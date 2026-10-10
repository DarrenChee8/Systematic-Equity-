"""Point-in-time fundamentals from SEC EDGAR's XBRL "company facts" API.

Every figure carries the date it was filed, so a signal at month-end t can be limited
to what had actually been published by t.

Known limits:
- Only companies that still have a ticker in SEC's current list are found, so most
  delisted companies have no fundamentals (survivorship bias, same direction as prices).
- XBRL data is reliable from about 2010.
- Flow items (earnings, cash flow) use the latest annual report, so they can be up to
  about 15 months old. Trailing-twelve-month figures are a later refinement.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# For each item, XBRL tags in order of preference.
TAGS = {
    "net_income": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss",
                   "IncomeLossFromContinuingOperations"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit": ["GrossProfit"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets": ["Assets"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "shares": ["WeightedAverageNumberOfDilutedSharesOutstanding",
               "EntityCommonStockSharesOutstanding"],
}
ALL_TAGS = {t for tags in TAGS.values() for t in tags}
COLUMNS = ["ticker", "tag", "start", "end", "val", "filed", "form"]
CACHE_VERSION = "v2"   # bump when TAGS change, so cached downloads are refreshed

# SEC's ticker list points only at a company's current registrant. Where a company has
# re-registered under a new number, its history sits under the old one. Add known cases here.
EXTRA_CIKS = {"XOM": [34088], "BKNG": [1075531]}


def user_agent() -> str:
    """SEC requires a contact name and email on every request. Read from env or .env."""
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("SEC_USER_AGENT="):
                ua = line.split("=", 1)[1].strip().strip('"')
    if not ua:
        raise RuntimeError('Create a file named .env containing: SEC_USER_AGENT="Your Name you@example.com"')
    return ua


def _get(url: str, headers: dict, tries: int = 4):
    for i in range(tries):
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        time.sleep(2 * (i + 1))  # throttled or server error: back off and retry
    return None


def extract_facts(js: dict, ticker: str) -> pd.DataFrame:
    rows = []
    for taxonomy in ("us-gaap", "dei"):
        for tag, node in js.get("facts", {}).get(taxonomy, {}).items():
            if tag not in ALL_TAGS:
                continue
            for unit, recs in node.get("units", {}).items():
                if unit not in ("USD", "shares"):
                    continue
                for r in recs:
                    if str(r.get("form", "")).startswith("10-") and r.get("val") is not None:
                        rows.append((ticker, tag, r.get("start"), r["end"], float(r["val"]),
                                     r["filed"], r["form"]))
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def download_fundamentals(tickers: list[str], cache_dir: str = "data") -> pd.DataFrame:
    cache = Path(cache_dir)
    path = cache / f"fundamentals_{CACHE_VERSION}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    headers = {"User-Agent": user_agent()}
    listing = _get(TICKER_URL, headers)
    if listing is None:
        raise RuntimeError("Could not download the SEC ticker list. Check the .env contact line.")
    cik = {v["ticker"].replace("-", "."): int(v["cik_str"]) for v in listing.values()}
    per = cache / f"edgar_{CACHE_VERSION}"
    per.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if t in cik]
    print(f"SEC EDGAR: {len(todo)} of {len(tickers)} tickers found in the SEC list")
    for i, t in enumerate(todo, 1):
        f = per / f"{t}.parquet"
        if not f.exists():
            parts = []
            for c in dict.fromkeys([cik[t], *EXTRA_CIKS.get(t, [])]):
                js = _get(FACTS_URL.format(cik=c), headers)
                if js:
                    parts.append(extract_facts(js, t))
                time.sleep(0.15)  # stay well under SEC's 10 requests/second limit
            df = pd.concat(parts).drop_duplicates() if parts else pd.DataFrame(columns=COLUMNS)
            df.to_parquet(f)
        if i % 25 == 0 or i == len(todo):
            print(f"  fundamentals {i}/{len(todo)}", flush=True)
    frames = [pd.read_parquet(per / f"{t}.parquet") for t in todo]
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out.to_parquet(path)
    return out


def split_adjust(facts: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Restate share counts onto today's share basis, to match split-adjusted prices.

    A count filed before a 4-for-1 split is multiplied by 4.
    """
    cum = splits.where(splits > 0, 1.0).cumprod()
    out = facts.copy()
    factor = np.ones(len(out))
    filed = out["filed"].values
    for tkr, idx in out.groupby("ticker").indices.items():
        if tkr not in cum.columns:
            continue
        c = cum[tkr]
        pos = c.index.searchsorted(filed[idx], side="right") - 1
        before = np.where(pos >= 0, c.values[np.clip(pos, 0, None)], 1.0)
        factor[idx] = c.values[-1] / before
    out["val"] = out["val"] * factor
    return out


def asof_panel(facts: pd.DataFrame, tags: list[str], month_ends: pd.DatetimeIndex,
               duration: str = "instant", max_age_days: int = 550, lag_days: int = 1) -> pd.DataFrame:
    """Latest value known at each month-end. Index: month_ends, columns: tickers.

    duration: 'instant' (balance sheet), 'annual' (12-month flows) or 'any'.
    A filing becomes usable lag_days after its filing date. Values whose period ended
    more than max_age_days before the month-end are dropped as stale.
    """
    f = facts[facts["tag"].isin(tags) & (facts["end"] <= facts["filed"])].copy()  # drop mis-dated records
    if duration == "instant":
        f = f[f["start"].isna()]
    elif duration == "annual":
        f = f[(f["end"] - f["start"]).dt.days.between(350, 380)]
    f["prio"] = f["tag"].map({t: i for i, t in enumerate(tags)})
    f["avail"] = f["filed"] + pd.Timedelta(days=lag_days)
    f = f.sort_values(["ticker", "avail", "end", "prio"], ascending=[True, True, True, False])
    # keep only the newest reporting period known so far (drops prior-year comparatives)
    f = f[f["end"] >= f.groupby("ticker")["end"].cummax()]
    f = f.drop_duplicates(["ticker", "avail"], keep="last")
    vals = f.pivot(index="avail", columns="ticker", values="val")
    ends = f.pivot(index="avail", columns="ticker", values="end")
    grid = vals.index.union(month_ends)
    vals = vals.reindex(grid).ffill().reindex(month_ends)
    ends = ends.reindex(grid).ffill().reindex(month_ends)
    age = ends.apply(lambda col: (col.index.to_series() - col).dt.days)
    return vals.where(age <= max_age_days)


# ---- trailing-twelve-month and quarterly figures -----------------------------------
def _periods(facts: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    """One row per reporting period, as first filed, using the preferred tag available."""
    f = facts[facts["tag"].isin(tags) & facts["start"].notna() & (facts["end"] <= facts["filed"])].copy()
    f["prio"] = f["tag"].map({t: i for i, t in enumerate(tags)})
    f = f.sort_values(["prio", "filed"]).drop_duplicates(["ticker", "start", "end"], keep="first")
    f["days"] = (f["end"] - f["start"]).dt.days
    return f[["ticker", "start", "end", "val", "filed", "days"]]


def _as_facts(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df[["ticker", "end", "val", "filed"]].copy()
    out["tag"], out["start"], out["form"] = name, pd.NaT, ""
    return out[COLUMNS]


def ttm_facts(facts: pd.DataFrame, tags: list[str], name: str) -> pd.DataFrame:
    """Trailing-twelve-month value at each period end, dated by when it became knowable.

    Full year: the annual figure. Interim: year-to-date + previous full year - the same
    year-to-date a year earlier. All three numbers are public by the interim filing date.
    Output has the same columns as raw facts (tag = name), so asof_panel can read it.
    """
    p = _periods(facts, tags)
    annual = p[p["days"].between(350, 380)]
    interim = p[p["days"].between(80, 290)]
    prev = annual.assign(next_start=annual["end"] + pd.Timedelta(days=1))[
        ["ticker", "next_start", "start", "val", "filed"]
    ].rename(columns={"start": "a_start", "val": "a_val", "filed": "a_filed"})
    m = interim.merge(prev, left_on=["ticker", "start"], right_on=["ticker", "next_start"])
    last_year = interim[["ticker", "start", "days", "val", "filed"]].rename(
        columns={"start": "a_start", "days": "p_days", "val": "p_val", "filed": "p_filed"})
    m = m.merge(last_year, on=["ticker", "a_start"])
    m["gap"] = (m["days"] - m["p_days"]).abs()
    m = m[m["gap"] <= 10].sort_values("gap").drop_duplicates(["ticker", "end"], keep="first")
    m["val"] = m["val"] + m["a_val"] - m["p_val"]
    m["filed"] = m[["filed", "a_filed", "p_filed"]].max(axis=1)
    return _as_facts(pd.concat([annual, m]), name)


def quarterly_facts(facts: pd.DataFrame, tags: list[str], name: str) -> pd.DataFrame:
    """Single-quarter values, from three-month figures or differences of year-to-date figures.

    The fourth quarter is the full year minus nine months, known when the annual report is filed.
    """
    p = _periods(facts, tags)
    p = p[p["days"] <= 380].sort_values(["ticker", "start", "end"])
    g = p.groupby(["ticker", "start"])
    step = p["days"] - g["days"].shift(1)
    first = g.cumcount() == 0
    direct = first & p["days"].between(80, 100)
    derived = ~first & step.between(80, 100)
    q = p.copy()
    q["val"] = np.where(direct, p["val"], p["val"] - g["val"].shift(1))
    q["filed"] = np.where(direct, p["filed"], np.maximum(p["filed"], g["filed"].shift(1).fillna(p["filed"])))
    q["filed"] = pd.to_datetime(q["filed"])
    q = q[direct | derived].sort_values("days").drop_duplicates(["ticker", "end"], keep="first")
    return _as_facts(q, name)


def surprise_facts(quarterly: pd.DataFrame, name: str, lookback: int = 8, min_quarters: int = 4) -> pd.DataFrame:
    """Standardised unexpected value: (this quarter - same quarter last year) / volatility of that change.

    A simple, analyst-free measure of earnings surprise (seasonal random walk).
    """
    q = quarterly.sort_values(["ticker", "end"]).copy()
    g = q.groupby("ticker")
    year_gap = (q["end"] - g["end"].shift(4)).dt.days
    change = (q["val"] - g["val"].shift(4)).where(year_gap.between(350, 380))
    q["change"] = change
    scale = q.groupby("ticker")["change"].transform(
        lambda c: c.shift(1).rolling(lookback, min_periods=min_quarters).std())
    q["val"] = (change / scale.where(scale > 0)).clip(-10, 10)
    return _as_facts(q.dropna(subset=["val"]), name)
