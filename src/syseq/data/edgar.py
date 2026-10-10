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
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
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
    path = cache / "fundamentals.parquet"
    if path.exists():
        return pd.read_parquet(path)
    headers = {"User-Agent": user_agent()}
    listing = _get(TICKER_URL, headers)
    if listing is None:
        raise RuntimeError("Could not download the SEC ticker list. Check the .env contact line.")
    cik = {v["ticker"].replace("-", "."): int(v["cik_str"]) for v in listing.values()}
    per = cache / "edgar"
    per.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if t in cik]
    print(f"SEC EDGAR: {len(todo)} of {len(tickers)} tickers found in the SEC list")
    for i, t in enumerate(todo, 1):
        f = per / f"{t}.parquet"
        if not f.exists():
            js = _get(FACTS_URL.format(cik=cik[t]), headers)
            (extract_facts(js, t) if js else pd.DataFrame(columns=COLUMNS)).to_parquet(f)
            time.sleep(0.15)  # stay well under SEC's 10 requests/second limit
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
    f = facts[facts["tag"].isin(tags)].copy()
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
