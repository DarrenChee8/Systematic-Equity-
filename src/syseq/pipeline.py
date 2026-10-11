"""Build everything the later stages need: data, point-in-time fundamentals, signals, returns.

One function, so every milestone script works from identical inputs.
"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from syseq.data.edgar import (TAGS, asof_panel, clean_share_counts, download_fundamentals, market_cap_panel, quarterly_facts,
                              split_adjust, surprise_facts, ttm_facts)
from syseq.data.free import FreeData
from syseq.evaluation.ic import forward_returns, neutral_returns
from syseq.signals.fundamental import quality_characteristics, value_characteristics
from syseq.signals.lowrisk import low_beta, low_volatility
from syseq.signals.momentum import momentum, monthly_prices
from syseq.signals.transforms import composite, sector_zscore

THEME_GROUPS = {
    "VALUE": ["earnings_yield", "book_to_price", "cashflow_yield", "sales_yield"],
    "QUALITY": ["roe", "cash_profitability", "gross_profitability", "low_accruals", "low_leverage"],
    "LOW_RISK": ["low_volatility", "low_beta"],
    "MOMENTUM": ["momentum"],
    "SURPRISE": ["earnings_surprise", "revenue_surprise"],
}


def build(config_path: str = "config/base.yaml", data_dir: str = "data", verbose: bool = True) -> SimpleNamespace:
    cfg = yaml.safe_load(Path(config_path).read_text())
    start = cfg["start"]
    src = FreeData(data_dir)

    # ---- data ---------------------------------------------------------------
    members = src.membership(start, cfg["end"])
    daily = src.prices(members.columns.tolist(), "2008-06-01", cfg["end"])
    daily = daily.where(daily > 0)        # Yahoo's adjusted prices are occasionally zero or negative
    monthly = monthly_prices(daily)
    in_window = members.loc[start:].any()
    universe = [t for t in monthly.columns if in_window.get(t, False)]
    close, splits = src.market_data(universe, "2008-06-01", cfg["end"])
    facts = download_fundamentals(universe, data_dir)
    sectors = src.sectors()
    dates, cols = monthly.index, monthly.columns
    mask = members.reindex(index=dates, columns=cols).fillna(False).astype(bool)

    # ---- point-in-time fundamentals ----------------------------------------
    def ttm(item):
        return asof_panel(ttm_facts(facts, TAGS[item], item), [item], dates).reindex(columns=cols)

    def surprise(item):
        s = surprise_facts(quarterly_facts(facts, TAGS[item], item), item)
        return asof_panel(s, [item], dates, max_age_days=200).reindex(columns=cols)

    net_income, cfo, revenue, gross_profit = ttm("net_income"), ttm("cfo"), ttm("revenue"), ttm("gross_profit")
    equity = asof_panel(facts, TAGS["equity"], dates).reindex(columns=cols)
    assets = asof_panel(facts, TAGS["assets"], dates).reindex(columns=cols)
    share_facts = clean_share_counts(split_adjust(facts[facts["tag"].isin(TAGS["shares"])], splits))
    shares = asof_panel(share_facts, TAGS["shares"], dates, "any", max_age_days=400).reindex(columns=cols)
    market_cap = market_cap_panel(shares, close.resample("ME").last().reindex(index=dates, columns=cols))

    # ---- characteristics -> sector-neutral signals --------------------------
    raw = {}
    raw.update(value_characteristics(net_income, equity, cfo, market_cap, revenue))
    raw.update(quality_characteristics(net_income, equity, cfo, assets, gross_profit))
    raw["low_volatility"] = low_volatility(daily).reindex(dates)
    raw["low_beta"] = low_beta(daily).reindex(dates)
    m = cfg["signals"]["momentum"]
    raw["momentum"] = momentum(monthly, m["lookback_months"], m["skip_months"])
    raw["earnings_surprise"] = surprise("net_income")
    raw["revenue_surprise"] = surprise("revenue")

    coverage = {k: (v.where(mask).notna().sum(axis=1) / mask.sum(axis=1)).loc[start:].mean()
                for k, v in raw.items()}
    dropped = [k for k, c in coverage.items() if c < 0.30]
    if dropped and verbose:
        print("Dropped for low coverage (<30% of index members):", dropped)
    z = {k: sector_zscore(v.where(mask), sectors) for k, v in raw.items() if k not in dropped}
    themes = {name: composite([z[k] for k in parts if k in z], sectors, min_parts=1)
              for name, parts in THEME_GROUPS.items() if any(k in z for k in parts)}

    # ---- returns ------------------------------------------------------------
    fwd = forward_returns(monthly)
    beta = 0.67 * (-raw["low_beta"]) + 0.33      # shrink estimated betas toward 1
    fwd_neutral = neutral_returns(fwd, beta, sectors, mask)

    return SimpleNamespace(
        cfg=cfg, start=start, members=members, daily=daily, monthly=monthly, universe=universe,
        sectors=sectors, mask=mask, raw=raw, z=z, themes=themes, fwd=fwd, fwd_neutral=fwd_neutral,
        beta=beta, market_cap=market_cap,
        fundamentals=dict(net_income=net_income, cfo=cfo, revenue=revenue, gross_profit=gross_profit,
                          equity=equity, assets=assets))
