"""Milestone 2b: stronger inputs.

Changes from milestone 2:
- earnings, cash flow, revenue and gross profit are trailing-twelve-month, not last annual
- more XBRL labels for earnings, plus history for companies that re-registered with the SEC
- new characteristics: sales yield, gross profitability
- new theme: earnings and revenue surprise

Run from the repo root. The first run re-downloads fundamentals (about 15-20 minutes).
"""
from pathlib import Path

import pandas as pd
import yaml

from syseq.data.edgar import (TAGS, asof_panel, download_fundamentals, quarterly_facts,
                              split_adjust, surprise_facts, ttm_facts)
from syseq.data.free import FreeData
from syseq.evaluation.correlation import series_correlation, signal_correlation
from syseq.evaluation.ic import forward_returns, neutral_returns
from syseq.evaluation.report import evaluate_signals, signal_figure
from syseq.signals.fundamental import quality_characteristics, value_characteristics
from syseq.signals.lowrisk import low_beta, low_volatility
from syseq.signals.momentum import momentum, monthly_prices
from syseq.signals.transforms import composite, sector_zscore

cfg = yaml.safe_load(Path("config/base.yaml").read_text())
START = cfg["start"]
src = FreeData("data")

# ---- data -------------------------------------------------------------------
members = src.membership(START, cfg["end"])
daily = src.prices(members.columns.tolist(), "2008-06-01", cfg["end"])
monthly = monthly_prices(daily)
in_window = members.loc[START:].any()
universe = [t for t in monthly.columns if in_window.get(t, False)]
close, splits = src.market_data(universe, "2008-06-01", cfg["end"])
facts = download_fundamentals(universe, "data")
sectors = src.sectors()
dates = monthly.index
cols = monthly.columns
mask = members.reindex(index=dates, columns=cols).fillna(False).astype(bool)

# ---- point-in-time fundamentals --------------------------------------------
def ttm(item):
    return asof_panel(ttm_facts(facts, TAGS[item], item), [item], dates).reindex(columns=cols)


def surprise(item):
    s = surprise_facts(quarterly_facts(facts, TAGS[item], item), item)
    return asof_panel(s, [item], dates, max_age_days=200).reindex(columns=cols)


net_income, cfo, revenue, gross_profit = ttm("net_income"), ttm("cfo"), ttm("revenue"), ttm("gross_profit")
equity = asof_panel(facts, TAGS["equity"], dates).reindex(columns=cols)
assets = asof_panel(facts, TAGS["assets"], dates).reindex(columns=cols)
share_facts = split_adjust(facts[facts["tag"].isin(TAGS["shares"])], splits)
shares = asof_panel(share_facts, TAGS["shares"], dates, "any", max_age_days=400).reindex(columns=cols)
market_cap = shares * close.resample("ME").last().reindex(index=dates, columns=cols)

# ---- characteristics -> sector-neutral signals ------------------------------
raw = {}
raw.update(value_characteristics(net_income, equity, cfo, market_cap, revenue))
raw.update(quality_characteristics(net_income, equity, cfo, assets, gross_profit))
raw["low_volatility"] = low_volatility(daily).reindex(dates)
raw["low_beta"] = low_beta(daily).reindex(dates)
m = cfg["signals"]["momentum"]
raw["momentum"] = momentum(monthly, m["lookback_months"], m["skip_months"])
raw["earnings_surprise"] = surprise("net_income")
raw["revenue_surprise"] = surprise("revenue")

coverage = {k: (v.where(mask).notna().sum(axis=1) / mask.sum(axis=1)).loc[START:].mean() for k, v in raw.items()}
dropped = [k for k, c in coverage.items() if c < 0.30]
if dropped:
    print("Dropped for low coverage (<30% of index members):", dropped)
z = {k: sector_zscore(v.where(mask), sectors) for k, v in raw.items() if k not in dropped}

groups = {
    "VALUE": ["earnings_yield", "book_to_price", "cashflow_yield", "sales_yield"],
    "QUALITY": ["roe", "cash_profitability", "gross_profitability", "low_accruals", "low_leverage"],
    "LOW_RISK": ["low_volatility", "low_beta"],
    "MOMENTUM": ["momentum"],
    "SURPRISE": ["earnings_surprise", "revenue_surprise"],
}
themes = {name: composite([z[k] for k in parts if k in z], sectors, min_parts=1)
          for name, parts in groups.items() if any(k in z for k in parts)}
themes["COMBINED"] = composite(list(themes.values()), sectors)
signals = {**z, **themes}
theme_names = list(themes)

# ---- sanity check -----------------------------------------------------------
check = [t for t in ["AAPL", "MSFT", "JPM", "XOM", "WMT", "JNJ"] if t in cols]
last = market_cap[check].dropna(how="all").index[-1]
sanity = pd.DataFrame({
    "market_cap_bn": market_cap.loc[last, check] / 1e9,
    "ttm_net_income_bn": net_income.loc[last, check] / 1e9,
    "ttm_revenue_bn": revenue.loc[last, check] / 1e9,
    "ttm_cash_flow_bn": cfo.loc[last, check] / 1e9,
    "pe": market_cap.loc[last, check] / net_income.loc[last, check],
    "earnings_surprise": raw["earnings_surprise"].loc[last, check],
})

# ---- evaluation -------------------------------------------------------------
fwd = forward_returns(monthly)
beta = 0.67 * (-raw["low_beta"]) + 0.33   # shrink estimated betas toward 1
fwd_neutral = neutral_returns(fwd, beta, sectors, mask)
q = cfg["evaluation"]["quantiles"]
summary_raw, _, _ = evaluate_signals(signals, fwd, mask, START, q)
summary, spreads, _ = evaluate_signals(signals, fwd_neutral, mask, START, q)
sig_corr = signal_correlation({k: signals[k].loc[START:] for k in theme_names})
pay_corr = series_correlation({k: spreads[k] for k in theme_names})

pd.set_option("display.width", 200, "display.max_columns", None)
print(f"\nSanity check at {last.date()} (trailing twelve months)\n", sanity.round(2))
print("\nRAW returns: signal summary\n", summary_raw.round(3))
print("\nMARKET- AND SECTOR-NEUTRAL returns: signal summary\n", summary.round(3))
print("\nCorrelation between theme scores\n", sig_corr.round(2))
print("\nCorrelation between neutral theme payoffs\n", pay_corr.round(2))

Path("reports").mkdir(exist_ok=True)
summary.round(4).to_csv("reports/m2b_signal_summary.csv")
summary_raw.round(4).to_csv("reports/m2b_signal_summary_raw.csv")
pay_corr.round(3).to_csv("reports/m2b_theme_payoff_correlation.csv")
sanity.round(3).to_csv("reports/m2b_sanity.csv")
signal_figure({"raw": summary_raw["mean_ic"], "neutral": summary["mean_ic"]},
              spreads, pay_corr, theme_names, "reports/m2b_signals.png")
print("\nSaved reports/m2b_signals.png and reports/m2b_*.csv")
