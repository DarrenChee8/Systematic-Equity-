"""Milestone 2b: stronger inputs.

Changes from milestone 2:
- earnings, cash flow, revenue and gross profit are trailing-twelve-month, not last annual
- more XBRL labels for earnings, plus history for companies that re-registered with the SEC
- new characteristics: sales yield, gross profitability
- new theme: earnings and revenue surprise

Run from the repo root. Data and signals come from syseq.pipeline.build().
"""
from pathlib import Path

import pandas as pd

from syseq.evaluation.correlation import series_correlation, signal_correlation
from syseq.evaluation.report import evaluate_signals, signal_figure
from syseq.pipeline import build
from syseq.signals.transforms import composite

p = build()
START = p.start
themes = dict(p.themes)
themes["COMBINED"] = composite(list(p.themes.values()), p.sectors)
signals = {**p.z, **themes}
theme_names = list(themes)

# ---- sanity check -----------------------------------------------------------
f = p.fundamentals
check = [t for t in ["AAPL", "MSFT", "JPM", "XOM", "WMT", "JNJ"] if t in p.monthly.columns]
last = p.market_cap[check].dropna(how="all").index[-1]
sanity = pd.DataFrame({
    "market_cap_bn": p.market_cap.loc[last, check] / 1e9,
    "ttm_net_income_bn": f["net_income"].loc[last, check] / 1e9,
    "ttm_revenue_bn": f["revenue"].loc[last, check] / 1e9,
    "ttm_cash_flow_bn": f["cfo"].loc[last, check] / 1e9,
    "pe": p.market_cap.loc[last, check] / f["net_income"].loc[last, check],
    "earnings_surprise": p.raw["earnings_surprise"].loc[last, check],
})

# ---- evaluation -------------------------------------------------------------
q = p.cfg["evaluation"]["quantiles"]
summary_raw, _, _ = evaluate_signals(signals, p.fwd, p.mask, START, q)
summary, spreads, _ = evaluate_signals(signals, p.fwd_neutral, p.mask, START, q)
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
