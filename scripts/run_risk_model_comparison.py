"""Milestone 5b: which risk model predicts risk best?

Runs the full backtest under several risk-model settings and compares predicted with realised
tracking error. Models are judged on risk accuracy only. Choosing a risk model by the returns
it happened to produce would be overfitting.
Run from the repo root. Takes about ten minutes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from syseq.backtest.engine import run
from syseq.backtest.performance import period_returns
from syseq.evaluation.ic import rank_ic
from syseq.model.combine import combine, walk_forward_weights
from syseq.pipeline import build
from syseq.portfolio.optimiser import Limits

p = build()
pc = p.cfg["portfolio"]
names = list(p.themes)
ics = pd.DataFrame({k: rank_ic(p.themes[k], p.fwd_neutral) for k in names}).loc[p.start:]
score = combine(p.themes, walk_forward_weights(ics, "equal").reindex(p.monthly.index).ffill().bfill(), p.sectors)
limits = Limits(pc["tracking_error"], pc["max_stock_active"], pc["max_sector_active"],
                pc["max_beta_active"], pc["max_turnover"])
complete = p.monthly.index[p.monthly.index <= p.daily.index[-1]]

variants = {
    "2 years, 15 factors (original)": dict(risk_window=504, n_factors=15),
    "1 year of history": dict(risk_window=252, n_factors=15),
    "4 years of history": dict(risk_window=1008, n_factors=15),
    "5 factors": dict(risk_window=504, n_factors=5),
    "30 factors": dict(risk_window=504, n_factors=30),
    "recency weighted, 6-month half-life": dict(risk_window=504, n_factors=15, halflife=126),
    "recency weighted, 3-month half-life": dict(risk_window=504, n_factors=15, halflife=63),
    "original + calibration": dict(risk_window=504, n_factors=15, calibrate=True),
    "6-month half-life + calibration": dict(risk_window=504, n_factors=15, halflife=126, calibrate=True),
}

rows = {}
for name, kw in variants.items():
    w, b, diag = run(score, p.market_cap, p.daily, p.monthly, p.mask, p.sectors, p.beta, p.fwd,
                     pc["start"], limits, verbose=False, issuer=p.issuer, **kw)
    w = w[w.index < complete[-1]]
    r = period_returns(w, b.loc[w.index], p.fwd)
    predicted = diag["tracking_error_calibrated"].astype(float).reindex(r.index)
    rolling = r["active"].rolling(12).std() * np.sqrt(12)
    rows[name] = {
        "predicted_te": predicted.mean(),
        "realised_te": r["active"].std() * np.sqrt(12),
        "bias_statistic": (r["active"] / (predicted / np.sqrt(12))).std(),
        "highest_12m_te": rolling.max(),
        "worst_month_vs_benchmark": r["active"].abs().max(),
        "holdings": diag["holdings"].astype(float).mean(),
    }
    print(f"  {name}: realised {rows[name]['realised_te']:.2%}, bias {rows[name]['bias_statistic']:.2f}", flush=True)

table = pd.DataFrame(rows).T
pd.set_option("display.width", 200, "display.max_columns", None)
show = table.copy()
for c in ["predicted_te", "realised_te", "highest_12m_te", "worst_month_vs_benchmark"]:
    show[c] = (show[c] * 100).round(2)
print("\nRisk model comparison (tracking errors in % a year; bias statistic 1.00 = perfect)\n",
      show.round({"bias_statistic": 2, "holdings": 0}))
Path("reports").mkdir(exist_ok=True)
table.round(5).to_csv("reports/m5b_risk_models.csv")
print("\nSaved reports/m5b_risk_models.csv")
