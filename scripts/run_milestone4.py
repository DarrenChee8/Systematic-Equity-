"""Milestone 4: from scores to holdings.

Each month: estimate a risk model, then choose the portfolio with the highest combined score
that stays within the risk limits. Saves the holdings for milestone 5.
Run from the repo root. Takes about five minutes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from syseq.backtest.engine import run
from syseq.model.combine import combine, walk_forward_weights
from syseq.evaluation.ic import rank_ic
from syseq.pipeline import build
from syseq.portfolio.optimiser import Limits

p = build()
pc = p.cfg["portfolio"]
names = list(p.themes)

# Milestone 3 found nothing better than equal weights, so that is the forecast carried forward.
ics = pd.DataFrame({k: rank_ic(p.themes[k], p.fwd_neutral) for k in names}).loc[p.start:]
theme_weights = walk_forward_weights(ics, "equal").reindex(p.monthly.index).ffill().bfill()
score = combine(p.themes, theme_weights, p.sectors)

limits = Limits(pc["tracking_error"], pc["max_stock_active"], pc["max_sector_active"],
                pc["max_beta_active"], pc["max_turnover"])
print("Rebalancing monthly...")
weights, bench, diag = run(score, p.market_cap, p.daily, p.monthly, p.mask, p.sectors, p.beta, p.fwd,
                           pc["start"], limits, pc["risk_window_days"], pc["risk_factors"], issuer=p.issuer)

Path("data").mkdir(exist_ok=True)
weights.to_parquet("data/m4_weights.parquet")
bench.to_parquet("data/m4_benchmark.parquet")
score.to_parquet("data/m4_score.parquet")

numeric = diag.drop(columns="status").astype(float)
summary = pd.DataFrame({"average": numeric.mean(), "minimum": numeric.min(), "maximum": numeric.max()})
summary.loc["turnover", "average"] = numeric["turnover"].iloc[1:].mean()    # first month is the initial purchase
limit_col = {"tracking_error": limits.tracking_error, "max_stock_active": limits.max_stock,
             "max_sector_active": limits.max_sector, "active_beta": limits.max_beta,
             "turnover": limits.max_turnover}
summary["limit"] = pd.Series(limit_col)

last = weights.index[-1]
w, b = weights.loc[last], bench.loc[last]
held = b.index[b > 0].union(w.index[w > 0])
diag.to_parquet("data/m4_diagnostics.parquet")
sector = p.sectors.reindex(held).fillna("Unknown")
book = pd.DataFrame({"sector": sector, "portfolio": w[held], "benchmark": b[held]})
book["active"] = book["portfolio"] - book["benchmark"]
book["score"] = score.loc[last].reindex(held)
for k in names:
    book[k.lower()] = p.themes[k].loc[last].reindex(held)
top = book.sort_values("active", ascending=False).head(10)
bottom = book.sort_values("active").head(10)
by_sector = book.groupby("sector")[["portfolio", "benchmark", "active"]].sum().sort_values("active", ascending=False)

pd.set_option("display.width", 220, "display.max_columns", None)
print(f"\nRebalances: {len(weights)} from {weights.index[0].date()} to {last.date()}")
print("Optimiser status:", diag["status"].value_counts().to_dict())
print("\nPortfolio characteristics over time (limits where they apply)\n", summary.round(4))
print(f"\nLargest benchmark weights at {last.date()} (%), to compare with the real S&P 500\n",
      (b.sort_values(ascending=False).head(10) * 100).round(2).to_dict())
print(f"\nLargest overweights at {last.date()} (weights in %, scores in standard deviations)\n",
      top.assign(**{c: top[c] * 100 for c in ["portfolio", "benchmark", "active"]}).round(2))
print("\nLargest underweights\n",
      bottom.assign(**{c: bottom[c] * 100 for c in ["portfolio", "benchmark", "active"]}).round(2))
print("\nSector weights (%)\n", (by_sector * 100).round(2))
print(f"\nAverage score: portfolio {(w[held] * book['score'].fillna(0)).sum():.2f}, "
      f"benchmark {(b[held] * book['score'].fillna(0)).sum():.2f}")

Path("reports").mkdir(exist_ok=True)
diag.to_csv("reports/m4_diagnostics.csv")
summary.round(5).to_csv("reports/m4_summary.csv")
book.sort_values("active", ascending=False).round(5).to_csv("reports/m4_latest_holdings.csv")
(by_sector * 100).round(3).to_csv("reports/m4_latest_sectors.csv")

fig, ax = plt.subplots(2, 2, figsize=(14, 8))
numeric[["holdings", "benchmark_names"]].plot(ax=ax[0, 0], title="Number of holdings")
(numeric[["tracking_error", "active_share"]] * 100).plot(ax=ax[0, 1], secondary_y="active_share",
                                                        title="Predicted tracking error (%, left) and active share (%, right)")
(numeric["turnover"].iloc[1:] * 100).plot(ax=ax[1, 0], title="One-way turnover per rebalance (%)")
ax[1, 0].axhline(limits.max_turnover * 100, color="grey", lw=0.8, ls="--")
(by_sector["active"] * 100).plot.barh(ax=ax[1, 1], title=f"Sector active weights at {last.date()} (%)")
ax[1, 1].axvline(0, color="grey", lw=0.8)
ax[1, 1].invert_yaxis()
fig.tight_layout()
fig.savefig("reports/m4_portfolio.png", dpi=150)
print("\nSaved reports/m4_portfolio.png, reports/m4_*.csv and data/m4_*.parquet")
