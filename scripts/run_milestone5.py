"""Milestone 5: how did the milestone 4 holdings perform?

Measures return against the benchmark, realised against predicted risk, the effect of trading
costs, and which themes the active return came from.
Run from the repo root, after run_milestone4.py (it reads the holdings that script saved).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from syseq.backtest.attribution import attribute
from syseq.backtest.performance import calendar_years, net_of_costs, period_returns, summary, traded_value
from syseq.pipeline import build

for f in ("data/m4_weights.parquet", "data/m4_benchmark.parquet", "data/m4_diagnostics.parquet"):
    if not Path(f).exists():
        raise SystemExit(f"{f} not found. Run: python scripts/run_milestone4.py")

p = build()
weights = pd.read_parquet("data/m4_weights.parquet")
bench = pd.read_parquet("data/m4_benchmark.parquet")
diag = pd.read_parquet("data/m4_diagnostics.parquet")

# Drop the final rebalance if the month after it is not yet complete.
complete = p.monthly.index[p.monthly.index <= p.daily.index[-1]]
weights = weights[weights.index < complete[-1]]
bench = bench.loc[weights.index]

rate = p.cfg["costs"]["rate_bps"]
gross = period_returns(weights, bench, p.fwd)
traded = traded_value(weights, p.fwd)
perf = net_of_costs(gross, traded, rate)

stats = pd.DataFrame({"gross": summary(perf["portfolio"], perf["benchmark"]),
                      f"net ({rate}bp)": summary(perf["portfolio_net"], perf["benchmark"])})
years = calendar_years(perf["portfolio_net"], perf["benchmark"])
sens = pd.DataFrame({f"{c}bp": summary(net_of_costs(gross, traded, c)["portfolio_net"], gross["benchmark"])
                     [["excess_return", "information_ratio"]] for c in p.cfg["costs"]["sensitivity_bps"]}).T
annual_turnover = traded.iloc[1:].mean() / 2 * 12          # one-way, excluding the initial purchase

attr = attribute(weights, bench, p.fwd, p.themes, p.beta, p.sectors, p.market_cap)
contrib = (attr.mean() * 12).rename("contribution_per_year")
contrib["TOTAL (gross active, arithmetic)"] = attr.sum(axis=1).mean() * 12
contrib[f"costs ({rate}bp)"] = -perf["cost"].mean() * 12

predicted = diag["tracking_error"].astype(float).reindex(perf.index)
realised_te = perf["active"].std() * np.sqrt(12)
rolling_te = perf["active"].rolling(12).std() * np.sqrt(12)

pd.set_option("display.width", 200, "display.max_columns", None)
pct = lambda s: (s * 100).round(2)
print(f"\nBacktest: {len(perf)} months, {perf.index[0].date()} to "
      f"{(perf.index[-1] + pd.offsets.MonthEnd(1)).date()}")
ratio_rows = ["information_ratio", "t_stat", "beta", "hit_rate"]
show = stats.copy()
show.loc[~show.index.isin(ratio_rows)] = show.loc[~show.index.isin(ratio_rows)] * 100
print("\nPerformance (% a year, except ratios, t-stat, beta and hit rate)\n", show.round(2))
print(f"\nTracking error: predicted {predicted.mean():.2%}, realised {realised_te:.2%}")
print(f"One-way turnover: {annual_turnover:.0%} a year")
print(f"Largest portfolio weight in stocks with no return in a month: {perf['weight_without_return'].max():.2%}")
print("\nNet excess return by trading cost assumption (% a year)\n",
      sens.assign(excess_return=pct(sens["excess_return"])).round(2))
print("\nCalendar years, net of costs (%)\n",
      years.assign(**{c: pct(years[c]) for c in ["portfolio", "benchmark", "excess"]}))
print("\nWhere the active return came from (% a year)\n", pct(contrib))

Path("reports").mkdir(exist_ok=True)
stats.round(5).to_csv("reports/m5_performance.csv")
years.round(5).to_csv("reports/m5_calendar_years.csv")
contrib.round(6).to_csv("reports/m5_attribution.csv")
sens.round(5).to_csv("reports/m5_cost_sensitivity.csv")
perf.round(6).to_csv("reports/m5_monthly_returns.csv")

end = perf.index + pd.offsets.MonthEnd(1)
fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))
pd.DataFrame({"portfolio (net)": (1 + perf["portfolio_net"]).cumprod().values,
              "benchmark": (1 + perf["benchmark"]).cumprod().values}, index=end).plot(
    ax=ax[0, 0], logy=True, title="Growth of 1 (log scale)")
pd.DataFrame({"gross": perf["active"].cumsum().values * 100,
              "net": perf["active_net"].cumsum().values * 100}, index=end).plot(
    ax=ax[0, 1], title="Cumulative active return (%)")
ax[0, 1].axhline(0, color="grey", lw=0.8)
pd.DataFrame({"realised (rolling 12 months)": rolling_te.values * 100,
              "predicted": predicted.values * 100}, index=end).plot(
    ax=ax[1, 0], title="Tracking error (% a year)")
parts = pct(contrib.drop("TOTAL (gross active, arithmetic)"))
parts.plot.barh(ax=ax[1, 1], color=["#3b6ea5"] * (len(parts) - 1) + ["#b04a4a"],
                title="Contribution to active return (% a year)")
ax[1, 1].axvline(0, color="grey", lw=0.8)
ax[1, 1].invert_yaxis()
fig.tight_layout()
fig.savefig("reports/m5_backtest.png", dpi=150)
print("\nSaved reports/m5_backtest.png and reports/m5_*.csv")
