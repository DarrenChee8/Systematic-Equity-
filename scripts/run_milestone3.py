"""Milestone 3: how should the five themes be weighted?

Compares equal weights with three data-driven schemes whose weights are re-estimated every
month from past data only, and with hindsight weights fitted on the full sample.
Run from the repo root. Uses cached data, so it takes a few minutes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from syseq.evaluation.ic import rank_ic
from syseq.evaluation.report import evaluate_signals
from syseq.model.combine import combine, in_sample_weights, walk_forward_weights
from syseq.pipeline import build

MIN_MONTHS = 36          # history needed before data-driven weights are trusted

p = build()
themes, names = p.themes, list(p.themes)

# Monthly IC of each theme against market- and sector-neutral returns: the evidence weights are built from.
ics = pd.DataFrame({k: rank_ic(themes[k], p.fwd_neutral) for k in names}).loc[p.start:]

weights = {
    "equal": walk_forward_weights(ics, "equal", MIN_MONTHS),
    "ic": walk_forward_weights(ics, "ic", MIN_MONTHS),
    "ic_ir": walk_forward_weights(ics, "ic_ir", MIN_MONTHS),
    "mean_variance": walk_forward_weights(ics, "mean_variance", MIN_MONTHS),
    "mean_variance_5y": walk_forward_weights(ics, "mean_variance", MIN_MONTHS, window=60),
    "HINDSIGHT": in_sample_weights(ics, "mean_variance"),
}
models = {k: combine(themes, w, p.sectors) for k, w in weights.items()}

# Compare only after the burn-in, when every method has had the chance to differ from equal weights.
test_start = str(ics.index[MIN_MONTHS + 1].date())
summary, spreads, model_ics = evaluate_signals(models, p.fwd_neutral, p.mask, test_start,
                                               p.cfg["evaluation"]["quantiles"])
theme_summary, _, _ = evaluate_signals(themes, p.fwd_neutral, p.mask, test_start,
                                       p.cfg["evaluation"]["quantiles"])
latest = pd.DataFrame({k: w.iloc[-1] for k, w in weights.items()}).T
average = pd.DataFrame({k: w.loc[test_start:].mean() for k, w in weights.items()}).T

pd.set_option("display.width", 200, "display.max_columns", None)
print(f"\nTest period: {test_start} to {model_ics['equal'].index[-1].date()} "
      f"({len(model_ics['equal'])} months), market- and sector-neutral returns")
print("\nThemes on their own over the test period\n", theme_summary.round(3))
print("\nCombination methods\n", summary.round(3))
print("\nAverage weights over the test period\n", average.round(2))
print("\nLatest weights\n", latest.round(2))

Path("reports").mkdir(exist_ok=True)
summary.round(4).to_csv("reports/m3_combination_summary.csv")
average.round(4).to_csv("reports/m3_average_weights.csv")
weights["mean_variance"].round(4).to_csv("reports/m3_weights_mean_variance.csv")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
pd.DataFrame(model_ics).cumsum().plot(ax=ax[0], title="Cumulative IC by combination method")
ax[0].axhline(0, color="grey", lw=0.8)
weights["mean_variance"].loc[test_start:].plot.area(ax=ax[1], title="Walk-forward mean-variance weights", lw=0)
ax[1].set_ylim(0, 1)
ax[1].legend(loc="upper left", fontsize=8)
colours = ["#b04a4a" if k == "HINDSIGHT" else "#3b6ea5" for k in summary.index]
summary["mean_ic"].plot.bar(ax=ax[2], color=colours, title="Mean IC (red = hindsight, not achievable)")
ax[2].axhline(0, color="grey", lw=0.8)
ax[2].tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig("reports/m3_combination.png", dpi=150)
print("\nSaved reports/m3_combination.png and reports/m3_*.csv")
