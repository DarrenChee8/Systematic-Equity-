"""Milestone 1: download data, build 12-1 momentum, test it. Run from the repo root."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from syseq.data.free import FreeData
from syseq.evaluation.ic import forward_returns, ic_decay, ic_summary, rank_ic
from syseq.evaluation.quantiles import annualised_excess, long_short_stats, quantile_returns
from syseq.signals.momentum import momentum, monthly_prices
from syseq.signals.transforms import winsorise, zscore

cfg = yaml.safe_load(Path("config/base.yaml").read_text())
src = FreeData("data")

members = src.membership(cfg["start"], cfg["end"])
daily = src.prices(members.columns.tolist(), "2008-06-01", cfg["end"])  # extra history for lookback
monthly = monthly_prices(daily)

m = cfg["signals"]["momentum"]
raw = momentum(monthly, m["lookback_months"], m["skip_months"])

# keep only stocks that were in the index on each date (point-in-time universe)
mask = members.reindex(index=raw.index, columns=raw.columns).fillna(False).astype(bool)
signal = zscore(winsorise(raw.where(mask)))
fwd = forward_returns(monthly)

coverage = (signal.notna().sum(axis=1) / mask.sum(axis=1)).loc[cfg["start"]:]
ic = rank_ic(signal, fwd).loc[cfg["start"]:]
qret = quantile_returns(signal, fwd, cfg["evaluation"]["quantiles"]).loc[cfg["start"]:]

print(f"\nPrice coverage of index members: mean {coverage.mean():.0%}, min {coverage.min():.0%}")
print("(below 100% = delisted stocks Yahoo no longer carries: survivorship bias)\n")
print("IC summary\n", ic_summary(ic).round(3), "\n")
print("Quintile annualised excess return\n", annualised_excess(qret).round(4), "\n")
print("Top minus bottom quintile\n", long_short_stats(qret).round(3))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
annualised_excess(qret).mul(100).plot.bar(ax=ax[0], title="Annualised excess return by quintile (%)")
ax[0].set_xlabel("Quintile (1 = lowest momentum)")
ic.rolling(12).mean().plot(ax=ax[1], title="Rolling 12-month mean IC")
ax[1].axhline(0, color="grey", lw=0.8)
ic_decay(signal, monthly).plot.bar(ax=ax[2], title="IC decay (months ahead)")
fig.tight_layout()
Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/m1_momentum.png", dpi=150)
print("\nSaved reports/m1_momentum.png")
