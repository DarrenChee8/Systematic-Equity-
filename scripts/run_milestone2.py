"""Milestone 2: value, quality, low-risk and momentum signals, scored within sectors.

Run from the repo root. The first run downloads prices again (with split data) and
fundamentals for each company from SEC EDGAR, which takes roughly 15-20 minutes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from syseq.data.edgar import TAGS, asof_panel, download_fundamentals, split_adjust
from syseq.data.free import FreeData
from syseq.evaluation.correlation import series_correlation, signal_correlation
from syseq.evaluation.ic import forward_returns, ic_summary, neutral_returns, rank_ic
from syseq.evaluation.quantiles import long_short_stats, quantile_returns
from syseq.signals.fundamental import quality_characteristics, value_characteristics
from syseq.signals.lowrisk import low_beta, low_volatility
from syseq.signals.momentum import momentum, monthly_prices
from syseq.signals.transforms import composite, sector_zscore

cfg = yaml.safe_load(Path("config/base.yaml").read_text())
START = cfg["start"]
src = FreeData("data")

# ---- data -------------------------------------------------------------------
members = src.membership(START, cfg["end"])
daily = src.prices(members.columns.tolist(), "2008-06-01", cfg["end"])   # total-return prices
monthly = monthly_prices(daily)
in_window = members.loc[START:].any()
universe = [t for t in monthly.columns if in_window.get(t, False)]

close, splits = src.market_data(universe, "2008-06-01", cfg["end"])
facts = download_fundamentals(universe, "data")
sectors = src.sectors()

dates = monthly.index
mask = members.reindex(index=dates, columns=monthly.columns).fillna(False).astype(bool)

# ---- point-in-time fundamentals --------------------------------------------
def panel(item, duration):
    return asof_panel(facts, TAGS[item], dates, duration).reindex(columns=monthly.columns)

net_income, cfo = panel("net_income", "annual"), panel("cfo", "annual")
equity, assets = panel("equity", "instant"), panel("assets", "instant")
share_facts = split_adjust(facts[facts["tag"].isin(TAGS["shares"])], splits)
shares = asof_panel(share_facts, TAGS["shares"], dates, "any", max_age_days=400).reindex(columns=monthly.columns)
market_cap = shares * close.resample("ME").last().reindex(index=dates, columns=monthly.columns)

# ---- characteristics -> sector-neutral signals ------------------------------
raw = {}
raw.update(value_characteristics(net_income, equity, cfo, market_cap))
raw.update(quality_characteristics(net_income, equity, cfo, assets))
raw["low_volatility"] = low_volatility(daily).reindex(dates)
raw["low_beta"] = low_beta(daily).reindex(dates)
m = cfg["signals"]["momentum"]
raw["momentum"] = momentum(monthly, m["lookback_months"], m["skip_months"])

z = {k: sector_zscore(v.where(mask), sectors) for k, v in raw.items()}
themes = {
    "VALUE": composite([z["earnings_yield"], z["book_to_price"], z["cashflow_yield"]], sectors),
    "QUALITY": composite([z["roe"], z["cash_profitability"], z["low_accruals"], z["low_leverage"]], sectors),
    "LOW_RISK": composite([z["low_volatility"], z["low_beta"]], sectors),
    "MOMENTUM": z["momentum"],
}
themes["COMBINED"] = composite(list(themes.values()), sectors)   # equal-weight preview of milestone 3
signals = {**z, **themes}

# ---- sanity check: do the fundamentals look right for names we know? --------
check = [t for t in ["AAPL", "MSFT", "JPM", "XOM", "WMT", "JNJ"] if t in monthly.columns]
last = market_cap[check].dropna(how="all").index[-1]
sanity = pd.DataFrame({
    "market_cap_bn": market_cap.loc[last, check] / 1e9,
    "net_income_bn": net_income.loc[last, check] / 1e9,
    "equity_bn": equity.loc[last, check] / 1e9,
    "assets_bn": assets.loc[last, check] / 1e9,
    "pe": market_cap.loc[last, check] / net_income.loc[last, check],
    "roe": raw["roe"].loc[last, check],
})

# ---- evaluation -------------------------------------------------------------
# "raw": plain next-month returns. "neutral": market (beta) and sector effects removed,
# which is the fair test for a stock-selection signal.
fwd = forward_returns(monthly)
beta = 0.67 * (-raw["low_beta"]) + 0.33   # shrink estimated betas toward 1 (Blume adjustment)
fwd_neutral = neutral_returns(fwd, beta, sectors, mask)


def evaluate(returns):
    rows, spreads = {}, {}
    for name, s in signals.items():
        ic = rank_ic(s, returns).loc[START:]
        q = quantile_returns(s, returns, cfg["evaluation"]["quantiles"]).loc[START:]
        spreads[name] = q.iloc[:, -1] - q.iloc[:, 0]
        cover = (s.notna().sum(axis=1) / mask.sum(axis=1)).loc[START:].mean()
        rows[name] = {**ic_summary(ic)[["mean_ic", "t_stat", "hit_rate"]],
                      **long_short_stats(q)[["ann_return", "sharpe", "max_drawdown"]],
                      "coverage": cover}
    table = pd.DataFrame(rows).T.rename(columns={"ann_return": "q5_minus_q1", "sharpe": "ls_sharpe"})
    return table, spreads


summary_raw, spreads_raw = evaluate(fwd)
summary, spreads = evaluate(fwd_neutral)

theme_names = list(themes)
sig_corr = signal_correlation({k: signals[k].loc[START:] for k in theme_names})
pay_corr = series_correlation({k: spreads[k] for k in theme_names})

pd.set_option("display.width", 200, "display.max_columns", None)
print(f"\nSanity check at {last.date()} (compare with figures you know)\n", sanity.round(2))
print(f"\nSectors known for {sectors.reindex(universe).notna().mean():.0%} of universe tickers")
print("\nRAW returns: signal summary\n", summary_raw.round(3))
print("\nMARKET- AND SECTOR-NEUTRAL returns: signal summary\n", summary.round(3))
print("\nCorrelation between theme scores (are they picking the same stocks?)\n", sig_corr.round(2))
print("\nCorrelation between neutral theme payoffs (do they work at the same times?)\n", pay_corr.round(2))

Path("reports").mkdir(exist_ok=True)
summary.round(4).to_csv("reports/m2_signal_summary.csv")
summary_raw.round(4).to_csv("reports/m2_signal_summary_raw.csv")
sig_corr.round(3).to_csv("reports/m2_theme_score_correlation.csv")
pay_corr.round(3).to_csv("reports/m2_theme_payoff_correlation.csv")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
pd.DataFrame({"raw": summary_raw["mean_ic"], "neutral": summary["mean_ic"]}).plot.barh(
    ax=ax[0], title="Mean monthly rank IC")
ax[0].axvline(0, color="grey", lw=0.8); ax[0].invert_yaxis()
(pd.DataFrame({k: spreads[k] for k in theme_names}).cumsum() * 100).plot(
    ax=ax[1], title="Neutral top-minus-bottom quintile return (%, cumulative)")
ax[1].axhline(0, color="grey", lw=0.8)
im = ax[2].imshow(pay_corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax[2].set_xticks(range(len(theme_names)), theme_names, rotation=45, ha="right")
ax[2].set_yticks(range(len(theme_names)), theme_names)
for i in range(len(theme_names)):
    for j in range(len(theme_names)):
        ax[2].text(j, i, f"{pay_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax[2].set_title("Correlation of neutral theme payoffs")
fig.colorbar(im, ax=ax[2], fraction=0.046)
fig.tight_layout()
fig.savefig("reports/m2_signals.png", dpi=150)
print("\nSaved reports/m2_signals.png and reports/m2_*.csv")
