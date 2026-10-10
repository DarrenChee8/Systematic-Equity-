"""Score a set of signals against a return panel and draw the standard three-panel chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .ic import ic_summary, rank_ic
from .quantiles import long_short_stats, quantile_returns


def evaluate_signals(signals: dict, returns: pd.DataFrame, mask: pd.DataFrame, start: str, q: int = 5):
    """Returns (summary table, monthly top-minus-bottom spreads, monthly ICs)."""
    rows, spreads, ics = {}, {}, {}
    for name, s in signals.items():
        ic = rank_ic(s, returns).loc[start:]
        qr = quantile_returns(s, returns, q).loc[start:]
        ics[name], spreads[name] = ic, qr.iloc[:, -1] - qr.iloc[:, 0]
        cover = (s.notna().sum(axis=1) / mask.sum(axis=1)).loc[start:].mean()
        rows[name] = {**ic_summary(ic)[["mean_ic", "t_stat", "hit_rate"]],
                      **long_short_stats(qr)[["ann_return", "sharpe", "max_drawdown"]],
                      "coverage": cover}
    table = pd.DataFrame(rows).T.rename(columns={"ann_return": "q5_minus_q1", "sharpe": "ls_sharpe"})
    return table, spreads, ics


def signal_figure(ic_columns: dict, spreads: dict, pay_corr: pd.DataFrame, themes: list[str], path: str):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    pd.DataFrame(ic_columns).plot.barh(ax=ax[0], title="Mean monthly rank IC")
    ax[0].axvline(0, color="grey", lw=0.8)
    ax[0].invert_yaxis()
    (pd.DataFrame({k: spreads[k] for k in themes}).cumsum() * 100).plot(
        ax=ax[1], title="Neutral top-minus-bottom quintile return (%, cumulative)")
    ax[1].axhline(0, color="grey", lw=0.8)
    im = ax[2].imshow(pay_corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[2].set_xticks(range(len(themes)), themes, rotation=45, ha="right")
    ax[2].set_yticks(range(len(themes)), themes)
    for i in range(len(themes)):
        for j in range(len(themes)):
            ax[2].text(j, i, f"{pay_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax[2].set_title("Correlation of neutral theme payoffs")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
