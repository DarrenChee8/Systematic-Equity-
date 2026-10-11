"""Where did the active return come from?

Each month, stock returns are explained by a regression on sector, beta and the theme scores:

    return = sector effect + beta x market effect + sum(theme score x theme payoff) + residual

The portfolio's active return then splits exactly into:
- each theme: (active exposure to the theme) x (that theme's payoff this month)
- beta and sectors: the same, for the exposures the optimiser is meant to keep small
- stock-specific: active weights x residuals, the part no common effect explains
"""
import numpy as np
import pandas as pd


def attribute(weights: pd.DataFrame, benchmark: pd.DataFrame, fwd: pd.DataFrame,
              themes: dict[str, pd.DataFrame], beta: pd.DataFrame, sectors: pd.Series,
              market_cap: pd.DataFrame) -> pd.DataFrame:
    """Monthly contributions to active return. Columns: themes, 'beta', 'sectors', 'stock_specific'."""
    names = list(themes)
    rows = {}
    for d in weights.index.intersection(fwd.index):
        b = benchmark.loc[d]
        r = fwd.loc[d].reindex(b.index[b > 0]).dropna()
        if len(r) < 50:
            continue
        tickers = r.index
        active = (weights.loc[d].reindex(tickers).fillna(0.0) - b.reindex(tickers)).values
        dummies = pd.get_dummies(sectors.reindex(tickers).fillna("Unknown")).astype(float)
        bt = beta.loc[d].reindex(tickers).fillna(1.0).values
        z = np.column_stack([themes[k].loc[d].reindex(tickers).fillna(0.0).values for k in names])
        x = np.column_stack([dummies.values, bt - bt.mean(), z])
        sw = np.sqrt(market_cap.loc[d].reindex(tickers).fillna(market_cap.loc[d].median()).clip(lower=1).values)
        coef, *_ = np.linalg.lstsq(x * sw[:, None], r.values * sw, rcond=None)
        resid = r.values - x @ coef
        n_sec = dummies.shape[1]
        out = {k: float((active @ z[:, i]) * coef[n_sec + 1 + i]) for i, k in enumerate(names)}
        out["beta"] = float((active @ (bt - bt.mean())) * coef[n_sec])
        out["sectors"] = float((active @ dummies.values) @ coef[:n_sec])
        out["stock_specific"] = float(active @ resid)
        rows[d] = out
    return pd.DataFrame(rows).T
