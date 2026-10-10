"""Quantile sorts: equal-weighted next-month return of each signal bucket."""
import numpy as np
import pandas as pd


def quantile_returns(signal: pd.DataFrame, fwd: pd.DataFrame, q: int = 5, min_names: int = 50) -> pd.DataFrame:
    rows = {}
    for d in signal.index.intersection(fwd.index):
        pair = pd.concat([signal.loc[d], fwd.loc[d]], axis=1, keys=["s", "r"]).dropna()
        if len(pair) < min_names:
            continue
        bucket = pd.qcut(pair["s"].rank(method="first"), q, labels=range(1, q + 1))
        rows[d] = pair.groupby(bucket, observed=True)["r"].mean()
    return pd.DataFrame(rows).T


def annualised_excess(qret: pd.DataFrame) -> pd.Series:
    """Annualised return of each bucket minus the average of all buckets."""
    excess = qret.sub(qret.mean(axis=1), axis=0)
    return (1 + excess).prod() ** (12 / len(excess)) - 1


def long_short_stats(qret: pd.DataFrame) -> pd.Series:
    ls = qret.iloc[:, -1] - qret.iloc[:, 0]
    return pd.Series({
        "ann_return": ls.mean() * 12,
        "ann_vol": ls.std() * np.sqrt(12),
        "sharpe": ls.mean() / ls.std() * np.sqrt(12),
        "max_drawdown": ((1 + ls).cumprod() / (1 + ls).cumprod().cummax() - 1).min(),
    })
