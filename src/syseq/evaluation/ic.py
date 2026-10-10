"""Information coefficient: rank correlation between today's signal and next period's return."""
import numpy as np
import pandas as pd


def forward_returns(monthly: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return from month-end t to t+horizon, stored at t."""
    return monthly.shift(-horizon) / monthly - 1


def rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 20) -> pd.Series:
    out = {}
    for d in signal.index.intersection(fwd.index):
        pair = pd.concat([signal.loc[d], fwd.loc[d]], axis=1).dropna()
        if len(pair) >= min_names:
            out[d] = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return pd.Series(out, name="ic")


def ic_summary(ic: pd.Series) -> pd.Series:
    n = ic.count()
    return pd.Series({
        "mean_ic": ic.mean(),
        "ic_std": ic.std(),
        "ic_ir": ic.mean() / ic.std(),                  # consistency of the signal
        "t_stat": ic.mean() / ic.std() * np.sqrt(n),
        "hit_rate": (ic > 0).mean(),
        "months": n,
    })


def ic_decay(signal: pd.DataFrame, monthly: pd.DataFrame, horizons=range(1, 13)) -> pd.Series:
    """Mean IC against the single month h months ahead: shows how fast the signal fades."""
    one = forward_returns(monthly, 1)
    return pd.Series({h: rank_ic(signal, one.shift(-(h - 1))).mean() for h in horizons}, name="ic")


def neutral_returns(fwd: pd.DataFrame, beta: pd.DataFrame, sectors: pd.Series,
                    mask: pd.DataFrame) -> pd.DataFrame:
    """Forward returns with market and sector effects removed.

    Step 1: subtract beta x market return (market = equal-weighted universe), using the
    beta known at the start of the month. Step 2: subtract the sector average.
    What is left is the stock-specific return a stock-selection signal should predict.
    """
    f = fwd.where(mask)
    resid = f - beta.mul(f.mean(axis=1), axis=0)
    groups = sectors.reindex(resid.columns).fillna("Unknown")
    out = resid.copy()
    for _, cols in groups.groupby(groups).groups.items():
        out[cols] = resid[cols].sub(resid[cols].mean(axis=1), axis=0)
    return out
