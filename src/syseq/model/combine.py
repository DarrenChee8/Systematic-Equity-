"""Combine theme scores into one forecast, with weights that use past information only.

At month-end t the latest usable evidence is the IC for month t-1: that IC compares scores
at t-1 with the return from t-1 to t, which has just been observed. Weights for t therefore
use ICs up to and including t-1 (the `shift(1)` below).
"""
import numpy as np
import pandas as pd

from syseq.signals.transforms import sector_zscore


def _normalise(w: np.ndarray) -> np.ndarray:
    """Negative weights are set to zero; if nothing is positive, fall back to equal weights."""
    w = np.where(np.isfinite(w), np.clip(w, 0, None), 0.0)
    return w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))


def weight_equal(ics: pd.DataFrame) -> np.ndarray:
    return np.full(ics.shape[1], 1 / ics.shape[1])


def weight_ic(ics: pd.DataFrame) -> np.ndarray:
    """In proportion to each theme's average IC."""
    return _normalise(ics.mean().values)


def weight_icir(ics: pd.DataFrame) -> np.ndarray:
    """In proportion to IC consistency (mean / standard deviation)."""
    return _normalise((ics.mean() / ics.std()).values)


def weight_mean_variance(ics: pd.DataFrame, shrink: float = 0.5) -> np.ndarray:
    """Maximise combined IC per unit of IC risk, allowing for correlation between themes.

    w is proportional to inverse(covariance) x mean. The covariance is shrunk halfway to its
    diagonal, because a covariance matrix estimated from a short history is unreliable.
    """
    cov = ics.cov().values
    cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
    try:
        return _normalise(np.linalg.solve(cov, ics.mean().values))
    except np.linalg.LinAlgError:
        return weight_equal(ics)


METHODS = {"equal": weight_equal, "ic": weight_ic, "ic_ir": weight_icir,
           "mean_variance": weight_mean_variance}


def walk_forward_weights(ics: pd.DataFrame, method: str, min_months: int = 36,
                         window: int | None = None) -> pd.DataFrame:
    """Weights per month from ICs observed before that month.

    min_months: equal weights are used until this much history exists.
    window: None = all history so far (expanding); a number = only the latest `window` months.
    """
    fn = METHODS[method]
    known = ics.shift(1)                       # at month t, only ICs through t-1 are known
    rows = {}
    for i, d in enumerate(ics.index):
        hist = known.iloc[: i + 1].dropna(how="any")
        enough = len(hist) >= min_months
        if window:
            hist = hist.iloc[-window:]
        rows[d] = fn(hist) if enough else weight_equal(ics)
    return pd.DataFrame(rows, index=ics.columns).T


def in_sample_weights(ics: pd.DataFrame, method: str = "mean_variance") -> pd.DataFrame:
    """Weights fitted on the whole sample and applied to all of it: hindsight, for comparison only."""
    w = METHODS[method](ics.dropna(how="any"))
    return pd.DataFrame(np.tile(w, (len(ics), 1)), index=ics.index, columns=ics.columns)


def combine(themes: dict[str, pd.DataFrame], weights: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Weighted average of theme scores per stock, ignoring themes a stock lacks, then re-standardised."""
    names = list(weights.columns)
    ref = themes[names[0]]
    w = weights.reindex(ref.index)
    total = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)
    weight_sum = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)
    for k in names:
        s = themes[k]
        total = total.add(s.mul(w[k], axis=0).fillna(0.0))
        weight_sum = weight_sum.add(s.notna().mul(w[k], axis=0))
    return sector_zscore(total / weight_sum.where(weight_sum > 0), sectors)
