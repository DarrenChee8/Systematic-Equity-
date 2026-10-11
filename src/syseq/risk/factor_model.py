"""Statistical factor risk model.

Estimating how 500 stocks co-move directly needs about 125,000 correlations from a few hundred
days of data, which is mostly noise. A factor model assumes instead that stocks move together
only through a small number of common factors (found here by principal components), plus a
stock-specific part that is independent across stocks:

    covariance = X X' + diag(specific variance)

where X holds each stock's exposure to each factor, scaled by the factor's volatility.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskModel:
    tickers: list[str]
    exposures: np.ndarray        # N x K, annualised: X X' is the common-factor covariance
    specific_var: np.ndarray     # N, annualised

    def covariance(self) -> np.ndarray:
        return self.exposures @ self.exposures.T + np.diag(self.specific_var)

    def variance(self, weights: np.ndarray) -> float:
        common = self.exposures.T @ weights
        return float(common @ common + (self.specific_var * weights ** 2).sum())

    def volatility(self, weights: np.ndarray) -> float:
        return float(np.sqrt(self.variance(weights)))

    def subset(self, tickers: list[str]) -> "RiskModel":
        pos = [self.tickers.index(t) for t in tickers]
        return RiskModel(list(tickers), self.exposures[pos], self.specific_var[pos])


def estimate(daily_returns: pd.DataFrame, n_factors: int = 15, min_coverage: float = 0.8,
             periods_per_year: int = 252, halflife: float | None = None) -> RiskModel:
    """Fit on a window of daily returns (rows: days, columns: tickers).

    Stocks with too little history in the window are left out of the model.
    halflife (in days) weights recent days more heavily: a day `halflife` days ago counts
    half as much as today. None weights every day equally.
    """
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)   # tame data glitches
    r = daily_returns.loc[:, daily_returns.notna().mean() >= min_coverage]
    r = r.dropna(how="all")
    t, n = r.shape
    k = min(n_factors, n - 1, t - 1)
    if halflife is None:
        x = (r - r.mean()).fillna(0.0).values
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        loadings = vt[:k].T                                   # N x K
        factor_var = s[:k] ** 2 / (t - 1)
        resid = x - (x @ loadings) @ loadings.T
        specific = resid.var(axis=0, ddof=1)
    else:
        w = 0.5 ** (np.arange(t - 1, -1, -1) / halflife)
        w = w / w.sum()
        mean = (r.fillna(0.0).values * w[:, None]).sum(axis=0) / np.maximum(
            (r.notna().values * w[:, None]).sum(axis=0), 1e-12)
        x = (r - mean).fillna(0.0).values * np.sqrt(w)[:, None]   # rows scaled so x'x is the weighted covariance
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        loadings = vt[:k].T
        factor_var = s[:k] ** 2
        resid = x - (x @ loadings) @ loadings.T
        specific = (resid ** 2).sum(axis=0)
    specific = np.maximum(specific, 0.1 * np.median(specific))   # floor: no stock is risk-free
    return RiskModel(list(r.columns), loadings * np.sqrt(factor_var * periods_per_year),
                     specific * periods_per_year)
