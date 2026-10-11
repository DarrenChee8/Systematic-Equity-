"""Monthly rebalancing loop.

At each month-end: define the investable set and benchmark, estimate risk from past returns,
optimise, record the holdings. Between rebalances, weights drift with returns. Performance,
costs and attribution are computed from the recorded weights in milestone 5.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from syseq.portfolio.optimiser import Limits, diagnostics, optimise
from syseq.risk.factor_model import estimate


def benchmark_weights(market_cap: pd.Series, eligible: list[str]) -> pd.Series:
    """Capitalisation weights over the investable set, an approximation of the index's own weighting."""
    cap = market_cap.reindex(eligible).dropna()
    cap = cap[cap > 0]
    return cap / cap.sum()


def drift(weights: pd.Series, returns: pd.Series) -> pd.Series:
    """Weights after one period of returns, before trading. A missing return is treated as zero."""
    grown = weights * (1 + returns.reindex(weights.index).fillna(0.0))
    return grown / grown.sum()


def run(scores: pd.DataFrame, market_cap: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame,
        mask: pd.DataFrame, sectors: pd.Series, beta: pd.DataFrame, fwd: pd.DataFrame,
        start: str, limits: Limits = Limits(), risk_window: int = 504, n_factors: int = 15,
        verbose: bool = True, issuer: pd.Series | None = None, halflife: float | None = None,
        calibrate: bool = False, calibration_window: int = 36, calibration_min: int = 24):
    """Returns (portfolio weights, benchmark weights, diagnostics), each indexed by rebalance date.

    calibrate: correct the risk model for its own track record. Each month the realised active
    return is divided by what the model predicted; if those ratios have been more volatile than 1
    over the past `calibration_window` months, the model has been under-predicting, and the
    tracking-error budget is tightened by that factor. Uses past months only.
    """
    daily_ret = daily.pct_change(fill_method=None)
    dates = [d for d in scores.loc[start:].index if d <= daily.index[-1]]
    weights, bench, diag = {}, {}, {}
    previous, last, surprises = None, None, []
    for i, d in enumerate(dates):
        window = daily_ret.loc[:d].iloc[-risk_window:]
        risk = estimate(window, n_factors, halflife=halflife)
        if last is not None and last in fwd.index:          # last month's outcome is now known
            r = fwd.loc[last]
            realised = float(((weights[last] - bench[last]).reindex(r.index).fillna(0.0) * r.fillna(0.0)).sum())
            predicted = diag[last]["tracking_error"] / np.sqrt(12)
            if predicted > 1e-6:                          # skip months held at the benchmark
                surprises.append(realised / predicted)
        scale = 1.0
        if calibrate and len(surprises) >= calibration_min:
            scale = float(np.clip(np.std(surprises[-calibration_window:]), 0.7, 2.5))
        month_limits = replace(limits, tracking_error=limits.tracking_error / scale)
        priced = monthly.loc[d].dropna().index
        eligible = [t for t in mask.columns[mask.loc[d]] if t in risk.tickers and t in priced]
        b = benchmark_weights(market_cap.loc[d], eligible)
        if len(b) < 50:
            continue
        res = optimise(scores.loc[d], b, risk, sectors, month_limits, beta.loc[d], previous, issuer)
        weights[d], bench[d] = res.weights, b
        diag[d] = {**diagnostics(res.weights, b, risk, sectors, beta.loc[d], previous, issuer),
                   "risk_scale": scale, "status": res.status}
        diag[d]["tracking_error_calibrated"] = diag[d]["tracking_error"] * scale
        last = d
        previous = drift(res.weights, fwd.loc[d]) if d in fwd.index else res.weights
        if verbose and (i % 24 == 0 or i == len(dates) - 1):
            print(f"  rebalanced {d.date()}  holdings {diag[d]['holdings']}  "
                  f"tracking error {diag[d]['tracking_error']:.2%}", flush=True)
    return (pd.DataFrame(weights).T.fillna(0.0), pd.DataFrame(bench).T.fillna(0.0), pd.DataFrame(diag).T)
