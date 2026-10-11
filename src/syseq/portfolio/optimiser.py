"""Turn scores into portfolio weights.

Maximise the portfolio's score, subject to:
- fully invested, no short positions
- each stock within +/- max_stock of its benchmark weight
- each sector within +/- max_sector of its benchmark weight
- predicted tracking error no higher than the target
- portfolio beta close to the benchmark's
- trading no more than max_turnover of the portfolio (one-way)

"Active weight" means portfolio weight minus benchmark weight; it is the bet being taken.
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from syseq.risk.factor_model import RiskModel


@dataclass
class Limits:
    tracking_error: float = 0.025
    max_stock: float = 0.02
    max_sector: float = 0.06
    max_beta: float | None = 0.03
    max_turnover: float | None = 0.10      # one-way, per rebalance


@dataclass
class Result:
    weights: pd.Series
    status: str        # 'optimal', 'relaxed_turnover' or 'benchmark'


def optimise(score: pd.Series, benchmark: pd.Series, risk: RiskModel, sectors: pd.Series,
             limits: Limits = Limits(), beta: pd.Series | None = None,
             previous: pd.Series | None = None) -> Result:
    """All inputs are indexed by ticker; `benchmark` defines the investable set and sums to 1."""
    tickers = list(benchmark.index)
    n = len(tickers)
    b = benchmark.values
    alpha = score.reindex(tickers).fillna(0.0).values         # no score = no view
    rm = risk.subset(tickers)
    groups = sectors.reindex(tickers).fillna("Unknown")
    membership = pd.get_dummies(groups).values.T.astype(float)   # sectors x stocks

    w = cp.Variable(n)
    active = w - b
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.abs(active) <= limits.max_stock,
        cp.abs(membership @ active) <= limits.max_sector,
        cp.sum_squares(rm.exposures.T @ active)
        + cp.sum_squares(cp.multiply(np.sqrt(rm.specific_var), active)) <= limits.tracking_error ** 2,
    ]
    if beta is not None and limits.max_beta is not None:
        constraints.append(cp.abs(beta.reindex(tickers).fillna(1.0).values @ active) <= limits.max_beta)

    def solve(extra):
        problem = cp.Problem(cp.Maximize(alpha @ w), constraints + extra)
        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return None
        if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            return None
        out = np.clip(w.value, 0, None)
        out[out < 1e-6] = 0.0                                   # drop dust positions
        return pd.Series(out / out.sum(), index=tickers)

    if previous is not None and limits.max_turnover is not None:
        prev = previous.reindex(tickers).fillna(0.0).values
        forced = 1 - prev.sum()             # weight in stocks that left the investable set must be sold
        cap = limits.max_turnover + max(forced, 0)
        sol = solve([cp.norm1(w - prev) / 2 <= cap])
        if sol is not None:
            return Result(sol, "optimal")
        sol = solve([])                     # the limits could not all be met within the trading cap
        return Result(sol, "relaxed_turnover") if sol is not None else Result(benchmark.copy(), "benchmark")
    sol = solve([])
    return Result(sol, "optimal") if sol is not None else Result(benchmark.copy(), "benchmark")


def diagnostics(weights: pd.Series, benchmark: pd.Series, risk: RiskModel, sectors: pd.Series,
                beta: pd.Series | None = None, previous: pd.Series | None = None) -> dict:
    tickers = list(benchmark.index)
    w, b = weights.reindex(tickers).fillna(0.0), benchmark
    a = w - b
    rm = risk.subset(tickers)
    groups = sectors.reindex(tickers).fillna("Unknown")
    out = {
        "holdings": int((w > 1e-6).sum()),
        "benchmark_names": len(tickers),
        "active_share": float(a.abs().sum() / 2),
        "tracking_error": rm.volatility(a.values),
        "max_stock_active": float(a.abs().max()),
        "max_sector_active": float(a.groupby(groups).sum().abs().max()),
    }
    if beta is not None:
        out["active_beta"] = float((beta.reindex(tickers).fillna(1.0) * a).sum())
    if previous is not None:
        out["turnover"] = float((w - previous.reindex(tickers).fillna(0.0)).abs().sum() / 2
                                + previous.drop(tickers, errors="ignore").sum() / 2)
    return out
