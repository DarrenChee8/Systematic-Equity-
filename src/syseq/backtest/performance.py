"""Measure what the recorded holdings earned: returns, trading costs and summary statistics."""
import numpy as np
import pandas as pd

from .engine import drift


def period_returns(weights: pd.DataFrame, benchmark: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """Return of portfolio and benchmark over the month after each rebalance.

    Weights at month-end t earn the return from t to t+1. A stock with no return that month
    (for example delisted, with no price) is given zero, and the weight affected is reported.
    """
    rows = {}
    for d in weights.index.intersection(fwd.index):
        r = fwd.loc[d]
        if r.notna().sum() == 0:
            continue
        w, b = weights.loc[d], benchmark.loc[d]
        rows[d] = {"portfolio": float((w * r.reindex(w.index).fillna(0.0)).sum()),
                   "benchmark": float((b * r.reindex(b.index).fillna(0.0)).sum()),
                   "weight_without_return": float(w[r.reindex(w.index).isna()].sum())}
    out = pd.DataFrame(rows).T
    out["active"] = out["portfolio"] - out["benchmark"]
    return out


def traded_value(weights: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Buys plus sells at each rebalance, as a fraction of the portfolio.

    Compares the new target with the previous holdings after a month of price drift.
    The first rebalance buys the whole portfolio (traded value 1).
    """
    out, previous = {}, None
    for d in weights.index:
        w = weights.loc[d]
        w = w[w > 0]
        if previous is None:
            out[d] = 1.0
        else:
            names = w.index.union(previous.index)
            out[d] = float((w.reindex(names).fillna(0.0) - previous.reindex(names).fillna(0.0)).abs().sum())
        previous = drift(w, fwd.loc[d]) if d in fwd.index else w
    return pd.Series(out)


def net_of_costs(returns: pd.DataFrame, traded: pd.Series, rate_bps: float) -> pd.DataFrame:
    """Charge rate_bps per unit traded against the portfolio. The benchmark is treated as costless."""
    out = returns.copy()
    out["cost"] = traded.reindex(out.index).fillna(0.0) * rate_bps / 1e4
    out["portfolio_net"] = out["portfolio"] - out["cost"]
    out["active_net"] = out["portfolio_net"] - out["benchmark"]
    return out


def _annual(r: pd.Series, periods: int = 12) -> float:
    return float((1 + r).prod() ** (periods / len(r)) - 1)


def _drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summary(portfolio: pd.Series, benchmark: pd.Series, periods: int = 12) -> pd.Series:
    active = portfolio - benchmark
    te = active.std() * np.sqrt(periods)
    excess = _annual(portfolio, periods) - _annual(benchmark, periods)
    return pd.Series({
        "return": _annual(portfolio, periods),
        "benchmark_return": _annual(benchmark, periods),
        "excess_return": excess,
        "tracking_error": te,
        "information_ratio": excess / te if te > 0 else np.nan,
        "t_stat": active.mean() / active.std() * np.sqrt(len(active)) if active.std() > 0 else np.nan,
        "volatility": portfolio.std() * np.sqrt(periods),
        "benchmark_volatility": benchmark.std() * np.sqrt(periods),
        "beta": portfolio.cov(benchmark) / benchmark.var(),
        "hit_rate": float((active > 0).mean()),
        "max_drawdown": _drawdown(portfolio),
        "benchmark_max_drawdown": _drawdown(benchmark),
        "worst_relative_drawdown": float(((1 + active).cumprod() / (1 + active).cumprod().cummax() - 1).min()),
    })


def calendar_years(portfolio: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Compound returns by the calendar year in which each month's return was earned."""
    year = (portfolio.index + pd.offsets.MonthEnd(1)).year
    out = pd.DataFrame({"portfolio": (1 + portfolio).groupby(year).prod() - 1,
                        "benchmark": (1 + benchmark).groupby(year).prod() - 1,
                        "months": portfolio.groupby(year).size()})
    out["excess"] = out["portfolio"] - out["benchmark"]
    return out[["portfolio", "benchmark", "excess", "months"]]
