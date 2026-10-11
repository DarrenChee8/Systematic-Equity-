import numpy as np
import pandas as pd

from syseq.backtest.attribution import attribute
from syseq.backtest.performance import (calendar_years, net_of_costs, period_returns, summary,
                                        traded_value)

IDX = pd.date_range("2020-01-31", periods=24, freq="ME")
COLS = [f"S{i:02d}" for i in range(60)]


def _world(seed=0):
    rng = np.random.default_rng(seed)
    fwd = pd.DataFrame(rng.normal(0.01, 0.05, (24, 60)), IDX, COLS)
    bench = pd.DataFrame(1 / 60, IDX, COLS)
    tilt = pd.DataFrame(rng.normal(0, 0.004, (24, 60)), IDX, COLS)
    w = (bench + tilt).clip(lower=0)
    return fwd, bench, w.div(w.sum(axis=1), axis=0)


def test_period_returns_and_missing_prices():
    fwd, bench, w = _world()
    r = period_returns(w, bench, fwd)
    d = IDX[3]
    assert np.isclose(r.loc[d, "portfolio"], (w.loc[d] * fwd.loc[d]).sum())
    assert np.isclose(r.loc[d, "active"], r.loc[d, "portfolio"] - r.loc[d, "benchmark"])
    fwd.loc[d, "S00"] = np.nan
    r2 = period_returns(w, bench, fwd)
    assert np.isclose(r2.loc[d, "weight_without_return"], w.loc[d, "S00"])


def test_traded_value_and_costs():
    fwd, bench, w = _world()
    still = pd.DataFrame(np.tile(w.iloc[0].values, (24, 1)), IDX, COLS)
    flat = fwd * 0
    t = traded_value(still, flat)
    assert t.iloc[0] == 1.0 and np.allclose(t.iloc[1:], 0)           # no price moves, no changes: no trades
    swap = still.copy()
    swap.iloc[1:, 0], swap.iloc[1:, 1] = still.iloc[0, 1], still.iloc[0, 0]
    t2 = traded_value(swap, flat)
    assert np.isclose(t2.iloc[1], 2 * abs(still.iloc[0, 0] - still.iloc[0, 1]))
    perf = net_of_costs(period_returns(w, bench, fwd), traded_value(w, fwd), 10)
    assert np.allclose(perf["portfolio"] - perf["portfolio_net"], perf["cost"]) and (perf["cost"] > 0).all()
    assert np.isclose(perf["cost"].iloc[0], 10 / 1e4)                # buying the whole portfolio at 10bp


def test_summary_and_calendar_years():
    b = pd.Series(0.01, IDX)
    p = b + 0.001
    p.iloc[::2] += 0.002
    s = summary(p, b)
    assert s["excess_return"] > 0 and s["tracking_error"] > 0 and s["hit_rate"] == 1.0
    assert np.isclose(s["benchmark_return"], 1.01 ** 12 - 1)
    y = calendar_years(p, b)
    assert y.loc[2020, "months"] == 11 and y.loc[2021, "months"] == 12   # January's weights earn February's return
    assert np.isclose(y.loc[2021, "benchmark"], 1.01 ** 12 - 1)


def test_attribution_adds_up_and_finds_the_driver():
    rng = np.random.default_rng(2)
    good = pd.DataFrame(rng.normal(size=(24, 60)), IDX, COLS)
    noise = pd.DataFrame(rng.normal(size=(24, 60)), IDX, COLS)
    fwd = 0.02 * good + pd.DataFrame(rng.normal(0, 0.01, (24, 60)), IDX, COLS)   # returns driven by 'good'
    bench = pd.DataFrame(1 / 60, IDX, COLS)
    w = (bench + 0.004 * good).clip(lower=0)
    w = w.div(w.sum(axis=1), axis=0)                                             # portfolio tilts to 'good'
    sectors = pd.Series(np.repeat(["A", "B", "C"], 20), index=COLS)
    beta = pd.DataFrame(1.0, IDX, COLS)
    cap = pd.DataFrame(1e9, IDX, COLS)
    a = attribute(w, bench, fwd, {"GOOD": good, "NOISE": noise}, beta, sectors, cap)
    active = period_returns(w, bench, fwd)["active"]
    assert np.allclose(a.sum(axis=1), active.loc[a.index])                       # contributions sum to the total
    assert a["GOOD"].mean() > 5 * abs(a["NOISE"].mean()) and a["GOOD"].mean() > 0


def _engine_inputs(seed=0, months=40, n=60):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2017-01-02", periods=21 * (months + 26))
    tick = [f"S{i:02d}" for i in range(n)]
    market = rng.normal(0.0004, 0.01, len(days))
    daily_ret = np.outer(market, np.linspace(0.7, 1.3, n)) + rng.normal(0, 0.015, (len(days), n))
    daily = pd.DataFrame(100 * np.cumprod(1 + daily_ret, axis=0), days, tick)
    monthly = daily.resample("ME").last()
    fwd = monthly.shift(-1) / monthly - 1
    idx = monthly.index
    scores = pd.DataFrame(rng.normal(size=(len(idx), n)), idx, tick)
    cap = pd.DataFrame(np.tile(rng.lognormal(3, 1, n), (len(idx), 1)), idx, tick)
    mask = pd.DataFrame(True, idx, tick)
    sectors = pd.Series(np.repeat(["A", "B", "C", "D"], n // 4), index=tick)
    beta = pd.DataFrame(1.0, idx, tick)
    return scores, cap, daily, monthly, mask, sectors, beta, fwd, str(idx[26].date())


def test_engine_calibration_uses_only_past_outcomes():
    from syseq.backtest.engine import run
    from syseq.portfolio.optimiser import Limits
    scores, cap, daily, monthly, mask, sectors, beta, fwd, start = _engine_inputs()
    lim = Limits(tracking_error=0.03, max_stock=0.05, max_sector=0.2, max_beta=None, max_turnover=None)
    args = (scores, cap, daily, monthly, mask, sectors, beta)
    kw = dict(risk_window=252, n_factors=5, verbose=False, calibrate=True, calibration_min=6, calibration_window=12)
    w, b, d = run(*args, fwd, start, lim, **kw)
    scale = d["risk_scale"].astype(float)
    assert (scale.iloc[:6] == 1.0).all() and scale.between(0.7, 2.5).all() and (scale.iloc[6:] != 1.0).any()
    assert np.allclose(d["tracking_error_calibrated"].astype(float),
                       d["tracking_error"].astype(float) * scale)
    assert (d["tracking_error_calibrated"].astype(float) <= 0.03 + 1e-4).all()
    k = 20                                               # corrupt returns from month k onward
    bad = fwd.copy()
    bad.loc[w.index[k]:] = bad.loc[w.index[k]:] * 5
    w2, _, d2 = run(*args, bad, start, lim, **kw)
    assert np.allclose(w.iloc[: k + 1].values, w2.iloc[: k + 1].values, atol=1e-6)   # decisions up to k unchanged
    assert d2["risk_scale"].astype(float).iloc[k + 1:].max() > scale.iloc[k + 1:].max()


def test_recency_weighted_risk_model_reacts_to_recent_volatility():
    from syseq.risk.factor_model import estimate
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, (500, 30))
    r[-60:] *= 3                                          # the last three months are three times as volatile
    df = pd.DataFrame(r, pd.bdate_range("2019-01-01", periods=500), [f"S{i}" for i in range(30)])
    w = np.full(30, 1 / 30)
    assert estimate(df, 5, halflife=42).volatility(w) > 1.4 * estimate(df, 5).volatility(w)
