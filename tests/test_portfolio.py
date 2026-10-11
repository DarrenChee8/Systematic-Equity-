import numpy as np
import pandas as pd

from syseq.backtest.engine import benchmark_weights, drift
from syseq.portfolio.optimiser import Limits, diagnostics, optimise
from syseq.risk.factor_model import estimate

TICKERS = [f"S{i:02d}" for i in range(60)]


def _returns(seed=0, days=500):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, days)
    betas = np.linspace(0.6, 1.4, 60)
    r = np.outer(market, betas) + rng.normal(0, 0.012, (days, 60))
    return pd.DataFrame(r, index=pd.bdate_range("2019-01-01", periods=days), columns=TICKERS), betas


def _setup(seed=0):
    r, betas = _returns(seed)
    rng = np.random.default_rng(seed + 1)
    cap = pd.Series(rng.lognormal(3, 1, 60), index=TICKERS)
    bench = cap / cap.sum()
    score = pd.Series(rng.normal(size=60), index=TICKERS)
    sectors = pd.Series(np.repeat(["A", "B", "C", "D"], 15), index=TICKERS)
    return estimate(r, 5), bench, score, sectors, pd.Series(betas, index=TICKERS)


def test_risk_model_matches_sample_risk():
    r, betas = _returns()
    rm = estimate(r, 5)
    w = np.full(60, 1 / 60)
    realised = (r @ w).std() * np.sqrt(252)
    assert abs(rm.volatility(w) / realised - 1) < 0.1          # market-like portfolio
    assert np.allclose(rm.covariance(), rm.covariance().T) and (rm.specific_var > 0).all()
    assert rm.subset(TICKERS[:10]).exposures.shape == (10, 5)
    high, low = np.zeros(60), np.zeros(60)
    high[-10:], low[:10] = 0.1, 0.1
    assert rm.volatility(high) > rm.volatility(low)             # high-beta stocks are riskier


def test_risk_model_skips_short_histories():
    r, _ = _returns()
    r.iloc[:400, 3] = np.nan
    assert "S03" not in estimate(r, 5).tickers


def test_optimiser_respects_every_limit():
    rm, bench, score, sectors, beta = _setup()
    lim = Limits(tracking_error=0.02, max_stock=0.02, max_sector=0.03, max_beta=0.02, max_turnover=None)
    res = optimise(score, bench, rm, sectors, lim, beta)
    w = res.weights
    d = diagnostics(w, bench, rm, sectors, beta)
    assert res.status == "optimal" and np.isclose(w.sum(), 1) and (w >= 0).all()
    assert d["tracking_error"] <= 0.02 + 1e-4 and d["max_stock_active"] <= 0.02 + 1e-5
    assert d["max_sector_active"] <= 0.03 + 1e-5 and abs(d["active_beta"]) <= 0.02 + 1e-5
    assert (w * score).sum() > (bench * score).sum()            # tilts toward high scores


def test_no_view_gives_the_benchmark_and_tighter_risk_means_smaller_bets():
    rm, bench, score, sectors, beta = _setup()
    flat = optimise(score * 0, bench, rm, sectors, Limits(max_turnover=None), beta, previous=None)
    assert diagnostics(flat.weights, bench, rm, sectors)["tracking_error"] <= 0.025 + 1e-4
    tight = optimise(score, bench, rm, sectors, Limits(tracking_error=0.005, max_turnover=None), beta)
    loose = optimise(score, bench, rm, sectors, Limits(tracking_error=0.03, max_turnover=None), beta)
    assert diagnostics(tight.weights, bench, rm, sectors)["active_share"] < \
        diagnostics(loose.weights, bench, rm, sectors)["active_share"]


def test_turnover_cap_and_forced_sales():
    rm, bench, score, sectors, beta = _setup()
    lim = Limits(max_turnover=0.05)
    res = optimise(score, bench, rm, sectors, lim, beta, previous=bench)
    assert res.status == "optimal" and (res.weights - bench).abs().sum() / 2 <= 0.05 + 1e-5
    prev = pd.concat([bench * 0.9, pd.Series({"GONE": 0.1})])    # 10% sits in a stock that left the index
    res2 = optimise(score, bench, rm, sectors, lim, beta, previous=prev)
    assert res2.status == "optimal" and np.isclose(res2.weights.sum(), 1)


def test_benchmark_and_drift():
    cap = pd.Series({"A": 300.0, "B": 100.0, "C": np.nan, "D": 50.0})
    b = benchmark_weights(cap, ["A", "B", "C"])
    assert list(b.index) == ["A", "B"] and np.isclose(b["A"], 0.75)
    d = drift(pd.Series({"A": 0.5, "B": 0.5}), pd.Series({"A": 1.0}))   # A doubles, B has no return
    assert np.isclose(d["A"], 2 / 3)


def test_issuer_limit_covers_share_classes_together():
    rm, bench, score, sectors, beta = _setup()
    score = score.copy()
    score[["S00", "S01"]] = 10.0                                # two classes of one very attractive company
    issuer = pd.Series(TICKERS, index=TICKERS)
    issuer["S01"] = "S00"
    lim = Limits(tracking_error=0.05, max_stock=0.02, max_sector=0.2, max_beta=None, max_turnover=None)
    apart = optimise(score, bench, rm, sectors, lim, beta)
    together = optimise(score, bench, rm, sectors, lim, beta, issuer=issuer)
    extra = lambda w: (w - bench)[["S00", "S01"]].sum()
    assert extra(apart.weights) > 0.03 and extra(together.weights) <= 0.02 + 1e-5
    assert diagnostics(together.weights, bench, rm, sectors, issuer=issuer)["max_stock_active"] <= 0.02 + 1e-5
