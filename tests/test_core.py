import numpy as np
import pandas as pd

from syseq.data.universe import reconstruct_membership
from syseq.evaluation.ic import forward_returns, rank_ic
from syseq.evaluation.quantiles import quantile_returns
from syseq.signals.momentum import momentum
from syseq.signals.transforms import winsorise, zscore


def _prices(n_months=60, n_stocks=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    rets = rng.normal(0.01, 0.05, (n_months, n_stocks))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx,
                        columns=[f"S{i}" for i in range(n_stocks)])


def test_momentum_formula():
    p = _prices()
    sig = momentum(p)
    assert np.isclose(sig.iloc[20, 3], p.iloc[19, 3] / p.iloc[8, 3] - 1)
    assert sig.iloc[:12].isna().all().all()


def test_momentum_has_no_lookahead():
    """Changing future prices must not change today's signal."""
    p = _prices()
    tampered = p.copy()
    tampered.iloc[31:] *= 5
    pd.testing.assert_frame_equal(momentum(p).iloc[:31], momentum(tampered).iloc[:31])


def test_forward_return_alignment():
    p = _prices()
    fwd = forward_returns(p)
    assert np.isclose(fwd.iloc[10, 0], p.iloc[11, 0] / p.iloc[10, 0] - 1)
    assert fwd.iloc[-1].isna().all()


def test_ic_perfect_and_random():
    p = _prices()
    fwd = forward_returns(p)
    assert np.allclose(rank_ic(fwd, fwd), 1.0)          # perfect foresight
    noise = pd.DataFrame(np.random.default_rng(1).normal(size=p.shape), p.index, p.columns)
    assert abs(rank_ic(noise, fwd).mean()) < 0.05       # noise predicts nothing


def test_quantiles_monotonic_for_perfect_signal():
    p = _prices()
    fwd = forward_returns(p)
    q = quantile_returns(fwd, fwd).mean()
    assert q.is_monotonic_increasing and len(q) == 5


def test_transforms():
    p = _prices()
    z = zscore(winsorise(p))
    assert np.allclose(z.mean(axis=1), 0) and np.allclose(z.std(axis=1), 1)


def test_membership_reconstruction():
    changes = pd.DataFrame({"date": pd.to_datetime(["2020-06-15"]), "added": ["NEW"], "removed": ["OLD"]})
    me = pd.date_range("2020-04-30", "2020-08-31", freq="ME")
    m = reconstruct_membership(["A", "NEW"], changes, me)
    assert m.loc["2020-05-31", "OLD"] and not m.loc["2020-05-31", "NEW"]
    assert m.loc["2020-06-30", "NEW"] and not m.loc["2020-06-30", "OLD"]
    assert m["A"].all()


def test_membership_from_spans():
    from syseq.data.universe import membership_from_spans
    spans = pd.DataFrame({
        "ticker": ["A", "B", "B"],
        "start_date": pd.to_datetime(["2019-01-01", "2019-01-01", "2020-07-15"]),
        "end_date": pd.to_datetime([None, "2020-05-15", None]),
    })
    me = pd.date_range("2020-04-30", "2020-08-31", freq="ME")
    m = membership_from_spans(spans, me)
    assert m["A"].all()
    assert m["B"].tolist() == [True, False, False, True, True]


def test_neutral_returns_remove_market_and_sector():
    from syseq.evaluation.ic import neutral_returns
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    cols = [f"S{i}" for i in range(40)]
    beta = pd.DataFrame(np.tile(np.linspace(0.5, 1.5, 40), (24, 1)), idx, cols)
    mkt = pd.Series(rng.normal(0.01, 0.05, 24), idx)
    sect = pd.Series(["A"] * 20 + ["B"] * 20, index=cols)
    bump = pd.Series([0.02] * 20 + [-0.02] * 20, index=cols)
    noise = pd.DataFrame(rng.normal(0, 0.001, (24, 40)), idx, cols)
    fwd = beta.mul(mkt, axis=0) + bump + noise
    mask = pd.DataFrame(True, idx, cols)
    res = neutral_returns(fwd, beta, sect, mask)
    assert abs(res.corrwith(beta, axis=1).mean()) < 0.5 and res.abs().max().max() < 0.02
    assert np.allclose(res[cols[:20]].mean(axis=1), 0)
