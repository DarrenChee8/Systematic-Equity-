import numpy as np
import pandas as pd

from syseq.model.combine import combine, in_sample_weights, walk_forward_weights

IDX = pd.date_range("2015-01-31", periods=80, freq="ME")


def _ics(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"good": rng.normal(0.05, 0.05, 80), "flat": rng.normal(0.0, 0.05, 80),
                         "bad": rng.normal(-0.03, 0.05, 80)}, index=IDX)


def test_weights_are_valid_and_start_equal():
    for method in ("equal", "ic", "ic_ir", "mean_variance"):
        w = walk_forward_weights(_ics(), method, min_months=36)
        assert np.allclose(w.sum(axis=1), 1) and (w >= 0).all().all()
        assert np.allclose(w.iloc[:36], 1 / 3)             # not enough history yet
    w = walk_forward_weights(_ics(), "ic", min_months=36)
    assert w["good"].iloc[-1] > 0.6 and w["bad"].iloc[-1] == 0   # learns to favour the good theme


def test_weights_never_use_current_or_future_ics():
    ics = _ics()
    tampered = ics.copy()
    tampered.iloc[50:] = 9.0
    for method in ("ic", "ic_ir", "mean_variance"):
        a = walk_forward_weights(ics, method, 36)
        b = walk_forward_weights(tampered, method, 36)
        pd.testing.assert_frame_equal(a.iloc[:51], b.iloc[:51])   # month 50 uses ICs up to month 49 only
        assert not np.allclose(a.iloc[51], b.iloc[51])


def test_rolling_window_and_hindsight():
    ics = _ics()
    ics.iloc[:40, 0] = -0.1                                 # 'good' was bad early on
    recent = walk_forward_weights(ics, "ic", 36, window=24)
    full = walk_forward_weights(ics, "ic", 36)
    assert recent["good"].iloc[-1] > full["good"].iloc[-1]  # the short window forgets the bad start
    h = in_sample_weights(_ics())
    assert (h.nunique() == 1).all() and np.isclose(h.iloc[0].sum(), 1)


def test_combine_weights_and_missing_themes():
    cols = [f"S{i}" for i in range(30)]
    rng = np.random.default_rng(1)
    a = pd.DataFrame(rng.normal(size=(80, 30)), IDX, cols)
    b = pd.DataFrame(rng.normal(size=(80, 30)), IDX, cols)
    b.iloc[:, 0] = np.nan                                    # one stock lacks theme b
    sectors = pd.Series("X", index=cols)
    w = pd.DataFrame({"a": 1.0, "b": 0.0}, index=IDX)
    only_a = combine({"a": a, "b": b}, w, sectors)
    assert np.allclose(only_a.rank(axis=1), a.rank(axis=1))  # all weight on a reproduces a's ranking
    half = combine({"a": a, "b": b}, pd.DataFrame({"a": 0.5, "b": 0.5}, index=IDX), sectors)
    assert half.iloc[:, 0].notna().all()                     # falls back to the theme it does have
