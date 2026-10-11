import numpy as np
import pandas as pd

from syseq.data.edgar import asof_panel, extract_facts, split_adjust
from syseq.evaluation.correlation import signal_correlation
from syseq.signals.transforms import composite, sector_zscore

ME = pd.date_range("2020-01-31", "2021-12-31", freq="ME")


def _facts(rows):
    df = pd.DataFrame(rows, columns=["ticker", "tag", "start", "end", "val", "filed", "form"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def test_asof_uses_only_filed_data():
    f = _facts([
        ("A", "Assets", None, "2019-12-31", 100.0, "2020-02-20", "10-K"),
        ("A", "Assets", None, "2020-03-31", 110.0, "2020-05-05", "10-Q"),
        ("A", "Assets", None, "2019-12-31", 999.0, "2020-05-05", "10-Q"),  # comparative, older period
    ])
    p = asof_panel(f, ["Assets"], ME)["A"]
    assert np.isnan(p["2020-01-31"])                 # nothing filed yet
    assert p["2020-02-29"] == 100 and p["2020-04-30"] == 100   # Q1 exists but is not yet filed
    assert p["2020-05-31"] == 110                    # newest period wins over the comparative


def test_asof_filing_lag_and_staleness():
    f = _facts([("A", "Assets", None, "2019-12-31", 100.0, "2020-02-29", "10-K")])
    p = asof_panel(f, ["Assets"], ME)["A"]
    assert np.isnan(p["2020-02-29"]) and p["2020-03-31"] == 100   # usable the day after filing
    assert p["2021-06-30"] == 100 and np.isnan(p["2021-07-31"])   # dropped once over 550 days old


def test_asof_annual_filter_restatement_and_tag_priority():
    f = _facts([
        ("A", "NetIncomeLoss", "2019-01-01", "2019-12-31", 40.0, "2020-02-20", "10-K"),
        ("A", "ProfitLoss", "2019-01-01", "2019-12-31", 44.0, "2020-02-20", "10-K"),
        ("A", "NetIncomeLoss", "2020-01-01", "2020-03-31", 9.0, "2020-05-05", "10-Q"),   # quarter: ignored
        ("A", "NetIncomeLoss", "2019-01-01", "2019-12-31", 38.0, "2020-08-10", "10-K/A"),  # restated
    ])
    p = asof_panel(f, ["NetIncomeLoss", "ProfitLoss"], ME, "annual")["A"]
    assert p["2020-03-31"] == 40 and p["2020-06-30"] == 40 and p["2020-08-31"] == 38


def test_split_adjust():
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    splits = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    splits.loc["2020-08-31", "A"] = 4.0
    f = _facts([("A", "s", None, "2020-06-30", 100.0, "2020-07-30", "10-Q"),
                ("A", "s", None, "2020-09-30", 400.0, "2020-10-30", "10-Q"),
                ("B", "s", None, "2020-06-30", 50.0, "2020-07-30", "10-Q")])
    assert split_adjust(f, splits)["val"].tolist() == [400.0, 400.0, 50.0]


def test_extract_facts():
    js = {"facts": {
        "us-gaap": {"Assets": {"units": {"USD": [
            {"end": "2020-12-31", "val": 5, "filed": "2021-02-01", "form": "10-K"},
            {"end": "2020-12-31", "val": 5, "filed": "2021-02-01", "form": "8-K"}]}},
            "Goodwill": {"units": {"USD": [{"end": "2020-12-31", "val": 1, "filed": "2021-02-01", "form": "10-K"}]}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2021-01-25", "val": 10, "filed": "2021-02-01", "form": "10-K"}]}}}}}
    df = extract_facts(js, "A")
    assert sorted(df["tag"]) == ["Assets", "EntityCommonStockSharesOutstanding"]
    assert df["start"].isna().all() and str(df["filed"].dtype).startswith("datetime")


def test_sector_zscore_and_composite():
    rng = np.random.default_rng(0)
    cols = [f"S{i}" for i in range(40)]
    x = pd.DataFrame(rng.normal(size=(6, 40)), index=ME[:6], columns=cols)
    x.iloc[:, :20] += 10                                  # one sector has much higher values
    sectors = pd.Series(["Tech"] * 20 + ["Banks"] * 17, index=cols[:37])   # 3 tickers unknown
    z = sector_zscore(x, sectors)
    assert np.allclose(z.iloc[:, :20].mean(axis=1), 0)    # the sector effect is removed
    assert z[cols[37:]].isna().all().all()                # group of 3 is below min_names
    c = composite([z, z], sectors)
    assert np.allclose(c.iloc[:, :20].std(axis=1), 1)


def test_signal_correlation():
    rng = np.random.default_rng(1)
    a = pd.DataFrame(rng.normal(size=(12, 50)))
    c = signal_correlation({"a": a, "neg": -a, "noise": pd.DataFrame(rng.normal(size=(12, 50)))})
    assert np.isclose(c.loc["a", "neg"], -1) and abs(c.loc["a", "noise"]) < 0.15


def _q(t, s, e, v, filed, tag="NetIncomeLoss"):
    return (t, tag, s, e, v, filed, "10-Q")


def _year(y, q1, h1, m9, fy, t="A"):
    """A calendar-year filer: year-to-date figures filed about 35 days after each quarter."""
    return [_q(t, f"{y}-01-01", f"{y}-03-31", q1, f"{y}-05-05"),
            _q(t, f"{y}-01-01", f"{y}-06-30", h1, f"{y}-08-05"),
            _q(t, f"{y}-01-01", f"{y}-09-30", m9, f"{y}-11-05"),
            _q(t, f"{y}-01-01", f"{y}-12-31", fy, f"{y + 1}-02-20")]


def test_ttm_and_quarterly():
    from syseq.data.edgar import quarterly_facts, ttm_facts
    f = _facts(_year(2019, 10, 22, 36, 50) + _year(2020, 13, 28, 45, 64))
    t = ttm_facts(f, ["NetIncomeLoss"], "ni").set_index("end")
    assert t.loc["2019-12-31", "val"] == 50
    assert t.loc["2020-03-31", "val"] == 13 + 50 - 10          # YTD + prior year - prior YTD
    assert t.loc["2020-09-30", "val"] == 45 + 50 - 36
    assert t.loc["2020-03-31", "filed"] == pd.Timestamp("2020-05-05")
    assert pd.Timestamp("2019-03-31") not in t.index            # no prior year to build from
    q = quarterly_facts(f, ["NetIncomeLoss"], "ni").set_index("end")["val"]
    assert q["2019-03-31"] == 10 and q["2019-06-30"] == 12 and q["2019-12-31"] == 14
    assert q["2020-12-31"] == 19 and len(q) == 8


def test_ttm_panel_is_point_in_time():
    from syseq.data.edgar import ttm_facts
    f = _facts(_year(2019, 10, 22, 36, 50) + _year(2020, 13, 28, 45, 64))
    p = asof_panel(ttm_facts(f, ["NetIncomeLoss"], "ni"), ["ni"], ME)["A"]
    assert p["2020-04-30"] == 50 and p["2020-05-31"] == 53      # Q1 TTM only after the 10-Q is filed
    assert p["2021-01-31"] == 59 and p["2021-02-28"] == 64


def test_surprise():
    from syseq.data.edgar import quarterly_facts, surprise_facts
    rows = []
    for i, y in enumerate(range(2014, 2021)):
        b = 10 + i + (i % 2)                                    # uneven growth so the scale is non-zero
        rows += _year(y, b, 2 * b, 3 * b, 4 * b)
    rows[-1] = _q("A", "2020-01-01", "2020-12-31", 4 * b + 20, "2021-02-20")   # big Q4 beat
    s = surprise_facts(quarterly_facts(_facts(rows), ["NetIncomeLoss"], "ni"), "sue").set_index("end")["val"]
    assert s["2020-12-31"] > 3 and abs(s["2020-09-30"]) < 3
    assert s.index.min() >= pd.Timestamp("2016-01-01")          # needs history before it can score


def test_bad_dates_are_ignored():
    f = _facts([("A", "Assets", None, "2019-12-31", 100.0, "2020-02-20", "10-K"),
                ("A", "Assets", None, "2029-12-31", 5.0, "2020-03-20", "10-Q"),   # impossible period end
                ("A", "Assets", None, "2020-03-31", 110.0, "2020-05-05", "10-Q")])
    assert asof_panel(f, ["Assets"], ME)["A"]["2020-05-31"] == 110


def test_clean_share_counts_drops_unit_errors_but_keeps_real_changes():
    from syseq.data.edgar import clean_share_counts
    ends = pd.date_range("2015-03-31", periods=16, freq="QE")
    vals = [100.0 + i for i in range(16)]
    vals[7] = 100_000_000.0                                  # a count a million times too large
    vals[11] = 0.11                                          # a count filed in thousands
    rows = [("A", "s", None, e, v, e + pd.Timedelta(days=30), "10-Q") for e, v in zip(ends, vals)]
    rows += [("B", "s", None, "2019-12-31", 50.0, "2020-02-01", "10-K"),
             ("B", "s", None, "2020-03-31", 50.0, "2020-05-01", "10-Q"),
             ("B", "s", None, "2020-06-30", 95.0, "2020-08-01", "10-Q"),    # a merger nearly doubles it: real
             ("B", "s", None, "2020-09-30", 96.0, "2020-11-01", "10-Q")]
    out = clean_share_counts(_facts(rows))
    a = out[out["ticker"] == "A"]["val"]
    assert len(a) == 14 and a.max() < 200 and a.min() > 50
    assert len(out[out["ticker"] == "B"]) == 4


def test_market_cap_repairs_units_and_share_classes():
    from syseq.data.edgar import market_cap_panel
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    shares = pd.DataFrame({"OK": [1e9, 1e9], "MILLIONS": [700.0, 700.0], "THOUSANDS": [5e5, 5e5],
                           "CLS_A": [3e9, 3e9], "CLS_B": [3e9, 3e9], "NONE": [np.nan, np.nan]}, index=idx)
    close = pd.DataFrame(100.0, index=idx, columns=shares.columns)
    cap = market_cap_panel(shares, close).iloc[0]
    assert cap["OK"] == 1e11
    assert cap["MILLIONS"] == 7e10 and cap["THOUSANDS"] == 5e10      # scaled up to plausible values
    assert cap["CLS_A"] == cap["CLS_B"] == 1.5e11                    # one company, not counted twice
    assert np.isnan(cap["NONE"])
