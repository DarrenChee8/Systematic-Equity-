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
