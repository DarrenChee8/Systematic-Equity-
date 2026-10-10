"""Free data: Yahoo Finance prices and a public S&P 500 membership history.

Known biases (quantified later against Sharadar):
- Yahoo has no prices for most delisted companies, so failed or acquired stocks
  drop out of the sample -> survivorship bias, results flattered.
- The membership history is a community-maintained file, not an official S&P record.
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .base import DataSource
from .universe import membership_from_spans

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Membership spans (ticker, start_date, end_date) maintained at github.com/fja05680/sp500
MEMBERS_CSV = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
HEADERS = {"User-Agent": "syseq-research/0.1 (personal research project)"}


class FreeData(DataSource):
    def __init__(self, cache_dir: str = "data"):
        self.cache = Path(cache_dir)
        self.cache.mkdir(exist_ok=True)

    # ---- universe -------------------------------------------------------
    def sectors(self) -> pd.Series:
        """Current GICS sector per ticker, from Wikipedia. Stocks no longer in the index have none."""
        path = self.cache / "sectors.csv"
        if path.exists():
            return pd.read_csv(path, index_col=0).iloc[:, 0]
        html = requests.get(WIKI, headers=HEADERS, timeout=30).text
        tables = pd.read_html(StringIO(html))
        current = next(t for t in tables if "Symbol" in t.columns)
        out = current.set_index("Symbol")["GICS Sector"].rename("sector")
        out.to_csv(path)
        return out

    def membership(self, start: str, end: str | None = None) -> pd.DataFrame:
        path = self.cache / "membership.parquet"
        if path.exists():
            return pd.read_parquet(path)
        spans = pd.read_csv(MEMBERS_CSV, parse_dates=["start_date", "end_date"])
        end = end or pd.Timestamp.today().normalize()
        month_ends = pd.date_range(start, end, freq="ME")
        out = membership_from_spans(spans, month_ends)
        out.to_parquet(path)
        return out

    # ---- prices ---------------------------------------------------------
    def prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        import yfinance as yf

        path = self.cache / "prices.parquet"
        if path.exists():
            return pd.read_parquet(path)
        yahoo = {t: t.replace(".", "-") for t in tickers}  # BRK.B -> BRK-B
        raw = yf.download(list(yahoo.values()), start=start, end=end,
                          auto_adjust=True, progress=True, threads=True)["Close"]
        raw = raw.rename(columns={v: k for k, v in yahoo.items()}).dropna(axis=1, how="all")
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.to_parquet(path)
        return raw

    def market_data(self, tickers: list[str], start: str, end: str | None = None):
        """Split-adjusted (not dividend-adjusted) close and split ratios, for market cap."""
        import time

        import yfinance as yf

        cpath, spath = self.cache / "close.parquet", self.cache / "splits.parquet"
        if cpath.exists() and spath.exists():
            return pd.read_parquet(cpath), pd.read_parquet(spath)
        yahoo = {t: t.replace(".", "-") for t in tickers}
        back = {v: k for k, v in yahoo.items()}

        def fetch(symbols):
            raw = yf.download(symbols, start=start, end=end, auto_adjust=False,
                              actions=True, progress=True, threads=True)
            fields = raw.columns.get_level_values(0)
            close = raw["Close"]
            splits = raw["Stock Splits"] if "Stock Splits" in fields else close * 0
            return close, splits

        close, splits = fetch(list(yahoo.values()))
        missing = [c for c in close.columns if close[c].isna().all()]
        if missing:  # one retry, in case Yahoo throttled part of the first request
            time.sleep(5)
            c2, s2 = fetch(missing)
            close[c2.columns] = c2.reindex(close.index)
            splits[s2.columns] = s2.reindex(splits.index)
        out = []
        for df in (close, splits):
            df = df.rename(columns=back)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            out.append(df)
        close, splits = out
        keep = close.columns[close.notna().any()]
        close, splits = close[keep], splits[keep].fillna(0.0)
        close.to_parquet(cpath)
        splits.to_parquet(spath)
        return close, splits
