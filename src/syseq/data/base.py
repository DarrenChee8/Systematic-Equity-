"""Interface every data source implements, so free data can be swapped for Sharadar later."""
from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    @abstractmethod
    def prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        """Daily total-return (adjusted) prices. Index: dates, columns: tickers."""

    @abstractmethod
    def membership(self, start: str, end: str | None = None) -> pd.DataFrame:
        """Point-in-time index membership. Index: month-ends, columns: tickers, values: bool."""
