"""12-1 price momentum (Jegadeesh & Titman 1993).

Return over the past 12 months, skipping the most recent month (which tends to reverse).
"""
import pandas as pd


def monthly_prices(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("ME").last()


def momentum(monthly: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """Signal known at month-end t: price[t-skip] / price[t-lookback] - 1.

    Uses only prices up to t, so it can be traded at the close of month t.
    """
    return monthly.shift(skip) / monthly.shift(lookback) - 1
