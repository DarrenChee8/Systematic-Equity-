"""Low-risk characteristics: low volatility and low beta. Higher = lower risk."""
import pandas as pd


def daily_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    return daily_prices.pct_change(fill_method=None)


def low_volatility(daily_prices: pd.DataFrame, window: int = 252, min_periods: int = 200) -> pd.DataFrame:
    vol = daily_returns(daily_prices).rolling(window, min_periods=min_periods).std()
    return -vol.resample("ME").last()


def low_beta(daily_prices: pd.DataFrame, window: int = 252, min_periods: int = 200) -> pd.DataFrame:
    """Beta to the equal-weighted average of all stocks in the price file."""
    r = daily_returns(daily_prices)
    mkt = r.mean(axis=1)
    cov = r.rolling(window, min_periods=min_periods).cov(mkt)
    var = mkt.rolling(window, min_periods=min_periods).var()
    return -cov.div(var, axis=0).resample("ME").last()
