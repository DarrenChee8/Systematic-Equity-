"""Cross-sectional transforms: each row (date) is treated independently."""
import pandas as pd


def winsorise(x: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    lo = x.quantile(lower, axis=1)
    hi = x.quantile(upper, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
