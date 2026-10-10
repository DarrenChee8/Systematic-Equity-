"""Value and quality characteristics from point-in-time fundamentals.

Inputs are month-end panels (index: dates, columns: tickers). Higher = more attractive.
"""
import numpy as np
import pandas as pd


def _ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def value_characteristics(net_income, equity, cfo, market_cap) -> dict[str, pd.DataFrame]:
    return {
        "earnings_yield": _ratio(net_income, market_cap),
        "book_to_price": _ratio(equity, market_cap),
        "cashflow_yield": _ratio(cfo, market_cap),
    }


def quality_characteristics(net_income, equity, cfo, assets) -> dict[str, pd.DataFrame]:
    return {
        "roe": _ratio(net_income, equity),                 # profitability
        "cash_profitability": _ratio(cfo, assets),         # cash earnings per unit of assets
        "low_accruals": -_ratio(net_income - cfo, assets), # earnings backed by cash
        "low_leverage": _ratio(equity, assets),            # balance-sheet strength
    }
