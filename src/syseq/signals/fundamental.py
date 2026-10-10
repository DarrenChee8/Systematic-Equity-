"""Value and quality characteristics from point-in-time fundamentals.

Inputs are month-end panels (index: dates, columns: tickers). Higher = more attractive.
"""
import numpy as np
import pandas as pd


def _ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def value_characteristics(net_income, equity, cfo, market_cap, revenue=None) -> dict[str, pd.DataFrame]:
    extra = {} if revenue is None else {"sales_yield": _ratio(revenue, market_cap)}
    return {
        **extra,
        "earnings_yield": _ratio(net_income, market_cap),
        "book_to_price": _ratio(equity, market_cap),
        "cashflow_yield": _ratio(cfo, market_cap),
    }


def quality_characteristics(net_income, equity, cfo, assets, gross_profit=None) -> dict[str, pd.DataFrame]:
    extra = {} if gross_profit is None else {"gross_profitability": _ratio(gross_profit, assets)}
    return {
        **extra,
        "roe": _ratio(net_income, equity),                 # profitability
        "cash_profitability": _ratio(cfo, assets),         # cash earnings per unit of assets
        "low_accruals": -_ratio(net_income - cfo, assets), # earnings backed by cash
        "low_leverage": _ratio(equity, assets),            # balance-sheet strength
    }
