# systematic-equity

A long-only, benchmark-relative, multi-factor US equity model, built stage by stage:
data -> signals -> model -> risk -> portfolio -> backtest -> paper trading.
The design follows the process published by large systematic managers.

## Status

| Milestone | Content | State |
|---|---|---|
| 1 | Data loader, point-in-time S&P 500 universe, 12-1 momentum, IC and quintile tests | done |
| 2 | Value, quality, low-risk signals; sector-neutral scores; signal correlations | next |
| 3 | Signal combination, walk-forward weights | |
| 4 | Risk model and constrained optimiser | |
| 5 | Backtest with costs and attribution | |
| 6 | IBKR paper trading | |

## Run it

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_milestone1.py
```

The first run downloads about 18 years of daily prices and caches them in `data/`.
Delete `data/*.parquet` to refresh.

## Known biases in the free data

- **Survivorship**: Yahoo Finance carries few delisted stocks, so companies that failed
  or were acquired are missing. The run prints index-member price coverage per month.
- **Membership**: taken from a community-maintained history file (github.com/fja05680/sp500), not an official S&P record.

Both will be measured by rerunning on point-in-time data (Sharadar).

## Method notes

- Signals at month-end *t* use data up to *t* only; returns are measured from *t* to *t+1*.
  `tests/test_core.py` checks this by tampering with future prices.
- Momentum is the 12-month return skipping the latest month (Jegadeesh and Titman, 1993).
