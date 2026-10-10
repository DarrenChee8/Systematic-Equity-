# systematic-equity

A long-only, benchmark-relative, multi-factor US equity model, built stage by stage:
data -> signals -> model -> risk -> portfolio -> backtest -> paper trading.
The design follows the process published by large systematic managers.

## Status

| Milestone | Content | State |
|---|---|---|
| 1 | Data loader, point-in-time S&P 500 universe, 12-1 momentum, IC and quintile tests | done |
| 2 | Value, quality, low-risk signals from SEC EDGAR fundamentals; sector-neutral scores; signal correlations | done |
| 3 | Signal combination, walk-forward weights | next |
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
python scripts/run_milestone2.py
```

The first run downloads about 18 years of daily prices and caches them in `data/`.
Delete `data/*.parquet` to refresh.

Milestone 2 reads company fundamentals from SEC EDGAR, which requires a contact line.
Create a file named `.env` in the repo root (it is gitignored):

```
SEC_USER_AGENT="Your Name you@example.com"
```

## Results so far

**Milestone 1: 12-1 momentum, S&P 500, 2010-2026 (201 months).**
Mean monthly rank IC 0.006 (t = 0.42), hit rate 52%. Top-minus-bottom quintile return
1.6% a year with 16% volatility and a 41% maximum drawdown. On its own, momentum has no
statistically detectable edge in this universe and period, and it suffers sharp reversals
(2016-17, 2021-22).

![Momentum diagnostics](reports/m1_momentum.png)

## Known biases in the free data

- **Survivorship**: Yahoo Finance carries few delisted stocks, so companies that failed
  or were acquired are missing. The run prints index-member price coverage per month.
- **Membership**: taken from a community-maintained history file (github.com/fja05680/sp500), not an official S&P record.

- **Fundamentals**: SEC EDGAR only lists companies that still file, so most delisted
  companies have no fundamentals. Flow items use the latest annual report.
- **Sectors**: today's GICS sectors from Wikipedia are applied to the whole history;
  stocks that have left the index are grouped as "Unknown".

These will be measured by rerunning on point-in-time data (Sharadar).

## Method notes

- Signals at month-end *t* use data up to *t* only; returns are measured from *t* to *t+1*.
  `tests/test_core.py` checks this by tampering with future prices.
- A filing is usable from the day after its SEC filing date; restatements replace earlier
  values only from the date they were filed.
- Signals are winsorised and z-scored within sector, so stocks are ranked against peers.
- Momentum is the 12-month return skipping the latest month (Jegadeesh and Titman, 1993).
