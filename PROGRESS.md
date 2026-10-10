# Progress log

## Where things stand (end of 10 Oct 2026)

| Milestone | State |
|---|---|
| 1. Data loader, S&P 500 universe, momentum, IC and quintile tests | Run on real data, pushed |
| 2. Value, quality, low-risk signals from SEC fundamentals, neutral evaluation | Run on real data, pushed |
| 2b. Trailing-twelve-month fundamentals, surprise signals, extra labels | Code written and tested (20 tests). **Not yet run on this PC** |
| 3. Signal combination | Not started |

## Findings so far

- S&P 500, 2010-2026, market- and sector-neutral: no theme has a significant IC.
  Value -0.006 (t -0.59), quality 0.002 (0.24), low risk 0.016 (1.19), momentum 0.002 (0.13),
  equal-weight combination 0.009 (0.86).
- Value and momentum payoffs are strongly negatively correlated (-0.69).
- A trial of milestone 2b on the cached data (earnings-based parts only) gave the same
  conclusion: combination IC 0.009 (t 0.90).
- Price coverage of index members averages 96% (minimum 91%); the rest is survivorship bias.

## Next steps

1. Run `python scripts/run_milestone2b.py` (re-downloads fundamentals, 15-20 minutes).
2. Check the sanity table: Exxon should now show earnings; revenue columns should be filled.
   The old SEC numbers for Exxon and Booking in `EXTRA_CIKS` are unverified.
3. Commit and push milestone 2b.
4. Milestone 3 (combination), 4 (optimiser), 5 (backtest) on the current universe.
5. Then one month of Sharadar to rerun on a broad universe that includes delisted stocks.

## Open items

- Some tickers failed to download from Yahoo that should exist (for example BK, MMC, IPG).
- Repo name on GitHub still has a trailing hyphen (`Systematic-Equity-`).
- Sectors are today's classification; stocks that left the index are grouped as "Unknown".
