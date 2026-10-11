# Progress log

## Where things stand (11 Oct 2026)

| Milestone | State |
|---|---|
| 1. Data, universe, momentum, signal tests | Done |
| 2, 2b. Five themes from SEC fundamentals, neutral testing | Done |
| 3. Combination model, methodology document | Done |
| 4. Risk model, optimiser, monthly holdings | Done |
| 5. Backtest, costs, attribution | Done |
| 6. IBKR paper trading | Not started |
| 7. Point-in-time data (Sharadar), wider universe, original extension | Not started |

## Findings so far

- S&P 500, 2010-2026: no theme has a statistically significant IC; equal-weight combination
  IC about 0.01 (t about 0.9 to 1.2). Value and momentum payoffs correlate at about -0.65.
- Data-driven theme weights do not beat equal weights out of sample; hindsight weights look best.
- Backtest 2011-2026: excess return 0.29% a year gross, 0.15% net of 5bp costs, information
  ratio under 0.1. Realised tracking error 3.7% against 2.5% predicted. Turnover 121% a year.

## Open items

- Risk model under-predicts tracking error by about half: consider a longer window, shrinkage,
  or scaling the target.
- SEC share counts needed three rule-based repairs (median filter, units, share classes).
- Some tickers fail to download from Yahoo that should exist (for example BK, MMC, IPG).
- Sectors are today's classification; former index members are grouped as "Unknown".
- Benchmark covers about 60% of the index in 2011, nearly all by 2026.
