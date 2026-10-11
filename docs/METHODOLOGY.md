# Methodology

What each stage does, why it is done that way, and how to read its output.

## 1. Universe

**What.** Every stock in the S&P 500 at each month-end, 2010 to date: about 500 at a time,
roughly 800 different companies over the period.

**Why.** A backtest must use the stocks an investor could have picked from at the time. Using
today's index members for the past would include only companies that survived and grew.
Membership comes from a history of joining and leaving dates.

**Why not a global index.** Large systematic managers run global universes (MSCI World, or
tens of thousands of stocks). Free data limits this project to the US: SEC filings are free and
dated, while equivalent global fundamentals and MSCI's constituent history are paid.

## 2. Point-in-time data

**What.** Every fundamental figure is used only from the day after it was filed with the SEC.

**Why.** A company's December year-end results are published in February. A model that "knows"
them in January is using the future (look-ahead bias) and will look better than it could
ever have been in practice.

**Trailing twelve months.** Earnings, cash flow, revenue and gross profit are the latest four
quarters: year-to-date + previous full year - the same year-to-date a year earlier.

## 3. Characteristics, signals and themes

Stocks are **not** sorted into categories. Every stock gets a score on every theme, every month.

1. **Characteristic**: a raw ratio, such as earnings / market value.
2. **Signal**: the characteristic compared with sector peers. Extreme values are capped
   (winsorised at the 1st and 99th percentile), then converted to a z-score within the sector.
   A bank is ranked against banks, because sectors have structurally different ratios.
3. **Theme**: the average of related signals, re-standardised.

| Theme | Signals | Idea |
|---|---|---|
| Value | Earnings, book, cash flow and sales, each / market value | Cheap relative to fundamentals |
| Quality | Return on equity, cash flow / assets, gross profit / assets, low accruals, low leverage | Profitable, cash-backed, sound balance sheet |
| Low risk | Low 12-month volatility, low beta | Steadier stocks |
| Momentum | 12-month return, skipping the latest month | Recent winners |
| Surprise | Year-on-year change in quarterly earnings and revenue, / the usual size of that change | Results better than their own history implies |

## 4. Testing a signal

Each test asks: did high-scoring stocks beat low-scoring stocks the following month?

| Measure | Meaning |
|---|---|
| Mean IC | Average monthly rank correlation between score and next-month return. 0 = no power; 0.02-0.05 is good |
| t-stat | Mean IC / its standard error. Above about 2 is distinguishable from luck |
| Hit rate | Share of months with a positive IC |
| Q5 minus Q1 | Annual return of the top fifth of stocks minus the bottom fifth |
| Sharpe | That return / its volatility |
| Max drawdown | Worst peak-to-trough loss of top minus bottom |
| Coverage | Share of index members that have a score |

**Raw and neutral returns.** Raw is the plain next-month return. Neutral first removes beta x
market return and then the sector average. Neutral is the fair test of stock selection: without
it, a low-risk signal looks terrible in a rising market simply because it holds low-beta stocks.

## 5. Combining themes

**What.** One forecast per stock, as a weighted average of the five theme scores.

**Weighting schemes.**
- *Equal*: 20% each.
- *IC*: in proportion to each theme's past average IC.
- *IC consistency*: in proportion to past IC / its volatility.
- *Mean-variance*: also allows for correlation between themes, with the covariance shrunk
  towards its diagonal because short histories estimate it poorly.

**Walk-forward.** Weights for a month are estimated from ICs observed before that month, and
re-estimated monthly. The first 36 months use equal weights.

**Hindsight benchmark.** Weights fitted on the whole period and applied to that same period.
No investor could have had them. The gap between hindsight and walk-forward results measures
how much a backtest flatters a model when its parameters are chosen with knowledge of the outcome.

## 6. Known limitations

- **Survivorship.** Free prices and filings omit most companies that were delisted. About 4%
  of index-member months are missing, more in the early years.
- **Membership and sectors.** Membership is from a community-maintained file. Sectors are
  today's classification applied to the past; former members without one are grouped as "Unknown".
- **Beta estimates.** Neutral returns depend on an estimated beta; estimation error tends to
  flatter low-beta stocks in rising markets. Betas are shrunk toward 1 to reduce this.
- **Multiple testing.** Many signals and schemes are tried. One of twenty crossing a t-stat of
  2 by chance is expected.
