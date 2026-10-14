# 01 - Intraday Momentum and Core Signals

Why this folder: Report B identifies intraday momentum (price leaving its normal open-to-now range, confirmed by VWAP, on volatile days) as the only edge with peer-reviewed support on index futures. These are the primary sources behind the recommended "gated Noise Area breakout" candidate and the second uncorrelated entry (late-day momentum).

Every link below came up in a search on 2026-10-12 and points at a publisher, university, central bank or author-hosted page. Open each link and confirm before relying on it. Items marked [IN REPO] were in the T58 backtester's previous research folder (see `05_Repo_Research_Library_Index`).

## Core signal

1. Zarattini, Aziz, Barbon - "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)", Swiss Finance Institute Research Paper 24-97 (SSRN 4824172).
   - Role: the Noise Area system (19.6%/yr, 1.33 Sharpe net of costs, SPY 2007 to early 2024). This is the base of the candidate strategy. Read the band construction and cost model before coding.
   - Links: https://www.sfi.ch/de/publications/n-24-97-beat-the-market-an-effective-intraday-momentum-strategy-for-s-p500-etf-spy
     https://www.alexandria.unisg.ch/handle/20.500.14171/122124
     https://ideas.repec.org/p/chf/rpseri/rp2497.html
     https://concretumgroup.com/beat-the-market-an-effective-intraday-momentum-strategy-for-sp500-etf-spy/

2. Gao, Han, Li, Zhou - "Market Intraday Momentum", Journal of Financial Economics 2018 (SSRN 2440866).
   - Role: first-30-minute return predicts last-30-minute return; stronger on volatile, high-volume and news days. Basis for the volatility gate and the late-day momentum second entry.
   - Link: https://profiles.wustl.edu/en/publications/market-intraday-momentum/

3. Baltussen, Da, Lammers, Martens - "Hedging Demand and Market Intraday Momentum", Journal of Financial Economics 2021.
   - Role: finds the effect in 60+ futures contracts (1974-2020) and links it to dealer gamma hedging; momentum only on negative net-gamma days. Basis for the regime gate.
   - Links: https://pure.eur.nl/en/publications/hedging-demand-and-market-intraday-momentum/
     https://www3.nd.edu/~zda/intramom.pdf (appears to be an author-hosted copy; confirm the title on opening)

4. Zarattini, Aziz - "VWAP: The Holy Grail for Day Trading Systems" (SSRN 4631351). [IN REPO: 07_Volume_and_Other_Strategies]
   - Role: VWAP as side filter and trailing reference. Report B warns an independent rebuild decays to about 0.7 Sharpe out of sample and vanishes at realistic spreads.

5. Zarattini, Barbon, Aziz - "A Profitable Day Trading Strategy for the U.S. Equity Market" (SSRN 4729284). [IN REPO: 02_Momentum and Trend Following, file "A Profitable Day Trading Strategy.pdf"]

## Trend / time-series momentum (slower context and sizing ideas)

6. Moskowitz, Ooi, Pedersen - "Time Series Momentum", Journal of Financial Economics 2012.
   - Link: https://papers.ssrn.com/abstract=2089463

7. Barroso, Santa-Clara - "Momentum Has Its Moments", Journal of Financial Economics 2015. Volatility-scaling a momentum signal cuts crash risk and lifts Sharpe; relevant to sizing a trend-style entry.
   - Link: https://ciencia.ucp.pt/en/publications/momentum-has-its-moments/

8. Hurst, Ooi, Pedersen - "A Century of Evidence on Trend-Following Investing". [IN REPO: 02_Momentum and Trend Following]

9. Daniel, Moskowitz - "Momentum Crashes". [IN REPO: 02_Momentum and Trend Following]

## Not-yet-sourced items from Report B (find and add by hand)

- Fetna, "Opening-Range Breakout Does Not Survive Trading Costs", SSRN, Sep 2026 (the key negative result: 225 futures variants, median gross edge -0.01 ticks). Report B read it only through search excerpts; I did not locate it independently.
- Subhani, "Opening-Range Breakouts on Event-Free Fridays", SSRN, Jul 2026 (deflated Sharpe and permutation nulls).
- Heston, Korajczyk, Sadka - "Intraday Patterns in the Cross-Section of Stock Returns", Journal of Finance 2010.
- Lou, Polk, Skouras - "A Tug of War: Overnight Versus Intraday Expected Returns", JFE 2019.
- Cont, Kukanov, Stoikov - "The Price Impact of Order Book Events", J. Financial Econometrics 2014 (order-flow imbalance; Report B says skip for eval trading).

## How this feeds the backtester

Test the Noise Area signal at both 1-minute and 30-minute evaluation, with and without the VWAP side filter and the volatility gate, and count every variant for the deflated Sharpe step (see folder 02).
