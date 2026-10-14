# Passing Prop Evals With Systematic Strategies: Evidence, Math and a Simple Spec

Oct 12, 2026 · Owen Mantz

(Text transcription of the PDF, 13 pages. Tables are flattened to rows.)

## Bottom line

No strategy found, in academic or practitioner sources, has an audited record of passing evals at 90% or better. The best-evidenced simple edge is intraday trend-following that fires when price leaves its normal opening-to-now range, and its published results shrink under independent replication and costs.

**What is real:** Intraday momentum is documented across assets and decades (Gao et al., JFE 2018; Baltussen et al., JFE 2021), and it is stronger on volatile days. The Noise Area system of Zarattini, Aziz and Barbon reports 19.6% a year and a 1.33 Sharpe ratio net of costs on SPY, 2007 to early 2024.

**What did not hold up:** Opening-range breakout earned about nothing before costs across 225 futures variants over 16 years (Fetna, Sep 2026). An independent rebuild of the VWAP "holy grail" decays to a Sharpe near 0.7 out of sample and loses its edge at realistic spreads. A separate QuantConnect build of the Noise Area system shows a 37% win rate and a 0.49 Sharpe (settings not verified against the paper).

**What the eval math says:** With a $15 round-trip cost, a 45% win rate and $540 winners against $300 losers (expectancy about $63 per trade net), pass is about 53% at 3 trades a day with adaptive sizing, bust about 9%. At a true 40% win rate pass falls to 26% and bust rises to 28%.

**Recommendation:** A volatility-gated Noise Area breakout with a VWAP trailing stop and adaptive size. Treat it as a candidate to falsify in your backtester, not a proven pass.

**What unlocks 90%+:** Roughly $120 or more of expectancy per trade after costs (per the targets in Report A). No source demonstrates that on a futures eval ruleset.

## 1. What the eval math demands

With costs in, a $50K eval passes 88% or more of the time only at about 50% wins with a 2:1 payoff, or 45% wins with 2.5:1 or better; at 40% wins it needs 3:1 to reach 87%.

Simulation rebuilt from Report A: $50K account, $3,000 target, $2,000 end-of-day trailing drawdown locking at start + $100, 22 trading days, 50% best-day consistency check at pass, independent trades. Runs reproduce Report A's profiles A, B and D within about a point (A: 93.5% pass, 0.8% bust vs Report A's 94 and 0). Table adds $15 round-trip cost, $300 average loss, 3 trades a day and the adaptive sizing rule (size = cushion above trailing floor / $2,000, clipped 0.35 to 1.25). Each cell is pass % / bust %, 10,000 paths per cell.

| Win rate | 1.5:1 payoff | 2:1 | 2.5:1 | 3:1 |
|---|---|---|---|---|
| 35% | 2.6 / 74.9 | 15.1 / 45.1 | 40.2 / 24.3 | 64.6 / 13.2 |
| 40% | 9.8 / 46.1 | 38.6 / 19.0 | 68.6 / 8.2 | 86.6 / 3.7 |
| 45% | 28.7 / 19.8 | 67.1 / 5.5 | 89.2 / 1.9 | 96.7 / 0.7 |
| 50% | 56.6 / 5.8 | 88.1 / 1.2 | 97.6 / 0.4 | 99.6 / 0.1 |
| 55% | 82.2 / 1.2 | 97.6 / 0.3 | 99.7 / 0.1 | 100.0 / 0.0 |

| Candidate profile ($300 loser, $15 cost, 3 trades/day) | Fixed size pass/bust % | Adaptive pass/bust % |
|---|---|---|
| 45% wins, $540 winners | 63.1 / 31.5 | 52.8 / 9.4 |
| 40% wins, $540 winners | 36.7 / 56.0 | 25.7 / 28.1 |
| 50% wins, $450 winners | 65.4 / 27.6 | 56.2 / 5.9 |
| 40% wins, $600 winners | 49.1 / 46.0 | 37.8 / 20.0 |

Three readings:

1. Costs are the hidden tax. At 35% wins and 1.5:1 the gross edge is negative; even at 40% and 2:1 the $15 cost removes a quarter of expectancy.
2. Trend-style win rates sit in the weak corner. 45% wins with $540 winners and $300 losers has a gross profit factor of 1.47. That passes 53% with adaptive sizing and 63% with fixed size, but fixed size busts 31% of the time.
3. An overstated win rate is the main risk. The same profile at a true 40% win rate has gross profit factor 1.20 and passes 26% (adaptive) with 28% bust; at 35% expectancy is negative and bust is 57%.

The simulation draws trades independently, so real losing streaks are fatter and bust numbers are optimistic. The $600 daily stop changed results by about a point in every run, so it is a guard against rule breaches rather than an edge.

## 2. Strategy families and what the evidence says

Only intraday momentum has peer-reviewed support on index futures; opening-range breakout and VWAP-cross systems lose most of their edge under costs, and options-positioning ideas work better as regime filters than as entry signals.

- **Opening-range breakout (ORB):** Fetna (SSRN, Sep 2026): median gross edge of -0.01 ticks across 225 futures variants; 155 sit within two ticks of zero; 5-minute range is worst. Subhani (SSRN, Jul 2026) tests an event-free Friday version with deflated Sharpe and permutation nulls. Strong published results are for single stocks filtered by abnormal volume (Zarattini et al., 2024). USE: skip as a standalone index-futures signal.
- **VWAP trend (long above, short below):** Zarattini and Aziz report $25,000 growing to $192,656 on QQQ with 9.4% max drawdown and 2.1 Sharpe. An independent rebuild finds the in-sample effect is real, decays toward 0.7 Sharpe out of sample, and vanishes at realistic spreads. USE: VWAP as a side filter and trailing reference, not as the signal.
- **Noise Area intraday momentum:** Zarattini, Aziz and Barbon (SFI 24-97): trades when price leaves the average open-to-now move, with trailing stops; 19.6% a year, 1.33 Sharpe net of costs, SPY 2007 to early 2024. A secondary summary reports Sharpe rising to 3.50 when VIX is above 40. One independent QuantConnect build shows 37% wins and 0.49 Sharpe. USE: best simple candidate; needs futures and cost validation.
- **First-half-hour to last-half-hour momentum:** Gao, Han, Li and Zhou (JFE 2018): return from prior close to the first 30 minutes predicts the last 30 minutes, more strongly on volatile, high-volume and major-news days. Baltussen, Da, Lammers and Martens (JFE 2021) find it in over 60 futures, 1974 to 2020, and tie it to gamma hedging. USE: one trade a day with a small edge; bias or exit-timing filter.
- **Mean reversion to VWAP bands:** widely published setup, but a ten-year DAX test reports weak results. Heston, Korajczyk and Sadka (JF 2010) tie short-term reversal in stocks to liquidity imbalances lasting under an hour. USE: high win rate with small winners is the trap Report A flags. Skip.
- **Pre-FOMC and overnight drift:** Lucca and Moench (NY Fed SR 512): the 24 hours before scheduled FOMC announcements carried over 80% of the equity premium across their 17-year sample. Lou, Polk and Skouras (JFE 2019): strategy profits split between overnight and intraday components, often with opposite signs. USE: few events a year, needs overnight holds; calendar filter only.
- **Dealer gamma, GEX walls, expected-move levels:** Baltussen et al. find intraday momentum only on negative net-gamma days. Adams et al. find lower intraday volatility and stronger reversals when 0DTE gamma is present. A Princeton thesis (Apr 2026) does not support hedging feedback as the main channel. Vendors say public open interest does not identify who holds the gamma, so GEX is estimated. USE: regime gate (trend vs reversion day); call/put walls as levels are unproven.
- **Volatility-managed sizing:** Moreira and Muir (JF 2017): less risk in high-volatility periods raises Sharpe ratios; DeMiguel et al. note other work finds out-of-sample and cost problems. USE: fits the eval's trailing floor; size by cushion.
- **Order-flow imbalance:** Cont, Kukanov and Stoikov (2014): short-horizon price changes are mainly driven by imbalance at best bid and ask. USE: needs book data and speed; prop firms offer no colocation. Skip.
- **Variance risk premium:** Bollerslev, Tauchen and Zhou (RFS 2009): implied minus realized variance predicts returns most strongly at the quarterly horizon. USE: horizon too long for a 30-day eval.
- **Classic technical rules:** Brock, Lakonishok and LeBaron (JF 1992) found support for MA and range-break rules; Sullivan, Timmermann and White (JF 1999) re-test for data snooping; a futures study finds technical rules generally unprofitable in US futures after that correction. USE: warning about parameter searching, not a signal.

Two cross-cutting facts: Bailey and Lopez de Prado show the best of many backtested variants overstates its Sharpe unless the number of trials is deflated, and their overfitting simulator shows optimized strategies commonly do poorly on fresh data. Disclosed pass rates cluster at 5-20% of attempts (Report A), while one operator blog puts first-attempt rates at 15-25% (Track360), so edge stability matters more than a headline Sharpe.

## 3. Recommended simple strategy: gated Noise Area breakout

Trade one index future in the direction of price once it leaves its normal open-to-now range and sits on the matching side of VWAP, only on volatile days, with size tied to the drawdown cushion. The Noise Area construction and VWAP trail come from the cited work; entry timing, initial stop, gates and risk rules are the author's own design choices, untested starting values.

**Instrument and session:** NQ or ES (MNQ or MES while testing), regular session only, flat by 15:55 ET, no overnight holds. On NQ ($20/pt) a $300 loser is a 15-point stop; on ES ($50/pt) it is 6 points.

**Signal (the Noise Area):** At each minute t compute the average absolute move from the day's open to time t over the last 14 sessions. Upper band = today's open + that average; lower band = open - the average. Follows the Concretum Bands description of the paper's construction; check the paper for its exact lookback and band adjustments.

**Entry:**

1. Long when a 1-minute close is above the upper band and above session VWAP; short when below the lower band and below VWAP.
2. Test both 1-minute and 30-minute evaluation; a public TradingView port of the paper runs on 30-minute bars and reports weaker results than the paper.
3. Skip the first 5 minutes after the open and any bar within 2 minutes of a scheduled release.
4. At most 2 entries a day, one per direction.

**Volatility gate (hypothesis to test):** Trade only when VIX is at least 18 or the day's realized range is above its 20-day median. Basis: intraday momentum is stronger on volatile days (Gao et al.). When a net-gamma estimate is available, add a second gate: skip long breakouts in positive-gamma conditions (Baltussen et al. find no significant intraday momentum there). FRED's VIXCLS series gives daily closes for the gate's history. The 18 threshold is a starting point, not a sourced figure.

**Exit:**

1. Initial stop at the Noise Area midpoint or a fixed $300 loss, whichever is closer.
2. Trail the stop to the nearer of VWAP and the band the trade broke out of, once price is one initial risk unit in profit.
3. Hard time exit at 15:55 ET; minimum hold of 2 minutes enforced in code.
4. No profit target. The edge comes from a few large winners; capping them removes the payoff ratio the math needs.

**Size and risk:** Risk $300 per trade at full size, then scale by cushion: size = (equity - trailing floor) / $2,000, clipped to 0.35-1.25 (Report A's adaptive rule; cut bust from 31.5% to 9.4% on the 45%/$540 profile for about ten points of pass). Daily loss stop at -$600 and a kill switch after 4 consecutive losses; neither adds edge but both protect against a rule breach. Stop trading for the eval once profit reaches $3,000 and the best day is under 40% of total profit; otherwise keep trading small until the consistency check clears.

**Honest pass-rate estimate:** If live edge matches the published shape (45% wins, $540 winners, $300 losers) expect about 53% pass and 9% bust. If closer to the independent replications, it fails. Two to three trades a day is likely more than this signal generates on its own, the main reason to test a second, uncorrelated entry before relying on the 53% figure.

What the strategy must show before you pay for an eval (after costs, out of sample):

| Metric | Minimum | Why |
|---|---|---|
| Average winner / average loser | At least 1.8, ideally 2.5 | Payoff drives pass odds |
| Win rate | At least 45% | At 40% pass falls to 26% at 1.8:1 |
| Expectancy per trade | At least $63; $120 for 90%+ pass odds | Net of $15 round trip |
| Trades per day | 2 to 3 | Day-5 odds in Report A rise from 38-52% at 2/day to 61-90% at 3-4 |
| Profit factor | At least 1.5 after costs | At 1.2 pass was 26% |
| Best day / total profit | At most 40% | Room under 30-50% consistency caps |
| Sample | At least 300 out-of-sample trades | Narrows the win-rate interval to about 5 points |

## 4. Other viable strategies, ranked

| Rank | Strategy | Role | What must be true | Evidence quality |
|---|---|---|---|---|
| 1 | Gated Noise Area breakout | Core signal | Payoff 1.8:1 or better at 45% wins after costs on futures | One working paper, one weaker independent replication |
| 2 | Late-day momentum: go with sign of first-half-hour and second-to-last-half-hour returns, hold into the close | Second trade a day, uncorrelated entry time | Per-trade edge large enough to clear $15 costs on NQ/ES | Peer-reviewed (Gao et al., Baltussen et al.); QuantRocket tutorial applies it on SPY with VIX-above-20 filter |
| 3 | Volatility and gamma regime gate | Turns core signal on/off | Threshold (VIX, realized range, net gamma) survives walk-forward testing | Peer-reviewed for momentum vs gamma; gamma estimates are vendor-model dependent |
| 4 | Mean reversion to VWAP bands with time stop, positive-gamma days only | Complement on quiet days | Payoff at least 1:1; below that need about 80% wins | Weak; vendor backtests only |
| 5 | Opening-range breakout, event-free days | Diversifier only | Positive after one tick of slippage per side | Fetna finds none on futures; Subhani reports low correlation with the index |
| 6 | Pre-FOMC calendar overlay | Do not fade strength into scheduled FOMC announcements | Eval rules allow the holding window | Lucca and Moench; sample ends before 2012, decay unknown |

Level ideas from options markets: the one-standard-deviation daily move = VIX / sqrt(252) x price; a TradingView expected-move script says price closes inside it about two days in three. Useful for sanity-checking a stop or target, but no peer-reviewed support for trading the edges themselves; call and put walls rest on estimated dealer positioning.

What not to build: volatility selling (needs options; futures evals generally do not offer them); overnight holds (depend on firm rules); order-flow strategies (need book data and latency); high-win-rate scalps with small winners (Report A shows 46% bust at 70% win rate with $300 winners vs $600 losers).

## 5. Validation protocol before paying for an eval

Run these seven checks in order; a failure at any step sends the strategy back to design.

1. Log every variant you test. Count them, compute a deflated Sharpe ratio against that count, and estimate the probability of backtest overfitting with combinatorially symmetric cross-validation.
2. Walk forward with parameters frozen. Fit on one window, test on the next, never refit on the test window. The independent VWAP rebuild shows what skipping this costs.
3. Charge real costs. At least $15 round trip on NQ/ES plus one tick of slippage per side, and actual commission. Fetna's finding: most breakout edge sits inside the spread.
4. Run a random-anchor control. Re-run the signal anchored to a random time instead of the session open. If the edge survives, it is volatility expansion rather than information in the open.
5. Split performance by regime: by VIX bucket, by event days, and if available by net-gamma sign. Be positive in the regimes you trade and know which you lose in.
6. Feed the trade list into the eval simulator. Resample real trade outcomes (not an assumed win rate) through the eval rules, under both end-of-day and intraday trailing drawdown, using the lower bound of your win-rate interval.
7. Paper trade two weeks to confirm fills, hold times and trade counts match the backtest, and read the firm's automation, news, hold-time and consistency rules for both eval and funded stage.

## Caveats and open questions

- Source depth: abstracts and summary pages for most academic items, not full papers, plus one full QuantConnect listing. Fetna and Subhani pages were read only through search excerpts. Check the paper's exact band construction and cost model before coding.
- Paper vs futures: Noise Area and VWAP results are for SPY and QQQ shares. Futures costs, the 22-day clock and trailing floor are different; independent replications are practitioner builds, not peer review.
- Secondary figures: the 3.50 Sharpe above VIX 40 comes from a Substack summary; the 70% share of days that implied vol exceeds realized comes from a glossary page.
- Simulation limits: independent trades, flat $15 costs, no slippage tail or gap risk. Real streaks cluster; bust figures are optimistic.
- Regime decay: published intraday edges tend to shrink after publication.
- Open question: which firm and plan will you test on? Consistency, hold-time and automation rules vary, and the 40% best-day cap assumes a 50% rule.

This is research, not financial advice.

## Sources (as listed in the PDF, as of Oct 12, 2026)

**Academic and central-bank research:** Gao, Han, Li, Zhou, Market intraday momentum (JFE 2018; SSRN 2440866). Baltussen, Da, Lammers, Martens, Hedging demand and market intraday momentum (JFE 2021). Zarattini, Aziz, Barbon, Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY) (SFI 24-97; SSRN 4824172). Zarattini, Aziz, VWAP: The Holy Grail for Day Trading Systems (SSRN 4631351). Zarattini, Barbon, Aziz, A Profitable Day Trading Strategy for the U.S. Equity Market (SSRN 4729284). Fetna, Opening-Range Breakout Does Not Survive Trading Costs (SSRN, Sep 2026). Subhani, Opening-Range Breakouts on Event-Free Fridays (SSRN, Jul 2026). Lucca, Moench, The Pre-FOMC Announcement Drift (NY Fed SR 512; JF 2015). Lou, Polk, Skouras, A tug of war: overnight versus intraday expected returns (JFE 2019). Heston, Korajczyk, Sadka, Intraday patterns in the cross-section of stock returns (JF 2010). Moreira, Muir, Volatility-managed portfolios (NBER 22208; JF 2017). DeMiguel, Martin-Utrera, Uppal, A multifactor perspective on volatility-managed portfolios (JF). Bollerslev, Tauchen, Zhou, Expected stock returns and variance risk premia (RFS 2009). Cont, Kukanov, Stoikov, The price impact of order book events (J. Financial Econometrics 2014). Adams et al., Do S&P500 0DTE options increase market volatility? Princeton ORFE senior thesis, 0DTE Options and Intraday Volatility (Apr 2026). Bailey, Lopez de Prado, The Deflated Sharpe Ratio (SSRN 2460551). Bailey, Borwein, Lopez de Prado, Zhu, The Probability of Backtest Overfitting (SSRN 2326253); Bailey et al., Statistical Overfitting and Backtest Performance (SSRN 2507040). Brock, Lakonishok, LeBaron (JF 1992); Sullivan, Timmermann, White (JF 1999).

**Data:** Federal Reserve Bank of St. Louis, CBOE Volatility Index: VIX (VIXCLS), daily close.

**Replications and practitioner pages:** QuantConnect (independent VWAP "Holy Grail" reproduction and Noise Area SPY build); QuantRocket (First half hour predicts last half hour); Concretum Bands and a TradingView port of the Noise Area strategy; Quantified Strategies; SpotGamma and Quantower (gamma exposure); Backtrex (VWAP band fade on the DAX); Volatility Box; Track360; Keyt; NexusFi; Report A.
