# Passing a PropEval in 30 Days: The Math and What Quants Do

Oct 12, 2026 · Owen Mantz

(Text transcription of the PDF "Passing a PropEval in 30 Days: The Math and What Quants Do", 10 pages. Tables are flattened to rows.)

## Bottom line

Your spec is reachable on paper, and the binding constraint is not the $3,000 target or the drawdown: it is banking $1,500 in the first 5 trading days. In my simulation of a $50K eval ($3,000 target, $2,000 end-of-day trailing drawdown, 22 trading days), strategies with a 55-75% win rate and $250-$500 average winners pass 91-99% of the time at 2 trades a day with 0-2% bust. Only 38-52% of those also have $1,500 banked by day 5. At 3-4 trades a day that rises to 61-90%.

- Pass vs. bust is easy; speed is hard. Once the average winner is at least as big as the average loser, failure comes from running out of time, not from hitting the drawdown.
- High win rate with small winners is the trap. With $300 winners against $600 losers, a 70% win rate busts 46% of the time and only 85% wins pass reliably.
- Frequency is the lever. One trade a day passes only 51-72% of the time. You need roughly 3-4 trades a day of $250+ average winners, so "3 a week" is a floor, not the plan.
- The real risk is an overstated edge. At 1:1 payoffs, a true win rate 10 points below your backtest (65% instead of 75%) cuts pass from 99% to 80%. Payoff ratios of 1.5:1 or better make the plan far more forgiving.
- Most people fail. Disclosed pass rates cluster at 5-20% of attempts, so a backtested edge has to survive costs, slippage and regime change before these numbers mean anything.

## The ruleset I modeled

A typical $50K futures evaluation; the $3,000 target matches the common ~6% of account size used by most futures firms (Prop Firm Compared). Rules vary by firm and plan. Trailing drawdown follows your highest balance, so profit raises the floor until it locks. A consistency rule means one huge day can force you to keep trading after you hit the target.

| Rule modeled | What current firms show |
|---|---|
| Profit target $3,000 | About 6% of account size; one 50K plan lists $3,000 with a 2-day minimum (Phidias) |
| Max drawdown $2,000, trailing on end-of-day closes, locking at start + $100 | Industry has moved toward end-of-day trailing, more forgiving than intraday (DamnPropFirms); one 50K plan lists $2,000 EOD |
| Consistency: best day at most 50% of total profit, checked at pass | Topstep 50%; Apex none in evaluation, 50% funded; Alpha Futures varies by tier (GodLove University); other firms use 30-45% |
| Daily loss limit: none | Optional; some firms have none, some hard-breach |
| Time: 22 trading days (~30 calendar days) | Deadlines vary by firm; the 30-day window is your own constraint |
| Costs: none | Not modeled; see execution section |

## What your spec implies mathematically

W = average winner, L = average loser, p = win rate, b = W/L.

```
E = pW - (1-p)L                (expectancy per trade)
trades to target ~ 3000 / E
sigma = (W+L) * sqrt(p(1-p))
break-even win rate = L / (W+L)
```

Worked example: p = 70%, W = L = $300 gives E = $120 per trade and sigma ~ $275. The target takes about 25 trades on average, roughly 12 days at 2 trades a day. The simulation's median was 11 days.

Size check on index futures (standard CME point values): a $250-$300 winner is about 12-15 points on one NQ contract ($20/point) or 5-6 points on one ES contract ($50/point). One micro contract would need a 125-point move. So the spec implies one full-size contract with stops of roughly the same distance.

| Requirement | What it actually demands |
|---|---|
| $3,000 in under 30 days | $136 a day over 22 trading days, or about 12 net winners at $250 |
| At least 3 trades a week | Only a floor. 3 trades at $250 is $750 a week, so alone it cannot reach $3,000 in 4 weeks |
| $1,500 in the first 5 days, $250+ per trade | At least 6 net winning trades in week 1. At +/-$300 payoffs that means 8 wins out of 10 trades, or 10 out of 15 |
| High win rate | Break-even is 50% at 1:1, 40% at 1.5:1, 33% at 2:1, but 67% at 0.5:1. High win rate only helps if winners are not much smaller than losers |
| Pass before drawdown | Loss size must fit the buffer. A $300 loss uses 15% of a $2,000 buffer; a $500 loss uses 25%; a vendor guide suggests never risking more than 25% of remaining drawdown on one trade (HuntersAlgo) |
| Hold longer than 2 minutes | Rules out tick scalping. A design constraint, not a math one |

## Pass probability by win rate and payoff ratio

Each cell is pass % / bust % within 22 trading days, $300 average winner, 2 trades a day, rules above. The rest of each row is timeouts. b = winner size / loser size (b = 0.5 means $300 winners vs $600 losers).

| Win rate | b=0.5 | b=1 | b=1.5 | b=2 |
|---|---|---|---|---|
| 50% | 1 / 99 | 13 / 36 | 39 / 5 | 63 / 0 |
| 55% | 2 / 95 | 32 / 18 | 65 / 1 | 84 / 0 |
| 60% | 7 / 86 | 56 / 6 | 85 / 0 | 95 / 0 |
| 65% | 19 / 69 | 80 / 2 | 96 / 0 | 99 / 0 |
| 70% | 39 / 46 | 94 / 0 | 99 / 0 | 100 / 0 |
| 75% | 65 / 24 | 99 / 0 | 100 / 0 | 100 / 0 |
| 80% | 87 / 8 | 100 / 0 | 100 / 0 | 100 / 0 |
| 85% | 97 / 2 | 100 / 0 | 100 / 0 | 100 / 0 |

Three readings:

1. Below b = 1 the drawdown is the enemy. With $600 losers only three losses fit in a $2,000 buffer. You need roughly 80% wins before bust risk fades, and break-even is already 67%.
2. At b >= 1.5 the failure mode flips to running out of time. Bust is near zero even at 50-55% wins, so the question becomes whether you take enough trades.
3. Overstating your edge is expensive. At b = 1, a true 65% instead of a backtested 75% cuts pass from 99% to 80%, and 60% cuts it to 56%. At b = 1.5 the same drop goes 100% to 96% to 85%. A bigger payoff ratio buys robustness.

For comparison, LuxAlgo's open-source simulator reports about 50% pass for a 45% win rate, 1.4R winner trader risking 1% per trade, and about 95% at 48% wins, 1.6R and 0.75% risk, on a CFD 100K two-step challenge (LuxAlgo). Rules differ, but the pattern is the same: win rate, payoff and risk per trade have to be judged together.

## Candidate profiles against your full spec

Six profiles that all clear $250 average winners, tested at 2, 3 and 4 trades a day. Pass, bust and median days are at 2 trades a day. Last three columns = share of runs with $1,500+ banked after 5 trading days (2/day, 3/day, 4/day).

| Profile | Pass % | Bust % | Median days | $1.5K by day 5: 2/day, 3/day, 4/day |
|---|---|---|---|---|
| A: 70%, $300/$300 | 94 | 0 | 11 | 38, 72, 78 |
| B: 75%, $300/$300 | 99 | 0 | 9 | 52, 85, 90 |
| C: 60%, $375/$250 | 91 | 1 | 11 | 38, 61, 76 |
| D: 55%, $500/$250 | 96 | 2 | 9 | 50, 82, 87 |
| E: 50%, $500/$250 | 88 | 5 | 10 | 38, 70, 75 |
| F: 85%, $250/$500 | 96 | 1 | 9 | 54, 82, 83 |

- Pass odds reach 98-100% at 3-4 trades a day for every profile except E, which stays at 96%.
- Trades per day is the biggest lever. At 1 trade a day pass odds drop to 51-72% and the day-5 target is hit 0-26% of the time.
- Pass-to-bust is at least 18 to 1 for every profile. E is lowest at 88 to 5. F is the only profile that wins with small winners, and it needs a real 85% win rate.
- Intraday trailing drawdown costs 0-3 points. If your firm trails intraday instead of on closes, bust rises, e.g. 5% to 7% on E at 2/day and 1% to 3% on D at 3/day.
- Profit factor runs 2.0-3.0 across the profiles (A 2.3, B 3.0, C 2.3, D 2.4, E 2.0, F 2.8). Treat roughly 2.0 after costs as the screen.

## Adaptive risk: what it buys and what it costs

Adaptive sizing works as bust insurance and costs a few points of pass probability. Rule tested: trade size = cushion above the trailing floor divided by $2,000, limited to 0.35x at the low end and 1.25x at the high end. Size shrinks after losses and presses modestly when ahead. Also tested a stop for the day at -$600. All runs 2 trades/day.

| Profile | Fixed size pass/bust | Adaptive pass/bust | Adaptive + $600 day stop pass/bust |
|---|---|---|---|
| A: 70%, $300/$300 | 94 / 0 | 89 / 0 | 88 / 0 |
| B: 75%, $300/$300 | 99 / 0 | 98 / 0 | 97 / 0 |
| C: 60%, $375/$250 | 92 / 1 | 87 / 0 | 87 / 0 |
| D: 55%, $500/$250 | 96 / 2 | 94 / 0 | 93 / 0 |
| E: 50%, $500/$250 | 88 / 5 | 81 / 0 | 81 / 0 |
| F: 85%, $250/$500 | 96 / 1 | 92 / 0 | 93 / 0 |

- It removes the bust tail. Bust falls to 0% in every profile, biggest gain on low-win-rate profile E (5% to 0%).
- The cost is time, not losses. Smaller size after a drawdown slows the climb, so more runs time out. Pass drops 1-7 points. The share with $1,500 by day 5 barely changes, because early on the cushion is near full and size is near 1x.
- A $600 daily stop added almost nothing in this model (+/-1 point). It matters more when trades cluster and losses arrive in bursts, which this independent-trade model understates.
- Where to use it: low win rate or heavy payoff profiles (C, D, E) gain the most protection. For high win rate at 1:1 (A, B) fixed size is nearly as safe and faster.
- Practitioners describe the same idea: start with one micro contract, scale to two only after about $500 of profit, throttle exposure when volatility spikes (HuntersAlgo, FeneFX).

## HFT, hold time and execution reality

True HFT is off the table, and a 2-minute hold floor keeps you clear of the rules firms actually enforce. No prop firm offers colocation, direct market access or microsecond execution, and most prohibit HFT outright (Audacity Capital).

- Hold-time floors run from about 5 seconds to 2 minutes. Some firms flag a pattern of sub-30-second holds with tiny profit per trade (NexusFi). Code a minimum hold into the bot.
- Automation is allowed at most firms, not all. Across 128 firms tracked by one directory, 58 of 95 with a documented policy allow EAs or algos and 37 prohibit them (PropFirmMap). Permissions can differ between evaluation and funded account.
- News and slippage clauses can void a payout. One firm in the research prohibits Tier 1 news once funded.
- Your "HFT" is really 3-4 trades a day = 66-88 trades over 22 days. Costs scale with that count. As an assumption (not sourced), a $10-20 all-in round trip takes 4-8% off a $250 winner. At E = $120 a trade, $15 of costs cuts expectancy to $105 and adds about 3 trades to the target.
- Practical rule: model costs and one tick of slippage per side in your backtest before reading any pass probability here.

## What algo and quant traders are doing now

Most public material comes from vendor and affiliate blogs: common practice, not proof of results. No audited track record of a strategy meeting your exact spec was found.

Strategy families in use:

- Opening range breakout: fully mechanical; narrow-range sessions produce more fake-outs; results depend on range length (5, 15 or 30 min) (PickMyTrade).
- Mean reversion at session extremes with a hard time stop; never fade a trending session (For Traders).
- Trend pullback and breakout retest; wait for the retest on index futures and gold (For Traders).

Risk layer around the signal:

- Hard daily loss stop, trade caps, session filters, automatic flatten before close or news (HuntersAlgo).
- Risk 0.5-1% of account per trade in forex-style challenges; futures cap of about 25% of remaining drawdown per trade.
- Size that shrinks as the buffer shrinks, plus throttling when volatility or liquidity deteriorates (FeneFX).

Execution and validation:

- Signals coded in TradingView and sent by webhook to the broker, or native EAs on MT5/cTrader, often on a VPS.
- Monte Carlo against the firm's exact ruleset before paying. LuxAlgo publishes an open-source engine (prop-firm-sim) returning pass probability, which rule kills attempts, expected fees and a risk-per-trade sweep; it separates the risk that maximizes pass odds from the risk that maximizes expected value.

## The economics

- Disclosed pass rates: Topstep reports 16.8% of Combines started in 2025 were completed; Tradeify reports 17.2% of evaluations from Aug 2025 to Jul 2026 (Thor Trade Copier). Industry-wide figures cluster around 5-20%.
- One aggregator says only about 7% of traders who pass later withdraw money (QuantVPS). Secondhand; directional.
- Automation is spreading: at one multi-asset firm the share of active traders using an API or automated system rose from 34% to 76% in four months (Keyt). Firms respond with hold-time, frequency and consistency rules.
- Vendors advertise "10-15% monthly gains under 5% drawdown" for forex mean reversion (PickMyTrade). Unverified marketing claim.
- Read of the sources: optimize expected value after fees and payouts, not just probability of passing. A strategy tuned only to pass can fail on the funded account's tighter consistency and payout rules.

## Spec sheet: what the strategy has to show (after commissions and slippage)

| Metric | Target | Why |
|---|---|---|
| Payoff and win rate | Either 1:1 with at least 70% wins, or at least 1.5:1 with 55-60% wins or more | Both clear 91%+ pass at 2/day with 0-2% bust (profiles A, B, C, D) |
| Average winner | $250-$500 | Meets per-trade floor; $1,500 takes 6 net winners at $250 or 3 at $500 |
| Average loser | At most $300-$400, never above the winner unless win rate is 80%+ | $300 is 15% of the $2,000 buffer |
| Expectancy | At least $120 per trade after costs | About 25 trades to the target |
| Profit factor | At least 2.0 | All six tested profiles sit at 2.0-3.0 |
| Trades per day | 3-4 on average, never fewer than 2 | Takes day-5 odds from 38-52% to 61-90% |
| Hold time | 2-30 minutes, minimum enforced in code | Your own rule; avoids hold-time flags |
| Worst losing streak in test | At most 4 trades, worst day under about $700 | Keeps a bad run well inside the buffer |
| Best day / total profit | At most 40% | Room under 30-50% consistency caps |
| Sample size | At least 300 out-of-sample trades | With 300 trades and 65% win rate the 95% interval is about +/-5.4 points |
| Adaptive layer | Size scales with cushion, cap about 1.25x | Cuts bust to about 0% for a few points of pass |

Before paying for an evaluation:

1. Run the sim with the lower bound of your win-rate interval, not the point estimate. A 5-point haircut moves pass by 5-25 points at 1:1.
2. Stress costs: add one tick of slippage per side and your real commission.
3. Run it under both end-of-day and intraday trailing drawdown rules.
4. Paper trade two weeks live to confirm fills and hold times match the backtest.
5. Read the firm's rules for automation, news, hold time and consistency in both the evaluation and the funded phase.

## Caveats

- Independent trades: real trades cluster by regime, so losing streaks are fatter than modeled and bust numbers are optimistic.
- No costs, no slippage, no gaps. Fixed dollar outcomes per trade; fat-tailed losses not modeled.
- One ruleset: $50K, $2,000 trailing on closes, locking at start + $100, 50% consistency checked at pass, no daily loss limit. 20,000 paths per cell, seed 42, sampling error about +/-0.7 points.
- Pass is not payout. Funded-account rules, payout thresholds and the 5-20% disclosed pass rates all sit on top of these numbers.
- This is not financial advice.

## Sources (as listed in the PDF; figures come from search-result excerpts, as of Oct 12, 2026)

Futures Prop Firm Evaluation Rules Explained (GodLove University); Prop Firm Rules Explained, October 2026 (Prop Firm Compared); Best Futures Prop Firms 2026 (Phidias); Futures Prop Firm Rules (DamnPropFirms); Prop firm challenge pass rate (Thor Trade Copier); Prop Firm Statistics 2026 (QuantVPS); Prop Firm Trading Rules Explained (PropFirmMap); Automated Trading and EA Rules (NexusFi); 9 Best Prop Firms That Allow HFT (Audacity Capital); As algo trading grows, are prop firm rules keeping up? (Keyt); Prop Firm Evaluation Automation Guide (HuntersAlgo); Opening Range Breakout Automation (PickMyTrade); 5 Proven Strategies to Pass a Prop Firm Challenge (For Traders); Prop Trading Strategies (FeneFX); Prop Firm Sim (LuxAlgo).
