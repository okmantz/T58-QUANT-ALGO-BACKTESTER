# Prop-firm lifecycle objective

The default optimisation target is the probability of completing the firm's evaluation **and** qualifying for a payout, not raw profit.
`app/prop/lifecycle.py` replays resampled trade paths through the real rules and reports separately:

- **Evaluation**: P(reach target before drawdown breach), P(pass), P(daily-loss violation), P(fail by drawdown / time limit), P(consistency satisfied | target hit),
  time to target, and P(pass within N calendar days) over *all* attempts.
- **Funded**: P(funded), Kaplan-Meier survival after N days (censored accounts are not counted as survivors), P(1st/2nd/3rd payout), expected payout.
- **Economics**: P(pass AND first payout), expected net per attempt after fees, expected attempts per payout.

**Why "8-30 days" and "180 days" disagreed**: the simulator reported *trading* days among *passing* attempts only; the Full Pipeline reported *calendar* days across
replay windows including attempts that never passed. The lifecycle report states each clock explicitly and gives the unconditional curve.

**Net-profit mode** (selectable on every form, `objective_mode=net_profit`) skips the prop simulation and runs `app/validation/net_profit_tester.py` for live
accounts: expectancy, profit factor, Monte Carlo P(profit), drawdown percentiles, cost stress and stability.

**Canonical engine**: `run_execution` is the reference. `app/backtest/reconcile.py` verifies the vectorized fast path reproduces it trade-for-trade.
