# Guides

Short guides that used to be separate files.

## Installing on a Chromebook

1. Turn on Linux: **Settings > About ChromeOS > Linux development environment > Turn on**.
2. Download the **Chromebook** zip (most Chromebooks) or **Chromebook-ARM** (ARM models, e.g. MediaTek/Snapdragon). Not sure? In the Linux terminal run `uname -m`: `x86_64` = Chromebook, `aarch64` = Chromebook-ARM.
3. Move the zip into **My files > Linux files**, then in the Linux terminal:
   ```
   unzip T58-*-Chromebook*.zip -d t58 && cd t58 && bash install_chromebook.sh
   ```
4. The installer adds a **T58** launcher to the ChromeOS app drawer and opens the app in your browser.

Needs Debian 12 (the default Linux container). If the app does not start, run `./T58-Web-App` from the terminal to see the error.

## Investing (paper mode by default)

Hedge Fund Manager > **Investing**.

- **Discover**: search stocks (via the SEC ticker list), indices (S&P 500, Nasdaq, Dow...), ETFs and crypto. Each stock page shows price, 52-week range, annual
  revenue, net income, assets, liabilities and equity from SEC filings, recent 10-K/10-Q/8-K links, and a **T58 value score** (0-100) with the reasons behind it.
- **Undervalued scan**: ranks a large-cap universe by earnings yield, price/book, price/free-cash-flow, return on equity, margins, growth, balance sheet and
  consistency, penalising stocks that are cheap but collapsing (value traps).
- **My portfolio / Trade**: manual market orders (dollars or shares). **Paper** (default, $10,000 of practice money, stored locally), **Alpaca paper**, or **Live**
  (real money; needs Alpaca keys saved in Settings and a confirmation tick on every session).
- **Automated**: six questions (reaction to a 20% drop, horizon, goal, hands-on level, allowed assets, max tolerable loss) set the risk score, number of
  holdings, per-name cap, cash floor, value/momentum/quality mix, rebalance frequency, crypto cap and stop-loss. Algos: momentum, mean reversion, low-vol trend,
  value, and the T58 composite. **Backtest** any price-based algo over up to 10 years against an equal-weight benchmark, then apply it to the account.
- **SEC setup**: the SEC requires every client to identify itself. Enter "Your Name you@example.com" (or set `T58_SEC_USER_AGENT`). It is sent only to sec.gov.

Honest limits: fundamentals are not back-tested (today's filings would leak the future), so historical tests use price factors only; screens rank candidates and
do not predict returns; free price data (Yahoo/Stooq) can lag or fail, and the tab reports the error instead of guessing. Not investment advice.

## Prop-firm lifecycle

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
