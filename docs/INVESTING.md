# Investing

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
