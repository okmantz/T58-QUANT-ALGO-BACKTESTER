# Index of the Research Folder (previous contents)

Snapshot of 2026-10-12 of the research folder as it stood before it was replaced by these documents: 44 PDF files (43 distinct papers; the Kelly paper was stored twice) in 7 subfolders, plus README.md, EVOLUTION_LAB_ROADMAP.md (now docs/) and audit_2026_10/ scripts (now scripts/audit_2026_10/) (combo.py, fast.py, lab.py, oos.py, scan2.py, scan3.py).

The PDFs listed here have since been removed from the repository; this index is kept so the reading lists and reports can still refer to them by name. It maps each to the two reports so you can see what was covered and what the new reading lists add. Papers were classified from their file names and titles only; the 44 PDFs were not read.

## 01_Validation and Overfitting (8) -> Report B section 5 (validation protocol)

- A Novel Approach to Trading Strategy Parameter Optimization
- A Validated Volatility-Volume-Gap Classifier for Regime Identification MNQ Data (on-instrument: MNQ)
- Backtest Overfitting in Financial Markets
- DETERMINING OPTIMAL TRADING RULES
- Do Simple Technical Trading Rules Survive Realistic Costs
- Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures (on-instrument: MNQ)
- The Deflated Sharpe Ratio
- The Probability of Backtest Overfitting

## 02_Momentum and Trend Following (7) -> Report B section 2 (strategy families)

- A Century of Evidence on Trend-Following Investing
- A Profitable Day Trading Strategy (Zarattini, Barbon, Aziz)
- Do Momentum and Reversal Strategies Work in Commodity Futures
- Momentum Crashes
- Momentum Strategies in Futures Markets and Trend-Following Funds
- Profitability of Momentum Strategies
- Trend Following, Risk Parity and Momentum in Commodity Futures

## 03_Mean Reversion Strategies (4) -> Report B ranks mean reversion low unless payoff >= 1:1

- A Comparative Study of Breakout and Mean-Reversion Strategies
- Analysis of the Bollinger Bands and Mean Reversion Trading Strategy in SP 500
- Design and Development of Mean Reversion Strategies on QuantConnect Platform
- Mean Reversion Trading Strategies and the Black Swan Events

## 04_Machine Learning and AI Trading Systems (9) -> neither report endorses ML as an edge source; treat as search-method background

- 101 Formulaic Alphas
- A Hybrid AI-Driven Trading System Integrating Technical Analysis, Machine Learning and Financial Sentiment
- A Reinforcement Learning and Genetic Algorithm Framework for Intraday Index Futures Trading
- Adaptive Limits of Algorithmic Cryptocurrency Trading from a Complex Adaptive Systems Perspective
- Deep Reinforcement Learning on a Multi-Asset (title truncated in filename)
- Deep Reinforcement Learning for Trading
- Expectancy-Optimised Algorithmic Trading on Gold Spot Markets
- Improving Performance with Fast Alphas
- Machine Learning in Futures Markets

## 05_Market Microstructure and Order Flow (3) -> Report B: needs book data and speed; background only

- How Order Flow Affects Stock Price
- Order Flow and Exchange Rate Dynamics in Electronic Brokerage System Data
- Order Flow and Exchange Rate Dynamics

## 06_Prop Firm Economics and Risk Sizing (4 files, 3 distinct papers) -> Report A (math and economics)

- An Actuarial Analysis of Proprietary Trading Firm Challenges
- Day Trading Adapting to Change
- The Kelly Criterion and the Stock Market (two copies: TheKellyCriterionandtheStockMarket.pdf and The Kelly Criterion and the Stock Market.pdf)

## 07_Volume_and_Other_Strategies (9) -> mostly foundational finance theory

- Capital Asset Prices: Market Equilibrium
- Efficient Capital Markets (Fama 1970)
- Forecasting Intraday Trading Volume
- Managerial Behavior, Agency Costs and Ownership Structure
- Modern Portfolio Theory 1950
- The Limits of Arbitrage
- The Pricing of Options and Corporate Liabilities
- The Arbitrage Theory of Capital Asset Pricing
- Volume Weighted Average Price (VWAP): The Holy Grail for Day Trading Systems (Zarattini, Aziz)

## Observations

- Strong coverage: validation and overfitting, momentum, prop-firm economics, MNQ-specific work.
- Not in the repo and needed by Report B's recommended strategy: the Noise Area paper (Zarattini, Aziz, Barbon, SFI 24-97), Gao et al. and Baltussen et al. on intraday momentum, Moreira-Muir on volatility-managed sizing, the multiple-testing papers (Harvey-Liu-Zhu, White), the pre-FOMC paper, and FRED VIXCLS. These are in the reading lists in folders 01 to 04.
- Folder 07 was mostly foundational theory (CAPM, Black-Scholes, Fama, Markowitz, agency costs). It is low-leverage for the eval goal. The AI assist picks excerpts by plain keyword overlap, so off-topic papers can add noise to its suggestions.
- Folder 04's machine-learning papers carry the same caution as Report B: no audited result at the eval spec.
- A few entries (for example the QuantConnect-platform and cryptocurrency papers) may have been practitioner or student work. Worth a second look before treating them as peer-reviewed evidence.
- The two copies of the Kelly paper were duplicate content in the app's index.
