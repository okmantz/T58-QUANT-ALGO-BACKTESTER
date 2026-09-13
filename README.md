# T58 Trading — Quant Algo Backtester

Full Tutorial: https://drive.google.com/file/d/1sgmIN7Q5vTkf8PESkHDFuSnRRUq1TYFp/view?usp=drive_link

Web Version: https://github.com/user-attachments/assets/a8a5fa7d-a95a-437e-9146-ff7ac05143b5

The one-stop shop for taking a trading idea from "here's a script" to
"here's a validated, prop-firm-ready strategy" — without leaving one app.
Import or write a strategy in any of four formats, validate it against real
prop-firm rules, search for a version that actually holds up out-of-sample,
optionally let a local AI help along the way, and walk away with one report
that answers the only question that actually matters:

> **"If I trade this strategy under these prop-firm rules, what is the probability that I pass the evaluation and reach my first payout?"**

This is a prop-firm-first backtester, not a traditional long-term investment
backtester. The core loop:

```
Market Data + Strategy + Risk + Prop Rules
        -> Historical Backtest
        -> Prop-Firm Simulation
        -> Monte Carlo Simulation (thousands of simulated accounts)
        -> Probability of Success
        -> Comprehensive Report
```

Everything else builds on that loop, organized into workflow stages:

- **Create** — Speed Run, Generate Strategies (AI), Research Agent
- **Test** — Run & Report, Payout Probability, Prop-Firm Recommender
- **Optimize** — Search Lab, Iterative Refinement, Full Pipeline, Quick
  Optimize, Multi-Objective Optimization, Evolution Lab
- **Validate** — Walk-Forward Optimization, Walk-Forward-Aware GA, CPCV /
  PBO, Parameter Sensitivity, Regime Survival Matrix
- **Champion** — Multi-Asset Portfolio, Multi-Strategy Ensemble, Family
  Diversity
- **Deployment** — Forward Test (MT5 demo), Overnight Autopilot,
  Auto-Retune, Deploy Live (connection management), Live Market monitor

Plus **Quant Lab** (a dozen standalone analysis tools), **Options Outlook**,
an **AI Assistant** dashboard (news + market scanner + chat), and a
beginner-friendly **Resources** guide. An optional local **AI Assist**
(Ollama) can participate in several of these — see below.

Three ways to run it: a **Windows desktop app (.exe)**, a **local Python app**
(any OS), or a **mobile-friendly web app** you open in a phone browser.
Nearly every feature has full web/desktop parity — the exceptions (MT5-only
Deployment tools, a couple of small picker UIs) are called out where they
come up below.

## Running it

**Windows `.exe`, no Python needed:** push this repo to GitHub, then go to
**Actions → build-exe → (latest run)** and download the
**T58-Quant-Algo-Backtester-windows-exe** artifact. Tagging a release also
attaches it to a GitHub Release directly.

**Local Python app (Windows/macOS/Linux):**

```bash
git clone <this-repo-url>
cd T58-Quant-Algo-Backtester
python3 -m venv .venv                 # Windows: py -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate.bat
pip install -r config/requirements.txt
python -m app.main                    # desktop GUI
```

Add `--cli --csv data/examples/EURUSD_5M_sample.csv --sims 10000` to run
headless instead; see `python -m app.main --help` for the full flag list.
Reports land in `reports/report.{json,html}` — `report.html` is
self-contained and print-to-PDF friendly.

**Mobile / web app:** Tkinter can't run on a phone, so mobile access is a
lightweight Flask web app reusing the exact same engine. Easiest path:
grab `T58-Web-App-Windows.zip` from [Releases](../../releases), run
`T58-Web-App.exe`, and scan the QR code it shows with your phone (same
Wi-Fi). From source instead: `python run_web.py`, which prints your LAN
address and a QR code. Away from home, [Tailscale](https://tailscale.com)
(free) lets your phone reach it from anywhere with no port forwarding —
details on the app's own **Phone access** page. Full walkthrough:
[`HOW_TO_OPEN_ON_YOUR_PHONE.md`](HOW_TO_OPEN_ON_YOUR_PHONE.md).

## Core workflow — Run & Report

1. **Upload Market Data** — CSV/TSV/Parquet/`.zip`/`.7z`, auto column
   mapping, gap/duplicate detection. A sample dataset ships at
   `data/examples/EURUSD_5M_sample.csv`; broader datasets (XAUUSD, EURUSD,
   GBPUSD, indices, etc.) ship under `data/raw`. Multiple files can be
   selected for multi-timeframe analysis (e.g. 60m bias + 15m zone + 5m
   entry). Alpaca API fetch is also supported for US equities/crypto.
2. **Import/Create a Strategy** — four formats, all reduced to the same
   standardized signal series:
   - **Manual Builder** — a full no-code visual builder: entry conditions
     across price/EMA/SMA/WMA/VWAP/RSI/MACD/ATR/Bollinger and
     market-structure concepts (swing highs/lows, liquidity sweeps, break
     of structure, fair value gaps, order blocks, session ranges); exits
     via fixed/ATR stops & targets, trailing stop, break-even, time-based
     exit, and more.
   - **Python** — upload/paste a `.py` exposing `generate_signals(df)`.
   - **PineScript** and **MQL5** — real parsers for a tested subset of each
     language (see below); anything unsupported fails loudly rather than
     producing a silently inaccurate backtest.

   Every strategy can be checked for **lookahead bias**, which re-runs
   signal generation on truncated data and flags the exact bar where a
   leak first appears. A persistent **Strategy Library** saves any
   strategy with status tags and its full backtest/search history.
3. **Enter Prop-Firm Rules** — account size, profit target, daily loss
   limit, max drawdown (trailing/static, intrabar/EOD), consistency rule,
   payout threshold/cap/frequency. **Quick-fill from a preset** — FTMO,
   Apex, TopStep, The5%ers, FundedNext, and Lucid rules are built in
   (dated, with a note on what to re-verify — firms change terms often).
4. **Configure Risk & Execution** — fixed-$ or %-of-equity risk per trade,
   commission, slippage, spread, pip size (with a "detect from data"
   helper).
5. **Backtest → Prop Simulation → Monte Carlo → Report** — one click, or
   the `--cli` flag.

## Create

- **Speed Run** — an all-in-one discover-to-verdict tool for a hard
  deadline: a fast, wide multi-family search feeding straight into Full
  Pipeline validation of the top survivors, picking the best
  READY/MARGINAL winner. If nothing clears the bar, it turns the run's own
  rejection reasons into concrete "what to try next" suggestions.
- **Generate Strategies (AI)** — a local Ollama model drafts a new
  strategy from a plain-language idea, saved to the library as a `draft`.
- **Research Agent** — a tool-calling AI that investigates a strategy over
  several reasoning steps, each a read-only call into this app's own
  engine (backtest, prop-sim, Monte Carlo, walk-forward, regime,
  sensitivity) — never an invented number.

## Test

- **Run & Report** — the core workflow above.
- **Payout Probability** — a lifecycle simulation carrying a strategy
  through funding milestones (eval pass → funded → payout #1 → payout #2
  → …), reporting a funnel of probabilities at each stage instead of one
  flat number. Includes an optional **account-scaling stress test**: model
  a funded account scaling up after a run of consecutive payouts (a common
  shape at several modern prop firms) and see how it changes your expected
  total payout.
- **Prop-Firm Recommender** — the reverse question: given your strategy's
  own trade sequence, scores it against every prop-firm preset (or a
  chosen subset) and returns a ranked table of which firm's rules it
  actually fits best, instead of checking one firm at a time by hand.

## Optimize

- **Search Lab** — discovers and validates *many* candidates in one run
  via a 5-stage funnel: generate a candidate pool from a named strategy
  family, cheap Stage-1 filter, GA refinement, a full validation gate
  (Monte Carlo + walk-forward holdout + robustness + Deflated Sharpe), then
  a leaderboard and champion promotion.
- **Iterative Refinement** — a genetic-algorithm parameter search across
  all four strategy sources, toward a configurable fitness metric.
- **Full Pipeline** — one button chaining baseline backtest → lookahead
  check → walk-forward-aware GA → re-validated report → OOS/holdout checks
  → a plain READY/MARGINAL/NOT READY verdict with reasons. Winners are
  auto-saved to the Strategy Library.
- **Quick Optimize** — a lighter, single-strategy version of the above.
- **Multi-Objective Optimization** — a real NSGA-II Pareto front across
  several objectives at once (e.g. Sharpe, drawdown, eval-pass
  probability) instead of collapsing them into one score.
- **Evolution Lab** — runs generate → filter → validate → keep-winners →
  mutate unattended, for as long as you leave it running, ranked by a
  composite **PROP FITNESS** score (pass probability × payout probability
  × robustness × OOS consistency, penalized for thin samples, overfitting,
  and concentration risk — not raw net profit). Every genome's backtest
  now runs under the same daily-loss/drawdown circuit breakers your prop
  rules define, so a candidate that would blow the account gets stopped
  during its own backtest rather than only being caught later. Checkpoints
  after every generation, so STOP-then-START resumes exactly where it left
  off. Candidates are Manual Strategy Builder configs, not mutated
  uploaded code.

## Validate

Five statistical-rigor tools answering different "how much should I trust
this backtest?" questions:

- **Walk-Forward Optimization** — rolling/anchored folds, a fresh GA per
  fold, chained into one continuous OOS equity curve.
- **Walk-Forward-Aware GA** — the same GA, scored only on chained OOS data
  so the search can't just curve-fit harder; reports an overfitting gap.
- **CPCV / PBO** — Combinatorial Purged Cross-Validation stress-tests one
  strategy across many train/test partitions; Probability of Backtest
  Overfitting checks a *pool* of candidates for whether the apparent best
  one is really better than a coin flip out-of-sample.
- **Parameter Sensitivity** — 1D sweeps of every tunable parameter with
  cliff detection, plus an optional 2D heatmap.
- **Regime Survival Matrix** — classifies every bar into a regime (trend,
  volatility, session, environment) and attributes the strategy's own
  trades to whichever regime was active, surfacing where its losers
  concentrate.

## Champion

- **Multi-Asset Portfolio** — runs a strategy across several instruments,
  computes their correlation, re-weights risk, and merges legs into one
  shared account curve.
- **Multi-Strategy Ensemble** — combines several different,
  weakly-correlated strategies on the same instrument (blend or vote).
- **Family Diversity** — reports per-family performance from a Search Lab
  run, so you can see if the leaderboard is genuinely diverse.
- An **Automated Portfolio Composer** (under Quant Lab) searches the
  Strategy Library for the best N-strategy combination automatically.

Cost-stress-adjusted fitness (re-backtesting every GA candidate at
inflated spread/slippage/commission) and a declarative adaptive-risk
layer (de-risk after N losses, cut size on a bad day, coast near a profit
target) apply across every optimizer above.

## Deployment

- **Forward Test (MT5 demo)** — deploys any Strategy Library strategy to a
  free MT5 demo account and watches it trade forward against real broker
  prices, reusing the exact signal/sizing engine the backtester uses.
  Journals every trade, flags win-rate drift, ships a kill switch. Demo
  accounts only. Desktop-only (needs a running MT5 terminal).
- **Overnight Autopilot** — chains Speed Run discovery straight into a
  live MT5 forward test of the winner: runs the search, and if it lands on
  a READY/MARGINAL strategy, starts forward-testing it automatically and
  writes one dated report to read in the morning. It can't finish a full
  forward-test evaluation overnight (that's a live, ongoing process) — what
  it gets you is "the search ran and a winner is already live on your demo
  account" instead of needing to babysit the search, then load, then start
  the forward test by hand. Desktop-only (same MT5 requirement as Forward
  Test); available from both the Forward Test tab and Speed Run.
- **Auto-Retune** — a one-click check against Strategy Health/Drift
  Monitor: if a running forward test's realized performance has drifted
  outside its predicted band, this re-runs Quick Optimize against it
  automatically and shows you the before/after, instead of you needing to
  notice the drift flag and start the re-tune by hand yourself.
- **Deploy Live** — connection management for a real, funded prop-firm
  account. **Actual live order placement is a deliberate stub** — this is
  connection/credential plumbing only, desktop-only, pending a proper
  authentication layer before going further.
- **Live Market monitor** — a read-only view of live bars/trades over your
  MT5/Alpaca connection.

## Quant Lab

A dozen standalone analysis tools: Universal Strategy Translator (Manual
config → clean PineScript v5 or MQL5), Auto Regime Selector, Strategy
Health / Drift Monitor, Automated Portfolio Composer, Pairs
Screener/Backtest, Options Pricing Calculator, Order Book Simulator,
Sentiment-Price Correlation, Portfolio Optimizer (Markowitz), Volatility
Surface, and Factor Model.

## Options Outlook & AI Assistant

**Options Outlook** generates deterministic call/put candidates via
Black-Scholes, with an optional Ollama-ranked narrative on top. The **AI
Assistant** dashboard combines a ForexFactory news panel, a best-markets
scanner, and chat, reading live bars from your MT5/Alpaca connection.

## Resources

A curated beginner's guide to trading fundamentals — market structure,
liquidity, supply & demand, entry models — for anyone who wants a running
start before backtesting their first strategy.

## AI Assist (optional, local Ollama)

Several tools above can optionally call a local [Ollama](https://ollama.com)
model — Full Pipeline's GA search, the Research Agent, Generate
Strategies, and Options Outlook/AI Assistant's narrative layer. Every
GA-relevant suggestion still passes through the same backtest → prop-sim →
Monte Carlo pipeline as any other candidate: the model proposes numbers
for already-discovered tunable parameters, never writes strategy code, and
can never displace a genuinely better candidate the GA already found.

**Setup:** install Ollama and pull a model (`ollama pull llama3.1`), then
check **Enable** and hit **Test Connection** wherever a page has an **AI
Assist** section. Off by default everywhere; an unreachable Ollama
degrades quietly rather than breaking the feature.

## Strategy format support

**PineScript (subset):** `open/high/low/close`, `input.int`/`input.float`,
`ta.sma`/`ta.ema`/`ta.wma`/`ta.rsi`, `ta.crossover`/`ta.crossunder`,
`strategy.entry`/`strategy.close` with `when=` conditions, and
`// T58_SL_PIPS=20` / `// T58_TP_PIPS=40` directive comments. Not
supported: custom functions, arrays, `security()`/multi-timeframe
requests, plotting/alerts.

**MQL5 (subset):** direct-value `iMA()`/`iRSI()` calls, C-style
conditions, `trade.Buy`/`trade.Sell`/`OrderSend()` for entries, the same
`T58_SL_PIPS`/`T58_TP_PIPS` directives. Not supported: `CopyBuffer()`
indicator handles, custom indicators, multi-symbol logic, trailing stops.

Both fail loudly with a clear error on anything unsupported, rather than
silently producing an inaccurate backtest.

## Tests

```bash
pytest -q tests
```

## Disclaimer

Simulated results are estimates derived from historical data and
resampling. Past performance and simulated outcomes do not guarantee
future results.
