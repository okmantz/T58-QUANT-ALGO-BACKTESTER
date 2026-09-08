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

Everything else in the app builds on that loop. It's organized into the same
six workflow stages the sidebar uses, plus a few standalone tools:

- **Create** — Speed Run (discover → validate a strategy end-to-end in one
  run against a hard deadline), Generate Strategies (AI drafts a strategy
  from a plain-language idea), Research Agent (a tool-calling AI that
  investigates using the app's own engine, never invented numbers)
- **Test** — Run & Report (the core loop above), Payout Probability
  (lifecycle simulation through funding milestones)
- **Optimize** — Search Lab, Iterative Refinement, Full Pipeline, Quick
  Optimize, Multi-Objective Optimization, Evolution Lab
- **Validate** — Walk-Forward Optimization, Walk-Forward-Aware GA, CPCV /
  PBO, Parameter Sensitivity, Regime Survival Matrix
- **Champion** — Multi-Asset Portfolio, Multi-Strategy Ensemble, Family
  Diversity
- **Deployment** — Forward Test (MT5 demo), Deploy Live (funded-account
  connection management — live order placement is a deliberate stub, see
  below), Live Market monitor

Plus: **Quant Lab** (a dozen standalone analysis tools — pairs screening,
options pricing, portfolio optimization, volatility surface, and more),
**Options Outlook** (deterministic Black-Scholes option candidates,
optionally AI-ranked), the **AI Assistant** (forex-factory news + a
best-markets scanner + chat, over your MT5/Alpaca feed), and **Resources**
(a beginner trading-education guide). All are described in more detail
below. An optional local **AI Assist** (Ollama) can participate in several
of these — see **AI Assist** further down.

Three ways to run it: a **Windows desktop app (.exe)**, a **local Python app**
(any OS), or a **mobile-friendly web app** you open in a phone browser. Every
feature above has full web/desktop parity except two (a PBO candidate-pool
picker and a 2D sensitivity heatmap picker, both awaiting a small UI, backend
already there) and the MT5-dependent Deployment tabs, which are inherently
desktop-only (explained where they come up below).

## 1. Windows `.exe`

Every push to `main` (and every `vX.Y.Z` tag) triggers `.github/workflows/build-exe.yml`,
which builds a single-file Windows executable with PyInstaller and uploads it
as a workflow artifact — no local Python install needed to get the `.exe`:

1. Push this repo to GitHub.
2. Go to **Actions → build-exe → (latest run)**.
3. Download the **T58-Quant-Algo-Backtester-windows-exe** artifact — it contains `T58-Quant-Algo-Backtester.exe`.
4. Tagging a release (`git tag v0.1.0 && git push --tags`) also attaches the `.exe` directly to a GitHub Release.

To build it locally on Windows instead:

```bat
pip install -r config/requirements.txt pyinstaller
pyinstaller --noconfirm --onefile --windowed --name T58-Quant-Algo-Backtester ^
  --paths . --add-data "data/examples;data/examples" run_app.py
```

**Important:** the PyInstaller entry point is `run_app.py`, at the repo root
— **not** `app/main.py`. `run_app.py` exists specifically so PyInstaller's
import analysis can resolve the `app` package correctly; pointing it at
`app/main.py` directly produces a broken `.exe` that crashes on launch with
`ModuleNotFoundError: No module named 'app.ui.main_window'`. Keep the entry
point as `run_app.py` if you ever rebuild the workflow or the local command
by hand.

The `.exe` launches the same Tkinter desktop GUI described below. A handful
of example strategies under `strategies/` (Python, PineScript, MQL5) ship
bundled inside the `.exe` itself and self-seed into the Strategy Library the
first time it runs, so the library isn't empty on a fresh install.

## 2. Local Python app (any OS)

Works on Windows, macOS, and Linux — the only two things that trip people up
are (a) not being *inside* the extracted/cloned folder yet, since
`config/requirements.txt` is a path relative to the repo root, and (b) some
Linux distros (Debian, Ubuntu, Linux Mint) only ever install a `python3`
command, never a plain `python` — see the notes below the block if either
happens to you.

```bash
# 0. Get the code, then move INTO that folder -- every command after this
#    assumes you're standing inside it (extract first if you downloaded a
#    "Code -> Download ZIP" instead of using git clone).
git clone <this-repo-url>
cd T58-Quant-Algo-Backtester

# 1. Create a virtual environment (isolates this app's packages).
python3 -m venv .venv                 # Windows: py -m venv .venv

# 2. Activate it -- run ONE of these:
source .venv/bin/activate             # macOS / Linux, bash or zsh
.venv\Scripts\activate.bat            # Windows, Command Prompt (cmd.exe)
.venv\Scripts\Activate.ps1            # Windows, PowerShell

# 3. Install dependencies (relative to the repo root you cd'd into above).
pip install -r config/requirements.txt

# 4. Run it.
python -m app.main                    # desktop GUI
python -m app.main --cli --csv data/examples/EURUSD_5M_sample.csv --sims 10000   # headless
```

**Notes if a command above didn't work:**

- **`Command 'python' not found, did you mean 'python3'`**: use `python3`
  only for creating the venv (step 1) — once it's *activated* (step 2), the
  plain `python` command always points at the venv's own Python, so steps
  3-4 work as written on every OS including Linux.
- **`ERROR: Could not open requirements file`**: you're not standing in the
  repo folder — `cd` into it first, then check with
  `ls config/requirements.txt` (macOS/Linux) or `dir config\requirements.txt`
  (Windows).
- **PowerShell says running scripts is disabled**: run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
  `.venv\Scripts\activate.bat` in Command Prompt instead.
- You'll know the venv is active because your prompt gets a `(.venv)`
  prefix — if not, steps 3/4 will hit your system Python instead.

CLI output is written to `reports/report.{json,html}` plus `report_summary.csv`
and `report_trades.csv`. Open `report.html` in a browser — it's self-contained
(charts included) and print-to-PDF friendly. See **CLI reference** below for
every other headless mode.

## 3. Mobile app (web / installable PWA)

Tkinter (the desktop GUI toolkit) can't run on a phone, so mobile access is a
lightweight **Flask web app that reuses the exact same engine** — no logic
is duplicated between desktop and mobile.

### Easiest: download `T58-Web-App.exe` (no Python install, no terminal)

Grab `T58-Web-App-Windows.zip` from [GitHub Releases](../../releases) (built
by `.github/workflows/build-web-exe.yml`), extract it, and double-click
`T58-Web-App.exe`. It finds your PC's Wi-Fi address and pops up a **QR
code** — scan it with your phone (same Wi-Fi network) to open the app, then
use your browser's **"Add to Home Screen"** for a real app icon. Full
step-by-step: [`HOW_TO_OPEN_ON_YOUR_PHONE.md`](HOW_TO_OPEN_ON_YOUR_PHONE.md).

Your phone is a remote screen for the app running on your PC — no hosting
account, no Play Store — but your PC does need to stay on while you use it.

### Alternative: run it from source

```bash
pip install -r config/requirements.txt
python run_web.py
```

This prints your LAN address, opens a QR code, and serves on
`http://0.0.0.0:5000` — `python -m app.web.server` works identically. The
running app also has this same address + QR code any time at its **Phone
access** sidebar link (`/mobile-access`).

**Getting "site can't be reached" on your phone almost always means:**

- You typed `https://` instead of `http://` — this server doesn't speak HTTPS.
- You typed `127.0.0.1`/`localhost` instead of the LAN address the
  banner/`/mobile-access` page shows — those only ever mean "this computer."
- Phone and computer aren't on the same Wi-Fi (phone on cellular data, or a
  guest/isolated network).
- Windows Firewall is blocking it — allow Python for Private networks under
  **Windows Security → Firewall & network protection**.

From your phone's browser, tap **Share → Add to Home Screen** (iOS) or the
**Install app** prompt (Android Chrome) to install it as a standalone PWA.
This is designed to run entirely on your own Wi-Fi, for free.

### Away from home? Use Tailscale (works from anywhere)

The plain LAN address only works while your phone shares Wi-Fi with this
computer. [Tailscale](https://tailscale.com) is a free, private VPN mesh
that lets your phone reach this app from any network — no port forwarding,
nothing exposed to the open internet.

1. On **this computer**, install Tailscale from
   [tailscale.com/download](https://tailscale.com/download) and sign in.
2. On **your phone**, install the Tailscale app and sign in with the
   **same account**.
3. Start the web app as usual. The console banner now shows a second
   `100.x.x.x` address that works from anywhere; `/mobile-access` shows the
   same with its own QR code and setup instructions if Tailscale isn't
   detected yet.
4. Open that address (or scan its QR code) on your phone.

Purely additive and optional — without Tailscale installed, everything works
exactly as before.

**Want the literal desktop app (Tkinter) on your phone?** Not something this
app can add code for, but Tailscale plus a remote-desktop tool (Windows'
built-in Remote Desktop, or Tailscale SSH + VNC) can stream the actual
desktop window to your phone — noticeably less responsive than the
purpose-built web app above, which remains the recommended mobile path.

## Core workflow — Run & Report

<img width="1927" height="1038" alt="image" src="https://github.com/user-attachments/assets/ef78c799-b48f-4fb8-8b1c-6d2f7c8f8e9a" />

1. **Upload Market Data** — CSV import with auto column-mapping, timestamp/OHLC
   validation, duplicate & gap detection (`app/data/importer.py`). A sample
   dataset ships at `data/examples/EURUSD_5M_sample.csv`; broader historical
   datasets (XAUUSD, EURUSD, GBPUSD, S&P 500, NASDAQ, etc.) ship under
   `data/raw`.

   **Supported file types**, dispatched by extension:
   - `.csv` / `.tsv` / `.txt` — delimiter and header/no-header both
     auto-detected; headerless 6-column files are assumed
     `timestamp, open, high, low, close, volume`.
   - `.parquet` — via `pyarrow` (already bundled/required).
   - `.zip` / `.7z` archives — opened automatically, the OHLCV member inside
     is located and read (`.7z` needs `py7zr`, also bundled).
   - **Column names**: a wide alias list maps common vendor/broker names
     (`ts`, `Gmt time`, `o/h/l/c/v`, etc.) to the standard schema, with a
     test-parse fallback for anything unrecognized. Unused extra columns are
     ignored rather than causing an import error.

   The picker supports selecting **more than one file** for multi-timeframe
   analysis (e.g. 60-minute bias + 15-minute zone + 5-minute entry). The
   finest file becomes the base timeframe; coarser files merge onto it
   (`app/data/multi_timeframe.py`, as-of/backward merge, no lookahead) as
   `tfNN_open/high/low/close/volume` columns usable directly in the strategy
   builder.

   **Alpaca API fetch** (`app/data/alpaca_source.py`): pulls bars directly
   from Alpaca (US equities + crypto only) using saved API keys and saves
   them into `data/raw/` alongside local files.

2. **Import/Create a Strategy** — four adapters, all reduced to the same
   standardized `-1/0/1` signal series before hitting the backtest engine
   (`app/strategy/`):

   - **Manual Builder** — a full, no-code visual builder
     (`app/ui/condition_builder.py` + `app/strategy/manual.py`): strategy
     info (name, instrument, timeframe, session, direction); entry
     conditions across Price/EMA/SMA/WMA/VWAP/RSI/MACD/ATR/Bollinger/Highest-
     Lowest/Volume/Candle shape/Cross Above-Below, AND/OR-chained; advanced
     market-structure conditions (Swing High/Low, Liquidity Sweep, Break of
     Structure, Change of Character, Fair Value Gap, Order Block, Session/
     Previous-Day/Opening-Range High-Low, ATR/Volatility Regime); exits via
     fixed or ATR-based Take Profit/Stop Loss, ATR trailing stop,
     break-even, time-based exit, max bars in trade, opposite-signal exit,
     and indicator exit conditions.
   - **Python** — upload/paste a `.py` exposing `generate_signals(df)` (see
     `app/strategy/python.py`'s docstring for the `.attrs` mechanism used
     for per-trade dynamic stop/target/trailing distances, and the
     multi-timeframe-bias lookahead trap it specifically warns about).
   - **PineScript** (`app/strategy/pinescript.py`) and **MQL5**
     (`app/strategy/mql5.py`) — real parsers for a tested subset of each
     language (see below), not full implementations. Anything unsupported
     raises a clear `StrategyError` rather than silently producing an
     inaccurate backtest.

   Every strategy, regardless of source, can be checked for **lookahead
   bias** (`app/strategy/lookahead_check.py`): it re-runs signal generation
   on data truncated right after real signal checkpoints and diffs the
   result against the full-data run, naming the exact bar a leak first
   appears at. The same bug class — a naive higher-timeframe filter leaking
   the still-forming current HTF bar — has been caught this way in real
   uploaded strategies and flipped their reported result from profitable to
   a clear loser; see `app/strategy/mtf.py`'s docstring for the numbers.

   **Strategy Library** (`app/strategy/library.py`): any Python/PineScript/
   MQL5/Manual strategy can be saved inside the app's own persistent data
   folder — shows up in the library picker on every future run, with status
   tags and backtest/lookahead/search result history.

3. **Enter Prop-Firm Rules** — account size, eval profit target, daily loss
   limit, max drawdown (trailing/static, intrabar/end-of-day), consistency
   rule, minimum trading days, payout threshold/cap/frequency, required
   buffer, max position size (`app/prop/simulator.py::PropRules`).

4. **Configure Risk & Execution** — fixed-$ or %-of-equity risk per trade,
   max trades/day, commission, slippage, spread, pip size
   (`app/backtest/risk.py::RiskConfig`). **"Detect pip size from data"**
   suggests a value from what's loaded in Step 1 — leaving pip size at its
   FX default against a non-FX instrument (stocks, indices, crypto, JPY
   pairs) is the single most common cause of a nonsensical position size.

5. **Backtest → Prop Simulation → Monte Carlo → Report** — one click, or the
   `--cli` flag.

## Create

- **Speed Run** — an all-in-one discover-to-verdict tool for when you have a
  hard deadline: chains a fast, wide multi-family search into parallel Full
  Pipeline validation of the top survivors and picks the best READY/MARGINAL
  winner (`app/orchestration/speed_run.py`). If nothing clears the bar, it
  mines the run's own rejection reasons into concrete "what to try next"
  suggestions instead of a bare "no winner."
- **Generate Strategies (AI)** — a local Ollama model drafts a new strategy
  file from a plain-language idea, saved into the Strategy Library tagged
  `draft` (`app/ai/strategy_generator.py`). A draft is a starting point, not
  a validated strategy — run it through Search Lab/Full Pipeline before
  trusting it.
- **Research Agent** — a ReAct-style tool-calling AI that investigates a
  strategy across several reasoning steps, each a read-only call into this
  app's own already-validated engine (backtest, prop-sim, Monte Carlo,
  walk-forward, regime, sensitivity, cost-stress) — never a guess, never an
  invented number. See **AI Assist** below for full setup and the "engine is
  the authority" design rule.

## Test

- **Run & Report** — the core workflow above.
- **Payout Probability** — a lifecycle simulation carrying a strategy through
  configurable funding milestones (eval pass → funded → first payout → next
  payout tier), reporting a funnel of probabilities at each stage rather
  than one flat "pass probability" number (`app/prop/survival_engine.py`).

## Optimize

- **Search Lab** — discovers and validates *many* candidates in one run
  instead of tuning one you already picked, via a 5-stage funnel
  (`app/search/batch_runner.py`): **Generate** a candidate pool (a
  combinatorial grid over a named-hypothesis family — trend/breakout,
  multi-timeframe pullback, mean-reversion band, volatility breakout,
  session/time-of-day effect, volume imbalance, statistical pairs, plus
  several more families tuned for prop-eval math, see
  `app/search/strategy_space.py::FAMILIES` — or over an uploaded strategy's
  own parameters); **Stage 1** cheap filter (trade count + profit factor);
  **Stage 2** GA refinement (same engine as Iterative Refinement, including
  cost-stress penalty); **Stage 3** validation gate (full Monte Carlo +
  walk-forward holdout + parameter-neighborhood robustness + Deflated
  Sharpe); **Stage 4/5** leaderboard (SQLite, `app/search/results_db.py`)
  and champion promotion to a full standalone report. Auto-relaxes Stage 1
  thresholds if nothing survives, rather than dead-ending.
- **Iterative Refinement** — a genetic-algorithm parameter search: re-runs
  the current strategy with mutated numeric parameters on the same data,
  keeps the best performers each generation (elitism + tournament selection
  + random immigrants), toward a configurable fitness metric (composite
  prop score, eval-pass probability, first-payout probability, expected
  payout, net profit, profit factor, Sharpe —
  `app/optimize/refinement.py::FITNESS_METRICS`). Works across all four
  strategy sources via shared gene discovery
  (`app/optimize/parameter_space.py`, `code_parameter_space.py`): Manual
  Builder numeric fields, `SCREAMING_SNAKE_CASE` Python constants,
  `input.int()`/`input.float()` in PineScript, `iMA()`/`iRSI()` periods in
  MQL5. Produces its own report and an "apply best config back to Strategy"
  button — the normal Run & Report pipeline is unaffected unless enabled.
- **Full Pipeline** — one button that hands off between the tools above
  automatically: baseline backtest + lookahead check → walk-forward-aware GA
  search (optionally AI-assisted) → re-validated final report → OOS fold
  check → holdout check → a plain READY/MARGINAL/NOT READY verdict with the
  reasons behind it (`app/orchestration/full_pipeline.py`). For Python/
  PineScript/MQL5 strategies, the winner is also saved into the Strategy
  Library tagged `validated`. The report surfaces execution-integrity
  warnings (pip-size mismatches, gap-through stop fills), the verdict, and
  the winning parameters front and center rather than buried in a log.
- **Quick Optimize** — a lighter single-strategy version of the above: pick
  a Strategy Library entry, click Optimize, and the same walk-forward-aware
  GA auto-tunes it toward pass/win-rate/payout targets, saving the result as
  a new draft (`app/orchestration/quick_optimize.py`).
- **Multi-Objective Optimization** — a real NSGA-II implementation
  (non-dominated sorting + crowding distance) producing a genuine Pareto
  front across several objectives at once (e.g. Sharpe, max drawdown,
  eval-pass probability) instead of collapsing them into one weighted score
  (`app/optimize/multi_objective.py`). Picking a winner from the front is
  left as a judgment call.
- **Evolution Lab** — runs the whole generate → filter → validate →
  keep-the-winners → mutate loop unattended, for as long as you leave it
  running (`app/evolution/`):

  ```
  RESEARCH (knowledge graph) -> GENERATE -> PRE-FILTER + BACKTEST
      -> ROBUSTNESS + OOS + MONTE CARLO + PROP SIMULATION -> CPCV / PBO
      -> STRESS TEST -> CLUSTER -> KEEP TOP N -> record -> MUTATE -> repeat
  ```

  Candidates are ranked by **PROP FITNESS** (`app/evolution/prop_fitness.py`)
  — pass probability × payout probability × robustness × OOS consistency
  over drawdown, penalized for thin trade counts, parameter sensitivity,
  high PBO, in/out-of-sample degradation, profit concentration, and long
  losing streaks — not raw net profit. Every candidate tested is logged
  (pass or fail, with the specific rejection reason) to a durable on-disk
  file, and a **knowledge graph** of structural feature vectors → outcomes
  lets later generations lean on what's historically worked. Auto-relaxes
  thresholds after repeated empty generations. Progress (generation number,
  elites, leaderboard, journal) checkpoints to disk after every generation,
  so STOP-then-START (even in a new session) resumes exactly where it left
  off, as long as the same data is loaded; RESET discards the checkpoint.
  Runs on a background thread with its own worker-process pool — safe to
  leave running for hours while working in other tabs.

  Scope, stated plainly: candidates are Manual Strategy Builder configs
  generated from `app.search.strategy_space`'s families — this does not
  mutate uploaded Python/PineScript/MQL5 files.

## Validate

Five statistical-rigor tools, each answering a different "how much should I
trust this backtest?" question a single in-sample run can't answer alone.
All reuse the same gene-discovery/GA machinery as Iterative Refinement, so
they work consistently across every strategy source.

- **Walk-Forward Optimization** (`app/validation/walk_forward_opt.py`):
  rolling/anchored folds, a *fresh* GA search per fold's train window
  applied unchanged to that fold's held-out test window, chained into ONE
  continuous out-of-sample equity curve.
- **Walk-Forward-Aware GA** (`app/optimize/walkforward_ga.py`): the same GA
  as Iterative Refinement, but every candidate's fitness is scored *only* on
  chained OOS fold data — never the training windows — so the search can't
  just curve-fit harder. Reports an "overfitting gap" (in-sample fitness vs.
  chained-OOS fitness of the winner).
- **CPCV / PBO** (`app/validation/cpcv.py`): Combinatorial Purged
  Cross-Validation stress-tests one strategy across many combinatorial
  train/test partitions; Probability of Backtest Overfitting
  (Bailey/López de Prado) checks a *pool* of candidates and reports the odds
  that whichever looks best in-sample is, out-of-sample, no better than a
  coin flip.
- **Parameter Sensitivity** (`app/validation/sensitivity.py`): 1D sweeps of
  every tunable parameter (±X%, with "cliff" detection for a knife-edge
  parameter vs. a stable plateau), plus an optional 2D heatmap.
- **Regime Survival Matrix** (`app/validation/regime_matrix.py`): a
  trade-attribution tool, not a re-validation one — runs the backtest once
  over the whole dataset, classifies every bar into a regime along four
  causal dimensions (trend, volatility, session, and a trending/ranging/
  breakout-style "environment" split), then attributes each of the strategy's own trades to whichever regime was
  active at entry, surfacing which specific market conditions its losers
  are concentrated in so you can gate the strategy off in those conditions
  going forward.

Report generation lives in `app/reports/validation_reports.py` (JSON +
focused HTML per feature, reusing `app/reports/charts.py`'s SVG chart
helpers, including the 2D heatmap). See **CLI reference** below.

**On account-state logic:** `generate_signals(df)` (and its Manual/
PineScript/MQL5 equivalents) is still called once, statelessly, over the
whole dataset before any P&L exists, so no strategy source can implement
its own account-state-dependent logic directly. `app/backtest/adaptive_risk.py`
covers the common cases as a declarative, engine-level layer instead (see
below) — de-risk after N losses, cut size once a daily loss threshold is
hit, coast once X% of the way to a profit target.

## Champion

- **Multi-Asset Portfolio** (`app/portfolio/portfolio.py`): runs a strategy
  across several instruments, computes their return correlation matrix,
  re-weights each instrument's risk (correlated legs sized down), and
  merges every leg's trades into one shared account equity curve — modeling
  "one prop account, several instruments," with a static (whole-window)
  correlation pass and chronological trade-close merging rather than a
  margin-constrained concurrent-position engine (see the module's own
  docstring for the full reasoning). A library-based leg picker lets you add
  legs straight from the Strategy Library.
- **Multi-Strategy Ensemble** (`app/ensemble/ensemble.py`): the mirror case
  — several *different*, weakly-correlated strategies combined on the
  *same* instrument. Two modes: **blend** (each leg trades independently at
  a correlation-adjusted risk weight) and **vote** (combines every leg's
  signal into one majority/threshold-vote entry; risk management inherited
  from the first-listed leg).
- **Family Diversity** — reads a completed Search Lab run and reports
  per-family performance, so you can see whether the leaderboard is
  genuinely diverse or dominated by near-identical variants of one family.

Also available here (and cross-referenced in Optimize/Test): an **Automated
Portfolio Composer** under Quant Lab (below) searches the Strategy Library
for the best N-strategy combination by combined eval-pass probability,
rather than requiring you to hand-pick legs.

**Cost-stress-adjusted fitness, everywhere a GA runs:** Iterative
Refinement, Search Lab's Stage 2, and the Walk-Forward-Aware GA all also
re-backtest every candidate at spread/slippage/commission multiplied by a
configurable stress multiplier (default 2x) and blend that into the fitness
score the GA actually selects on (`app/optimize/refinement.py::apply_cost_stress_penalty`)
— on by default; reported statistics stay nominal, only the GA's selection
signal is adjusted.

**Declarative adaptive risk layer** (`app/backtest/adaptive_risk.py`):
engine-level money-management rules — `consecutive_losses`,
`daily_loss_pct`, `daily_profit_pct`, `progress_to_target_pct` — each
scaling new-entry position size by a configured multiplier once triggered;
multiple active rules stack multiplicatively, and every trade records which
rule(s) were active when it opened. CLI: `--adaptive-risk-rules
'{"rules": [{"trigger": "consecutive_losses", "threshold": 2,
"risk_multiplier": 0.5}], "profit_target_amount_pct": 8.0}'`.

## Deployment

- **Forward Test (MT5 demo)** — deploys any Strategy Library strategy to a
  free MetaTrader 5 demo account and watches it trade forward against real
  broker prices, bar by bar, instead of a CSV — the bridge between "the
  backtest looks good" and "I'd trust this with real money." Reuses the
  exact signal engine and position-sizing math the backtester uses, polls
  only fully-closed bars, enforces the same daily-loss circuit breaker,
  reconciles with MT5's actual open positions on restart, journals every
  trade to a local SQLite log, flags (doesn't auto-stop on) win-rate drift,
  and ships a kill switch. **Demo accounts only** — no live/funded order
  path exists in this module. Requires Windows, a running MT5 terminal
  logged into a demo account, and `pip install MetaTrader5` (conditional in
  `config/requirements.txt` on Windows). Desktop-only: there's no web
  equivalent of "the MT5 terminal on your desk" to connect to from a phone
  browser — the web app explains this plainly if you land on this tab there.
  See `strategies/SCREENING_RESULTS.md` for an honest read on which bundled
  library strategies currently show a real edge.
- **Deploy Live** — connection management for a *real, funded* prop-firm
  account (`app/live_deploy/`): saved/tested account credentials (OS
  keyring-backed, supports multiple named accounts) and a curated prop-firm
  reference list. **Actual live order placement is a deliberate stub, not
  wired up yet** — turning it on is a separate, later decision once this
  plumbing has been used and reviewed. Desktop-only for two reasons: the
  same MT5-terminal-on-this-machine constraint as Forward Test, and because
  this web server has no authentication and listens on every network
  interface — building live-credential entry on top of that would be a real
  step backward in safety before a proper login system exists in front of
  it.
- **Live Market monitor** — a read-only view of live market bars/trades over
  your MT5/Alpaca connection.

## Quant Lab

A dozen standalone analysis tools that don't fit the backtest-a-strategy
workflow above, each its own page (`app/quant_lab/` plus a few tools that
live alongside their subject matter elsewhere in the codebase):
**Universal Strategy Translator** (`app/strategy/translator.py` — turns a
Manual Builder config into clean, broker-facing PineScript v5 or MQL5, and
refuses to translate unsafe SMC-style constructs by name), **Auto Regime
Selector** (`app/strategy/auto_regime_selector.py` — assigns the best
validated strategy per market regime), **Strategy Health / Drift Monitor**
(`app/monitoring/strategy_health.py` — compares forward-test trades against
a strategy's own Monte Carlo distribution and flags drift), **Automated
Portfolio Composer** (`app/portfolio/composer.py` — searches the Strategy
Library for the best N-strategy combo), **Pairs Screener** and **Pairs
Backtest**, **Options Pricing Calculator**, **Order Book Simulator**,
**Sentiment-Price Correlation**, **Portfolio Optimizer** (Markowitz),
**Volatility Surface**, and **Factor Model**.

## Options Outlook

Deterministic call/put candidate generation via Black-Scholes
(`app/ai/options_outlook.py`) for a chosen symbol/horizon, with an optional
Ollama-ranked narrative layer on top (`app/ai/trading_assistant.py`) — same
"app computes the facts, Ollama explains them" split used throughout.

## AI Assistant

A dashboard combining a ForexFactory news panel, a best-markets scanner
across forex/futures/crypto, and a chat panel (`app/ai/t58_strategy_engine.py`,
`app/ai/news_forexfactory.py`, `app/ai/market_scanner.py`,
`app/ai/trading_assistant.py`), reading live bars from your MT5 connection
(falling back to Alpaca for crypto if MT5 isn't connected). This app has no
fundamentals/news data source of its own beyond ForexFactory's calendar, so
treat the macro bias it surfaces as a starting point, not a complete picture
— the panel says so plainly in its own `macro_note` field.

## Resources

A curated beginner's guide to trading fundamentals — market structure,
liquidity, supply & demand, and entry models — for anyone using this app
who wants a running start before backtesting their first strategy. Purely
educational, no engine dependency.

## AI Assist (optional, local Ollama)

Several of the tools above can optionally call a local
[Ollama](https://ollama.com) model — Full Pipeline's GA search (candidate
parameter suggestions once per generation), the Research Agent, Generate
Strategies, and Options Outlook/AI Assistant's narrative layer. Every
GA-relevant suggestion still passes through the exact same backtest →
prop-simulation → Monte Carlo pipeline as any other candidate: the model
only ever proposes numbers for a strategy's already-discovered tunable
parameters, never writes or edits strategy code, and can never displace a
genuinely better candidate the GA already found.

**Setup:**

1. Install Ollama and pull a model — free, runs entirely on your own
   machine: **[ollama.com/download](https://ollama.com/download)**, then
   `ollama pull llama3.1` (or any model you prefer).
2. Wherever a page has an **AI Assist** section, check **Enable**, and hit
   **Test Connection** to confirm it's reachable.
3. Use the feature as normal — nothing else changes.

Off by default, everywhere. Leaving it disabled runs every feature above
exactly as if AI Assist didn't exist; an unreachable/misconfigured Ollama
degrades the same way everywhere — a failed attempt or two, logged once,
and the feature quietly continues without it. An optional API key field
supports a remote/proxied Ollama endpoint instead of a local install.

**Research Agent, specifically**, layers two more pieces on top:

```
RESEARCH LIBRARY (research/ papers, notes)     T58 RESEARCH MEMORY
        │                                (every strategy ever tested,
        └──────────────┬─────────────────  SQLite + semantic index)
                        ↓                              │
             local Ollama embeddings ───────────────────
                        ↓
                 local vector store (data/ai_memory/*.json)
                        ↓
              T58 AI RESEARCH AGENT (Ollama)
     proposes which read-only tool to call next, reasons over
     the result, repeats up to N steps, then answers
                        ↓
    run_backtest / run_prop_simulation / run_monte_carlo /
    run_walk_forward / run_regime_analysis / run_parameter_sensitivity /
    run_cost_stress / compare_strategies / search_research / search_experiments
```

The one rule that matters: the quantitative engine is the authority, never
the model's own judgment — there is no `edit_strategy_code` tool. The agent
can recommend a next step in plain language, but turning that into a tested
strategy still goes through Iterative Refinement/Quick Optimize/Full
Pipeline like any human-typed idea would. Plain keyword search over
`research/` works with zero setup; `ollama pull nomic-embed-text` (or
another embedding model) plus **EMBED RESEARCH LIBRARY** on the Research
Agent page blends in real semantic search. Fine-tuning a model on
accumulated experiments is intentionally not built — RAG plus the growing
Research Memory table gets most of the value without the training
infrastructure.

## PineScript support (subset)

Supported: `open/high/low/close/hl2/hlc3/ohlc4`, `input.int`/`input.float`,
`ta.sma`/`ta.ema`/`ta.wma`/`ta.rsi`, `ta.crossover`/`ta.crossunder`, boolean
rule variables (`and`/`or`/comparisons`), `strategy.entry(..., when=...)` and
`strategy.close(..., when=...)` inline or inside an `if` block, and
`// T58_SL_PIPS=20` / `// T58_TP_PIPS=40` directive comments (Pine's own
`strategy.exit()` uses absolute price offsets, not a portable "pips"
concept). Not supported: custom functions, arrays/matrices,
`security()`/multi-timeframe requests, plotting/alerts, and any `ta.*`
function beyond the list above. `input.int()`/`input.float()` values work as
a `ta.*` length argument but not as a general numeric constant inside a
comparison expression — use a literal there instead.

## MQL5 support (subset)

Supported: direct-value `iMA(...)` (`MODE_SMA`/`MODE_EMA`/`MODE_LWMA`) and
`iRSI(...)` calls, C-style boolean conditions, `if (cond) { ... }` (Allman
and K&R) plus single-statement `if (cond) stmt;`,
`trade.Buy`/`trade.Sell`/`OrderSend(...)` for entries,
`trade.PositionClose`/`OrderClose` for exits, and the same
`T58_SL_PIPS`/`T58_TP_PIPS` directive comments. Not supported:
`CopyBuffer()`-based indicator handles, custom indicators, arrays/structs,
multi-symbol/multi-timeframe logic, any indicator beyond iMA/iRSI, and
trailing stops. `iMA()`'s `shift` argument is parsed but not used — there's
no "previous bar's MA" here, so a true crossover *event* isn't expressible
(only sustained-state comparisons). (The Manual Builder's own trailing
stop/break-even support is not subject to any of this.)

## Engines

- **Backtest engine** (`app/backtest/`): bar-by-bar execution with intrabar
  stop-loss/take-profit checks — ATR-based dynamic distances, a ratcheting
  ATR trailing stop, break-even management — producing a trade list, equity
  curve, and full statistics (returns, win/loss, risk, risk-adjusted
  ratios). Also home to `run_holdout_comparison()` (a single chronological
  in-sample/OOS split, distinct from fold-based Walk-Forward Optimization)
  and `app/strategy/lookahead_check.py`.
- **Prop-firm simulator** (`app/prop/simulator.py`): walks a chronological
  trade P&L sequence through the configured rules, determining pass/fail,
  days to pass, payout events, and failure cause — the same function
  powers both the single historical run and every Monte Carlo iteration.
- **Monte Carlo engine** (`app/monte_carlo/engine.py`): resamples the
  historical trade sequence (bootstrap/shuffle/block-bootstrap, plus
  optional slippage stress) thousands of times and re-runs the prop
  simulator on each — pass probability, first-payout probability, speed,
  financial outcome, and risk distributions.
- **Optimization engines** (`app/optimize/`): shared GA operators
  (crossover/mutation/tournament selection/elitism/random immigrants)
  powering Iterative Refinement, Search Lab's Stage 2, the Walk-Forward-
  Aware GA, and (via non-dominated sorting) Multi-Objective Optimization.
  `refinement.py` also owns the cost-stress penalty shared across all three.
- **Report generator** (`app/reports/generator.py` + `charts.py`, plus
  `refinement_report.py` and `validation_reports.py`): JSON, a flattened
  summary CSV, a trades CSV, and a self-contained HTML report with inline
  SVG charts (no plotting dependency) — equity curves, Monte Carlo
  histograms, chained OOS equity curves, Pareto-front convergence,
  sensitivity heatmaps — plus execution-integrity warnings and, for Full
  Pipeline, the verdict and winning parameters as banners/tables up top.

## CLI reference

Every feature that predates the web-app rewrite is also available headless
via `python -m app.main --cli <flag> ...` (newer additions — Speed Run,
Quick Optimize, Evolution Lab, Regime Survival Matrix, Family Diversity,
Generate Strategies, Payout Probability, Quant Lab, Options Outlook, the AI
Assistant — are web/desktop-GUI-only for now, no CLI flag). The base flags
(`--csv`, `--output`, `--sims`) apply throughout; run
`python -m app.main --help` for the full, current list with defaults — this
table is a summary, not the source of truth.

| Flag | Runs |
|---|---|
| `--refine` (+ `--refine-population/-generations/-metric/-seed`) | Iterative Refinement |
| `--search` (+ `--search-mode/-family/-strategy-file/-grid-points/-max-candidates/-workers/-min-trades/-min-profit-factor/-stage1-top-n/-stage2-top-n/-ga-population/-ga-generations/-full-mc-sims/-walk-forward-folds/-robustness-neighbors/-metric/-seed/-db/-no-promote`) | Search Lab, Stages 1-5 |
| `--wfo` (+ `--wfo-folds/-window-mode/-train-frac/-population/-generations/-metric/-seed`) | Walk-Forward Optimization |
| `--cpcv` (+ `--cpcv-groups/-test-groups/-metric/-max-paths`) | Combinatorial Purged Cross-Validation |
| `--pbo` (+ `--pbo-groups/-test-groups/-metric/-max-paths/-candidates/-seed`) | Probability of Backtest Overfitting |
| `--sensitivity` (+ `--sensitivity-metric/-pct-range/-steps/-heatmap`) | Parameter Sensitivity |
| `--portfolio` (+ `--portfolio-csv` [repeatable, 2+ required] `/-balance/-correlation-strength`) | Multi-Asset Portfolio |
| `--multi-objective` (+ `--mo-objectives/-population/-generations/-seed`) | Multi-Objective Optimization |
| `--wfga` (+ `--wfga-folds/-window-mode/-population/-generations/-metric/-seed`) | Walk-Forward-Aware GA |
| `--ensemble` (+ `--ensemble-strategy` [repeatable, 2+ required] `/-mode/-min-agreement/-balance/-correlation-strength`) | Multi-strategy ensemble, blend or vote |
| `--full-pipeline` (+ `--fp-folds/-window-mode/-population/-generations/-metric/-final-mc-sims/-seed/-no-save-to-library`) | Full Pipeline: baseline → GA → re-validated report → OOS/holdout checks → verdict |

`--refine` and `--search` additionally accept `--refine-no-cost-stress` /
`--refine-cost-stress-multiplier` / `--refine-cost-stress-weight` and
`--search-no-cost-stress` / `--search-cost-stress-multiplier` /
`--search-cost-stress-weight` (cost-stress fitness, on by default). `--search`
also accepts `--pair-csv <path>` to merge in a second instrument so the
`stat_pairs` family can be searched. Plain `--cli` accepts
`--adaptive-risk-rules '<json>'`.

Each of the nine `--wfo`/`--cpcv`/`--pbo`/`--sensitivity`/`--portfolio`/
`--multi-objective`/`--wfga`/`--ensemble`/`--full-pipeline` runs is mutually
exclusive with the others and with `--search`/plain `--cli`; pick one per
invocation. All write reports under `--output` (default `reports/`).

## MVP scope decisions

- PineScript/MQL5 support a real, tested *subset* of each language rather
  than a full parser/runtime — anything unsupported fails loudly instead of
  producing a silently inaccurate backtest.
- Report export is JSON + CSV + HTML instead of PDF, to avoid a heavy
  rendering dependency — any browser prints `report.html` to PDF for free.
- The desktop GUI is built with Tkinter (Python's standard library) — zero
  extra GUI-framework install burden, packages into a Windows `.exe` with
  PyInstaller without code changes. Entry point: repo-root `run_app.py`.
- Mobile access is a web app (Flask + installable PWA) rather than a native
  iOS/Android build — reuses the engine with zero duplication, no App
  Store/Play Store submission; needs the Flask server running somewhere
  reachable (your own Wi-Fi, or Tailscale for anywhere-access).
- One open position at a time (consistent with the standardized long/flat/
  short signal model); no partial fills or multi-leg positions.
- Multi-timeframe analysis is an as-of merge onto the finest selected
  timeframe rather than fully separate per-timeframe backtests — keeps
  every strategy source working against one dataframe unchanged.
- Multi-Asset Portfolio uses a static (whole-window) correlation pass and
  chronological trade-close merging rather than a margin-constrained
  concurrent-position engine — see `app/portfolio/portfolio.py`'s docstring.
- Evolution Lab candidates are Manual Strategy Builder configs only — it
  does not mutate uploaded Python/PineScript/MQL5 files.
- Deploy Live's live order placement is a deliberate stub — connection
  management is built and usable, actual funded-account trading is a
  separate, later decision (see **Deployment** above).
- Forward Test and Deploy Live are desktop-only (MT5 terminal + Windows
  dependency, and — for Deploy Live — this server having no authentication
  layer yet). Everything else has full web/desktop parity except a PBO
  candidate-pool picker and a 2D sensitivity-heatmap picker, both awaiting
  a small UI on top of an already-built backend.

## Project layout

```
T58-Quant-Algo-Backtester/
├── run_app.py                  # PyInstaller entry point (desktop) -- must stay at repo root
├── run_web.py                  # PyInstaller entry point (web/phone) -- must stay at repo root
├── config/                     # pyproject.toml, requirements.txt
├── docs/                       # WEB_PARITY_ROADMAP.md, ENGINE_PARITY_CHECKLIST.md
├── app/
│   ├── main.py                 # entry point (GUI, or --cli headless -- see CLI reference)
│   ├── ui/
│   │   ├── main_window.py      # Tkinter desktop GUI -- every tab described above
│   │   └── condition_builder.py  # visual condition-row widget used by the Manual Builder
│   ├── web/                    # Flask mobile/web app (same engine, new front end)
│   │   ├── server.py
│   │   ├── quant_lab_routes.py / options_outlook_routes.py / ai_assistant_routes.py  # Blueprints
│   │   ├── live_market.py / launcher.py / network_info.py
│   │   ├── templates/          # one page per feature, plus shared partials (_sidebar.html, etc.)
│   │   └── static/             # manifest.json, service worker, icons, theme.css
│   ├── data/                   # importer, storage, multi_timeframe, pairs, alpaca_source
│   ├── strategy/                # manual / python / pinescript / mql5 adapters
│   │   ├── indicators.py / expr.py / manual.py / python.py / pinescript.py / mql5.py
│   │   ├── mtf.py               # safe "last fully-closed HTF bar" helper
│   │   ├── lookahead_check.py   # generic, code-agnostic lookahead-bias detector
│   │   ├── translator.py        # Manual config -> PineScript v5 / MQL5
│   │   ├── auto_regime_selector.py
│   │   └── library.py           # persistent Strategy Library
│   ├── backtest/                 # execution engine, risk sizing, statistics, adaptive_risk.py
│   ├── ensemble/ensemble.py       # multi-strategy ensembles (blend or vote)
│   ├── orchestration/             # full_pipeline.py, speed_run.py, quick_optimize.py, resource_guard.py
│   ├── evolution/                 # Evolution Lab: engine.py, checkpoint.py, prop_fitness.py, knowledge_graph.py
│   ├── ai/                        # Ollama-backed features (all off by default)
│   │   ├── ollama_client.py / ollama_settings.py
│   │   ├── strategy_generator.py / research_library.py / vector_store.py
│   │   ├── experiment_memory.py / research_agent.py / research_loop.py
│   │   ├── options_outlook.py / trading_assistant.py / t58_strategy_engine.py
│   │   └── news_forexfactory.py / market_scanner.py / market_intelligence.py
│   ├── optimize/                  # parameter_space.py, refinement.py, multi_objective.py, walkforward_ga.py
│   ├── validation/                # walk_forward_opt.py, cpcv.py, sensitivity.py
│   ├── portfolio/                 # portfolio.py (Multi-Asset), composer.py (Automated Composer)
│   ├── search/                    # Search Lab: strategy_space.py, batch_runner.py, robustness.py, results_db.py
│   ├── quant_lab/                  # the 12 standalone analysis tools
│   ├── monitoring/strategy_health.py
│   ├── forward_test/                # MT5 demo forward-testing
│   ├── live_deploy/                  # live-account connection management (order placement stubbed)
│   ├── lab/strategy_lab.py
│   ├── scoring/t58_scorecard.py
│   ├── prop/simulator.py          # prop-firm rules + account simulator, survival_engine.py
│   ├── monte_carlo/engine.py
│   └── reports/                   # generator.py, refinement_report.py, validation_reports.py, charts.py
├── data/
│   ├── examples/                  # sample OHLCV dataset for immediate testing
│   └── raw/                       # bundled datasets for common forex pairs & indices
├── strategies/                    # persistent Strategy Library storage (python/pinescript/mql5 + metadata)
├── tests/                         # pytest unit tests for every engine
└── .github/workflows/
    ├── build.yml                  # runs pytest on push/PR
    ├── build-exe.yml              # builds & uploads the Windows .exe (desktop)
    └── build-web-exe.yml          # builds & uploads the Windows .exe (web/phone)
```

## Tests

```bash
pytest -q tests
```

## Disclaimer

Simulated results are estimates derived from historical data and
resampling. Past performance and simulated outcomes do not guarantee future
results.
