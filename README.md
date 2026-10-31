<p align="center">
  <img src="assets/logo/t58-logo-horizontal-dark.png" alt="T58 Quant Algo Backtester" width="480" />
</p>

<p align="center">
  <sub>STRATEGY RESEARCH &nbsp;/&nbsp; VALIDATION &nbsp;/&nbsp; ROBUSTNESS &nbsp;/&nbsp; PROP SIMULATION</sub>
</p>

<img width="1280" height="420" alt="T58 Quant Algo Backtester — dashboard preview" src="assets/images/readme-banner.png" />

<div align="center">


# T58 Trading — Quant Algo Backtester

</div>

---

Full tutorial: https://drive.google.com/file/d/1pO-jS8W7A2mh8HVCpxZ0Sd2h2AHaUeW_/view?usp=drivesdk

Web App: https://script.google.com/macros/s/AKfycbxOtM8dHxQnR-TsLNsgHxr-vYElG1he4XpHNPvJSsNWq1bP__pjy1PRfmBo-cbaIs7wFQ/exec

---

A prop-firm-first strategy lab. Bring a strategy (Manual Builder, Python, PineScript or MQL5), test it against real
prop-firm rules, search for a version that holds up out-of-sample, and get one report that answers:

> **"If I trade this strategy under these prop-firm rules, what is the probability that I pass the evaluation and reach my first payout?"**

```
Market data + Strategy + Risk + Prop rules
  -> Backtest -> Prop-firm simulation -> Monte Carlo -> Probability of success -> Report
```

It runs as a Windows desktop app (.exe), a local Python app on any OS, or a web app you can open from your phone.
Everything stays on your computer.

## Contents

[Run it](#run-it) · [Workflow](#workflow) · [Discovery Engine](#discovery-engine) · [Verdicts and scoring](#verdicts-and-scoring) ·
[Monte Carlo](#monte-carlo) · [Champion and Deployment](#champion-and-deployment) · [Library and research](#library-and-research) ·
[Account, ranks and themes](#account-ranks-and-themes) · [Settings and defaults](#settings-and-defaults) · [Strategy formats](#strategy-formats) · [Tests](#tests)

## Run it

**Windows `.exe` (no Python):** in GitHub go to **Actions → desktop-cross-platform → latest run** and download
**T58-Quant-Algo-Backtester-Windows**. Extract the inner zip into a short folder such as `C:\T58\` and double-click
`T58-Quant-Algo-Backtester.exe` (keep `_internal` beside it). Market data is a separate download (**Actions → build-data-pack →
T58-Market-Data**); extract it next to the `.exe`. Tagged releases (`v*`) attach the app zip, data zip and `.sha256` files directly.

**Web app, any OS (Python 3.10+):**

| OS | Start it |
| --- | --- |
| Windows | double-click `launchers\run_web.bat` |
| macOS | double-click `launchers/run_web.command` (first time: right-click, Open; if it will not run, `chmod +x launchers/run_web.command launchers/run_web.sh`) |
| Linux | `bash launchers/run_web.sh` |
| Manual | `pip install -r config/requirements.txt` then `python launchers/run_web.py` |

The first run creates a private `.venv`. If port 5000 is busy the app picks the next free one and prints the address and a QR code.
For your phone, scan the QR code on the same Wi-Fi, or use [Tailscale](https://tailscale.com) away from home
([`HOW TO OPEN ON YOUR PHONE.md`](HOW%20TO%20OPEN%20ON%20YOUR%20PHONE.md)). User data lives in `%LOCALAPPDATA%` (Windows),
`~/Library/Application Support` (macOS) or `~/.local/share` (Linux) when the install folder is read-only.

**Desktop GUI / headless:** `python -m app.main`, or `python -m app.main --cli --csv data/raw/EURUSD/EURUSD_5M_sample.csv --sims 10000`
(`--help` lists every flag). Reports land in `reports/report.{json,html}`; the HTML is self-contained and prints to PDF.

`python scripts/web_smoke_test.py` starts the real server and checks every main page. The **web-cross-platform** and
**desktop-cross-platform** Actions run the tests and the smoke test on Windows, macOS and Linux.

New buyers: [docs/buyer/INSTALL.md](docs/buyer/INSTALL.md), then [docs/buyer/QUICKSTART.md](docs/buyer/QUICKSTART.md).

## Workflow

1. **Data** — CSV/TSV/Parquet/`.zip`/`.7z` with automatic column mapping and gap/duplicate checks, or Alpaca for US equities and crypto.
   Datasets live in `data/raw`; drag a folder onto the Data Center to import it. The full history ships as the separate
   `T58-Market-Data.zip` (redistribution is licensed); the app zip carries only a small sample.
2. **Strategy** — Manual Builder (no-code: price, EMA/SMA/WMA, VWAP, RSI, MACD, ATR, Bollinger, market-structure concepts; fixed/ATR stops,
   trailing, break-even, time exits), Python (`generate_signals(df)`), PineScript or MQL5 (tested subsets, see below). Every strategy can be
   checked for **lookahead bias**: signals are re-computed on truncated data and the first bar that changes is reported. The Strategy Library
   keeps status tags and each strategy's backtest and search history.
3. **Prop rules** — account size, target, daily loss, max drawdown (trailing/static, intrabar/EOD), consistency, payout rules. Presets for FTMO,
   Apex, TopStep, The5%ers, FundedNext and Lucid carry a verified date; check it before trading real money.
4. **Risk and execution** — fixed-$ or %-equity risk, commission, slippage, spread, pip size.
5. **Run** — backtest, prop simulation, Monte Carlo and report in one click.

| Stage | Tools |
|---|---|
| **Create** | Strategy Discovery Engine (pick a market and prop firm; Speed Run, Search Lab, Evolution Lab and the deep Forge gauntlet are its presets), plain-English / Python / PineScript / MQL5 import, the no-code Manual builder, Generate Strategies (AI), Research Agent (read-only engine calls, never invented numbers) |
| **Test** | Run & Report, Payout Probability (eval pass, funded, payout 1, 2 ... with an optional scaling stress test), Prop-Firm Recommender (ranks every preset against your trade sequence) |
| **Optimize** | Search Lab (5-stage funnel to a leaderboard), Iterative Refinement (GA), Full Pipeline, Optimize (Quick Optimize: searches the most recent 150k bars by default, adjustable under Advanced), Multi-Objective (NSGA-II Pareto front), Evolution Lab (unattended generate, filter, validate, mutate, ranked by PROP FITNESS) |
| **Validate** | One page, eight checks you tick and run together or alone: Monte Carlo, CPCV, PBO, Walk-Forward, Walk-Forward-Aware GA, Parameter Sensitivity, Parameter Robustness, Regime Survival |
| **Champion** | Multi-Asset Portfolio, Multi-Strategy Ensemble, Family Diversity, Automated Portfolio Composer |
| **Deployment** | Forward Test (MT5 demo), Webhook Relay (TradingView), Overnight Autopilot, Auto-Retune, Deploy Live, Live PnL pill, Live Market monitor |
| **Quant Lab** | Strategy Translator (to PineScript v5 / MQL5), Auto Regime Selector, Health / Drift Monitor, Pairs, Options Pricing, Order Book Simulator, Sentiment Correlation, Markowitz, Volatility Surface, Factor Model |

**Optimizer defaults.** Every GA form defaults to the same fitness, `pass_gradient`; Speed Run uses `edge_lb_r` (lower bound of mean R after 2x costs).
The optimizers share one persistent worker pool, memoize backtests by effective parameter values, and score plateau-robust finalists by perturbing the
winner's neighbours. After a GA run, **Map the winner's neighbourhood** opens Parameter Robustness, which shades the neighbouring parameter grid and
says in plain words whether the winner sits on a plateau or a lone spike.

**Deployment notes.** Deploy Live connects MT5, cTrader, DXtrade, TradeLocker or Tradovate with enforced guardrails (news blackout, no weekend
holds, no hedging, size cap, daily-loss and drawdown halts); start on a demo. Forward Test and Overnight Autopilot need a running MT5 terminal
(desktop only). Webhook Relay needs a public https address and a TradingView plan with webhooks; it feeds a T58 paper account and optionally a demo broker.

## Discovery Engine

Finds new strategies without hand-picking parameters. Effort budgets are **5,000+ (Quick), 10,000+ (Standard) and 15,000+ (Deep)** evaluated
configurations. Candidates are drawn from paper-sourced families, the full indicator pool (about 300 indicators as trend / revert / slope rules) and
the graveyard (known-dead neighbourhoods are skipped). Every configuration is optimized, then verified with real-engine backtests plus a lookahead
test: a prefix-invariance check on the signals and a second probe on an early slice of the history, so a signal that peeks at the future is rejected.
A finalist must also clear **Full Pipeline's own preflight** (the same sizing resolution on the full-size contract, at least 100 trades and at most 25% of
signals skipped for sizing) before it is called a winner, so every winner can be run through Full Pipeline. The trial count feeds the Deflated Sharpe
correction. Details: [docs/DISCOVERY_ENGINE.md](docs/DISCOVERY_ENGINE.md).

- **Strategy Graveyard** — searchable by market and text, with **See all** and **Revive**, which re-optimizes a dead strategy on a chosen market.
  New installs ship with 1,100+ measured failures (rebuild with `scripts/build_graveyard_seed.py`, extend with `scripts/expand_graveyard_seed.py`,
  fold in your own with `scripts/export_graveyard_seed.py`).
- **Trial ledger** (`/trial-ledger`) — a plain-language record of how many things have been tried, which is what multiple-testing corrections need.

## Verdicts and scoring

## The verdict ladder (the headline everywhere)

READY used to read as "buy the evaluation". It no longer does. Every page headlines one of five stages and shows the legacy
READY / MARGINAL / NOT READY label only as detail:

| Stage | Meaning | Reached by |
|---|---|---|
| REJECT | A hard gate failed, the pooled edge test FAILED, or the EV test says the fee is not worth it | backtest |
| LEAD | No hard failure, but the edge is not proven (thin trades, lower bound not above zero) | backtest |
| FORWARD-TEST CANDIDATE | Cleared the backtest gates and the EV test: permission to paper-trade under the SPRT monitor, nothing more | backtest |
| EVALUATE | The forward test accepted H1 (SPRT): buy the cheapest evaluation | recorded forward-test result only |
| FUNDED | Evaluation passed and a payout was received | recorded live facts only |

FORWARD-TEST CANDIDATE is the realistic output of a search. A backtest alone can never produce EVALUATE or FUNDED.

On the Full Pipeline result, a **verdict waterfall** lists each step in order (score, gate, cap, advisory) so you can see exactly what moved the verdict,
and what single change would move it.

**Is there an edge? The pooled test.** One strategy on one market at 7 trades a month cannot prove a +0.10R edge in six years; that is a power problem,
not strictness. So the edge question is asked pooled: the same fixed parameters on 8-12 markets, R after 2x costs, one day-block bootstrap over
trading days (`app/validation/pooled_edge.py`). Search Lab ranks on `edge_lb_r`, needs 100+ Stage-1 trades, skips candidates whose round-trip cost exceeds
0.15R and prints a random-entry control; `app/validation/pipeline_controls.py` shows on the Dashboard that a zero-edge stream almost never passes and a planted
+0.20-0.30R edge usually does. `python scripts/bootstrap_data.py` rebuilds the data manifest and calendars and warns about missing instrument specs or markets.

### How READY and the T58 score work

READY means every hard gate passes **and** the T58 score is 75 or more (85+ earns the Elite label). Gates only ever lower a verdict. Click any verdict pill
on the Dashboard or Champion page for a plain-English reason. Source: `app/scoring/t58_scorecard.py`, `app/orchestration/full_pipeline.py`, `app/orchestration/audit_layers.py`.

### The T58 score (weights sum to 100)

| Score component | Weight | Measures |
|---|---:|---|
| Pass probability | 10 | Per-attempt Monte Carlo pass rate, trade-sample-aware lower bound |
| First payout (pass AND paid) | 8 | Same Monte Carlo, lower bound |
| Risk of ruin (inverted) | 6 | Evaluation-stage breach only |
| Funded survival (inverted) | 6 | Funded-stage breach only |
| Attempt replay pass | 15 | The real engine re-run from many real start dates (independent of the Monte Carlo) |
| Expectancy lower bound | 15 | Lower bound of average R per trade after 2x costs: 0R scores 0, 0.15R scores 100 |
| Walk-forward stability | 15 | Test metric / train metric, capped at 1.0 |
| Drawdown | 8 | Max drawdown as a share of the firm's drawdown limit |
| Monte Carlo robustness | 5 | Seed-to-seed spread of the pass rate |
| Parameter stability | 5 | Neighbour-parameter robustness |
| Parsimony | 4 | Fewer degrees of freedom scores higher |
| CPCV / PBO (supporting) | 3 | Out-of-sample retention across CPCV paths |

A component's share of the score is its weight / 100.

The score is the weighted average over the components that were **measured** (a missing check is never scored as a
zero, and thin coverage caps the verdict instead). Tiers: **85+ Elite, 75+ Strong, 65+ Promising,
below 65 Reject.** Elite and Strong are READY (if every gate also passes), Promising is MARGINAL, Reject is NOT READY.

**T58 conformance is a badge, not a score term.** It measures how much of the entry logic is written in Market
Structure / Liquidity / Supply & Demand terms. That is a *style*, not survival, and prop firms do not score
vocabulary, so it is shown beside the score and **never** affects the score, the weight coverage or READY.

The four Monte Carlo components (pass, paid, ruin, funded survival) are capped at 30 of the 100 points because they come
from one resampled trade list; the other 70 points are evidence independent of it (replay, expectancy lower bound,

### The gates (defaults)

| Gate | Effect |
|---|---|
| Lookahead bias detected (or the check crashed) | NOT READY |
| Risk of ruin above 20% | NOT READY |
| Walk-forward not stable, CPCV not robust, ICIR fail, deflated Sharpe (PSR 0.95) fail, PBO above 0.5 | NOT READY |
| Per-attempt pass lower bound below 70% | NOT READY |
| Holdout (needs **30+ trades**) loses money **and** the upper 95% bound on per-trade expectancy is still below zero | NOT READY |
| Holdout loses money but the loss is not statistically convincing | Capped at MARGINAL |
| Fewer than **300** trades (`min_trades_for_ready`; the 100-trade preflight floor only decides whether the run is analysed at all); scorecard coverage below 70%; a core component missing | Capped at MARGINAL |
| Pass-and-paid lower bound below 35%; 30-day payout rate below 35% | Capped at MARGINAL |
| Random-entry null p above 0.10 | Capped at MARGINAL |
| **2× costs keep less than 25% of base net profit** (was: net profit of $1 passed) | Capped at MARGINAL |
| Seed spread above 10 points | Capped at MARGINAL |
| Attempt replay (**48** real start dates) disagrees with Monte Carlo: the two 95% intervals do not overlap **and** the gap is 10+ points | Capped at MARGINAL |
| No clean holdout, unverified firm rules, failed data gate, search-bias haircut | Capped at MARGINAL |

### What READY effectively asks for

Real scorecard runs at the gate floors (every other component set to the value shown, walk-forward 100):

| Pass lower bound | Paid lower bound | Ruin | Other components | Score | Verdict |
|---:|---:|---:|---:|---:|---|
| 70% | 35% | 20% | 80 | 72.9 | MARGINAL (Promising): gate floors alone are not enough |
| 80% | 45% | 10% | 85 | 80.0 | READY (Strong), if every gate also passes |
| 85% | 65% | 8% | 80 | 83.0 | READY (Strong), if every gate also passes |
| 90% | 60% | 5% | 90 | 87.1 | READY (Elite), if every gate also passes |

So READY now asks for the gates plus a score of 75: roughly 80-85% per-attempt pass lower bound, 45-65% paid lower
bound and ruin under about 10%, with decent (not perfect) secondary components. Elite (85+) is a label for

### Expected value beside READY

The EV verdict (HIGH CONFIDENCE / FORWARD-TEST CANDIDATE / NOT WORTH A FEE / INSUFFICIENT EVIDENCE) answers the
economic question: `EV = p × q × V − p × A − c`. For firms that charge a **monthly subscription** (Topstep's Trading
Combine, $49 per month) `c` is the monthly price × the months the evaluation is expected to take (median time to pass,
rounded up to whole 30-day periods), so slow passers are no longer priced as if they paid for one month.

## Monte Carlo

One method is used everywhere: **day-block bootstrap with 1-day blocks**. Whole session days are resampled, so every trade of a day stays together and the
per-day rules (daily loss limit, consistency, winning-day counts) bind as often as in real life.

It was chosen by test, not by taste. Trades were generated from processes whose true per-attempt pass rate is known (i.i.d. wins, hot/cold regimes, a shared
daily shock, streaky wins); each method estimated that rate from about 9 months of history. Mean absolute error in pass-rate points (first run, 8 histories each):
1-day blocks 7.9-10.9 versus 3-day blocks 12.5-15.1 and 5-day blocks 14.2-17.8 on the i.i.d. and daily-shock processes, with 1-day blocks about equal to the plain
i.i.d. bootstrap and no worse on bias. On the regime and streaky processes every method failed about equally: one short history cannot reveal regimes it did not
contain, which is why the score also leans on replay from real start dates. The experiment is `python scripts/mc_method_calibration.py 30 out.csv` (slow: about 20 minutes per process); rerun it before changing the default. A 30-history run was still finishing when this was written, so the numbers above come from the 8-history run.

## Champion and Deployment

Cost-stress-adjusted fitness (every GA candidate re-backtested at inflated spread, slippage and commission) and a declarative adaptive-risk layer
(de-risk after N losses, cut size on a bad day, coast near the target) apply across every optimizer. The Dashboard's equity curve lets you select several
strategies to compare, and you can promote a strategy from there. See the Deployment table above for the live tools.

## Library and research

- **Library** (Playground): your book shelf with progress, an in-app PDF reader for PDFs you add, a list of free legal books to read online (statistics,
  machine learning, forecasting, optimization, probability), a Bible reader (KJV offline, ESV with your own free key) and focus sessions.
- **Research papers**: `research/NN_*/Reading_List.md` lists 225 papers in 12 topic folders (momentum, validation, volatility, sizing, execution, machine learning,
  multiple testing, regimes, trend and mean reversion, prop-firm behaviour), each with a one-line role and links. Drop PDFs anywhere under `research/` and
  the optional AI assist reads them when suggesting parameters (keyword matching only, nothing leaves your machine). Links were compiled without being opened:
  run `python scripts/check_paper_links.py` to find dead ones, or add `--download` to save the arXiv PDFs. See [research/README.md](research/README.md).
- **Global Events**: a plain-language session brief, the economic calendar and an optional X feed (Bearer token under API Keys, paid X plan; see
  [docs/GLOBAL_EVENTS.md](docs/GLOBAL_EVENTS.md)).
- **Options Outlook** (Black-Scholes candidates with an optional AI narrative), the **AI Assistant** dashboard (news, scanner, chat) and a beginner **Resources** guide.

## Account, ranks and themes

**Account Settings** holds your research profile: a rank and title earned from XP, collectible metallic badges, a three-badge showcase, shareable
achievement cards and downloadable certificates (SVG or PNG). The **Achievements** page filters by status, difficulty and category. Six scenes unlock
through achievements (Obsidian Gold, Sakura Reverie, Abyssal Blue, Endless Sands, Coastal Italy, Bioluminescence; locked tiles show what to do), five
whole-interface themes and six **Workspace presets** (Obsidian, Cyberpunk, Nordic, Evergreen, Sovereign, Midnight -- each an earned reward) can be switched on,
and your theme, preset, accent and background sync with the account. The Sakura, Abyssal and other animated scenes follow one shared clock, so the
Sakura sky moves through sunrise, morning, noon, afternoon, sunset, evening and night continuously, whichever page you are on. The bell beside your picture is the notification inbox; a weekly digest arrives on Sundays.
Personalization also offers the original mountains, the Nordic Kingdom fjord (available as soon as you are signed in) and four seasonal scenes, and an optional Bible verse bar (off by default). There are no online
leaderboards: nothing in the app talks to a T58 server for that.

## Settings and defaults

- **AI Assist** is off by default. Ollama (local, free) is the default provider; **API Keys → AI provider** switches every feature to Claude or OpenAI with your key
  (it falls back to Ollama if no key is saved). The model only proposes values for already-discovered tunable parameters, never code, and every suggestion goes
  through the same backtest, prop-sim and Monte Carlo pipeline.
- **Costs are never zero by default**: realistic spread and slippage plus $3 per trade commission (`app/backtest/risk.py`). You can set any cost to 0.
- **One account size**: RiskConfig and prop rules both default to $100,000.
- **Presets with no verified stamp** are labelled unverified and cannot reach READY (`app/prop/presets.py`).
- **Licensing**: master key (offline, owner), Google Sheet keys (customers) or an optional license server. See [docs/LICENSING.md](docs/LICENSING.md).
- **Speed switches** (environment variables): `T58_MC_VECTORIZE=0` (Monte Carlo falls back to the per-path loop), `T58_LOOKAHEAD_DISK_CACHE=0`,
  `T58_CACHE_DIR`, `T58_MODULE_CACHE=0` (a strategy with mutable module state sets `T58_STATEFUL = True`).

## Strategy formats

**PineScript (subset):** `open/high/low/close`, `input.int`/`input.float`, `ta.sma`/`ta.ema`/`ta.wma`/`ta.rsi`, `ta.crossover`/`ta.crossunder`,
`strategy.entry`/`strategy.close` with `when=`, and `// T58_SL_PIPS=20` / `// T58_TP_PIPS=40`. Not supported: custom functions, arrays, `security()`, plotting, alerts.

**MQL5 (subset):** direct-value `iMA()`/`iRSI()`, C-style conditions, `trade.Buy`/`trade.Sell`/`OrderSend()`, the same SL/TP directives.
Not supported: `CopyBuffer()` handles, custom indicators, multi-symbol logic, trailing stops.

Both fail loudly on anything unsupported instead of producing a silently wrong backtest.

## Tests

```bash
pytest -q tests
```

Brand colors: `assets/brand/palette.md`.

## Disclaimer

Simulated results are estimates derived from historical data and resampling. Past performance and simulated outcomes do not guarantee future results.
