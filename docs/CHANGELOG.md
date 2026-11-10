# Changelog

## 2026-11-10 - 30-day prop evaluation research initiative
- **New evaluator** `app/prop/thirty_day.py`: P(pass within 30/60 calendar days of purchase), purchased every weekday, floating drawdown/daily loss, cost stress, regimes, losing streaks and a dependence-aware 95% interval. It never reports eventual/chain MC pass as a 30-day number.
- **Audit fixes**: `sizing_summary.skip_rate` is now per signal (was episodes/bars); `run_attempt_replay` gains opt-in calendar-day, truncation and mid-signal controls (defaults unchanged); rolling-evaluation labels say *trading* days.
- **Research stack**: feasibility screen, opportunity scanner (NQ/ES/GC), frozen-entry exit lab, shared-risk portfolio builder, validation gate with independence audit and forward registry, plus `scripts/run_prop30_research.py` and `scripts/audit_gad_engine.py`. See `docs/PROP30_AUDIT.md`.
- **Result**: the bundled data does not establish a strategy meeting the 65% (95% lower bound) 30-day target; the existing NQ Gap-Aligned Drive passes 0.0% within 30 days.
- **Tests**: `tests/journal` suite skips cleanly when `app/journal` modules are absent.

## 2026-11-09 (second pass) - tabs, demo deploy, trimmed navigation
- **Account Settings** is one tabbed page: Account, Notifications (with the Discord and briefing cards), API Keys, Achievements, Support. The old `/settings/api-keys`, `/support`, `/achievements`, `/settings/notifications` URLs redirect to the matching tab.
- **University** is three separate pages: Courses, Resources, User Manual (rewritten). `/education/knowledge-check` still redirects to the course module.
- **Deploy** has a *T58 demo account* pane (pick a saved strategy, get the TradingView alert) beside the MT5 pane; `/demo-account` redirects there.
- **Removed pages**: Prop Firm Rulebook page and Trader Profile (the rulebook library stays inside the pipeline and discovery engine, whose run notes now record looks and G0 status).
- **Quant Lab menu** lists only Quant Lab and Research Lab. Investing: click a holding for details and a one-click sell (`/investing/api/close`).
- **Global Events map** is readable in light mode; the Endless Sands scene has a walking traveller.
- **Vectorized screen** now covers breakeven and time-stop exits (scalar-parity tested). Resting orders, partial exits, flatten rules and prop state still go to the full engine.
- **Bundled data** is ES, NQ and GC only (`data/data_pack.json` refrozen). The pooled-edge test needs 8 markets and reports INSUFFICIENT until the extended pack is added.
- **TradingView bridge**: a Pine *strategy* template in addition to the settings indicator, plus a Bridge page (`/pine-indicator/bridge`) that decodes and verifies a settings ID.
- **Reports** open with a plain-English summary; validation reports explain each check in a sentence.

## 2026-11-09 - University, starter data, planner and a calmer interface
- **First launch**: the first screen is always the license key + email page. A leftover app password from an earlier install is cleared on first activation, so a fresh install never asks for a password it does not have. Re-activating on an existing install keeps your password.
- **Starter data**: ES, NQ and GC history ship inside the app and are copied into your data folder on first run (each file once; delete one and it stays deleted). GC is a plain dataset: it is not part of the sealed-holdout data pack.
- **Payout Planner** (Test): expected attempts to a first payout, expected fee spend, expected calendar days, the budget for 90% confidence, Lucid 50k vs 100k side by side, and the effect of running two to four evaluations in parallel. Enter a price for any account the fee table does not list.
- **TradingView bridge**: every report carries a settings ID. Add the *T58 Settings* Pine indicator once, paste the ID, and the chart shows the exact configuration of the backtest. A cut-off ID is caught by its checksum.
- **Adaptive risk modes**: Conservative (survival first), Balanced (the previous behaviour, unchanged) and Accelerated (sizes up to 1.25x while every protection is quiet and drops the extra the instant one fires). No mode can override the account's hard daily-loss / drawdown protections.
- **Reports**: tap-to-exclude breakdowns by weekday, killzone and hour, a month-by-year heatmap, and plain-words ablation ("With this rule: -18.3% result, 377 trades removed").
- **Data health cards** with one-click fixes (back-adjust contract rolls, drop duplicates, sort, remove impossible bars). Fixes always save a new file; your original is never touched.
- **Indicator cache**: a shared disk tier keyed by a content hash of the data, so parallel workers reuse each other's indicator series.
- **Dashboard**: an optional, collapsed-by-default live P&L panel, earned by promoting a first strategy to champion and switched on under Account > Personalize.
- **T58 prop-firm demo**: every rule shown in plain words with where you stand, a pass-chance estimate once 15 trades have closed, and a card on the Deploy page.
- **Pages**: "Education (Course)" is now **University**. Search Lab, Evolution Lab, Forge, Speed Run and Multi-Instrument Speed Run are retired as pages (their searches are Strategy Discovery presets). Research Agent and Research Director are one **AI Research** page. Optimize / Validate / Test lead with three steps: strategy, market data, timeframe.
- **CI**: a fast lane (unit + parity, `pytest -m "fast and not slow"`) on every push, and the full suite in four shards (`--shard N/4`) nightly, on tags and on demand.
- **Tests**: the suite no longer needs a committed EURUSD sample: a deterministic stand-in is generated for the session and removed afterwards.



## 2026-11-07

Implements the independent code audit (all findings except the LAN warning, which was deliberately skipped), plus the Replay / Live Monitor / prop-simulator overhaul and a payout-forecast layer on validation.

## Read this first: numbers that move

**Equity-curve timing (audit finding 4).** Equity is now marked when a signal-exit is *scheduled*, not one bar earlier, matching how fills actually happen. Reported **max drawdown and Sharpe shift slightly** on strategies with signal exits; trade lists are unchanged. No existing test pinned the old numbers, so none needed re-baselining; new tests cover the timing and the portfolio-engine mirror. Switch: `_EQUITY_SIGNAL_EXIT_TIMING` in `app/backtest/execution.py`; the same ordering is mirrored bit-for-bit in `portfolio_engine.py`. Re-run saved strategies before comparing to old reports.

## Performance and reliability
- **Skip-memo in the bar loop (finding 1).** Repeated identical skipped entries are cached (keyed on direction, equity, stop/target/trail distances, and the raw price for fallback stops); only *skipped* plans are cached and it is disabled under adaptive risk. Parity tests compare trades, equity and counters against the old path on plain, adaptive-risk and prop-account configs. Measured 4.6-5.7x on a workload that was ~99.8% skips; the gain scales with the share of skipped entries, so ordinary strategies will see less.
- **Day-index cache lock + `_LOOKAHEAD_CACHE` (finding 3).** Removes a rare crash in long jobs.
- **`dayofweek` -> `day_of_week` (finding 5)** across 18 files.
- **Process workers for sweeps (finding 2).** Walk-forward, sensitivity and parameter robustness run on a spawn-safe process pool (`app/validation/process_sweep.py`, same initializer pattern as Search Lab) with a serial fallback. Tested on a spawn pool on Linux; **not yet run on the Windows build**.
- **Speed Run Phase 2 stays on threads.** The benchmark (`scripts/bench_speed_run_phase2.py`) did not finish in the time available, so the move was not made without a measurement. Re-run it on a real machine before changing.
- **Build hygiene (findings 6/7).** `config/requirements-core.txt` and `requirements-extras.txt` added (`requirements.txt` remains the full install for CI/release); the web build job now `needs: test`. A fully pinned cross-platform lock must be generated on your machine (`pip freeze > config/requirements.lock`).

## Interactive Replay and Live Monitor
Shared chart kit (`t58-chart-kit.js`) and rewritten drawing tools (`chart_drawing_tools.js`); both pages use it, Live Monitor panes 2-4 included.
- Chart types: candles, hollow, bars, line, area, baseline, Heikin-Ashi.
- Indicators: SMA, EMA, WMA, Bollinger, VWAP, Donchian, Supertrend on price; RSI, MACD, Stochastic, ATR in their own panes; volume.
- Drawing: trend, ray, extended, horizontal/vertical line, arrow, channel, Fib retracement/extension, measure, box, ellipse, path, text, long/short R:R. Select, move, edit handles, colour/width/dash/fill, lock, clone, magnet, undo/redo, hide/lock all, per-symbol persistence.
- Settings: 5 themes, candle colours, grid, crosshair, log/%/indexed price axis, timezone, hi/lo and last-price lines, watermark, legend, full-screen, PNG screenshot. Preferences persist.
- Replay: follow mode, entry-to-exit lines, SL/TP zones, go-to-time, step size, open-trade chip, new shortcuts (Home/End, B/S/C, Shift+arrows). The chart never receives future bars.
- Replay prop panel: trailing / end-of-day / static drawdown, daily loss, min days, consistency rule, sticky pass/fail banner, firm-preset dropdown.

## Prop firm simulator
- New rules: end-of-day trailing drawdown, floor lock at starting balance, daily-loss lock-the-day, eval time limit.
- Choosing a firm preset fills rules, fees and profit split; the preset note shows whether the firm allows algos.
- Rough EV card, plus a **Payout forecast** panel (below).

## Payout forecast (validation)
Answers "will this get paid?", not "does this pass our gates?".
- Per firm preset: Monte Carlo plus a rolling replay from real start dates; Wilson lower bounds; the decision uses the *lower* of the two.
- P(paid | pass) and expected payout, then **ranked by conservative EV per month**. Trades are rescaled to each firm's account size at constant risk %.
- Graded **OUT-OF-SAMPLE** when run on the held-out 30%, otherwise **IN-SAMPLE (optimistic)**.
- Verdicts: PAYS / MARGINAL / DOES NOT PAY / NO FEE DATA. Informational in the full pipeline: it never changes the pipeline verdict.
- Calibration ledger (`payout_forecast_ledger.jsonl`): log real eval outcomes; forecasts are shrunk toward reality when they have been too optimistic, never raised.
- `score_vs_payout`: Spearman check of whether the T58 score actually tracks payouts on your saved strategies, and it will say so when it does not.
- Routes: `POST /api/payout-forecast/run`, `POST /api/payout-forecast/outcome`, `GET /api/payout-forecast/calibration`.

## Known limits
- The forecast ranks firm fit and records real outcomes; it does not trade on live firm accounts. The existing Deploy connectors are unchanged.
- LFS market-data files were unavailable during testing; tests needing them were skipped or fail for that reason.
- Three `test_translator` failures and one 1-minute-data strategy test fail in this checkout for reasons unrelated to these changes (a shipped expression-style strategy; missing 1m data).

## 2026-11-04 reliability audit

Goal: fewer false rejections of strategies that really have an edge, without loosening what protects against fake edges. Every change
replaces a measurement that conflated two things with one that measures the thing it names. Faster strategies are preferred over slower
ones (70% in 30 days beats 80% in 150 days) as a **ranking/score** input, never as a gate.

## Calendar days, not trade-days
* `app/prop/account.py`, `simulator.py`, `vector_account.py`: every attempt records the CALENDAR days it lived (`calendar_days_to_pass`,
  first-attempt state/days). `days_to_pass` stays a trading-day count (days with a trade) and is no longer used as a calendar figure.
* `app/monte_carlo/engine.py`: `median_calendar_days_to_pass`, `pass_within_days_pct {30,60,90,180}`, `attempt_outcomes`,
  `attempt_calendar_days`, `speed_factor`, `speed_adjusted_pass_pct`. The legacy Wilson interval now uses a fixed 30-day attempt window.
* `app/scoring/speed.py`: `speed_factor = (30 / calendar_days) ** 0.5` (1.0 at <= 30 days). 70%@30d = 70.0, 80%@150d = 35.8.
  Used in the scorecard (`speed_efficiency`, weight 4), sweep ranking, planted-edge checks and `book_prop_account`.

## Report items
| Item | Change | Files |
|---|---|---|
| P0-1 | Outer (nested) bootstrap pass interval replaces the n_eff Wilson interval (which measured slowness) | `monte_carlo/outer_bootstrap.py`, `engine.py`, `full_pipeline.py` (`outer_bootstrap_*`) |
| P0-2 | DSR on unseen (holdout + OOS) trades; failure = score input (`dsr_unseen`) + cap, not NOT READY (`dsr_hard_gate=False`); same-fingerprint ledger trials decay on a 5-year half-life instead of never | `audit_layers.unseen_dsr`, `full_pipeline.py`, `validation/trial_ledger.py`, `scoring/t58_scorecard.py` |
| P0-3 | PBO: midrank ties, one representative per correlation cluster, informational (`pbo_informational=True`) | `validation/cpcv.py`, `validation/effective_trials.py`, `full_pipeline.py` |
| P1-1 | One uncertainty treatment per quantity: when the holdout confirms the edge, the search-bias deflation is not applied again | `audit_layers.py`, `ev_verdict.py` |
| P1-2 | Expectancy gate uses the UNSEEN lower bound at 2x costs; the in-sample bound only ranks | `full_pipeline._attach_unseen_cost_bound` |
| P1-3 | Stale (old but primary-verified) rules stamp = banner + deployment-checklist blocker, not a verdict demotion; never-verified rules still cap | `prop/presets.rules_stamp_state`, `audit_layers.py`, `orchestration/deployment_checklist.py` |
| P1-4 | Data manifest: READY from >= 3 years when the cross-market transfer test passes; otherwise 3-5 years still blocks READY. Synthetic / unreadable / roll-gap data stay hard blockers | `data/data_manifest.py` |
| P1-5 | Seed gate: 20 seeds x 300 sims, only the spread in EXCESS of Monte Carlo noise counts; fewer than 20 seeds = gate dropped | `full_pipeline.py` |
| P2-1 | FORWARD-TEST CANDIDATE earned on edge evidence alone (holdout confirmed, null ok, unseen cost bound > 0, EV > 0) | `orchestration/verdict_ladder.py`, `audit_layers.edge_evidence_for_ladder` |
| P2-2 | Empirical-Bayes shrinkage of the unseen edge (only ever lowers the EV pass rate), family prior, SPRT forward-test plan | `audit_layers.edge_shrinkage_and_plan` |
| P2-3 | GA objective default `edge_lb_r`; leaderboard clustering helper; edge book scored as ONE prop account | `full_pipeline.py`, `validation/effective_trials.cluster_leaderboard`, `portfolio/edge_portfolio.book_prop_account` |
| P3 | Sizing rule fixed from the firm's limit before the search (no dev result used); planted-edge tests extended | `full_pipeline._size_by_rule` (`size_rule="rule"`), `tests/orchestration/test_planted_edge_calibration.py` |

## Fix 1 -- per-attempt subscription billing in the EV
`orchestration/ev_verdict.py`: each simulated attempt is billed `monthly_fee x ceil(calendar_days / 30)` (29 -> 1, 30 -> 1, 31 -> 2, 61 -> 3);
passes also pay the activation fee. `EV(p) = p(qV - A - C_pass) - (1-p) C_fail`, break-even is the exact root, `ev_per_month` is reported,
and the pass-rate lower bound feeds an EV lower bound. Tests in `tests/packaging/test_gate_tightening_2026_10_15.py`.

## Fix 2
* 2a `orchestration/gate_histogram.py` + `scripts/gate_kill_histogram.py`: fail rate per gate, fail rate among runs with point pass >= 80%, sole-killer counts.
* 2b `scripts/planted_drift_sweep.py [--check]`: planted drift 0 / 0.05 / 0.10 / 0.15 / 0.20 / 0.30R x 20 seeds through the statistical gate stack (the nightly regression; targets: 0.15R detected >= 75%, 0R false positives <= 5%, 0.12R forward-test candidate >= 60%).
* 2c per-family DSR scope with fingerprint clustering and the ledger decay change above; the DSR on unseen data uses the clustered effective trial count.

## Measured planted-drift results (20 seeds, topstep_50k rules, 1 trade/day, $150 risk, 5 years)
| planted edge | READY-equivalent detected | FORWARD-TEST CANDIDATE | old stack passes | mean pass % | pass LB (new / old) |
|---|---|---|---|---|---|
| 0.00R | 0% | 0% | 0% | 19.7 | 8.6 / 6.5 |
| 0.05R | 0% | 5% | 0% | 36.6 | 20.0 / 16.5 |
| 0.10R | 0% | 40% | 0% | 56.3 | 36.9 / 31.7 |
| 0.12R | 0% | 60% | 0% | 63.3 | 44.3 / 38.1 |
| 0.15R | 15% | 80% | 0% | 74.0 | 57.1 / 48.8 |
| 0.20R | 65% | 95% | 20% | 85.8 | 73.1 / 62.9 |
| 0.30R | 100% | 100% | 100% | 96.9 | 92.5 / 80.4 |

Targets: 0R false positives <= 5% **met (0%)**; 0.12R forward-test candidate >= 60% **met (60%)**; 0.15R forward-test candidate 80%.
0.15R READY-equivalent detection >= 75% is **not met (15%)**. READY-equivalent includes the 70% pass-probability lower-bound bar, and a 0.15R edge
traded once a day on a 50K account only passes ~74% of attempts, so the bar cannot be cleared by any honest interval; the new interval raised the
0.15R lower bound from 48.8 to 57.1 and the old stack detected 0%. With 2-3 trades a day the 0.15R pass rate is ~78-80% and detection is 4 of 8 seeds
(0R stays 0 of 8). I did not lower the 70% bar to hit the number: it is the HIGH-CONFIDENCE label, and under P2-1 a strategy no longer needs it to
reach FORWARD-TEST CANDIDATE. `scripts/planted_drift_sweep.py --check` will therefore report the 0.15R READY target as missed until you decide
whether to relax that bar or size/trade more often.

## Not verified here
The genetic-search stage of the full pipeline was not run end to end on real market data (none in the sandbox); the planted-drift sweep exercises the
statistical gates on synthetic trade streams. Run `python scripts/planted_drift_sweep.py --check` and a real pipeline run before relying on the new defaults.

## Follow-up pass (same audit, app + docs consistency)
| Area | Change | Files |
|---|---|---|
| Prop presets | 25k accounts for FTMO, Apex (EOD and intraday) and Lucid; every preset carries a rules source link; a 30-day in-app *rules re-verified* acknowledgement (`/api/prop-presets/<key>/acknowledge`, stored in `data/preset_stamps.json`). 25k prices are unknown (None), never guessed | `prop/presets.py`, `_lifecycle_rules_js.html` |
| Edition 1.1 / Rulebook | Edge Rulebook page and calculators (z-bar, trades needed, required lift, alpha ledger, binary screen), LLR kill line in the forward-test monitor, `rulebook_screen` on every Full Pipeline result | `validation/rulebook.py`, `forward_test/sprt_monitor.py`, `templates/rulebook.html` |
| Champion | T58 score per row, 1st/2nd/3rd podium, per-row promotion assessment and *Promote to Champion*, replacement comparison against an overlapping champion; the rerun-checks block was removed | `scoring/champion_board.py`, `templates/champion_simple.html` |
| Consistency | One source for the READY bar (`orchestration/ready_bar.py`) feeds the Test side panel and popups; Full Pipeline fitness default is `edge_lb_r` on the page as in the config; stale "100 trades", "last 20%" and "Lucid only verified" text removed | `templates/full_pipeline.html`, `scoring/metric_defs.py` |
| UX | Test / Validate Advanced options split into labelled sections with one-line explanations; Run Batch / Schedule Batch moved into Test → Advanced → Batch & scheduling; Create page examples and flow strip | `templates/full_pipeline.html`, `validate_simple.html`, `section_start_here.html` |
| Deploy / Dashboard | `/api/portfolio` (per-account firm, P&L, open and past trades, health); Deploy overview with next-step line; Dashboard cards default OFF, enabled under Personalize only when an account has trading activity | `live_deploy/portfolio.py`, `static/t58-portfolio.js` |
| Global Events | Regime intelligence with printed rules, macro releases, central banks, sessions, futures stats, vol regime, cross-asset, correlations, geopolitics | `global_events/intel.py`, `static/global-events/intel.js` |
| Replay / prop sim | Next / previous trade jump, keyboard shortcuts, max speed; equity-floor-target chart, risk-per-trade sweep with a recommended level, Wilson ranges | `replay_view.html`, `static/t58-propsim.js` |
| Data | Frozen NQ / ES / MNQ / MES pack with a calendar-fixed sealed window and content hashes | `data/data_pack.py`, `scripts/freeze_data_pack.py`, `data/data_pack.json` |
| AI | Auto provider chain (saved cloud key first, Ollama fallback), longer Ollama timeouts, fake-Ollama-server tests | `ai/ollama_settings.py`, `ai/ollama_transport.py`, `tests/ai/` |
| Speed | Promotion steps reuse one built strategy per candidate instead of re-importing it for every slice | `discovery_engine/engine_steps.py` |

Done in the 2026-11-05 follow-up pass (see the section at the end of this file): E8 One 25k and LucidFlex 25k/50k/100k presets (UNVERIFIED -- secondary sources only), the GA-vs-random proof harness (`app/optimize/ga_vs_random.py`, `python -m scripts.run_ga_vs_random`), elite fitness-noise re-evaluation (`app/optimize/elite_reeval.py`), the n_eff 0.80/0.90/0.95 sensitivity display and the per-step promotion pass/fail summary on the Discovery Engine results page. Still open: the desktop (tkinter) Champion rerun button, and the surrogate false-discovery study at +0.10/+0.20/+0.30R -- the script (`python -m scripts.run_surrogate_study`) runs for hours at the full 60 null runs, and the gate (`gate_status`) stays CLOSED until a fresh saved report shows < 5% false discoveries.


## 2026-11-05 follow-up pass

* **Adaptive risk is ON by default** for Test (Full Pipeline): `FullPipelineConfig.adaptive_risk_enabled = True`, the Test page box is pre-ticked, and the CLI and Speed Run inherit it. Optimize/Refine forms are unchanged (the Test page notes the two must match to be comparable).
* **Presets:** `e8_one_25k` (scaled from the verified $100k E8 One configuration -- E8 One is configurable, no $25k page was found) and `lucid_flex_25k/50k/100k` ($1,250/$3,000/$6,000 targets, $1,000/$2,000/$3,000 loss allowance, 50% evaluation consistency; eval drawdown type assumed EOD trailing). All are unstamped, so they cannot support a READY verdict until you re-read the firm's page and acknowledge them. Prices are None.
* **GA vs random:** equal-budget paired comparison with an exact sign test. On the built-in sphere, Rastrigin and needle landscapes the GA won 29-1, 29-1 and 30-0 of 30 paired seeds. This proves the search loop climbs; it does not prove any strategy has an edge.
* **Elite re-evaluation:** the top 3 candidates are re-scored on 2 fresh Monte-Carlo seeds and the best average wins (`RefinementConfig.elite_reeval_count/repeats`; 0 disables). The report is on `RefinementResult.elite_reeval`.
* **n_eff sensitivity:** the effective-trial count at correlation cut-offs 0.80/0.90/0.95 is shown beside the Discovery Engine result (`TrialVerdictInputs.sensitivity`).
* **Promotion display:** each champion shows "N of 5 checks passed, M failed, K not run".
* **Optimize page** advanced options regrouped into six sections (same field names).
* **Fixes found while making the suite clean:** the vectorised Monte-Carlo kernel skipped the "first attempt busted" bookkeeping on steps where every live path failed (e.g. the inactivity rule), so `attempt_outcomes` disagreed with the per-path loop (10 parity cases failed); `default_library.py` listed an uncommitted strategy file; alpaca tests now skip when the optional `alpaca-py` is missing.
* **Speed:** Monte-Carlo percentiles sort each array once; `_periodic_max_drawdown` no longer copies the whole equity frame. Profiling a full pipeline run showed the remaining time is the bar-by-bar execution loop and pandas expression evaluation -- no further change there was safe to make without risking the numbers.
