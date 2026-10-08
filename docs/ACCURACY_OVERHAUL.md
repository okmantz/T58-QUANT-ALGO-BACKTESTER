# T58 accuracy overhaul + discovery layer (2026-10-07)

Goal: every backtest mode reports what a funded prop account would see (same trades, sizing, costs, account rules), and a discovery layer turns an idea into a tested, deflated, break-it-checked, recorded hypothesis.

## What changed (by report finding)

| Finding | Fix | Where |
|---|---|---|
| Cost units: instrument spreads/slippage were stored in *ticks* but applied as *pips* (ES: 4x too small) | ticks -> price -> pips via `tick_size/pip_size`; `round_trip_cost_dollars()` (ES default $54.20/contract) | `app/data/instrument_specs.py`, `app/search/instrument_risk.py` |
| Sizing ignored costs, so ~every stopped trade "overshot"; fractional/zero-contract rescue distorted risk | One sizing routine `RiskConfig.size_for_stop`: whole contracts, costs (exit spread+slippage+commission) inside the budget, four modes `skip / fit_stop / fixed_contracts / micro_fallback`, explicit skip reasons | `app/backtest/risk.py`, `execution.py`, `vectorized_fastpath.py` |
| Five different account-rule implementations | One `PropAccount` rule engine used by the bar engine, `simulate_account`, attempt replay and Monte Carlo (legacy semantics reproduced exactly; verified against a frozen copy of the old simulator) | `app/prop/account.py`, `simulator.py` |
| Drawdown/daily loss only on realized P&L; no firm-day logic in the bar engine | `account_model="prop"`: floating (intrabar) trailing/static drawdown, floating daily loss with fail or lock-day, HWM bases (`realized/eod/floating`), trailing lock, profit target, min days, consistency, eval time limit, inactivity, payouts; liquidation at the floor *unless the stop is hit first*; each attempt is a fresh purchased account (`attempt_id` on trades and equity rows, `attempt_mode="chain"|"single"`, `stop_on_pass`) | `app/backtest/execution.py` |
| Rolling "evaluation windows" re-fed a fixed trade list | `run_attempt_replay`: re-runs the real engine from many start dates as fresh accounts; pass / bust / unresolved, bust-before-pass, expected accounts per pass, Wilson CI, sample-floor flag | `app/prop/attempt_replay.py` |
| Monte Carlo cut sessions in half; ruin = "any attempt in a rebuy chain died" (~100%) | `day_block_bootstrap` default (whole session days stay together); `risk_of_ruin_pct` = failed attempts / attempts; `bust_before_pass_probability`, `expected_attempts_to_pass`, `sample_ok` / `sample_notes` | `app/monte_carlo/engine.py` |
| GA climbed noise at ~14 trades; cost stress forgot per-contract commission | fitness shrinkage below 15 source trades; stressed risk scales `commission_per_contract` | `app/optimize/refinement.py` |
| Verdict could be READY on thin or contradicted evidence | sample-floor and replay-agreement gates (can only demote READY -> MARGINAL) | `app/orchestration/full_pipeline.py` (`_make_verdict` wrapper) |
| Statistics overshoot measured against the bare stop | measured against `risk_at_stop_dollars`; trades segmented by `attempt_id` | `app/backtest/statistics.py` |
| Structural problems found only after a run | `run_preflight` (pip/contract mismatch, zero costs, unaffordable stop, stops-to-bust, daily-limit tightness, sample sizes) | `app/validation/preflight.py` |

## Discovery layer

`app/discovery/` + `app/backtest/resting_orders.py` + `app/ai/llm_client.py`, CLI: `python cli.py discover --idea "..." --data ES.csv [NQ.csv] --timeframes 15min 1h --symbol ES`.

1. **Idea -> rule spec** (`idea_compiler`, `rule_spec`): a language model (optional) or a deterministic keyword compiler produces a *whitelisted, bounds-checked* spec; hallucinated kinds/params are rejected, never coerced. The reason a fallback was used is recorded on the hypothesis.
2. **Resting orders / persistent zones** (`resting_orders`): limit orders that fill only if price trades through, fair-value-gap zones, same sizing and costs as the engine, stop booked before target on ambiguous bars.
3. **Hypothesis object** (`hypothesis`): mechanism, falsifiers, markets, timeframes, every experiment (win or lose) and the number of variants tried; JSON store, nothing deleted.
4. **Experiment grid** (`experiment_runner`): markets x timeframes with the real engine; every cell's Sharpe is **deflated** for the cells tried plus all earlier variants on that hypothesis.
5. **Break-it battery** (`break_battery`): random-timing null distribution with p-value, held-out segment, 1.5x costs, volatility regimes, parameter neighbors, consecutive time folds, cross-market. Verdict `survived / broken / inconclusive`; tests that cannot run are reported `not_run`, never silently passed.

## How to turn the new account model on

```python
risk = RiskConfig(..., account_model="prop", prop_account_rules=PropRules(..., dd_basis="floating", daily_loss_basis="floating"),
                  sizing_mode="fit_stop")
trades, equity = run_execution(df, signals, risk, stop_loss_pips, take_profit_pips, attempt_mode="chain")
equity.attrs["prop_attempts"]    # one record per purchased account
```
`account_model="legacy"` (default) keeps every previous behaviour; `PropRules.dd_basis="legacy"` reproduces the old simulator exactly.

## Second pass (2026-10-07, later)

| Item | What exists now | Where |
|---|---|---|
| 1-minute intrabar fills | When one bar could touch both stop and target, the 1-minute path decides which came first (stop first if the same minute touches both); trades carry `fill_resolution="intrabar"` | `execution.py`, `engine.run_backtest(intrabar_df=)`, `data/timeframe_resample.resample_with_intrabar` |
| GA reliability | <15 trades = -inf, <100 scaled down, skip-rate penalty; used by walk-forward GA with lookahead check skipped for speed | `optimize/refinement.py`, `optimize/walkforward_ga.py` |
| Preflight gate | Quick Optimize and Full Pipeline stop when the baseline has <100 trades or >20% of signals skipped for sizing (`preflight_enforce`) | `validation/preflight.py`, `orchestration/*` |
| Holdout | `run_holdout_comparison(continuous_account=True)`: holdout continues the account instead of starting empty; reports `min_holdout_trades` | `backtest/engine.py` |
| Presets | `dd_basis`, `trailing_lock`, lock offset, contract caps; Lucid 50k updated; `compare_rules_to_preset` | `prop/presets.py` |
| Reports | per-attempt sawtooth equity curve, reworded overshoot note, reliability header, battery section | `reports/*` |
| Continuous contracts | difference (Panama) back-adjustment of roll gaps | `data/continuous_contract.py` |
| Web / desktop forms | instrument-first fields, sizing mode, mismatch blocking | `web/accuracy_form.py`, `_accuracy_fields.html`, `ui/main_window.py` |
| Zones in the engine | `StrategyResult.entry_orders`, manual `zone_entry`, strict causality check, fast path refuses them | `strategy/zones.py`, `strategy/base.py`, `strategy/manual.py`, `backtest/engine.py` |
| Null gate | random-entry distribution + p-value; can only demote READY -> MARGINAL | `research/director.py`, `full_pipeline.py` |
| Hypotheses | linked to experiment memory; already-tested warning | `ai/experiment_memory.py`, `discovery/experiment_runner.py` |
| Papers -> hypotheses | claim extraction | `discovery/paper_hypotheses.py` |
| Hosted LLM | Claude/OpenAI keys used by generator, research loop and agent | `ai/llm_client.py` |
| Speed | lookahead check cached by strategy structure | `backtest/engine.py` |
| "Your idea" screen | `/discover`, job-based, shows rule, grid, battery | `web/discover_routes.py` |
| Translator | zone entries rendered to Pine and MQL5 | `strategy/translator.py` |
| Real-account check | compares your real sessions to a replay | `validation/real_account_check.py` |
| Data coverage | `python scripts/data_coverage_audit.py` -> `docs/DATA_COVERAGE.md` | `scripts/` |

## Honest limits

* **Only Lucid 50k was updated, and only against a secondary source (checked 2026-10-07).** Other presets are unverified. Check each firm's current rulebook before trusting a pass probability.
* **The real-account check has not been run.** It needs your ~20 real sessions (entries, exits, sizes, fees). It is the only test that proves the model matches a funded account.
* The execution loop is **not compiled** (no numba); the speed gain is the lookahead cache only.
* Pine/MQL5 zone output is generated text; it has **not been compiled or run** in TradingView/MetaEditor. No cTrader zone output.
* Back-adjustment exists as a module but is **not wired into the importer**; run it explicitly on roll-contract data.
* Hosted LLM calls were written against the documented APIs and **not exercised live**.
* The keyword idea compiler is crude: it can ignore numbers in the idea (e.g. "stop 1 ATR") and fall back to defaults. Read the rule shown on the "Your idea" screen before trusting the result.
* The stop-in-dollars limit in the GA is enforced by the skip-rate penalty, not by a bound on the stop gene.
* Monte Carlo still resamples a trade list; use the attempt replay as the cross-check.
* The break-it "out-of-sample" segment is only out-of-sample if the rule was not tuned on it.
* `sizing_mode="skip"` keeps a bounded dead-lock guard (single contract when worst case <=1.5x budget, tagged `sized_above_risk_target`).
* Defaults changed: Monte Carlo method is `day_block_bootstrap`; `risk_of_ruin_pct` is the per-attempt bust rate; the web form defaults sizing to `fit_stop`; the preflight gate blocks runs under 100 trades.
