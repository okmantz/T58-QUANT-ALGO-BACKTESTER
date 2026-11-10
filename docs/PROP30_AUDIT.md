# 30-Day Prop Evaluation: technical audit, findings and architecture (2026-11-10)

Target: pass a $50,000 evaluation (+6%, 4% max drawdown, 2.4% daily loss, Lucid-style floating/EOD rules) within 30 CALENDAR days
of purchase, with the 95% lower bound on P(pass) at or above 65%.

**Bottom line: the data in this repository does not establish that any strategy achieves this, and the existing NQ Gap-Aligned
Opening Drive does not come close (0.0% within 30 and 60 calendar days, 95% upper bound 5.0%). No success is claimed.**

## 1. Why 99.8% (Monte Carlo) vs 12.5% (actual-date replay)

They answer different questions, and the report mixed them under one "pass" label.

| Number | What it really measures |
|---|---|
| 99.8% MC "eventual" / chain pass | Probability that repeated attempts eventually pass at some time; no calendar limit |
| `per_attempt_pass_probability` | One attempt, MC-resampled trades, horizon counted in TRADES/trading days |
| 12.5% replay | Fresh account purchased on real dates, run through the real engine, fixed horizon |
| 30.8% historical evaluation | Unlimited-time evaluation from one start |
| **0.0% (this audit)** | Purchase on every weekday, clock from PURCHASE in calendar days, floating DD/daily loss |

Concrete unit/clock mismatches found:

1. `days_to_pass` counts days that contained a trade; `calendar_days_to_pass` starts at the attempt's FIRST TRADE day. A sparse strategy
   (about 9 trades per 30 days) is reported as passing in far fewer "days" than elapsed. `orchestration/full_pipeline.py` converts
   median trading days with x7/5, which is only valid for a daily-trading strategy.
2. `run_attempt_replay` allowed starts up to `horizon/4` before the data ends, so up to a quarter of the windows were truncated and
   their unresolved outcomes counted as failures (downward bias), and it started accounts "mid-signal" on level signals.
3. The skip metric divided signal EPISODES by BARS: `sizing_summary.skip_rate` printed 0.6% where 4.0% of signals (28 of 696) were
   skipped. The warning text and the report disagreed. (Fixed; the new field `signals_seen` is the denominator.)
4. 2,236 "skipped" vs 0%: the 2,236 was bar-level retries of the same unaffordable signals (`skipped_for_sizing_bars` 3,717 here), the 0% was
   the per-signal rate computed on the wrong base. Both are now labelled and reconciled.
5. MC used the 300 development trades; the report's total is 373 including the 73 holdout trades, so the holdout was both reported and
   already used for sizing/selection: it is not independent.

## 2. Adaptive risk and sizing

On the bundled NQ data (668 trades): adaptive risk reports an average multiplier of 0.41 and is active on 85% of trades (intended
risk $126.45, actual stop risk $69.97 vs $90.96 without adaptive). Net P&L is $11,644 with adaptive vs $11,833 without (1.6% apart) and
the set of traded signals is identical: in this run the throttle is almost inert in P&L terms, which is consistent with whole-micro-contract
rounding absorbing it (not isolated further). Reporting "0.48x" as a risk reduction overstates what the account experienced.

## 3. Exit engine

Verified by regression tests (`tests/backtest/test_exit_engine_audit_2026_11_10.py`): stop wins when stop and target share a bar; a gap
through the stop fills near the open, worse than the stop; the target fills at the target; entries pay spread/slippage. No bug found.

## 4. Research/validation independence

The bundled data has a sealed window (from 2025-09-01) that earlier experiments already read; the GAD run's 73-trade holdout lies inside the
same file. Nothing in the bundle can be certified as never observed. The new gate (`app/validation/prop30_validation.py`) therefore
refuses to call any holdout independent unless it starts after the selection cut-off plus embargo and overlaps no declared observed range,
and it never reports CONFIRMED without forward results.

## 5. What was built

| Phase | Module |
|---|---|
| Evaluator (calendar days from purchase, floating DD, dependence-aware CI) | `app/prop/thirty_day.py` |
| Frequency/edge feasibility | `app/prop/feasibility.py` |
| Replay fixes (opt-in, defaults unchanged) | `app/prop/attempt_replay.py`: `calendar_days_to_pass`, `n_truncated`, `pass_within_days`, `min_horizon_coverage`, `arm_signals_after_start` |
| Skip-rate unit fix | `app/backtest/execution.py`, `portfolio_engine.py` |
| Opportunity scanner (NQ/ES/GC; CL has no data in the bundle) | `app/discovery/opportunity_scanner.py` |
| Exit lab on frozen entries | `app/research/exit_lab.py` |
| Portfolio: correlation, shared risk, regime gate, allocation | `app/portfolio/prop_portfolio.py` |
| Validation gate and forward registry | `app/validation/prop30_validation.py` |
| Pipeline / CLIs | `app/research/prop30_pipeline.py`, `scripts/run_prop30_research.py`, `scripts/audit_gad_engine.py` |

Thresholds were not changed. Run: `python scripts/run_prop30_research.py` (writes `reports/prop30/`).

## 6. Results (discovery 2020-2022, development 2023-H1 2024)

See `docs/PROP30_RESEARCH_REPORT.md`. 150 scanner cells on NQ/ES/GC: the best has P(pass<=30d) 8.4% (95% lower 3.3%) on discovery
and 4.9% on development. The best exit policy on the best entries reached 19.0% (lower 8.2%) in-sample; the best 3-leg portfolio 27.7%
(lower 16.4%) in-sample with 60% bust. All fail the 65% lower-bound target by a wide margin.

## 7. Why, and what would change it (i.i.d. upper bound)

At a 0.3 R net edge per trade a strategy needs about 40 trades per 30 days for ~72% (0.5 R: ~10-20 trades). The scanned families give
17 trades per 30 days at 0.03-0.06 R net edge. The binding constraints are the 4% trailing drawdown against a 6% target and the
net-of-cost edge, not the number of ideas tried. Recommended: new genuine edge sources with higher net R per trade or much higher frequency
(more liquid intraday instruments, true multi-instrument concurrency), a CL dataset, and a forward test of any survivor before any claim.

## 8. Limits of this work

- Scanner families are simple, costs are fixed estimates (`INSTRUMENTS` table), and all scans are in-sample on discovery; survivors are hypotheses.
- Portfolio weights are chosen and scored on the same data (flagged in the report).
- Block robustness is contiguous-block dispersion, not a full CPCV purge/embargo.
- `tests/suite/test_journal_suite.py` is skipped because `app/journal/{behavior,capture,statistics}.py` are absent from the checkout supplied.
