# Corrections and change log: report of 2026-10-20

Report: "T58 Backtester: Why Nothing Reaches READY and What To Change".

## Statements in the report that were already stale when checked against the repo

| Report statement | Repo state before these changes |
|---|---|
| The 30-day payout gate rejects | It was already a soft gate. Unchanged. |
| No EV verdict module | `app/scoring/ev_verdict.py` existed. It is now the rejecting bar and is printed beside READY in every report. |
| ORB `time_of_day == 0.627` is not caught | A float-equality lint existed for `time_of_day`. It now covers every float-valued operand, not just `time_of_day`. |
| Preflight allows heavy skipping | Baseline preflight already blocked at 20% skipped signals. It is now 25% with a "SIZING ERROR:" prefix. |

## Measured corrections

* Trade frequency is about 7 trades per month on the audited strategies, not the figure implied in the report. SPRT planning uses 6.7 per month for the months-to-decide estimate.
* Power: 237 trades for +0.21R, 117 for +0.30R, 1045 for +0.10R (sd 1.3R, alpha 5%, power 80%).
* SPRT (H0 = 0R, H1 = 0.21R): about 103 trades expected under H0 and 146 under H1. That is 15 to 22 months with one edge and 4 to 5.5 months with four independent edges.

## What changed

Gates replaced (ruin, null test, 2x cost stress, holdout and lookahead are NOT loosened):
1. ICIR and half-life gate is advisory. `edge_ic_gate` tests signed R. Old functions remain.
2. Walk-forward uses a pooled purged OOS mean-R test with a day-block bootstrap lower bound above 0 and at least half of merged folds positive. FAIL rejects, NOT_PROVEN caps at MARGINAL, UNPROVEN means it could not run. Efficiency is diagnostic.
3. The fixed 70% per-attempt bar is now a HIGH CONFIDENCE label and a MARGINAL cap (`per_attempt_pass_mode="cap"`). Set `"hard"` for the old behaviour.
4. DSR keeps its 0.95 bar but counts effective (clustered) trials, with an aging trial ledger.
5. PBO and CPCV can use `mean_r`.
6. The EV test rejects (`ev_hard_bar=True`). NOT WORTH A FEE demotes to NOT READY. INSUFFICIENT EVIDENCE caps at MARGINAL.
7. Edge-decay flag (by-year edge and rolling slope) caps READY at MARGINAL.

Measurements fixed: tick-based cost model with a 2x stress that does not scale commission; blank spread and slippage defaults in web forms resolve from the instrument; sizing-skip limit; scorecard prefers `generalization_score`.

New:
* Session, calendar, event, daily-context, VIX and rates primitives (`app/strategy/indicators_v11.py`).
* Eleven new strategy families (`app/search/families_v11.py`).
* Event calendar, macro data, data rules, and update scripts.
* Two-stage optimizer (`app/optimize/two_stage.py`), edge portfolio (`app/portfolio/edge_portfolio.py`), verdict ladder (`app/orchestration/verdict_ladder.py`).
* Pre-registration (`app/research/preregistered_tests.py`) and `scripts/run_preregistered_families.py`.
* Forward-test harness: frozen code hash, fill ledger, SPRT monitor with kill switch, paper session, bias journal.
* Cross-market instrument specs.

## Limits

* Nothing here was re-run on your strategies or real data. Pre-registered runs on synthetic data correctly return NOT SUPPORTED.
* Data fetchers (`scripts/update_*.py`, `snapshot_gex.py`) need network access on your machine. CPI and PCE schedules need a FRED key.
* `vix_backwardation_dip` needs VIX and VIX3M in `data/macro`; `leader_follow` needs `frame.attrs['leader']`.
* Timestamps are treated as US/Eastern wall-clock. Use `frame.attrs['tz_offset_hours']` to shift otherwise.
