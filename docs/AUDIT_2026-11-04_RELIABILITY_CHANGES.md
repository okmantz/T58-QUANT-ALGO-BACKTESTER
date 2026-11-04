# 2026-11-04 reliability audit -- what changed and where

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
