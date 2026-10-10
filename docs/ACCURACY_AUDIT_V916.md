# v9.16 -- Full Pipeline audit fixes (2026-10-10)

Source: the "T58 Full Pipeline Audit" (Oct 10, 2026). Each row = one audited defect, what changed, where, and the test that pins it.

| Audit # | Defect | Change | Files | Test |
|---|---|---|---|---|
| P0-1 | Pipeline optimized "eventually passes", not "paid within 30 days" | New evidence + gate: fresh accounts are bought on many real start dates and run through the real engine for `payout_horizon_days` (35); the share paid within `payout_window_days` (30 calendar days) is reported and READY needs >= `payout30_min_pct` (35%) | `app/orchestration/full_pipeline.py`, `app/prop/attempt_replay.py`, `app/backtest/execution.py` | `test_verdict_wrapper_demotes_ready_on_low_30_day_payout_rate`, `test_attempt_replay_tracks_first_payout_within_30_days` |
| P0-2 | Pass-odds evidence was in-sample | The payout replay runs on the dev slice AND on the untouched holdout; the gate uses the holdout when it holds >= `payout30_min_effective_windows` independent windows | `full_pipeline.py` (`_payout_window_evidence`) | same |
| P0-3 / P1-4 | Intervals assumed independent overlapping starts | Moving-block bootstrap (+ Wilson on the effective sample size) -> `pass_ci_honest`, `payout_within_ci`, `n_effective`; shown in the verdict reasons / report | `attempt_replay.py` | `test_block_bootstrap_*` |
| P1-5 | EOD trailing peak ratcheted after every winning trade | Bar engine passes `is_last_of_day` only for non-EOD bases; EOD high-water mark moves at the session roll (`PropAccount.end_day`) | `execution.py` | `test_eod_trailing_peak_moves_only_at_session_close` (fails on the old code) |
| P1-6 | No flatten-by-close rule | `PropRules.flatten_time_ct` ("15:45" CT = 4:45 pm ET): open position closed at that bar's open, entries blocked until the 17:00 CT roll; Lucid presets set it | `simulator.py`, `presets.py`, `execution.py` | `test_session_flatten_*` |
| P1-7 | Roll heuristic matched ordinary weekend gaps | Equity-index products: roll candidates only in Mar/Jun/Sep/Dec, days 4-21 | `app/data/importer.py` | `test_roll_detection_ignores_weekend_gaps_in_non_roll_months` |
| P1-8 | `to_prop_rules()` dropped fields | Copies `trailing_distance_basis`, `max_eval_calendar_days`, `flatten_time_ct`; Lucid 50K uses the fixed-dollar ("account") trailing distance | `presets.py` | `test_lucid_preset_carries_every_rule_field` |
| P1-9 | 3x-risk loss clamp hid gap losses in prop runs | In prop runs the clamp is the account's max drawdown (explicit caller values win) | `app/backtest/risk.py` | `test_prop_runs_do_not_clamp_losses_at_three_r` |
| speed | Two tight-stop trades out of 4,600 disabled the whole GA search | ATR scale-mismatch warning (which gates the GA) now needs >= 5% of trades (min 3) | `execution.py` | existing `test_risk_pip_size`, `test_reliability_improvements` |
| speed | Slow break-it gates ran even when they could not change the verdict | Random-entry null and seed-stability MC are skipped when the verdict is already below READY (they can only demote READY); the null runs 100 seeds first and a second batch only when 0.03 < p < 0.30 | `full_pipeline.py`, config `null_first_batch` | existing `test_null_gate` |

## Not done (still open)
* The GA's own fitness is still the Monte-Carlo `eval_pass_probability`; the 30-day payout is a verdict gate and report evidence, not yet a search objective.
* Other firms' presets are unverified against firm rulebooks; only Lucid 50K carries `flatten_time_ct` and the fixed-dollar trailing distance.
* Mid-session contract rolls are still not detected (no contract calendar in the repo).
* Selection bias from the GA is reported (`selection_bias_caveat`), not deflated.
