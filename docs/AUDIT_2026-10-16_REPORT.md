# T58 Quant Algo Backtester — Fresh-Eyes Audit (2026-10-16)

Scope: speed and smoothness, UI and deeper issues, the READY verdict and T58 score in depth, and whether the
prop-firm metrics are fair. Every number below was measured on this repo (Python 3.13, 2 CPUs, no numba) or
computed by calling the repo's own scoring code. Where something is a judgement rather than a measurement, it says so.

---

## 0. Bottom line

1. **READY is far stricter than the published gates say.** A strategy sitting exactly on the documented gate
   minimums (pass lower bound 70%, paid lower bound 35%, ruin 20%) scores **65.8 = "Research" → MARGINAL**. Even a 98%
   pass lower bound, 92% paid lower bound and 1% ruin scores only **84.3 → MARGINAL** if the secondary components are
   merely "good". To reach 85 you need roughly **pass LB ≥ 90%, paid LB ≥ 75%, ruin ≤ 5% and every secondary component
   in the 80s** (walk-forward 85, MC robustness 90, parameters 80, expectancy 70 ≈ 0.42R, drawdown 80). The real bar is
   the 85-point cut, not the 70/35/20 gates.
2. **Two real bugs fixed.** (a) Walk-forward optimization scored every candidate `-inf` when the fitness metric was
   `payout_30d` (the default), so per-fold "optimization" was choosing between identical `-inf`s. (b) The verdict core
   contained its whole tail (scorecard lines, tier mapping, caps, holdout gate) **three times**.
3. **Speed:** the walk-forward fix makes the search real, which costs about 13 s on the 84k-bar 1-minute test; the
   Monte Carlo bootstrap and the warning flood are now cheap; the remaining time is process-pool spawn/pickle and PBO.
   Full Pipeline on 3 months of MNQ 1-minute data: **~140 s on this 2-core box** (your machine will be faster).
4. **Prop-firm metrics: mostly fair, three things are not.** See section 5. The EV verdict has fee data for
   **2 of 19 presets**, so for ~90% of presets the "should I pay for this?" answer is always "INSUFFICIENT EVIDENCE".
5. Reports are redesigned (verdict hero + score gauge + component bars, KPI tiles, humanised labels, grouped
   collapsible tables, print CSS) with no information removed; the PDF export got the same treatment.

---

## 1. What changed in this delivery

| File | Change |
|---|---|
| `app/validation/walk_forward_opt.py` | Four `compute_fitness` calls now pass `trades=` and `prop_rules=` (was `-inf` for `payout_30d` / `avg_give_back_r`). |
| `app/orchestration/full_pipeline.py` | Removed the triplicated verdict tail in `_make_verdict_core` (one copy remains, behaviour identical to the copy that was effective). Fixed the "EV EV VERDICT" doubled prefix. |
| `app/monte_carlo/engine.py` | Block-bootstrap index builder vectorised; verified bit-identical (including RNG state afterwards) over 200 seeds × 6 sizes × 7 block sizes. |
| `app/backtest/execution.py` | Per-entry MARKET IMPACT warnings aggregated into one warning per backtest (one `warnings.warn` per oversized entry before). |
| `app/reports/theme.py` (new) | Shared design system for every report. |
| `app/reports/generator.py` | New full-report layout; restored 5 sentences that had been cut off mid-word by a literal `[...]` in the repo copy. |
| `app/reports/pdf_export.py` | Styled PDF (title band, verdict hero, KPI tiles, banded tables, page footer). Now also shows the per-attempt headline numbers the HTML uses. |
| `app/reports/validation_reports.py`, `survival_report.py`, `refinement_report.py`, `app/search/search_report.py`, `app/validation/report_export.py` | Same theme; no fields dropped. |
| `data/examples/` removed | See section 6. Sample references repointed to `data/raw/EURUSD/EURUSD_5M_sample.csv` (byte-identical file). |
| `data/data_manifest.json`, `docs/DATA_COVERAGE.md` | Removed 17 phantom rows (test litter: `a.csv`, `b.csv`, `eurusd.csv`, `leaky.csv`, `EURUSD_5M_sample (N).csv`) that made every Full Pipeline start print "manifest lists N files that are not on disk". |
| tests | `test_w4_realism_validation.py` holdout fixture updated to the 30-trade policy (see 3.4); sample-path repoints. |

---

## 2. Speed and smoothness

### 2.1 Measured time split (MNQ 1m, 84,077 bars, default config, 2 CPUs)

| Phase | Time |
|---|---|
| Walk-forward-aware GA (Step 2) | ~56 s before the fix, ~69 s after (the WFO phase now evaluates real fitness) |
| of which per-fold re-optimization | ~34 s |
| Final validation + Monte Carlo | ~1.5 s |
| OOS folds, holdout, ICIR, DSR | ~5 s |
| PBO gate | 24–44 s (depends on contention) |
| Process-pool spawn + pickling | ~17 s across the run |

The `-inf` fitness in the old WFO phase was visible in the logs as `best fitness so far = -inf` on every
generation (15 lines in the old run, 0 now; the GA now reports real values, e.g. `3.075`).

### 2.2 Done

* Vectorised block bootstrap (was a Python `while` loop per simulated path).
* One aggregated market-impact warning instead of one per oversized entry (each one is captured by `warnings.catch_warnings(record=True)` and stored in `BacktestResult.warnings`).
* WFO fitness fix (correctness first; it also removes wasted generations spent on all-`-inf` populations).

I also tried replacing the `Period`-keyed groupby in `_periodic_max_drawdown` with integer keys. It was
numerically identical but **not faster** (0.084 s vs 0.091 s on 300k bars), so I left the file untouched.

### 2.3 Next, ranked by value (not applied — each is a bigger, riskier change)

1. **One persistent worker pool for GA → WFO → PBO → MC** instead of spawning per stage (~17 s of ~140 s).
2. **Memoize backtests by parameter hash** across GA generations and between the GA and the per-fold WFO; many
   candidates repeat.
3. **PBO**: reuse the trade lists the GA already produced rather than re-running backtests for each CPCV path.
4. **Optional numba** for the execution loop (the dominant CPU cost); keep the pure-Python path as fallback.
5. **Reports**: Plotly loads from a CDN, so the Trade Visualization tab is blank offline; bundle it or inline the
   minified library once per report folder.
6. Keep `T58_TEST_CLEAN_LITTER=1` in your dev shell: the suite writes ~110 stray files into `data/raw`,
   `strategies/` and `reports/` otherwise (documented in `tests/conftest.py`).

---

## 3. App improvements: UI and deeper issues

### 3.1 Fixed
* **WFO `-inf` fitness** (section 0).
* **Triplicated verdict tail** — three copies of scorecard lines, tier mapping, trade-count cap, coverage cap and
  holdout gate. Reasons were being appended three times, and the tier mapping was recomputed (resetting `verdict`)
  between copies.
* **Truncated report text** — five sentences in `generator.py` literally ended `payou[...]`, `i[...]`, etc.
* **Manifest/coverage litter** — see section 1.
* **Reports** — raw snake_case keys, 4-decimal floats (`initial_balance 100,000.0000`), dicts dumped as Python
  `repr` (`prop_account_rules`), "$-1,234.50", the verdict as a flat bullet list, headline numbers buried under
  the config table.

### 3.2 Found, not fixed (needs your decision)
* **Missing futures contract-roll data gate.** `tests/packaging/test_review_2026_10_15.py::test_data_gate_flags_unadjusted_roll_gaps_...`
  imports `looks_like_futures` and `roll_gap_counts` from `app.data.data_manifest`; neither exists. Either the feature
  was never merged or it was lost in a refactor. Until it exists, unadjusted NQ/ES continuous contracts get no
  roll-gap warning.
* **EV verdict is blind for most firms**: `FIRM_FEES` has 2 rows for 19 presets.
* **Preset verification is a snapshot**: 10 of 19 presets are READY-eligible; the stamps are dated 2026-09-13 …
  2026-10-13 and never expire. Firms change rules monthly; an eligibility stamp older than ~45 days should cap
  READY the same way an unverified one does.
* **DATA GATE needs ≥5 years** and every futures file in `data/raw` that the pipeline can trade (MNQ/MES/MGC/ES/NQ
  1-minute and 5-minute) covers **~3 months** (2026-01-20 → 2026-04-15). Those strategies can never be READY no
  matter how good they are. That is a data problem, not a strategy problem.
* **Tiny post-embargo folds in Step 4** on short histories: with a 15-day dataset the OOS fold check runs on folds
  of a few days; the verdict says so, but the fold check should be skipped (UNPROVEN) below a minimum fold length
  instead of reporting a number.

### 3.3 UI polish list (low risk)
* Put the verdict + headline numbers first (done in the saved report). The live job page was not part of this change.
* Show **why** a cap applied next to the score (done in the report hero: Blocking / Capping / Notes).
* Replace the label **READY** with something that says what it means (see section 4.5).

### 3.4 Stale tests
Five tests in `tests/packaging/test_w4_realism_validation.py` failed before my changes: their holdout fixture has 12
trades, but the 2026-10-15 policy (`min_holdout_trades_for_gate = 30`) intentionally stopped convicting on tiny
holdouts. The fixture now uses 40 trades; the policy is unchanged.

---

## 4. The READY verdict and the T58 score, in depth

### 4.1 How a verdict is produced
1. **Hard gates → NOT READY** (any one is enough): Monte Carlo risk of ruin above the cap (20%); per-attempt pass
   lower bound below 70%; walk-forward (or CPCV) fails; ICIR/decay/Bonferroni gate measurable and failed;
   DSR (probabilistic Sharpe < 0.95) or PBO (> 0.5) fail; holdout lost money **and** the upper 95% bound on
   per-trade expectancy is still below zero (30+ holdout trades); lookahead / random-entry null (p > 0.10) /
   cost-stress (retains < 25%) / payout-in-30-days (< 35%) / attempt-replay disagreement.
2. **Score → tier → provisional verdict**: Elite/Strong (≥85) → READY, Promising/Research (≥65) → MARGINAL,
   below 65 → NOT READY.
3. **Caps (demote only, never promote)**: < 100 trades; score coverage < 70% or a core component missing;
   holdout empty; holdout lost but inconclusive; neither WFO/CPCV nor holdout produced evidence; data history < 5
   years; firm rules unverified; ruin window not covered by the data.
4. **EV verdict** sits beside it (needs a `FIRM_FEES` row): HIGH CONFIDENCE / FORWARD-TEST CANDIDATE /
   NOT WORTH A FEE / INSUFFICIENT EVIDENCE.

### 4.2 The score
`score = Σ(weight × value) / Σ(weight of components present)`, weights sum to 120:

| Component | Weight | Share | How the 0–100 value is made |
|---|---|---|---|
| Pass probability | 25 | 20.8% | Wilson 95% **lower bound** of per-attempt pass rate (%) |
| First payout probability | 20 | 16.7% | Wilson lower bound of P(pass **and** paid) per purchased attempt |
| Risk of ruin | 20 | 16.7% | 100 − ruin% (ruin = busts within 90 calendar days) |
| Walk-forward stability | 15 | 12.5% | WFO efficiency (or CPCV if it is the primary test) |
| Monte Carlo robustness | 10 | 8.3% | seed-to-seed dispersion of the pass rate (0 pts → 100, ≥20 pts → 0) |
| Drawdown | 10 | 8.3% | 100 × (1 − maxDD / firm DD limit); using the whole allowance → 0 |
| Parameter stability | 5 | 4.2% | neighbourhood robustness |
| Expectancy | 5 | 4.2% | average R: 0R → 0, 0.3R → 50, ≥0.6R → 100 |
| Parsimony | 5 | 4.2% | fewer free parameters score higher |
| CPCV (supporting) | 5 | 4.2% | only when CPCV is not the primary test |

Tiers: 92 Elite, 85 Strong, 75 Promising, 65 Research, below that Reject. T58 conformance is informational only.

### 4.3 What the score actually demands (computed with the repo's `compute_t58_score`)

Other components held at "good": WFO 70, MC robustness 85, parameter 70, expectancy 50 (0.3R), drawdown 65, parsimony 75, CPCV 65.

| Pass LB | Paid LB | Ruin | Score | Tier | Verdict |
|---|---|---|---|---|---|
| 70% | 35% | 20% | 65.8 | Research | MARGINAL (gate minimums) |
| 75% | 45% | 12% | 69.9 | Research | MARGINAL |
| 80% | 55% | 10% | 72.9 | Research | MARGINAL |
| 85% | 65% | 8% | 76.0 | Promising | MARGINAL |
| 90% | 75% | 5% | 79.2 | Promising | MARGINAL |
| 95% | 85% | 3% | 82.2 | Promising | MARGINAL |
| 98% | 92% | 1% | 84.3 | Promising | MARGINAL |

Other cases:

* All three survival components perfect, but WFO stability = 0 and MC robustness = 0 (everything else 100): **79.2**.
  A single hard-gate failure in those two areas is also caught separately by the gates, so the score does not need to
  be this punitive on its own.
* Same inputs with the five weakest components omitted: **76.7** at 75% coverage vs **76.9** with them — the
  coverage cap (70%) is the thing that stops omission from helping, and it works.
* A zero-edge strategy with a low drawdown and high parsimony: **32.5**. The score ranks correctly at the bottom.

### 4.4 Findings
1. **The score is not calibrated to the gates.** The gates say 70/35/20; the score needs roughly 90/75/5 *plus*
   secondary components in the 80s to reach 85 (computed: 90/75/5 with WFO 85, MC 90, parameters 80, expectancy 70,
   drawdown 80, parsimony 85, CPCV 80 → exactly 85.0; the same survival numbers with "good" secondaries → 79.2).
   The 33% of weight on expectancy, drawdown, parameter stability, parsimony and CPCV sits on scales that few
   realistic strategies saturate (expectancy needs ≥0.6R for 100; the drawdown component is 100 only if you use none of
   the firm's allowance). READY therefore means "excellent on everything", but the UI and docs present it as "cleared
   the gates".
2. **Pass, payout and ruin overlap.** 65 of 120 weight (54%) measures one underlying quantity (does an account survive
   and get paid). They are correlated by construction (an eval failure is a bust; a bust raises ruin). That is fine as a
   ranking aid; it is double counting if you read the score as 10 independent pieces of evidence.
3. **The 70% pass gate and the 20% ruin cap are not independent.** Ruin counts every bust within 90 days, including
   eval failures, so `pass ≥ 80%` is implied unless attempts time out unresolved. The effective pass bar is ~80%.
4. **Max drawdown component penalises the firm's own allowance**, not the strategy: a strategy at 50% of a 10% limit
   scores 50 here. Risk-of-ruin already prices this, so drawdown is a second charge for the same risk.
5. **Coverage and core-component caps are sound** and prevented a real bug (omitting weak components used to raise the
   score from 70.9 to 83.2 in the earlier audit).

### 4.5 Recommendation
* Keep the gates. Change what READY *means*: **READY = every hard gate passed AND score ≥ 75**, and keep ≥85 as an
  **"Elite"** label, not as the READY threshold. On the table above, 75 corresponds to roughly 85% pass LB, 65%
  paid LB, 8% ruin with good secondary components, which is already a very strong strategy.
* Rename the labels so they say what they claim: `READY` → `VALIDATED – FORWARD TEST`, `MARGINAL` →
  `NEEDS MORE EVIDENCE`. Nothing here is permission to fund an account; the EV verdict already says so.
* I have **not** applied this change: it is a policy decision and it changes many existing test expectations.
  It is a one-line change in the tier→verdict mapping in `_make_verdict_core`. Say the word and I will apply it and
  update the affected tests.

---

## 5. Are the prop-firm metrics fair? Should we uphold them?

Short answer: **uphold the structure, fix the calibration, complete the data.**

| Gate / metric | Threshold | Fair? | Verdict |
|---|---|---|---|
| Per-attempt pass **lower bound** | ≥ 70% | Conservative but principled (judges the pessimistic end of the Wilson interval, not the point estimate). Economic break-even is often single-digit % (the EV module's own example: 8.9%). | **Keep** as the HIGH-CONFIDENCE bar; rely on the EV verdict for "is it worth a fee". |
| Risk-of-ruin cap | ≤ 20% | Fair and it is the right hard gate: it is the one metric that is a genuine pass/fail for money. It is **time-bounded (90 days)**, so longer data no longer inflates it. | **Keep.** Show bankroll ruin separately. |
| Paid lower bound / payout in 30 days | ≥ 35% | Fair; measures the product pass × paid, which is what the buyer cares about. | **Keep.** |
| Min trades for READY | ≥ 100 | Fair; thinner samples give wide intervals. | **Keep** (consider 150 for 1-minute intraday). |
| Holdout gate | 20% reserved, ≥30 trades, reject only if the upper 95% bound on expectancy < 0 | Fair and statistically careful (a 10-trade holdout rejected 15–35% of real edges). | **Keep.** |
| Random-entry null | p ≤ 0.10 | Fair; a one-sided 10% is lenient, which is appropriate for a single test. | **Keep.** |
| Cost stress | retains ≥ 25% of profit | Fair, slightly lenient. | **Keep.** |
| DSR / PBO | ≥ 0.95 / ≤ 0.5 | Standard thresholds. | **Keep.** |
| Data history | ≥ 5 years for READY | Fair in principle (regime coverage) but binary, and your futures data is ~3 months, so READY is impossible. | **Keep the cap, get longer futures history**, or tier it (1 y / 3 y / 5 y). |
| Unverified firm rules | caps at MARGINAL | Fair. | **Keep** and add a staleness cap (~45 days). |
| T58 score ≥ 85 | Strong | **Not calibrated** (section 4). | **Change** (section 4.5). |
| EV verdict | needs `FIRM_FEES` row | Right idea (the only economic answer), unusable today for 17 of 19 presets. | **Complete the table** (price, activation fee, reset price, billing period, source URL, date). |

Things that make verdicts **unfair to a good strategy**:
1. READY needs near-flawless secondary components (4.4 #1).
2. Short data caps everything at MARGINAL regardless of quality (data, not strategy).
3. A 70% per-attempt pass requirement is far above what an economic edge needs, so "NOT READY" does not mean
   "not worth trying" — check the EV line before discarding.

Things that make verdicts **unfair in the other direction (too lenient)**:
1. Rule presets can go stale silently (no expiry).
2. Very short folds and holdouts can still produce a number; where a check cannot be measured it should read
   UNPROVEN, and visibly so (it mostly does — keep that discipline).

---

## 6. Files: remove and keep

**Removed**
* `data/examples/.gitkeep`, `data/examples/EURUSD_5M_sample.csv`. The file was a byte-identical duplicate of
  `data/raw/EURUSD/EURUSD_5M_sample.csv`. Every reference was repointed (list below). Nothing else depended on it.
* Phantom manifest rows (in `data/data_manifest.json` and `docs/DATA_COVERAGE.md`).

**References handled for `data/examples`:** `app/main.py` (CLI default and docstring), `app/data/downloader.py`
(docstring), `README.md`, `docs/buyer/INSTALL.md`, `docs/buyer/QUICKSTART.md`, `build_tools/package_release.py`,
`build_tools/spec_common.py`, `.github/workflows/web-cross-platform.yml`,
`tests/packaging/test_package_release.py`, and 18 test files that read the sample CSV.

**Keep** (per your standing instruction): everything under `data/raw`, `app/data/downloader.py`,
`app/data/ctrader_export.py`, `app/validation/pre_purchase_gate.py`.

**Safe to delete if you want a leaner repo (your call, I have not touched them):**
`scripts/audit_2026_10/*` (one-off research scripts; their outputs are in `strategies/AUDIT_2026-10_RESULTS.md`),
`reports/EXAMPLE_search_leaderboard.html` (a sample artefact), `HOW_TO_OPEN_ON_YOUR_PHONE.md` only if you no longer
document phone access, and `docs/WEB_PARITY_ROADMAP.md` / `docs/EVOLUTION_LAB_ROADMAP.md` once their items ship.

---

## 7. Verification

* **Full suite on the patched tree: 3,414 passed, 35 skipped, 5 failed (72 min on 2 CPUs).**
* The 5 failures also fail on an untouched clone of your HEAD (run separately), so none comes from this delivery:
  * `tests/packaging/test_review_2026_10_15.py::test_data_gate_flags_unadjusted_roll_gaps_...` — the roll-gap
    functions the test imports do not exist (section 3.2).
  * `tests/ui/test_v911_stop_hang_layout.py::test_stop_during_step2_ga_reaches_terminal_state_fast` — orphan GA worker
    processes after stop.
  * `tests/web/test_web_forge_loop_mode.py::test_forge_loop_mode_end_to_end` — forge job did not finish in 120 s
    (very likely just the 2-CPU sandbox; re-run on your machine).
  * `tests/web/test_web_multi_instrument_search.py::test_multi_instrument_search_loop_mode_save_to_library` — no champion produced.
  * `tests/web/test_web_search.py::test_search_family_grid_python_end_to_end` — expects a `python` family result, got `manual`.
  The last three look like time- or data-sensitive tests; worth a look on your machine, but unrelated to these changes.
* The patch was applied to a fresh clone of your repo; the report, validation-report, packaging, walk-forward and
  headline-flag suites pass there (116 passed).
* Bit-identical check for the vectorised bootstrap; numerical check for the (rejected) drawdown change.
* End-to-end Full Pipeline run on real data (MNQ 1m) completes; the redesigned HTML and PDF were rendered and
  inspected.
