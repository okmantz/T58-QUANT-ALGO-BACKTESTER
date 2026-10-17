# Fresh-Eyes Audit — what changed (2026-10-17)

Everything below is implemented and covered by tests in this drop. Where something could **not** be done, or rests on a
judgement call, it says so.

## 1. Windows CI failure
`encoding="utf-8", errors="replace"` added to the `subprocess.run` calls in `tests/packaging/test_audit_regressions.py` and
`tests/web/test_telemetry_js.py`. The desktop job also sets `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`, and both workflows
now ignore doc-only pushes (`paths-ignore`).

## 3. One Monte Carlo no longer decides 54% of the score
`app/scoring/t58_scorecard.py` (`SCORECARD_VERSION 2026-10-17.2`, weights sum to **100**):

| Evidence | Weight | Source |
|---|---|---|
| pass 10 · payout 8 · ruin −6 · funded survival 6 | **30** (the whole Monte Carlo family) | one resample |
| attempt **replay** pass rate | 15 | real engine, real start dates |
| **expectancy lower bound** | 15 | t / bootstrap, after 2× costs |
| walk-forward stability | 15 | |
| MC robustness 5 · parameter stability 5 · drawdown 8 · parsimony 4 · CPCV 3 | 25 | |

`mc_weight_share` and `missing_independent` are reported on every result. READY now needs the replay and the expectancy
bound to have actually been measured (`evidence_required`); missing evidence caps at MARGINAL and never lifts a score.

## 4. The "95% lower bound" now reflects the trade sample
New `app/scoring/edge_bounds.py`. `effective_independent_attempts` = the smaller of (history days ÷ 30) and (source trades ÷ 20),
never above the simulated count. `MonteCarloResult.per_attempt_pass_ci95_sample` / `per_attempt_paid_ci95_sample` carry a
Wilson interval at that effective n (typically ±10–25 points instead of ±1). The replay's dependence-aware interval now feeds
the score (15 points) instead of only demoting. *Kept deliberately:* the 70% pass **gate** still uses the old interval so the
bar is not loosened silently; the honest interval drives the score and is shown in the reasons.

## 5. Edge size is a hard gate
`expectancy_bound()` → mean R per trade after 2× costs, lower bound = the **more conservative** of t-based and bootstrap.
`_make_verdict`: upper bound < 0 → **NOT READY**; lower bound ≤ `min_expectancy_lb_r` (default 0) → **MARGINAL**.
`min_trades_for_ready` raised 100 → **300** (average R at 100 trades is the same size as the edge being measured).

## 6. Eval breach vs funded breach
`MonteCarloResult.eval_breach_pct` and `funded_breach_within_horizon_pct` (+ `funded_attempts`) are reported separately,
scored separately, and funded breach > 50% (≥30 funded attempts) caps at MARGINAL. The existing combined ruin gate is
unchanged.

## 7. Rules and fees verified against each firm's own pages (read 2026-10-17)

READY now requires a **primary** (firm's own page) stamp ≤ 30 days old (`is_ready_eligible`). Before: 9 unstamped,
5 owner-attested, 1 secondary, 4 primary. After: **14 primary**, 5 deliberately left unstamped.

**Corrections to the previous values**

| Preset | Was → Now |
|---|---|
| lucid_100k | daily loss 2.4% hard-fail → **1.8%, soft lock-day**; no MLL lock → **locks at +$100 offset**; 6 minis |
| apex_50k_eod / 100k_eod | daily loss hard-fail → **soft (pauses the day)**; 30-day limit; contracts 6 / 8 |
| apex_100k_eod / 100k_intraday | payout min winning day $250 / $200 → **$300 / $250** |
| topstep_100k | Combine consistency 50% → **55%** |
| fundednext_25k / 100k | evaluation consistency 40% → **none** (40% is a funded-reward rule); intrabar/floating equity drawdown |
| the5ers (both) | target 8% → **10%**; consistency 30% → **none**; daily loss 5% → **3%** (see conflict below) |
| e8_one_100k | target 10% → **9%**; static → **trailing, locks at start**; minimum days 10 → **0** |
| funderpro_one_phase_100k | minimum days 5 → **0**; added **40%** best-day rule |
| atlas_funded_1step_100k | daily 5% → **4%**; max DD 10% → **7%**; minimum days 5 → **3** |
| futures_desk_50k | trailing → **static**; minimum days 1 → **7**; 20% consistency → none |

**Fees corrected:** FundedNext 25K $189.99 → **$199.99**, 100K $529.99 → **$549.99**; Atlas 100K $478 → **$538**, split 80% → **90%**;
Futures Desk $144 → **$139**, reset $75 → **$99**.

**Still unverified — left unstamped on purpose (READY stays capped for these)**
- **The5%ers** (both): two official pages disagree on the daily loss (5% vs 3%); the stricter 3% is modelled. The preset's **$20k size does not exist** on their site (2.5k/5k/10k/25k/50k/100k).
- **Atlas**: official pages disagree on the target (10% vs 11%).
- **FunderPro**: page does not say whether the 6% drawdown is static or trailing; no $100K fee or EA policy.
- **Futures Desk**: page does not state which account size each plan is.
- **FTMO fees** are behind a login (rules are fine); **Lucid prices** are not on its help centre; **Topstep's** help-centre article fetch was blocked, so its $3,000 / $6,000 targets rest on a dated secondary page.
- Apex's EOD threshold locking differs by platform; the preset models the stricter *never locks*.

**Automation warnings you should act on:** **Apex prohibits automation** outright; **FundedNext $50k+ must be traded manually**;
The5%ers needs written approval first. A bot cannot be READY on those regardless of the numbers.

## 8. Vectorized fast path
- **Trailing stops** now run on the fast path (`trail_distances`, per-bar). It follows the scalar engine's exact order — update best price → resting-stop check → ratchet → stop/take check — including `fit_stop` shrinking the trail. Pinned by trade-by-trade parity tests against `run_execution` (6 cases) plus a guard that the array is not silently ignored. Gap-Aligned and Lucid V2 are now eligible.
- **Null test** (`random_entry_distribution`) evaluates seeds as columns of one vectorised pass (parity-tested to 1e-9 against the engine, p-value identical). It falls back to the engine for a seed with no trades, any batch failure, or a run config the fast path doesn't model (`risk_is_vectorizable`: time stop, session flatten, weekend block, prop account state).
- **Not done:** the **attempt replay** and payout-window replays. Their pass/bust/payout state machine lives only inside `run_execution` (`attempt_mode="single"`), and approximating it in the fast path would give a *different answer*, not a faster one. Routing them needs the prop-account state machine ported to numpy first — a separate, larger job.

## 2. Evolution Lab
New `app/evolution/lab_extras.py` (pure, unit-tested) hooked into `EvolutionRunner`; all switchable in `EvolutionConfig`.

1. **Parity gate** — `attach_parity()` / `POST /api/evolution/lab/parity` compares a candidate's trades with an MT5 or account CSV (entry-time match, P&L correlation, total-P&L gap). Finalists read **NOT READY** until parity passes. *The Lab cannot run MT5 itself; you export the trades and upload them.*
2. **Random controls** — `random_controls_per_generation` (default 3) seeded random-entry strategies ride in every pool through the same filters, are never bred or crowned, and feed a "does the search beat chance?" scoreboard (one-sided Fisher on stage-1 pass rate + best evolved PF vs the 95th control percentile). It says "not enough data" until ≥20 of each.
3. **Tribes** — "consistency" and "growth" breed apart (own elites, own RNG) until `tribe_merge_generation` (default 10), then pool.
4. **15-trade decay check** — a stress survivor whose last 15 trades have gone cold is demoted with a plain-language graveyard note.
5. **Incubation ramp** — each finalist carries a 25% → 50% → 75% → 100% risk schedule (advance only if the last 15 live trades aren't decaying and drawdown is inside the backtest worst case).
6. **Hour-of-week spread** — 168-bucket spread shape; survivors are re-priced at the spread of the hour they traded in and demoted if the edge disappears. *The shape is proxied from bar range ÷ √volume because OHLC data has no quotes; swap in measured spreads when you have them.*

The weights in the tribe scores and the decay/ramp thresholds are documented judgement calls, not fitted values.

**UI:** a live gallery on the Evolution page — one equity-curve card per evaluated mutation (elites outlined, controls dashed), tribe tags, the chance scoreboard, and a finalists/readiness panel with the parity upload. It polls a small in-memory feed only while the tab is visible, keeps ≤48 canvases, and the finalist report runs on its own thread, so it does not slow the rest of the app.

## 9. Consolidation
Repo root is now: `README.md`, `LICENSE.md`, `HOW_TO_OPEN_ON_YOUR_PHONE.md`, `.gitignore` (+ the folders).

| Moved / removed | To |
|---|---|
| `run_web.py`, `run_app.py`, `cli.py`, `run_web.bat`, `run_web.command`, `run_web.sh` | `launchers/` (they put the repo root on `sys.path`; PyInstaller specs, CI and README updated) |
| `render.yaml` | `license_server/render.yaml` — **in Render's Blueprint screen set "Blueprint file path" to `license_server/render.yaml`** |
| `pytest.ini` | **deleted** (it only said `testpaths = tests`; CI always passes the path — run `pytest tests`) |
| `account_settings.json` (runtime file with a password hash) | added to `.gitignore` |

Also safe to delete (no references): `assets/icons/.gitkeep`, `assets/images/.gitkeep`, `assets/logo/.gitkeep`
(the folders are non-empty). The `data/raw` market files and `strategies/` are yours — untouched.
