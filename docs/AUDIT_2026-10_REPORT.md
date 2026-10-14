# T58 audit report -- 11 October 2026

Scope: whole repo, the 15 shipped strategies (3 manual JSON, 9 rule-based Python, 3 ML), and whether the app can
"genuinely find, test and validate" a strategy with an edge for prop-firm use.

## 1. Headline

1. **The validation machinery is largely sound; the strategies are not.** The pipeline correctly refused every strategy
   it was given. Of 15, one has any measurable edge (NQ opening-range drive/fade) and even that one fails the ruin gate.
2. **I found and fixed one real bug that inflated failures** (rolling evaluation) and corrected one preset that was
   both inaccurate and unverified (Topstep $50K). Both have regression tests.
3. **Nothing in this repo will currently produce a READY verdict, and that is the correct output.** The honest path to
   READY is a better edge, not looser gates. Do not lower the gates.

## 2. What was fixed (all in the zip)

| # | Problem | Fix | Files |
|---|---|---|---|
| 1 | **Rolling-evaluation bug.** `simulate_account` runs the whole remaining trade tail, so a funded-stage bust *after* a successful pass was relabelled an evaluation failure for every window long enough to contain it. Pass rates therefore *fell* as the window got longer, which is impossible (a longer window only adds chances to pass). | A pass inside the window now takes precedence; only a failure before (or without) a pass counts against the evaluation. | `app/prop/rolling_evaluation.py`, `tests/test_rolling_evaluation.py` (+3 tests; the pass-then-bust test fails on the old code) |
| 2 | **Topstep $50K preset inaccurate and unverified**, so the pipeline could never return READY for it ("UNVERIFIED FIRM RULES"). Missing trailing lock, wrong consistency %, wrong open-loss handling. | `trailing_lock=True` (MLL stops trailing at the starting balance), consistency 55%, `floating_drawdown_mode="adverse"`, stamped `verified_on=2026-10-11` + source URL, note records the 5-mini / 50-micro contract cap. | `app/prop/presets.py`, 2 tests updated that encoded the old values |
| 3 | **Hand-typed / imported manual configs** could compare a 0/1 flag (`time_of_day`) to a non-0/1 constant, creating a branch that never fires. The GA already freezes these genes (your earlier fix); nothing caught the config if it arrived any other way. | `validate_bounded_conditions` now flags boolean operands compared to non-boolean constants, surfaced through the existing "can never be true" warning path. | `app/strategy/manual.py`, `tests/test_bounded_oscillator_genes.py` (+3 tests) |
| 4 | **Stale docs.** `SCREENING_RESULTS.md` described seven strategies that no longer exist and was still cited. | Replaced by `strategies/AUDIT_2026-10_RESULTS.md`; README and `docs/ACCURACY_OVERHAUL.md` updated. | `strategies/*`, `docs/ACCURACY_OVERHAUL.md` |
| 5 | **Dead strategies and their scaffolding** (14 strategies, their generator script, tests that loaded them). | Removed via `cleanup.sh`; the ML retrain-protocol tests now use a fixture copy of the one ML file; the NQ ORB manual JSON used by a gene test is now `tests/fixtures/nq_orb_manual.json`. | see `cleanup.sh` |

## 3. Still open (not changed -- needs your decision or an outside fact)

| Priority | Issue | Why it matters | Recommended action |
|---|---|---|---|
| High | **No `FIRM_FEES` row for Topstep.** The EV verdict returns INSUFFICIENT EVIDENCE for it. | No economic verdict (is the evaluation fee worth it) is possible without it. | Add the Combine price and reset fee with a source URL and date. I did not invent a price. |
| High | **Selection bias is only as good as the trial count you enter.** The pipeline can't know about the ~1,800 grid evaluations and 18 variants in `scripts/audit_2026_10`. | DSR / PBO / Bonferroni will flatter any strategy chosen from a large search unless `external_trial_count` is set. | Always set `external_trial_count` to the real number of configurations tried before the pipeline run. |
| High | **Lucid preset rests on a secondary source** (tradetanto.com plus price aggregators). | A READY verdict on a firm whose rules were not read from the firm. | Confirm on the firm's own site before paying; the preset is stamped `secondary`. |
| Med | **Monte Carlo "risk of ruin" counts any breach of a purchased account, including after payout.** | It is stricter than "will I pass the evaluation" (e.g. ~60% per-attempt pass but 60.8% ruin for the NQ keeper). Not a bug -- a different question -- but it is easy to misread. | Keep the gate; read per-attempt pass and ruin together. |
| Med | **Unadjusted roll gaps / synthetic sample files** in `data/raw` (AAA/BBB/EURUSD samples; manifest lists 11 missing files). | Gaps at contract rolls can fake a stop-out or a runaway trade; manifest drift is a warning on every run. | Re-import continuous contracts fully back-adjusted; regenerate the data manifest; keep synthetic samples out of `data/raw`. |
| Med | `rolling_evaluation`: `was_making_progress` (profit at window close) is a heuristic, and `reached_first_payout` is not limited to the window. | Only labels *why* a window failed / flags payouts; does not change pass/fail. | Advisory. |
| Med | **Sizing preflight blocks wide-stop strategies** under `skip` ("50% of signals skipped"). | Correct behaviour, but it hides the real constraint: a $2,000 MLL cannot absorb a 240-pt stop at 1 micro. | Use `fixed_contracts`/`micro_fallback` knowingly; treat stop width vs MLL as a design input. |
| Med | **Pre-existing test failures (identical on the untouched repo):** `test_research_agent` (6, needs a local Ollama), `test_strategy_space` (4 families), `test_speed_optimizations::test_ga_flatline_stop_can_be_disabled`, `test_audit_findings_2026_09::...no_cooldown_behavior`. Also `test_translator_targets` globs `strategies/manual/*.json`, and other tests write `searchlab_*` files (and `.meta.json` sidecars) into that folder first, so it fails in a full run. | A red suite hides real regressions. | Make the translator test use a tmp dir / exclude `*.meta.json`; make tests that write to `strategies/` use a temp library; triage the 4-family and cooldown failures separately. None are caused by this audit's changes. |
| Low | `test_alpaca_source` fails only because `alpaca-py` is not installed here. | Environmental. | Install it or skip the test in CI. |

## 4. Why nothing reaches READY, and what would

The ruin gate is a barrier-hitting problem: with a trailing MLL of $2,000 on a $50K account, the probability of
ever being stopped out depends on the per-trade Sharpe and on how large one losing trade is relative to the MLL.
A synthetic strategy with a per-trade Sharpe around 0.2 and bounded risk clears every gate (I tested this), so the
gates are *reachable* -- they are simply demanding. What the real NQ strategy shows:

* Edge about **0.19 R/trade all-sample, ~0.085 R/trade on the final holdout** -- roughly half of what is needed.
* One trade at a 1.0x-opening-range stop risks ~$480 on 1 MNQ, i.e. a quarter of the MLL. Tightening the stop
  (the original 22-pt cap) removed most of the edge.
* The edge is decaying: Topstep 30-day rolling pass rate 10.6% (all history) vs 3.8% (2025 onward).

**What would move it to READY:** a second, uncorrelated edge on a different market (ES/GC edges were negative or
nil), or a materially better signal on NQ (more selective entries with a tighter, structurally meaningful stop).
Both are research, not configuration. Parameter-grid search on the same data will not get there honestly.

## 5. Strategy decisions

See `strategies/AUDIT_2026-10_RESULTS.md` for the table. In short:

* **KEEP + improve:** NQ Drive+Fail -> `NQ1! Opening Drive 5m` (stop 1.0x OR, no fixed target, 15:45 ET exit, run on MNQ).
* **CUT:** everything else. ES Failed Breakout is the nearest runner-up (PF 1.06 dev / 1.12 OOS) but too thin to carry costs.
* **ML strategies:** CUT. They lose on every market (PF 0.58-0.93) or barely trade. A classifier on lagged OHLC features
  of 15m bars has no structural reason to beat costs; one file is retained purely as a test fixture.

## 6. Full Pipeline result for the keeper

NOT READY. Risk of ruin 60.8% (cap 20%); attempt replay 62% passed / 12% busted of 8 starts; 0% first payout inside 30
days on dev and holdout replays; ICIR / Bonferroni gate not passed; lookahead check passed; EV verdict INSUFFICIENT
EVIDENCE (no fee row). Run config: `topstep_50k` (now verified), 1 MNQ fixed, `skip_optimization`, `external_trial_count=1800`.
