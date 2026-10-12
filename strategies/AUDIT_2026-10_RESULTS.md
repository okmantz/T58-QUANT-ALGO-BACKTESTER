# Strategy audit -- October 2026

Replaces the stale 2026-08-29 `SCREENING_RESULTS.md` (which described seven files that no longer exist).

**Bottom line: of the 15 strategies that were in this folder, one has any measurable edge, and that edge is
borderline for the app's READY gates. Fourteen were removed. Nothing here is a validated, READY-grade strategy.**

Data: back-adjusted continuous futures 1m bars (NQ, ES, GC) resampled to 5m, Topstep-style $50K rules, engine
costs from the instrument specs. Dev / validation / final splits were kept separate; "OOS" below is the held-out
final slice. Rolling pass-rate = share of real historical start dates that passed the evaluation inside the window.

## Verdicts

| Strategy | Dev PF (n) | OOS PF (n) | Verdict |
|---|---|---|---|
| NQ1! Drive+Fail 5m | 1.18 (886) | 1.15 (208) | **KEEP -> rewritten as `NQ1! Opening Drive 5m`** |
| ES1! Failed Breakout 5m | 1.06 (416) | 1.12 (142) | CUT -- positive but thin (PF 1.06 / 1.12) with a pass rate well below NQ's. Closest runner-up; revisit if more data accumulates. |
| NQ1! Opening Range Breakout 5m | 0.95 (733) | 1.00 (170) | CUT -- no edge |
| NQ1! Trend Vol Filter 5m | 1.00 (664) | 0.86 (140) | CUT -- no edge, OOS negative |
| ES1! Opening Range Breakout 5m | 0.77 (514) | 0.96 (178) | CUT -- negative |
| ES1! Trend Vol Filter 5m | 0.90 (1082) | 0.76 (234) | CUT -- negative |
| GC1! Opening Range Breakout 5m | 0.77 (445) | 0.61 (105) | CUT -- negative |
| GC1! Trend Vol Filter 5m | 0.74 (656) | 0.80 (277) | CUT -- negative |
| GC1! COMEX Drive 5m | 0.98 (639) | 0.99 (217) | CUT -- no edge |
| ml_classifier_direction | 0 - 187 trades, PF 0.58 on GC | -- | CUT (kept only as a test fixture for the walk-forward retrain protocol) |
| ml_classifier_gbdt_direction | PF 0.64 / 0.93 / 0.74 (ES/NQ/GC) | -- | CUT -- loses on every market |
| ml_classifier_random_forest | PF 0.71 on GC, 13 trades on NQ | -- | CUT -- loses or does not trade |
| 3 manual JSON strategies | -- | -- | CUT -- boolean-flag/GA artefacts, no edge; see docs/AUDIT_2026-10_REPORT.md |

## The one keeper: `NQ1! Opening Drive 5m`

Changes from Drive+Fail: stop = 1.0x opening range in both modes, no fixed target (rides to the 15:45 ET exit),
22-pt stop cap removed (run on MNQ with `micro_fallback` sizing so wide stops still fit the risk budget).

Honest status:

* Chosen after **18 variants** on the dev/validation split, on top of ~1,800 earlier grid evaluations. Selection bias is real.
  Set `external_trial_count=1800` (or higher) when running the Full Pipeline so DSR/PBO see it.
* Edge: ~0.19 R/trade all-sample, **~0.085 R/trade on the final holdout**. Roughly 0.2 is needed to clear the pass-probability
  gates comfortably.
* The edge **decays**: on 1x MNQ the Topstep $50K rolling 30-day pass rate is ~10.6% over all history but 3.8% from 2025 on;
  60/120/250-day windows from 2025 on are 0%. Lucid $50K holds up better (~24% from 2025 at 2x-3x MNQ), but Lucid's rules
  rest on a secondary source -- verify on lucidtrading.com before paying.
* **Full Pipeline result for the rewrite: NOT READY.** Run on the now-verified `topstep_50k` preset, 1 MNQ fixed contract,
  `skip_optimization`, `external_trial_count=1800`. Hard gate failed: Monte Carlo risk of ruin **60.8%** vs the 20% cap
  (the pre-rewrite file scored 63.4%). Other readings: ~60% per-attempt evaluation pass across the reset chain; 8-start
  attempt replay 62% passed / 12% busted (95% CI 25-100%); **0% reached a first payout inside 30 days** on both the dev
  and holdout replays; the ICIR / Bonferroni gate did not pass; EV verdict INSUFFICIENT EVIDENCE (no `FIRM_FEES` row for
  Topstep -- add one with a source URL and date). Lookahead check passed.
* Why ruin is high: one trade at a 1.0x-OR stop risks ~$480 on 1 MNQ, a quarter of Topstep's $2,000 trailing MLL, and
  1 micro is the smallest size there is. The stop width, not the edge, is what the ruin gate is punishing. Tighter-stop
  variants lose most of the edge (that is why the 22-pt cap version was replaced), so this is a genuine trade-off.

Do not buy an evaluation on the strength of this file. Paper-trade it first.
