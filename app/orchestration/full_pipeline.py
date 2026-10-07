"""
Full Pipeline -- one button that runs everything.

Every other tab in this app is a separate tool: Run & Report backtests
one fixed configuration, Iterative Refinement tunes it in-sample,
Walk-Forward-Aware GA tunes it against out-of-sample folds, Validation
Lab checks robustness after the fact. Getting from "here's a strategy
file" to "here's the best, validated version of it, ready for a prop
firm" means running several of those in the right order and carrying
the winner from one into the next by hand.

This module does that hand-off automatically:

    Step 1  Baseline           -- one backtest of the strategy exactly as
                                   given, plus a lookahead-bias check for
                                   code strategies. Fails fast (in under a
                                   second) if this produces zero trades,
                                   rather than wasting minutes discovering
                                   that fact three more times over.
    Step 2  Robust optimization -- app.optimize.walkforward_ga's GA, which
                                   scores every candidate ONLY on chained
                                   out-of-sample fold performance (never
                                   in-sample), specifically so the "best"
                                   configuration it finds is one that
                                   generalizes rather than one that just
                                   curve-fits the baseline harder. Skipped
                                   gracefully (not a failure) if the
                                   strategy has no tunable numeric
                                   parameters -- the baseline then IS the
                                   final configuration.
    Step 3  Final validation    -- the winning configuration is re-run
                                   through a full backtest, prop-firm
                                   simulation, and full-fidelity Monte
                                   Carlo on `dev_df` -- the same
                                   pre-holdout slice Steps 1-2 already
                                   used (more simulations than the search
                                   phase used, since this is the one that
                                   counts) -- NOT the whole dataset; see
                                   Step 5 below and
                                   FullPipelineConfig.reserve_true_holdout
                                   for where the genuinely-unseen tail
                                   comes in. (DOC-003: this step used to
                                   be described as running on "the whole
                                   dataset," which was true before the
                                   circularity fix landed but never
                                   updated afterward -- the code has been
                                   correct since; only this docstring was
                                   stale.)
    Step 4  Out-of-sample check -- app.search.robustness.run_walk_forward
                                   on the exact winning configuration, NO
                                   further re-tuning: does this exact
                                   strategy keep working across several
                                   distinct historical stretches?
    Step 5  Holdout check       -- the same chronological in-sample/
                                   holdout split every other pipeline run
                                   in this app already does -- and, as of
                                   the circularity fix (see
                                   FullPipelineConfig.reserve_true_holdout
                                   and CIRCULARITY_AUDIT.md), a GENUINE
                                   one: Steps 1-4 above only ever see the
                                   first (1 - holdout_frac) of the data,
                                   so this step is the first time the
                                   final configuration is evaluated
                                   against bars that had no chance to
                                   influence its own selection.
    Step 6  Report + save       -- one full HTML/JSON report (the same
                                   generate_full_report every other run
                                   produces) for the FINAL strategy, a
                                   plain-language verdict, and -- for code
                                   strategies -- the winning source saved
                                   straight into the Strategy Library,
                                   tagged "validated", ready to hand to a
                                   prop firm or plug back into the app.

Every step is best-effort past Step 2: a step that can't run (e.g. not
enough bars for a walk-forward check) is recorded as skipped with a
reason, never allowed to take down a run that otherwise succeeded.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import pandas as pd

from app.backtest.engine import BacktestResult, run_backtest, run_holdout_comparison
from app.backtest.adaptive_risk import build_limit_aware_preset
from app.backtest.statistics import format_run_summary_line
from app.data.timeframe_resample import describe_resolved_timeframe
from app.backtest.risk import (
    RiskConfig,
    account_size_mismatch_message,
    has_impossible_condition,
    has_instrument_scale_mismatch,
    position_sizing_deviation_message,
    with_prop_safety_defaults,
)
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, default_method_for_adaptive_risk, run_monte_carlo
from app.optimize.code_parameter_space import patched_source_for_strategy
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig, preflight_signal_check
from app.optimize.walkforward_ga import WalkforwardGAResult, run_walkforward_aware_refinement
from app.orchestration.resource_guard import safe_worker_count
from app.prop.simulator import AccountSimResult, PropRules, simulate_account
from app.reports.crash_log import log_crash
from app.scoring.parsimony import ParsimonyResult, compute_parsimony
from app.scoring.t58_scorecard import T58ScorecardResult, score_from_results
from app.search.robustness import (
    DeflatedSharpeGateResult,
    WalkForwardResult,
    count_all_trials,
    deflated_sharpe_gate,
    run_walk_forward,
)
from app.search.strategy_space import build_strategy_from_spec
from app.strategy.base import Strategy
from app.strategy.library import StrategyAlreadyExists, provenance_stamped_name, safe_filename_stem, \
    save_strategy_replacing_version, save_strategy_text, set_strategy_status, record_backtest_result, \
    record_optimize_result, record_validation_result, record_champion_check_result
from app.validation.cpcv import CPCVError, CPCVResult, PBOGateResult, compute_pbo, pbo_gate, run_cpcv
from app.validation.icir import ICIRGateResult, run_icir_gate_from_backtest
from app.validation.regime_matrix import RegimeMatrixResult, build_regime_matrix

ProgressCallback = Callable[[str], None]


@dataclass
class FullPipelineConfig:
    n_folds: int = 4
    window_mode: str = "rolling"           # for the GA's internal fold split
    ga_population: int = 12
    ga_generations: int = 6
    ga_search_mc_sims: int = 200
    # UPGRADE (optimizer core): see QuickOptimizeConfig.optimizer_mode's
    # identical field/docstring -- same default, same downstream effect.
    optimizer_mode: str = "genetic"
    fitness_metric: str = "eval_pass_probability"
    final_mc_sims: int = 10_000
    # Step 1's baseline Monte Carlo run is diagnostic only -- it's logged
    # and reported alongside the final numbers, but never feeds the
    # verdict (see _make_verdict, which only reads final_mc) and gets
    # thrown away entirely whenever Step 2 finds a better configuration.
    # Running it at the same full_mc_sims fidelity as the run that
    # actually counts wasted a meaningful share of every pipeline run for
    # no quality benefit, so it defaults lower here -- this is a pure
    # speed change with no effect on the report's headline numbers.
    baseline_mc_sims: int = 2_000
    holdout_frac: float = 0.2
    # Circularity fix (see CIRCULARITY_AUDIT.md): when True (default),
    # the LAST holdout_frac fraction of `df` is carved off and hidden
    # from every step that selects or scores parameters -- Step 1's
    # baseline, Step 2's GA search (and therefore every fold it builds
    # via app.validation.walk_forward_opt.build_folds), Step 3's final
    # re-validation, Step 4's post-hoc walk-forward check, and CPCV (if
    # enabled) -- all run on this smaller "dev" slice only. Step 5's
    # run_holdout_comparison then re-splits the ORIGINAL, full `df` by
    # this same holdout_frac, which reproduces the exact dev/holdout
    # boundary above -- so its holdout half is, for the first time,
    # genuinely never seen by anything upstream of it. Before this flag
    # existed, Step 2's chained-OOS fold search already spanned the
    # entire dataset (folds walk forward to the end of `df`), so Step
    # 5's "holdout" overlapped data the GA had already selected against
    # -- see CIRCULARITY_AUDIT.md Finding 1 for the full trace. Setting
    # this False restores that old (circular) behavior, e.g. for
    # comparing against a report generated before this fix.
    reserve_true_holdout: bool = True
    oos_check_folds: int = 4               # for the post-hoc run_walk_forward check
    oos_check_metric: str = "eval_pass_probability"
    random_seed: int | None = 42
    # Adaptive, limit-aware position sizing (see app.backtest.adaptive_risk):
    # when enabled, every backtest this pipeline runs -- the baseline, the
    # walk-forward-aware GA's own search, and the final validated run --
    # uses a graduated risk-throttle preset derived from THIS run's own
    # PropRules (cut size as the account nears its daily-loss/drawdown
    # floor; optionally lock in a good day's profit). Off by default so
    # existing behavior/reports are unchanged unless explicitly turned on.
    # This directly targets the eval_pass_probability objective itself,
    # independent of whatever edge the strategy has.
    adaptive_risk_enabled: bool = False
    adaptive_risk_daily_profit_lock_pct: float | None = 80.0
    save_to_library: bool = True           # code strategies only -- manual configs aren't files
    library_status: str | None = None      # None = auto-pick from the READY/MARGINAL/NOT READY
                                            # verdict (see _VERDICT_TO_LIBRARY_STATUS below);
                                            # pass an explicit STRATEGY_STATUSES value to always
                                            # use that one regardless of verdict.
    # Passed straight through to Step 2's walk-forward-aware GA (by far
    # the most expensive step in a typical run) -- see
    # app.optimize.walkforward_ga.run_walkforward_aware_refinement for
    # what these control. parallel=True (the default) is a pure speed
    # change: it evaluates a generation's candidates across worker
    # processes instead of one at a time, with automatic fallback to a
    # single process if that can't be set up, so it never changes which
    # configuration wins.
    parallel_search: bool = True
    parallel_search_max_workers: int | None = None

    # -- Pipeline reorg: scorecard verdict + ruin hard gate --------------
    # The ONE legitimate hard gate (pipeline reorg plan section 2/19):
    # risk of ruin has a genuinely binary quality that the other metrics
    # don't, so it's checked BEFORE scoring rather than folded in as just
    # another weighted component. A strategy failing this gate is always
    # "NOT READY" regardless of how good everything else looks -- ruin
    # still also appears INSIDE the scorecard below (as a ranking signal
    # among strategies that already passed this cap), it's just no
    # longer the only thing standing between "READY" and "NOT READY".
    risk_of_ruin_cap: float = 20.0

    # -- Pipeline reorg: minimum trade-count floor for READY (2026-09-24) -
    # Found via an external RoboQuant comparison: a strategy that scored
    # well enough for the "READY" tier on just 40 trades over a 6-year
    # window (one trade responsible for a third of its entire gross
    # profit, and too few distinct periods for the ICIR/Bonferroni
    # significance gate to even run) is not something the T58 Score alone
    # protects against -- that score is about the QUALITY of the evidence
    # it was given, not whether there was ENOUGH of it. This is a second,
    # separate hard floor (not scored/weighted like the rest -- either the
    # sample is big enough to trust or it isn't): a strategy that would
    # otherwise land on "READY" is capped at "MARGINAL" when its final
    # backtest's total_trades falls short of this, with the reason spelled
    # out in verdict_reasons. Does not affect MARGINAL/NOT READY verdicts
    # (a strategy already below READY isn't made worse by a thin sample --
    # that's already reflected in its score).
    min_trades_for_ready: int = 100

    # -- Lock-parameters / fixed-backtest mode (2026-09-24) --------------
    # Found via an external RoboQuant comparison: comparing "the same
    # strategy" between T58 and another tool is only valid when NEITHER
    # side silently re-optimizes it -- Full Pipeline's GA search (Step 2)
    # otherwise always tries to improve on whatever was supplied, which
    # is exactly right for finding a strategy but wrong for reproducing
    # or cross-checking one you already have fixed parameters for. When
    # True, Step 2 is skipped entirely (same "skipped" bookkeeping/report
    # fields as the existing pip-scale-mismatch/impossible-condition
    # fast-skips just below use) and every later step runs against the
    # SUPPLIED strategy exactly as given -- final_parameters/baseline_
    # parameters stay empty and the report title is never tagged
    # "GA-modified" (see _finish's mutated_by_ga), because nothing was
    # mutated. Steps 1, 3-7 (baseline backtest, re-validated Monte Carlo,
    # OOS/holdout checks, verdict) still run normally against that fixed
    # configuration.
    skip_optimization: bool = False

    # -- Pipeline reorg: one canonical robustness test (Option A) --------
    # "walk_forward" (default -- unchanged behavior) keeps using Step 4's
    # existing run_walk_forward result as the scorecard's
    # walk_forward_stability component. "cpcv" instead runs CPCV/PBO
    # (app.validation.cpcv) as the PRIMARY generalization test and feeds
    # ITS efficiency into that same scorecard slot -- the two are never
    # both scored as primary at once, so robustness is never counted
    # twice under two different names (pipeline reorg plan section 11).
    primary_robustness_method: str = "walk_forward"   # "walk_forward" | "cpcv"
    # When True AND primary_robustness_method is "walk_forward", CPCV
    # additionally runs as a SECOND, low-weight SUPPORTING diagnostic
    # (scorecard's cpcv_supporting component, 5 points) rather than a
    # gate -- lets you see whether CPCV's more expensive multi-path
    # check actually changes anything before promoting it to primary.
    # Ignored (never runs) when primary_robustness_method is "cpcv",
    # since that would double-count the same evidence twice.
    cpcv_supporting_enabled: bool = False
    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2
    # -- Validation hard gates (v5) --------------------------------------
    # Walk-forward, CPCV, and the ICIR/significance gate used to be
    # diagnostic-only in this pipeline (scored or noted, never rejecting)
    # while the only hard gates were lookahead-bias and risk-of-ruin. They
    # are now hard gates: a candidate that FAILS any enabled one is NOT
    # READY, with the reason recorded in verdict_reasons -- a strong
    # backtest that doesn't generalize is exactly what these tests exist
    # to catch, and "strong but fragile" must not be crowned.
    # validation_gates_advisory_only=True restores the old behavior for
    # debugging: gate failures are still computed and recorded in
    # verdict_reasons (prefixed ADVISORY) but no longer force NOT READY.
    validation_gates_advisory_only: bool = False
    # DSR gate: the champion's Sharpe, deflated for how many configurations
    # the search actually evaluated -- count_all_trials() counts the baseline
    # PLUS every GA genome the inner loop backtested (not just leaderboard
    # survivors), fixing the old optimistic undercount. Rejects when the
    # probabilistic Sharpe falls below dsr_min_probabilistic_sharpe. 0.95 is
    # Bailey & de Prado's recommended bar and matches deflated_sharpe_ratio's
    # own default -- a gate should be no looser than the diagnostic it
    # promotes.
    dsr_gate_enabled: bool = True
    dsr_min_probabilistic_sharpe: float = 0.95
    # D4(b): hand-tuned / externally-evaluated configurations that never
    # went through this pipeline's own GA still count as trials the
    # searcher "had a chance to get lucky on" -- without this, a
    # hand-tuned strategy arrives at the DSR gate with n_trials=1 (just
    # the baseline), which understates the multiple-testing burden and
    # makes the deflated Sharpe optimistic. Set to the number of
    # distinct configurations evaluated outside this pipeline (manual
    # spot-checks, AI-suggested candidates, earlier runs); it is threaded
    # into count_all_trials(..., extra_trial_counts=(...)) at the DSR
    # step. 0 (default) = this pipeline saw every evaluation itself.
    external_trial_count: int = 0
    # PBO gate: the genuine multi-candidate Probability of Backtest
    # Overfitting (Bailey et al. 2017) over the GA's final-generation
    # leaderboard -- "did the SELECTION process pick signal or noise?".
    # Rejects when PBO exceeds pbo_max (0.5 = the coin-flip line: above it
    # the in-sample winner lands in the bottom half out-of-sample more
    # often than random -- see app.validation.cpcv.pbo_gate).
    # pbo_max_candidates caps the pool (leaderboard, best-first) and
    # pbo_max_paths caps the CPCV paths, bounding the extra backtests this
    # adds (candidates x paths x 2 backtests each).
    pbo_gate_enabled: bool = True
    pbo_max: float = 0.5
    pbo_max_candidates: int = 8
    pbo_max_paths: int = 10
    # D1: the holdout gate rejects (NOT READY) when the holdout check RAN
    # with at least this many trades AND lost money (net < 0 or profit
    # factor < 1.0) while the in-sample run traded. Below this trade
    # count the holdout is too thin to convict -- it stays UNPROVEN (the
    # empty-holdout MARGINAL cap still applies), not FAILED.
    min_holdout_trades_for_gate: int = 10
    # B2(d): the acceptance verdict gates on the PER-ATTEMPT pass odds,
    # not the inflated chain-level number. A strategy whose per-attempt
    # pass CI lower bound (fallback: per-attempt point estimate) sits
    # below this bar cannot be called READY -- it is a hard validation
    # gate, so the verdict is NOT READY (advisory mode records without
    # rejecting). 70.0 is the bar historically used for "high chance of
    # passing an eval". The gate is skipped when the MC result carries
    # no per-attempt information at all (legacy results predating the
    # v6 MC fields) -- a gate that can't measure can't convict, the
    # same couldn't-run/UNPROVEN distinction the ICIR gate makes.
    min_per_attempt_pass_pct: float = 70.0
    # Regime performance stays purely informational (pipeline reorg plan
    # section 13/14) -- attached to the report/result for a human to
    # read, never scored or gated. Reuses the SAME trades final_bt
    # already produced (build_regime_matrix, not run_regime_matrix) so
    # this costs no extra backtest -- cheap enough to default on.
    regime_diagnostics_enabled: bool = True

    # UPGRADE (prop-firm reset-on-breach as the search basis): threaded
    # into Step 1's baseline MC, Step 2's GA search (via mc_config), and
    # Step 3's final re-validated MC/single-run below, so a blown account
    # is scored the way a real prop trader would actually handle it --
    # reset and keep going -- instead of as a dead end. False (default)
    # is byte-identical to every run before this field existed; the web/
    # desktop Full Pipeline form defaults its own checkbox to CHECKED.
    reset_on_breach: bool = False

    # VERSIONING (stop-making-copies fix): same fields/contract as
    # QuickOptimizeConfig.library_ref/replace_existing (see that class's
    # docstring for the full rationale) -- when the strategy Full
    # Pipeline is running came from the Strategy Library, library_ref is
    # its (strategy_type, filename), and replace_existing=True makes the
    # save step in _finish overwrite that SAME file in place (archiving
    # the version being replaced, carrying forward last_run/
    # last_optimize/last_validation/etc.) instead of writing yet another
    # provenance-stamped copy. False (default) is byte-identical to every
    # run before these fields existed.
    library_ref: tuple[str, str] | None = None
    replace_existing: bool = False


@dataclass
class FullPipelineResult:
    strategy_source_type: str
    strategy_display_name: str

    baseline_bt: BacktestResult
    baseline_single_run: AccountSimResult
    baseline_mc: MonteCarloResult
    lookahead_summary: str | None

    refinement_ran: bool
    refinement_skip_reason: str | None
    ga_result: WalkforwardGAResult | None

    final_source_type: str
    final_config: dict | None
    final_code_text: str | None
    final_code_extension: str | None

    final_bt: BacktestResult
    final_single_run: AccountSimResult
    final_mc: MonteCarloResult
    final_holdout: dict | None

    # Full-history (complete df, not just dev_df) backtest/single-run used
    # for the HTML report's chart and payout stats -- see the comment above
    # full_history_bt's own computation. Equal to final_bt/final_single_run
    # whenever reserve_true_holdout is off (dev_df already is the full df).
    full_history_single_run: AccountSimResult

    oos_validation: WalkForwardResult | None
    oos_validation_skip_reason: str | None

    icir_gate: "ICIRGateResult | None"
    icir_gate_skip_reason: str | None

    verdict: str                # "READY" | "MARGINAL" | "NOT READY"
    verdict_reasons: list[str]

    # -- Pipeline reorg additions -----------------------------------------
    scorecard: "T58ScorecardResult | None"       # the continuous 0-100 score/tier behind `verdict`
    risk_of_ruin_hard_fail: bool                 # True if verdict is NOT READY solely because of the ruin cap
    lookahead_hard_fail: bool                    # True if verdict is NOT READY solely because of a confirmed lookahead-bias leak
    risk_of_ruin_cap: float                      # the cap that was actually in force for this run (FullPipelineConfig.risk_of_ruin_cap) -- kept on the result so downstream guidance (pipeline_guide.after_full_pipeline) can quote the exact threshold a rejected strategy missed, even if a caller ran with a non-default cap
    parsimony: "ParsimonyResult | None"
    cpcv_result: "CPCVResult | None"             # populated if primary_robustness_method=="cpcv" OR cpcv_supporting_enabled
    cpcv_skip_reason: str | None
    regime_result: "RegimeMatrixResult | None"   # report-only diagnostic, never gated/scored
    regime_skip_reason: str | None

    saved_library_path: Path | None
    saved_library_note: str | None

    report_paths: dict
    elapsed_seconds: float
    warnings: list = field(default_factory=list)
    # RISK-001: set when the caller's RiskConfig.initial_balance didn't
    # match PropRules.account_size before this run -- see
    # app.backtest.risk.account_size_mismatch_message. Also present in
    # `warnings` above; broken out here so callers/UI can give it the
    # same dedicated prominence Quick Optimize gives its own copy of
    # this field.
    account_mismatch_warning: str | None = None
    # -- v5 validation hard gates -----------------------------------------
    dsr_gate_result: "DeflatedSharpeGateResult | None" = None
    dsr_skip_reason: str | None = None
    pbo_gate_result: "PBOGateResult | None" = None
    pbo_skip_reason: str | None = None
    # Names of the validation gates (walk_forward / cpcv / icir / dsr /
    # pbo) whose failure forced this run's NOT READY verdict -- empty when
    # the verdict came from the scorecard or another hard gate. Lets
    # callers/UI say exactly which filter rejected the candidate without
    # re-parsing verdict_reasons.
    validation_gate_failures: list = field(default_factory=list)


def _display_name(strategy: Strategy) -> str:
    if strategy.source_type == "manual":
        return strategy.config.get("name", "Manual Strategy")
    if strategy.source_type == "python":
        return Path(strategy.file_path).stem
    if strategy.source_type == "pinescript":
        return "PineScript Strategy"
    if strategy.source_type == "mql5":
        return "MQL5 Strategy"
    return "Strategy"


def _spec_for_manual(config: dict) -> dict:
    return {"source_type": "manual", "config": config}


def _spec_for_code(source_type: str, code_text: str, extension: str) -> dict:
    return {"source_type": source_type, "code_text": code_text, "code_extension": extension}


def _holdout_untestable_note(holdout: dict | None) -> str | None:
    """A holdout that produced ZERO trades while the in-sample run traded
    proves nothing either way -- it is not a pass. Before this check the
    verdict was silent about it, so a strategy could show a strong headline
    (in-sample backtest + Monte Carlo resampled from those same trades) with
    an empty out-of-sample check and nothing in the report saying so. The
    usual cause on futures is position sizing flooring every entry to 0
    whole contracts in the holdout period (price level / volatility rose, so
    one contract's stop risk now exceeds the per-trade risk budget)."""
    if not holdout:
        return None
    hs = holdout.get("holdout_statistics") or {}
    ins = holdout.get("in_sample_statistics") or {}
    if (hs.get("total_trades") or 0) == 0 and (ins.get("total_trades") or 0) > 0:
        return (
            "HOLDOUT UNTESTED: the strategy took 0 trades in the untouched holdout period "
            f"({(holdout.get('holdout_period') or ['?', '?'])[0]} to {(holdout.get('holdout_period') or ['?', '?'])[1]}) "
            f"after taking {ins.get('total_trades')} in-sample, so the holdout confirms nothing. Check the "
            "execution warnings: if entries were skipped because sizing rounded down to 0 contracts, the "
            "stop is too wide for your risk budget at that period's price/volatility (tighten the stop, "
            "raise risk-per-trade, or use a smaller contract)."
        )
    return None


def _leaderboard_candidate_spec(cand, final_source_type: str) -> dict:
    """Builds a compute_pbo()/DSR-trial-ready candidate spec from one
    WalkforwardGACandidate (the GA's final-generation leaderboard).

    WalkforwardGACandidate doesn't carry source_type -- but a GA run never
    mixes source types, so the run's own final_source_type applies to every
    leaderboard genome. Manual genomes carry .config; code genomes carry
    .code_text/.code_extension."""
    if cand.config is not None:
        return {"source_type": "manual", "config": cand.config}
    if final_source_type == "manual":
        raise ValueError("GA leaderboard candidate has no config for a manual-type run.")
    return {
        "source_type": final_source_type,
        "code_text": cand.code_text,
        "code_extension": cand.code_extension,
    }


def _pnl_skew_kurtosis(pnls: list[float]) -> tuple[float, float]:
    """Trade-PnL skew/kurtosis for the DSR gate's PSR denominator -- Sharpe
    ratios on fat-tailed, skewed trade distributions (typical for
    short-RR, high win-rate prop strategies) are noisier than the same
    Sharpe on a symmetric distribution, and the PSR formula accounts for
    that directly. Falls back to the normal defaults (0.0 / 3.0) when
    there isn't enough data to estimate."""
    import numpy as np

    arr = np.asarray([p for p in pnls if math.isfinite(p)], dtype=float)
    if arr.size < 4:
        return 0.0, 3.0
    mean = arr.mean()
    sd = arr.std(ddof=1)
    if sd <= 0:
        return 0.0, 3.0
    z = (arr - mean) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    if not (math.isfinite(skew) and math.isfinite(kurt)):
        return 0.0, 3.0
    return skew, kurt


def _validation_gate_failure_names(verdict_reasons: list[str]) -> list[str]:
    """Extracts which validation gates rejected the candidate from the
    recorded HARD VALIDATION GATE FAILED reasons -- lets callers/UI name
    the failing filter(s) without re-parsing prose."""
    names = []
    for r in verdict_reasons or []:
        if "HARD VALIDATION GATE FAILED" not in r:
            continue
        low = r.lower()
        if "walk-forward" in low:
            names.append("walk_forward")
        elif "cpcv" in low:
            names.append("cpcv")
        elif "icir" in low:
            names.append("icir")
        elif "deflated sharpe" in low:
            names.append("dsr")
        elif "pbo gate" in low:
            names.append("pbo")
    # de-dupe, preserving order
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def _make_verdict(
    final_mc: MonteCarloResult,
    oos_validation: WalkForwardResult | None,
    icir_gate: "ICIRGateResult | None" = None,
    statistics=None,
    prop_rules: PropRules | None = None,
    risk_of_ruin_cap: float = 20.0,
    parsimony: "ParsimonyResult | None" = None,
    cpcv_primary_result: "CPCVResult | None" = None,
    cpcv_supporting_result: "CPCVResult | None" = None,
    lookahead_bug_detected: bool = False,
    min_trades_for_ready: int = 100,
    holdout: dict | None = None,
    # D1: minimum holdout trade count for the holdout gate to count as
    # "ran" (see FullPipelineConfig.min_holdout_trades_for_gate).
    min_holdout_trades_for_gate: int = 10,
    # B2(d): minimum per-attempt pass odds for the acceptance verdict
    # (see FullPipelineConfig.min_per_attempt_pass_pct).
    min_per_attempt_pass_pct: float = 70.0,
    # -- v5 validation hard gates -----------------------------------------
    # dsr_gate_result / pbo_gate_result: pass the computed gate results
    # (see run_full_pipeline Steps 6c/6d) -- a FAILED gate rejects the
    # candidate (NOT READY) with its reason recorded, unless
    # gates_advisory_only=True, which records the failure as ADVISORY
    # without changing the verdict (debugging mode).
    dsr_gate_result: "DeflatedSharpeGateResult | None" = None,
    pbo_gate_result: "PBOGateResult | None" = None,
    primary_robustness_method: str = "walk_forward",
    gates_advisory_only: bool = False,
) -> tuple[str, list[str], "T58ScorecardResult", bool, bool]:
    """Pipeline reorg item #1: the verdict is now a hard safety gate
    (risk of ruin, and -- FIX (audit) -- a confirmed lookahead-bias leak)
    followed by app.scoring.t58_scorecard's continuous, missing-aware
    score -- NOT independent boolean checks ANDed together. See that
    module's docstring for why: requiring every one of several
    imperfect, correlated tests to pass simultaneously can reject a
    genuinely good strategy just because one noisy measurement
    disagreed with the others (pipeline reorg plan section 4). A
    confirmed lookahead leak is different in kind from those noisy
    measurements -- like risk of ruin, it's treated as a genuine
    pass/fail requirement, not evidence to weigh, because a strategy
    whose signal depends on future data isn't "somewhat trustworthy":
    every number this pipeline reports for it (backtest, Monte Carlo,
    walk-forward, CPCV) is downstream of that same leaky signal and
    therefore equally untrustworthy.

    Returns (verdict, verdict_reasons, scorecard_result,
    risk_of_ruin_hard_fail, lookahead_hard_fail).
    `verdict` stays one of the same three strings ("READY" / "MARGINAL" /
    "NOT READY") every existing caller and test already expects --
    Elite/Strong tiers -> READY, Promising/Research -> MARGINAL, Reject
    (or a hard-fail) -> NOT READY. `scorecard_result` carries the
    actual continuous score/tier/component breakdown for anything that
    wants more resolution than the 3-way verdict (e.g. the leaderboard).

    cpcv_primary_result: pass this when primary_robustness_method=="cpcv"
    for this run -- its efficiency fills the SAME scorecard slot
    oos_validation would otherwise fill (never both at once).
    cpcv_supporting_result: pass this when CPCV ran as a second,
    non-primary diagnostic (cpcv_supporting_enabled) -- scored in its
    own small-weight component instead.
    lookahead_bug_detected: the re-checked (final-configuration, not
    baseline) result of app.strategy.lookahead_check.check_for_lookahead
    -- see run_full_pipeline's "Lookahead re-check on the ACTUAL final
    candidate" step. Defaults to False so every existing caller/test
    that doesn't pass it keeps its prior behavior exactly."""
    reasons: list[str] = []
    ruin = final_mc.risk_of_ruin_pct
    ruin_hard_fail = ruin > risk_of_ruin_cap

    # -- The other hard gate: a confirmed lookahead-bias leak ------------
    # Checked before the ruin gate's early-return so a leaky strategy is
    # always reported as a lookahead failure first, even if its (equally
    # untrustworthy) simulated ruin number also happens to look bad --
    # the leak is the actionable root cause, not the ruin number.
    if lookahead_bug_detected:
        scorecard = score_from_results(
            mc_result=final_mc, walk_forward_result=oos_validation, statistics=statistics,
            prop_max_drawdown_pct=getattr(prop_rules, "max_drawdown_pct", None),
            parsimony_result=parsimony, cpcv_supporting_result=cpcv_supporting_result,
        )
        reasons.append(
            "HARD SAFETY GATE FAILED: the lookahead-bias check detected that this strategy's "
            "signal depends on data that had not happened yet as of its own bar -- see the "
            "lookahead check log/report above for exactly which bar. This strategy is NOT READY "
            "regardless of how the rest of the evidence looks, because every number this pipeline "
            "reports for it (backtest, Monte Carlo, walk-forward, CPCV) is downstream of that same "
            "leaky signal and is therefore equally unreliable."
        )
        reasons.append(f"For reference, {scorecard.render_line()} (not the reason for this verdict).")
        return "NOT READY", reasons, scorecard, False, True

    # -- Continuous, missing-aware scorecard (computed either way, so a
    # rejected strategy's other numbers still show up in the report and
    # leaderboard rather than vanishing behind a bare "NOT READY") ------
    # "Primary generalization test" is whichever of walk-forward/CPCV this
    # run actually used -- they share the same scorecard slot
    # (_walk_forward_score / _cpcv_score both map onto the same 0-100
    # "generalization efficiency" scale), never scored as two components
    # at once (pipeline reorg plan section 11).
    if cpcv_primary_result is not None:
        from app.scoring.t58_scorecard import _cpcv_score as _score_cpcv_as_primary
        scorecard = score_from_results(
            mc_result=final_mc, statistics=statistics,
            prop_max_drawdown_pct=getattr(prop_rules, "max_drawdown_pct", None),
            parsimony_result=parsimony, cpcv_supporting_result=cpcv_supporting_result,
        )
        from app.scoring.t58_scorecard import T58ScorecardInputs, compute_t58_score
        inputs = T58ScorecardInputs(**{k: v["value"] for k, v in scorecard.components.items()})
        inputs.walk_forward_stability = _score_cpcv_as_primary(cpcv_primary_result)
        scorecard = compute_t58_score(inputs)
    else:
        scorecard = score_from_results(
            mc_result=final_mc, walk_forward_result=oos_validation, statistics=statistics,
            prop_max_drawdown_pct=getattr(prop_rules, "max_drawdown_pct", None),
            parsimony_result=parsimony, cpcv_supporting_result=cpcv_supporting_result,
        )

    # -- The one legitimate hard gate ------------------------------------
    if ruin_hard_fail:
        reasons.append(
            f"HARD SAFETY GATE FAILED: Monte Carlo risk of ruin is {ruin:.1f}%, above the "
            f"configured cap of {risk_of_ruin_cap:.1f}%. This strategy is NOT READY regardless of "
            f"how the rest of the evidence looks -- ruin is the one metric this pipeline treats as "
            f"a genuine pass/fail requirement rather than evidence to weigh (see "
            f"FullPipelineConfig.risk_of_ruin_cap)."
        )
        reasons.append(f"For reference, {scorecard.render_line()} (not the reason for this verdict).")
        _hold_note = _holdout_untestable_note(holdout)
        if _hold_note:
            reasons.append(_hold_note)
        reasons.append(
            "This verdict does NOT lock the strategy: it can still be sent through the Validate hub "
            "(CPCV, Walk-Forward, Sensitivity, Regime Matrix) to see where it is weak."
        )
        return "NOT READY", reasons, scorecard, True, False

    if oos_validation is None and cpcv_primary_result is None:
        reasons.append(
            "Primary generalization test couldn't run (not enough data) -- scored as UNPROVEN "
            "(missing, not failing) rather than penalized as if it had failed."
        )

    # -- v5 validation hard gates -----------------------------------------
    # Walk-forward, CPCV, ICIR, DSR, and PBO graduate here from diagnostic
    # to REJECTING gates. The rule for each: a gate that RAN and FAILED
    # rejects the candidate; a gate that COULDN'T RUN stays UNPROVEN /
    # advisory (missing evidence is not the same as bad evidence) --
    # EXCEPT CPCV-as-primary, which this run explicitly selected as its
    # generalization test: choosing it and then getting no result is NOT
    # TESTED, and NOT TESTED is rejected (a run cannot pass on a
    # generalization test that never ran).
    validation_gate_failures: list[str] = []
    icir_hard_failed = False

    def _record_gate_failure(reason: str) -> None:
        validation_gate_failures.append(reason)
        tag = (
            "ADVISORY ONLY (validation_gates_advisory_only=True -- not rejecting)"
            if gates_advisory_only
            else "HARD VALIDATION GATE FAILED"
        )
        reasons.append(
            f"{tag}: {reason} This strategy is NOT READY -- a strong backtest that does not "
            f"generalize is exactly what these gates exist to catch."
            if not gates_advisory_only else
            f"{tag}: {reason}"
        )

    if oos_validation is not None and not oos_validation.is_stable:
        _record_gate_failure(
            f"the walk-forward check is NOT stable: walk-forward efficiency "
            f"{oos_validation.walk_forward_efficiency:.2f} is below the "
            f"{oos_validation.stability_threshold:.2f} stability threshold "
            f"across {oos_validation.n_folds} fold(s) -- this configuration does not retain "
            f"its edge on unseen folds."
        )
    if primary_robustness_method == "cpcv":
        if cpcv_primary_result is None:
            _record_gate_failure(
                "CPCV was selected as this run's primary robustness method "
                "(primary_robustness_method='cpcv') but produced no result -- NOT TESTED."
            )
        elif not cpcv_primary_result.is_robust:
            _record_gate_failure(
                f"the CPCV check is NOT robust: mean out-of-sample metric "
                f"{cpcv_primary_result.mean_oos_metric:.3f} vs in-sample "
                f"{cpcv_primary_result.mean_is_metric:.3f} across "
                f"{cpcv_primary_result.n_paths} path(s) -- the edge does not survive "
                f"partition choice."
            )
    if icir_gate is not None and not icir_gate.ok:
        # Genuine failure rejects: the ICIR was measurable in-sample AND
        # out-of-sample, but retention/decay/significance didn't hold. When
        # either ICIR is None the gate COULDN'T COMPUTE (too few distinct
        # periods with trades) -- that is UNPROVEN, not FAILED: a gate that
        # can't measure can't convict, the same couldn't-run/UNPROVEN
        # distinction the walk-forward gate makes (thin-data situations are
        # caught by min_trades_for_ready and the scorecard's
        # available-checks count instead).
        icir_measurable = (
            icir_gate.in_sample_icir is not None
            and icir_gate.out_sample_icir is not None
        )
        if icir_measurable:
            # In advisory mode the failure is recorded above as ADVISORY but the
            # old informational note still applies (the gate didn't reject) --
            # only suppress the duplicate note when the gate actually rejected.
            icir_hard_failed = not gates_advisory_only
            _record_gate_failure(
                "the ICIR / signal-decay / Bonferroni-corrected significance gate DID NOT PASS "
                "(" + " ".join(icir_gate.reasons) + ")."
            )
        # not measurable -> UNPROVEN: no rejection; the supporting-diagnostic
        # note below still records what the gate found.
    if dsr_gate_result is not None and not dsr_gate_result.passed:
        _record_gate_failure(dsr_gate_result.reason)
    if pbo_gate_result is not None and not pbo_gate_result.passed:
        _record_gate_failure(pbo_gate_result.reason)

    # -- B2(d): per-attempt pass-odds gate ----------------------------------
    # The Monte Carlo chain-level pass probability is inflated (it
    # counts "passed at least once over N attempts"), so the acceptance
    # verdict gates on the PER-ATTEMPT pass odds: the Wilson-95% lower
    # bound when present, else the per-attempt point estimate. Skipped
    # when the MC result carries no per-attempt information at all
    # (legacy results predating the v6 MC fields) -- a gate that can't
    # measure can't convict.
    from app.scoring.t58_scorecard import _preferred_pass_ci
    _pass_ci = _preferred_pass_ci(final_mc)
    _pass_lo = None
    if _pass_ci is not None:
        _pass_lo = _pass_ci[0]
    elif getattr(final_mc, "per_attempt_pass_probability", None):
        _pass_lo = final_mc.per_attempt_pass_probability
    if _pass_lo is not None and _pass_lo < min_per_attempt_pass_pct:
        _ci_txt = (
            f" (Wilson-95% CI {_pass_ci[0]:.1f}%–{_pass_ci[1]:.1f}%)"
            if _pass_ci is not None
            else " (per-attempt point estimate; no CI recorded)"
        )
        _record_gate_failure(
            f"per-attempt eval pass odds too low: the per-attempt pass lower bound is "
            f"{_pass_lo:.1f}%{_ci_txt} -- below the {min_per_attempt_pass_pct:.0f}% bar "
            f"(the chain-level number is inflated by counting multiple attempts per "
            f"simulation). See FullPipelineConfig.min_per_attempt_pass_pct."
        )

    if validation_gate_failures and not gates_advisory_only:
        reasons.append(f"For reference, {scorecard.render_line()} (not the reason for this verdict).")
        _hold_note = _holdout_untestable_note(holdout)
        if _hold_note:
            reasons.append(_hold_note)
        reasons.append(
            "This verdict does NOT lock the strategy: it can still be sent through the Validate hub "
            "(CPCV, Walk-Forward, Sensitivity, Regime Matrix) to see where it is weak."
        )
        return "NOT READY", reasons, scorecard, False, False

    if icir_gate is None:
        reasons.append(
            "ICIR / signal-decay / Bonferroni-corrected significance gate couldn't run -- kept as "
            "a supporting diagnostic only, not part of the T58 Score."
        )
    elif not icir_gate.ok and not icir_hard_failed:
        reasons.append(
            "Supporting diagnostic: did NOT pass the ICIR / signal-decay / Bonferroni-corrected "
            "significance gate (" + " ".join(icir_gate.reasons) + "). Not scored into the T58 Score "
            "or gated on -- section 14 of the pipeline reorg plan treats this as informational once "
            "a primary generalization test already ran."
        )

    # UPGRADE (buried-position-sizing-deviation): informational, not
    # gated/scored -- same treatment as the ICIR gate note just above --
    # but now actually said in the verdict, not just left as a number in
    # a report table nothing points at. See app.backtest.risk.position_
    # sizing_deviation_message; this is also already in `reasons`'
    # sibling BacktestResult.warnings (final_bt.warnings) whenever it
    # fired, so this line specifically calls it out as relevant to THIS
    # verdict rather than requiring a person to notice it buried there.
    if statistics is not None:
        sizing_note = position_sizing_deviation_message(statistics.to_dict())
        if sizing_note:
            reasons.append(f"Supporting diagnostic: {sizing_note}")

    reasons.append(scorecard.render_line())
    for name, comp in scorecard.components.items():
        if comp["value"] is not None:
            reasons.append(f"  {name}: {comp['value']:.1f}/100 (weight {comp['weight']:+.0f})")
    reasons.extend(scorecard.notes)

    if scorecard.tier in ("Elite", "Strong"):
        verdict = "READY"
    elif scorecard.tier in ("Promising", "Research"):
        verdict = "MARGINAL"
    else:
        verdict = "NOT READY"

    # -- Minimum trade-count floor for READY (see FullPipelineConfig.
    # min_trades_for_ready's own docstring for why this exists) ----------
    total_trades = getattr(statistics, "total_trades", None) if statistics is not None else None
    if verdict == "READY" and total_trades is not None and total_trades < min_trades_for_ready:
        verdict = "MARGINAL"
        reasons.append(
            f"CAPPED AT MARGINAL: the T58 Score alone would have called this READY, but the final "
            f"backtest only produced {total_trades} trade(s), below the {min_trades_for_ready}-trade "
            "floor this pipeline requires before trusting a result enough to call it READY -- a small "
            "sample can look strong by chance (or on the strength of one or two outsized trades) "
            "regardless of how good its score is. See FullPipelineConfig.min_trades_for_ready."
        )

    # -- D1: holdout gate ------------------------------------------------
    # A holdout that RAN with enough trades to mean something (>=
    # min_holdout_trades_for_gate) and LOST money (net < 0 or profit
    # factor < 1.0) while the in-sample run traded is a hard rejection:
    # the edge did not survive untouched data. An empty/thin holdout is
    # UNPROVEN, not FAILED (the empty-holdout MARGINAL cap below still
    # applies) -- a gate that can't measure can't convict, the same
    # couldn't-run distinction the walk-forward gate makes.
    _ho = holdout or {}
    _ho_stats = _ho.get("holdout_statistics") or {}
    _ho_in_stats = _ho.get("in_sample_statistics") or {}
    _ho_trades = _ho_stats.get("total_trades") or 0
    _ho_in_trades = _ho_in_stats.get("total_trades") or 0
    # The gate "ran" only when the holdout produced enough trades to be
    # a real check AND the in-sample run actually traded (otherwise there
    # is no in-sample edge to falsify). D5 keys off this same flag.
    holdout_gate_ran = bool(
        _ho_trades >= min_holdout_trades_for_gate and _ho_in_trades > 0
    )
    if holdout_gate_ran:
        _ho_net = _ho_stats.get("net_profit")
        _ho_pf = _ho_stats.get("profit_factor")
        if (_ho_net is not None and _ho_net < 0) or (_ho_pf is not None and _ho_pf < 1.0):
            _net_txt = f"${_ho_net:,.2f}" if _ho_net is not None else "n/a"
            _pf_txt = f"{_ho_pf:.2f}" if _ho_pf is not None else "n/a"
            _record_gate_failure(
                f"the holdout check LOST money on untouched data: holdout net {_net_txt} "
                f"(profit factor {_pf_txt}) over {_ho_trades} trade(s) while the in-sample "
                f"run took {_ho_in_trades} trade(s) -- the edge did not survive data the "
                f"search never saw."
            )
    if validation_gate_failures and not gates_advisory_only:
        reasons.append(f"For reference, {scorecard.render_line()} (not the reason for this verdict).")
        _hold_note = _holdout_untestable_note(holdout)
        if _hold_note:
            reasons.append(_hold_note)
        reasons.append(
            "This verdict does NOT lock the strategy: it can still be sent through the Validate hub "
            "(CPCV, Walk-Forward, Sensitivity, Regime Matrix) to see where it is weak."
        )
        return "NOT READY", reasons, scorecard, False, False

    # -- D5: WF-starved cap -----------------------------------------------
    # When BOTH the primary generalization test (walk-forward or CPCV)
    # and the holdout gate failed to produce a real check, a READY
    # verdict would rest on in-sample evidence alone -- cap at MARGINAL.
    # (In addition to the existing empty-holdout cap below, which only
    # fires when the holdout ran-but-empty.)
    _wf_missing = oos_validation is None and cpcv_primary_result is None
    if verdict == "READY" and _wf_missing and not holdout_gate_ran:
        verdict = "MARGINAL"
        reasons.append(
            "CAPPED AT MARGINAL: neither the walk-forward check nor CPCV produced a result, "
            "and the holdout gate did not run with enough trades to count as a check -- a READY "
            "verdict on in-sample evidence alone is not trusted. See "
            "FullPipelineConfig.min_holdout_trades_for_gate."
        )

    _hold_note = _holdout_untestable_note(holdout)
    if _hold_note:
        reasons.append(_hold_note)
        if verdict == "READY":
            verdict = "MARGINAL"
            reasons.append("CAPPED AT MARGINAL: a strategy is not called READY while its holdout check is empty.")

    return verdict, reasons, scorecard, False, False


# What each Full Pipeline verdict tags a newly-saved strategy with in the
# Strategy Library when FullPipelineConfig.library_status is left as None
# (the default) -- READY strategies still land on "validated" rather than
# jumping straight to "ready_for_demo"/"ready_for_live", since those last
# two stages are meant to reflect actual demo/live trading experience, not
# just a clean backtest -- promote it yourself once you've watched it run.
_VERDICT_TO_LIBRARY_STATUS = {
    "READY": "validated",
    "MARGINAL": "tested_passed",
    "NOT READY": "tested_failed",
}


def _library_save_note_suffix(verdict: str, verdict_reasons: list[str]) -> str:
    """A short, human-readable explanation appended to the log line shown
    right after Full Pipeline saves/replaces a Strategy Library entry.

    FIX (2026-09): before this existed, the log only ever said something
    like "...status: tested_failed" with no reason attached -- confusing
    whenever the Dashboard's single-run pass/fail pill (a DIFFERENT,
    narrower check -- see run_history.record_run's single_run_passed,
    which only reflects whether ONE historical run cleared prop-firm
    rules) showed "pass" for the very same strategy. Full Pipeline's
    verdict is a fuller check -- Monte Carlo risk of ruin, out-of-sample
    validation, trade-count floor, T58 Score -- and it and the single-run
    pill CAN legitimately disagree; the fix is explaining why, right
    where the status gets set, not leaving it unexplained."""
    if verdict == "READY":
        return ""
    headline = verdict_reasons[0] if verdict_reasons else ""
    return (
        f" Full Pipeline verdict: {verdict}"
        + (f" -- {headline}" if headline else "")
        + " (this is a fuller check than the Dashboard scorecard's single-run pass/fail pill, which only "
          "reflects one historical run and doesn't weigh Monte Carlo risk of ruin, out-of-sample "
          "validation, or trade count -- the two can disagree)."
    )


class FullPipelineCancelled(Exception):
    """Raised out of run_full_pipeline when a caller-supplied cancel_event
    is set between steps -- see run_full_pipeline's cancel_event param.
    Deliberately a distinct class from FullPipelineBatchCancelled (defined
    further below) even though they mean the same thing, so a caller that
    only runs single strategies doesn't need to import the batch module
    just to catch this."""


def run_full_pipeline(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    output_dir: str | Path,
    cfg: FullPipelineConfig | None = None,
    progress_cb: ProgressCallback | None = None,
    instrument: str = "unknown",
    ollama_settings: "OllamaSettings | None" = None,
    report_basename: str = "full_pipeline_report",
    cancel_event: threading.Event | None = None,
) -> FullPipelineResult:
    """
    report_basename: filename stem (no extension) for the written report,
    e.g. "full_pipeline_report" -> full_pipeline_report.html /
    full_pipeline_report.json in `output_dir`. Defaults to the fixed name
    every single-strategy run has always used. IMPORTANT for callers that
    run this in a loop (see run_full_pipeline_batch below): every call
    with the same output_dir AND the same report_basename overwrites the
    previous call's report -- pass a distinct report_basename per
    strategy when running more than one against the same output_dir.

    cancel_event: optional. Checked between each of the 7 steps below
    (not sub-step-by-sub-step -- Step 2's own GA already checks its own
    cancellation deep inside the worker-pool loop for the batch path, but
    a single-strategy run stopping between steps rather than mid-GA-
    generation is still a large improvement over "no stop button at all",
    which was the actual prior behavior of the web app's single Full
    Pipeline run). Raises FullPipelineCancelled the moment it's noticed;
    callers should treat that the same as FullPipelineBatchCancelled.

    ollama_settings: optional. When provided and `.is_usable` (enabled,
    with a host configured -- see app.ai.ollama_settings), Step 2's
    walk-forward-aware GA asks a local Ollama model for candidate
    parameter values once per generation and seeds them into that
    generation's population alongside the normal random/bred candidates
    (see app.optimize.walkforward_ga's ai_suggest_cb). Every suggestion
    still goes through the exact same backtest/prop-sim/Monte Carlo
    evaluation as any other candidate -- the model proposes numbers for
    already-existing tunable parameters, never code. None/disabled (the
    default) runs exactly as before this parameter existed; any failure
    to reach Ollama degrades to the same "AI assist is off" behavior
    without interrupting the pipeline.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            log("\nStop requested -- ending this Full Pipeline run.")
            raise FullPipelineCancelled("Full Pipeline stopped by user.")

    cfg = cfg or FullPipelineConfig()
    t0 = time.time()
    warnings: list[str] = []
    display_name = _display_name(strategy)

    # -- Circularity fix: reserve a true holdout BEFORE anything below
    # gets to see it (see CIRCULARITY_AUDIT.md and
    # FullPipelineConfig.reserve_true_holdout's own docstring) ----------
    # `df` keeps meaning "the full dataset" everywhere below (Step 5's
    # run_holdout_comparison call and the final report still use it, so
    # the report's price chart and Step 5's own re-derived split are
    # unaffected). `dev_df` is what Steps 1-4, CPCV, and regime
    # diagnostics use instead -- a strategy, once selected using ONLY
    # dev_df, sees the true holdout tail for the very first time in
    # Step 5, not before.
    if cfg.reserve_true_holdout:
        _holdout_split_idx = int(len(df) * (1 - cfg.holdout_frac))
        _holdout_split_idx = max(1, min(_holdout_split_idx, len(df) - 1)) if len(df) > 1 else len(df)
        dev_df = df.iloc[:_holdout_split_idx].reset_index(drop=True)
        log(
            f"Reserving the final {cfg.holdout_frac:.0%} of the dataset ({len(df) - len(dev_df):,} of "
            f"{len(df):,} bars) as a true holdout -- Steps 1-4 below only see the first {len(dev_df):,} "
            f"bars; Step 5 is the first time the reserved bars are used at all."
        )
    else:
        dev_df = df

    # Step 2's GA below loads its own full copy of `dev_df` into each worker
    # process it spawns (same pattern as Search Lab / Evolution Lab -- see
    # app.orchestration.resource_guard's module docstring). On a large
    # dataset (e.g. years of 1-minute bars), letting that default to
    # os.cpu_count() -- especially when this is itself one of several
    # strategies running in a batch, or another heavy job is running at
    # the same time -- risks exhausting system memory well before CPU.
    # Only overrides when the caller hasn't already pinned an explicit
    # value (batch mode already computes its own per-item split; this
    # additionally caps that against available memory).
    _safe_fp_workers = safe_worker_count(dev_df, requested=cfg.parallel_search_max_workers)
    if _safe_fp_workers != (cfg.parallel_search_max_workers or _safe_fp_workers):
        log(
            f"Reducing Full Pipeline GA worker processes from {cfg.parallel_search_max_workers} to "
            f"{_safe_fp_workers} -- {len(dev_df):,} bars is large enough that more full copies of it "
            f"(one per worker) would risk exhausting available memory."
        )
    cfg = replace(cfg, parallel_search_max_workers=_safe_fp_workers)

    # RISK-001: detect (and log) a RiskConfig.initial_balance / PropRules.
    # account_size mismatch BEFORE with_prop_safety_defaults silently
    # reconciles it below -- same "!!!"-prefixed prominence this pipeline
    # already gives the pip_size/instrument-scale mismatch warning, since
    # every number this run produces was computed against the corrected
    # balance, not whatever the caller originally passed in.
    account_mismatch_warning = account_size_mismatch_message(risk.initial_balance, prop_rules.account_size)
    if account_mismatch_warning is not None:
        log(f"  !!! {account_mismatch_warning}")
        warnings.append(account_mismatch_warning)

    # Automatically ties the raw execution engine's account-blown circuit
    # breaker (app.backtest.execution) to whatever max-drawdown floor this
    # PROP FIRM actually enforces, so a single misconfigured/gapped trade
    # can never report a loss bigger than the account the prop simulation
    # is about to test it against. No-op if `risk` already set its own
    # max_account_drawdown_pct explicitly. Also forces risk.initial_balance
    # to match prop_rules.account_size (see RISK-001 note on this function).
    risk = with_prop_safety_defaults(risk, prop_rules)

    # FIX (2026-09-18): cfg.reset_on_breach was already threaded into the
    # POST-HOC scoring layer below (simulate_account / MonteCarloConfig)
    # but never into the RiskConfig the RAW baseline/GA-search/final
    # backtests actually run against -- so with the checkbox checked, the
    # underlying trade sequence for every stage of this pipeline still
    # permanently stopped opening new trades the instant it first blew the
    # configured drawdown floor, often within the first few trades of a
    # multi-year dataset. See RiskConfig.reset_on_breach's own docstring.
    # This one-line fix is what actually makes "score on the basis that a
    # blown account gets a fresh eval and keeps going" true end to end.
    risk = replace(risk, reset_on_breach=cfg.reset_on_breach)

    adaptive_risk = build_limit_aware_preset(prop_rules, daily_profit_lock_pct=cfg.adaptive_risk_daily_profit_lock_pct) \
        if cfg.adaptive_risk_enabled else None
    if adaptive_risk is not None:
        log(f"Adaptive risk enabled: {len(adaptive_risk.rules)} limit-aware throttle rule(s) applied to every backtest below.")

    # -- Step 1: baseline -----------------------------------------------
    log(f"Step 1/7: Baseline run for '{display_name}'...")
    preflight_signal_check(dev_df, strategy, risk, "Full Pipeline")
    baseline_bt = run_backtest(dev_df, strategy, risk, adaptive_risk=adaptive_risk)
    for w in baseline_bt.warnings:
        log(f"  WARNING: {w}")
        warnings.append(w)

    lookahead_summary = None
    lookahead_bug_detected = False
    if strategy.source_type in ("manual", "python", "pinescript", "mql5"):
        try:
            from app.strategy.lookahead_check import check_for_lookahead
            lookahead_result = check_for_lookahead(strategy, dev_df, max_signal_checkpoints=8)
            lookahead_summary = lookahead_result.summary()
            lookahead_bug_detected = lookahead_result.bug_detected
            log(f"  Lookahead check: {lookahead_summary}")
        except Exception:
            log("  Lookahead check failed to run (skipped, best-effort only).")
    # NOTE: this baseline-strategy check is re-run below (see "Lookahead
    # check (final configuration)") against whatever strategy actually
    # ends up as `final_strategy` after Step 2's optimization -- that
    # re-check, not this one, is what _make_verdict() below treats as a
    # hard gate. This one only seeds the log/report early and covers the
    # (common) case where refinement never runs or never changes the
    # verdict-relevant answer.

    pnls = [t.pnl for t in baseline_bt.trades]
    dates = [t.entry_time for t in baseline_bt.trades]
    baseline_single_run = simulate_account(pnls, dates, prop_rules, reset_on_breach=cfg.reset_on_breach)
    baseline_mc = run_monte_carlo(
        baseline_bt.trades, prop_rules,
        MonteCarloConfig(method=default_method_for_adaptive_risk(adaptive_risk), n_simulations=cfg.baseline_mc_sims, random_seed=cfg.random_seed, reset_on_breach=cfg.reset_on_breach),
    )
    log(
        format_run_summary_line(
            "  Baseline", len(baseline_bt.trades), baseline_bt.statistics,
            baseline_mc.evaluation_pass_probability, baseline_mc.first_payout_probability,
            baseline_mc.per_attempt_pass_probability, baseline_mc.per_attempt_payout_probability,
            baseline_mc.total_independent_attempts,
        )
    )

    # -- Step 2: robust (walk-forward-aware) optimization ----------------
    _check_cancel()
    log("Step 2/7: Searching for a more robust configuration (walk-forward-aware GA)...")
    refinement_ran = False
    refinement_skip_reason = None
    ga_result: WalkforwardGAResult | None = None
    final_source_type = strategy.source_type
    final_config = strategy.config if strategy.source_type == "manual" else None
    final_code_text, final_code_ext = (None, None)
    if strategy.source_type != "manual":
        final_code_text, final_code_ext = patched_source_for_strategy(strategy, [], [])

    ai_suggest_cb = None
    if ollama_settings is not None and ollama_settings.is_usable:
        from app.ai.ollama_client import OllamaClient
        from app.ai.research_library import find_relevant_excerpts
        from app.optimize.gene_fitness_analysis import analyze_gene_fitness_correlation

        ollama_client = OllamaClient(ollama_settings)
        # Captures Step 1's baseline stats once -- good enough context for
        # every generation's request without re-running anything extra.
        # A live "how's the search going" readout would need re-summarizing
        # the current best each generation; left for a future pass.
        baseline_stats_summary = {
            k: v for k, v in baseline_bt.statistics.to_dict().items()
            if k in ("net_profit", "win_rate", "profit_factor", "max_drawdown_pct", "total_trades")
        }
        prop_rules_summary = {
            "account_size": prop_rules.account_size,
            "max_drawdown_pct": prop_rules.max_drawdown_pct,
            "daily_loss_limit_pct": prop_rules.daily_loss_limit_pct,
            "evaluation_profit_target_pct": prop_rules.evaluation_profit_target_pct,
        }

        # Circuit breaker: a genuinely slow/unreachable/misconfigured
        # Ollama would otherwise pay its full timeout on EVERY generation
        # (confirmed via a real run: 7 straight "didn't respond in time"
        # messages, one per generation, ~90s+ each wasted for nothing).
        # After 2 consecutive failures, stop trying for the rest of this
        # run and say so once -- the search still proceeds exactly as if
        # AI assist were off.
        consecutive_failures = 0
        gave_up = False

        def ai_suggest_cb(genes: list, population: list) -> list:
            nonlocal consecutive_failures, gave_up
            if gave_up:
                return []
            # Stage 4 of the quant loop framework ("analyze why the losers
            # failed, feed that back into generation") computed as plain
            # statistics over the population the GA already evaluated --
            # no extra backtests, no AI call. Only the resulting few lines
            # of text are spent as prompt tokens below, which is what
            # keeps this "systematic first, AI only where it must be."
            analysis = analyze_gene_fitness_correlation(genes, population)
            feedback_lines = analysis.summary_lines(top_n=3)

            # Same "retrieval is free, AI is only for the last step"
            # principle as the gene-fitness analysis above: a plain
            # keyword search over whatever's in the research/ folder,
            # queried on the strategy's name and its genes' own labels
            # (e.g. "EMA period", "session filter") since those are
            # exactly the terms a relevant paper would use. Costs nothing
            # extra beyond folder-mtime bookkeeping and returns [] with
            # an empty research/ folder -- identical to this feature not
            # existing at all.
            research_query = f"{display_name} {strategy.source_type} " + " ".join(g.label for g in genes)
            research_excerpts = find_relevant_excerpts(research_query, max_excerpts=2)

            result = ollama_client.suggest_parameter_adjustments(
                strategy_name=display_name,
                source_type=strategy.source_type,
                genes=genes,
                baseline_stats=baseline_stats_summary,
                prop_rules_summary=prop_rules_summary,
                failure_analysis_lines=feedback_lines,
                research_excerpts=research_excerpts,
            )
            if result.error:
                log(f"  AI assist: {result.error}")
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    gave_up = True
                    log("  AI assist: giving up after 2 consecutive failures -- "
                        "continuing the search without it for the rest of this run.")
                return []
            consecutive_failures = 0
            if feedback_lines and not analysis.note:
                log(f"  AI assist: seeded with {len(feedback_lines)} observed parameter pattern(s) from this search.")
            return result.genomes

    # Fast-skip: a pip_size/instrument-scale mismatch (see the
    # pip_scale_mismatch AND atr_scale_mismatch warnings in
    # app.backtest.execution) invalidates every position size and every
    # stop distance the baseline computed -- spending a full
    # multi-generation GA search (typically the single most expensive
    # step of a Full Pipeline run, 100-270s in a real 23-strategy batch)
    # tuning parameters against numbers that can't be trusted is pure
    # wasted wall-clock time. The fix is changing risk.pip_size to match
    # the instrument (see suggest_pip_size), not anything the GA can
    # search its way around. Skip straight to reporting NOT READY with
    # the actionable reason instead.
    #
    # Both warnings are matched here (not just the price-ratio one)
    # because a fixed-pips stop can pass the price-ratio check -- look
    # like a perfectly ordinary fraction of price -- while still being
    # tiny next to the instrument's own actual volatility (ATR); a
    # high-priced but volatile instrument such as an equity index is the
    # case that price-ratio alone misses.
    instrument_mismatch = has_instrument_scale_mismatch(baseline_bt.warnings)
    # Same fast-skip logic as the pip_size/instrument-scale mismatch above,
    # for the same reason: a condition comparing a bounded oscillator
    # (RSI, Stochastic, MFI, ...) to a threshold outside its possible
    # range (e.g. "RSI > 102.56") can never be satisfied, permanently
    # disabling that branch of the strategy's logic -- see
    # app.strategy.manual.validate_bounded_conditions. Before the GA
    # gene-bounds fix in app.optimize.parameter_space, a GA search could
    # itself introduce this by mutating a comparison threshold outside
    # the compared indicator's range; that path is now closed, but a
    # hand-typed or already-saved config can still arrive here with one,
    # and searching for "better" parameters around dead logic is exactly
    # as wasted as searching around an unreliable pip_size.
    impossible_condition = has_impossible_condition(baseline_bt.warnings)
    if cfg.skip_optimization:
        refinement_skip_reason = (
            "Skipped optimization search: FullPipelineConfig.skip_optimization is set (lock-"
            "parameters / fixed-backtest mode). Every step below ran against the SUPPLIED "
            "strategy exactly as given -- no parameter was changed from what was provided."
        )
        log(f"  Optimization skipped: {refinement_skip_reason}")
        ga_result = None
    elif instrument_mismatch or impossible_condition:
        if instrument_mismatch:
            refinement_skip_reason = (
                "Skipped optimization search: the baseline run flagged a pip_size/"
                "instrument-scale mismatch (see the WARNING above). Every position "
                "size and stop distance this backtest computed is unreliable, so "
                "searching for 'better' parameters against those numbers would "
                "waste the search budget without producing a trustworthy result. "
                "Set risk.pip_size to match this instrument (e.g. 0.01 for gold/"
                "JPY pairs, 1.0 for high-priced indices/stocks -- see "
                "app.backtest.risk.suggest_pip_size) and re-run."
            )
        else:
            reason_line = next(w for w in baseline_bt.warnings if "can never be true" in w)
            refinement_skip_reason = (
                f"Skipped optimization search: {reason_line} Fix the impossible "
                "condition (or remove it) before searching -- optimizing around "
                "permanently dead logic wastes the search budget without producing "
                "a trustworthy result."
            )
        log(f"  Optimization skipped: {refinement_skip_reason}")
        ga_result = None
    else:
        try:
            refine_cfg = RefinementConfig(
                population_size=cfg.ga_population,
                generations=cfg.ga_generations,
                fitness_metric=cfg.fitness_metric,
                search_monte_carlo_sims=cfg.ga_search_mc_sims,
                random_seed=cfg.random_seed,
                optimizer_mode=cfg.optimizer_mode,
                # UPGRADE (GA-searches-what-it's-graded-on): the search
                # itself is now penalized for elevated Monte Carlo risk of
                # ruin using the SAME cap _make_verdict hard-vetoes on
                # below, instead of only finding out after Step 2 already
                # picked a "winner" -- see RefinementConfig.risk_of_ruin_cap.
                risk_of_ruin_cap=cfg.risk_of_ruin_cap,
            )
            ga_result = run_walkforward_aware_refinement(
                dev_df, strategy, risk, prop_rules,
                MonteCarloConfig(method=default_method_for_adaptive_risk(adaptive_risk), n_simulations=cfg.ga_search_mc_sims, random_seed=cfg.random_seed, reset_on_breach=cfg.reset_on_breach),
                refinement_config=refine_cfg,
                n_folds=cfg.n_folds, window_mode=cfg.window_mode,
                progress_cb=lambda m: log(f"  {m}"),
                ai_suggest_cb=ai_suggest_cb,
                parallel=cfg.parallel_search,
                max_workers=cfg.parallel_search_max_workers,
                adaptive_risk=adaptive_risk,
            )
            refinement_ran = True
            if ga_result.best.oos_trade_count > 0:
                final_config = ga_result.best.config
                final_code_text = ga_result.best.code_text
                final_code_ext = ga_result.best.code_extension
                log(
                    f"  Winning configuration: chained-OOS fitness {ga_result.best.fitness:.3f} "
                    f"({ga_result.best.oos_trade_count} OOS trades across {ga_result.n_folds} fold(s))."
                )
                if ga_result.overfitting_gap is not None and ga_result.overfitting_gap > 0:
                    warnings.append(
                        f"Overfitting gap (in-sample fitness minus chained-OOS fitness): "
                        f"{ga_result.overfitting_gap:.3f}. A large positive gap means the winning "
                        f"configuration looks noticeably better in-sample than out-of-sample."
                    )
                warnings.extend(ga_result.warnings)
            else:
                warnings.append(
                    "The walk-forward-aware GA's best candidate still produced zero out-of-sample "
                    "trades -- keeping the original baseline configuration as final instead of a "
                    "'winner' that never actually traded out-of-sample."
                )
        except RefinementError as exc:
            refinement_skip_reason = str(exc)
            log(f"  Optimization skipped: {exc}")

    # -- Build the final strategy -----------------------------------------------
    if final_source_type == "manual":
        final_spec = _spec_for_manual(final_config)
    else:
        final_spec = _spec_for_code(final_source_type, final_code_text, final_code_ext)

    # SPEED (2026-10-06): when Step 2's search returns a configuration
    # identical to the one handed in (the GA's best genome IS the baseline
    # genome -- Owen's 2020-2026 ES run printed byte-identical Baseline
    # and Final parameter tables), Step 3 used to re-run the entire
    # full-length backtest to produce a bit-identical trade list, then
    # Step 3's Monte Carlo re-scored it. Spec equality is exact (same
    # JSON-able dict), so reuse Step 1's backtest wholesale -- same
    # trades, same statistics, zero statistical change; the 10,000-sim
    # final Monte Carlo below still runs on those trades (Step 1's own
    # MC was only baseline_mc_sims and is NOT reused).
    if strategy.source_type == "manual":
        _baseline_spec = _spec_for_manual(strategy.config)
    else:
        _base_text, _base_ext = patched_source_for_strategy(strategy, [], [])
        _baseline_spec = _spec_for_code(strategy.source_type, _base_text, _base_ext)
    _final_is_baseline = final_spec == _baseline_spec

    from tempfile import mkdtemp
    from shutil import rmtree
    final_tmp_dir = Path(mkdtemp(prefix="t58_fullpipeline_")) if final_source_type != "manual" else None
    try:
        final_strategy = build_strategy_from_spec(final_spec, final_tmp_dir)

        # -- Step 3: final validation ------------------------------------
        _check_cancel()
        if _final_is_baseline:
            log("Step 3/7: Final validation -- the winning configuration is identical to the "
                "baseline, reusing the Step 1 backtest instead of re-running it...")
            final_bt = baseline_bt
        else:
            log("Step 3/7: Final validation (full backtest, prop simulation, Monte Carlo)...")
            final_bt = run_backtest(dev_df, final_strategy, risk, adaptive_risk=adaptive_risk)
            for w in final_bt.warnings:
                log(f"  WARNING: {w}")
                warnings.append(w)
        if not final_bt.trades:
            # Should not happen (the GA never returns a worse-than-baseline
            # candidate, and baseline already passed preflight), but never
            # trust that blindly -- fall back to the baseline strategy/spec.
            warnings.append(
                "The selected final configuration unexpectedly produced zero trades on "
                "re-validation -- falling back to the original baseline configuration."
            )
            final_source_type = strategy.source_type
            final_config = strategy.config if strategy.source_type == "manual" else None
            if strategy.source_type != "manual":
                final_code_text, final_code_ext = patched_source_for_strategy(strategy, [], [])
            final_spec = (
                _spec_for_manual(final_config) if final_source_type == "manual"
                else _spec_for_code(final_source_type, final_code_text, final_code_ext)
            )
            final_strategy = build_strategy_from_spec(final_spec, final_tmp_dir)
            final_bt = baseline_bt

        # -- Lookahead re-check on the ACTUAL final candidate ----------------
        # FIX (audit): the check above only ever looked at the strategy as
        # handed in, before Step 2's optimization had a chance to run. This
        # is the one _make_verdict() below actually gates on -- Step 2 can
        # change parameters (or, for a fallback, revert to baseline), so the
        # candidate that gets a verdict is re-checked directly rather than
        # trusting the earlier check's target strategy instance to still be
        # representative. Cheap: a handful of truncated re-generate() calls,
        # same cost class as the baseline check above.
        if final_source_type in ("manual", "python", "pinescript", "mql5"):
            try:
                from app.strategy.lookahead_check import check_for_lookahead
                final_lookahead_result = check_for_lookahead(final_strategy, dev_df, max_signal_checkpoints=8)
                lookahead_summary = final_lookahead_result.summary()
                lookahead_bug_detected = final_lookahead_result.bug_detected
                log(f"  Lookahead check (final configuration): {lookahead_summary}")
            except Exception:
                log("  Lookahead check on final configuration failed to run (skipped, best-effort only).")
        else:
            # Manual (indicator-builder) strategies are causal by
            # construction (see app.strategy.lookahead_check's own scope
            # note) -- no re-check needed even if refinement changed
            # numeric parameters, since the underlying evaluation code
            # never changes shape.
            lookahead_bug_detected = False

        pnls = [t.pnl for t in final_bt.trades]
        dates = [t.entry_time for t in final_bt.trades]
        final_single_run = simulate_account(pnls, dates, prop_rules, reset_on_breach=cfg.reset_on_breach)

        # -- Full-history backtest for reporting/charting only ---------------
        # final_bt above is intentionally dev_df-only when reserve_true_holdout
        # is on, so the READY/MARGINAL/NOT READY verdict, t58_score, and
        # final_mc never see the reserved holdout tail before Step 5 -- that
        # separation is the whole point of the holdout design (see the
        # comment above dev_df's own definition) and must not change. But the
        # final HTML report's interactive trade chart and "single historical
        # run" prop-firm summary were BOTH built from that same dev_df-only
        # final_bt, plotted against price_df=df (the FULL dataset) -- so the
        # chart's price line ran all the way to the end while its trade
        # markers and equity curve simply stopped wherever dev_df ended,
        # rendering as a flat line for the reserved tail. That flat line was
        # never "the strategy hit its profit target and stopped trading" --
        # run_backtest has no profit-target-based halting at all, and
        # app.prop.simulator.simulate_account already transitions
        # evaluation -> funded and keeps walking every remaining trade,
        # tracking payout cycles, rather than stopping there -- it was
        # purely this dev-only-data-plotted-against-full-length-price-axis
        # mismatch. full_history_bt/full_history_single_run below re-run the
        # SAME final strategy across the complete `df` purely so the report
        # reflects it, without touching final_bt/final_single_run/final_mc's
        # dev-only inputs anywhere above or below this block.
        if cfg.reserve_true_holdout and len(dev_df) < len(df):
            full_history_bt = run_backtest(df, final_strategy, risk, adaptive_risk=adaptive_risk)
            full_history_pnls = [t.pnl for t in full_history_bt.trades]
            full_history_dates = [t.entry_time for t in full_history_bt.trades]
            full_history_single_run = simulate_account(
                full_history_pnls, full_history_dates, prop_rules, reset_on_breach=cfg.reset_on_breach,
            )
        else:
            full_history_bt = final_bt
            full_history_single_run = final_single_run

        final_mc = run_monte_carlo(
            final_bt.trades, prop_rules,
            MonteCarloConfig(method=default_method_for_adaptive_risk(adaptive_risk), n_simulations=cfg.final_mc_sims, random_seed=cfg.random_seed, reset_on_breach=cfg.reset_on_breach),
            # MC-004: these trades came from Step 2's GA search over this
            # same dev_df when refinement actually ran -- see
            # run_monte_carlo's docstring. Steps 4-6 below provide the
            # genuinely independent evidence this number alone doesn't.
            selection_bias_caveat=refinement_ran,
        )
        log(
            format_run_summary_line(
                "  Final", len(final_bt.trades), final_bt.statistics,
                final_mc.evaluation_pass_probability, final_mc.first_payout_probability,
                final_mc.per_attempt_pass_probability, final_mc.per_attempt_payout_probability,
                final_mc.total_independent_attempts,
            )
        )

        # -- Step 4: out-of-sample fold check (no re-tuning) --------------
        _check_cancel()
        log("Step 4/7: Out-of-sample fold check (same configuration, no further tuning)...")
        oos_validation = None
        oos_skip_reason = None
        # VAL-005 fix: Step 2's GA already selected the winning genome
        # using chained fitness over its OWN fold test windows on this
        # same dev_df -- without embargoing past those bars, this step's
        # EXPANDING fold construction substantially or fully overlapped 3
        # of 4 "out-of-sample" fold test windows with the GA's own
        # (verified numerically), making most of this "independence" an
        # illusion. ga_result.max_test_bar_used (None if refinement
        # didn't run, or the strategy had no tunable parameters) tells
        # run_walk_forward exactly where to start instead.
        embargo_start_bar = (
            ga_result.max_test_bar_used if (refinement_ran and ga_result is not None) else None
        )
        try:
            oos_validation = run_walk_forward(
                dev_df, lambda: build_strategy_from_spec(final_spec, final_tmp_dir), risk,
                n_folds=cfg.oos_check_folds, metric=cfg.oos_check_metric,
                prop_rules=prop_rules, mc_cfg=MonteCarloConfig(n_simulations=cfg.ga_search_mc_sims, random_seed=cfg.random_seed, reset_on_breach=cfg.reset_on_breach),
                embargo_start_bar=embargo_start_bar,
            )
            if oos_validation is None:
                oos_skip_reason = "Not enough bars to build the requested number of out-of-sample folds."
                if embargo_start_bar is not None:
                    oos_skip_reason += (
                        f" (after embargoing the first {embargo_start_bar:,} bar(s) the "
                        "optimization search already used, to keep this check genuinely independent)."
                    )
            else:
                for w in oos_validation.warnings:
                    log(f"  {w}")
                    warnings.append(w)
                log(
                    f"  Walk-forward efficiency {oos_validation.walk_forward_efficiency:.2f} "
                    f"({'stable' if oos_validation.is_stable else 'NOT stable'})."
                )
        except Exception as exc:  # noqa: BLE001 -- best-effort validation step
            oos_skip_reason = f"Out-of-sample check failed to run: {exc}"
            log(f"  {oos_skip_reason}")

        # -- Step 5: holdout check ----------------------------------------
        # Deliberately the ORIGINAL, full `df` here (not dev_df) -- this
        # is what makes the holdout genuine: re-splitting the full
        # dataset by the same holdout_frac reproduces dev_df's own
        # cutoff exactly, so the tail half this compares against is the
        # same bars Steps 1-4 above never got to see (see
        # FullPipelineConfig.reserve_true_holdout).
        _check_cancel()
        log("Step 5/7: Out-of-sample holdout check...")
        try:
            final_holdout = run_holdout_comparison(df, final_strategy, risk, holdout_frac=cfg.holdout_frac, adaptive_risk=adaptive_risk)
        except Exception:
            final_holdout = None
            log("  Holdout check skipped (not enough data to split).")

        # -- Step 6: ICIR / signal-decay / Bonferroni-corrected gate ------
        # The quant loop framework's "out-of-sample gate": scores the
        # strategy's directional signal with the standard IC/ICIR metric,
        # checks whether its predictive power decays too fast to trade,
        # and requires the result to still be statistically significant
        # after correcting for how many candidates the GA actually tried
        # (Bonferroni). All pure arithmetic over trades already produced
        # above -- no AI, no extra network calls, no extra backtests
        # beyond the same in-sample/holdout split run_holdout_comparison
        # just used. See app.validation.icir. Uses the full `df` (like
        # Step 5, not dev_df) -- it doesn't select or score parameters,
        # so it isn't part of the circularity Step 2's search creates;
        # it's simply re-deriving its own in-sample/holdout split.
        _check_cancel()
        log("Step 6/7: ICIR / signal-decay / Bonferroni-corrected significance gate...")
        icir_gate = None
        icir_gate_skip_reason = None
        try:
            n_candidates_tested = 1  # the baseline itself always counts as one candidate tried
            if refinement_ran and ga_result is not None:
                # FIX (Bonferroni-vs-actual-evaluations): ga_result.total_
                # evaluations is how many genomes the GA actually
                # backtested -- population*(generations+1) is only the
                # CONFIGURED budget, which now can be smaller than that
                # whenever auto-shrink-on-low-trades reduced generations
                # (see RefinementConfig.auto_shrink_on_low_trades) or
                # larger whenever AI-assist/parallel evaluation added
                # extra candidates. Bonferroni-correcting for the budget
                # instead of what actually ran either under- or over-
                # states how many independent trials this significance
                # gate needs to account for.
                n_candidates_tested = max(1, ga_result.total_evaluations)
            icir_gate = run_icir_gate_from_backtest(
                df, final_strategy, risk, n_tests=n_candidates_tested, holdout_frac=cfg.holdout_frac,
            )
            log(f"  {'PASSED' if icir_gate.ok else 'DID NOT PASS'} "
                f"(Bonferroni-corrected for {n_candidates_tested} candidate(s) tried).")
            for reason in icir_gate.reasons:
                log(f"    {reason}")
        except Exception as exc:  # noqa: BLE001 -- best-effort validation step
            icir_gate_skip_reason = f"ICIR gate failed to run: {exc}"
            log(f"  {icir_gate_skip_reason}")

        # -- Pipeline reorg extra: CPCV/PBO (Option A -- one canonical
        # generalization test, section 11) ----------------------------------
        # cfg.primary_robustness_method selects whether CPCV or the
        # walk-forward check above (Step 4) is the PRIMARY generalization
        # test feeding the scorecard. cfg.cpcv_supporting_enabled instead
        # runs CPCV as a second, SUPPORTING diagnostic alongside
        # walk-forward-as-primary -- the two settings are mutually
        # exclusive in what they feed (see _make_verdict), so CPCV never
        # gets counted as evidence twice under two different names.
        _check_cancel()
        cpcv_primary_result = None
        cpcv_supporting_result = None
        cpcv_skip_reason = None
        run_cpcv_this_time = (cfg.primary_robustness_method == "cpcv") or cfg.cpcv_supporting_enabled
        if run_cpcv_this_time:
            log("Step 6b/7: CPCV / PBO (out-of-sample generalization, multi-path)...")
            try:
                cpcv_result = run_cpcv(
                    dev_df, lambda: build_strategy_from_spec(final_spec, final_tmp_dir), risk,
                    n_groups=cfg.cpcv_n_groups, n_test_groups=cfg.cpcv_n_test_groups,
                    metric=cfg.oos_check_metric, prop_rules=prop_rules,
                    mc_cfg=MonteCarloConfig(n_simulations=cfg.ga_search_mc_sims, random_seed=cfg.random_seed, reset_on_breach=cfg.reset_on_breach),
                )
                log(
                    f"  CPCV: {cpcv_result.n_paths} path(s), mean OOS/IS "
                    f"{cpcv_result.mean_oos_metric:.3f}/{cpcv_result.mean_is_metric:.3f} "
                    f"({'robust' if cpcv_result.is_robust else 'NOT robust'})."
                )
                if cfg.primary_robustness_method == "cpcv":
                    cpcv_primary_result = cpcv_result
                else:
                    cpcv_supporting_result = cpcv_result
            except CPCVError as exc:
                cpcv_skip_reason = f"CPCV could not run: {exc}"
                log(f"  {cpcv_skip_reason}")
            except Exception as exc:  # noqa: BLE001 -- best-effort validation step
                cpcv_skip_reason = f"CPCV failed to run: {exc}"
                log(f"  {cpcv_skip_reason}")
        elif cfg.primary_robustness_method == "cpcv":
            cpcv_skip_reason = "CPCV was selected as the primary robustness method but did not run (see log above)."

        # -- v5 Step 6c/7: Deflated Sharpe gate (multiple-testing) --------
        # The champion's Sharpe, deflated for the search's FULL multiple-
        # testing burden via count_all_trials() (baseline + every GA genome
        # the inner loop actually backtested -- the old leaderboard-length
        # undercount made this optimistic). The trial-Sharpe spread comes
        # from re-backtesting the GA's final-generation leaderboard once
        # each on dev_df -- the search's own candidates, not an assumed
        # textbook spread. Known conservative limitation, documented in
        # the log line: the final generation already converged, so its
        # spread is a LOWER bound on the true cross-trial dispersion --
        # the deflation is weaker than textbook, which makes a REJECT here
        # unambiguous (it failed even the lenient version).
        _check_cancel()
        dsr_gate_result = None
        dsr_skip_reason = None
        if cfg.dsr_gate_enabled:
            log("Step 6c/7: Deflated Sharpe gate (multiple-testing correction)...")
            try:
                from app.backtest.engine import run_backtest as _run_backtest_dsr
                n_trials = count_all_trials(
                    baseline_count=1,
                    ga_total_evaluations=(ga_result.total_evaluations
                                          if (refinement_ran and ga_result is not None) else 0),
                    # D4(b): hand-tuned / externally-evaluated configs
                    # count too -- otherwise a hand-tuned strategy gets
                    # n_trials=1 and an optimistic deflated Sharpe.
                    extra_trial_counts=(cfg.external_trial_count,),
                )
                trial_sharpes: list[float] = []
                if refinement_ran and ga_result is not None and ga_result.leaderboard:
                    for cand in ga_result.leaderboard:
                        try:
                            cand_spec = _leaderboard_candidate_spec(cand, final_source_type)
                            cand_bt = _run_backtest_dsr(
                                dev_df, build_strategy_from_spec(cand_spec, final_tmp_dir), risk)
                            trial_sharpes.append(float(cand_bt.statistics.sharpe_ratio))
                        except Exception:  # noqa: BLE001 -- one bad genome must not kill the gate
                            continue
                if not trial_sharpes:
                    # No GA leaderboard to sample a spread from (refinement
                    # didn't run or produced nothing re-backtestable): fall
                    # back to the two Sharpe observations the pipeline
                    # genuinely has. expected_max_sharpe() needs >= 2 finite
                    # values for a spread; with fewer the benchmark is 0
                    # and the gate degrades to a plain PSR-vs-zero check,
                    # which the log says out loud rather than hiding.
                    trial_sharpes = [float(final_bt.statistics.sharpe_ratio)]
                    dsr_skip_reason = (
                        "DSR gate ran on a degenerate trial pool (no GA leaderboard Sharpes available) -- "
                        "treated as a plain significance check, not a full multiple-testing correction."
                    )
                    log(f"  {dsr_skip_reason}")
                _pnls = [t.pnl for t in final_bt.trades]
                _skew, _kurt = _pnl_skew_kurtosis(_pnls)
                dsr_gate_result = deflated_sharpe_gate(
                    float(final_bt.statistics.sharpe_ratio),
                    trial_sharpes,
                    n_trials,
                    len(final_bt.trades),
                    returns_skew=_skew,
                    returns_kurtosis=_kurt,
                    min_probabilistic_sharpe=cfg.dsr_min_probabilistic_sharpe,
                )
                log(f"  {dsr_gate_result.reason}")
            except Exception as exc:  # noqa: BLE001 -- best-effort validation step
                dsr_skip_reason = f"DSR gate failed to run: {exc}"
                log(f"  {dsr_skip_reason}")

        # -- v5 Step 6d/7: PBO gate (was the SELECTION process overfit?) ---
        # Genuine Bailey et al. (2017) PBO over the GA's final-generation
        # leaderboard: for every CPCV path, rank the pool in-sample, take
        # the IS winner, check its OOS rank. Above pbo_max the search was
        # selecting noise, so the champion is rejected no matter how good
        # it looks. Needs >= 2 candidates -- with fewer the measurement is
        # degenerate by construction and the gate stays out of the way.
        _check_cancel()
        pbo_gate_result = None
        pbo_skip_reason = None
        if cfg.pbo_gate_enabled:
            log("Step 6d/7: PBO gate (probability of backtest overfitting)...")
            try:
                pbo_specs = []
                if refinement_ran and ga_result is not None and ga_result.leaderboard:
                    for cand in ga_result.leaderboard[: max(int(cfg.pbo_max_candidates), 2)]:
                        try:
                            pbo_specs.append(_leaderboard_candidate_spec(cand, final_source_type))
                        except Exception:  # noqa: BLE001 -- one bad genome must not kill the gate
                            continue
                if len(pbo_specs) >= 2:
                    pbo_res = compute_pbo(
                        dev_df, pbo_specs, risk,
                        n_groups=cfg.cpcv_n_groups, n_test_groups=cfg.cpcv_n_test_groups,
                        max_paths=cfg.pbo_max_paths,
                        metric=cfg.oos_check_metric, prop_rules=prop_rules,
                        mc_cfg=MonteCarloConfig(n_simulations=cfg.ga_search_mc_sims, random_seed=cfg.random_seed, reset_on_breach=cfg.reset_on_breach),
                    )
                    pbo_gate_result = pbo_gate(pbo_res, max_pbo=cfg.pbo_max)
                    log(f"  {pbo_gate_result.reason}")
                else:
                    pbo_skip_reason = (
                        f"PBO gate not run: only {len(pbo_specs)} candidate(s) available "
                        "(need >= 2 for the measurement to be meaningful)."
                    )
                    log(f"  {pbo_skip_reason}")
            except Exception as exc:  # noqa: BLE001 -- best-effort validation step
                pbo_skip_reason = f"PBO gate failed to run: {exc}"
                log(f"  {pbo_skip_reason}")

        # -- Pipeline reorg extra: parsimony (section 24) --------------------
        # Reward strategies with fewer unnecessary degrees of freedom --
        # a small, additive scorecard component, never a gate. Cheap
        # (static analysis of the final strategy's own config/source),
        # so this always runs.
        parsimony_result = compute_parsimony(final_strategy)
        log(f"  Parsimony: {parsimony_result.notes[0] if parsimony_result.notes else 'not scored'}")

        # -- Pipeline reorg extra: regime diagnostics (report-only,
        # section 13/14 -- never scored, never gated) -----------------------
        regime_result = None
        regime_skip_reason = None
        if cfg.regime_diagnostics_enabled:
            try:
                from app.validation.regime_matrix import build_regime_matrix
                regime_result = build_regime_matrix(dev_df, final_bt.trades, risk.initial_balance)
                worst = regime_result.disable_regimes()
                if worst:
                    log(f"  Regime diagnostics: {len(worst)} regime(s) flagged as candidates to disable -- see report (informational only).")
                else:
                    log("  Regime diagnostics: no regime flagged for disabling.")
            except Exception as exc:  # noqa: BLE001 -- report-only diagnostic, never allowed to affect the verdict
                regime_skip_reason = f"Regime diagnostics failed to run: {exc}"
                log(f"  {regime_skip_reason}")

        # -- Step 7: report + save -----------------------------------------
        _check_cancel()
        log("Step 7/7: Generating final report...")
        verdict, verdict_reasons, scorecard, risk_of_ruin_hard_fail, lookahead_hard_fail = _make_verdict(
            final_mc, oos_validation, icir_gate,
            statistics=final_bt.statistics, prop_rules=prop_rules,
            risk_of_ruin_cap=cfg.risk_of_ruin_cap, parsimony=parsimony_result,
            cpcv_primary_result=cpcv_primary_result, cpcv_supporting_result=cpcv_supporting_result,
            lookahead_bug_detected=lookahead_bug_detected,
            min_trades_for_ready=cfg.min_trades_for_ready,
            holdout=final_holdout,
            min_holdout_trades_for_gate=cfg.min_holdout_trades_for_gate,
            min_per_attempt_pass_pct=cfg.min_per_attempt_pass_pct,
            dsr_gate_result=dsr_gate_result, pbo_gate_result=pbo_gate_result,
            primary_robustness_method=cfg.primary_robustness_method,
            gates_advisory_only=cfg.validation_gates_advisory_only,
        )

        elapsed = time.time() - t0
        return _finish(
            strategy, display_name, baseline_bt, baseline_single_run, baseline_mc, lookahead_summary,
            refinement_ran, refinement_skip_reason, ga_result,
            final_source_type, final_config, final_code_text, final_code_ext,
            final_bt, final_single_run, final_mc, final_holdout,
            full_history_bt, full_history_single_run,
            oos_validation, oos_skip_reason, icir_gate, icir_gate_skip_reason, verdict, verdict_reasons,
            scorecard, risk_of_ruin_hard_fail, lookahead_hard_fail, parsimony_result,
            cpcv_primary_result or cpcv_supporting_result, cpcv_skip_reason, regime_result, regime_skip_reason,
            df, prop_rules, risk, cfg, elapsed, warnings, log, output_dir,
            instrument, report_basename, account_mismatch_warning,
            dsr_gate_result=dsr_gate_result, dsr_skip_reason=dsr_skip_reason,
            pbo_gate_result=pbo_gate_result, pbo_skip_reason=pbo_skip_reason,
        )
    finally:
        if final_tmp_dir is not None:
            rmtree(final_tmp_dir, ignore_errors=True)


def _finish(
    strategy, display_name, baseline_bt, baseline_single_run, baseline_mc, lookahead_summary,
    refinement_ran, refinement_skip_reason, ga_result,
    final_source_type, final_config, final_code_text, final_code_ext,
    final_bt, final_single_run, final_mc, final_holdout,
    full_history_bt, full_history_single_run,
    oos_validation, oos_skip_reason, icir_gate, icir_gate_skip_reason, verdict, verdict_reasons,
    scorecard, risk_of_ruin_hard_fail, lookahead_hard_fail, parsimony_result, cpcv_result, cpcv_skip_reason,
    regime_result, regime_skip_reason,
    df, prop_rules, risk, cfg, elapsed, warnings, log, output_dir,
    instrument="unknown", report_basename="full_pipeline_report", account_mismatch_warning=None,
    dsr_gate_result: "DeflatedSharpeGateResult | None" = None,
    dsr_skip_reason: str | None = None,
    pbo_gate_result: "PBOGateResult | None" = None,
    pbo_skip_reason: str | None = None,
) -> FullPipelineResult:
    """Writes the report + (for code strategies) saves the winner into the
    Strategy Library. Split out of run_full_pipeline only to keep that
    function's main try/finally block readable."""
    from app.reports.generator import generate_full_report
    from tempfile import mkdtemp
    from shutil import rmtree

    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
    final_strategy_name = f"{display_name} (Full Pipeline)"

    # 2026-09-17 Quick-Optimize-vs-Full-Pipeline naming-drift fix, point (3):
    # final_config/final_code_text above only ever get replaced with the
    # GA's mutated winner when refinement actually ran AND that winner
    # produced OOS trades (see the "if ga_result.best.oos_trade_count > 0"
    # branch just before this function is called) -- any other case keeps
    # the ORIGINAL baseline configuration, whose display_name is still
    # accurate. save_name below is what both the saved filename AND (for
    # manual configs) the saved JSON's own "name" field are derived from --
    # see app.strategy.library.provenance_stamped_name's docstring for the
    # exact bug this closes (the same fix as app.orchestration.
    # quick_optimize.run_quick_optimize's save_name).
    mutated_by_ga = refinement_ran and ga_result is not None and ga_result.best.oos_trade_count > 0
    save_name = (
        provenance_stamped_name(display_name, origin="full_pipeline", seed=cfg.random_seed)
        if mutated_by_ga else display_name
    )
    # PARAMETER-FIDELITY FIX (2026-09-24): make the report's own title say
    # when the backtested parameters are NOT the ones display_name/save_
    # name describe -- see the baseline_parameters comment further below
    # for the exact confusion this closes. Only the title changes here;
    # save_name (what gets written to disk) is untouched.
    if mutated_by_ga:
        final_strategy_name = f"{display_name} (Full Pipeline, GA-modified parameters -- see Final vs Baseline Parameters)"

    # Pipeline reorg: every record_backtest_result() call below also
    # stamps these three fields into the strategy's "last_run" metadata
    # -- this is the ONLY change needed to make app.scoring.leaderboard's
    # cross-tool Final Selection leaderboard (item #5/#3 of the reorg
    # plan) possible, since list_saved_strategies() already surfaces
    # this same metadata for every strategy in the library regardless of
    # which tool (Full Pipeline, Forge, Evolution Lab, Search Lab) wrote
    # it last.
    t58_score = scorecard.score if scorecard is not None else None
    t58_tier = scorecard.tier if scorecard is not None else None
    parsimony_score = parsimony_result.score if parsimony_result is not None else None

    final_parameters = None
    baseline_parameters = None
    if ga_result is not None and ga_result.genes:
        final_parameters = {
            gene.label: (str(int(round(value))) if gene.is_int else f"{value:.4f}".rstrip("0").rstrip("."))
            for gene, value in zip(ga_result.genes, ga_result.best.genome)
        }
        # PARAMETER-FIDELITY FIX (2026-09-24): found via an external
        # RoboQuant comparison -- the GA can (and, on that comparison's ES
        # strategy, did) mutate periods/thresholds far away from whatever
        # was originally supplied, while final_strategy_name below kept
        # showing the SUPPLIED strategy's static, hand-typed name (which
        # can itself have specific parameter values baked into it, e.g.
        # "...optimized (vwma31/86, ema458/244, rsi19/9, 1H)..."). Anyone
        # comparing this report's headline numbers against a backtest of
        # the ORIGINALLY-NAMED parameters elsewhere was silently comparing
        # two different rules. baseline_parameters (each gene's pre-GA
        # value, already tracked on GeneMeta/CodeGene.base_value for the
        # search bounds themselves) lets the report show both sets side
        # by side instead of only the final one.
        baseline_parameters = {
            gene.label: (str(int(round(gene.base_value))) if gene.is_int else f"{gene.base_value:.4f}".rstrip("0").rstrip("."))
            for gene in ga_result.genes
        }

    # BUGFIX (2026-09-19): `final_strategy` (the Strategy object) is a local
    # of run_full_pipeline's try-block and was never threaded into this
    # split-out function, so this call always raised NameError before a
    # report could ever be written (every Full Pipeline run hit this).
    # Rebuild the equivalent Strategy object here from final_config /
    # final_source_type, which _are_ passed through -- describe_resolved_timeframe
    # only reads the strategy's declared timeframe, so this is sufficient.
    if final_source_type == "manual":
        _report_strategy = build_strategy_from_spec(_spec_for_manual(final_config), None)
    else:
        _tmp_report_dir = Path(mkdtemp(prefix="t58_fullpipeline_report_"))
        try:
            _report_strategy = build_strategy_from_spec(
                _spec_for_code(final_source_type, final_code_text, final_code_ext), _tmp_report_dir
            )
        finally:
            rmtree(_tmp_report_dir, ignore_errors=True)

    report_paths = generate_full_report(
        output_dir=output_dir,
        strategy_name=final_strategy_name,
        strategy_source_type=final_source_type,
        instrument=instrument,
        timeframe=describe_resolved_timeframe(_report_strategy, df),
        backtest_period=period,
        backtest_result=full_history_bt,
        prop_rules=prop_rules,
        prop_single_run=full_history_single_run,
        monte_carlo_result=final_mc,
        basename=report_basename,
        holdout_comparison=final_holdout,
        risk_config=risk,
        price_df=df,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        final_parameters=final_parameters,
        baseline_parameters=baseline_parameters,
    )

    saved_library_path = None
    saved_library_note = None
    if cfg.save_to_library and final_source_type in ("python", "pinescript", "mql5") and final_code_text:
        ext = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}[final_source_type]
        status_to_set = cfg.library_status or _VERDICT_TO_LIBRARY_STATUS.get(verdict, "tested_passed")
        if cfg.replace_existing and cfg.library_ref:
            # VERSIONING: same in-place-replace this Full Pipeline run
            # started from gets applied to what it produces too -- see
            # app.orchestration.quick_optimize's identical branch (and
            # app.strategy.library.save_strategy_replacing_version's own
            # docstring) for the full rationale. Archives the version
            # being replaced and carries forward every existing
            # pipeline-progress field on that file instead of starting a
            # fresh, disconnected copy.
            ref_type, ref_filename = cfg.library_ref
            try:
                saved_library_path = save_strategy_replacing_version(final_code_text, ref_type, ref_filename)
                filename = ref_filename
                set_strategy_status(ref_type, filename, status_to_set)
                record_backtest_result(ref_type, filename, {
                    "trades": len(final_bt.trades),
                    "net_profit": round(final_bt.statistics.net_profit, 2),
                    "win_rate": round(final_bt.statistics.win_rate, 1),
                    "max_dd": round(final_bt.statistics.max_drawdown_pct, 2),
                    "eval_pass_probability": round(final_mc.evaluation_pass_probability, 1),
                    "first_payout_probability": round(final_mc.first_payout_probability, 1),
                    "verdict": verdict,
                    "report_html": str(report_paths["html"]),
                    "t58_score": round(t58_score, 1) if t58_score is not None else None,
                    "t58_tier": t58_tier,
                    "parsimony_score": round(parsimony_score, 1) if parsimony_score is not None else None,
                    "risk_of_ruin_pct": round(final_mc.risk_of_ruin_pct, 1),
                    "risk_of_ruin_hard_fail": risk_of_ruin_hard_fail,
                    "lookahead_hard_fail": lookahead_hard_fail,
                    "first_payout_amount": full_history_single_run.first_payout_amount,
                    "days_to_first_payout": full_history_single_run.first_payout_day_index,
                    "total_payouts_full_history": len(full_history_single_run.payouts),
                })
                if refinement_ran and ga_result is not None:
                    record_optimize_result(ref_type, filename, {
                        "method": "full_pipeline_ga", "oos_trade_count": ga_result.best.oos_trade_count,
                    })
                if oos_validation is not None or cpcv_result is not None:
                    record_validation_result(ref_type, filename, {
                        "method": "cpcv" if cpcv_result is not None else "walk_forward",
                        "efficiency": getattr(cpcv_result, "mean_oos_metric", None) if cpcv_result is not None else getattr(oos_validation, "efficiency", None),
                    })
                record_champion_check_result(ref_type, filename, {
                    "verdict": verdict,
                    "t58_score": round(t58_score, 1) if t58_score is not None else None,
                    "t58_tier": t58_tier,
                })
                saved_library_note = f"Replaced '{filename}' in the Strategy Library (previous version archived, status: {status_to_set})." + _library_save_note_suffix(verdict, verdict_reasons)
                log(f"  {saved_library_note}")
            except Exception as exc:  # noqa: BLE001 -- saving to the library is a convenience, not core output
                saved_library_note = f"Could not replace the Strategy Library entry: {exc}"
                log(f"  {saved_library_note}")
        else:
            base_name = safe_filename_stem(save_name, "full_pipeline_strategy")
            filename = f"{base_name}_pipeline{ext}"
            try:
                try:
                    saved_library_path = save_strategy_text(final_code_text, filename, final_source_type, overwrite=False)
                except StrategyAlreadyExists:
                    filename = f"{base_name}_pipeline_{int(time.time())}{ext}"
                    saved_library_path = save_strategy_text(final_code_text, filename, final_source_type, overwrite=False)
                set_strategy_status(final_source_type, filename, status_to_set)
                record_backtest_result(final_source_type, filename, {
                    "trades": len(final_bt.trades),
                    "net_profit": round(final_bt.statistics.net_profit, 2),
                    "win_rate": round(final_bt.statistics.win_rate, 1),
                    "max_dd": round(final_bt.statistics.max_drawdown_pct, 2),
                    "eval_pass_probability": round(final_mc.evaluation_pass_probability, 1),
                    "first_payout_probability": round(final_mc.first_payout_probability, 1),
                    "verdict": verdict,
                    "report_html": str(report_paths["html"]),
                    "t58_score": round(t58_score, 1) if t58_score is not None else None,
                    "t58_tier": t58_tier,
                    "parsimony_score": round(parsimony_score, 1) if parsimony_score is not None else None,
                    "risk_of_ruin_pct": round(final_mc.risk_of_ruin_pct, 1),
                    "risk_of_ruin_hard_fail": risk_of_ruin_hard_fail,
                    "lookahead_hard_fail": lookahead_hard_fail,
                    # ADDED (2026-09-26): first-payout economics from the
                    # full-history single run (see full_history_single_run
                    # above) -- the Strategy Library couldn't show either of
                    # these before because nothing recorded them here.
                    "first_payout_amount": full_history_single_run.first_payout_amount,
                    "days_to_first_payout": full_history_single_run.first_payout_day_index,
                    "total_payouts_full_history": len(full_history_single_run.payouts),
                })
                # Pipeline-progress tracker (Create/Test/Optimize/Validate/
                # Champion Check/Ready -- see app.strategy.library.
                # compute_pipeline_progress): Full Pipeline's Step 2 GA re-
                # optimization, Step 4/6b OOS or CPCV validation, and its own
                # final verdict all happen inside this ONE run, so all three
                # downstream stages get stamped here in addition to the
                # "Test" stage record_backtest_result already covers above.
                if refinement_ran and ga_result is not None:
                    record_optimize_result(final_source_type, filename, {
                        "method": "full_pipeline_ga", "oos_trade_count": ga_result.best.oos_trade_count,
                    })
                if oos_validation is not None or cpcv_result is not None:
                    record_validation_result(final_source_type, filename, {
                        "method": "cpcv" if cpcv_result is not None else "walk_forward",
                        "efficiency": getattr(cpcv_result, "mean_oos_metric", None) if cpcv_result is not None else getattr(oos_validation, "efficiency", None),
                    })
                record_champion_check_result(final_source_type, filename, {
                    "verdict": verdict,
                    "t58_score": round(t58_score, 1) if t58_score is not None else None,
                    "t58_tier": t58_tier,
                })
                saved_library_note = f"Saved to the Strategy Library as '{filename}' (status: {status_to_set})." + _library_save_note_suffix(verdict, verdict_reasons)
                log(f"  {saved_library_note}")
            except Exception as exc:  # noqa: BLE001 -- saving to the library is a convenience, not core output
                saved_library_note = f"Could not save to the Strategy Library: {exc}"
                log(f"  {saved_library_note}")
    elif cfg.save_to_library and final_source_type == "manual" and final_config:
        # Manual Strategy Builder / Search Lab / Evolution Lab configs are
        # dicts, not source files -- but app.strategy.library already has a
        # first-class "manual" strategy type that stores exactly this shape
        # as JSON (see library.py's STRATEGY_TYPES and Evolution Lab's own
        # PROMOTE button, _promote_evolution_leader_record). This used to
        # fall through to the "nothing to save" message below even though
        # the library could save it perfectly well -- fixed by using the
        # same json.dumps(config, indent=2) + save_strategy_text(..., "manual",
        # ...) pattern Evolution Lab already relies on. Also mirrors the
        # saved JSON into final_code_text/final_code_extension so a "view
        # code" action downstream (e.g. Speed Run's candidate list) has
        # something to show for a manual-builder winner too, not just for
        # python/pinescript/mql5 ones.
        status_to_set = cfg.library_status or _VERDICT_TO_LIBRARY_STATUS.get(verdict, "tested_passed")
        config_to_save = dict(final_config)
        if cfg.replace_existing and cfg.library_ref:
            ref_type, ref_filename = cfg.library_ref
            config_text = json.dumps(config_to_save, indent=2)
            try:
                saved_library_path = save_strategy_replacing_version(config_text, ref_type, ref_filename)
                filename = ref_filename
                set_strategy_status(ref_type, filename, status_to_set)
                record_backtest_result(ref_type, filename, {
                    "trades": len(final_bt.trades),
                    "net_profit": round(final_bt.statistics.net_profit, 2),
                    "win_rate": round(final_bt.statistics.win_rate, 1),
                    "max_dd": round(final_bt.statistics.max_drawdown_pct, 2),
                    "eval_pass_probability": round(final_mc.evaluation_pass_probability, 1),
                    "first_payout_probability": round(final_mc.first_payout_probability, 1),
                    "verdict": verdict,
                    "report_html": str(report_paths["html"]),
                    "t58_score": round(t58_score, 1) if t58_score is not None else None,
                    "t58_tier": t58_tier,
                    "parsimony_score": round(parsimony_score, 1) if parsimony_score is not None else None,
                    "risk_of_ruin_pct": round(final_mc.risk_of_ruin_pct, 1),
                    "risk_of_ruin_hard_fail": risk_of_ruin_hard_fail,
                    "lookahead_hard_fail": lookahead_hard_fail,
                    "first_payout_amount": full_history_single_run.first_payout_amount,
                    "days_to_first_payout": full_history_single_run.first_payout_day_index,
                    "total_payouts_full_history": len(full_history_single_run.payouts),
                })
                if refinement_ran and ga_result is not None:
                    record_optimize_result(ref_type, filename, {
                        "method": "full_pipeline_ga", "oos_trade_count": ga_result.best.oos_trade_count,
                    })
                if oos_validation is not None or cpcv_result is not None:
                    record_validation_result(ref_type, filename, {
                        "method": "cpcv" if cpcv_result is not None else "walk_forward",
                        "efficiency": getattr(cpcv_result, "mean_oos_metric", None) if cpcv_result is not None else getattr(oos_validation, "efficiency", None),
                    })
                record_champion_check_result(ref_type, filename, {
                    "verdict": verdict,
                    "t58_score": round(t58_score, 1) if t58_score is not None else None,
                    "t58_tier": t58_tier,
                })
                saved_library_note = f"Replaced '{filename}' in the Strategy Library (previous version archived, status: {status_to_set})." + _library_save_note_suffix(verdict, verdict_reasons)
                log(f"  {saved_library_note}")
                final_code_text = config_text
                final_code_ext = ".json"
            except Exception as exc:  # noqa: BLE001 -- saving to the library is a convenience, not core output
                saved_library_note = f"Could not replace the Strategy Library entry: {exc}"
                log(f"  {saved_library_note}")
        else:
            base_name = safe_filename_stem(save_name, "full_pipeline_strategy")
            filename = f"{base_name}_pipeline.json"
            # 2026-09-17 naming-drift fix, continued: overwrite the saved
            # JSON's own "name" field with save_name too when mutated --
            # otherwise the file on disk would still claim the ORIGINAL
            # (now-inaccurate) name internally even though its filename was
            # fixed. Mirrors app.orchestration.quick_optimize's identical fix.
            if mutated_by_ga:
                config_to_save["name"] = save_name
            config_text = json.dumps(config_to_save, indent=2)
            try:
                try:
                    saved_library_path = save_strategy_text(config_text, filename, "manual", overwrite=False)
                except StrategyAlreadyExists:
                    filename = f"{base_name}_pipeline_{int(time.time())}.json"
                    saved_library_path = save_strategy_text(config_text, filename, "manual", overwrite=False)
                set_strategy_status("manual", filename, status_to_set)
                record_backtest_result("manual", filename, {
                    "trades": len(final_bt.trades),
                    "net_profit": round(final_bt.statistics.net_profit, 2),
                    "win_rate": round(final_bt.statistics.win_rate, 1),
                    "max_dd": round(final_bt.statistics.max_drawdown_pct, 2),
                    "eval_pass_probability": round(final_mc.evaluation_pass_probability, 1),
                    "first_payout_probability": round(final_mc.first_payout_probability, 1),
                    "verdict": verdict,
                    "report_html": str(report_paths["html"]),
                    "t58_score": round(t58_score, 1) if t58_score is not None else None,
                    "t58_tier": t58_tier,
                    "parsimony_score": round(parsimony_score, 1) if parsimony_score is not None else None,
                    "risk_of_ruin_pct": round(final_mc.risk_of_ruin_pct, 1),
                    "risk_of_ruin_hard_fail": risk_of_ruin_hard_fail,
                    "lookahead_hard_fail": lookahead_hard_fail,
                    "first_payout_amount": full_history_single_run.first_payout_amount,
                    "days_to_first_payout": full_history_single_run.first_payout_day_index,
                    "total_payouts_full_history": len(full_history_single_run.payouts),
                })
                if refinement_ran and ga_result is not None:
                    record_optimize_result("manual", filename, {
                        "method": "full_pipeline_ga", "oos_trade_count": ga_result.best.oos_trade_count,
                    })
                if oos_validation is not None or cpcv_result is not None:
                    record_validation_result("manual", filename, {
                        "method": "cpcv" if cpcv_result is not None else "walk_forward",
                        "efficiency": getattr(cpcv_result, "mean_oos_metric", None) if cpcv_result is not None else getattr(oos_validation, "efficiency", None),
                    })
                record_champion_check_result("manual", filename, {
                    "verdict": verdict,
                    "t58_score": round(t58_score, 1) if t58_score is not None else None,
                    "t58_tier": t58_tier,
                })
                saved_library_note = f"Saved to the Strategy Library as '{filename}' (status: {status_to_set})." + _library_save_note_suffix(verdict, verdict_reasons)
                log(f"  {saved_library_note}")
                final_code_text = config_text
                final_code_ext = ".json"
            except Exception as exc:  # noqa: BLE001 -- saving to the library is a convenience, not core output
                saved_library_note = f"Could not save to the Strategy Library: {exc}"
                log(f"  {saved_library_note}")
    elif final_source_type == "manual":
        saved_library_note = (
            "Manual Strategy Builder configuration produced, but nothing was saved (saving to the "
            "Strategy Library is turned off, or no configuration was available) -- copy the winning "
            "settings from the report, or use 'Apply Best Config to Strategy Tab' after a standalone "
            "Iterative Refinement run."
        )

    log(f"\nFull Pipeline complete in {elapsed:.1f}s. Verdict: {verdict}.")

    try:
        from app.ai.experiment_memory import record_experiment

        record_experiment(
            origin="full_pipeline",
            strategy_name=final_strategy_name,
            source_type=final_source_type,
            instrument=instrument,
            verdict=verdict,
            trades=len(final_bt.trades),
            net_profit=final_bt.statistics.net_profit,
            win_rate=final_bt.statistics.win_rate,
            profit_factor=final_bt.statistics.profit_factor,
            max_drawdown_pct=final_bt.statistics.max_drawdown_pct,
            eval_pass_probability=final_mc.evaluation_pass_probability,
            first_payout_probability=final_mc.first_payout_probability,
            risk_of_ruin_pct=final_mc.risk_of_ruin_pct,
            lesson="; ".join(verdict_reasons) if verdict_reasons else "",
        )
    except Exception:
        pass  # T58 Research Memory is a bonus record -- never let it affect a completed Full Pipeline run

    return FullPipelineResult(
        strategy_source_type=strategy.source_type,
        strategy_display_name=display_name,
        baseline_bt=baseline_bt,
        baseline_single_run=baseline_single_run,
        baseline_mc=baseline_mc,
        lookahead_summary=lookahead_summary,
        refinement_ran=refinement_ran,
        refinement_skip_reason=refinement_skip_reason,
        ga_result=ga_result,
        final_source_type=final_source_type,
        final_config=final_config,
        final_code_text=final_code_text,
        final_code_extension=final_code_ext,
        final_bt=final_bt,
        final_single_run=final_single_run,
        final_mc=final_mc,
        final_holdout=final_holdout,
        full_history_single_run=full_history_single_run,
        oos_validation=oos_validation,
        oos_validation_skip_reason=oos_skip_reason,
        icir_gate=icir_gate,
        icir_gate_skip_reason=icir_gate_skip_reason,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        scorecard=scorecard,
        risk_of_ruin_hard_fail=risk_of_ruin_hard_fail,
        lookahead_hard_fail=lookahead_hard_fail,
        risk_of_ruin_cap=cfg.risk_of_ruin_cap,
        parsimony=parsimony_result,
        cpcv_result=cpcv_result,
        cpcv_skip_reason=cpcv_skip_reason,
        dsr_gate_result=dsr_gate_result,
        dsr_skip_reason=dsr_skip_reason,
        pbo_gate_result=pbo_gate_result,
        pbo_skip_reason=pbo_skip_reason,
        validation_gate_failures=_validation_gate_failure_names(verdict_reasons),
        regime_result=regime_result,
        regime_skip_reason=regime_skip_reason,
        saved_library_path=saved_library_path,
        saved_library_note=saved_library_note,
        report_paths=report_paths,
        elapsed_seconds=elapsed,
        account_mismatch_warning=account_mismatch_warning,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Batch Full Pipeline -- run the WHOLE 7-step pipeline (not just a plain
# backtest) against every strategy in a list, one after another.
# ---------------------------------------------------------------------------

# How long the parallel batch pool can go with ZERO completions before a
# still-pending item is treated as stalled (a wedged worker, or one that
# silently died) rather than just legitimately slow. Generous by design --
# a real 7-step Full Pipeline run can take a long time per item -- but
# finite, unlike the bare `as_completed()` this replaced. Module-level (not
# a literal inline) so tests can shrink it instead of waiting 30 minutes.
FULL_PIPELINE_BATCH_STALL_TIMEOUT_SECONDS = 1800.0


def _drain_batch_pool_futures(
    pool, futures: dict, cancel_event: "threading.Event | None", on_done, log,
    stall_timeout: float = FULL_PIPELINE_BATCH_STALL_TIMEOUT_SECONDS,
) -> None:
    """Consumes {future: (i, label)} as each item's pipeline run finishes --
    used by run_full_pipeline_batch's parallel path INSTEAD of Python's
    `for future in as_completed(futures)`, which blocks until the NEXT
    future completes with no timeout at all: one wedged worker (a hung GA
    search inside one item's pipeline, or a worker process that silently
    died) used to block the WHOLE batch indefinitely, with no way to stop
    it either -- exactly the "the full pipeline stalled when I tried batch
    generations" report this fixes. This is the same fix shape already
    applied to Search Lab's 3 stages and the Evolution Lab (see
    app.search.batch_runner._drain_futures / app.evolution.engine's
    version) -- Full Pipeline's batch path had its own separate, still
    unfixed, copy of the old blocking pattern.

    Polls with a 1s timeout so cancel_event gets checked regardless of how
    long any individual item's pipeline takes, and raises TimeoutError if
    stall_timeout passes with zero completions (the caller's existing
    BrokenProcessPool `except` already falls back to running whatever
    hadn't finished serially in-process, so a genuine stall recovers via
    that same path rather than needing a second one)."""
    pending = set(futures.keys())
    last_progress = time.time()
    while pending:
        if cancel_event is not None and cancel_event.is_set():
            log("\nStop requested -- cancelling remaining strategy(ies) and shutting down workers...")
            for fut in pending:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise FullPipelineBatchCancelled("Full Pipeline batch stopped by user.")
        done, pending = futures_wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
        if done:
            last_progress = time.time()
            for future in done:
                on_done(futures[future], future)
        elif pending and (time.time() - last_progress) > stall_timeout:
            raise TimeoutError(
                f"no progress for {int(stall_timeout)}s across {len(pending)} "
                f"still-running strategy(ies) -- worker pool appears stalled"
            )


class FullPipelineBatchCancelled(Exception):
    """Raised when the caller sets ``cancel_event`` mid-batch. Not an
    error -- the UI catches this to report a clean user-requested stop
    rather than a crash. Every item that had already finished before the
    stop is still in the returned summary's outcomes (batch_progress.json
    is also already up to date via _write_progress), so stopping a batch
    partway through doesn't lose the work it already did."""


@dataclass
class FullPipelineBatchItem:
    label: str                                    # display name (e.g. the library filename)
    strategy: Strategy
    library_ref: tuple[str, str] | None = None     # (strategy_type, filename) -- library-sourced items only


@dataclass
class FullPipelineBatchOutcome:
    label: str
    ok: bool
    reason: str | None = None                      # set when ok is False
    verdict: str | None = None                      # "READY" / "MARGINAL" / "NOT READY"
    trades: int = 0
    net_profit: float = 0.0
    eval_pass_probability: float = 0.0
    report_html: Path | None = None
    result: "FullPipelineResult | None" = None


@dataclass
class FullPipelineBatchSummary:
    outcomes: list = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def succeeded(self) -> list:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list:
        return [o for o in self.outcomes if not o.ok]


def _batch_item_worker(
    i: int, label: str, strategy: Strategy, df: pd.DataFrame, risk: RiskConfig,
    prop_rules: PropRules, output_dir: str | Path, cfg: FullPipelineConfig,
    instrument: str, ollama_settings: "OllamaSettings | None", report_basename: str,
) -> tuple[int, str, bool, "FullPipelineResult | None", str | None]:
    """Module-level (picklable) target for the batch's ProcessPoolExecutor.
    Runs exactly one item's full pipeline with no progress_cb (a Tkinter-
    bound callback can't cross a process boundary) -- per-item step-by-step
    logging is only available in the serial (max_parallel_strategies=1)
    path; the parallel path logs only start/finish per item. Returns a
    plain tuple instead of raising so one bad strategy can never take down
    `as_completed` for the rest of the batch."""
    try:
        result = run_full_pipeline(
            df, strategy, risk, prop_rules, output_dir, cfg,
            progress_cb=None, instrument=instrument, ollama_settings=ollama_settings,
            report_basename=report_basename,
        )
        return (i, label, True, result, None)
    except Exception as exc:  # noqa: BLE001 -- one bad strategy must not stop the batch
        log_crash(f"Full Pipeline batch item {i} ({label})", exc=exc)
        return (i, label, False, None, str(exc))


@dataclass
class FullPipelineSweepResult:
    per_timeframe: dict          # label -> FullPipelineResult
    best_timeframe: str
    best_result: "FullPipelineResult"
    skipped: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)


_VERDICT_RANK = {"READY": 2, "MARGINAL": 1, "NOT READY": 0}


def _sweep_rank(result: "FullPipelineResult") -> tuple:
    score = getattr(getattr(result, "scorecard", None), "score", None)
    return (
        _VERDICT_RANK.get(result.verdict, 0),
        score if score is not None else -1.0,
        result.final_mc.evaluation_pass_probability,
    )


def run_full_pipeline_sweep(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    output_dir: str | Path,
    timeframes: list[str],
    cfg: FullPipelineConfig | None = None,
    progress_cb: ProgressCallback | None = None,
    instrument: str = "unknown",
    ollama_settings: "OllamaSettings | None" = None,
    report_basename: str = "full_pipeline_report",
    cancel_event: threading.Event | None = None,
) -> FullPipelineSweepResult:
    """Runs the ENTIRE Full Pipeline once per requested timeframe, each on
    `df` resampled to that bar size (so one 1-minute file can be judged on
    5m/15m/30m/1h/4h), and returns every result plus the winner.

    Winner = verdict (READY > MARGINAL > NOT READY), then T58 score, then
    eval-pass probability. Each timeframe keeps its own full report
    (`<report_basename>__<label>`). Manual strategies are stamped with the
    timeframe they were found on so a saved winner keeps trading on it.
    In-place replace is disabled for sweeps (several timeframes must not
    all overwrite one library file). An empty/unusable timeframe list
    falls back to one native run. FullPipelineCancelled propagates.
    """
    from app.data.timeframe_sweep import build_timeframe_sweep, stamp_manual_timeframe

    cfg = cfg or FullPipelineConfig()
    plan = build_timeframe_sweep(df, timeframes or [])
    pairs = [(t.label, t.dataframe) for t in plan.targets] if plan.targets else [("native", df)]
    sweeping = bool(plan.targets)
    run_cfg = replace(cfg, replace_existing=False, library_ref=None) if sweeping and len(pairs) > 1 else cfg

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    results: dict = {}
    errors: dict = {}
    for label, target_df in pairs:
        if cancel_event is not None and cancel_event.is_set():
            raise FullPipelineCancelled("Full Pipeline sweep stopped by request.")
        log(f"=== Timeframe {label}: starting Full Pipeline ({len(target_df):,} bars) ===")
        run_strategy = stamp_manual_timeframe(strategy, label) if sweeping else strategy
        try:
            results[label] = run_full_pipeline(
                target_df, run_strategy, risk, prop_rules, output_dir, run_cfg,
                progress_cb=(lambda m, _l=label: log(f"[{_l}] {m}")),
                instrument=instrument, ollama_settings=ollama_settings,
                report_basename=f"{report_basename}__{label}" if sweeping else report_basename,
                cancel_event=cancel_event,
            )
        except FullPipelineCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 -- one bad timeframe must not kill the rest
            errors[label] = str(exc)
            log(f"[{label}] skipped -- {exc}")

    if not results:
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items()) or "no timeframe could be resampled from this dataset."
        raise RefinementError(f"Full Pipeline's timeframe sweep produced no usable timeframe -- {detail}")

    best = max(results, key=lambda k: _sweep_rank(results[k]))
    if sweeping and len(results) > 1:
        results[best].warnings.append(
            f"Timeframe sweep: {best} was the best of {len(results)} timeframes tested on the same history "
            "(verdict, then T58 score, then eval-pass probability). Choosing a winner among several is a "
            "mild selection effect -- confirm it on data this run never saw (forward test) before going live."
        )
    return FullPipelineSweepResult(
        per_timeframe=results, best_timeframe=best, best_result=results[best], skipped=plan.skipped, errors=errors,
    )


def run_full_pipeline_batch(
    df: pd.DataFrame,
    items: list[FullPipelineBatchItem],
    risk: RiskConfig,
    prop_rules: PropRules,
    output_dir: str | Path,
    cfg: FullPipelineConfig | None = None,
    instrument: str = "unknown",
    ollama_settings: "OllamaSettings | None" = None,
    progress_cb: ProgressCallback | None = None,
    max_parallel_strategies: int = 1,
    cancel_event: threading.Event | None = None,
) -> FullPipelineBatchSummary:
    """Runs app.orchestration.full_pipeline.run_full_pipeline (the full
    baseline -> walk-forward-aware GA -> final validation -> OOS check ->
    holdout check -> ICIR gate -> report pipeline, not just a plain
    backtest) against every item in `items`, writing one full report per
    strategy and recording each result back onto its own Strategy Library
    metadata exactly like a single Full Pipeline run already does.

    This exists specifically because Strategy Library multi-select +
    the batch queue previously only ever fed app.orchestration.batch_test
    (a plain backtest -> prop-sim -> Monte Carlo pipeline) -- there was no
    way to run the full 7-step Full Pipeline against more than one
    strategy without loading and running each one individually. This is
    the batch equivalent of that: same idea as run_batch_test, just
    calling the heavier, more thorough pipeline per item instead.

    One bad strategy (backtest error, zero trades, a GA/validation step
    that fails) is recorded as a failed outcome and the rest of the batch
    keeps going -- it never aborts the whole run. Every item uses the same
    `cfg` (GA population/generations/etc.) and the same `risk`/`prop_rules`
    -- whatever is configured on the Full Pipeline tab at the moment the
    batch is started.

    max_parallel_strategies: 1 (default) preserves the original strictly-
    sequential behavior with full live per-step logging. Set higher (e.g.
    3-4 on an 8+ core machine) to run that many strategies' pipelines
    concurrently in separate worker processes -- this was the single
    biggest lever for cutting a large batch's wall-clock time (a real
    23-strategy/~1-hour batch is CPU-bound almost entirely inside Step 2's
    GA search, which already parallelizes ACROSS a genome population; this
    additionally parallelizes ACROSS strategies). To avoid oversubscribing
    the machine, each item's own GA worker-process count
    (cfg.parallel_search_max_workers) is automatically capped to roughly
    os.cpu_count() // max_parallel_strategies (minimum 1) whenever the
    caller hasn't already pinned an explicit value; pass a `cfg` with
    parallel_search_max_workers set to override this. Per-item live
    progress logs are unavailable in this mode (see _batch_item_worker) --
    only start/finish lines are logged for each item; drop back to 1 if
    you need the detailed step-by-step console output for every strategy."""
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    cfg = cfg or FullPipelineConfig()
    t0 = time.time()
    outcomes_by_index: dict[int, FullPipelineBatchOutcome] = {}

    # Every full 7-step pipeline in this batch can easily run for tens of
    # minutes each (a 6-year, 1-minute-bar dataset especially so); the
    # per-item HTML report only appears once THAT item fully finishes, and
    # nothing at all was ever written summarizing the batch as a whole. If
    # the process dies partway through (crash, forced shutdown, OOM) -- as
    # opposed to one strategy cleanly failing, which _record already
    # handles -- everything that HAD finished earlier in the batch was
    # otherwise invisible unless you already knew to go hunting for each
    # item's individual report file. This writes a small running summary
    # to `output_dir/batch_progress.json` after every item finishes (pass
    # or fail), so a batch that's interrupted at item 3 of 7 still leaves
    # a readable record of what happened to items 1-3.
    progress_path = Path(output_dir) / "batch_progress.json"

    def _write_progress(done_count: int) -> None:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            payload = {
                "started_at": t0,
                "updated_at": time.time(),
                "total_items": len(items),
                "completed_items": done_count,
                "outcomes": [
                    {
                        "index": i,
                        "label": outcomes_by_index[i].label,
                        "ok": outcomes_by_index[i].ok,
                        "verdict": outcomes_by_index[i].verdict,
                        "reason": outcomes_by_index[i].reason,
                        "report_html": str(outcomes_by_index[i].report_html) if outcomes_by_index[i].report_html else None,
                    }
                    for i in sorted(outcomes_by_index)
                ],
            }
            tmp = progress_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            tmp.replace(progress_path)  # atomic-ish: a crash mid-write can't corrupt the last good file
        except Exception:
            pass  # best-effort -- must never break the batch itself

    def _record(i: int, label: str, ok: bool, result, reason: str | None) -> None:
        if not ok:
            log(f"  Skipped -- Full Pipeline error: {reason}")
            outcomes_by_index[i] = FullPipelineBatchOutcome(label, ok=False, reason=reason)
            _write_progress(len(outcomes_by_index))
            return
        if item_library_refs.get(i) is not None:
            try:
                from app.strategy.library import record_backtest_result
                strategy_type, filename = item_library_refs[i]
                record_backtest_result(strategy_type, filename, {
                    "trades": len(result.final_bt.trades),
                    "net_profit": round(result.final_bt.statistics.net_profit, 2),
                    "win_rate": round(result.final_bt.statistics.win_rate, 1),
                    "max_dd": round(result.final_bt.statistics.max_drawdown_pct, 2),
                    "eval_pass_probability": round(result.final_mc.evaluation_pass_probability, 1),
                    "first_payout_probability": round(result.final_mc.first_payout_probability, 1),
                    "verdict": result.verdict,
                    "report_html": str(result.report_paths["html"]),
                    "first_payout_amount": result.full_history_single_run.first_payout_amount,
                    "days_to_first_payout": result.full_history_single_run.first_payout_day_index,
                    "total_payouts_full_history": len(result.full_history_single_run.payouts),
                })
                from app.strategy.library import (
                    record_optimize_result as _rec_opt,
                    record_validation_result as _rec_val,
                    record_champion_check_result as _rec_champ,
                )
                if result.refinement_ran and result.ga_result is not None:
                    _rec_opt(strategy_type, filename, {
                        "method": "full_pipeline_ga", "oos_trade_count": result.ga_result.best.oos_trade_count,
                    })
                if result.oos_validation is not None or result.cpcv_result is not None:
                    _rec_val(strategy_type, filename, {
                        "method": "cpcv" if result.cpcv_result is not None else "walk_forward",
                        "efficiency": getattr(result.cpcv_result, "mean_oos_metric", None) if result.cpcv_result is not None else getattr(result.oos_validation, "efficiency", None),
                    })
                _rec_champ(strategy_type, filename, {
                    "verdict": result.verdict,
                    "t58_score": round(result.scorecard.score, 1) if result.scorecard is not None else None,
                    "t58_tier": result.scorecard.tier if result.scorecard is not None else None,
                })
            except Exception:  # noqa: BLE001 -- recording to the library is a convenience, not core output
                pass
        log(
            f"  Verdict: {result.verdict}  |  Trades: {len(result.final_bt.trades)}  |  "
            f"Net profit: ${result.final_bt.statistics.net_profit:,.2f}  |  "
            f"Eval pass probability: {result.final_mc.evaluation_pass_probability:.1f}%  |  "
            f"Report: {result.report_paths['html'].name}"
        )
        outcomes_by_index[i] = FullPipelineBatchOutcome(
            label, ok=True, verdict=result.verdict,
            trades=len(result.final_bt.trades),
            net_profit=result.final_bt.statistics.net_profit,
            eval_pass_probability=result.final_mc.evaluation_pass_probability,
            report_html=result.report_paths["html"], result=result,
        )
        _write_progress(len(outcomes_by_index))

    item_library_refs = {i: item.library_ref for i, item in enumerate(items, start=1)}
    safe_names = {}
    for i, item in enumerate(items, start=1):
        safe_names[i] = re.sub(r"[^A-Za-z0-9_-]+", "_", item.label) or f"strategy_{i}"

    if max_parallel_strategies <= 1 or len(items) <= 1:
        for i, item in enumerate(items, start=1):
            if cancel_event is not None and cancel_event.is_set():
                log(
                    f"\nStop requested -- {len(outcomes_by_index)}/{len(items)} strategy(ies) already "
                    f"finished and recorded above; the rest of the batch will not run."
                )
                raise FullPipelineBatchCancelled("Full Pipeline batch stopped by user.")
            log(f"\n===== [{i}/{len(items)}] Full Pipeline: {item.label} =====")

            def item_log(msg: str, _label=item.label) -> None:
                log(f"  {msg}")

            try:
                result = run_full_pipeline(
                    df, item.strategy, risk, prop_rules, output_dir, cfg,
                    progress_cb=item_log, instrument=instrument, ollama_settings=ollama_settings,
                    report_basename=f"full_pipeline_{i:03d}_{safe_names[i]}",
                )
            except Exception as exc:  # noqa: BLE001 -- one bad strategy must not stop the batch
                _record(i, item.label, False, None, str(exc))
                continue
            _record(i, item.label, True, result, None)
    else:
        per_item_workers = max(1, (os.cpu_count() or 4) // max_parallel_strategies)
        item_cfg = cfg if cfg.parallel_search_max_workers is not None else replace(
            cfg, parallel_search_max_workers=per_item_workers,
        )
        log(
            f"\nRunning {len(items)} strategies with up to {max_parallel_strategies} in parallel "
            f"(each capped to {item_cfg.parallel_search_max_workers} GA worker process(es))..."
        )
        try:
            with ProcessPoolExecutor(max_workers=max_parallel_strategies) as pool:
                futures = {
                    pool.submit(
                        _batch_item_worker, i, item.label, item.strategy, df, risk, prop_rules,
                        output_dir, item_cfg, instrument, ollama_settings,
                        f"full_pipeline_{i:03d}_{safe_names[i]}",
                    ): (i, item.label)
                    for i, item in enumerate(items, start=1)
                }
                def _on_item_done(label_tuple, future):
                    i, label = label_tuple
                    log(f"\n===== [{i}/{len(items)}] Full Pipeline: {label} (finished) =====")
                    try:
                        _, _, ok, result, reason = future.result()
                    except Exception as exc:  # noqa: BLE001 -- worker crash must not stop the batch
                        ok, result, reason = False, None, str(exc)
                    _record(i, label, ok, result, reason)

                _drain_batch_pool_futures(pool, futures, cancel_event, _on_item_done, log)
        except FullPipelineBatchCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 -- e.g. BrokenProcessPool, or the stall
            # TimeoutError raised above: either way the whole pool is
            # unusable, which normally surfaces here rather than from an
            # individual future.result() call. Whatever didn't finish yet
            # falls back to running serially in THIS process instead of
            # the entire rest of the batch silently vanishing with no
            # report and no recorded reason.
            log(f"\nParallel batch pool failed ({exc}) -- finishing the remaining strategy(ies) one at a time...")
            log_crash("Full Pipeline batch (worker pool)", exc=exc, extra=f"{len(outcomes_by_index)}/{len(items)} item(s) had already finished.")
            remaining = [
                (i, item) for i, item in enumerate(items, start=1)
                if i not in outcomes_by_index
            ]
            for i, item in remaining:
                if cancel_event is not None and cancel_event.is_set():
                    log(
                        f"\nStop requested -- {len(outcomes_by_index)}/{len(items)} strategy(ies) "
                        f"already finished and recorded above; the rest of the batch will not run."
                    )
                    raise FullPipelineBatchCancelled("Full Pipeline batch stopped by user.")
                log(f"\n===== [{i}/{len(items)}] Full Pipeline: {item.label} =====")

                def item_log(msg: str) -> None:
                    log(f"  {msg}")

                try:
                    result = run_full_pipeline(
                        df, item.strategy, risk, prop_rules, output_dir, item_cfg,
                        progress_cb=item_log, instrument=instrument, ollama_settings=ollama_settings,
                        report_basename=f"full_pipeline_{i:03d}_{safe_names[i]}",
                    )
                except Exception as item_exc:  # noqa: BLE001
                    _record(i, item.label, False, None, str(item_exc))
                    continue
                _record(i, item.label, True, result, None)

    outcomes = [outcomes_by_index[i] for i in sorted(outcomes_by_index)]
    elapsed = time.time() - t0
    log(
        f"\nBatch Full Pipeline complete in {elapsed:.1f}s. {len(items)} strategy(ies) attempted, "
        f"{sum(1 for o in outcomes if o.ok)} produced a report."
    )
    return FullPipelineBatchSummary(outcomes=outcomes, elapsed_seconds=elapsed)
