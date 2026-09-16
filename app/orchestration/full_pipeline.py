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
from app.backtest.risk import (
    RiskConfig,
    account_size_mismatch_message,
    has_impossible_condition,
    has_instrument_scale_mismatch,
    with_prop_safety_defaults,
)
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from app.optimize.code_parameter_space import patched_source_for_strategy
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig, preflight_signal_check
from app.optimize.walkforward_ga import WalkforwardGAResult, run_walkforward_aware_refinement
from app.orchestration.resource_guard import safe_worker_count
from app.prop.simulator import AccountSimResult, PropRules, simulate_account
from app.reports.crash_log import log_crash
from app.scoring.parsimony import ParsimonyResult, compute_parsimony
from app.scoring.t58_scorecard import T58ScorecardResult, score_from_results
from app.search.robustness import WalkForwardResult, run_walk_forward
from app.search.strategy_space import build_strategy_from_spec
from app.strategy.base import Strategy
from app.strategy.library import StrategyAlreadyExists, safe_filename_stem, save_strategy_text, set_strategy_status, \
    record_backtest_result
from app.validation.cpcv import CPCVError, CPCVResult, run_cpcv
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
    # Regime performance stays purely informational (pipeline reorg plan
    # section 13/14) -- attached to the report/result for a human to
    # read, never scored or gated. Reuses the SAME trades final_bt
    # already produced (build_regime_matrix, not run_regime_matrix) so
    # this costs no extra backtest -- cheap enough to default on.
    regime_diagnostics_enabled: bool = True


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

    oos_validation: WalkForwardResult | None
    oos_validation_skip_reason: str | None

    icir_gate: "ICIRGateResult | None"
    icir_gate_skip_reason: str | None

    verdict: str                # "READY" | "MARGINAL" | "NOT READY"
    verdict_reasons: list[str]

    # -- Pipeline reorg additions -----------------------------------------
    scorecard: "T58ScorecardResult | None"       # the continuous 0-100 score/tier behind `verdict`
    risk_of_ruin_hard_fail: bool                 # True if verdict is NOT READY solely because of the ruin cap
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
) -> tuple[str, list[str], "T58ScorecardResult", bool]:
    """Pipeline reorg item #1: the verdict is now a hard safety gate
    (risk of ruin) followed by app.scoring.t58_scorecard's continuous,
    missing-aware score -- NOT five independent boolean checks ANDed
    together. See that module's docstring for why: requiring every one
    of several imperfect, correlated tests to pass simultaneously can
    reject a genuinely good strategy just because one noisy measurement
    disagreed with the others (pipeline reorg plan section 4).

    Returns (verdict, verdict_reasons, scorecard_result, risk_of_ruin_hard_fail).
    `verdict` stays one of the same three strings ("READY" / "MARGINAL" /
    "NOT READY") every existing caller and test already expects --
    Elite/Strong tiers -> READY, Promising/Research -> MARGINAL, Reject
    (or a ruin hard-fail) -> NOT READY. `scorecard_result` carries the
    actual continuous score/tier/component breakdown for anything that
    wants more resolution than the 3-way verdict (e.g. the leaderboard).

    cpcv_primary_result: pass this when primary_robustness_method=="cpcv"
    for this run -- its efficiency fills the SAME scorecard slot
    oos_validation would otherwise fill (never both at once).
    cpcv_supporting_result: pass this when CPCV ran as a second,
    non-primary diagnostic (cpcv_supporting_enabled) -- scored in its
    own small-weight component instead."""
    reasons: list[str] = []
    ruin = final_mc.risk_of_ruin_pct
    ruin_hard_fail = ruin > risk_of_ruin_cap

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
        return "NOT READY", reasons, scorecard, True

    if oos_validation is None and cpcv_primary_result is None:
        reasons.append(
            "Primary generalization test couldn't run (not enough data) -- scored as UNPROVEN "
            "(missing, not failing) rather than penalized as if it had failed."
        )
    if icir_gate is None:
        reasons.append(
            "ICIR / signal-decay / Bonferroni-corrected significance gate couldn't run -- kept as "
            "a supporting diagnostic only, not part of the T58 Score."
        )
    elif not icir_gate.ok:
        reasons.append(
            "Supporting diagnostic: did NOT pass the ICIR / signal-decay / Bonferroni-corrected "
            "significance gate (" + " ".join(icir_gate.reasons) + "). Not scored into the T58 Score "
            "or gated on -- section 14 of the pipeline reorg plan treats this as informational once "
            "a primary generalization test already ran."
        )

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
    return verdict, reasons, scorecard, False


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
    if strategy.source_type in ("python", "pinescript", "mql5"):
        try:
            from app.strategy.lookahead_check import check_for_lookahead
            lookahead_result = check_for_lookahead(strategy, dev_df, max_signal_checkpoints=8)
            lookahead_summary = lookahead_result.summary()
            log(f"  Lookahead check: {lookahead_summary}")
        except Exception:
            log("  Lookahead check failed to run (skipped, best-effort only).")

    pnls = [t.pnl for t in baseline_bt.trades]
    dates = [t.entry_time for t in baseline_bt.trades]
    baseline_single_run = simulate_account(pnls, dates, prop_rules)
    baseline_mc = run_monte_carlo(baseline_bt.trades, prop_rules, MonteCarloConfig(n_simulations=cfg.baseline_mc_sims, random_seed=cfg.random_seed))
    log(
        f"  Baseline: {len(baseline_bt.trades)} trades, net ${baseline_bt.statistics.net_profit:,.2f}, "
        f"eval pass {baseline_mc.evaluation_pass_probability:.1f}%, payout {baseline_mc.first_payout_probability:.1f}%."
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
    if instrument_mismatch or impossible_condition:
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
            )
            ga_result = run_walkforward_aware_refinement(
                dev_df, strategy, risk, prop_rules,
                MonteCarloConfig(n_simulations=cfg.ga_search_mc_sims, random_seed=cfg.random_seed),
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

    from tempfile import mkdtemp
    from shutil import rmtree
    final_tmp_dir = Path(mkdtemp(prefix="t58_fullpipeline_")) if final_source_type != "manual" else None
    try:
        final_strategy = build_strategy_from_spec(final_spec, final_tmp_dir)

        # -- Step 3: final validation ------------------------------------
        _check_cancel()
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

        pnls = [t.pnl for t in final_bt.trades]
        dates = [t.entry_time for t in final_bt.trades]
        final_single_run = simulate_account(pnls, dates, prop_rules)
        final_mc = run_monte_carlo(
            final_bt.trades, prop_rules, MonteCarloConfig(n_simulations=cfg.final_mc_sims, random_seed=cfg.random_seed),
            # MC-004: these trades came from Step 2's GA search over this
            # same dev_df when refinement actually ran -- see
            # run_monte_carlo's docstring. Steps 4-6 below provide the
            # genuinely independent evidence this number alone doesn't.
            selection_bias_caveat=refinement_ran,
        )
        log(
            f"  Final: {len(final_bt.trades)} trades, net ${final_bt.statistics.net_profit:,.2f}, "
            f"eval pass {final_mc.evaluation_pass_probability:.1f}%, payout {final_mc.first_payout_probability:.1f}%."
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
                prop_rules=prop_rules, mc_cfg=MonteCarloConfig(n_simulations=cfg.ga_search_mc_sims, random_seed=cfg.random_seed),
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
            final_holdout = run_holdout_comparison(df, final_strategy, risk, holdout_frac=cfg.holdout_frac)
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
                n_candidates_tested = cfg.ga_population * (cfg.ga_generations + 1)
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
                    mc_cfg=MonteCarloConfig(n_simulations=cfg.ga_search_mc_sims, random_seed=cfg.random_seed),
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
        verdict, verdict_reasons, scorecard, risk_of_ruin_hard_fail = _make_verdict(
            final_mc, oos_validation, icir_gate,
            statistics=final_bt.statistics, prop_rules=prop_rules,
            risk_of_ruin_cap=cfg.risk_of_ruin_cap, parsimony=parsimony_result,
            cpcv_primary_result=cpcv_primary_result, cpcv_supporting_result=cpcv_supporting_result,
        )

        elapsed = time.time() - t0
        return _finish(
            strategy, display_name, baseline_bt, baseline_single_run, baseline_mc, lookahead_summary,
            refinement_ran, refinement_skip_reason, ga_result,
            final_source_type, final_config, final_code_text, final_code_ext,
            final_bt, final_single_run, final_mc, final_holdout,
            oos_validation, oos_skip_reason, icir_gate, icir_gate_skip_reason, verdict, verdict_reasons,
            scorecard, risk_of_ruin_hard_fail, parsimony_result,
            cpcv_primary_result or cpcv_supporting_result, cpcv_skip_reason, regime_result, regime_skip_reason,
            df, prop_rules, risk, cfg, elapsed, warnings, log, output_dir,
            instrument, report_basename, account_mismatch_warning,
        )
    finally:
        if final_tmp_dir is not None:
            rmtree(final_tmp_dir, ignore_errors=True)


def _finish(
    strategy, display_name, baseline_bt, baseline_single_run, baseline_mc, lookahead_summary,
    refinement_ran, refinement_skip_reason, ga_result,
    final_source_type, final_config, final_code_text, final_code_ext,
    final_bt, final_single_run, final_mc, final_holdout,
    oos_validation, oos_skip_reason, icir_gate, icir_gate_skip_reason, verdict, verdict_reasons,
    scorecard, risk_of_ruin_hard_fail, parsimony_result, cpcv_result, cpcv_skip_reason,
    regime_result, regime_skip_reason,
    df, prop_rules, risk, cfg, elapsed, warnings, log, output_dir,
    instrument="unknown", report_basename="full_pipeline_report", account_mismatch_warning=None,
) -> FullPipelineResult:
    """Writes the report + (for code strategies) saves the winner into the
    Strategy Library. Split out of run_full_pipeline only to keep that
    function's main try/finally block readable."""
    from app.reports.generator import generate_full_report

    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
    final_strategy_name = f"{display_name} (Full Pipeline)"

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
    if ga_result is not None and ga_result.genes:
        final_parameters = {
            gene.label: (str(int(round(value))) if gene.is_int else f"{value:.4f}".rstrip("0").rstrip("."))
            for gene, value in zip(ga_result.genes, ga_result.best.genome)
        }

    report_paths = generate_full_report(
        output_dir=output_dir,
        strategy_name=final_strategy_name,
        strategy_source_type=final_source_type,
        instrument=instrument,
        timeframe="unknown",
        backtest_period=period,
        backtest_result=final_bt,
        prop_rules=prop_rules,
        prop_single_run=final_single_run,
        monte_carlo_result=final_mc,
        basename=report_basename,
        holdout_comparison=final_holdout,
        risk_config=risk,
        price_df=df,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        final_parameters=final_parameters,
    )

    saved_library_path = None
    saved_library_note = None
    if cfg.save_to_library and final_source_type in ("python", "pinescript", "mql5") and final_code_text:
        ext = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}[final_source_type]
        base_name = safe_filename_stem(display_name, "full_pipeline_strategy")
        filename = f"{base_name}_pipeline{ext}"
        try:
            try:
                saved_library_path = save_strategy_text(final_code_text, filename, final_source_type, overwrite=False)
            except StrategyAlreadyExists:
                filename = f"{base_name}_pipeline_{int(time.time())}{ext}"
                saved_library_path = save_strategy_text(final_code_text, filename, final_source_type, overwrite=False)
            set_strategy_status(final_source_type, filename, cfg.library_status or _VERDICT_TO_LIBRARY_STATUS.get(verdict, "tested_passed"))
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
            })
            saved_library_note = f"Saved to the Strategy Library as '{filename}' (status: {cfg.library_status})."
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
        base_name = safe_filename_stem(display_name, "full_pipeline_strategy")
        filename = f"{base_name}_pipeline.json"
        config_text = json.dumps(final_config, indent=2)
        try:
            try:
                saved_library_path = save_strategy_text(config_text, filename, "manual", overwrite=False)
            except StrategyAlreadyExists:
                filename = f"{base_name}_pipeline_{int(time.time())}.json"
                saved_library_path = save_strategy_text(config_text, filename, "manual", overwrite=False)
            set_strategy_status("manual", filename, cfg.library_status or _VERDICT_TO_LIBRARY_STATUS.get(verdict, "tested_passed"))
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
            })
            saved_library_note = f"Saved to the Strategy Library as '{filename}' (status: {cfg.library_status})."
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
        oos_validation=oos_validation,
        oos_validation_skip_reason=oos_skip_reason,
        icir_gate=icir_gate,
        icir_gate_skip_reason=icir_gate_skip_reason,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        scorecard=scorecard,
        risk_of_ruin_hard_fail=risk_of_ruin_hard_fail,
        parsimony=parsimony_result,
        cpcv_result=cpcv_result,
        cpcv_skip_reason=cpcv_skip_reason,
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
