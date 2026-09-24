"""
Quick Optimize -- "select a strategy, click Optimize."

Full Pipeline (app.orchestration.full_pipeline) is the right tool when you
want the whole 7-step validated pipeline (baseline -> GA -> re-validation ->
OOS fold check -> holdout check -> significance gate -> report) for one
strategy at a time. That is deliberately thorough, and deliberately slow.

Sometimes what you actually want is much narrower: "take this strategy
that's already in my library, automatically try a bunch of different
parameter values against it, and hand me back the version that does best
against my eval-pass / payout / win-rate targets" -- without a full
validation report, and without leaving the Strategy Library screen.

This module is exactly that: it runs Full Pipeline's Step 2 (the same
walk-forward-aware GA -- identical code, so results carry the same
overfitting protection) in isolation, re-backtests the winning
configuration once to get real before/after numbers, and -- unless told
not to -- saves the optimized version into the Strategy Library as a new
file (never overwriting the original) so both versions stay available for
comparison and for staging into a batch/Full Pipeline run later.

This is intentionally a subset of Full Pipeline, not a replacement for it.
A strategy that comes out of Quick Optimize looking great has only been
checked against the same chained out-of-sample folds the GA already
optimizes for -- it has NOT been through the OOS holdout check or the
ICIR/Bonferroni significance gate that Full Pipeline runs afterward
specifically to catch a GA that got lucky. Treat a strong Quick Optimize
result as "worth a real Full Pipeline run," not as a finished answer.

A pip_size/instrument-scale mismatch (see app.backtest.risk's
has_instrument_scale_mismatch) is escalated the same way Full Pipeline
escalates it -- into a dedicated, hard-to-miss
QuickOptimizeResult.instrument_mismatch_warning field and a "!!!"-
prefixed log line -- rather than sitting quietly inside `warnings`
alongside routine notices. Unlike Full Pipeline, this tool does NOT skip
its GA search when it detects this (Quick Optimize's whole point is a
fast, narrower check), so a mismatch here means every number this run
reports is unreliable, not just slower to produce.

2026-09-17 Quick-Optimize-vs-Full-Pipeline follow-up (the 5-point fix):
a Quick Optimize result that LOOKED finished (a clean eval-pass
percentage, no visible caveats) was exactly what made the earlier
discrepancy so confusing -- someone reading only this tool's output had
no way to tell it apart from a validated one. This module now closes
that gap several ways at once:

  1. QuickOptimizeResult.validated is now hard-coded False and
     .result_banner/.result_banner_detail carry an explicit, impossible-
     to-miss "NOT OOS VALIDATED" warning into every caller (web job
     status, desktop summary line) -- this tool can no longer be
     mistaken for a finished answer just because it didn't say
     otherwise.
  2. A percentage built on too few out-of-sample trades is next to
     meaningless (a 99% eval-pass over 12 trades is noise, not edge).
     MIN_CREDIBLE_OOS_TRADES below gates .min_trade_count_met and puts
     an explicit "UNRELIABLE -- only N trades" qualifier on the
     displayed number rather than a bare percentage.
  3. The saved file's name/filename no longer silently drifts from its
     own parameters -- see app.strategy.library.provenance_stamped_name,
     used here whenever the GA actually mutated the config.
  4. The SAME ICIR/signal-decay/Bonferroni significance gate and
     parsimony score Full Pipeline's Step 6 runs are now surfaced here
     too (best-effort, same as Full Pipeline treats them), so Quick
     Optimize is never the tool that "looked safe" purely because it
     never checked.
  5. QuickOptimizeConfig.reserve_holdout (default False, opt-in) carves
     off the same trailing holdout_frac of `df` Full Pipeline reserves,
     runs the baseline/GA/final steps only against the remaining dev
     slice, and additionally reports the final configuration's
     performance on the untouched holdout tail -- so a caller who wants
     a less-than-100%-circular number from this fast tool can get one,
     without waiting for a full Full Pipeline run. This is still not a
     replacement for Full Pipeline's holdout check (no ICIR/Bonferroni
     re-check of the holdout segment itself, no OOS-fold walk-forward
     re-verification) -- it's a lighter, optional sanity check, and the
     "NOT OOS VALIDATED" banner above still applies even when it's on.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from app.backtest.adaptive_risk import build_limit_aware_preset
from app.backtest.engine import run_backtest
from app.backtest.statistics import reset_chain_note
from app.backtest.risk import (
    RiskConfig,
    account_size_mismatch_message,
    has_impossible_condition,
    has_instrument_scale_mismatch,
    instrument_scale_mismatch_message,
    with_prop_safety_defaults,
)
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, default_method_for_adaptive_risk, run_monte_carlo
from app.optimize.code_parameter_space import patched_source_for_strategy
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig, preflight_signal_check
from app.optimize.walkforward_ga import WalkforwardGAResult, run_walkforward_aware_refinement
from app.prop.simulator import PropRules, simulate_account
from app.scoring.parsimony import ParsimonyResult, compute_parsimony
from app.search.strategy_space import build_strategy_from_spec
from app.strategy.base import Strategy
from app.strategy.library import (
    StrategyAlreadyExists,
    provenance_stamped_name,
    record_optimize_result,
    safe_filename_stem,
    save_strategy_text,
    set_strategy_status,
)
from app.validation.icir import ICIRGateResult, run_icir_gate_from_backtest

ProgressCallback = "Callable[[str], None]"

_EXT_FOR_SOURCE = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}

# Point (2) of the 2026-09-17 fix: below this many out-of-sample trades, an
# eval-pass/payout percentage is treated as statistically unreliable rather
# than a trustworthy headline number -- a handful of resampled trades can
# easily land at 90%+ by chance. 100 is the same rough floor Full Pipeline's
# own ICIR gate effectively requires to say anything meaningful about
# significance (see app.validation.icir's period-based IC calculation);
# picking the same number here means "Quick Optimize says this is credible"
# and "Full Pipeline's significance gate has a chance of agreeing" point in
# the same direction, instead of Quick Optimize using a looser bar.
MIN_CREDIBLE_OOS_TRADES = 100

# Point (1): the fixed, always-shown banner text making clear that NO
# Quick Optimize result -- regardless of how strong its numbers look -- has
# been checked against genuinely unseen data by an independent gate the way
# a Full Pipeline run has. Kept as module-level constants (not built fresh
# per result) so every caller (web, desktop) renders byte-identical text.
RESEARCH_RESULT_BANNER = "\u26a0\ufe0f RESEARCH RESULT \u2014 NOT OOS VALIDATED"
RESEARCH_RESULT_BANNER_DETAIL = (
    "Run Full Pipeline to determine whether this result survives unseen-data validation."
)


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


@dataclass
class QuickOptimizeConfig:
    ga_population: int = 16
    ga_generations: int = 8
    fitness_metric: str = "eval_pass_probability"
    # UPGRADE (optimizer core): explicit optimizer mode -- "genetic"
    # (default, byte-identical to every run before this field existed),
    # or the optional TPE/CMA-ES samplers (see
    # app.optimize.refinement.OPTIMIZER_MODES). Threaded straight into the
    # RefinementConfig passed to run_walkforward_aware_refinement below --
    # every downstream step (cost-stress penalty, the chained-OOS fitness,
    # the final re-validation) runs identically regardless of which mode
    # is picked.
    optimizer_mode: str = "genetic"
    ga_search_mc_sims: int = 200
    final_mc_sims: int = 1000
    n_folds: int = 4
    window_mode: str = "rolling"          # "rolling" or "anchored"
    # Fixed default (was None, i.e. a fresh random search every run) so a
    # Quick Optimize run against the same strategy/data/config is
    # reproducible run-to-run -- matching FullPipelineConfig.random_seed's
    # own default of 42. Pass an explicit seed (or None) to override.
    random_seed: int | None = 42
    parallel: bool = True
    parallel_max_workers: int | None = None
    save_to_library: bool = True          # code strategies only -- manual configs aren't files
    # Same limit-aware risk-throttle preset as Full Pipeline (see
    # app.backtest.adaptive_risk.build_limit_aware_preset) -- off by
    # default; applies to the baseline run, the GA search itself, and
    # the final re-validated run alike.
    adaptive_risk_enabled: bool = False
    adaptive_risk_daily_profit_lock_pct: float | None = 80.0
    # Must be one of app.strategy.library's STATUS_LABELS_ORDERED. "draft" is
    # deliberately conservative -- Quick Optimize only re-runs the GA + one
    # backtest, not Full Pipeline's OOS holdout check or significance gate,
    # so the result hasn't earned "tested_passed"/"validated" yet.
    library_status: str = "draft"

    # UPGRADE (prop-firm reset-on-breach as the search basis): threaded into
    # the baseline/GA-search/final Monte Carlo passes and single-run prop
    # summaries below, so a blown account is scored the way a real prop
    # trader would actually handle it -- reset and keep going -- instead of
    # as a dead end. False (default) is byte-identical to every run before
    # this field existed; the web/desktop Quick Optimize form defaults its
    # own checkbox to CHECKED.
    reset_on_breach: bool = False

    # Point (5) of the 2026-09-17 fix: opt-in, lightweight holdout check.
    # False (default) is byte-identical to every run before this existed --
    # baseline/GA search/final re-validation all see the whole `df`, exactly
    # as before. True carves off the LAST holdout_frac of `df` (same
    # chronological split Full Pipeline uses -- see
    # FullPipelineConfig.reserve_true_holdout) and restricts the baseline,
    # the GA's own chained-OOS-fold search, and the final re-validation
    # backtest/Monte Carlo to the remaining dev slice only; the winning
    # configuration is then ALSO backtested (and Monte-Carlo'd) once against
    # the untouched holdout slice, and those numbers are reported alongside
    # -- see QuickOptimizeResult.holdout_trades and friends. This is
    # deliberately NOT the same as Full Pipeline's own holdout check: there
    # is no OOS-fold walk-forward re-verification and no ICIR/Bonferroni
    # re-check of the holdout segment specifically (see run_significance_
    # diagnostics below for what IS run, against the dev/search data). It
    # exists so a caller who wants a less-than-100%-circular number out of
    # this fast tool can opt into one without waiting for a full Full
    # Pipeline run -- not so this tool can claim to replace that run.
    reserve_holdout: bool = False
    holdout_frac: float = 0.2


@dataclass
class QuickOptimizeResult:
    strategy_display_name: str
    source_type: str

    baseline_trades: int
    baseline_net_profit: float
    baseline_win_rate: float
    baseline_eval_pass_probability: float
    baseline_payout_probability: float

    optimized_trades: int
    optimized_net_profit: float
    optimized_win_rate: float
    optimized_eval_pass_probability: float
    optimized_payout_probability: float

    ga_result: WalkforwardGAResult | None
    improved: bool                       # optimized beats baseline on eval-pass, then win-rate
    final_parameters: dict[str, str] | None
    final_code_text: str | None
    final_code_extension: str | None
    saved_library_path: Path | None
    saved_library_note: str | None
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)
    # Escalated copy of app.backtest.risk.instrument_scale_mismatch_message
    # when either the baseline or the optimized run flagged a pip_size/
    # instrument-scale mismatch -- None otherwise. Unlike Full Pipeline
    # (which skips its GA search on this), Quick Optimize still runs the
    # search either way -- this field exists purely so the UI can give
    # this warning the same visual weight Full Pipeline does, instead of
    # it sitting quietly inside `warnings` at the same level as everything
    # else (see the Quick-Optimize-vs-Full-Pipeline diagnosis this fixes).
    instrument_mismatch_warning: str | None = None
    # Escalated copy of any app.strategy.manual.validate_bounded_conditions
    # warning (a condition comparing a bounded oscillator to a threshold
    # outside its possible range, e.g. "RSI > 102.56") found on either the
    # baseline or optimized run -- same treatment as
    # instrument_mismatch_warning above, and for the same reason: this
    # silently disables part of the strategy's logic while every other
    # number keeps reporting normally, so it needs the same visual
    # weight, not a line buried in `warnings`.
    invalid_condition_warning: str | None = None
    # RISK-001: set when the caller's RiskConfig.initial_balance didn't
    # match PropRules.account_size before this run -- see
    # app.backtest.risk.account_size_mismatch_message. The run still
    # proceeds (with_prop_safety_defaults makes account_size win), but
    # this needs the same "!!!"-prefixed prominence as the other two
    # warnings above since every number below was computed against the
    # corrected balance, not whatever the caller originally passed in.
    account_mismatch_warning: str | None = None

    # -- 2026-09-17 Quick-Optimize-vs-Full-Pipeline follow-up fields ------

    # Point (1): ALWAYS False. Quick Optimize never runs an independent
    # out-of-sample gate on the final configuration the way Full Pipeline's
    # Step 5 holdout check + Step 6 ICIR/Bonferroni gate do -- even with
    # reserve_holdout=True (point 5), the holdout check here is a single
    # plain backtest, not re-verified against its own significance test.
    # This field exists so a caller can branch on it (`if result.validated`)
    # instead of re-deriving "is this actually validated" from several
    # other fields every time; it is intentionally not a settable/computed
    # property, so nothing here can ever accidentally start reporting True.
    validated: bool = False
    # Fixed banner text (see RESEARCH_RESULT_BANNER / _DETAIL above) --
    # every UI surface (web job status, desktop summary) renders these
    # instead of a bare percentage so a strong-looking result can't be
    # mistaken for a finished, validated one.
    result_banner: str = RESEARCH_RESULT_BANNER
    result_banner_detail: str = RESEARCH_RESULT_BANNER_DETAIL

    # Point (2): the number of out-of-sample trades the winning
    # configuration's headline optimized_eval_pass_probability is actually
    # built on (ga_result.best.oos_trade_count when the GA found a winner,
    # else len(final_bt.trades)), and whether that clears
    # MIN_CREDIBLE_OOS_TRADES. A caller/template should show optimized_
    # eval_pass_probability as a plain percentage only when this is True --
    # otherwise pair it with an explicit "too few trades to be reliable"
    # qualifier (see trade_count_warning).
    oos_trade_count: int = 0
    min_trade_count_met: bool = False
    trade_count_warning: str | None = None

    # Point (4): the same ICIR/signal-decay/Bonferroni significance gate
    # and parsimony score Full Pipeline's Step 6 runs, applied here against
    # the same data the GA searched over (df, or dev_df when
    # reserve_holdout=True) with n_tests set from cfg.ga_population *
    # (cfg.ga_generations + 1) -- identical formula to Full Pipeline's own
    # n_candidates_tested, so the two tools' significance bars are directly
    # comparable. None when the strategy produced too few trades to run the
    # gate at all (see icir_gate_skip_reason) -- best-effort, like Full
    # Pipeline's own copy of this step, never allowed to fail the run.
    icir_gate: ICIRGateResult | None = None
    icir_gate_skip_reason: str | None = None
    significance_note: str | None = None
    parsimony_result: ParsimonyResult | None = None
    parsimony_note: str | None = None

    # Point (5): populated only when cfg.reserve_holdout=True. The winning
    # configuration's own performance on the trailing holdout_frac slice
    # that the baseline/GA search/final re-validation above never saw.
    # None (not zero) in every field when reserve_holdout=False, so a
    # caller can tell "holdout wasn't requested" apart from "holdout ran
    # and found zero trades."
    holdout_enabled: bool = False
    holdout_trades: int | None = None
    holdout_net_profit: float | None = None
    holdout_win_rate: float | None = None
    holdout_eval_pass_probability: float | None = None
    holdout_payout_probability: float | None = None
    holdout_note: str | None = None


def run_quick_optimize(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    cfg: QuickOptimizeConfig | None = None,
    progress_cb=None,
    cancel_event=None,
) -> QuickOptimizeResult:
    """Runs the walk-forward-aware GA against `strategy` and returns a
    before/after comparison. Raises RefinementError (same exception Full
    Pipeline surfaces for this step) if the strategy has no tunable
    parameters, or produces zero baseline trades, so the caller can show a
    clear message instead of a stack trace.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    cfg = cfg or QuickOptimizeConfig()
    t0 = time.time()
    warnings: list[str] = []
    display_name = _display_name(strategy)

    log(f"  {RESEARCH_RESULT_BANNER}")
    log(f"  {RESEARCH_RESULT_BANNER_DETAIL}")

    # Point (5) of the 2026-09-17 fix: opt-in holdout carve-out, same
    # chronological split Full Pipeline's FullPipelineConfig.reserve_true_
    # holdout uses. `dev_df` is what the baseline, the GA's own chained-OOS
    # search, and the final re-validation below all see; `holdout_df` is
    # held back and only used once, at the very end, purely to report the
    # winning configuration's performance on data none of the above ever
    # touched. When reserve_holdout is off (default), dev_df IS df and
    # holdout_df is empty -- byte-identical to this function's behavior
    # before this option existed.
    if cfg.reserve_holdout:
        _n = len(df)
        _split_idx = int(_n * (1 - cfg.holdout_frac))
        _split_idx = max(1, min(_split_idx, _n - 1)) if _n > 1 else _n
        dev_df = df.iloc[:_split_idx].reset_index(drop=True)
        holdout_df = df.iloc[_split_idx:].reset_index(drop=True)
        log(
            f"  Reserving the final {cfg.holdout_frac:.0%} of the dataset ({len(df) - len(dev_df):,} of "
            f"{len(df):,} bars) as a holdout -- the baseline, search, and final re-validation below only "
            f"see the first {len(dev_df):,} bars; the holdout numbers reported at the end are the first "
            f"time the reserved bars are used at all."
        )
    else:
        dev_df = df
        holdout_df = df.iloc[0:0]

    # RISK-001: detect (and log) a RiskConfig.initial_balance / PropRules.
    # account_size mismatch BEFORE it gets silently reconciled below --
    # same escalation prominence as the instrument-mismatch/impossible-
    # condition warnings that already exist on this result.
    account_mismatch_warning = account_size_mismatch_message(risk.initial_balance, prop_rules.account_size)
    if account_mismatch_warning is not None:
        log(f"  !!! {account_mismatch_warning}")
        warnings.append(account_mismatch_warning)
    risk = with_prop_safety_defaults(risk, prop_rules)
    # FIX (2026-09-18): see RiskConfig.reset_on_breach's docstring -- cfg.
    # reset_on_breach was already threaded into simulate_account/
    # MonteCarloConfig below but never into the RiskConfig the raw
    # baseline/GA-search/final/holdout backtests actually run against.
    risk = replace(risk, reset_on_breach=cfg.reset_on_breach)

    log(f"Checking '{display_name}' produces trades on this data...")
    preflight_signal_check(dev_df, strategy, risk, "Quick Optimize")

    adaptive_risk = build_limit_aware_preset(prop_rules, daily_profit_lock_pct=cfg.adaptive_risk_daily_profit_lock_pct) \
        if cfg.adaptive_risk_enabled else None
    if adaptive_risk is not None:
        log(f"Adaptive risk enabled: {len(adaptive_risk.rules)} limit-aware throttle rule(s) applied.")

    log("Running baseline backtest...")
    baseline_bt = run_backtest(dev_df, strategy, risk, adaptive_risk=adaptive_risk)
    warnings.extend(baseline_bt.warnings)

    # Escalate a pip_size/instrument-scale mismatch to the SAME prominence
    # Full Pipeline gives it, rather than letting it sit quietly inside
    # `warnings` -- Full Pipeline detects this and skips its GA search
    # entirely (see app.orchestration.full_pipeline); Quick Optimize still
    # runs the search either way (this tool's whole point is a fast,
    # narrower check), but the person running it needs to see this just as
    # clearly, since every number below is unreliable while it's present.
    instrument_mismatch_warning = None
    if has_instrument_scale_mismatch(baseline_bt.warnings):
        instrument_mismatch_warning = instrument_scale_mismatch_message(risk.pip_size)
        log(f"  !!! {instrument_mismatch_warning}")
        log(
            "  !!! Continuing anyway (Quick Optimize does not skip its search on this, unlike "
            "Full Pipeline) -- but treat every number below as unreliable until pip_size is fixed."
        )

    invalid_condition_warning = None
    if has_impossible_condition(baseline_bt.warnings):
        invalid_condition_warning = next(w for w in baseline_bt.warnings if "can never be true" in w)
        log(f"  !!! {invalid_condition_warning}")

    baseline_pnls = [t.pnl for t in baseline_bt.trades]
    baseline_dates = [t.entry_time for t in baseline_bt.trades]
    simulate_account(baseline_pnls, baseline_dates, prop_rules, reset_on_breach=cfg.reset_on_breach)  # surfaces any account-sim issues early
    baseline_mc = run_monte_carlo(
        baseline_bt.trades, prop_rules,
        MonteCarloConfig(method=default_method_for_adaptive_risk(adaptive_risk), n_simulations=cfg.final_mc_sims, random_seed=cfg.random_seed, reset_on_breach=cfg.reset_on_breach),
    )
    log(
        f"Baseline: {len(baseline_bt.trades)} trades, net ${baseline_bt.statistics.net_profit:,.2f}, "
        f"win rate {baseline_bt.statistics.win_rate:.1f}%, eval pass {baseline_mc.evaluation_pass_probability:.1f}%, "
        f"payout {baseline_mc.first_payout_probability:.1f}%."
        + reset_chain_note(
            baseline_bt.statistics, baseline_mc.evaluation_pass_probability, baseline_mc.first_payout_probability,
            baseline_mc.per_attempt_pass_probability, baseline_mc.per_attempt_payout_probability,
            baseline_mc.total_independent_attempts,
        )
    )

    log(f"Searching for a more robust configuration ({cfg.ga_generations} generations x {cfg.ga_population} candidates)...")
    refine_cfg = RefinementConfig(
        population_size=cfg.ga_population,
        generations=cfg.ga_generations,
        fitness_metric=cfg.fitness_metric,
        search_monte_carlo_sims=cfg.ga_search_mc_sims,
        random_seed=cfg.random_seed,
        optimizer_mode=cfg.optimizer_mode,
    )
    ga_result = run_walkforward_aware_refinement(
        dev_df, strategy, risk, prop_rules,
        MonteCarloConfig(method=default_method_for_adaptive_risk(adaptive_risk), n_simulations=cfg.ga_search_mc_sims, random_seed=cfg.random_seed, reset_on_breach=cfg.reset_on_breach),
        refinement_config=refine_cfg,
        n_folds=cfg.n_folds, window_mode=cfg.window_mode,
        progress_cb=lambda m: log(f"  {m}"),
        parallel=cfg.parallel, max_workers=cfg.parallel_max_workers,
        adaptive_risk=adaptive_risk,
        cancel_event=cancel_event,
    )
    warnings.extend(ga_result.warnings)

    # Point (3) of the 2026-09-17 fix: only a real GA winner (this `else`
    # branch) has actually mutated the config away from its starting
    # parameters -- the `if` branch below returns the ORIGINAL, unchanged
    # strategy, so its name is still accurate and must NOT be provenance-
    # stamped. mutated_by_ga is threaded through to the library-save block
    # further down so it stamps (and, for manual configs, rewrites the
    # saved JSON's own "name" field) only when the parameters on disk would
    # otherwise no longer match the name they're filed under.
    mutated_by_ga = ga_result.best.oos_trade_count > 0
    if not mutated_by_ga:
        warnings.append(
            "The GA's best candidate still produced zero out-of-sample trades -- returning the "
            "original configuration unchanged rather than a 'winner' that never actually traded."
        )
        final_source_type = strategy.source_type
        final_config = strategy.config if strategy.source_type == "manual" else None
        final_code_text, final_code_ext = (None, None)
        if strategy.source_type != "manual":
            final_code_text, final_code_ext = patched_source_for_strategy(strategy, [], [])
        final_parameters = None
    else:
        log(
            f"Winning configuration: chained-OOS fitness {ga_result.best.fitness:.3f} "
            f"({ga_result.best.oos_trade_count} OOS trades across {ga_result.n_folds} fold(s))."
        )
        if ga_result.overfitting_gap is not None and ga_result.overfitting_gap > 0:
            warnings.append(
                f"Overfitting gap (in-sample fitness minus chained-OOS fitness): "
                f"{ga_result.overfitting_gap:.3f}. Large positive = looks better in-sample than out-of-sample."
            )
        final_source_type = strategy.source_type
        final_config = ga_result.best.config
        final_code_text = ga_result.best.code_text
        final_code_ext = ga_result.best.code_extension
        final_parameters = {
            gene.label: (str(int(round(value))) if gene.is_int else f"{value:.4f}".rstrip("0").rstrip("."))
            for gene, value in zip(ga_result.genes, ga_result.best.genome)
        } if ga_result.genes else None

    log("Re-running the full backtest + Monte Carlo on the winning configuration...")
    if final_source_type == "manual":
        final_spec = {"source_type": "manual", "config": final_config}
    else:
        final_spec = {"source_type": final_source_type, "code_text": final_code_text, "code_extension": final_code_ext}
    final_strategy = build_strategy_from_spec(final_spec)
    final_bt = run_backtest(dev_df, final_strategy, risk, adaptive_risk=adaptive_risk)
    warnings.extend(final_bt.warnings)
    if instrument_mismatch_warning is None and has_instrument_scale_mismatch(final_bt.warnings):
        # Same mismatch, just not visible until the optimized parameters
        # were actually run -- risk.pip_size is unchanged from the
        # baseline check above, so this is the identical underlying
        # problem, not a new one the GA introduced.
        instrument_mismatch_warning = instrument_scale_mismatch_message(risk.pip_size)
        log(f"  !!! {instrument_mismatch_warning}")
    if invalid_condition_warning is None and has_impossible_condition(final_bt.warnings):
        invalid_condition_warning = next(w for w in final_bt.warnings if "can never be true" in w)
        log(f"  !!! {invalid_condition_warning}")
    final_mc = run_monte_carlo(
        final_bt.trades, prop_rules,
        MonteCarloConfig(method=default_method_for_adaptive_risk(adaptive_risk), n_simulations=cfg.final_mc_sims, random_seed=cfg.random_seed, reset_on_breach=cfg.reset_on_breach),
        # MC-004: these trades came straight out of this run's own GA
        # search over this same data -- see run_monte_carlo's docstring.
        selection_bias_caveat=(ga_result.best.oos_trade_count > 0),
    )
    log(
        f"Optimized: {len(final_bt.trades)} trades, net ${final_bt.statistics.net_profit:,.2f}, "
        f"win rate {final_bt.statistics.win_rate:.1f}%, eval pass {final_mc.evaluation_pass_probability:.1f}%, "
        f"payout {final_mc.first_payout_probability:.1f}%."
        + reset_chain_note(
            final_bt.statistics, final_mc.evaluation_pass_probability, final_mc.first_payout_probability,
            final_mc.per_attempt_pass_probability, final_mc.per_attempt_payout_probability,
            final_mc.total_independent_attempts,
        )
    )

    # Point (2) of the 2026-09-17 fix: the headline eval-pass/payout
    # percentage above is only as credible as the number of out-of-sample
    # trades behind it. Use the GA's own OOS trade count when it found a
    # real winner (the number the chained-OOS fitness above was actually
    # scored on); fall back to the final backtest's own trade count for the
    # zero-OOS-trades branch, where oos_trade_count is 0 by definition but
    # final_bt.trades is the unchanged original strategy's own trades.
    oos_trade_count = ga_result.best.oos_trade_count if mutated_by_ga else len(final_bt.trades)
    min_trade_count_met = oos_trade_count >= MIN_CREDIBLE_OOS_TRADES
    trade_count_warning = None
    if not min_trade_count_met:
        trade_count_warning = (
            f"UNRELIABLE -- only {oos_trade_count} out-of-sample trade(s) behind this result "
            f"(want {MIN_CREDIBLE_OOS_TRADES}+). A percentage built on this few trades can easily "
            f"land wherever it landed by chance; treat it as noise, not edge, until more trades "
            f"are available."
        )
        log(f"  !!! {trade_count_warning}")

    # Point (4): the same ICIR/signal-decay/Bonferroni significance gate
    # and parsimony score Full Pipeline's Step 6 runs -- best-effort, never
    # allowed to fail this run. Run against dev_df (the same data the GA
    # searched over, exactly like Full Pipeline runs its own copy of this
    # step against `df`/dev_df rather than the held-out tail), using the
    # identical n_tests formula Full Pipeline uses so the two tools'
    # Bonferroni corrections are directly comparable.
    log("Checking signal significance (ICIR / signal-decay / Bonferroni) and parsimony...")
    icir_gate = None
    icir_gate_skip_reason = None
    significance_note = None
    try:
        # FIX (Bonferroni-vs-actual-evaluations): use the GA's own
        # total_evaluations (how many genomes it actually backtested)
        # rather than the CONFIGURED population*(generations+1) budget --
        # see app.orchestration.full_pipeline's identical fix for why
        # these two can now differ (auto-shrink-on-low-trades, AI-assist).
        n_candidates_tested = max(1, ga_result.total_evaluations) if mutated_by_ga else 1
        icir_gate = run_icir_gate_from_backtest(
            dev_df, final_strategy, risk, n_tests=n_candidates_tested, holdout_frac=cfg.holdout_frac,
        )
        significance_note = (
            f"{'PASSED' if icir_gate.ok else 'DID NOT PASS'} "
            f"(Bonferroni-corrected for {n_candidates_tested} candidate(s) tried)."
        )
        log(f"  {significance_note}")
        for reason in icir_gate.reasons:
            log(f"    {reason}")
    except Exception as exc:  # noqa: BLE001 -- best-effort validation step, same treatment as Full Pipeline's copy
        icir_gate_skip_reason = f"ICIR gate failed to run: {exc}"
        log(f"  {icir_gate_skip_reason}")

    parsimony_result = compute_parsimony(final_strategy)
    parsimony_note = parsimony_result.notes[0] if parsimony_result.notes else None
    if parsimony_note:
        log(f"  Parsimony: {parsimony_note}")

    # Point (5): the winning configuration's performance on the untouched
    # holdout tail, when reserve_holdout=True. A single plain backtest +
    # Monte Carlo -- not re-run through the ICIR gate above, and not a
    # substitute for Full Pipeline's own Step 4/5/6 -- see this function's
    # docstring and QuickOptimizeConfig.reserve_holdout for exactly what
    # this does and does not check.
    holdout_trades = holdout_net_profit = holdout_win_rate = None
    holdout_eval_pass_probability = holdout_payout_probability = None
    holdout_note = None
    if cfg.reserve_holdout:
        if len(holdout_df) == 0:
            holdout_note = "Holdout slice was empty -- nothing to check."
            log(f"  {holdout_note}")
        else:
            log(f"Backtesting the winning configuration on the reserved holdout ({len(holdout_df):,} bars, never seen above)...")
            holdout_bt = run_backtest(holdout_df, final_strategy, risk, adaptive_risk=adaptive_risk)
            warnings.extend(holdout_bt.warnings)
            holdout_trades = len(holdout_bt.trades)
            holdout_net_profit = holdout_bt.statistics.net_profit
            holdout_win_rate = holdout_bt.statistics.win_rate
            if holdout_trades > 0:
                holdout_mc = run_monte_carlo(
                    holdout_bt.trades, prop_rules,
                    MonteCarloConfig(method=default_method_for_adaptive_risk(adaptive_risk), n_simulations=cfg.final_mc_sims, random_seed=cfg.random_seed, reset_on_breach=cfg.reset_on_breach),
                    selection_bias_caveat=False,  # this slice was never used to select/score the winner
                )
                holdout_eval_pass_probability = holdout_mc.evaluation_pass_probability
                holdout_payout_probability = holdout_mc.first_payout_probability
                holdout_note = (
                    f"Holdout: {holdout_trades} trades, net ${holdout_net_profit:,.2f}, "
                    f"win rate {holdout_win_rate:.1f}%, eval pass {holdout_eval_pass_probability:.1f}%, "
                    f"payout {holdout_payout_probability:.1f}%."
                    + reset_chain_note(
                        holdout_bt.statistics, holdout_eval_pass_probability, holdout_payout_probability,
                        holdout_mc.per_attempt_pass_probability, holdout_mc.per_attempt_payout_probability,
                        holdout_mc.total_independent_attempts,
                    )
                )
            else:
                holdout_note = "Holdout: 0 trades -- the winning configuration never traded on the reserved slice."
            log(f"  {holdout_note}")

    improved = (
        final_mc.evaluation_pass_probability > baseline_mc.evaluation_pass_probability
        or (
            final_mc.evaluation_pass_probability == baseline_mc.evaluation_pass_probability
            and final_bt.statistics.win_rate > baseline_bt.statistics.win_rate
        )
    )

    # Point (3) of the 2026-09-17 fix: a real GA winner's parameters no
    # longer match `display_name` (the ORIGINAL strategy's name) -- see
    # app.strategy.library.provenance_stamped_name's own docstring for the
    # exact bug this closes. save_name is what both the filename AND (for
    # manual configs, below) the saved JSON's own "name" field are derived
    # from; the unmutated branch keeps display_name exactly as before.
    save_name = (
        provenance_stamped_name(display_name, origin="quick_optimize", seed=cfg.random_seed)
        if mutated_by_ga else display_name
    )

    saved_library_path = None
    saved_library_note = None
    if cfg.save_to_library and final_source_type in ("python", "pinescript", "mql5") and final_code_text:
        ext = _EXT_FOR_SOURCE[final_source_type]
        base_name = safe_filename_stem(save_name, "optimized_strategy")
        filename = f"{base_name}_optimized{ext}"
        try:
            try:
                saved_library_path = save_strategy_text(final_code_text, filename, final_source_type, overwrite=False)
            except StrategyAlreadyExists:
                filename = f"{base_name}_optimized_{int(time.time())}{ext}"
                saved_library_path = save_strategy_text(final_code_text, filename, final_source_type, overwrite=False)
            set_strategy_status(final_source_type, filename, cfg.library_status)
            record_optimize_result(final_source_type, filename, {
                "improved": improved,
                "trades": len(final_bt.trades),
                "net_profit": round(final_bt.statistics.net_profit, 2),
                "win_rate": round(final_bt.statistics.win_rate, 1),
                "max_dd": round(final_bt.statistics.max_drawdown_pct, 2),
                "eval_pass_probability": round(final_mc.evaluation_pass_probability, 1),
                "first_payout_probability": round(final_mc.first_payout_probability, 1),
                "baseline_eval_pass_probability": round(baseline_mc.evaluation_pass_probability, 1),
            })
            saved_library_note = f"Saved to the Strategy Library as '{filename}' (status: {cfg.library_status})."
            log(saved_library_note)
        except Exception as exc:  # noqa: BLE001 -- saving is a convenience, not the core result
            saved_library_note = f"Could not save to the Strategy Library: {exc}"
            log(saved_library_note)
    elif cfg.save_to_library and final_source_type == "manual" and final_config:
        # Same fix as app.orchestration.full_pipeline._finish -- manual/
        # Search-Lab configs are dicts, not files, but app.strategy.library
        # already has a first-class "manual" type that stores exactly this
        # shape as JSON, so there's no real reason this used to give up.
        base_name = safe_filename_stem(save_name, "optimized_strategy")
        filename = f"{base_name}_optimized.json"
        # Point (3), continued: overwrite the saved JSON's own "name" field
        # with save_name too when mutated -- otherwise the file on disk
        # would still claim the ORIGINAL (now-inaccurate) name internally
        # even though its filename was fixed, which is exactly the
        # "stale name, mutated body" state that caused the original bug.
        config_to_save = dict(final_config)
        if mutated_by_ga:
            config_to_save["name"] = save_name
        config_text = json.dumps(config_to_save, indent=2)
        try:
            try:
                saved_library_path = save_strategy_text(config_text, filename, "manual", overwrite=False)
            except StrategyAlreadyExists:
                filename = f"{base_name}_optimized_{int(time.time())}.json"
                saved_library_path = save_strategy_text(config_text, filename, "manual", overwrite=False)
            set_strategy_status("manual", filename, cfg.library_status)
            record_optimize_result("manual", filename, {
                "improved": improved,
                "trades": len(final_bt.trades),
                "net_profit": round(final_bt.statistics.net_profit, 2),
                "win_rate": round(final_bt.statistics.win_rate, 1),
                "max_dd": round(final_bt.statistics.max_drawdown_pct, 2),
                "eval_pass_probability": round(final_mc.evaluation_pass_probability, 1),
                "first_payout_probability": round(final_mc.first_payout_probability, 1),
                "baseline_eval_pass_probability": round(baseline_mc.evaluation_pass_probability, 1),
            })
            saved_library_note = f"Saved to the Strategy Library as '{filename}' (status: {cfg.library_status})."
            log(saved_library_note)
            final_code_text = config_text
            final_code_ext = ".json"
        except Exception as exc:  # noqa: BLE001 -- saving is a convenience, not the core result
            saved_library_note = f"Could not save to the Strategy Library: {exc}"
            log(saved_library_note)
    elif final_source_type == "manual":
        saved_library_note = (
            "Manual Strategy Builder configuration produced, but nothing was saved (saving to the "
            "Strategy Library is turned off) -- copy the winning parameters from this result into "
            "the Strategy tab."
        )

    elapsed = time.time() - t0
    log(f"Quick Optimize complete in {elapsed:.1f}s.")

    try:
        from app.ai.experiment_memory import record_experiment

        record_experiment(
            origin="quick_optimize",
            strategy_name=f"{display_name} (Quick Optimize)",
            source_type=final_source_type,
            verdict="IMPROVED" if improved else "NO IMPROVEMENT",
            trades=len(final_bt.trades),
            net_profit=final_bt.statistics.net_profit,
            win_rate=final_bt.statistics.win_rate,
            profit_factor=final_bt.statistics.profit_factor,
            max_drawdown_pct=final_bt.statistics.max_drawdown_pct,
            eval_pass_probability=final_mc.evaluation_pass_probability,
            first_payout_probability=final_mc.first_payout_probability,
            risk_of_ruin_pct=final_mc.risk_of_ruin_pct,
            lesson="; ".join(warnings[:3]) if warnings else "",
        )
    except Exception:
        pass  # T58 Research Memory is a bonus record -- never let it affect a completed Quick Optimize run

    return QuickOptimizeResult(
        strategy_display_name=display_name,
        source_type=strategy.source_type,
        baseline_trades=len(baseline_bt.trades),
        baseline_net_profit=baseline_bt.statistics.net_profit,
        baseline_win_rate=baseline_bt.statistics.win_rate,
        baseline_eval_pass_probability=baseline_mc.evaluation_pass_probability,
        baseline_payout_probability=baseline_mc.first_payout_probability,
        optimized_trades=len(final_bt.trades),
        optimized_net_profit=final_bt.statistics.net_profit,
        optimized_win_rate=final_bt.statistics.win_rate,
        optimized_eval_pass_probability=final_mc.evaluation_pass_probability,
        optimized_payout_probability=final_mc.first_payout_probability,
        ga_result=ga_result,
        improved=improved,
        final_parameters=final_parameters,
        final_code_text=final_code_text,
        final_code_extension=final_code_ext,
        saved_library_path=saved_library_path,
        saved_library_note=saved_library_note,
        elapsed_seconds=elapsed,
        warnings=warnings,
        instrument_mismatch_warning=instrument_mismatch_warning,
        invalid_condition_warning=invalid_condition_warning,
        account_mismatch_warning=account_mismatch_warning,
        validated=False,
        result_banner=RESEARCH_RESULT_BANNER,
        result_banner_detail=RESEARCH_RESULT_BANNER_DETAIL,
        oos_trade_count=oos_trade_count,
        min_trade_count_met=min_trade_count_met,
        trade_count_warning=trade_count_warning,
        icir_gate=icir_gate,
        icir_gate_skip_reason=icir_gate_skip_reason,
        significance_note=significance_note,
        parsimony_result=parsimony_result,
        parsimony_note=parsimony_note,
        holdout_enabled=cfg.reserve_holdout,
        holdout_trades=holdout_trades,
        holdout_net_profit=holdout_net_profit,
        holdout_win_rate=holdout_win_rate,
        holdout_eval_pass_probability=holdout_eval_pass_probability,
        holdout_payout_probability=holdout_payout_probability,
        holdout_note=holdout_note,
    )


# ---------------------------------------------------------------------------
# Timeframe sweep -- "optimize this strategy's parameters, AND tell me which
# of these timeframes it should even be trading on." Deliberately an outer
# loop around run_quick_optimize above, not a change to it: each requested
# timeframe gets its own complete, independent Quick Optimize run (same GA,
# same OOS-fold scoring, same holdout/ICIR machinery -- nothing about a
# single run's own rigor changes), and this just collects the N results and
# says which one won. See app.data.timeframe_sweep for exactly how a
# timeframe list gets turned into resampled DataFrames (and what a bad/
# too-fine/duplicate entry does -- skipped, never fatal to the others).
# ---------------------------------------------------------------------------

@dataclass
class QuickOptimizeSweepResult:
    per_timeframe: dict  # dict[str, QuickOptimizeResult], insertion order == timeframes tried
    best_timeframe: str
    best_result: QuickOptimizeResult
    skipped: list = field(default_factory=list)   # list[app.data.timeframe_sweep.SweepSkipped]
    errors: dict = field(default_factory=dict)     # dict[str, str] -- timeframe label -> why it failed


def run_quick_optimize_sweep(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    timeframes: list[str],
    cfg: QuickOptimizeConfig | None = None,
    progress_cb=None,
    cancel_event=None,
) -> QuickOptimizeSweepResult:
    """Runs run_quick_optimize once per timeframe in `timeframes`, on `df`
    resampled to each one, and returns every timeframe's result alongside
    which one won.

    An empty `timeframes` (or one where every entry gets skipped by the
    sweep -- e.g. everything requested was finer than the loaded data)
    falls back to a single run against `df` completely unresampled, under
    the label "native" -- so calling this with no timeframes is never
    worse than calling run_quick_optimize directly.

    "Won" is judged the same way a single run's own `.improved` flag is:
    higher optimized_eval_pass_probability first, optimized_win_rate as
    the tiebreaker (see QuickOptimizeResult.improved's own comment) --
    never net_profit alone, since eval-pass/payout probability is what
    this app treats as the real target everywhere else.

    For a manual (JSON) Strategy Builder strategy, the CANDIDATE timeframe
    being tried is stamped onto a COPY of its config's own top-level
    "timeframe" key before that timeframe's run -- so if it wins, the
    version saved to the Strategy Library already declares the timeframe
    it was found on (see app.data.timeframe_resample), and re-running it
    later anywhere else in the app automatically resamples to that same
    bar size rather than silently reverting to whatever's loaded. Python/
    PineScript/MQL5 strategies are swept the same way (their code runs
    against each resampled DataFrame in turn) but are NOT auto-stamped
    this way -- declare their timeframe the normal way (a module-level
    TIMEFRAME= constant, or a `// T58_TIMEFRAME=` directive) if you want
    the same guarantee for a non-manual winner.
    """
    from app.data.timeframe_sweep import build_timeframe_sweep

    plan = build_timeframe_sweep(df, timeframes or [])
    target_pairs = [(t.label, t.dataframe) for t in plan.targets] if plan.targets else [("native", df)]

    per_timeframe: dict[str, QuickOptimizeResult] = {}
    errors: dict[str, str] = {}

    for label, target_df in target_pairs:
        if cancel_event is not None and cancel_event.is_set():
            break
        if progress_cb:
            progress_cb(f"=== Timeframe {label}: starting Quick Optimize ({len(target_df):,} bars) ===")

        run_strategy = strategy
        if (
            plan.targets and getattr(strategy, "source_type", None) == "manual"
            and isinstance(getattr(strategy, "config", None), dict)
        ):
            from app.strategy.manual import ManualStrategy

            stamped_config = json.loads(json.dumps(strategy.config))  # cheap, dependency-free deep copy
            stamped_config["timeframe"] = label
            run_strategy = ManualStrategy(stamped_config)

        def _scoped_log(msg: str, _label=label) -> None:
            if progress_cb:
                progress_cb(f"[{_label}] {msg}")

        try:
            result = run_quick_optimize(
                target_df, run_strategy, risk, prop_rules, cfg=cfg,
                progress_cb=_scoped_log if progress_cb else None, cancel_event=cancel_event,
            )
            per_timeframe[label] = result
        except RefinementError as exc:
            errors[label] = str(exc)
            _scoped_log(f"skipped -- {exc}")

    if not per_timeframe:
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items()) if errors else "no timeframe could be resampled from this dataset."
        raise RefinementError(f"Quick Optimize's timeframe sweep produced no usable timeframe -- {detail}")

    best_label = max(
        per_timeframe,
        key=lambda k: (per_timeframe[k].optimized_eval_pass_probability, per_timeframe[k].optimized_win_rate),
    )

    return QuickOptimizeSweepResult(
        per_timeframe=per_timeframe, best_timeframe=best_label, best_result=per_timeframe[best_label],
        skipped=plan.skipped, errors=errors,
    )
