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
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.backtest.adaptive_risk import build_limit_aware_preset
from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig, has_instrument_scale_mismatch, instrument_scale_mismatch_message
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from app.optimize.code_parameter_space import patched_source_for_strategy
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig, preflight_signal_check
from app.optimize.walkforward_ga import WalkforwardGAResult, run_walkforward_aware_refinement
from app.prop.simulator import PropRules, simulate_account
from app.search.strategy_space import build_strategy_from_spec
from app.strategy.base import Strategy
from app.strategy.library import StrategyAlreadyExists, save_strategy_text, set_strategy_status

ProgressCallback = "Callable[[str], None]"

_EXT_FOR_SOURCE = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}


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

    log(f"Checking '{display_name}' produces trades on this data...")
    preflight_signal_check(df, strategy, risk, "Quick Optimize")

    adaptive_risk = build_limit_aware_preset(prop_rules, daily_profit_lock_pct=cfg.adaptive_risk_daily_profit_lock_pct) \
        if cfg.adaptive_risk_enabled else None
    if adaptive_risk is not None:
        log(f"Adaptive risk enabled: {len(adaptive_risk.rules)} limit-aware throttle rule(s) applied.")

    log("Running baseline backtest...")
    baseline_bt = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)
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

    baseline_pnls = [t.pnl for t in baseline_bt.trades]
    baseline_dates = [t.entry_time for t in baseline_bt.trades]
    simulate_account(baseline_pnls, baseline_dates, prop_rules)  # surfaces any account-sim issues early
    baseline_mc = run_monte_carlo(
        baseline_bt.trades, prop_rules, MonteCarloConfig(n_simulations=cfg.final_mc_sims, random_seed=cfg.random_seed)
    )
    log(
        f"Baseline: {len(baseline_bt.trades)} trades, net ${baseline_bt.statistics.net_profit:,.2f}, "
        f"win rate {baseline_bt.statistics.win_rate:.1f}%, eval pass {baseline_mc.evaluation_pass_probability:.1f}%, "
        f"payout {baseline_mc.first_payout_probability:.1f}%."
    )

    log(f"Searching for a more robust configuration ({cfg.ga_generations} generations x {cfg.ga_population} candidates)...")
    refine_cfg = RefinementConfig(
        population_size=cfg.ga_population,
        generations=cfg.ga_generations,
        fitness_metric=cfg.fitness_metric,
        search_monte_carlo_sims=cfg.ga_search_mc_sims,
        random_seed=cfg.random_seed,
    )
    ga_result = run_walkforward_aware_refinement(
        df, strategy, risk, prop_rules,
        MonteCarloConfig(n_simulations=cfg.ga_search_mc_sims, random_seed=cfg.random_seed),
        refinement_config=refine_cfg,
        n_folds=cfg.n_folds, window_mode=cfg.window_mode,
        progress_cb=lambda m: log(f"  {m}"),
        parallel=cfg.parallel, max_workers=cfg.parallel_max_workers,
        adaptive_risk=adaptive_risk,
        cancel_event=cancel_event,
    )
    warnings.extend(ga_result.warnings)

    if ga_result.best.oos_trade_count == 0:
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
    final_bt = run_backtest(df, final_strategy, risk, adaptive_risk=adaptive_risk)
    warnings.extend(final_bt.warnings)
    if instrument_mismatch_warning is None and has_instrument_scale_mismatch(final_bt.warnings):
        # Same mismatch, just not visible until the optimized parameters
        # were actually run -- risk.pip_size is unchanged from the
        # baseline check above, so this is the identical underlying
        # problem, not a new one the GA introduced.
        instrument_mismatch_warning = instrument_scale_mismatch_message(risk.pip_size)
        log(f"  !!! {instrument_mismatch_warning}")
    final_mc = run_monte_carlo(
        final_bt.trades, prop_rules, MonteCarloConfig(n_simulations=cfg.final_mc_sims, random_seed=cfg.random_seed)
    )
    log(
        f"Optimized: {len(final_bt.trades)} trades, net ${final_bt.statistics.net_profit:,.2f}, "
        f"win rate {final_bt.statistics.win_rate:.1f}%, eval pass {final_mc.evaluation_pass_probability:.1f}%, "
        f"payout {final_mc.first_payout_probability:.1f}%."
    )

    improved = (
        final_mc.evaluation_pass_probability > baseline_mc.evaluation_pass_probability
        or (
            final_mc.evaluation_pass_probability == baseline_mc.evaluation_pass_probability
            and final_bt.statistics.win_rate > baseline_bt.statistics.win_rate
        )
    )

    saved_library_path = None
    saved_library_note = None
    if cfg.save_to_library and final_source_type in ("python", "pinescript", "mql5") and final_code_text:
        ext = _EXT_FOR_SOURCE[final_source_type]
        base_name = Path(display_name).stem.replace(" ", "_") or "optimized_strategy"
        filename = f"{base_name}_optimized{ext}"
        try:
            try:
                saved_library_path = save_strategy_text(final_code_text, filename, final_source_type, overwrite=False)
            except StrategyAlreadyExists:
                filename = f"{base_name}_optimized_{int(time.time())}{ext}"
                saved_library_path = save_strategy_text(final_code_text, filename, final_source_type, overwrite=False)
            set_strategy_status(final_source_type, filename, cfg.library_status)
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
        base_name = Path(display_name).stem.replace(" ", "_") or "optimized_strategy"
        filename = f"{base_name}_optimized.json"
        config_text = json.dumps(final_config, indent=2)
        try:
            try:
                saved_library_path = save_strategy_text(config_text, filename, "manual", overwrite=False)
            except StrategyAlreadyExists:
                filename = f"{base_name}_optimized_{int(time.time())}.json"
                saved_library_path = save_strategy_text(config_text, filename, "manual", overwrite=False)
            set_strategy_status("manual", filename, cfg.library_status)
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
    )
