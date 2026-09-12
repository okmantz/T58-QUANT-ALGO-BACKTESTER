"""
Evolution Lab -- natural-selection-based strategy discovery.

    RESEARCH (knowledge graph, informs which families/features get
              weighted into GENERATE)
        v
    GENERATE ~N STRATEGIES   (app.search.strategy_space, all families)
        v
    PRE-FILTER               (one cheap backtest: trades/profit-factor/DD)
        v
    BACKTEST                 (full dataset, already done above)
        v
    ROBUSTNESS FILTER        (app.search.robustness.parameter_neighborhood_robustness)
        v
    OOS FILTER                (app.search.robustness.run_walk_forward)
        v
    MONTE CARLO               (app.monte_carlo.engine.run_monte_carlo)
        v
    PROP SIMULATION            (app.prop.simulator.simulate_account)
        v
    CPCV / PBO                (app.validation.cpcv -- REAL combinatorial
                                purged CV + genuine multi-candidate PBO,
                                applied to the best `cpcv_top_n` survivors
                                only -- it's the most expensive stage)
        v
    STRESS TEST                (re-run at N-x execution costs)
        v
    CLUSTER                    (correlation-dedupe on daily P&L, so the
                                 top 10 aren't 10 near-identical variants
                                 of the same winner)
        v
    KEEP TOP N -> record to knowledge graph -> MUTATE -> repeat

This reuses this app's existing, already-tested building blocks end to
end (Search Lab's strategy generator, the walk-forward GA's mutation
operators, robustness/walk-forward/CPCV/PBO, Monte Carlo, the prop
simulator) rather than reimplementing any of them -- see the imports
below for exactly which module each stage delegates to. The new code
here is the composite PROP FITNESS scoring (app.evolution.prop_fitness),
the knowledge graph (app.evolution.knowledge_graph), and the generation
loop itself (this module).

Known scope limits (stated plainly rather than glossed over):
- Candidates are Manual Strategy Builder configs generated from
  app.search.strategy_space's families -- this does not mutate uploaded
  Python/PineScript/MQL5 files. Manual configs are what Search Lab
  already generates and mutates today, so this is the same scope as the
  system Owen asked to have "combined," not a new restriction.
- PRE-FILTER and the ROBUSTNESS/OOS/MONTE CARLO/PROP SIMULATION stage
  (by far the two most expensive, once-per-candidate stages) now run
  across a ProcessPoolExecutor pool, the same pattern
  app.search.batch_runner already uses for its own stage1/2/3 and
  app.orchestration.full_pipeline uses for its multi-strategy batch --
  see EvolutionConfig.parallel_workers. CPCV/PBO, the stress test, and
  clustering stay serial (they only run against the small `cpcv_top_n`
  survivor pool per generation, not the full population, so parallelizing
  them buys much less for the added complexity).
"""
from __future__ import annotations

import multiprocessing
import os
import gc
import random
import tempfile
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np

from app.backtest.adaptive_risk import build_limit_aware_preset
from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.evolution import checkpoint as evo_checkpoint
from app.evolution.family_budget import FamilyBudgetTracker
from app.evolution.knowledge_graph import DEFAULT_KG_PATH, KnowledgeGraph, feature_vector_for_spec
from app.evolution.prop_fitness import PropFitnessBreakdown, compute_prop_fitness
from app.evolution.surrogate import FamilySurrogateBank
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.parameter_space import apply_genome, extract_genome
from app.optimize.refinement import _mutate, _stressed_risk_config
from app.orchestration.resource_guard import safe_worker_count
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.reports.crash_log import log_crash
from app.search.graveyard import GraveyardEntry, param_signature, record_rejections
from app.search.robustness import parameter_neighborhood_robustness, run_walk_forward
from app.search.strategy_space import (
    StrategySpaceError,
    build_strategy_from_spec,
    generate_search_space,
    list_families,
)
from app.strategy.library import (
    StrategyAlreadyExists, save_strategy_metadata, save_strategy_text, set_strategy_status,
)
from app.validation.cpcv import CPCVError, compute_pbo, run_cpcv

def evolution_stats_metadata(record: dict, generation: int | None = None) -> dict:
    """Builds the \"evolution\" metadata block attached to a strategy's
    library sidecar (see app.strategy.library.save_strategy_metadata) so
    the Dashboard / Strategy Library can show what the Evolution Lab
    actually found -- fitness score, eval-pass/first-payout probability,
    robustness, walk-forward efficiency -- instead of a strategy that
    was promoted from the lab looking exactly like any hand-built one
    with no lab context attached. `record` is one candidate's
    EvolutionCandidateRecord.to_checkpoint_dict() (or the equivalent
    dict loaded back from a checkpoint/leaderboard). Used by both the
    auto-save-every-generation path (_maybe_save_to_library below) and
    the web/desktop \"Promote to Strategy Library\" button, so both
    routes into the library carry identical stats.
    """
    stats = record.get("stats") or {}
    mc_summary = record.get("mc_summary") or {}
    robustness = record.get("robustness") or {}
    walk_forward = record.get("walk_forward") or {}
    fitness = record.get("fitness") or {}
    meta = record.get("meta") or {}
    return {
        "source": "evolution_lab",
        "family": meta.get("family"),
        "candidate_id": record.get("candidate_id"),
        "generation": generation,
        "fitness_score": fitness.get("final_score"),
        "eval_pass_probability": mc_summary.get("evaluation_pass_probability"),
        "first_payout_probability": mc_summary.get("first_payout_probability"),
        # The honest, held-out-fold estimate of eval_pass_probability from
        # CPCV -- only populated for candidates that made the top
        # cpcv_top_n pool. Prefer THIS over the raw eval_pass_probability
        # above when deciding whether a candidate is worth promoting: the
        # raw number is Monte-Carlo-resampled from a single in-sample
        # backtest over the exact data the GA searched/selected against,
        # so it systematically overstates a genome's real edge. See
        # EvolutionCandidateRecord.cpcv_oos_eval_pass_probability's
        # docstring for the full explanation.
        "cpcv_oos_eval_pass_probability": record.get("cpcv_oos_eval_pass_probability"),
        "cpcv_degradation": record.get("cpcv_degradation"),
        "robustness_stability": robustness.get("stability_ratio"),
        "walk_forward_efficiency": walk_forward.get("walk_forward_efficiency"),
        "net_profit": stats.get("net_profit"),
        "max_drawdown_pct": stats.get("max_drawdown_pct"),
        "total_trades": stats.get("total_trades"),
        "pbo": record.get("pbo"),
        "stressed_ok": record.get("stressed_ok"),
    }


ProgressCallback = "Callable[[str], None]"


# ---------------------------------------------------------------------------
# Worker process state & tasks (module-level so ProcessPoolExecutor can
# pickle/import them; state is per-process, populated once by
# _evo_init_worker -- same pattern app.search.batch_runner already uses).
# ---------------------------------------------------------------------------

_EVO_WORKER: dict = {}


def _evo_init_worker(df_pickle_path: str, risk_kwargs: dict, prop_kwargs: dict, adaptive_risk=None) -> None:
    global _EVO_WORKER
    _EVO_WORKER["df"] = pd.read_pickle(df_pickle_path)
    _EVO_WORKER["risk"] = RiskConfig(**risk_kwargs)
    _EVO_WORKER["prop_rules"] = PropRules(**prop_kwargs)
    _EVO_WORKER["adaptive_risk"] = adaptive_risk


def _evo_worker_ready_ping() -> bool:
    """No-op task -- see app.search.batch_runner._worker_ready_ping (the
    same fix, ported here for the exact same reason: a ProcessPoolExecutor
    only runs _evo_init_worker's full-dataset unpickling lazily, on each
    worker's FIRST real task, so without this it silently eats into
    _drain_futures' stall-timeout budget on a large dataset instead of
    being its own visible, accounted-for step."""
    return True


def _evo_prefilter_task(
    cid: str, spec: dict, meta: dict,
    min_trades: int, min_profit_factor: float, max_drawdown_pct: float, max_drawdown_buffer_mult: float,
    prefilter_max_bars: int | None = None,
):
    """One candidate's PRE-FILTER: build + backtest + the cheap pass/fail
    test, run in a worker process. Mirrors
    EvolutionRunner._prefilter_one's body exactly -- that instance method
    is what actually runs this when parallel_workers==1 or the pool is
    unavailable, so there's exactly one place the filter logic lives;
    this just re-parametrizes it without `self` so it can cross a
    process boundary. Returns a plain tuple (never raises) so one bad
    generated spec can't take down the whole prefilter batch.

    `prefilter_max_bars`, when set, backtests only the most recent N bars
    of the loaded dataset for THIS cheap pass only -- the worker's full
    dataset (_EVO_WORKER["df"]) is untouched, so any later stage that
    re-fetches it (full-eval, robustness, etc.) still sees every bar.
    See EvolutionConfig.prefilter_max_bars for why this exists.
    """
    df, risk = _EVO_WORKER["df"], _EVO_WORKER["risk"]
    if prefilter_max_bars and len(df) > prefilter_max_bars:
        df = df.tail(prefilter_max_bars)
    try:
        strategy = build_strategy_from_spec(spec)
        bt = run_backtest(df, strategy, risk, adaptive_risk=_EVO_WORKER.get("adaptive_risk"))
    except Exception as exc:  # noqa: BLE001 -- a bad generated config must not kill the pool
        return (cid, spec, meta, None, ["build_or_backtest_error"], str(exc)[:300], None)
    if not bt.trades:
        return (cid, spec, meta, None, ["no_trades"], None, None)

    stats = bt.statistics.to_dict()
    pf = stats.get("profit_factor", 0.0)
    pf_val = 10.0 if pf == float("inf") else float(pf or 0.0)
    n_trades = stats.get("total_trades", 0)
    max_dd = stats.get("max_drawdown_pct", 0.0) or 0.0

    reasons = []
    if n_trades < min_trades:
        reasons.append("min_trades")
    if pf_val < min_profit_factor:
        reasons.append("profit_factor")
    if max_dd > max_drawdown_pct * max_drawdown_buffer_mult:
        reasons.append("max_drawdown")
    if stats.get("net_profit", 0.0) <= 0:
        reasons.append("unprofitable")

    if reasons:
        return (cid, spec, meta, None, reasons, None, stats)
    return (cid, spec, meta, bt, [], None, stats)


def _evo_full_eval_task(
    cid: str, spec: dict, meta: dict, bt,
    mc_sims: int, robustness_perturbation_frac: float, robustness_neighbors: int,
    robustness_min_stability: float, walk_forward_folds: int, walk_forward_metric: str,
    min_trades_target_for_fitness: int, random_seed: int, fitness_goal: "dict | str | None" = "balanced",
):
    """One PRE-FILTER survivor's ROBUSTNESS / OOS / MONTE CARLO / PROP
    SIMULATION scoring, run in a worker process. Mirrors
    EvolutionRunner._full_eval_one's body exactly (same one-source-of-
    truth reasoning as _evo_prefilter_task above)."""
    df, risk, prop_rules = _EVO_WORKER["df"], _EVO_WORKER["risk"], _EVO_WORKER["prop_rules"]
    adaptive_risk = _EVO_WORKER.get("adaptive_risk")
    if adaptive_risk is not None:
        # Re-run at THIS stage under the limit-aware throttle -- the
        # pre-filter's `bt` (passed in) was deliberately run at plain
        # nominal sizing for pre-filter speed/consistency; the full
        # eval's own Monte Carlo/fitness must reflect the throttled
        # sizing that would actually be deployed.
        strategy = build_strategy_from_spec(spec)
        bt = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)
    stats = bt.statistics.to_dict()
    trade_pnls = [t.pnl for t in bt.trades]
    trade_dates = [t.entry_time for t in bt.trades]

    mc_cfg = MonteCarloConfig(n_simulations=mc_sims, random_seed=random_seed)
    mc = run_monte_carlo(bt.trades, prop_rules, mc_cfg)
    mc_summary = {
        "evaluation_pass_probability": mc.evaluation_pass_probability,
        "first_payout_probability": mc.first_payout_probability,
    }
    single_run = simulate_account(trade_pnls, trade_dates, prop_rules)
    summarize_single_run(single_run)  # surfaces prop-sim issues early; summary itself not needed downstream here

    robustness_dict = None
    try:
        robustness = parameter_neighborhood_robustness(
            spec, df, risk, prop_rules, mc_cfg,
            fitness_metric="eval_pass_probability",
            perturbation_frac=robustness_perturbation_frac,
            n_neighbors=robustness_neighbors,
            seed=random_seed,
            stability_threshold=robustness_min_stability,
        )
        if robustness is not None:
            robustness_dict = {"stability_ratio": robustness.stability_ratio, "is_stable": robustness.is_stable}
    except Exception:
        pass

    wf_dict = None
    try:
        wf = run_walk_forward(
            df, lambda spec=spec: build_strategy_from_spec(spec), risk,
            n_folds=walk_forward_folds, metric=walk_forward_metric,
            prop_rules=prop_rules, mc_cfg=mc_cfg,
        )
        if wf is not None:
            wf_dict = {"walk_forward_efficiency": wf.walk_forward_efficiency, "is_stable": wf.is_stable}
    except Exception:
        pass

    fitness = compute_prop_fitness(
        stats, mc_summary, robustness_dict, wf_dict, trade_pnls,
        min_trades_target=min_trades_target_for_fitness,
        weights=fitness_goal,
    )
    return EvolutionCandidateRecord(
        candidate_id=cid, spec=spec, meta=meta, stats=stats, mc_summary=mc_summary,
        robustness=robustness_dict, walk_forward=wf_dict, fitness=fitness, trade_pnls=trade_pnls,
        trades=bt.trades,
    )


@dataclass
class EvolutionConfig:
    population_size: int = 60
    elite_keep: int = 10
    families: list[str] | None = None          # None = every family in list_families()
    # When `families` is None (the caller wants "every family," not a
    # specific list), auto-excludes any family app.search.family_health
    # flags as a dead end (tested at least family_health_min_samples times
    # across every past Search Lab/Evolution Lab run combined, with zero
    # successes) -- so a fresh run's compute budget goes toward families
    # that have shown ANY signal instead of re-proving the same negative
    # result generation after generation. Never overrides an EXPLICIT
    # `families` list (the caller asked for those specifically), and never
    # excludes every registered family (see
    # app.search.family_health.apply_family_exclusions).
    auto_exclude_dead_end_families: bool = True
    family_health_min_samples: int = 30
    # Floor on how few families an exclusion pass is allowed to leave in
    # play -- see app.search.family_health.apply_family_exclusions' own
    # docstring for the real report this fixes ("every time I run the
    # evolution lab, it creates the same three strategies"): the old rule
    # only ever guaranteed "not zero," which let exclusions silently
    # accumulate over many sessions until just a handful of survivor
    # families were ever left in play.
    family_health_min_active_families: int = 6
    grid_points_per_gene: int = 3

    # Pre-filter (Stage 1, cheap)
    min_trades: int = 20
    min_profit_factor: float = 1.05
    max_drawdown_buffer_mult: float = 1.5
    # FIX (2026-09-11): _handle_empty_prefilter used to multiply this by
    # 1.25 every auto-relax cycle with NO ceiling -- on a long run stuck
    # empty for hundreds of generations (see the module-level note on
    # _handle_empty_prefilter) this compounds to an astronomically large,
    # physically meaningless number (observed: >10^13x in a real run's
    # log) that no longer means anything as a risk control. Once the
    # buffer crosses this cap, any real drawdown is already being
    # accepted -- capping it doesn't change what passes, it just stops
    # the number itself from becoming nonsense and misleading anyone
    # reading the log into thinking drawdown was ever the bottleneck.
    max_drawdown_buffer_mult_cap: float = 8.0

    # Robustness / OOS
    robustness_neighbors: int = 4
    robustness_perturbation_frac: float = 0.15
    robustness_min_stability: float = 0.4
    walk_forward_folds: int = 4
    walk_forward_metric: str = "eval_pass_probability"

    # Monte Carlo
    mc_sims: int = 1000

    # CPCV / PBO -- only run against the best `cpcv_top_n` survivors, since
    # genuine CPCV re-backtests every candidate up to `cpcv_max_paths` times.
    cpcv_top_n: int = 8
    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2
    cpcv_max_paths: int = 10
    cpcv_metric: str = "eval_pass_probability"

    # Stress test
    stress_cost_multiplier: float = 2.0

    # Cluster (dedupe near-identical survivors before picking the top N)
    cluster_correlation_threshold: float = 0.85

    # Mutation / next generation
    mutation_rate: float = 0.35
    mutation_strength: float = 0.25
    random_immigrant_frac: float = 0.3
    # Family diversity: without these two, a GA that finds one working
    # family early (e.g. mtf_pullback) starves every other family of
    # both fresh candidates AND elite/breeding slots within a handful of
    # generations -- not because the other families don't work, but
    # because uniform-random immigrant sampling over the pooled grid is
    # size-biased toward whichever family happens to have the biggest
    # grid, and unconstrained elite selection lets one high-scoring
    # family's descendants fill every breeding slot. Both floors below
    # exist specifically so "seed the GA with structurally different
    # edges" stays true for the whole run, not just generation 0.
    min_immigrants_per_family: int = 2   # every family gets at least this many fresh candidates, every generation
    max_elite_frac_per_family: float = 0.5   # no single family may hold more than this share of the elite/breeding pool

    # Adaptive family budget -- the graduated, intra-run counterpart to
    # family_health's binary cross-run exclusion (see
    # app.evolution.family_budget's module docstring for the full
    # rationale): every generation, the per-family immigrant count
    # above is scaled by a rolling multiplier based on THIS run's own
    # pre-filter/stress survival rate per family, so a family that's
    # actually converting into survivors gets more of the population
    # budget and a family that's producing nothing (but hasn't crossed
    # family_health's much stricter 30-sample dead-end bar) gets less --
    # never below min_immigrants_per_family, and always recoverable if
    # the family starts working again later in the run.
    adaptive_family_budget_enabled: bool = True
    adaptive_family_budget_window: int = 10
    adaptive_family_budget_min_frac: float = 0.4
    adaptive_family_budget_max_frac: float = 2.5

    # Stress-failure stagnation: FIX (2026-09-11) -- the generation loop
    # used to fall back to `stress_survivors or cpcv_pool` when NOTHING
    # survived the stress test, which let a candidate that had just
    # FAILED stress testing at N-x costs get promoted to the leaderboard
    # as "WINNER," logged into the journal with a confidence label, and
    # (worst of all) bred from as next generation's elite seed -- see
    # _run_generation's own comment at the cluster/elite step for the
    # full explanation. That fallback is now removed; when stress
    # produces zero survivors, elites for the next generation come from
    # near-miss seeding instead (same mechanism _handle_empty_prefilter
    # already uses for an empty pre-filter). This threshold controls how
    # many CONSECUTIVE stress-failure generations trigger a full,
    # elites-bypassing random-immigrant reset generation, to break out of
    # a GA that's stuck refining mutations of an idea that has already
    # proven it cannot survive realistic costs.
    stress_failure_stagnation_threshold: int = 5

    # Surrogate-model-guided search (replaces blind mutation for elite
    # breeding once enough history exists): a per-family Gaussian Process
    # fit on every fully-evaluated manual-config candidate's genome ->
    # fitness, proposing next-generation children by Upper Confidence
    # Bound instead of random mutation/crossover. See app.evolution.surrogate
    # for why this is pure numpy (no scipy/sklearn dependency) and why it
    # can only ever make proposals SMARTER, never required -- every
    # fallback path below is plain mutation, unchanged from before.
    surrogate_guided_search: bool = True
    surrogate_min_observations: int = 8
    surrogate_kappa: float = 1.5
    surrogate_pool_size: int = 300

    min_trades_target_for_fitness: int = 30
    max_generations: int | None = None          # None = run until stop() is called
    random_seed: int = 42

    # "Loop mode" -- Owen's ask: Evolution Lab already runs generation
    # after generation until stop() is called or max_generations is hit,
    # but it never stops ITSELF just because a genuinely good candidate
    # already showed up; someone has to notice the leaderboard and click
    # STOP. Setting a target here makes a run stop on its own the first
    # generation a leaderboard candidate's target_metric clears
    # target_eval_pass_pct -- checked in _run_loop right after each
    # generation's leaderboard update. None (the default) preserves the
    # exact old behavior (run forever / to max_generations regardless of
    # what's on the leaderboard).
    target_eval_pass_pct: float | None = None
    # "cpcv_oos_eval_pass_probability" (the honest held-out estimate) is
    # the default and strongly recommended metric -- see
    # EvolutionCandidateRecord's own field comment for why the raw
    # mc_summary one overstates a genome's real edge. "eval_pass_probability"
    # (the raw in-sample Monte Carlo number) is offered for a caller that
    # explicitly wants the old, less trustworthy leaderboard-sort metric
    # for some other reason.
    target_metric: str = "cpcv_oos_eval_pass_probability"

    # What to actually optimize for. Either a named preset (see
    # app.evolution.prop_fitness.FITNESS_GOAL_PRESETS -- "balanced" is the
    # original, unweighted PROP FITNESS formula), or a custom dict of
    # per-component weights (any keys omitted fall back to the default
    # weight for that component). See compute_prop_fitness's own
    # docstring for exactly what each weight does.
    fitness_goal: "dict | str | None" = "balanced"

    # Same limit-aware risk-throttle preset as Full Pipeline / Quick
    # Optimize (see app.backtest.adaptive_risk.build_limit_aware_preset)
    # -- off by default. When enabled, every candidate's pre-filter
    # backtest, full-eval backtest, and stress test all run under the
    # SAME throttled sizing that would actually be deployed, so a
    # candidate's fitness reflects survivability under throttling rather
    # than nominal unthrottled sizing.
    adaptive_risk_enabled: bool = False
    adaptive_risk_daily_profit_lock_pct: float | None = 80.0

    # Parallelism for PRE-FILTER and the ROBUSTNESS/OOS/MONTE CARLO/PROP
    # SIMULATION stage -- the two stages that run once per candidate and
    # dominate a generation's wall-clock time. None (default) auto-picks
    # os.cpu_count() (minimum 1); set to 1 to force the old fully serial
    # behavior (e.g. for debugging, or a machine where spawning worker
    # processes is undesirable). The pool is created once per run (not
    # once per generation) and reused across generations to avoid paying
    # worker-startup cost repeatedly.
    parallel_workers: int | None = None

    # Every this many generations, the worker pool is torn down and
    # lazily recreated fresh on the next generation that needs it --
    # cheap routine upkeep against the slow worker-process memory
    # fragmentation that a long, many-generation run can otherwise build
    # up until it eventually throws a MemoryError (see _run_loop's
    # MemoryError handling for the full rationale). Set to 0/None to
    # disable and only recycle reactively, after an actual MemoryError.
    pool_recycle_every_generations: int = 25

    save_to_library: bool = True
    library_status: str = "draft"
    knowledge_graph_path: str = str(DEFAULT_KG_PATH)

    # Checkpoint / resume -- so STOP then START again continues from the
    # last completed generation (same elites, leaderboard, journal)
    # instead of starting a brand new run from scratch. Resuming is
    # refused (with a clear log message, falling back to a fresh run)
    # if the market data being started with doesn't match what the
    # checkpoint was built from -- see app.evolution.checkpoint.
    resume_from_checkpoint: bool = True
    checkpoint_path: str = str(evo_checkpoint.default_checkpoint_path())
    tested_log_path: str = str(evo_checkpoint.default_tested_log_path())

    # Auto-relax -- if this many generations IN A ROW produce zero
    # PRE-FILTER survivors, the pre-filter thresholds are automatically
    # loosened once (same idea as Search Lab's own Stage 1 auto-relax)
    # instead of the run silently grinding forever with an empty
    # leaderboard and no indication why.
    # Lowered 3 -> 2 (2026-09-03): on a large intraday dataset (e.g. a
    # multi-year 1-minute feed), a single generation's PRE-FILTER stage
    # alone can take hours (50 candidates x one real backtest each over
    # millions of bars) -- waiting for 3 of those in a row before ever
    # relaxing meant an overnight run that never got there at all. See
    # prefilter_max_bars below for the other half of this fix.
    auto_relax_after_empty_generations: int = 2

    # PRE-FILTER-only bar cap (2026-09-03): the PRE-FILTER stage is
    # supposed to be the CHEAP stage -- one plain backtest per candidate,
    # just to weed out obvious non-starters before the expensive
    # robustness/OOS/Monte Carlo/CPCV stage runs on the survivors. On a
    # multi-year 1-minute dataset (millions of bars), even that "cheap"
    # backtest is expensive enough that a whole night can produce only 1-2
    # generations -- which is indistinguishable, from the log alone, from
    # the run being stuck (see _finish_empty_generation's elapsed-time
    # logging below, added for exactly this). Setting this caps PRE-FILTER
    # backtests to the most recent N bars ONLY; every survivor still goes
    # through the full stack (robustness/OOS/Monte Carlo/CPCV/stress) on
    # the COMPLETE dataset once it clears this cheap first pass -- nothing
    # ever gets a final verdict off less than the full data. None (the
    # default) auto-selects a cap once the loaded dataset exceeds
    # AUTO_PREFILTER_BAR_THRESHOLD bars (see _resolved_prefilter_max_bars),
    # and leaves normal-sized datasets (15m/1h/daily feeds) completely
    # untouched. Set to 0 to force no cap regardless of dataset size.
    prefilter_max_bars: int | None = None


# A multi-year 1-minute-bar dataset (Owen's GC1! feed: ~2.04M bars) is
# roughly this size or larger; a typical 15m/1h/daily feed is nowhere
# close. Only datasets at or above this size get an automatic PRE-FILTER
# cap when prefilter_max_bars is left at its default (None) -- see
# EvolutionConfig.prefilter_max_bars and EvolutionRunner._resolved_prefilter_max_bars.
AUTO_PREFILTER_BAR_THRESHOLD = 500_000
AUTO_PREFILTER_BAR_CAP = 250_000


@dataclass
class EvolutionCandidateRecord:
    candidate_id: str
    spec: dict
    meta: dict
    stats: dict | None = None
    mc_summary: dict | None = None
    robustness: dict | None = None
    walk_forward: dict | None = None
    pbo: float | None = None
    cpcv_degradation: float | None = None
    # The actual out-of-sample estimate CPCV computed for this candidate's
    # eval_pass_probability -- as opposed to mc_summary's
    # "evaluation_pass_probability", which is Monte-Carlo-resampled from a
    # SINGLE in-sample backtest over the whole dataset the GA searched on.
    # See _cpcv_and_pbo's docstring note: after many generations of
    # selection pressure evaluated against that same data, the winning
    # genome is systematically the one that happened to fit that data's
    # noise best, not necessarily the one with a real edge -- this field is
    # the honest number computed on genuinely held-out combinatorial-purged
    # folds instead, and is what a promoted strategy's Full Pipeline result
    # should be expected to resemble, not the raw in-sample one.
    cpcv_oos_eval_pass_probability: float | None = None
    stressed_ok: bool | None = None
    fitness: object = None                     # PropFitnessBreakdown
    trade_pnls: list = field(default_factory=list)
    trades: list = field(default_factory=list)  # raw Trade objects, for date-aligned cluster correlation

    def to_checkpoint_dict(self) -> dict:
        """Serializes everything needed to redisplay this record and seed
        future generations -- NOT the raw `trades` objects (not JSON-
        serializable and only needed transiently for same-generation
        cluster-dedupe correlation), and pnls are capped since a
        checkpoint is meant to be a small, fast-to-load file, not a full
        trade-by-trade record (the strategy itself is always saved to
        the Strategy Library separately, in full, if save_to_library is on)."""
        return {
            "candidate_id": self.candidate_id,
            "spec": self.spec,
            "meta": self.meta,
            "stats": self.stats,
            "mc_summary": self.mc_summary,
            "robustness": self.robustness,
            "walk_forward": self.walk_forward,
            "pbo": self.pbo,
            "cpcv_degradation": self.cpcv_degradation,
            "cpcv_oos_eval_pass_probability": self.cpcv_oos_eval_pass_probability,
            "stressed_ok": self.stressed_ok,
            "fitness": self.fitness.to_dict() if self.fitness is not None else None,
            "trade_pnls": self.trade_pnls[:500],
        }


def _record_from_dict(d: dict) -> EvolutionCandidateRecord:
    fitness = PropFitnessBreakdown(**d["fitness"]) if d.get("fitness") else None
    return EvolutionCandidateRecord(
        candidate_id=d.get("candidate_id", ""),
        spec=d.get("spec") or {},
        meta=d.get("meta") or {},
        stats=d.get("stats"),
        mc_summary=d.get("mc_summary"),
        robustness=d.get("robustness"),
        walk_forward=d.get("walk_forward"),
        pbo=d.get("pbo"),
        cpcv_degradation=d.get("cpcv_degradation"),
        cpcv_oos_eval_pass_probability=d.get("cpcv_oos_eval_pass_probability"),
        stressed_ok=d.get("stressed_ok"),
        fitness=fitness,
        trade_pnls=d.get("trade_pnls") or [],
        trades=[],
    )


def _daily_pnl_series(trades) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    rows = [(pd.Timestamp(t.exit_time or t.entry_time).normalize(), t.pnl) for t in trades]
    s = pd.Series([r[1] for r in rows], index=[r[0] for r in rows])
    return s.groupby(level=0).sum()


class EvolutionRunner:
    """Owns one Evolution Lab run: a background thread cycling through
    generations until stop() is called or max_generations is reached.
    Not thread-safe against being start()ed twice concurrently -- callers
    (the UI) are expected to check .is_running first, same convention as
    every other background job in this app."""

    def __init__(
        self,
        df: pd.DataFrame,
        risk: RiskConfig,
        prop_rules: PropRules,
        cfg: EvolutionConfig | None = None,
        progress_cb=None,
    ):
        self.df = df
        self.risk = risk
        self.prop_rules = prop_rules
        self.cfg = cfg or EvolutionConfig()
        self.adaptive_risk = build_limit_aware_preset(
            prop_rules, daily_profit_lock_pct=self.cfg.adaptive_risk_daily_profit_lock_pct,
        ) if self.cfg.adaptive_risk_enabled else None
        self.progress_cb = progress_cb
        self.knowledge_graph = KnowledgeGraph(Path(self.cfg.knowledge_graph_path))

        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self.is_running = False
        self.generation = 0
        self.leaderboard: list[EvolutionCandidateRecord] = []      # current top N (all-time best seen)
        self.journal: list[str] = []                                 # numbered HYPOTHESIS-style entries
        self._elites: list[tuple[dict, dict]] = []                    # seeds mutated children next gen
        # FIX (2026-09-03): before this, when a generation produced zero
        # true PRE-FILTER survivors, self._elites stayed [] and
        # _generate_population(gen, []) took the "no elites" branch --
        # 100% fresh random immigrants, every single generation. With a
        # compound bar (min trades AND profit_factor>=1.0 AND net_profit>0
        # AND a drawdown ceiling) on costly/real-spread data, blind random
        # sampling can run for dozens of generations without ever landing
        # inside that region by chance -- which is exactly what a batch log
        # showing "0/52 survived" for 20 straight generations looks like:
        # not "no strategy exists," but "the GA never got a foothold to
        # start climbing from." near_miss_seeds holds the best-scoring
        # PRE-FILTER *failures* each generation (ranked by profit factor,
        # not required to pass) so _generate_population can mutate around
        # the closest-to-profitable configs instead of only reshuffling
        # randomly until a true survivor appears on its own.
        self._near_miss_seeds: list[tuple[dict, dict]] = []
        self._consecutive_empty_generations = 0
        # FIX (2026-09-11): companion counter to _consecutive_empty_generations,
        # but for "reached full-eval and even CPCV, but NOTHING survived the
        # stress test" -- see stress_failure_stagnation_threshold's docstring
        # and _run_generation's cluster/elite step.
        self._consecutive_stress_failures = 0
        self._force_full_immigrant_next = False
        self._pending_family_counts: dict | None = None
        self._family_budget = FamilyBudgetTracker(
            window=self.cfg.adaptive_family_budget_window,
            min_frac=self.cfg.adaptive_family_budget_min_frac,
            max_frac=self.cfg.adaptive_family_budget_max_frac,
        )
        self.resumed = False                                          # set True if a checkpoint was loaded
        self._target_reached_by: EvolutionCandidateRecord | None = None
        self._pool: ProcessPoolExecutor | None = None
        self._pool_tmp_dir: tempfile.TemporaryDirectory | None = None
        self._surrogate = (
            FamilySurrogateBank(
                min_observations=self.cfg.surrogate_min_observations,
                kappa=self.cfg.surrogate_kappa,
            ) if self.cfg.surrogate_guided_search else None
        )

        self._load_checkpoint_if_compatible()
        self._apply_family_health_exclusions()

    def _apply_family_health_exclusions(self) -> None:
        """If the caller didn't pin an explicit family list (cfg.families
        is None, meaning "search every family"), resolves that ONCE here
        -- rather than on every generation -- to every family EXCEPT any
        app.search.family_health flags as a dead end. Mutates
        self.cfg.families directly so _generate_population's existing
        "families is None -> every family" logic doesn't need to change
        at all; it just sees an already-resolved list. A caller who DID
        pin specific families is never touched, even if one of them is
        itself flagged dead-end -- that's an explicit choice to keep
        testing it anyway (e.g. on a new instrument), not a mistake to
        correct.

        Also records the outcome onto self.family_health_status (a plain
        dict, not just a one-time log line) -- real report this addresses:
        "every time I run the evolution lab, it creates the same three
        strategies" turned out to be caused by this exact mechanism
        silently collapsing the active family list over many past
        sessions, with the only visibility being a console log line that
        was easy to miss and impossible to check after the fact. status()
        surfaces this so the UI can show it persistently instead."""
        self.family_health_status: dict = {"applied": False, "excluded": [], "active_family_count": None}
        if self.cfg.families is not None or not self.cfg.auto_exclude_dead_end_families:
            return
        try:
            from app.search.family_health import apply_family_exclusions
            survivors, excluded = apply_family_exclusions(
                min_samples=self.cfg.family_health_min_samples,
                min_active_families=self.cfg.family_health_min_active_families,
            )
        except Exception:  # noqa: BLE001 -- a family-health scan failing must never block starting a run
            return
        self.family_health_status["applied"] = True
        self.family_health_status["excluded"] = excluded
        if excluded:
            self._log(
                f"Auto-excluding {len(excluded)} dead-end famil{'y' if len(excluded) == 1 else 'ies'} "
                f"(tested {self.cfg.family_health_min_samples}+ times across past runs with zero "
                f"successes): {', '.join(excluded)}."
                + (
                    f" ({len(survivors)} famil{'y' if len(survivors) == 1 else 'ies'} still active.)"
                    if survivors is not None else
                    f" Would have left fewer than {self.cfg.family_health_min_active_families} families "
                    f"active -- searching all {len(list_families())} registered families anyway rather "
                    f"than collapsing the search space down to a stagnant handful."
                )
            )
        if survivors is not None:
            self.cfg.families = survivors
            self.family_health_status["active_family_count"] = len(survivors)
        else:
            self.family_health_status["active_family_count"] = len(list_families())

    # -- worker pool (PRE-FILTER + full-eval parallelism) ------------------
    def _ensure_pool(self) -> ProcessPoolExecutor | None:
        """Lazily creates the ProcessPoolExecutor used by _prefilter and
        _full_eval, reused across every generation of this run (workers
        are expensive to start, cheap to keep alive). Returns None (never
        raises) if parallel_workers resolves to 1 or the pool fails to
        start for any reason -- both _prefilter and _full_eval fall back
        to their serial per-candidate loop in that case, so a machine
        where spawning worker processes doesn't work still runs, just
        without the speedup."""
        if self._pool is not None:
            return self._pool
        workers = self.cfg.parallel_workers
        if workers is None:
            workers = os.cpu_count() or 1
        # Each worker below loads its OWN full copy of self.df (see
        # _evo_init_worker) -- on a large dataset this pool can, on its
        # own or stacked with another heavy job (Full Pipeline, Speed Run)
        # running at the same time, exhaust system memory well before CPU.
        # See app.orchestration.resource_guard for the full rationale.
        safe_workers = safe_worker_count(self.df, requested=workers)
        if safe_workers < workers:
            self._log(
                f"Reducing Evolution Lab worker processes from {workers} to {safe_workers} -- "
                f"{len(self.df):,} bars is large enough that {workers} full copies of it (one per "
                f"worker) would risk exhausting available memory."
            )
        workers = safe_workers
        if workers <= 1:
            return None
        try:
            self._pool_tmp_dir = tempfile.TemporaryDirectory(prefix="t58_evolution_")
            df_path = Path(self._pool_tmp_dir.name) / "data.pkl"
            self.df.to_pickle(df_path)
            self._pool = ProcessPoolExecutor(
                max_workers=workers, initializer=_evo_init_worker,
                initargs=(str(df_path), asdict(self.risk), asdict(self.prop_rules), self.adaptive_risk),
                # spawn, not the platform default fork: this runner always
                # lives inside a background thread (see start()), and
                # forking a multi-threaded process is unsafe/deprecated --
                # same reasoning as app.search.batch_runner's own pool.
                mp_context=multiprocessing.get_context("spawn"),
            )
            self._warm_up_pool(self._pool, workers, len(self.df))
            return self._pool
        except Exception:
            self._log(
                "Could not start a worker process pool for this run -- "
                "continuing single-process (slower, but still fully "
                "functional)."
            )
            self._shutdown_pool()
            return None

    def _warm_up_pool(self, pool: ProcessPoolExecutor, workers: int, n_bars: int) -> None:
        """Forces every worker's one-time _evo_init_worker (full-dataset
        unpickle) to run and complete before this pool is handed any real
        candidates -- see app.search.batch_runner._warm_up_pool's
        docstring for the full "why" (same fix, same bug, ported here for
        this runner's own separate pool/drain_futures pair). Without this,
        a large dataset's per-worker unpickling time silently counted
        against _drain_futures' stall-timeout with zero real progress to
        show for it, which is a very plausible explanation for "loaded
        2,353,209 bars ... generation 1 ... never continued": every
        worker was likely still loading that dataset, not wedged on a
        candidate, when the stall-timeout fired and started skip-and-
        respawn cycling -- paying the same large unpickling cost again
        on every respawn, without ever actually reaching generation 2.
        """
        t0 = time.monotonic()
        futures = [pool.submit(_evo_worker_ready_ping) for _ in range(max(1, workers))]
        try:
            done, pending = futures_wait(set(futures), timeout=max(60.0, n_bars / 2000.0))
        except Exception:
            return
        elapsed = time.monotonic() - t0
        if pending:
            self._log(
                f"  ** Worker pool warm-up: {len(pending)}/{len(futures)} worker(s) still hadn't "
                f"finished loading this {n_bars:,}-bar dataset after {elapsed:.0f}s. Continuing anyway "
                f"-- if this generation immediately reports every candidate as stalled/skipped, this "
                f"dataset is too large for this machine to hold {workers} full in-memory copies of "
                f"comfortably; try fewer parallel workers or a smaller/downsampled dataset."
            )
        elif elapsed > 5.0:
            self._log(f"  Worker pool ready ({workers} worker(s) loaded {n_bars:,} bars in {elapsed:.0f}s).")

    def _shutdown_pool(self) -> None:
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            # cancel_futures=True above only cancels futures that hadn't
            # started yet -- a worker already executing a candidate (e.g.
            # one that's hung, or just slow) keeps running until it
            # finishes on its own unless explicitly killed here. Without
            # this, STOP could mark the run as stopped while an orphaned
            # worker process kept chewing CPU/RAM in the background, and
            # a genuinely wedged worker would never go away at all until
            # the whole server restarted. Best-effort: `_processes` is a
            # private ProcessPoolExecutor attribute, not public API, so
            # this is wrapped defensively rather than relied on.
            try:
                for proc in list(getattr(self._pool, "_processes", {}).values()):
                    if proc.is_alive():
                        proc.terminate()
            except Exception:
                pass
            self._pool = None
        if self._pool_tmp_dir is not None:
            try:
                self._pool_tmp_dir.cleanup()
            except Exception:
                pass
            self._pool_tmp_dir = None

    # -- checkpoint / resume ----------------------------------------------
    def _load_checkpoint_if_compatible(self) -> None:
        """Loads generation/elites/leaderboard/journal from disk if the
        config asks to resume AND the checkpoint was built from the same
        market data this runner is starting with. Otherwise starts clean
        (still logging why, so a mismatched-data situation isn't silent)."""
        if not self.cfg.resume_from_checkpoint:
            return
        saved = evo_checkpoint.load_checkpoint(Path(self.cfg.checkpoint_path))
        if saved is None:
            return
        current_fp = evo_checkpoint.data_fingerprint(self.df)
        if saved.data_fingerprint and saved.data_fingerprint != current_fp:
            self._log(
                f"Found a saved Evolution Lab checkpoint (generation {saved.generation}, "
                f"{len(saved.leaderboard)} on its leaderboard) but it was built from different "
                f"market data than what's loaded now -- starting a fresh run instead of resuming "
                f"against mismatched data. (Use the same data file to resume that run.)"
            )
            return
        try:
            self.generation = saved.generation
            self._elites = [(e["spec"], e["meta"]) for e in saved.elites]
            self.leaderboard = [_record_from_dict(r) for r in saved.leaderboard]
            self.journal = list(saved.journal)
            self.resumed = True
            self._log(
                f"Resuming Evolution Lab from checkpoint: generation {self.generation}, "
                f"{len(self.leaderboard)} on the leaderboard, {len(self.journal)} journal "
                f"entries carried over. Click STOP at any time -- progress keeps saving after "
                f"every generation."
            )
        except Exception:
            self._log("Found a saved Evolution Lab checkpoint but couldn't load it cleanly -- starting fresh.")
            self.generation = 0
            self._elites = []
            self.leaderboard = []
            self.journal = []
            self.resumed = False

    def _save_checkpoint(self, next_generation: int) -> None:
        try:
            ckpt = evo_checkpoint.EvolutionCheckpoint(
                generation=next_generation,
                elites=[{"spec": s, "meta": m} for s, m in self._elites],
                leaderboard=[r.to_checkpoint_dict() for r in self.leaderboard],
                journal=list(self.journal),
                data_fingerprint=evo_checkpoint.data_fingerprint(self.df),
                saved_at=pd.Timestamp.now("UTC").isoformat(),
            )
            evo_checkpoint.save_checkpoint(ckpt, Path(self.cfg.checkpoint_path))
        except Exception:
            pass  # checkpointing is best-effort -- must never break the run itself

    def reset(self) -> None:
        """Discards the on-disk checkpoint and tested-candidates log so the
        next START begins a genuinely fresh run. Only safe to call while
        not running."""
        evo_checkpoint.clear_checkpoint(Path(self.cfg.checkpoint_path))
        evo_checkpoint.clear_tested_log(Path(self.cfg.tested_log_path))
        self.generation = 0
        self._elites = []
        self.leaderboard = []
        self.journal = []
        self.resumed = False
        self._target_reached_by = None

    # -- public controls ------------------------------------------------
    def start(self) -> None:
        if self.is_running:
            return
        self._stop_flag.clear()
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()

    def stop_and_wait(self, timeout: float = 5.0) -> bool:
        """Signals stop and blocks up to `timeout` seconds for the
        background thread to actually exit, so a caller (the web STOP
        route) can reflect the real state in its own response instead of
        the page still showing RUNNING until the next status poll.
        Returns True once the thread has genuinely stopped (now
        realistic within a second or two thanks to _drain_futures'
        polling, where it previously could hang indefinitely)."""
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return not self.is_running

    def status(self) -> dict:
        return {
            "running": self.is_running,
            "generation": self.generation,
            "leaderboard_size": len(self.leaderboard),
            "resumed": self.resumed,
            "family_health": getattr(self, "family_health_status", None),
            # Adaptive family budget (this run's own rolling per-family
            # survival rate -> immigrant-count multiplier) -- distinct
            # from family_health above, which is the binary, cross-run
            # dead-end exclusion. See app.evolution.family_budget.
            "family_budget": self._family_budget.status(),
            "consecutive_empty_generations": self._consecutive_empty_generations,
            "consecutive_stress_failures": self._consecutive_stress_failures,
            "max_drawdown_buffer_mult": self.cfg.max_drawdown_buffer_mult,
            # Loop mode -- see EvolutionConfig.target_eval_pass_pct.
            "target_eval_pass_pct": self.cfg.target_eval_pass_pct,
            "target_reached": self._target_reached_by is not None,
            "target_reached_candidate_id": (
                self._target_reached_by.candidate_id if self._target_reached_by else None
            ),
        }

    def graveyard_summary(self, top_n: int = 25) -> list[dict]:
        """The Strategy Graveyard's 'dead neighborhoods, most-tested
        first' view for this run -- see app.search.graveyard. Reads the
        same on-disk log _write_graveyard_entries appends to, so this
        reflects every stress-test failure from this run AND any prior
        resumed run against the same checkpoint path."""
        from app.search.graveyard import load_graveyard, summarize_graveyard
        path = Path(self.cfg.tested_log_path).with_name("strategy_graveyard.jsonl")
        rows = load_graveyard(path)
        return [c.to_dict() for c in summarize_graveyard(rows, top_n=top_n)]

    def finalists_report(self, top_n: int = 10, window_trading_days: int | None = None) -> list[dict]:
        """Runs the full payout funnel + Pareto frontier (see
        app.evolution.finalists) on the current leaderboard. Expensive
        relative to a status() call (a full Monte Carlo survival
        analysis per finalist, plus an even more expensive Rolling
        Evaluation Windows scan per finalist if window_trading_days is
        given) -- call on demand, not every generation."""
        from app.evolution.finalists import build_finalist_reports, pareto_frontier_for_finalists
        reports = build_finalist_reports(
            self.leaderboard, self.prop_rules, top_n=top_n, window_trading_days=window_trading_days,
        )
        pareto_frontier_for_finalists(reports)
        return [r.to_dict() for r in reports]

    def tested_candidates(self, limit: int = 500) -> list[dict]:
        """The "what was actually tested" record -- every candidate the
        PRE-FILTER stage has backtested this run (and prior resumed runs
        against the same data), pass or fail, with the reason it was
        rejected if it was. Read from disk so it survives a restart same
        as the checkpoint does."""
        return evo_checkpoint.read_tested_rows(Path(self.cfg.tested_log_path), limit=limit)

    # -- internals --------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.progress_cb:
            self.progress_cb(msg)

    def _drain_futures(self, futures: dict, on_result, stall_timeout: float = 240.0) -> None:
        """Consumes a {future: label} dict as futures complete, calling
        on_result(label, future) for each one -- used by both _prefilter
        and _full_eval instead of Python's `for f in as_completed(futures)`.

        as_completed() with no timeout blocks until the NEXT future
        completes, however long that takes, and the STOP button only
        works by setting a flag this loop checks between iterations --
        so one hung candidate (a pathological generated config, a wedged
        worker process, anything) used to make the whole generation --
        and STOP along with it -- block indefinitely. This polls with a
        short timeout instead, so the stop flag gets checked roughly
        once a second regardless of how long any individual candidate
        takes, and abandons the remaining futures immediately (rather
        than waiting on them) the moment a stop is requested; the actual
        worker processes are then force-terminated by _shutdown_pool.

        That alone only makes the STOP BUTTON responsive during a stall --
        it does nothing for a stall nobody notices in time to click Stop
        for (an unattended overnight run), which is exactly the "loaded
        2,353,209 bars ... generation 1 ... and never continued" report
        this second half addresses: one wedged worker (or a worker that
        silently died) left the remaining futures pending forever, and
        with nothing left to become newly "done", the loop above just
        polled quietly forever. This mirrors the stall-timeout-then-
        pool-respawn fix already shipped for Search Lab
        (app.search.batch_runner._drain_futures): if `stall_timeout`
        seconds pass with ZERO futures completing while at least one is
        still pending, every remaining pending future is assumed wedged.
        The pool's still-alive worker processes are terminated outright
        (cancel_futures=True alone only drops futures that hadn't
        STARTED yet -- it doesn't stop a worker already inside a hung
        call), a fresh pool is spawned via _ensure_pool() for the rest of
        this run, and the stuck candidate(s) are reported to on_result
        with fut=None (skipped, never scored) so the generation can
        finish with everything else instead of hanging forever.
        """
        pending = set(futures.keys())
        last_progress = time.monotonic()
        while pending:
            if self._stop_flag.is_set():
                for fut in pending:
                    fut.cancel()  # only frees futures that hadn't started yet; see _shutdown_pool for the rest
                return
            done, pending = futures_wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            if done:
                last_progress = time.monotonic()
                for fut in done:
                    on_result(futures[fut], fut)
                continue
            if not pending:
                break
            stalled_for = time.monotonic() - last_progress
            if stalled_for < stall_timeout:
                continue
            stuck_labels = [futures[f] for f in pending]
            self._log(
                f"  ** No progress for {int(stalled_for)}s -- {len(pending)} candidate(s) appear stuck "
                f"(a worker likely hung on one pathological candidate, or its process died silently): "
                f"{stuck_labels}. Terminating the stuck worker process(es), marking those candidate(s) "
                f"as skipped, and continuing with a freshly-spawned worker pool instead of hanging forever."
            )
            for fut in list(pending):
                fut.cancel()
            try:
                for proc in list(getattr(self._pool, "_processes", {}).values()):
                    if proc.is_alive():
                        proc.terminate()
            except Exception:
                pass
            try:
                if self._pool is not None:
                    self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._pool = None
            self._ensure_pool()  # respawn a fresh pool for the caller to keep submitting to
            for fut in pending:
                on_result(futures[fut], None)  # None fut == skipped, never scored
            pending = set()

    def _check_target_reached(self) -> "EvolutionCandidateRecord | None":
        """Loop mode's stop condition -- see EvolutionConfig.target_eval_pass_pct's
        own comment. Scans the current leaderboard (already sorted by
        fitness, but a high-fitness candidate is not necessarily the one
        clearing the target metric, so this checks every leaderboard row,
        not just [0]) for the first one whose target_metric value already
        clears the configured threshold. Returns None immediately if no
        target is configured, so this is a cheap no-op for every existing
        caller that never sets one."""
        if self.cfg.target_eval_pass_pct is None:
            return None
        for record in self.leaderboard:
            if self.cfg.target_metric == "cpcv_oos_eval_pass_probability":
                value = record.cpcv_oos_eval_pass_probability
            else:
                value = (record.mc_summary or {}).get("evaluation_pass_probability")
            if value is not None and value >= self.cfg.target_eval_pass_pct:
                return record
        return None

    def _run_loop(self) -> None:
        gen = self.generation
        try:
            while not self._stop_flag.is_set():
                if self.cfg.max_generations is not None and gen >= self.cfg.max_generations:
                    break
                try:
                    gen = self._run_one_generation(gen)
                except MemoryError:
                    # A long unattended run (this loop has no natural end --
                    # see max_generations's docstring) can run the OS out of
                    # memory well after CPU/worker-count sizing looked fine
                    # at startup: long-lived worker processes doing
                    # thousands of numpy/pandas allocate/free cycles
                    # gradually fragment their heap (glibc malloc arenas
                    # often never hand freed memory back to the OS), so
                    # "still have plenty of RAM" at generation 1 does not
                    # mean the same is true at generation 200 -- eventually
                    # even a small array allocation can fail. Previously
                    # this propagated straight to the outer except below,
                    # which logged "Evolution Lab crashed" and stopped the
                    # run entirely, requiring Owen to notice and manually
                    # click START again (which does resume from checkpoint,
                    # but silently ran into the exact same wall on the next
                    # long run). Instead: recycle the worker pool (a fresh
                    # pool means fresh worker processes with a clean heap),
                    # permanently halve the worker count for the rest of
                    # this run so the same generation doesn't immediately
                    # hit the same wall again, checkpoint what's already
                    # been found, and keep going.
                    self._log(
                        "  ** Ran out of memory mid-generation. Recycling the worker pool and halving "
                        "the worker count for the rest of this run (this is usually long-run memory "
                        "fragmentation in the worker processes, not a sign anything found so far is "
                        "invalid) -- progress up to this point is saved."
                    )
                    log_crash("Evolution Lab", exc=MemoryError(f"generation={gen}"), extra="auto-recovered: pool recycled, workers halved")
                    self._shutdown_pool()
                    current_workers = self.cfg.parallel_workers or (os.cpu_count() or 1)
                    self.cfg.parallel_workers = max(1, current_workers // 2)
                    self._save_checkpoint(next_generation=gen + 1)
                    gen += 1
                    continue
                gen += 1
                target_hit = self._check_target_reached()
                if target_hit is not None:
                    self._target_reached_by = target_hit
                    self._log(
                        f"  Loop mode target reached: candidate {target_hit.candidate_id} cleared "
                        f"{self.cfg.target_eval_pass_pct:.0f}% on {self.cfg.target_metric} "
                        f"at generation {gen - 1} -- stopping automatically."
                    )
                    break
                # Recycle the pool periodically regardless of errors --
                # cheap insurance against the same slow fragmentation
                # described above ever building up far enough to hit a
                # MemoryError in the first place. Workers are lazily
                # recreated by the next generation's _ensure_pool() call.
                if (
                    self.cfg.pool_recycle_every_generations
                    and self._pool is not None
                    and gen % self.cfg.pool_recycle_every_generations == 0
                ):
                    self._log(f"  Recycling worker pool after {self.cfg.pool_recycle_every_generations} generations (routine memory upkeep).")
                    self._shutdown_pool()
                gc.collect()
        except Exception as exc:
            self._log("Evolution Lab crashed:\n" + traceback.format_exc())
            # Written straight to disk (data/logs/crash_log.txt), independent
            # of this in-memory log widget/callback -- if the process itself
            # is killed (e.g. out-of-memory) shortly after this, the GUI log
            # above may never actually get painted or persisted anywhere.
            log_crash("Evolution Lab", exc=exc, extra=f"generation={gen}")
        finally:
            self._shutdown_pool()
            self.is_running = False
            self._log("Evolution Lab stopped. Progress is saved -- clicking START again resumes from here.")

    def _run_one_generation(self, gen: int) -> int:
        """Runs exactly one generation end to end and returns the same
        `gen` it was given (the caller increments). Factored out of
        _run_loop so a MemoryError raised anywhere in here can be caught
        by _run_loop around a single, well-defined unit of work instead of
        needing a matching except clause at every one of the loop's four
        `continue` points -- see _run_loop's MemoryError handling.
        """
        self.generation = gen
        t0 = time.time()
        self._log(f"===== GENERATION {gen} =====")

        # Breed from real elites once we have any; otherwise breed
        # from the best near-misses found so far rather than
        # resampling purely at random every generation (see
        # self._near_miss_seeds's docstring in __init__).
        breeding_pool = self._elites or self._near_miss_seeds
        if not self._elites and self._near_miss_seeds:
            self._log(f"  (no true survivor yet -- breeding from the {len(self._near_miss_seeds)} "
                      f"closest near-misses found so far instead of pure random search)")
        population = self._generate_population(gen, breeding_pool)
        self._log(f"GENERATE: {len(population)} candidates.")
        if self._stop_flag.is_set():
            return gen

        stage1_survivors, rejection_counts, near_miss_top = self._prefilter(population, gen)
        prefilter_elapsed = time.time() - t0
        self._log(f"PRE-FILTER + BACKTEST: {len(stage1_survivors)}/{len(population)} survived "
                  f"(took {prefilter_elapsed:.1f}s).")
        self._record_family_budget_prefilter(population, stage1_survivors)
        if not stage1_survivors:
            self._handle_empty_prefilter(gen, rejection_counts)
            if near_miss_top:
                self._near_miss_seeds = near_miss_top
        if self._stop_flag.is_set() or not stage1_survivors:
            self._finish_empty_generation(gen, population, elapsed=time.time() - t0)
            self._save_checkpoint(next_generation=gen + 1)
            del population
            return gen
        self._consecutive_empty_generations = 0

        evaluated = self._full_eval(stage1_survivors)
        self._log(f"ROBUSTNESS / OOS / MONTE CARLO / PROP SIMULATION: {len(evaluated)} candidates scored.")
        if self._surrogate is not None:
            self._record_surrogate_observations(evaluated)
        if self._stop_flag.is_set() or not evaluated:
            self._finish_empty_generation(gen, population, evaluated, elapsed=time.time() - t0)
            self._save_checkpoint(next_generation=gen + 1)
            del population, stage1_survivors, evaluated
            return gen

        cpcv_pool = self._cpcv_and_pbo(evaluated)
        self._log(f"CPCV / PBO: re-scored top {len(cpcv_pool)} candidates.")

        stress_survivors = self._stress_test(cpcv_pool)
        self._log(f"STRESS TEST ({self.cfg.stress_cost_multiplier:g}x costs): "
                  f"{len(stress_survivors)}/{len(cpcv_pool)} still fitness-positive.")
        self._record_family_budget_stress(cpcv_pool, stress_survivors)

        # FIX (2026-09-11): this used to be `self._cluster(stress_survivors
        # or cpcv_pool)` -- when NOTHING survives the stress test (which,
        # per Owen's own multi-hundred-generation log, was true almost
        # every generation), that fallback quietly promoted a candidate
        # that had just FAILED stress testing at N-x costs to "WINNER,"
        # put it on the leaderboard, and -- critically -- fed it into
        # self._elites below, so the NEXT generation bred mutated children
        # from a strategy already proven not to survive realistic costs.
        # Repeated over hundreds of generations, that is exactly the
        # "100% similarity to previously tested strategies, no novel
        # component" stagnation pattern in the uploaded log: the GA was
        # refining a dead neighborhood because nothing ever told it the
        # neighborhood was dead.
        #
        # Now: clustering/elite-seeding only ever draws from GENUINE
        # stress survivors. Every cpcv_pool candidate that did NOT survive
        # stress gets a Strategy Graveyard entry (with the actual
        # robustness/CPCV/Monte Carlo numbers that killed it -- see
        # _graveyard_entries_for_stress_failures) instead of silently
        # vanishing or getting mislabeled a winner.
        stress_failures = [r for r in cpcv_pool if id(r) not in {id(s) for s in stress_survivors}]
        if stress_failures:
            self._write_graveyard_entries(stress_failures, gen, stage_died="stress")

        if stress_survivors:
            self._consecutive_stress_failures = 0
            clustered = self._cluster(stress_survivors)
            clustered.sort(key=lambda r: r.fitness.final_score, reverse=True)
            new_elites = self._diversify_elites(clustered)
        else:
            # No genuine survivor this generation -- breed next generation
            # from the closest near-misses (ranked by fitness, same
            # mechanism _handle_empty_prefilter already uses) instead of
            # from a disguised failure. Leaderboard/library are untouched
            # this generation (nothing here has earned a spot on either).
            clustered = []
            new_elites = []
            ranked_failures = sorted(cpcv_pool, key=lambda r: r.fitness.final_score, reverse=True)
            self._near_miss_seeds = [(r.spec, r.meta) for r in ranked_failures[: self.cfg.elite_keep]]
            self._consecutive_stress_failures += 1
            self._log(
                f"  No candidate survived the stress test this generation -- breeding next generation "
                f"from the {len(self._near_miss_seeds)} closest near-misses instead of a disguised failure."
            )
            if self._consecutive_stress_failures >= self.cfg.stress_failure_stagnation_threshold:
                self._log(
                    f"  STAGNATION: {self._consecutive_stress_failures} generations in a row produced zero "
                    f"stress-test survivors -- forcing a full random-immigrant generation next round instead "
                    f"of continuing to refine mutations of ideas that keep failing under realistic costs."
                )
                self._force_full_immigrant_next = True
                self._consecutive_stress_failures = 0

        self._record_generation_to_knowledge_graph(evaluated, {r.candidate_id for r in new_elites})
        self._append_tested_log_full_eval(evaluated, gen)
        if new_elites:
            self._update_leaderboard(new_elites)
            self._maybe_save_to_library(new_elites)
        self._write_journal_entry(gen, population, stage1_survivors, evaluated, cpcv_pool, stress_survivors, new_elites)

        elapsed = time.time() - t0
        self._log(f"Generation {gen} complete in {elapsed:.1f}s. Best fitness so far: "
                  f"{self.leaderboard[0].fitness.final_score:.2f}" if self.leaderboard else f"Generation {gen} complete in {elapsed:.1f}s.")

        if new_elites:
            self._elites = [(r.spec, r.meta) for r in new_elites]
        else:
            # Nothing genuinely survived -- clear stale elites rather than
            # keep breeding from last generation's (this run's _generate_
            # population already falls back to self._near_miss_seeds,
            # updated above, whenever self._elites is empty).
            self._elites = []
        self._save_checkpoint(next_generation=gen + 1)
        # Every one of these can hold a full backtest's worth of Trade
        # objects (evaluated/cpcv_pool/stress_survivors/clustered all
        # reference the SAME EvolutionCandidateRecord instances, so this is
        # only a handful of distinct objects, not 4x the memory) -- explicit
        # cleanup here plus the periodic pool recycle in _run_loop is what
        # keeps a long, many-generation run from accumulating enough
        # fragmentation to eventually throw a MemoryError (see _run_loop).
        del population, stage1_survivors, evaluated, cpcv_pool, stress_survivors, clustered, new_elites
        return gen

    def _handle_empty_prefilter(self, gen: int, rejection_counts: dict) -> None:
        """Logs WHY nothing survived (instead of just "0 survived", which
        gives no way to tell "your data/settings genuinely can't produce
        a passing strategy" from "something's silently broken"), and
        auto-relaxes the pre-filter thresholds once every
        `auto_relax_after_empty_generations` consecutive empty
        generations -- same philosophy as Search Lab's own Stage 1
        auto-relax (app.search.batch_runner), so a run doesn't grind for
        hours with a leaderboard stuck at 0 for no visible reason."""
        breakdown = ", ".join(f"{k}: {v}" for k, v in rejection_counts.items() if v) or "no candidates built successfully"
        self._log(f"  Rejection breakdown -- {breakdown}.")
        self._consecutive_empty_generations += 1
        if self._consecutive_empty_generations >= self.cfg.auto_relax_after_empty_generations:
            old_trades, old_pf, old_dd = self.cfg.min_trades, self.cfg.min_profit_factor, self.cfg.max_drawdown_buffer_mult
            self.cfg.min_trades = max(5, int(self.cfg.min_trades * 0.6))
            self.cfg.min_profit_factor = max(1.0, round(self.cfg.min_profit_factor * 0.9, 3))
            # FIX (2026-09-11): capped -- see max_drawdown_buffer_mult_cap's
            # docstring. Below the cap this behaves exactly as before.
            self.cfg.max_drawdown_buffer_mult = min(
                self.cfg.max_drawdown_buffer_mult_cap,
                round(self.cfg.max_drawdown_buffer_mult * 1.25, 3),
            )
            # "Floored" only looks at min_trades/min_profit_factor -- the
            # two levers that actually gate "unprofitable"/"profit_factor"
            # rejections, which is what a real stuck run's breakdown
            # usually shows (see the docstring above). The drawdown
            # buffer is deliberately excluded from this check: it can
            # still be climbing toward its own cap for several more
            # cycles after trades/PF are floored, and waiting for IT to
            # also cap out before escaping would delay the fix on
            # exactly the runs that need it soonest.
            floored = self.cfg.min_trades == old_trades == 5 and self.cfg.min_profit_factor == old_pf == 1.0
            self._log(
                f"  AUTO-RELAX: {self._consecutive_empty_generations} generations in a row produced zero "
                f"pre-filter survivors, so the thresholds were automatically loosened -- min trades "
                f"{old_trades} -> {self.cfg.min_trades}, min profit factor {old_pf:.2f} -> "
                f"{self.cfg.min_profit_factor:.2f}, drawdown buffer x{self.cfg.max_drawdown_buffer_mult:.2f}"
                f"{' (capped)' if self.cfg.max_drawdown_buffer_mult >= self.cfg.max_drawdown_buffer_mult_cap else ''}. "
                f"If generations keep coming back empty even after this, the market data or the selected "
                f"families likely can't produce a profitable strategy at all on this instrument/timeframe."
            )
            # FIX (2026-09-11): the block above used to be the ENTIRE
            # response to "stuck empty," forever -- but min_trades and
            # min_profit_factor both hit hard floors quickly (5 trades,
            # PF 1.0), after which every subsequent "AUTO-RELAX" cycle
            # was a no-op for both of them and only kept inflating the
            # (now capped) drawdown buffer, a constraint the rejection
            # breakdown usually shows was never the actual blocker
            # (see the module-level real-run example: "profit_factor: 91,
            # unprofitable: 91" -- zero drawdown rejections). Once
            # thresholds are genuinely floored, relaxing them further
            # changes nothing; the real fix is to stop breeding
            # children from whatever's currently seeding this dead
            # neighborhood and give every family a fresh, full-strength
            # random trial instead.
            if floored:
                self._log(
                    "  STAGNATION: pre-filter thresholds are already at their floor (min trades 5, "
                    "min profit factor 1.0) -- further relaxing them will not help. Forcing a full "
                    "random-immigrant generation (bypassing elite mutation) to give every active family "
                    "a fresh, full-strength trial instead of continuing to refine whatever seeded this "
                    "dead neighborhood."
                )
                self._force_full_immigrant_next = True
            self._consecutive_empty_generations = 0

    def _finish_empty_generation(self, gen, population, evaluated=None, elapsed: float | None = None) -> None:
        suffix = f" (took {elapsed:.1f}s)" if elapsed is not None else ""
        self._log(f"Generation {gen}: nothing survived far enough to update the leaderboard{suffix} -- continuing to the next generation.")

    # -- GENERATE ---------------------------------------------------------
    def _record_surrogate_observations(self, evaluated: list["EvolutionCandidateRecord"]) -> None:
        """Feeds every fully-evaluated manual-config candidate's genome ->
        final_score into this run's per-family surrogate bank. Best-effort:
        a malformed config/gene mismatch skips that one candidate rather
        than aborting the generation -- the surrogate is an optimization
        hint, never load-bearing."""
        for r in evaluated:
            if r.spec.get("source_type") != "manual" or r.fitness is None:
                continue
            config = r.spec.get("config")
            if not config:
                continue
            try:
                genes = extract_genome(config)
                if not genes:
                    continue
                # extract_genome(config) reads each gene's CURRENT value out
                # of this exact candidate's config into base_value, so this
                # already reflects what this candidate actually tested.
                genome_norm = np.array([
                    (g.base_value - g.lo) / (g.hi - g.lo) if g.hi > g.lo else 0.5
                    for g in genes
                ])
                fam = r.meta.get("family", "?")
                self._surrogate.observe(fam, genome_norm, r.fitness.final_score)
            except Exception:  # noqa: BLE001 -- surrogate learning must never break a run
                continue

    def _generate_population(self, gen: int, elites: list[tuple[dict, dict]]) -> list[tuple[str, dict, dict]]:
        """Returns a list of (candidate_id, spec, meta). Generation 0 is
        pure random family sampling. Later generations mix mutated
        children of the previous top N with a fresh slice of random
        immigrants for diversity (same random-immigrant idea the
        walk-forward GA already uses, applied at the population level).

        Immigrants are sampled PER FAMILY (stratified), not as one pooled
        draw across every family's combined grid -- a pooled draw is
        size-biased toward whichever family happens to have the largest
        parameter grid, and once the immigrant budget shrinks after
        generation 0 (see min_immigrants_per_family's docstring on
        EvolutionConfig), a size-biased pooled draw can go whole
        generations without a single candidate from a smaller family.
        Stratifying, with a floor of min_immigrants_per_family per
        family, is what keeps "mean reversion" / "volatility breakout" /
        "session timing" / "stat pairs" etc. genuinely in contention for
        the entire run instead of only at generation 0.
        """
        seed = self.cfg.random_seed + gen

        # FIX (2026-09-11): when _handle_empty_prefilter's floor-detection
        # trips (see its docstring), bypass elite mutation entirely for
        # ONE generation -- pure random stratified immigrants across every
        # active family at full population size, same as generation 0.
        # This is what actually breaks a GA that's spent many generations
        # only ever refining mutations of whatever seeded the current
        # elite/near-miss pool.
        force_full_immigrant = self._force_full_immigrant_next
        if force_full_immigrant:
            self._force_full_immigrant_next = False
            self._log(f"  GENERATE: forcing a full random-immigrant generation (elite mutation skipped this round).")
            elites = []

        n_immigrants = self.cfg.population_size if not elites else max(1, int(self.cfg.population_size * self.cfg.random_immigrant_frac))

        active_families = list(self.cfg.families) if self.cfg.families else list(list_families().keys())
        n_fam = max(1, len(active_families))
        base_per_family = max(self.cfg.min_immigrants_per_family, n_immigrants // n_fam)

        # Adaptive family budget (see app.evolution.family_budget's module
        # docstring): scales base_per_family by this run's own rolling
        # per-family survival rate. A family with no history yet gets
        # multiplier 1.0 (base_per_family unchanged); disabled entirely
        # falls back to the plain uniform base_per_family for every family,
        # identical to behavior before this existed.
        budget_multipliers = (
            self._family_budget.multipliers(active_families)
            if self.cfg.adaptive_family_budget_enabled else {fam: 1.0 for fam in active_families}
        )

        out: list[tuple[str, dict, dict]] = []
        for i, fam in enumerate(active_families):
            per_family = max(self.cfg.min_immigrants_per_family, round(base_per_family * budget_multipliers.get(fam, 1.0)))
            try:
                fam_space = generate_search_space(
                    mode="family", family=fam,
                    max_candidates=per_family, seed=seed + i * 7919,  # distinct seed per family, still reproducible
                    grid_points_per_gene=self.cfg.grid_points_per_gene,
                )
            except StrategySpaceError:
                # e.g. stat_pairs requested with no pair data merged in --
                # skip that one family rather than failing the whole generation.
                continue
            out.extend((cid, spec, fam_space.meta[cid]) for cid, spec in fam_space.candidates.items())

        if elites:
            rng = random.Random(seed)
            np_rng = np.random.default_rng(seed)
            n_children = max(0, self.cfg.population_size - len(out))
            per_elite = max(1, n_children // len(elites))
            for spec, meta in elites:
                config = spec.get("config")
                if not config:
                    continue
                genes = extract_genome(config)
                if not genes:
                    continue
                base_genome = [g.base_value for g in genes]
                fam = meta.get("family", "mutant")

                # Surrogate-guided proposal: once this family has enough
                # observed (genome -> fitness) history, propose children by
                # UCB over the fitted GP instead of blind mutation. Falls
                # through to plain mutation for whatever the surrogate
                # doesn't cover (not enough history yet, a fit failure, or
                # simply per_elite - len(surrogate_children) leftover slots)
                # so a cold-start family or a failed fit never loses
                # candidates, it just gets the same behavior as before.
                surrogate_children_norm = None
                if self._surrogate is not None:
                    surrogate_children_norm = self._surrogate.propose(
                        fam, per_elite, np_rng, len(genes), pool_size=self.cfg.surrogate_pool_size,
                    )

                n_from_surrogate = 0
                if surrogate_children_norm is not None:
                    for genome_norm in surrogate_children_norm:
                        child_genome = [
                            float(g.lo + float(v) * (g.hi - g.lo)) for g, v in zip(genes, genome_norm)
                        ]
                        child_genome = [
                            float(round(v)) if g.is_int else v for g, v in zip(genes, child_genome)
                        ]
                        child_config = apply_genome(config, genes, child_genome)
                        child_spec = {"source_type": "manual", "config": child_config}
                        cid = f"{fam}-gen{gen}-{rng.randrange(10**8):08x}"
                        out.append((cid, child_spec, {"family": fam, "params": {}, "mutated_from": meta.get("family"), "surrogate": True}))
                        n_from_surrogate += 1

                for _ in range(max(0, per_elite - n_from_surrogate)):
                    child_genome = _mutate(base_genome, genes, self.cfg.mutation_rate, self.cfg.mutation_strength, rng)
                    child_config = apply_genome(config, genes, child_genome)
                    child_spec = {"source_type": "manual", "config": child_config}
                    cid = f"{fam}-gen{gen}-{rng.randrange(10**8):08x}"
                    out.append((cid, child_spec, {"family": fam, "params": {}, "mutated_from": meta.get("family")}))
        return out[: max(self.cfg.population_size, len(out))]

    def _diversify_elites(self, clustered: list[EvolutionCandidateRecord]) -> list[EvolutionCandidateRecord]:
        """Picks the elite/breeding pool for the next generation off
        `clustered` (already fitness-ranked, best first), capping any one
        family's share at max_elite_frac_per_family instead of just
        taking the top elite_keep outright.

        Without this cap, one family scoring even slightly better early
        can fill every elite slot for the rest of the run -- since
        elites are what _generate_population mutates into next
        generation's children, an all-one-family elite pool means every
        "new" candidate after generation 0 is really just a variation on
        that one family's entry logic, no matter how diverse the fresh
        immigrants are. Backfills with the next-best candidates from
        OTHER families first; only falls back to filling remaining slots
        regardless of family once every other family's candidates are
        exhausted (so a genuinely one-family-survives generation still
        fills its elite_keep quota rather than wasting slots).
        """
        cap = max(1, int(self.cfg.elite_keep * self.cfg.max_elite_frac_per_family))
        family_counts: dict[str, int] = {}
        picked: list[EvolutionCandidateRecord] = []
        deferred: list[EvolutionCandidateRecord] = []
        for r in clustered:
            fam = r.meta.get("family", "?")
            if family_counts.get(fam, 0) < cap:
                picked.append(r)
                family_counts[fam] = family_counts.get(fam, 0) + 1
            else:
                deferred.append(r)
            if len(picked) >= self.cfg.elite_keep:
                break
        if len(picked) < self.cfg.elite_keep:
            picked.extend(deferred[: self.cfg.elite_keep - len(picked)])
        return picked

    # -- PRE-FILTER + BACKTEST --------------------------------------------
    def _resolved_prefilter_max_bars(self) -> int | None:
        """Resolves EvolutionConfig.prefilter_max_bars to an actual cap:
        an explicit positive value is used as-is, 0 means "no cap" no
        matter the dataset size, and None (the default) auto-caps only
        once the loaded dataset is at or above AUTO_PREFILTER_BAR_THRESHOLD
        -- see that constant and EvolutionConfig.prefilter_max_bars for
        the full reasoning. Logs once per run (not once per generation)
        so it's visible without being noisy."""
        configured = self.cfg.prefilter_max_bars
        if configured == 0:
            return None
        if configured:
            return configured
        if len(self.df) >= AUTO_PREFILTER_BAR_THRESHOLD:
            if not getattr(self, "_logged_auto_prefilter_cap", False):
                self._logged_auto_prefilter_cap = True
                self._log(
                    f"  NOTE: loaded dataset has {len(self.df):,} bars (>= {AUTO_PREFILTER_BAR_THRESHOLD:,}) -- "
                    f"auto-capping the cheap PRE-FILTER backtest to the most recent {AUTO_PREFILTER_BAR_CAP:,} "
                    f"bars for speed. Every PRE-FILTER survivor still gets its full robustness/OOS/Monte "
                    f"Carlo/CPCV/prop-simulation evaluation on the COMPLETE dataset -- only this cheap first "
                    f"pass is capped. Set 'Pre-filter bars cap' explicitly (or to 0 for no cap) to override."
                )
            return AUTO_PREFILTER_BAR_CAP
        return None

    def _prefilter(self, population: list[tuple[str, dict, dict]], gen: int):
        """Runs every candidate in `population` through build+backtest+the
        cheap pass/fail test. Dispatches across the worker pool (see
        _ensure_pool) when one is available, falling back to a plain
        serial loop -- in-process, calling the exact same
        _evo_prefilter_task function each candidate would run in a
        worker -- if the pool couldn't be started or parallel_workers==1.
        Either path produces identical survivors/rejection_counts/
        tested_rows; only the wall-clock time differs."""
        survivors = []
        near_misses: list[tuple[float, dict, dict]] = []  # (rank_score, spec, meta) for candidates that traded but failed a gate
        rejection_counts = {
            "build_or_backtest_error": 0, "no_trades": 0, "min_trades": 0,
            "profit_factor": 0, "max_drawdown": 0, "unprofitable": 0,
        }
        tested_rows = []

        def _consume(cid, spec, meta, bt, reasons, error, stats):
            # Mirrors _evo_prefilter_task's four possible return shapes
            # exactly (see its docstring/body): a build/backtest error, a
            # zero-trade candidate, a candidate that ran but failed one or
            # more cheap filters, or a genuine survivor.
            if error is not None:
                rejection_counts["build_or_backtest_error"] += 1
                tested_rows.append(self._tested_row(cid, meta, gen, passed=False, reasons=reasons, error=error))
                return
            if stats is None:
                rejection_counts["no_trades"] += 1
                tested_rows.append(self._tested_row(cid, meta, gen, passed=False, reasons=reasons))
                return
            n_trades = stats.get("total_trades", 0)
            pf = stats.get("profit_factor", 0.0)
            pf_val = 10.0 if pf == float("inf") else float(pf or 0.0)
            max_dd = stats.get("max_drawdown_pct", 0.0) or 0.0
            tested_rows.append(self._tested_row(
                cid, meta, gen, passed=not reasons, reasons=reasons,
                n_trades=n_trades, profit_factor=pf_val, net_profit=stats.get("net_profit"),
                max_drawdown_pct=max_dd,
            ))
            if reasons:
                for r in reasons:
                    rejection_counts[r] += 1
                # Rank near-misses mainly on profit factor (the gate that's
                # hardest to clear by luck), with net profit as a tiebreak --
                # a candidate at pf=0.97 with a small loss is a genuinely
                # closer miss than one at pf=0.2, even though both failed.
                rank_score = (pf_val, stats.get("net_profit") or 0.0)
                near_misses.append((rank_score, spec, meta))
            else:
                survivors.append((cid, spec, meta, bt))

        pool = self._ensure_pool()
        prefilter_max_bars = self._resolved_prefilter_max_bars()
        if pool is None:
            # Serial fallback still calls _evo_prefilter_task (single
            # source of truth for the filter logic) -- it just runs it
            # in-process instead of in a worker, so _EVO_WORKER (normally
            # populated once per worker process by _evo_init_worker) needs
            # seeding here too.
            _EVO_WORKER["df"], _EVO_WORKER["risk"] = self.df, self.risk
            _EVO_WORKER["adaptive_risk"] = self.adaptive_risk
            for cid, spec, meta in population:
                if self._stop_flag.is_set():
                    break
                cid_out, spec_out, meta_out, bt, reasons, error, stats = _evo_prefilter_task(
                    cid, spec, meta, self.cfg.min_trades, self.cfg.min_profit_factor,
                    self.prop_rules.max_drawdown_pct, self.cfg.max_drawdown_buffer_mult,
                    prefilter_max_bars,
                )
                _consume(cid_out, spec_out, meta_out, bt, reasons, error, stats)
        else:
            futures = {
                pool.submit(
                    _evo_prefilter_task, cid, spec, meta, self.cfg.min_trades, self.cfg.min_profit_factor,
                    self.prop_rules.max_drawdown_pct, self.cfg.max_drawdown_buffer_mult,
                    prefilter_max_bars,
                ): (cid, spec, meta)
                for cid, spec, meta in population
            }

            # Periodic "N/total evaluated" progress, logged roughly every
            # 10% of the batch (at least every candidate on tiny batches).
            # Without this, a large dataset where each candidate's
            # backtest genuinely takes tens of seconds produces total
            # silence between "GENERATE: N candidates" and either the
            # next generation's log line or a stall warning many minutes
            # later -- indistinguishable from a real hang even when the
            # run is working exactly as intended. See _drain_futures'
            # docstring for the separate, real-stall-recovery half of
            # this fix; this half is purely about visibility during a
            # slow-but-healthy run.
            total = len(futures)
            log_every = max(1, total // 10)
            done_count = 0

            def _on_result(label, future):
                nonlocal done_count
                cid, spec, meta = label
                if future is None:
                    # Stall-recovery skip (see _drain_futures) -- the worker
                    # handling this candidate was terminated as wedged, not
                    # actually evaluated. Recorded honestly as an error, not
                    # as a pass/fail on the strategy itself.
                    _consume(cid, spec, meta, None, ["build_or_backtest_error"], "skipped: worker pool stalled", None)
                else:
                    try:
                        _, _, _, bt, reasons, error, stats = future.result()
                    except Exception as exc:  # noqa: BLE001 -- a dead worker must not kill the generation
                        _consume(cid, spec, meta, None, ["build_or_backtest_error"], str(exc)[:300], None)
                        done_count += 1
                        return
                    _consume(cid, spec, meta, bt, reasons, error, stats)
                done_count += 1
                if done_count % log_every == 0 or done_count == total:
                    self._log(f"  PRE-FILTER: {done_count}/{total} candidate(s) evaluated...")

            self._drain_futures(futures, _on_result)

        evo_checkpoint.append_tested_rows(tested_rows, Path(self.cfg.tested_log_path))
        near_misses.sort(key=lambda t: t[0], reverse=True)
        near_miss_top = [(spec, meta) for _score, spec, meta in near_misses[: self.cfg.elite_keep]]
        return survivors, rejection_counts, near_miss_top

    def _tested_row(self, cid: str, meta: dict, gen: int, passed: bool, reasons: list[str],
                     n_trades=None, profit_factor=None, net_profit=None, max_drawdown_pct=None,
                     error: str | None = None) -> dict:
        """One row of the durable "what was tested" log -- deliberately
        flat/small (no spec/config blob) so the log stays cheap to append
        to and to read back for a multi-hour, many-thousand-candidate
        run; the full spec for anything that mattered (an elite/leaderboard
        entry) is separately captured in the checkpoint and, when
        save_to_library is on, the Strategy Library."""
        return {
            "generation": gen,
            "candidate_id": cid,
            "family": meta.get("family", "?"),
            "stage": "prefilter",
            "passed": passed,
            "reasons": reasons,
            "n_trades": n_trades,
            "profit_factor": profit_factor,
            "net_profit": net_profit,
            "max_drawdown_pct": max_drawdown_pct,
            "error": error,
        }

    def _append_tested_log_full_eval(self, evaluated: list["EvolutionCandidateRecord"], gen: int) -> None:
        """Appends a second row for every candidate that made it past
        PRE-FILTER into the expensive robustness/OOS/Monte Carlo/prop-sim
        stage, this time with the PROP FITNESS score -- so "what was
        tested" shows not just the cheap pass/fail but how far a
        candidate actually got and how good it turned out to be."""
        rows = []
        for r in evaluated:
            rows.append({
                "generation": gen,
                "candidate_id": r.candidate_id,
                "family": r.meta.get("family", "?"),
                "stage": "full_eval",
                "passed": True,
                "reasons": [],
                "n_trades": (r.stats or {}).get("total_trades"),
                "profit_factor": (r.stats or {}).get("profit_factor"),
                "net_profit": (r.stats or {}).get("net_profit"),
                "max_drawdown_pct": (r.stats or {}).get("max_drawdown_pct"),
                "fitness_score": r.fitness.final_score if r.fitness else None,
                "pass_probability": (r.mc_summary or {}).get("evaluation_pass_probability"),
                "error": None,
            })
        evo_checkpoint.append_tested_rows(rows, Path(self.cfg.tested_log_path))

    # -- ROBUSTNESS + OOS + MONTE CARLO + PROP SIMULATION ------------------
    def _full_eval(self, stage1_survivors) -> list[EvolutionCandidateRecord]:
        """Runs every PRE-FILTER survivor through ROBUSTNESS / OOS / MONTE
        CARLO / PROP SIMULATION scoring. Same dispatch-to-pool-or-fall-
        back-serial shape as _prefilter (see its docstring) -- both paths
        call _evo_full_eval_task, so there's one place this logic lives."""
        args = (
            self.cfg.mc_sims, self.cfg.robustness_perturbation_frac, self.cfg.robustness_neighbors,
            self.cfg.robustness_min_stability, self.cfg.walk_forward_folds, self.cfg.walk_forward_metric,
            self.cfg.min_trades_target_for_fitness, self.cfg.random_seed, self.cfg.fitness_goal,
        )
        records: list[EvolutionCandidateRecord] = []
        eval_pool = self._ensure_pool()
        if eval_pool is None:
            _EVO_WORKER["df"], _EVO_WORKER["risk"], _EVO_WORKER["prop_rules"] = self.df, self.risk, self.prop_rules
            _EVO_WORKER["adaptive_risk"] = self.adaptive_risk
            for cid, spec, meta, bt in stage1_survivors:
                if self._stop_flag.is_set():
                    break
                try:
                    records.append(_evo_full_eval_task(cid, spec, meta, bt, *args))
                except Exception:
                    self._log(f"  full-eval error on {cid}:\n" + traceback.format_exc())
        else:
            futures = {
                eval_pool.submit(_evo_full_eval_task, cid, spec, meta, bt, *args): cid
                for cid, spec, meta, bt in stage1_survivors
            }

            # See the matching progress-logging comment in _prefilter --
            # full-eval's per-candidate robustness/OOS/Monte Carlo/prop-sim
            # work is more expensive than the pre-filter pass, so silence
            # here is even more likely to read as a hang than a run.
            total = len(futures)
            log_every = max(1, total // 10)
            done_count = 0

            def _on_result(cid, future):
                nonlocal done_count
                if future is None:
                    # Stall-recovery skip (see _drain_futures) -- this
                    # candidate's worker was terminated as wedged; it never
                    # produced a scored record.
                    self._log(f"  full-eval skipped {cid}: worker pool stalled (stall recovery).")
                else:
                    try:
                        records.append(future.result())
                    except Exception:  # noqa: BLE001 -- a dead worker must not kill the generation
                        self._log(f"  full-eval error on {cid}:\n" + traceback.format_exc())
                done_count += 1
                if done_count % log_every == 0 or done_count == total:
                    self._log(f"  FULL-EVAL: {done_count}/{total} candidate(s) evaluated (robustness/OOS/Monte Carlo/prop-sim)...")

            self._drain_futures(futures, _on_result)
        return records

    # -- CPCV / PBO ---------------------------------------------------------
    def _cpcv_and_pbo(self, evaluated: list[EvolutionCandidateRecord]) -> list[EvolutionCandidateRecord]:
        """Re-scores the top `cpcv_top_n` candidates (by raw fitness) with
        Combinatorial Purged Cross-Validation + PBO -- this is the ONLY
        point in the whole Evolution Lab pipeline that evaluates a
        candidate against data it wasn't itself selected against.
        Everything upstream (pre-filter, full eval's own Monte Carlo/
        robustness/walk-forward) all runs the backtest over the SAME
        entire `self.df` the GA has been searching and selecting against
        for potentially hundreds of generations -- so a high raw
        eval_pass_probability there is exactly as likely to mean "this
        genome happened to fit this dataset's noise well" as "this
        genome has a real edge." This is precisely why a strategy
        promoted straight off the raw leaderboard number can look like
        40%/30% here and come back 2%/1% out of Full Pipeline's much
        stricter walk-forward-search + genuinely-held-out holdout split:
        Full Pipeline is measuring something CPCV also measures here
        (out-of-sample performance) that the rest of this pipeline
        never does. See cpcv_oos_eval_pass_probability's docstring."""
        pool = sorted(evaluated, key=lambda r: r.fitness.final_score, reverse=True)[: self.cfg.cpcv_top_n]
        if len(pool) < 2:
            return pool

        pbo_value = None
        try:
            pbo_result = compute_pbo(
                self.df, [r.spec for r in pool], self.risk,
                n_groups=self.cfg.cpcv_n_groups, n_test_groups=self.cfg.cpcv_n_test_groups,
                metric=self.cfg.cpcv_metric, max_paths=self.cfg.cpcv_max_paths,
                prop_rules=self.prop_rules,
            )
            pbo_value = pbo_result.pbo
        except Exception:
            pass

        for r in pool:
            if self._stop_flag.is_set():
                break
            cpcv_degradation = None
            oos_metric = None
            try:
                cpcv_result = run_cpcv(
                    self.df, lambda spec=r.spec: build_strategy_from_spec(spec), self.risk,
                    n_groups=self.cfg.cpcv_n_groups, n_test_groups=self.cfg.cpcv_n_test_groups,
                    metric=self.cfg.cpcv_metric, max_paths=self.cfg.cpcv_max_paths,
                    prop_rules=self.prop_rules,
                )
                cpcv_degradation = cpcv_result.degradation
                if self.cfg.cpcv_metric == "eval_pass_probability":
                    oos_metric = cpcv_result.mean_oos_metric
            except CPCVError:
                pass
            except Exception:
                pass
            r.pbo = pbo_value
            r.cpcv_degradation = cpcv_degradation
            r.cpcv_oos_eval_pass_probability = oos_metric
            raw_pct = (r.mc_summary or {}).get("evaluation_pass_probability")
            if oos_metric is not None and raw_pct is not None:
                flag = " <-- big gap, likely overfit; verify with Full Pipeline before treating as real" \
                    if raw_pct - oos_metric > 15 else ""
                self._log(
                    f"  CPCV [{r.candidate_id}]: raw in-sample eval pass {raw_pct:.1f}% -> "
                    f"out-of-sample estimate {oos_metric:.1f}%{flag}"
                )
            r.fitness = compute_prop_fitness(
                r.stats, r.mc_summary, r.robustness, r.walk_forward, r.trade_pnls,
                pbo=pbo_value, cpcv_degradation=cpcv_degradation,
                min_trades_target=self.cfg.min_trades_target_for_fitness,
                weights=self.cfg.fitness_goal,
            )
        return pool

    # -- STRESS TEST ---------------------------------------------------------
    def _stress_test(self, pool: list[EvolutionCandidateRecord]) -> list[EvolutionCandidateRecord]:
        stressed_risk = _stressed_risk_config(self.risk, self.cfg.stress_cost_multiplier)
        survivors = []
        for r in pool:
            if self._stop_flag.is_set():
                break
            try:
                strategy = build_strategy_from_spec(r.spec)
                bt = run_backtest(self.df, strategy, stressed_risk, adaptive_risk=self.adaptive_risk)
                r.stressed_ok = bool(bt.trades) and bt.statistics.net_profit > 0
            except Exception:
                r.stressed_ok = False
            if r.stressed_ok:
                survivors.append(r)
        return survivors

    # -- CLUSTER -------------------------------------------------------------
    def _cluster(self, pool: list[EvolutionCandidateRecord]) -> list[EvolutionCandidateRecord]:
        """Correlation-dedupe on DAILY P&L (date-aligned, zero-filled on
        days either strategy didn't trade) -- comparing raw trade-by-trade
        pnl lists positionally would be meaningless since two strategies
        rarely have the same number of trades on the same days."""
        ranked = sorted(pool, key=lambda r: r.fitness.final_score, reverse=True)
        kept: list[EvolutionCandidateRecord] = []
        kept_series: list[pd.Series] = []
        for r in ranked:
            series = _daily_pnl_series(r.trades)
            is_duplicate = False
            for other in kept_series:
                if series.empty or other.empty:
                    continue
                idx = series.index.union(other.index)
                a = series.reindex(idx, fill_value=0.0)
                b = other.reindex(idx, fill_value=0.0)
                if len(idx) < 5 or a.std() == 0 or b.std() == 0:
                    continue
                corr = a.corr(b)
                if corr is not None and corr >= self.cfg.cluster_correlation_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(r)
                kept_series.append(series)
        return kept

    # -- knowledge graph / leaderboard / library / journal -------------------
    def _record_generation_to_knowledge_graph(self, evaluated: list[EvolutionCandidateRecord], elite_ids: set) -> None:
        for r in evaluated:
            fv = feature_vector_for_spec(r.spec, r.meta)
            outcome = {
                "passed": r.candidate_id in elite_ids,
                "final_score": r.fitness.final_score if r.fitness else None,
                "generation": self.generation,
            }
            try:
                self.knowledge_graph.record(fv, outcome)
            except Exception:
                pass

    def _update_leaderboard(self, new_elites: list[EvolutionCandidateRecord]) -> None:
        combined = {r.candidate_id: r for r in (self.leaderboard + new_elites)}
        ranked = sorted(combined.values(), key=lambda r: r.fitness.final_score, reverse=True)
        self.leaderboard = ranked[: self.cfg.elite_keep]

    def _maybe_save_to_library(self, new_elites: list[EvolutionCandidateRecord]) -> None:
        if not self.cfg.save_to_library:
            return
        for r in new_elites:
            config = r.spec.get("config")
            if not config:
                continue
            import json
            text = json.dumps(config, indent=2)
            base_name = f"evolab_gen{self.generation}_{r.meta.get('family', 'strategy')}_{r.candidate_id[-6:]}"
            filename = f"{base_name}.json"
            try:
                try:
                    save_strategy_text(text, filename, "manual", overwrite=False)
                except StrategyAlreadyExists:
                    continue
                set_strategy_status("manual", filename, self.cfg.library_status)
                # Tagged + given a lab-stats sidecar so this appears in the
                # Strategy Library / Dashboard as a distinguishable
                # Evolution Lab result, filterable by the existing tag
                # filter, with fitness/MC/robustness context attached --
                # not just another anonymous "manual" strategy file that
                # has to be found and read to know where it came from.
                save_strategy_metadata(
                    "manual", filename,
                    {
                        "tags": ["evolution-lab"],
                        "description": (
                            f"Evolution Lab, generation {self.generation}, family "
                            f"'{r.meta.get('family', '?')}' -- PROP FITNESS "
                            f"{r.fitness.final_score:.2f}" if r.fitness else
                            f"Evolution Lab, generation {self.generation}, family '{r.meta.get('family', '?')}'"
                        ),
                        "evolution": evolution_stats_metadata(r.to_checkpoint_dict(), generation=self.generation),
                    },
                    merge=True,
                )
            except Exception:
                pass

    def _write_journal_entry(self, gen, population, stage1_survivors, evaluated, cpcv_pool, stress_survivors, new_elites) -> None:
        n = len(self.journal) + 1
        winner = new_elites[0] if new_elites else None
        confidence = "LOW"
        if winner is not None and winner.fitness and winner.fitness.final_score > 0:
            n_similar_records = self.knowledge_graph.query_similar(feature_vector_for_spec(winner.spec, winner.meta), top_k=20)
            n_similar = sum(1 for s, _ in n_similar_records if s >= 0.6)
            stable = bool(winner.robustness and winner.robustness.get("is_stable"))
            if stable and n_similar >= 5:
                confidence = "HIGH"
            elif stable or n_similar >= 2:
                confidence = "MEDIUM"

        lines = [
            f"HYPOTHESIS #{n} (generation {gen})",
            "",
            f"TEST: {len(population)} strategy variants",
            f"RESULT: {len(stage1_survivors)} survived initial screening",
            f"OOS/ROBUSTNESS: {len(evaluated)} survived",
            f"CPCV: {len(cpcv_pool)} re-scored",
            f"STRESS: {len(stress_survivors)} survived {self.cfg.stress_cost_multiplier:g}x costs",
        ]
        if winner is not None:
            lines.append(f"WINNER: {winner.candidate_id}  (PROP FITNESS {winner.fitness.final_score:.2f})")
            lines.append(f"CONFIDENCE: {confidence}")
            lines.append("")
            lines.append(self.knowledge_graph.describe(feature_vector_for_spec(winner.spec, winner.meta)))
        else:
            # FIX (2026-09-11): previously unreachable in practice --
            # new_elites used to always contain the best PRE-stress
            # candidate even when stress_survivors was empty (see
            # _run_one_generation's old `stress_survivors or cpcv_pool`
            # fallback), so this branch almost never fired even on a run
            # where every single generation's "winner" had actually failed
            # the stress test. Now that new_elites is only ever populated
            # from genuine stress survivors, this is the honest, common
            # case for a hard search -- shown with the closest near-miss
            # for context instead of a bare "none," so the journal still
            # tells a story (see app.search.graveyard for the full
            # per-candidate failure detail behind this near-miss).
            lines.append("WINNER: none this generation (no candidate survived the stress test)")
            lines.append("CONFIDENCE: LOW")
            near_miss = sorted(cpcv_pool, key=lambda r: r.fitness.final_score, reverse=True)[:1]
            if near_miss:
                nm = near_miss[0]
                lines.append("")
                lines.append(
                    f"Closest near-miss: {nm.candidate_id} (PROP FITNESS {nm.fitness.final_score:.2f}) -- "
                    f"failed the stress test at {self.cfg.stress_cost_multiplier:g}x costs; see the Strategy "
                    f"Graveyard for full detail on why."
                )
        entry = "\n".join(lines)
        self.journal.append(entry)
        self._log("\n" + entry + "\n")

    # -- adaptive family budget bookkeeping -----------------------------------
    def _record_family_budget_prefilter(self, population: list, stage1_survivors: list) -> None:
        """Tallies this generation's per-family tested/prefilter-passed
        counts and pre-registers them with the family budget tracker --
        finished off by _record_family_budget_stress once the stress
        stage has also run, so both halves of one generation's counts
        land in the tracker together (see app.evolution.family_budget)."""
        if not self.cfg.adaptive_family_budget_enabled:
            return
        tested: dict[str, int] = {}
        for _cid, _spec, meta in population:
            fam = meta.get("family", "?")
            tested[fam] = tested.get(fam, 0) + 1
        passed: dict[str, int] = {}
        for item in stage1_survivors:
            meta = item[2]  # (cid, spec, meta, bt)
            fam = meta.get("family", "?")
            passed[fam] = passed.get(fam, 0) + 1
        self._pending_family_counts = {
            fam: {"tested": n, "prefilter_passed": passed.get(fam, 0), "stress_passed": 0}
            for fam, n in tested.items()
        }

    def _record_family_budget_stress(self, cpcv_pool: list, stress_survivors: list) -> None:
        if not self.cfg.adaptive_family_budget_enabled:
            return
        pending = getattr(self, "_pending_family_counts", None)
        if pending is None:
            return
        survivor_ids = {id(r) for r in stress_survivors}
        for r in cpcv_pool:
            fam = (r.meta or {}).get("family", "?")
            if fam not in pending:
                pending[fam] = {"tested": 0, "prefilter_passed": 0, "stress_passed": 0}
            if id(r) in survivor_ids:
                pending[fam]["stress_passed"] += 1
        self._family_budget.record_generation(pending)
        self._pending_family_counts = None

    # -- strategy graveyard ----------------------------------------------------
    def _write_graveyard_entries(self, records: list, gen: int, stage_died: str) -> None:
        """Builds and persists one GraveyardEntry per record that reached
        full evaluation and CPCV but died at `stage_died` -- see
        app.search.graveyard's module docstring for why this is separate
        from the cheap prefilter rejection log."""
        entries = []
        for r in records:
            fam = (r.meta or {}).get("family", "?")
            config = (r.spec or {}).get("config")
            robustness = r.robustness or {}
            stability_pct = None
            if robustness.get("stability_ratio") is not None:
                stability_pct = max(0.0, min(1.0, float(robustness["stability_ratio"]))) * 100.0
            oos_result = None
            if r.walk_forward is not None:
                eff = r.walk_forward.get("walk_forward_efficiency")
                if eff is not None:
                    oos_result = "negative" if eff < 0 else "positive"
            raw_pass_pct = (r.mc_summary or {}).get("evaluation_pass_probability")
            mc_failure_pct = (100.0 - raw_pass_pct) if raw_pass_pct is not None else None

            reason = (r.fitness.notes[0] if (r.fitness and r.fitness.notes) else None) or (
                f"Failed the stress test at {self.cfg.stress_cost_multiplier:g}x execution costs "
                "-- net profit went negative once realistic costs were applied."
            )
            entries.append(GraveyardEntry(
                candidate_id=r.candidate_id, family=fam, generation=gen, stage_died=stage_died,
                reason=reason, oos_result=oos_result,
                neighbor_robustness_pct=stability_pct,
                monte_carlo_failure_pct=mc_failure_pct,
                prop_sim_pass_pct=raw_pass_pct,
                cpcv_oos_pass_pct=r.cpcv_oos_eval_pass_probability,
                pbo=r.pbo,
                fitness_score=r.fitness.final_score if r.fitness else None,
                param_signature=param_signature(fam, config),
                notes=list(r.fitness.notes) if r.fitness else [],
            ))
        try:
            record_rejections(entries, Path(self.cfg.tested_log_path).with_name("strategy_graveyard.jsonl"))
        except Exception:  # noqa: BLE001 -- graveyard logging is diagnostic, never allowed to break a run
            pass
