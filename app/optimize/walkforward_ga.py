"""
Walk-forward-aware genetic algorithm.

Iterative Refinement's GA (app.optimize.refinement) scores every
candidate genome by backtesting it on the WHOLE dataset and computing
fitness from that single in-sample run. That means the GA is free to
evolve toward whatever squeezes the most fitness out of that one
historical window -- exactly the overfitting failure mode this app has
hit for real, more than once, via lookahead bugs that inflated in-sample
numbers (see the champion-selection history in this project). A GA is a
particularly efficient way to find and exploit noise, precisely because
it tries so many variations.

This module runs the identical GA operators (crossover, mutation,
tournament selection, elitism, random immigrants -- all imported directly
from app.optimize.refinement so the two never drift apart) but scores
each genome differently: it splits the data into several chronological
folds (see app.validation.walk_forward_opt.build_folds), and a genome's
fitness is computed ONLY from backtesting that SAME fixed genome on each
fold's held-out test slice and chaining the results -- never from the
training slices, and never from the full dataset. A genome that only
works on one specific historical stretch, rather than generalizing
across several distinct ones, will simply score lower here and get
selected against -- which is the entire point.

This directly answers "stop the optimizer from just curve-fitting
harder": the optimizer can still evolve toward whatever works, but
"works" is now defined as "keeps working on data it never got to fit
against," across every generation, not just at a final holdout check
tacked on at the end.
"""
from __future__ import annotations

import inspect
import math
import multiprocessing
import os
import random
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np

from app.backtest.engine import run_backtest
from app.backtest.execution import Trade
from app.backtest.risk import RiskConfig
from app.backtest.statistics import compute_statistics
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.monte_carlo.slippage_model import SessionVolatilitySlippageConfig
from app.optimize.code_parameter_space import materialize_code_strategy
from app.optimize.parameter_space import GeneMeta, RefinementError, apply_genome
from app.orchestration.resource_guard import safe_worker_count_for_bytes
from app.optimize.refinement import (
    RefinementConfig,
    _build_adapter,
    _crossover,
    _mutate,
    _random_gene_value,
    _stressed_risk_config,
    _tournament_select,
    apply_cost_stress_penalty,
    compute_fitness,
    preflight_signal_check,
)
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.strategy.base import Strategy
from app.strategy.manual import ManualStrategy
from app.validation.walk_forward_opt import build_folds

ProgressCallback = Callable[[str], None]

# Below this many total genome evaluations (population_size * (generations
# + 1)), the fixed cost of spinning up worker processes (spawn, not fork --
# see the "why spawn" note further down) outweighs any benefit, so small/
# test-scale runs stay on a single process exactly as before this
# parallelization existed.
_MIN_EVALUATIONS_FOR_PARALLEL = 16


class _CodeStrategyShim:
    """A minimal stand-in for a real Strategy object, carrying only what
    materialize_code_strategy() actually reads (source_type, plus either
    file_path or code) -- used inside a worker process, which never has
    the original live Strategy instance, only these few picklable
    attributes copied out of it up front."""
    __slots__ = ("source_type", "file_path", "code")

    def __init__(self, source_type: str, file_path=None, code=None):
        self.source_type = source_type
        self.file_path = file_path
        self.code = code


def _chained_fitness(
    strategy_to_run, slices: list[pd.DataFrame], risk_to_use: RiskConfig,
    prop_rules: PropRules, mc_cfg: MonteCarloConfig, fitness_metric: str,
    adaptive_risk=None, risk_of_ruin_cap: float | None = None,
) -> tuple[float, int]:
    """Backtests `strategy_to_run` (a fixed, already-built genome) on every
    fold's held-out test slice, chains the resulting trades into one
    equity curve, and scores that with the same fitness metric everywhere
    else in this app uses. Module-level (not a closure) so it can be
    called identically from the main process (serial path) or from inside
    a worker process (parallel path) with no duplicated logic.

    risk_of_ruin_cap: forwarded straight to compute_fitness -- see
    app.optimize.refinement._apply_ruin_penalty. None (default) means no
    penalty, byte-identical to before this parameter existed."""
    all_trades: list[Trade] = []
    for test_df in slices:
        bt = run_backtest(test_df, strategy_to_run, risk_to_use, adaptive_risk=adaptive_risk)
        all_trades.extend(bt.trades)
    if not all_trades:
        return float("-inf"), 0
    equity = risk_to_use.initial_balance
    rows = []
    ordered = sorted(all_trades, key=lambda t: t.exit_time)
    for t in ordered:
        equity += t.pnl if math.isfinite(t.pnl) else 0.0
        rows.append({"timestamp": t.exit_time, "equity": equity})
    equity_curve = pd.DataFrame(rows)
    stats = compute_statistics(all_trades, equity_curve, initial_balance=risk_to_use.initial_balance)
    pnls = [t.pnl for t in all_trades]
    dates = [t.entry_time for t in all_trades]
    single_run = simulate_account(pnls, dates, prop_rules, reset_on_breach=mc_cfg.reset_on_breach)
    mc = run_monte_carlo(all_trades, prop_rules, mc_cfg)
    prop_summary = summarize_single_run(single_run)
    fitness = compute_fitness(stats.to_dict(), prop_summary, mc, fitness_metric, risk_of_ruin_cap=risk_of_ruin_cap)
    return (fitness if math.isfinite(fitness) else float("-inf")), len(all_trades)


# ---------------------------------------------------------------------------
# Worker process state & task (module-level so ProcessPoolExecutor can
# pickle/import them; state is per-process, populated once by _ga_worker_init
# -- mirrors app.search.batch_runner's proven "load shared state once at
# pool startup, workers return only small picklable results" pattern).
# ---------------------------------------------------------------------------

_GA_WORKER: dict = {}


def _ga_worker_init(
    kind: str, manual_config: dict | None, code_shim: "_CodeStrategyShim | None",
    genes: list, test_slice_paths: list[str], risk_kwargs: dict, prop_kwargs: dict,
    tmp_dir_path: str, search_mc_kwargs: dict, fitness_metric: str,
    cost_stress_enabled: bool, cost_stress_multiplier: float, cost_stress_penalty_weight: float,
    adaptive_risk=None, risk_of_ruin_cap: float | None = None,
) -> None:
    global _GA_WORKER
    # BUG FIX (2026-09): dataclasses.asdict() -- used at the pool-creation
    # call site to make MonteCarloConfig picklable as plain kwargs --
    # recurses into EVERY nested dataclass field, not just the top-level
    # one. MonteCarloConfig.session_slippage is itself a dataclass
    # (SessionVolatilitySlippageConfig), so search_mc_kwargs["session_
    # slippage"] arrives here as a plain dict, not an object. Reconstructing
    # MonteCarloConfig(**search_mc_kwargs) without first rebuilding that
    # nested field left session_slippage as a bare dict on the resulting
    # MonteCarloConfig -- which then raised
    # `AttributeError: 'dict' object has no attribute 'enabled'` the
    # instant a genome's fitness evaluation reached
    # apply_session_volatility_slippage(trades, cfg.session_slippage)
    # inside run_monte_carlo, on EVERY worker, for EVERY genome, every
    # generation. Because this is caught by evaluate_batch's blanket
    # except and silently falls back to a single process, this never
    # produced a wrong number -- it just meant this pool never actually
    # ran a single genome in parallel, silently paying full sequential
    # cost on every Quick Optimize / Full Pipeline run that used this GA
    # (i.e. every one of them, since both share this exact function).
    search_mc_kwargs = dict(search_mc_kwargs)
    session_slippage_kwargs = search_mc_kwargs.get("session_slippage")
    if isinstance(session_slippage_kwargs, dict):
        search_mc_kwargs["session_slippage"] = SessionVolatilitySlippageConfig(**session_slippage_kwargs)
    _GA_WORKER = {
        "kind": kind,
        "manual_config": manual_config,
        "code_shim": code_shim,
        "genes": genes,
        "test_slices": [pd.read_pickle(p) for p in test_slice_paths],
        "risk": RiskConfig(**risk_kwargs),
        "prop_rules": PropRules(**prop_kwargs),
        "tmp_dir": Path(tmp_dir_path),
        "search_mc_cfg": MonteCarloConfig(**search_mc_kwargs),
        "fitness_metric": fitness_metric,
        "cost_stress_enabled": cost_stress_enabled,
        "cost_stress_multiplier": cost_stress_multiplier,
        "cost_stress_penalty_weight": cost_stress_penalty_weight,
        "adaptive_risk": adaptive_risk,
        "risk_of_ruin_cap": risk_of_ruin_cap,
    }


def _ga_worker_build(genome: list):
    w = _GA_WORKER
    if w["kind"] == "manual":
        return ManualStrategy(apply_genome(w["manual_config"], w["genes"], genome))
    return materialize_code_strategy(w["code_shim"], w["genes"], genome, w["tmp_dir"])


def _ga_eval_task(genome: list) -> tuple[float, int]:
    """Runs in a worker process: builds the strategy for this one genome
    and scores it exactly like the serial oos_fitness() below does."""
    w = _GA_WORKER
    strategy = _ga_worker_build(genome)
    fitness, trade_count = _chained_fitness(
        strategy, w["test_slices"], w["risk"], w["prop_rules"], w["search_mc_cfg"], w["fitness_metric"],
        w.get("adaptive_risk"), w.get("risk_of_ruin_cap"),
    )
    if w["cost_stress_enabled"] and w["cost_stress_penalty_weight"] > 0 and math.isfinite(fitness):
        stressed_risk = _stressed_risk_config(w["risk"], w["cost_stress_multiplier"])
        stressed_fitness, _ = _chained_fitness(
            strategy, w["test_slices"], stressed_risk, w["prop_rules"], w["search_mc_cfg"], w["fitness_metric"],
            w.get("adaptive_risk"), w.get("risk_of_ruin_cap"),
        )
        fitness = apply_cost_stress_penalty(fitness, stressed_fitness, w["cost_stress_penalty_weight"])
    return fitness, trade_count


@dataclass
class WalkforwardGACandidate:
    genome: list
    fitness: float
    oos_trade_count: int
    in_sample_fitness: float | None = None  # same genome's fitness on the FULL df, for an overfitting-gap readout
    config: dict | None = None
    code_text: str | None = None
    code_extension: str | None = None


@dataclass
class WalkforwardGAGenerationSummary:
    generation: int
    best_fitness: float
    mean_fitness: float


@dataclass
class WalkforwardGAResult:
    refinement_config: RefinementConfig
    n_folds: int
    window_mode: str
    genes: list
    best: WalkforwardGACandidate
    generation_history: list  # list[WalkforwardGAGenerationSummary]
    leaderboard: list  # list[WalkforwardGACandidate], final generation sorted best-first
    overfitting_gap: float | None  # best.in_sample_fitness - best.fitness; large positive = curve-fit, likely to disappoint live
    elapsed_seconds: float
    warnings: list = field(default_factory=list)
    # VAL-005: 0-based, exclusive-end bar position (in the `df` this GA
    # was run against) of the furthest-forward bar any of its fold TEST
    # windows touched -- None if no folds were built. See
    # app.search.robustness.run_walk_forward's embargo_start_bar.
    max_test_bar_used: int | None = None
    # UPGRADE (optimizer core): evaluation-count transparency -- see
    # app.optimize.refinement.RefinementResult's identical field.
    total_evaluations: int = 0


class WalkforwardGACancelled(Exception):
    """Raised out of run_walkforward_aware_refinement when a caller-
    supplied cancel_event is set -- checked once per generation, so a
    Stop click takes effect at the next generation boundary rather than
    instantly. Used by Quick Optimize's web job (previously had no way to
    stop a run in progress at all)."""


def run_walkforward_aware_refinement(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    refinement_config: RefinementConfig | None = None,
    n_folds: int = 4,
    window_mode: str = "rolling",
    train_frac: float = 0.6,
    progress_cb: ProgressCallback | None = None,
    ai_suggest_cb: Callable[[list], list[list[float]]] | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    adaptive_risk=None,
    cancel_event: "threading.Event | None" = None,
) -> WalkforwardGAResult:
    """
    parallel: when True (the default) and the search is large enough to be
    worth it (see _MIN_EVALUATIONS_FOR_PARALLEL), each generation's genome
    evaluations run across multiple worker processes instead of one at a
    time in the calling process -- this is normally THE dominant cost of
    a Full Pipeline run, so this can cut a multi-minute search down
    substantially on a multi-core machine. Genome GENERATION (which
    genomes get tried) always stays single-process and deterministic;
    only backtesting/scoring an already-decided genome is parallelized, so
    results are the same as the fully-serial path modulo which CPU
    happened to run which candidate. Any failure to start the worker pool
    falls back to the single-process path automatically. max_workers caps
    how many processes are used (default: up to os.cpu_count()).

    ai_suggest_cb: optional, called with the strategy's discovered `genes`
    list once per generation (including generation 0's initial population)
    when the optional AI assistant (see app.ai.ollama_client) is enabled.
    Returns a list of already-clamped-and-validated genomes to inject into
    that generation's population, replacing some of what would otherwise
    be random immigrants/offspring -- never an elite, so a bad AI
    suggestion can never displace a genuinely better candidate, only
    compete for the non-elite slots on equal footing. Any exception the
    callback raises, or an empty list, is treated exactly like AI assist
    being off: the generation proceeds with its normal random/bred
    population, unchanged.

    Callbacks may optionally accept a SECOND argument: the prior
    generation's already-evaluated population as `[(genome, fitness), ...]`
    (an empty list for generation 0, before anything has been evaluated).
    This is the systematic "Stage 4 analysis and feedback" hook from the
    quant loop framework -- see app.optimize.gene_fitness_analysis, which
    a two-argument callback can run itself (pure statistics, no extra
    backtests, no AI call) to tell an AI assistant which parameter regions
    are already known to score well or badly before asking it for new
    candidates. Detected via the callback's signature so existing
    single-argument callbacks (including every one in this app's own test
    suite) keep working unchanged.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    _ai_cb_wants_population = False
    if ai_suggest_cb is not None:
        try:
            params = list(inspect.signature(ai_suggest_cb).parameters.values())
            _ai_cb_wants_population = len(params) >= 2 or any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
            )
        except (TypeError, ValueError):
            _ai_cb_wants_population = False

    def ai_genomes(genes_for_cb: list, population_for_cb: list | None = None) -> list[list[float]]:
        if ai_suggest_cb is None:
            return []
        try:
            if _ai_cb_wants_population:
                return ai_suggest_cb(genes_for_cb, population_for_cb or []) or []
            return ai_suggest_cb(genes_for_cb) or []
        except Exception:
            return []

    cfg = refinement_config or RefinementConfig(population_size=12, generations=6, search_monte_carlo_sims=200)
    t0 = time.time()
    warnings: list[str] = []
    # Evaluation-count transparency (see app.optimize.refinement's
    # identical RefinementResult.total_evaluations) -- a single-element
    # list rather than a plain int so evaluate_batch's closure below can
    # mutate it without a `nonlocal` declaration.
    total_evaluations = [0]

    folds = build_folds(df, n_folds=n_folds, window_mode=window_mode, train_frac=train_frac, warnings=warnings)
    if not folds:
        raise RefinementError(
            "Not enough bars to build the requested number of walk-forward folds for the GA."
        )
    if len(folds) < n_folds:
        # VAL-005: build_folds already appended the per-fold reason(s) to
        # `warnings` above -- this adds the summary a person actually
        # scans for: "the chained-OOS fitness below reflects fewer folds
        # than you configured."
        warnings.append(
            f"Only {len(folds)} of the {n_folds} requested walk-forward fold(s) were usable -- "
            "the chained out-of-sample fitness, generation-history log, and overfitting_gap "
            "below were all computed from those folds only."
        )
    test_slices = [f.test_df for f in folds]
    # VAL-005: the furthest-forward bar (0-based, exclusive end) any of
    # this GA's fold TEST windows touched in the original `df` -- Full
    # Pipeline's Step 4 out-of-sample check uses this to embargo its own
    # fold construction so it doesn't re-test bars this search already
    # selected the winning genome against. See
    # app.search.robustness.run_walk_forward's embargo_start_bar.
    max_test_bar_used = max((f.test_end_bar for f in folds), default=None)

    preflight_signal_check(df, strategy, risk, "Walk-Forward-Aware GA")

    tmp_dir: Path | None = None
    if strategy.source_type == "python":
        tmp_dir = Path(tempfile.mkdtemp(prefix="t58_wfga_"))

    try:
        genes, build = _build_adapter(strategy, tmp_dir)
        if not genes:
            raise RefinementError(
                "This strategy has no tunable numeric parameters -- there is nothing "
                "for a walk-forward-aware GA to search over."
            )

        rng = random.Random(cfg.random_seed)
        search_mc_cfg = replace(mc_config, n_simulations=cfg.search_monte_carlo_sims)

        def oos_fitness(genome: list) -> tuple[float, int]:
            """Fitness computed ONLY from chaining every fold's held-out test slice,
            cost-stress-adjusted the same way Iterative Refinement's plain GA is
            (see app.optimize.refinement.apply_cost_stress_penalty) -- a genome that
            only survives on nominal costs AND only on one historical stretch is
            exactly the double-overfitting failure mode this module plus that
            adjustment together are meant to catch."""
            candidate_strategy = build(genome)
            fitness, trade_count = _chained_fitness(
                candidate_strategy, test_slices, risk, prop_rules, search_mc_cfg, cfg.fitness_metric,
                adaptive_risk, cfg.risk_of_ruin_cap,
            )
            if cfg.cost_stress_enabled and cfg.cost_stress_penalty_weight > 0 and math.isfinite(fitness):
                stressed_risk = _stressed_risk_config(risk, cfg.cost_stress_multiplier)
                stressed_fitness, _ = _chained_fitness(
                    candidate_strategy, test_slices, stressed_risk, prop_rules, search_mc_cfg, cfg.fitness_metric,
                    adaptive_risk, cfg.risk_of_ruin_cap,
                )
                fitness = apply_cost_stress_penalty(fitness, stressed_fitness, cfg.cost_stress_penalty_weight)
            return fitness, trade_count

        def full_df_fitness(genome: list) -> float:
            candidate_strategy = build(genome)
            bt = run_backtest(df, candidate_strategy, risk, adaptive_risk=adaptive_risk)
            if not bt.trades:
                return float("-inf")
            pnls = [t.pnl for t in bt.trades]
            dates = [t.entry_time for t in bt.trades]
            single_run = simulate_account(pnls, dates, prop_rules, reset_on_breach=search_mc_cfg.reset_on_breach)
            mc = run_monte_carlo(bt.trades, prop_rules, search_mc_cfg)
            fitness = compute_fitness(
                bt.statistics.to_dict(), summarize_single_run(single_run), mc, cfg.fitness_metric,
                risk_of_ruin_cap=cfg.risk_of_ruin_cap,
            )
            return fitness if math.isfinite(fitness) else float("-inf")

        class _Cand:
            __slots__ = ("genome", "fitness", "trade_count")

            def __init__(self, genome, fitness, trade_count):
                self.genome = genome
                self.fitness = fitness
                self.trade_count = trade_count

        def make(genome: list) -> "_Cand":
            fitness, tc = oos_fitness(genome)
            return _Cand(genome, fitness, tc)

        # -- Optional cross-process parallel evaluation -----------------
        # Every genome's fitness is independent of every other genome's --
        # only the GA operators that DECIDE which genomes to try
        # (tournament selection, crossover, mutation, random immigrants)
        # need a fixed sequence against `rng`. So genome GENERATION always
        # stays serial and deterministic below (the exact same genomes,
        # in the exact same order, as the fully-serial version), and only
        # the potentially expensive per-genome backtest+Monte-Carlo
        # evaluation is farmed out across worker processes, mirroring
        # app.search.batch_runner's already-proven ProcessPoolExecutor
        # pattern. Falls back to the single-process path on any setup
        # failure (pickling issue, no spare cores, etc.) so this can never
        # turn a working run into a broken one.
        estimated_evaluations = cfg.population_size * (cfg.generations + 1)
        cpu_count = os.cpu_count() or 1
        pool: ProcessPoolExecutor | None = None
        pool_tmp_dir: Path | None = None
        use_parallel = bool(parallel) and estimated_evaluations >= _MIN_EVALUATIONS_FOR_PARALLEL and cpu_count > 1
        if use_parallel:
            try:
                from dataclasses import asdict

                pool_tmp_dir = Path(tempfile.mkdtemp(prefix="t58_wfga_pool_"))
                slice_paths = []
                for i, test_df in enumerate(test_slices):
                    p = pool_tmp_dir / f"fold_{i}.pkl"
                    test_df.to_pickle(p)
                    slice_paths.append(str(p))

                if strategy.source_type == "manual":
                    worker_kind, manual_cfg, code_shim = "manual", strategy.config, None
                else:
                    worker_kind, manual_cfg = strategy.source_type, None
                    code_shim = _CodeStrategyShim(
                        strategy.source_type,
                        file_path=getattr(strategy, "file_path", None),
                        code=getattr(strategy, "code", None),
                    )

                n_workers = max(1, min(max_workers or cpu_count, cpu_count, cfg.population_size))
                # Every worker's initializer loads ALL of slice_paths (every
                # walk-forward fold), not just one -- so the right memory
                # estimate for safe_worker_count_for_bytes is the SUM across
                # all folds, not any single fold. Without this, this pool
                # used raw os.cpu_count() workers regardless of dataset size
                # or available memory, which is what was behind the
                # "unable to allocate" MemoryErrors and BrokenProcessPool
                # crashes seen in Quick Optimize and Full Pipeline's GA
                # stage on large (multi-year, 1-minute) datasets -- see
                # app.orchestration.resource_guard for the full rationale.
                try:
                    total_fold_bytes = sum(float(s.memory_usage(deep=True).sum()) for s in test_slices)
                except Exception:
                    total_fold_bytes = 0.0
                n_workers = safe_worker_count_for_bytes(
                    total_fold_bytes, requested=n_workers, max_candidates_in_flight=cfg.population_size,
                )
                pool = ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_ga_worker_init,
                    initargs=(
                        worker_kind, manual_cfg, code_shim, genes, slice_paths,
                        asdict(risk), asdict(prop_rules), str(pool_tmp_dir),
                        asdict(search_mc_cfg), cfg.fitness_metric,
                        cfg.cost_stress_enabled, cfg.cost_stress_multiplier, cfg.cost_stress_penalty_weight,
                        adaptive_risk, cfg.risk_of_ruin_cap,
                    ),
                    # Runs from a background thread in real usage (the
                    # desktop GUI's Full Pipeline / Walk-Forward-Aware GA
                    # tabs, the web app's search job) -- see
                    # app.search.batch_runner's identical note for why
                    # spawn is used instead of the platform-default fork.
                    mp_context=multiprocessing.get_context("spawn"),
                )
                log(f"Parallel search enabled across {n_workers} worker process(es).")
            except Exception as exc:  # noqa: BLE001 -- must never break the GA
                log(f"Parallel search unavailable ({exc}) -- continuing on a single process.")
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
                pool = None

        def evaluate_batch(genome_list: list[list[float]]) -> list["_Cand"]:
            if not genome_list:
                return []
            if pool is not None:
                try:
                    futures = {pool.submit(_ga_eval_task, g): i for i, g in enumerate(genome_list)}
                    results: list = [None] * len(genome_list)
                    for fut in as_completed(futures):
                        i = futures[fut]
                        fitness, tc = fut.result()
                        results[i] = _Cand(genome_list[i], fitness, tc)
                    total_evaluations[0] += len(results)
                    return results
                except Exception as exc:  # noqa: BLE001 -- fall back to serial for this batch
                    log(f"  Parallel evaluation failed ({exc}) -- falling back to a single process.")
            results = [make(g) for g in genome_list]
            total_evaluations[0] += len(results)
            return results

        try:
            log(f"Analyzing strategy parameters... found {len(genes)} tunable parameter(s). "
                f"Fitness will be scored on {len(test_slices)} chained out-of-sample fold(s).")

            baseline = make([g.base_value for g in genes])
            total_evaluations[0] += 1

            # UPGRADE (auto-shrink on low trade count): a wider search (more
            # population/generations) tries more candidates, which directly
            # raises the Bonferroni-corrected significance bar every
            # candidate has to clear afterward (see app.validation.icir,
            # and FullPipelineConfig/QuickOptimizeConfig's n_candidates_
            # tested calculations) -- punishing exactly the naturally-
            # selective strategy families whose chained-OOS trade count is
            # already thin. The baseline's own trade_count (computed above,
            # independent of how many generations run) is the cheapest
            # possible signal for "does this data/strategy combination
            # support a search this wide": if it's below
            # min_oos_trades_per_candidate, generations are reduced
            # proportionally BEFORE any generation actually runs, for every
            # optimizer_mode (genetic/tpe/cma_es all read cfg.generations),
            # rather than searching wide first and finding out only at the
            # significance gate that it never had enough data to justify
            # that many trials.
            if (
                cfg.auto_shrink_on_low_trades
                and cfg.generations > 1
                and 0 < baseline.trade_count < cfg.min_oos_trades_per_candidate
            ):
                shrink_ratio = baseline.trade_count / cfg.min_oos_trades_per_candidate
                shrunk_generations = max(1, math.ceil(cfg.generations * shrink_ratio))
                if shrunk_generations < cfg.generations:
                    auto_shrink_note = (
                        f"Auto-shrink: the baseline configuration produced only "
                        f"{baseline.trade_count} chained out-of-sample trade(s) across "
                        f"{len(test_slices)} fold(s) (want {cfg.min_oos_trades_per_candidate}+ to "
                        f"justify a search this wide) -- reduced generations from {cfg.generations} "
                        f"to {shrunk_generations} so this search doesn't inflate the number of "
                        "candidates tried (and therefore the Bonferroni-corrected significance bar "
                        "every candidate has to clear -- see app.validation.icir) beyond what this "
                        "much data can actually support. Add more historical data, use a higher-"
                        "frequency signal, or set RefinementConfig.auto_shrink_on_low_trades=False "
                        "to search the full configured width anyway."
                    )
                    log(f"  {auto_shrink_note}")
                    warnings.append(auto_shrink_note)
                    cfg = replace(cfg, generations=shrunk_generations)

            if cfg.optimizer_mode == "genetic":
                population = [baseline]
                # Generation 0: nothing evaluated yet, so the population snapshot
                # a two-argument callback receives is empty -- it has only the
                # gene definitions to work with, same as before this hook existed.
                ai_seed_genomes = [g for g in ai_genomes(genes, []) if len(g) == len(genes)]
                ai_seed_genomes = ai_seed_genomes[: max(0, cfg.population_size - len(population))]
                if ai_seed_genomes:
                    population.extend(evaluate_batch(ai_seed_genomes))
                    log(f"AI assist: seeded {len(ai_seed_genomes)} candidate(s) into the initial population.")
                remaining = cfg.population_size - len(population)
                if remaining > 0:
                    random_genomes = [[_random_gene_value(g, rng) for g in genes] for _ in range(remaining)]
                    population.extend(evaluate_batch(random_genomes))

                best_ever = max(population, key=lambda c: c.fitness)
                gen_summaries = [_summary(0, population)]
                log(f"Generation 0: best OOS fitness={gen_summaries[0].best_fitness:.3f}")

                for gen in range(1, cfg.generations + 1):
                    if cancel_event is not None and cancel_event.is_set():
                        log("Stop requested -- ending the search at the current generation.")
                        raise WalkforwardGACancelled("Walk-forward-aware GA search stopped by user.")
                    population.sort(key=lambda c: c.fitness, reverse=True)
                    elites = population[: cfg.elite_count]
                    next_pop = list(elites)
                    n_immigrants = max(1, round(cfg.population_size * cfg.random_immigrants_frac))
                    n_bred = max(cfg.population_size - len(elites) - n_immigrants, 0)

                    bred_genomes = []
                    for _ in range(n_bred):
                        pa = _tournament_select(population, rng)
                        pb = _tournament_select(population, rng)
                        child_genome = _crossover(pa.genome, pb.genome, rng)
                        child_genome = _mutate(child_genome, genes, cfg.mutation_rate, cfg.mutation_strength, rng)
                        bred_genomes.append(child_genome)
                    if bred_genomes:
                        next_pop.extend(evaluate_batch(bred_genomes))

                    # AI-suggested genomes take up to n_immigrants of the
                    # remaining slots (never an elite slot -- see the docstring
                    # above), so a fresh round of suggestions each generation can
                    # actually influence the search as it progresses, not just at
                    # the start. Whatever's left over still falls back to random
                    # immigrants exactly as before.
                    remaining = cfg.population_size - len(next_pop)
                    ai_added = 0
                    if remaining > 0:
                        # `population` here is still the PRIOR generation's fully
                        # evaluated set (before this generation's next_pop
                        # replaces it below), so this is exactly the "population
                        # this callback should analyze" snapshot for its optional
                        # second argument.
                        prior_population_snapshot = [(c.genome, c.fitness) for c in population]
                        ai_batch: list[list[float]] = []
                        for ai_genome in ai_genomes(genes, prior_population_snapshot):
                            if ai_added >= n_immigrants or len(next_pop) + len(ai_batch) >= cfg.population_size:
                                break
                            if len(ai_genome) == len(genes):
                                ai_batch.append(ai_genome)
                                ai_added += 1
                        if ai_batch:
                            next_pop.extend(evaluate_batch(ai_batch))
                            log(f"AI assist: seeded {ai_added} candidate(s) into generation {gen}.")
                    remaining = cfg.population_size - len(next_pop)
                    if remaining > 0:
                        random_genomes = [[_random_gene_value(g, rng) for g in genes] for _ in range(remaining)]
                        next_pop.extend(evaluate_batch(random_genomes))

                    population = next_pop
                    gen_best = max(population, key=lambda c: c.fitness)
                    if gen_best.fitness > best_ever.fitness:
                        best_ever = gen_best
                    gen_summaries.append(_summary(gen, population))
                    log(f"Generation {gen}/{cfg.generations}: best OOS fitness={gen_summaries[-1].best_fitness:.3f} "
                        f"mean={gen_summaries[-1].mean_fitness:.3f}")
            else:
                # TPE / CMA-ES -- see app.optimize.refinement.OPTIMIZER_MODES
                # for what each mode is. Both propose a whole BATCH of
                # population_size genomes per "generation" and hand that
                # batch to the exact same `evaluate_batch` every genetic
                # generation already uses above -- so the worker-pool
                # parallelism, its memory-safety sizing, and the fallback-
                # to-serial-on-failure behavior are all reused completely
                # unchanged regardless of which mode is running. AI-suggestion
                # injection (ai_suggest_cb) is a population-feedback hook
                # specific to the genetic mode's generational structure and
                # is not called in these two branches -- see the warning
                # appended below when both are configured together.
                if ai_suggest_cb is not None:
                    warnings.append(
                        f"An AI-suggestion callback was provided, but optimizer_mode='{cfg.optimizer_mode}' "
                        "does not use it -- AI-assisted genome suggestions are currently genetic-mode only."
                    )
                if cfg.optimizer_mode == "tpe":
                    population, gen_summaries = _run_tpe_batches(genes, evaluate_batch, cfg, log)
                elif cfg.optimizer_mode == "cma_es":
                    population, gen_summaries = _run_cma_es_batches(genes, evaluate_batch, cfg, log)
                else:
                    raise RefinementError(
                        f"Unknown optimizer_mode '{cfg.optimizer_mode}'. Supported: genetic, tpe, cma_es."
                    )
                # Leaderboard/generation-history shape stays exactly
                # population_size (matching what the genetic branch's own
                # `population` holds at the end of its last generation) --
                # baseline is still considered for best_ever below, just
                # not folded into the reported leaderboard batch itself.
                best_ever = max(population + [baseline], key=lambda c: c.fitness)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
            if pool_tmp_dir is not None:
                shutil.rmtree(pool_tmp_dir, ignore_errors=True)

        in_sample_fitness = full_df_fitness(best_ever.genome)
        overfitting_gap = None
        if math.isfinite(in_sample_fitness) and math.isfinite(best_ever.fitness):
            overfitting_gap = in_sample_fitness - best_ever.fitness
            if overfitting_gap > 0.3 * (abs(in_sample_fitness) or 1.0):
                warnings.append(
                    "The best genome's in-sample (full-dataset) fitness is substantially "
                    "higher than its chained out-of-sample fitness -- this is exactly the "
                    "gap Iterative Refinement's plain in-sample GA cannot see, and is a "
                    "sign of continued overfitting risk even after walk-forward-aware selection."
                )

        config_snapshot, code_text, code_ext = (None, None, None)
        if strategy.source_type == "manual":
            from app.optimize.parameter_space import apply_genome
            config_snapshot = apply_genome(strategy.config, genes, best_ever.genome)
        else:
            from app.optimize.code_parameter_space import patched_source_for_strategy
            code_text, code_ext = patched_source_for_strategy(strategy, genes, best_ever.genome)

        best_candidate = WalkforwardGACandidate(
            genome=best_ever.genome, fitness=best_ever.fitness, oos_trade_count=best_ever.trade_count,
            in_sample_fitness=in_sample_fitness, config=config_snapshot, code_text=code_text, code_extension=code_ext,
        )
        leaderboard = [
            WalkforwardGACandidate(genome=c.genome, fitness=c.fitness, oos_trade_count=c.trade_count)
            for c in sorted(population, key=lambda c: c.fitness, reverse=True)
        ]

        elapsed = time.time() - t0
        log(f"Walk-forward-aware GA complete in {elapsed:.1f}s.")

        return WalkforwardGAResult(
            refinement_config=cfg,
            n_folds=len(folds),
            window_mode=window_mode,
            genes=genes,
            best=best_candidate,
            generation_history=gen_summaries,
            leaderboard=leaderboard,
            overfitting_gap=overfitting_gap,
            elapsed_seconds=elapsed,
            warnings=warnings,
            max_test_bar_used=max_test_bar_used,
            total_evaluations=total_evaluations[0],
        )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _summary(gen: int, population: list) -> WalkforwardGAGenerationSummary:
    finite = [c.fitness for c in population if math.isfinite(c.fitness)]
    best = max((c.fitness for c in population), default=float("-inf"))
    mean = (sum(finite) / len(finite)) if finite else float("-inf")
    return WalkforwardGAGenerationSummary(generation=gen, best_fitness=best, mean_fitness=mean)


# ---------------------------------------------------------------------------
# TPE / CMA-ES batch search -- see app.optimize.refinement's identical-in-
# spirit _run_tpe_search / _run_cma_es_search for the single-genome version
# used by Iterative Refinement. The difference here is BATCHED: each
# "generation" proposes a whole population_size batch of genomes up front
# and hands it to the caller's `evaluate_batch`, so the exact same worker-
# pool parallelism (and its memory-safety sizing / serial fallback) that
# already makes the genetic mode fast on a multi-core machine works
# identically for these two modes -- neither one bypasses it.
# ---------------------------------------------------------------------------

def _run_tpe_batches(genes: list, evaluate_batch: Callable, cfg: RefinementConfig, log) -> tuple[list, list]:
    try:
        import optuna
    except ImportError as exc:
        raise RefinementError(
            "optimizer_mode='tpe' requires the optuna package (pip install optuna) -- "
            "it isn't a hard dependency of the rest of this app, only of TPE search."
        ) from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=cfg.random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    all_population: list = []
    gen_summaries: list[WalkforwardGAGenerationSummary] = []
    total_batches = cfg.generations + 1  # matches genetic mode's gen-0 initial population + `generations` more

    for gen in range(total_batches):
        trials = []
        genomes = []
        for _ in range(cfg.population_size):
            trial = study.ask()
            genome = []
            for gene in genes:
                if gene.is_int:
                    v = trial.suggest_int(gene.label, int(round(gene.lo)), int(round(gene.hi)))
                else:
                    v = trial.suggest_float(gene.label, float(gene.lo), float(gene.hi))
                genome.append(float(v))
            trials.append(trial)
            genomes.append(genome)

        batch = evaluate_batch(genomes)
        for trial, cand in zip(trials, batch):
            study.tell(trial, cand.fitness if math.isfinite(cand.fitness) else -1e18)

        all_population.extend(batch)
        gen_summaries.append(_summary(gen, batch))
        log(f"TPE batch {gen}/{cfg.generations}: best OOS fitness={gen_summaries[-1].best_fitness:.3f} "
            f"mean={gen_summaries[-1].mean_fitness:.3f}")

    return all_population[-cfg.population_size:], gen_summaries


def _run_cma_es_batches(genes: list, evaluate_batch: Callable, cfg: RefinementConfig, log) -> tuple[list, list]:
    try:
        import cma
    except ImportError as exc:
        raise RefinementError(
            "optimizer_mode='cma_es' requires the cma package (pip install cma) -- "
            "it isn't a hard dependency of the rest of this app, only of CMA-ES search."
        ) from exc

    # Genes normalized to a shared 0..1 range internally (see
    # app.optimize.refinement._run_cma_es_search's identical rationale) so
    # a period-2-to-30 gene and an ATR-multiplier 0.5-to-5 gene don't get a
    # badly mismatched step size.
    es = cma.CMAEvolutionStrategy(
        [0.5] * len(genes), 0.3,
        {"bounds": [0.0, 1.0], "popsize": cfg.population_size, "verbose": -9,
         # UPGRADE (reproducibility bug fix): see app.optimize.refinement's
         # identical fix for the full investigation -- "seed" alone seeds
         # numpy's GLOBAL random state, but cma's default "randn" reads
         # from that same global state, so the backtest/OOS-fold scoring
         # this loop runs between every ask()/tell() pair silently broke
         # reproducibility for any search past one generation. An isolated
         # RandomState passed as "randn" fixes it.
         "randn": np.random.RandomState(cfg.random_seed or 0).randn},
    )

    def _denormalize(unit_genome: list) -> list:
        out = []
        for u, gene in zip(unit_genome, genes):
            v = gene.lo + max(0.0, min(1.0, u)) * (gene.hi - gene.lo)
            out.append(float(round(v)) if gene.is_int else float(v))
        return out

    all_population: list = []
    gen_summaries: list[WalkforwardGAGenerationSummary] = []
    last_batch: list = []
    gen = 0
    while gen <= cfg.generations and not es.stop():
        solutions = es.ask()
        genomes = [_denormalize(s) for s in solutions]
        batch = evaluate_batch(genomes)
        penalties = [-c.fitness if math.isfinite(c.fitness) else 1e12 for c in batch]
        es.tell(solutions, penalties)

        all_population.extend(batch)
        last_batch = batch
        gen_summaries.append(_summary(gen, batch))
        log(f"CMA-ES generation {gen}/{cfg.generations}: best OOS fitness={gen_summaries[-1].best_fitness:.3f} "
            f"mean={gen_summaries[-1].mean_fitness:.3f}")
        gen += 1

    return last_batch or all_population, gen_summaries

