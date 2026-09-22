"""
Multi-market aggregate scoring.

Every other search tool in this app (Search Lab's own multi-instrument
mode, Evolution Lab's MultiInstrumentEvolutionGroup) runs the SAME search
independently per instrument and then, at best, pools the separate winners
into one leaderboard afterward (see app.search.budget_allocator.
pooled_cross_instrument_leaderboard) -- which still ends up as "the best ES
strategy" and "the best NQ strategy" side by side, not one strategy whose
behavior has been checked against not depending on either market alone.

This module does the other thing: for a GIVEN CANDIDATE (one genome), it
backtests that SAME candidate against every selected market's own data and
aggregates the per-market fitness scores (mean, dispersion, worst-case)
into ONE robustness score -- and then searches (genetic / TPE / CMA-ES,
same three modes as app.optimize.refinement) for the genome that scores
best on THAT aggregate, not on any one market alone. Conceptually:

                 PARAMETERS
                     |
        +------------+------------+
        |            |            |
       ES           NQ           GC
        |            |            |
       84%          91%          78%
        +------------+------------+
                     |
             ROBUSTNESS SCORE

Reuses the exact same per-candidate scoring primitive every other search
tool uses (app.optimize.refinement._evaluate -> run_backtest -> prop
simulation -> Monte Carlo) and the same low-level genetic operators
(_crossover/_mutate/_random_gene_value/_tournament_select/_build_adapter)
walkforward_ga.py already reuses from refinement.py -- this is a new
AGGREGATION and SEARCH-OBJECTIVE, not a new backtest engine.

Does NOT (yet) parallelize across a worker pool the way
app.optimize.walkforward_ga does -- each candidate here already does
len(markets) backtests serially, and adding cross-process parallelism on
top of that is a real, separate piece of engineering (pickling N
DataFrames per worker, etc.) rather than a natural extension of this
module. Flagged in this module's own delivery notes, not silently skipped.
"""
from __future__ import annotations

import math
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import (
    FITNESS_METRICS,
    OPTIMIZER_MODES,
    RefinementConfig,
    _build_adapter,
    _crossover,
    _evaluate,
    _mutate,
    _random_gene_value,
    apply_cost_stress_penalty,
)
from app.prop.simulator import PropRules
from app.strategy.base import Strategy

ProgressCallback = Callable[[str], None]

AGGREGATION_METHODS: dict[str, str] = {
    "mean_minus_dispersion": (
        "Mean minus dispersion (recommended) -- rewards consistent performance across "
        "every market and penalizes a candidate that's great on one and mediocre on "
        "the rest, without being as pessimistic as pure worst-case"
    ),
    "worst_case": (
        "Worst-case -- a candidate's score IS its single weakest market; the most "
        "conservative choice, biased toward candidates with no bad markets at all"
    ),
    "mean": (
        "Mean -- plain average across markets; the least conservative choice, since "
        "one standout market can compensate for weak ones"
    ),
}

# A candidate that produced zero trades (fitness = -inf) on ANY selected
# market is never robust by this module's own definition, regardless of
# how well it did elsewhere -- a strategy that doesn't even trade GC isn't
# "a GC-robust strategy" no matter how it scores on ES and NQ. This is
# enforced for every aggregation method, not just worst_case.
_ANY_MARKET_DEAD_PENALTY = float("-inf")


@dataclass
class PerMarketScore:
    market: str
    fitness: float
    trade_count: int


@dataclass
class MultiMarketCandidate:
    generation: int
    genome: list
    per_market: list[PerMarketScore]
    mean_fitness: float
    worst_case_fitness: float
    dispersion: float  # population std-dev of per-market fitness (0 if any market is non-finite)
    robustness_score: float  # the value the search actually selects on -- see `aggregation`
    config: dict | None = None
    code_text: str | None = None
    code_extension: str | None = None


@dataclass
class MultiMarketGenerationSummary:
    generation: int
    best_robustness_score: float
    mean_robustness_score: float


@dataclass
class MultiMarketSearchResult:
    refinement_config: RefinementConfig
    aggregation: str
    fitness_metric: str
    markets: list[str]
    genes: list
    baseline: MultiMarketCandidate
    best: MultiMarketCandidate
    generation_history: list  # list[MultiMarketGenerationSummary]
    leaderboard: list  # list[MultiMarketCandidate], final generation/batch, sorted best-first
    total_evaluations: int  # candidates scored, NOT backtests run (each candidate = len(markets) backtests)
    elapsed_seconds: float
    warnings: list = field(default_factory=list)


def _aggregate(per_market: list[PerMarketScore], aggregation: str) -> tuple[float, float, float, float]:
    """Returns (mean, worst_case, dispersion, robustness_score)."""
    fitnesses = [p.fitness for p in per_market]
    if any(not math.isfinite(f) for f in fitnesses):
        # See _ANY_MARKET_DEAD_PENALTY's own docstring above.
        finite = [f for f in fitnesses if math.isfinite(f)]
        mean = (sum(finite) / len(finite)) if finite else float("-inf")
        worst_case = min(fitnesses) if fitnesses else float("-inf")
        return mean, worst_case, 0.0, _ANY_MARKET_DEAD_PENALTY

    mean = sum(fitnesses) / len(fitnesses)
    worst_case = min(fitnesses)
    variance = sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)
    dispersion = variance ** 0.5

    if aggregation == "worst_case":
        robustness = worst_case
    elif aggregation == "mean":
        robustness = mean
    else:  # "mean_minus_dispersion", also the fallback for an unrecognized value
        robustness = mean - dispersion
    return mean, worst_case, dispersion, robustness


def evaluate_multi_market(
    genome: list,
    genes: list,
    build: Callable,
    dfs: dict[str, pd.DataFrame],
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_cfg: MonteCarloConfig,
    fitness_metric: str,
    aggregation: str = "mean_minus_dispersion",
    generation: int = 0,
    keep_full: bool = False,
    cost_stress_multiplier: float | None = None,
    cost_stress_penalty_weight: float = 0.0,
    adaptive_risk=None,
) -> MultiMarketCandidate:
    """Backtests ONE genome against every market in `dfs`. Builds a FRESH
    strategy instance per market (via `build(genome)` inside the loop)
    rather than reusing one instance across all of them -- testing this
    module surfaced a real, pre-existing statefulness bug where reusing
    one ManualStrategy instance across multiple backtests against
    DIFFERENT data can leak state between them (confirmed directly: the
    exact same genome against the exact same GC data scored differently
    depending on whether that strategy instance had already backtested
    ES and NQ first). Every other search tool in this app already avoids
    this by construction -- refinement.py and walkforward_ga.py both call
    `build(genome)` fresh immediately before every single `_evaluate`
    call, never reusing an instance across separate backtests -- this
    module just hadn't followed that same discipline yet. The underlying
    engine bug itself is not fixed here (that's a separate, deeper
    investigation into the strategy engine); this only ensures THIS
    module never triggers it.
    `keep_full` only affects whether the candidate carries a config/code
    snapshot (for the baseline and the final best-ever candidate) --
    per-market bt_result/mc_result objects are never retained here (there
    would be one set per market per candidate, which is a lot of memory
    for no report that reads them)."""
    per_market: list[PerMarketScore] = []
    for market_label, df in dfs.items():
        strategy = build(genome)
        fitness, stats, _prop_summary, _mc_summary, bt_result, _mc_result, _single_run = _evaluate(
            df, strategy, risk, prop_rules, mc_cfg, fitness_metric, keep_full=False,
            cost_stress_multiplier=cost_stress_multiplier,
            cost_stress_penalty_weight=cost_stress_penalty_weight,
            adaptive_risk=adaptive_risk,
        )
        trade_count = len(bt_result.trades) if bt_result is not None else (
            stats.get("total_trades", 0) if stats else 0
        )
        per_market.append(PerMarketScore(market=market_label, fitness=fitness, trade_count=trade_count))

    mean_fitness, worst_case, dispersion, robustness = _aggregate(per_market, aggregation)

    config, code_text, code_ext = (None, None, None)
    if keep_full:
        snapshot_strategy = build(genome)
        if genes and hasattr(snapshot_strategy, "config"):
            from app.optimize.parameter_space import apply_genome
            config = apply_genome(snapshot_strategy.config, genes, genome)
        elif genes:
            from app.optimize.code_parameter_space import patched_source_for_strategy
            code_text, code_ext = patched_source_for_strategy(snapshot_strategy, genes, genome)

    return MultiMarketCandidate(
        generation=generation, genome=list(genome), per_market=per_market,
        mean_fitness=mean_fitness, worst_case_fitness=worst_case, dispersion=dispersion,
        robustness_score=robustness, config=config, code_text=code_text, code_extension=code_ext,
    )


def _summarize(gen: int, population: list[MultiMarketCandidate]) -> MultiMarketGenerationSummary:
    finite = [c.robustness_score for c in population if math.isfinite(c.robustness_score)]
    best = max((c.robustness_score for c in population), default=float("-inf"))
    mean = (sum(finite) / len(finite)) if finite else float("-inf")
    return MultiMarketGenerationSummary(generation=gen, best_robustness_score=best, mean_robustness_score=mean)


def _tournament_select_by_robustness(
    population: list[MultiMarketCandidate], rng: random.Random, k: int = 3,
) -> MultiMarketCandidate:
    """Same tournament-selection shape as app.optimize.refinement's
    _tournament_select, but reading `.robustness_score` instead of
    `.fitness` -- MultiMarketCandidate deliberately doesn't call its
    aggregate score `fitness` (it isn't one backtest's fitness, it's an
    aggregate across markets), so that function's hardcoded `.fitness`
    attribute access can't be reused directly here."""
    k = min(k, len(population))
    contenders = rng.sample(population, k)
    return max(contenders, key=lambda c: c.robustness_score)


def run_multi_market_search(
    dfs: dict[str, pd.DataFrame],
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    refinement_config: RefinementConfig | None = None,
    aggregation: str = "mean_minus_dispersion",
    progress_cb: ProgressCallback | None = None,
) -> MultiMarketSearchResult:
    """The search loop itself. `refinement_config.optimizer_mode` selects
    genetic (default) / tpe / cma_es exactly as it does everywhere else in
    this app -- all three call the SAME `evaluate_multi_market` above, so
    switching modes never changes what a candidate is scored on, only how
    the next genome to try is chosen (same principle as
    app.optimize.refinement and app.optimize.walkforward_ga)."""
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if len(dfs) < 2:
        raise RefinementError(
            "Multi-market aggregate scoring needs at least 2 markets selected -- "
            "with only one, this is just Iterative Refinement with extra steps."
        )
    if aggregation not in AGGREGATION_METHODS:
        raise RefinementError(
            f"Unknown aggregation method '{aggregation}'. Supported: {list(AGGREGATION_METHODS)}."
        )

    cfg = refinement_config or RefinementConfig(population_size=12, generations=6, search_monte_carlo_sims=200)
    t0 = time.time()
    warnings: list[str] = []
    markets = list(dfs.keys())

    tmp_dir: Path | None = None
    if strategy.source_type == "python":
        tmp_dir = Path(tempfile.mkdtemp(prefix="t58_multimkt_"))

    try:
        genes, build = _build_adapter(strategy, tmp_dir)
        if not genes:
            raise RefinementError(
                "This strategy has no tunable numeric parameters -- there is nothing "
                "for a multi-market search to search over."
            )

        rng = random.Random(cfg.random_seed)
        search_mc_cfg = MonteCarloConfig(
            n_simulations=cfg.search_monte_carlo_sims, random_seed=mc_config.random_seed,
            reset_on_breach=mc_config.reset_on_breach,
        )
        total_evaluations = [0]
        all_evaluated: list[MultiMarketCandidate] = []

        def evaluate(genome: list, generation: int, keep_full: bool = False) -> MultiMarketCandidate:
            cand = evaluate_multi_market(
                genome, genes, build, dfs, risk, prop_rules, search_mc_cfg, cfg.fitness_metric,
                aggregation=aggregation, generation=generation, keep_full=keep_full,
                cost_stress_multiplier=cfg.cost_stress_multiplier if cfg.cost_stress_enabled else None,
                cost_stress_penalty_weight=cfg.cost_stress_penalty_weight if cfg.cost_stress_enabled else 0.0,
            )
            total_evaluations[0] += 1
            all_evaluated.append(cand)
            return cand

        log(f"Analyzing strategy parameters... found {len(genes)} tunable parameter(s). "
            f"Scoring across {len(markets)} market(s): {', '.join(markets)}.")

        baseline = evaluate([g.base_value for g in genes], 0, keep_full=True)
        log(
            f"Baseline robustness score ({aggregation}): {baseline.robustness_score:.3f} "
            f"(mean={baseline.mean_fitness:.3f}, worst={baseline.worst_case_fitness:.3f})"
        )

        def _run_genetic() -> tuple[list[MultiMarketCandidate], list[MultiMarketGenerationSummary]]:
            population = [baseline]
            while len(population) < cfg.population_size:
                population.append(evaluate([_random_gene_value(g, rng) for g in genes], 0))
            history = [_summarize(0, population)]
            log(f"Generation 0: best robustness={history[0].best_robustness_score:.3f}")

            for gen in range(1, cfg.generations + 1):
                population.sort(key=lambda c: c.robustness_score, reverse=True)
                elites = population[: cfg.elite_count]
                next_pop = list(elites)
                n_immigrants = max(1, round(cfg.population_size * cfg.random_immigrants_frac))
                n_bred = max(cfg.population_size - len(elites) - n_immigrants, 0)

                for _ in range(n_bred):
                    pa = _tournament_select_by_robustness(population, rng, k=3)
                    pb = _tournament_select_by_robustness(population, rng, k=3)
                    child = _crossover(pa.genome, pb.genome, rng)
                    child = _mutate(child, genes, cfg.mutation_rate, cfg.mutation_strength, rng)
                    next_pop.append(evaluate(child, gen))
                while len(next_pop) < cfg.population_size:
                    next_pop.append(evaluate([_random_gene_value(g, rng) for g in genes], gen))

                population = next_pop
                history.append(_summarize(gen, population))
                log(f"Generation {gen}/{cfg.generations}: best robustness={history[-1].best_robustness_score:.3f} "
                    f"mean={history[-1].mean_robustness_score:.3f}")
            return population, history

        def _run_tpe() -> tuple[list[MultiMarketCandidate], list[MultiMarketGenerationSummary]]:
            try:
                import optuna
            except ImportError as exc:
                raise RefinementError(
                    "optimizer_mode='tpe' requires the optuna package (pip install optuna)."
                ) from exc
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.random_seed))
            history: list[MultiMarketGenerationSummary] = []
            last_batch: list[MultiMarketCandidate] = []
            for gen in range(cfg.generations + 1):
                trials, batch = [], []
                for _ in range(cfg.population_size):
                    trial = study.ask()
                    genome = []
                    for gene in genes:
                        if gene.is_int:
                            genome.append(float(trial.suggest_int(gene.label, int(round(gene.lo)), int(round(gene.hi)))))
                        else:
                            genome.append(float(trial.suggest_float(gene.label, float(gene.lo), float(gene.hi))))
                    cand = evaluate(genome, gen)
                    trials.append(trial)
                    batch.append(cand)
                for trial, cand in zip(trials, batch):
                    study.tell(trial, cand.robustness_score if math.isfinite(cand.robustness_score) else -1e18)
                last_batch = batch
                history.append(_summarize(gen, batch))
                log(f"TPE batch {gen}/{cfg.generations}: best robustness={history[-1].best_robustness_score:.3f}")
            return last_batch, history

        def _run_cma_es() -> tuple[list[MultiMarketCandidate], list[MultiMarketGenerationSummary]]:
            try:
                import cma
            except ImportError as exc:
                raise RefinementError(
                    "optimizer_mode='cma_es' requires the cma package (pip install cma)."
                ) from exc
            es = cma.CMAEvolutionStrategy(
                [0.5] * len(genes), 0.3,
                {"bounds": [0.0, 1.0], "popsize": cfg.population_size, "verbose": -9,
                 # UPGRADE (reproducibility bug fix): see app.optimize.
                 # refinement's identical fix for the full rationale --
                 # "seed" alone seeds numpy's GLOBAL random state, but
                 # cma's default "randn" reads from that same global
                 # state, so anything else in this process touching numpy
                 # randomness between ask() calls (every backtest this
                 # loop runs) silently breaks reproducibility. An isolated
                 # RandomState passed as "randn" fixes it.
                 "randn": np.random.RandomState(cfg.random_seed or 0).randn},
            )

            def _denorm(u):
                out = []
                for v, gene in zip(u, genes):
                    x = gene.lo + max(0.0, min(1.0, v)) * (gene.hi - gene.lo)
                    out.append(float(round(x)) if gene.is_int else float(x))
                return out

            history: list[MultiMarketGenerationSummary] = []
            last_batch: list[MultiMarketCandidate] = []
            gen = 0
            while gen <= cfg.generations and not es.stop():
                solutions = es.ask()
                genomes = [_denorm(s) for s in solutions]
                batch = [evaluate(g, gen) for g in genomes]
                penalties = [-c.robustness_score if math.isfinite(c.robustness_score) else 1e12 for c in batch]
                es.tell(solutions, penalties)
                last_batch = batch
                history.append(_summarize(gen, batch))
                log(f"CMA-ES generation {gen}/{cfg.generations}: best robustness={history[-1].best_robustness_score:.3f}")
                gen += 1
            return last_batch, history

        if cfg.optimizer_mode == "genetic":
            population, generation_history = _run_genetic()
        elif cfg.optimizer_mode == "tpe":
            population, generation_history = _run_tpe()
        elif cfg.optimizer_mode == "cma_es":
            population, generation_history = _run_cma_es()
        else:
            raise RefinementError(f"Unknown optimizer_mode '{cfg.optimizer_mode}'.")

        best_ever = max(all_evaluated, key=lambda c: c.robustness_score)

        log("Re-scoring the best-ever candidate at full Monte Carlo fidelity across every market...")
        best_final = evaluate_multi_market(
            best_ever.genome, genes, build, dfs, risk, prop_rules, mc_config, cfg.fitness_metric,
            aggregation=aggregation, generation=best_ever.generation, keep_full=True,
            cost_stress_multiplier=cfg.cost_stress_multiplier if cfg.cost_stress_enabled else None,
            cost_stress_penalty_weight=cfg.cost_stress_penalty_weight if cfg.cost_stress_enabled else 0.0,
        )

        leaderboard = sorted(population, key=lambda c: c.robustness_score, reverse=True)
        elapsed = time.time() - t0
        log(f"Multi-market search complete in {elapsed:.1f}s ({total_evaluations[0]} candidates, "
            f"{total_evaluations[0] * len(markets)} total backtests).")

        return MultiMarketSearchResult(
            refinement_config=cfg, aggregation=aggregation, fitness_metric=cfg.fitness_metric,
            markets=markets, genes=genes, baseline=baseline, best=best_final,
            generation_history=generation_history, leaderboard=leaderboard,
            total_evaluations=total_evaluations[0], elapsed_seconds=elapsed, warnings=warnings,
        )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
