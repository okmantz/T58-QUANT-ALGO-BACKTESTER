"""
Iterative Refinement engine.

A small, dependency-free genetic algorithm that searches a strategy's
numeric parameter space for the configuration that scores best on a chosen
fitness metric, evaluated on the SAME historical dataset used for the
normal backtest. Works across all four strategy sources this app supports:

  Manual        -- every tunable numeric leaf in the config dict
                   (see app.optimize.parameter_space)
  Python        -- every top-level SCREAMING_SNAKE_CASE numeric constant
                   (see app.optimize.code_parameter_space)
  PineScript    -- every input.int()/input.float() default, plus the
                   T58_SL_PIPS/T58_TP_PIPS directives
  MQL5          -- every literal iMA()/iRSI() period, plus the same
                   T58_SL_PIPS/T58_TP_PIPS directives

Each "generation":
  1. keeps the top `elite_count` candidates from the current population unchanged
  2. breeds the rest via tournament-selected crossover + mutation
  3. injects a small fraction of fresh, fully-random candidates ("random
     immigrants") to keep the search from collapsing onto one local optimum
  4. re-evaluates the new population (backtest -> prop simulation ->
     a *cheap* Monte Carlo pass) and scores it

Across generations the population's parameters converge toward whatever
scores highest on the fitness metric -- the "genetic algorithm-like
optimization" Owen asked for. After the last generation, the single
best-ever candidate found across the whole search is re-evaluated once
more with the SAME Monte Carlo fidelity (n_simulations) as the main
pipeline, plus an out-of-sample holdout check, so the final report's
headline numbers are as trustworthy as the first report's.

IMPORTANT -- this is an in-sample search. It will always find *something*
that looks better on the exact historical window it was run against, even
if that improvement is pure noise. The holdout check and the prominent
overfitting-risk note in the generated report exist specifically to catch
that. Iterative Refinement should never be treated as proof a strategy
is better -- only as a faster way to generate candidates worth falsifying.
"""
from __future__ import annotations

import math
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np

from app.backtest.adaptive_risk import AdaptiveRiskConfig
from app.backtest.engine import BacktestResult, run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from app.optimize.code_parameter_space import discover_code_genes, materialize_code_strategy, patched_source_for_strategy
from app.optimize.parameter_space import GeneMeta, RefinementError, apply_genome, extract_genome
from app.prop.simulator import AccountSimResult, PropRules, simulate_account, summarize_single_run
from app.strategy.base import Strategy
from app.strategy.manual import ManualStrategy

ProgressCallback = Callable[[str], None]

CODE_SOURCE_TYPES = {"python", "pinescript", "mql5"}
SUPPORTED_SOURCE_TYPES = {"manual"} | CODE_SOURCE_TYPES

CODE_EXTENSIONS = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}

# Human-readable labels for the fitness-metric dropdown in the UI. Keys
# here are the exact strings accepted by RefinementConfig.fitness_metric.
#
# Most of these metrics (eval_pass_probability, prop_guide_score,
# composite_prop_score, first_payout_probability, expected_payout) are only
# meaningful if you're actually being scored against a prop firm's
# evaluation/funded-account rules (see app.prop.simulator.PropRules) -- they
# read PropRules and the Monte Carlo account simulation to score candidates.
#
# "net_profit" is the odd one out: compute_fitness() below reads it straight
# off the backtest's own statistics and never touches PropRules or the MC
# account simulation at all. That makes it the right choice for long-term /
# buy-and-hold-style retail trading where there is no prop firm evaluation to
# pass -- you just want the configuration that made the most money over the
# full test period, full stop.
FITNESS_METRICS: dict[str, str] = {
    "eval_pass_probability": "Eval Pass Probability -- reach target before hitting a limit (recommended for prop firms)",
    "prop_guide_score": "Prop-Oriented Guide Score",
    "composite_prop_score": "Composite Prop Score",
    "first_payout_probability": "First Payout Probability",
    "fastest_payout": "Fastest Payout -- optimizes for SPEED to first payout under a safety floor (for tight deadlines)",
    "expected_payout": "Expected Payout ($)",
    "net_profit": "Long-Term Net Profit ($) -- for long-term trading, no prop firm",
    "profit_factor": "Profit Factor",
    "sharpe_ratio": "Sharpe Ratio",
}

# Explicit optimizer modes. All three run through the exact SAME
# per-candidate execution path (_evaluate -> run_backtest -> prop
# simulation -> Monte Carlo, below) and feed the exact same downstream
# machinery (plateau-robust selection, cost-stress penalty, the final
# full-fidelity re-evaluation, the holdout check) -- only HOW the next
# genome to try is chosen differs. Picking a mode never changes what a
# candidate is scored on, only how the search space is explored.
OPTIMIZER_MODES: dict[str, str] = {
    "genetic": "Genetic Algorithm -- tournament selection + crossover/mutation + random immigrants (default, dependency-free)",
    "tpe": "TPE / Bayesian (Optuna) -- models which regions of the parameter space score well and samples more there each trial",
    "cma_es": "CMA-ES -- adapts a search distribution's shape/scale to the space's curvature; strong on continuous, correlated parameters",
}


@dataclass
class RefinementConfig:
    enabled: bool = False
    fitness_metric: str = "eval_pass_probability"
    population_size: int = 10
    generations: int = 5
    elite_count: int = 2
    mutation_rate: float = 0.35        # probability each gene mutates in a bred child
    mutation_strength: float = 0.25    # mutation step size, as a fraction of that gene's search range
    random_immigrants_frac: float = 0.15  # fraction of each new generation that is freshly randomized
    search_monte_carlo_sims: int = 500    # cheap MC used while searching, for speed
    random_seed: int | None = 42
    # Cost-stress: while ranking candidates, ALSO backtest each one at
    # spread_pips/slippage_pips/commission_per_trade multiplied by
    # cost_stress_multiplier, and blend that stressed-cost fitness into the
    # score the GA actually selects on (see _apply_cost_stress below). This
    # is what makes the search prefer strategies whose edge SURVIVES worse
    # execution, rather than strategies that only look good under the
    # default (fairly forgiving) cost assumptions -- without this, a
    # strategy that is pure curve-fit to optimistic fills scores identically
    # to a strategy with a real edge, right up until Stage 3's cost-ladder
    # check (which only reports, it doesn't feed back into what gets bred).
    cost_stress_enabled: bool = True
    cost_stress_multiplier: float = 2.0
    cost_stress_penalty_weight: float = 0.35   # 0 = ignore stress entirely, 1 = fully penalize any degradation

    # Plateau-robust champion selection: a GA searching a noisy in-sample
    # fitness surface will happily hand back a genome that sits on a thin,
    # one-bar-wide spike -- often just curve-fit noise -- instead of a
    # nearby, slightly-lower-scoring region that holds up under a small
    # nudge to any one parameter. When enabled (the default), the top
    # `plateau_finalist_pool` genomes found anywhere in the search are each
    # perturbed by +/- `plateau_neighbor_step_frac` of that gene's own
    # search range in every dimension, one dimension at a time; the genome
    # whose local neighborhood scores best on average (least penalized by
    # how much that average varies) is promoted to `best_ever` in place of
    # whichever raw single point happened to score highest. Set to False to
    # restore the old raw-best-fitness behavior.
    plateau_robust_selection: bool = True
    plateau_neighbor_step_frac: float = 0.08
    plateau_finalist_pool: int = 5

    # Explicit optimizer mode -- see OPTIMIZER_MODES above. "genetic" is
    # the exact pre-existing algorithm (default, zero behavior change).
    # "tpe" and "cma_es" spend the SAME evaluation budget
    # (population_size * (generations + 1), matching the genetic mode's
    # initial population + bred generations) choosing genomes a different
    # way, via the optional `optuna` / `cma` packages respectively.
    optimizer_mode: str = "genetic"

    def __post_init__(self):
        self.population_size = max(int(self.population_size), 4)
        self.generations = max(int(self.generations), 1)
        self.elite_count = max(1, min(int(self.elite_count), self.population_size - 1))
        self.mutation_rate = min(max(float(self.mutation_rate), 0.0), 1.0)
        self.mutation_strength = min(max(float(self.mutation_strength), 0.01), 1.0)
        self.random_immigrants_frac = min(max(float(self.random_immigrants_frac), 0.0), 0.9)
        self.search_monte_carlo_sims = max(int(self.search_monte_carlo_sims), 50)
        self.cost_stress_multiplier = max(float(self.cost_stress_multiplier), 1.0)
        self.cost_stress_penalty_weight = min(max(float(self.cost_stress_penalty_weight), 0.0), 1.0)
        self.plateau_neighbor_step_frac = min(max(float(self.plateau_neighbor_step_frac), 0.01), 0.5)
        self.plateau_finalist_pool = max(int(self.plateau_finalist_pool), 1)
        if self.fitness_metric not in FITNESS_METRICS:
            raise RefinementError(f"Unknown fitness metric '{self.fitness_metric}'.")
        if self.optimizer_mode not in OPTIMIZER_MODES:
            raise RefinementError(
                f"Unknown optimizer_mode '{self.optimizer_mode}'. Supported: {list(OPTIMIZER_MODES)}."
            )


@dataclass
class Candidate:
    generation: int
    genome: list
    fitness: float
    source_type: str = "manual"
    config: dict | None = None              # Manual Strategy config dict (manual only)
    code_text: str | None = None            # patched source text (python/pinescript/mql5 only)
    code_extension: str | None = None       # ".py" / ".pine" / ".mq5" (code strategies only)
    statistics: dict | None = None          # BacktestStatistics.to_dict()
    prop_summary: dict | None = None        # summarize_single_run(...)
    mc_summary: dict | None = None          # a few key Monte Carlo fields (cheap to keep for every candidate)
    # Full objects are only kept for the baseline and the final best
    # candidate (see _evaluate keep_full=) -- keeping them for every
    # candidate in every generation would mean holding population_size *
    # generations full equity curves in memory at once.
    bt_result: BacktestResult | None = None
    mc_result: MonteCarloResult | None = None
    single_run: AccountSimResult | None = None


@dataclass
class GenerationSummary:
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float   # average normalized population std-dev across genes; falls as the GA converges


@dataclass
class RefinementResult:
    refinement_config: RefinementConfig
    fitness_metric: str
    source_type: str
    genes: list  # list[GeneMeta] (manual) or list[CodeGene] (python/pinescript/mql5)
    baseline: Candidate
    best: Candidate
    generation_history: list  # list[GenerationSummary]
    leaderboard: list  # list[Candidate], final generation, sorted best-first
    holdout_comparison: dict | None
    elapsed_seconds: float
    warnings: list = field(default_factory=list)
    # None when plateau_robust_selection was off, or no finite candidate ever
    # existed to select from. Otherwise: raw_best_fitness (the single highest
    # in-sample fitness seen anywhere in the search), chosen_fitness (the
    # plateau-robust pick's own fitness), swapped (whether the robust pick
    # differs from the raw optimum), and neighbor_step_frac (for the report).
    plateau_robustness: dict | None = None
    # Evaluation-count transparency: exactly how many candidates this run
    # actually backtested (baseline + every genome tried across the whole
    # search, before plateau-robustness's cheap neighbor probes, which are
    # deliberately excluded -- see `track=False` in run_iterative_refinement).
    total_evaluations: int = 0
    # "Distributions, not just a winner": median vs. best across every
    # candidate this run evaluated, for whichever metrics are computed per-
    # candidate already (eval/first-payout pass probability from the cheap
    # per-candidate Monte Carlo pass, max drawdown from its own backtest
    # statistics). None only when zero candidates produced a finite result.
    # See _compute_distribution_summary's own docstring for exactly what
    # this is -- and, as important, what it is NOT (no per-candidate
    # OOS/regime numbers exist inside this loop, so those are never
    # fabricated here).
    distribution_summary: dict | None = None


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def _band(value: float, floor: float, ceiling: float) -> float:
    """0 at/below `floor`, ramps linearly to 1 at/above `ceiling`. Small
    helper for scoring a raw metric against one of the guide's target
    bands without repeating the same clamp-and-scale logic for each one."""
    if not math.isfinite(value):
        return 0.0
    if value <= floor:
        return 0.0
    if value >= ceiling:
        return 1.0
    return (value - floor) / (ceiling - floor)


def _prop_guide_score(stats: dict, mc: MonteCarloResult) -> float:
    """Scores a strategy against the PROP-ORIENTED STRATEGY GENERATION
    GUIDE's "ideal performance profile" (its section 4 target table)
    instead of collapsing everything to net profit, or even just eval-pass
    probability alone.

    Two strategies with identical eval-pass probability can still be very
    different bets to fund: one might clear the bar with a P95 drawdown
    that hugs the firm's actual limit, a handful of oversized trades, or
    too few trades to trust the number at all -- exactly the failure
    patterns the guide calls out by name in its "do not optimize for
    these alone" section. This blends the guide's own target bands (win
    rate 50-65%, profit factor >= 1.3, 100+ trades, P95 drawdown under
    roughly half of a typical prop firm's overall drawdown limit) into
    the pass/payout-probability objective the guide is actually chasing,
    so the GA can't win by exploiting the numbers this metric doesn't
    look at.

    Deliberately NOT normalized to a clean 0-1 range -- what matters for
    the GA/refinement's tournament selection is relative ordering between
    candidates, not the absolute scale."""
    eval_pass = mc.evaluation_pass_probability / 100.0
    payout = mc.first_payout_probability / 100.0
    ruin_penalty = mc.risk_of_ruin_pct / 100.0

    pf = stats.get("profit_factor", 0.0)
    pf_score = _band(pf if math.isfinite(pf) else 0.0, 1.0, 1.5)

    trade_score = _band(stats.get("total_trades", 0), 30, 150)

    win_rate = stats.get("win_rate", 0.0)
    if 50.0 <= win_rate <= 65.0:
        win_rate_score = 1.0
    elif win_rate < 50.0:
        win_rate_score = _band(win_rate, 30.0, 50.0)
    else:
        win_rate_score = max(0.0, 1.0 - (win_rate - 65.0) / 25.0)

    # P95 drawdown vs a typical ~10%-max-drawdown prop firm: the guide
    # wants P95 drawdown under roughly 50-60% of the firm's actual limit.
    # compute_fitness doesn't have the active PropRules in scope here, so
    # this uses that typical 10% figure as a stand-in rather than the
    # exact configured limit -- close enough to penalize a strategy that
    # runs uncomfortably close to ANY reasonable drawdown limit, without
    # needing to thread PropRules through every call site of this metric.
    dd_score = 1.0 - _band(mc.p95_drawdown_pct, 6.0, 10.0)

    return (
        eval_pass * 0.40
        + payout * 0.25
        + pf_score * 0.10
        + trade_score * 0.08
        + win_rate_score * 0.07
        + dd_score * 0.10
        - ruin_penalty * 0.15
    )


def _fastest_payout_score(mc: MonteCarloResult) -> float:
    """Optimizes DIRECTLY for speed to first payout, not just the probability
    of eventually getting there -- the gap flagged in this app's own review:
    every other metric here (first_payout_probability, composite_prop_score,
    prop_guide_score) treats a candidate that reaches payout in a median 9
    days exactly the same as one that takes 40, as long as both eventually
    clear the bar. For a genuinely time-constrained goal ("I need a payout
    ASAP"), speed itself is the objective -- and median_days_to_first_payout
    (falling back to median_days_to_pass for a candidate whose simulations
    never got as far as a funded payout) is a number every Monte Carlo run
    already computes (see MonteCarloResult), just never fed into a fitness
    metric before this one.

    Blended with a safety floor so it can't win by being reckless: a
    candidate that reaches payout in 3 days on the rare 2% of simulations
    that don't blow up first isn't "faster" than one that reaches it in 10
    days on 70% of simulations -- it's just less likely to ever get there,
    which the raw median days figure alone would not penalize (a median is
    only computed over the simulations that DID reach payout; it says
    nothing about how rare those simulations were). eval_pass_probability
    gates the score for exactly this reason: below MIN_VIABLE_PASS_PROBABILITY
    the score collapses toward zero regardless of how fast the rare
    successes are, so a search using this metric can't win by concentrating
    everything into a wild, mostly-losing strategy in the hope of an
    occasional lightning-fast pass. The gate is soft (squared ratio, not a
    hard cutoff) so a GA still has a gradient to climb even below the floor
    instead of a flat zero that gives it nothing to optimize against.
    """
    MIN_VIABLE_PASS_PROBABILITY = 40.0  # below this, speed doesn't matter -- it barely ever passes at all
    MAX_MEANINGFUL_DAYS = 60.0          # candidates at/above this all score ~0 -- no upside in ranking "45 days" vs "90 days"

    days = mc.median_days_to_first_payout
    if days is None:
        days = mc.median_days_to_pass  # never reached a first payout in ANY simulation -- fall back to days-to-pass
    if days is None or days <= 0:
        return 0.0  # never passed in any simulation at all -- no speed to reward

    speed_component = max(0.0, 1.0 - (float(days) / MAX_MEANINGFUL_DAYS))  # 1.0 = instant, 0.0 = at/beyond the cutoff

    pass_prob = max(0.0, min(100.0, mc.evaluation_pass_probability)) / 100.0
    safety_gate = min(1.0, pass_prob / (MIN_VIABLE_PASS_PROBABILITY / 100.0)) ** 2

    return speed_component * safety_gate * 100.0  # 0..100 scale, consistent with the other metrics here


def compute_fitness(stats: dict, prop_summary: dict | None, mc: MonteCarloResult, metric: str) -> float:
    if metric == "net_profit":
        return float(stats.get("net_profit", 0.0))
    if metric == "profit_factor":
        pf = stats.get("profit_factor", 0.0)
        return 10.0 if pf == float("inf") else float(pf)
    if metric == "sharpe_ratio":
        return float(stats.get("sharpe_ratio", 0.0))
    if metric == "eval_pass_probability":
        return float(mc.evaluation_pass_probability)
    if metric == "first_payout_probability":
        return float(mc.first_payout_probability)
    if metric == "fastest_payout":
        return _fastest_payout_score(mc)
    if metric == "expected_payout":
        return float(mc.expected_payout)
    if metric == "composite_prop_score":
        return float(
            mc.evaluation_pass_probability * 0.5
            + mc.first_payout_probability * 0.3
            - mc.risk_of_ruin_pct * 0.2
        )
    if metric == "prop_guide_score":
        return _prop_guide_score(stats, mc)
    raise RefinementError(f"Unknown fitness metric '{metric}'.")


def _stressed_risk_config(risk: RiskConfig, multiplier: float) -> RiskConfig:
    """A copy of `risk` with every execution-cost assumption (spread,
    slippage, commission) scaled up by `multiplier` -- used to re-backtest
    a candidate under deliberately worse fills, never to change position
    sizing or account rules."""
    return replace(
        risk,
        spread_pips=risk.spread_pips * multiplier,
        slippage_pips=risk.slippage_pips * multiplier,
        commission_per_trade=risk.commission_per_trade * multiplier,
    )


def apply_cost_stress_penalty(nominal_fitness: float, stressed_fitness: float, weight: float) -> float:
    """
    Blends a stressed-cost fitness value into a nominal one, penalizing
    degradation without rewarding a stressed run that (by noise) scores
    slightly ABOVE nominal. Metric-agnostic: works the same whether
    `metric` is a raw dollar figure, a ratio, or a 0-1 probability, because
    the penalty is expressed as a FRACTION of the nominal score, not an
    absolute offset.

        weight=0.0 -> stress is ignored entirely (returns nominal_fitness)
        weight=1.0 -> full erosion under stress drives fitness to exactly 0

    A candidate that is already unprofitable/invalid at nominal cost
    (nominal_fitness <= 0, or non-finite) is returned unchanged -- there is
    no meaningful "how much of the edge survived" to measure once there
    was no edge at nominal cost either, and stressing it further would
    double-penalize a candidate Stage 1/the GA's own selection pressure
    already rejects on nominal grounds.
    """
    if weight <= 0 or not math.isfinite(nominal_fitness) or nominal_fitness <= 0:
        return nominal_fitness
    if not math.isfinite(stressed_fitness):
        degradation = 1.0
    else:
        degradation = max(0.0, (nominal_fitness - stressed_fitness) / abs(nominal_fitness))
        degradation = min(degradation, 1.0)
    return nominal_fitness - weight * degradation * nominal_fitness


def _mc_summary(mc: MonteCarloResult) -> dict:
    return {
        "evaluation_pass_probability": mc.evaluation_pass_probability,
        "first_payout_probability": mc.first_payout_probability,
        "failure_before_payout_probability": mc.failure_before_payout_probability,
        "expected_payout": mc.expected_payout,
        "risk_of_ruin_pct": mc.risk_of_ruin_pct,
        "median_drawdown_pct": mc.median_drawdown_pct,
        "n_simulations": mc.n_simulations,
    }


def _evaluate(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_cfg: MonteCarloConfig,
    metric: str,
    keep_full: bool = False,
    cost_stress_multiplier: float | None = None,
    cost_stress_penalty_weight: float = 0.0,
    adaptive_risk: AdaptiveRiskConfig | None = None,
):
    """Runs one full backtest -> prop sim -> Monte Carlo pass for one strategy instance.

    When `cost_stress_multiplier` is given (and `cost_stress_penalty_weight` >
    0), ALSO re-runs the same strategy on the same data with spread/
    slippage/commission scaled up by that multiplier, and blends that
    stressed-cost result into the returned fitness via
    apply_cost_stress_penalty(). The returned `statistics`/`prop_summary`/
    `mc_summary` always describe the NOMINAL run (so reports keep showing
    real, un-stressed numbers) -- only the scalar fitness the GA selects on
    is cost-stress-adjusted.

    adaptive_risk: same declarative, limit-aware position-sizing overlay
    Quick Optimize and Full Pipeline already accept (see
    app.backtest.adaptive_risk) -- applied identically to every backtest
    below (baseline, every GA candidate, the stressed-cost pass, and the
    final full-fidelity re-run) so a strategy searched here with this
    enabled produces a result that's actually comparable to a Quick
    Optimize/Full Pipeline run made with the same setting, instead of
    silently diverging the way an unthrottled run can (see the 2026-09-16
    Quick-Optimize-vs-Full-Pipeline diagnosis this closes for Iterative
    Refinement).
    """
    bt_result = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)

    if not bt_result.trades:
        return (
            float("-inf"),
            bt_result.statistics.to_dict(),
            None,
            None,
            bt_result if keep_full else None,
            None,
            None,
        )

    trade_pnls = [t.pnl for t in bt_result.trades]
    trade_dates = [t.entry_time for t in bt_result.trades]
    single_run = simulate_account(trade_pnls, trade_dates, prop_rules)
    mc_result = run_monte_carlo(bt_result.trades, prop_rules, mc_cfg)
    prop_summary = summarize_single_run(single_run)

    fitness = compute_fitness(bt_result.statistics.to_dict(), prop_summary, mc_result, metric)
    if not math.isfinite(fitness):
        fitness = float("-inf")

    if cost_stress_multiplier and cost_stress_penalty_weight > 0 and math.isfinite(fitness):
        stressed_risk = _stressed_risk_config(risk, cost_stress_multiplier)
        stressed_bt = run_backtest(df, strategy, stressed_risk, adaptive_risk=adaptive_risk)
        if stressed_bt.trades:
            stressed_pnls = [t.pnl for t in stressed_bt.trades]
            stressed_dates = [t.entry_time for t in stressed_bt.trades]
            stressed_single_run = simulate_account(stressed_pnls, stressed_dates, prop_rules)
            stressed_mc = run_monte_carlo(stressed_bt.trades, prop_rules, mc_cfg)
            stressed_fitness = compute_fitness(
                stressed_bt.statistics.to_dict(), summarize_single_run(stressed_single_run), stressed_mc, metric,
            )
        else:
            stressed_fitness = float("-inf")
        fitness = apply_cost_stress_penalty(fitness, stressed_fitness, cost_stress_penalty_weight)

    return (
        fitness,
        bt_result.statistics.to_dict(),
        prop_summary,
        _mc_summary(mc_result),
        bt_result if keep_full else None,
        mc_result if keep_full else None,
        single_run if keep_full else None,
    )


# ---------------------------------------------------------------------------
# Genetic operators (source-type agnostic: only touch .lo/.hi/.is_int/.base_value,
# which GeneMeta and CodeGene both expose)
# ---------------------------------------------------------------------------

def _random_gene_value(gene, rng: random.Random) -> float:
    v = rng.uniform(gene.lo, gene.hi)
    return float(round(v)) if gene.is_int else float(v)


def _crossover(genome_a: list, genome_b: list, rng: random.Random) -> list:
    return [genome_a[i] if rng.random() < 0.5 else genome_b[i] for i in range(len(genome_a))]


def _mutate(genome: list, genes: list, rate: float, strength: float, rng: random.Random) -> list:
    out = list(genome)
    for i, gene in enumerate(genes):
        if rng.random() < rate:
            span = gene.hi - gene.lo
            delta = rng.uniform(-strength, strength) * span
            v = min(max(out[i] + delta, gene.lo), gene.hi)
            out[i] = float(round(v)) if gene.is_int else float(v)
    return out


def _neighbor_genomes(genome: list, genes: list, step_frac: float) -> list[list]:
    """One perturbed genome per (gene, direction) pair: nudge that single
    gene by +/- step_frac of its own search range, clamped to [lo, hi] and
    rounded for integer genes, leaving every other gene untouched. This
    probes the immediate neighborhood one dimension at a time (2 * len(genes)
    points) rather than a single random perturbation, so a spike that's
    fragile in even one parameter's direction is caught."""
    out: list[list] = []
    for i, gene in enumerate(genes):
        span = gene.hi - gene.lo
        if span <= 0:
            continue
        step = step_frac * span
        for sign in (1.0, -1.0):
            v = min(max(genome[i] + sign * step, gene.lo), gene.hi)
            v = float(round(v)) if gene.is_int else float(v)
            if v == genome[i]:
                continue
            neighbor = list(genome)
            neighbor[i] = v
            out.append(neighbor)
    return out


def _plateau_robustness_score(center_fitness: float, neighbor_fitnesses: list[float]) -> float:
    """Higher is more plateau-like: rewards a neighborhood that scores well
    on average, penalized by how much it varies. A thin spike has a high
    center_fitness but neighbors that collapse -- low mean, high spread --
    and scores worse here than a slightly-lower, flatter region despite
    having the higher raw fitness."""
    values = [v for v in ([center_fitness] + neighbor_fitnesses) if math.isfinite(v)]
    if not values:
        return float("-inf")
    mean_v = sum(values) / len(values)
    if len(values) < 2:
        return mean_v
    variance = sum((v - mean_v) ** 2 for v in values) / len(values)
    return mean_v - (variance ** 0.5)


def _select_plateau_robust(
    candidates: list[Candidate],
    genes: list,
    cfg: RefinementConfig,
    evaluate_cheap,
) -> tuple[Candidate, dict | None]:
    """Picks a plateau-robust champion from every finite candidate the
    search evaluated. Returns (chosen_candidate, report_dict | None) --
    report_dict is None only when there was nothing finite to choose from
    (evaluate_cheap already handles that case by returning the raw best)."""
    finite = [c for c in candidates if math.isfinite(c.fitness)]
    if not finite:
        raw_best = max(candidates, key=lambda c: c.fitness)
        return raw_best, None

    raw_best = max(finite, key=lambda c: c.fitness)

    # De-duplicate by genome so an identical genome re-evaluated across
    # generations (e.g. via elitism) isn't scored as its own finalist twice.
    seen: set[tuple] = set()
    pool: list[Candidate] = []
    for c in sorted(finite, key=lambda c: c.fitness, reverse=True):
        key = tuple(round(v, 6) for v in c.genome)
        if key in seen:
            continue
        seen.add(key)
        pool.append(c)
        if len(pool) >= cfg.plateau_finalist_pool:
            break

    best_candidate = raw_best
    best_score = float("-inf")
    for candidate in pool:
        neighbor_genomes = _neighbor_genomes(candidate.genome, genes, cfg.plateau_neighbor_step_frac)
        neighbor_fitnesses = [evaluate_cheap(g).fitness for g in neighbor_genomes]
        score = _plateau_robustness_score(candidate.fitness, neighbor_fitnesses)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    report = {
        "raw_best_fitness": raw_best.fitness,
        "chosen_fitness": best_candidate.fitness,
        "swapped": tuple(round(v, 6) for v in best_candidate.genome) != tuple(round(v, 6) for v in raw_best.genome),
        "neighbor_step_frac": cfg.plateau_neighbor_step_frac,
        "finalist_pool_size": len(pool),
    }
    return best_candidate, report


def _tournament_select(population: list[Candidate], rng: random.Random, k: int = 3) -> Candidate:
    k = min(k, len(population))
    contenders = rng.sample(population, k)
    return max(contenders, key=lambda c: c.fitness)


def _diversity(population: list[Candidate], genes: list) -> float:
    if not genes or len(population) < 2:
        return 0.0
    total = 0.0
    for i, gene in enumerate(genes):
        span = (gene.hi - gene.lo) or 1.0
        vals = [c.genome[i] for c in population]
        mean_v = sum(vals) / len(vals)
        var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
        total += (var ** 0.5) / span
    return total / len(genes)


def _summarize_generation(gen: int, population: list[Candidate], genes: list) -> GenerationSummary:
    finite = [c.fitness for c in population if math.isfinite(c.fitness)]
    best = max((c.fitness for c in population), default=float("-inf"))
    mean = (sum(finite) / len(finite)) if finite else float("-inf")
    worst = min(finite) if finite else float("-inf")
    return GenerationSummary(
        generation=gen, best_fitness=best, mean_fitness=mean, worst_fitness=worst,
        diversity=_diversity(population, genes),
    )


# ---------------------------------------------------------------------------
# Explicit optimizer modes -- TPE (Optuna) and CMA-ES.
#
# Both take the SAME inputs (genes, an `evaluate(genome, generation) ->
# Candidate` closure already bound to _evaluate/run_backtest, and the same
# RefinementConfig) and return the SAME shape the genetic-mode loop already
# produces: (final_population, generation_history, best_ever). Everything
# downstream of the search loop in run_iterative_refinement -- plateau-
# robust selection, the cost-stress penalty already baked into `fitness`,
# the final full-fidelity re-evaluation, the holdout check -- is untouched
# by which of the three ran, because all three only ever call the same
# `evaluate` closure to decide what a genome scores. Neither library is a
# hard dependency of the rest of the app -- see ml_classifier_gbdt_
# direction.py's identical lazy-import pattern for lightgbm.
# ---------------------------------------------------------------------------

def _genome_to_dict(genome: list, genes: list) -> dict:
    return {gene.label: value for gene, value in zip(genes, genome)}


def _run_genetic_search(
    genes: list, evaluate: Callable, cfg: "RefinementConfig", log: ProgressCallback,
    rng: random.Random, baseline: Candidate,
) -> tuple[list, list["GenerationSummary"], Candidate]:
    """The original genetic algorithm, extracted verbatim (no behavior
    change -- see tests/test_refinement.py's full existing suite, which
    covers this exact code unchanged) so app.optimize.multi_market_refinement
    can reuse it for multi-market aggregate scoring instead of duplicating
    it. Takes the already-evaluated `baseline` candidate (tracked by the
    caller before this runs, same as every mode) and returns (population,
    generation_history, best_ever) -- exactly what run_iterative_refinement
    used to build inline."""
    population: list[Candidate] = [baseline]
    while len(population) < cfg.population_size:
        population.append(evaluate([_random_gene_value(g, rng) for g in genes], 0))

    best_ever = max(population, key=lambda c: c.fitness)
    generation_history: list[GenerationSummary] = [_summarize_generation(0, population, genes)]
    log(
        f"Generation 0 (initial population of {cfg.population_size}): "
        f"best={generation_history[0].best_fitness:.3f}  mean={generation_history[0].mean_fitness:.3f}"
    )

    for gen in range(1, cfg.generations + 1):
        population.sort(key=lambda c: c.fitness, reverse=True)
        elites = population[: cfg.elite_count]
        next_pop: list[Candidate] = list(elites)

        n_immigrants = max(1, round(cfg.population_size * cfg.random_immigrants_frac))
        n_bred = max(cfg.population_size - len(elites) - n_immigrants, 0)

        for _ in range(n_bred):
            parent_a = _tournament_select(population, rng)
            parent_b = _tournament_select(population, rng)
            child_genome = _crossover(parent_a.genome, parent_b.genome, rng)
            child_genome = _mutate(child_genome, genes, cfg.mutation_rate, cfg.mutation_strength, rng)
            next_pop.append(evaluate(child_genome, gen))

        while len(next_pop) < cfg.population_size:
            next_pop.append(evaluate([_random_gene_value(g, rng) for g in genes], gen))

        population = next_pop
        gen_summary = _summarize_generation(gen, population, genes)
        generation_history.append(gen_summary)

        gen_best = max(population, key=lambda c: c.fitness)
        if gen_best.fitness > best_ever.fitness:
            best_ever = gen_best

        log(
            f"Generation {gen}/{cfg.generations}: best={gen_summary.best_fitness:.3f}  "
            f"mean={gen_summary.mean_fitness:.3f}  diversity={gen_summary.diversity:.3f}  "
            f"(best-ever={best_ever.fitness:.3f})"
        )
    return population, generation_history, best_ever


def _run_tpe_search(
    genes: list, evaluate: Callable, cfg: "RefinementConfig", log: ProgressCallback,
) -> tuple[list["GenerationSummary"], int]:
    """Runs TPE search via the SAME `evaluate` closure the genetic loop
    uses. Returns (generation_history, last_batch_size) -- NOT the
    Candidate objects themselves, since `evaluate` already appends every
    one of them, in completion order, to the caller's own all_evaluated
    list; the caller recovers the final "generation" as
    all_evaluated[-last_batch_size:], exactly mirroring what the genetic
    loop's own `population` variable holds at the end of its last
    generation."""
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

    batch: list[Candidate] = []
    generation_history: list[GenerationSummary] = []
    gen_counter = [0]

    def objective(trial: "optuna.Trial") -> float:
        genome = []
        for gene in genes:
            if gene.is_int:
                v = trial.suggest_int(gene.label, int(round(gene.lo)), int(round(gene.hi)))
            else:
                v = trial.suggest_float(gene.label, float(gene.lo), float(gene.hi))
            genome.append(float(v))
        candidate = evaluate(genome, gen_counter[0])
        batch.append(candidate)
        if len(batch) >= cfg.population_size:
            generation_history.append(_summarize_generation(gen_counter[0], list(batch), genes))
            log(
                f"TPE batch {gen_counter[0]}/{cfg.generations}: "
                f"best={generation_history[-1].best_fitness:.3f}  mean={generation_history[-1].mean_fitness:.3f}"
            )
            batch.clear()
            gen_counter[0] += 1
        return candidate.fitness if math.isfinite(candidate.fitness) else -1e18

    total_trials = cfg.population_size * (cfg.generations + 1)
    study.optimize(objective, n_trials=total_trials, show_progress_bar=False)
    if batch:
        generation_history.append(_summarize_generation(gen_counter[0], list(batch), genes))
    last_batch_size = min(cfg.population_size, total_trials)
    return generation_history, last_batch_size


def _run_cma_es_search(
    genes: list, evaluate: Callable, cfg: "RefinementConfig", log: ProgressCallback,
) -> tuple[list["GenerationSummary"], int]:
    """Same contract as _run_tpe_search above: returns (generation_history,
    last_batch_size), not Candidate objects."""
    try:
        import cma
    except ImportError as exc:
        raise RefinementError(
            "optimizer_mode='cma_es' requires the cma package (pip install cma) -- "
            "it isn't a hard dependency of the rest of this app, only of CMA-ES search."
        ) from exc

    x0 = [(gene.lo + gene.hi) / 2.0 for gene in genes]
    spans = [(gene.hi - gene.lo) or 1.0 for gene in genes]
    sigma0 = 0.3  # in the same normalized-by-span units bounds/scaling below use
    # cma treats every dimension as sharing one sigma unless told otherwise --
    # normalize each gene to a 0..1 range internally and rescale on the way
    # out, so genes with very different spans (e.g. an RSI period 2-30 next
    # to an ATR stop multiplier 0.5-5) don't get a badly mismatched step size.
    es = cma.CMAEvolutionStrategy(
        [0.5] * len(genes), sigma0,
        {"bounds": [0.0, 1.0], "popsize": cfg.population_size, "verbose": -9,
         # UPGRADE (reproducibility bug fix, found and root-caused via a
         # sibling module's test failure -- see app.optimize.multi_market
         # for the full investigation): "seed" alone seeds numpy's GLOBAL
         # random state at construction time, but cma's default "randn"
         # option is np.random.randn -- the SAME global function -- so
         # every ask() after the first draws from whatever state numpy's
         # global RNG happens to be in by then. The backtest/Monte Carlo
         # pass this loop runs between every ask()/tell() pair is exactly
         # the kind of code that disturbs it, so "seed" alone silently
         # broke reproducibility for any search that ran more than one
         # generation. An isolated np.random.RandomState instance, passed
         # as "randn" instead, gives CMA-ES a private generator nothing
         # else in this process can perturb -- verified by deliberately
         # disturbing numpy's global state between ask() calls in testing
         # and confirming the sequence stayed identical anyway.
         "randn": np.random.RandomState(cfg.random_seed or 0).randn},
    )

    def _denormalize(unit_genome: list) -> list:
        out = []
        for u, gene in zip(unit_genome, genes):
            v = gene.lo + max(0.0, min(1.0, u)) * (gene.hi - gene.lo)
            out.append(float(round(v)) if gene.is_int else float(v))
        return out

    generation_history: list[GenerationSummary] = []
    last_batch_size = cfg.population_size
    gen = 0
    while gen <= cfg.generations and not es.stop():
        solutions = es.ask()
        batch_candidates: list[Candidate] = []
        penalties: list[float] = []
        for unit_genome in solutions:
            genome = _denormalize(unit_genome)
            candidate = evaluate(genome, gen)
            batch_candidates.append(candidate)
            # cma minimizes -- feed it the negative fitness. A non-finite
            # fitness (zero trades) becomes a large-but-finite penalty
            # rather than +/-inf, which cma's internal covariance update
            # cannot handle.
            penalties.append(-candidate.fitness if math.isfinite(candidate.fitness) else 1e12)
        es.tell(solutions, penalties)
        generation_history.append(_summarize_generation(gen, batch_candidates, genes))
        log(
            f"CMA-ES generation {gen}/{cfg.generations}: "
            f"best={generation_history[-1].best_fitness:.3f}  mean={generation_history[-1].mean_fitness:.3f}"
        )
        last_batch_size = len(batch_candidates)
        gen += 1
    return generation_history, last_batch_size


def _compute_distribution_summary(all_evaluated: list[Candidate], cfg: "RefinementConfig") -> dict | None:
    """"Distributions, not just a winner": median vs. best across every
    candidate this run actually evaluated, for the metrics computed for
    EVERY candidate already (not just the final champion) -- the cheap
    per-candidate Monte Carlo pass's eval/first-payout probability, and
    that candidate's own backtest max drawdown. Deliberately does NOT
    report an OOS-pass or regime-stability distribution: neither is
    computed per-candidate inside this search loop (only once, at the
    very end, for the single chosen champion via the holdout check) --
    fabricating a per-candidate number for either here would be exactly
    the kind of invented statistic this app avoids elsewhere.
    """
    finite = [c for c in all_evaluated if math.isfinite(c.fitness) and c.mc_summary]
    if not finite:
        return None

    def _pctile(values: list[float], p: float) -> float:
        s = sorted(values)
        if not s:
            return float("nan")
        k = (len(s) - 1) * p
        f, c = math.floor(k), math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] + (s[c] - s[f]) * (k - f)

    eval_pass = [c.mc_summary["evaluation_pass_probability"] for c in finite]
    payout_pass = [c.mc_summary["first_payout_probability"] for c in finite]
    max_dd = [
        c.statistics.get("max_drawdown_pct") for c in finite
        if c.statistics and c.statistics.get("max_drawdown_pct") is not None
    ]
    fitnesses = [c.fitness for c in finite]

    # "Robust candidates" -- a deliberately simple, transparent bar: how
    # many DIFFERENT-GENOME candidates this run evaluated cleared >=80% of
    # the raw best fitness found anywhere. Not a re-implementation of
    # plateau-robust selection's own neighborhood-perturbation logic
    # (that answers "does this ONE candidate hold up to nudging its own
    # parameters"); this instead answers "how many genuinely different
    # points in the space this run tried are almost as good as the best
    # one" -- a rough proxy for how wide/flat the good region of the
    # space is, not a claim of statistical robustness.
    best_fitness = max(fitnesses)
    robust_threshold = best_fitness * 0.8 if best_fitness > 0 else best_fitness * 1.2
    robust_count = sum(1 for f in fitnesses if f >= robust_threshold)

    return {
        "candidates_tested": len(all_evaluated),
        "candidates_with_trades": len(finite),
        "eval_pass_probability": {
            "median": _pctile(eval_pass, 0.5), "best": max(eval_pass),
        },
        "first_payout_probability": {
            "median": _pctile(payout_pass, 0.5), "best": max(payout_pass),
        },
        "max_drawdown_pct": (
            {"median": _pctile(max_dd, 0.5), "best": min(max_dd)} if max_dd else None
        ),
        "robust_candidates": robust_count,
        "robust_threshold_fraction_of_best": 0.8 if best_fitness > 0 else 1.2,
    }


_NO_PARAMS_MESSAGE = {
    "manual": (
        "No tunable numeric parameters were found in this strategy configuration. "
        "Iterative Refinement optimizes Manual Strategy Builder parameters "
        "(indicator periods, comparison thresholds, stop loss / take profit / "
        "trailing stop / break-even values). Add at least one indicator-based "
        "condition, or a Fixed or ATR-based stop/target, and try again."
    ),
    "python": (
        "No tunable numeric parameters were found in this Python strategy. Iterative "
        "Refinement optimizes any top-level SCREAMING_SNAKE_CASE numeric constant "
        "(e.g. `EMA_FAST = 10`, `STOP_LOSS_PIPS = 20`) -- add at least one such "
        "constant and reference it inside generate_signals() to make it tunable."
    ),
    "pinescript": (
        "No tunable numeric parameters were found in this PineScript strategy. "
        "Iterative Refinement optimizes input.int()/input.float() values and the "
        "// T58_SL_PIPS= / // T58_TP_PIPS= directives -- add at least one of these."
    ),
    "mql5": (
        "No tunable numeric parameters were found in this MQL5 strategy. Iterative "
        "Refinement optimizes literal iMA()/iRSI() period arguments and the "
        "// T58_SL_PIPS= / // T58_TP_PIPS= directives -- add at least one of these."
    ),
}


def _build_adapter(strategy: Strategy, tmp_dir: Path | None):
    """
    Returns (genes, build_fn) where build_fn(genome) -> a fresh Strategy
    instance of the same source type as `strategy` with that genome applied.
    """
    source_type = strategy.source_type
    if source_type == "manual":
        genes = extract_genome(strategy.config)

        def build(genome: list):
            return ManualStrategy(apply_genome(strategy.config, genes, genome))

        return genes, build

    if source_type in CODE_SOURCE_TYPES:
        genes = discover_code_genes(strategy)

        def build(genome: list):
            return materialize_code_strategy(strategy, genes, genome, tmp_dir)

        return genes, build

    raise RefinementError(
        f"Iterative Refinement does not support strategy source type '{source_type}'."
    )


def preflight_signal_check(
    df: pd.DataFrame, strategy: Strategy, risk: RiskConfig, feature_name: str,
    adaptive_risk: AdaptiveRiskConfig | None = None,
) -> None:
    """
    Runs ONE cheap, unmodified backtest of `strategy` on the FULL `df`
    before any fold-splitting or GA/NSGA-II search begins, and raises a
    clear RefinementError if it produces zero trades.

    Why this exists: Walk-Forward Optimization, Multi-Objective search,
    and the Walk-Forward-Aware GA all score every candidate (across every
    fold and every generation) the same way -- backtest it and read off
    stats. If the UNMODIFIED baseline strategy already produces zero
    trades on the WHOLE dataset, every single candidate downstream is
    guaranteed to also produce zero trades (folds are strict subsets of
    the same data, and no amount of numeric-parameter tuning fixes a
    strategy that structurally never fires on this data/timeframe). The
    old behavior was to grind through every fold and every generation
    anyway and hand back a report that's all zeros / -inf / "infeasible"
    with no explanation -- expensive AND confusing. This catches it in
    under a second, before any of that work starts.

    This is deliberately NOT run for Iterative Refinement's own baseline
    (run_iterative_refinement already computes and reports that baseline
    as part of its normal flow) -- only for the three heavier fold/
    population-based searches that would otherwise waste real time
    re-discovering the same "zero trades" fact many times over.
    """
    try:
        bt = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)
    except Exception:
        # Let the caller's own error handling deal with a strategy that
        # can't even run once -- this check is only about "runs fine but
        # never fires," not about strategies that crash outright.
        return

    if bt.trades:
        return
    raise RefinementError(
        f"{feature_name} can't proceed: the strategy, unmodified, produced "
        f"ZERO trades on the entire dataset ({len(df)} bars, "
        f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}) before any "
        f"optimization even began. Every fold and every candidate downstream "
        f"would also score zero trades -- that's not a search-quality problem, "
        f"it's this strategy never firing on this data at all, so the search "
        f"was stopped instead of grinding through folds/generations for a "
        f"guaranteed-empty result.\n\n"
        f"Common causes, roughly in order of likelihood:\n"
        f"  - The strategy filters entries to specific hours-of-day (a London/"
        f"NY \"session\" window) but this data is daily bars or otherwise has "
        f"no real intraday hour information -- every bar's hour is constant, "
        f"so an hour-of-day filter excludes 100% of bars. Check the strategy "
        f"source for an hour/session filter if this data isn't intraday.\n"
        f"  - Not enough bars for the strategy's slowest indicator to warm up "
        f"(e.g. a 200-period moving average on a dataset with only a few "
        f"hundred bars).\n"
        f"  - The data's price scale, symbol, or timeframe doesn't match what "
        f"the strategy was written/tuned for.\n"
        f"  - The strategy's entry conditions are just very strict for this "
        f"particular instrument/period.\n\n"
        f"Try running a plain Run & Report (Step 5) on this same data/strategy "
        f"pair first -- if that also shows 0 trades, the fix is in the "
        f"strategy or the data pairing, not in this search."
    )



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_iterative_refinement(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    refinement_config: RefinementConfig,
    progress_cb: ProgressCallback | None = None,
    adaptive_risk: AdaptiveRiskConfig | None = None,
) -> RefinementResult:
    """
    strategy: an already-built Strategy instance (ManualStrategy,
    PythonStrategy, PineScriptStrategy, or MQL5Strategy) -- e.g. whatever
    the UI/CLI already constructed for the normal run.
    mc_config: the SAME MonteCarloConfig used for the main pipeline run --
    its n_simulations is used for the baseline and the final best-candidate
    evaluation; the search phase uses refinement_config.search_monte_carlo_sims
    instead, for speed.
    adaptive_risk: same declarative, limit-aware position-sizing overlay
    accepted by Quick Optimize and Full Pipeline (see
    app.backtest.adaptive_risk.build_limit_aware_preset) -- None/omitted
    runs exactly as before this parameter existed. Applied to the
    baseline backtest, every GA candidate, and the final full-fidelity
    re-run, so a result produced here is only comparable to a Quick
    Optimize/Full Pipeline result made with the SAME setting -- mixing
    an enabled run here with a disabled one there (or vice versa) is not
    an apples-to-apples comparison, same caveat that already exists on
    those other two tools' "Enable adaptive, limit-aware position
    sizing" checkboxes.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if adaptive_risk is not None and adaptive_risk.enabled:
        log(f"Adaptive risk enabled: {len(adaptive_risk.rules)} limit-aware throttle rule(s) applied.")

    if strategy.source_type not in SUPPORTED_SOURCE_TYPES:
        raise RefinementError(
            f"Iterative Refinement does not support strategy source type "
            f"'{strategy.source_type}'."
        )

    t0 = time.time()
    cfg = refinement_config
    warnings: list[str] = []
    source_type = strategy.source_type

    tmp_dir: Path | None = None
    if source_type == "python":
        # PythonStrategy only accepts a file path, so mutated candidates are
        # written to throwaway temp .py files here, cleaned up in `finally`.
        tmp_dir = Path(tempfile.mkdtemp(prefix="t58_refine_"))

    try:
        genes, build = _build_adapter(strategy, tmp_dir)
        if not genes:
            raise RefinementError(_NO_PARAMS_MESSAGE.get(source_type, _NO_PARAMS_MESSAGE["manual"]))

        rng = random.Random(cfg.random_seed)
        search_mc_cfg = replace(mc_config, n_simulations=cfg.search_monte_carlo_sims)

        def snapshot(genome: list) -> tuple[dict | None, str | None, str | None]:
            """Only computed for the baseline and best-ever candidate (see keep_full)."""
            if source_type == "manual":
                return apply_genome(strategy.config, genes, genome), None, None
            code_text, ext = patched_source_for_strategy(strategy, genes, genome)
            return None, code_text, ext

        all_evaluated: list[Candidate] = []

        def evaluate(genome: list, generation: int, keep_full: bool = False, track: bool = True) -> Candidate:
            candidate_strategy = build(genome)
            fitness, stats, prop_summary, mc_summary, bt_full, mc_full, single_full = _evaluate(
                df, candidate_strategy, risk, prop_rules, search_mc_cfg, cfg.fitness_metric, keep_full=keep_full,
                cost_stress_multiplier=cfg.cost_stress_multiplier if cfg.cost_stress_enabled else None,
                cost_stress_penalty_weight=cfg.cost_stress_penalty_weight if cfg.cost_stress_enabled else 0.0,
                adaptive_risk=adaptive_risk,
            )
            config, code_text, code_ext = (None, None, None)
            if keep_full:
                config, code_text, code_ext = snapshot(genome)
            result = Candidate(
                generation=generation, genome=list(genome), fitness=fitness, source_type=source_type,
                config=config, code_text=code_text, code_extension=code_ext,
                statistics=stats, prop_summary=prop_summary, mc_summary=mc_summary,
                bt_result=bt_full, mc_result=mc_full, single_run=single_full,
            )
            # `track=False` is used for the plateau-robustness probes below --
            # those are cheap neighborhood checks around already-evaluated
            # finalists, not new search points, so they don't belong in the
            # finalist pool themselves (nor would re-adding them there change
            # anything, since de-duplication is by genome).
            if track:
                all_evaluated.append(result)
            return result

        log(f"Analyzing strategy parameters... found {len(genes)} tunable parameter(s).")

        baseline_genome = [g.base_value for g in genes]
        baseline = evaluate(baseline_genome, 0, keep_full=True)
        if not math.isfinite(baseline.fitness):
            warnings.append(
                "The current (baseline) configuration produced no trades on this data "
                "-- there is no known-good baseline to compare the search against."
            )
        log(f"Baseline fitness ({FITNESS_METRICS[cfg.fitness_metric]}): {baseline.fitness:.3f}")

        if cfg.optimizer_mode == "genetic":
            population, generation_history, best_ever = _run_genetic_search(genes, evaluate, cfg, log, rng, baseline)
        else:
            # TPE / CMA-ES -- see OPTIMIZER_MODES and _run_tpe_search /
            # _run_cma_es_search's own docstrings. Both call the SAME
            # `evaluate` closure as the genetic branch above, so `baseline`
            # (already evaluated and tracked before this if/else) and every
            # trial they run land in `all_evaluated` in one consistent
            # order regardless of mode.
            mode_label = OPTIMIZER_MODES[cfg.optimizer_mode].split(" -- ")[0]
            planned = cfg.population_size * (cfg.generations + 1)
            log(f"Running {mode_label} search ({planned} evaluations planned)...")
            if cfg.optimizer_mode == "tpe":
                generation_history, last_batch_size = _run_tpe_search(genes, evaluate, cfg, log)
            else:
                generation_history, last_batch_size = _run_cma_es_search(genes, evaluate, cfg, log)
            population = all_evaluated[-last_batch_size:] if last_batch_size else list(all_evaluated)
            best_ever = max(all_evaluated, key=lambda c: c.fitness)

        plateau_report: dict | None = None
        if cfg.plateau_robust_selection:
            log(
                f"Checking the top {min(cfg.plateau_finalist_pool, len(all_evaluated))} candidate(s) "
                "for neighborhood-robust (plateau) selection..."
            )
            evaluate_cheap = lambda genome: evaluate(genome, generation=-1, keep_full=False, track=False)  # noqa: E731
            best_ever, plateau_report = _select_plateau_robust(all_evaluated, genes, cfg, evaluate_cheap)
            if plateau_report is not None and plateau_report["swapped"]:
                log(
                    f"Raw optimum (fitness={plateau_report['raw_best_fitness']:.3f}) looks like a thin spike; "
                    f"using the plateau-robust pick instead (fitness={plateau_report['chosen_fitness']:.3f}, "
                    f"+/-{plateau_report['neighbor_step_frac']:.0%} per parameter holds up)."
                )
                warnings.append(
                    "Iterative Refinement's raw best-scoring configuration sat on a narrow parameter "
                    "spike (a small nudge to at least one parameter meaningfully hurt its score). The "
                    "configuration actually returned is a nearby, neighborhood-robust pick instead -- "
                    "see plateau_robustness in the result for both fitness values. Disable via "
                    "plateau_robust_selection=False to restore the old raw-best behavior."
                )

        log("Running full-fidelity Monte Carlo on the best-ever configuration...")
        best_strategy = build(best_ever.genome)
        final_fitness, final_stats, final_prop, final_mc_summary, final_bt, final_mc, final_single = _evaluate(
            df, best_strategy, risk, prop_rules, mc_config, cfg.fitness_metric, keep_full=True,
            cost_stress_multiplier=cfg.cost_stress_multiplier if cfg.cost_stress_enabled else None,
            cost_stress_penalty_weight=cfg.cost_stress_penalty_weight if cfg.cost_stress_enabled else 0.0,
            adaptive_risk=adaptive_risk,
        )
        final_config, final_code_text, final_code_ext = snapshot(best_ever.genome)
        best_final = Candidate(
            generation=best_ever.generation, genome=best_ever.genome, fitness=final_fitness,
            source_type=source_type, config=final_config, code_text=final_code_text, code_extension=final_code_ext,
            statistics=final_stats, prop_summary=final_prop, mc_summary=final_mc_summary,
            bt_result=final_bt, mc_result=final_mc, single_run=final_single,
        )

        holdout_comparison = None
        if final_bt is not None and final_bt.trades:
            log("Running out-of-sample holdout check on the optimized configuration...")
            try:
                holdout_comparison = run_holdout_comparison(
                    df, build(best_ever.genome), risk, holdout_frac=0.2,
                )
            except Exception:
                warnings.append(
                    "Holdout check on the optimized configuration could not be "
                    "completed (not enough data to split)."
                )

        leaderboard = sorted(population, key=lambda c: c.fitness, reverse=True)
        elapsed = time.time() - t0
        distribution_summary = _compute_distribution_summary(all_evaluated, cfg)
        log(f"Iterative Refinement complete in {elapsed:.1f}s ({len(all_evaluated)} candidates evaluated).")

        return RefinementResult(
            refinement_config=cfg,
            fitness_metric=cfg.fitness_metric,
            source_type=source_type,
            genes=genes,
            baseline=baseline,
            best=best_final,
            generation_history=generation_history,
            leaderboard=leaderboard,
            holdout_comparison=holdout_comparison,
            elapsed_seconds=elapsed,
            warnings=warnings,
            plateau_robustness=plateau_report,
            total_evaluations=len(all_evaluated),
            distribution_summary=distribution_summary,
        )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
