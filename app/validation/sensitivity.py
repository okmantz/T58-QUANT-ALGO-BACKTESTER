"""
Parameter sensitivity sweeps and heatmaps.

app.search.robustness.parameter_neighborhood_robustness() already answers
"are nearby parameter values roughly as good?" with a single scalar
stability ratio -- useful for an automated pass/fail gate, but it doesn't
show WHERE a strategy's edge lives or WHAT the drop-off looks like. This
module produces the actual curves/grids so a person can look at them:

  compute_1d_sensitivity() -- for a chosen parameter, evaluate the
      strategy at N evenly-spaced values across +/- pct_range of its
      current value (clipped to that parameter's normal search bounds),
      holding every other parameter fixed. Flags a "cliff" wherever the
      metric drops sharply between adjacent steps, vs. a "plateau" where
      it degrades gradually.

  compute_2d_heatmap() -- the same idea for a PAIR of parameters at once,
      producing a 2D grid suitable for a heatmap chart. This is the more
      informative view when two parameters interact (e.g. a fast/slow
      moving-average pair), since a 1D sweep of either one alone can look
      robust while the pair together sits on a narrow ridge.

Works across all source types (manual/python/pinescript/mql5) via the
same gene discovery Iterative Refinement already uses.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig, build_run_context
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import _build_adapter, compute_fitness
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.strategy.base import Strategy


def _sweep_worker_count(n_items: int) -> int:
    """Bounded thread count for parallelizing independent sensitivity/
    heatmap evaluations. Threads, not processes: each evaluation builds
    its OWN fresh Strategy instance (see app.optimize.refinement.
    _build_adapter / materialize_python_strategy's uuid-named temp file)
    and never touches shared mutable state, and run_backtest/
    run_monte_carlo do enough real numpy/pandas work to release the GIL
    for a meaningful fraction of each call -- the same reasoning that
    made ThreadPoolExecutor a real win for Walk-Forward Opt's previously
    unparallelized loop. No memory-duplication guard is needed here (see
    app.orchestration.resource_guard's docstring on why ProcessPoolExecutor
    needs one): a thread pool shares the parent process's one copy of
    `df`, it doesn't multiply it per worker. Capped at 8 regardless of
    core count -- past that, thread-scheduling/GIL overhead on this kind
    of workload stops paying for itself.
    """
    return max(1, min(n_items, os.cpu_count() or 4, 8))


class _NullExecutor:
    """Trivial stand-in used when there's only one item to evaluate (a
    single value/cell) -- skips ThreadPoolExecutor's setup/teardown cost
    entirely rather than spinning up a pool for one item."""

    def map(self, fn, iterable):
        return [fn(x) for x in iterable]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _executor_for(n_items: int):
    if n_items <= 1:
        return _NullExecutor()
    return ThreadPoolExecutor(max_workers=_sweep_worker_count(n_items))


def _metric_for(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    metric: str,
) -> float:
    bt = run_backtest(df, strategy, risk)
    if not bt.trades:
        return float("-inf")
    if metric in ("net_profit", "profit_factor", "sharpe_ratio", "win_rate", "expectancy", "max_drawdown_pct"):
        stats = bt.statistics.to_dict()
        v = stats.get(metric, 0.0)
        if v == float("inf"):
            return 10.0
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0
    # Anything else (composite_prop_score, eval_pass_probability, ...) needs the full prop+MC pipeline.
    pnls = [t.pnl for t in bt.trades]
    dates = [t.entry_time for t in bt.trades]
    single_run = simulate_account(pnls, dates, prop_rules)
    mc = run_monte_carlo(bt.trades, prop_rules, mc_config)
    prop_summary = summarize_single_run(single_run)
    fitness = compute_fitness(stats := bt.statistics.to_dict(), prop_summary, mc, metric)
    return fitness if math.isfinite(fitness) else float("-inf")


@dataclass
class Sensitivity1DResult:
    gene_label: str
    base_value: float
    base_metric: float
    values: list
    metric_values: list
    metric: str
    max_pct_drop_between_adjacent_steps: float
    cliff_detected: bool
    cliff_threshold: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _sweep_values(base_value: float, lo: float, hi: float, is_int: bool, pct_range: float, n_steps: int) -> list[float]:
    span = max(abs(base_value) * pct_range, (hi - lo) * 0.02 if hi > lo else 1.0)
    sweep_lo = max(base_value - span, lo)
    sweep_hi = min(base_value + span, hi)
    if sweep_hi <= sweep_lo:
        sweep_lo, sweep_hi = lo, hi
    raw = np.linspace(sweep_lo, sweep_hi, max(n_steps, 3))
    if is_int:
        raw = sorted(set(int(round(v)) for v in raw))
        return [float(v) for v in raw]
    return [float(v) for v in raw]


def compute_base_metric(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    metric: str = "profit_factor",
    tmp_dir: Path | None = None,
) -> float:
    """The strategy's metric at its CURRENT (unperturbed) parameter
    values -- the same "no change" baseline both compute_1d_sensitivity
    and compute_2d_heatmap compute internally by default. Exposed as its
    own function so a caller that needs the SAME baseline for several
    sweeps/heatmaps in one pipeline (see
    app.validation.parameter_robustness.compute_parameter_robustness,
    which used to trigger this identical full backtest+Monte-Carlo run
    once per parameter PLUS once per heatmap pair -- 7+ redundant runs at
    default settings) can compute it exactly ONCE and pass it into every
    downstream call via their own base_metric= parameter, instead of
    quietly recomputing an identical result over and over.
    """
    risk = build_run_context(risk, prop_rules)
    genes, build = _build_adapter(strategy, tmp_dir)
    if not genes:
        raise RefinementError(
            "This strategy has no tunable numeric parameters."
        )
    return _metric_for(df, build([g.base_value for g in genes]), risk, prop_rules, mc_config, metric)


def compute_1d_sensitivity(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    metric: str = "profit_factor",
    pct_range: float = 0.5,
    n_steps: int = 9,
    max_params: int = 8,
    cliff_threshold_pct: float = 40.0,
    tmp_dir: Path | None = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    base_metric: Optional[float] = None,
) -> list[Sensitivity1DResult]:
    """
    Sweeps every tunable numeric parameter (up to max_params, in discovery
    order) independently across +/- pct_range of its current value,
    holding all other parameters at their base value.

    progress_cb (optional): called with a short human-readable string
    after each parameter's sweep finishes, e.g. "Sensitivity: swept
    ema_fast (2/6)." -- so a job page polling for status has something
    to show while this runs instead of going silent for however long the
    full sweep takes. Never required; a caller with no job to report to
    can simply omit it.

    base_metric (optional): the strategy's metric at its CURRENT
    (unperturbed) parameter values. Every one of the max_params sweeps
    needs this same number as its "no change" baseline, and it's an
    identical, deterministic computation every time (same df/strategy/
    risk/prop_rules/mc_config/metric) -- so by default this computes it
    ONCE up front and reuses it, rather than the pre-2026-09 behavior of
    silently recomputing an identical full backtest+Monte-Carlo run once
    per parameter (6 wasted evaluations at the default max_params=6).
    Pass this in explicitly when a caller (e.g. compute_parameter_
    robustness, which also needs the same baseline for its 2D heatmaps)
    has already computed it, to skip the computation here entirely.
    """
    risk = build_run_context(risk, prop_rules)
    genes, build = _build_adapter(strategy, tmp_dir)
    if not genes:
        raise RefinementError(
            "This strategy has no tunable numeric parameters to run a sensitivity sweep on."
        )

    active_genes = genes[:max_params]
    if base_metric is None:
        base_metric = _metric_for(df, build([g.base_value for g in genes]), risk, prop_rules, mc_config, metric)

    results: list[Sensitivity1DResult] = []
    for pi, (gi, gene) in enumerate(list(enumerate(genes))[:max_params]):
        values = _sweep_values(gene.base_value, gene.lo, gene.hi, gene.is_int, pct_range, n_steps)

        def _eval_one(v, _gi=gi):
            genome = [g.base_value for g in genes]
            genome[_gi] = v
            candidate = build(genome)
            return _metric_for(df, candidate, risk, prop_rules, mc_config, metric)

        with _executor_for(len(values)) as ex:
            metric_values = list(ex.map(_eval_one, values))

        finite_vals = [m for m in metric_values if math.isfinite(m)]
        max_drop_pct = 0.0
        for a, b in zip(metric_values, metric_values[1:]):
            if math.isfinite(a) and math.isfinite(b) and a > 0:
                drop = (a - b) / abs(a) * 100.0
                max_drop_pct = max(max_drop_pct, drop)
            elif math.isfinite(a) and not math.isfinite(b):
                max_drop_pct = max(max_drop_pct, 100.0)

        results.append(Sensitivity1DResult(
            gene_label=gene.label,
            base_value=gene.base_value,
            base_metric=base_metric,
            values=values,
            metric_values=metric_values,
            metric=metric,
            max_pct_drop_between_adjacent_steps=max_drop_pct,
            cliff_detected=max_drop_pct >= cliff_threshold_pct,
            cliff_threshold=cliff_threshold_pct,
        ))
        if progress_cb is not None:
            try:
                progress_cb(f"Sensitivity: swept {gene.label} ({pi + 1}/{len(active_genes)}).")
            except Exception:  # noqa: BLE001 -- a broken progress callback must never break the sweep
                pass
    return results


@dataclass
class Sensitivity2DResult:
    gene_a_label: str
    gene_b_label: str
    a_values: list
    b_values: list
    grid: list  # list[list[float]], grid[i][j] = metric at (a_values[i], b_values[j])
    base_metric: float
    metric: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def compute_2d_heatmap(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    gene_label_a: str,
    gene_label_b: str,
    metric: str = "profit_factor",
    pct_range: float = 0.5,
    n_steps: int = 7,
    tmp_dir: Path | None = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    base_metric: Optional[float] = None,
) -> Sensitivity2DResult:
    """
    progress_cb / base_metric: same purpose as compute_1d_sensitivity's
    own params of the same name -- an optional status callback (called
    once per completed grid ROW, e.g. "Heatmap ema_fast x ema_slow: row
    3/7."), and an optional precomputed baseline to skip a redundant
    identical backtest+Monte-Carlo run when the caller (e.g.
    compute_parameter_robustness) already has one on hand.
    """
    risk = build_run_context(risk, prop_rules)
    genes, build = _build_adapter(strategy, tmp_dir)
    if not genes:
        raise RefinementError("This strategy has no tunable numeric parameters.")

    label_to_idx = {g.label: i for i, g in enumerate(genes)}
    if gene_label_a not in label_to_idx or gene_label_b not in label_to_idx:
        raise RefinementError(
            f"Unknown parameter label(s). Available: {sorted(label_to_idx)}"
        )
    ia, ib = label_to_idx[gene_label_a], label_to_idx[gene_label_b]
    ga, gb = genes[ia], genes[ib]

    a_values = _sweep_values(ga.base_value, ga.lo, ga.hi, ga.is_int, pct_range, n_steps)
    b_values = _sweep_values(gb.base_value, gb.lo, gb.hi, gb.is_int, pct_range, n_steps)

    base_genome = [g.base_value for g in genes]
    if base_metric is None:
        base_metric = _metric_for(df, build(base_genome), risk, prop_rules, mc_config, metric)

    # PERFORMANCE (2026-09): this grid is n_steps x n_steps independent
    # full backtest+Monte-Carlo evaluations -- 49 of them at the default
    # n_steps=7 -- that used to run one at a time in a plain nested loop.
    # Flattening to (av, bv) pairs and evaluating a whole ROW concurrently
    # (bounded thread pool -- see _executor_for's docstring for why
    # threads, not processes, are safe here) is what actually cuts this
    # from "however long 49 sequential runs take" down to roughly
    # (49 / worker_count) runs' worth of wall-clock time.
    def _eval_cell(bv, _av=None):
        genome = list(base_genome)
        genome[ia] = _av
        genome[ib] = bv
        return _metric_for(df, build(genome), risk, prop_rules, mc_config, metric)

    grid: list[list[float]] = []
    for ri, av in enumerate(a_values):
        with _executor_for(len(b_values)) as ex:
            row = list(ex.map(lambda bv, _av=av: _eval_cell(bv, _av), b_values))
        grid.append(row)
        if progress_cb is not None:
            try:
                progress_cb(f"Heatmap {gene_label_a} x {gene_label_b}: row {ri + 1}/{len(a_values)}.")
            except Exception:  # noqa: BLE001 -- a broken progress callback must never break the sweep
                pass

    return Sensitivity2DResult(
        gene_a_label=gene_label_a,
        gene_b_label=gene_label_b,
        a_values=a_values,
        b_values=b_values,
        grid=grid,
        base_metric=base_metric,
        metric=metric,
    )


def list_tunable_parameters(strategy: Strategy, tmp_dir: Path | None = None) -> list[str]:
    """Convenience helper for a UI/CLI to populate a parameter picker."""
    genes, _build = _build_adapter(strategy, tmp_dir)
    return [g.label for g in genes]
