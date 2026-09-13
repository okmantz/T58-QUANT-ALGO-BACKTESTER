"""
Strategy Compare -- the direct, visual side-by-side comparison view Owen
asked for: pick 2-4 saved strategies, see their equity curves,
Monte-Carlo pass probability, and Prop Fitness scores in one place.

Before this file existed, "comparison" only happened implicitly: the
Research Agent's compare_strategies tool (app.ai.research_agent) does
something similar but is buried behind an AI chat turn, and the Family
Diversity report compares GA-generation siblings, not arbitrary
Strategy-Library picks. This is the direct screen someone would actually
reach for first.

Mirrors app.ai.research_agent._tool_compare_strategies' approach (backtest
each candidate fresh against the same dataset/risk config -- never trust
possibly-stale metadata from a strategy's last unrelated run) but adds
the two numbers that screen didn't compute: Monte Carlo eval-pass
probability and a Prop Fitness score, per candidate, using the same
app.evolution.prop_fitness.compute_prop_fitness formula the Evolution Lab
ranks candidates with elsewhere in the app.

HONEST SIMPLIFICATION: Prop Fitness normally also factors in parameter-
neighborhood robustness and walk-forward out-of-sample consistency (see
compute_prop_fitness's own docstring) -- both are expensive, multi-run
analyses that don't belong in a quick side-by-side compare click. Here
they're passed as None, which compute_prop_fitness treats as a neutral
0.5 for each rather than 0 or 1 -- meaning this view's Prop Fitness
numbers are directly comparable to EACH OTHER (all computed the exact
same simplified way) but will read differently from a number the same
strategy got out of a full Evolution Lab run, which has robustness/
walk-forward data. compare_strategies() labels every result with a
`prop_fitness_is_simplified: True` flag so no caller can present this as
if it were a Evolution-Lab-grade score in a report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.evolution.prop_fitness import compute_prop_fitness
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules
from app.search.strategy_space import build_strategy_from_spec
from app.strategy.library import load_strategy_text


class CompareError(Exception):
    """Raised for bad input (too few/many candidates) before any backtest runs."""


@dataclass
class CompareCandidateResult:
    strategy_type: str
    filename: str
    ok: bool
    error: str | None = None
    stats: dict = field(default_factory=dict)             # app.backtest.statistics summary
    equity_curve: list[float] = field(default_factory=list)  # downsampled, for charting
    eval_pass_probability_pct: float = 0.0
    first_payout_probability_pct: float = 0.0
    prop_fitness_score: float = 0.0
    prop_fitness_is_simplified: bool = True


def _downsample(values: list[float], max_points: int = 300) -> list[float]:
    if len(values) <= max_points:
        return values
    step = len(values) / max_points
    return [values[int(i * step)] for i in range(max_points)]


def compare_strategies(
    candidates: list[tuple[str, str]],   # [(strategy_type, filename), ...], 2-4 entries
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_n_simulations: int = 1000,
    tmp_dir: str | None = None,
) -> list[CompareCandidateResult]:
    if not (2 <= len(candidates) <= 4):
        raise CompareError(f"Compare 2-4 strategies at a time, got {len(candidates)}.")

    mc_cfg = MonteCarloConfig(n_simulations=mc_n_simulations)
    results: list[CompareCandidateResult] = []

    for strategy_type, filename in candidates:
        try:
            code_text = load_strategy_text(strategy_type, filename)
            spec = {"source_type": strategy_type, "code_text": code_text}
            strategy = build_strategy_from_spec(spec, tmp_dir=tmp_dir)
            bt = run_backtest(df, strategy, risk)
            stats = bt.statistics.to_dict()

            if not bt.trades:
                results.append(CompareCandidateResult(
                    strategy_type=strategy_type, filename=filename, ok=False,
                    error="Zero trades produced on this dataset -- nothing to compare.",
                ))
                continue

            mc_result = run_monte_carlo(bt.trades, prop_rules, mc_cfg)
            mc_summary = mc_result.to_dict()

            fitness = compute_prop_fitness(
                stats=stats, mc_summary=mc_summary,
                robustness_dict=None, walk_forward_dict=None,
                trade_pnls=[t.pnl for t in bt.trades],
            )

            equity = bt.equity_curve["equity"].tolist() if len(bt.equity_curve) else []

            results.append(CompareCandidateResult(
                strategy_type=strategy_type, filename=filename, ok=True,
                stats={k: stats.get(k) for k in (
                    "total_trades", "net_profit", "return_pct", "win_rate", "profit_factor",
                    "expectancy", "sharpe_ratio", "max_drawdown_pct", "max_losing_streak",
                )},
                equity_curve=_downsample(equity),
                eval_pass_probability_pct=mc_result.evaluation_pass_probability,
                first_payout_probability_pct=mc_result.first_payout_probability,
                prop_fitness_score=round(fitness.final_score, 2),
            ))
        except Exception as exc:  # noqa: BLE001 -- one bad candidate must not sink the whole comparison
            results.append(CompareCandidateResult(
                strategy_type=strategy_type, filename=filename, ok=False, error=str(exc),
            ))

    return results


def compare_to_dicts(results: list[CompareCandidateResult]) -> list[dict]:
    """Plain-dict form for JSON API responses / template rendering."""
    return [r.__dict__ for r in results]
