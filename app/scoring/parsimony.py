"""
Parsimony scoring -- item #24 of the pipeline reorg plan.

Rewards strategies that reach their results with FEWER unnecessary degrees
of freedom (indicators, entry/exit conditions, tunable numeric parameters),
without ever treating simplicity as automatically good or complexity as
automatically bad:

    "The ideal strategy is not necessarily the simplest possible strategy.
     It is the simplest strategy that demonstrates a genuine and
     sufficiently robust edge."

This module only COUNTS degrees of freedom and maps that count onto a
0-100 "higher is better" score -- it makes no judgment about whether the
strategy is actually profitable or robust. That's why it plugs into
app.scoring.t58_scorecard as a single, small-weight component (5 points)
rather than a gate: see that module's _WEIGHTS table.

Degrees of freedom, by source type:

    manual   -- number of indicators used, plus the number of individual
                entry/exit conditions (visual builder's entry_conditions /
                exit_conditions lists, or a rough count of AND/OR-joined
                terms for the older raw-expression long_entry/short_entry
                fields).
    python / pinescript / mql5
             -- number of tunable numeric parameters this app's own
                Iterative Refinement / GA already discovers for that
                source type (app.optimize.code_parameter_space) -- this
                deliberately reuses the exact same "what counts as a
                parameter" definition the optimizer already uses, so the
                parsimony score and the search space it's penalizing are
                always talking about the same thing.

Nothing here invents a new strategy-analysis capability; it only counts
what other modules already expose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.strategy.base import Strategy


@dataclass
class ParsimonyResult:
    score: float | None            # 0-100, higher = fewer unnecessary degrees of freedom; None if uncountable
    degrees_of_freedom: int
    breakdown: dict = field(default_factory=dict)   # what was counted -- shown in the evidence panel
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _count_legacy_expression_terms(expr: str | None) -> int:
    """Crude but honest count of independent conditions in an older
    raw-expression long_entry/short_entry/long_exit/short_exit string:
    every 'and'/'or' boundary is one more term being combined. An empty
    or missing expression counts as zero, not one -- 'no rule here'
    shouldn't cost a strategy a degree of freedom."""
    if not expr or not str(expr).strip():
        return 0
    parts = re.split(r"\s+(?:and|or)\s+", str(expr), flags=re.IGNORECASE)
    return len([p for p in parts if p.strip()])


def _manual_degrees_of_freedom(config: dict) -> tuple[int, dict]:
    indicators = config.get("indicators", []) or []
    n_indicators = len(indicators)

    entries = config.get("entry_conditions", {}) or {}
    exits = config.get("exit_conditions", {}) or {}
    n_visual_conditions = (
        len(entries.get("long", []) or [])
        + len(entries.get("short", []) or [])
        + len(exits.get("long", []) or [])
        + len(exits.get("short", []) or [])
    )

    n_legacy_conditions = (
        _count_legacy_expression_terms(config.get("long_entry"))
        + _count_legacy_expression_terms(config.get("short_entry"))
        + _count_legacy_expression_terms(config.get("long_exit"))
        + _count_legacy_expression_terms(config.get("short_exit"))
    )

    dof = n_indicators + n_visual_conditions + n_legacy_conditions
    return dof, {
        "indicators": n_indicators,
        "visual_conditions": n_visual_conditions,
        "legacy_conditions": n_legacy_conditions,
    }


def _code_degrees_of_freedom(strategy: Strategy) -> tuple[int, dict]:
    from app.optimize.code_parameter_space import (
        discover_mql5_parameters, discover_pinescript_parameters, discover_python_parameters,
    )
    genes: list = []
    if strategy.source_type == "python":
        genes = discover_python_parameters(strategy.file_path)
    elif strategy.source_type == "pinescript":
        genes = discover_pinescript_parameters(strategy.code)
    elif strategy.source_type == "mql5":
        genes = discover_mql5_parameters(strategy.code)
    return len(genes), {"tunable_parameters": len(genes)}


def compute_degrees_of_freedom(strategy: Strategy) -> tuple[int, dict]:
    """Returns (dof_count, breakdown_dict). Raises whatever the underlying
    discover_* / config access raises -- compute_parsimony() below is the
    best-effort wrapper callers should actually use."""
    if strategy.source_type == "manual":
        return _manual_degrees_of_freedom(getattr(strategy, "config", {}) or {})
    return _code_degrees_of_freedom(strategy)


# Scoring curve: 0 degrees of freedom (the ceiling, essentially never hit
# in practice) -> 100. Score falls off gently at first -- a strategy with
# 2-4 conditions is still "simple" by any reasonable trader's definition
# -- then more steeply past roughly _DOF_HALF_SCORE, where added degrees
# of freedom start buying curve-fit risk faster than they buy genuine
# expressiveness. This is a heuristic curve, not a statistically derived
# one -- same caveat as every *_score helper in t58_scorecard.py.
_DOF_HALF_SCORE = 6.0  # degrees of freedom at which the score crosses 50


def dof_to_score(dof: int) -> float:
    if dof <= 0:
        return 100.0
    ratio = dof / _DOF_HALF_SCORE
    score = 100.0 / (1.0 + ratio ** 1.5)
    return max(0.0, min(100.0, score))


def compute_parsimony(strategy: Strategy) -> ParsimonyResult:
    """Best-effort entry point -- never raises. A strategy type this
    module doesn't know how to count (or a malformed config) returns
    score=None (NOT 0 -- an uncountable strategy isn't the same as a
    maximally-complex one) so t58_scorecard's missing-aware weighting
    simply leaves parsimony out of that strategy's score instead of
    penalizing it for a counting failure."""
    try:
        dof, breakdown = compute_degrees_of_freedom(strategy)
    except Exception as exc:  # noqa: BLE001 -- counting degrees of freedom is a bonus signal, not core output
        return ParsimonyResult(
            score=None, degrees_of_freedom=0, breakdown={},
            notes=[f"Could not count degrees of freedom ({exc}) -- parsimony not scored for this strategy."],
        )
    score = dof_to_score(dof)
    breakdown_str = ", ".join(f"{k}={v}" for k, v in breakdown.items()) or "none counted"
    return ParsimonyResult(
        score=score, degrees_of_freedom=dof, breakdown=breakdown,
        notes=[f"{dof} counted degree(s) of freedom ({breakdown_str}) -> parsimony score {score:.1f}/100."],
    )
