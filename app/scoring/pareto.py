"""
Pareto Frontier -- "there isn't one best strategy, there's a frontier of
tradeoffs."

Owen's ask: instead of a single PROP FITNESS number crowning one
"winner," find every candidate that isn't strictly beaten on every
objective by some other candidate (the non-dominated set), and label
points on that frontier Conservative / Balanced / Aggressive so a human
can pick a tradeoff instead of the pipeline picking one number for them.

This module is intentionally generic -- it operates on plain dicts of
named metrics, not on EvolutionCandidateRecord or any other app-specific
type, so it works the same way for Evolution Lab finalists, Search Lab
Stage 4 survivors, or Full Pipeline batch results. Callers decide which
metrics matter and which direction is "better" (maximize vs minimize).

Dominance: point A dominates point B if A is >= B on every maximized
metric and <= B on every minimized metric, with at least one strict
inequality. A is on the frontier if nothing dominates A. Ties (two
identical points) are all kept, not arbitrarily deduped -- a caller who
wants one representative per identical point can dedupe upstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ParetoPoint:
    key: Any                       # caller-supplied identifier (e.g. candidate_id)
    metrics: dict[str, float]       # the raw metric values this point was ranked on
    dominated: bool = False
    dominated_by: list = field(default_factory=list)   # keys of points that dominate this one
    label: str | None = None       # "Conservative" / "Balanced" / "Aggressive" / None

    def to_dict(self) -> dict:
        return {
            "key": self.key, "metrics": dict(self.metrics), "dominated": self.dominated,
            "dominated_by": list(self.dominated_by), "label": self.label,
        }


def _dominates(a: dict[str, float], b: dict[str, float], directions: dict[str, str]) -> bool:
    """True if `a` dominates `b`: at least as good on every objective,
    strictly better on at least one. `directions[metric]` is "max" or
    "min". Missing metrics on either side are treated as a tie on that
    metric (neither side gets credit), so a partially-missing point
    still participates instead of being silently excluded."""
    at_least_as_good = True
    strictly_better = False
    for m, direction in directions.items():
        av, bv = a.get(m), b.get(m)
        if av is None or bv is None:
            continue
        if direction == "min":
            av, bv = -av, -bv
        if av < bv:
            at_least_as_good = False
            break
        if av > bv:
            strictly_better = True
    return at_least_as_good and strictly_better


def compute_pareto_frontier(
    points: list[dict],
    metrics: dict[str, str],
    key_fn: Callable[[dict], Any] | None = None,
) -> list[ParetoPoint]:
    """points: list of plain dicts, each holding at least the keys named
    in `metrics` (values coercible to float; missing/non-numeric values
    are tolerated -- see _dominates). metrics: {metric_name: "max"|"min"}
    -- e.g. {"eval_pass_probability": "max", "max_drawdown_pct": "min"}.
    key_fn: extracts a caller-meaningful identifier from each point dict
    (defaults to the point's own "candidate_id", falling back to its
    index if that key is absent).

    Returns one ParetoPoint per input point, frontier members first
    (dominated=False), each also carrying who dominates it so a caller
    can explain "why wasn't this one on the frontier" rather than just
    dropping it silently.
    """
    key_fn = key_fn or (lambda p, _i=[0]: p.get("candidate_id"))
    parsed: list[ParetoPoint] = []
    for i, p in enumerate(points):
        vals: dict[str, float] = {}
        for m in metrics:
            v = p.get(m)
            try:
                vals[m] = float(v) if v is not None else None
            except (TypeError, ValueError):
                vals[m] = None
        key = key_fn(p) if key_fn(p) is not None else i
        parsed.append(ParetoPoint(key=key, metrics=vals))

    for i, a in enumerate(parsed):
        for j, b in enumerate(parsed):
            if i == j:
                continue
            if _dominates(b.metrics, a.metrics, metrics):
                a.dominated = True
                a.dominated_by.append(b.key)

    return parsed


def label_frontier(
    frontier: list[ParetoPoint],
    conservative_metric: str,
    aggressive_metric: str,
) -> list[ParetoPoint]:
    """Labels the NON-dominated subset of `frontier` (leaves dominated
    points unlabeled -- they aren't real choices) along a single axis
    from `conservative_metric` (highest = Conservative -- typically
    eval-pass probability or robustness, "safest bet") to
    `aggressive_metric` (highest = Aggressive -- typically raw return or
    payout size). Frontier members strictly between the two extremes are
    "Balanced". A frontier of 1 point is just labeled "Balanced" (there's
    no tradeoff to describe with one point); a frontier of 2 points
    labels the conservative-leaning one "Conservative" and the other
    "Aggressive" with no "Balanced" in between.
    """
    live = [p for p in frontier if not p.dominated]
    if not live:
        return frontier
    if len(live) == 1:
        live[0].label = "Balanced"
        return frontier

    ranked = sorted(
        live,
        key=lambda p: (p.metrics.get(conservative_metric) if p.metrics.get(conservative_metric) is not None else float("-inf")),
        reverse=True,
    )
    ranked[0].label = "Conservative"
    ranked[-1].label = "Aggressive"
    for p in ranked[1:-1]:
        p.label = "Balanced"
    return frontier


def pareto_report(frontier: list[ParetoPoint]) -> str:
    """Plain-text summary: the frontier members (with labels) first,
    then a one-line count of how many candidates were dominated (and by
    the frontier, not lost -- see ParetoPoint.dominated_by for detail
    on any specific one)."""
    live = [p for p in frontier if not p.dominated]
    dead = [p for p in frontier if p.dominated]
    lines = [f"Pareto Frontier: {len(live)} non-dominated candidate(s) of {len(frontier)} evaluated.", ""]
    for p in sorted(live, key=lambda p: p.label or ""):
        metric_str = ", ".join(f"{k}={v:.2f}" for k, v in p.metrics.items() if v is not None)
        tag = f" [{p.label}]" if p.label else ""
        lines.append(f"  {p.key}{tag} -- {metric_str}")
    if dead:
        lines.append("")
        lines.append(f"{len(dead)} candidate(s) dominated on every objective by something on the frontier.")
    return "\n".join(lines)
