"""
Shared "distributions, not just a winner" summary -- generalizes
app.optimize.refinement._compute_distribution_summary (which only ever
worked on that module's own Candidate dataclass) so Search Lab and
Evolution Lab's own batch-level results can report the same median-vs-best
table and robust-candidate count, from their own, differently-shaped
result records, without duplicating this logic three times.

Works on a list of plain dicts, each expected to have:
    "fitness"      -- float, this candidate's own optimization score
    "mc_summary"   -- dict with "evaluation_pass_probability" and
                       "first_payout_probability" keys (or None)
    "statistics"   -- dict with a "max_drawdown_pct" key (or None)
Any record missing "mc_summary" (e.g. a candidate that never reached a
full evaluation) is simply excluded from every distribution below, exactly
like the original -- this never fabricates a per-candidate number for a
metric that wasn't actually computed for that candidate.
"""
from __future__ import annotations

import math


def _pctile(values: list[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def compute_distribution_summary(records: list[dict], total_tested: int | None = None) -> dict | None:
    """`records` -- plain dicts as described in this module's own
    docstring. `total_tested` lets a caller report a "candidates_tested"
    count larger than len(records) when records only holds the subset
    that reached a full evaluation (e.g. Search Lab's stage3 leaderboard,
    where stage1/stage2 filtered out far more candidates than stage3 ever
    sees) -- defaults to len(records) when omitted."""
    finite = [
        r for r in records
        if r.get("mc_summary") and isinstance(r.get("fitness"), (int, float)) and math.isfinite(r["fitness"])
    ]
    if not finite:
        return None

    eval_pass = [r["mc_summary"]["evaluation_pass_probability"] for r in finite]
    payout_pass = [r["mc_summary"]["first_payout_probability"] for r in finite]
    max_dd = [
        r["statistics"]["max_drawdown_pct"] for r in finite
        if r.get("statistics") and r["statistics"].get("max_drawdown_pct") is not None
    ]
    fitnesses = [r["fitness"] for r in finite]

    # See app.optimize.refinement._compute_distribution_summary's own
    # comment for exactly what this is and isn't claiming -- a simple,
    # transparent "how many genuinely different points in the space this
    # run tried are almost as good as the best one" count, not a
    # robustness test.
    best_fitness = max(fitnesses)
    robust_threshold = best_fitness * 0.8 if best_fitness > 0 else best_fitness * 1.2
    robust_count = sum(1 for f in fitnesses if f >= robust_threshold)

    return {
        "candidates_tested": total_tested if total_tested is not None else len(records),
        "candidates_with_trades": len(finite),
        "eval_pass_probability": {"median": _pctile(eval_pass, 0.5), "best": max(eval_pass)},
        "first_payout_probability": {"median": _pctile(payout_pass, 0.5), "best": max(payout_pass)},
        "max_drawdown_pct": (
            {"median": _pctile(max_dd, 0.5), "best": min(max_dd)} if max_dd else None
        ),
        "robust_candidates": robust_count,
        "robust_threshold_fraction_of_best": 0.8 if best_fitness > 0 else 1.2,
    }
