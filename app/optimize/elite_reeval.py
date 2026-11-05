"""Fitness-noise re-evaluation of elites.

The GA picks its winner by a Monte-Carlo fitness that is itself noisy: the top-scoring candidate of thousands is partly the one
the noise favoured (winner's curse). Re-scoring the few best candidates on fresh Monte-Carlo seeds and ranking by the AVERAGE of all
their scores removes that luck from the selection. ``reevaluate_elites`` is pure (the scorer is injected) so it is cheap to test.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence


def reevaluate_elites(elites: Sequence, score_with_seed: Callable[[list, int], float], *, repeats: int = 2, base_seed: int = 1000):
    """``elites``: objects with ``.genome`` and ``.fitness`` (best first). ``score_with_seed(genome, seed) -> fitness``.
    Returns ``(best, report)``: ``best`` is the elite with the highest mean over its original + re-scored fitness; ``report`` lists each
    elite's original score, re-scores, mean, and the 'luck' (original - mean) so the page can say how much of the lead was noise."""
    rows = []
    for rank, e in enumerate(elites):
        scores = [float(e.fitness)]
        for r in range(max(0, int(repeats))):
            try:
                v = float(score_with_seed(list(e.genome), base_seed + r))
            except Exception:  # noqa: BLE001 -- a re-score that cannot run is not evidence either way
                continue
            if math.isfinite(v):
                scores.append(v)
        finite = [s for s in scores if math.isfinite(s)]
        mean = sum(finite) / len(finite) if finite else float("-inf")
        rows.append({"rank": rank, "original": float(e.fitness), "rescored": scores[1:], "mean": mean,
                     "luck": (float(e.fitness) - mean) if finite else 0.0})
    if not rows:
        return None, {"elites": [], "swapped": False}
    i = max(range(len(rows)), key=lambda k: rows[k]["mean"])
    return elites[i], {"elites": rows, "chosen_rank": i, "swapped": i != 0, "repeats": int(repeats)}
