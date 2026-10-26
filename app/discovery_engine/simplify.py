"""Prefer a stable, simple winner: a structural depth cap and a pruning pass.

``cap_structure``   hard limits on how many entry / exit conditions a candidate may carry (the engine applies it to every child, so
                    the search can never reward complexity that the complexity charge in the fitness only discourages)
``prune``           remove each condition in turn and keep the removal whenever the edge lower bound does not fall by more than
                    ``tolerance`` R; repeat until nothing more can go. A condition that the edge does not need is a degree of
                    freedom the search fitted to noise.
Both work on a Manual-config spec (``{"source_type": "manual", "config": {...}}`` or the bare config) and never mutate their input.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable

SIDES = ("long", "short")


@dataclass(frozen=True)
class StructureCap:
    max_entry_conditions_per_side: int = 4
    max_exit_conditions_per_side: int = 2


DEFAULT_CAP = StructureCap()


def _cfg(spec: dict) -> dict:
    return spec["config"] if isinstance(spec, dict) and isinstance(spec.get("config"), dict) else spec


def cap_structure(spec: dict, cap: StructureCap = DEFAULT_CAP) -> tuple[dict, int]:
    """(spec with at most ``cap`` conditions per side, number of conditions removed). Drops from the END (latest-added) first."""
    out = copy.deepcopy(spec)
    cfg, removed = _cfg(out), 0
    for key, limit in (("entry_conditions", cap.max_entry_conditions_per_side), ("exit_conditions", cap.max_exit_conditions_per_side)):
        block = cfg.get(key) or {}
        for side in SIDES:
            conds = block.get(side)
            if isinstance(conds, list) and len(conds) > limit:
                removed += len(conds) - limit
                block[side] = conds[:limit]
    return out, removed


def within_cap(spec: dict, cap: StructureCap = DEFAULT_CAP) -> bool:
    return cap_structure(spec, cap)[1] == 0


def _slots(spec: dict) -> list[tuple[str, str, int]]:
    cfg, out = _cfg(spec), []
    for key in ("entry_conditions", "exit_conditions"):
        block = cfg.get(key) or {}
        for side in SIDES:
            for i in range(len(block.get(side) or [])):
                out.append((key, side, i))
    return out


def _without(spec: dict, slot: tuple[str, str, int]) -> dict:
    out = copy.deepcopy(spec)
    key, side, i = slot
    del _cfg(out)[key][side][i]
    return out


@dataclass
class PruneResult:
    spec: dict
    removed: list = field(default_factory=list)       # [(key, side, condition_dict)]
    score_before: float = float("-inf")
    score_after: float = float("-inf")
    evaluations: int = 0

    def to_dict(self) -> dict:
        return {"removed": [(k, s, c) for k, s, c in self.removed], "score_before": self.score_before,
                "score_after": self.score_after, "evaluations": self.evaluations}


def prune(spec: dict, score: Callable[[dict], float], tolerance: float = 0.005, min_entry_conditions: int = 1,
          max_passes: int = 6) -> PruneResult:
    """Greedy backward elimination. ``score(spec)`` is the shared objective (higher = better, -inf = unmeasurable).

    A condition is removed when the pruned spec scores at least ``score_before - tolerance``; the pass removing the condition whose
    removal helps MOST goes first. Entry sides never drop below ``min_entry_conditions`` conditions on a side that has any."""
    cur = copy.deepcopy(spec)
    res = PruneResult(cur, score_before=float(score(cur)))
    res.evaluations = 1
    best = res.score_before
    if not math.isfinite(best):
        res.score_after = best
        return res
    for _ in range(int(max_passes)):
        cands = []
        for slot in _slots(cur):
            key, side, i = slot
            if key == "entry_conditions" and len(_cfg(cur)[key][side]) <= int(min_entry_conditions):
                continue
            trial = _without(cur, slot)
            s = float(score(trial))
            res.evaluations += 1
            if math.isfinite(s) and s >= best - float(tolerance):
                cands.append((s, slot, trial))
        if not cands:
            break
        s, slot, trial = max(cands, key=lambda x: x[0])
        res.removed.append((slot[0], slot[1], copy.deepcopy(_cfg(cur)[slot[0]][slot[1]][slot[2]])))
        cur, best = trial, max(s, best - float(tolerance))
    res.spec, res.score_after = cur, float(score(cur))
    res.evaluations += 1
    return res
