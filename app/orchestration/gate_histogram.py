"""Gate-kill histogram (2026-11-04 audit, Fix 2a).

Every full-pipeline run stores a ``gate_vector`` (see ``full_pipeline._gate_vector``): each verdict gate evaluated independently, with a
status of pass / fail / cap / n/a / info. This module aggregates those vectors over a folder of stored runs and answers the question the
audit asked -- "which gate is actually doing the killing?" -- three ways:

  * fail rate        share of runs in which the gate failed or capped (blocked READY),
  * fail rate among  the same, restricted to runs whose point-estimate pass probability was >= ``strong_pass_pct`` (80%) -- a gate that
    strong runs         kills a lot of strong runs is rejecting strategies the headline number likes,
  * sole killer      runs in which this gate was the ONLY blocking gate (removing it alone would have changed the verdict).

Pure functions over dicts plus a directory scanner; no pipeline imports, never raises on a malformed file (it is counted and skipped).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

BLOCKING = ("fail", "cap")


def _walk(obj, key):
    """Yield every value stored under ``key`` anywhere inside a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def _find_gate_vector(doc) -> dict | None:
    for gv in _walk(doc, "gate_vector"):
        if isinstance(gv, dict) and isinstance(gv.get("gates"), list):
            return gv
    return None


def _find_point_pass(doc) -> float | None:
    """Point-estimate per-attempt pass probability in PERCENT, wherever the run stored it (percent keys are used as they are)."""
    for key in ("headline_pass_pct", "per_attempt_pass_probability", "per_attempt_pass_pct"):
        for v in _walk(doc, key):
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f:
                return f
    return None


def load_runs(root: str | Path) -> tuple[list, int]:
    """(runs, n_unreadable): each run is {"path", "gate_vector", "point_pass_pct"}; only JSON files that contain a gate vector count."""
    runs, bad = [], 0
    for p in sorted(Path(root).rglob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        gv = _find_gate_vector(doc)
        if gv is None:
            continue
        runs.append({"path": str(p), "gate_vector": gv, "point_pass_pct": _find_point_pass(doc)})
    return runs, bad


def aggregate(runs: list, strong_pass_pct: float = 80.0) -> dict:
    """Histogram over ``runs`` (see ``load_runs``). Returns {"n_runs", "n_strong", "gates": {id: {...}}, "ranking": [...]}."""
    n = len(runs)
    strong = [r for r in runs if r.get("point_pass_pct") is not None and float(r["point_pass_pct"]) >= strong_pass_pct]
    labels: dict = {}
    blocked = Counter()
    blocked_strong = Counter()
    sole = Counter()
    measured = Counter()
    for r in runs:
        gates = (r.get("gate_vector") or {}).get("gates") or []
        blockers = [g["id"] for g in gates if g.get("status") in BLOCKING]
        is_strong = r in strong
        for g in gates:
            gid = g.get("id")
            labels.setdefault(gid, g.get("label", gid))
            if g.get("status") not in ("n/a", None):
                measured[gid] += 1
        for gid in set(blockers):
            blocked[gid] += 1
            if is_strong:
                blocked_strong[gid] += 1
        if len(set(blockers)) == 1:
            sole[blockers[0]] += 1
    out: dict = defaultdict(dict)
    for gid in labels:
        out[gid] = {
            "label": labels[gid], "measured": measured[gid], "blocked": blocked[gid],
            "fail_rate": (blocked[gid] / n) if n else None,
            "blocked_among_strong": blocked_strong[gid],
            "fail_rate_among_strong": (blocked_strong[gid] / len(strong)) if strong else None,
            "sole_killer": sole[gid],
        }
    ranking = sorted(out, key=lambda g: (-out[g]["sole_killer"], -out[g]["blocked"]))
    return {"n_runs": n, "n_strong": len(strong), "strong_pass_pct": strong_pass_pct, "gates": dict(out), "ranking": ranking}


def render(hist: dict) -> str:
    lines = [f"Gate-kill histogram over {hist['n_runs']} run(s); {hist['n_strong']} with point pass >= {hist['strong_pass_pct']:g}%.",
             f"{'gate':<14}{'blocked':>8}{'fail%':>8}{'strong%':>9}{'sole killer':>13}  label"]
    for gid in hist["ranking"]:
        g = hist["gates"][gid]
        fr = "-" if g["fail_rate"] is None else f"{g['fail_rate'] * 100:.0f}"
        fs = "-" if g["fail_rate_among_strong"] is None else f"{g['fail_rate_among_strong'] * 100:.0f}"
        lines.append(f"{gid:<14}{g['blocked']:>8}{fr:>8}{fs:>9}{g['sole_killer']:>13}  {g['label']}")
    return "\n".join(lines)
