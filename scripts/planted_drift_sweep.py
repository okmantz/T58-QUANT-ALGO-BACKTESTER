"""Planted-drift sweep (2026-11-04 audit, Fix 2b): plant a KNOWN edge in synthetic trades, run the gate stack, and report how often each
edge level is detected. Acceptance targets from the roadmap: at 0.15R detection >= 75%, at 0R false positives <= 5%, and at 0.12R the
FORWARD-TEST CANDIDATE rate >= 60%.

Usage: python scripts/planted_drift_sweep.py [--seeds 20] [--edges 0 0.05 0.10 0.15 0.20 0.30] [--check] [--json out.json]
``--check`` exits non-zero when a target is missed (the nightly regression). The sweep covers the STATISTICAL gate stack (outer-bootstrap pass
interval, pooled OOS, holdout confirmation, unseen expectancy at 2x costs, unseen DSR) on planted trade streams; it does not run the
genetic search, so it isolates whether the gates can see an edge that is really there."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.validation.planted_edge import render_sweep, sweep  # noqa: E402

TARGETS = {0.15: ("detected_pct", ">=", 75.0), 0.0: ("detected_pct", "<=", 5.0), 0.12: ("forward_candidate_pct", ">=", 60.0)}


def evaluate_targets(rows: list) -> list:
    by = {round(r["edge_r"], 3): r for r in rows}
    fails = []
    for edge, (key, op, bar) in TARGETS.items():
        r = by.get(round(edge, 3))
        if r is None:
            continue
        v = r[key]
        if (op == ">=" and v < bar) or (op == "<=" and v > bar):
            fails.append(f"edge {edge:g}R: {key} = {v:.0f}% (needs {op} {bar:g}%)")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--edges", type=float, nargs="*", default=[0.0, 0.05, 0.10, 0.12, 0.15, 0.20, 0.30])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = sweep(edges=tuple(a.edges), seeds=range(a.seeds))
    print(render_sweep(rows))
    fails = evaluate_targets(rows)
    for f in fails:
        print("MISSED:", f)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 1 if (a.check and fails) else 0


if __name__ == "__main__":
    raise SystemExit(main())
