"""Remove fake/test rows from data/run_history.json (2026-10 audit).

A row is a test run when ANY of:
  * instrument is 'TEST' / 'TEST.csv' (case-insensitive), or ends with '_test'
  * strategy name starts with 'test_' or 'batch_test'
The leaderboard and any "best strategy" view read this file, so rows produced by unit tests / smoke runs on synthetic
frames (e.g. +$490,627 'Search Lab' results on instrument 'TEST') must not sit beside real runs.

    python scripts/prune_run_history.py            # rewrite in place (writes data/run_history.removed.json with what it dropped)
    python scripts/prune_run_history.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "run_history.json"


def is_test_run(r: dict) -> bool:
    inst = str(r.get("instrument", "")).strip().lower()
    name = str(r.get("strategy_name", "")).strip().lower()
    return inst in ("test", "test.csv") or inst.endswith("_test") or name.startswith(("test_", "batch_test"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = json.loads(PATH.read_text(encoding="utf-8"))
    keep = [r for r in rows if not is_test_run(r)]
    drop = [r for r in rows if is_test_run(r)]
    print(f"{len(rows)} rows: keeping {len(keep)}, dropping {len(drop)} test run(s)")
    if args.dry_run:
        return
    PATH.with_name("run_history.removed.json").write_text(json.dumps(drop, indent=1), encoding="utf-8")
    PATH.write_text(json.dumps(keep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
