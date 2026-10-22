#!/usr/bin/env python3
"""Fold YOUR OWN local Strategy Graveyard into the seed that ships with the app.

Reads every data/evolution/strategy_graveyard__<instrument>__<timeframe>.jsonl on this machine and writes
app/search/seed_data/graveyard_seed.jsonl (merging with the rows already in the seed, de-duplicated by candidate_id), so the next
build/download starts with those failures already in the Graveyard. Nothing is recomputed -- rows are copied as-is.

    python scripts/export_graveyard_seed.py                       # uses this machine's app data folder
    python scripts/export_graveyard_seed.py --evolution-dir PATH  # or point at a copied data/evolution folder

Instrument and timeframe come from the file names, which are the sanitized labels the app itself uses; rows land in
the same-named files on the machine that installs the seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = ROOT / "app" / "search" / "seed_data" / "graveyard_seed.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evolution-dir", type=Path, default=None)
    a = ap.parse_args()
    if a.evolution_dir is None:
        from app.data.storage import get_app_base_dir
        a.evolution_dir = get_app_base_dir() / "data" / "evolution"
    merged: dict[str, dict] = {}
    if SEED.exists():
        for line in SEED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                merged[r["candidate_id"]] = r
    added = 0
    for p in sorted(a.evolution_dir.glob("strategy_graveyard__*__*.jsonl")):
        parts = p.stem.split("__")
        if len(parts) < 3:
            continue
        inst, tf = parts[1], parts[2]
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = r.get("candidate_id")
            if not cid or cid in merged:
                continue
            r["_seed_instrument"], r["_seed_timeframe"] = inst, tf
            merged[cid] = r
            added += 1
    SEED.parent.mkdir(parents=True, exist_ok=True)
    with SEED.open("w", encoding="utf-8") as f:
        for r in merged.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"added {added} rows from {a.evolution_dir}; seed now holds {len(merged)} -> {SEED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
