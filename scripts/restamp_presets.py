"""Re-stamp prop-firm presets after YOU re-read the firm's own rules page.

    python scripts/restamp_presets.py ftmo_100k topstep_50k --date 2026-11-01
    python scripts/restamp_presets.py --list          # show every stamp and how many days remain before it goes stale

Writes data/preset_stamps.json (read by app/prop/presets.py). READY needs a fresh stamp (30 days). Only stamp rows whose
numbers you actually re-checked; a stamp is a claim, not a formality."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "preset_stamps.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    from app.prop import presets
    by = {p.key: p for p in presets.PROP_FIRM_PRESETS}
    if a.list or not a.keys:
        today = dt.date.today()
        for k, st in sorted(presets._VERIFIED_STAMPS.items()):
            left = 30 - (today - dt.date.fromisoformat(st[0])).days
            print(f"{k:22s} stamped {st[0]}  {'STALE' if left < 0 else f'{left} days left'}")
        return 0
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    for k in a.keys:
        if k not in by:
            print(f"unknown preset: {k}")
            return 2
        old = presets._VERIFIED_STAMPS.get(k, ("", by[k].source_url or "", "primary"))
        cur[k] = {"date": a.date, "url": old[1] or by[k].source_url or "", "tier": old[2] or "primary"}
    OUT.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    print(f"stamped {len(a.keys)} preset(s) as {a.date} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
