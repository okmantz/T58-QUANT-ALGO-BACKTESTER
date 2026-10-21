"""One-command data bootstrap (report 2026-10-20, section 5).

    python scripts/bootstrap_data.py            # rebuild manifest, write event calendar, check clock + specs + coverage, print banner
    python scripts/bootstrap_data.py --no-clock # skip the (slower) per-file session-clock check

Prints a LOUD banner listing everything that will silently weaken results: files without an instrument spec (unmeasured costs), files
that cannot be parsed, datasets with a misaligned session clock, target markets that are missing, and which pooled-edge tests therefore
cannot run on the full 8-12 market universe. Exit code 1 when a blocking problem exists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The cross-market universe the primary edge test pools over (report section 2). Roots, in priority order.
TARGET_MARKETS = ("ES", "NQ", "YM", "RTY", "GC", "CL", "ZN", "ZB", "6E", "EURUSD", "GBPUSD", "USDJPY")
MIN_POOL_MARKETS = 8


def _banner(lines: list[str], char: str = "!") -> str:
    w = max(len(x) for x in lines) + 4
    return "\n".join([char * w] + [f"{char} {x.ljust(w - 4)} {char}" for x in lines] + [char * w])


def coverage_report(manifest: dict) -> dict:
    files = manifest.get("files", [])
    have = {}
    for f in files:
        sym = (f.get("symbol") or "").upper()
        if sym and (f.get("bars") or 0) > 0:
            have[sym] = max(have.get(sym, 0.0), float(f.get("years") or 0.0))
    missing = [m for m in TARGET_MARKETS if m not in have and not any(h.startswith(m) or m.startswith(h) for h in have)]
    return {
        "markets_present": sorted(have),
        "missing_targets": missing,
        "pool_size": len([m for m in TARGET_MARKETS if m not in missing]),
        "no_spec": [f["file"] for f in files if f.get("spec_resolved") is False],
        "lfs_pointers": [f["file"] for f in files if "GIT-LFS" in str(f.get("flags", ""))],
        "unparseable": [f["file"] for f in files if (f.get("bars") or 0) == 0 and "GIT-LFS" not in str(f.get("flags", ""))],
        "synthetic": [f["file"] for f in files if f.get("synthetic")],
        "short": [f["file"] for f in files if f.get("years") and f["years"] < 5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clock", action="store_true")
    a = ap.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_data_manifest.py")], check=True, cwd=ROOT)
    try:
        from app.data.event_calendar import builtin_events, write_events_csv
        ev = builtin_events()
        for name in ev["event"].unique():
            write_events_csv(ev[ev["event"] == name], name)
        print(f"event calendar: {len(ev)} FOMC/NFP events {ev['timestamp'].min().date()} .. {ev['timestamp'].max().date()}")
    except Exception as exc:  # noqa: BLE001
        print(f"event calendar skipped: {exc}")
    manifest = json.loads((ROOT / "data" / "data_manifest.json").read_text(encoding="utf-8"))
    cov = coverage_report(manifest)
    problems, blocking = [], False
    if cov["pool_size"] < MIN_POOL_MARKETS:
        problems.append(f"Only {cov['pool_size']} of the {MIN_POOL_MARKETS}+ markets needed for the POOLED edge test are present.")
        problems.append(f"Missing: {', '.join(cov['missing_targets'])}. Pooled tests run on what exists and are labelled UNDERPOWERED.")
    if cov["no_spec"]:
        blocking = True
        problems.append(f"{len(cov['no_spec'])} file(s) have NO instrument spec, so costs are unmeasured (backtests would be too optimistic):")
        problems += [f"   - {x}" for x in cov["no_spec"][:8]]
    if cov["lfs_pointers"]:
        blocking = True
        problems.append(f"{len(cov['lfs_pointers'])} file(s) are git-LFS pointers, not data - run `git lfs pull`: " + ", ".join(cov["lfs_pointers"][:4]))
    if cov["unparseable"]:
        problems.append(f"{len(cov['unparseable'])} file(s) have 0 parseable bars and are SKIPPED: " + ", ".join(cov["unparseable"][:6]))
    if cov["synthetic"]:
        problems.append("Synthetic data present (never READY-eligible): " + ", ".join(cov["synthetic"][:4]))
    if not a.no_clock:
        try:
            import pandas as pd
            from app.data.session_clock import clock_quality_issues, infer_symbol
            raw = ROOT / "data" / "raw"
            for f in manifest.get("files", []):
                if f.get("bars") and "1m" in str(f.get("bar", "")) or str(f.get("bar", "")).endswith("00:01:00"):
                    p = raw / f["file"]
                    if p.suffix == ".parquet":
                        df = pd.read_parquet(p).head(200_000)
                        for sev, msg in clock_quality_issues(df, infer_symbol(f["file"])):
                            problems.append(f"CLOCK [{sev}] {f['file']}: {msg}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"clock check could not run: {exc}")
    print()
    print(_banner(["DATA BOOTSTRAP " + ("- BLOCKING PROBLEMS" if blocking else "- WARNINGS")] + problems) if problems
          else _banner(["DATA BOOTSTRAP OK: manifest rebuilt, every file has a spec, pooled universe complete."], "="))
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
