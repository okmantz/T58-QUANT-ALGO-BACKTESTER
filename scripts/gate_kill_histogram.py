"""Which gate kills the most runs? Usage: python scripts/gate_kill_histogram.py [reports_dir] [--strong 80] [--json out.json]

Scans stored full-pipeline run JSON under ``reports_dir`` (default ./reports) for ``gate_vector`` records and prints the fail rate per gate,
the fail rate among runs whose point pass probability was >= --strong percent, and how often each gate was the sole blocker."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.orchestration.gate_histogram import aggregate, load_runs, render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="reports")
    ap.add_argument("--strong", type=float, default=80.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs, bad = load_runs(a.root)
    h = aggregate(runs, a.strong)
    print(render(h))
    if bad:
        print(f"({bad} unreadable JSON file(s) skipped)")
    if a.json:
        Path(a.json).write_text(json.dumps(h, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
