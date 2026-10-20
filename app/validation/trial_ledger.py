"""Persistent trial ledger -- an honest count of how many variants of a strategy idea were ever evaluated.

Deflated Sharpe / Bonferroni gates are only fair if they know how many things you tried. Hand-tuning, optimizer runs and prior
pipeline runs all add trials; forgetting them flatters the verdict. Each pipeline run records its own evaluations here and the
next run for the same strategy name automatically adds them to external_trial_count.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PATH = Path(__file__).resolve().parents[2] / "reports" / "trial_ledger.json"


def _key(label) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label or "unknown").lower()).strip("_") or "unknown"


def _load() -> dict:
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def total(label) -> int:
    return int(_load().get(_key(label), 0))


def record(label, n: int) -> int:
    n = max(0, int(n))
    with _LOCK:
        data = _load()
        k = _key(label)
        data[k] = int(data.get(k, 0)) + n
        try:
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            _PATH.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        except Exception:
            pass
        return data[k]
