"""Persistent trial ledger -- an honest count of how many variants of a strategy idea were ever evaluated.

Deflated Sharpe / Bonferroni gates are only fair if they know how many things you tried. Hand-tuning, optimizer runs and prior
pipeline runs all add trials; forgetting them flatters the verdict. Each pipeline run records its own evaluations here and the
next run for the same strategy name automatically adds them to external_trial_count.

2026-10-20 (report section 4, "ledger aging"): entries are now timestamped and old trials count less. A search run three years ago
on different data is weak evidence that today's idea was overfit to it, while counting it forever makes a long-lived idea
undeployable. ``total(label)`` weights each entry by 0.5 ** (age / half_life_days); legacy integer entries (no timestamp) keep full
weight. ``raw_total`` ignores aging. Family-level ledgers use the same functions with a label such as ``family:event_day``.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_PATH = Path(__file__).resolve().parents[2] / "reports" / "trial_ledger.json"
DEFAULT_HALF_LIFE_DAYS = 730.0


def _key(label) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label or "unknown").lower()).strip("_") or "unknown"


def _load() -> dict:
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(data: dict) -> None:
    """Atomic write (temp file + replace) so a crash mid-write can never truncate the whole ledger."""
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".trial_ledger.", suffix=".tmp", dir=str(_PATH.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(data, indent=1, sort_keys=True))
            os.replace(tmp, _PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        pass


def _entries(value) -> list[dict]:
    """Normalise a stored value (legacy int or list of {t, n}) into entries; legacy ints have t = None."""
    if isinstance(value, list):
        return [e for e in value if isinstance(e, dict) and "n" in e]
    try:
        n = int(value)
    except (TypeError, ValueError):
        return []
    return [{"t": None, "n": n}] if n else []


def raw_total(label) -> int:
    return int(sum(int(e.get("n", 0)) for e in _entries(_load().get(_key(label), 0))))


def total(label, half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS, now: float | None = None) -> int:
    """Aged trial count (rounded up so one real trial never rounds to zero). ``half_life_days=None`` disables aging."""
    ents = _entries(_load().get(_key(label), 0))
    if half_life_days is None:
        return int(sum(int(e["n"]) for e in ents))
    now = time.time() if now is None else now
    acc = 0.0
    for e in ents:
        n = int(e.get("n", 0))
        t = e.get("t")
        w = 1.0 if t is None else 0.5 ** (max(0.0, now - float(t)) / 86400.0 / float(half_life_days))
        acc += n * w
    return int(-(-acc // 1)) if acc > 0 else 0


CLOCK_EPOCHS_KEY = "_clock_epochs"


def mark_clock_epoch(epoch_id: str, note: str, now: float | None = None) -> None:
    """Annotate (never delete) the ledger: every trial recorded before ``now`` ran on the misaligned session clock.

    Stored under a reserved key that ``total`` / ``raw_total`` never read, so trial counts -- and the deflated-Sharpe penalty
    that depends on them -- are unchanged. ``clock_epochs()`` returns the annotations for reports."""
    with _LOCK:
        data = _load()
        eps = data.get(CLOCK_EPOCHS_KEY) if isinstance(data.get(CLOCK_EPOCHS_KEY), list) else []
        if any(isinstance(e, dict) and e.get("epoch") == epoch_id for e in eps):
            return
        eps.append({"epoch": str(epoch_id), "t": time.time() if now is None else float(now), "note": str(note)})
        data[CLOCK_EPOCHS_KEY] = eps
        _write(data)


def clock_epochs() -> list[dict]:
    v = _load().get(CLOCK_EPOCHS_KEY)
    return [e for e in v if isinstance(e, dict)] if isinstance(v, list) else []


def record(label, n: int, now: float | None = None) -> int:
    n = max(0, int(n))
    with _LOCK:
        data = _load()
        k = _key(label)
        ents = _entries(data.get(k, 0))
        ents.append({"t": time.time() if now is None else float(now), "n": n})
        # compact: merge entries within the same UTC day so the file does not grow with every run
        merged: dict = {}
        for e in ents:
            day = None if e.get("t") is None else int(float(e["t"]) // 86400)
            m = merged.setdefault(day, {"t": e.get("t"), "n": 0})
            m["n"] += int(e["n"])
            if e.get("t") is not None:
                m["t"] = max(float(m["t"] or 0), float(e["t"]))
        data[k] = list(merged.values())
        _write(data)
        return int(sum(int(e["n"]) for e in data[k]))


# ---------------------------------------------------------------------------
# Family-level bookings for Search Lab / Evolution Lab (2026-10-24)
#
# A four-hour Search Lab run evaluates thousands of candidates (Stage-1 kills
# included) and none of that used to reach the ledger, so later Deflated
# Sharpe / Bonferroni verdicts were not deflated by it. ``book_run`` books a
# run's evaluated count under ``family:<name>``; ``family_prior`` reads it
# back. Bookings are idempotent per ``run_key`` (a retried/resumed run never
# double counts) and the bookkeeping lives under reserved keys that ``total``
# / ``raw_total`` never read.
# ---------------------------------------------------------------------------
BOOKED_KEY = "_booked"


def family_label(name) -> str:
    """Ledger label for a family: ``family:<name>`` (already-prefixed names pass through)."""
    s = str(name or "unknown").strip()
    return s if s.lower().startswith("family:") else f"family:{s}"


def book_run(run_key: str, counts: dict, now: float | None = None) -> dict:
    """Book ``{family_name: n_evaluated}`` for one run, once. Returns the counts actually booked ({} if already booked)."""
    run_key = str(run_key or "")
    clean = {family_label(k): max(0, int(v)) for k, v in (counts or {}).items() if v and int(v) > 0}
    if not clean:
        return {}
    with _LOCK:
        data = _load()
        booked = data.get(BOOKED_KEY) if isinstance(data.get(BOOKED_KEY), dict) else {}
        if run_key and run_key in booked:
            return {}
        t = time.time() if now is None else float(now)
        for label, n in clean.items():
            k = _key(label)
            ents = _entries(data.get(k, 0))
            ents.append({"t": t, "n": n})
            merged: dict = {}
            for e in ents:
                day = None if e.get("t") is None else int(float(e["t"]) // 86400)
                m = merged.setdefault(day, {"t": e.get("t"), "n": 0})
                m["n"] += int(e["n"])
                if e.get("t") is not None:
                    m["t"] = max(float(m["t"] or 0), float(e["t"]))
            data[k] = list(merged.values())
        if run_key:
            booked[run_key] = t
            if len(booked) > 2000:  # bounded
                for old in sorted(booked, key=booked.get)[: len(booked) - 2000]:
                    booked.pop(old, None)
            data[BOOKED_KEY] = booked
        _write(data)
    return clean


def family_prior(name, half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS, now: float | None = None) -> int:
    """Aged trial count previously booked for a family (0 when none)."""
    if not name:
        return 0
    return total(family_label(name), half_life_days=half_life_days, now=now)
