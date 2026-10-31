"""Persistent trial ledger -- an honest count of how many variants of a strategy idea were ever evaluated.

Deflated Sharpe / Bonferroni gates are only fair if they know how many things you tried. Hand-tuning, optimizer runs and prior
pipeline runs all add trials; forgetting them flatters the verdict. Each pipeline run records its own evaluations here and the
next run for the same strategy name automatically adds them to external_trial_count.

2026-10-20 (report section 4, "ledger aging"): entries are now timestamped and old trials count less. A search run three years ago
on different data is weak evidence that today's idea was overfit to it, while counting it forever makes a long-lived idea
undeployable. ``total(label)`` weights each entry by 0.5 ** (age / half_life_days); legacy integer entries (no timestamp) keep full
weight. ``raw_total`` ignores aging. Family-level ledgers use the same functions with a label such as ``family:event_day``.

2026-10-25 (Strategy Discovery Engine, report "Tier 2"): aging is only honest when the DATA changed. A search run on the same
historical window two years ago carries exactly the same selection bias today, so it must not fade. Entries may now carry a
data fingerprint (``fp``, see ``app.discovery_engine.provenance.data_fingerprint``). When the caller passes ``current_fp`` to
``total`` / ``family_prior``, entries whose fingerprint EQUALS it keep full weight; entries with a different (or no) fingerprint
age as before. Callers that pass no ``current_fp`` get the old behaviour byte for byte.

2026-10-31 audit (section 4, "the trial ledger is a counter, not a ledger"): the ledger now FAILS LOUDLY. A corrupt or unreadable file
used to load as an empty ledger (trial count 0, so the deflated-Sharpe bar silently fell) and a failed write was swallowed while
``record`` still returned a total. Now:

* ``_load`` raises ``TrialLedgerError`` for a file that exists but cannot be parsed as a JSON object, after copying the bad file to
  ``trial_ledger.corrupt-<hash>.json`` for forensics. The original is left in place, so every later call keeps raising until a human
  repairs or restores it -- it never reads as empty. A file that does not exist is a genuinely empty ledger.
* ``_write`` raises ``TrialLedgerError`` when the atomic write fails, so ``record`` / ``book_run`` / ``mark_clock_epoch`` abort the
  run instead of reporting a total that was never saved.
* Same-data trials never decay: callers pass ``current_fp`` (``data_fingerprint(df)``) and ``data_fp``; ``decay_report`` says which
  rule fired for a label so a report can state it.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

class TrialLedgerError(RuntimeError):
    """The trial ledger is corrupt, unreadable or unwritable. Trial counts feed the multiple-testing penalty, so a ledger that cannot be
    trusted must stop the run rather than read as empty."""


_LOCK = threading.RLock()
_PATH = Path(__file__).resolve().parents[2] / "reports" / "trial_ledger.json"
DEFAULT_HALF_LIFE_DAYS = 730.0


def _key(label) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label or "unknown").lower()).strip("_") or "unknown"


def _load() -> dict:
    """The ledger as a dict. A missing file is an empty ledger; a file that exists but is unreadable or not a JSON object raises."""
    try:
        raw = _PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise TrialLedgerError(f"Trial ledger {_PATH} could not be read ({exc}). Refusing to treat it as empty: that would drop the "
                               "trial count to zero and lower the deflated-Sharpe bar.") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        _quarantine(raw)
        raise TrialLedgerError(f"Trial ledger {_PATH} is corrupt ({exc}). A copy was saved beside it as trial_ledger.corrupt-*.json. "
                               "Restore it from backup or repair it by hand; it is NOT being treated as empty.") from exc
    if not isinstance(data, dict):
        _quarantine(raw)
        raise TrialLedgerError(f"Trial ledger {_PATH} does not hold a JSON object (found {type(data).__name__}); refusing to use it.")
    return data


def _quarantine(raw: str) -> None:
    """Best-effort forensic copy of a corrupt ledger (named by content hash, so repeated failures do not pile up copies)."""
    try:
        import hashlib
        tag = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:10]
        dest = _PATH.with_name(f"trial_ledger.corrupt-{tag}.json")
        if not dest.exists():
            dest.write_text(raw, encoding="utf-8")
    except OSError:
        pass


def _write(data: dict) -> None:
    """Atomic write (temp file + replace) so a crash mid-write can never truncate the whole ledger. Raises ``TrialLedgerError`` on failure."""
    tmp = None
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".trial_ledger.", suffix=".tmp", dir=str(_PATH.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=1, sort_keys=True))
        os.replace(tmp, _PATH)
    except Exception as exc:  # noqa: BLE001 - re-raised as the ledger's own error
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise TrialLedgerError(f"Trial ledger {_PATH} could not be written ({exc}). The trials of this run were NOT recorded, so the "
                               "run is aborted rather than reporting a count that was never saved.") from exc


def _entries(value) -> list[dict]:
    """Normalise a stored value (legacy int or list of {t, n}) into entries; legacy ints have t = None."""
    if isinstance(value, list):
        return [e for e in value if isinstance(e, dict) and "n" in e]
    try:
        n = int(value)
    except (TypeError, ValueError):
        return []
    return [{"t": None, "n": n}] if n else []


def _compact(ents: list[dict]) -> list[dict]:
    """Merge entries within the same UTC day AND the same data fingerprint so the file does not grow with every run."""
    merged: dict = {}
    for e in ents:
        day = None if e.get("t") is None else int(float(e["t"]) // 86400)
        key = (day, e.get("fp"))
        m = merged.setdefault(key, {"t": e.get("t"), "n": 0})
        if e.get("fp"):
            m["fp"] = e["fp"]
        m["n"] += int(e["n"])
        if e.get("t") is not None:
            m["t"] = max(float(m["t"] or 0), float(e["t"]))
    return list(merged.values())


def raw_total(label) -> int:
    return int(sum(int(e.get("n", 0)) for e in _entries(_load().get(_key(label), 0))))


def total(label, half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS, now: float | None = None,
          current_fp: str | None = None) -> int:
    """Aged trial count (rounded up so one real trial never rounds to zero). ``half_life_days=None`` disables aging.

    ``current_fp``: fingerprint of the data window being judged today. Entries recorded on that SAME window do not decay
    (the selection bias they created is still fully present); entries on any other window age with the half-life."""
    ents = _entries(_load().get(_key(label), 0))
    if half_life_days is None:
        return int(sum(int(e["n"]) for e in ents))
    now = time.time() if now is None else now
    acc = 0.0
    for e in ents:
        n = int(e.get("n", 0))
        t = e.get("t")
        same_window = bool(current_fp) and e.get("fp") == current_fp
        w = 1.0 if (t is None or same_window) else 0.5 ** (max(0.0, now - float(t)) / 86400.0 / float(half_life_days))
        acc += n * w
    return int(-(-acc // 1)) if acc > 0 else 0


def decay_report(label, half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS, now: float | None = None,
                 current_fp: str | None = None) -> dict:
    """Which aging rule fired for ``label`` -- so a report can state it instead of leaving the reader to guess.

    Returns ``{"raw": int, "aged": int, "same_data": int, "aged_by_half_life": int, "rule": str}`` where ``same_data`` counts trials
    recorded on exactly ``current_fp`` (never decayed) and ``aged_by_half_life`` counts the rest."""
    ents = _entries(_load().get(_key(label), 0))
    raw = int(sum(int(e.get("n", 0)) for e in ents))
    same = int(sum(int(e.get("n", 0)) for e in ents if current_fp and e.get("fp") == current_fp))
    aged = total(label, half_life_days=half_life_days, now=now, current_fp=current_fp)
    if half_life_days is None:
        rule = "no aging (half-life disabled): every trial counts in full"
    elif current_fp and same:
        rule = f"data fingerprint {current_fp} matched {same} trial(s): they do not decay; the other {raw - same} age with a {half_life_days:g}-day half-life"
    else:
        rule = f"no trial shares the current data fingerprint: all age with a {half_life_days:g}-day half-life"
    return {"raw": raw, "aged": aged, "same_data": same, "aged_by_half_life": raw - same, "rule": rule}


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


def record(label, n: int, now: float | None = None, data_fp: str | None = None) -> int:
    n = max(0, int(n))
    with _LOCK:
        data = _load()
        k = _key(label)
        ents = _entries(data.get(k, 0))
        ent = {"t": time.time() if now is None else float(now), "n": n}
        if data_fp:
            ent["fp"] = str(data_fp)
        ents.append(ent)
        # compact: merge entries within the same UTC day (and data window) so the file does not grow with every run
        data[k] = _compact(ents)
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
KINDS_KEY = "_kinds"      # per-run audit of WHAT was booked (prefilter kills, GA inner loop, grammar mutations, ...)


def family_label(name) -> str:
    """Ledger label for a family: ``family:<name>`` (already-prefixed names pass through)."""
    s = str(name or "unknown").strip()
    return s if s.lower().startswith("family:") else f"family:{s}"


def book_run(run_key: str, counts: dict, now: float | None = None, data_fp: str | None = None,
             kinds: dict | None = None) -> dict:
    """Book ``{family_name: n_evaluated}`` for one run, once. Returns the counts actually booked ({} if already booked).

    ``data_fp`` stamps the entries with the data window they were evaluated on (see ``total(current_fp=...)``); ``kinds`` is an
    optional ``{kind: n}`` audit trail (``prefilter_kills``, ``ga_inner_loop``, ``grammar_mutations`` ...) kept under a reserved key."""
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
            ent = {"t": t, "n": n}
            if data_fp:
                ent["fp"] = str(data_fp)
            ents.append(ent)
            data[k] = _compact(ents)
        if run_key and kinds:
            kd = data.get(KINDS_KEY) if isinstance(data.get(KINDS_KEY), dict) else {}
            kd[run_key] = {str(a): int(b) for a, b in kinds.items() if b}
            if len(kd) > 2000:
                for old in sorted(kd)[: len(kd) - 2000]:
                    kd.pop(old, None)
            data[KINDS_KEY] = kd
        if run_key:
            booked[run_key] = t
            if len(booked) > 2000:  # bounded
                for old in sorted(booked, key=booked.get)[: len(booked) - 2000]:
                    booked.pop(old, None)
            data[BOOKED_KEY] = booked
        _write(data)
    return clean


def family_prior(name, half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS, now: float | None = None,
                 current_fp: str | None = None) -> int:
    """Aged trial count previously booked for a family (0 when none). ``current_fp``: see ``total``."""
    if not name:
        return 0
    return total(family_label(name), half_life_days=half_life_days, now=now, current_fp=current_fp)


def run_kinds(run_key: str) -> dict:
    """The ``{kind: n}`` audit trail booked for one run (empty when none)."""
    kd = _load().get(KINDS_KEY)
    v = kd.get(run_key) if isinstance(kd, dict) else None
    return dict(v) if isinstance(v, dict) else {}


def all_family_labels() -> list[str]:
    """Every ``family:<name>`` label that has at least one booking."""
    return sorted(k for k in _load() if k.startswith("family_") or k.startswith("family:"))
