"""
Persisted "current strategy" pointer + its validation checklist.

This is a thin, dependency-free layer alongside run_history.py, answering
two questions the UI needs and nothing tracked before:

  1. "Which strategy is the person currently focused on?"
  2. "Which of the deeper validation tools (Walk-Forward Opt, CPCV,
     Sensitivity, Walk-Forward GA, Regime Survival Matrix) has actually
     been run against THAT strategy, and what did each one find?"

Nothing here is inferred or guessed:
  - The current strategy is only ever set by an explicit action (the
    "Set as current" button on the Dashboard scorecard, or the query-string
    action /run redirects into after a backtest -- see server.py).
  - A validation result is only ever recorded against the current strategy
    when the identity a validation job just ran against matches the
    current strategy's own identity exactly. Running CPCV against some
    other, unrelated strategy never silently overwrites the checklist for
    whatever you've marked current.
  - A tool with no strict pass/fail verdict (Sensitivity, today) records
    passed=None -- "this has been run", not "and it passed". Nothing here
    invents a threshold the underlying tool doesn't itself compute.

IDENTITY (FIX 2026-09): every function here now takes an optional
``library_type``/``library_filename`` pair -- the exact (strategy_type,
filename) reference a saved Strategy Library entry is keyed by everywhere
else in the app (see app.strategy.library.record_validation_result and
friends). When present, THAT is the identity used for both "is this the
current strategy" and "which strategy did this validation run just test"
-- not the free-text display name.

This matters because the display name shown as "current strategy" is
often a long, tool-generated label (e.g. "T58 Gold Trend Breakout -- Full
Pipeline optimized (bos59/15, ema75-66 & 20-93, adx28-53, 1H), set risk at
0.6%"), which will almost never equal the plain filename/base name a later
CPCV/WFO/etc. run resolves the SAME strategy under when loaded fresh from
the library picker. Matching on the free-text name alone (the original
design) silently failed in exactly that case: a validation run genuinely
against the tracked strategy would get filed under a different key, and
the Dashboard/Validate checklist would keep showing 0/5 forever even
though every check had actually been run.

The free-text (strategy_name, instrument) pair is kept and still used as
a fallback for entries with no library identity (a one-off pasted/uploaded
strategy never saved to the library has no filename to key by), and for
reading data recorded before this fix -- record lookups always try the
library-identity key FIRST, then fall back to the legacy name+instrument
key, so nothing already recorded is lost.

Storage is two small JSON files under data/config/, next to run_history.json
and ui_theme.json. Both web and desktop already share get_app_base_dir(),
so this file is usable from either build.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

_CURRENT_FILENAME = "current_strategy.json"
_CHECKLIST_FILENAME = "validation_checklist.json"

VALIDATION_KINDS = ("wfo", "cpcv", "sensitivity", "wfga", "regime_matrix")

VALIDATION_LABELS = {
    "wfo": "Walk-Forward Opt",
    "cpcv": "CPCV / PBO",
    "sensitivity": "Sensitivity",
    "wfga": "Walk-Forward GA",
    "regime_matrix": "Regime Survival Matrix",
}

# Where each check lives in the guided nav, for the Validate hub's "Run" links.
VALIDATION_HREFS = {
    "wfo": "/walk-forward-opt",
    "cpcv": "/cpcv",
    "sensitivity": "/sensitivity",
    "wfga": "/walk-forward-ga",
    "regime_matrix": "/regime-matrix",
}


def _config_dir() -> Path:
    # Imported locally (not at module load time) so tests can monkeypatch
    # app.data.storage.get_app_base_dir and have it actually take effect --
    # matches the pattern app.ui.main_window's theme persistence already uses.
    from app.data.storage import get_app_base_dir

    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def strategy_key(strategy_name: str, instrument: str) -> str:
    """Legacy, name-based identity for a (strategy, instrument) pair --
    kept as the fallback key (and for reading data recorded before the
    library-identity fix above) when no library_type/library_filename is
    available. Case-insensitive: the same strategy re-run against the
    same instrument is "the same strategy" even if capitalization drifts."""
    return f"{(strategy_name or '').strip().lower()}::{(instrument or '').strip().lower()}"


def _library_key(library_type: str, library_filename: str) -> Optional[str]:
    """The preferred, stable identity: a saved Strategy Library entry's
    own (strategy_type, filename) -- immune to a display name changing
    across pipeline stages (e.g. Full Pipeline decorating it with
    "-- Full Pipeline optimized (...)"). Returns None when either half is
    missing, so callers can fall back to strategy_key() instead."""
    library_type = (library_type or "").strip().lower()
    library_filename = (library_filename or "").strip().lower()
    if not library_type or not library_filename:
        return None
    return f"lib::{library_type}::{library_filename}"


def _candidate_keys(strategy_name: str, instrument: str, library_type: str = "", library_filename: str = "") -> list[str]:
    """Every key worth trying for this identity, most-specific first: the
    library-identity key (if we have one), then the legacy name+instrument
    key. Used identically for both writes (record_validation always
    writes the most-specific key it has) and reads (get_checklist/
    robustness_score try each in order until one has data)."""
    keys = []
    lib_key = _library_key(library_type, library_filename)
    if lib_key:
        keys.append(lib_key)
    keys.append(strategy_key(strategy_name, instrument))
    return keys


def get_current_strategy() -> Optional[dict]:
    """Returns {"strategy_name", "instrument", "timeframe", "library_type",
    "library_filename", "set_at"} or None if nothing has been set yet (or
    the file is missing/corrupt -- never raises). "library_type"/
    "library_filename" default to "" for entries set before this field
    existed -- callers should treat an empty value as "no library
    identity known" and fall back to name-based matching."""
    path = _config_dir() / _CURRENT_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("strategy_name"):
            return None
        data.setdefault("library_type", "")
        data.setdefault("library_filename", "")
        return data
    except Exception:
        return None


def set_current_strategy(
    strategy_name: str,
    instrument: str,
    timeframe: str = "",
    *,
    library_type: str = "",
    library_filename: str = "",
) -> dict:
    """Marks (strategy_name, instrument) as the strategy the person is
    currently focused on. Pass library_type/library_filename whenever the
    caller was able to resolve this strategy to an actual saved Strategy
    Library entry (see app.web.server._find_library_match_for_current_strategy)
    -- this is what lets record_validation/get_checklist and the Champion
    Board's "what am I working on" panel all agree on the same strategy
    instead of independently guessing from the display name."""
    data = {
        "strategy_name": strategy_name,
        "instrument": instrument,
        "timeframe": timeframe,
        "library_type": library_type or "",
        "library_filename": library_filename or "",
        "set_at": time.time(),
    }
    (_config_dir() / _CURRENT_FILENAME).write_text(json.dumps(data), encoding="utf-8")
    return data


def clear_current_strategy() -> None:
    path = _config_dir() / _CURRENT_FILENAME
    if path.exists():
        path.unlink()


def _load_checklist_store() -> dict:
    path = _config_dir() / _CHECKLIST_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_checklist_store(store: dict) -> None:
    (_config_dir() / _CHECKLIST_FILENAME).write_text(json.dumps(store), encoding="utf-8")


def record_validation(
    strategy_name: str,
    instrument: str,
    kind: str,
    *,
    passed: Optional[bool],
    summary: str = "",
    report_html: str = "",
    library_type: str = "",
    library_filename: str = "",
    report_files: Optional[dict] = None,
) -> None:
    """Records one validation tool's result against this strategy's
    identity. Best-effort: never raises -- a checklist-recording failure
    must not break the validation job that produced it.

    Pass library_type/library_filename (the exact library_ref every job
    runner in app.web.server already has on hand) whenever this run's
    strategy came from -- or was saved to -- the Strategy Library: this
    is what makes the write land under the SAME key set_current_strategy
    used, instead of under a separate, never-looked-up name-based key.
    Without it, a validation run against a strategy tracked by its
    library identity would previously all silently disappear -- the
    Dashboard/Validate checklist would keep reading 0/5 even though every
    check had genuinely been run.

    Pass report_files (e.g. {"html": <abs path>, "json": <abs path>} --
    the actual filesystem paths a tool's own generate_..._report() call
    returned) to also mirror this run's report into the strategy's
    consolidated reports/by_strategy/<strategy>/ folder -- see
    app.reports.strategy_folder. Optional: a caller with only a URL
    (report_html) and no filesystem path can omit this and the checklist
    entry is still recorded, just without a copy in the per-strategy
    folder."""
    if kind not in VALIDATION_KINDS:
        return
    try:
        key = _candidate_keys(strategy_name, instrument, library_type, library_filename)[0]
        store = _load_checklist_store()
        entry = store.setdefault(key, {"strategy_name": strategy_name, "instrument": instrument, "checks": {}})
        entry["strategy_name"] = strategy_name
        entry["instrument"] = instrument
        if library_type and library_filename:
            entry["library_type"] = library_type
            entry["library_filename"] = library_filename
        entry.setdefault("checks", {})[kind] = {
            "ran": True,
            "passed": passed,
            "summary": summary,
            "report_html": report_html,
            "recorded_at": time.time(),
        }
        _save_checklist_store(store)
    except Exception:
        pass

    if report_files:
        try:
            from app.reports.strategy_folder import record_report

            record_report(
                strategy_name, instrument, kind,
                library_type=library_type, library_filename=library_filename,
                passed=passed, summary=summary, source_files=report_files,
            )
        except Exception:
            pass


def get_checklist(strategy_name: str, instrument: str, *, library_type: str = "", library_filename: str = "") -> dict:
    """Always returns every kind in VALIDATION_KINDS, defaulting an unset
    one to {"ran": False, "passed": None, "summary": "", "report_html": ""}.

    Looks up the library-identity key first (when library_type/
    library_filename are given), then falls back to the legacy
    name+instrument key -- so a strategy tracked by its library identity
    still finds checks that were recorded (by an older build, or by a
    caller that couldn't resolve a library match) under the name-only
    key, rather than silently reporting everything as not-run."""
    store = _load_checklist_store()
    checks: dict = {}
    for key in _candidate_keys(strategy_name, instrument, library_type, library_filename):
        candidate_checks = store.get(key, {}).get("checks", {})
        if candidate_checks:
            checks = candidate_checks
            break
    return {
        kind: checks.get(kind, {"ran": False, "passed": None, "summary": "", "report_html": ""})
        for kind in VALIDATION_KINDS
    }


def robustness_score(strategy_name: str, instrument: str, *, library_type: str = "", library_filename: str = "") -> dict:
    """A transparent count, not a fabricated single number: how many of
    the 5 checks have been run at all, and of those, how many passed
    (only where the tool itself computes a verdict -- Sensitivity's
    passed=None entries are excluded from decided_count/passed_count,
    not counted as failures)."""
    checklist = get_checklist(strategy_name, instrument, library_type=library_type, library_filename=library_filename)
    ran = [c for c in checklist.values() if c["ran"]]
    decided = [c for c in ran if c["passed"] is not None]
    passed = [c for c in decided if c["passed"]]
    return {
        "ran_count": len(ran),
        "total_count": len(VALIDATION_KINDS),
        "decided_count": len(decided),
        "passed_count": len(passed),
        "pct_run": round(100.0 * len(ran) / len(VALIDATION_KINDS), 1),
        "pct_passed_of_decided": round(100.0 * len(passed) / len(decided), 1) if decided else None,
    }
