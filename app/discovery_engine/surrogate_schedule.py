"""Keep the surrogate-study report fresh without anyone remembering to run it (2026-11-05 audit).

The v13 family batches stay locked until ``reports/surrogate_gate.json`` exists, is under 90 days old, matches the current search code
and shows the lab's false-discovery rate and power. That is the right design, but a report that does not ship and silently goes stale
leaves ~92 families idle. This module makes the study a scheduled background job:

    study_due()                  is the gate closed for a reason a fresh study would fix (missing / stale / code changed / too thin)?
    maybe_start_background_study()   start ``python -m scripts.run_surrogate_study`` at LOW priority, at most one at a time and at most one
                                 attempt per ``MIN_RETRY_DAYS``; never raises; a no-op under pytest or when disabled.
    study_status()               plain-language state for the UI ("locked: study queued / running / not due").

The gate itself is untouched: nothing here opens it or lowers any bar. Only a completed study that clears the thresholds does.
Disable with T58_AUTO_SURROGATE_STUDY=0.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MIN_RETRY_DAYS = 7.0
_ROOT = Path(__file__).resolve().parents[2]
# Reasons a re-run can fix. A report that RAN and showed a high false-discovery rate or low power is a finding, not a staleness problem.
_FIXABLE = ("no surrogate-study report", "schema", "older than", "code changed", "only ", "too few planted")


def _state_path() -> Path:
    from app.discovery_engine.surrogate import report_path
    return report_path().with_name("surrogate_study_state.json")


def _read_state() -> dict:
    try:
        d = json.loads(_state_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _write_state(d: dict) -> None:
    try:
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        if sys.platform.startswith("win"):
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, timeout=5).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def study_due(path: Path | None = None) -> dict:
    """{"due": bool, "reasons": [...]}: due when the gate is closed and every closing reason is one a fresh study can fix."""
    from app.discovery_engine.surrogate import gate_status
    st = gate_status(path)
    if st["open"]:
        return {"due": False, "reasons": []}
    fixable = [r for r in st["reasons"] if any(k in r for k in _FIXABLE)]
    only_fixable = bool(fixable) and len(fixable) == len(st["reasons"])
    return {"due": only_fixable, "reasons": st["reasons"]}


def study_status() -> dict:
    """{"state": open|running|queued|waiting|measured_failed, "text": str}"""
    from app.discovery_engine.surrogate import gate_status
    st = gate_status()
    if st["open"]:
        return {"state": "open", "text": "Discovery families: unlocked (surrogate study is fresh)."}
    s = _read_state()
    if s.get("pid") and _pid_alive(s["pid"]):
        return {"state": "running", "text": "Discovery families: locked while the surrogate study runs in the background (it re-measures the false-discovery rate and power)."}
    due = study_due()
    if due["due"]:
        return {"state": "queued", "text": "Discovery families: locked until the surrogate study is refreshed; it starts automatically in the background. " + "; ".join(due["reasons"])}
    return {"state": "measured_failed", "text": "Discovery families: locked because the surrogate study measured a problem: " + "; ".join(st["reasons"])}


def maybe_start_background_study(*, force: bool = False, now: float | None = None) -> dict:
    """Start the study if it is due. Returns {"started": bool, "why": str}. Never raises."""
    try:
        if os.environ.get("T58_AUTO_SURROGATE_STUDY", "1") == "0" and not force:
            return {"started": False, "why": "disabled by T58_AUTO_SURROGATE_STUDY=0"}
        if "pytest" in sys.modules and not force:
            return {"started": False, "why": "not started under pytest"}
        due = study_due()
        if not due["due"] and not force:
            return {"started": False, "why": "gate is open or the study is not what is blocking it"}
        s = _read_state()
        if s.get("pid") and _pid_alive(s["pid"]):
            return {"started": False, "why": "a study is already running"}
        now = time.time() if now is None else now
        if not force and (now - float(s.get("last_attempt", 0))) / 86400.0 < MIN_RETRY_DAYS:
            return {"started": False, "why": f"last attempt was under {MIN_RETRY_DAYS:g} days ago"}
        log = _state_path().with_name("surrogate_study.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.pop("T58_KEEP_WORKER_PRIORITY", None)        # children lower their own priority (app/__init__.py)
        kw: dict = {"cwd": str(_ROOT), "env": env, "stdout": open(log, "ab"), "stderr": subprocess.STDOUT}
        if sys.platform.startswith("win"):
            kw["creationflags"] = 0x00004000 | 0x08000000     # BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW
        else:
            kw["preexec_fn"] = lambda: os.nice(10)
        proc = subprocess.Popen([sys.executable, "-m", "scripts.run_surrogate_study"], **kw)
        _write_state({"pid": proc.pid, "started_at": now, "last_attempt": now})
        return {"started": True, "why": "gate closed for a fixable reason: " + "; ".join(due["reasons"])}
    except Exception as exc:  # noqa: BLE001
        return {"started": False, "why": f"could not start: {exc}"}
