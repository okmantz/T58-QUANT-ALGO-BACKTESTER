"""Runs "Start forward test" in the background and reports live progress.

Why this exists: connecting can legitimately take minutes the first time (installing Python + the MetaTrader5 package
inside Wine, opening MT5, waiting for it to load). Doing that inside the HTTP request left the page frozen on
"Starting..." and then dumped one raw error string. Now the request returns at once; this job does the work in a thread
and the page polls ``snapshot()`` to draw a checklist, and, if something fails, a plain-language problem card.

One job at a time (a forward test owns the single global MT5 connection).
"""
from __future__ import annotations

import threading
import time
import traceback

from app.forward_test import mt5_diagnostics as diag

STEPS = [
    ("prepare", "Check your MT5 setup"),
    ("helper", "Start T58's MT5 connection helper"),
    ("terminal", "Open MetaTrader 5"),
    ("login", "Log in to your account"),
    ("trading", "Confirm Algo Trading is on"),
    ("strategy", "Start your strategy"),
]
_ORDER = [s for s, _ in STEPS]


class StartJob:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._reset_locked()

    # ---- state -------------------------------------------------------------------------------------------
    def _reset_locked(self) -> None:
        self.state = "idle"          # idle | running | ok | error
        self.current = None
        self.texts: dict = {}
        self.done: set = set()
        self.problem: dict | None = None
        self.error_step: str | None = None
        self.started_at = 0.0
        self.finished_at = 0.0

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self.state == "running"

    def progress(self, step: str, text: str = "") -> None:
        """Called from the connector/engine threads. Entering a step completes every earlier step."""
        if step not in _ORDER:
            step = self.current or "prepare"
        with self._lock:
            idx = _ORDER.index(step)
            for s in _ORDER[:idx]:
                self.done.add(s)
            self.current = step
            if text:
                self.texts[step] = text

    def snapshot(self) -> dict:
        with self._lock:
            steps = []
            for sid, label in STEPS:
                if sid in self.done or self.state == "ok":
                    st = "done"
                elif self.state == "error" and sid == self.error_step:
                    st = "error"
                elif self.state == "running" and sid == self.current:
                    st = "active"
                else:
                    st = "pending"
                steps.append({"id": sid, "label": label, "status": st, "text": self.texts.get(sid, "")})
            end = self.finished_at or time.time()
            return {"state": self.state, "steps": steps, "problem": self.problem,
                    "elapsed": int(end - self.started_at) if self.started_at else 0,
                    "current_text": self.texts.get(self.current or "", "")}

    # ---- run ---------------------------------------------------------------------------------------------
    def begin(self, session, on_success) -> bool:
        """Starts ``session.start()`` in a thread. False if a job is already running."""
        with self._lock:
            if self.state == "running":
                return False
            self._reset_locked()
            self.state = "running"
            self.current = "prepare"
            self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, args=(session, on_success), daemon=True, name="t58-ft-start")
        self._thread.start()
        return True

    def _run(self, session, on_success) -> None:
        try:
            ok, message = session.start(progress=self.progress)
            if ok:
                try:
                    on_success(session)
                finally:
                    with self._lock:
                        self.state = "ok"
                        self.finished_at = time.time()
                return
            problem = getattr(session, "last_problem", None) or diag.unknown_problem(message).to_dict()
        except Exception as exc:  # noqa: BLE001 -- the thread must always end in a visible state
            problem = diag.unknown_problem(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}").to_dict()
        with self._lock:
            self.error_step = self.current or "prepare"
            self.problem = problem
            self.state = "error"
            self.finished_at = time.time()

    def clear(self) -> None:
        with self._lock:
            if self.state != "running":
                self._reset_locked()
