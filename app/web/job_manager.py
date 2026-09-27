"""
Generic in-memory background-job tracker.

app.web.server hand-rolls this exact pattern independently for roughly
18 different long-running tools (Search Lab, Full Pipeline, Walk-Forward
Opt, Walk-Forward GA, CPCV, PBO, Sensitivity, Parameter Robustness, Quick
Optimize, Multi-Objective, Research Agent, Research Loop, Multi-Market,
Multi-Instrument Evolution, ...): a module-level ``dict[str, dict]`` + a
``threading.Lock`` + a ``threading.Thread(daemon=True)`` whose target
mutates the dict under the lock, plus a hand-written "append one log
line" helper and a ``/<tool>/job/<id>/status.json`` route that reads it
back out. Every one of the ~18 copies has to be kept correct
independently -- a fix to one (proper pruning, a race in the log-append
helper, a consistent shape for the status payload) doesn't reach the
other 17.

This module is the shared primitive that pattern should be built on.
``app.web.server``'s Walk-Forward Opt and CPCV job runners have been
migrated onto it as a concrete, working example (see ``_run_wfo_job``/
``_run_cpcv_job`` and their ``/start``/``/job/<id>``/``/status.json``
routes) -- every other job type listed above still uses its own
hand-rolled dict-and-lock for now. Deliberately NOT a big-bang rewrite of
all 18 at once: this file has no test coverage of its own yet beyond
``tests/test_job_manager.py``, and migrating every long-running tool in
a single pass, in a codebase this size, with no way to exercise the real
UI here, is a good way to silently break one of them. The intended path
is to migrate the rest incrementally, one tool per change, following the
exact shape WFO/CPCV now use.

Thread-safety: every method takes an internal lock; job dicts handed
back to callers are shallow copies, so a caller mutating what ``get()``
returned never corrupts the manager's own state.

Design notes:
  - ``create()`` returns a fresh job_id and stores the initial fields --
    call it BEFORE starting the worker thread (exactly like the old
    ``_WFO_JOBS[job_id] = {...}`` pattern), then pass that job_id into
    the thread's target so it can call back into ``log()``/``update()``/
    ``finish()``/``fail()`` as it progresses.
  - ``update()`` merges fields into an existing job (e.g. stashing a
    non-JSON-safe ``result`` object, or a ``report_html`` URL) without
    disturbing its log or done/error state.
  - ``finish()``/``fail()`` are ``update()`` with ``done`` set for you,
    so a job runner can't accidentally leave ``done`` unset on one exit
    path and hang the status page forever polling.
  - ``prune()`` evicts finished jobs older than a max age. Nothing here
    calls it automatically (a caller decides its own cadence -- e.g. once
    per new job created is enough for how infrequently these heavy jobs
    run) -- see app.web.server's ``_prune_old_jobs`` for the convention
    the migrated routes use. This is what keeps a long-uptime install's
    job dict from growing without bound, which the fully-manual version
    of this pattern never addressed for any of the 18 tools.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional


class JobManager:
    """Thread-safe tracker for background jobs identified by a short hex
    id. One instance is meant to be shared by every job-based tool that
    migrates onto it (see app.web.server's module-level ``JOB_MANAGER``)
    -- ids are globally unique (uuid4), so multiple tools sharing one
    instance never collide."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, **initial: Any) -> str:
        """Registers a new job and returns its id. Always sets
        done=False, error=None, result=None, log=[], started_at=now
        unless the caller explicitly overrides one of those via
        ``initial`` (e.g. passing a non-empty starting ``log``)."""
        job_id = uuid.uuid4().hex[:12]
        job = {
            "log": [],
            "done": False,
            "error": None,
            "result": None,
            "started_at": time.time(),
        }
        job.update(initial)
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Returns a shallow copy of the job's current state, or None if
        no job with this id exists (never raises KeyError)."""
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def log(self, job_id: str, message: str) -> None:
        """Appends one line to a job's log. Silently does nothing if the
        job id is unknown (e.g. the process restarted and lost in-memory
        state) -- a progress line is never worth raising over."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job["log"].append(message)

    def update(self, job_id: str, **fields: Any) -> None:
        """Merges fields into a job's stored dict. Silently does nothing
        for an unknown job id."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(fields)

    def finish(self, job_id: str, **fields: Any) -> None:
        """update() plus done=True -- the single call every job runner's
        success path should end on, so "did this job finish" is never
        ambiguous."""
        self.update(job_id, done=True, **fields)

    def fail(self, job_id: str, error: str) -> None:
        """update() plus done=True, error=<message> -- the single call
        every job runner's except-block should end on."""
        self.update(job_id, done=True, error=error)

    def prune(self, max_age_seconds: float) -> int:
        """Evicts finished (done=True) jobs whose started_at is older
        than max_age_seconds. Returns the number evicted. Never touches
        a still-running job, however old -- a slow overnight job must
        never be pruned out from under its own status page."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            stale = [
                jid for jid, job in self._jobs.items()
                if job.get("done") and job.get("started_at", time.time()) < cutoff
            ]
            for jid in stale:
                del self._jobs[jid]
            return len(stale)

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)
