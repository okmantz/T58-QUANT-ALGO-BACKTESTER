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

Project tagging (added for the Project Chat feature -- see
app.orchestration.projects / app.web.project_routes):
  - ``create()`` auto-stamps a ``project_id`` field on every new job IF
    the caller didn't already pass one explicitly AND an "active
    project" getter has been registered via
    ``set_active_project_getter()``. This is deliberately NOT a change
    to any of the ~18 existing ``JOB_MANAGER.create(...)`` call sites in
    app.web.server -- every one of them keeps working exactly as before
    (project_id just comes back as None) unless project_routes.py's
    getter is wired in AND a project is actually active for the current
    request. This is how Search Lab, WFO, CPCV, Quick Optimize, and
    every other job type get associated with whichever project is open
    in the floating chat widget with zero per-tool code changes.
  - ``list_jobs()`` is the read-only feed app.web.project_routes polls
    for a project's Activity tab: every known job (optionally filtered
    to one project_id), newest first, each carrying its own job_id.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Optional


class JobManager:
    """Thread-safe tracker for background jobs identified by a short hex
    id. One instance is meant to be shared by every job-based tool that
    migrates onto it (see app.web.server's module-level ``JOB_MANAGER``)
    -- ids are globally unique (uuid4), so multiple tools sharing one
    instance never collide."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        # Set via set_active_project_getter() (see that method's own
        # docstring) -- instance-level, deliberately NOT a module global,
        # so a test (or a future second JobManager) can register its own
        # getter without affecting any other instance.
        self._active_project_getter: Optional[Callable[[], Optional[str]]] = None

    def set_active_project_getter(self, getter: Optional[Callable[[], Optional[str]]]) -> None:
        """Registers the function create() calls (with no arguments) to
        find "whichever project is currently active for this request",
        so a newly created job can be auto-tagged with it -- see
        app.web.project_routes, which calls this once at import time
        against the shared JOB_MANAGER singleton below. Pass None to
        unregister (mainly for test isolation). Never raises itself; a
        getter that raises when CALLED is treated as "no active project"
        -- see create()'s own try/except around that call."""
        self._active_project_getter = getter

    def create(self, **initial: Any) -> str:
        """Registers a new job and returns its id. Always sets
        done=False, error=None, result=None, log=[], started_at=now,
        project_id=None unless the caller explicitly overrides one of
        those via ``initial`` (e.g. passing a non-empty starting
        ``log``, or an explicit ``project_id``).

        project_id auto-tagging: if the caller didn't pass project_id
        explicitly AND set_active_project_getter() has registered a
        getter, that getter is called (with no args) and its result
        (if any) becomes this job's project_id. This is deliberately NOT
        a change to any pre-existing call site -- every job type keeps
        working exactly as before (project_id just comes back None)
        unless a getter has actually been registered AND returns
        something truthy for the current call."""
        job_id = uuid.uuid4().hex[:12]
        job = {
            "log": [],
            "done": False,
            "error": None,
            "result": None,
            "started_at": time.time(),
            "project_id": None,
        }
        job.update(initial)
        if job.get("project_id") is None and self._active_project_getter is not None:
            try:
                job["project_id"] = self._active_project_getter()
            except Exception:
                pass  # a broken getter must never break job creation itself
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

    def list_jobs(self, project_id: Optional[str] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Returns shallow copies of jobs, each with its own ``job_id``
        merged in, newest (by started_at) first. Pass project_id to see
        only jobs tagged with that project (see create()'s auto-tagging
        above); omit it to see every job regardless of project. This is
        read-only and non-destructive -- it never prunes or mutates
        anything, so it's safe to call from a polling route on every
        request. ``limit`` caps how many are returned after sorting."""
        with self._lock:
            snapshot = [
                {"job_id": jid, **job} for jid, job in self._jobs.items()
                if project_id is None or job.get("project_id") == project_id
            ]
        snapshot.sort(key=lambda j: j.get("started_at", 0.0), reverse=True)
        if limit is not None:
            snapshot = snapshot[:limit]
        return snapshot


# Shared singleton -- every job-based tool in app.web.server (Search Lab,
# WFO, CPCV, Quick Optimize, Full Pipeline, Multi-Objective, Multi-Market,
# ...) and the newer app.web.project_routes blueprint import THIS instance
# rather than constructing their own, so ids stay globally unique and a
# job started from any tool is visible to the Activity feed. Living here
# (rather than being instantiated in server.py, which imports this
# module) avoids a circular import between server.py and project_routes.py.
JOB_MANAGER = JobManager()
