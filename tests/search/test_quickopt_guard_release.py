"""Regression tests: a finished Quick Optimize must free the shared
HEAVY_JOB_GUARD slot (Owen, 2026-10-06).

Bug report: Quick Optimize finished, but starting Full Pipeline right
after was refused with "Quick Optimize is already running on this
server." Root cause: /recovery/quick-optimize acquires JOB_QUICK_OPTIMIZE
from HEAVY_JOB_GUARD, but the background runners (_run_quickopt_job /
_run_quickopt_sweep_job) never released it -- the only release() calls
were in the recovery route's synchronous setup error paths -- and no
health check was registered for the slot, so it never self-healed.
Every other heavy job stayed refused until a server restart.

These tests drive the real runner functions synchronously with the GA
entry points stubbed out, so they are fast and deterministic. The
guard is a process-wide singleton, so an autouse fixture leaves it
free (and the active-job registry empty) around every test.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.web.server as server
from app.orchestration.resource_guard import JOB_FULL_PIPELINE, JOB_QUICK_OPTIMIZE
from app.optimize.walkforward_ga import WalkforwardGACancelled

_GUARD = server.HEAVY_JOB_GUARD
_MANAGER = server.JOB_MANAGER


@pytest.fixture(autouse=True)
def _clean_guard_state():
    _GUARD.release(JOB_QUICK_OPTIMIZE)
    _GUARD.release(JOB_FULL_PIPELINE)
    getattr(server, "_QUICKOPT_ACTIVE_JOB_IDS", set()).clear()
    yield
    _GUARD.release(JOB_QUICK_OPTIMIZE)
    _GUARD.release(JOB_FULL_PIPELINE)
    getattr(server, "_QUICKOPT_ACTIVE_JOB_IDS", set()).clear()


def _make_job() -> str:
    return _MANAGER.create(log=["test job"], instrument="TEST")


def test_completed_quickopt_job_releases_the_guard(monkeypatch):
    """Owen's exact flow: guard acquired (recovery route), QO job runs
    to completion, then Full Pipeline must be allowed to start."""
    monkeypatch.setattr(server, "run_quick_optimize", lambda *a, **k: SimpleNamespace())
    assert _GUARD.try_acquire(JOB_QUICK_OPTIMIZE) is True

    server._run_quickopt_job(_make_job(), None, None, None, None, None)

    assert _GUARD.active_name is None
    assert _GUARD.try_acquire(JOB_FULL_PIPELINE) is True


def test_failed_quickopt_job_releases_the_guard(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "run_quick_optimize", _boom)
    assert _GUARD.try_acquire(JOB_QUICK_OPTIMIZE) is True

    server._run_quickopt_job(_make_job(), None, None, None, None, None)

    assert _GUARD.active_name is None


def test_cancelled_quickopt_job_releases_the_guard(monkeypatch):
    def _cancelled(*a, **k):
        raise WalkforwardGACancelled("stopped")

    monkeypatch.setattr(server, "run_quick_optimize", _cancelled)
    assert _GUARD.try_acquire(JOB_QUICK_OPTIMIZE) is True

    server._run_quickopt_job(_make_job(), None, None, None, None, None)

    assert _GUARD.active_name is None


def test_completed_sweep_job_releases_the_guard(monkeypatch):
    sweep = SimpleNamespace(
        per_timeframe={}, skipped=[], errors={},
        best_result=SimpleNamespace(), best_timeframe=None,
    )
    monkeypatch.setattr(server, "run_quick_optimize_sweep", lambda *a, **k: sweep)
    assert _GUARD.try_acquire(JOB_QUICK_OPTIMIZE) is True

    server._run_quickopt_sweep_job(_make_job(), None, None, None, None, None, ["5"])

    assert _GUARD.active_name is None


def test_guard_self_heals_when_the_slot_leaks_with_no_job_running():
    """Defense in depth (the guard's own registered-health-check
    mechanism): a leaked Quick Optimize slot with no QO job actually
    running must not block the next heavy job."""
    assert _GUARD.try_acquire(JOB_QUICK_OPTIMIZE) is True  # leaked slot, no job
    assert _GUARD.try_acquire(JOB_FULL_PIPELINE) is True  # self-healed


def test_guard_stays_held_while_a_quickopt_job_is_genuinely_running():
    """The health check must not free the slot early: while a QO job
    is registered active and not done, other heavy jobs stay refused;
    once it finishes, the slot frees on the next acquire attempt."""
    job_id = _make_job()
    server._QUICKOPT_ACTIVE_JOB_IDS.add(job_id)
    assert _GUARD.try_acquire(JOB_QUICK_OPTIMIZE) is True
    assert _GUARD.try_acquire(JOB_FULL_PIPELINE) is False

    _MANAGER.finish(job_id)
    assert _GUARD.try_acquire(JOB_FULL_PIPELINE) is True
