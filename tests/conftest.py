"""Repo-wide pytest fixtures.

FIX (audit, Sep 2026): this file didn't exist before. The CI failures
that prompted this fix (see CHANGES_SUMMARY.md) were two tests in
tests/test_web_speed_run_loop_mode.py flaking ONLY when run as part of
the full ~1650-test suite (`pytest tests`), never in isolation --
exactly the signature of state leaking between tests rather than a bug
in either test itself.

The concrete mechanism: app.orchestration.resource_guard.HEAVY_JOB_GUARD
is a single process-wide named-slot lock, imported as one shared
singleton by both app.web.server and app.ui.main_window, and every test
in this suite runs inside the SAME Python process (pytest doesn't fork
a fresh interpreter per test). If any test that starts a heavy job
(Search Lab, Evolution Lab, Forge, Speed Run, Full Pipeline, Quick
Optimize, Walk-Forward GA, Multi-Objective, WFO, CPCV -- see
resource_guard.py's own docstring for the full list) ends without its
background thread reaching the `finally: HEAVY_JOB_GUARD.release(...)`
that normally frees its slot -- a raised exception in a place that
skips the finally, a daemon thread still mid-run when the test function
returns, an assertion failure that skips cleanup code below it -- the
guard's slot stays claimed for the rest of the ENTIRE test session,
regardless of which test file runs next. Every subsequent test that
tries to start a DIFFERENT heavy job then gets refused (a 409 response
instead of the 302 redirect it expected), which is exactly the kind of
mismatch that produces confusing downstream errors far from the actual
cause -- e.g. a test extracting a job id from a redirect Location
header that was never sent.

This was already partly self-healing for two job types (Evolution Lab
and Multi-Instrument Evolution Lab each register a HEAVY_JOB_GUARD
health check in app/web/server.py, so a stuck slot for THOSE two
self-clears the next time anything tries to acquire the guard) but
every other job type had no such check, so a leak from any of THOSE
could still wedge the whole rest of the suite. Search Lab, Forge, Speed
Run, and Multi-Instrument Speed Run each now register one too (see
app/web/server.py) as defense-in-depth for a real running app, not just
for tests. But relying on every current and future heavy-job type
remembering to register a health check is fragile, and doesn't help at
all for a job type whose health check itself raises (try_acquire
treats a broken check as "still active", by design, so it fails safe
rather than falsely freeing a slot that might be genuinely busy) or for
non-HEAVY_JOB_GUARD global state a test might leak (e.g. a job-tracking
dict that isn't cleared).

The robust, general fix for the TEST SUITE specifically -- as opposed
to a real running app, where the guard's job-name slot has to persist
across requests on purpose -- is to guarantee every test starts and
ends with a clean HEAVY_JOB_GUARD, full stop, regardless of what any
other test did or how it exited. That's what this fixture does. It is
test-only (this file is never imported by the application itself) and
changes no production behavior.
"""
from __future__ import annotations

import pytest

from app.orchestration.resource_guard import HEAVY_JOB_GUARD


@pytest.fixture(autouse=True)
def _reset_heavy_job_guard():
    """Runs around every single test in this suite. Clears whatever
    HEAVY_JOB_GUARD slot is held before the test starts (so a leak from
    an EARLIER test can never block this one) and again after it ends
    (so a leak from THIS test can never block a LATER one) -- belt and
    suspenders, since either direction alone would still let one bad
    test corrupt every test around it.

    Uses release() rather than reaching into the guard's private
    _active_name -- release(name) is a no-op if name doesn't hold the
    slot, so this is always safe to call even when nothing is held."""
    stuck = HEAVY_JOB_GUARD.active_name
    if stuck is not None:
        HEAVY_JOB_GUARD.release(stuck)
    yield
    stuck = HEAVY_JOB_GUARD.active_name
    if stuck is not None:
        HEAVY_JOB_GUARD.release(stuck)
