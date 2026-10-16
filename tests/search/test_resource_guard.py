"""Tests for app.orchestration.resource_guard.HeavyJobGuard -- in
particular the health-check self-heal, added after a real bug where
Evolution Lab's slot could stay held forever (its only release path was
the web app's /evolution/status.json route noticing is_running had gone
False -- if nothing polled it, or its background thread got wedged, every
OTHER heavy job -- Search Lab included -- was refused indefinitely with
no way to recover short of restarting the server)."""
from app.orchestration.resource_guard import HeavyJobGuard


def test_try_acquire_blocks_a_second_job_while_first_is_active():
    guard = HeavyJobGuard()
    assert guard.try_acquire("A") is True
    assert guard.try_acquire("B") is False
    assert guard.active_name == "A"


def test_release_frees_the_slot_for_another_job():
    guard = HeavyJobGuard()
    guard.try_acquire("A")
    guard.release("A")
    assert guard.active_name is None
    assert guard.try_acquire("B") is True


def test_release_is_a_no_op_if_a_different_job_holds_the_slot():
    guard = HeavyJobGuard()
    guard.try_acquire("A")
    guard.release("B")  # B never held it -- must not clear A's slot
    assert guard.active_name == "A"


def test_self_heals_a_stale_slot_via_registered_health_check():
    """The core regression case: job A's slot is held, but A's own
    health check now reports it's no longer actually active (its thread
    died/finished without ever calling release()). A second job must be
    able to acquire instead of being refused forever."""
    guard = HeavyJobGuard()
    still_running = {"A": False}  # A has already stopped for real
    guard.register_health_check("A", lambda: still_running["A"])
    guard.try_acquire("A")

    assert guard.try_acquire("B") is True
    assert guard.active_name == "B"


def test_does_not_self_heal_while_health_check_reports_active():
    guard = HeavyJobGuard()
    still_running = {"A": True}
    guard.register_health_check("A", lambda: still_running["A"])
    guard.try_acquire("A")

    assert guard.try_acquire("B") is False
    assert guard.active_name == "A"

    still_running["A"] = False
    assert guard.try_acquire("B") is True


def test_a_broken_health_check_fails_safe_and_keeps_the_slot_held():
    guard = HeavyJobGuard()

    def _boom():
        raise RuntimeError("boom")

    guard.register_health_check("A", _boom)
    guard.try_acquire("A")

    assert guard.try_acquire("B") is False
    assert guard.active_name == "A"


def test_unregistered_job_names_behave_exactly_as_before():
    guard = HeavyJobGuard()
    guard.try_acquire("A")
    assert guard.try_acquire("B") is False  # no health check for "A" -- no self-heal, unchanged behavior
