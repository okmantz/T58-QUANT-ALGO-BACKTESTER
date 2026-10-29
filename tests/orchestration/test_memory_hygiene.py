"""Memory hygiene: the Full Pipeline must give memory back, and the guards must notice swap, not just free RAM."""
from __future__ import annotations

import types

import pytest

from app.orchestration import memory_hygiene as mh
from app.orchestration import resource_guard as rg


def test_release_memory_never_raises_and_reports_rss():
    out = mh.release_memory("test")
    assert out["reason"] == "test" and "rss_before" in out and "rss_after" in out


def test_status_has_the_fields_the_ui_needs():
    st = mh.memory_status()
    assert "pid" in st and "history" in st and "ollama_models" in st


def test_pressure_triggers_on_heavy_swap_even_with_free_ram(monkeypatch):
    if mh.psutil is None:
        pytest.skip("psutil not installed")
    gb = 1024 ** 3
    monkeypatch.setattr(mh.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=16 * gb, available=8 * gb))
    monkeypatch.setattr(mh.psutil, "swap_memory", lambda: types.SimpleNamespace(total=8 * gb, used=6 * gb))
    assert mh.memory_pressure() is True
    monkeypatch.setattr(mh.psutil, "swap_memory", lambda: types.SimpleNamespace(total=8 * gb, used=0))
    assert mh.memory_pressure() is False


def test_resource_guard_pressure_check_is_swap_aware(monkeypatch):
    if rg.psutil is None:
        pytest.skip("psutil not installed")
    gb = 1024 ** 3
    monkeypatch.setattr(rg.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=16 * gb, available=10 * gb))
    monkeypatch.setattr(rg.psutil, "swap_memory", lambda: types.SimpleNamespace(total=8 * gb, used=6 * gb))
    assert rg.memory_pressure_ok() is False
    assert rg.shrink_pool_for_memory_pressure(8) == 4


def test_releasing_the_heavy_job_slot_triggers_cleanup(monkeypatch):
    seen = []
    monkeypatch.setattr(mh, "release_memory", lambda reason="": seen.append(reason) or {})
    guard = rg.HeavyJobGuard()
    assert guard.try_acquire("job-x")
    guard.release("job-x")
    import time

    for _ in range(50):
        if seen:
            break
        time.sleep(0.02)
    assert seen == ["after:job-x"]


def test_batch_outcomes_do_not_retain_full_results_by_default():
    from app.orchestration.full_pipeline import FullPipelineBatchOutcome

    assert FullPipelineBatchOutcome("x", ok=True).result is None
    import inspect
    from app.orchestration import full_pipeline

    assert inspect.signature(full_pipeline.run_full_pipeline_batch).parameters["keep_results"].default is False


def test_idle_cleanup_closes_pools_and_clears_caches():
    done = mh.idle_cleanup()
    assert done.get("stage_pool") == "closed" and "resampled bars" in done.get("caches_cleared", [])


def test_idle_watchdog_can_be_started_twice_and_stopped():
    mh.start_idle_watchdog(interval=3600)
    mh.start_idle_watchdog(interval=3600)
    mh.stop_idle_watchdog()
