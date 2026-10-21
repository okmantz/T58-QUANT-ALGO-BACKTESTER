"""Shared fixtures: every test gets its own throw-away SQLite file; the poller is never started."""
import pytest

from app.global_events import store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("T58_GLOBAL_EVENTS_DB", str(tmp_path / "ge.db"))
    monkeypatch.setenv("T58_GLOBAL_EVENTS_POLLER", "0")
    store._initialised.clear()
    yield
    store._initialised.clear()
