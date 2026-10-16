"""Tests for app.orchestration.projects -- the persistent Project data
model backing the floating Project Chat widget (Phase 1/2 of the
StarNet-style feature; see that module's own docstring).

Mirrors app.ai.ollama_settings's own test convention exactly:
monkeypatch get_app_base_dir() to a pytest tmp_path so nothing here ever
touches a developer's real app-data folder."""
from __future__ import annotations

import pytest

from app.orchestration import projects


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(projects, "get_app_base_dir", lambda: tmp_path)


def test_create_project_persists_and_is_returned_by_list():
    project = projects.create_project("Falsification Kit Rebuild")
    assert project["name"] == "Falsification Kit Rebuild"
    assert project["chat_history"] == []
    assert project["created_at"] == project["updated_at"]

    listed = projects.list_projects()
    assert len(listed) == 1
    assert listed[0]["id"] == project["id"]
    assert listed[0]["name"] == "Falsification Kit Rebuild"
    assert listed[0]["message_count"] == 0


def test_create_project_blank_name_gets_a_sane_default():
    project = projects.create_project("   ")
    assert project["name"].startswith("Project ")


def test_get_project_round_trips_full_data():
    created = projects.create_project("Freight Signal")
    fetched = projects.get_project(created["id"])
    assert fetched == created


def test_get_unknown_project_raises_not_found():
    with pytest.raises(projects.ProjectNotFound):
        projects.get_project("does-not-exist")


def test_project_exists():
    created = projects.create_project("X")
    assert projects.project_exists(created["id"]) is True
    assert projects.project_exists("nope") is False


def test_rename_project_updates_name_and_timestamp():
    created = projects.create_project("Old Name")
    renamed = projects.rename_project(created["id"], "New Name")
    assert renamed["name"] == "New Name"
    assert renamed["updated_at"] >= created["updated_at"]
    assert projects.get_project(created["id"])["name"] == "New Name"


def test_rename_project_rejects_blank_name():
    created = projects.create_project("Keep Me")
    with pytest.raises(ValueError):
        projects.rename_project(created["id"], "   ")


def test_rename_unknown_project_raises_not_found():
    with pytest.raises(projects.ProjectNotFound):
        projects.rename_project("nope", "New Name")


def test_delete_project_removes_it():
    created = projects.create_project("Temp")
    projects.delete_project(created["id"])
    assert projects.project_exists(created["id"]) is False
    assert projects.list_projects() == []


def test_delete_unknown_project_raises_not_found():
    with pytest.raises(projects.ProjectNotFound):
        projects.delete_project("nope")


def test_append_chat_message_persists_in_order():
    created = projects.create_project("Chatty")
    projects.append_chat_message(created["id"], "user", "hello")
    updated = projects.append_chat_message(created["id"], "assistant", "hi there")

    assert [t["role"] for t in updated["chat_history"]] == ["user", "assistant"]
    assert [t["content"] for t in updated["chat_history"]] == ["hello", "hi there"]
    assert all("ts" in t for t in updated["chat_history"])

    # Persisted, not just returned -- a fresh read sees the same history.
    reread = projects.get_project(created["id"])
    assert reread["chat_history"] == updated["chat_history"]


def test_append_chat_message_bumps_message_count_in_list():
    created = projects.create_project("Counted")
    projects.append_chat_message(created["id"], "user", "one")
    projects.append_chat_message(created["id"], "assistant", "two")
    listed = projects.list_projects()
    assert listed[0]["message_count"] == 2


def test_append_chat_message_rejects_unknown_role():
    created = projects.create_project("Strict")
    with pytest.raises(ValueError):
        projects.append_chat_message(created["id"], "system", "nope")


def test_append_chat_message_unknown_project_raises_not_found():
    with pytest.raises(projects.ProjectNotFound):
        projects.append_chat_message("nope", "user", "hi")


def test_stored_history_is_trimmed_to_max_stored_messages(monkeypatch):
    monkeypatch.setattr(projects, "MAX_STORED_MESSAGES", 4)
    created = projects.create_project("Trimmed")
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        projects.append_chat_message(created["id"], role, f"message {i}")

    history = projects.get_project(created["id"])["chat_history"]
    assert len(history) == 4
    # Oldest two ("message 0", "message 1") dropped -- the most recent
    # MAX_STORED_MESSAGES are kept, in order.
    assert [t["content"] for t in history] == ["message 2", "message 3", "message 4", "message 5"]


def test_get_chat_history_respects_limit():
    created = projects.create_project("Limited")
    for i in range(5):
        projects.append_chat_message(created["id"], "user", str(i))
    assert [t["content"] for t in projects.get_chat_history(created["id"], limit=2)] == ["3", "4"]
    assert len(projects.get_chat_history(created["id"])) == 5


def test_list_projects_sorted_most_recently_updated_first():
    first = projects.create_project("First")
    second = projects.create_project("Second")
    # Touch "first" again so it becomes the most recently updated.
    projects.rename_project(first["id"], "First Renamed")
    listed = projects.list_projects()
    assert listed[0]["id"] == first["id"]
    assert listed[1]["id"] == second["id"]


def test_list_projects_skips_unreadable_files(tmp_path):
    projects.create_project("Good One")
    projects_dir = tmp_path / "data" / "projects"
    (projects_dir / "corrupt.json").write_text("{not valid json", encoding="utf-8")

    listed = projects.list_projects()
    assert len(listed) == 1
    assert listed[0]["name"] == "Good One"
