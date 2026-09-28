"""Tests for app.web.project_routes -- run against a bare Flask app with
only this blueprint registered (same isolation convention as
tests/test_web_ai_assistant_routes.py), never touching real app data."""
from __future__ import annotations

import pytest
from flask import Flask

from app.orchestration import projects
from app.web import project_routes as m
from app.web.job_manager import JOB_MANAGER


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(projects, "get_app_base_dir", lambda: tmp_path)
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(m.project_bp)

    @app.route("/_start_job")
    def _start_job():  # stands in for any real tool's JOB_MANAGER.create(...) call
        from flask import jsonify
        return jsonify({"job_id": JOB_MANAGER.create()})

    return app.test_client()


def _create(client, name="Proj"):
    return client.post("/api/projects", json={"name": name}).get_json()["project"]


def test_create_and_list_and_active(client):
    p = _create(client, "Alpha")
    listed = client.get("/api/projects").get_json()
    assert [x["name"] for x in listed["projects"]] == ["Alpha"]
    assert listed["active_project_id"] == p["id"]  # creating activates
    assert client.get("/api/projects/active").get_json()["project"]["id"] == p["id"]


def test_activate_switch_and_deactivate(client):
    a, b = _create(client, "A"), _create(client, "B")
    assert client.get("/api/projects").get_json()["active_project_id"] == b["id"]
    assert client.post(f"/api/projects/{a['id']}/activate").get_json() == {"ok": True}
    assert client.get("/api/projects").get_json()["active_project_id"] == a["id"]
    client.post("/api/projects/deactivate")
    assert client.get("/api/projects/active").get_json()["project"] is None


def test_unknown_project_is_404_json(client):
    for resp in (client.get("/api/projects/nope"), client.post("/api/projects/nope/activate"),
                 client.get("/api/projects/nope/activity"),
                 client.post("/api/projects/nope/chat", json={"message": "hi"})):
        assert resp.status_code == 404 and resp.is_json


def test_rename_and_delete_clears_active(client):
    p = _create(client, "Old")
    assert client.patch(f"/api/projects/{p['id']}", json={"name": "New"}).get_json()["project"]["name"] == "New"
    assert client.delete(f"/api/projects/{p['id']}").get_json() == {"ok": True}
    assert client.get("/api/projects/active").get_json()["project"] is None


def test_chat_persists_both_turns_and_sends_prior_history(client, monkeypatch):
    seen = {}

    def fake_chat(self, project_name, user_message, history=None, activity_lines=None):
        seen.update(name=project_name, history=list(history or []), activity=activity_lines)
        return f"echo: {user_message}", None

    monkeypatch.setattr(m.ProjectChatClient, "chat", fake_chat)
    p = _create(client, "Chatty")

    first = client.post(f"/api/projects/{p['id']}/chat", json={"message": "one"}).get_json()
    assert first["reply"] == "echo: one" and first["error"] is None
    assert seen["history"] == []  # this turn's own message isn't duplicated into history

    client.post(f"/api/projects/{p['id']}/chat", json={"message": "two"})
    assert [t["content"] for t in seen["history"]] == ["one", "echo: one"]

    saved = projects.get_project(p["id"])["chat_history"]
    assert [t["content"] for t in saved] == ["one", "echo: one", "two", "echo: two"]


def test_chat_error_keeps_user_message_but_saves_no_assistant_turn(client, monkeypatch):
    monkeypatch.setattr(m.ProjectChatClient, "chat", lambda self, *a, **k: ("", "Ollama down"))
    p = _create(client)
    data = client.post(f"/api/projects/{p['id']}/chat", json={"message": "hi"}).get_json()
    assert data["error"] == "Ollama down"
    assert [t["role"] for t in projects.get_project(p["id"])["chat_history"]] == ["user"]


def test_chat_rejects_empty_message(client):
    p = _create(client)
    resp = client.post(f"/api/projects/{p['id']}/chat", json={"message": "   "})
    assert resp.status_code == 400


def test_activity_feed_shows_only_this_projects_jobs_and_is_json_safe(client):
    a, b = _create(client, "A"), _create(client, "B")
    mine = JOB_MANAGER.create(project_id=a["id"], instrument="ES")
    JOB_MANAGER.create(project_id=b["id"])
    JOB_MANAGER.update(mine, result=object())  # non-JSON-safe result must not break the feed
    JOB_MANAGER.log(mine, "step 1")

    jobs = client.get(f"/api/projects/{a['id']}/activity").get_json()["jobs"]
    assert [j["job_id"] for j in jobs] == [mine]
    assert jobs[0]["instrument"] == "ES" and jobs[0]["log_tail"] == ["step 1"]
    assert jobs[0]["has_result"] is True and "result" not in jobs[0]


def test_jobs_created_in_a_session_are_auto_tagged_with_its_active_project(client):
    p = _create(client, "Tagged")  # activates it in this client's session

    job_id = client.get("/_start_job").get_json()["job_id"]
    assert JOB_MANAGER.get(job_id)["project_id"] == p["id"]

    client.post("/api/projects/deactivate")
    untagged = client.get("/_start_job").get_json()["job_id"]
    assert JOB_MANAGER.get(untagged)["project_id"] is None
