"""Tests for app.web.telemetry_routes (progress, swarm, universe, survivors)
and the pages/partials that render them."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from flask import Flask

from app.orchestration import swarm as swarm_module
from app.orchestration.swarm import SwarmCache, SwarmResult
from app.web import telemetry_routes as tr
from app.web.job_manager import JOB_MANAGER

TEMPLATES = Path(__file__).resolve().parent.parent / "app" / "web" / "templates"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(tr, "SWARM_CACHE", SwarmCache())
    tr.set_evolution_source(None)
    app = Flask(__name__)
    app.register_blueprint(tr.telemetry_bp)
    yield app.test_client()
    tr.set_evolution_source(None)


# ---- progress ------------------------------------------------------------------

def test_progress_route_returns_the_live_snapshot(client):
    job = JOB_MANAGER.create(progress_kind="search")
    JOB_MANAGER.log(job, "Search space ready: 100 candidate(s) (family).")
    data = client.get(f"/api/telemetry/job/{job}/progress").get_json()
    assert data["found"] and not data["done"]
    assert data["progress"]["phase_index"] == 1 and data["progress"]["candidates_total"] == 100
    JOB_MANAGER.finish(job)
    assert client.get(f"/api/telemetry/job/{job}/progress").get_json()["done"] is True


def test_progress_route_unknown_job_and_untracked_job(client):
    resp = client.get("/api/telemetry/job/nope/progress")
    assert resp.status_code == 404 and resp.get_json()["found"] is False
    plain = JOB_MANAGER.create()
    data = client.get(f"/api/telemetry/job/{plain}/progress").get_json()
    assert data["found"] is True and data["progress"] is None


# ---- swarm ---------------------------------------------------------------------

def _wait_ready(client, url, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        d = client.get(url).get_json()
        if d["status"] not in ("running",):
            return d
        time.sleep(0.02)
    raise AssertionError("swarm never finished")


def test_swarm_job_states(client, monkeypatch):
    assert client.get("/api/telemetry/swarm/job/nope").status_code == 404

    running = JOB_MANAGER.create()
    d = client.get(f"/api/telemetry/swarm/job/{running}").get_json()
    assert d["status"] == "pending"                       # chart keeps polling

    no_data = JOB_MANAGER.create(); JOB_MANAGER.finish(no_data)
    d = client.get(f"/api/telemetry/swarm/job/{no_data}").get_json()
    assert d["status"] == "unavailable" and "reason" in d  # chart stops polling


def test_swarm_job_builds_from_the_search_summary_and_caches(client, monkeypatch):
    calls = []
    monkeypatch.setattr(tr, "_load_search_rows", lambda db, run, limit: [{"candidate_id": "c1"}])
    monkeypatch.setattr(tr, "build_swarm", lambda df, risk, rows, progress_cb=None: (calls.append(rows), SwarmResult(requested=1, computed=1))[1])
    summary = SimpleNamespace(db_path="x.db", run_id="r1")
    job = JOB_MANAGER.create(); JOB_MANAGER.finish(job, summary=summary, df=object(), risk=object())
    url = f"/api/telemetry/swarm/job/{job}"
    d = _wait_ready(client, url)
    assert d["status"] == "ready" and d["swarm"]["computed"] == 1
    client.get(url); assert len(calls) == 1               # cached
    client.get(url + "?refresh=1"); _wait_ready(client, url)
    assert len(calls) == 2                                # refresh recomputes


def test_swarm_job_works_for_a_speed_run_result(client, monkeypatch):
    monkeypatch.setattr(tr, "_load_search_rows", lambda *a: [])
    monkeypatch.setattr(tr, "build_swarm", lambda *a, **k: SwarmResult())
    result = SimpleNamespace(search_summary=SimpleNamespace(db_path="x", run_id="r"))
    job = JOB_MANAGER.create(); JOB_MANAGER.finish(job, result=result, df=object(), risk=object())
    assert _wait_ready(client, f"/api/telemetry/swarm/job/{job}")["status"] == "ready"


def test_swarm_error_is_reported_not_raised(client, monkeypatch):
    def boom(*a, **k): raise RuntimeError("engine exploded")
    monkeypatch.setattr(tr, "_load_search_rows", lambda *a: [])
    monkeypatch.setattr(tr, "build_swarm", boom)
    job = JOB_MANAGER.create()
    JOB_MANAGER.finish(job, summary=SimpleNamespace(db_path="x", run_id="r"), df=object(), risk=object())
    d = _wait_ready(client, f"/api/telemetry/swarm/job/{job}")
    assert d["status"] == "error" and "engine exploded" in d["error"]


def test_swarm_evolution_pending_then_built_from_the_elite(client, monkeypatch):
    assert client.get("/api/telemetry/swarm/evolution").get_json()["status"] == "pending"
    tr.set_evolution_source(lambda: None)
    assert client.get("/api/telemetry/swarm/evolution").get_json()["status"] == "pending"

    rec = SimpleNamespace(candidate_id="e1", meta={"family": "trend_breakout"},
                          spec={"source_type": "manual", "config": {"a": 1}})
    runner = SimpleNamespace(leaderboard=[rec], df=object(), risk=object())
    tr.set_evolution_source(lambda: runner)
    seen = []
    monkeypatch.setattr(tr, "build_swarm", lambda df, risk, cands, progress_cb=None: (seen.append(cands), SwarmResult(computed=1))[1])
    d = _wait_ready(client, "/api/telemetry/swarm/evolution")
    assert d["status"] == "ready"
    assert seen[0][0]["candidate_id"] == "e1" and seen[0][0]["family"] == "trend_breakout" and seen[0][0]["config"] == {"a": 1}


# ---- universe / survivors --------------------------------------------------------

def test_universe_route_validates_sources_and_returns_layout(client, monkeypatch):
    seen = {}
    def fake(search_dir, sources):
        seen["sources"] = list(sources)
        return {"dots": [], "clusters": [], "total": 0, "shown": 0, "truncated": False, "sources": list(sources)}
    monkeypatch.setattr(tr.universe_module, "load_universe", fake)
    client.get("/api/telemetry/universe?sources=search,bogus,library")
    assert seen["sources"] == ["search", "library"]
    client.get("/api/telemetry/universe?sources=bogus")
    assert seen["sources"] == ["search", "evolution", "library"]     # nothing valid -> everything


def test_survivors_route(client, monkeypatch):
    monkeypatch.setattr(tr.survivors_module, "load_library_funnel", lambda: {"total": 3, "stages": [], "ready_verdict_count": 1})
    assert client.get("/api/telemetry/survivors").get_json()["total"] == 3


def test_unexpected_errors_are_json_not_html(client, monkeypatch):
    def boom(*a, **k): raise RuntimeError("disk on fire")
    monkeypatch.setattr(tr.survivors_module, "load_library_funnel", boom)
    resp = client.get("/api/telemetry/survivors")
    assert resp.status_code == 500 and resp.is_json and "disk on fire" in resp.get_json()["error"]


# ---- pages ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_client():
    from app.web.server import app
    return app.test_client()


def test_universe_page_renders_with_sidebar_and_chat(real_client):
    html = real_client.get("/universe").get_data(as_text=True)
    assert "data-t58-universe" in html and "data-t58-survivors" in html
    assert "/static/telemetry.js" in html and "t58-pc-launcher" in html
    assert 'href="/universe"' in html                                   # sidebar link


def test_evolution_status_json_carries_progress_when_running(real_client):
    data = real_client.get("/evolution/status.json").get_json()
    assert data["started"] is False and "log" in data                   # unchanged when idle


def test_static_assets_are_served(real_client):
    for path in ("/static/telemetry.js", "/static/telemetry.css", "/static/project_chat.js"):
        assert real_client.get(path).status_code == 200, path


def test_search_and_speed_run_job_pages_carry_banner_swarm_and_chat(real_client):
    for tool, prefix in (("Search Lab", "/search"), ("Speed Run", "/speed-run")):
        job = JOB_MANAGER.create(tool=tool, progress_kind="search")
        html = real_client.get(f"{prefix}/job/{job}").get_data(as_text=True)
        assert f"/api/telemetry/job/{job}/progress" in html, tool
        assert f"/api/telemetry/swarm/job/{job}" in html, tool
        assert "t58-pc-launcher" in html, tool


def test_evolution_page_has_banner_and_swarm(real_client):
    html = real_client.get("/evolution").get_data(as_text=True)
    assert 'data-progress-url="/evolution/status.json"' in html and 'data-progress-key="progress"' in html
    assert "/api/telemetry/swarm/evolution" in html


def test_dashboard_has_the_survivors_strip_and_universe_link(real_client):
    html = real_client.get("/dashboard").get_data(as_text=True)
    assert "data-t58-survivors" in html and 'href="/universe"' in html
    assert 'id="universe-graph"' in html                                 # the existing card is untouched


def test_every_page_that_can_show_a_job_has_the_chat_widget():
    """Job-status pages don't include the sidebar; each must pull the widget in
    itself, and nothing may include it twice (two launchers would collide)."""
    missing, doubled = [], []
    for path in sorted(TEMPLATES.glob("*.html")):
        if path.name.startswith("_") or path.name in ("activate.html", "lock.html"):
            continue
        text = path.read_text(encoding="utf-8")
        has_sidebar = "_sidebar.html" in text
        has_embed = "_project_chat_embed.html" in text
        if not (has_sidebar or has_embed):
            missing.append(path.name)
        if has_sidebar and has_embed:
            doubled.append(path.name)
    assert not missing, f"pages without the Project Chat widget: {missing}"
    assert not doubled, f"pages that would load the widget twice: {doubled}"
