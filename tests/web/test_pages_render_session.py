from __future__ import annotations

import pytest

from app.web.server import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("T58_JOB_HISTORY_PATH", str(tmp_path / "jh.json"))
    app.config["TESTING"] = True
    return app.test_client()


@pytest.mark.parametrize("url", ["/library", "/runs", "/discovery-engine", "/api/running-jobs.json"])
def test_pages_render(client, url):
    assert client.get(url).status_code == 200


def test_library_has_retire_and_next_step(client):
    t = client.get("/library").get_data(as_text=True)
    assert "Retire to graveyard" in t and "d-nextcard" in t


def test_discovery_page_promises_counts(client):
    t = client.get("/discovery-engine").get_data(as_text=True)
    assert "5,000+" in t and "10,000+" in t and "15,000+" in t


def test_retire_route_unknown_strategy(client):
    r = client.post("/strategies/retire", data={"strategy_type": "python", "filename": "nope.py"},
                    headers={"X-Requested-With": "fetch"})
    assert r.status_code in (400, 404)
