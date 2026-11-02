"""A damaged saved experiment (or any lab failure) must never turn a Research Lab page into a blank server error."""
import json

import pytest

from app.research_labs import common as C
from app.web.server import app as flask_app

SLUGS = ["volume-profile", "order-flow", "order-book", "lead-lag", "stat-arb", "event-study", "seasonality", "hypothesis"]


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


@pytest.fixture
def junk_store(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_base_dir", lambda: tmp_path)
    for lab in ("order_flow", "order_book", "lead_lag", "seasonality"):
        d = tmp_path / "data" / "research_labs" / lab
        d.mkdir(parents=True)
        (d / "a.json").write_text("[1, 2, 3]")                                   # not an object
        (d / "b.json").write_text('{"results": [], "created_utc": "yesterday"}')  # wrong types
        (d / "c.json").write_text('{"id": 5, "title": null, "results": {"headline": {"x": 1}}}')
        (d / "d.json").write_text("{truncated")                                  # invalid JSON
        (d / "e.json").write_bytes(b"\xff\xfe\x00bad")                           # bad encoding
        (d / "ok.json").write_text(json.dumps({"id": "ok1", "lab": lab, "title": "T", "created_utc": 5, "results": {"headline": "fine"}}))
    return tmp_path


def test_list_experiments_survives_junk(junk_store):
    rows = C.list_experiments(lab="order_flow")
    assert any(r["headline"] == "fine" for r in rows)
    assert all(isinstance(r["created_utc"], float) for r in rows)
    assert C.list_experiments(limit=50)


@pytest.mark.parametrize("slug", SLUGS)
def test_every_lab_page_opens_even_with_damaged_history(client, junk_store, slug):
    r = client.get(f"/research-labs/{slug}")
    assert r.status_code == 200, r.get_data(as_text=True)[:300]


def test_hub_opens_with_damaged_history(client, junk_store):
    assert client.get("/research-labs/").status_code == 200


def test_unexpected_error_shows_readable_page(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(C, "list_experiments", boom)
    r = client.get("/research-labs/order-flow")
    body = r.get_data(as_text=True)
    assert r.status_code == 500 and "synthetic failure" in body and "Research Labs" in body
    r = client.post("/research-labs/order-flow/run", data={})
    assert r.status_code in (400, 500) and r.get_json()["ok"] is False
