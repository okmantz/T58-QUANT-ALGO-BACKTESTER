"""The Rescue Search page and its job API."""
from app.web.server import app


def test_rescue_page_loads_and_is_linked_from_the_optimize_hub():
    c = app.test_client()
    r = c.get("/rescue")
    assert r.status_code == 200 and b"Rescue a weak strategy" in r.data
    assert b'href="/rescue"' in c.get("/optimize-hub").data or b'href="/rescue"' in c.get("/optimize").data


def test_rescue_start_rejects_missing_data_with_a_plain_message():
    c = app.test_client()
    r = c.post("/api/rescue/start", data={"strategy_pick": "manual::nope.json", "existing_dataset": ""})
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_rescue_status_unknown_job_is_404():
    assert app.test_client().get("/api/rescue/does-not-exist").status_code == 404
