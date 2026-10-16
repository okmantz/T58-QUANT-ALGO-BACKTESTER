"""Tests for the Strategy Library web page (/library) and its supporting
routes: /strategies/save-code and the return_to=library redirect target."""
from __future__ import annotations

from app.strategy.library import (
    delete_saved_strategy, load_strategy_text, save_strategy_metadata,
    save_strategy_text, set_strategy_tags,
)
from app.web.server import app


def _client():
    app.config["TESTING"] = True
    return app.test_client()


def _make_test_strategy(name="lib_route_test.py", strategy_type="python"):
    save_strategy_text("def generate_signals(df):\n    return df['close'] * 0", name, strategy_type, overwrite=True)
    save_strategy_metadata(strategy_type, name, {"description": "A test strategy", "market": "EURUSD"})
    set_strategy_tags(strategy_type, name, ["test", "demo"])


def test_library_page_renders_and_lists_a_saved_strategy():
    _make_test_strategy()
    try:
        r = _client().get("/library")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert "lib_route_test.py" in body
        assert "A test strategy" in body
    finally:
        delete_saved_strategy("python", "lib_route_test.py")


def test_save_code_route_persists_an_edit():
    _make_test_strategy()
    try:
        client = _client()
        r = client.post("/strategies/save-code", data={
            "strategy_type": "python", "filename": "lib_route_test.py",
            "code": "def generate_signals(df):\n    return df['close'] * 1  # edited",
        })
        assert r.status_code == 200
        assert r.get_json()["ok"] is True
        assert "edited" in load_strategy_text("python", "lib_route_test.py")
    finally:
        delete_saved_strategy("python", "lib_route_test.py")


def test_save_code_route_rejects_unknown_strategy_type():
    r = _client().post("/strategies/save-code", data={
        "strategy_type": "not_a_real_type", "filename": "x.py", "code": "x = 1",
    })
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_metadata_and_status_updates_redirect_back_to_library_when_asked():
    _make_test_strategy()
    try:
        client = _client()
        r1 = client.post("/strategies/metadata", data={
            "strategy_type": "python", "filename": "lib_route_test.py",
            "description": "Updated", "market": "GBPUSD", "tags": "a,b", "return_to": "library",
        })
        assert r1.status_code == 302
        assert r1.headers["Location"].startswith("/library")

        r2 = client.post("/strategies/status", data={
            "strategy_type": "python", "filename": "lib_route_test.py",
            "status": "validated", "return_to": "library",
        })
        assert r2.status_code == 302
        assert r2.headers["Location"].startswith("/library")
    finally:
        delete_saved_strategy("python", "lib_route_test.py")


def test_metadata_update_without_return_to_still_redirects_to_index_unchanged():
    """The new return_to=library branch must not change the pre-existing
    default behavior for callers that don't pass it."""
    _make_test_strategy()
    try:
        client = _client()
        r = client.post("/strategies/metadata", data={
            "strategy_type": "python", "filename": "lib_route_test.py",
            "description": "x", "market": "", "tags": "",
        })
        assert r.status_code == 302
        assert r.headers["Location"].startswith("/") and "/library" not in r.headers["Location"]
    finally:
        delete_saved_strategy("python", "lib_route_test.py")


def test_full_pipeline_form_accepts_load_strategy_query_params():
    """Just confirms the page still renders fine with the Strategy
    Library's pre-load query params present -- the actual pre-selection
    happens client-side in JS, which this test can't execute, but the
    page must not error out on their presence."""
    r = _client().get("/full-pipeline?load_strategy_type=python&load_strategy_name=whatever.py")
    assert r.status_code == 200


def test_quick_optimize_form_accepts_load_strategy_query_params():
    r = _client().get("/quick-optimize?load_strategy_type=manual&load_strategy_name=whatever.json")
    assert r.status_code == 200
