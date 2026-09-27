"""
Tests for app.web.ai_assistant_routes -- specifically the _json_safe
error-handling wrapper (BUGFIX Sep 2026: every route in this blueprint
used to have zero exception handling, so an unhandled exception fell
through to Flask's default HTML error page instead of JSON, breaking
every button's `(await fetch(...)).json()` call in ai_assistant.html
with "Unexpected token '<' ... is not valid JSON") and the new
"Analyze a Symbol" chart-picker route (api_analyze_symbol).

No real MT5/network connection exists in this sandbox, so these tests
either (a) confirm every route still returns valid JSON with graceful
"no data" messaging when the data feed is unavailable (the pre-existing,
correct behavior for a disconnected feed), or (b) monkeypatch an inner
function to explicitly raise, proving the wrapper turns that into JSON
instead of an HTML 500 page.
"""
from flask import Flask

from app.web import ai_assistant_routes as m


def _make_client():
    app = Flask(__name__)
    app.register_blueprint(m.ai_assistant_bp)
    return app.test_client()


def test_outlook_route_survives_no_data_feed():
    client = _make_client()
    resp = client.get("/assistant/api/outlook")
    assert resp.status_code == 200
    assert resp.is_json
    data = resp.get_json()
    assert "text" in data


def test_trade_of_the_day_route_survives_no_data_feed():
    client = _make_client()
    resp = client.get("/assistant/api/trade-of-the-day")
    assert resp.status_code == 200
    assert resp.is_json


def test_json_safe_wrapper_turns_an_unhandled_exception_into_json():
    """The core regression test: force an exception inside the outlook
    pipeline and confirm the route returns a normal JSON body (with the
    real error message) instead of Flask's HTML error page -- exactly
    the bug that produced "Unexpected token '<'" in the browser."""
    client = _make_client()
    m._cache.clear()  # force a fresh compute instead of a cached result from an earlier test

    def boom():
        raise ValueError("simulated crash for test")

    original = m._compute_rankings
    m._compute_rankings = boom
    try:
        resp = client.get("/assistant/api/outlook")
    finally:
        m._compute_rankings = original
        m._cache.clear()

    assert resp.status_code == 200
    assert resp.is_json  # the whole point: never an HTML error page
    data = resp.get_json()
    assert "simulated crash for test" in data["error"]


def test_json_safe_wrapper_covers_trade_of_the_day_too():
    client = _make_client()
    m._cache.clear()

    def boom():
        raise RuntimeError("simulated trade-of-the-day crash")

    original = m._compute_news
    m._compute_news = boom
    try:
        resp = client.get("/assistant/api/trade-of-the-day")
    finally:
        m._compute_news = original
        m._cache.clear()

    assert resp.status_code == 200
    assert resp.is_json
    assert "simulated trade-of-the-day crash" in resp.get_json()["error"]


def test_analyze_symbol_requires_a_symbol():
    client = _make_client()
    resp = client.get("/assistant/api/analyze-symbol")
    assert resp.status_code == 400
    assert resp.is_json
    assert "symbol" in resp.get_json()["error"].lower()


def test_analyze_symbol_reports_no_data_gracefully_not_as_a_crash():
    """With no MT5/Alpaca connection in this sandbox, an arbitrary symbol
    must come back as a clear, actionable "no data" JSON message -- never
    an unhandled exception."""
    client = _make_client()
    resp = client.get("/assistant/api/analyze-symbol?symbol=ES1!")
    assert resp.status_code == 200
    assert resp.is_json
    data = resp.get_json()
    assert data["error"] is not None
    assert "ES1!" in data["error"]


def test_analyze_symbol_survives_an_unhandled_exception():
    client = _make_client()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated bar-fetch crash")

    original = m._bar_fetcher
    m._bar_fetcher = boom
    try:
        resp = client.get("/assistant/api/analyze-symbol?symbol=ES1!")
    finally:
        m._bar_fetcher = original

    assert resp.status_code == 200
    assert resp.is_json
    assert "simulated bar-fetch crash" in resp.get_json()["error"]
