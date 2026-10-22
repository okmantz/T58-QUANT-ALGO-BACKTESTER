import pytest

from app.global_events import x_feed


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.setattr(x_feed, "_path", lambda: tmp_path / "x.json")
    x_feed._invalidate(); x_feed._USER_ID_CACHE.clear()
    yield


def test_normalize():
    assert x_feed.normalize_username("@Fed") == "Fed"
    assert x_feed.normalize_username("https://x.com/Fed/status/1") == "Fed"
    assert x_feed.normalize_username("twitter.com/Fed") == "Fed"
    assert x_feed.normalize_username("bad name!") is None
    assert x_feed.normalize_username("a" * 16) is None


def test_add_remove_dedupe():
    assert x_feed.add_account("@A")[0]
    ok, msg, accts = x_feed.add_account("a")
    assert not ok and accts == ["A"]
    assert x_feed.remove_account("@a") == []


def test_feed_requires_token_and_accounts(monkeypatch):
    monkeypatch.setattr(x_feed, "bearer_token", lambda: "")
    x_feed.add_account("A")
    assert x_feed.feed()["configured"] is False


class R:
    def __init__(self, code, js): self.status_code, self._j = code, js
    def json(self): return self._j


def test_feed_collects_posts_and_errors(monkeypatch):
    monkeypatch.setattr(x_feed, "bearer_token", lambda: "tok")
    x_feed.add_account("good"); x_feed.add_account("limited")

    def fake(path, token, params=None):
        if path.endswith("/good"): return R(200, {"data": {"id": "1", "name": "Good"}})
        if path.endswith("/limited"): return R(429, {})
        return R(200, {"data": [{"id": "9", "text": "hi", "created_at": "2026-10-22T10:00:00Z"}]})

    monkeypatch.setattr(x_feed, "_get", fake)
    d = x_feed.feed(force=True)
    assert [p["id"] for p in d["posts"]] == ["9"] and d["posts"][0]["url"].endswith("/good/status/9")
    assert "429" in d["errors"]["limited"]


def test_routes(monkeypatch):
    from flask import Flask
    from app.global_events.routes import global_events_bp
    app = Flask(__name__); app.register_blueprint(global_events_bp)
    c = app.test_client()
    assert c.post("/events/api/x/accounts", json={"username": "@Fed"}).status_code == 200
    assert c.post("/events/api/x/accounts", json={"username": "no way"}).status_code == 400
    assert c.get("/events/api/x").get_json()["accounts"] == ["Fed"]
