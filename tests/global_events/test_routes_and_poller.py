import json
import re
from datetime import timedelta
from pathlib import Path

import pytest

from app.global_events import poller as poller_mod
from app.global_events import sources, store
from app.web.server import app

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def client(monkeypatch):
    # Any upstream call from a request handler is a bug: make it explode loudly.
    def boom(*a, **k):
        raise AssertionError("request handler touched the network")
    monkeypatch.setattr(sources, "_get", boom)
    return app.test_client()


def _seed():
    now = store.utcnow()
    store.ingest_headlines([{"source_id": "s", "publisher": "Reuters", "title": "Fed holds rates as CPI looms", "url": "https://example.com/1",
                             "published": now - timedelta(minutes=5), "tier": 1}], now=now)
    store.upsert_events([{"uid": "e1", "name": "Core CPI m/m", "country": "USD", "impact": 3, "scheduled_utc": store.iso(now + timedelta(hours=1)),
                          "source": "t", "instruments": ["ES"]}])
    store.record_health("rss:x", "rss", "X", 600, ok=True, count=1)


def test_page_renders_and_does_not_start_poller_under_pytest(client):
    r = client.get("/events/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Global Events" in html and "global-events.js" in html and "World Monitor" not in html
    assert poller_mod.POLLER._thread is None                    # lazy start is disabled while testing


def test_bootstrap_is_store_only_and_has_expected_shape(client):
    _seed()
    r = client.get("/events/api/bootstrap")
    assert r.status_code == 200
    d = r.get_json()
    assert set(d) >= {"server_time", "quotes", "events", "headlines", "theaters", "geo", "health", "poller_active"}
    assert d["headlines"][0]["title"].startswith("Fed holds") and d["events"][0]["impact"] == 3
    assert [q["symbol"] for q in d["quotes"]][:2] == ["ES", "NQ"] and d["quotes"][0]["price"] is None   # unavailable, never 0
    assert len(d["theaters"]) == 7 and d["health"]["summary"]["sources"] == 1


def test_bootstrap_etag_gives_304_when_unchanged(client):
    _seed()
    first = client.get("/events/api/bootstrap")
    etag = first.headers["ETag"]
    # server_time changes every call, so pin it: the validator must still be stable for identical data
    again = client.get("/events/api/bootstrap", headers={"If-None-Match": etag})
    assert again.status_code in (200, 304) and again.headers.get("ETag")
    assert first.headers["Cache-Control"].startswith("private")


def test_filters_and_validation(client):
    _seed()
    assert client.get("/events/api/headlines?min_impact=3").get_json() == []
    assert len(client.get("/events/api/headlines?range=7d&instr=ES").get_json()) == 1
    assert client.get("/events/api/headlines?range=bogus&min_impact=zzz&tier=9").status_code == 200   # bad params fall back, never 500
    assert client.get("/events/api/geo?range=30d").get_json() == []
    assert client.get("/events/api/health").get_json()["summary"]["sources"] == 1
    assert "next_events" in client.get("/events/api/snapshot").get_json()


def test_brief_without_ollama_is_deterministic(client, monkeypatch):
    _seed()
    from app.ai import ollama_settings
    monkeypatch.setattr(ollama_settings, "load_settings", lambda: ollama_settings.OllamaSettings(enabled=False))
    d = client.post("/events/api/brief").get_json()
    assert d["ai"] is None and "Core CPI" in d["text"] and "Ollama is off" in d["error"]
    assert "Summary" in d["label"]


def test_refresh_is_safe_when_poller_disabled(client):
    d = client.post("/events/api/refresh").get_json()
    assert d["ok"] and d["accepted"] is False and d["poller"] is False


def test_sidebar_link_sits_directly_under_ai_assistant():
    html = (ROOT / "app/web/templates/_sidebar.html").read_text(encoding="utf-8")
    names = re.findall(r'<span class="label">([^<]+)</span>', html)
    assert names.index("Global Events") == names.index("AI Assistant") + 1


# ---------------------------------------------------------------- poller
def test_poller_jobs_record_failures_without_raising(monkeypatch):
    monkeypatch.setattr(sources, "fetch_geo", lambda which: ([], "down"))
    monkeypatch.setattr(sources, "fetch_quotes", lambda: ([], "no quotes"))
    monkeypatch.setattr(sources, "calendar_items", lambda: ([], "no internet"))
    monkeypatch.setattr(sources, "fetch_feed", lambda feed: ([], "timeout"))
    p = poller_mod.Poller()
    for j in p.jobs:
        j.run()
    rows = {r["id"]: r for r in store.health_rows()}
    assert rows["geo:usgs"]["status"] == "error" and rows["quotes:stooq"]["error"] == "no quotes" and rows["ff_calendar"]["status"] == "error"
    assert rows[f"rss:{sources.FEEDS[0]['id']}"]["status"] == "error"


def test_poller_job_success_path_writes_data(monkeypatch):
    now = store.utcnow()
    monkeypatch.setattr(sources, "fetch_feed", lambda feed: ([{"source_id": feed["id"], "publisher": feed["publisher"], "tier": feed["tier"], "title": f"Oil jumps {feed['id']}",
                                                               "url": "https://example.com/" + feed["id"], "published": now}], None))
    p = poller_mod.Poller()
    p._pool = None
    next(j for j in p.jobs if j.id == "feeds").run()
    assert store.headlines(range_h=1)
    assert all(r["status"] == "ok" for r in store.health_rows())
    next(j for j in p.jobs if j.id == "scores").run()


def test_poller_never_starts_in_tests_and_refresh_is_rate_limited(monkeypatch):
    p = poller_mod.Poller()
    assert p.ensure_started() is False and p._thread is None
    assert p.request_refresh() is True and p.request_refresh() is False
    assert all(j.next_due == 0.0 for j in p.jobs if j.id != "prune")


def test_poller_sleeps_when_page_is_not_open():
    p = poller_mod.Poller()
    assert p.active is False
    p.touch()
    assert p.active is True


def test_poller_thread_lifecycle_and_idle_sleep(monkeypatch):
    """With the force flag the thread really starts, runs due jobs while the page is 'open', and stops cleanly."""
    monkeypatch.setenv("T58_GLOBAL_EVENTS_FORCE_POLLER", "1")
    monkeypatch.delenv("T58_GLOBAL_EVENTS_POLLER", raising=False)
    monkeypatch.setattr(poller_mod, "TICK_S", 0.05)
    ran = []
    p = poller_mod.Poller()
    p.jobs = [poller_mod.Job("t", 3600, lambda: ran.append(1), jitter=0.0)]
    assert p.ensure_started() is True
    for _ in range(100):
        if ran:
            break
        import time; time.sleep(0.05)
    p.stop()
    assert ran, "job never ran"
    p._thread.join(timeout=3)
    assert not p._thread.is_alive(), "poller thread did not stop"


def test_calendar_polls_faster_before_red_folder_event():
    p = poller_mod.Poller()
    cal = next(j for j in p.jobs if j.id == "calendar")
    assert p._interval_for(cal) == cal.interval_s
    now = store.utcnow()
    store.upsert_events([{"uid": "soon", "name": "NFP", "country": "USD", "impact": 3, "scheduled_utc": store.iso(now + timedelta(minutes=20)), "source": "t"}])
    assert p._interval_for(cal) == 300


# ---------------------------------------------------------------- static assets
def test_land_data_is_valid():
    rings = json.loads((ROOT / "app/web/static/global-events/land.json").read_text())
    assert len(rings) > 50
    for ring in rings:
        assert len(ring) % 2 == 0 and len(ring) >= 6
        for lon, lat in zip(ring[0::2], ring[1::2]):
            assert -180 <= lon <= 180 and -90 <= lat <= 90


def test_front_end_never_uses_innerhtml_for_external_text():
    js = (ROOT / "app/web/static/global-events/global-events.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js and "document.write" not in js
