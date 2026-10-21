import sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.global_events import store

NOW = datetime(2026, 10, 21, 14, 0, tzinfo=timezone.utc)


def _h(i, title, publisher, tier, minutes_ago, url=None):
    return {"source_id": f"s{i}", "publisher": publisher, "title": title, "url": url or f"https://example.com/{i}",
            "published": NOW - timedelta(minutes=minutes_ago), "tier": tier}


def test_ingest_dedupes_clusters_and_lifts_capped_impact():
    n = store.ingest_headlines([
        _h(1, "Iran tanker strike raises Hormuz risk, crude jumps", "Aggregator Blog", 4, 30),
    ], now=NOW)
    assert n == 1
    low = store.headlines(range_h=24, now=NOW)
    assert low[0]["impact"] <= 1                                  # tier-4, single source: capped
    n = store.ingest_headlines([
        _h(2, "Iran tanker strike raises Hormuz risk, oil climbs", "Reuters", 1, 25),
        _h(2, "Iran tanker strike raises Hormuz risk, oil climbs", "Reuters", 1, 25),   # exact duplicate -> ignored
    ], now=NOW)
    assert n == 1
    rows = store.headlines(range_h=24, now=NOW)
    assert len(rows) == 1                                         # one story, not two rows
    assert rows[0]["outlets"] == 2 and rows[0]["tier"] == 1       # best source first
    assert rows[0]["impact"] >= 1


def test_rejects_rows_without_url_or_title_and_future_dates():
    assert store.ingest_headlines([_h(1, "", "X", 2, 1), {**_h(2, "t", "X", 2, 1), "url": ""}], now=NOW) == 0
    store.ingest_headlines([{**_h(3, "Fed holds rates", "Reuters", 1, 0), "published": NOW + timedelta(days=3)}], now=NOW)
    assert store.headlines(range_h=1, now=NOW)                    # clamped to "now", not hidden in the future


def test_headline_filters():
    store.ingest_headlines([_h(1, "OPEC signals output cut", "Reuters", 1, 10), _h(2, "Bitcoin ETF flows turn positive", "CoinDesk", 3, 20)], now=NOW)
    assert len(store.headlines(range_h=24, instrument="CL", now=NOW)) == 1
    assert len(store.headlines(range_h=24, min_impact=2, now=NOW)) == 1
    assert store.headlines(range_h=24, tier_max=1, now=NOW)[0]["publisher"] == "Reuters"


def test_events_upsert_updates_actual_without_duplicating():
    uid = store.event_uid("ff", "Core CPI m/m", "USD", "2026-10-22T12:30:00Z")
    base = {"uid": uid, "name": "Core CPI m/m", "country": "USD", "impact": 3, "scheduled_utc": "2026-10-22T12:30:00Z", "source": "ff", "instruments": ["ES"]}
    store.upsert_events([base])
    store.upsert_events([{**base, "actual": "0.4%"}])
    ev = store.events(NOW, NOW + timedelta(days=2))
    assert len(ev) == 1 and ev[0]["actual"] == "0.4%" and ev[0]["impact"] == 3


def test_ff_release_maps_to_correct_utc_across_both_dst_transitions():
    """08:30 US Eastern must land on 13:30Z in winter and 12:30Z in summer, either side of both 2026 transitions."""
    from app.ai import news_forexfactory as ff

    ny = ZoneInfo("America/New_York")
    cases = [(datetime(2026, 3, 6, 8, 30, tzinfo=ny), "13:30"), (datetime(2026, 3, 9, 8, 30, tzinfo=ny), "12:30"),
             (datetime(2026, 10, 30, 8, 30, tzinfo=ny), "12:30"), (datetime(2026, 11, 2, 8, 30, tzinfo=ny), "13:30")]
    for local, expect in cases:
        assert store.iso(local)[11:16] == expect
        ev = ff._parse_event({"title": "CPI", "country": "USD", "impact": "High", "date": local.isoformat()})
        assert store.iso(ev.when)[11:16] == expect


def test_geo_maps_theater_and_market_relevance():
    store.upsert_geo([
        {"uid": "q1", "source": "usgs", "kind": "quake", "lat": 24.0, "lon": 121.0, "severity": 2, "magnitude": 6.1, "title": "M6.1 Taiwan", "occurred_utc": store.iso(NOW)},
        {"uid": "q2", "source": "usgs", "kind": "quake", "lat": -40.0, "lon": 0.0, "severity": 0, "magnitude": 4.6, "title": "M4.6 mid-ocean", "occurred_utc": store.iso(NOW)},
        {"uid": "q3", "source": "usgs", "kind": "quake", "lat": 29.5, "lon": -95.0, "severity": 1, "magnitude": 5.0, "title": "M5 Gulf Coast", "occurred_utc": store.iso(NOW)},
    ])
    by = {g["title"]: g for g in store.geo_events(range_h=24, now=NOW)}
    assert by["M6.1 Taiwan"]["theater"] == "taiwan" and by["M6.1 Taiwan"]["mkt"]
    assert not by["M4.6 mid-ocean"]["mkt"] and by["M4.6 mid-ocean"]["theater"] is None
    assert by["M5 Gulf Coast"]["theater"] == "hazard"            # near a refining hub with no theater box


def test_quotes_na_is_never_zero():
    store.insert_quotes([{"symbol": "ES", "ts_utc": "2026-10-21T14:00:00Z", "price": 5900.0, "chg": 5.0, "chg_pct": 0.1, "source": "t"}])
    q = {x["symbol"]: x for x in store.quotes_latest(["ES", "NQ"])}
    assert q["ES"]["price"] == 5900.0 and q["NQ"]["price"] is None and q["NQ"]["chg_pct"] is None and q["NQ"]["spark"] == []


def test_health_status_progression():
    store.record_health("a", "rss", "A", 600, ok=True, count=5)
    store.record_health("b", "rss", "B", 600, ok=False, error="boom")
    store.record_health("c", "rss", "C", 600, ok=True, count=0)
    rows = {r["id"]: r for r in store.health_rows()}
    assert rows["a"]["status"] == "ok" and rows["b"]["status"] == "error" and rows["c"]["status"] == "empty"
    assert rows["b"]["error"] == "boom"
    later = store.utcnow() + timedelta(seconds=700)
    assert {r["id"]: r["status"] for r in store.health_rows(later)}["a"] == "stale"
    assert {r["id"]: r["status"] for r in store.health_rows(store.utcnow() + timedelta(seconds=2000))}["a"] == "very_stale"
    # a later failure keeps the last good record count / time but flips the status
    store.record_health("a", "rss", "A", 600, ok=False, error="down")
    a = {r["id"]: r for r in store.health_rows()}["a"]
    assert a["status"] == "error" and a["count"] == 5 and a["last_ok"]
    assert store.health_summary()["overall"] in ("ok", "degraded")


def test_theater_scores_calibrate_then_score():
    store.ingest_headlines([_h(i, f"Iran tanker strike number {i} near Hormuz", f"Pub{i}", 1, 10 + i) for i in range(4)], now=NOW)
    res = {r["id"]: r for r in store.compute_theater_scores(NOW)}
    assert res["mideast"]["components"]["calibrating"] is True
    assert res["mideast"]["score"] >= res["korea"]["score"]
    scores = {t["id"]: t for t in store.theater_scores()}
    assert scores["mideast"]["score"] is not None and scores["mideast"]["top"]["title"]


def test_prune_and_snapshot():
    old = NOW - timedelta(days=40)
    store.ingest_headlines([{**_h(1, "Old story about oil", "Reuters", 1, 0), "published": old}], now=old)
    store.prune(NOW)
    with store.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM ge_headlines").fetchone()[0] == 0
    snap = store.context_snapshot()
    assert set(snap) >= {"generated_utc", "next_events", "headlines", "theaters", "freshness", "health"}


def test_schema_is_wal_and_idempotent():
    store.connect().close()
    store._initialised.clear()
    c = store.connect()
    assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    c.close()
    assert isinstance(sqlite3.sqlite_version, str)
