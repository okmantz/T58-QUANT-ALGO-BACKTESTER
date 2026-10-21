"""Contract tests: parsers must survive realistic payloads, missing fields and garbage."""
import json
from datetime import datetime, timezone

from app.global_events import sources

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>
<item><title>Fed holds rates steady</title><link>https://example.com/a</link><pubDate>Wed, 21 Oct 2026 14:00:00 GMT</pubDate></item>
<item><title>No link item</title></item>
<item><title>Bad date</title><link>https://example.com/b</link><pubDate>not a date</pubDate></item>
<item><title>javascript link</title><link>javascript:alert(1)</link></item>
</channel></rss>"""

GNEWS = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Oil jumps on supply fears - Reuters</title><link>https://news.google.com/x</link><pubDate>Wed, 21 Oct 2026 13:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Atom headline</title>
<link href="https://example.com/atom"/><updated>2026-10-21T12:00:00Z</updated></entry></feed>"""


def test_parse_rss_skips_bad_items_and_unsafe_links():
    items = sources.parse_feed(RSS, "Example")
    titles = [i["title"] for i in items]
    assert titles == ["Fed holds rates steady", "Bad date"]
    assert items[0]["published"] == datetime(2026, 10, 21, 14, 0, tzinfo=timezone.utc)
    assert items[1]["published"] is None and items[0]["publisher"] == "Example"


def test_parse_google_news_uses_real_publisher_and_strips_suffix():
    it = sources.parse_feed(GNEWS, "Google News")[0]
    assert it["publisher"] == "Reuters" and it["title"] == "Oil jumps on supply fears"


def test_parse_atom():
    it = sources.parse_feed(ATOM, "Example")[0]
    assert it["url"] == "https://example.com/atom" and it["published"].hour == 12


def test_parse_garbage_returns_empty():
    assert sources.parse_feed(b"<html><body>blocked</body>", "X") == []
    assert sources.parse_feed(b"\x00\x01 not xml", "X") == []


def test_google_news_url_is_encoded():
    assert "q=site%3Areuters.com+markets" in sources.google_news_url("site:reuters.com markets")


def test_parse_usgs():
    data = {"features": [
        {"id": "us1", "properties": {"mag": 6.4, "time": 1792587600000, "place": "x", "title": "M 6.4 - Taiwan", "url": "https://earthquake.usgs.gov/e/us1"},
         "geometry": {"coordinates": [121.1, 24.2, 10]}},
        {"id": "bad", "properties": {"mag": None, "time": 1}, "geometry": {"coordinates": [0, 0]}},
        {"id": "nogeo", "properties": {"mag": 5, "time": 1}, "geometry": {}},
    ]}
    rows = sources.parse_usgs(data)
    assert len(rows) == 1 and rows[0]["severity"] == 2 and rows[0]["lat"] == 24.2 and rows[0]["uid"] == "usgs:us1"
    assert rows[0]["occurred_utc"].endswith("Z")


def test_parse_eonet_skips_quakes_and_handles_polygons():
    data = {"events": [
        {"id": "EONET_1", "title": "Wildfire X", "link": "https://eonet.gsfc.nasa.gov/1", "categories": [{"id": "wildfires"}],
         "geometry": [{"date": "2026-10-20T10:00:00Z", "type": "Point", "coordinates": [-120.5, 38.2]}]},
        {"id": "EONET_2", "title": "Quake", "categories": [{"id": "earthquakes"}], "geometry": [{"date": "2026-10-20T10:00:00Z", "type": "Point", "coordinates": [1, 1]}]},
        {"id": "EONET_3", "title": "Storm Y", "categories": [{"id": "severeStorms"}],
         "geometry": [{"date": "2026-10-20T10:00:00Z", "type": "Polygon", "coordinates": [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]]}]},
        {"id": "EONET_4", "title": "No geometry", "categories": [{"id": "floods"}], "geometry": []},
    ]}
    rows = {r["uid"]: r for r in sources.parse_eonet(data)}
    assert set(rows) == {"eonet:EONET_1", "eonet:EONET_3"}
    assert rows["eonet:EONET_1"]["kind"] == "fire" and rows["eonet:EONET_3"]["kind"] == "storm"
    assert 10 < rows["eonet:EONET_3"]["lon"] < 20


def test_parse_gdacs_namespaces():
    xml = b"""<?xml version="1.0"?><rss xmlns:gdacs="http://www.gdacs.org" xmlns:geo="http://www.w3.org/2003/01/geo/wgs84_pos#"><channel>
    <item><title>Red alert for tropical cyclone X</title><link>https://www.gdacs.org/r</link><pubDate>Wed, 21 Oct 2026 09:00:00 GMT</pubDate>
    <gdacs:eventtype>TC</gdacs:eventtype><gdacs:eventid>1001</gdacs:eventid><gdacs:alertlevel>Red</gdacs:alertlevel><geo:Point><geo:lat>15.5</geo:lat><geo:long>125.0</geo:long></geo:Point></item>
    <item><title>no coords</title></item></channel></rss>"""
    rows = sources.parse_gdacs(xml)                      # geo:lat / geo:long are nested inside geo:Point
    assert len(rows) == 1
    r = rows[0]
    assert (r["lat"], r["lon"], r["severity"], r["kind"], r["uid"]) == (15.5, 125.0, 3, "storm", "gdacs:TC1001")
    assert sources.parse_gdacs(b"<html>blocked") == []


def test_parse_stooq_rejects_nd_and_computes_change():
    csv_text = "Symbol,Date,Time,Open,High,Low,Close,Volume\nES.F,2026-10-21,14:00:00,5880,5900,5870,5890,1000\nNQ.F,N/D,N/D,N/D,N/D,N/D,N/D,N/D\n"
    rows = sources.parse_stooq(csv_text, datetime(2026, 10, 21, 14, 0, tzinfo=timezone.utc))
    assert len(rows) == 1 and rows[0]["symbol"] == "ES" and rows[0]["chg"] == 10 and round(rows[0]["chg_pct"], 3) == round(10 / 5880 * 100, 3)


def test_fetch_never_raises_on_network_failure(monkeypatch):
    monkeypatch.setattr(sources, "_get", lambda *a, **k: (None, "ConnectionError: offline", {}))
    assert sources.fetch_feed(sources.FEEDS[0]) == ([], "ConnectionError: offline")
    assert sources.fetch_geo("usgs") == ([], "ConnectionError: offline")
    assert sources.fetch_quotes() == ([], "ConnectionError: offline")


def test_fetch_geo_reports_parse_errors(monkeypatch):
    monkeypatch.setattr(sources, "_get", lambda *a, **k: (b"<html>blocked</html>", None, {}))
    items, err = sources.fetch_geo("usgs")
    assert items == [] and err and "parse error" in err
    assert json.dumps(sources.FEEDS[0])      # catalogue is plain data


def test_catalogue_is_well_formed():
    ids = [f["id"] for f in sources.FEEDS]
    assert len(ids) == len(set(ids)) >= 15
    for f in sources.FEEDS:
        assert f["tier"] in (1, 2, 3, 4) and f["publisher"] and (f.get("url", "").startswith("https://") or f.get("gnews"))
