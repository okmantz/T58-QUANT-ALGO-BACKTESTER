"""Upstream fetchers + pure parsers for Global Events.

Parsers are pure (bytes/JSON in, plain dicts out) so they are unit-tested against saved sample payloads. Fetchers never
raise: they return ``(items, error)`` and the poller records the outcome in the source-health table. Nothing here is
called from a request handler.

The endpoints below come from each provider's public documentation; none were live-tested when this module was written
(the build sandbox had no internet), which is exactly why every source reports its own health in the UI.
"""
from __future__ import annotations

import csv
import io
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from . import store

USER_AGENT = "T58-GlobalEvents/1.0 (+local desktop app; contact: support@t58trading.com)"
MAX_BYTES = 2_000_000
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 8

# --------------------------------------------------------------------------- source catalogue (our own curation)
# tier: 1 wire / official, 2 major outlet, 3 specialist / aggregator, 4 commentary.
FEEDS: tuple[dict, ...] = (
    {"id": "fed_press", "publisher": "Federal Reserve", "tier": 1, "url": "https://www.federalreserve.gov/feeds/press_all.xml"},
    {"id": "ecb_press", "publisher": "European Central Bank", "tier": 1, "url": "https://www.ecb.europa.eu/rss/press.html"},
    {"id": "sec_press", "publisher": "SEC", "tier": 1, "url": "https://www.sec.gov/news/pressreleases.rss"},
    {"id": "eia_today", "publisher": "EIA", "tier": 1, "url": "https://www.eia.gov/rss/todayinenergy.xml"},
    {"id": "gn_reuters", "publisher": "Reuters", "tier": 1, "gnews": "site:reuters.com markets OR economy OR oil when:1d"},
    {"id": "gn_reuters_world", "publisher": "Reuters", "tier": 1, "gnews": "site:reuters.com world conflict OR sanctions OR tariffs when:1d"},
    {"id": "gn_ap", "publisher": "Associated Press", "tier": 1, "gnews": "site:apnews.com business OR world when:1d"},
    {"id": "gn_bloomberg", "publisher": "Bloomberg", "tier": 1, "gnews": "site:bloomberg.com markets when:1d"},
    {"id": "cnbc_top", "publisher": "CNBC", "tier": 2, "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"},
    {"id": "cnbc_markets", "publisher": "CNBC", "tier": 2, "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html"},
    {"id": "mw_top", "publisher": "MarketWatch", "tier": 2, "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories"},
    {"id": "bbc_business", "publisher": "BBC News", "tier": 2, "url": "https://feeds.bbci.co.uk/news/business/rss.xml"},
    {"id": "bbc_world", "publisher": "BBC News", "tier": 2, "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"id": "npr_business", "publisher": "NPR", "tier": 2, "url": "https://feeds.npr.org/1006/rss.xml"},
    {"id": "aljazeera", "publisher": "Al Jazeera", "tier": 2, "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    {"id": "guardian_world", "publisher": "The Guardian", "tier": 2, "url": "https://www.theguardian.com/world/rss"},
    {"id": "dw_all", "publisher": "DW", "tier": 2, "url": "https://rss.dw.com/rdf/rss-en-all"},
    {"id": "yahoo_fin", "publisher": "Yahoo Finance", "tier": 3, "url": "https://finance.yahoo.com/news/rssindex"},
    {"id": "oilprice", "publisher": "OilPrice.com", "tier": 3, "url": "https://oilprice.com/rss/main"},
    {"id": "coindesk", "publisher": "CoinDesk", "tier": 3, "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
)
FEED_MAX_STALE_SEC = 30 * 60

QUOTE_SYMBOLS: tuple[tuple[str, str, str], ...] = (   # (display, stooq symbol, label)
    ("ES", "es.f", "S&P 500 fut"), ("NQ", "nq.f", "Nasdaq fut"), ("YM", "ym.f", "Dow fut"), ("RTY", "rty.f", "Russell fut"),
    ("DXY", "dx.f", "Dollar idx"), ("GC", "gc.f", "Gold"), ("CL", "cl.f", "WTI crude"), ("BTC", "btc.v", "Bitcoin"), ("VIX", "^vix", "VIX"),
)


def _get(url: str, headers: dict | None = None) -> tuple[bytes | None, str | None, dict]:
    """GET with a hard size cap. Returns (body, error, response_headers)."""
    import requests

    try:
        with requests.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True,
                          headers={"User-Agent": USER_AGENT, "Accept": "*/*", **(headers or {})}) as r:
            if r.status_code == 304:
                return None, "not_modified", dict(r.headers)
            r.raise_for_status()
            buf = bytearray()
            for chunk in r.iter_content(65536):
                buf.extend(chunk)
                if len(buf) > MAX_BYTES:
                    return None, "response too large", {}
            return bytes(buf), None, dict(r.headers)
    except Exception as exc:  # noqa: BLE001 -- a dead source must never take the poller down
        return None, f"{type(exc).__name__}: {str(exc)[:160]}", {}


# --------------------------------------------------------------------------- RSS / Atom
def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el: ET.Element, name: str) -> ET.Element | None:
    for ch in el:
        if _local(ch.tag) == name:
            return ch
    return None


def _parse_date(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        d = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    if d is None:
        return None
    return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)


def parse_feed(body: bytes, default_publisher: str, limit: int = 25) -> list[dict]:
    """RSS 2.0 / RDF / Atom -> [{title, url, published, publisher}]. Tolerates missing fields; returns [] on bad XML."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    out = []
    for el in root.iter():
        name = _local(el.tag)
        if name not in ("item", "entry"):
            continue
        title = _text(_child(el, "title"))
        link_el = _child(el, "link")
        url = (link_el.get("href") if link_el is not None and link_el.get("href") else _text(link_el)) or _text(_child(el, "guid"))
        pub = _parse_date(_text(_child(el, "pubDate")) or _text(_child(el, "published")) or _text(_child(el, "updated")) or _text(_child(el, "date")))
        publisher = default_publisher
        src = _child(el, "source")
        if src is not None and _text(src):   # Google News carries the real outlet here
            publisher = _text(src)
            if title.endswith(" - " + publisher):
                title = title[: -(len(publisher) + 3)]
        if title and url.startswith("http"):
            out.append({"title": title, "url": url, "published": pub, "publisher": publisher})
        if len(out) >= limit:
            break
    return out


def google_news_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"


_etag_cache: dict[str, dict] = {}


def fetch_feed(feed: dict) -> tuple[list[dict], str | None]:
    url = google_news_url(feed["gnews"]) if feed.get("gnews") else feed["url"]
    hdr = {}
    cached = _etag_cache.get(url)
    if cached:
        if cached.get("etag"):
            hdr["If-None-Match"] = cached["etag"]
        if cached.get("modified"):
            hdr["If-Modified-Since"] = cached["modified"]
    body, err, resp_h = _get(url, hdr)
    if err == "not_modified":
        return [], None
    if err or body is None:
        return [], err or "empty response"
    _etag_cache[url] = {"etag": resp_h.get("ETag"), "modified": resp_h.get("Last-Modified")}
    items = parse_feed(body, feed["publisher"])
    for it in items:
        it["source_id"] = feed["id"]
        it["tier"] = feed["tier"]
    return items, (None if items else "feed parsed but contained no items")


# --------------------------------------------------------------------------- hazards
USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson"
EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&days=14&limit=120"
GDACS_URL = "https://www.gdacs.org/xml/rss.xml"
_EONET_SKIP = {"earthquakes", "seaLakeIce", "manmade", "snow", "tempExtremes", "dustHaze"}
_EONET_KIND = {"wildfires": "fire", "volcanoes": "volcano", "severeStorms": "storm", "floods": "flood", "landslides": "flood", "drought": "other"}


def parse_usgs(data: dict) -> list[dict]:
    out = []
    for f in data.get("features", []):
        p, g = f.get("properties") or {}, f.get("geometry") or {}
        coords = g.get("coordinates") or []
        mag, t = p.get("mag"), p.get("time")
        if len(coords) < 2 or mag is None or t is None:
            continue
        sev = 3 if mag >= 7 else 2 if mag >= 6 else 1 if mag >= 5 else 0
        out.append({"uid": f"usgs:{f.get('id')}", "source": "usgs", "kind": "quake", "lat": coords[1], "lon": coords[0], "severity": sev,
                    "magnitude": float(mag), "title": p.get("title") or f"M{mag} earthquake", "url": p.get("url"),
                    "occurred_utc": store.iso(datetime.fromtimestamp(t / 1000, tz=timezone.utc))})
    return out


def _first_point(geom: dict) -> tuple[float, float] | None:
    c = geom.get("coordinates")
    if not c:
        return None
    if geom.get("type") == "Point" and len(c) >= 2:
        return float(c[0]), float(c[1])
    # Polygon -> centroid of the outer ring's vertices (cheap, good enough for a map dot)
    try:
        ring = c[0] if geom.get("type") == "Polygon" else c[0][0]
        xs = [float(p[0]) for p in ring]
        ys = [float(p[1]) for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    except (TypeError, IndexError, ValueError, ZeroDivisionError):
        return None


def parse_eonet(data: dict) -> list[dict]:
    out = []
    for ev in data.get("events", []):
        cats = [c.get("id") for c in ev.get("categories", [])]
        if not cats or any(c in _EONET_SKIP for c in cats):
            continue
        geoms = ev.get("geometry") or []
        if not geoms:
            continue
        last = geoms[-1]
        pt = _first_point(last)
        when = _parse_date(last.get("date", ""))
        if not pt or not when:
            continue
        kind = _EONET_KIND.get(cats[0], "other")
        out.append({"uid": f"eonet:{ev.get('id')}", "source": "eonet", "kind": kind, "lat": pt[1], "lon": pt[0],
                    "severity": 2 if kind in ("volcano", "storm") else 1, "magnitude": None, "title": ev.get("title", "Natural event"),
                    "url": ev.get("link"), "occurred_utc": store.iso(when)})
    return out


def parse_gdacs(body: bytes) -> list[dict]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    kinds = {"EQ": "quake", "TC": "storm", "FL": "flood", "VO": "volcano", "WF": "fire", "DR": "other"}
    levels = {"green": 1, "orange": 2, "red": 3}
    out = []
    for item in root.iter("item"):
        d = {_local(ch.tag): ch for ch in item.iter() if ch is not item}   # flat by local name: geo:lat/long sit inside geo:Point
        lat, lon = _text(d.get("lat")), _text(d.get("long"))
        if not lat or not lon:
            continue
        try:
            latf, lonf = float(lat), float(lon)
        except ValueError:
            continue
        etype = _text(d.get("eventtype"))
        when = _parse_date(_text(d.get("pubDate"))) or datetime.now(timezone.utc)
        out.append({"uid": f"gdacs:{etype}{_text(d.get('eventid'))}", "source": "gdacs", "kind": kinds.get(etype, "other"), "lat": latf, "lon": lonf,
                    "severity": levels.get(_text(d.get("alertlevel")).lower(), 1), "magnitude": None, "title": _text(d.get("title")) or "GDACS alert",
                    "url": _text(d.get("link")) or None, "occurred_utc": store.iso(when)})
    return out


def fetch_geo(which: str) -> tuple[list[dict], str | None]:
    url = {"usgs": USGS_URL, "eonet": EONET_URL, "gdacs": GDACS_URL}[which]
    body, err, _ = _get(url)
    if err or body is None:
        return [], err or "empty response"
    try:
        items = parse_gdacs(body) if which == "gdacs" else (parse_usgs if which == "usgs" else parse_eonet)(json.loads(body))
    except (ValueError, AttributeError) as exc:
        return [], f"parse error: {exc}"
    return items, None


# --------------------------------------------------------------------------- calendar (reuses the app's existing feed)
USD_INSTRUMENTS = ["ES", "NQ", "YM", "ZN", "DXY", "GC"]


def calendar_items() -> tuple[list[dict], str | None]:
    from app.ai import news_forexfactory as ff

    res = ff.fetch_calendar(timeout=10)
    if res.error and not res.events:
        return [], res.error
    impact = {"High": 3, "Medium": 2, "Low": 1}
    out = []
    for e in res.events:
        if e.when is None or e.impact not in impact:
            continue
        when = store.iso(e.when)
        instruments = USD_INSTRUMENTS if e.currency == "USD" else list(ff.CURRENCY_TO_SYMBOLS.get(e.currency, []))
        out.append({"uid": store.event_uid("ff", e.title, e.currency, when), "kind": "econ", "name": e.title, "country": e.currency,
                    "impact": impact[e.impact], "scheduled_utc": when, "forecast": e.forecast or None, "previous": e.previous or None,
                    "actual": e.actual or None, "instruments": instruments, "source": "ff_calendar"})
    return out, None


# --------------------------------------------------------------------------- quotes (Stooq CSV)
def parse_stooq(text: str, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    by_symbol = {s[1].upper(): s[0] for s in QUOTE_SYMBOLS}
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        sym = by_symbol.get((row.get("Symbol") or "").upper())
        try:
            close, op = float(row["Close"]), float(row["Open"])
        except (KeyError, TypeError, ValueError):
            continue   # "N/D" rows -> unavailable, never zero
        out.append({"symbol": sym or row.get("Symbol"), "ts_utc": store.iso(now), "price": close, "chg": round(close - op, 4),
                    "chg_pct": round((close - op) / op * 100, 3) if op else None, "source": "stooq"})
    return [q for q in out if q["symbol"]]


def fetch_quotes() -> tuple[list[dict], str | None]:
    syms = "+".join(s[1] for s in QUOTE_SYMBOLS)
    body, err, _ = _get(f"https://stooq.com/q/l/?s={syms}&f=sd2t2ohlcv&h&e=csv")
    if err or body is None:
        return [], err or "empty response"
    rows = parse_stooq(body.decode("utf-8", "replace"))
    return rows, (None if rows else "no parseable quotes (N/D)")
