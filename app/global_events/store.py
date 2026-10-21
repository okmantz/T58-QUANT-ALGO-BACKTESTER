"""SQLite store for Global Events (WAL mode, one writer lock, short-lived connections).

Everything the page shows is read from here. All timestamps are stored as UTC ISO-8601 strings (``...Z``) and only
converted to the viewer's time zone in the browser -- an explicit test pins the 08:30 ET -> UTC mapping across both DST
transitions, because the timezone bug class has bitten this desk before.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import statistics
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import classify, theaters as th

_WRITE_LOCK = threading.RLock()
_initialised: set[str] = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS ge_source_health (
  source_id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT,
  last_attempt_utc TEXT, last_ok_utc TEXT, last_error TEXT,
  record_count INTEGER DEFAULT 0, max_stale_sec INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS ge_headlines (
  id INTEGER PRIMARY KEY, source_id TEXT NOT NULL, publisher TEXT NOT NULL, title TEXT NOT NULL, title_norm TEXT NOT NULL,
  url TEXT NOT NULL, published_utc TEXT, fetched_utc TEXT NOT NULL, tier INTEGER, category TEXT,
  impact_raw INTEGER NOT NULL DEFAULT 0, impact INTEGER NOT NULL DEFAULT 0, instruments TEXT, theaters TEXT, cluster_id INTEGER,
  UNIQUE(source_id, title_norm)
);
CREATE INDEX IF NOT EXISTS ix_ge_head_pub ON ge_headlines(published_utc DESC);
CREATE INDEX IF NOT EXISTS ix_ge_head_cluster ON ge_headlines(cluster_id);
CREATE TABLE IF NOT EXISTS ge_events (
  id INTEGER PRIMARY KEY, event_uid TEXT NOT NULL UNIQUE, kind TEXT NOT NULL, name TEXT NOT NULL, country TEXT,
  impact INTEGER NOT NULL, scheduled_utc TEXT NOT NULL, forecast TEXT, previous TEXT, actual TEXT,
  instruments TEXT, source TEXT NOT NULL, updated_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ge_evt_time ON ge_events(scheduled_utc);
CREATE TABLE IF NOT EXISTS ge_quotes (
  symbol TEXT NOT NULL, ts_utc TEXT NOT NULL, price REAL, chg REAL, chg_pct REAL, source TEXT NOT NULL,
  PRIMARY KEY(symbol, ts_utc)
);
CREATE TABLE IF NOT EXISTS ge_geo_events (
  id INTEGER PRIMARY KEY, uid TEXT NOT NULL UNIQUE, source TEXT NOT NULL, kind TEXT NOT NULL, lat REAL, lon REAL,
  theater TEXT, severity INTEGER NOT NULL DEFAULT 0, magnitude REAL, title TEXT NOT NULL, url TEXT,
  occurred_utc TEXT NOT NULL, fetched_utc TEXT NOT NULL, market_relevant INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_ge_geo_time ON ge_geo_events(occurred_utc DESC);
CREATE TABLE IF NOT EXISTS ge_theater_scores (
  theater TEXT NOT NULL, ts_utc TEXT NOT NULL, score REAL NOT NULL, components TEXT NOT NULL, PRIMARY KEY(theater, ts_utc)
);
CREATE TABLE IF NOT EXISTS ge_alerts (
  id INTEGER PRIMARY KEY, ts_utc TEXT NOT NULL, type TEXT NOT NULL, ref_id INTEGER, title TEXT NOT NULL,
  severity INTEGER NOT NULL, dedupe_key TEXT NOT NULL, ack INTEGER NOT NULL DEFAULT 0
);
"""


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def db_path() -> Path:
    override = os.environ.get("T58_GLOBAL_EVENTS_DB")
    if override:
        return Path(override)
    from app.data.storage import get_app_base_dir

    return get_app_base_dir() / "data" / "global_events" / "global_events.db"


def connect() -> sqlite3.Connection:
    path = db_path()
    key = str(path)
    if key not in _initialised:
        with _WRITE_LOCK:
            if key not in _initialised:
                path.parent.mkdir(parents=True, exist_ok=True)
                c = sqlite3.connect(path, timeout=10)
                try:
                    c.execute("PRAGMA journal_mode=WAL")
                    c.executescript(SCHEMA)
                    c.commit()
                finally:
                    c.close()
                _initialised.add(key)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _loads(s: str | None) -> list:
    try:
        return json.loads(s) if s else []
    except (ValueError, TypeError):
        return []


# --------------------------------------------------------------------------- health
def record_health(source_id: str, kind: str, label: str, max_stale_sec: int, *, ok: bool, count: int = 0, error: str | None = None) -> None:
    now = iso(utcnow())
    with _WRITE_LOCK, connect() as c:
        c.execute(
            "INSERT INTO ge_source_health(source_id, kind, label, last_attempt_utc, last_ok_utc, last_error, record_count, max_stale_sec)"
            " VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET kind=excluded.kind, label=excluded.label,"
            " last_attempt_utc=excluded.last_attempt_utc, max_stale_sec=excluded.max_stale_sec,"
            " last_ok_utc=CASE WHEN ? THEN excluded.last_ok_utc ELSE ge_source_health.last_ok_utc END,"
            " last_error=CASE WHEN ? THEN NULL ELSE excluded.last_error END,"
            " record_count=CASE WHEN ? THEN excluded.record_count ELSE ge_source_health.record_count END",
            (source_id, kind, label, now, now if ok else None, None if ok else (error or "error")[:300], count, max_stale_sec,
             1 if ok else 0, 1 if ok else 0, 1 if ok else 0),
        )


def health_rows(now: datetime | None = None) -> list[dict]:
    now = now or utcnow()
    out = []
    with connect() as c:
        rows = c.execute("SELECT * FROM ge_source_health ORDER BY kind, source_id").fetchall()
    for r in rows:
        ok_at = classify.parse_when(r["last_ok_utc"])
        age = (now - ok_at).total_seconds() if ok_at else None
        if not r["enabled"]:
            status = "disabled"
        elif ok_at is None:
            status = "error" if r["last_error"] else "no_data"
        elif r["last_error"]:
            status = "error"
        elif age is not None and age > 3 * r["max_stale_sec"]:
            status = "very_stale"
        elif age is not None and age > r["max_stale_sec"]:
            status = "stale"
        elif r["record_count"] == 0:
            status = "empty"
        else:
            status = "ok"
        out.append({"id": r["source_id"], "kind": r["kind"], "label": r["label"] or r["source_id"], "status": status,
                    "last_ok": r["last_ok_utc"], "age_sec": None if age is None else int(age), "count": r["record_count"],
                    "error": r["last_error"]})
    return out


def health_summary(rows: list[dict] | None = None) -> dict:
    rows = health_rows() if rows is None else rows
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    freshest = min((r["age_sec"] for r in rows if r["age_sec"] is not None), default=None)
    overall = "no_data" if not rows else ("ok" if counts.get("ok", 0) >= max(1, len(rows) // 2) else "degraded")
    return {"overall": overall, "counts": counts, "freshest_age_sec": freshest, "sources": len(rows)}


# --------------------------------------------------------------------------- headlines
def ingest_headlines(items: list[dict], now: datetime | None = None) -> int:
    """Insert new headlines (dedupe on source+normalised title), cluster near-duplicates, apply tier-capped impact.

    Each item: source_id, publisher, title, url, published (datetime|None), tier (int|None). Returns rows inserted."""
    now = now or utcnow()
    inserted = 0
    with _WRITE_LOCK, connect() as c:
        recent = c.execute(
            "SELECT id, cluster_id, title, publisher, published_utc, fetched_utc FROM ge_headlines WHERE COALESCE(published_utc, fetched_utc) >= ?",
            (iso(now - timedelta(hours=12)),),
        ).fetchall()
        existing = []
        for r in recent:
            w = classify.parse_when(r["published_utc"] or r["fetched_utc"]) or now
            existing.append((r["cluster_id"] or r["id"], classify.tokens(r["title"]), r["publisher"], w))
        for it in items:
            title = (it.get("title") or "").strip()
            url = (it.get("url") or "").strip()
            if not title or not url:
                continue
            norm = classify.normalize_title(title)
            pub: datetime = it.get("published") or now
            if pub > now + timedelta(hours=2):
                pub = now
            info = classify.classify(title)
            tk = classify.tokens(title)
            cid, corroborated = classify.cluster_into(existing, tk, it["publisher"], pub)
            try:
                cur = c.execute(
                    "INSERT INTO ge_headlines(source_id, publisher, title, title_norm, url, published_utc, fetched_utc, tier, category,"
                    " impact_raw, impact, instruments, theaters, cluster_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (it["source_id"], it["publisher"], title[:400], norm, url[:600], iso(pub), iso(now), it.get("tier"), info["category"],
                     info["impact_raw"], classify.effective_impact(info["impact_raw"], it.get("tier"), corroborated),
                     json.dumps(info["instruments"]), json.dumps(info["theaters"]), cid),
                )
            except sqlite3.IntegrityError:
                continue
            inserted += 1
            rid = cur.lastrowid
            if cid is None:
                c.execute("UPDATE ge_headlines SET cluster_id=? WHERE id=?", (rid, rid))
                cid = rid
            existing.append((cid, tk, it["publisher"], pub))
            if corroborated:  # a second publisher confirms the story: lift every capped member of the cluster
                c.execute("UPDATE ge_headlines SET impact=impact_raw WHERE cluster_id=? AND impact<impact_raw", (cid,))
        c.commit()
    return inserted


def headlines(range_h: float = 24, min_impact: int = 0, theater: str | None = None, instrument: str | None = None,
              tier_max: int | None = None, limit: int = 40, now: datetime | None = None) -> list[dict]:
    """Clustered headline list: one row per story (best source first), newest/highest-impact first."""
    now = now or utcnow()
    since = iso(now - timedelta(hours=range_h))
    sql = ("SELECT * FROM ge_headlines WHERE COALESCE(published_utc, fetched_utc) >= ? AND impact >= ?")
    args: list = [since, int(min_impact)]
    if theater:
        sql += " AND theaters LIKE ?"
        args.append(f'%"{theater}"%')
    if instrument:
        sql += " AND instruments LIKE ?"
        args.append(f'%"{instrument}"%')
    if tier_max:
        sql += " AND (tier IS NOT NULL AND tier <= ?)"
        args.append(int(tier_max))
    sql += " ORDER BY COALESCE(published_utc, fetched_utc) DESC LIMIT 600"
    with connect() as c:
        rows = c.execute(sql, args).fetchall()
    groups: dict[int, list[sqlite3.Row]] = {}
    for r in rows:
        groups.setdefault(r["cluster_id"] or r["id"], []).append(r)
    out = []
    for members in groups.values():
        best = sorted(members, key=lambda r: ((r["tier"] or 9), -(len(r["title"]))))[0]
        out.append({
            "id": best["id"], "title": best["title"], "url": best["url"], "publisher": best["publisher"], "tier": best["tier"],
            "published": best["published_utc"], "category": best["category"], "impact": max(m["impact"] for m in members),
            "instruments": _loads(best["instruments"]), "theaters": _loads(best["theaters"]),
            "outlets": len({m["publisher"] for m in members}),
            "also": sorted({m["publisher"] for m in members if m["publisher"] != best["publisher"]})[:4],
        })
    out.sort(key=lambda h: (h["published"] or ""), reverse=True)
    out.sort(key=lambda h: h["impact"], reverse=True)
    return out[:limit]


# --------------------------------------------------------------------------- events (calendar)
def event_uid(source: str, name: str, country: str, scheduled_utc: str) -> str:
    return hashlib.sha1(f"{source}|{name}|{country}|{scheduled_utc}".encode()).hexdigest()[:20]


def upsert_events(items: list[dict]) -> int:
    now = iso(utcnow())
    n = 0
    with _WRITE_LOCK, connect() as c:
        for e in items:
            c.execute(
                "INSERT INTO ge_events(event_uid, kind, name, country, impact, scheduled_utc, forecast, previous, actual, instruments, source, updated_utc)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(event_uid) DO UPDATE SET impact=excluded.impact, forecast=excluded.forecast,"
                " previous=excluded.previous, actual=excluded.actual, updated_utc=excluded.updated_utc",
                (e["uid"], e.get("kind", "econ"), e["name"], e.get("country"), e["impact"], e["scheduled_utc"], e.get("forecast"),
                 e.get("previous"), e.get("actual"), json.dumps(e.get("instruments", [])), e["source"], now),
            )
            n += 1
        c.commit()
    return n


def events(from_utc: datetime, to_utc: datetime, min_impact: int = 1, limit: int = 60) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM ge_events WHERE scheduled_utc >= ? AND scheduled_utc <= ? AND impact >= ? ORDER BY scheduled_utc LIMIT ?",
            (iso(from_utc), iso(to_utc), int(min_impact), limit)).fetchall()
    return [{"id": r["id"], "name": r["name"], "country": r["country"], "impact": r["impact"], "when": r["scheduled_utc"],
             "forecast": r["forecast"], "previous": r["previous"], "actual": r["actual"], "instruments": _loads(r["instruments"])} for r in rows]


# --------------------------------------------------------------------------- geo
def upsert_geo(items: list[dict]) -> int:
    now = iso(utcnow())
    n = 0
    with _WRITE_LOCK, connect() as c:
        for g in items:
            theater = th.theater_for_point(g.get("lon"), g.get("lat"))
            hub = th.near_hazard_hub(g.get("lon"), g.get("lat"))
            if hub and not theater:
                theater = "hazard"
            c.execute(
                "INSERT INTO ge_geo_events(uid, source, kind, lat, lon, theater, severity, magnitude, title, url, occurred_utc, fetched_utc, market_relevant)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(uid) DO UPDATE SET severity=excluded.severity, magnitude=excluded.magnitude,"
                " title=excluded.title, fetched_utc=excluded.fetched_utc, theater=excluded.theater, market_relevant=excluded.market_relevant",
                (g["uid"], g["source"], g["kind"], g.get("lat"), g.get("lon"), theater, g.get("severity", 0), g.get("magnitude"), g["title"][:240],
                 g.get("url"), g["occurred_utc"], now, 1 if (hub or theater) else 0),
            )
            n += 1
        c.commit()
    return n


def geo_events(range_h: float = 72, min_severity: int = 0, limit: int = 400, now: datetime | None = None) -> list[dict]:
    now = now or utcnow()
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM ge_geo_events WHERE occurred_utc >= ? AND severity >= ? ORDER BY occurred_utc DESC LIMIT ?",
            (iso(now - timedelta(hours=range_h)), int(min_severity), limit)).fetchall()
    return [{"id": r["id"], "kind": r["kind"], "lat": r["lat"], "lon": r["lon"], "sev": r["severity"], "mag": r["magnitude"],
             "title": r["title"], "url": r["url"], "t": r["occurred_utc"], "theater": r["theater"], "mkt": bool(r["market_relevant"]),
             "src": r["source"]} for r in rows]


# --------------------------------------------------------------------------- quotes
def insert_quotes(rows: list[dict]) -> int:
    n = 0
    with _WRITE_LOCK, connect() as c:
        for q in rows:
            c.execute("INSERT OR REPLACE INTO ge_quotes(symbol, ts_utc, price, chg, chg_pct, source) VALUES(?,?,?,?,?,?)",
                      (q["symbol"], q["ts_utc"], q["price"], q.get("chg"), q.get("chg_pct"), q["source"]))
            n += 1
        c.commit()
    return n


def quotes_latest(symbols: list[str], spark_points: int = 24) -> list[dict]:
    out = []
    with connect() as c:
        for s in symbols:
            rows = c.execute("SELECT ts_utc, price, chg, chg_pct FROM ge_quotes WHERE symbol=? ORDER BY ts_utc DESC LIMIT ?", (s, spark_points)).fetchall()
            if not rows:
                out.append({"symbol": s, "price": None, "chg_pct": None, "ts": None, "spark": []})
                continue
            first = rows[0]
            out.append({"symbol": s, "price": first["price"], "chg": first["chg"], "chg_pct": first["chg_pct"], "ts": first["ts_utc"],
                        "spark": [r["price"] for r in reversed(rows) if r["price"] is not None]})
    return out


# --------------------------------------------------------------------------- theater heat
def _day_bucket(dt: datetime, now: datetime) -> int:
    return int((now - dt).total_seconds() // 86400)


def compute_theater_scores(now: datetime | None = None) -> list[dict]:
    """Compute and persist Theater Heat for every theater from the stored headlines + hazard events."""
    now = now or utcnow()
    tier_w = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.3}
    with connect() as c:
        heads = c.execute(
            "SELECT tier, impact, theaters, published_utc, fetched_utc, cluster_id, id FROM ge_headlines WHERE COALESCE(published_utc, fetched_utc) >= ?",
            (iso(now - timedelta(days=15)),)).fetchall()
        geos = c.execute("SELECT theater, severity, occurred_utc FROM ge_geo_events WHERE occurred_utc >= ? AND theater IS NOT NULL",
                         (iso(now - timedelta(days=2)),)).fetchall()
    per_day: dict[str, dict[int, float]] = {t.id: {} for t in th.THEATERS}
    hi_6h: dict[str, int] = {t.id: 0 for t in th.THEATERS}
    seen_clusters: set[tuple[str, int]] = set()
    covered_days: set[int] = set()
    for r in heads:
        when = classify.parse_when(r["published_utc"] or r["fetched_utc"]) or now
        day = _day_bucket(when, now)
        if day < 0 or day > 14:
            continue
        covered_days.add(day)
        w = tier_w.get(r["tier"] or 3, 0.5) * (1.0 + 0.5 * (r["impact"] or 0))
        for tid in _loads(r["theaters"]):
            if tid not in per_day:
                continue
            key = (tid, r["cluster_id"] or r["id"])
            if key in seen_clusters:   # a story carried by 7 outlets counts once
                continue
            seen_clusters.add(key)
            per_day[tid][day] = per_day[tid].get(day, 0.0) + w
            if (r["impact"] or 0) >= 2 and (now - when).total_seconds() <= 6 * 3600:
                hi_6h[tid] += 1
    hz: dict[str, int] = {t.id: 0 for t in th.THEATERS}
    for g in geos:
        if g["severity"] >= 2 and g["theater"] in hz:
            hz[g["theater"]] += 1
    results = []
    ts = iso(now)
    calibrating = len(covered_days) < th.CALIBRATION_DAYS
    with _WRITE_LOCK, connect() as c:
        for t in th.THEATERS:
            hits_24 = per_day[t.id].get(0, 0.0)
            hist = [per_day[t.id].get(d, 0.0) for d in range(1, 15)]
            mean = statistics.fmean(hist) if hist else 0.0
            std = statistics.pstdev(hist) if len(hist) > 1 else 0.0
            score, comps = th.heat(hits_24, mean, std, hi_6h[t.id], hz[t.id], calibrating)
            # previous score ~24h ago for the delta arrow
            prev = c.execute("SELECT score FROM ge_theater_scores WHERE theater=? AND ts_utc<=? ORDER BY ts_utc DESC LIMIT 1",
                             (t.id, iso(now - timedelta(hours=24)))).fetchone()
            c.execute("INSERT OR REPLACE INTO ge_theater_scores(theater, ts_utc, score, components) VALUES(?,?,?,?)",
                      (t.id, ts, score, json.dumps(comps)))
            results.append({"id": t.id, "score": score, "delta": None if not prev else round(score - prev["score"], 1), "components": comps})
        c.commit()
    return results


def theater_scores() -> list[dict]:
    """Latest score per theater joined with static theater info and its top headline."""
    out = []
    with connect() as c:
        for t in th.THEATERS:
            r = c.execute("SELECT score, components, ts_utc FROM ge_theater_scores WHERE theater=? ORDER BY ts_utc DESC LIMIT 1", (t.id,)).fetchone()
            prev = c.execute("SELECT score FROM ge_theater_scores WHERE theater=? AND ts_utc<=? ORDER BY ts_utc DESC LIMIT 1",
                             (t.id, iso(utcnow() - timedelta(hours=24)))).fetchone()
            top = c.execute("SELECT title, url, publisher FROM ge_headlines WHERE theaters LIKE ? AND impact>=1 ORDER BY COALESCE(published_utc, fetched_utc) DESC LIMIT 1",
                            (f'%"{t.id}"%',)).fetchone()
            item = {"id": t.id, "name": t.name, "short": t.short, "instruments": list(t.instruments), "bbox": t.bbox, "label_at": t.label_at,
                    "score": None, "delta": None, "components": None, "ts": None, "top": None}
            if r:
                item.update(score=r["score"], components=json.loads(r["components"]), ts=r["ts_utc"],
                            delta=None if not prev else round(r["score"] - prev["score"], 1))
            if top:
                item["top"] = {"title": top["title"], "url": top["url"], "publisher": top["publisher"]}
            out.append(item)
    return out


# --------------------------------------------------------------------------- housekeeping + snapshot
def prune(now: datetime | None = None) -> None:
    now = now or utcnow()
    with _WRITE_LOCK, connect() as c:
        c.execute("DELETE FROM ge_headlines WHERE COALESCE(published_utc, fetched_utc) < ?", (iso(now - timedelta(days=30)),))
        c.execute("DELETE FROM ge_geo_events WHERE occurred_utc < ?", (iso(now - timedelta(days=30)),))
        c.execute("DELETE FROM ge_quotes WHERE ts_utc < ?", (iso(now - timedelta(days=7)),))
        c.execute("DELETE FROM ge_theater_scores WHERE ts_utc < ?", (iso(now - timedelta(days=30)),))
        c.execute("DELETE FROM ge_events WHERE scheduled_utc < ?", (iso(now - timedelta(days=60)),))
        c.commit()


def context_snapshot(max_headlines: int = 8) -> dict:
    now = utcnow()
    hs = health_rows(now)
    return {
        "generated_utc": iso(now),
        "next_events": events(now - timedelta(minutes=30), now + timedelta(hours=48), min_impact=2, limit=6),
        "headlines": [{"title": h["title"], "publisher": h["publisher"], "impact": h["impact"], "published": h["published"], "instruments": h["instruments"]}
                      for h in headlines(range_h=12, min_impact=2, limit=max_headlines, now=now)],
        "theaters": [{"theater": t["short"], "score": t["score"], "delta": t["delta"]} for t in theater_scores() if t["score"] is not None],
        "freshness": {h["id"]: h["age_sec"] for h in hs},
        "health": health_summary(hs),
    }
