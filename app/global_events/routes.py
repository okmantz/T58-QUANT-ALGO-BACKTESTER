"""Flask blueprint for Global Events: one page shell + a handful of read-only JSON endpoints.

Handlers read the local store only (see package docstring). ``/events/api/bootstrap`` returns everything the first paint
needs in a single small payload, with an ETag so unchanged polls cost a 304 and no body.
"""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta

from flask import Blueprint, Response, jsonify, render_template, request

from . import brief as brief_mod
from . import sources, store
from .poller import POLLER

global_events_bp = Blueprint("global_events", __name__, url_prefix="/events")

RANGE_HOURS = {"24h": 24, "3d": 72, "7d": 168, "30d": 720}
QUOTE_ORDER = [s[0] for s in sources.QUOTE_SYMBOLS]


def _hours(default: str = "24h") -> float:
    return float(RANGE_HOURS.get(request.args.get("range", default), RANGE_HOURS[default]))


def _int(name: str, default: int, lo: int = 0, hi: int = 9) -> int:
    try:
        return max(lo, min(hi, int(request.args.get(name, default))))
    except (TypeError, ValueError):
        return default


def _json_etag(payload: dict) -> Response:
    body = json.dumps(payload, separators=(",", ":"), default=str)
    etag = hashlib.md5(body.encode()).hexdigest()  # noqa: S324 -- cache validator, not security
    if request.headers.get("If-None-Match") == etag:
        resp = Response(status=304)
    else:
        resp = Response(body, mimetype="application/json")
    resp.headers["ETag"] = etag
    resp.headers["Cache-Control"] = "private, max-age=0, must-revalidate"
    return resp


@global_events_bp.route("/")
def page():
    POLLER.ensure_started()
    return render_template("global_events.html", active_page="global_events")


@global_events_bp.route("/api/bootstrap")
def api_bootstrap():
    POLLER.ensure_started()
    now = store.utcnow()
    hrs = _hours()
    health = store.health_rows(now)
    payload = {
        "server_time": store.iso(now),
        "range": request.args.get("range", "24h"),
        "quotes": store.quotes_latest(QUOTE_ORDER),
        "quote_labels": {s[0]: s[2] for s in sources.QUOTE_SYMBOLS},
        "events": store.events(now - timedelta(minutes=45), now + timedelta(hours=72), min_impact=2, limit=24),
        "headlines": store.headlines(range_h=hrs, min_impact=_int("min_impact", 0), theater=request.args.get("theater") or None,
                                     instrument=request.args.get("instr") or None, limit=30, now=now),
        "theaters": store.theater_scores(),
        "geo": store.geo_events(range_h=max(hrs, 72), limit=300, now=now),
        "health": {"summary": store.health_summary(health), "sources": health},
        "poller_active": POLLER.active,
    }
    return _json_etag(payload)


@global_events_bp.route("/api/headlines")
def api_headlines():
    POLLER.touch()
    return jsonify(store.headlines(range_h=_hours(), min_impact=_int("min_impact", 0), theater=request.args.get("theater") or None,
                                   instrument=request.args.get("instr") or None, tier_max=_int("tier", 0, 0, 4) or None, limit=_int("limit", 40, 1, 100)))


@global_events_bp.route("/api/events")
def api_events():
    POLLER.touch()
    now = store.utcnow()
    return jsonify(store.events(now - timedelta(hours=2), now + timedelta(hours=_hours("3d")), min_impact=_int("min_impact", 1, 0, 3)))


@global_events_bp.route("/api/geo")
def api_geo():
    POLLER.touch()
    return jsonify(store.geo_events(range_h=_hours("3d"), min_severity=_int("min_severity", 0, 0, 3)))


@global_events_bp.route("/api/health")
def api_health():
    rows = store.health_rows()
    return jsonify({"summary": store.health_summary(rows), "sources": rows})


@global_events_bp.route("/api/refresh", methods=["POST"])
def api_refresh():
    started = POLLER.ensure_started()
    accepted = POLLER.request_refresh() if started else False
    return jsonify({"ok": True, "accepted": accepted, "poller": started})


@global_events_bp.route("/api/brief", methods=["POST"])
def api_brief():
    POLLER.touch()
    return jsonify(brief_mod.make_brief())


@global_events_bp.route("/api/snapshot")
def api_snapshot():
    return jsonify(store.context_snapshot())
