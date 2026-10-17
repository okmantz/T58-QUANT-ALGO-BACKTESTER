"""
Evolution Lab live feed (2026-10-17).

  GET  /api/evolution/lab/feed?since=<seq>   new equity-curve cards + the random-control scoreboard + parity status
  GET  /api/evolution/lab/finalists          finalist funnel / readiness / incubation plan (computed on a background
                                             thread; answers {"status": "computing"} until it is done)
  POST /api/evolution/lab/parity             attach a second engine's trades (JSON list or CSV text) to a candidate

Everything here is read-only against the runner except the parity attach, and nothing blocks: the feed reads an
in-memory ring buffer, the finalist report runs on its own thread, and the page polls only while its tab is visible.
The Evolution runner is a server.py global, so it is handed in as a getter (importing server from here would be circular).
"""
from __future__ import annotations

import csv
import io
import threading
import time
from typing import Any, Callable, Optional

from flask import Blueprint, jsonify, request

evolution_lab_bp = Blueprint("evolution_lab", __name__)

_runner_source: Optional[Callable[[], Any]] = None
_FINALISTS: dict = {"key": None, "state": "idle", "rows": None, "at": 0.0, "error": None}
_FINALISTS_LOCK = threading.Lock()


def set_runner_source(getter: Optional[Callable[[], Any]]) -> None:
    global _runner_source
    _runner_source = getter


def _runner():
    return _runner_source() if _runner_source else None


@evolution_lab_bp.route("/api/evolution/lab/feed")
def lab_feed():
    r = _runner()
    if r is None:
        return jsonify({"seq": 0, "items": [], "lab": None, "running": False})
    try:
        since = int(request.args.get("since", 0))
        limit = max(1, min(int(request.args.get("limit", 80)), 240))
    except ValueError:
        since, limit = 0, 80
    feed = r.curve_store.since(since, limit)
    # A restarted runner starts its sequence at 1 again; tell the browser to drop what it holds.
    reset = since > feed["seq"]
    return jsonify({**feed, "reset": reset, "lab": r.lab_status(), "running": bool(r.is_running)})


@evolution_lab_bp.route("/api/evolution/lab/finalists")
def lab_finalists():
    r = _runner()
    if r is None:
        return jsonify({"status": "idle", "rows": []})
    key = (id(r), r.generation, len(r.leaderboard), tuple(sorted(r.parity)))
    with _FINALISTS_LOCK:
        st = dict(_FINALISTS)
        if st["key"] == key and st["state"] in ("computing", "done"):
            return jsonify({"status": st["state"], "rows": st["rows"] or [], "error": st["error"]})
        _FINALISTS.update(key=key, state="computing", rows=None, error=None, at=time.time())

    def work():
        try:
            rows = r.finalists_report(top_n=5)
            with _FINALISTS_LOCK:
                if _FINALISTS["key"] == key:
                    _FINALISTS.update(state="done", rows=rows)
        except Exception as exc:  # noqa: BLE001
            with _FINALISTS_LOCK:
                if _FINALISTS["key"] == key:
                    _FINALISTS.update(state="done", rows=[], error=str(exc)[:300])

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"status": "computing", "rows": []})


def parse_reference_trades(text: str) -> list[dict]:
    """CSV text from an MT5 report / broker export -> [{entry_time, direction, pnl}]. Accepts common header names; a
    'type' column of buy/sell (or long/short) is mapped to +1/-1."""
    rows = list(csv.DictReader(io.StringIO(text.strip())))
    if not rows:
        return []
    low = {k.strip().lower(): k for k in rows[0].keys() if k}

    def col(*names):
        for n in names:
            if n in low:
                return low[n]
        return None

    c_time = col("entry_time", "open_time", "time", "open time", "entry time", "opened")
    c_dir = col("direction", "type", "side")
    c_pnl = col("pnl", "profit", "net_pnl", "net profit", "p&l")
    if not (c_time and c_dir and c_pnl):
        raise ValueError("CSV needs columns for entry time, direction/type, and pnl/profit.")
    out = []
    for r in rows:
        d = str(r[c_dir]).strip().lower()
        if d in ("buy", "long", "1", "+1"):
            dirn = 1
        elif d in ("sell", "short", "-1"):
            dirn = -1
        else:
            continue  # balance rows, deposits, etc.
        try:
            out.append({"entry_time": r[c_time], "direction": dirn, "pnl": float(str(r[c_pnl]).replace(",", ""))})
        except (TypeError, ValueError):
            continue
    return out


@evolution_lab_bp.route("/api/evolution/lab/parity", methods=["POST"])
def lab_parity():
    r = _runner()
    if r is None:
        return jsonify({"error": "No Evolution Lab run is loaded."}), 409
    body = request.get_json(silent=True) or {}
    cid = body.get("candidate_id") or request.form.get("candidate_id")
    if not cid:
        return jsonify({"error": "candidate_id is required."}), 400
    try:
        trades = body.get("trades")
        if trades is None:
            f = request.files.get("file")
            text = f.read().decode("utf-8", errors="replace") if f else (body.get("csv") or request.form.get("csv") or "")
            trades = parse_reference_trades(text)
        res = r.attach_parity(cid, trades)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001 -- malformed upload: tell the user, don't 500
        return jsonify({"error": f"Could not compare trades: {exc}"}), 400
    return jsonify({"candidate_id": cid, "parity": res})
