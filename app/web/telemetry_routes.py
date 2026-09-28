"""
Run telemetry -- web API + the Universe Map page.

Four read-only surfaces built on modules that hold all the logic:

  /api/telemetry/job/<job_id>/progress   live phase banner numbers
                                          (app.orchestration.run_progress)
  /api/telemetry/swarm/job/<job_id>      equity-curve swarm for a finished
  /api/telemetry/swarm/evolution         Search Lab / Speed Run job, or for the
                                          Evolution Lab's current elite
                                          (app.orchestration.swarm)
  /api/telemetry/universe                every strategy as a dot, by family
                                          (app.orchestration.universe)
  /api/telemetry/survivors               pipeline funnel counts
                                          (app.orchestration.survivors)
  /universe                              the page that draws the universe

Registered as a Blueprint like the other feature areas; app/web/server.py's
change is one import, one register_blueprint call and one
``set_evolution_source`` call (the Evolution runner is a server.py global, so
it is handed in rather than imported -- importing server from here would be
circular).

Nothing here mutates anything. Swarm computation runs on a background thread
(SWARM_CACHE) so a request never blocks for the backtests; the chart polls.
"""
from __future__ import annotations

import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from flask import Blueprint, jsonify, render_template, request

from app.data.storage import get_app_base_dir
from app.orchestration import survivors as survivors_module
from app.orchestration import universe as universe_module
from app.orchestration.swarm import MAX_CURVES, SWARM_CACHE, build_swarm
from app.web.job_manager import JOB_MANAGER

telemetry_bp = Blueprint("telemetry", __name__)

_logger = logging.getLogger(__name__)

_evolution_source: Optional[Callable[[], Any]] = None


def set_evolution_source(getter: Optional[Callable[[], Any]]) -> None:
    """server.py registers ``lambda: _EVOLUTION_RUNNER`` here. The getter must
    return the current EvolutionRunner or None."""
    global _evolution_source
    _evolution_source = getter


def _search_dir() -> Path:
    return get_app_base_dir() / "reports" / "search"


def _json_safe(view):
    """Same guarantee the other blueprints give: an unexpected exception is a
    JSON error body, never Flask's HTML error page (which would make every
    fetch(...).json() in the browser throw "Unexpected token '<'")."""

    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        try:
            return view(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- last line of defense
            _logger.exception("Telemetry route %s failed", view.__name__)
            return jsonify({
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            }), 500

    return wrapped


# ---------------------------------------------------------------------------
# Live progress
# ---------------------------------------------------------------------------

@telemetry_bp.route("/api/telemetry/job/<job_id>/progress")
@_json_safe
def job_progress(job_id: str):
    job = JOB_MANAGER.get(job_id)
    if job is None:
        return jsonify({"found": False, "done": False, "progress": None}), 404
    return jsonify({"found": True, "done": bool(job.get("done")), "progress": JOB_MANAGER.get_progress(job_id)})


# ---------------------------------------------------------------------------
# Equity swarm
# ---------------------------------------------------------------------------

def _search_summary_of(job: dict) -> Any:
    """The SearchSummary behind a Search Lab job (job['summary']) or a Speed
    Run job (job['result'].search_summary), else None."""
    summary = job.get("summary")
    if summary is not None:
        return summary
    return getattr(job.get("result"), "search_summary", None)


def _load_search_rows(db_path: str, run_id: str, limit: int) -> list[dict]:
    from app.search.results_db import ResultsDB

    with ResultsDB(db_path) as db:
        return db.live_leaderboard(run_id, top_n=limit)


def _swarm_reply(key: str, compute: Callable, force: bool):
    entry = SWARM_CACHE.get(key)
    if entry is None or (force and entry["status"] != "running"):
        entry = SWARM_CACHE.start(key, compute, force=force)
    return jsonify({
        "status": entry["status"], "done": entry["done"], "total": entry["total"],
        "swarm": entry["result"], "error": entry["error"],
    })


@telemetry_bp.route("/api/telemetry/swarm/job/<job_id>")
@_json_safe
def swarm_for_job(job_id: str):
    job = JOB_MANAGER.get(job_id)
    if job is None:
        return jsonify({"status": "unavailable", "reason": "Job not found (the server may have restarted)."}), 404
    summary = _search_summary_of(job)
    if not job.get("done"):
        # "pending" tells the chart to keep polling; "unavailable" tells it to stop.
        return jsonify({"status": "pending", "reason": "The swarm is drawn once the run finishes."})
    df, risk = job.get("df"), job.get("risk")
    if summary is None or df is None or risk is None:
        return jsonify({
            "status": "unavailable",
            "reason": "This run type doesn't keep the data needed to redraw its candidates.",
        })

    def compute(progress_cb):
        rows = _load_search_rows(summary.db_path, summary.run_id, MAX_CURVES)
        return build_swarm(df, risk, rows, progress_cb=progress_cb)

    return _swarm_reply(f"job:{job_id}", compute, force=request.args.get("refresh") == "1")


@telemetry_bp.route("/api/telemetry/swarm/evolution")
@_json_safe
def swarm_for_evolution():
    runner = _evolution_source() if _evolution_source is not None else None
    if runner is None or not getattr(runner, "leaderboard", None):
        return jsonify({"status": "pending", "reason": "Waiting for Evolution Lab's first elite candidates…"})
    records = list(runner.leaderboard)
    candidates = []
    for r in records[:MAX_CURVES]:
        cand = {"candidate_id": r.candidate_id, "family": (r.meta or {}).get("family", "")}
        cand.update(r.spec or {})
        candidates.append(cand)
    # Key on the elite set itself so the swarm refreshes when the elite changes.
    key = "evolution:" + ",".join(sorted(str(c["candidate_id"]) for c in candidates))
    df, risk = runner.df, runner.risk

    def compute(progress_cb):
        return build_swarm(df, risk, candidates, progress_cb=progress_cb)

    return _swarm_reply(key, compute, force=request.args.get("refresh") == "1")


# ---------------------------------------------------------------------------
# Universe map + survivors funnel
# ---------------------------------------------------------------------------

_VALID_SOURCES = ("search", "evolution", "library")


@telemetry_bp.route("/api/telemetry/universe")
@_json_safe
def universe_data():
    raw = request.args.get("sources", "")
    wanted = [s for s in (x.strip() for x in raw.split(",")) if s in _VALID_SOURCES] or list(_VALID_SOURCES)
    return jsonify(universe_module.load_universe(_search_dir(), wanted))


@telemetry_bp.route("/api/telemetry/survivors")
@_json_safe
def survivors_data():
    return jsonify(survivors_module.load_library_funnel())


@telemetry_bp.route("/universe")
def universe_page():
    return render_template("universe.html", active_page="universe")
