"""
Projects -- web API for the floating "Project Chat" widget.

Phase 1 (persistent, memory-carrying AI chat) + Phase 2 (wiring
app.web.job_manager.JOB_MANAGER jobs to a project + a read-only activity
feed) of the StarNet-style feature Owen asked for. No delegation yet --
see app.ai.project_chat's module docstring for what's explicitly still
future work (Phase 3).

Registered as a Flask Blueprint, same pattern as app/web/quant_lab_routes.py
/ app/web/risk_sweep_routes.py etc: this whole feature area lives in one
file, and app/web/server.py's own change is a single import + one
`app.register_blueprint(project_bp)` line.

"Active project" is per-browser-session (Flask's signed session cookie,
already configured in server.py via app.secret_key) rather than a single
global -- so multiple people/tabs hitting the same server (e.g. Owen on
his phone and his desktop at once) each get their own idea of which
project is "active" without one clobbering the other's job tagging.

Job-tagging mechanism: on import, this module registers itself with
JOB_MANAGER via set_active_project_getter(), pointing at
_get_active_project_id() below. From that point on, EVERY job started
anywhere in the app (Search Lab, WFO, CPCV, Quick Optimize, ...) via
JOB_MANAGER.create() gets auto-tagged with whatever project is active
for that request's session -- see app.web.job_manager.JobManager.create's
own docstring. No other file needed to change for this to work.
"""
from __future__ import annotations

import functools
import logging
import traceback

from flask import Blueprint, jsonify, request, session

from app.ai.ollama_settings import load_settings as load_ollama_settings
from app.ai.project_chat import ProjectChatClient
from app.orchestration import projects
from app.web.job_manager import JOB_MANAGER

project_bp = Blueprint("projects", __name__, url_prefix="/api/projects")

_logger = logging.getLogger(__name__)

_SESSION_KEY = "active_project_id"

# How many of a project's most recent jobs the Activity tab shows, and how
# many of those get summarized into the chat model's own context (a
# shorter subset -- the full list is for the human to look at, only the
# highlights are worth spending prompt tokens on).
ACTIVITY_FEED_LIMIT = 50
ACTIVITY_CONTEXT_LIMIT = 8

# Fields on a job dict that are safe to hand straight to jsonify (plain
# JSON-serializable types only). A job's "result" field in particular can
# hold arbitrary domain dataclasses/objects that aren't JSON-safe (every
# job type stores whatever its own report layer produced there) -- the
# Activity feed only needs to show status, not replay the full report, so
# unknown/non-primitive fields are simply omitted rather than risking a
# 500 on jsonify(). The tool-specific status/report pages already exist
# for anyone who wants the full result.
_JSON_SAFE_TYPES = (str, int, float, bool, type(None))


def _json_safe(view):
    """Same guarantee app.web.ai_assistant_routes._json_safe already
    established for that blueprint: ANY unhandled exception in a route
    here becomes a normal 200 JSON body with the real error message,
    never Flask's default HTML error page -- so the widget's
    `(await fetch(...)).json()` calls never throw "Unexpected token '<'"
    on a bug here. Duplicated rather than imported to avoid a
    cross-blueprint import, matching how every other blueprint in this
    app already keeps its own copy of small helpers like this."""

    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        try:
            return view(*args, **kwargs)
        except projects.ProjectNotFound:
            return jsonify({"error": "Project not found.", "reply": "", "projects": [], "jobs": []}), 404
        except Exception as exc:  # noqa: BLE001 -- last line of defense, see docstring
            _logger.exception("Project route %s failed", view.__name__)
            return jsonify({
                "error": f"{type(exc).__name__}: {exc}",
                "reply": "", "projects": [], "jobs": [],
                "traceback": traceback.format_exc(limit=6),
            }), 500

    return wrapped


def _get_active_project_id() -> str | None:
    """The getter registered with JOB_MANAGER (see module docstring).
    Reads from Flask's session, which only exists inside a request
    context -- wrapped in a try/except by job_manager.create() itself,
    so this never needs to guard against being called outside one."""
    return session.get(_SESSION_KEY)


JOB_MANAGER.set_active_project_getter(_get_active_project_id)


def _job_summary(job: dict) -> dict:
    """Strips a raw JOB_MANAGER job dict down to JSON-safe fields for the
    Activity feed (see _JSON_SAFE_TYPES above), and adds a couple of
    small display conveniences (elapsed_seconds, a short log tail)."""
    import time as _time

    safe = {k: v for k, v in job.items() if k not in ("result", "log") and isinstance(v, _JSON_SAFE_TYPES)}
    log = job.get("log") or []
    safe["log_tail"] = [line for line in log[-5:] if isinstance(line, str)]
    started_at = job.get("started_at")
    if isinstance(started_at, (int, float)):
        safe["elapsed_seconds"] = round(_time.time() - started_at, 1)
    safe["has_result"] = job.get("result") is not None
    return safe


def _activity_summary_lines(project_id: str) -> list[str]:
    """Deterministic, non-AI plain-text lines describing a project's most
    recent jobs -- passed to app.ai.project_chat as grounding context,
    same principle as app.ai.ollama_client's failure_analysis_lines /
    research_excerpts: the SUMMARIZING is free, only the resulting short
    text costs prompt tokens."""
    jobs = JOB_MANAGER.list_jobs(project_id=project_id, limit=ACTIVITY_CONTEXT_LIMIT)
    lines = []
    for job in jobs:
        status = "done" if job.get("done") else "running"
        if job.get("error"):
            status = f"failed ({job['error']})"
        elif job.get("cancelled"):
            status = "cancelled"
        label = job.get("instrument") or job.get("tool") or job["job_id"]
        lines.append(f"Job {job['job_id']} ({label}): {status}")
    return lines


@project_bp.route("", methods=["GET"])
@_json_safe
def list_projects_route():
    return jsonify({"projects": projects.list_projects(), "active_project_id": _get_active_project_id()})


@project_bp.route("", methods=["POST"])
@_json_safe
def create_project_route():
    data = request.get_json(force=True, silent=True) or {}
    project = projects.create_project(data.get("name", ""))
    session[_SESSION_KEY] = project["id"]  # a newly created project becomes the active one immediately
    return jsonify({"project": project})


@project_bp.route("/active", methods=["GET"])
@_json_safe
def get_active_project_route():
    """Lets the widget restore state on page load/reload: which project
    (if any) is active for this browser session, plus its full data so
    the widget doesn't need a second round trip."""
    active_id = _get_active_project_id()
    if not active_id or not projects.project_exists(active_id):
        return jsonify({"project": None})
    return jsonify({"project": projects.get_project(active_id)})


@project_bp.route("/deactivate", methods=["POST"])
@_json_safe
def deactivate_project_route():
    session.pop(_SESSION_KEY, None)
    return jsonify({"ok": True})


@project_bp.route("/<project_id>", methods=["GET"])
@_json_safe
def get_project_route(project_id: str):
    return jsonify({"project": projects.get_project(project_id)})


@project_bp.route("/<project_id>", methods=["PATCH"])
@_json_safe
def rename_project_route(project_id: str):
    data = request.get_json(force=True, silent=True) or {}
    project = projects.rename_project(project_id, data.get("name", ""))
    return jsonify({"project": project})


@project_bp.route("/<project_id>", methods=["DELETE"])
@_json_safe
def delete_project_route(project_id: str):
    projects.delete_project(project_id)
    if _get_active_project_id() == project_id:
        session.pop(_SESSION_KEY, None)
    return jsonify({"ok": True})


@project_bp.route("/<project_id>/activate", methods=["POST"])
@_json_safe
def activate_project_route(project_id: str):
    if not projects.project_exists(project_id):
        raise projects.ProjectNotFound(project_id)
    session[_SESSION_KEY] = project_id
    return jsonify({"ok": True})


@project_bp.route("/<project_id>/chat", methods=["POST"])
@_json_safe
def chat_route(project_id: str):
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "", "error": "Empty message."}), 400

    project = projects.get_project(project_id)  # raises ProjectNotFound -> 404, via _json_safe
    projects.append_chat_message(project_id, "user", message)

    activity_lines = _activity_summary_lines(project_id)
    client = ProjectChatClient(load_ollama_settings())
    # Send the history as it stood BEFORE this turn's user message (the
    # message itself is appended separately by chat()/_build_messages),
    # so it isn't duplicated in the prompt.
    reply, error = client.chat(
        project["name"], message, history=project.get("chat_history", []), activity_lines=activity_lines,
    )
    if error:
        return jsonify({"reply": "", "error": error})

    updated = projects.append_chat_message(project_id, "assistant", reply)
    return jsonify({"reply": reply, "error": None, "project": updated})


@project_bp.route("/<project_id>/activity", methods=["GET"])
@_json_safe
def activity_route(project_id: str):
    if not projects.project_exists(project_id):
        raise projects.ProjectNotFound(project_id)
    jobs = JOB_MANAGER.list_jobs(project_id=project_id, limit=ACTIVITY_FEED_LIMIT)
    return jsonify({"jobs": [_job_summary(j) for j in jobs]})
