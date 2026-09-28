"""
Persistent "Projects" for the floating Project Chat widget.

A Project is a lightweight, named container for a persistent AI
conversation -- the "memory-carrying AI Assist" half of the StarNet-style
feature Owen asked for (see app.ai.project_chat for the Ollama side, and
app.web.project_routes for the web API). Deliberately NOT a strategy
container: it doesn't hold strategy code, datasets, or backtest configs
-- see STRATEGY_FORMAT.md / app.strategy.library for that. A project is
just: a name, a running chat transcript, and (via app.web.job_manager's
project_id auto-tagging) a read-only view of whichever background jobs
were started while this project was the active one.

Storage mirrors app.strategy.library's own convention exactly: one JSON
file per project under the same persistent, frozen-exe-aware app data
root (see app.data.storage.get_app_base_dir), so a project survives
restarts and (in dev, not a packaged .exe -- see library.py's own note
about this) shows up in the git repo:

    <app base dir>/data/projects/<project_id>.json

No delegation logic lives here yet (Phase 3, not built): a project has
no concept of "which job types it's allowed to start" or "spawn a Search
Lab run" -- it is purely the chat + activity-feed data model for now.

Thread-safety: a single process-wide lock serializes every read-modify-
write against a project file, mirroring app.web.job_manager.JobManager's
own reasoning -- these files are small and writes are infrequent (one
per chat turn), so a simple global lock is plenty and never a real
bottleneck.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from app.data.storage import get_app_base_dir

# Keeps a long-lived project's on-disk file (and every prompt built from
# it) bounded -- see app.ai.project_chat, which further trims to only the
# last ~10 turns for the actual Ollama prompt. This cap is just about the
# STORED transcript never growing forever, not prompt size.
MAX_STORED_MESSAGES = 200

_LOCK = threading.Lock()


class ProjectNotFound(Exception):
    """Raised when a project_id doesn't correspond to any saved project."""


def _projects_dir() -> Path:
    d = get_app_base_dir() / "data" / "projects"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_path(project_id: str) -> Path:
    # project_id is always a uuid4 hex string minted by create_project()
    # below, never taken from user input directly -- but guard the
    # filename anyway (same defensive habit as app.strategy.library's
    # safe_filename_stem) rather than trust that invariant forever.
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", project_id)
    return _projects_dir() / f"{safe}.json"


def _now() -> float:
    return time.time()


def _read_project_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_project_file(path: Path, data: dict[str, Any]) -> None:
    # Write-then-rename would be the fully-crash-safe version, but these
    # files are tiny (a name + a capped chat transcript) and writes only
    # happen from within a request holding _LOCK, so a plain write
    # matches every other small-JSON-sidecar in this codebase (e.g.
    # app.ai.ollama_settings's own settings file).
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    """The list_projects() shape -- everything EXCEPT the full chat
    transcript, so the project picker in the widget can list many
    projects cheaply without pulling every message of every one."""
    return {
        "id": data["id"],
        "name": data["name"],
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
        "message_count": len(data.get("chat_history", [])),
    }


def list_projects() -> list[dict[str, Any]]:
    """Every saved project, most-recently-updated first. Never raises --
    a corrupt/unreadable project file is skipped rather than blowing up
    the whole list (matches app.ai.ollama_settings.load_settings's
    fail-safe convention)."""
    out: list[dict[str, Any]] = []
    for path in _projects_dir().glob("*.json"):
        try:
            out.append(_summary(_read_project_file(path)))
        except Exception:
            continue
    out.sort(key=lambda p: p["updated_at"], reverse=True)
    return out


def create_project(name: str) -> dict[str, Any]:
    """Creates and persists a new, empty-history project. Blank names
    fall back to a timestamped default rather than failing -- the
    widget's "+ New Project" button should always succeed."""
    name = (name or "").strip() or f"Project {time.strftime('%Y-%m-%d %H:%M')}"
    project_id = uuid.uuid4().hex
    now = _now()
    data = {
        "id": project_id,
        "name": name,
        "created_at": now,
        "updated_at": now,
        "chat_history": [],
    }
    with _LOCK:
        _write_project_file(_project_path(project_id), data)
    return data


def get_project(project_id: str) -> dict[str, Any]:
    """Full project including its chat_history. Raises ProjectNotFound
    for an unknown id -- callers (the web routes) turn that into a 404
    JSON response rather than a 500."""
    path = _project_path(project_id)
    with _LOCK:
        if not path.exists():
            raise ProjectNotFound(project_id)
        return _read_project_file(path)


def project_exists(project_id: str) -> bool:
    return _project_path(project_id).exists()


def rename_project(project_id: str, name: str) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise ValueError("Project name can't be empty.")
    with _LOCK:
        path = _project_path(project_id)
        if not path.exists():
            raise ProjectNotFound(project_id)
        data = _read_project_file(path)
        data["name"] = name
        data["updated_at"] = _now()
        _write_project_file(path, data)
        return data


def delete_project(project_id: str) -> None:
    path = _project_path(project_id)
    with _LOCK:
        if not path.exists():
            raise ProjectNotFound(project_id)
        path.unlink()


def append_chat_message(project_id: str, role: str, content: str) -> dict[str, Any]:
    """Appends one {role, content, ts} turn and persists it, trimming the
    STORED transcript to the most recent MAX_STORED_MESSAGES (oldest
    dropped first) so a project chatted with daily for months doesn't
    grow its JSON file without bound. Returns the full updated project.
    role is whatever the caller passes ("user"/"assistant") -- this
    module doesn't validate it, matching Ollama's own /api/chat message
    shape exactly so app.ai.project_chat can pass history straight
    through with no reshaping."""
    if role not in ("user", "assistant"):
        raise ValueError(f"Unknown chat role: {role!r}")
    with _LOCK:
        path = _project_path(project_id)
        if not path.exists():
            raise ProjectNotFound(project_id)
        data = _read_project_file(path)
        history = data.get("chat_history", [])
        history.append({"role": role, "content": content, "ts": _now()})
        if len(history) > MAX_STORED_MESSAGES:
            history = history[-MAX_STORED_MESSAGES:]
        data["chat_history"] = history
        data["updated_at"] = _now()
        _write_project_file(path, data)
        return data


def get_chat_history(project_id: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Convenience read-only accessor -- the most recent `limit` turns
    (or all of them), oldest first, exactly as get_project()['chat_history']
    already returns them."""
    data = get_project(project_id)
    history = data.get("chat_history", [])
    return history[-limit:] if limit else history
