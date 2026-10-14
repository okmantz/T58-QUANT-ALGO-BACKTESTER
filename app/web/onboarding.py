"""
First-run onboarding state for the web app.

The guided tour used to decide "have I shown this?" from browser storage
alone. That is fragile: the launcher opens the app at 127.0.0.1 while a phone
uses the LAN address (two origins = two separate storages), a reinstall keeps
the old browser flag, and some desktop WebViews block storage entirely. So the
SERVER now owns the flag -- a tiny JSON file next to the license state in
data/config/ -- and the dashboard tells the tour script whether to auto-start.

Lifecycle: nothing recorded yet -> the first Dashboard visit after the license
is activated auto-starts the tour; finishing, skipping or closing it records
"seen" and it never auto-starts again. "Take the 60-second tour" on the
Dashboard always replays it.

Every function is best-effort: a read/write failure never blocks the app, and
an unreadable flag means "don't auto-start" rather than nagging on every load.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.data.storage import get_app_base_dir

FLAG_NAME = "web_onboarding.json"


def _flag_path() -> Path:
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d / FLAG_NAME


def tour_seen() -> bool:
    """True once the first-run tour was finished or dismissed on this install."""
    path = _flag_path()
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- corrupt flag: treat as seen, never nag
        return True
    return bool(data.get("tour_seen"))


def should_autostart_tour() -> bool:
    return not tour_seen()


def mark_tour_seen(how: str = "done") -> None:
    try:
        _flag_path().write_text(
            json.dumps({"tour_seen": True, "how": str(how)[:32],
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass


def reset_tour() -> None:
    try:
        _flag_path().unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
