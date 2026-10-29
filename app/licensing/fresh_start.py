"""One-time clean slate for the owner (master key).

The first time the master key signs in on a device, the app starts like a brand-new install: saved runs, achievements,
look-and-feel choices, journal, AI memory and the account vault copy are MOVED (never deleted) into a dated backup folder.
After that a marker file is written and everything saves normally -- this never runs again on that device.

Kept on purpose: strategies (your code), data/raw (price files), data/config (API keys / broker settings) and the license.
Delete `fresh_start_done.json` in the user-data folder to get another clean start.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

MARKER_NAME = "fresh_start_done.json"

# Relative to the app base dir. Everything here is moved to the backup folder.
_MOVE_DIRS = (
    "reports", "data/evolution", "data/discovery", "data/discovery_engine", "data/journal", "data/ai_memory",
    "data/projects", "data/forward_test",
)
_MOVE_FILES = (
    "account_settings.json", "data/run_history.json", "data/achievements.json", "data/briefing.json",
    "data/ui_prefs.json", "data/inbox.json", "data/library/paper_state.json", "data/library/focus_sessions.json",
)


def _marker() -> Path:
    from app.data.storage import user_data_dir
    return user_data_dir() / MARKER_NAME


def epoch() -> str:
    """Changes only when a fresh start ran; the web page uses it to clear its own saved browser settings once."""
    try:
        return str(json.loads(_marker().read_text(encoding="utf-8")).get("epoch", ""))
    except Exception:  # noqa: BLE001
        return ""


def needed() -> bool:
    return not _marker().exists()


def _move(src: Path, dest_root: Path, base: Path) -> int:
    if not src.exists():
        return 0
    dest = dest_root / src.relative_to(base)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return 1


def run_if_needed() -> bool:
    """Run the clean slate if it hasn't run on this device. Never raises. Returns True when it ran now."""
    try:
        import os
        # Never during tests or when explicitly disabled (T58_NO_FRESH_START=1).
        if os.environ.get("T58_NO_FRESH_START") == "1":
            return False
        if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get("T58_FORCE_FRESH_START") != "1":
            return False
        if not needed():
            return False
        from app.data.storage import get_app_base_dir, user_data_dir
        base = get_app_base_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = user_data_dir() / f"fresh_start_backup_{stamp}"
        moved = 0
        for rel in _MOVE_FILES:
            moved += _move(base / rel, backup, base)
        for rel in _MOVE_DIRS:
            p = base / rel
            if not p.is_dir():
                continue
            if rel == "reports":
                # keep the folder itself (the app expects it) and move its contents
                for child in list(p.iterdir()):
                    moved += _move(child, backup, base)
            else:
                moved += _move(p, backup, base)
        # The account vault would silently restore everything we just moved on the next sign-in.
        try:
            from app.persistence import vault
            for acct_id in vault.candidate_ids():
                acct = vault.vault_root() / acct_id
                if acct.exists():
                    dest = backup / "vault" / acct.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(acct), str(dest))
        except Exception:  # noqa: BLE001
            pass
        user_data_dir().mkdir(parents=True, exist_ok=True)
        _marker().write_text(json.dumps({
            "epoch": uuid.uuid4().hex[:12], "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backup": str(backup), "moved": moved,
        }), encoding="utf-8")
        return True
    except Exception:  # noqa: BLE001
        return False
