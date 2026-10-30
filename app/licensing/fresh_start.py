"""Clean slate for the owner (master key).

EVERY time the master key is entered (and the first time it is found saved on a device), the app starts like a brand-new install: saved runs, achievements,
look-and-feel choices, journal, AI memory and the account vault copy are MOVED (never deleted) into a dated backup folder.
After that a marker file is written and everything saves normally until the master key is entered again.

The strategy library goes back to the strategies committed to the repo. Kept on purpose: data/raw (price files), data/config (API keys / broker settings) and the license.
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


def run_if_needed(force: bool = False) -> bool:
    """Run the clean slate if it hasn't run on this device (or always, with force=True -- used every time the master key is entered).
    Never raises. Returns True when it ran now."""
    try:
        import os
        # Never during tests or when explicitly disabled (T58_NO_FRESH_START=1).
        if os.environ.get("T58_NO_FRESH_START") == "1":
            return False
        if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get("T58_FORCE_FRESH_START") != "1":
            return False
        if not force and not needed():
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
        try:    # back to the 5 default strategies; everything else is parked in the same backup folder
            from app.strategy import default_library
            moved += default_library.reset_to_defaults(backup / "strategies")
            default_library.forget_account()
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


def clear_for_account_switch() -> int:
    """A DIFFERENT account is signing in on this device: park the previous account's runs, settings and achievements in a dated
    backup folder (their own vault copy keeps them for their next sign-in) so the new account starts from the defaults.
    Strategies are handled separately by app.strategy.default_library. Never raises; returns files/folders moved."""
    try:
        import os
        if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get("T58_FORCE_FRESH_START") != "1":
            return 0
        from app.data.storage import get_app_base_dir, user_data_dir
        base = get_app_base_dir()
        backup = user_data_dir() / f"account_switch_backup_{time.strftime('%Y%m%d-%H%M%S')}"
        moved = 0
        for rel in _MOVE_FILES:
            moved += _move(base / rel, backup, base)
        for rel in _MOVE_DIRS:
            p = base / rel
            if not p.is_dir():
                continue
            if rel == "reports":
                for child in list(p.iterdir()):
                    moved += _move(child, backup, base)
            else:
                moved += _move(p, backup, base)
        # the browser clears its own saved look-and-feel once when the epoch changes
        user_data_dir().mkdir(parents=True, exist_ok=True)
        _marker().write_text(json.dumps({"epoch": uuid.uuid4().hex[:12], "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                         "backup": str(backup), "moved": moved, "reason": "account switch"}), encoding="utf-8")
        return moved
    except Exception:  # noqa: BLE001
        return 0
