"""Server-side copy of the browser preferences (background, theme, accent colour, verse bar, playground scores, ...).

WHY: those choices used to live ONLY in the browser's localStorage, which belongs to one origin (host + port) of one browser
profile. A reinstall, a different port (5000 is often busy, so the launcher falls back to another) or a different browser all
started from a blank slate -- "nothing saved". The page now mirrors every `t58-*` localStorage key to this small JSON file on
the server (see /static/t58-prefs-sync.js), and the file is part of the account vault (app.persistence.vault), so it follows
the license key through a reinstall too.

Only keys starting with `t58-` are kept, values must be short strings, and the file is capped, so a buggy page can't bloat it.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

PREFIX = "t58-"
MAX_VALUE_CHARS = 200_000
MAX_KEYS = 400
_LOCK = threading.Lock()


def _path() -> Path:
    from app.data.storage import get_app_base_dir
    return get_app_base_dir() / "data" / "ui_prefs.json"


def valid_key(key: object) -> bool:
    return isinstance(key, str) and key.startswith(PREFIX) and 4 < len(key) <= 80 and all(c.isalnum() or c in "-_." for c in key)


def load() -> dict[str, str]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if valid_key(k) and isinstance(v, str)}


def update(changes: dict) -> dict[str, str]:
    """Apply {key: value-or-None}; None removes the key. Returns the full saved mapping."""
    with _LOCK:
        cur = load()
        for k, v in (changes or {}).items():
            if not valid_key(k):
                continue
            if v is None:
                cur.pop(k, None)
            elif isinstance(v, str) and len(v) <= MAX_VALUE_CHARS:
                cur[k] = v
        if len(cur) > MAX_KEYS:
            cur = dict(list(cur.items())[-MAX_KEYS:])
        p = _path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(cur, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, p)
        except OSError:
            pass
        return cur
