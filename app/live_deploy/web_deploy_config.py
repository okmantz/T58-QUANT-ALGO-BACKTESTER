"""
Web live-deploy access control (v7, P0-3).

The web ``/deploy-live/*`` routes connect to REAL funded accounts and
place REAL orders. They must never be reachable by anyone who happens to
reach the server's port. This module holds the two decisions:

1. ``T58_ENABLE_WEB_LIVE_DEPLOY`` (env, default OFF): the routes are
   entirely disabled unless the operator explicitly opts in.

2. When enabled, the request must ALSO pass the web app's own
   authentication -- the optional account password lock
   (app.accounts.settings; ``session["t58_unlocked"]``). If no lock
   password is set, enabling the flag alone is NOT enough: the routes
   refuse until the operator sets a password and unlocks, i.e.
   "putting the server behind auth" is a prerequisite, not a suggestion.

Kept Flask-free on purpose so the decision logic is unit-testable
without importing the whole web server; server.py translates the
(allowed, status_code, message) triple into a response.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ENV_FLAG = "T58_ENABLE_WEB_LIVE_DEPLOY"

# Typed confirmation for the in-app switch (see enable_from_app).
ENABLE_PHRASE = "ENABLE LIVE DEPLOY"


def _flag_path() -> Path:
    from app.data.storage import get_app_base_dir
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d / "web_live_deploy.json"


def _persisted_enabled() -> bool:
    try:
        data = json.loads(_flag_path().read_text(encoding="utf-8"))
        return bool(data.get("enabled"))
    except Exception:  # noqa: BLE001 -- missing/corrupt file == not enabled (fail closed)
        return False


def set_persisted_enabled(enabled: bool) -> None:
    """Writes (or clears) the in-app opt-in. Callers MUST have already
    checked the request is allowed to do this -- see enable_from_app."""
    _flag_path().write_text(json.dumps({
        "enabled": bool(enabled),
        "changed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }), encoding="utf-8")


def web_live_deploy_enabled() -> bool:
    """True when the operator opted in -- either via the environment
    variable (the original, still-supported way) or via the in-app switch
    (persisted; only ever set from a loopback request that is already
    password-unlocked -- see enable_from_app)."""
    if os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return _persisted_enabled()


def enable_from_app(*, is_loopback: bool, has_password: bool, unlocked: bool, typed_phrase: str) -> tuple[bool, str]:
    """In-app equivalent of setting the env flag, with the SAME assurances:
    the request must come from this computer itself (not another device on
    the network), an app password must exist, the browser session must be
    unlocked with it, and the operator must type the confirmation phrase.
    Returns (ok, message)."""
    if not is_loopback:
        return False, ("This switch can only be flipped from the computer the app is running on "
                       "(open http://127.0.0.1:5000 there), not from another device.")
    if not has_password:
        return False, "Set an app password first (Settings -> Account) -- live trading is never available without one."
    if not unlocked:
        return False, "Unlock this browser session with your app password first."
    if (typed_phrase or "").strip() != ENABLE_PHRASE:
        return False, f"Type {ENABLE_PHRASE} exactly, in capitals, to confirm."
    set_persisted_enabled(True)
    return True, "Web live deployment is now enabled on this computer."


def setup_checklist(has_password: bool, unlocked: bool) -> list[dict]:
    """The same three gates check_web_live_deploy enforces, as a checklist
    the locked page can render (so the person sees exactly what is missing
    instead of a raw error)."""
    enabled = web_live_deploy_enabled()
    return [
        {"key": "password", "ok": bool(has_password), "title": "Set an app password",
         "detail": "Settings -> Account. Live trading is never available without the password lock.",
         "href": "/settings/account", "label": "Open Account settings"},
        {"key": "unlock", "ok": bool(has_password and unlocked), "title": "Unlock this browser session",
         "detail": "Enter your app password once in this browser.", "href": "/lock?next=/deploy-live",
         "label": "Unlock"},
        {"key": "enable", "ok": enabled, "title": "Turn on live deployment",
         "detail": f"Either start the app with {ENV_FLAG}=1, or use the switch below (type {ENABLE_PHRASE}).",
         "href": "", "label": ""},
    ]


def check_web_live_deploy(has_password: bool, unlocked: bool) -> tuple[bool, int, str]:
    """Decide whether a /deploy-live/* request may proceed.

    Returns (allowed, http_status, message).
    """
    if not web_live_deploy_enabled():
        return (
            False,
            403,
            "Web live deployment is DISABLED. To enable it, set the "
            f"{ENV_FLAG}=1 environment variable on the server AND set an "
            "app password (Settings -> Account) -- enabling the flag "
            "without the password lock keeps these routes refused.",
        )
    if not has_password or not unlocked:
        return (
            False,
            401,
            "Web live deployment requires authentication: set an app "
            "password (Settings -> Account) and unlock this browser "
            "session before starting, stopping, or killing a live "
            "deployment from the web UI.",
        )
    return True, 200, ""
