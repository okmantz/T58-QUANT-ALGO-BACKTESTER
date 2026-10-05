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

import os

ENV_FLAG = "T58_ENABLE_WEB_LIVE_DEPLOY"


def web_live_deploy_enabled() -> bool:
    """True only when the operator explicitly opted in via env."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


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
