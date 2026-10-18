"""
Subscription / license info -- locally stored fields for the Account tab's
Subscription section (Plan, License Key, License Status, Renewal/Expiration
Date).

This app has NO license server of its own today (see app.accounts.settings'
own docstring on the same point for auth) -- there is nothing here to
"activate" against over the network. What this module gives the Account
tab is a place to RECORD those four fields locally (e.g. once T58 Trading
stands up a real licensing backend, or for tracking a manually-issued key)
and a best-effort, honest-about-its-limits FORMAT check on a pasted key --
never a real verification, and the UI must not claim otherwise.

Wiring this to a real license server later is a matter of replacing
`check_license_key_format` with an actual HTTP call and leaving everything
else (storage shape, the four fields, the template) unchanged.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.data.storage import get_app_base_dir

SUBSCRIPTION_FILENAME = "subscription.json"

# The plan every signed-in (licensed) user is on. The Account tab shows this
# as a dropdown instead of a free-text box (2026-10-17), so nobody has to type
# their plan name and it can never be mistyped or left blank.
PREMIUM_PLAN = "Premium Subscription"
PLAN_CHOICES = (PREMIUM_PLAN,)

LICENSE_STATUS_CHOICES = ("unset", "active", "trial", "expired", "cancelled")
# Statuses a person (or a future license server) set on purpose -- these are
# honoured as-is. Anything else ("unset", "trial", blank) reads as Active once
# the person is signed in, because reaching the Account tab at all means the
# license gate already let them in.
_EXPLICIT_INACTIVE_STATUSES = ("expired", "cancelled")

# XXXX-XXXX-XXXX-XXXX, letters/digits only -- a reasonable generic shape;
# tighten this once a real key format exists.
_KEY_FORMAT_RE = re.compile(r"^[A-Z0-9]{4}(-[A-Z0-9]{4}){3}$")


@dataclass
class SubscriptionInfo:
    plan: str = ""
    license_key: str = ""
    license_status: str = "unset"
    renewal_date: str = ""  # free-text (e.g. "2027-01-15") -- no server to validate this against yet


def _subscription_path() -> Path:
    return get_app_base_dir() / SUBSCRIPTION_FILENAME


def load_subscription() -> SubscriptionInfo:
    path = _subscription_path()
    if not path.exists():
        return SubscriptionInfo()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        status = str(data.get("license_status", "unset"))
        if status not in LICENSE_STATUS_CHOICES:
            status = "unset"
        return SubscriptionInfo(
            plan=str(data.get("plan", "")),
            license_key=str(data.get("license_key", "")),
            license_status=status,
            renewal_date=str(data.get("renewal_date", "")),
        )
    except Exception:  # noqa: BLE001 -- a corrupt/unreadable file must never crash the Account tab
        return SubscriptionInfo()


def effective_subscription(info: SubscriptionInfo | None = None) -> SubscriptionInfo:
    """What the Account tab shows: the stored record with the plan normalised to
    "Premium Subscription" and the status shown as "active" once the person is
    signed in (unless an expired/cancelled status was recorded on purpose).
    Pure -- never writes anything."""
    info = info if info is not None else load_subscription()
    status = info.license_status if info.license_status in _EXPLICIT_INACTIVE_STATUSES else "active"
    return SubscriptionInfo(
        plan=PREMIUM_PLAN,
        license_key=info.license_key,
        license_status=status,
        renewal_date=info.renewal_date,
    )


def save_subscription(info: SubscriptionInfo) -> None:
    path = _subscription_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")


def check_license_key_format(license_key: str) -> tuple[bool, str]:
    """A SHAPE check only ("does this look like a key"), never a real
    verification -- there is no license server to ask. Returns
    (looks_valid, message)."""
    key = (license_key or "").strip().upper()
    if not key:
        return False, "No license key entered."
    if _KEY_FORMAT_RE.match(key):
        return True, "Key format looks valid (format check only -- not verified against a license server)."
    return False, "Key doesn't match the expected XXXX-XXXX-XXXX-XXXX format."
