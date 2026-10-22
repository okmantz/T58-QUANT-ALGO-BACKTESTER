"""
Subscription info for the Account tab -- Plan, License Key, License Status and
Renewal/Expiration Date.

Everything here is DERIVED from the license saved on this device (see
app.licensing.client); none of it is typed in by the person and none of it is
stored separately. That is the point: the plan a customer sees on the Account
page is the PLAN column of your license sheet for the key they activated with
(Pro Monthly / Pro Annual / Lifetime / Trial / Complimentary), delivered in the
sheet's activate/validate answer and refreshed at every check-in. The master key
is always Lifetime. A customer can't change any of it from the UI -- the plan
dropdown is locked and the server never reads a plan from the form.

The older free-text subscription.json record this module used to read and write
is gone; a leftover subscription.json on disk is simply ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# The five plans of the license sheet (license_sheet/Code.gs PLANS). Re-exported
# from the licensing client so there is one list.
from app.licensing.client import KNOWN_PLANS as PLAN_CHOICES

# Shown only when a license is active but carries no plan at all (allowlist mode,
# or a sheet whose Apps Script has not been redeployed with plan support yet).
PREMIUM_PLAN = "Premium Subscription"

LICENSE_STATUS_CHOICES = ("unset", "active", "trial", "expired", "cancelled", "revoked", "suspended")
# Statuses the license server/sheet reported on purpose -- shown as-is. Anything
# else reads as Active, because reaching the Account tab means the license gate
# already let this device in.
_EXPLICIT_INACTIVE_STATUSES = ("expired", "cancelled", "revoked", "suspended")

NO_EXPIRY_TEXT = "No expiry"


@dataclass
class SubscriptionInfo:
    plan: str = ""
    license_key: str = ""
    license_status: str = "unset"
    renewal_date: str = ""  # YYYY-MM-DD, or NO_EXPIRY_TEXT when the license never expires


def _date_only(iso: str | None) -> str:
    """'2027-01-15T00:00:00.000Z' -> '2027-01-15'. Unparseable text is shown as-is."""
    text = (iso or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text[:10]


def effective_subscription(state=None) -> SubscriptionInfo:
    """What the Account tab shows, built from the saved license state (or the
    `state` passed in -- anything with the LicenseState fields). Pure: never
    writes anything and never touches the network."""
    from app.licensing import client

    state = state if state is not None else client.load_state()
    key = (getattr(state, "license_key", "") or "").strip()
    if not key:
        return SubscriptionInfo(plan="", license_key="", license_status="unset", renewal_date="")

    plan = client.current_plan(state) or PREMIUM_PLAN
    reported = (getattr(state, "status", "") or "").strip().lower()
    status = reported if reported in _EXPLICIT_INACTIVE_STATUSES else "active"
    renewal = _date_only(getattr(state, "expires_at", None)) or NO_EXPIRY_TEXT
    return SubscriptionInfo(plan=plan, license_key=key, license_status=status, renewal_date=renewal)
