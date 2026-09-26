"""
T58 License Server -- a small, self-hosted Flask app that is the ONLY
thing standing between a copied ZIP of the T58 Quant Algo Backtester and
someone actually being able to use it. Run this yourself (a free-tier
Render/Fly.io/PythonAnywhere/Railway instance, or any always-on machine
you already have) -- no third-party licensing service, no per-seat fees
to anyone else. See README.md in this folder for setup.

Endpoints
---------
Client-facing (called by the desktop app -- app/licensing/client.py):
    POST /activate    {email, license_key, device_id} -> binds this
                       license to this device on first use.
    POST /validate     {email, license_key, device_id} -> checked
                       periodically while the app runs.
    POST /deactivate   {email, license_key, device_id} -> frees the
                       device binding so the license can move to a new
                       machine ("logout" / "deactivate this device").

Admin-only (require header X-Admin-Token: <ADMIN_TOKEN env var>):
    POST /admin/licenses                    create a new license
    GET  /admin/licenses                    list all licenses
    GET  /admin/licenses/<key>              one license's full record
    POST /admin/licenses/<key>/revoke       status -> revoked
    POST /admin/licenses/<key>/suspend      status -> suspended
    POST /admin/licenses/<key>/reactivate   status -> active
    POST /admin/licenses/<key>/extend       {days} -> pushes expires_at out

Whop subscription webhook (see README.md for how to point Whop at this):
    POST /webhook/whop   Verified with the Standard Webhooks scheme Whop
                         actually signs with (HMAC-SHA256 of
                         "{id}.{timestamp}.{raw_body}", header
                         webhook-signature: "v1,<base64>"). On
                         membership.went_invalid, automatically revokes
                         the matching license -- this is what actually
                         answers "if they stop paying, rescind access"
                         without you doing anything by hand. On
                         membership.went_valid for an existing (e.g.
                         previously lapsed) license, reactivates it.
                         A brand-new membership with no matching license
                         yet is logged, not auto-created -- see README.md
                         for why key issuance for a NEW customer still
                         has one manual (or admin_cli.py-scripted) step.
"""
from __future__ import annotations

import base64
import hmac
import hashlib
import os
import time
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, jsonify, request

import db

app = Flask(__name__)
db.init_db()

ADMIN_TOKEN = os.environ.get("T58_LICENSE_ADMIN_TOKEN", "")
WHOP_WEBHOOK_SECRET = os.environ.get("T58_WHOP_WEBHOOK_SECRET", "")  # the "whsec_..." value from your Whop webhook settings
WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS = 5 * 60  # matches the Standard Webhooks spec's own recommended replay window


def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not ADMIN_TOKEN:
            return jsonify({"ok": False, "error": "Server misconfigured: T58_LICENSE_ADMIN_TOKEN is not set."}), 500
        provided = request.headers.get("X-Admin-Token", "")
        if not hmac.compare_digest(provided, ADMIN_TOKEN):
            return jsonify({"ok": False, "error": "Invalid or missing X-Admin-Token header."}), 401
        return fn(*args, **kwargs)
    return wrapper


def _license_public_view(lic: dict) -> dict:
    """Never echo device_id back in a client-facing (non-admin) response
    -- it's a device fingerprint, not something the client needs told to
    it, and there's no reason to give a would-be forger confirmation of
    what value would need to match."""
    return {
        "status": lic["status"],
        "email": lic["email"],
        "plan": lic["plan"],
        "expires_at": lic["expires_at"],
    }


def _check_license(email: str, license_key: str, device_id: str, allow_bind: bool):
    """Shared logic for /activate and /validate. Returns (http_status,
    response_dict). Never raises -- every failure path is a clean,
    specific error the client can show the person directly."""
    if not email or not license_key or not device_id:
        return 400, {"ok": False, "error": "email, license_key, and device_id are all required."}

    # SHARED TRIAL KEYS (2026-09-26): a trial_key is not tied to any one
    # email at creation, so it can never be found in the `licenses` table
    # -- check it FIRST and hand off to the dedicated trial flow (see
    # db.redeem_trial) rather than falling through to "not_found" below.
    # This means the existing app/client code -- which already only
    # knows how to POST email/license_key/device_id to /activate and
    # /validate and read back {ok, status, expires_at} -- needs zero
    # changes to support a trial key; it's just another string someone
    # can type into the same "license key" field.
    trial_key = db.get_trial_key(license_key.strip().upper())
    if trial_key is not None:
        return _check_trial(trial_key, email, device_id)

    lic = db.get_license(license_key.strip().upper())
    if lic is None:
        return 404, {"ok": False, "error": "not_found"}

    lic = db.mark_expired_if_past_due(lic)

    if lic["email"].lower() != email.strip().lower():
        return 403, {"ok": False, "error": "email_mismatch"}

    if lic["status"] != "active":
        return 403, {"ok": False, "error": lic["status"], **_license_public_view(lic)}

    if lic["device_id"] is None:
        if allow_bind:
            db.bind_device(lic["license_key"], device_id)
            lic["device_id"] = device_id
        # else: unbound but this call site doesn't bind -- treat as fine,
        # the next /activate call will bind it.
    elif lic["device_id"] != device_id:
        return 403, {"ok": False, "error": "device_mismatch"}

    db.touch_validated(lic["license_key"])
    return 200, {"ok": True, **_license_public_view(lic)}


def _check_trial(trial_key: dict, email: str, device_id: str):
    """The trial-key half of _check_license -- see db.redeem_trial for
    the actual per-person-once-per-3-days rule. Response shape matches
    _license_public_view exactly (status/email/plan/expires_at) so the
    client can't tell a trial redemption apart from a normal license
    response and needs no special-case handling for it."""
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    try:
        redemption = db.redeem_trial(trial_key["trial_key"], email, device_id, ip_address=ip_address)
    except db.TrialConflict as exc:
        reason = "This trial key has already been used on a different device." if exc.reason == "device" \
            else "This trial key has already been used with a different email address."
        return 403, {"ok": False, "error": "trial_already_used", "message": reason}
    except ValueError as exc:
        return 403, {"ok": False, "error": "trial_invalid", "message": str(exc)}

    expires = datetime.fromisoformat(redemption["expires_at"])
    if expires <= datetime.now(timezone.utc):
        return 403, {
            "ok": False, "error": "expired",
            "status": "expired", "email": redemption["email"], "plan": "trial",
            "expires_at": redemption["expires_at"],
        }
    return 200, {
        "ok": True, "status": "active", "email": redemption["email"], "plan": "trial",
        "expires_at": redemption["expires_at"],
    }


@app.route("/activate", methods=["POST"])
def activate():
    data = request.get_json(silent=True) or request.form
    status, body = _check_license(
        data.get("email", ""), data.get("license_key", ""), data.get("device_id", ""), allow_bind=True,
    )
    return jsonify(body), status


@app.route("/validate", methods=["POST"])
def validate():
    data = request.get_json(silent=True) or request.form
    status, body = _check_license(
        data.get("email", ""), data.get("license_key", ""), data.get("device_id", ""), allow_bind=True,
    )
    return jsonify(body), status


@app.route("/deactivate", methods=["POST"])
def deactivate():
    """Frees the device binding -- lets the same license be activated on
    a different machine afterward. Requires knowing the license's own
    current device_id (sent back by the client, which stored it locally
    at activation time) so this can't be used to bump someone else's
    active session without also knowing that value."""
    data = request.get_json(silent=True) or request.form
    license_key = (data.get("license_key") or "").strip().upper()
    device_id = data.get("device_id", "")
    lic = db.get_license(license_key)
    if lic is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    if lic["device_id"] and lic["device_id"] != device_id:
        return jsonify({"ok": False, "error": "device_mismatch"}), 403
    db.clear_device(license_key)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@app.route("/admin/licenses", methods=["POST"])
@require_admin
def admin_create_license():
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip()
    if not email:
        return jsonify({"ok": False, "error": "email is required."}), 400
    plan = data.get("plan", "")
    days = data.get("days")
    days = int(days) if days not in (None, "") else None
    whop_membership_id = data.get("whop_membership_id") or None
    lic = db.create_license(email, plan=plan, days=days, whop_membership_id=whop_membership_id)
    return jsonify({"ok": True, "license": lic}), 201


@app.route("/admin/licenses", methods=["GET"])
@require_admin
def admin_list_licenses():
    return jsonify({"ok": True, "licenses": db.list_licenses()})


@app.route("/admin/licenses/<license_key>", methods=["GET"])
@require_admin
def admin_get_license(license_key):
    lic = db.get_license(license_key.strip().upper())
    if lic is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify({"ok": True, "license": lic})


# ---------------------------------------------------------------------------
# Admin: shared trial keys (see db.redeem_trial for the actual rule)
# ---------------------------------------------------------------------------

@app.route("/admin/trial-keys", methods=["POST"])
@require_admin
def admin_create_trial_key():
    data = request.get_json(silent=True) or request.form
    trial_key = (data.get("trial_key") or "").strip()
    if not trial_key:
        return jsonify({"ok": False, "error": "trial_key is required."}), 400
    days = int(data.get("days") or 3)
    try:
        tk = db.create_trial_key(trial_key, days=days)
    except Exception as exc:  # sqlite3.IntegrityError on a duplicate key phrase
        return jsonify({"ok": False, "error": f"Could not create trial key: {exc}"}), 400
    return jsonify({"ok": True, "trial_key": tk}), 201


@app.route("/admin/trial-keys", methods=["GET"])
@require_admin
def admin_list_trial_keys():
    return jsonify({"ok": True, "trial_keys": db.list_trial_keys()})


@app.route("/admin/trial-keys/<trial_key>/disable", methods=["POST"])
@require_admin
def admin_disable_trial_key(trial_key):
    tk = db.set_trial_key_status(trial_key, "disabled")
    if tk is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify({"ok": True, "trial_key": tk})


@app.route("/admin/trial-keys/<trial_key>/enable", methods=["POST"])
@require_admin
def admin_enable_trial_key(trial_key):
    tk = db.set_trial_key_status(trial_key, "active")
    if tk is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify({"ok": True, "trial_key": tk})


@app.route("/admin/trial-redemptions", methods=["GET"])
@require_admin
def admin_list_trial_redemptions():
    """Every (device, email) pair that has ever redeemed a trial, newest
    first -- this is Owen's own visibility into potential trial-key
    abuse (e.g. many redemptions from the same ip_address in a short
    window), since device_id/email alone can't be hard-blocked across a
    fresh pairing of both (see db.redeem_trial's docstring)."""
    return jsonify({"ok": True, "redemptions": db.list_trial_redemptions()})


def _admin_set_status_route(license_key, status):
    lic = db.get_license(license_key.strip().upper())
    if lic is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    lic = db.set_status(lic["license_key"], status)
    return jsonify({"ok": True, "license": lic})


@app.route("/admin/licenses/<license_key>/revoke", methods=["POST"])
@require_admin
def admin_revoke(license_key):
    return _admin_set_status_route(license_key, "revoked")


@app.route("/admin/licenses/<license_key>/suspend", methods=["POST"])
@require_admin
def admin_suspend(license_key):
    return _admin_set_status_route(license_key, "suspended")


@app.route("/admin/licenses/<license_key>/reactivate", methods=["POST"])
@require_admin
def admin_reactivate(license_key):
    return _admin_set_status_route(license_key, "active")


@app.route("/admin/licenses/<license_key>/extend", methods=["POST"])
@require_admin
def admin_extend(license_key):
    data = request.get_json(silent=True) or request.form
    days = data.get("days")
    if days in (None, ""):
        return jsonify({"ok": False, "error": "days is required."}), 400
    lic = db.get_license(license_key.strip().upper())
    if lic is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    lic = db.extend_license(lic["license_key"], int(days))
    return jsonify({"ok": True, "license": lic})


# ---------------------------------------------------------------------------
# Whop subscription webhook
# ---------------------------------------------------------------------------

def _verify_standard_webhook(raw_body: bytes, headers) -> bool:
    """Whop signs webhooks with the open Standard Webhooks spec (see
    https://github.com/standard-webhooks/standard-webhooks), NOT a
    Whop-specific scheme -- this is a real, documented signature check,
    not a guess. Signed content is "{id}.{timestamp}.{raw_body}", HMAC-
    SHA256'd with the base64-decoded portion of your whsec_... secret,
    then base64-encoded again; the header carries it as "v1,<that>".
    Also enforces a 5-minute timestamp tolerance against replay, per the
    spec's own recommendation."""
    if not WHOP_WEBHOOK_SECRET:
        return False
    webhook_id = headers.get("webhook-id", "")
    webhook_timestamp = headers.get("webhook-timestamp", "")
    webhook_signature = headers.get("webhook-signature", "")
    if not (webhook_id and webhook_timestamp and webhook_signature):
        return False

    try:
        ts = int(webhook_timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS:
        return False

    secret = WHOP_WEBHOOK_SECRET
    if secret.startswith("whsec_"):
        secret = secret[len("whsec_"):]
    try:
        secret_bytes = base64.b64decode(secret)
    except Exception:  # noqa: BLE001
        secret_bytes = secret.encode("utf-8")

    signed_content = f"{webhook_id}.{webhook_timestamp}.".encode("utf-8") + raw_body
    expected = base64.b64encode(hmac.new(secret_bytes, signed_content, hashlib.sha256).digest()).decode("ascii")

    # webhook-signature can list multiple space-separated "v1,<sig>" values
    # (e.g. during secret rotation) -- a match against any of them is valid.
    for candidate in webhook_signature.split():
        _version, _, sig = candidate.partition(",")
        if sig and hmac.compare_digest(sig, expected):
            return True
    return False


def _extract_whop_identifiers(payload: dict) -> tuple[str | None, str | None]:
    """Whop's own field naming for these two identifiers has shifted
    across API versions/event shapes, so this tries a few plausible
    locations rather than betting on exactly one -- returns
    (whop_membership_id, email), either of which may be None."""
    data = payload.get("data", payload)
    membership_id = (
        data.get("membership_id") or data.get("id")
        or (data.get("membership") or {}).get("id")
    )
    email = (
        data.get("email")
        or (data.get("user") or {}).get("email")
        or (data.get("member") or {}).get("email")
    )
    return membership_id, email


@app.route("/webhook/whop", methods=["POST"])
def webhook_whop():
    raw_body = request.get_data()
    if not _verify_standard_webhook(raw_body, request.headers):
        return jsonify({"ok": False, "error": "invalid_signature"}), 401

    payload = request.get_json(silent=True) or {}
    event_type = payload.get("type") or payload.get("action") or ""
    membership_id, email = _extract_whop_identifiers(payload)

    lic = None
    if membership_id:
        lic = db.find_by_whop_membership_id(membership_id)
    if lic is None and email:
        matches = db.find_by_email(email)
        lic = matches[0] if matches else None

    if event_type.endswith("went_invalid"):
        if lic is not None:
            db.set_status(lic["license_key"], "revoked")
        # No matching license is not an error -- e.g. a membership that
        # was never activated in T58 at all. Always 200 either way so
        # Whop doesn't retry (see this file's module docstring).
        return jsonify({"ok": True, "matched": lic is not None})

    if event_type.endswith("went_valid"):
        if lic is not None and lic["status"] in ("revoked", "expired", "suspended"):
            db.set_status(lic["license_key"], "active")
        # A brand-new membership with no existing license is logged for
        # the admin to issue a key for (via admin_cli.py) -- see
        # README.md's "New customer flow" section for why this one step
        # stays manual rather than guessing at auto-delivery.
        if lic is None:
            app.logger.info(
                "Whop membership.went_valid with no matching T58 license yet "
                "(membership_id=%s, email=%s) -- issue one with admin_cli.py.",
                membership_id, email,
            )
        return jsonify({"ok": True, "matched": lic is not None})

    return jsonify({"ok": True, "ignored_event": event_type})


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    app.run(host="0.0.0.0", port=port)
