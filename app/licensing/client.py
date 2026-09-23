"""
T58 Licensing Client -- the ONLY part of the app that talks to the
license server (see /license_server at the repo root). Deliberately
isolated in this one small package (this file + gate.py's UI) so that
removing licensing entirely means deleting this package and one function
call in app/main.py -- nothing in app/backtest, app/strategy, app/web, or
any other part of the app ever imports from here or knows this exists.

Storage pattern mirrors app.ai.ollama_settings / app.data.alpaca_credentials
exactly: OS keyring when available for the one real secret (the license
key), a lightly-obfuscated local JSON file otherwise. Like every other
credential in this app, this is meant to keep a casual glance (or a
stray screenshot) from exposing the key, not to defeat someone who is
deliberately reverse-engineering the binary.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from app.data.storage import get_app_base_dir

SERVICE_NAME = "T58PropAlgoBacktester"
KEYRING_USERNAME = "license"

# Point this at wherever you deploy license_server/ (see its README.md).
# Overridable via the T58_LICENSE_SERVER_URL environment variable so the
# same built .exe can point at a staging server during testing without a
# rebuild -- set the env var, or just edit this default before building.
DEFAULT_SERVER_URL = "https://license.yourdomain.example.com"

# How long the app keeps working after the LAST successful ONLINE
# validation if the license server can't be reached at all (as opposed
# to being reached and saying "revoked"/"expired", which is never
# forgiven by the grace period). Protects a legitimate, currently-paying
# customer from a hotel wifi outage or your server having a bad day.
OFFLINE_GRACE_DAYS = 3


@dataclass
class LicenseState:
    email: str = ""
    license_key: str = ""
    device_id: str = ""
    status: str = ""  # active | expired | revoked | suspended | "" (never activated)
    expires_at: str | None = None
    last_validated_at: str | None = None  # ISO timestamp of the last successful ONLINE check


def _config_dir() -> Path:
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path() -> Path:
    return _config_dir() / "license_state.json"


def _key_fallback_path() -> Path:
    return _config_dir() / "license_key.txt"


def _obfuscate(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _deobfuscate(value: str) -> str:
    try:
        return base64.b64decode(value.encode("ascii")).decode("utf-8")
    except Exception:  # noqa: BLE001
        return ""


def _try_keyring():
    try:
        import keyring  # type: ignore
        from keyring.backends.fail import Keyring as FailKeyring  # type: ignore

        backend = keyring.get_keyring()
        if isinstance(backend, FailKeyring):
            return None
        return keyring
    except Exception:  # noqa: BLE001
        return None


def device_id() -> str:
    """A stable-per-install identifier -- not cryptographically strong,
    just enough to bind a license to "this machine" so a copied ZIP on a
    second computer can't silently reuse an already-activated license
    file: activate()/validate() there will report device_mismatch, since
    the server sees a different device_id than the one the license is
    bound to."""
    raw = f"{uuid.getnode()}-{platform.node()}-{platform.system()}-{platform.machine()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def save_state(state: LicenseState) -> None:
    payload = asdict(state)
    key = payload.pop("license_key", "")
    _state_path().write_text(json.dumps(payload), encoding="utf-8")

    kr = _try_keyring()
    if kr is not None:
        try:
            if key:
                kr.set_password(SERVICE_NAME, KEYRING_USERNAME, key)
            else:
                kr.delete_password(SERVICE_NAME, KEYRING_USERNAME)
            _key_fallback_path().unlink(missing_ok=True)
            return
        except Exception:  # noqa: BLE001
            pass  # fall through to the file-based fallback below

    if key:
        _key_fallback_path().write_text(_obfuscate(key), encoding="utf-8")
    else:
        _key_fallback_path().unlink(missing_ok=True)


def load_state() -> LicenseState:
    path = _state_path()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}

    key = ""
    kr = _try_keyring()
    if kr is not None:
        try:
            key = kr.get_password(SERVICE_NAME, KEYRING_USERNAME) or ""
        except Exception:  # noqa: BLE001
            key = ""
    if not key:
        fallback = _key_fallback_path()
        if fallback.exists():
            try:
                key = _deobfuscate(fallback.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                key = ""

    return LicenseState(
        email=data.get("email", ""), license_key=key, device_id=data.get("device_id", ""),
        status=data.get("status", ""), expires_at=data.get("expires_at"),
        last_validated_at=data.get("last_validated_at"),
    )


def clear_state() -> None:
    """Used by both explicit logout/deactivation and by activate() when
    swapping to a brand-new key -- never leaves a stale key sitting in
    the keyring/fallback file after a state change."""
    _state_path().unlink(missing_ok=True)
    _key_fallback_path().unlink(missing_ok=True)
    kr = _try_keyring()
    if kr is not None:
        try:
            kr.delete_password(SERVICE_NAME, KEYRING_USERNAME)
        except Exception:  # noqa: BLE001
            pass


def _server_url() -> str:
    return os.environ.get("T58_LICENSE_SERVER_URL", DEFAULT_SERVER_URL)


def _post(path: str, payload: dict, timeout: float = 10.0) -> tuple[bool, dict, str | None]:
    """Returns (network_ok, response_json, error_detail).

    network_ok=False means the server could not be reached at all
    (offline, DNS failure, timeout, connection refused) -- this is what
    triggers the offline grace period in validate() below. network_ok=True
    with response["ok"]=False means the server WAS reached and gave a
    real, authoritative answer (e.g. "revoked") -- never subject to the
    grace period, on purpose."""
    url = _server_url().rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode("utf-8")), None
    except HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            body = {"ok": False, "error": f"http_{exc.code}"}
        return True, body, None
    except (URLError, TimeoutError, OSError) as exc:
        return False, {}, str(exc)


_ERROR_MESSAGES = {
    "not_found": "That license key wasn't found. Double-check it and try again.",
    "email_mismatch": "That license key is registered to a different email address.",
    "device_mismatch": "That license is already active on a different device. Deactivate it there first (Account \u2192 License \u2192 Deactivate this device), or contact support.",
    "expired": "This license has expired. Renew your subscription to keep using T58.",
    "revoked": "This license has been revoked.",
    "suspended": "This license is temporarily suspended. Contact support if you believe this is a mistake.",
}


def _error_message(code: str) -> str:
    return _ERROR_MESSAGES.get(code, f"License error: {code}")


def activate(email: str, license_key: str) -> tuple[bool, str]:
    """Returns (ok, message). Persists the new state locally on success."""
    email = email.strip()
    license_key = license_key.strip().upper()
    if not email or not license_key:
        return False, "Enter both your email and license key."

    did = device_id()
    ok, body, err = _post("/activate", {"email": email, "license_key": license_key, "device_id": did})
    if not ok:
        return False, f"Couldn't reach the license server: {err}"
    if not body.get("ok"):
        return False, _error_message(body.get("error", "unknown_error"))

    save_state(LicenseState(
        email=email, license_key=license_key, device_id=did, status=body.get("status", "active"),
        expires_at=body.get("expires_at"), last_validated_at=datetime.now(timezone.utc).isoformat(),
    ))
    return True, "Activated."


def validate() -> tuple[bool, str]:
    """Checks the currently-saved license. Returns (ok_to_run, message).

    Three cases:
      1. Never activated locally -> (False, ...).
      2. Server reachable -> the real, current status decides ok_to_run;
         the local cache is refreshed either way.
      3. Server unreachable -> falls back to the offline grace period:
         ok_to_run is True only if the last successful ONLINE validation
         was both recent enough (OFFLINE_GRACE_DAYS) AND was itself
         "active" at the time -- a license that was already
         expired/revoked the last time it COULD reach the server does
         not get a free pass just because the network is down now.
    """
    state = load_state()
    if not state.license_key or not state.email:
        return False, "Not activated."

    ok, body, err = _post("/validate", {
        "email": state.email, "license_key": state.license_key, "device_id": state.device_id or device_id(),
    })
    if ok:
        if body.get("ok"):
            state.status = body.get("status", "active")
            state.expires_at = body.get("expires_at")
            state.last_validated_at = datetime.now(timezone.utc).isoformat()
            save_state(state)
            return True, "Active."
        state.status = body.get("error", "invalid")
        save_state(state)
        return False, _error_message(body.get("error", "unknown_error"))

    if state.last_validated_at and state.status == "active":
        try:
            last = datetime.fromisoformat(state.last_validated_at)
            age_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400.0
            if age_days <= OFFLINE_GRACE_DAYS:
                remaining = max(OFFLINE_GRACE_DAYS - age_days, 0.0)
                return True, f"Offline -- last verified {age_days:.1f} day(s) ago ({remaining:.1f} day(s) of offline use left before this needs to reconnect)."
        except ValueError:
            pass
    return False, f"Couldn't reach the license server, and the offline grace period has run out. Reconnect to the internet and try again. ({err})"


def deactivate() -> tuple[bool, str]:
    """Frees this device's binding on the server (so the license can be
    activated on a different machine) and clears everything saved
    locally -- the "logout / deactivate this device" the person asked
    for. Always clears the local state even if the server can't be
    reached, so this is never a way to get stuck logged in."""
    state = load_state()
    if not state.license_key:
        clear_state()
        return True, "Nothing was activated."
    ok, body, err = _post("/deactivate", {"license_key": state.license_key, "device_id": state.device_id or device_id()})
    clear_state()
    if not ok:
        return True, f"Deactivated locally. Couldn't reach the server to free this device slot remotely ({err}) -- contact support if you need that done."
    if not body.get("ok"):
        return True, f"Deactivated locally. Server said: {_error_message(body.get('error', 'unknown_error'))}"
    return True, "Deactivated."
