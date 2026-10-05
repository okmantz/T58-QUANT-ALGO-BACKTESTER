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

# --- License server URL (v7) ---------------------------------------------------
# Where license_server/ is deployed. There is deliberately NO placeholder
# default anymore: the old "https://license.yourdomain.example.com" meant
# every paying customer's activation died against a dead domain with a
# confusing network error. Resolution order:
#   1. T58_LICENSE_SERVER_URL environment variable (runtime -- the way a
#      from-source dev run or a staging deploy points the app somewhere
#      without a rebuild; also the local-testing path in
#      license_server/README.md).
#   2. app/licensing/build_config.py LICENSE_SERVER_URL (baked in at
#      build time -- the GitHub release workflows generate build_config.py
#      from the T58_LICENSE_SERVER_URL repo secret right before PyInstaller
#      runs, so the shipped .exe carries the real URL; see
#      license_server/DEPLOY.md step 7).
#   3. None -- and then the app FAILS LOUDLY at startup (see
#      check_license_server_configured() and its call in
#      app/licensing/gate.py's ensure_licensed()) instead of silently
#      falling back to a dead URL. The master-key activation path is fully
#      offline and keeps working regardless (see the master-key section).
def _baked_license_server_url() -> str:
    """The build-time URL from app/licensing/build_config.py, or "".

    build_config.py is GENERATED at release-build time (never committed
    with real values -- see build_config.py.example); its absence on a
    from-source dev checkout is normal, not an error."""
    try:
        from app.licensing import build_config
    except ImportError:
        return ""
    return str(getattr(build_config, "LICENSE_SERVER_URL", "") or "").strip().rstrip("/")


def resolve_license_server_url() -> str | None:
    """The effective license-server URL, or None if none is configured."""
    env_value = os.environ.get("T58_LICENSE_SERVER_URL", "").strip().rstrip("/")
    if env_value:
        return env_value
    baked = _baked_license_server_url()
    return baked or None


_NO_LICENSE_SERVER_URL_MESSAGE = (
    "This T58 build has no license server URL configured, so it cannot activate "
    "or validate a server license. Set the T58_LICENSE_SERVER_URL environment "
    "variable, or rebuild with the URL baked in (see license_server/DEPLOY.md "
    "step 7). A master key still activates fully offline -- enter it in the "
    "activation window if you have one."
)


def check_license_server_configured() -> tuple[bool, str]:
    """(configured, message) -- the fail-LOUD check for a missing server URL.

    Called from app/licensing/gate.py's ensure_licensed() BEFORE the
    activation window is shown: a licensed build with no URL can never
    activate anyone, so the startup must say so plainly instead of letting
    the person type a key into a form that is guaranteed to fail."""
    url = resolve_license_server_url()
    if url:
        return True, url
    return False, _NO_LICENSE_SERVER_URL_MESSAGE

# How long the app keeps working after the LAST successful ONLINE
# validation if the license server can't be reached at all (as opposed
# to being reached and saying "revoked"/"expired", which is never
# forgiven by the grace period). Protects a legitimate, currently-paying
# customer from a hotel wifi outage or your server having a bad day.
OFFLINE_GRACE_DAYS = 3

# Startup speed (Oct 2026). validate() used to block every launch on a network round
# trip (10s timeout) -- and main() called it twice, plus a third time right after
# activating. Now: a launch within FAST_START_MAX_AGE_HOURS of the last successful
# online check opens immediately and re-verifies in the background (a revoked /
# expired license is still caught: the background check records the new status, and
# the next launch then does a full blocking validate). Shorter timeout when a real
# check is needed.
FAST_START_MAX_AGE_HOURS = 6.0
VALIDATE_TIMEOUT_S = 6.0

# --- Master key ------------------------------------------------------------
# A single, permanent, fully-offline override for the app's own owner --
# the owner shouldn't depend on a live server, a device binding, or an
# internet connection just to open their own software.
#
# Only the SHA-256 HASH of the real key is ever configured -- never the
# key itself: matching it requires knowing the actual key, not just
# reading this source, the same way a password hash doesn't reveal the
# password. The key is handed to the owner once and never
# printed/logged/stored in plaintext anywhere by this module --
# activate()/validate() below only ever hash the entered value and
# compare (UPPERCASED with surrounding whitespace stripped -- see
# _is_master_key).
#
# v7 (Oct 2026): the old hardcoded _DEFAULT_MASTER_KEY_HASH is GONE --
# a hash baked into a public repo is un-revocable by design, so the
# default is now EMPTY. The hash comes from build-time configuration:
#   1. T58_MASTER_LICENSE_KEY_HASH environment variable (when set) --
#      used directly at runtime AND read by the release workflows at
#      build time to bake the hash into build_config.py for the .exe
#      (see license_server/DEPLOY.md step 7 -- Owen sets this per
#      release, never committed to the repo).
#   2. app/licensing/build_config.py MASTER_LICENSE_KEY_HASH (the
#      build-time-baked value).
#   3. Nothing -- and then the master-key path is simply UNAVAILABLE in
#      this build, with the clear hint below on every master-key-shaped
#      attempt. Never silently insecure: an unset hash can never
#      accidentally validate anything.
#
# Rotation = new hash -> rebuild -> re-ship; old builds keep honoring
# their old hash until they are replaced.
def _baked_master_key_hash() -> str:
    """The build-time master-key hash from app/licensing/build_config.py,
    or "". See _baked_license_server_url() above for why its absence is
    normal on a from-source checkout."""
    try:
        from app.licensing import build_config
    except ImportError:
        return ""
    return str(getattr(build_config, "MASTER_LICENSE_KEY_HASH", "") or "").strip()


def _master_key_hash() -> str | None:
    """The effective master-key hash, or None if none is configured.
    Precedence: T58_MASTER_LICENSE_KEY_HASH env var (when set) >
    build_config.MASTER_LICENSE_KEY_HASH (baked at build time, when set) >
    None (master-key path unavailable in this build)."""
    env_value = os.environ.get("T58_MASTER_LICENSE_KEY_HASH", "").strip()
    if env_value:
        return env_value
    baked = _baked_master_key_hash()
    return baked or None


def _is_master_key(license_key: str) -> bool:
    configured = _master_key_hash()
    if not configured or not license_key:
        return False
    return hashlib.sha256(license_key.strip().upper().encode("utf-8")).hexdigest() == configured


def master_key_configured() -> bool:
    """Whether a master key hash is available in this environment."""
    return _master_key_hash() is not None


_MASTER_KEY_UNSET_HINT = (
    "If this was a master-key attempt: no master key is configured in this build. "
    "Set the T58_MASTER_LICENSE_KEY_HASH environment variable (to the SHA-256 hash "
    "of the key, uppercased and stripped) at build/startup time before a master key "
    "can activate anything. See app/licensing/client.py for how to compute the hash."
)


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


# Session-only override for the "Remember this license on this device"
# checkbox (see gate.py's ActivationWindow -- previously visible but
# completely unwired; every launch persisted regardless of its state).
# When set, save_state()/load_state()/clear_state() all operate on this
# in-process value instead of touching disk/keyring at all, so unchecking
# the box means the activation genuinely does not survive past this
# process exiting -- the next launch finds no saved license and shows
# the activation window again, exactly like a real "session only" mode
# should. None means "no override" (the normal, persistent path below).
_session_state: LicenseState | None = None
_session_only_active = False


def _use_session_only(state: LicenseState) -> None:
    """Called by activate() instead of a real save_state() when the
    person unchecked "Remember this license on this device". Does NOT
    touch whatever was already persisted on disk/keyring from a prior,
    remembered activation -- unrelated to this session -- it only makes
    THIS process's load_state() calls return `state` from memory until
    the process exits or clear_state()/a remembered activate() replaces
    it."""
    global _session_state, _session_only_active
    _session_state = state
    _session_only_active = True


def save_state(state: LicenseState) -> None:
    global _session_state
    if _session_only_active:
        _session_state = state
        return

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
    if _session_only_active:
        return _session_state if _session_state is not None else LicenseState()

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
    the keyring/fallback file after a state change. Also drops any
    session-only state and turns the override back off, so a fresh
    activate() right after a clear starts from a clean slate either way."""
    global _session_state, _session_only_active
    _session_state = None
    _session_only_active = False
    _state_path().unlink(missing_ok=True)
    _key_fallback_path().unlink(missing_ok=True)
    kr = _try_keyring()
    if kr is not None:
        try:
            kr.delete_password(SERVICE_NAME, KEYRING_USERNAME)
        except Exception:  # noqa: BLE001
            pass


def _server_url() -> str | None:
    """The resolved license-server URL, or None when none is configured.
    (v7: the old placeholder default is gone -- see
    resolve_license_server_url() above.)"""
    return resolve_license_server_url()


def _post(path: str, payload: dict, timeout: float = 10.0) -> tuple[bool, dict, str | None]:
    """Returns (network_ok, response_json, error_detail).

    network_ok=False means the server could not be reached at all
    (offline, DNS failure, timeout, connection refused) -- this is what
    triggers the offline grace period in validate() below. network_ok=True
    with response["ok"]=False means the server WAS reached and gave a
    real, authoritative answer (e.g. "revoked") -- never subject to the
    grace period, on purpose.

    v7: a missing server URL is its own clear error (not a crash on
    None.rstrip, and not a confusing DNS failure against a placeholder).
    The offline grace period still applies to a genuinely unreachable
    CONFIGURED server; a build with NO server configured fails loudly
    earlier, at ensure_licensed()."""
    base = _server_url()
    if not base:
        return False, {}, _NO_LICENSE_SERVER_URL_MESSAGE
    url = base.rstrip("/") + path
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


def activate(email: str, license_key: str, remember: bool = True) -> tuple[bool, str]:
    """Returns (ok, message).

    `remember=True` (the default, and the only behavior that existed
    before this parameter did) persists the activated state to the OS
    keyring/local file exactly as before, so it's still there next
    launch. `remember=False` is the actual implementation of the
    activation window's "Remember this license on this device" checkbox
    when unchecked: the license is fully usable for the REST OF THIS
    PROCESS (validate() below sees it, every feature works normally),
    but nothing is written to disk/keyring -- closing and reopening the
    app finds no saved license and shows the activation window again,
    same as a brand-new install. See _use_session_only()'s docstring for
    the mechanics."""
    email = email.strip()
    license_key = license_key.strip().upper()
    if not email or not license_key:
        return False, "Enter both your email and license key."
    persist = save_state if remember else _use_session_only

    if _is_master_key(license_key):
        # Permanent, offline activation -- no server contacted, no device
        # binding, no expiry. The hash lives ONLY in the
        # T58_MASTER_LICENSE_KEY_HASH environment variable (see the
        # comment on _master_key_hash above).
        persist(LicenseState(
            email=email, license_key=license_key, device_id=device_id(), status="active",
            expires_at=None, last_validated_at=datetime.now(timezone.utc).isoformat(),
        ))
        return True, "Activated (master key -- no license server required)."

    did = device_id()
    ok, body, err = _post("/activate", {"email": email, "license_key": license_key, "device_id": did})
    if not ok:
        return False, f"Couldn't reach the license server: {err}{'' if master_key_configured() else ' ' + _MASTER_KEY_UNSET_HINT}"
    if not body.get("ok"):
        msg = _error_message(body.get("error", "unknown_error"))
        if not master_key_configured():
            # A master-key attempt is indistinguishable from a wrong key
            # here -- the only way to tell the operator what to do is to
            # say so on every failed activation when no master key is
            # configured.
            msg += " " + _MASTER_KEY_UNSET_HINT
        return False, msg

    persist(LicenseState(
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

    if _is_master_key(state.license_key):
        # Never contacts the server, never expires, never subject to the
        # offline grace period -- see _master_key_hash's comment.
        return True, "Active (master key)."

    ok, body, err = _post("/validate", {
        "email": state.email, "license_key": state.license_key, "device_id": state.device_id or device_id(),
    }, timeout=VALIDATE_TIMEOUT_S)
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


def validate_fast_start() -> tuple[bool, str]:
    """validate(), but never makes the person wait on the network at launch
    when they were verified online recently -- see FAST_START_MAX_AGE_HOURS."""
    state = load_state()
    if state.license_key and state.email and state.status == "active" and state.last_validated_at:
        if _is_master_key(state.license_key):
            return True, "Active (master key)."
        try:
            last = datetime.fromisoformat(state.last_validated_at)
            age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
        except ValueError:
            age_h = None
        if age_h is not None and 0 <= age_h <= FAST_START_MAX_AGE_HOURS:
            import threading
            threading.Thread(target=validate, daemon=True, name="t58-license-recheck").start()
            return True, "Active (verified recently)."
    return validate()


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
    if _is_master_key(state.license_key):
        clear_state()
        return True, "Deactivated (master key -- nothing to free on a server)."
    ok, body, err = _post("/deactivate", {"license_key": state.license_key, "device_id": state.device_id or device_id()})
    clear_state()
    if not ok:
        return True, f"Deactivated locally. Couldn't reach the server to free this device slot remotely ({err}) -- contact support if you need that done."
    if not body.get("ok"):
        return True, f"Deactivated locally. Server said: {_error_message(body.get('error', 'unknown_error'))}"
    return True, "Deactivated."
