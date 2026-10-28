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
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

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
    the person type a key into a form that is guaranteed to fail.

    A configured allowlist URL (see the allowlist section below) counts as
    configured too -- that mode needs no license server at all."""
    url = resolve_allowlist_url() or resolve_sheet_url() or resolve_license_server_url()
    if url:
        return True, url
    return False, _NO_LICENSE_SERVER_URL_MESSAGE


# --- Google Sheet mode (license_sheet/Code.gs) -----------------------------------
# The license list lives in a PRIVATE Google Sheet that you manage by hand
# (key / customer / email / status / expiry / bound device). A small Apps Script
# bound to that sheet is deployed as a web app; this client asks it
# "activate / validate / deactivate" over a GET request and it answers with the
# same JSON the old license server used ({"ok": ..., "error": ..., "status": ...,
# "expires_at": ...}). The sheet itself is never exposed -- the script only ever
# answers about the one key it is asked about. See license_sheet/SETUP.md.
# Resolution order for the URL, highest first:
#   1. T58_LICENSE_SHEET_URL environment variable
#   2. app/licensing/build_config.py LICENSE_SHEET_URL (baked at build time)
# Precedence between modes: allowlist URL > sheet URL > license-server URL.
def _baked_sheet_url() -> str:
    try:
        from app.licensing import build_config
    except ImportError:
        return ""
    return str(getattr(build_config, "LICENSE_SHEET_URL", "") or "").strip()


def resolve_sheet_url() -> str | None:
    """The effective Google Sheet web-app URL, or None when sheet mode is off."""
    env_value = os.environ.get("T58_LICENSE_SHEET_URL", "").strip()
    if env_value:
        return env_value
    return _baked_sheet_url() or None


def _sheet_get(path: str, payload: dict, timeout: float) -> tuple[bool, dict, str | None]:
    """Same contract as _post(): (network_ok, response_json, error_detail).
    Apps Script web apps answer GETs reliably (and redirect to a googleusercontent
    URL, which urllib follows), so each request is a GET with query parameters.
    Anything that is not a clean JSON answer -- offline, an HTTP error, a Google
    sign-in/error page because the web app was deployed with the wrong access
    setting, a transient "busy" -- counts as UNREACHABLE (offline grace applies),
    never as "revoked", so a deployment mistake can't lock paying customers out."""
    base = resolve_sheet_url()
    params = {"action": path.strip("/")}
    params.update({k: v for k, v in payload.items() if v is not None})
    url = base + ("&" if "?" in base else "?") + urlencode(params)
    req = urllib_request.Request(url, headers={"User-Agent": "T58-License/1"})
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        return False, {}, f"license sheet returned HTTP {exc.code}"
    except (URLError, TimeoutError, OSError) as exc:
        return False, {}, str(exc)
    try:
        body = json.loads(raw)
    except ValueError:
        body = None
    if not isinstance(body, dict) or "ok" not in body:
        return False, {}, ("unexpected reply from the license sheet -- check the Apps Script web app is "
                           "deployed with \"Who has access: Anyone\"")
    if body.get("error") in ("busy", "not_configured"):
        # Transient / setup problems on YOUR side are never a verdict on the customer.
        return False, {}, "the license sheet is busy or not set up -- try again in a moment"
    return True, body, None


# --- Allowlist mode (no license server, no hosting) ------------------------------
# Serverless licensing: you keep a list of license keys; the app is built with
# the URL of a small PUBLIC text file that holds only the SHA-256 HASH of every
# currently-valid key (one per line). At activation and on every launch the app
# downloads that file and checks that the hash of the entered key is in it.
# Revoking a key = deleting its line from the file (commit it); every install
# that can reach the file stops working the next time it checks. No server to
# run, nothing to pay for -- GitHub (or any static host) just serves a text file.
#
# Why hashes and a PUBLIC file: a private repo can't be read by shipped software
# without embedding an access token, and that token would expose your source.
# A hash of a high-entropy random key (build_tools/license_keys.py makes
# ~100-bit keys) can't be reversed, so publishing the hashes leaks nothing.
#
# What this mode does NOT do (needs a server): bind a key to one machine, or
# stop a customer from handing their key to a friend until you notice and
# revoke it. Resolution order for the URL, highest first:
#   1. T58_LICENSE_ALLOWLIST_URL environment variable
#   2. app/licensing/build_config.py LICENSE_ALLOWLIST_URL (baked at build time)
#   3. None -> the app falls back to the license-server mode above.
# When an allowlist URL is set it takes precedence over the license server.
ALLOWLIST_TIMEOUT_S = 8.0


def _baked_allowlist_url() -> str:
    try:
        from app.licensing import build_config
    except ImportError:
        return ""
    return str(getattr(build_config, "LICENSE_ALLOWLIST_URL", "") or "").strip()


def resolve_allowlist_url() -> str | None:
    """The effective allowlist URL, or None when allowlist mode is off."""
    env_value = os.environ.get("T58_LICENSE_ALLOWLIST_URL", "").strip()
    if env_value:
        return env_value
    return _baked_allowlist_url() or None


def key_hash(license_key: str) -> str:
    """SHA-256 hex of the key, uppercased and stripped -- the same
    normalization the master-key check uses."""
    return hashlib.sha256(license_key.strip().upper().encode("utf-8")).hexdigest()


_HASH_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")


def _parse_allowlist(text: str) -> set[str]:
    """Accepts a plain text file (one hash per line, '#' comments) OR a CSV /
    published Google Sheet export (hash in any column, extra columns such as
    "issued_to" are ignored). A row containing the word "revoked" is skipped,
    so a key can be revoked by marking its row instead of deleting it."""
    hashes: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        if "revoked" in line.lower():
            continue
        match = _HASH_RE.search(line)
        if match:
            hashes.add(match.group(0).lower())
    return hashes


def _fetch_allowlist(timeout: float = ALLOWLIST_TIMEOUT_S) -> tuple[bool, set[str], str | None]:
    """Returns (reached, hashes, error). reached=False covers everything that
    is not an authoritative answer -- offline, DNS failure, HTTP error (404,
    5xx), or a file with no valid hashes in it (e.g. an accidental empty
    commit) -- so a bad publish can never lock out every paying customer; it
    just triggers the offline grace period instead of a mass revocation."""
    url = resolve_allowlist_url()
    if not url:
        return False, set(), "no allowlist URL configured"
    req = urllib_request.Request(url, headers={"User-Agent": "T58-License/1", "Cache-Control": "no-cache"})
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return False, set(), str(exc)
    hashes = _parse_allowlist(text)
    if not hashes:
        return False, set(), "the license list was empty or unreadable"
    return True, hashes, None


def _offline_grace_result(state: "LicenseState", err: str | None, what: str = "license server") -> tuple[bool, str]:
    """The shared offline-grace decision: ok only if the last successful
    ONLINE check was recent enough AND was itself "active"."""
    if state.last_validated_at and state.status == "active":
        try:
            last = datetime.fromisoformat(state.last_validated_at)
            age_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400.0
            if age_days <= OFFLINE_GRACE_DAYS:
                remaining = max(OFFLINE_GRACE_DAYS - age_days, 0.0)
                return True, f"Offline -- last verified {age_days:.1f} day(s) ago ({remaining:.1f} day(s) of offline use left before this needs to reconnect)."
        except ValueError:
            pass
    return False, f"Couldn't reach the {what}, and the offline grace period has run out. Reconnect to the internet and try again. ({err})"

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
# Resolution order (highest wins):
#   1. T58_MASTER_LICENSE_KEY_HASH environment variable (when set) --
#      runtime override AND the rotation path: point it at a new hash and
#      every build honors the new key immediately, no rebuild needed.
#   2. app/licensing/build_config.py MASTER_LICENSE_KEY_HASH (baked at
#      build time from the T58_MASTER_LICENSE_KEY_HASH repo secret -- see
#      license_server/DEPLOY.md step 7; never committed to the repo).
#   3. _DEFAULT_MASTER_KEY_HASH below -- compiled into every build, so the
#      owner's key works with zero configuration: from-source dev runs,
#      CI-built .exes, any machine, no env vars, no server.
#
# On the compiled-in default living in a public repo (Oct 2026 -- Owen's
# explicit direction, reversing the v7 "no baked default" call): a
# SHA-256 hash is not the key. Recovering the key from the hash means a
# preimage attack against a 192-bit-entropy key -- computationally
# infeasible. What IS true, and worth knowing: anyone reading the repo
# can see which hash the builds accept, and already-shipped builds keep
# honoring their baked hash until replaced -- but that was already true
# of the build_config.py path, so this default is no worse on that axis.
# The plaintext key itself never appears in the repo, in any
# build_config, or in any log.
#
# Rotation = new key -> new hash -> set the env var (instant, no rebuild)
# or set the repo secret and rebuild -> re-ship; old builds keep honoring
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


# SHA-256 hex digest of the owner's master key -- the key itself, uppercased
# and stripped of surrounding whitespace, then hashed, exactly as
# _is_master_key() checks it. This is the zero-configuration fallback so
# the master key activates offline in every build with nothing to set up.
_DEFAULT_MASTER_KEY_HASH = "5cce872f1fdac2d290e817e707376212e224c96a1d97976c2809e7ec45922f5c"


def _master_key_hash() -> str | None:
    """The effective master-key hash, or None if none is configured.
    Precedence: T58_MASTER_LICENSE_KEY_HASH env var (when set) >
    build_config.MASTER_LICENSE_KEY_HASH (baked at build time, when set) >
    _DEFAULT_MASTER_KEY_HASH (compiled in -- the owner's key works with
    zero configuration in every build)."""
    env_value = os.environ.get("T58_MASTER_LICENSE_KEY_HASH", "").strip()
    if env_value:
        return env_value
    baked = _baked_master_key_hash()
    if baked:
        return baked
    return _DEFAULT_MASTER_KEY_HASH or None


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


# --- Plans -------------------------------------------------------------------
# The plan a license key belongs to. The source of truth is the PLAN column of
# the license sheet (license_sheet/Code.gs PLANS) -- or the license server's
# `plan` field -- and it reaches the app in the activate/validate answer. The
# Account page shows it read-only; nothing a customer types can change it.
# The master key is always Lifetime.
MASTER_PLAN = "Lifetime"
KNOWN_PLANS = ("Pro Monthly", "Pro Annual", "Lifetime", "Trial", "Complimentary")
_PLAN_ALIASES = {
    "monthly": "Pro Monthly", "pro monthly": "Pro Monthly", "month": "Pro Monthly",
    "annual": "Pro Annual", "yearly": "Pro Annual", "pro annual": "Pro Annual", "pro yearly": "Pro Annual",
    "annually": "Pro Annual", "year": "Pro Annual",
    "lifetime": "Lifetime", "life": "Lifetime", "master": "Lifetime",
    "trial": "Trial", "complimentary": "Complimentary", "comp": "Complimentary",
}


def normalize_plan(raw: object) -> str:
    """Map whatever the sheet/server sent to one of KNOWN_PLANS (case, spacing and
    "monthly"/"annual" style aliases tolerated). An unrecognised non-empty value is
    kept as typed, so the Account page always matches what the sheet says; blank
    stays blank."""
    text = " ".join(str(raw or "").replace("_", " ").replace("-", " ").split())
    if not text:
        return ""
    return _PLAN_ALIASES.get(text.lower(), text)


@dataclass
class LicenseState:
    email: str = ""
    license_key: str = ""
    device_id: str = ""
    status: str = ""  # active | expired | revoked | suspended | "" (never activated)
    expires_at: str | None = None
    last_validated_at: str | None = None  # ISO timestamp of the last successful ONLINE check
    plan: str = ""  # the plan this key belongs to (see normalize_plan); "" = the license carried none


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
        last_validated_at=data.get("last_validated_at"), plan=data.get("plan", "") or "",
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
    if resolve_sheet_url():
        return _sheet_get(path, payload, timeout)
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
    "bad_request": "The license check couldn't read this device's details. Restart the app and try again.",
    "inactive": "This license isn't activated on this device. Enter your key again to activate it.",
    "not_configured": "The license sheet hasn't been set up yet. Contact support.",
}


def _error_message(code: str) -> str:
    return _ERROR_MESSAGES.get(code, f"License error: {code}")


def _activate_impl(email: str, license_key: str, remember: bool = True) -> tuple[bool, str]:
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
        # binding, no expiry. The hash comes from the T58_MASTER_LICENSE_KEY_HASH
        # env var, build_config.py, or the compiled-in default (see the
        # comment on _master_key_hash above) -- the plaintext key is never
        # stored anywhere by this module.
        persist(LicenseState(
            email=email, license_key=license_key, device_id=device_id(), status="active",
            expires_at=None, last_validated_at=datetime.now(timezone.utc).isoformat(),
            plan=MASTER_PLAN,
        ))
        return True, "Activated (master key -- no license server required)."

    did = device_id()

    if resolve_allowlist_url():
        # Allowlist mode: no server. The key is valid iff its hash is on the
        # published list right now. The email is just a label here.
        reached, hashes, err = _fetch_allowlist()
        if not reached:
            return False, f"Couldn't check the license list: {err}. Check your internet connection and try again."
        if key_hash(license_key) not in hashes:
            return False, "That license key wasn't recognized, or it has been revoked."
        persist(LicenseState(
            email=email, license_key=license_key, device_id=did, status="active",
            expires_at=None, last_validated_at=datetime.now(timezone.utc).isoformat(),
        ))
        return True, "Activated."

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
        plan=normalize_plan(body.get("plan")),
    ))
    return True, "Activated."


def activate(email: str, license_key: str, remember: bool = True) -> tuple[bool, str]:
    """Sign in. See _activate_impl for the license logic.

    v10 FIX: the saved-runs restore used to run only at the very end of the license-SERVER path, so signing in with the master
    key (which returns early, offline) or through the sheet / allowlist modes never restored anything -- a reinstall came back
    empty. It now runs after EVERY successful sign-in, whichever kind of key was used. Best effort: it never blocks or fails
    the activation itself."""
    ok, message = _activate_impl(email, license_key, remember=remember)
    if ok:
        try:
            from app.persistence import vault
            vault.on_activated()
        except Exception:  # noqa: BLE001
            pass
    return ok, message


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

    if resolve_allowlist_url():
        reached, hashes, err = _fetch_allowlist(timeout=VALIDATE_TIMEOUT_S)
        if reached:
            if key_hash(state.license_key) in hashes:
                state.status = "active"
                state.last_validated_at = datetime.now(timezone.utc).isoformat()
                save_state(state)
                return True, "Active."
            state.status = "revoked"
            save_state(state)
            return False, _error_message("revoked")
        return _offline_grace_result(state, err, what="license list")

    ok, body, err = _post("/validate", {
        "email": state.email, "license_key": state.license_key, "device_id": state.device_id or device_id(),
    }, timeout=VALIDATE_TIMEOUT_S)
    if ok:
        if body.get("ok"):
            state.status = body.get("status", "active")
            state.expires_at = body.get("expires_at")
            if "plan" in body:  # an upgrade/downgrade made in the sheet shows up at the next check-in
                state.plan = normalize_plan(body.get("plan"))
            state.last_validated_at = datetime.now(timezone.utc).isoformat()
            save_state(state)
            return True, "Active."
        state.status = body.get("error", "invalid")
        save_state(state)
        return False, _error_message(body.get("error", "unknown_error"))

    return _offline_grace_result(state, err)


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
    if resolve_allowlist_url():
        clear_state()  # allowlist mode has no server-side device slot to free
        return True, "Deactivated."
    ok, body, err = _post("/deactivate", {"license_key": state.license_key, "device_id": state.device_id or device_id()})
    clear_state()
    if not ok:
        return True, f"Deactivated locally. Couldn't reach the server to free this device slot remotely ({err}) -- contact support if you need that done."
    if not body.get("ok"):
        return True, f"Deactivated locally. Server said: {_error_message(body.get('error', 'unknown_error'))}"
    return True, "Deactivated."


# --- Plan shown on the Account page -------------------------------------------
REFRESH_MIN_INTERVAL_S = 600.0  # at most one background re-check per 10 minutes (unless forced)
_last_refresh_monotonic = 0.0


def current_plan(state: "LicenseState | None" = None) -> str:
    """The plan of the license saved on this device: Lifetime for the master key,
    otherwise whatever the sheet/server last reported ("" when it reported none)."""
    state = state if state is not None else load_state()
    if state.license_key and _is_master_key(state.license_key):
        return MASTER_PLAN
    return normalize_plan(state.plan)


def refresh_license_info(force: bool = False) -> bool:
    """Best-effort re-check so the Account page matches the license sheet. Called when
    the Account page opens: it asks the sheet/server again only if the saved plan is
    missing (a license activated before plans existed) or the last check-in is older
    than REFRESH_MIN_INTERVAL_S, and never more than once per interval unless `force`.
    Never raises and never touches the network for the master key or allowlist mode
    (neither carries a plan). Returns True if a check was actually attempted."""
    global _last_refresh_monotonic
    import time

    try:
        state = load_state()
        if not state.license_key or not state.email:
            return False
        if _is_master_key(state.license_key) or resolve_allowlist_url():
            return False
        now = time.monotonic()
        if not force and _last_refresh_monotonic and now - _last_refresh_monotonic < REFRESH_MIN_INTERVAL_S:
            return False
        stale = True
        if state.last_validated_at:
            try:
                age_s = (datetime.now(timezone.utc) - datetime.fromisoformat(state.last_validated_at)).total_seconds()
                stale = age_s > REFRESH_MIN_INTERVAL_S
            except ValueError:
                stale = True
        if not (force or not state.plan or stale):
            return False
        _last_refresh_monotonic = now
        validate()
        return True
    except Exception:  # noqa: BLE001 -- the Account page must always render
        return False
