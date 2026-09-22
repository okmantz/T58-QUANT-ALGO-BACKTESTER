"""
Account Settings -- the person's own profile fields, stored locally.

Deliberately as minimal as app.web.notifications' non-secret fields: this
app has no server-side account system, no cloud login, and no remote
identity of any kind. Everything here (name/username/email, and an
OPTIONAL local lock password) lives in one small JSON file on THIS
computer, exactly like NotificationSettings' non-secret fields --
there is no "your account" anywhere else for this to sync with.

The optional password is a LOCAL APP LOCK, not a login to anything.
Setting one asks for that password before this computer's browser can
open any page of the (already-running) web app -- useful mainly for the
Phone Access / LAN-exposed case (see mobile_access.html) where anyone on
the same Wi-Fi could otherwise just open the URL. Leaving it blank (the
default) means the app behaves exactly as before this existed: no lock,
no prompt, nothing gating any page. See app.web.server's `_account_lock_gate`
before_request hook for where this is actually enforced.

Never stores a plaintext password -- only a salted PBKDF2 hash, in the
same file, using the stdlib's own `hashlib.pbkdf2_hmac` (no extra
dependency needed for this).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from app.data.storage import get_app_base_dir

ACCOUNT_SETTINGS_FILENAME = "account_settings.json"

_PBKDF2_ITERATIONS = 260_000
_PBKDF2_ALGO = "sha256"


@dataclass
class AccountSettings:
    display_name: str = ""
    username: str = ""
    email: str = ""
    company: str = ""  # optional -- e.g. "T58 Trading", shown on exported reports if set
    password_hash: str = ""  # "pbkdf2$<algo>$<iterations>$<salt_hex>$<hash_hex>", or "" if no lock is set
    # UPGRADE (Account tab pass): filename only (e.g. "profile_picture.png"),
    # relative to the same config directory _settings_path() lives in --
    # see app.web.server's /settings/account/profile-picture routes for
    # where the actual image bytes are written/served. Never a full path
    # (this file may move between machines/backups) and never the image
    # bytes themselves (this is a small JSON file, not a blob store).
    profile_picture_filename: str = ""

    @property
    def has_password(self) -> bool:
        return bool(self.password_hash)


def _settings_path() -> Path:
    return get_app_base_dir() / ACCOUNT_SETTINGS_FILENAME


def load_account_settings() -> AccountSettings:
    path = _settings_path()
    if not path.exists():
        return AccountSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AccountSettings(
            display_name=str(data.get("display_name", "")),
            username=str(data.get("username", "")),
            email=str(data.get("email", "")),
            company=str(data.get("company", "")),
            password_hash=str(data.get("password_hash", "")),
            profile_picture_filename=str(data.get("profile_picture_filename", "")),
        )
    except Exception:  # noqa: BLE001 -- a corrupt/unreadable file must never crash the Account tab
        return AccountSettings()


def save_account_settings(settings: AccountSettings) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def hash_password(plaintext: str) -> str:
    """Salts + hashes `plaintext` for storage. Empty input returns "" (the
    "no lock set" sentinel), so clearing the password field and saving
    is exactly how a lock gets removed."""
    if not plaintext:
        return ""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(_PBKDF2_ALGO, plaintext.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2${_PBKDF2_ALGO}${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(plaintext: str, password_hash: str) -> bool:
    """Constant-time check of `plaintext` against a hash produced by
    hash_password(). Never raises -- a malformed/corrupt stored hash is
    treated as "does not match" rather than crashing the lock screen."""
    if not password_hash:
        return False
    try:
        algo_tag, algo, iterations_s, salt_hex, hash_hex = password_hash.split("$")
        if algo_tag != "pbkdf2":
            return False
        digest = hashlib.pbkdf2_hmac(algo, plaintext.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations_s))
        return hmac.compare_digest(digest.hex(), hash_hex)
    except Exception:  # noqa: BLE001
        return False
