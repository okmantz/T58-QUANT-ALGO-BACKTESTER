"""
Persistent settings for saved LIVE prop-firm account connections.

Deliberately separate from app.forward_test.mt5_settings, which manages
exactly one demo account: this supports multiple named live accounts
(you may have funded accounts at more than one firm), and every account
here represents real capital, so it's kept as its own explicit module
rather than a "second mode" bolted onto the demo settings file.

Storage mirrors the same split used everywhere else in this app: a plain
JSON file for non-secret fields (nickname, firm, platform, login, server,
terminal path), keyed by a stable per-account id; each account's password
goes to the OS keyring under that same id, falling back to a
lightly-obfuscated local file if no keyring backend is available (same
fallback app.forward_test.mt5_settings and app.ai.ollama_settings use).
Passwords are never written into the plain JSON file.
"""
from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.data.storage import get_app_base_dir

SERVICE_NAME = "T58PropAlgoBacktester"


@dataclass
class LiveAccount:
    id: str | None          # None for a not-yet-saved account; a stable uuid4 hex once saved
    nickname: str
    firm_name: str
    platform: str
    login: str
    server: str
    password: str = ""      # secret -- keyring-backed, populated on load for programmatic use only
    terminal_path: str = ""
    # Non-MT5 platforms need more than login/password/server (OAuth client id+secret+refresh
    # token for cTrader; app id/secret+cid/sec for Tradovate; a DXtrade base_url; etc.) -- rather
    # than adding a new dataclass field per platform, every platform-specific credential lives
    # here as plain key/value strings. See app.live_deploy.broker_registry.build_adapter for
    # which keys each platform requires. Secret-shaped values in here (client_secret,
    # refresh_token, app_secret, sec) are stored keyring-backed exactly like `password` --
    # everything else (host, is_live, domain, ctid_trader_account_id) is not treated as secret
    # and lives in the plain JSON file.
    extra_credentials: dict = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        if self.platform == "MT4/MT5":
            return bool(self.login.strip()) and bool(self.server.strip()) and bool(self.password)
        return bool(self.login.strip()) or bool(self.extra_credentials)


# Extra-credential keys treated as secrets (keyring-backed like `password`, never written to
# the plain accounts JSON file). Everything else in extra_credentials is written in plain JSON.
_SECRET_EXTRA_KEYS = {"client_secret", "refresh_token", "app_secret", "sec", "access_token"}


def _config_dir() -> Path:
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _accounts_path() -> Path:
    return _config_dir() / "live_accounts.json"


def _password_fallback_path(account_id: str) -> Path:
    return _config_dir() / f"live_password_{account_id}.txt"


def _try_keyring():
    try:
        import keyring  # type: ignore
        from keyring.backends.fail import Keyring as FailKeyring  # type: ignore

        backend = keyring.get_keyring()
        if isinstance(backend, FailKeyring):
            return None
        return keyring
    except Exception:
        return None


def _obfuscate(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _deobfuscate(value: str) -> str:
    try:
        return base64.b64decode(value.encode("ascii")).decode("utf-8")
    except Exception:
        return ""


def _keyring_username(account_id: str) -> str:
    return f"live_account_{account_id}"


def _load_raw_list() -> list[dict]:
    path = _accounts_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_password(account_id: str, password: str) -> None:
    kr = _try_keyring()
    if kr is not None:
        try:
            username = _keyring_username(account_id)
            if password:
                kr.set_password(SERVICE_NAME, username, password)
            else:
                kr.delete_password(SERVICE_NAME, username)
            _password_fallback_path(account_id).unlink(missing_ok=True)
            return
        except Exception:
            pass  # fall through to the file-based fallback below

    fallback = _password_fallback_path(account_id)
    if password:
        fallback.write_text(_obfuscate(password), encoding="utf-8")
    else:
        fallback.unlink(missing_ok=True)


def _load_password(account_id: str) -> str:
    kr = _try_keyring()
    if kr is not None:
        try:
            pw = kr.get_password(SERVICE_NAME, _keyring_username(account_id))
            if pw:
                return pw
        except Exception:
            pass
    fallback = _password_fallback_path(account_id)
    if fallback.exists():
        try:
            return _deobfuscate(fallback.read_text(encoding="utf-8"))
        except Exception:
            return ""
    return ""


def _extra_keyring_username(account_id: str, key: str) -> str:
    return f"live_account_{account_id}_extra_{key}"


def _save_extra_credentials(account_id: str, extra: dict) -> dict:
    """Splits extra_credentials into secret keys (keyring-backed, one
    entry per key, same fallback as `password`) and non-secret keys
    (returned as-is for the plain JSON payload). Only overwrites a
    secret's keyring entry when a new non-empty value was actually
    provided, matching save_account's existing password behavior --
    editing an account's host/domain shouldn't force re-entering its
    client_secret."""
    plain: dict = {}
    for key, value in (extra or {}).items():
        if key in _SECRET_EXTRA_KEYS:
            if value:
                _save_password(f"{account_id}_extra_{key}", str(value))
        else:
            plain[key] = value
    return plain


def _load_extra_credentials(account_id: str, plain: dict) -> dict:
    out = dict(plain)
    for key in _SECRET_EXTRA_KEYS:
        val = _load_password(f"{account_id}_extra_{key}")
        if val:
            out[key] = val
    return out


def save_account(account: LiveAccount) -> str:
    """Creates a new account (assigning it an id) or updates an existing
    one (matched by id). Returns the account's id."""
    account_id = account.id or uuid.uuid4().hex
    accounts = _load_raw_list()
    payload = {
        "id": account_id,
        "nickname": account.nickname.strip(),
        "firm_name": account.firm_name,
        "platform": account.platform,
        "login": account.login.strip(),
        "server": account.server.strip(),
        "terminal_path": (account.terminal_path or "").strip(),
        "extra_credentials": _save_extra_credentials(account_id, account.extra_credentials),
    }
    accounts = [a for a in accounts if a.get("id") != account_id]
    accounts.append(payload)
    _accounts_path().write_text(json.dumps(accounts, indent=2), encoding="utf-8")

    # An account being edited without re-entering the password keeps its
    # existing one -- only overwrite the keyring entry when a new,
    # non-empty password was actually provided.
    if account.password:
        _save_password(account_id, account.password)
    return account_id


def load_accounts() -> list[LiveAccount]:
    """Never raises -- returns an empty list if nothing is saved yet or
    the settings file is corrupt. Passwords ARE populated on the returned
    objects (for internal use connecting to MT5) -- callers must never
    write a loaded account's password back into a visible UI field."""
    accounts = []
    for raw in _load_raw_list():
        account_id = raw.get("id")
        if not account_id:
            continue
        accounts.append(LiveAccount(
            id=account_id,
            nickname=raw.get("nickname") or "",
            firm_name=raw.get("firm_name") or "",
            platform=raw.get("platform") or "",
            login=raw.get("login") or "",
            server=raw.get("server") or "",
            password=_load_password(account_id),
            terminal_path=raw.get("terminal_path") or "",
            extra_credentials=_load_extra_credentials(account_id, raw.get("extra_credentials") or {}),
        ))
    return accounts


def delete_account(account_id: str) -> None:
    accounts = [a for a in _load_raw_list() if a.get("id") != account_id]
    _accounts_path().write_text(json.dumps(accounts, indent=2), encoding="utf-8")
    _save_password(account_id, "")  # clears both keyring and fallback file
    for key in _SECRET_EXTRA_KEYS:
        _save_password(f"{account_id}_extra_{key}", "")
