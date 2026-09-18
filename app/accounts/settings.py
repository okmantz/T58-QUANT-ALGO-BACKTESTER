"""
Account Settings -- the person's own display name/email, stored locally.

Deliberately as minimal as app.web.notifications' non-secret fields: no
login, no password, no auth of any kind (this app has no server-side
account system to log into) -- just the couple of fields the ACCOUNT
tab/page needs to let someone put their own name/email on their local
install (e.g. so exported reports or future notification defaults can
reference it). Lives in its own module/file (account_settings.json)
rather than being folded into NotificationSettings so the two stay
independently loadable/savable -- Account Settings has no SMTP/secret
field at all, so it needs none of that module's keyring/obfuscation
machinery.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.data.storage import get_app_base_dir

ACCOUNT_SETTINGS_FILENAME = "account_settings.json"


@dataclass
class AccountSettings:
    display_name: str = ""
    email: str = ""
    company: str = ""  # optional -- e.g. "T58 Trading", shown on exported reports if set


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
            email=str(data.get("email", "")),
            company=str(data.get("company", "")),
        )
    except Exception:  # noqa: BLE001 -- a corrupt/unreadable file must never crash the Account tab
        return AccountSettings()


def save_account_settings(settings: AccountSettings) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
