"""
Persistent settings for the API keys that don't already have their own
storage module -- FRED, OpenAI, Claude/Anthropic, and London Strategic
Edge. Ollama (app.ai.ollama_settings), Alpaca (app.data.alpaca_credentials),
and broker/prop-firm accounts (app.live_deploy.live_settings) already have
their own modules with their own multi-field shapes; this one exists so
the remaining single-API-key-style integrations don't each need a
near-duplicate ~130-line module of their own. The unified /settings/
api-keys page (see app.web.server) reads from ALL FOUR modules and
presents them together, categorized, in one place -- that page is the
actual point of this consolidation, not this file by itself.

Same storage split as every sibling module: every field here is a bare
secret string, so every one of them goes to the OS keyring when available,
falling back to a lightly-obfuscated local file otherwise (same fallback
app.ai.ollama_settings/app.data.alpaca_credentials/app.live_deploy.
live_settings all use). Nothing non-secret lives in this file at all --
there's no host/model/enabled-style field for any of these four, just a
key each.

London Strategic Edge: stored as a plain opaque API key. UPGRADE (2026-09):
now backed by a real integration -- app.data.london_strategic_edge_source
fetches OHLCV candles with it (wired into the main Market Data page's "Or
fetch data from London Strategic Edge" card, same convention as the
Alpaca card next to it) and Test Connection performs a real, cheap
catalog lookup instead of reporting "not implemented".
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from app.data.storage import get_app_base_dir

SERVICE_NAME = "T58PropAlgoBacktester"

_FIELDS = ("fred_api_key", "openai_api_key", "claude_api_key", "london_strategic_edge_key")


@dataclass
class ApiKeysSettings:
    fred_api_key: str = ""
    openai_api_key: str = ""
    claude_api_key: str = ""
    london_strategic_edge_key: str = ""


def _config_dir() -> Path:
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def _fallback_path(field: str) -> Path:
    return _config_dir() / f"api_key_{field}.txt"


def save_settings(settings: ApiKeysSettings) -> None:
    """Each of the four fields is saved independently -- an empty string
    deletes that one key (keyring entry + fallback file) without
    disturbing the other three."""
    kr = _try_keyring()
    for field in _FIELDS:
        value = (getattr(settings, field) or "").strip()
        if kr is not None:
            try:
                if value:
                    kr.set_password(SERVICE_NAME, field, value)
                else:
                    kr.delete_password(SERVICE_NAME, field)
                _fallback_path(field).unlink(missing_ok=True)
                continue
            except Exception:
                pass  # fall through to the file-based fallback below
        if value:
            _fallback_path(field).write_text(_obfuscate(value), encoding="utf-8")
        else:
            _fallback_path(field).unlink(missing_ok=True)


def load_settings() -> ApiKeysSettings:
    """Never raises -- a missing or corrupt key just loads as an empty
    string for that one field."""
    kr = _try_keyring()
    values = {}
    for field in _FIELDS:
        value = ""
        if kr is not None:
            try:
                value = kr.get_password(SERVICE_NAME, field) or ""
            except Exception:
                value = ""
        if not value:
            fallback = _fallback_path(field)
            if fallback.exists():
                try:
                    value = _deobfuscate(fallback.read_text(encoding="utf-8"))
                except Exception:
                    value = ""
        values[field] = value
    return ApiKeysSettings(**values)
