"""Persistent settings for the optional local-Ollama AI trading assistant.

Off by default, everywhere. Nothing in this app calls Ollama unless
`OllamaSettings.enabled` is True AND the person has explicitly turned it on
in the UI -- see app.ai.ollama_client and the "AI Assist" section on the
Full Pipeline tab.

Mirrors app.data.alpaca_credentials's storage pattern exactly (OS keyring
when available, lightly-obfuscated local JSON fallback otherwise) for the
one field that's actually secret -- `api_key`, only relevant for a
remote/proxied Ollama endpoint that sits behind auth; a genuinely local
Ollama install (the default, and the easy-setup path this is built for)
needs no key at all. host/model/enabled aren't secrets, so they're stored
in a small plain JSON file next to the credentials file.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from app.data.storage import get_app_base_dir

SERVICE_NAME = "T58PropAlgoBacktester"
KEYRING_USERNAME = "ollama"

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"
# Chart/trade screenshot analysis needs a vision-capable model -- a plain
# text model like the DEFAULT_MODEL above can't see the image at all.
# llava is the smallest/most commonly-pulled Ollama vision model; anything
# multimodal Ollama supports (e.g. llama3.2-vision, bakllava) also works.
DEFAULT_VISION_MODEL = "llava"

# How long Ollama keeps a model resident in memory after a request, for
# every interactive/frequently-called path (chat, screenshot analysis,
# daily brief, watchlist, outlook, research agent/loop, embeddings).
# Ollama's own server-side default is 5 minutes -- fine for a one-off
# generation, but this app's AI Assistant/Research Agent/Research Loop are
# used in bursts of several calls a few minutes apart (a user reading a
# reply and typing the next question, or a research loop's own internal
# multi-step calls), so the model was frequently unloading and reloading
# from disk between calls -- the actual cause of "the AI model takes a
# while" on anything past the very first request of a session. Passing
# this explicitly on every such call keeps the model warm through a
# realistic gap between turns without pinning it forever (see
# app.ai.strategy_generator.KEEP_ALIVE for the one deliberately-shorter
# exception, a single one-shot code-generation call).
INTERACTIVE_KEEP_ALIVE = "30m"


@dataclass
class OllamaSettings:
    enabled: bool = False
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    api_key: str = ""  # optional -- only needed for a remote Ollama behind auth
    vision_model: str = DEFAULT_VISION_MODEL  # used only for screenshot analysis (chart/trade images)

    @property
    def is_usable(self) -> bool:
        return self.enabled and bool(self.host.strip())


def _config_dir() -> Path:
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _settings_path() -> Path:
    return _config_dir() / "ollama_settings.json"


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


def save_settings(settings: OllamaSettings) -> None:
    """Persists enabled/host/model to a plain local JSON file, and api_key
    (if any) preferentially to the OS keyring, matching
    app.data.alpaca_credentials's exact split between non-secret and
    secret fields."""
    payload = {
        "enabled": bool(settings.enabled),
        "host": (settings.host or DEFAULT_HOST).strip(),
        "model": (settings.model or DEFAULT_MODEL).strip(),
        "vision_model": (settings.vision_model or DEFAULT_VISION_MODEL).strip(),
    }
    _settings_path().write_text(json.dumps(payload), encoding="utf-8")

    api_key = (settings.api_key or "").strip()
    kr = _try_keyring()
    if kr is not None:
        try:
            if api_key:
                kr.set_password(SERVICE_NAME, KEYRING_USERNAME, api_key)
            else:
                kr.delete_password(SERVICE_NAME, KEYRING_USERNAME)
            _api_key_fallback_path().unlink(missing_ok=True)
            return
        except Exception:
            pass  # fall through to the file-based fallback below

    if api_key:
        _api_key_fallback_path().write_text(_obfuscate(api_key), encoding="utf-8")
    else:
        _api_key_fallback_path().unlink(missing_ok=True)


def _api_key_fallback_path() -> Path:
    return _config_dir() / "ollama_api_key.txt"


def load_settings() -> OllamaSettings:
    """Returns saved settings, or the (disabled) defaults if nothing has
    been saved yet -- never raises, so callers never need a try/except
    just to read config."""
    path = _settings_path()
    enabled, host, model, vision_model = False, DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_VISION_MODEL
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            enabled = bool(data.get("enabled", False))
            host = data.get("host") or DEFAULT_HOST
            model = data.get("model") or DEFAULT_MODEL
            vision_model = data.get("vision_model") or DEFAULT_VISION_MODEL
        except Exception:
            pass

    api_key = ""
    kr = _try_keyring()
    if kr is not None:
        try:
            api_key = kr.get_password(SERVICE_NAME, KEYRING_USERNAME) or ""
        except Exception:
            api_key = ""
    if not api_key:
        fallback = _api_key_fallback_path()
        if fallback.exists():
            try:
                api_key = _deobfuscate(fallback.read_text(encoding="utf-8"))
            except Exception:
                api_key = ""

    return OllamaSettings(enabled=enabled, host=host, model=model, api_key=api_key, vision_model=vision_model)
