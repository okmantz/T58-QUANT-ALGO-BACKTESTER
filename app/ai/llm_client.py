"""
LLM CLIENT ABSTRACTION (2026-10-07 discovery layer).

The discovery layer needs ONE narrow capability from a language model: turn a
trader's free-text idea into a JSON rule spec (app.discovery.rule_spec). It
must never need a particular vendor, and it must keep working with no model at
all. So callers depend on this tiny protocol:

    client.complete(prompt: str, *, system: str | None = None) -> str

Implementations:
  * NullClient      -- raises LLMUnavailable; callers fall back to the
                       deterministic keyword compiler.
  * ScriptedClient  -- returns canned replies (tests, offline demos).
  * OllamaTextClient-- the app's local Ollama server (app.ai.ollama_settings),
                       plain text completion over HTTP.

The model never executes anything: its text is parsed into a declarative spec
and every field is validated against whitelisted building blocks and numeric
bounds before a single bar is tested (see app.discovery.rule_spec).
"""
from __future__ import annotations

from typing import Protocol


class LLMUnavailable(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> str: ...


class NullClient:
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        raise LLMUnavailable("no language model configured")


class ScriptedClient:
    def __init__(self, replies):
        self._replies = list(replies) if not isinstance(replies, str) else [replies]
        self.calls: list[str] = []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(prompt)
        if not self._replies:
            raise LLMUnavailable("scripted client exhausted")
        return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]


class OllamaTextClient:
    def __init__(self, settings=None, timeout: int = 90):
        from app.ai import ollama_settings as _os
        self.settings = settings or _os.load_settings()
        self.timeout = timeout

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        s = self.settings
        if not getattr(s, "is_usable", False):
            raise LLMUnavailable("Ollama is not enabled in settings")
        try:
            import requests
            headers = {"Authorization": f"Bearer {s.api_key}"} if s.api_key else {}
            body = {"model": s.model, "prompt": prompt, "stream": False, "format": "json"}
            if system:
                body["system"] = system
            r = requests.post(s.host.rstrip("/") + "/api/generate", json=body, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return str(r.json().get("response", ""))
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"Ollama request failed: {exc}") from exc
