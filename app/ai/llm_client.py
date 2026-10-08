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
    def __init__(self, settings=None, timeout: int = 90, json_mode: bool = True):
        from app.ai import ollama_settings as _os
        self.settings = settings or _os.load_settings()
        self.timeout = timeout
        self.json_mode = json_mode

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        s = self.settings
        if not getattr(s, "is_usable", False):
            raise LLMUnavailable("Ollama is not enabled in settings")
        try:
            import requests
            headers = {"Authorization": f"Bearer {s.api_key}"} if s.api_key else {}
            body = {"model": s.model, "prompt": prompt, "stream": False}
            if self.json_mode:
                body["format"] = "json"
            if system:
                body["system"] = system
            r = requests.post(s.host.rstrip("/") + "/api/generate", json=body, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return str(r.json().get("response", ""))
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"Ollama request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Hosted backends using the API keys already stored by Settings (app.accounts.api_keys)
# ---------------------------------------------------------------------------
class AnthropicClient:
    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5", timeout: int = 120, max_tokens: int = 4096):
        self.api_key, self.model, self.timeout, self.max_tokens = api_key, model, timeout, max_tokens

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        import requests
        body = {"model": self.model, "max_tokens": self.max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=self.timeout,
                              headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"Anthropic request failed: {exc}") from exc


class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: int = 120):
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        import requests
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", json={"model": self.model, "messages": msgs},
                              headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout)
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"])
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"OpenAI request failed: {exc}") from exc


def stored_keys():
    try:
        from app.accounts.api_keys import load_settings
        return load_settings()
    except Exception:  # noqa: BLE001
        return None


def remote_client():
    """Anthropic if a Claude key is stored, else OpenAI if an OpenAI key is stored, else None."""
    k = stored_keys()
    if k is None:
        return None
    if getattr(k, "claude_api_key", ""):
        return AnthropicClient(k.claude_api_key)
    if getattr(k, "openai_api_key", ""):
        return OpenAIClient(k.openai_api_key)
    return None


def preferred_client(ollama_settings=None):
    """Local Ollama when enabled (private, free), else a hosted model with the stored key, else NullClient."""
    try:
        if ollama_settings is None:
            from app.ai import ollama_settings as _os
            ollama_settings = _os.load_settings()
        if getattr(ollama_settings, "is_usable", False):
            return OllamaTextClient(ollama_settings)
    except Exception:  # noqa: BLE001
        pass
    return remote_client() or NullClient()


def remote_available() -> bool:
    return remote_client() is not None


def complete_text(prompt: str, *, system: str | None = None, ollama_settings=None) -> tuple[str | None, str | None]:
    """(text, None) or (None, error). Never raises."""
    c = preferred_client(ollama_settings)
    try:
        return c.complete(prompt, system=system), None
    except LLMUnavailable as exc:
        return None, str(exc)
