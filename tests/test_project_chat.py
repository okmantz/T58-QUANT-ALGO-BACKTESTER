"""Tests for app.ai.project_chat -- prompt building and the fail-safe
paths of ProjectChatClient.chat, with no live Ollama server."""
from __future__ import annotations

import requests

from app.ai.ollama_settings import OllamaSettings
from app.ai.project_chat import (
    MAX_HISTORY_TURNS_IN_PROMPT, ProjectChatClient, build_system_prompt,
)


def _client(enabled=True):
    return ProjectChatClient(OllamaSettings(enabled=enabled, host="http://localhost:11434", model="m"))


def test_system_prompt_names_the_project_and_forbids_pretending_to_run_jobs():
    prompt = build_system_prompt("Falsification Kit Rebuild")
    assert "Falsification Kit Rebuild" in prompt
    assert "cannot start, stop, or configure" in prompt
    assert "Recent background jobs" not in prompt


def test_system_prompt_includes_activity_lines_when_given():
    prompt = build_system_prompt("P", ["Job abc (ES): running", "Job def (NQ): done"])
    assert "Recent background jobs" in prompt
    assert "Job abc (ES): running" in prompt
    assert "Job def (NQ): done" in prompt


def test_build_messages_trims_history_and_appends_user_turn_last():
    client = _client()
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}", "ts": 1.0}
               for i in range(MAX_HISTORY_TURNS_IN_PROMPT + 6)]
    messages = client._build_messages("SYS", "new question", history)

    assert messages[0] == {"role": "system", "content": "SYS"}
    assert messages[-1] == {"role": "user", "content": "new question"}
    assert len(messages) == MAX_HISTORY_TURNS_IN_PROMPT + 2
    assert messages[1]["content"] == "m6"  # oldest turns dropped
    assert all(set(m) == {"role", "content"} for m in messages)  # "ts" stripped


def test_chat_when_ollama_disabled_returns_error_without_network(monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("must not call the network when disabled")

    monkeypatch.setattr(requests, "post", no_network)
    reply, error = _client(enabled=False).chat("P", "hi")
    assert reply == "" and "isn't enabled" in error


def test_chat_success_returns_reply(monkeypatch):
    captured = {}

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "hello back"}}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"], captured["body"] = url, json
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    reply, error = _client().chat("P", "hi", history=[], activity_lines=["Job x (ES): done"])

    assert (reply, error) == ("hello back", None)
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["stream"] is False
    assert "Job x (ES): done" in captured["body"]["messages"][0]["content"]


def test_chat_connection_error_is_fail_safe(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(requests, "post", boom)
    reply, error = _client().chat("P", "hi")
    assert reply == "" and "Couldn't reach Ollama" in error


def test_chat_timeout_is_fail_safe(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(requests, "post", boom)
    reply, error = _client().chat("P", "hi")
    assert reply == "" and "didn't respond in time" in error
