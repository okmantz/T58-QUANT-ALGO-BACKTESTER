"""AI provider routing: Ollama default; Claude/OpenAI used only when selected AND keyed."""
import dataclasses

import pytest

from app.ai import ollama_settings as os_, ollama_transport as tr


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    os_._invalidate_cache()
    monkeypatch.setattr(os_, "_config_dir", lambda: tmp_path)
    monkeypatch.setattr(os_, "_try_keyring", lambda: None)
    keys = {"claude_api_key": "", "openai_api_key": ""}
    monkeypatch.setattr(os_, "remote_api_key", lambda p: keys["claude_api_key" if p == "claude" else "openai_api_key"])
    yield keys
    os_._invalidate_cache()


def test_default_is_ollama():
    assert os_.saved_provider() == "ollama"
    assert os_.effective_provider(os_.OllamaSettings()) == "ollama"


def test_selected_provider_without_key_falls_back_to_ollama(_isolated):
    os_.save_settings(os_.OllamaSettings(provider="claude"))
    assert os_.saved_provider() == "claude"
    assert os_.effective_provider(None) == "ollama"


def test_selected_provider_with_key_routes_remote_and_is_usable(_isolated, monkeypatch):
    _isolated["claude_api_key"] = "sk-ant-x"
    os_.save_settings(os_.OllamaSettings(enabled=False, provider="claude"))
    s = os_.load_settings()
    assert s.is_usable  # usable even though local Ollama is disabled
    seen = {}

    def fake(provider, messages, **kw):
        seen.update(provider=provider, messages=messages, kw=kw)
        return "hello"

    monkeypatch.setattr(tr, "_remote_complete", fake)
    assert tr.generate(s, "hi", system="sys", format_json=True) == "hello"
    assert seen["provider"] == "claude" and seen["kw"]["format_json"] is True
    assert tr.chat(s, [{"role": "user", "content": "x"}]) == "hello"


def test_fresh_settings_object_still_honors_saved_choice(_isolated):
    _isolated["openai_api_key"] = "sk-x"
    os_.save_settings(os_.OllamaSettings(provider="openai"))
    assert os_.effective_provider(os_.OllamaSettings(enabled=True)) == "openai"


def test_model_overrides_persist(_isolated):
    os_.save_remote_models(claude_model="claude-x")
    os_.save_settings(os_.OllamaSettings(provider="claude"))
    assert os_.saved_remote_model("claude") == "claude-x"
    assert os_.saved_remote_model("openai") == "gpt-4o-mini"
