"""Tests for app.accounts.api_keys and the Discord/Telegram extension to
app.web.notifications."""
from __future__ import annotations

from app.accounts.api_keys import ApiKeysSettings, load_settings, save_settings
from app.web.notifications import NotificationSettings, load_notification_settings, save_notification_settings


def test_api_keys_round_trip():
    save_settings(ApiKeysSettings(
        fred_api_key="fred-1", openai_api_key="sk-openai", claude_api_key="sk-claude",
        london_strategic_edge_key="lse-1",
    ))
    loaded = load_settings()
    assert loaded.fred_api_key == "fred-1"
    assert loaded.openai_api_key == "sk-openai"
    assert loaded.claude_api_key == "sk-claude"
    assert loaded.london_strategic_edge_key == "lse-1"


def test_api_keys_clearing_one_field_leaves_the_others_intact():
    save_settings(ApiKeysSettings(fred_api_key="a", openai_api_key="b", claude_api_key="c", london_strategic_edge_key="d"))
    save_settings(ApiKeysSettings(fred_api_key="", openai_api_key="b", claude_api_key="c", london_strategic_edge_key="d"))
    loaded = load_settings()
    assert loaded.fred_api_key == ""
    assert loaded.openai_api_key == "b"
    assert loaded.claude_api_key == "c"
    assert loaded.london_strategic_edge_key == "d"


def test_notification_settings_discord_and_telegram_round_trip():
    save_notification_settings(NotificationSettings(
        discord_webhook_url="https://discord.com/api/webhooks/1/abc",
        telegram_bot_token="123:ABC", telegram_chat_id="456",
    ))
    loaded = load_notification_settings()
    assert loaded.discord_webhook_url == "https://discord.com/api/webhooks/1/abc"
    assert loaded.telegram_bot_token == "123:ABC"
    assert loaded.telegram_chat_id == "456"
    assert loaded.discord_is_usable
    assert loaded.telegram_is_usable


def test_notification_settings_telegram_needs_both_token_and_chat_id():
    save_notification_settings(NotificationSettings(telegram_bot_token="123:ABC", telegram_chat_id=""))
    loaded = load_notification_settings()
    assert not loaded.telegram_is_usable

    save_notification_settings(NotificationSettings(telegram_bot_token="", telegram_chat_id="456"))
    loaded = load_notification_settings()
    assert not loaded.telegram_is_usable


def test_notify_job_finished_does_not_raise_with_discord_and_telegram_configured():
    """Best-effort delivery -- an unreachable/fake webhook or bot token
    must never raise into the caller, since job-completion paths call
    this unconditionally."""
    from app.web.notifications import notify_job_finished
    save_notification_settings(NotificationSettings(
        discord_webhook_url="https://discord.com/api/webhooks/0/fake",
        telegram_bot_token="0:FAKE", telegram_chat_id="0",
    ))
    notify_job_finished(None, "Test Job", "did a thing")  # must not raise
