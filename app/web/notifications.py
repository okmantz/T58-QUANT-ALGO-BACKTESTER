"""
Best-effort "notify me when done" webhook helper.

Closes the gap flagged in WEB_PARITY_ROADMAP-adjacent notes: every
long-running job (Evolution Lab, Search Lab, Full Pipeline batch) needed a
human to come back and watch a progress page, with no email/SMS/webhook
integration anywhere in the app besides a static "join our Discord" link.

This does NOT attempt to be a full notification service (no email/SMS
provider, no stored credentials) -- it posts a plain JSON payload to a
webhook URL the user supplies at job-start time, which is exactly what a
Discord "Incoming Webhook", a Slack "Incoming Webhook", a Telegram bot
webhook proxy, or a personal endpoint (e.g. a Zapier/IFTTT catch hook)
already expects, with zero new third-party accounts or stored secrets on
T58's side. If the URL happens to be a Discord webhook, the payload also
includes a `content` field so it renders as a normal Discord message with
no further configuration.
"""
from __future__ import annotations

import base64
import json
import smtplib
import threading
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

import requests

from app.data.storage import get_app_base_dir

MAX_WEBHOOK_URL_LENGTH = 2048

SERVICE_NAME = "T58PropAlgoBacktester"
KEYRING_USERNAME = "notifications_smtp_password"


@dataclass
class NotificationSettings:
    """Email "notify me when a job finishes" settings, stored locally.

    Mirrors app.ai.ollama_settings' exact split between non-secret fields
    (plain local JSON) and the one genuinely secret field, smtp_password
    (OS keyring when available, lightly-obfuscated local file otherwise).
    Email is entirely optional -- everything here defaults to off/blank,
    and notify_job_finished() below is a no-op for email whenever
    email_enabled is False or the SMTP fields aren't filled in, exactly
    like send_job_notification() is already a no-op for a blank webhook
    URL.
    """

    notify_email: str = ""
    notify_phone: str = ""  # reserved for a future SMS-via-carrier-gateway path
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    email_enabled: bool = False

    @property
    def is_usable(self) -> bool:
        return bool(
            self.email_enabled and self.notify_email.strip() and self.smtp_host.strip()
            and self.smtp_from.strip(),
        )


def _config_dir() -> Path:
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _settings_path() -> Path:
    return _config_dir() / "notification_settings.json"


def _password_fallback_path() -> Path:
    return _config_dir() / "notification_smtp_password.txt"


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


def save_notification_settings(settings: NotificationSettings) -> None:
    """Persists everything except smtp_password to a plain local JSON
    file, and smtp_password preferentially to the OS keyring -- same
    split app.ai.ollama_settings uses for its one secret field."""
    payload = {
        "notify_email": (settings.notify_email or "").strip(),
        "notify_phone": (settings.notify_phone or "").strip(),
        "smtp_host": (settings.smtp_host or "").strip(),
        "smtp_port": int(settings.smtp_port or 587),
        "smtp_username": (settings.smtp_username or "").strip(),
        "smtp_from": (settings.smtp_from or "").strip(),
        "email_enabled": bool(settings.email_enabled),
    }
    _settings_path().write_text(json.dumps(payload), encoding="utf-8")

    password = (settings.smtp_password or "").strip()
    kr = _try_keyring()
    if kr is not None:
        try:
            if password:
                kr.set_password(SERVICE_NAME, KEYRING_USERNAME, password)
            else:
                kr.delete_password(SERVICE_NAME, KEYRING_USERNAME)
            _password_fallback_path().unlink(missing_ok=True)
            return
        except Exception:
            pass  # fall through to the file-based fallback below

    if password:
        _password_fallback_path().write_text(_obfuscate(password), encoding="utf-8")
    else:
        _password_fallback_path().unlink(missing_ok=True)


def load_notification_settings() -> NotificationSettings:
    """Returns saved settings, or the (disabled) defaults if nothing has
    been saved yet -- never raises, so callers never need a try/except
    just to read config."""
    path = _settings_path()
    notify_email = notify_phone = smtp_host = smtp_username = smtp_from = ""
    smtp_port = 587
    email_enabled = False
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            notify_email = data.get("notify_email") or ""
            notify_phone = data.get("notify_phone") or ""
            smtp_host = data.get("smtp_host") or ""
            smtp_port = int(data.get("smtp_port") or 587)
            smtp_username = data.get("smtp_username") or ""
            smtp_from = data.get("smtp_from") or ""
            email_enabled = bool(data.get("email_enabled", False))
        except Exception:
            pass

    smtp_password = ""
    kr = _try_keyring()
    if kr is not None:
        try:
            smtp_password = kr.get_password(SERVICE_NAME, KEYRING_USERNAME) or ""
        except Exception:
            smtp_password = ""
    if not smtp_password:
        fallback = _password_fallback_path()
        if fallback.exists():
            try:
                smtp_password = _deobfuscate(fallback.read_text(encoding="utf-8"))
            except Exception:
                smtp_password = ""

    return NotificationSettings(
        notify_email=notify_email, notify_phone=notify_phone, smtp_host=smtp_host,
        smtp_port=smtp_port, smtp_username=smtp_username, smtp_password=smtp_password,
        smtp_from=smtp_from, email_enabled=email_enabled,
    )


def _send_email_notification(settings: NotificationSettings, job_kind: str, summary: str, job_url: str | None) -> None:
    """Fires an SMTP email in a background thread -- never blocks or
    raises into the caller, matching send_job_notification's webhook
    behavior exactly. A no-op whenever email notifications aren't fully
    configured (see NotificationSettings.is_usable)."""
    if not settings.is_usable:
        return

    msg = EmailMessage()
    msg["Subject"] = f"T58 -- {job_kind} finished"
    msg["From"] = settings.smtp_from.strip()
    msg["To"] = settings.notify_email.strip()
    body = f"{job_kind} finished: {summary}"
    if job_url:
        body += f"\n\n{job_url}"
    msg.set_content(body)

    def _send() -> None:
        try:
            with smtplib.SMTP(settings.smtp_host.strip(), int(settings.smtp_port or 587), timeout=15) as server:
                server.starttls()
                if settings.smtp_username.strip():
                    server.login(settings.smtp_username.strip(), settings.smtp_password)
                server.send_message(msg)
        except Exception:
            # Best-effort only -- an email failure must never surface as
            # a job failure, and there's no one left waiting on this
            # thread's return value.
            pass

    threading.Thread(target=_send, daemon=True).start()


def notify_job_finished(webhook_url: str | None, job_kind: str, summary: str, job_url: str | None = None) -> None:
    """Single call site every job-completion path uses -- fires BOTH the
    per-job webhook (if a URL was supplied for this run) and the
    account-wide email notification (if configured and enabled in
    Settings -> Notification settings). Each channel is independently
    optional and independently best-effort; a failure or absence of one
    never affects the other."""
    send_job_notification(webhook_url, job_kind, summary, job_url=job_url)
    _send_email_notification(load_notification_settings(), job_kind, summary, job_url)


def send_job_notification(webhook_url: str | None, job_kind: str, summary: str, job_url: str | None = None) -> None:
    """Fires a webhook POST in a background thread -- never blocks or
    raises into the caller. A bad/unreachable URL should never take down
    (or even delay) the job it's reporting on; failures are simply
    swallowed here since there's no user-facing surface waiting on this
    call's result.

    webhook_url: None/empty -- a no-op, so every call site can call this
    unconditionally without an `if notify_url:` guard of its own.
    job_kind: short label, e.g. "Full Pipeline (batch)", "Evolution Lab".
    summary: one or two plain-English sentences of what finished.
    job_url: optional absolute or relative link back to the job's own
    status/report page.
    """
    if not webhook_url or not webhook_url.strip():
        return
    url = webhook_url.strip()
    if len(url) > MAX_WEBHOOK_URL_LENGTH or not (url.startswith("http://") or url.startswith("https://")):
        return

    message = f"T58 -- {job_kind} finished: {summary}"
    if job_url:
        message += f"\n{job_url}"
    payload = {
        "content": message,        # Discord/Slack-compatible field name
        "text": message,           # Slack-compatible alternate field name
        "job_kind": job_kind,
        "summary": summary,
        "job_url": job_url,
    }

    def _post() -> None:
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception:
            # Best-effort only -- a webhook failure must never surface as
            # a job failure, and there's no one left waiting on this
            # thread's return value.
            pass

    threading.Thread(target=_post, daemon=True).start()
