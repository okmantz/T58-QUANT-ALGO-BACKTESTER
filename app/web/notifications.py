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

import threading

import requests

MAX_WEBHOOK_URL_LENGTH = 2048


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
