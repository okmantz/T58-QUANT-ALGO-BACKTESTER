"""
Webhook Relay routes -- TradingView alert -> T58 forward test.

Two very different kinds of route live here:

  * ``POST /webhook/tradingview/<token>`` is called by TradingView's servers,
    not by a browser, so it is exempt from the license redirect, the account
    lock and CSRF (see the exempt prefixes in app/web/server.py). The secret
    token in the URL is its credential; it also refuses to act unless the app
    is licensed, answers quickly (TradingView drops a webhook after ~3
    seconds, so the demo-broker leg runs on a background thread), and is
    rate limited and size capped.
  * Everything under ``/webhook-relay`` is the normal gated setup page.

The engine itself is app/webhook_relay/ -- this file is only HTTP.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from flask import Blueprint, current_app, jsonify, render_template, request

from app.webhook_relay import config as relay_config
from app.webhook_relay import engine, journal
from app.webhook_relay.paper import PaperLedger
from app.webhook_relay.signals import MAX_BODY_BYTES, SignalError, parse_alert, pine_alert_message

webhook_relay_bp = Blueprint("webhook_relay", __name__)

_RATE: dict[str, deque] = defaultdict(deque)
RATE_LIMIT_PER_MINUTE = 120


def _rate_limited(key: str) -> bool:
    now = time.monotonic()
    window = _RATE[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        return True
    window.append(now)
    return False


# ---------------------------------------------------------------- public endpoint

@webhook_relay_bp.route("/webhook/tradingview/<token>", methods=["POST"])
def tradingview_webhook(token: str):
    license_ok = current_app.config.get("t58_license_check", lambda: True)
    if not license_ok():
        return jsonify({"ok": False, "error": "T58 is not activated on this machine."}), 403
    relay = relay_config.find_by_token(token)
    if relay is None:
        return jsonify({"ok": False, "error": "Unknown webhook URL."}), 404
    if _rate_limited(relay.id):
        return jsonify({"ok": False, "error": "Too many alerts per minute."}), 429
    if request.content_length is not None and request.content_length > MAX_BODY_BYTES:
        return jsonify({"ok": False, "error": "Alert body too large."}), 413
    body = request.get_data(cache=False)[: MAX_BODY_BYTES + 1]
    try:
        signal = parse_alert(body)
    except SignalError as exc:
        journal.record(relay.id, status="error", message=str(exc), ticker="", root="", target="", price=None,
                       qty=None, source="tradingview", paper=None, broker=None,
                       raw=body[:300].decode("utf-8", errors="replace"))
        return jsonify({"ok": False, "error": str(exc)}), 400
    result = engine.handle(relay, signal)
    code = 200 if result["ok"] else (403 if result["status"] == "rejected" else 422)
    return jsonify({"ok": result["ok"], "status": result["status"], "message": result["message"]}), code


@webhook_relay_bp.route("/webhook/tradingview/<token>/ping", methods=["GET"])
def tradingview_ping(token: str):
    relay = relay_config.find_by_token(token)
    if relay is None:
        return jsonify({"ok": False, "error": "Unknown webhook URL."}), 404
    return jsonify({"ok": True, "relay": relay.name, "enabled": relay.enabled})


# ---------------------------------------------------------------- setup page (gated)

def _demo_accounts() -> list[dict]:
    try:
        from app.live_deploy.connect_flow import account_is_live
        from app.live_deploy.live_settings import load_accounts

        rows = []
        for a in load_accounts():
            live = account_is_live(a)
            rows.append({"id": a.id, "label": f"{a.nickname} ({a.platform})", "live": live,
                         "env": "LIVE" if live else ("demo" if live is False else "unknown")})
        return rows
    except Exception:  # noqa: BLE001 -- the page must open even if accounts can't be read
        return []


def _relay_view(relay, public_url: str) -> dict:
    d = relay.to_public()
    d["url"] = f"{public_url or request.host_url.rstrip('/')}/webhook/tradingview/{relay.token}"
    d["alert_message"] = pine_alert_message(relay.passphrase)
    d["symbol_map_text"] = "\n".join(f"{k}={v}" for k, v in relay.symbol_map.items())
    d["allowed_text"] = ", ".join(relay.allowed_tickers)
    return d


@webhook_relay_bp.route("/webhook-relay")
def webhook_relay_page():
    public_url = relay_config.get_public_url()
    relays = [_relay_view(r, public_url) for r in relay_config.list_relays()]
    return render_template(
        "webhook_relay.html", active_page="webhook_relay", relays=relays, public_url=public_url,
        local_url=request.host_url.rstrip("/"), host=request.host, accounts=_demo_accounts(),
    )


def _state(relay_id: str):
    relay = relay_config.get_relay(relay_id)
    if relay is None:
        return None
    return {
        "ok": True,
        "enabled": relay.enabled,
        "paper": PaperLedger(relay.id).snapshot(),
        "signals": journal.recent(relay.id, 40),
    }


@webhook_relay_bp.route("/webhook-relay/relays/<relay_id>/state.json")
def relay_state(relay_id: str):
    state = _state(relay_id)
    return (jsonify(state), 200) if state else (jsonify({"ok": False, "error": "Unknown relay."}), 404)


@webhook_relay_bp.route("/webhook-relay/public-url", methods=["POST"])
def relay_public_url():
    try:
        url = relay_config.set_public_url(request.form.get("public_url", ""))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "public_url": url})


@webhook_relay_bp.route("/webhook-relay/relays", methods=["POST"])
def relay_create():
    relay = relay_config.create_relay({"name": request.form.get("name", "")})
    return jsonify({"ok": True, "id": relay.id})


@webhook_relay_bp.route("/webhook-relay/relays/<relay_id>/save", methods=["POST"])
def relay_save(relay_id: str):
    values = {k: v for k, v in request.form.items()}
    # Unchecked checkboxes are absent from a form post: treat them as off.
    for flag in ("enabled", "allow_short", "force_default_qty", "confirm_demo"):
        values.setdefault(flag, "0")
    relay = relay_config.update_relay(relay_id, values)
    if relay is None:
        return jsonify({"ok": False, "error": "Unknown relay."}), 404
    return jsonify({"ok": True})


@webhook_relay_bp.route("/webhook-relay/relays/<relay_id>/rotate", methods=["POST"])
def relay_rotate(relay_id: str):
    relay = relay_config.rotate_token(relay_id)
    if relay is None:
        return jsonify({"ok": False, "error": "Unknown relay."}), 404
    return jsonify({"ok": True})


@webhook_relay_bp.route("/webhook-relay/relays/<relay_id>/delete", methods=["POST"])
def relay_delete(relay_id: str):
    if not relay_config.delete_relay(relay_id):
        return jsonify({"ok": False, "error": "Unknown relay."}), 404
    PaperLedger(relay_id).reset()
    journal.clear(relay_id)
    return jsonify({"ok": True})


@webhook_relay_bp.route("/webhook-relay/relays/<relay_id>/reset-paper", methods=["POST"])
def relay_reset_paper(relay_id: str):
    if relay_config.get_relay(relay_id) is None:
        return jsonify({"ok": False, "error": "Unknown relay."}), 404
    PaperLedger(relay_id).reset()
    journal.clear(relay_id)
    return jsonify({"ok": True})


@webhook_relay_bp.route("/webhook-relay/relays/<relay_id>/test", methods=["POST"])
def relay_test(relay_id: str):
    """Fire a test alert through the exact same path a real one takes
    (same parser, rules, paper ledger and broker leg), so the person can
    check the whole chain before touching TradingView."""
    relay = relay_config.get_relay(relay_id)
    if relay is None:
        return jsonify({"ok": False, "error": "Unknown relay."}), 404
    import json as _json

    payload = {
        "ticker": request.form.get("ticker", "NQ1!") or "NQ1!",
        "position": request.form.get("target", "long"),
        "price": request.form.get("price", ""),
        "time": str(time.time()),
    }
    qty = request.form.get("quantity", "").strip()
    if qty:
        payload["position_size"] = qty
    if relay.passphrase:
        payload["passphrase"] = relay.passphrase
    try:
        signal = parse_alert(_json.dumps(payload))
    except SignalError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    result = engine.handle(relay, signal, source="test", wait_for_broker=True)
    return jsonify({"ok": result["ok"], "status": result["status"], "message": result["message"],
                    "broker": result.get("broker")})
