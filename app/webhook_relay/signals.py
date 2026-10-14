"""
Parse a TradingView alert body into one normalized ``Signal``.

TradingView sends whatever text is in the alert's Message box: JSON when the
message is valid JSON (Content-Type application/json), otherwise the raw text
as text/plain. Both are accepted. The recommended message (see
``pine_alert_message``) uses TradingView's strategy placeholders so a Pine
strategy's own entries, exits and reversals arrive without extra Pine code.

The relay thinks in TARGET POSITIONS (long / short / flat), not raw orders:
that makes duplicate and out-of-order alerts harmless (a second "long" while
already long is a no-op) and lets a reversal arrive as one message.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs

MAX_BODY_BYTES = 32 * 1024

LONG_WORDS = {"buy", "long", "enter_long", "entry_long", "open_long", "bull"}
SHORT_WORDS = {"sell", "short", "enter_short", "entry_short", "open_short", "bear"}
FLAT_WORDS = {"flat", "exit", "close", "close_all", "closeall", "exit_all", "exit_long", "exit_short",
              "close_long", "close_short", "cancel"}


class SignalError(ValueError):
    """The alert body could not be understood (the message says why)."""


@dataclass
class Signal:
    target: str                      # "long" | "short" | "flat"
    ticker: str                      # as TradingView sent it, e.g. "NQ1!" or "CME_MINI:NQ1!"
    quantity: float | None = None    # contracts/lots requested by the alert (None = use the relay default)
    price: float | None = None       # fill/close price TradingView reported
    stop_loss: float | None = None
    take_profit: float | None = None
    passphrase: str = ""
    token: str = ""
    sent_at: str = ""
    raw: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        return "|".join([self.target, self.ticker.upper(), str(self.quantity), str(self.price), self.sent_at])


def _num(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        text = str(value).strip().replace(",", "")
        if not text or text.lower() in ("nan", "none", "null") or "{{" in text:
            return None
        out = float(text)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _first(data: dict, *keys):
    lowered = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def _word(value) -> str:
    return re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())


def _decode(raw: bytes | str) -> dict:
    if isinstance(raw, bytes):
        if len(raw) > MAX_BODY_BYTES:
            raise SignalError("Alert body is too large.")
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw
    text = text.strip()
    if not text:
        raise SignalError("Empty alert body. Paste the T58 alert message into the TradingView alert's Message box.")
    if text[0] in "{[":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SignalError(f"Alert body looks like JSON but is not valid JSON ({exc.msg}).") from exc
        if not isinstance(data, dict):
            raise SignalError("Alert JSON must be an object like {\"ticker\": ..., \"action\": ...}.")
        return data
    if "=" in text and ("action" in text.lower() or "position" in text.lower()):
        return {k: v[0] for k, v in parse_qs(text.replace(",", "&").replace("\n", "&")).items()}
    # Bare word: "buy" / "sell" / "flat"
    return {"action": text.split()[0]}


def parse_alert(raw: bytes | str, default_ticker: str = "") -> Signal:
    data = _decode(raw)
    ticker = str(_first(data, "ticker", "symbol", "instrument", "syminfo.ticker") or default_ticker or "").strip()
    if "{{" in ticker:
        ticker = default_ticker
    if not ticker:
        raise SignalError("No ticker in the alert (expected a \"ticker\" field, e.g. {{ticker}}).")

    position = _word(_first(data, "position", "market_position", "strategy.market_position"))
    action = _word(_first(data, "action", "side", "order_action", "sentiment", "strategy.order.action"))
    # market_position (the position AFTER this order) is the most reliable
    # signal for a Pine strategy; fall back to the order action.
    word = position if position in LONG_WORDS | SHORT_WORDS | FLAT_WORDS else action
    if word in LONG_WORDS:
        target = "long"
    elif word in SHORT_WORDS:
        target = "short"
    elif word in FLAT_WORDS:
        target = "flat"
    else:
        raise SignalError(
            f"Could not tell what to do from action={action or '-'!r} position={position or '-'!r}. "
            "Use buy/sell/long/short/flat/exit."
        )
    # {{strategy.order.action}} = "sell" while market_position = "flat" is an exit, not a short:
    # position wins when it is present, which the branch above already guarantees.

    # For a non-flat target the size AFTER the order (strategy.position_size) is the right
    # quantity: a Pine reversal's order contracts are close + open combined (e.g. 2 for
    # long 1 -> short 1), which would double the new position.
    qty = None
    if target != "flat":
        qty = _num(_first(data, "position_size", "market_position_size", "strategy.position_size"))
    if qty is None or abs(qty) == 0:
        qty = _num(_first(data, "quantity", "qty", "contracts", "size", "strategy.order.contracts"))
    if qty is not None:
        qty = abs(qty) or None
    if target == "flat":
        qty = None

    return Signal(
        target=target, ticker=ticker, quantity=qty,
        price=_num(_first(data, "price", "close", "fill_price", "strategy.order.price")),
        stop_loss=_num(_first(data, "stop_loss", "sl", "stop")),
        take_profit=_num(_first(data, "take_profit", "tp", "target")),
        passphrase=str(_first(data, "passphrase", "password", "secret") or ""),
        token=str(_first(data, "token") or ""),
        sent_at=str(_first(data, "time", "timenow", "timestamp") or ""),
        raw={k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in data.items()
             if str(k).lower() not in ("passphrase", "password", "secret", "token")},
    )


def pine_alert_message(passphrase: str = "") -> str:
    """The Message to paste into a TradingView STRATEGY alert. Uses
    TradingView's placeholders, which it fills in when the alert fires."""
    body = [
        '"ticker": "{{ticker}}"',
        '"action": "{{strategy.order.action}}"',
        '"position": "{{strategy.market_position}}"',
        '"quantity": "{{strategy.order.contracts}}"',
        '"position_size": "{{strategy.position_size}}"',
        '"price": "{{strategy.order.price}}"',
        '"time": "{{timenow}}"',
    ]
    if passphrase:
        body.append(f'"passphrase": "{passphrase}"')
    return "{\n  " + ",\n  ".join(body) + "\n}"
