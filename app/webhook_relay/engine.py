"""
The relay engine: one alert in, one position change out.

    TradingView alert -> parse (signals.py) -> checks -> paper ledger (always)
                                                       -> demo broker (optional, background)

Safety rules baked in (not settings):
  * A broker leg only ever runs against a DEMO/practice account: cTrader,
    Tradovate and TradeLocker accounts report their environment and a live one
    is refused; MT5 and DXtrade can't report it, so the user must tick
    "this is a demo account" on the relay first.
  * Quantity is capped by the relay's max_qty whatever the alert says.
  * Duplicate alerts inside a short window are ignored.
Quantity is sent to the broker in its OWN units (whole contracts on Tradovate,
lots on MT5/cTrader/TradeLocker/DXtrade) -- the same number the alert carries.
"""
from __future__ import annotations

import hmac
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.webhook_relay import journal
from app.webhook_relay.config import RelayConfig, root_symbol
from app.webhook_relay.paper import PaperLedger
from app.webhook_relay.signals import Signal

# One worker: broker legs for successive alerts stay in the order they arrived.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t58-relay")
_DEDUPE: dict[str, float] = {}
_DEDUPE_LOCK = threading.Lock()
_ADAPTERS: dict[str, object] = {}
_ADAPTER_LOCK = threading.Lock()


def _is_duplicate(relay: RelayConfig, signal: Signal) -> bool:
    if relay.dedupe_seconds <= 0:
        return False
    key = f"{relay.id}|{signal.fingerprint()}"
    now = time.monotonic()
    with _DEDUPE_LOCK:
        for k in [k for k, t in _DEDUPE.items() if now - t > 3600]:
            del _DEDUPE[k]
        last = _DEDUPE.get(key)
        _DEDUPE[key] = now
        return last is not None and now - last < relay.dedupe_seconds


def _decide_qty(relay: RelayConfig, signal: Signal) -> tuple[float, str]:
    qty = relay.default_qty if (relay.force_default_qty or signal.quantity is None) else signal.quantity
    note = ""
    if qty > relay.max_qty:
        note = f"quantity {qty:g} capped to max {relay.max_qty:g}"
        qty = relay.max_qty
    return qty, note


def handle(relay: RelayConfig, signal: Signal, *, wait_for_broker: bool = False, source: str = "tradingview") -> dict:
    """Process one signal. Returns {"ok", "status", "message", "id", ...}.
    status: executed | no_change | duplicate | rejected | error."""
    root = root_symbol(signal.ticker)
    base = {"ticker": signal.ticker, "root": root, "target": signal.target,
            "price": signal.price, "source": source}

    def done(status: str, message: str, ok: bool, **extra) -> dict:
        rec_id = journal.record(relay.id, status=status, message=message, qty=extra.get("qty", signal.quantity),
                                paper=extra.get("paper"), broker=extra.get("broker"), **base)
        return {"ok": ok, "status": status, "message": message, "id": rec_id, **{k: v for k, v in extra.items() if k != "broker"}}

    if not relay.enabled:
        return done("rejected", "This relay is switched off.", False)
    if relay.passphrase and not hmac.compare_digest(relay.passphrase.encode(), (signal.passphrase or "").encode()):
        return done("rejected", "Wrong or missing passphrase in the alert.", False)
    if relay.allowed_tickers and root not in relay.allowed_tickers:
        return done("rejected", f"{root} is not in this relay's allowed tickers ({', '.join(relay.allowed_tickers)}).", False)
    if signal.target == "short" and not relay.allow_short:
        return done("rejected", "Short signals are switched off for this relay.", False)
    if _is_duplicate(relay, signal):
        return done("duplicate", "Identical alert received a moment ago; ignored.", True)

    qty, note = _decide_qty(relay, signal) if signal.target != "flat" else (0.0, "")
    ledger = PaperLedger(relay.id)
    try:
        actions = ledger.apply(root, signal.target, qty, signal.price)
    except ValueError as exc:
        return done("error", str(exc), False)

    if not actions:
        message = f"Already {signal.target}; nothing to do." if signal.target != "flat" else "Already flat; nothing to do."
        status = "no_change"
    else:
        message = "; ".join(_describe(a) for a in actions) + (f" ({note})" if note else "")
        status = "executed"

    broker_pending = bool(relay.broker_account_id) and status == "executed"
    rec = done(status, message, True, qty=qty or None, paper=actions,
               broker={"state": "queued"} if broker_pending else None)
    if broker_pending:
        def job():
            result = _broker_leg(relay, root, signal.target, qty)
            journal.patch(relay.id, rec["id"], broker=result)
            return result
        future = _EXECUTOR.submit(job)
        if wait_for_broker:
            rec["broker"] = future.result(timeout=60)
    return rec


def _describe(a: dict) -> str:
    if a["action"] == "open":
        return f"opened {a['side']} {a['qty']:g} {a['root']} @ {a['price']:g} (paper)"
    if a["action"] == "add":
        return f"added {a['qty']:g} {a['root']} @ {a['price']:g} (paper)"
    verb = "closed" if a["action"] == "close" else "reduced"
    return f"{verb} {a['side']} {a['qty']:g} {a['root']} @ {a['exit']:g}, P&L ${a['pnl']:,.2f} (paper)"


# ---- optional demo-broker leg ------------------------------------------------------

def _adapter_for(account):
    from app.live_deploy.broker_registry import build_adapter

    with _ADAPTER_LOCK:
        adapter = _ADAPTERS.get(account.id)
        if adapter is None:
            adapter = build_adapter(account)
            _ADAPTERS[account.id] = adapter
        return adapter


def _drop_adapter(account_id: str) -> None:
    with _ADAPTER_LOCK:
        adapter = _ADAPTERS.pop(account_id, None)
    try:
        if adapter is not None:
            adapter.disconnect()
    except Exception:  # noqa: BLE001
        pass


def check_demo_account(relay: RelayConfig):
    """(account, error). error is None when the relay may trade this account."""
    from app.live_deploy.connect_flow import account_is_live
    from app.live_deploy.live_settings import load_accounts

    account = next((a for a in load_accounts() if a.id == relay.broker_account_id), None)
    if account is None:
        return None, "The saved broker account for this relay no longer exists (Deploy Live -> accounts)."
    live = account_is_live(account)
    if live is True:
        return account, ("This account is set to LIVE. The webhook relay only trades demo/practice accounts; "
                         "switch the account to its demo environment in Deploy Live.")
    if live is None and not relay.confirm_demo:
        return account, (f"{account.platform} can't tell us if this is a demo account. Tick \"This is a demo/practice "
                         "account\" on the relay to allow it.")
    return account, None


def _broker_leg(relay: RelayConfig, root: str, target: str, qty: float) -> dict:
    account, error = check_demo_account(relay)
    if error:
        return {"state": "refused", "ok": False, "message": error}
    symbol = relay.symbol_map.get(root, root)
    try:
        adapter = _adapter_for(account)
        conn = adapter.ensure_connected()
        if not conn.ok:
            _drop_adapter(account.id)
            return {"state": "failed", "ok": False, "message": f"Broker connection failed: {conn.message}"}
        orders: list[str] = []
        positions = [p for p in adapter.get_open_positions(symbol)]
        want = {"long": 1, "short": -1}.get(target, 0)
        current = sum(p.volume * p.direction for p in positions)
        wrong_side = [p for p in positions if want == 0 or p.direction != want]
        for p in wrong_side:
            r = adapter.close_position(p.ticket, comment="T58 relay close")
            orders.append(f"close {p.volume:g}: {'ok' if r.ok else r.message}")
            if not r.ok:
                return {"state": "failed", "ok": False, "message": "; ".join(orders)}
        if want != 0:
            have = sum(p.volume for p in positions if p.direction == want)
            if have > qty + 1e-9:        # reduce = close and reopen at the new size
                for p in [p for p in positions if p.direction == want]:
                    r = adapter.close_position(p.ticket, comment="T58 relay resize")
                    orders.append(f"close {p.volume:g}: {'ok' if r.ok else r.message}")
                have = 0.0
            need = qty - have
            if need > 1e-9:
                r = adapter.place_market_order(symbol, want, need, comment="T58 relay")
                orders.append(f"{'buy' if want == 1 else 'sell'} {need:g} {symbol}: {'ok' if r.ok else r.message}")
                if not r.ok:
                    return {"state": "failed", "ok": False, "message": "; ".join(orders)}
        return {"state": "sent", "ok": True, "message": "; ".join(orders) or "Broker already in the target position.",
                "account": account.nickname, "symbol": symbol}
    except Exception as exc:  # noqa: BLE001 -- never let a broker error escape the worker thread
        _drop_adapter(account.id)
        return {"state": "failed", "ok": False, "message": f"{type(exc).__name__}: {exc}"}
