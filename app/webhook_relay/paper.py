"""
T58 paper account for the webhook relay -- the forward test itself.

Every accepted alert moves a simulated position at the price TradingView
reported, so a Pine strategy gets a live, out-of-sample track record without
risking anything and without a broker. Costs use the instrument's default
round-turn commission from app.data.instrument_specs; there is NO slippage
model (fills are at the alert's price), so treat results as optimistic by
roughly a tick or two per trade.

State is one small JSON file per relay under data/webhook_relay/.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.data.storage import get_app_base_dir

_LOCK = threading.RLock()
MAX_CLOSED_TRADES = 5000


def relay_dir() -> Path:
    d = get_app_base_dir() / "data" / "webhook_relay"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def instrument_economics(root: str) -> tuple[float, float, str]:
    """(dollars per 1.0 price move per contract, commission per round-turn
    contract, label). Unknown symbols fall back to 1.0 / 0.0 -- P&L is then
    in raw price units, and the label says so."""
    from app.data.instrument_specs import get_instrument_spec

    spec = get_instrument_spec(root)
    if spec is not None and spec.tick_size:
        return spec.tick_value / spec.tick_size, float(spec.default_commission_round_turn or 0.0), spec.description
    return 1.0, 0.0, "price units (symbol not in the instrument table)"


class PaperLedger:
    def __init__(self, relay_id: str):
        self.relay_id = relay_id
        self.path = relay_dir() / f"{relay_id}_paper.json"

    # ---- persistence
    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            data.setdefault("positions", {})
            data.setdefault("closed", [])
            data.setdefault("last_price", {})
            return data
        except Exception:  # noqa: BLE001
            return {"positions": {}, "closed": [], "last_price": {}, "since": _now()}

    def _save(self, state: dict) -> None:
        state["closed"] = state["closed"][-MAX_CLOSED_TRADES:]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(self.path)

    def reset(self) -> None:
        with _LOCK:
            self.path.unlink(missing_ok=True)

    # ---- trading
    @staticmethod
    def _close_leg(state: dict, root: str, pos: dict, qty: float, price: float, why: str) -> dict:
        per_point, commission, _ = instrument_economics(root)
        direction = 1 if pos["side"] == "long" else -1
        gross = (price - pos["entry"]) * direction * qty * per_point
        cost = commission * qty
        trade = {
            "root": root, "side": pos["side"], "qty": qty, "entry": pos["entry"], "exit": price,
            "gross": round(gross, 2), "costs": round(cost, 2), "pnl": round(gross - cost, 2),
            "opened_at": pos["opened_at"], "closed_at": _now(), "why": why,
        }
        state["closed"].append(trade)
        return trade

    def apply(self, root: str, target: str, qty: float | None, price: float | None) -> list[dict]:
        """Move this root's paper position to ``target`` ('long'/'short'/'flat')
        at ``price``. Returns the list of actions taken ([] = already there)."""
        with _LOCK:
            state = self._load()
            pos = state["positions"].get(root)
            actions: list[dict] = []
            if price is not None:
                state["last_price"][root] = price

            if target == "flat":
                if pos:
                    if price is None:
                        raise ValueError("The exit alert had no price, so the paper fill cannot be priced. "
                                         "Add \"price\": \"{{strategy.order.price}}\" to the alert message.")
                    trade = self._close_leg(state, root, pos, pos["qty"], price, "exit")
                    del state["positions"][root]
                    actions.append({"action": "close", **trade})
                self._save(state)
                return actions

            if price is None:
                raise ValueError("The alert had no price, so the paper fill cannot be priced. "
                                 "Add \"price\": \"{{strategy.order.price}}\" to the alert message.")
            qty = float(qty or 1.0)

            if pos and pos["side"] != target:
                trade = self._close_leg(state, root, pos, pos["qty"], price, "reverse")
                actions.append({"action": "close", **trade})
                del state["positions"][root]
                pos = None

            if pos is None:
                state["positions"][root] = {"side": target, "qty": qty, "entry": price, "opened_at": _now()}
                actions.append({"action": "open", "root": root, "side": target, "qty": qty, "price": price})
            elif abs(pos["qty"] - qty) > 1e-9:
                if qty > pos["qty"]:   # scale in: average the entry
                    add = qty - pos["qty"]
                    pos["entry"] = (pos["entry"] * pos["qty"] + price * add) / qty
                    pos["qty"] = qty
                    actions.append({"action": "add", "root": root, "side": target, "qty": add, "price": price})
                else:                  # scale out
                    cut = pos["qty"] - qty
                    trade = self._close_leg(state, root, pos, cut, price, "scale-out")
                    pos["qty"] = qty
                    actions.append({"action": "reduce", **trade})
            self._save(state)
            return actions

    # ---- reporting
    def snapshot(self) -> dict:
        with _LOCK:
            state = self._load()
        closed = state["closed"]
        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        equity, peak, max_dd = 0.0, 0.0, 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        open_rows = []
        unreal = 0.0
        for root, pos in state["positions"].items():
            per_point, _c, _l = instrument_economics(root)
            last = state["last_price"].get(root)
            direction = 1 if pos["side"] == "long" else -1
            u = (last - pos["entry"]) * direction * pos["qty"] * per_point if last is not None else None
            if u is not None:
                unreal += u
            open_rows.append({**pos, "root": root, "last": last, "unrealized": None if u is None else round(u, 2)})
        gross_win, gross_loss = sum(wins), -sum(losses)
        return {
            "open": open_rows,
            "trades": len(closed),
            "realized": round(sum(pnls), 2),
            "unrealized": round(unreal, 2),
            "win_rate": round(100.0 * len(wins) / len(closed), 1) if closed else None,
            "avg_win": round(gross_win / len(wins), 2) if wins else None,
            "avg_loss": round(-gross_loss / len(losses), 2) if losses else None,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
            "max_drawdown": round(max_dd, 2),
            "since": state.get("since", ""),
            "recent_trades": list(reversed(closed[-15:])),
        }
