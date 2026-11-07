"""Paper brokerage account (JSON on disk). Same interface as the live broker so manual and automated flows are identical."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()


def _path() -> Path:
    p = Path(os.environ.get("T58_INVEST_DIR") or (Path.home() / ".t58" / "investing"))
    p.mkdir(parents=True, exist_ok=True)
    return p / "paper_account.json"


class PaperAccount:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else _path()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:  # noqa: BLE001
                pass
        return {"cash": 10_000.0, "start_cash": 10_000.0, "positions": {}, "orders": [], "created": time.time(), "settings": {}}

    def _save(self, d: dict) -> None:
        self.path.write_text(json.dumps(d, indent=1))

    def reset(self, cash: float = 10_000.0) -> dict:
        with _lock:
            d = {"cash": float(cash), "start_cash": float(cash), "positions": {}, "orders": [], "created": time.time(), "settings": {}}
            self._save(d)
            return d

    def state(self, prices: dict | None = None) -> dict:
        d = self._load()
        prices = prices or {}
        pos, total = [], d["cash"]
        for sym, p in d["positions"].items():
            px = prices.get(sym, p["avg_cost"])
            mv = p["qty"] * px
            total += mv
            pos.append({"symbol": sym, "qty": p["qty"], "avg_cost": p["avg_cost"], "price": px, "market_value": mv,
                        "pnl": (px - p["avg_cost"]) * p["qty"], "pnl_pct": (px / p["avg_cost"] - 1) * 100 if p["avg_cost"] else 0})
        return {"mode": "paper", "cash": d["cash"], "equity": total, "start_cash": d["start_cash"], "pnl": total - d["start_cash"],
                "pnl_pct": (total / d["start_cash"] - 1) * 100 if d["start_cash"] else 0, "positions": sorted(pos, key=lambda x: -x["market_value"]),
                "orders": d["orders"][-50:][::-1], "settings": d.get("settings", {})}

    def order(self, symbol: str, side: str, qty: float, price: float, note: str = "") -> dict:
        symbol, side = symbol.upper(), side.lower()
        if side not in ("buy", "sell") or qty <= 0 or price <= 0:
            raise ValueError("Order needs side buy/sell, positive quantity and a price.")
        with _lock:
            d = self._load()
            cost = qty * price
            pos = d["positions"].get(symbol)
            if side == "buy":
                if cost > d["cash"] + 1e-9:
                    raise ValueError(f"Not enough paper cash: need ${cost:,.2f}, have ${d['cash']:,.2f}.")
                d["cash"] -= cost
                if pos:
                    tq = pos["qty"] + qty
                    pos["avg_cost"] = (pos["avg_cost"] * pos["qty"] + cost) / tq
                    pos["qty"] = tq
                else:
                    d["positions"][symbol] = {"qty": qty, "avg_cost": price}
            else:
                if not pos or qty > pos["qty"] + 1e-9:
                    raise ValueError(f"You only hold {pos['qty'] if pos else 0} of {symbol}.")
                d["cash"] += cost
                pos["qty"] -= qty
                if pos["qty"] <= 1e-9:
                    del d["positions"][symbol]
            o = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "symbol": symbol, "side": side, "qty": qty, "price": price, "note": note}
            d["orders"].append(o)
            self._save(d)
            return o

    def set_settings(self, **kw) -> None:
        with _lock:
            d = self._load()
            d.setdefault("settings", {}).update(kw)
            self._save(d)

    def rebalance(self, targets: dict, prices: dict, min_trade: float = 25.0, note: str = "auto-rebalance") -> list:
        """Move the account toward target weights. Sells first, then buys. Returns the orders placed."""
        st = self.state(prices)
        eq = st["equity"]
        held = {p["symbol"]: p for p in st["positions"]}
        placed = []
        for sym in list(held):
            want = targets.get(sym, 0.0) * eq
            diff = want - held[sym]["market_value"]
            if diff < -min_trade:
                placed.append(self.order(sym, "sell", min(held[sym]["qty"], -diff / prices[sym]), prices[sym], note))
        for sym, w in targets.items():
            if sym not in prices:
                continue
            have = held.get(sym, {}).get("market_value", 0.0)
            diff = w * eq - have
            cash = self._load()["cash"]
            if diff > min_trade and cash > min_trade:
                placed.append(self.order(sym, "buy", min(diff, cash) / prices[sym], prices[sym], note))
        return placed
