"""Optional LIVE brokerage via Alpaca (uses the keys already saved in Settings). Off unless the user explicitly selects Live mode.
Defaults to Alpaca's paper-trading endpoint; the real-money endpoint needs ``live=True`` AND an explicit confirmation flag from the UI."""
from __future__ import annotations

from app.data import alpaca_credentials


class BrokerError(RuntimeError):
    pass


class AlpacaBroker:
    def __init__(self, live: bool = False):
        c = alpaca_credentials.load_credentials()
        if not c or not c.is_usable:
            raise BrokerError("No Alpaca keys saved. Add them under Market Data / Settings first.")
        self.live = live
        self.base = "https://api.alpaca.markets" if live else "https://paper-api.alpaca.markets"
        self.h = {"APCA-API-KEY-ID": c.api_key, "APCA-API-SECRET-KEY": c.secret_key}

    def _req(self, method, path, **kw):
        import requests
        r = requests.request(method, self.base + path, headers=self.h, timeout=20, **kw)
        if r.status_code >= 300:
            raise BrokerError(f"Alpaca {r.status_code}: {r.text[:200]}")
        return r.json() if r.text else {}

    def state(self, prices=None) -> dict:
        a = self._req("GET", "/v2/account")
        pos = [{"symbol": p["symbol"], "qty": float(p["qty"]), "avg_cost": float(p["avg_entry_price"]), "price": float(p["current_price"]),
                "market_value": float(p["market_value"]), "pnl": float(p["unrealized_pl"]), "pnl_pct": float(p["unrealized_plpc"]) * 100}
               for p in self._req("GET", "/v2/positions")]
        eq, last = float(a["equity"]), float(a.get("last_equity") or a["equity"])
        return {"mode": "live" if self.live else "alpaca-paper", "cash": float(a["cash"]), "equity": eq, "start_cash": last, "pnl": eq - last,
                "pnl_pct": (eq / last - 1) * 100 if last else 0, "positions": pos, "orders": [], "settings": {}}

    def order(self, symbol, side, qty, price=None, note=""):
        sym = symbol.upper().replace("-USD", "/USD")
        o = self._req("POST", "/v2/orders", json={"symbol": sym, "qty": str(round(qty, 6)), "side": side, "type": "market",
                                                  "time_in_force": "gtc" if "/" in sym else "day"})
        return {"t": o.get("submitted_at", ""), "symbol": symbol, "side": side, "qty": qty, "price": price or 0, "note": note or o.get("id", "")}
