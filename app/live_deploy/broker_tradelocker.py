"""
TradeLocker adapter -- TradeLocker is the platform behind several
FundedNext account types and a growing list of other forex/CFD prop
firms as an alternative to MT5/cTrader.

TradeLocker's API is REST + JSON, documented at https://tradelocker.com
(developer docs linked from your broker/firm dashboard once you have an
account). No local terminal, no paid API tier as of this writing --
verify current terms, since access models across all these newer
platforms move faster than MT4/MT5's.

Auth model: email + password + the specific "server" your account lives
on (shown in your TradeLocker dashboard, analogous to an MT5 server
name), exchanged for a short-lived access token + a refresh token this
adapter uses to silently re-authenticate.

HONESTY NOTE (see broker_base.py's module docstring): written against
TradeLocker's published REST reference, not exercised against a live
account by anyone on this project -- of the four new adapters in this
package, this is the one with the least certain endpoint/field-name
accuracy, since TradeLocker's public docs are newer and thinner than
MT5/cTrader/Tradovate's. Treat every endpoint path and field name below
as a first draft to verify against your own account's actual API
responses (call each endpoint once from a REPL and print the raw JSON
before trusting the parsed fields) rather than as confirmed-correct.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from app.live_deploy.broker_base import (
    BrokerAdapter, ConnectionResult, OpenPosition, OrderResult,
)

_LIVE_BASE = "https://live.tradelocker.com/backend-api"
_DEMO_BASE = "https://demo.tradelocker.com/backend-api"

_RESOLUTION_MAP = {1: "1", 5: "5", 15: "15", 30: "30", 60: "60", 240: "240", 1440: "1D"}


def is_available() -> bool:
    return True


class TradeLockerBrokerAdapter(BrokerAdapter):
    platform_name = "TradeLocker"

    def __init__(self, email: str, password: str, server: str, is_live: bool = False):
        self.email = email
        self.password = password
        self.server = server
        self.is_live = is_live
        self.base_url = _LIVE_BASE if is_live else _DEMO_BASE
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self.account_id: str | None = None
        self.acc_num: int | None = None
        self._session = requests.Session()

    def connect(self) -> ConnectionResult:
        try:
            resp = self._session.post(f"{self.base_url}/auth/jwt/token", json={
                "email": self.email, "password": self.password, "server": self.server,
            }, timeout=15)
            data = resp.json()
        except Exception as exc:
            return ConnectionResult(ok=False, message=f"TradeLocker auth request failed: {exc}")
        if "accessToken" not in data:
            return ConnectionResult(ok=False, message=f"TradeLocker login rejected: {data}")
        self.access_token = data["accessToken"]
        self.refresh_token = data.get("refreshToken")
        self._session.headers.update({"Authorization": f"Bearer {self.access_token}"})

        try:
            accounts = self._session.get(f"{self.base_url}/auth/jwt/all-accounts", timeout=15).json()
            accts = accounts.get("accounts", accounts) if isinstance(accounts, dict) else accounts
            if not accts:
                return ConnectionResult(ok=False, message="Login succeeded but no TradeLocker accounts were found.")
            self.account_id = str(accts[0]["id"])
            self.acc_num = accts[0].get("accNum", 0)
        except Exception as exc:
            return ConnectionResult(ok=False, message=f"Could not resolve account id: {exc}")

        summary = self.account_summary() or {}
        return ConnectionResult(
            ok=True, message="Connected.", account_login=self.account_id,
            account_server=f"TradeLocker-{self.server}",
            balance=summary.get("balance"), equity=summary.get("equity"), currency=summary.get("currency"),
        )

    def _headers(self) -> dict:
        return {"accNum": str(self.acc_num)} if self.acc_num is not None else {}

    def disconnect(self) -> None:
        self.access_token = None
        self._session.headers.pop("Authorization", None)

    def is_alive(self) -> bool:
        if not self.access_token:
            return False
        try:
            r = self._session.get(f"{self.base_url}/trade/accounts/{self.account_id}/state",
                                   headers=self._headers(), timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def account_summary(self) -> Optional[dict]:
        if not self.access_token:
            return None
        try:
            state = self._session.get(f"{self.base_url}/trade/accounts/{self.account_id}/state",
                                       headers=self._headers(), timeout=15).json()
            d = state.get("d", state)
            return {
                "login": self.account_id, "server": self.server,
                "balance": d.get("balance"), "equity": d.get("equity", d.get("balance")),
                "margin_free": d.get("availableFunds"), "currency": d.get("currency", "USD"),
            }
        except Exception:
            return None

    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        resolution = _RESOLUTION_MAP.get(timeframe_minutes)
        if resolution is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_minutes} minutes. Supported: {sorted(_RESOLUTION_MAP)}.")
        resp = self._session.get(f"{self.base_url}/trade/history", params={
            "routeId": symbol, "resolution": resolution, "from": 0, "to": 0, "count": count + 1,
        }, headers=self._headers(), timeout=20)
        data = resp.json()
        d = data.get("d", data)
        bars = d.get("barDetails", d if isinstance(d, list) else [])
        rows = [{
            "timestamp": pd.to_datetime(b.get("t") or b.get("timestamp"), unit="ms", utc=True),
            "open": float(b["o"]), "high": float(b["h"]), "low": float(b["l"]),
            "close": float(b["c"]), "volume": float(b.get("v", 0)),
        } for b in bars]
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df.iloc[:-1] if len(df) > 1 else df

    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]:
        try:
            resp = self._session.get(f"{self.base_url}/trade/accounts/{self.account_id}/positions",
                                      headers=self._headers(), timeout=15).json()
            positions = resp.get("d", {}).get("positions", []) if isinstance(resp, dict) else resp
        except Exception:
            return []
        out = []
        for p in positions:
            sym = p.get("tradableInstrumentName") or str(p.get("tradableInstrumentId"))
            if symbol and sym != symbol:
                continue
            side = str(p.get("side", "")).lower()
            out.append(OpenPosition(
                ticket=str(p["id"]), symbol=sym, direction=1 if side == "buy" else -1,
                volume=float(p.get("qty", 0)), open_price=float(p.get("avgPrice", 0)),
                sl=float(p.get("stopLoss", 0) or 0), tp=float(p.get("takeProfit", 0) or 0),
                open_time=pd.to_datetime(p.get("openDate", pd.Timestamp.utcnow()), utc=True),
                profit=float(p.get("unrealizedPl", 0) or 0),
            ))
        return out

    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Live", deviation: int = 20,
    ) -> OrderResult:
        try:
            body = {
                "tradableInstrumentId": symbol, "qty": volume,
                "side": "buy" if direction == 1 else "sell", "type": "market", "validity": "IOC",
            }
            if sl_price:
                body["stopLoss"] = sl_price
            if tp_price:
                body["takeProfit"] = tp_price
            resp = self._session.post(f"{self.base_url}/trade/accounts/{self.account_id}/orders",
                                       json=body, headers=self._headers(), timeout=20)
            data = resp.json()
            if resp.status_code >= 400:
                return OrderResult(ok=False, message=str(data))
            order_id = data.get("d", {}).get("orderId") if isinstance(data, dict) else None
            return OrderResult(ok=True, message="Order submitted.", ticket=str(order_id), volume=volume)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))

    def close_position(self, ticket: str, comment: str = "T58 Live close") -> OrderResult:
        try:
            resp = self._session.delete(
                f"{self.base_url}/trade/accounts/{self.account_id}/positions/{ticket}",
                headers=self._headers(), timeout=20,
            )
            if resp.status_code >= 400:
                return OrderResult(ok=False, message=resp.text)
            return OrderResult(ok=True, message="Closed.", ticket=ticket)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))
