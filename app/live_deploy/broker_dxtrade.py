"""
DXtrade adapter -- DXtrade (Devexperts) is the platform behind FTMO's
DXtrade accounts and a number of other prop firms' non-MetaTrader
offering.

REST + JSON, session-token auth (no OAuth2, no local terminal). Your
broker/firm's own DXtrade web platform URL is normally also its API
host -- e.g. a firm on `https://dxtrade.somefirm.com` typically exposes
its API under the same host. There is no single universal DXtrade host
the way there is for cTrader or Tradovate; `base_url` below MUST be set
per firm from your account's own login page/dashboard.

HONESTY NOTE (see broker_base.py's module docstring): DXtrade is
white-labeled per broker/firm, so exact endpoint paths and field names
can vary between deployments more than the other three new adapters.
This has been written against Devexperts' commonly-published REST
pattern (session login -> bearer-style token -> /accounts, /positions,
/orders), not exercised against any specific firm's live deployment.
Confirm your firm's exact API base path (some prefix with `/api`, some
with `/dxsca-web`) before trusting this against a funded account.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from app.live_deploy.broker_base import (
    BrokerAdapter, ConnectionResult, OpenPosition, OrderResult,
)

_RESOLUTION_MAP = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 240: "4h", 1440: "1d"}


def is_available() -> bool:
    return True


class DXtradeBrokerAdapter(BrokerAdapter):
    platform_name = "DXtrade"

    def __init__(self, base_url: str, username: str, password: str, domain: str = "default", account_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.domain = domain
        self.account_id = account_id
        self.token: str | None = None
        self._session = requests.Session()

    def connect(self) -> ConnectionResult:
        try:
            resp = self._session.post(f"{self.base_url}/login", json={
                "username": self.username, "domain": self.domain, "password": self.password,
            }, timeout=15)
            data = resp.json()
        except Exception as exc:
            return ConnectionResult(ok=False, message=f"DXtrade login request failed: {exc}")
        token = data.get("sessionToken") or data.get("token") or resp.headers.get("Authorization")
        if not token:
            return ConnectionResult(ok=False, message=f"DXtrade login rejected: {data}")
        self.token = token
        self._session.headers.update({"Authorization": f"DXAPI {self.token}"})

        if self.account_id is None:
            try:
                accounts = self._session.get(f"{self.base_url}/accounts", timeout=15).json()
                accts = accounts.get("accounts", accounts) if isinstance(accounts, dict) else accounts
                if not accts:
                    return ConnectionResult(ok=False, message="Login succeeded but no DXtrade accounts were found.")
                self.account_id = accts[0].get("account") or accts[0].get("id")
            except Exception as exc:
                return ConnectionResult(ok=False, message=f"Could not resolve account id: {exc}")

        summary = self.account_summary() or {}
        return ConnectionResult(
            ok=True, message="Connected.", account_login=str(self.account_id), account_server=self.base_url,
            balance=summary.get("balance"), equity=summary.get("equity"), currency=summary.get("currency"),
        )

    def disconnect(self) -> None:
        self.token = None
        self._session.headers.pop("Authorization", None)

    def is_alive(self) -> bool:
        if not self.token:
            return False
        try:
            r = self._session.get(f"{self.base_url}/accounts/{self.account_id}", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def account_summary(self) -> Optional[dict]:
        if not self.token:
            return None
        try:
            acc = self._session.get(f"{self.base_url}/accounts/{self.account_id}", timeout=15).json()
            return {
                "login": self.account_id, "server": self.base_url,
                "balance": acc.get("balance"), "equity": acc.get("equity", acc.get("balance")),
                "margin_free": acc.get("availableBalance"), "currency": acc.get("currency", "USD"),
            }
        except Exception:
            return None

    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        resolution = _RESOLUTION_MAP.get(timeframe_minutes)
        if resolution is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_minutes} minutes. Supported: {sorted(_RESOLUTION_MAP)}.")
        resp = self._session.get(f"{self.base_url}/marketdata/{symbol}/candles", params={
            "resolution": resolution, "count": count + 1,
        }, timeout=20)
        data = resp.json()
        candles = data.get("candles", data if isinstance(data, list) else [])
        rows = [{
            "timestamp": pd.to_datetime(c.get("time") or c.get("timestamp"), unit="ms", utc=True),
            "open": float(c["open"]), "high": float(c["high"]), "low": float(c["low"]),
            "close": float(c["close"]), "volume": float(c.get("volume", 0)),
        } for c in candles]
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df.iloc[:-1] if len(df) > 1 else df

    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]:
        try:
            resp = self._session.get(f"{self.base_url}/accounts/{self.account_id}/positions", timeout=15).json()
            positions = resp.get("positions", resp) if isinstance(resp, dict) else resp
        except Exception:
            return []
        out = []
        for p in positions:
            sym = p.get("symbol") or p.get("instrument")
            if symbol and sym != symbol:
                continue
            qty = float(p.get("qty", p.get("quantity", 0)))
            out.append(OpenPosition(
                ticket=str(p.get("id") or p.get("positionId")), symbol=sym,
                direction=1 if qty > 0 else -1, volume=abs(qty),
                open_price=float(p.get("averagePrice", p.get("price", 0))),
                sl=float(p.get("stopLoss", 0) or 0), tp=float(p.get("takeProfit", 0) or 0),
                open_time=pd.to_datetime(p.get("openTime", pd.Timestamp.utcnow()), utc=True),
                profit=float(p.get("unrealizedPnl", p.get("pnl", 0)) or 0),
            ))
        return out

    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Live", deviation: int = 20,
    ) -> OrderResult:
        try:
            body = {
                "accountId": self.account_id, "symbol": symbol,
                "side": "buy" if direction == 1 else "sell", "quantity": volume,
                "orderType": "market", "comment": comment,
            }
            if sl_price:
                body["stopLoss"] = sl_price
            if tp_price:
                body["takeProfit"] = tp_price
            resp = self._session.post(f"{self.base_url}/orders", json=body, timeout=20)
            data = resp.json()
            if resp.status_code >= 400:
                return OrderResult(ok=False, message=str(data))
            return OrderResult(ok=True, message="Order submitted.", ticket=str(data.get("orderId") or data.get("id")), volume=volume)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))

    def close_position(self, ticket: str, comment: str = "T58 Live close") -> OrderResult:
        try:
            resp = self._session.post(f"{self.base_url}/positions/{ticket}/close", json={"comment": comment}, timeout=20)
            if resp.status_code >= 400:
                return OrderResult(ok=False, message=resp.text)
            return OrderResult(ok=True, message="Closed.", ticket=ticket)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))
