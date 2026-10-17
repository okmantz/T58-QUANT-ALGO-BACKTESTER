"""
Tradovate adapter -- closes the futures-prop-firm gap flagged in
prop_firms.py (Apex Trader Funding, TopStep, MyFundedFutures all route
through Tradovate on at least one account type; this app previously had
zero futures-platform connectivity).

Tradovate's API is plain REST + JSON (no protobuf, no local terminal),
which makes it the most straightforward of the four new adapters. Setup:
  1. Register a free developer application at https://api.tradovate.com
     (v7 link fix 2026-10-05: the old https://tradovate.com/api returns 404;
     an "app id" and "app secret" -- the app registration itself is
     free; verify current terms there, they do change access tiers over
     time). For development/testing, Tradovate's demo environment at
     demo.tradovateapi.com works against a free practice/sim account with
     no live funding needed.
  2. This adapter authenticates with your Tradovate username/password
     (same credentials as the Tradovate web/desktop platform) plus the
     app id/secret, and exchanges them for a short-lived access token it
     refreshes automatically.

HONESTY NOTE (see broker_base.py's module docstring): written against
Tradovate's published REST reference, not exercised against a live or
demo account by anyone on this project. The historical-bars endpoint in
particular is the part most likely to need adjustment -- Tradovate's real
market-data history API is WebSocket-streamed rather than a simple REST
GET, and the REST fallback used here may only return a shorter window
than `count` bars. Test against a demo account before trusting it for
signal generation.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd
import requests

from app.live_deploy.broker_base import (
    BrokerAdapter, ConnectionResult, OpenPosition, OrderResult,
)

_LIVE_BASE = "https://live.tradovateapi.com/v1"
_DEMO_BASE = "https://demo.tradovateapi.com/v1"

_TIMEFRAME_TO_CHART = {
    1: ("MinuteBar", 1), 5: ("MinuteBar", 5), 15: ("MinuteBar", 15),
    30: ("MinuteBar", 30), 60: ("MinuteBar", 60), 1440: ("DailyBar", 1),
}


def is_available() -> bool:
    return True  # pure REST via `requests`, already an app dependency -- no optional import to guard


class TradovateBrokerAdapter(BrokerAdapter):
    platform_name = "Tradovate"

    def __init__(self, username: str, password: str, app_id: str, app_secret: str,
                 cid: str, sec: str, is_live: bool = False, account_id: int | None = None):
        self.username = username
        self.password = password
        self.app_id = app_id
        self.app_secret = app_secret
        self.cid = cid          # Tradovate "client id" issued with your app registration
        self.sec = sec          # Tradovate "client secret" issued with your app registration
        self.is_live = is_live
        self.account_id = account_id  # resolved on connect() if not supplied
        self.base_url = _LIVE_BASE if is_live else _DEMO_BASE
        self.access_token: str | None = None
        self._session = requests.Session()

    def connect(self) -> ConnectionResult:
        try:
            resp = self._session.post(f"{self.base_url}/auth/accesstokenrequest", json={
                "name": self.username, "password": self.password,
                "appId": self.app_id, "appVersion": "1.0",
                "cid": self.cid, "sec": self.sec,
            }, timeout=15)
            data = resp.json()
        except Exception as exc:
            return ConnectionResult(ok=False, message=f"Tradovate auth request failed: {exc}")

        if "accessToken" not in data:
            return ConnectionResult(ok=False, message=f"Tradovate login rejected: {data.get('errorText', data)}")
        self.access_token = data["accessToken"]
        self._session.headers.update({"Authorization": f"Bearer {self.access_token}"})

        if self.account_id is None:
            try:
                accounts = self._session.get(f"{self.base_url}/account/list", timeout=15).json()
                if not accounts:
                    return ConnectionResult(ok=False, message="Login succeeded but no Tradovate accounts were found.")
                self.account_id = accounts[0]["id"]
            except Exception as exc:
                return ConnectionResult(ok=False, message=f"Could not resolve account id: {exc}")

        summary = self.account_summary() or {}
        return ConnectionResult(
            ok=True, message="Connected.", account_login=str(self.account_id),
            account_server="Tradovate-live" if self.is_live else "Tradovate-demo",
            balance=summary.get("balance"), equity=summary.get("equity"), currency="USD",
        )

    def disconnect(self) -> None:
        self.access_token = None
        self._session.headers.pop("Authorization", None)

    def is_alive(self) -> bool:
        if not self.access_token:
            return False
        try:
            return self._session.get(f"{self.base_url}/account/item?id={self.account_id}", timeout=10).status_code == 200
        except Exception:
            return False

    def _post(self, path: str, payload: dict, timeout: float = 20):
        r = self._session.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        try:
            data = r.json()
        except ValueError:
            data = {"raw": r.text}
        return r.status_code, data

    def account_summary(self) -> Optional[dict]:
        if not self.access_token:
            return None
        try:
            # getcashbalancesnapshot is a POST taking {accountId}
            status, cash = self._post("/cashBalance/getcashbalancesnapshot", {"accountId": self.account_id}, 15)
            if status >= 400 or not isinstance(cash, dict):
                return None
            bal = cash.get("totalCashValue", cash.get("cashBalance"))
            net_liq = cash.get("netLiq")
            if net_liq is None:
                net_liq = (bal or 0) + (cash.get("openPnL") or 0) if bal is not None else None
            return {
                "login": self.account_id, "server": "Tradovate",
                "balance": bal, "equity": net_liq,
                "margin_free": None, "currency": "USD",
            }
        except Exception:
            return None

    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        # Tradovate only serves chart history over its WebSocket market-data
        # service; there is no REST equivalent. The webhook relay (TradingView
        # supplies the signal) never needs bars from the broker, so refuse
        # clearly instead of returning wrong data.
        raise ValueError(
            "Tradovate chart history is WebSocket-only and not implemented in this adapter. "
            "Use a data source (Databento/Polygon/Alpaca/...) for bars; Tradovate is used for execution only."
        )

    def _contract_name(self, contract_id) -> str:
        cache = self.__dict__.setdefault("_contract_cache", {})
        if contract_id in cache:
            return cache[contract_id]
        try:
            r = self._session.get(f"{self.base_url}/contract/item", params={"id": contract_id}, timeout=10)
            name = r.json().get("name") or str(contract_id)
        except Exception:
            name = str(contract_id)
        cache[contract_id] = name
        return name

    def _contract_id(self, symbol: str):
        try:
            r = self._session.get(f"{self.base_url}/contract/find", params={"name": symbol}, timeout=10)
            return r.json().get("id")
        except Exception:
            return None

    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]:
        try:
            positions = self._session.get(f"{self.base_url}/position/list", timeout=15).json()
        except Exception:
            return []
        out = []
        for p in positions:
            if p.get("accountId") != self.account_id or p.get("netPos", 0) == 0:
                continue
            sym = p.get("contractName") or self._contract_name(p.get("contractId"))
            if symbol and sym != symbol:
                continue
            net = p["netPos"]
            out.append(OpenPosition(
                ticket=str(p["id"]), symbol=sym, direction=1 if net > 0 else -1,
                volume=abs(net), open_price=float(p.get("netPrice") or p.get("prevPrice") or 0.0), sl=0.0, tp=0.0,
                open_time=pd.to_datetime(p.get("timestamp", pd.Timestamp.now(tz="UTC")), utc=True),
                profit=float(p.get("openPl") or 0.0),
            ))
        return out

    def to_broker_qty(
        self, units: float, contract_size: Optional[float], units_per_lot: Optional[float],
    ) -> float:
        # Tradovate's orderQty is a WHOLE NUMBER OF CONTRACTS. One contract
        # = contract_size sizing units (ES: 50.0 units = 1 contract).
        # Floor, never round up; refuse (0.0) when contract_size is
        # missing -- without it we'd be guessing, and a guess here is a
        # 50x order on real capital (v7 P0-1).
        if not contract_size or contract_size <= 0:
            return 0.0
        return float(math.floor(units / contract_size + 1e-9))

    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Live", deviation: int = 20,
    ) -> OrderResult:
        # `volume` here is broker-native: WHOLE CONTRACTS, converted via
        # to_broker_qty by the caller -- never raw sizing units (v7 P0-1).
        try:
            status, data = self._post("/order/placeorder", {
                "accountSpec": self.username, "accountId": self.account_id, "symbol": symbol,
                "action": "Buy" if direction == 1 else "Sell",
                "orderQty": int(volume), "orderType": "Market", "isAutomated": True,
            })
            if status >= 400 or not isinstance(data, dict):
                return OrderResult(ok=False, message=f"HTTP {status}: {data}")
            if data.get("failureReason") or data.get("failureText") or "orderId" not in data:
                return OrderResult(ok=False, message=data.get("failureText") or data.get("errorText") or str(data))
            order_id = data["orderId"]
            price = self._fill_price(order_id)
            return OrderResult(ok=True, message="Order submitted.", ticket=str(order_id),
                               price=price, volume=volume)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))

    def _fill_price(self, order_id) -> Optional[float]:
        import time as _t
        for _ in range(3):
            try:
                r = self._session.get(f"{self.base_url}/fill/ldeps", params={"masterids": order_id}, timeout=10)
                fills = r.json()
                if isinstance(fills, list) and fills:
                    qty = sum(float(f.get("qty") or 0) for f in fills)
                    if qty > 0:
                        return sum(float(f.get("price") or 0) * float(f.get("qty") or 0) for f in fills) / qty
            except Exception:
                pass
            _t.sleep(0.4)
        return None

    def close_position(self, ticket: str, comment: str = "T58 Live close") -> OrderResult:
        # v7 P0-4: verify the API response before reporting success -- the
        # old code returned ok=True even when Tradovate rejected the
        # liquidation, leaving the session journal believing a position
        # was flat while it was still open.
        try:
            pos = next((p for p in self._session.get(f"{self.base_url}/position/list", timeout=15).json()
                        if str(p.get("id")) == str(ticket)), None)
            if pos is None:
                return OrderResult(ok=False, message=f"Position {ticket} not found on Tradovate.")
            status, data = self._post("/order/liquidateposition", {
                "accountId": self.account_id, "contractId": pos["contractId"],
                "admin": False, "customTag50": "T58",
            })
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))
        if status >= 400:
            detail = data.get("failureText") or data.get("errorText") if isinstance(data, dict) else None
            return OrderResult(ok=False, message=f"Tradovate rejected the close (HTTP {status}): {detail or data}")
        if isinstance(data, dict) and ("failureReason" in data or "failureText" in data):
            return OrderResult(
                ok=False,
                message=f"Tradovate rejected the close: {data.get('failureText') or data.get('failureReason')}",
            )
        px = self._fill_price(data["orderId"]) if isinstance(data, dict) and data.get("orderId") else None
        return OrderResult(ok=True, message="Close submitted.", ticket=ticket, price=px)
