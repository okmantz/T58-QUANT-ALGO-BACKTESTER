"""
Tradovate adapter -- closes the futures-prop-firm gap flagged in
prop_firms.py (Apex Trader Funding, TopStep, MyFundedFutures all route
through Tradovate on at least one account type; this app previously had
zero futures-platform connectivity).

Tradovate's API is plain REST + JSON (no protobuf, no local terminal),
which makes it the most straightforward of the four new adapters. Setup:
  1. Register a free developer application at https://tradovate.com/api
     (an "app id" and "app secret" -- the app registration itself is
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

    def account_summary(self) -> Optional[dict]:
        if not self.access_token:
            return None
        try:
            cash = self._session.get(f"{self.base_url}/cashBalance/getcashbalancesnapshot",
                                      params={"accountId": self.account_id}, timeout=15).json()
            return {
                "login": self.account_id, "server": "Tradovate",
                "balance": cash.get("cashBalance") or cash.get("amount"),
                "equity": cash.get("netLiq") or cash.get("cashBalance"),
                "margin_free": None, "currency": cash.get("currency", "USD"),
            }
        except Exception:
            return None

    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        chart_kind, unit = _TIMEFRAME_TO_CHART.get(timeframe_minutes, (None, None))
        if chart_kind is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_minutes} minutes. Supported: {sorted(_TIMEFRAME_TO_CHART)}.")
        # Tradovate's real-time chart history is normally streamed over their
        # WebSocket `md` service (md.tradovateapi.com); this REST call hits
        # their historical-quote endpoint as a simpler (but less complete)
        # substitute -- see module docstring.
        resp = self._session.get(f"{self.base_url}/md/getchart", params={
            "symbol": symbol, "chartDescription": chart_kind, "elementSize": unit,
            "elementSizeUnit": "UnderlyingUnits", "withHistogram": "false", "count": count + 1,
        }, timeout=20)
        data = resp.json()
        bars = data.get("bars", data if isinstance(data, list) else [])
        rows = [{
            "timestamp": pd.to_datetime(b["timestamp"], utc=True),
            "open": float(b["open"]), "high": float(b["high"]),
            "low": float(b["low"]), "close": float(b["close"]), "volume": float(b.get("upVolume", 0) + b.get("downVolume", 0)),
        } for b in bars]
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df.iloc[:-1] if len(df) > 1 else df

    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]:
        try:
            positions = self._session.get(f"{self.base_url}/position/list", timeout=15).json()
        except Exception:
            return []
        out = []
        for p in positions:
            if p.get("accountId") != self.account_id or p.get("netPos", 0) == 0:
                continue
            sym = p.get("contractName") or str(p.get("contractId"))
            if symbol and sym != symbol:
                continue
            net = p["netPos"]
            out.append(OpenPosition(
                ticket=str(p["id"]), symbol=sym, direction=1 if net > 0 else -1,
                volume=abs(net), open_price=p.get("netPrice", 0.0), sl=0.0, tp=0.0,
                open_time=pd.to_datetime(p.get("timestamp", pd.Timestamp.utcnow()), utc=True),
                profit=p.get("openPl", 0.0),
            ))
        return out

    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Live", deviation: int = 20,
    ) -> OrderResult:
        try:
            resp = self._session.post(f"{self.base_url}/order/placeorder", json={
                "accountId": self.account_id, "symbol": symbol,
                "action": "Buy" if direction == 1 else "Sell",
                "orderQty": int(volume), "orderType": "Market",
            }, timeout=20)
            data = resp.json()
            if "orderId" not in data and "failureReason" in data:
                return OrderResult(ok=False, message=data.get("failureText", str(data)))
            return OrderResult(ok=True, message="Order submitted.", ticket=str(data.get("orderId")), volume=volume)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))

    def close_position(self, ticket: str, comment: str = "T58 Live close") -> OrderResult:
        try:
            resp = self._session.post(f"{self.base_url}/order/liquidateposition", json={
                "accountId": self.account_id, "positionId": int(ticket),
            }, timeout=20)
            data = resp.json()
            return OrderResult(ok=True, message="Close submitted.", ticket=ticket)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))
