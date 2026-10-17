"""
TradeLocker adapter (REST + JSON) -- the platform behind several prop firms.

Request shapes follow the official ``tradelocker`` Python client
(github.com/TradeLocker/tradelocker-python):

* auth      POST /auth/jwt/token     {"email","password","server"} -> accessToken/refreshToken
* refresh   POST /auth/jwt/refresh   {"refreshToken"}
* accounts  GET  /auth/jwt/all-accounts -> {"accounts":[{id, accNum, name, currency, ...}]}
* every /trade/* call needs the header ``accNum``
* config    GET  /trade/config  -> column names for array-shaped payloads
* instruments GET /trade/accounts/{id}/instruments -> d.instruments[] with
              tradableInstrumentId, name, routes[{id,type:"TRADE"|"INFO"}]
* positions GET  /trade/accounts/{id}/positions -> d.positions = LIST OF LISTS
* state     GET  /trade/accounts/{id}/state     -> d.accountDetailsData = LIST
* order     POST /trade/accounts/{id}/orders  (routeId = the instrument's TRADE route)
* close     DELETE /trade/positions/{positionId}  {"qty": "0"} (0 = whole position)
* history   GET  /trade/history?tradableInstrumentId&routeId(INFO)&resolution&from&to (ms)

Not exercised against a live account by this project (no account available);
the unit tests drive it against a fake server that mimics the shapes above.
Run it on a DEMO account first.
"""
from __future__ import annotations

import math
import time
from typing import Any, Optional

import pandas as pd
import requests

from app.live_deploy.broker_base import (
    BrokerAdapter, ConnectionResult, OpenPosition, OrderResult,
)

_LIVE_BASE = "https://live.tradelocker.com/backend-api"
_DEMO_BASE = "https://demo.tradelocker.com/backend-api"

_RESOLUTION_MAP = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1H", 240: "4H", 1440: "1D"}

# Default column order used only if /trade/config is unavailable.
_DEFAULT_POSITION_COLS = ["id", "tradableInstrumentId", "routeId", "side", "qty", "avgPrice",
                          "stopLossId", "takeProfitId", "openDate", "unrealizedPl", "strategyId"]
_DEFAULT_ORDER_COLS = ["id", "tradableInstrumentId", "routeId", "qty", "side", "type", "status",
                       "filledQty", "avgPrice", "price", "stopPrice", "validity", "expireDate",
                       "createdDate", "lastModified", "isOpen", "positionId", "stopLoss",
                       "stopLossType", "takeProfit", "takeProfitType", "strategyId"]
_DEFAULT_STATE_COLS = ["balance", "projectedBalance", "availableFunds", "blockedBalance",
                       "cashBalance", "unsettledCash", "withdrawalAvailable", "stocksValue",
                       "optionValue", "initialMarginReq", "maintMarginReq", "marginAvailable",
                       "accountEquity", "marginLevel"]


def is_available() -> bool:
    return True


class TradeLockerBrokerAdapter(BrokerAdapter):
    platform_name = "TradeLocker"

    def __init__(self, email: str, password: str, server: str, is_live: bool = False,
                 base_url: str | None = None):
        self.email = email
        self.password = password
        self.server = server
        self.is_live = is_live
        self.base_url = (base_url or (_LIVE_BASE if is_live else _DEMO_BASE)).rstrip("/")
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self.account_id: str | None = None
        self.acc_num: int | None = None
        self.currency: str | None = None
        self._session = requests.Session()
        self._config: dict | None = None
        self._instruments: list[dict] | None = None
        self._token_time = 0.0

    # ------------------------------------------------------------------ http
    def _hdrs(self, acc_num: bool = True) -> dict:
        h = {}
        if self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        if acc_num and self.acc_num is not None:
            h["accNum"] = str(self.acc_num)
        return h

    def _refresh(self) -> bool:
        if not self.refresh_token:
            return False
        try:
            r = self._session.post(f"{self.base_url}/auth/jwt/refresh",
                                   json={"refreshToken": self.refresh_token}, timeout=15)
            d = r.json()
            if r.status_code < 400 and d.get("accessToken"):
                self.access_token = d["accessToken"]
                self.refresh_token = d.get("refreshToken", self.refresh_token)
                self._token_time = time.time()
                return True
        except Exception:
            pass
        return False

    def _req(self, method: str, path: str, *, params: dict | None = None,
             json: Any = None, timeout: float = 20) -> Any:
        """Authenticated request; refreshes the token once on 401. Raises on HTTP
        errors with the body in the message so it lands in the journal."""
        url = f"{self.base_url}{path}"
        for attempt in (0, 1):
            r = self._session.request(method, url, params=params, json=json,
                                      headers=self._hdrs(), timeout=timeout)
            if r.status_code == 401 and attempt == 0 and self._refresh():
                continue
            break
        try:
            data = r.json() if r.text else {}
        except ValueError:
            data = {"raw": r.text}
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {data}")
        if isinstance(data, dict) and data.get("s") == "error":
            raise RuntimeError(f"TradeLocker error: {data.get('errmsg') or data}")
        return data

    def _cols(self, key: str, default: list[str]) -> list[str]:
        try:
            if self._config is None:
                self._config = self._req("GET", "/trade/config").get("d", {})
            cols = [c["id"] for c in self._config[key]["columns"]]
            return cols or default
        except Exception:
            return default

    # --------------------------------------------------------------- session
    def connect(self) -> ConnectionResult:
        try:
            resp = self._session.post(f"{self.base_url}/auth/jwt/token", json={
                "email": self.email, "password": self.password, "server": self.server,
            }, timeout=15)
            data = resp.json()
        except Exception as exc:
            return ConnectionResult(ok=False, message=f"TradeLocker auth request failed: {exc}")
        if not isinstance(data, dict) or "accessToken" not in data:
            return ConnectionResult(ok=False, message=f"TradeLocker login rejected: {data}")
        self.access_token = data["accessToken"]
        self.refresh_token = data.get("refreshToken")
        self._token_time = time.time()
        try:
            r = self._session.get(f"{self.base_url}/auth/jwt/all-accounts",
                                  headers=self._hdrs(acc_num=False), timeout=15).json()
            accts = r.get("accounts", []) if isinstance(r, dict) else r
            if not accts:
                return ConnectionResult(ok=False, message="Login succeeded but no TradeLocker accounts were found.")
            chosen = accts[0]
            self.account_id = str(chosen["id"])
            self.acc_num = int(chosen.get("accNum", 0) or 0)
            self.currency = chosen.get("currency")
        except Exception as exc:
            return ConnectionResult(ok=False, message=f"Could not resolve account id: {exc}")
        self._config = None
        self._instruments = None
        summary = self.account_summary() or {}
        return ConnectionResult(
            ok=True, message="Connected.", account_login=self.account_id,
            account_server=f"TradeLocker-{self.server}",
            balance=summary.get("balance"), equity=summary.get("equity"),
            currency=summary.get("currency") or self.currency,
        )

    def disconnect(self) -> None:
        self.access_token = None
        self.refresh_token = None

    def is_alive(self) -> bool:
        if not self.access_token or not self.account_id:
            return False
        try:
            self._req("GET", f"/trade/accounts/{self.account_id}/state", timeout=10)
            return True
        except Exception:
            return False

    def account_summary(self) -> Optional[dict]:
        if not self.access_token or not self.account_id:
            return None
        try:
            d = self._req("GET", f"/trade/accounts/{self.account_id}/state").get("d", {})
            vals = d.get("accountDetailsData")
            if isinstance(vals, list):
                cols = self._cols("accountDetailsConfig", _DEFAULT_STATE_COLS)
                d = dict(zip(cols, vals))
            bal = d.get("balance")
            eq = d.get("accountEquity", d.get("projectedBalance", bal))
            return {"login": self.account_id, "server": self.server, "balance": bal, "equity": eq,
                    "margin_free": d.get("availableFunds"), "currency": self.currency or "USD"}
        except Exception:
            return None

    # ----------------------------------------------------------- instruments
    def _load_instruments(self) -> list[dict]:
        if self._instruments is None:
            d = self._req("GET", f"/trade/accounts/{self.account_id}/instruments").get("d", {})
            self._instruments = d.get("instruments", []) if isinstance(d, dict) else []
        return self._instruments

    def _instrument(self, symbol: str) -> dict:
        s = str(symbol).strip()
        up = s.upper()
        insts = self._load_instruments()
        for key in ("name", "localizedName"):
            for i in insts:
                if str(i.get(key, "")).upper() == up:
                    return i
        for i in insts:
            if str(i.get("tradableInstrumentId")) == s:
                return i
        # tolerate suffixes / separators: "EUR/USD", "EURUSD.r"
        flat = "".join(ch for ch in up if ch.isalnum())
        for i in insts:
            n = "".join(ch for ch in str(i.get("name", "")).upper() if ch.isalnum())
            if n == flat or (n and flat.startswith(n) and len(flat) - len(n) <= 2):
                return i
        raise ValueError(f"Instrument '{symbol}' not found on this TradeLocker account.")

    @staticmethod
    def _route(inst: dict, kind: str) -> str:
        routes = [str(r["id"]) for r in inst.get("routes", []) if r.get("type") == kind]
        if not routes:
            raise ValueError(f"No {kind} route for instrument {inst.get('name')}.")
        return routes[0]

    # ------------------------------------------------------------------ data
    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        resolution = _RESOLUTION_MAP.get(timeframe_minutes)
        if resolution is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_minutes} minutes. Supported: {sorted(_RESOLUTION_MAP)}.")
        inst = self._instrument(symbol)
        now_ms = int(time.time() * 1000)
        span_ms = (count + 5) * timeframe_minutes * 60_000 * 2  # x2 covers weekends
        data = self._req("GET", "/trade/history", params={
            "tradableInstrumentId": inst["tradableInstrumentId"],
            "routeId": self._route(inst, "INFO"),
            "resolution": resolution, "from": now_ms - span_ms, "to": now_ms,
        })
        bars = (data.get("d") or {}).get("barDetails", []) if isinstance(data, dict) else []
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if not bars:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame([{
            "timestamp": pd.to_datetime(int(b["t"]), unit="ms", utc=True),
            "open": float(b["o"]), "high": float(b["h"]), "low": float(b["l"]),
            "close": float(b["c"]), "volume": float(b.get("v", 0) or 0),
        } for b in bars]).sort_values("timestamp").reset_index(drop=True)
        df = df.iloc[:-1] if len(df) > 1 else df  # drop the still-forming bar
        return df.tail(count).reset_index(drop=True)

    # ------------------------------------------------------------- positions
    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]:
        try:
            d = self._req("GET", f"/trade/accounts/{self.account_id}/positions").get("d", {})
            rows = d.get("positions", []) if isinstance(d, dict) else []
            cols = self._cols("positionsConfig", _DEFAULT_POSITION_COLS)
        except Exception:
            return []
        try:
            id_to_name = {int(i["tradableInstrumentId"]): str(i.get("name"))
                          for i in self._load_instruments()}
        except Exception:
            id_to_name = {}
        want = None
        if symbol:
            try:
                want = int(self._instrument(symbol)["tradableInstrumentId"])
            except Exception:
                want = None
        out = []
        for row in rows:
            p = dict(zip(cols, row)) if isinstance(row, (list, tuple)) else dict(row)
            iid = int(p.get("tradableInstrumentId", 0) or 0)
            if want is not None and iid != want:
                continue
            side = str(p.get("side", "")).lower()
            try:
                opened = pd.to_datetime(int(p.get("openDate")), unit="ms", utc=True)
            except Exception:
                opened = pd.Timestamp.now(tz="UTC")
            out.append(OpenPosition(
                ticket=str(p["id"]), symbol=id_to_name.get(iid, str(iid)),
                direction=1 if side == "buy" else -1,
                volume=float(p.get("qty", 0) or 0), open_price=float(p.get("avgPrice", 0) or 0),
                sl=0.0, tp=0.0, open_time=opened, profit=float(p.get("unrealizedPl", 0) or 0),
            ))
        return out

    def to_broker_qty(
        self, units: float, contract_size: Optional[float], units_per_lot: Optional[float],
    ) -> float:
        # TradeLocker order `qty` is in lots. Floor to 0.01, never round up;
        # refuse (0.0) when the lot conversion factor is missing.
        if not units_per_lot or units_per_lot <= 0:
            return 0.0
        return math.floor(units / units_per_lot * 100 + 1e-9) / 100.0

    # ---------------------------------------------------------------- orders
    def _order_fill(self, order_id: str) -> tuple[Optional[float], Optional[str]]:
        """Best-effort: (fill price, positionId) of an order from ordersHistory."""
        try:
            cols = self._cols("ordersHistoryConfig", _DEFAULT_ORDER_COLS)
            for _ in range(3):
                d = self._req("GET", f"/trade/accounts/{self.account_id}/ordersHistory").get("d", {})
                for row in d.get("ordersHistory", []):
                    o = dict(zip(cols, row)) if isinstance(row, (list, tuple)) else dict(row)
                    if str(o.get("id")) == str(order_id) and str(o.get("status", "")).lower() == "filled":
                        px = o.get("avgPrice")
                        pos = o.get("positionId")
                        return (float(px) if px else None), (str(pos) if pos else None)
                time.sleep(0.4)
        except Exception:
            pass
        return None, None

    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Live", deviation: int = 20,
    ) -> OrderResult:
        # `volume` is broker-native LOTS (see to_broker_qty).
        try:
            inst = self._instrument(symbol)
            body: dict[str, Any] = {
                "price": None,
                "qty": str(volume),
                "routeId": self._route(inst, "TRADE"),
                "side": "buy" if direction == 1 else "sell",
                "validity": "IOC",
                "tradableInstrumentId": str(inst["tradableInstrumentId"]),
                "type": "market",
            }
            if sl_price:
                body.update(stopLoss=float(sl_price), stopLossType="absolute")
            if tp_price:
                body.update(takeProfit=float(tp_price), takeProfitType="absolute")
            data = self._req("POST", f"/trade/accounts/{self.account_id}/orders", json=body)
            order_id = (data.get("d") or {}).get("orderId")
            if not order_id:
                return OrderResult(ok=False, message=f"Order not accepted: {data}")
            price, pos_id = self._order_fill(str(order_id))
            return OrderResult(ok=True, message="Order submitted.",
                               ticket=str(pos_id or order_id), price=price, volume=volume)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))

    def close_position(self, ticket: str, comment: str = "T58 Live close") -> OrderResult:
        try:
            price = None
            try:
                for p in self.get_open_positions():
                    if p.ticket == str(ticket):
                        price = None  # exit price is filled in by the relay from the quote/fill
                        break
            except Exception:
                pass
            self._req("DELETE", f"/trade/positions/{ticket}", json={"qty": "0"})
            return OrderResult(ok=True, message="Close order sent.", ticket=str(ticket), price=price)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))
