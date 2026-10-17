"""
cTrader Open API adapter -- closes the FTMO/FundedNext/E8/The5%ers
cTrader-account gap flagged in prop_firms.py.

UNLIKE MT4/MT5 (a local terminal the MetaTrader5 package talks to over
IPC), cTrader Open API is a remote TCP+Protobuf service you authenticate
against with an OAuth2 app -- there is no local terminal to install.
Setup, once, free, no paid tier involved:
  1. Create a free account at https://openapi.ctrader.com and register an
     "application" -- this gives you a Client ID and Client Secret.
  2. Run that app's OAuth2 authorization-code flow once in a browser
     (the portal walks you through the URL) to get a long-lived refresh
     token for YOUR trading account. This adapter only needs the
     resulting client_id / client_secret / refresh_token -- it refreshes
     the short-lived access token itself.
  3. Pick host=demo for a demo/practice account, host=live for a real
     funded account, matching whichever your prop firm issued.

Requires `pip install ctrader-open-api` (official Spotware package,
Twisted-based, free, no paid tier). Import is guarded exactly like
MetaTrader5 in mt5_connector.py: everywhere else in the app loads cleanly
even when it's absent.

HONESTY NOTE: written against the installed ctrader-open-api package and its
protobuf definitions (field names checked against the package), and unit
tested with a fake transport; not exercised against a live cTrader account.
Test on a demo account first. All socket I/O runs on the Twisted reactor
thread; this adapter talks to it only via blockingCallFromThread, so calling
it from Flask/worker threads is safe. The reactor runs once per process.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import pandas as pd

from app.live_deploy.broker_base import (
    BrokerAdapter, ConnectionResult, OpenPosition, OrderResult,
)

try:
    from ctrader_open_api import Client, TcpProtocol
    from ctrader_open_api.endpoints import EndPoints
    from ctrader_open_api.protobuf import Protobuf
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAAccountAuthReq,
        ProtoOAGetTrendbarsReq,
        ProtoOANewOrderReq,
        ProtoOAClosePositionReq,
        ProtoOAAmendPositionSLTPReq,
        ProtoOAReconcileReq,
        ProtoOATraderReq,
        ProtoOASymbolsListReq,
        ProtoOASymbolByIdReq,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
        ProtoOAOrderType, ProtoOATradeSide, ProtoOATrendbarPeriod,
    )
    from twisted.internet import reactor, threads
    _CTRADER_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001
    Client = None  # type: ignore
    _CTRADER_IMPORT_ERROR = str(exc)

_TIMEFRAME_MAP = {
    1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4", 1440: "D1",
}
_REACTOR_LOCK = threading.Lock()


def is_available() -> bool:
    return Client is not None


def unavailable_reason() -> str:
    if is_available():
        return ""
    if _CTRADER_IMPORT_ERROR and "No module named" in _CTRADER_IMPORT_ERROR:
        return "The ctrader-open-api package isn't installed. Run `pip install ctrader-open-api`."
    return f"ctrader-open-api package failed to load: {_CTRADER_IMPORT_ERROR}"


def _ensure_reactor() -> None:
    """Start the (one-per-process) Twisted reactor on a daemon thread."""
    with _REACTOR_LOCK:
        if reactor.running:
            return
        t = threading.Thread(target=lambda: reactor.run(installSignalHandlers=False),
                             daemon=True, name="t58-ctrader-reactor")
        t.start()
        for _ in range(100):
            if reactor.running:
                return
            time.sleep(0.05)


class CTraderBrokerAdapter(BrokerAdapter):
    """One instance per account. Synchronous facade over the Twisted client."""

    platform_name = "cTrader"

    def __init__(self, client_id: str, client_secret: str, refresh_token: str,
                 ctid_trader_account_id: int, host: str = "demo", response_timeout: float = 15.0):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token: str | None = None
        self.account_id = int(ctid_trader_account_id)
        self.host = host  # "demo" or "live"
        self.response_timeout = response_timeout
        self._client = None
        self._connected = False
        self._symbol_cache: dict[int, dict] = {}   # symbolId -> {"name", "lot_size", ...}
        self._money_digits = 2

    # -- transport -----------------------------------------------------------

    def _call(self, request):
        """Send one request from any thread, wait for the reply, return the
        decoded payload message. Raises on timeout, transport failure or a
        cTrader error response."""
        timeout = self.response_timeout
        try:
            raw = threads.blockingCallFromThread(
                reactor, self._client.send, request, responseTimeoutInSeconds=timeout)
        except Exception as exc:  # twisted Failure wrappers / TimeoutError
            raise RuntimeError(f"{type(request).__name__} failed: {exc!r}") from exc
        payload = Protobuf.extract(raw)
        name = type(payload).__name__
        if name in ("ProtoOAErrorRes", "ProtoErrorRes", "ProtoOAOrderErrorEvent"):
            raise RuntimeError(f"cTrader error {getattr(payload, 'errorCode', '')}: {getattr(payload, 'description', '')}")
        if name == "ProtoOAExecutionEvent" and payload.executionType in (7, 8):  # rejected
            raise RuntimeError(f"cTrader rejected the order (errorCode={payload.errorCode or 'n/a'})")
        return payload

    # -- connection lifecycle ------------------------------------------------

    def connect(self) -> ConnectionResult:
        if not is_available():
            return ConnectionResult(ok=False, message=unavailable_reason())
        try:
            self.access_token = self._refresh_access_token()
        except Exception as exc:
            return ConnectionResult(ok=False, message=f"OAuth token refresh failed: {exc}")

        self.disconnect()
        endpoint = EndPoints.PROTOBUF_LIVE_HOST if self.host == "live" else EndPoints.PROTOBUF_DEMO_HOST
        connected = threading.Event()
        try:
            _ensure_reactor()

            def _start():
                self._client = Client(endpoint, EndPoints.PROTOBUF_PORT, TcpProtocol)
                self._client.setConnectedCallback(lambda _c: connected.set())
                self._client.setDisconnectedCallback(lambda _c, _r: setattr(self, "_connected", False))
                self._client.startService()

            threads.blockingCallFromThread(reactor, _start)
            if not connected.wait(timeout=self.response_timeout):
                self.disconnect()
                return ConnectionResult(ok=False, message="Timed out connecting to cTrader Open API.")
            self._call(ProtoOAApplicationAuthReq(clientId=self.client_id, clientSecret=self.client_secret))
            self._call(ProtoOAAccountAuthReq(ctidTraderAccountId=self.account_id, accessToken=self.access_token))
        except Exception as exc:  # noqa: BLE001
            self.disconnect()
            return ConnectionResult(ok=False, message=f"cTrader connect failed: {exc}")

        self._connected = True
        self._symbol_cache = {}
        summary = self.account_summary() or {}
        return ConnectionResult(
            ok=True, message=f"Connected ({self.host}).", account_login=str(self.account_id),
            account_server=f"cTrader-{self.host}", balance=summary.get("balance"),
            equity=summary.get("equity"), currency=summary.get("currency"),
        )

    def _refresh_access_token(self) -> str:
        """OAuth2 refresh-token grant (plain HTTPS)."""
        import requests
        resp = requests.post(
            "https://openapi.ctrader.com/apps/token",
            data={
                "grant_type": "refresh_token", "refresh_token": self.refresh_token,
                "client_id": self.client_id, "client_secret": self.client_secret,
            }, timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if "accessToken" not in data and "access_token" not in data:
            raise RuntimeError(f"Unexpected token response: {data}")
        new_refresh = data.get("refreshToken") or data.get("refresh_token")
        if new_refresh:
            self.refresh_token = new_refresh
        return data.get("accessToken") or data.get("access_token")

    def disconnect(self) -> None:
        client, self._client = self._client, None
        self._connected = False
        if client is not None:
            try:
                threads.blockingCallFromThread(reactor, client.stopService)
            except Exception:  # noqa: BLE001
                pass

    def is_alive(self) -> bool:
        return bool(self._connected and self._client is not None and getattr(self._client, "isConnected", True))

    def account_summary(self) -> Optional[dict]:
        if not self._client:
            return None
        try:
            trader = self._call(ProtoOATraderReq(ctidTraderAccountId=self.account_id)).trader
            self._money_digits = int(getattr(trader, "moneyDigits", 2) or 2)
            bal = trader.balance / (10 ** self._money_digits)
            return {"login": self.account_id, "server": f"cTrader-{self.host}",
                    "balance": bal, "equity": bal, "margin_free": None, "currency": None}
        except Exception:
            self._connected = False
            return None

    # -- symbols -------------------------------------------------------------

    def _ensure_symbol_cache(self) -> None:
        if self._symbol_cache:
            return
        resp = self._call(ProtoOASymbolsListReq(ctidTraderAccountId=self.account_id))
        for s in resp.symbol:
            self._symbol_cache[int(s.symbolId)] = {"name": s.symbolName, "lot_size": None, "digits": None}

    def _symbol_details(self, symbol_id: int) -> dict:
        meta = self._symbol_cache[symbol_id]
        if meta.get("lot_size") is None:
            resp = self._call(ProtoOASymbolByIdReq(ctidTraderAccountId=self.account_id, symbolId=[symbol_id]))
            for sym in resp.symbol:
                if int(sym.symbolId) == symbol_id:
                    meta.update(lot_size=int(sym.lotSize or 100000), digits=int(sym.digits),
                                min_volume=int(sym.minVolume or 1), step_volume=int(sym.stepVolume or 1))
            meta.setdefault("lot_size", 100000)
            if meta["lot_size"] is None:
                meta["lot_size"] = 100000
        return meta

    def _resolve_symbol_id(self, symbol: str) -> int:
        self._ensure_symbol_cache()
        want = str(symbol).upper()
        for symbol_id, meta in self._symbol_cache.items():
            if str(meta.get("name", "")).upper() == want:
                return symbol_id
        known = sorted(meta.get("name", "") for meta in self._symbol_cache.values())
        raise ValueError(
            f"cTrader symbol '{symbol}' was not found on this account. "
            f"Known symbols ({len(known)}): {', '.join(known[:25])}" + (", ..." if len(known) > 25 else "")
        )

    # -- market data ---------------------------------------------------------

    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        period = _TIMEFRAME_MAP.get(timeframe_minutes)
        if period is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_minutes} minutes. Supported: {sorted(_TIMEFRAME_MAP)}.")
        symbol_id = self._resolve_symbol_id(symbol)
        now_ms = int(time.time() * 1000)
        span = (count + 5) * timeframe_minutes * 60_000 * 2
        resp = self._call(ProtoOAGetTrendbarsReq(
            ctidTraderAccountId=self.account_id, symbolId=symbol_id,
            period=getattr(ProtoOATrendbarPeriod, period), fromTimestamp=now_ms - span,
            toTimestamp=now_ms, count=count + 1,
        ))
        rows = []
        for bar in resp.trendbar:
            low = bar.low / 100000.0
            rows.append({
                "timestamp": pd.to_datetime(bar.utcTimestampInMinutes * 60, unit="s", utc=True),
                "open": low + bar.deltaOpen / 100000.0, "high": low + bar.deltaHigh / 100000.0,
                "low": low, "close": low + bar.deltaClose / 100000.0, "volume": float(bar.volume),
            })
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        df = df.iloc[:-1] if len(df) > 1 else df  # drop still-forming bar
        return df.tail(count).reset_index(drop=True)

    # -- positions -----------------------------------------------------------

    def _reconcile(self):
        return list(self._call(ProtoOAReconcileReq(ctidTraderAccountId=self.account_id)).position)

    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]:
        try:
            raw = self._reconcile()
            self._ensure_symbol_cache()
        except Exception:
            return []
        out = []
        for pos in raw:
            sid = int(pos.tradeData.symbolId)
            sym_name = self._symbol_cache.get(sid, {}).get("name", str(sid))
            if symbol and sym_name.upper() != str(symbol).upper():
                continue
            lot = self._symbol_details(sid)["lot_size"] if sid in self._symbol_cache else 100000
            out.append(OpenPosition(
                ticket=str(pos.positionId), symbol=sym_name,
                direction=1 if pos.tradeData.tradeSide == ProtoOATradeSide.BUY else -1,
                volume=pos.tradeData.volume / float(lot), open_price=float(pos.price),
                sl=float(getattr(pos, "stopLoss", 0.0) or 0.0), tp=float(getattr(pos, "takeProfit", 0.0) or 0.0),
                open_time=pd.to_datetime(pos.tradeData.openTimestamp, unit="ms", utc=True),
                profit=0.0,
            ))
        return out

    def to_broker_qty(
        self, units: float, contract_size: Optional[float], units_per_lot: Optional[float],
    ) -> float:
        # LOTS floored to 0.01; place_market_order converts to the symbol's
        # protocol volume (lots * lotSize). Refuses (0.0) without units_per_lot.
        if not units_per_lot or units_per_lot <= 0:
            return 0.0
        return math.floor(units / units_per_lot * 100 + 1e-9) / 100.0

    # -- orders --------------------------------------------------------------

    @staticmethod
    def _round_volume(lots: float, meta: dict) -> int:
        vol = int(round(lots * meta["lot_size"]))
        step = max(int(meta.get("step_volume") or 1), 1)
        vol = (vol // step) * step
        return max(vol, int(meta.get("min_volume") or 0)) if vol > 0 else 0

    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Live", deviation: int = 20,
    ) -> OrderResult:
        # `volume` is broker-native LOTS (see to_broker_qty).
        try:
            symbol_id = self._resolve_symbol_id(symbol)
            meta = self._symbol_details(symbol_id)
            proto_volume = self._round_volume(volume, meta)
            if proto_volume <= 0:
                return OrderResult(ok=False, message=f"{volume} lots is below the minimum size for {symbol}.")
            before = {p.positionId for p in self._reconcile()}
            ev = self._call(ProtoOANewOrderReq(
                ctidTraderAccountId=self.account_id, symbolId=symbol_id,
                orderType=ProtoOAOrderType.MARKET,
                tradeSide=ProtoOATradeSide.BUY if direction == 1 else ProtoOATradeSide.SELL,
                volume=proto_volume, comment=comment[:100],
            ))
            price = position_id = None
            if getattr(ev, "executionType", None) == 3 and ev.deal.executionPrice:   # ORDER_FILLED
                price = float(ev.deal.executionPrice)
                position_id = int(ev.deal.positionId or ev.position.positionId or 0) or None
            if position_id is None or price is None:
                for _ in range(6):                                  # accepted but not yet filled
                    new = [p for p in self._reconcile()
                           if p.positionId not in before and int(p.tradeData.symbolId) == symbol_id]
                    if new:
                        position_id, price = int(new[0].positionId), float(new[0].price)
                        break
                    time.sleep(0.5)
            if position_id is None:
                return OrderResult(ok=False, message="cTrader accepted the order but no position appeared; check the platform.")
            if sl_price or tp_price:
                try:
                    req = ProtoOAAmendPositionSLTPReq(ctidTraderAccountId=self.account_id, positionId=position_id)
                    if sl_price:
                        req.stopLoss = sl_price
                    if tp_price:
                        req.takeProfit = tp_price
                    self._call(req)
                except Exception as exc:  # noqa: BLE001
                    return OrderResult(ok=True, message=f"Filled; SL/TP not set: {exc}",
                                       ticket=str(position_id), price=price, volume=volume)
            return OrderResult(ok=True, message="Filled.", ticket=str(position_id), price=price, volume=volume)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))

    def close_position(self, ticket: str, comment: str = "T58 Live close") -> OrderResult:
        try:
            pos = next((p for p in self._reconcile() if str(p.positionId) == str(ticket)), None)
            if pos is None:
                return OrderResult(ok=False, message=f"Position {ticket} not found on cTrader.")
            ev = self._call(ProtoOAClosePositionReq(
                ctidTraderAccountId=self.account_id, positionId=int(ticket), volume=pos.tradeData.volume))
            px = float(ev.deal.executionPrice) if getattr(ev, "deal", None) is not None and ev.deal.executionPrice else None
            return OrderResult(ok=True, message="Closed.", ticket=str(ticket), price=px)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))
