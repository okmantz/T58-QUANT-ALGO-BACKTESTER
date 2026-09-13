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

HONESTY NOTE (see broker_base.py's module docstring): this has been
written against cTrader Open API's public protobuf message definitions
and OAuth2 docs, but has not been run against a live cTrader account by
anyone on this project. Test it against a demo account first. The
Twisted reactor can only run once per process -- if you already run
another Twisted-based integration in this app, they need to share the
single global reactor rather than each starting their own.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import pandas as pd

from app.live_deploy.broker_base import (
    BrokerAdapter, ConnectionResult, OpenPosition, OrderResult,
)

try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol
    from ctrader_open_api.endpoints import EndPoints
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq, ProtoOAAccountAuthReq, ProtoOAGetTrendbarsReq,
        ProtoOANewOrderReq, ProtoOAClosePositionReq, ProtoOAReconcileReq,
        ProtoOATraderReq, ProtoOAGetAccountListByAccessTokenReq,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
        ProtoOAOrderType, ProtoOATradeSide, ProtoOATrendbarPeriod,
    )
    from twisted.internet import reactor, defer
    _CTRADER_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001
    Client = None  # type: ignore
    _CTRADER_IMPORT_ERROR = str(exc)

_TIMEFRAME_MAP = {
    1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4", 1440: "D1",
}


def is_available() -> bool:
    return Client is not None


def unavailable_reason() -> str:
    if is_available():
        return ""
    if _CTRADER_IMPORT_ERROR and "No module named" in _CTRADER_IMPORT_ERROR:
        return "The ctrader-open-api package isn't installed. Run `pip install ctrader-open-api`."
    return f"ctrader-open-api package failed to load: {_CTRADER_IMPORT_ERROR}"


class CTraderBrokerAdapter(BrokerAdapter):
    """One instance per live account. Bridges the package's async/Twisted
    callback style into the synchronous BrokerAdapter interface using a
    background reactor thread plus blocking queues -- the rest of the app
    (execution_engine.LiveExecutionSession) never has to know cTrader's
    API is event-driven under the hood."""

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
        self._reactor_thread: Optional[threading.Thread] = None
        self._symbol_cache: dict[str, dict] = {}

    # -- connection lifecycle ------------------------------------------------

    def connect(self) -> ConnectionResult:
        if not is_available():
            return ConnectionResult(ok=False, message=unavailable_reason())
        try:
            self.access_token = self._refresh_access_token()
        except Exception as exc:
            return ConnectionResult(ok=False, message=f"OAuth token refresh failed: {exc}")

        endpoint = EndPoints.PROTOBUF_LIVE_HOST if self.host == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self._client = Client(endpoint, EndPoints.PROTOBUF_PORT, TcpProtocol)

        started = threading.Event()
        error_box: dict = {}

        def _on_connected(_client):
            try:
                self._send_and_wait(ProtoOAApplicationAuthReq(
                    clientId=self.client_id, clientSecret=self.client_secret,
                ))
                self._send_and_wait(ProtoOAAccountAuthReq(
                    ctidTraderAccountId=self.account_id, accessToken=self.access_token,
                ))
            except Exception as exc:  # noqa: BLE001
                error_box["error"] = str(exc)
            finally:
                started.set()

        self._client.setConnectedCallback(_on_connected)
        self._client.startService()

        if self._reactor_thread is None or not self._reactor_thread.is_alive():
            self._reactor_thread = threading.Thread(
                target=lambda: reactor.run(installSignalHandlers=False), daemon=True,
            )
            self._reactor_thread.start()

        if not started.wait(timeout=self.response_timeout):
            return ConnectionResult(ok=False, message="Timed out connecting to cTrader Open API.")
        if "error" in error_box:
            return ConnectionResult(ok=False, message=error_box["error"])

        self._connected = True
        summary = self.account_summary() or {}
        return ConnectionResult(
            ok=True, message=f"Connected ({self.host}).", account_login=str(self.account_id),
            account_server=f"cTrader-{self.host}", balance=summary.get("balance"),
            equity=summary.get("equity"), currency=summary.get("currency"),
        )

    def _refresh_access_token(self) -> str:
        """OAuth2 refresh-token grant -- a plain HTTPS POST, unrelated to
        the Twisted TCP connection above."""
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
        return data.get("accessToken") or data.get("access_token")

    def _send_and_wait(self, request):
        """Sends one protobuf request and blocks for its matching
        response via the package's deferred, converted to a plain
        threading-friendly wait. Raises on timeout or an error response."""
        result_q: "queue.Queue" = queue.Queue(maxsize=1)
        d = self._client.send(request)
        d.addCallback(lambda resp: result_q.put(("ok", resp)))
        d.addErrback(lambda err: result_q.put(("err", err)))
        try:
            status, payload = result_q.get(timeout=self.response_timeout)
        except queue.Empty:
            raise TimeoutError(f"No response from cTrader for {type(request).__name__} within {self.response_timeout}s")
        if status == "err":
            raise RuntimeError(str(payload))
        return payload

    def disconnect(self) -> None:
        try:
            if self._client is not None:
                self._client.stopService()
        except Exception:
            pass
        self._connected = False

    def is_alive(self) -> bool:
        return self._connected and self._client is not None

    def account_summary(self) -> Optional[dict]:
        if not self._connected:
            return None
        try:
            resp = self._send_and_wait(ProtoOATraderReq(ctidTraderAccountId=self.account_id))
            trader = resp.trader
            return {
                "login": self.account_id, "server": f"cTrader-{self.host}",
                "balance": trader.balance / 100.0,  # cTrader reports money in cents/hundredths
                "equity": getattr(trader, "equity", trader.balance) / 100.0,
                "margin_free": None, "currency": getattr(trader, "depositCurrency", None),
            }
        except Exception:
            self._connected = False
            return None

    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        period = _TIMEFRAME_MAP.get(timeframe_minutes)
        if period is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_minutes} minutes. Supported: {sorted(_TIMEFRAME_MAP)}.")
        symbol_id = self._resolve_symbol_id(symbol)
        resp = self._send_and_wait(ProtoOAGetTrendbarsReq(
            ctidTraderAccountId=self.account_id, symbolId=symbol_id,
            period=getattr(ProtoOATrendbarPeriod, period), count=count + 1,
        ))
        rows = []
        for bar in resp.trendbar:
            low = bar.low / 100000.0
            rows.append({
                "timestamp": pd.to_datetime(bar.utcTimestampInMinutes * 60, unit="s", utc=True),
                "open": low + bar.deltaOpen / 100000.0, "high": low + bar.deltaHigh / 100000.0,
                "low": low, "close": low + bar.deltaClose / 100000.0, "volume": float(bar.volume),
            })
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df.iloc[:-1] if len(df) > 1 else df  # drop still-forming bar, matching MT5Connector's contract

    def _resolve_symbol_id(self, symbol: str) -> int:
        """cTrader addresses symbols by an integer id, not a name -- the
        mapping is per-account (via ProtoOASymbolsListReq) and cached
        here since it doesn't change mid-session. Left as a documented
        TODO rather than guessed: wire this to a ProtoOASymbolsListReq
        call and a name->id lookup the first time a symbol is needed."""
        raise NotImplementedError(
            "Symbol name -> cTrader symbolId lookup isn't wired up yet. Call "
            "ProtoOASymbolsListReq(ctidTraderAccountId=...) once after connect() and cache the "
            "name->id map on this instance before using fetch_completed_bars/place_market_order."
        )

    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]:
        resp = self._send_and_wait(ProtoOAReconcileReq(ctidTraderAccountId=self.account_id))
        out = []
        for pos in resp.position:
            sym_name = self._symbol_cache.get(pos.tradeData.symbolId, {}).get("name", str(pos.tradeData.symbolId))
            if symbol and sym_name != symbol:
                continue
            out.append(OpenPosition(
                ticket=str(pos.positionId), symbol=sym_name,
                direction=1 if pos.tradeData.tradeSide == ProtoOATradeSide.BUY else -1,
                volume=pos.tradeData.volume / 100.0, open_price=pos.price,
                sl=getattr(pos, "stopLoss", 0.0) or 0.0, tp=getattr(pos, "takeProfit", 0.0) or 0.0,
                open_time=pd.to_datetime(pos.tradeData.openTimestamp, unit="ms", utc=True),
                profit=getattr(pos, "usdConversionRate", 1.0) and pos.grossProfit / 100.0 if hasattr(pos, "grossProfit") else 0.0,
            ))
        return out

    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Live", deviation: int = 20,
    ) -> OrderResult:
        try:
            symbol_id = self._resolve_symbol_id(symbol)
            req = ProtoOANewOrderReq(
                ctidTraderAccountId=self.account_id, symbolId=symbol_id,
                orderType=ProtoOAOrderType.MARKET,
                tradeSide=ProtoOATradeSide.BUY if direction == 1 else ProtoOATradeSide.SELL,
                volume=int(volume * 100), comment=comment[:100],
            )
            if sl_price:
                req.stopLoss = sl_price
            if tp_price:
                req.takeProfit = tp_price
            resp = self._send_and_wait(req)
            return OrderResult(ok=True, message="Filled.", ticket=str(getattr(resp, "orderId", None)))
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))

    def close_position(self, ticket: str, comment: str = "T58 Live close") -> OrderResult:
        try:
            resp = self._send_and_wait(ProtoOAClosePositionReq(
                ctidTraderAccountId=self.account_id, positionId=int(ticket),
            ))
            return OrderResult(ok=True, message="Closed.", ticket=ticket)
        except Exception as exc:
            return OrderResult(ok=False, message=str(exc))
