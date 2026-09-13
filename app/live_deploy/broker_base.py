"""
BrokerAdapter -- the common interface every platform connector (MT5,
cTrader, Tradovate, TradeLocker, DXtrade) implements, so the rest of the
app (execution_engine.LiveExecutionSession, the Deploy Live routes) can
run a strategy against ANY connected funded account without knowing or
caring which platform it lives on.

This is new: previously app.forward_test.mt5_connector.MT5Connector was
the only connector in the codebase, and it was wired directly into
ForwardTestSession with no interface boundary -- there was nothing another
platform's connector could implement to slot into the same code path.

The dataclasses here (ConnectionResult, OrderResult, OpenPosition) are
intentionally IDENTICAL in shape to the ones already defined inline in
app.forward_test.mt5_connector, so MT5Connector can be wrapped by
broker_mt5.MT5BrokerAdapter with zero changes to MT5Connector itself --
see that file's docstring for why MT5Connector stays untouched rather than
being refactored to subclass this.

Every new adapter in this package (broker_ctrader.py, broker_tradovate.py,
broker_tradelocker.py, broker_dxtrade.py) is written against each
platform's real, documented, publicly-available API. None of them have
been exercised against a live account by anyone on this project -- there
was no account to test against. Treat first use with any of them as an
integration test: run it against a demo/practice account before pointing
it at a funded one, and expect to file a bug against whichever adapter you
hit first. This is explicitly flagged rather than hidden because the
alternative (a confident-looking connector that silently mis-executes on a
funded account) is worse than an honest "not yet battle-tested" label.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ConnectionResult:
    ok: bool
    message: str
    account_login: str | None = None
    account_server: str | None = None
    balance: float | None = None
    equity: float | None = None
    currency: str | None = None
    resolved_terminal_path: str | None = None  # MT5-only; always None for REST-based platforms


@dataclass
class OrderResult:
    ok: bool
    message: str
    ticket: str | None = None   # order/deal/position id, stringified -- platforms use ints, uuids, or opaque strings
    price: float | None = None
    volume: float | None = None


@dataclass
class OpenPosition:
    ticket: str
    symbol: str
    direction: int  # 1 long, -1 short
    volume: float
    open_price: float
    sl: float
    tp: float
    open_time: pd.Timestamp
    profit: float


class BrokerAdapter(ABC):
    """Every method here mirrors a method MT5Connector already had, so
    execution_engine.LiveExecutionSession (the generalized successor to
    app.forward_test.engine.ForwardTestSession) can treat any adapter
    identically. Implementations should never raise out of these methods
    for ordinary failure conditions (bad credentials, rejected order,
    network hiccup) -- return ok=False / None with a human-readable
    message instead, matching MT5Connector's existing convention, since
    LiveExecutionSession logs `.message` directly into the session journal
    a real user will read while real capital is on the line.
    """

    platform_name: str = "unknown"

    @abstractmethod
    def connect(self) -> ConnectionResult: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_alive(self) -> bool: ...

    def ensure_connected(self, max_attempts: int = 3, retry_delay_seconds: float = 2.0) -> ConnectionResult:
        """Default retry wrapper, identical in behavior to
        MT5Connector.ensure_connected. Adapters can override this if a
        platform needs different reconnect semantics (e.g. refreshing an
        OAuth token instead of a full re-login)."""
        import time as _time
        if self.is_alive():
            return ConnectionResult(ok=True, message="Connected.")
        last_result = ConnectionResult(ok=False, message="Not connected.")
        for attempt in range(1, max_attempts + 1):
            last_result = self.connect()
            if last_result.ok:
                return last_result
            if attempt < max_attempts:
                _time.sleep(retry_delay_seconds)
        return last_result

    @abstractmethod
    def account_summary(self) -> Optional[dict]: ...

    @abstractmethod
    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame: ...

    def latest_completed_bar_time(self, symbol: str, timeframe_minutes: int) -> Optional[pd.Timestamp]:
        df = self.fetch_completed_bars(symbol, timeframe_minutes, 1)
        if df.empty:
            return None
        return df["timestamp"].iloc[-1]

    @abstractmethod
    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]: ...

    @abstractmethod
    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Live", deviation: int = 20,
    ) -> OrderResult: ...

    @abstractmethod
    def close_position(self, ticket: str, comment: str = "T58 Live close") -> OrderResult: ...

    def close_all(self, symbol: Optional[str] = None) -> list[OrderResult]:
        return [self.close_position(p.ticket) for p in self.get_open_positions(symbol)]
