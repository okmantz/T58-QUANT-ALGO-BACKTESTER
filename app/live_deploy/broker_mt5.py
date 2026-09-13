"""
Wraps the existing, already-working app.forward_test.mt5_connector.MT5Connector
so it satisfies the new BrokerAdapter interface, without changing
MT5Connector itself.

Why wrap instead of refactor MT5Connector to subclass BrokerAdapter
directly: MT5Connector is used today by the Forward Test tab against a
*demo* account, is already tested in that role, and its OrderResult/
OpenPosition/ConnectionResult field types (int tickets, etc.) are baked
into app.forward_test.engine and app.forward_test.journal. Changing its
field types to match BrokerAdapter's (stringified tickets, so every
platform can share one type) would risk breaking Forward Test for a
purely cosmetic gain. This thin wrapper is the entire cost of
interoperability instead.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from app.forward_test.mt5_connector import MT5Connector
from app.live_deploy.broker_base import (
    BrokerAdapter, ConnectionResult, OpenPosition, OrderResult,
)


class MT5BrokerAdapter(BrokerAdapter):
    platform_name = "MT4/MT5"

    def __init__(self, login: str, password: str, server: str, terminal_path: str = ""):
        self._conn = MT5Connector(login=login, password=password, server=server, terminal_path=terminal_path)

    def connect(self) -> ConnectionResult:
        r = self._conn.connect()
        return ConnectionResult(
            ok=r.ok, message=r.message,
            account_login=str(r.account_login) if r.account_login is not None else None,
            account_server=r.account_server, balance=r.balance, equity=r.equity,
            currency=r.currency, resolved_terminal_path=r.resolved_terminal_path,
        )

    def disconnect(self) -> None:
        self._conn.disconnect()

    def is_alive(self) -> bool:
        return self._conn.is_alive()

    def account_summary(self) -> Optional[dict]:
        return self._conn.account_summary()

    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        return self._conn.fetch_completed_bars(symbol, timeframe_minutes, count)

    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]:
        return [
            OpenPosition(
                ticket=str(p.ticket), symbol=p.symbol, direction=p.direction, volume=p.volume,
                open_price=p.open_price, sl=p.sl, tp=p.tp, open_time=p.open_time, profit=p.profit,
            )
            for p in self._conn.get_open_positions(symbol)
        ]

    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Live", deviation: int = 20,
    ) -> OrderResult:
        r = self._conn.place_market_order(
            symbol, direction, volume, sl_price=sl_price, tp_price=tp_price,
            comment=comment[:31], deviation=deviation,  # MT5 comment field is capped at 31 chars
        )
        return OrderResult(ok=r.ok, message=r.message, ticket=str(r.ticket) if r.ticket is not None else None,
                            price=r.price, volume=r.volume)

    def close_position(self, ticket: str, comment: str = "T58 Live close") -> OrderResult:
        r = self._conn.close_position(int(ticket), comment=comment[:31])
        return OrderResult(ok=r.ok, message=r.message, ticket=str(r.ticket) if r.ticket is not None else None,
                            price=r.price, volume=r.volume)
