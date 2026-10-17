"""cTrader adapter logic with a fake transport (real protobuf messages, no sockets)."""
from __future__ import annotations

import pytest

pytest.importorskip("ctrader_open_api")

from ctrader_open_api.messages import OpenApiMessages_pb2 as m  # noqa: E402
from ctrader_open_api.messages import OpenApiModelMessages_pb2 as mm  # noqa: E402

from app.live_deploy import broker_ctrader as bc  # noqa: E402


def _adapter(handler):
    a = bc.CTraderBrokerAdapter("id", "sec", "rt", 123)
    a._client = object()
    a._connected = True
    a._call = lambda req: handler(req)
    return a


def _symbols():
    r = m.ProtoOASymbolsListRes()
    s = r.symbol.add()
    s.symbolId, s.symbolName = 7, "EURUSD"
    return r


def _by_id():
    r = m.ProtoOASymbolByIdRes()
    s = r.symbol.add()
    s.symbolId, s.lotSize, s.digits, s.minVolume, s.stepVolume = 7, 10_000_000, 5, 100_000, 100_000
    return r


def _pos(pid, price=1.1, vol=10_000_000, side=mm.ProtoOATradeSide.BUY):
    p = mm.ProtoOAPosition()
    p.positionId, p.price = pid, price
    p.tradeData.symbolId, p.tradeData.volume, p.tradeData.tradeSide = 7, vol, side
    p.tradeData.openTimestamp = 1_700_000_000_000
    return p


def test_volume_uses_symbol_lot_size_not_hundredths():
    sent = {}
    state = {"pos": []}

    def h(req):
        if isinstance(req, m.ProtoOASymbolsListReq):
            return _symbols()
        if isinstance(req, m.ProtoOASymbolByIdReq):
            return _by_id()
        if isinstance(req, m.ProtoOAReconcileReq):
            r = m.ProtoOAReconcileRes()
            r.position.extend(state["pos"])
            return r
        if isinstance(req, m.ProtoOANewOrderReq):
            sent["vol"] = req.volume
            ev = m.ProtoOAExecutionEvent()
            ev.executionType = 3
            ev.deal.executionPrice, ev.deal.positionId = 1.1234, 555
            return ev
        raise AssertionError(type(req))

    a = _adapter(h)
    r = a.place_market_order("EURUSD", 1, 0.5)
    assert r.ok and r.ticket == "555" and r.price == 1.1234
    assert sent["vol"] == 5_000_000          # 0.5 lot * lotSize(10,000,000 = 100k units in cents)


def test_accepted_then_position_polled(monkeypatch):
    monkeypatch.setattr(bc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def h(req):
        if isinstance(req, m.ProtoOASymbolsListReq):
            return _symbols()
        if isinstance(req, m.ProtoOASymbolByIdReq):
            return _by_id()
        if isinstance(req, m.ProtoOAReconcileReq):
            calls["n"] += 1
            r = m.ProtoOAReconcileRes()
            if calls["n"] >= 3:
                r.position.append(_pos(900, price=1.2))
            return r
        ev = m.ProtoOAExecutionEvent()
        ev.executionType = 2   # accepted only
        return ev

    r = _adapter(h).place_market_order("EURUSD", -1, 1.0)
    assert r.ok and r.ticket == "900" and r.price == 1.2


def test_close_sends_position_volume_and_reads_exit():
    got = {}

    def h(req):
        if isinstance(req, m.ProtoOASymbolsListReq):
            return _symbols()
        if isinstance(req, m.ProtoOAReconcileReq):
            r = m.ProtoOAReconcileRes()
            r.position.append(_pos(42, vol=2_000_000))
            return r
        if isinstance(req, m.ProtoOAClosePositionReq):
            got["vol"] = req.volume
            ev = m.ProtoOAExecutionEvent()
            ev.executionType = 3
            ev.deal.executionPrice = 1.31
            return ev
        raise AssertionError(type(req))

    r = _adapter(h).close_position("42")
    assert r.ok and r.price == 1.31 and got["vol"] == 2_000_000
    assert not _adapter(lambda req: m.ProtoOAReconcileRes()).close_position("1").ok


def test_positions_converted_to_lots():
    def h(req):
        if isinstance(req, m.ProtoOASymbolsListReq):
            return _symbols()
        if isinstance(req, m.ProtoOASymbolByIdReq):
            return _by_id()
        r = m.ProtoOAReconcileRes()
        r.position.append(_pos(1, vol=3_000_000, side=mm.ProtoOATradeSide.SELL))
        return r

    p = _adapter(h).get_open_positions("eurusd")[0]
    assert p.volume == pytest.approx(0.3) and p.direction == -1 and p.symbol == "EURUSD"


def test_unknown_symbol_and_error_surface():
    a = _adapter(lambda req: _symbols())
    r = a.place_market_order("XYZ", 1, 1.0)
    assert not r.ok and "not found" in r.message
