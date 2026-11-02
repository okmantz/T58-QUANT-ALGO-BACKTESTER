"""End-to-end test of T58's own MT5 bridge: a real RPyC server + the real client, with a stub MetaTrader5 module."""
import socket
import sys
import threading
import time
import types
from collections import namedtuple

import numpy as np
import pytest

pytest.importorskip("rpyc")

from app.forward_test import mt5_bridge, mt5_bridge_manager as mgr, mt5_connector as mc, mt5_rpc_server as srv

Acct = namedtuple("Acct", "login server balance equity currency")
Pos = namedtuple("Pos", "ticket type symbol volume price_open sl tp time profit")
Res = namedtuple("Res", "retcode order deal comment price volume")
Tick = namedtuple("Tick", "bid ask time")


def _stub():
    m = types.ModuleType("MetaTrader5")
    m.TIMEFRAME_M15 = 15
    m.ORDER_TYPE_BUY, m.ORDER_TYPE_SELL, m.TRADE_ACTION_DEAL, m.TRADE_RETCODE_DONE = 0, 1, 1, 10009
    m.ORDER_TIME_GTC, m.ORDER_FILLING_IOC, m.DEAL_ENTRY_OUT = 0, 1, 1
    state = {"init": None, "last": None}
    m.initialize = lambda **kw: state.update(init=kw) or True
    m.login = lambda *a, **k: True
    m.shutdown = lambda: True
    m.last_error = lambda: (1, "Success")
    m.account_info = lambda: Acct(1234, "Demo", 10000.0, 10010.5, "USD")
    m.terminal_info = lambda: Acct(0, "t", 0.0, 0.0, "")
    dt = np.dtype([("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"), ("tick_volume", "<u8")])
    m.copy_rates_from_pos = lambda sym, tf, pos, n: np.array([(1700000000 + 900 * i, 1.0 + i, 2.0 + i, 0.5, 1.5 + i, 10) for i in range(n)], dtype=dt)
    m.positions_get = lambda **k: (Pos(7, 0, "XAUUSD", 0.1, 2000.0, 1990.0, 2020.0, 1700000000, 1.5),)
    m.order_send = lambda req: state.update(last=req) or Res(10009, 55, 66, "done", 2000.5, 0.1)
    m.symbol_info_tick = lambda sym: Tick(2000.0, 2000.5, 1700000000)
    m.symbol_select = lambda s, e=True: True
    m.positions_total = lambda: 1
    m.history_deals_get = lambda a, b, **k: None
    m._state = state
    return m


@pytest.fixture()
def bridge(monkeypatch):
    stub = _stub()
    monkeypatch.setattr(srv, "_MT5", stub)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    from rpyc.utils.server import ThreadedServer
    server = ThreadedServer(srv.T58MT5Service, hostname="127.0.0.1", port=port,
                            protocol_config={"allow_public_attrs": False, "allow_pickle": False})
    threading.Thread(target=server.start, daemon=True).start()
    for _ in range(50):
        if mgr.port_open("127.0.0.1", port):
            break
        time.sleep(0.1)
    yield stub, port
    server.close()


def test_client_roundtrip(bridge):
    stub, port = bridge
    c = mt5_bridge.MT5BridgeClient("127.0.0.1", port)
    assert c.TIMEFRAME_M15 == 15 and c.ORDER_TYPE_SELL == 1
    assert c.initialize(path=r"C:\MT5\terminal64.exe", timeout=1000) is True
    assert stub._state["init"]["path"] == r"C:\MT5\terminal64.exe"
    info = c.account_info()
    assert info.login == 1234 and info.equity == 10010.5 and info._asdict()["currency"] == "USD"
    assert c.last_error() == (1, "Success")
    rates = c.copy_rates_from_pos("XAUUSD", c.TIMEFRAME_M15, 0, 5)
    assert isinstance(rates, np.ndarray) and len(rates) == 5 and rates["close"][2] == 3.5
    pos = c.positions_get(symbol="XAUUSD")
    assert pos[0].ticket == 7
    res = c.order_send({"action": c.TRADE_ACTION_DEAL, "volume": 0.1})
    assert res.retcode == c.TRADE_RETCODE_DONE and stub._state["last"]["volume"] == 0.1
    assert c.history_deals_get(__import__("datetime").datetime(2024, 1, 1), __import__("datetime").datetime(2024, 1, 2)) is None
    with pytest.raises(AttributeError):
        c.not_a_function


def test_disallowed_function_rejected(bridge):
    _, port = bridge
    c = mt5_bridge.MT5BridgeClient("127.0.0.1", port)
    with pytest.raises(Exception):
        c._conn.root.call("__import__", "[]", "{}", "")


def test_token_enforced(bridge, monkeypatch):
    _, port = bridge
    monkeypatch.setattr(srv.T58MT5Service, "token", "secret")
    with pytest.raises(Exception):
        mt5_bridge.MT5BridgeClient("127.0.0.1", port, token="wrong")
    assert mt5_bridge.MT5BridgeClient("127.0.0.1", port, token="secret").account_info().login == 1234


def test_full_connector_flow_over_real_bridge(bridge, monkeypatch):
    """Save-account -> Connect -> bars -> positions -> order, all through the real client/server, no mt5linux."""
    stub, port = bridge
    monkeypatch.setattr(mc, "mt5", None)
    monkeypatch.setattr(mc, "_NATIVE_MT5", None)
    monkeypatch.setattr(mc, "_BRIDGE_KEY", None)
    monkeypatch.setattr(mc, "platform_kind", lambda: "linux")
    monkeypatch.setattr(mc.MT5Connector, "launch_terminal", lambda self, wait_seconds=45.0: None)
    conn = mc.MT5Connector("1234", "pw", "Demo", "", bridge_host="127.0.0.1", bridge_port=port)
    res = conn.connect()
    assert res.ok, res.message
    assert mc.backend_name() == "bridge"
    assert conn.is_alive()
    pos = conn.get_open_positions("XAUUSD")
    assert pos and pos[0].ticket == 7
    order = conn.place_market_order("XAUUSD", 1, 0.1, 1990.0, 2020.0)
    assert order.ok, order.message
    assert stub._state["last"]["type"] == 0 and stub._state["last"]["sl"] == 1990.0
    bars = conn.fetch_completed_bars("XAUUSD", 15, 3)
    assert len(bars) >= 2


def test_manager_reports_missing_wine(monkeypatch):
    monkeypatch.setattr(mgr, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(mgr, "find_wine", lambda: None)
    monkeypatch.setattr(mgr.sys, "platform", "linux")
    ok, msg = mgr.ensure_bridge("localhost", 18812)
    assert not ok and "Wine" in msg


def test_wine_path():
    assert mgr.wine_path("/home/u/x/mt5_rpc_server.py") == "Z:\\home\\u\\x\\mt5_rpc_server.py"


def test_manager_autostarts_server_with_fake_wine(tmp_path, monkeypatch):
    """Fake `wine` (a shell script that runs host python3) + fake python.exe + stub MetaTrader5: ensure_bridge must find the
    prefix, verify packages, start the server and see it listening."""
    import os
    import stat
    prefix = tmp_path / "pfx"
    exe = prefix / "drive_c" / "Program Files" / "MT5" / "terminal64.exe"
    exe.parent.mkdir(parents=True); exe.write_bytes(b"MZ")
    (prefix / "drive_c" / "Python311").mkdir(); (prefix / "drive_c" / "Python311" / "python.exe").write_bytes(b"MZ")
    stubdir = tmp_path / "stub"; stubdir.mkdir()
    (stubdir / "MetaTrader5.py").write_text("def last_error():\n    return (1,'ok')\n")
    wine = tmp_path / "wine"
    script = (
        "#!/bin/sh\nshift\n"
        "for a in \"$@\"; do case \"$a\" in Z:*) a=$(printf '%s' \"$a\" | sed 's/^Z://; s#\\\\#/#g');; esac; set -- \"$@\" \"$a\"; shift; done\n"
        "PYTHONPATH=\"" + str(stubdir) + "\" exec " + sys.executable + " \"$@\"\n")
    wine.write_text(script)
    wine.chmod(wine.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(mgr, "find_wine", lambda: str(wine))
    monkeypatch.setattr(mgr.sys, "platform", "linux")
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    try:
        ok, msg = mgr.ensure_bridge("localhost", port, str(exe), install=False, wait=20)
        assert ok, msg
        c = mt5_bridge.MT5BridgeClient("127.0.0.1", port)
        assert c.last_error() == (1, "ok")
    finally:
        mgr.stop_bridge()
