"""MT5 connector across operating systems, with fake backends (no terminal needed)."""
import sys
import types

import pytest

from app.forward_test import mt5_connector as mc


def test_platform_kind(monkeypatch):
    for plat, expect in (("win32", "windows"), ("darwin", "macos"), ("linux", "linux")):
        monkeypatch.setattr(mc.sys, "platform", plat)
        monkeypatch.setattr(mc.os, "name", "nt" if plat == "win32" else "posix")
        assert mc.platform_kind() == expect


def test_to_wine_path():
    assert mc.to_wine_path("/home/u/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe") == r"C:\Program Files\MetaTrader 5\terminal64.exe"
    assert mc.to_wine_path(r"C:\MT5\terminal64.exe") == r"C:\MT5\terminal64.exe"
    assert mc.to_wine_path("") == ""
    assert mc.to_wine_path("/opt/other/terminal64.exe") == "/opt/other/terminal64.exe"


def test_find_terminal_in_wine_prefix(tmp_path, monkeypatch):
    exe = tmp_path / ".wine" / "drive_c" / "Program Files" / "MetaTrader 5" / "terminal64.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    mac = tmp_path / "Library" / "Application Support" / "net.metaquotes.wine.metatrader5" / "drive_c" / "Program Files" / "Broker MT5" / "terminal64.exe"
    mac.parent.mkdir(parents=True)
    mac.write_bytes(b"MZ")
    monkeypatch.setattr(mc.os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
    monkeypatch.delenv("WINEPREFIX", raising=False)
    monkeypatch.setattr(mc, "platform_kind", lambda: "linux")
    found = mc.find_terminal_candidates()
    assert str(exe) in found and str(mac) in found


def test_unavailable_reason_is_os_specific(monkeypatch):
    monkeypatch.setattr(mc, "mt5", None)
    monkeypatch.setattr(mc, "_bridge_package_available", lambda: False)
    monkeypatch.setattr(mc, "platform_kind", lambda: "linux")
    assert "bridge" in mc.unavailable_reason().lower() and "Linux" in mc.unavailable_reason()
    monkeypatch.setattr(mc, "platform_kind", lambda: "macos")
    assert "macOS" in mc.unavailable_reason()
    monkeypatch.setattr(mc, "platform_kind", lambda: "windows")
    assert "MetaTrader5" in mc.unavailable_reason()


class _FakeAccount:
    login, server, balance, equity, currency = 1234, "Demo", 10000.0, 10010.0, "USD"


class _FakeBridgeMT5:
    def __init__(self, host="localhost", port=18812):
        self.host, self.port, self.init_kw = host, port, None
        self.TIMEFRAME_M15 = 15

    def initialize(self, **kw):
        self.init_kw = kw
        return True

    def account_info(self):
        return _FakeAccount()

    def login(self, *a, **k):
        return True

    def shutdown(self):
        return True

    def last_error(self):
        return (0, "ok")

    def terminal_info(self):
        return object()


def test_connect_through_bridge_translates_path(monkeypatch):
    fake_mod = types.SimpleNamespace(MetaTrader5=_FakeBridgeMT5)
    monkeypatch.setitem(sys.modules, "mt5linux", fake_mod)
    monkeypatch.setattr(mc, "mt5", None)
    monkeypatch.setattr(mc, "_NATIVE_MT5", None)
    monkeypatch.setattr(mc, "_BRIDGE_KEY", None)
    monkeypatch.setattr(mc, "_bridge_package_available", lambda: True)
    monkeypatch.setattr(mc, "platform_kind", lambda: "linux")
    monkeypatch.setattr(mc.MT5Connector, "launch_terminal", lambda self, wait_seconds=45.0: None)
    conn = mc.MT5Connector("1234", "pw", "Demo", "/home/u/.wine/drive_c/Program Files/MT5/terminal64.exe",
                           bridge_host="10.0.0.5", bridge_port=19000)
    res = conn.connect()
    assert res.ok, res.message
    assert mc.backend_name() == "bridge"
    assert (mc.mt5.host, mc.mt5.port) == ("10.0.0.5", 19000)
    assert mc.mt5.init_kw["path"] == r"C:\Program Files\MT5\terminal64.exe"
    assert conn.is_alive()


def test_bridge_unreachable_gives_actionable_message(monkeypatch):
    def boom(host, port):
        raise ConnectionRefusedError("refused")
    monkeypatch.setitem(sys.modules, "mt5linux", types.SimpleNamespace(MetaTrader5=boom))
    monkeypatch.setattr(mc, "mt5", None)
    monkeypatch.setattr(mc, "_NATIVE_MT5", None)
    monkeypatch.setattr(mc, "_bridge_package_available", lambda: True)
    res = mc.MT5Connector("1", "p", "s").connect()
    assert not res.ok and "bridge" in res.message.lower() and "mt5_bridge_server" in res.message


def test_bridge_endpoint_precedence(monkeypatch, tmp_path):
    from app.forward_test import mt5_settings
    monkeypatch.setattr(mt5_settings, "_config_dir", lambda: tmp_path)
    mt5_settings.save_settings(mt5_settings.MT5Settings(login="1", server="s", bridge_host="saved", bridge_port=111))
    monkeypatch.delenv("T58_MT5_BRIDGE_HOST", raising=False)
    monkeypatch.delenv("T58_MT5_BRIDGE_PORT", raising=False)
    assert mc.bridge_endpoint() == ("saved", 111)
    monkeypatch.setenv("T58_MT5_BRIDGE_HOST", "envhost")
    assert mc.bridge_endpoint() == ("envhost", 111)
    assert mc.bridge_endpoint("arg", 5) == ("arg", 5)
