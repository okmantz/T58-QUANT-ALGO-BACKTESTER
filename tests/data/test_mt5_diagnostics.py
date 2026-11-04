"""Plain-language problems, pre-flight, and the order-side fixes (lots, fill modes, Algo Trading) with a fake MT5."""
from collections import namedtuple
from types import SimpleNamespace

import pytest

from app.forward_test import mt5_bridge_manager as mgr
from app.forward_test import mt5_connector as mc
from app.forward_test import mt5_diagnostics as diag


# ---------------------------------------------------------------- problems read like instructions, not codes

@pytest.mark.parametrize("code,kind", [(-10003, "terminal_not_found"), (-10004, "terminal_slow"), (-10005, "terminal_slow"),
                                      (-2, "login_failed"), (-6, "login_failed"), (-1, "terminal_error"), (-777, "connect_failed")])
def test_mt5_error_codes_map_to_problems_with_steps(code, kind):
    p = diag.problem_from_mt5_error(code, "whatever")
    assert p.kind == kind and p.title and p.summary and p.steps
    assert str(code) in p.technical                      # the raw code is kept, but not the headline
    assert str(code) not in p.title


def test_login_problem_warns_about_investor_password():
    assert any("investor" in s for s in diag.problem_from_mt5_error(-6, "Authorization failed").steps)


def test_plain_text_contains_title_steps_commands_and_technical():
    p = diag.wine_missing()
    p.technical = "raw detail"
    text = p.plain_text()
    assert p.title in text and "1. " in text and "raw detail" in text


def test_order_help_covers_the_retcodes_people_hit():
    assert "Algo Trading" in diag.order_help(10027)
    assert "minimum lot" in diag.order_help(10014)
    assert "closed" in diag.order_help(10018)
    assert diag.order_help(99999) == ""


@pytest.mark.parametrize("log,kind", [
    ("python.exe: can't open file 'x': [Errno 2] No such file or directory", "bridge_file_unreachable"),
    ("zsh: Bad CPU type in executable: wine64", "needs_rosetta"),
    ("ModuleNotFoundError: No module named 'MetaTrader5'", "bridge_packages_missing"),
    ("OSError: [WinError 10048] Only one usage of each socket address", "bridge_port_busy"),
    ("something nobody has seen before", "bridge_exit"),
])
def test_bridge_failure_classification(log, kind):
    p = diag.bridge_failed(2, log, "/p", "/tmp/mt5_bridge.log")
    assert p.kind == kind and "exit code 2" in p.technical


# ---------------------------------------------------------------- pre-flight

def test_preflight_linux_without_wine_says_exactly_what_to_do(monkeypatch):
    monkeypatch.setattr(mc, "platform_kind", lambda: "linux")
    monkeypatch.setattr(mc, "_bridge_package_available", lambda: True)
    monkeypatch.setattr(mc, "find_terminal_candidates", lambda: [])
    monkeypatch.setattr(mgr, "find_wine", lambda: None)
    monkeypatch.setattr(mgr, "port_open", lambda *a, **k: False)
    out = diag.preflight("")
    by = {c["id"]: c for c in out["checks"]}
    assert not out["ready"] and by["wine"]["status"] == "action" and by["wine"]["commands"]
    assert by["terminal"]["status"] == "action"
    assert "need" in out["headline"]


def test_preflight_ready_when_only_first_time_setup_remains(monkeypatch, tmp_path):
    exe = tmp_path / "pfx" / "drive_c" / "Program Files" / "MT5" / "terminal64.exe"
    exe.parent.mkdir(parents=True); exe.write_bytes(b"MZ")
    monkeypatch.setattr(mc, "platform_kind", lambda: "macos")
    monkeypatch.setattr(mc, "_bridge_package_available", lambda: True)
    monkeypatch.setattr(mc, "find_terminal_candidates", lambda: [str(exe)])
    monkeypatch.setattr(mgr, "find_wine", lambda: "/opt/homebrew/bin/wine64")
    monkeypatch.setattr(mgr, "port_open", lambda *a, **k: False)
    out = diag.preflight("")
    assert out["ready"] and out["first_time_setup"]          # no Wine Python yet: T58 does it, user does nothing
    assert {c["status"] for c in out["checks"]} <= {"ok", "auto"}


def test_preflight_windows_native_package(monkeypatch):
    monkeypatch.setattr(mc, "platform_kind", lambda: "windows")
    monkeypatch.setattr(mc, "_NATIVE_MT5", None)
    monkeypatch.setattr(mc, "find_terminal_candidates", lambda: [r"C:\Program Files\MT5\terminal64.exe"])
    out = diag.preflight("")
    by = {c["id"]: c for c in out["checks"]}
    assert by["package"]["status"] == "action" and "pip install MetaTrader5" in by["package"]["commands"]


# ---------------------------------------------------------------- connector with a fake terminal

SymInfo = namedtuple("SymInfo", "trade_contract_size volume_step volume_min volume_max point")
Res = namedtuple("Res", "retcode order deal comment price volume")
Tick = namedtuple("Tick", "bid ask time")


class FakeMT5:
    TRADE_RETCODE_DONE = 10009
    ORDER_TYPE_BUY, ORDER_TYPE_SELL, TRADE_ACTION_DEAL, ORDER_TIME_GTC = 0, 1, 1, 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2

    def __init__(self, info=SymInfo(100.0, 0.01, 0.01, 50.0, 0.01), accepted_fills=(1, 0, 2), trade_allowed=True, login=1234):
        self.info, self.accepted, self.trade_allowed, self._login = info, accepted_fills, trade_allowed, login
        self.sent = []

    def initialize(self, **kw): return True
    def shutdown(self): return True
    def last_error(self): return (1, "ok")
    def account_info(self): return SimpleNamespace(login=self._login, server="Demo", balance=1.0, equity=1.0, currency="USD")
    def terminal_info(self): return SimpleNamespace(trade_allowed=self.trade_allowed)
    def symbol_select(self, s, e=True): return s != "NOPE"
    def symbol_info(self, s): return None if s == "NOPE" else self.info
    def symbol_info_tick(self, s): return Tick(2000.0, 2000.5, 0)
    def history_deals_get(self, **k): return None

    def order_send(self, req):
        self.sent.append(dict(req))
        if req["type_filling"] in self.accepted:
            return Res(10009, 5, 6, "done", 2000.5, req["volume"])
        return Res(10030, 0, 0, "Unsupported filling mode", 0.0, 0.0)


@pytest.fixture()
def fake(monkeypatch):
    def make(**kw):
        f = FakeMT5(**kw)
        monkeypatch.setattr(mc, "mt5", f)
        monkeypatch.setattr(mc, "_NATIVE_MT5", f)
        monkeypatch.setattr(mc, "platform_kind", lambda: "windows")
        monkeypatch.setattr(mc.MT5Connector, "launch_terminal", lambda self, wait_seconds=45.0: None)
        return f, mc.MT5Connector("1234", "pw", "Demo")
    return make


def test_units_are_converted_to_lots_with_broker_contract_size(fake):
    f, c = fake()                                         # 100 oz per lot, 0.01 step
    assert c.units_to_lots("XAUUSD", 250.0) == (2.5, None)        # 250 units / 100 = 2.5 lots, NOT 250 lots
    f.info = SymInfo(100000.0, 0.01, 0.01, 50.0, 0.00001)
    lots, _ = c.units_to_lots("EURUSD", 123456.0)
    assert lots == 1.23                                    # rounded DOWN to the step: never risk more than budgeted


def test_below_minimum_lot_skips_instead_of_oversizing(fake):
    f, c = fake()
    lots, note = c.units_to_lots("XAUUSD", 0.4)           # 0.004 lots < 0.01 minimum
    assert lots == 0 and "minimum" in note and "Skipped" in note


def test_lots_are_clamped_to_broker_maximum(fake):
    f, c = fake(info=SymInfo(1.0, 1.0, 1.0, 20.0, 0.01))
    assert c.units_to_lots("X", 500.0) == (20.0, None)


def test_unsupported_filling_mode_is_retried_with_the_next_mode(fake):
    f, c = fake(accepted_fills=(2,))                      # broker only takes RETURN
    r = c.place_market_order("XAUUSD", 1, 0.1, 1990.0, 2020.0)
    assert r.ok, r.message
    assert [s["type_filling"] for s in f.sent] == [1, 0, 2]


def test_rejection_message_explains_itself(fake):
    f, c = fake(accepted_fills=())
    r = c.place_market_order("XAUUSD", 1, 0.1)
    assert not r.ok and "10030" in r.message and "filling" in r.message.lower()


def test_algo_trading_off_is_caught_at_connect_with_fix_steps(fake):
    f, c = fake(trade_allowed=False)
    res = c.connect()
    assert not res.ok and res.problem["kind"] == "algo_trading_off"
    assert any("Algo Trading" in s for s in res.problem["steps"])


def test_connect_ok_when_algo_trading_on_and_reports_progress(fake):
    f, c = fake()
    seen = []
    c.on_progress = lambda step, text: seen.append(step)
    res = c.connect()
    assert res.ok and res.problem is None
    assert seen[0] == "prepare" and "trading" in seen


def test_unknown_symbol_gives_a_symbol_problem(fake):
    f, c = fake()
    c.connect()
    assert c.check_symbol("XAUUSD") is None
    p = c.check_symbol("NOPE")
    assert p.kind == "symbol_not_found" and "NOPE" in p.title


def test_non_numeric_login_is_explained(fake):
    f, _ = fake()
    res = mc.MT5Connector("abc", "pw", "Demo").connect()
    assert not res.ok and res.problem["kind"] == "login_not_numeric"
