"""v7 P0 live-deployment safety tests: sizing-unit -> broker-quantity
conversion per adapter, futures-prop-safe guardrail defaults, live
max-drawdown halt, disconnect N-strikes halt, and the web deploy
flag/auth gate."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.live_deploy.broker_base import BrokerAdapter, ConnectionResult, OrderResult
from app.live_deploy.broker_ctrader import CTraderBrokerAdapter
from app.live_deploy.broker_dxtrade import DXtradeBrokerAdapter
from app.live_deploy.broker_mt5 import MT5BrokerAdapter
from app.live_deploy.broker_tradelocker import TradeLockerBrokerAdapter
from app.live_deploy.broker_tradovate import TradovateBrokerAdapter
from app.live_deploy.execution_engine import LiveExecutionConfig, LiveExecutionSession
from app.live_deploy.guardrails import (
    DEFAULT_MAX_LOT_SIZE,
    contract_size_for_symbol,
    enforced_guardrails_summary,
    futures_prop_safe_rules,
    normalize_blackout_text,
    resolve_units_per_lot,
)
from app.live_deploy.web_deploy_config import check_web_live_deploy, web_live_deploy_enabled
from app.forward_test.journal import ForwardTestJournal
from app.prop.simulator import PropRules


# ---------------------------------------------------------------- adapters

def _tradovate():
    return TradovateBrokerAdapter(
        username="u", password="p", app_id="a", app_secret="s", cid="c", sec="s2",
    )


def test_tradovate_es_one_contract():
    """The headline bug: $50k / 1% risk / 10-pt ES stop = 50.0 sizing units
    = ONE contract. The old code sent orderQty=int(50)=50 contracts."""
    assert _tradovate().to_broker_qty(50.0, contract_size=50.0, units_per_lot=None) == 1.0


def test_tradovate_mes_micros():
    assert _tradovate().to_broker_qty(10.0, contract_size=5.0, units_per_lot=None) == 2.0


def test_tradovate_sub_one_contract_returns_zero_never_fractional():
    # 25 units on ES = half a contract: must skip (0), never 0.5 or 1.
    assert _tradovate().to_broker_qty(25.0, contract_size=50.0, units_per_lot=None) == 0.0
    assert _tradovate().to_broker_qty(49.9, contract_size=50.0, units_per_lot=None) == 0.0


def test_tradovate_refuses_without_contract_size():
    # No silent defaults that could oversize: missing factor -> 0 (skip).
    assert _tradovate().to_broker_qty(50.0, contract_size=None, units_per_lot=None) == 0.0
    assert _tradovate().to_broker_qty(50.0, contract_size=0.0, units_per_lot=None) == 0.0
    assert _tradovate().to_broker_qty(50.0, contract_size=-5.0, units_per_lot=None) == 0.0


def test_tradovate_floors_never_rounds_up():
    assert _tradovate().to_broker_qty(149.9, contract_size=50.0, units_per_lot=None) == 2.0


def _mt5():
    return MT5BrokerAdapter(login="x", password="y", server="z")


def test_mt5_fx_one_lot():
    """FX standard: 100,000 sizing units = 1.0 MT5 lot."""
    assert _mt5().to_broker_qty(100_000.0, contract_size=None, units_per_lot=100_000.0) == 1.0


def test_mt5_floors_to_centilot_never_rounds_up():
    assert _mt5().to_broker_qty(150_000.0, contract_size=None, units_per_lot=100_000.0) == 1.5
    assert _mt5().to_broker_qty(149_999.0, contract_size=None, units_per_lot=100_000.0) == 1.49


def test_mt5_refuses_without_units_per_lot():
    # The old code passed raw units as lots (a 100,000-lot order). Now: refuse.
    assert _mt5().to_broker_qty(100_000.0, contract_size=None, units_per_lot=None) == 0.0


def test_mt5_futures_via_contract_size():
    # MT5 futures: 1 lot = 1 contract -> units_per_lot = contract_size.
    assert _mt5().to_broker_qty(50.0, contract_size=50.0, units_per_lot=50.0) == 1.0


def _ctrader():
    return CTraderBrokerAdapter(
        client_id="a", client_secret="b", refresh_token="c", ctid_trader_account_id=1,
    )


def test_ctrader_lots_floored_to_hundredth():
    """cTrader volume is integer 1/100-lots; to_broker_qty returns lots
    floored to 0.01 so int(lots*100) is exact."""
    assert _ctrader().to_broker_qty(100_000.0, contract_size=None, units_per_lot=100_000.0) == 1.0
    assert _ctrader().to_broker_qty(150_500.0, contract_size=None, units_per_lot=100_000.0) == 1.5
    assert _ctrader().to_broker_qty(100_000.0, contract_size=None, units_per_lot=None) == 0.0


def test_tradelocker_and_dxtrade_lots():
    tl = TradeLockerBrokerAdapter(email="e", password="p", server="s")
    dx = DXtradeBrokerAdapter(username="u", password="p", base_url="https://x")
    assert tl.to_broker_qty(200_000.0, contract_size=None, units_per_lot=100_000.0) == 2.0
    assert dx.to_broker_qty(200_000.0, contract_size=None, units_per_lot=100_000.0) == 2.0
    assert tl.to_broker_qty(200_000.0, contract_size=None, units_per_lot=None) == 0.0
    assert dx.to_broker_qty(200_000.0, contract_size=None, units_per_lot=None) == 0.0


# ------------------------------------------------------- tradovate close fix

class _FakeResp:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


def test_tradovate_close_position_verifies_response():
    adapter = _tradovate()

    adapter._session.post = lambda *a, **k: _FakeResp({"failureReason": "Unknown", "failureText": "No such position"})
    r = adapter.close_position("123")
    assert r.ok is False and "No such position" in r.message

    adapter._session.post = lambda *a, **k: _FakeResp({}, status_code=500)
    r = adapter.close_position("123")
    assert r.ok is False and "500" in r.message

    adapter._session.post = lambda *a, **k: _FakeResp({"id": 123})
    r = adapter.close_position("123")
    assert r.ok is True


# ------------------------------------------------------- guardrail defaults

def test_futures_prop_safe_rules_defaults():
    rules = futures_prop_safe_rules(account_size=50_000.0)
    assert rules.news_blackout_windows.strip() != ""  # blackout ON with a default window set
    assert rules.weekend_hold_allowed is False
    assert rules.hedging_allowed is False
    assert rules.max_lot_size == DEFAULT_MAX_LOT_SIZE == 5.0


def test_guardrail_overrides_are_explicit_opt_in():
    rules = futures_prop_safe_rules(
        50_000.0, weekend_hold_allowed=True, hedging_allowed=True, max_lot_size=None,
    )
    assert rules.weekend_hold_allowed is True
    assert rules.hedging_allowed is True
    assert rules.max_lot_size is None


def test_blackout_text_normalization():
    assert normalize_blackout_text("07:25-07:40, 12:55-13:10") == "07:25-07:40\n12:55-13:10"
    assert normalize_blackout_text("07:25-07:40;12:55-13:10") == "07:25-07:40\n12:55-13:10"


def test_enforced_summary_names_the_values():
    rules = futures_prop_safe_rules(50_000.0)
    text = enforced_guardrails_summary(rules)
    assert "BLOCKED" in text and "5" in text and "max-drawdown halt" in text


def test_contract_size_lookup():
    assert contract_size_for_symbol("ES") == 50.0
    assert contract_size_for_symbol("es") == 50.0
    assert contract_size_for_symbol("ESZ25") == 50.0  # month code stripped
    assert contract_size_for_symbol("MES") == 5.0
    assert contract_size_for_symbol("EURUSD") is None


def test_resolve_units_per_lot():
    assert resolve_units_per_lot(50.0) == 50.0            # futures: 1 lot/contract = contract_size
    assert resolve_units_per_lot(None) == 100_000.0      # FX standard
    assert resolve_units_per_lot(None, 100_000.0) == 100_000.0
    assert resolve_units_per_lot(50.0, 25.0) == 25.0     # explicit wins


# ------------------------------------------------------- engine safety tests

class _FlatStrategy:
    def generate(self, df):
        n = len(df)
        return SimpleNamespace(
            signals=pd.Series([0] * n, index=df.index),
            stop_loss_distance=None, stop_loss_pips=None,
            take_profit_distance=None, take_profit_pips=None,
        )


def _bars(n=10):
    ts = pd.date_range("2026-10-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": 5000.0, "high": 5001.0,
        "low": 4999.0, "close": 5000.0, "volume": 100.0,
    })


class _FakeBroker(BrokerAdapter):
    platform_name = "Fake"

    def __init__(self, equity=50_000.0, alive=True):
        self.equity = equity  # test mutates this to simulate market moves
        self._alive = alive
        self.flatten_calls = 0
        self.last_order_qty = None

    def connect(self):
        return ConnectionResult(ok=True, message="ok", account_login="1",
                                account_server="fake", balance=50_000.0, equity=self.equity)

    def disconnect(self):
        pass

    def is_alive(self):
        return self._alive

    def ensure_connected(self):
        if self._alive:
            return ConnectionResult(ok=True, message="ok")
        return ConnectionResult(ok=False, message="link down")

    def account_summary(self):
        return {"balance": 50_000.0, "equity": self.equity}

    def fetch_completed_bars(self, symbol, timeframe_minutes, count):
        return _bars()

    def get_open_positions(self, symbol=None):
        return []

    def place_market_order(self, symbol, direction, volume, sl_price=None, tp_price=None,
                           comment="T58 Live", deviation=20):
        self.last_order_qty = volume
        return OrderResult(ok=True, message="ok", ticket="1", price=5000.0, volume=volume)

    def close_position(self, ticket, comment="T58 Live close"):
        return OrderResult(ok=True, message="closed", ticket=ticket)

    def close_all(self, symbol=None):
        self.flatten_calls += 1
        return []

    def to_broker_qty(self, units, contract_size, units_per_lot):
        return float(units)  # 1:1 for the engine-level tests


def _session(tmp_path, broker, **rule_overrides):
    journal = ForwardTestJournal(db_path=tmp_path / "j.db")
    rules = PropRules(account_size=50_000.0, max_drawdown_pct=10.0, **rule_overrides)
    cfg = LiveExecutionConfig(
        symbol="ES", timeframe_minutes=15,
        risk=RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=1.0, pip_size=1.0),
        prop_rules=rules, contract_size=50.0, units_per_lot=50.0,
    )
    return LiveExecutionSession(
        strategy=_FlatStrategy(), strategy_type="manual", strategy_filename="flat",
        broker=broker, journal=journal, config=cfg,
    )


def _begun_session(tmp_path, broker, **rule_overrides):
    """A session past start() but WITHOUT the background poll thread, so
    _poll_once() can be driven deterministically in tests."""
    sess = _session(tmp_path, broker, **rule_overrides)
    conn = broker.connect()
    assert conn.ok
    sess.status.connected = True
    sess.status.balance = conn.balance
    sess.status.equity = conn.equity
    sess._session_id = sess.journal.start_session("manual", "flat", "ES", 15, "1", "fake")
    sess._dd_peak = conn.equity or conn.balance or 50_000.0
    sess._stop_flag.clear()
    sess.status.running = True
    return sess


def test_max_drawdown_breach_flattens_and_halts_trailing(tmp_path):
    # 10% of 50k = 5k -> trailing floor 45k; equity 44k breaches.
    broker = _FakeBroker(equity=50_000.0)
    sess = _begun_session(tmp_path, broker)
    sess._poll_once()   # equity 50k: peak anchored, no breach
    assert sess.status.alert is None
    broker.equity = 44_000.0
    sess._poll_once()   # breach
    assert broker.flatten_calls == 1
    assert sess.status.alert is not None and "MAX DRAWDOWN" in sess.status.alert
    assert sess._stop_flag.is_set()


def test_max_drawdown_breach_flattens_and_halts_static(tmp_path):
    broker = _FakeBroker(equity=60_000.0)
    sess = _begun_session(tmp_path, broker, drawdown_type="static")
    sess._poll_once()   # 60k: peak moves up, static floor stays 45k
    assert sess.status.alert is None
    broker.equity = 44_000.0
    sess._poll_once()   # 44k <= 45k: breach
    assert broker.flatten_calls == 1
    assert sess.status.alert is not None and "static" in sess.status.alert


def test_no_breach_no_halt(tmp_path):
    broker = _FakeBroker(equity=50_000.0)
    sess = _begun_session(tmp_path, broker)
    for eq in (48_000.0, 47_000.0, 46_000.0):
        broker.equity = eq
        sess._poll_once()
    assert broker.flatten_calls == 0
    assert sess.status.alert is None
    assert not sess._stop_flag.is_set()


def test_disconnect_three_strikes_flattens_and_halts(tmp_path):
    broker = _FakeBroker(equity=50_000.0, alive=False)
    sess = _begun_session(tmp_path, broker)
    sess._poll_once()
    assert sess.status.alert is None and broker.flatten_calls == 0
    sess._poll_once()
    assert sess.status.alert is None and broker.flatten_calls == 0
    sess._poll_once()  # third consecutive failure
    assert broker.flatten_calls == 1
    assert sess.status.alert is not None and "CONNECTION LOST" in sess.status.alert
    assert sess._stop_flag.is_set()


def test_disconnect_recovery_resets_counter(tmp_path):
    broker = _FakeBroker(equity=50_000.0, alive=False)
    sess = _begun_session(tmp_path, broker)
    sess._poll_once()
    sess._poll_once()
    assert sess._consecutive_poll_failures == 2
    broker._alive = True
    sess._poll_once()  # success resets
    assert sess._consecutive_poll_failures == 0
    assert sess.status.alert is None
    assert not sess._stop_flag.is_set()


def test_config_requires_sizing_factors():
    with pytest.raises(TypeError):
        LiveExecutionConfig(
            symbol="ES", timeframe_minutes=15,
            risk=RiskConfig(), prop_rules=PropRules(),
        )


class _LongStrategy:
    """Always-long, 10-point stop -- drives _open_new_trade end to end."""

    def generate(self, df):
        n = len(df)
        return SimpleNamespace(
            signals=pd.Series([1] * n, index=df.index),
            stop_loss_distance=pd.Series([10.0] * n, index=df.index),
            stop_loss_pips=None, take_profit_distance=None, take_profit_pips=None,
        )


class _TradovateSizedBroker(_FakeBroker):
    """Fake broker with Tradovate's real conversion: whole contracts."""

    def to_broker_qty(self, units, contract_size, units_per_lot):
        import math
        if not contract_size or contract_size <= 0:
            return 0.0
        return float(math.floor(units / contract_size + 1e-9))


def test_engine_converts_units_to_broker_qty_on_entry(tmp_path):
    """$50k / 1% / 10-pt ES stop -> 50.0 sizing units -> the broker must
    receive orderQty 1 (one contract), not 50."""
    broker = _TradovateSizedBroker(equity=50_000.0)
    journal = ForwardTestJournal(db_path=tmp_path / "j.db")
    cfg = LiveExecutionConfig(
        symbol="ES", timeframe_minutes=15,
        risk=RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=1.0,
                        pip_size=1.0, contract_size=50.0),
        prop_rules=PropRules(account_size=50_000.0, max_drawdown_pct=10.0),
        contract_size=50.0, units_per_lot=50.0,
    )
    sess = LiveExecutionSession(
        strategy=_LongStrategy(), strategy_type="manual", strategy_filename="long",
        broker=broker, journal=journal, config=cfg,
    )
    sess._session_id = journal.start_session("manual", "long", "ES", 15, "1", "fake")
    sess._dd_peak = 50_000.0
    sess._stop_flag.clear()
    sess._open_new_trade(1, _LongStrategy().generate(_bars()), 5000.0)
    assert broker.last_order_qty == 1.0


def test_engine_skips_sub_one_contract_entry(tmp_path):
    """Risk too small for one whole contract -> to_broker_qty -> 0 ->
    the entry is skipped, never a fractional contract."""
    broker = _TradovateSizedBroker(equity=50_000.0)
    journal = ForwardTestJournal(db_path=tmp_path / "j.db")
    cfg = LiveExecutionConfig(
        symbol="ES", timeframe_minutes=15,
        # 0.1% of 50k = $50 risk on a 10-pt stop = 5 units < 1 contract
        risk=RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=0.1,
                        pip_size=1.0, contract_size=50.0),
        prop_rules=PropRules(account_size=50_000.0, max_drawdown_pct=10.0),
        contract_size=50.0, units_per_lot=50.0,
    )
    sess = LiveExecutionSession(
        strategy=_LongStrategy(), strategy_type="manual", strategy_filename="long",
        broker=broker, journal=journal, config=cfg,
    )
    sess._session_id = journal.start_session("manual", "long", "ES", 15, "1", "fake")
    sess._dd_peak = 50_000.0
    sess._stop_flag.clear()
    sess._open_new_trade(1, _LongStrategy().generate(_bars()), 5000.0)
    assert broker.last_order_qty is None  # no order placed
    assert sess.status.open_position_ticket is None


# ------------------------------------------------------- web deploy flag/auth

def test_web_deploy_disabled_by_default(monkeypatch):
    monkeypatch.delenv("T58_ENABLE_WEB_LIVE_DEPLOY", raising=False)
    assert web_live_deploy_enabled() is False
    allowed, code, msg = check_web_live_deploy(True, True)
    assert allowed is False and code == 403
    assert "T58_ENABLE_WEB_LIVE_DEPLOY" in msg


def test_web_deploy_enabled_still_requires_auth(monkeypatch):
    monkeypatch.setenv("T58_ENABLE_WEB_LIVE_DEPLOY", "1")
    assert web_live_deploy_enabled() is True
    # flag on, but no password set -> refused
    allowed, code, _ = check_web_live_deploy(False, False)
    assert allowed is False and code == 401
    # flag on, password set, session not unlocked -> refused
    allowed, code, _ = check_web_live_deploy(True, False)
    assert allowed is False and code == 401
    # flag on + password + unlocked -> allowed
    allowed, code, _ = check_web_live_deploy(True, True)
    assert allowed is True and code == 200
