"""The Guardian inside a live execution session, driven against the existing fake-broker harness."""
from types import SimpleNamespace

import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.forward_test.journal import ForwardTestJournal
from app.guardian.engine import GuardianRules
from app.live_deploy import guardian_setup
from app.live_deploy.execution_engine import LiveExecutionConfig, LiveExecutionSession
from app.prop.simulator import PropRules
from tests.backtest.test_live_sizing_v7 import _bars, _FakeBroker


class _LongStrategy:
    def generate(self, df):
        return SimpleNamespace(signals=pd.Series([1] * len(df), index=df.index), stop_loss_distance=pd.Series([10.0] * len(df), index=df.index), stop_loss_pips=None,
                               take_profit_distance=None, take_profit_pips=None)


class _Broker(_FakeBroker):
    def __init__(self, equity=50_000.0, balance=50_000.0):
        super().__init__(equity=equity)
        self.balance = balance
        self.orders = 0

    def account_summary(self):
        return {"balance": self.balance, "equity": self.equity}

    def fetch_completed_bars(self, symbol, tf, count):
        self.orders += 0
        df = _bars()
        df["timestamp"] = df["timestamp"] + pd.Timedelta(minutes=15 * getattr(self, "_shift", 0))
        return df

    def place_market_order(self, *a, **k):
        self.orders += 1
        return super().place_market_order(*a, **k)


@pytest.fixture(autouse=True)
def _tmp_latest(tmp_path, monkeypatch):
    from app.guardian import engine
    monkeypatch.setattr(engine, "_latest_path", lambda: tmp_path / "latest.json")


def _begun(tmp_path, broker, guardian, **cfg_kw):
    cfg = LiveExecutionConfig(symbol="ES", timeframe_minutes=15, risk=RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=1.0, pip_size=1.0),
                              prop_rules=PropRules(account_size=50_000.0, max_drawdown_pct=10.0), contract_size=50.0, units_per_lot=50.0,
                              guardian=guardian, **cfg_kw)
    sess = LiveExecutionSession(strategy=_LongStrategy(), strategy_type="manual", strategy_filename="long", broker=broker,
                                journal=ForwardTestJournal(db_path=tmp_path / "j.db"), config=cfg)
    conn = broker.connect()
    sess.status.connected, sess.status.balance, sess.status.equity = True, conn.balance, conn.equity
    sess._session_id = sess.journal.start_session("manual", "long", "ES", 15, "1", "fake")
    sess._dd_peak = conn.equity
    sess._stop_flag.clear()
    sess.status.running = True
    return sess


RULES = dict(account_size=50_000, daily_loss_limit=1_200, max_drawdown=4_000, drawdown_type="static")


def test_guardian_off_is_unchanged(tmp_path):
    b = _Broker()
    s = _begun(tmp_path, b, None)
    s._poll_once()
    assert b.orders == 1 and s.status.guardian is None


def test_guardian_reports_and_allows_entry_when_clear(tmp_path):
    b = _Broker()
    s = _begun(tmp_path, b, GuardianRules(**RULES))
    s._poll_once()
    assert s.status.guardian["level"] == "ok" and b.orders == 1


def test_guardian_blocks_entries_near_the_daily_limit(tmp_path):
    b = _Broker()
    s = _begun(tmp_path, b, GuardianRules(**RULES))
    s._poll_once()                                   # first poll fixes the day's opening balance at 50,000
    s._guardian._sod = 50_000.0
    b.balance = b.equity = 48_850.0                  # 96% of the 1,200 limit used
    b._shift = 1
    s.status.open_position_ticket = None
    before = b.orders
    s._poll_once()
    assert s.status.guardian["action"]["can_open"] is False and b.orders == before
    assert "Guardian" in (s.status.alert or "")


def test_guardian_flattens_and_halts_for_the_day(tmp_path):
    b = _Broker()
    s = _begun(tmp_path, b, GuardianRules(**RULES))
    s._poll_once()
    s._guardian._sod = 50_000.0
    b.balance = b.equity = 48_700.0                  # past the daily limit
    s._poll_once()
    assert s.status.guardian["action"]["flatten"] and s.status.halted_reason and "halted for the day" in s.status.halted_reason


def test_reduced_size_when_braking(tmp_path):
    full = _Broker()
    s1 = _begun(tmp_path, full, GuardianRules(**RULES))
    s1._poll_once()
    reduced = _Broker()
    s2 = _begun(tmp_path, reduced, GuardianRules(**RULES))
    s2._poll_once()                                  # seeds the opening balance
    s2._guardian._sod = 50_000.0
    reduced.balance = reduced.equity = 49_000.0      # ~83% used: reduce
    reduced._shift = 1
    s2.status.open_position_ticket = None
    s2._poll_once()
    assert reduced.orders == 2 and reduced.last_order_qty < full.last_order_qty


def test_optimizer_clamps_risk_to_hard_limits(tmp_path):
    b = _Broker()
    s = _begun(tmp_path, b, GuardianRules(**RULES), optimizer={"stage": "scaling", "preset": "accelerated", "sizes_risk": True})
    scale = s._optimizer_scale(50_000.0)
    from app.prop import payout_optimizer as po
    assert scale > 0 and scale * s.cfg.risk.risk_amount(50_000.0) <= po.HARD_MAX_SHARE_OF_CUSHION * 5_000.0 + 1e-6


def test_form_setup_builds_guardian_and_optimizer():
    cfg = SimpleNamespace(guardian=None, guardian_preset="balanced", optimizer=None)
    lines = guardian_setup.apply({"payout_preset": "conservative", "payout_stage": "newly_funded", "optimizer_sizes_risk": "1"}, cfg,
                                 account_size=50_000, daily_loss_pct=2.4, max_dd_pct=4.0, profit_target_pct=6.0)
    assert cfg.guardian.daily_loss_limit == 1_200 and cfg.guardian.max_drawdown == 2_000 and cfg.guardian.profit_target is None
    assert cfg.guardian_preset == "conservative" and cfg.optimizer == {"stage": "newly_funded", "preset": "conservative", "sizes_risk": True}
    assert any("never overridden" in ln for ln in lines)
    cfg2 = SimpleNamespace(guardian=1, guardian_preset="x", optimizer=None)
    assert any("OFF" in ln for ln in guardian_setup.apply({"guardian_on": "0"}, cfg2, account_size=50_000, daily_loss_pct=2, max_dd_pct=4, profit_target_pct=6))
    assert cfg2.guardian is None


def test_form_setup_uses_firm_rules_when_asked():
    cfg = SimpleNamespace(guardian=None, guardian_preset="balanced", optimizer=None)
    guardian_setup.apply({"firm_preset": "lucid_50k", "payout_stage": "evaluation"}, cfg, account_size=1, daily_loss_pct=1, max_dd_pct=1, profit_target_pct=1)
    assert cfg.guardian.max_contracts == 4 and cfg.guardian.firm == "Lucid Trading"


def test_optimizer_routes():
    from app.web.server import app
    c = app.test_client()
    j = c.get("/api/payout-optimizer/stages?preset=conservative&account_size=50000&daily_loss=4&max_dd=4").json
    assert j["ok"] and [s["stage"] for s in j["stages"]][0] == "evaluation" and len(j["stages"]) == 5
    assert c.get("/api/payout-optimizer/stages?account_size=abc").status_code == 400
    assert len(c.get("/api/prop-presets/keys").json["presets"]) >= 20
