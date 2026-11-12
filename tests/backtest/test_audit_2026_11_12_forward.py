from types import SimpleNamespace

import pandas as pd

from app.backtest.risk import RiskConfig
from app.forward_test.engine import ForwardTestConfig, ForwardTestSession


class _Conn:
    def __init__(self, profit=0.0):
        self.profit = profit

    def get_open_positions(self, symbol):
        return [SimpleNamespace(ticket=7, profit=self.profit)]


def _session(profit, **kw):
    cfg = ForwardTestConfig(symbol="ES", timeframe_minutes=5, risk=RiskConfig(initial_balance=10_000.0, daily_loss_limit_pct=2.0), **kw)
    strat = SimpleNamespace(config={"a": 1})
    s = ForwardTestSession(strat, "manual", "x.json", _Conn(profit), SimpleNamespace(), cfg)
    s.status.open_position_ticket = 7
    return s


def test_daily_breaker_counts_unrealized_loss():
    assert _session(-250.0)._daily_loss_breached() is True      # -250 floating vs a $200 limit, nothing realized yet
    assert _session(-50.0)._daily_loss_breached() is False


def test_frozen_guard_blocks_entries_when_config_changes():
    s = _session(0.0)
    s._sequential_checks()
    assert s._entries_blocked_reason is None
    s.cfg.strategy_config["a"] = 2            # "I just tweaked a threshold"
    s._sequential_checks()
    assert "FROZEN-CODE GUARD" in s._entries_blocked_reason


def test_sprt_futility_or_drawdown_blocks_entries():
    s = _session(0.0, expected_max_drawdown_r=3.0)
    for _ in range(5):
        s._sprt.add(-1.0, day="2026-11-12")
    s._sequential_checks()
    assert s._entries_blocked_reason and "FORWARD-TEST HALT" in s._entries_blocked_reason
