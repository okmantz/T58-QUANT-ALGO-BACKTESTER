import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.backtest.invariants import InvariantViolation, check_trade, enforce


def _t(**kw):
    base = dict(entry_time=pd.Timestamp("2024-01-02"), exit_time=pd.Timestamp("2024-01-03"), direction=1, entry_price=100.0,
                exit_price=99.0, size=1.0, pnl=-1.0, pnl_pct=-0.01, exit_reason="stop_loss", commission=0.5, equity_after=1.0,
                exit_cause="stop_loss")
    base.update(kw)
    return Trade(**base)


def test_clean_stop_out_passes():
    assert check_trade(_t()) == []


def test_profitable_stop_out_flagged():
    assert any("never be profitable" in v for v in check_trade(_t(pnl=5.0)))


def test_trailing_stop_may_profit():
    assert check_trade(_t(exit_cause="trailing_stop", exit_price=101.0, pnl=1.0)) == []


def test_short_stop_on_winning_side_flagged():
    assert check_trade(_t(direction=-1, exit_price=99.0, pnl=-1.0))


def test_time_travel_nan_and_policy(monkeypatch):
    assert check_trade(_t(exit_time=pd.Timestamp("2023-01-01")))
    assert check_trade(_t(pnl=float("nan")))
    monkeypatch.setenv("T58_INVARIANTS", "raise")
    with pytest.raises(InvariantViolation):
        enforce([_t(pnl=5.0)])
    monkeypatch.setenv("T58_INVARIANTS", "off")
    assert enforce([_t(pnl=5.0)]) == []
    monkeypatch.setenv("T58_INVARIANTS", "warn")
    with pytest.warns(RuntimeWarning):
        enforce([_t(pnl=5.0)])
