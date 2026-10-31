"""Equity ledger identity: final == start + sum(trade P&L) + logged adjustments."""
import pytest

from app.backtest.engine import run_backtest
from app.backtest.equity_reconciliation import EquityReconciliationError, enforce, recon_policy, reconcile_equity
from app.backtest.risk import RiskConfig
from tests.backtest.test_backtest import _sma_strategy, _trending_df


def test_balanced_ledger_is_ok():
    r = reconcile_equity(50000, [100, -40, 25], [("payout", -500)], 50000 + 85 - 500)
    assert r.ok and r.difference == pytest.approx(0) and r.payouts == pytest.approx(500)


def test_unexplained_equity_change_fails():
    r = reconcile_equity(50000, [100, -40], [], 50000 + 60 - 3000)
    assert not r.ok and r.difference == pytest.approx(-3000)
    assert "3,000" in r.message()


def test_strict_raises_warn_warns_off_is_silent(monkeypatch):
    bad = reconcile_equity(1000, [10], [], 5)
    monkeypatch.setenv("T58_EQUITY_RECON", "strict")
    with pytest.raises(EquityReconciliationError):
        enforce(bad)
    monkeypatch.setenv("T58_EQUITY_RECON", "warn")
    with pytest.warns(RuntimeWarning):
        enforce(bad)
    monkeypatch.setenv("T58_EQUITY_RECON", "off")
    enforce(bad)
    monkeypatch.setenv("T58_EQUITY_RECON", "garbage")
    assert recon_policy() == "strict"


def test_engine_run_balances_in_strict_mode(monkeypatch):
    monkeypatch.setenv("T58_EQUITY_RECON", "strict")
    res = run_backtest(_trending_df(600, seed=3), _sma_strategy(), RiskConfig(initial_balance=10000, pip_size=0.0001))
    assert res.statistics.total_trades >= 1       # strict mode would have raised if the ledger did not balance
