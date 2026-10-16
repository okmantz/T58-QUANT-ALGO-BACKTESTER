from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.forward_test.journal import ForwardTestJournal
from app.monte_carlo.engine import MonteCarloResult
from app.orchestration import auto_retune
from app.orchestration.auto_retune import AutoRetuneOutcome, maybe_trigger_retune
from app.orchestration.quick_optimize import QuickOptimizeResult
from app.prop.simulator import PropRules
from app.backtest.risk import RiskConfig


def _mc_result(**overrides) -> MonteCarloResult:
    defaults = dict(
        n_simulations=1000, evaluation_pass_probability=0.6, first_payout_probability=0.5,
        failure_before_payout_probability=0.4, multiple_payout_probability=0.2,
        median_days_to_pass=20.0, median_days_to_first_payout=40.0, average_days_to_first_payout=42.0,
        median_return_pct=8.0, mean_return_pct=7.5, expected_payout=1000.0, median_payout=900.0,
        total_simulated_withdrawals=500_000.0, median_drawdown_pct=3.0, p95_drawdown_pct=6.0,
        worst_drawdown_pct=9.0, risk_of_ruin_pct=2.0, median_max_losing_streak=4.0, worst_max_losing_streak=9,
        return_percentiles={"5": 1.0, "25": 5.0, "50": 8.0, "75": 11.0, "95": 15.0},
        drawdown_percentiles={"5": 1.0, "25": 2.0, "50": 3.0, "75": 4.5, "95": 6.0},
        days_to_payout_distribution=[],
        return_distribution=[1.0, 4.0, 6.0, 8.0, 8.0, 9.0, 11.0, 13.0, 15.0, 15.5] * 100,
        drawdown_distribution=[1.0, 2.0, 3.0, 4.0, 6.0] * 200,
    )
    defaults.update(overrides)
    return MonteCarloResult(**defaults)


def _journal_with_trades(pnls: list[float]) -> tuple[ForwardTestJournal, int]:
    journal = ForwardTestJournal(db_path=Path(":memory:"))
    session_id = journal.start_session("python", "strat.py", "XAUUSD", 15, "12345", "Demo-Server")
    for i, pnl in enumerate(pnls):
        trade_id = journal.record_open(session_id, mt5_ticket=i, direction=1, volume=0.1,
                                        entry_price=2000.0, sl_price=1990.0, tp_price=2020.0)
        journal.record_close(trade_id, exit_price=2000.0 + pnl, pnl=pnl)
    return journal, session_id


def _fake_quick_optimize_result(improved=True) -> QuickOptimizeResult:
    return QuickOptimizeResult(
        strategy_display_name="test_strat", source_type="python",
        baseline_trades=10, baseline_net_profit=100.0, baseline_win_rate=50.0,
        baseline_eval_pass_probability=40.0, baseline_payout_probability=30.0,
        optimized_trades=10, optimized_net_profit=200.0, optimized_win_rate=60.0,
        optimized_eval_pass_probability=55.0, optimized_payout_probability=45.0,
        ga_result=None, improved=improved, final_parameters=None, final_code_text=None,
        final_code_extension=None, saved_library_path=None, saved_library_note=None,
        elapsed_seconds=1.0,
    )


def test_no_trigger_when_healthy(monkeypatch):
    journal, session_id = _journal_with_trades([80.0] * 30)
    called = {"n": 0}
    monkeypatch.setattr(auto_retune, "run_quick_optimize", lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    outcome = maybe_trigger_retune(
        journal, session_id, "test_strat", _mc_result(), account_balance=10_000,
        df=pd.DataFrame(), strategy=object(), risk=RiskConfig(), prop_rules=PropRules(),
    )
    assert isinstance(outcome, AutoRetuneOutcome)
    assert not outcome.triggered
    assert outcome.retune_result is None
    assert called["n"] == 0


def test_trigger_on_critical_drawdown_and_calls_quick_optimize(monkeypatch):
    journal, session_id = _journal_with_trades([50.0, 50.0, 50.0, -2000.0])
    monkeypatch.setattr(auto_retune, "run_quick_optimize", lambda *a, **k: _fake_quick_optimize_result())

    outcome = maybe_trigger_retune(
        journal, session_id, "test_strat", _mc_result(), account_balance=10_000,
        df=pd.DataFrame(), strategy=object(), risk=RiskConfig(), prop_rules=PropRules(),
        severity_threshold="warning",
    )
    assert outcome.triggered
    assert outcome.trigger_reason is not None
    assert outcome.retune_result is not None
    assert outcome.retune_result.improved is True
    assert outcome.error is None


def test_retune_failure_is_captured_not_raised(monkeypatch):
    journal, session_id = _journal_with_trades([50.0, 50.0, 50.0, -2000.0])

    def _boom(*a, **k):
        raise RuntimeError("optimizer blew up")

    monkeypatch.setattr(auto_retune, "run_quick_optimize", _boom)

    outcome = maybe_trigger_retune(
        journal, session_id, "test_strat", _mc_result(), account_balance=10_000,
        df=pd.DataFrame(), strategy=object(), risk=RiskConfig(), prop_rules=PropRules(),
    )
    assert outcome.triggered
    assert outcome.retune_result is None
    assert "optimizer blew up" in outcome.error


def test_invalid_severity_threshold_raises():
    journal, session_id = _journal_with_trades([80.0] * 30)
    with pytest.raises(ValueError):
        maybe_trigger_retune(
            journal, session_id, "test_strat", _mc_result(), account_balance=10_000,
            df=pd.DataFrame(), strategy=object(), risk=RiskConfig(), prop_rules=PropRules(),
            severity_threshold="not_a_real_level",
        )


def test_critical_threshold_not_triggered_by_mere_watch(monkeypatch):
    # Modest drift that should stay at "watch"/"ok", not reach "critical".
    journal, session_id = _journal_with_trades([70.0] * 30)
    called = {"n": 0}
    monkeypatch.setattr(auto_retune, "run_quick_optimize", lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    outcome = maybe_trigger_retune(
        journal, session_id, "test_strat", _mc_result(), account_balance=10_000,
        df=pd.DataFrame(), strategy=object(), risk=RiskConfig(), prop_rules=PropRules(),
        severity_threshold="critical",
    )
    assert not outcome.triggered
    assert called["n"] == 0
