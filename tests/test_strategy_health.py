from __future__ import annotations

from pathlib import Path

import pytest

from app.forward_test.journal import ForwardTestJournal
from app.monitoring.strategy_health import (
    StrategyHealthError,
    check_strategy_health,
)
from app.monte_carlo.engine import MonteCarloResult


def _mc_result(**overrides) -> MonteCarloResult:
    defaults = dict(
        n_simulations=1000,
        evaluation_pass_probability=0.6,
        first_payout_probability=0.5,
        failure_before_payout_probability=0.4,
        multiple_payout_probability=0.2,
        median_days_to_pass=20.0,
        median_days_to_first_payout=40.0,
        average_days_to_first_payout=42.0,
        median_return_pct=8.0,
        mean_return_pct=7.5,
        expected_payout=1000.0,
        median_payout=900.0,
        total_simulated_withdrawals=500_000.0,
        median_drawdown_pct=3.0,
        p95_drawdown_pct=6.0,
        worst_drawdown_pct=9.0,
        risk_of_ruin_pct=2.0,
        median_max_losing_streak=4.0,
        worst_max_losing_streak=9,
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


def test_no_closed_trades_raises():
    journal, session_id = _journal_with_trades([])
    with pytest.raises(StrategyHealthError):
        check_strategy_health(journal, session_id, "test_strat", _mc_result(), account_balance=10_000)


def test_healthy_performance_produces_no_critical_flags():
    # 30 winning trades, comfortably within the predicted band's median.
    journal, session_id = _journal_with_trades([80.0] * 30)
    result = check_strategy_health(journal, session_id, "test_strat", _mc_result(), account_balance=10_000)
    assert result.n_closed_trades == 30
    assert result.overall_severity in ("ok", "watch")
    assert not any(f.severity == "critical" for f in result.flags)


def test_severe_losses_flag_return_drift():
    # Deep losses, well outside even the 5th-percentile predicted band.
    journal, session_id = _journal_with_trades([-500.0] * 30)
    result = check_strategy_health(journal, session_id, "test_strat", _mc_result(), account_balance=10_000)
    return_flags = [f for f in result.flags if f.metric == "return"]
    assert return_flags
    assert return_flags[0].severity in ("warning", "critical")


def test_drawdown_beyond_worst_case_is_critical():
    # A large realized drawdown -- one huge loss after modest gains.
    journal, session_id = _journal_with_trades([50.0, 50.0, 50.0, -2000.0])
    result = check_strategy_health(journal, session_id, "test_strat", _mc_result(), account_balance=10_000)
    dd_flags = [f for f in result.flags if f.metric == "drawdown"]
    assert dd_flags
    assert dd_flags[0].severity == "critical"


def test_losing_streak_beyond_worst_case_is_critical():
    pnls = [-10.0] * 10  # streak of 10 exceeds worst_max_losing_streak=9
    journal, session_id = _journal_with_trades(pnls)
    result = check_strategy_health(journal, session_id, "test_strat", _mc_result(), account_balance=10_000)
    streak_flags = [f for f in result.flags if f.metric == "losing_streak"]
    assert streak_flags
    assert streak_flags[0].severity == "critical"


def test_small_sample_warning_present():
    journal, session_id = _journal_with_trades([10.0, -5.0, 8.0])
    result = check_strategy_health(journal, session_id, "test_strat", _mc_result(), account_balance=10_000)
    assert any("Only 3 closed trades" in w for w in result.warnings)


def test_invalid_account_balance_raises():
    journal, session_id = _journal_with_trades([10.0])
    with pytest.raises(StrategyHealthError):
        check_strategy_health(journal, session_id, "test_strat", _mc_result(), account_balance=0)


def test_to_dict_and_render_table_roundtrip():
    journal, session_id = _journal_with_trades([10.0, 20.0, -5.0])
    result = check_strategy_health(journal, session_id, "test_strat", _mc_result(), account_balance=10_000)
    d = result.to_dict()
    assert d["session_id"] == session_id
    assert "overall_severity" in d
    table = result.render_table()
    assert "test_strat" in table
