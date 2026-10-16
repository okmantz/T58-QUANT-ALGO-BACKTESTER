"""
Tests for the 2026-09-24 "headline risk flags" upgrade: app.reports.
generator._headline_risk_flags / _headline_warnings_banner, and their
wiring into build_report's returned dict.

Added after an external RoboQuant comparison found a Full Pipeline report
where a single trade was 31.8% of all gross profit (net profit negative
without it) and the ICIR/Bonferroni significance gate couldn't run at
all -- both facts were on record in the report's raw JSON, but neither
was visible anywhere without reading the concentration-check table and
verdict_reasons list closely. These tests pin that both findings now
surface as short, prominent strings any UI can render without its own
re-derivation logic.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.reports.generator import (
    _headline_risk_flags,
    _headline_warnings_banner,
    build_report,
)


@dataclass
class _FakeStats:
    total_trades: int


def test_no_flags_on_a_clean_diversified_result():
    concentration = {
        "best_trade_pnl": 100.0, "best_trade_pct_of_gross_profit": 8.0,
        "net_profit_excluding_best_trade": 900.0,
    }
    flags = _headline_risk_flags(concentration, verdict_reasons=None, statistics=_FakeStats(total_trades=300))
    assert flags == []


def test_flags_when_removing_best_trade_goes_negative():
    concentration = {
        "best_trade_pnl": 981.0, "best_trade_pct_of_gross_profit": 31.8,
        "net_profit_excluding_best_trade": -365.0,
    }
    flags = _headline_risk_flags(concentration, verdict_reasons=None, statistics=_FakeStats(total_trades=40))
    joined = " ".join(flags)
    assert "Single-trade concentration" in joined
    assert "31.8" in joined.replace(".0", ".0") or "32%" in joined or "31.8%" in joined or "32.0%" in joined
    assert "NEGATIVE" in joined


def test_flags_material_concentration_even_without_going_negative():
    concentration = {
        "best_trade_pnl": 500.0, "best_trade_pct_of_gross_profit": 40.0,
        "net_profit_excluding_best_trade": 200.0,
    }
    flags = _headline_risk_flags(concentration, verdict_reasons=None, statistics=_FakeStats(total_trades=300))
    assert any("Single-trade concentration" in f for f in flags)
    assert not any("NEGATIVE" in f for f in flags)


def test_flags_failed_icir_gate_from_verdict_reasons():
    concentration = {"best_trade_pct_of_gross_profit": 5.0, "net_profit_excluding_best_trade": 100.0}
    reasons = [
        "Supporting diagnostic: did NOT pass the ICIR / signal-decay / Bonferroni-corrected "
        "significance gate (not enough distinct time periods with trades).",
    ]
    flags = _headline_risk_flags(concentration, verdict_reasons=reasons, statistics=_FakeStats(total_trades=300))
    assert any("Signal significance UNPROVEN" in f for f in flags)


def test_no_significance_flag_when_gate_passed():
    concentration = {"best_trade_pct_of_gross_profit": 5.0, "net_profit_excluding_best_trade": 100.0}
    reasons = ["T58 Score: 82.0/100 -- Elite"]
    flags = _headline_risk_flags(concentration, verdict_reasons=reasons, statistics=_FakeStats(total_trades=300))
    assert not any("Signal significance" in f for f in flags)


def test_flags_small_sample_under_fifty_trades():
    concentration = {"best_trade_pct_of_gross_profit": 5.0, "net_profit_excluding_best_trade": 100.0}
    flags = _headline_risk_flags(concentration, verdict_reasons=None, statistics=_FakeStats(total_trades=40))
    assert any("Small sample" in f for f in flags)
    assert any("40 trade" in f for f in flags)


def test_no_small_sample_flag_at_or_above_fifty_trades():
    concentration = {"best_trade_pct_of_gross_profit": 5.0, "net_profit_excluding_best_trade": 100.0}
    flags = _headline_risk_flags(concentration, verdict_reasons=None, statistics=_FakeStats(total_trades=50))
    assert not any("Small sample" in f for f in flags)


def test_missing_statistics_does_not_crash_and_skips_sample_flag():
    concentration = {"best_trade_pct_of_gross_profit": 5.0, "net_profit_excluding_best_trade": 100.0}
    flags = _headline_risk_flags(concentration, verdict_reasons=None, statistics=None)
    assert not any("Small sample" in f for f in flags)


def test_banner_empty_string_when_no_flags():
    assert _headline_warnings_banner([]) == ""


def test_banner_uses_its_own_css_class_not_verdict_banner():
    """Must never be mistaken for (or styled identically to) the
    Full-Pipeline-only verdict banner -- this fires for ANY report."""
    html = _headline_warnings_banner(["\u26a0 something fragile"])
    assert "risk-flags-banner" in html
    assert 'class="verdict-banner' not in html
    assert "something fragile" in html


def test_build_report_dict_carries_headline_warnings_key(monkeypatch):
    """build_report itself must compute and attach headline_warnings for
    every report, not just Full Pipeline ones -- Run & Report and every
    other caller get this for free."""
    from app.backtest.execution import Trade
    from app.backtest.engine import BacktestResult
    from app.backtest.statistics import BacktestStatistics
    import pandas as pd

    trades = [
        Trade(
            entry_time=pd.Timestamp("2024-01-01"), exit_time=pd.Timestamp("2024-01-02"),
            direction=1, entry_price=100.0, exit_price=110.0, size=10.0, pnl=981.0,
            pnl_pct=0.1, exit_reason="take_profit", commission=0.0, equity_after=50981.0,
        ),
        Trade(
            entry_time=pd.Timestamp("2024-01-03"), exit_time=pd.Timestamp("2024-01-04"),
            direction=1, entry_price=100.0, exit_price=95.0, size=10.0, pnl=-365.0,
            pnl_pct=-0.05, exit_reason="stop_loss", commission=0.0, equity_after=50616.0,
        ),
    ]
    stats = BacktestStatistics(
        net_profit=616.0, gross_profit=981.0, gross_loss=-365.0, return_pct=1.2,
        average_trade=308.0, win_rate=50.0, loss_rate=50.0, average_winner=981.0,
        average_loser=-365.0, largest_winner=981.0, largest_loser=-365.0,
        max_drawdown=-365.0, max_drawdown_pct=0.7, average_drawdown_pct=0.5,
        max_daily_drawdown_pct=0.5, max_weekly_drawdown_pct=0.5, max_losing_streak=1,
        max_winning_streak=1, profit_factor=2.69, expectancy=308.0, average_r=0.5,
        risk_reward=2.69, sharpe_ratio=0.5, sortino_ratio=0.5, calmar_ratio=1.7,
        total_trades=2,
    )
    bt_result = BacktestResult(
        strategy_name="probe", trades=trades,
        equity_curve=pd.DataFrame({"timestamp": [trades[0].entry_time], "equity": [50616.0]}),
        statistics=stats, initial_balance=50000.0, warnings=[],
    )

    from app.prop.simulator import PropRules
    from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
    from app.prop.simulator import simulate_account

    rules = PropRules()
    mc_result = run_monte_carlo(trades, rules, MonteCarloConfig(n_simulations=50, random_seed=1))
    single_run = simulate_account([t.pnl for t in trades], [t.exit_time for t in trades], rules)

    report = build_report(
        "probe", "manual", "ES", "1h", ("2024-01-01", "2024-01-04"),
        bt_result, rules, single_run, mc_result,
    )
    assert "headline_warnings" in report
    assert any("Single-trade concentration" in f for f in report["headline_warnings"])
