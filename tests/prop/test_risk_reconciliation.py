"""FIX (RESET-ACCT-001/002, RISK-RECON): tests for two related fixes made / Regression coverage for a 2026-09-24 investigation (T58 ES VWMA Trend"""
from __future__ import annotations

import pandas as pd
import pytest
from app.backtest.execution import Trade, run_execution
from app.backtest.risk import RiskConfig
from app.backtest.statistics import (
    compute_risk_reconciliation,
    compute_statistics,
    format_run_summary_line,
    reset_chain_note,
)
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules
import numpy as np
from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskRule
from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig, position_sizing_deviation_message
from app.backtest.statistics import compute_risk_reconciliation
from app.strategy.manual import ManualStrategy


# ---- from test_reset_accounting_and_risk_reconciliation.py ----

def _dummy_trade(entry_time, exit_time, pnl, equity_after, **kwargs) -> Trade:
    return Trade(
        entry_time=entry_time, exit_time=exit_time, direction=1,
        entry_price=100.0, exit_price=100.0 + pnl, size=kwargs.pop("size", 1.0),
        pnl=pnl, pnl_pct=0.0, exit_reason="signal", commission=0.0,
        equity_after=equity_after, **kwargs,
    )


def test_final_segment_net_profit_is_only_the_last_accounts_own_pnl():
    ts = pd.date_range("2024-01-01 00:00", periods=6, freq="h")
    equity_df = pd.DataFrame({
        "timestamp": ts,
        "equity": [10_000.0, 12_000.0, 4_000.0, 10_000.0, 9_500.0, 10_800.0],
    })
    # One reset, right before the trade exiting at ts[3].
    equity_df.attrs["account_reset_events"] = [{
        "reset_at": ts[3], "equity_before_reset": 4_000.0,
        "equity_after_forced_close": 4_000.0, "drawdown_pct": 60.0,
    }]
    trades = [
        _dummy_trade(ts[0], ts[1], 2_000.0, 12_000.0),
        _dummy_trade(ts[1], ts[2], -8_000.0, 4_000.0),   # busts old account -> reset
        _dummy_trade(ts[3], ts[4], -500.0, 9_500.0),     # new account, trade 1
        _dummy_trade(ts[4], ts[5], 1_300.0, 10_800.0),   # new account, trade 2
    ]
    stats = compute_statistics(trades, equity_df, initial_balance=10_000.0)

    # Cumulative across BOTH accounts, unchanged semantics.
    assert stats.net_profit == pytest.approx(2_000.0 - 8_000.0 - 500.0 + 1_300.0)
    assert stats.account_reset_count == 1
    assert stats.is_reset_chain is True
    # The CURRENT account's own P&L: just the last two trades.
    assert stats.final_segment_net_profit == pytest.approx(-500.0 + 1_300.0)
    assert stats.final_segment_trade_count == 2


def test_no_reset_final_segment_equals_net_profit():
    ts = pd.date_range("2024-01-01 00:00", periods=3, freq="h")
    equity_df = pd.DataFrame({"timestamp": ts, "equity": [10_000.0, 10_500.0, 11_000.0]})
    trades = [
        _dummy_trade(ts[0], ts[1], 500.0, 10_500.0),
        _dummy_trade(ts[1], ts[2], 500.0, 11_000.0),
    ]
    stats = compute_statistics(trades, equity_df, initial_balance=10_000.0)
    assert stats.account_reset_count == 0
    assert stats.is_reset_chain is False
    assert stats.final_segment_net_profit == pytest.approx(stats.net_profit)
    assert stats.final_segment_trade_count == len(trades)


def test_forced_close_trade_at_the_reset_instant_stays_in_the_old_segment():
    """The trade whose exit_time == the reset's own timestamp is the
    forced-close that CAUSED the reset -- it belongs to the dying
    account, not the fresh one that starts immediately after."""
    ts = pd.date_range("2024-01-01 00:00", periods=3, freq="h")
    equity_df = pd.DataFrame({"timestamp": ts, "equity": [10_000.0, 4_000.0, 10_000.0]})
    equity_df.attrs["account_reset_events"] = [{
        "reset_at": ts[1], "equity_before_reset": 4_000.0,
        "equity_after_forced_close": 4_000.0, "drawdown_pct": 60.0,
    }]
    trades = [
        _dummy_trade(ts[0], ts[1], -6_000.0, 4_000.0),  # exits exactly at reset_at
    ]
    stats = compute_statistics(trades, equity_df, initial_balance=10_000.0)
    # No trade exists yet in the post-reset segment, so the "last segment"
    # (the segment of the last trade in the list) is still segment 0 --
    # this forced-close trade itself, correctly kept with the dying
    # account it closed out rather than the fresh one that starts after it.
    assert stats.final_segment_trade_count == 1
    assert stats.final_segment_net_profit == pytest.approx(-6_000.0)


def test_risk_reconciliation_flags_a_position_size_cap():
    ts = pd.date_range("2024-01-01", periods=1, freq="h")[0]
    # Configured to risk $500, but the position was actually only sized
    # to risk $200 at its own stop (a cap/throttle engaged).
    trades = [Trade(
        entry_time=ts, exit_time=ts, direction=1, entry_price=100.0, exit_price=98.0,
        size=100.0, pnl=-200.0, pnl_pct=-2.0, exit_reason="stop_loss", commission=0.0,
        equity_after=9_800.0, initial_risk=2.0, intended_risk_dollars=500.0,
    )]
    recon = compute_risk_reconciliation(trades)
    assert recon["avg_intended_risk_dollars"] == pytest.approx(500.0)
    assert recon["avg_actual_stop_risk_dollars"] == pytest.approx(200.0)  # 2.0 * 100
    assert recon["pct_trades_position_capped"] == pytest.approx(100.0)
    assert recon["pct_trades_risk_overshoot"] == pytest.approx(0.0)


def test_risk_reconciliation_flags_a_gap_through_overshoot():
    ts = pd.date_range("2024-01-01", periods=1, freq="h")[0]
    # Sized to risk exactly $500 at its stop, but a gap-through fill
    # realized a much larger loss than that.
    trades = [Trade(
        entry_time=ts, exit_time=ts, direction=1, entry_price=100.0, exit_price=90.0,
        size=100.0, pnl=-1_500.0, pnl_pct=-15.0, exit_reason="stop_loss", commission=0.0,
        equity_after=8_500.0, initial_risk=5.0, intended_risk_dollars=500.0,
    )]
    recon = compute_risk_reconciliation(trades)
    assert recon["avg_actual_stop_risk_dollars"] == pytest.approx(500.0)  # 5.0 * 100
    assert recon["pct_trades_position_capped"] == pytest.approx(0.0)
    assert recon["pct_trades_risk_overshoot"] == pytest.approx(100.0)
    assert recon["avg_realized_loss_on_losers"] == pytest.approx(1_500.0)


def test_risk_reconciliation_empty_for_trades_with_no_recorded_risk():
    ts = pd.date_range("2024-01-01", periods=1, freq="h")[0]
    trades = [_dummy_trade(ts, ts, 100.0, 10_100.0)]  # no initial_risk / intended_risk_dollars
    recon = compute_risk_reconciliation(trades)
    assert recon["avg_intended_risk_dollars"] == 0.0
    assert recon["avg_actual_stop_risk_dollars"] == 0.0
    assert recon["pct_trades_position_capped"] == 0.0
    assert recon["pct_trades_risk_overshoot"] == 0.0


def test_execution_records_intended_risk_dollars_at_entry():
    """RISK-RECON end-to-end: run_execution should tag every trade with
    the raw %-of-equity target it was sized against, independent of
    whatever the actual sized stop risk (initial_risk * size) works out
    to."""
    ts = pd.date_range("2024-01-01", periods=10, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [100.0] * 10, "high": [101.0] * 10, "low": [99.0] * 10, "close": [100.0] * 10,
    })
    signals = pd.Series([1] + [0] * 9, index=df.index)
    risk = RiskConfig(initial_balance=50_000.0, risk_value=0.5, pip_size=0.01)  # 0.5% of 50k = $250
    trades, _equity_df = run_execution(df, signals, risk, stop_loss_pips=20, take_profit_pips=40)
    assert len(trades) >= 1
    t = trades[0]
    assert t.intended_risk_dollars == pytest.approx(250.0)


def _prop_rules(**overrides) -> PropRules:
    defaults = dict(account_size=50_000.0, evaluation_profit_target_pct=8.0,
                     daily_loss_limit_pct=5.0, max_drawdown_pct=10.0)
    defaults.update(overrides)
    return PropRules(**defaults)


def _trades_from_pnls(pnls: list[float]) -> list[Trade]:
    ts = pd.date_range("2024-01-01", periods=len(pnls), freq="D")
    return [
        Trade(entry_time=t, exit_time=t, direction=1, entry_price=100.0, exit_price=100.0,
              size=1.0, pnl=p, pnl_pct=0.0, exit_reason="signal", commission=0.0, equity_after=0.0)
        for t, p in zip(ts, pnls)
    ]


def test_per_attempt_matches_chain_level_when_reset_on_breach_is_off():
    """Default (non-reset) case: every simulated path has exactly one
    attempt, so the new per-attempt fields must be byte-identical to the
    existing chain-level fields -- no behavior change for the common
    case."""
    pnls = [500.0, -200.0, 800.0, -300.0, 1_200.0] * 10
    trades = _trades_from_pnls(pnls)
    rules = _prop_rules()
    mc = run_monte_carlo(trades, rules, MonteCarloConfig(n_simulations=200, reset_on_breach=False))
    assert mc.per_attempt_pass_probability == pytest.approx(mc.evaluation_pass_probability)
    assert mc.per_attempt_payout_probability == pytest.approx(mc.first_payout_probability)
    assert mc.total_independent_attempts == mc.n_simulations


def test_per_attempt_can_diverge_from_chain_level_when_reset_on_breach_is_on():
    """With reset_on_breach on and a strategy that busts often, a long
    chain can have a much higher "did at least one attempt pass" rate
    than the true per-attempt rate -- this is exactly the ChatGPT-flagged
    confusion. Assert the two fields are computed independently and the
    per-attempt rate is never (mathematically cannot be) higher than the
    any-attempt chain rate."""
    pnls = ([-4_900.0] + [100.0] * 3) * 30  # mostly a small bust, few small wins
    trades = _trades_from_pnls(pnls)
    rules = _prop_rules(max_drawdown_pct=10.0, account_size=50_000.0)
    mc = run_monte_carlo(trades, rules, MonteCarloConfig(n_simulations=150, reset_on_breach=True))
    assert mc.total_independent_attempts >= mc.n_simulations
    assert mc.per_attempt_pass_probability <= mc.evaluation_pass_probability + 1e-9
    assert mc.per_attempt_payout_probability <= mc.first_payout_probability + 1e-9


def test_reset_chain_note_empty_when_no_resets_occurred():
    ts = pd.date_range("2024-01-01", periods=2, freq="h")
    equity_df = pd.DataFrame({"timestamp": ts, "equity": [10_000.0, 10_500.0]})
    trades = [_dummy_trade(ts[0], ts[1], 500.0, 10_500.0)]
    stats = compute_statistics(trades, equity_df, initial_balance=10_000.0)
    assert reset_chain_note(stats, 80.0, 60.0) == ""
    line = format_run_summary_line("Baseline", 1, stats, 80.0, 60.0)
    assert line == "Baseline: 1 trades, net $500.00, eval pass 80.0%, payout 60.0%."


def test_reset_chain_note_explains_cumulative_vs_final_segment():
    ts = pd.date_range("2024-01-01 00:00", periods=4, freq="h")
    equity_df = pd.DataFrame({"timestamp": ts, "equity": [10_000.0, 2_000.0, 10_000.0, 10_200.0]})
    equity_df.attrs["account_reset_events"] = [{
        "reset_at": ts[2], "equity_before_reset": 2_000.0,
        "equity_after_forced_close": 2_000.0, "drawdown_pct": 80.0,
    }]
    trades = [
        _dummy_trade(ts[0], ts[1], -8_000.0, 2_000.0),
        _dummy_trade(ts[2], ts[3], 200.0, 10_200.0),
    ]
    stats = compute_statistics(trades, equity_df, initial_balance=10_000.0)
    note = reset_chain_note(stats, 0.0, 0.0, per_attempt_eval_pass_pct=5.0, per_attempt_payout_pct=2.0, total_attempts=1000)
    assert "CUMULATIVE P&L" in note
    assert "$200.00" in note  # final_segment_net_profit
    assert "per-ATTEMPT rate" in note
    assert "1,000 independent attempts" in note


# ---- from test_risk_reconciliation_no_throttle.py ----

def _es_like_df(n=800, seed=11):
    """Synthetic ES-scaled bars: price ~5000-6000, enough range for a
    multi-point ATR stop, so position_size() produces a non-trivial
    fractional 'units' figure (not always exactly 1.0) -- a more honest
    check than a flat/trivial series would be."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    walk = np.cumsum(rng.normal(0, 3.0, n))
    close = 5500 + walk
    high = close + rng.random(n) * 4.0
    low = close - rng.random(n) * 4.0
    openp = close + rng.normal(0, 1.0, n)
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": high, "low": low, "close": close})


def _atr_trend_config():
    """A simple ATR-stop strategy so most trades carry a real
    Trade.initial_risk (needed for compute_risk_reconciliation to have
    anything to reconcile)."""
    return {
        "name": "es reconciliation probe",
        "indicators": [
            {"type": "ema", "period": 20, "column": "close", "as": "ema20"},
        ],
        "long_entry": "close > ema20", "long_exit": "close < ema20",
        "short_entry": "close < ema20", "short_exit": "close > ema20",
        "risk_management": {
            "stop_type": "atr", "stop_value": 2.0, "stop_atr_period": 14,
            "target_type": "atr", "target_value": 3.0, "target_atr_period": 14,
        },
    }


def _es_risk_no_caps(**overrides) -> RiskConfig:
    base = dict(
        initial_balance=50_000.0, risk_mode="percent", risk_value=1.0,
        pip_size=1.0, max_position_size=None, contract_size=None,
        commission_per_trade=0.0, slippage_pips=0.0, spread_pips=0.0,
    )
    base.update(overrides)
    return RiskConfig(**base)


def test_reconciliation_exact_with_no_caps_or_throttle():
    df = _es_like_df()
    strategy = ManualStrategy(_atr_trend_config())
    risk = _es_risk_no_caps()

    result = run_backtest(df, strategy, risk)
    recon = compute_risk_reconciliation(result.trades)

    assert result.statistics.total_trades > 5, "need a real sample for this check to mean anything"
    # No cap, no contract-size rounding, no adaptive throttle configured ->
    # actual stop risk must equal intended risk, not just be close to it.
    assert recon["avg_intended_risk_dollars"] == pytest_approx(recon["avg_actual_stop_risk_dollars"])
    assert recon["pct_trades_position_capped"] == 0.0
    assert recon["avg_adaptive_risk_multiplier"] == 1.0
    assert recon["pct_trades_adaptive_throttle_active"] == 0.0
    # And therefore no position-sizing-deviation warning should fire at all.
    assert position_sizing_deviation_message(result.statistics.__dict__) is None


def test_reconciliation_attributes_gap_to_adaptive_throttle_when_active():
    df = _es_like_df(seed=23)
    strategy = ManualStrategy(_atr_trend_config())
    risk = _es_risk_no_caps()
    # A throttle that always cuts size in half from the very first trade,
    # so it's unambiguously "active" and "material" on effectively every
    # entry regardless of this run's actual drawdown/streak path.
    adaptive = AdaptiveRiskConfig(
        enabled=True,
        rules=(AdaptiveRiskRule(trigger="drawdown_pct", threshold=0.0, risk_multiplier=0.5),),
    )

    result = run_backtest(df, strategy, risk, adaptive_risk=adaptive)
    recon = compute_risk_reconciliation(result.trades)

    assert result.statistics.total_trades > 5
    assert recon["pct_trades_adaptive_throttle_active"] > 50.0
    assert recon["avg_adaptive_risk_multiplier"] < 0.98
    # Actual risk should now come in materially below intended, driven by
    # the throttle -- and the resulting stats should say so by name.
    stats_dict = result.statistics.__dict__
    msg = position_sizing_deviation_message(stats_dict)
    assert msg is not None
    assert "adaptive-risk throttle" in msg
    assert "rules out the adaptive-risk throttle" not in msg


def pytest_approx(value, rel=1e-6, abs_=1e-6):
    import pytest as _pytest
    return _pytest.approx(value, rel=rel, abs=abs_)
