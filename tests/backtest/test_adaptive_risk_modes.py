"""Three adaptive-risk modes: ordering of caution, and acceleration can never override a protection."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.backtest.adaptive_risk import (
    MAX_BASE_MULTIPLIER, AdaptiveRiskConfig, AdaptiveRiskRule, AdaptiveRiskState, adaptive_from_settings,
    build_limit_aware_preset, build_mode_preset, normalize_adaptive_mode,
)

RULES = SimpleNamespace(daily_loss_limit_pct=4.0, max_drawdown_pct=8.0)


def _state(daily_loss_pct=0.0, dd_pct=0.0):
    st = AdaptiveRiskState(initial_balance=100_000.0)
    st.day_realized_pnl = -daily_loss_pct * 1000
    st.cumulative_realized_pnl = -dd_pct * 1000
    st.peak_realized_balance = 100_000.0
    return st


def test_unknown_mode_falls_back_to_balanced():
    assert normalize_adaptive_mode("YOLO") == "balanced" and normalize_adaptive_mode(None) == "balanced"
    assert normalize_adaptive_mode(" Accelerated ") == "accelerated"


def test_balanced_is_exactly_the_original_preset():
    a = build_mode_preset(RULES, "balanced")
    b = build_limit_aware_preset(RULES, daily_profit_lock_pct=80.0)
    assert [(r.trigger, r.threshold, r.risk_multiplier) for r in a.rules] == [(r.trigger, r.threshold, r.risk_multiplier) for r in b.rules]
    assert a.base_multiplier == 1.0


def test_healthy_account_sizes_conservative_below_balanced_below_accelerated():
    healthy = {m: _state().active_multiplier(build_mode_preset(RULES, m), 10.0) for m in ("conservative", "balanced", "accelerated")}
    assert healthy["conservative"] < healthy["balanced"] < healthy["accelerated"] <= MAX_BASE_MULTIPLIER


def test_conservative_steps_down_before_balanced():
    st = _state(daily_loss_pct=1.6)      # 40% of a 4% daily limit
    assert st.active_multiplier(build_mode_preset(RULES, "conservative"), 10.0) < 0.75
    assert st.active_multiplier(build_mode_preset(RULES, "balanced"), 10.0) == 1.0


def test_acceleration_is_dropped_the_moment_any_protection_fires():
    st = _state(daily_loss_pct=2.5)      # past 60% of the daily limit -> accelerated's first step
    acc = build_mode_preset(RULES, "accelerated")
    assert st.active_multiplier(acc, 10.0) == pytest.approx(0.5)     # no 1.25x on top of the throttle
    locked = AdaptiveRiskState(initial_balance=100_000.0)
    locked.day_realized_pnl = 4000.0
    assert locked.active_multiplier(acc, 10.0) == 0.0               # a lock still locks


def test_high_volatility_also_drops_acceleration():
    assert _state().active_multiplier(build_mode_preset(RULES, "accelerated"), 99.0) <= 0.25


def test_base_multiplier_is_hard_capped():
    assert AdaptiveRiskConfig(enabled=True, base_multiplier=9.0).base_multiplier == MAX_BASE_MULTIPLIER
    assert AdaptiveRiskConfig(enabled=True, base_multiplier=0.0).base_multiplier > 0


def test_pipeline_helper_keeps_the_balanced_lock_setting_and_modes_own_lock_otherwise():
    bal = adaptive_from_settings(RULES, "balanced", 50.0)
    assert bal.rules[-1].threshold == pytest.approx(4.0 * 0.5)
    con = adaptive_from_settings(RULES, "conservative", 50.0)
    assert con.rules[-1].threshold == pytest.approx(4.0 * 0.6)
