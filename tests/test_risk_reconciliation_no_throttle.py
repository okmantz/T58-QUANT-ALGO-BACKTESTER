"""
Regression coverage for a 2026-09-24 investigation (T58 ES VWMA Trend
Pullback vs a RoboQuant comparison): a Full Pipeline report showed 60% of
trades sized materially BELOW the configured risk target on ES, with
avg_actual_stop_risk_dollars running ~41% below avg_intended_risk_dollars
-- but the report's own risk_config had max_position_size=None and
contract_size=None, so neither of the two documented capping mechanisms
could explain it, and adaptive risk was never surfaced in the report at
all to check the third.

This file pins two things going forward:

1. compute_risk_reconciliation's own arithmetic is exact (actual == intended,
   not just "close") for a point-value-heavy instrument (ES, pip_size=1.0)
   when NEITHER max_position_size NOR contract_size is set and NO adaptive
   risk config is active -- i.e. the reconciliation gap in a report like
   that one can ONLY be caused by a cap, contract-size rounding, or an
   active adaptive-risk throttle, never a silent unit-conversion bug. If
   this ever stops being exact, something in risk.position_size /
   execution.py's sizing has regressed.

2. When an adaptive-risk throttle IS active and materially shrinking size,
   compute_risk_reconciliation's new avg_adaptive_risk_multiplier / pct_
   trades_adaptive_throttle_active fields, and position_sizing_deviation_
   message's attribution built from them, correctly name it as the cause
   instead of leaving it as an unresolved guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskRule
from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig, position_sizing_deviation_message
from app.backtest.statistics import compute_risk_reconciliation
from app.strategy.manual import ManualStrategy


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
