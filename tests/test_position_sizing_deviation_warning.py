"""Tests for the buried-position-sizing-deviation upgrade:
app.backtest.risk.position_sizing_deviation_message / has_position_sizing_
deviation, and its wiring into app.backtest.engine.run_backtest's
BacktestResult.warnings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig, has_position_sizing_deviation, position_sizing_deviation_message
from app.strategy.manual import ManualStrategy


def test_message_none_when_below_threshold():
    stats = {"pct_trades_position_capped": 5.0, "pct_trades_risk_overshoot": 3.0}
    assert position_sizing_deviation_message(stats) is None


def test_message_fires_on_material_capping():
    stats = {"pct_trades_position_capped": 89.0, "pct_trades_risk_overshoot": 2.0}
    msg = position_sizing_deviation_message(stats)
    assert msg is not None
    assert "89%" in msg
    assert "BELOW" in msg
    assert has_position_sizing_deviation([msg])


def test_message_fires_on_material_overshoot():
    stats = {"pct_trades_position_capped": 1.0, "pct_trades_risk_overshoot": 40.0}
    msg = position_sizing_deviation_message(stats)
    assert msg is not None
    assert "40%" in msg
    assert "gap-through" in msg


def test_has_position_sizing_deviation_false_when_absent():
    assert not has_position_sizing_deviation(["some other warning", "another one"])


def _synthetic_df(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    walk = np.cumsum(rng.normal(0, 0.4, n))
    close = 1000 + walk
    high = close + rng.random(n) * 0.5
    low = close - rng.random(n) * 0.5
    openp = close + rng.normal(0, 0.05, n)
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": high, "low": low, "close": close})


def _flip_flop_config():
    return {
        "name": "rsi flip-flop",
        "indicators": [{"type": "rsi", "period": 5, "column": "close", "as": "rsi5"}],
        "long_entry": "rsi5 < 50", "long_exit": "rsi5 > 60",
        "short_entry": "rsi5 > 50", "short_exit": "rsi5 < 40",
        "stop_loss_pips": 5, "take_profit_pips": 8,
    }


def test_run_backtest_never_crashes_regardless_of_warning():
    """Smoke test: whatever the real pct values come out to on a normal
    run, run_backtest must not raise, and warnings stays a plain list of
    strings."""
    df = _synthetic_df()
    strategy = ManualStrategy(_flip_flop_config())
    risk = RiskConfig(pip_size=1.0, contract_size=None)
    result = run_backtest(df, strategy, risk)
    assert isinstance(result.warnings, list)
    assert all(isinstance(w, str) for w in result.warnings)
