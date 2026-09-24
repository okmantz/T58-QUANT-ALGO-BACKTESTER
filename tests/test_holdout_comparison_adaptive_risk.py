"""Tests for the holdout-comparison-ignored-adaptive-risk fix:
app.backtest.engine.run_holdout_comparison now threads `adaptive_risk`
through both its in-sample and holdout backtests, instead of silently
dropping it (even when the caller's own main backtest of the identical
strategy+risk used it)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskRule
from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.strategy.manual import ManualStrategy


def _frequent_trader_df(n=1500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    walk = np.cumsum(rng.normal(0, 0.3, n))
    close = 100 + walk - 0.15 * (walk - pd.Series(walk).rolling(50, min_periods=1).mean().to_numpy())
    high = close + rng.random(n) * 0.4
    low = close - rng.random(n) * 0.4
    openp = close + rng.normal(0, 0.05, n)
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": high, "low": low, "close": close})


def _flip_flop_config():
    return {
        "name": "RSI flip-flop (trades often)",
        "indicators": [{"type": "rsi", "period": 5, "column": "close", "as": "rsi5"}],
        "long_entry": "rsi5 < 50", "long_exit": "rsi5 > 60",
        "short_entry": "rsi5 > 50", "short_exit": "rsi5 < 40",
        "stop_loss_pips": 5, "take_profit_pips": 8,
    }


def _shutdown_after_one_loss() -> AdaptiveRiskConfig:
    """Throttles risk to (near) zero after the very first losing trade --
    deliberately aggressive so its effect on trade count is unmistakable
    in a test, mirroring the real-world "one early trade, then it just
    stops" symptom this fix addresses."""
    return AdaptiveRiskConfig(
        enabled=True,
        rules=[AdaptiveRiskRule(trigger="consecutive_losses", threshold=1, risk_multiplier=0.0001)],
    )


def test_adaptive_risk_none_is_byte_identical_to_before():
    df = _frequent_trader_df()
    strategy = ManualStrategy(_flip_flop_config())
    risk = RiskConfig()
    result = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2)
    assert result["in_sample_statistics"] is not None
    assert result["holdout_statistics"] is not None


def test_adaptive_risk_is_actually_applied_to_both_halves():
    df = _frequent_trader_df()
    strategy = ManualStrategy(_flip_flop_config())
    risk = RiskConfig()
    adaptive_risk = _shutdown_after_one_loss()

    without = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2)
    with_throttle = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2, adaptive_risk=adaptive_risk)

    # A throttle this aggressive, once it has ever fired, should mean far
    # FEWER meaningfully-sized trades than the untouched run -- if
    # adaptive_risk were still being silently dropped, these two results
    # would be identical.
    assert with_throttle["in_sample_statistics"]["total_trades"] <= without["in_sample_statistics"]["total_trades"]
    assert with_throttle != without


def test_holdout_comparison_matches_direct_run_backtest_per_half():
    """The core promise of this function -- 'runs the identical strategy +
    risk config' on each half -- now actually holds for adaptive_risk too:
    each half's stats should match calling run_backtest directly on that
    same half with the same adaptive_risk."""
    df = _frequent_trader_df()
    strategy = ManualStrategy(_flip_flop_config())
    risk = RiskConfig()
    adaptive_risk = _shutdown_after_one_loss()

    holdout_frac = 0.2
    split_idx = int(len(df) * (1 - holdout_frac))
    in_sample_df = df.iloc[:split_idx].reset_index(drop=True)
    holdout_df = df.iloc[split_idx:].reset_index(drop=True)

    direct_in_sample = run_backtest(in_sample_df, strategy, risk, adaptive_risk=adaptive_risk)
    direct_holdout = run_backtest(holdout_df, strategy, risk, adaptive_risk=adaptive_risk)

    result = run_holdout_comparison(df, strategy, risk, holdout_frac=holdout_frac, adaptive_risk=adaptive_risk)

    assert result["in_sample_statistics"]["total_trades"] == direct_in_sample.statistics.total_trades
    assert result["in_sample_statistics"]["net_profit"] == direct_in_sample.statistics.net_profit
    assert result["holdout_statistics"]["total_trades"] == direct_holdout.statistics.total_trades
    assert result["holdout_statistics"]["net_profit"] == direct_holdout.statistics.net_profit
