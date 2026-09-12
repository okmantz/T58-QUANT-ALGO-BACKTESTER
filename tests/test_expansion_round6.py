"""Tests for expansion round 6: 5 more indicators (Williams %R, Rate of
Change, Awesome Oscillator, Chaikin Money Flow, Parabolic SAR) in
app.strategy.indicators, their dispatch through app.strategy.manual's
condition engine, and their 5 new families in app.search.strategy_space
(covered end-to-end for "produces at least one trade" by the existing
parametrized test in test_strategy_space.py; this file adds indicator-
level unit coverage that test doesn't).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.indicators import (
    awesome_oscillator, build_indicator_series, chaikin_money_flow, parabolic_sar, roc, williams_r,
)
from app.strategy.manual import ManualStrategy


def _trending_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.cumsum(rng.normal(0.05, 0.5, n))
    close = 100 + drift
    high = close + rng.random(n) * 0.5
    low = close - rng.random(n) * 0.5
    openp = close + rng.normal(0, 0.1, n)
    vol = rng.random(n) * 1000
    return pd.DataFrame({
        "timestamp": ts, "open": openp, "high": high, "low": low, "close": close, "volume": vol,
    })


def test_williams_r_is_bounded_minus_100_to_0():
    df = _trending_df()
    result = williams_r(df, 14)
    assert (result >= -100).all() and (result <= 0).all()


def test_roc_is_zero_for_a_flat_price_series():
    n = 60
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    flat = pd.Series([100.0] * n)
    df = pd.DataFrame({"timestamp": ts, "open": flat, "high": flat, "low": flat, "close": flat, "volume": 100.0})
    result = roc(df, 10)
    assert np.allclose(result.to_numpy(), 0.0)


def test_roc_reacts_to_a_real_price_move():
    df = _trending_df(n=60)
    baseline = roc(df, 10)
    df.loc[50, "close"] = df.loc[50, "close"] + 50  # sharp spike
    spiked = roc(df, 10)
    assert spiked.iloc[50] > baseline.iloc[50]


def test_awesome_oscillator_is_sma5_minus_sma34_of_midpoint():
    df = _trending_df(n=60)
    ao = awesome_oscillator(df)
    midpoint = (df["high"] + df["low"]) / 2.0
    expected = midpoint.rolling(5, min_periods=5).mean() - midpoint.rolling(34, min_periods=34).mean()
    valid = expected.notna()
    assert np.allclose(ao[valid].to_numpy(), expected[valid].to_numpy())


def test_chaikin_money_flow_is_bounded_roughly_minus1_to_1():
    df = _trending_df()
    cmf = chaikin_money_flow(df, 20)
    valid = cmf.dropna()
    assert len(valid) > 0
    assert (valid >= -1.0001).all() and (valid <= 1.0001).all()


def test_chaikin_money_flow_is_zero_without_a_volume_column():
    df = _trending_df().drop(columns=["volume"])
    cmf = chaikin_money_flow(df, 20)
    assert (cmf == 0.0).all()


def test_parabolic_sar_direction_is_always_plus_or_minus_one():
    df = _trending_df(n=500)
    _, direction = parabolic_sar(df)
    assert set(direction.unique()) <= {1.0, -1.0}
    assert direction.isna().sum() == 0  # unlike supertrend, psar has no ATR warmup period


def test_parabolic_sar_actually_flips_on_a_realistic_series():
    df = _trending_df(n=1000, seed=7)
    _, direction = parabolic_sar(df)
    flips = (direction.diff().fillna(0) != 0).sum()
    assert flips > 0


def test_parabolic_sar_never_sits_on_the_wrong_side_of_price_without_flipping():
    """Regression guard for the exact class of bug supertrend hit last
    round: SAR must stay below the bar's own low while in an uptrend (and
    above the bar's own high while in a downtrend) -- if it doesn't, that
    means price already crossed it without the direction flipping, which
    would make the flip-based entry/exit condition fire too late or never."""
    df = _trending_df(n=1000, seed=3)
    sar, direction = parabolic_sar(df)
    up = direction == 1.0
    down = direction == -1.0
    assert (sar[up] <= df["low"][up]).all()
    assert (sar[down] >= df["high"][down]).all()


@pytest.mark.parametrize("kind", ["williams_r", "roc", "awesome_oscillator", "cmf", "psar_line", "psar_direction"])
def test_build_indicator_series_dispatches_every_new_kind(kind):
    df = _trending_df()
    series = build_indicator_series(df, kind, period=14, column="close")
    assert len(series) == len(df)


@pytest.mark.parametrize("kind", ["williams_r", "roc", "awesome_oscillator", "cmf", "psar_direction"])
def test_manual_strategy_condition_engine_accepts_every_new_kind(kind):
    df = _trending_df()
    config = {
        "name": "round 6 kind dispatch smoke test",
        "entry_conditions": {
            "long": [{"left": {"type": kind, "period": 10}, "operator": ">", "right": {"type": "value", "value": -1e9}}],
            "short": [],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": {"stop_type": "atr", "stop_value": 1.5, "target_type": "atr", "target_value": 2.0},
    }
    strategy = ManualStrategy(config)
    result = strategy.generate(df)
    assert len(result.signals) == len(df)
