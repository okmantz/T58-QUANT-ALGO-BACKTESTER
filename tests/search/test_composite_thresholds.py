"""Tests for app.strategy.composite_thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.composite_thresholds import (
    ThresholdError,
    cross_level,
    cross_lines,
    derivative_threshold,
    hold_level,
    in_range,
    kurtosis_threshold,
    mix_thresholds,
    preview_signal,
    skew_threshold,
    stdv_bands_threshold,
)


def _oscillating_df(n=600, seed=11):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 100.0
    rows = []
    for i in range(n):
        step = 0.1 * np.sin(i / 40) + rng.normal(0, 0.05)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.02))
        l = min(o, c) - abs(rng.normal(0, 0.02))
        rows.append((ts[i], o, h, l, c, 100 + rng.integers(0, 50)))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_cross_level_fires_only_on_the_crossing_bar():
    df = _oscillating_df()
    sig = cross_level(df, "rsi", 14, 30, direction="above")
    assert sig.triggered.dtype == bool
    # every trigger bar must have RSI > 30 while the PRIOR bar was <= 30
    from app.strategy.indicators import build_indicator_series
    rsi = build_indicator_series(df, "rsi", period=14)
    idx = sig.triggered[sig.triggered].index
    for i in idx:
        assert rsi.loc[i] > 30
        assert rsi.shift(1).loc[i] <= 30


def test_cross_level_rejects_bad_direction():
    df = _oscillating_df()
    with pytest.raises(ThresholdError):
        cross_level(df, "rsi", 14, 30, direction="sideways")


def test_cross_lines_fast_above_slow():
    df = _oscillating_df()
    sig = cross_lines(df, "ema", 9, "ema", 21, direction="above")
    assert sig.n_triggers >= 0
    assert "ema(9)" in sig.label and "ema(21)" in sig.label


def test_in_range_rejects_inverted_bounds():
    df = _oscillating_df()
    with pytest.raises(ThresholdError):
        in_range(df, "rsi", 14, 60, 40)


def test_in_range_all_triggers_within_bounds():
    df = _oscillating_df()
    sig = in_range(df, "rsi", 14, 40, 60)
    from app.strategy.indicators import build_indicator_series
    rsi = build_indicator_series(df, "rsi", period=14)
    idx = sig.triggered[sig.triggered].index
    assert ((rsi.loc[idx] >= 40) & (rsi.loc[idx] <= 60)).all()


def test_hold_level_requires_minimum_consecutive_bars():
    df = _oscillating_df()
    sig_short = hold_level(df, "rsi", 14, 50, direction="above", min_bars=1)
    sig_long = hold_level(df, "rsi", 14, 50, direction="above", min_bars=10)
    # A longer required hold can never have MORE trigger bars than a
    # shorter one over the same series.
    assert sig_long.n_triggers <= sig_short.n_triggers


def test_hold_level_rejects_bad_min_bars():
    df = _oscillating_df()
    with pytest.raises(ThresholdError):
        hold_level(df, "rsi", 14, 50, min_bars=0)


def test_stdv_bands_threshold_runs():
    df = _oscillating_df()
    sig = stdv_bands_threshold(df)
    assert sig.n_triggers >= 0


def test_skew_and_kurtosis_thresholds_run():
    df = _oscillating_df()
    sk = skew_threshold(df)
    ku = kurtosis_threshold(df)
    assert sk.n_triggers >= 0
    assert ku.n_triggers >= 0


def test_derivative_threshold_runs_and_rejects_bad_direction():
    df = _oscillating_df()
    sig = derivative_threshold(df, direction="above")
    assert sig.n_triggers >= 0
    with pytest.raises(ThresholdError):
        derivative_threshold(df, direction="sideways")


def test_mix_thresholds_and_is_subset_of_each_input():
    df = _oscillating_df()
    a = cross_level(df, "rsi", 14, 50, direction="above")
    b = in_range(df, "rsi", 14, 30, 70)
    combo = mix_thresholds([a, b], mode="and")
    # AND can never have more triggers than either input alone.
    assert combo.n_triggers <= a.n_triggers
    assert combo.n_triggers <= b.n_triggers
    assert (combo.triggered == (a.triggered & b.triggered)).all()


def test_mix_thresholds_or_is_superset_of_each_input():
    df = _oscillating_df()
    a = cross_level(df, "rsi", 14, 50, direction="above")
    b = in_range(df, "rsi", 14, 30, 70)
    combo = mix_thresholds([a, b], mode="or")
    assert combo.n_triggers >= a.n_triggers
    assert combo.n_triggers >= b.n_triggers
    assert (combo.triggered == (a.triggered | b.triggered)).all()


def test_mix_thresholds_rejects_empty_list_and_bad_mode():
    with pytest.raises(ThresholdError):
        mix_thresholds([])
    df = _oscillating_df()
    a = cross_level(df, "rsi", 14, 50, direction="above")
    with pytest.raises(ThresholdError):
        mix_thresholds([a], mode="xor")


def test_preview_signal_reports_no_triggers_warning():
    df = _oscillating_df()
    sig = in_range(df, "rsi", 14, 0, 0.001)  # essentially impossible
    preview = preview_signal(df, sig)
    assert preview.n_triggers == 0
    assert preview.warnings
    assert "Signal:" in preview.render_summary()
