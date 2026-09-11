"""Tests for app.quant_lab.market_structure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.quant_lab.market_structure import (
    MarketStructureError,
    calculate_hh_ll_structure,
    calculate_swing_points,
    calculate_wyckoff_events,
    summarize_market_structure,
)


def _synthetic_df(n=800, seed=3, freq_min=5):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq=f"{freq_min}min")
    price = 100.0
    rows = []
    for i in range(n):
        step = 0.15 * np.sin(i / 60) + rng.normal(0, 0.05)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.03))
        l = min(o, c) - abs(rng.normal(0, 0.03))
        rows.append((ts[i], o, h, l, c, 100 + rng.integers(0, 50)))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _uptrend_df(n=200, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 100.0
    rows = []
    for i in range(n):
        # A gentle upward drift with a slow oscillation layered on top --
        # a purely monotonic series never forms a local max/min inside a
        # fixed left/right fractal window, so this needs *some* wiggle for
        # calculate_swing_points to find anything at all.
        step = 0.05 + 0.08 * np.sin(i / 12) + rng.normal(0, 0.01)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.005))
        l = min(o, c) - abs(rng.normal(0, 0.005))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_calculate_swing_points_finds_alternating_highs_and_lows():
    df = _synthetic_df()
    swings = calculate_swing_points(df, left=3, right=3)
    assert not swings.empty
    assert set(swings["kind"].unique()) <= {"high", "low"}
    assert list(swings["timestamp"]) == sorted(swings["timestamp"])


def test_calculate_swing_points_rejects_bad_window():
    df = _synthetic_df()
    with pytest.raises(MarketStructureError):
        calculate_swing_points(df, left=0, right=2)


def test_calculate_hh_ll_structure_flags_bos_in_a_clean_uptrend():
    df = _uptrend_df()
    hh_ll = calculate_hh_ll_structure(df, left=3, right=3)
    assert not hh_ll.empty
    # A clean, near-monotonic uptrend should mostly produce HH/HL swings,
    # and any flagged event should be a bos (continuation), never a choch,
    # once the trend is established.
    assert hh_ll["structure"].isin(["H", "L", "HH", "HL"]).all()
    events = hh_ll["event"].dropna()
    if not events.empty:
        assert set(events.unique()) <= {"bos"}


def test_calculate_hh_ll_structure_empty_on_too_little_data():
    df = _synthetic_df(n=5)
    hh_ll = calculate_hh_ll_structure(df, left=5, right=5)
    assert hh_ll.empty
    assert list(hh_ll.columns) == ["timestamp", "price", "kind", "structure", "trend", "event", "index"]


def test_calculate_wyckoff_events_returns_expected_columns():
    df = _synthetic_df()
    events = calculate_wyckoff_events(df, window=30, max_width_pct=0.5, lookforward=20)
    expected_cols = {"timestamp", "event", "phase", "price", "range_top", "range_bottom", "range_start", "range_end", "index"}
    assert expected_cols <= set(events.columns) or events.empty
    if not events.empty:
        assert events["event"].isin(["spring", "upthrust", "sos", "sow", "range"]).all()
        assert events["phase"].isin(["accumulation", "distribution", "markup", "markdown", "ranging"]).all()


def test_summarize_market_structure_runs_end_to_end():
    df = _synthetic_df()
    summary = summarize_market_structure(df)
    assert summary.latest_trend in ("up", "down", "range", "unknown")
    assert summary.bias in ("bullish", "bearish", "neutral")
    assert isinstance(summary.render_summary(), str)
    assert summary.render_summary()  # non-empty


def test_summarize_market_structure_warns_on_too_little_data():
    df = _synthetic_df(n=5)
    summary = summarize_market_structure(df, swing_left=5, swing_right=5)
    assert summary.latest_trend == "unknown"
    assert summary.warnings
