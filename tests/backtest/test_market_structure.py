"""Tests for app.quant_lab.market_structure. / Regression test for the swing-sort-stability fix in"""
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
from app.quant_lab.market_structure import (
    _label_swing_structure,
    calculate_hh_ll_structure,
    calculate_swing_points,
)


# ---- from test_market_structure.py ----

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


# ---- from test_market_structure_sort_stability.py ----

def _dual_swing_df(n=60, dual_bar=10):
    """Flat bars except one bar that is simultaneously a fractal high and low."""
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    high = np.full(n, 100.0)
    low = np.full(n, 99.0)
    high[dual_bar] = 110.0
    low[dual_bar] = 90.0
    mid = (high + low) / 2
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": mid,
            "high": high,
            "low": low,
            "close": mid,
            "volume": 100.0,
        }
    )


def _kinds_at(swings: pd.DataFrame, ts) -> list:
    return list(swings.loc[swings["timestamp"] == ts, "kind"])


def test_same_bar_pair_ordered_low_before_high():
    df = _dual_swing_df()
    swings = calculate_swing_points(df, left=2, right=2)
    dual_ts = df["timestamp"].iloc[10]
    # Both swings detected, deterministic order: low first, then high.
    assert _kinds_at(swings, dual_ts) == ["low", "high"]


def test_same_bar_pair_identical_across_frame_lengths():
    """The exact reported failure: full vs truncated runs must agree."""
    df = _dual_swing_df()
    full = calculate_swing_points(df, left=2, right=2)
    trunc30 = calculate_swing_points(df.iloc[:30], left=2, right=2)
    trunc45 = calculate_swing_points(df.iloc[:45], left=2, right=2)
    dual_ts = df["timestamp"].iloc[10]

    for frame in (trunc30, trunc45):
        # Same-bar pair order identical to the full run.
        assert _kinds_at(frame, dual_ts) == _kinds_at(full, dual_ts) == ["low", "high"]
        # Every timestamp in the safely-interior region classifies identically.
        # (Bars within `right` of the truncation edge are excluded -- the
        # fractal loop legitimately cannot see them there.)
        interior = df["timestamp"].iloc[:25]
        for ts in interior:
            assert _kinds_at(frame, ts) == _kinds_at(full, ts), f"mismatch at {ts}"


def test_labeled_structure_identical_across_frame_lengths():
    df = _dual_swing_df()
    full = _label_swing_structure(calculate_swing_points(df, left=2, right=2))
    trunc = _label_swing_structure(calculate_swing_points(df.iloc[:30], left=2, right=2))
    interior = df["timestamp"].iloc[:25]
    full_i = full[full["timestamp"].isin(interior)].reset_index(drop=True)
    trunc_i = trunc[trunc["timestamp"].isin(interior)].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_i, trunc_i)


def test_hh_ll_events_identical_on_dual_bar():
    df = _dual_swing_df()
    full = calculate_hh_ll_structure(df, left=2, right=2)
    trunc = calculate_hh_ll_structure(df.iloc[:30], left=2, right=2)
    dual_ts = df["timestamp"].iloc[10]
    full_ev = full.loc[full["timestamp"] == dual_ts].sort_values("kind")
    trunc_ev = trunc.loc[trunc["timestamp"] == dual_ts].sort_values("kind")
    pd.testing.assert_frame_equal(
        full_ev.reset_index(drop=True), trunc_ev.reset_index(drop=True)
    )
