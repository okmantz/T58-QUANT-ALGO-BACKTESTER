"""Regression test for the swing-sort-stability fix in
app/quant_lab/market_structure.py::calculate_swing_points.

Bug: the final sort_values("timestamp") used quicksort (not stable), so a bar
that is simultaneously a fractal high AND a fractal low produced its two swing
rows in nondeterministic order depending on frame length -- flipping downstream
bos/choch attribution between full and truncated runs (seen on MGC 15m).

Fix: deterministic tie-break (lows before highs) + kind="stable".
"""
import numpy as np
import pandas as pd

from app.quant_lab.market_structure import (
    _label_swing_structure,
    calculate_hh_ll_structure,
    calculate_swing_points,
)


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
