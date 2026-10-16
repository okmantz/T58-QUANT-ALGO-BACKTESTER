"""Tests for the ib_contraction_ratio operand added to app.strategy.manual
(today's Initial Balance range vs its own trailing average -- the
volatility-contraction filter behind IB/opening-range breakout systems)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.manual import ManualStrategy


def _session_df(daily_ib_ranges: list[float], bars_per_day: int = 8, ib_bars: int = 2) -> pd.DataFrame:
    """Builds one synthetic RTH session per entry in `daily_ib_ranges`.

    Each day runs 09:30..09:30+bars_per_day*15min in 15-minute bars. The
    first `ib_bars` bars (09:30-10:00) are the Initial Balance window and
    are constructed so (high - low) over exactly that window equals the
    requested day's IB range -- bar 0 sets the low, bar (ib_bars-1) sets
    the high, dead center of the day's base price so later bars can move
    around without ever exceeding the IB window's own high/low (which
    would change ITS range, not just the day's price action).
    """
    rows = []
    base_price = 1000.0
    for day_idx, ib_range in enumerate(daily_ib_ranges):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_idx)
        # Skip weekends so trading-day spacing stays intuitive, though the
        # implementation itself does not care about calendar days.
        day = day + pd.Timedelta(days=2 * (day_idx // 5))
        session_open = day + pd.Timedelta(hours=9, minutes=30)
        for bar in range(bars_per_day):
            ts = session_open + pd.Timedelta(minutes=15 * bar)
            if bar == 0:
                low = base_price - ib_range / 2
                high = low + 0.01
                close = low + 0.005
            elif bar == ib_bars - 1:
                high = base_price + ib_range / 2
                low = high - 0.01
                close = high - 0.005
            else:
                # Comfortably inside the IB window's own high/low so it
                # never redefines that day's IB range.
                low = base_price - 0.01
                high = base_price + 0.01
                close = base_price
            rows.append({"timestamp": ts, "open": close, "high": high, "low": low,
                         "close": close, "volume": 100.0})
    return pd.DataFrame(rows)


def test_ib_contraction_ratio_matches_hand_computed_value():
    # 10 prior days all with IB range 10.0, then one day with IB range 4.0.
    # Trailing 10-day average of the prior days is 10.0, so today's ratio
    # should be 4.0 / 10.0 == 0.4.
    ranges = [10.0] * 10 + [4.0]
    df = _session_df(ranges)
    strat = ManualStrategy({
        "entry_conditions": {
            "long": [{"left": {"type": "ib_contraction_ratio", "session_start": "09:30",
                                "session_end": "10:00", "lookback": 10},
                       "operator": ">", "right": {"type": "value", "value": -1}}],
            "long_connectors": [], "short": [], "short_connectors": [],
        },
        "exit_conditions": {"long": [], "short": []},
    })
    work = strat._build_indicators(df)
    ratio = strat._series_from_operand(
        work, {"type": "ib_contraction_ratio", "session_start": "09:30", "session_end": "10:00", "lookback": 10}
    )
    # Only bars strictly after that day's own 10:00 IB close carry ITS
    # freshly revealed ratio -- bars before 10:00 that same day still show
    # whatever the previous completed day last revealed (see the
    # look-ahead test below), so restrict the check to post-close bars.
    last_day = df["timestamp"].iloc[-1].normalize()
    after_close_today = (df["timestamp"].dt.normalize() == last_day) & (df["timestamp"].dt.time > pd.Timestamp("10:00").time())
    last_day_ratio = ratio[after_close_today].dropna().unique()
    assert len(last_day_ratio) == 1
    assert last_day_ratio[0] == pytest.approx(0.4, rel=1e-6)


def test_ib_contraction_ratio_is_nan_before_trailing_history_exists():
    # Only 3 prior days for a lookback of 10 -- still computable (rolling
    # mean uses min_periods=1), but the FIRST day in the dataset has no
    # prior day at all, so its ratio must be NaN (fail closed, not a
    # fabricated baseline of "compare to itself").
    ranges = [8.0, 6.0, 12.0]
    df = _session_df(ranges)
    strat = ManualStrategy({
        "entry_conditions": {"long": [], "long_connectors": [], "short": [], "short_connectors": []},
        "exit_conditions": {"long": [], "short": []},
    })
    work = strat._build_indicators(df)
    ratio = strat._series_from_operand(
        work, {"type": "ib_contraction_ratio", "session_start": "09:30", "session_end": "10:00", "lookback": 10}
    )
    first_day_mask = df["timestamp"] < (df["timestamp"].iloc[0] + pd.Timedelta(hours=2))
    assert ratio[first_day_mask].isna().all()


def test_ib_contraction_ratio_does_not_leak_the_still_forming_ib_window():
    # Within the IB window itself (before session_end), the ratio for
    # TODAY must not yet be revealed -- it can only reflect a prior,
    # already-completed day's value (or NaN on day 1).
    ranges = [10.0, 10.0, 4.0]
    df = _session_df(ranges, bars_per_day=8, ib_bars=2)
    strat = ManualStrategy({
        "entry_conditions": {"long": [], "long_connectors": [], "short": [], "short_connectors": []},
        "exit_conditions": {"long": [], "short": []},
    })
    work = strat._build_indicators(df)
    ratio = strat._series_from_operand(
        work, {"type": "ib_contraction_ratio", "session_start": "09:30", "session_end": "10:00", "lookback": 10}
    )
    # The very first bar of the third day (09:30, still inside the IB
    # window) must not already show that day's own 0.4 ratio.
    third_day_start = df["timestamp"].iloc[16]  # day index 2, bar 0 (8 bars/day)
    idx = df.index[df["timestamp"] == third_day_start][0]
    assert ratio.loc[idx] != pytest.approx(0.4, rel=1e-6)


def test_ib_contraction_condition_only_fires_on_the_contracted_day():
    # End-to-end through ManualStrategy.generate(): a long-only strategy
    # that requires ib_contraction_ratio <= 0.5 (today's IB is at most
    # half its trailing 10-day average) should only ever be eligible to
    # enter on the one day built to satisfy that.
    ranges = [10.0] * 10 + [4.0] + [10.0]
    df = _session_df(ranges)
    config = {
        "entry_conditions": {
            "long": [
                {"left": {"type": "ib_contraction_ratio", "session_start": "09:30",
                          "session_end": "10:00", "lookback": 10},
                 "operator": "<=", "right": {"type": "value", "value": 0.5}},
            ],
            "long_connectors": [],
            "short": [], "short_connectors": [],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": {"opposite_signal_exit": True},
    }
    strat = ManualStrategy(config)
    work = strat._build_indicators(df)
    long_mask = strat._combine_conditions(work, config["entry_conditions"]["long"],
                                           config["entry_conditions"]["long_connectors"])
    # Exclude bars still carrying the PRIOR day's revealed ratio forward
    # (ffill holds a day's value until that day's own IB closes) -- a real
    # strategy would normally pair this filter with an intraday time
    # window anyway; here we isolate the ratio's own correctness by only
    # looking at bars at/after each day's 10:00 IB close.
    after_close = df["timestamp"].dt.time > pd.Timestamp("10:00").time()
    eligible_days = df.loc[long_mask & after_close, "timestamp"].dt.normalize().unique()
    # Only the 11th day (index 10, the 4.0-range day) can ever satisfy the
    # <= 0.5 contraction filter; the 10.0-range days (ratio 1.0) cannot,
    # and the very first 10 days have no trailing history at all (NaN).
    contracted_day = df["timestamp"].iloc[0].normalize() + pd.Timedelta(days=10) + pd.Timedelta(days=2 * (10 // 5))
    assert set(eligible_days) == {contracted_day}
