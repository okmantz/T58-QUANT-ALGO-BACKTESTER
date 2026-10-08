"""Back-adjusted continuous futures contracts.

An unadjusted 'ES1!' style series jumps by the roll spread (calendar spread
between the expiring and the next contract) at every roll. Those jumps are
not tradable: a strategy can book hundreds of points of fake profit or loss
on a roll bar. This module finds likely roll gaps and removes them with the
standard difference (Panama) back-adjustment: every bar BEFORE a roll is
shifted by the gap, so the most recent prices stay real and older prices are
shifted. Differences, not ratios, are used so point-based stops keep their
dollar meaning; the shifted history can drift from real price levels, so use
the adjusted frame for backtesting and keep the original for display.

Detection is deliberately conservative and reports what it did; nothing is
changed silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RollEvent:
    index: int
    timestamp: pd.Timestamp
    gap: float            # open[i] - close[i-1] that was removed


def detect_rolls(
    df: pd.DataFrame,
    tr_multiple: float = 3.0,
    min_gap_hours: float = 2.0,
    roll_days: range | None = range(6, 17),
    roll_months: tuple[int, ...] | None = None,
) -> list[RollEvent]:
    """Gaps at a session break that are `tr_multiple` x the median true range
    and fall in the roll week of the month (optionally only roll_months, e.g.
    (3, 6, 9, 12) for equity index futures). roll_days=None disables the
    calendar filter (then only size and the session break decide)."""
    if len(df) < 3:
        return []
    ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]), axis=0)
    med_tr = float(np.nanmedian(tr))
    if not np.isfinite(med_tr) or med_tr <= 0:
        return []
    dt_h = ts.diff().dt.total_seconds().to_numpy() / 3600.0
    out: list[RollEvent] = []
    for i in range(1, len(df)):
        gap = o[i] - c[i - 1]
        if abs(gap) <= tr_multiple * med_tr or not (dt_h[i] >= min_gap_hours):
            continue
        day, month = ts.iloc[i].day, ts.iloc[i].month
        if roll_days is not None and day not in roll_days:
            continue
        if roll_months is not None and month not in roll_months:
            continue
        out.append(RollEvent(i, ts.iloc[i], float(gap)))
    return out


def back_adjust(df: pd.DataFrame, rolls: list[RollEvent] | None = None, **detect_kwargs) -> tuple[pd.DataFrame, list[RollEvent]]:
    """Returns (adjusted_df, rolls). Bars before each roll are shifted by the
    gap (cumulatively), the latest segment is untouched. Adds a boolean
    `roll_bar` column and `adj_offset` (the total shift applied to that bar)."""
    rolls = detect_rolls(df, **detect_kwargs) if rolls is None else rolls
    out = df.copy().reset_index(drop=True)
    offset = np.zeros(len(out))
    for r in sorted(rolls, key=lambda r: r.index):
        offset[: r.index] += r.gap
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].to_numpy(float) + offset
    out["adj_offset"] = offset
    out["roll_bar"] = False
    for r in rolls:
        out.loc[r.index, "roll_bar"] = True
    return out, rolls


def describe(rolls: list[RollEvent]) -> str:
    if not rolls:
        return "No contract-roll gaps detected."
    big = max(rolls, key=lambda r: abs(r.gap))
    return (f"{len(rolls)} roll gap(s) removed by difference back-adjustment; largest {big.gap:+.2f} on {big.timestamp}.")
