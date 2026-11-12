"""Continuous-contract roll handling (2026-11-11).

A spliced continuous contract jumps by the calendar spread at every roll. That jump is not tradable P&L, yet it looks exactly like an
overnight gap, so it leaks into any gap filter / gap-fade / opening-range logic ("gap-aligned ORB") and into stop-gap fills.

Policy used by the scanner (``roll_policy``):
  * ``"exclude"`` (default): detect the roll sessions and skip every entry on them (they are counted as skipped signals);
  * ``"adjust"``: difference-back-adjust the history so the roll gap is removed from every price before the roll (use when the
    series is UNadjusted); the same sessions are NOT excluded because the gap no longer exists;
  * ``"none"``: leave the data alone (only correct for a series already back-adjusted, or one that is a single contract).

Detection needs no vendor roll calendar: a session break whose open-vs-prior-close gap is an outlier (``k`` x the median absolute break
gap) AND that falls within ``window_days`` of an expiry for that product's roll cycle is a roll. The expiry cycle is the quarterly
(Mar/Jun/Sep/Dec) third Friday for equity index futures, every month for CL, and the even months (Feb/Apr/Jun/Aug/Oct/Dec) for GC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLL_MONTHS = {
    "NQ": (3, 6, 9, 12), "ES": (3, 6, 9, 12), "RTY": (3, 6, 9, 12), "YM": (3, 6, 9, 12),
    "MNQ": (3, 6, 9, 12), "MES": (3, 6, 9, 12), "M2K": (3, 6, 9, 12), "MYM": (3, 6, 9, 12),
    "GC": (2, 4, 6, 8, 10, 12), "MGC": (2, 4, 6, 8, 10, 12),
    "CL": tuple(range(1, 13)), "MCL": tuple(range(1, 13)),
}
ROLL_POLICIES = ("exclude", "adjust", "none")


def third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    first_fri = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
    return first_fri + pd.Timedelta(days=14)


def _near_expiry(ts: pd.Timestamp, months, window_days: int) -> bool:
    for dy, mo in ((ts.year, ts.month), (ts.year + (ts.month == 12), ts.month % 12 + 1), (ts.year - (ts.month == 1), (ts.month - 2) % 12 + 1)):
        if mo in months and abs((ts.normalize() - third_friday(int(dy), int(mo))).days) <= window_days:
            return True
    return False


def detect_roll_bars(bars: pd.DataFrame, instrument: str, *, k: float = 4.0, window_days: int = 9, break_minutes: float = 30.0) -> np.ndarray:
    """Indices (into ``bars``) of the FIRST bar after each detected roll gap. ``bars`` needs timestamp/open/close."""
    n = len(bars)
    if n < 3:
        return np.zeros(0, dtype=np.int64)
    ts = pd.to_datetime(bars["timestamp"]).reset_index(drop=True)
    gap_t = ts.diff().dt.total_seconds().to_numpy() / 60.0
    brk = np.flatnonzero(np.nan_to_num(gap_t, nan=0.0) >= break_minutes)
    if brk.size == 0:
        return np.zeros(0, dtype=np.int64)
    o = bars["open"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    gaps = o[brk] - c[brk - 1]
    med = float(np.median(np.abs(gaps))) or 1e-12
    months = ROLL_MONTHS.get(str(instrument).upper(), (3, 6, 9, 12))
    out = []
    for j, g in zip(brk, gaps):
        if abs(g) > k * med and _near_expiry(pd.Timestamp(ts.iloc[j]), months, window_days):
            out.append(int(j))
    return np.asarray(out, dtype=np.int64)


def back_adjust(bars: pd.DataFrame, roll_bars: np.ndarray) -> pd.DataFrame:
    """Difference back-adjustment: shift every price BEFORE each roll by that roll's gap so the series is continuous."""
    out = bars.copy()
    if len(roll_bars) == 0:
        return out
    px = ["open", "high", "low", "close"]
    arr = {k: out[k].to_numpy(float).copy() for k in px}
    for j in sorted(map(int, roll_bars), reverse=True):
        gap = arr["open"][j] - arr["close"][j - 1]
        for k in px:
            arr[k][:j] = arr[k][:j] + gap
    for k in px:
        out[k] = arr[k]
    return out


def session_ordinals(bars: pd.DataFrame, session_roll_ny_min: int = 18 * 60) -> np.ndarray:
    """Futures-session ordinal of every bar (a session starts 18:00 New York the evening before its calendar date)."""
    ts = pd.to_datetime(bars["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    ny = ts.dt.tz_convert("America/New_York")
    mins = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
    dates = np.asarray([d.toordinal() for d in ny.dt.date], dtype=np.int64)
    return dates + (mins >= session_roll_ny_min).astype(np.int64)


def roll_session_dates(bars: pd.DataFrame, roll_bars: np.ndarray, session_roll_ny_min: int = 18 * 60) -> set:
    """Futures-session ordinals that contain a roll gap bar."""
    if len(roll_bars) == 0:
        return set()
    sess = session_ordinals(bars, session_roll_ny_min)
    return {int(sess[j]) for j in roll_bars}
