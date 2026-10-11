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

AUDIT FIX (Oct 2026, data-layer review of the shipped NQ/ES/GC 1-minute
vendor files). The original detector (`detect_rolls` / `back_adjust` below,
kept unchanged for backward compatibility) looked ONLY for a large gap after a
session break of >= 2 hours. On the real vendor series that is exactly
backwards: the vendor splices the next contract in at the first bar of a new
date, with NO session break (NQ +296.75 pts on 2024-12-17 00:00 UTC, ES
+70.75 on 2024-12-16, GC +25..+39 each two months), so every real roll was
MISSED -- while ~35-45 ordinary Sunday-open / holiday gaps per instrument
(real market moves, e.g. the 2020-03-08 oil gap or the 2025-06-15 weekend
strike gap) were wrongly classed as rolls and erased from history. GC also
carries windows where the vendor alternated bar-by-bar between two contract
months (alternating +/-33 pt flips on 2026-05-25/26), which is pure phantom
volatility. `back_adjust_continuous` below fixes both:

  * isolated SPLICE rolls: a jump >= `splice_k` x the robust scale of normal
    contiguous 1-bar jumps on the FIRST bar of a new date with no real time
    break; at most one per calendar month; optionally restricted to roll
    months / days of the month;
  * INTERLEAVED-CONTRACT "teleport" bars: a bar that jumps by the calendar
    spread and jumps back by the same amount on the next bar WITHOUT travelling
    through the levels (tiny own range). Each jump is treated as a roll event so
    the cumulative back-adjustment cancels the flip exactly. Real news spikes
    sweep through the levels (large bar range) and are never touched;
  * ordinary session-break gaps are NEVER touched (they are real price
    moves a position held across the break would really have experienced).
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


# ---------------------------------------------------------------------------
# Splice / interleave detection for vendor "continuous" futures (audit fix)
# ---------------------------------------------------------------------------

@dataclass
class ContinuousCleanReport:
    splice_rolls: list            # list[RollEvent] -- isolated contract splices
    interleave_events: list       # list[RollEvent] -- individual flips inside interleaved windows
    interleave_windows: list      # list[(start_ts, end_ts, n_flips, median_abs_flip)]
    ambiguous_weekend_gaps: list  # large weekend/holiday-open gaps in a roll window that were NOT adjusted
    jump_scale: float

    @property
    def rolls(self) -> list:
        return sorted(self.splice_rolls + self.interleave_events, key=lambda r: r.index)

    def describe(self) -> str:
        if not self.rolls:
            return "No contract-roll splices or interleaved-contract bars detected."
        parts = []
        if self.splice_rolls:
            big = max(self.splice_rolls, key=lambda r: abs(r.gap))
            parts.append(f"{len(self.splice_rolls)} contract splice(s) back-adjusted (largest {big.gap:+.2f} on {big.timestamp})")
        if self.interleave_windows:
            parts.append(f"{len(self.interleave_windows)} interleaved-contract window(s) repaired "
                         f"({len(self.interleave_events)} contract-flip jump(s), e.g. {self.interleave_windows[0][0]})")
        msg = "; ".join(parts) + "."
        if self.ambiguous_weekend_gaps:
            msg += (f" {len(self.ambiguous_weekend_gaps)} large weekend/holiday-open gap(s) inside a roll window were "
                    "left unadjusted (real price moves cannot be told apart from a weekend splice).")
        return msg


def _arrays(df: pd.DataFrame):
    ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    jump = np.full(len(df), np.nan)
    jump[1:] = o[1:] - c[:-1]
    t_ns = ts.dt.tz_localize(None).to_numpy("datetime64[ns]") if ts.dt.tz is not None else ts.to_numpy("datetime64[ns]")
    dt_min = np.full(len(df), np.inf)
    dt_min[1:] = (t_ns[1:] - t_ns[:-1]).astype("timedelta64[s]").astype(float) / 60.0
    day = t_ns.astype("datetime64[D]")
    day_change = np.zeros(len(df), dtype=bool)
    day_change[1:] = day[1:] != day[:-1]
    return ts, jump, dt_min, day_change


def detect_splices_and_interleaves(
    df: pd.DataFrame,
    *,
    splice_k: float = 4.0,
    interleave_k: float = 5.0,
    max_break_minutes: float = 90.0,
    flip_window_bars: int = 120,
    flip_size_tol: float = 0.25,
    flip_range_frac: float = 0.35,
    roll_months: tuple[int, ...] | None = None,
    roll_days: range | None = None,
    one_per_month: bool = True,
) -> ContinuousCleanReport:
    """See the module docstring. `roll_months`/`roll_days` (e.g. (3,6,9,12) and
    range(4, 22) for equity-index futures) restrict where an isolated splice may
    be accepted; None accepts any month/day (use for metals/energies whose roll
    months differ by contract)."""
    empty = ContinuousCleanReport([], [], [], [], 0.0)
    if df is None or len(df) < 50:
        return empty
    ts, jump, dt_min, day_change = _arrays(df)
    contiguous = dt_min <= max_break_minutes
    ordinary = contiguous & ~day_change & np.isfinite(jump)
    if ordinary.sum() < 100:
        return empty
    scale = float(np.quantile(np.abs(jump[ordinary]), 0.999))
    # Relative floor: a vendor whose bars always open exactly at the previous close
    # has a zero jump scale, which would make ANY nonzero jump look infinite.
    scale = max(scale, 1e-4 * float(np.nanmedian(df["close"].to_numpy(float))))
    if not np.isfinite(scale) or scale <= 0:
        return empty
    absj = np.abs(np.where(np.isfinite(jump), jump, 0.0))

    # -- interleaved-contract "teleport" bars ---------------------------------
    # The vendor sometimes prints a bar from the OTHER contract month between
    # two bars of the active one: the bar jumps by the calendar spread (~33 pts
    # in GC) relative to the previous close, and jumps back by the same amount
    # on the next bar. Signature (all required): jump IN and jump OUT are both
    # >= interleave_k x scale, opposite in sign and equal in size (within
    # `flip_size_tol`), AND the bar itself did not travel (its high-low range is
    # a small fraction of the jump -- a real news spike sweeps through the
    # levels, so its bar range is at least comparable to the move). Real
    # whipsaws (CPI/NFP/FOMC) fail the range test and are never touched.
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    n = len(df)
    interleave_events: list[RollEvent] = []
    windows: list = []
    used = np.zeros(n, dtype=bool)
    idx = np.arange(1, n - 1)
    j_in = jump[idx]
    j_out = jump[idx + 1]
    ok = (
        contiguous[idx] & contiguous[idx + 1]
        & np.isfinite(j_in) & np.isfinite(j_out)
        & (np.abs(j_in) >= interleave_k * scale) & (np.abs(j_out) >= interleave_k * scale)
        & (np.sign(j_in) != np.sign(j_out))
        & (np.abs(np.abs(j_in) - np.abs(j_out)) <= flip_size_tol * np.maximum(np.abs(j_in), np.abs(j_out)))
        & ((high[idx] - low[idx]) <= flip_range_frac * np.abs(j_in))
    )
    spikes = idx[ok]
    if len(spikes):
        by_index: dict = {}   # overlapping spike pairs share a jump: register each bar's jump ONCE
        for i in spikes:
            i = int(i)
            for k_ in (i, i + 1):
                by_index.setdefault(k_, RollEvent(k_, ts.iloc[k_], float(jump[k_])))
            used[i] = used[i + 1] = True
        interleave_events = [by_index[k_] for k_ in sorted(by_index)]
        # group consecutive spikes (<= flip_window_bars apart) into reported windows
        grp_start = int(spikes[0]); prev = int(spikes[0]); cnt = 1
        mags = [abs(float(jump[prev]))]
        for i in spikes[1:]:
            i = int(i)
            if i - prev <= flip_window_bars:
                cnt += 1; mags.append(abs(float(jump[i])))
            else:
                windows.append((ts.iloc[grp_start], ts.iloc[prev + 1], cnt, float(np.median(mags))))
                grp_start = i; cnt = 1; mags = [abs(float(jump[i]))]
            prev = i
        windows.append((ts.iloc[grp_start], ts.iloc[prev + 1], cnt, float(np.median(mags))))

    # -- isolated splices: first bar of a new date, no real time break --------
    cand = np.flatnonzero(contiguous & day_change & np.isfinite(jump) & (absj >= splice_k * scale) & ~used)
    kept: dict = {}
    for i in cand:
        t = ts.iloc[int(i)]
        if roll_months is not None and t.month not in roll_months:
            continue
        if roll_days is not None and t.day not in roll_days:
            continue
        key = (t.year, t.month) if one_per_month else int(i)
        if key not in kept or absj[int(i)] > absj[kept[key]]:
            kept[key] = int(i)
    splices = [RollEvent(i, ts.iloc[i], float(jump[i])) for i in sorted(kept.values())]

    # -- informational: big weekend/holiday-open gaps in roll windows (NOT adjusted) --
    ambiguous: list = []
    if roll_months is not None:
        brk = np.flatnonzero((dt_min >= 120) & np.isfinite(jump) & (absj >= splice_k * scale * 4))
        for i in brk:
            t = ts.iloc[int(i)]
            if t.month in roll_months and (roll_days is None or t.day in roll_days):
                ambiguous.append((t, float(jump[int(i)])))
    return ContinuousCleanReport(splices, interleave_events, windows, ambiguous, scale)


def back_adjust_continuous(df: pd.DataFrame, report: ContinuousCleanReport | None = None, **detect_kwargs) -> tuple[pd.DataFrame, ContinuousCleanReport]:
    """Difference back-adjustment using detect_splices_and_interleaves. Same output
    columns as `back_adjust` (`adj_offset`, `roll_bar`) so downstream code is unchanged."""
    report = detect_splices_and_interleaves(df, **detect_kwargs) if report is None else report
    out, _ = back_adjust(df, rolls=report.rolls)
    return out, report
