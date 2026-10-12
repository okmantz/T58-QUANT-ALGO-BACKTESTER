"""Futures contract calendar for ES / NQ (quarterly) and GC (bi-monthly).

Why (2026-10 audit, weak point 5): continuous-contract files splice the next contract in at a
roll date. If the splice happens mid-session and nobody knows when it *should* have happened,
the roll spread shows up as a fake trading gain or loss. ``app.data.continuous_contract`` finds
likely splices from the price series alone; this module supplies the CALENDAR side so the two
can be cross-checked:

    expected_roll_windows(symbol, start, end)     -> where a roll SHOULD be
    audit_rolls(df, symbol)                        -> detected jumps vs expected windows:
                                                      unexplained jumps, missing rolls, mid-session splices

Dates here are rule-based approximations, NOT an exchange feed:
  * ES / NQ (CME equity index): expire the 3rd Friday of Mar/Jun/Sep/Dec; volume rolls ~8
    calendar days earlier (the Thursday of the week before expiry week).
  * GC (COMEX gold): active months Feb/Apr/Jun/Aug/Oct/Dec. Last trade is the 3rd-last business
    day of the delivery month; first notice is the last business day of the month before. Volume
    rolls ~1-3 business days before first notice -> modelled as a window ending at first notice.
Verify against the CME contract calendar before relying on a specific date.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

EQUITY_INDEX_SYMBOLS = ("ES", "NQ", "MES", "MNQ", "YM", "RTY", "M2K")
GOLD_SYMBOLS = ("GC", "MGC")
_EQ_MONTHS = (3, 6, 9, 12)
_GC_MONTHS = (2, 4, 6, 8, 10, 12)


def _norm(symbol: str) -> str:
    s = (symbol or "").upper().strip().rstrip("!").rstrip("1").replace("/", "")
    return s


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(days=14)


def _last_business_day(year: int, month: int) -> date:
    d = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _business_days_back(d: date, n: int) -> date:
    while n > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


@dataclass
class RollWindow:
    contract_month: str       # e.g. "2025-03"
    expiry: date              # last trade / expiry (approx.)
    start: date               # earliest plausible roll
    end: date                 # latest plausible roll

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


def expected_roll_windows(symbol: str, start, end) -> list[RollWindow]:
    """Roll windows for contracts expiring between ``start`` and ``end`` (+/- the window)."""
    sym = _norm(symbol)
    s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    out: list[RollWindow] = []
    for year in range(s.year - 1, e.year + 2):
        if sym in EQUITY_INDEX_SYMBOLS:
            for m in _EQ_MONTHS:
                exp = third_friday(year, m)
                out.append(RollWindow(f"{year}-{m:02d}", exp, exp - timedelta(days=12), exp - timedelta(days=5)))
        elif sym in GOLD_SYMBOLS:
            for m in _GC_MONTHS:
                fnd = _last_business_day(year if m > 1 else year - 1, m - 1 if m > 1 else 12)
                last_trade = _business_days_back(_last_business_day(year, m), 2)
                out.append(RollWindow(f"{year}-{m:02d}", last_trade, _business_days_back(fnd, 4), fnd))
        else:
            raise ValueError(f"No contract calendar for {symbol!r}; supported: {EQUITY_INDEX_SYMBOLS + GOLD_SYMBOLS}")
    lo, hi = s - timedelta(days=45), e + timedelta(days=45)
    return sorted((w for w in out if lo <= w.expiry <= hi), key=lambda w: w.expiry)


@dataclass
class RollAudit:
    symbol: str
    n_expected: int
    n_detected_jumps: int
    unexplained_jumps: list = field(default_factory=list)   # big jumps NOT near an expected roll
    mid_session_splices: list = field(default_factory=list) # jumps near a roll that happened with no session break
    missing_rolls: list = field(default_factory=list)       # expected windows (inside the data) with no jump
    notes: list = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.unexplained_jumps or self.mid_session_splices)

    def render(self) -> str:
        lines = [f"CONTRACT ROLL AUDIT {self.symbol}: expected rolls {self.n_expected}, big jumps {self.n_detected_jumps}, "
                 f"unexplained {len(self.unexplained_jumps)}, mid-session splices {len(self.mid_session_splices)}, "
                 f"expected-but-absent {len(self.missing_rolls)}  -> {'CLEAN' if self.clean else 'NEEDS ATTENTION'}"]
        for k, items in (("unexplained", self.unexplained_jumps), ("mid-session splice", self.mid_session_splices), ("no jump in window", self.missing_rolls)):
            for it in items[:10]:
                lines.append(f"  {k}: {it}")
        lines += [f"  NOTE: {n}" for n in self.notes]
        return "\n".join(lines)


def audit_rolls(df: pd.DataFrame, symbol: str, *, jump_k: float = 8.0, session_break_minutes: float = 60.0) -> RollAudit:
    """Compare close-to-open jumps in a (continuous) price frame with the expected roll calendar.

    A "big jump" is |open[i] - close[i-1]| >= jump_k x the median absolute bar-to-bar close change.
    A jump inside an expected window with NO session break before it is a mid-session splice:
    the file switches contract in the middle of a session, so the spread is booked as P&L.
    Needs columns timestamp, open, close."""
    d = df[["timestamp", "open", "close"]].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    if len(d) < 50:
        return RollAudit(symbol, 0, 0, notes=["too few bars to audit"])
    gap = (d["open"] - d["close"].shift(1)).to_numpy()
    step = d["close"].diff().abs().to_numpy()
    scale = float(np.nanmedian(step[step > 0])) if np.any(step > 0) else 0.0
    thr = jump_k * scale if scale > 0 else np.inf
    dt_min = d["timestamp"].diff().dt.total_seconds().to_numpy() / 60.0
    idx = np.where(np.abs(np.nan_to_num(gap)) >= thr)[0]
    windows = expected_roll_windows(symbol, d["timestamp"].iloc[0], d["timestamp"].iloc[-1])
    audit = RollAudit(symbol, len(windows), len(idx))
    hit = set()
    bar_min = float(np.nanmedian(dt_min[1:])) if len(dt_min) > 1 else 1.0
    for i in idx:
        ts = d["timestamp"].iloc[i]
        day = ts.date()
        w = next((w for w in windows if w.start - timedelta(days=2) <= day <= w.end + timedelta(days=2)), None)
        desc = f"{ts} gap {gap[i]:+.2f}"
        if w is None:
            audit.unexplained_jumps.append(desc + " (not near any expected roll window)")
            continue
        hit.add(w.contract_month)
        if dt_min[i] <= max(session_break_minutes, 3 * bar_min):
            audit.mid_session_splices.append(desc + f" (window {w.contract_month}; {dt_min[i]:.0f} min after previous bar -> spliced mid-session)")
    first, last = d["timestamp"].iloc[0].date(), d["timestamp"].iloc[-1].date()
    for w in windows:
        if first <= w.start and w.end <= last and w.contract_month not in hit:
            audit.missing_rolls.append(f"{w.contract_month}: no jump between {w.start} and {w.end} (smooth/back-adjusted, or roll not in the file)")
    audit.notes.append("Calendar windows are rule-based approximations; verify against the CME calendar. A smooth (back-adjusted) series legitimately shows no jumps.")
    return audit
