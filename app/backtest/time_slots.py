"""Blocked entry slots: refuse NEW entries in specific weekday/hour buckets.

This is what lets the optimizer act on the Full Pipeline's "worst day/time" findings
(e.g. "Thursday 11:00 loses the most") instead of only printing them.

Clock: the DATA's own wall clock (the same clock the Full Pipeline heatmap uses to bucket trades by
entry time), checked against the bar the trade actually FILLS on (so it matches the report exactly).

Text format (stored in RiskConfig.blocked_entry_slots), tokens separated by newline, comma or ';':
    THU 11          Thursday, the 11:00-11:59 hour
    THU 11-13       Thursday hours 11, 12 and 13 (inclusive)
    FRI *           all of Friday
    * 15            15:00-15:59 every day
Days: MON TUE WED THU FRI SAT SUN (or full names, or 0-6 with Mon=0). Hours: 0-23.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

DAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
_DAY_LOOKUP = {}
for _i, _n in enumerate(DAY_NAMES):
    _DAY_LOOKUP[_n] = _i
    _DAY_LOOKUP[_n.title()] = _i
for _i, _n in enumerate(["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]):
    _DAY_LOOKUP[_n] = _i


def _parse_day(tok: str):
    t = tok.strip().upper()
    if t in ("*", "ALL", "ANY", "EVERY"):
        return None
    if t in _DAY_LOOKUP:
        return _DAY_LOOKUP[t]
    if t.isdigit() and 0 <= int(t) <= 6:
        return int(t)
    raise ValueError(f"unknown weekday '{tok}'")


def _parse_hours(tok: str):
    t = tok.strip()
    if t in ("*", ""):
        return None
    t = t.split(":")[0] if re.fullmatch(r"\d{1,2}:\d{2}", t) else t
    m = re.fullmatch(r"(\d{1,2})\s*-\s*(\d{1,2})", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
    elif t.isdigit():
        a = b = int(t)
    else:
        raise ValueError(f"unknown hour '{tok}'")
    if not (0 <= a <= 23 and 0 <= b <= 23 and a <= b):
        raise ValueError(f"hour out of range '{tok}'")
    return (a, b)


def parse_slots(raw) -> list[tuple]:
    """-> list of (weekday|None, (hour_lo, hour_hi)|None). Raises ValueError on a malformed token."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        raw = "\n".join(str(x) for x in raw)
    out = []
    for piece in re.split(r"[\n,;]+", str(raw)):
        piece = piece.strip()
        if not piece or piece.startswith("#"):
            continue
        parts = piece.split(None, 1)
        if len(parts) == 1:
            parts = [parts[0], "*"]
        day, hrs = _parse_day(parts[0]), _parse_hours(parts[1])
        if day is None and hrs is None:
            raise ValueError("a slot cannot block everything ('* *')")
        out.append((day, hrs))
    return out


def format_slots(slots: Iterable[tuple]) -> str:
    lines = []
    for day, hrs in slots:
        d = "*" if day is None else DAY_NAMES[day]
        if hrs is None:
            h = "*"
        elif hrs[0] == hrs[1]:
            h = f"{hrs[0]:02d}"
        else:
            h = f"{hrs[0]:02d}-{hrs[1]:02d}"
        lines.append(f"{d} {h}")
    return "\n".join(lines)


def merge_slot_text(*texts) -> str:
    """Union of several slot texts, de-duplicated, stable order."""
    seen, out = set(), []
    for t in texts:
        for s in parse_slots(t):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return format_slots(out)


def slot_label(day, hrs) -> str:
    d = "every day" if day is None else DAY_NAMES[day].title()
    if hrs is None:
        return d
    h = f"{hrs[0]:02d}:00" if hrs[0] == hrs[1] else f"{hrs[0]:02d}:00-{hrs[1]:02d}:59"
    return f"{d} {h}"


def slot_mask(weekday: np.ndarray, hour: np.ndarray, slots: list[tuple]) -> np.ndarray:
    """Boolean mask over bars: True = a trade filling on this bar is refused."""
    mask = np.zeros(len(weekday), dtype=bool)
    for day, hrs in slots:
        m = np.ones(len(weekday), dtype=bool)
        if day is not None:
            m &= weekday == day
        if hrs is not None:
            m &= (hour >= hrs[0]) & (hour <= hrs[1])
        mask |= m
    return mask


def slot_mask_for_timestamps(timestamps, raw) -> np.ndarray:
    """Convenience: timestamps = a pandas Series/Index of the data's own (possibly tz-aware) clock."""
    import pandas as pd
    slots = parse_slots(raw)
    idx = pd.DatetimeIndex(timestamps)
    return slot_mask(idx.dayofweek.to_numpy(), idx.hour.to_numpy(), slots)
