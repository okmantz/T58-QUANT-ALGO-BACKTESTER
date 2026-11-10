"""Actionable data-health cards: what is wrong with this file, in plain words, and a one-click fix where one is safe.

Each card is a dict: {id, severity ("bad"|"warn"|"info"), title, detail, fix} where fix is None or {"id", "label"}.
apply_fix() returns a cleaned COPY of the frame; callers save it as a new file so the original is never overwritten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.continuous_contract import back_adjust_continuous, detect_splices_and_interleaves

FIX_ROLLS = "back_adjust"
FIX_DUPES = "drop_duplicates"
FIX_SORT = "sort_by_time"
FIX_BAD_BARS = "drop_bad_bars"
FIXES = (FIX_ROLLS, FIX_DUPES, FIX_SORT, FIX_BAD_BARS)
_PRICE = ("open", "high", "low", "close")


def _bad_bar_mask(df: pd.DataFrame) -> np.ndarray:
    o, h, l, c = (df[k].to_numpy(float) for k in _PRICE)
    return ~np.isfinite(o + h + l + c) | (np.minimum.reduce([o, h, l, c]) <= 0) | (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)


def build_cards(df: pd.DataFrame, name: str = "This file") -> list[dict]:
    cards: list[dict] = []
    ts = pd.to_datetime(df["timestamp"])
    if not ts.is_monotonic_increasing:
        cards.append({"id": "order", "severity": "bad", "title": "Bars are out of time order",
                      "detail": f"{name} has rows that are not in time order, which silently corrupts indicators and fills.",
                      "fix": {"id": FIX_SORT, "label": "Sort by time"}})
    dupes = int(ts.duplicated().sum())
    if dupes:
        cards.append({"id": "dupes", "severity": "bad", "title": f"{dupes:,} duplicate timestamp(s)",
                      "detail": "Two bars share the same time. The engine would treat them as two separate moments.",
                      "fix": {"id": FIX_DUPES, "label": "Keep the last bar for each time"}})
    bad = int(_bad_bar_mask(df).sum())
    if bad:
        cards.append({"id": "bad_bars", "severity": "bad", "title": f"{bad:,} impossible bar(s)",
                      "detail": "Bars with a high below the low, a close outside the range, a zero/negative price or a missing value.",
                      "fix": {"id": FIX_BAD_BARS, "label": "Remove those bars"}})
    if len(df) >= 50:
        try:
            ordered = df.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
            report = detect_splices_and_interleaves(ordered)
        except Exception:  # noqa: BLE001 -- a detector hiccup must not hide the other cards
            report = None
        if report is not None and report.rolls:
            points = float(sum(abs(r.gap) for r in report.rolls))
            cards.append({"id": "rolls", "severity": "warn",
                          "title": f"About {points:,.0f} points of contract-roll splice",
                          "detail": f"{report.describe()} Unadjusted, every one of these shows up as a fake overnight gap a strategy can 'trade'. "
                                    "Back-adjusting removes the gaps and keeps the recent prices true.",
                          "fix": {"id": FIX_ROLLS, "label": "Apply back-adjustment"}})
    if not cards:
        cards.append({"id": "clean", "severity": "info", "title": "No problems found",
                      "detail": f"{name} is in time order, has no duplicate or impossible bars and shows no contract-roll splices.",
                      "fix": None})
    return cards


def apply_fix(df: pd.DataFrame, fix_id: str) -> tuple[pd.DataFrame, str]:
    """(cleaned copy, one-line summary). Raises ValueError for an unknown fix id."""
    if fix_id == FIX_SORT:
        return df.sort_values("timestamp", kind="stable").reset_index(drop=True), "Sorted the bars by time."
    if fix_id == FIX_DUPES:
        out = df.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        return out, f"Removed {len(df) - len(out):,} duplicate bar(s)."
    if fix_id == FIX_BAD_BARS:
        out = df.loc[~_bad_bar_mask(df)].reset_index(drop=True)
        return out, f"Removed {len(df) - len(out):,} impossible bar(s)."
    if fix_id == FIX_ROLLS:
        ordered = df.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        out, report = back_adjust_continuous(ordered)
        return out.drop(columns=["roll_bar"], errors="ignore"), report.describe()
    raise ValueError(f"Unknown fix: {fix_id!r}")
