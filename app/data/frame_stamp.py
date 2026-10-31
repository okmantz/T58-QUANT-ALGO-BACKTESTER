"""Stamp every loaded price frame with what was done to it (2026-10-31 audit, section 6).

The audit found that nothing in a Full Pipeline report said whether the NQ frame it ran on was back-adjusted: the raw vendor file carries
14 fake overnight jumps (quarterly contract splices, all upward, +76 to +297 points) that contaminate every gap and overnight feature.
``frame_stamp`` measures it from the frame itself and ``frame_stamp_line`` prints it in a report header:

    adjusted (yes/no), roll count, residual un-adjusted splices / interleaves still in the frame, bars, span and the data fingerprint
    (the same fingerprint the trial ledger uses, ``app.discovery_engine.provenance.data_fingerprint``).

``assert_no_residual_rolls`` is the hard check the pipeline data gate and the test-suite use: a frame that still carries contract splices
is not fit for a gap or overnight result.
"""
from __future__ import annotations

import pandas as pd

from app.data.continuous_contract import detect_splices_and_interleaves

_PRICE_COLS = {"timestamp", "open", "high", "low", "close"}


class ResidualRollError(ValueError):
    """The frame still carries contract-roll splices that back-adjustment should have removed."""


def _roll_count(df: pd.DataFrame) -> int:
    """Rolls recorded by the importer: ``roll_bar`` flags, else non-trivial steps in ``adj_offset``, else the attrs text."""
    if "roll_bar" in df.columns:
        try:
            return int(pd.Series(df["roll_bar"]).fillna(False).astype(bool).sum())
        except Exception:  # noqa: BLE001
            pass
    if "adj_offset" in df.columns:
        off = pd.to_numeric(df["adj_offset"], errors="coerce").fillna(0.0).to_numpy()
        if len(off) > 1:
            return int((abs(off[1:] - off[:-1]) > 1e-9).sum())
    return 0


def residual_rolls(df: pd.DataFrame | None) -> tuple[int, int] | None:
    """(isolated splices, interleaved flips) still detectable in ``df``; None when it cannot be measured (too short / missing columns)."""
    if df is None or len(df) < 1000 or not _PRICE_COLS <= set(df.columns):
        return None
    rep = detect_splices_and_interleaves(df)
    return len(rep.splice_rolls), len(rep.interleave_events)


def frame_stamp(df: pd.DataFrame | None, label: str | None = None) -> dict:
    """What the frame is: adjusted or not, how many rolls were removed, how many remain, size, span and fingerprint."""
    from app.discovery_engine.provenance import data_fingerprint
    if df is None or len(df) == 0:
        return {"label": label, "bars": 0, "adjusted": False, "roll_count": 0, "residual_splices": None, "residual_interleaves": None,
                "fingerprint": "empty", "start": None, "end": None}
    attrs = getattr(df, "attrs", {}) or {}
    rolls = _roll_count(df)
    adjusted = bool(attrs.get("roll_adjusted")) or rolls > 0
    res = residual_rolls(df)
    start = end = None
    if "timestamp" in df.columns:
        start, end = str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1])
    return {
        "label": label, "bars": int(len(df)), "adjusted": adjusted, "roll_count": rolls,
        "residual_splices": None if res is None else res[0], "residual_interleaves": None if res is None else res[1],
        "fingerprint": data_fingerprint(df), "start": start, "end": end,
        "adjustment_note": str(attrs.get("roll_adjustment") or ""),
    }


def frame_stamp_line(stamp: dict | None) -> str:
    """One line for a report header. Never raises."""
    if not stamp:
        return ""
    try:
        res_s, res_i = stamp.get("residual_splices"), stamp.get("residual_interleaves")
        if res_s is None:
            resid = "residual rolls not measurable"
        elif res_s + (res_i or 0) == 0:
            resid = "0 residual contract splices"
        else:
            resid = f"WARNING {res_s} un-adjusted contract splice(s) + {res_i or 0} interleaved flip(s) remain in this frame"
        adj = "back-adjusted" if stamp.get("adjusted") else "NOT back-adjusted"
        return (f"Data frame: {stamp.get('label') or 'unnamed'}; {stamp.get('bars', 0):,} bars; {adj} ({stamp.get('roll_count', 0)} roll(s) "
                f"removed); {resid}; fingerprint {stamp.get('fingerprint')}.")
    except Exception:  # noqa: BLE001
        return ""


def assert_no_residual_rolls(df: pd.DataFrame, label: str | None = None) -> dict:
    """Raise ``ResidualRollError`` if ``df`` still carries contract splices or interleaved flips; return its stamp otherwise."""
    st = frame_stamp(df, label)
    n = (st.get("residual_splices") or 0) + (st.get("residual_interleaves") or 0)
    if n:
        raise ResidualRollError(frame_stamp_line(st))
    return st
