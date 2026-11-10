"""
Economic-calendar news restrictions for the Evaluation Guardian.

Many prop firms bar opening (or holding) positions in a short window around high-impact releases. This module answers one
question from the scheduled-event calendar (``app.data.event_calendar``: FOMC, NFP, CPI, PCE): "is ``now`` inside a restricted
window, and what is the next event?". It reads the SCHEDULE only -- event times are public and known in advance -- and never a
realised print.

Calendar timestamps are US/Eastern wall-clock (tz-naive); ``now`` may be tz-aware (any zone) or naive, in which case it is taken
as US/Eastern. CPI and PCE only appear once ``scripts/update_event_calendar.py`` has been run with a FRED key; until then the
built-in FOMC and NFP schedule is what is enforced, and ``coverage`` says so, so the page never implies more protection than exists.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
DEFAULT_IMPACTS = ("High",)


def to_eastern(now: datetime | None) -> datetime:
    """``now`` as a naive US/Eastern wall-clock datetime (what the calendar uses)."""
    if now is None:
        now = datetime.now(ET)
    if now.tzinfo is None:
        return now
    return now.astimezone(ET).replace(tzinfo=None)


def _load(events: pd.DataFrame | None) -> pd.DataFrame:
    if events is not None:
        return events
    from app.data.event_calendar import load_events
    try:
        return load_events()
    except Exception:  # noqa: BLE001 - a missing calendar must degrade to "no events", never crash a live poll
        return pd.DataFrame(columns=["timestamp", "event", "impact"])


def coverage(events: pd.DataFrame | None = None) -> dict:
    """Which event types the calendar actually holds right now (so the UI can say what is, and is not, enforced)."""
    df = _load(events)
    have = sorted({str(e) for e in df["event"].unique()}) if len(df) else []
    missing = [e for e in ("FOMC", "NFP", "CPI", "PCE") if e not in have]
    return {"types": have, "missing": missing,
            "text": ("News restrictions cover " + ", ".join(have) + ".") + (
                f" {', '.join(missing)} not loaded (run scripts/update_event_calendar.py with a free FRED key)." if missing else "")}


def news_state(now: datetime | None = None, *, before_min: int = 5, after_min: int = 5, horizon_min: int = 120,
               impacts: tuple = DEFAULT_IMPACTS, events: pd.DataFrame | None = None) -> dict:
    """{"in_window", "event", "starts", "ends", "minutes_to_next", "next_event", "next_at"} for ``now``."""
    t = pd.Timestamp(to_eastern(now))
    df = _load(events)
    out = {"in_window": False, "event": None, "starts": None, "ends": None, "minutes_to_next": None, "next_event": None, "next_at": None}
    if not len(df):
        return out
    df = df[df["impact"].isin(impacts)] if "impact" in df.columns else df
    if not len(df):
        return out
    ts = pd.to_datetime(df["timestamp"])
    lo, hi = t - timedelta(minutes=after_min), t + timedelta(minutes=max(before_min, horizon_min))
    near = df[(ts >= lo) & (ts <= hi)].assign(_ts=ts[(ts >= lo) & (ts <= hi)]).sort_values("_ts")
    for _, r in near.iterrows():
        at = pd.Timestamp(r["_ts"])
        start, end = at - timedelta(minutes=before_min), at + timedelta(minutes=after_min)
        if start <= t <= end:
            out.update(in_window=True, event=str(r["event"]), starts=start.isoformat(), ends=end.isoformat())
        if at >= t and out["next_event"] is None:
            out.update(next_event=str(r["event"]), next_at=at.isoformat(), minutes_to_next=round((at - t).total_seconds() / 60.0, 1))
    return out


def blackout_lines_for_day(day, *, before_min: int = 5, after_min: int = 5, to_server_offset_hours: int = -1,
                           impacts: tuple = DEFAULT_IMPACTS, events: pd.DataFrame | None = None) -> str:
    """The day's restricted windows as ``HH:MM-HH:MM`` lines in SERVER time, the format ``execution_engine.parse_blackout_windows``
    reads. ``to_server_offset_hours`` converts US/Eastern to the broker's clock (-1 = US Central, what Tradovate and most US-futures
    servers run on)."""
    df = _load(events)
    if not len(df):
        return ""
    ts = pd.to_datetime(df["timestamp"])
    d = pd.Timestamp(day).normalize()
    todays = df[(ts >= d) & (ts < d + pd.Timedelta(days=1))]
    if "impact" in todays.columns:
        todays = todays[todays["impact"].isin(impacts)]
    lines = []
    for at in sorted(pd.to_datetime(todays["timestamp"])):
        a = at + timedelta(hours=to_server_offset_hours) - timedelta(minutes=before_min)
        b = at + timedelta(hours=to_server_offset_hours) + timedelta(minutes=after_min)
        if a.date() == b.date() == (at + timedelta(hours=to_server_offset_hours)).date():
            lines.append(f"{a:%H:%M}-{b:%H:%M}")
    return "\n".join(lines)
