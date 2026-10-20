"""Scheduled macro-event calendar (FOMC, NFP, CPI, PCE) as a clean, free, forward-usable table.

Report 2026-10-20 (section 5): "free event calendars now". Event TIMES are public schedules, so they are known in advance
and using them as features carries no lookahead -- provided the table is built from the schedule and never from realised
surprises. This module therefore stores only (timestamp, event, impact); it never stores the print.

Sources, in order:
  1. Built-in rules/tables: FOMC decision dates 2022-2026 (14:00 ET) and the NFP rule (BLS: third Friday after the end of the
     reference week that contains the 12th, 08:30 ET). The NFP rule reproduces the published dates, including the
     "second Friday" months, but does NOT know about shutdown-driven delays -- override in the CSV if one occurs.
  2. ``data/events/<event>.csv`` (columns: timestamp, event, impact) -- hand-maintained or written by
     ``fetch_fred_release_dates`` / ``write_events_csv``. CPI and PCE have no closed-form rule, so they come only from here
     (run ``python -m scripts.update_event_calendar --fred-key KEY``); until populated, CPI/PCE features are neutral.
All timestamps are US/Eastern wall-clock, tz-naive, to match the exchange-time bars the backtester consumes.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

EVENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "events"
EVENT_TYPES = ("FOMC", "NFP", "CPI", "PCE")

# FOMC policy-decision dates (second day of each meeting), statement at 14:00 ET.
FOMC_DECISION_DATES = (
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
)
_RELEASE_TIME = {"FOMC": (14, 0), "NFP": (8, 30), "CPI": (8, 30), "PCE": (8, 30)}


def nfp_release_date(year: int, month: int) -> _dt.date:
    """Release date of the Employment Situation for the reference month ENDING in (year, month) -- i.e. the report published
    in that month's following weeks. BLS rule: third Friday after the end (Saturday) of the week containing the 12th."""
    twelfth = _dt.date(year, month, 12)
    # week runs Sunday..Saturday; python weekday(): Mon=0..Sun=6
    days_to_sat = (5 - twelfth.weekday()) % 7
    week_end = twelfth + _dt.timedelta(days=days_to_sat)
    return week_end + _dt.timedelta(days=20)  # first Friday = +6, third = +20


def nfp_dates(start_year: int, end_year: int) -> list[_dt.date]:
    """Release dates for every report PUBLISHED from January of start_year through December of end_year."""
    ref = [(start_year - 1, 12)] + [(y, m) for y in range(start_year, end_year + 1) for m in range(1, 12 + 1)]
    out = sorted({nfp_release_date(y, m) for y, m in ref})
    return [d for d in out if start_year <= d.year <= end_year]


def _frame(event: str, dates, impact: str = "High") -> pd.DataFrame:
    h, m = _RELEASE_TIME[event]
    ts = [pd.Timestamp(d) + pd.Timedelta(hours=h, minutes=m) for d in dates]
    return pd.DataFrame({"timestamp": ts, "event": event, "impact": impact})


def builtin_events(start_year: int = 2022, end_year: int = 2026) -> pd.DataFrame:
    parts = [_frame("FOMC", FOMC_DECISION_DATES), _frame("NFP", nfp_dates(start_year, end_year))]
    return pd.concat(parts, ignore_index=True)


def read_events_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "timestamp" not in cols or "event" not in cols:
        raise ValueError(f"{path}: needs columns timestamp,event[,impact]")
    out = pd.DataFrame({"timestamp": pd.to_datetime(df[cols["timestamp"]]),
                        "event": df[cols["event"]].astype(str).str.upper().str.strip()})
    out["impact"] = df[cols["impact"]].astype(str) if "impact" in cols else "High"
    if getattr(out["timestamp"].dt, "tz", None) is not None:
        out["timestamp"] = out["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    return out


def load_events(events_dir: str | Path | None = None, include_builtin: bool = True) -> pd.DataFrame:
    """Built-in + CSV events, deduplicated on (timestamp, event), sorted. CSV rows win over built-ins for the same event/day."""
    d = Path(events_dir) if events_dir else EVENTS_DIR
    parts = [builtin_events()] if include_builtin else []
    if d.exists():
        for f in sorted(d.glob("*.csv")):
            try:
                parts.append(read_events_csv(f))
            except Exception:  # noqa: BLE001 -- a broken file must not break a backtest
                continue
    if not parts:
        return pd.DataFrame(columns=["timestamp", "event", "impact"])
    df = pd.concat(parts, ignore_index=True)
    df["day"] = df["timestamp"].dt.normalize()
    # a CSV row overrides a built-in row of the same event on a DIFFERENT date within 7 days (e.g. a delayed NFP)
    df = df.drop_duplicates(["timestamp", "event"], keep="last")
    return df.drop(columns="day").sort_values("timestamp").reset_index(drop=True)


def write_events_csv(df: pd.DataFrame, event: str, events_dir: str | Path | None = None) -> Path:
    d = Path(events_dir) if events_dir else EVENTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["event"] = event.upper()
    path = d / f"{event.lower()}.csv"
    out[["timestamp", "event", "impact"]].to_csv(path, index=False)
    return path


def fetch_fred_release_dates(api_key: str, release_id: int, event: str, start: str = "2015-01-01", timeout: int = 30) -> pd.DataFrame:
    """Release dates from the free FRED API (release/dates). Known ids: CPI=10, Employment Situation=50, PCE (Personal Income and
    Outlays)=54. Needs network access and a free API key; returns the schedule at the event's usual release time."""
    import requests
    r = requests.get("https://api.stlouisfed.org/fred/release/dates", timeout=timeout, params={
        "release_id": release_id, "api_key": api_key, "file_type": "json", "include_release_dates_with_no_data": "true",
        "realtime_start": start, "sort_order": "asc", "limit": 10000})
    r.raise_for_status()
    dates = [x["date"] for x in r.json().get("release_dates", [])]
    return _frame(event.upper(), dates)


def event_day_table(events: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (date, event) for calendar-flag features."""
    ev = load_events() if events is None else events
    t = ev.copy()
    t["date"] = pd.to_datetime(t["timestamp"]).dt.normalize()
    return t[["date", "event", "timestamp"]]
