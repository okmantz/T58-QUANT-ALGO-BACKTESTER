"""
Historical economic-calendar alignment for event-driven strategies.

app.ai.news_forexfactory already fetches ForexFactory's public calendar
feed -- but that feed only ever returns the CURRENT week (nfs.faireconomy.
media serves a rolling "this week" JSON document, not a historical
archive). A backtest needs a per-bar feature computed over the entire
historical dataset being tested, which a live "this week" feed cannot
provide by itself.

This module bridges that gap the same honest, minimal way
app.data.pairs bridges the single-instrument engine to a second
instrument: it does NOT change the engine or add a network dependency to
every backtest. Instead:

  1. `record_current_week(path)` appends whatever `news_forexfactory.
     fetch_calendar()` returns right now into a local, ever-growing CSV
     history file (deduplicated by title+currency+timestamp) -- called
     opportunistically (e.g. once a day the app is open) so a genuine
     historical archive accumulates over time. Day one has zero rows;
     a year in, it has a year of real events. This is the same
     "build it up for real, don't fake it" approach as this app's own
     falsification-kit ethos.
  2. `load_calendar_history(path)` reads that CSV back into a plain
     DataFrame (timestamp, currency, impact).
  3. `merge_news_features(df, calendar_df, symbol, impact_levels=("High",))`
     merges two new columns into a COPY of `df`, aligned by timestamp,
     exactly like app.data.pairs.merge_pair_series merges `pair_close`:
       - minutes_since_high_impact_news: minutes since the most recent
         qualifying event at-or-before this bar (NaN if none yet).
       - minutes_until_high_impact_news: minutes until the next
         qualifying event after this bar (NaN if none scheduled within
         the merged calendar's own coverage).
     "Qualifying" means impact in `impact_levels` AND the event's
     currency affects `symbol` (app.ai.news_forexfactory.
     CURRENCY_TO_SYMBOLS -- the same mapping the live dashboard uses).

KNOWN, EXPLICIT LIMITATION (same spirit as app.data.pairs' own
docstring): until a real historical archive has been built up via
`record_current_week` over time, or a person supplies their own
historical calendar CSV (same three columns), a backtest run through
this module will show no news features for older data -- both new
columns come back all-NaN, and app.search.strategy_space's calendar
families correctly produce zero candidates from missing data rather
than a misleadingly-confident backtest. This is NOT a live news feed
wired into every backtest; it's a local log this app's own calendar
requests build up.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from app.ai.news_forexfactory import CURRENCY_TO_SYMBOLS, CalendarResult

CALENDAR_HISTORY_COLUMNS = ["timestamp", "title", "currency", "impact"]
DEFAULT_SINCE_COLUMN = "minutes_since_high_impact_news"
DEFAULT_UNTIL_COLUMN = "minutes_until_high_impact_news"


class EconomicCalendarError(Exception):
    """Raised when historical calendar data can't be read or merged."""


def record_current_week(result: CalendarResult, path: str | Path) -> int:
    """Appends every timestamped event in `result` to the CSV at `path`,
    creating it with a header if it doesn't exist yet. Deduplicates
    against what's already on disk by (timestamp, title, currency) so
    calling this repeatedly (e.g. once per day the dashboard is open)
    never double-counts the same event across overlapping weekly fetches.
    Returns the number of NEW rows actually written."""
    path = Path(path)
    existing_keys: set[tuple[str, str, str]] = set()
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row.get("timestamp", ""), row.get("title", ""), row.get("currency", "")))

    new_rows = []
    for event in result.events:
        if event.when is None:
            continue
        ts_str = event.when.isoformat()
        key = (ts_str, event.title, event.currency)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_rows.append(key + (event.impact,))

    if not new_rows:
        return 0

    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CALENDAR_HISTORY_COLUMNS)
        writer.writerows(new_rows)
    return len(new_rows)


def load_calendar_history(path: str | Path) -> pd.DataFrame:
    """Reads the CSV `record_current_week` builds up (or an equivalent
    hand-supplied CSV with the same `timestamp,title,currency,impact`
    columns -- e.g. a purchased historical calendar export) into a plain
    DataFrame with a parsed, UTC `timestamp` column. Returns an empty
    (but correctly-shaped) DataFrame if the file doesn't exist yet,
    rather than raising -- day one of a fresh install has no history,
    and that's an expected, not an error, state."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=CALENDAR_HISTORY_COLUMNS)
    df = pd.read_csv(path)
    missing = [c for c in CALENDAR_HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise EconomicCalendarError(f"Calendar history file is missing column(s): {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def merge_news_features(
    df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    symbol: str,
    impact_levels: tuple[str, ...] = ("High",),
    since_column: str = DEFAULT_SINCE_COLUMN,
    until_column: str = DEFAULT_UNTIL_COLUMN,
) -> pd.DataFrame:
    """Returns a COPY of `df` with two extra columns -- see module
    docstring. `calendar_df` is whatever `load_calendar_history` returns
    (or an equivalent DataFrame with `timestamp`/`currency`/`impact`
    columns). Only events whose currency is in
    app.ai.news_forexfactory.CURRENCY_TO_SYMBOLS[currency] for `symbol`
    are considered -- same watch-list convention the live dashboard uses.

    Backward-looking only (merge_asof direction="backward"/"forward" on
    an already-sorted, already-filtered event list) -- a bar can only
    ever see events that have already printed for `minutes_since_...`,
    and `minutes_until_...` is a genuinely forward-looking SCHEDULE
    lookup (the calendar itself is known in advance, unlike price),
    which is standard practice for backtesting scheduled-release
    strategies -- the release TIME is public knowledge well before the
    release, only its content is not.
    """
    if "timestamp" not in df.columns:
        raise EconomicCalendarError("df must have a 'timestamp' column to align on.")
    for col in ("timestamp", "currency", "impact"):
        if col not in calendar_df.columns:
            raise EconomicCalendarError(f"calendar_df is missing required column '{col}'.")

    relevant_currencies = {c for c, symbols in CURRENCY_TO_SYMBOLS.items() if symbol in symbols}
    qualifying = calendar_df[
        calendar_df["impact"].isin(impact_levels) & calendar_df["currency"].isin(relevant_currencies)
    ].copy()
    qualifying["timestamp"] = pd.to_datetime(qualifying["timestamp"], utc=True)
    qualifying = qualifying.sort_values("timestamp")

    left = df.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    left_sorted = left.sort_values("timestamp")

    if qualifying.empty:
        left_sorted[since_column] = float("nan")
        left_sorted[until_column] = float("nan")
        return left_sorted.set_index(left_sorted.index).reindex(left.index)

    events = qualifying[["timestamp"]].rename(columns={"timestamp": "event_time"})

    since = pd.merge_asof(left_sorted, events, left_on="timestamp", right_on="event_time", direction="backward")
    left_sorted[since_column] = (since["timestamp"] - since["event_time"]).dt.total_seconds() / 60.0

    until = pd.merge_asof(left_sorted, events, left_on="timestamp", right_on="event_time", direction="forward")
    left_sorted[until_column] = (until["event_time"] - until["timestamp"]).dt.total_seconds() / 60.0

    merged = left_sorted.set_index(left_sorted.index).reindex(left.index)
    return merged
