"""Tests for B2-3 (session trading day + source_timezone stamp) and the
trailing-partial-bar drop at import (Oct 2026)."""

import datetime
import io

import pandas as pd
import pytest

from app.data.importer import import_csv, import_csv_bytes
from app.data.trading_day import trading_day

# 2026-04-13 is a Monday; April 2026 is CDT (UTC-5).
MONDAY = pd.Timestamp("2026-04-13")


def _csv(rows: list[tuple[str, float]]) -> io.StringIO:
    """Build a minimal 1m-spaced OHLCV CSV from (timestamp, close) pairs."""
    lines = ["timestamp,open,high,low,close,volume"]
    for ts, close in rows:
        lines.append(f"{ts},{close},{close},{close},{close},10")
    return io.StringIO("\n".join(lines) + "\n")


def _regular_1m(start: pd.Timestamp, n: int) -> list[tuple[str, float]]:
    return [
        ((start + pd.Timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"), 100.0 + i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------
# trading_day: roll-hour boundary (CME futures roll at 17:00 CT)
# ---------------------------------------------------------------------


def test_trading_day_roll_boundary_1659_vs_1700_ct():
    # Naive timestamps are assumed UTC; April is CDT (UTC-5).
    before_roll = pd.Timestamp("2026-04-13 21:59:00")  # 16:59 CT Monday
    at_roll = pd.Timestamp("2026-04-13 22:00:00")  # 17:00 CT Monday
    assert trading_day(before_roll) == datetime.date(2026, 4, 13)
    assert trading_day(at_roll) == datetime.date(2026, 4, 14)


def test_trading_day_late_evening_belongs_to_next_day():
    # 18:00 CT Monday -> Tuesday's trading day (spec's canonical example).
    assert trading_day(pd.Timestamp("2026-04-13 23:00:00")) == datetime.date(2026, 4, 14)


def test_trading_day_early_morning_belongs_to_same_day():
    # 00:30 CT Monday is still Monday's session (opened Sunday 17:00 CT).
    assert trading_day(pd.Timestamp("2026-04-13 05:30:00")) == datetime.date(2026, 4, 13)


def test_trading_day_naive_assumed_utc():
    # 00:30 UTC Monday = 19:30 CT Sunday -> rolls to MONDAY's session.
    # If the naive timestamp were wrongly assumed to be CT, this would
    # report Sunday instead.
    assert trading_day(pd.Timestamp("2026-04-13 00:30:00")) == datetime.date(2026, 4, 13)


def test_trading_day_tz_aware_converted():
    aware_utc = pd.Timestamp("2026-04-13 22:00:00", tz="UTC")
    assert trading_day(aware_utc) == datetime.date(2026, 4, 14)
    # 17:30 America/New_York = 16:30 CT -> still Monday.
    aware_et = pd.Timestamp("2026-04-13 17:30:00", tz="America/New_York")
    assert trading_day(aware_et) == datetime.date(2026, 4, 13)


def test_trading_day_custom_roll_hour():
    # roll_hour=0 degenerates to a calendar-day boundary in tz.
    ts = pd.Timestamp("2026-04-13 23:30:00")  # 18:30 CT
    assert trading_day(ts, roll_hour=0) == datetime.date(2026, 4, 13)
    assert trading_day(ts) == datetime.date(2026, 4, 14)


def test_trading_day_bad_roll_hour():
    with pytest.raises(ValueError):
        trading_day(pd.Timestamp("2026-04-13 12:00:00"), roll_hour=24)


# ---------------------------------------------------------------------
# importer: source_timezone stamp
# ---------------------------------------------------------------------


def test_source_timezone_stamp_defaults_to_utc_for_naive():
    rows = _regular_1m(pd.Timestamp("2023-01-02 00:00:00"), 5)
    result = import_csv(_csv(rows))
    assert result.is_valid
    assert result.dataframe.attrs["source_timezone"] == "UTC"


def test_source_timezone_stamp_records_tz_aware_input():
    rows = _regular_1m(pd.Timestamp("2023-01-02 00:00:00"), 5)
    aware_rows = [
        (
            pd.Timestamp(ts).tz_localize("UTC").tz_convert("UTC+05:00").isoformat(),
            c,
        )
        for ts, c in rows
    ]
    result = import_csv(_csv(aware_rows))
    assert result.is_valid
    assert result.dataframe.attrs["source_timezone"] == "UTC+05:00"


# ---------------------------------------------------------------------
# importer: trailing partial bar drop
# ---------------------------------------------------------------------


def test_trailing_partial_bar_dropped_when_off_cadence():
    # Bundled-MGC signature: 1m bars, final gap 13 minutes -> last bar dropped.
    rows = _regular_1m(pd.Timestamp("2023-01-02 00:00:00"), 10)
    rows.append(("2023-01-02 00:22:00", 110.0))  # 13-min final gap
    result = import_csv(_csv(rows))
    assert result.is_valid
    df = result.dataframe
    assert len(df) == 10
    assert df["timestamp"].iloc[-1] == pd.Timestamp("2023-01-02 00:09:00")
    assert any("trailing bar" in i.message for i in result.issues)


def test_trailing_bar_kept_when_opted_out():
    rows = _regular_1m(pd.Timestamp("2023-01-02 00:00:00"), 10)
    rows.append(("2023-01-02 00:22:00", 110.0))
    result = import_csv(_csv(rows), drop_trailing_partial_bar=False)
    assert result.is_valid
    assert len(result.dataframe) == 11


def test_trailing_bar_kept_when_historical_complete():
    rows = _regular_1m(pd.Timestamp("2023-01-02 00:00:00"), 10)
    rows.append(("2023-01-02 00:22:00", 110.0))
    result = import_csv(_csv(rows), historical_complete=True)
    assert result.is_valid
    assert len(result.dataframe) == 11


def test_trailing_bar_kept_when_on_cadence_and_stale():
    # Complete export: cadence holds to the end, data is old -> nothing dropped.
    rows = _regular_1m(pd.Timestamp("2023-01-02 00:00:00"), 10)
    result = import_csv(_csv(rows))
    assert result.is_valid
    assert len(result.dataframe) == 10
    assert not any("trailing bar" in i.message for i in result.issues)


def test_trailing_bar_dropped_when_fresh_even_if_on_cadence():
    # On-cadence but the last bar is newer than one interval relative to
    # import time -> still forming, dropped.
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
    rows = _regular_1m(now - pd.Timedelta(minutes=9), 10)
    result = import_csv(_csv(rows))
    assert result.is_valid
    assert len(result.dataframe) == 9
    assert any("trailing bar" in i.message for i in result.issues)


def test_trailing_bar_opt_out_flows_through_import_csv_bytes():
    rows = _regular_1m(pd.Timestamp("2023-01-02 00:00:00"), 10)
    rows.append(("2023-01-02 00:22:00", 110.0))
    content = _csv(rows).getvalue().encode()
    dropped = import_csv_bytes(content)
    assert len(dropped.dataframe) == 10
    kept = import_csv_bytes(content, drop_trailing_partial_bar=False)
    assert len(kept.dataframe) == 11
