"""Tests for app.data.economic_calendar -- the historical calendar
accumulation/merge module backing the economic_calendar_* strategy
families."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from app.ai.news_forexfactory import CalendarResult, NewsEvent
from app.data.economic_calendar import (
    EconomicCalendarError,
    load_calendar_history,
    merge_news_features,
    record_current_week,
)


def _event(title, currency, impact, when):
    return NewsEvent(title=title, currency=currency, impact=impact, when=when)


def test_record_current_week_creates_file_and_returns_new_row_count(tmp_path):
    path = tmp_path / "calendar_history.csv"
    result = CalendarResult(events=[
        _event("NFP", "USD", "High", datetime(2026, 1, 2, 13, 30, tzinfo=timezone.utc)),
        _event("CPI", "USD", "High", datetime(2026, 1, 3, 13, 30, tzinfo=timezone.utc)),
    ])
    written = record_current_week(result, path)
    assert written == 2
    assert path.exists()


def test_record_current_week_deduplicates_across_calls(tmp_path):
    path = tmp_path / "calendar_history.csv"
    result = CalendarResult(events=[
        _event("NFP", "USD", "High", datetime(2026, 1, 2, 13, 30, tzinfo=timezone.utc)),
    ])
    first = record_current_week(result, path)
    second = record_current_week(result, path)
    assert first == 1
    assert second == 0


def test_record_current_week_skips_events_with_no_timestamp(tmp_path):
    path = tmp_path / "calendar_history.csv"
    result = CalendarResult(events=[_event("Bank Holiday", "USD", "Holiday", None)])
    written = record_current_week(result, path)
    assert written == 0
    assert not path.exists()


def test_load_calendar_history_returns_empty_frame_when_file_missing(tmp_path):
    df = load_calendar_history(tmp_path / "does_not_exist.csv")
    assert df.empty
    assert list(df.columns) == ["timestamp", "title", "currency", "impact"]


def test_load_calendar_history_round_trips_recorded_events(tmp_path):
    path = tmp_path / "calendar_history.csv"
    result = CalendarResult(events=[
        _event("NFP", "USD", "High", datetime(2026, 1, 2, 13, 30, tzinfo=timezone.utc)),
    ])
    record_current_week(result, path)
    df = load_calendar_history(path)
    assert len(df) == 1
    assert df.iloc[0]["currency"] == "USD"
    assert df.iloc[0]["impact"] == "High"


def test_load_calendar_history_raises_on_malformed_csv(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("not,the,right,columns\n1,2,3,4\n")
    with pytest.raises(EconomicCalendarError):
        load_calendar_history(path)


def _bars(n=20, start="2026-01-01", freq="15min"):
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})


def test_merge_news_features_computes_minutes_since_and_until():
    df = _bars(n=20)
    calendar_df = pd.DataFrame({
        "timestamp": [df["timestamp"].iloc[5]],
        "title": ["NFP"], "currency": ["USD"], "impact": ["High"],
    })
    out = merge_news_features(df, calendar_df, symbol="EURUSD")
    assert out["minutes_since_high_impact_news"].iloc[:5].isna().all()
    assert out["minutes_since_high_impact_news"].iloc[5] == 0
    assert out["minutes_since_high_impact_news"].iloc[6] == 15
    assert out["minutes_until_high_impact_news"].iloc[4] == 15
    assert out["minutes_until_high_impact_news"].iloc[6:].isna().all()


def test_merge_news_features_ignores_currencies_irrelevant_to_symbol():
    df = _bars(n=10)
    calendar_df = pd.DataFrame({
        "timestamp": [df["timestamp"].iloc[3]],
        "title": ["BOJ Rate"], "currency": ["JPY"], "impact": ["High"],
    })
    # EURUSD's relevant currencies are EUR/USD, not JPY -- this event
    # should never show up as a feature for it.
    out = merge_news_features(df, calendar_df, symbol="EURUSD")
    assert out["minutes_since_high_impact_news"].isna().all()
    assert out["minutes_until_high_impact_news"].isna().all()


def test_merge_news_features_respects_impact_levels_filter():
    df = _bars(n=10)
    calendar_df = pd.DataFrame({
        "timestamp": [df["timestamp"].iloc[3]],
        "title": ["Minor release"], "currency": ["USD"], "impact": ["Low"],
    })
    out = merge_news_features(df, calendar_df, symbol="EURUSD", impact_levels=("High",))
    assert out["minutes_since_high_impact_news"].isna().all()


def test_merge_news_features_handles_an_empty_calendar():
    df = _bars(n=10)
    calendar_df = pd.DataFrame(columns=["timestamp", "title", "currency", "impact"])
    out = merge_news_features(df, calendar_df, symbol="EURUSD")
    assert out["minutes_since_high_impact_news"].isna().all()
    assert out["minutes_until_high_impact_news"].isna().all()
    assert len(out) == len(df)


def test_merge_news_features_raises_without_timestamp_column():
    df = _bars(n=5).drop(columns=["timestamp"])
    calendar_df = pd.DataFrame(columns=["timestamp", "title", "currency", "impact"])
    with pytest.raises(EconomicCalendarError):
        merge_news_features(df, calendar_df, symbol="EURUSD")


def test_merge_news_features_preserves_original_row_order():
    df = _bars(n=15)
    calendar_df = pd.DataFrame({
        "timestamp": [df["timestamp"].iloc[7]],
        "title": ["NFP"], "currency": ["USD"], "impact": ["High"],
    })
    out = merge_news_features(df, calendar_df, symbol="EURUSD")
    assert (out["timestamp"].to_numpy() == df["timestamp"].to_numpy()).all()
