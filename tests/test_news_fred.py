from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.ai import news_fred
from app.ai.news_forexfactory import CalendarResult, NewsEvent


def _fred_payload(*rows):
    return {"release_dates": list(rows)}


def test_fetch_calendar_no_key_returns_error_no_exception():
    result = news_fred.fetch_calendar("")
    assert result.error is not None
    assert result.events == []


def test_fetch_calendar_parses_and_tags_high_impact():
    payload = _fred_payload(
        {"release_name": "Employment Situation", "date": "2026-10-02"},
        {"release_name": "Some Minor Regional Survey", "date": "2026-10-03"},
    )
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: payload)
        mock_get.return_value.raise_for_status = lambda: None
        result = news_fred.fetch_calendar("FAKEKEY")
    assert result.error is None
    assert len(result.events) == 2
    high = [e for e in result.events if e.title == "Employment Situation"][0]
    assert high.impact == "High"
    assert high.currency == "USD"
    minor = [e for e in result.events if e.title == "Some Minor Regional Survey"][0]
    assert minor.impact == "Medium"


def test_fetch_calendar_connection_error_fails_soft():
    import requests

    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("boom")):
        result = news_fred.fetch_calendar("FAKEKEY")
    assert result.events == []
    assert "FRED" in result.error


def test_fetch_calendar_bad_format_returns_error():
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"nope": []})
        mock_get.return_value.raise_for_status = lambda: None
        result = news_fred.fetch_calendar("FAKEKEY")
    assert result.error is not None


def _event(title, currency="USD", impact="High", day=2):
    return NewsEvent(title=title, currency=currency, impact=impact,
                      when=datetime(2026, 10, day, 12, 0, tzinfo=timezone.utc))


def test_merge_calendars_both_ok_dedupes_by_title_currency_date():
    primary = CalendarResult(events=[_event("Employment Situation")])
    secondary = CalendarResult(events=[_event("Employment Situation"), _event("CPI", day=3)])
    merged = news_fred.merge_calendars(primary, secondary)
    assert merged.error is None
    titles = sorted(e.title for e in merged.events)
    assert titles == ["CPI", "Employment Situation"]
    # ForexFactory's (primary's) copy of the duplicate wins -- same object.
    assert [e for e in merged.events if e.title == "Employment Situation"][0] is primary.events[0]


def test_merge_calendars_primary_failed_falls_back_to_secondary_no_error():
    primary = CalendarResult(error="Couldn't reach the ForexFactory calendar feed (no internet?).")
    secondary = CalendarResult(events=[_event("CPI")])
    merged = news_fred.merge_calendars(primary, secondary)
    assert merged.error is None
    assert len(merged.events) == 1


def test_merge_calendars_secondary_failed_falls_back_to_primary_no_error():
    primary = CalendarResult(events=[_event("CPI")])
    secondary = CalendarResult(error="No FRED API key saved yet.")
    merged = news_fred.merge_calendars(primary, secondary)
    assert merged.error is None
    assert len(merged.events) == 1


def test_merge_calendars_both_failed_combines_errors():
    primary = CalendarResult(error="ForexFactory is down.")
    secondary = CalendarResult(error="No FRED API key saved yet.")
    merged = news_fred.merge_calendars(primary, secondary)
    assert merged.error is not None
    assert "ForexFactory is down." in merged.error
    assert "FRED" in merged.error
