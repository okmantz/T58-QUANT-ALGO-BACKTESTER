from unittest.mock import MagicMock, patch

import requests

from app.ai import news_forexfactory


def _ok_response(payload):
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    resp.raise_for_status = lambda: None
    return resp


def test_fetch_calendar_success_uses_browser_headers():
    payload = [{"title": "Fed Speech", "country": "USD", "impact": "Medium", "date": "2026-10-02T08:30:00-04:00"}]
    with patch("requests.get", return_value=_ok_response(payload)) as mock_get:
        result = news_forexfactory.fetch_calendar()
    assert result.error is None
    assert len(result.events) == 1
    _, kwargs = mock_get.call_args
    assert "Mozilla" in kwargs["headers"]["User-Agent"]
    assert mock_get.call_args[0][0] == news_forexfactory.CALENDAR_URL


def test_fetch_calendar_falls_back_to_mirror_on_connection_error():
    payload = [{"title": "CPI", "country": "USD", "impact": "High", "date": "2026-10-02T08:30:00-04:00"}]
    calls = []

    def fake_get(url, timeout=None, headers=None):
        calls.append(url)
        if url == news_forexfactory.CALENDAR_URL:
            raise requests.exceptions.ConnectionError("primary down")
        return _ok_response(payload)

    with patch("requests.get", side_effect=fake_get):
        result = news_forexfactory.fetch_calendar()
    assert result.error is None
    assert len(result.events) == 1
    assert calls == [news_forexfactory.CALENDAR_URL, news_forexfactory.CALENDAR_URL_FALLBACK]


def test_fetch_calendar_both_endpoints_down_reports_clear_error():
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("all down")):
        result = news_forexfactory.fetch_calendar()
    assert result.events == []
    assert result.error is not None


def test_fetch_calendar_non_connection_error_does_not_try_fallback():
    calls = []

    def fake_get(url, timeout=None, headers=None):
        calls.append(url)
        raise ValueError("weird parse failure")

    with patch("requests.get", side_effect=fake_get):
        result = news_forexfactory.fetch_calendar()
    assert result.error is not None
    assert calls == [news_forexfactory.CALENDAR_URL]
