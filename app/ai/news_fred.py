"""FRED Economic Release Calendar -- a second, official source for the
AI Assistant's News panel, alongside app.ai.news_forexfactory's
ForexFactory feed.

Why a second source at all: ForexFactory's feed (nfs.faireconomy.media)
is an undocumented, unofficial third-party JSON endpoint with no SLA --
it moves between hostnames periodically and multiple public forum
threads (MQL5, ForexFactory itself) report it going dark for days at a
time. FRED's `/fred/releases/dates` endpoint is the opposite: an
official, documented, key-authenticated API from the St. Louis Fed
(the same api.stlouisfed.org host app.accounts.api_keys' "Test
Connection" for FRED already calls) that lists every upcoming US
economic-data release by name and date.

Trade-off, stated plainly: FRED's release calendar gives release NAME +
DATE only -- no forecast/previous/actual figures (that would require a
per-series follow-up call for each of dozens of releases, which is a
different, heavier feature). ForexFactory's feed already has
forecast/previous/actual. So this is deliberately a fallback/
supplement, not a replacement: `merge_calendars()` below prefers
ForexFactory's richer events and only adds FRED's when ForexFactory
didn't return that event, or didn't return anything at all.

Every function here follows the exact same fail-soft convention as
news_forexfactory: no FRED key configured, or the request fails for
any reason, returns an empty CalendarResult with `.error` set -- never
an exception, and never silently blocking the News panel from at least
showing whatever ForexFactory did return.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.ai.news_forexfactory import CalendarResult, NewsEvent

FRED_RELEASES_DATES_URL = "https://api.stlouisfed.org/fred/releases/dates"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_DAYS_AHEAD = 14
DEFAULT_DAYS_BEHIND = 3

# FRED doesn't tag its own impact level the way ForexFactory does -- this
# is a plain, hand-maintained set of release NAMES (as FRED's own
# `release_name` field spells them) that move USD pairs/indices the way
# ForexFactory's "High" impact does. Anything else FRED returns is
# tagged "Medium" rather than guessed at. Kept as a plain set (same
# convention as news_forexfactory.CURRENCY_TO_SYMBOLS) so it's easy to
# extend by just adding a name here.
HIGH_IMPACT_RELEASE_NAMES = {
    "Employment Situation",
    "Consumer Price Index",
    "Producer Price Index",
    "Gross Domestic Product",
    "Personal Income and Outlays",
    "Advance Monthly Sales for Retail and Food Services",
    "Federal Open Market Committee",
}


def _parse_release_date(raw: dict) -> NewsEvent | None:
    try:
        date_str = raw.get("date")
        name = str(raw.get("release_name", "")).strip()
        if not date_str or not name:
            return None
        when = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        impact = "High" if name in HIGH_IMPACT_RELEASE_NAMES else "Medium"
        return NewsEvent(title=name, currency="USD", impact=impact, when=when)
    except Exception:
        return None


def fetch_calendar(
    api_key: str,
    days_ahead: int = DEFAULT_DAYS_AHEAD,
    days_behind: int = DEFAULT_DAYS_BEHIND,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> CalendarResult:
    """Fetches every US economic release scheduled between `days_behind`
    days ago and `days_ahead` days from now. Returns an empty
    CalendarResult with `.error` set if no key is configured or the
    request fails for any reason -- never raises."""
    if not api_key:
        return CalendarResult(error="No FRED API key saved yet.")

    import requests

    today = date.today()
    params = {
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": (today - timedelta(days=days_behind)).isoformat(),
        "realtime_end": (today + timedelta(days=days_ahead)).isoformat(),
        "include_release_dates_with_no_data": "false",
        "limit": 1000,
    }
    try:
        resp = requests.get(FRED_RELEASES_DATES_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.ConnectionError as exc:
        return CalendarResult(error=f"Couldn't reach the FRED API (no internet?): {exc}")
    except requests.exceptions.Timeout:
        return CalendarResult(error="FRED API timed out.")
    except Exception as exc:  # noqa: BLE001 -- includes HTTPError (e.g. bad/revoked key)
        return CalendarResult(error=f"FRED rejected the request or is unreachable: {exc}")

    raw_dates = payload.get("release_dates")
    if not isinstance(raw_dates, list):
        return CalendarResult(error="FRED releases/dates returned an unexpected format.")

    events = [e for e in (_parse_release_date(r) for r in raw_dates) if e is not None]
    events.sort(key=lambda e: (e.when is None, e.when))
    return CalendarResult(events=events)


def merge_calendars(primary: CalendarResult, secondary: CalendarResult) -> CalendarResult:
    """Combines a ForexFactory CalendarResult (`primary`, richer:
    forecast/previous/actual) with a FRED one (`secondary`, name+date
    only) into one CalendarResult for the News panel.

    - Both succeeded: union of events, deduplicated by
      (title, currency, calendar date) so the same release reported by
      both sources doesn't show twice -- ForexFactory's copy wins the
      dedup (it has the richer forecast/previous/actual fields).
    - Only one succeeded: that one's events, with NO error surfaced --
      this is the actual point of having two sources: a ForexFactory
      outage no longer blanks the whole News panel as long as a FRED
      key is configured, and vice versa.
    - Both failed: both error messages, so the person can tell FRED's
      key needs attention vs. ForexFactory being down vs. both.
    """
    if primary.error and secondary.error:
        return CalendarResult(error=f"{primary.error} | FRED: {secondary.error}")
    if primary.error:
        return secondary
    if secondary.error:
        return primary

    seen = {(e.title, e.currency, e.when.date()) for e in primary.events if e.when is not None}
    combined = list(primary.events)
    for e in secondary.events:
        key = (e.title, e.currency, e.when.date()) if e.when is not None else None
        if key is not None and key in seen:
            continue
        combined.append(e)
        if key is not None:
            seen.add(key)
    combined.sort(key=lambda e: (e.when is None, e.when))
    return CalendarResult(events=combined)
