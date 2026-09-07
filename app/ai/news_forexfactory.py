"""News Intelligence: fetches ForexFactory's public economic-calendar feed
and turns it into structured events the rest of the AI assistant (and the
dashboard's News panel) can use.

ForexFactory itself doesn't publish a documented public API, but it serves
a plain JSON calendar feed (used by many third-party calendar widgets)
at nfs.faireconomy.media -- no key, no login, no scraping of the HTML
site required. That endpoint is what this module calls. If it's ever
unreachable (network blocked, feed moved, offline dev machine), every
function here fails soft: an empty list plus a human-readable `error`,
never an exception -- same convention as app.ai.ollama_client, so a
broken internet connection degrades the dashboard's News panel to
"no data" instead of crashing the whole assistant.

Deliberately does no interpretation: this module only parses and
classifies (impact level, which of Owen's watched symbols an event
affects). Whether a given event actually matters for a trade idea is
Layer 3's job (app.ai.trading_assistant), same "Ollama interprets,
the app calculates" split as everywhere else in this package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
DEFAULT_TIMEOUT_SECONDS = 15

# Which of Owen's watched symbols each currency's news can move. Kept as a
# plain dict (not auto-derived) so it stays easy to read/extend -- add a
# symbol here and its news-risk flag on the dashboard starts working
# immediately, no code changes elsewhere.
CURRENCY_TO_SYMBOLS: dict[str, list[str]] = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
            "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "US30", "NAS100", "SPX500"],
    "EUR": ["EURUSD", "EURJPY", "EURGBP"],
    "GBP": ["GBPUSD", "GBPJPY", "EURGBP"],
    "JPY": ["USDJPY", "EURJPY", "GBPJPY"],
    "AUD": ["AUDUSD"],
    "CAD": ["USDCAD"],
    "CHF": ["USDCHF"],
    "NZD": ["NZDUSD"],
    "CNY": ["AUDUSD", "XAUUSD"],
}

IMPACT_ORDER = {"High": 3, "Medium": 2, "Low": 1, "Holiday": 0}


@dataclass
class NewsEvent:
    title: str
    currency: str
    impact: str          # "High" | "Medium" | "Low" | "Holiday"
    when: datetime | None
    forecast: str = ""
    previous: str = ""
    actual: str = ""

    @property
    def affected_symbols(self) -> list[str]:
        return CURRENCY_TO_SYMBOLS.get(self.currency, [])

    def minutes_until(self, now: datetime | None = None) -> float | None:
        if self.when is None:
            return None
        now = now or datetime.now(timezone.utc)
        return (self.when - now).total_seconds() / 60.0


@dataclass
class CalendarResult:
    events: list[NewsEvent] = field(default_factory=list)
    error: str | None = None


def _parse_event(raw: dict) -> NewsEvent | None:
    try:
        when = None
        date_str = raw.get("date")
        if date_str:
            # Feed uses ISO-8601 with an embedded offset, e.g. "2026-09-08T08:30:00-04:00".
            when = datetime.fromisoformat(date_str)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            when = when.astimezone(timezone.utc)
        return NewsEvent(
            title=str(raw.get("title", "")).strip(),
            currency=str(raw.get("country", "")).strip().upper(),
            impact=str(raw.get("impact", "")).strip() or "Low",
            when=when,
            forecast=str(raw.get("forecast", "") or ""),
            previous=str(raw.get("previous", "") or ""),
            actual=str(raw.get("actual", "") or ""),
        )
    except Exception:
        return None


def fetch_calendar(timeout: int = DEFAULT_TIMEOUT_SECONDS) -> CalendarResult:
    """Fetches this week's calendar. Never raises. On any failure returns
    an empty CalendarResult with `.error` set, so callers (the Flask route
    and app.ai.trading_assistant) can display "news unavailable" instead
    of a stack trace."""
    import requests

    try:
        resp = requests.get(CALENDAR_URL, timeout=timeout, headers={"User-Agent": "T58-Trading-Assistant/1.0"})
        resp.raise_for_status()
        raw_events = resp.json()
    except requests.exceptions.ConnectionError:
        return CalendarResult(error="Couldn't reach the ForexFactory calendar feed (no internet?).")
    except requests.exceptions.Timeout:
        return CalendarResult(error="ForexFactory calendar feed timed out.")
    except Exception as exc:
        return CalendarResult(error=f"Couldn't load the calendar feed: {exc}")

    if not isinstance(raw_events, list):
        return CalendarResult(error="Calendar feed returned an unexpected format.")

    events = [e for e in (_parse_event(r) for r in raw_events) if e is not None]
    events.sort(key=lambda e: (e.when is None, e.when))
    return CalendarResult(events=events)


def high_impact_events(result: CalendarResult, upcoming_only: bool = True) -> list[NewsEvent]:
    now = datetime.now(timezone.utc)
    events = [e for e in result.events if e.impact == "High"]
    if upcoming_only:
        events = [e for e in events if e.when is not None and e.when >= now]
    return events


def next_high_impact_event(result: CalendarResult) -> NewsEvent | None:
    upcoming = high_impact_events(result, upcoming_only=True)
    return upcoming[0] if upcoming else None


def news_risk_for_symbol(result: CalendarResult, symbol: str, within_minutes: float = 90.0) -> str:
    """Returns "high" if a high-impact event affecting this symbol lands
    within `within_minutes`, "medium" if a medium-impact one does, else
    "none". This is the flag app.ai.t58_strategy_engine.MarketSnapshot's
    news_risk field is meant to be filled from."""
    now = datetime.now(timezone.utc)
    level = "none"
    for event in result.events:
        if symbol not in event.affected_symbols or event.when is None:
            continue
        minutes = (event.when - now).total_seconds() / 60.0
        if not (0 <= minutes <= within_minutes):
            continue
        if event.impact == "High":
            return "high"
        if event.impact == "Medium":
            level = "medium"
    return level
