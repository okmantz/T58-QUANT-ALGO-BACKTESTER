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

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
# Fallback mirror -- the exact same feed on FaireEconomy's CDN subdomain.
# Real-world evidence (mirrored complaints in MQL5/ForexFactory forums)
# is that this endpoint moves between nfs./cdn-nfs. periodically, and
# that some ISPs/networks block whichever one isn't currently the
# "documented" one. Tried only if the primary fails to connect at all.
CALENDAR_URL_FALLBACK = "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"
DEFAULT_TIMEOUT_SECONDS = 15
# A generic User-Agent string ("T58-Trading-Assistant/1.0") is what this
# module used to send -- some CDNs/bot-protection layers (Cloudflare
# included) silently reject non-browser-looking User-Agents even when
# the actual network path is fine, which reads to the caller as "no
# internet" when the real cause is "this looks like a bot". A real
# browser UA plus a plain Accept header (mirroring what a browser tab
# would send) avoids that class of false "no internet" failure.
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

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
    of a stack trace.

    Tries the primary URL first, then the fallback mirror ONLY if the
    primary couldn't be reached at all (ConnectionError) -- a real
    "revoked"/"malformed" style response from the primary is treated as
    authoritative and not retried against the mirror, same philosophy
    as app.licensing.client's network_ok vs. response-said-no
    distinction."""
    import requests

    last_error = None
    for url in (CALENDAR_URL, CALENDAR_URL_FALLBACK):
        try:
            resp = requests.get(url, timeout=timeout, headers=_BROWSER_HEADERS)
            resp.raise_for_status()
            raw_events = resp.json()
        except requests.exceptions.ConnectionError as exc:
            last_error = f"Couldn't reach the ForexFactory calendar feed (no internet?): {exc}"
            continue  # try the fallback mirror before giving up
        except requests.exceptions.Timeout:
            last_error = "ForexFactory calendar feed timed out."
            continue
        except Exception as exc:
            return CalendarResult(error=f"Couldn't load the calendar feed: {exc}")

        if not isinstance(raw_events, list):
            last_error = "Calendar feed returned an unexpected format."
            continue

        events = [e for e in (_parse_event(r) for r in raw_events) if e is not None]
        events.sort(key=lambda e: (e.when is None, e.when))
        return CalendarResult(events=events)

    return CalendarResult(error=last_error or "Couldn't reach the ForexFactory calendar feed.")


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


# ---------------------------------------------------------------------------
# Recent-data-surprise bias (Sep 2026): a second, FUNDAMENTAL-flavored proxy
# alongside app.ai.market_intelligence.daily_trend_bias's technical one.
# Owen asked for Best Markets to "also draw from FRED and Forex Factory" --
# this is that: for each currency, compares already-released actual vs.
# forecast on recent High/Medium impact events and scores whether the data
# has been surprising bullish or bearish for that currency.
#
# Deliberately narrow, and labeled as a proxy rather than a real fundamental
# model: it only reads the numbers the calendar feed already gives us
# (actual vs. forecast), it has no idea which release matters most this
# week, and its "higher is bullish" assumption is wrong for a small set of
# indicators (unemployment/claims-style releases), which LOWER_IS_BETTER_
# KEYWORDS below corrects for by name -- anything not in that list defaults
# to "higher actual than forecast = bullish for the currency". FRED-sourced
# events (see app.ai.news_fred) rarely populate actual/forecast -- FRED's
# calendar is name+date only -- so in practice this mostly scores
# ForexFactory's numbers; FRED still contributes event coverage/timing.
LOWER_IS_BETTER_KEYWORDS = (
    "unemployment", "jobless claims", "initial claims", "continuing claims",
)
DEFAULT_SURPRISE_LOOKBACK_HOURS = 96.0


def _parse_calendar_number(raw: str) -> float | None:
    """Parses a ForexFactory-style actual/forecast string ("3.2%", "175K",
    "-0.3%", "1.75M") into a bare float. Returns None for blank/dash
    ("not yet released") or anything unparseable -- callers must treat
    that as "no surprise to score", never as zero."""
    s = (raw or "").strip().replace(",", "")
    if not s or s in ("-", "\u2014"):
        return None
    match = re.match(r"^([+-]?\d*\.?\d+)\s*([KMB%]?)$", s, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    multiplier = {"k": 1e3, "m": 1e6, "b": 1e9}.get(match.group(2).lower(), 1.0)
    return value * multiplier


def recent_data_surprise_bias_by_currency(
    result: CalendarResult,
    lookback_hours: float = DEFAULT_SURPRISE_LOOKBACK_HOURS,
) -> dict[str, str]:
    """Returns {currency: "bullish" | "bearish" | "neutral"} from already-
    released High/Medium impact events in the last `lookback_hours`. A
    currency with no scorable recent releases simply doesn't appear in the
    returned dict (callers should default missing currencies to
    "neutral"), rather than this function inventing a bias from nothing."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)
    scores: dict[str, float] = {}

    for event in result.events:
        if event.impact not in ("High", "Medium") or event.when is None:
            continue
        if not (cutoff <= event.when <= now):
            continue
        actual = _parse_calendar_number(event.actual)
        forecast = _parse_calendar_number(event.forecast)
        if actual is None or forecast is None or actual == forecast:
            continue
        sign = 1.0 if actual > forecast else -1.0
        if any(kw in event.title.lower() for kw in LOWER_IS_BETTER_KEYWORDS):
            sign = -sign
        weight = 2.0 if event.impact == "High" else 1.0
        scores[event.currency] = scores.get(event.currency, 0.0) + sign * weight

    biases: dict[str, str] = {}
    for currency, score in scores.items():
        if score > 0.5:
            biases[currency] = "bullish"
        elif score < -0.5:
            biases[currency] = "bearish"
        else:
            biases[currency] = "neutral"
    return biases
