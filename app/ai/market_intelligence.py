"""Shared "gather real market facts across forex/crypto/futures" logic,
used by BOTH the web AI Assistant blueprint (app.web.ai_assistant_routes)
and the desktop AI Assistant tab (app.ui.main_window).

BUGFIX (Sep 2026): this logic used to live ONLY inline inside the web
blueprint. The desktop tab's own `_ai_market_context()` always passed
`rankings=[]` into app.ai.trading_assistant.build_context() -- so Daily
Brief/chat/the new Basic Outlook button on the desktop app silently never
saw "which markets moved best" even though the identical feature on the
web app already worked correctly. Extracting the real implementation here
(rather than copy-pasting it into main_window.py a second time) means the
two UIs can no longer drift apart on this again.

Data source: the app's existing global MT5 connection (app.web.live_market
-- the same one the Live Market tab manages) for forex/futures/broker-
crypto-CFDs, with app.data's Alpaca source as a key-free-tier fallback for
crypto when MT5 isn't configured or doesn't list a symbol. Both desktop
and web already depend on app.web.live_market for other features, so
importing it from the desktop side here is not a new dependency.

Macro bias caveat (same one the web dashboard surfaces in `macro_note`):
this app has no fundamentals/rates/positioning data source, so
macro_bias_by_symbol is a plain technical PROXY (daily EMA50 vs EMA200
trend), not a true fundamental read. compute_market_structure_notes()
below is a SEPARATE, complementary signal -- real, deterministic
BOS/ChoCH (break of structure / change of character) and Wyckoff
spring/upthrust/SOS/SOW + phase detection (app.quant_lab.market_structure,
ported from the HyperTA project's Structures module) -- fed into Owen
AI's chat context (see app.ai.trading_assistant.build_context) so the
model has real computed structure facts for its top-ranked symbols
instead of only an EMA-cross proxy, or having to eyeball structure from
price alone.
"""
from __future__ import annotations

import pandas as pd

from app.ai import market_scanner, news_forexfactory
from app.quant_lab.market_structure import MarketStructureError, summarize_market_structure
from app.strategy.indicators import ema


def bar_fetcher(symbol: str, timeframe_minutes: int, count: int):
    """Adapts the app's existing data sources to app.ai.market_scanner's
    BarFetcher shape. Tries the shared MT5 connection first, then Alpaca
    (crypto/US-listed only) if MT5 has nothing for this symbol."""
    from app.web import live_market

    bars = live_market.fetch_mt5_bars(symbol, timeframe_minutes, count)
    if not bars:
        bars = live_market.fetch_alpaca_bars(symbol, "Crypto", timeframe_minutes, count)
    if not bars:
        return None
    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def daily_trend_bias(symbol: str) -> str:
    """Technical PROXY for macro_bias -- see module docstring's caveat.
    Daily EMA50 above EMA200 -> 'bullish', below -> 'bearish', otherwise
    'neutral'."""
    try:
        df = bar_fetcher(symbol, 1440, 260)
        if df is None or len(df) < 210:
            return "neutral"
        e50, e200 = ema(df["close"], 50), ema(df["close"], 200)
        if float(e50.iloc[-1]) > float(e200.iloc[-1]):
            return "bullish"
        if float(e50.iloc[-1]) < float(e200.iloc[-1]):
            return "bearish"
        return "neutral"
    except Exception:
        return "neutral"


def get_universe() -> dict[str, list[str]]:
    return market_scanner.DEFAULT_UNIVERSE


def compute_news() -> "news_forexfactory.CalendarResult":
    return news_forexfactory.fetch_calendar()


def compute_rankings(news_result=None) -> tuple[list, list[str]]:
    """Returns (rankings, errors). Pass an already-fetched `news_result`
    (e.g. from a caller that also needs it separately) to avoid a second
    calendar fetch."""
    universe = get_universe()
    all_symbols = [s for symbols in universe.values() for s in symbols]
    macro_bias = {s: daily_trend_bias(s) for s in all_symbols}

    if news_result is None:
        news_result = compute_news()
    news_risk = {s: news_forexfactory.news_risk_for_symbol(news_result, s) for s in all_symbols}

    return market_scanner.rank_markets(
        universe, bar_fetcher=bar_fetcher, macro_bias_by_symbol=macro_bias, news_risk_by_symbol=news_risk,
    )


def market_structure_note_for_symbol(symbol: str) -> str:
    """One-line, plain-language BOS/ChoCH/Wyckoff summary for `symbol` on
    daily bars -- see this module's docstring for why this exists
    alongside (not instead of) daily_trend_bias()'s EMA-cross proxy.
    Returns a short "no data" note rather than raising, matching
    daily_trend_bias()'s own fail-safe convention (a chat context should
    degrade gracefully for one bad symbol, not break the whole context)."""
    try:
        df = bar_fetcher(symbol, 1440, 260)
        if df is None or len(df) < 50:
            return "not enough daily bars for structure analysis"
        summary = summarize_market_structure(df)
        note = f"trend={summary.latest_trend}, bias={summary.bias}"
        if summary.latest_structure_event:
            note += f", last event={summary.latest_structure_event}"
        if summary.latest_wyckoff_phase:
            note += f", wyckoff={summary.latest_wyckoff_phase}"
        return note
    except MarketStructureError as exc:
        return f"structure analysis unavailable: {exc}"
    except Exception:
        return "structure analysis unavailable"


def compute_market_structure_notes(symbols: list[str], max_symbols: int = 5) -> dict[str, str]:
    """Computes market_structure_note_for_symbol() for up to `max_symbols`
    of `symbols` -- capped because this fetches+analyzes daily bars per
    symbol, and the chat context only needs this for the handful of
    symbols actually relevant to the current conversation (typically the
    top-ranked markets), not the whole universe on every message."""
    return {s: market_structure_note_for_symbol(s) for s in symbols[:max_symbols]}
