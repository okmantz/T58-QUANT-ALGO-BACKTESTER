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
trend), not a true fundamental read.
"""
from __future__ import annotations

import pandas as pd

from app.ai import market_scanner, news_forexfactory
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
