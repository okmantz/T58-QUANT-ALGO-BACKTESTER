"""Best Markets scanner: pulls bars for a watchlist of forex/crypto/futures
symbols, computes plain-statistics movement rankings (momentum,
ATR-normalized move), runs each through app.ai.t58_strategy_engine, and
returns a sorted "which ones moved best / which ones are actually
worth Owen's attention" table for the dashboard's Best Markets panel.

Data source is injected as a plain callable (`bar_fetcher`) rather than
this module owning a connection itself, so it stays source-agnostic and
unit-testable with fake data. In the actual web app
(app/web/ai_assistant_routes.py) that callable is wired to
app.web.live_market's existing global MT5 connection for forex/futures/
broker-crypto-CFDs (the same one the Live Market tab already manages --
reusing it rather than opening a second connection, since the underlying
MetaTrader5 package is a single connection to one terminal per process),
with app.data.alpaca_source as a key-free-tier fallback for crypto/US
equities when MT5 isn't configured or doesn't list a symbol.

Nothing here is AI -- it is Layer 1/2 (deterministic facts) from the
proposed architecture. app.ai.trading_assistant is the only place an LLM
touches this data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from app.ai import t58_strategy_engine as t58
from app.strategy.indicators import atr

# (symbol, timeframe_minutes, count) -> OHLCV DataFrame (standard
# timestamp/open/high/low/close/volume schema) or None if unavailable.
BarFetcher = Callable[[str, int, int], Optional[pd.DataFrame]]

# Default watchlist across the three asset classes Owen asked for. Purely a
# starting point -- callers (the Flask route) can pass their own list.
DEFAULT_UNIVERSE: dict[str, list[str]] = {
    "forex": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAUUSD"],
    "crypto": ["BTCUSD", "ETHUSD"],
    "futures": ["US30", "NAS100", "SPX500"],  # MT5 CFD naming varies by broker; adjust to match your symbol list
}


@dataclass
class SymbolFetchResult:
    symbol: str
    h1: pd.DataFrame | None = None
    m15: pd.DataFrame | None = None
    error: str | None = None


@dataclass
class MarketRanking:
    symbol: str
    asset_class: str
    momentum_pct: float
    atr_normalized_move: float
    t58_assessment: "t58.T58Assessment"
    snapshot: "t58.MarketSnapshot"


def fetch_symbol_bars(symbol: str, bar_fetcher: BarFetcher, h1_count: int = 300, m15_count: int = 200) -> SymbolFetchResult:
    """Never raises -- any failure is captured on `.error` so one bad
    symbol doesn't stop the rest of the scan."""
    try:
        h1 = bar_fetcher(symbol, 60, h1_count)
        m15 = bar_fetcher(symbol, 15, m15_count)
        if h1 is None or h1.empty:
            return SymbolFetchResult(symbol=symbol, error="No H1 bars returned for this symbol.")
        return SymbolFetchResult(symbol=symbol, h1=h1, m15=m15)
    except Exception as exc:
        return SymbolFetchResult(symbol=symbol, error=str(exc))


def _momentum_pct(frame: pd.DataFrame, bars: int = 24) -> float:
    """% price change over the last `bars` H1 candles (default 24 = last day)."""
    if len(frame) <= bars:
        return 0.0
    then, now = float(frame["close"].iloc[-bars - 1]), float(frame["close"].iloc[-1])
    if then == 0:
        return 0.0
    return (now - then) / then * 100.0


def _atr_normalized_move(frame: pd.DataFrame, bars: int = 24, atr_period: int = 14) -> float:
    """How many ATRs the price has moved over `bars` candles -- comparable
    across instruments with very different price scales/pip values, unlike
    raw percentage change (a $200 gold move and a 0.002 EURUSD move aren't
    otherwise comparable)."""
    if len(frame) <= max(bars, atr_period):
        return 0.0
    atr_series = atr(frame, atr_period)
    current_atr = float(atr_series.iloc[-1])
    if current_atr <= 0:
        return 0.0
    then, now = float(frame["close"].iloc[-bars - 1]), float(frame["close"].iloc[-1])
    return abs(now - then) / current_atr


def rank_markets(
    universe: dict[str, list[str]],
    bar_fetcher: BarFetcher,
    macro_bias_by_symbol: dict[str, str] | None = None,
    news_risk_by_symbol: dict[str, str] | None = None,
) -> tuple[list[MarketRanking], list[str]]:
    """Scans every symbol in `universe`, returns (rankings sorted best
    first by T58 score then ATR-normalized move, list of per-symbol error
    strings for anything that couldn't be fetched). macro_bias_by_symbol
    should come from app.ai.trading_assistant's macro read (see that
    module) -- defaults every symbol to "neutral" if not supplied, which
    will correctly keep every assessment at WAIT/PASS rather than
    inventing a directional bias from price action alone."""
    macro_bias_by_symbol = macro_bias_by_symbol or {}
    news_risk_by_symbol = news_risk_by_symbol or {}
    rankings: list[MarketRanking] = []
    errors: list[str] = []

    for asset_class, symbols in universe.items():
        for symbol in symbols:
            fetched = fetch_symbol_bars(symbol, bar_fetcher)
            if fetched.error or fetched.h1 is None or len(fetched.h1) < 60:
                errors.append(f"{symbol}: {fetched.error or 'not enough bars returned'}")
                continue
            snapshot = t58.build_market_snapshot(
                symbol=symbol,
                h1_frame=fetched.h1,
                m15_frame=fetched.m15,
                macro_bias=macro_bias_by_symbol.get(symbol, "neutral"),
                news_risk=news_risk_by_symbol.get(symbol, "none"),
            )
            assessment = t58.assess(snapshot)
            rankings.append(MarketRanking(
                symbol=symbol,
                asset_class=asset_class,
                momentum_pct=_momentum_pct(fetched.h1),
                atr_normalized_move=_atr_normalized_move(fetched.h1),
                t58_assessment=assessment,
                snapshot=snapshot,
            ))

    rankings.sort(key=lambda r: (r.t58_assessment.score, r.atr_normalized_move), reverse=True)
    return rankings, errors


def ranking_to_dict(r: MarketRanking) -> dict:
    """Flat, JSON-safe representation for the Flask API / dashboard table."""
    return {
        "symbol": r.symbol,
        "asset_class": r.asset_class,
        "score": r.t58_assessment.score,
        "status": r.t58_assessment.status,
        "direction": r.t58_assessment.direction,
        "momentum_pct": round(r.momentum_pct, 3),
        "atr_normalized_move": round(r.atr_normalized_move, 2),
        "zone": r.snapshot.location.zone,
        "ema_alignment": r.snapshot.ema.alignment,
        "target": r.t58_assessment.target,
        "missing": r.t58_assessment.missing,
        "news_risk": r.snapshot.news_risk,
    }
