"""Owen AI Assistant -- web routes.

Wires app.ai.t58_strategy_engine / app.ai.news_forexfactory /
app.ai.market_scanner / app.ai.trading_assistant into the mobile/desktop
browser app as one dashboard page (News panel + Best Markets panel + Owen
AI chat panel) plus a small JSON API those panels poll.

Registered as a Flask Blueprint, same pattern as app/web/quant_lab_routes.py
-- kept out of the already-3,700-line app/web/server.py.

Data source note: live bars come from this app's existing global MT5
connection (app.web.live_market -- the same one the Live Market tab uses)
for forex/futures/broker-crypto-CFDs, since that's the only source in this
repo that covers all three of Owen's asset classes from one account. If
MT5 isn't connected, crypto symbols additionally try Alpaca (no paid tier
required) as a fallback; forex/futures simply show "no data" with a clear
reason rather than a stack trace, exactly like the Live Market tab does.

Macro bias caveat (surfaced honestly in the API's `macro_note` field and
worth knowing before trusting the rankings): this app has no fundamentals/
rates/positioning data source, so macro_bias_by_symbol is a plain
technical PROXY (daily EMA50 vs EMA200 trend), not a true fundamental
read. Ask Owen AI chat directly for a fundamental macro take -- the model
can reason about that from its own knowledge/training even though the
deterministic engine can't compute it -- and treat the scanner's status
column as technical-only until you do.
"""
from __future__ import annotations

import time

import pandas as pd
from flask import Blueprint, jsonify, render_template, request

from app.ai import market_scanner, news_forexfactory, t58_strategy_engine as t58
from app.ai import trading_assistant
from app.ai.ollama_settings import OllamaSettings, load_settings as load_ollama_settings, save_settings as save_ollama_settings

ai_assistant_bp = Blueprint("ai_assistant", __name__, url_prefix="/assistant")

_CACHE_TTL_NEWS = 300      # seconds -- the calendar doesn't change second to second
_CACHE_TTL_RANKINGS = 60   # seconds -- bounds how often we hammer the MT5/Alpaca connection
_cache: dict = {}


def _cached(key: str, ttl: float, compute):
    entry = _cache.get(key)
    now = time.time()
    if entry and now - entry[0] < ttl:
        return entry[1]
    value = compute()
    _cache[key] = (now, value)
    return value


def _bar_fetcher(symbol: str, timeframe_minutes: int, count: int):
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


def _daily_trend_bias(symbol: str) -> str:
    """Technical PROXY for macro_bias -- see module docstring's caveat.
    Daily EMA50 above EMA200 -> 'bullish', below -> 'bearish', otherwise
    'neutral'. Deliberately simple and explainable, not a claim of real
    fundamental analysis."""
    try:
        df = _bar_fetcher(symbol, 1440, 260)
        if df is None or len(df) < 210:
            return "neutral"
        from app.strategy.indicators import ema
        e50, e200 = ema(df["close"], 50), ema(df["close"], 200)
        if float(e50.iloc[-1]) > float(e200.iloc[-1]):
            return "bullish"
        if float(e50.iloc[-1]) < float(e200.iloc[-1]):
            return "bearish"
        return "neutral"
    except Exception:
        return "neutral"


def _get_universe() -> dict[str, list[str]]:
    return market_scanner.DEFAULT_UNIVERSE


def _compute_news():
    result = news_forexfactory.fetch_calendar()
    return result


def _compute_rankings():
    universe = _get_universe()
    all_symbols = [s for symbols in universe.values() for s in symbols]
    macro_bias = {s: _daily_trend_bias(s) for s in all_symbols}

    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    news_risk = {s: news_forexfactory.news_risk_for_symbol(news_result, s) for s in all_symbols}

    rankings, errors = market_scanner.rank_markets(
        universe, bar_fetcher=_bar_fetcher, macro_bias_by_symbol=macro_bias, news_risk_by_symbol=news_risk,
    )
    return rankings, errors


@ai_assistant_bp.route("/")
def dashboard():
    saved_ai = load_ollama_settings()
    return render_template(
        "ai_assistant.html",
        active_page="ai_assistant",
        ollama_enabled=saved_ai.enabled, ollama_host=saved_ai.host, ollama_model=saved_ai.model,
    )


@ai_assistant_bp.route("/api/news")
def api_news():
    result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    if result.error:
        return jsonify({"error": result.error, "events": []})
    upcoming = [e for e in result.events if e.when is None or e.minutes_until() is None or e.minutes_until() >= -60]
    return jsonify({
        "error": None,
        "events": [
            {
                "title": e.title, "currency": e.currency, "impact": e.impact,
                "when": e.when.isoformat() if e.when else None,
                "minutes_until": e.minutes_until(), "forecast": e.forecast, "previous": e.previous,
                "actual": e.actual, "affected_symbols": e.affected_symbols,
            }
            for e in upcoming[:40]
        ],
    })


@ai_assistant_bp.route("/api/rankings")
def api_rankings():
    rankings, errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    return jsonify({
        "rankings": [market_scanner.ranking_to_dict(r) for r in rankings],
        "errors": errors,
        "macro_note": (
            "Bias shown is a technical proxy (daily 50/200 EMA trend), not fundamental macro analysis -- "
            "ask the chat for a fundamental read."
        ),
    })


@ai_assistant_bp.route("/api/snapshot/<symbol>")
def api_snapshot(symbol: str):
    h1 = _bar_fetcher(symbol, 60, 300)
    m15 = _bar_fetcher(symbol, 15, 200)
    if h1 is None or h1.empty:
        return jsonify({"error": f"No data available for {symbol}."}), 404
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    snapshot = t58.build_market_snapshot(
        symbol=symbol, h1_frame=h1, m15_frame=m15,
        macro_bias=_daily_trend_bias(symbol),
        news_risk=news_forexfactory.news_risk_for_symbol(news_result, symbol),
    )
    assessment = t58.assess(snapshot)
    return jsonify({
        "symbol": symbol,
        "status": assessment.status, "direction": assessment.direction, "score": assessment.score,
        "checklist": assessment.checklist, "missing": assessment.missing,
        "invalidation": assessment.invalidation, "target": assessment.target,
        "ema_alignment": snapshot.ema.alignment, "zone": snapshot.location.zone,
    })


@ai_assistant_bp.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        settings = OllamaSettings(
            enabled=bool(data.get("enabled", False)),
            host=(data.get("host") or "").strip() or OllamaSettings().host,
            model=(data.get("model") or "").strip() or OllamaSettings().model,
            api_key=(data.get("api_key") or "").strip(),
        )
        save_ollama_settings(settings)
        return jsonify({"ok": True})
    settings = load_ollama_settings()
    return jsonify({"enabled": settings.enabled, "host": settings.host, "model": settings.model})


@ai_assistant_bp.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    mode = data.get("mode") or "personal"
    history = data.get("history") or []
    if not message:
        return jsonify({"error": "Empty message."}), 400

    rankings, _errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    context = trading_assistant.build_context(rankings, news_result.events)

    client = trading_assistant.TradingAssistantClient(load_ollama_settings())
    reply, error = client.ask(message, context, mode=mode, history=history)
    if error:
        return jsonify({"reply": "", "error": error})
    return jsonify({"reply": reply, "error": None})


@ai_assistant_bp.route("/api/daily-brief")
def api_daily_brief():
    rankings, _errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    context = trading_assistant.build_context(rankings, news_result.events)
    client = trading_assistant.TradingAssistantClient(load_ollama_settings())
    reply, error = client.daily_brief(context)
    return jsonify({"reply": reply, "error": error})


@ai_assistant_bp.route("/api/watchlist", methods=["POST"])
def api_watchlist():
    data = request.get_json(force=True, silent=True) or {}
    symbols = data.get("symbols") or []
    rankings, _errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    context = trading_assistant.build_context(rankings, news_result.events, watchlist_symbols=symbols)
    client = trading_assistant.TradingAssistantClient(load_ollama_settings())
    reply, error = client.watchlist(context)
    return jsonify({"reply": reply, "error": error})


@ai_assistant_bp.route("/api/pre-trade-check", methods=["POST"])
def api_pre_trade_check():
    data = request.get_json(force=True, silent=True) or {}
    symbol = (data.get("symbol") or "").strip()
    if not symbol:
        return jsonify({"error": "Missing 'symbol'."}), 400
    rankings, _errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    context = trading_assistant.build_context(rankings, news_result.events, watchlist_symbols=[symbol])
    client = trading_assistant.TradingAssistantClient(load_ollama_settings())
    reply, error = client.pre_trade_check(context, symbol)
    return jsonify({"reply": reply, "error": error})
