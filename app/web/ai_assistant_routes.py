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

import base64
import time

from flask import Blueprint, Response, jsonify, render_template, request

from app.ai import market_intelligence, market_scanner, news_forexfactory, t58_strategy_engine as t58
from app.ai import trading_assistant
from app.ai.ollama_settings import OllamaSettings, load_settings as load_ollama_settings, save_settings as save_ollama_settings

ai_assistant_bp = Blueprint("ai_assistant", __name__, url_prefix="/assistant")

_CACHE_TTL_NEWS = 300      # seconds -- the calendar doesn't change second to second
_CACHE_TTL_RANKINGS = 60   # seconds -- bounds how often we hammer the MT5/Alpaca connection
_CACHE_TTL_STRUCTURE = 300  # seconds -- daily-bar BOS/ChoCH/Wyckoff facts don't change within a minute
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
    return market_intelligence.bar_fetcher(symbol, timeframe_minutes, count)


def _daily_trend_bias(symbol: str) -> str:
    return market_intelligence.daily_trend_bias(symbol)


def _get_universe() -> dict[str, list[str]]:
    return market_intelligence.get_universe()


def _compute_news():
    return market_intelligence.compute_news()


def _compute_rankings():
    # BUGFIX (Sep 2026): this used to duplicate app.ai.market_intelligence's
    # logic inline -- see that module's docstring for why it was pulled out
    # (the desktop AI Assistant tab needed the exact same scan and had
    # drifted, always sending empty rankings). Delegating here means this
    # blueprint and the desktop tab can no longer disagree about what
    # "best markets" means.
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    return market_intelligence.compute_rankings(news_result=news_result)


def _structure_notes_for(rankings) -> dict:
    """Real, deterministic BOS/ChoCH/Wyckoff facts for the top-ranked
    symbols -- see app.ai.market_intelligence's module docstring. Cached
    separately from rankings (longer TTL: these come from DAILY bars, so
    they can't meaningfully change within the rankings' own 60s window)
    so a chat message doesn't pay for a fresh structure analysis every
    time rankings happen to refresh."""
    top_symbols = tuple(r.symbol for r in rankings[:5])
    return _cached(f"structure:{top_symbols}", _CACHE_TTL_STRUCTURE,
                   lambda: market_intelligence.compute_market_structure_notes(list(top_symbols)))


@ai_assistant_bp.route("/")
def dashboard():
    saved_ai = load_ollama_settings()
    return render_template(
        "ai_assistant.html",
        active_page="ai_assistant",
        ollama_enabled=saved_ai.enabled, ollama_host=saved_ai.host, ollama_model=saved_ai.model,
        ollama_vision_model=getattr(saved_ai, "vision_model", "llava"),
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
            vision_model=(data.get("vision_model") or "").strip() or OllamaSettings().vision_model,
        )
        save_ollama_settings(settings)
        return jsonify({"ok": True})
    settings = load_ollama_settings()
    return jsonify({
        "enabled": settings.enabled, "host": settings.host, "model": settings.model,
        "vision_model": getattr(settings, "vision_model", "llava"),
    })


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
    structure_notes = _structure_notes_for(rankings)
    context = trading_assistant.build_context(rankings, news_result.events, market_structure_by_symbol=structure_notes)

    client = trading_assistant.TradingAssistantClient(load_ollama_settings())
    reply, error = client.ask(message, context, mode=mode, history=history)
    if error:
        return jsonify({"reply": "", "error": error})
    return jsonify({"reply": reply, "error": None})


@ai_assistant_bp.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """Streaming twin of /api/chat -- same context/history handling, but
    the Ollama reply is forwarded to the browser as it's generated
    (newline-delimited JSON chunks: {"text": "..."} per piece, then one
    final {"done": true} or {"error": "..."}) instead of only after the
    whole reply is ready. This is the real fix for "the AI assistant
    takes forever": total local-model generation time is unchanged, but
    the person sees the first words almost immediately instead of a
    spinner for the full duration. /api/chat is left in place unchanged
    for any other caller that still wants one blocking JSON response."""
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    mode = data.get("mode") or "personal"
    history = data.get("history") or []
    if not message:
        return jsonify({"error": "Empty message."}), 400

    rankings, _errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    structure_notes = _structure_notes_for(rankings)
    context = trading_assistant.build_context(rankings, news_result.events, market_structure_by_symbol=structure_notes)
    client = trading_assistant.TradingAssistantClient(load_ollama_settings())

    def generate():
        import json as _json
        for chunk in client.ask_stream(message, context, mode=mode, history=history):
            yield _json.dumps(chunk) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@ai_assistant_bp.route("/api/daily-brief")
def api_daily_brief():
    rankings, _errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    structure_notes = _structure_notes_for(rankings)
    context = trading_assistant.build_context(rankings, news_result.events, market_structure_by_symbol=structure_notes)
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


@ai_assistant_bp.route("/api/outlook")
def api_outlook():
    """Backs the page's "Generate Basic Outlook" button -- see
    app.ai.trading_assistant.MARKET_OUTLOOK_SYSTEM_PROMPT /
    build_deterministic_outlook(). Always returns deterministic text (every
    figure already computed by app.ai.market_scanner/news_forexfactory);
    appends Ollama's narrative read on top only if it's enabled/reachable,
    same fallback behavior as the desktop AI Assistant tab's identical
    button."""
    rankings, _errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    structure_notes = _structure_notes_for(rankings)
    context = trading_assistant.build_context(rankings, news_result.events, market_structure_by_symbol=structure_notes)

    deterministic = trading_assistant.build_deterministic_outlook(context)
    settings = load_ollama_settings()
    if not settings.is_usable:
        text = deterministic + "\n\n(Ollama isn't enabled -- showing deterministic data only. Turn it on in Ollama settings below for a narrative read too.)"
        return jsonify({"text": text, "error": None})

    client = trading_assistant.TradingAssistantClient(settings)
    reply, error = client.market_outlook(context)
    if error:
        text = deterministic + f"\n\n(Ollama narrative unavailable: {error})"
    else:
        text = deterministic + "\n\n--- Owen AI's read ---\n" + reply
    return jsonify({"text": text, "error": None})


@ai_assistant_bp.route("/api/analyze-screenshot", methods=["POST"])
def api_analyze_screenshot():
    """Chart screenshot -> exact trading plan, or trade screenshot ->
    session-review breakdown. Expects multipart/form-data: `image` (the
    file), `kind` ("chart" or "trade"), optional `notes`. Requires a
    vision-capable Ollama model (see OllamaSettings.vision_model) --
    the default text model can't see the image at all."""
    kind = (request.form.get("kind") or "chart").strip().lower()
    notes = (request.form.get("notes") or "").strip()
    image_file = request.files.get("image")
    if image_file is None or not image_file.filename:
        return jsonify({"error": "No image uploaded."}), 400
    if kind not in ("chart", "trade"):
        return jsonify({"error": "kind must be 'chart' or 'trade'."}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded image was empty."}), 400
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    client = trading_assistant.TradingAssistantClient(load_ollama_settings())
    if kind == "chart":
        rankings, _errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
        news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
        context = trading_assistant.build_context(rankings, news_result.events)
        reply, error = client.analyze_chart_screenshot(image_b64, context=context, extra_notes=notes)
    else:
        reply, error = client.analyze_trade_screenshot(image_b64, extra_notes=notes)

    if error:
        return jsonify({"reply": "", "error": error})
    return jsonify({"reply": reply, "error": None})


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
