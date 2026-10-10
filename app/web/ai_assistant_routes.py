"""T58 AI Assistant -- web routes.

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
worth knowing before trusting the rankings): each ranking's technical
bias/status is still a plain PROXY (daily EMA50 vs EMA200 trend), not a
true fundamental read -- that scoring hasn't changed. Since Sep 2026 each
ranking also carries a `fundamental_bias` field built from recent FRED/
ForexFactory data surprises (see app.ai.market_intelligence's docstring
and app.ai.news_forexfactory.recent_data_surprise_bias_by_currency) --
Owen AI chat can reason about the two agreeing or conflicting; ask it
directly for that read.
"""
from __future__ import annotations

import base64
import functools
import logging
import time
import traceback

from flask import Blueprint, Response, jsonify, render_template, request

from app.ai import ai_director, market_intelligence, market_scanner, news_forexfactory, t58_strategy_engine as t58
from app.ai import trading_assistant
from app.ai.ollama_settings import OllamaSettings, load_settings as load_ollama_settings, save_settings as save_ollama_settings

ai_assistant_bp = Blueprint("ai_assistant", __name__, url_prefix="/assistant")

_logger = logging.getLogger(__name__)

_CACHE_TTL_NEWS = 300      # seconds -- the calendar doesn't change second to second
_CACHE_TTL_RANKINGS = 60   # seconds -- bounds how often we hammer the MT5/Alpaca connection
_CACHE_TTL_STRUCTURE = 300  # seconds -- daily-bar BOS/ChoCH/Wyckoff facts don't change within a minute
_cache: dict = {}

# v9.15: AI Director budgets. The panel used to sit on "Loading... /
# Generating..." indefinitely because this path stacked an unbounded
# sequential market scan (per-symbol feed attempts in
# app.ai.market_intelligence, each now individually bounded there)
# in front of a 600s Ollama completion. Each half now has its own
# finite budget, enforced through the SHARED transport's own timeout
# parameters (no second timeout stack): on expiry the endpoint still
# returns the deterministic directive list plus the transport's named
# error, so the UI always reaches a terminal state.
DIRECTOR_RANKINGS_TIMEOUT_S = 25.0
DIRECTOR_OLLAMA_TOTAL_TIMEOUT_S = 45.0
DIRECTOR_OLLAMA_STALL_TIMEOUT_S = 15.0
DIRECTOR_OLLAMA_FIRST_TOKEN_TIMEOUT_S = 20.0


def _json_safe(view):
    """BUGFIX (Sep 2026): every route in this blueprint used to have zero
    exception handling. Any unhandled exception anywhere in the
    rankings/news/context/Ollama pipeline (a bad symbol, a flaky data
    feed, a genuine bug) fell straight through to Flask's DEFAULT error
    handler, which renders an HTML page -- not JSON. Every button on the
    AI Assistant page does `(await fetch(...)).json()`, so an HTML
    response there throws "Unexpected token '<' ... is not valid JSON"
    in the browser and the button looks dead, with zero indication of
    what actually broke ("Generate Basic Outlook" reported exactly this
    symptom). This decorator is the one place that guarantee is now
    enforced: ANY exception raised by a wrapped view is caught here and
    turned into a normal 200 JSON response carrying the real error
    message (never swallowed silently -- also logged server-side with
    the full traceback for anything deeper than a bad symbol/timeout),
    so a caller's existing `data.error || data.text || ...` fallback
    handling already in ai_assistant.html's JS just works instead of the
    fetch throwing. Apply this to EVERY route in this blueprint that can
    touch live data, Ollama, or the strategy engine -- i.e. all of them
    except the plain HTML page route.
    """

    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        try:
            return view(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: this is the last line of defense
            _logger.exception("AI Assistant route %s failed", view.__name__)
            message = f"{type(exc).__name__}: {exc}"
            return jsonify({
                "error": message,
                "reply": "", "text": "", "rankings": [], "events": [],
                "traceback": traceback.format_exc(limit=6),
            })

    return wrapped



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
@_json_safe
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
@_json_safe
def api_rankings():
    rankings, errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    return jsonify({
        "rankings": [market_scanner.ranking_to_dict(r) for r in rankings],
        "errors": errors,
        "macro_note": (
            "Technical bias (used for each symbol's score) is a proxy: daily 50/200 EMA trend. "
            "\"fundamental_bias\" alongside it comes from recent FRED/ForexFactory data surprises "
            "(actual vs. forecast on released high/medium-impact releases) -- ask the chat how the "
            "two agree or conflict for a fuller read."
        ),
    })


@ai_assistant_bp.route("/api/snapshot/<symbol>")
@_json_safe
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
@_json_safe
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
@_json_safe
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
@_json_safe
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
@_json_safe
def api_daily_brief():
    rankings, _errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    structure_notes = _structure_notes_for(rankings)
    context = trading_assistant.build_context(rankings, news_result.events, market_structure_by_symbol=structure_notes)
    client = trading_assistant.TradingAssistantClient(load_ollama_settings())
    reply, error = client.daily_brief(context)
    return jsonify({"reply": reply, "error": error})


@ai_assistant_bp.route("/api/trade-of-the-day")
@_json_safe
def api_trade_of_the_day():
    """One-button "best trade of the day for MES/MNQ/MGC" -- scans exactly
    app.ai.market_scanner.DEFAULT_UNIVERSE["micro_futures"] (not the full
    AI Assistant universe) so the reply is never diluted by unrelated
    forex/crypto rankings, then hands Ollama the SAME real, deterministic
    T58 checklist + concrete entry/stop/target price levels every other
    AI Assistant reply is grounded in -- see
    app.ai.trading_assistant.TradingAssistantClient.trade_of_the_day's own
    docstring for why Ollama is never asked to invent a price level."""
    micro_futures_universe = {"micro_futures": market_scanner.DEFAULT_UNIVERSE["micro_futures"]}
    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    rankings, errors = market_intelligence.compute_rankings(news_result=news_result, universe=micro_futures_universe)
    structure_notes = _structure_notes_for(rankings)
    context = trading_assistant.build_context(rankings, news_result.events, market_structure_by_symbol=structure_notes)
    client = trading_assistant.TradingAssistantClient(load_ollama_settings())
    reply, error = client.trade_of_the_day(context)
    return jsonify({"reply": reply, "error": error, "fetch_errors": errors})


@ai_assistant_bp.route("/api/watchlist", methods=["POST"])
@_json_safe
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
@_json_safe
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
        text = deterministic + "\n\n--- T58 AI's read ---\n" + reply
    return jsonify({"text": text, "error": None})


def _rankings_within_budget():
    """Today's rankings, or ([], honest-note) if the scan doesn't
    finish within DIRECTOR_RANKINGS_TIMEOUT_S. The scan keeps running
    in its worker (and populates the shared rankings cache when it
    lands); the Director simply stops waiting on it and computes its
    deterministic list without today's market-alignment bonus rather
    than hanging the panel. This is a wait bound on OUR scan, not a
    new data/Ollama timeout stack."""
    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="t58-director-scan")
    try:
        future = executor.submit(_cached, "rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
        rankings, _errors = future.result(timeout=DIRECTOR_RANKINGS_TIMEOUT_S)
        return rankings, None
    except concurrent.futures.TimeoutError:
        return [], (
            f"Best Markets scan didn't finish within {DIRECTOR_RANKINGS_TIMEOUT_S:.0f}s -- "
            "showing the library priority list without today's market alignment."
        )
    except Exception:  # noqa: BLE001 -- a scan failure degrades the same way
        return [], "Best Markets scan was unavailable -- showing the library priority list without today's market alignment."
    finally:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)


def _compute_director_directives():
    """Shared by /api/director and /api/director/stream. Surveys the
    WHOLE Strategy Library (all three languages) plus today's Best
    Markets scan, and returns (directives, memory_counts) -- see
    app.ai.ai_director for the deterministic scoring itself. Never
    raises: a library/DB read failure just means an empty directive list
    (the deterministic text below already handles that gracefully),
    never a 500 for the panel.

    v9.15: returns a third value, `scan_note` -- None when the market
    scan landed in budget, otherwise an honest note that directives
    were computed without it."""
    from app.ai import experiment_memory
    from app.strategy import library

    try:
        strategies = library.list_saved_strategies()
    except Exception:
        strategies = []
    rankings, scan_note = _rankings_within_budget()
    ranking_dicts = [market_scanner.ranking_to_dict(r) for r in rankings] if rankings else []
    try:
        memory_counts = experiment_memory.get_summary_counts()
    except Exception:
        memory_counts = {"total": 0, "by_verdict": {}, "top_strategies": {}}
    directives = ai_director.compute_directives(strategies, rankings=ranking_dicts)
    return directives, memory_counts, scan_note


def _director_budgets() -> dict:
    """The Director's Ollama budgets, forwarded to the shared v9.14
    transport via TradingAssistantClient (see module constants)."""
    return {
        "total_timeout": DIRECTOR_OLLAMA_TOTAL_TIMEOUT_S,
        "stall_timeout": DIRECTOR_OLLAMA_STALL_TIMEOUT_S,
        "first_token_timeout": DIRECTOR_OLLAMA_FIRST_TOKEN_TIMEOUT_S,
    }


@ai_assistant_bp.route("/api/director")
@_json_safe
def api_director():
    """Backs the AI Director panel. Always returns the deterministic
    priority list + fallback text (every figure computed by
    app.strategy.library / app.ai.market_scanner / app.ai.experiment_memory);
    appends Ollama's narrative briefing on top only if it's enabled/
    reachable -- identical fallback posture to /api/outlook."""
    directives, memory_counts, scan_note = _compute_director_directives()
    deterministic = ai_director.build_deterministic_briefing(directives, memory_counts)
    if scan_note:
        deterministic += f"\n\n({scan_note})"
    payload = {"directives": [d.to_dict() for d in directives], "memory_counts": memory_counts}

    settings = load_ollama_settings()
    if not settings.is_usable:
        text = deterministic + "\n\n(Ollama isn't enabled -- showing the deterministic priority list only. " \
            "Turn it on in Ollama settings below for a narrative briefing too.)"
        payload.update({"text": text, "error": None})
        return jsonify(payload)

    client = trading_assistant.TradingAssistantClient(settings)
    user_message = ai_director.build_director_prompt(directives, memory_counts)
    reply, error = client.director_briefing(user_message, **_director_budgets())
    if error:
        text = deterministic + f"\n\n(Ollama narrative unavailable: {error})"
    else:
        text = deterministic + "\n\n--- T58 AI Director's briefing ---\n" + reply
    payload.update({"text": text, "error": None})
    return jsonify(payload)


@ai_assistant_bp.route("/api/director/stream", methods=["POST"])
@_json_safe
def api_director_stream():
    """Streaming twin of /api/director's Ollama narrative half -- same
    newline-delimited-JSON convention as /api/chat/stream. The browser is
    expected to have already rendered the deterministic list from a
    prior /api/director call before triggering this; this endpoint only
    ever streams the narrative briefing text."""
    directives, memory_counts, _scan_note = _compute_director_directives()
    settings = load_ollama_settings()
    if not settings.is_usable:
        def generate_off():
            import json as _json
            yield _json.dumps({"error": "Ollama isn't enabled -- turn it on in Ollama settings below."}) + "\n"
        return Response(generate_off(), mimetype="application/x-ndjson")

    client = trading_assistant.TradingAssistantClient(settings)
    user_message = ai_director.build_director_prompt(directives, memory_counts)

    def generate():
        import json as _json
        # Bounded by the shared transport's stall/first-token/total
        # deadlines (see _director_budgets): a stalled model yields
        # one terminal {"error": "...named cause..."} chunk instead
        # of the browser's "Generating..." spinning forever.
        for chunk in client.director_briefing_stream(user_message, **_director_budgets()):
            yield _json.dumps(chunk) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@ai_assistant_bp.route("/api/analyze-screenshot", methods=["POST"])
@_json_safe
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
@_json_safe
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


@ai_assistant_bp.route("/api/analyze-symbol")
@_json_safe
def api_analyze_symbol():
    """Backs the "Analyze a Symbol" chart-picker button -- pick ANY
    symbol (not limited to the fixed AI Assistant universe or the
    micro_futures Trade-of-the-Day universe) and get Owen's exact T58
    checklist plus concrete entry/stop/target price levels for it,
    using live data + today's news, exactly the same way trade_of_the_day
    does for its fixed three symbols. Always returns the deterministic
    report (see app.ai.trading_assistant.build_deterministic_symbol_report
    -- every figure computed by app.ai.t58_strategy_engine.assess, never
    invented); Ollama's narrative is appended on top only if it's
    enabled/reachable, same fallback posture as /api/outlook.
    Query param: ?symbol=ES1! (case-sensitive match to however your data
    feed names it -- same convention as every other symbol field in this
    app)."""
    symbol = (request.args.get("symbol") or "").strip()
    if not symbol:
        return jsonify({"error": "Missing 'symbol' query parameter.", "reply": "", "text": ""}), 400

    h1 = _bar_fetcher(symbol, 60, 300)
    m15 = _bar_fetcher(symbol, 15, 200)
    if h1 is None or h1.empty or len(h1) < 60:
        return jsonify({
            "error": f"No usable H1 data for '{symbol}' (need at least 60 bars). Check the symbol name "
                     f"matches your data feed/MT5 exactly, and that MT5/Alpaca is connected.",
            "reply": "", "text": "",
        })

    news_result = _cached("news", _CACHE_TTL_NEWS, _compute_news)
    macro_bias = market_intelligence.daily_trend_bias(symbol)
    news_risk = news_forexfactory.news_risk_for_symbol(news_result, symbol)
    snapshot = t58.build_market_snapshot(
        symbol=symbol, h1_frame=h1, m15_frame=m15, macro_bias=macro_bias, news_risk=news_risk,
    )
    assessment = t58.assess(snapshot)

    fundamental_bias = "neutral"
    try:
        currency_bias = news_forexfactory.recent_data_surprise_bias_by_currency(news_result)
        fundamental_bias = market_intelligence.fundamental_bias_for_symbol(symbol, currency_bias)
    except Exception:
        pass  # cosmetic-only field; never let a currency-mapping miss break the whole analysis

    ranking_dict = {
        "symbol": symbol,
        "asset_class": "custom",
        "score": assessment.score,
        "status": assessment.status,
        "direction": assessment.direction,
        "momentum_pct": round(market_scanner._momentum_pct(h1), 3),
        "atr_normalized_move": round(market_scanner._atr_normalized_move(h1), 2),
        "zone": snapshot.location.zone,
        "ema_alignment": snapshot.ema.alignment,
        "target": assessment.target,
        "missing": assessment.missing,
        "news_risk": snapshot.news_risk,
        "fundamental_bias": fundamental_bias,
        "entry_price": assessment.entry_price,
        "stop_price": assessment.stop_price,
        "target_price": assessment.target_price,
    }

    deterministic = trading_assistant.build_deterministic_symbol_report(ranking_dict)
    settings = load_ollama_settings()
    if not settings.is_usable:
        text = deterministic + "\n\n(Ollama isn't enabled -- showing deterministic data only.)"
        return jsonify({"text": text, "error": None, "assessment": ranking_dict})

    rankings, _errors = _cached("rankings", _CACHE_TTL_RANKINGS, _compute_rankings)
    context = trading_assistant.build_context(rankings, news_result.events, symbol_assessment=ranking_dict)
    client = trading_assistant.TradingAssistantClient(settings)
    reply, error = client.analyze_symbol(context, symbol)
    if error:
        text = deterministic + f"\n\n(Ollama narrative unavailable: {error})"
    else:
        text = deterministic + "\n\n--- T58 AI's read ---\n" + reply
    return jsonify({"text": text, "error": None, "assessment": ranking_dict})
