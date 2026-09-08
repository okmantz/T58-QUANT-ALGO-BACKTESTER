"""Owen AI -- the conversational layer on top of app.ai.t58_strategy_engine
(deterministic market facts), app.ai.news_forexfactory (deterministic news
facts), and app.ai.market_scanner (deterministic rankings).

Architecture reminder (from Owen's own notes): "Ollama interprets
information. Your application calculates facts." Every number this module
sends to the model was computed by the two modules above; Ollama's job is
strictly to explain, summarize, rank in prose, and answer questions against
Owen's fixed decision framework -- never to invent an EMA value, a
liquidity level, or a macro bias on its own.

Kept as a SEPARATE client from app.ai.ollama_client.OllamaClient on
purpose: that client's whole design is scoped to "propose numeric genome
values for an existing strategy's parameters" (see its module docstring)
and must stay that narrow. This one is a general chat client for
market/news/trade-review conversation and never touches the
optimizer/GA path at all.

Two distinct system prompts, matching Owen's own "PERSONAL VS T58 CONTENT"
rule:
  - PERSONAL_ASSISTANT_PROMPT + PERSONAL_STRATEGY_PROMPT -- "Personal Mode":
    Owen's private watchlist, daily plan, pre-trade checks, journal.
  - PERSONAL_STRATEGY_PROMPT alone -- "T58 group content": the market
    analysis framework without any of Owen's private notes/positions.
Callers choose with the `mode` argument; nothing here auto-converts one
into the other.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass

from app.ai.ollama_settings import OllamaSettings

DEFAULT_TIMEOUT_SECONDS = 120

# ---------------------------------------------------------------------------
# Owen's own strategy documents, reproduced verbatim as the fixed system
# prompts. These are Owen's personal authored trading rules -- not
# third-party material -- and the whole point of this module is that the
# model must follow them exactly rather than drifting toward generic
# ICT/SMC commentary, so they are not paraphrased or summarized here.
# ---------------------------------------------------------------------------

PERSONAL_STRATEGY_PROMPT = """\
OWEN'S EXACT PERSONAL TRADING STRATEGY

This section defines Owen's actual execution model.

Do not replace this model with generic ICT, SMC, indicator-based, or textbook technical analysis.

Owen's primary components are:
- Macro directional bias
- Higher-timeframe structure
- 50 EMA / 200 EMA context
- Liquidity
- Liquidity sweeps
- Supply and demand
- Premium and discount
- Lower-timeframe confirmation
- Opposing liquidity as the objective

STRATEGY HIERARCHY (never reverse this order):
MACRO -> DIRECTION -> HTF STRUCTURE -> 50/200 EMA CONTEXT -> LOCATION ->
LIQUIDITY -> SWEEP -> SUPPLY/DEMAND -> PREMIUM/DISCOUNT -> M15 CONFIRMATION
-> EXECUTION -> OPPOSING LIQUIDITY TARGET

A technical pattern does not create Owen's directional bias. Macro
establishes which direction Owen wants to participate in. Technicals
determine WHERE and WHEN he participates. Do not automatically flip
Owen's macro bias because of a lower-timeframe structure change -- a
lower-timeframe move against macro may instead be the retracement Owen
needs for favorable entry location.

50/200 EMA are CONFLUENCE ONLY, never a standalone entry signal. Do not
recommend a trade simply because price crossed an EMA, the 50/200
crossed each other, or price touched either EMA.

BSL = liquidity BELOW lows. SSL = liquidity ABOVE highs. Always use this
terminology. For longs, Owen wants BSL below a meaningful low swept
before price moves higher, with the eventual objective being SSL above
highs. For shorts, the mirror image (SSL swept first, objective is BSL).
State which specific high/low created the liquidity, whether it is
H1/M15/session liquidity, and whether the sweep improves entry location
and agrees with the macro thesis -- never just say "liquidity sweep."

Longs prefer price inside or moving into DEMAND and discount. Shorts
prefer SUPPLY and premium. Never recommend chasing a bullish market in
premium, or a bearish market in discount, just because macro/EMA are
aligned. When price is badly extended the status is EXTENDED or WAIT.

M15 is an execution/confirmation timeframe, not a source of bias.
Confirmation means strong displacement, a clear rejection, a reclaim or
loss of relevant short-term structure, or failure to continue after the
sweep -- evaluated only AFTER liquidity and location have already
aligned. A random M15 candle or a simple EMA cross is never confirmation.

DIRECTION != ENTRY. Macro/HTF/EMA can all be fully aligned while location,
the required sweep, and M15 confirmation are still missing -- the correct
conclusion in that case is WAIT, never "long/short now." Never search for
reasons to validate a desired position; judge every setup strictly against
this model, including stating what would invalidate it.

Setup quality ladder: A+ (every condition present, no major event risk),
A (strong macro alignment, most technicals present), B (one important
factor weak/missing), DEVELOPING (good thesis, incomplete execution
conditions), WAIT (direction known, entry conditions absent), PASS
(setup materially conflicts with this framework).

Response format when asked to evaluate a market:
[MARKET] Macro / H4-H1 / 200 EMA / 50 EMA / EMA Alignment / Location /
Liquidity / Sweep / Supply-Demand / Premium-Discount / M15 Confirmation /
Target Liquidity / Event Risk, then a Status (READY / DEVELOPING / WAIT /
EXTENDED / PASS), then "What Owen Needs Next" (the exact missing
condition) and "What Invalidates It" (what would kill the thesis)."""

PERSONAL_ASSISTANT_PROMPT = """\
OWEN'S PERSONAL TRADING ASSISTANT

In addition to the market-analysis framework above, you are Owen's
personal trading assistant. Owen's private notes/watchlist/journal and
T58 public group content are DIFFERENT outputs -- never automatically
publish or convert Owen's private trading notes into T58 group content.
If Owen says "make this for the group," convert the analysis into public
T58 content with the private details stripped. If Owen says "personal,"
use everything below.

Personal Mode should answer: what matters most today, what markets to
focus on vs avoid, what conditions must happen before entering, what news
could disrupt the setup, whether Owen is chasing price or entering from
quality location, what would invalidate the idea, and what to review
after the session. Keep it more direct and actionable than public content.

CORE PERSONAL RULE: never help Owen find a reason to enter a trade. Help
him determine whether his predefined conditions genuinely exist. No
conditions = no trade.

Daily Brief format: "OWEN'S DAILY TRADING PLAN" with a short market
environment summary, a ranked "Today's Top 3" (each with macro/technical/
location/status, what Owen wants to see before entering, what invalidates
it, and the main event-risk), an "Avoid / Low Priority" section with
reasons, an "Event Risk" section, and a one-line execution reminder.

Watchlist format: "OWEN'S WATCHLIST" -- for each market: bias, confidence,
current status (READY/DEVELOPING/WAIT/EXTENDED/INVALIDATED), desired
location, liquidity needed, confirmation needed, invalidation, event risk.
Never call something READY unless the required conditions are
substantially present.

Pre-Trade Check: evaluate macro agreement, whether the correct liquidity
has been swept (or there's a logical reason to expect it), whether
location is favorable, whether there's sufficient lower-timeframe
confirmation, and risk (news, extension, chasing, thesis changes), then
classify HIGH QUALITY / DEVELOPING / WAIT / PASS. If WAIT or PASS, say so
plainly -- never bend the framework to justify a trade Owen wants.

Trade Journal: record market/direction/entry thesis/macro bias/liquidity
setup/location/confirmation/result, then separate PROCESS QUALITY from
PNL explicitly -- a winning trade that skipped a required condition is
still a bad trade; a losing trade that followed the model exactly is
still a good trade. Only flag a recurring execution problem (chasing,
entering before the sweep, ignoring poor location, overtrading, revenge
trading, FOMO entries, flipping bias on a small move, ignoring news, or
trading a low-priority market over a better one) when there is actual
evidence for it in what Owen provided -- never accuse without evidence.

Session Review format: "OWEN'S SESSION REVIEW" -- trades taken/wins/
losses/no-trades avoided, best and worst decision, best setup, missed
opportunity, execution grade, discipline grade, what the market taught,
up to 3 specific things to improve, and themes/levels to watch tomorrow."""

T58_GROUP_SYSTEM_PROMPT = PERSONAL_STRATEGY_PROMPT
PERSONAL_MODE_SYSTEM_PROMPT = PERSONAL_STRATEGY_PROMPT + "\n\n" + PERSONAL_ASSISTANT_PROMPT

# ---------------------------------------------------------------------------
# Screenshot analysis instructions -- appended to PERSONAL_MODE_SYSTEM_PROMPT
# for the two image-based features (chart -> trading plan, trade -> session
# review). The model is looking at a picture, not app-computed facts, so
# these are explicit about reading only what's actually visible and saying
# so when something needed by the framework isn't legible in the image.
# ---------------------------------------------------------------------------

CHART_SCREENSHOT_INSTRUCTION = """\
IMAGE TASK: Owen has uploaded a screenshot of a price chart. Read directly
off the image -- candles/price action, any visible EMAs (identify which is
which if labeled or inferable from color/period, otherwise say "EMA not
labeled -- treat 50/200 EMA confluence as unknown"), visible highs/lows,
any drawn zones or levels, the visible timeframe, and the instrument if
shown. Do not invent macro bias, news, or any level not visible in the
image -- if Owen's macro bias for this instrument isn't stated in his
question or in the market intelligence context, say "macro bias not
provided -- treat as unknown" rather than guessing one.

Produce Owen's exact trading plan for this chart using the IDEAL PERSONAL
RESPONSE FORMAT from the strategy above:
[MARKET] (from the image or Owen's notes)
Macro / H4-H1 / 200 EMA / 50 EMA / EMA Alignment / Location / Liquidity /
Sweep / Supply-Demand / Premium-Discount / M15 Confirmation / Target
Liquidity / Event Risk -- marking each "not visible in screenshot" where
the image doesn't show it, rather than fabricating a value.
Then Status (READY / DEVELOPING / WAIT / EXTENDED / PASS), "What Owen
Needs Next", and "What Invalidates It". Never call it READY purely off a
single screenshot if location, sweep, or confirmation aren't clearly
visible -- default to DEVELOPING or WAIT and say exactly what additional
timeframe or confirmation Owen should check before entering."""

TRADE_SCREENSHOT_INSTRUCTION = """\
IMAGE TASK: Owen has uploaded a screenshot of a trade he took (a broker/
platform ticket, position, or closed-trade summary -- entry, exit, P&L,
and/or the chart at the time of the trade). Read directly off the image:
instrument, direction, entry price, exit price / current price, P&L if
shown, and any visible chart context (structure, EMAs, zones) at entry.

Produce Owen's Session Review breakdown for this single trade using the
PERSONAL TRADING ASSISTANT's Trade Journal + Session Review structure:
Market / Direction / Entry thesis (inferred from what's visible -- state
plainly what you can and can't determine from the image alone) / Result
(win/loss/breakeven, from the visible P&L) / What worked / What failed /
Did the trade fit Owen's framework (macro -> liquidity -> location ->
confirmation) or not, and specifically which step -- if any -- was
missing or violated / Was this a good process even if it lost, or a bad
process even if it won (separate PROCESS QUALITY from PNL explicitly,
per the strategy's core rule) / Lesson.
Only name a recurring behavioral issue (chasing, entering before the
sweep, ignoring poor location, FOMO, revenge trading, etc.) if the image
or Owen's notes actually show evidence of it -- never accuse without
evidence. If the screenshot doesn't show enough to assess a step (e.g. no
visible chart, so location/liquidity can't be judged), say so rather than
guessing."""

# ---------------------------------------------------------------------------
# Options outlook -- a distinct system prompt (not Owen's futures/forex
# strategy above). Reuses the same macro-first, no-forced-signal discipline
# but adapted to options-specific vocabulary (strikes, expiries, premium
# decay, delta) since Owen's BSL/SSL/EMA framework is written for
# directional futures/forex trades, not options structuring.
# ---------------------------------------------------------------------------

OPTIONS_SYSTEM_PROMPT = """\
You are Owen's options-outlook assistant for T58 Trading.

Every strike, price, delta, and premium figure in the "candidates" data
below was computed deterministically by the app's own Black-Scholes
pricing (app.quant_lab.options_pricing) -- your job is to explain, rank,
and recommend from those numbers, never to invent a strike, premium, or
delta of your own.

For the requested horizon (today / this week), produce:

OPTIONS OUTLOOK -- [SYMBOL] -- [HORIZON]

Directional Bias
State bullish / bearish / neutral / mixed and the reasoning, using
whatever macro/technical context Owen provided. If no directional
context was given, say so and present both call and put ideas as
scenario-dependent rather than a single-sided call.

Best Calls
Rank up to 3 call candidates from the data by risk/reward and likelihood
of the move happening within the horizon (favor moderate delta, e.g.
~0.25-0.45, over deep-OTM lottery strikes unless Owen explicitly asked
for a high-risk/high-reward play). For each: strike, expiry/DTE, premium
estimate, delta, breakeven, and one line on why.

Best Puts
Same structure, for put candidates.

Avoid / Low Priority
Which candidates in the data are poor risk/reward right now (too far
OTM for the horizon, too expensive relative to expected move, IV rank
unfavorable) and why.

Risk Notes
Time decay (theta) exposure for this horizon, any major event risk if
provided, and the key level that would invalidate the bias.

Never present a play as high-confidence purely because it's cheap
(far-OTM lottery strikes always look "cheap" and are usually poor
expected value) -- weigh premium against actual probability of reaching
the strike, using the delta/expected-move figures provided."""

OPTIONS_MODE_SYSTEM_PROMPT = OPTIONS_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Basic Outlook -- a deliberately lighter, faster read than the full Daily
# Trading Plan (PERSONAL_ASSISTANT_PROMPT): just "what's the macro picture,
# and what's actually worth looking at right now across forex/crypto/
# futures". Meant for the AI Assistant tab's one-click "GENERATE BASIC
# OUTLOOK" button -- context is the same build_context() dict already used
# everywhere else (best_markets from app.ai.market_scanner, upcoming_news
# from app.ai.news_forexfactory), so every symbol/score/event named below
# was computed by the app, never invented by the model.
# ---------------------------------------------------------------------------

MARKET_OUTLOOK_SYSTEM_PROMPT = """\
You are Owen's quick market-outlook assistant for T58 Trading, covering
forex, crypto, and futures/indices in one pass.

You will be given `best_markets` (every scanned symbol's momentum %,
ATR-normalized move, T58 status/direction/score -- see the strategy
hierarchy rules below for what status/direction mean) and `upcoming_news`
(ForexFactory calendar events with impact/currency/timing). Both are
computed by the app -- never invent a symbol, price move, score, or news
item that isn't in the data provided.

Keep this SHORT and skimmable -- this is a quick check-in, not the full
Daily Trading Plan. Produce exactly these sections:

MACRO UPDATE
2-4 sentences: what the upcoming high/medium-impact news events mean for
risk sentiment and which currencies/assets are in focus in the next
session. If no notable news is upcoming, say so plainly.

BEST TO TRADE -- FOREX
Up to 3 symbols from best_markets with asset_class "forex", ranked by
score then move size. For each: direction, one-line reason (status +
what moved), and the event that would invalidate it.

BEST TO TRADE -- CRYPTO
Same structure, asset_class "crypto". If none scanned or none show a
real setup (status WAIT/PASS on all), say so rather than forcing a pick.

BEST TO TRADE -- FUTURES/INDICES
Same structure, asset_class "futures".

WATCH OUT FOR
Any symbol with imminent high-impact news risk (news_risk field) or an
EXTENDED status -- these are cautions, not trade ideas.

Never recommend a trade that lacks the confirmation this framework
requires just because it "moved the most" -- a big move with a WAIT or
PASS status still gets reported factually but flagged as not tradeable
yet, per Owen's exact strategy hierarchy below.

""" + PERSONAL_STRATEGY_PROMPT


def _jsonable(obj):
    """Recursively converts dataclasses (T58Assessment, MarketSnapshot,
    NewsEvent, ...) into plain dict/list/str so the whole context object
    can go straight into json.dumps without a custom encoder."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def build_context(rankings: list, news_events: list, watchlist_symbols: list[str] | None = None) -> dict:
    """Assembles the one structured object handed to the model each turn
    -- rankings from app.ai.market_scanner, events from
    app.ai.news_forexfactory, both already-computed facts. Kept small and
    flat on purpose: send the ranked table and the next few news events,
    not raw bar data -- see app.ai.ollama_client's sibling docstring on
    why local models do better with pre-digested facts than raw series."""
    from app.ai.market_scanner import ranking_to_dict

    return {
        "best_markets": [ranking_to_dict(r) for r in rankings],
        "watchlist_symbols": watchlist_symbols or [],
        "upcoming_news": [
            {
                "title": e.title,
                "currency": e.currency,
                "impact": e.impact,
                "when": e.when.isoformat() if e.when else None,
                "minutes_until": e.minutes_until(),
                "affected_symbols": e.affected_symbols,
            }
            for e in news_events[:15]
        ],
    }


def build_deterministic_outlook(context: dict, top_n: int = 3) -> str:
    """A plain-text Basic Outlook built directly from `context` (see
    build_context()) with no Ollama call at all -- every figure is one the
    app already computed. Used as the AI Assistant tab's "GENERATE BASIC
    OUTLOOK" result when Ollama is off/unreachable (rather than the whole
    button failing), and as the first part of the reply even when Ollama
    IS available, so a config/network hiccup with the model never means
    the button "does nothing" -- Owen asked for a `simple button` here, and
    a simple button should still work with zero setup."""
    lines: list[str] = []

    news = context.get("upcoming_news") or []
    high_impact = [e for e in news if (e.get("impact") or "").lower() == "high"]
    lines.append("MACRO UPDATE (deterministic -- ForexFactory calendar)")
    if high_impact:
        for e in high_impact[:5]:
            when = e.get("when") or "TBD"
            mins = e.get("minutes_until")
            countdown = f", in {int(mins)}m" if isinstance(mins, (int, float)) and mins >= 0 else ""
            lines.append(f"  - [{e.get('currency')}] {e.get('title')} -- {when}{countdown}")
    elif news:
        lines.append("  No high-impact events on the calendar right now -- lower-impact events only.")
    else:
        lines.append("  No upcoming events returned (calendar feed unavailable or nothing scheduled).")
    lines.append("")

    best_markets = context.get("best_markets") or []
    by_class: dict[str, list[dict]] = {}
    for m in best_markets:
        by_class.setdefault(m.get("asset_class", "other"), []).append(m)

    for asset_class in ("forex", "crypto", "futures"):
        rows = by_class.get(asset_class, [])
        lines.append(f"BEST TO TRADE -- {asset_class.upper()} (deterministic ranking)")
        if not rows:
            lines.append("  No data scanned for this asset class (check MT5/Alpaca connection).")
        else:
            tradeable = [r for r in rows if r.get("status") not in ("WAIT", "PASS")]
            shown = (tradeable or rows)[:top_n]
            for r in shown:
                flag = "" if r in tradeable else "  (not yet tradeable -- status below)"
                lines.append(
                    f"  - {r.get('symbol')}: {r.get('direction', 'n/a')} | status {r.get('status')} | "
                    f"score {r.get('score')} | moved {r.get('momentum_pct')}% "
                    f"({r.get('atr_normalized_move')} ATR){flag}"
                )
        lines.append("")

    at_risk = [m for m in best_markets if m.get("news_risk") and m.get("news_risk") != "none"]
    extended = [m for m in best_markets if m.get("status") == "EXTENDED"]
    if at_risk or extended:
        lines.append("WATCH OUT FOR")
        for m in at_risk:
            lines.append(f"  - {m.get('symbol')}: news risk ({m.get('news_risk')})")
        for m in extended:
            lines.append(f"  - {m.get('symbol')}: EXTENDED -- avoid chasing")

    return "\n".join(lines).rstrip()


class TradingAssistantClient:
    def __init__(self, settings: OllamaSettings, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        self.settings = settings
        self.timeout = timeout

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _chat(self, system_prompt: str, user_message: str, history: list[dict] | None = None) -> tuple[str, str | None]:
        """Calls Ollama's /api/chat (non-streaming). Returns (reply, error)
        -- reply is "" and error is set on any failure, mirroring
        app.ai.ollama_client's fail-safe convention exactly."""
        import requests

        if not self.settings.is_usable:
            return "", "Ollama isn't enabled/configured yet. Turn it on and set a host in AI Assistant settings."

        host = (self.settings.host or "").rstrip("/")
        messages = [{"role": "system", "content": system_prompt}]
        for turn in (history or [])[-10:]:  # keep the last 10 turns -- plenty of context, bounded prompt size
            messages.append(turn)
        messages.append({"role": "user", "content": user_message})

        try:
            resp = requests.post(
                f"{host}/api/chat",
                headers=self._headers(),
                json={"model": self.settings.model, "messages": messages, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            reply = (data.get("message") or {}).get("content", "")
            return reply, None
        except requests.exceptions.ConnectionError:
            return "", f"Couldn't reach Ollama at {host} (is `ollama serve` running?)."
        except requests.exceptions.Timeout:
            return "", f"Ollama at {host} didn't respond in time."
        except Exception as exc:
            return "", f"Ollama request failed: {exc}"

    def _chat_vision(
        self, system_prompt: str, user_message: str, image_b64: str, model: str | None = None,
    ) -> tuple[str, str | None]:
        """Same as _chat, but attaches one base64-encoded image to the user
        turn via Ollama's `images` field (the standard way Ollama's
        /api/chat accepts an image for a multimodal model -- see
        https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-images).
        `model` overrides self.settings.model for this call only -- vision
        needs a separate, explicitly multimodal model (e.g. llava,
        llama3.2-vision); the default text model can't see the image at
        all and most will simply ignore the `images` field or error."""
        import requests

        if not self.settings.is_usable:
            return "", "Ollama isn't enabled/configured yet. Turn it on and set a host in AI Assistant settings."

        host = (self.settings.host or "").rstrip("/")
        vision_model = model or getattr(self.settings, "vision_model", "") or "llava"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message, "images": [image_b64]},
        ]
        try:
            resp = requests.post(
                f"{host}/api/chat",
                headers=self._headers(),
                json={"model": vision_model, "messages": messages, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            reply = (data.get("message") or {}).get("content", "")
            return reply, None
        except requests.exceptions.ConnectionError:
            return "", f"Couldn't reach Ollama at {host} (is `ollama serve` running?)."
        except requests.exceptions.Timeout:
            return "", f"Ollama at {host} didn't respond in time (vision models can be slow -- try a smaller model)."
        except Exception as exc:
            return "", (
                f"Ollama vision request failed: {exc}. Make sure `{vision_model}` is a vision-capable "
                f"model that's actually been pulled (e.g. `ollama pull llava`)."
            )

    def analyze_chart_screenshot(self, image_b64: str, context: dict | None = None, extra_notes: str = "") -> tuple[str, str | None]:
        """Chart screenshot -> exact trading plan in Owen's format. `context`
        (from build_context) is optional extra market intelligence (macro
        bias, news) the deterministic scanner already knows about the
        instrument, if Owen's provided one; the image itself is the
        primary source of truth for what's actually on the chart."""
        import json

        system_prompt = PERSONAL_MODE_SYSTEM_PROMPT + "\n\n" + CHART_SCREENSHOT_INSTRUCTION
        parts = []
        if context:
            parts.append(
                "Additional market intelligence computed by the app (may or may not cover this "
                f"instrument):\n{json.dumps(_jsonable(context), indent=2)}"
            )
        if extra_notes.strip():
            parts.append(f"Owen's notes: {extra_notes.strip()}")
        parts.append("Analyze the attached chart screenshot and produce the trading plan now.")
        user_message = "\n\n".join(parts)
        return self._chat_vision(system_prompt, user_message, image_b64)

    def analyze_trade_screenshot(self, image_b64: str, extra_notes: str = "") -> tuple[str, str | None]:
        """Trade/PnL screenshot -> Session-Review-style breakdown of what
        went right or wrong, per PERSONAL_ASSISTANT_PROMPT's Trade Journal
        rules (process quality kept separate from PnL)."""
        system_prompt = PERSONAL_MODE_SYSTEM_PROMPT + "\n\n" + TRADE_SCREENSHOT_INSTRUCTION
        parts = []
        if extra_notes.strip():
            parts.append(f"Owen's notes on this trade: {extra_notes.strip()}")
        parts.append("Analyze the attached trade screenshot and produce the review now.")
        user_message = "\n\n".join(parts)
        return self._chat_vision(system_prompt, user_message, image_b64)

    def options_outlook(self, symbol: str, horizon: str, candidates: list[dict], notes: str = "") -> tuple[str, str | None]:
        """`candidates` is the deterministic call/put strike list from
        app.ai.options_outlook.build_candidates (spot, strikes, premiums,
        deltas, breakevens already computed) -- the model ranks and
        explains them, it never invents its own strikes or prices.
        `horizon` is a free label, e.g. "today" or "this week"."""
        import json

        user_message = (
            f"Symbol: {symbol}\nHorizon: {horizon}\n"
            f"Candidate calls/puts (all figures computed by Black-Scholes, not by you):\n"
            f"{json.dumps(_jsonable(candidates), indent=2)}\n"
        )
        if notes.strip():
            user_message += f"\nOwen's directional/context notes: {notes.strip()}\n"
        user_message += "\nProduce the Options Outlook now in the exact format specified."
        return self._chat(OPTIONS_MODE_SYSTEM_PROMPT, user_message)

    def ask(self, question: str, context: dict, mode: str = "personal", history: list[dict] | None = None) -> tuple[str, str | None]:
        system_prompt = PERSONAL_MODE_SYSTEM_PROMPT if mode == "personal" else T58_GROUP_SYSTEM_PROMPT
        import json
        user_message = (
            f"Current market intelligence (all facts below were computed by the app, not by you -- "
            f"treat them as ground truth):\n{json.dumps(_jsonable(context), indent=2)}\n\n"
            f"Owen's question: {question}"
        )
        return self._chat(system_prompt, user_message, history=history)

    def daily_brief(self, context: dict) -> tuple[str, str | None]:
        return self.ask(
            "Generate today's Daily Trading Plan in the exact format specified.",
            context, mode="personal",
        )

    def watchlist(self, context: dict) -> tuple[str, str | None]:
        return self.ask(
            "Generate my personal watchlist in the exact format specified, using only the symbols "
            "in watchlist_symbols and best_markets.",
            context, mode="personal",
        )

    def market_outlook(self, context: dict) -> tuple[str, str | None]:
        """Backs the AI Assistant tab's "GENERATE BASIC OUTLOOK" button --
        see MARKET_OUTLOOK_SYSTEM_PROMPT. Lighter/faster than daily_brief();
        doesn't require `mode` since there's only one outlook format."""
        import json
        user_message = (
            "Current market intelligence (all facts below were computed by the app, not by you -- "
            f"treat them as ground truth):\n{json.dumps(_jsonable(context), indent=2)}\n\n"
            "Produce the Basic Outlook now in the exact format specified."
        )
        return self._chat(MARKET_OUTLOOK_SYSTEM_PROMPT, user_message)

    def pre_trade_check(self, context: dict, symbol: str) -> tuple[str, str | None]:
        return self.ask(
            f"Run the Pre-Trade Check for {symbol} using the exact format specified. "
            f"Do not manipulate the framework to justify a trade -- if conditions are missing, say WAIT or PASS.",
            context, mode="personal",
        )

    def session_review(self, context: dict, trades_summary: str) -> tuple[str, str | None]:
        return self.ask(
            f"Generate the End-of-Day Session Review in the exact format specified, based on these trades/notes "
            f"Owen provided:\n{trades_summary}",
            context, mode="personal",
        )
