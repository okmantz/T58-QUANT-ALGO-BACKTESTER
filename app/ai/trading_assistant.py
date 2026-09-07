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
