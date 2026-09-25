"""T58 Capability Reference -- the single source of truth an LLM should be
grounded against before claiming any T58 feature is or isn't supported.

Why this exists: Roboquant's own changelog documents them hitting and
fixing exactly this bug class -- their AI assistant told a user a feature
wasn't supported when it actually was, because the assistant's own belief
about the engine had gone stale relative to the real parser. Auditing this
app's equivalent surface (app.ai.strategy_generator's hand-typed PineScript/
MQL5 "STRICT OUTPUT CONTRACT" strings) found that exact bug already
present here: app.strategy.pinescript's real parser has supported
ta.atr, ta.vwap, ta.highest, ta.lowest, ta.stdev, and destructured ta.macd,
plus ATR-multiple-based T58_SL_ATR_MULT/T58_TP_ATR_MULT/T58_ATR_PERIOD
stop/target directives, for a while now -- and app.strategy.mql5's real
parser has supported iATR, iBands, iHighest, and iLowest -- none of which
the Strategy Generator's static contract text ever mentioned. Every
PineScript/MQL5 strategy that feature has ever drafted was silently
restricted to a stale subset of what the engine can actually run, and the
chat assistant had no way to answer "does T58 support X?" other than
guessing.

The fix here is structural, not a one-time text correction: every function
below INTROSPECTS the real parser modules' own compiled regex patterns
(app.strategy.pinescript / app.strategy.mql5), the real indicator library
(app.strategy.indicators), and the real manual/JSON condition-builder
source (app.strategy.manual) at call time, rather than repeating a
hand-typed list that can drift again the next time someone adds a new
ta.*/i*/indicator/condition-kind without remembering to update a separate
prompt string somewhere else. Whenever those modules gain new support,
this reference -- and therefore the Strategy Generator's contract text and
the chat assistant's grounding note -- picks it up automatically, with
zero further maintenance and no second list to keep in sync.

Cheap and pure: everything here is regex/introspection over already-
imported modules, no I/O, no network. Cached at module level (each
process's parser source doesn't change while it's running) so this is
safe to call on every chat message or generation request without adding
meaningful latency.
"""
from __future__ import annotations

import inspect
import re
from functools import lru_cache

_MANUAL_KIND_RE = re.compile(r'kind\s*==\s*"([a-z0-9_]+)"')
_TA_ALTERNATION_RE = re.compile(r"\(([a-z_]+(?:\|[a-z_]+)+)\)")


@lru_cache(maxsize=1)
def pinescript_capabilities() -> dict:
    """Extracted straight from app.strategy.pinescript's own compiled
    regexes (_TA_CALL_RE, _TA_ATR_RE, _TA_VWAP_RE, _TA_HIGHEST_RE,
    _TA_LOWEST_RE, _TA_STDEV_RE, _DESTRUCTURE_MACD_RE, the SL/TP directive
    regexes, _PRICE_ALIASES) -- never hand-typed, so it can't go stale
    relative to the real parser the way strategy_generator's old static
    contract text did."""
    from app.strategy import pinescript as ps

    ta_funcs: set[str] = set()
    m = _TA_ALTERNATION_RE.search(ps._TA_CALL_RE.pattern)
    if m:
        ta_funcs.update(m.group(1).split("|"))
    ta_funcs.update({"crossover", "crossunder"})  # _CROSS_CALL_RE
    if hasattr(ps, "_DESTRUCTURE_MACD_RE"):
        ta_funcs.add("macd")
    if hasattr(ps, "_TA_ATR_RE"):
        ta_funcs.add("atr")
    if hasattr(ps, "_TA_VWAP_RE"):
        ta_funcs.add("vwap")
    if hasattr(ps, "_TA_HIGHEST_RE"):
        ta_funcs.add("highest")
    if hasattr(ps, "_TA_LOWEST_RE"):
        ta_funcs.add("lowest")
    if hasattr(ps, "_TA_STDEV_RE"):
        ta_funcs.add("stdev")

    directives = {"T58_SL_PIPS", "T58_TP_PIPS"}
    if hasattr(ps, "_SL_ATR_DIRECTIVE_RE"):
        directives.add("T58_SL_ATR_MULT")
    if hasattr(ps, "_TP_ATR_DIRECTIVE_RE"):
        directives.add("T58_TP_ATR_MULT")
    if hasattr(ps, "_ATR_PERIOD_DIRECTIVE_RE"):
        directives.add("T58_ATR_PERIOD")

    return {
        "ta_functions": sorted(ta_funcs),
        "price_references": sorted(getattr(ps, "_PRICE_ALIASES", set())),
        "directives": sorted(directives),
    }


@lru_cache(maxsize=1)
def mql5_capabilities() -> dict:
    """Extracted straight from app.strategy.mql5's own compiled regexes
    and _MODE_TO_FUNC -- same never-hand-typed guarantee as
    pinescript_capabilities()."""
    from app.strategy import mql5

    indicators: set[str] = set()
    if hasattr(mql5, "_IMA_RE"):
        indicators.add("iMA")
    if hasattr(mql5, "_IRSI_RE"):
        indicators.add("iRSI")
    if hasattr(mql5, "_IATR_RE"):
        indicators.add("iATR")
    if hasattr(mql5, "_IBANDS_RE"):
        indicators.add("iBands")
    if hasattr(mql5, "_IHIGHEST_RE"):
        indicators.add("iHighest")
    if hasattr(mql5, "_ILOWEST_RE"):
        indicators.add("iLowest")

    ma_modes = sorted(getattr(mql5, "_MODE_TO_FUNC", {}).keys())

    directives = {"T58_SL_PIPS", "T58_TP_PIPS"}
    if hasattr(mql5, "_SL_ATR_DIRECTIVE_RE"):
        directives.add("T58_SL_ATR_MULT")
    if hasattr(mql5, "_TP_ATR_DIRECTIVE_RE"):
        directives.add("T58_TP_ATR_MULT")
    if hasattr(mql5, "_ATR_PERIOD_DIRECTIVE_RE"):
        directives.add("T58_ATR_PERIOD")

    return {
        "indicator_functions": sorted(indicators),
        "ma_modes": ma_modes,
        "directives": sorted(directives),
    }


@lru_cache(maxsize=1)
def python_indicator_library() -> list[str]:
    """Every public, callable indicator implementation in
    app.strategy.indicators -- what a hand-written or AI-drafted Python
    strategy can compute inline. The Python output contract itself stays
    an open pandas/numpy sandbox (not a fixed whitelist -- unlike
    PineScript/MQL5, arbitrary pandas/numpy expressions already work), but
    this list is useful grounding context for chat/generation either way:
    "here's what already exists in this codebase, port from it instead of
    reinventing it.\""""
    from app.strategy import indicators as ind

    return sorted(
        name for name, obj in inspect.getmembers(ind, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == ind.__name__
    )


@lru_cache(maxsize=1)
def manual_condition_kinds() -> list[str]:
    """The `kind` values app.strategy.manual's condition builder actually
    dispatches on -- extracted from its own source text (the dispatch is a
    plain if/elif chain, not a lookup table), so a new condition kind
    added there is picked up here automatically the next time this
    process starts, with no second list to keep in sync."""
    from app.strategy import manual

    source = inspect.getsource(manual)
    return sorted(set(_MANUAL_KIND_RE.findall(source)))


def build_pinescript_contract_text() -> str:
    """Replaces app.ai.strategy_generator's old hand-typed
    _PINESCRIPT_CONTRACT string -- generated from
    pinescript_capabilities() so it can never silently under- or
    over-state what the real parser supports again."""
    caps = pinescript_capabilities()
    param_funcs = [f for f in caps["ta_functions"] if f not in ("crossover", "crossunder", "atr", "vwap", "macd")]
    ta_list = ", ".join(f"ta.{f}(src, len)" for f in param_funcs)
    return f"""\
Output PineScript v5. This app's parser only understands a RESTRICTED
SUBSET -- anything outside it fails to load. You may ONLY use:
- Price references: {", ".join(caps["price_references"])}
- x = input.int(20, ...) / input.float(1.5, ...) -- becomes a constant
  using the given default value; no other input.* types
- x = {ta_list}
- x = ta.atr(len) -- no source argument (always operates on the bar's OHLC)
- x = ta.vwap() -- no arguments (resets every session)
- x = ta.crossover(a, b), ta.crossunder(a, b)
- [macdLine, signalLine, histLine] = ta.macd(src, fastLen, slowLen, signalLen)
- No other ta.* function -- anything not listed above fails to parse
- Boolean rule variables built from comparisons/and/or/not over the above,
  e.g. longCondition = ta.crossover(fast, slow) and rsiVal < 70
- Entries, inline or inside an if block:
    strategy.entry("Long", strategy.long, when=longCondition)
    if longCondition
        strategy.entry("Long", strategy.long)
- Exits: strategy.close("Long", when=exitLongCondition)
- Stop-loss/take-profit as special directive comments (not strategy.exit
  price offsets) -- either fixed pips:
    // T58_SL_PIPS=20
    // T58_TP_PIPS=40
  or ATR-multiple based:
    // T58_ATR_PERIOD=14
    // T58_SL_ATR_MULT=2.0
    // T58_TP_ATR_MULT=3.0
Do NOT use: custom functions, arrays/matrices, security()/multi-timeframe
requests, repainting constructs, plotting, alerts, or any ta.* function
not listed above -- all of these fail to parse.
"""


def build_mql5_contract_text() -> str:
    """Replaces app.ai.strategy_generator's old hand-typed
    _MQL5_CONTRACT string -- generated from mql5_capabilities()."""
    caps = mql5_capabilities()
    modes = " / ".join(caps["ma_modes"])
    indicators = ", ".join(caps["indicator_functions"])
    return f"""\
Output MQL5 Expert Advisor source. This app's parser only understands a
RESTRICTED SUBSET -- anything outside it fails to load. You may ONLY use:
- Direct-value indicator calls (the simplified/legacy calling style):
    double fastMA = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_SMA, PRICE_CLOSE);
    double slowMA = iMA(_Symbol, PERIOD_CURRENT, 30, 0, MODE_EMA, PRICE_CLOSE);
    double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
    double atrVal = iATR(_Symbol, PERIOD_CURRENT, 14);
    double bandsMid = iBands(_Symbol, PERIOD_CURRENT, 20, 0, 2.0, PRICE_CLOSE, MODE_MAIN);
    double hh = iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, 20, 0);
    double ll = iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, 20, 0);
  (only {modes} as MA modes; only {indicators} as indicator calls)
- Boolean conditions with C-style operators: > < >= <= == != && || !
- if (condition) {{ ... }} or single-statement if (condition) statement;
- Entries inside a condition's guard: trade.Buy(...) / trade.Sell(...)
  (or OrderSend(..., ORDER_TYPE_BUY/ORDER_TYPE_SELL, ...))
- Exits inside a condition's guard: trade.PositionClose(...) / OrderClose(...)
- Stop-loss/take-profit as special directive comments -- either fixed pips:
    // T58_SL_PIPS=20
    // T58_TP_PIPS=40
  or ATR-multiple based:
    // T58_ATR_PERIOD=14
    // T58_SL_ATR_MULT=2.0
    // T58_TP_ATR_MULT=3.0
Do NOT use: CopyBuffer()-based indicator handles, custom indicators,
arrays/structs, multi-symbol/multi-timeframe logic, trailing stops, or any
indicator beyond {indicators} -- all of these fail to parse.
"""


def chat_capability_note() -> dict:
    """Compact, JSON-safe summary for
    app.ai.trading_assistant.build_context's `engine_capabilities` field --
    kept small on purpose (this rides along on every chat/daily-brief/
    outlook request, so it stays a short fact block, not a full contract
    dump). Ground every "is X supported" answer against this instead of
    guessing."""
    ps = pinescript_capabilities()
    mq = mql5_capabilities()
    return {
        "python_indicators_available": python_indicator_library(),
        "manual_json_condition_types": manual_condition_kinds(),
        "pinescript_ta_functions": [f"ta.{f}" for f in ps["ta_functions"]],
        "pinescript_directives": ps["directives"],
        "mql5_indicator_functions": mq["indicator_functions"],
        "mql5_directives": mq["directives"],
    }
