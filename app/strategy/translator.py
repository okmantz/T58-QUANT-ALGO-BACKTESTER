"""
Universal Strategy Translator -- Manual Strategy Builder config (T58's
internal signal representation; see app/strategy/manual.py's docstring and
app/strategy/base.py's StrategyResult) -> clean, standalone PineScript v5
and MQL5 source.

Why this direction, not full bidirectional code-gen: T58 already parses
PineScript and MQL5 INTO signals (app/strategy/pinescript.py,
app/strategy/mql5.py) for backtesting -- that half of the problem is
solved. What was missing is the other direction: a strategy discovered or
optimized inside T58 (the Manual Strategy Builder, Evolution Lab, the
walk-forward-aware GA, Quick Optimize, Iterative Refinement -- every one
of which operates on Manual Strategy Builder config dicts, per
app/optimize/refinement.py and app/optimize/walkforward_ga.py) had no
path to TradingView or a live MT5 account except hand-porting the JSON
logic by eye. This module closes that gap: it walks the SAME
entry_conditions / exit_conditions / risk_management structure
app/strategy/manual.py itself evaluates, and emits equivalent source in
each target language.

Deliberately one-directional and honest about its supported subset (same
philosophy as the Pine/MQL5 *parsers*: fail loudly on anything not
supported rather than silently emit code that looks right and isn't):

  Supported operand kinds: price fields (open/high/low/close), numeric
  constants, sma/ema/wma/rsi/atr, macd/macd_signal/macd_histogram,
  bollinger_mid/upper/lower, highest_high/lowest_low, candle_direction --
  i.e. exactly the indicator kinds app/strategy/indicators.py's
  build_indicator_series() can compute from a single native platform
  function call, so the emitted code isn't secretly re-implementing
  something T58-specific.

  Supported operators: > >= < <= == != and "crosses above"/"crosses
  below" (rendered as ta.crossover/ta.crossunder in Pine -- the SAME
  primitive app/strategy/pinescript.py's own parser recognizes, so a
  translated-then-re-uploaded file still loads; rendered as an explicit
  current-vs-prior-bar comparison in MQL5, since T58's MQL5 parser has no
  crossover primitive to match).

  Supported risk management: fixed-pips or ATR-multiple stop/target
  (rendered as REAL broker-facing exit orders, not just a comment
  directive), max-bars-in-trade, a single daily clock-time exit, and
  long/short/both direction restriction.

  NOT auto-coded (each is instead surfaced as a clearly labeled TODO
  comment naming the exact configured values, non-fatal -- see
  _UNSUPPORTED_RISK_NOTE): trailing stop and break-even. Both require
  stateful, position-lifecycle-aware logic (a mutable stop level that
  moves as the trade develops) that is easy to get subtly wrong, and a
  subtly wrong stop on a live account is exactly the failure mode this
  codebase is built to avoid (see app/strategy/pinescript.py's own
  "fail loudly rather than silently produce an inaccurate ... result"
  principle) -- so these are flagged for the trader to wire up natively
  in their platform's own trailing-stop/break-even mechanism instead of
  guessing at translated stateful code.

  NOT supported at all (raises TranslationError naming the construct,
  before any code is emitted): the Manual Builder's SMC-style/session/
  regime operand kinds (liquidity_sweep, break_of_structure,
  change_of_character, fair_value_gap, order_block, session_high/low,
  previous_day_*, opening_range_*, time_of_day, atr_regime/
  volatility_regime), and expression-string strategies (long_entry="close
  > sma_20") rather than the structured entry_conditions/exit_conditions
  format -- these don't reduce to a single-line indicator comparison a
  retail platform's own native functions can express directly, and a
  hand-port is safer than a confidently-wrong one.

Both outputs also carry the T58_SL_PIPS / T58_SL_ATR_MULT-style directive
comments the Pine/MQL5 *parsers* already recognize (see those modules'
docstrings), so a strategy round-tripped back into T58 later (after any
manual edits made on TradingView/MT5) still has its stop/target settings
recognized automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TranslationError(Exception):
    """Raised when a Manual Strategy Builder config uses a construct this
    translator does not know how to safely re-render in the target
    language. Raised BEFORE any output file is produced -- there is no
    such thing as a partially-translated file from this module."""


_PRICE_FIELDS = {"open", "high", "low", "close", "volume"}
_SIMPLE_INDICATOR_KINDS = {"sma", "ema", "wma", "rsi", "atr", "highest_high", "lowest_low"}
_MULTI_OUTPUT_KINDS = {"macd", "macd_signal", "macd_histogram", "bollinger_mid", "bollinger_upper", "bollinger_lower"}
_SUPPORTED_INDICATOR_KINDS = _SIMPLE_INDICATOR_KINDS | _MULTI_OUTPUT_KINDS | {"candle_direction"}

_UNSUPPORTED_KIND_HINT = {
    "liquidity_sweep": "structural liquidity-sweep detection",
    "break_of_structure": "market-structure break detection",
    "bos": "market-structure break detection",
    "change_of_character": "structural character-shift detection",
    "choch": "structural character-shift detection",
    "fair_value_gap": "fair-value-gap detection",
    "fvg": "fair-value-gap detection",
    "order_block": "order-block detection",
    "session_high": "session-relative level tracking",
    "session_low": "session-relative level tracking",
    "previous_day_high": "prior-day level tracking",
    "previous_day_low": "prior-day level tracking",
    "previous_day_close": "prior-day level tracking",
    "opening_range_high": "opening-range level tracking",
    "opening_range_low": "opening-range level tracking",
    "time_of_day": "session time-window filtering",
    "atr_regime": "ATR-regime classification",
    "volatility_regime": "volatility-regime classification",
    "swing_high": "confirmed swing-point detection",
    "swing_low": "confirmed swing-point detection",
}


@dataclass(frozen=True)
class _Operand:
    kind: str  # "constant" | "price" | one of _SUPPORTED_INDICATOR_KINDS
    value: float | None = None
    field: str = "close"
    period: int = 14

    def key(self) -> tuple:
        return (self.kind, self.field, self.period)


def _parse_operand(raw: Any, side: str) -> _Operand:
    if isinstance(raw, (int, float)):
        return _Operand(kind="constant", value=float(raw))
    if raw is None:
        raise TranslationError(f"The {side} side of a condition is missing a value.")
    if isinstance(raw, str):
        name = raw.lower().strip()
        if name in _PRICE_FIELDS:
            return _Operand(kind="price", field=name)
        try:
            return _Operand(kind="constant", value=float(raw))
        except ValueError as exc:
            raise TranslationError(f"Unknown {side}-side operand '{raw}'.") from exc
    if not isinstance(raw, dict):
        raise TranslationError(f"Invalid {side}-side operand: {raw!r}")

    kind = str(raw.get("type", raw.get("source", "close"))).lower().strip()
    field_name = str(raw.get("field", "close")).lower().strip()
    try:
        period = max(int(raw.get("period", 14) or 14), 1)
    except (TypeError, ValueError):
        period = 14

    if kind in {"value", "constant", "number"}:
        try:
            return _Operand(kind="constant", value=float(raw.get("value", 0)))
        except (TypeError, ValueError) as exc:
            raise TranslationError("A numeric condition value is required.") from exc
    if kind in {"price", "open", "high", "low", "close", "volume"}:
        col = field_name if kind == "price" else kind
        if col not in _PRICE_FIELDS:
            raise TranslationError(f"Unknown price field '{col}'.")
        return _Operand(kind="price", field=col)
    if kind in _SUPPORTED_INDICATOR_KINDS:
        return _Operand(kind=kind, field=field_name, period=period)

    hint = _UNSUPPORTED_KIND_HINT.get(kind)
    if hint:
        raise TranslationError(
            f"'{kind}' ({hint}) has no direct equivalent as a single native indicator call on a "
            "retail platform -- port this condition by hand rather than risk a silently-wrong "
            "translation. Everything else in this strategy can still be translated separately."
        )
    raise TranslationError(f"Unsupported condition source '{kind}' for translation.")


def _normalize_operator(op: str) -> str:
    op = op.strip().lower()
    if op in {">", "greater than", "gt"}:
        return ">"
    if op in {">=", "greater than or equal", "gte"}:
        return ">="
    if op in {"<", "less than", "lt"}:
        return "<"
    if op in {"<=", "less than or equal", "lte"}:
        return "<="
    if op in {"==", "equal to", "equals", "eq"}:
        return "=="
    if op in {"!=", "not equal", "neq"}:
        return "!="
    if op in {"cross above", "crosses above"}:
        return "crosses_above"
    if op in {"cross below", "crosses below"}:
        return "crosses_below"
    raise TranslationError(f"Unsupported condition operator '{op}' for translation.")


@dataclass(frozen=True)
class _Condition:
    left: _Operand
    operator: str
    right: _Operand


def _parse_condition(raw: dict) -> _Condition:
    if not isinstance(raw, dict):
        raise TranslationError(f"Invalid condition: {raw!r}")
    left = _parse_operand(raw.get("left", raw.get("source", "close")), "left")
    right = _parse_operand(raw.get("right", raw.get("value", 0)), "right")
    operator = _normalize_operator(str(raw.get("operator", ">")))
    return _Condition(left=left, operator=operator, right=right)


@dataclass(frozen=True)
class _ConditionGroup:
    conditions: tuple
    connectors: tuple  # "AND"/"OR" joining conditions[i] and conditions[i+1]


def _parse_group(conditions_raw: list | None, connectors_raw: list | None) -> _ConditionGroup:
    conditions = tuple(_parse_condition(c) for c in (conditions_raw or []))
    connectors = tuple(str(c).upper() for c in (connectors_raw or []))
    return _ConditionGroup(conditions=conditions, connectors=connectors)


@dataclass
class ParsedStrategy:
    name: str
    direction: str  # "long" | "short" | "both"
    long_entry: _ConditionGroup
    short_entry: _ConditionGroup
    long_exit: _ConditionGroup
    short_exit: _ConditionGroup
    max_bars_in_trade: int | None
    time_exit: str | None
    stop_type: str | None      # "fixed" | "atr" | None
    stop_value: float | None
    stop_atr_period: int
    target_type: str | None
    target_value: float | None
    target_atr_period: int
    trailing_enabled: bool
    trailing_value: float | None
    trailing_atr_period: int
    breakeven_enabled: bool
    breakeven_trigger_r: float | None


def parse_manual_config(config: dict) -> ParsedStrategy:
    """Validates and normalizes a Manual Strategy Builder config into the
    structured form the two code generators below both consume. Raises
    TranslationError immediately on anything unsupported -- callers
    should not catch-and-continue partway through, since there is no
    partial output from this module."""
    if not isinstance(config, dict):
        raise TranslationError("Manual strategy configuration must be a dictionary.")
    entries = config.get("entry_conditions")
    if not entries:
        raise TranslationError(
            "Only Manual Strategy Builder configs using the visual-builder 'entry_conditions' format "
            "can be translated -- expression-string strategies (long_entry='close > sma_20') aren't a "
            "structured representation this translator can safely re-render construct-by-construct."
        )
    exits = config.get("exit_conditions", {}) or {}
    rm = config.get("risk_management", {}) or {}

    direction = str(config.get("market", {}).get("direction", config.get("direction", "Both"))).lower()
    if direction not in {"long", "short", "both"}:
        direction = "both"

    time_cfg = rm.get("time_based_exit", {}) or {}
    trailing = rm.get("trailing_stop", {}) or {}
    breakeven = rm.get("break_even", {}) or {}

    max_bars = rm.get("max_bars_in_trade")
    try:
        max_bars = max(int(max_bars), 1) if max_bars else None
    except (TypeError, ValueError):
        max_bars = None

    stop_type = str(rm.get("stop_type", "")).lower() or None
    if stop_type not in {"fixed", "atr"} or rm.get("stop_value") in (None, ""):
        stop_type = None
    target_type = str(rm.get("target_type", "")).lower() or None
    if target_type not in {"fixed", "atr"} or rm.get("target_value") in (None, ""):
        target_type = None

    return ParsedStrategy(
        name=str(config.get("name", "T58 Strategy")).strip() or "T58 Strategy",
        direction=direction,
        long_entry=_parse_group(entries.get("long", []), entries.get("long_connectors")),
        short_entry=_parse_group(entries.get("short", []), entries.get("short_connectors")),
        long_exit=_parse_group(exits.get("long", []), exits.get("long_connectors")),
        short_exit=_parse_group(exits.get("short", []), exits.get("short_connectors")),
        max_bars_in_trade=max_bars,
        time_exit=str(time_cfg["time"]) if time_cfg.get("enabled") and time_cfg.get("time") else None,
        stop_type=stop_type,
        stop_value=float(rm["stop_value"]) if stop_type else None,
        stop_atr_period=max(int(rm.get("stop_atr_period", 14) or 14), 1),
        target_type=target_type,
        target_value=float(rm["target_value"]) if target_type else None,
        target_atr_period=max(int(rm.get("target_atr_period", 14) or 14), 1),
        trailing_enabled=bool(trailing.get("enabled")) and trailing.get("value") not in (None, ""),
        trailing_value=float(trailing["value"]) if trailing.get("value") not in (None, "") else None,
        trailing_atr_period=max(int(trailing.get("atr_period", 14) or 14), 1),
        breakeven_enabled=bool(breakeven.get("enabled")) and breakeven.get("trigger_r") not in (None, ""),
        breakeven_trigger_r=float(breakeven["trigger_r"]) if breakeven.get("trigger_r") not in (None, "") else None,
    )


def _all_operands(parsed: ParsedStrategy):
    for group in (parsed.long_entry, parsed.short_entry, parsed.long_exit, parsed.short_exit):
        for cond in group.conditions:
            yield cond.left
            yield cond.right


def _collect_indicators(parsed: ParsedStrategy) -> dict[tuple, _Operand]:
    seen: dict[tuple, _Operand] = {}
    for op in _all_operands(parsed):
        if op.kind not in ("constant", "price"):
            seen.setdefault(op.key(), op)
    return seen


def _fmt_num(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _pine_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _risk_todo_lines(parsed: ParsedStrategy, comment_prefix: str) -> list[str]:
    lines: list[str] = []
    if parsed.trailing_enabled:
        lines.append(
            f"{comment_prefix} TODO(T58): trailing stop not auto-translated -- configure a "
            f"{_fmt_num(parsed.trailing_value)} x ATR({parsed.trailing_atr_period}) trailing stop "
            "natively on this platform."
        )
    if parsed.breakeven_enabled:
        lines.append(
            f"{comment_prefix} TODO(T58): break-even not auto-translated -- move the stop to entry "
            f"once open profit reaches {_fmt_num(parsed.breakeven_trigger_r)}R natively on this platform."
        )
    return lines


# ---------------------------------------------------------------------------
# PineScript v5 generation
# ---------------------------------------------------------------------------

def _pine_source(field_name: str) -> str:
    return field_name if field_name in _PRICE_FIELDS else "close"


def _pine_declare_indicators(indicators: dict[tuple, _Operand]) -> tuple[list[str], dict[tuple, str]]:
    lines: list[str] = []
    var_map: dict[tuple, str] = {}

    macd_fields = sorted({op.field for op in indicators.values() if op.kind in ("macd", "macd_signal", "macd_histogram")})
    for fld in macd_fields:
        src = _pine_source(fld)
        base = f"macd_{fld}"
        lines.append(f"[{base}_line, {base}_signal, {base}_hist] = ta.macd({src}, 12, 26, 9)")
        var_map[("macd", fld, 0)] = f"{base}_line"
        var_map[("macd_signal", fld, 0)] = f"{base}_signal"
        var_map[("macd_histogram", fld, 0)] = f"{base}_hist"

    bb_groups = sorted({(op.field, op.period) for op in indicators.values()
                         if op.kind in ("bollinger_mid", "bollinger_upper", "bollinger_lower")})
    for fld, period in bb_groups:
        src = _pine_source(fld)
        base = f"bb_{fld}_{period}"
        lines.append(f"[{base}_mid, {base}_upper, {base}_lower] = ta.bb({src}, {period}, 2.0)")
        var_map[("bollinger_mid", fld, period)] = f"{base}_mid"
        var_map[("bollinger_upper", fld, period)] = f"{base}_upper"
        var_map[("bollinger_lower", fld, period)] = f"{base}_lower"

    for key, op in indicators.items():
        if op.kind in ("macd", "macd_signal", "macd_histogram", "bollinger_mid", "bollinger_upper", "bollinger_lower"):
            continue
        src = _pine_source(op.field)
        name = f"{op.kind}_{op.period}_{op.field}"
        if op.kind == "sma":
            lines.append(f"{name} = ta.sma({src}, {op.period})")
        elif op.kind == "ema":
            lines.append(f"{name} = ta.ema({src}, {op.period})")
        elif op.kind == "wma":
            lines.append(f"{name} = ta.wma({src}, {op.period})")
        elif op.kind == "rsi":
            lines.append(f"{name} = ta.rsi({src}, {op.period})")
        elif op.kind == "atr":
            lines.append(f"{name} = ta.atr({op.period})")
        elif op.kind == "highest_high":
            lines.append(f"{name} = ta.highest(high, {op.period})")
        elif op.kind == "lowest_low":
            lines.append(f"{name} = ta.lowest(low, {op.period})")
        elif op.kind == "candle_direction":
            lines.append(f"{name} = close > open ? 1 : close < open ? -1 : 0")
        else:  # pragma: no cover -- guarded by _SUPPORTED_INDICATOR_KINDS
            raise TranslationError(f"Unhandled indicator kind '{op.kind}' during PineScript generation.")
        var_map[key] = name

    return lines, var_map


def _pine_render_operand(op: _Operand, var_map: dict[tuple, str]) -> str:
    if op.kind == "constant":
        return _fmt_num(op.value)
    if op.kind == "price":
        return _pine_source(op.field)
    return var_map[op.key()]


def _pine_render_condition(cond: _Condition, var_map: dict[tuple, str]) -> str:
    left = _pine_render_operand(cond.left, var_map)
    right = _pine_render_operand(cond.right, var_map)
    if cond.operator == "crosses_above":
        return f"ta.crossover({left}, {right})"
    if cond.operator == "crosses_below":
        return f"ta.crossunder({left}, {right})"
    return f"{left} {cond.operator} {right}"


def _pine_render_group(group: _ConditionGroup, var_map: dict[tuple, str]) -> str:
    if not group.conditions:
        return "false"
    parts = [_pine_render_condition(c, var_map) for c in group.conditions]
    expr = parts[0]
    for i, connector in enumerate(group.connectors):
        if i + 1 < len(parts):
            op = "and" if connector == "AND" else "or"
            expr = f"({expr}) {op} ({parts[i + 1]})"
    return expr


def to_pinescript(config: dict) -> str:
    """Renders a Manual Strategy Builder config as a standalone PineScript
    v5 strategy() script, ready to paste into TradingView's Pine Editor."""
    parsed = parse_manual_config(config)
    indicators = _collect_indicators(parsed)
    indicator_lines, var_map = _pine_declare_indicators(indicators)

    allow_long = parsed.direction in ("long", "both")
    allow_short = parsed.direction in ("short", "both")

    lines: list[str] = []
    lines.append("//@version=5")
    lines.append(
        f'strategy("{_pine_escape(parsed.name)}", overlay=true, process_orders_on_close=true, '
        "default_qty_type=strategy.percent_of_equity, default_qty_value=100)"
    )
    lines.append("")
    lines.append("// Generated by the T58 Universal Strategy Translator from a Manual Strategy")
    lines.append("// Builder config. Review before trading real or paper capital.")
    if parsed.stop_type == "fixed" or parsed.target_type == "fixed":
        lines.append(
            "// T58_SL_PIPS / T58_TP_PIPS below is expressed in the SAME unit as the Manual"
        )
        lines.append("// Strategy Builder's stop/target values -- set pipSize to match your instrument.")
    lines.append("")
    lines.extend(indicator_lines)
    if indicator_lines:
        lines.append("")

    lines.append(f"longEntryCond = {_pine_render_group(parsed.long_entry if allow_long else _ConditionGroup((), ()), var_map)}")
    lines.append(f"shortEntryCond = {_pine_render_group(parsed.short_entry if allow_short else _ConditionGroup((), ()), var_map)}")
    lines.append(f"longExitCond = {_pine_render_group(parsed.long_exit, var_map)}")
    lines.append(f"shortExitCond = {_pine_render_group(parsed.short_exit, var_map)}")
    lines.append("")

    if parsed.max_bars_in_trade:
        lines.append(f"// T58_MAX_BARS_IN_TRADE={parsed.max_bars_in_trade}")
        lines.append("var int barsInTrade = 0")
        lines.append("barsInTrade := strategy.position_size != 0 ? barsInTrade + 1 : 0")
        lines.append(f"maxBarsExit = barsInTrade >= {parsed.max_bars_in_trade}")
        lines.append("longExitCond := longExitCond or maxBarsExit")
        lines.append("shortExitCond := shortExitCond or maxBarsExit")
        lines.append("")

    if parsed.time_exit:
        try:
            hh, mm = (int(x) for x in parsed.time_exit.split(":")[:2])
        except ValueError:
            raise TranslationError(f"Time-based exit '{parsed.time_exit}' must be in HH:MM format.")
        lines.append(f"// T58_TIME_EXIT={parsed.time_exit}")
        lines.append(f"timeExitCond = (hour > {hh}) or (hour == {hh} and minute >= {mm})")
        lines.append("longExitCond := longExitCond or timeExitCond")
        lines.append("shortExitCond := shortExitCond or timeExitCond")
        lines.append("")

    lines.append("if longEntryCond")
    lines.append('    strategy.entry("Long", strategy.long)')
    lines.append("if shortEntryCond")
    lines.append('    strategy.entry("Short", strategy.short)')
    lines.append('strategy.close("Long", when=longExitCond)')
    lines.append('strategy.close("Short", when=shortExitCond)')
    lines.append("")

    if parsed.stop_type or parsed.target_type:
        lines.append('pipSize = input.float(0.0001, title="Pip Size (match your instrument, e.g. 0.01 for gold)")')
        if parsed.stop_type == "fixed":
            lines.append(f"// T58_SL_PIPS={_fmt_num(parsed.stop_value)}")
            lines.append(f"slDist = {_fmt_num(parsed.stop_value)} * pipSize")
        elif parsed.stop_type == "atr":
            lines.append(f"// T58_SL_ATR_MULT={_fmt_num(parsed.stop_value)}")
            lines.append(f"// T58_ATR_PERIOD={parsed.stop_atr_period}")
            lines.append(f"slAtr = ta.atr({parsed.stop_atr_period})")
            lines.append(f"slDist = slAtr * {_fmt_num(parsed.stop_value)}")
        if parsed.target_type == "fixed":
            lines.append(f"// T58_TP_PIPS={_fmt_num(parsed.target_value)}")
            lines.append(f"tpDist = {_fmt_num(parsed.target_value)} * pipSize")
        elif parsed.target_type == "atr":
            lines.append(f"// T58_TP_ATR_MULT={_fmt_num(parsed.target_value)}")
            lines.append(f"// T58_ATR_PERIOD={parsed.target_atr_period}")
            lines.append(f"tpAtr = ta.atr({parsed.target_atr_period})")
            lines.append(f"tpDist = tpAtr * {_fmt_num(parsed.target_value)}")

        sl_arg = "strategy.position_avg_price - slDist" if parsed.stop_type else "na"
        sl_short_arg = "strategy.position_avg_price + slDist" if parsed.stop_type else "na"
        tp_arg = "strategy.position_avg_price + tpDist" if parsed.target_type else "na"
        tp_short_arg = "strategy.position_avg_price - tpDist" if parsed.target_type else "na"
        lines.append(f'strategy.exit("Long Exit", from_entry="Long", stop={sl_arg}, limit={tp_arg})')
        lines.append(f'strategy.exit("Short Exit", from_entry="Short", stop={sl_short_arg}, limit={tp_short_arg})')
        lines.append("")

    lines.extend(_risk_todo_lines(parsed, "//"))
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# MQL5 generation
# ---------------------------------------------------------------------------

_MQL5_KIND_INFO = {
    "sma": ("iMA", "MODE_SMA"), "ema": ("iMA", "MODE_EMA"), "wma": ("iMA", "MODE_LWMA"),
    "rsi": ("iRSI", None), "atr": ("iATR", None),
}


def _mql5_price_call(field_name: str, shift: str) -> str:
    fn = {"open": "iOpen", "high": "iHigh", "low": "iLow", "close": "iClose"}.get(field_name, "iClose")
    return f"{fn}(_Symbol, PERIOD_CURRENT, {shift})"


def _mql5_declare_indicators(indicators: dict[tuple, _Operand]) -> tuple[list[str], list[str], list[str], dict[tuple, str]]:
    """Returns (global declarations, OnInit handle-creation lines,
    OnTick CopyBuffer lines, {operand_key: array_variable_name})."""
    globals_: list[str] = []
    init_lines: list[str] = []
    tick_lines: list[str] = []
    var_map: dict[tuple, str] = {}

    macd_fields = sorted({op.field for op in indicators.values() if op.kind in ("macd", "macd_signal", "macd_histogram")})
    for fld in macd_fields:
        handle = f"h_macd_{fld}"
        globals_.append(f"int {handle};")
        init_lines.append(f'{handle} = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_{fld.upper()});')
        for buf_idx, kind in ((0, "macd"), (1, "macd_signal"), (2, "macd_histogram")):
            arr = f"{handle}_buf{buf_idx}"
            globals_.append(f"double {arr}[];")
            tick_lines.append(f"ArraySetAsSeries({arr}, true);")
            tick_lines.append(f"CopyBuffer({handle}, {buf_idx}, 0, 3, {arr});")
            var_map[(kind, fld, 0)] = arr

    bb_groups = sorted({(op.field, op.period) for op in indicators.values()
                         if op.kind in ("bollinger_mid", "bollinger_upper", "bollinger_lower")})
    for fld, period in bb_groups:
        handle = f"h_bb_{fld}_{period}"
        globals_.append(f"int {handle};")
        init_lines.append(f'{handle} = iBands(_Symbol, PERIOD_CURRENT, {period}, 0, 2.0, PRICE_{fld.upper()});')
        for buf_idx, kind in ((0, "bollinger_mid"), (1, "bollinger_upper"), (2, "bollinger_lower")):
            arr = f"{handle}_buf{buf_idx}"
            globals_.append(f"double {arr}[];")
            tick_lines.append(f"ArraySetAsSeries({arr}, true);")
            tick_lines.append(f"CopyBuffer({handle}, {buf_idx}, 0, 3, {arr});")
            var_map[(kind, fld, period)] = arr

    for key, op in indicators.items():
        if op.kind in ("macd", "macd_signal", "macd_histogram", "bollinger_mid", "bollinger_upper", "bollinger_lower"):
            continue
        if op.kind == "candle_direction":
            continue  # rendered inline from price calls, no handle needed
        if op.kind in ("highest_high", "lowest_low"):
            continue  # rendered inline via ArrayMaximum/ArrayMinimum over price calls
        fn, mode = _MQL5_KIND_INFO[op.kind]
        handle = f"h_{op.kind}_{op.period}_{op.field}"
        arr = f"{handle}_buf"
        globals_.append(f"int {handle};")
        globals_.append(f"double {arr}[];")
        if fn == "iMA":
            init_lines.append(f'{handle} = iMA(_Symbol, PERIOD_CURRENT, {op.period}, 0, {mode}, PRICE_{op.field.upper()});')
        elif fn == "iRSI":
            init_lines.append(f'{handle} = iRSI(_Symbol, PERIOD_CURRENT, {op.period}, PRICE_{op.field.upper()});')
        elif fn == "iATR":
            init_lines.append(f'{handle} = iATR(_Symbol, PERIOD_CURRENT, {op.period});')
        tick_lines.append(f"ArraySetAsSeries({arr}, true);")
        tick_lines.append(f"CopyBuffer({handle}, 0, 0, 3, {arr});")
        var_map[key] = arr

    return globals_, init_lines, tick_lines, var_map


def _mql5_render_scalar(op: _Operand, var_map: dict[tuple, str], shift: str = "0") -> str:
    """Renders op's value at `shift` (0 = current/most recent closed bar, 1 = prior)."""
    if op.kind == "constant":
        return _fmt_num(op.value)
    if op.kind == "price":
        return _mql5_price_call(op.field, shift)
    if op.kind == "candle_direction":
        c, o = _mql5_price_call("close", shift), _mql5_price_call("open", shift)
        return f"({c} > {o} ? 1 : ({c} < {o} ? -1 : 0))"
    if op.kind == "highest_high":
        return f"iHigh(_Symbol, PERIOD_CURRENT, iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, {op.period}, {shift}))"
    if op.kind == "lowest_low":
        return f"iLow(_Symbol, PERIOD_CURRENT, iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, {op.period}, {shift}))"
    return f"{var_map[op.key()]}[{shift}]"


def _mql5_render_condition(cond: _Condition, var_map: dict[tuple, str]) -> str:
    if cond.operator in ("crosses_above", "crosses_below"):
        left0, right0 = _mql5_render_scalar(cond.left, var_map, "0"), _mql5_render_scalar(cond.right, var_map, "0")
        left1, right1 = _mql5_render_scalar(cond.left, var_map, "1"), _mql5_render_scalar(cond.right, var_map, "1")
        if cond.operator == "crosses_above":
            return f"(({left0} > {right0}) && ({left1} <= {right1}))"
        return f"(({left0} < {right0}) && ({left1} >= {right1}))"
    left = _mql5_render_scalar(cond.left, var_map, "0")
    right = _mql5_render_scalar(cond.right, var_map, "0")
    return f"({left} {cond.operator} {right})"


def _mql5_render_group(group: _ConditionGroup, var_map: dict[tuple, str]) -> str:
    if not group.conditions:
        return "false"
    parts = [_mql5_render_condition(c, var_map) for c in group.conditions]
    expr = parts[0]
    for i, connector in enumerate(group.connectors):
        if i + 1 < len(parts):
            op = "&&" if connector == "AND" else "||"
            expr = f"({expr} {op} {parts[i + 1]})"
    return expr


def to_mql5(config: dict) -> str:
    """Renders a Manual Strategy Builder config as a standalone MQL5
    Expert Advisor, ready to compile in MetaEditor and attach to a live
    or demo MT5 account. Uses CTrade + handle/CopyBuffer indicators (the
    modern MQL5 idiom), not the legacy OrderSend() calling style."""
    parsed = parse_manual_config(config)
    indicators = _collect_indicators(parsed)
    globals_, init_lines, tick_lines, var_map = _mql5_declare_indicators(indicators)

    # ATR-multiple stop/target need their own dedicated handle regardless of
    # whether the strategy's entry/exit conditions happen to reference an
    # ATR operand at the same period -- risk management is generated
    # independently of the condition expressions above.
    risk_atr_handles: dict[int, str] = {}
    for period in filter(None, [
        parsed.stop_atr_period if parsed.stop_type == "atr" else None,
        parsed.target_atr_period if parsed.target_type == "atr" else None,
    ]):
        if period in risk_atr_handles:
            continue
        handle = f"h_risk_atr_{period}"
        arr = f"{handle}_buf"
        globals_.append(f"int {handle};")
        globals_.append(f"double {arr}[];")
        init_lines.append(f"{handle} = iATR(_Symbol, PERIOD_CURRENT, {period});")
        tick_lines.append(f"ArraySetAsSeries({arr}, true);")
        tick_lines.append(f"CopyBuffer({handle}, 0, 0, 3, {arr});")
        risk_atr_handles[period] = arr

    allow_long = parsed.direction in ("long", "both")
    allow_short = parsed.direction in ("short", "both")

    lines: list[str] = []
    lines.append("//+------------------------------------------------------------------+")
    lines.append(f"//| {parsed.name[:64]:<64} |")
    lines.append("//| Generated by the T58 Universal Strategy Translator from a Manual  |")
    lines.append("//| Strategy Builder config. Review before trading a live account.    |")
    lines.append("//+------------------------------------------------------------------+")
    lines.append("#include <Trade\\Trade.mqh>")
    lines.append("CTrade trade;")
    lines.append("")
    lines.append('input double LotSize = 0.10;')
    lines.append("")
    lines.extend(globals_)
    if parsed.max_bars_in_trade:
        lines.append(f"// T58_MAX_BARS_IN_TRADE={parsed.max_bars_in_trade}")
        lines.append("int barsInTrade = 0;")
    lines.append("")

    lines.append("int OnInit()")
    lines.append("{")
    lines.extend(f"   {line}" for line in init_lines)
    lines.append("   return(INIT_SUCCEEDED);")
    lines.append("}")
    lines.append("")

    lines.append("void OnTick()")
    lines.append("{")
    lines.append("   if (Bars(_Symbol, PERIOD_CURRENT) < 50) return; // let indicators warm up")
    lines.extend(f"   {line}" for line in tick_lines)
    lines.append("")
    long_entry_expr = _mql5_render_group(parsed.long_entry if allow_long else _ConditionGroup((), ()), var_map)
    short_entry_expr = _mql5_render_group(parsed.short_entry if allow_short else _ConditionGroup((), ()), var_map)
    long_exit_expr = _mql5_render_group(parsed.long_exit, var_map)
    short_exit_expr = _mql5_render_group(parsed.short_exit, var_map)

    lines.append(f"   bool longEntryCond = {long_entry_expr};")
    lines.append(f"   bool shortEntryCond = {short_entry_expr};")
    lines.append(f"   bool longExitCond = {long_exit_expr};")
    lines.append(f"   bool shortExitCond = {short_exit_expr};")
    lines.append("")

    if parsed.max_bars_in_trade:
        lines.append("   if (PositionSelect(_Symbol)) { barsInTrade++; } else { barsInTrade = 0; }")
        lines.append(f"   bool maxBarsExit = barsInTrade >= {parsed.max_bars_in_trade};")
        lines.append("   longExitCond = longExitCond || maxBarsExit;")
        lines.append("   shortExitCond = shortExitCond || maxBarsExit;")
        lines.append("")

    if parsed.time_exit:
        try:
            hh, mm = (int(x) for x in parsed.time_exit.split(":")[:2])
        except ValueError:
            raise TranslationError(f"Time-based exit '{parsed.time_exit}' must be in HH:MM format.")
        lines.append(f"   // T58_TIME_EXIT={parsed.time_exit}")
        lines.append("   MqlDateTime tm; TimeToStruct(TimeCurrent(), tm);")
        lines.append(f"   bool timeExitCond = (tm.hour > {hh}) || (tm.hour == {hh} && tm.min >= {mm});")
        lines.append("   longExitCond = longExitCond || timeExitCond;")
        lines.append("   shortExitCond = shortExitCond || timeExitCond;")
        lines.append("")

    lines.append("   if (PositionSelect(_Symbol))")
    lines.append("   {")
    lines.append("      long posType = PositionGetInteger(POSITION_TYPE);")
    lines.append("      if (posType == POSITION_TYPE_BUY && longExitCond) trade.PositionClose(_Symbol);")
    lines.append("      if (posType == POSITION_TYPE_SELL && shortExitCond) trade.PositionClose(_Symbol);")
    lines.append("      return;")
    lines.append("   }")
    lines.append("")

    sl_tp_setup: list[str] = []
    if parsed.stop_type or parsed.target_type:
        sl_tp_setup.append('// T58_SL_PIPS / T58_SL_ATR_MULT below use the point size of THIS symbol.')
        if parsed.stop_type == "fixed":
            sl_tp_setup.append(f"   // T58_SL_PIPS={_fmt_num(parsed.stop_value)}")
            sl_tp_setup.append(f"   double slDist = {_fmt_num(parsed.stop_value)} * _Point;")
        elif parsed.stop_type == "atr":
            sl_tp_setup.append(f"   // T58_SL_ATR_MULT={_fmt_num(parsed.stop_value)}")
            sl_tp_setup.append(f"   // T58_ATR_PERIOD={parsed.stop_atr_period}")
            atr_expr = f"{risk_atr_handles[parsed.stop_atr_period]}[0]"
            sl_tp_setup.append(f"   double slDist = {atr_expr} * {_fmt_num(parsed.stop_value)};")
        if parsed.target_type == "fixed":
            sl_tp_setup.append(f"   // T58_TP_PIPS={_fmt_num(parsed.target_value)}")
            sl_tp_setup.append(f"   double tpDist = {_fmt_num(parsed.target_value)} * _Point;")
        elif parsed.target_type == "atr":
            sl_tp_setup.append(f"   // T58_TP_ATR_MULT={_fmt_num(parsed.target_value)}")
            sl_tp_setup.append(f"   // T58_ATR_PERIOD={parsed.target_atr_period}")
            atr_expr = f"{risk_atr_handles[parsed.target_atr_period]}[0]"
            sl_tp_setup.append(f"   double tpDist = {atr_expr} * {_fmt_num(parsed.target_value)};")

    lines.append("   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);")
    lines.append("   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);")
    lines.extend(sl_tp_setup)
    lines.append("")
    lines.append("   if (longEntryCond)")
    lines.append("   {")
    sl_long = "ask - slDist" if parsed.stop_type else "0"
    tp_long = "ask + tpDist" if parsed.target_type else "0"
    lines.append(f'      trade.Buy(LotSize, _Symbol, ask, {sl_long}, {tp_long});')
    lines.append("   }")
    lines.append("   else if (shortEntryCond)")
    lines.append("   {")
    sl_short = "bid + slDist" if parsed.stop_type else "0"
    tp_short = "bid - tpDist" if parsed.target_type else "0"
    lines.append(f'      trade.Sell(LotSize, _Symbol, bid, {sl_short}, {tp_short});')
    lines.append("   }")
    lines.append("}")
    lines.append("")
    lines.extend(_risk_todo_lines(parsed, "//"))
    return "\n".join(lines).rstrip() + "\n"
