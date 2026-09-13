"""
cTrader (cAlgo/C#) exporter -- adds a fourth code-generation target to
app.strategy.translator's Manual Strategy Builder config -> native
platform code pipeline (Pine/MQL5/Python already existed; this is the
`to_ctrader` this module was missing, requested specifically so a
strategy can be deployed to a cTrader-based prop firm account WITHOUT
needing live API access at all -- compile the .cs file in cAlgo,
backtest/attach it there directly).

Deliberately implemented as its own module rather than appended into the
already-1900-line translator.py: it reuses that module's parsing
(`parse_manual_config`, `_collect_indicators`, `_fmt_num`,
`_risk_todo_lines`, `_config_directive_lines`, `TranslationError`) so a
config only needs to be validated once and every renderer shares the
exact same "fail loudly on anything unsupported" posture, but keeps the
cAlgo-specific C# rendering logic separable and independently reviewable.

SUPPORTED, mirroring to_mql5's scope minus two indicator families (see
NOT SUPPORTED below): price fields, sma/ema/wma, rsi, atr, highest_high/
lowest_low, candle_direction; >,>=,<,<=,==,!=, crosses above/below;
fixed-pips or ATR-multiple stop/target (rendered as real
ExecuteMarketOrder stopLossPips/takeProfitPips, converted from price-unit
ATR distances via Symbol.PipSize); max-bars-in-trade; a single daily
clock-time exit; long/short/both direction restriction. Trailing stop and
break-even are surfaced as TODO comments, identically to to_mql5, for the
same stateful-logic-is-risky-to-auto-translate reason.

NOT SUPPORTED YET (raises TranslationError before any code is emitted):
macd, macd_signal, macd_histogram, bollinger_mid/upper/lower, and vwap.
cAlgo's own Indicators.MacdCrossOver/BollingerBands classes DO support
these natively -- they were left out of this first pass specifically to
avoid guessing at exact cAlgo API property names/signatures without a
cTrader account to compile and verify against (see this module's broader
"untested" caveat below), rather than because they're structurally hard.
Adding them is a natural follow-up once the rest of this exporter has
been compiled and run once in cAlgo.

HONESTY NOTE: this generates C# against cAlgo's public Robot API
(OnStart/OnBar, Bars, Indicators.*, ExecuteMarketOrder, Positions) from
Spotware's published documentation, but has not been compiled or run in
cAlgo by anyone on this project. Open the generated .cs file as a New
cBot in cAlgo, paste it in, and fix any compiler errors before trusting
it against a demo account -- the most likely rough edges are exact
Indicators.* method overload signatures, which move between cAlgo API
versions.
"""
from __future__ import annotations

from app.strategy.translator import (
    ParsedStrategy, TranslationError, _ConditionGroup, _Condition, _Operand,
    _collect_indicators, _config_directive_lines, _fmt_num, _risk_todo_lines,
    parse_manual_config,
)

_NOT_YET_SUPPORTED = {"macd", "macd_signal", "macd_histogram", "bollinger_mid", "bollinger_upper", "bollinger_lower", "vwap"}

_MA_TYPE = {"sma": "MovingAverageType.Simple", "ema": "MovingAverageType.Exponential", "wma": "MovingAverageType.Weighted"}


def _price_series(field: str) -> str:
    return {"open": "Bars.OpenPrices", "high": "Bars.HighPrices", "low": "Bars.LowPrices",
            "close": "Bars.ClosePrices", "volume": "Bars.TickVolumes"}.get(field, "Bars.ClosePrices")


def _declare_indicators(indicators: dict[tuple, _Operand]) -> tuple[list[str], dict[tuple, str]]:
    """Returns (field declarations to place before OnStart, and the
    operand-key -> C# variable-name map used when rendering conditions)."""
    fields: list[str] = []
    var_map: dict[tuple, str] = {}
    init_lines: list[str] = []

    for key, op in indicators.items():
        if op.kind in _NOT_YET_SUPPORTED:
            raise TranslationError(
                f"'{op.kind}' isn't supported by the cTrader exporter yet -- see ctrader_export.py's "
                "module docstring for why and what to do instead. Everything else in this strategy "
                "can still be exported separately."
            )
        if op.kind == "candle_direction":
            continue  # rendered inline
        if op.kind in ("highest_high", "lowest_low"):
            continue  # rendered inline via a small helper method

        name = f"_{op.kind}_{op.field}_{op.period}"
        if op.kind in ("sma", "ema", "wma"):
            fields.append(f"private MovingAverage {name};")
            init_lines.append(f"{name} = Indicators.MovingAverage({_price_series(op.field)}, {op.period}, {_MA_TYPE[op.kind]});")
            var_map[key] = f"{name}.Result"
        elif op.kind == "rsi":
            fields.append(f"private RelativeStrengthIndex {name};")
            init_lines.append(f"{name} = Indicators.RelativeStrengthIndex({_price_series(op.field)}, {op.period});")
            var_map[key] = f"{name}.Result"
        elif op.kind == "atr":
            fields.append(f"private AverageTrueRange {name};")
            init_lines.append(f"{name} = Indicators.AverageTrueRange({op.period}, MovingAverageType.Simple);")
            var_map[key] = f"{name}.Result"
        else:
            raise TranslationError(f"Unhandled indicator kind '{op.kind}' in the cTrader exporter.")

    return fields, init_lines, var_map


def _render_scalar_series(op: _Operand, var_map: dict[tuple, str], shift: int) -> str:
    """Same as _render_scalar but always indexes a DataSeries-shaped
    reference with .Last(shift) -- needed because MovingAverage/RSI/ATR
    results are already ".Result" DataSeries, so a *second* .Last() call
    is required to look back further than the current bar."""
    if op.kind == "constant":
        return _fmt_num(op.value)
    if op.kind == "price":
        return f"{_price_series(op.field)}.Last({shift})"
    if op.kind == "candle_direction":
        c, o = f"Bars.ClosePrices.Last({shift})", f"Bars.OpenPrices.Last({shift})"
        return f"({c} > {o} ? 1 : ({c} < {o} ? -1 : 0))"
    if op.kind == "highest_high":
        return f"HighestHigh({op.period}, {shift})"
    if op.kind == "lowest_low":
        return f"LowestLow({op.period}, {shift})"
    return f"{var_map[op.key()]}.Last({shift})"


def _render_condition(cond: _Condition, var_map: dict[tuple, str]) -> str:
    if cond.operator in ("crosses_above", "crosses_below"):
        left0, right0 = _render_scalar_series(cond.left, var_map, 0), _render_scalar_series(cond.right, var_map, 0)
        left1, right1 = _render_scalar_series(cond.left, var_map, 1), _render_scalar_series(cond.right, var_map, 1)
        if cond.operator == "crosses_above":
            return f"(({left0} > {right0}) && ({left1} <= {right1}))"
        return f"(({left0} < {right0}) && ({left1} >= {right1}))"
    left = _render_scalar_series(cond.left, var_map, 0)
    right = _render_scalar_series(cond.right, var_map, 0)
    op = "==" if cond.operator == "==" else ("!=" if cond.operator == "!=" else cond.operator)
    return f"({left} {op} {right})"


def _render_group(group: _ConditionGroup, var_map: dict[tuple, str]) -> str:
    if not group.conditions:
        return "false"
    parts = [_render_condition(c, var_map) for c in group.conditions]
    expr = parts[0]
    for i, connector in enumerate(group.connectors):
        if i + 1 < len(parts):
            op = "&&" if connector == "AND" else "||"
            expr = f"({expr} {op} {parts[i + 1]})"
    return expr


def to_ctrader(config: dict, class_name: str = "T58Strategy") -> str:
    """Renders a Manual Strategy Builder config as a standalone cAlgo
    cBot (.cs). Paste the output into a New cBot in cAlgo, build, and
    backtest/attach it there -- exactly the same "compile before trusting
    it" posture the MQL5 exporter expects for MetaEditor."""
    parsed = parse_manual_config(config)
    indicators = _collect_indicators(parsed)
    fields, init_lines, var_map = _declare_indicators(indicators)

    needs_hh_ll = any(op.kind in ("highest_high", "lowest_low") for op in indicators.values())

    risk_atr_fields: dict[int, str] = {}
    risk_atr_init: list[str] = []
    for period in filter(None, [
        parsed.stop_atr_period if parsed.stop_type == "atr" else None,
        parsed.target_atr_period if parsed.target_type == "atr" else None,
    ]):
        if period in risk_atr_fields:
            continue
        name = f"_riskAtr{period}"
        fields.append(f"private AverageTrueRange {name};")
        risk_atr_init.append(f"{name} = Indicators.AverageTrueRange({period}, MovingAverageType.Simple);")
        risk_atr_fields[period] = name

    allow_long = parsed.direction in ("long", "both")
    allow_short = parsed.direction in ("short", "both")
    class_name = "".join(ch for ch in class_name if ch.isalnum()) or "T58Strategy"

    lines: list[str] = []
    lines.append("// " + "-" * 68)
    lines.append(f"// {parsed.name[:64]}")
    lines.append("// Generated by the T58 Universal Strategy Translator (cTrader exporter)")
    lines.append("// from a Manual Strategy Builder config. Build in cAlgo and test on a")
    lines.append("// demo account before trading it live -- see ctrader_export.py's docstring.")
    lines.append("// " + "-" * 68)
    lines.append("using cAlgo.API;")
    lines.append("using cAlgo.API.Indicators;")
    lines.append("using cAlgo.API.Internals;")
    lines.append("")
    lines.append("namespace cAlgo.Robots")
    lines.append("{")
    lines.append(f'    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]')
    lines.append(f"    public class {class_name} : Robot")
    lines.append("    {")
    lines.append('        [Parameter("Volume (lots)", DefaultValue = 0.1)]')
    lines.append("        public double VolumeLots { get; set; }")
    lines.append("")
    lines.extend(f"        {line}" for line in fields)
    if parsed.max_bars_in_trade:
        lines.append(f"        // T58_MAX_BARS_IN_TRADE={parsed.max_bars_in_trade}")
        lines.append("        private int _barsInTrade = 0;")
    lines.append("")

    lines.append("        protected override void OnStart()")
    lines.append("        {")
    lines.extend(f"            {line}" for line in init_lines)
    lines.extend(f"            {line}" for line in risk_atr_init)
    lines.append("        }")
    lines.append("")

    lines.append("        protected override void OnBar()")
    lines.append("        {")
    lines.append("            if (Bars.Count < 50) return; // let indicators warm up")
    lines.append("")

    long_entry_expr = _render_group(parsed.long_entry if allow_long else _ConditionGroup((), ()), var_map)
    short_entry_expr = _render_group(parsed.short_entry if allow_short else _ConditionGroup((), ()), var_map)
    long_exit_expr = _render_group(parsed.long_exit, var_map)
    short_exit_expr = _render_group(parsed.short_exit, var_map)

    lines.append(f"            bool longEntryCond = {long_entry_expr};")
    lines.append(f"            bool shortEntryCond = {short_entry_expr};")
    lines.append(f"            bool longExitCond = {long_exit_expr};")
    lines.append(f"            bool shortExitCond = {short_exit_expr};")
    lines.append("")

    if parsed.max_bars_in_trade:
        lines.append("            var openPos = Positions.Find(Label, SymbolName);")
        lines.append("            _barsInTrade = openPos != null ? _barsInTrade + 1 : 0;")
        lines.append(f"            bool maxBarsExit = _barsInTrade >= {parsed.max_bars_in_trade};")
        lines.append("            longExitCond = longExitCond || maxBarsExit;")
        lines.append("            shortExitCond = shortExitCond || maxBarsExit;")
        lines.append("")

    if parsed.time_exit:
        try:
            hh, mm = (int(x) for x in parsed.time_exit.split(":")[:2])
        except ValueError:
            raise TranslationError(f"Time-based exit '{parsed.time_exit}' must be in HH:MM format.")
        lines.append(f"            // T58_TIME_EXIT={parsed.time_exit}")
        lines.append(f"            bool timeExitCond = Server.Time.Hour > {hh} || (Server.Time.Hour == {hh} && Server.Time.Minute >= {mm});")
        lines.append("            longExitCond = longExitCond || timeExitCond;")
        lines.append("            shortExitCond = shortExitCond || timeExitCond;")
        lines.append("")

    lines.append("            var position = Positions.Find(Label, SymbolName);")
    lines.append("            if (position != null)")
    lines.append("            {")
    lines.append("                if (position.TradeType == TradeType.Buy && longExitCond) ClosePosition(position);")
    lines.append("                if (position.TradeType == TradeType.Sell && shortExitCond) ClosePosition(position);")
    lines.append("                return;")
    lines.append("            }")
    lines.append("")

    sl_tp_setup: list[str] = []
    sl_pips_expr = "(double?)null"
    tp_pips_expr = "(double?)null"
    if parsed.stop_type == "fixed":
        sl_tp_setup.append(f"            // T58_SL_PIPS={_fmt_num(parsed.stop_value)}")
        sl_pips_expr = _fmt_num(parsed.stop_value)
    elif parsed.stop_type == "atr":
        sl_tp_setup.append(f"            // T58_SL_ATR_MULT={_fmt_num(parsed.stop_value)}")
        sl_tp_setup.append(f"            // T58_ATR_PERIOD={parsed.stop_atr_period}")
        sl_tp_setup.append(f"            double slDist = {risk_atr_fields[parsed.stop_atr_period]}.Result.Last(0) * {_fmt_num(parsed.stop_value)};")
        sl_pips_expr = "slDist / Symbol.PipSize"
    if parsed.target_type == "fixed":
        sl_tp_setup.append(f"            // T58_TP_PIPS={_fmt_num(parsed.target_value)}")
        tp_pips_expr = _fmt_num(parsed.target_value)
    elif parsed.target_type == "atr":
        sl_tp_setup.append(f"            // T58_TP_ATR_MULT={_fmt_num(parsed.target_value)}")
        sl_tp_setup.append(f"            // T58_ATR_PERIOD={parsed.target_atr_period}")
        sl_tp_setup.append(f"            double tpDist = {risk_atr_fields[parsed.target_atr_period]}.Result.Last(0) * {_fmt_num(parsed.target_value)};")
        tp_pips_expr = "tpDist / Symbol.PipSize"

    lines.extend(sl_tp_setup)
    lines.append("")
    lines.append("            if (longEntryCond)")
    lines.append("            {")
    lines.append(f'                ExecuteMarketOrder(TradeType.Buy, SymbolName, Symbol.QuantityToVolumeInUnits(VolumeLots), Label, {sl_pips_expr}, {tp_pips_expr});')
    lines.append("            }")
    lines.append("            else if (shortEntryCond)")
    lines.append("            {")
    lines.append(f'                ExecuteMarketOrder(TradeType.Sell, SymbolName, Symbol.QuantityToVolumeInUnits(VolumeLots), Label, {sl_pips_expr}, {tp_pips_expr});')
    lines.append("            }")
    lines.append("        }")

    if needs_hh_ll:
        lines.append("")
        lines.append("        private double HighestHigh(int period, int shift)")
        lines.append("        {")
        lines.append("            double best = double.MinValue;")
        lines.append("            for (int i = shift; i < shift + period; i++) best = System.Math.Max(best, Bars.HighPrices.Last(i));")
        lines.append("            return best;")
        lines.append("        }")
        lines.append("")
        lines.append("        private double LowestLow(int period, int shift)")
        lines.append("        {")
        lines.append("            double best = double.MaxValue;")
        lines.append("            for (int i = shift; i < shift + period; i++) best = System.Math.Min(best, Bars.LowPrices.Last(i));")
        lines.append("            return best;")
        lines.append("        }")

    lines.append("")
    lines.append('        private string Label => "T58Live";')
    lines.append("    }")
    lines.append("}")

    todo = _risk_todo_lines(parsed, "//")
    if todo:
        lines.append("")
        lines.extend(f"    {t}" for t in todo)
    directive = _config_directive_lines(config, "//")
    if directive:
        lines.append("")
        lines.extend(f"    {d}" for d in directive)

    return "\n".join(lines).rstrip() + "\n"
