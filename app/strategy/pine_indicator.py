"""Pine Script (TradingView) INDICATOR creator.

Three ways in, one pipeline out:

1. **Idea -> indicator, offline**: ``build_from_idea`` parses a plain-English idea ("EMA 21/50/200 ribbon with RSI 14 and Bollinger
   bands, mark liquidity sweeps") into components from a catalog of hand-written, pattern-safe Pine v5 building blocks and assembles
   them. No AI needed; deterministic.
2. **Idea -> indicator, with AI**: ``generate_with_llm`` asks the configured model (local Ollama, or the user's hosted key) for a script
   under strict Pine v5 rules, lints the answer, and feeds the lint errors back for up to ``max_repairs`` repair rounds. If the model is
   unavailable or cannot produce lint-clean code, the offline builder is used instead (and the result says so).
3. **Code in**: ``check_and_fix`` takes pasted code, auto-fixes the mechanical v4->v5 problems, then lints.

HONESTY ABOUT "COMPILES": TradingView's compiler is not available outside TradingView. ``lint`` is a static checker for the error classes
that most often stop a script from compiling (version line, declaration, v4 leftovers, unbalanced brackets, indentation rules, global-scope-only calls,
re-assignment with ``=``, unknown ``ta./math./str./...`` members, plot limits, ...). When the optional ``pynescript`` package is installed the code is also
parsed with a real Pine grammar (``grammar_check``). Both lower the risk a lot -- neither is a guarantee. The result always carries ``verified_by``
so the UI can say exactly what was checked, and tells the user to paste into TradingView's Pine Editor and press "Add to chart".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

PINE_VERSION = 5

# ---------------------------------------------------------------------------------------------------------------
# Known built-in members (Pine v5). A member outside these lists is reported as an error (typo / hallucination).
# ---------------------------------------------------------------------------------------------------------------
TA_FUNCS = {
    "alma", "atr", "barssince", "bb", "bbw", "cci", "change", "cmo", "cog", "correlation", "cross", "crossover", "crossunder", "cum",
    "dev", "dmi", "ema", "falling", "highest", "highestbars", "hma", "kc", "kcw", "linreg", "lowest", "lowestbars", "macd", "max", "median",
    "mfi", "min", "mode", "mom", "percentile_linear_interpolation", "percentile_nearest_rank", "percentrank", "pivot_point_levels",
    "pivothigh", "pivotlow", "range", "rising", "rma", "roc", "rsi", "sar", "sma", "stdev", "stoch", "supertrend", "swma", "tr", "tsi",
    "valuewhen", "variance", "vwap", "vwma", "wma", "wpr",
}
TA_VARS = {"accdist", "iii", "nvi", "obv", "pvi", "pvt", "tr", "vwap", "wad", "wvad"}
MATH_FUNCS = {"abs", "acos", "asin", "atan", "avg", "ceil", "cos", "exp", "floor", "log", "log10", "max", "min", "pow", "random", "round", "round_to_mintick",
              "sign", "sin", "sqrt", "sum", "tan", "todegrees", "toradians"}
MATH_CONSTS = {"e", "phi", "pi", "rphi"}
STR_FUNCS = {"contains", "endswith", "format", "format_time", "length", "lower", "match", "pos", "repeat", "replace", "replace_all", "split", "startswith",
             "substring", "tonumber", "tostring", "trim", "upper"}
COLORS = {"aqua", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "orange", "purple", "red", "silver", "teal", "white", "yellow"}
COLOR_FUNCS = {"b", "from_gradient", "g", "new", "r", "rgb", "t"}
INPUT_FUNCS = {"bool", "color", "float", "int", "price", "session", "source", "string", "symbol", "text_area", "time", "timeframe", "enum"}
NAMESPACES = {"ta", "math", "str", "color", "input", "array", "matrix", "map", "request", "ticker", "timeframe", "syminfo", "barstate", "session", "plot", "line", "label",
              "box", "table", "chart", "alert", "location", "shape", "size", "position", "extend", "xloc", "yloc", "hline", "display", "format", "currency", "dayofweek",
              "barmerge", "adjustment", "backadjustment", "settlement_as_close", "text", "font", "label", "linefill", "polyline", "runtime", "log", "strategy", "order", "scale", "dividends", "earnings", "splits"}
BUILTIN_SERIES = {"open", "high", "low", "close", "volume", "time", "time_close", "time_tradingday", "hl2", "hlc3", "ohlc4", "hlcc4", "bar_index", "last_bar_index",
                  "last_bar_time", "na", "dayofmonth", "dayofweek", "hour", "minute", "month", "second", "year", "weekofyear", "timenow", "true", "false"}
V4_BARE = {  # v4 bare function -> v5 replacement
    "sma": "ta.sma", "ema": "ta.ema", "rma": "ta.rma", "wma": "ta.wma", "vwma": "ta.vwma", "rsi": "ta.rsi", "atr": "ta.atr", "macd": "ta.macd",
    "stoch": "ta.stoch", "cci": "ta.cci", "mfi": "ta.mfi", "bb": "ta.bb", "highest": "ta.highest", "lowest": "ta.lowest", "crossover": "ta.crossover",
    "crossunder": "ta.crossunder", "cross": "ta.cross", "change": "ta.change", "mom": "ta.mom", "roc": "ta.roc", "stdev": "ta.stdev", "linreg": "ta.linreg",
    "pivothigh": "ta.pivothigh", "pivotlow": "ta.pivotlow", "valuewhen": "ta.valuewhen", "barssince": "ta.barssince", "cum": "ta.cum", "tr": "ta.tr",
    "supertrend": "ta.supertrend", "sar": "ta.sar", "wpr": "ta.wpr", "hma": "ta.hma", "swma": "ta.swma", "rising": "ta.rising", "falling": "ta.falling",
    "dmi": "ta.dmi", "kc": "ta.kc", "tsi": "ta.tsi", "percentrank": "ta.percentrank", "correlation": "ta.correlation", "variance": "ta.variance",
    "highestbars": "ta.highestbars", "lowestbars": "ta.lowestbars",
}
V4_MATH = {"abs": "math.abs", "sqrt": "math.sqrt", "pow": "math.pow", "log": "math.log", "exp": "math.exp", "max": "math.max", "min": "math.min", "round": "math.round",
           "floor": "math.floor", "ceil": "math.ceil", "sign": "math.sign", "avg": "math.avg", "sum": "math.sum"}
GLOBAL_ONLY_CALLS = ("plot", "plotshape", "plotchar", "plotcandle", "plotbar", "bgcolor", "barcolor", "hline", "fill", "alertcondition")


@dataclass
class Issue:
    level: str          # "error" | "warning" | "info"
    line: int           # 1-based, 0 = whole script
    message: str
    fix: str = ""

    def to_dict(self) -> dict:
        return {"level": self.level, "line": self.line, "message": self.message, "fix": self.fix}


def _strip_strings_comments(line: str) -> str:
    """The line with string literals blanked and the // comment removed (so identifiers inside strings are never linted)."""
    out, i, n, quote = [], 0, len(line), ""
    while i < n:
        ch = line[i]
        if quote:
            if ch == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if ch == quote:
                quote = ""
                out.append(ch)
            else:
                out.append(" ")
        else:
            if ch in "\"'":
                quote = ch
                out.append(ch)
            elif ch == "/" and i + 1 < n and line[i + 1] == "/":
                break
            else:
                out.append(ch)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------------------------------------------
def lint(code: str) -> list[Issue]:
    issues: list[Issue] = []
    raw_lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not code.strip():
        return [Issue("error", 0, "The script is empty.")]
    # --- version + declaration -----------------------------------------------------------------------------
    ver = None
    for ln in raw_lines:
        m = re.match(r"^\s*//\s*@version\s*=\s*(\d+)", ln)
        if m:
            ver = int(m.group(1))
            break
        if ln.strip() and not ln.strip().startswith("//"):
            break
    if ver is None:
        issues.append(Issue("error", 1, "Missing '//@version=5' as the first line.", "Add //@version=5 at the very top."))
    elif ver < 5:
        issues.append(Issue("error", 1, f"Pine v{ver} syntax is outdated; this tool targets v5 (v6 also compiles most v5 code).", "Use //@version=5."))
    decl = [i + 1 for i, ln in enumerate(raw_lines) if re.match(r"^\s*(indicator|strategy|library|study)\s*\(", _strip_strings_comments(ln))]
    if not decl:
        issues.append(Issue("error", 0, "No indicator(...) declaration found.", 'Add: indicator("My Indicator", overlay=true)'))
    elif len(decl) > 1:
        issues.append(Issue("error", decl[1], "More than one script declaration."))
    for i, ln in enumerate(raw_lines, 1):
        s = _strip_strings_comments(ln)
        if re.match(r"^\s*strategy\s*\(", s):
            issues.append(Issue("error", i, "This is a strategy(...) script, but this tool creates indicators.", "Use indicator(...) and plot signals instead of strategy.entry()."))
        if re.match(r"^\s*study\s*\(", s):
            issues.append(Issue("error", i, "study() was removed in Pine v5.", "Use indicator(...)."))
    # --- balance of brackets & quotes (whole script) -------------------------------------------------------
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for i, ln in enumerate(raw_lines, 1):
        s = _strip_strings_comments(ln)
        if s.count('"') % 2 or s.count("'") % 2:
            issues.append(Issue("error", i, "Unterminated string literal."))
        for ch in s:
            if ch in "([{":
                stack.append((ch, i))
            elif ch in pairs:
                if not stack or stack[-1][0] != pairs[ch]:
                    issues.append(Issue("error", i, f"Unmatched '{ch}'."))
                else:
                    stack.pop()
    for ch, ln in stack[:3]:
        issues.append(Issue("error", ln, f"'{ch}' is never closed."))
    # --- line-level checks ------------------------------------------------------------------------------------
    declared_global: dict[str, int] = {}
    plot_count = 0
    in_block_indent = [0]
    prev_continues = False
    paren_depth = 0
    for i, ln in enumerate(raw_lines, 1):
        s = _strip_strings_comments(ln)
        if not s.strip():
            continue
        if "\t" in ln[: len(ln) - len(ln.lstrip())] and " " in ln[: len(ln) - len(ln.lstrip())]:
            issues.append(Issue("error", i, "Mixed tabs and spaces in indentation."))
        indent = len(ln) - len(ln.lstrip(" "))
        is_tab = ln.startswith("\t")
        continuation = paren_depth > 0 or prev_continues
        if not continuation and indent > 0 and not is_tab and indent % 4 != 0:
            issues.append(Issue("error", i, f"Indentation of {indent} spaces: local blocks must be indented by a multiple of 4 spaces (or a tab).",
                                "Use 4 spaces per block level."))
        if continuation and indent > 0 and not is_tab and indent % 4 == 0 and paren_depth == 0 and prev_continues:
            issues.append(Issue("warning", i, "Wrapped line indented by a multiple of 4 spaces can be read as a new block; use e.g. 2 or 6 spaces.", ""))
        paren_depth += s.count("(") + s.count("[") - s.count(")") - s.count("]")
        paren_depth = max(paren_depth, 0)
        stripped = s.rstrip()
        prev_continues = bool(re.search(r"(\?|:|\+|-|\*|/|%|,|\band\b|\bor\b|=|==|!=|<=|>=|<|>)\s*$", stripped)) and not stripped.endswith("//")
        body = s.strip()
        # v4 leftovers
        m = re.search(r"(?<![\w.])(sma|ema|rma|wma|vwma|rsi|atr|macd|stoch|cci|mfi|highest|lowest|crossover|crossunder|cross|change|mom|roc|stdev|linreg|pivothigh|pivotlow|valuewhen|barssince|supertrend|sar|wpr|hma|cum)\s*\(", s)
        if m and not re.match(r"^\s*(//)", ln):
            issues.append(Issue("error", i, f"'{m.group(1)}(' is a Pine v4 call.", f"Use {V4_BARE[m.group(1)]}(...)."))
        m = re.search(r"(?<![\w.])(abs|sqrt|pow|exp|round|floor|ceil|sign|avg)\s*\(", s)
        if m:
            issues.append(Issue("error", i, f"'{m.group(1)}(' needs the math. prefix in v5.", f"Use {V4_MATH[m.group(1)]}(...)."))
        if re.search(r"(?<![\w.])security\s*\(", s):
            issues.append(Issue("error", i, "security() is request.security() in v5.", "Use request.security(...)."))
        if re.search(r"(?<![\w.])(color\.new|input)\s*\(", s) is None and re.search(r"(?<![\w.])input\s*\(\s*[^)]*type\s*=", s):
            issues.append(Issue("error", i, "input(type=...) is v4; use input.int / input.float / input.bool / input.string ...", ""))
        if re.search(r"\bnz\(|\bna\(", s) is None and re.search(r"\bvarip?\b", s) and False:
            pass
        # namespaced members
        for ns, member in re.findall(r"\b(ta|math|str|color|input)\.([A-Za-z_]\w*)", s):
            known = {"ta": TA_FUNCS | TA_VARS, "math": MATH_FUNCS | MATH_CONSTS, "str": STR_FUNCS, "color": COLORS | COLOR_FUNCS, "input": INPUT_FUNCS}[ns]
            if member not in known:
                near = _closest(member, known)
                issues.append(Issue("error", i, f"Unknown {ns}.{member}.", f"Did you mean {ns}.{near}?" if near else ""))
        # global-only calls inside a block
        if indent > 0 and not continuation:
            m = re.match(r"^(plot|plotshape|plotchar|plotcandle|plotbar|bgcolor|barcolor|hline|fill|alertcondition)\s*\(", body)
            if m:
                issues.append(Issue("error", i, f"{m.group(1)}() cannot be called inside an if/for/function block (it must be at global scope).",
                                    "Compute a series inside the block, then call plot at the left margin."))
        # counts
        if re.match(r"^(plot|plotshape|plotchar|plotcandle|plotbar|hline|bgcolor|barcolor)\s*\(", body) and indent == 0:
            plot_count += 1
        # global redeclaration with '='
        if indent == 0 and not continuation:
            md = re.match(r"^(?:(?:var|varip)\s+)?(?:(?:int|float|bool|string|color|series\s+\w+|simple\s+\w+)\s+)?([A-Za-z_]\w*)\s*=(?!=)", body)
            if md and not re.match(r"^(if|for|while|switch)\b", body):
                name = md.group(1)
                if name in declared_global:
                    issues.append(Issue("error", i, f"'{name}' is already declared (line {declared_global[name]}); re-assign with ':=' instead of '='.", f"{name} := ..."))
                else:
                    declared_global[name] = i
            if re.match(r"^\[[^\]]+\]\s*=", body):  # tuple assignment
                for nm in re.findall(r"[A-Za-z_]\w*", body.split("]")[0]):
                    declared_global.setdefault(nm, i)
        if re.search(r"(?<![=!<>:])=(?!=)\s*$", body) is None and re.match(r"^(if|for|while)\b.*:\s*$", body):
            issues.append(Issue("error", i, "Pine does not use a trailing ':' after if/for/while.", "Remove the colon."))
        if re.search(r"\b(for)\b.*\bin\b.*:\s*$", body):
            issues.append(Issue("error", i, "Pine's for-in has no trailing colon."))
        if re.search(r"\blookahead\s*=\s*barmerge\.lookahead_on\b", s) and "[1]" not in "".join(raw_lines[max(0, i - 3): i + 1]):
            issues.append(Issue("warning", i, "request.security with lookahead_on and no [1] offset repaints (uses future data on history).", "Offset the expression with [1] or use lookahead_off."))
        if re.search(r"\bvar\s+(?!int|float|bool|string|color|label|line|box|table)\w+\s*$", body):
            issues.append(Issue("error", i, "'var' needs an assignment.", ""))
    if plot_count > 64:
        issues.append(Issue("error", 0, f"{plot_count} plot calls: TradingView allows at most 64 plot counts per script."))
    if decl and not any(re.match(r"^(plot|plotshape|plotchar|plotcandle|plotbar|bgcolor|barcolor|hline|alertcondition|label\.new|line\.new|box\.new|table\.new|fill)\b",
                                 _strip_strings_comments(l).strip()) or re.search(r"\b(label|line|box|table)\.new\s*\(", _strip_strings_comments(l)) for l in raw_lines):
        issues.append(Issue("error", 0, "The script draws nothing (no plot/plotshape/label/line/box/bgcolor).", "Add at least one plot(...)."))
    # --- unknown identifiers: very conservative (only obvious v4 series like 'open[1]' are fine) ------------------
    return sorted(issues, key=lambda x: (x.line, x.level != "error"))


def _closest(name: str, options) -> str:
    import difflib

    m = difflib.get_close_matches(name, list(options), n=1, cutoff=0.6)
    return m[0] if m else ""


def grammar_check(code: str) -> tuple[Optional[bool], str]:
    """(True/False, message) from the optional ``pynescript`` Pine grammar; (None, reason) when it is not installed."""
    try:
        from pynescript.ast import parse
    except Exception:  # noqa: BLE001
        return None, "pynescript is not installed, so only the built-in linter ran."
    try:
        parse(code if code.endswith("\n") else code + "\n")
        return True, "Parsed with a real Pine grammar (pynescript)."
    except Exception as exc:  # noqa: BLE001
        return False, f"Pine grammar parse failed: {str(exc)[:200]}"


# ---------------------------------------------------------------------------------------------------------------
# Auto-fix (mechanical v4 -> v5 repairs)
# ---------------------------------------------------------------------------------------------------------------
def autofix(code: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    text = code.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    lines = text.split("\n")
    if not any(re.match(r"^\s*//\s*@version", l) for l in lines[:5]):
        lines.insert(0, f"//@version={PINE_VERSION}")
        changes.append("Added //@version=5.")
    elif any(re.match(r"^\s*//\s*@version\s*=\s*[1-4]\b", l) for l in lines[:5]):
        lines = [re.sub(r"@version\s*=\s*[1-4]\b", "@version=5", l, count=1) if re.match(r"^\s*//\s*@version", l) else l for l in lines]
        changes.append("Upgraded //@version to 5.")
    out = []
    for ln in lines:
        head, comment = ln, ""
        s = _strip_strings_comments(ln)
        if "//" in ln and len(s) < len(ln):
            idx = len(s)
            head, comment = ln[:idx], ln[idx:]
        new = head
        new = re.sub(r"^(\s*)study\s*\(", r"\1indicator(", new)
        new = re.sub(r"(?<![\w.])security\s*\(", "request.security(", new)
        for old, rep in V4_BARE.items():
            new = re.sub(rf"(?<![\w.]){old}\s*\(", rep + "(", new)
        for old, rep in V4_MATH.items():
            new = re.sub(rf"(?<![\w.]){old}\s*\(", rep + "(", new)
        m_in = re.search(r"(?<![\w.])input\s*\(([^()]*?),\s*type\s*=\s*input\.(\w+)([^()]*)\)", new)
        if m_in:
            kind = {"integer": "int", "resolution": "timeframe"}.get(m_in.group(2), m_in.group(2))
            rest = m_in.group(3).strip()
            new = new[:m_in.start()] + f"input.{kind}({m_in.group(1).strip()}{rest})" + new[m_in.end():]
        # v4 colors written bare:  color=red  ->  color.red
        new = re.sub(r"(?<![\w.])(color\s*=\s*)(aqua|black|blue|fuchsia|gray|green|lime|maroon|navy|olive|orange|purple|red|silver|teal|white|yellow)\b", r"\1color.\2", new)
        new = re.sub(r"(?<![\w.])(textcolor\s*=\s*)(aqua|black|blue|fuchsia|gray|green|lime|maroon|navy|olive|orange|purple|red|silver|teal|white|yellow)\b", r"\1color.\2", new)
        if new != head:
            changes.append(f"Converted v4 syntax: {head.strip()[:60]!r}")
        out.append(new + comment)
    return "\n".join(out), list(dict.fromkeys(changes))[:20]


# ---------------------------------------------------------------------------------------------------------------
# Component catalog (offline builder)
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Component:
    id: str
    title: str
    overlay: bool
    keywords: tuple[str, ...]
    defaults: dict
    render: Callable[[dict], tuple[list[str], list[str]]]       # params -> (input/body lines, plot lines)


def _c_ma(kind: str) -> Callable[[dict], tuple[list[str], list[str]]]:
    def r(p):
        periods = p.get("periods") or [21]
        colors = ["color.new(color.aqua, 0)", "color.new(color.yellow, 0)", "color.new(color.orange, 0)", "color.new(color.fuchsia, 0)", "color.new(color.white, 0)"]
        body, plots = [], []
        for n, per in enumerate(periods[:5], 1):
            body += [f'{kind}Len{n} = input.int({int(per)}, "{kind.upper()} {n} length", minval=1, group="{kind.upper()}")',
                     f"{kind}Val{n} = ta.{kind}(close, {kind}Len{n})"]
            plots.append(f'plot({kind}Val{n}, "{kind.upper()} {n}", color={colors[(n - 1) % len(colors)]}, linewidth=2)')
        return body, plots
    return r


def _c_vwap(p):
    return ['vwapVal = ta.vwap(hlc3)'], ['plot(vwapVal, "VWAP", color=color.new(color.purple, 0), linewidth=2)']


def _c_bb(p):
    return ['bbLen = input.int(%d, "BB length", minval=1, group="Bollinger")' % int(p.get("period", 20)),
            'bbMult = input.float(%s, "BB multiplier", minval=0.1, step=0.1, group="Bollinger")' % p.get("mult", 2.0),
            '[bbMid, bbUp, bbLo] = ta.bb(close, bbLen, bbMult)'], \
           ['plot(bbMid, "BB basis", color=color.new(color.gray, 0))', 'bbUpP = plot(bbUp, "BB upper", color=color.new(color.blue, 0))',
            'bbLoP = plot(bbLo, "BB lower", color=color.new(color.blue, 0))', 'fill(bbUpP, bbLoP, color=color.new(color.blue, 90), title="BB fill")']


def _c_supertrend(p):
    return ['stFactor = input.float(%s, "Supertrend factor", minval=0.5, step=0.1, group="Supertrend")' % p.get("factor", 3.0),
            'stAtr = input.int(%d, "Supertrend ATR length", minval=1, group="Supertrend")' % int(p.get("period", 10)),
            '[stVal, stDir] = ta.supertrend(stFactor, stAtr)'], \
           ['plot(stDir < 0 ? stVal : na, "Supertrend up", color=color.new(color.green, 0), style=plot.style_linebr, linewidth=2)',
            'plot(stDir > 0 ? stVal : na, "Supertrend down", color=color.new(color.red, 0), style=plot.style_linebr, linewidth=2)']


def _c_donchian(p):
    return ['dcLen = input.int(%d, "Donchian length", minval=1, group="Donchian")' % int(p.get("period", 20)),
            'dcUp = ta.highest(high, dcLen)', 'dcLo = ta.lowest(low, dcLen)', 'dcMid = math.avg(dcUp, dcLo)'], \
           ['plot(dcUp, "Donchian upper", color=color.new(color.teal, 0))', 'plot(dcLo, "Donchian lower", color=color.new(color.teal, 0))',
            'plot(dcMid, "Donchian mid", color=color.new(color.gray, 50))']


def _c_prev_day(p):
    return ['pdHigh = request.security(syminfo.tickerid, "D", high[1], lookahead=barmerge.lookahead_on)',
            'pdLow = request.security(syminfo.tickerid, "D", low[1], lookahead=barmerge.lookahead_on)',
            'pdClose = request.security(syminfo.tickerid, "D", close[1], lookahead=barmerge.lookahead_on)'], \
           ['plot(pdHigh, "Prev day high", color=color.new(color.red, 20), style=plot.style_linebr)',
            'plot(pdLow, "Prev day low", color=color.new(color.green, 20), style=plot.style_linebr)',
            'plot(pdClose, "Prev day close", color=color.new(color.gray, 20), style=plot.style_linebr)']


def _c_sweep(p):
    return ['swLen = input.int(%d, "Sweep lookback", minval=2, group="Liquidity sweeps")' % int(p.get("period", 20)),
            'priorLow = ta.lowest(low, swLen)[1]', 'priorHigh = ta.highest(high, swLen)[1]',
            'sslSweep = low < priorLow and close > priorLow', 'bslSweep = high > priorHigh and close < priorHigh'], \
           ['plotshape(sslSweep, "SSL sweep (buy-side reversal)", style=shape.triangleup, location=location.belowbar, color=color.new(color.green, 0), size=size.small)',
            'plotshape(bslSweep, "BSL sweep (sell-side reversal)", style=shape.triangledown, location=location.abovebar, color=color.new(color.red, 0), size=size.small)']


def _c_fvg(p):
    return ['fvgMin = input.float(%s, "Min gap (x ATR)", minval=0, step=0.05, group="Fair value gaps")' % p.get("min_atr", 0.25),
            'fvgAtr = ta.atr(14)', 'bullFvg = low > high[2] and (low - high[2]) > fvgMin * fvgAtr', 'bearFvg = high < low[2] and (low[2] - high) > fvgMin * fvgAtr'], \
           ['plotshape(bullFvg, "Bullish FVG", style=shape.diamond, location=location.belowbar, color=color.new(color.teal, 0), size=size.tiny)',
            'plotshape(bearFvg, "Bearish FVG", style=shape.diamond, location=location.abovebar, color=color.new(color.maroon, 0), size=size.tiny)']


def _c_pivots(p):
    return ['pvLeft = input.int(%d, "Pivot left", minval=1, group="Swing pivots")' % int(p.get("left", 5)),
            'pvRight = input.int(%d, "Pivot right", minval=1, group="Swing pivots")' % int(p.get("right", 5)),
            'pivHigh = ta.pivothigh(high, pvLeft, pvRight)', 'pivLow = ta.pivotlow(low, pvLeft, pvRight)'], \
           ['plotshape(not na(pivHigh), "Swing high", style=shape.circle, location=location.abovebar, color=color.new(color.red, 0), size=size.tiny, offset=-pvRight)',
            'plotshape(not na(pivLow), "Swing low", style=shape.circle, location=location.belowbar, color=color.new(color.green, 0), size=size.tiny, offset=-pvRight)']


def _c_cross(p):
    return ['xFast = input.int(%d, "Fast EMA", minval=1, group="EMA cross")' % int(p.get("fast", 9)),
            'xSlow = input.int(%d, "Slow EMA", minval=2, group="EMA cross")' % int(p.get("slow", 21)),
            'emaFast = ta.ema(close, xFast)', 'emaSlow = ta.ema(close, xSlow)', 'crossUp = ta.crossover(emaFast, emaSlow)', 'crossDn = ta.crossunder(emaFast, emaSlow)'], \
           ['plot(emaFast, "Fast EMA", color=color.new(color.aqua, 0))', 'plot(emaSlow, "Slow EMA", color=color.new(color.orange, 0))',
            'plotshape(crossUp, "Bull cross", style=shape.labelup, location=location.belowbar, color=color.new(color.green, 0), text="Buy", textcolor=color.white, size=size.small)',
            'plotshape(crossDn, "Bear cross", style=shape.labeldown, location=location.abovebar, color=color.new(color.red, 0), text="Sell", textcolor=color.white, size=size.small)',
            'alertcondition(crossUp, "Bull cross", "Fast EMA crossed above slow EMA")', 'alertcondition(crossDn, "Bear cross", "Fast EMA crossed below slow EMA")']


def _c_atr_stop(p):
    return ['asLen = input.int(%d, "ATR length", minval=1, group="ATR stop")' % int(p.get("period", 14)),
            'asMult = input.float(%s, "ATR multiple", minval=0.1, step=0.1, group="ATR stop")' % p.get("mult", 2.0),
            'atrUp = close - asMult * ta.atr(asLen)', 'atrDn = close + asMult * ta.atr(asLen)'], \
           ['plot(atrUp, "ATR stop (long)", color=color.new(color.green, 40), style=plot.style_linebr)', 'plot(atrDn, "ATR stop (short)", color=color.new(color.red, 40), style=plot.style_linebr)']


def _c_volspike(p):
    return ['vsLen = input.int(%d, "Volume average length", minval=2, group="Volume spike")' % int(p.get("period", 20)),
            'vsMult = input.float(%s, "Spike multiple", minval=1, step=0.1, group="Volume spike")' % p.get("mult", 2.0),
            'volSpike = volume > vsMult * ta.sma(volume, vsLen)'], \
           ['plotshape(volSpike, "Volume spike", style=shape.square, location=location.bottom, color=color.new(color.yellow, 0), size=size.tiny)']


def _c_rsi(p):
    return ['rsiLen = input.int(%d, "RSI length", minval=1, group="RSI")' % int(p.get("period", 14)),
            'rsiOb = input.int(70, "Overbought", group="RSI")', 'rsiOs = input.int(30, "Oversold", group="RSI")', 'rsiVal = ta.rsi(close, rsiLen)'], \
           ['plot(rsiVal, "RSI", color=color.new(color.purple, 0), linewidth=2)', 'hline(70, "Overbought", color=color.new(color.red, 50))',
            'hline(30, "Oversold", color=color.new(color.green, 50))', 'hline(50, "Mid", color=color.new(color.gray, 70))']


def _c_macd(p):
    return ['macdFast = input.int(%d, "Fast", minval=1, group="MACD")' % int(p.get("fast", 12)), 'macdSlow = input.int(%d, "Slow", minval=2, group="MACD")' % int(p.get("slow", 26)),
            'macdSig = input.int(%d, "Signal", minval=1, group="MACD")' % int(p.get("signal", 9)), '[macdLine, macdSignal, macdHist] = ta.macd(close, macdFast, macdSlow, macdSig)'], \
           ['plot(macdHist, "Histogram", style=plot.style_columns, color=macdHist >= 0 ? color.new(color.teal, 30) : color.new(color.red, 30))',
            'plot(macdLine, "MACD", color=color.new(color.blue, 0))', 'plot(macdSignal, "Signal", color=color.new(color.orange, 0))']


def _c_stoch(p):
    return ['stK = input.int(%d, "%%K length", minval=1, group="Stochastic")' % int(p.get("period", 14)), 'stSmooth = input.int(3, "%K smoothing", minval=1, group="Stochastic")',
            'stD = input.int(3, "%D smoothing", minval=1, group="Stochastic")', 'kRaw = ta.stoch(close, high, low, stK)', 'kLine = ta.sma(kRaw, stSmooth)', 'dLine = ta.sma(kLine, stD)'], \
           ['plot(kLine, "%K", color=color.new(color.blue, 0))', 'plot(dLine, "%D", color=color.new(color.orange, 0))', 'hline(80, "Upper", color=color.new(color.red, 50))',
            'hline(20, "Lower", color=color.new(color.green, 50))']


def _c_adx(p):
    return ['adxLen = input.int(%d, "ADX length", minval=1, group="ADX")' % int(p.get("period", 14)), '[diPlus, diMinus, adxVal] = ta.dmi(adxLen, adxLen)'], \
           ['plot(adxVal, "ADX", color=color.new(color.white, 0), linewidth=2)', 'plot(diPlus, "+DI", color=color.new(color.green, 0))', 'plot(diMinus, "-DI", color=color.new(color.red, 0))',
            'hline(25, "Trend threshold", color=color.new(color.gray, 60))']


def _c_atr(p):
    return ['atrLen = input.int(%d, "ATR length", minval=1, group="ATR")' % int(p.get("period", 14)), 'atrVal = ta.atr(atrLen)'], ['plot(atrVal, "ATR", color=color.new(color.orange, 0), linewidth=2)']


def _c_cci(p):
    return ['cciLen = input.int(%d, "CCI length", minval=1, group="CCI")' % int(p.get("period", 20)), 'cciVal = ta.cci(hlc3, cciLen)'], \
           ['plot(cciVal, "CCI", color=color.new(color.aqua, 0))', 'hline(100, "Upper", color=color.new(color.red, 50))', 'hline(-100, "Lower", color=color.new(color.green, 50))', 'hline(0, "Zero", color=color.new(color.gray, 60))']


def _c_mfi(p):
    return ['mfiLen = input.int(%d, "MFI length", minval=1, group="MFI")' % int(p.get("period", 14)), 'mfiVal = ta.mfi(hlc3, mfiLen)'], \
           ['plot(mfiVal, "MFI", color=color.new(color.teal, 0))', 'hline(80, "Upper", color=color.new(color.red, 50))', 'hline(20, "Lower", color=color.new(color.green, 50))']


def _c_obv(p):
    return ['obvVal = ta.obv'], ['plot(obvVal, "OBV", color=color.new(color.blue, 0))']


def _c_cvd(p):
    return ['cvdRange = high - low', 'cvdDelta = cvdRange > 0 ? volume * ((close - low) - (high - close)) / cvdRange : 0.0', 'cvdVal = ta.cum(cvdDelta)'], \
           ['plot(cvdVal, "CVD (bar-location proxy)", color=color.new(color.aqua, 0), linewidth=2)']


def _c_relvol(p):
    return ['rvLen = input.int(%d, "Average length", minval=2, group="Relative volume")' % int(p.get("period", 20)), 'rvVal = volume / ta.sma(volume, rvLen)'], \
           ['plot(rvVal, "Relative volume", style=plot.style_columns, color=rvVal >= 1.5 ? color.new(color.orange, 20) : color.new(color.gray, 50))', 'hline(1, "Average", color=color.new(color.gray, 60))']


def _c_roc(p):
    return ['rocLen = input.int(%d, "ROC length", minval=1, group="Momentum")' % int(p.get("period", 10)), 'rocVal = ta.roc(close, rocLen)'], \
           ['plot(rocVal, "ROC %", color=color.new(color.fuchsia, 0))', 'hline(0, "Zero", color=color.new(color.gray, 60))']


CATALOG: list[Component] = [
    Component("ema", "EMA", True, ("ema", "exponential"), {"periods": [21]}, _c_ma("ema")),
    Component("sma", "SMA", True, ("sma", "simple moving", "moving average"), {"periods": [50]}, _c_ma("sma")),
    Component("vwap", "VWAP", True, ("vwap",), {}, _c_vwap),
    Component("bb", "Bollinger Bands", True, ("bollinger", "bband", "bands"), {"period": 20, "mult": 2.0}, _c_bb),
    Component("supertrend", "Supertrend", True, ("supertrend", "super trend"), {"factor": 3.0, "period": 10}, _c_supertrend),
    Component("donchian", "Donchian Channel", True, ("donchian", "channel breakout", "range breakout"), {"period": 20}, _c_donchian),
    Component("prevday", "Previous-day levels", True, ("previous day", "prior day", "prev day", "pdh", "pdl", "yesterday"), {}, _c_prev_day),
    Component("sweep", "Liquidity sweeps", True, ("sweep", "liquidity", "stop run", "stop hunt", "ssl", "bsl"), {"period": 20}, _c_sweep),
    Component("fvg", "Fair value gaps", True, ("fair value gap", "fvg", "imbalance gap"), {"min_atr": 0.25}, _c_fvg),
    Component("pivots", "Swing pivots", True, ("pivot", "swing high", "swing low", "swing point", "structure"), {"left": 5, "right": 5}, _c_pivots),
    Component("cross", "EMA cross signals", True, ("cross", "crossover", "golden cross", "signal arrows", "buy sell"), {"fast": 9, "slow": 21}, _c_cross),
    Component("atrstop", "ATR stop levels", True, ("atr stop", "trailing stop", "stop loss levels", "atr levels"), {"period": 14, "mult": 2.0}, _c_atr_stop),
    Component("volspike", "Volume spikes", True, ("volume spike", "volume surge", "high volume"), {"period": 20, "mult": 2.0}, _c_volspike),
    Component("rsi", "RSI", False, ("rsi", "relative strength"), {"period": 14}, _c_rsi),
    Component("macd", "MACD", False, ("macd",), {"fast": 12, "slow": 26, "signal": 9}, _c_macd),
    Component("stoch", "Stochastic", False, ("stochastic", "stoch"), {"period": 14}, _c_stoch),
    Component("adx", "ADX / DMI", False, ("adx", "dmi", "trend strength"), {"period": 14}, _c_adx),
    Component("atr", "ATR", False, ("atr", "average true range", "volatility"), {"period": 14}, _c_atr),
    Component("cci", "CCI", False, ("cci", "commodity channel"), {"period": 20}, _c_cci),
    Component("mfi", "MFI", False, ("mfi", "money flow"), {"period": 14}, _c_mfi),
    Component("obv", "OBV", False, ("obv", "on balance volume", "on-balance"), {}, _c_obv),
    Component("cvd", "CVD (proxy)", False, ("cvd", "cumulative delta", "delta"), {}, _c_cvd),
    Component("relvol", "Relative volume", False, ("relative volume", "rvol"), {"period": 20}, _c_relvol),
    Component("roc", "Rate of change", False, ("roc", "momentum", "rate of change"), {"period": 10}, _c_roc),
]
_BY_ID = {c.id: c for c in CATALOG}
# 'atr stop' must win over bare 'atr'; 'relative volume' over 'rsi' style words, etc.: longest keyword match wins per component.


def _numbers_after(text: str, kw_start: int, limit: int = 3) -> list[int]:
    tail = text[kw_start: kw_start + 40]
    m = re.match(r"[^0-9a-z]{0,12}((?:\d{1,4}\s*(?:,|/|and|&|-|\s)\s*){0,%d}\d{1,4})" % (limit - 1), tail.split("\n")[0])
    return [int(x) for x in re.findall(r"\d{1,4}", m.group(1))] if m else []


def pick_components(idea: str) -> list[tuple[Component, dict]]:
    text = " " + idea.lower() + " "
    hits: list[tuple[int, Component, int]] = []   # (position, component, keyword length)
    claimed: list[tuple[int, int]] = []
    candidates = sorted(((len(k), c, k) for c in CATALOG for k in c.keywords), key=lambda t: -t[0])
    chosen: dict[str, tuple[int, int]] = {}
    for klen, comp, kw in candidates:
        for m in re.finditer(r"(?<![a-z])" + re.escape(kw) + r"(?:s|es)?(?![a-z])", text):
            span = (m.start(), m.end())
            if any(not (span[1] <= a or span[0] >= b) for a, b in claimed):
                continue
            claimed.append(span)
            if comp.id not in chosen:
                chosen[comp.id] = span
            break
    out = []
    for comp_id, span in sorted(chosen.items(), key=lambda kv: kv[1][0]):
        comp = _BY_ID[comp_id]
        params = dict(comp.defaults)
        nums = _numbers_after(text, span[1])
        if nums:
            if "periods" in params:
                params["periods"] = [n for n in nums if 1 <= n <= 1000][:5] or params["periods"]
            elif comp_id == "macd" and len(nums) >= 3:
                params.update(fast=nums[0], slow=nums[1], signal=nums[2])
            elif comp_id == "cross" and len(nums) >= 2:
                params.update(fast=min(nums[:2]), slow=max(nums[:2]))
            elif "period" in params and 1 <= nums[0] <= 1000:
                params["period"] = nums[0]
        out.append((comp, params))
    return out


def _title_from(idea: str, comps: list[Component]) -> str:
    base = " + ".join(c.title for c in comps[:3])
    if len(comps) > 3:
        base += f" +{len(comps) - 3}"
    return base or "Custom indicator"


def assemble(parts: list[tuple[Component, dict]], title: str | None = None, overlay: bool | None = None) -> str:
    if overlay is None:
        overlay = all(c.overlay for c, _ in parts)
    seen_inputs: set[str] = set()
    body: list[str] = []
    plots: list[str] = []
    for comp, params in parts:
        b, p = comp.render(params)
        for ln in b:
            key = ln.split("=")[0].strip()
            if ln.startswith("[") or key not in seen_inputs or True:
                body.append(ln)
        plots += p
    title = (title or _title_from("", [c for c, _ in parts])).replace('"', "'")[:60]
    short = title if len(title) <= 24 else title[:23].rstrip(" +") + "…"
    head = [f"//@version={PINE_VERSION}", f'indicator("{title}", shorttitle="{short}", overlay={"true" if overlay else "false"}, max_labels_count=100)', ""]
    return "\n".join(head + body + [""] + plots) + "\n"


@dataclass
class IndicatorResult:
    code: str
    title: str
    source: str                       # "builder" | "llm" | "user-code"
    issues: list[dict] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    extra_scripts: list[dict] = field(default_factory=list)   # e.g. the pane indicators when the idea mixed overlay + pane studies
    ok: bool = False
    verified_by: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    grammar: Optional[bool] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _finalize(code: str, source: str, title: str, comps: list[str], changes: list[str] | None = None, notes: list[str] | None = None,
              extra: list[dict] | None = None) -> IndicatorResult:
    issues = lint(code)
    g_ok, g_msg = grammar_check(code)
    verified = ["built-in Pine v5 linter"] + ([g_msg] if g_ok else [])
    notes = list(notes or [])
    if g_ok is None:
        notes.append(g_msg)
    elif g_ok is False:
        issues.append(Issue("error", 0, g_msg))
    errors = [i for i in issues if i.level == "error"]
    notes.append("TradingView's own compiler could not be run here: paste the code into the Pine Editor and press 'Add to chart' as the final check.")
    return IndicatorResult(code=code, title=title, source=source, issues=[i.to_dict() for i in issues], changes=changes or [], components=comps,
                           extra_scripts=extra or [], ok=not errors, verified_by=verified, notes=notes, grammar=g_ok)


def build_from_idea(idea: str) -> IndicatorResult:
    parts = pick_components(idea)
    notes: list[str] = []
    if not parts:
        parts = [(_BY_ID["ema"], {"periods": [20, 50]}), (_BY_ID["rsi"], dict(_BY_ID["rsi"].defaults))]
        notes.append("No known indicator names were recognised in the idea, so a starter EMA(20/50) + RSI(14) set was generated. Name the studies you want "
                     "(e.g. 'EMA 21, Bollinger bands, RSI 14, liquidity sweeps') for a closer match.")
    overlay_parts = [(c, p) for c, p in parts if c.overlay]
    pane_parts = [(c, p) for c, p in parts if not c.overlay]
    comps = [c.title for c, _ in parts]
    extra: list[dict] = []
    if overlay_parts and pane_parts:
        main = assemble(overlay_parts)
        pane = assemble(pane_parts)
        notes.append("Your idea mixes price-overlay studies and oscillator studies. TradingView scripts are either on the price chart or in their own pane, so it was "
                     "split into two scripts: the overlay (below) and a separate pane script (see 'Additional script').")
        res = _finalize(main, "builder", _title_from(idea, [c for c, _ in overlay_parts]), comps, notes=notes)
        pr = _finalize(pane, "builder", _title_from(idea, [c for c, _ in pane_parts]), [c.title for c, _ in pane_parts])
        res.extra_scripts = [{"title": pr.title, "code": pr.code, "ok": pr.ok, "issues": pr.issues}]
        res.ok = res.ok and pr.ok
        return res
    code = assemble(parts)
    return _finalize(code, "builder", _title_from(idea, [c for c, _ in parts]), comps, notes=notes)


def check_and_fix(code: str) -> IndicatorResult:
    fixed, changes = autofix(code)
    m = re.search(r'indicator\(\s*"([^"]+)"', fixed)
    return _finalize(fixed, "user-code", m.group(1) if m else "Indicator", [], changes)


# ---------------------------------------------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior TradingView Pine Script developer. Write ONE Pine Script v5 INDICATOR that compiles on the first try.
Hard rules:
- First line exactly //@version=5 ; second statement is indicator("Title", overlay=true|false). Never strategy(), never study().
- Use only v5 namespaces: ta.sma ta.ema ta.rsi ta.atr ta.macd ta.stoch ta.bb ta.crossover ta.highest ta.lowest ta.pivothigh ..., math.*, str.*, request.security, input.int/float/bool/string/source.
- Indent blocks with exactly 4 spaces. Wrapped lines must NOT be indented by a multiple of 4.
- plot(), plotshape(), hline(), bgcolor(), fill(), alertcondition() are called at global scope only (never inside if/for/functions).
- Declare a variable once with '='; change it later with ':='. Use 'var' for values that persist across bars.
- Tuple functions return tuples: [mid, up, lo] = ta.bb(close, 20, 2). ta.dmi returns [diPlus, diMinus, adx].
- Do not use lookahead_on without a [1] offset (repainting). At most 60 plots. Add input() controls with sensible defaults and group names.
- Do not invent functions. If unsure, use a simpler standard construction.
Output ONLY the code in one ```pine fenced block. No prose."""


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:pine|pinescript|pine-script)?\s*\n(.*?)```", text, re.S | re.I)
    code = (m.group(1) if m else text).strip("\n")
    return code + "\n"


def _repair_prompt(idea: str, code: str, issues: list[Issue]) -> str:
    msgs = "\n".join(f"- line {i.line}: {i.message}" + (f" ({i.fix})" if i.fix else "") for i in issues if i.level == "error")[:3000]
    return (f"Idea: {idea}\n\nThe script below has compile problems. Fix ALL of them and return the complete corrected script in one ```pine block.\n\n"
            f"Problems:\n{msgs}\n\nScript:\n```pine\n{code}\n```")


def generate_with_llm(idea: str, *, complete: Callable[[str, str], tuple[Optional[str], Optional[str]]] | None = None, max_repairs: int = 3) -> IndicatorResult:
    """``complete(prompt, system) -> (text|None, error|None)``; defaults to the app's configured model. Falls back to the offline builder."""
    if complete is None:
        from app.ai.llm_client import complete_text

        def complete(prompt, system):  # type: ignore[no-redef]
            return complete_text(prompt, system=system)

    notes: list[str] = []
    text, err = complete(f"Create a TradingView indicator for this idea:\n{idea}", SYSTEM_PROMPT)
    if text is None:
        fb = build_from_idea(idea)
        fb.notes.insert(0, f"AI was unavailable ({err}); built from the offline component library instead.")
        return fb
    code, changes = autofix(_extract_code(text))
    for attempt in range(max_repairs + 1):
        issues = lint(code)
        g_ok, _ = grammar_check(code)
        errors = [i for i in issues if i.level == "error"]
        if not errors and g_ok is not False:
            m = re.search(r'indicator\(\s*"([^"]+)"', code)
            res = _finalize(code, "llm", m.group(1) if m else "Indicator", [], changes, notes + ([f"AI draft needed {attempt} repair round(s)."] if attempt else []))
            return res
        if attempt == max_repairs:
            break
        extra = list(errors)
        if g_ok is False:
            extra.append(Issue("error", 0, "The script does not parse as Pine v5 (check brackets, indentation and statement syntax)."))
        text, err = complete(_repair_prompt(idea, code, extra), SYSTEM_PROMPT)
        if text is None:
            break
        code, more = autofix(_extract_code(text))
        changes += more
    fb = build_from_idea(idea)
    fb.notes.insert(0, "The AI draft still had compile problems after repair attempts, so the offline component library was used instead. "
                       "Its last draft is available under 'AI draft (unverified)'.")
    fb.extra_scripts.append({"title": "AI draft (unverified)", "code": code, "ok": False, "issues": [i.to_dict() for i in lint(code)]})
    return fb
