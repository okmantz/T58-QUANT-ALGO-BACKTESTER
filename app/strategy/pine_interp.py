"""
Pine Script interpreter -- the fallback behind app.strategy.pinescript.PineScriptStrategy.

The fast line-based parser in pinescript.py handles the common shapes (and is vectorized, so the optimizer can call it
thousands of times). Real-world scripts use more of the language: `var` state, `:=` re-assignment, if/else blocks that
assign values, for/while loops, user functions, arrays, ternaries, history references on any series (`x[1]`), the full
ta.* library, `request.security()` for higher timeframes and strategy.exit() with stop/limit/profit/loss. This module
runs those scripts bar by bar and returns the same +1/0/-1 signal series + stop/target distances every adapter returns.

What is simulated: Pine's execution model (script runs on each bar close, orders fill at the next bar open, one net
position, pyramiding from the strategy() header). What is not: intrabar order logic beyond stop/limit hit tests,
margin, commissions (the backtest engine applies its own costs), labels/lines/boxes/tables (accepted and ignored),
user-defined types and methods, maps/matrices (rejected with a clear message).

Never guesses: an unknown function or syntax raises StrategyError naming it.
"""
from __future__ import annotations

import math
import re
import statistics
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.strategy.base import StrategyError, StrategyResult
from app.strategy.interp_common import (
    NAN, SimBroker, bar_minutes, completed_map, guess_mintick, isnan, parse_tf_minutes, resample_to, time_index,
)

# ---------------------------------------------------------------------------------------------------------------
# Lexing / parsing
# ---------------------------------------------------------------------------------------------------------------
_TOKEN = re.compile(
    r"""\s*(?:
        (?P<num>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)
      | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
      | (?P<color>\#[0-9A-Fa-f]{6,8})
      | (?P<name>[A-Za-z_]\w*)
      | (?P<op>:=|==|!=|<=|>=|=>|\+=|-=|\*=|/=|%=|[-+*/%<>=()\[\],.?:])
    )""", re.X)

_TYPE_WORDS = {"int", "float", "bool", "string", "color", "label", "line", "box", "table", "series", "simple", "const",
               "input", "linefill", "polyline", "chart.point"}
_KEYWORDS = {"if", "else", "for", "to", "by", "while", "switch", "break", "continue", "var", "varip", "import",
             "export", "and", "or", "not", "true", "false", "in"}


class _Tok:
    __slots__ = ("kind", "val")

    def __init__(self, kind, val):
        self.kind, self.val = kind, val

    def __repr__(self):  # pragma: no cover
        return f"{self.kind}:{self.val}"


def _tokenize(text: str) -> list[_Tok]:
    out, pos, n = [], 0, len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            raise StrategyError(f"PineScript: cannot read '{text[pos:pos + 20].strip()}' (unsupported syntax).")
        pos = m.end()
        kind = m.lastgroup
        out.append(_Tok(kind, m.group(kind)))
    return out


class _Parser:
    """Pratt parser over one logical line's tokens. AST nodes are tuples; `id` slots give each call/index site its own
    persistent state."""

    def __init__(self, toks: list[_Tok], ids: Callable[[], int]):
        self.t, self.p, self.nid = toks, 0, ids

    def peek(self, k=0):
        i = self.p + k
        return self.t[i] if i < len(self.t) else None

    def eat(self):
        tok = self.t[self.p]
        self.p += 1
        return tok

    def is_op(self, v, k=0):
        tk = self.peek(k)
        return tk is not None and tk.kind == "op" and tk.val == v

    def is_kw(self, v):
        tk = self.peek()
        return tk is not None and tk.kind == "name" and tk.val == v

    def expect(self, v):
        if not self.is_op(v):
            got = self.peek()
            raise StrategyError(f"PineScript: expected '{v}' but found '{got.val if got else 'end of line'}'.")
        self.p += 1

    def done(self):
        return self.p >= len(self.t)

    # precedence climbing ---------------------------------------------------------------------------
    def expr(self):
        return self.ternary()

    def ternary(self):
        c = self.or_()
        if self.is_op("?"):
            self.p += 1
            a = self.ternary()
            self.expect(":")
            b = self.ternary()
            return ("tern", c, a, b)
        return c

    def or_(self):
        left = self.and_()
        while self.is_kw("or"):
            self.p += 1
            left = ("or", left, self.and_())
        return left

    def and_(self):
        left = self.eq()
        while self.is_kw("and"):
            self.p += 1
            left = ("and", left, self.eq())
        return left

    def eq(self):
        left = self.cmp()
        while self.peek() and self.peek().kind == "op" and self.peek().val in ("==", "!="):
            op = self.eat().val
            left = ("bin", op, left, self.cmp())
        return left

    def cmp(self):
        left = self.add()
        while self.peek() and self.peek().kind == "op" and self.peek().val in ("<", ">", "<=", ">="):
            op = self.eat().val
            left = ("bin", op, left, self.add())
        return left

    def add(self):
        left = self.mul()
        while self.peek() and self.peek().kind == "op" and self.peek().val in ("+", "-"):
            op = self.eat().val
            left = ("bin", op, left, self.mul())
        return left

    def mul(self):
        left = self.unary()
        while self.peek() and self.peek().kind == "op" and self.peek().val in ("*", "/", "%"):
            op = self.eat().val
            left = ("bin", op, left, self.unary())
        return left

    def unary(self):
        tk = self.peek()
        if tk is not None and tk.kind == "op" and tk.val in ("-", "+"):
            self.p += 1
            e = self.unary()
            return ("un", tk.val, e)
        if tk is not None and tk.kind == "name" and tk.val == "not":
            self.p += 1
            return ("not", self.unary())
        return self.postfix()

    def postfix(self):
        e = self.primary()
        while True:
            if self.is_op("."):
                self.p += 1
                nm = self.eat()
                if nm.kind != "name":
                    raise StrategyError("PineScript: expected a name after '.'.")
                if e[0] == "name":
                    e = ("name", e[1] + "." + nm.val)
                else:
                    e = ("attr", e, nm.val)
            elif self.is_op("("):
                e = self.call(e)
            elif self.is_op("["):
                self.p += 1
                idx = self.expr()
                self.expect("]")
                e = ("idx", e, idx, self.nid())
            else:
                return e

    def call(self, fn):
        self.expect("(")
        args, kwargs = [], {}
        while not self.is_op(")"):
            tk, nx = self.peek(), self.peek(1)
            if tk.kind == "name" and nx is not None and nx.kind == "op" and nx.val == "=":
                self.p += 2
                kwargs[tk.val] = self.expr()
            else:
                args.append(self.expr())
            if self.is_op(","):
                self.p += 1
            elif not self.is_op(")"):
                raise StrategyError("PineScript: expected ',' or ')' in a call.")
        self.expect(")")
        return ("call", fn, args, kwargs, self.nid())

    def primary(self):
        tk = self.peek()
        if tk is None:
            raise StrategyError("PineScript: unexpected end of expression.")
        if tk.kind == "num":
            self.p += 1
            v = float(tk.val)
            return ("num", int(v) if re.fullmatch(r"\d+", tk.val) else v)
        if tk.kind == "str":
            self.p += 1
            return ("str", bytes(tk.val[1:-1], "utf-8").decode("unicode_escape"))
        if tk.kind == "color":
            self.p += 1
            return ("num", 0)
        if tk.kind == "name":
            self.p += 1
            if tk.val == "true":
                return ("num", True)
            if tk.val == "false":
                return ("num", False)
            return ("name", tk.val)
        if tk.kind == "op" and tk.val == "(":
            self.p += 1
            e = self.expr()
            self.expect(")")
            return e
        if tk.kind == "op" and tk.val == "[":
            self.p += 1
            items = []
            while not self.is_op("]"):
                items.append(self.expr())
                if self.is_op(","):
                    self.p += 1
            self.expect("]")
            return ("tuple", items)
        raise StrategyError(f"PineScript: unexpected '{tk.val}'.")


# ---- statements ---------------------------------------------------------------------------------------------
_FUNCDEF = re.compile(r"^(?:export\s+)?(?:method\s+)?([A-Za-z_]\w*)\s*\(([^()]*)\)\s*=>\s*(.*)$")
_ASSIGN_OPS = (":=", "+=", "-=", "*=", "/=", "%=", "=")
_NS_PREFIX = re.compile(r"\.new<[^>]*>\(")


def _strip_comment(line: str) -> str:
    out, q, i = [], None, 0
    while i < len(line):
        ch = line[i]
        if q:
            out.append(ch)
            if ch == "\\" and i + 1 < len(line):
                out.append(line[i + 1])
                i += 1
            elif ch == q:
                q = None
        else:
            if ch in "\"'":
                q = ch
                out.append(ch)
            elif ch == "/" and line[i:i + 2] == "//":
                break
            else:
                out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _logical_lines(code: str) -> list[tuple[int, str]]:
    """(indent, text) per logical statement. A line continues when brackets are open, it ends in an operator/comma, or the
    next line is indented deeper than the block but is not a new block (Pine's wrapped-line rule: indent not a multiple of
    4 spaces / deeper continuation)."""
    raw = [(_strip_comment(l.expandtabs(4))) for l in code.splitlines()]
    out: list[tuple[int, str]] = []
    buf, depth, ind = "", 0, 0
    for line in raw:
        if not line.strip():
            continue
        if buf == "":
            ind = len(line) - len(line.lstrip(" "))
            buf = line.strip()
        else:
            buf += " " + line.strip()
        depth = 0
        q = None
        for ch in buf:
            if q:
                if ch == q:
                    q = None
            elif ch in "\"'":
                q = ch
            elif ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
        tail = buf.rstrip()
        cont = depth > 0 or re.search(r"(?:[-+*/%<>=,?:(\[]|\band|\bor|=>)$", tail) is not None and not tail.endswith("=>")
        if cont and depth <= 0 and tail.endswith(("=>",)):
            cont = False
        if not cont:
            out.append((ind, buf))
            buf = ""
    if buf:
        out.append((ind, buf))
    # second pass: lines starting with an operator belong to the previous statement
    merged: list[tuple[int, str]] = []
    for ind, txt in out:
        if merged and re.match(r"^(?:and\b|or\b|[?:+\-*/<>=%.])", txt) and not txt.startswith(("=>",)) and \
                ind > merged[-1][0] and not re.match(r"^(?:-?\d)", txt[1:2] or ""):
            merged[-1] = (merged[-1][0], merged[-1][1] + " " + txt)
        else:
            merged.append((ind, txt))
    return merged


class _Stmt:
    pass


def _find_assign(toks: list[_Tok]):
    depth = 0
    for i, t in enumerate(toks):
        if t.kind == "op":
            if t.val in "([":
                depth += 1
            elif t.val in ")]":
                depth -= 1
            elif depth == 0 and t.val in _ASSIGN_OPS:
                return i, t.val
    return None, None


class _BlockParser:
    def __init__(self, lines: list[tuple[int, str]]):
        self.lines, self.i = lines, 0
        self._id = 0
        self.funcs: dict[str, tuple] = {}
        self.header: dict[str, Any] = {}

    def nid(self):
        self._id += 1
        return self._id

    def parse_all(self):
        return self.block(self.lines[0][0] if self.lines else 0)

    def block(self, indent: int):
        stmts = []
        while self.i < len(self.lines):
            ind, txt = self.lines[self.i]
            if ind < indent:
                break
            if ind > indent:
                # unexpected deeper line: treat as continuation of the block
                indent = ind
            s = self.statement(ind, txt)
            if s is not None:
                stmts.append(s)
        return stmts

    def child_block(self, parent_indent: int):
        if self.i < len(self.lines) and self.lines[self.i][0] > parent_indent:
            return self.block(self.lines[self.i][0])
        return []

    def expr_from(self, text: str):
        toks = _tokenize(_NS_PREFIX.sub(".new_any(", text))
        p = _Parser(toks, self.nid)
        e = p.expr()
        if not p.done():
            raise StrategyError(f"PineScript: cannot parse '{text}' (leftover '{p.peek().val}').")
        return e

    def statement(self, ind: int, txt: str):
        self.i += 1
        txt = txt.strip()
        if txt.startswith(("import ", "type ", "enum ", "indicator(", "library(")) or txt.startswith("//@"):
            if txt.startswith(("type ", "enum ")):
                raise StrategyError("PineScript: user-defined types/enums are not supported by the interpreter.")
            return None
        head = re.match(r"^([A-Za-z_]\w*)", txt)
        kw = head.group(1) if head else ""
        if kw == "if" and (len(txt) == 2 or not txt[2].isalnum() and txt[2] != "_"):
            return self.if_stmt(ind, txt)
        if kw == "for":
            return self.for_stmt(ind, txt)
        if kw == "while":
            cond = self.expr_from(txt[5:])
            body = self.child_block(ind)
            return ("while", cond, body)
        if kw == "switch":
            return self.switch_stmt(ind, txt)
        if txt in ("break", "continue"):
            return (txt,)
        m = _FUNCDEF.match(txt)
        if m and kw not in ("if", "for", "while", "switch") and not re.match(r"^[A-Za-z_]\w*\s*\(.*\)\s*=\s", txt):
            name, params, body = m.groups()
            if txt.startswith("method "):
                raise StrategyError("PineScript: methods are not supported by the interpreter.")
            plist = []
            for prm in [x.strip() for x in params.split(",") if x.strip()]:
                nm, _, dflt = prm.partition("=")
                nm = nm.strip().split()[-1]
                plist.append((nm, self.expr_from(dflt) if dflt.strip() else None))
            if body.strip():
                fb = [("expr", self.expr_from(body))]
            else:
                fb = self.child_block(ind)
            self.funcs[name] = (plist, fb)
            return None
        toks = _tokenize(_NS_PREFIX.sub(".new_any(", txt))
        pos, op = _find_assign(toks)
        if pos is not None:
            left, right = toks[:pos], toks[pos + 1:]
            persist = False
            while left and left[0].kind == "name" and left[0].val in ("var", "varip"):
                persist = True
                left = left[1:]
            # strip leading type words ("float x", "series int x", "array<float> a" has been normalised)
            while len(left) >= 2 and left[0].kind == "name" and left[1].kind == "name" and \
                    (left[0].val in _TYPE_WORDS or left[0].val.startswith("array")):
                left = left[1:]
            if len(left) >= 2 and left[0].kind == "name" and left[1].kind == "name":
                left = left[1:]            # e.g. a user type name; keep the variable name
            tgt = None
            if len(left) == 1 and left[0].kind == "name":
                tgt = ("var", left[0].val)
            elif left and left[0].kind == "op" and left[0].val == "[" and left[-1].val == "]":
                names = [t.val for t in left[1:-1] if t.kind == "name"]
                tgt = ("tuple", names)
            else:
                raise StrategyError(f"PineScript: unsupported assignment target near '{txt[:40]}'.")
            if not right:
                # value comes from the following block (x = \n if ...)  -- rare; treat as block expression
                raise StrategyError("PineScript: empty right-hand side.")
            first = right[0]
            if first.kind == "name" and first.val in ("if", "for", "while", "switch") and len(right) >= 1:
                rhs_stmt = self.compound_from_tokens(right, ind)
                return ("assign", tgt, op, ("blockexpr", rhs_stmt), persist, self.nid())
            node = self.expr_from_tokens(right)
            return ("assign", tgt, op, node, persist, self.nid())
        return ("expr", self.expr_from(txt))

    def expr_from_tokens(self, toks):
        p = _Parser(toks, self.nid)
        e = p.expr()
        if not p.done():
            raise StrategyError(f"PineScript: cannot parse expression (leftover '{p.peek().val}').")
        return e

    def compound_from_tokens(self, toks, ind):
        text = " ".join(("'%s'" % t.val if t.kind == "str" else t.val) for t in toks)
        kw = toks[0].val
        if kw == "if":
            return self.if_stmt(ind, text, consumed=True)
        if kw == "for":
            return self.for_stmt(ind, text, consumed=True)
        if kw == "switch":
            return self.switch_stmt(ind, text, consumed=True)
        raise StrategyError("PineScript: unsupported block expression.")

    def if_stmt(self, ind: int, txt: str, consumed=False):
        cond = self.expr_from(txt[2:])
        body = self.child_block(ind)
        orelse = []
        if self.i < len(self.lines) and self.lines[self.i][0] == ind and re.match(r"^else\b", self.lines[self.i][1]):
            _, etxt = self.lines[self.i]
            self.i += 1
            rest = etxt[4:].strip()
            if rest.startswith("if"):
                orelse = [self.if_stmt(ind, rest, consumed=True)]
            elif rest:
                orelse = [("expr", self.expr_from(rest))]
            else:
                orelse = self.child_block(ind)
        return ("if", cond, body, orelse)

    def for_stmt(self, ind: int, txt: str, consumed=False):
        m = re.match(r"^for\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s+to\s+(.+?)(?:\s+by\s+(.+))?$", txt)
        if m:
            var, a, b, step = m.groups()
            body = self.child_block(ind)
            return ("for", var, self.expr_from(a), self.expr_from(b), self.expr_from(step) if step else None, body)
        m = re.match(r"^for\s+([A-Za-z_]\w*)\s+in\s+(.+)$", txt)
        if m:
            var, arr = m.groups()
            return ("foreach", var, self.expr_from(arr), self.child_block(ind))
        raise StrategyError(f"PineScript: unsupported for-loop '{txt[:50]}'.")

    def switch_stmt(self, ind: int, txt: str, consumed=False):
        subject = txt[6:].strip()
        subj = self.expr_from(subject) if subject else None
        cases, default = [], None
        if self.i < len(self.lines) and self.lines[self.i][0] > ind:
            cind = self.lines[self.i][0]
            while self.i < len(self.lines) and self.lines[self.i][0] == cind:
                _, ctxt = self.lines[self.i]
                self.i += 1
                head, sep, body = self._split_arrow(ctxt)
                if not sep:
                    raise StrategyError("PineScript: malformed switch case.")
                if body.strip():
                    bstmts = [("expr", self.expr_from(body))]
                else:
                    bstmts = self.child_block(cind)
                if head.strip() == "":
                    default = bstmts
                else:
                    cases.append((self.expr_from(head), bstmts))
        return ("switch", subj, cases, default)

    @staticmethod
    def _split_arrow(s):
        depth, q = 0, None
        for i in range(len(s) - 1):
            ch = s[i]
            if q:
                if ch == q:
                    q = None
            elif ch in "\"'":
                q = ch
            elif ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif depth == 0 and s[i:i + 2] == "=>":
                return s[:i], "=>", s[i + 2:]
        return s, "", ""


# ---------------------------------------------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------------------------------------------
class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Frame:
    __slots__ = ("vars", "hist", "state", "inited")

    def __init__(self):
        self.vars: dict[str, Any] = {}
        self.hist: dict[str, list] = {}
        self.state: dict[Any, Any] = {}
        self.inited: set = set()


def _num(x):
    if x is None:
        return NAN
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    return x


def _truthy(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float):
        return x == x and x != 0.0
    return bool(x)


def _arith(op, a, b):
    if isinstance(a, str) or isinstance(b, str):
        if op == "+":
            return str(a) + str(b)
    a, b = _num(a), _num(b)
    try:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b if b != 0 else (NAN if a == 0 or a != a else math.copysign(math.inf, a))
        if op == "%":
            return math.fmod(a, b) if b != 0 else NAN
    except (TypeError, ValueError):
        return NAN
    raise StrategyError(f"PineScript: operator {op}")


def _cmp(op, a, b):
    if a is None or b is None:
        return False
    if isinstance(a, float) and a != a or isinstance(b, float) and b != b:
        return op == "!="
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b
    if op == ">":
        return a > b
    if op == "<=":
        return a <= b
    return a >= b


class PineRuntime:
    def __init__(self, df: pd.DataFrame, parser: _BlockParser, stmts, overrides=None, is_child=False,
                 parent: "PineRuntime | None" = None):
        self.df = df
        self.n = len(df)
        self.On = df["open"].to_numpy(float)
        self.Hn = df["high"].to_numpy(float)
        self.Ln = df["low"].to_numpy(float)
        self.Cn = df["close"].to_numpy(float)
        self.Vn = df["volume"].to_numpy(float) if "volume" in df.columns else np.zeros(self.n)
        self.O, self.H, self.L, self.C, self.V = (self.On.tolist(), self.Hn.tolist(), self.Ln.tolist(),
                                                  self.Cn.tolist(), self.Vn.tolist())
        idx = time_index(df)
        self.idx = idx
        self.TS = (idx.values.astype("datetime64[s]").astype("int64") * 1000).astype(float)
        self.HOUR = idx.hour.to_numpy()
        self.MIN = idx.minute.to_numpy()
        self.DOW = ((idx.dayofweek.to_numpy() + 1) % 7 + 1)      # Pine: Sunday=1 .. Saturday=7
        self.MONTH = idx.month.to_numpy()
        self.YEAR = idx.year.to_numpy()
        self.DAY = idx.day.to_numpy()
        self.bar_min = bar_minutes(df)
        self.mintick = guess_mintick(df)
        self.parser = parser
        self.stmts = stmts
        self.funcs = parser.funcs
        self.overrides = overrides or {}
        self.i = 0
        self.g = _Frame()
        self.frame = self.g
        self.call_frames: dict[int, _Frame] = {}
        self.broker: SimBroker | None = None
        self.sec_cache: dict = {}
        self.is_child = is_child
        self.parent = parent
        self.warnings: list[str] = []
        self.arrays_seen = False

    # -- variable access --------------------------------------------------------------------------------
    def lookup(self, name: str):
        fr = self.frame
        if name in fr.vars:
            return fr.vars[name]
        if fr is not self.g and name in self.g.vars:
            return self.g.vars[name]
        return self.builtin(name)

    def builtin(self, name: str):
        i = self.i
        b = _BUILTIN_SERIES.get(name)
        if b is not None:
            return b(self, i)
        if name in _CONSTS:
            return _CONSTS[name]
        if name in self.funcs:
            raise StrategyError(f"PineScript: function '{name}' used without ()")
        if name == "na":
            return NAN
        raise StrategyError(f"PineScript: unknown variable or function '{name}'.")

    def hist_at(self, name: str, k: int):
        j = self.i - k
        if j < 0:
            return NAN
        fr = self.frame if name in self.frame.vars or self.frame is self.g else self.g
        if name not in fr.vars and name in self.g.vars:
            fr = self.g
        h = fr.hist.get(name)
        if h is not None:
            if k == 0:
                return fr.vars.get(name, NAN)
            return h[j] if j < len(h) else NAN
        # builtin series
        arr = _BUILTIN_ARRAYS.get(name)
        if arr is not None:
            return arr(self)[j]
        if name in fr.vars:
            return NAN
        raise StrategyError(f"PineScript: cannot take history of '{name}'.")


_BUILTIN_SERIES: dict[str, Callable] = {
    "open": lambda r, i: r.O[i], "high": lambda r, i: r.H[i], "low": lambda r, i: r.L[i], "close": lambda r, i: r.C[i],
    "volume": lambda r, i: r.V[i],
    "hl2": lambda r, i: (r.H[i] + r.L[i]) / 2.0, "hlc3": lambda r, i: (r.H[i] + r.L[i] + r.C[i]) / 3,
    "ohlc4": lambda r, i: (r.O[i] + r.H[i] + r.L[i] + r.C[i]) / 4,
    "hlcc4": lambda r, i: (r.H[i] + r.L[i] + 2 * r.C[i]) / 4,
    "ta.tr": lambda r, i: float(_tr_vec(r)[i]), "ta.obv": lambda r, i: float(_obv_vec(r)[i]),
    "ta.accdist": lambda r, i: float(_accdist_vec(r)[i]), "ta.pvt": lambda r, i: float(_pvt_vec(r)[i]),
    "ta.vwap": lambda r, i: float(_vwap_vec(r)[i]),
    "time": lambda r, i: r.TS[i], "time_close": lambda r, i: r.TS[i] + r.bar_min * 60000.0,
    "time_tradingday": lambda r, i: r.TS[i],
    "timenow": lambda r, i: r.TS[-1], "last_bar_time": lambda r, i: r.TS[-1],
    "bar_index": lambda r, i: i, "last_bar_index": lambda r, i: r.n - 1,
    "hour": lambda r, i: int(r.HOUR[i]), "minute": lambda r, i: int(r.MIN[i]), "second": lambda r, i: 0,
    "dayofweek": lambda r, i: int(r.DOW[i]), "month": lambda r, i: int(r.MONTH[i]), "year": lambda r, i: int(r.YEAR[i]),
    "dayofmonth": lambda r, i: int(r.DAY[i]),
    "weekofyear": lambda r, i: int(r.idx[i].isocalendar()[1]),
    "barstate.isfirst": lambda r, i: i == 0, "barstate.islast": lambda r, i: i == r.n - 1,
    "barstate.isconfirmed": lambda r, i: True, "barstate.ishistory": lambda r, i: True,
    "barstate.isrealtime": lambda r, i: False, "barstate.isnew": lambda r, i: True,
    "barstate.islastconfirmedhistory": lambda r, i: i == r.n - 1,
    "syminfo.mintick": lambda r, i: r.mintick, "syminfo.pointvalue": lambda r, i: 1.0,
    "syminfo.tickerid": lambda r, i: "SYMBOL", "syminfo.ticker": lambda r, i: "SYMBOL",
    "syminfo.currency": lambda r, i: "USD", "syminfo.type": lambda r, i: "futures", "syminfo.timezone": lambda r, i: "UTC",
    "syminfo.prefix": lambda r, i: "", "syminfo.root": lambda r, i: "SYMBOL",
    "timeframe.period": lambda r, i: _period_label(r.bar_min), "timeframe.multiplier": lambda r, i: _tf_mult(r.bar_min),
    "timeframe.isintraday": lambda r, i: r.bar_min < 1440, "timeframe.isdaily": lambda r, i: 1440 <= r.bar_min < 10080,
    "timeframe.isweekly": lambda r, i: 10080 <= r.bar_min < 43200, "timeframe.ismonthly": lambda r, i: r.bar_min >= 43200,
    "timeframe.isminutes": lambda r, i: r.bar_min < 1440, "timeframe.isseconds": lambda r, i: False,
    "timeframe.isdwm": lambda r, i: r.bar_min >= 1440,
    "session.ismarket": lambda r, i: True, "session.isregular": lambda r, i: True,
    "session.isfirstbar": lambda r, i: i == 0 or r.DAY[i] != r.DAY[i - 1],
    "session.islastbar": lambda r, i: i == r.n - 1 or r.DAY[i] != r.DAY[i + 1],
    "session.isfirstbar_regular": lambda r, i: i == 0 or r.DAY[i] != r.DAY[i - 1],
    "session.islastbar_regular": lambda r, i: i == r.n - 1 or r.DAY[i] != r.DAY[i + 1],
    "session.ispremarket": lambda r, i: False, "session.ispostmarket": lambda r, i: False,
    "strategy.position_size": lambda r, i: r.broker.pos if r.broker else 0,
    "strategy.position_avg_price": lambda r, i: r.broker.avg if r.broker else NAN,
    "strategy.opentrades": lambda r, i: (1 if r.broker and r.broker.pos else 0),
    "strategy.closedtrades": lambda r, i: len(r.broker.closed) if r.broker else 0,
    "strategy.netprofit": lambda r, i: r.broker.netprofit if r.broker else 0.0,
    "strategy.equity": lambda r, i: r.initial_capital + (r.broker.netprofit if r.broker else 0.0) +
    ((r.C[i] - r.broker.avg) * r.broker.pos if r.broker and r.broker.pos and not isnan(r.broker.avg) else 0.0),
    "strategy.initial_capital": lambda r, i: r.initial_capital,
    "strategy.openprofit": lambda r, i: ((r.C[i] - r.broker.avg) * r.broker.pos
                                         if r.broker and r.broker.pos and not isnan(r.broker.avg) else 0.0),
    "strategy.grossprofit": lambda r, i: sum(max(0.0, d * (x - e) * q) for d, e, x, q in (r.broker.closed if r.broker else [])),
    "strategy.grossloss": lambda r, i: -sum(max(0.0, -d * (x - e) * q) for d, e, x, q in (r.broker.closed if r.broker else [])),
    "strategy.wintrades": lambda r, i: sum(1 for d, e, x, q in (r.broker.closed if r.broker else []) if d * (x - e) > 0),
    "strategy.losstrades": lambda r, i: sum(1 for d, e, x, q in (r.broker.closed if r.broker else []) if d * (x - e) < 0),
}
PineRuntime.initial_capital = 100000.0   # type: ignore[attr-defined]

_BUILTIN_ARRAYS: dict[str, Callable] = {
    "open": lambda r: r.On, "high": lambda r: r.Hn, "low": lambda r: r.Ln, "close": lambda r: r.Cn, "volume": lambda r: r.Vn,
    "hl2": lambda r: (r.Hn + r.Ln) / 2, "hlc3": lambda r: (r.Hn + r.Ln + r.Cn) / 3,
    "ohlc4": lambda r: (r.On + r.Hn + r.Ln + r.Cn) / 4, "hlcc4": lambda r: (r.Hn + r.Ln + 2 * r.Cn) / 4,
    "ta.tr": lambda r: _tr_vec(r), "ta.obv": lambda r: _obv_vec(r), "ta.accdist": lambda r: _accdist_vec(r),
    "ta.pvt": lambda r: _pvt_vec(r),
    "time": lambda r: r.TS, "bar_index": lambda r: np.arange(r.n, dtype=float),
    "hour": lambda r: r.HOUR.astype(float), "minute": lambda r: r.MIN.astype(float),
    "dayofweek": lambda r: r.DOW.astype(float), "month": lambda r: r.MONTH.astype(float),
    "year": lambda r: r.YEAR.astype(float), "dayofmonth": lambda r: r.DAY.astype(float),
}


def _tf_mult(m):
    if m >= 43200:
        return int(m // 43200)
    if m >= 10080:
        return int(m // 10080)
    if m >= 1440:
        return int(m // 1440)
    return int(m)


def _period_label(m):
    if m >= 43200:
        return f"{int(m // 43200)}M"
    if m >= 10080:
        return f"{int(m // 10080)}W"
    if m >= 1440:
        return "D" if m == 1440 else f"{int(m // 1440)}D"
    return str(int(m))


_CONSTS: dict[str, Any] = {
    "na": NAN, "true": True, "false": False,
    "strategy.long": "long", "strategy.short": "short", "strategy.cash": "cash", "strategy.fixed": "fixed",
    "strategy.percent_of_equity": "pct", "strategy.direction.long": "long", "strategy.direction.short": "short",
    "strategy.direction.all": "all", "strategy.oca.none": "none", "strategy.oca.cancel": "cancel",
    "strategy.oca.reduce": "reduce",
    "math.pi": math.pi, "math.e": math.e, "math.phi": 1.618033988749895, "math.rphi": 0.6180339887498949,
    "dayofweek.sunday": 1, "dayofweek.monday": 2, "dayofweek.tuesday": 3, "dayofweek.wednesday": 4,
    "dayofweek.thursday": 5, "dayofweek.friday": 6, "dayofweek.saturday": 7,
    "barmerge.lookahead_on": True, "barmerge.lookahead_off": False, "barmerge.gaps_on": True, "barmerge.gaps_off": False,
    "extend.none": 0, "extend.left": 1, "extend.right": 2, "extend.both": 3,
    "size.auto": "auto", "size.tiny": "tiny", "size.small": "small", "size.normal": "normal", "size.large": "large",
    "size.huge": "huge",
    "display.all": 0, "display.none": 0, "display.pane": 0, "display.data_window": 0, "display.status_line": 0,
    "location.top": 0, "location.bottom": 0, "location.abovebar": 0, "location.belowbar": 0, "location.absolute": 0,
    "shape.triangleup": 0, "shape.triangledown": 0, "shape.circle": 0, "shape.cross": 0, "shape.xcross": 0,
    "shape.diamond": 0, "shape.square": 0, "shape.arrowup": 0, "shape.arrowdown": 0, "shape.flag": 0,
    "shape.labelup": 0, "shape.labeldown": 0,
    "plot.style_line": 0, "plot.style_histogram": 0, "plot.style_columns": 0, "plot.style_circles": 0,
    "plot.style_stepline": 0, "plot.style_area": 0, "plot.style_cross": 0, "plot.style_linebr": 0,
    "hline.style_solid": 0, "hline.style_dashed": 0, "hline.style_dotted": 0,
    "line.style_solid": 0, "line.style_dashed": 0, "line.style_dotted": 0,
    "label.style_label_up": 0, "label.style_label_down": 0, "label.style_none": 0,
    "xloc.bar_index": 0, "xloc.bar_time": 0, "yloc.price": 0, "yloc.abovebar": 0, "yloc.belowbar": 0,
    "currency.USD": "USD", "currency.EUR": "EUR", "scale.none": 0, "scale.left": 0, "scale.right": 0,
    "format.price": 0, "format.volume": 0, "format.percent": 0, "format.inherit": 0,
    "alert.freq_once_per_bar": 0, "alert.freq_once_per_bar_close": 0, "alert.freq_all": 0,
    "text.align_left": 0, "text.align_center": 0, "text.align_right": 0,
    "position.top_left": 0, "position.top_right": 0, "position.bottom_left": 0, "position.bottom_right": 0,
    "position.middle_center": 0, "position.top_center": 0, "position.bottom_center": 0,
}
for _c in ("red", "green", "blue", "white", "black", "gray", "silver", "orange", "yellow", "aqua", "fuchsia", "lime",
           "maroon", "navy", "olive", "purple", "teal"):
    _CONSTS["color." + _c] = 0


# ---------------------------------------------------------------------------------------------------------------
# helper vectors (ta.tr, ta.obv, ...) -- implemented in pine_ta, re-exported under the names used above
# ---------------------------------------------------------------------------------------------------------------
from app.strategy import pine_ta as T  # noqa: E402

_tr_vec, _obv_vec, _accdist_vec, _pvt_vec, _vwap_vec = T.tr_vec, T.obv_vec, T.accdist_vec, T.pvt_vec, T.vwap_vec


# ---------------------------------------------------------------------------------------------------------------
# function table.  kind: "p" plain fn(*a, **k) | "s" stateful fn(rt, st, *a, **k) | "r" runtime fn(rt, *a, **k)
# ---------------------------------------------------------------------------------------------------------------
def _noop(*a, **k):
    return None


def _inp(rt, defval=None, *a, **k):
    return defval


def _tf_seconds(tf=None):
    m = parse_tf_minutes(tf)
    return float(m * 60) if m else 0.0


def _timeframe_change(rt, st, tf=None):
    m = parse_tf_minutes(tf) or rt.bar_min
    sec = m * 60.0
    cur = math.floor(rt.TS[rt.i] / 1000.0 / sec)
    prev = st.get("p")
    st["p"] = cur
    return prev is not None and cur != prev


def _time_fn(rt, st, tf=None, session=None, timezone=None):
    """time(tf, session): the bar's time when it is inside `session` ("0930-1600", optional ":23456" day list), else na."""
    if session in (None, ""):
        return rt.TS[rt.i]
    s = str(session).split(":")[0]
    try:
        a, b = s.split("-")
        a, b = int(a[:2]) * 60 + int(a[2:]), int(b[:2]) * 60 + int(b[2:])
    except ValueError:
        return rt.TS[rt.i]
    t = int(rt.HOUR[rt.i]) * 60 + int(rt.MIN[rt.i])
    inside = (a <= t < b) if a < b else (t >= a or t < b)
    if ":" in str(session):
        days = str(session).split(":")[1]
        if days and str(int(rt.DOW[rt.i])) not in days:
            inside = False
    return rt.TS[rt.i] if inside else NAN


def _timestamp(*a, **k):
    vals = [int(x) for x in a if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if len(vals) == 3:
        vals += [0, 0, 0]
    elif len(vals) == 5:
        vals += [0]
    try:
        return pd.Timestamp(*vals[:6]).value / 1e6
    except Exception:  # noqa: BLE001
        return NAN


def _ent(rt, direction="long", qty=None, limit=None, stop=None, comment=None, when=True, **k):
    return _entry(rt, None, direction, qty, limit, stop, when)


def _entry(rt, id_, direction, qty=None, limit=None, stop=None, when=True):
    if not _truthy(when):
        return None
    if rt.broker is None:
        return None
    d = 1 if str(direction) == "long" else -1
    q = _num(qty) if qty is not None and not isnan(_num(qty)) and _num(qty) > 0 else None
    if (limit is not None and not isnan(_num(limit))) or (stop is not None and not isnan(_num(stop))):
        rt.warn_once("limit/stop entry orders are filled as market orders at the next open")
    rt.broker.entry(d, q)
    return None


def _strategy_entry(rt, id=None, direction="long", qty=None, limit=None, stop=None, oca_name=None, oca_type=None,
                    comment=None, alert_message=None, when=True, **k):
    return _entry(rt, id, direction, qty, limit, stop, when)


def _strategy_exit(rt, id=None, from_entry=None, qty=None, qty_percent=None, profit=None, limit=None, loss=None,
                   stop=None, trail_price=None, trail_points=None, trail_offset=None, oca_name=None, comment=None,
                   comment_profit=None, comment_loss=None, comment_trailing=None, alert_message=None,
                   alert_profit=None, alert_loss=None, alert_trailing=None, when=True, **k):
    if not _truthy(when) or rt.broker is None:
        return None
    if trail_price is not None or trail_points is not None or trail_offset is not None:
        rt.warn_once("trailing exits (trail_*) are not simulated by the interpreter; use a fixed stop here or set a trailing stop in the backtest settings")
    rt.broker.set_exit(stop=_num(stop) if stop is not None else None, limit=_num(limit) if limit is not None else None,
                       stop_ticks=_num(loss) if loss is not None else None,
                       limit_ticks=_num(profit) if profit is not None else None)
    return None


def _strategy_close(rt, id=None, comment=None, qty=None, qty_percent=None, alert_message=None, immediately=False,
                    when=True, **k):
    if _truthy(when) and rt.broker is not None:
        rt.broker.close()
    return None


def _strategy_close_all(rt, comment=None, alert_message=None, immediately=False, when=True, **k):
    return _strategy_close(rt, when=when)


def _security(rt, st, args, kwargs, site):
    """request.security(symbol, timeframe, expression, gaps, lookahead): `args` are the compiled (unevaluated) argument
    closures so the expression can be run on the HTF series, never on the current bar."""
    names = ["symbol", "timeframe", "expression", "gaps", "lookahead", "ignore_invalid_symbol", "currency"]
    a = dict(zip(names, args))
    a.update(kwargs)
    tf_fn, ex_fn = a.get("timeframe"), a.get("expression")
    if tf_fn is None or ex_fn is None:
        raise StrategyError("PineScript: request.security needs symbol, timeframe and expression.")
    tf = tf_fn(rt)
    look = _truthy(a["lookahead"](rt)) if "lookahead" in a else False
    mins = parse_tf_minutes(tf) if tf not in (None, "", "timeframe.period") else None
    if not mins or mins <= rt.bar_min * 1.0001:
        return ex_fn(rt)
    key = (site, mins)
    ent = rt.sec_cache.get(key)
    if ent is None:
        htf = resample_to(rt.df, mins)
        if len(htf) < 2:
            raise StrategyError(f"PineScript: not enough data to build the {tf} timeframe for request.security.")
        child = PineRuntime(htf, rt.parser, [], is_child=True, parent=rt)
        child.broker = None
        for nm, val in rt.g.vars.items():
            if isinstance(val, (int, float, str, bool)):
                child.g.vars[nm] = val
        vals = []
        for j in range(len(htf)):
            child.i = j
            vals.append(ex_fn(child))
            child.end_bar()
        pos = completed_map(rt.idx, rt.bar_min, time_index(htf), mins)
        if look:
            if _lagged_only(a["expression_node"]) if "expression_node" in a else False:
                start = time_index(htf).values.astype("datetime64[s]").astype("int64")
                base = rt.idx.values.astype("datetime64[s]").astype("int64")
                pos = np.searchsorted(start, base, side="right") - 1
            else:
                rt.warn_once("request.security lookahead_on without a [1] offset would leak future bars; using completed higher-timeframe bars only")
        ent = rt.sec_cache[key] = (vals, pos)
    vals, pos = ent
    j = int(pos[rt.i])
    if j < 0:
        return NAN
    return vals[j]


def _lagged_only(node) -> bool:
    """True when every price series in the expression is read with a [1] (or deeper) offset, which is the documented
    non-repainting idiom for request.security(..., lookahead=barmerge.lookahead_on)."""
    series = set(_FAST_SERIES) | {"hl2", "hlc3", "ohlc4", "hlcc4"}

    def walk(n, lagged):
        if isinstance(n, list):
            return [x for c in n for x in walk(c, lagged)]
        if isinstance(n, dict):
            return [x for c in n.values() for x in walk(c, lagged)]
        if not isinstance(n, tuple) or not n:
            return []
        if n[0] == "idx":
            ok = n[2][0] == "num" and isinstance(n[2][1], (int, float)) and n[2][1] >= 1
            return walk(n[1], lagged or ok)
        if n[0] == "name":
            return [lagged] if n[1] in series else []
        if n[0] == "call":
            if n[1][0] == "name" and n[1][1] in FUNCS and FUNCS[n[1][1]][0] != "s":
                return walk(n[2], lagged) + walk(n[3], lagged)
            return [lagged] if lagged else [False]
        return [x for c in n[1:] for x in walk(c, lagged)]
    flags = walk(node, False)
    return bool(flags) and all(flags)


FUNCS: dict[str, tuple] = {}


def _reg(kind, fn, *names):
    for n in names:
        FUNCS[n] = (kind, fn)


for _n, _f in {
    "sma": T.ta_sma, "ema": T.ta_ema, "rma": T.ta_rma, "wma": T.ta_wma, "vwma": T.ta_vwma, "swma": T.ta_swma,
    "hma": T.ta_hma, "alma": T.ta_alma, "linreg": T.ta_linreg, "tr": T.ta_tr, "atr": T.ta_atr, "rsi": T.ta_rsi,
    "stdev": T.ta_stdev, "variance": T.ta_variance, "dev": T.ta_dev, "median": T.ta_median,
    "percentrank": T.ta_percentrank, "correlation": T.ta_correlation, "highest": T.ta_highest, "lowest": T.ta_lowest,
    "highestbars": T.ta_highestbars, "lowestbars": T.ta_lowestbars, "range": T.ta_range, "change": T.ta_change,
    "mom": T.ta_mom, "roc": T.ta_roc, "cum": T.ta_cum, "sum": T.ta_sum, "crossover": T.ta_cross_over,
    "crossunder": T.ta_cross_under, "cross": T.ta_cross, "rising": T.ta_rising, "falling": T.ta_falling,
    "barssince": T.ta_barssince, "valuewhen": T.ta_valuewhen, "macd": T.ta_macd, "bb": T.ta_bb, "bbw": T.ta_bbw,
    "kc": T.ta_kc, "kcw": T.ta_kcw, "stoch": T.ta_stoch, "wpr": T.ta_wpr, "cci": T.ta_cci, "cmo": T.ta_cmo,
    "mfi": T.ta_mfi, "dmi": T.ta_dmi, "adx": T.ta_adx, "supertrend": T.ta_supertrend, "sar": T.ta_sar,
    "pivothigh": T.ta_pivothigh, "pivotlow": T.ta_pivotlow, "vwap": T.ta_vwap, "tsi": T.ta_tsi,
}.items():
    _reg("s", _f, "ta." + _n, _n)
FUNCS.pop("tr", None)
FUNCS.pop("range", None)
FUNCS.pop("sum", None)
FUNCS.pop("cum", None)
FUNCS.pop("change", None)
_reg("s", T.ta_change, "change")
_reg("s", T.ta_cum, "cum")
_reg("s", T.ta_sum, "sum")
_reg("s", T.ta_tr, "tr")
_reg("s", T.ta_cross_over, "cross_over")
_reg("s", T.fixnan, "fixnan")
for _n, _f in {"abs": abs, "max": T.m_max, "min": T.m_min, "sqrt": T._m1(math.sqrt), "pow": T._m1(math.pow),
               "log": T._m1(math.log), "log10": T._m1(math.log10), "exp": T._m1(math.exp), "ceil": T._m1(lambda x: math.ceil(x)),
               "floor": T._m1(lambda x: math.floor(x)), "round": T.m_round, "sign": T.m_sign, "avg": T.m_avg,
               "sum": T.m_sum, "sin": T._m1(math.sin), "cos": T._m1(math.cos), "tan": T._m1(math.tan),
               "asin": T._m1(math.asin), "acos": T._m1(math.acos), "atan": T._m1(math.atan),
               "todegrees": T._m1(math.degrees), "toradians": T._m1(math.radians),
               "random": lambda mn=0.0, mx=1.0, seed=None: (mn + (mx - mn) * 0.5)}.items():
    _reg("p", _f, "math." + _n)
for _n in ("abs", "max", "min", "sqrt", "pow", "log", "log10", "exp", "ceil", "floor", "sign", "sin", "cos", "tan",
           "asin", "acos", "atan"):
    _reg("p", FUNCS["math." + _n][1], _n)
_reg("p", T.m_round, "round")
_reg("p", T.m_avg, "avg")
_reg("p", lambda x, precision=None: x, "math.round_to_mintick")
_reg("p", T.nz, "nz")
_reg("p", T.is_na, "na")
_reg("p", T.iff, "iff")
_reg("p", T.to_int, "int")
_reg("p", T.to_float, "float")
_reg("p", T.to_bool, "bool")
_reg("p", lambda x=None: ("" if x is None else str(x)), "str.tostring_plain")
_reg("p", T.tostring, "str.tostring", "tostring")
_reg("p", lambda s: len(str(s)), "str.length")
_reg("p", lambda s, sub: str(sub) in str(s), "str.contains")
_reg("p", lambda s, sub: str(s).startswith(str(sub)), "str.startswith")
_reg("p", lambda s, sub: str(s).endswith(str(sub)), "str.endswith")
_reg("p", lambda s: str(s).upper(), "str.upper")
_reg("p", lambda s: str(s).lower(), "str.lower")
_reg("p", lambda s, *a: str(s), "str.format")
_reg("p", lambda s: T._f(s), "str.tonumber")
_reg("p", lambda *a, **k: 0, "color.new", "color.rgb", "color.from_gradient")
_reg("p", lambda *a, **k: 0, "color.r", "color.g", "color.b", "color.t")
for _n in ("array.new_float", "array.new_int", "array.new_bool", "array.new_string", "array.new_any", "array.new_color",
           "array.new_label", "array.new_line", "array.new_box"):
    _reg("p", T.arr_new, _n)
_reg("p", T.arr_from, "array.from")
for _n, _f in {"push": T.arr_push, "get": T.arr_get, "set": T.arr_set, "size": T.arr_size, "pop": T.arr_pop,
               "shift": T.arr_shift, "unshift": T.arr_unshift, "clear": T.arr_clear, "insert": T.arr_insert,
               "remove": T.arr_remove, "sum": T.arr_sum, "avg": T.arr_avg, "max": T.arr_max, "min": T.arr_min,
               "includes": T.arr_includes, "indexof": T.arr_indexof, "first": T.arr_first, "last": T.arr_last,
               "copy": T.arr_copy, "sort": T.arr_sort, "stdev": T.arr_stdev, "median": T.arr_median}.items():
    _reg("p", _f, "array." + _n)
_reg("r", lambda rt, tf=None: _tf_seconds(tf), "timeframe.in_seconds")
_reg("s", _timeframe_change, "timeframe.change")
_reg("s", _time_fn, "time")
_reg("p", _timestamp, "timestamp")
_reg("r", _strategy_entry, "strategy.entry", "strategy.order")
_reg("r", _strategy_exit, "strategy.exit")
_reg("r", _strategy_close, "strategy.close")
_reg("r", _strategy_close_all, "strategy.close_all")
for _n in ("strategy.cancel", "strategy.cancel_all", "strategy.risk.allow_entry_in", "strategy.risk.max_drawdown",
           "strategy.risk.max_intraday_loss", "strategy.risk.max_cons_loss_days", "strategy.risk.max_intraday_filled_orders",
           "strategy.risk.max_position_size", "strategy", "indicator", "study", "library", "plot", "plotshape", "plotchar",
           "plotcandle", "plotbar", "plotarrow", "hline", "fill", "bgcolor", "barcolor", "alert", "alertcondition",
           "max_bars_back", "label.new", "label.set_text", "label.set_xy", "label.set_color", "label.set_style",
           "label.delete", "label.set_x", "label.set_y", "label.set_text_color", "line.new", "line.set_xy1", "line.set_xy2",
           "line.delete", "line.set_color", "line.set_x2", "line.set_y2", "box.new", "box.delete", "box.set_right",
           "box.set_bgcolor", "box.set_top", "box.set_bottom", "box.set_left", "table.new", "table.cell",
           "table.set_bgcolor", "table.delete", "table.clear", "log.info", "log.warning", "log.error", "runtime.error",
           "linefill.new", "chart.point.new"):
    _reg("p", _noop, _n)
for _n in ("input", "input.int", "input.float", "input.bool", "input.string", "input.source", "input.timeframe",
           "input.color", "input.symbol", "input.session", "input.price", "input.time", "input.text_area",
           "input.enum"):
    _reg("i", _inp, _n)
for _n in ("request.dividends", "request.earnings", "request.splits", "request.financial", "request.economic",
           "request.quandl", "request.security_lower_tf", "request.seed", "ticker.new", "ticker.modify"):
    _reg("x", None, _n)
FUNCS["ticker.heikinashi"] = ("p", lambda *a: "HA")
FUNCS["ticker.standard"] = ("p", lambda *a: "STD")
FUNCS["syminfo.tickerid_fn"] = ("p", _noop)


# ---------------------------------------------------------------------------------------------------------------
# compile AST -> closures
# ---------------------------------------------------------------------------------------------------------------
_FAST_SERIES = {"open": "O", "high": "H", "low": "L", "close": "C", "volume": "V"}


def _compile_expr(node, ctx):
    k = node[0]
    if k == "num" or k == "str":
        v = node[1]
        return lambda rt: v
    if k == "name":
        n = node[1]
        fs = _FAST_SERIES.get(n)
        if fs:
            return lambda rt, fs=fs: getattr(rt, fs)[rt.i]
        return lambda rt, n=n: rt.lookup(n)
    if k == "bin":
        op, a, b = node[1], _compile_expr(node[2], ctx), _compile_expr(node[3], ctx)
        if op in ("+", "-", "*", "/", "%"):
            return lambda rt, op=op, a=a, b=b: _arith(op, a(rt), b(rt))
        return lambda rt, op=op, a=a, b=b: _cmp(op, a(rt), b(rt))
    if k == "and":
        a, b = _compile_expr(node[1], ctx), _compile_expr(node[2], ctx)
        return lambda rt: _truthy(a(rt)) and _truthy(b(rt))
    if k == "or":
        a, b = _compile_expr(node[1], ctx), _compile_expr(node[2], ctx)
        return lambda rt: _truthy(a(rt)) or _truthy(b(rt))
    if k == "not":
        a = _compile_expr(node[1], ctx)
        return lambda rt: not _truthy(a(rt))
    if k == "un":
        a = _compile_expr(node[2], ctx)
        if node[1] == "-":
            return lambda rt: -_num(a(rt))
        return a
    if k == "tern":
        c, a, b = (_compile_expr(x, ctx) for x in node[1:4])
        return lambda rt: a(rt) if _truthy(c(rt)) else b(rt)
    if k == "tuple":
        items = [_compile_expr(x, ctx) for x in node[1]]
        return lambda rt: tuple(f(rt) for f in items)
    if k == "idx":
        return _compile_idx(node, ctx)
    if k == "call":
        return _compile_call(node, ctx)
    if k == "attr":
        raise StrategyError("PineScript: member access on a value is only supported for arrays/strings (use array.get(), ...).")
    raise StrategyError(f"PineScript: unsupported expression node {k}.")


def _compile_idx(node, ctx):
    base, kexp, sid = node[1], _compile_expr(node[2], ctx), node[3]
    if base[0] == "name" and base[1] not in ctx["parser"].funcs and "." not in base[1] or \
            (base[0] == "name" and base[1] in _BUILTIN_ARRAYS):
        name = base[1]

        def f(rt, name=name):
            k = kexp(rt)
            if isinstance(k, float):
                if k != k:
                    return NAN
                k = int(k)
            if k == 0:
                return rt.lookup(name)
            return rt.hist_at(name, k)
        return f
    inner = _compile_expr(base, ctx)

    def g(rt, inner=inner, sid=sid):
        v = inner(rt)
        st = rt.frame.state.setdefault(("idx", sid), {"b": []})
        st["b"].append(v)
        if len(st["b"]) > 4096:
            del st["b"][:2048]
        k = kexp(rt)
        if isinstance(k, float):
            if k != k:
                return NAN
        k = int(k)
        b = st["b"]
        if k == 0:
            return v
        if isinstance(v, list):          # array element access  arr[i] is not Pine; treat as history
            pass
        return b[-1 - k] if len(b) > k else NAN
    return g


def _compile_call(node, ctx):
    fnode, args, kwargs, sid = node[1], node[2], node[3], node[4]
    parser: _BlockParser = ctx["parser"]
    cargs = [_compile_expr(a, ctx) for a in args]
    ckw = {k: _compile_expr(v, ctx) for k, v in kwargs.items()}
    if fnode[0] != "name":
        raise StrategyError("PineScript: unsupported call target.")
    name = fnode[1]

    if name in parser.funcs:
        return _compile_user_call(name, cargs, ckw, sid, ctx)

    ent = FUNCS.get(name) or FUNCS.get("ta." + name) or FUNCS.get("math." + name)
    if ent is None and name == "request.security":
        ent = ("sec", None)
    if ent is None and name in ("security", "request.security"):
        ent = ("sec", None)
    if ent is None and "." in name:
        head, _, meth = name.partition(".")
        if head not in ("ta", "math", "array", "str", "strategy", "request", "color", "input", "timeframe", "barmerge",
                        "syminfo", "session", "label", "line", "box", "table", "log", "runtime", "chart", "ticker"):
            # method-call syntax on a variable:  arr.push(x)  ->  array.push(arr, x)
            tgt = FUNCS.get("array." + meth) or FUNCS.get("str." + meth)
            if tgt is not None:
                recv = _compile_expr(("name", head), ctx)
                kind, fn = tgt
                return lambda rt: fn(recv(rt), *[a(rt) for a in cargs], **{k: v(rt) for k, v in ckw.items()})
    if ent is None:
        raise StrategyError(f"PineScript: function '{name}' is not supported by the interpreter.")
    kind, fn = ent
    if kind == "x":
        raise StrategyError(f"PineScript: '{name}' needs external data and is not supported.")
    if kind == "sec":
        node_expr = args[2] if len(args) > 2 else kwargs.get("expression")
        site = sid

        def sec(rt):
            def lag_check():
                return False
            st = rt.frame.state.setdefault(("sec", site), {})
            names = ["symbol", "timeframe", "expression", "gaps", "lookahead", "ignore_invalid_symbol", "currency"]
            a = dict(zip(names, cargs))
            a.update(ckw)
            a["expression_node"] = node_expr
            return _security_call(rt, st, a, site)
        return sec
    if kind == "i":
        title_ix = 1

        def inp(rt):
            vals = [a(rt) for a in cargs]
            kw = {k: v(rt) for k, v in ckw.items()}
            return _inp_value(rt, name, vals, kw)
        return inp
    if kind == "p":
        return lambda rt: fn(*[a(rt) for a in cargs], **{k: v(rt) for k, v in ckw.items()})
    if kind == "r":
        return lambda rt: fn(rt, *[a(rt) for a in cargs], **{k: v(rt) for k, v in ckw.items()})

    def stateful(rt):
        st = rt.frame.state.setdefault(sid, {})
        return fn(rt, st, *[a(rt) for a in cargs], **{k: v(rt) for k, v in ckw.items()})
    return stateful


def _inp_value(rt, name, vals, kw):
    default = vals[0] if vals else kw.get("defval")
    title = kw.get("title") or (vals[1] if len(vals) > 1 and isinstance(vals[1], str) else None)
    tgt = rt.assign_target
    for key in (tgt, title):
        if key is not None and key in rt.overrides:
            v = rt.overrides[key]
            return type(default)(v) if isinstance(default, (int, float)) and not isinstance(default, bool) and v is not None else v
    return default


def _security_call(rt, st, a, site):
    # translate the arg-closure dict into _security's calling convention
    args = [a.get(n) for n in ("symbol", "timeframe", "expression")]
    kw = {k: v for k, v in a.items() if k in ("gaps", "lookahead", "expression_node")}
    kw2 = {k: v for k, v in kw.items()}
    return _security(rt, st, [x for x in args], kw2, site)


def _compile_user_call(name, cargs, ckw, sid, ctx):
    parser: _BlockParser = ctx["parser"]
    plist, body_nodes = parser.funcs[name]
    body = _compile_block(body_nodes, ctx)
    pdef = [(p, _compile_expr(d, ctx) if d is not None else None) for p, d in plist]

    def call(rt):
        argv = [a(rt) for a in cargs]
        kwv = {k: v(rt) for k, v in ckw.items()}
        fr = rt.call_frames.get(sid)
        if fr is None:
            fr = rt.call_frames[sid] = _Frame()
        saved = rt.frame
        vals = {}
        for ix, (pn, dflt) in enumerate(pdef):
            if ix < len(argv):
                vals[pn] = argv[ix]
            elif pn in kwv:
                vals[pn] = kwv[pn]
            elif dflt is not None:
                vals[pn] = dflt(rt)
            else:
                vals[pn] = NAN
        fr.vars.update(vals)
        rt.frame = fr
        try:
            res = body(rt)
        finally:
            rt.frame = saved
        for nm, v in fr.vars.items():
            fr.hist.setdefault(nm, []).append(v)
        return res
    return call


def _compile_stmt(node, ctx):
    k = node[0]
    if k == "expr":
        e = _compile_expr(node[1], ctx)
        return e
    if k == "assign":
        return _compile_assign(node, ctx)
    if k == "if":
        c = _compile_expr(node[1], ctx)
        body = _compile_block(node[2], ctx)
        orelse = _compile_block(node[3], ctx) if node[3] else None

        def f(rt):
            if _truthy(c(rt)):
                return body(rt)
            if orelse is not None:
                return orelse(rt)
            return None
        return f
    if k == "for":
        var, a, b = node[1], _compile_expr(node[2], ctx), _compile_expr(node[3], ctx)
        step = _compile_expr(node[4], ctx) if node[4] is not None else None
        body = _compile_block(node[5], ctx)

        def f(rt):
            lo, hi = int(_num(a(rt))), int(_num(b(rt)))
            st = int(_num(step(rt))) if step else (1 if hi >= lo else -1)
            if st == 0:
                st = 1
            last = None
            ix = lo
            guard = 0
            while (ix <= hi) if st > 0 else (ix >= hi):
                rt.frame.vars[var] = ix
                try:
                    last = body(rt)
                except _Break:
                    break
                except _Continue:
                    pass
                ix += st
                guard += 1
                if guard > 1_000_000:
                    raise StrategyError("PineScript: a for-loop ran more than 1,000,000 iterations.")
            return last
        return f
    if k == "foreach":
        var, arr = node[1], _compile_expr(node[2], ctx)
        body = _compile_block(node[3], ctx)

        def f(rt):
            last = None
            for v in list(arr(rt)):
                rt.frame.vars[var] = v
                try:
                    last = body(rt)
                except _Break:
                    break
                except _Continue:
                    pass
            return last
        return f
    if k == "while":
        c, body = _compile_expr(node[1], ctx), _compile_block(node[2], ctx)

        def f(rt):
            last, guard = None, 0
            while _truthy(c(rt)):
                try:
                    last = body(rt)
                except _Break:
                    break
                except _Continue:
                    pass
                guard += 1
                if guard > 1_000_000:
                    raise StrategyError("PineScript: a while-loop ran more than 1,000,000 iterations.")
            return last
        return f
    if k == "switch":
        subj = _compile_expr(node[1], ctx) if node[1] is not None else None
        cases = [(_compile_expr(c, ctx), _compile_block(b, ctx)) for c, b in node[2]]
        default = _compile_block(node[3], ctx) if node[3] else None

        def f(rt):
            sv = subj(rt) if subj else None
            for c, b in cases:
                cv = c(rt)
                if (subj is not None and _cmp("==", sv, cv)) or (subj is None and _truthy(cv)):
                    return b(rt)
            return default(rt) if default else None
        return f
    if k == "break":
        def f(rt):
            raise _Break()
        return f
    if k == "continue":
        def f(rt):
            raise _Continue()
        return f
    if k == "blockexpr":
        return _compile_stmt(node[1], ctx)
    raise StrategyError(f"PineScript: unsupported statement '{k}'.")


def _compile_block(nodes, ctx):
    fs = [_compile_stmt(n, ctx) for n in nodes]
    if not fs:
        return lambda rt: None
    if len(fs) == 1:
        return fs[0]

    def f(rt):
        v = None
        for x in fs:
            v = x(rt)
        return v
    return f


def _find_frame(rt, name):
    if name in rt.frame.vars or rt.frame is rt.g:
        return rt.frame
    return rt.g if name in rt.g.vars else rt.frame


def _compile_assign(node, ctx):
    _, tgt, op, rhs_node, persist, sid = node
    rhs_is_input = rhs_node[0] == "call" and rhs_node[1][0] == "name" and rhs_node[1][1].startswith("input")
    rhs = _compile_expr(rhs_node[1], ctx) if rhs_node[0] == "blockexpr" and False else None
    rhs = _compile_stmt(rhs_node, ctx) if rhs_node[0] == "blockexpr" else _compile_expr(rhs_node, ctx)
    if tgt[0] == "tuple":
        names = tgt[1]

        def ft(rt):
            val = rhs(rt)
            if not isinstance(val, (tuple, list)):
                val = (val,) * len(names)
            for nm, v in zip(names, val):
                if nm != "_":
                    rt.frame.vars[nm] = v
            return val
        return ft
    name = tgt[1]
    bin_op = op[0] if op in ("+=", "-=", "*=", "/=", "%=") else None

    def f(rt):
        fr = rt.frame
        if op == "=":
            if persist:
                if (sid, id(fr)) in fr.inited:
                    return fr.vars.get(name)
                fr.inited.add((sid, id(fr)))
            if rhs_is_input:
                rt.assign_target = name
            val = rhs(rt)
            rt.assign_target = None
            fr.vars[name] = val
            return val
        fr = _find_frame(rt, name)
        val = rhs(rt)
        if bin_op:
            val = _arith(bin_op, fr.vars.get(name, NAN), val)
        fr.vars[name] = val
        return val
    return f


PineRuntime.assign_target = None   # type: ignore[attr-defined]
PineRuntime.warn_once = lambda self, msg: (self.warnings.append(msg) if msg not in self.warnings else None)  # type: ignore[attr-defined]


def _end_bar(self):
    i = self.i
    hist = self.g.hist
    for nm, v in self.g.vars.items():
        h = hist.get(nm)
        if h is None:
            h = hist[nm] = []
        pad = i - len(h)
        if pad > 0:
            h.extend([NAN] * pad)
        if len(h) == i:
            h.append(v)
        else:
            h[i] = v


PineRuntime.end_bar = _end_bar  # type: ignore[attr-defined]


def _hist_at(self, name, k):
    j = self.i - k
    if j < 0:
        return NAN
    fr = self.frame
    if fr is not self.g and name not in fr.vars:
        fr = self.g
    h = fr.hist.get(name)
    if h is not None:
        if fr is self.g:
            return h[j] if j < len(h) else NAN
        idx = len(h) - k
        return h[idx] if idx >= 0 else NAN
    arr = _BUILTIN_ARRAYS.get(name)
    if arr is not None:
        return float(arr(self)[j])
    if name in fr.vars or name in self.g.vars:
        return NAN
    raise StrategyError(f"PineScript: cannot take history of '{name}'.")


PineRuntime.hist_at = _hist_at  # type: ignore[attr-defined]

_SL_ATR = re.compile(r"T58_SL_ATR_MULT\s*=\s*([\d.]+)")
_TP_ATR = re.compile(r"T58_TP_ATR_MULT\s*=\s*([\d.]+)")
_ATR_P = re.compile(r"T58_ATR_PERIOD\s*=\s*(\d+)")
_SL_PIPS = re.compile(r"T58_SL_PIPS\s*=\s*([\d.]+)")
_TP_PIPS = re.compile(r"T58_TP_PIPS\s*=\s*([\d.]+)")
# Any call that can open a position. A script with none of these can never trade, so running it would silently return
# all-zero signals instead of telling the user the script has no entry logic.
_ENTRY_CALL = re.compile(r"\bstrategy\s*\.\s*(?:entry|order)\s*\(")
_LINE_COMMENT = re.compile(r"//[^\n]*")


class PineInterpreter:
    """Parse once, run on any dataframe. `overrides` maps an input's variable name or title to a new value."""

    def __init__(self, code: str, overrides: dict | None = None):
        self.code = code
        self.overrides = overrides or {}
        lines = _logical_lines(code)
        if not lines:
            raise StrategyError("PineScript source is empty.")
        self.parser = _BlockParser(lines)
        self.ast = self.parser.parse_all()
        self.ctx = {"parser": self.parser}
        self.prog = _compile_block(self.ast, self.ctx)
        self.header = self._header()
        self.warnings: list[str] = []
        self._has_entry_call = bool(_ENTRY_CALL.search(_LINE_COMMENT.sub("", code)))

    def _header(self) -> dict:
        for n in self.ast:
            if n[0] == "expr" and n[1][0] == "call" and n[1][1] == ("name", "strategy"):
                try:
                    probe = PineRuntime(pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                                                      "volume": [0.0]}, index=pd.date_range("2020-01-01", periods=1)),
                                        self.parser, [])
                    out = {}
                    for k, v in n[1][3].items():
                        try:
                            out[k] = _compile_expr(v, self.ctx)(probe)
                        except Exception:  # noqa: BLE001
                            pass
                    return out
                except Exception:  # noqa: BLE001
                    return {}
        return {}

    def run(self, df: pd.DataFrame, max_bars: int | None = None) -> StrategyResult:
        if len(df) < 5:
            raise StrategyError("PineScript: not enough bars to run the interpreter.")
        rt = PineRuntime(df, self.parser, self.ast, overrides=self.overrides)
        rt.initial_capital = float(self.header.get("initial_capital", 100000.0) or 100000.0)
        qty = self.header.get("default_qty_value", 1.0)
        rt.broker = SimBroker(df, default_qty=float(qty) if isinstance(qty, (int, float)) and qty == qty and qty > 0 else 1.0,
                              pyramiding=int(self.header.get("pyramiding", 0) or 0))
        n = len(df)
        for i in range(n):
            rt.i = i
            rt.frame = rt.g
            rt.broker.begin_bar(i, rt.mintick)
            try:
                self.prog(rt)
            except (_Break, _Continue):
                pass
            rt.end_bar()
            rt.broker.end_bar(i, rt.mintick)
        # Checked after the run so a more specific runtime error (unsupported function etc.) always wins.
        if not self._has_entry_call:
            raise StrategyError(
                "PineScript: no strategy.entry(...) or strategy.order(...) call was found, so this script can never "
                "open a trade. Add an entry call (inline `when=` or inside an `if` block)."
            )
        self.warnings = list(rt.warnings)
        sig = pd.Series(rt.broker.signal, index=df.index)
        sl, tp = rt.broker.distances(df.index)
        return self._finish(df, sig, sl, tp)

    def _finish(self, df, sig, sl, tp) -> StrategyResult:
        code = self.code
        from app.strategy.indicators import atr as atr_fn
        sl_pips = float(_SL_PIPS.search(code).group(1)) if _SL_PIPS.search(code) else None
        tp_pips = float(_TP_PIPS.search(code).group(1)) if _TP_PIPS.search(code) else None
        m1, m2 = _SL_ATR.search(code), _TP_ATR.search(code)
        if m1 or m2:
            period = int(_ATR_P.search(code).group(1)) if _ATR_P.search(code) else 14
            a = atr_fn(df, period)
            if m1:
                sl, sl_pips = a * float(m1.group(1)), None
            if m2:
                tp, tp_pips = a * float(m2.group(1)), None
        return StrategyResult(name="PineScript Strategy (interpreted)", source_type="pinescript", signals=sig.astype(int),
                              stop_loss_pips=sl_pips if sl is None else None,
                              take_profit_pips=tp_pips if tp is None else None,
                              stop_loss_distance=sl, take_profit_distance=tp)
