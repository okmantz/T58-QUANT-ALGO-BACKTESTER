"""
MQL5 language front end for app.strategy.mql5_interp: preprocessor, lexer, parser and a compiler that turns the AST into
Python closures.

Supported: globals, `input`/`sinput` parameters, enums, simple structs, functions (value + reference params, defaults),
if/else, for, while, do-while, switch/case, break/continue/return, ternary, casts, compound assignment, ++/--, arrays
(fixed + dynamic, multi-dimensional via nested lists), `#define NAME value` constants, built-in classes (CTrade, CPositionInfo,
CSymbolInfo, CAccountInfo) and structs (MqlRates, MqlTick, MqlTradeRequest, MqlTradeResult, MqlDateTime).

Not supported (raises StrategyError with the exact construct): user-defined classes, templates, pointers/`new`, function-like
macros, #import of DLLs, iCustom() and other external-data calls.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable

from app.strategy.base import StrategyError

NAN = float("nan")

# ---------------------------------------------------------------------------------------------------------------
# preprocessor
# ---------------------------------------------------------------------------------------------------------------


def strip_comments(code: str) -> str:
    out, i, n, q = [], 0, len(code), None
    while i < n:
        ch = code[i]
        if q:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 1
            elif ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            out.append(ch)
        elif code.startswith("//", i):
            while i < n and code[i] != "\n":
                i += 1
            continue
        elif code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
            continue
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def preprocess(code: str) -> tuple[str, dict[str, str]]:
    """Removes #include/#property/#import blocks and collects simple object-like #define constants."""
    defines: dict[str, str] = {}
    lines, in_import = [], False
    for raw in strip_comments(code).splitlines():
        s = raw.strip()
        if s.startswith("#"):
            m = re.match(r"#\s*define\s+([A-Za-z_]\w*)(\()?\s*(.*)$", s)
            if m:
                if m.group(2):
                    raise StrategyError(f"MQL5: function-like macro '{m.group(1)}(...)' is not supported by the interpreter.")
                defines[m.group(1)] = m.group(3).strip()
            elif re.match(r"#\s*import", s):
                in_import = not in_import
            lines.append("")
            continue
        lines.append("" if in_import else raw)
    return "\n".join(lines), defines


# ---------------------------------------------------------------------------------------------------------------
# lexer
# ---------------------------------------------------------------------------------------------------------------
_TOK = re.compile(r"""\s*(?:
    (?P<num>0[xX][0-9a-fA-F]+|\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)[fFuUlL]*
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<chr>'(?:[^'\\]|\\.)')
  | (?P<id>[A-Za-z_]\w*)
  | (?P<op><<=|>>=|\+\+|--|\+=|-=|\*=|/=|%=|&=|\|=|\^=|==|!=|<=|>=|&&|\|\||<<|>>|::|->|[-+*/%<>=!~&|^?:;,.(){}\[\]])
)""", re.X)

KEYWORD_TYPES = {"void", "int", "uint", "long", "ulong", "short", "ushort", "char", "uchar", "bool", "double", "float",
                 "string", "datetime", "color"}
BUILTIN_CLASS_TYPES = {"CTrade", "CPositionInfo", "CSymbolInfo", "CAccountInfo", "CHistoryOrderInfo", "CDealInfo",
                       "COrderInfo", "MqlRates", "MqlTick", "MqlTradeRequest", "MqlTradeResult", "MqlTradeCheckResult",
                       "MqlDateTime", "MqlParam", "CArrayDouble", "CArrayInt"}
ENUM_TYPES = {"ENUM_TIMEFRAMES", "ENUM_MA_METHOD", "ENUM_APPLIED_PRICE", "ENUM_ORDER_TYPE", "ENUM_POSITION_TYPE",
              "ENUM_TRADE_REQUEST_ACTIONS", "ENUM_APPLIED_VOLUME", "ENUM_STO_PRICE", "ENUM_LINE_STYLE",
              "ENUM_BASE_CORNER", "ENUM_DAY_OF_WEEK", "ENUM_ORDER_TYPE_FILLING", "ENUM_ORDER_TYPE_TIME",
              "ENUM_DEAL_TYPE", "ENUM_INIT_RETCODE", "ENUM_ALIGN_MODE", "ENUM_ANCHOR_POINT", "ENUM_OBJECT",
              "ENUM_CHART_MODE", "ENUM_ACCOUNT_MARGIN_MODE", "ENUM_SYMBOL_CALC_MODE", "ENUM_ENVELOPE_PRICE"}
QUALIFIERS = {"const", "static", "virtual", "public", "private", "protected", "extern", "input", "sinput", "volatile"}


class Tok:
    __slots__ = ("k", "v")

    def __init__(self, k, v):
        self.k, self.v = k, v

    def __repr__(self):  # pragma: no cover
        return f"{self.k}:{self.v}"


def tokenize(text: str, defines: dict[str, str]) -> list[Tok]:
    out, pos, n = [], 0, len(text)
    while pos < n:
        m = _TOK.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            raise StrategyError(f"MQL5: cannot read '{text[pos:pos + 24].strip()}' (unsupported syntax).")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
        if kind == "id" and val in defines:
            sub = tokenize(defines[val], {k: v for k, v in defines.items() if k != val})
            out.extend(sub)
            continue
        out.append(Tok(kind, val))
    return out


# ---------------------------------------------------------------------------------------------------------------
# AST + parser
# ---------------------------------------------------------------------------------------------------------------
class Parser:
    def __init__(self, toks: list[Tok]):
        self.t, self.p = toks, 0
        self.types = set(KEYWORD_TYPES) | set(BUILTIN_CLASS_TYPES) | set(ENUM_TYPES)
        self.enums: dict[str, int] = {}
        self.structs: dict[str, list] = {}
        self.funcs: dict[str, dict] = {}
        self.globals: list = []       # ("var", type, name, dims, init, is_input)
        self.inputs: dict[str, Any] = {}

    # -- token helpers
    def peek(self, k=0):
        i = self.p + k
        return self.t[i] if i < len(self.t) else None

    def is_op(self, v, k=0):
        t = self.peek(k)
        return t is not None and t.k == "op" and t.v == v

    def is_id(self, v, k=0):
        t = self.peek(k)
        return t is not None and t.k == "id" and t.v == v

    def eat(self):
        t = self.t[self.p]
        self.p += 1
        return t

    def expect(self, v):
        if not self.is_op(v):
            t = self.peek()
            raise StrategyError(f"MQL5: expected '{v}' but found '{t.v if t else 'end of file'}'.")
        self.p += 1

    def accept(self, v):
        if self.is_op(v):
            self.p += 1
            return True
        return False

    def ident(self):
        t = self.peek()
        if t is None or t.k != "id":
            raise StrategyError(f"MQL5: expected a name but found '{t.v if t else 'end of file'}'.")
        self.p += 1
        return t.v

    # -- top level
    def parse(self):
        while self.peek() is not None:
            if self.accept(";"):
                continue
            self.top_level()
        return self

    def is_type_start(self, k=0) -> bool:
        i = k
        while True:
            t = self.peek(i)
            if t is None or t.k != "id":
                return False
            if t.v in QUALIFIERS:
                i += 1
                continue
            return t.v in self.types

    def parse_type(self):
        quals = set()
        while self.peek() and self.peek().k == "id" and self.peek().v in QUALIFIERS:
            quals.add(self.eat().v)
        t = self.peek()
        if t is None or t.k != "id":
            raise StrategyError("MQL5: expected a type.")
        name = self.eat().v
        if self.is_op("<"):
            raise StrategyError("MQL5: templates are not supported by the interpreter.")
        if self.is_op("*"):
            raise StrategyError("MQL5: pointers are not supported by the interpreter.")
        return name, quals

    def top_level(self):
        t = self.peek()
        if t.k != "id":
            raise StrategyError(f"MQL5: unexpected '{t.v}' at top level.")
        if t.v == "enum":
            return self.parse_enum()
        if t.v == "struct":
            return self.parse_struct()
        if t.v in ("class", "interface", "template", "namespace", "typedef", "union"):
            raise StrategyError(f"MQL5: '{t.v}' (user-defined classes/templates) is not supported by the interpreter.")
        if t.v in ("input", "sinput") and self.is_id("group", 1) and self.peek(2) is not None and self.peek(2).k == "str":
            self.p += 3
            return
        typ, quals = self.parse_type()
        by_ref = self.accept("&")
        name = self.ident()
        if self.is_op("("):
            return self.parse_function(typ, name)
        self.p -= 1  # re-read the first name as a declarator
        self.var_decls(typ, quals, is_global=True)

    def parse_enum(self):
        self.eat()
        name = None
        if self.peek().k == "id":
            name = self.eat().v
            self.types.add(name)
        self.expect("{")
        val = 0
        while not self.is_op("}"):
            nm = self.ident()
            if self.accept("="):
                e = self.expr()
                val = int(_const_eval(e, self.enums))
            self.enums[nm] = val
            val += 1
            self.accept(",")
        self.expect("}")
        self.accept(";")

    def parse_struct(self):
        self.eat()
        name = self.ident()
        self.types.add(name)
        self.expect("{")
        fields = []
        while not self.is_op("}"):
            typ, quals = self.parse_type()
            while True:
                fname = self.ident()
                dims = []
                while self.accept("["):
                    dims.append(None if self.is_op("]") else self.expr())
                    self.expect("]")
                fields.append((typ, fname, dims))
                if not self.accept(","):
                    break
            self.expect(";")
        self.expect("}")
        self.accept(";")
        self.structs[name] = fields

    def parse_function(self, rtype, name):
        self.expect("(")
        params = []
        while not self.is_op(")"):
            if self.is_id("void") and self.is_op(")", 1):
                self.eat()
                break
            ptype, _q = self.parse_type()
            ref = self.accept("&")
            pname = self.ident() if self.peek().k == "id" else f"_p{len(params)}"
            is_arr = False
            while self.accept("["):
                is_arr = True
                if not self.is_op("]"):
                    self.expr()
                self.expect("]")
            default = self.expr() if self.accept("=") else None
            params.append((ptype, pname, ref, is_arr, default))
            if not self.accept(","):
                break
        self.expect(")")
        if self.accept(";"):          # forward declaration
            return
        body = self.block()
        self.funcs[name] = {"ret": rtype, "params": params, "body": body}

    # -- statements
    def block(self):
        self.expect("{")
        stmts = []
        while not self.is_op("}"):
            if self.peek() is None:
                raise StrategyError("MQL5: missing closing '}'.")
            stmts.append(self.statement())
        self.expect("}")
        return ("block", stmts)

    def statement(self):
        t = self.peek()
        if self.accept(";"):
            return ("block", [])
        if self.is_op("{"):
            return self.block()
        if t.k == "id":
            v = t.v
            if v == "if":
                self.eat()
                self.expect("(")
                c = self.expr()
                self.expect(")")
                a = self.statement()
                b = None
                if self.is_id("else"):
                    self.eat()
                    b = self.statement()
                return ("if", c, a, b)
            if v == "for":
                self.eat()
                self.expect("(")
                init = []
                if not self.is_op(";"):
                    if self.is_type_start():
                        init = [self.var_decls_stmt()]
                    else:
                        init = [("expr", self.expr())]
                        while self.accept(","):
                            init.append(("expr", self.expr()))
                        self.expect(";")
                else:
                    self.eat()
                cond = None if self.is_op(";") else self.expr()
                self.expect(";")
                step = []
                if not self.is_op(")"):
                    step.append(self.expr())
                    while self.accept(","):
                        step.append(self.expr())
                self.expect(")")
                return ("for", init, cond, step, self.statement())
            if v == "while":
                self.eat()
                self.expect("(")
                c = self.expr()
                self.expect(")")
                return ("while", c, self.statement())
            if v == "do":
                self.eat()
                body = self.statement()
                if not self.is_id("while"):
                    raise StrategyError("MQL5: expected 'while' after do-block.")
                self.eat()
                self.expect("(")
                c = self.expr()
                self.expect(")")
                self.accept(";")
                return ("dowhile", body, c)
            if v == "switch":
                return self.switch_stmt()
            if v == "return":
                self.eat()
                e = None if self.is_op(";") else self.expr()
                self.expect(";")
                return ("return", e)
            if v == "break":
                self.eat()
                self.expect(";")
                return ("break",)
            if v == "continue":
                self.eat()
                self.expect(";")
                return ("continue",)
            if v in ("class", "struct", "enum", "delete", "new"):
                raise StrategyError(f"MQL5: '{v}' inside a function is not supported by the interpreter.")
            if self.is_type_start() and not (self.peek(1) and self.peek(1).k == "op" and self.peek(1).v in ("(", ".", "::")):
                return self.var_decls_stmt()
        e = self.expr()
        self.expect(";")
        return ("expr", e)

    def var_decls_stmt(self):
        typ, quals = self.parse_type()
        node = self.var_decls(typ, quals, is_global=False)
        return node

    def var_decls(self, typ, quals, is_global):
        decls = []
        while True:
            self.accept("&")
            name = self.ident()
            dims = []
            while self.accept("["):
                dims.append(None if self.is_op("]") else self.expr())
                self.expect("]")
            init = None
            if self.accept("="):
                init = self.initializer()
            elif self.is_op("(") and typ in self.types:            # constructor-style init, e.g. CTrade trade(...)
                self.eat()
                args = []
                while not self.is_op(")"):
                    args.append(self.expr())
                    self.accept(",")
                self.expect(")")
                init = ("ctor", args)
            decls.append((typ, name, dims, init, "input" in quals or "sinput" in quals, "static" in quals))
            if not self.accept(","):
                break
        self.expect(";")
        if is_global:
            self.globals.extend(decls)
            return None
        return ("decl", decls)

    def initializer(self):
        if self.is_op("{"):
            self.eat()
            items = []
            while not self.is_op("}"):
                items.append(self.initializer())
                self.accept(",")
            self.expect("}")
            return ("arrinit", items)
        return self.expr()

    def switch_stmt(self):
        self.eat()
        self.expect("(")
        subj = self.expr()
        self.expect(")")
        self.expect("{")
        cases, default_ix = [], None
        cur = None
        while not self.is_op("}"):
            if self.is_id("case"):
                self.eat()
                e = self.expr()
                self.expect(":")
                cur = (e, [])
                cases.append(cur)
            elif self.is_id("default"):
                self.eat()
                self.expect(":")
                cur = (None, [])
                default_ix = len(cases)
                cases.append(cur)
            else:
                if cur is None:
                    raise StrategyError("MQL5: statement before first case in switch.")
                cur[1].append(self.statement())
        self.expect("}")
        return ("switch", subj, cases, default_ix)

    # -- expressions (precedence climbing)
    def expr(self):
        return self.assign()

    def assign(self):
        left = self.ternary()
        t = self.peek()
        if t is not None and t.k == "op" and t.v in ("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="):
            self.eat()
            right = self.assign()
            return ("assign", t.v, left, right)
        return left

    def ternary(self):
        c = self.binary(0)
        if self.accept("?"):
            a = self.assign()
            self.expect(":")
            b = self.assign()
            return ("tern", c, a, b)
        return c

    _LEVELS = [("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", ">", "<=", ">="), ("<<", ">>"),
               ("+", "-"), ("*", "/", "%")]

    def binary(self, lvl):
        if lvl >= len(self._LEVELS):
            return self.unary()
        left = self.binary(lvl + 1)
        ops = self._LEVELS[lvl]
        while self.peek() is not None and self.peek().k == "op" and self.peek().v in ops:
            op = self.eat().v
            right = self.binary(lvl + 1)
            left = ("logic" if op in ("&&", "||") else "bin", op, left, right)
        return left

    def unary(self):
        t = self.peek()
        if t is None:
            raise StrategyError("MQL5: unexpected end of expression.")
        if t.k == "op":
            if t.v in ("!", "-", "+", "~"):
                self.eat()
                return ("un", t.v, self.unary())
            if t.v in ("++", "--"):
                self.eat()
                return ("preinc", t.v, self.unary())
            if t.v == "(" and self.peek(1) and self.peek(1).k == "id" and self.peek(1).v in self.types and self.is_op(")", 2):
                typ = self.peek(1).v
                self.p += 3
                return ("cast", typ, self.unary())
            if t.v == "&":
                self.eat()
                return self.unary()
        if t.k == "id" and t.v in ("new", "delete", "sizeof"):
            raise StrategyError(f"MQL5: '{t.v}' is not supported by the interpreter.")
        return self.postfix()

    def postfix(self):
        e = self.primary()
        while True:
            if self.is_op("("):
                self.eat()
                args = []
                while not self.is_op(")"):
                    args.append(self.expr())
                    if not self.accept(","):
                        break
                self.expect(")")
                e = ("call", e, args)
            elif self.is_op("["):
                self.eat()
                idx = self.expr()
                self.expect("]")
                e = ("index", e, idx)
            elif self.is_op(".") or self.is_op("->"):
                self.eat()
                e = ("member", e, self.ident())
            elif self.is_op("::"):
                self.eat()
                nm = self.ident()
                e = ("name", nm)
            elif self.is_op("++") or self.is_op("--"):
                op = self.eat().v
                e = ("postinc", op, e)
            else:
                return e

    def primary(self):
        t = self.eat()
        if t.k == "num":
            v = t.v
            if v.lower().startswith("0x"):
                return ("num", int(v, 16))
            if re.fullmatch(r"\d+", v):
                return ("num", int(v))
            return ("num", float(v))
        if t.k == "str":
            return ("str", bytes(t.v[1:-1], "utf-8").decode("unicode_escape"))
        if t.k == "chr":
            s = bytes(t.v[1:-1], "utf-8").decode("unicode_escape")
            return ("num", ord(s[0]) if s else 0)
        if t.k == "id":
            return ("name", t.v)
        if t.k == "op" and t.v == "(":
            e = self.expr()
            self.expect(")")
            return e
        raise StrategyError(f"MQL5: unexpected '{t.v}' in expression.")


def _const_eval(node, enums):
    k = node[0]
    if k == "num":
        return node[1]
    if k == "name":
        if node[1] in enums:
            return enums[node[1]]
        raise StrategyError(f"MQL5: '{node[1]}' is not a compile-time constant here.")
    if k == "un" and node[1] == "-":
        return -_const_eval(node[2], enums)
    if k == "bin":
        a, b = _const_eval(node[2], enums), _const_eval(node[3], enums)
        return {"+": a + b, "-": a - b, "*": a * b, "<<": int(a) << int(b), "|": int(a) | int(b)}.get(node[1], 0)
    raise StrategyError("MQL5: enum value is not a compile-time constant.")


# ---------------------------------------------------------------------------------------------------------------
# runtime values
# ---------------------------------------------------------------------------------------------------------------
class BreakSig:
    pass


class ContSig:
    pass


BREAK, CONT = BreakSig(), ContSig()


class Ret:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


class Ref:
    """By-reference scalar parameter (`double &x`)."""
    __slots__ = ("get", "set")

    def __init__(self, get, set_):
        self.get, self.set = get, set_


class MqlArray(list):
    """List with MetaTrader's series flag and element type (for structs created on resize)."""
    series = False
    elem = None


class MqlStruct:
    def __init__(self, fields=None):
        object.__setattr__(self, "_f", dict(fields or {}))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        if name.startswith("_"):
            raise AttributeError(name)
        return 0

    def __setattr__(self, name, value):
        self._f[name] = value

    def clear(self):
        for k in list(self._f):
            v = self._f[k]
            self._f[k] = "" if isinstance(v, str) else 0


def to_int(x):
    if x.__class__.__name__ == "Handle":      # indicator handles survive being stored in an `int`
        return x
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if x != x or x in (math.inf, -math.inf):
            return 0
        return int(x)
    if isinstance(x, str):
        try:
            return int(float(x))
        except ValueError:
            return 0
    if x is None:
        return 0
    return x       # Handle objects etc.


def to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return 0.0
    if x is None:
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def to_bool(x):
    if isinstance(x, float):
        return x != 0.0 and x == x
    return bool(x)


def to_str(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        return f"{x:.8f}".rstrip("0").rstrip(".") if x == x else "nan"
    return str(x)


COERCE: dict[str, Callable] = {
    "int": to_int, "uint": to_int, "long": to_int, "ulong": to_int, "short": to_int, "ushort": to_int, "char": to_int,
    "uchar": to_int, "datetime": to_int, "color": to_int, "double": to_float, "float": to_float, "bool": to_bool,
    "string": to_str,
}


def _cdiv(a, b):
    if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) and not isinstance(b, bool):
        if b == 0:
            return 0
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b >= 0) else -q
    try:
        return a / b
    except ZeroDivisionError:
        return NAN if (a == 0 or a != a) else math.copysign(math.inf, a)
    except TypeError:
        return NAN


def _cmod(a, b):
    try:
        if isinstance(a, int) and isinstance(b, int):
            return int(math.fmod(a, b)) if b else 0
        return math.fmod(a, b) if b else NAN
    except (TypeError, ValueError):
        return NAN


def _binop(op):
    if op == "+":
        return lambda a, b: (a + b) if not (isinstance(a, str) or isinstance(b, str)) else (to_str(a) + to_str(b))
    ops = {
        "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": _cdiv, "%": _cmod,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b, ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
        "<<": lambda a, b: to_int(a) << to_int(b), ">>": lambda a, b: to_int(a) >> to_int(b),
        "&": lambda a, b: to_int(a) & to_int(b), "|": lambda a, b: to_int(a) | to_int(b), "^": lambda a, b: to_int(a) ^ to_int(b),
    }
    return ops[op]


# ---------------------------------------------------------------------------------------------------------------
# compiler: AST -> closures f(rt, env)
# ---------------------------------------------------------------------------------------------------------------
class Compiler:
    def __init__(self, parser: Parser):
        self.p = parser
        self.func_types: dict[str, str] = {}
        self.global_types: dict[str, str] = {}
        self.cur_types: dict[str, str] = {}
        self.cur_refs: set[str] = set()
        self.cur_func = ""
        self.compiled: dict[str, Callable] = {}

    # -- expressions
    def ex(self, n) -> Callable:
        k = n[0]
        if k == "num" or k == "str":
            v = n[1]
            return lambda rt, env, v=v: v
        if k == "name":
            return self.ex_name(n[1])
        if k == "bin":
            op, a, b = n[1], self.ex(n[2]), self.ex(n[3])
            fn = _binop(op)

            def f(rt, env, a=a, b=b, fn=fn):
                return fn(a(rt, env), b(rt, env))
            return f
        if k == "logic":
            a, b = self.ex(n[2]), self.ex(n[3])
            if n[1] == "&&":
                return lambda rt, env: to_bool(a(rt, env)) and to_bool(b(rt, env))
            return lambda rt, env: to_bool(a(rt, env)) or to_bool(b(rt, env))
        if k == "un":
            a = self.ex(n[2])
            op = n[1]
            if op == "!":
                return lambda rt, env: not to_bool(a(rt, env))
            if op == "-":
                return lambda rt, env: -a(rt, env)
            if op == "~":
                return lambda rt, env: ~to_int(a(rt, env))
            return a
        if k == "tern":
            c, a, b = self.ex(n[1]), self.ex(n[2]), self.ex(n[3])
            return lambda rt, env: a(rt, env) if to_bool(c(rt, env)) else b(rt, env)
        if k == "cast":
            fn = COERCE.get(n[1])
            a = self.ex(n[2])
            if fn is None:
                return a
            return lambda rt, env: fn(a(rt, env))
        if k == "assign":
            return self.ex_assign(n)
        if k in ("preinc", "postinc"):
            return self.ex_inc(n)
        if k == "index":
            a, i = self.ex(n[1]), self.ex(n[2])

            def f(rt, env):
                arr = a(rt, env)
                ix = to_int(i(rt, env))
                try:
                    return arr[ix] if ix >= 0 else NAN
                except (IndexError, TypeError):
                    return NAN if not isinstance(arr, str) else ""
            return f
        if k == "member":
            a = self.ex(n[1])
            name = n[2]
            return lambda rt, env: getattr(a(rt, env), name)
        if k == "call":
            return self.ex_call(n)
        raise StrategyError(f"MQL5: unsupported expression '{k}'.")

    def ex_name(self, name: str):
        if name in self.cur_refs:
            def fr(rt, env):
                v = env[name]
                return v.get() if v.__class__ is Ref else v
            return fr

        def f(rt, env):
            if name in env:
                return env[name]
            g = rt.globals
            if name in g:
                return g[name]
            return rt.constant(name)
        return f

    def lvalue(self, n):
        """(getter, setter, type) for an assignable expression."""
        k = n[0]
        if k == "name":
            name = n[1]
            typ = self.cur_types.get(name) or self.global_types.get(name)
            coerce = COERCE.get(typ) if typ else None
            if name in self.cur_refs:
                def get(rt, env):
                    v = env[name]
                    return v.get() if v.__class__ is Ref else v

                def set_(rt, env, val):
                    v = env[name]
                    if coerce:
                        val = coerce(val)
                    if v.__class__ is Ref:
                        v.set(val)
                    else:
                        env[name] = val
                return get, set_

            def get(rt, env):
                if name in env:
                    return env[name]
                if name in rt.globals:
                    return rt.globals[name]
                return rt.constant(name)

            def set_(rt, env, val):
                if coerce:
                    val = coerce(val)
                if name in env:
                    env[name] = val
                else:
                    rt.globals[name] = val
            return get, set_
        if k == "index":
            a, i = self.ex(n[1]), self.ex(n[2])
            base_t = n[1][1] if n[1][0] == "name" else None
            arr_t = (self.cur_types.get(base_t) or self.global_types.get(base_t)) if base_t else None
            coerce = COERCE.get(arr_t) if arr_t else None

            def get(rt, env):
                arr = a(rt, env)
                ix = to_int(i(rt, env))
                try:
                    return arr[ix]
                except (IndexError, TypeError):
                    return NAN

            def set_(rt, env, val):
                arr = a(rt, env)
                ix = to_int(i(rt, env))
                if coerce:
                    val = coerce(val)
                try:
                    arr[ix] = val
                except IndexError:
                    raise StrategyError("MQL5: array index out of range (the EA would stop with 'array out of range').")
            return get, set_
        if k == "member":
            a, name = self.ex(n[1]), n[2]

            def get(rt, env):
                return getattr(a(rt, env), name)

            def set_(rt, env, val):
                setattr(a(rt, env), name, val)
            return get, set_
        raise StrategyError("MQL5: invalid assignment target.")

    def ex_assign(self, n):
        op, target, rhs = n[1], n[2], self.ex(n[3])
        get, set_ = self.lvalue(target)
        if op == "=":
            def f(rt, env):
                v = rhs(rt, env)
                set_(rt, env, v)
                return v
            return f
        fn = _binop(op[:-1])

        def g(rt, env):
            v = fn(get(rt, env), rhs(rt, env))
            set_(rt, env, v)
            return v
        return g

    def ex_inc(self, n):
        kind, op, target = n[0], n[1], n[2]
        get, set_ = self.lvalue(target)
        d = 1 if op == "++" else -1
        if kind == "preinc":
            def f(rt, env):
                v = get(rt, env) + d
                set_(rt, env, v)
                return v
            return f

        def g(rt, env):
            old = get(rt, env)
            set_(rt, env, old + d)
            return old
        return g

    def ex_call(self, n):
        callee, args = n[1], n[2]
        cargs = [self.ex(a) for a in args]
        if callee[0] == "member":
            obj = self.ex(callee[1])
            meth = callee[2]
            lv = [self._maybe_lv(a) for a in args]

            def mc(rt, env):
                o = obj(rt, env)
                fn = getattr(o, meth, None)
                if fn is None:
                    raise StrategyError(f"MQL5: '{meth}' is not a supported method here.")
                return fn(*[a(rt, env) for a in cargs])
            return mc
        if callee[0] != "name":
            raise StrategyError("MQL5: unsupported call target.")
        name = callee[1]
        if name in self.p.funcs:
            spec = self.p.funcs[name]
            params = spec["params"]
            lvs = []
            for ix, a in enumerate(args):
                if ix < len(params) and params[ix][2] and not params[ix][3] and a[0] in ("name", "index", "member"):
                    lvs.append(self.lvalue(a))
                else:
                    lvs.append(None)

            def uc(rt, env):
                vals = []
                for ix, a in enumerate(cargs):
                    lv = lvs[ix]
                    if lv is not None:
                        vals.append(Ref(lambda e=env, lv=lv: lv[0](rt, e), lambda v, e=env, lv=lv: lv[1](rt, e, v)))
                    else:
                        vals.append(a(rt, env))
                return rt.call_user(name, vals)
            return uc
        if name in COERCE and len(args) == 1:
            fn = COERCE[name]
            return lambda rt, env: fn(cargs[0](rt, env))

        def bc(rt, env):
            fn = rt.builtins.get(name)
            if fn is None:
                raise StrategyError(f"MQL5: function '{name}' is not supported by the interpreter.")
            return fn(rt, *[a(rt, env) for a in cargs])
        return bc

    def _maybe_lv(self, a):
        return None

    # -- statements
    def st(self, n) -> Callable:
        k = n[0]
        if k == "block":
            fs = [self.st(s) for s in n[1]]
            if not fs:
                return lambda rt, env: None
            if len(fs) == 1:
                return fs[0]

            def blk(rt, env):
                for f in fs:
                    r = f(rt, env)
                    if r is not None:
                        return r
                return None
            return blk
        if k == "expr":
            e = self.ex(n[1])

            def es(rt, env):
                e(rt, env)
            return es
        if k == "decl":
            return self.st_decl(n[1])
        if k == "if":
            c, a = self.ex(n[1]), self.st(n[2])
            b = self.st(n[3]) if n[3] is not None else None

            def ifs(rt, env):
                if to_bool(c(rt, env)):
                    return a(rt, env)
                if b is not None:
                    return b(rt, env)
                return None
            return ifs
        if k == "for":
            init = [self.st(s) for s in n[1]]
            cond = self.ex(n[2]) if n[2] is not None else None
            step = [self.ex(s) for s in n[3]]
            body = self.st(n[4])

            def fors(rt, env):
                for s in init:
                    s(rt, env)
                guard = 0
                while cond is None or to_bool(cond(rt, env)):
                    r = body(rt, env)
                    if r is not None:
                        if r is BREAK:
                            break
                        if r is not CONT:
                            return r
                    for s in step:
                        s(rt, env)
                    guard += 1
                    if guard > 5_000_000:
                        raise StrategyError("MQL5: a for-loop ran more than 5,000,000 iterations.")
                return None
            return fors
        if k == "while":
            c, body = self.ex(n[1]), self.st(n[2])

            def whs(rt, env):
                guard = 0
                while to_bool(c(rt, env)):
                    r = body(rt, env)
                    if r is not None:
                        if r is BREAK:
                            break
                        if r is not CONT:
                            return r
                    guard += 1
                    if guard > 5_000_000:
                        raise StrategyError("MQL5: a while-loop ran more than 5,000,000 iterations.")
                return None
            return whs
        if k == "dowhile":
            body, c = self.st(n[1]), self.ex(n[2])

            def dws(rt, env):
                guard = 0
                while True:
                    r = body(rt, env)
                    if r is not None:
                        if r is BREAK:
                            break
                        if r is not CONT:
                            return r
                    guard += 1
                    if guard > 5_000_000 or not to_bool(c(rt, env)):
                        break
                return None
            return dws
        if k == "switch":
            subj = self.ex(n[1])
            cases = [(self.ex(c) if c is not None else None, [self.st(s) for s in body]) for c, body in n[2]]
            dix = n[3]

            def sws(rt, env):
                v = subj(rt, env)
                start = None
                for ix, (c, _b) in enumerate(cases):
                    if c is not None and c(rt, env) == v:
                        start = ix
                        break
                if start is None:
                    start = dix
                if start is None:
                    return None
                for _c, body in cases[start:]:
                    for s in body:
                        r = s(rt, env)
                        if r is not None:
                            if r is BREAK:
                                return None
                            return r
                return None
            return sws
        if k == "return":
            e = self.ex(n[1]) if n[1] is not None else None
            rtype = self.func_types.get(self.cur_func)
            coerce = COERCE.get(rtype) if rtype else None

            def rets(rt, env):
                v = e(rt, env) if e is not None else None
                if coerce is not None and v is not None:
                    v = coerce(v)
                return Ret(v)
            return rets
        if k == "break":
            return lambda rt, env: BREAK
        if k == "continue":
            return lambda rt, env: CONT
        raise StrategyError(f"MQL5: unsupported statement '{k}'.")

    def st_decl(self, decls):
        fs = []
        for typ, name, dims, init, is_input, is_static in decls:
            self.cur_types[name] = typ
            fs.append(self.decl_one(typ, name, dims, init, is_static))

        def run(rt, env):
            for f in fs:
                f(rt, env)
        return run

    def decl_one(self, typ, name, dims, init, is_static=False):
        make = self.make_value(typ, dims, init)
        func = self.cur_func

        def f(rt, env):
            if is_static:
                key = (func, name)
                if key in rt.statics:
                    env[name] = rt.statics[key]
                    return
                v = make(rt, env)
                rt.statics[key] = v
                env[name] = v
                return
            env[name] = make(rt, env)
        return f

    def make_value(self, typ, dims, init):
        coerce = COERCE.get(typ)
        dim_fns = [self.ex(d) if d is not None else None for d in dims]
        initf = self.init_expr(init, typ) if init is not None else None

        def mk(rt, env):
            if dims:
                sizes = [to_int(d(rt, env)) if d is not None else 0 for d in dim_fns]
                arr = rt.new_array(typ, sizes)
                if initf is not None:
                    iv = initf(rt, env)
                    if isinstance(iv, list):
                        for ix, v in enumerate(iv):
                            if ix < len(arr):
                                arr[ix] = coerce(v) if coerce else v
                            else:
                                arr.append(coerce(v) if coerce else v)
                return arr
            if initf is not None:
                v = initf(rt, env)
                if coerce is not None:
                    if typ in ("int", "uint", "long", "ulong", "short", "ushort") and v.__class__.__name__ == "Handle":
                        return v
                    return coerce(v)
                return v
            return rt.default_value(typ)
        return mk

    def init_expr(self, init, typ):
        if init[0] == "arrinit":
            items = [self.init_expr(i, typ) for i in init[1]]
            return lambda rt, env: [f(rt, env) for f in items]
        if init[0] == "ctor":
            args = [self.ex(a) for a in init[1]]
            return lambda rt, env: rt.construct(typ, [a(rt, env) for a in args])
        return self.ex(init)

    # -- functions
    def compile_function(self, name: str):
        spec = self.p.funcs[name]
        self.cur_func = name
        self.cur_types = {p[1]: p[0] for p in spec["params"]}
        self.cur_refs = {p[1] for p in spec["params"] if p[2] and not p[3]}
        body = self.st(spec["body"])
        defaults = [self.ex(p[4]) if p[4] is not None else None for p in spec["params"]]
        return body, defaults

    def compile_all(self):
        for name, spec in self.p.funcs.items():
            self.func_types[name] = spec["ret"]
        for typ, name, dims, init, is_input, is_static in self.p.globals:
            self.global_types[name] = typ
        out = {}
        for name in self.p.funcs:
            out[name] = self.compile_function(name)
        self.cur_func, self.cur_types, self.cur_refs = "", {}, set()
        gi = []
        for typ, name, dims, init, is_input, is_static in self.p.globals:
            gi.append((name, is_input, typ, self.make_value(typ, dims, init)))
        return out, gi
