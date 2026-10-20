"""
Pine `ta.*`, `math.*`, `array.*`, `str.*` and helper functions for the interpreter in pine_interp.py.

Every stateful function has the signature fn(rt, st, *args, **kwargs): `st` is a dict private to that call site, so two
`ta.ema(close, 20)` calls on one line never share state (this is exactly Pine's own model). Warm-up bars return NaN
(Pine's `na`), seeds follow Pine's definitions (ema/rma seed with the SMA of the first `length` values).
"""
from __future__ import annotations

import math
import statistics

import numpy as np

NAN = float("nan")


def _f(x) -> float:
    if x is None:
        return NAN
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _nan(x) -> bool:
    return x != x


def _buf(st, key, val, keep):
    b = st.get(key)
    if b is None:
        b = st[key] = []
    b.append(val)
    if len(b) > keep * 4 + 256:
        del b[: len(b) - keep - 64]
    return b


# ---------------------------------------------------------------------------------------------------------------
# moving averages (incremental)
# ---------------------------------------------------------------------------------------------------------------
def _sma(st, key, src, n):
    b = _buf(st, key, src, n)
    if len(b) < n:
        return NAN
    return sum(b[-n:]) / n


def _rma(st, key, src, n):
    """Wilder smoothing, seeded with the SMA of the first n valid values."""
    s = st.setdefault(key, {"seed": [], "v": None})
    if _nan(src):
        return NAN if s["v"] is None else s["v"]
    if s["v"] is None:
        s["seed"].append(src)
        if len(s["seed"]) < n:
            return NAN
        s["v"] = sum(s["seed"]) / n
        s["seed"] = []
        return s["v"]
    s["v"] = (s["v"] * (n - 1) + src) / n
    return s["v"]


def _ema(st, key, src, n):
    s = st.setdefault(key, {"seed": [], "v": None})
    if _nan(src):
        return NAN if s["v"] is None else s["v"]
    a = 2.0 / (n + 1)
    if s["v"] is None:
        s["seed"].append(src)
        if len(s["seed"]) < n:
            return NAN
        s["v"] = sum(s["seed"]) / n
        s["seed"] = []
        return s["v"]
    s["v"] = a * src + (1 - a) * s["v"]
    return s["v"]


def _wma(st, key, src, n):
    b = _buf(st, key, src, n)
    if len(b) < n:
        return NAN
    w = b[-n:]
    den = n * (n + 1) / 2.0
    return sum((k + 1) * v for k, v in enumerate(w)) / den


def ta_sma(rt, st, source, length=None):
    return _sma(st, "b", _f(source), max(int(length), 1))


def ta_ema(rt, st, source, length=None):
    return _ema(st, "e", _f(source), max(int(length), 1))


def ta_rma(rt, st, source, length=None):
    return _rma(st, "r", _f(source), max(int(length), 1))


def ta_wma(rt, st, source, length=None):
    return _wma(st, "w", _f(source), max(int(length), 1))


def ta_vwma(rt, st, source, length=None):
    n = max(int(length), 1)
    pv = _buf(st, "pv", _f(source) * rt.V[rt.i], n)
    vv = _buf(st, "v", rt.V[rt.i], n)
    if len(pv) < n:
        return NAN
    d = sum(vv[-n:])
    return sum(pv[-n:]) / d if d else NAN


def ta_swma(rt, st, source):
    b = _buf(st, "b", _f(source), 4)
    if len(b) < 4:
        return NAN
    w = b[-4:]
    return (w[0] * 1 + w[1] * 2 + w[2] * 2 + w[3] * 1) / 6.0


def ta_hma(rt, st, source, length=None):
    n = max(int(length), 1)
    half, sq = max(n // 2, 1), max(int(math.sqrt(n)), 1)
    a = _wma(st, "h1", _f(source), half)
    b = _wma(st, "h2", _f(source), n)
    return _wma(st, "h3", 2 * a - b if not (_nan(a) or _nan(b)) else NAN, sq)


def ta_alma(rt, st, source, length=9, offset=0.85, sigma=6.0, floor=False):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n)
    if len(b) < n:
        return NAN
    m = offset * (n - 1)
    s = n / sigma
    w = [math.exp(-((k - m) ** 2) / (2 * s * s)) for k in range(n)]
    return sum(x * y for x, y in zip(b[-n:], w)) / sum(w)


def ta_linreg(rt, st, source, length=None, offset=0):
    n = max(int(length), 2)
    b = _buf(st, "b", _f(source), n)
    if len(b) < n:
        return NAN
    y = np.asarray(b[-n:], dtype=float)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * (n - 1 - offset))


# ---------------------------------------------------------------------------------------------------------------
# oscillators / volatility
# ---------------------------------------------------------------------------------------------------------------
def _tr_now(rt, st):
    i = rt.i
    if i == 0:
        return rt.H[0] - rt.L[0]
    pc = rt.C[i - 1]
    return max(rt.H[i] - rt.L[i], abs(rt.H[i] - pc), abs(rt.L[i] - pc))


def ta_tr(rt, st, handle_na=False):
    return _tr_now(rt, st)


def ta_atr(rt, st, length=None):
    return _rma(st, "a", _tr_now(rt, st), max(int(length), 1))


def ta_rsi(rt, st, source, length=None):
    n = max(int(length), 1)
    src = _f(source)
    prev = st.get("p")
    st["p"] = src
    if prev is None or _nan(prev) or _nan(src):
        return NAN
    ch = src - prev
    up = _rma(st, "u", max(ch, 0.0), n)
    dn = _rma(st, "d", max(-ch, 0.0), n)
    if _nan(up) or _nan(dn):
        return NAN
    if dn == 0:
        return 100.0
    if up == 0:
        return 0.0
    return 100.0 - 100.0 / (1.0 + up / dn)


def ta_stdev(rt, st, source, length=None, biased=True):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n)
    if len(b) < n:
        return NAN
    w = np.asarray(b[-n:], dtype=float)
    return float(np.std(w, ddof=0 if biased else 1))


def ta_variance(rt, st, source, length=None, biased=True):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n)
    if len(b) < n:
        return NAN
    return float(np.var(np.asarray(b[-n:], dtype=float), ddof=0 if biased else 1))


def ta_dev(rt, st, source, length=None):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n)
    if len(b) < n:
        return NAN
    w = b[-n:]
    m = sum(w) / n
    return sum(abs(x - m) for x in w) / n


def ta_median(rt, st, source, length=None):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n)
    if len(b) < n:
        return NAN
    return float(np.median(b[-n:]))


def ta_percentrank(rt, st, source, length=None):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n + 1)
    if len(b) < n + 1:
        return NAN
    cur = b[-1]
    return 100.0 * sum(1 for x in b[-n - 1:-1] if x <= cur) / n


def ta_correlation(rt, st, source1, source2, length=None):
    n = max(int(length), 2)
    a = _buf(st, "a", _f(source1), n)
    c = _buf(st, "c", _f(source2), n)
    if len(a) < n:
        return NAN
    x, y = np.asarray(a[-n:]), np.asarray(c[-n:])
    if x.std() == 0 or y.std() == 0:
        return NAN
    return float(np.corrcoef(x, y)[0, 1])


def _default_hi(rt, a, b):
    if b is None:
        return rt.H[rt.i], a
    return a, b


def ta_highest(rt, st, a, b=None):
    src, n = _default_hi(rt, a, b)
    n = max(int(n), 1)
    buf = _buf(st, "b", _f(src), n)
    if len(buf) < n:
        return NAN
    return max(buf[-n:])


def ta_lowest(rt, st, a, b=None):
    src = rt.L[rt.i] if b is None else a
    n = a if b is None else b
    n = max(int(n), 1)
    buf = _buf(st, "b", _f(src), n)
    if len(buf) < n:
        return NAN
    return min(buf[-n:])


def ta_highestbars(rt, st, a, b=None):
    src, n = _default_hi(rt, a, b)
    n = max(int(n), 1)
    buf = _buf(st, "b", _f(src), n)
    if len(buf) < n:
        return NAN
    w = buf[-n:]
    m = max(w)
    return -float(n - 1 - max(k for k, v in enumerate(w) if v == m))


def ta_lowestbars(rt, st, a, b=None):
    src = rt.L[rt.i] if b is None else a
    n = a if b is None else b
    n = max(int(n), 1)
    buf = _buf(st, "b", _f(src), n)
    if len(buf) < n:
        return NAN
    w = buf[-n:]
    m = min(w)
    return -float(n - 1 - max(k for k, v in enumerate(w) if v == m))


def ta_range(rt, st, source, length=None):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n)
    return NAN if len(b) < n else max(b[-n:]) - min(b[-n:])


def ta_change(rt, st, source, length=1):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n + 1)
    return NAN if len(b) < n + 1 else b[-1] - b[-1 - n]


def ta_mom(rt, st, source, length=None):
    return ta_change(rt, st, source, length)


def ta_roc(rt, st, source, length=None):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n + 1)
    if len(b) < n + 1 or b[-1 - n] == 0:
        return NAN
    return 100.0 * (b[-1] - b[-1 - n]) / b[-1 - n]


def ta_cum(rt, st, source):
    v = _f(source)
    st["s"] = st.get("s", 0.0) + (0.0 if _nan(v) else v)
    return st["s"]


def ta_sum(rt, st, source, length=None):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n)
    return NAN if len(b) < n else sum(b[-n:])


def ta_cross_over(rt, st, a, b):
    a, b = _f(a), _f(b)
    pa, pb = st.get("a"), st.get("b")
    st["a"], st["b"] = a, b
    if pa is None or _nan(pa) or _nan(pb) or _nan(a) or _nan(b):
        return False
    return a > b and pa <= pb


def ta_cross_under(rt, st, a, b):
    a, b = _f(a), _f(b)
    pa, pb = st.get("a"), st.get("b")
    st["a"], st["b"] = a, b
    if pa is None or _nan(pa) or _nan(pb) or _nan(a) or _nan(b):
        return False
    return a < b and pa >= pb


def ta_cross(rt, st, a, b):
    a, b = _f(a), _f(b)
    pa, pb = st.get("a"), st.get("b")
    st["a"], st["b"] = a, b
    if pa is None or _nan(pa) or _nan(pb) or _nan(a) or _nan(b):
        return False
    return (a > b and pa <= pb) or (a < b and pa >= pb)


def ta_rising(rt, st, source, length=None):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n + 1)
    return len(b) >= n + 1 and all(b[-k] > b[-k - 1] for k in range(1, n + 1))


def ta_falling(rt, st, source, length=None):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n + 1)
    return len(b) >= n + 1 and all(b[-k] < b[-k - 1] for k in range(1, n + 1))


def ta_barssince(rt, st, condition):
    if condition and not _nan(_f(condition)):
        st["c"] = 0
    elif "c" in st:
        st["c"] += 1
    return float(st["c"]) if "c" in st else NAN


def ta_valuewhen(rt, st, condition, source, occurrence=0):
    lst = st.setdefault("l", [])
    if condition and not _nan(_f(condition)):
        lst.append(_f(source))
        if len(lst) > 200:
            del lst[:-100]
    k = int(occurrence)
    return lst[-1 - k] if len(lst) > k else NAN


def ta_macd(rt, st, source, fastlen=12, slowlen=26, siglen=9):
    src = _f(source)
    fast = _ema(st, "f", src, int(fastlen))
    slow = _ema(st, "s", src, int(slowlen))
    m = fast - slow if not (_nan(fast) or _nan(slow)) else NAN
    sig = _ema(st, "g", m, int(siglen))
    return (m, sig, m - sig if not (_nan(m) or _nan(sig)) else NAN)


def ta_bb(rt, st, source, length=20, mult=2.0):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n)
    if len(b) < n:
        return (NAN, NAN, NAN)
    w = np.asarray(b[-n:], dtype=float)
    basis, dev = float(w.mean()), float(w.std()) * mult
    return (basis, basis + dev, basis - dev)


def ta_bbw(rt, st, source, length=20, mult=2.0):
    basis, up, lo = ta_bb(rt, st, source, length, mult)
    return (up - lo) / basis if basis and not _nan(basis) else NAN


def ta_kc(rt, st, source, length=20, mult=2.0, use_true_range=True):
    n = max(int(length), 1)
    basis = _ema(st, "e", _f(source), n)
    rng = _tr_now(rt, st) if use_true_range else rt.H[rt.i] - rt.L[rt.i]
    r = _rma(st, "r", rng, n)
    if _nan(basis) or _nan(r):
        return (NAN, NAN, NAN)
    return (basis, basis + mult * r, basis - mult * r)


def ta_kcw(rt, st, source, length=20, mult=2.0, use_true_range=True):
    basis, up, lo = ta_kc(rt, st, source, length, mult, use_true_range)
    return (up - lo) / basis if basis and not _nan(basis) else NAN


def ta_stoch(rt, st, source, high, low, length=None):
    n = max(int(length), 1)
    hh = _buf(st, "h", _f(high), n)
    ll = _buf(st, "l", _f(low), n)
    if len(hh) < n:
        return NAN
    hi, lo = max(hh[-n:]), min(ll[-n:])
    return 100.0 * (_f(source) - lo) / (hi - lo) if hi != lo else 0.0


def ta_wpr(rt, st, length=None):
    n = max(int(length), 1)
    hh = _buf(st, "h", rt.H[rt.i], n)
    ll = _buf(st, "l", rt.L[rt.i], n)
    if len(hh) < n:
        return NAN
    hi, lo = max(hh[-n:]), min(ll[-n:])
    return 100.0 * (rt.C[rt.i] - hi) / (hi - lo) if hi != lo else 0.0


def ta_cci(rt, st, source, length=None):
    n = max(int(length), 1)
    b = _buf(st, "b", _f(source), n)
    if len(b) < n:
        return NAN
    w = b[-n:]
    m = sum(w) / n
    d = sum(abs(x - m) for x in w) / n
    return (w[-1] - m) / (0.015 * d) if d else 0.0


def ta_cmo(rt, st, source, length=None):
    n = max(int(length), 1)
    src = _f(source)
    prev = st.get("p")
    st["p"] = src
    if prev is None or _nan(prev) or _nan(src):
        return NAN
    ch = src - prev
    ups = _buf(st, "u", max(ch, 0.0), n)
    dns = _buf(st, "d", max(-ch, 0.0), n)
    if len(ups) < n:
        return NAN
    su, sd = sum(ups[-n:]), sum(dns[-n:])
    return 100.0 * (su - sd) / (su + sd) if (su + sd) else 0.0


def ta_mfi(rt, st, source, length=None):
    n = max(int(length), 1)
    src = _f(source)
    prev = st.get("p")
    st["p"] = src
    if prev is None:
        return NAN
    flow = src * rt.V[rt.i]
    pos = _buf(st, "pos", flow if src > prev else 0.0, n)
    neg = _buf(st, "neg", flow if src < prev else 0.0, n)
    if len(pos) < n:
        return NAN
    p, q = sum(pos[-n:]), sum(neg[-n:])
    return 100.0 if q == 0 else 100.0 - 100.0 / (1.0 + p / q)


def ta_dmi(rt, st, dilen=14, adxsmoothing=14):
    i = rt.i
    n, m = max(int(dilen), 1), max(int(adxsmoothing), 1)
    if i == 0:
        return (NAN, NAN, NAN)
    up = rt.H[i] - rt.H[i - 1]
    dn = rt.L[i - 1] - rt.L[i]
    plus = up if (up > dn and up > 0) else 0.0
    minus = dn if (dn > up and dn > 0) else 0.0
    tr = _rma(st, "tr", _tr_now(rt, st), n)
    pdm = _rma(st, "p", plus, n)
    mdm = _rma(st, "m", minus, n)
    if _nan(tr) or tr == 0 or _nan(pdm) or _nan(mdm):
        return (NAN, NAN, NAN)
    pdi, mdi = 100.0 * pdm / tr, 100.0 * mdm / tr
    s = pdi + mdi
    dx = 100.0 * abs(pdi - mdi) / s if s else 0.0
    adx = _rma(st, "x", dx, m)
    return (pdi, mdi, adx)


def ta_adx(rt, st, a=14, b=None):
    return ta_dmi(rt, st, a, a if b is None else b)[2]


def ta_supertrend(rt, st, factor, atrperiod):
    """Pine's own definition (direction -1 = up-trend with the line under price, +1 = down-trend)."""
    i = rt.i
    n = max(int(atrperiod), 1)
    a = _rma(st, "a", _tr_now(rt, st), n)
    hl2 = (rt.H[i] + rt.L[i]) / 2.0
    if _nan(a):
        st["has_atr"] = False
        return (NAN, 1)
    up, lo = hl2 + factor * a, hl2 - factor * a
    pl, pu = st.get("lo", 0.0), st.get("up", 0.0)
    pc = rt.C[i - 1] if i > 0 else rt.C[i]
    lo = lo if (lo > pl or pc < pl) else pl
    up = up if (up < pu or pc > pu) else pu
    if not st.get("has_atr"):
        d = 1
    elif st.get("line") == pu:
        d = -1 if rt.C[i] > up else 1
    else:
        d = 1 if rt.C[i] < lo else -1
    st["has_atr"] = True
    line = lo if d == -1 else up
    st["lo"], st["up"], st["line"] = lo, up, line
    return (line, d)


def ta_sar(rt, st, start=0.02, inc=0.02, maximum=0.2):
    i = rt.i
    if i == 0:
        return NAN
    s = st.get("s")
    if s is None:
        long_ = rt.C[i] >= rt.C[i - 1]
        st.update(s=rt.L[i - 1] if long_ else rt.H[i - 1], long=long_, ep=rt.H[i] if long_ else rt.L[i], af=start)
        return st["s"]
    s, long_, ep, af = st["s"], st["long"], st["ep"], st["af"]
    s = s + af * (ep - s)
    if long_:
        s = min(s, rt.L[i - 1], rt.L[i - 2] if i > 1 else rt.L[i - 1])
        if rt.L[i] < s:
            long_, s, ep, af = False, ep, rt.L[i], start
        elif rt.H[i] > ep:
            ep, af = rt.H[i], min(af + inc, maximum)
    else:
        s = max(s, rt.H[i - 1], rt.H[i - 2] if i > 1 else rt.H[i - 1])
        if rt.H[i] > s:
            long_, s, ep, af = True, ep, rt.H[i], start
        elif rt.L[i] < ep:
            ep, af = rt.L[i], min(af + inc, maximum)
    st.update(s=s, long=long_, ep=ep, af=af)
    return s


def _pivot(rt, st, src, left, right, hi):
    left, right = int(left), int(right)
    keep = left + right + 1
    b = _buf(st, "b", _f(src), keep)
    if len(b) < keep:
        return NAN
    w = b[-keep:]
    c = w[left]
    if _nan(c):
        return NAN
    others_l, others_r = w[:left], w[left + 1:]
    if hi:
        ok = all(c > x for x in others_l) and all(c >= x for x in others_r)
    else:
        ok = all(c < x for x in others_l) and all(c <= x for x in others_r)
    return c if ok else NAN


def ta_pivothigh(rt, st, a, b, c=None):
    if c is None:
        return _pivot(rt, st, rt.H[rt.i], a, b, True)
    return _pivot(rt, st, a, b, c, True)


def ta_pivotlow(rt, st, a, b, c=None):
    if c is None:
        return _pivot(rt, st, rt.L[rt.i], a, b, False)
    return _pivot(rt, st, a, b, c, False)


def ta_vwap(rt, st, source=None, anchor=None, stdev_mult=None):
    i = rt.i
    day = (rt.YEAR[i], rt.MONTH[i], rt.DAY[i])
    if st.get("day") != day:
        st.update(day=day, pv=0.0, v=0.0)
    src = (rt.H[i] + rt.L[i] + rt.C[i]) / 3.0 if source is None else _f(source)
    st["pv"] += src * rt.V[i]
    st["v"] += rt.V[i]
    return st["pv"] / st["v"] if st["v"] else src


def ta_tsi(rt, st, source, short_length=5, long_length=25):
    src = _f(source)
    prev = st.get("p")
    st["p"] = src
    if prev is None:
        return NAN
    ch = src - prev
    num = _ema(st, "n2", _ema(st, "n1", ch, int(long_length)), int(short_length))
    den = _ema(st, "d2", _ema(st, "d1", abs(ch), int(long_length)), int(short_length))
    return num / den if den and not _nan(den) else NAN


# ---------------------------------------------------------------------------------------------------------------
# vector helpers for `ta.tr` / `ta.obv` style variables
# ---------------------------------------------------------------------------------------------------------------
def tr_vec(rt):
    v = rt.__dict__.get("_tr")
    if v is None:
        pc = np.concatenate(([rt.Cn[0]], rt.Cn[:-1]))
        v = np.maximum(rt.Hn - rt.Ln, np.maximum(np.abs(rt.Hn - pc), np.abs(rt.Ln - pc)))
        v[0] = rt.Hn[0] - rt.Ln[0]
        rt._tr = v
    return v


def obv_vec(rt):
    v = rt.__dict__.get("_obv")
    if v is None:
        d = np.sign(np.diff(rt.Cn, prepend=rt.Cn[0]))
        v = np.cumsum(d * rt.Vn)
        rt._obv = v
    return v


def accdist_vec(rt):
    v = rt.__dict__.get("_ad")
    if v is None:
        rng = rt.Hn - rt.Ln
        mfm = np.where(rng > 0, ((rt.Cn - rt.Ln) - (rt.Hn - rt.Cn)) / np.where(rng > 0, rng, 1), 0.0)
        v = np.cumsum(mfm * rt.Vn)
        rt._ad = v
    return v


def pvt_vec(rt):
    v = rt.__dict__.get("_pvt")
    if v is None:
        pc = np.concatenate(([rt.Cn[0]], rt.Cn[:-1]))
        v = np.cumsum(np.where(pc != 0, (rt.Cn - pc) / np.where(pc != 0, pc, 1), 0.0) * rt.Vn)
        rt._pvt = v
    return v


def vwap_vec(rt):
    v = rt.__dict__.get("_vwap")
    if v is None:
        import pandas as pd
        tp = (rt.Hn + rt.Ln + rt.Cn) / 3.0
        day = pd.Series(rt.idx.normalize())
        pv = pd.Series(tp * rt.Vn).groupby(day.values).cumsum().to_numpy()
        vv = pd.Series(rt.Vn).groupby(day.values).cumsum().to_numpy()
        v = np.where(vv > 0, pv / np.where(vv > 0, vv, 1), tp)
        rt._vwap = v
    return v


# ---------------------------------------------------------------------------------------------------------------
# math / conversion / misc
# ---------------------------------------------------------------------------------------------------------------
def _m1(fn):
    def w(x, *a):
        x = _f(x)
        if _nan(x):
            return NAN
        try:
            return fn(x, *a)
        except (ValueError, OverflowError, ZeroDivisionError):
            return NAN
    return w


def m_max(*a):
    v = [_f(x) for x in a]
    return NAN if any(_nan(x) for x in v) else max(v)


def m_min(*a):
    v = [_f(x) for x in a]
    return NAN if any(_nan(x) for x in v) else min(v)


def m_round(x, precision=None):
    x = _f(x)
    if _nan(x):
        return NAN
    if precision is None:
        return int(math.floor(x + 0.5))
    p = 10 ** int(precision)
    return math.floor(x * p + 0.5) / p


def m_sign(x):
    x = _f(x)
    return NAN if _nan(x) else (1 if x > 0 else (-1 if x < 0 else 0))


def m_avg(*a):
    v = [_f(x) for x in a]
    return NAN if any(_nan(x) for x in v) else sum(v) / len(v)


def m_sum(*a):
    return sum(_f(x) for x in a)


def nz(x, repl=0.0):
    if x is None:
        return repl
    if isinstance(x, float) and x != x:
        return repl
    return x


def is_na(x=None):
    return x is None or (isinstance(x, float) and x != x)


def iff(c, a, b):
    return a if c else b


def to_int(x):
    x = _f(x)
    return NAN if _nan(x) else int(x)


def to_float(x):
    return _f(x)


def to_bool(x):
    x = _f(x)
    return False if _nan(x) else x != 0


def fixnan(rt, st, x):
    if not is_na(x):
        st["v"] = x
        return x
    return st.get("v", NAN)


def tostring(x, fmt=None):
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        if isinstance(x, float) and _nan(x):
            return "NaN"
        if isinstance(fmt, str) and "#" in fmt:
            dec = len(fmt.split(".")[1]) if "." in fmt else 0
            return f"{x:.{dec}f}"
        return str(x)
    return str(x)


# ---------------------------------------------------------------------------------------------------------------
# arrays (plain Python lists)
# ---------------------------------------------------------------------------------------------------------------
def _arr(a):
    if not isinstance(a, list):
        raise_ = __import__("app.strategy.base", fromlist=["StrategyError"]).StrategyError
        raise raise_("PineScript: array function used on a non-array value.")
    return a


def arr_new(size=0, initial=NAN):
    return [initial] * int(size)


def arr_from(*a):
    return list(a)


def arr_push(a, v):
    _arr(a).append(v)


def arr_get(a, i):
    a = _arr(a)
    i = int(i)
    return a[i] if -len(a) <= i < len(a) else NAN


def arr_set(a, i, v):
    a = _arr(a)
    i = int(i)
    if 0 <= i < len(a):
        a[i] = v


def arr_size(a):
    return len(_arr(a))


def arr_pop(a):
    a = _arr(a)
    return a.pop() if a else NAN


def arr_shift(a):
    a = _arr(a)
    return a.pop(0) if a else NAN


def arr_unshift(a, v):
    _arr(a).insert(0, v)


def arr_clear(a):
    _arr(a).clear()


def arr_insert(a, i, v):
    _arr(a).insert(int(i), v)


def arr_remove(a, i):
    a = _arr(a)
    i = int(i)
    return a.pop(i) if 0 <= i < len(a) else NAN


def arr_sum(a):
    return sum(_f(x) for x in _arr(a))


def arr_avg(a):
    a = _arr(a)
    return sum(_f(x) for x in a) / len(a) if a else NAN


def arr_max(a, nth=0):
    a = sorted((x for x in _arr(a) if not is_na(x)), reverse=True)
    return a[int(nth)] if len(a) > int(nth) else NAN


def arr_min(a, nth=0):
    a = sorted(x for x in _arr(a) if not is_na(x))
    return a[int(nth)] if len(a) > int(nth) else NAN


def arr_includes(a, v):
    return v in _arr(a)


def arr_indexof(a, v):
    a = _arr(a)
    return a.index(v) if v in a else -1


def arr_first(a):
    a = _arr(a)
    return a[0] if a else NAN


def arr_last(a):
    a = _arr(a)
    return a[-1] if a else NAN


def arr_copy(a):
    return list(_arr(a))


def arr_sort(a, order=None):
    _arr(a).sort(reverse=(order == "desc"))


def arr_stdev(a):
    a = [_f(x) for x in _arr(a)]
    return float(np.std(a)) if a else NAN


def arr_median(a):
    a = [_f(x) for x in _arr(a) if not is_na(x)]
    return float(np.median(a)) if a else NAN
