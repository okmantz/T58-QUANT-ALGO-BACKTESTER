"""Indicators and strategy families for the Discovery Lab.

Every family is ``build(ind, p) -> (long, short)`` -- two boolean arrays that are True on the bar whose CLOSE completes the setup. The
trade is entered at the NEXT bar's open, so nothing here can see the future. ``ind`` is an ``Indicators`` cache over one OHLC frame.

All arrays are plain numpy; NaNs (warm-up) never fire a signal.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------------------------- indicators
def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _std(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def _shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if k < len(x):
        out[k:] = x[:-k]
    return out


class Indicators:
    """Memoised indicator bank over one OHLC(V) frame."""

    def __init__(self, df: pd.DataFrame):
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.n = len(self.c)
        ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.Series(pd.RangeIndex(self.n))
        self.ts = ts
        self.hour = ts.dt.hour.to_numpy() if hasattr(ts, "dt") else np.zeros(self.n, int)
        self.day = (ts.dt.normalize().astype("int64").to_numpy() if hasattr(ts, "dt") else np.arange(self.n))
        self._m: dict = {}

    def memo(self, key, fn):
        if key not in self._m:
            self._m[key] = fn()
        return self._m[key]

    def ema(self, n): return self.memo(("ema", n), lambda: _ema(self.c, n))
    def sma(self, n): return self.memo(("sma", n), lambda: _sma(self.c, n))

    def tr(self):
        return self.memo("tr", lambda: np.maximum(self.h - self.l, np.maximum(np.abs(self.h - _shift(self.c)), np.abs(self.l - _shift(self.c)))))

    def atr(self, n=14): return self.memo(("atr", n), lambda: _wilder(np.nan_to_num(self.tr(), nan=0.0), n))

    def rsi(self, n=14):
        def f():
            d = np.diff(self.c, prepend=self.c[0])
            up, dn = _wilder(np.maximum(d, 0), n), _wilder(np.maximum(-d, 0), n)
            return 100 - 100 / (1 + up / np.where(dn == 0, 1e-12, dn))
        return self.memo(("rsi", n), f)

    def bb(self, n, k):
        def f():
            m, s = self.sma(n), _std(self.c, n)
            return m, m + k * s, m - k * s, s
        return self.memo(("bb", n, k), f)

    def donch(self, n):
        def f():
            hi = pd.Series(self.h).rolling(n, min_periods=n).max().to_numpy()
            lo = pd.Series(self.l).rolling(n, min_periods=n).min().to_numpy()
            return _shift(hi), _shift(lo)             # prior n bars only
        return self.memo(("don", n), f)

    def stoch(self, n):
        def f():
            hi = pd.Series(self.h).rolling(n, min_periods=n).max().to_numpy()
            lo = pd.Series(self.l).rolling(n, min_periods=n).min().to_numpy()
            return 100 * (self.c - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
        return self.memo(("sto", n), f)

    def macd(self, a, b, s):
        def f():
            line = _ema(self.c, a) - _ema(self.c, b)
            sig = pd.Series(line).ewm(span=s, adjust=False, min_periods=s).mean().to_numpy()
            return line, sig
        return self.memo(("macd", a, b, s), f)

    def adx(self, n=14):
        def f():
            up, dn = np.diff(self.h, prepend=self.h[0]), -np.diff(self.l, prepend=self.l[0])
            pdm, mdm = np.where((up > dn) & (up > 0), up, 0.0), np.where((dn > up) & (dn > 0), dn, 0.0)
            a = self.atr(n)
            pdi, mdi = 100 * _wilder(pdm, n) / np.where(a == 0, np.nan, a), 100 * _wilder(mdm, n) / np.where(a == 0, np.nan, a)
            dx = 100 * np.abs(pdi - mdi) / np.where(pdi + mdi == 0, np.nan, pdi + mdi)
            return _wilder(np.nan_to_num(dx), n), pdi, mdi
        return self.memo(("adx", n), f)

    def cci(self, n):
        def f():
            tp = (self.h + self.l + self.c) / 3
            m = _sma(tp, n)
            md = pd.Series(tp).rolling(n, min_periods=n).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True).to_numpy()
            return (tp - m) / (0.015 * np.where(md == 0, np.nan, md))
        return self.memo(("cci", n), f)

    def z(self, n):
        return self.memo(("z", n), lambda: (self.c - _sma(self.c, n)) / np.where(_std(self.c, n) == 0, np.nan, _std(self.c, n)))

    def keltner(self, n, m):
        def f():
            mid, a = self.ema(n), self.atr(n)
            return mid, mid + m * a, mid - m * a
        return self.memo(("kel", n, m), f)

    def supertrend(self, n, m):
        def f():
            a = self.atr(n)
            hl2 = (self.h + self.l) / 2
            ub, lb = hl2 + m * a, hl2 - m * a
            d = np.zeros(self.n, dtype=np.int8)
            fu, fl = ub.copy(), lb.copy()
            for i in range(1, self.n):
                if np.isnan(a[i]):
                    continue
                fu[i] = ub[i] if (ub[i] < fu[i - 1] or self.c[i - 1] > fu[i - 1] or np.isnan(fu[i - 1])) else fu[i - 1]
                fl[i] = lb[i] if (lb[i] > fl[i - 1] or self.c[i - 1] < fl[i - 1] or np.isnan(fl[i - 1])) else fl[i - 1]
                d[i] = 1 if self.c[i] > fu[i - 1] else (-1 if self.c[i] < fl[i - 1] else d[i - 1])
            return d
        return self.memo(("st", n, m), f)

    def roc(self, n): return self.memo(("roc", n), lambda: self.c / _shift(self.c, n) - 1.0)

    def opening_range(self, k):
        """High/low of each day's first k bars, valid only AFTER those k bars have closed (NaN before)."""
        def f():
            s = pd.DataFrame({"d": self.day, "h": self.h, "l": self.l})
            pos = s.groupby("d").cumcount().to_numpy()
            hi = s.where(pos < k).groupby("d")["h"].transform("max").to_numpy()
            lo = s.where(pos < k).groupby("d")["l"].transform("min").to_numpy()
            hi, lo = np.where(pos >= k, hi, np.nan), np.where(pos >= k, lo, np.nan)
            return hi, lo, pos
        return self.memo(("or", k), f)


def _cross_up(a, b): return (a > b) & (_shift(a) <= _shift(b))
def _cross_dn(a, b): return (a < b) & (_shift(a) >= _shift(b))
def _ok(*arrs):
    m = np.ones(len(arrs[0]), bool)
    for a in arrs:
        m &= ~np.isnan(a)
    return m


# ---------------------------------------------------------------------------------------------------------------- families
@dataclass(frozen=True)
class Family:
    name: str
    label: str
    group: str            # trend | reversion | breakout | momentum | session
    about: str
    grid: dict
    build: Callable


def _f_ema_cross(I, p):
    f, s = I.ema(p["fast"]), I.ema(p["slow"])
    adx = I.adx(14)[0]
    ok = _ok(f, s) & (adx >= p["adx_min"]) if p["adx_min"] else _ok(f, s)
    return _cross_up(f, s) & ok, _cross_dn(f, s) & ok


def _f_donchian(I, p):
    hi, lo = I.donch(p["n"])
    trend = I.ema(p["trend"]) if p["trend"] else None
    ok = _ok(hi, lo)
    lg, sh = (I.c > hi) & ok, (I.c < lo) & ok
    if trend is not None:
        lg &= I.c > trend
        sh &= I.c < trend
    return lg & ~_shift(lg.astype(float), 1).astype(bool), sh & ~_shift(sh.astype(float), 1).astype(bool)


def _f_rsi_rev(I, p):
    r = I.rsi(p["n"])
    ok = _ok(r)
    lg, sh = _cross_up(r, np.full(I.n, p["lo"])) & ok, _cross_dn(r, np.full(I.n, 100 - p["lo"])) & ok
    if p["trend"]:
        t = I.sma(p["trend"])
        lg &= I.c > t
        sh &= I.c < t
    return lg, sh


def _f_boll_rev(I, p):
    m, up, lo, _ = I.bb(p["n"], p["k"])
    ok = _ok(m)
    return _cross_up(I.c, lo) & ok & (_shift(I.c) < _shift(lo)), _cross_dn(I.c, up) & ok & (_shift(I.c) > _shift(up))


def _f_boll_break(I, p):
    m, up, lo, s = I.bb(p["n"], 2.0)
    width = (up - lo) / np.where(m == 0, np.nan, m)
    q = pd.Series(width).rolling(p["look"], min_periods=p["look"]).quantile(p["squeeze_q"]).to_numpy()
    squeezed = _shift((width <= q).astype(float)).astype(bool)
    ok = _ok(m, q)
    return (I.c > up) & squeezed & ok, (I.c < lo) & squeezed & ok


def _f_macd(I, p):
    line, sig = I.macd(p["a"], p["b"], p["s"])
    ok = _ok(line, sig)
    lg, sh = _cross_up(line, sig) & ok, _cross_dn(line, sig) & ok
    if p["zero"]:
        lg &= line < 0
        sh &= line > 0
    return lg, sh


def _f_stoch(I, p):
    k = I.stoch(p["n"])
    ok = _ok(k)
    return _cross_up(k, np.full(I.n, p["lo"])) & ok, _cross_dn(k, np.full(I.n, 100 - p["lo"])) & ok


def _f_keltner(I, p):
    mid, up, lo = I.keltner(p["n"], p["m"])
    ok = _ok(mid)
    return (I.c > up) & ~(_shift(I.c) > _shift(up)) & ok, (I.c < lo) & ~(_shift(I.c) < _shift(lo)) & ok


def _f_supertrend(I, p):
    d = I.supertrend(p["n"], p["m"])
    pd_ = _shift(d.astype(float)).astype(int)
    return (d == 1) & (pd_ == -1), (d == -1) & (pd_ == 1)


def _f_zrev(I, p):
    z = I.z(p["n"])
    ok = _ok(z)
    return _cross_up(z, np.full(I.n, -p["z"])) & ok, _cross_dn(z, np.full(I.n, p["z"])) & ok


def _f_momo(I, p):
    r = I.roc(p["n"])
    thr = p["thr_atr"] * I.atr(14) / np.where(I.c == 0, np.nan, I.c) * np.sqrt(p["n"])
    ok = _ok(r, thr)
    lg, sh = (r > thr) & ok, (r < -thr) & ok
    return lg & ~_shift(lg.astype(float)).astype(bool), sh & ~_shift(sh.astype(float)).astype(bool)


def _f_cci(I, p):
    c = I.cci(p["n"])
    ok = _ok(c)
    return _cross_up(c, np.full(I.n, -p["lvl"])) & ok, _cross_dn(c, np.full(I.n, p["lvl"])) & ok


def _f_pullback(I, p):
    t, r = I.ema(p["trend"]), I.rsi(p["n"])
    ok = _ok(t, r)
    up, dn = (I.c > t) & ok, (I.c < t) & ok
    return up & _cross_up(r, np.full(I.n, p["dip"])), dn & _cross_dn(r, np.full(I.n, 100 - p["dip"]))


def _f_inside(I, p):
    inside = (I.h < _shift(I.h)) & (I.l > _shift(I.l))
    prev_in = _shift(inside.astype(float)).astype(bool)
    t = I.ema(p["trend"]) if p["trend"] else None
    lg, sh = prev_in & (I.c > _shift(I.h, 2)), prev_in & (I.c < _shift(I.l, 2))
    if t is not None:
        ok = _ok(t)
        lg &= (I.c > t) & ok
        sh &= (I.c < t) & ok
    return lg, sh


def _f_orb(I, p):
    hi, lo, pos = I.opening_range(p["k"])
    ok = _ok(hi, lo)
    first = pd.Series((I.c > hi) & ok).groupby(I.day).cumsum().to_numpy() == 1
    firs = pd.Series((I.c < lo) & ok).groupby(I.day).cumsum().to_numpy() == 1
    return ((I.c > hi) & ok & first), ((I.c < lo) & ok & firs)


def _f_failbreak(I, p):
    hi, lo = I.donch(p["n"])
    ok = _ok(hi, lo)
    return (I.l < lo) & (I.c > lo) & ok, (I.h > hi) & (I.c < hi) & ok          # sweep the range extreme, close back inside


def _f_adx_di(I, p):
    adx, pdi, mdi = I.adx(p["n"])
    ok = _ok(adx) & (adx >= p["min"])
    return _cross_up(pdi, mdi) & ok, _cross_dn(pdi, mdi) & ok


def _f_wpr(I, p):
    w = I.stoch(p["n"]) - 100                                      # Williams %R in [-100, 0]
    ok = _ok(w)
    return _cross_up(w, np.full(I.n, -100 + p["lo"])) & ok, _cross_dn(w, np.full(I.n, -p["lo"])) & ok


FAMILIES: dict[str, Family] = {f.name: f for f in (
    Family("ema_cross", "EMA cross", "trend", "Fast EMA crosses slow EMA, optionally only when ADX says the market is trending.",
           {"fast": [8, 12, 20, 30], "slow": [40, 55, 80, 120], "adx_min": [0, 18, 25]}, _f_ema_cross),
    Family("supertrend", "Supertrend flip", "trend", "ATR trailing band flips side: a clean trend-following trigger.",
           {"n": [7, 10, 14, 20], "m": [2.0, 2.5, 3.0, 4.0]}, _f_supertrend),
    Family("adx_di", "ADX / DI cross", "trend", "+DI crosses -DI while ADX confirms trend strength.",
           {"n": [10, 14, 20], "min": [15, 20, 25, 30]}, _f_adx_di),
    Family("macd", "MACD signal cross", "trend", "MACD line crosses its signal line; optional 'only from the far side of zero'.",
           {"a": [8, 12, 16], "b": [21, 26, 34], "s": [7, 9, 12], "zero": [0, 1]}, _f_macd),
    Family("donchian", "Donchian breakout", "breakout", "Close breaks the prior N-bar high/low, optionally with an EMA trend filter.",
           {"n": [20, 30, 55, 80, 120], "trend": [0, 100, 200]}, _f_donchian),
    Family("keltner", "Keltner breakout", "breakout", "Close pushes outside an ATR channel around an EMA (volatility expansion).",
           {"n": [14, 20, 30, 50], "m": [1.5, 2.0, 2.5, 3.0]}, _f_keltner),
    Family("squeeze", "Bollinger squeeze breakout", "breakout", "Volatility was compressed (narrow bands), then price breaks out.",
           {"n": [20, 30], "look": [100, 200], "squeeze_q": [0.1, 0.2, 0.3]}, _f_boll_break),
    Family("inside_bar", "Inside-bar breakout", "breakout", "An inside bar (compression) followed by a break of the mother bar.",
           {"trend": [0, 50, 100, 200]}, _f_inside),
    Family("orb", "Opening-range breakout", "session", "First break of the day's opening range (first k bars). Intraday data only.",
           {"k": [2, 3, 4, 6, 8, 12]}, _f_orb),
    Family("rsi_rev", "RSI reversal", "reversion", "RSI leaves oversold/overbought; optional 'only with the long-term trend'.",
           {"n": [2, 3, 5, 7, 14], "lo": [10, 20, 30], "trend": [0, 100, 200]}, _f_rsi_rev),
    Family("boll_rev", "Bollinger reversion", "reversion", "Close re-enters the bands after an excursion outside them.",
           {"n": [14, 20, 30, 50], "k": [1.5, 2.0, 2.5, 3.0]}, _f_boll_rev),
    Family("zscore_rev", "Z-score reversion", "reversion", "Price stretched N sigma from its mean starts to snap back.",
           {"n": [20, 30, 50, 100], "z": [1.5, 2.0, 2.5, 3.0]}, _f_zrev),
    Family("stoch_rev", "Stochastic reversal", "reversion", "Stochastic leaves an extreme zone.",
           {"n": [5, 9, 14, 21], "lo": [10, 15, 20, 25]}, _f_stoch),
    Family("cci_rev", "CCI extreme reversal", "reversion", "CCI exits a +/- extreme.",
           {"n": [14, 20, 40], "lvl": [100, 150, 200]}, _f_cci),
    Family("willr", "Williams %R reversal", "reversion", "Williams %R leaves an extreme.",
           {"n": [10, 14, 21], "lo": [10, 15, 20]}, _f_wpr),
    Family("failed_break", "Failed-breakout sweep", "reversion", "Price sweeps the N-bar extreme then closes back inside (stop-run reversal).",
           {"n": [10, 20, 40, 60]}, _f_failbreak),
    Family("pullback", "Trend pullback", "trend", "In an EMA trend, RSI dips then turns back with the trend.",
           {"trend": [50, 100, 200], "n": [3, 5, 14], "dip": [30, 40, 50]}, _f_pullback),
    Family("momentum", "Time-series momentum", "momentum", "N-bar return beats a volatility-scaled threshold; ride it.",
           {"n": [10, 20, 40, 80], "thr_atr": [0.5, 1.0, 1.5]}, _f_momo),
)}

EXIT_GRID = {"stop_atr": [1.5, 2.0, 2.5, 3.0, 4.0], "target_r": [1.0, 1.5, 2.0, 3.0, 5.0]}


def grid_size(fam: Family) -> int:
    n = 1
    for v in fam.grid.values():
        n *= len(v)
    return n


def all_param_sets(fam: Family) -> list[dict]:
    keys = list(fam.grid)
    out = [dict(zip(keys, combo)) for combo in itertools.product(*(fam.grid[k] for k in keys))]
    if fam.name == "ema_cross":
        out = [p for p in out if p["fast"] < p["slow"]]
    if fam.name == "macd":
        out = [p for p in out if p["a"] < p["b"]]
    return out


def neighbours(fam: Family, params: dict) -> list[dict]:
    """Grid neighbours: change ONE parameter by one step. Used by the robustness check."""
    out = []
    for k, vals in fam.grid.items():
        if len(vals) < 2 or params[k] not in vals:
            continue
        i = vals.index(params[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                q = dict(params)
                q[k] = vals[j]
                out.append(q)
    return out


def signals(name: str, df: pd.DataFrame, params: dict, ind: Indicators | None = None):
    ind = ind or Indicators(df)
    lg, sh = FAMILIES[name].build(ind, params)
    return np.nan_to_num(lg, nan=False).astype(bool), np.nan_to_num(sh, nan=False).astype(bool), ind
