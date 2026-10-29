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
        # Exchange clock (US Central for CME data) when the frame carries it: session windows and "hour of day" effects must follow
        # the exchange's day, not UTC. Falls back to the timestamp's own clock.
        xt = None
        if "exchange_time" in df.columns:
            try:
                xt = pd.to_datetime(df["exchange_time"])
            except Exception:  # noqa: BLE001
                xt = None
        if xt is not None and hasattr(xt, "dt"):
            self.xhour = xt.dt.hour.to_numpy()
            self.xdow = xt.dt.dayofweek.to_numpy()
        else:
            self.xhour = self.hour
            self.xdow = ts.dt.day_of_week.to_numpy() if hasattr(ts, "dt") else np.zeros(self.n, int)
        if "trading_day" in df.columns:
            try:
                self.tday = pd.to_datetime(df["trading_day"]).astype("int64").to_numpy()
            except Exception:  # noqa: BLE001
                self.tday = self.day
        else:
            self.tday = self.day
        self._m: dict = {}
        self.df = df

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

    def day_pos(self):
        """Bar number within its trading day (0 = first bar) and that day's first open."""
        def f():
            s_ = pd.DataFrame({"d": self.tday, "o": self.o})
            pos = s_.groupby("d").cumcount().to_numpy()
            dopen = s_.groupby("d")["o"].transform("first").to_numpy()
            return pos, dopen
        return self.memo("daypos", f)

    def prev_day_close(self):
        """Close of the previous trading day, known from the first bar of the new day onward."""
        def f():
            s_ = pd.DataFrame({"d": self.tday, "c": self.c})
            last = s_.groupby("d")["c"].transform("last").to_numpy()
            first_idx = s_.groupby("d").cumcount().to_numpy() == 0
            prev = np.full(self.n, np.nan)
            cur = np.nan
            lastc = np.nan
            for i in range(self.n):
                if first_idx[i]:
                    cur = lastc
                prev[i] = cur
                lastc = self.c[i]
            return prev
        return self.memo("pdc", f)

    def atr_ratio(self, n=200):
        return self.memo(("atrr", n), lambda: self.atr(14) / np.where(_sma(np.nan_to_num(self.atr(14)), n) == 0, np.nan, _sma(np.nan_to_num(self.atr(14)), n)))

    def opening_range(self, k):
        """High/low of each day's first k bars, valid only AFTER those k bars have closed (NaN before)."""
        def f():
            s = pd.DataFrame({"d": self.day, "h": self.h, "l": self.l})
            pos = s.groupby("d").cumcount().to_numpy()
            cond = pd.Series(pos < k, index=s.index)
            hi = s["h"].where(cond).groupby(s["d"]).transform("max").to_numpy()
            lo = s["l"].where(cond).groupby(s["d"]).transform("min").to_numpy()
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
    source: str = ""       # which paper in the Library / which known effect this family is drawn from
    categorical: tuple = ()  # grid keys whose values are NAMES, not an ordered scale: never 'one step' apart (neighbours skip them)


def _f_ind_rule(I, p):
    from app.discovery_lab import indicator_pool as _P
    return _P.rule_signals(I, p)


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
    pd_ = np.nan_to_num(_shift(d.astype(float)), nan=0.0).astype(int)
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


def _f_ibs(I, p):
    rng = np.where(I.h - I.l == 0, np.nan, I.h - I.l)
    ibs = (I.c - I.l) / rng
    t = I.ema(p["trend"]) if p["trend"] else None
    ok = _ok(ibs)
    lg, sh = (ibs < p["lo"]) & ok, (ibs > 1 - p["lo"]) & ok
    if t is not None:
        lg &= I.c > t
        sh &= I.c < t
    return lg, sh


def _f_streak(I, p):
    d = np.sign(np.diff(I.c, prepend=I.c[0]))
    dn = pd.Series((d < 0).astype(float)).rolling(p["n"], min_periods=p["n"]).sum().to_numpy() == p["n"]
    up = pd.Series((d > 0).astype(float)).rolling(p["n"], min_periods=p["n"]).sum().to_numpy() == p["n"]
    if p["trend"]:
        t = I.ema(p["trend"])
        dn &= I.c > t
        up &= I.c < t
    return dn, up


def _f_tod(I, p):
    hit = (I.xhour == p["hour"])
    if p["dow"] >= 0:
        hit &= (I.xdow == p["dow"])
    return (hit if p["dir"] == 1 else np.zeros(I.n, bool)), (hit if p["dir"] == -1 else np.zeros(I.n, bool))


def _f_gap(I, p):
    pos, dopen = I.day_pos()
    pdc = I.prev_day_close()
    gap = (dopen - pdc) / np.where(I.atr(14) == 0, np.nan, I.atr(14))
    ok = _ok(gap) & (pos == 0)
    big_up, big_dn = (gap > p["k"]) & ok, (gap < -p["k"]) & ok
    return (big_up, big_dn) if p["mode"] == 1 else (big_dn, big_up)           # go with the gap, or fade it


def _f_sess_mom(I, p):
    pos, dopen = I.day_pos()
    mv = (I.c - dopen) / np.where(I.atr(14) == 0, np.nan, I.atr(14))
    hit = (I.xhour == p["hour"]) & _ok(mv)
    up, dn = hit & (mv > p["thr"]), hit & (mv < -p["thr"])
    return (up, dn) if p["dir"] == 1 else (dn, up)                              # follow the session's move, or fade it


def _f_price_ma(I, p):
    m = I.ema(p["n"])
    ok = _ok(m)
    return _cross_up(I.c, m) & ok, _cross_dn(I.c, m) & ok


def _f_nr(I, p):
    rng = I.h - I.l
    nr = rng == pd.Series(rng).rolling(p["n"], min_periods=p["n"]).min().to_numpy()
    prev_nr = _shift(nr.astype(float)).astype(bool)
    return prev_nr & (I.c > _shift(I.h)), prev_nr & (I.c < _shift(I.l))


def _f_body(I, p):
    rng = np.where(I.h - I.l == 0, np.nan, I.h - I.l)
    body = (I.c - I.o) / rng
    ok = _ok(body)
    big_dn, big_up = (body < -p["thr"]) & ok, (body > p["thr"]) & ok
    lg, sh = (big_dn, big_up) if p["mode"] == 1 else (big_up, big_dn)           # fade the big bar, or ride it
    if p["trend"]:
        t = I.ema(p["trend"])
        lg &= I.c > t
        sh &= I.c < t
    return lg, sh


def _f_distma(I, p):
    d = (I.c - I.ema(p["n"])) / np.where(I.atr(14) == 0, np.nan, I.atr(14))
    ok = _ok(d)
    return _cross_up(d, np.full(I.n, -p["k"])) & ok, _cross_dn(d, np.full(I.n, p["k"])) & ok


def _f_volexp(I, p):
    fast = I.atr(5)
    slow = I.atr(p["slow"])
    ratio = fast / np.where(slow == 0, np.nan, slow)
    ok = _ok(ratio) & (ratio > p["thr"])
    return ok & (I.c > I.o), ok & (I.c < I.o)


def _f_lowrev(I, p):
    lo = pd.Series(I.c).rolling(p["n"], min_periods=p["n"]).min().to_numpy()
    hi = pd.Series(I.c).rolling(p["n"], min_periods=p["n"]).max().to_numpy()
    ok = _ok(lo, hi)
    lg, sh = (I.c <= lo) & ok, (I.c >= hi) & ok
    if p["trend"]:
        t = I.ema(p["trend"])
        lg &= I.c > t
        sh &= I.c < t
    return lg, sh


# ---------------------------------------------------------------------------------------------------------------- filters
HOUR_WINDOWS = {"all": None, "rth": (8, 15), "morning": (8, 11), "afternoon": (11, 15), "overnight": (17, 8), "us_open": (8, 10)}
FILTER_GRID = {
    "side": ["both", "long", "short"],
    "trend": [0, 50, 100, 200],
    "hours": list(HOUR_WINDOWS),
    "vol": ["all", "high", "low"],
}


def apply_filters(I, lg, sh, flt: dict | None):
    """Optional regime filters, all causal (each looks only at information up to the signal bar's close):
    side (long/short only), trend (long only above / short only below an EMA), hours (exchange-clock session window),
    vol (ATR above/below its own 200-bar average -- the volatility regime idea from the Library's regime papers)."""
    if not flt:
        return lg, sh
    side = flt.get("side", "both")
    if side == "long":
        sh = np.zeros_like(sh)
    elif side == "short":
        lg = np.zeros_like(lg)
    tr = flt.get("trend", 0)
    if tr:
        t = I.ema(int(tr))
        okt = ~np.isnan(t)
        lg = lg & (I.c > t) & okt
        sh = sh & (I.c < t) & okt
    win = HOUR_WINDOWS.get(flt.get("hours", "all"))
    if win:
        a, b = win
        m = ((I.xhour >= a) & (I.xhour < b)) if a < b else ((I.xhour >= a) | (I.xhour < b))
        lg, sh = lg & m, sh & m
    v = flt.get("vol", "all")
    if v != "all":
        r = I.atr_ratio(200)
        m = (r > 1.1) if v == "high" else (r < 0.9)
        m = m & ~np.isnan(r)
        lg, sh = lg & m, sh & m
    return lg, sh


from app.discovery_lab.indicator_pool import POOL_KINDS as _POOL_KINDS  # noqa: E402

FAMILIES: dict[str, Family] = {f.name: f for f in (
    Family("ema_cross", "EMA cross", "trend", "Fast EMA crosses slow EMA, optionally only when ADX says the market is trending.",
           {"fast": [8, 12, 20, 30], "slow": [40, 55, 80, 120], "adx_min": [0, 18, 25]}, _f_ema_cross, 'Momentum Strategies in Futures Markets and Trend-Following Funds'),
    Family("supertrend", "Supertrend flip", "trend", "ATR trailing band flips side: a clean trend-following trigger.",
           {"n": [7, 10, 14, 20], "m": [2.0, 2.5, 3.0, 4.0]}, _f_supertrend, 'Trend Following, Risk Parity and Momentum in Commodity Futures'),
    Family("adx_di", "ADX / DI cross", "trend", "+DI crosses -DI while ADX confirms trend strength.",
           {"n": [10, 14, 20], "min": [15, 20, 25, 30]}, _f_adx_di),
    Family("macd", "MACD signal cross", "trend", "MACD line crosses its signal line; optional 'only from the far side of zero'.",
           {"a": [8, 12, 16], "b": [21, 26, 34], "s": [7, 9, 12], "zero": [0, 1]}, _f_macd),
    Family("donchian", "Donchian breakout", "breakout", "Close breaks the prior N-bar high/low, optionally with an EMA trend filter.",
           {"n": [20, 30, 55, 80, 120], "trend": [0, 100, 200]}, _f_donchian, 'Comparative Study of Breakout and Mean-Reversion Strategies'),
    Family("keltner", "Keltner breakout", "breakout", "Close pushes outside an ATR channel around an EMA (volatility expansion).",
           {"n": [14, 20, 30, 50], "m": [1.5, 2.0, 2.5, 3.0]}, _f_keltner),
    Family("squeeze", "Bollinger squeeze breakout", "breakout", "Volatility was compressed (narrow bands), then price breaks out.",
           {"n": [20, 30], "look": [100, 200], "squeeze_q": [0.1, 0.2, 0.3]}, _f_boll_break),
    Family("inside_bar", "Inside-bar breakout", "breakout", "An inside bar (compression) followed by a break of the mother bar.",
           {"trend": [0, 50, 100, 200]}, _f_inside),
    Family("orb", "Opening-range breakout", "session", "First break of the day's opening range (first k bars). Intraday data only.",
           {"k": [2, 3, 4, 6, 8, 12]}, _f_orb),
    Family("rsi_rev", "RSI reversal", "reversion", "RSI leaves oversold/overbought; optional 'only with the long-term trend'.",
           {"n": [2, 3, 5, 7, 14], "lo": [10, 20, 30], "trend": [0, 100, 200]}, _f_rsi_rev, 'Design and Development of Mean Reversion Strategies'),
    Family("boll_rev", "Bollinger reversion", "reversion", "Close re-enters the bands after an excursion outside them.",
           {"n": [14, 20, 30, 50], "k": [1.5, 2.0, 2.5, 3.0]}, _f_boll_rev, 'Analysis of the Bollinger Bands and Mean Reversion Trading Strategy in the S&P 500'),
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
    Family("ibs_rev", "Internal-bar-strength reversion", "reversion", "Close near the bar's low (or high) tends to bounce -- the classic index 'IBS' effect, long-biased in an uptrend.",
           {"lo": [0.05, 0.1, 0.15, 0.2, 0.3], "trend": [0, 50, 100, 200]}, _f_ibs, "101 Formulaic Alphas; Mean-reversion papers (S&P 500)"),
    Family("streak_rev", "Consecutive-close reversal", "reversion", "N lower closes in a row inside an uptrend (or N higher in a downtrend) -- short-term exhaustion.",
           {"n": [2, 3, 4, 5, 6], "trend": [0, 100, 200]}, _f_streak, "Mean Reversion Trading Strategies and the Black Swan Events"),
    Family("low_rev", "N-bar extreme reversion", "reversion", "Close at the lowest (highest) close of the last N bars while the bigger trend is intact.",
           {"n": [5, 10, 20, 40, 60], "trend": [0, 100, 200]}, _f_lowrev, "Comparative Study of Breakout and Mean-Reversion Strategies"),
    Family("dist_ma_rev", "Stretch-from-average reversion", "reversion", "Price stretched k ATR below (above) its EMA snaps back across the threshold.",
           {"n": [20, 50, 100], "k": [1.0, 1.5, 2.0, 3.0, 4.0]}, _f_distma, "Bollinger/mean-reversion papers"),
    Family("body_bar", "Big-bar fade or follow", "momentum", "A bar whose body fills most of its range is faded (mode 1) or ridden (mode -1).",
           {"thr": [0.6, 0.75, 0.9], "mode": [1, -1], "trend": [0, 100, 200]}, _f_body, "101 Formulaic Alphas (body / range alphas)"),
    Family("vol_expansion", "Volatility-expansion thrust", "momentum", "Short-term ATR jumps well above its longer average and the bar closes with direction -- ride the expansion.",
           {"thr": [1.3, 1.6, 2.0, 2.5], "slow": [50, 100]}, _f_volexp, "Volatility-Volume-Gap regime classifier; Structural Limits of OHLCV Signals"),
    Family("price_ma", "Price crosses long average", "trend", "Close crosses a long EMA: slow trend-following in the spirit of the century-long trend evidence.",
           {"n": [50, 100, 150, 200, 300, 500]}, _f_price_ma, "A Century of Evidence on Trend-Following Investing"),
    Family("nr_break", "Narrow-range breakout", "breakout", "The narrowest bar of the last N is followed by a break of its high/low.",
           {"n": [4, 7, 10, 14]}, _f_nr, "Breakout vs mean-reversion study"),
    Family("tod_hold", "Time-of-day entry", "session", "Enter at a fixed exchange-clock hour (optionally one weekday) and let the exit rules decide -- tests intraday/overnight drift.",
           {"hour": list(range(0, 24)), "dir": [1, -1], "dow": [-1]}, _f_tod, "Intraday momentum / overnight drift literature; Day Trading: Adapting to Change"),
    Family("gap_play", "Opening gap go-or-fade", "session", "A gap of k ATR at the new session's first bar is followed (mode 1) or faded (mode -1).",
           {"k": [0.25, 0.5, 0.75, 1.0, 1.5], "mode": [1, -1]}, _f_gap, "Volatility-Volume-Gap classifier (MNQ)"),
    Family("session_mom", "Session momentum / fade", "session", "At a chosen hour, if the session has moved more than k ATR from its open, follow it (or fade it).",
           {"hour": [9, 10, 11, 12, 13, 14], "thr": [0.25, 0.5, 1.0, 1.5], "dir": [1, -1]}, _f_sess_mom, "Intraday momentum (first part of the day predicts the last)"),
    Family("pullback", "Trend pullback", "trend", "In an EMA trend, RSI dips then turns back with the trend.",
           {"trend": [50, 100, 200], "n": [3, 5, 14], "dip": [30, 40, 50]}, _f_pullback),
    Family("ind_rule", "Any-indicator rule", "pool", "One of ~300 causal indicators (every family the app can compute), standardised against its own trailing window: ride the shift, fade the extreme, or follow its turn.",
           {"kind": list(_POOL_KINDS), "period": [7, 14, 21, 34], "win": [100, 250, 500], "k": [1.0, 1.5, 2.0], "mode": ["trend", "revert", "slope"]}, _f_ind_rule,
           "Indicator pool: every causal indicator in the app (base + v7-v13) against the same exits and filters as every other family.", categorical=("kind", "mode")),
    Family("momentum", "Time-series momentum", "momentum", "N-bar return beats a volatility-scaled threshold; ride it.",
           {"n": [10, 20, 40, 80], "thr_atr": [0.5, 1.0, 1.5]}, _f_momo, 'Do Momentum and Reversal Strategies Work in Commodity Futures'),
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
        if len(vals) < 2 or params[k] not in vals or k in fam.categorical:
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
