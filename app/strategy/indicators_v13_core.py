"""v13 core helpers + indicator groups A-F (report 2026-10-25, "Top 100 indicators to add", items 1-61).

  A  volatility / range estimators      1-10     yang_zhang_vol ... downside_dev, ulcer_index
  B  adaptive filters / trend quality   11-24    kalman_*, ehlers_supersmoother, ehlers_itrend, mama/fama, frama, alma, zlema, t3, ...
  C  cycles / spectral                  25-30    hilbert_period, ehlers_sine, cyber_cycle, acf_period, wavelet_*, lowfreq_power_share
  D  randomness / complexity            31-37    dfa_exponent, perm_entropy, sample_entropy, katz_fractal_dim, vr_z, runs_z, ljung_box_q
  E  momentum / oscillators             38-51    tsmom_blend, smi, cmo, ppo, pvo, rvgi, inv_fisher_rsi, wavetrend, force_index_norm, ...
  F  volume / liquidity / microstructure 52-61   rvol_tod, amihud, corwin_schultz, roll_spread, kyle_lambda, bvc_imbalance, vpin, ...

Every builder has the dispatcher signature ``(frame, period, column, lookback) -> Series`` aligned to ``frame.index``.

LOOKAHEAD RULES (checked by tests/backtest/test_indicators_v13.py with a prefix-invariance test on EVERY kind and a behavioural
check on a manual strategy for the highest-priority ones): the value at bar t uses bars <= t only. Recursions run forward;
anything that is "refit" (GARCH level, HMM, k-means) uses data strictly before the refit bar; strided estimators (DFA, sample
entropy, spectral share) compute on the window ending at the last update bar <= t and carry the value forward (stale, never
future). A series that cannot be built returns NaN (a condition on NaN is False) -- never a crash and never a neutral fake.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

try:  # numba is optional (config/requirements.txt lists it); the plain-python fallback is identical, just slower
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    HAVE_NUMBA = False

    def njit(*_a, **_k):  # type: ignore[no-redef]
        def deco(fn):
            return fn
        return deco if not (len(_a) == 1 and callable(_a[0])) else _a[0]

V13Builder = Callable[[pd.DataFrame, int, str, int], pd.Series]

LN2 = float(np.log(2.0))
BUILDERS: dict[str, V13Builder] = {}
DESCRIPTIONS: dict[str, str] = {}
NUMBER: dict[str, int] = {}                      # kind -> the report's indicator number (1..100)
BOUNDED: dict[str, tuple[float, float]] = {}     # exact bounds of the produced series
VALUE_RANGES: dict[str, tuple[float, float]] = {}   # soft practical ranges for comparison thresholds
PERIOD_RANGES: dict[str, tuple[int, int]] = {}
NO_PERIOD: list[str] = []
FLAGS: list[str] = []


def register(kind: str, number: int, desc: str, *, bounded: tuple | None = None, soft: tuple | None = None,
             period: tuple | None = None, no_period: bool = False, flag: bool = False):
    """Decorator: add a builder to the v13 registry with its metadata (single source for every consumer)."""
    def deco(fn: V13Builder) -> V13Builder:
        BUILDERS[kind] = fn
        DESCRIPTIONS[kind] = desc
        NUMBER[kind] = int(number)
        if bounded:
            BOUNDED[kind] = (float(bounded[0]), float(bounded[1]))
        if soft:
            VALUE_RANGES[kind] = (float(soft[0]), float(soft[1]))
        if period:
            PERIOD_RANGES[kind] = (int(period[0]), int(period[1]))
        if no_period:
            NO_PERIOD.append(kind)
        if flag:
            FLAGS.append(kind)
        return fn
    return deco


# -------------------------------------------------------------------------------------------------------- helpers
def nan_like(f: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=f.index, dtype=float)


def clamp_p(period, lo: int, hi: int, default: int) -> int:
    try:
        v = int(period)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v if v > 0 else default))


def safe(fn: V13Builder) -> V13Builder:
    """Wrap a builder: any failure -> an all-NaN series (condition False), the v11/v12 convention."""
    def wrapped(f, p, c, l):
        try:
            from app.strategy.frame_attrs import attrs_context, scalar_attrs
            heavy = dict(getattr(f, "attrs", {}) or {})
            g = f.copy(deep=False)
            g.attrs = scalar_attrs(heavy)             # pandas deep-copies attrs on every combine: keep DataFrames out of them
            with attrs_context(heavy):
                out = fn(g, p, c, l)
            if not isinstance(out, pd.Series):
                out = pd.Series(np.asarray(out, dtype=float), index=f.index)
            out = pd.Series(out.to_numpy(dtype=float), index=f.index)
            return out.replace([np.inf, -np.inf], np.nan)
        except Exception:  # noqa: BLE001
            return nan_like(f)
    wrapped.__name__ = getattr(fn, "__name__", "v13")
    wrapped.__doc__ = fn.__doc__
    return wrapped


def px(f: pd.DataFrame, col: str = "close") -> pd.Series:
    return f[col].astype(float)


def log_ret(f: pd.DataFrame) -> pd.Series:
    c = px(f)
    return np.log(c.where(c > 0)).diff()


def log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.astype(float), b.astype(float)
    ok = (a > 0) & (b > 0)
    return pd.Series(np.where(ok, np.log(a.where(ok, 1.0) / b.where(ok, 1.0)), np.nan), index=a.index)


def vol_col(f: pd.DataFrame) -> pd.Series:
    return f["volume"].astype(float) if "volume" in f.columns else pd.Series(1.0, index=f.index)


def atr_series(f: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l_, c = px(f, "high"), px(f, "low"), px(f)
    tr = pd.concat([h - l_, (h - c.shift(1)).abs(), (l_ - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n // 2)).mean()


def fir(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Causal FIR: out[t] = sum_k w[k] * x[t-k] (w[0] weights the CURRENT bar). NaN until len(w) valid values exist."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    bad = ~np.isfinite(x)
    xz = np.where(bad, 0.0, x)
    out = np.convolve(xz, np.asarray(w, dtype=float), mode="full")[:n]
    nb = np.convolve(bad.astype(float), np.ones(len(w)), mode="full")[:n]
    out[nb > 0] = np.nan
    out[: len(w) - 1] = np.nan
    return out


def iir(b, a, x: np.ndarray, warm: int) -> np.ndarray:
    """Causal IIR filter (scipy lfilter) seeded with the first value; the first ``warm`` outputs are NaN."""
    from scipy.signal import lfilter
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if not ok.any():
        return out
    first = int(np.argmax(ok))
    seg = pd.Series(x[first:]).ffill().to_numpy()
    zi = _steady_zi(b, a, seg[0])
    y, _ = lfilter(b, a, seg, zi=zi)
    out[first:] = y
    out[: first + warm] = np.nan
    return out


def _steady_zi(b, a, x0: float) -> np.ndarray:
    from scipy.signal import lfilter_zi
    try:
        return lfilter_zi(b, a) * x0
    except Exception:  # noqa: BLE001
        return np.zeros(max(len(a), len(b)) - 1)


def strided(n: int, step: int, start: int, fn, out_dtype=float) -> np.ndarray:
    """Evaluate ``fn(t)`` (window ending at bar t, inclusive) at t = start, start+step, ... and carry the value FORWARD."""
    out = np.full(n, np.nan, dtype=out_dtype)
    last = np.nan
    for t in range(start, n):
        if (t - start) % step == 0:
            v = fn(t)
            last = v if np.isfinite(v) else np.nan
        out[t] = last
    return out


def roll_ema(x: pd.Series, n: int) -> pd.Series:
    n = max(int(n), 1)
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def tod_baseline(values: pd.Series, f: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """(mean, std) of ``values`` at the SAME clock minute over the previous ``period`` occurrences (current bar excluded)."""
    from app.strategy.indicators_v11 import _minute_of_day, _ts
    key = _minute_of_day(_ts(f))
    if key.isna().all():
        return nan_like(f), nan_like(f)
    n = max(3, period // 3)
    prior = values.groupby(key, sort=False).shift(1)
    g = prior.groupby(key, sort=False)
    mean = g.transform(lambda s: s.rolling(period, min_periods=n).mean())
    std = g.transform(lambda s: s.rolling(period, min_periods=n).std())
    return mean, std


# --------------------------------------------------------------------------------------------------- A: volatility
@register("yang_zhang_vol", 1, "Yang-Zhang volatility (drift- and gap-robust OHLC estimator), percent per bar, over N bars.",
          soft=(0.0, 5.0), period=(5, 200))
@safe
def yang_zhang_vol(f, p, c, l):
    n = clamp_p(p, 5, 500, 20)
    o, h, lo, cl = px(f, "open"), px(f, "high"), px(f, "low"), px(f)
    ro = log_ratio(o, cl.shift(1))
    rc = log_ratio(cl, o)
    rs = log_ratio(h, cl) * log_ratio(h, o) + log_ratio(lo, cl) * log_ratio(lo, o)
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    var = ro.rolling(n, min_periods=n).var() + k * rc.rolling(n, min_periods=n).var() + (1 - k) * rs.rolling(n, min_periods=n).mean()
    return np.sqrt(var.clip(lower=0.0)) * 100.0


@register("rogers_satchell_vol", 2, "Rogers-Satchell drift-robust range volatility, percent per bar, over N bars.",
          soft=(0.0, 5.0), period=(5, 200))
@safe
def rogers_satchell_vol(f, p, c, l):
    n = clamp_p(p, 2, 500, 20)
    o, h, lo, cl = px(f, "open"), px(f, "high"), px(f, "low"), px(f)
    rs = log_ratio(h, cl) * log_ratio(h, o) + log_ratio(lo, cl) * log_ratio(lo, o)
    return np.sqrt(rs.rolling(n, min_periods=n).mean().clip(lower=0.0)) * 100.0


@register("semivol_up", 3, "Realized upside semi-volatility (percent per bar) over N bars.", soft=(0.0, 5.0), period=(5, 200))
@safe
def semivol_up(f, p, c, l):
    n = clamp_p(p, 3, 500, 20)
    r = log_ret(f)
    return np.sqrt((r.clip(lower=0.0) ** 2).rolling(n, min_periods=n).mean()) * 100.0


@register("semivol_down", 3, "Realized downside semi-volatility (percent per bar) over N bars.", soft=(0.0, 5.0), period=(5, 200))
@safe
def semivol_down(f, p, c, l):
    n = clamp_p(p, 3, 500, 20)
    r = log_ret(f)
    return np.sqrt((r.clip(upper=0.0) ** 2).rolling(n, min_periods=n).mean()) * 100.0


@register("semivar_ratio", 3, "Upside share of realized variance over N bars (0..1); below 0.5 = bad volatility dominates.",
          bounded=(0.0, 1.0), period=(5, 200))
@safe
def semivar_ratio(f, p, c, l):
    n = clamp_p(p, 3, 500, 20)
    r = log_ret(f)
    up = (r.clip(lower=0.0) ** 2).rolling(n, min_periods=n).sum()
    dn = (r.clip(upper=0.0) ** 2).rolling(n, min_periods=n).sum()
    return up / (up + dn).replace(0, np.nan)


@register("bipower_jump_ratio", 4, "Share of realized variance that is jump (RV - bipower variation)/RV over N bars (0..1).",
          bounded=(0.0, 1.0), period=(10, 200))
@safe
def bipower_jump_ratio(f, p, c, l):
    n = clamp_p(p, 5, 500, 30)
    r = log_ret(f)
    rv = (r ** 2).rolling(n, min_periods=n).sum()
    bv = (np.pi / 2.0) * (r.abs() * r.abs().shift(1)).rolling(n, min_periods=n).sum()
    return ((rv - bv) / rv.replace(0, np.nan)).clip(0.0, 1.0)


@register("ewma_vol", 5, "RiskMetrics EWMA volatility, percent per bar (period = span; lambda 0.94 when period <= 0).",
          soft=(0.0, 5.0), period=(5, 200))
@safe
def ewma_vol(f, p, c, l):
    r = log_ret(f)
    try:
        span = int(p)
    except (TypeError, ValueError):
        span = 0
    alpha = (1.0 - 0.94) if span <= 0 else 2.0 / (span + 1.0)
    var = (r ** 2).ewm(alpha=alpha, adjust=False, min_periods=max(5, int(1 / alpha) // 2)).mean()
    return np.sqrt(var) * 100.0


@njit(cache=False)
def _garch_run(r, omega_scale, alpha, beta, warm):
    n = len(r)
    out = np.full(n, np.nan)
    s = 0.0
    cnt = 0
    var = np.nan
    for t in range(n):
        x = r[t]
        if np.isnan(x):
            out[t] = var if not np.isnan(var) else np.nan
            continue
        if cnt >= warm:
            uncond = s / cnt                      # expanding unconditional variance of returns BEFORE this bar
            omega = (1.0 - alpha - beta) * uncond * omega_scale
            if np.isnan(var):
                var = uncond
            var = omega + alpha * x * x + beta * var
            out[t] = var
        s += x * x
        cnt += 1
    return out


@register("garch_vol", 6, "GARCH(1,1) conditional volatility (variance targeting on the expanding past; fixed alpha 0.08 / beta 0.90), percent per bar.",
          soft=(0.0, 5.0), period=(30, 500))
@safe
def garch_vol(f, p, c, l):
    warm = clamp_p(p, 20, 2000, 100)
    r = log_ret(f).to_numpy(dtype=float)
    return pd.Series(np.sqrt(_garch_run(r, 1.0, 0.08, 0.90, warm)) * 100.0, index=f.index)


@register("vol_of_vol", 7, "Volatility of volatility: rolling std of the N-bar volatility divided by its mean (coefficient of variation).",
          soft=(0.0, 3.0), period=(10, 200))
@safe
def vol_of_vol(f, p, c, l):
    n = clamp_p(p, 5, 300, 20)
    v = log_ret(f).rolling(n, min_periods=n).std()
    return v.rolling(n, min_periods=n).std() / v.rolling(n, min_periods=n).mean().replace(0, np.nan)


@register("vol_term_ratio", 8, "Short over long realized volatility (N/4 bars over N bars): >1 expansion, <1 contraction.",
          soft=(0.0, 4.0), period=(8, 300))
@safe
def vol_term_ratio(f, p, c, l):
    n = clamp_p(p, 8, 600, 40)
    r = log_ret(f)
    s = r.rolling(max(2, n // 4), min_periods=max(2, n // 4)).std()
    return s / r.rolling(n, min_periods=n).std().replace(0, np.nan)


@njit(cache=False)
def _rank_in_window(x, win):
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(win - 1, n):
        v = x[t]
        if np.isnan(v):
            continue
        c = 0
        m = 0
        for k in range(t - win + 1, t + 1):
            y = x[k]
            if not np.isnan(y):
                m += 1
                if y <= v:
                    c += 1
        if m >= win // 2:
            out[t] = 100.0 * c / m
    return out


@register("vol_cone_pct", 9, "Percentile (0..100) of the current N-bar volatility within its own trailing 10N-bar history.",
          bounded=(0.0, 100.0), period=(10, 100))
@safe
def vol_cone_pct(f, p, c, l):
    n = clamp_p(p, 5, 200, 20)
    v = log_ret(f).rolling(n, min_periods=n).std().to_numpy(dtype=float)
    return pd.Series(_rank_in_window(v, min(10 * n, 2000)), index=f.index)


@register("downside_dev", 10, "Downside deviation (RMS of negative returns, percent per bar) over N bars.", soft=(0.0, 5.0), period=(5, 200))
@safe
def downside_dev(f, p, c, l):
    n = clamp_p(p, 3, 500, 20)
    r = log_ret(f).clip(upper=0.0)
    return np.sqrt((r ** 2).rolling(n, min_periods=n).mean()) * 100.0


@register("ulcer_index", 10, "Ulcer index of price: RMS percent drawdown from the N-bar high.", soft=(0.0, 10.0), period=(5, 200))
@safe
def ulcer_index(f, p, c, l):
    n = clamp_p(p, 3, 500, 14)
    cl = px(f)
    dd = 100.0 * (cl / cl.rolling(n, min_periods=n).max() - 1.0)
    return np.sqrt((dd ** 2).rolling(n, min_periods=n).mean())


# ---------------------------------------------------------------------------------- B: adaptive filters / trend quality
@njit(cache=False)
def _kalman_llt(y, q_level, q_slope):
    """Local-linear-trend Kalman filter, unit observation noise (the gain is scale free). Returns level, slope, slope se."""
    n = len(y)
    lv = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    se = np.full(n, np.nan)
    started = False
    x0 = 0.0
    x1 = 0.0
    p00 = 1e3
    p01 = 0.0
    p11 = 1e3
    s_innov = 0.0
    seen = 0
    for t in range(n):
        v = y[t]
        if np.isnan(v):
            continue
        if not started:
            x0 = v
            started = True
            lv[t] = x0
            continue
        # predict
        x0p = x0 + x1
        x1p = x1
        p00p = p00 + 2.0 * p01 + p11 + q_level
        p01p = p01 + p11
        p11p = p11 + q_slope
        # update (R = 1)
        s = p00p + 1.0
        k0 = p00p / s
        k1 = p01p / s
        innov = v - x0p
        x0 = x0p + k0 * innov
        x1 = x1p + k1 * innov
        p00 = (1.0 - k0) * p00p
        p01 = (1.0 - k0) * p01p
        p11 = p11p - k1 * p01p
        # innovation scale: EWMA of squared innovation / S (>= 0)
        seen += 1
        a = 0.02 if seen > 50 else 1.0 / seen
        s_innov = (1.0 - a) * s_innov + a * (innov * innov / s)
        lv[t] = x0
        sl[t] = x1
        se[t] = np.sqrt(max(p11, 0.0) * max(s_innov, 1e-18))
    return lv, sl, se


def _kalman(f, p):
    n = clamp_p(p, 5, 300, 30)
    c = px(f).to_numpy(dtype=float)
    return _kalman_llt(c, 1.0 / (n * n), 1.0 / (n ** 4) * 4.0), n


@register("kalman_level", 11, "Kalman local-linear-trend filtered level, expressed as (level - close) / ATR(14): where the adaptive fair value sits.",
          soft=(-5.0, 5.0), period=(5, 300))
@safe
def kalman_level(f, p, c, l):
    (lv, _sl, _se), _n = _kalman(f, p)
    return (pd.Series(lv, index=f.index) - px(f)) / atr_series(f).replace(0, np.nan)


@register("kalman_slope", 11, "Kalman filtered slope as percent of price per bar.", soft=(-0.5, 0.5), period=(5, 300))
@safe
def kalman_slope(f, p, c, l):
    (_lv, sl, _se), _n = _kalman(f, p)
    return pd.Series(sl, index=f.index) / px(f).replace(0, np.nan) * 100.0


@register("kalman_slope_z", 11, "Kalman filtered slope divided by its own standard error: a trend large against its uncertainty.",
          soft=(-6.0, 6.0), period=(5, 300))
@safe
def kalman_slope_z(f, p, c, l):
    (_lv, sl, se), _n = _kalman(f, p)
    s = pd.Series(sl, index=f.index)
    e = pd.Series(se, index=f.index).replace(0, np.nan)
    return (s / e)


@register("ehlers_supersmoother", 12, "Ehlers 2-pole Super Smoother of close, as (smooth - close) / ATR(14).", soft=(-5.0, 5.0), period=(4, 200))
@safe
def ehlers_supersmoother(f, p, c, l):
    n = clamp_p(p, 4, 400, 10)
    a1 = np.exp(-np.sqrt(2.0) * np.pi / n)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2.0) * np.pi / n)
    c2, c3 = b1, -a1 * a1
    c1 = 1.0 - c2 - c3
    x = px(f).to_numpy(dtype=float)
    xin = (x + np.concatenate([[x[0]], x[:-1]])) / 2.0
    y = iir([c1], [1.0, -c2, -c3], xin, warm=2 * n)
    return (pd.Series(y, index=f.index) - px(f)) / atr_series(f).replace(0, np.nan)


@register("ehlers_itrend", 13, "Ehlers Instantaneous Trendline, as (trendline - close) / ATR(14).", soft=(-5.0, 5.0), period=(5, 200))
@safe
def ehlers_itrend(f, p, c, l):
    n = clamp_p(p, 5, 400, 20)
    a = 2.0 / (n + 1.0)
    x = px(f).to_numpy(dtype=float)
    b = [a - a * a / 4.0, 0.5 * a * a, -(a - 0.75 * a * a)]
    den = [1.0, -2.0 * (1.0 - a), (1.0 - a) ** 2]
    y = iir(b, den, x, warm=n)
    return (pd.Series(y, index=f.index) - px(f)) / atr_series(f).replace(0, np.nan)


@njit(cache=False)
def _hilbert(price):
    """Ehlers homodyne discriminator: dominant-cycle period, MAMA, FAMA and the I/Q phase (degrees). Causal recursion."""
    n = len(price)
    period = np.full(n, np.nan)
    mama = np.full(n, np.nan)
    fama = np.full(n, np.nan)
    phase_out = np.full(n, np.nan)
    smooth = np.zeros(n)
    detr = np.zeros(n)
    I1 = np.zeros(n)
    Q1 = np.zeros(n)
    I2 = np.zeros(n)
    Q2 = np.zeros(n)
    Re = np.zeros(n)
    Im = np.zeros(n)
    per = np.zeros(n)
    sper = np.zeros(n)
    phase = np.zeros(n)
    m_prev = np.nan
    f_prev = np.nan
    for t in range(n):
        if t < 7 or np.isnan(price[t]) or np.isnan(price[t - 1]) or np.isnan(price[t - 2]) or np.isnan(price[t - 3]):
            smooth[t] = price[t] if not np.isnan(price[t]) else 0.0
            continue
        smooth[t] = (4.0 * price[t] + 3.0 * price[t - 1] + 2.0 * price[t - 2] + price[t - 3]) / 10.0
        adj = 0.075 * per[t - 1] + 0.54
        detr[t] = (0.0962 * smooth[t] + 0.5769 * smooth[t - 2] - 0.5769 * smooth[t - 4] - 0.0962 * smooth[t - 6]) * adj
        Q1[t] = (0.0962 * detr[t] + 0.5769 * detr[t - 2] - 0.5769 * detr[t - 4] - 0.0962 * detr[t - 6]) * adj
        I1[t] = detr[t - 3]
        jI = (0.0962 * I1[t] + 0.5769 * I1[t - 2] - 0.5769 * I1[t - 4] - 0.0962 * I1[t - 6]) * adj
        jQ = (0.0962 * Q1[t] + 0.5769 * Q1[t - 2] - 0.5769 * Q1[t - 4] - 0.0962 * Q1[t - 6]) * adj
        i2 = I1[t] - jQ
        q2 = Q1[t] + jI
        I2[t] = 0.2 * i2 + 0.8 * I2[t - 1]
        Q2[t] = 0.2 * q2 + 0.8 * Q2[t - 1]
        re = I2[t] * I2[t - 1] + Q2[t] * Q2[t - 1]
        im = I2[t] * Q2[t - 1] - Q2[t] * I2[t - 1]
        Re[t] = 0.2 * re + 0.8 * Re[t - 1]
        Im[t] = 0.2 * im + 0.8 * Im[t - 1]
        pr = per[t - 1]
        if Im[t] != 0.0 and Re[t] != 0.0:
            pr = 360.0 / (np.arctan(Im[t] / Re[t]) * 180.0 / np.pi)
        if per[t - 1] > 0.0:
            if pr > 1.5 * per[t - 1]:
                pr = 1.5 * per[t - 1]
            if pr < 0.67 * per[t - 1]:
                pr = 0.67 * per[t - 1]
        if pr < 6.0:
            pr = 6.0
        if pr > 50.0:
            pr = 50.0
        per[t] = 0.2 * pr + 0.8 * per[t - 1]
        sper[t] = 0.33 * per[t] + 0.67 * sper[t - 1]
        if I1[t] != 0.0:
            phase[t] = np.arctan(Q1[t] / I1[t]) * 180.0 / np.pi
        else:
            phase[t] = phase[t - 1]
        delta = phase[t - 1] - phase[t]
        if delta < 1.0:
            delta = 1.0
        alpha = 0.5 / delta
        if alpha < 0.05:
            alpha = 0.05
        if alpha > 0.5:
            alpha = 0.5
        if np.isnan(m_prev):
            m_prev = price[t]
            f_prev = price[t]
        m_prev = alpha * price[t] + (1.0 - alpha) * m_prev
        f_prev = 0.5 * alpha * m_prev + (1.0 - 0.5 * alpha) * f_prev
        if t >= 40:
            period[t] = sper[t]
            mama[t] = m_prev
            fama[t] = f_prev
            phase_out[t] = phase[t]
    return period, mama, fama, phase_out


def _hilbert_series(f):
    return _hilbert(px(f).to_numpy(dtype=float))


@register("mama_fama_spread", 14, "MESA adaptive MA: (MAMA - FAMA) / ATR(14); the sign is the cycle-adaptive crossover state.",
          soft=(-5.0, 5.0), period=(0, 0), no_period=True)
@safe
def mama_fama_spread(f, p, c, l):
    _per, mama, fama, _ph = _hilbert_series(f)
    return pd.Series(mama - fama, index=f.index) / atr_series(f).replace(0, np.nan)


@register("mama_dist", 14, "Distance of close above MAMA in ATR units.", soft=(-6.0, 6.0), period=(0, 0), no_period=True)
@safe
def mama_dist(f, p, c, l):
    _per, mama, _fama, _ph = _hilbert_series(f)
    return (px(f) - pd.Series(mama, index=f.index)) / atr_series(f).replace(0, np.nan)


@njit(cache=False)
def _frama(price, high, low, n):
    out = np.full(len(price), np.nan)
    half = n // 2
    prev = np.nan
    for t in range(n - 1, len(price)):
        hh1 = -1e300
        ll1 = 1e300
        hh2 = -1e300
        ll2 = 1e300
        for k in range(t - half + 1, t + 1):
            hh1 = max(hh1, high[k])
            ll1 = min(ll1, low[k])
        for k in range(t - n + 1, t - half + 1):
            hh2 = max(hh2, high[k])
            ll2 = min(ll2, low[k])
        hh = max(hh1, hh2)
        ll = min(ll1, ll2)
        n1 = (hh1 - ll1) / half
        n2 = (hh2 - ll2) / half
        n3 = (hh - ll) / n
        if n1 > 0 and n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
        else:
            d = 1.0
        alpha = np.exp(-4.6 * (d - 1.0))
        if alpha < 0.01:
            alpha = 0.01
        if alpha > 1.0:
            alpha = 1.0
        if np.isnan(prev):
            prev = price[t]
        prev = alpha * price[t] + (1.0 - alpha) * prev
        out[t] = prev
    return out


@register("frama_dist", 15, "Fractal Adaptive MA: (close - FRAMA) / ATR(14).", soft=(-8.0, 8.0), period=(8, 200))
@safe
def frama_dist(f, p, c, l):
    n = clamp_p(p, 8, 400, 16)
    n -= n % 2
    fr = _frama(px(f).to_numpy(float), px(f, "high").to_numpy(float), px(f, "low").to_numpy(float), n)
    return (px(f) - pd.Series(fr, index=f.index)) / atr_series(f).replace(0, np.nan)


@register("alma_dist", 16, "ALMA (offset 0.85, sigma 6): (close - ALMA) / ATR(14).", soft=(-8.0, 8.0), period=(5, 200))
@safe
def alma_dist(f, p, c, l):
    n = clamp_p(p, 3, 400, 9)
    m = 0.85 * (n - 1)
    s = n / 6.0
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * s * s))
    w = w / w.sum()
    out = fir(px(f).to_numpy(float), w[::-1])       # w[::-1][0] weights the current bar: ALMA puts most weight near the end
    return (px(f) - pd.Series(out, index=f.index)) / atr_series(f).replace(0, np.nan)


@register("zlema_dist", 17, "Zero-lag EMA: (close - ZLEMA) / ATR(14).", soft=(-8.0, 8.0), period=(5, 200))
@safe
def zlema_dist(f, p, c, l):
    n = clamp_p(p, 3, 400, 20)
    lag = (n - 1) // 2
    x = px(f)
    z = roll_ema(2 * x - x.shift(lag), n)
    return (x - z) / atr_series(f).replace(0, np.nan)


@register("t3_dist", 18, "Tillson T3 (v=0.7): (close - T3) / ATR(14).", soft=(-8.0, 8.0), period=(3, 100))
@safe
def t3_dist(f, p, c, l):
    n = clamp_p(p, 2, 200, 5)
    v = 0.7
    x = px(f)
    e = x
    es = []
    for _ in range(6):
        e = e.ewm(span=n, adjust=False, min_periods=1).mean()
        es.append(e)
    c1 = -v ** 3
    c2 = 3 * v ** 2 + 3 * v ** 3
    c3 = -6 * v ** 2 - 3 * v - 3 * v ** 3
    c4 = 1 + 3 * v + v ** 3 + 3 * v ** 2
    t3 = c1 * es[5] + c2 * es[4] + c3 * es[3] + c4 * es[2]
    t3[: 6 * n] = np.nan
    return (x - t3) / atr_series(f).replace(0, np.nan)


@njit(cache=False)
def _mcginley(price, n):
    out = np.full(len(price), np.nan)
    md = np.nan
    for t in range(len(price)):
        v = price[t]
        if np.isnan(v):
            continue
        if np.isnan(md):
            md = v
        ratio = v / md if md != 0 else 1.0
        den = 0.6 * n * ratio ** 4
        if den < 1.0:
            den = 1.0
        md = md + (v - md) / den
        if t >= n:
            out[t] = md
    return out


@register("mcginley_dist", 19, "McGinley Dynamic: (close - MD) / ATR(14).", soft=(-8.0, 8.0), period=(5, 200))
@safe
def mcginley_dist(f, p, c, l):
    n = clamp_p(p, 3, 400, 14)
    md = _mcginley(px(f).to_numpy(float), n)
    return (px(f) - pd.Series(md, index=f.index)) / atr_series(f).replace(0, np.nan)


@register("gaussian_slope", 20, "Ehlers 2-pole Gaussian filter slope, percent of price per bar.", soft=(-0.5, 0.5), period=(5, 200))
@safe
def gaussian_slope(f, p, c, l):
    n = clamp_p(p, 4, 400, 20)
    beta = (1 - np.cos(2 * np.pi / n)) / (np.sqrt(2.0) - 1.0)
    alpha = -beta + np.sqrt(beta * beta + 2 * beta)
    y = iir([alpha * alpha], [1.0, -2 * (1 - alpha), (1 - alpha) ** 2], px(f).to_numpy(float), warm=3 * n)
    s = pd.Series(y, index=f.index)
    return s.diff() / px(f).replace(0, np.nan) * 100.0


@register("schaff_trend_cycle", 21, "Schaff Trend Cycle (MACD stochastic of stochastic), 0..100.", bounded=(0.0, 100.0), period=(5, 100))
@safe
def schaff_trend_cycle(f, p, c, l):
    cyc = clamp_p(p, 5, 200, 10)
    x = px(f)
    macd = x.ewm(span=23, adjust=False, min_periods=23).mean() - x.ewm(span=50, adjust=False, min_periods=50).mean()

    def stoch(s):
        lo, hi = s.rolling(cyc, min_periods=cyc).min(), s.rolling(cyc, min_periods=cyc).max()
        return 100.0 * (s - lo) / (hi - lo).replace(0, np.nan)

    pf = stoch(macd).ewm(alpha=0.5, adjust=False, ignore_na=True).mean()
    return stoch(pf).ewm(alpha=0.5, adjust=False, ignore_na=True).mean().clip(0.0, 100.0)


@register("vhf", 22, "Vertical Horizontal Filter: trend versus congestion, (max close - min close) / sum |dclose| over N bars.",
          bounded=(0.0, 1.0), period=(5, 200))
@safe
def vhf(f, p, c, l):
    n = clamp_p(p, 3, 400, 28)
    x = px(f)
    num = x.rolling(n, min_periods=n).max() - x.rolling(n, min_periods=n).min()
    den = x.diff().abs().rolling(n, min_periods=n).sum()
    return (num / den.replace(0, np.nan)).clip(0.0, 1.0)


@njit(cache=False)
def _mann_kendall(x, n):
    out = np.full(len(x), np.nan)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    for t in range(n - 1, len(x)):
        s = 0.0
        ok = True
        for i in range(t - n + 1, t):
            if np.isnan(x[i]):
                ok = False
                break
            for j in range(i + 1, t + 1):
                d = x[j] - x[i]
                if d > 0:
                    s += 1.0
                elif d < 0:
                    s -= 1.0
        if not ok or np.isnan(x[t]):
            continue
        if s > 0:
            out[t] = (s - 1.0) / np.sqrt(var_s)
        elif s < 0:
            out[t] = (s + 1.0) / np.sqrt(var_s)
        else:
            out[t] = 0.0
    return out


@register("mann_kendall_z", 23, "Mann-Kendall trend statistic (z) of close over N bars: non-parametric trend significance.",
          soft=(-4.0, 4.0), period=(8, 100))
@safe
def mann_kendall_z(f, p, c, l):
    n = clamp_p(p, 8, 150, 20)
    return pd.Series(_mann_kendall(px(f).to_numpy(float), n), index=f.index)


@register("slope_tstat", 24, "t-statistic of the OLS slope of close on time over N bars: trend significance, not just slope.",
          soft=(-8.0, 8.0), period=(8, 200))
@safe
def slope_tstat(f, p, c, l):
    n = clamp_p(p, 5, 400, 20)
    y = px(f)
    xs = np.arange(n, dtype=float)
    xm = xs.mean()
    sxx = ((xs - xm) ** 2).sum()
    sy = y.rolling(n, min_periods=n).sum()
    syy = (y ** 2).rolling(n, min_periods=n).sum()
    sxy = pd.Series(fir(y.to_numpy(float), (xs)[::-1]), index=y.index)       # sum_k x_k * y[t-(n-1-k)] ; x_{n-1} weights the current bar
    cov = sxy - xm * sy
    slope = cov / sxx
    sst = syy - sy ** 2 / n
    sse = (sst - slope * cov).clip(lower=0.0)
    se = np.sqrt(sse / max(n - 2, 1) / sxx)
    return slope / se.replace(0, np.nan)


# --------------------------------------------------------------------------------------------- C: cycles / spectral
@register("hilbert_period", 25, "Hilbert-transform (homodyne) dominant cycle period in bars (6..50): adaptive lookback source.",
          bounded=(6.0, 50.0), period=(0, 0), no_period=True)
@safe
def hilbert_period(f, p, c, l):
    per, _m, _f, _ph = _hilbert_series(f)
    return pd.Series(per, index=f.index).clip(6.0, 50.0)


def _dc_phase(f):
    """Ehlers DC phase from the dominant-cycle window (degrees), causal."""
    per, _m, _fm, _ph = _hilbert_series(f)
    price = px(f).to_numpy(float)
    return _dc_phase_nb(price, np.nan_to_num(per, nan=0.0))


@njit(cache=False)
def _dc_phase_nb(price, per):
    n = len(price)
    out = np.full(n, np.nan)
    prev = 0.0
    for t in range(n):
        if per[t] < 6.0 or t < 60:
            continue
        w = int(per[t] + 0.5)
        if w < 6:
            w = 6
        if w > 50:
            w = 50
        re = 0.0
        im = 0.0
        for k in range(w):
            ang = 2.0 * np.pi * k / w
            v = price[t - k]
            re += np.sin(ang) * v
            im += np.cos(ang) * v
        if abs(im) > 1e-12 * max(1.0, abs(re)):
            ph = np.arctan(re / im) * 180.0 / np.pi
        else:
            ph = 90.0 if re > 0 else -90.0
        ph += 90.0
        ph += 360.0 / w
        if im < 0:
            ph += 180.0
        if ph > 315.0:
            ph -= 360.0
        out[t] = ph
        prev = ph
    return out


@register("ehlers_sine", 26, "Ehlers sinewave of the dominant-cycle phase (-1..1): cycle-mode turning points.", bounded=(-1.0, 1.0), no_period=True)
@safe
def ehlers_sine(f, p, c, l):
    return pd.Series(np.sin(np.radians(_dc_phase(f))), index=f.index)


@register("ehlers_sine_lead", 26, "Ehlers lead sine (phase + 45 degrees); crossing ehlers_sine marks cycle turns.", bounded=(-1.0, 1.0), no_period=True)
@safe
def ehlers_sine_lead(f, p, c, l):
    return pd.Series(np.sin(np.radians(_dc_phase(f) + 45.0)), index=f.index)


@register("cyber_cycle", 27, "Ehlers Cyber Cycle (band-pass cycle component) in ATR units; period sets alpha = 2/(N+1).",
          soft=(-3.0, 3.0), period=(5, 100))
@safe
def cyber_cycle(f, p, c, l):
    n = clamp_p(p, 4, 200, 14)
    a = 2.0 / (n + 1.0)
    x = px(f)
    sm = ((x + 2 * x.shift(1) + 2 * x.shift(2) + x.shift(3)) / 6.0).to_numpy(float)
    k = (1 - 0.5 * a) ** 2
    y = iir([k, -2 * k, k], [1.0, -2 * (1 - a), (1 - a) ** 2], sm, warm=3 * n)
    return pd.Series(y, index=f.index) / atr_series(f).replace(0, np.nan)


@njit(cache=False)
def _acf_period(x, window, step, lo, hi):
    n = len(x)
    out = np.full(n, np.nan)
    last = np.nan
    for t in range(window + hi, n):
        if (t - (window + hi)) % step == 0:
            best = -2.0
            bestlag = np.nan
            num_ok = True
            for lag in range(lo, hi + 1):
                sx = 0.0
                sy = 0.0
                sxx = 0.0
                syy = 0.0
                sxy = 0.0
                for k in range(window):
                    a = x[t - k]
                    b = x[t - k - lag]
                    if np.isnan(a) or np.isnan(b):
                        num_ok = False
                    sx += a
                    sy += b
                    sxx += a * a
                    syy += b * b
                    sxy += a * b
                dx = window * sxx - sx * sx
                dy = window * syy - sy * sy
                if dx <= 0 or dy <= 0:
                    continue
                r = (window * sxy - sx * sy) / np.sqrt(dx * dy)
                if r > best:
                    best = r
                    bestlag = lag
            if num_ok and best > -2.0:
                last = bestlag
            else:
                last = np.nan
        out[t] = last
    return out


@register("acf_period", 28, "Autocorrelation-periodogram dominant period in bars (10..48), roofing-filtered, updated every 4 bars.",
          bounded=(10.0, 48.0), period=(0, 0), no_period=True)
@safe
def acf_period(f, p, c, l):
    x = px(f)
    hp = (x - x.ewm(span=48, adjust=False).mean()).to_numpy(float)          # high-pass (roofing): removes the trend component
    ss = iir([0.2, 0.2], [1.0, -0.6], hp, warm=10)
    return pd.Series(_acf_period(ss, 48, 4, 10, 48), index=f.index)


def _haar_energy(r: np.ndarray, scales: int, win: int) -> np.ndarray:
    """Causal Haar-MODWT detail energy per scale: shape (scales, n)."""
    n = len(r)
    out = np.full((scales, n), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(r))])
    for j in range(1, scales + 1):
        half = 2 ** (j - 1)
        idx = np.arange(n)
        a = cs[idx + 1] - cs[np.maximum(idx + 1 - half, 0)]
        b = cs[np.maximum(idx + 1 - half, 0)] - cs[np.maximum(idx + 1 - 2 * half, 0)]
        w = (a - b) / (2.0 * half)
        w[: 2 * half - 1] = np.nan
        e = pd.Series(w * w).rolling(win, min_periods=win).sum().to_numpy()
        out[j - 1] = e
    return out


@register("wavelet_dominant_scale", 29, "Haar-MODWT scale (1..6; scale j spans 2^j bars) carrying the most return energy over the last N bars.",
          bounded=(1.0, 6.0), period=(32, 400))
@safe
def wavelet_dominant_scale(f, p, c, l):
    win = clamp_p(p, 16, 1000, 64)
    r = log_ret(f).to_numpy(float)
    e = _haar_energy(r, 6, win)
    ok = np.isfinite(e).all(axis=0)
    out = np.full(len(r), np.nan)
    out[ok] = np.argmax(e[:, ok], axis=0) + 1
    return pd.Series(out, index=f.index)


@register("wavelet_low_share", 29, "Share (0..1) of Haar-MODWT return energy in scales 4-6 (16-64 bar horizons) over the last N bars.",
          bounded=(0.0, 1.0), period=(32, 400))
@safe
def wavelet_low_share(f, p, c, l):
    win = clamp_p(p, 16, 1000, 64)
    r = log_ret(f).to_numpy(float)
    e = _haar_energy(r, 6, win)
    tot = e.sum(axis=0)
    return pd.Series(np.where(tot > 0, e[3:].sum(axis=0) / np.where(tot > 0, tot, 1.0), np.nan), index=f.index)


@register("lowfreq_power_share", 30, "Share (0..1) of the window's spectral power in periods >= N/4 bars (window N, refreshed every N/8 bars): trending vs noisy.",
          bounded=(0.0, 1.0), period=(32, 400))
@safe
def lowfreq_power_share(f, p, c, l):
    n = clamp_p(p, 32, 1000, 128)
    r = log_ret(f).to_numpy(float)
    step = max(1, n // 8)

    def at(t):
        w = r[t - n + 1: t + 1]
        if not np.isfinite(w).all():
            return np.nan
        sp = np.abs(np.fft.rfft(w - w.mean())) ** 2
        tot = sp[1:].sum()
        return sp[1:5].sum() / tot if tot > 0 else np.nan

    return pd.Series(strided(len(r), step, n, at), index=f.index)


# ----------------------------------------------------------------------------------------- D: randomness / complexity
@njit(cache=False)
def _dfa_alpha(x, scales):
    """DFA-1 exponent of the integrated, demeaned window."""
    n = len(x)
    m = x.mean()
    y = np.cumsum(x - m)
    logs = np.zeros(len(scales))
    logf = np.zeros(len(scales))
    cnt = 0
    for si in range(len(scales)):
        s = scales[si]
        nseg = n // s
        if nseg < 2:
            continue
        acc = 0.0
        for g in range(nseg):
            lo = g * s
            sx = 0.0
            sy = 0.0
            sxx = 0.0
            sxy = 0.0
            for k in range(s):
                sx += k
                sy += y[lo + k]
                sxx += k * k
                sxy += k * y[lo + k]
            den = s * sxx - sx * sx
            b = (s * sxy - sx * sy) / den
            a = (sy - b * sx) / s
            for k in range(s):
                e = y[lo + k] - (a + b * k)
                acc += e * e
        fl = np.sqrt(acc / (nseg * s))
        if fl > 0:
            logs[cnt] = np.log(s)
            logf[cnt] = np.log(fl)
            cnt += 1
    if cnt < 3:
        return np.nan
    mx = 0.0
    my = 0.0
    for i in range(cnt):
        mx += logs[i]
        my += logf[i]
    mx /= cnt
    my /= cnt
    num = 0.0
    den = 0.0
    for i in range(cnt):
        num += (logs[i] - mx) * (logf[i] - my)
        den += (logs[i] - mx) ** 2
    return num / den if den > 0 else np.nan


@register("dfa_exponent", 31, "Detrended-fluctuation exponent of returns over N bars (refreshed every N/8 bars): 0.5 random, >0.5 persistent, <0.5 anti-persistent.",
          soft=(0.2, 1.2), period=(64, 1000))
@safe
def dfa_exponent(f, p, c, l):
    n = clamp_p(p, 64, 2000, 256)
    r = log_ret(f).to_numpy(float)
    scales = np.array([s for s in (4, 8, 16, 32, 64) if s <= n // 4], dtype=np.int64)
    step = max(1, n // 8)

    def at(t):
        w = r[t - n + 1: t + 1]
        if not np.isfinite(w).all() or np.std(w) == 0:
            return np.nan
        return float(_dfa_alpha(w, scales))

    return pd.Series(strided(len(r), step, n, at), index=f.index)


@register("perm_entropy", 32, "Normalized permutation entropy (order 3) of close over N bars (0..1): low = structure, high = noise.",
          bounded=(0.0, 1.0), period=(20, 500))
@safe
def perm_entropy(f, p, c, l):
    n = clamp_p(p, 20, 1000, 60)
    x = px(f)
    a, b, d = x, x.shift(1), x.shift(2)
    code = ((a > b).astype(int) * 4 + (b > d).astype(int) * 2 + (a > d).astype(int)).where(d.notna() & b.notna())
    ent = pd.Series(0.0, index=f.index)
    cnt_ok = code.notna().astype(float).rolling(n, min_periods=n).sum()
    for k in range(8):
        pk = (code == k).astype(float).rolling(n, min_periods=n).sum() / cnt_ok.replace(0, np.nan)
        ent = ent - (pk * np.log(pk.where(pk > 0, 1.0)))
    return (ent / np.log(6.0)).where(cnt_ok >= n).clip(0.0, 1.0)


@njit(cache=False)
def _sampen(x, m, r):
    n = len(x)
    a = 0
    b = 0
    for i in range(n - m):
        for j in range(i + 1, n - m):
            d = 0.0
            for k in range(m):
                dd = abs(x[i + k] - x[j + k])
                if dd > d:
                    d = dd
            if d <= r:
                b += 1
                if abs(x[i + m] - x[j + m]) <= r:
                    a += 1
    if a == 0 or b == 0:
        return np.nan
    return -np.log(a / b)


@register("sample_entropy", 33, "Sample entropy (m=2, r=0.2 std) of returns over N bars, refreshed every N/5 bars: regularity of recent returns.",
          soft=(0.0, 3.0), period=(30, 200))
@safe
def sample_entropy(f, p, c, l):
    n = clamp_p(p, 30, 300, 60)
    r = log_ret(f).to_numpy(float)
    step = max(1, n // 5)

    def at(t):
        w = r[t - n + 1: t + 1]
        if not np.isfinite(w).all():
            return np.nan
        s = np.std(w)
        return float(_sampen(w, 2, 0.2 * s)) if s > 0 else np.nan

    return pd.Series(strided(len(r), step, n, at), index=f.index)


@register("katz_fractal_dim", 34, "Katz fractal dimension of close over N bars (1..2): roughness of the price path (2 = noisy, 1 = straight).",
          bounded=(1.0, 2.0), period=(8, 200))
@safe
def katz_fractal_dim(f, p, c, l):
    n = clamp_p(p, 5, 400, 30)
    x = px(f)
    length = x.diff().abs().rolling(n - 1, min_periods=n - 1).sum()
    diam = x.rolling(n, min_periods=n).max() - x.rolling(n, min_periods=n).min()
    steps = n - 1
    d = np.log10(steps) / (np.log10(steps) + np.log10((diam / length.replace(0, np.nan)).clip(lower=1e-12)))
    return d.clip(1.0, 2.0)


@register("vr_z", 35, "Lo-MacKinlay heteroskedasticity-robust variance-ratio z for lag q = period (2..30), window max(10q, 100), zero-mean returns.",
          soft=(-6.0, 6.0), period=(2, 30))
@safe
def vr_z(f, p, c, l):
    q = clamp_p(p, 2, 30, 4)
    return _vr_z(f, q)


def _vr_z(f, q: int) -> pd.Series:
    win = max(10 * q, 100)
    r = log_ret(f)
    r2 = r ** 2
    s1 = r2.rolling(win, min_periods=win).sum()
    rq = r.rolling(q, min_periods=q).sum()
    vr = (rq ** 2).rolling(win, min_periods=win).sum() / (q * s1.replace(0, np.nan))
    theta = pd.Series(0.0, index=f.index)
    for j in range(1, q):
        dj = (r2 * r2.shift(j)).rolling(win, min_periods=win).sum() * win / (s1 ** 2).replace(0, np.nan)
        theta = theta + (2.0 * (q - j) / q) ** 2 * dj
    return (vr - 1.0) * np.sqrt(win) / np.sqrt(theta.replace(0, np.nan))


@register("vr_profile_z", 35, "Mean heteroskedasticity-robust variance-ratio z across lags 2, 4, 8 and 16: >0 trending, <0 mean-reverting.",
          soft=(-6.0, 6.0), no_period=True)
@safe
def vr_profile_z(f, p, c, l):
    return pd.concat([_vr_z(f, q) for q in (2, 4, 8, 16)], axis=1).mean(axis=1, skipna=False)


@register("runs_z", 36, "Wald-Wolfowitz runs-test z of return signs over N bars: <0 streaky (trending), >0 alternating.",
          soft=(-4.0, 4.0), period=(20, 400))
@safe
def runs_z(f, p, c, l):
    n = clamp_p(p, 20, 600, 60)
    r = log_ret(f)
    s = np.sign(r).where(r != 0)
    pos = (s > 0).astype(float).where(s.notna())
    npos = pos.rolling(n, min_periods=n).sum()
    nneg = (1.0 - pos).rolling(n, min_periods=n).sum()
    chg = (s != s.shift(1)).astype(float).where(s.notna() & s.shift(1).notna())
    runs = 1.0 + chg.rolling(n - 1, min_periods=n - 1).sum()
    nn = npos + nneg
    mu = 2 * npos * nneg / nn.replace(0, np.nan) + 1.0
    var = 2 * npos * nneg * (2 * npos * nneg - nn) / ((nn ** 2) * (nn - 1)).replace(0, np.nan)
    return (runs - mu) / np.sqrt(var.replace(0, np.nan))


@register("ljung_box_q", 37, "Ljung-Box Q over return autocorrelations at lags 1-10 on a window of N bars (chi-square 10 df; 18.3 is the 5% level).",
          soft=(0.0, 60.0), period=(50, 500))
@safe
def ljung_box_q(f, p, c, l):
    n = clamp_p(p, 50, 1000, 100)
    r = log_ret(f)
    den = (r ** 2).rolling(n, min_periods=n).sum().replace(0, np.nan)
    q = pd.Series(0.0, index=f.index)
    for k in range(1, 11):
        rho = (r * r.shift(k)).rolling(n, min_periods=n).sum() / den
        q = q + rho ** 2 / (n - k)
    return n * (n + 2) * q


# -------------------------------------------------------------------------------------- E: momentum / oscillators
@register("tsmom_blend", 38, "Vol-normalized multi-horizon time-series momentum: mean of ret_h / (ewma vol * sqrt(h)) for h = N, 3N, 12N.",
          soft=(-3.0, 3.0), period=(5, 120))
@safe
def tsmom_blend(f, p, c, l):
    n = clamp_p(p, 3, 250, 21)
    r = log_ret(f)
    vol = np.sqrt((r ** 2).ewm(span=max(20, n), adjust=False, min_periods=20).mean())
    parts = []
    for h in (n, 3 * n, 12 * n):
        parts.append(r.rolling(h, min_periods=h).sum() / (vol * np.sqrt(h)).replace(0, np.nan))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False)


@register("smi", 39, "Stochastic Momentum Index (double-smoothed distance from range midpoint), -100..100.", bounded=(-100.0, 100.0), period=(5, 100))
@safe
def smi(f, p, c, l):
    n = clamp_p(p, 3, 200, 10)
    s = max(2, n // 3)
    hh, ll = px(f, "high").rolling(n, min_periods=n).max(), px(f, "low").rolling(n, min_periods=n).min()
    d = px(f) - (hh + ll) / 2.0
    rng = hh - ll
    num = d.ewm(span=s, adjust=False).mean().ewm(span=s, adjust=False).mean()
    den = (rng / 2.0).ewm(span=s, adjust=False).mean().ewm(span=s, adjust=False).mean()
    return (100.0 * num / den.replace(0, np.nan)).clip(-100.0, 100.0)


@register("cmo", 40, "Chande Momentum Oscillator, -100..100.", bounded=(-100.0, 100.0), period=(3, 100))
@safe
def cmo(f, p, c, l):
    n = clamp_p(p, 3, 300, 14)
    d = px(f).diff()
    up = d.clip(lower=0.0).rolling(n, min_periods=n).sum()
    dn = (-d).clip(lower=0.0).rolling(n, min_periods=n).sum()
    return (100.0 * (up - dn) / (up + dn).replace(0, np.nan)).clip(-100.0, 100.0)


def _ppo(x: pd.Series, fast: int) -> pd.Series:
    slow = max(fast + 1, int(round(fast * 26.0 / 12.0)))
    ef, es = roll_ema(x, fast), roll_ema(x, slow)
    return 100.0 * (ef - es) / es.replace(0, np.nan)


@register("ppo", 41, "Percentage Price Oscillator (fast = N, slow = 26/12 N), percent.", soft=(-5.0, 5.0), period=(3, 100))
@safe
def ppo(f, p, c, l):
    return _ppo(px(f), clamp_p(p, 2, 200, 12))


@register("pvo", 41, "Percentage Volume Oscillator (fast = N, slow = 26/12 N), percent.", soft=(-100.0, 200.0), period=(3, 100))
@safe
def pvo(f, p, c, l):
    return _ppo(vol_col(f), clamp_p(p, 2, 200, 12))


@register("rvgi", 42, "Relative Vigor Index (close-open vs high-low conviction), -1..1.", bounded=(-1.0, 1.0), period=(3, 100))
@safe
def rvgi(f, p, c, l):
    n = clamp_p(p, 3, 200, 10)
    w = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
    num = pd.Series(fir((px(f) - px(f, "open")).to_numpy(float), w), index=f.index)
    den = pd.Series(fir((px(f, "high") - px(f, "low")).to_numpy(float), w), index=f.index)
    return (num.rolling(n, min_periods=n).sum() / den.rolling(n, min_periods=n).sum().replace(0, np.nan)).clip(-1.0, 1.0)


@register("inv_fisher_rsi", 43, "Inverse Fisher transform of smoothed RSI, -1..1.", bounded=(-1.0, 1.0), period=(3, 100))
@safe
def inv_fisher_rsi(f, p, c, l):
    from app.strategy.indicators import rsi
    n = clamp_p(p, 3, 200, 5)
    v = 0.1 * (rsi(px(f), n) - 50.0)
    w = np.arange(1, 10, dtype=float)
    vs = pd.Series(fir(v.to_numpy(float), w[::-1] / w.sum()), index=f.index)
    e = np.exp(2.0 * vs.clip(-20, 20))
    return ((e - 1.0) / (e + 1.0)).clip(-1.0, 1.0)


@register("wavetrend", 44, "WaveTrend oscillator (channel index smoothed), typically -100..100 with reversals beyond +-60.",
          soft=(-150.0, 150.0), period=(5, 100))
@safe
def wavetrend(f, p, c, l):
    n = clamp_p(p, 3, 200, 10)
    ap = (px(f, "high") + px(f, "low") + px(f)) / 3.0
    esa = roll_ema(ap, n)
    d = roll_ema((ap - esa).abs(), n)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    return roll_ema(ci, 21)


@register("force_index_norm", 45, "Elder Force Index (EMA of dclose*volume) divided by the EMA of |dclose|*volume: -1..1.",
          bounded=(-1.0, 1.0), period=(2, 100))
@safe
def force_index_norm(f, p, c, l):
    n = clamp_p(p, 2, 200, 13)
    d = px(f).diff()
    v = vol_col(f)
    return (roll_ema(d * v, n) / roll_ema(d.abs() * v, n).replace(0, np.nan)).clip(-1.0, 1.0)


@register("elder_impulse", 45, "Elder Impulse system state: +1 when EMA(N) and MACD histogram both rise, -1 when both fall, else 0.",
          bounded=(-1.0, 1.0), period=(5, 100))
@safe
def elder_impulse(f, p, c, l):
    n = clamp_p(p, 3, 200, 13)
    x = px(f)
    e = roll_ema(x, n)
    macd = roll_ema(x, 12) - roll_ema(x, 26)
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    up = (e.diff() > 0) & (hist.diff() > 0)
    dn = (e.diff() < 0) & (hist.diff() < 0)
    ok = e.diff().notna() & hist.diff().notna() & (x.expanding().count() > 40)
    return pd.Series(np.where(up, 1.0, np.where(dn, -1.0, 0.0)), index=f.index).where(ok)


@register("balance_of_power", 46, "Balance of Power: SMA of (close-open)/(high-low), -1..1.", bounded=(-1.0, 1.0), period=(2, 100))
@safe
def balance_of_power(f, p, c, l):
    n = clamp_p(p, 1, 200, 14)
    rng = (px(f, "high") - px(f, "low")).replace(0, np.nan)
    return ((px(f) - px(f, "open")) / rng).rolling(n, min_periods=n).mean().clip(-1.0, 1.0)


@register("ease_of_movement", 47, "Ease of Movement (SMA of midpoint move per unit volume/range), scaled by its own 5N-bar std: z-like.",
          soft=(-4.0, 4.0), period=(5, 100))
@safe
def ease_of_movement(f, p, c, l):
    n = clamp_p(p, 3, 200, 14)
    h, lo = px(f, "high"), px(f, "low")
    mid = (h + lo) / 2.0
    box = (vol_col(f) / (h - lo).replace(0, np.nan))
    emv = (mid - mid.shift(1)) / box.replace(0, np.nan)
    sm = emv.rolling(n, min_periods=n).mean()
    return sm / emv.rolling(5 * n, min_periods=5 * n).std().replace(0, np.nan)


@register("klinger_osc", 48, "Klinger Volume Oscillator (EMA34-EMA55 of volume force) divided by 100x the 55-bar EMA of volume.",
          soft=(-3.0, 3.0), period=(0, 0), no_period=True)
@safe
def klinger_osc(f, p, c, l):
    h, lo, cl = px(f, "high"), px(f, "low"), px(f)
    hlc = h + lo + cl
    trend = np.sign(hlc.diff()).replace(0, np.nan).ffill()
    dm = h - lo
    run = (trend != trend.shift(1)).cumsum()
    cm = dm.groupby(run).cumsum() + dm.shift(1).groupby(run).transform("first")
    vf = vol_col(f) * (2.0 * (dm / cm.replace(0, np.nan) - 1.0)).abs() * trend * 100.0
    kvo = roll_ema(vf, 34) - roll_ema(vf, 55)
    return kvo / (100.0 * roll_ema(vol_col(f), 55)).replace(0, np.nan)


def _smma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


@register("alligator_spread", 49, "Williams Alligator (lips - jaw) / ATR(14): awake trend (large |.|) versus sleeping (near 0).",
          soft=(-8.0, 8.0), period=(5, 60))
@safe
def alligator_spread(f, p, c, l):
    scale = clamp_p(p, 5, 100, 13) / 13.0
    med = (px(f, "high") + px(f, "low")) / 2.0
    jaw = _smma(med, max(3, round(13 * scale))).shift(max(1, round(8 * scale)))
    lips = _smma(med, max(2, round(5 * scale))).shift(max(1, round(3 * scale)))
    return (lips - jaw) / atr_series(f).replace(0, np.nan)


@njit(cache=False)
def _td(close, low, high):
    n = len(close)
    setup = np.zeros(n)
    cd = np.zeros(n)
    bs = 0
    ss = 0
    b_on = False
    s_on = False
    bcount = 0
    scount = 0
    for t in range(n):
        if t >= 4:
            bs = bs + 1 if close[t] < close[t - 4] else 0
            ss = ss + 1 if close[t] > close[t - 4] else 0
            if bs > 9:
                bs = 1 if close[t] < close[t - 4] else 0
            if ss > 9:
                ss = 1 if close[t] > close[t - 4] else 0
            if bs == 9:
                b_on = True
                bcount = 0
                s_on = False
                scount = 0
            if ss == 9:
                s_on = True
                scount = 0
                b_on = False
                bcount = 0
        if b_on and t >= 2 and close[t] <= low[t - 2]:
            bcount += 1
        if s_on and t >= 2 and close[t] >= high[t - 2]:
            scount += 1
        setup[t] = bs if bs > 0 else -ss
        if b_on:
            cd[t] = bcount
            if bcount >= 13:
                b_on = False
                bcount = 0
        elif s_on:
            cd[t] = -scount
            if scount >= 13:
                s_on = False
                scount = 0
    return setup, cd


@register("td_setup", 50, "TD Sequential setup count: +1..+9 for consecutive closes below the close 4 bars earlier (buy setup), -1..-9 for sell setups.",
          bounded=(-9.0, 9.0), no_period=True)
@safe
def td_setup(f, p, c, l):
    s, _cd = _td(px(f).to_numpy(float), px(f, "low").to_numpy(float), px(f, "high").to_numpy(float))
    return pd.Series(s, index=f.index)


@register("td_countdown", 50, "TD Sequential countdown after a completed setup: +1..+13 (buy) / -1..-13 (sell), 0 when not counting.",
          bounded=(-13.0, 13.0), no_period=True)
@safe
def td_countdown(f, p, c, l):
    _s, cd = _td(px(f).to_numpy(float), px(f, "low").to_numpy(float), px(f, "high").to_numpy(float))
    return pd.Series(cd, index=f.index)


@register("mass_index", 51, "Mass Index: sum over N bars of EMA9(range)/EMA9(EMA9(range)); a bulge above 27 then below 26.5 flags reversals.",
          soft=(5.0, 40.0), period=(10, 60))
@safe
def mass_index(f, p, c, l):
    n = clamp_p(p, 5, 100, 25)
    rng = px(f, "high") - px(f, "low")
    e1 = roll_ema(rng, 9)
    e2 = roll_ema(e1, 9)
    return (e1 / e2.replace(0, np.nan)).rolling(n, min_periods=n).sum()


# --------------------------------------------------------------------- F: volume / liquidity / microstructure (OHLCV)
@register("rvol_tod", 52, "Volume against the mean volume of the SAME clock minute over the previous N sessions (1.0 = typical for that time of day).",
          soft=(0.0, 10.0), period=(5, 60))
@safe
def rvol_tod(f, p, c, l):
    n = clamp_p(p, 5, 120, 20)
    mean, _ = tod_baseline(vol_col(f), f, n)
    return vol_col(f) / mean.replace(0, np.nan)


@register("amihud", 53, "Amihud illiquidity: mean(|return| / dollar volume) over N bars, x1e9 (rises when price moves a lot on little volume).",
          soft=(0.0, 1000.0), period=(5, 200))
@safe
def amihud(f, p, c, l):
    n = clamp_p(p, 3, 400, 20)
    dv = (px(f) * vol_col(f)).replace(0, np.nan)
    return (log_ret(f).abs() / dv).rolling(n, min_periods=n).mean() * 1e9


@register("corwin_schultz", 54, "Corwin-Schultz high-low spread estimate in basis points, mean over N bars (negatives floored at 0).",
          soft=(0.0, 100.0), period=(5, 200))
@safe
def corwin_schultz(f, p, c, l):
    n = clamp_p(p, 3, 400, 20)
    h, lo = px(f, "high"), px(f, "low")
    beta = (log_ratio(h, lo) ** 2) + (log_ratio(h.shift(1), lo.shift(1)) ** 2)
    gamma = log_ratio(pd.concat([h, h.shift(1)], axis=1).max(axis=1), pd.concat([lo, lo.shift(1)], axis=1).min(axis=1)) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return s.clip(lower=0.0).rolling(n, min_periods=n).mean() * 1e4


@register("roll_spread", 55, "Roll (1984) spread estimate from the serial covariance of price changes, basis points of price, over N bars.",
          soft=(0.0, 100.0), period=(10, 300))
@safe
def roll_spread(f, p, c, l):
    n = clamp_p(p, 10, 600, 30)
    d = px(f).diff()
    cov = d.rolling(n, min_periods=n).cov(d.shift(1))
    return 2.0 * np.sqrt((-cov).clip(lower=0.0)) / px(f).replace(0, np.nan) * 1e4


@register("kyle_lambda", 56, "Kyle's lambda: rolling slope of the return on signed volume, scaled to the return impact of an average-size bar (bps).",
          soft=(-50.0, 50.0), period=(20, 500))
@safe
def kyle_lambda(f, p, c, l):
    n = clamp_p(p, 10, 800, 60)
    v = vol_col(f)
    sv = np.sign(px(f).diff()) * v
    r = log_ret(f)
    slope = r.rolling(n, min_periods=n).cov(sv) / sv.rolling(n, min_periods=n).var().replace(0, np.nan)
    return slope * v.rolling(n, min_periods=n).mean() * 1e4


def _bvc_buy_frac(f, n: int) -> pd.Series:
    from scipy.special import ndtr
    d = px(f).diff()
    sd = d.rolling(max(20, n), min_periods=20).std()
    return pd.Series(ndtr((d / sd.replace(0, np.nan)).to_numpy(float)), index=f.index)


@register("bvc_imbalance", 57, "Bulk Volume Classification imbalance over N bars: sum(V*(2*Phi(dP/sigma)-1))/sum(V), -1..1.",
          bounded=(-1.0, 1.0), period=(5, 300))
@safe
def bvc_imbalance(f, p, c, l):
    n = clamp_p(p, 3, 600, 20)
    v = vol_col(f)
    buy = _bvc_buy_frac(f, n)
    return ((v * (2.0 * buy - 1.0)).rolling(n, min_periods=n).sum() / v.rolling(n, min_periods=n).sum().replace(0, np.nan)).clip(-1.0, 1.0)


@register("vpin", 58, "VPIN order-flow toxicity (bar-bucket version): sum|V_buy - V_sell| / sum V over N bars, 0..1.",
          bounded=(0.0, 1.0), period=(5, 300))
@safe
def vpin(f, p, c, l):
    n = clamp_p(p, 3, 600, 50)
    v = vol_col(f)
    buy = _bvc_buy_frac(f, n)
    return ((v * (2.0 * buy - 1.0)).abs().rolling(n, min_periods=n).sum() / v.rolling(n, min_periods=n).sum().replace(0, np.nan)).clip(0.0, 1.0)


@register("volume_clock_return", 59, "Return over the last N average-bars of VOLUME (volume time, bar-approximated; exact dollar bars need ticks), percent.",
          soft=(-5.0, 5.0), period=(3, 200))
@safe
def volume_clock_return(f, p, c, l):
    n = clamp_p(p, 2, 400, 20)
    v = vol_col(f).fillna(0.0)
    cum = v.cumsum().to_numpy(float)
    avg = v.rolling(100, min_periods=20).mean().to_numpy(float)
    target = cum - n * avg
    idx = np.searchsorted(cum, target, side="left")
    c0 = px(f).to_numpy(float)
    ref = np.where((idx >= 0) & (idx < len(c0)) & np.isfinite(avg), c0[np.clip(idx, 0, len(c0) - 1)], np.nan)
    return pd.Series((c0 / ref - 1.0) * 100.0, index=f.index)


@njit(cache=False)
def _weis(close, atr, vol, k):
    n = len(close)
    wave = np.zeros(n)
    direction = 0
    ext = np.nan
    acc = 0.0
    for t in range(n):
        if np.isnan(atr[t]) or np.isnan(close[t]):
            wave[t] = acc * direction if direction != 0 else 0.0
            continue
        if np.isnan(ext):
            ext = close[t]
        if direction >= 0:
            if close[t] > ext:
                ext = close[t]
            if direction == 0 and close[t] - ext < -k * atr[t]:
                direction = -1
                ext = close[t]
                acc = 0.0
            elif direction == 1 and ext - close[t] > k * atr[t]:
                direction = -1
                ext = close[t]
                acc = 0.0
            elif direction == 0 and close[t] - ext > k * atr[t]:
                direction = 1
        else:
            if close[t] < ext:
                ext = close[t]
            if close[t] - ext > k * atr[t]:
                direction = 1
                ext = close[t]
                acc = 0.0
        acc += vol[t]
        wave[t] = acc * direction
    return wave


@register("weis_wave", 60, "Weis Wave volume: cumulative volume of the current swing wave (swing = reversal of N/10 ATR), signed by direction, in 20-bar average volumes.",
          soft=(-60.0, 60.0), period=(10, 100))
@safe
def weis_wave(f, p, c, l):
    k = clamp_p(p, 5, 100, 20) / 10.0
    v = vol_col(f)
    w = _weis(px(f).to_numpy(float), atr_series(f).to_numpy(float), v.fillna(0.0).to_numpy(float), k)
    return pd.Series(w, index=f.index) / v.rolling(20, min_periods=10).mean().replace(0, np.nan)


@register("updown_vol_ratio", 61, "(up-bar volume - down-bar volume) / total volume over N bars, -1..1.", bounded=(-1.0, 1.0), period=(3, 300))
@safe
def updown_vol_ratio(f, p, c, l):
    n = clamp_p(p, 2, 600, 20)
    v = vol_col(f)
    up = v.where(px(f) > px(f, "open"), 0.0).rolling(n, min_periods=n).sum()
    dn = v.where(px(f) < px(f, "open"), 0.0).rolling(n, min_periods=n).sum()
    return ((up - dn) / (up + dn).replace(0, np.nan)).clip(-1.0, 1.0)


@register("vw_close_location", 61, "Volume-weighted close location value over N bars ((C-L)-(H-C))/(H-L), -1..1.", bounded=(-1.0, 1.0), period=(3, 300))
@safe
def vw_close_location(f, p, c, l):
    n = clamp_p(p, 2, 600, 20)
    h, lo, cl, v = px(f, "high"), px(f, "low"), px(f), vol_col(f)
    clv = (((cl - lo) - (h - cl)) / (h - lo).replace(0, np.nan))
    return ((clv * v).rolling(n, min_periods=n).sum() / v.rolling(n, min_periods=n).sum().replace(0, np.nan)).clip(-1.0, 1.0)
