"""v13 FAMILY-SUPPORT primitives (report item number 0: not among the 100 indicators).

The 100 new strategy families need a handful of building blocks that are not on the indicator list: model outputs (ridge /
analog / triple-barrier / meta-label forecasts), multi-timeframe consensus, a cycle-adaptive breakout, basket spreads. They are
registered through the same registry (``number = 0``), obey the same causality rules (prefix-invariance tested) and are listed in
``V13_SUPPORT_KINDS`` so nothing pretends they are part of the 100.

MODEL CAUSALITY (the part that is easy to get wrong): a model that is fit on labels needs the label to be KNOWN. Every model below
refits on a fixed schedule at bar T using only rows whose label resolved strictly before T (``resolve_idx < T``), and applies that
fit to bars T .. T+step-1. Features at bar t use bars <= t only. Training windows are trailing, never the full sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.frame_attrs import frame_attr
from app.strategy.indicators_v13_core import (
    atr_series, clamp_p, log_ret, nan_like, njit, px, register, safe, vol_col,
)
from app.strategy.indicators_v13_market import _align_companion, _attr_frame, _basket_closes, _daily, _frame_day, _prior_map

SUPPORT_NAMES: list[str] = []


def _support(kind: str, desc: str, **kw):
    def deco(fn):
        register(kind, 0, desc, **kw)(fn)
        SUPPORT_NAMES.append(kind)
        return fn
    return deco


# ------------------------------------------------------------------------------------------------- simple structure
@_support("pdc_dist", "(close - PRIOR day's close) / prior daily ATR: how far price has stretched from yesterday's settlement.",
          soft=(-8.0, 8.0), no_period=True)
@safe
def pdc_dist(f, p, c, l):
    d = _daily(f)
    return (px(f) - _prior_map(f, d["close"])) / _prior_map(f, d["atr14"]).replace(0, np.nan)


@_support("box_position", "Position of close inside the box of the PRIOR N bars (0 = bottom, 1 = top; a breakout is > 1 or < 0).", soft=(-0.5, 1.5), period=(8, 200))
@safe
def box_position(f, p, c, l):
    n = clamp_p(p, 5, 400, 30)
    hi, lo = px(f, "high").shift(1).rolling(n, min_periods=n).max(), px(f, "low").shift(1).rolling(n, min_periods=n).min()
    return ((px(f) - lo) / (hi - lo).replace(0, np.nan)).clip(-0.5, 1.5)


@_support("ou_halflife", "Ornstein-Uhlenbeck half-life (bars) of the close's deviation from its N-bar mean, AR(1) over 3N bars (0..1000).",
          bounded=(0.0, 1000.0), period=(10, 200))
@safe
def ou_halflife(f, p, c, l):
    n = clamp_p(p, 8, 400, 30)
    x = px(f)
    d = x - x.rolling(n, min_periods=n).mean()
    w = 3 * n
    phi = d.rolling(w, min_periods=w).cov(d.shift(1)) / d.shift(1).rolling(w, min_periods=w).var().replace(0, np.nan)
    hl = pd.Series(np.where(phi >= 1.0, 1000.0, np.where(phi <= 0.0, 0.5, -np.log(2.0) / np.log(phi.clip(1e-6, 0.999999)))), index=f.index)
    return hl.where(phi.notna()).clip(0.0, 1000.0)


@_support("cycle_breakout_state", "Breakout of the high/low of the last (mult x dominant-cycle period) bars: +1 above, -1 below, else 0; period = lookback multiplier x10.",
          bounded=(-1.0, 1.0), period=(5, 30))
@safe
def cycle_breakout_state(f, p, c, l):
    from app.strategy.indicators_v13_core import BUILDERS
    mult = clamp_p(p, 5, 40, 10) / 10.0
    per = BUILDERS["hilbert_period"](f, 0, "close", None).to_numpy(float)
    h, lo, cl = px(f, "high"), px(f, "low"), px(f)
    out = np.full(len(f), np.nan)
    hi_by, lo_by = {}, {}
    for w in range(6, 51):
        ww = max(2, int(round(w * mult)))
        hi_by[w] = h.shift(1).rolling(ww, min_periods=ww).max().to_numpy(float)
        lo_by[w] = lo.shift(1).rolling(ww, min_periods=ww).min().to_numpy(float)
    c0 = cl.to_numpy(float)
    wi = np.where(np.isfinite(per), np.clip(np.round(per), 6, 50), 0).astype(int)
    for w in range(6, 51):
        m = wi == w
        if m.any():
            up = c0[m] > hi_by[w][m]
            dn = c0[m] < lo_by[w][m]
            ok = np.isfinite(hi_by[w][m]) & np.isfinite(lo_by[w][m])
            out[m] = np.where(ok, np.where(up, 1.0, np.where(dn, -1.0, 0.0)), np.nan)
    return pd.Series(out, index=f.index)


def _freq_minutes(f: pd.DataFrame) -> float:
    ts = pd.to_datetime(f["timestamp"]) if "timestamp" in f.columns else pd.Series(f.index)
    d = ts.diff().dt.total_seconds().dropna()
    return float(d.median() / 60.0) if len(d) else 1.0


def _htf_state(f: pd.DataFrame, rule: str, ema_n: int = 20) -> pd.Series:
    """+1/-1 trend state of the last COMPLETED higher-timeframe bar (close vs EMA), mapped to each base bar."""
    ts = pd.to_datetime(f["timestamp"])
    s = pd.DataFrame({"c": px(f).to_numpy()}, index=ts)
    htf = s["c"].resample(rule, label="left", closed="left").last().dropna()
    e = htf.ewm(span=ema_n, adjust=False, min_periods=ema_n).mean()
    state = pd.Series(np.where(htf > e, 1.0, np.where(htf < e, -1.0, 0.0)), index=htf.index).where(e.notna())
    bar_len = pd.tseries.frequencies.to_offset(rule)
    avail = pd.Series(state.to_numpy(), index=htf.index + bar_len)                     # known only when the HTF bar CLOSES
    left = pd.DataFrame({"t": ts.reset_index(drop=True)})
    right = pd.DataFrame({"t": avail.index, "v": avail.to_numpy()}).sort_values("t")
    m = pd.merge_asof(left.reset_index().sort_values("t"), right, on="t", direction="backward").sort_values("index")
    return pd.Series(m["v"].to_numpy(float), index=f.index)


@_support("htf_consensus", "Trend consensus (-4..4): base-timeframe EMA state plus the last COMPLETED 15-minute, 1-hour and daily bar states (close vs EMA20).",
          bounded=(-4.0, 4.0), period=(5, 100))
@safe
def htf_consensus(f, p, c, l):
    n = clamp_p(p, 5, 200, 20)
    base = pd.Series(np.where(px(f) > px(f).ewm(span=n, adjust=False, min_periods=n).mean(), 1.0, -1.0), index=f.index).where(
        px(f).ewm(span=n, adjust=False, min_periods=n).mean().notna())
    fm = _freq_minutes(f)
    parts = [base]
    for rule, mins in (("15min", 15.0), ("1h", 60.0), ("1D", 1440.0)):
        if fm < mins:
            parts.append(_htf_state(f, rule))
    out = pd.concat(parts, axis=1)
    return out.sum(axis=1, skipna=False)


@_support("nested_squeeze", "Multi-scale squeeze count (0..3): scales N, 4N, 16N whose Bollinger bandwidth sits in the bottom 20% of its trailing 200 values.",
          bounded=(0.0, 3.0), period=(10, 60))
@safe
def nested_squeeze(f, p, c, l):
    n = clamp_p(p, 8, 100, 20)
    x = px(f)
    total = pd.Series(0.0, index=f.index)
    ok = pd.Series(True, index=f.index)
    for k in (1, 4, 16):
        w = n * k
        bw = x.rolling(w, min_periods=w).std() / x.rolling(w, min_periods=w).mean().replace(0, np.nan)
        thr = bw.rolling(200, min_periods=100).quantile(0.2)
        total = total + (bw <= thr).astype(float)
        ok = ok & thr.notna()
    return total.where(ok)


# ------------------------------------------------------------------------------------------- vote and expectancy
@_support("vote_score", "Weak-signal ensemble vote (-8..8): signs of EMA12/26, MACD histogram, ROC(10), RSI-50, Donchian mid, Kalman slope, BVC imbalance, linear-regression slope.",
          bounded=(-8.0, 8.0), period=(5, 60))
@safe
def vote_score(f, p, c, l):
    from app.strategy.indicators import build_indicator_series
    from app.strategy.indicators_v13_core import BUILDERS
    n = clamp_p(p, 5, 100, 14)
    x = px(f)
    macd = x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    votes = [
        np.sign(x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean()),
        np.sign(hist),
        np.sign(x.pct_change(n)),
        np.sign(build_indicator_series(f, "rsi", period=n) - 50.0),
        np.sign(x - (px(f, "high").rolling(2 * n).max() + px(f, "low").rolling(2 * n).min()) / 2.0),
        np.sign(BUILDERS["kalman_slope"](f, 30, "close", None)),
        np.sign(BUILDERS["bvc_imbalance"](f, n, "close", None)),
        np.sign(build_indicator_series(f, "linreg_slope", period=n)) if True else None,
    ]
    return pd.concat(votes, axis=1).sum(axis=1, skipna=False)


@_support("base_signal_expectancy", "t-statistic of the recent per-bar P&L (in ATR) of an always-in EMA20/50 trend rule over N bars: the base signal's own recent expectancy.",
          soft=(-6.0, 6.0), period=(20, 500))
@safe
def base_signal_expectancy(f, p, c, l):
    n = clamp_p(p, 10, 1000, 100)
    x = px(f)
    state = np.sign(x.ewm(span=20, adjust=False).mean() - x.ewm(span=50, adjust=False).mean())
    atr = atr_series(f).shift(1)
    pnl = state.shift(1) * x.diff() / atr.replace(0, np.nan)
    return pnl.rolling(n, min_periods=n).mean() / pnl.rolling(n, min_periods=n).std().replace(0, np.nan) * np.sqrt(n)


# ------------------------------------------------------------------------------------------------ model forecasts
def _features(f: pd.DataFrame) -> pd.DataFrame:
    """Causal feature matrix (every column uses bars <= t)."""
    from app.strategy.indicators import rsi
    r = log_ret(f)
    vol = r.rolling(50, min_periods=20).std()
    X = {}
    for k in range(1, 6):
        X[f"r{k}"] = r.shift(k - 1) / vol
    X["m10"] = r.rolling(10).sum() / (vol * np.sqrt(10))
    X["m20"] = r.rolling(20).sum() / (vol * np.sqrt(20))
    X["m50"] = r.rolling(50).sum() / (vol * np.sqrt(50))
    X["volr"] = vol / r.rolling(200, min_periods=50).std() - 1.0
    X["rsi"] = (rsi(px(f), 14) - 50.0) / 25.0
    return pd.DataFrame(X, index=f.index).clip(-5.0, 5.0)


def _seasonal(f: pd.DataFrame) -> pd.DataFrame:
    from app.strategy.indicators_v11 import _minute_of_day, _ts
    t = _ts(f)
    m = _minute_of_day(t).astype(float)
    dow = t.dt.dayofweek
    cols = {"s1": np.sin(2 * np.pi * m / 1440.0), "c1": np.cos(2 * np.pi * m / 1440.0),
            "s2": np.sin(4 * np.pi * m / 1440.0), "c2": np.cos(4 * np.pi * m / 1440.0)}
    for d in range(4):
        cols[f"d{d}"] = (dow == d).astype(float)
    return pd.DataFrame(cols, index=f.index)


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def _purged_cv_lambda(X: np.ndarray, y: np.ndarray, lams, k: int = 4, embargo: int = 10) -> float:
    n = len(y)
    edges = np.linspace(0, n, k + 1).astype(int)
    best, best_err = lams[0], np.inf
    for lam in lams:
        err = 0.0
        for i in range(k):
            lo, hi = edges[i], edges[i + 1]
            tr = np.r_[0:max(0, lo - embargo), min(n, hi + embargo):n]
            if len(tr) < 50:
                continue
            b = _ridge_fit(X[tr], y[tr], lam)
            err += float(np.mean((y[lo:hi] - X[lo:hi] @ b) ** 2))
        if err < best_err:
            best, best_err = lam, err
    return best


def _ridge_series(f: pd.DataFrame, X: pd.DataFrame, refit: int, window: int, cv: bool) -> pd.Series:
    r = log_ret(f)
    vol = r.rolling(50, min_periods=20).std()
    y = (r.shift(-1) / vol).clip(-5.0, 5.0).to_numpy(float)        # next-bar return in vol units: KNOWN at bar j+1
    Xv = np.column_stack([X.to_numpy(float), np.ones(len(X))])
    n = len(f)
    out = np.full(n, np.nan)
    first = max(300, window // 4)
    t = first
    while t < n:
        lo = max(60, t - window)
        hi = t - 1                              # rows j <= t-2 have y_j = r_{j+1}/vol_j known strictly before t
        Xt, yt = Xv[lo:hi], y[lo:hi]
        ok = np.isfinite(Xt).all(axis=1) & np.isfinite(yt)
        if ok.sum() > 150:
            lam = _purged_cv_lambda(Xt[ok], yt[ok], (1.0, 10.0, 100.0, 1000.0)) if cv else 100.0
            b = _ridge_fit(Xt[ok], yt[ok], lam)
            end = min(n, t + refit)
            seg = Xv[t:end]
            pred = seg @ b
            sd = float(np.std(Xt[ok] @ b)) or 1.0
            out[t:end] = np.where(np.isfinite(seg).all(axis=1), pred / sd, np.nan)
        t += refit
    return pd.Series(out, index=f.index)


@_support("ridge_forecast", "Rolling ridge forecast of the next-bar return (standardized): causal linear model on 9 features, refit every N bars on a trailing window.",
          soft=(-4.0, 4.0), period=(100, 2000))
@safe
def ridge_forecast(f, p, c, l):
    n = clamp_p(p, 50, 5000, 250)
    return _ridge_series(f, _features(f), n, min(6000, 20 * n), False)


@_support("seasonal_cv_forecast", "Seasonality-adjusted ridge forecast (hour-of-day harmonics + weekday terms) with the penalty chosen by PURGED k-fold CV inside each training window.",
          soft=(-4.0, 4.0), period=(100, 2000))
@safe
def seasonal_cv_forecast(f, p, c, l):
    n = clamp_p(p, 50, 5000, 250)
    X = pd.concat([_features(f), _seasonal(f)], axis=1)
    return _ridge_series(f, X, n, min(6000, 20 * n), True)


@njit(cache=False)
def _analog(z, w, h, lib, k, step, fwd):
    """For every ``step``-th bar t: mean forward return of the k library windows closest (L2) to the window ending at t."""
    n = len(z)
    out = np.full(n, np.nan)
    last = np.nan
    for t in range(w + h + 50, n):
        if (t - (w + h + 50)) % step == 0:
            lo = max(w, t - h - lib)
            hi = t - h                       # a library window ending at j needs fwd[j] (j..j+h) fully known by t: j <= t-h-1
            m = hi - lo
            if m < 50:
                last = np.nan
            else:
                best_d = np.full(k, 1e300)
                best_f = np.zeros(k)
                for j in range(lo, hi):
                    d = 0.0
                    for q in range(w):
                        a = z[t - q] - z[j - q]
                        d += a * a
                        if d > best_d[k - 1]:
                            break
                    if d < best_d[k - 1]:
                        pos = k - 1
                        while pos > 0 and best_d[pos - 1] > d:
                            best_d[pos] = best_d[pos - 1]
                            best_f[pos] = best_f[pos - 1]
                            pos -= 1
                        best_d[pos] = d
                        best_f[pos] = fwd[j]
                last = np.mean(best_f)
        out[t] = last
    return out


@_support("analog_match", "Analog forecast: mean forward 10-bar return (vol units) of the 20 historical windows most similar to the last N returns (library = trailing 500 windows, refreshed every 25 bars).",
          soft=(-3.0, 3.0), period=(10, 60))
@safe
def analog_match(f, p, c, l):
    w = clamp_p(p, 8, 100, 20)
    h = 10
    r = log_ret(f)
    vol = r.rolling(50, min_periods=20).std()
    z = (r / vol).fillna(0.0).clip(-5.0, 5.0).to_numpy(float)
    fwd = (r.rolling(h).sum().shift(-h) / (vol * np.sqrt(h))).fillna(0.0).to_numpy(float)    # sum of the NEXT h returns (known h bars later)
    return pd.Series(_analog(z, w, h, 500, 20, 25, fwd), index=f.index)


def _logit_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0, iters: int = 25) -> np.ndarray:
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (p - y) + lam * b
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X + lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b = b - step
        if np.max(np.abs(step)) < 1e-6:
            break
    return b


@njit(cache=False)
def _barrier_labels(high, low, close, atr, mult, horizon):
    """Triple-barrier label per bar (+1 upper first, -1 lower first, 0 neither, NaN no ATR) and the bar index at which it RESOLVED."""
    n = len(close)
    lab = np.full(n, np.nan)
    res = np.full(n, np.nan)
    for i in range(n - 1):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        up = close[i] + mult * a
        dn = close[i] - mult * a
        end = min(n - 1, i + horizon)
        lab[i] = 0.0
        res[i] = end
        for j in range(i + 1, end + 1):
            hit_up = high[j] >= up
            hit_dn = low[j] <= dn
            if hit_up and hit_dn:
                lab[i] = 0.0
                res[i] = j
                break
            if hit_up:
                lab[i] = 1.0
                res[i] = j
                break
            if hit_dn:
                lab[i] = -1.0
                res[i] = j
                break
    return lab, res


@_support("triple_barrier_prob", "Supervised triple-barrier classifier: P(upper 1-ATR barrier hit before the lower within 24 bars) - 0.5, logistic model refit every N bars on labels RESOLVED before the refit.",
          soft=(-0.5, 0.5), period=(100, 2000))
@safe
def triple_barrier_prob(f, p, c, l):
    refit = clamp_p(p, 50, 5000, 250)
    X = np.column_stack([_features(f).to_numpy(float), np.ones(len(f))])
    lab, res = _barrier_labels(px(f, "high").to_numpy(float), px(f, "low").to_numpy(float), px(f).to_numpy(float), atr_series(f).to_numpy(float), 1.0, 24)
    n = len(f)
    out = np.full(n, np.nan)
    t = 400
    while t < n:
        lo = max(0, t - 6000)
        idx = np.arange(lo, t)
        ok = np.isfinite(lab[idx]) & (lab[idx] != 0) & np.isfinite(res[idx]) & (res[idx] < t) & np.isfinite(X[idx]).all(axis=1)     # resolved BEFORE t
        if ok.sum() > 200:
            ii = idx[ok]
            b = _logit_fit(X[ii], (lab[ii] > 0).astype(float))
            end = min(n, t + refit)
            seg = X[t:end]
            pr = 1.0 / (1.0 + np.exp(-np.clip(seg @ b, -30, 30)))
            out[t:end] = np.where(np.isfinite(seg).all(axis=1), pr - 0.5, np.nan)
        t += refit
    return pd.Series(out, index=f.index)


@_support("meta_label_prob", "Meta-labeling: P(the primary EMA12/48 signal's direction pays over the next 12 bars), logistic model refit every N bars on RESOLVED outcomes; 0..1.",
          bounded=(0.0, 1.0), period=(100, 2000))
@safe
def meta_label_prob(f, p, c, l):
    refit = clamp_p(p, 50, 5000, 250)
    x = px(f)
    d = np.sign(x.ewm(span=12, adjust=False).mean() - x.ewm(span=48, adjust=False).mean()).replace(0, np.nan)
    F = _features(f)
    X = np.column_stack([F.to_numpy(float) * d.to_numpy(float)[:, None], np.abs(F[["volr"]].to_numpy(float)), np.ones(len(f))])
    h = 12
    fwd = (x.shift(-h) - x) / atr_series(f)
    y = ((d * fwd) > 0).astype(float).where(fwd.notna() & d.notna()).to_numpy(float)
    n = len(f)
    out = np.full(n, np.nan)
    t = 400
    while t < n:
        lo = max(0, t - 6000)
        hi = t - h - 1                       # outcome of row j is known at bar j+h <= t-1
        ok = np.isfinite(y[lo:hi]) & np.isfinite(X[lo:hi]).all(axis=1)
        if ok.sum() > 200:
            b = _logit_fit(X[lo:hi][ok], y[lo:hi][ok])
            end = min(n, t + refit)
            seg = X[t:end]
            out[t:end] = np.where(np.isfinite(seg).all(axis=1), 1.0 / (1.0 + np.exp(-np.clip(seg @ b, -30, 30))), np.nan)
        t += refit
    return pd.Series(out, index=f.index)


# ------------------------------------------------------------------------------------------------ basket spreads
@_support("basket_pc_spread_z", "Cointegrated-basket spread: z-score of the smallest-variance principal-component spread of log prices (own + frame.attrs['basket']) over N bars, weights refit every N/4 bars (Johansen is not available; a stationarity gate is basket_pc_phi).",
          soft=(-5.0, 5.0), period=(60, 1000))
@safe
def basket_pc_spread_z(f, p, c, l):
    return _basket_pc(f, p)[0]


@_support("basket_pc_phi", "AR(1) coefficient of the basket principal-component spread (below ~0.98 = mean-reverting): the stationarity gate for basket_pc_spread_z.",
          soft=(0.0, 1.05), period=(60, 1000))
@safe
def basket_pc_phi(f, p, c, l):
    return _basket_pc(f, p)[1]


def _basket_pc(f, p):
    cl = _basket_closes(f)
    if cl is None:
        return nan_like(f), nan_like(f)
    n = clamp_p(p, 40, 2000, 120)
    L = np.log(np.column_stack([px(f).to_numpy(float), cl.to_numpy(float)]))
    N = len(f)
    z = np.full(N, np.nan)
    phi = np.full(N, np.nan)
    step = max(10, n // 4)
    t = n + 1
    while t < N:
        win = L[t - n:t]
        if np.isfinite(win).all():
            ev, evec = np.linalg.eigh(np.cov(win, rowvar=False))
            w = evec[:, 0]
            spread = win @ w
            mu, sd = spread.mean(), spread.std()
            a = np.polyfit(spread[:-1] - mu, spread[1:] - mu, 1)[0] if sd > 0 else np.nan
            end = min(N, t + step)
            seg = L[t:end] @ w
            z[t:end] = (seg - mu) / sd if sd > 0 else np.nan
            phi[t:end] = a
        t += step
    return pd.Series(z, index=f.index), pd.Series(phi, index=f.index)


@njit(cache=False)
def _kalman_hedge(y, x, delta):
    n = len(y)
    out = np.full(n, np.nan)
    beta = np.zeros(2)
    P = np.eye(2)
    Vw = delta / (1.0 - delta)
    Ve = 1e-3
    for t in range(n):
        if np.isnan(y[t]) or np.isnan(x[t]):
            continue
        P = P + Vw * np.eye(2)
        h = np.array([x[t], 1.0])
        yhat = beta[0] * x[t] + beta[1]
        e = y[t] - yhat
        S = h @ P @ h + Ve
        K = P @ h / S
        beta = beta + K * e
        P = P - np.outer(K, h) @ P
        Ve = 0.99 * Ve + 0.01 * e * e
        if t > 100:
            out[t] = e / np.sqrt(S)
    return out


@_support("kalman_hedge_spread_z", "Dynamic-hedge spread: standardized Kalman innovation of own log price regressed on the benchmark/companion (frame.attrs['benchmark']) with a time-varying hedge ratio.",
          soft=(-5.0, 5.0), no_period=True)
@safe
def kalman_hedge_spread_z(f, p, c, l):
    other = _attr_frame(f, "benchmark")
    if other is None:
        return nan_like(f)
    x = np.log(_align_companion(f, other)).to_numpy(float)
    return pd.Series(_kalman_hedge(np.log(px(f)).to_numpy(float), x, 1e-4), index=f.index)



# ============================================================================================ session / structure states
# Used by the session, breakout and fade families. Everything is bar-t-causal: day-to-date aggregates are cumulative within the
# trading day, "opening range" values appear only once the range window has CLOSED, and prior-period levels come through the
# shift(1) daily table.
def _mso(f):
    from app.strategy.indicators_v11 import minutes_since_open
    return minutes_since_open(f, 0, "close", None)


def _day(f):
    return _frame_day(f).reset_index(drop=True).to_numpy()


def _rth(f):
    m = _mso(f)
    return (m >= 0) & (m < 390)


def _day_open(f):
    """Open of the first RTH bar of the trading day, defined only from that bar onward (never before it)."""
    rth = _rth(f)
    op = px(f, "open").where(rth)
    return op.groupby(_day(f)).transform("first").where(rth)


def _datr(f):
    return _prior_map(f, _daily(f)["atr14"]).replace(0, np.nan)


@_support("day_open_dist", "(close - today's RTH open) / prior daily ATR; NaN outside the regular session.", soft=(-8.0, 8.0), no_period=True)
@safe
def day_open_dist(f, p, c, l):
    return (px(f) - _day_open(f)) / _datr(f)


@_support("morning_move", "Today's move from the RTH open to the close of the last bar within the first N minutes (frozen after N), in prior daily ATR.",
          soft=(-8.0, 8.0), period=(30, 240))
@safe
def morning_move(f, p, c, l):
    m = clamp_p(p, 15, 360, 120)
    rth, mso = _rth(f), _mso(f)
    morning = px(f).where(rth & (mso <= m))
    last = morning.groupby(_day(f)).ffill()
    return ((last - _day_open(f)) / _datr(f)).where(rth)


def _or_levels(f, m):
    rth, mso = _rth(f), _mso(f)
    inside = rth & (mso < m)
    g = _day(f)
    hi = px(f, "high").where(inside).groupby(g).cummax().groupby(g).ffill()
    lo = px(f, "low").where(inside).groupby(g).cummin().groupby(g).ffill()
    done = rth & (mso >= m)
    return hi.where(done), lo.where(done), done


@_support("or_width_atr", "Opening-range (first N minutes) width in prior daily ATR; available once the range window has closed.", soft=(0.0, 6.0), period=(5, 90))
@safe
def or_width_atr(f, p, c, l):
    hi, lo, _ = _or_levels(f, clamp_p(p, 5, 120, 15))
    return (hi - lo) / _datr(f)


@_support("or_state", "Close vs the opening range of the first N minutes: +1 above its high, -1 below its low, 0 inside (after the range closes).",
          bounded=(-1.0, 1.0), period=(5, 90))
@safe
def or_state(f, p, c, l):
    hi, lo, done = _or_levels(f, clamp_p(p, 5, 120, 15))
    x = px(f)
    return pd.Series(np.where(x > hi, 1.0, np.where(x < lo, -1.0, 0.0)), index=f.index).where(done & hi.notna())


@_support("or_failure", "Failed opening-range break within 2 hours of the range closing: +1 a downside break that closed back inside (long), -1 an upside break that closed back inside.",
          bounded=(-1.0, 1.0), period=(5, 90))
@safe
def or_failure(f, p, c, l):
    m = clamp_p(p, 5, 120, 30)
    hi, lo, done = _or_levels(f, m)
    x, g, mso = px(f), _day(f), _mso(f)
    broke_dn = ((x < lo) & done).astype(float).groupby(g).cummax().groupby(g).shift(1)
    broke_up = ((x > hi) & done).astype(float).groupby(g).cummax().groupby(g).shift(1)
    long_ = (broke_dn == 1.0) & (x >= lo) & (x <= hi)
    short_ = (broke_up == 1.0) & (x <= hi) & (x >= lo)
    out = pd.Series(np.where(long_ & ~short_, 1.0, np.where(short_ & ~long_, -1.0, 0.0)), index=f.index)
    return out.where(done & hi.notna() & (mso <= m + 120))


@_support("round_tag", "First tag of a round number (step = 10^(digits-1)): +1 a low that touched the level below price and closed above (long), -1 a high that touched the level above and closed below; only if the level was not within 0.5 ATR of the prior N bars.",
          bounded=(-1.0, 1.0), period=(20, 200))
@safe
def round_tag(f, p, c, l):
    n = clamp_p(p, 10, 500, 50)
    x, h, lo, atr = px(f), px(f, "high"), px(f, "low"), atr_series(f).shift(1)
    pc = x.shift(1)
    step = 10.0 ** (np.floor(np.log10(pc.clip(lower=1e-9))) - 1.0)
    up_lvl, dn_lvl = np.ceil(pc / step) * step, np.floor(pc / step) * step
    near = 0.1 * atr
    tag_up = (h >= up_lvl - near) & (x < up_lvl)
    tag_dn = (lo <= dn_lvl + near) & (x > dn_lvl)
    fresh_up = h.shift(1).rolling(n, min_periods=n).max() < up_lvl - 0.5 * atr
    fresh_dn = lo.shift(1).rolling(n, min_periods=n).min() > dn_lvl + 0.5 * atr
    out = np.where(tag_dn & fresh_dn & ~(tag_up & fresh_up), 1.0, np.where(tag_up & fresh_up & ~(tag_dn & fresh_dn), -1.0, 0.0))
    return pd.Series(out, index=f.index).where(atr.notna() & fresh_up.notna())


@_support("nr_breakout", "NRn breakout: the PREVIOUS bar had the narrowest range of the last N bars; +1 if this close breaks its high, -1 its low, else 0.",
          bounded=(-1.0, 1.0), period=(4, 20))
@safe
def nr_breakout(f, p, c, l):
    n = clamp_p(p, 3, 60, 7)
    h, lo, x = px(f, "high"), px(f, "low"), px(f)
    rng = h - lo
    nr = (rng <= rng.rolling(n, min_periods=n).min()).shift(1)
    up, dn = x > h.shift(1), x < lo.shift(1)
    out = pd.Series(np.where(nr == True, np.where(up, 1.0, np.where(dn, -1.0, 0.0)), 0.0), index=f.index)     # noqa: E712
    return out.where(rng.rolling(n, min_periods=n).min().shift(1).notna())


@_support("inside_break", "Inside-bar break: the previous bar sat inside the bar before it; +1 if this close exceeds the mother bar's high, -1 its low, else 0.",
          bounded=(-1.0, 1.0), no_period=True)
@safe
def inside_break(f, p, c, l):
    h, lo, x = px(f, "high"), px(f, "low"), px(f)
    inside = ((h.shift(1) <= h.shift(2)) & (lo.shift(1) >= lo.shift(2)))
    out = np.where(inside, np.where(x > h.shift(2), 1.0, np.where(x < lo.shift(2), -1.0, 0.0)), 0.0)
    return pd.Series(out, index=f.index).where(h.shift(2).notna())


@_support("mo_range_break", "Month opening-range break: high/low of the first N trading days of the month; +1/-1/0 for a close above/below/inside it on later days (NaN during the first N days).",
          bounded=(-1.0, 1.0), period=(2, 10))
@safe
def mo_range_break(f, p, c, l):
    n = clamp_p(p, 1, 15, 3)
    d = _daily(f)
    ym = pd.Series(pd.DatetimeIndex(d.index).to_period("M").astype(str), index=d.index)
    rank = d.groupby(ym).cumcount() + 1
    hi = d["high"].where(rank <= n).groupby(ym).cummax().groupby(ym).ffill().where(rank > n)
    lo = d["low"].where(rank <= n).groupby(ym).cummin().groupby(ym).ffill().where(rank > n)
    day = _frame_day(f)
    H, L = day.map(hi).to_numpy(float), day.map(lo).to_numpy(float)
    x = px(f).to_numpy(float)
    out = np.where(x > H, 1.0, np.where(x < L, -1.0, 0.0))
    return pd.Series(np.where(np.isfinite(H) & np.isfinite(L), out, np.nan), index=f.index)


@_support("breakout_retest", "Breakout-retest-continuation: within 20 bars of a close beyond the prior N-bar high/low, +1 when price dips back to within 0.25 ATR of that level and closes above it (-1 mirror).",
          bounded=(-1.0, 1.0), period=(10, 100))
@safe
def breakout_retest(f, p, c, l):
    n = clamp_p(p, 5, 300, 30)
    h, lo, x, atr = px(f, "high"), px(f, "low"), px(f), atr_series(f).shift(1)
    hl, ll = h.shift(1).rolling(n, min_periods=n).max(), lo.shift(1).rolling(n, min_periods=n).min()
    up_lvl = hl.where(x > hl).ffill(limit=20).shift(1)
    dn_lvl = ll.where(x < ll).ffill(limit=20).shift(1)
    long_ = (lo <= up_lvl + 0.25 * atr) & (x > up_lvl)
    short_ = (h >= dn_lvl - 0.25 * atr) & (x < dn_lvl)
    out = pd.Series(np.where(long_ & ~short_, 1.0, np.where(short_ & ~long_, -1.0, 0.0)), index=f.index)
    return out.where(hl.notna() & atr.notna())


@_support("wedge_break", "Triangle/wedge break: highs and lows of the last N bars converge (high slope < 0 < low slope); +1 if close exceeds the projected upper line, -1 the lower, else 0. OLS lines, not Theil-Sen.",
          bounded=(-1.0, 1.0), period=(20, 120))
@safe
def wedge_break(f, p, c, l):
    n = clamp_p(p, 10, 400, 40)
    h, lo, x = px(f, "high"), px(f, "low"), px(f)
    t = pd.Series(np.arange(len(f), dtype=float), index=f.index)
    vt = t.rolling(n, min_periods=n).var()

    def line(y):
        slope = y.rolling(n, min_periods=n).cov(t) / vt
        fitted_end = y.rolling(n, min_periods=n).mean() + slope * ((n - 1) / 2.0)
        return slope, fitted_end
    sh, eh = line(h)
    sl, el = line(lo)
    up_proj, dn_proj = (eh + sh).shift(1), (el + sl).shift(1)
    converge = ((sh < 0) & (sl > 0)).shift(1) & (up_proj > dn_proj)
    out = np.where(converge & (x > up_proj), 1.0, np.where(converge & (x < dn_proj), -1.0, 0.0))
    return pd.Series(out, index=f.index).where(up_proj.notna() & dn_proj.notna())


@_support("po3_state", "Power-of-Three daily model: today swept the PRIOR day's low then closed back above it and above today's open (+1), or the high-side mirror (-1); RTH only.",
          bounded=(-1.0, 1.0), no_period=True)
@safe
def po3_state(f, p, c, l):
    g = _day(f)
    d = _daily(f)
    pdl, pdh = _prior_map(f, d["low"]), _prior_map(f, d["high"])
    lo_td = px(f, "low").groupby(g).cummin()
    hi_td = px(f, "high").groupby(g).cummax()
    x, op = px(f), _day_open(f)
    long_ = (lo_td < pdl) & (x > pdl) & (x > op)
    short_ = (hi_td > pdh) & (x < pdh) & (x < op)
    out = pd.Series(np.where(long_ & ~short_, 1.0, np.where(short_ & ~long_, -1.0, 0.0)), index=f.index)
    return out.where(_rth(f) & pdl.notna() & op.notna())


@_support("spread_z", "Z-score of the Corwin-Schultz spread estimate (N bars) against its own trailing 500 values: a liquidity-vacuum detector that needs no instrument-specific threshold.",
          soft=(-4.0, 8.0), period=(5, 60))
@safe
def spread_z(f, p, c, l):
    from app.strategy.indicators_v13_core import BUILDERS
    s = BUILDERS["corwin_schultz"](f, clamp_p(p, 5, 100, 20), "close", None)
    return (s - s.rolling(500, min_periods=100).mean()) / s.rolling(500, min_periods=100).std().replace(0, np.nan)


@_support("amihud_z", "Z-score of the Amihud illiquidity measure (N bars) against its own trailing 500 values.", soft=(-4.0, 8.0), period=(5, 100))
@safe
def amihud_z(f, p, c, l):
    from app.strategy.indicators_v13_core import BUILDERS
    s = BUILDERS["amihud"](f, clamp_p(p, 5, 200, 20), "close", None)
    return (s - s.rolling(500, min_periods=100).mean()) / s.rolling(500, min_periods=100).std().replace(0, np.nan)


@_support("day_range_used", "Today's high-low so far as a multiple of the prior 14-day average daily range; RTH only.", soft=(0.0, 4.0), no_period=True)
@safe
def day_range_used(f, p, c, l):
    g = _day(f)
    rng = px(f, "high").groupby(g).cummax() - px(f, "low").groupby(g).cummin()
    return (rng / _datr(f)).where(_rth(f))


@_support("day_range_pos", "Where the close sits in today's range so far (0 = at the low, 1 = at the high); RTH only.", bounded=(0.0, 1.0), no_period=True)
@safe
def day_range_pos(f, p, c, l):
    g = _day(f)
    hi, lo = px(f, "high").groupby(g).cummax(), px(f, "low").groupby(g).cummin()
    return ((px(f) - lo) / (hi - lo).replace(0, np.nan)).clip(0.0, 1.0).where(_rth(f))


def _comp_ret_z(f, key, p):
    other = _attr_frame(f, key)
    if other is None:
        return nan_like(f)
    x = np.log(_align_companion(f, other))
    r = x.diff(clamp_p(p, 1, 500, 20))
    return r / r.rolling(500, min_periods=100).std().replace(0, np.nan)


@_support("oil_ret_z", "Z-scored N-bar log return of the oil companion (frame.attrs['oil']); NaN without it.", soft=(-5.0, 5.0), period=(5, 200))
@safe
def oil_ret_z(f, p, c, l):
    return _comp_ret_z(f, "oil", p)


@_support("copper_ret_z", "Z-scored N-bar log return of the copper companion (frame.attrs['copper']); NaN without it.", soft=(-5.0, 5.0), period=(5, 200))
@safe
def copper_ret_z(f, p, c, l):
    return _comp_ret_z(f, "copper", p)


V13_SUPPORT_KINDS: tuple[str, ...] = tuple(SUPPORT_NAMES)
SUPPORT_NEEDS_COMPANION = ("basket_pc_spread_z", "basket_pc_phi", "kalman_hedge_spread_z", "oil_ret_z", "copper_ret_z")
