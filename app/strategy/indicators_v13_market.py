"""v13 indicator groups G-K (report 2026-10-25, items 62-100): market structure, cross-asset / macro (Ext), crypto, seasonality, regime models.

  F (tail)  62  open-interest change and price-OI regime (Ext)      63 event-anchored VWAP (OHLCV)
  G  structure   64-73   ATR zigzag, TPO profile, naked POC, weekly/monthly levels, equal highs/lows, premium/discount, gap fill, candle anatomy, box, Theil-Sen
  H  cross-asset 74-89   beta/residual, absorption ratio, cross-sectional momentum (companion frames); credit/rates/vol/COT/... (Ext feeds with release lags)
  I  crypto      90-92   funding z, OI + liquidation proxy, perp-spot basis (Ext)
  J  seasonality 93-94   causal time-of-day and calendar seasonality, session handoff ratios
  K  regime      95-100  HMM filtered prob, Markov-switching variance, BOCPD, CUSUM, causal clusters, drawdown state

External data: every Ext indicator reads its series ONLY through ``app.data.macro_data.ext_aligned`` (daily feeds are delayed by the
feed's release lag; intraday feeds are as-of merged on the bar time). A missing feed gives NaN (a condition on it is False -- no
trades, never a fake neutral value). Companion data (benchmark, basket, commodity legs) is passed in ``frame.attrs``:
``benchmark`` (DataFrame), ``basket`` ({symbol: DataFrame}), ``copper`` / ``gold`` / ``silver`` / ``oil`` (DataFrames);
companion bars are aligned with ``merge_asof(direction='backward')`` so a bar never sees a companion bar stamped later than itself.
"""
from __future__ import annotations

from math import lgamma

import numpy as np
import pandas as pd

from app.strategy.frame_attrs import frame_attr
from app.strategy.indicators_v13_core import (
    atr_series, clamp_p, iir, log_ret, nan_like, njit, px, register, roll_ema, safe, strided, vol_col,
)

EXT_KINDS: set[str] = set()          # kinds that need an external feed (macro_data.EXT_FEEDS)
COMPANION_KINDS: set[str] = set()    # kinds that need companion frames in frame.attrs
INSTRUMENT_TIED: set[str] = set()    # kinds whose feed belongs to ONE instrument (their families cannot pool)


def _mark(kind: str, ext: bool = False, companion: bool = False, tied: bool = False) -> None:
    if ext:
        EXT_KINDS.add(kind)
    if companion:
        COMPANION_KINDS.add(kind)
    if tied:
        INSTRUMENT_TIED.add(kind)


# --------------------------------------------------------------------------------------------------------- helpers
def _v11():
    from app.strategy import indicators_v11 as v11
    return v11


def _frame_day(f):
    return _v11()._frame_day(f)


def _ts(f):
    return _v11()._ts(f)


def _daily(f):
    return _v11()._daily_table(f)


def _prior_map(f, s):
    return _v11()._prior_day_map(f, s)


def _align_companion(f: pd.DataFrame, other: pd.DataFrame, col: str = "close") -> pd.Series:
    """The companion's ``col`` as of each bar of ``f`` (backward as-of on timestamp); NaN where it has no bar yet."""
    if not isinstance(other, pd.DataFrame) or other.empty:
        return nan_like(f)
    o = other.copy()
    if "timestamp" not in o.columns:
        if isinstance(o.index, pd.DatetimeIndex):
            o = o.reset_index().rename(columns={o.index.name or "index": "timestamp"})
        else:
            return nan_like(f)
    left = pd.DataFrame({"t": pd.to_datetime(f["timestamp"] if "timestamp" in f.columns else f.index).reset_index(drop=True)})
    right = pd.DataFrame({"t": pd.to_datetime(o["timestamp"]).reset_index(drop=True), "v": pd.to_numeric(o[col], errors="coerce").reset_index(drop=True)}).sort_values("t")
    m = pd.merge_asof(left.reset_index().sort_values("t"), right, on="t", direction="backward").sort_values("index")
    return pd.Series(m["v"].to_numpy(dtype=float), index=f.index)


def _attr_frame(f: pd.DataFrame, key: str):
    v = frame_attr(f, key)
    return v if isinstance(v, pd.DataFrame) and not v.empty else None


def _basket_closes(f: pd.DataFrame) -> pd.DataFrame | None:
    b = frame_attr(f, "basket")
    if not isinstance(b, dict) or len(b) < 2:
        return None
    cols = {str(k): _align_companion(f, v) for k, v in b.items() if isinstance(v, pd.DataFrame) and not v.empty}
    return pd.DataFrame(cols) if len(cols) >= 2 else None


# ===================================================================================================== F tail: 62, 63
@register("oi_change_pct", 62, "Open-interest change over N daily observations, percent (prior-day OI; Ext feed open_interest_<SYMBOL>.csv, release lag 1 day).",
          soft=(-30.0, 30.0), period=(1, 60))
@safe
def oi_change_pct(f, p, c, l):
    from app.data import macro_data as md
    s = md.ext_series(f, "open_interest")
    n = clamp_p(p, 1, 120, 1)
    return pd.Series(md.ext_aligned(f, "open_interest", (s.pct_change(n) * 100.0).dropna()), index=f.index)


_mark("oi_change_pct", ext=True, tied=True)


@register("price_oi_regime", 62, "Four-quadrant positioning of the PRIOR day: +2 price up/OI up, +1 price up/OI down, -1 price down/OI down, -2 price down/OI up.",
          bounded=(-2.0, 2.0), no_period=True)
@safe
def price_oi_regime(f, p, c, l):
    from app.data import macro_data as md
    s = md.ext_series(f, "open_interest")
    if not len(s):
        return nan_like(f)
    oi = pd.Series(md.ext_aligned(f, "open_interest", s.pct_change(1).dropna()), index=f.index)
    pr = _prior_map(f, _daily(f)["close"].pct_change())
    up, dn = pr > 0, pr < 0
    oup, odn = oi > 0, oi < 0
    out = np.select([up & oup, up & odn, dn & odn, dn & oup], [2.0, 1.0, -1.0, -2.0], default=0.0)
    return pd.Series(out, index=f.index).where(oi.notna() & pr.notna())


_mark("price_oi_regime", ext=True, tied=True)


def _avwap(f: pd.DataFrame, start_flag: pd.Series) -> pd.Series:
    """VWAP (typical price, volume weighted; equal weights when the volume column is empty) from the most recent anchor bar."""
    tp = (px(f, "high") + px(f, "low") + px(f)) / 3.0
    v = vol_col(f).fillna(0.0)
    if float(v.sum()) <= 0:
        v = pd.Series(1.0, index=f.index)
    grp = start_flag.fillna(False).astype(bool).cumsum()
    num = (tp * v).groupby(grp).cumsum()
    den = v.groupby(grp).cumsum().replace(0, np.nan)
    out = num / den
    return out.where(grp > 0)


def _avwap_dist(f: pd.DataFrame, start_flag: pd.Series) -> pd.Series:
    return (px(f) - _avwap(f, start_flag)) / atr_series(f).replace(0, np.nan)


def _first_of(key: pd.Series) -> pd.Series:
    return key != key.shift(1)


@register("avwap_dist_week", 63, "Distance of close from the VWAP anchored at the week's first bar, in ATR(14).", soft=(-8.0, 8.0), no_period=True)
@safe
def avwap_dist_week(f, p, c, l):
    wk = _frame_day(f).dt.to_period("W")
    return _avwap_dist(f, _first_of(wk.astype(str)))


@register("avwap_dist_month", 63, "Distance of close from the VWAP anchored at the month's first bar, in ATR(14).", soft=(-10.0, 10.0), no_period=True)
@safe
def avwap_dist_month(f, p, c, l):
    mo = _frame_day(f).dt.to_period("M")
    return _avwap_dist(f, _first_of(mo.astype(str)))


@register("avwap_dist_fomc", 63, "Distance of close from the VWAP anchored at the open of the most recent FOMC decision day, in ATR(14).",
          soft=(-10.0, 10.0), no_period=True)
@safe
def avwap_dist_fomc(f, p, c, l):
    v11 = _v11()
    flag = v11.event_day_fomc(f, 0, "close", None).fillna(0.0) > 0.5
    day = _frame_day(f)
    first_of_event_day = flag & (day != day.shift(1))
    return _avwap_dist(f, first_of_event_day)


@register("avwap_dist_gap", 63, "Distance of close from the VWAP anchored at the open of the last day that opened with a gap > 0.5 daily ATR, in ATR(14).",
          soft=(-10.0, 10.0), no_period=True)
@safe
def avwap_dist_gap(f, p, c, l):
    d = _daily(f)
    day = _frame_day(f)
    day_open = f["open"].astype(float).groupby(day).transform("first")
    prev_close = _prior_map(f, d["close"])
    prev_atr = _prior_map(f, d["atr14"])
    gap = (day_open - prev_close).abs() / prev_atr.replace(0, np.nan)
    return _avwap_dist(f, (gap > 0.5) & (day != day.shift(1)))


@register("avwap_dist_swing", 63, "Distance of close from the VWAP anchored at the last CONFIRMED swing (pivot of N bars each side; known N bars later), in ATR(14).",
          soft=(-10.0, 10.0), period=(2, 30))
@safe
def avwap_dist_swing(f, p, c, l):
    k = clamp_p(p, 2, 60, 5)
    h, lo = px(f, "high"), px(f, "low")
    win = 2 * k + 1
    hi_piv = (h == h.rolling(win, center=True, min_periods=win).max())
    lo_piv = (lo == lo.rolling(win, center=True, min_periods=win).min())
    piv = (hi_piv | lo_piv)
    # a pivot at bar i is KNOWN at bar i+k; anchor the VWAP at bar i but only from i+k on
    pos = pd.Series(np.where(piv, np.arange(len(f)), np.nan), index=f.index).shift(k).ffill()
    tp = (h + lo + px(f)) / 3.0
    v = vol_col(f).fillna(0.0)
    if float(v.sum()) <= 0:
        v = pd.Series(1.0, index=f.index)
    cpv, cv = (tp * v).cumsum().to_numpy(float), v.cumsum().to_numpy(float)
    a = pos.to_numpy(float)
    ok = np.isfinite(a)
    ai = np.where(ok, a, 0).astype(int)
    base_pv = np.where(ai > 0, cpv[np.maximum(ai - 1, 0)], 0.0)
    base_v = np.where(ai > 0, cv[np.maximum(ai - 1, 0)], 0.0)
    den = cv - base_v
    vw = np.where(ok & (den > 0), (cpv - base_pv) / np.where(den > 0, den, 1.0), np.nan)
    return (px(f) - pd.Series(vw, index=f.index)) / atr_series(f).replace(0, np.nan)


# ================================================================================================ G: market structure
@njit(cache=False)
def _zigzag(high, low, atr, mult):
    n = len(high)
    cur = np.full(n, np.nan)
    prev = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    dur = np.full(n, np.nan)
    d = 0
    hi = high[0]
    lo = low[0]
    pivot = np.nan
    pivot_i = 0
    ext = np.nan
    ext_i = 0
    last_leg = np.nan
    for t in range(n):
        a = atr[t]
        if np.isnan(a):
            continue
        thr = mult * a
        if d == 0:
            if high[t] > hi:
                hi = high[t]
            if low[t] < lo:
                lo = low[t]
            if high[t] - lo >= thr and high[t] >= hi:
                d = 1
                pivot = lo
                pivot_i = t
                ext = high[t]
                ext_i = t
            elif hi - low[t] >= thr and low[t] <= lo:
                d = -1
                pivot = hi
                pivot_i = t
                ext = low[t]
                ext_i = t
            continue
        if d == 1:
            if high[t] > ext:
                ext = high[t]
                ext_i = t
            elif ext - low[t] >= thr:
                last_leg = (ext - pivot) / a
                pivot = ext
                pivot_i = ext_i
                d = -1
                ext = low[t]
                ext_i = t
        else:
            if low[t] < ext:
                ext = low[t]
                ext_i = t
            elif high[t] - ext >= thr:
                last_leg = (ext - pivot) / a
                pivot = ext
                pivot_i = ext_i
                d = 1
                ext = high[t]
                ext_i = t
        leg = ext - pivot
        cur[t] = leg / a
        prev[t] = last_leg
        ret[t] = abs(ext - (high[t] + low[t]) / 2.0) / abs(leg) if leg != 0 else 0.0
        dur[t] = t - pivot_i
    return cur, prev, ret, dur


def _zz(f, p):
    m = clamp_p(p, 1, 20, 3) / 1.0
    return _zigzag(px(f, "high").to_numpy(float), px(f, "low").to_numpy(float), atr_series(f).to_numpy(float), m)


@register("zigzag_leg_atr", 64, "ATR zigzag: signed size of the running swing leg (extreme minus last confirmed pivot) in ATR; period = reversal threshold in ATRs.",
          soft=(-30.0, 30.0), period=(1, 10))
@safe
def zigzag_leg_atr(f, p, c, l):
    return pd.Series(_zz(f, p)[0], index=f.index)


@register("zigzag_prev_leg_atr", 64, "ATR zigzag: signed size of the previous COMPLETED swing leg in ATR.", soft=(-30.0, 30.0), period=(1, 10))
@safe
def zigzag_prev_leg_atr(f, p, c, l):
    return pd.Series(_zz(f, p)[1], index=f.index)


@register("zigzag_retrace", 64, "ATR zigzag: how far the bar has retraced from the running leg's extreme, as a fraction of that leg (0 at the extreme).",
          soft=(0.0, 1.5), period=(1, 10))
@safe
def zigzag_retrace(f, p, c, l):
    return pd.Series(_zz(f, p)[2], index=f.index)


@register("zigzag_leg_bars", 64, "ATR zigzag: bars since the last confirmed pivot (duration of the running leg).", soft=(0.0, 500.0), period=(1, 10))
@safe
def zigzag_leg_bars(f, p, c, l):
    return pd.Series(_zz(f, p)[3], index=f.index)


def _day_profiles(f: pd.DataFrame, nbins: int = 40) -> pd.DataFrame:
    """Per trading day TPO profile from 30-minute periods: poc, vah, val, poor_high, poor_low, single_share (day-level, COMPLETE days)."""
    t = _ts(f)
    day = _frame_day(f)
    period = (t.dt.hour * 2 + t.dt.minute // 30).astype(int)
    h, lo = px(f, "high").to_numpy(float), px(f, "low").to_numpy(float)
    df = pd.DataFrame({"day": day.to_numpy(), "per": period.to_numpy(), "h": h, "l": lo})
    rows = []
    for d, g in df.groupby("day", sort=True):
        pg = g.groupby("per").agg(h=("h", "max"), l=("l", "min"))
        top, bot = float(pg["h"].max()), float(pg["l"].min())
        if not np.isfinite(top) or not np.isfinite(bot) or top <= bot:
            rows.append((d, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, top, bot))
            continue
        edges = np.linspace(bot, top, nbins + 1)
        cnt = np.zeros(nbins)
        i0 = np.clip(np.searchsorted(edges, pg["l"].to_numpy(), side="right") - 1, 0, nbins - 1)
        i1 = np.clip(np.searchsorted(edges, pg["h"].to_numpy(), side="right") - 1, 0, nbins - 1)
        for a, b in zip(i0, i1):
            cnt[a:b + 1] += 1
        mid = (nbins - 1) / 2.0
        poc_i = int(np.argmax(cnt - 1e-6 * np.abs(np.arange(nbins) - mid)))
        lo_i = hi_i = poc_i
        total, acc = cnt.sum(), cnt[poc_i]
        while acc < 0.7 * total and (lo_i > 0 or hi_i < nbins - 1):
            up = cnt[hi_i + 1] if hi_i < nbins - 1 else -1
            dn = cnt[lo_i - 1] if lo_i > 0 else -1
            if up >= dn:
                hi_i += 1
                acc += cnt[hi_i]
            else:
                lo_i -= 1
                acc += cnt[lo_i]
        centers = (edges[:-1] + edges[1:]) / 2.0
        rows.append((d, centers[poc_i], edges[hi_i + 1], edges[lo_i], float(cnt[-1] >= 2), float(cnt[0] >= 2), float((cnt == 1).mean()), top, bot))
    return pd.DataFrame(rows, columns=["day", "poc", "vah", "val", "poor_high", "poor_low", "single_share", "top", "bot"]).set_index("day")


def _prior_profile_map(f: pd.DataFrame, col: str) -> pd.Series:
    prof = _day_profiles(f)
    return _prior_map(f, prof[col])


@register("tpo_poc_dist", 65, "(close - PRIOR session TPO point of control) / prior daily ATR.", soft=(-8.0, 8.0), no_period=True)
@safe
def tpo_poc_dist(f, p, c, l):
    return (px(f) - _prior_profile_map(f, "poc")) / _prior_map(f, _daily(f)["atr14"]).replace(0, np.nan)


@register("tpo_va_position", 65, "Position of close in the PRIOR session's TPO value area: 0 = value-area low, 1 = value-area high (outside < 0 or > 1).",
          soft=(-3.0, 4.0), no_period=True)
@safe
def tpo_va_position(f, p, c, l):
    vah, val = _prior_profile_map(f, "vah"), _prior_profile_map(f, "val")
    return ((px(f) - val) / (vah - val).replace(0, np.nan)).clip(-5.0, 6.0)


@register("tpo_poor_high", 65, "1 on every bar of a session that follows a day with a POOR HIGH (two or more TPOs at the extreme: an unfinished auction).",
          bounded=(0.0, 1.0), no_period=True, flag=True)
@safe
def tpo_poor_high(f, p, c, l):
    return _prior_profile_map(f, "poor_high")


@register("tpo_poor_low", 65, "1 on every bar of a session that follows a day with a POOR LOW.", bounded=(0.0, 1.0), no_period=True, flag=True)
@safe
def tpo_poor_low(f, p, c, l):
    return _prior_profile_map(f, "poor_low")


@register("tpo_single_print_share", 65, "Share (0..1) of the PRIOR session's price bins printed by a single TPO period (thin, fast-travel zones).",
          bounded=(0.0, 1.0), no_period=True)
@safe
def tpo_single_print_share(f, p, c, l):
    return _prior_profile_map(f, "single_share")


def _naked(f: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    prof = _day_profiles(f)
    days = list(prof.index)
    poc, top, bot = prof["poc"].to_numpy(float), prof["top"].to_numpy(float), prof["bot"].to_numpy(float)
    day = _frame_day(f)
    close = px(f).to_numpy(float)
    pos = {d: np.where(day.to_numpy() == d)[0] for d in days}
    atr_prior = _prior_map(f, _daily(f)["atr14"]).to_numpy(float)
    dist = np.full(len(f), np.nan)
    cnt = np.full(len(f), np.nan)
    naked: list[float] = []
    for i, d in enumerate(days):
        idx = pos[d]
        if naked:
            lv = np.asarray(naked)
            cl = close[idx][:, None]
            j = np.argmin(np.abs(lv[None, :] - cl), axis=1)
            nearest = lv[j]
            dist[idx] = (nearest - close[idx]) / np.where(atr_prior[idx] > 0, atr_prior[idx], np.nan)
            cnt[idx] = len(lv)
        else:
            cnt[idx] = 0.0
        # after the day completes: it tests older naked POCs, and its own POC becomes a candidate
        if np.isfinite(top[i]):
            naked = [x for x in naked if not (bot[i] <= x <= top[i])]
        if np.isfinite(poc[i]):
            naked.append(float(poc[i]))
            naked = naked[-30:]
    return pd.Series(dist, index=f.index), pd.Series(cnt, index=f.index)


@register("naked_poc_dist", 66, "Signed distance (ATR) from close to the NEAREST untested prior-session point of control (positive = above price).",
          soft=(-15.0, 15.0), no_period=True)
@safe
def naked_poc_dist(f, p, c, l):
    return _naked(f)[0]


@register("naked_poc_count", 66, "Number of untested (naked) prior-session points of control still open (last 30 sessions).",
          soft=(0.0, 30.0), no_period=True)
@safe
def naked_poc_count(f, p, c, l):
    return _naked(f)[1]


def _period_table(f: pd.DataFrame, freq: str) -> tuple[pd.Series, pd.DataFrame]:
    key = _frame_day(f).dt.to_period(freq).astype(str)
    g = pd.DataFrame({"k": key, "o": px(f, "open"), "h": px(f, "high"), "l": px(f, "low"), "c": px(f)}).groupby("k", sort=True)
    tab = pd.DataFrame({"open": g["o"].first(), "high": g["h"].max(), "low": g["l"].min(), "close": g["c"].last()})
    return key, tab


def _level_dist(f: pd.DataFrame, freq: str, which: str, prior: bool) -> pd.Series:
    key, tab = _period_table(f, freq)
    col = tab[which].shift(1) if prior else tab[which]
    lvl = key.map(col)
    return (px(f) - lvl) / _prior_map(f, _daily(f)["atr14"]).replace(0, np.nan)


def _reg_level(kind: str, freq: str, which: str, prior: bool, desc: str):
    @register(kind, 67, desc, soft=(-30.0, 30.0), no_period=True)
    @safe
    def _fn(f, p, c, l):
        return _level_dist(f, freq, which, prior)
    _fn.__name__ = kind
    return _fn


for _k, _fq, _w, _pr, _d in (
    ("wk_open_dist", "W", "open", False, "(close - this week's open) / prior daily ATR."),
    ("pwk_high_dist", "W", "high", True, "(close - PRIOR week's high) / prior daily ATR."),
    ("pwk_low_dist", "W", "low", True, "(close - PRIOR week's low) / prior daily ATR."),
    ("pwk_close_dist", "W", "close", True, "(close - PRIOR week's close) / prior daily ATR."),
    ("mo_open_dist", "M", "open", False, "(close - this month's open) / prior daily ATR."),
    ("pmo_high_dist", "M", "high", True, "(close - PRIOR month's high) / prior daily ATR."),
    ("pmo_low_dist", "M", "low", True, "(close - PRIOR month's low) / prior daily ATR."),
    ("pmo_close_dist", "M", "close", True, "(close - PRIOR month's close) / prior daily ATR."),
):
    _reg_level(_k, _fq, _w, _pr, _d)


@njit(cache=False)
def _equal_density(high, low, atr, k, win, tol, want_high):
    n = len(high)
    out = np.full(n, np.nan)
    lvl = np.full(n, np.nan)       # confirmed swing level at its confirmation bar
    for i in range(k, n - k):
        ok = True
        for j in range(1, k + 1):
            if want_high:
                if high[i] <= high[i - j] or high[i] < high[i + j]:
                    ok = False
                    break
            else:
                if low[i] >= low[i - j] or low[i] > low[i + j]:
                    ok = False
                    break
        if ok:
            lvl[i + k] = high[i] if want_high else low[i]
    for t in range(win, n):
        if np.isnan(atr[t]):
            continue
        best = np.nan
        for s in range(t - win + 1, t + 1):
            v = lvl[s]
            if np.isnan(v):
                continue
            if np.isnan(best) or (want_high and v > best) or ((not want_high) and v < best):
                best = v
        if np.isnan(best):
            out[t] = 0.0
            continue
        c = 0
        for s in range(t - win + 1, t + 1):
            v = lvl[s]
            if not np.isnan(v) and abs(v - best) <= tol * atr[t]:
                c += 1
        out[t] = c
    return out


@register("equal_highs_density", 68, "Count of confirmed swing highs within 0.15 ATR of the highest swing high in the last N bars (clustered resting liquidity above).",
          soft=(0.0, 6.0), period=(10, 300))
@safe
def equal_highs_density(f, p, c, l):
    w = clamp_p(p, 10, 600, 60)
    return pd.Series(_equal_density(px(f, "high").to_numpy(float), px(f, "low").to_numpy(float), atr_series(f).to_numpy(float), 2, w, 0.15, True), index=f.index)


@register("equal_lows_density", 68, "Count of confirmed swing lows within 0.15 ATR of the lowest swing low in the last N bars (resting liquidity below).",
          soft=(0.0, 6.0), period=(10, 300))
@safe
def equal_lows_density(f, p, c, l):
    w = clamp_p(p, 10, 600, 60)
    return pd.Series(_equal_density(px(f, "high").to_numpy(float), px(f, "low").to_numpy(float), atr_series(f).to_numpy(float), 2, w, 0.15, False), index=f.index)


def _pd_pos(f: pd.DataFrame, k: int) -> pd.Series:
    h, lo = px(f, "high"), px(f, "low")
    win = 2 * k + 1
    sh = h.where(h == h.rolling(win, center=True, min_periods=win).max()).shift(k).ffill()
    sl = lo.where(lo == lo.rolling(win, center=True, min_periods=win).min()).shift(k).ffill()
    rng = (sh - sl).where(sh > sl)
    return ((px(f) - sl) / rng.replace(0, np.nan)).clip(-0.5, 1.5)


@register("premium_discount", 69, "Position of close between the last confirmed swing low (0 = discount) and swing high (1 = premium); period = pivot width.",
          soft=(-0.5, 1.5), period=(2, 60))
@safe
def premium_discount(f, p, c, l):
    return _pd_pos(f, clamp_p(p, 2, 100, 5))


@register("premium_discount_multi", 69, "Mean premium/discount position across three swing scales (pivot widths N/2, N, 3N).",
          soft=(-0.5, 1.5), period=(4, 60))
@safe
def premium_discount_multi(f, p, c, l):
    n = clamp_p(p, 4, 100, 10)
    return pd.concat([_pd_pos(f, max(2, n // 2)), _pd_pos(f, n), _pd_pos(f, 3 * n)], axis=1).mean(axis=1, skipna=False)


def _gap_state(f: pd.DataFrame):
    d = _daily(f)
    day = _frame_day(f)
    op = f["open"].astype(float).groupby(day).transform("first")
    prev_close = _prior_map(f, d["close"])
    gap = op - prev_close
    run_lo = px(f, "low").groupby(day).cummin()
    run_hi = px(f, "high").groupby(day).cummax()
    up = gap > 0
    filled = np.where(up, (op - run_lo) / gap.where(gap != 0), (run_hi - op) / (-gap).where(gap != 0))
    t = _ts(f)
    mins = (t - t.groupby(day).transform("first")).dt.total_seconds() / 60.0
    return gap, pd.Series(filled, index=f.index), mins


@register("gap_fill_progress", 70, "Fraction (0..1) of the opening gap closed so far today (NaN on days without a gap).", bounded=(0.0, 1.0), no_period=True)
@safe
def gap_fill_progress(f, p, c, l):
    gap, filled, _ = _gap_state(f)
    return filled.clip(0.0, 1.0).where(gap.abs() > 0)


@register("gap_fill_minutes", 70, "Minutes since today's open on days that opened with a gap (NaN otherwise): how long the gap has had to fill.",
          soft=(0.0, 1500.0), no_period=True)
@safe
def gap_fill_minutes(f, p, c, l):
    gap, _, mins = _gap_state(f)
    return mins.where(gap.abs() > 0)


def _hl(f):
    return (px(f, "high") - px(f, "low")).replace(0, np.nan)


@register("body_ratio", 71, "Candle body / range (0..1).", bounded=(0.0, 1.0), no_period=True)
@safe
def body_ratio(f, p, c, l):
    return ((px(f) - px(f, "open")).abs() / _hl(f)).clip(0.0, 1.0)


@register("upper_wick_ratio", 71, "Upper wick / range (0..1).", bounded=(0.0, 1.0), no_period=True)
@safe
def upper_wick_ratio(f, p, c, l):
    return ((px(f, "high") - pd.concat([px(f), px(f, "open")], axis=1).max(axis=1)) / _hl(f)).clip(0.0, 1.0)


@register("lower_wick_ratio", 71, "Lower wick / range (0..1).", bounded=(0.0, 1.0), no_period=True)
@safe
def lower_wick_ratio(f, p, c, l):
    return ((pd.concat([px(f), px(f, "open")], axis=1).min(axis=1) - px(f, "low")) / _hl(f)).clip(0.0, 1.0)


@register("close_location_value", 71, "Close location value ((C-L)-(H-C))/(H-L), -1..1.", bounded=(-1.0, 1.0), no_period=True)
@safe
def close_location_value(f, p, c, l):
    return ((((px(f) - px(f, "low")) - (px(f, "high") - px(f))) / _hl(f))).clip(-1.0, 1.0)


@njit(cache=False)
def _box_touch(high, low, win):
    n = len(high)
    out = np.full(n, np.nan)
    for t in range(win - 1, n):
        top = -1e300
        bot = 1e300
        for k in range(t - win + 1, t + 1):
            top = max(top, high[k])
            bot = min(bot, low[k])
        z = 0.1 * (top - bot)
        a = 0
        b = 0
        for k in range(t - win + 1, t + 1):
            if high[k] >= top - z:
                a += 1
            if low[k] <= bot + z:
                b += 1
        out[t] = min(a, b)
    return out


@register("box_tightness", 72, "Consolidation box height (N-bar high - low) in ATR(14): small = tight box.", soft=(0.0, 30.0), period=(8, 200))
@safe
def box_tightness(f, p, c, l):
    n = clamp_p(p, 5, 400, 30)
    return (px(f, "high").rolling(n, min_periods=n).max() - px(f, "low").rolling(n, min_periods=n).min()) / atr_series(f).replace(0, np.nan)


@register("box_touches", 72, "Edge touches of the N-bar box: min(top-zone touches, bottom-zone touches), zones = outer 10% of the box.",
          soft=(0.0, 20.0), period=(8, 200))
@safe
def box_touches(f, p, c, l):
    n = clamp_p(p, 5, 400, 30)
    return pd.Series(_box_touch(px(f, "high").to_numpy(float), px(f, "low").to_numpy(float), n), index=f.index)


@register("box_duration", 72, "Consecutive bars for which the N-bar box has stayed within 3 ATR(14) (age of the consolidation).",
          soft=(0.0, 2000.0), period=(8, 200))
@safe
def box_duration(f, p, c, l):
    n = clamp_p(p, 5, 400, 30)
    ok = box_tightness(f, n, "close", None) <= 3.0
    grp = (~ok).cumsum()
    return ok.astype(float).groupby(grp).cumsum().where(ok.notna())


@njit(cache=False)
def _theil_sen(y, win, atr):
    n = len(y)
    out = np.full(n, np.nan)
    m = win * (win - 1) // 2
    slopes = np.empty(m)
    resid = np.empty(win)
    for t in range(win - 1, n):
        c = 0
        bad = False
        for i in range(win - 1):
            yi = y[t - win + 1 + i]
            if np.isnan(yi):
                bad = True
                break
            for j in range(i + 1, win):
                slopes[c] = (y[t - win + 1 + j] - yi) / (j - i)
                c += 1
        if bad or np.isnan(y[t]) or np.isnan(atr[t]) or atr[t] <= 0:
            continue
        b = np.median(slopes[:c])
        for i in range(win):
            resid[i] = y[t - win + 1 + i] - b * i
        a = np.median(resid)
        out[t] = (y[t] - (a + b * (win - 1))) / atr[t]
    return out


@register("theil_sen_dist", 73, "Distance (ATR) of close from the Theil-Sen robust trendline fitted over the last N bars (N <= 40).",
          soft=(-6.0, 6.0), period=(6, 40))
@safe
def theil_sen_dist(f, p, c, l):
    n = clamp_p(p, 6, 40, 20)
    return pd.Series(_theil_sen(px(f).to_numpy(float), n, atr_series(f, n).to_numpy(float)), index=f.index)


# ============================================================================ H: cross-asset, macro and positioning
@register("rolling_beta", 74, "Rolling beta of own returns on the benchmark (frame.attrs['benchmark']) over 3N bars.", soft=(-3.0, 3.0), period=(10, 200))
@safe
def rolling_beta(f, p, c, l):
    bench = _attr_frame(f, "benchmark")
    if bench is None:
        return nan_like(f)
    n = clamp_p(p, 10, 400, 30)
    r, rb = log_ret(f), np.log(_align_companion(f, bench)).diff()
    return r.rolling(3 * n, min_periods=3 * n).cov(rb) / rb.rolling(3 * n, min_periods=3 * n).var().replace(0, np.nan)


_mark("rolling_beta", companion=True)


@register("resid_return", 74, "Benchmark-neutral return over N bars (percent): sum of r - beta*r_bench, beta estimated over 3N bars.",
          soft=(-5.0, 5.0), period=(5, 200))
@safe
def resid_return(f, p, c, l):
    bench = _attr_frame(f, "benchmark")
    if bench is None:
        return nan_like(f)
    n = clamp_p(p, 5, 400, 20)
    r, rb = log_ret(f), np.log(_align_companion(f, bench)).diff()
    beta = r.rolling(3 * n, min_periods=3 * n).cov(rb) / rb.rolling(3 * n, min_periods=3 * n).var().replace(0, np.nan)
    return (r - beta.shift(1) * rb).rolling(n, min_periods=n).sum() * 100.0


_mark("resid_return", companion=True)


@register("absorption_ratio", 75, "Absorption ratio: share of basket variance carried by the first principal component of the last N bars (0..1; frame.attrs['basket']).",
          bounded=(0.0, 1.0), period=(30, 500))
@safe
def absorption_ratio(f, p, c, l):
    cl = _basket_closes(f)
    if cl is None:
        return nan_like(f)
    n = clamp_p(p, 30, 1000, 60)
    r = np.column_stack([log_ret(f).to_numpy(float), np.log(cl).diff().to_numpy(float)])

    def at(t):
        w = r[t - n + 1: t + 1]
        if not np.isfinite(w).all() or (w.std(axis=0) == 0).any():
            return np.nan
        ev = np.linalg.eigvalsh(np.corrcoef(w, rowvar=False))
        return float(ev[-1] / ev.sum())

    return pd.Series(strided(len(f), max(1, n // 6), n, at), index=f.index)


_mark("absorption_ratio", companion=True)


@register("xs_momentum_rank", 76, "Cross-sectional momentum: percentile rank (0..100) of own N-bar return among the basket (frame.attrs['basket']).",
          bounded=(0.0, 100.0), period=(5, 500))
@safe
def xs_momentum_rank(f, p, c, l):
    cl = _basket_closes(f)
    if cl is None:
        return nan_like(f)
    n = clamp_p(p, 2, 1000, 20)
    mine = px(f).pct_change(n)
    others = cl.pct_change(n)
    allr = pd.concat([mine.rename("__own__"), others], axis=1)
    rank = allr.rank(axis=1, pct=True)["__own__"]
    return (rank * 100.0).where(allr.notna().all(axis=1))


_mark("xs_momentum_rank", companion=True)


def _ext_chg(f, feed: str, n: int, transform) -> pd.Series:
    from app.data import macro_data as md
    s = md.ext_series(f, feed)
    if not len(s):
        return nan_like(f)
    return pd.Series(md.ext_aligned(f, feed, transform(s, n).dropna()), index=f.index)


def _reg_ext(kind: str, number: int, feed: str, desc: str, transform, *, soft=None, bounded=None, period=None, no_period=False, tied=False):
    @register(kind, number, desc, soft=soft, bounded=bounded, period=period, no_period=no_period)
    @safe
    def _fn(f, p, c, l):
        n = clamp_p(p, 1, 2000, (period[0] if period else 1) if period else 1)
        return _ext_chg(f, feed, n, transform)
    _fn.__name__ = kind
    _mark(kind, ext=True, tied=tied)
    return _fn


def _level(s, n):
    return s


def _diff(s, n):
    return s.diff(n)


def _pct_rank(s, n):
    n = max(int(n), 20)
    return s.rolling(n, min_periods=max(10, n // 4)).apply(lambda w: 100.0 * float((w[:-1] < w[-1]).mean()) if len(w) > 1 else np.nan, raw=True)


def _zscore(s, n):
    n = max(int(n), 10)
    return (s - s.rolling(n, min_periods=max(5, n // 4)).mean()) / s.rolling(n, min_periods=max(5, n // 4)).std().replace(0, np.nan)


_reg_ext("hy_spread", 77, "hy_oas", "High-yield OAS level (percent), known with a 2-day lag (FRED BAMLH0A0HYM2).", _level, soft=(1.0, 20.0), no_period=True)
_reg_ext("hy_spread_chg", 77, "hy_oas", "High-yield OAS change over N daily observations (percentage points).", _diff, soft=(-3.0, 5.0), period=(1, 120))
_reg_ext("real_yield_chg", 78, "real_yield_10y", "10-year real-yield change over N daily observations (percentage points).", _diff, soft=(-1.0, 1.0), period=(1, 120))
_reg_ext("breakeven_chg", 79, "breakeven_10y", "10-year breakeven-inflation change over N daily observations (percentage points).", _diff, soft=(-1.0, 1.0), period=(1, 120))
_reg_ext("move_pct", 80, "move", "MOVE index percentile (0..100) within the last N observations.", _pct_rank, bounded=(0.0, 100.0), period=(60, 1000))
_reg_ext("vvix_pct", 81, "vvix", "VVIX percentile (0..100) within the last N observations.", _pct_rank, bounded=(0.0, 100.0), period=(60, 1000))
_reg_ext("skew_pct", 81, "skew", "CBOE SKEW percentile (0..100) within the last N observations.", _pct_rank, bounded=(0.0, 100.0), period=(60, 1000))
_reg_ext("term_structure_slope", 84, "term_slope", "Futures term-structure slope (next/front - 1): >0 contango, <0 backwardation (per-instrument feed).",
         _level, soft=(-0.2, 0.2), no_period=True, tied=True)
_reg_ext("cot_net_spec_pct", 85, "cot_net_spec", "COT net speculator position percentile (0..100) within the last N weekly reports; known only after publication (4-day lag).",
         _pct_rank, bounded=(0.0, 100.0), period=(26, 520), tied=True)
_reg_ext("putcall_z", 86, "putcall", "Put/call ratio z-score over the last N observations.", _zscore, soft=(-4.0, 4.0), period=(20, 500))
_reg_ext("econ_surprise_chg", 87, "econ_surprise", "Economic surprise index change over N daily observations.", _diff, soft=(-100.0, 100.0), period=(1, 120))
_reg_ext("rate_expect_shift", 88, "fed_funds_path", "Policy-rate expectation shift (fed funds futures path change around meetings), as supplied by the feed.",
         _level, soft=(-1.0, 1.0), no_period=True)
_reg_ext("fx_carry_diff", 89, "rate_diff", "FX carry: interest-rate differential (base minus quote) for the pair, percent (per-pair feed).",
         _level, soft=(-10.0, 10.0), no_period=True, tied=True)
_reg_ext("fx_carry_chg", 89, "rate_diff", "FX carry: change of the rate differential over N daily observations.", _diff, soft=(-3.0, 3.0), period=(1, 250), tied=True)
_reg_ext("funding_rate_z", 90, "funding_rate", "Perpetual funding-rate z-score over the last N settlements (crowded leverage), dated at settlement.",
         _zscore, soft=(-4.0, 4.0), period=(10, 500), tied=True)
_reg_ext("perp_spot_basis", 92, "perp_premium", "Perp-minus-spot premium, basis points (leverage demand; converges when leverage unwinds).",
         lambda s, n: s * 1e4, soft=(-100.0, 200.0), no_period=True, tied=True)


def _macro13(f, name: str) -> pd.Series:
    m = frame_attr(f, "macro")
    if isinstance(m, dict) and name in m:
        return m[name]
    from app.data.macro_data import load_series
    return load_series(name)


@register("vrp", 82, "Variance risk premium: prior-day VIX minus the instrument's 21-day realized volatility (annualized percent); Ext (VIX).",
          soft=(-30.0, 30.0), period=(5, 60))
@safe
def vrp(f, p, c, l):
    v11 = _v11()
    s = _macro13(f, "VIX")
    if not len(s):
        return nan_like(f)
    n = clamp_p(p, 5, 120, 21)
    d = _daily(f)
    rv = np.log(d["close"]).diff().rolling(n, min_periods=n).std() * np.sqrt(252.0) * 100.0
    return pd.Series(v11._prior(f, s), index=f.index) - _prior_map(f, rv)


_mark("vrp", ext=True)


def _leg(f: pd.DataFrame, key: str) -> pd.Series | None:
    aliases = {"copper": ("HG", "COPPER", "XCUUSD"), "gold": ("GC", "MGC", "XAUUSD", "GOLD"), "silver": ("SI", "XAGUSD", "SILVER"), "oil": ("CL", "MCL", "OIL", "WTI", "USOIL")}
    other = _attr_frame(f, key)
    if other is not None:
        return _align_companion(f, other)
    sym = ""
    attrs = getattr(f, "attrs", {}) or {}
    for k in ("symbol", "instrument", "market", "name"):
        if attrs.get(k):
            sym = str(attrs[k]).upper()
            break
    if sym and any(sym.startswith(a) for a in aliases[key]):
        return px(f)
    return None


def _reg_ratio(kind: str, num: str, den: str, desc: str):
    @register(kind, 83, desc, soft=(-4.0, 4.0), period=(10, 1000))
    @safe
    def _fn(f, p, c, l):
        a, b = _leg(f, num), _leg(f, den)
        if a is None or b is None:
            return nan_like(f)
        n = clamp_p(p, 10, 2000, 100)
        r = np.log(a / b)
        return (r - r.rolling(n, min_periods=n).mean()) / r.rolling(n, min_periods=n).std().replace(0, np.nan)
    _fn.__name__ = kind
    _mark(kind, companion=True)
    return _fn


_reg_ratio("ratio_copper_gold", "copper", "gold", "Copper/gold ratio z-score over N bars (companion frames or the frame itself).")
_reg_ratio("ratio_gold_silver", "gold", "silver", "Gold/silver ratio z-score over N bars.")
_reg_ratio("ratio_oil_gold", "oil", "gold", "Oil/gold ratio z-score over N bars.")


@register("oi_liq_proxy", 91, "Liquidation proxy: open-interest DROP over N samples (percent) times the bar's range in ATR(20); large = forced selling/covering.",
          soft=(0.0, 100.0), period=(1, 60))
@safe
def oi_liq_proxy(f, p, c, l):
    from app.data import macro_data as md
    s = md.ext_series(f, "crypto_oi")
    if not len(s):
        return nan_like(f)
    n = clamp_p(p, 1, 200, 4)
    chg = pd.Series(md.ext_aligned(f, "crypto_oi", (s.pct_change(n) * 100.0).dropna()), index=f.index)
    rng = (px(f, "high") - px(f, "low")) / atr_series(f, 20).replace(0, np.nan)
    return (-chg).clip(lower=0.0) * rng


_mark("oi_liq_proxy", ext=True, tied=True)


# ===================================================================================================== J: seasonality
def _expanding_t(r: pd.Series, key: pd.Series, min_n: int) -> pd.Series:
    """t-statistic of the expanding mean of ``r`` within ``key`` using STRICTLY EARLIER occurrences only."""
    r = r.fillna(0.0)
    g = r.groupby(key, sort=False)
    n = g.cumcount().astype(float)
    csum = g.cumsum() - r
    csq = (r ** 2).groupby(key, sort=False).cumsum() - r ** 2
    mean = csum / n.replace(0, np.nan)
    var = (csq - n * mean ** 2) / (n - 1).replace(0, np.nan)
    t = mean / np.sqrt((var / n).replace(0, np.nan))
    return t.where(n >= min_n)


@register("seas_tod", 93, "Causal time-of-day seasonality: t-statistic of the expanding mean 1-bar return of THIS clock minute, earlier occurrences only (>= 20).",
          soft=(-5.0, 5.0), no_period=True)
@safe
def seas_tod(f, p, c, l):
    v11 = _v11()
    key = v11._minute_of_day(_ts(f))
    return _expanding_t(log_ret(f) * 1e4, key, 20)


def _daily_seasonal(f: pd.DataFrame, keyfn, min_n: int) -> pd.Series:
    d = _daily(f)
    r = np.log(d["close"]).diff() * 1e4
    key = keyfn(d.index)
    t = _expanding_t(r, pd.Series(key, index=d.index), min_n)
    return pd.Series(_frame_day(f).map(t).to_numpy(dtype=float), index=f.index)


@register("seas_dow", 93, "Causal day-of-week seasonality: t-statistic of the expanding mean daily return of this weekday (earlier days only, >= 20).",
          soft=(-5.0, 5.0), no_period=True)
@safe
def seas_dow(f, p, c, l):
    return _daily_seasonal(f, lambda ix: pd.DatetimeIndex(ix).day_of_week.to_numpy(), 20)


@register("seas_dom", 93, "Causal day-of-month seasonality: t-statistic of the expanding mean daily return of this calendar day (earlier days only, >= 10).",
          soft=(-5.0, 5.0), no_period=True)
@safe
def seas_dom(f, p, c, l):
    return _daily_seasonal(f, lambda ix: pd.DatetimeIndex(ix).day.to_numpy(), 10)


@register("seas_woy", 93, "Causal week-of-year seasonality: t-statistic of the expanding mean daily return of this ISO week (earlier days only, >= 10).",
          soft=(-5.0, 5.0), no_period=True)
@safe
def seas_woy(f, p, c, l):
    return _daily_seasonal(f, lambda ix: pd.DatetimeIndex(ix).isocalendar().week.to_numpy().astype(int), 10)


def _sessions(f: pd.DataFrame):
    t = _ts(f)
    m = t.dt.hour * 60 + t.dt.minute
    tday = (t + pd.Timedelta(hours=6)).dt.normalize()
    sess = pd.Series(np.where((m >= 18 * 60) | (m < 3 * 60), 0, np.where(m < 8 * 60, 1, 2)), index=f.index)
    return tday, sess


def _sess_running(f: pd.DataFrame):
    tday, sess = _sessions(f)
    h, lo = px(f, "high"), px(f, "low")
    keys = [tday, sess]
    return tday, sess, h.groupby(keys).cummax(), lo.groupby(keys).cummin()


def _complete(f, tday, sess, s: int, rh, rl):
    """High/low of the COMPLETE session ``s`` of each trading day, visible only to bars of LATER sessions."""
    hi = rh.where(sess == s).groupby(tday).transform("last")     # groupby-last skips NaN: the final bar of session ``s``
    lo = rl.where(sess == s).groupby(tday).transform("last")
    return hi.where(sess > s), lo.where(sess > s)


def _handoff(f):
    tday, sess, rh, rl = _sess_running(f)
    ah, al = _complete(f, tday, sess, 0, rh, rl)
    lh, ll = _complete(f, tday, sess, 1, rh, rl)
    return tday, sess, rh, rl, ah, al, lh, ll


@register("handoff_london_asia", 94, "London session range to date (or complete) divided by the COMPLETE Asia session range of the same trading day (exchange clock).",
          soft=(0.0, 5.0), no_period=True)
@safe
def handoff_london_asia(f, p, c, l):
    tday, sess, rh, rl, ah, al, _lh, _ll = _handoff(f)
    return ((rh - rl) / (ah - al).replace(0, np.nan)).where(sess == 1)


@register("handoff_ny_london", 94, "New York range to date divided by the COMPLETE London session range of the same trading day.",
          soft=(0.0, 5.0), no_period=True)
@safe
def handoff_ny_london(f, p, c, l):
    tday, sess, rh, rl, _ah, _al, lh, ll = _handoff(f)
    return ((rh - rl) / (lh - ll).replace(0, np.nan)).where(sess == 2)


@register("handoff_pos_asia", 94, "Close position (0..1, outside = beyond) within the Asia session range (running during Asia, complete afterwards).",
          soft=(-1.0, 2.0), no_period=True)
@safe
def handoff_pos_asia(f, p, c, l):
    tday, sess, rh, rl, ah, al, _lh, _ll = _handoff(f)
    hi, lo = ah.where(sess > 0, rh), al.where(sess > 0, rl)
    return ((px(f) - lo) / (hi - lo).replace(0, np.nan)).clip(-3.0, 4.0)


@register("handoff_pos_london", 94, "Close position (0..1, outside = beyond) within the London session range (NaN during Asia).",
          soft=(-1.0, 2.0), no_period=True)
@safe
def handoff_pos_london(f, p, c, l):
    tday, sess, rh, rl, _ah, _al, lh, ll = _handoff(f)
    hi, lo = lh.where(sess > 1, rh), ll.where(sess > 1, rl)
    return ((px(f) - lo) / (hi - lo).replace(0, np.nan)).clip(-3.0, 4.0).where(sess >= 1)


# ================================================================================================= K: regime models
@njit(cache=False)
def _hmm_em(x, fit_means, iters):
    """EM for a 2-state Gaussian HMM on the 1-D series ``x``. Returns (mu0, mu1, v0, v1, a00, a11); state 1 is the higher mean (or variance).
    Scalar arithmetic only in the inner loops (small-array allocation per step made the earlier version ~100x slower)."""
    n = len(x)
    s = np.std(x)
    if s <= 0:
        return 0.0, 0.0, 1.0, 1.0, 0.95, 0.95
    if fit_means:
        mu0, mu1 = -0.1 * s, 0.1 * s
        v0, v1 = s * s, s * s
    else:
        mu0, mu1 = 0.0, 0.0
        v0, v1 = s * s * 0.5, s * s * 1.8
    a00, a01, a10, a11 = 0.95, 0.05, 0.05, 0.95
    pi1 = 0.5
    al0 = np.zeros(n)
    al1 = np.zeros(n)
    be0 = np.zeros(n)
    be1 = np.zeros(n)
    sc = np.zeros(n)
    b0 = np.zeros(n)
    b1 = np.zeros(n)
    floor_v = 1e-10 * s * s + 1e-18
    for _ in range(iters):
        for t in range(n):
            b0[t] = np.exp(-0.5 * (x[t] - mu0) ** 2 / v0) / np.sqrt(2 * np.pi * v0) + 1e-300
            b1[t] = np.exp(-0.5 * (x[t] - mu1) ** 2 / v1) / np.sqrt(2 * np.pi * v1) + 1e-300
        al0[0] = (1.0 - pi1) * b0[0]
        al1[0] = pi1 * b1[0]
        sc[0] = al0[0] + al1[0]
        al0[0] /= sc[0]
        al1[0] /= sc[0]
        for t in range(1, n):
            n0 = (al0[t - 1] * a00 + al1[t - 1] * a10) * b0[t]
            n1 = (al0[t - 1] * a01 + al1[t - 1] * a11) * b1[t]
            sc[t] = n0 + n1
            al0[t] = n0 / sc[t]
            al1[t] = n1 / sc[t]
        be0[n - 1] = 1.0
        be1[n - 1] = 1.0
        for t in range(n - 2, -1, -1):
            be0[t] = (a00 * b0[t + 1] * be0[t + 1] + a01 * b1[t + 1] * be1[t + 1]) / sc[t + 1]
            be1[t] = (a10 * b0[t + 1] * be0[t + 1] + a11 * b1[t + 1] * be1[t + 1]) / sc[t + 1]
        x00 = 0.0
        x01 = 0.0
        x10 = 0.0
        x11 = 0.0
        w0 = 0.0
        w1 = 0.0
        m0 = 0.0
        m1 = 0.0
        g00 = 0.0
        g10 = 0.0
        for t in range(n):
            g0 = al0[t] * be0[t]
            g1 = al1[t] * be1[t]
            tot_g = g0 + g1
            g0 /= tot_g
            g1 /= tot_g
            if t == 0:
                g00, g10 = g0, g1
            w0 += g0
            w1 += g1
            m0 += g0 * x[t]
            m1 += g1 * x[t]
            if t < n - 1:
                t00 = al0[t] * a00 * b0[t + 1] * be0[t + 1]
                t01 = al0[t] * a01 * b1[t + 1] * be1[t + 1]
                t10 = al1[t] * a10 * b0[t + 1] * be0[t + 1]
                t11 = al1[t] * a11 * b1[t + 1] * be1[t + 1]
                tt = t00 + t01 + t10 + t11
                if tt > 0:
                    x00 += t00 / tt
                    x01 += t01 / tt
                    x10 += t10 / tt
                    x11 += t11 / tt
        r0 = x00 + x01
        r1 = x10 + x11
        if r0 > 0:
            a00, a01 = x00 / r0, x01 / r0
        if r1 > 0:
            a10, a11 = x10 / r1, x11 / r1
        pi1 = g10
        if w0 > 1e-9:
            if fit_means:
                mu0 = m0 / w0
            q = 0.0
            for t in range(n):
                g0 = al0[t] * be0[t]
                g1 = al1[t] * be1[t]
                q += (g0 / (g0 + g1)) * (x[t] - mu0) ** 2
            v0 = max(q / w0, floor_v)
        if w1 > 1e-9:
            if fit_means:
                mu1 = m1 / w1
            q = 0.0
            for t in range(n):
                g0 = al0[t] * be0[t]
                g1 = al1[t] * be1[t]
                q += (g1 / (g0 + g1)) * (x[t] - mu1) ** 2
            v1 = max(q / w1, floor_v)
    if (fit_means and mu0 > mu1) or ((not fit_means) and v0 > v1):
        return mu1, mu0, v1, v0, a11, a00
    return mu0, mu1, v0, v1, a00, a11


@njit(cache=False)
def _hmm_filter(x, start, end, prm, state):
    """Forward-filter x[start:end] with parameters ``prm`` = (mu0, mu1, v0, v1, a00, a11); ``state`` = P(state 1) carried across segments."""
    out = np.full(end - start, np.nan)
    p1 = state
    for t in range(start, end):
        if np.isnan(x[t]):
            out[t - start] = p1
            continue
        pr1 = p1 * prm[5] + (1.0 - p1) * (1.0 - prm[4])
        pr0 = 1.0 - pr1
        l0 = np.exp(-0.5 * (x[t] - prm[0]) ** 2 / prm[2]) / np.sqrt(2 * np.pi * prm[2]) + 1e-300
        l1 = np.exp(-0.5 * (x[t] - prm[1]) ** 2 / prm[3]) / np.sqrt(2 * np.pi * prm[3]) + 1e-300
        p1 = pr1 * l1 / (pr1 * l1 + pr0 * l0)
        out[t - start] = p1
    return out


def _hmm_prob(f: pd.DataFrame, fit_means: bool, refit: int, window: int) -> pd.Series:
    r = (log_ret(f) * 100.0).to_numpy(float)
    n = len(r)
    out = np.full(n, np.nan)
    first = max(window // 2, 200)
    if n <= first:
        return pd.Series(out, index=f.index)
    state = 0.5
    t = first
    while t < n:
        lo = max(1, t - window)
        seg = r[lo:t]                                     # data STRICTLY BEFORE the refit bar
        seg = seg[np.isfinite(seg)]
        if len(seg) < 100:
            t += refit
            continue
        prm = np.array(_hmm_em(seg.astype(np.float64), fit_means, 15), dtype=np.float64)
        end = min(n, t + refit)
        out[t:end] = _hmm_filter(r, t, end, prm, state)
        state = out[end - 1] if np.isfinite(out[end - 1]) else state
        t = end
    return pd.Series(out, index=f.index)


@register("hmm_bull_prob", 95, "HMM forward-filtered probability of the high-mean (bull) state of a 2-state Gaussian HMM, refit every N bars on a trailing window (never smoothed).",
          bounded=(0.0, 1.0), period=(100, 2000))
@safe
def hmm_bull_prob(f, p, c, l):
    n = clamp_p(p, 100, 5000, 500)
    return _hmm_prob(f, True, n, min(5000, 10 * n))


@register("ms_var_prob", 96, "Markov-switching variance: filtered probability of the HIGH-volatility state (zero-mean 2-state HMM), refit every N bars on a trailing window.",
          bounded=(0.0, 1.0), period=(100, 2000))
@safe
def ms_var_prob(f, p, c, l):
    n = clamp_p(p, 100, 5000, 500)
    return _hmm_prob(f, False, n, min(5000, 10 * n))


@njit(cache=False)
def _bocpd(z, hazard, rmax, k):
    n = len(z)
    out = np.full(n, np.nan)
    R = np.zeros(rmax + 1)
    R[0] = 1.0
    mu = np.zeros(rmax + 1)
    kap = np.ones(rmax + 1)
    alp = np.ones(rmax + 1)
    bet = np.ones(rmax + 1)
    for t in range(n):
        x = z[t]
        if np.isnan(x):
            continue
        pred = np.zeros(rmax + 1)
        for r in range(rmax + 1):
            if R[r] < 1e-300:
                continue
            nu = 2.0 * alp[r]
            sc2 = bet[r] * (kap[r] + 1.0) / (alp[r] * kap[r])
            lp = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * np.log(nu * np.pi * sc2) - (nu + 1) / 2 * np.log(1 + (x - mu[r]) ** 2 / (nu * sc2))
            pred[r] = np.exp(lp)
        newR = np.zeros(rmax + 1)
        cp = 0.0
        for r in range(rmax + 1):
            g = R[r] * pred[r]
            cp += g * hazard
            if r < rmax:
                newR[r + 1] += g * (1.0 - hazard)
            else:
                newR[r] += g * (1.0 - hazard)
        newR[0] = cp
        s = newR.sum()
        if s <= 0:
            newR[:] = 0.0
            newR[0] = 1.0
        else:
            newR /= s
        nmu = np.zeros(rmax + 1)
        nka = np.ones(rmax + 1)
        nal = np.ones(rmax + 1)
        nbe = np.ones(rmax + 1)
        for r in range(rmax):
            nmu[r + 1] = (kap[r] * mu[r] + x) / (kap[r] + 1.0)
            nka[r + 1] = kap[r] + 1.0
            nal[r + 1] = alp[r] + 0.5
            nbe[r + 1] = bet[r] + kap[r] * (x - mu[r]) ** 2 / (2.0 * (kap[r] + 1.0))
        # keep the tail bucket's statistics from growing without bound
        nmu[rmax] = mu[rmax]
        nka[rmax] = kap[rmax]
        nal[rmax] = alp[rmax]
        nbe[rmax] = bet[rmax]
        R, mu, kap, alp, bet = newR, nmu, nka, nal, nbe
        c = 0.0
        for r in range(k):
            c += R[r]
        out[t] = c
    return out


@register("bocpd_prob", 97, "Bayesian online change-point probability: posterior mass on run lengths < N (a new regime began in the last N bars), hazard 1/250.",
          bounded=(0.0, 1.0), period=(2, 20))
@safe
def bocpd_prob(f, p, c, l):
    k = clamp_p(p, 2, 40, 5)
    r = log_ret(f)
    sd = np.sqrt((r ** 2).ewm(span=100, adjust=False, min_periods=30).mean().shift(1))      # scale known BEFORE the bar
    z = (r / sd.replace(0, np.nan)).to_numpy(float)
    return pd.Series(_bocpd(z, 1.0 / 250.0, 200, k), index=f.index)


@njit(cache=False)
def _cusum(z, kdrift, h):
    n = len(z)
    out = np.full(n, np.nan)
    sp = 0.0
    sn = 0.0
    for t in range(n):
        x = z[t]
        if np.isnan(x):
            continue
        sp = max(0.0, sp + x - kdrift)
        sn = max(0.0, sn - x - kdrift)
        out[t] = sp - sn
        if sp > h or sn > h:
            sp = 0.0
            sn = 0.0
    return out


@register("cusum_stat", 98, "Two-sided CUSUM of standardized returns (drift 0.5 sigma, reset at 5): + persistent upward drift, - downward; sigma = N-bar EWMA known before the bar.",
          soft=(-8.0, 8.0), period=(20, 500))
@safe
def cusum_stat(f, p, c, l):
    n = clamp_p(p, 10, 1000, 100)
    r = log_ret(f)
    sd = np.sqrt((r ** 2).ewm(span=n, adjust=False, min_periods=max(10, n // 4)).mean().shift(1))
    return pd.Series(_cusum((r / sd.replace(0, np.nan)).to_numpy(float), 0.5, 5.0), index=f.index)


@register("cluster_regime", 99, "Causal k-means regime label (0 = calm ... 2 = turbulent) on volatility, trend and volume features, refit every ~5N bars on the trailing window.",
          bounded=(0.0, 2.0), period=(10, 200))
@safe
def cluster_regime(f, p, c, l):
    n = clamp_p(p, 10, 400, 30)
    r = log_ret(f)
    vol = r.rolling(n, min_periods=n).std()
    trend = r.rolling(n, min_periods=n).sum() / (vol * np.sqrt(n)).replace(0, np.nan)
    v = vol_col(f)
    vz = (v - v.rolling(5 * n, min_periods=n).mean()) / v.rolling(5 * n, min_periods=n).std().replace(0, np.nan)
    X = np.column_stack([vol.to_numpy(float), trend.abs().to_numpy(float), vz.fillna(0.0).to_numpy(float)])
    N = len(X)
    out = np.full(N, np.nan)
    refit = max(100, 5 * n)
    win = 3000
    t = max(3 * n, 300)
    while t < N:
        lo = max(0, t - win)
        tr = X[lo:t]
        tr = tr[np.isfinite(tr).all(axis=1)]
        if len(tr) < 150:
            t += refit
            continue
        mu, sd = tr.mean(axis=0), tr.std(axis=0)
        sd[sd == 0] = 1.0
        Z = (tr - mu) / sd
        srt = np.argsort(Z[:, 0])
        cent = np.array([Z[srt[int(q * (len(Z) - 1))]] for q in (0.15, 0.5, 0.85)])
        for _ in range(12):
            lab = np.argmin(((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2), axis=1)
            new = np.array([Z[lab == k].mean(axis=0) if (lab == k).any() else cent[k] for k in range(3)])
            if np.allclose(new, cent):
                break
            cent = new
        order = np.argsort(cent[:, 0])                  # rank clusters by volatility so labels keep their meaning across refits
        rank = np.empty(3)
        rank[order] = np.arange(3)
        end = min(N, t + refit)
        seg = X[t:end]
        ok = np.isfinite(seg).all(axis=1)
        zl = np.argmin(((((seg - mu) / sd)[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2), axis=1)
        out[t:end] = np.where(ok, rank[zl], np.nan)
        t = end
    return pd.Series(out, index=f.index)


@njit(cache=False)
def _dd_state(close, win):
    n = len(close)
    depth = np.full(n, np.nan)
    tu = np.full(n, np.nan)
    rec = np.full(n, np.nan)
    for t in range(win - 1, n):
        pk = -1e300
        pi = t
        for k in range(t - win + 1, t + 1):
            if close[k] >= pk:
                pk = close[k]
                pi = k
        tr = 1e300
        for k in range(pi, t + 1):
            if close[k] < tr:
                tr = close[k]
        depth[t] = 100.0 * (close[t] / pk - 1.0)
        tu[t] = (t - pi) / win
        rec[t] = 1.0 if pk <= tr else (close[t] - tr) / (pk - tr)
    return depth, tu, rec


@register("dd_depth_pct", 100, "Drawdown depth: percent below the N-bar closing high (<= 0).", bounded=(-100.0, 0.0), period=(10, 1000))
@safe
def dd_depth_pct(f, p, c, l):
    return pd.Series(_dd_state(px(f).to_numpy(float), clamp_p(p, 5, 2000, 100))[0], index=f.index)


@register("dd_time_under", 100, "Time under water: bars since the N-bar closing high divided by N (0..1).", bounded=(0.0, 1.0), period=(10, 1000))
@safe
def dd_time_under(f, p, c, l):
    return pd.Series(_dd_state(px(f).to_numpy(float), clamp_p(p, 5, 2000, 100))[1], index=f.index)


@register("dd_recovery", 100, "Recovery fraction: how much of the drop from the N-bar high to the trough since it has been regained (0..1).",
          bounded=(0.0, 1.0), period=(10, 1000))
@safe
def dd_recovery(f, p, c, l):
    return pd.Series(_dd_state(px(f).to_numpy(float), clamp_p(p, 5, 2000, 100))[2], index=f.index)
