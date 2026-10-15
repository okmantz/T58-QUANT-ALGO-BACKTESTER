"""
NQ Lucid50k Drive v2 - opening-drive momentum, 20pt stop, 120pt runner, 1 trade/day equivalent.
Research-grade candidate; see report for honest verdict.
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "nq_lucid50k_drive_v2"
TIMEFRAME = "1m"
WARMUP_BARS = 0
MAX_TRADES_PER_DAY = 1

_TZ = "America/New_York"
_CFG = {'bar_minutes': 1, 'trend_days': 50, 'atr_days': 14, 'components': [{'m': 5, 'm_end': 30, 'feat': 'ret', 'th': 0.2, 'mode': 'mom', 'trend': 'none', 'S': 20, 'T': 120, 'exit': 955}]}


def _sessions(df):
    ts = pd.to_datetime(df["timestamp"])
    loc = ts.dt.tz_convert(_TZ) if ts.dt.tz is not None else ts.dt.tz_localize("UTC").dt.tz_convert(_TZ)
    mins = (loc.dt.hour * 60 + loc.dt.minute).to_numpy()
    day = loc.dt.normalize().dt.tz_localize(None).to_numpy()
    return mins, day


def generate_signals(df):
    n = len(df)
    bar = int(_CFG["bar_minutes"])
    mins, day = _sessions(df)
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float) if "volume" in df else np.ones(n)
    sig = np.zeros(n); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], n]
    nd = len(starts)
    OPEN, END = 570, 960
    # per-session stats from completed sessions only
    rng = np.full(nd, np.nan); rclose = np.full(nd, np.nan); ropen = np.full(nd, np.nan)
    for di, (a, b) in enumerate(zip(starts, ends)):
        m = mins[a:b]; r = (m >= OPEN) & (m < END)
        if r.any():
            ropen[di] = o[a:b][r][0]
        if r.sum() >= 40:
            rng[di] = h[a:b][r].max() - l[a:b][r].min()
            rclose[di] = c[a:b][r][-1]
    atr = np.full(nd, np.nan); sma = np.full(nd, np.nan); pclose = np.full(nd, np.nan)
    hist = []
    for di in range(nd):
        if len(hist) >= _CFG["trend_days"]:
            hh = np.asarray(hist)
            atr[di] = rng[hh[-_CFG["atr_days"]:]].mean()
            sma[di] = rclose[hh[-_CFG["trend_days"]:]].mean()
            pclose[di] = rclose[hh[-1]]
        if np.isfinite(rclose[di]):
            hist.append(di)
    for di, (a, b) in enumerate(zip(starts, ends)):
        if not (np.isfinite(atr[di]) and np.isfinite(sma[di]) and np.isfinite(ropen[di])):
            continue
        m = mins[a:b]; oo = o[a:b]; hi = h[a:b]; lo = l[a:b]; cc = c[a:b]; vv = v[a:b]
        rth = (m >= OPEN) & (m < END)
        if not rth.any():
            continue
        idx = np.flatnonzero(rth)
        trend = 1 if pclose[di] > sma[di] else -1
        gap = ropen[di] - pclose[di]
        tp_ = (hi + lo + cc) / 3.0
        cv = np.cumsum(np.where(rth, vv, 0.0)); cpv = np.cumsum(np.where(rth, tp_ * vv, 0.0))
        busy_until = 0   # minute-of-day before which no new position may start
        for comp in _CFG["components"]:
            m_end = comp.get("m_end", comp["m"])
            f = comp["feat"]
            fired = -1; d = 0
            for mm in range(comp["m"], m_end + 1, bar):
                dec_min = OPEN + mm + bar
                kk = np.flatnonzero(m + bar == dec_min)
                if len(kk) == 0:
                    continue
                k = int(kk[0])
                if m[k] + bar < busy_until or not rth[k]:
                    continue
                if f == "ret":
                    x = (cc[k] - oo[idx[0]]) / atr[di]
                elif f == "gap":
                    x = gap / atr[di]
                elif f == "vwap":
                    vw = cpv[k] / max(cv[k], 1.0)
                    x = (cc[k] - vw) / atr[di]
                else:
                    break
                if not np.isfinite(x) or abs(x) < comp["th"]:
                    continue
                dd = 1 if x > 0 else -1
                if comp["mode"] == "fade":
                    dd = -dd
                if comp["trend"] == "with" and dd != trend:
                    continue
                if comp["trend"] == "against" and dd != -trend:
                    continue
                fired = k; d = dd
                break
            if fired < 0:
                continue
            k = fired
            exit_min = comp["exit"]
            if exit_min <= m[k] + bar:
                continue
            pos = np.flatnonzero((np.arange(len(m)) >= k) & (m < exit_min) & rth)
            if len(pos) == 0:
                continue
            sig[a + pos] = d
            sl[a + pos] = comp["S"]
            if comp["T"] > 0:
                tp[a + pos] = comp["T"]
            busy_until = exit_min
    out = pd.Series(sig.astype(int), index=df.index)
    out.attrs["stop_loss_distance"] = pd.Series(sl, index=df.index)
    if np.isfinite(tp).any():
        out.attrs["take_profit_distance"] = pd.Series(tp, index=df.index)
    return out
