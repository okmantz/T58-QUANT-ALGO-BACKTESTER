"""
V1T v2 -- NQ 1m opening-momentum rule, locked config + one pre-registered second decision window

CHANGES IN v2 (from the 2026-10-26 audit)
  1. LOCKED. Run the pipeline with skip_optimization=True so PBO sees one real candidate and DSR a trial count near 1.
     Report the 52 configurations already tried in external_trial_count, plus 1 for the extra window below = 53
     (52 if you run with component 2 removed). The PBO 1.000 failure came from ranking 8 near-identical candidates;
     the edge itself is flat across nearby settings (holdout PF 1.55 vs dev 1.33, ruin 0.1%, EV +$815/attempt).
  2. ONE EXTRA VARIANT, pre-registered: component 2 is the Trend Momentum Continuation logic (move >= 0.35 x ATR14
     from the open, WITH the 50-day trend, decision window 10:30-12:30 ET) added as a later decision window, same
     S=40 / T=120 / exit 15:55 as component 1. Because component 1 holds to 15:55 once it fires, component 2 can only
     trade on days component 1 did not -- so it adds trades without overlapping positions. Compare with and without it
     (delete the second dict from _CFG["components"] to get the locked original).
  3. NOT IN CODE (engine settings): the stop here is a fixed 40 pts (~$80 on MNQ), so at the $503 intended risk the
     account sits at the 1-contract floor on every trade. Max DD was 2.7% vs the 4% trailing limit; use the Risk Sweep
     to find the exact risk level (roughly 1.3x headroom) rather than guessing. Do not distort the strategy to chase the
     30-day payout metric; median days to payout is ~200 for this edge size.
  No other rule changed. Nothing here has been re-measured.
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "v1t"
TIMEFRAME = "1m"
WARMUP_BARS = 0
MAX_TRADES_PER_DAY = 3

_TZ = "America/New_York"
_CFG = {
    'bar_minutes': 1, 'trend_days': 50, 'atr_days': 14,
    'components': [
        # component 1: original V1T rule (locked)
        {'m': 5, 'm_end': 30, 'feat': 'ret', 'th': 0.3, 'mode': 'mom', 'trend': 'none', 'S': 40, 'T': 120, 'exit': 955},
        # component 2: pre-registered extra variant -- trend-aligned momentum, decision closes 10:30..12:30 ET
        {'m': 59, 'm_end': 179, 'feat': 'ret', 'th': 0.35, 'mode': 'mom', 'trend': 'with', 'S': 40, 'T': 120, 'exit': 955},
    ],
}


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
