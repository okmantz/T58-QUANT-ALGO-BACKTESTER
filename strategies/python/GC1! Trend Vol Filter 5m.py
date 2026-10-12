"""
GC1! Trend Continuation with Volatility Filter (5m, max 2 trades/day)  --  Oct-2026 audit candidate, NO PROVEN EDGE.

RATIONALE (stated before looking at results): intraday index/metal moves trend more than a random walk when the higher-timeframe
drift is clear and volatility is orderly. Trade only WITH the slow trend (fast EMA vs slow EMA, price on the same side) and only on a
fresh 24-bar channel break (momentum confirmation), and stand aside when current ATR is abnormally HIGH versus its own recent
median (news/shock regimes where stops are gapped and spreads widen) or abnormally LOW (no follow-through). A hypothesis, not a proven edge.

Rules (exchange clock America/New_York, 5-minute bars, entries 08:35 ET-11:20 ET):
  TREND  : EMA(20) > EMA(100) and close > EMA(100) for longs (mirror for shorts).
  BREAK  : close > highest high of the previous 24 bars (mirror for shorts).
  VOL    : ATR(14) / rolling-median ATR(14) over 288 bars between 0.7 and 1.8.
  STOP   : 2.0x ATR(14), floored 1.0, capped STOP_CAP_PTS (4.4 pts). TARGET: TP_RR x stop. Time exit 13:20 ET.
Causality: every indicator at bar t uses data up to and including bar t only (EMA/rolling windows are backward-looking; the
channel excludes the current bar).
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "GC Trend+Vol 5m"
TIMEFRAME = "5m"
WARMUP_BARS = 600
MAX_TRADES_PER_DAY = 2
EXCLUDE_DAYS_OF_WEEK = [5, 6]

TZ = "America/New_York"
ENTRY_START = 515
ENTRY_END = 680
SESSION_EXIT = 800
EMA_FAST = 20
EMA_SLOW = 100
DONCHIAN_BARS = 24
VOL_WINDOW = 288
VOL_MIN = 0.7
VOL_MAX = 1.8
ATR_STOP_MULT = 2.0
TP_RR = 1.5
STOP_CAP_PTS = 4.4

# Behaviour rules (0 = off). Searchable by the optimizer within app.backtest.behavior_rules.BEHAVIOR_BOUNDS.
BEHAVIOR_STOP_AFTER_LOSSES = 0
BEHAVIOR_SIZE_DOWN_PROGRESS_PCT = 0.0
BEHAVIOR_SIZE_DOWN_MULT = 0.5
BEHAVIOR_NEWS_BLOCK_MIN = 0
BEHAVIOR_FLATTEN_BEFORE_CLOSE_MIN = 0


def generate_signals(df):
    ts = pd.to_datetime(df["timestamp"])
    loc = ts.dt.tz_convert(TZ) if ts.dt.tz is not None else ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    mins = (loc.dt.hour * 60 + loc.dt.minute).to_numpy()
    day = loc.dt.normalize().dt.tz_localize(None).to_numpy()
    hi = df["high"].astype(float); lo = df["low"].astype(float); cl = df["close"].astype(float)
    pc = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    atr_med = atr.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).median()
    ratio = (atr / atr_med).to_numpy()
    ef = cl.ewm(span=EMA_FAST, adjust=False).mean().to_numpy()
    es = cl.ewm(span=EMA_SLOW, adjust=False).mean().to_numpy()
    up = hi.rolling(DONCHIAN_BARS).max().shift(1).to_numpy()
    dn = lo.rolling(DONCHIAN_BARS).min().shift(1).to_numpy()
    c = cl.to_numpy(); a = atr.to_numpy()
    n = len(df)
    sig = np.zeros(n); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]]); ends = np.r_[starts[1:], n]
    for s0, e0 in zip(starts, ends):
        for k in range(s0, e0):
            if mins[k] < ENTRY_START or mins[k] >= ENTRY_END:
                continue
            if not (np.isfinite(ratio[k]) and VOL_MIN <= ratio[k] <= VOL_MAX and np.isfinite(up[k]) and np.isfinite(a[k])):
                continue
            side = 0
            if ef[k] > es[k] and c[k] > es[k] and c[k] > up[k]:
                side = 1
            elif ef[k] < es[k] and c[k] < es[k] and c[k] < dn[k]:
                side = -1
            if not side:
                continue
            sd = float(min(max(ATR_STOP_MULT * a[k], 1.0), STOP_CAP_PTS))
            sig[k:e0] = side; sl[k] = sd; tp[k] = sd * TP_RR
            late = np.flatnonzero(mins[k:e0] >= SESSION_EXIT)
            if late.size:
                sig[k + late[0]:e0] = 0
            break
    s = pd.Series(sig, index=df.index)
    s.attrs["stop_loss_distance"] = pd.Series(sl, index=df.index)
    s.attrs["take_profit_distance"] = pd.Series(tp, index=df.index)
    return s
