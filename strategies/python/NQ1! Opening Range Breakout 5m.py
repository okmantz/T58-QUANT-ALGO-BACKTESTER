"""
NQ1! Opening-Range Breakout (5m, max 1 trade(s)/day)  --  Oct-2026 audit candidate, NO PROVEN EDGE.

RATIONALE (why it might work, stated before looking at results): the first 15 minutes after the NYSE cash open concentrate
the day's overnight-inventory repricing; a decisive close outside that range, on a day when the range is neither tiny (no
information) nor huge (move already spent), tends to be followed by directional follow-through for a few hours as slower
participants chase. It is a documented, widely traded pattern, which also means it may be arbitraged away. Treat as a hypothesis.

Rules (exchange clock America/New_York, 5-minute bars):
  RANGE  : high/low of the first 15 minutes from 09:30.
  FILTER : range >= 0.3x and <= 1.5x the mean of the PRIOR 14 sessions' RTH ranges (prior sessions only).
  ENTRY  : first bar CLOSE beyond the range before 11:30 ET -> trade that way (engine fills at next bar open).
  STOP   : the range width (distance to the opposite side), capped at STOP_CAP_PTS (22.0 pts ~ one contract inside a ~1% risk budget on $50K).
  TARGET : TP_RR x stop. Time exit 15:45 ET. No trades Sat/Sun.
Behaviour rules the optimizer may search are declared BEHAVIOR_* (all OFF by default).
Causality: only bars up to the signal bar are used; the range filter uses prior sessions only.
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "NQ ORB 5m"
TIMEFRAME = "5m"
WARMUP_BARS = 300
MAX_TRADES_PER_DAY = 1
EXCLUDE_DAYS_OF_WEEK = [5, 6]

TZ = "America/New_York"
SESSION_OPEN = 570          # minutes after midnight, exchange clock
SESSION_EXIT = 945
ENTRY_END = 690
OR_MIN = 15
MIN_RANGE_ATR = 0.3
MAX_RANGE_ATR = 1.5
TP_RR = 1.5
STOP_CAP_PTS = 22.0

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
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    n = len(df)
    sig = np.zeros(n); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]]); ends = np.r_[starts[1:], n]
    rth = []
    for a, b in zip(starts, ends):
        m = mins[a:b]; r = (m >= SESSION_OPEN) & (m < SESSION_EXIT)
        rth.append(h[a:b][r].max() - l[a:b][r].min() if r.any() else np.nan)
    atr = pd.Series(rth).rolling(14, min_periods=5).mean().shift(1).to_numpy()      # PRIOR sessions only
    for di, (a, b) in enumerate(zip(starts, ends)):
        if not np.isfinite(atr[di]):
            continue
        m = mins[a:b]
        orm = (m >= SESSION_OPEN) & (m < SESSION_OPEN + OR_MIN)
        if orm.sum() < OR_MIN // 5:
            continue
        oh = h[a:b][orm].max(); ol = l[a:b][orm].min(); rng = oh - ol
        if rng <= 0 or rng < MIN_RANGE_ATR * atr[di] or rng > MAX_RANGE_ATR * atr[di]:
            continue
        for i in range(len(m)):
            if m[i] < SESSION_OPEN + OR_MIN or m[i] >= ENTRY_END:
                continue
            side = 1 if c[a + i] > oh else (-1 if c[a + i] < ol else 0)
            if not side:
                continue
            sd = float(min(max(rng, 1.0), STOP_CAP_PTS))
            k = a + i
            sig[k:b] = side; sl[k] = sd; tp[k] = sd * TP_RR
            late = np.flatnonzero(mins[k:b] >= SESSION_EXIT)
            if late.size:
                sig[k + late[0]:b] = 0
            break
    s = pd.Series(sig, index=df.index)
    s.attrs["stop_loss_distance"] = pd.Series(sl, index=df.index)
    s.attrs["take_profit_distance"] = pd.Series(tp, index=df.index)
    return s
