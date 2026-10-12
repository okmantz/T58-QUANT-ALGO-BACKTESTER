"""One-off generator for the six audit strategies (kept for reproducibility; the .py strategy files are the artefacts)."""
from pathlib import Path

ORB = '''"""
{SYM}1! Opening-Range Breakout (5m, max {MAXT} trade(s)/day)  --  Oct-2026 audit candidate, NO PROVEN EDGE.

RATIONALE (why it might work, stated before looking at results): the first {OR_MIN} minutes after the {OPEN_NAME} open concentrate
the day's overnight-inventory repricing; a decisive close outside that range, on a day when the range is neither tiny (no
information) nor huge (move already spent), tends to be followed by directional follow-through for a few hours as slower
participants chase. It is a documented, widely traded pattern, which also means it may be arbitraged away. Treat as a hypothesis.

Rules (exchange clock {TZ}, 5-minute bars):
  RANGE  : high/low of the first {OR_MIN} minutes from {OPEN_H:02d}:{OPEN_M:02d}.
  FILTER : range >= {MIN_R}x and <= {MAX_R}x the mean of the PRIOR 14 sessions' RTH ranges (prior sessions only).
  ENTRY  : first bar CLOSE beyond the range before {ENTRY_END} -> trade that way (engine fills at next bar open).
  STOP   : the range width (distance to the opposite side), capped at STOP_CAP_PTS ({CAP} pts ~ one contract inside a ~1% risk budget on $50K).
  TARGET : TP_RR x stop. Time exit {EXIT_NAME}. No trades Sat/Sun.
Behaviour rules the optimizer may search are declared BEHAVIOR_* (all OFF by default).
Causality: only bars up to the signal bar are used; the range filter uses prior sessions only.
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "{SYM} ORB 5m"
TIMEFRAME = "5m"
WARMUP_BARS = 300
MAX_TRADES_PER_DAY = {MAXT}
EXCLUDE_DAYS_OF_WEEK = [5, 6]

TZ = "America/New_York"
SESSION_OPEN = {OPEN_MIN}          # minutes after midnight, exchange clock
SESSION_EXIT = {EXIT_MIN}
ENTRY_END = {ENTRY_END_MIN}
OR_MIN = {OR_MIN}
MIN_RANGE_ATR = {MIN_R}
MAX_RANGE_ATR = {MAX_R}
TP_RR = {RR}
STOP_CAP_PTS = {CAP}

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
'''

TREND = '''"""
{SYM}1! Trend Continuation with Volatility Filter (5m, max {MAXT} trades/day)  --  Oct-2026 audit candidate, NO PROVEN EDGE.

RATIONALE (stated before looking at results): intraday index/metal moves trend more than a random walk when the higher-timeframe
drift is clear and volatility is orderly. Trade only WITH the slow trend (fast EMA vs slow EMA, price on the same side) and only on a
fresh {DON}-bar channel break (momentum confirmation), and stand aside when current ATR is abnormally HIGH versus its own recent
median (news/shock regimes where stops are gapped and spreads widen) or abnormally LOW (no follow-through). A hypothesis, not a proven edge.

Rules (exchange clock {TZ}, 5-minute bars, entries {ENTRY_START}-{ENTRY_END}):
  TREND  : EMA({FAST}) > EMA({SLOW}) and close > EMA({SLOW}) for longs (mirror for shorts).
  BREAK  : close > highest high of the previous {DON} bars (mirror for shorts).
  VOL    : ATR(14) / rolling-median ATR(14) over {VOLWIN} bars between {VMIN} and {VMAX}.
  STOP   : {ATRSTOP}x ATR(14), floored 1.0, capped STOP_CAP_PTS ({CAP} pts). TARGET: TP_RR x stop. Time exit {EXIT_NAME}.
Causality: every indicator at bar t uses data up to and including bar t only (EMA/rolling windows are backward-looking; the
channel excludes the current bar).
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "{SYM} Trend+Vol 5m"
TIMEFRAME = "5m"
WARMUP_BARS = 600
MAX_TRADES_PER_DAY = {MAXT}
EXCLUDE_DAYS_OF_WEEK = [5, 6]

TZ = "America/New_York"
ENTRY_START = {ENTRY_START_MIN}
ENTRY_END = {ENTRY_END_MIN}
SESSION_EXIT = {EXIT_MIN}
EMA_FAST = {FAST}
EMA_SLOW = {SLOW}
DONCHIAN_BARS = {DON}
VOL_WINDOW = {VOLWIN}
VOL_MIN = {VMIN}
VOL_MAX = {VMAX}
ATR_STOP_MULT = {ATRSTOP}
TP_RR = {RR}
STOP_CAP_PTS = {CAP}

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
'''

SPEC = {
 "ES": dict(CAP=8.5, OPEN=(9, 30), EXIT=(15, 45), ENTRY_END=(11, 30), OPEN_NAME="NYSE cash", EXIT_NAME="15:45 ET", RR=1.5, MIN_R=0.3, MAX_R=1.5, ATRSTOP=2.0),
 "NQ": dict(CAP=22.0, OPEN=(9, 30), EXIT=(15, 45), ENTRY_END=(11, 30), OPEN_NAME="NYSE cash", EXIT_NAME="15:45 ET", RR=1.5, MIN_R=0.3, MAX_R=1.5, ATRSTOP=2.0),
 "GC": dict(CAP=4.4, OPEN=(8, 20), EXIT=(13, 20), ENTRY_END=(10, 30), OPEN_NAME="COMEX pit", EXIT_NAME="13:20 ET", RR=1.5, MIN_R=0.3, MAX_R=1.5, ATRSTOP=2.0),
}
out = Path(__file__).resolve().parents[1] / "strategies" / "python"
for sym, s in SPEC.items():
    om = s["OPEN"][0] * 60 + s["OPEN"][1]
    ex = s["EXIT"][0] * 60 + s["EXIT"][1]
    ee = s["ENTRY_END"][0] * 60 + s["ENTRY_END"][1]
    base = dict(SYM=sym, TZ="America/New_York", MAXT=1, OR_MIN=15, OPEN_H=s["OPEN"][0], OPEN_M=s["OPEN"][1], OPEN_MIN=om,
                EXIT_MIN=ex, ENTRY_END_MIN=ee, ENTRY_END=f"{s['ENTRY_END'][0]:02d}:{s['ENTRY_END'][1]:02d} ET", MIN_R=s["MIN_R"],
                MAX_R=s["MAX_R"], RR=s["RR"], CAP=s["CAP"], OPEN_NAME=s["OPEN_NAME"], EXIT_NAME=s["EXIT_NAME"])
    (out / f"{sym}1! Opening Range Breakout 5m.py").write_text(ORB.format(**base))
    tstart = om + 15
    t = dict(base, MAXT=2, FAST=20, SLOW=100, DON=24, VOLWIN=288, VMIN=0.7, VMAX=1.8, ATRSTOP=s["ATRSTOP"],
             ENTRY_START_MIN=tstart, ENTRY_START=f"{tstart // 60:02d}:{tstart % 60:02d} ET",
             ENTRY_END_MIN=ex - 120, ENTRY_END=f"{(ex - 120) // 60:02d}:{(ex - 120) % 60:02d} ET")
    (out / f"{sym}1! Trend Vol Filter 5m.py").write_text(TREND.format(**t))
print("ok")
