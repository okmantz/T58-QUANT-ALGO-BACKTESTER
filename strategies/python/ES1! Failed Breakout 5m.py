"""
ES1! Failed-Breakout Fade (5m, one trade/day)

Honest research candidate (Oct-2026 audit) -- NOT a guaranteed winner; see strategies/AUDIT_2026-10_RESULTS.md for
its measured numbers and the verdict the app gave it.

Logic (all times New York exchange clock, DST-aware; 5-minute bars; ONE trade per day):
  FAIL : 09:30-09:45 range; a pierce of one side that closes back inside is faded to the other side, stop 1.0x range, target 2.5x stop.
  Range must be >= 0.3x the mean prior-session range.
Stops are capped so ONE contract fits a ~1% risk budget on a $50K account with the app's spec costs
(cap 8.5 pts = ~$470 risk/contract). Time exit 15:45 ET. Entry fills at the NEXT bar's open (engine lag).
Causality: opening-range stats use only bars up to the signal bar; the range-size filter uses only PRIOR sessions.
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "ES Fail 5m"
TIMEFRAME = "5m"
WARMUP_BARS = 300
MAX_TRADES_PER_DAY = 1                 # level-based signals would otherwise re-enter after a stop-out
EXCLUDE_DAYS_OF_WEEK = [5, 6]

TZ = "America/New_York"
SESSION_OPEN = 570            # minutes after midnight, exchange clock
SESSION_EXIT = 945
MODES = ["fail"]                       # subset of ["drive", "fail"]
DRIVE_OR_MIN, DRIVE_BIAS, DRIVE_SF, DRIVE_RR = 30, 0.25, 0.5, 2.5
FAIL_OR_MIN, FAIL_WINDOW_MIN, FAIL_SF, FAIL_RR = 15, 60, 1.0, 2.5
MIN_RANGE_ATR = 0.3                # opening range must be >= this x mean prior-session range
STOP_CAP_PTS = 8.5


def generate_signals(df):
    ts = pd.to_datetime(df["timestamp"])
    loc = ts.dt.tz_convert(TZ) if ts.dt.tz is not None else ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    mins = (loc.dt.hour * 60 + loc.dt.minute).to_numpy()
    day = loc.dt.normalize().dt.tz_localize(None).to_numpy()
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    n = len(df)
    sig = np.zeros(n); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]]); ends = np.r_[starts[1:], n]
    rth_rng = []
    for a, b in zip(starts, ends):
        m = mins[a:b]; r = (m >= SESSION_OPEN) & (m < SESSION_EXIT)
        rth_rng.append(h[a:b][r].max() - l[a:b][r].min() if r.any() else np.nan)
    atr = pd.Series(rth_rng).rolling(14, min_periods=5).mean().shift(1).to_numpy()   # PRIOR sessions only

    def place(k, b, side, sd, rr):
        sd = float(min(max(sd, 1.0), STOP_CAP_PTS))
        sig[k:b] = side; sl[k] = sd; tp[k] = sd * rr
        late = np.flatnonzero(mins[k:b] >= SESSION_EXIT)
        if late.size:
            sig[k + late[0]:b] = 0
            if late[0] == 0:
                sig[k] = 0

    for di, (a, b) in enumerate(zip(starts, ends)):
        if not np.isfinite(atr[di]):
            continue
        m = mins[a:b]; cand = []                      # (bar index within day, side, stop dist, rr)
        if "drive" in MODES:
            orm = (m >= SESSION_OPEN) & (m < SESSION_OPEN + DRIVE_OR_MIN)
            if orm.sum() >= DRIVE_OR_MIN // 5:
                oh = h[a:b][orm].max(); ol = l[a:b][orm].min(); rng = oh - ol
                last = np.flatnonzero(orm)[-1]
                if rng > 0 and rng >= MIN_RANGE_ATR * atr[di]:
                    mid = (oh + ol) / 2
                    cl = c[a:b][last]
                    side = 1 if cl > mid + DRIVE_BIAS * rng else (-1 if cl < mid - DRIVE_BIAS * rng else 0)
                    if side:
                        cand.append((last, side, rng * DRIVE_SF, DRIVE_RR))
        if "fail" in MODES:
            orm = (m >= SESSION_OPEN) & (m < SESSION_OPEN + FAIL_OR_MIN)
            if orm.sum() >= FAIL_OR_MIN // 5:
                oh = h[a:b][orm].max(); ol = l[a:b][orm].min(); rng = oh - ol
                if rng > 0 and rng >= MIN_RANGE_ATR * atr[di]:
                    pierced = 0
                    for i in range(len(m)):
                        if not (SESSION_OPEN + FAIL_OR_MIN <= m[i] <= SESSION_OPEN + FAIL_OR_MIN + FAIL_WINDOW_MIN):
                            continue
                        hh, ll, cc = h[a + i], l[a + i], c[a + i]
                        if hh > oh and pierced == 0: pierced = 1
                        if ll < ol and pierced == 0: pierced = -1
                        if pierced == 1 and ol < cc < oh: side = -1
                        elif pierced == -1 and ol < cc < oh: side = 1
                        else: continue
                        cand.append((i, side, rng * FAIL_SF, FAIL_RR)); break
        if not cand:
            continue
        i, side, sd, rr = min(cand, key=lambda x: x[0])      # first signal of the day wins
        place(a + i, b, side, sd, rr)

    s = pd.Series(sig, index=df.index)
    s.attrs["stop_loss_distance"] = pd.Series(sl, index=df.index)
    s.attrs["take_profit_distance"] = pd.Series(tp, index=df.index)
    return s
