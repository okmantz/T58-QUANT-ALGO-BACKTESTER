"""
T58 Gap-Aligned Opening Drive, fixed 3:1 target (GAD-F) -- ES / NQ (MES / MNQ), 1-minute data, flat by 15:55 ET.

MECHANISM (Rulebook Stage 0).  Who pays you: participants who must trade the overnight imbalance at the open.  When the overnight gap and the first
15-180 minutes of cash trading point the SAME way, close near the extreme of the session range and on the correct side of VWAP, the imbalance has not been
absorbed; price keeps going far enough, often enough, to pay a 3:1 target.  Sign: continuation (long after gap-up + up-drive, short after gap-down + down-drive).
Segment: entry 15-180 minutes after 09:30 ET, exit at stop, target or 15:55 ET.  Payoff b = 3 (fixed in advance).  Cost c in R: ES 0.5 pt, NQ 2 pt round turn (0.03R measured).

WHAT THIS FILE IS.  It is "NQ1! Context Aligned Drive 1m v2" with three overlays REMOVED (each one was a degree of freedom, not a mechanism):
  * the daily 50/200 EMA context filter (it deleted ~55% of trades and almost every short; the drive edge is present without it),
  * the trailing stop (replaced by a fixed target, so every trade has one pre-declared payoff and the Rulebook payoff grid applies),
  * the static news blackout (use the prop-rule news blackout in the lab instead).
Nothing here was tuned on the 2025-09-01+ sealed window.  Parameters are CAD v2's own, unchanged: min move 0.15 ATR, close in the outer 30% of the since-open range,
beyond VWAP, move <= 0.6 ATR (do not chase), stop = 0.7 x |move| clipped to [0.05, 0.27] x ATR14, decisions at minutes 15/30/45/90/120/180, one trade per day.

STOPS AND TARGETS.  stop_loss_distance and take_profit_distance are written on EVERY row of the signal segment (the engine fills on the bar after the decision bar).
The strategy walks the bars itself, mirroring the engine (stop first if one bar touches both), so the signal ends on the exit bar and the engine cannot re-enter.
MAX_TRADES_PER_DAY = 1 is a second guard.

SIZING (lab setting, not code).  Trade MNQ / MES.  Risk about $110-150 per trade on Lucid 50k.  NQ stops reach 0.27 x ATR14 = ~60 points = $120 per MNQ contract, so a
$110 budget skips the widest NQ stops; MES has finer granularity.  Never trade full-size NQ on a 50k account with this stop.
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "T58 Gap Aligned Drive RW Tight Stop 1m"
EXCLUDE_DAYS_OF_WEEK = [5, 6]
MAX_TRADES_PER_DAY = 1
WARMUP_BARS = 0

TZ = "America/New_York"
TS_IS_BAR_END = False
DECISION_MINUTES = (15, 30, 45, 90, 120, 180)
MIN_MOVE_ATR = 0.15
EXT_MAX_ATR = 0.6
CLOSE_POS = 0.7
STOP_FRAC = 0.7
MIN_STOP_ATR = 0.05
MAX_STOP_ATR = 0.15
REWARD_MULT = 3.0          # fixed payoff b = 3
ATR_DAYS = 14
FLAT_MIN = 385             # minutes after 09:30 -> 15:55
SESSION_START = 9 * 60 + 30
SESSION_MIN = 390
SKIP_ROLL_WINDOW = True    # calendar rule (causal): no entries from 10 days before the 3rd Friday of Mar/Jun/Sep/Dec through 1 day after


def _roll_window(day):
    """True if `day` (datetime.date) sits in the quarterly contract-roll window. Pure calendar -- uses no price data."""
    if day.month not in (3, 6, 9, 12):
        return False
    first = pd.Timestamp(day.year, day.month, 1)
    f3 = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)      # 3rd Friday
    dt = pd.Timestamp(day)
    return (f3 - pd.Timedelta(days=10)) <= dt <= (f3 + pd.Timedelta(days=1))


def _bar_minutes(ts):
    d = ts.diff().dt.total_seconds().dropna()
    d = d[d > 0]
    return int(round(float(d.median()) / 60.0)) if len(d) else 1


def generate_signals(df):
    n = len(df)
    sig = np.zeros(n)
    sl = np.full(n, np.nan)
    tp = np.full(n, np.nan)

    def _out():
        bad = (sig != 0) & (~np.isfinite(sl) | ~np.isfinite(tp))
        if bad.any():
            sig[bad] = 0
        out = pd.Series(sig.astype(int), index=df.index)
        out.attrs["stop_loss_distance"] = pd.Series(sl, index=df.index)
        out.attrs["take_profit_distance"] = pd.Series(tp, index=df.index)
        return out

    ts = pd.to_datetime(df["timestamp"])
    loc = ts.dt.tz_convert(TZ) if ts.dt.tz is not None else ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    bar = _bar_minutes(loc)
    if bar != 1:
        raise ValueError(f"Load a 1-minute dataset; this one looks like {bar}-minute bars.")
    k_min = (loc.dt.hour * 60 + loc.dt.minute - SESSION_START).to_numpy()
    in_rth = (k_min >= 0) & (k_min < SESSION_MIN)
    date = loc.dt.date.to_numpy()

    t = pd.DataFrame({"date": date[in_rth], "slot": k_min[in_rth], "pos": np.flatnonzero(in_rth)})
    t = t.drop_duplicates(["date", "slot"])
    if t.empty:
        return _out()
    codes, days = pd.factorize(t["date"], sort=True)
    S = SESSION_MIN
    pos = np.full((len(days), S), -1, dtype=np.int64)
    pos[codes, t["slot"].to_numpy()] = t["pos"].to_numpy()
    nd = len(days)
    ok = pos >= 0

    def grab(col, fill):
        a = df[col].to_numpy(float) if col in df.columns else np.full(n, fill)
        return np.where(ok, a[np.where(ok, pos, 0)], np.nan)

    O, H, L, C = grab("open", np.nan), grab("high", np.nan), grab("low", np.nan), grab("close", np.nan)
    V = np.nan_to_num(grab("volume", 1.0), nan=0.0)
    tpr = np.nan_to_num((H + L + C) / 3.0, nan=0.0)
    cum_v = np.cumsum(V, axis=1)
    vwap = np.where(cum_v > 0, np.cumsum(tpr * V, axis=1) / np.where(cum_v > 0, cum_v, 1.0), C)

    with np.errstate(all="ignore"):
        day_range = np.nanmax(H, axis=1) - np.nanmin(L, axis=1)
    last_close = np.full(nd, np.nan)
    for d in range(nd):
        row = C[d][np.isfinite(C[d])]
        if len(row):
            last_close[d] = row[-1]

    flat_slot = FLAT_MIN
    seen = np.cumsum(ok, axis=1)
    checks = {m - 1 for m in DECISION_MINUTES}              # slot index of the bar that CLOSES at minute m

    for d in range(ATR_DAYS + 1, nd):
        if SKIP_ROLL_WINDOW and _roll_window(days[d]):
            continue
        o = O[d, 0]
        pc = last_close[d - 1]
        if not (np.isfinite(o) and np.isfinite(pc)):
            continue
        atr = np.nanmean(day_range[d - ATR_DAYS: d])
        if not np.isfinite(atr) or atr <= 0:
            continue
        gap_dir = np.sign(o - pc)
        for k in sorted(checks):
            if k >= flat_slot - 1 or pos[d, k] < 0 or seen[d, k] < 0.97 * (k + 1):
                continue
            c = C[d, k]
            hh = np.nanmax(H[d, :k + 1]); ll = np.nanmin(L[d, :k + 1])
            if not (np.isfinite(c) and hh > ll):
                continue
            mv = c - o
            if abs(mv) < MIN_MOVE_ATR * atr or abs(mv) > EXT_MAX_ATR * atr:
                continue
            p = (c - ll) / (hh - ll)
            s = 0
            if mv > 0 and p >= CLOSE_POS and c > vwap[d, k]:
                s = 1
            elif mv < 0 and p <= 1.0 - CLOSE_POS and c < vwap[d, k]:
                s = -1
            if s == 0 or gap_dir != s:
                continue
            if pos[d, k + 1] < 0:                           # fill bar missing: skip rather than enter blind
                continue
            stop_d = float(np.clip(STOP_FRAC * abs(mv), MIN_STOP_ATR * atr, MAX_STOP_ATR * atr))
            tgt_d = REWARD_MULT * stop_d
            entry = O[d, k + 1]
            if not np.isfinite(entry):
                continue
            stop = entry - s * stop_d
            target = entry + s * tgt_d
            # walk forward to find the exit (mirrors the engine: stop first when one bar touches both)
            k_exit = flat_slot
            for j in range(k + 1, flat_slot):
                if pos[d, j] < 0:
                    continue
                if s == 1:
                    if O[d, j] <= stop or L[d, j] <= stop or H[d, j] >= target:
                        k_exit = j
                        break
                else:
                    if O[d, j] >= stop or H[d, j] >= stop or L[d, j] <= target:
                        k_exit = j
                        break
            seg = pos[d, k:k_exit]
            seg = seg[seg >= 0]
            sig[seg] = s
            sl[seg] = stop_d
            tp[seg] = tgt_d
            break                                           # one trade per day
    return _out()
