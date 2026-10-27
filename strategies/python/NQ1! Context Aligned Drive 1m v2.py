"""
T58 NQ/MNQ Context-Aligned Opening Drive v2 (futures, 1m data, flat by 15:55 ET)
Derived from "NQ1! Gap-Aligned Drive Ensemble 1m" + HTF context filter + extension filter (unchanged from v1).

CHANGES IN v2 (from the 2026-10-26 audit)
  1. STOP ON EVERY SIGNAL ROW. v1 wrote stop_loss_distance / trailing_stop_distance only on the DECISION bar, while the
     engine fills on the NEXT bar. That is the most likely source of the 32 fallback-stop (~200 pt) trades, the 9 gap-through
     fills and the $1,221 excursion. v2 writes both distances on every bar of the signal segment, so whichever row the
     engine reads as the entry carries the strategy stop.
  2. SKIP INSTEAD OF ENTERING BLIND. A signal is dropped if the fill bar (k+1) is missing from the resampled series, and a
     final guard zeroes any signal row that still lacks a finite stop (the engine can never fall back to 1%-of-price).
  3. NEWS BLACKOUT (static fallback). No new entry whose decision close falls within +/-3 min of 08:30, 10:00 or 14:00 ET.
     Side effect: the 10:00 decision time (minute 30) can no longer fire. This is a time-of-day approximation; the engine's
     app/data/economic_calendar.py + Prop-Rules news blackout (3 min either side, high-impact only) is the precise version.
     Use that and set NEWS_BLACKOUT = False here if you do not want both. Open positions are NOT flattened before a release.
  4. STRATEGY_NAME made unique (v1 reused the Gap file's name).
  No new entry filters were added. HTF_FILTER and EXT_MAX_ATR are unchanged (EXT_MAX_ATR=0.6 is still an untuned first guess).

ENGINE SETTINGS (lab settings, not code -- strategies cannot see their own P&L)
  * MNQ sizing: contract_size 2.0, pip_size 1.0. One NQ contract exceeded the risk budget on 17 entries; MNQ fixes that.
  * Risk per trade 0.25% (about $125 on $50K) per the Risk Sweep on the first 80% of data (survival score 54.0, eval pass
    84.7%, first payout 72.2%, max DD 3.6%). 0.35%/0.50% breach the 4% trailing limit; 1.00% gave 57% ruin.
    The sweep uses the survival engine, not the pipeline's 90-day ruin metric: re-run the Full Pipeline at 0.25% to confirm.
  * Trial count: 4 variants (base, +HTF, +EXT, +both) + this v2 = report honestly in external_trial_count; skip_optimization=True.

HONEST STATUS (v1 engine run 2026-10-19; v2 NOT yet re-measured):
  NQ/MNQ  PF 1.49 dev / 1.42 holdout, ~495 trades, +0.21R after 2x costs. Works on NQ/MNQ only (ES, GC do not).
  v1 Full pipeline: NOT READY (risk of ruin 36-57% at 1% risk). Re-run at 0.25% after this change.
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "NQ Context Aligned Drive 1m"
EXCLUDE_DAYS_OF_WEEK = [5, 6]
WARMUP_BARS = 0                  # the strategy keeps itself flat until it has 14 complete prior sessions

TZ = "America/New_York"
TS_IS_BAR_END = False            # True if your data stamps each bar with its CLOSE time (most vendors: open time)

DECISION_MINUTES = (15, 30, 45, 90, 120, 180)   # minutes after the 09:30 open (check at the close of that minute)
HTF_FILTER = 1                   # 1 = trade only with the daily 50/200 EMA context (prior closes only)
EXT_MAX_ATR = 0.6                # >0: skip when the drive already exceeds this x ATR14 (do not chase)
MIN_MOVE_ATR = 0.15              # |close - open| >= this * ATR14
ATR_DAYS = 14
CLOSE_POS = 0.7                  # close must be in the top (or bottom) 30% of the since-open range
REQUIRE_GAP_ALIGN = True         # the filter that mattered most
STOP_FRAC = 0.7                  # stop = STOP_FRAC * |move|
MIN_STOP_ATR = 0.05              # stop floor as a fraction of ATR14 (daily range): ~15 NQ pts at the 6-year average
MAX_STOP_ATR = 0.27              # stop cap as a fraction of ATR14: ~80 NQ pts at the 6-year average
TRAIL_MULT = 2.0                 # trailing distance = TRAIL_MULT * initial stop
FLAT_MIN = 385                   # minutes after 09:30 -> 15:55
SESSION_START = 9 * 60 + 30
SESSION_MIN = 390

NEWS_BLACKOUT = True             # static time-of-day blackout on new entries (see header, item 3)
NEWS_TIMES = (8 * 60 + 30, 10 * 60, 14 * 60)    # 08:30, 10:00, 14:00 ET as minutes after midnight
NEWS_WINDOW_MIN = 3              # +/- minutes around each release


def _bar_minutes(ts):
    d = ts.diff().dt.total_seconds().dropna()
    d = d[d > 0]
    return int(round(float(d.median()) / 60.0)) if len(d) else 1


def _in_news(minute_of_day):
    return any(abs(minute_of_day - t) <= NEWS_WINDOW_MIN for t in NEWS_TIMES)


def generate_signals(df):
    n = len(df)
    sig = np.zeros(n)
    sl = np.full(n, np.nan)
    tr = np.full(n, np.nan)

    def _out():
        # Safety net: never emit a position row without a strategy stop (engine would apply a ~1%-of-price fallback).
        bad = (sig != 0) & ~np.isfinite(sl)
        if bad.any():
            sig[bad] = 0
        out = pd.Series(sig.astype(int), index=df.index)
        out.attrs["stop_loss_distance"] = pd.Series(sl, index=df.index)
        out.attrs["trailing_stop_distance"] = pd.Series(tr, index=df.index)
        return out

    ts = pd.to_datetime(df["timestamp"])
    loc = ts.dt.tz_convert(TZ) if ts.dt.tz is not None else ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    bar = _bar_minutes(loc)
    if bar not in (1, 5):
        raise ValueError(f"Load a 1-minute (or 5-minute) dataset; this one looks like {bar}-minute bars.")
    if TS_IS_BAR_END:
        loc = loc - pd.Timedelta(minutes=bar)
    S = SESSION_MIN // bar
    k_min = (loc.dt.hour * 60 + loc.dt.minute - SESSION_START).to_numpy()
    in_rth = (k_min >= 0) & (k_min < SESSION_MIN)
    date = loc.dt.date.to_numpy()

    t = pd.DataFrame({"date": date[in_rth], "slot": k_min[in_rth] // bar, "pos": np.flatnonzero(in_rth)})
    t = t.drop_duplicates(["date", "slot"])
    if t.empty:
        return _out()
    codes, days = pd.factorize(t["date"], sort=True)
    pos = np.full((len(days), S), -1, dtype=np.int64)
    pos[codes, t["slot"].to_numpy()] = t["pos"].to_numpy()
    nd = len(days)
    ok = pos >= 0

    def grab(col, fill):
        a = df[col].to_numpy(float) if col in df.columns else np.full(n, fill)
        return np.where(ok, a[np.where(ok, pos, 0)], np.nan)

    O, H, L, C = grab("open", np.nan), grab("high", np.nan), grab("low", np.nan), grab("close", np.nan)
    V = np.nan_to_num(grab("volume", 1.0), nan=0.0)

    tp = np.nan_to_num((H + L + C) / 3.0, nan=0.0)
    cum_v = np.cumsum(V, axis=1)
    vwap = np.where(cum_v > 0, np.cumsum(tp * V, axis=1) / np.where(cum_v > 0, cum_v, 1.0), C)

    with np.errstate(all="ignore"):
        day_range = np.nanmax(H, axis=1) - np.nanmin(L, axis=1)
    # last close of each complete-enough session (previous session's final traded price)
    last_close = np.full(nd, np.nan)
    for d in range(nd):
        row = C[d][np.isfinite(C[d])]
        if len(row):
            last_close[d] = row[-1]

    ema50 = pd.Series(last_close).ffill().ewm(span=50, adjust=False).mean().to_numpy()
    ema200 = pd.Series(last_close).ffill().ewm(span=200, adjust=False).mean().to_numpy()
    flat_slot = max(0, (FLAT_MIN - bar) // bar)      # signal is 0 on this bar -> engine exits at the 15:55 open
    checks = {}
    for m in DECISION_MINUTES:
        if m % bar == 0:
            checks[m // bar - 1] = True              # slot index of the bar that CLOSES at minute m

    # bars seen so far today, by slot (causal: never uses how many bars the day ends up having)
    seen = np.cumsum(ok, axis=1)
    for d in range(ATR_DAYS + 1, nd):
        o = O[d, 0]
        pc = last_close[d - 1]
        if not (np.isfinite(o) and np.isfinite(pc)):
            continue
        atr = np.nanmean(day_range[d - ATR_DAYS: d])
        if not np.isfinite(atr) or atr <= 0:
            continue
        gap_dir = np.sign(o - pc)
        if d - 1 >= 200:
            dtr = 1 if (ema50[d - 1] > ema200[d - 1] and last_close[d - 1] > ema200[d - 1]) else (-1 if (ema50[d - 1] < ema200[d - 1] and last_close[d - 1] < ema200[d - 1]) else 0)
        else:
            dtr = 0

        side = 0
        sig_start = -1
        pending_fill = -1
        entry = stop = stop_d = trail_d = ext = 0.0
        last_exit_slot = -1

        def close_out(k_exit):
            nonlocal side, sig_start
            if sig_start >= 0 and k_exit > sig_start:
                seg = pos[d, sig_start:k_exit]
                seg = seg[seg >= 0]
                sig[seg] = side
                sl[seg] = stop_d          # stop + trail on EVERY row of the segment, not just the decision bar
                tr[seg] = trail_d
            side = 0
            sig_start = -1

        for k in range(S):
            if pos[d, k] < 0:
                if k >= flat_slot and side != 0:
                    close_out(k)
                    break
                continue
            if k >= flat_slot:
                if side != 0:
                    close_out(k)
                break

            # 1) fill of the pending entry at this bar's open
            if side != 0 and k == pending_fill:
                entry = O[d, k]
                stop = entry - side * stop_d
                ext = entry

            # 2) manage the open position inside this bar (mirrors the engine: stop first, then trail update)
            if side != 0 and k >= pending_fill:
                hit = (O[d, k] <= stop or L[d, k] <= stop) if side == 1 else (O[d, k] >= stop or H[d, k] >= stop)
                if hit:
                    close_out(k)
                    last_exit_slot = k
                else:
                    if side == 1:
                        ext = max(ext, H[d, k])
                        if ext - entry >= 1.0 * stop_d:
                            stop = max(stop, ext - trail_d)
                    else:
                        ext = min(ext, L[d, k])
                        if entry - ext >= 1.0 * stop_d:
                            stop = min(stop, ext + trail_d)

            # 3) decision at the close of this bar
            if side == 0 and k in checks and k > last_exit_slot and seen[d, k] >= 0.97 * (k + 1):
                if NEWS_BLACKOUT and _in_news(SESSION_START + (k + 1) * bar):
                    continue
                c = C[d, k]
                hh = np.nanmax(H[d, :k + 1])
                ll = np.nanmin(L[d, :k + 1])
                rng = hh - ll
                mv = c - o
                if rng > 0 and abs(mv) >= MIN_MOVE_ATR * atr:
                    p = (c - ll) / rng
                    s = 0
                    if mv > 0 and p >= CLOSE_POS and c > vwap[d, k]:
                        s = 1
                    elif mv < 0 and p <= 1.0 - CLOSE_POS and c < vwap[d, k]:
                        s = -1
                    if s != 0 and REQUIRE_GAP_ALIGN and gap_dir != s:
                        s = 0
                    if s != 0 and HTF_FILTER and dtr != s:
                        s = 0
                    if s != 0 and EXT_MAX_ATR > 0 and abs(mv) > EXT_MAX_ATR * atr:
                        s = 0
                    # no fill bar (missing from the series) -> skip the signal instead of entering without a stop
                    if s != 0 and (k + 1 >= flat_slot or pos[d, k + 1] < 0):
                        s = 0
                    if s != 0:
                        stop_d = float(np.clip(STOP_FRAC * abs(mv), MIN_STOP_ATR * atr, MAX_STOP_ATR * atr))
                        trail_d = TRAIL_MULT * stop_d
                        side = s
                        sig_start = k
                        pending_fill = k + 1
        else:
            if side != 0:
                close_out(S)

    return _out()
