"""
NQ1! Gated Noise Area Breakout (1-minute data, decisions every 30 minutes)

Research candidate -- NOT a validated strategy. Nothing here has been tuned; every threshold below is a
starting value. Run it through the app's validation (walk-forward, DSR/PBO, cost stress) before trusting it.

LOAD A 1-MINUTE DATASET. (5-minute data also runs, as a coarser approximation; do not use anything else.)
Do NOT declare TIMEFRAME: the strategy reads the native bar size of the dataset and needs 1m for exact stops.

RULES (all times America/New_York, DST-aware, regular session 09:30-16:00 only)
  Noise Area  sigma(t) = mean over the previous 14 sessions of |close(t) / open - 1|, where t is the minute of the
              session. Upper band = today's 09:30 open * (1 + sigma(t)); lower band = open * (1 - sigma(t)).
              (Zarattini / Aziz / Barbon style intraday-momentum bands.)
  Decision    only on the close of the bar that ends at a 30-minute mark: 10:00, 10:30, ... 15:30.
  Entry       LONG  if close > upper band AND close > session VWAP.
              SHORT if close < lower band AND close < session VWAP.
              One entry per direction per day, at most 2 per day (MAX_TRADES_PER_DAY). The engine fills at the
              NEXT bar's open.
  Gate        trade a session only if the prior session's range >= the median range of the 20 sessions before it.
              Optional VIX leg: set VIX_CSV to a FRED VIXCLS csv path and the gate becomes
              "prior-day VIX close >= 18 OR range leg". [MY DESIGN, untested]
  Stop        STOP_MODE "structural" (default): the distance from the signal close back to the trail level
              (the point where the breakout is invalidated), clipped to [20, 100] points and attached to the trade
              as stop_loss_distance. [MY DESIGN] The first version used |close - open| clipped to [4, 15] points;
              on real NQ that is inside one-minute noise (10% win rate, PF 0.63 on 2024+), so it was replaced.
              "open" keeps the old behaviour.
  Trail       once a trade is +1R (close-based), exit when a bar CLOSES beyond the trail level.
              TRAIL_MODE "loose": level = the lower of (upper band, VWAP) for longs, the higher of (lower band, VWAP)
              for shorts, i.e. exit only after losing BOTH. "tight": exit after losing EITHER. The level only ratchets.
              The research text said "nearer of", which is ambiguous, so both are exposed. Test both.
  Exit        stop, trail, or flat at 15:55 ET (signal goes flat on the 15:54 bar, engine fills at the 15:55 open).

WHAT THE ENGINE MUST DO (lab settings, not strategy code -- strategies cannot see their own P&L)
  * Daily loss limit (about $600 on a $50K eval) and a loss-streak halt come from the lab / prop-rule settings.
  * Position size comes from the lab's risk-per-trade setting using the stop distance this file attaches.
    Stops are 20-100 NQ points = $400-$2,000 per NQ contract, which can exceed a $2,000 eval drawdown on its own,
    so RUN IT ON MNQ SIZING (contract_size 2.0, $300 risk per trade). The NQ price series is the same.
  * Costs (spread / slippage / commission) are lab settings. Budget about 1 tick of slippage per side.

DIFFERENCES FROM THE STANDALONE noise_area_nq.py
  * The trail exit is close-based (exit on the next bar open) instead of an intrabar stop order.
  * The stop distance is measured from the signal bar's close, not the next open (the fill is not known yet).
  * Gap adjustment of the bands from the paper is not implemented.
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "NQ Gated Noise Area 1m"
MAX_TRADES_PER_DAY = 2          # level-based signals would otherwise re-enter after an engine stop-out
EXCLUDE_DAYS_OF_WEEK = [5, 6]
WARMUP_BARS = 0                 # the strategy keeps itself flat until it has 21 complete sessions of history

TZ = "America/New_York"
TS_IS_BAR_END = False           # True if your data stamps each bar with its CLOSE time (most vendors: open time)
LOOKBACK_DAYS = 14
SIGNAL_EVERY_MIN = 30
FIRST_SIGNAL_MIN = 30           # minutes after 09:30 -> 10:00
LAST_ENTRY_MIN = 360            # 15:30
FLAT_MIN = 385                  # 15:55
RANGE_LOOKBACK = 20
GATE = "range"                  # "range" | "vix_or_range" | "none"
VIX_CSV = ""                    # optional FRED VIXCLS csv (date, value); used only when GATE == "vix_or_range"
VIX_MIN = 18.0
STOP_MODE = "structural"        # "structural": distance back to the trail level; "open": |close - day open|
MIN_STOP_PTS = 20.0
MAX_STOP_PTS = 100.0
TRAIL_MODE = "loose"            # "loose" | "tight"
TRAIL_ACTIVATE_R = 1.0
TICK = 0.25

SESSION_START = 9 * 60 + 30
SESSION_MIN = 390


def _bar_minutes(ts):
    d = ts.diff().dt.total_seconds().dropna()
    d = d[d > 0]
    return int(round(float(d.median()) / 60.0)) if len(d) else 1


def _vix_by_date():
    if not VIX_CSV:
        return None
    try:
        v = pd.read_csv(VIX_CSV)
        d = pd.to_datetime(v.iloc[:, 0], errors="coerce").dt.date
        x = pd.to_numeric(v.iloc[:, 1], errors="coerce")
        s = pd.Series(x.values, index=d).dropna().sort_index()
        return s
    except Exception:  # noqa: BLE001  -- a bad optional file must not kill the run
        return None


def generate_signals(df):
    n = len(df)
    sig = np.zeros(n)
    sl = np.full(n, np.nan)

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
        out = pd.Series(sig.astype(int), index=df.index)
        out.attrs["stop_loss_distance"] = pd.Series(sl, index=df.index)
        return out
    # One row per session, one column per slot; a missing bar is NaN / -1 (never invented). Sessions are
    # processed bar by bar, so a partial TODAY (live data, truncated history) trades exactly as it would
    # have with the full file -- the signal at a bar never depends on whether later bars exist.
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
    day_open = O[:, 0]
    with np.errstate(all="ignore"):
        day_range = (np.nanmax(H, axis=1) - np.nanmin(L, axis=1)) / day_open

    vix = _vix_by_date() if GATE == "vix_or_range" else None
    vix_dates = list(vix.index) if vix is not None else []

    start = max(LOOKBACK_DAYS, RANGE_LOOKBACK + 1)
    for d in range(start, nd):
        # ---- gate: only information known before the open --------------------------------------------------
        med = np.nanmedian(day_range[d - RANGE_LOOKBACK - 1: d - 1])
        range_ok = bool(day_range[d - 1] >= med)
        vix_ok = False
        if vix is not None:
            j = np.searchsorted(np.array(vix_dates, dtype="datetime64[D]"), np.datetime64(days[d]), side="left") - 1
            vix_ok = bool(j >= 0 and vix.values[j] >= VIX_MIN)
        gate = True if GATE == "none" else (range_ok or vix_ok if GATE == "vix_or_range" else range_ok)
        if not gate:
            continue

        o = day_open[d]
        if not np.isfinite(o):
            continue                                     # no 09:30 bar today -> no reference price, sit out
        with np.errstate(all="ignore"):
            sigma = np.nanmean(np.abs(C[d - LOOKBACK_DAYS: d] / day_open[d - LOOKBACK_DAYS: d, None] - 1.0), axis=0)
        ub = o * (1.0 + sigma)
        lb = o * (1.0 - sigma)
        flat_slot = max(0, (FLAT_MIN - bar) // bar)     # signal is 0 on this bar -> engine exits at the 15:55 open

        used = {1: False, -1: False}
        n_entries = 0
        side = 0
        pending_fill = -1                               # slot at which the engine would fill the entry
        entry = stop = stop_d = 0.0
        active = False
        trail = np.nan
        sig_start = -1

        def close_out(k_exit):
            """Position ends: the signal is 0 from slot k_exit on; fill the 1/-1 run before it."""
            nonlocal side, sig_start
            if sig_start >= 0 and k_exit > sig_start:
                seg = pos[d, sig_start:k_exit]
                sig[seg[seg >= 0]] = side
            side = 0
            sig_start = -1

        for k in range(S):
            if pos[d, k] < 0 or not np.isfinite(sigma[k]):
                if k >= flat_slot and side != 0:
                    close_out(k)
                    break
                continue                                  # missing bar: nothing is seen, nothing is done
            if k >= flat_slot:
                if side != 0:
                    close_out(k)
                break

            # 1) fill of a pending entry at this bar's open (engine fills at the bar after the signal bar)
            if side != 0 and k == pending_fill:
                entry = O[d, k]
                stop = entry - side * stop_d
                active = False
                trail = np.nan

            # 2) manage the open position on this bar
            if side != 0 and k >= pending_fill:
                hit = (O[d, k] <= stop or L[d, k] <= stop) if side == 1 else (O[d, k] >= stop or H[d, k] >= stop)
                if hit:
                    close_out(k)                         # stop-out inside bar k: be flat on this bar
                else:
                    if active and ((side == 1 and C[d, k] < trail) or (side == -1 and C[d, k] > trail)):
                        close_out(k)                     # trail exit decided at this close, fills at k+1 open
                    else:
                        if not active and side * (C[d, k] - entry) >= TRAIL_ACTIVATE_R * stop_d:
                            active = True
                        if active:
                            if side == 1:
                                cand = min(ub[k], vwap[d, k]) if TRAIL_MODE == "loose" else max(ub[k], vwap[d, k])
                                cand = min(cand, C[d, k] - TICK)
                                trail = cand if np.isnan(trail) else max(trail, cand)
                            else:
                                cand = max(lb[k], vwap[d, k]) if TRAIL_MODE == "loose" else min(lb[k], vwap[d, k])
                                cand = max(cand, C[d, k] + TICK)
                                trail = cand if np.isnan(trail) else min(trail, cand)

            # 3) decision at the close of this bar
            m = (k + 1) * bar
            if (side == 0 and m % SIGNAL_EVERY_MIN == 0 and FIRST_SIGNAL_MIN <= m <= LAST_ENTRY_MIN
                    and n_entries < MAX_TRADES_PER_DAY):
                c = C[d, k]
                s = 1 if (c > ub[k] and c > vwap[d, k]) else (-1 if (c < lb[k] and c < vwap[d, k]) else 0)
                if s != 0 and not used[s]:
                    side, used[s] = s, True
                    n_entries += 1
                    if STOP_MODE == "structural":
                        if s == 1:
                            lvl = min(ub[k], vwap[d, k]) if TRAIL_MODE == "loose" else max(ub[k], vwap[d, k])
                        else:
                            lvl = max(lb[k], vwap[d, k]) if TRAIL_MODE == "loose" else min(lb[k], vwap[d, k])
                        raw_stop = abs(c - lvl)
                    else:
                        raw_stop = abs(c - o)
                    stop_d = float(np.clip(raw_stop, MIN_STOP_PTS, MAX_STOP_PTS))
                    sl[pos[d, k]] = stop_d
                    sig_start = k
                    pending_fill = k + 1
        else:
            if side != 0:
                close_out(S)

    out = pd.Series(sig.astype(int), index=df.index)
    out.attrs["stop_loss_distance"] = pd.Series(sl, index=df.index)
    return out
