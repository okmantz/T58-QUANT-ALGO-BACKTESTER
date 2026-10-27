"""
NQ1! Gap-Aligned Opening Drive Ensemble v2 (1-minute data, six decision times, wide trailing stop)

Research candidate built 2026-10-12 from the T58 research library (intraday-momentum papers) and a
6-year NQ 1-minute test. NOT a guarantee of passing anything -- see "WHAT WAS MEASURED" below.

LOAD A 1-MINUTE NQ (or MNQ) DATASET. 5-minute data also runs as a coarser approximation.
Do NOT declare TIMEFRAME: the strategy reads the dataset's native bar size.

CHANGES IN v2 (from the 2026-10-26 audit)
  1. RULES LOCKED. No rule or parameter changed. Run the pipeline with skip_optimization=True and enter the real number
     of variants evaluated in external_trial_count (NOT the 313 the search burned through -- the DSR gate was rejecting
     the strategy because of that search: Sharpe 0.153 vs expected best-by-chance 0.397).
  2. STOP ON EVERY SIGNAL ROW. v1 wrote stop_loss_distance / trailing_stop_distance only on the DECISION bar, while the
     engine fills on the NEXT bar. That is the most likely source of the 153 fallback-stop (1%-of-price) trades (31%).
     v2 writes both on every bar of the signal segment. A signal is also skipped if the fill bar is missing, and a final
     guard zeroes any position row without a finite stop.
  3. MNQ SIZING (engine setting): contract_size 2.0, pip_size 1.0. 742 entries (60%) were skipped under NQ sizing because
     one NQ contract at a 15-80 pt stop exceeded the $500 risk budget; that biased the sample toward tight stops.
     Re-measure after switching -- the extra wide-stop trades may be better or worse.
  4. OPTIONAL PRE-REGISTERED REGIME FILTER (REGIME_FILTER, default OFF = locked config). Skip sessions whose ATR14 is below
     the 25th percentile of its own trailing 250-session history (quiet markets: opening drives do not extend; targets
     the 2025 decay, PF 0.99). Evaluate on the DEVELOPMENT slice only, count as ONE extra trial, drop if it does not help.
  5. Do NOT raise risk to fix the EV. Break-even pass rate 16.9% vs deflated holdout 4.9% is an edge problem, not sizing.

IDEA (one sentence)
  When NQ has already moved a meaningful distance from the 09:30 open, is closing near the extreme of that
  move, sits on the matching side of session VWAP, AND the overnight gap points the same way, the move tends
  to keep going -- ride it with a wide trailing stop and no profit target.
  Sources: Gao/Han/Li/Zhou (JFE 2018) prior-close-to-first-30-minutes return predicts later intraday returns;
  Baltussen/Da/Lammers/Martens (JFE 2021); Zarattini/Aziz/Barbon (SFI 24-97) Noise Area; Lou/Polk/Skouras
  (JFE 2019) overnight vs intraday split (the gap filter).

RULES (America/New_York, regular session 09:30-16:00 only)
  Decision times  6 checks, at the CLOSE of the bar ending 09:45, 10:00, 10:15, 11:00, 11:30, 12:30 ET.
  Drive           move = close(t) - open(09:30).  Required: |move| >= 0.15 * ATR14(daily range, prior 14 sessions).
  Extreme         LONG  needs close in the top 30% of the high-low range since the open;
                  SHORT needs it in the bottom 30%.
  VWAP            LONG needs close > session VWAP; SHORT needs close < session VWAP.
  Gap filter      the 09:30 open must be on the SAME side of the prior session's last close as the trade
                  (gap up -> longs only, gap down -> shorts only).
  Entry           engine fills at the NEXT bar's open.  One entry per decision time, one position at a time
                  (a later check can fire after an earlier trade has exited).
  Stop            0.7 * |move|, clipped to [15, 80] NQ points, attached as stop_loss_distance.
  Trail           trailing_stop_distance = 2.0 * initial stop.
  Target          none.
  Exit            stop, trail, or flat at 15:55 ET.
  Weekdays only; a check is skipped if more than 3% of the bars so far today are missing (data gaps).

WHAT WAS MEASURED (v1, NQ 1-min, 2020-01 to 2026-02, $0.25/side slippage + $1.42/micro commission)
  Holdout caveat: the 2024-07+ numbers were visible while two choices (gap filter, 0.7 stop fraction) were made.
  Development 2020-01..2024-06 : 756 trades, avg +0.28R, profit factor 1.54, win rate ~40%
  Holdout     2024-07..2026-02 : 297 trades, avg +0.16R, profit factor 1.22   (pipeline audit: PF 1.12, Sharpe -0.018, 111 trades)
  2025 was roughly flat (PF 0.99).  v2 has NOT been re-measured.

ENGINE SETTINGS (lab settings, not strategy code)
  * MNQ sizing (above). Risk per trade: use the Risk Sweep; v1 guidance was $450-$600 on a $50K account, adaptive throttle on.
  * Costs: ~1 tick slippage per side. Check your firm's rules on automation.

DIFFERENCES FROM THE RESEARCH HARNESS
  * Slippage is charged by the engine. The engine's trail is bar-high/low based and starts from entry.
  * A new check is only allowed on a bar strictly after the bar in which the previous trade was stopped out.
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "NQ Gap-Aligned Drive Ensemble 1m"
EXCLUDE_DAYS_OF_WEEK = [5, 6]
WARMUP_BARS = 0                  # the strategy keeps itself flat until it has 14 complete prior sessions

TZ = "America/New_York"
TS_IS_BAR_END = False            # True if your data stamps each bar with its CLOSE time (most vendors: open time)

DECISION_MINUTES = (15, 30, 45, 90, 120, 180)   # minutes after the 09:30 open (check at the close of that minute)
MIN_MOVE_ATR = 0.15              # |close - open| >= this * ATR14
ATR_DAYS = 14
CLOSE_POS = 0.7                  # close must be in the top (or bottom) 30% of the since-open range
REQUIRE_GAP_ALIGN = True         # the filter that mattered most
STOP_FRAC = 0.7                  # stop = STOP_FRAC * |move|
MIN_STOP_PTS = 15.0
MAX_STOP_PTS = 80.0
TRAIL_MULT = 2.0                 # trailing distance = TRAIL_MULT * initial stop
FLAT_MIN = 385                   # minutes after 09:30 -> 15:55
SESSION_START = 9 * 60 + 30
SESSION_MIN = 390

REGIME_FILTER = False            # pre-registered extra trial: skip quiet-regime sessions (dev slice only; see header item 4)
REGIME_LOOKBACK = 250            # trailing sessions of ATR14 history
REGIME_PCT = 25                  # skip if today's ATR14 is below this percentile of that history
REGIME_MIN_HIST = 60


def _bar_minutes(ts):
    d = ts.diff().dt.total_seconds().dropna()
    d = d[d > 0]
    return int(round(float(d.median()) / 60.0)) if len(d) else 1


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

    # ATR14 per session (prior 14 sessions only) -- used for sizing the drive and for the optional regime filter
    atr_arr = np.full(nd, np.nan)
    for d in range(ATR_DAYS, nd):
        atr_arr[d] = np.nanmean(day_range[d - ATR_DAYS: d])

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
        atr = atr_arr[d]
        if not np.isfinite(atr) or atr <= 0:
            continue
        if REGIME_FILTER:
            hist = atr_arr[max(0, d - REGIME_LOOKBACK): d]
            hist = hist[np.isfinite(hist)]
            if len(hist) >= REGIME_MIN_HIST and atr < np.percentile(hist, REGIME_PCT):
                continue
        gap_dir = np.sign(o - pc)

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
                    # no fill bar (missing from the series) -> skip the signal instead of entering without a stop
                    if s != 0 and (k + 1 >= flat_slot or pos[d, k + 1] < 0):
                        s = 0
                    if s != 0:
                        stop_d = float(np.clip(STOP_FRAC * abs(mv), MIN_STOP_PTS, MAX_STOP_PTS))
                        trail_d = TRAIL_MULT * stop_d
                        side = s
                        sig_start = k
                        pending_fill = k + 1
        else:
            if side != 0:
                close_out(S)

    return _out()
