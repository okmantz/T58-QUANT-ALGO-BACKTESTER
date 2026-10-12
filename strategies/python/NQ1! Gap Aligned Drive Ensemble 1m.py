"""
NQ1! Gap-Aligned Opening Drive Ensemble (1-minute data, six decision times, wide trailing stop)

Research candidate built 2026-10-12 from the T58 research library (intraday-momentum papers) and a
6-year NQ 1-minute test. NOT a guarantee of passing anything -- see "WHAT WAS MEASURED" below.

LOAD A 1-MINUTE NQ (or MNQ) DATASET. 5-minute data also runs as a coarser approximation.
Do NOT declare TIMEFRAME: the strategy reads the dataset's native bar size.

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
                  (gap up -> longs only, gap down -> shorts only).  [This single filter roughly doubled the
                  average R per trade and survived a shuffled-gap permutation test.]
  Entry           engine fills at the NEXT bar's open.  One entry per decision time, one position at a time
                  (a later check can fire after an earlier trade has exited).
  Stop            0.7 * |move|, clipped to [15, 80] NQ points, attached as stop_loss_distance.
  Trail           trailing_stop_distance = 2.0 * initial stop.  The initial stop is tighter than the trail, so
                  the trail only starts to bite after the trade is ~+1R in profit.
  Target          none (the edge is a few large winners; capping them did not help in testing).
  Exit            stop, trail, or flat at 15:55 ET.
  Weekdays only; partial-session days (holidays, half-days, data gaps) are skipped.

WHAT WAS MEASURED (NQ 1-min, 2020-01 to 2026-02, $0.25/side slippage + $1.42/micro commission)
  Holdout caveat: the 2024-07+ numbers were visible while two choices (the gap filter and the 0.7 stop
  fraction) were being made, so treat the holdout as "mostly clean", not pristine.
  Development 2020-01..2024-06 : 756 trades, avg +0.28R, profit factor 1.54, win rate ~40%
  Holdout     2024-07..2026-02 : 297 trades, avg +0.16R, profit factor 1.22
  Doubling slippage to 2 ticks per side: avg R falls only ~4%.   2025 was roughly flat (PF 0.99).
  Eval simulation ($50K, $3,000 target, $2,000 end-of-day trailing drawdown, 30 calendar days, rolling
  non-overlapping windows, 70 windows): ~29% pass / ~10% bust at $600 risk per trade (adaptive sizing) when the
  firm has NO best-day consistency rule; ~16% pass with a 50% best-day rule.  95% interval for the 29% is
  roughly 19-40%.  At $450 risk: ~23% pass / ~1% bust.  Longest losing streak in the sample: 13 trades.
  Repo engine check (run_backtest, MNQ sizing, 1.2% risk, 0.25 spread + 0.25 slippage): PF 1.45 development,
  1.19 holdout.  See research/T58_FINAL_STRATEGY_REPORT.md.

WHAT THE ENGINE MUST DO (lab settings, not strategy code -- strategies cannot see their own P&L)
  * Run on MNQ sizing: contract_size 2.0, pip_size 1.0.  NQ stops of 15-80 points are $300-$1,600 per full
    contract and will not fit a $2,000 drawdown; micros let the risk setting size the trade.
  * Risk per trade about $450 (conservative) to $600 (max pass odds) on a $50K account, i.e. risk_value
    0.9%-1.2%.  Turn on the adaptive-risk throttle (size = cushion above the trailing floor / $2,000,
    clipped 0.35x..1.25x).  Daily loss limit about $600.  Costs: ~1 tick slippage per side.
  * No news/weekend settings are needed (flat by 15:55), but check your firm's rules on automation.

DIFFERENCES FROM THE RESEARCH HARNESS
  * Slippage is charged by the engine, not here.  The engine's trail is bar-high/low based and starts from
    entry (research started it at +1R); expect small differences in individual trades.
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

    flat_slot = max(0, (FLAT_MIN - bar) // bar)      # signal is 0 on this bar -> engine exits at the 15:55 open
    checks = {}
    for m in DECISION_MINUTES:
        if m % bar == 0:
            checks[m // bar - 1] = True              # slot index of the bar that CLOSES at minute m

    n_bars = ok.sum(axis=1)
    for d in range(ATR_DAYS + 1, nd):
        if n_bars[d] < 0.97 * S:                      # holiday / half-day / data-gap session: sit out
            continue
        o = O[d, 0]
        pc = last_close[d - 1]
        if not (np.isfinite(o) and np.isfinite(pc)):
            continue
        atr = np.nanmean(day_range[d - ATR_DAYS: d])
        if not np.isfinite(atr) or atr <= 0:
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
                sig[seg[seg >= 0]] = side
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
            if side == 0 and k in checks and k > last_exit_slot:
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
                    if s != 0:
                        stop_d = float(np.clip(STOP_FRAC * abs(mv), MIN_STOP_PTS, MAX_STOP_PTS))
                        trail_d = TRAIL_MULT * stop_d
                        side = s
                        sig_start = k
                        pending_fill = k + 1
                        sl[pos[d, k]] = stop_d
                        tr[pos[d, k]] = trail_d
        else:
            if side != 0:
                close_out(S)

    return _out()
