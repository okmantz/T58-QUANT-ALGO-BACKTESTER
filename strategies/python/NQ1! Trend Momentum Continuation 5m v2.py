"""
NQ1! Trend-Aligned Momentum Continuation v2 (5m, one trade/day max)

Built for the Lucid Trading $50K LucidPro evaluation. Written against the T58 backtester's Python strategy contract
(generate_signals(df) -> -1/0/1 plus attrs["stop_loss_distance"]).

CHANGES IN v2 (from the 2026-10-26 audit)
  1. ATR-BASED STOP. v1 hard-coded STOP_PTS = 80 (about $160 on MNQ), which is a different risk in 2020 (NQ ~9,000) than
     in 2025 (>20,000) and was always smaller than the intended risk, so the account always traded one micro. v2 uses
     STOP_ATR_FRAC x ATR14 (0.27 reproduces ~80 pts at the 6-year average range, so behaviour is preserved on average
     while risk becomes comparable across years). Same convention as the Gap and Context files.
  2. NO 1.5R TARGET, TRAILING STOP INSTEAD. The Gap file's testing found capping winners did not help; v1 capped every
     winner at 1.5R (51% win rate, 1.13 reward/risk). v2 sets TARGET_R = 0 and attaches trailing_stop_distance =
     TRAIL_MULT x stop (same 2.0x as Gap/Context). Set TARGET_R back to 1.5 and TRAIL_MULT to 0 to reproduce v1 exactly
     (apart from the stop size).
  3. A stop is written on every signal row (it already was in v1) and an ATR guard skips days where the stop is not finite.

STATUS / HONEST ADVICE
  v1 audit: avg R 0.088, PF 1.20 in-sample / 1.05 holdout, holdout Sharpe -0.392, deflated holdout pass 0.0% vs 36.8%
  break-even, median 178 days to first payout. 241 configurations were already evaluated (expected best-by-chance Sharpe
  0.137 vs its 0.103). DO NOT RUN ANOTHER SEARCH ON THIS FILE. This v2 exists to be compared once, as a pre-registered
  variant. The preferred route is to fold its logic into V1T as a second decision window (done in "NQ1! V1T 1m v2.py");
  if the v2 holdout Sharpe is still <= 0, retire this file.

Edge being harvested: intraday momentum (Gao et al. JFE 2018; Baltussen et al. JFE 2021; Zarattini/Aziz/Barbon 2024).

Rules (New York exchange clock, DST-aware, 5-minute bars, regular session 09:30-16:00 ET):
  TREND    prior session close > its 50-session mean close -> longs only; below -> shorts only.   (prior data only)
  TRIGGER  first 5m bar CLOSE inside [ENTRY_FIRST_MIN, ENTRY_LAST_MIN] where |close - session open| >= MOVE_ATR x ATR14
           and the move's sign agrees with TREND. ATR14 = mean RTH high-low range of the previous 14 sessions.
  ENTRY    next bar's open (engine lag). One entry per session.
  STOP     STOP_ATR_FRAC x ATR14 points from entry, attached per trade as stop_loss_distance.
  TRAIL    trailing_stop_distance = TRAIL_MULT x stop.
  EXIT     stop, trail, (optional TARGET_R x stop target), or flat at EXIT_MIN (15:55 ET).
Causality: every input at bar t is a close at or before t, or a prior-session statistic. Nothing looks ahead.

Run it on MNQ sizing with the Lucid 50K preset; size by dollar risk through the lab (Risk Sweep), not in this file.
"""
import numpy as np
import pandas as pd

STRATEGY_NAME = "NQ Trend Momentum Continuation 5m"
TIMEFRAME = "5m"
WARMUP_BARS = 0
MAX_TRADES_PER_DAY = 1
EXCLUDE_DAYS_OF_WEEK = [5, 6]

TZ = "America/New_York"
SESSION_OPEN = 570          # 09:30 ET in minutes after midnight
SESSION_END = 960           # 16:00 ET
EXIT_MIN = 955              # flat at 15:55 ET
ENTRY_FIRST_MIN = 630       # first trigger bar closes no earlier than 10:30 ET
ENTRY_LAST_MIN = 750        # last trigger bar closes no later than 12:30 ET
ATR_DAYS = 14
TREND_DAYS = 50
MOVE_ATR = 0.35             # |open->now move| must be >= this x ATR14
STOP_ATR_FRAC = 0.27        # stop = this x ATR14 (v1: fixed 80 pts, about 0.27 x the 6-year average range)
TARGET_R = 0.0              # profit target = this x stop distance (0 = no target; v1 used 1.5)
TRAIL_MULT = 2.0            # trailing distance = this x stop (0 = no trail)


def generate_signals(df):
    n = len(df)
    sig = np.zeros(n)
    sl = np.full(n, np.nan)
    tp = np.full(n, np.nan)
    tr = np.full(n, np.nan)

    ts = pd.to_datetime(df["timestamp"])
    loc = ts.dt.tz_convert(TZ) if ts.dt.tz is not None else ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    mins = (loc.dt.hour * 60 + loc.dt.minute).to_numpy()
    day = loc.dt.normalize().dt.tz_localize(None).to_numpy()
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], n]
    nd = len(starts)

    # Per-session RTH statistics. A session only contributes to LATER sessions' statistics once it is complete, and the
    # current session's own validity depends only on its open (known at the open) -- never on how many bars it ends up having.
    rng = np.full(nd, np.nan)
    rclose = np.full(nd, np.nan)
    ropen = np.full(nd, np.nan)
    for di, (a, b) in enumerate(zip(starts, ends)):
        m = mins[a:b]
        r = (m >= SESSION_OPEN) & (m < SESSION_END)
        if r.any():
            ropen[di] = o[a:b][r][0]
        if r.sum() >= 60:
            rng[di] = h[a:b][r].max() - l[a:b][r].min()
            rclose[di] = c[a:b][r][-1]

    atr = np.full(nd, np.nan)
    sma = np.full(nd, np.nan)
    pclose = np.full(nd, np.nan)
    hist_idx = []                                   # indices of completed sessions seen so far
    for di in range(nd):
        if len(hist_idx) >= TREND_DAYS:
            prev = np.asarray(hist_idx)
            atr[di] = rng[prev[-ATR_DAYS:]].mean()
            sma[di] = rclose[prev[-TREND_DAYS:]].mean()
            pclose[di] = rclose[prev[-1]]
        if np.isfinite(rclose[di]):
            hist_idx.append(di)

    for di, (a, b) in enumerate(zip(starts, ends)):
        if not (np.isfinite(atr[di]) and np.isfinite(sma[di]) and np.isfinite(pclose[di]) and np.isfinite(ropen[di])):
            continue
        stop_d = STOP_ATR_FRAC * atr[di]
        if not np.isfinite(stop_d) or stop_d <= 0:
            continue
        trend = 1 if pclose[di] > sma[di] else -1
        m = mins[a:b]
        sess_open = ropen[di]
        thr = MOVE_ATR * atr[di]
        triggered = False
        # Causal state machine, one pass forward through the day: the signal at bar i depends only on bars <= i.
        for i in range(b - a):
            if m[i] >= EXIT_MIN or m[i] < SESSION_OPEN:
                triggered = triggered and m[i] < EXIT_MIN and m[i] >= SESSION_OPEN
                continue
            if not triggered:
                bar_close_min = m[i] + 5          # the bar stamped m[i] closes 5 minutes later
                if ENTRY_FIRST_MIN <= bar_close_min <= ENTRY_LAST_MIN:
                    move = c[a + i] - sess_open
                    if abs(move) >= thr and (1 if move > 0 else -1) == trend:
                        triggered = True
            if triggered:
                sig[a + i] = trend
                sl[a + i] = stop_d
                if TARGET_R > 0:
                    tp[a + i] = stop_d * TARGET_R
                if TRAIL_MULT > 0:
                    tr[a + i] = stop_d * TRAIL_MULT
    out = pd.Series(sig.astype(int), index=df.index)
    out.attrs["stop_loss_distance"] = pd.Series(sl, index=df.index)
    if TARGET_R > 0:
        out.attrs["take_profit_distance"] = pd.Series(tp, index=df.index)
    if TRAIL_MULT > 0:
        out.attrs["trailing_stop_distance"] = pd.Series(tr, index=df.index)
    return out
