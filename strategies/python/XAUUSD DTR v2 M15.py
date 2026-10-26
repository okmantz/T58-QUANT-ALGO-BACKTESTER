"""DTR Gold Sweep-Reclaim v2 (research candidate, pipeline Series contract).

Changes vs XAUUSD_DTR_Pipeline_1m_DualPath.py (see bottom of file for the audit trail):
  * FIX  pandas>=3 unit bug: DatetimeIndex.asi8 is microseconds, so the old `== 60*10**9`
         1-minute guard rejected real M1 data and the `// (24*3600*10**9)` day key collapsed
         ~1000 calendar days into one bucket (the 3-trades/day cap became ~3 trades per 1000 days).
  * Timeframe-parametric: every length is declared in BARS for the chosen TIMEFRAME, so the same
    logic runs on M5/M15 where $0.36 round-trip cost is a small fraction of the stop.
  * Cost-aware viability: stop must be >= MIN_STOP_TO_COST x round-trip cost.
  * FIX  signal/engine desync: fill, stop, target and hold deadline are recomputed from the NEXT bar's
         open (where the engine fills), so the strategy no longer re-enters with the 1% fallback stop.

RESULT (honest): NOT READY. On 6y GC1! sized as MGC (Lucid 50k, ~$0.36/oz cost), no timeframe has an edge:
  M1 IS PF 0.68 | M5 0.82 | M15 0.62 (1x cost); validation PF 0.91 / 0.75 / 0.71; zero-cost PF ~0.985.
  Official pipeline on this M15 file: NOT READY, score 9.4. ~15 variants tried; final 20% holdout never read.
  Use this file as a correct, cost-aware base for a NEW signal, not as a passing strategy.
  Still a research strategy: no sizing, no account risk (host applies RiskConfig), stop/target
    distances exported through Series.attrs. Timestamps/sessions are UTC.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

STRATEGY_NAME = "DTR Gold Sweep-Reclaim v2"
TIMEFRAME = "15m"
HTF_TIMEFRAMES = ["1h"]

ATR_PERIOD = 14
EMA_FAST = 20
EMA_SLOW = 96
POOL_LOOKBACK = 16
WARMUP_BARS = 120
MAX_TRADES_PER_DAY = 3
REGIME_ON = 1

SWEEP_BUFFER_ATR = 0.15
XAU_SWEEP_MIN_USD = 0.20
XAU_SWEEP_MAX_USD = 0.40
RECLAIM_TIMEOUT_BARS = 2
CONFIRM_TIMEOUT_BARS = 3
RETEST_TIMEOUT_BARS = 3
BOS_LOOKBACK = 4
BOS_BREAK_ATR = 0.05
DISPLACEMENT_ATR = 1.0
RETEST_BAND_ATR = 0.10
MAX_RETEST_CLOSE_GAP_ATR = 0.25
BREAK_RECLAIM_ATR = 0.05
ALLOW_PATH_A = True
ALLOW_PATH_B = True

STOP_ATR_BUFFER = 0.10
MIN_STOP_ATR = 0.6
MAX_STOP_ATR = 3.5
TARGET_RR = 2.0
MAX_HOLD_BARS = 24
SESSION_START_UTC = 7
SESSION_END_UTC = 20
MIN_STOP_TO_COST = 6.0
ROUND_TRIP_COST = 0.36          # gold $/oz: ~1 tick spread + ~1 tick slippage + $1.60 MGC commission /10oz


@dataclass
class Confirm:
    candle_i: int
    midpoint: float
    expiry_i: int


@dataclass
class Setup:
    direction: int
    pool: float
    sweep_i: int
    extreme: float
    reclaim_i: int = -1
    reclaim_high: float = np.nan
    reclaim_low: float = np.nan
    path_a: Confirm | None = None
    path_b: Confirm | None = None


def _atr(high, low, close, n):
    previous = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def _prior_extreme(data, window, side):
    s = pd.Series(data).shift(1).rolling(window, min_periods=window)
    return (s.min() if side == "low" else s.max()).to_numpy()


def _pick_pool(i, direction, lo, hi, price_l, price_h, buf, h1_lo, h1_hi):
    levels = (lo[i], h1_lo[i]) if direction == 1 else (hi[i], h1_hi[i])
    for p in levels:
        if not np.isfinite(p):
            continue
        if direction == 1 and price_l <= p - buf:
            return float(p)
        if direction == -1 and price_h >= p + buf:
            return float(p)
    return None


def _retest(i, direction, confirmation, lo, hi, close, atr):
    if confirmation is None or i <= confirmation.candle_i or i > confirmation.expiry_i:
        return False
    midpoint = confirmation.midpoint
    band = RETEST_BAND_ATR * atr
    touched = lo <= midpoint + band and hi >= midpoint - band
    reclaimed_mid = close >= midpoint if direction == 1 else close <= midpoint
    not_chasing = abs(close - midpoint) <= MAX_RETEST_CLOSE_GAP_ATR * atr
    return bool(touched and reclaimed_mid and not_chasing)


def generate_signals(df: pd.DataFrame) -> pd.Series:
    columns = ("open", "high", "low", "close")
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError("Missing required OHLC columns: " + ", ".join(missing))
    n = len(df)
    sig = np.zeros(n, dtype=int)
    sl = np.full(n, np.nan)
    tp = np.full(n, np.nan)
    signal_path = np.full(n, "", dtype=object)
    if n == 0:
        return _pack(df, sig, sl, tp, signal_path)

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    if "timestamp" in df.columns:
        times = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    elif isinstance(df.index, pd.DatetimeIndex):
        times = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
    else:
        raise ValueError("Provide UTC timestamp column or a DatetimeIndex")
    if n >= 3 and (np.diff(times.values.astype("datetime64[ns]").astype(np.int64)) <= 0).any():
        raise ValueError("Timestamps must be strictly increasing and unique")
    hours = times.hour.to_numpy()
    minutes = times.minute.to_numpy()
    # unit-safe UTC day ordinal (independent of the datetime64 resolution pandas hands us)
    days = times.tz_convert("UTC").tz_localize(None).values.astype("datetime64[D]").astype(np.int64)

    atr = _atr(h, l, c, int(ATR_PERIOD))
    ema_fast = pd.Series(c).ewm(span=int(EMA_FAST), adjust=False).mean().to_numpy()
    ema_slow = pd.Series(c).ewm(span=int(EMA_SLOW), adjust=False).mean().to_numpy()
    prior_lo = _prior_extreme(l, int(POOL_LOOKBACK), "low")
    prior_hi = _prior_extreme(h, int(POOL_LOOKBACK), "high")
    bos_hi = _prior_extreme(h, int(BOS_LOOKBACK), "high")
    bos_lo = _prior_extreme(l, int(BOS_LOOKBACK), "low")
    h1_lo = df["tf60_low"].to_numpy(float) if "tf60_low" in df else np.full(n, np.nan)
    h1_hi = df["tf60_high"].to_numpy(float) if "tf60_high" in df else np.full(n, np.nan)

    setups = {1: None, -1: None}
    daily_count = {}
    active = None
    last_exit = -1
    min_start = max(int(WARMUP_BARS), int(EMA_SLOW), int(POOL_LOOKBACK), int(BOS_LOOKBACK))
    # session end expressed as minutes-of-day so it works for any bar size
    end_min = SESSION_END_UTC * 60
    for i in range(n):
        if i < min_start or not (np.isfinite(atr[i]) and atr[i] > 0):
            continue
        if not np.all(np.isfinite([o[i], h[i], l[i], c[i]])):
            continue
        hour = hours[i]
        mod = hour * 60 + minutes[i]
        in_session = SESSION_START_UTC <= hour < SESSION_END_UTC
        force_flat = mod >= end_min - 5

        if active is not None:
            d, sd_a, sig_i, stop_px, target_px, deadline = active
            sig[i] = d
            if i == sig_i + 1:
                # model the ENGINE fill: next-bar open, with stop/target distances measured from that fill
                fill = o[i]
                stop_px = fill - d * sd_a
                target_px = fill + d * TARGET_RR * sd_a
                active = (d, sd_a, sig_i, stop_px, target_px, deadline)
            if i > sig_i:
                hit_stop = (l[i] <= stop_px) if d == 1 else (h[i] >= stop_px)
                hit_target = (h[i] >= target_px) if d == 1 else (l[i] <= target_px)
                if hit_stop or hit_target or force_flat or i >= deadline:
                    sig[i] = 0
                    active = None
                    last_exit = i
            if active is not None:
                continue
            setups = {1: None, -1: None}
            continue

        if not in_session:
            setups = {1: None, -1: None}
            continue
        if i <= last_exit:
            continue

        eligible_long = (ema_fast[i] > ema_slow[i] and c[i] > ema_slow[i]) if REGIME_ON else True
        eligible_short = (ema_fast[i] < ema_slow[i] and c[i] < ema_slow[i]) if REGIME_ON else True
        buf = float(np.clip(SWEEP_BUFFER_ATR * atr[i], XAU_SWEEP_MIN_USD, XAU_SWEEP_MAX_USD))
        candidates = []

        for d in (1, -1):
            setup = setups[d]
            if setup is None:
                pool = _pick_pool(i, d, prior_lo, prior_hi, l[i], h[i], buf, h1_lo, h1_hi)
                if pool is not None:
                    setups[d] = Setup(direction=d, pool=pool, sweep_i=i,
                                      extreme=l[i] if d == 1 else h[i])
                    setup = setups[d]
                else:
                    continue
            else:
                if setup.reclaim_i < 0:
                    setup.extreme = min(setup.extreme, l[i]) if d == 1 else max(setup.extreme, h[i])

            if setup.reclaim_i < 0:
                if i - setup.sweep_i > RECLAIM_TIMEOUT_BARS:
                    setups[d] = None
                    continue
                reclaimed = c[i] > setup.pool if d == 1 else c[i] < setup.pool
                if reclaimed:
                    setup.reclaim_i = i
                    setup.reclaim_high = h[i]
                    setup.reclaim_low = l[i]
                continue

            if (d == 1 and l[i] < setup.extreme - STOP_ATR_BUFFER * atr[i]) or (
                d == -1 and h[i] > setup.extreme + STOP_ATR_BUFFER * atr[i]
            ):
                setups[d] = None
                continue

            if i - setup.reclaim_i > CONFIRM_TIMEOUT_BARS and setup.path_a is None and setup.path_b is None:
                setups[d] = None
                continue

            if i > setup.reclaim_i and i - setup.reclaim_i <= CONFIRM_TIMEOUT_BARS:
                directional_body = (c[i] - o[i]) if d == 1 else (o[i] - c[i])
                displacement = directional_body >= DISPLACEMENT_ATR * atr[i]
                structure_break = (c[i] > bos_hi[i] + BOS_BREAK_ATR * atr[i]) if d == 1 else (
                    c[i] < bos_lo[i] - BOS_BREAK_ATR * atr[i])
                directional_move = (c[i] > c[setup.reclaim_i]) if d == 1 else (c[i] < c[setup.reclaim_i])
                if ALLOW_PATH_A and setup.path_a is None and directional_move and (displacement or structure_break):
                    setup.path_a = Confirm(i, (h[i] + l[i]) / 2, i + RETEST_TIMEOUT_BARS)
                breakout_reclaim = (c[i] > setup.reclaim_high + BREAK_RECLAIM_ATR * atr[i]) if d == 1 else (
                    c[i] < setup.reclaim_low - BREAK_RECLAIM_ATR * atr[i])
                if ALLOW_PATH_B and setup.path_b is None and breakout_reclaim:
                    setup.path_b = Confirm(i, (h[i] + l[i]) / 2, i + RETEST_TIMEOUT_BARS)

            if setup.path_a and i > setup.path_a.expiry_i:
                setup.path_a = None
            if setup.path_b and i > setup.path_b.expiry_i:
                setup.path_b = None

            path = "A" if _retest(i, d, setup.path_a, l[i], h[i], c[i], atr[i]) else (
                "B" if _retest(i, d, setup.path_b, l[i], h[i], c[i], atr[i]) else None)
            if path is None:
                continue
            if not ((d == 1 and eligible_long) or (d == -1 and eligible_short)):
                continue
            if daily_count.get(int(days[i]), 0) >= MAX_TRADES_PER_DAY:
                continue
            stop_px = setup.extreme - STOP_ATR_BUFFER * atr[i] if d == 1 else (
                setup.extreme + STOP_ATR_BUFFER * atr[i])
            sd = d * (c[i] - stop_px)
            if not (np.isfinite(sd) and sd > 0 and MIN_STOP_ATR * atr[i] <= sd <= MAX_STOP_ATR * atr[i]):
                continue
            if sd < MIN_STOP_TO_COST * ROUND_TRIP_COST:
                continue
            candidates.append((d, path, float(sd), float(stop_px)))

        if candidates:
            if len({item[0] for item in candidates}) > 1:
                setups = {1: None, -1: None}
                continue
            d, path, sd, stop_px = candidates[0]
            sig[i] = d
            sl[i] = sd
            tp[i] = TARGET_RR * sd
            signal_path[i] = path
            daily_count[int(days[i])] = daily_count.get(int(days[i]), 0) + 1
            active = (d, float(sd), i, float(stop_px), float(c[i] + d * TARGET_RR * sd), i + MAX_HOLD_BARS)
            setups = {1: None, -1: None}

    return _pack(df, sig, sl, tp, signal_path)


def _pack(df, sig, sl, tp, path):
    output = pd.Series(sig, index=df.index, name="DTR_signal")
    output.attrs["stop_loss_distance"] = pd.Series(sl, index=df.index)
    output.attrs["take_profit_distance"] = pd.Series(tp, index=df.index)
    output.attrs["entry_path"] = pd.Series(path, index=df.index)
    return output
