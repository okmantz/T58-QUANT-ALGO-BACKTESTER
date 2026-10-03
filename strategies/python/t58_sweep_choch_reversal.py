# =============================================================================
# EXPERIMENTAL -- DO NOT FUND WITHOUT FORWARD VALIDATION
# =============================================================================
# T58 Sweep-ChoCH Reversal did NOT pass the acceptance contract:
#   1 of 9 criteria (only the zero-lookahead-warnings check).
#   Per-attempt eval-pass 41.8% (bar: 70%) | payout 30.1% (bar: 50%) |
#   risk of ruin 60.1% (bar: <=10%) | walk-forward efficiency 0.119 (bar: 0.6) |
#   86 trades (bar: >=100) | maxDD -3.25% (bar: <=2%).
# MNQ-only thin edge (PF 1.19); MGC and MES variants were negative on every
# configuration tested. Genuine T58 market-structure/liquidity/supply-demand
# logic (conformance 90/100), intraday only, flat 17:00 UTC, $200 fixed risk.
# Full validation + gap analysis: strategy-review-2026-10-03/new-strategy.md
# (2026-10-03). Logic below is UNCHANGED from the validated finalist.
# =============================================================================

"""
T58 Sweep-ChoCH Reversal -- intraday reversal strategy for micro futures (MGC / MNQ / MES).

FINALIST CONFIGURATION (MNQ): parameters below are the output of the app's own
Quick Optimize walk-forward-aware GA (fitness = eval-pass probability, 3 chained
out-of-sample folds, locked 80/20 holdout), seeded from a hand-designed base.
See new-strategy.md for full validation and the honest gap analysis -- this
configuration does NOT clear the acceptance contract.

Genuine Market Structure / Liquidity / Supply-Demand logic (no relabeled moving-average crossovers):

  1. MARKET STRUCTURE first -- change of character from a fractal-swing
     detector that ports app.quant_lab.market_structure's swing/label/
     trend/event logic one-for-one, with one fix: same-bar swings are
     ordered deterministically (lows before highs), because that module's
     unstable timestamp sort makes bos/choch attribution differ between a
     full run and a truncated run (the app's own lookahead gate flags it).
     A structure event is emitted on its CONFIRMATION bar (swing position +
     right bars) -- the first bar a live trader could have known, so there
     is zero lookahead. Direction comes from the swing label itself:
     ChoCH+Higher-High = bullish, ChoCH+Lower-Low = bearish.
  2. LIQUIDITY second -- engineered liquidity raid AFTER the structure
     shift: the bar wicks through the prior SWEEP_LOOKBACK-bar extreme
     (sell-side liquidity below the lows for longs, buy-side above the highs
     for shorts) and closes back inside. Classic stop-hunt / turtle-soup
     footprint, now trading WITH the fresh structure instead of against it.
  3. SUPPLY/DEMAND location -- premium/discount: the dealing range is the
     prior DEAL_RANGE_BARS-bar high/low (shifted one bar so the current bar
     can never define its own range). Longs only in discount (close below
     equilibrium + buffer), shorts only in premium.

Entry (all conditions on the same 15m bar, AND) -- two entry modes:
  ENTRY_MODE="sweep" (default):
    long  = bullish liquidity sweep on this bar
            AND a bullish structure confirmation within the last STRUCT_LOOKBACK bars
            AND close in discount AND 1h HTF momentum up AND in session
  ENTRY_MODE="breakout":
    long  = bullish structure confirmation ON this bar
            AND 1h HTF momentum up AND in session (premium/discount optional)
  short = mirror image.

Exits: ATR-based stop/target (attached per-trade via signals.attrs so the
engine sizes and protects exactly what the strategy intends), a
MAX_BARS_IN_TRADE cap, and a hard time flatten -- no entries after
ENTRY_END, flat at FLAT_TIME. The strategy NEVER holds overnight.

Causality audit (every input at bar i uses only bars <= i):
  - sweep: low/high/close of bar i vs rolling extremes of bars < i (shift(1)).
  - Structure: fractal swing at position p confirmed at p + STRUCT_WINDOW; the
    event series is 1.0 only on the confirmation bar.
  - dealing range: high/low shifted by one bar before the rolling window.
  - HTF bias: tf60_* columns merged by the engine's own lookahead-safe
    higher-timeframe merge (only fully-closed 1h bars are ever visible).
  - session clock: the bar's own timestamp.

Tunable parameters are top-level SCREAMING_SNAKE_CASE numbers so the app's
own optimizer (Quick Optimize / Evolution Lab code path) can tune them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "T58 Sweep-ChoCH Reversal (micros, 15m)"

TIMEFRAME = "15m"
HTF_TIMEFRAMES = ["1h"]
WARMUP_BARS = 320

# ---------------------------------------------------------------------------
# Tunable parameters (the app's optimizer patches these SCREAMING_SNAKE_CASE
# numeric literals directly -- keep them top-level plain assignments).
# ---------------------------------------------------------------------------
SWEEP_LOOKBACK = 8          # bars defining the liquidity reference extreme
STRUCTURE_MODE = "choch"    # "choch" = reversal (fade the sweep with fresh structure);
                            # "bos" = continuation (buy the sweep-dip in the trend direction)
ENTRY_MODE = "sweep"        # "sweep" = enter on the sweep-reclaim bar (needs prior structure);
                            # "breakout" = enter on the structure confirmation bar itself
STRUCT_WINDOW = 5           # fractal swing window (left=right) for the real structure detector
STRUCT_LOOKBACK = 15
DEAL_RANGE_BARS = 63        # dealing-range lookback in 15m bars (~16h)
PREM_DISC_BUFFER_ATR = 0  # premium/discount buffer in ATRs (0 = strict equilibrium line)
ATR_PERIOD = 13
STOP_ATR_MULT = 1.2
TARGET_ATR_MULT = 3
HTF_BIAS_BARS = 6           # 1h bars for higher-timeframe momentum bias
MAX_BARS_IN_TRADE = 12      # 12 x 15m = 3h cap
USE_HTF_BIAS = 1            # 1 = require 1h momentum alignment, 0 = no HTF filter
USE_PREM_DISC = 1           # 1 = require premium/discount location, 0 = trade anywhere
TRADE_LONGS = 1             # 1 = take long setups, 0 = longs disabled
TRADE_SHORTS = 1            # 1 = take short setups, 0 = shorts disabled

# Session clock (UTC -- data timestamps are UTC). Entries only in the US
# morning window; hard flatten well before the close. Never holds overnight.
ENTRY_START_MIN = 720
ENTRY_END_MIN = 16 * 60 + 30     # 16:30 UTC
FLAT_MIN = 17 * 60               # 17:00 UTC


def _structure_events(df: pd.DataFrame, window: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Bullish/bearish structure confirmation series from fractal swings.

    This is a faithful, DETERMINISTIC port of
    app.quant_lab.market_structure.calculate_hh_ll_structure's detection +
    labeling + trend/event logic. It exists because that module's
    ``sort_values("timestamp")`` uses pandas' default quicksort, which is NOT
    stable: two swings on the SAME bar (a bar that is simultaneously a fractal
    high and a fractal low) can come out in either order depending on the
    total frame length, so the running-trend attribution -- and hence whether
    an event is classified "bos" or "choch" -- differs between a full run and
    a truncated run. The app's own lookahead gate correctly flags that as
    lookahead. Here, same-bar swings are always processed lows-before-highs,
    making the event series a pure causal function of bars <= the
    confirmation bar: identical on any truncation.

    mode="choch": change of character (HH vs prior downtrend / LL vs prior
      uptrend) -- reversal structure.
    mode="bos": break of structure IN the trend direction (HH while already up
      / LL while already down) -- continuation structure.
    Returns two boolean numpy arrays (positional, len(df)) True only on the
    confirmation bar -- no lookahead by construction."""
    n = len(df)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    window = max(int(window), 1)
    mode = str(mode).lower()
    if n < 2 * window + 1 or mode not in ("choch", "bos"):
        return bull, bear

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    # 1. Fractal swings (same definition as the reference detector: unique
    #    max/min over [i-left, i+right]).
    swings: list[tuple[int, str, float]] = []  # (pos, kind, price)
    for i in range(window, n - window):
        wh = highs[i - window: i + window + 1]
        wl = lows[i - window: i + window + 1]
        if highs[i] == np.nanmax(wh) and np.sum(wh == highs[i]) == 1:
            swings.append((i, "high", float(highs[i])))
        if lows[i] == np.nanmin(wl) and np.sum(wl == lows[i]) == 1:
            swings.append((i, "low", float(lows[i])))
    # 2. Deterministic order: chronological, lows before highs on the same bar.
    swings.sort(key=lambda s: (s[0], 0 if s[1] == "low" else 1))

    # 3. HH/HL/LH/LL labeling + running trend + bos/choch events (same rules
    #    as the reference detector).
    last_high = None
    last_low = None
    trend = "range"
    for pos, kind, price in swings:
        if kind == "high":
            label = "H" if last_high is None else ("HH" if price > last_high else "LH")
            last_high = price
        else:
            label = "L" if last_low is None else ("HL" if price > last_low else "LL")
            last_low = price
        event = None
        if label == "HH":
            if trend == "down":
                event = "choch"
            elif trend == "up":
                event = "bos"
            trend = "up"
        elif label == "LL":
            if trend == "up":
                event = "choch"
            elif trend == "down":
                event = "bos"
            trend = "down"
        elif label == "HL":
            if trend != "down":
                trend = "up"
        elif label == "LH":
            if trend != "up":
                trend = "down"
        if event == mode:
            confirm = pos + window
            if 0 <= confirm < n:
                if label == "HH":
                    bull[confirm] = True
                elif label == "LL":
                    bear[confirm] = True
    return bull, bear


# (legacy name kept for compatibility)
def _choch_events(df: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
    return _structure_events(df, window, "choch")


def generate_signals(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    flat = pd.Series(0, index=df.index, dtype=int)
    if n < 2 * STRUCT_WINDOW + 1:
        return flat

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    # ---- 1. Liquidity sweep (causal: reference extremes use bars < i) ----
    lb = max(int(SWEEP_LOOKBACK), 1)
    prior_low = pd.Series(low).shift(1).rolling(lb, min_periods=lb).min().to_numpy()
    prior_high = pd.Series(high).shift(1).rolling(lb, min_periods=lb).max().to_numpy()
    sweep_bull = (low < prior_low) & (close > prior_low)     # sell-side raid, reclaimed
    sweep_bear = (high > prior_high) & (close < prior_high)  # buy-side raid, rejected

    # ---- 2. Market structure: real fractal ChoCH, confirmation-bar aligned --
    # The ChoCH must PRECEDE the sweep: a structure shift, then a liquidity
    # raid against it that gets reclaimed -- the turtle-soup entry.
    struct_bull, struct_bear = _structure_events(df, STRUCT_WINDOW, STRUCTURE_MODE)
    cl = max(int(STRUCT_LOOKBACK), 1)
    prior_struct_bull = pd.Series(struct_bull).shift(1).rolling(cl, min_periods=1).max().to_numpy() > 0
    prior_struct_bear = pd.Series(struct_bear).shift(1).rolling(cl, min_periods=1).max().to_numpy() > 0

    # ---- 3. Premium/discount location (dealing range excludes current bar) --
    # ATR is computed here (before entries) so the buffer can use it.
    from app.strategy.indicators import atr as atr_fn
    atr_vals = atr_fn(df, period=max(int(ATR_PERIOD), 1)).to_numpy(dtype=float)
    drb = max(int(DEAL_RANGE_BARS), 10)
    dr_high = pd.Series(high).shift(1).rolling(drb, min_periods=drb).max().to_numpy()
    dr_low = pd.Series(low).shift(1).rolling(drb, min_periods=drb).min().to_numpy()
    equilibrium = (dr_high + dr_low) / 2.0
    buf = PREM_DISC_BUFFER_ATR * atr_vals
    in_discount = close < equilibrium + buf
    in_premium = close > equilibrium - buf

    # ---- 4. Higher-timeframe bias (engine-merged, only closed 1h bars) -----
    htf_close = pd.to_numeric(df.get("tf60_close", pd.Series(np.nan, index=df.index)),
                              errors="coerce")
    htf_bars = max(int(HTF_BIAS_BARS), 1)
    htf_up = (htf_close > htf_close.shift(htf_bars)).fillna(False).to_numpy()
    htf_down = (htf_close < htf_close.shift(htf_bars)).fillna(False).to_numpy()

    # ---- 5. Session clock -------------------------------------------------
    ts = pd.to_datetime(df["timestamp"])
    mins = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    in_session = (mins >= ENTRY_START_MIN) & (mins < ENTRY_END_MIN)
    flat_zone = mins >= FLAT_MIN

    # ---- entries ----------------------------------------------------------
    htf_long_ok = htf_up if USE_HTF_BIAS else np.ones(n, dtype=bool)
    htf_short_ok = htf_down if USE_HTF_BIAS else np.ones(n, dtype=bool)
    pd_long_ok = in_discount if USE_PREM_DISC else np.ones(n, dtype=bool)
    pd_short_ok = in_premium if USE_PREM_DISC else np.ones(n, dtype=bool)
    if str(ENTRY_MODE).lower() == "breakout":
        # Enter ON the structure confirmation bar: buy the break itself.
        long_entry = struct_bull & pd_long_ok & htf_long_ok & in_session
        short_entry = struct_bear & pd_short_ok & htf_short_ok & in_session
    else:
        # Enter on the sweep-reclaim bar after a structure confirmation.
        long_entry = sweep_bull & prior_struct_bull & pd_long_ok & htf_long_ok & in_session
        short_entry = sweep_bear & prior_struct_bear & pd_short_ok & htf_short_ok & in_session

    # ---- ATR stop/target distances (attached per entry bar) ---------------
    stop_dist = STOP_ATR_MULT * atr_vals
    target_dist = TARGET_ATR_MULT * atr_vals

    # ---- stateful position loop: max bars + hard time flatten -------------
    pos = np.zeros(n, dtype=int)
    stop_attr = np.full(n, np.nan)
    target_attr = np.full(n, np.nan)
    position = 0
    bars_held = 0
    max_bars = max(int(MAX_BARS_IN_TRADE), 1)
    for i in range(n):
        if position == 0:
            if flat_zone[i]:
                continue                      # never enter in/after the flatten zone
            if long_entry[i] and TRADE_LONGS:
                position, bars_held = 1, 0
            elif short_entry[i] and TRADE_SHORTS:
                position, bars_held = -1, 0
        else:
            bars_held += 1
            if bars_held >= max_bars or flat_zone[i]:
                position, bars_held = 0, 0    # max-bars cap or session flatten
        pos[i] = position
        if position != 0 and bars_held == 0:
            stop_attr[i] = stop_dist[i]
            target_attr[i] = target_dist[i]

    signals = pd.Series(pos, index=df.index, dtype=int)
    signals.attrs["stop_loss_distance"] = pd.Series(stop_attr, index=df.index)
    signals.attrs["take_profit_distance"] = pd.Series(target_attr, index=df.index)
    return signals
