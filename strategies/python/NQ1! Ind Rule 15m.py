"""
futures_NQ.F Ind rule 15m
Found by the T58 Discovery Lab on futures_NQ.F (15-minute bars).
Idea: Any-indicator rule (kind=base_signal_expectancy period=7 win=250 k=2.0 mode=trend) [side=long, trend=200, hours=afternoon] | stop 3 ATR, target 0.75R, exit after 120 bars
Drawn from: Indicator pool: every causal indicator in the app (base + v7-v13) against the same exits and filters as every other family.
Train t=4.783  validate t=1.53  sealed test 0.2322R over 30 trades.
Estimated evaluation pass chance 50.3% at 1.5% risk per trade (a bootstrap estimate, not a promise).
A lead, not a verdict -- run it through Full Pipeline and Validate before trusting it.
Entries fire on the bar CLOSE that completes the setup (filled next bar). Stop = 3 x ATR(14), target = 0.75 x stop, time limit 120 bars.
The signal is causal: it never looks at a bar after the one it is deciding on.
"""
import numpy as np
import pandas as pd

from app.discovery_lab import families as _F

STRATEGY_NAME = "futures_NQ.F Ind rule 15m"
TIMEFRAME = "15m"
CONFIG = {"family": "ind_rule", "params": {"kind": "base_signal_expectancy", "period": 7, "win": 250, "k": 2.0, "mode": "trend"}}
FILTERS = {"side": "long", "trend": 200, "hours": "afternoon"}
STOP_ATR = 3
TARGET_R = 0.75
HOLD_BARS = 120
WARMUP = 250


def _levels(lg, sh, o, h, l, c, atr):
    """Position carried bar by bar. level[t] is the side to hold AFTER bar t closes (+1 long, -1 short, 0 flat). It depends only on
    bars up to t: a trade ends on the bar whose range reaches its stop or target, or HOLD_BARS after entry; at the end of the data an
    unresolved trade is simply still open."""
    n = len(c)
    level = np.zeros(n, dtype=np.int8)
    sig = np.flatnonzero(lg ^ sh)
    free = 0
    for i in sig:
        if i < free or i < WARMUP:
            continue
        a = atr[i]
        if not (a > 0):
            continue
        d = 1 if lg[i] else -1
        if i >= n - 1:
            # The setup completes on the LAST bar we have. The fill (next bar's open) has not happened yet, but the decision -- "be
            # long/short after this bar closes" -- is already known from bars 0..i. Dropping it here (the old behaviour) made the
            # signal on the final bar differ from the signal on the very same bar when more data follows, which is exactly what the
            # lookahead test flags: it truncates the data right after a signal bar. Emit it and stop; nothing later is knowable.
            level[i:] = d
            break
        e = o[i + 1]
        sd = STOP_ATR * a
        stop = e - d * sd
        tgt = e + d * TARGET_R * sd
        jend = i + HOLD_BARS
        exit_j = -1
        for j in range(i + 1, min(jend, n - 1) + 1):
            if d == 1:
                hit = (o[j] <= stop) or (l[j] <= stop) or (o[j] >= tgt) or (h[j] >= tgt)
            else:
                hit = (o[j] >= stop) or (h[j] >= stop) or (o[j] <= tgt) or (l[j] <= tgt)
            if hit:
                exit_j = j
                break
        if exit_j < 0 and jend <= n - 1:
            exit_j = jend                       # time limit reached inside the data
        if exit_j < 0:                          # still open when the data ends
            level[i:] = d
            break
        level[i:exit_j] = d                     # flat again from the exit bar itself
        free = exit_j + 1
    return level


def generate_signals(df: pd.DataFrame) -> pd.Series:
    ind = _F.Indicators(df)
    lg, sh = _F.FAMILIES[CONFIG["family"]].build(ind, CONFIG["params"])
    lg, sh = np.nan_to_num(lg, nan=False).astype(bool), np.nan_to_num(sh, nan=False).astype(bool)
    lg, sh = _F.apply_filters(ind, lg, sh, FILTERS)
    atr = np.nan_to_num(ind.atr(14), nan=0.0)
    level = _levels(lg, sh, ind.o, ind.h, ind.l, ind.c, atr)
    sig = pd.Series(level.astype(int), index=df.index)
    stop = pd.Series(atr * STOP_ATR, index=df.index).replace(0, np.nan)
    sig.attrs["stop_loss_distance"] = stop.to_numpy()
    sig.attrs["take_profit_distance"] = (stop * TARGET_R).to_numpy()
    return sig
