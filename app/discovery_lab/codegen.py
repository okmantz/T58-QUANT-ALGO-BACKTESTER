"""Turn a Discovery Lab candidate into a standalone Python strategy file for the Strategy Library (then Full Pipeline, Validate...).

The saved strategy is CAUSAL by construction: the signal on bar t is decided only from bars 0..t. The previous version simulated the whole
history up front to decide when to hold a position, which silently used the end of the data as a trade exit -- so the signal on the last bars
changed whenever data was cut, and Full Pipeline's lookahead test (correctly) rejected it. This one carries a position through the bars
exactly as a live system would: it goes long/short on the setup bar, stays in while the stop or target has not been hit, goes flat on the
bar where either is hit (or after the time limit), and simply stays in at the end of the data.
"""
from __future__ import annotations

import json


_TF = {1: "1m", 3: "3m", 5: "5m", 10: "10m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 1440: "1d"}


def tf_text(minutes) -> str:
    """Bar size in minutes -> the label used in file names and TIMEFRAME ('15m', '1h', '4h', '1d')."""
    m = int(minutes or 0)
    if m <= 0:
        return ""
    return _TF.get(m) or (f"{m // 1440}d" if m % 1440 == 0 else f"{m // 60}h" if m % 60 == 0 else f"{m}m")


def render(c: dict, instrument: str, minutes: int, name: str) -> str:
    cfg = {"family": c["family"], "params": c["params"]}
    flt = c.get("flt") or {}
    tf = tf_text(minutes) or None
    tests = c.get("test", {})
    hold = int(c.get("hold", 400))
    target = float(c["target_r"])
    no_target = target >= 50
    src = c.get("family_source") or ""
    eng = c.get("engine") or {}
    return f'''"""
{name}
Found by the T58 Discovery Lab on {instrument} ({minutes}-minute bars).
Idea: {c.get("label", "")}
{("Drawn from: " + src) if src else ""}
Train t={c.get("train", {}).get("t")}  validate t={c.get("val", {}).get("t")}  sealed test {tests.get("mean")}R over {tests.get("n")} trades.
{("Estimated evaluation pass chance " + str(c.get("pass_prob")) + "% at " + str(c.get("suggested_risk_pct")) + "% risk per trade (a bootstrap estimate, not a promise).") if c.get("pass_prob") is not None else ""}
A lead, not a verdict -- run it through Full Pipeline and Validate before trusting it.
Entries fire on the bar CLOSE that completes the setup (filled next bar). Stop = {c["stop_atr"]:g} x ATR(14), {"no profit target" if no_target else "target = " + format(target, "g") + " x stop"}, time limit {hold} bars.
The signal is causal: it never looks at a bar after the one it is deciding on.
"""
import numpy as np
import pandas as pd

from app.discovery_lab import families as _F

STRATEGY_NAME = {json.dumps(name)}
{("TIMEFRAME = " + json.dumps(tf)) if tf else ""}
CONFIG = {json.dumps(cfg)}
FILTERS = {json.dumps(flt)}
STOP_ATR = {c["stop_atr"]!r}
TARGET_R = {target!r}
HOLD_BARS = {hold!r}
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
    sig.attrs["take_profit_distance"] = {"(stop * TARGET_R).to_numpy()" if not no_target else "np.full(len(df), np.nan)"}
    return sig
'''
