"""Turn a Discovery Lab candidate into a standalone Python strategy file for the Strategy Library (then Full Pipeline, Validate...)."""
from __future__ import annotations

import json


def render(c: dict, instrument: str, minutes: int, name: str) -> str:
    cfg = {"family": c["family"], "params": c["params"]}
    tf = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 240: "4h", 1440: "1d"}.get(int(minutes))
    tests = c.get("test", {})
    return f'''"""
{name}
Found by the T58 Discovery Lab on {instrument} ({minutes}-minute bars).
Idea: {c.get("label", "")}
Train t={c.get("train", {}).get("t")}  validate t={c.get("val", {}).get("t")}  sealed test {tests.get("mean")}R over {tests.get("n")} trades.
A lead, not a verdict -- run it through Full Pipeline and Validate before trusting it.
Entries fire on the bar CLOSE that completes the setup (filled next bar). Stop = {c["stop_atr"]:g} x ATR(14), target = {c["target_r"]:g} x stop.
"""
import numpy as np
import pandas as pd

from app.discovery_lab import families as _F
from app.discovery_lab.engine import simulate

STRATEGY_NAME = {json.dumps(name)}
{("TIMEFRAME = " + json.dumps(tf)) if tf else ""}
CONFIG = {json.dumps(cfg)}
STOP_ATR = {c["stop_atr"]!r}
TARGET_R = {c["target_r"]!r}


def generate_signals(df: pd.DataFrame) -> pd.Series:
    lg, sh, ind = _F.signals(CONFIG["family"], df, CONFIG["params"])
    # The engine treats signals as LEVELS (flat when the signal returns to 0), so hold the side from the setup bar until this
    # trade's stop/target resolves -- the same non-overlapping trades the Discovery Lab simulated.
    ent, ex, _ = simulate(lg & ~sh, sh & ~lg, ind, STOP_ATR, TARGET_R, 0.0)
    level = np.zeros(len(df), dtype=int)
    for e, x in zip(ent, ex):
        level[e - 1:x] = 1 if lg[e - 1] else -1
    sig = pd.Series(level, index=df.index)
    atr = pd.Series(ind.atr(14), index=df.index)
    stop = (atr * STOP_ATR).replace(0, np.nan)
    sig.attrs["stop_loss_distance"] = stop.to_numpy()
    sig.attrs["take_profit_distance"] = (stop * TARGET_R).to_numpy()
    return sig
'''
