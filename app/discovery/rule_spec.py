"""
RULE SPEC -- the one declarative, validated description of a tradeable idea.

Whatever produces it (a person filling a form, the deterministic keyword
compiler, a language model) the result is a plain dict:

    {"kind": "donchian_breakout", "params": {"lookback": 40, "stop_atr": 2.0, "target_r": 2.0},
     "direction": "both"}

`validate_spec` accepts ONLY whitelisted kinds and clamps nothing silently: an
unknown kind/param or an out-of-bounds value is an error that names the field
and the allowed range, so a model hallucination can never reach the backtester.
`build_plan(df, spec)` turns a valid spec into either a per-bar signal plan
(market-order ideas -> run_execution) or a list of resting limit orders
(zone ideas -> app.backtest.resting_orders). No lookahead: every condition at
bar i uses data up to and including bar i; the engine fills at the next open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# kind -> {param: (default, lo, hi, is_int)}
SPEC_SCHEMA: dict[str, dict[str, tuple]] = {
    "ma_cross": {"fast": (20, 2, 100, True), "slow": (60, 5, 400, True), "stop_atr": (2.0, 0.5, 8.0, False), "target_r": (2.0, 0.0, 8.0, False)},
    "donchian_breakout": {"lookback": (40, 5, 300, True), "stop_atr": (2.0, 0.5, 8.0, False), "target_r": (2.0, 0.0, 8.0, False)},
    "rsi_reversion": {"period": (14, 2, 40, True), "lo": (30, 3, 45, False), "hi": (70, 55, 97, False), "stop_atr": (2.0, 0.5, 8.0, False), "target_r": (1.5, 0.0, 8.0, False)},
    "zscore_reversion": {"lookback": (50, 10, 300, True), "z_entry": (2.0, 0.8, 4.5, False), "stop_atr": (2.5, 0.5, 8.0, False), "target_r": (1.5, 0.0, 8.0, False)},
    "momentum": {"lookback": (50, 5, 400, True), "threshold_atr": (1.0, 0.0, 6.0, False), "stop_atr": (2.5, 0.5, 8.0, False), "target_r": (2.0, 0.0, 8.0, False)},
    "fvg_retest": {"min_gap_atr": (0.3, 0.05, 3.0, False), "expiry_bars": (30, 3, 200, True), "stop_buffer_atr": (0.25, 0.0, 3.0, False), "target_r": (2.0, 0.5, 8.0, False)},
}
DIRECTIONS = ("both", "long", "short")


class SpecError(ValueError):
    pass


def validate_spec(spec: dict) -> dict:
    """Returns a normalized copy (defaults filled, ints coerced) or raises SpecError."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be an object")
    kind = spec.get("kind")
    if kind not in SPEC_SCHEMA:
        raise SpecError(f"unknown kind {kind!r}; allowed: {sorted(SPEC_SCHEMA)}")
    schema = SPEC_SCHEMA[kind]
    params_in = spec.get("params") or {}
    if not isinstance(params_in, dict):
        raise SpecError("params must be an object")
    unknown = set(params_in) - set(schema)
    if unknown:
        raise SpecError(f"unknown params for {kind}: {sorted(unknown)}; allowed: {sorted(schema)}")
    params = {}
    for name, (default, lo, hi, is_int) in schema.items():
        v = params_in.get(name, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise SpecError(f"{kind}.{name} must be a number, got {v!r}")
        if not np.isfinite(v) or v < lo or v > hi:
            raise SpecError(f"{kind}.{name}={v:g} outside allowed range [{lo:g}, {hi:g}]")
        params[name] = int(round(v)) if is_int else v
    if kind == "ma_cross" and params["fast"] >= params["slow"]:
        raise SpecError("ma_cross.fast must be smaller than ma_cross.slow")
    if kind == "rsi_reversion" and params["lo"] >= params["hi"]:
        raise SpecError("rsi_reversion.lo must be below rsi_reversion.hi")
    direction = spec.get("direction", "both")
    if direction not in DIRECTIONS:
        raise SpecError(f"direction must be one of {DIRECTIONS}")
    return {"kind": kind, "params": params, "direction": direction}


def default_spec(kind: str) -> dict:
    return validate_spec({"kind": kind})


def spec_param_ranges(kind: str) -> dict:
    return {k: (lo, hi, is_int) for k, (d, lo, hi, is_int) in SPEC_SCHEMA[kind].items()}


# ----------------------------------------------------------------------------
@dataclass
class SignalPlan:
    signals: pd.Series | None = None
    stop_loss_distance: pd.Series | None = None
    take_profit_distance: pd.Series | None = None
    orders: list = field(default_factory=list)       # LimitOrder list for resting ideas
    mode: str = "signals"                             # "signals" | "resting"


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    c = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c).abs(), (df["low"] - c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _hold(entry_long, entry_short, exit_long, exit_short) -> np.ndarray:
    n = len(entry_long)
    out = np.zeros(n, dtype=np.int8)
    pos = 0
    for i in range(n):
        if pos == 0:
            if entry_long[i]:
                pos = 1
            elif entry_short[i]:
                pos = -1
        elif pos == 1:
            if entry_short[i]:
                pos = -1
            elif exit_long[i]:
                pos = 0
        else:
            if entry_long[i]:
                pos = 1
            elif exit_short[i]:
                pos = 0
        out[i] = pos
    return out


def build_plan(df: pd.DataFrame, spec: dict) -> SignalPlan:
    spec = validate_spec(spec)
    kind, p, direction = spec["kind"], spec["params"], spec["direction"]
    close, high, low = df["close"], df["high"], df["low"]
    atr = _atr(df)

    if kind == "fvg_retest":
        from app.backtest.resting_orders import zones_fvg
        orders = zones_fvg(df, min_gap_atr=p["min_gap_atr"], expiry_bars=p["expiry_bars"],
                           stop_buffer_atr=p["stop_buffer_atr"], target_r=p["target_r"])
        if direction == "long":
            orders = [o for o in orders if o.side == 1]
        elif direction == "short":
            orders = [o for o in orders if o.side == -1]
        return SignalPlan(orders=orders, mode="resting")

    z = np.zeros(len(df), dtype=bool)
    if kind == "ma_cross":
        f = close.rolling(p["fast"]).mean()
        s = close.rolling(p["slow"]).mean()
        el = ((f > s) & (f.shift(1) <= s.shift(1))).to_numpy()
        es = ((f < s) & (f.shift(1) >= s.shift(1))).to_numpy()
        sig = _hold(el, es, z, z)
    elif kind == "donchian_breakout":
        hi = high.rolling(p["lookback"]).max().shift(1)
        lo = low.rolling(p["lookback"]).min().shift(1)
        sig = _hold((close > hi).to_numpy(), (close < lo).to_numpy(), z, z)
    elif kind == "rsi_reversion":
        d = close.diff()
        up = d.clip(lower=0).ewm(alpha=1 / p["period"], adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / p["period"], adjust=False).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        sig = _hold((rsi < p["lo"]).to_numpy(), (rsi > p["hi"]).to_numpy(), (rsi > 50).to_numpy(), (rsi < 50).to_numpy())
    elif kind == "zscore_reversion":
        m = close.rolling(p["lookback"]).mean()
        sd = close.rolling(p["lookback"]).std()
        zs = (close - m) / sd.replace(0, np.nan)
        sig = _hold((zs < -p["z_entry"]).to_numpy(), (zs > p["z_entry"]).to_numpy(), (zs > -0.2).to_numpy(), (zs < 0.2).to_numpy())
    else:  # momentum
        mv = (close - close.shift(p["lookback"])) / atr
        sig = _hold((mv > p["threshold_atr"]).to_numpy(), (mv < -p["threshold_atr"]).to_numpy(),
                    (mv < 0).to_numpy(), (mv > 0).to_numpy())

    if direction == "long":
        sig = np.where(sig > 0, sig, 0)
    elif direction == "short":
        sig = np.where(sig < 0, sig, 0)
    stop = (atr * p["stop_atr"])
    tp = (stop * p["target_r"]) if p.get("target_r") else None
    return SignalPlan(
        signals=pd.Series(sig.astype(int), index=df.index),
        stop_loss_distance=stop, take_profit_distance=tp, mode="signals",
    )
