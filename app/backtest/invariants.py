"""Cheap trade-level invariants the engine must satisfy on EVERY backtest (2026-11-06 audit, "invariant self-checks").

A whole class of engine bugs (a sign flip on a short, a stop that fills on the wrong side, a time-travelling exit, a NaN that slips
through) shows up as a trade that cannot exist. Checking for those costs microseconds and catches them on the run that produced them,
instead of in a result someone later trusts.

Only claims that are true by construction are asserted. In particular: a trade that exited at its ORIGINAL fixed stop (exit_cause
"stop_loss" -- trailing stops and breakeven stops are tagged separately and CAN be profitable) can never be profitable, because the
stop sits on the losing side of entry, a gap fills further away, and costs only subtract.

Policy (env ``T58_INVARIANTS``): ``warn`` (default: RuntimeWarning, run continues), ``raise`` (InvariantViolation), ``off``.
Results never change: this only reports.
"""
from __future__ import annotations

import math
import os
import warnings


class InvariantViolation(AssertionError):
    """The engine produced a trade that is impossible by construction."""


def policy() -> str:
    v = os.environ.get("T58_INVARIANTS", "warn").strip().lower()
    return v if v in ("warn", "raise", "off") else "warn"


def _finite(*xs) -> bool:
    return all(x is not None and isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def check_trade(t, index: int | None = None) -> list[str]:
    """Violations for one trade (empty list = clean)."""
    out: list[str] = []
    tag = f"trade #{index}" if index is not None else "trade"
    if not _finite(t.entry_price, t.exit_price, t.size, t.pnl, t.commission):
        out.append(f"{tag}: non-finite price/size/pnl/commission")
        return out
    if t.direction not in (1, -1):
        out.append(f"{tag}: direction {t.direction!r} is not +1/-1")
    if t.entry_price <= 0 or t.exit_price <= 0:
        out.append(f"{tag}: non-positive price (entry {t.entry_price}, exit {t.exit_price})")
    if t.size <= 0:
        out.append(f"{tag}: non-positive size {t.size}")
    if t.commission < -1e-9:
        out.append(f"{tag}: negative commission {t.commission} (costs cannot pay you)")
    try:
        if t.exit_time < t.entry_time:
            out.append(f"{tag}: exits ({t.exit_time}) before it enters ({t.entry_time})")
    except TypeError:
        pass
    if getattr(t, "exit_cause", None) == "stop_loss" and t.pnl > 1e-9:
        out.append(f"{tag}: exited at its original fixed stop yet made +{t.pnl:.2f} (a stop-out can never be profitable)")
    if getattr(t, "exit_cause", None) == "stop_loss" and t.direction in (1, -1):
        # the fixed stop sits on the losing side of entry; the fill can only be at or beyond it
        if t.direction * (t.exit_price - t.entry_price) > 1e-9 * max(1.0, abs(t.entry_price)):
            out.append(f"{tag}: stop-loss exit price {t.exit_price} is on the winning side of entry {t.entry_price}")
    return out


def check_trades(trades) -> list[str]:
    out: list[str] = []
    for i, t in enumerate(trades or []):
        out.extend(check_trade(t, i))
    return out


def enforce(trades, where: str = "backtest") -> list[str]:
    """Run the checks and apply the policy. Returns the violations found (also when only warning)."""
    pol = policy()
    if pol == "off":
        return []
    v = check_trades(trades)
    if v:
        msg = (f"Engine invariant violation in {where}: {len(v)} impossible trade(s). First: {v[0]}. "
               "This is an engine/data bug, not a strategy result -- do not trust this run's numbers.")
        if pol == "raise":
            raise InvariantViolation(msg + " | " + " | ".join(v[:5]))
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return v
