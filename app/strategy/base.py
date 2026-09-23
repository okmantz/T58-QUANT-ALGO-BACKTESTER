"""
Standardized strategy representation.

Regardless of source (manual rule builder, Python, PineScript, MQL5), every
strategy is reduced to a function that consumes an OHLCV DataFrame and
produces a standardized signal series. This lets the backtest engine remain
completely agnostic to strategy origin.

Signal convention:
    1  -> enter/hold long
    -1 -> enter/hold short
    0  -> flat / no position
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


class StrategyError(Exception):
    """Raised when a strategy is invalid or cannot be converted to signals."""


@dataclass
class StrategyResult:
    name: str
    source_type: str  # "manual" | "python" | "pinescript" | "mql5"
    signals: pd.Series  # indexed like the input dataframe, values in {-1, 0, 1}
    stop_loss_pips: float | None = None
    take_profit_pips: float | None = None
    # Optional per-bar stop/target distances in raw price units (e.g. an
    # ATR-multiple stop). When set, these take precedence over the fixed
    # pip-based fields above. Indexed like `signals`.
    stop_loss_distance: pd.Series | None = None
    take_profit_distance: pd.Series | None = None
    # Optional per-bar trailing-stop distance in raw price units (e.g. an
    # ATR-multiple trailing stop, fixed at trade entry).
    trailing_stop_distance: pd.Series | None = None
    # Move the stop to break-even once open profit reaches this multiple
    # of the trade's initial risk (e.g. 1.0 == "+1R").
    breakeven_trigger_r: float | None = None
    # Optional scale-out/partial-profit-taking config: {"r_multiple": float,
    # "fraction": float in (0, 1], "move_stop_to_breakeven": bool}. Once open
    # profit reaches `r_multiple` times the trade's initial risk, closes
    # `fraction` of the ORIGINAL position size at that level (a real
    # settled trade of its own, tagged exit_reason="partial_take_profit")
    # and, if move_stop_to_breakeven, tightens the remaining position's
    # stop to entry -- the common prop-firm "bank some of the daily-loss-
    # limit-safe portion of a trade early" technique. Fires at most once
    # per trade. None/omitted = no partial exit, identical to every
    # backtest run before this field existed.
    partial_exit: dict | None = None


class Strategy:
    """Base class all strategy adapters implement."""

    source_type: str = "base"

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        raise NotImplementedError

    @staticmethod
    def _validate_signals(signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        if len(signals) != len(df):
            raise StrategyError(
                f"Strategy produced {len(signals)} signals for {len(df)} bars; lengths must match."
            )
        signals = signals.fillna(0).clip(-1, 1).round().astype(int)
        return signals


def signals_from_conditions(
    index: pd.Index,
    long_entry: pd.Series,
    long_exit: pd.Series,
    short_entry: pd.Series,
    short_exit: pd.Series,
    allow_opposite_signal_flip: bool = True,
) -> pd.Series:
    """
    Shared stateful long/flat/short position loop used by every strategy
    adapter (Manual, PineScript, MQL5) once each has reduced its rules down
    to four boolean condition series. Keeping this in one place guarantees
    all strategy sources behave identically given the same conditions.

    allow_opposite_signal_flip: when True (default, and the only behavior
    prior versions had), an opposite-direction entry signal while a
    position is open immediately reverses it. When False ("Opposite Signal
    Exit" turned off in the Manual Strategy Builder), an opposite entry
    signal is ignored while a position is open; the position can only be
    closed by its own exit conditions, stop loss, take profit, a
    time-based exit, or the max-bars-in-trade limit.
    """
    le, lx = long_entry.values, long_exit.values
    se, sx = short_entry.values, short_exit.values

    position = 0
    out = np.zeros(len(index), dtype=int)
    for i in range(len(index)):
        if position == 0:
            if le[i]:
                position = 1
            elif se[i]:
                position = -1
        elif position == 1:
            if lx[i]:
                position = 0
            elif allow_opposite_signal_flip and se[i]:
                position = -1
        elif position == -1:
            if sx[i]:
                position = 0
            elif allow_opposite_signal_flip and le[i]:
                position = 1
        out[i] = position

    return pd.Series(out, index=index)


# ---------------------------------------------------------------------------
# Day-of-week trading restriction -- ONE shared, cross-source mechanism that
# any strategy (Manual, Python, PineScript, or MQL5) can use to exclude
# specific calendar days from trading entirely, independent of whatever
# entry/exit logic the strategy itself contains.
#
# This is deliberately separate from Manual Strategy Builder's own
# `day_of_week` CONDITION (see app.strategy.manual) -- that's an
# entry-condition a single Manual strategy can choose to build its signal
# around (e.g. "only enter on Tuesdays"). This is a blanket override applied
# to the FINAL signal series AFTER strategy.generate() returns (see
# apply_days_of_week_exclusion() below, and its one caller,
# app.backtest.engine.run_backtest -- the one chokepoint every tool in the
# app funnels through), so it forces flat (0) on every excluded day
# regardless of source type or what the strategy's own logic would
# otherwise have signaled, exactly matching a request like "don't let this
# specific strategy trade on Sundays" without needing to touch that
# strategy's actual entry/exit logic at all.
#
# Convention: 0=Monday .. 6=Sunday, matching pandas' own Series.dt.dayofweek
# (and Manual Strategy Builder's pre-existing `day_of_week` condition, for
# consistency across the app) -- NOT Python's calendar.weekday() (same
# numbering, mentioned only to rule out confusion) and NOT a 1-indexed or
# Sunday-first scheme.
#
# How each source declares it -- a strategy that declares nothing here is
# completely unaffected, byte-identical to every backtest before this
# existed:
#
#   Manual:      config["filters"] = {"days_of_week": {"exclude": [6]}}
#                (a new top-level `filters` block, parallel to the
#                existing `risk_management` block)
#
#   Python:      EXCLUDE_DAYS_OF_WEEK = [6]     (module-level constant,
#                same convention as STOP_LOSS_PIPS/TIMEFRAME/
#                RETRAIN_PER_FOLD -- see app.strategy.python's docstring)
#
#   PineScript:  // T58_EXCLUDE_DAYS=6           (comma-separated ints,
#   MQL5:        // T58_EXCLUDE_DAYS=6            e.g. "0,6" for Monday
#                and Sunday both -- same directive-comment convention as
#                T58_SL_PIPS/T58_TIMEFRAME/T58_HTF; see app.strategy.
#                pinescript's docstring for why a `//` directive is the
#                mechanism for these two languages)
# ---------------------------------------------------------------------------

_EXCLUDE_DAYS_DIRECTIVE_RE = re.compile(r"T58_EXCLUDE_DAYS\s*=\s*(\S+)")


def _coerce_day_list(raw) -> set[int]:
    """Normalizes whatever a source handed us (a list, a single int/str,
    or a comma-separated directive string) into a set of valid weekday
    ints (0-6). Anything unparsable or out of range is silently dropped
    rather than raising -- a typo'd day in a filter degrades to "no
    restriction for that entry," not a broken backtest."""
    if raw is None:
        return set()
    if isinstance(raw, (str, int)):
        raw = [raw]
    days: set[int] = set()
    for item in raw:
        try:
            day = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if 0 <= day <= 6:
            days.add(day)
    return days


def resolve_excluded_days_of_week(strategy) -> set[int]:
    """Which weekday(s) (0=Monday..6=Sunday) `strategy` has declared should
    never trade, regardless of source type. Returns an empty set for any
    strategy that declares nothing -- the fully backward-compatible
    default. See the module comment above for exactly how each source
    type declares this."""
    source_type = getattr(strategy, "source_type", None)

    if source_type == "manual":
        config = getattr(strategy, "config", None)
        if not isinstance(config, dict):
            return set()
        filters = config.get("filters", {}) or {}
        dow = filters.get("days_of_week", {}) or {}
        return _coerce_day_list(dow.get("exclude"))

    if source_type == "python":
        # PythonStrategy.module_attr() loads the module fresh and fails
        # soft (returns the default) rather than raising -- see its own
        # docstring in app.strategy.python.
        module_attr = getattr(strategy, "module_attr", None)
        if module_attr is None:
            return set()
        return _coerce_day_list(module_attr("EXCLUDE_DAYS_OF_WEEK", None))

    if source_type in ("pinescript", "mql5"):
        code = getattr(strategy, "code", None)
        if not code:
            return set()
        match = _EXCLUDE_DAYS_DIRECTIVE_RE.search(code)
        if not match:
            return set()
        return _coerce_day_list(match.group(1).split(","))

    return set()


def apply_days_of_week_exclusion(df: pd.DataFrame, signals: pd.Series, strategy) -> pd.Series:
    """Forces `signals` flat (0) on every bar that falls on a weekday
    `strategy` has excluded (see resolve_excluded_days_of_week above). A
    strategy that excludes nothing gets `signals` back completely
    unchanged -- byte-identical to before this feature existed. Requires
    df's standard `timestamp` column; silently a no-op without it rather
    than raising, since this must never be the reason a backtest that
    isn't even using this feature breaks. Purely positional (matches by
    row position, not by index label) since `signals` is only guaranteed
    to be the same LENGTH as `df`, not share its index."""
    excluded = resolve_excluded_days_of_week(strategy)
    if not excluded or "timestamp" not in df.columns:
        return signals
    dow = pd.to_datetime(df["timestamp"]).dt.dayofweek.to_numpy()
    mask = np.isin(dow, list(excluded))
    if not mask.any():
        return signals
    arr = signals.to_numpy(copy=True)
    arr[mask] = 0
    return pd.Series(arr, index=signals.index)
