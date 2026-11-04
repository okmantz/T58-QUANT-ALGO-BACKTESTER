"""Speed preference for prop-firm strategies (2026-11-04 audit).

A strategy that passes 70% of the time inside 30 calendar days is worth more than one that passes 80% of the time but needs 150:
fees are billed by the month, capital and attention sit idle, and rules change under a slow account. Speed is therefore a RANKING and
SCORING input, never a reliability gate -- the pass-probability interval measures how sure we are the edge is real, and a slow edge is
exactly as real as a fast one.

``speed_factor(days)`` is 1.0 at or under ``REFERENCE_DAYS`` (30 calendar days, one billing period) and falls as
``(REFERENCE_DAYS / days) ** exponent``. With the default exponent 0.5:

    30 days -> 1.00    60 -> 0.71    90 -> 0.58    120 -> 0.50    150 -> 0.45    300 -> 0.32

so 70% @ 30 days scores 70.0 while 80% @ 150 days scores 35.8, and 90% @ 45 days (73.5) still beats 70% @ 30 days: a large reliability
gain can buy back a modest delay. The exponent is deliberately gentle; raise it to prefer speed harder.

Everything here takes CALENDAR days. ``app.prop`` also exposes ``days_to_pass`` as a TRADING-day count (days that contained a trade);
that number understates calendar time and must never be passed in here.

Pure functions, no app imports.
"""
from __future__ import annotations

import math

REFERENCE_DAYS = 30.0
DEFAULT_SPEED_EXPONENT = 0.5
TRADING_TO_CALENDAR = 7.0 / 5.0      # only for callers that hold a trading-day count and nothing better


def trading_to_calendar_days(trading_days: float | None) -> float | None:
    """Rough calendar days for a TRADING-day count (x7/5). Use only when the simulator's calendar figure is unavailable."""
    if trading_days is None:
        return None
    try:
        v = float(trading_days)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v * TRADING_TO_CALENDAR


def speed_factor(calendar_days: float | None, reference_days: float = REFERENCE_DAYS,
                 exponent: float = DEFAULT_SPEED_EXPONENT) -> float | None:
    """1.0 for <= reference_days, then (reference/days)**exponent. None when the time to pass is unknown (nothing passed)."""
    if calendar_days is None:
        return None
    try:
        d = float(calendar_days)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d) or d <= 0:
        return None
    ref = max(float(reference_days), 1e-9)
    if d <= ref:
        return 1.0
    return float((ref / d) ** max(float(exponent), 0.0))


def speed_adjusted_pass_pct(pass_pct: float | None, calendar_days: float | None, **kw) -> float | None:
    """pass probability (percent) x speed factor. A strategy that never passes has no time to pass and scores 0."""
    if pass_pct is None:
        return None
    f = speed_factor(calendar_days, **kw)
    if f is None:
        return 0.0 if float(pass_pct) <= 0.0 else float(pass_pct)   # no passes recorded but a non-zero rate: do not invent a penalty
    return float(pass_pct) * f


def speed_score(calendar_days: float | None, **kw) -> float | None:
    """0-100 scorecard component: 100 x speed_factor. None (missing, not zero) when the time to pass is unknown."""
    f = speed_factor(calendar_days, **kw)
    return None if f is None else 100.0 * f


def prefer_faster(a_pass_pct: float, a_days: float | None, b_pass_pct: float, b_days: float | None, **kw) -> int:
    """-1 when A is preferred, 1 when B is, 0 on an exact tie, comparing speed-adjusted pass rates."""
    a = speed_adjusted_pass_pct(a_pass_pct, a_days, **kw) or 0.0
    b = speed_adjusted_pass_pct(b_pass_pct, b_days, **kw) or 0.0
    return -1 if a > b else (1 if b > a else 0)
