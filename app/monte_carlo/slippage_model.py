"""
Session- and volatility-aware slippage model for Monte Carlo.

Before this file existed, app.monte_carlo.engine's only slippage
modeling was `_apply_slippage_stress`: a single flat percentage applied
identically to every trade's P&L regardless of when it happened or how
volatile the market was at the time. That's a reasonable stress-test
knob, but it can't distinguish a trade opened during the London/New York
overlap (deep liquidity, typically tight fills) from one opened during
the Sydney/Tokyo session dead zone or straight into a weekend gap
(thin liquidity, realistically much worse fills) -- which is exactly
where Monte Carlo output is least trustworthy for a real prop-firm
account, since those are also the conditions where a real broker's
execution most diverges from a backtest's assumed close-price fill.

WHAT THIS ACTUALLY MODELS, honestly: each historical Trade already
carries its real `entry_time` (session-classifiable) and `initial_risk`
(the |entry - stop| distance the strategy itself chose at entry, in raw
price units) -- the ONLY per-trade signal available at this stage of the
pipeline without re-reading the original OHLC dataset. `initial_risk`
relative to `entry_price` is used here as a volatility PROXY: a strategy
that set a wider stop (in % of price terms) at entry was very likely
responding to a more volatile market right then, via whatever
ATR/indicator logic it uses (see app.backtest.execution's stop-distance
resolution). This is not a substitute for a real historical spread/
volatility feed -- it's the best signal already flowing through this
app's existing trade objects, used honestly rather than inventing a
volatility number no upstream module actually computed.

Applied ONCE, deterministically, to the historical trade P&L pool BEFORE
Monte Carlo resampling -- not re-randomized per simulation the way
`_apply_slippage_stress` is -- because a given historical trade's session
and volatility-at-entry are fixed facts about when it happened, not
something that should vary run-to-run. The existing flat
`slippage_stress_pct` in MonteCarloConfig can still be layered on top
afterward as an extra, run-time stress multiplier; the two are
independent and both optional.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtest.execution import Trade

# UTC hour ranges for the four major FX sessions. Overlaps are the
# deepest-liquidity windows (tightest realistic slippage); the Sydney/
# Tokyo-only stretch and the pre-Sydney "dead zone" are the thinnest.
_SESSION_RANGES = {
    "sydney_tokyo": (21, 7),     # 21:00-07:00 UTC (wraps midnight)
    "london": (7, 12),
    "london_ny_overlap": (12, 16),
    "new_york": (16, 21),
}


def classify_session(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    hour = ts.hour
    if hour >= 12 and hour < 16:
        return "london_ny_overlap"
    if hour >= 7 and hour < 12:
        return "london"
    if hour >= 16 and hour < 21:
        return "new_york"
    return "sydney_tokyo"  # 21:00-07:00, wrapping midnight


def is_weekend_open_gap(ts: pd.Timestamp) -> bool:
    """True for a trade entered within the first hour of the week's
    trading (Sunday 21:00-22:00 UTC through Monday's open, roughly) --
    the single worst realistic-slippage window for FX/CFD, where a
    backtest's assumed fill-at-close is most likely to be badly wrong
    against a real broker's Sunday-open gap."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return (ts.weekday() == 6 and ts.hour >= 21) or (ts.weekday() == 0 and ts.hour < 1)


@dataclass
class SessionVolatilitySlippageConfig:
    enabled: bool = False
    base_slippage_pct_of_price: float = 0.00005  # ~0.5 pip on a 5-digit FX pair, before multipliers
    session_multipliers: dict = None  # filled in __post_init__ with the defaults below
    volatility_sensitivity: float = 1.0  # scales how strongly wider initial_risk (as % of price) increases slippage
    weekend_gap_multiplier: float = 6.0  # extra multiplier for trades entered in the Sunday-open gap window

    def __post_init__(self):
        if self.session_multipliers is None:
            self.session_multipliers = {
                "london_ny_overlap": 0.7,   # deepest liquidity -- tighter than the base rate
                "london": 1.0,
                "new_york": 1.0,
                "sydney_tokyo": 1.8,        # thinnest liquidity -- meaningfully worse fills
            }


def _trade_slippage_pct_of_price(trade: Trade, cfg: SessionVolatilitySlippageConfig) -> float:
    session = classify_session(trade.entry_time)
    mult = cfg.session_multipliers.get(session, 1.0)
    if is_weekend_open_gap(trade.entry_time):
        mult *= cfg.weekend_gap_multiplier

    vol_mult = 1.0
    if trade.initial_risk and trade.entry_price:
        risk_pct_of_price = abs(trade.initial_risk) / abs(trade.entry_price)
        # A strategy's own "typical" stop is usually well under 5% of price;
        # scale linearly above that baseline rather than picking an absolute
        # ATR threshold this module has no independent way to calibrate.
        vol_mult = 1.0 + cfg.volatility_sensitivity * max(0.0, risk_pct_of_price - 0.01) * 20.0

    return cfg.base_slippage_pct_of_price * mult * vol_mult


def apply_session_volatility_slippage(trades: list[Trade], cfg: SessionVolatilitySlippageConfig) -> np.ndarray:
    """Returns an array of trade P&Ls, same length/order as `trades`,
    each reduced by that trade's own session- and volatility-scaled
    slippage cost (always a cost, in the direction that hurts -- this
    models WORSE fills than the backtest assumed, never better). A no-op
    (returns trades' pnl unchanged) when cfg.enabled is False."""
    if not cfg.enabled or not trades:
        return np.array([t.pnl for t in trades], dtype=float)

    adjusted = np.empty(len(trades), dtype=float)
    for i, t in enumerate(trades):
        slip_pct = _trade_slippage_pct_of_price(t, cfg)
        cost = abs(t.entry_price) * slip_pct * abs(t.size)
        adjusted[i] = t.pnl - cost
    return adjusted
