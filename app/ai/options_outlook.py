"""Deterministic options-play candidate generator for the Options Outlook
tab/page.

Architecture mirrors app.ai.trading_assistant's own rule ("Ollama
interprets information, your application calculates facts"): every
strike, premium, delta, and breakeven a person sees here is computed by
app.quant_lab.options_pricing's from-scratch Black-Scholes implementation
-- nothing here calls out to an LLM. app.ai.trading_assistant.options_outlook
takes this module's `build_candidates()` output and asks the model to
rank/explain it in prose; this module never talks to Ollama itself.

Strike selection: rather than pricing an arbitrary strike ladder, this
builds candidates around a target-delta grid (roughly ATM, ~25-delta, and
~10-delta on both sides) because that's how options traders actually
think about risk/reward -- "give me the ~30-delta call" is a more useful
question than "price every $5 strike from here to the moon". Strikes are
snapped to the nearest `strike_increment` so the numbers look like real
tradeable strikes (e.g. whole dollars for a $50 stock, $5 increments for
a $400 stock) instead of Black-Scholes' exact-but-unlistable solve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.quant_lab.options_pricing import (
    OptionsPricingError, black_scholes_greeks, black_scholes_price, norm_cdf,
)

# Target deltas (absolute value) used to pick candidate strikes around the
# current spot price. 0.50 ~ ATM, 0.25 ~ a "normal" directional play,
# 0.10 ~ a cheap, lower-probability lottery-style play. Kept small and
# fixed on purpose -- a long strike ladder is more data than a trader
# scanning a daily/weekly outlook actually wants to read.
TARGET_DELTAS = (0.50, 0.25, 0.10)

TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0


class OptionsOutlookError(Exception):
    pass


@dataclass
class OptionCandidate:
    option_type: str        # "call" | "put"
    strike: float
    target_delta: float     # the delta this strike was selected to approximate
    delta: float             # actual Black-Scholes delta at the snapped strike
    premium: float
    breakeven: float
    theta_per_day: float
    gamma: float
    vega_per_vol_point: float
    dte_days: int
    expected_move: float
    prob_itm_proxy: float   # |delta| is the standard quick proxy for prob. of finishing ITM

    def to_dict(self) -> dict:
        return {
            "option_type": self.option_type,
            "strike": round(self.strike, 4),
            "target_delta": self.target_delta,
            "delta": round(self.delta, 4),
            "premium": round(self.premium, 4),
            "breakeven": round(self.breakeven, 4),
            "theta_per_day": round(self.theta_per_day, 4),
            "gamma": round(self.gamma, 6),
            "vega_per_vol_point": round(self.vega_per_vol_point, 4),
            "dte_days": self.dte_days,
            "expected_move": round(self.expected_move, 4),
            "prob_itm_proxy": round(self.prob_itm_proxy, 4),
        }


def expected_move(spot: float, iv: float, dte_days: int) -> float:
    """Standard one-standard-deviation expected move approximation:
    spot * iv * sqrt(T), T in years. This is the same "expected move"
    number options desks quote alongside an expiry -- roughly a 68%
    confidence range, not a prediction of direction."""
    if spot <= 0 or iv <= 0 or dte_days <= 0:
        return 0.0
    t_years = dte_days / CALENDAR_DAYS_PER_YEAR
    return spot * iv * math.sqrt(t_years)


def _delta_solve_strike(
    spot: float, t_years: float, r: float, iv: float, option_type: str, target_delta: float,
    q: float = 0.0, tolerance: float = 1e-4, max_iterations: int = 60,
) -> float:
    """Bisection solve for the strike whose Black-Scholes delta matches
    `target_delta` (given as a positive magnitude for both calls and
    puts). Bisection over a wide, sane strike bracket is used instead of
    a closed-form delta-to-strike inversion because it's a few extra
    lines and is robust regardless of how black_scholes_greeks' internals
    evolve -- correctness over cleverness for a strike picker that only
    runs a handful of times per outlook request."""
    option_type = option_type.lower()
    lo, hi = spot * 0.2, spot * 3.0

    def delta_at(k: float) -> float:
        g = black_scholes_greeks(spot, k, t_years, r, iv, option_type=option_type, q=q)
        return abs(g.delta)

    # |delta(K)| is monotonic in strike but in OPPOSITE directions for
    # calls vs. puts: a call's |delta| falls as strike rises (1 deep ITM
    # at low K -> 0 deep OTM at high K), while a put's |delta| rises as
    # strike rises (0 deep OTM at low K -> 1 deep ITM at high K). The
    # bisection step direction has to match whichever shape is in play,
    # or it converges to the wrong root for puts.
    decreasing_in_strike = option_type == "call"
    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        d = delta_at(mid)
        if abs(d - target_delta) < tolerance:
            return mid
        need_higher_strike = (d > target_delta) if decreasing_in_strike else (d < target_delta)
        if need_higher_strike:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _snap(value: float, increment: float) -> float:
    if increment <= 0:
        return value
    return round(value / increment) * increment


def build_candidates(
    spot: float, iv: float, dte_days: int, r: float = 0.045, q: float = 0.0,
    strike_increment: float | None = None,
) -> list[OptionCandidate]:
    """Builds the call/put candidate grid for one (spot, iv, dte) scenario.

    spot: current underlying price.
    iv: annualized implied/estimated volatility, decimal (e.g. 0.22).
    dte_days: calendar days to the expiry being considered (use ~1-2 for
        "today"/0DTE, ~5-9 for "this week").
    r: annualized risk-free rate, decimal -- defaults to a reasonable
        current short-term rate assumption; pass your own for accuracy.
    q: continuous dividend yield, decimal (0 for most futures/FX/no- or
        low-dividend names).
    strike_increment: rounds candidate strikes to a realistic listed
        increment (e.g. 1 for a $50 stock, 5 for SPX-like products,
        0.0001-scale for FX). Defaults to a rough guess off spot if not
        given.
    """
    if spot <= 0:
        raise OptionsOutlookError("Spot price must be positive.")
    if iv <= 0:
        raise OptionsOutlookError("Implied/estimated volatility must be positive.")
    if dte_days <= 0:
        raise OptionsOutlookError("Days to expiry must be positive.")

    if strike_increment is None:
        if spot < 25:
            strike_increment = 0.5
        elif spot < 200:
            strike_increment = 1.0
        elif spot < 1000:
            strike_increment = 5.0
        else:
            strike_increment = 10.0

    t_years = dte_days / CALENDAR_DAYS_PER_YEAR
    move = expected_move(spot, iv, dte_days)
    candidates: list[OptionCandidate] = []

    for option_type in ("call", "put"):
        for target_delta in TARGET_DELTAS:
            try:
                raw_strike = _delta_solve_strike(spot, t_years, r, iv, option_type, target_delta, q=q)
                strike = max(strike_increment, _snap(raw_strike, strike_increment))
                premium = black_scholes_price(spot, strike, t_years, r, iv, option_type=option_type, q=q)
                greeks = black_scholes_greeks(spot, strike, t_years, r, iv, option_type=option_type, q=q)
            except OptionsPricingError:
                continue

            breakeven = strike + premium if option_type == "call" else strike - premium
            candidates.append(OptionCandidate(
                option_type=option_type,
                strike=strike,
                target_delta=target_delta,
                delta=greeks.delta,
                premium=premium,
                breakeven=breakeven,
                theta_per_day=greeks.theta / CALENDAR_DAYS_PER_YEAR,
                gamma=greeks.gamma,
                vega_per_vol_point=greeks.vega / 100.0,
                dte_days=dte_days,
                expected_move=move,
                prob_itm_proxy=abs(greeks.delta),
            ))

    # Calls ascending by strike (near -> far OTM), puts descending (near ->
    # far OTM) -- reads naturally as "closest/most likely play first".
    calls = sorted([c for c in candidates if c.option_type == "call"], key=lambda c: c.strike)
    puts = sorted([c for c in candidates if c.option_type == "put"], key=lambda c: -c.strike)
    return calls + puts


def candidates_to_dicts(candidates: list[OptionCandidate]) -> list[dict]:
    return [c.to_dict() for c in candidates]


HORIZON_PRESETS: dict[str, int] = {
    "today": 1,
    "this_week": 6,
}
