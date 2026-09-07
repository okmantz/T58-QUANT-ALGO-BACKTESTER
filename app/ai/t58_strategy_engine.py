"""Deterministic implementation of Owen's exact personal trading model
(see the "OWEN'S EXACT PERSONAL TRADING STRATEGY" doc): macro -> HTF
structure -> 50/200 EMA context -> location -> liquidity -> sweep ->
supply/demand -> premium/discount -> M15 confirmation -> execution ->
opposing-liquidity target.

Deliberately mirrors app.ai.ollama_client's separation of concerns: this
module is Layer 1 + Layer 2 (deterministic facts and rules) from the
proposed architecture -- EMA values, swing highs/lows, liquidity pools,
sweeps, premium/discount, and the resulting T58 status are all plain
Python/pandas, computed the same way every time from the same bars.
Nothing here calls an LLM. app.ai.trading_assistant is Layer 3: it takes
the MarketSnapshot/T58Assessment this module produces and asks Ollama to
explain/summarize it -- the model never invents these facts itself.

Every function here is pure (frame in, dataclass out) so it can be unit
tested without a live data feed, a running Ollama, or a broker
connection -- same testing philosophy as app.strategy.indicators.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.strategy.indicators import atr, ema

# ---------------------------------------------------------------------------
# Tunable constants -- all named here so the scoring rubric is auditable in
# one place rather than scattered as magic numbers through the functions.
# ---------------------------------------------------------------------------
SWING_LOOKBACK = 3          # bars each side for a fractal swing high/low
DEALING_RANGE_LOOKBACK = 50  # bars used to define the current premium/discount range
SWEEP_LOOKBACK_BARS = 20     # how recently a swing point must have formed to count as "live" liquidity
DISPLACEMENT_ATR_MULT = 1.4  # an M15 candle this many multiples of its own ATR counts as "displacement"


@dataclass
class EMAContext:
    ema50: float
    ema200: float
    price: float
    price_vs_50: str   # "above" | "below"
    price_vs_200: str  # "above" | "below"
    alignment: str      # "bullish" | "bearish" | "mixed"


@dataclass
class LiquidityPool:
    price: float
    bar_index: int
    swept: bool


@dataclass
class LiquidityContext:
    bsl: list[LiquidityPool] = field(default_factory=list)  # pools BELOW price (Owen's BSL terminology)
    ssl: list[LiquidityPool] = field(default_factory=list)  # pools ABOVE price (Owen's SSL terminology)

    def nearest_unswept_bsl(self) -> LiquidityPool | None:
        pools = [p for p in self.bsl if not p.swept]
        return max(pools, key=lambda p: p.price) if pools else None

    def nearest_unswept_ssl(self) -> LiquidityPool | None:
        pools = [p for p in self.ssl if not p.swept]
        return min(pools, key=lambda p: p.price) if pools else None


@dataclass
class LocationContext:
    range_high: float
    range_low: float
    midpoint: float
    zone: str  # "premium" | "discount" | "equilibrium"
    extended: bool  # price beyond the dealing range entirely -- Owen's "chasing" flag


@dataclass
class ConfirmationContext:
    displacement: bool
    direction: str  # "bullish" | "bearish" | "none"
    detail: str


@dataclass
class MarketSnapshot:
    """The one structured object handed to the LLM -- see doc 3's
    MarketSnapshot schema. Everything on this object is a fact computed by
    this module; Ollama only ever reasons over it, never regenerates it."""
    symbol: str
    price: float
    macro_bias: str            # "bullish" | "bearish" | "neutral" -- supplied by the caller, see note below
    macro_confidence: float
    macro_drivers: list[str]
    h1_structure: str          # "bullish" | "bearish" | "mixed"
    ema: EMAContext
    location: LocationContext
    liquidity: LiquidityContext
    m15: ConfirmationContext
    news_risk: str = "none"    # "none" | "medium" | "high" -- set by the caller from news_forexfactory output


@dataclass
class T58Assessment:
    symbol: str
    direction: str          # "long" | "short" | "none"
    status: str             # "READY" | "DEVELOPING" | "WAIT" | "EXTENDED" | "PASS"
    score: int               # 0-100, see _score() for the rubric
    checklist: dict[str, bool]
    missing: list[str]
    invalidation: str
    target: str


def _swing_points(frame: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> tuple[list[LiquidityPool], list[LiquidityPool]]:
    """Simple fractal swing detection: a bar is a swing high if its high is
    the max within +/-lookback bars, a swing low if its low is the min.
    These are Owen's "meaningful highs/lows" that create BSL/SSL -- kept
    simple and explainable on purpose rather than a denser structure
    algorithm, since the model must be able to say *which* high/low
    created the liquidity (per the strategy doc's Step 5)."""
    highs, lows = [], []
    h, l = frame["high"].values, frame["low"].values
    n = len(frame)
    for i in range(lookback, n - lookback):
        window_h = h[i - lookback: i + lookback + 1]
        window_l = l[i - lookback: i + lookback + 1]
        if h[i] == window_h.max():
            highs.append(LiquidityPool(price=float(h[i]), bar_index=i, swept=False))
        if l[i] == window_l.min():
            lows.append(LiquidityPool(price=float(l[i]), bar_index=i, swept=False))
    return highs, lows


def _mark_sweeps(pools: list[LiquidityPool], frame: pd.DataFrame, side: str) -> None:
    """Mutates `pools` in place: a pool counts as swept if any LATER bar's
    wick traded through it and then closed back on the origin side (the
    doc's "attack/sweep... price rejects/reclaims" behavior) -- a close
    that stays through the level is treated as a genuine breakout, not a
    sweep, and the pool is left unswept so the model doesn't mistake trend
    continuation for liquidity-grab-and-reverse."""
    close = frame["close"].values
    high = frame["high"].values
    low = frame["low"].values
    n = len(frame)
    for pool in pools:
        for j in range(pool.bar_index + 1, n):
            if side == "low" and low[j] < pool.price and close[j] > pool.price:
                pool.swept = True
                break
            if side == "high" and high[j] > pool.price and close[j] < pool.price:
                pool.swept = True
                break


def _liquidity_context(frame: pd.DataFrame) -> LiquidityContext:
    swing_highs, swing_lows = _swing_points(frame)
    _mark_sweeps(swing_lows, frame, side="low")
    _mark_sweeps(swing_highs, frame, side="high")
    price = float(frame["close"].iloc[-1])
    # BSL = liquidity BELOW lows (below current price); SSL = liquidity ABOVE highs.
    recent_cutoff = len(frame) - SWEEP_LOOKBACK_BARS * 4  # keep a generous window of "still relevant" pools
    bsl = [p for p in swing_lows if p.price < price and p.bar_index >= max(0, recent_cutoff)]
    ssl = [p for p in swing_highs if p.price > price and p.bar_index >= max(0, recent_cutoff)]
    return LiquidityContext(bsl=bsl, ssl=ssl)


def _location_context(frame: pd.DataFrame, lookback: int = DEALING_RANGE_LOOKBACK) -> LocationContext:
    window = frame.tail(lookback)
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    midpoint = (range_high + range_low) / 2.0
    price = float(frame["close"].iloc[-1])
    extended = price > range_high or price < range_low
    if extended:
        zone = "premium" if price > range_high else "discount"
    elif price > midpoint:
        zone = "premium"
    elif price < midpoint:
        zone = "discount"
    else:
        zone = "equilibrium"
    return LocationContext(range_high=range_high, range_low=range_low, midpoint=midpoint, zone=zone, extended=extended)


def _ema_context(frame: pd.DataFrame) -> EMAContext:
    ema50 = ema(frame["close"], 50)
    ema200 = ema(frame["close"], 200)
    price = float(frame["close"].iloc[-1])
    e50, e200 = float(ema50.iloc[-1]), float(ema200.iloc[-1])
    price_vs_50 = "above" if price > e50 else "below"
    price_vs_200 = "above" if price > e200 else "below"
    if e50 > e200 and price_vs_50 == "above":
        alignment = "bullish"
    elif e50 < e200 and price_vs_50 == "below":
        alignment = "bearish"
    else:
        alignment = "mixed"
    return EMAContext(ema50=e50, ema200=e200, price=price, price_vs_50=price_vs_50, price_vs_200=price_vs_200, alignment=alignment)


def _m15_confirmation(m15_frame: pd.DataFrame | None) -> ConfirmationContext:
    """Confirmation, per the doc, means "strong displacement / clear
    rejection / reclaim of structure" AFTER liquidity+location have
    aligned -- never a bare EMA cross or random movement. Proxied here as
    a candle whose true range exceeds DISPLACEMENT_ATR_MULT times its own
    recent ATR, closing strongly in one direction (close in the top/bottom
    third of its range). This is intentionally a narrow, literal reading
    of "displacement" -- it will say "none" more often than a looser
    definition would, which matches the doc's instruction not to
    manufacture confirmation that isn't really there."""
    if m15_frame is None or len(m15_frame) < 20:
        return ConfirmationContext(displacement=False, direction="none", detail="No M15 data supplied.")
    frame = m15_frame.tail(30)
    atr_series = atr(frame, 14)
    last = frame.iloc[-1]
    last_atr = float(atr_series.iloc[-1])
    candle_range = float(last["high"] - last["low"])
    if last_atr <= 0 or candle_range < DISPLACEMENT_ATR_MULT * last_atr:
        return ConfirmationContext(displacement=False, direction="none", detail="No displacement candle on M15.")
    close_position = (last["close"] - last["low"]) / candle_range if candle_range else 0.5
    if close_position >= 0.66:
        return ConfirmationContext(True, "bullish", "M15 displacement candle closed in the top third of its range.")
    if close_position <= 0.33:
        return ConfirmationContext(True, "bearish", "M15 displacement candle closed in the bottom third of its range.")
    return ConfirmationContext(False, "none", "M15 range expanded but closed mid-candle -- not a clean confirmation.")


def build_market_snapshot(
    symbol: str,
    h1_frame: pd.DataFrame,
    m15_frame: pd.DataFrame | None,
    macro_bias: str = "neutral",
    macro_confidence: float = 0.5,
    macro_drivers: list[str] | None = None,
    news_risk: str = "none",
) -> MarketSnapshot:
    """Builds the full deterministic snapshot for one symbol from H1 bars
    (structure/EMA/liquidity/location) plus optional M15 bars
    (confirmation). macro_bias/macro_confidence/macro_drivers are supplied
    by the caller (see app.ai.trading_assistant) rather than computed here
    -- fundamental/macro read is explicitly NOT something this app can
    derive from OHLC bars alone (see that module's docstring for how it's
    actually sourced), and the strategy doc is explicit that a technical
    pattern must never manufacture the macro bias."""
    ema_ctx = _ema_context(h1_frame)
    location = _location_context(h1_frame)
    liquidity = _liquidity_context(h1_frame)
    m15 = _m15_confirmation(m15_frame)
    h1_structure = ema_ctx.alignment  # simple, literal reading: EMA alignment stands in for "H1 structure" label
    return MarketSnapshot(
        symbol=symbol,
        price=float(h1_frame["close"].iloc[-1]),
        macro_bias=macro_bias,
        macro_confidence=macro_confidence,
        macro_drivers=macro_drivers or [],
        h1_structure=h1_structure,
        ema=ema_ctx,
        location=location,
        liquidity=liquidity,
        m15=m15,
        news_risk=news_risk,
    )


def assess(snapshot: MarketSnapshot) -> T58Assessment:
    """Runs Owen's exact hierarchy against the snapshot and returns a
    status. NEVER concludes READY on direction alone -- see the doc's
    "DIRECTION != ENTRY" example, reproduced faithfully here: macro+HTF+EMA
    can all be green while location/liquidity/confirmation are still
    missing, and the result must still be WAIT."""
    direction = "none"
    if snapshot.macro_bias == "bullish":
        direction = "long"
    elif snapshot.macro_bias == "bearish":
        direction = "short"

    checklist: dict[str, bool] = {}
    missing: list[str] = []

    checklist["macro_established"] = snapshot.macro_bias in ("bullish", "bearish")
    if not checklist["macro_established"]:
        missing.append("Macro bias is neutral/mixed -- no directional thesis yet.")

    checklist["ema_supportive"] = (
        (direction == "long" and snapshot.ema.alignment == "bullish")
        or (direction == "short" and snapshot.ema.alignment == "bearish")
    )
    if checklist["macro_established"] and not checklist["ema_supportive"]:
        missing.append("50/200 EMA context does not support the macro thesis yet.")

    checklist["favorable_location"] = (
        (direction == "long" and snapshot.location.zone == "discount")
        or (direction == "short" and snapshot.location.zone == "premium")
    )
    if checklist["macro_established"] and not checklist["favorable_location"]:
        missing.append(f"Price is in {snapshot.location.zone}, not the favorable zone for a {direction or 'directional'} setup.")

    target_pool = None
    sweep_ok = False
    if direction == "long":
        target_pool = snapshot.liquidity.nearest_unswept_ssl()
        source_pool = snapshot.liquidity.nearest_unswept_bsl()
        sweep_ok = source_pool is None  # nearest BSL below already swept (none left unswept nearby)
        if source_pool is not None:
            missing.append(f"BSL at {source_pool.price:.5g} has not been swept yet.")
    elif direction == "short":
        target_pool = snapshot.liquidity.nearest_unswept_bsl()
        source_pool = snapshot.liquidity.nearest_unswept_ssl()
        sweep_ok = source_pool is None
        if source_pool is not None:
            missing.append(f"SSL at {source_pool.price:.5g} has not been swept yet.")
    checklist["liquidity_swept"] = sweep_ok

    checklist["m15_confirmation"] = (
        (direction == "long" and snapshot.m15.direction == "bullish")
        or (direction == "short" and snapshot.m15.direction == "bearish")
    )
    if checklist["macro_established"] and checklist["favorable_location"] and not checklist["m15_confirmation"]:
        missing.append("M15 confirmation (displacement / rejection / reclaim) has not appeared yet.")

    checklist["not_extended"] = not snapshot.location.extended
    if snapshot.location.extended:
        missing.append("Price is extended beyond the recent dealing range -- this would be chasing.")

    checklist["no_high_impact_news_imminent"] = snapshot.news_risk != "high"
    if snapshot.news_risk == "high":
        missing.append("High-impact news is imminent for this symbol.")

    # --- Score: matches the "T58 Opportunity Score" weighting from the
    # architecture note (macro 20 / structure+EMA 15 / location 15 /
    # liquidity 15 / confirmation 15 / event risk -5 / extension -5),
    # renormalized here to the checklist actually computed above.
    score = 0
    score += 20 if checklist["macro_established"] else 0
    score += 15 if checklist["ema_supportive"] else 0
    score += 15 if checklist["favorable_location"] else 0
    score += 15 if checklist["liquidity_swept"] else 0
    score += 15 if checklist["m15_confirmation"] else 0
    score += 10 if not snapshot.location.extended else 0
    score -= 5 if snapshot.news_risk == "high" else 0
    score -= 5 if snapshot.location.extended else 0
    score = max(0, min(100, score))

    # --- Status: literal reading of the doc's setup-quality ladder.
    if snapshot.location.extended and checklist["macro_established"]:
        status = "EXTENDED"
    elif not checklist["macro_established"]:
        status = "PASS"
        direction = "none"
    elif checklist["ema_supportive"] and checklist["favorable_location"] and checklist["liquidity_swept"] and checklist["m15_confirmation"] and checklist["no_high_impact_news_imminent"]:
        status = "READY"
    elif checklist["ema_supportive"] and (checklist["favorable_location"] or checklist["liquidity_swept"]):
        status = "DEVELOPING"
    else:
        status = "WAIT"

    if target_pool is not None:
        target = f"{'SSL' if direction == 'long' else 'BSL'} at {target_pool.price:.5g}"
    else:
        target = "No logical opposing liquidity identified yet."

    if direction == "long":
        invalidation = "Price expands directly into SSL without first sweeping BSL / reaching discount, or macro flips bearish."
    elif direction == "short":
        invalidation = "Price expands directly into BSL without first sweeping SSL / reaching premium, or macro flips bullish."
    else:
        invalidation = "No directional thesis to invalidate -- macro is neutral."

    return T58Assessment(
        symbol=snapshot.symbol, direction=direction, status=status, score=score,
        checklist=checklist, missing=missing, invalidation=invalidation, target=target,
    )
