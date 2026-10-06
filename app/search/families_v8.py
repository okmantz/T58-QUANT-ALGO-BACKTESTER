"""
v8 (2026-10-05) -- eight new template families for intraday discovery,
each built on a primitive no prior family actually used (four of them on
the new app.strategy.indicators_v8 terminals).

Every family here follows app.search.strategy_space's existing
conventions:

* Conditions are built from the same operand dicts ManualStrategy
  dispatches (app.strategy.manual) -- every terminal used below
  (overnight_gap_atr, session_vwap/vwap_upper_2/vwap_lower_2/vwap_zscore,
  linreg_slope/linreg_r2, relative_volume, liquidity_sweep, candle_range,
  atr, atr_percentile, efficiency_ratio, donchian_mid_distance,
  fractal_strength, time_of_day, previous_day_close, rsi, ema, close) is
  already registered and verified causal in app.strategy.manual and/or
  app.search.grammar. The four v8 indicator kinds used here come from
  app.strategy.indicators_v8 (wired through the v8 fallback in
  build_indicator_series); the rest were already registered.
* Scale safety: a raw price/ATR-scale indicator is only ever compared
  to ANOTHER price/ATR-scale indicator, never to a hardcoded absolute
  number -- only genuinely dimensionless reads (overnight_gap_atr,
  vwap_zscore, donchian_mid_distance, efficiency_ratio, linreg_r2, a
  0-100 percentile, an RSI 0-100) are ever compared to a plain number.
* Instrument awareness: no family hardcodes a pip_size or any FX-scale
  assumption. All stops/targets are ATR multiples resolved against the
  instrument's own volatility at backtest time.
* Lookahead: every family is exercised by tests/test_families_v8.py,
  which builds candidates on synthetic ES-scale OHLC and runs
  app.strategy.lookahead_check.check_for_lookahead on each -- a family
  that fails that check fails the suite.
* Registration: V8_FAMILIES is merged into app.search.strategy_space.
  FAMILIES (see the import block at the end of strategy_space.py), so
  Search Lab (generate_search_space) and Evolution Lab (which draws its
  immigrant families from list_families()) both pick these up with no
  further wiring. HYPOTHESIS_QUESTIONS additions live in
  V8_HYPOTHESIS_QUESTIONS in this module and are merged the same way.

Each family below carries a comment naming the existing family it is
deliberately NOT duplicating, so a reviewer can verify at a glance that
these are genuinely new hypotheses, not renames.
"""

from __future__ import annotations

from app.search.strategy_space import SkeletonSpec, _cond, _ind, _risk_management, _val


def _apply_time_based_exit(config: dict) -> dict:
    flat_time = config.pop("_time_based_exit", None)
    if flat_time:
        config["risk_management"]["time_based_exit"] = {"enabled": True, "time": flat_time}
    return config


def _vwap_kw() -> dict:
    # Operand knobs app.strategy.manual._vwap_profile_series reads (same
    # convention as app.search.families_v7's vwap families and the
    # grammar's own _vwap_builder): session VWAP rolled at 17:00 CT.
    return {"roll_hour": 17, "value_area_sigma": 1.0, "extreme_sigma": 2.0,
            "poc_buckets": 30}


# ---------------------------------------------------------------------------
# v8-1: Overnight Gap Fade, ATR-gated (small gaps only)
#   NOT a duplicate of: overnight_gap_fade (fades EVERY gap regardless of
#   size -- no magnitude filter at all), or gap_and_go_continuation (the
#   opposite direction bet: trades WITH big gaps).
#   Hypothesis: small overnight gaps (a fraction of ATR) are noise that
#   mean-reverts toward the prior close; large gaps are information and
#   should NOT be faded. The new overnight_gap_atr terminal (gap in ATR
#   units -- dimensionless, comparable across ES/NQ/GC/EURUSD) is the
#   gate: only gaps with |gap| <= gap_max_atr_mult are faded.
# ---------------------------------------------------------------------------

def _build_overnight_gap_fade_atr_gated(p: dict) -> dict:
    start, end = p["session_start"], p["session_end"]
    gap_max, gap_min = p["gap_max_atr_mult"], p["gap_min_atr_mult"]
    return {
        "name": f"ATR-Gated Gap Fade ({start}-{end}, |gap|<={gap_max}x ATR)",
        "entry_conditions": {
            "long": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), "<", {"type": "previous_day_close"}),
                _cond({"type": "overnight_gap_atr", "period": p["atr_period"]}, ">", _val(-gap_max)),
                _cond({"type": "overnight_gap_atr", "period": p["atr_period"]}, "<", _val(-gap_min)),
            ],
            "long_connectors": ["AND", "AND", "AND"],
            "short": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), ">", {"type": "previous_day_close"}),
                _cond({"type": "overnight_gap_atr", "period": p["atr_period"]}, "<", _val(gap_max)),
                _cond({"type": "overnight_gap_atr", "period": p["atr_period"]}, ">", _val(gap_min)),
            ],
            "short_connectors": ["AND", "AND", "AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), ">", {"type": "previous_day_close"})],
            "short": [_cond(_ind("close", 1), "<", {"type": "previous_day_close"})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
        "_time_based_exit": p["flat_time"],
    }


_OVERNIGHT_GAP_FADE_ATR_GATED = SkeletonSpec(
    name="overnight_gap_fade_atr_gated",
    label="Overnight Gap Fade, ATR-gated (small gaps only)",
    description=(
        "Fades the overnight gap toward the prior day's close ONLY when the gap is small in "
        "volatility units (|gap| <= gap_max_atr_mult, via the dimensionless overnight_gap_atr "
        "terminal) -- big gaps are information and are deliberately NOT faded (that's "
        "gap_and_go_continuation's hypothesis, the opposite bet). A gap-minimum keeps "
        "dust-sized gaps out. Force-flattened by clock. Distinct from overnight_gap_fade, "
        "which fades every gap regardless of size."
    ),
    param_grid={
        "session_start": ["08:30", "13:30"],
        "session_end": ["09:30", "14:30"],
        "flat_time": ["16:00"],
        "atr_period": [14],
        "gap_max_atr_mult": [0.5, 1.0],
        "gap_min_atr_mult": [0.05],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.0],
        "max_bars": [12, 24],
    },
    build=lambda p: _apply_time_based_exit(_build_overnight_gap_fade_atr_gated(p)),
    valid=lambda p: p["session_start"] < p["session_end"] and p["gap_min_atr_mult"] < p["gap_max_atr_mult"],
)


# ---------------------------------------------------------------------------
# v8-2: NY Lunch Session-VWAP Band Fade
#   NOT a duplicate of: vwap_reversion (daily-VWAP LINE crosses, traded
#   all session -- no band, no time gate), vwap_band_breakout (trades
#   THROUGH the bands -- the opposite direction bet), or
#   vwap_value_area_breakout (also a breakout).
#   Hypothesis: the 11:00-13:00 CT lunch liquidity trough is when price
#   most reliably mean-reverts; fading the session-VWAP +/-2-sigma band
#   back toward the anchored VWAP, only inside that window, with a
#   vwap_zscore distance gate so entries need a genuine stretch, and
#   force-flat before the afternoon session resumes.
# ---------------------------------------------------------------------------

def _build_ny_lunch_vwap_band_fade(p: dict) -> dict:
    start, end = p["lunch_start"], p["lunch_end"]
    kw = _vwap_kw()
    z_min = p["z_min"]
    return {
        "name": f"NY Lunch VWAP Band Fade ({start}-{end}, z>{z_min})",
        "entry_conditions": {
            "long": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), "<", {"type": "vwap_lower_2", **kw}),
                _cond({"type": "vwap_zscore", **kw}, "<", _val(-z_min)),
            ],
            "long_connectors": ["AND", "AND"],
            "short": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), ">", {"type": "vwap_upper_2", **kw}),
                _cond({"type": "vwap_zscore", **kw}, ">", _val(z_min)),
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), "crosses above", {"type": "session_vwap", **kw})],
            "short": [_cond(_ind("close", 1), "crosses below", {"type": "session_vwap", **kw})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
        "_time_based_exit": p["flat_time"],
    }


_NY_LUNCH_VWAP_BAND_FADE = SkeletonSpec(
    name="ny_lunch_vwap_band_fade",
    label="NY Lunch Session-VWAP Band Fade (fade +/-2 sigma back to VWAP)",
    description=(
        "Fades the session-anchored VWAP +/-2-sigma band back toward the VWAP, but ONLY "
        "inside the 11:00-13:00 CT lunch liquidity trough -- the hypothesis is that the "
        "trough is when mean reversion is most reliable. A vwap_zscore distance gate "
        "requires a genuine stretch, and everything is force-flat before the afternoon "
        "session. Distinct from vwap_reversion (line crosses, all session) and from the "
        "VWAP breakout families (opposite direction bet)."
    ),
    param_grid={
        "lunch_start": ["11:00"],
        "lunch_end": ["13:00"],
        "flat_time": ["13:45"],
        "z_min": [1.5, 2.0],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [12, 24],
    },
    build=lambda p: _apply_time_based_exit(_build_ny_lunch_vwap_band_fade(p)),
    valid=lambda p: p["lunch_start"] < p["lunch_end"],
)


# ---------------------------------------------------------------------------
# v8-3: Linear-Regression-Confirmed Trend Continuation
#   NOT a duplicate of: any MA-alignment trend family (wma_ribbon_trend,
#   hma_trend_following, supertrend_trend_following, macd_cross_trend,
#   aroon_trend_strength_breakout, vortex_trend_strength_breakout) -- all
#   of those define "trend" via moving-average geometry. This one defines
#   trend via regression: enter in the direction of linreg_slope only when
#   linreg_r2 says the recent move is actually LINEAR (a clean trend)
#   rather than a jagged lurch with the same slope. Exits on a slope sign
#   flip. The two v8/v7 regression terminals are the entire signal.
# ---------------------------------------------------------------------------

def _build_linreg_confirmed_trend(p: dict) -> dict:
    period, r2_min = p["linreg_period"], p["r2_min"]
    return {
        "name": f"LinReg-Confirmed Trend (period={period}, R2>{r2_min})",
        "entry_conditions": {
            "long": [
                _cond({"type": "linreg_slope", "period": period, "field": "close"}, ">", _val(0.0)),
                _cond({"type": "linreg_r2", "period": period, "field": "close"}, ">", _val(r2_min)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "linreg_slope", "period": period, "field": "close"}, "<", _val(0.0)),
                _cond({"type": "linreg_r2", "period": period, "field": "close"}, ">", _val(r2_min)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond({"type": "linreg_slope", "period": period, "field": "close"}, "<", _val(0.0))],
            "short": [_cond({"type": "linreg_slope", "period": period, "field": "close"}, ">", _val(0.0))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_LINREG_CONFIRMED_TREND = SkeletonSpec(
    name="linreg_confirmed_trend_continuation",
    label="LinReg-Confirmed Trend (slope + R-squared linearity gate)",
    description=(
        "Trades in the direction of the rolling linear-regression slope, but ONLY when the "
        "rolling R-squared says the recent move is genuinely linear (a clean trend) rather "
        "than a jagged lurch that happens to have the same slope. Exits on a slope sign "
        "flip. Defines 'trend' via regression geometry, not moving-average alignment -- "
        "a different mechanism from every MA/cross-based trend family."
    ),
    param_grid={
        "linreg_period": [14, 30],
        "r2_min": [0.5, 0.65],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_linreg_confirmed_trend,
)


# ---------------------------------------------------------------------------
# v8-4: RSI-2 Extreme Reversion, volume-confirmed
#   NOT a duplicate of: rsi_extreme_reversion (RSI 7/14 extremes, NO
#   volume gate) or connors_rsi_extreme_reversion (trades the Connors
#   composite -- price-RSI + streak-RSI + percentile rank -- not raw
#   RSI-2, and has no participation filter).
#   Hypothesis: the classic Connors RSI-2 setup -- fade a 2-period RSI
#   extreme -- but only when relative volume confirms real capitulation
#   participation rather than a thin, meaningless print. Exits when RSI-2
#   mean-reverts through 50.
# ---------------------------------------------------------------------------

def _build_rsi2_volume_confirmed_reversion(p: dict) -> dict:
    rsi_low, rsi_high = p["rsi_low"], 100 - p["rsi_low"]
    return {
        "name": f"RSI2 Volume-Confirmed Reversion (rsi2 {rsi_low}/{rsi_high})",
        "entry_conditions": {
            "long": [
                _cond(_ind("rsi", 2), "<", _val(rsi_low)),
                _cond({"type": "relative_volume", "period": p["vol_period"]}, ">", _val(p["vol_mult"])),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("rsi", 2), ">", _val(rsi_high)),
                _cond({"type": "relative_volume", "period": p["vol_period"]}, ">", _val(p["vol_mult"])),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", 2), "crosses above", _val(50))],
            "short": [_cond(_ind("rsi", 2), "crosses below", _val(50))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_RSI2_VOLUME_CONFIRMED_REVERSION = SkeletonSpec(
    name="rsi2_volume_confirmed_reversion",
    label="RSI-2 Extreme Reversion, volume-confirmed (capitulation fade)",
    description=(
        "Fades a 2-period RSI extreme (the classic Connors RSI-2 setup), but ONLY when "
        "relative volume clears a threshold -- real capitulation participation, not a "
        "thin meaningless print. Exits when RSI-2 mean-reverts through 50. Distinct from "
        "rsi_extreme_reversion (longer RSI, no volume gate) and connors_rsi_extreme_reversion "
        "(Connors composite, no participation filter)."
    ),
    param_grid={
        "rsi_low": [8, 12],
        "vol_period": [10, 20],
        "vol_mult": [1.5, 2.0],
        "stop_atr_mult": [0.75, 1.0, 1.5],
        "target_atr_mult": [1.5, 2.0, 3.0],
        "max_bars": [None, 24],
    },
    build=_build_rsi2_volume_confirmed_reversion,
)


# ---------------------------------------------------------------------------
# v8-5: Turtle Soup -- displacement-filtered failed-breakout fade
#   NOT a duplicate of: liquidity_sweep_reversal (fires on EVERY wick-poke
#   sweep of a prior extreme, however small). The turtle-soup hypothesis
#   is stricter: fade the failed breakout only when the breakout bar
#   showed genuine DISPLACEMENT -- its own range exceeded the recent ATR
#   AND volatility was already running hot (atr_percentile gate) -- i.e.
#   trapped breakout-chasers with real size behind them, not noise.
#   Same "proven base hypothesis plus exactly one new filter" pattern as
#   the expansion-round-2 families.
# ---------------------------------------------------------------------------

def _build_turtle_soup_displacement_fade(p: dict) -> dict:
    lookback, atr_period = p["lookback"], p["atr_period"]
    return {
        "name": f"Turtle Soup Displacement Fade (lb={lookback}, vol pct>{p['vol_pct']})",
        "entry_conditions": {
            # Bearish turtle soup: wick above the prior high, close back
            # below it (the liquidity_sweep bearish primitive), on a bar
            # whose own range exceeded the ATR, while ATR runs hot.
            "short": [
                _cond({"type": "liquidity_sweep", "lookback": lookback, "direction": "bearish"}, "is true", _val(1)),
                _cond({"type": "candle_range"}, ">", {"type": "atr", "period": atr_period}),
                _cond({"type": "atr_percentile", "period": atr_period}, ">", _val(p["vol_pct"])),
            ],
            "short_connectors": ["AND", "AND"],
            "long": [
                _cond({"type": "liquidity_sweep", "lookback": lookback, "direction": "bullish"}, "is true", _val(1)),
                _cond({"type": "candle_range"}, ">", {"type": "atr", "period": atr_period}),
                _cond({"type": "atr_percentile", "period": atr_period}, ">", _val(p["vol_pct"])),
            ],
            "long_connectors": ["AND", "AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_TURTLE_SOUP_DISPLACEMENT_FADE = SkeletonSpec(
    name="turtle_soup_displacement_fade",
    label="Turtle Soup (displacement-filtered failed-breakout fade)",
    description=(
        "Fades a failed N-bar breakout (wick through the prior extreme, close back inside) "
        "ONLY when the breakout bar showed genuine displacement -- its own range exceeded "
        "the recent ATR while ATR itself runs hot (atr_percentile gate): trapped "
        "breakout-chasers with real size behind them, not wick-poke noise. A stricter, "
        "displacement-graded version of the plain liquidity_sweep_reversal hypothesis."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "atr_period": [14],
        "vol_pct": [60, 75],
        "stop_atr_mult": [0.75, 1.0, 1.5],
        "target_atr_mult": [1.5, 2.0, 3.0],
        "max_bars": [None, 24],
    },
    build=_build_turtle_soup_displacement_fade,
)


# ---------------------------------------------------------------------------
# v8-6: Efficiency-Ratio-Gated Trend Pullback
#   NOT a duplicate of: mtf_pullback (RSI dip + EMA, no quality filter),
#   atr_regime_trend_pullback (ATR-regime filter),
#   choppiness_regime_trend_pullback (choppiness-oscillator filter), or
#   volume_confirmed_trend_pullback (volume filter). Kaufman's efficiency
#   ratio is a genuinely different trend-quality read: directional
#   efficiency (net travel vs total wander), not oscillation or
#   volatility level. The hypothesis: pullbacks are only worth buying
#   inside trends that are travelling efficiently; choppy "trends" that
#   score low on ER are just noise and get stood down.
# ---------------------------------------------------------------------------

def _build_efficiency_ratio_trend_pullback(p: dict) -> dict:
    ema_fast, ema_slow = p["ema_fast"], p["ema_slow"]
    rsi_period = p["rsi_period"]
    return {
        "name": f"ER-Gated Pullback (ema {ema_fast}/{ema_slow}, ER>{p['er_min']})",
        "entry_conditions": {
            "long": [
                _cond(_ind("ema", ema_fast), ">", _ind("ema", ema_slow)),
                _cond({"type": "efficiency_ratio", "period": p["er_period"]}, ">", _val(p["er_min"])),
                _cond(_ind("rsi", rsi_period), "<", _val(p["rsi_pullback_low"])),
            ],
            "long_connectors": ["AND", "AND"],
            "short": [
                _cond(_ind("ema", ema_fast), "<", _ind("ema", ema_slow)),
                _cond({"type": "efficiency_ratio", "period": p["er_period"]}, ">", _val(p["er_min"])),
                _cond(_ind("rsi", rsi_period), ">", _val(p["rsi_pullback_high"])),
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), ">", _val(p["rsi_pullback_high"]))],
            "short": [_cond(_ind("rsi", rsi_period), "<", _val(p["rsi_pullback_low"]))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_EFFICIENCY_RATIO_TREND_PULLBACK = SkeletonSpec(
    name="efficiency_ratio_trend_pullback",
    label="Efficiency-Ratio-Gated Trend Pullback (Kaufman ER quality filter)",
    description=(
        "Buys RSI pullbacks inside an EMA trend ONLY when Kaufman's efficiency ratio says "
        "the trend is travelling efficiently (net move vs total wander) -- choppy "
        "'trends' that score low on ER are stood down as noise. A different trend-quality "
        "mechanism from the ATR-regime, choppiness, and volume pullback filters."
    ),
    param_grid={
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "er_period": [14],
        "er_min": [0.4, 0.55],
        "rsi_period": [14],
        "rsi_pullback_low": [35, 40],
        "rsi_pullback_high": [55, 60],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 24],
    },
    build=_build_efficiency_ratio_trend_pullback,
    valid=lambda p: p["ema_fast"] < p["ema_slow"] and p["rsi_pullback_low"] < p["rsi_pullback_high"],
)


# ---------------------------------------------------------------------------
# v8-7: Donchian Channel Position Reversion
#   NOT a duplicate of: mean_reversion_band (Bollinger SIGMA bands --
#   standard-deviation geometry), range_midpoint_fade (fades a CLOCK-TIME
#   range's midpoint), or donchian_channel_turtle_breakout (trades
#   THROUGH the channel -- the opposite bet).
#   Hypothesis: fade extreme Donchian channel position back toward the
#   channel midpoint. The channel here is range-based (highest high /
#   lowest low), not sigma-based, and the entry is the dimensionless
#   donchian_mid_distance terminal (channel position -0.5..+0.5), so the
#   threshold means the same thing on every instrument.
# ---------------------------------------------------------------------------

def _build_donchian_channel_position_reversion(p: dict) -> dict:
    period, pos = p["donchian_period"], p["pos_thresh"]
    return {
        "name": f"Donchian Position Reversion (dc{period}, |pos|>{pos})",
        "entry_conditions": {
            "long": [
                _cond({"type": "donchian_mid_distance", "period": period}, "<", _val(-pos)),
            ],
            "short": [
                _cond({"type": "donchian_mid_distance", "period": period}, ">", _val(pos)),
            ],
        },
        "exit_conditions": {
            "long": [_cond({"type": "donchian_mid_distance", "period": period}, "crosses above", _val(0.0))],
            "short": [_cond({"type": "donchian_mid_distance", "period": period}, "crosses below", _val(0.0))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_DONCHIAN_CHANNEL_POSITION_REVERSION = SkeletonSpec(
    name="donchian_channel_position_reversion",
    label="Donchian Channel Position Reversion (fade channel extremes to mid)",
    description=(
        "Fades extreme Donchian channel position back toward the channel midpoint, entered "
        "on the dimensionless donchian_mid_distance terminal (-0.5 at the lower band, "
        "+0.5 at the upper). The channel is range-based (highest high/lowest low), a "
        "different anchor from Bollinger's sigma bands and from clock-time ranges."
    ),
    param_grid={
        "donchian_period": [20, 40],
        "pos_thresh": [0.3, 0.4],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [None, 48],
    },
    build=_build_donchian_channel_position_reversion,
)


# ---------------------------------------------------------------------------
# v8-8: Fractal Swing Strength Exhaustion Fade
#   NOT a duplicate of: wide_range_bar_exhaustion_fade (ONE bar's own
#   high-low range -- a single-bar read) or volume_climax_reversal (reads
#   climax off relative VOLUME). This one measures multi-bar swing
#   structure: the ATR-normalized size of the most recent CONFIRMED
#   fractal swing envelope (the new fractal_strength terminal). When
#   swings are printing at several ATRs in size and the latest bar closes
#   against the fade direction, the move is read as climactic and faded.
# ---------------------------------------------------------------------------

def _build_fractal_strength_exhaustion_fade(p: dict) -> dict:
    return {
        "name": f"Fractal Strength Exhaustion Fade (strength>{p['strength_thresh']})",
        "entry_conditions": {
            "long": [
                _cond({"type": "fractal_strength", "period": p["atr_period"]}, ">", _val(p["strength_thresh"])),
                _cond({"type": "candle_direction", "direction": "bearish"}, "is true", _val(1)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "fractal_strength", "period": p["atr_period"]}, ">", _val(p["strength_thresh"])),
                _cond({"type": "candle_direction", "direction": "bullish"}, "is true", _val(1)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_FRACTAL_STRENGTH_EXHAUSTION_FADE = SkeletonSpec(
    name="fractal_strength_exhaustion_fade",
    label="Fractal Swing Strength Exhaustion Fade (climactic swings)",
    description=(
        "Fades a move when the ATR-normalized size of the most recent confirmed fractal "
        "swing envelope runs extreme (fractal_strength terminal) and the latest bar closes "
        "against the fade direction -- a multi-bar structural exhaustion read, distinct "
        "from the single-bar range read (wide_range_bar_exhaustion_fade) and the volume "
        "read (volume_climax_reversal)."
    ),
    param_grid={
        "atr_period": [14, 20],
        "strength_thresh": [2.0, 3.0],
        "stop_atr_mult": [0.75, 1.0, 1.5],
        "target_atr_mult": [1.5, 2.0, 3.0],
        "max_bars": [None, 24],
    },
    build=_build_fractal_strength_exhaustion_fade,
)


V8_FAMILIES: dict[str, SkeletonSpec] = {
    _OVERNIGHT_GAP_FADE_ATR_GATED.name: _OVERNIGHT_GAP_FADE_ATR_GATED,
    _NY_LUNCH_VWAP_BAND_FADE.name: _NY_LUNCH_VWAP_BAND_FADE,
    _LINREG_CONFIRMED_TREND.name: _LINREG_CONFIRMED_TREND,
    _RSI2_VOLUME_CONFIRMED_REVERSION.name: _RSI2_VOLUME_CONFIRMED_REVERSION,
    _TURTLE_SOUP_DISPLACEMENT_FADE.name: _TURTLE_SOUP_DISPLACEMENT_FADE,
    _EFFICIENCY_RATIO_TREND_PULLBACK.name: _EFFICIENCY_RATIO_TREND_PULLBACK,
    _DONCHIAN_CHANNEL_POSITION_REVERSION.name: _DONCHIAN_CHANNEL_POSITION_REVERSION,
    _FRACTAL_STRENGTH_EXHAUSTION_FADE.name: _FRACTAL_STRENGTH_EXHAUSTION_FADE,
}

V8_FAMILY_NAMES: list[str] = list(V8_FAMILIES.keys())

V8_HYPOTHESIS_QUESTIONS: dict[str, str] = {
    name: spec.description for name, spec in V8_FAMILIES.items()
}
