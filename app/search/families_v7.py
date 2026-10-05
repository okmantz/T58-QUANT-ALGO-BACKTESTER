"""
v7 (2026-10-05, worker B) -- Workstream B: ten new template families for
futures/CFD intraday discovery.

Every family here follows app.search.strategy_space's existing conventions:

* Conditions are built from the same operand dicts ManualStrategy
  dispatches (app.strategy.manual) -- every terminal used below
  (session_high/session_low, vwap_upper_2/vwap_lower_2/vwap_vah/vwap_val/
  session_vwap, rsi_divergence, atr_expansion/atr_contraction, keltner_*,
  bollinger_*, heikin_ashi_*, time_of_day, percentage_change, bos) is
  already registered and verified causal in app.strategy.manual and/or
  app.search.grammar. No new indicator math was needed, so there is no
  new grammar-registration dependency at merge.
* Scale safety: a raw price/ATR-scale indicator is only ever compared to
  ANOTHER price/ATR-scale indicator, never to a hardcoded absolute
  number -- only genuinely dimensionless reads (RSI 0-100, a z-score, the
  tristate divergence flag, a 1/0 regime flag, a percentage_change ratio)
  are ever compared to a plain number.
* Instrument awareness: NO family hardcodes a pip_size or any FX-scale
  assumption. All stops/targets are ATR multiples (resolved against the
  instrument's own volatility at backtest time), and pip_size itself is
  owned by RiskConfig -- auto-detected per dataset by the v7 pip_size UX
  (see app.web.templates._pip_size_autodetect and the run_search
  backstop in app.search.batch_runner). That is what "instrument-aware
  defaults" means for a template family: the family is scale-agnostic by
  construction, so the same grid runs on ES, NQ, GC, or EURUSD without a
  silent FX-pip trap.
* Lookahead: every family is exercised by tests/test_families_v7.py,
  which builds candidates on synthetic ES-scale OHLC and runs
  app.strategy.lookahead_check.check_for_lookahead on each -- a family
  that fails that check fails the suite.
* Registration: V7_FAMILIES is merged into app.search.strategy_space.
  FAMILIES (see the import block at the end of strategy_space.py), so
  Search Lab (generate_search_space) and Evolution Lab (which draws its
  immigrant families from list_families()) both pick these up with no
  further wiring. HYPOTHESIS_QUESTIONS additions live in
  V7_HYPOTHESIS_QUESTIONS in this module and are merged the same way.

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


# ---------------------------------------------------------------------------
# v7-B1: Session Range Breakout (Asia/London/New York)
#   NOT a duplicate of: session_time_effect (opening-range breakout inside
#   one window), prev_day_range_breakout (prior DAY's high/low),
#   session_gated_liquidity_sweep (sweep, not breakout), or
#   session_extreme_fade (fades the session extreme instead of trading
#   through it).
#   Hypothesis: the range built during one session (e.g. Asia) becomes the
#   reference level for the next session's (e.g. London's) directional
#   order flow -- a break of that range during the later window continues.
# ---------------------------------------------------------------------------

def _build_session_range_breakout(p: dict) -> dict:
    rs, re = p["range_session_start"], p["range_session_end"]
    ts, te = p["trade_session_start"], p["trade_session_end"]
    return {
        "name": f"Session Range Breakout ({rs}-{re} range, trade {ts}-{te})",
        "entry_conditions": {
            "long": [
                _cond({"type": "time_of_day", "session_start": ts, "session_end": te}, "is true", _val(1)),
                # session_high is the EXPANDING high of the range session;
                # after that session ends the final value is forward-filled,
                # so during the trade window it is exactly "the range's high".
                _cond(_ind("close", 1), ">",
                      {"type": "session_high", "session_start": rs, "session_end": re}),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "time_of_day", "session_start": ts, "session_end": te}, "is true", _val(1)),
                _cond(_ind("close", 1), "<",
                      {"type": "session_low", "session_start": rs, "session_end": re}),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
        "_time_based_exit": p["flat_time"],
    }


_SESSION_RANGE_BREAKOUT = SkeletonSpec(
    name="session_range_breakout",
    label="Session Range Breakout (Asia/London/New York range break)",
    description=(
        "Builds the reference range from one session (e.g. Asia 18:00-02:00) and trades its "
        "breakout only during a LATER session window (e.g. London 07:00-11:00) -- betting the "
        "overnight range becomes the next session's directional launchpad. Force-flattens by "
        "clock time so a trade opened late in the window can't run into hours the hypothesis "
        "says nothing about. Distinct from session_time_effect (one-window opening-range "
        "breakout) and prev_day_range_breakout (daily, not session, reference)."
    ),
    param_grid={
        "range_session_start": ["18:00", "02:00"],
        "range_session_end": ["02:00", "07:00"],
        "trade_session_start": ["07:00", "08:30"],
        "trade_session_end": ["11:00", "12:00"],
        "flat_time": ["16:00"],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [12, 24],
    },
    build=lambda p: _apply_time_based_exit(_build_session_range_breakout(p)),
    valid=lambda p: (
        (p["range_session_start"], p["range_session_end"])
        != (p["trade_session_start"], p["trade_session_end"])
    ),
)


# ---------------------------------------------------------------------------
# v7-B2: VWAP Band Breakout (+/-2 sigma)
#   NOT a duplicate of: vwap_reversion (fades the stretch AWAY from VWAP),
#   vwap_trend_continuation (buys shallow pullbacks WHILE holding above
#   VWAP -- never requires leaving the band), or vwap_bollinger_pullback
#   (pullback to VWAP inside a Bollinger band).
#   Hypothesis: a close that escapes beyond the second VWAP sigma band is
#   institutional acceptance of a new price level -- momentum continues
#   rather than snapping back.
# ---------------------------------------------------------------------------

def _build_vwap_band_breakout(p: dict) -> dict:
    sigma = p["extreme_sigma"]
    vwap_kw = {"roll_hour": p["roll_hour"], "extreme_sigma": sigma,
               "value_area_sigma": p["value_area_sigma"]}
    return {
        "name": f"VWAP Band Breakout (+/-{sigma} sigma)",
        "entry_conditions": {
            "long": [_cond(_ind("close", 1), "crosses above", {"type": "vwap_upper_2", **vwap_kw})],
            "short": [_cond(_ind("close", 1), "crosses below", {"type": "vwap_lower_2", **vwap_kw})],
        },
        "exit_conditions": {
            # Back inside the extreme band = acceptance failed; the session
            # VWAP itself is the deeper invalidation for the runner.
            "long": [_cond(_ind("close", 1), "crosses below", {"type": "session_vwap", **vwap_kw})],
            "short": [_cond(_ind("close", 1), "crosses above", {"type": "session_vwap", **vwap_kw})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_VWAP_BAND_BREAKOUT = SkeletonSpec(
    name="vwap_band_breakout",
    label="VWAP Band Breakout (beyond +/-2 sigma, momentum continuation)",
    description=(
        "Trades WITH a close that escapes beyond the second VWAP sigma band -- the mirror "
        "image of vwap_reversion: instead of fading the stretch, this bets the escape is "
        "institutional acceptance of a new price level and momentum keeps running. Exits "
        "when price falls back through the session VWAP itself."
    ),
    param_grid={
        "extreme_sigma": [1.5, 2.0],
        "value_area_sigma": [1.0],
        "roll_hour": [17],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [24, 48],
    },
    build=_build_vwap_band_breakout,
)


# ---------------------------------------------------------------------------
# v7-B3: Opening Momentum Burst (time-gated ignition)
#   NOT a duplicate of: pct_change_momentum_burst (not time-gated -- fires
#   on any bar of the day), gap_and_go_continuation (needs an overnight
#   gap, not an intraday burst), or session_time_effect (opening-RANGE
#   breakout, not a momentum-ignition burst).
#   Hypothesis: a sharp percentage burst inside the first minutes of the
#   cash session is informed order flow marking the day's direction, and
#   it keeps running further than a same-sized burst at a random hour.
# ---------------------------------------------------------------------------

def _build_opening_momentum_burst(p: dict) -> dict:
    start, end = p["session_start"], p["session_end"]
    return {
        "name": f"Opening Momentum Burst ({start}-{end})",
        "entry_conditions": {
            "long": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("percentage_change", p["pct_period"]), ">", _val(p["pct_threshold"])),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("percentage_change", p["pct_period"]), "<", _val(-p["pct_threshold"])),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
        "_time_based_exit": p["flat_time"],
    }


_OPENING_MOMENTUM_BURST = SkeletonSpec(
    name="opening_momentum_burst",
    label="Opening Momentum Burst (first-minutes ignition, time-gated)",
    description=(
        "Buys a sharp percentage-change burst ONLY inside the opening window of the cash "
        "session -- betting early informed flow marks the day's direction and runs further "
        "than a same-sized burst at a random hour. Force-flattens by clock time. Distinct "
        "from pct_change_momentum_burst (no time gate) and gap_and_go_continuation (needs a "
        "gap, not an intraday burst)."
    ),
    param_grid={
        "session_start": ["08:30"],
        "session_end": ["09:15", "10:00"],
        "pct_period": [3, 5],
        "pct_threshold": [0.15, 0.25],
        "flat_time": ["16:00"],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0],
        "max_bars": [12, 24],
    },
    build=lambda p: _apply_time_based_exit(_build_opening_momentum_burst(p)),
    valid=lambda p: p["session_start"] < p["session_end"],
)


# ---------------------------------------------------------------------------
# v7-B4: Keltner Band Breakout (no squeeze precondition)
#   NOT a duplicate of: keltner_squeeze_breakout (REQUIRES a Bollinger/
#   Keltner squeeze first -- this one takes every band break),
#   donchian_channel_turtle_breakout (Donchian = pure price extremes, no
#   volatility normalization), or trend_breakout (Donchian-style N-bar
#   break gated on an EMA trend filter).
#   Hypothesis: an ATR-normalized band break (Keltner) is a cleaner
#   continuation trigger than a raw price-extreme break, because the band
#   already accounts for the instrument's own recent volatility.
# ---------------------------------------------------------------------------

def _build_keltner_band_breakout(p: dict) -> dict:
    period = p["period"]
    return {
        "name": f"Keltner Band Breakout (period={period})",
        "entry_conditions": {
            "long": [_cond(_ind("close", 1), "crosses above", _ind("keltner_upper", period))],
            "short": [_cond(_ind("close", 1), "crosses below", _ind("keltner_lower", period))],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), "crosses below", _ind("keltner_mid", period))],
            "short": [_cond(_ind("close", 1), "crosses above", _ind("keltner_mid", period))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_KELTNER_BAND_BREAKOUT = SkeletonSpec(
    name="keltner_band_breakout",
    label="Keltner Band Breakout (ATR-normalized band break, no squeeze needed)",
    description=(
        "Trades every Keltner-channel breakout, with no squeeze precondition -- the band is "
        "already ATR-normalized, so a break is a volatility-adjusted continuation signal on "
        "its own. Exits back at the channel midline. Distinct from keltner_squeeze_breakout "
        "(which waits for compression first) and the Donchian turtle family (raw price "
        "extremes, no volatility normalization)."
    ),
    param_grid={
        "period": [14, 20, 30],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [24, 48],
    },
    build=_build_keltner_band_breakout,
)


# ---------------------------------------------------------------------------
# v7-B5: Bollinger Squeeze Expansion
#   NOT a duplicate of: volatility_contraction_squeeze (ATR-contraction +
#   ATR-range breakout -- this one enters on a BOLLINGER band break),
#   ttm_squeeze_momentum_breakout (TTM composite squeeze + momentum
#   direction), or bollinger_band_walk_continuation (rides the band with
#   NO squeeze requirement).
#   Hypothesis: after volatility compression, the first Bollinger-band
#   break in either direction marks the expansion leg -- trade the break,
#   not the squeeze itself.
# ---------------------------------------------------------------------------

def _build_bollinger_squeeze_expansion(p: dict) -> dict:
    bb_period, atr_period = p["bb_period"], p["atr_period"]
    return {
        "name": f"Bollinger Squeeze Expansion (bb{bb_period}, squeeze-gated)",
        "entry_conditions": {
            "long": [
                # atr_contraction is the dedicated 1/0 leg (see
                # app.search.grammar BOOLEAN_KINDS) -- "is true" is safe here.
                _cond({"type": "atr_contraction", "period": atr_period}, "is true", _val(1)),
                _cond(_ind("close", 1), "crosses above",
                      {"type": "bollinger_upper", "period": bb_period, "field": "close"}),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "atr_contraction", "period": atr_period}, "is true", _val(1)),
                _cond(_ind("close", 1), "crosses below",
                      {"type": "bollinger_lower", "period": bb_period, "field": "close"}),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), "crosses below",
                           {"type": "bollinger_mid", "period": bb_period, "field": "close"})],
            "short": [_cond(_ind("close", 1), "crosses above",
                            {"type": "bollinger_mid", "period": bb_period, "field": "close"})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_BOLLINGER_SQUEEZE_EXPANSION = SkeletonSpec(
    name="bollinger_squeeze_expansion",
    label="Bollinger Squeeze Expansion (compression, then band break)",
    description=(
        "Waits for ATR contraction (the squeeze) and then trades the FIRST Bollinger-band "
        "break in either direction -- the expansion leg, not the compression. Exits back at "
        "the band midline. Distinct from volatility_contraction_squeeze (ATR-range entry), "
        "the TTM squeeze family (composite squeeze + momentum filter), and the band-walk "
        "family (no squeeze requirement at all)."
    ),
    param_grid={
        "bb_period": [14, 20],
        "atr_period": [14],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [24, 48],
    },
    build=_build_bollinger_squeeze_expansion,
)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# v7-B6: RSI Divergence Reversal
#   NOT a duplicate of: rsi_extreme_reversion (level-based fade, no
#   divergence), obv_divergence_trend_confirmation (OBV-based and a
#   CONTINUATION read), connors_rsi_extreme_reversion (level-based), or
#   change_of_character_reversal_scalp (structure-based, no oscillator).
#   Hypothesis: price making a new extreme that RSI does not confirm is
#   exhaustion -- fade it, but only once a break-of-structure confirms the
#   turn, so the entry isn't catching a still-running knife.
# ---------------------------------------------------------------------------

def _build_rsi_divergence_reversal(p: dict) -> dict:
    div_kw = {"rsi_period": p["rsi_period"], "div_lookback": p["div_lookback"]}
    bos_lb = p["bos_lookback"]
    return {
        "name": f"RSI Divergence Reversal (div lb={p['div_lookback']})",
        "entry_conditions": {
            # rsi_divergence is TRISTATE (+1 bullish / -1 bearish / 0):
            # compare with ==, never "is true" (bool(-1) is True).
            "long": [
                _cond({"type": "rsi_divergence", **div_kw}, "==", _val(1)),
                _cond({"type": "bos", "lookback": bos_lb, "direction": "bullish"}, "is true", _val(1)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "rsi_divergence", **div_kw}, "==", _val(-1)),
                _cond({"type": "bos", "lookback": bos_lb, "direction": "bearish"}, "is true", _val(1)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond({"type": "rsi_divergence", **div_kw}, "==", _val(-1))],
            "short": [_cond({"type": "rsi_divergence", **div_kw}, "==", _val(1))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_RSI_DIVERGENCE_REVERSAL = SkeletonSpec(
    name="rsi_divergence_reversal",
    label="RSI Divergence Reversal (exhaustion fade, BOS-confirmed)",
    description=(
        "Fades a classic RSI-vs-price divergence (price new extreme, RSI not confirming), "
        "but only after a break-of-structure confirms the turn -- so the fade isn't catching "
        "a still-running move. Exits on the opposite divergence. Distinct from the "
        "level-based RSI fade families and from the OBV-divergence continuation family."
    ),
    param_grid={
        "rsi_period": [14],
        "div_lookback": [15, 20],
        "bos_lookback": [10, 20],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [24, 48],
    },
    build=_build_rsi_divergence_reversal,
)

# ---------------------------------------------------------------------------
# v7-B7: Choppiness-Gated Momentum (expansion-regime momentum)
#   NOT a duplicate of: momentum_continuation (no regime gate -- fires in
#   chop too), atr_regime_trend_pullback (PULLBACK entry, not a momentum
#   surge entry), or choppiness_regime_trend_pullback (pullback gated on
#   the choppiness INDEX, not the ATR expansion flag).
#   Hypothesis: momentum continuation works when volatility is EXPANDING
#   (a real directional regime) and fails in chop -- so only take the
#   momentum surge when the ATR-expansion flag agrees.
# ---------------------------------------------------------------------------

def _build_choppiness_gated_momentum(p: dict) -> dict:
    rsi_period, rsi_threshold = p["rsi_period"], p["rsi_threshold"]
    return {
        "name": f"Choppiness-Gated Momentum (rsi{rsi_period}>{rsi_threshold}, expansion only)",
        "entry_conditions": {
            "long": [
                _cond(_ind("rsi", rsi_period), ">", _val(rsi_threshold)),
                _cond({"type": "macd_histogram"}, ">", _val(0.0)),
                _cond({"type": "atr_expansion", "period": p["atr_period"]}, "is true", _val(1)),
            ],
            "long_connectors": ["AND", "AND"],
            "short": [
                _cond(_ind("rsi", rsi_period), "<", _val(100 - rsi_threshold)),
                _cond({"type": "macd_histogram"}, "<", _val(0.0)),
                _cond({"type": "atr_expansion", "period": p["atr_period"]}, "is true", _val(1)),
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), "<", _val(50))],
            "short": [_cond(_ind("rsi", rsi_period), ">", _val(50))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_CHOPPINESS_GATED_MOMENTUM = SkeletonSpec(
    name="choppiness_gated_momentum",
    label="Choppiness-Gated Momentum (expansion-regime momentum only)",
    description=(
        "The momentum-continuation core (RSI extremity + MACD histogram agreement), but "
        "taken ONLY while the ATR-expansion flag says volatility is genuinely expanding -- "
        "standing down in chop, where momentum surges are most often fakeouts. Distinct "
        "from the ungated momentum family and from the regime-gated pullback families."
    ),
    param_grid={
        "rsi_period": [7, 14],
        "rsi_threshold": [60, 65],
        "atr_period": [14],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [24],
    },
    build=_build_choppiness_gated_momentum,
)

# ---------------------------------------------------------------------------
# v7-B8: Choppiness-Gated Range Fade (contraction-regime fade)
#   NOT a duplicate of: mean_reversion_band (no regime gate),
#   choppiness_regime_trend_pullback (pullback WITH the trend in a chop
#   regime -- this one fades AGAINST the stretch), or
#   bollinger_squeeze_expansion (B5 above -- trades the band BREAK after
#   compression; this one fades the stretch DURING compression).
#   Hypothesis: when volatility is compressing (ATR contraction), stretches
#   to the Bollinger band are range noise, not breakouts -- fade them back
#   toward the midline, and stand down entirely once expansion resumes.
# ---------------------------------------------------------------------------

def _build_choppiness_gated_range_fade(p: dict) -> dict:
    bb_period, atr_period = p["bb_period"], p["atr_period"]
    return {
        "name": f"Choppiness-Gated Range Fade (bb{bb_period}, contraction only)",
        "entry_conditions": {
            "long": [
                _cond(_ind("close", 1), "<",
                      {"type": "bollinger_lower", "period": bb_period, "field": "close"}),
                _cond({"type": "atr_contraction", "period": atr_period}, "is true", _val(1)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("close", 1), ">",
                      {"type": "bollinger_upper", "period": bb_period, "field": "close"}),
                _cond({"type": "atr_contraction", "period": atr_period}, "is true", _val(1)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), ">",
                           {"type": "bollinger_mid", "period": bb_period, "field": "close"})],
            "short": [_cond(_ind("close", 1), "<",
                            {"type": "bollinger_mid", "period": bb_period, "field": "close"})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_CHOPPINESS_GATED_RANGE_FADE = SkeletonSpec(
    name="choppiness_gated_range_fade",
    label="Choppiness-Gated Range Fade (fade the band stretch in compression)",
    description=(
        "Fades Bollinger-band stretches back toward the midline, but ONLY while the "
        "ATR-contraction flag says volatility is compressing -- in compression, band "
        "touches are range noise; in expansion they are breakouts and this family stands "
        "down. The deliberate mirror of B5 (which trades the break AFTER compression) and "
        "of B7 (which trades momentum DURING expansion)."
    ),
    param_grid={
        "bb_period": [14, 20],
        "atr_period": [14],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.0],
        "max_bars": [12, 24],
    },
    build=_build_choppiness_gated_range_fade,
)

# ---------------------------------------------------------------------------
# v7-B9: Heikin-Ashi Reversal (washout + HA flip)
#   NOT a duplicate of: heikin_ashi_trend_continuation (trades WITH the HA
#   trend on no-wick candles -- this one fades the EXTENDED move on the
#   reversal candle), rsi_extreme_reversion (raw-price RSI fade, no HA
#   structure), or change_of_character_reversal_scalp (raw-price CHoCH).
#   Hypothesis: after an extended smoothed run, the first Heikin-Ashi
#   reversal candle -- confirmed by a washed-out fast RSI -- marks an
#   exhaustion point worth fading for a quick snap-back.
# ---------------------------------------------------------------------------

def _build_heikin_ashi_reversal(p: dict) -> dict:
    rsi_period, rsi_low = p["rsi_period"], p["rsi_low"]
    return {
        "name": f"Heikin-Ashi Reversal (rsi{rsi_period} washout + HA flip)",
        "entry_conditions": {
            "long": [
                _cond(_ind("heikin_ashi_close", 1), "crosses above", _ind("heikin_ashi_open", 1)),
                _cond(_ind("rsi", rsi_period), "<", _val(rsi_low)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("heikin_ashi_close", 1), "crosses below", _ind("heikin_ashi_open", 1)),
                _cond(_ind("rsi", rsi_period), ">", _val(100 - rsi_low)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("heikin_ashi_close", 1), "crosses below", _ind("heikin_ashi_open", 1))],
            "short": [_cond(_ind("heikin_ashi_close", 1), "crosses above", _ind("heikin_ashi_open", 1))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_HEIKIN_ASHI_REVERSAL = SkeletonSpec(
    name="heikin_ashi_reversal",
    label="Heikin-Ashi Reversal (washout + HA flip fade)",
    description=(
        "Fades the extended move at the first Heikin-Ashi reversal candle, confirmed by a "
        "washed-out fast RSI -- the smoothed-candle analogue of an exhaustion fade. The "
        "mirror image of heikin_ashi_trend_continuation: that family rides the HA trend, "
        "this one bets the first HA flip after a washout snaps back."
    ),
    param_grid={
        "rsi_period": [2, 3],
        "rsi_low": [20, 30],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [12, 24],
    },
    build=_build_heikin_ashi_reversal,
    valid=lambda p: p["rsi_low"] < 50,
)

# ---------------------------------------------------------------------------
# v7-B10: VWAP Value-Area Breakout
#   NOT a duplicate of: vwap_band_breakout (B2 above -- the +/-sigma
#   EXTREME band; this is the VALUE AREA edge), vwap_reversion (fades back
#   TO VWAP), or volume_profile_value_area_fade (fades back INTO the value
#   area -- this one trades the EXIT from it as continuation).
#   Hypothesis: leaving the session VWAP value area is institutional
#   acceptance of trade outside the day's fair zone -- the move keeps
#   going rather than snapping back inside.
# ---------------------------------------------------------------------------

def _build_vwap_value_area_breakout(p: dict) -> dict:
    va_kw = {"roll_hour": p["roll_hour"], "value_area_sigma": p["value_area_sigma"]}
    return {
        "name": "VWAP Value-Area Breakout (leave the value area)",
        "entry_conditions": {
            "long": [_cond(_ind("close", 1), "crosses above", {"type": "vwap_vah", **va_kw})],
            "short": [_cond(_ind("close", 1), "crosses below", {"type": "vwap_val", **va_kw})],
        },
        "exit_conditions": {
            # Back inside the value area = acceptance failed.
            "long": [_cond(_ind("close", 1), "crosses below", {"type": "vwap_vah", **va_kw})],
            "short": [_cond(_ind("close", 1), "crosses above", {"type": "vwap_val", **va_kw})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_VWAP_VALUE_AREA_BREAKOUT = SkeletonSpec(
    name="vwap_value_area_breakout",
    label="VWAP Value-Area Breakout (exit the value area, continuation)",
    description=(
        "Trades WITH a close that leaves the session VWAP value area (above VAH / below "
        "VAL) -- betting the exit is institutional acceptance of trade outside the day's "
        "fair zone, so the move continues. The mirror of volume_profile_value_area_fade "
        "(which fades back inside) and the value-area sibling of B2's sigma-band breakout."
    ),
    param_grid={
        "value_area_sigma": [1.0],
        "roll_hour": [17],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [24, 48],
    },
    build=_build_vwap_value_area_breakout,
)


# ---------------------------------------------------------------------------
# Registration -- merged into app.search.strategy_space.FAMILIES (and
# HYPOTHESIS_QUESTIONS) by the import block at the end of strategy_space.py.
# ---------------------------------------------------------------------------

V7_FAMILIES: dict[str, SkeletonSpec] = {
    _SESSION_RANGE_BREAKOUT.name: _SESSION_RANGE_BREAKOUT,
    _VWAP_BAND_BREAKOUT.name: _VWAP_BAND_BREAKOUT,
    _OPENING_MOMENTUM_BURST.name: _OPENING_MOMENTUM_BURST,
    _KELTNER_BAND_BREAKOUT.name: _KELTNER_BAND_BREAKOUT,
    _BOLLINGER_SQUEEZE_EXPANSION.name: _BOLLINGER_SQUEEZE_EXPANSION,
    _RSI_DIVERGENCE_REVERSAL.name: _RSI_DIVERGENCE_REVERSAL,
    _CHOPPINESS_GATED_MOMENTUM.name: _CHOPPINESS_GATED_MOMENTUM,
    _CHOPPINESS_GATED_RANGE_FADE.name: _CHOPPINESS_GATED_RANGE_FADE,
    _HEIKIN_ASHI_REVERSAL.name: _HEIKIN_ASHI_REVERSAL,
    _VWAP_VALUE_AREA_BREAKOUT.name: _VWAP_VALUE_AREA_BREAKOUT,
}

V7_HYPOTHESIS_QUESTIONS: dict[str, str] = {
    "session_range_breakout": "When price breaks the range built during a prior session (e.g. Asia) during a later session window (e.g. London), does it continue?",
    "vwap_band_breakout": "When price escapes beyond the second VWAP sigma band, is that acceptance of a new level (momentum continues)?",
    "opening_momentum_burst": "Does a sharp percentage burst inside the opening minutes keep running further than a same-sized burst at a random hour?",
    "keltner_band_breakout": "Does an ATR-normalized (Keltner) band break continue without needing a prior squeeze?",
    "bollinger_squeeze_expansion": "After volatility compression, does the first Bollinger-band break mark the expansion leg?",
    "rsi_divergence_reversal": "When RSI diverges from a price extreme and structure confirms the turn, does price reverse?",
    "choppiness_gated_momentum": "Does momentum continuation work when volatility is expanding, avoiding chop?",
    "choppiness_gated_range_fade": "In volatility compression, do Bollinger-band stretches fade back to the midline?",
    "heikin_ashi_reversal": "After a washout, does the first Heikin-Ashi reversal candle mark a snap-back?",
    "vwap_value_area_breakout": "When price leaves the session VWAP value area, does it keep going?",
}

V7_FAMILY_NAMES: list[str] = list(V7_FAMILIES.keys())
