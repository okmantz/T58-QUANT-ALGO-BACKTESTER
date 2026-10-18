"""
v9 (2026-10-17) -- eight gamma-exposure and price-structure template families.

They plug into Search Lab and Evolution Lab exactly like v7/v8 (merged into app.search.strategy_space.FAMILIES at the end
of that module) and reuse the same conventions: ATR-multiple stops/targets, dimensionless thresholds only, no hardcoded
pip size, lookahead-checked by tests/strategy/test_families_v9.py.

The gex_* terminals use real gamma levels when data/gamma has them and a clearly-labelled prior-day proxy otherwise
(see app.strategy.indicators_v9), so every family also runs on any CSV you load.

Each family names the existing family it is NOT duplicating:
  gex_wall_fade               vs bollinger/vwap band fades: the band is a gamma wall, not a statistic
  gex_flip_trend              vs ema trend families: direction comes from which side of the gamma flip price is on
  gex_flip_reclaim            vs level_reclaim-style families: the level is the flip, gated by regime
  negative_gamma_breakout     vs trend_breakout: only breaks out when dealers are short gamma (they chase the move)
  positive_gamma_pin_fade     vs vwap_reversion: late-session only, when long-gamma dealers pin price
  gamma_convexity_momentum    vs momentum families: needs price ACCELERATION, not just speed
  ema_stack_bull_flag_volume  vs mtf_pullback: a measured impulse pole + tight flag + volume expansion
  call_wall_breakout          vs trend_breakout: the break must clear the call wall, in negative gamma
"""
from __future__ import annotations

from app.search.strategy_space import SkeletonSpec, _cond, _ind, _risk_management, _val


def _rm(p: dict) -> dict:
    return _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"])


def _g(kind: str, period: int = 14) -> dict:
    return {"type": kind, "period": period, "field": "close"}


_RISK_GRID = {"stop_atr_mult": [1.0, 1.5], "target_atr_mult": [1.5, 2.5], "max_bars": [24, 48]}


def _sides(long_c: list, short_c: list) -> dict:
    return {"long": long_c, "long_connectors": ["AND"] * (len(long_c) - 1),
            "short": short_c, "short_connectors": ["AND"] * (len(short_c) - 1)}


# 1 -- fade the gamma walls in a positive-gamma day
def _build_gex_wall_fade(p: dict) -> dict:
    d = p["wall_atr"]
    return {
        "name": f"GEX Wall Fade (<= {d} ATR from wall)",
        "entry_conditions": _sides(
            [_cond(_g("gex_regime"), ">", _val(0)), _cond(_g("gex_put_wall_distance"), "<", _val(d)),
             _cond(_g("gex_put_wall_distance"), ">", _val(-d)), _cond(_ind("rsi", 14), "<", _val(p["rsi_cap"]))],
            [_cond(_g("gex_regime"), ">", _val(0)), _cond(_g("gex_call_wall_distance"), "<", _val(d)),
             _cond(_g("gex_call_wall_distance"), ">", _val(-d)), _cond(_ind("rsi", 14), ">", _val(100 - p["rsi_cap"]))]),
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _rm(p),
    }


_GEX_WALL_FADE = SkeletonSpec(
    name="gex_wall_fade", label="GEX wall fade (positive gamma)",
    description="In a positive-gamma regime, dealer hedging pushes price back from the largest put/call gamma strikes. "
                "Buys near the put wall and sells near the call wall when RSI agrees.",
    param_grid={"wall_atr": [0.3, 0.6], "rsi_cap": [35, 45], **_RISK_GRID},
    build=_build_gex_wall_fade, valid=lambda p: p["target_atr_mult"] > p["stop_atr_mult"] * 0.9,
)


# 2 -- trend with the side of the flip
def _build_gex_flip_trend(p: dict) -> dict:
    return {
        "name": f"GEX Flip Trend (EMA{p['ema_period']} stack)",
        "entry_conditions": _sides(
            [_cond(_g("gex_flip_distance"), ">", _val(p["flip_atr"])), _cond(_ind("ema_stack_score", p["ema_period"]), ">", _val(0.5))],
            [_cond(_g("gex_regime"), "<", _val(0)), _cond(_g("gex_flip_distance"), "<", _val(-p["flip_atr"])),
             _cond(_ind("ema_stack_score", p["ema_period"]), "<", _val(-0.5))]),
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _rm(p),
    }


_GEX_FLIP_TREND = SkeletonSpec(
    name="gex_flip_trend", label="GEX flip trend",
    description="Longs above the gamma flip with a bullish EMA stack; shorts only below the flip in negative gamma, where "
                "dealers amplify moves.",
    param_grid={"flip_atr": [0.25, 0.75], "ema_period": [9, 13], **_RISK_GRID},
    build=_build_gex_flip_trend, valid=lambda p: True,
)


# 3 -- flip reclaim
def _build_gex_flip_reclaim(p: dict) -> dict:
    return {
        "name": "GEX Flip Reclaim",
        "entry_conditions": _sides(
            [_cond(_g("gex_flip_distance"), "cross above", _val(0)), _cond(_g("volume_expansion", 20), ">", _val(p["vol_x"]))],
            [_cond(_g("gex_flip_distance"), "cross below", _val(0)), _cond(_g("volume_expansion", 20), ">", _val(p["vol_x"]))]),
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _rm(p),
    }


_GEX_FLIP_RECLAIM = SkeletonSpec(
    name="gex_flip_reclaim", label="GEX flip reclaim / loss",
    description="Trades the cross of the gamma-flip level with a volume-expansion confirmation: reclaiming it moves "
                "dealers toward stabilising flow, losing it toward accelerating flow.",
    param_grid={"vol_x": [1.0, 1.3], **_RISK_GRID}, build=_build_gex_flip_reclaim, valid=lambda p: True,
)


# 4 -- breakout only in negative gamma
def _build_negative_gamma_breakout(p: dict) -> dict:
    n = p["lookback"]
    bull = {"type": "bos", "lookback": n, "direction": "bullish"}
    bear = {"type": "bos", "lookback": n, "direction": "bearish"}
    return {
        "name": f"Negative-Gamma Breakout ({n} bars)",
        "entry_conditions": _sides(
            [_cond(_g("gex_regime"), "<", _val(0)), _cond(bull, "is true", _val(1)),
             _cond(_g("volume_expansion", 20), ">", _val(p["vol_x"]))],
            [_cond(_g("gex_regime"), "<", _val(0)), _cond(bear, "is true", _val(1)),
             _cond(_g("volume_expansion", 20), ">", _val(p["vol_x"]))]),
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _rm(p),
    }


_NEGATIVE_GAMMA_BREAKOUT = SkeletonSpec(
    name="negative_gamma_breakout", label="Negative-gamma breakout",
    description="Donchian-style breakouts taken only when dealers are net short gamma (they chase price) and volume expands.",
    param_grid={"lookback": [20, 40], "vol_x": [1.0, 1.4], **_RISK_GRID},
    build=_build_negative_gamma_breakout, valid=lambda p: True,
)


# 5 -- late-session pin fade
def _build_positive_gamma_pin_fade(p: dict) -> dict:
    z = p["z"]
    zv = {"type": "vwap_zscore", "period": 20, "roll_hour": 17, "value_area_sigma": 1.0, "extreme_sigma": 2.0, "poc_buckets": 30}
    return {
        "name": f"Positive-Gamma Pin Fade (z>{z}, late session)",
        "entry_conditions": _sides(
            [_cond(_g("gex_regime"), ">", _val(0)), _cond(_g("theta_clock"), ">", _val(p["clock"])), _cond(zv, "<", _val(-z))],
            [_cond(_g("gex_regime"), ">", _val(0)), _cond(_g("theta_clock"), ">", _val(p["clock"])), _cond(zv, ">", _val(z))]),
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _rm(p),
    }


_POSITIVE_GAMMA_PIN_FADE = SkeletonSpec(
    name="positive_gamma_pin_fade", label="Positive-gamma pin fade (late session)",
    description="Late in the session, long-gamma dealers pin price toward the VWAP; fades stretches beyond z sigma.",
    param_grid={"z": [1.5, 2.0], "clock": [0.55, 0.75], **_RISK_GRID},
    build=_build_positive_gamma_pin_fade, valid=lambda p: True,
)


# 6 -- acceleration momentum
def _build_gamma_convexity_momentum(p: dict) -> dict:
    c = p["convexity"]
    return {
        "name": f"Gamma Convexity Momentum (>{c} ATR)",
        "entry_conditions": _sides(
            [_cond(_g("gamma_convexity", p["period"]), ">", _val(c)), _cond(_ind("ema_stack_score", 9), ">", _val(0))],
            [_cond(_g("gamma_convexity", p["period"]), "<", _val(-c)), _cond(_ind("ema_stack_score", 9), "<", _val(0))]),
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _rm(p),
    }


_GAMMA_CONVEXITY_MOMENTUM = SkeletonSpec(
    name="gamma_convexity_momentum", label="Gamma convexity momentum",
    description="Enters when price is accelerating (positive convexity) in the direction of the EMA stack.",
    param_grid={"convexity": [0.5, 1.0], "period": [8, 14], **_RISK_GRID},
    build=_build_gamma_convexity_momentum, valid=lambda p: True,
)


# 7 -- EMA stack + bull flag + volume
def _build_ema_stack_bull_flag_volume(p: dict) -> dict:
    return {
        "name": "EMA Stack Bull/Bear Flag + Volume Expansion",
        "entry_conditions": _sides(
            [_cond(_ind("ema_stack_score", p["ema_period"]), ">", _val(0.5)), _cond(_g("bull_flag_score", p["flag_bars"]), ">", _val(p["pole_atr"])),
             _cond(_g("volume_expansion", 20), ">", _val(p["vol_x"]))],
            [_cond(_ind("ema_stack_score", p["ema_period"]), "<", _val(-0.5)), _cond(_g("bull_flag_score", p["flag_bars"]), "<", _val(-p["pole_atr"])),
             _cond(_g("volume_expansion", 20), ">", _val(p["vol_x"]))]),
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _rm(p),
    }


_EMA_STACK_BULL_FLAG_VOLUME = SkeletonSpec(
    name="ema_stack_bull_flag_volume", label="EMA stack + flag + volume expansion",
    description="Trend continuation: aligned EMA stack, an impulse pole followed by a tight flag, then a volume expansion.",
    param_grid={"ema_period": [9, 13], "flag_bars": [5, 8], "pole_atr": [2.0, 3.0], "vol_x": [1.1, 1.4], **_RISK_GRID},
    build=_build_ema_stack_bull_flag_volume, valid=lambda p: True,
)


# 8 -- call wall breakout
def _build_call_wall_breakout(p: dict) -> dict:
    return {
        "name": "Call-Wall Breakout (negative gamma)",
        "entry_conditions": _sides(
            [_cond(_g("gex_regime"), "<", _val(0)), _cond(_g("gex_call_wall_distance"), "<", _val(0)),
             _cond(_g("gex_call_wall_distance"), ">", _val(-p["clear_atr"])), _cond(_g("volume_expansion", 20), ">", _val(p["vol_x"]))],
            [_cond(_g("gex_regime"), "<", _val(0)), _cond(_g("gex_put_wall_distance"), "<", _val(0)),
             _cond(_g("gex_put_wall_distance"), ">", _val(-p["clear_atr"])), _cond(_g("volume_expansion", 20), ">", _val(p["vol_x"]))]),
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _rm(p),
    }


_CALL_WALL_BREAKOUT = SkeletonSpec(
    name="call_wall_breakout", label="Call/put wall breakout (negative gamma)",
    description="When dealers are short gamma, a decisive close through the call (or put) wall removes the pinning force "
                "and hedging flow accelerates the move.",
    param_grid={"clear_atr": [0.5, 1.0], "vol_x": [1.0, 1.3], **_RISK_GRID},
    build=_build_call_wall_breakout, valid=lambda p: True,
)


V9_FAMILIES: dict[str, SkeletonSpec] = {s.name: s for s in (
    _GEX_WALL_FADE, _GEX_FLIP_TREND, _GEX_FLIP_RECLAIM, _NEGATIVE_GAMMA_BREAKOUT, _POSITIVE_GAMMA_PIN_FADE,
    _GAMMA_CONVEXITY_MOMENTUM, _EMA_STACK_BULL_FLAG_VOLUME, _CALL_WALL_BREAKOUT)}
V9_FAMILY_NAMES: list[str] = list(V9_FAMILIES)
V9_HYPOTHESIS_QUESTIONS: dict[str, str] = {n: s.description for n, s in V9_FAMILIES.items()}
