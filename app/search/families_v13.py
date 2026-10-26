"""v13 (2026-10-25) -- the report's "Top 100 strategy families to add".

Every row of the report is here with its number. Each is a PRE-REGISTERABLE hypothesis (``app.discovery.hypothesis`` via
``preregister_v13_families``): one mechanism, a small fixed grid (grid size = trials booked to the lifetime ledger), a kill
condition written down BEFORE the first run, ATR-multiple risk, and primitives from ``indicators_v13*`` / ``indicators_v11``.

STATUS (be explicit about what a family can and cannot honestly do):

  ready        runs on OHLCV bars alone (``needs`` says if a calendar or extended-hours bars are also needed; without them the
               family's primitives are NaN, the condition is False and it trades nothing)
  needs_data   registered, but trades only when its companion frames (``frame.attrs['basket'|'benchmark'|'copper'|'oil'|...]``) or its
               Ext feed (``app.data.macro_data.EXT_FEEDS`` / ``macro_data.update_ext``) are supplied. Missing data -> zero trades, never a
               fake neutral value
  blocked      NOT registered, because the bar-based engine cannot model it honestly (see ``V13_BLOCKED`` for the reason of each):
               tick/order-book families (56, 57, 59), scale-in/pyramiding (6, 95: the engine has no add-to-position), a calendar-spread
               instrument (18), a published-imbalance feed with no adapter (39), a contract-roll calendar (45)

ROLLOUT (the report: "do not turn on the new batches until the lab-level false-discovery rate is under 5%"): the report's FIRST 12
are batch 1; the rest follow in batches of 10-15 ordered by whether the family can pool across markets. ``new_batches_enabled`` in
``app.discovery_engine.surrogate`` is the gate; until it passes ``active_family_names()`` excludes every v13 family from "all
families" searches (an explicit single-family request still runs them, exactly like RETIRED_FAMILIES). Instrument-tied families skip
the pooled cross-market test and get the stricter holdout plus a mandatory forward test (``V13_INSTRUMENT_TIED``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.search.strategy_space import SkeletonSpec, _cond, _ind, _risk_management, _val

_RG = {"stop_atr_mult": [1.5, 3.0], "target_atr_mult": [3.0, 6.0], "max_bars": [48, 144]}
_GOOD = lambda p: p["target_atr_mult"] >= p["stop_atr_mult"]  # noqa: E731


def _rm(p: dict) -> dict:
    return _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"])


def g(kind: str, period: int = 0) -> dict:
    return {"type": kind, "period": int(period), "field": "close"}


PX = {"type": "close", "period": 1, "field": "close"}


def c(kind: str, period: int, op: str, v: float) -> dict:
    return _cond(g(kind, period), op, _val(v))


def win(a: float, b: float) -> list:
    return [c("minutes_since_open", 0, ">=", a), c("minutes_since_open", 0, "<=", b)]


def late(a: float, b: float) -> list:
    """The last ``a``..``b`` minutes before the close."""
    return [c("minutes_to_close", 0, ">=", b), c("minutes_to_close", 0, "<=", a)]


def after_clock(end_min: float) -> list:
    return [c("minutes_since_open", 0, ">", end_min), c("minutes_since_open", 0, "<", 0)]


def _sides(long_c: list, short_c: list) -> dict:
    return {"long": long_c, "long_connectors": ["AND"] * max(len(long_c) - 1, 0),
            "short": short_c, "short_connectors": ["AND"] * max(len(short_c) - 1, 0)}


def _exits(long_c: list, short_c: list | None = None, mode: str = "OR") -> dict:
    short_c = long_c if short_c is None else short_c
    return {"long": long_c, "long_connectors": [mode] * max(len(long_c) - 1, 0),
            "short": short_c, "short_connectors": [mode] * max(len(short_c) - 1, 0)}


@dataclass(frozen=True)
class FamilyMeta:
    number: int
    name: str
    group: str
    needs: str
    status: str            # ready | needs_data | blocked
    tied: bool             # instrument-tied: no pooled cross-market test; stricter holdout + mandatory forward test
    batch: int             # rollout batch (1 = the report's first 12)
    kill: str              # kill condition, written BEFORE the first run
    hypothesis: str
    aux: str = ""          # what it needs besides OHLCV
    reason: str = ""       # blocked: why


V13_META: dict[str, FamilyMeta] = {}
V13_FAMILIES: dict[str, SkeletonSpec] = {}
V13_BLOCKED: dict[int, FamilyMeta] = {}
RETIRED_V13: dict[str, str] = {}
V13_NUMBER: dict[str, int] = {}

_FIRST_12 = {12, 13, 16, 23, 25, 28, 30, 31, 34, 36, 37, 42}


def _batch_for(num: int, group: str, status: str, tied: bool, needs: str) -> int:
    if num in _FIRST_12:
        return 1
    if status == "blocked":
        return 9
    if num in (56, 57, 59):
        return 9
    if "Ext" in needs or "Basket" in needs or tied:
        return 5 if num < 60 else 6
    return {"Trend and momentum": 2, "Mean reversion": 2, "Breakout and structure": 2, "Session and calendar": 3,
            "Microstructure and order flow": 3, "Volatility regime and premium": 4, "Statistical and model-based": 4,
            "Position management": 4}.get(group, 4)


def fam(num: int, name: str, group: str, label: str, desc: str, kill: str, needs: str, grid: dict,
        long: Callable | None = None, short: Callable | None = None, exits: Callable | None = None, *,
        status: str = "ready", tied: bool = False, aux: str = "", calendar: bool = False, valid: Callable | None = None,
        retired_to: str | None = None) -> None:
    def build(p: dict) -> dict:
        lc = long(p) if long else []
        sc = short(p) if short else []
        out = {"name": f"{label} [{', '.join(f'{k}={v}' for k, v in p.items() if k not in _RG)}]",
               "entry_conditions": _sides(lc, sc), "risk_management": _rm(p)}
        if exits:
            ex = exits(p)
            out["exit_conditions"] = _exits(*ex) if isinstance(ex, tuple) else _exits(ex)
        return out

    meta = FamilyMeta(num, name, group, needs, status, tied, _batch_for(num, group, status, tied, needs), kill, desc, aux)
    V13_META[name] = meta
    V13_NUMBER[name] = num
    ok = valid or _GOOD
    V13_FAMILIES[name] = SkeletonSpec(name=name, label=label, description=desc, param_grid={**grid, **_RG}, build=build,
                                      valid=lambda p, _v=ok: _GOOD(p) and _v(p), requires_calendar_data=calendar)
    if retired_to:
        RETIRED_V13[name] = retired_to


def blocked(num: int, name: str, group: str, label: str, desc: str, kill: str, needs: str, reason: str) -> None:
    V13_BLOCKED[num] = FamilyMeta(num, name, group, needs, "blocked", True, 9, kill, desc, reason=reason, aux=label)
    V13_NUMBER[name] = num


T, M, B, S_, MS, V, X, P = ("Trend and momentum", "Mean reversion", "Breakout and structure", "Session and calendar",
                            "Microstructure and order flow", "Volatility regime and premium", "Cross-asset and macro",
                            "Statistical and model-based")
PM = "Position management"
SLOPE = lambda per: ("kalman_slope_z", per)   # noqa: E731


def _up(p, k="z"):
    return p[k]


# ==================================================================================================== 1. trend and momentum
fam(1, "tsmom_vol_targeted", T, "Vol-targeted multi-horizon time-series momentum",
    "Blended short/medium/long momentum signals scaled by volatility persist (closest existing: ts_daily_trend_hold).",
    "Kill if the blended-momentum edge_lb_r is not above ts_daily_trend_hold's on the same data, or pooled mean R <= 0 across 8+ markets.",
    "OHLCV", {"per": [20, 60], "thr": [0.5, 1.0]},
    lambda p: [c("tsmom_blend", p["per"], ">", p["thr"])], lambda p: [c("tsmom_blend", p["per"], "<", -p["thr"])])
fam(2, "kalman_slope_trend", T, "Kalman-slope trend following",
    "Trends whose filtered slope is large against its own uncertainty continue.",
    "Kill if sealed-slice mean R <= 0 or the null-gate p >= 0.05.", "OHLCV", {"per": [30, 100], "z": [1.5, 2.5]},
    lambda p: [c("kalman_slope_z", p["per"], ">", p["z"])], lambda p: [c("kalman_slope_z", p["per"], "<", -p["z"])])
fam(3, "supersmoother_itrend_cross", T, "Super Smoother / Instantaneous Trendline cross",
    "Low-lag Ehlers filters time trends better than EMA crosses.",
    "Kill if not better than ema_cross_trend's edge_lb_r on the same data (it is a re-skin of an EMA cross).", "OHLCV",
    {"per": [10, 30], "slope": [0.0, 0.3]},
    # both kinds are (filter - close) / ATR: NEGATIVE when price is above the filter
    lambda p: [c("ehlers_supersmoother", p["per"], "<", -p["slope"]), c("ehlers_itrend", p["per"], "<", 0.0)],
    lambda p: [c("ehlers_supersmoother", p["per"], ">", p["slope"]), c("ehlers_itrend", p["per"], ">", 0.0)])
fam(4, "dual_momentum_rotation", T, "Dual-momentum rotation",
    "Assets with positive absolute AND relative momentum keep leading.",
    "Kill if the basket long/short spread has mean R <= 0 net of cost, or fewer than 5 instruments are available.", "Basket",
    {"per": [60, 250], "top": [70, 85]},
    lambda p: [c("xs_momentum_rank", p["per"], ">", p["top"]), c("tsmom_blend", 60, ">", 0.0)],
    lambda p: [c("xs_momentum_rank", p["per"], "<", 100 - p["top"]), c("tsmom_blend", 60, "<", 0.0)], status="needs_data", tied=True,
    aux="frame.attrs['basket'] with >= 4 instruments")
fam(5, "trend_realized_vol_exit", T, "Trend with realized-volatility exit",
    "Exiting when volatility expands AGAINST the position protects open profit.",
    "Kill if the vol-expansion exit does not beat the same entries with a plain ATR stop (paired mean R difference <= 0).", "OHLCV",
    {"per": [50], "vt": [1.5, 2.0]},
    lambda p: [c("kalman_slope_z", p["per"], ">", 1.5)], lambda p: [c("kalman_slope_z", p["per"], "<", -1.5)],
    lambda p: ([c("vol_term_ratio", 20, ">", p["vt"]), c("kalman_slope_z", p["per"], "<", 0.0)],
                [c("vol_term_ratio", 20, ">", p["vt"]), c("kalman_slope_z", p["per"], ">", 0.0)], "AND"))
blocked(6, "breakout_atr_pyramiding", T, "Breakout with ATR-step pyramiding",
        "Adding at fixed ATR steps in a confirmed breakout raises payoff without raising initial risk.",
        "n/a -- not runnable", "OHLCV", "the backtest engine has no scale-in / add-to-position; a faithful test needs partial entries and a "
                                         "combined-risk cap, and modelling it as separate entries would overstate the edge")
fam(7, "slope_tstat_trend", T, "Slope t-statistic gated trend",
    "Only statistically significant slopes continue.",
    "Kill if edge_lb_r is not above the ungated kalman_slope_trend on the same data.", "OHLCV", {"per": [30, 80], "t": [2.0, 3.0]},
    lambda p: [c("slope_tstat", p["per"], ">", p["t"])], lambda p: [c("slope_tstat", p["per"], "<", -p["t"])])
fam(8, "persistence_gated_pullback", T, "Persistence-gated pullback",
    "Pullbacks work only when DFA/Hurst shows persistence.",
    "Kill if the DFA gate adds nothing: pullback mean R with the gate <= without it.", "OHLCV", {"dfa": [0.55, 0.6], "rsi": [30, 40]},
    lambda p: [c("dfa_exponent", 256, ">", p["dfa"]), c("kalman_slope_z", 60, ">", 1.0), c("rsi", 7, "<", p["rsi"])],
    lambda p: [c("dfa_exponent", 256, ">", p["dfa"]), c("kalman_slope_z", 60, "<", -1.0), c("rsi", 7, ">", 100 - p["rsi"])])
fam(9, "mtf_consensus_trend", T, "Multi-timeframe trend consensus",
    "Agreement across the base, 15m, 1h and daily trend states beats any one of them.",
    "Kill if consensus mean R <= the single best timeframe's on the same data.", "OHLCV", {"k": [3, 4], "ema": [20, 50]},
    lambda p: [c("htf_consensus", p["ema"], ">=", p["k"])], lambda p: [c("htf_consensus", p["ema"], "<=", -p["k"])])
fam(10, "cycle_adaptive_breakout", T, "Cycle-adaptive breakout",
    "A breakout lookback that follows the dominant cycle period beats a fixed one.",
    "Kill if not better than trend_breakout with the best fixed lookback (paired).", "OHLCV", {"mult": [8, 15, 25]},
    lambda p: [c("cycle_breakout_state", p["mult"], ">", 0.5)], lambda p: [c("cycle_breakout_state", p["mult"], "<", -0.5)])

# ======================================================================================================== 2. mean reversion
fam(11, "kalman_fair_value_reversion", M, "Reversion to Kalman fair value",
    "Deviations from an adaptive fair value, measured in sigma, mean-revert.",
    "Kill if edge_lb_r <= 0 or not better than rsi_extreme_reversion.", "OHLCV", {"per": [30, 100], "z": [1.5, 2.5]},
    # kalman_level = (fair value - close) / ATR: POSITIVE when price sits below fair value (a long)
    lambda p: [c("kalman_level", p["per"], ">", p["z"])], lambda p: [c("kalman_level", p["per"], "<", -p["z"])],
    lambda p: ([c("kalman_level", p["per"], "<", 0.0)], [c("kalman_level", p["per"], ">", 0.0)]))
fam(12, "halflife_gated_reversion", M, "Half-life-gated reversion",
    "Reversion works only when the estimated OU half-life is shorter than the maximum hold.",
    "Kill if gated mean R <= ungated mean R, or fewer than 100 trades.", "OHLCV", {"z": [1.5, 2.0], "hl": [12, 24]},
    lambda p: [c("kalman_level", 60, ">", p["z"]), c("ou_halflife", 30, "<", p["hl"])],
    lambda p: [c("kalman_level", 60, "<", -p["z"]), c("ou_halflife", 30, "<", p["hl"])],
    lambda p: ([c("kalman_level", 60, "<", 0.0)], [c("kalman_level", 60, ">", 0.0)]))
fam(13, "pctb_variance_ratio_reversion", M, "Percent-b reversion with variance-ratio gate",
    "Band fades work only when the variance ratio shows anti-persistence.",
    "Kill if VR-gated mean R <= ungated, or the gate removes > 90% of trades.", "OHLCV", {"n": [20, 40], "vr": [-1.0, -2.0]},
    lambda p: [_cond(PX, "<", _ind("bollinger_lower", p["n"])), c("vr_z", 10, "<", p["vr"])],
    lambda p: [_cond(PX, ">", _ind("bollinger_upper", p["n"])), c("vr_z", 10, "<", p["vr"])],
    lambda p: ([_cond(PX, ">", _ind("bollinger_mid", p["n"]))], [_cond(PX, "<", _ind("bollinger_mid", p["n"]))]))
fam(14, "prior_close_magnet", M, "Prior-close magnet",
    "Large intraday deviations from the prior close drift back toward it (closest existing: overnight_gap_fade).",
    "Kill if mean R <= 0 or it is not distinct from overnight_gap_fade (trade-overlap > 0.8).", "OHLCV", {"d": [0.6, 1.0]},
    lambda p: win(60, 300) + [c("pdc_dist", 0, "<", -p["d"])], lambda p: win(60, 300) + [c("pdc_dist", 0, ">", p["d"])],
    lambda p: ([c("pdc_dist", 0, ">", 0.0)] + [], [c("pdc_dist", 0, "<", 0.0)]))
fam(15, "anchored_vwap_reversion", M, "Weekly or anchored VWAP reversion",
    "Price returns to higher-timeframe VWAP after stretching.",
    "Kill if mean R <= 0 for both anchors.", "OHLCV", {"anchor": ["avwap_dist_week", "avwap_dist_month"], "z": [1.5, 2.5]},
    lambda p: [c(p["anchor"], 0, "<", -p["z"])], lambda p: [c(p["anchor"], 0, ">", p["z"])],
    lambda p: ([c(p["anchor"], 0, ">", 0.0)], [c(p["anchor"], 0, "<", 0.0)]))
fam(16, "consolidation_box_fade", M, "Consolidation-box edge fade",
    "Edges of a confirmed multi-touch box reject price.",
    "Kill if edge_lb_r <= 0 or fewer than 100 trades.", "OHLCV", {"n": [40, 80], "touches": [3, 4]},
    lambda p: [c("box_touches", p["n"], ">=", p["touches"]), c("box_position", p["n"], "<", 0.1), c("box_position", p["n"], ">", -0.2)],
    lambda p: [c("box_touches", p["n"], ">=", p["touches"]), c("box_position", p["n"], ">", 0.9), c("box_position", p["n"], "<", 1.2)])
fam(17, "cointegrated_basket_reversion", M, "Cointegrated basket reversion",
    "A stationarity-gated basket spread mean-reverts (closest existing: stat_pairs, pairs only). Johansen is not available in this "
    "environment; the spread is the smallest principal component with an AR(1) stationarity gate.",
    "Kill if the gated spread has mean R <= 0, or the AR(1) gate passes < 5% of bars.", "Basket", {"n": [120, 300], "z": [2.0, 2.5]},
    lambda p: [c("basket_pc_phi", p["n"], "<", 0.98), c("basket_pc_spread_z", p["n"], "<", -p["z"])],
    lambda p: [c("basket_pc_phi", p["n"], "<", 0.98), c("basket_pc_spread_z", p["n"], ">", p["z"])],
    lambda p: ([c("basket_pc_spread_z", p["n"], ">", 0.0)], [c("basket_pc_spread_z", p["n"], "<", 0.0)]), status="needs_data", tied=True,
    aux="frame.attrs['basket'] (>= 2 other instruments)")
blocked(18, "calendar_spread_reversion", M, "Calendar-spread reversion", "Front-to-next contract spreads revert to a stable relationship.",
        "n/a -- not runnable", "Two contracts", "needs a front/next-contract spread instrument series; the bundled data is single continuous contracts")
fam(19, "eod_morning_reversal", M, "End-of-day reversal of the morning trend",
    "Strong morning trends partly reverse into the close.",
    "Kill if mean R <= 0 or the sign of the effect flips between halves of the sample.", "OHLCV", {"m": [0.5, 0.8], "late": [90, 150]},
    lambda p: late(p["late"], 20) + [c("morning_move", 120, "<", -p["m"])], lambda p: late(p["late"], 20) + [c("morning_move", 120, ">", p["m"])],
    lambda p: (after_clock(385), after_clock(385)))
fam(20, "round_number_tag_fade", M, "Round-number tag exhaustion fade",
    "A first tag of a major round number exhausts and rejects.",
    "Kill if the first-tag effect is not larger than the effect at random non-round levels (shuffled-level null).", "OHLCV",
    {"fresh": [50, 150]}, lambda p: [c("round_tag", p["fresh"], ">", 0.5)], lambda p: [c("round_tag", p["fresh"], "<", -0.5)], tied=True)
fam(21, "multiday_reversal_index", M, "Multi-day short-term reversal on index futures",
    "Last week's loser bounces over the next few days.",
    "Kill if mean R <= 0 on any of the index markets tested, or the effect needs > 5 days to appear.", "OHLCV",
    {"ema": [5, 10], "d": [1.0, 2.0]},
    lambda p: win(0, 60) + [c("daily_ema_distance", p["ema"], "<", -p["d"])], lambda p: win(0, 60) + [c("daily_ema_distance", p["ema"], ">", p["d"])],
    tied=True)
fam(22, "opening_drive_failure_reversal", M, "Opening-drive failure reversal",
    "A first-30-minute extreme that fails by mid-morning reverses.",
    "Kill if mean R <= 0 or fewer than 100 trades.", "OHLCV", {"m": [15, 30]},
    lambda p: [c("or_failure", p["m"], ">", 0.5)], lambda p: [c("or_failure", p["m"], "<", -0.5)],
    lambda p: (after_clock(p["m"] + 150), after_clock(p["m"] + 150)))

# ============================================================================================== 3. breakout and structure
fam(23, "nr_breakout", B, "NR7 and NR4 narrow-range breakout", "The narrowest range in 7 or 4 bars precedes expansion.",
    "Kill if mean R <= the same breakout on random (non-NR) bars.", "OHLCV", {"n": [4, 7]},
    lambda p: [c("nr_breakout", p["n"], ">", 0.5), c("kalman_slope_z", 50, ">", 0.0)], lambda p: [c("nr_breakout", p["n"], "<", -0.5), c("kalman_slope_z", 50, "<", 0.0)])
fam(24, "inside_bar_trend_break", B, "Inside-bar breakout with trend filter", "An inside bar resolves in the trend direction.",
    "Kill if the trend filter adds nothing: mean R with <= without.", "OHLCV", {"per": [50, 100]},
    lambda p: [c("inside_break", 0, ">", 0.5), c("kalman_slope_z", p["per"], ">", 0.5)], lambda p: [c("inside_break", 0, "<", -0.5), c("kalman_slope_z", p["per"], "<", -0.5)])
fam(25, "weekly_range_breakout", B, "Weekly range breakout", "A break of the prior week's high or low continues.",
    "Kill if mean R <= 0 or the pooled test fails on 8+ markets.", "OHLCV", {"cap": [1.5, 3.0], "off": [0.0, 0.5]},
    lambda p: [c("pwk_high_dist", 0, ">", p["off"]), c("pwk_high_dist", 0, "<", p["cap"] + 1.5)],
    lambda p: [c("pwk_low_dist", 0, "<", -p["off"]), c("pwk_low_dist", 0, ">", -(p["cap"] + 1.5))])
fam(26, "monthly_opening_range_breakout", B, "Monthly opening-range breakout", "The first days of the month set the month's range.",
    "Kill if mean R <= 0 (only ~12 independent months a year: needs 10+ years of data to be testable at all).", "OHLCV", {"days": [3, 5]},
    lambda p: [c("mo_range_break", p["days"], ">", 0.5)], lambda p: [c("mo_range_break", p["days"], "<", -0.5)])
fam(27, "vol_of_vol_compression_release", B, "Volatility-of-volatility compression release",
    "Low vol-of-vol followed by a break signals a regime shift.",
    "Kill if mean R <= the same break without the compression gate.", "OHLCV", {"vov": [0.3, 0.5], "n": [30, 60]},
    lambda p: [c("vol_of_vol", 30, "<", p["vov"]), c("box_position", p["n"], ">", 1.0)], lambda p: [c("vol_of_vol", 30, "<", p["vov"]), c("box_position", p["n"], "<", 0.0)])
fam(28, "breakout_retest_continuation", B, "Breakout, retest, continuation", "A broken level retested from the other side holds and continues.",
    "Kill if mean R <= the plain breakout's on the same data.", "OHLCV", {"n": [30, 60]},
    lambda p: [c("breakout_retest", p["n"], ">", 0.5)], lambda p: [c("breakout_retest", p["n"], "<", -0.5)])
fam(29, "triangle_wedge_breakout", B, "Triangle and wedge breakout", "Convergence of trendlines resolves with momentum (OLS lines, not Theil-Sen).",
    "Kill if mean R <= the breakout of a random box of the same width.", "OHLCV", {"n": [40, 80]},
    lambda p: [c("wedge_break", p["n"], ">", 0.5)], lambda p: [c("wedge_break", p["n"], "<", -0.5)])
fam(30, "low_volume_node_traversal", B, "Low-volume-node traversal", "Price accelerates through thin volume toward the next high-volume node.",
    "Kill if mean R <= 0 when price is in a thin (single-print-heavy) area vs elsewhere.", "OHLCV", {"sp": [0.3, 0.5]},
    lambda p: [c("tpo_single_print_share", 0, ">", p["sp"]), c("kalman_slope_z", 30, ">", 1.0)], lambda p: [c("tpo_single_print_share", 0, ">", p["sp"]), c("kalman_slope_z", 30, "<", -1.0)])
fam(31, "value_area_migration", B, "Value-area migration continuation",
    "A session accepted above (below) the prior session's value area continues higher (lower).",
    "Kill if mean R <= 0 or acceptance above the prior value area is no better than a random price level.", "OHLCV", {"win": [60, 120]},
    lambda p: win(p["win"], p["win"] + 120) + [c("tpo_va_position", 0, ">", 1.0), c("daily_trend_state", 0, ">", -0.5)],
    lambda p: win(p["win"], p["win"] + 120) + [c("tpo_va_position", 0, "<", 0.0), c("daily_trend_state", 0, "<", 0.5)],
    lambda p: (after_clock(p["win"] + 120), after_clock(p["win"] + 120)))
fam(32, "gap_and_hold_acceptance", B, "Gap-and-hold acceptance", "A gap that holds through the first hour continues.",
    "Kill if mean R <= 0 or it duplicates overnight_gap_fade with the sign flipped (paired).", "OHLCV", {"g": [0.3, 0.6], "fill": [0.25, 0.5]},
    lambda p: win(60, 150) + [c("pdc_dist", 0, ">", p["g"]), c("gap_fill_progress", 0, "<", p["fill"])],
    lambda p: win(60, 150) + [c("pdc_dist", 0, "<", -p["g"]), c("gap_fill_progress", 0, "<", p["fill"])], lambda p: (after_clock(300), after_clock(300)))
fam(33, "nested_squeeze_release", B, "Nested multi-timeframe squeeze", "Compression on several timeframes at once releases harder.",
    "Kill if mean R <= the single-scale squeeze release.", "OHLCV", {"k": [2, 3], "n": [20, 30]},
    lambda p: [c("nested_squeeze", p["n"], ">=", p["k"]), c("box_position", p["n"], ">", 1.0)], lambda p: [c("nested_squeeze", p["n"], ">=", p["k"]), c("box_position", p["n"], "<", 0.0)])
fam(34, "power_of_three_daily", B, "Power of Three daily model", "Accumulation, a manipulation sweep of the prior day's extreme, then distribution through the day's range.",
    "Kill if mean R <= 0 or the sweep adds nothing over trading from the open in the same direction.", "OHLCV", {"win": [60, 180]},
    lambda p: win(p["win"], 330) + [c("po3_state", 0, ">", 0.5)], lambda p: win(p["win"], 330) + [c("po3_state", 0, "<", -0.5)], lambda p: (after_clock(385), after_clock(385)))

# ================================================================================================== 4. session and calendar
fam(35, "london_ny_handoff", S_, "London-New York overlap handoff", "The London move into the overlap predicts New York's first hour.",
    "Kill if mean R <= 0 or needs 24h bars that the data lacks (then it trades nothing).", "OHLCV (24h bars)", {"strength": [0.5, 1.0]},
    lambda p: win(0, 60) + [c("handoff_pos_london", 0, ">", 1.0), c("handoff_ny_london", 0, ">", p["strength"])],
    lambda p: win(0, 60) + [c("handoff_pos_london", 0, "<", 0.0), c("handoff_ny_london", 0, ">", p["strength"])], lambda p: (after_clock(75), after_clock(75)), tied=True,
    aux="24-hour bars")
fam(36, "ny_open_drive_continue_or_fade", S_, "New York open drive: continue or fade", "The first-15-minute range size decides continuation versus fade.",
    "Kill if neither the continue nor the fade variant has mean R > 0, or the two have the same sign of effect for both range sizes.", "OHLCV",
    {"w": [0.3, 0.6], "mode": ["continue", "fade"]},
    lambda p: win(15, 90) + [c("or_state", 15, ">" if p["mode"] == "continue" else "<", 0.5 if p["mode"] == "continue" else -0.5),
                              c("or_width_atr", 15, ">" if p["mode"] == "continue" else "<", p["w"])],
    lambda p: win(15, 90) + [c("or_state", 15, "<" if p["mode"] == "continue" else ">", -0.5 if p["mode"] == "continue" else 0.5),
                              c("or_width_atr", 15, ">" if p["mode"] == "continue" else "<", p["w"])],
    lambda p: (after_clock(120), after_clock(120)))
fam(37, "midday_lull_expansion", S_, "Midday lull to afternoon expansion", "A compressed midday range precedes a 2 pm expansion.",
    "Kill if mean R <= the same break without the compression condition.", "OHLCV", {"tight": [4.0, 8.0], "n": [24, 36]},
    lambda p: win(270, 360) + [c("box_tightness", p["n"], "<", p["tight"]), c("box_position", p["n"], ">", 1.0)],
    lambda p: win(270, 360) + [c("box_tightness", p["n"], "<", p["tight"]), c("box_position", p["n"], "<", 0.0)], lambda p: (after_clock(385), after_clock(385)))
fam(38, "power_hour_continuation", S_, "Power-hour continuation", "The afternoon trend persists into the last hour.",
    "Kill if mean R <= 0 or equals the effect of holding the same side all day.", "OHLCV", {"d": [0.4, 0.8]},
    lambda p: late(60, 15) + [c("day_open_dist", 0, ">", p["d"]), c("kalman_slope_z", 30, ">", 0.0)],
    lambda p: late(60, 15) + [c("day_open_dist", 0, "<", -p["d"]), c("kalman_slope_z", 30, "<", 0.0)], lambda p: (after_clock(385), after_clock(385)))
blocked(39, "closing_imbalance_reversion", S_, "Closing-imbalance reversion", "Large published close imbalances revert after the close.",
        "n/a -- not runnable", "Ext", "needs the exchange's published closing-imbalance feed, and there is no adapter; add one in macro_data.EXT_FEEDS first")
fam(40, "monday_gap_reversal", S_, "Monday gap and reversal model", "Weekend gaps with no news partly fill on Monday.",
    "Kill if mean R <= 0 or Monday gaps fill no more often than weekday gaps.", "OHLCV", {"g": [0.3, 0.6]},
    lambda p: win(0, 120) + [c("weekday_number", 0, "<", 0.5), c("pdc_dist", 0, "<", -p["g"]), c("event_day_any", 0, "<", 0.5)],
    lambda p: win(0, 120) + [c("weekday_number", 0, "<", 0.5), c("pdc_dist", 0, ">", p["g"]), c("event_day_any", 0, "<", 0.5)],
    lambda p: ([c("pdc_dist", 0, ">", 0.0)], [c("pdc_dist", 0, "<", 0.0)]))
fam(41, "friday_derisk_drift", S_, "Friday de-risk drift", "Positions are cut into the weekend, creating a late-Friday drift (direction is itself the test: both sides are booked).",
    "Kill if neither side has mean R > 0 with p < 0.05 after the Bonferroni charge for both sides.", "OHLCV", {"side": ["long", "short"], "late": [120, 240]},
    lambda p: (late(p["late"], 10) + [c("weekday_number", 0, ">", 3.5)]) if p["side"] == "long" else [],
    lambda p: (late(p["late"], 10) + [c("weekday_number", 0, ">", 3.5)]) if p["side"] == "short" else [], lambda p: (after_clock(385), after_clock(385)))
fam(42, "pre_fomc_drift", S_, "Pre-FOMC announcement drift", "Equity index futures drift upward before announcements.",
    "Kill if mean R <= 0 over 20+ events or the effect disappears after 2015 (published decay).", "OHLCV plus calendar", {"h": [180, 360]},
    lambda p: [c("minutes_to_fomc", 0, ">", 0.0), c("minutes_to_fomc", 0, "<=", p["h"]), c("event_day_fomc", 0, ">", 0.5)], None,
    calendar=True, tied=True, aux="economic calendar (FOMC dates)")
fam(43, "month_end_rebalancing_reversal", S_, "Month-end and quarter-end rebalancing reversal", "Rebalancing flows reverse the prior month's move.",
    "Kill if mean R <= 0 across months (few independent observations: needs 15+ years).", "OHLCV", {"d": [0.5, 1.0]},
    lambda p: [c("month_end_window", 0, ">", 0.5), c("mo_open_dist", 0, "<", -p["d"])], lambda p: [c("month_end_window", 0, ">", 0.5), c("mo_open_dist", 0, ">", p["d"])])
fam(44, "roll_week_vol_fade", S_, "Roll-week and triple-witching volatility model", "Roll and expiry weeks have a distinct volatility pattern: spikes there fade.",
    "Kill if mean R <= the same fade in ordinary weeks.", "OHLCV", {"rv": [2.0, 3.0]},
    lambda p: [c("triple_witching", 0, ">", 0.5), c("rvol_tod", 20, ">", p["rv"]), c("roc", 3, "<", 0.0)],
    lambda p: [c("triple_witching", 0, ">", 0.5), c("rvol_tod", 20, ">", p["rv"]), c("roc", 3, ">", 0.0)])
blocked(45, "futures_roll_day_dislocation", S_, "Futures roll-day dislocation fade", "Roll-day price jumps unrelated to news revert.",
        "n/a -- not runnable", "OHLCV", "needs a per-contract roll calendar; the bundled series are continuous and already adjusted")
fam(46, "thin_session_fade", S_, "Half-day and thin-liquidity session fade", "Moves in thin sessions revert when liquidity returns (operationalised as the session before an exchange holiday).",
    "Kill if mean R <= 0 or fewer than 100 trades (holiday sessions are rare).", "OHLCV", {"d": [0.4, 0.8]},
    lambda p: [c("pre_holiday", 0, ">", 0.5), c("pdc_dist", 0, "<", -p["d"])], lambda p: [c("pre_holiday", 0, ">", 0.5), c("pdc_dist", 0, ">", p["d"])])
fam(47, "seasonal_tilt_dom_woy", S_, "Day-of-month and week-of-year tilt", "A causally estimated seasonal tilt adds edge to another signal (closest existing: day_of_week_seasonality).",
    "Kill if the tilted signal's mean R <= the untilted signal's (paired).", "OHLCV", {"season": ["seas_dom", "seas_woy", "seas_tod"], "s": [0.5, 1.0]},
    lambda p: [c("kalman_slope_z", 50, ">", 1.0), c(p["season"], 0, ">", p["s"])], lambda p: [c("kalman_slope_z", 50, "<", -1.0), c(p["season"], 0, "<", -p["s"])])
fam(48, "seasonal_tilt_hour_of_day", S_, "Hour-of-day return tilt", "A causally estimated hourly drift adds edge to another signal.",
    "Merged into seasonal_tilt_dom_woy (same construction, seas_tod): not run on its own.", "OHLCV", {"s": [0.5, 1.0]},
    lambda p: [c("kalman_slope_z", 50, ">", 1.0), c("seas_tod", 0, ">", p["s"])], lambda p: [c("kalman_slope_z", 50, "<", -1.0), c("seas_tod", 0, "<", -p["s"])],
    retired_to="seasonal_tilt_dom_woy")

# ============================================================================================ 5. microstructure and order flow
fam(49, "order_flow_toxicity_spike", MS, "Order-flow toxicity spike", "A VPIN or imbalance spike precedes short-horizon continuation or reversal, depending on regime (mode is booked).",
    "Kill if neither mode has mean R > 0.", "OHLCV", {"vp": [0.5, 0.56], "mode": ["continue", "fade"]},
    lambda p: [c("vpin", 50, ">", p["vp"]), c("bvc_imbalance", 20, ">" if p["mode"] == "continue" else "<", 0.3 if p["mode"] == "continue" else -0.3)],
    lambda p: [c("vpin", 50, ">", p["vp"]), c("bvc_imbalance", 20, "<" if p["mode"] == "continue" else ">", -0.3 if p["mode"] == "continue" else 0.3)])
fam(50, "delta_divergence_extremes", MS, "Delta divergence at session extremes", "Price making a new extreme without matching cumulative delta fails.",
    "Kill if mean R <= 0 (BVC imbalance is a bar-based estimate, not true delta).", "OHLCV", {"n": [50, 100], "imb": [0.0, 0.1]},
    lambda p: [c("box_position", p["n"], "<", 0.02), c("bvc_imbalance", p["n"], ">", p["imb"])], lambda p: [c("box_position", p["n"], ">", 0.98), c("bvc_imbalance", p["n"], "<", -p["imb"])])
fam(51, "naked_poc_magnet", MS, "Naked POC magnet", "Untested prior points of control pull price back.",
    "Kill if the touch rate of naked POCs is no higher than of random price levels.", "OHLCV", {"d": [0.5, 1.0]},
    lambda p: [c("naked_poc_count", 0, ">=", 1.0), c("naked_poc_dist", 0, "<", -p["d"])], lambda p: [c("naked_poc_count", 0, ">=", 1.0), c("naked_poc_dist", 0, ">", p["d"])],
    lambda p: ([c("naked_poc_dist", 0, ">", -0.2)], [c("naked_poc_dist", 0, "<", 0.2)]))
fam(52, "poor_high_low_repair", MS, "Poor high and low repair", "Unfinished auctions at session extremes get revisited.",
    "Kill if mean R <= 0 or poor extremes are revisited no more often than clean ones.", "OHLCV", {"win": [30, 120]},
    lambda p: win(0, p["win"]) + [c("tpo_poor_high", 0, ">", 0.5)], lambda p: win(0, p["win"]) + [c("tpo_poor_low", 0, ">", 0.5)], lambda p: (after_clock(p["win"] + 90), after_clock(p["win"] + 90)))
fam(53, "equal_highs_stop_run", MS, "Equal-highs stop run with volume", "A sweep of clustered equal highs or lows on a volume spike reverses (closest existing: liquidity_sweep_reversal).",
    "Kill if not distinct from liquidity_sweep_reversal (Jaccard of conditions >= 0.8 => retire) or mean R <= 0.", "OHLCV", {"dens": [2.0, 3.0], "rv": [1.2, 1.6]},
    lambda p: [c("equal_lows_density", 100, ">", p["dens"]), c("rvol_tod", 20, ">", p["rv"]), c("lower_wick_ratio", 0, ">", 0.4)],
    lambda p: [c("equal_highs_density", 100, ">", p["dens"]), c("rvol_tod", 20, ">", p["rv"]), c("upper_wick_ratio", 0, ">", 0.4)])
fam(54, "liquidity_vacuum_fade", MS, "Liquidity-vacuum fade", "A spike in estimated spread reverts as liquidity returns.",
    "Kill if mean R <= 0 or the spread z-score is just a volatility proxy (partial effect after controlling for ATR is zero).", "OHLCV", {"z": [1.5, 2.5]},
    lambda p: [c("spread_z", 20, ">", p["z"]), c("roc", 3, "<", 0.0)], lambda p: [c("spread_z", 20, ">", p["z"]), c("roc", 3, ">", 0.0)])
fam(55, "illiquidity_premium_drift", MS, "Illiquidity-premium drift", "Instruments with rising Amihud illiquidity earn a multi-day premium (single-instrument form: long after an illiquidity z-score spike).",
    "Kill if mean R <= 0; the cross-sectional claim itself needs a basket and is not tested here.", "OHLCV", {"z": [1.0, 2.0]},
    lambda p: [c("amihud_z", 20, ">", p["z"])], None, tied=True)
blocked(56, "large_trade_imbalance_continuation", MS, "Large-trade imbalance continuation", "Persistent large-lot imbalance continues for a few bars.",
        "n/a -- not runnable", "Tick", "needs trade-level data; bar volume cannot identify lot sizes. Tick families depend on latency and queue position that a bar-based prop simulation cannot model honestly")
blocked(57, "dollar_bar_momentum", MS, "Dollar-bar momentum", "Momentum measured in volume time is cleaner than in clock time.",
        "n/a -- not runnable", "Tick", "needs tick data to build dollar bars (volume_clock_return is the bar-based approximation, used by other families)")
fam(58, "premarket_level_acceptance", MS, "Pre-market level acceptance or rejection", "Pre-market highs and lows act as same-day support and resistance (acceptance = holds beyond the overnight extreme).",
    "Kill if mean R <= 0 for both acceptance and rejection.", "OHLCV with extended hours", {"mode": ["accept", "reject"], "win": [60, 120]},
    lambda p: win(30, p["win"] + 30) + [c("overnight_position", 0, ">" if p["mode"] == "accept" else "<", 1.0 if p["mode"] == "accept" else 0.15)],
    lambda p: win(30, p["win"] + 30) + [c("overnight_position", 0, "<" if p["mode"] == "accept" else ">", 0.0 if p["mode"] == "accept" else 0.85)],
    lambda p: (after_clock(p["win"] + 90), after_clock(p["win"] + 90)), aux="extended-hours bars")
blocked(59, "order_book_imbalance_scalp", MS, "Order-book imbalance micro-scalp", "Top-of-book imbalance predicts the next tick direction.",
        "n/a -- not runnable", "Tick, order book", "needs level-2 data and a queue-position model; a bar simulation would invent fills")

# ========================================================================================= 6. volatility regime and premium
fam(60, "variance_risk_premium_timing", V, "Variance-risk-premium timing", "A high gap of implied over realized volatility favors long equity exposure.",
    "Kill if mean R <= 0 or the VRP gate adds nothing over a plain trend filter.", "Ext", {"v": [2.0, 5.0], "per": [20]},
    lambda p: [c("vrp", p["per"], ">", p["v"])], None, status="needs_data", tied=True, aux="Ext feed: VIX (macro cache)")
fam(61, "vix_term_throttled_trend", V, "VIX term-structure-throttled trend", "Trend exposure should shrink when the VIX curve inverts.",
    "Kill if the throttled trend's mean R <= the unthrottled trend's (paired).", "VIX feed", {"per": [50, 100], "z": [1.0, 1.5]},
    lambda p: [c("kalman_slope_z", p["per"], ">", p["z"]), c("vix_term_slope", 0, "<", 0.0)], lambda p: [c("kalman_slope_z", p["per"], "<", -p["z"]), c("vix_term_slope", 0, "<", 0.0)],
    status="needs_data", tied=True, aux="macro VIX + VIX3M")
fam(62, "vol_managed_exposure", V, "Vol-managed index exposure", "Scaling long exposure inversely to volatility improves risk-adjusted return (single-instrument form: be long only when volatility is in its low range).",
    "Kill if the Sharpe is not above buy-and-hold's on the same data.", "OHLCV", {"q": [30, 50]},
    lambda p: [c("vol_cone_pct", 60, "<", p["q"]), c("kalman_slope_z", 100, ">", 0.0)], None,
    lambda p: ([c("vol_cone_pct", 60, ">", p["q"] + 30)], []), tied=True)
fam(63, "vvix_spike_breakout", V, "VVIX spike breakout", "A VVIX spike precedes a volatility regime change in index futures.",
    "Kill if mean R <= 0 on 3+ index markets.", "Ext", {"q": [90, 95]},
    lambda p: [c("vvix_pct", 250, ">", p["q"]), c("box_position", 40, ">", 1.0)], lambda p: [c("vvix_pct", 250, ">", p["q"]), c("box_position", 40, "<", 0.0)],
    status="needs_data", tied=True, aux="Ext feed: vvix")
fam(64, "skew_extreme_reversal", V, "Skew-extreme reversal", "Extreme CBOE SKEW readings precede mean reversion of the recent move.",
    "Kill if mean R <= 0 or fewer than 100 trades (SKEW extremes are rare).", "Ext", {"q": [90, 95]},
    lambda p: [c("skew_pct", 250, ">", p["q"]), c("kalman_level", 60, ">", 1.5)], lambda p: [c("skew_pct", 250, ">", p["q"]), c("kalman_level", 60, "<", -1.5)],
    status="needs_data", tied=True, aux="Ext feed: skew")
fam(65, "post_event_vol_crush", V, "Post-event volatility crush", "Realized volatility collapses after scheduled events, favoring fades.",
    "Kill if mean R <= 0 or equals the same fade at non-event times.", "OHLCV plus calendar", {"start": [90, 150], "rsi": [20, 30]},
    lambda p: [c("minutes_since_fomc", 0, ">=", p["start"]), c("minutes_since_fomc", 0, "<=", p["start"] + 240), c("rsi", 7, "<", p["rsi"])],
    lambda p: [c("minutes_since_fomc", 0, ">=", p["start"]), c("minutes_since_fomc", 0, "<=", p["start"] + 240), c("rsi", 7, ">", 100 - p["rsi"])], calendar=True, tied=True, aux="economic calendar")
fam(66, "intraday_vol_exhaustion_fade", V, "Intraday volatility-exhaustion fade", "A range expansion that has used most of the typical daily range fades.",
    "Kill if mean R <= 0 or the effect is explained by plain overextension (pdc_dist) alone.", "OHLCV", {"used": [0.9, 1.2], "edge": [0.1, 0.2]},
    lambda p: win(120, 360) + [c("day_range_used", 0, ">", p["used"]), c("day_range_pos", 0, "<", p["edge"])],
    lambda p: win(120, 360) + [c("day_range_used", 0, ">", p["used"]), c("day_range_pos", 0, ">", 1 - p["edge"])], lambda p: (after_clock(385), after_clock(385)))
fam(67, "forecast_gated_entries", V, "Forecast-gated entries", "Enter a base signal only when forecast volatility exceeds realized volatility.",
    "Kill if gated mean R <= ungated mean R.", "OHLCV", {"per": [50, 100]},
    lambda p: [c("kalman_slope_z", p["per"], ">", 1.0), _cond(g("garch_vol", 100), ">", g("yang_zhang_vol", 100))],
    lambda p: [c("kalman_slope_z", p["per"], "<", -1.0), _cond(g("garch_vol", 100), ">", g("yang_zhang_vol", 100))])
fam(68, "unexplained_jump_fade", V, "Unexplained-jump fade", "A statistically detected jump with no scheduled event reverts.",
    "Kill if mean R <= 0 or jumps that coincide with events behave the same (then it is not 'unexplained').", "OHLCV", {"j": [0.15, 0.25]},
    lambda p: [c("bipower_jump_ratio", 50, ">", p["j"]), c("event_day_any", 0, "<", 0.5), c("roc", 1, "<", 0.0)],
    lambda p: [c("bipower_jump_ratio", 50, ">", p["j"]), c("event_day_any", 0, "<", 0.5), c("roc", 1, ">", 0.0)])
fam(69, "downside_semivariance_stand_aside", V, "Downside-semivariance stand-aside", "Stand aside when downside semivariance is elevated.",
    "Kill if the stand-aside rule's mean R <= the unfiltered base signal's.", "OHLCV", {"r": [0.55, 0.65]},
    lambda p: [c("kalman_slope_z", 60, ">", 1.0), c("semivar_ratio", 50, "<", p["r"])], lambda p: [c("kalman_slope_z", 60, "<", -1.0), c("semivar_ratio", 50, ">", 1 - p["r"])])

# ==================================================================================================== 7. cross-asset and macro
fam(70, "risk_on_off_composite", X, "Risk-on and risk-off composite", "A composite of credit and the dollar predicts index direction.",
    "Kill if mean R <= the credit-only or dollar-only version.", "Basket, Ext", {"hy": [0.0, 0.1]},
    lambda p: [c("hy_spread_chg", 20, "<", -p["hy"]), c("dollar_change_20d", 0, "<", 0.0)], lambda p: [c("hy_spread_chg", 20, ">", p["hy"]), c("dollar_change_20d", 0, ">", 0.0)],
    status="needs_data", tied=True, aux="Ext feed: hy_oas; macro dollar")
fam(71, "dollar_driven_fx_trend", X, "Dollar-driven FX trend", "A dollar-index trend carries through the majors (dollar_sign = -1 for pairs quoted XXXUSD, +1 for USDXXX; both are booked).",
    "Kill if the dollar signal adds nothing over the pair's own trend.", "Basket", {"d": [1.0, 2.0], "dollar_sign": [-1, 1]},
    lambda p: [c("dollar_change_20d", 0, ">" if p["dollar_sign"] > 0 else "<", p["d"] if p["dollar_sign"] > 0 else -p["d"])],
    lambda p: [c("dollar_change_20d", 0, "<" if p["dollar_sign"] > 0 else ">", -p["d"] if p["dollar_sign"] > 0 else p["d"])], status="needs_data", tied=True, aux="macro dollar index")
fam(72, "yield_curve_regime_tilt", X, "Yield-curve regime tilt", "The curve's regime tilts equity-index and gold direction (curve_10y2y exists).",
    "Kill if mean R <= 0 or the regime flips fewer than 5 times in the sample (then n is tiny).", "Ext", {"slope": [0.0, 0.5]},
    lambda p: [c("curve_10y2y", 0, ">", p["slope"]), c("kalman_slope_z", 100, ">", 0.0)], lambda p: [c("curve_10y2y", 0, "<", -p["slope"]), c("kalman_slope_z", 100, "<", 0.0)],
    status="needs_data", tied=True, aux="macro 10y and 2y yields")
fam(73, "real_yield_gold", X, "Real-yield-to-gold model", "Gold reacts inversely to 10-year real-yield changes.",
    "Kill if mean R <= 0 on gold or the relation is unstable across halves.", "Ext", {"t": [0.05, 0.1], "per": [5, 20]},
    lambda p: [c("real_yield_chg", p["per"], "<", -p["t"])], lambda p: [c("real_yield_chg", p["per"], ">", p["t"])], status="needs_data", tied=True, aux="Ext feed: real_yield")
fam(74, "copper_gold_equity_tilt", X, "Copper-to-gold equity tilt", "A rising copper/gold ratio favors growth assets.",
    "Kill if mean R <= 0 or equals a plain trend.", "Basket", {"z": [1.0, 2.0], "per": [60, 250]},
    lambda p: [c("ratio_copper_gold", p["per"], ">", p["z"])], lambda p: [c("ratio_copper_gold", p["per"], "<", -p["z"])], status="needs_data", tied=True, aux="frame.attrs['copper'] and ['gold']")
fam(75, "oil_usdcad_leadlag", X, "Oil-to-USDCAD lead-lag", "Crude moves lead USDCAD (inverse); both signs are booked via oil_sign.",
    "Kill if mean R <= 0 or the lead-lag correlation is not significant at lags 1-5.", "Basket", {"z": [1.5, 2.5], "oil_sign": [-1, 1]},
    lambda p: [c("oil_ret_z", 20, ">" if p["oil_sign"] > 0 else "<", p["z"] if p["oil_sign"] > 0 else -p["z"])],
    lambda p: [c("oil_ret_z", 20, "<" if p["oil_sign"] > 0 else ">", -p["z"] if p["oil_sign"] > 0 else p["z"])], status="needs_data", tied=True, aux="frame.attrs['oil']")
fam(76, "audusd_copper_leadlag", X, "AUDUSD versus copper", "Copper leads the Australian dollar.",
    "Kill if mean R <= 0 or the lead-lag is not significant.", "Basket", {"z": [1.5, 2.5]},
    lambda p: [c("copper_ret_z", 20, ">", p["z"])], lambda p: [c("copper_ret_z", 20, "<", -p["z"])], status="needs_data", tied=True, aux="frame.attrs['copper']")
fam(77, "gold_silver_ratio_rotation", X, "Gold-silver ratio rotation", "Extremes in the ratio revert (trade_gold = 1 shorts gold when the ratio is high; 0 trades silver).",
    "Kill if mean R <= 0 or the ratio does not mean-revert (ADF fails).", "Basket", {"z": [2.0, 2.5], "per": [250, 500], "trade_gold": [0, 1]},
    lambda p: [c("ratio_gold_silver", p["per"], "<" if p["trade_gold"] else ">", -p["z"] if p["trade_gold"] else p["z"])],
    lambda p: [c("ratio_gold_silver", p["per"], ">" if p["trade_gold"] else "<", p["z"] if p["trade_gold"] else -p["z"])], status="needs_data", tied=True, aux="frame.attrs['gold'] and ['silver']")
fam(78, "inflation_surprise_reaction", X, "Inflation-surprise reaction", "Surprises move the dollar and gold in a consistent direction (surprise_sign booked both ways).",
    "Kill if mean R <= 0 over 30+ releases.", "Ext", {"s": [5.0, 15.0], "surprise_sign": [-1, 1]},
    lambda p: [c("econ_surprise_chg", 5, ">" if p["surprise_sign"] > 0 else "<", p["s"] if p["surprise_sign"] > 0 else -p["s"])],
    lambda p: [c("econ_surprise_chg", 5, "<" if p["surprise_sign"] > 0 else ">", -p["s"] if p["surprise_sign"] > 0 else p["s"])], status="needs_data", tied=True, aux="Ext feed: econ_surprise")
fam(79, "cot_extremes_contrarian", X, "COT extremes contrarian", "Extreme speculator positioning precedes reversals.",
    "Kill if mean R <= 0 (weekly data: only ~50 independent observations a year).", "Ext", {"q": [90, 95]},
    lambda p: [c("cot_net_spec_pct", 156, "<", 100 - p["q"])], lambda p: [c("cot_net_spec_pct", 156, ">", p["q"])], status="needs_data", tied=True, aux="Ext feed: cot_net_spec_<SYMBOL>")
fam(80, "putcall_extremes_contrarian", X, "Put/call extremes contrarian", "Extreme put/call readings precede reversals.",
    "Kill if mean R <= 0 or the effect is gone after 2015.", "Ext", {"z": [1.5, 2.5]},
    lambda p: [c("putcall_z", 250, ">", p["z"])], lambda p: [c("putcall_z", 250, "<", -p["z"])], status="needs_data", tied=True, aux="Ext feed: putcall")
fam(81, "credit_spread_risk_off", X, "Credit-spread risk-off exit", "Widening high-yield spreads signal exiting long equity exposure.",
    "Kill if the exit rule's mean R <= buy-and-hold's.", "Ext", {"w": [0.2, 0.5]},
    lambda p: [c("hy_spread_chg", 20, "<", 0.0), c("kalman_slope_z", 100, ">", 0.0)], lambda p: [c("hy_spread_chg", 20, ">", p["w"]), c("kalman_slope_z", 100, "<", 0.0)],
    status="needs_data", tied=True, aux="Ext feed: hy_oas")
fam(82, "xs_momentum_basket", X, "Cross-sectional momentum basket", "Long the top and short the bottom of a basket of FX majors or crypto.",
    "Kill if the long/short spread's mean R <= 0 net of cost or the basket has < 5 instruments.", "Basket", {"per": [60, 250], "q": [20, 30]},
    lambda p: [c("xs_momentum_rank", p["per"], ">", 100 - p["q"])], lambda p: [c("xs_momentum_rank", p["per"], "<", p["q"])], status="needs_data", tied=True, aux="frame.attrs['basket']")
fam(83, "fx_carry_plus_momentum", X, "FX carry plus momentum", "Rate differential and trend together beat either alone.",
    "Kill if the combination's mean R <= the better of the two alone.", "Basket, Ext", {"carry": [0.5, 1.0]},
    lambda p: [c("fx_carry_diff", 0, ">", p["carry"]), c("kalman_slope_z", 100, ">", 0.5)], lambda p: [c("fx_carry_diff", 0, "<", -p["carry"]), c("kalman_slope_z", 100, "<", -0.5)],
    status="needs_data", tied=True, aux="Ext feed: rate_diff")
fam(84, "metals_dispersion", X, "Metals-complex dispersion", "Relative strength within gold, silver, platinum and copper persists (the same construction as xs_momentum_basket on a metals basket: retired as a near-duplicate).",
    "Merged into xs_momentum_basket.", "Basket", {"per": [60, 250]},
    lambda p: [c("xs_momentum_rank", p["per"], ">", 75)], lambda p: [c("xs_momentum_rank", p["per"], "<", 25)], status="needs_data", tied=True, aux="frame.attrs['basket'] of metals",
    retired_to="xs_momentum_basket")

# ===================================================================================================== 8. statistical and model
fam(85, "hmm_regime_switch", P, "HMM-gated trend or reversion switch", "A filtered regime probability selects between trend and reversion rules (see existing regime_router).",
    "Kill if the switched rule's mean R <= the better fixed rule's.", "OHLCV", {"p": [0.7, 0.85]},
    lambda p: [c("hmm_bull_prob", 500, ">", p["p"]), c("kalman_slope_z", 60, ">", 0.5)], lambda p: [c("hmm_bull_prob", 500, "<", 1 - p["p"]), c("kalman_slope_z", 60, "<", -0.5)])
fam(86, "changepoint_trend_start", P, "Change-point trend start", "Entering when a change-point probability spikes catches new trends early.",
    "Kill if mean R <= the trend entry without the change-point condition.", "OHLCV", {"cp": [0.3, 0.5]},
    lambda p: [c("bocpd_prob", 10, ">", p["cp"]), c("kalman_slope_z", 30, ">", 0.5)], lambda p: [c("bocpd_prob", 10, ">", p["cp"]), c("kalman_slope_z", 30, "<", -0.5)])
fam(87, "cusum_structural_break", P, "CUSUM structural-break entry", "A persistent drift detected by CUSUM continues.",
    "Kill if mean R <= 0.", "OHLCV", {"h": [3.0, 5.0], "per": [50, 200]},
    lambda p: [c("cusum_stat", p["per"], ">", p["h"])], lambda p: [c("cusum_stat", p["per"], "<", -p["h"])])
fam(88, "kalman_dynamic_hedge_spread", P, "Kalman dynamic-hedge spread", "A spread with a time-varying hedge ratio reverts better than a fixed ratio.",
    "Kill if not better than the fixed-ratio stat_pairs.", "Basket", {"z": [1.5, 2.5]},
    lambda p: [c("kalman_hedge_spread_z", 0, "<", -p["z"])], lambda p: [c("kalman_hedge_spread_z", 0, ">", p["z"])], lambda p: ([c("kalman_hedge_spread_z", 0, ">", 0.0)], [c("kalman_hedge_spread_z", 0, "<", 0.0)]),
    status="needs_data", tied=True, aux="frame.attrs['benchmark']")
fam(89, "meta_labeling_filter", P, "Meta-labeling filter", "A secondary model that decides whether to take a primary signal improves precision.",
    "Kill if the meta-filtered mean R <= the primary's (paired).", "OHLCV", {"pm": [0.55, 0.6], "refit": [250, 500]},
    lambda p: [_cond(_ind("ema", 12), ">", _ind("ema", 48)), c("meta_label_prob", p["refit"], ">", p["pm"])], lambda p: [_cond(_ind("ema", 12), "<", _ind("ema", 48)), c("meta_label_prob", p["refit"], ">", p["pm"])])
fam(90, "triple_barrier_classifier", P, "Triple-barrier classifier", "A supervised model on barrier-labeled outcomes finds structure the rules miss.",
    "Kill if mean R <= 0 out of sample, or the model's AUC on the sealed slice is < 0.51.", "OHLCV", {"t": [0.03, 0.06], "refit": [250, 500]},
    lambda p: [c("triple_barrier_prob", p["refit"], ">", p["t"])], lambda p: [c("triple_barrier_prob", p["refit"], "<", -p["t"])])
fam(91, "weak_signal_ensemble_vote", P, "Weak-signal ensemble vote", "An n-of-m vote across many weak, uncorrelated signals beats any one.",
    "Kill if the vote's mean R <= the best single voter's.", "OHLCV", {"k": [4, 6], "n": [14, 30]},
    lambda p: [c("vote_score", p["n"], ">=", p["k"])], lambda p: [c("vote_score", p["n"], "<=", -p["k"])])
fam(92, "rolling_ridge_forecast", P, "Rolling ridge forecast", "A causal linear model on a small feature set forecasts the next return sign.",
    "Kill if the forecast's information coefficient on the sealed slice is <= 0.", "OHLCV", {"z": [0.5, 1.0], "refit": [250, 500]},
    lambda p: [c("ridge_forecast", p["refit"], ">", p["z"])], lambda p: [c("ridge_forecast", p["refit"], "<", -p["z"])])
fam(93, "analog_pattern_match", P, "Analog pattern match", "Windows that resemble past windows resolve similarly.",
    "Kill if mean R <= 0 or the forecast IC is not above that of shuffled library windows.", "OHLCV", {"w": [20, 40], "z": [0.3, 0.6]},
    lambda p: [c("analog_match", p["w"], ">", p["z"])], lambda p: [c("analog_match", p["w"], "<", -p["z"])])
fam(94, "seasonal_cv_forecast_family", P, "Seasonality-adjusted cross-validated forecast", "A forecast with seasonal terms, tested under purged cross-validation.",
    "Kill if sealed-slice IC <= 0 or it does not beat rolling_ridge_forecast.", "OHLCV", {"z": [0.5, 1.0], "refit": [250, 500]},
    lambda p: [c("seasonal_cv_forecast", p["refit"], ">", p["z"])], lambda p: [c("seasonal_cv_forecast", p["refit"], "<", -p["z"])])

# ======================================================================================== 9. position management, crypto-specific
blocked(95, "anti_martingale_pyramiding", PM, "Anti-martingale pyramiding", "Adding to winners with a hard risk cap raises the expectancy of a trend signal.",
        "n/a -- not runnable", "OHLCV", "the backtest engine has no scale-in / add-to-position")
fam(96, "equity_curve_throttle", PM, "Equity-curve-aware throttle", "Trading a base signal only while its own recent expectancy is positive.",
    "Kill if the throttled mean R <= the unthrottled base signal's (paired).", "OHLCV", {"n": [100, 250], "t": [0.0, 1.0]},
    lambda p: [_cond(_ind("ema", 12), ">", _ind("ema", 48)), c("base_signal_expectancy", p["n"], ">", p["t"])], lambda p: [_cond(_ind("ema", 12), "<", _ind("ema", 48)), c("base_signal_expectancy", p["n"], ">", p["t"])])
fam(97, "time_stop_scalp", PM, "Time-stop scalp", "Trades with no follow-through within N bars are likely losers (the time stop is the max_bars grid: 6/12/24 here replace the default).",
    "Kill if mean R with the time stop <= the same entries held to stop/target.", "OHLCV", {"n": [6, 12, 24]},
    lambda p: [c("kalman_slope_z", 30, ">", 1.0), c("roc", 3, ">", 0.0)], lambda p: [c("kalman_slope_z", 30, "<", -1.0), c("roc", 3, "<", 0.0)],
    lambda p: ([c("kalman_slope_z", 30, "<", 0.0)], [c("kalman_slope_z", 30, ">", 0.0)]))
fam(98, "funding_rate_extreme_contrarian", PM, "Funding-rate extreme contrarian", "Extreme perpetual funding precedes liquidation reversals.",
    "Kill if mean R <= 0 or the effect disappears after fees.", "Ext", {"z": [2.0, 3.0]},
    lambda p: [c("funding_rate_z", 100, "<", -p["z"])], lambda p: [c("funding_rate_z", 100, ">", p["z"])], status="needs_data", tied=True, aux="Ext feed: funding_rate (crypto perps)")
fam(99, "liquidation_cascade_reversal", PM, "Liquidation-cascade reversal", "A drop in open interest with range expansion marks forced selling that reverses.",
    "Kill if mean R <= 0.", "Ext", {"q": [90, 95]},
    lambda p: [c("oi_liq_proxy", 5, ">", p["q"]), c("roc", 3, "<", 0.0)], lambda p: [c("oi_liq_proxy", 5, ">", p["q"]), c("roc", 3, ">", 0.0)], status="needs_data", tied=True, aux="Ext feed: open_interest")
fam(100, "perp_spot_basis_convergence", PM, "Perp-spot basis convergence", "A wide basis converges as leverage unwinds.",
    "Kill if mean R <= 0.", "Ext", {"b": [20.0, 50.0]},
    lambda p: [c("perp_spot_basis", 0, "<", -p["b"])], lambda p: [c("perp_spot_basis", 0, ">", p["b"])], status="needs_data", tied=True, aux="Ext feed: perp_premium")

# ------------------------------------------------------------------------------------------------------ exported structure
# Metadata keys the status of every family carries.
V13_FAMILY_NAMES: list[str] = list(V13_FAMILIES)
V13_HYPOTHESIS_QUESTIONS: dict[str, str] = {n: s.description for n, s in V13_FAMILIES.items()}
V13_FAMILY_GROUPS: dict[str, str] = {n: m.group for n, m in V13_META.items()}
V13_INSTRUMENT_TIED: frozenset = frozenset(n for n, m in V13_META.items() if m.tied)
V13_REQUIRES_AUX: dict[str, str] = {n: m.aux for n, m in V13_META.items() if m.aux}
V13_NEEDS_DATA: frozenset = frozenset(n for n, m in V13_META.items() if m.status == "needs_data")
V13_ALL_NUMBERS: dict[int, str] = {**{m.number: n for n, m in V13_META.items()}, **{k: m.name for k, m in V13_BLOCKED.items()}}


def families_in_batch(batch: int) -> list[str]:
    return [n for n, m in V13_META.items() if m.batch == batch and n not in RETIRED_V13]
