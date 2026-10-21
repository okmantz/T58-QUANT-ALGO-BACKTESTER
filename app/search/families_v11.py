"""v11 (2026-10-20) -- structural / event / calendar / regime template families (report sections 5-6).

The report's diagnosis was that the search space was oscillator-threshold templates on 5m bars. Each family below is a
PRE-REGISTRABLE hypothesis: one economic reason, a small fixed grid (the grid size IS the trial count that gets booked to the
trial ledger), ATR-multiple risk, and primitives from app.strategy.indicators_v11 (clock, calendar, event, overnight, daily
context, regime). Nothing here is tuned on the user's data; run them through app.research.preregistered_tests.

  event_post_release_drift    vs trend families: only the window after a scheduled release (FOMC/NFP/CPI), direction = the
                              release reaction already printed
  event_overreaction_fade     vs rsi fades: only inside the post-release window, only on an extreme reaction
  overnight_extreme_fade      vs band fades: the band is the overnight range, only in the first RTH hours
  overnight_range_trend       vs trend_breakout: break from overnight range only when the DAILY trend agrees
  daily_trend_pullback_rth    vs mtf_pullback: higher-timeframe state is the prior daily stack, entry in the opening hours
  daily_momentum_open_drive   vs momentum families: time-series momentum of the prior daily distance, taken at the open
  turn_of_month_flow_long     calendar flow: pension/fund rebalancing around month turn; long only
  opex_pin_fade               vs vwap_reversion: expiry-week afternoons, when pinning is strongest
  pre_holiday_drift_long      calendar drift: the session before an exchange holiday; long only
  vix_backwardation_dip       regime: buy a sharp dip only while VIX is in backwardation (needs data/macro VIX + VIX3M)
  leader_follow               lead-lag: follow the leader instrument's recent move (needs frame.attrs['leader'])

Families depending on macro or leader data produce zero signals when that data is absent (their primitives return NaN/0), so
they can never create a trade out of missing inputs. Relative-value pairs (NQ/ES, gold/silver) already exist as `stat_pairs`.
"""
from __future__ import annotations

from app.search.strategy_space import SkeletonSpec, _cond, _ind, _risk_management, _val


def _rm(p: dict) -> dict:
    return _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"])


def _g(kind: str, period: int = 0) -> dict:
    return {"type": kind, "period": period, "field": "close"}


# 2026-10-20 (report section 3, "widen exits"): ATR stops of 1-1.5 and targets of 1.5-2.5 are inside the noise floor of 1m/5m bars, so
# costs dominate (a 0.15R round-trip cost is an edge killer). Wider stops/targets and longer holds shrink cost-in-R and let a real
# structural edge show. Position size stays constant-RISK: size_for_stop() divides the risk budget by the stop distance, so a wider
# ATR stop automatically trades fewer contracts - that IS the volatility-scaled sizing, with no extra gene needed.
_RISK_GRID = {"stop_atr_mult": [1.0, 2.0, 3.0], "target_atr_mult": [2.0, 4.0, 6.0], "max_bars": [36, 96]}
_GOOD = lambda p: p["target_atr_mult"] >= p["stop_atr_mult"]  # noqa: E731


def _sides(long_c: list, short_c: list) -> dict:
    return {"long": long_c, "long_connectors": ["AND"] * max(len(long_c) - 1, 0),
            "short": short_c, "short_connectors": ["AND"] * max(len(short_c) - 1, 0)}


def _exits(long_c: list, short_c: list | None = None, mode: str = "OR") -> dict:
    """Explicit exit rules. The signal loop is a position state machine: without an exit condition a position opened inside a
    window stays 'on' after the window closes and the engine re-enters on every stop/target, trading hours the hypothesis says
    nothing about. Every windowed family therefore flattens when its window ends."""
    short_c = long_c if short_c is None else short_c
    return {"long": long_c, "long_connectors": [mode] * max(len(long_c) - 1, 0),
            "short": short_c, "short_connectors": [mode] * max(len(short_c) - 1, 0)}


def _after_clock(end_min: float) -> list:
    """True once the RTH clock is past ``end_min`` minutes after the open, or before the open (overnight)."""
    return [_cond(_g("minutes_since_open"), ">", _val(end_min)), _cond(_g("minutes_since_open"), "<", _val(0))]


def _event_key(p: dict) -> str:
    return str(p["event"]).lower()


# 1 -- post-release drift
def _build_event_drift(p: dict) -> dict:
    ev = _event_key(p)
    win = [_cond(_g(f"minutes_since_{ev}"), ">=", _val(p["start_min"])), _cond(_g(f"minutes_since_{ev}"), "<=", _val(p["start_min"] + p["window_min"]))]
    return {
        "name": f"{p['event']} post-release drift ({p['start_min']}m+{p['window_min']}m)",
        "entry_conditions": _sides(win + [_cond(_ind("roc", p["roc_bars"]), ">", _val(p["roc_min"]))],
                                   win + [_cond(_ind("roc", p["roc_bars"]), "<", _val(-p["roc_min"]))]),
        "exit_conditions": _exits([_cond(_g(f"minutes_since_{ev}"), ">", _val(p["start_min"] + p["window_min"]))]),
        "risk_management": _rm(p),
    }


_EVENT_DRIFT = SkeletonSpec(
    name="event_post_release_drift", label="Event post-release drift",
    description="After a scheduled macro release (FOMC/NFP), the reaction printed in the first minutes continues for a while as "
                "slower participants reprice. Enters with the reaction between start_min and start_min+window_min after the release.",
    param_grid={"event": ["FOMC", "NFP"], "start_min": [5, 15], "window_min": [45], "roc_bars": [3], "roc_min": [0.05, 0.15], **_RISK_GRID},
    build=_build_event_drift, valid=_GOOD,
)


# 2 -- overreaction fade
def _build_event_fade(p: dict) -> dict:
    ev = _event_key(p)
    win = [_cond(_g(f"minutes_since_{ev}"), ">=", _val(p["start_min"])), _cond(_g(f"minutes_since_{ev}"), "<=", _val(p["start_min"] + 90))]
    return {
        "name": f"{p['event']} overreaction fade (RSI {p['rsi_cap']})",
        "entry_conditions": _sides(win + [_cond(_ind("rsi", 7), "<", _val(p["rsi_cap"]))], win + [_cond(_ind("rsi", 7), ">", _val(100 - p["rsi_cap"]))]),
        "exit_conditions": _exits([_cond(_g(f"minutes_since_{ev}"), ">", _val(p["start_min"] + 90))]),
        "risk_management": _rm(p),
    }


_EVENT_FADE = SkeletonSpec(
    name="event_overreaction_fade", label="Event overreaction fade",
    description="An extreme short-term reaction (RSI(7) at the tails) inside 90 minutes of a scheduled release often over-shoots "
                "thin liquidity and retraces. Fades only in that window.",
    param_grid={"event": ["FOMC", "NFP"], "start_min": [10, 30], "rsi_cap": [15, 25], **_RISK_GRID},
    build=_build_event_fade, valid=_GOOD,
)


# 3 -- overnight extreme fade
def _build_overnight_fade(p: dict) -> dict:
    rth = [_cond(_g("minutes_since_open"), ">=", _val(0)), _cond(_g("minutes_since_open"), "<=", _val(p["window_min"]))]
    return {
        "name": f"Overnight extreme fade (<{p['edge']}/>{1 - p['edge']:.2f} of range)",
        "entry_conditions": _sides(rth + [_cond(_g("overnight_position"), "<", _val(p["edge"]))], rth + [_cond(_g("overnight_position"), ">", _val(1 - p["edge"]))]),
        "exit_conditions": _exits(_after_clock(p["window_min"])), "risk_management": _rm(p),
    }


_OVERNIGHT_FADE = SkeletonSpec(
    name="overnight_extreme_fade", label="Overnight-range extreme fade",
    description="In the first RTH hours, price sitting at the extreme of the overnight range tends to be pulled back into it "
                "(overnight positioning unwinds against the cash-session order book).",
    param_grid={"edge": [0.1, 0.2], "window_min": [60, 120], **_RISK_GRID}, build=_build_overnight_fade, valid=_GOOD,
)


# 4 -- overnight range trend with daily agreement
def _build_overnight_trend(p: dict) -> dict:
    rth = [_cond(_g("minutes_since_open"), ">=", _val(0)), _cond(_g("minutes_since_open"), "<=", _val(p["window_min"]))]
    return {
        "name": f"Overnight range trend (range >= {p['min_range']} ATR)",
        "entry_conditions": _sides(
            rth + [_cond(_g("daily_trend_state"), ">", _val(0.5)), _cond(_g("overnight_position"), ">", _val(0.8)), _cond(_g("overnight_range_atr"), ">", _val(p["min_range"]))],
            rth + [_cond(_g("daily_trend_state"), "<", _val(-0.5)), _cond(_g("overnight_position"), "<", _val(0.2)), _cond(_g("overnight_range_atr"), ">", _val(p["min_range"]))]),
        "exit_conditions": _exits(_after_clock(p["window_min"])), "risk_management": _rm(p),
    }


_OVERNIGHT_TREND = SkeletonSpec(
    name="overnight_range_trend", label="Overnight range trend (daily-trend aligned)",
    description="When the daily EMA stack and the overnight session agree and the overnight range is meaningful, a hold at the "
                "overnight extreme in the opening hours is continuation, not exhaustion.",
    param_grid={"min_range": [0.3, 0.6], "window_min": [90], **_RISK_GRID}, build=_build_overnight_trend, valid=_GOOD,
)


# 5 -- daily trend pullback
def _build_daily_pullback(p: dict) -> dict:
    rth = [_cond(_g("minutes_since_open"), ">=", _val(0)), _cond(_g("minutes_since_open"), "<=", _val(p["window_min"]))]
    return {
        "name": f"Daily-trend pullback at open (EMA{p['ema']} dist <= {p['dist']})",
        "entry_conditions": _sides(
            rth + [_cond(_g("daily_trend_state"), ">", _val(0.5)), _cond(_g("daily_ema_distance", p["ema"]), "<", _val(p["dist"])), _cond(_ind("rsi", 14), "<", _val(p["rsi_cap"]))],
            rth + [_cond(_g("daily_trend_state"), "<", _val(-0.5)), _cond(_g("daily_ema_distance", p["ema"]), ">", _val(-p["dist"])), _cond(_ind("rsi", 14), ">", _val(100 - p["rsi_cap"]))]),
        "exit_conditions": _exits(_after_clock(p["window_min"])), "risk_management": _rm(p),
    }


_DAILY_PULLBACK = SkeletonSpec(
    name="daily_trend_pullback_rth", label="Daily-trend pullback (opening hours)",
    description="Buys short-term weakness when the prior-day EMA50/200 stack is bullish and price has pulled back toward the daily "
                "EMA; mirror image for shorts. Daily state is strictly the prior completed day.",
    param_grid={"ema": [20, 50], "dist": [0.5, 1.0], "rsi_cap": [40], "window_min": [120], **_RISK_GRID},
    build=_build_daily_pullback, valid=_GOOD,
)


# 6 -- daily time-series momentum at the open
def _build_daily_mom(p: dict) -> dict:
    rth = [_cond(_g("minutes_since_open"), ">=", _val(p["start_min"])), _cond(_g("minutes_since_open"), "<=", _val(p["start_min"] + 30))]
    return {
        "name": f"Daily momentum open drive (dist > {p['dist']} ATR)",
        "entry_conditions": _sides(rth + [_cond(_g("daily_ema_distance", 50), ">", _val(p["dist"])), _cond(_ind("roc", 6), ">", _val(0))],
                                   rth + [_cond(_g("daily_ema_distance", 50), "<", _val(-p["dist"])), _cond(_ind("roc", 6), "<", _val(0))]),
        "exit_conditions": _exits(_after_clock(p["start_min"] + 30)), "risk_management": _rm(p),
    }


_DAILY_MOM = SkeletonSpec(
    name="daily_momentum_open_drive", label="Daily time-series momentum (open drive)",
    description="Time-series momentum: when the prior daily close is far above (below) its daily EMA50, the first RTH half-hour "
                "drive in the same direction continues.",
    param_grid={"dist": [1.0, 2.0], "start_min": [5, 30], **_RISK_GRID}, build=_build_daily_mom, valid=_GOOD,
)


# 7 -- turn of month flow
def _build_tom(p: dict) -> dict:
    rth = [_cond(_g("turn_of_month"), "is true", _val(1)), _cond(_g("minutes_since_open"), ">=", _val(p["start_min"])), _cond(_g("minutes_since_open"), "<=", _val(p["start_min"] + 60))]
    return {
        "name": "Turn-of-month flow (long)", "entry_conditions": _sides(rth + [_cond(_ind("ema", p["ema"]), "<", {"type": "close", "period": 1, "field": "close"})], []),
        "exit_conditions": _exits(_after_clock(p["start_min"] + 60), []), "risk_management": _rm(p),
    }


_TOM = SkeletonSpec(
    name="turn_of_month_flow_long", label="Turn-of-month flow (long)",
    description="Fund and pension inflows and rebalancing cluster around the month turn (last trading day through the 3rd). "
                "Long in the opening hour when price is above its short EMA. Long only; no shorting of a flow edge.",
    param_grid={"ema": [20, 50], "start_min": [5, 60], **_RISK_GRID}, build=_build_tom, valid=_GOOD,
)


# 8 -- opex pin fade
def _build_opex(p: dict) -> dict:
    late = [_cond(_g("opex_week"), "is true", _val(1)), _cond(_g("minutes_to_close"), ">", _val(0)), _cond(_g("minutes_to_close"), "<", _val(p["late_min"]))]
    z = {"type": "vwap_zscore", "period": 20, "field": "close"}
    return {
        "name": f"Opex-week afternoon pin fade (|z|>{p['z']})",
        "entry_conditions": _sides(late + [_cond(z, "<", _val(-p["z"]))], late + [_cond(z, ">", _val(p["z"]))]),
        "exit_conditions": _exits([_cond(_g("minutes_to_close"), "<", _val(0)), _cond(_g("minutes_to_close"), ">", _val(p["late_min"]))]),
        "risk_management": _rm(p),
    }


_OPEX = SkeletonSpec(
    name="opex_pin_fade", label="Opex-week afternoon pin fade",
    description="In monthly options-expiry week, late-session dealer hedging pins price toward high-open-interest strikes; "
                "stretches from VWAP in the last hours fade back.",
    param_grid={"z": [1.5, 2.0], "late_min": [120, 180], **_RISK_GRID}, build=_build_opex, valid=_GOOD,
)


# 9 -- pre-holiday drift
def _build_preholiday(p: dict) -> dict:
    rth = [_cond(_g("pre_holiday"), "is true", _val(1)), _cond(_g("minutes_since_open"), ">=", _val(p["start_min"])), _cond(_g("minutes_since_open"), "<=", _val(p["start_min"] + 30))]
    return {"name": "Pre-holiday drift (long)", "entry_conditions": _sides(rth, []), "exit_conditions": _exits(_after_clock(p["start_min"] + 30), []),
            "risk_management": _rm(p)}


_PREHOLIDAY = SkeletonSpec(
    name="pre_holiday_drift_long", label="Pre-holiday drift (long)",
    description="Equity indices have drifted up into exchange holidays (reduced selling, positioning ahead of the break). Long "
                "only, entered in the first RTH hour of the last session before a closure.",
    param_grid={"start_min": [5, 30], **_RISK_GRID}, build=_build_preholiday, valid=_GOOD,
)


# 10 -- VIX backwardation dip
def _build_vix_dip(p: dict) -> dict:
    return {
        "name": f"VIX backwardation dip (slope > {p['slope']})",
        "entry_conditions": _sides([_cond(_g("vix_term_slope"), ">", _val(p["slope"])), _cond(_ind("rsi", 7), "<", _val(p["rsi_cap"])),
                                    _cond(_g("rth_flag"), "is true", _val(1))], []),
        "exit_conditions": _exits([_cond(_ind("rsi", 7), ">", _val(50)), _cond(_g("rth_flag"), "is false", _val(0))], []), "risk_management": _rm(p),
    }


_VIX_DIP = SkeletonSpec(
    name="vix_backwardation_dip", label="VIX backwardation dip buy",
    description="When the VIX curve is inverted (stress) the market is statistically stretched; a sharp short-term dip inside "
                "that regime mean-reverts. Needs data/macro VIX and VIX3M (python -m scripts.update_macro_data); zero signals without them.",
    param_grid={"slope": [0.0, 0.05], "rsi_cap": [20, 30], **_RISK_GRID}, build=_build_vix_dip, valid=_GOOD,
)


# 11 -- leader follow
def _build_leader(p: dict) -> dict:
    lead = {"type": "leader_return", "period": p["bars"], "field": "close"}
    return {
        "name": f"Leader follow ({p['bars']} bars, >{p['thr']} ATR)",
        "entry_conditions": _sides([_cond(lead, ">", _val(p["thr"])), _cond(_ind("roc", p["bars"]), "<", _val(p["own_cap"]))],
                                   [_cond(lead, "<", _val(-p["thr"])), _cond(_ind("roc", p["bars"]), ">", _val(-p["own_cap"]))]),
        "exit_conditions": _exits([_cond(lead, "<", _val(p["thr"] / 4.0))], [_cond(lead, ">", _val(-p["thr"] / 4.0))]), "risk_management": _rm(p),
    }


_LEADER = SkeletonSpec(
    name="leader_follow", label="Lead-lag: follow the leader",
    description="When a leading instrument (set frame.attrs['leader']) has moved by more than a threshold in its own ATR and this "
                "instrument has not yet followed, trade the catch-up. Zero signals without a leader frame.",
    param_grid={"bars": [3, 6], "thr": [0.5, 1.0], "own_cap": [0.05], **_RISK_GRID}, build=_build_leader, valid=_GOOD,
)


# 12 -- time-series trend, held: the pure multi-market trend hypothesis (pool over 8-12 markets)
def _build_ts_trend(p: dict) -> dict:
    rth = [_cond(_g("minutes_since_open"), ">=", _val(5)), _cond(_g("minutes_since_open"), "<=", _val(p["window_min"]))]
    return {
        "name": f"Time-series trend hold (daily stack, EMA{p['ema']} dist > {p['dist']})",
        "entry_conditions": _sides(
            rth + [_cond(_g("daily_trend_state"), ">", _val(0.5)), _cond(_g("daily_ema_distance", p["ema"]), ">", _val(p["dist"]))],
            rth + [_cond(_g("daily_trend_state"), "<", _val(-0.5)), _cond(_g("daily_ema_distance", p["ema"]), "<", _val(-p["dist"]))]),
        "exit_conditions": _exits([_cond(_g("daily_trend_state"), "<", _val(0.5))], [_cond(_g("daily_trend_state"), ">", _val(-0.5))]),
        "risk_management": _rm(p),
    }


_TS_TREND = SkeletonSpec(
    name="ts_daily_trend_hold", label="Time-series trend (daily stack, multi-day hold)",
    description="Enters in the direction of the prior-day EMA50/200 stack when price is already extended from its daily EMA, early in "
                "the session, and holds until the stack flips. The classic managed-futures trend hypothesis; meant to be judged "
                "POOLED across 8-12 markets, never on one instrument.",
    param_grid={"ema": [20, 50], "dist": [0.0, 0.5], "window_min": [60], "stop_atr_mult": [2.0, 3.0, 4.0],
                "target_atr_mult": [6.0, 10.0], "max_bars": [288, 1440]},
    build=_build_ts_trend, valid=_GOOD,
)


# 13 -- intraday momentum: how the day has gone so far predicts the last hour
def _build_close_momentum(p: dict) -> dict:
    win = [_cond(_g("minutes_to_close"), "<=", _val(p["start_before_close"])), _cond(_g("minutes_to_close"), ">=", _val(5))]
    pdc = {"type": "previous_day_close"}
    return {
        "name": f"Intraday momentum into the close (day return > {p['min_atr_move']} ATR, last {p['start_before_close']} min)",
        "entry_conditions": _sides(
            win + [_cond(_ind("close", 1), ">", pdc), _cond(_ind("roc", p["roc"]), ">", _val(0))],
            win + [_cond(_ind("close", 1), "<", pdc), _cond(_ind("roc", p["roc"]), "<", _val(0))]),
        "exit_conditions": _exits([_cond(_g("minutes_to_close"), "<", _val(2)), _cond(_g("minutes_to_close"), ">", _val(p["start_before_close"] + 600))]),
        "risk_management": _rm(p),
    }


_CLOSE_MOM = SkeletonSpec(
    name="intraday_momentum_close", label="Intraday momentum into the close",
    description="Documented intraday-momentum effect: the session's direction so far (price vs prior close) continues through the "
                "final 60-90 minutes as hedging and rebalancing flows arrive. Flat before the cash close.",
    param_grid={"start_before_close": [60, 90], "roc": [6, 12], "min_atr_move": [0.0], **_RISK_GRID},
    build=_build_close_momentum, valid=_GOOD,
)


# 14 -- overnight vs intraday decomposition: the same asset, only one half of the day
def _build_overnight_intraday(p: dict) -> dict:
    mode = p["mode"]
    if mode == "overnight":
        # Windows are 15 minutes wide so they contain a bar on 1m, 5m and 15m data alike.
        enter = [_cond(_g("minutes_to_close"), "<=", _val(15)), _cond(_g("minutes_to_close"), ">=", _val(1))]
        exit_ = [_cond(_g("minutes_since_open"), ">=", _val(5)), _cond(_g("minutes_since_open"), "<=", _val(120))]
    else:
        enter = [_cond(_g("minutes_since_open"), ">=", _val(0)), _cond(_g("minutes_since_open"), "<=", _val(14))]
        exit_ = [_cond(_g("minutes_to_close"), "<=", _val(5)), _cond(_g("minutes_to_close"), ">=", _val(0))]
    return {
        "name": f"{mode.title()} drift ({'long' if p['direction'] > 0 else 'short'})",
        "entry_conditions": _sides(enter if p["direction"] > 0 else [], enter if p["direction"] < 0 else []),
        "exit_conditions": _exits(exit_, exit_),
        "risk_management": _rm(p),
    }


_OVN_INTRA = SkeletonSpec(
    name="overnight_vs_intraday_drift", label="Overnight vs intraday drift",
    description="Holds ONLY the overnight session (buy the close, sell the next open) or ONLY the cash session (buy the open, sell the "
                "close). Equity-index drift is known to be concentrated in one half of the day; this family measures which half, per "
                "market, instead of assuming it. Long and short both in the grid so the sign is tested, not presumed.",
    param_grid={"mode": ["overnight", "intraday"], "direction": [1, -1], "stop_atr_mult": [2.0, 3.0, 4.0],
                "target_atr_mult": [6.0, 10.0], "max_bars": [288, 600]},
    build=_build_overnight_intraday, valid=_GOOD,
)


# 15 -- volatility-regime gated breakout (self-contained: ATR percentile instead of VIX)
def _build_vol_regime_breakout(p: dict) -> dict:
    # Regime = the PRIOR completed day's range in daily ATR (calm yesterday). A rolling intraday ATR percentile cannot be used here:
    # RTH-open volatility ranks high against the quiet overnight bars, so that gate is almost never true at the open (measured: 28 of
    # 5,339 opening-window bars on NQ).
    gate = [_cond(_g("prior_day_range_atr", 14), "<=", _val(p["max_prior_range"])),
            _cond(_g("minutes_since_open"), ">=", _val(0)), _cond(_g("minutes_since_open"), "<=", _val(p["window_min"]))]
    return {
        "name": f"Vol-regime breakout (prior-day range <= {p['max_prior_range']} ATR, {p['lookback']}-bar break)",
        "entry_conditions": _sides(gate + [{"left": {"type": "bos", "lookback": p["lookback"], "direction": "bullish"}, "operator": "is true", "right": _val(1)}],
                                   gate + [{"left": {"type": "bos", "lookback": p["lookback"], "direction": "bearish"}, "operator": "is true", "right": _val(1)}]),
        "exit_conditions": _exits(_after_clock(p["window_min"])), "risk_management": _rm(p),
    }


_VOL_REGIME = SkeletonSpec(
    name="vol_regime_gated_breakout", label="Volatility-regime gated breakout",
    description="Takes an N-bar breakout in the opening window only after a CALM prior day (range below a threshold in daily ATR), "
                "where range expansion is least priced in; flat when the window ends. Needs no VIX feed.",
    param_grid={"max_prior_range": [0.8, 1.1], "lookback": [20, 40], "window_min": [90], **_RISK_GRID},
    build=_build_vol_regime_breakout, valid=_GOOD,
)


V11_FAMILIES: dict[str, SkeletonSpec] = {s.name: s for s in (
    _EVENT_DRIFT, _EVENT_FADE, _OVERNIGHT_FADE, _OVERNIGHT_TREND, _DAILY_PULLBACK, _DAILY_MOM, _TOM, _OPEX, _PREHOLIDAY, _VIX_DIP, _LEADER,
    _TS_TREND, _CLOSE_MOM, _OVN_INTRA, _VOL_REGIME)}
V11_FAMILY_NAMES: list[str] = list(V11_FAMILIES)
V11_HYPOTHESIS_QUESTIONS: dict[str, str] = {n: s.description for n, s in V11_FAMILIES.items()}
V11_FAMILY_GROUPS: dict[str, str] = {
    "event_post_release_drift": "event_driven", "event_overreaction_fade": "event_driven",
    "overnight_extreme_fade": "overnight_session", "overnight_range_trend": "overnight_session",
    "daily_trend_pullback_rth": "trend_following", "daily_momentum_open_drive": "momentum",
    "turn_of_month_flow_long": "calendar_seasonality", "opex_pin_fade": "calendar_seasonality", "pre_holiday_drift_long": "calendar_seasonality",
    "vix_backwardation_dip": "macro_regime", "leader_follow": "relative_strength",
    "ts_daily_trend_hold": "trend_following", "intraday_momentum_close": "momentum",
    "overnight_vs_intraday_drift": "overnight_session", "vol_regime_gated_breakout": "volatility_contraction",
}
# Families whose inputs live OUTSIDE the OHLCV frame: they trade nothing when the input is absent (by construction).
V11_REQUIRES_AUX: dict[str, str] = {"vix_backwardation_dip": "data/macro VIX + VIX3M", "leader_follow": "frame.attrs['leader']"}
