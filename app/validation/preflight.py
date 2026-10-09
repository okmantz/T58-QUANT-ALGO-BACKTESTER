"""
PREFLIGHT (2026-10-07 accuracy overhaul) -- structural checks that run BEFORE a
backtest/optimization is trusted. Each finding is an actionable sentence with a
severity: "block" (the numbers would be misleading), "warn" (read with care) or
"info". Nothing here changes a result; it tells the trader which assumption is
about to produce a number that a funded account would not reproduce.

Checks:
  * instrument/pip-size agreement with the known contract spec;
  * zero spread/slippage/commission on an instrument that has real costs;
  * the risk budget can actually buy one contract at the strategy's stop
    (sizing feasibility, per sizing mode) -- the #1 reason a "tested" account
    silently trades 0 or 1 contract;
  * the daily-loss limit vs. one stop-out, and drawdown room vs. one stop-out
    (a rule structure where a single normal loss locks or busts the account);
  * how many consecutive stop-outs the account survives;
  * sample floors: bars, trading days, trades, versus the evaluation horizon.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PreflightIssue:
    severity: str          # "block" | "warn" | "info"
    code: str
    message: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class PreflightReport:
    issues: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return any(i.severity == "block" for i in self.issues)

    def by_severity(self, sev: str) -> list:
        return [i for i in self.issues if i.severity == sev]

    def render(self) -> str:
        if not self.issues:
            return "Preflight: no structural problems found."
        tag = {"block": "BLOCK", "warn": "WARN ", "info": "INFO "}
        return "\n".join(f"[{tag.get(i.severity, i.severity)}] {i.message}" for i in self.issues)

    def to_dict(self) -> dict:
        return {"blocked": self.blocked, "issues": [i.to_dict() for i in self.issues], "facts": self.facts}


MIN_TRADES = 30
MIN_TRADING_DAYS = 40


def run_preflight(
    df,
    risk,
    rules=None,
    stop_loss_pips: float | None = None,
    stop_distance_price: float | None = None,
    trades=None,
    symbol: str | None = None,
    eval_days: int | None = None,
) -> PreflightReport:
    rep = PreflightReport()
    add = lambda sev, code, msg: rep.issues.append(PreflightIssue(sev, code, msg))  # noqa: E731

    spec = None
    try:
        from app.data.instrument_specs import get_any_instrument_spec as get_instrument_spec, guess_any_instrument_symbol as guess_instrument_symbol
        sym = guess_instrument_symbol(symbol) or symbol
        spec = get_instrument_spec(sym) if sym else None
    except Exception:  # noqa: BLE001
        spec = None

    # -- instrument agreement ---------------------------------------------
    if spec is not None:
        rep.facts["instrument"] = spec.symbol
        if abs(float(risk.pip_size) - float(spec.pip_size)) > 1e-12:
            add("block", "pip_size_mismatch",
                f"pip_size={risk.pip_size:g} but {spec.symbol} uses {spec.pip_size:g}: every stop, "
                "target, cost and position size would be scaled wrongly.")
        if risk.contract_size and abs(float(risk.contract_size) - float(spec.contract_size)) > 1e-9:
            add("warn", "contract_size_mismatch",
                f"contract_size={risk.contract_size:g} but {spec.symbol} is ${spec.contract_size:g}/point.")
        if (risk.spread_pips + risk.slippage_pips) == 0 and not getattr(risk, "commission_per_contract", 0):
            add("block", "zero_costs",
                f"Spread, slippage and commission are all zero on {spec.symbol}; a funded account pays "
                f"about ${spec.round_trip_cost_dollars():,.2f} per round turn per contract.")
        elif (risk.spread_pips + risk.slippage_pips) == 0:
            add("warn", "zero_spread_slippage", "Spread and slippage are zero; fills will be optimistic.")

    # -- sizing feasibility ------------------------------------------------
    D = stop_distance_price
    if D is None and stop_loss_pips:
        D = float(stop_loss_pips) * float(risk.pip_size)
    if D:
        eq = float(risk.initial_balance)
        dec = risk.size_for_stop(eq, float(D))
        rep.facts["sizing_mode"] = risk.sizing_mode
        rep.facts["budget_dollars"] = dec.budget
        if dec.skip_reason:
            sev = "block" if risk.sizing_mode == "skip" else "warn"
            add(sev, "sizing_infeasible",
                f"At a {D:g}-point stop the risk budget (${dec.budget:,.0f}) cannot buy one whole contract "
                f"({dec.skip_reason}). In sizing_mode='{risk.sizing_mode}' this strategy would take no trade at "
                "all (or only the ones with a tighter stop). Raise the risk, tighten the stop, or use "
                "fit_stop / micro_fallback.")
        else:
            rep.facts["contracts_at_start"] = dec.contracts
            if dec.above_budget:
                add("warn", "above_budget", "The minimum contract risks more than the budget.")
            if dec.stop_capped:
                add("info", "stop_capped",
                    f"fit_stop will shrink the stop to {dec.stop_distance:g} points "
                    f"({dec.stop_scale:.0%} of the strategy's own): the traded system differs from the "
                    "backtested one on wide-stop setups.")
            # rule structure
            if rules is not None and dec.risk_at_stop:
                acct = float(rules.account_size)
                dd_room = acct * float(rules.max_drawdown_pct) / 100.0
                dl = acct * float(rules.daily_loss_limit_pct) / 100.0 if rules.daily_loss_limit_pct else None
                rep.facts.update({"risk_at_stop": dec.risk_at_stop, "drawdown_room": dd_room, "daily_limit": dl})
                n_to_bust = dd_room / dec.risk_at_stop if dec.risk_at_stop else None
                rep.facts["consecutive_stops_to_bust"] = n_to_bust
                if n_to_bust is not None and n_to_bust < 3:
                    add("block", "drawdown_too_tight",
                        f"One stop-out costs ${dec.risk_at_stop:,.0f}; the drawdown room is ${dd_room:,.0f} -- "
                        f"only {n_to_bust:.1f} consecutive stops bust the account.")
                elif n_to_bust is not None and n_to_bust < 6:
                    add("warn", "drawdown_tight",
                        f"Only {n_to_bust:.1f} consecutive stop-outs reach the drawdown limit.")
                if dl is not None and dec.risk_at_stop > 0.6 * dl:
                    add("warn", "daily_limit_tight",
                        f"One stop-out (${dec.risk_at_stop:,.0f}) is {dec.risk_at_stop / dl:.0%} of the daily "
                        f"loss limit (${dl:,.0f}); two losses in a day lock or fail the account.")

    # -- sample floors -------------------------------------------------------
    try:
        n_bars = len(df)
        rep.facts["bars"] = n_bars
        import pandas as pd
        ts = pd.to_datetime(df["timestamp"])
        n_days = int(ts.dt.normalize().nunique())
        rep.facts["trading_days"] = n_days
        if n_days < MIN_TRADING_DAYS:
            add("warn", "short_history", f"Only {n_days} trading days of data (< {MIN_TRADING_DAYS}); pass probabilities will be noise.")
        if eval_days and n_days < 2 * eval_days:
            add("warn", "history_vs_eval",
                f"History covers {n_days} days but the evaluation window is {eval_days}: fewer than two independent attempts fit.")
    except Exception:  # noqa: BLE001
        pass
    if trades is not None:
        rep.facts["trades"] = len(trades)
        if len(trades) < MIN_TRADES:
            add("warn", "few_trades", f"Only {len(trades)} trades (< {MIN_TRADES}); statistics and Monte Carlo are not reliable.")
    return rep


def baseline_preflight(
    baseline_bt,
    df,
    risk,
    rules=None,
    symbol: str | None = None,
    stop_loss_pips: float | None = None,
    min_trades: int = 100,
    max_skip_ratio: float = 0.20,
    min_per_direction: int = 30,
) -> PreflightReport:
    """Preflight that also looks at the strategy's own baseline run: the
    plan's step-0 gate. Blocks (does not merely warn) on fewer than
    `min_trades` trades, more than `max_skip_ratio` of signals skipped for
    sizing, and a direction that never fires when the other one does."""
    rep = run_preflight(df, risk, rules, stop_loss_pips=stop_loss_pips, trades=baseline_bt.trades, symbol=symbol)
    add = lambda sev, code, msg: rep.issues.append(PreflightIssue(sev, code, msg))  # noqa: E731
    rep.issues = [i for i in rep.issues if i.code != "few_trades"]
    n = len(baseline_bt.trades)
    rep.facts["baseline_trades"] = n
    if n < min_trades:
        add("block", "trade_floor",
            f"This configuration produces {n} trades on the development data (minimum {min_trades}). "
            "Statistics, the Monte Carlo and any optimizer score on fewer trades are noise.")
    try:
        halt = baseline_bt.equity_curve.attrs.get("sizing_halt", {}) or {}
        ratio = float(halt.get("skip_ratio", 0.0))
        rep.facts["sizing_skip_ratio"] = ratio
        if ratio > max_skip_ratio:
            # Dollar terms (A1): say which lever fixes this, in dollars,
            # using only facts actually available -- never invented numbers.
            try:
                budget = float(risk.risk_amount(float(risk.initial_balance)))
            except Exception:  # noqa: BLE001
                budget = None
            if risk.risk_mode == "fixed":
                budget_txt = f"risk budget is ${budget:,.0f} per trade (fixed ${float(risk.risk_value):,.0f})" if budget is not None else "risk budget is a fixed dollar amount"
            else:
                budget_txt = (
                    f"risk budget is ${budget:,.0f} per trade ({float(risk.risk_value):g}% of ${float(risk.initial_balance):,.0f})"
                    if budget is not None else f"risk budget is {float(risk.risk_value):g}% per trade"
                )
            # One-contract risk at the strategy's stop, if computable.
            needed = None
            stop_dist = None
            for _key in ("stop_distance_price", "stop_distance", "one_contract_risk_at_stop", "risk_at_stop_dollars", "stop_dollars"):
                if halt.get(_key):
                    try:
                        _val = float(halt[_key])
                    except Exception:  # noqa: BLE001
                        continue
                    if _key in ("one_contract_risk_at_stop", "risk_at_stop_dollars", "stop_dollars"):
                        needed = _val
                    else:
                        stop_dist = _val
                    break
            if stop_dist is None and stop_loss_pips:
                try:
                    stop_dist = float(stop_loss_pips) * float(risk.pip_size)
                except Exception:  # noqa: BLE001
                    stop_dist = None
            if needed is None and stop_dist:
                try:
                    units = float(risk.contract_size) if risk.contract_size else 1.0
                    needed = float(risk.worst_case_loss(units, float(stop_dist)))
                except Exception:  # noqa: BLE001
                    needed = None
            if needed is None:
                for _trade in (getattr(baseline_bt, "trades", None) or []):
                    _rat = getattr(_trade, "risk_at_stop_dollars", None)
                    if _rat:
                        try:
                            needed = float(_rat)
                        except Exception:  # noqa: BLE001
                            needed = None
                        break
            if needed is not None:
                rep.facts["one_contract_risk_at_stop"] = needed
            # Micro-contract alternative, named when a mapping exists.
            micro_txt = "a micro contract"
            try:
                from app.data.instrument_specs import (
                    KNOWN_INSTRUMENTS,
                    guess_any_instrument_symbol,
                    micro_equivalent,
                )

                _sym = halt.get("symbol") or halt.get("instrument") or symbol
                _root = guess_any_instrument_symbol(str(_sym)) if _sym else None
                if _root is None and risk.contract_size:
                    for _spec in KNOWN_INSTRUMENTS.values():
                        if abs(float(_spec.contract_size) - float(risk.contract_size)) < 1e-9 and abs(float(_spec.pip_size) - float(risk.pip_size)) < 1e-12:
                            _root = _spec.symbol
                            break
                _micro = micro_equivalent(_root) if _root else None
                if _micro is not None:
                    _vs = f" vs ${float(risk.contract_size):g}/point here" if risk.contract_size else ""
                    micro_txt = (
                        f"the micro contract {_micro.symbol} (micro equivalent of {_root}, "
                        f"${_micro.contract_size:g}/point{_vs})"
                    )
                elif _root:
                    micro_txt = f"a micro contract, if one exists for {_root}"
            except Exception:  # noqa: BLE001
                micro_txt = "a micro contract (for example MES for ES, MNQ for NQ, or MGC for GC)"
            one_contract_txt = f" One contract at the strategy's stop risks about ${needed:,.0f}." if needed is not None else ""
            msg = (
                f"{ratio:.0%} of signals ({halt.get('skipped', 0)}) were skipped because the risk budget "
                f"cannot buy one contract at the strategy's stop (limit {max_skip_ratio:.0%}). The {budget_txt}.{one_contract_txt} "
                f"Fixes, in dollars: raise the risk budget per trade; switch to {micro_txt} with sizing_mode='micro_fallback'; "
                f"use sizing_mode='fit_stop' and consider lowering the don't-shrink-below lever "
                f"(fit_stop_min_fraction is currently {float(getattr(risk, 'fit_stop_min_fraction', 0.2)):g} -- a capped stop below that "
                "fraction of the strategy's stop is skipped instead of shrunk); tighten the strategy stop; or raise the risk budget. "
                f"Reasons: {halt.get('skip_reasons', {})}"
            )
            if needed is not None and budget is not None:
                msg += f" This would need ≈ ${needed:,.0f} per trade at the current stop, vs ${budget:,.0f} configured."
            add("block", "sizing_skips", msg)
    except Exception:  # noqa: BLE001
        pass
    longs = sum(1 for t in baseline_bt.trades if t.direction == 1)
    shorts = sum(1 for t in baseline_bt.trades if t.direction == -1)
    rep.facts.update({"long_trades": longs, "short_trades": shorts})
    if n >= min_trades and (longs == 0 or shorts == 0):
        side = "short" if shorts == 0 else "long"
        add("warn", "one_sided",
            f"No {side} trade fired in the whole sample ({longs} long / {shorts} short): that entry rule is never true "
            "on this data, so the strategy is effectively one-directional.")
    elif n >= min_trades and min(longs, shorts) < min_per_direction:
        add("warn", "thin_side", f"Only {min(longs, shorts)} trades on one side ({longs} long / {shorts} short).")
    return rep


def enforce_preflight(rep: PreflightReport, feature_name: str) -> None:
    """Raises RefinementError (the exception both pipelines already turn
    into a clean 'stopped' result) when the report has a blocking issue."""
    if not rep.blocked:
        return
    from app.optimize.parameter_space import RefinementError
    raise RefinementError(f"{feature_name} stopped at preflight, before any search:\n" + rep.render())


# ---------------------------------------------------------------------------
# Sizing remedies + the no-stop default (v9.10) ------------------------------
#
# Owen's discretionary reality check (2026-10-09): "I'm able to easily take
# a trade with 1 mini contract and risk less than $500. It's pretty
# flexible, just depending on where I put my stop." Correct -- risk is
# stop distance x $/point, and the stop is a CHOICE. The pipeline broke
# that twice over for stop-less strategies: the engine silently invented
# a 1%-of-price stop (~78 pts on ES = ~$3,900 on one mini, a stop no
# discretionary trader would place), then fit_stop refused to shrink
# below 20% of that invented distance, so preflight BLOCKed every run.
# The fix order below mirrors how a trader actually solves it: a sane,
# visible stop first; then mini if it fits, micro if it doesn't; BLOCK
# only when even one micro at that stop busts the budget.

SIZING_BLOCK_CODES = frozenset({"sizing_skips", "sizing_infeasible"})

# fit_stop auto-remedy hard floor: a capped stop is never shrunk below
# this fraction of the strategy's own stop. Below that it is a different
# strategy, and the run blocks instead (the BLOCK message says so).
FIT_STOP_HARD_FLOOR = 0.10

# The no-stop default: stop = 2 x ATR(14). On ES 5m that is ~6-12 pts --
# the $300-600 zone on one mini at a $500 budget, i.e. where a
# discretionary trader actually risks. Documented here, printed in the
# run log (points and dollars) by the caller, and recorded on the result.
DEFAULT_NO_STOP_ATR_MULTIPLE = 2.0
DEFAULT_NO_STOP_ATR_PERIOD = 14


def strategy_defines_stop(strategy) -> bool:
    """True when the strategy carries its own stop, so the pipeline must
    not substitute one. Manual (JSON) strategies define one via a
    risk_management stop_type of fixed/atr with a value, a legacy
    top-level stop_loss_pips, or a zone_entry block (zones carry their
    own stop geometry). Code strategies are assumed to manage their own
    stops (their signals carry the distances)."""
    if getattr(strategy, "source_type", None) != "manual":
        return True
    cfg = getattr(strategy, "config", None) or {}
    if cfg.get("zone_entry"):
        return True
    rm = cfg.get("risk_management") or {}
    if str(rm.get("stop_type", "")).lower() in ("fixed", "atr") and rm.get("stop_value") not in (None, ""):
        return True
    return cfg.get("stop_loss_pips") not in (None, "")


def apply_no_stop_default(strategy, df):
    """(strategy, note | None). For a stop-less MANUAL strategy: a copy
    whose risk_management carries the ATR default above, plus a note
    describing the rule and the median stop (points) it produces on
    `df`. Anything else comes back unchanged with note=None. The
    caller's strategy object is never mutated."""
    if strategy_defines_stop(strategy):
        return strategy, None
    from app.strategy.manual import ManualStrategy

    new_cfg = copy.deepcopy(getattr(strategy, "config", {}) or {})
    rm = dict(new_cfg.get("risk_management") or {})
    rm.update({
        "stop_type": "atr",
        "stop_value": DEFAULT_NO_STOP_ATR_MULTIPLE,
        "stop_atr_period": DEFAULT_NO_STOP_ATR_PERIOD,
    })
    new_cfg["risk_management"] = rm
    note = {
        "rule": "atr",
        "multiple": DEFAULT_NO_STOP_ATR_MULTIPLE,
        "period": DEFAULT_NO_STOP_ATR_PERIOD,
        "median_stop_points": None,
        "placeholder_stop_points": None,
    }
    try:
        from app.strategy.indicators import build_indicator_series

        atr = build_indicator_series(df, "atr", period=DEFAULT_NO_STOP_ATR_PERIOD, column="close")
        med = float(atr.median())
        if med == med and med > 0:  # not NaN
            note["median_stop_points"] = DEFAULT_NO_STOP_ATR_MULTIPLE * med
        note["placeholder_stop_points"] = 0.01 * float(df["close"].median())
    except Exception:  # noqa: BLE001 -- the note explains; it must never block a run
        pass
    return ManualStrategy(new_cfg), note


def sizing_remedy_candidates(risk, symbol=None):
    """Ordered (kind, detail, remedied RiskConfig) fallbacks for a
    sizing-blocked run, cheapest-to-the-strategy first:

    (a) micro_fallback -- the SAME stop distance on the micro contract's
        $/point (the remedy the sizing_skips message itself names);
    (b) fit_stop with the don't-shrink-below floor lowered to
        FIT_STOP_HARD_FLOOR -- only when fit_stop is the run's mode, and
        never below that hard floor.

    Neither changes the strategy; (a) never changes the stop at all. The
    caller re-runs and re-checks preflight per candidate and logs any
    adoption. If every candidate still blocks, the original BLOCK
    stands (remedy (c): even one micro at the stop busts the budget)."""
    from dataclasses import replace as _dc_replace

    out = []
    mode = getattr(risk, "sizing_mode", "skip")
    if mode == "rr_planned":
        # Owen's model already IS the mini -> micro -> skip ladder run
        # against planned risk; what remains (budget, target) is the
        # user's call, so there is nothing honest to auto-adopt.
        return out
    if mode != "micro_fallback":
        micro = None
        try:
            from app.data.instrument_specs import guess_any_instrument_symbol, micro_equivalent

            root = guess_any_instrument_symbol(str(symbol)) if symbol else None
            micro = micro_equivalent(root) if root else None
        except Exception:  # noqa: BLE001
            micro = None
        if micro is not None:
            upd = {"sizing_mode": "micro_fallback"}
            if not getattr(risk, "micro_contract_size", None):
                upd["micro_contract_size"] = float(micro.contract_size)
            if getattr(risk, "micro_commission_per_contract", None) is None:
                upd["micro_commission_per_contract"] = float(micro.default_commission_round_turn)
            out.append(("micro_fallback",
                        {"micro_symbol": micro.symbol,
                         "micro_contract_size": float(micro.contract_size)},
                        _dc_replace(risk, **upd)))
    if mode == "fit_stop" and float(getattr(risk, "fit_stop_min_fraction", 0.2) or 0.2) > FIT_STOP_HARD_FLOOR:
        out.append(("fit_stop_floor",
                    {"fit_stop_min_fraction": FIT_STOP_HARD_FLOOR},
                    _dc_replace(risk, fit_stop_min_fraction=FIT_STOP_HARD_FLOOR)))
    return out


_PRESET_COMPARE_FIELDS = (
    "account_size", "evaluation_profit_target_pct", "daily_loss_limit_pct", "max_drawdown_pct",
    "drawdown_type", "consistency_rule_pct", "min_trading_days", "dd_basis", "daily_loss_action",
    "trailing_lock", "max_contracts",
)


def compare_rules_to_preset(rules, preset_key: str) -> list[PreflightIssue]:
    """Differences between the PropRules a run is configured with and the
    named firm preset (e.g. 'lucid_50k'). One 'warn' issue per differing
    field: a config that silently disagrees with the firm is the commonest
    way a pass probability stops describing a real account."""
    from app.prop.presets import get_preset
    preset = get_preset(preset_key)
    if preset is None:
        return [PreflightIssue("warn", "unknown_preset", f"No preset named {preset_key!r}.")]
    want = preset.to_prop_rules()
    out = []
    for f in _PRESET_COMPARE_FIELDS:
        a, b = getattr(rules, f, None), getattr(want, f, None)
        if a != b:
            out.append(PreflightIssue(
                "warn", "preset_mismatch",
                f"{f}: config has {a!r} but {preset.label} uses {b!r}"
                + (f" (checked {preset.rules_checked_on})" if preset.rules_checked_on else " (not re-verified)") + "."))
    return out
