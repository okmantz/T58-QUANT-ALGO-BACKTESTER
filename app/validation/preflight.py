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
            add("block", "sizing_skips",
                f"{ratio:.0%} of signals ({halt.get('skipped', 0)}) were skipped because the risk budget "
                f"cannot buy one contract at the strategy's stop (limit {max_skip_ratio:.0%}). Use "
                "sizing_mode='fit_stop' or 'micro_fallback', a micro contract, or a tighter stop. "
                f"Reasons: {halt.get('skip_reasons', {})}")
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
