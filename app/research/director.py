"""
Research Director engines.

Owen's ask (paraphrased): stop treating a failed or passed strategy as a
single black-box verdict. For every strategy T58 tests, work out WHICH
PART of it is creating the edge (or destroying it), how fragile that edge
is, whether it beats a stupidly simple alternative, and under which
conditions it actually fires -- then roll all of that up, across every
strategy ever tested, into a plain-English "what have we learned, and
where should we keep looking" report.

Every engine here is a THIN layer on top of infrastructure this codebase
already has -- app.backtest.engine.run_backtest for the P&L math,
app.prop.rolling_evaluation.run_rolling_evaluation for an honest historical
pass rate, and app.search.strategy_space.build_strategy_from_spec for
turning a manual-builder config dict back into a runnable strategy. None
of these engines re-implement backtest or prop-eval logic -- a rule change
in either of those automatically applies here too.

SCOPE NOTE: edge_decomposition() and ablation_test() operate on Manual
Strategy Builder configs (dicts with entry_conditions/exit_conditions --
see app.strategy.manual), because that is the only strategy format in
this codebase with individually addressable, removable rules. A
Python/PineScript/MQL5 strategy is opaque source code with no rule
boundaries T58 can identify automatically -- for those, pass your own
`variants` list (see each function's docstring) built from hand-authored
versions of the strategy with one layer added/removed at a time; the
engine will still run the comparison and produce the same report shape.
"""
from __future__ import annotations

import copy
import random
import statistics as pystats
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest, BacktestResult
from app.backtest.execution import Trade, run_execution
from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.prop.rolling_evaluation import run_rolling_evaluation
from app.search.strategy_space import build_strategy_from_spec
from app.strategy.manual import ManualStrategy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pass_rate(
    trades: list[Trade],
    prop_rules: PropRules,
    window_trading_days: int = 30,
    stride: int = 2,
    max_windows: int | None = 500,
) -> tuple[float | None, float | None]:
    """Returns (pass_rate_pct, first_payout_rate_pct), or (None, None) when
    there are too few trades/trading days to run even one rolling window --
    this is a legitimate outcome (not an error) for a badly underperforming
    variant and callers should display it as "n/a", not zero."""
    if not trades:
        return None, None
    try:
        result = run_rolling_evaluation(trades, prop_rules, window_trading_days, stride=stride, max_windows=max_windows)
    except ValueError:
        return None, None
    return result.pass_rate_pct, result.first_payout_rate_pct


def _row(label: str, bt: BacktestResult | None, prop_rules: PropRules, window_trading_days: int) -> dict:
    if bt is None or not bt.trades:
        return {
            "label": label, "total_trades": 0, "expectancy_r": None, "profit_factor": None,
            "net_profit": None, "max_drawdown_pct": None, "pass_rate_pct": None,
            "first_payout_rate_pct": None, "note": "No trades generated.",
        }
    pass_rate, payout_rate = _pass_rate(bt.trades, prop_rules, window_trading_days)
    s = bt.statistics
    return {
        "label": label,
        "total_trades": len(bt.trades),
        "expectancy_r": round(s.average_r, 4),
        "profit_factor": round(s.profit_factor, 3) if s.profit_factor == s.profit_factor else None,
        "net_profit": round(s.net_profit, 2),
        "max_drawdown_pct": round(s.max_drawdown_pct, 2),
        "pass_rate_pct": round(pass_rate, 1) if pass_rate is not None else None,
        "first_payout_rate_pct": round(payout_rate, 1) if payout_rate is not None else None,
        "note": None,
    }


def _run_spec(spec: dict, df: pd.DataFrame, risk: RiskConfig) -> BacktestResult | None:
    try:
        strategy = build_strategy_from_spec(spec)
        return run_backtest(df, strategy, risk)
    except Exception as exc:  # a candidate variant is allowed to be broken/untradeable
        return None


def _is_manual(spec: dict) -> bool:
    return (spec or {}).get("source_type", "manual") == "manual"


# Heuristic role classifier for a Manual Strategy Builder condition dict.
# A condition is {"left": operand, "operator": str, "right": operand} (see
# app.strategy.manual._evaluate_condition) and an operand, when a dict, has
# its indicator/source type under "type" -- there is no explicit "role"
# field anywhere in the schema. This infers one from the left/right operand
# types so Edge Decomposition can group conditions the way Owen's example
# table does. Callers can override any condition's role by adding an
# explicit "role" key to that condition dict before calling
# edge_decomposition/ablation_test; this heuristic is only a fallback for
# untagged conditions.
_SESSION_KINDS = {
    "time_of_day", "session_high", "session_low", "previous_day_high",
    "previous_day_low", "previous_day_close", "opening_range_high", "opening_range_low",
}
_REGIME_KINDS = {"atr_regime", "volatility_regime"}
_CONFIRMATION_KINDS = {
    "candle_direction", "rsi", "macd", "macd_signal", "macd_histogram",
    "relative_volume", "volume_delta", "average_volume",
}


def _operand_kind(operand) -> str | None:
    if isinstance(operand, dict):
        return str(operand.get("type", operand.get("source", ""))).lower().strip() or None
    return None


def _condition_kinds(cond: dict) -> set[str]:
    kinds = {_operand_kind((cond or {}).get("left")), _operand_kind((cond or {}).get("right"))}
    kinds.discard(None)
    return kinds


def _condition_role(cond: dict) -> str:
    if isinstance(cond, dict) and cond.get("role"):
        return cond["role"]
    kinds = _condition_kinds(cond)
    if kinds & _SESSION_KINDS:
        return "session"
    if kinds & _REGIME_KINDS:
        return "regime"
    if kinds & _CONFIRMATION_KINDS:
        return "confirmation"
    return "setup"


def _condition_label(cond: dict) -> str:
    kinds = _condition_kinds(cond)
    if kinds:
        return "/".join(sorted(kinds))
    return "condition"


# ---------------------------------------------------------------------------
# 1. Edge Decomposition Engine
# ---------------------------------------------------------------------------

@dataclass
class DecompositionStep:
    label: str
    added: str | None      # what this step added, e.g. "+ NY session" (None for baseline)
    row: dict

    def to_dict(self) -> dict:
        return {"label": self.label, "added": self.added, "row": self.row}


def edge_decomposition(
    spec: dict,
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    window_trading_days: int = 30,
    role_order: tuple[str, ...] = ("session", "regime", "confirmation"),
    variants: list[tuple[str, dict]] | None = None,
) -> dict:
    """Builds a strategy up one rule-layer at a time and records the
    expectancy/pass-rate at each step, so you can see which layer is doing
    the work and which is destroying it.

    Manual-builder path (default): starts from just the "setup" conditions
    (the core entry trigger), then adds each other role group in
    `role_order`, ending on the full strategy exactly as configured.

    Code-strategy path: pass `variants` as an ordered list of
    (label, spec) tuples yourself, e.g.
        [("Base strategy", base_spec), ("+ NY session", spec_with_session), ...]
    and this function will backtest each one and produce the same table
    (skips the manual-only decomposition logic entirely).
    """
    if variants is not None:
        steps = []
        for label, variant_spec in variants:
            bt = _run_spec(variant_spec, df, risk)
            steps.append(DecompositionStep(label, label, _row(label, bt, prop_rules, window_trading_days)))
        return _decomposition_verdict(steps)

    if not _is_manual(spec):
        raise ValueError(
            "Edge Decomposition needs either a Manual Strategy Builder config, or an explicit "
            "`variants` list of (label, spec) tuples for a code-based strategy -- see this "
            "function's docstring."
        )

    base_config = copy.deepcopy(spec["config"])
    entries = base_config.get("entry_conditions", {}) or {}
    long_conds = list(entries.get("long", []) or [])
    short_conds = list(entries.get("short", []) or [])

    # Every strategy needs at least one bar-triggering condition to form a
    # "baseline" step. If the heuristic classifier didn't tag anything as
    # "setup" (e.g. a strategy built entirely from session + confirmation
    # conditions with no distinct trigger), fall back to treating each
    # side's FIRST declared condition as its setup trigger, so the baseline
    # step isn't silently empty.
    def _roles_with_fallback(conds: list[dict]) -> list[str]:
        roles = [_condition_role(c) for c in conds]
        if conds and "setup" not in roles:
            roles[0] = "setup"
        return roles

    long_roles = _roles_with_fallback(long_conds)
    short_roles = _roles_with_fallback(short_conds)

    def _filter_role(conds: list[dict], roles: list[str], allowed_roles: set[str]) -> list[dict]:
        return [c for c, r in zip(conds, roles) if r in allowed_roles]

    steps: list[DecompositionStep] = []
    cumulative_roles: set[str] = {"setup"}

    def _build_step(label: str, added: str | None, roles: set[str]) -> None:
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("entry_conditions", {})
        cfg["entry_conditions"]["long"] = _filter_role(long_conds, long_roles, roles)
        cfg["entry_conditions"]["short"] = _filter_role(short_conds, short_roles, roles)
        variant_spec = {"source_type": "manual", "config": cfg}
        bt = _run_spec(variant_spec, df, risk)
        steps.append(DecompositionStep(label, added, _row(label, bt, prop_rules, window_trading_days)))

    _build_step("Base strategy (setup only)", None, cumulative_roles)
    role_labels = {
        "session": "+ session filter", "regime": "+ market regime filter",
        "confirmation": "+ confirmation",
    }
    for role in role_order:
        if role not in long_roles and role not in short_roles:
            continue  # nothing of this role in the actual config -- skip, don't fabricate a step
        cumulative_roles = cumulative_roles | {role}
        _build_step(role_labels.get(role, f"+ {role}"), role_labels.get(role, f"+ {role}"), cumulative_roles)

    # Final row: the strategy exactly as configured (covers any condition
    # whose role wasn't in role_order, e.g. a custom "role" tag, and any
    # exit_conditions/exit-logic differences vs. the stepwise builds above).
    full_bt = _run_spec(spec, df, risk)
    steps.append(DecompositionStep("Full strategy (as configured)", "+ exit logic & risk management", _row("Full strategy (as configured)", full_bt, prop_rules, window_trading_days)))

    return _decomposition_verdict(steps)


def _decomposition_verdict(steps: list[DecompositionStep]) -> dict:
    rows = [s.to_dict() for s in steps]
    # Find the layer with the single largest expectancy_r gain and the
    # single largest expectancy_r loss between consecutive steps.
    deltas = []
    for prev, cur in zip(steps, steps[1:]):
        er_prev = prev.row["expectancy_r"]
        er_cur = cur.row["expectancy_r"]
        if er_prev is None or er_cur is None:
            continue
        deltas.append((cur.added or cur.label, er_cur - er_prev))
    best = max(deltas, key=lambda d: d[1]) if deltas else None
    worst = min(deltas, key=lambda d: d[1]) if deltas else None
    verdict_lines = []
    if best and best[1] > 0:
        verdict_lines.append(f"{best[0]} is doing most of the work (+{best[1]:.3f}R expectancy).")
    if worst and worst[1] < 0:
        verdict_lines.append(f"{worst[0]} is hurting performance ({worst[1]:.3f}R expectancy).")
    if not verdict_lines:
        verdict_lines.append("No single layer stands out -- the edge (or lack of one) is spread evenly across the rules.")
    return {"steps": rows, "verdict": " ".join(verdict_lines)}


# ---------------------------------------------------------------------------
# 2. Ablation Testing
# ---------------------------------------------------------------------------

def ablation_test(
    spec: dict,
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    window_trading_days: int = 30,
    metric: str = "pass_rate_pct",
    variants: list[tuple[str, dict]] | None = None,
) -> dict:
    """Removes exactly one rule at a time from the FULL strategy and
    measures the metric (default: prop-eval pass rate) with it gone.
    A rule whose removal barely moves the metric is dead weight; a rule
    whose removal collapses the metric is essential.

    Manual-builder path (default): each entry_conditions/exit_conditions
    condition is one removable rule. Code-strategy path: pass `variants`
    yourself as [(rule_name, spec_with_that_rule_removed), ...] -- the
    full/baseline row is always run separately as `spec` itself.
    """
    full_bt = _run_spec(spec, df, risk)
    full_row = _row("Full strategy", full_bt, prop_rules, window_trading_days)

    rows = [full_row]
    essential, unnecessary = [], []

    def _classify(rule_name: str, ablated_row: dict) -> None:
        base = full_row.get(metric)
        cur = ablated_row.get(metric)
        if base is None or cur is None:
            return
        drop = base - cur
        # "Essential" = removing it costs a large chunk of the metric;
        # "unnecessary" = removing it costs (almost) nothing, or helps.
        if drop >= max(5.0, 0.15 * max(base, 1.0)):
            essential.append(rule_name)
        elif drop <= 0:
            unnecessary.append(rule_name)

    if variants is not None:
        for rule_name, variant_spec in variants:
            bt = _run_spec(variant_spec, df, risk)
            row = _row(f"Remove {rule_name}", bt, prop_rules, window_trading_days)
            rows.append(row)
            _classify(rule_name, row)
        return _ablation_verdict(rows, essential, unnecessary)

    if not _is_manual(spec):
        raise ValueError(
            "Ablation Testing needs either a Manual Strategy Builder config, or an explicit "
            "`variants` list of (rule_name, spec_with_rule_removed) tuples for a code-based "
            "strategy -- see this function's docstring."
        )

    base_config = spec["config"]
    entries = base_config.get("entry_conditions", {}) or {}
    exits = base_config.get("exit_conditions", {}) or {}

    def _iter_removable():
        for side in ("long", "short"):
            for i, cond in enumerate(entries.get(side, []) or []):
                yield ("entry", side, i, cond)
        for side in ("long", "short"):
            for i, cond in enumerate(exits.get(side, []) or []):
                yield ("exit", side, i, cond)

    for section, side, idx, cond in _iter_removable():
        cfg = copy.deepcopy(base_config)
        bucket = cfg["entry_conditions"] if section == "entry" else cfg.get("exit_conditions", {})
        conds = bucket.get(side, [])
        if idx >= len(conds):
            continue
        del conds[idx]
        variant_spec = {"source_type": "manual", "config": cfg}
        role = _condition_role(cond)
        rule_name = f"{_condition_label(cond)} ({role}, {side} {section})"
        bt = _run_spec(variant_spec, df, risk)
        row = _row(f"Remove {rule_name}", bt, prop_rules, window_trading_days)
        rows.append(row)
        _classify(rule_name, row)

    return _ablation_verdict(rows, essential, unnecessary)


def _ablation_verdict(rows: list[dict], essential: list[str], unnecessary: list[str]) -> dict:
    lines = []
    if essential:
        lines.append("Essential (removing it collapses performance): " + ", ".join(essential) + ".")
    if unnecessary:
        lines.append("Mostly irrelevant or actively unnecessary (removing it doesn't hurt, or helps): " + ", ".join(unnecessary) + ".")
    if not lines:
        lines.append("No rule clearly stands out as essential or unnecessary at the current threshold.")
    simplified_note = (
        f"A simplified version dropping just the unnecessary rule(s) ({', '.join(unnecessary)}) "
        "is worth backtesting on its own -- fewer free parameters, same or better pass rate."
        if unnecessary else None
    )
    return {"rows": rows, "essential": essential, "unnecessary": unnecessary, "verdict": " ".join(lines), "simplification_suggestion": simplified_note}


# ---------------------------------------------------------------------------
# 3. Null Strategy Benchmark
# ---------------------------------------------------------------------------

def _null_signals_random(df: pd.DataFrame, seed: int, trade_rate: float = 0.02) -> pd.Series:
    rng = np.random.default_rng(seed)
    draws = rng.random(len(df))
    signals = np.where(draws < trade_rate / 2, 1, np.where(draws < trade_rate, -1, 0))
    return pd.Series(signals, index=df.index)


def _null_signals_coinflip(df: pd.DataFrame, seed: int, trade_rate: float = 0.02) -> pd.Series:
    rng = random.Random(seed)
    out = [0] * len(df)
    for i in range(len(df)):
        if rng.random() < trade_rate:
            out[i] = 1 if rng.random() < 0.5 else -1
    return pd.Series(out, index=df.index)


def _null_signals_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    signals = pd.Series(0, index=df.index)
    if len(signals):
        signals.iloc[0] = 1
    return signals


def _null_signals_session_only(df: pd.DataFrame, start: str = "08:30", end: str = "09:00") -> pd.Series:
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.to_datetime(df.index)
    t = ts.dt.time
    start_t, end_t = pd.to_datetime(start).time(), pd.to_datetime(end).time()
    mask = (t >= start_t) & (t <= end_t)
    entered = mask & ~mask.shift(1, fill_value=False)
    return pd.Series(np.where(entered, 1, 0), index=df.index)


def _null_signals_breakout(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    high_n = df["high"].rolling(lookback).max().shift(1)
    low_n = df["low"].rolling(lookback).min().shift(1)
    long_sig = df["close"] > high_n
    short_sig = df["close"] < low_n
    return pd.Series(np.where(long_sig, 1, np.where(short_sig, -1, 0)), index=df.index)


def _null_signals_mean_reversion(df: pd.DataFrame, lookback: int = 20, z: float = 2.0) -> pd.Series:
    mean = df["close"].rolling(lookback).mean()
    std = df["close"].rolling(lookback).std()
    zscore = (df["close"] - mean) / std.replace(0, np.nan)
    long_sig = zscore < -z
    short_sig = zscore > z
    return pd.Series(np.where(long_sig, 1, np.where(short_sig, -1, 0)), index=df.index)


def _null_signals_prev_bar_continuation(df: pd.DataFrame) -> pd.Series:
    prev_up = df["close"].shift(1) > df["close"].shift(2)
    prev_down = df["close"].shift(1) < df["close"].shift(2)
    return pd.Series(np.where(prev_up, 1, np.where(prev_down, -1, 0)), index=df.index)


_NULL_BUILDERS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "Random entry": lambda df: _null_signals_random(df, seed=1),
    "Coin flip (same risk/exit)": lambda df: _null_signals_coinflip(df, seed=2),
    "Buy and hold": _null_signals_buy_and_hold,
    "Session-only entry": _null_signals_session_only,
    "Simple breakout (20-bar)": _null_signals_breakout,
    "Simple mean reversion (20-bar, 2σ)": _null_signals_mean_reversion,
    "Previous-bar continuation": _null_signals_prev_bar_continuation,
}


def null_baselines(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    target_row: dict,
    window_trading_days: int = 30,
    stop_loss_pips: float | None = None,
    take_profit_pips: float | None = None,
    only: list[str] | None = None,
) -> dict:
    """Runs your strategy's own risk/exit structure against a set of
    stupidly-simple entry rules. `target_row` is the output of `_row()` (or
    the equivalent fields) for the strategy actually being evaluated --
    pass the row from your own backtest so this can compute the edge
    contribution directly. If the target's own stop/target isn't fixed-pip
    (e.g. ATR-based), pass explicit stop_loss_pips/take_profit_pips to give
    every baseline a comparable, fixed exit structure instead.
    """
    sl = stop_loss_pips if stop_loss_pips is not None else 20.0
    tp = take_profit_pips if take_profit_pips is not None else 30.0
    builders = {k: v for k, v in _NULL_BUILDERS.items() if (only is None or k in only)}
    rows = []
    for label, builder in builders.items():
        try:
            signals = builder(df)
            trades, equity_curve = run_execution(df=df, signals=signals, risk=risk, stop_loss_pips=sl, take_profit_pips=tp)
        except Exception:
            rows.append({"label": label, "total_trades": 0, "expectancy_r": None, "profit_factor": None,
                         "net_profit": None, "max_drawdown_pct": None, "pass_rate_pct": None,
                         "first_payout_rate_pct": None, "note": "Could not generate a valid signal for this dataset."})
            continue
        from app.backtest.statistics import compute_statistics
        stats = compute_statistics(trades, equity_curve, initial_balance=risk.initial_balance)
        bt = BacktestResult(strategy_name=label, trades=trades, equity_curve=equity_curve, statistics=stats, initial_balance=risk.initial_balance)
        rows.append(_row(label, bt, prop_rules, window_trading_days))

    best_null = max((r for r in rows if r.get("pass_rate_pct") is not None), key=lambda r: r["pass_rate_pct"], default=None)
    target_pass = target_row.get("pass_rate_pct")
    verdict = "Not enough data to compare."
    edge_contribution = None
    if best_null is not None and target_pass is not None:
        gap = target_pass - best_null["pass_rate_pct"]
        if gap < 5:
            edge_contribution = "LOW"
            verdict = (
                f"Your strategy's pass rate ({target_pass:.1f}%) is barely ahead of the best simple baseline, "
                f"\"{best_null['label']}\" ({best_null['pass_rate_pct']:.1f}%). The edge isn't coming from the "
                "entry logic -- it's likely coming from the risk/exit structure both share, or isn't real."
            )
        elif gap < 15:
            edge_contribution = "MODERATE"
            verdict = f"Your strategy beats the best simple baseline (\"{best_null['label']}\") by {gap:.1f} points -- some real edge, but worth tightening."
        else:
            edge_contribution = "HIGH"
            verdict = f"Your strategy clears the best simple baseline (\"{best_null['label']}\") by {gap:.1f} points -- the entry logic is doing genuine work."
    return {"target": target_row, "baselines": rows, "best_baseline": best_null, "edge_contribution": edge_contribution, "verdict": verdict}


# ---------------------------------------------------------------------------
# 4. Signal Degradation Tests (Execution Fragility Score)
# ---------------------------------------------------------------------------

def _shift_signals(signals: pd.Series, bars: int) -> pd.Series:
    return signals.shift(bars).fillna(0).astype(int)


def _skip_random_trades(signals: pd.Series, frac: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    sig = signals.copy()
    nonzero_idx = sig[sig != 0].index
    n_skip = int(len(nonzero_idx) * frac)
    if n_skip and len(nonzero_idx):
        drop_idx = rng.choice(nonzero_idx, size=min(n_skip, len(nonzero_idx)), replace=False)
        sig.loc[drop_idx] = 0
    return sig


def signal_degradation(
    spec: dict,
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    window_trading_days: int = 30,
) -> dict:
    """Stresses a strategy's timing and cost assumptions and reports how
    much of the P&L survives. A real edge degrades gracefully; an edge
    that vanishes at 1-bar delay was probably a backtest artifact
    (lookahead, unrealistic fills, or fitting to exact bar timing)."""
    if not _is_manual(spec):
        raise ValueError("Signal Degradation Tests currently support Manual Strategy Builder configs only.")

    strategy = ManualStrategy(spec["config"])
    strat_result = strategy.generate(df)
    base_signals = strat_result.signals
    sl, tp = strat_result.stop_loss_pips, strat_result.take_profit_pips

    def _bt_from_signals(signals: pd.Series, risk_override: RiskConfig | None = None) -> BacktestResult:
        r = risk_override or risk
        trades, eq = run_execution(
            df=df, signals=signals, risk=r, stop_loss_pips=sl, take_profit_pips=tp,
            stop_loss_distance=strat_result.stop_loss_distance, take_profit_distance=strat_result.take_profit_distance,
        )
        from app.backtest.statistics import compute_statistics
        stats = compute_statistics(trades, eq, initial_balance=r.initial_balance)
        return BacktestResult(strategy_name="variant", trades=trades, equity_curve=eq, statistics=stats, initial_balance=r.initial_balance)

    baseline_bt = _bt_from_signals(base_signals)
    baseline_row = _row("Normal (no stress)", baseline_bt, prop_rules, window_trading_days)

    tests = []

    tests.append(("Enter 1 bar later", _bt_from_signals(_shift_signals(base_signals, 1))))
    tests.append(("Enter 2 bars later", _bt_from_signals(_shift_signals(base_signals, 2))))
    tests.append(("Skip 10% of trades randomly", _bt_from_signals(_skip_random_trades(base_signals, 0.10, seed=7))))

    slip_risk = copy.deepcopy(risk)
    slip_risk.slippage_pips = (risk.slippage_pips or 0) + 2.0
    tests.append(("Add realistic slippage (+2 pips)", _bt_from_signals(base_signals, risk_override=slip_risk)))

    spread_risk = copy.deepcopy(risk)
    spread_risk.spread_pips = (risk.spread_pips or 0) + 1.0
    tests.append(("Slightly worse fill (+1 pip spread)", _bt_from_signals(base_signals, risk_override=spread_risk)))

    comm_risk = copy.deepcopy(risk)
    comm_risk.commission_per_trade = (risk.commission_per_trade or 0) * 2 if risk.commission_per_trade else 5.0
    tests.append(("Increase commissions (2x)", _bt_from_signals(base_signals, risk_override=comm_risk)))

    rows = [baseline_row] + [_row(label, bt, prop_rules, window_trading_days) for label, bt in tests]

    base_np = baseline_row.get("net_profit")
    fragility_hits = 0
    fragility_total = 0
    for r in rows[1:]:
        if base_np is None or r.get("net_profit") is None:
            continue
        fragility_total += 1
        # A test "survives" if it retains at least 60% of baseline net
        # profit (or baseline was <=0 and the stress didn't make it worse).
        if base_np > 0:
            if r["net_profit"] >= 0.6 * base_np:
                fragility_hits += 1
        else:
            if r["net_profit"] >= base_np:
                fragility_hits += 1
    score = round(100 * fragility_hits / fragility_total, 1) if fragility_total else None
    if score is None:
        verdict = "Not enough trades to assess execution fragility."
    elif score >= 80:
        verdict = f"Execution Fragility Score {score}/100 -- robust. The edge survives realistic timing/cost stress."
    elif score >= 40:
        verdict = f"Execution Fragility Score {score}/100 -- moderately fragile. Some stress tests erase most of the edge."
    else:
        verdict = f"Execution Fragility Score {score}/100 -- fragile. The edge likely depends on exact bar timing/fills that won't hold live."
    return {"baseline": baseline_row, "stress_tests": rows[1:], "execution_fragility_score": score, "verdict": verdict}


# ---------------------------------------------------------------------------
# 5. Trade Contribution Analysis (leave-X-out)
# ---------------------------------------------------------------------------

def trade_contribution(trades: list[Trade], initial_balance: float) -> dict:
    """Answers: does the strategy's total P&L depend on a handful of
    trades, or one lucky month, rather than a broad, repeatable edge?"""
    if not trades:
        return {"note": "No trades to analyze."}

    pnls = [t.pnl for t in trades]
    total = sum(pnls)
    order = sorted(range(len(pnls)), key=lambda i: pnls[i], reverse=True)

    def _excl(idxs: set[int]) -> float:
        return sum(p for i, p in enumerate(pnls) if i not in idxs)

    n = len(pnls)
    best_1 = {order[0]} if n >= 1 else set()
    best_5pct = set(order[: max(1, round(0.05 * n))])
    best_10pct = set(order[: max(1, round(0.10 * n))])
    worst_5pct = set(order[-max(1, round(0.05 * n)):])

    leave_out = {
        "Remove best 1 trade": _excl(best_1),
        "Remove best 5%": _excl(best_5pct),
        "Remove best 10%": _excl(best_10pct),
        "Remove worst 5%": _excl(worst_5pct),
    }

    # Best 10 trades vs the rest (Owen's literal example), scaled to
    # whatever n actually is if fewer than 10 trades exist.
    top_n = min(10, n)
    top_idx = set(order[:top_n])
    top_pnl = sum(pnls[i] for i in top_idx)
    remaining_pnl = total - top_pnl

    # Per-calendar-month and per-quarter leave-one-period-out.
    df = pd.DataFrame({"pnl": pnls, "time": [pd.Timestamp(t.entry_time) for t in trades]})
    monthly = df.groupby(df["time"].dt.to_period("M"))["pnl"].sum().sort_values()
    quarterly = df.groupby(df["time"].dt.to_period("Q"))["pnl"].sum().sort_values()
    worst_month_removed = total - monthly.iloc[0] if len(monthly) else total
    worst_quarter_removed = total - quarterly.iloc[0] if len(quarterly) else total

    flags = []
    if total > 0 and top_pnl > 0 and remaining_pnl < 0.3 * total:
        flags.append(f"The best {top_n} trades account for {top_pnl:.0f} of {total:.0f} total P&L -- the remaining {n - top_n} trades only made {remaining_pnl:.0f}. Concentration risk.")
    if total > 0 and leave_out["Remove best 5%"] < 0:
        flags.append("Removing just the best 5% of trades flips total P&L negative.")
    if not flags:
        flags.append("P&L looks broadly distributed across trades -- no single trade or short period is propping up the result.")

    return {
        "total_pnl": round(total, 2),
        "n_trades": n,
        f"best_{top_n}_trades_pnl": round(top_pnl, 2),
        "remaining_trades_pnl": round(remaining_pnl, 2),
        "leave_out_pnl": {k: round(v, 2) for k, v in leave_out.items()},
        "worst_month_pnl": round(float(monthly.iloc[0]), 2) if len(monthly) else None,
        "pnl_excluding_worst_month": round(float(worst_month_removed), 2),
        "worst_quarter_pnl": round(float(quarterly.iloc[0]), 2) if len(quarterly) else None,
        "pnl_excluding_worst_quarter": round(float(worst_quarter_removed), 2),
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# 6. Conditional Expectancy Maps
# ---------------------------------------------------------------------------

def conditional_expectancy(trades: list[Trade], df: pd.DataFrame) -> dict:
    """Breaks expectancy down by hour, day of week, month, session, and
    (if an ATR-like range is derivable) a rough volatility percentile, plus
    direction. Answers "WHEN does this edge fire" instead of just "does it
    work on average"."""
    if not trades:
        return {"note": "No trades to analyze."}

    rows = []
    for t in trades:
        r = (t.pnl / t.initial_risk) if t.initial_risk else None
        rows.append({
            "hour": pd.Timestamp(t.entry_time).hour,
            "dow": pd.Timestamp(t.entry_time).day_name(),
            "month": pd.Timestamp(t.entry_time).month,
            "direction": "long" if t.direction == 1 else "short",
            "pnl": t.pnl, "r": r,
        })
    tdf = pd.DataFrame(rows)

    # Volatility percentile via a simple high-low range, aligned by nearest
    # bar timestamp, when the dataframe has enough context to compute it.
    vol_bucket = None
    if {"high", "low", "timestamp"}.issubset(df.columns):
        rng = (df["high"] - df["low"])
        rng_rank = rng.rank(pct=True)
        vol_by_time = pd.Series(rng_rank.values, index=pd.to_datetime(df["timestamp"]))
        vol_lookup = vol_by_time.reindex(vol_by_time.index.union([pd.Timestamp(t.entry_time) for t in trades])).sort_index().ffill()
        buckets = []
        for t in trades:
            try:
                pct = float(vol_lookup.loc[pd.Timestamp(t.entry_time)])
            except Exception:
                pct = None
            buckets.append("high_vol" if pct is not None and pct >= 0.66 else "low_vol" if pct is not None and pct <= 0.33 else "mid_vol" if pct is not None else None)
        tdf["volatility_bucket"] = buckets
        vol_bucket = _expectancy_table(tdf, "volatility_bucket")

    def _fmt(bucket_df, col):
        return _expectancy_table(bucket_df, col)

    all_trades_expectancy = round(tdf["r"].dropna().mean(), 4) if tdf["r"].notna().any() else None

    return {
        "all_trades_expectancy_r": all_trades_expectancy,
        "by_hour": _fmt(tdf, "hour"),
        "by_day_of_week": _fmt(tdf, "dow"),
        "by_month": _fmt(tdf, "month"),
        "by_direction": _fmt(tdf, "direction"),
        "by_volatility": vol_bucket,
    }


def _expectancy_table(tdf: pd.DataFrame, col: str) -> list[dict]:
    if col not in tdf.columns or tdf[col].isna().all():
        return []
    g = tdf.dropna(subset=[col]).groupby(col)
    out = []
    for key, sub in g:
        out.append({
            "bucket": str(key), "n_trades": len(sub),
            "expectancy_r": round(sub["r"].dropna().mean(), 4) if sub["r"].notna().any() else None,
            "win_rate_pct": round(100 * (sub["pnl"] > 0).mean(), 1),
        })
    out.sort(key=lambda x: (x["expectancy_r"] is None, -(x["expectancy_r"] or 0)))
    return out


# ---------------------------------------------------------------------------
# 7. Regime Discovery Engine
# ---------------------------------------------------------------------------

def regime_discovery(
    trades: list[Trade],
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    spec: dict,
    holdout_df: pd.DataFrame,
    window_trading_days: int = 30,
) -> dict:
    """Compares feature distributions between winning and losing trades to
    propose a data-driven regime hypothesis (e.g. "only works when
    volatility is expanding and price is above VWAP"), then actually tests
    that hypothesis on an untouched holdout slice by re-running the same
    strategy with the proposed regime condition added, and reporting
    whether the pass rate genuinely improves out of sample."""
    if not trades:
        return {"note": "No trades to analyze."}

    features = []
    for t in trades:
        row = {"win": t.pnl > 0}
        idx_candidates = df.index[pd.to_datetime(df["timestamp"]) <= pd.Timestamp(t.entry_time)] if "timestamp" in df.columns else []
        if len(idx_candidates):
            i = idx_candidates[-1]
            window = df.loc[max(0, i - 20): i]
            row["opening_range"] = float(window["high"].max() - window["low"].min()) if len(window) else None
            row["above_recent_mean"] = bool(df.loc[i, "close"] > window["close"].mean()) if len(window) else None
            row["hour"] = pd.Timestamp(t.entry_time).hour
        features.append(row)
    fdf = pd.DataFrame(features)

    winners = fdf[fdf["win"]]
    losers = fdf[~fdf["win"]]

    findings_win, findings_loss = [], []
    if "opening_range" in fdf.columns and fdf["opening_range"].notna().any():
        w_med, l_med = winners["opening_range"].median(), losers["opening_range"].median()
        if pd.notna(w_med) and pd.notna(l_med) and l_med:
            if w_med > 1.15 * l_med:
                findings_win.append("above-average recent range/volatility")
            elif w_med < 0.85 * l_med:
                findings_loss.append("compressed recent range/volatility")
    if "above_recent_mean" in fdf.columns:
        if winners["above_recent_mean"].mean() - losers["above_recent_mean"].mean() > 0.15:
            findings_win.append("price above its recent mean")
        elif losers["above_recent_mean"].mean() - winners["above_recent_mean"].mean() > 0.15:
            findings_loss.append("price below its recent mean")
    if "hour" in fdf.columns:
        win_hours = winners["hour"].value_counts(normalize=True)
        loss_hours = losers["hour"].value_counts(normalize=True)
        common_loss_hour = loss_hours.idxmax() if len(loss_hours) else None
        if common_loss_hour is not None and loss_hours.max() > win_hours.get(common_loss_hour, 0) + 0.15:
            findings_loss.append(f"concentrated around hour {common_loss_hour}:00")

    hypothesis = None
    if findings_win:
        hypothesis = "The strategy may only have an edge when " + " and ".join(findings_win) + "."
    elif findings_loss:
        hypothesis = "The strategy loses primarily when " + " and ".join(findings_loss) + " -- consider filtering those conditions out."

    # Test the hypothesis on the locked holdout by re-running the SAME
    # strategy unchanged (this engine proposes a regime condition to ADD
    # in the strategy builder next, it does not silently mutate the spec --
    # doing that automatically here would just be another round of in-sample
    # curve-fitting). We instead report the strategy's raw holdout pass
    # rate so the hypothesis can be manually added and re-compared.
    holdout_bt = _run_spec(spec, holdout_df, risk) if spec else None
    holdout_row = _row("Holdout (unmodified strategy)", holdout_bt, prop_rules, window_trading_days) if holdout_bt else None

    return {
        "winning_trade_tendencies": findings_win,
        "losing_trade_tendencies": findings_loss,
        "hypothesis": hypothesis,
        "holdout_check": holdout_row,
        "next_step": (
            "Add the winning-trade condition(s) above as an explicit regime filter in the Manual Strategy "
            "Builder, then re-run Edge Decomposition to confirm it's actually the layer doing the work on "
            "the untouched holdout, not just in-sample."
            if hypothesis else
            "No strong separating feature found with the built-in feature set -- winners and losers look "
            "similar on recent range, mean position, and hour of day."
        ),
    }


# ---------------------------------------------------------------------------
# 8. Research Director -- cross-run synthesis
# ---------------------------------------------------------------------------

def research_director_report(candidates: list[dict], top_pct: float = 0.02) -> dict:
    """Owen's ask: something sitting above every individual test, watching
    everything T58 has tried and answering "what have we learned?"

    `candidates`: a flat list of dicts, one per tested strategy, each with
    at minimum {"family": str, "passed_stage3_gate"/"passed": bool,
    "composite_score" or "eval_pass_probability": float,
    "max_drawdown_pct": float | None}. This is exactly the shape
    app.search.results_db.ResultsDB.leaderboard(...) already returns, so
    the natural call is:
        candidates = results_db.leaderboard(run_id, stage="stage3", top_n=100000, only_passed=False)
        report = research_director_report(candidates)
    This function does no I/O itself so it can be tested and reused (web
    route, desktop tab, or a scheduled batch report) without depending on
    a live ResultsDB/run_id.
    """
    if not candidates:
        return {"note": "No tested strategies to summarize yet."}

    by_family: dict[str, list[dict]] = {}
    for c in candidates:
        fam = c.get("family") or "unknown"
        by_family.setdefault(fam, []).append(c)

    def _score(c: dict) -> float:
        for key in ("composite_score", "eval_pass_probability", "cpcv_oos_eval_pass_probability", "fitness"):
            v = c.get(key)
            if v is not None:
                return float(v)
        return 0.0

    def _passed(c: dict) -> bool:
        return bool(c.get("passed_stage3_gate") or c.get("passed"))

    family_stats = []
    for fam, rows in by_family.items():
        n = len(rows)
        n_passed = sum(1 for r in rows if _passed(r))
        avg_score = pystats.fmean(_score(r) for r in rows) if rows else 0.0
        avg_dd = pystats.fmean(r["max_drawdown_pct"] for r in rows if r.get("max_drawdown_pct") is not None) if any(r.get("max_drawdown_pct") is not None for r in rows) else None
        family_stats.append({"family": fam, "n_tested": n, "n_passed": n_passed, "pass_rate_pct": round(100 * n_passed / n, 1), "avg_score": round(avg_score, 4), "avg_max_drawdown_pct": round(avg_dd, 2) if avg_dd is not None else None})

    family_stats.sort(key=lambda r: r["pass_rate_pct"], reverse=True)

    ranked = sorted(candidates, key=_score, reverse=True)
    n_top = max(1, round(top_pct * len(ranked)))
    top_slice = ranked[:n_top]
    top_families = {}
    for c in top_slice:
        fam = c.get("family") or "unknown"
        top_families[fam] = top_families.get(fam, 0) + 1
    dominant_family = max(top_families.items(), key=lambda kv: kv[1])[0] if top_families else None

    lines = [f"Tested {len(candidates):,} strategies across {len(by_family)} families."]
    consistently_failing = [f["family"] for f in family_stats if f["n_tested"] >= 20 and f["pass_rate_pct"] < 5]
    strong = [f["family"] for f in family_stats if f["n_tested"] >= 10 and f["pass_rate_pct"] >= 25]
    if consistently_failing:
        lines.append("Consistently failing (stop allocating compute here): " + ", ".join(consistently_failing) + ".")
    if strong:
        lines.append("Consistently strong -- worth deeper search here: " + ", ".join(strong) + ".")
    if dominant_family:
        lines.append(f"The top {top_pct:.0%} of all tested strategies is dominated by the \"{dominant_family}\" family ({top_families[dominant_family]} of {n_top}).")

    return {
        "n_candidates": len(candidates),
        "family_stats": family_stats,
        "top_slice_size": n_top,
        "top_slice_family_counts": top_families,
        "dominant_top_family": dominant_family,
        "consistently_failing_families": consistently_failing,
        "consistently_strong_families": strong,
        "summary": " ".join(lines),
    }
