"""Planted-edge calibration (2026-11-04 audit, Fix 2b / roadmap phase 1).

A verdict stack is only trustworthy if it can be shown to (a) almost never pass a strategy with NO edge and (b) usually pass one whose edge
is real and tradable. This module plants a KNOWN net edge (in R per trade, after costs) into synthetic trade streams, runs them through
the repo's own gate functions, and reports detection rate, first-killing gate and zero-edge false-positive rate.

``make_planted_trades``   an i.i.d. win/loss stream with a chosen true expectancy (default RR 1.5, one trade per weekday, ~5 years).
``run_gate_stack``        the STATISTICAL gates only (not the GA): outer-bootstrap pass interval, EV bar, pooled out-of-sample test, untouched
                          holdout confirmation, expectancy lower bound at 2x costs on UNSEEN data, deflated Sharpe on unseen data.
``sweep``                 replicates the above over edges and seeds -> detection table.

The full-pipeline sweep (data -> GA -> holdout -> verdict) lives in ``scripts/planted_drift_sweep.py``; this module is its gate-level core and
what the nightly regression (tests/orchestration/test_planted_edge_calibration.py) runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_RR = 1.5
DEFAULT_RISK_DOLLARS = 150.0
DEFAULT_COST_R = 0.04            # round-trip cost per trade in R, charged at 1x; the 2x stress doubles it
# Assumed per-trade-Sharpe spread of a converged GA leaderboard (sd ~0.03): the deflated-Sharpe benchmark scales with it. An assumption of the
# planted test, not a measurement -- the audit's own numbers (a 0.15R edge clearing DSR 5% of the time at 3,000 trials) imply a spread near this.
TRIAL_SHARPE_SPREAD = 0.04 * np.linspace(-1.2, 1.2, 7)


def win_probability_for_edge(edge_r: float, rr: float = DEFAULT_RR) -> float:
    """P(win) that gives expectancy ``edge_r`` R per trade with wins of +rr R and losses of -1 R: p*rr - (1-p) = edge."""
    return min(max((edge_r + 1.0) / (rr + 1.0), 0.0), 1.0)


def make_planted_trades(edge_r: float, *, n_years: float = 5.0, trades_per_day: int = 1, rr: float = DEFAULT_RR,
                        risk_dollars: float = DEFAULT_RISK_DOLLARS, seed: int = 0, cost_r: float = DEFAULT_COST_R,
                        start: str = "2021-01-04"):
    """A synthetic trade list whose TRUE net expectancy (after ``cost_r`` per trade) is ``edge_r``. Wins pay +rr R, losses -1 R, then the
    cost is subtracted from every trade. Trades are spaced one set per weekday. Each Trade carries initial_risk=1.0 and size=risk_dollars so
    ``pnl / (initial_risk * size)`` is exactly R (the convention app.scoring.edge_bounds.r_multiples uses)."""
    from app.backtest.execution import Trade
    rng = np.random.default_rng(int(seed))
    days = pd.bdate_range(start=start, periods=int(round(252 * n_years)))
    gross_edge = edge_r + cost_r
    p_win = win_probability_for_edge(gross_edge, rr)
    out: list = []
    equity = 50_000.0
    for d in days:
        for k in range(int(trades_per_day)):
            win = bool(rng.random() < p_win)
            r = (rr if win else -1.0) - cost_r
            pnl = r * risk_dollars
            entry = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=30 + 30 * k)
            equity += pnl
            out.append(Trade(
                entry_time=entry, exit_time=entry + pd.Timedelta(minutes=20), direction=1 if rng.random() < 0.5 else -1,
                entry_price=100.0, exit_price=100.0 + (r if True else 0.0), size=risk_dollars, pnl=pnl, pnl_pct=pnl / 50_000.0 * 100.0,
                exit_reason="take_profit" if win else "stop_loss", commission=cost_r * risk_dollars, equity_after=equity,
                initial_risk=1.0, intended_risk_dollars=risk_dollars,
            ))
    return out


def stressed_r(trades, cost_r: float = DEFAULT_COST_R, multiplier: float = 2.0) -> list[float]:
    """R multiples of ``trades`` after charging the cost ``multiplier`` times instead of once (the extra (multiplier-1) x cost_r comes off every trade)."""
    from app.scoring.edge_bounds import r_multiples
    return [x - (float(multiplier) - 1.0) * cost_r for x in r_multiples(trades)]


@dataclass
class GateStackResult:
    edge_r: float
    seed: int
    pass_pct: float = 0.0
    pass_lb: float = 0.0
    ev_positive: bool = False
    pooled_oos_status: str = "n/a"
    holdout_confirmed: bool = False
    unseen_expectancy_lb: float | None = None
    unseen_expectancy_ok: bool = False
    dsr_unseen_ok: bool = False
    ev_label: str = ""                 # label from the production ``evaluate_ev`` (firm fee + payout), not a fixed pass-rate number
    ev_break_even_p: float | None = None
    legacy_pass_lb: float = 0.0
    legacy_would_pass: bool = False
    detected: bool = False
    forward_candidate: bool = False   # edge evidence alone (P2-1): holdout confirmed + unseen expectancy LB>0 at 2x costs + EV>0 + pooled OOS not FAIL
    first_killer: str | None = None
    detail: dict = field(default_factory=dict)
    detail_dsr: dict | None = None
    _dev_trades: object = None


def _production_ev(res, mc, rules) -> bool:
    """EV test through ``evaluate_ev`` exactly as the pipeline feeds it: the firm's fee row, the simulated first-payout size and the
    outer-bootstrap pass lower bound. A rules object without a fee row has no economic verdict, which is NOT a pass."""
    try:
        from app.orchestration.ev_verdict import evaluate_ev
        from app.prop.presets import get_fee_info
        fee = get_fee_info(getattr(rules, "preset_key", None) or "")
        if fee is None:
            return False
        fp = float(mc.first_payout_probability or 0.0) / 100.0
        target = float(rules.account_size) * float(rules.evaluation_profit_target_pct) / 100.0
        gross = (float(mc.expected_payout) / fp) if fp > 0 else None
        # bankroll ruin before first payout and per-attempt subscription billing, computed the way audit_layers._compute_ev computes them
        from app.monte_carlo.bankroll import BankrollConfig, simulate_bankroll_survival
        from app.orchestration.ev_verdict import billing_profile as _bp
        from app.prop.survival_engine import ResetEconomics
        eco = ResetEconomics(evaluation_fee=fee.challenge_price or 0.0, reset_fee=fee.reset_price, profit_split_pct=fee.profit_split_pct, max_attempts=5)
        br = simulate_bankroll_survival(getattr(res, "_dev_trades", None) or [], rules, BankrollConfig(
            starting_bankroll=1000.0, reset_economics=eco, n_simulations=300, method="block_bootstrap", stop_after_first_payout=True))
        prof = (_bp(list(getattr(mc, "attempt_outcomes", None) or []), list(getattr(mc, "attempt_calendar_days", None) or []),
                    int(getattr(fee, "billing_period_days", 30))) if fee.monthly_subscription else None)
        ev = evaluate_ev(
            bankroll_ruin_pct=float(br.probability_ruin_before_first_payout), billing_profile=prof,
            legacy_verdict="MARGINAL", holdout_pass_rate=res.pass_pct / 100.0, window_payout_rate=fp,
            payout_value=min(gross, target) if gross else target, challenge_price=fee.challenge_price, reset_price=fee.reset_price,
            profit_split_pct=fee.profit_split_pct, activation_fee=fee.activation_fee, monthly_subscription=bool(fee.monthly_subscription),
            holdout_pass_rate_lb=(res.pass_lb / 100.0 if getattr(mc, "pass_interval_method", "") == "outer_bootstrap" else None),
            selection_handled_by_holdout=True, parameters_tuned=False)
        res.ev_label, res.ev_break_even_p = ev.label, ev.break_even_p
        return ev.label in ("HIGH CONFIDENCE", "FORWARD-TEST CANDIDATE")
    except Exception:  # noqa: BLE001
        return False


def _pooled_oos(r_oos: list[float], days_oos: list) -> str:
    from app.validation.edge_stats import pooled_oos_edge
    try:
        pe = pooled_oos_edge([r_oos], [days_oos])
        return str(getattr(pe, "status", "n/a"))
    except Exception:  # noqa: BLE001
        return "n/a"


def run_gate_stack(edge_r: float, seed: int, *, n_years: float = 5.0, rules=None, mc_sims: int = 1500, outer_reps: int = 16, outer_sims: int = 300,
                   holdout_frac: float = 0.20, oos_frac: float = 0.35, pass_lb_bar: float = 70.0, n_trials: int = 3000,
                   cost_r: float = DEFAULT_COST_R) -> GateStackResult:
    """Statistical gate stack on one planted stream (see module docstring). ``detected`` is the redesigned stack's READY-equivalent: the edge
    evidence (pooled OOS not FAIL, holdout confirms, unseen expectancy LB > 0 at 2x costs) AND the business evidence (outer-bootstrap pass LB
    clears ``pass_lb_bar``). ``legacy_would_pass`` re-computes the OLD stack (Wilson interval over median-days-to-pass windows, DSR on all
    trades at ``n_trials`` trials, expectancy LB at 2x costs) so the two can be compared on the same streams."""
    from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
    from app.orchestration.audit_layers import holdout_confirmation
    from app.prop.presets import get_preset
    from app.scoring.edge_bounds import expectancy_bound, r_multiples, sample_aware_interval
    rules = rules or get_preset("topstep_50k").to_prop_rules()
    trades = make_planted_trades(edge_r, n_years=n_years, seed=seed, cost_r=cost_r)
    n = len(trades)
    n_hold = int(n * holdout_frac)
    dev, hold = trades[: n - n_hold], trades[n - n_hold:]
    n_oos = int(len(dev) * oos_frac)
    oos = dev[len(dev) - n_oos:]
    res = GateStackResult(edge_r=float(edge_r), seed=int(seed))

    cfg = MonteCarloConfig(n_simulations=int(mc_sims), random_seed=int(seed) + 1, reset_on_breach=False,
                           outer_bootstrap_reps=int(outer_reps), outer_bootstrap_sims=int(outer_sims))
    mc = run_monte_carlo(dev, rules, cfg)
    res.pass_pct = float(mc.per_attempt_pass_probability)
    res.pass_lb = float(mc.per_attempt_pass_ci95_sample[0])
    # old stack's pass lower bound: Wilson over windows sized by the median TRADING days to pass
    old_days = max(30.0, float(mc.median_days_to_pass or 0.0))
    (old_lo, _), _ = sample_aware_interval(res.pass_pct, len(dev), int(mc.n_source_calendar_days), attempt_days=old_days,
                                           n_simulated_attempts=float(mc.total_independent_attempts))
    res.legacy_pass_lb = float(old_lo)

    # edge evidence on UNSEEN data
    r_oos = stressed_r(oos, cost_r, 1.0)
    days_oos = [pd.Timestamp(t.entry_time).normalize() for t in oos]
    res.pooled_oos_status = _pooled_oos(r_oos, days_oos)
    conf = holdout_confirmation(hold, r_oos, min_trades=30)
    res.holdout_confirmed = bool(conf["confirmed"])
    unseen = stressed_r(hold, cost_r, 2.0) + stressed_r(oos, cost_r, 2.0)
    eb = expectancy_bound(unseen)
    res.unseen_expectancy_lb = eb.get("lb")
    res.unseen_expectancy_ok = eb.get("lb") is not None and eb["lb"] > 0.0
    # EV bar: the PRODUCTION evaluate_ev with this firm's fee, split and payout (2026-11 audit -- it used to be a fixed ``pass_lb >= 15``).
    res._dev_trades = dev
    res.ev_positive = _production_ev(res, mc, rules)
    res._dev_trades = None
    # DSR on unseen data: the PRODUCTION function the pipeline calls (audit_layers.unseen_dsr), at the stack's effective trial count.
    from app.orchestration.audit_layers import unseen_dsr
    _du = unseen_dsr(unseen, max(2, int(n_trials // 10)))
    res.dsr_unseen_ok = bool(_du["passed"]) if _du is not None else False
    res.detail_dsr = None if _du is None else {"psr": _du["psr"], "basis": _du.get("spread_basis")}
    kills = []
    if res.pooled_oos_status == "FAIL":
        kills.append("oos")
    if not res.holdout_confirmed:
        kills.append("holdout")
    if not res.unseen_expectancy_ok:
        kills.append("expectancy_unseen")
    if not res.dsr_unseen_ok:
        kills.append("dsr_unseen")
    if not res.ev_positive:
        kills.append("ev")
    if res.pass_lb < pass_lb_bar:
        kills.append("pass_lb")
    res.first_killer = kills[0] if kills else None
    res.detected = not kills
    res.forward_candidate = bool(res.holdout_confirmed and res.unseen_expectancy_ok and res.ev_positive and res.pooled_oos_status != "FAIL")
    # the old stack: in-sample expectancy LB at 2x costs, Wilson LB >= bar, in-sample DSR at n_trials
    r_dev2 = stressed_r(dev, cost_r, 2.0)
    ebd = expectancy_bound(r_dev2)
    try:
        from app.search.robustness import deflated_sharpe_gate
        a = np.asarray(r_multiples(dev), dtype=float)
        sr = float(a.mean() / a.std(ddof=1)) if a.std(ddof=1) > 0 else 0.0
        g2 = deflated_sharpe_gate(sr, list(sr + TRIAL_SHARPE_SPREAD), n_trials, len(a), returns_skew=0.0, returns_kurtosis=3.0)
        old_dsr = bool(g2.passed)
    except Exception:  # noqa: BLE001
        old_dsr = True
    res.legacy_would_pass = bool(res.legacy_pass_lb >= pass_lb_bar and (ebd.get("lb") or -1) > 0 and old_dsr)
    res.detail = {"holdout": conf.get("reason"), "mc_interval_method": mc.pass_interval_method, "median_calendar_days_to_pass": mc.median_calendar_days_to_pass}
    return res


def sweep(edges=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30), seeds=range(20), **kw) -> list[dict]:
    """Detection table: for each planted edge, the share of seeds the redesigned stack detects, the legacy stack's share, the mean pass rate and
    interval, and which gate kills most often."""
    rows = []
    for e in edges:
        rs = [run_gate_stack(e, s, **kw) for s in seeds]
        kill: dict = {}
        for r in rs:
            if r.first_killer:
                kill[r.first_killer] = kill.get(r.first_killer, 0) + 1
        rows.append({
            "edge_r": float(e), "n": len(rs),
            "detected_pct": 100.0 * sum(r.detected for r in rs) / len(rs),
            "forward_candidate_pct": 100.0 * sum(r.forward_candidate for r in rs) / len(rs),
            "legacy_pct": 100.0 * sum(r.legacy_would_pass for r in rs) / len(rs),
            "mean_pass_pct": float(np.mean([r.pass_pct for r in rs])), "mean_pass_lb": float(np.mean([r.pass_lb for r in rs])),
            "mean_legacy_lb": float(np.mean([r.legacy_pass_lb for r in rs])),
            "first_killer": max(kill, key=kill.get) if kill else None, "killers": kill,
        })
    return rows


def render_sweep(rows: list[dict]) -> str:
    lines = ["edge(R)  detect%  fwd-cand%  legacy%  pass%   passLB(new)  passLB(old)  top killer"]
    for r in rows:
        lines.append(f"{r['edge_r']:>6.2f}  {r['detected_pct']:>7.0f}  {r['forward_candidate_pct']:>9.0f}  {r['legacy_pct']:>7.0f}  {r['mean_pass_pct']:>5.1f}   {r['mean_pass_lb']:>10.1f}  "
                     f"{r['mean_legacy_lb']:>10.1f}  {r['first_killer'] or '-'}")
    return "\n".join(lines)
