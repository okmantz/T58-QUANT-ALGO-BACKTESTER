"""Post-verdict honesty layers for Full Pipeline (2026-10 audit).

``apply_audit_layers`` takes the verdict the legacy pipeline produced and applies the audit's
demote-only rules, then computes the EV verdict next to it. It can only make a verdict
LESS optimistic, never more:

  * lookahead check CRASHED            -> NOT READY (a crash must not read as a pass)
  * clean holdout missing / unusable   -> READY capped at MARGINAL
  * data gate (synthetic, <5 years)    -> READY capped at MARGINAL
  * preset rules unverified            -> READY capped at MARGINAL
  * search-bias-deflated pass rate     -> READY capped at MARGINAL when it falls below the READY bar

Every demotion inserts its reason FIRST in ``verdict_reasons`` so anything that prints the
headline reason (reports, library save notes) leads with the demotion.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from app.orchestration.ev_verdict import EVVerdict, deflate_pass_rate, evaluate_ev


def _preset_matches_rules(rules) -> tuple[bool, str]:
    """True when PropRules still equal the verified preset they came from (a user who edited the
    drawdown or target by hand is no longer testing the verified rules)."""
    key = getattr(rules, "preset_key", None)
    if not key:
        return False, "these PropRules were not built from a firm preset"
    try:
        from app.prop.presets import get_preset, ready_eligibility
        p = get_preset(key)
    except Exception as exc:  # noqa: BLE001
        return False, f"preset '{key}' could not be loaded ({exc})"
    _ok, _why = ready_eligibility(p)
    if not _ok:
        return False, _why
    base = p.to_prop_rules()
    for f in ("account_size", "evaluation_profit_target_pct", "daily_loss_limit_pct", "max_drawdown_pct", "drawdown_type",
              "dd_basis", "daily_loss_basis", "trailing_lock", "consistency_rule_pct", "min_trading_days"):
        if getattr(rules, f, None) != getattr(base, f, None):
            return False, f"PropRules.{f}={getattr(rules, f, None)!r} differs from verified preset '{key}' ({getattr(base, f, None)!r})"
    return True, ""


def holdout_trades_for(df, dev_df, strategy, risk, adaptive_risk=None) -> list:
    """Trades whose entry falls after the last dev bar, from ONE continuous-account run over the full frame
    (so the holdout starts from the account state development left behind, like run_holdout_comparison)."""
    from app.backtest.engine import run_backtest
    full = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk, lookahead_check="skip")
    cut = pd.Timestamp(dev_df["timestamp"].iloc[-1])
    return [t for t in full.trades if pd.Timestamp(t.entry_time) > cut]


def holdout_pass_rate_mc(trades, prop_rules, *, sims: int = 2000, seed: int = 7, reset_on_breach: bool = False):
    """Per-attempt pass probability (FRACTION) of the holdout trades via excursion-aware Monte Carlo.
    Returns (pass_fraction, MonteCarloResult) or (None, None) if there are too few trades."""
    from app.monte_carlo.engine import MIN_TRADES_FOR_VERDICT, MonteCarloConfig, run_monte_carlo
    if len(trades) < MIN_TRADES_FOR_VERDICT:
        return None, None
    mc = run_monte_carlo(trades, prop_rules, MonteCarloConfig(n_simulations=sims, random_seed=seed, reset_on_breach=reset_on_breach),
                         selection_bias_caveat=False)
    return float(mc.per_attempt_pass_probability) / 100.0, mc


def apply_audit_layers(
    verdict: str, reasons: list, *, cfg, prop_rules, df, dev_df, instrument_label, final_strategy, risk, adaptive_risk,
    final_mc, final_holdout, payout_ev: dict | None, lookahead_check_crashed: bool, n_trials: int,
    log=lambda m: None,
):
    """Returns (verdict, reasons, ev_verdict: EVVerdict, evidence: dict)."""
    reasons = list(reasons)
    rank = {"NOT READY": 0, "MARGINAL": 1, "READY": 2}
    evidence: dict = {}

    def demote(to: str, reason: str):
        nonlocal verdict
        if rank.get(verdict, 0) > rank[to]:
            reasons.insert(0, reason)
            verdict = to
        elif reason not in reasons:
            reasons.append(reason)

    # 1. lookahead crash
    if lookahead_check_crashed:
        demote("NOT READY", "LOOKAHEAD CHECK CRASHED: the behavioural lookahead check raised instead of finishing, so the "
                            "strategy was never cleared of using future data. A crash is treated as a failure, not a pass.")

    # 2. clean holdout
    holdout_reserved = bool(getattr(cfg, "reserve_true_holdout", True)) and dev_df is not None and len(dev_df) < len(df)
    holdout_ok = bool(final_holdout and final_holdout.get("holdout_ok", True) and (final_holdout.get("holdout_trades") or 0) > 0)
    evidence["holdout_reserved"], evidence["holdout_ok"] = holdout_reserved, holdout_ok
    if not getattr(cfg, "require_clean_holdout_for_ready", True):
        pass
    elif not holdout_reserved:
        demote("MARGINAL", "NO CLEAN HOLDOUT: reserve_true_holdout was off (or the data too short to split), so every pass rate here was "
                           "measured on bars the search could see. READY requires a clean holdout.")
    elif not holdout_ok:
        demote("MARGINAL", "HOLDOUT UNUSABLE: the untouched holdout produced too few trades (or none) to confirm anything; READY requires it.")

    # 3. data gate
    from app.data.data_manifest import check_dataset
    gate = check_dataset(df, instrument_label)
    evidence["data_gate"] = gate.to_dict()
    if not gate.ok_for_ready:
        for r in gate.reasons:
            demote("MARGINAL", r)

    # 4. verified rules
    ok, why = _preset_matches_rules(prop_rules)
    evidence["rules_verified"] = ok
    if not ok:
        demote("MARGINAL", f"UNVERIFIED FIRM RULES: {why}. Pass/fail on unverified rules is not evidence for a purchase; READY is not allowed.")

    # 5. holdout pass rate (clean) + search-bias deflation
    p_hold = None
    hold_trades: list = []
    if holdout_reserved and holdout_ok:
        try:
            hold_trades = holdout_trades_for(df, dev_df, final_strategy, risk, adaptive_risk)
            p_hold, hold_mc = holdout_pass_rate_mc(hold_trades, prop_rules, sims=int(getattr(cfg, "holdout_mc_sims", 2000)), seed=getattr(cfg, "random_seed", 7) or 7)
            if hold_mc is not None:
                evidence["holdout_mc"] = {"pass_pct": hold_mc.per_attempt_pass_probability, "n_trades": len(hold_trades),
                                          "ci95": list(hold_mc.per_attempt_pass_ci95), "ruin_pct": hold_mc.risk_of_ruin_pct,
                                          "median_payout": hold_mc.median_payout, "expected_payout": hold_mc.expected_payout,
                                          "first_payout_pct": hold_mc.first_payout_probability,
                                          "median_days_to_pass": hold_mc.median_days_to_pass}
                log(f"  Clean-holdout Monte Carlo: per-attempt pass {hold_mc.per_attempt_pass_probability:.1f}% on {len(hold_trades)} holdout trades.")
        except Exception as exc:  # noqa: BLE001 -- failure = no evidence, never a pass
            log(f"  Clean-holdout Monte Carlo skipped: {exc}")
            evidence["holdout_mc_error"] = str(exc)
    min_pct = float(getattr(cfg, "min_per_attempt_pass_pct", 70.0))
    n_eff = float((payout_ev or {}).get("oos").n_effective) if (payout_ev or {}).get("oos") is not None else None
    if n_eff is None or n_eff <= 0:
        n_eff = max(10.0, len(hold_trades) / 3.0) if hold_trades else 30.0
    headline_p = float(final_mc.per_attempt_pass_probability) / 100.0
    deflated = deflate_pass_rate(headline_p, int(n_trials), n_eff)
    evidence["headline_pass_pct"], evidence["deflated_headline_pass_pct"] = headline_p * 100, deflated * 100
    evidence["n_trials"], evidence["n_effective"] = int(n_trials), n_eff
    if verdict == "READY" and deflated * 100.0 < min_pct <= headline_p * 100.0:
        demote("MARGINAL", f"SEARCH BIAS: the headline per-attempt pass rate {headline_p * 100:.0f}% falls to ~{deflated * 100:.0f}% after "
                           f"deflating for {n_trials} configurations tried (~{n_eff:.0f} independent observations); below the {min_pct:.0f}% READY bar.")

    # 6. EV verdict beside READY
    ev = _compute_ev(verdict, cfg, prop_rules, payout_ev, p_hold, hold_trades, evidence, int(n_trials), n_eff,
                     lookahead_ok=not lookahead_check_crashed, ok_rules=ok, log=log)
    return verdict, reasons, ev, evidence


def _compute_ev(verdict, cfg, prop_rules, payout_ev, p_hold, hold_trades, evidence, n_trials, n_eff, *, lookahead_ok, ok_rules, log) -> EVVerdict:
    from app.prop.presets import get_fee_info
    fee = get_fee_info(getattr(prop_rules, "preset_key", None) or "")
    oos = (payout_ev or {}).get("oos")
    win = int(getattr(cfg, "payout_window_days", 30))
    r30 = oos.payout_within_rate.get(win) if (oos is not None and getattr(oos, "payout_tracked", False)) else None
    hm = evidence.get("holdout_mc") or {}
    # V: gross first payout ~ the per-paid-path payout, capped at one profit target (a first payout is realistically about that size)
    target_usd = float(prop_rules.account_size) * float(prop_rules.evaluation_profit_target_pct) / 100.0
    fp = float(hm.get("first_payout_pct", 0.0)) / 100.0
    gross = (float(hm.get("expected_payout", 0.0)) / fp) if fp > 0 else None
    payout_value = min(gross, target_usd) if gross else target_usd
    ruin = None
    if fee is not None and hold_trades:
        try:
            from app.monte_carlo.bankroll import BankrollConfig, simulate_bankroll_survival
            from app.prop.survival_engine import ResetEconomics
            eco = ResetEconomics(evaluation_fee=fee.challenge_price or 0.0, reset_fee=fee.reset_price, profit_split_pct=fee.profit_split_pct,
                                 max_attempts=int(getattr(cfg, "ev_max_attempts", 5)))
            br = simulate_bankroll_survival(hold_trades, prop_rules, BankrollConfig(
                starting_bankroll=float(getattr(cfg, "ev_bankroll", 1000.0)), reset_economics=eco, n_simulations=int(getattr(cfg, "ev_bankroll_sims", 1000)),
                method="block_bootstrap", stop_after_first_payout=True))
            ruin = float(br.probability_ruin_before_first_payout)
            evidence["bankroll"] = {"ruin_pct": ruin, "p_first_payout_pct": br.probability_reach_first_payout, "expected_net": br.expected_net_profit,
                                    "bankroll": float(getattr(cfg, "ev_bankroll", 1000.0))}
        except Exception as exc:  # noqa: BLE001
            log(f"  Bankroll survival skipped: {exc}")
            evidence["bankroll_error"] = str(exc)
    from app.orchestration.ev_verdict import evaluation_months
    _sub = bool(fee.monthly_subscription) if fee else False
    _months = evaluation_months(hm.get("median_days_to_pass"), int(getattr(fee, "billing_period_days", 30)) if fee else 30) if _sub else 1
    evidence["subscription_months"] = _months if _sub else None
    ev = evaluate_ev(
        challenge_months=_months, monthly_subscription=_sub,
        legacy_verdict=verdict, holdout_pass_rate=p_hold, window_payout_rate=r30,
        payout_value=payout_value, challenge_price=fee.challenge_price if fee else None,
        reset_price=fee.reset_price if fee else None, profit_split_pct=fee.profit_split_pct if fee else None,
        activation_fee=(fee.activation_fee if fee else 0.0),
        bankroll_ruin_pct=ruin, ruin_cap_pct=float(getattr(cfg, "ev_ruin_cap_pct", getattr(cfg, "risk_of_ruin_cap", 20.0))),
        margin_multiple=float(getattr(cfg, "ev_margin_multiple", 1.5)), margin_points=float(getattr(cfg, "ev_margin_points", 0.05)),
        n_trials=n_trials, n_effective=n_eff, rules_verified=ok_rules, lookahead_ok=lookahead_ok,
        holdout_clean=bool(evidence.get("holdout_reserved") and evidence.get("holdout_ok")),
        automation_confirmed=fee.written_confirmation if fee else None,
    )
    if fee is None:
        ev.reasons.append("No fee row for this preset in app.prop.presets.FIRM_FEES (or hand-typed rules): add one with a source URL and date.")
    return ev
