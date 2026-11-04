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


_STALE_NOTE: list = []


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
        # 2026-11-04 (P1-3): a primary-verified but merely OLD stamp is not a verdict problem (see rules_stamp_state)
        from app.prop.presets import rules_stamp_state
        _state, _swhy = rules_stamp_state(p)
        if _state != "stale":
            return False, _why
        _STALE_NOTE.append(_swhy)
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


def edge_shrinkage_and_plan(unseen_r, cfg=None) -> dict | None:
    """Empirical-Bayes shrinkage of the unseen-data mean R toward the family prior, plus the forward-test (SPRT) plan (2026-11-04, P2-2).

    A strategy picked as the best of a search carries an optimistic edge estimate; a family prior pulls it back by how little data it has
    (a 100-trade winner moves a lot, a 2,000-trade one barely). ``family_prior_mu0`` / ``family_prior_tau`` (FullPipelineConfig, defaults
    0.0R and 0.15R) are the mean and spread of true per-trade edges across strategies of this kind -- an assumption until the ledger holds
    enough strategies for ``edge_stats.estimate_family_prior``. Returns observed / shrunk mean R and shrunk lower bound, the SPRT boundaries
    for H1 = the shrunk edge (floored at ``sprt_min_h1_r``), the trades and months a forward test needs, or None with < 30 unseen trades."""
    try:
        import numpy as _np
        from app.validation.edge_stats import (DEFAULT_TRADE_SD_R, empirical_bayes_shrink, shrunk_lower_bound, sprt_boundaries, trades_needed)
        a = _np.asarray([float(x) for x in (unseen_r or []) if x is not None and x == x], dtype=float)
        if len(a) < 30:
            return None
        n, xbar = int(len(a)), float(a.mean())
        sd = float(a.std(ddof=1)) or DEFAULT_TRADE_SD_R
        mu0 = float(getattr(cfg, "family_prior_mu0", 0.0)); tau = float(getattr(cfg, "family_prior_tau", 0.15))
        shrunk = empirical_bayes_shrink(n, xbar, sd, mu0, tau)
        lb = shrunk_lower_bound(n, xbar, sd, mu0, tau)
        h1 = max(float(shrunk), float(getattr(cfg, "sprt_min_h1_r", 0.05)))
        bnd = sprt_boundaries(h1, sd_r=sd)
        need = trades_needed(h1, sd_r=sd)
        out = {"n": n, "observed_mean_r": xbar, "sd_r": sd, "family_mu0": mu0, "family_tau": tau, "shrunk_mean_r": float(shrunk), "shrunk_lb_r": float(lb),
               "sprt": {"h1_r": h1, "h0_r": 0.0, "alpha": bnd.alpha, "beta": bnd.beta, "accept_h1_at": bnd.accept_h1_at, "accept_h0_at": bnd.accept_h0_at,
                        "expected_trades_if_h1": bnd.expected_trades_h1, "expected_trades_if_h0": bnd.expected_trades_h0},
               "trades_needed_fixed_sample": need}
        return out
    except Exception:  # noqa: BLE001
        return None


def _shifted_trades(trades, delta_r: float) -> list:
    """Copies of ``trades`` with each trade's P&L lowered by ``delta_r`` x its planned dollar risk (trades without a planned risk are kept)."""
    from dataclasses import replace as _rep
    out = []
    for t in trades or []:
        ir, size, pnl = getattr(t, "initial_risk", None), getattr(t, "size", None), getattr(t, "pnl", None)
        if pnl is None or not ir or ir <= 0 or not size:
            out.append(t)
            continue
        try:
            out.append(_rep(t, pnl=float(pnl) - float(delta_r) * float(ir) * float(size)))
        except Exception:  # noqa: BLE001
            out.append(t)
    return out


def edge_unseen_r_list(oos_fold_r, hold_trades) -> list:
    from app.scoring.edge_bounds import r_multiples as _rm
    return [float(x) for x in (oos_fold_r or []) if x is not None and x == x] + (_rm(hold_trades) if hold_trades else [])


def unseen_dsr(unseen_r, effective_trials: int, *, min_psr: float = 0.95) -> dict | None:
    """Deflated Sharpe on UNSEEN trades only (holdout + walk-forward out-of-sample R), 2026-11-04 audit P0-2.

    The old gate deflated the IN-SAMPLE Sharpe by the search-wide trial count, so a bigger search was punished for being big even when the
    edge then held up on untouched data. The unseen trades were never searched over, so only the trials that could have SELECTED this
    strategy (the per-family, fingerprint-clustered, decayed effective count) matter. Returns {psr, passed, n, sharpe, trials, score}
    or None when there are too few trades. Never raises."""
    try:
        import numpy as _np
        from app.search.robustness import deflated_sharpe_gate
        from app.scoring.t58_scorecard import dsr_unseen_to_score
        from app.validation.planted_edge import TRIAL_SHARPE_SPREAD
        a = _np.asarray([float(x) for x in (unseen_r or []) if x is not None and x == x], dtype=float)
        if len(a) < 30 or float(a.std(ddof=1)) <= 0:
            return None
        sr = float(a.mean() / a.std(ddof=1))
        sd0 = float(a.std(ddof=0))
        sk = float(((a - a.mean()) ** 3).mean() / sd0 ** 3) if sd0 > 0 else 0.0
        ku = float(((a - a.mean()) ** 4).mean() / sd0 ** 4) if sd0 > 0 else 3.0
        trials = max(2, int(effective_trials))
        g = deflated_sharpe_gate(sr, list(sr + TRIAL_SHARPE_SPREAD), trials, len(a), returns_skew=sk, returns_kurtosis=ku,
                                 min_probabilistic_sharpe=float(min_psr))
        return {"psr": float(g.probabilistic_sharpe), "passed": bool(g.passed), "n": int(len(a)), "sharpe": sr, "trials": trials,
                "score": dsr_unseen_to_score(float(g.probabilistic_sharpe)), "reason": g.reason}
    except Exception:  # noqa: BLE001
        return None


def edge_evidence_for_ladder(confirmation, null_ev, cost_ev, ev, cfg, verdict) -> dict:
    """The four facts that earn FORWARD-TEST CANDIDATE on edge evidence alone (2026-11-04, P2-1): holdout confirmed; random-entry null not
    rejected; unseen expectancy lower bound > 0 at stressed costs; EV per attempt > 0. A hard NOT READY from lookahead / ruin never qualifies."""
    try:
        null_p = None if not null_ev else null_ev.get("p_value")
        null_ok = null_p is None or float(null_p) <= float(getattr(cfg, "null_p_max", 0.10))
        eb = (cost_ev or {}).get("unseen_expectancy_bound") or {}
        cost_ok = eb.get("lb") is not None and float(eb["lb"]) > float(getattr(cfg, "min_expectancy_lb_r", 0.0))
        hard = verdict == "NOT READY" and not any(("EV" in str(x) and "NOT WORTH" in str(x)) for x in [])
        out = {"holdout_confirmed": bool((confirmation or {}).get("confirmed")), "null_ok": bool(null_ok and null_p is not None), "cost_ok": bool(cost_ok),
               "ev_mean": getattr(ev, "ev", None)}
        out["null_p"] = null_p
        return out
    except Exception:  # noqa: BLE001
        return {}


def _mc_summary(mc, n_trades: int, basis: str) -> dict:
    return {"pass_pct": mc.per_attempt_pass_probability, "n_trades": n_trades, "ci95": list(mc.per_attempt_pass_ci95), "ruin_pct": mc.risk_of_ruin_pct,
            "median_payout": mc.median_payout, "expected_payout": mc.expected_payout, "first_payout_pct": mc.first_payout_probability,
            "median_days_to_pass": mc.median_days_to_pass, "basis": basis,
            # 2026-11-04: calendar-day bookkeeping, outer-bootstrap lower bound and per-attempt billing records
            "median_calendar_days_to_pass": getattr(mc, "median_calendar_days_to_pass", None),
            "pass_lb_pct": (list(getattr(mc, "per_attempt_pass_ci95_sample", None) or [0.0, 0.0]) or [0.0])[0],
            "pass_interval_method": getattr(mc, "pass_interval_method", None),
            "attempt_outcomes": list(getattr(mc, "attempt_outcomes", None) or []),
            "attempt_calendar_days": list(getattr(mc, "attempt_calendar_days", None) or [])}


def holdout_confirmation(hold_trades, oos_fold_r=None, *, min_trades: int = 30) -> dict:
    """Is the edge CONFIRMED on unseen data? Needs >= ``min_trades`` holdout trades, a positive holdout mean R, and a positive 95% lower
    bound on mean R over the pooled unseen evidence (walk-forward out-of-sample R + holdout R). Never raises."""
    from app.scoring.edge_bounds import expectancy_bound, r_multiples
    out = {"confirmed": False, "n_holdout": len(hold_trades or []), "holdout_mean_r": None, "pooled_n": 0, "pooled_lb_r": None, "reason": ""}
    try:
        r_h = r_multiples(hold_trades)
    except Exception:  # noqa: BLE001
        r_h = []
    if len(r_h) < int(min_trades):
        out["reason"] = f"only {len(r_h)} holdout trades with a planned risk (need {int(min_trades)}) -- not enough to confirm anything"
        return out
    out["holdout_mean_r"] = float(sum(r_h) / len(r_h))
    pooled = [float(x) for x in (oos_fold_r or []) if x is not None and x == x] + [float(x) for x in r_h]
    eb = expectancy_bound(pooled)
    out["pooled_n"], out["pooled_lb_r"], out["pooled_mean_r"] = int(eb.get("n", 0)), eb.get("lb"), eb.get("mean")
    if out["holdout_mean_r"] <= 0:
        out["reason"] = f"holdout mean R is {out['holdout_mean_r']:+.3f} over {len(r_h)} trades (not positive)"
    elif eb.get("lb") is None or eb["lb"] <= 0:
        out["reason"] = (f"holdout mean R {out['holdout_mean_r']:+.3f} over {len(r_h)} trades, but the pooled unseen-data 95% lower bound "
                         f"({'n/a' if eb.get('lb') is None else format(eb['lb'], '+.3f')}R over {out['pooled_n']} trades) is not above zero")
    else:
        out["confirmed"] = True
        out["reason"] = (f"holdout mean R {out['holdout_mean_r']:+.3f} over {len(r_h)} trades; pooled unseen-data lower bound {eb['lb']:+.3f}R "
                         f"over {out['pooled_n']} trades")
    return out


def apply_audit_layers(
    verdict: str, reasons: list, *, cfg, prop_rules, df, dev_df, instrument_label, final_strategy, risk, adaptive_risk,
    final_mc, final_holdout, payout_ev: dict | None, lookahead_check_crashed: bool, n_trials: int,
    log=lambda m: None, parameters_tuned: bool = True, final_trades=None, oos_fold_r=None, dsr_gate_result=None, null_ev=None, cost_ev=None, transfer_ev=None,
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
    gate = check_dataset(df, instrument_label, transfer_passed=(str((transfer_ev or {}).get('status', '')).upper() == 'PASS') if transfer_ev else None)
    evidence["data_gate"] = gate.to_dict()
    if not gate.ok_for_ready:
        for r in gate.reasons:
            demote("MARGINAL", r)

    # 4. verified rules
    _STALE_NOTE.clear()
    ok, why = _preset_matches_rules(prop_rules)
    evidence["rules_verified"] = ok
    if ok and _STALE_NOTE:
        evidence["rules_stamp"] = {"state": "stale", "note": _STALE_NOTE[0]}
        reasons.append("RULES STAMP STALE (banner, not a verdict change): " + _STALE_NOTE[0] + ". It is a blocker on the deployment checklist.")
    else:
        evidence["rules_stamp"] = {"state": "ok" if ok else "unverified", "note": why}
    if not ok:
        demote("MARGINAL", f"UNVERIFIED FIRM RULES: {why}. Pass/fail on unverified rules is not evidence for a purchase; READY is not allowed.")

    # 5. holdout evidence (pooled) + search-bias deflation
    # 2026-10-27 audit (fix 2). The holdout used to be MC-simulated all the way to a 10% target from ~100 trades. At sane per-trade risk
    # that needs ~200 trades, so a real edge scored 19% on pure sampling noise and was then deflated to 0%. The holdout is now used for
    # what it can actually decide -- CONFIRMATION that the edge is real on unseen data -- and the pass rate comes from the full-sample
    # Monte Carlo (still search-bias deflated below). Confirmation = at least ``min_holdout_trades_for_gate`` holdout trades, a positive
    # holdout mean R, and a positive 95% lower bound on mean R over the POOLED unseen evidence (every walk-forward out-of-sample trade
    # plus the holdout). Not confirmed -> the old holdout-only Monte Carlo is the (conservative) pass-rate source, as before.
    p_hold = None
    hold_trades: list = []
    pooled_mc = None
    min_pct = float(getattr(cfg, "min_per_attempt_pass_pct", 70.0))
    headline_p = float(final_mc.per_attempt_pass_probability) / 100.0
    if holdout_reserved and holdout_ok:
        try:
            hold_trades = holdout_trades_for(df, dev_df, final_strategy, risk, adaptive_risk)
        except Exception as exc:  # noqa: BLE001 -- failure = no evidence, never a pass
            log(f"  Clean-holdout trades could not be computed: {exc}")
            evidence["holdout_mc_error"] = str(exc)
    confirmation = holdout_confirmation(hold_trades, oos_fold_r, min_trades=int(getattr(cfg, "min_holdout_trades_for_gate", 30)))
    evidence["holdout_confirmation"] = confirmation
    if hold_trades:
        log(f"  Holdout confirmation: {confirmation['reason']}")
    if confirmation["confirmed"] and getattr(cfg, "pool_holdout_evidence", True):
        pooled_mc, p_hold = final_mc, headline_p
        evidence["holdout_mc"] = _mc_summary(final_mc, len(hold_trades), basis="pooled")
        log(f"  Holdout confirms the edge, so the pass rate comes from the full-sample Monte Carlo ({final_mc.per_attempt_pass_probability:.1f}%) "
            "rather than from a 10%-target simulation of the short holdout slice.")
    elif hold_trades:
        try:
            p_hold, hold_mc = holdout_pass_rate_mc(hold_trades, prop_rules, sims=int(getattr(cfg, "holdout_mc_sims", 2000)), seed=getattr(cfg, "random_seed", 7) or 7)
            if hold_mc is not None:
                evidence["holdout_mc"] = _mc_summary(hold_mc, len(hold_trades), basis="holdout")
                log(f"  Clean-holdout Monte Carlo: per-attempt pass {hold_mc.per_attempt_pass_probability:.1f}% on {len(hold_trades)} holdout trades.")
        except Exception as exc:  # noqa: BLE001 -- failure = no evidence, never a pass
            log(f"  Clean-holdout Monte Carlo skipped: {exc}")
            evidence["holdout_mc_error"] = str(exc)
    if pooled_mc is not None:
        from app.scoring.edge_bounds import effective_independent_attempts
        try:
            _days = int((pd.Timestamp(df["timestamp"].iloc[-1]) - pd.Timestamp(df["timestamp"].iloc[0])).days)
        except Exception:  # noqa: BLE001
            _days = 0
        n_eff = effective_independent_attempts(len(final_trades or []) + len(hold_trades), _days)
    else:
        n_eff = float((payout_ev or {}).get("oos").n_effective) if (payout_ev or {}).get("oos") is not None else None
        if n_eff is None or n_eff <= 0:
            n_eff = max(10.0, len(hold_trades) / 3.0) if hold_trades else 30.0
    # 2026-11-04 audit (P1-1): ONE uncertainty treatment per quantity. When the holdout CONFIRMED the edge on untouched data, selection bias
    # has been handled by the holdout; deflating the same pass rate again for the number of configurations tried double-counts it.
    selection_handled = bool(confirmation["confirmed"] and holdout_reserved and holdout_ok and pooled_mc is not None
                             and getattr(cfg, "holdout_replaces_deflation", True))
    evidence["selection_handled_by_holdout"] = selection_handled
    deflated = headline_p if selection_handled else deflate_pass_rate(headline_p, int(n_trials), n_eff, tunable=parameters_tuned)
    evidence["headline_pass_pct"], evidence["deflated_headline_pass_pct"] = headline_p * 100, deflated * 100
    evidence["n_trials"], evidence["n_effective"], evidence["parameters_tuned"] = int(n_trials), n_eff, bool(parameters_tuned)
    # 2026-10-31 audit (section 4, defect 5): the pass-rate haircut and the DSR gate used two different "trial counts" under one name.
    # Label this one. It is THIS RUN's search evaluations only; the DSR gate's ``raw_trials`` / ``effective_trials`` (see
    # DeflatedSharpeGateResult) also include the ledger, external trials and the family prior.
    evidence["haircut_trials"] = int(n_trials)
    evidence["haircut_trials_basis"] = ("this run's search evaluations only; the Deflated Sharpe gate's raw_trials / effective_trials "
                                        "also include the trial ledger, external trials and the family prior")
    if (not selection_handled) and verdict == "READY" and deflated * 100.0 < min_pct <= headline_p * 100.0:
        demote("MARGINAL", f"SEARCH BIAS: the headline per-attempt pass rate {headline_p * 100:.0f}% falls to ~{deflated * 100:.0f}% after "
                           f"deflating for {n_trials} configurations tried (~{n_eff:.0f} independent observations); below the {min_pct:.0f}% READY bar.")

    # 5b. Deflated Sharpe on UNSEEN data (2026-11-04, P0-2): a score input + READY cap, never a NOT READY by itself.
    try:
        from app.scoring.edge_bounds import r_multiples as _rm
        _unseen_r = [float(x) for x in (oos_fold_r or []) if x is not None and x == x] + (_rm(hold_trades) if hold_trades else [])
    except Exception:  # noqa: BLE001
        _unseen_r = []
    _eff_trials = getattr(dsr_gate_result, "effective_trials", None) or getattr(dsr_gate_result, "n_trials", None) or int(n_trials)
    _du = unseen_dsr(_unseen_r, int(_eff_trials), min_psr=float(getattr(cfg, "dsr_min_probabilistic_sharpe", 0.95)))
    evidence["dsr_unseen"] = _du
    if _du is not None:
        log(f"  Deflated Sharpe on unseen data: PSR {_du['psr']:.3f} over {_du['n']} unseen trades vs {_du['trials']} effective trial(s) "
            f"({'passes' if _du['passed'] else 'below the bar'}).")
        if not _du["passed"] and verdict == "READY":
            demote("MARGINAL", f"DEFLATED SHARPE ON UNSEEN DATA: probabilistic Sharpe {_du['psr']:.3f} over {_du['n']} holdout + out-of-sample trades "
                               f"is below {float(getattr(cfg, 'dsr_min_probabilistic_sharpe', 0.95)):.2f} after deflating for {_du['trials']} effective trial(s); "
                               "capped at MARGINAL (a score input, not a rejection).")

    # 5c. Empirical-Bayes shrinkage + forward-test plan (2026-11-04, P2-2). The shrunk edge can only LOWER the pass rate the EV uses.
    _shr = edge_shrinkage_and_plan(_unseen_r, cfg)
    evidence["edge_shrinkage"] = _shr
    if _shr is not None:
        log(f"  Edge shrinkage: unseen mean {_shr['observed_mean_r']:+.3f}R over {_shr['n']} trades -> {_shr['shrunk_mean_r']:+.3f}R toward the family prior "
            f"({_shr['family_mu0']:+.2f}R, spread {_shr['family_tau']:.2f}R); forward test (SPRT H1 {_shr['sprt']['h1_r']:+.2f}R) expects ~"
            f"{_shr['sprt']['expected_trades_if_h1']:.0f} trades to confirm.")
        _delta = float(_shr["observed_mean_r"]) - float(_shr["shrunk_mean_r"])
        if (pooled_mc is not None and p_hold is not None and _delta > 1e-6 and _shr["n"] >= int(getattr(cfg, "shrink_pass_min_unseen_trades", 100))
                and getattr(cfg, "shrink_adjust_pass_rate", True)):
            try:
                _adj_trades = _shifted_trades(list(final_trades or []) + list(hold_trades), _delta)
                _p_adj, _ = holdout_pass_rate_mc(_adj_trades, prop_rules, sims=int(getattr(cfg, "holdout_mc_sims", 2000)) // 2 or 1000,
                                                 seed=getattr(cfg, "random_seed", 7) or 7)
                if _p_adj is not None:
                    evidence["edge_shrinkage"]["adjusted_pass_pct"] = _p_adj * 100.0
                    if _p_adj < p_hold:                           # only ever lowers
                        log(f"  Pass rate re-simulated with the edge shrunk by {_delta:.3f}R per trade: {p_hold * 100:.1f}% -> {_p_adj * 100:.1f}% (EV uses the lower).")
                        p_hold = float(_p_adj)
                        evidence["pass_rate_shrunk_for_ev"] = True
            except Exception as exc:  # noqa: BLE001
                evidence["edge_shrinkage_error"] = str(exc)

    # 6. EV verdict beside READY
    bankroll_trades = (list(final_trades or []) + list(hold_trades)) if pooled_mc is not None else hold_trades
    ev = _compute_ev(verdict, cfg, prop_rules, payout_ev, p_hold, bankroll_trades, evidence, int(n_trials), n_eff,
                     lookahead_ok=not lookahead_check_crashed, ok_rules=ok, log=log, parameters_tuned=parameters_tuned,
                     selection_handled=selection_handled, pass_lb_pct=(evidence.get("holdout_mc") or {}).get("pass_lb_pct"),
                     pass_interval_method=(evidence.get("holdout_mc") or {}).get("pass_interval_method"))
    # 7. EV hard bar (2026-10-20): the fixed per-attempt 70% bar is a HIGH CONFIDENCE label, not the rejecting test. What rejects
    #    is a computable economics failure: EV <= 0 or below margin with every input present -> NOT READY; missing inputs cap at MARGINAL.
    if getattr(cfg, "ev_hard_bar", True):
        if ev.label == "NOT WORTH A FEE" and not ev.inputs_missing:
            demote("NOT READY", "EV TEST FAILED: " + " ".join(ev.reasons[-2:]) + " The strategy does not earn back the fee at its measured, "
                                "deflated pass rate -- no pass-rate bar is needed to see that.")
        elif ev.label == "INSUFFICIENT EVIDENCE":
            demote("MARGINAL", "EV EVIDENCE INCOMPLETE: " + " ".join(ev.reasons[:1]) + " READY needs an economic verdict.")
    # 8. ladder stage printed next to the verdict everywhere
    from app.orchestration.verdict_ladder import ladder_stage
    _edge_ev = edge_evidence_for_ladder(confirmation, null_ev, cost_ev, ev, cfg, verdict)
    evidence["edge_evidence"] = _edge_ev
    _lad = ladder_stage(verdict, ev.label, pooled_status=evidence.get("pooled_status"),
                        hard_failed=(verdict == "NOT READY" and ev.label == "NOT WORTH A FEE"),
                        per_attempt_pass_lb_pct=None, high_confidence_pct=min_pct,
                        edge_evidence=_edge_ev,
                        caps=[r for r in reasons if isinstance(r, str) and ("CAPPED" in r or "NOT PROVEN" in r.upper())][:4])
    evidence["ladder"] = _lad.to_dict()
    try:
        from app.orchestration.deployment_checklist import build_checklist
        from app.prop.presets import get_fee_info
        evidence["deployment_checklist"] = build_checklist(evidence, fee=get_fee_info(getattr(prop_rules, "preset_key", None) or ""), ladder_stage=_lad.stage)
    except Exception as exc:  # noqa: BLE001
        evidence["deployment_checklist_error"] = str(exc)
    return verdict, reasons, ev, evidence


def _compute_ev(verdict, cfg, prop_rules, payout_ev, p_hold, hold_trades, evidence, n_trials, n_eff, *, lookahead_ok, ok_rules, log,
                parameters_tuned: bool = True, selection_handled: bool = False, pass_lb_pct=None, pass_interval_method=None) -> EVVerdict:
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
    # 2026-10-19: the 30-day window is a tracked metric, not the EV driver. A strategy that pays out in 45 days is
    # still worth the fee, so EV uses the longer-horizon per-attempt paid probability (same holdout) when it is higher.
    if fp > 0 and p_hold is not None:
        _long = min(fp, float(p_hold))
        evidence["ev_payout_basis"] = {"window_rate": r30, "long_horizon_rate": _long}
        r30 = _long if r30 is None else max(float(r30), _long)
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
    from app.orchestration.ev_verdict import billing_profile as _billing_profile, evaluation_months
    _sub = bool(fee.monthly_subscription) if fee else False
    _period = int(getattr(fee, "billing_period_days", 30)) if fee else 30
    # 2026-11-04 (Fix 1): bill every simulated attempt for the CALENDAR days IT lived (fast failures are cheap, slow passes cost more);
    # fall back to the scalar median only when per-attempt records are missing. The median fallback uses calendar days when available.
    _profile = _billing_profile(hm.get("attempt_outcomes") or [], hm.get("attempt_calendar_days") or [], _period) if _sub else None
    _cal = hm.get("median_calendar_days_to_pass")
    _months = (evaluation_months(_cal, _period, calendar_days=True) if _cal
               else evaluation_months(hm.get("median_days_to_pass"), _period)) if _sub else 1
    evidence["subscription_months"] = _months if _sub else None
    evidence["billing_profile"] = _profile
    # the pass-rate lower bound is only a true edge-uncertainty bound when it came from the outer bootstrap; the legacy Wilson bound
    # conflates slowness with weakness and is not used for the EV bar.
    _p_lb = (float(pass_lb_pct) / 100.0) if (pass_lb_pct is not None and pass_interval_method == "outer_bootstrap") else None
    ev = evaluate_ev(
        challenge_months=_months, monthly_subscription=_sub,
        legacy_verdict=verdict, holdout_pass_rate=p_hold, window_payout_rate=r30,
        payout_value=payout_value, challenge_price=fee.challenge_price if fee else None,
        reset_price=fee.reset_price if fee else None, profit_split_pct=fee.profit_split_pct if fee else None,
        activation_fee=(fee.activation_fee if fee else 0.0),
        bankroll_ruin_pct=ruin, ruin_cap_pct=float(getattr(cfg, "ev_ruin_cap_pct", getattr(cfg, "risk_of_ruin_cap", 20.0))),
        margin_multiple=float(getattr(cfg, "ev_margin_multiple", 1.5)), margin_points=float(getattr(cfg, "ev_margin_points", 0.05)),
        n_trials=n_trials, n_effective=n_eff, rules_verified=ok_rules, lookahead_ok=lookahead_ok,
        holdout_clean=bool(evidence.get("holdout_reserved") and evidence.get("holdout_ok")), parameters_tuned=parameters_tuned,
        automation_confirmed=fee.written_confirmation if fee else None,
        holdout_pass_rate_lb=_p_lb, billing_profile=_profile, selection_handled_by_holdout=selection_handled,
    )
    if fee is None:
        ev.reasons.append("No fee row for this preset in app.prop.presets.FIRM_FEES (or hand-typed rules): add one with a source URL and date.")
    return ev
