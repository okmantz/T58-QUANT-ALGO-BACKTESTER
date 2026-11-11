"""Real (backtest-engine) implementations of the promotion steps, plus helpers shared by the labs.

``build_steps(...)`` returns a ``promotion.Steps`` wired to the actual engine; the labs call it once per run and hand it to
``run_promotion_sequence`` / ``finalize_loop_stop``.
"""
from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from app.discovery_engine import objective as obj
from app.discovery_engine import provenance as prov
from app.discovery_engine.pool_stats import daily_series_from_trades
from app.discovery_engine.promotion import PromotionConfig, StepResult, Steps, deflated_step, pooled_step, sealed_step


_TMP_DIRS: list = []


def _cleanup_tmp() -> None:
    for d in _TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_tmp)


_STRATEGY_CACHE: dict = {}
_STRATEGY_CACHE_MAX = 64


def _strategy(spec: dict):
    """Live Strategy for a candidate spec. The temp dir (python candidates load lazily from it) lives until process exit.

    Memoised by the spec's canonical JSON: the promotion steps evaluate the same champion on the dev slice, every pooled market and
    the sealed slice, and each used to write a fresh temp dir and re-import the candidate (disk churn and import time per call).
    A strategy object is a pure function of its spec, so sharing one across read-only backtests is safe."""
    import json
    from app.search.strategy_space import build_strategy_from_spec
    try:
        key = json.dumps(spec, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 - unhashable spec: just build it
        key = None
    if key is not None and key in _STRATEGY_CACHE:
        return _STRATEGY_CACHE[key]
    tmp = Path(tempfile.mkdtemp(prefix="t58_promo_"))
    _TMP_DIRS.append(tmp)
    strat = build_strategy_from_spec(spec, tmp)
    if key is not None:
        if len(_STRATEGY_CACHE) >= _STRATEGY_CACHE_MAX:
            _STRATEGY_CACHE.pop(next(iter(_STRATEGY_CACHE)))
        _STRATEGY_CACHE[key] = strat
    return strat


def evaluate_spec(spec: dict, df: pd.DataFrame, risk, cost_multiplier: float = 1.0):
    """Backtest ``spec`` on ``df`` at ``cost_multiplier`` x costs. Returns the trade list ([] on any failure -- no evidence is not a pass)."""
    from app.backtest.engine import run_backtest
    from app.optimize.two_stage import stress_risk
    try:
        rk = stress_risk(risk, cost_multiplier) if cost_multiplier and cost_multiplier != 1.0 else risk
        return list(run_backtest(df, _strategy(spec), rk, lookahead_check="skip").trades)
    except Exception:  # noqa: BLE001
        return []


def edge_record(trades, spec: dict | None = None) -> dict:
    """{edge_lb_r, mean_r, n_trades, t_stat} -- what provenance records and what reproduction compares."""
    sc = obj.score_edge(trades, spec)
    return {"edge_lb_r": None if not np.isfinite(sc.edge_lb_r) else float(sc.edge_lb_r), "mean_r": sc.mean_r,
            "n_trades": sc.n_trades, "t_stat": sc.t_stat}


def thirty_day_record(trades, risk, rules, df, *, label: str = "", instrument: str = "") -> dict | None:
    """Score a champion's trades on the 30-day prop objective: P(pass <= 30 calendar days of purchase) under ``rules``, with the
    dependence-aware 95% interval (``app.prop.thirty_day.evaluate_thirty_day``). Returns a JSON-safe dict, or ``None`` when it cannot be
    measured (no trades, no rules, a ledger that cannot be built). Dollars are the champion's own account dollars.

    Measured MAE/MFE are required by floating-drawdown rules; if a trade list carries none the closed-P&L proxy is used and the record says
    so (``excursions_proxy``) -- it is optimistic, never silent."""
    if rules is None or not trades:
        return None
    try:
        from app.prop.thirty_day import ExcursionsMissingError, ThirtyDayConfig, TradeLedger, evaluate_thirty_day, rank_key
        led = TradeLedger.from_trades(list(trades), risk, strategy=label, instrument=instrument)
        if not len(led):
            return None
        ts = pd.to_datetime(df["timestamp"]) if df is not None and "timestamp" in getattr(df, "columns", []) else None
        d0 = ts.iloc[0] if ts is not None and len(ts) else None
        d1 = ts.iloc[-1] if ts is not None and len(ts) else None
        proxy = False
        try:
            res = evaluate_thirty_day(led, rules, data_start=d0, data_end=d1, cfg=ThirtyDayConfig(), with_sensitivity=False)
        except ExcursionsMissingError:
            proxy = True
            res = evaluate_thirty_day(led, rules, data_start=d0, data_end=d1, cfg=ThirtyDayConfig(allow_proxy_excursions=True), with_sensitivity=False)
        notes = list(res.notes)
        acct = float(getattr(rules, "account_size", 0.0) or 0.0)
        bal = float(getattr(risk, "initial_balance", 0.0) or 0.0)
        if acct > 0 and bal > 0 and abs(acct - bal) / acct > 0.01:
            notes.append(f"the strategy was sized on a ${bal:,.0f} account but scored on a ${acct:,.0f} evaluation: dollars are NOT rescaled.")
        return {"p_pass_30": res.p_pass_30, "lower_bound": res.lower_bound, "upper_bound": float(res.interval.get("upper", 1.0)),
                "p_bust_30": res.p_bust_30, "p_pass_60": res.p_pass_60, "median_days_to_pass": res.median_days_to_pass,
                "n_windows": res.n_windows, "n_independent": float(res.interval.get("n_independent", 0.0)),
                "meets_target_lower_bound": bool(res.meets_target_lower_bound), "target_probability": res.target_probability,
                "excursions_proxy": proxy, "rank_key": [float(x) for x in rank_key(res)], "notes": notes}
    except Exception:  # noqa: BLE001 -- the 30-day record is extra evidence; it never fails a run
        return None


def per_market_r(spec: dict, frames: dict, risks: dict, cost_multiplier: float = 2.0) -> dict:
    """Run the candidate's FIXED config on every market -> {symbol: (R list, day list)}. A market that fails is reported empty."""
    from app.optimize.two_stage import trade_r_and_days
    out = {}
    for sym, df in (frames or {}).items():
        risk = (risks or {}).get(sym)
        if df is None or risk is None or not len(df):
            out[sym] = ([], [])
            continue
        tr = evaluate_spec(spec, df, risk, cost_multiplier)
        out[sym] = trade_r_and_days(tr) if tr else ([], [])
    return out


def build_steps(*, dev_df: pd.DataFrame, sealed_df: pd.DataFrame | None, risk, frames: dict | None = None, risks: dict | None = None,
                pool_matrix: np.ndarray | None = None, n_trials: int = 1, cfg: PromotionConfig | None = None,
                holdout_budget=None, null_runner=None, sealed_ledger=None) -> Steps:
    """Wire the five steps to the real engine.

    ``frames`` / ``risks``: {SYMBOL: DataFrame} / {SYMBOL: RiskConfig} for the pooled test (the candidate's own market should be one
    of them). ``pool_matrix``: T x N daily results of the WHOLE evaluated pool (for the Sharpe spread). ``null_runner`` lets tests
    substitute the 200-seed null."""
    cfg = cfg or PromotionConfig()

    def reproduction(c, ctx):
        p = c.get("provenance")
        if p is None:
            return StepResult("reproduction", False, detail="no provenance recorded with this champion -- cannot be reproduced")
        p = p if isinstance(p, prov.Provenance) else prov.Provenance.from_dict(p)
        spec = c.get("spec")
        trades = evaluate_spec(spec, dev_df, risk, 1.0)
        rr = prov.verify_reproduction(p, edge_record(trades, spec), df=dev_df, abs_tol=cfg.repro_abs_tol, rel_tol=cfg.repro_rel_tol)
        return StepResult("reproduction", rr.reproduced, detail=rr.reason, data=rr.to_dict())

    def rulebook_g3(c, ctx):
        if str(getattr(cfg, "objective_mode", "prop")) == "net_profit":
            return StepResult("rulebook_g3", True, skipped=True, detail="not applicable: net-profit objective (prop-firm rules and Rulebook gates are not applied)")
        from app.validation import rulebook as rb
        trades = evaluate_spec(c.get("spec"), dev_df, risk, 1.0)
        g = rb.g3_development_evidence(trades, n_looks=max(1, int(n_trials)))
        # Neighbour robustness and lookahead are produced later (GA plateau probe, full pipeline); here a check that has not run yet is
        # PENDING, not failed -- but any check that ran and failed stops the candidate, and a pending check is named in the detail.
        pending = [k.name for k in g.checks if k.ok is None]
        failing = g.failing
        detail = (f"E_plan {g.data.get('e_plan', float('nan')):+.3f}R after N={int(n_trials)} looks (needs +0.05R; observed must reach "
                  f"{g.data.get('required_observed_edge', float('nan')):+.3f}R)" if g.data else "too few trades with a planned risk")
        if failing:
            detail = f"{failing[0].name}: {failing[0].detail}. " + detail
        if pending and not failing:
            detail += f" Pending: {'; '.join(pending)}."
        return StepResult("rulebook_g3", not failing, detail=detail, data={**(g.data or {}), "pending": pending, "failing": [f.name for f in failing]})

    def sequence_deflated(c, ctx):
        trades = evaluate_spec(c.get("spec"), dev_df, risk, 1.0)
        s = daily_series_from_trades(trades, unit="r")
        cal = s.index
        obs = s.reindex(pd.bdate_range(cal.min(), cal.max()) if len(cal) > 1 else cal).fillna(0.0).to_numpy() if len(cal) else np.zeros(0)
        return deflated_step(obs, pool_matrix, n_trials, min_psr=cfg.dsr_min_psr)

    def null_gate(c, ctx):
        runner = null_runner
        if runner is None:
            from app.validation.null_edge_gate import run_null_edge_gate
            runner = run_null_edge_gate
        r = runner(dev_df, _strategy(c.get("spec")), risk, n_seeds=cfg.null_seeds, p_max=cfg.null_p_max)
        return StepResult("null_gate", bool(r.passed), detail=r.render(), data=r.to_dict())

    def pooled_edge(c, ctx):
        per = per_market_r(c.get("spec"), frames or {}, risks or {}, cfg.cost_multiplier)
        return pooled_step(per, cfg)

    def sealed_slice(c, ctx):
        if sealed_df is None or not len(sealed_df):
            return StepResult("sealed_slice", False, detail="no sealed slice was reserved for this run")
        # Lifetime ledger (2026-11 audit): every read of this exact sealed slice, in any run, is a booked look. The k-th read must clear a
        # Bonferroni-spent confidence and past ``sealed_max_lifetime_looks`` the slice is refused -- it is development data by then.
        from app.discovery_engine.holdout import HoldoutExhausted, SealedLedger
        ledger = sealed_ledger if sealed_ledger is not None else SealedLedger(max_looks=cfg.sealed_max_lifetime_looks)
        fp = prov.data_fingerprint(sealed_df)
        try:
            looks = ledger.spend(fp)
        except HoldoutExhausted as exc:
            return StepResult("sealed_slice", False, detail=str(exc), data={"sealed_lifetime_looks": ledger.reads(fp), "refused": True})
        sealed = sealed_df                          # the per-run HoldoutBudget no longer caps a run's champions at one: each is a booked look
        if holdout_budget is not None:
            holdout_budget.sealed_reads += 1
            holdout_budget.log.append({"look": "sealed", "who": "promotion", "lifetime_looks": looks})
        from app.optimize.two_stage import trade_r_and_days
        tr = evaluate_spec(c.get("spec"), sealed, risk, cfg.cost_multiplier)
        rs, days = trade_r_and_days(tr) if tr else ([], [])
        return sealed_step(rs, days, tied=ctx["tied"], cfg=cfg, looks=looks)

    return Steps(reproduction=reproduction, rulebook_g3=rulebook_g3, sequence_deflated=sequence_deflated, null_gate=null_gate,
                 pooled_edge=pooled_edge, sealed_slice=sealed_slice)
