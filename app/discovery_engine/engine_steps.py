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


def _strategy(spec: dict):
    """Live Strategy for a candidate spec. The temp dir (python candidates load lazily from it) lives until process exit."""
    from app.search.strategy_space import build_strategy_from_spec
    tmp = Path(tempfile.mkdtemp(prefix="t58_promo_"))
    _TMP_DIRS.append(tmp)
    return build_strategy_from_spec(spec, tmp)


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
                holdout_budget=None, null_runner=None) -> Steps:
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
        if holdout_budget is not None:
            sealed = holdout_budget.read_sealed(sealed_df, who="promotion")
        else:
            sealed = sealed_df
        from app.optimize.two_stage import trade_r_and_days
        tr = evaluate_spec(c.get("spec"), sealed, risk, cfg.cost_multiplier)
        rs, days = trade_r_and_days(tr) if tr else ([], [])
        return sealed_step(rs, days, tied=ctx["tied"], cfg=cfg)

    return Steps(reproduction=reproduction, sequence_deflated=sequence_deflated, null_gate=null_gate,
                 pooled_edge=pooled_edge, sealed_slice=sealed_slice)
