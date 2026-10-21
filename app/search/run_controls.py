"""Random-entry / shuffled-signal control for every Search Lab run (report 2026-10-20, section 3 and item 7).

The question a search must answer about ITSELF: "if I had searched with no signal at all, how many candidates would my Stage-1 rule
have passed, and how good would the best one have looked?" Selection bias is the dominant error in a large search - the best of 2,000
noise candidates always looks like an edge. This control draws random-entry candidates (random entry times, 50/50 sides, the median
stop/target and hold of the real candidates, the real costs) and applies the SAME Stage-1 rule. The result is published with the run:

  random_pass_rate          share of random-entry draws that pass Stage 1 (the false-positive rate of the filter)
  random_t_p95              95th percentile of the mean-R t-stat among random draws
  real_pass_rate            share of real Stage-1 candidates that passed
  finalists[i].beats_random the finalist's t-stat is above the random p95 (a finalist that cannot beat the best 5% of noise is noise)

Nothing here changes which candidates advance; it adds an honest yardstick next to them (and a CONTROL FAILED warning when the
filter is passing random entries about as often as real candidates).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd


def _median(values, default):
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.median(v)) if v else default


def _stats_for_signal(df, sig, risk, stop_pips, target_pips):
    from app.backtest.execution import run_execution
    from app.backtest.statistics import compute_statistics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trades, eq = run_execution(df=df, signals=pd.Series(sig, index=df.index), risk=risk, stop_loss_pips=stop_pips,
                                   take_profit_pips=target_pips, stop_loss_distance=None, take_profit_distance=None)
    if not trades:
        return None
    return compute_statistics(trades, eq, initial_balance=risk.initial_balance).to_dict()


def random_entry_control(df, risk, prop_rules, *, stop_pips: float, target_pips: float | None, hold_bars: int, n_entries: int,
                         filters: dict, n_draws: int = 30, seed: int = 20261020, log=lambda m: None) -> dict:
    """Run ``n_draws`` random-entry candidates through the Stage-1 rule. Returns the control summary (never raises)."""
    from app.research.director import _null_draw_signal
    from app.search.batch_runner import _mean_r_t_stat, _stage1_score
    rng = np.random.default_rng(seed)
    n = len(df)
    ts, passed, rows = [], 0, []
    for k in range(int(n_draws)):
        try:
            sig = _null_draw_signal(rng, n, max(1, int(hold_bars)), max(5, int(n_entries)), 0.5)
            st = _stats_for_signal(df, sig, risk, stop_pips, target_pips)
            if not st:
                continue
            rec = _stage1_score({"candidate_id": f"rand{k}"}, st, True, filters, prop_rules)
            t = _mean_r_t_stat(st)
            ts.append(t)
            passed += int(bool(rec.get("passed_stage1")))
            rows.append({"n_trades": int(st.get("total_trades", 0)), "average_r": float(st.get("average_r", 0.0) or 0.0), "t": float(t)})
        except Exception as exc:  # noqa: BLE001 - a control must never break a search
            log(f"  control draw {k} failed: {exc}")
    finite = [t for t in ts if math.isfinite(t)]
    return {
        "n_draws": len(ts),
        "random_pass_rate": (passed / len(ts)) if ts else None,
        "random_t_p95": float(np.percentile(finite, 95)) if finite else None,
        "random_t_max": float(max(finite)) if finite else None,
        "random_mean_r": float(np.mean([r["average_r"] for r in rows])) if rows else None,
        "stop_pips": float(stop_pips), "target_pips": None if target_pips is None else float(target_pips),
        "hold_bars": int(hold_bars), "n_entries": int(n_entries),
    }


def search_controls(df, risk, prop_rules, stage1_records: list, specs_by_id: dict, filters: dict, *, finalists: list | None = None,
                    n_draws: int = 30, log=lambda m: None) -> dict | None:
    """Build the control for a finished Stage 1: representative stop/target/hold/trade-count from the real candidates."""
    try:
        from app.search.batch_runner import _candidate_median_stop_pips, _mean_r_t_stat
        scored = [r for r in stage1_records if isinstance(r.get("statistics"), dict) and (r["statistics"].get("total_trades") or 0) > 0]
        if len(scored) < 5:
            return None
        stops, rrs, holds, ntr = [], [], [], []
        for r in scored[:200]:
            spec = specs_by_id.get(r.get("candidate_id")) or {}
            sp = _candidate_median_stop_pips(spec, risk, df)
            rm = ((spec.get("config") or {}).get("risk_management") or {})
            if sp:
                stops.append(sp)
                try:
                    if rm.get("target_value") and rm.get("stop_value"):
                        rrs.append(float(rm["target_value"]) / float(rm["stop_value"]))
                except (TypeError, ValueError):
                    pass
            if rm.get("max_bars_in_trade"):
                holds.append(rm["max_bars_in_trade"])
            ntr.append(r["statistics"].get("total_trades"))
        if not stops:
            return None
        stop = _median(stops, 0.0)
        rr = _median(rrs, 2.0)
        ctl = random_entry_control(df, risk, prop_rules, stop_pips=stop, target_pips=stop * rr, hold_bars=int(_median(holds, 24)),
                                   n_entries=int(_median(ntr, 100)), filters=filters, n_draws=n_draws, log=log)
        real_pass = sum(1 for r in stage1_records if r.get("passed_stage1"))
        ctl["real_pass_rate"] = real_pass / max(1, len(stage1_records))
        ctl["n_real_candidates"] = len(stage1_records)
        p95 = ctl.get("random_t_p95")
        ctl["finalists"] = [{"candidate_id": f.get("candidate_id"), "t_stat": float(_mean_r_t_stat(f.get("statistics") or {})),
                             "beats_random": bool(p95 is not None and _mean_r_t_stat(f.get("statistics") or {}) > p95)}
                            for f in (finalists or [])]
        rp = ctl.get("random_pass_rate")
        ctl["control_failed"] = bool(rp is not None and ctl["real_pass_rate"] > 0 and rp >= 0.5 * ctl["real_pass_rate"] and rp >= 0.05)
        return ctl
    except Exception as exc:  # noqa: BLE001
        log(f"  random-entry control skipped: {exc}")
        return None


def control_log_lines(ctl: dict | None) -> list[str]:
    if not ctl:
        return ["Random-entry control: not run (too few scored candidates)."]
    rp = ctl.get("random_pass_rate")
    lines = [
        f"Random-entry control ({ctl['n_draws']} draws, same stops/costs): {0 if rp is None else rp * 100:.0f}% of RANDOM candidates pass Stage 1 "
        f"vs {ctl.get('real_pass_rate', 0.0) * 100:.0f}% of the {ctl.get('n_real_candidates', 0)} real ones; random t-stat 95th pct = "
        f"{'n/a' if ctl.get('random_t_p95') is None else format(ctl['random_t_p95'], '.2f')}."
    ]
    if ctl.get("control_failed"):
        lines.append("  CONTROL FAILED: Stage 1 passes random entries almost as often as real candidates - survivors are selection noise until proven otherwise.")
    for f in ctl.get("finalists", [])[:5]:
        lines.append(f"  {f['candidate_id']}: t={f['t_stat']:.2f} -> {'beats' if f['beats_random'] else 'does NOT beat'} the best 5% of random entries")
    return lines
