"""
v7 (2026-10-05, worker B, workstream D) -- honest gate ledger for Quick
Optimize.

Quick Optimize is deliberately a lighter tool than Full Pipeline: it runs
Full Pipeline's Step 2 (the walk-forward-aware GA) in isolation, plus a
best-effort significance gate and a lookahead recheck, but NOT the
independent OOS holdout re-verification, DSR/PBO, or CPCV. A strong
Quick Optimize number used to be indistinguishable from a validated one
-- this module builds the explicit ran/skipped ledger the report and the
web job status page render, so "what was and wasn't validated" is never
left to inference.

Pure function of a QuickOptimizeResult's public fields (duck-typed -- no
import of app.orchestration.quick_optimize, so this stays importable
without the orchestration tree or Flask).
"""

from __future__ import annotations


def quickopt_gates_summary(result) -> dict:
    """Returns {"ran": [...], "skipped": [...]} for a QuickOptimizeResult.

    `result` is duck-typed on the QuickOptimizeResult fields
    (icir_gate, icir_gate_skip_reason, lookahead_note,
    holdout_enabled). Never raises -- a ledger that can't be built
    degrades to a conservative all-skipped entry rather than crashing a
    report render.
    """
    ran: list[str] = []
    skipped: list[str] = []
    try:
        ran.extend([
            "Walk-forward-scored GA fitness (chained out-of-sample folds -- "
            "the GA never scores on in-sample data)",
            "Prop-rule-aware scoring (per-attempt eval-pass / payout odds "
            "through the prop simulator, not raw in-sample profit factor)",
            "Instrument-scale mismatch escalation (pip_size sanity check)",
            "READY acceptance gate (>=70% per-attempt pass probability, no "
            "confirmed lookahead leak, credible OOS trade count)",
        ])
        skipped.extend([
            "Full Pipeline's independent out-of-sample holdout re-verification",
            "DSR / PBO overfitting gates",
            "CPCV (combinatorial purged cross-validation)",
            "Walk-forward-starved MARGINAL cap",
        ])
        if getattr(result, "icir_gate", None) is not None:
            ran.append(
                "ICIR / signal-decay / Bonferroni significance gate "
                "(best-effort, on the search data)"
            )
        else:
            skipped.append(
                "ICIR / signal-decay / Bonferroni significance gate "
                f"(could not run: {getattr(result, 'icir_gate_skip_reason', None) or 'too few trades'})"
            )
        if getattr(result, "lookahead_note", None):
            ran.append("Lookahead-bias recheck on the final configuration")
        else:
            skipped.append("Lookahead-bias recheck (did not run)")
        if getattr(result, "holdout_enabled", False):
            ran.append(
                "Light holdout check (one plain backtest on the reserved "
                "tail -- NOT re-verified; not a substitute for Full Pipeline)"
            )
        else:
            skipped.append("Holdout check (not requested for this run)")
    except Exception:  # noqa: BLE001 -- ledger must never crash a report
        skipped.append("Gate ledger could not be built for this result.")
    return {"ran": ran, "skipped": skipped}
