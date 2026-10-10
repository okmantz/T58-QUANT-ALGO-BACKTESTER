"""v9.15: THE canonical Monte Carlo display derivation.

Every surface that shows an "eval pass probability" or "first payout
probability" -- the Full Pipeline job page tiles, the HTML/JSON report
headline cards, the Dashboard/Champion scorecards fed from run history,
speed-run summaries -- must derive the number from THIS one function so
the surfaces can never disagree with each other again.

Why this exists (Owen, 2026-10-10): his NQ pipeline page read
"EVAL PASS PROBABILITY 90.0%" while the same run's report read 8.7%.
Both numbers were real MonteCarloResult fields answering different
questions:

- ``per_attempt_pass_probability`` -- passed evals / attempted evals
  across every independent account attempt. This is the metric the
  READY verdict gates on (lower bound of ``per_attempt_pass_ci95``),
  and the only one that answers "if I buy ONE account, what happens".
- ``evaluation_pass_probability`` -- "did >=1 attempt in a whole
  mechanical rebuy chain ever pass". With reset_on_breach on and
  hundreds of rebuys per path it inflates toward 100% almost by
  construction. It may only ever be displayed under the explicit
  label "any attempt in chain".

Display rule (mirrors MonteCarloResult.headline_* and the v6 gating
fallback): headline = per-attempt; if a result predates the per-attempt
fields (``total_independent_attempts`` == 0 and per-attempt is the 0.0
default while a chain estimate exists), fall back to the chain fields
so an old report never renders a fake 0.0%.

Nothing here changes any computation -- this module only decides which
already-computed field a display may call "the" probability.
"""
from __future__ import annotations


def _f(d: dict, key: str) -> float:
    try:
        return float(d.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ci(d: dict, key: str):
    v = d.get(key)
    try:
        lo, hi = float(v[0]), float(v[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if lo == 0.0 and hi == 0.0:  # (0, 0) == "unknown, not zero"
        return None
    return [lo, hi]


def monte_carlo_headline(mc: dict | None) -> dict:
    """Canonical display values for one Monte Carlo result dict (the
    shape MonteCarloResult.to_dict() produces)."""
    mc = mc or {}
    chain_pass = _f(mc, "evaluation_pass_probability")
    chain_payout = _f(mc, "first_payout_probability")
    per_pass = _f(mc, "per_attempt_pass_probability")
    per_payout = _f(mc, "per_attempt_payout_probability")
    attempts = int(mc.get("total_independent_attempts", 0) or 0)

    if attempts == 0 and per_pass == 0.0 and chain_pass > 0.0:
        # Result built before the per-attempt fields existed -- the
        # chain fields are the only estimate there is.
        eval_pass, payout, basis = chain_pass, chain_payout, "chain_fallback"
        eval_ci = payout_ci = None
    else:
        eval_pass, payout, basis = per_pass, per_payout, "per_attempt"
        eval_ci = _ci(mc, "per_attempt_pass_ci95")
        payout_ci = _ci(mc, "per_attempt_payout_ci95")

    return {
        # The headline pair: per-attempt (the gate metric).
        "eval_pass_probability": eval_pass,
        "eval_pass_basis": basis,
        "eval_pass_ci95": eval_ci,
        "first_payout_probability": payout,
        "first_payout_ci95": payout_ci,
        # Chain-level pair: display ONLY labeled "any attempt in chain".
        "any_attempt_eval_pass_probability": chain_pass,
        "any_attempt_first_payout_probability": chain_payout,
        "risk_of_ruin_pct": _f(mc, "risk_of_ruin_pct"),
        "total_independent_attempts": attempts,
    }
