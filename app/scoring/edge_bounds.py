"""Sample-uncertainty helpers for the verdict (2026-10-17 audit, items 4 and 5).

Two separate problems, both caused by treating a finite trade list as if it were the true edge:

1. ``sample_aware_interval`` -- the Monte Carlo's "95% interval" on pass / paid probability is a
   Wilson interval over ~10,000 SIMULATED attempts, so it is about +/-1 point wide and says nothing
   about whether the 135 or 267 trades that fed the simulation represent the real edge. The number
   of attempts that are genuinely independent is bounded by how many non-overlapping attempt
   windows the source history can hold, so the interval is taken over THAT count.

2. ``expectancy_bound`` -- a lower confidence bound on average R per trade (t-based and bootstrap,
   whichever is more conservative), used as a hard gate after 2x costs instead of leaning on a
   4%-weight score component.

Pure functions, numpy only, no app imports -- safe to use from the scorecard, the pipeline and tests.
"""
from __future__ import annotations

import math

import numpy as np

# Two-sided 95% critical values used when scipy is not wanted here.
DEFAULT_TRADES_PER_ATTEMPT = 10.0
_Z95 = 1.959964
_Z90_ONE_SIDED = 1.644854


def wilson_interval(successes: float, n: float, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval on a proportion, returned as (lo, hi) in [0, 1]. ``n`` may be fractional
    (an effective sample size). n <= 0 returns the uninformative (0, 1)."""
    if n is None or n <= 0:
        return (0.0, 1.0)
    p = max(0.0, min(1.0, successes / n))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def effective_independent_attempts(
    n_source_trades: int,
    n_source_calendar_days: int,
    attempt_days: float | None = None,
    trades_per_attempt: float | None = None,
    n_simulated_attempts: float | None = None,
) -> float:
    """How many statistically independent attempts the source history can support.

    Two ceilings, take the smaller:
      * calendar -- history days / typical attempt length (default 30 days, the payout window);
      * trades   -- source trades / trades one attempt consumes (default: 10; was 20 before the 2026-10-27 audit, which showed
        trades/20 left 593 trades at ~24 independent attempts and capped a planted 0.2R edge's pass-probability bound near 86%).
    Never more than the simulated count, and never below 1."""
    ad = float(attempt_days) if attempt_days and attempt_days > 0 else 30.0
    n_cal = float(n_source_calendar_days) / ad if n_source_calendar_days and n_source_calendar_days > 0 else None
    tpa = float(trades_per_attempt) if trades_per_attempt and trades_per_attempt > 0 else DEFAULT_TRADES_PER_ATTEMPT
    n_tr = float(n_source_trades) / tpa if n_source_trades and n_source_trades > 0 else None
    cands = [c for c in (n_cal, n_tr) if c is not None]
    if not cands:
        return 1.0
    n_eff = min(cands)
    if n_simulated_attempts is not None and n_simulated_attempts > 0:
        n_eff = min(n_eff, float(n_simulated_attempts))
    return max(1.0, n_eff)


def sample_aware_interval(
    p_pct: float,
    n_source_trades: int,
    n_source_calendar_days: int,
    attempt_days: float | None = None,
    trades_per_attempt: float | None = None,
    n_simulated_attempts: float | None = None,
) -> tuple[tuple[float, float], float]:
    """((lo_pct, hi_pct), n_eff): a 95% Wilson interval on a probability given in PERCENT, taken over the
    effective number of independent attempts rather than the number of simulated ones."""
    n_eff = effective_independent_attempts(
        n_source_trades, n_source_calendar_days, attempt_days, trades_per_attempt, n_simulated_attempts)
    p = max(0.0, min(100.0, float(p_pct))) / 100.0
    lo, hi = wilson_interval(p * n_eff, n_eff)
    return (lo * 100.0, hi * 100.0), n_eff


def r_multiples(trades) -> list[float]:
    """R multiples exactly as app.backtest.statistics defines them: pnl / (planned initial risk * size),
    for trades that carry a planned risk. Falls back to nothing (empty list) when none do."""
    out: list[float] = []
    for t in trades or []:
        ir = getattr(t, "initial_risk", None)
        size = getattr(t, "size", None)
        pnl = getattr(t, "pnl", None)
        if pnl is None or not ir or ir <= 0 or not size:
            continue
        out.append(float(pnl) / (float(ir) * float(size)))
    return out


def expectancy_bound(
    r: list[float] | np.ndarray,
    conf: float = 0.95,
    n_boot: int = 2000,
    seed: int = 7,
) -> dict:
    """One-sided lower and upper confidence bounds on mean R per trade.

    ``lb`` is the MORE CONSERVATIVE of the t-based bound and the bootstrap percentile bound, so a skewed
    sample (a few outsized winners) cannot flatter it. Returns n=0 and None bounds for fewer than 5 trades.
    """
    _src = [] if r is None else list(np.asarray(r, dtype=object).ravel())
    a = np.asarray([x for x in _src if x is not None and np.isfinite(x)], dtype=float)
    n = int(a.size)
    if n < 5:
        return {"n": n, "mean": float(a.mean()) if n else None, "std": None, "lb": None, "ub": None,
                "lb_t": None, "lb_boot": None, "conf": conf}
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    # one-sided critical value for the requested confidence (95% -> 1.645)
    z = _Z90_ONE_SIDED if abs(conf - 0.95) < 1e-9 else float(_norm_ppf(conf))
    se = sd / math.sqrt(n)
    # small-sample t correction (Cornish-Fisher style, good to ~1% for n >= 10)
    z_t = z + (z ** 3 + z) / (4.0 * max(n - 1, 1))
    lb_t, ub_t = mean - z_t * se, mean + z_t * se
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = a[idx].mean(axis=1)
    lb_b = float(np.percentile(means, (1.0 - conf) * 100.0))
    ub_b = float(np.percentile(means, conf * 100.0))
    return {"n": n, "mean": mean, "std": sd, "lb": float(min(lb_t, lb_b)), "ub": float(max(ub_t, ub_b)),
            "lb_t": float(lb_t), "lb_boot": lb_b, "conf": conf}


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's approximation, |err| < 1.2e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p = min(max(p, 1e-12), 1 - 1e-12)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r_ = q * q
    return (((((a[0]*r_+a[1])*r_+a[2])*r_+a[3])*r_+a[4])*r_+a[5])*q / (((((b[0]*r_+b[1])*r_+b[2])*r_+b[3])*r_+b[4])*r_+1)


# ---------------------------------------------------------------------------------------------------------
# 2026-10-20: empirical-Bayes shrinkage toward the strategy's FAMILY (report section 4). A 100-trade +0.4R winner from a family whose
# strategies usually deliver ~0R is mostly luck; shrink it, then size from the shrunk LOWER bound, never the point estimate.
# ---------------------------------------------------------------------------------------------------------
DEFAULT_FAMILY_PRIOR = (0.0, 0.15)   # skeptical: the typical strategy earns 0R, true means spread ~0.15R


def family_prior_from_results(means: list, sds: list, ns: list) -> tuple[float, float]:
    """(mu0, tau) from the family's recorded strategies (method of moments); the skeptical default when there is nothing to learn from."""
    from app.validation.edge_stats import estimate_family_prior
    if not means or len(means) < 2:
        return DEFAULT_FAMILY_PRIOR
    return estimate_family_prior(means, sds, ns)


def shrunk_expectancy_bound(r, mu0: float | None = None, tau: float | None = None, conf: float = 0.95) -> dict:
    """Expectancy bound AFTER shrinkage toward (mu0, tau). Returns the raw bound fields plus shrunk_mean / shrunk_lb / prior."""
    from app.validation.edge_stats import empirical_bayes_shrink, shrunk_lower_bound
    base = expectancy_bound(r, conf=conf)
    if base.get("n", 0) < 5 or base.get("std") in (None, 0):
        return {**base, "shrunk_mean": None, "shrunk_lb": None, "prior": None}
    if mu0 is None or tau is None:
        mu0, tau = DEFAULT_FAMILY_PRIOR
    z = _Z90_ONE_SIDED if abs(conf - 0.95) < 1e-9 else float(_norm_ppf(conf))
    return {**base, "shrunk_mean": empirical_bayes_shrink(base["n"], base["mean"], base["std"], mu0, tau),
            "shrunk_lb": shrunk_lower_bound(base["n"], base["mean"], base["std"], mu0, tau, z=z), "prior": {"mu0": mu0, "tau": tau}}
