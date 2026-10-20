"""Power-aware edge statistics (2026-10-20 report, sections 4, 5 and 9).

Everything here measures the quantity that actually matters for a trading edge -- mean R per trade -- and says
how much evidence a sample of a given size can really carry. Pure functions, numpy only, no app imports, so the
verdict, the walk-forward gate, the forward-test harness and the tests can all share one implementation.

Contents
    trades_needed / standard_error_table   how many trades confirm an edge of a given size (power formula)
    day_block_bootstrap_mean               bootstrap of mean R resampling whole DAYS (keeps same-day clustering)
    merge_small_folds / pooled_oos_edge    the walk-forward / CPCV gate: pooled out-of-sample mean R
    edge_by_year / rolling_mean_r_slope    edge-decay diagnostics that replace the signal half-life check
    sprt_*                                 Wald sequential probability ratio test for the forward test
    empirical_bayes_shrink                 pull a small-sample winner toward its family mean
    sizing helpers                         lower-bound sizing so a lucky sample cannot raise risk

Nothing here promotes a strategy by itself. These are the instruments the gates use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# One-sided / two-sided critical values (avoid a scipy dependency).
Z_ONE_SIDED_95 = 1.6448536269514722
Z_TWO_SIDED_95 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
DEFAULT_TRADE_SD_R = 1.3          # per-trade standard deviation of R the report's simulations assume


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam, |err| < 1.2e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p = min(max(float(p), 1e-12), 1 - 1e-12)
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
# Power: how many trades does it take to see an edge?
# ---------------------------------------------------------------------------------------------------------

def trades_needed(effect_r: float, sd_r: float = DEFAULT_TRADE_SD_R, alpha: float = 0.05, power: float = 0.80,
                  two_sided: bool = False) -> int | None:
    """Trades required for a z-test of mean R > 0 to detect a true mean of ``effect_r`` at the given power.

        n = ((z_alpha + z_power) * sd / effect)^2

    One-sided 5% / 80% power at sd 1.3R: +0.10R -> ~1,040 trades, +0.21R -> ~240, +0.30R -> ~120. Two-sided at
    +0.21R -> ~300. Returns None when effect_r <= 0 (no sample size can confirm a non-positive edge)."""
    if effect_r is None or not math.isfinite(effect_r) or effect_r <= 0 or sd_r is None or sd_r <= 0:
        return None
    z_a = _norm_ppf(1.0 - (alpha / 2.0 if two_sided else alpha))
    z_b = _norm_ppf(power)
    return int(math.ceil(((z_a + z_b) * float(sd_r) / float(effect_r)) ** 2))


def standard_error_table(trade_counts=(19, 100, 300, 495, 1000), sd_r: float = DEFAULT_TRADE_SD_R) -> list[dict]:
    """Standard error of mean R and the 95% half-width for each trade count (the report's section-5 table)."""
    out = []
    for n in trade_counts:
        se = float(sd_r) / math.sqrt(max(int(n), 1))
        out.append({"trades": int(n), "se_r": se, "half_width_95_r": Z_TWO_SIDED_95 * se})
    return out


def trades_still_needed(mean_r: float | None, sd_r: float | None, n_have: int, alpha: float = 0.05,
                        power: float = 0.80) -> int | None:
    """Additional trades needed before an edge of ``mean_r`` (the observed or lower-bound estimate) can be confirmed.
    None when mean_r <= 0; 0 when the sample already has enough."""
    need = trades_needed(mean_r if mean_r is not None else 0.0, sd_r if sd_r else DEFAULT_TRADE_SD_R, alpha, power)
    if need is None:
        return None
    return max(0, need - int(n_have))


# ---------------------------------------------------------------------------------------------------------
# Day-block bootstrap
# ---------------------------------------------------------------------------------------------------------

def _day_codes(days) -> tuple[np.ndarray, int]:
    """Integer code per trade for its calendar day (any hashable / datetime-like), plus the number of distinct days."""
    arr = np.asarray(list(days), dtype=object)
    try:
        import pandas as pd
        arr = pd.to_datetime(pd.Series(arr), utc=True, errors="coerce").dt.tz_localize(None).dt.normalize().to_numpy()
        codes = np.unique(arr.astype("datetime64[D]"), return_inverse=True)[1]
    except Exception:  # noqa: BLE001 -- fall back to raw equality
        codes = np.unique(arr.astype(str), return_inverse=True)[1]
    return np.asarray(codes, dtype=int), int(codes.max()) + 1 if len(codes) else 0


def day_block_bootstrap_mean(r, days=None, conf: float = 0.95, n_boot: int = 2000, seed: int = 11) -> dict:
    """Bootstrap of mean R per trade that resamples whole trading DAYS with replacement.

    Trades taken on the same day are correlated (same regime, same news), so resampling single trades understates the
    uncertainty. ``days`` is a per-trade sequence of timestamps / dates; when None every trade is its own block (plain
    bootstrap). Returns {n, n_days, mean, std, lb, ub, conf}: ``lb`` is the one-sided lower bound at ``conf`` (5th
    percentile at 95%), ``ub`` the matching upper bound. n < 5 returns None bounds."""
    a = np.asarray([x for x in (r if r is not None else []) if x is not None and np.isfinite(x)], dtype=float)
    n = int(a.size)
    if n < 5:
        return {"n": n, "n_days": 0, "mean": float(a.mean()) if n else None, "std": None, "lb": None, "ub": None, "conf": conf}
    if days is not None:
        d_list = list(days)
        # drop days of dropped (non-finite) trades to keep alignment
        r_all = np.asarray([x if x is not None else np.nan for x in r], dtype=float)
        keep = np.isfinite(r_all)
        d_list = [d for d, k in zip(d_list, keep) if k]
        codes, n_days = _day_codes(d_list)
    else:
        codes, n_days = np.arange(n), n
    day_sum = np.bincount(codes, weights=a, minlength=n_days)
    day_cnt = np.bincount(codes, minlength=n_days).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_days, size=(int(n_boot), n_days))
    sums = day_sum[idx].sum(axis=1)
    cnts = day_cnt[idx].sum(axis=1)
    means = sums / np.maximum(cnts, 1.0)
    return {
        "n": n, "n_days": int(n_days), "mean": float(a.mean()), "std": float(a.std(ddof=1)),
        "lb": float(np.percentile(means, (1.0 - conf) * 100.0)),
        "ub": float(np.percentile(means, conf * 100.0)),
        "conf": conf,
    }


# ---------------------------------------------------------------------------------------------------------
# Pooled out-of-sample edge (the walk-forward / CPCV gate)
# ---------------------------------------------------------------------------------------------------------

def merge_small_folds(fold_r: list, fold_days: list | None = None, min_trades: int = 30) -> tuple[list, list]:
    """Merge adjacent folds until every group holds at least ``min_trades`` trades (a fold of 8 trades says nothing).
    The trailing group is folded into its predecessor when it is still short. Returns (groups_r, groups_days)."""
    fold_days = fold_days if fold_days is not None else [None] * len(fold_r)
    groups_r: list[list] = []
    groups_d: list[list] = []
    cur_r: list = []
    cur_d: list = []
    for rs, ds in zip(fold_r, fold_days):
        cur_r.extend(list(rs))
        cur_d.extend(list(ds) if ds is not None else [None] * len(rs))
        if len(cur_r) >= min_trades:
            groups_r.append(cur_r)
            groups_d.append(cur_d)
            cur_r, cur_d = [], []
    if cur_r:
        if groups_r:
            groups_r[-1].extend(cur_r)
            groups_d[-1].extend(cur_d)
        else:
            groups_r.append(cur_r)
            groups_d.append(cur_d)
    return groups_r, groups_d


@dataclass
class PooledEdge:
    """Result of pooling out-of-sample R across folds. ``status`` is one of
    PASS         lower bound above zero and at least half the (merged) folds positive
    NOT_PROVEN   mean R positive but the sample cannot separate it from zero yet (needs ``trades_needed_more``)
    FAIL         mean R <= 0 over enough trades, or fewer than half the folds are positive with no proof of an edge
    UNPROVEN     fewer than ``min_pooled_trades`` pooled trades: the gate cannot convict or acquit"""
    status: str
    n_trades: int = 0
    n_groups: int = 0
    mean_r: float | None = None
    lb_r: float | None = None
    ub_r: float | None = None
    sd_r: float | None = None
    fraction_groups_positive: float | None = None
    group_means: list = field(default_factory=list)
    group_trades: list = field(default_factory=list)
    trades_needed_more: int | None = None
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def hard_failed(self) -> bool:
        return self.status == "FAIL"

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["passed"] = self.passed
        return d


def pooled_oos_edge(fold_r: list, fold_days: list | None = None, min_fold_trades: int = 30,
                    min_pooled_trades: int = 30, conf: float = 0.95, n_boot: int = 2000, seed: int = 11,
                    min_fraction_positive: float = 0.5) -> PooledEdge:
    """Gate on POOLED out-of-sample mean R per trade.

    1. merge folds under ``min_fold_trades`` trades into neighbours;
    2. pool every out-of-sample trade and take a day-block bootstrap lower bound on mean R;
    3. require the lower bound above zero AND at least half of the merged folds positive.
    A positive mean that the sample cannot yet separate from zero is NOT_PROVEN (power-aware), never FAIL."""
    groups_r, groups_d = merge_small_folds(fold_r, fold_days, min_fold_trades)
    pooled = [x for g in groups_r for x in g]
    pooled_days = [d for g in groups_d for d in g]
    n = len(pooled)
    if n < min_pooled_trades:
        return PooledEdge("UNPROVEN", n_trades=n, n_groups=len(groups_r),
                          note=f"only {n} pooled out-of-sample trade(s) (< {min_pooled_trades}); the gate cannot judge")
    have_days = all(d is not None for d in pooled_days)
    bs = day_block_bootstrap_mean(pooled, pooled_days if have_days else None, conf=conf, n_boot=n_boot, seed=seed)
    gm = [float(np.mean(g)) for g in groups_r if len(g)]
    frac_pos = float(np.mean([m > 0 for m in gm])) if gm else 0.0
    mean_r, lb, ub, sd = bs["mean"], bs["lb"], bs["ub"], bs["std"]
    more = trades_still_needed(mean_r if (mean_r and mean_r > 0) else None, sd, n)
    out = PooledEdge("NOT_PROVEN", n_trades=n, n_groups=len(groups_r), mean_r=mean_r, lb_r=lb, ub_r=ub, sd_r=sd,
                     fraction_groups_positive=frac_pos, group_means=gm, group_trades=[len(g) for g in groups_r],
                     trades_needed_more=more)
    if mean_r is None or mean_r <= 0:
        out.status = "FAIL"
        out.note = f"pooled out-of-sample mean R is {0.0 if mean_r is None else mean_r:+.3f}R over {n} trades"
    elif lb is not None and lb > 0 and frac_pos >= min_fraction_positive:
        out.status = "PASS"
        out.note = f"pooled out-of-sample mean R {mean_r:+.3f}R, day-block lower bound {lb:+.3f}R, {frac_pos * 100:.0f}% of folds positive"
    elif frac_pos < min_fraction_positive and (lb is None or lb <= 0):
        out.status = "FAIL"
        out.note = (f"only {frac_pos * 100:.0f}% of out-of-sample folds were positive and the pooled lower bound "
                    f"({'n/a' if lb is None else f'{lb:+.3f}R'}) does not clear zero")
    else:
        out.status = "NOT_PROVEN"
        out.note = (f"pooled out-of-sample mean R {mean_r:+.3f}R is positive but its lower bound "
                    f"({'n/a' if lb is None else f'{lb:+.3f}R'}) does not clear zero yet"
                    + (f"; about {more} more trades would confirm it" if more else ""))
    return out


# ---------------------------------------------------------------------------------------------------------
# Edge decay: edge by year and the rolling-mean-R slope (replaces the signal half-life check)
# ---------------------------------------------------------------------------------------------------------

def edge_by_year(r, times) -> list[dict]:
    """Mean R and trade count per calendar year (the year of the trade's timestamp)."""
    import pandas as pd
    rr = np.asarray([x if x is not None else np.nan for x in r], dtype=float)
    ts = pd.to_datetime(pd.Series(list(times)), utc=True, errors="coerce")
    ok = np.isfinite(rr) & ts.notna().to_numpy()
    if not ok.any():
        return []
    years = ts[ok].dt.year.to_numpy()
    vals = rr[ok]
    out = []
    for y in sorted(set(years.tolist())):
        v = vals[years == y]
        out.append({"year": int(y), "n": int(v.size), "mean_r": float(v.mean())})
    return out


def rolling_mean_r_slope(r, times, conf: float = 0.95, n_boot: int = 1000, seed: int = 13,
                         min_trades: int = 60) -> dict:
    """Slope of R against time, in R per YEAR, with a bootstrap interval (trades resampled with replacement).

    A decaying edge shows as a negative slope whose upper bound is also below zero. ``flag`` is True only in that case,
    so an edge that is merely noisy does not trigger it, and independent trades pass whatever their lag structure --
    which is exactly where the old autocorrelation half-life check went wrong."""
    import pandas as pd
    rr = np.asarray([x if x is not None else np.nan for x in r], dtype=float)
    ts = pd.to_datetime(pd.Series(list(times)), utc=True, errors="coerce")
    ok = np.isfinite(rr) & ts.notna().to_numpy()
    n = int(ok.sum())
    if n < min_trades:
        return {"n": n, "slope_per_year": None, "lb": None, "ub": None, "flag": False,
                "note": f"only {n} trade(s) (< {min_trades}); decay not measurable"}
    t_years = ((ts[ok] - ts[ok].min()).dt.total_seconds().to_numpy()) / (365.25 * 86400.0)
    y = rr[ok]
    if float(np.ptp(t_years)) < 0.5:
        return {"n": n, "slope_per_year": None, "lb": None, "ub": None, "flag": False,
                "note": "less than half a year of trades; decay not measurable"}

    def _slope(tt, yy):
        tc = tt - tt.mean()
        den = float((tc * tc).sum())
        return float((tc * (yy - yy.mean())).sum() / den) if den > 0 else 0.0

    s = _slope(t_years, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        boots[i] = _slope(t_years[idx], y[idx])
    lo = float(np.percentile(boots, (1.0 - conf) * 100.0 / 2.0))
    hi = float(np.percentile(boots, 100.0 - (1.0 - conf) * 100.0 / 2.0))
    flag = bool(hi < 0.0)
    return {"n": n, "slope_per_year": s, "lb": lo, "ub": hi, "flag": flag,
            "note": ("edge is decaying: mean R falls " f"{abs(s):.2f}R per year (95% interval {lo:+.2f} to {hi:+.2f})"
                     if flag else f"no significant decay (slope {s:+.2f}R per year, 95% interval {lo:+.2f} to {hi:+.2f})")}


# ---------------------------------------------------------------------------------------------------------
# Sequential probability ratio test (forward test)
# ---------------------------------------------------------------------------------------------------------

@dataclass
class SPRTBoundaries:
    h0_r: float
    h1_r: float
    sd_r: float
    alpha: float          # false-accept: accepting H1 when H0 is true
    beta: float           # false-reject: rejecting H1 when H1 is true
    accept_h1_at: float   # LLR >= this -> the edge is confirmed
    accept_h0_at: float   # LLR <= this -> no edge at the hypothesised size
    expected_trades_h0: float
    expected_trades_h1: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def sprt_boundaries(h1_r: float, sd_r: float = DEFAULT_TRADE_SD_R, h0_r: float = 0.0, alpha: float = 0.05,
                    beta: float = 0.20) -> SPRTBoundaries:
    """Wald boundaries for a normal mean with known sd. With H0 = 0R, H1 = +0.21R, sd 1.3R, 5% false-accept and 20%
    false-reject the expected stopping point is about 103 trades under H0 and about 146 under H1."""
    if h1_r <= h0_r:
        raise ValueError("h1_r must be greater than h0_r")
    a = math.log((1.0 - beta) / alpha)
    b = math.log(beta / (1.0 - alpha))
    delta = (h1_r - h0_r) ** 2 / (2.0 * sd_r ** 2)          # |expected LLR drift per trade|
    e0 = ((1.0 - alpha) * b + alpha * a) / (-delta)
    e1 = ((1.0 - beta) * a + beta * b) / delta
    return SPRTBoundaries(h0_r, h1_r, sd_r, alpha, beta, a, b, e0, e1)


def sprt_llr(r, h1_r: float, sd_r: float = DEFAULT_TRADE_SD_R, h0_r: float = 0.0) -> float:
    """Cumulative log-likelihood ratio of H1 vs H0 over the R values seen so far."""
    x = np.asarray([v for v in r if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return 0.0
    return float(((h1_r - h0_r) / sd_r ** 2 * (x - (h1_r + h0_r) / 2.0)).sum())


def sprt_decision(r, h1_r: float, sd_r: float = DEFAULT_TRADE_SD_R, h0_r: float = 0.0, alpha: float = 0.05,
                  beta: float = 0.20, max_trades: int | None = None) -> dict:
    """ACCEPT_H1 (edge confirmed), ACCEPT_H0 (stop: no edge of this size), or CONTINUE. ``max_trades`` (the futility
    truncation) turns CONTINUE into STOP_FUTILE once exceeded with the LLR still between the boundaries."""
    bnd = sprt_boundaries(h1_r, sd_r, h0_r, alpha, beta)
    xs = [v for v in r if v is not None and np.isfinite(v)]
    llr = sprt_llr(xs, h1_r, sd_r, h0_r)
    if llr >= bnd.accept_h1_at:
        d = "ACCEPT_H1"
    elif llr <= bnd.accept_h0_at:
        d = "ACCEPT_H0"
    elif max_trades is not None and len(xs) >= int(max_trades):
        d = "STOP_FUTILE"
    else:
        d = "CONTINUE"
    return {"decision": d, "llr": llr, "n": len(xs), "accept_h1_at": bnd.accept_h1_at, "accept_h0_at": bnd.accept_h0_at,
            "expected_trades_h0": bnd.expected_trades_h0, "expected_trades_h1": bnd.expected_trades_h1}


def months_to_decide(expected_trades: float, trades_per_month: float, n_uncorrelated_strategies: int = 1) -> float | None:
    """Calendar months until a sequential test is expected to stop, for a portfolio of uncorrelated strategies each
    trading ``trades_per_month``. One strategy at 6.7 trades/month needs 15-22 months; four need about 4-5.5."""
    rate = float(trades_per_month) * max(int(n_uncorrelated_strategies), 1)
    return None if rate <= 0 else float(expected_trades) / rate


# ---------------------------------------------------------------------------------------------------------
# Empirical-Bayes shrinkage toward a family mean
# ---------------------------------------------------------------------------------------------------------

def empirical_bayes_shrink(n: int, xbar: float, sigma: float, mu0: float, tau: float) -> float:
    """Shrunk mean = (n*xbar/sigma^2 + mu0/tau^2) / (n/sigma^2 + 1/tau^2).

    n trades, strategy mean ``xbar``, per-trade sd ``sigma``, family mean ``mu0``, spread of true means across the
    family ``tau``. A 100-trade winner is pulled strongly toward what its family usually delivers; a 2,000-trade one
    barely moves. tau <= 0 pins the estimate to the family mean."""
    n = max(int(n), 0)
    if sigma is None or sigma <= 0:
        return float(xbar)
    if tau is None or tau <= 0:
        return float(mu0)
    prec_data = n / (sigma ** 2)
    prec_prior = 1.0 / (tau ** 2)
    return float((prec_data * xbar + prec_prior * mu0) / (prec_data + prec_prior))


def estimate_family_prior(strategy_means: list, strategy_sds: list, strategy_ns: list) -> tuple[float, float]:
    """Method-of-moments (mu0, tau) from a family's strategies: mu0 = precision-weighted mean of the strategy means,
    tau^2 = max(0, variance of the means - mean sampling variance). Needs >= 2 strategies, else (mean or 0, 0.25)."""
    m = np.asarray(strategy_means, dtype=float)
    s = np.asarray(strategy_sds, dtype=float)
    k = np.asarray(strategy_ns, dtype=float)
    ok = np.isfinite(m) & np.isfinite(s) & np.isfinite(k) & (k > 0) & (s > 0)
    m, s, k = m[ok], s[ok], k[ok]
    if m.size == 0:
        return 0.0, 0.25
    samp_var = s ** 2 / k
    w = 1.0 / samp_var
    mu0 = float((w * m).sum() / w.sum())
    if m.size < 2:
        return mu0, 0.25
    tau2 = max(0.0, float(m.var(ddof=1)) - float(samp_var.mean()))
    return mu0, math.sqrt(tau2) if tau2 > 0 else 0.05


def shrunk_lower_bound(n: int, xbar: float, sigma: float, mu0: float, tau: float, z: float = Z_ONE_SIDED_95) -> float:
    """One-sided lower bound on the shrunk mean: shrunk mean minus z * posterior sd. Size from this, not from the point
    estimate, so a lucky sample cannot push risk up."""
    prec = (max(int(n), 0) / (sigma ** 2) if sigma and sigma > 0 else 0.0) + (1.0 / tau ** 2 if tau and tau > 0 else 0.0)
    post_sd = math.sqrt(1.0 / prec) if prec > 0 else float("inf")
    return empirical_bayes_shrink(n, xbar, sigma, mu0, tau) - z * post_sd


def size_from_lower_bound(lb_r: float | None, base_risk_pct: float = 0.5, max_risk_pct: float = 1.5,
                          full_size_edge_r: float = 0.20) -> float:
    """Risk per trade (percent of the account) scaled by the LOWER-BOUND edge, never the point estimate:
    zero when the bound is <= 0, base_risk_pct at ``full_size_edge_r``, capped at ``max_risk_pct``."""
    if lb_r is None or not math.isfinite(lb_r) or lb_r <= 0:
        return 0.0
    return float(min(max_risk_pct, base_risk_pct * (lb_r / max(full_size_edge_r, 1e-9))))
