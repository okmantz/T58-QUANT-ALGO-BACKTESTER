"""Pooled cross-market edge test (report 2026-10-20, P1): the primary "is there an edge?" question.

Power, not strictness, is what blocks a true edge from clearing the gates: one strategy on one market at ~7 trades a month cannot prove
+0.10R in six years. Pooling the SAME fixed-parameter strategy across 8-12 markets multiplies the sample (the report's table: a
+0.20R edge clears the joint gates 44% of the time on one market, ~81% pooled over 8; +0.10R: 9% -> 28%).

The three questions are kept separate, in this order:

  1. IS THERE AN EDGE?      ``pooled_edge_test``: day-block bootstrap of the pooled mean R after 2x costs  <- this module
  2. DOES IT SURVIVE?       cost stress, regimes, holdout, ruin (the existing gates, unchanged)
  3. ONLY THEN: SIZING/EV   prop-rule Monte Carlo, ``app.validation.ev_verdict``

Blocks are TRADING DAYS across ALL markets at once, so same-day cross-market correlation (index futures moving together on a FOMC day)
stays inside a block and the bootstrap does not pretend 8 correlated markets are 8 independent samples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.validation.edge_stats import DEFAULT_TRADE_SD_R, _day_codes, trades_needed

DEFAULT_COST_MULTIPLIER = 2.0
DEFAULT_MIN_TRADES = 100
DEFAULT_CONF = 0.95


@dataclass
class PooledEdgeResult:
    status: str                                   # PASS | UNPROVEN | FAIL | INSUFFICIENT
    n_trades: int = 0
    n_markets: int = 0
    n_days: int = 0
    mean_r: float | None = None
    lb_r: float | None = None                     # one-sided lower bound at ``conf`` (day-block bootstrap)
    ub_r: float | None = None
    t_stat: float | None = None                   # cluster-robust (by trading day) t of the pooled mean
    p_value: float | None = None                  # one-sided, H0: mean R <= 0
    cost_multiplier: float = DEFAULT_COST_MULTIPLIER
    conf: float = DEFAULT_CONF
    per_market: list = field(default_factory=list)
    trades_still_needed: int | None = None
    notes: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def hard_failed(self) -> bool:
        return self.status == "FAIL"

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        if self.mean_r is None:
            return f"POOLED EDGE: {self.status} ({'; '.join(self.notes) or 'not measurable'})"
        lb = "n/a" if self.lb_r is None else f"{self.lb_r:+.3f}"
        t = "n/a" if self.t_stat is None else f"{self.t_stat:.2f}"
        return (f"POOLED EDGE ({self.cost_multiplier:g}x costs): {self.status} - mean {self.mean_r:+.3f}R over {self.n_trades} trades, "
                f"{self.n_markets} market(s), {self.n_days} trading days; {self.conf * 100:.0f}% lower bound {lb}R, cluster t = {t}")


def _clean(rs, days):
    r = np.asarray([np.nan if x is None else x for x in rs], dtype=float)
    keep = np.isfinite(r)
    d = [d for d, k in zip(list(days), keep) if k] if days is not None else None
    return r[keep], d


def cluster_t_stat(r, days=None) -> float | None:
    """t-statistic of mean R with standard errors clustered by trading day (trades on one day are correlated).

    t = mean / sqrt( sum_d (sum_{i in d} (r_i - mean))^2 ) , the standard cluster-robust variance of a sample mean."""
    a = np.asarray([x for x in (r if r is not None else []) if x is not None and np.isfinite(x)], dtype=float)
    if a.size < 5:
        return None
    if days is not None:
        _, d = _clean(r, days)
        codes, n_days = _day_codes(d)
    else:
        codes, n_days = np.arange(a.size), a.size
    if n_days < 3:
        return None
    mean = float(a.mean())
    resid_by_day = np.bincount(codes, weights=a - mean, minlength=n_days)
    se = math.sqrt(float((resid_by_day ** 2).sum())) / a.size
    return mean / se if se > 0 else None


def pool_markets(per_market: dict) -> tuple[np.ndarray, list, list]:
    """per_market: {symbol: (r_list, day_list)} -> (r, days, market_labels), finite trades only, aligned."""
    rs: list = []
    ds: list = []
    ms: list = []
    for sym, (r, d) in per_market.items():
        r = list(r or [])
        d = list(d or [None] * len(r))
        for x, dd in zip(r, d):
            if x is not None and np.isfinite(x):
                rs.append(float(x))
                ds.append(dd)
                ms.append(sym)
    return np.asarray(rs, dtype=float), ds, ms


def pooled_edge_test(per_market: dict, *, cost_multiplier: float = DEFAULT_COST_MULTIPLIER, min_trades: int = DEFAULT_MIN_TRADES,
                     conf: float = DEFAULT_CONF, n_boot: int = 4000, seed: int = 11, sd_r: float = DEFAULT_TRADE_SD_R) -> PooledEdgeResult:
    """Day-block-bootstrapped pooled mean R across markets (R already measured at ``cost_multiplier`` x costs by the caller).

    Status:
      PASS          n >= min_trades and the one-sided ``conf`` lower bound of the pooled mean R is above zero
      FAIL          n >= min_trades and the UPPER bound is below zero (a convincing losing edge)
      UNPROVEN      enough trades, bound straddles zero (positive or not) - more markets / history / forward trades needed
      INSUFFICIENT  fewer than ``min_trades`` trades pooled
    """
    r, days, mk = pool_markets(per_market)
    res = PooledEdgeResult(status="INSUFFICIENT", n_trades=int(r.size), n_markets=len({m for m in mk}), cost_multiplier=float(cost_multiplier), conf=conf)
    for sym, (rr, dd) in per_market.items():
        a = np.asarray([x for x in (rr or []) if x is not None and np.isfinite(x)], dtype=float)
        res.per_market.append({"market": sym, "n": int(a.size), "mean_r": float(a.mean()) if a.size else None,
                               "t_stat": cluster_t_stat(list(rr or []), list(dd) if dd is not None else None) if a.size >= 5 else None})
    if r.size < 5:
        res.notes.append(f"only {int(r.size)} pooled trades")
        return res
    codes, n_days = _day_codes(days)
    res.n_days = int(n_days)
    res.mean_r = float(r.mean())
    res.t_stat = cluster_t_stat(list(r), days)
    day_sum = np.bincount(codes, weights=r, minlength=n_days)
    day_cnt = np.bincount(codes, minlength=n_days).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_days, size=(int(n_boot), n_days))
    means = day_sum[idx].sum(axis=1) / np.maximum(day_cnt[idx].sum(axis=1), 1.0)
    res.lb_r = float(np.percentile(means, (1.0 - conf) * 100.0))
    res.ub_r = float(np.percentile(means, conf * 100.0))
    res.p_value = float((1 + (means <= 0).sum()) / (int(n_boot) + 1))
    if r.size < min_trades:
        res.status = "INSUFFICIENT"
        res.notes.append(f"{int(r.size)} pooled trades < the {min_trades}-trade floor")
    elif res.ub_r < 0:
        res.status = "FAIL"
        res.notes.append("even the upper bound of the pooled mean R is below zero")
    elif res.lb_r > 0:
        res.status = "PASS"
    else:
        res.status = "UNPROVEN"
        if res.mean_r > 0:
            need = trades_needed(res.mean_r, sd_r)
            res.trades_still_needed = max(0, int(need) - int(r.size)) if need is not None else None
            res.notes.append(f"positive but the lower bound ({res.lb_r:+.3f}R) has not cleared zero")
        else:
            res.notes.append("pooled mean R is not positive")
    return res


def estimated_cost_in_r(risk, stop_distance: float) -> float:
    """Round-trip cost of one trade expressed in R for a stop of ``stop_distance`` PRICE units.

    cost = 2 sides x (spread + slippage) + commission, over the stop distance. A candidate whose cost is above ~0.1R cannot be rescued by
    any signal (report P2), so Search Lab rejects it before spending a backtest."""
    try:
        D = float(stop_distance)
        if not math.isfinite(D) or D <= 0:
            return float("nan")
        side = float(risk.side_cost_price())
        per_unit_comm = 0.0
        cs = getattr(risk, "contract_size", None)
        if cs:
            per_unit_comm += float(getattr(risk, "commission_per_contract", 0.0) or 0.0) / float(cs)
        fixed = float(getattr(risk, "commission_per_trade", 0.0) or 0.0)
        if fixed > 0:
            units = float(risk.risk_amount(getattr(risk, "initial_balance", 0.0) or 0.0)) / (D + side) if (D + side) > 0 else 0.0
            per_unit_comm += fixed / units if units > 0 else 0.0
        return (2.0 * side + per_unit_comm) / D
    except Exception:  # noqa: BLE001
        return float("nan")


def r_by_market_from_backtests(family_name: str, params: dict, frames: dict, risks: dict, *, cost_multiplier: float = DEFAULT_COST_MULTIPLIER,
                               log=lambda m: None) -> dict:
    """Run ONE fixed parameter set of a family on every market at ``cost_multiplier`` x costs; returns {symbol: (R list, day list)}.

    ``frames`` / ``risks`` are keyed by upper-case symbol. A market that errors or has no trades contributes an empty list (reported,
    never silently dropped from ``per_market``)."""
    from app.backtest.engine import run_backtest
    from app.optimize.two_stage import stress_risk, trade_r_and_days
    from app.search import strategy_space as sps
    from app.strategy.manual import ManualStrategy

    cfg = sps.FAMILIES[family_name].build(params)
    out: dict = {}
    for sym, df in frames.items():
        risk = risks.get(sym)
        if df is None or risk is None:
            out[sym] = ([], [])
            continue
        try:
            bt = run_backtest(df, ManualStrategy(cfg), stress_risk(risk, cost_multiplier), lookahead_check="skip")
            rs, days = trade_r_and_days(bt.trades)
            out[sym] = (rs, days)
        except Exception as exc:  # noqa: BLE001 - a failing market is reported as empty, not dropped
            log(f"  {sym}: failed ({exc})")
            out[sym] = ([], [])
    return out


def pooled_family_test(family_name: str, frames: dict, risks: dict, *, params: dict | None = None,
                       cost_multiplier: float = DEFAULT_COST_MULTIPLIER, min_trades: int = DEFAULT_MIN_TRADES,
                       n_boot: int = 4000, log=lambda m: None) -> PooledEdgeResult:
    """Primary edge test for a family: fixed middle-of-the-grid parameters (the pre-registration convention), 8-12 markets, 2x costs."""
    from app.research.preregistered_tests import canonical_params
    p = dict(params) if params else canonical_params(family_name)
    per = r_by_market_from_backtests(family_name, p, frames, risks, cost_multiplier=cost_multiplier, log=log)
    res = pooled_edge_test(per, cost_multiplier=cost_multiplier, min_trades=min_trades, n_boot=n_boot)
    res.notes.append(f"family={family_name} params={p}")
    return res


# ---------------------------------------------------------------------------------------------------- analytic power
def analytic_power_table(edges=(0.05, 0.10, 0.20, 0.30, 0.40), *, trades_per_month: float = 7.0, years: float = 6.2, holdout_frac: float = 0.20,
                         sd_r: float = DEFAULT_TRADE_SD_R, n_markets: int = 8, rho: float = 0.3, alpha: float = 0.05) -> list[dict]:
    """The report's simplified power model (normal approximation): P(one-sided ``alpha`` lower bound of mean R > 0) for the development
    sample, the locked holdout, both, and both pooled over ``n_markets`` with average cross-market trade correlation ``rho``.

    This is a stand-in for the joint gate logic; the empirical numbers come from ``app.validation.pipeline_controls`` (planted edges)."""
    z = _norm_ppf(1.0 - alpha)
    n_total = trades_per_month * 12.0 * years
    n_dev, n_ho = n_total * (1.0 - holdout_frac), n_total * holdout_frac
    eff_markets = n_markets / (1.0 + (n_markets - 1) * rho)      # effective independent markets under equicorrelation

    def power(edge: float, n: float) -> float:
        se = sd_r / math.sqrt(max(n, 1.0))
        return float(_norm_cdf(edge / se - z))

    rows = []
    for e in edges:
        pd_, ph = power(e, n_dev), power(e, n_ho)
        pdp, php = power(e, n_dev * eff_markets), power(e, n_ho * eff_markets)
        rows.append({"edge_r": e, "dev": pd_, "holdout": ph, "both": pd_ * ph, "both_pooled": pdp * php})
    return rows


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation (|error| < 1.2e-9) - avoids a scipy dependency in the core path.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
