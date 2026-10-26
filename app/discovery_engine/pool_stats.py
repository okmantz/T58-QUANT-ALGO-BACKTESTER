"""Pool-wide statistics: the multiple-testing controls that look at EVERY candidate that competed, not just the winner.

Deflated Sharpe corrects ONE champion for how many things were tried. Two things it cannot do, and this module does:

  1. FAMILY-WISE ERROR CONTROL over the whole pool's daily P&L matrix -- White's Reality Check, Hansen's SPA (the studentised,
     recentred refinement that is not fooled by a pool full of terrible candidates) and the Romano-Wolf stepdown (which says WHICH
     candidates survive, not just whether the best one does). All three use one stationary (Politis-Romano) bootstrap, so
     autocorrelation and volatility clustering in the daily P&L are preserved.
  2. PROBABILITY OF BACKTEST OVERFITTING (CSCV, Bailey/Borwein/Lopez de Prado/Zhu) computed on the WHOLE evaluated pool -- or a
     stratified sample of it -- instead of a pre-selected top 8. PBO is defined on the set of candidates that competed in-sample;
     on a pre-selected top-N it understates overfitting because the worst competitors have already been removed.

The matrix is ``T days x N candidates`` of daily R (or P&L); a day with no trade is 0.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_BOOT = 1000


# --------------------------------------------------------------------------------------------------- matrix building
def daily_series_from_trades(trades, unit: str = "r", label: str | None = None) -> pd.Series:
    """Per-calendar-day sum of trade results (by exit day). ``unit='r'`` uses R multiples (comparable across sizing);
    ``unit='pnl'`` uses account currency."""
    rows: dict = {}
    if unit == "r":
        from app.optimize.two_stage import trade_r_and_days
        rs, days = trade_r_and_days(trades)
        for r, d in zip(rs, days):
            if r is None or not np.isfinite(r) or d is None:
                continue
            day = pd.Timestamp(d).tz_localize(None).normalize() if pd.Timestamp(d).tzinfo else pd.Timestamp(d).normalize()
            rows[day] = rows.get(day, 0.0) + float(r)
    else:
        for t in trades or []:
            ts = getattr(t, "exit_time", None) or getattr(t, "entry_time", None)
            pnl = getattr(t, "pnl", None)
            if ts is None or pnl is None:
                continue
            tt = pd.Timestamp(ts)
            day = (tt.tz_localize(None) if tt.tzinfo else tt).normalize()
            rows[day] = rows.get(day, 0.0) + float(pnl)
    return pd.Series(rows, dtype=float, name=label).sort_index()


def build_matrix(series: dict, calendar: pd.DatetimeIndex | None = None) -> tuple[np.ndarray, list, pd.DatetimeIndex]:
    """{candidate_id: daily Series} -> (matrix T x N float64, ids, day index). Missing days are 0 (no trade)."""
    ids = list(series)
    if not ids:
        return np.zeros((0, 0)), [], pd.DatetimeIndex([])
    if calendar is None:
        idx = None
        for s in series.values():
            idx = s.index if idx is None else idx.union(s.index)
        calendar = pd.DatetimeIndex(sorted(idx)) if idx is not None else pd.DatetimeIndex([])
        # a continuous business-day calendar so idle stretches count as zero-P&L days (not skipped)
        if len(calendar) >= 2:
            has_weekend = bool((calendar.day_of_week >= 5).any())      # 24/7 or Sunday-open markets keep their weekend P&L
            calendar = (pd.date_range(calendar.min(), calendar.max(), freq="D") if has_weekend
                        else pd.bdate_range(calendar.min(), calendar.max()))
    mat = np.zeros((len(calendar), len(ids)), dtype=float)
    for j, cid in enumerate(ids):
        s = series[cid]
        if len(s):
            mat[:, j] = s.reindex(calendar).fillna(0.0).to_numpy(dtype=float)
    return mat, ids, calendar


# ------------------------------------------------------------------------------------------------ stationary bootstrap
def stationary_bootstrap_indices(T: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, T) integer indices of Politis-Romano stationary-bootstrap resamples (geometric block lengths, wrap-around)."""
    T = int(T)
    p = 1.0 / max(float(mean_block), 1.0)
    idx = np.empty((int(n_boot), T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, size=int(n_boot))
    new_block = rng.random((int(n_boot), T)) < p
    starts = rng.integers(0, T, size=(int(n_boot), T))
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def default_block_length(T: int) -> float:
    """Rule-of-thumb mean block length ~ T^(1/3) (Politis-White order of magnitude), at least 2."""
    return float(max(2.0, round(float(T) ** (1.0 / 3.0))))


def _bootstrap_means(d: np.ndarray, n_boot: int, mean_block: float, seed: int) -> np.ndarray:
    """(n_boot, N) bootstrap means of the columns of ``d`` (T x N)."""
    T, N = d.shape
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(T, n_boot, mean_block, rng)
    out = np.empty((int(n_boot), N), dtype=float)
    # chunk to bound memory: (chunk, T, N) would be large for big pools
    step = max(1, int(2_000_000 // max(T * N, 1)))
    for a in range(0, int(n_boot), step):
        b = min(int(n_boot), a + step)
        out[a:b] = d[idx[a:b]].mean(axis=1)
    return out


# --------------------------------------------------------------------------------------------- Reality Check / SPA / RW
@dataclass
class FamilyWiseResult:
    n_candidates: int
    n_days: int
    best_index: int | None
    best_id: object = None
    best_mean: float | None = None
    rc_p: float | None = None            # White's Reality Check p-value (H0: no candidate beats the benchmark)
    spa_p: float | None = None           # Hansen SPA_c p-value
    spa_p_lower: float | None = None     # SPA with the "liberal" recentring (lower bound on the p-value)
    spa_p_upper: float | None = None     # SPA with the "conservative" recentring (== Reality Check, upper bound)
    stepdown_rejected: list = field(default_factory=list)   # ids of candidates with a genuinely significant edge (Romano-Wolf)
    alpha: float = 0.05
    passed: bool = False
    reason: str = ""
    mean_block: float = 0.0
    n_boot: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def family_wise_tests(mat: np.ndarray, ids: list | None = None, *, benchmark: np.ndarray | float = 0.0, alpha: float = 0.05,
                      n_boot: int = DEFAULT_BOOT, mean_block: float | None = None, seed: int = 17,
                      min_days: int = 60) -> FamilyWiseResult:
    """White's Reality Check + Hansen's SPA + Romano-Wolf stepdown on a T x N matrix of daily results.

    ``benchmark`` is the null's daily result (0 = "no edge"). ``passed`` is True only when BOTH the Reality Check and SPA_c
    reject at ``alpha`` -- the stricter of the two decides, never the friendlier. With fewer than ``min_days`` days or no
    candidates the result is ``passed=False`` with a reason (no evidence is not a pass)."""
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[1] == 0:
        return FamilyWiseResult(0, 0, None, reason="no candidates", alpha=alpha)
    T, N = mat.shape
    ids = list(ids) if ids is not None else list(range(N))
    if T < int(min_days):
        return FamilyWiseResult(N, T, None, reason=f"only {T} days (< {min_days}) -- too few for a family-wise test", alpha=alpha)
    bench = float(benchmark) if np.ndim(benchmark) == 0 else np.asarray(benchmark, dtype=float).reshape(-1, 1)
    d = mat - bench
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    mb = float(mean_block) if mean_block else default_block_length(T)
    dbar = d.mean(axis=0)
    bm = _bootstrap_means(d, int(n_boot), mb, seed)                  # (B, N)
    root_t = math.sqrt(T)
    omega = np.maximum((root_t * bm).std(axis=0, ddof=1), 1e-12)     # bootstrap s.d. of sqrt(T) * mean, per candidate

    # White's Reality Check: max_k sqrt(T) * dbar_k against its bootstrap null (centred at each candidate's own mean)
    v = float(np.max(root_t * dbar))
    v_star = np.max(root_t * (bm - dbar), axis=1)
    rc_p = float((1 + np.sum(v_star >= v)) / (int(n_boot) + 1))

    # Hansen SPA: studentised statistic, recentred so clearly-poor candidates cannot inflate the null
    tstat = root_t * dbar / omega
    t_spa = float(max(np.max(tstat), 0.0))
    a_k = 0.25 * (T ** -0.25) * omega / root_t            # threshold below which a candidate is "clearly poor"
    g_c = dbar * (dbar >= -a_k)                            # consistent recentring (SPA_c)
    g_l = np.maximum(dbar, 0.0)                            # liberal recentring (SPA_l): poor candidates keep their negative mean
    g_u = dbar                                             # conservative recentring (SPA_u) == the Reality Check null
    def _spa_p(g):
        z = (bm - g) * root_t / omega
        t_star = np.maximum(np.max(z, axis=1), 0.0)
        return float((1 + np.sum(t_star >= t_spa)) / (int(n_boot) + 1))
    spa_p = _spa_p(g_c)
    spa_lo, spa_hi = _spa_p(g_l), _spa_p(g_u)

    # Romano-Wolf stepdown on the same draws
    tb = root_t * (bm - dbar) / omega                      # (B, N) bootstrap t-statistics under the null
    order = np.argsort(-tstat)
    rejected: list = []
    remaining = list(order)
    while remaining:
        crit = float(np.quantile(np.max(tb[:, remaining], axis=1), 1.0 - alpha))
        top = remaining[0]
        if tstat[top] > crit:
            rejected.append(ids[top])
            remaining = remaining[1:]
        else:
            break

    best = int(np.argmax(dbar))
    res = FamilyWiseResult(
        n_candidates=N, n_days=T, best_index=best, best_id=ids[best], best_mean=float(dbar[best]), rc_p=rc_p, spa_p=spa_p,
        spa_p_lower=spa_lo, spa_p_upper=spa_hi, stepdown_rejected=rejected, alpha=alpha, mean_block=mb, n_boot=int(n_boot),
    )
    res.passed = bool(rc_p <= alpha and spa_p <= alpha)
    res.reason = (
        f"Reality Check p={rc_p:.3f}, SPA p={spa_p:.3f} over {N} candidate(s) / {T} days -- "
        + ("the best candidate's edge survives the whole pool's data snooping" if res.passed else
           "after correcting for the whole pool, the best candidate's edge is not distinguishable from luck")
        + (f"; {len(rejected)} candidate(s) survive the Romano-Wolf stepdown" if rejected else "")
    )
    return res


# --------------------------------------------------------------------------------------------------------------- PBO
@dataclass
class PoolPBOResult:
    pbo: float | None
    n_candidates: int
    n_slices: int
    n_combinations: int
    logit_mean: float | None = None
    degradation: float | None = None       # mean (IS Sharpe of the IS-best) - (its OOS Sharpe)
    prob_loss: float | None = None         # P(the IS-best candidate has a negative OOS Sharpe)
    sampled: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _sharpe_cols(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1) if x.shape[0] > 1 else np.ones(x.shape[1])
    return np.where(sd > 1e-12, mu / np.where(sd > 1e-12, sd, 1.0), 0.0)


def pool_pbo(mat: np.ndarray, *, n_slices: int = 16, max_combinations: int = 2000, seed: int = 5, min_days: int = 64) -> PoolPBOResult:
    """Probability of Backtest Overfitting by CSCV over the whole pool's daily-result matrix.

    Splits the T days into ``n_slices`` equal contiguous slices, forms every (sampled) way of choosing half of them as
    in-sample, finds the in-sample-best candidate by Sharpe and records its OUT-of-sample rank among all candidates.
    PBO = share of splits where the in-sample winner ranks at or below the OOS median."""
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[1] < 2:
        return PoolPBOResult(None, int(mat.shape[1]) if mat.ndim == 2 else 0, 0, 0, note="need at least 2 candidates")
    T, N = mat.shape
    if T < int(min_days):
        return PoolPBOResult(None, N, 0, 0, note=f"only {T} days (< {min_days})")
    S = int(n_slices) - int(n_slices) % 2
    S = max(4, min(S, T // 4 - (T // 4) % 2 if T // 4 >= 4 else 4))
    L = T // S
    mat = np.nan_to_num(mat[: L * S], nan=0.0)
    blocks = [mat[i * L:(i + 1) * L] for i in range(S)]
    half = S // 2
    total_combos = math.comb(S, half)
    rng = np.random.default_rng(seed)
    if total_combos <= int(max_combinations):
        combos = list(itertools.combinations(range(S), half))
        sampled = False
    else:
        seen: set = set()
        while len(seen) < int(max_combinations):
            seen.add(tuple(sorted(rng.choice(S, size=half, replace=False).tolist())))
        combos = sorted(seen)
        sampled = True
    logits, degr, loss = [], [], []
    for c in combos:
        cs = set(c)
        is_x = np.vstack([blocks[i] for i in sorted(cs)])
        oos_x = np.vstack([blocks[i] for i in range(S) if i not in cs])
        sr_is, sr_oos = _sharpe_cols(is_x), _sharpe_cols(oos_x)
        best = int(np.argmax(sr_is))
        rank = float((sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1)) + 1.0   # 1 = worst .. N = best
        w = rank / (N + 1.0)
        logits.append(math.log(w / (1.0 - w)))
        degr.append(float(sr_is[best] - sr_oos[best]))
        loss.append(float(sr_oos[best] < 0))
    logits_a = np.asarray(logits)
    return PoolPBOResult(
        pbo=float(np.mean(logits_a <= 0.0)), n_candidates=N, n_slices=S, n_combinations=len(combos), logit_mean=float(logits_a.mean()),
        degradation=float(np.mean(degr)), prob_loss=float(np.mean(loss)), sampled=sampled,
    )


def stratified_sample(ids: list, strata: list, k: int, seed: int = 3) -> list[int]:
    """Indices of a stratified sample of ``k`` items (proportional allocation by stratum, >= 1 per stratum, deterministic).

    Used so PBO sees every family's candidates in proportion rather than only whoever ranked highest."""
    n = len(ids)
    if n <= int(k):
        return list(range(n))
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    quota = {s: max(1, int(round(int(k) * len(v) / n))) for s, v in groups.items()}
    # trim / top up to exactly k
    while sum(quota.values()) > int(k):
        s = max(quota, key=lambda z: quota[z])
        if quota[s] <= 1:
            break
        quota[s] -= 1
    out: list[int] = []
    for s, v in groups.items():
        q = min(quota[s], len(v))
        out.extend(sorted(rng.choice(v, size=q, replace=False).tolist()))
    if len(out) < int(k):
        rest = [i for i in range(n) if i not in set(out)]
        extra = rng.choice(rest, size=min(int(k) - len(out), len(rest)), replace=False).tolist() if rest else []
        out.extend(extra)
    return sorted(out[: int(k)])


def pool_report(series: dict, strata: dict | None = None, *, alpha: float = 0.05, n_boot: int = DEFAULT_BOOT, pbo_sample: int = 200,
                n_slices: int = 16, seed: int = 17) -> dict:
    """One-call summary used by the engine: family-wise tests on the FULL pool + PBO on a stratified sample of it."""
    mat, ids, cal = build_matrix(series)
    fw = family_wise_tests(mat, ids, alpha=alpha, n_boot=n_boot, seed=seed)
    pbo = PoolPBOResult(None, mat.shape[1] if mat.ndim == 2 else 0, 0, 0, note="no matrix")
    if mat.ndim == 2 and mat.shape[1] >= 2:
        strat = [(strata or {}).get(i, "?") for i in ids]
        sel = stratified_sample(ids, strat, pbo_sample, seed=seed)
        pbo = pool_pbo(mat[:, sel], n_slices=n_slices, seed=seed)
        pbo.note = (pbo.note + " " if pbo.note else "") + f"computed on {len(sel)} of {len(ids)} evaluated candidate(s)" \
            + (" (stratified by family)" if strata else "")
    return {"family_wise": fw.to_dict(), "pbo": pbo.to_dict(), "n_candidates": len(ids), "n_days": int(mat.shape[0]) if mat.ndim == 2 else 0}
