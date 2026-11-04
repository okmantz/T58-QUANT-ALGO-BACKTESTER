"""Outer bootstrap for the pass-rate interval (2026-11-04 audit, P0-1).

The old "sample-aware" interval was a Wilson interval over ``n_eff = min(history days / attempt days, trades / 10)`` where
``attempt days`` was the median time to pass. A strategy that needs 150 days to pass therefore got a fifth of the effective attempts of one
that needs 30 -- the interval measured SLOWNESS, not weakness. At a true 0.15R edge (real pass rate 92%) its lower bound averaged 60, under
the 70 bar.

The uncertainty that matters is uncertainty in the EDGE: would a different draw of the same strategy's trade history have produced a
materially different pass rate? That is what a nested bootstrap answers directly:

    1. resample whole SESSION DAYS of the source trade stream with replacement (same number of days, so the calendar structure,
       the daily-loss clustering and the same-day trade grouping are all kept);
    2. re-run the pass-rate Monte Carlo on that resampled stream (a few hundred paths is plenty);
    3. repeat ``reps`` times; the spread of the resulting pass rates is the edge uncertainty.

The reported lower bound is the ``quantile`` (default 10th) percentile of the replicates, taken AFTER removing the replicates' own
Monte Carlo noise (each inner run has a binomial standard error of its own, which would otherwise widen the interval for free).
The point estimate used for the expected-value test stays the ordinary full-sample Monte Carlo number.

Nothing here depends on how long attempts take, so a slow edge and a fast edge of equal reliability get the same interval.
"""
from __future__ import annotations

import math
from dataclasses import replace as _dc_replace

import numpy as np
import pandas as pd

DEFAULT_REPS = 40
DEFAULT_SIMS = 400
DEFAULT_QUANTILE = 0.10
MIN_REPS = 8


def resample_trades_by_day(trades, rng: np.random.Generator, groups=None, dates=None, block_days: int = 1) -> list:
    """One bootstrap replicate of a trade list: whole session days drawn with replacement (in blocks of ``block_days`` consecutive days),
    laid back onto the ORIGINAL day sequence so the replicate spans exactly the same calendar. Each drawn day's trades keep their
    intraday times; only the date is moved. Returns new Trade objects (the inputs are never mutated)."""
    from app.monte_carlo.engine import _day_groups
    if groups is None or dates is None:
        groups, dates = _day_groups(trades)
    nd = len(groups)
    if nd == 0:
        return []
    k = max(1, min(int(block_days), nd))
    nb = -(-nd // k)
    starts = rng.integers(0, nd - k + 1, size=nb)
    draw = (starts[:, None] + np.arange(k)[None, :]).reshape(-1)[:nd]
    out: list = []
    for slot, src in enumerate(draw):
        shift_days = int((pd.Timestamp(dates[slot]) - pd.Timestamp(dates[int(src)])).days)
        off = pd.DateOffset(days=shift_days) if shift_days else None
        for ti in groups[int(src)]:
            t = trades[int(ti)]
            if off is None:
                out.append(_dc_replace(t))
            else:
                out.append(_dc_replace(t, entry_time=pd.Timestamp(t.entry_time) + off, exit_time=pd.Timestamp(t.exit_time) + off))
    return out


def _deconvolved_quantile(values: np.ndarray, noise_var: float, q: float) -> float:
    """Quantile ``q`` of the replicate rates with the replicates' own Monte Carlo noise taken out: the spread around the median is scaled by
    sqrt(max(var_total - noise_var, 0) / var_total). With no usable spread the median is returned."""
    med = float(np.median(values))
    var_t = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
    if var_t <= 1e-12:
        return med
    scale = math.sqrt(max(var_t - max(noise_var, 0.0), 0.0) / var_t)
    raw = float(np.quantile(values, q))
    return med + (raw - med) * scale


def outer_bootstrap(trades, rules, base_cfg, *, reps: int = DEFAULT_REPS, sims: int = DEFAULT_SIMS, quantile: float = DEFAULT_QUANTILE,
                    seed: int | None = None, progress_cb=None) -> dict | None:
    """Nested bootstrap of the per-attempt pass and pass-AND-paid probabilities (PERCENT). Returns None when the stream is too small to
    resample meaningfully or every replicate failed.

    Result keys: method, reps, sims_per_rep, quantile, n_source_trades, pass {lo, median, hi, mean, sd_edge, sd_total, noise_sd, values},
    paid {...same...}, speed_adjusted {lo, median, hi}, median_calendar_days_to_pass (median over replicates)."""
    from app.monte_carlo.engine import MIN_TRADES_FOR_VERDICT, _day_groups, run_monte_carlo
    reps = int(reps)
    if reps < MIN_REPS or not trades or len(trades) < MIN_TRADES_FOR_VERDICT:
        return None
    groups, dates = _day_groups(trades)
    if len(groups) < 10:
        return None
    seed0 = int(base_cfg.random_seed if seed is None and base_cfg.random_seed is not None else (seed or 0))
    rng = np.random.default_rng(seed0 + 90_001)
    inner = _dc_replace(base_cfg, n_simulations=int(sims), outer_bootstrap_reps=0)
    pass_v: list[float] = []
    paid_v: list[float] = []
    sadj_v: list[float] = []
    days_v: list[float] = []
    pass_noise: list[float] = []
    paid_noise: list[float] = []
    for i in range(reps):
        try:
            sample = resample_trades_by_day(trades, rng, groups, dates, block_days=max(1, int(getattr(base_cfg, "day_block_days", 1) or 1)))
            res = run_monte_carlo(sample, rules, _dc_replace(inner, random_seed=seed0 + 17 * (i + 1)), selection_bias_caveat=False)
        except Exception:  # noqa: BLE001 -- one bad replicate must not sink the interval
            continue
        n_att = max(int(res.total_independent_attempts or 0), 1)
        p = float(res.per_attempt_pass_probability) / 100.0
        q = float(res.per_attempt_paid_probability) / 100.0
        pass_v.append(p * 100.0)
        paid_v.append(q * 100.0)
        pass_noise.append(p * (1.0 - p) / n_att * 1e4)
        paid_noise.append(q * (1.0 - q) / n_att * 1e4)
        sadj_v.append(float(getattr(res, "speed_adjusted_pass_pct", 0.0) or 0.0))
        d = getattr(res, "median_calendar_days_to_pass", None)
        if d is not None:
            days_v.append(float(d))
        if progress_cb is not None:
            try:
                progress_cb(i + 1, reps)
            except Exception:  # noqa: BLE001
                pass
    if len(pass_v) < MIN_REPS:
        return None

    def _summ(vals, noise):
        a = np.asarray(vals, dtype=float)
        nv = float(np.mean(noise)) if noise else 0.0
        sd_t = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
        return {
            "lo": _deconvolved_quantile(a, nv, quantile), "median": float(np.median(a)),
            "hi": _deconvolved_quantile(a, nv, 1.0 - quantile), "mean": float(a.mean()),
            "sd_total": sd_t, "noise_sd": math.sqrt(max(nv, 0.0)), "sd_edge": math.sqrt(max(sd_t * sd_t - nv, 0.0)),
            "values": [float(x) for x in a],
        }

    sa = np.asarray(sadj_v, dtype=float)
    return {
        "method": "outer_bootstrap", "reps": len(pass_v), "sims_per_rep": int(sims), "quantile": float(quantile),
        "n_source_trades": len(trades), "n_source_days": len(groups),
        "pass": _summ(pass_v, pass_noise), "paid": _summ(paid_v, paid_noise),
        "speed_adjusted": {"lo": float(np.quantile(sa, quantile)), "median": float(np.median(sa)), "hi": float(np.quantile(sa, 1.0 - quantile))},
        "median_calendar_days_to_pass": float(np.median(days_v)) if days_v else None,
    }
