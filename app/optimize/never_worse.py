"""Never-worse refinement guard (2026-10-31 audit, section 5, change 1).

The known-edge sweep showed refinement walking AWAY from a planted rule: on the thinned cells the planted EMA12/48 rule had +0.15R net in
development, the optimizer's champion had a per-trade Sharpe of 0.054 against about 0.13 for the planted rule, and the pipeline kept the
champion. Two seeds cannot separate a bug from bad luck, so the guard is built so the failure is impossible by construction:

    The PARENT rule is always a candidate. A mutated champion replaces it only if its lower-confidence-bound expectancy (net of
    stressed costs) on inner, purged folds of the DEVELOPMENT data beats the parent's by more than one standard error. Otherwise the
    parent is returned.

Mechanics
---------
* Both rules are backtested once on the development frame with every cost assumption scaled by ``cost_multiplier`` (default 2x).
* The frame's timeline is cut into ``n_folds`` contiguous blocks. A trade belongs to a fold only if it ENTERS and EXITS inside it (a trade
  that straddles a boundary is purged), so folds share no information.
* Each fold's mean R (pnl / dollars risked at the stop) is the unit of evidence; a fold with fewer than ``min_trades_per_fold`` trades
  is not evidence for either rule and is dropped for BOTH (paired comparison on the same folds).
* ``LCB = mean(fold means) - t * SE`` with a one-sided 95% t quantile. ``SE_diff`` is the standard error of the paired per-fold
  differences. The champion wins iff ``LCB_champion - LCB_parent > se_multiple * SE_diff`` (default one standard error).
* Fewer than ``min_valid_folds`` usable folds means the improvement cannot be confirmed, so the parent is kept; the one exception is a
  parent that has no usable folds at all while the champion does (there is nothing to protect).

Pure functions (``decide``) take trade lists so they unit-test without a backtest; ``run_never_worse`` backtests both rules.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DEFAULT_FOLDS = 5
DEFAULT_COST_MULTIPLIER = 2.0
DEFAULT_SE_MULTIPLE = 1.0
MIN_TRADES_PER_FOLD = 3
MIN_VALID_FOLDS = 3
CONFIDENCE = 0.95


@dataclass
class NeverWorseDecision:
    accept: bool                      # True = the champion replaces the parent; False = the parent is returned
    reason: str
    parent_lcb_r: float | None
    champion_lcb_r: float | None
    parent_mean_r: float | None
    champion_mean_r: float | None
    lcb_gain_r: float | None          # champion LCB - parent LCB
    se_diff_r: float | None           # standard error of the paired per-fold difference
    n_folds_used: int
    parent_trades: int
    champion_trades: int
    cost_multiplier: float = DEFAULT_COST_MULTIPLIER
    se_multiple: float = DEFAULT_SE_MULTIPLE

    def to_dict(self) -> dict:
        return asdict(self)


def trade_r(t) -> float | None:
    """R multiple of one trade: net P&L over the dollars risked at its stop. None when the risk is unknown."""
    risk = getattr(t, "risk_at_stop_dollars", None)
    if not risk:
        ir, sz = getattr(t, "initial_risk", None), getattr(t, "size", None)
        risk = (ir * sz) if (ir and sz) else None
    if not risk or not math.isfinite(float(risk)) or float(risk) <= 0:
        return None
    return float(t.pnl) / float(risk)


def _ns(x) -> int:
    return int(pd.Timestamp(pd.to_datetime(x, utc=True)).value)


def fold_bounds(timestamps, n_folds: int = DEFAULT_FOLDS) -> list[tuple[int, int]]:
    """``n_folds`` contiguous [start, end) blocks of equal bar count over the timeline, as UTC nanosecond integers."""
    ts = pd.to_datetime(pd.Series(timestamps), utc=True).reset_index(drop=True)
    n = len(ts)
    k = max(1, int(n_folds))
    if n < k * 2:
        return []
    cuts = [int(round(i * n / k)) for i in range(k + 1)]
    out = []
    for i in range(k):
        a, b = cuts[i], cuts[i + 1]
        start = int(ts.iloc[a].value)
        end = int(ts.iloc[b].value) if b < n else int(ts.iloc[n - 1].value) + 1
        out.append((start, end))
    return out


def fold_mean_r(trades, bounds: list[tuple[int, int]], min_trades: int = MIN_TRADES_PER_FOLD) -> list[float]:
    """Mean R per fold (NaN where the fold has fewer than ``min_trades`` purged trades)."""
    rows = []
    for t in trades or []:
        r = trade_r(t)
        if r is None:
            continue
        try:
            rows.append((_ns(t.entry_time), _ns(t.exit_time), r))
        except Exception:  # noqa: BLE001 - an unparseable timestamp is not evidence
            continue
    out = []
    for start, end in bounds:
        rs = [r for (e, x, r) in rows if e >= start and x < end]           # entered AND exited inside the fold: purged otherwise
        out.append(float(np.mean(rs)) if len(rs) >= min_trades else float("nan"))
    return out


def _t_quantile(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(CONFIDENCE, max(df, 1)))
    except Exception:  # noqa: BLE001
        return 1.645


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return m, se


def decide(
    parent_trades, champion_trades, timestamps, *, n_folds: int = DEFAULT_FOLDS, se_multiple: float = DEFAULT_SE_MULTIPLE,
    min_trades_per_fold: int = MIN_TRADES_PER_FOLD, min_valid_folds: int = MIN_VALID_FOLDS,
    cost_multiplier: float = DEFAULT_COST_MULTIPLIER,
) -> NeverWorseDecision:
    """Pure decision from two trade lists and the development frame's timestamps. See the module docstring."""
    npar, nchamp = len(parent_trades or []), len(champion_trades or [])
    base = dict(parent_trades=npar, champion_trades=nchamp, cost_multiplier=cost_multiplier, se_multiple=se_multiple)

    def keep(reason, **kw):
        d = dict(parent_lcb_r=None, champion_lcb_r=None, parent_mean_r=None, champion_mean_r=None, lcb_gain_r=None, se_diff_r=None,
                 n_folds_used=0)
        d.update(kw)
        return NeverWorseDecision(accept=False, reason=reason, **d, **base)

    bounds = fold_bounds(timestamps, n_folds)
    if not bounds:
        return keep("Too little development data to cut inner folds; the parent is kept.")
    p = np.array(fold_mean_r(parent_trades, bounds, min_trades_per_fold))
    c = np.array(fold_mean_r(champion_trades, bounds, min_trades_per_fold))
    c_ok = int(np.isfinite(c).sum())
    p_ok = int(np.isfinite(p).sum())
    if c_ok < min_valid_folds:
        return keep(f"The champion has only {c_ok} usable inner fold(s) (need {min_valid_folds}); its edge cannot be confirmed, so the "
                    "parent is kept.", n_folds_used=c_ok)
    if p_ok < min_valid_folds:
        # nothing to protect: the parent is not measurable on these folds and the champion is
        cm, cse = _mean_se(c[np.isfinite(c)])
        lcb = cm - _t_quantile(c_ok - 1) * cse if math.isfinite(cse) else cm
        return NeverWorseDecision(
            accept=True, reason=f"The parent has only {p_ok} usable inner fold(s) while the champion has {c_ok}; there is no measurable "
                                "parent edge to protect, so the champion is used.",
            parent_lcb_r=None, champion_lcb_r=lcb, parent_mean_r=None, champion_mean_r=cm, lcb_gain_r=None, se_diff_r=None,
            n_folds_used=c_ok, **base)
    both = np.isfinite(p) & np.isfinite(c)
    k = int(both.sum())
    if k < min_valid_folds:
        return keep(f"Only {k} inner fold(s) are usable for BOTH rules (need {min_valid_folds}); the parent is kept.", n_folds_used=k)
    pk, ck = p[both], c[both]
    tq = _t_quantile(k - 1)
    pm, pse = _mean_se(pk)
    cm, cse = _mean_se(ck)
    p_lcb, c_lcb = pm - tq * pse, cm - tq * cse
    _, se_diff = _mean_se(ck - pk)
    gain = c_lcb - p_lcb
    if not math.isfinite(gain) or not math.isfinite(se_diff):
        return keep("The paired standard error is undefined; the parent is kept.", n_folds_used=k)
    accept = gain > se_multiple * se_diff
    reason = (f"Champion LCB {c_lcb:+.3f}R vs parent {p_lcb:+.3f}R over {k} inner fold(s) at {cost_multiplier:g}x costs: gain "
              f"{gain:+.3f}R {'>' if accept else '<='} {se_multiple:g} x SE {se_diff:.3f}R -> "
              f"{'the champion replaces the parent.' if accept else 'the parent is kept (the improvement is not confirmed).'}")
    return NeverWorseDecision(accept=accept, reason=reason, parent_lcb_r=p_lcb, champion_lcb_r=c_lcb, parent_mean_r=pm, champion_mean_r=cm,
                              lcb_gain_r=gain, se_diff_r=se_diff, n_folds_used=k, **base)


def run_never_worse(
    df: pd.DataFrame, parent_strategy, champion_strategy, risk, adaptive_risk=None, *, n_folds: int = DEFAULT_FOLDS,
    cost_multiplier: float = DEFAULT_COST_MULTIPLIER, se_multiple: float = DEFAULT_SE_MULTIPLE,
) -> NeverWorseDecision:
    """Backtest both rules on ``df`` (the DEVELOPMENT data only -- the caller must not pass a sealed holdout) at stressed costs and decide."""
    from app.backtest.engine import run_backtest
    from app.optimize.refinement import _stressed_risk_config
    stressed = _stressed_risk_config(risk, max(float(cost_multiplier), 1.0))
    bt_p = run_backtest(df, parent_strategy, stressed, adaptive_risk=adaptive_risk, compute_stats=False, lookahead_check="skip")
    bt_c = run_backtest(df, champion_strategy, stressed, adaptive_risk=adaptive_risk, compute_stats=False, lookahead_check="skip")
    return decide(bt_p.trades, bt_c.trades, df["timestamp"], n_folds=n_folds, se_multiple=se_multiple, cost_multiplier=cost_multiplier)
