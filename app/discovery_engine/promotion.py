"""The mandatory promotion sequence.

A candidate that the search loop likes is NOT a result. Before it can be called a forward-test candidate it must pass, in this
order and with no step skippable:

  1. reproduction        re-run with the stored provenance (seed, data fingerprint, git hash, config); dropped if it does not reproduce
  2. sequence-deflated   Deflated Sharpe against the LIFETIME ledger count clustered to effective trials, using the Sharpe spread of
                         the whole evaluated pool
  3. null gate           beats random entry timing (``run_null_edge_gate``)
  4. pooled test         the SAME fixed parameters across 8-12 markets, day-block bootstrap at 2x costs (``pooled_edge_test``).
                         Instrument-tied families (GEX, event calendar, pairs, macro feeds) cannot be pooled: they get a STRICTER
                         sealed-slice bar and a mandatory forward test instead.
  5. sealed slice        the last, never-searched slice, read exactly once for this run

``run_promotion_sequence`` stops at the first failing step (later steps are recorded as ``skipped``) and returns a
``PromotionResult``. ``result.raise_for_failure()`` turns a failure into an exception for callers that must fail the run; the
loop-mode "stop at the first 70%" path uses exactly that, so there is no way to stop early and keep a champion that has not
passed all five.

Every step is injectable (``Steps``) so the sequence is unit-testable without a backtest engine.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

STEP_ORDER = ("reproduction", "sequence_deflated", "null_gate", "pooled_edge", "sealed_slice")

# Substrings that mark a spec as tied to ONE instrument's auxiliary data (cannot be pooled across markets).
INSTRUMENT_TIED_MARKERS = [
    "gex", "gamma", "dealer_", "event_", "minutes_since", "minutes_until", "pair_close", "pair_", "high_impact",
    "fomc", "cpi_", "nfp_", "macro_", "ext_", "cot_", "options_", "put_call", "vix_", "yield_", "opex",
]


class PromotionFailed(RuntimeError):
    def __init__(self, result: "PromotionResult"):
        super().__init__(f"promotion failed at step '{result.failed_step}': {result.failed_reason}")
        self.result = result


@dataclass
class StepResult:
    name: str
    passed: bool
    skipped: bool = False
    detail: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class PromotionResult:
    passed: bool = False
    steps: list = field(default_factory=list)
    failed_step: str | None = None
    failed_reason: str = ""
    poolable: bool = True
    forward_test_required: bool = True
    notes: list = field(default_factory=list)

    def step(self, name: str) -> StepResult | None:
        return next((s for s in self.steps if s.name == name), None)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    def render(self) -> str:
        lines = [f"PROMOTION: {'PASSED' if self.passed else 'FAILED at ' + str(self.failed_step)}"]
        for s in self.steps:
            tag = "skipped" if s.skipped else ("pass" if s.passed else "FAIL")
            lines.append(f"  [{tag:7s}] {s.name}: {s.detail}")
        if self.passed and self.forward_test_required:
            lines.append("  -> forward test is MANDATORY before any real-money use")
        return "\n".join(lines)

    def raise_for_failure(self) -> "PromotionResult":
        if not self.passed:
            raise PromotionFailed(self)
        return self


def is_instrument_tied(spec: dict | None, family: str | None = None, extra_markers=()) -> bool:
    """True when the candidate depends on auxiliary data that exists for one instrument only (cannot be pooled across markets)."""
    try:
        text = json.dumps(spec or {}, default=str).lower()
    except Exception:  # noqa: BLE001
        text = str(spec).lower()
    fam = (family or "").lower()
    markers = list(INSTRUMENT_TIED_MARKERS) + [str(m).lower() for m in extra_markers]
    if any(m in text for m in markers):
        return True
    if any(m in fam for m in ("gex", "gamma", "event", "pair", "calendar", "macro", "news")):
        return True
    try:
        from app.search import strategy_space as sps
        f = sps.FAMILIES.get(family or "")
        if f is not None and (getattr(f, "requires_pair_data", False) or getattr(f, "requires_calendar_data", False)):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


# ------------------------------------------------------------------------------------------------------------- defaults
@dataclass
class PromotionConfig:
    dsr_min_psr: float = 0.95
    null_seeds: int = 200
    null_p_max: float = 0.10
    pooled_min_trades: int = 100
    pooled_min_markets: int = 8
    pooled_conf: float = 0.95
    sealed_min_trades: int = 30
    sealed_conf_poolable: float = 0.90        # confirmation of an already-pooled-proven edge
    sealed_conf_tied: float = 0.975           # STRICTER for instrument-tied families (no pooled evidence)
    sealed_min_trades_tied: int = 50
    cost_multiplier: float = 2.0
    repro_abs_tol: float = 0.02
    repro_rel_tol: float = 0.10


@dataclass
class Steps:
    """Injectable step implementations. Each returns ``StepResult``. ``None`` falls back to the default implementation."""

    reproduction: Callable | None = None
    sequence_deflated: Callable | None = None
    null_gate: Callable | None = None
    pooled_edge: Callable | None = None
    sealed_slice: Callable | None = None


def _daily_sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1) if x.size > 2 else 0.0
    return float(x.mean() / sd) if sd > 0 else 0.0


def deflated_step(observed_daily: np.ndarray, pool_matrix: np.ndarray | None, n_trials: int, *, min_psr: float = 0.95) -> StepResult:
    """Step 2. Deflated Sharpe of the candidate's DAILY P&L against ``n_trials`` (effective, lifetime) and the Sharpe spread of the pool.

    Daily units throughout (observed, trial spread, T, skew, kurtosis) so the deflation is internally consistent."""
    from app.search.robustness import deflated_sharpe_gate
    obs = np.asarray(observed_daily, dtype=float)
    if obs.size < 30:
        return StepResult("sequence_deflated", False, detail=f"only {obs.size} daily observations -- cannot deflate")
    if pool_matrix is not None and getattr(pool_matrix, "ndim", 0) == 2 and pool_matrix.shape[1] >= 2:
        trial_sr = [_daily_sharpe(pool_matrix[:, j]) for j in range(pool_matrix.shape[1])]
    else:
        trial_sr = [_daily_sharpe(obs)]
    mu, sd = float(obs.mean()), float(obs.std(ddof=0))
    skew = float(np.mean(((obs - mu) / sd) ** 3)) if sd > 0 else 0.0
    kurt = float(np.mean(((obs - mu) / sd) ** 4)) if sd > 0 else 3.0
    gate = deflated_sharpe_gate(_daily_sharpe(obs), trial_sr, max(1, int(n_trials)), int(obs.size), returns_skew=skew,
                                returns_kurtosis=kurt, min_probabilistic_sharpe=min_psr)
    return StepResult("sequence_deflated", bool(gate.passed), detail=gate.reason,
                      data={"psr": gate.probabilistic_sharpe, "n_trials": gate.n_trials, "benchmark_sharpe": gate.benchmark_sharpe})


def pooled_step(per_market: dict, cfg: PromotionConfig) -> StepResult:
    """Step 4 (poolable families). Needs >= ``pooled_min_markets`` markets WITH trades; fewer is an honest INSUFFICIENT, not a pass."""
    from app.validation.pooled_edge import pooled_edge_test
    with_trades = [m for m, (rs, _d) in per_market.items() if rs]
    if len(with_trades) < cfg.pooled_min_markets:
        return StepResult("pooled_edge", False, detail=(
            f"only {len(with_trades)} market(s) with trades (< {cfg.pooled_min_markets}); run scripts/bootstrap_data.py to see which "
            f"target markets are missing -- a pooled test on fewer markets is not the pooled test"),
            data={"markets": sorted(with_trades), "status": "INSUFFICIENT"})
    res = pooled_edge_test(per_market, cost_multiplier=cfg.cost_multiplier, min_trades=cfg.pooled_min_trades, conf=cfg.pooled_conf)
    return StepResult("pooled_edge", res.passed, detail=res.render(), data=res.to_dict())


def sealed_step(rs: list, days: list, *, tied: bool, cfg: PromotionConfig) -> StepResult:
    """Step 5. One read of the sealed slice. The bar is a one-sided day-block-bootstrap lower bound above zero; STRICTER (higher
    confidence and more trades) for instrument-tied families, which have no pooled evidence behind them."""
    from app.validation.pooled_edge import pooled_edge_test
    conf = cfg.sealed_conf_tied if tied else cfg.sealed_conf_poolable
    need = cfg.sealed_min_trades_tied if tied else cfg.sealed_min_trades
    res = pooled_edge_test({"sealed": (rs, days)}, min_trades=need, conf=conf, cost_multiplier=cfg.cost_multiplier)
    kind = "instrument-tied (stricter bar)" if tied else "poolable"
    return StepResult("sealed_slice", res.passed, detail=f"[{kind}] {res.render()}", data=res.to_dict())


def _skipped(name: str, why: str) -> StepResult:
    return StepResult(name, False, skipped=True, detail=why)


def run_promotion_sequence(candidate: dict, steps: Steps | None = None, cfg: PromotionConfig | None = None,
                           family: str | None = None, extra_tied_markers=()) -> PromotionResult:
    """Run the five mandatory steps in order, stopping at the first failure.

    ``candidate`` is the champion record (``spec``, ``provenance`` ...); the step callables receive ``(candidate, ctx)`` where
    ``ctx`` is a dict carrying ``poolable``, ``tied`` and ``cfg``. A step that raises counts as a FAILURE (no evidence is not a pass)."""
    cfg = cfg or PromotionConfig()
    steps = steps or Steps()
    tied = is_instrument_tied(candidate.get("spec"), family or candidate.get("family"), extra_tied_markers)
    out = PromotionResult(poolable=not tied, forward_test_required=True)
    ctx = {"tied": tied, "poolable": not tied, "cfg": cfg}
    if tied:
        out.notes.append("instrument-tied family: cannot be pooled across markets -> stricter sealed-slice bar and a mandatory forward test")
    for name in STEP_ORDER:
        if not out.passed and out.failed_step is not None:
            out.steps.append(_skipped(name, f"not run: '{out.failed_step}' failed"))
            continue
        fn = getattr(steps, name, None)
        if name == "pooled_edge" and tied and fn is None:
            out.steps.append(StepResult(name, True, skipped=True, detail="not applicable: instrument-tied family cannot be pooled (stricter sealed bar applies)"))
            continue
        if fn is None:
            out.steps.append(StepResult(name, False, detail="no implementation supplied -- a step that cannot run is a failure"))
        else:
            try:
                r = fn(candidate, ctx)
            except Exception as exc:  # noqa: BLE001
                r = StepResult(name, False, detail=f"step crashed: {type(exc).__name__}: {exc}")
            if not isinstance(r, StepResult):
                r = StepResult(name, bool(r), detail=str(r))
            out.steps.append(r)
        last = out.steps[-1]
        if not last.passed and not last.skipped:
            out.failed_step, out.failed_reason = name, last.detail
    out.passed = out.failed_step is None
    out.forward_test_required = True      # always: a backtest-only result is never a deployment decision
    return out


def finalize_loop_stop(candidate: dict, steps: Steps, cfg: PromotionConfig | None = None, family: str | None = None) -> PromotionResult:
    """Loop-mode stop ("stop at the first 70%") entry point: the stop is allowed, but the champion is kept ONLY when the mandatory
    sequence passes. Raises ``PromotionFailed`` otherwise so the run is marked failed, not silently 'stopped'."""
    return run_promotion_sequence(candidate, steps, cfg, family).raise_for_failure()


def stable_neighbourhood(scores: list[float], min_fraction: float = 0.6, floor: float = 0.0) -> bool:
    """A 'stable' winner stays above ``floor`` in at least ``min_fraction`` of its parameter neighbours (used by the pruning pass)."""
    s = [x for x in scores if x is not None and math.isfinite(x)]
    return bool(s) and (sum(1 for x in s if x > floor) / len(s)) >= min_fraction
