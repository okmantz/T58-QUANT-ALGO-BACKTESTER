"""Power-aware SEQUENTIAL forward test for the 30-day prop objective (Wald SPRT, exact operating characteristic).

Why this exists (2026-11-11 audit). The old forward gate asked for ``Wilson 95% lower bound >= 65%`` on the resolved forward attempts. With
n = 12 attempts that needs 12/12 passes, and a strategy whose TRUE pass rate is 80% clears it only ~7% of the time: the gate was
statistically unreachable, so it could never confirm anything. Loosening the confidence level is NOT the fix (that just buys false
confirmations). The fix is a test that is designed for the question:

    H0: the strategy's true P(pass <= 30d) is ``p0`` (unacceptable)      H1: it is ``p1`` (the target)

and that stops the moment the evidence is conclusive either way, with its error rates fixed IN ADVANCE:

    alpha  P(confirm | true p = p0)   the false-confirmation rate (default 5%)
    beta   P(reject  | true p = p1)   so the POWER at the target is 1 - beta (default 80%)

Pre-registration. A design (p0, p1, alpha, beta, n_max) is registered once per configuration and cannot be changed afterwards
(``ForwardRegistry.register_design``); attempts are read in the order they were recorded and the FIRST boundary crossing is the decision,
so later results cannot overturn it, and nothing is "peeked" at. If ``n_max`` independent attempts resolve with no boundary crossed the
verdict is INCONCLUSIVE (not a confirmation).

Independence. Forward attempts must not overlap: an attempt purchased inside the horizon of the previous counted attempt shares its
market path, so it is excluded from the count and reported (``n_excluded_overlap``), never silently counted.

``operating_characteristic(p)`` is exact (dynamic programming over (attempts, passes)), so the design's power, false-confirmation rate and
expected number of attempts at any true pass rate are known BEFORE a single evaluation is bought.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import pandas as pd

ACCEPT_TARGET = "ACCEPT_TARGET"        # H1: evidence the true pass rate is at the target
REJECT_TARGET = "REJECT_TARGET"        # H0: evidence it is not
CONTINUE = "CONTINUE"                  # keep forward testing (budget remains)
INCONCLUSIVE = "INCONCLUSIVE"          # budget spent without a boundary crossing: NOT a confirmation


@dataclass(frozen=True)
class SequentialDesign:
    p0: float = 0.40                   # a pass rate that is NOT good enough to buy evaluations on
    p1: float = 0.65                   # the target pass rate
    alpha: float = 0.05                # P(confirm | p = p0)
    beta: float = 0.20                 # P(reject  | p = p1)   (power = 1 - beta)
    n_max: int = 40                    # attempt budget; truncation point of the test
    horizon_days: int = 30

    def __post_init__(self):
        if not (0.0 < self.p0 < self.p1 < 1.0):
            raise ValueError("need 0 < p0 < p1 < 1")
        if not (0.0 < self.alpha < 0.5 and 0.0 < self.beta < 0.5):
            raise ValueError("alpha and beta must be in (0, 0.5)")
        if int(self.n_max) < 1:
            raise ValueError("n_max must be >= 1")

    @property
    def upper(self) -> float:          # accept-H1 boundary on the log-likelihood ratio
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:          # accept-H0 boundary
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def step_pass(self) -> float:
        return math.log(self.p1 / self.p0)

    @property
    def step_fail(self) -> float:
        return math.log((1.0 - self.p1) / (1.0 - self.p0))

    def llr(self, passes: int, n: int) -> float:
        return passes * self.step_pass + (n - passes) * self.step_fail

    def to_dict(self) -> dict:
        return asdict(self)


def decide(design: SequentialDesign, outcomes) -> dict:
    """Run the SPRT over ``outcomes`` (iterable of bool/int in chronological order). The FIRST boundary crossing decides; anything after it
    is ignored."""
    passes = n = 0
    llr = 0.0
    for o in outcomes:
        n += 1
        passes += 1 if o else 0
        llr = design.llr(passes, n)
        if llr >= design.upper:
            return _result(design, ACCEPT_TARGET, passes, n, llr, stopped_at=n)
        if llr <= design.lower:
            return _result(design, REJECT_TARGET, passes, n, llr, stopped_at=n)
        if n >= design.n_max:
            return _result(design, INCONCLUSIVE, passes, n, llr, stopped_at=n)
    return _result(design, CONTINUE, passes, n, llr, stopped_at=None)


def _result(design: SequentialDesign, decision: str, passes: int, n: int, llr: float, stopped_at) -> dict:
    return {"decision": decision, "n": int(n), "passes": int(passes), "llr": float(llr), "accept_boundary": design.upper,
            "reject_boundary": design.lower, "stopped_at": stopped_at, "p_hat": (passes / n) if n else None,
            "posterior_p_ge_target": posterior_prob_at_least(passes, n, design.p1), "design": design.to_dict(),
            "attempts_still_possible": (max(0, design.n_max - n) if decision == CONTINUE else 0)}


def posterior_prob_at_least(passes: int, n: int, threshold: float, a: float = 1.0, b: float = 1.0, grid: int = 4000) -> float:
    """Bayesian read-out: P(true pass rate >= ``threshold`` | data) under a Beta(a, b) prior (uniform by default), by midpoint integration.
    Reported next to the SPRT as information only; the SPRT decision is the registered stopping rule."""
    if n < 0 or passes < 0 or passes > n:
        raise ValueError("invalid counts")
    A, B = a + passes, b + (n - passes)
    # log-density up to a constant, integrated numerically (no scipy dependency)
    lo, hi, num, den = 0.0, 1.0, 0.0, 0.0
    step = (hi - lo) / grid
    for i in range(grid):
        x = lo + (i + 0.5) * step
        w = math.exp((A - 1.0) * math.log(x) + (B - 1.0) * math.log(1.0 - x))
        den += w
        if x >= threshold:
            num += w
    return float(num / den) if den > 0 else 0.0


def operating_characteristic(design: SequentialDesign, p: float) -> dict:
    """EXACT behaviour of the registered design when the true pass probability is ``p``: P(confirm), P(reject), P(inconclusive) and the
    expected number of attempts. Dynamic programming over the (attempts, passes) lattice with absorbing boundaries."""
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    live = {0: 1.0}                                    # passes -> probability mass still undecided after n attempts
    p_acc = p_rej = exp_n = 0.0
    for n in range(1, int(design.n_max) + 1):
        nxt: dict = {}
        for k, mass in live.items():
            for ok, pr in ((1, p), (0, 1.0 - p)):
                if pr <= 0.0:
                    continue
                m = mass * pr
                kk = k + ok
                llr = design.llr(kk, n)
                if llr >= design.upper:
                    p_acc += m
                    exp_n += m * n
                elif llr <= design.lower:
                    p_rej += m
                    exp_n += m * n
                else:
                    nxt[kk] = nxt.get(kk, 0.0) + m
        live = nxt
    p_inc = float(sum(live.values()))
    exp_n += p_inc * design.n_max
    return {"p": float(p), "p_confirm": float(p_acc), "p_reject": float(p_rej), "p_inconclusive": p_inc, "expected_attempts": float(exp_n)}


def design_summary(design: SequentialDesign) -> dict:
    """What to know before buying a single evaluation: error rates at p0 and p1 and the expected cost of getting an answer."""
    at0, at1 = operating_characteristic(design, design.p0), operating_characteristic(design, design.p1)
    mid = operating_characteristic(design, 0.5 * (design.p0 + design.p1))
    return {"design": design.to_dict(), "false_confirmation_at_p0": at0["p_confirm"], "power_at_p1": at1["p_confirm"],
            "expected_attempts_at_p0": at0["expected_attempts"], "expected_attempts_at_p1": at1["expected_attempts"],
            "expected_attempts_midway": mid["expected_attempts"], "p_inconclusive_at_p1": at1["p_inconclusive"],
            "meets_error_budget": bool(at0["p_confirm"] <= design.alpha + 1e-9 and at1["p_confirm"] >= 1.0 - design.beta - 0.05)}


def independent_attempts(rows: list, horizon_days: int = 30) -> tuple:
    """``(kept, n_excluded)``: attempts in chronological order of their START date, dropping any whose start falls inside the horizon of
    the previously kept attempt (overlapping windows share one market path and are not independent trials)."""
    parsed = sorted(((pd.Timestamp(r["start"]), i, r) for i, r in enumerate(rows)), key=lambda t: (t[0], t[1]))
    kept, last, excluded = [], None, 0
    for st, _i, r in parsed:
        if last is not None and (st - last).days < horizon_days:
            excluded += 1
            continue
        kept.append(r)
        last = st
    return kept, excluded


def forward_outcome_flags(rows: list, horizon_days: int = 30) -> list:
    """Resolved attempts -> pass flags. A pass only counts if it came within the horizon (a slow pass is a miss for the 30-day objective)."""
    return [bool(r["outcome"] == "passed" and (r.get("days") is None or r["days"] <= horizon_days)) for r in rows if r["outcome"] != "open"]
