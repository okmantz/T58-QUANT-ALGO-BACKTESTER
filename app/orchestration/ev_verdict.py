"""Expected-value verdict that sits beside READY (2026-10 audit, build item 1).

READY asks "is the pass probability above 70%?". That is a confidence statement, not an
economic one. A challenge fee is a bet, so the question that decides whether to spend it is

        EV per attempt  =  p x q x V  -  p x A  -  c

    p  probability ONE fresh attempt passes the evaluation (measured on the CLEAN HOLDOUT)
    q  probability a passed account is then paid (derived: q = r / p, r = P(paid within the window))
    V  what the trader keeps from that payout (profit split x first-payout size)
    A  one-time activation fee charged on a PASS (0 for most firms; Topstep standard path $149)
    c  cost of the attempt (challenge price from app.prop.presets.FIRM_FEES)

Break-even pass rate is  p* = c / (q x V - A)  (c=$100, q=0.45, V=$2,500, A=0 -> 8.9%).

Labels (``EVVerdict.label``):
    HIGH CONFIDENCE       the legacy 70% gate (verdict == "READY") passed and the EV is positive
    FORWARD-TEST CANDIDATE clean-holdout pass rate is above break-even WITH A MARGIN, EV > 0,
                          bankroll ruin is under the cap, rules are verified, no lookahead problem
    NOT WORTH A FEE       evidence is complete and EV <= 0 or the margin/ruin conditions fail
    INSUFFICIENT EVIDENCE a required input is missing (no holdout, no fee row, no payout rate, ...)

This module never promotes the legacy verdict; it adds a second, economic answer next to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

HIGH_CONFIDENCE = "HIGH CONFIDENCE"
FORWARD_TEST_CANDIDATE = "FORWARD-TEST CANDIDATE"
NOT_WORTH_A_FEE = "NOT WORTH A FEE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT EVIDENCE"


def compute_ev(p: float, q: float, v: float, c: float, activation_fee: float = 0.0) -> float:
    """EV per attempt = p*q*V - p*A - c (all probabilities as fractions 0..1, money in dollars;
    A = one-time activation fee paid only when the evaluation is passed, default 0)."""
    return float(p) * float(q) * float(v) - float(p) * float(activation_fee) - float(c)


def break_even_pass_rate(c: float, q: float, v: float, activation_fee: float = 0.0) -> float | None:
    """p* = c / (q*V - A). None when q*V - A <= 0 (no pass rate can pay for the attempt)."""
    denom = float(q) * float(v) - float(activation_fee)
    if denom <= 0:
        return None
    return float(c) / denom


def expected_max_std_normal(n_trials: int) -> float:
    """Expected maximum of ``n_trials`` i.i.d. standard normals (Bailey/Lopez de Prado approximation).
    0 for n_trials <= 1."""
    n = int(n_trials)
    if n <= 1:
        return 0.0
    a = math.sqrt(2.0 * math.log(n))
    return a - (math.log(math.log(n)) + math.log(4.0 * math.pi)) / (2.0 * a) if n > 2 else 0.5642


def deflate_pass_rate(p: float, n_trials: int, n_effective: float) -> float:
    """Heuristic search-bias deflation of a pass rate (fraction 0..1).

    Picking the best of ``n_trials`` configurations inflates the winner's measured rate by about
    (sampling noise) x E[max of n_trials standard normals]. With ``n_effective`` independent
    observations behind the rate, noise sd = sqrt(p(1-p)/n_effective). Returns p minus that
    inflation, floored at 0. It is an approximation (trials are correlated, so it can over-
    correct) -- use it as a haircut, not as an exact estimate."""
    p = min(max(float(p), 0.0), 1.0)
    n_eff = max(float(n_effective), 1.0)
    sd = math.sqrt(max(p * (1.0 - p), 1e-9) / n_eff)
    return max(0.0, p - sd * expected_max_std_normal(n_trials))


@dataclass
class EVVerdict:
    label: str
    reasons: list = field(default_factory=list)
    p: float | None = None                 # holdout per-attempt pass rate used (fraction)
    p_deflated: float | None = None
    q: float | None = None
    v: float | None = None
    c: float | None = None
    ev: float | None = None
    break_even_p: float | None = None
    margin_required: float | None = None
    bankroll_ruin_pct: float | None = None
    ruin_cap_pct: float | None = None
    inputs_missing: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        f = lambda v, fmt: "n/a" if v is None else fmt.format(v)  # noqa: E731
        lines = [
            f"EV VERDICT: {self.label}",
            f"  p (clean-holdout pass) {f(self.p and self.p * 100, '{:.1f}%')}   deflated {f(self.p_deflated and self.p_deflated * 100, '{:.1f}%')}   "
            f"q (paid | passed) {f(self.q and self.q * 100, '{:.0f}%')}",
            f"  V ${f(self.v, '{:,.0f}')}   c ${f(self.c, '{:,.0f}')}   EV per attempt ${f(self.ev, '{:,.2f}')}   "
            f"break-even p {f(self.break_even_p and self.break_even_p * 100, '{:.1f}%')}",
        ]
        if self.bankroll_ruin_pct is not None:
            lines.append(f"  bankroll ruin before first payout {self.bankroll_ruin_pct:.1f}% (cap {f(self.ruin_cap_pct, '{:.0f}%')})")
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def evaluate_ev(
    *,
    legacy_verdict: str,
    holdout_pass_rate: float | None,
    window_payout_rate: float | None,
    payout_value: float | None,
    challenge_price: float | None,
    reset_price: float | None = None,
    profit_split_pct: float | None = None,
    activation_fee: float = 0.0,
    bankroll_ruin_pct: float | None = None,
    ruin_cap_pct: float = 20.0,
    margin_multiple: float = 1.5,
    margin_points: float = 0.05,
    n_trials: int = 1,
    n_effective: float | None = None,
    rules_verified: bool = True,
    lookahead_ok: bool = True,
    holdout_clean: bool = True,
    automation_confirmed: bool | None = None,
) -> EVVerdict:
    """Compute the EV verdict.

    holdout_pass_rate / window_payout_rate are FRACTIONS measured on the clean holdout only.
    ``window_payout_rate`` is the unconditional P(first payout within the window); q is derived as
    r / p. payout_value is the trader's net dollars per paid account WHEN profit_split_pct is None;
    otherwise it is the gross payout and the split is applied here.
    The pass rate is deflated for ``n_trials`` before the margin test (the EV itself is shown both ways).
    margin: p_deflated must be >= max(break_even * margin_multiple, break_even + margin_points).
    """
    v = EVVerdict(label=INSUFFICIENT_EVIDENCE)
    missing = []
    if not holdout_clean or holdout_pass_rate is None:
        missing.append("clean-holdout pass rate")
    if window_payout_rate is None:
        missing.append("holdout payout-within-window rate")
    if payout_value is None or payout_value <= 0:
        missing.append("payout value")
    if challenge_price is None:
        missing.append("firm fee row (challenge price)")
    v.inputs_missing = missing
    if missing:
        v.reasons.append("Missing: " + ", ".join(missing) + ". No economic verdict is possible without them.")
        return v

    p = min(max(float(holdout_pass_rate), 0.0), 1.0)
    r = min(max(float(window_payout_rate), 0.0), 1.0)
    net_v = float(payout_value) * (float(profit_split_pct) / 100.0 if profit_split_pct is not None else 1.0)
    # an attempt that has to be re-bought after a bust costs the reset price; the first attempt costs the challenge price
    c = float(challenge_price)
    q = min(r / p, 1.0) if p > 0 else 0.0
    n_eff = n_effective if (n_effective and n_effective > 0) else 30.0
    p_def = deflate_pass_rate(p, n_trials, n_eff)
    v.p, v.p_deflated, v.q, v.v, v.c = p, p_def, q, net_v, c
    a = max(float(activation_fee or 0.0), 0.0)
    v.ev = compute_ev(p_def, q, net_v, c, a)
    v.break_even_p = break_even_pass_rate(c, q, net_v, a)
    v.bankroll_ruin_pct, v.ruin_cap_pct = bankroll_ruin_pct, ruin_cap_pct

    reasons = v.reasons
    if reset_price is None:
        reasons.append("Reset price unknown -> the challenge price is used for re-buys as well.")
    if a > 0:
        reasons.append(f"Activation fee ${a:,.0f} is charged on every pass (EV = p x q x V - p x ${a:,.0f} - c).")
    if n_trials > 1:
        reasons.append(
            f"Pass rate deflated from {p * 100:.1f}% to {p_def * 100:.1f}% for {n_trials} configurations tried "
            f"(~{n_eff:.0f} independent observations; heuristic haircut, see deflate_pass_rate)."
        )
    if v.break_even_p is None:
        v.label = NOT_WORTH_A_FEE
        reasons.append(
            "q x V does not cover the activation fee"
            + (f" (${a:,.0f})" if a > 0 else "")
            + ": no pass rate pays for the fee."
        )
        return v
    need = max(v.break_even_p * margin_multiple, v.break_even_p + margin_points)
    v.margin_required = need
    blockers = []
    if v.ev <= 0:
        blockers.append(f"EV per attempt is ${v.ev:,.2f} (<= 0): the deflated holdout pass rate {p_def * 100:.1f}% is below break-even {v.break_even_p * 100:.1f}%.")
    elif p_def < need:
        blockers.append(f"deflated pass rate {p_def * 100:.1f}% is above break-even {v.break_even_p * 100:.1f}% but short of the required margin ({need * 100:.1f}%).")
    if bankroll_ruin_pct is None:
        v.inputs_missing.append("bankroll ruin")
        blockers.append("bankroll ruin was not computed.")
    elif bankroll_ruin_pct > ruin_cap_pct:
        blockers.append(f"bankroll ruin before first payout is {bankroll_ruin_pct:.1f}% (cap {ruin_cap_pct:.0f}%).")
    if not rules_verified:
        blockers.append("this firm preset's rules are not verified against a firm page (see app.prop.presets.is_ready_eligible).")
    if not lookahead_ok:
        blockers.append("the lookahead check failed or crashed.")
    if automation_confirmed is False:
        reasons.append("NOTE: the firm has not confirmed in writing that this automation is allowed (written_confirmation=False). Get it before paying.")

    if blockers:
        if v.ev > 0 and (bankroll_ruin_pct is None or not rules_verified or not lookahead_ok):
            v.label = INSUFFICIENT_EVIDENCE
        else:
            v.label = NOT_WORTH_A_FEE
        reasons.extend(blockers)
        return v

    v.label = HIGH_CONFIDENCE if legacy_verdict == "READY" else FORWARD_TEST_CANDIDATE
    reasons.append(
        f"EV ${v.ev:,.2f} per attempt with the pass rate {p_def * 100:.1f}% clearing break-even {v.break_even_p * 100:.1f}% "
        f"by the required margin ({need * 100:.1f}%), and bankroll ruin {bankroll_ruin_pct:.1f}% under the {ruin_cap_pct:.0f}% cap."
    )
    if v.label == FORWARD_TEST_CANDIDATE:
        reasons.append("This is NOT the 70% READY gate: it is permission to forward-test on a demo / the cheapest account, not proof of an edge.")
    return v
