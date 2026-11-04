"""Expected-value verdict that sits beside READY (2026-10 audit, build item 1).

READY asks "is the pass probability above 70%?". That is a confidence statement, not an
economic one. A challenge fee is a bet, so the question that decides whether to spend it is

        EV per attempt  =  p x q x V  -  p x A  -  c

    p  probability ONE fresh attempt passes the evaluation (measured on the CLEAN HOLDOUT)
    q  probability a passed account is then paid (derived: q = r / p, r = P(paid within the window))
    V  what the trader keeps from that payout (profit split x first-payout size)
    A  one-time activation fee charged on a PASS (0 for most firms; Topstep standard path $149)
    c  cost of the attempt (challenge price from app.prop.presets.FIRM_FEES). For a MONTHLY SUBSCRIPTION
       (Topstep's Trading Combine: $49 per month, not one-off) c = price x months the evaluation is expected
       to take (``evaluation_months``) -- charging one month made the EV optimistic for slow passers.

Break-even pass rate is  p* = c / (q x V - A)  (c=$100, q=0.45, V=$2,500, A=0 -> 8.9%).

2026-11-04 audit (Fix 1) -- per-attempt billing. With ONE scalar ``months`` every attempt was billed for the MEDIAN pass duration: a path that
breached in week 2 paid for the whole median pass, and a slow pass was billed too little (and the median was a TRADING-day count converted
as if calendar days). With a ``billing_profile`` (built from the Monte Carlo's per-attempt calendar days and outcomes by
``billing_profile``), every simulated attempt is billed ``fee x ceil(calendar days / billing period)`` for however long IT lived, passes
additionally pay the activation fee, and
        EV(p) = p x (q x V - A - C_pass) - (1 - p) x C_fail
with C_pass / C_fail the mean fee paid by attempts that did / did not pass. Break-even is the p that makes EV(p) = 0,
        p* = C_fail / (q x V - A - C_pass + C_fail),
so it reflects that failures are cheap-and-fast and passes are slow-and-costly (or the reverse) instead of one blended number.

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


def evaluation_months(median_days_to_pass: float | None, billing_period_days: int = 30,
                      trading_to_calendar: float = 7.0 / 5.0, *, calendar_days: bool = False) -> int:
    """Billing periods a subscription evaluation is expected to run, rounded UP to whole billing periods, never below 1. Unknown -> 1 (the caller says so).

    ``median_days_to_pass`` is a TRADING-day count (days with a trade) by default and is converted to calendar days (x7/5). Pass
    ``calendar_days=True`` when the number is already CALENDAR days (MonteCarloResult.median_calendar_days_to_pass, 2026-11-04) -- it is then
    not converted again. Prefer the per-attempt ``billing_profile`` path in ``evaluate_ev`` whenever per-attempt records exist."""
    if median_days_to_pass is None or not math.isfinite(float(median_days_to_pass)) or median_days_to_pass <= 0:
        return 1
    factor = 1.0 if calendar_days else trading_to_calendar
    return max(1, int(math.ceil(float(median_days_to_pass) * factor / max(int(billing_period_days), 1))))


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


def billing_periods(calendar_days: float | None, billing_period_days: int = 30) -> int:
    """Whole billing periods an attempt that lived ``calendar_days`` calendar days is charged for: ceil(days / period), never below 1.
    Unknown / non-positive -> 1 (one period: a purchase is always charged once)."""
    try:
        d = float(calendar_days)
    except (TypeError, ValueError):
        return 1
    if not math.isfinite(d) or d <= 0:
        return 1
    return max(1, int(math.ceil(d / max(int(billing_period_days), 1))))


def billing_profile(attempt_outcomes, attempt_calendar_days, billing_period_days: int = 30) -> dict | None:
    """Summarise the Monte Carlo's per-attempt records (MonteCarloResult.attempt_outcomes / attempt_calendar_days: 1 = passed, 2 = breached
    before passing, 0 = still open when the history ran out) into what the EV needs: the pass share observed, and the mean number of billing
    periods paid by attempts that passed vs those that did not. None when there are no records."""
    try:
        n = len(attempt_outcomes)
    except TypeError:
        return None
    if n == 0 or len(attempt_calendar_days) != n:
        return None
    out = list(attempt_outcomes)
    per = [billing_periods(d, billing_period_days) for d in attempt_calendar_days]
    passed = [per[i] for i in range(n) if out[i] == 1]
    other = [per[i] for i in range(n) if out[i] != 1]
    cal_all = [float(d) for d in attempt_calendar_days if d is not None and d > 0]
    return {
        "n": n, "p_observed": len(passed) / n, "billing_period_days": int(billing_period_days),
        "periods_pass": (sum(passed) / len(passed)) if passed else None,
        "periods_nonpass": (sum(other) / len(other)) if other else None,
        "periods_all": sum(per) / n,
        "share_busted_before_pass": sum(1 for o in out if o == 2) / n,
        "share_open": sum(1 for o in out if o == 0) / n,
        "mean_calendar_days": (sum(cal_all) / len(cal_all)) if cal_all else None,
    }


def attempt_cost(profile: dict, p: float, price: float) -> float:
    """Expected fee per attempt when a fraction ``p`` of attempts pass: price x (p x periods_pass + (1-p) x periods_nonpass). When one of the two groups is empty in the profile
    (everything passed, or nothing did) its periods fall back to the other group's, so the cost never silently becomes zero."""
    pp = profile.get("periods_pass")
    pn = profile.get("periods_nonpass")
    if pp is None and pn is None:
        pp = pn = profile.get("periods_all") or 1.0
    pp = pn if pp is None else pp
    pn = pp if pn is None else pn
    p = min(max(float(p), 0.0), 1.0)
    return float(price) * (p * float(pp) + (1.0 - p) * float(pn))


def compute_ev_attempts(p: float, q: float, v: float, profile: dict, price: float, activation_fee: float = 0.0) -> float:
    """EV per attempt with per-attempt billing: p x (q x V - A) - expected fee at pass rate p (see attempt_cost)."""
    return float(p) * (float(q) * float(v) - float(activation_fee)) - attempt_cost(profile, p, price)


def break_even_pass_rate_attempts(q: float, v: float, profile: dict, price: float, activation_fee: float = 0.0) -> float | None:
    """The pass rate at which EV(p) = 0 under per-attempt billing (EV is linear in p, so this is its exact root):
    p* = C_fail / (q x V - A - C_pass + C_fail). None when even a certain pass would not cover its own fee."""
    pp = profile.get("periods_pass")
    pn = profile.get("periods_nonpass")
    if pp is None and pn is None:
        pp = pn = profile.get("periods_all") or 1.0
    pp = pn if pp is None else pp
    pn = pp if pn is None else pn
    c_pass, c_fail = float(price) * float(pp), float(price) * float(pn)
    gain = float(q) * float(v) - float(activation_fee)
    denom = gain - c_pass + c_fail
    if gain - c_pass <= 0 or denom <= 0:
        return None
    return min(max(c_fail / denom, 0.0), 1.0)


def expected_max_std_normal(n_trials: int) -> float:
    """Expected maximum of ``n_trials`` i.i.d. standard normals (Bailey/Lopez de Prado approximation).
    0 for n_trials <= 1."""
    n = int(n_trials)
    if n <= 1:
        return 0.0
    a = math.sqrt(2.0 * math.log(n))
    return a - (math.log(math.log(n)) + math.log(4.0 * math.pi)) / (2.0 * a) if n > 2 else 0.5642


MAX_SEARCH_BIAS_HAIRCUT = 0.5   # the search-bias haircut never removes more than half of a measured pass rate


def deflate_pass_rate(p: float, n_trials: int, n_effective: float, *, tunable: bool = True,
                      max_haircut: float = MAX_SEARCH_BIAS_HAIRCUT) -> float:
    """Heuristic search-bias deflation of a pass rate (fraction 0..1).

    Picking the best of ``n_trials`` configurations inflates the winner's measured rate by about
    (sampling noise) x E[max of n_trials standard normals]. With ``n_effective`` independent
    observations behind the rate, noise sd = sqrt(p(1-p)/n_effective). It is an approximation
    (trials are correlated, so it over-corrects) -- use it as a haircut, not as an exact estimate.

    2026-10-27 audit (fix 3), two changes:
      * ``tunable=False`` (the strategy has no free parameters, so nothing was searched) applies NO deflation:
        a fixed rule is not a best-of-N pick however many raw configurations a run logged.
      * the haircut is bounded to ``max_haircut`` of the measured rate (default one half) instead of being allowed to
        drive a real 19% pass rate to 0.0%. The Deflated Sharpe gate still carries the full, unbounded multiple-testing
        correction, so a lucky pick cannot hide behind the cap."""
    p = min(max(float(p), 0.0), 1.0)
    if not tunable or int(n_trials) <= 1:
        return p
    n_eff = max(float(n_effective), 1.0)
    sd = math.sqrt(max(p * (1.0 - p), 1e-9) / n_eff)
    floor = p * (1.0 - min(max(float(max_haircut), 0.0), 1.0))
    return max(floor, p - sd * expected_max_std_normal(n_trials))


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
    # 2026-11-04: pass-rate lower bound (outer bootstrap, fraction) and the EV it implies; EV per calendar month of attempt time (speed-aware)
    p_lb: float | None = None
    ev_lb: float | None = None
    ev_per_month: float | None = None
    mean_attempt_calendar_days: float | None = None
    billing: dict = field(default_factory=dict)
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
    parameters_tuned: bool = True,
    automation_confirmed: bool | None = None,
    challenge_months: int = 1,
    monthly_subscription: bool = False,
    holdout_pass_rate_lb: float | None = None,
    billing_profile: dict | None = None,
    selection_handled_by_holdout: bool = False,
) -> EVVerdict:
    """Compute the EV verdict.

    holdout_pass_rate / window_payout_rate are FRACTIONS measured on the clean holdout only.
    ``window_payout_rate`` is the unconditional P(first payout within the window); q is derived as
    r / p. payout_value is the trader's net dollars per paid account WHEN profit_split_pct is None;
    otherwise it is the gross payout and the split is applied here.
    The pass rate is deflated for ``n_trials`` before the margin test (the EV itself is shown both ways).
    margin: p_deflated must be >= max(break_even * margin_multiple, break_even + margin_points).

    2026-11-04 audit (redesign, "business evidence"):
      * ``holdout_pass_rate_lb`` (fraction; the outer-bootstrap lower bound) makes the bar an EV LOWER BOUND: EV evaluated at that rate must be
        positive and the rate must clear the margin. The point estimate is still shown. When it is None the (deflated) point estimate is
        tested exactly as before.
      * ``selection_handled_by_holdout`` -- the edge was CONFIRMED on untouched data, so selection bias has already been dealt with once and the
        heuristic search-bias haircut is NOT applied a second time (audit P1-1: one uncertainty treatment per quantity).
      * ``billing_profile`` -- per-attempt subscription billing (see the module docstring); without it the scalar ``challenge_months`` is used.
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
    months = max(1, int(challenge_months)) if monthly_subscription else 1
    c = float(challenge_price) * months
    q = min(r / p, 1.0) if p > 0 else 0.0
    n_eff = n_effective if (n_effective and n_effective > 0) else 30.0
    if selection_handled_by_holdout:
        p_def = p                      # the holdout already handled selection; no second haircut on the same quantity
    else:
        p_def = deflate_pass_rate(p, n_trials, n_eff, tunable=parameters_tuned)
    a = max(float(activation_fee or 0.0), 0.0)
    per_attempt = bool(monthly_subscription and billing_profile and billing_profile.get("n"))
    p_lb = None if holdout_pass_rate_lb is None else min(max(float(holdout_pass_rate_lb), 0.0), p_def)
    if per_attempt:
        c = attempt_cost(billing_profile, p_def, float(challenge_price))
        v.p, v.p_deflated, v.q, v.v, v.c = p, p_def, q, net_v, c
        v.ev = compute_ev_attempts(p_def, q, net_v, billing_profile, float(challenge_price), a)
        v.break_even_p = break_even_pass_rate_attempts(q, net_v, billing_profile, float(challenge_price), a)
        v.ev_lb = None if p_lb is None else compute_ev_attempts(p_lb, q, net_v, billing_profile, float(challenge_price), a)
        v.billing = dict(billing_profile)
    else:
        v.p, v.p_deflated, v.q, v.v, v.c = p, p_def, q, net_v, c
        v.ev = compute_ev(p_def, q, net_v, c, a)
        v.break_even_p = break_even_pass_rate(c, q, net_v, a)
        v.ev_lb = None if p_lb is None else compute_ev(p_lb, q, net_v, c, a)
    v.p_lb = p_lb
    v.bankroll_ruin_pct, v.ruin_cap_pct = bankroll_ruin_pct, ruin_cap_pct
    _mad = (billing_profile or {}).get("mean_calendar_days") if billing_profile else None
    if _mad and _mad > 0:
        v.mean_attempt_calendar_days = float(_mad)
        v.ev_per_month = float(v.ev) / max(float(_mad) / float((billing_profile or {}).get("billing_period_days", 30) or 30), 1.0)

    reasons = v.reasons
    if reset_price is None:
        reasons.append("Reset price unknown -> the challenge price is used for re-buys as well.")
    if per_attempt:
        _bp = billing_profile
        reasons.append(
            f"Subscription pricing, billed per attempt: ${float(challenge_price):,.0f} per {int(_bp.get('billing_period_days', 30))} calendar days. "
            f"Attempts that passed paid {float(_bp.get('periods_pass') or 0):.1f} period(s) on average, those that did not paid "
            f"{float(_bp.get('periods_nonpass') or 0):.1f} -> expected fee ${c:,.0f} per attempt at this pass rate."
        )
    elif monthly_subscription:
        reasons.append(
            f"Subscription pricing: ${float(challenge_price):,.0f} per month x {months} expected month(s) "
            f"(median time to pass) = ${c:,.0f} per attempt."
            + (" Time to pass unknown -> one month charged; EV is optimistic for slow passers." if months == 1 and challenge_months <= 1 else "")
        )
    if a > 0:
        reasons.append(f"Activation fee ${a:,.0f} is charged on every pass (EV = p x q x V - p x ${a:,.0f} - c).")
    if selection_handled_by_holdout and n_trials > 1:
        reasons.append("No search-bias haircut: the edge was confirmed on untouched holdout data, which is what handles selection (one correction per quantity).")
    elif n_trials > 1 and not parameters_tuned:
        reasons.append("No search-bias deflation: the strategy has no tunable parameters, so no configuration was picked out of a search.")
    elif n_trials > 1:
        reasons.append(
            f"Pass rate deflated from {p * 100:.1f}% to {p_def * 100:.1f}% for {n_trials} configurations tried "
            f"(~{n_eff:.0f} independent observations; heuristic haircut capped at half the rate, see deflate_pass_rate)."
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
    # the rate the bar is tested at: the pass-rate LOWER bound when one was supplied (EV lower bound > 0 with a margin), else the point estimate
    p_test = v.p_lb if v.p_lb is not None else p_def
    ev_test = v.ev_lb if v.ev_lb is not None else v.ev
    _what = "lower-bound" if v.p_lb is not None else ("holdout" if selection_handled_by_holdout else "deflated holdout")
    if ev_test <= 0:
        blockers.append(f"EV per attempt is ${ev_test:,.2f} (<= 0) at the {_what} pass rate {p_test * 100:.1f}% (break-even {v.break_even_p * 100:.1f}%).")
    elif p_test < need:
        blockers.append(f"{_what} pass rate {p_test * 100:.1f}% is above break-even {v.break_even_p * 100:.1f}% but short of the required margin ({need * 100:.1f}%).")
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
        if ev_test > 0 and (bankroll_ruin_pct is None or not rules_verified or not lookahead_ok):
            v.label = INSUFFICIENT_EVIDENCE
        else:
            v.label = NOT_WORTH_A_FEE
        reasons.extend(blockers)
        return v

    v.label = HIGH_CONFIDENCE if legacy_verdict == "READY" else FORWARD_TEST_CANDIDATE
    reasons.append(
        f"EV ${v.ev:,.2f} per attempt" + (f" (${ev_test:,.2f} at the pass-rate lower bound)" if v.p_lb is not None else "")
        + f" with the {_what} pass rate {p_test * 100:.1f}% clearing break-even {v.break_even_p * 100:.1f}% "
        f"by the required margin ({need * 100:.1f}%), and bankroll ruin {bankroll_ruin_pct:.1f}% under the {ruin_cap_pct:.0f}% cap."
        + (f" EV per calendar month of attempt time: ${v.ev_per_month:,.2f}." if v.ev_per_month is not None else "")
    )
    if v.label == FORWARD_TEST_CANDIDATE:
        reasons.append("This is NOT the 70% READY gate: it is permission to forward-test on a demo / the cheapest account, not proof of an edge.")
    return v
