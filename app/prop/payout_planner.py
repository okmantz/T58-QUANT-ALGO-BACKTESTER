"""Payout-EV planner: turns the Monte Carlo's pass and first-payout probabilities into decisions.

The Monte Carlo already answers "what is the chance one account passes, and what is the chance a funded account pays out".
This module is the thin layer on top that answers what a trader actually has to decide:

* How many evaluations will I probably have to buy before the first payout?          -> ``expected_attempts``
* How much will that cost me in fees?                                                -> ``expected_fee_spend``
* How long will it take, in calendar days?                                           -> ``expected_days_to_first_payout``
* How much should I budget to be 90% sure of getting there?                          -> ``budget_for_confidence``
* Which account is better, Lucid 50k or 100k?                                        -> ``compare_options``
* What does running two evaluations at once do to the timeline and the bill?         -> ``parallel_plan``

The model (every attempt is one purchased evaluation):

* ``p``  chance an attempt passes the evaluation.
* ``q``  chance a PASSED (funded) account reaches its first payout before breaching.
* ``s = p * q``  chance one attempt ends in a first payout. The number of attempts until the first payout is geometric, so
  ``E[attempts] = 1 / s``.
* Fees: the first evaluation costs the challenge price, every re-buy costs the reset price (the challenge price when the firm
  publishes no reset price), and every PASS costs the activation fee. A passed account is the one that pays out with probability
  ``q`` so the expected number of passes before the first payout is ``1 / q``.
* Time: a failed attempt costs ``days_per_failed_attempt`` calendar days, the successful one costs ``days_to_pass`` plus
  ``days_pass_to_payout``.

Parallel evaluations: with ``m`` evaluations started together, the chance that at least one of a round ends in a payout is
``1 - (1 - s) ** m_eff`` where ``m_eff = 1 + (m - 1) * independence``. ``independence`` (0..1) says how differently the lanes
behave: the same strategy on the same days fails together, so the default is a cautious 0.5, and the plan also reports the two
bounds (fully independent, and no benefit at all) so the range is visible instead of one false-precise number.

Everything here is closed-form, deterministic and has no I/O. Money is USD, probabilities are 0..1 inside the module, and the
helper ``from_monte_carlo`` converts the Monte Carlo result's 0..100 percentages.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_PARALLEL_INDEPENDENCE = 0.5
MAX_ATTEMPTS_CAP = 10_000.0          # an attempt count above this means "effectively never"; keep the numbers printable


@dataclass
class PlanInputs:
    """Everything the planner needs for one account option. ``None`` money fields mean "unknown"; the planner then reports
    the fee-dependent outputs as ``None`` rather than guessing a price."""
    label: str
    p_pass: float                                  # 0..1 per attempt
    q_payout: float                                # 0..1 given a pass
    challenge_price: float | None = None           # one evaluation purchase (first attempt)
    reset_price: float | None = None               # re-buy after a failure; None -> challenge_price
    activation_fee: float = 0.0                    # charged on every pass
    profit_split_pct: float = 90.0                 # trader's share of payouts
    first_payout_gross: float | None = None        # expected gross first payout in USD (before the split); None = unknown
    days_to_pass: float | None = None              # calendar days for a passing evaluation
    days_pass_to_payout: float | None = None       # calendar days from funded to first payout
    days_per_failed_attempt: float | None = None   # calendar days a failed attempt lasts; None -> days_to_pass (assumed)
    periods_per_attempt: float = 1.0               # >1 for monthly-subscription firms: billing periods an attempt runs
    preset_key: str | None = None
    notes: list = field(default_factory=list)


def _clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def from_monte_carlo(label: str, mc, *, fee=None, preset_key: str | None = None, challenge_price: float | None = None,
                     reset_price: float | None = None, first_payout_gross: float | None = None) -> PlanInputs:
    """Build planner inputs from a ``MonteCarloResult`` (percentages 0..100) and, optionally, a ``FirmFeeInfo``.

    Explicit ``challenge_price`` / ``reset_price`` win over the fee table, which is how a trader enters a price the table does
    not have (the Lucid 100k price is not published in the fee table)."""
    p = _clamp01(float(getattr(mc, "per_attempt_pass_probability", 0.0) or getattr(mc, "evaluation_pass_probability", 0.0)) / 100.0)
    q_raw = getattr(mc, "per_attempt_payout_probability", 0.0)
    if not q_raw and p > 0:                       # older results: derive q from the chain-level numbers
        q_raw = 100.0 * _clamp01(float(getattr(mc, "first_payout_probability", 0.0)) / max(float(getattr(mc, "evaluation_pass_probability", 0.0)), 1e-9))
    q = _clamp01(float(q_raw) / 100.0)
    days_to_pass = getattr(mc, "median_days_to_pass", None)
    days_to_payout = getattr(mc, "median_days_to_first_payout", None)
    pass_to_payout = (float(days_to_payout) - float(days_to_pass)) if days_to_pass is not None and days_to_payout is not None else None
    if pass_to_payout is not None and pass_to_payout < 0:
        pass_to_payout = 0.0
    notes: list = []
    price = challenge_price if challenge_price is not None else getattr(fee, "challenge_price", None)
    reset = reset_price if reset_price is not None else getattr(fee, "reset_price", None)
    if price is None:
        notes.append("No published evaluation price for this account: enter one to see fee spend.")
    return PlanInputs(
        label=label, p_pass=p, q_payout=q, challenge_price=price, reset_price=reset,
        activation_fee=float(getattr(fee, "activation_fee", 0.0) or 0.0),
        profit_split_pct=float(getattr(fee, "profit_split_pct", 90.0) or 90.0),
        first_payout_gross=first_payout_gross if first_payout_gross is not None else (float(getattr(mc, "median_payout", 0.0) or 0.0) or None),
        days_to_pass=float(days_to_pass) if days_to_pass is not None else None,
        days_pass_to_payout=pass_to_payout, preset_key=preset_key, notes=notes,
    )


@dataclass
class Plan:
    label: str
    p_pass: float
    q_payout: float
    p_first_payout_per_attempt: float
    expected_attempts: float | None
    attempts_for_50pct: int | None
    attempts_for_90pct: int | None
    expected_fee_spend: float | None
    budget_for_90pct: float | None
    expected_days_to_first_payout: float | None
    expected_first_payout_net: float | None
    expected_net_after_fees: float | None
    roi_multiple: float | None
    days_assumed: bool
    verdict: str
    verdict_reason: str
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def p_first_payout_per_attempt(p: float, q: float) -> float:
    return _clamp01(p) * _clamp01(q)


def expected_attempts(p: float, q: float) -> float | None:
    s = p_first_payout_per_attempt(p, q)
    if s <= 0:
        return None
    return min(1.0 / s, MAX_ATTEMPTS_CAP)


def attempts_for_confidence(p: float, q: float, confidence: float) -> int | None:
    """Smallest number of attempts k with P(at least one first payout in k attempts) >= confidence."""
    s = p_first_payout_per_attempt(p, q)
    c = _clamp01(confidence)
    if s <= 0:
        return None
    if s >= 1:
        return 1
    if c <= 0:
        return 0
    return max(1, int(math.ceil(math.log(1.0 - min(c, 0.999999)) / math.log(1.0 - s))))


def _reset_price(inp: PlanInputs) -> float | None:
    return inp.reset_price if inp.reset_price is not None else inp.challenge_price


def expected_fee_spend(inp: PlanInputs) -> float | None:
    """Expected fees until the first payout: first evaluation + re-buys + activation on every pass.
    ``challenge + reset * (E[N] - 1) + activation / q`` scaled by billing periods for subscription firms."""
    if inp.challenge_price is None:
        return None
    n = expected_attempts(inp.p_pass, inp.q_payout)
    if n is None:
        return None
    reset = _reset_price(inp)
    periods = max(float(inp.periods_per_attempt), 1.0)
    evals = float(inp.challenge_price) * periods + float(reset) * periods * (n - 1.0)
    passes = 1.0 / max(_clamp01(inp.q_payout), 1e-9)
    return evals + float(inp.activation_fee) * min(passes, MAX_ATTEMPTS_CAP)


def _fee_for_attempts(inp: PlanInputs, k: int) -> float | None:
    if inp.challenge_price is None or k is None:
        return None
    reset = _reset_price(inp)
    periods = max(float(inp.periods_per_attempt), 1.0)
    # at the k-th attempt the expected number of passes is k * p (each pass is charged the activation fee)
    return (float(inp.challenge_price) * periods + float(reset) * periods * (k - 1)
            + float(inp.activation_fee) * k * _clamp01(inp.p_pass))


def expected_days(inp: PlanInputs) -> tuple[float | None, bool]:
    """(expected calendar days to the first payout, whether the failed-attempt length was assumed)."""
    n = expected_attempts(inp.p_pass, inp.q_payout)
    if n is None or inp.days_to_pass is None:
        return None, False
    success = float(inp.days_to_pass) + float(inp.days_pass_to_payout or 0.0)
    assumed = inp.days_per_failed_attempt is None
    failed = float(inp.days_per_failed_attempt if inp.days_per_failed_attempt is not None else inp.days_to_pass)
    return (n - 1.0) * failed + success, assumed


def _verdict(inp: PlanInputs, spend: float | None, net_first_payout: float | None) -> tuple[str, str]:
    s = p_first_payout_per_attempt(inp.p_pass, inp.q_payout)
    if s <= 0:
        return "NOT VIABLE", "No simulated attempt reached a first payout."
    n = 1.0 / s
    if spend is None:
        return "NEEDS A PRICE", "Enter the evaluation price to see what getting to a payout costs."
    if net_first_payout is None:
        return "NEEDS A PAYOUT SIZE", "The expected first payout is unknown, so the spend cannot be weighed against it."
    if net_first_payout <= spend:
        return "NOT WORTH THE FEES", (f"About {n:.1f} evaluations are needed per first payout, costing ${spend:,.0f} in fees "
                                      f"against ${net_first_payout:,.0f} expected from that first payout.")
    if n > 6:
        return "LONG SHOT", f"Pays for itself on average, but expect roughly {n:.1f} evaluations before the first payout."
    return "WORTH PLAYING", f"About {n:.1f} evaluations per first payout; the first payout covers the fees about {net_first_payout / max(spend, 1e-9):.1f}x."


def plan_option(inp: PlanInputs) -> Plan:
    s = p_first_payout_per_attempt(inp.p_pass, inp.q_payout)
    n = expected_attempts(inp.p_pass, inp.q_payout)
    spend = expected_fee_spend(inp)
    days, assumed = expected_days(inp)
    k90 = attempts_for_confidence(inp.p_pass, inp.q_payout, 0.90)
    k50 = attempts_for_confidence(inp.p_pass, inp.q_payout, 0.50)
    net_first = (float(inp.first_payout_gross) * float(inp.profit_split_pct) / 100.0) if inp.first_payout_gross else None
    net_after = (net_first - spend) if (net_first is not None and spend is not None) else None
    roi = (net_first / spend) if (net_first is not None and spend) else None
    verdict, reason = _verdict(inp, spend, net_first)
    notes = list(inp.notes)
    if assumed and days is not None:
        notes.append("Failed attempts are assumed to last as long as a passing one; real busts are usually faster, so this is conservative.")
    return Plan(
        label=inp.label, p_pass=inp.p_pass, q_payout=inp.q_payout, p_first_payout_per_attempt=s,
        expected_attempts=n, attempts_for_50pct=k50, attempts_for_90pct=k90, expected_fee_spend=spend,
        budget_for_90pct=_fee_for_attempts(inp, k90) if k90 else None, expected_days_to_first_payout=days,
        expected_first_payout_net=net_first, expected_net_after_fees=net_after, roi_multiple=roi, days_assumed=assumed,
        verdict=verdict, verdict_reason=reason, notes=notes,
    )


# --------------------------------------------------------------------------------------------------------------------------
# Parallel evaluations
# --------------------------------------------------------------------------------------------------------------------------
@dataclass
class ParallelPlan:
    lanes: int
    independence: float
    expected_rounds: float | None
    expected_days_to_first_payout: float | None
    expected_fee_spend: float | None
    days_if_independent: float | None
    days_if_fully_correlated: float | None
    days_saved: float | None
    extra_spend: float | None
    note: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _rounds(s: float, lanes_eff: float) -> float | None:
    s_m = 1.0 - (1.0 - _clamp01(s)) ** max(lanes_eff, 1.0)
    if s_m <= 0:
        return None
    return min(1.0 / s_m, MAX_ATTEMPTS_CAP)


def parallel_plan(inp: PlanInputs, lanes: int = 2, independence: float = DEFAULT_PARALLEL_INDEPENDENCE) -> ParallelPlan:
    """What running ``lanes`` evaluations at once does to the timeline and the bill. Rounds are synchronised (a simplification: a lane that
    fails restarts immediately in reality), every lane pays in every round, and the round that produces the payout is paid by all lanes."""
    lanes = max(int(lanes), 1)
    independence = _clamp01(independence)
    s = p_first_payout_per_attempt(inp.p_pass, inp.q_payout)
    single_days, _assumed = expected_days(inp)
    if s <= 0 or single_days is None:
        return ParallelPlan(lanes, independence, None, None, None, None, None, None, None,
                            "Not enough information (no payout in any simulated attempt, or no time data).")
    success = float(inp.days_to_pass) + float(inp.days_pass_to_payout or 0.0)
    failed = float(inp.days_per_failed_attempt if inp.days_per_failed_attempt is not None else inp.days_to_pass)

    def days_for(lanes_eff: float) -> float | None:
        r = _rounds(s, lanes_eff)
        return None if r is None else (r - 1.0) * failed + success

    eff = 1.0 + (lanes - 1) * independence
    rounds = _rounds(s, eff)
    days = days_for(eff)
    days_ind, days_corr = days_for(float(lanes)), days_for(1.0)
    spend = None
    single_spend = expected_fee_spend(inp)
    if inp.challenge_price is not None and rounds is not None:
        reset = _reset_price(inp)
        periods = max(float(inp.periods_per_attempt), 1.0)
        per_lane = float(inp.challenge_price) * periods + float(reset) * periods * (rounds - 1.0) + float(inp.activation_fee) * _clamp01(inp.p_pass) * rounds
        spend = lanes * per_lane
    saved = (single_days - days) if days is not None else None
    extra = (spend - single_spend) if (spend is not None and single_spend is not None) else None
    if lanes == 1:
        note = "One evaluation at a time."
    else:
        note = (f"Starting {lanes} evaluations together is only as good as they are different: the same strategy on the same days tends to fail "
                f"together, so the estimate assumes {independence:.0%} independence. Range: {days_corr:,.0f} days if they move as one, "
                f"{days_ind:,.0f} days if fully independent.")
    return ParallelPlan(lanes=lanes, independence=independence, expected_rounds=rounds, expected_days_to_first_payout=days,
                        expected_fee_spend=spend, days_if_independent=days_ind, days_if_fully_correlated=days_corr,
                        days_saved=saved, extra_spend=extra, note=note)


# --------------------------------------------------------------------------------------------------------------------------
# Comparing accounts
# --------------------------------------------------------------------------------------------------------------------------
def compare_options(options: list[PlanInputs]) -> dict:
    """Plan every option and name the best on each decision axis. Options whose value on an axis is unknown never win that axis."""
    plans = [plan_option(o) for o in options]

    def best(key, lowest=True):
        cands = [(getattr(p, key), p.label) for p in plans if getattr(p, key) is not None]
        if not cands:
            return None
        return (min if lowest else max)(cands)[1]

    return {
        "plans": [p.to_dict() for p in plans],
        "fastest": best("expected_days_to_first_payout"),
        "cheapest": best("expected_fee_spend"),
        "most_likely_per_attempt": best("p_first_payout_per_attempt", lowest=False),
        "best_net_after_fees": best("expected_net_after_fees", lowest=False),
        "headline": _headline(plans),
    }


def _headline(plans: list[Plan]) -> str:
    ranked = [p for p in plans if p.expected_net_after_fees is not None]
    if not ranked:
        return "Enter the missing prices to compare these accounts on cost."
    top = max(ranked, key=lambda p: p.expected_net_after_fees)
    if len(ranked) == 1:
        return f"{top.label}: expect about ${top.expected_net_after_fees:,.0f} after fees at the first payout."
    other = sorted(ranked, key=lambda p: p.expected_net_after_fees, reverse=True)[1]
    gap = top.expected_net_after_fees - other.expected_net_after_fees
    return (f"{top.label} leaves about ${gap:,.0f} more after fees at the first payout than {other.label} "
            f"({top.expected_net_after_fees:,.0f} vs {other.expected_net_after_fees:,.0f}).")
