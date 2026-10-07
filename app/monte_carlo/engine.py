"""
Monte Carlo Prop Simulation.

The primary feature of the application. Instead of asking only "was the
strategy profitable historically?", this engine asks: "if I ran this
strategy through thousands of simulated prop accounts with different
possible trade sequences, how often would I pass and actually get paid?"

Each simulation resamples the historical trade P&L sequence (shuffle /
bootstrap / block-bootstrap for loss-streak stress, with optional slippage
stress) and re-runs it through the exact same prop-rule account simulator
used for the single historical run (app.prop.simulator.simulate_account),
so results are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.execution import Trade
from app.monte_carlo.slippage_model import SessionVolatilitySlippageConfig, apply_session_volatility_slippage
from app.prop.simulator import PropRules, precompute_day_structure, simulate_account


@dataclass
class MonteCarloConfig:
    n_simulations: int = 10_000
    # P1-5: "block_bootstrap" is the default resampling. The old i.i.d.
    # "bootstrap" default destroyed the session clustering, regime
    # dependence, and loss-streak structure the drawdown gates actually
    # evaluate -- the pass probability being optimized was computed on
    # paths a real strategy would never produce. Explicit opt-out:
    # method="bootstrap" reproduces the pre-fix i.i.d. behavior exactly.
    method: str = "block_bootstrap"  # "shuffle" | "bootstrap" | "block_bootstrap"
    # Block size used when method == "block_bootstrap". None (default) =
    # auto: scaled to typical trades/day from the trade timestamps --
    # max(2, round(n_trades / n_trading_days)) -- so a block approximates
    # one trading day and preserves day-level clustering. An explicit int
    # overrides the heuristic. NOTE: session-conditioned block resampling
    # is NOT implemented -- _resample_pnls has no session labels, so
    # blocks are contiguous NON-CIRCULAR slices of the trade sequence
    # (v5: a block never wraps around the end of the series, which would
    # splice the series' head onto its tail and fabricate a continuity
    # that never existed).
    block_size: int | None = None
    slippage_stress_pct: float = 0.0  # extra % cost applied to every trade
    # Session/volatility-aware slippage (app.monte_carlo.slippage_model) -- applied ONCE to the
    # historical trade pool before resampling, independent of and in addition to
    # slippage_stress_pct above. Disabled by default (opt-in): every existing caller of
    # MonteCarloConfig() gets byte-identical output to before this field existed.
    session_slippage: SessionVolatilitySlippageConfig = field(default_factory=SessionVolatilitySlippageConfig)
    random_seed: int | None = 42
    # When True, each of the n_simulations resampled paths is run through
    # simulate_account with reset_on_breach=True instead of stopping at the
    # first bust: within that ONE resampled trade ordering, a bust snaps
    # the account back to account_size and keeps walking the rest of that
    # SAME path (see app.prop.simulator.simulate_account's docstring).
    # This answers "if I mechanically rebought every time I busted, how
    # many attempts before this simulated timeline ran out, and what
    # fraction passed" -- drawn from thousands of plausible resampled
    # orderings of the real trade history, rather than the single order
    # that actually happened (which is what
    # app.prop.survival_engine.simulate_reset_chain measures instead, by
    # drawing a fresh independent resample per attempt). Default False:
    # every existing caller gets byte-identical output to before this
    # field existed. Turning it on only changes evaluation_pass_
    # probability / first_payout_probability et al. to mean "at least one
    # attempt in the chain" rather than "the one attempt" -- see
    # MonteCarloResult's new attempts_* fields for the chain-level detail
    # that distinction papers over.
    reset_on_breach: bool = True  # 2026-09-29: paths keep going after a bust (fresh account) -- log, don't stop


def default_method_for_adaptive_risk(adaptive_risk) -> str:
    """P1-5: block bootstrap is now this engine's default resampling for
    every caller (see MonteCarloConfig.method) -- the i.i.d. `bootstrap`
    method explicitly discards any real streakiness/regime-clustering a
    strategy's trade sequence has (see run_monte_carlo's methodology_note
    below), which is a real mismatch once a regime-aware adaptive-risk
    throttle is in play (app.backtest.adaptive_risk's volatility_percentile
    trigger): the throttle's whole value proposition is that bad conditions
    cluster in TIME, but an i.i.d. resample scatters every trade's P&L
    independently across simulated paths, so the very clustering the
    throttle is built to react to never shows up in the ruin estimate it's
    supposed to be protecting. `block_bootstrap` preserves local runs of
    consecutive trades instead.

    Returns "block_bootstrap" always now (both branches) -- kept as a
    function (rather than deleted) so every existing caller keeps working
    unchanged. A caller that wants the old i.i.d. behavior regardless can
    still pass method="bootstrap" explicitly."""
    return "block_bootstrap"


@dataclass
class MonteCarloResult:
    n_simulations: int
    evaluation_pass_probability: float
    first_payout_probability: float
    failure_before_payout_probability: float
    multiple_payout_probability: float

    median_days_to_pass: float | None
    median_days_to_first_payout: float | None
    average_days_to_first_payout: float | None

    median_return_pct: float
    mean_return_pct: float
    expected_payout: float
    median_payout: float
    total_simulated_withdrawals: float

    median_drawdown_pct: float
    p95_drawdown_pct: float
    worst_drawdown_pct: float
    risk_of_ruin_pct: float
    median_max_losing_streak: float
    worst_max_losing_streak: int

    return_percentiles: dict = field(default_factory=dict)      # {5,25,50,75,95: pct}
    drawdown_percentiles: dict = field(default_factory=dict)
    days_to_payout_distribution: list = field(default_factory=list)
    # Full per-simulation distributions, kept for charting (e.g. a return
    # histogram). Not shown in the flat summary tables, only used by the
    # HTML report's charts.
    return_distribution: list = field(default_factory=list)
    drawdown_distribution: list = field(default_factory=list)

    # -- reset-on-breach chain summary (only meaningful when cfg.reset_on_breach
    # was True; all zero/empty at their defaults otherwise, matching this
    # result's shape before the reset_on_breach chain feature existed) --
    reset_on_breach: bool = False
    mean_attempts_per_path: float = 0.0          # avg # of mechanical rebuys per simulated path
    median_attempts_per_path: float = 0.0
    p95_attempts_per_path: float = 0.0
    any_attempt_pass_probability: float = 0.0    # % of paths where >=1 attempt in the chain passed eval
    any_attempt_payout_probability: float = 0.0  # % of paths where >=1 attempt in the chain reached payout
    attempts_distribution: list = field(default_factory=list)  # per-path total_attempts, for charting

    # FIX (RESET-ACCT-002): evaluation_pass_probability/first_payout_probability
    # above answer "did >=1 attempt in the (possibly long) reset chain ever
    # pass/reach payout" once reset_on_breach is on -- with hundreds of
    # mechanical rebuys per path (see mean_attempts_per_path), that can look
    # like a strong number even for a strategy whose SINGLE account has only
    # a modest real chance, because a long enough chain eventually clears a
    # low bar almost by construction. These three fields instead pool every
    # independent attempt across every simulated path (using each path's own
    # AccountSimResult.attempts_passed/attempts_reached_payout/total_attempts
    # -- see app.prop.simulator): per_attempt_pass_probability is passed
    # evals / attempted evals, and per_attempt_payout_probability is funded
    # attempts reaching a first payout / FUNDED (passed) attempts -- an
    # attempt that blew its eval never had a payout chance, so it does not
    # dilute the payout rate (Owen's accounting: pass two evals, one of
    # the two funded accounts pays out -> 50%). When reset_on_breach is
    # False, every path has exactly one attempt, so these are IDENTICAL to
    # evaluation_pass_probability/first_payout_probability -- byte-for-byte
    # no change for the default, far more common case. Only diverges from
    # the chain-level fields once reset_on_breach is on and a path's chain
    # runs more than one attempt.
    per_attempt_pass_probability: float = 0.0
    per_attempt_payout_probability: float = 0.0
    per_attempt_failure_before_payout_probability: float = 0.0
    total_independent_attempts: int = 0          # sum of total_attempts across every simulated path

    # v5 (2026-10-04): Wilson 95% confidence intervals (0-100 scale) for
    # the path-level pass and first-payout probabilities, computed over
    # the n_simulations simulated paths via the Wilson score interval
    # (see _wilson_score_interval). A point estimate vs a hard 70%/50%
    # acceptance bar flips inside MC noise -- the acceptance verdict
    # (app.scoring.t58_scorecard's score_from_results, which drives
    # app.orchestration.full_pipeline._make_verdict) gates on the LOWER
    # bound of these intervals, not the point estimate, so a strategy
    # only reads as "passing the bar" when the bar clears the pessimistic
    # end of the sampling noise. (0.0, 0.0) on a result built before this
    # change (e.g. deserialized) means "unknown, not zero".
    # v6 (2026-10-04): SUPERSEDED for gating -- the acceptance verdict now
    # gates on the PER-ATTEMPT intervals (per_attempt_pass_ci95 /
    # per_attempt_payout_ci95) below, falling back to these only for
    # results built before the per-attempt fields existed. Kept populated
    # for reporting continuity.
    pass_probability_ci95: tuple = (0.0, 0.0)
    payout_probability_ci95: tuple = (0.0, 0.0)

    # v6 (2026-10-04): Wilson 95% confidence intervals (0-100 scale) for
    # the PER-ATTEMPT pass and first-payout probabilities -- the honest
    # "will ONE account attempt succeed" numbers (see the per_attempt_*
    # fields above). The chain-level pass_probability_ci95 /
    # payout_probability_ci95 above are inflated almost by construction
    # once reset_on_breach is on (a long enough rebuy chain eventually
    # clears a low bar), so the acceptance verdict
    # (app.scoring.t58_scorecard.score_from_results -> _make_verdict)
    # gates on the LOWER bound of THESE per-attempt intervals, falling
    # back to the chain-level interval only for results built before
    # this change. The per-attempt pool treats every independent attempt
    # across every simulated path as one Bernoulli trial; attempts
    # within the same path share a resampled trade ordering, so this is
    # an approximation, not a proof of independence -- it is still the
    # right number to gate on, because the alternative (gating on the
    # chain-level interval) answers "did a mechanical rebuyer who
    # rebought hundreds of times eventually pass," which is not the
    # question anyone is asking. (0.0, 0.0) on a result built before
    # this change (e.g. deserialized) means "unknown, not zero".
    per_attempt_pass_ci95: tuple = (0.0, 0.0)
    per_attempt_payout_ci95: tuple = (0.0, 0.0)

    # MC-004: what this run's resampling actually did, in plain language,
    # so `evaluation_pass_probability` isn't read as a stronger claim than
    # it is. Every consumer of this number (Search Lab, Quick Optimizer,
    # Evolution Lab, Full Pipeline's verdict, CPCV) inherits this same
    # caveat -- see run_monte_carlo's docstring for the full reasoning.
    methodology_note: str = ""

    # -- headline numbers (Owen's accounting) --------------------------------
    # What reports/dashboards should show as "the" eval-pass / payout
    # rates: per-attempt (passed/attempted evals, funded reaching
    # payout/funded) when reset_on_breach chained attempts, because the
    # chain-level fields above then answer "did >=1 attempt in a long
    # mechanical rebuy chain ever pass" -- true but misleading as a
    # headline. When reset_on_breach is off the two are identical, so
    # these properties are always safe to display.
    @property
    def headline_evaluation_pass_probability(self) -> float:
        if self.reset_on_breach and self.total_independent_attempts:
            return self.per_attempt_pass_probability
        return self.evaluation_pass_probability

    @property
    def headline_first_payout_probability(self) -> float:
        if self.reset_on_breach and self.total_independent_attempts:
            return self.per_attempt_payout_probability
        return self.first_payout_probability

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def _max_losing_streak(pnls: np.ndarray) -> int:
    # P2-7: breakeven (pnl <= 0) counts as non-winning here -- consistent
    # with app.backtest.statistics' win-rate/max-consecutive-losers, which
    # likewise treat breakeven as a loss for streak purposes.
    best = cur = 0
    for p in pnls:
        cur = cur + 1 if p <= 0 else 0
        best = max(best, cur)
    return best


# v5 (2026-10-04): minimum trades for any MC-derived verdict. A fold/path
# with fewer trades than this contributes NO verdict (scores 0.0): the
# resample has too few independent trades for the pass probability -- and
# its Wilson CI -- to mean anything, and scoring such a slice at face
# value lets thin-slice noise climb the search objective. ~15 = a handful
# of trading days at the typical 2-5 trades/day, the smallest sample
# whose block bootstrap isn't pure noise.
MIN_TRADES_FOR_VERDICT = 15


def _wilson_score_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion, returned on the
    0-100 scale this module reports probabilities on. Unlike the Wald
    (normal-approx) interval, Wilson stays inside [0, 1] and stays honest
    at the boundaries (k=0, k=n) -- exactly where a 69.2%-vs-70%
    acceptance-verdict flip lives inside MC noise. The acceptance
    verdict (app.scoring.t58_scorecard.score_from_results via
    app.orchestration.full_pipeline._make_verdict) gates on the LOWER
    bound of the PER-ATTEMPT interval (per_attempt_pass_ci95), not the
    point estimate and not the chain-level pass_probability_ci95 -- the
    chain-level number answers "did a mechanical rebuyer who rebought
    hundreds of times eventually pass," which is not the question the
    verdict is meant to ask."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * ((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) ** 0.5 / denom
    lo = max(0.0, center - half) * 100.0
    hi = min(1.0, center + half) * 100.0
    return (lo, hi)


def _effective_block_size(cfg: MonteCarloConfig, n_trades: int, n_trading_days: int | None) -> int:
    """P1-5: resolves the block size for block bootstrap. An explicit
    cfg.block_size always wins; otherwise scale to typical trades/day from
    the trade timestamps: max(2, round(n_trades / n_trading_days)). When
    the day count is unknown (e.g. _resample_pnls called without dates),
    fall back to treating each trade as its own day, which lands on the
    max(2, ...) floor -- documented here so the fallback is visible."""
    if getattr(cfg, "block_size", None):
        return int(cfg.block_size)
    days = n_trading_days if n_trading_days and n_trading_days > 0 else n_trades
    return max(2, round(n_trades / days)) if days else 2


def _resample_indices(rng: np.random.Generator, n: int, method: str, block_size: int) -> np.ndarray:
    """Index array implementing each resampling method. Block bootstrap
    uses contiguous NON-CIRCULAR blocks (v5: a block never wraps around
    the end of the sequence -- start is drawn from 0..n-block_size so
    every block is a real contiguous slice; wrapping would splice the
    series' head onto its tail and fabricate continuity that never
    existed). Session-conditioned block resampling is NOT implemented --
    this function never sees session labels, so blocks are plain
    contiguous slices. Returning indices (rather than values) lets
    callers resample parallel per-trade arrays (P&L + initial risk) with
    the same draw."""
    if method == "shuffle":
        return rng.permutation(n)
    if method == "block_bootstrap":
        if block_size >= n:
            # Degenerate: the whole series is one block -- return it in
            # order rather than wrapping fragments.
            return np.arange(n)
        idx: list[int] = []
        while len(idx) < n:
            start = int(rng.integers(0, n - block_size + 1))
            idx.extend(start + j for j in range(block_size))
        return np.array(idx[:n])
    # default: iid bootstrap with replacement
    return rng.integers(0, n, size=n)


def _resample_pnls(
    rng: np.random.Generator, pnls: np.ndarray, cfg: MonteCarloConfig,
    n_trading_days: int | None = None,
) -> np.ndarray:
    n = len(pnls)
    block_size = _effective_block_size(cfg, n, n_trading_days)
    return np.asarray(pnls)[_resample_indices(rng, n, cfg.method, block_size)]


def _trade_dollar_risk(t: Trade) -> float:
    """Per-trade initial dollar risk for the P1-3 floating-drawdown proxy:
    the configured dollar risk when known, else initial_risk (price units)
    times size, else 0.0."""
    if getattr(t, "intended_risk_dollars", None):
        return float(t.intended_risk_dollars)
    if getattr(t, "initial_risk", None):
        return float(t.initial_risk) * float(getattr(t, "size", 0.0) or 0.0)
    return 0.0


def _apply_slippage_stress(pnls: np.ndarray, stress_pct: float) -> np.ndarray:
    if not stress_pct:
        return pnls
    factor = stress_pct / 100.0
    stressed = pnls.copy()
    stressed[stressed > 0] *= (1 - factor)
    stressed[stressed <= 0] *= (1 + factor)
    return stressed


def eval_pass_probability_for_trades(
    trades: list[Trade],
    rules: PropRules,
    mc_cfg: MonteCarloConfig | None = None,
) -> float:
    """
    Convenience wrapper around run_monte_carlo() that returns just the
    single number nearly every fold-level / candidate-level scoring path
    in the app actually wants: the PER-ATTEMPT probability of passing the
    prop evaluation -- i.e. MonteCarloResult.per_attempt_pass_probability:
    of every independent account attempt this Monte Carlo run
    represents, what fraction passed.

    BEHAVIOR CHANGE (v5, 2026-10-04): this used to return the
    CHAIN-LEVEL MonteCarloResult.evaluation_pass_probability ("did >=1
    attempt in the mechanically-rebought chain ever pass"). With
    reset_on_breach on (the default), a long chain eventually clears a
    low bar almost by construction, so the chain-level number looked
    strong even for a strategy whose single account had only a modest
    real chance -- every caller (WF, CPCV, regime selector, Forge) was
    climbing an inflated objective. The per-attempt number is the actual
    "will ONE account attempt succeed" question. When reset_on_breach
    is off, every path has exactly one attempt and the two are
    identical, so nothing changes for that case. The chain-level value
    remains available for reporting on the full MonteCarloResult --
    keep using run_monte_carlo()'s chain-level fields for display;
    do NOT use them for scoring.

    MIN-TRADES FLOOR (v5): a fold/path with fewer than
    MIN_TRADES_FOR_VERDICT (15) trades scores 0.0 -- it contributes no
    verdict, because the resample has too few independent trades for a
    pass probability (or its Wilson CI) to mean anything. Returns 0.0
    (not an exception) for an empty/too-small trade list, since "this
    slice produced nothing worth passing" is itself a valid, low, fold
    score rather than a hard failure.

    This is the shared primitive behind making "probability of passing"
    (rather than raw backtest profit, win rate, or R:R) the one thing
    every test in the app -- Iterative Refinement, Walk-Forward
    Optimization, the walk-forward-aware GA, CPCV, the Evolution Lab, and
    Quick Optimize/Full Pipeline -- actually optimizes and validates
    against, including at the per-fold / per-path level where those
    modules previously fell back to a plain backtest-stats metric like
    profit_factor because no Monte Carlo had been run yet for that slice
    of data.

    Uses a smaller default simulation count than a final-report Monte
    Carlo run (this is called once per fold/path/generation, often many
    times per search) -- callers that care about that tradeoff should
    pass their own mc_cfg.
    """
    if not trades or len(trades) < MIN_TRADES_FOR_VERDICT:
        return 0.0
    cfg = mc_cfg or MonteCarloConfig(n_simulations=500)
    try:
        result = run_monte_carlo(trades, rules, cfg)
    except ValueError:
        return 0.0
    return result.per_attempt_pass_probability


def _reset_on_breach_note(cfg: "MonteCarloConfig") -> str:
    """MC-005: plain-language addendum for when reset_on_breach is on --
    the headline pass/payout numbers now mean 'at least one attempt in a
    mechanically-rebought chain,' not 'the one attempt that happened to
    run,' and that distinction needs to be visible next to the number."""
    if cfg.method != "block_bootstrap":
        method_caveat = (
            " Note this run used i.i.d. resampling rather than block "
            "bootstrap -- see the block_bootstrap method for a version "
            "that keeps bad stretches of trades clustered together the "
            "way they'd actually cluster in reality, instead of smearing "
            "a bad week randomly across the whole simulated chain."
        )
    else:
        method_caveat = ""
    return (
        " reset_on_breach was ON: within each of those resampled paths, a bust did not end the "
        "simulation -- the account snapped back to a fresh balance and kept walking the rest of "
        "that same path, so evaluation_pass_probability / first_payout_probability above mean "
        "'at least one attempt in that path's chain succeeded,' not 'the one attempt succeeded.' "
        "See mean_attempts_per_path and attempts_distribution for how many mechanical rebuys that "
        "typically took, and app.monte_carlo.bankroll for turning this into an actual "
        f"fees-vs-payout survival question.{method_caveat}"
    )


def _methodology_note(
    cfg: "MonteCarloConfig", n_trades: int, selection_bias_caveat: bool,
    n_trading_days: int | None = None,
) -> str:
    """MC-004: plain-language description of what THIS run's resampling
    actually did, so evaluation_pass_probability isn't read as a
    stronger claim than it is. Two things this deliberately calls out
    that were previously only in code comments, not anywhere a person
    reading a report would see them:

    (1) The resampling method. The default ("bootstrap") is i.i.d.
    resampling with replacement -- every trade's P&L is treated as
    statistically independent of its neighbors, which discards any real
    streakiness/regime-clustering the strategy actually has. "shuffle"
    (no replacement) and "block_bootstrap" (preserves local runs of
    trades) make different, also-real tradeoffs; whichever ran, the
    person reading the number should know which.

    (2) The FIXED trading-day calendar. Every simulation reuses the
    exact same calendar dates from the historical trade sequence --
    only the P&L VALUES are resampled onto that fixed timeline. That
    means this Monte Carlo run answers "given this exact trade-timing
    skeleton, how do outcomes vary if the P&L values landed in a
    different order?", not "how would this strategy's whole trajectory
    (including trade frequency/timing) vary across different possible
    histories?" -- a materially narrower question than the headline
    number alone suggests.

    (3) P1-3: the account simulator trails drawdown on REALIZED balance
    only (trade-close equity) -- it does not model floating intrabar
    drawdown against the trailing peak. For prop firms whose trailing
    drawdown is enforced on real-time floating equity, this UNDERSTATES
    how often the account would actually fail: a simulated path that
    survives here could easily have breached mid-trade in reality, and
    neither the pass probability nor the risk-of-ruin above accounts for
    it. To degrade the check toward that stricter reality, re-run with
    PropRules(floating_drawdown_mode="adverse"), which assumes every
    trade drew down to its full initial risk before closing -- a
    documented, deliberately conservative approximation, not a measured
    floating-equity series.

    selection_bias_caveat: True when the caller knows (or can't rule
    out) that these trades came from a search/optimization step that
    already selected them for scoring well -- in that case this Monte
    Carlo run is resampling-order robustness LAYERED ON TOP OF whatever
    selection bias already exists in the input trades, not independent
    out-of-sample evidence on its own.
    """
    block_size = _effective_block_size(cfg, n_trades, n_trading_days)
    method_descriptions = {
        "bootstrap": (
            f"resampled {cfg.n_simulations:,} times with i.i.d. bootstrap (each of the "
            f"{n_trades} historical trades treated as independent -- any real streakiness "
            "in the strategy's actual results is not preserved)"
        ),
        "shuffle": (
            f"resampled {cfg.n_simulations:,} times by shuffling the order of the same "
            f"{n_trades} historical trades (no repeats -- the exact multiset of outcomes "
            "is preserved, only their order varies)"
        ),
        "block_bootstrap": (
            f"resampled {cfg.n_simulations:,} times with block bootstrap (block size "
            f"{block_size}, preserving short local runs of the {n_trades} historical "
            "trades rather than treating each one as fully independent)"
        ),
    }
    method_text = method_descriptions.get(
        cfg.method, f"resampled {cfg.n_simulations:,} times (method: {cfg.method})"
    )
    note = (
        f"Based on {n_trades} historical trades, {method_text}, replayed onto the SAME fixed "
        "trading-day calendar every time (only the P&L order/values vary -- trade timing and "
        "frequency do not). This measures robustness to trade-ordering on this exact trade "
        "sequence, not an independent out-of-sample test."
    )
    # P1-3: honest paragraph -- the sim trails on realized balance only.
    note += (
        " Drawdown-modeling caveat: the account simulator trails drawdown on REALIZED "
        "balance only (trade-close equity) -- it does not model floating intrabar drawdown "
        "against the trailing peak. For prop firms whose trailing drawdown is enforced on "
        "real-time floating equity, this UNDERSTATES how often the account would actually "
        "fail: a simulated path that survives here could have breached mid-trade in reality, "
        "and neither the pass probability nor the risk-of-ruin above accounts for it. "
        "Re-run with PropRules(floating_drawdown_mode=\"adverse\") to degrade the check "
        "with each trade's initial risk as a conservative floating-drawdown proxy "
        "(documented approximation -- assumes every trade drew down to its full initial "
        "risk before closing)."
    )
    if selection_bias_caveat:
        note += (
            " These trades came from an optimization/search step -- treat this result as "
            "resampling-order robustness on top of whatever selection bias already exists in "
            "how these trades were chosen, not as independent validation by itself."
        )
    if cfg.reset_on_breach:
        note += _reset_on_breach_note(cfg)
    # v5: ruin now counts ANY simulated account death in a path
    # (daily-loss-limit breach -- the dominant death mode -- max-drawdown
    # breach, or inactivity closure), not just max-drawdown breaches; and
    # the pass/payout probabilities above carry Wilson 95% confidence
    # intervals (per_attempt_pass_ci95 / per_attempt_payout_ci95 -- the
    # PER-ATTEMPT, single-account intervals, which supersede the legacy
    # chain-level pass_probability_ci95 / payout_probability_ci95), whose
    # LOWER bounds are what the acceptance verdict gates on.
    note += (
        " Risk of ruin counts any simulated account death (daily-loss-limit "
        "breach, max-drawdown breach, or inactivity closure), not just "
        "max-drawdown breaches. Wilson 95% CIs on the PER-ATTEMPT "
        "pass/payout probabilities (not the chain-level intervals) are "
        "reported alongside the point estimates; the acceptance verdict "
        "gates on their lower bounds."
    )
    return note


def run_monte_carlo(
    trades: list[Trade],
    rules: PropRules,
    cfg: MonteCarloConfig | None = None,
    selection_bias_caveat: bool = False,
) -> MonteCarloResult:
    """
    selection_bias_caveat: MC-004. Pass True when `trades` are known (or
    can't be ruled out) to have come from a search/optimization step that
    already selected them for scoring well on some metric -- e.g. a
    genome the walk-forward-aware GA just picked. Only changes the
    wording of the returned result's `methodology_note`; never changes
    any numeric output.
    """
    cfg = cfg or MonteCarloConfig()
    if not trades:
        raise ValueError("Cannot run Monte Carlo simulation with zero trades.")

    rng = np.random.default_rng(cfg.random_seed)
    base_pnls = apply_session_volatility_slippage(trades, cfg.session_slippage)
    base_dates = [pd.Timestamp(t.entry_time).normalize() for t in trades]
    n_trading_days = len(set(base_dates))
    # P1-5: effective block size scaled to typical trades/day (see
    # _effective_block_size); resolved once here so the methodology note
    # reports the number actually used.
    eff_block_size = _effective_block_size(cfg, len(trades), n_trading_days)
    # P1-3: per-trade initial dollar risks, resampled WITH the P&Ls (same
    # index draw) so each simulated path keeps each trade's own risk
    # paired with its P&L. Only consumed when
    # rules.floating_drawdown_mode == "adverse"; all-zero/None otherwise,
    # which keeps that check a no-op (byte-identical to not passing it).
    base_risks = np.array([_trade_dollar_risk(t) for t in trades], dtype=float)
    has_risks = bool(np.any(base_risks > 0))
    # Every simulation below reassigns the SAME fixed calendar dates
    # (base_dates never changes) to a resampled sequence of P&L values --
    # only sim_pnls' order/values differ per simulation. That means the
    # date-to-trading-day bookkeeping simulate_account would otherwise
    # rebuild from scratch (via per-trade pandas Timestamp parsing and
    # dict lookups) on every single one of cfg.n_simulations calls is
    # actually identical every time, so it's computed once here instead.
    # See app.prop.simulator.DayStructure's docstring for the full
    # reasoning; this is a pure performance change with no effect on any
    # output value.
    day_structure = precompute_day_structure(base_dates)

    passed_flags, first_payout_flags, failed_before_payout_flags, multiple_payout_flags = [], [], [], []
    days_to_pass_list, days_to_first_payout_list = [], []
    return_pcts, payout_amounts, drawdown_pcts, losing_streaks = [], [], [], []
    total_withdrawals = 0.0
    attempts_per_path: list[int] = []
    sum_attempts_passed = 0
    sum_attempts_reached_payout = 0
    sum_total_attempts = 0
    # v5: per-path account-death flag -- True when ANY attempt in this
    # path's reset chain died (breached a prop rule). The dominant death
    # mode in practice is failure_reason="daily_loss_limit" (a single bad
    # day kills the account long before the trailing max-drawdown floor is
    # ever touched), so counting only max-DD breaches -- the old
    # behavior -- made risk_of_ruin blind to how most simulated accounts
    # actually die.
    death_flags: list[bool] = []

    for _ in range(cfg.n_simulations):
        sim_idx = _resample_indices(rng, len(base_pnls), cfg.method, eff_block_size)
        sim_pnls = base_pnls[sim_idx]
        sim_pnls = _apply_slippage_stress(sim_pnls, cfg.slippage_stress_pct)
        sim_risks = base_risks[sim_idx] if has_risks else None

        result = simulate_account(
            sim_pnls, base_dates, rules, _day_structure=day_structure,
            reset_on_breach=cfg.reset_on_breach,
            trade_initial_risks=sim_risks,
        )

        passed_flags.append(result.passed_evaluation)
        first_payout_flags.append(result.reached_first_payout)
        failed_before_payout_flags.append(result.failed and not result.reached_first_payout)
        multiple_payout_flags.append(len(result.payouts) > 1)
        attempts_per_path.append(result.total_attempts)
        sum_attempts_passed += result.attempts_passed
        sum_attempts_reached_payout += result.attempts_reached_payout
        sum_total_attempts += result.total_attempts
        # v5: count EVERY account death, whatever the failure_reason
        # ("daily_loss_limit", "max_drawdown (...)", inactivity closure).
        # result.attempts always holds at least the attempt-#1 record, and
        # more than one record only when reset_on_breach rebuys into the
        # remaining history after a bust.
        death_flags.append(any(a.failed for a in result.attempts))

        if result.days_to_pass is not None:
            days_to_pass_list.append(result.days_to_pass)
        if result.first_payout_day_index is not None:
            days_to_first_payout_list.append(result.first_payout_day_index)

        return_pcts.append((result.final_balance - rules.account_size) / rules.account_size * 100.0)
        payout_amounts.append(result.total_payout_amount)
        total_withdrawals += result.total_payout_amount
        drawdown_pcts.append(result.max_drawdown_pct_reached)
        losing_streaks.append(_max_losing_streak(sim_pnls))

    passed_arr = np.array(passed_flags)
    first_payout_arr = np.array(first_payout_flags)
    failed_before_payout_arr = np.array(failed_before_payout_flags)
    multiple_payout_arr = np.array(multiple_payout_flags)
    return_arr = np.array(return_pcts)
    payout_arr = np.array(payout_amounts)
    dd_arr = np.array(drawdown_pcts)
    streak_arr = np.array(losing_streaks)

    # v5 (2026-10-04): ruin = the account DIED in this path -- ANY attempt
    # in the chain breached a prop rule -- not just "the max-drawdown
    # floor was hit". The old definition (drawdown_pct >=
    # max_drawdown_pct) ignored daily-loss-limit deaths, which are the
    # dominant death mode: most busts never get near the trailing DD
    # floor because a single bad day ends them first. Death counts are
    # path-level (did >=1 attempt die), consistent with the other
    # headline path-level probabilities on this result.
    death_arr = np.array(death_flags)
    ruin_arr = death_arr
    attempts_arr = np.array(attempts_per_path)

    def pct(arr, q):
        return float(np.percentile(arr, q)) if len(arr) else 0.0

    pass_ci95 = _wilson_score_interval(int(passed_arr.sum()), len(passed_arr))
    payout_ci95 = _wilson_score_interval(int(first_payout_arr.sum()), len(first_payout_arr))
    # v6 (2026-10-04): Wilson 95% CIs on the PER-ATTEMPT (single-account)
    # numbers -- the honest gate. See the per_attempt_*_ci95 fields on
    # MonteCarloResult for why these, not the chain-level intervals
    # above, are what the acceptance verdict gates on.
    per_attempt_pass_ci95 = _wilson_score_interval(int(sum_attempts_passed), int(sum_total_attempts))
    # Payout is gated on having PASSED the eval first, so its honest
    # denominator is funded (passed) attempts, not every eval attempt --
    # Owen's accounting: "of the accounts that got funded, how many
    # reached a first payout" (1 of 2 funded = 50%, even when other
    # eval attempts around them failed).
    per_attempt_payout_ci95 = _wilson_score_interval(int(sum_attempts_reached_payout), int(sum_attempts_passed))

    result = MonteCarloResult(
        n_simulations=cfg.n_simulations,
        evaluation_pass_probability=float(passed_arr.mean() * 100),
        first_payout_probability=float(first_payout_arr.mean() * 100),
        failure_before_payout_probability=float(failed_before_payout_arr.mean() * 100),
        multiple_payout_probability=float(multiple_payout_arr.mean() * 100),
        median_days_to_pass=float(np.median(days_to_pass_list)) if days_to_pass_list else None,
        median_days_to_first_payout=float(np.median(days_to_first_payout_list)) if days_to_first_payout_list else None,
        average_days_to_first_payout=float(np.mean(days_to_first_payout_list)) if days_to_first_payout_list else None,
        median_return_pct=pct(return_arr, 50),
        mean_return_pct=float(return_arr.mean()) if len(return_arr) else 0.0,
        expected_payout=float(payout_arr.mean()) if len(payout_arr) else 0.0,
        median_payout=pct(payout_arr, 50),
        total_simulated_withdrawals=float(total_withdrawals),
        median_drawdown_pct=pct(dd_arr, 50),
        p95_drawdown_pct=pct(dd_arr, 95),
        worst_drawdown_pct=float(dd_arr.max()) if len(dd_arr) else 0.0,
        risk_of_ruin_pct=float(ruin_arr.mean() * 100),
        median_max_losing_streak=float(np.median(streak_arr)) if len(streak_arr) else 0.0,
        worst_max_losing_streak=int(streak_arr.max()) if len(streak_arr) else 0,
        return_percentiles={q: pct(return_arr, q) for q in (5, 25, 50, 75, 95)},
        drawdown_percentiles={q: pct(dd_arr, q) for q in (5, 25, 50, 75, 95)},
        days_to_payout_distribution=days_to_first_payout_list,
        return_distribution=return_arr.tolist(),
        drawdown_distribution=dd_arr.tolist(),
        methodology_note=_methodology_note(cfg, len(trades), selection_bias_caveat, n_trading_days=n_trading_days),
        reset_on_breach=cfg.reset_on_breach,
        mean_attempts_per_path=float(attempts_arr.mean()) if len(attempts_arr) else 0.0,
        median_attempts_per_path=pct(attempts_arr, 50),
        p95_attempts_per_path=pct(attempts_arr, 95),
        any_attempt_pass_probability=float(passed_arr.mean() * 100),
        any_attempt_payout_probability=float(first_payout_arr.mean() * 100),
        attempts_distribution=attempts_arr.tolist(),
        per_attempt_pass_probability=float(sum_attempts_passed / sum_total_attempts * 100) if sum_total_attempts else 0.0,
        # Funded-only denominator (see the CI note above): of attempts
        # that PASSED their eval, the fraction that reached a first
        # payout. Attempts that blew the eval never had a payout chance
        # and must not dilute this number.
        per_attempt_payout_probability=float(sum_attempts_reached_payout / sum_attempts_passed * 100) if sum_attempts_passed else 0.0,
        per_attempt_failure_before_payout_probability=(
            float((sum_attempts_passed - sum_attempts_reached_payout) / sum_attempts_passed * 100) if sum_attempts_passed else 0.0
        ),
        total_independent_attempts=int(sum_total_attempts),
        pass_probability_ci95=pass_ci95,
        payout_probability_ci95=payout_ci95,
        per_attempt_pass_ci95=per_attempt_pass_ci95,
        per_attempt_payout_ci95=per_attempt_payout_ci95,
    )
    return result
