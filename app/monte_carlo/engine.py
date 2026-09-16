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
    method: str = "bootstrap"        # "shuffle" | "bootstrap" | "block_bootstrap"
    block_size: int = 5              # used when method == "block_bootstrap"
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
    reset_on_breach: bool = False


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

    # MC-004: what this run's resampling actually did, in plain language,
    # so `evaluation_pass_probability` isn't read as a stronger claim than
    # it is. Every consumer of this number (Search Lab, Quick Optimizer,
    # Evolution Lab, Full Pipeline's verdict, CPCV) inherits this same
    # caveat -- see run_monte_carlo's docstring for the full reasoning.
    methodology_note: str = ""

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def _max_losing_streak(pnls: np.ndarray) -> int:
    best = cur = 0
    for p in pnls:
        cur = cur + 1 if p <= 0 else 0
        best = max(best, cur)
    return best


def _resample_pnls(rng: np.random.Generator, pnls: np.ndarray, cfg: MonteCarloConfig) -> np.ndarray:
    n = len(pnls)
    if cfg.method == "shuffle":
        return rng.permutation(pnls)
    if cfg.method == "block_bootstrap":
        blocks = max(1, n // cfg.block_size)
        out = []
        while len(out) < n:
            start = rng.integers(0, n)
            block = [pnls[(start + j) % n] for j in range(cfg.block_size)]
            out.extend(block)
        return np.array(out[:n])
    # default: iid bootstrap with replacement
    idx = rng.integers(0, n, size=n)
    return pnls[idx]


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
    in the app actually wants: the probability of reaching the prop
    firm's profit target BEFORE hitting the daily-loss limit, the
    max-drawdown limit, or the consistency rule -- i.e.
    MonteCarloResult.evaluation_pass_probability.

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
    pass their own mc_cfg. Returns 0.0 (not an exception) for an
    empty/too-small trade list, since "this slice produced nothing worth
    passing" is itself a valid, low, fold score rather than a hard
    failure.
    """
    if not trades:
        return 0.0
    cfg = mc_cfg or MonteCarloConfig(n_simulations=500)
    try:
        result = run_monte_carlo(trades, rules, cfg)
    except ValueError:
        return 0.0
    return result.evaluation_pass_probability


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


def _methodology_note(cfg: "MonteCarloConfig", n_trades: int, selection_bias_caveat: bool) -> str:
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

    selection_bias_caveat: True when the caller knows (or can't rule
    out) that these trades came from a search/optimization step that
    already selected them for scoring well -- in that case this Monte
    Carlo run is resampling-order robustness LAYERED ON TOP OF whatever
    selection bias already exists in the input trades, not independent
    out-of-sample evidence on its own.
    """
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
            f"{cfg.block_size}, preserving short local runs of the {n_trades} historical "
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
    if selection_bias_caveat:
        note += (
            " These trades came from an optimization/search step -- treat this result as "
            "resampling-order robustness on top of whatever selection bias already exists in "
            "how these trades were chosen, not as independent validation by itself."
        )
    if cfg.reset_on_breach:
        note += _reset_on_breach_note(cfg)
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

    for _ in range(cfg.n_simulations):
        sim_pnls = _resample_pnls(rng, base_pnls, cfg)
        sim_pnls = _apply_slippage_stress(sim_pnls, cfg.slippage_stress_pct)

        result = simulate_account(
            sim_pnls, base_dates, rules, _day_structure=day_structure,
            reset_on_breach=cfg.reset_on_breach,
        )

        passed_flags.append(result.passed_evaluation)
        first_payout_flags.append(result.reached_first_payout)
        failed_before_payout_flags.append(result.failed and not result.reached_first_payout)
        multiple_payout_flags.append(len(result.payouts) > 1)
        attempts_per_path.append(result.total_attempts)

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

    ruin_arr = dd_arr >= rules.max_drawdown_pct  # account hit its max-drawdown floor at least once
    attempts_arr = np.array(attempts_per_path)

    def pct(arr, q):
        return float(np.percentile(arr, q)) if len(arr) else 0.0

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
        methodology_note=_methodology_note(cfg, len(trades), selection_bias_caveat),
        reset_on_breach=cfg.reset_on_breach,
        mean_attempts_per_path=float(attempts_arr.mean()) if len(attempts_arr) else 0.0,
        median_attempts_per_path=pct(attempts_arr, 50),
        p95_attempts_per_path=pct(attempts_arr, 95),
        any_attempt_pass_probability=float(passed_arr.mean() * 100),
        any_attempt_payout_probability=float(first_payout_arr.mean() * 100),
        attempts_distribution=attempts_arr.tolist(),
    )
    return result
