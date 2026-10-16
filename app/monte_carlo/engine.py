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
from typing import Any, NamedTuple

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
    method: str = "day_block_bootstrap"  # "shuffle" | "bootstrap" | "block_bootstrap" | "day_block_bootstrap" | "regime_shuffle"
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
    # ACCURACY OVERHAUL (2026-10-07): "day_block_bootstrap" (the new default)
    # resamples whole SESSION DAYS -- every trade of a day stays together and
    # keeps its day -- in blocks of `day_block_days` consecutive days. The
    # daily-loss limit, the consistency rule and the 5x$200 winning-day count
    # are all per-day rules; resampling single trades (or arbitrary trade
    # blocks) cuts days in half and changes how often those rules bind.
    day_block_days: int = 3
    # 2026-10 audit: feed each resampled trade's MAE/MFE (dollars) into the account so
    # floating-drawdown / intraday-trailing / floating-daily-loss rules are checked on the
    # open-P&L range, not just closed P&L. The excursions are resampled with the SAME index
    # draw as the P&Ls. Trades without measured excursions use the closed-P&L proxy and the
    # count is disclosed in MonteCarloResult.warnings-style notes (see excursion_note).
    use_excursions: bool = True
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
    # 2026-10 audit: default is now False. With True the headline "pass probability" meant
    # "at least one attempt in a mechanically rebought chain passed", which flatters the
    # per-attempt number. Per-attempt results (what one challenge fee buys) are the default;
    # pass reset_on_breach=True explicitly for chain / bankroll studies.
    reset_on_breach: bool = False
    # 2026-10-15 review (time-bounded ruin): the headline risk_of_ruin_pct now counts
    # only account deaths that happen within this many CALENDAR days of an attempt's
    # start (default 90 ~ a typical evaluation + first-payout window). The old
    # definition counted ANY breach across the whole replayed history, so the same
    # strategy scored a longer backtest as far riskier (18.5% ruin on 80 trades,
    # 59.3% on 157) purely because accounts had more time to bust. 90- and 180-day
    # ruin, ruin before first payout, and funded-account bust are always reported
    # alongside (see MonteCarloResult); lifetime-in-sample ruin stays available as
    # per_attempt_bust_probability.
    ruin_horizon_days: int = 90


# Fixed reporting windows for time-bounded ruin (calendar days).
RUIN_WINDOW_SHORT_DAYS = 90
RUIN_WINDOW_LONG_DAYS = 180


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

    # ACCURACY OVERHAUL (2026-10-07) -- honest attempt-level accounting.
    # bust_before_pass_probability: % of paths whose FIRST purchased account
    #   breached a rule before it passed the evaluation.
    # per_attempt_bust_probability: failed attempts / all attempts (the
    #   definition risk_of_ruin_pct now uses; see risk_of_ruin_definition).
    # expected_attempts_to_pass: purchased accounts per pass (total attempts /
    #   passed attempts across the simulated chains), None when none passed.
    bust_before_pass_probability: float = 0.0
    per_attempt_bust_probability: float = 0.0
    expected_attempts_to_pass: float | None = None
    risk_of_ruin_definition: str = "share of purchased accounts (attempts) that breached a firm rule"
    sample_ok: bool = True
    sample_notes: list = field(default_factory=list)
    n_source_trades: int = 0
    n_source_days: int = 0

    # -- 2026-10-15 review: time-bounded ruin + the pass-AND-paid product ------------------
    # risk_of_ruin_pct (above) is now ruin WITHIN `ruin_horizon_days` calendar days of an
    # attempt's start. The fields below break it out. All are % of purchased attempts unless
    # stated; all default to 0 so results built/deserialized before this change still load.
    ruin_horizon_days: int = 90
    ruin_within_90d_pct: float = 0.0
    ruin_within_180d_pct: float = 0.0
    ruin_before_first_payout_pct: float = 0.0     # attempts that failed having never reached a payout
    lifetime_ruin_pct: float = 0.0                # the OLD definition: any breach in the replayed history
    funded_bust_pct: float = 0.0                  # of FUNDED (passed) attempts, % that later breached -- lifetime in the replayed history
    n_source_calendar_days: int = 0               # calendar span of the source trades
    ruin_horizon_covered: bool = True             # False when the source history is shorter than ruin_horizon_days
    # Pass-AND-paid per purchased attempt = P(pass) x P(paid | passed): what the README
    # question actually asks. per_attempt_payout_probability stays the CONDITIONAL rate
    # (payouts per FUNDED account) and is shown next to it, never instead of it.
    per_attempt_paid_probability: float = 0.0
    per_attempt_paid_ci95: tuple = (0.0, 0.0)

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

    @property
    def headline_pass_and_paid_probability(self) -> float:
        """P(pass the evaluation AND reach a first payout) per purchased attempt --
        the product the README promises. Falls back to the path-level payout rate
        (identical when reset_on_breach is off) for results built before the
        per-attempt paid field existed."""
        if self.total_independent_attempts and (self.per_attempt_paid_probability or self.per_attempt_paid_ci95 != (0.0, 0.0)):
            return self.per_attempt_paid_probability
        return self.first_payout_probability

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def _max_losing_streak(pnls: np.ndarray) -> int:
    # P2-7: breakeven (pnl <= 0) counts as non-winning here -- consistent
    # with app.backtest.statistics' win-rate/max-consecutive-losers, which
    # likewise treat breakeven as a loss for streak purposes.
    # v9.13: vectorized run-length computation, integer-identical to the
    # per-element loop it replaces (a "loss" is pnl <= 0; NaN compares
    # False in both versions, so a NaN breaks a streak either way). This
    # runs once per simulated path -- at 10,000 paths x ~3,800 trades the
    # Python loop was seconds of pure overhead per Monte Carlo call.
    arr = np.asarray(pnls, dtype=float)
    if arr.size == 0:
        return 0
    is_loss = arr <= 0
    if not bool(is_loss.any()):
        return 0
    win_idx = np.flatnonzero(~is_loss)
    bounds = np.concatenate(([-1], win_idx, [arr.size]))
    return int(np.diff(bounds).max() - 1)


# v5 (2026-10-04): minimum trades for any MC-derived verdict. A fold/path
# with fewer trades than this contributes NO verdict (scores 0.0): the
# resample has too few independent trades for the pass probability -- and
# its Wilson CI -- to mean anything, and scoring such a slice at face
# value lets thin-slice noise climb the search objective. ~15 = a handful
# of trading days at the typical 2-5 trades/day, the smallest sample
# whose block bootstrap isn't pure noise.
MIN_TRADES_FOR_VERDICT = 15
MIN_DAYS_FOR_VERDICT = 20


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


def _day_groups(trades) -> tuple[list, list]:
    """Indices of trades per session day (chronological) and the day dates."""
    from app.data.trading_day import trading_day
    order = sorted(range(len(trades)), key=lambda i: pd.Timestamp(trades[i].entry_time))
    groups: list[list[int]] = []
    dates: list = []
    last = None
    for i in order:
        d = pd.Timestamp(trading_day(pd.Timestamp(trades[i].entry_time), tz="America/Chicago", roll_hour=17))
        if d != last:
            groups.append([])
            dates.append(d)
            last = d
        groups[-1].append(i)
    return [np.asarray(g, dtype=np.int64) for g in groups], dates


class _DayBlockTables(NamedTuple):
    """Per-run tables for the numpy day-block resampler: the cumulative offsets are computed ONCE
    (not once per path), so drawing/gathering a path is a handful of vectorized operations."""
    counts: np.ndarray    # trades in each source day group
    starts: np.ndarray    # offset of each group inside `order`
    order: np.ndarray     # all trade indices, groups concatenated in day order


def _day_block_tables(groups) -> _DayBlockTables:
    counts = np.fromiter((len(g) for g in groups), dtype=np.int64, count=len(groups))
    starts = np.cumsum(counts) - counts
    order = np.concatenate(groups).astype(np.int64) if len(groups) else np.zeros(0, dtype=np.int64)
    return _DayBlockTables(counts, starts, order)


def _draw_day_block_choices(rng, nd: int, block_days: int, n_paths: int) -> np.ndarray:
    """Group ids for ``n_paths`` paths at once: (n_paths, nd). ONE ``rng.integers`` call draws every
    block start for every path, path-major -- the identical stream the old per-path scalar draws
    consumed (ceil(nd/k) draws per path, in path order), so every result is unchanged."""
    k = max(1, min(int(block_days), nd))
    nb = -(-nd // k)
    starts = rng.integers(0, nd - k + 1, size=(n_paths, nb))
    return (starts[:, :, None] + np.arange(k)[None, None, :]).reshape(n_paths, nb * k)[:, :nd]


def _resample_day_blocks(rng, groups, dates, block_days: int, tables: "_DayBlockTables | None" = None):
    """Draws consecutive-day blocks (non-circular) until the path has as many
    day slots as the source. Returns (trade_index_array, DayStructure) -- the
    structure is built with numpy straight from the drawn slots, so no
    per-trade timestamp parsing happens inside the simulation loop.

    2026-10-15: numpy rewrite. All block starts for the path come from ONE
    ``rng.integers`` call and the trade gather uses precomputed cumulative
    offsets (``tables``, built once per run by the caller; computed here when
    omitted so existing callers keep working). Draw stream and output are
    identical to the previous per-block Python loop."""
    from app.prop.simulator import DayStructure
    nd = len(groups)
    tb = tables if tables is not None else _day_block_tables(groups)
    chosen = _draw_day_block_choices(rng, nd, block_days, 1)[0]
    cnt = tb.counts[chosen]
    total = int(cnt.sum())
    blk_end = np.cumsum(cnt)
    pos = np.arange(total) - np.repeat(blk_end - cnt, cnt)
    idx = tb.order[np.repeat(tb.starts[chosen], cnt) + pos]
    day_index = np.repeat(np.arange(nd), cnt)
    is_last = np.zeros(total, dtype=bool)
    is_last[blk_end - 1] = True
    ds = DayStructure(day_index_per_trade=day_index.tolist(), is_last_of_day=is_last.tolist(),
                      day_dates=list(dates), n_days=nd)
    return idx, ds


def _month_groups(dates) -> list:
    """Groups of consecutive day-group indices that share a calendar (year, month)."""
    out: list[list[int]] = []
    last = None
    for i, d in enumerate(dates):
        key = (pd.Timestamp(d).year, pd.Timestamp(d).month)
        if key != last:
            out.append([])
            last = key
        out[-1].append(i)
    return [np.asarray(g, dtype=np.int64) for g in out]


def _resample_month_blocks(rng, groups, dates, month_groups):
    """Regime-shuffle path (2026-10 audit): resample WHOLE CALENDAR MONTHS with replacement,
    keeping each month's days (and every day's trades) in their original order, until the
    path has as many day slots as the source. A strategy that only works in one regime (one
    strong month carrying the sample) shows up as a wide spread between paths, which
    day-level blocks of 3 days smooth away. Returns (trade_index_array, DayStructure)."""
    from app.prop.simulator import DayStructure
    nd = len(groups)
    nm = len(month_groups)
    chosen: list[int] = []
    while len(chosen) < nd:
        chosen.extend(month_groups[int(rng.integers(0, nm))].tolist())
    chosen = chosen[:nd]
    arrs = [groups[g] for g in chosen]
    idx = np.concatenate(arrs).astype(np.int64)
    counts = np.fromiter((len(a) for a in arrs), dtype=np.int64, count=nd)
    day_index = np.repeat(np.arange(nd), counts)
    is_last = np.zeros(len(idx), dtype=bool)
    is_last[np.cumsum(counts) - 1] = True
    ds = DayStructure(day_index_per_trade=day_index.tolist(), is_last_of_day=is_last.tolist(),
                      day_dates=list(dates), n_days=nd)
    return idx, ds


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
    excursion_set=None,
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
        "regime_shuffle": (
            f"resampled {cfg.n_simulations:,} times by drawing WHOLE CALENDAR MONTHS of the "
            f"{n_trades} historical trades with replacement (each month keeps its days and trades in "
            "order), so a strategy that only worked in one or two regimes shows a wide, honest spread"
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
    # P1-3 / 2026-10 audit: say exactly what drawdown evidence the paths carry.
    if excursion_set is not None:
        n_all = excursion_set.n_measured + excursion_set.n_proxied
        note += (
            f" Drawdown modeling: each resampled trade carried its own MAE/MFE (dollars) into the "
            f"account, so floating / intraday-trailing rules were checked on the open-P&L range, not "
            f"just closed P&L. {excursion_set.n_measured} of {n_all} trades "
            f"({excursion_set.measured_fraction * 100:.0f}%) had measured excursions; the other "
            f"{excursion_set.n_proxied} use the closed-P&L proxy (min(pnl,0) / max(pnl,0)), which "
            "understates intratrade drawdown for those trades."
        )
    else:
        note += (
            " Drawdown-modeling caveat: this run was made with use_excursions=False, so the account "
            "simulator trails drawdown on REALIZED balance only (trade-close equity). For firms that "
            "enforce trailing drawdown on real-time floating equity this UNDERSTATES how often the "
            "account would actually fail. Re-run with use_excursions=True (default) or "
            "PropRules(floating_drawdown_mode=\"adverse\")."
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
        "breach, max-drawdown breach, or inactivity closure) that happens within the "
        f"first {int(cfg.ruin_horizon_days)} calendar days of an attempt (time-bounded, so the "
        "length of the backtest does not change it; 90/180-day ruin, ruin before first payout "
        "and funded-account bust are reported separately). Wilson 95% CIs on the PER-ATTEMPT "
        "pass/payout probabilities (not the chain-level intervals) are "
        "reported alongside the point estimates; the acceptance verdict "
        "gates on their lower bounds."
    )
    return note


class _SimRow(NamedTuple):
    """Everything run_monte_carlo's aggregation consumes from ONE
    simulated path, in the aggregation's own terms. Computed by
    _mc_simulate_path either in-process (serial path) or in a worker
    process (parallel path) -- a pure function of the drawn path, so the
    two paths produce interchangeable rows."""
    passed: bool
    reached_first_payout: bool
    failed_before_payout: bool
    multiple_payouts: bool
    total_attempts: int
    attempts_passed: int
    attempts_reached_payout: int
    any_attempt_failed: bool
    first_attempt_bust: bool
    attempts_failed: int
    days_to_pass: Any
    first_payout_day_index: Any
    return_pct: float
    total_payout_amount: float
    max_drawdown_pct: float
    max_losing_streak: int
    # 2026-10-15 review: time-bounded ruin inputs (defaults keep any older constructor working).
    # failed_calendar_days = how many CALENDAR days each failed attempt lived (0 = unknown, which
    # the aggregator counts as "within any window"); carrying the raw days lets run_monte_carlo
    # apply any ruin horizon in the parent process, so no horizon has to be plumbed through the
    # worker-pool / shared-pool paths.
    failed_calendar_days: tuple = ()
    failed_before_payout_attempts: int = 0
    funded_failed: int = 0


def _mc_simulate_path(sim_pnls, base_dates, rules, day_structure, reset_on_breach, sim_risks) -> _SimRow:
    """Runs ONE resampled path through simulate_account and reduces it to
    a _SimRow. Extracted verbatim from run_monte_carlo's old serial loop
    (v9.13) so the serial and parallel paths share one implementation.

    2026-10 audit: ``sim_risks`` may be a 3-tuple ``(risks_or_None, mae, mfe)`` (the
    excursion bundle), so the worker/pool plumbing that carries ``sim_risks`` did not need
    a new field. A bare array / None keeps the legacy meaning."""
    sim_mae = sim_mfe = None
    if isinstance(sim_risks, tuple) and len(sim_risks) == 3:
        sim_risks, sim_mae, sim_mfe = sim_risks
    result = simulate_account(
        sim_pnls, base_dates, rules,
        _day_structure=day_structure,
        reset_on_breach=reset_on_breach,
        trade_initial_risks=sim_risks,
        trade_mae=sim_mae,
        trade_mfe=sim_mfe,
    )
    first_attempt = result.attempts[0] if result.attempts else None
    return _SimRow(
        passed=result.passed_evaluation,
        reached_first_payout=result.reached_first_payout,
        failed_before_payout=result.failed and not result.reached_first_payout,
        multiple_payouts=len(result.payouts) > 1,
        total_attempts=result.total_attempts,
        attempts_passed=result.attempts_passed,
        attempts_reached_payout=result.attempts_reached_payout,
        any_attempt_failed=any(a.failed for a in result.attempts),
        first_attempt_bust=bool(first_attempt and first_attempt.failed and not first_attempt.passed_evaluation),
        attempts_failed=sum(1 for a in result.attempts if a.failed),
        days_to_pass=result.days_to_pass,
        first_payout_day_index=result.first_payout_day_index,
        return_pct=(result.final_balance - rules.account_size) / rules.account_size * 100.0,
        total_payout_amount=result.total_payout_amount,
        max_drawdown_pct=result.max_drawdown_pct_reached,
        max_losing_streak=_max_losing_streak(sim_pnls),
        failed_calendar_days=tuple(int(a.calendar_days_used or 0) for a in result.attempts if a.failed),
        failed_before_payout_attempts=sum(1 for a in result.attempts if a.failed and not a.reached_first_payout),
        funded_failed=sum(1 for a in result.attempts if a.failed and a.passed_evaluation),
    )


# -- 2026-10-15: vectorized path evaluation ------------------------------------------------------
# One numpy state array per rule value, looping over the TRADE INDEX instead of over simulations
# (app.prop.vector_account). Same draws (batched, stream-identical), same per-path arithmetic, so the
# rows are identical to _mc_simulate_path's; rule sets the kernel does not cover fall back to the
# original per-path loop (see supports_vectorized). T58_MC_VECTORIZE=0 turns it off.
_VEC_MIN_SIMS = 48
_VEC_CELL_BUDGET_DAYBLOCK = 20_000_000     # (trades x paths) cells per chunk; ~9 bytes each
_VEC_CELL_BUDGET_GENERIC = 4_000_000       # generic path stacks float matrices -> smaller chunks


def _vec_rows_from_result(res, rules, tables) -> list:
    acct = float(rules.account_size)
    passed = res.passed.tolist()
    failed = res.failed.tolist()
    dtp = res.days_to_pass.tolist()
    fpd = res.first_payout_day.tolist()
    npay = res.n_payouts.tolist()
    tpay = res.total_payout.tolist()
    ret = ((res.final_balance - acct) / acct * 100.0).tolist()
    mdd = res.max_dd_pct.tolist()
    streak = res.max_losing_streak.tolist()
    cal_used = (tables.cal0[np.clip(res.last_slot, 0, len(tables.cal0) - 1)] + 1).tolist()
    rows = []
    for k in range(len(passed)):
        p, f = passed[k], failed[k]
        reached = fpd[k] >= 0
        rows.append(_SimRow(
            passed=p, reached_first_payout=reached,
            failed_before_payout=f and not reached,
            multiple_payouts=npay[k] > 1,
            total_attempts=1, attempts_passed=1 if p else 0, attempts_reached_payout=1 if reached else 0,
            any_attempt_failed=f, first_attempt_bust=f and not p,
            attempts_failed=1 if f else 0,
            days_to_pass=dtp[k] if dtp[k] >= 0 else None,
            first_payout_day_index=fpd[k] if reached else None,
            return_pct=ret[k], total_payout_amount=tpay[k], max_drawdown_pct=mdd[k],
            max_losing_streak=streak[k],
            failed_calendar_days=(int(cal_used[k] or 0),) if f else (),
            failed_before_payout_attempts=1 if (f and not reached) else 0,
            funded_failed=1 if (f and p) else 0,
        ))
    return rows


def _mc_rows_vectorized(*, n_sims, rng, cfg, rules, base_pnls, base_risks, has_risks, base_mae, base_mfe,
                        use_day_blocks, use_regime_shuffle, groups, gdates, tables, draw_one,
                        progress_cb, base_ds=None) -> list:
    from app.prop.vector_account import build_day_tables, simulate_paths
    stress = cfg.slippage_stress_pct
    rows: list = []
    done = 0

    def _report():
        if progress_cb is not None:
            try:
                progress_cb(done, n_sims)
            except Exception:  # noqa: BLE001 -- progress must never sink the run
                pass

    fast = use_day_blocks and not use_regime_shuffle and groups is not None
    if fast:
        nd = len(groups)
        day_tables = build_day_tables(gdates)
        avg_len = max(1, int(len(tables.order)))
        chunk = int(max(256, min(8192, _VEC_CELL_BUDGET_DAYBLOCK // max(avg_len, 1))))
        while done < n_sims:
            P = min(chunk, n_sims - done)
            chosen = _draw_day_block_choices(rng, nd, cfg.day_block_days, P)
            cnt = tables.counts[chosen]                       # (P, nd)
            path_len = cnt.sum(axis=1)
            L = int(path_len.max())
            fc = cnt.ravel()
            fg = chosen.ravel()
            total = int(fc.sum())
            blk_end = np.cumsum(fc)
            pos = np.arange(total) - np.repeat(blk_end - fc, fc)
            trade_idx = tables.order[np.repeat(tables.starts[fg], fc) + pos]
            row_of = np.repeat(np.arange(P), path_len)
            col_of = np.arange(total) - np.repeat(np.cumsum(path_len) - path_len, path_len)
            idx2 = np.zeros((L, P), dtype=np.int32)
            slot2 = np.full((L, P), -1, dtype=np.int32)
            last2 = np.zeros((L, P), dtype=bool)
            idx2[col_of, row_of] = trade_idx
            slot2[col_of, row_of] = np.repeat(np.tile(np.arange(nd), P), fc)
            last_flat = np.zeros(total, dtype=bool)
            last_flat[blk_end - 1] = True
            last2[col_of, row_of] = last_flat

            def get_step(t, idx2=idx2, slot2=slot2, last2=last2):
                ix = idx2[t]
                pn = _apply_slippage_stress(base_pnls[ix], stress)
                mae = mfe = None
                if base_mae is not None:
                    d = pn - base_pnls[ix]
                    mae = np.minimum(base_mae[ix] + np.minimum(d, 0.0), np.minimum(pn, 0.0))
                    mfe = np.maximum(base_mfe[ix] + np.maximum(d, 0.0), np.maximum(pn, 0.0))
                risk = base_risks[ix] if has_risks else None
                return pn, mae, mfe, risk, slot2[t], last2[t]

            res = simulate_paths(rules, path_len, get_step, L, day_tables)
            rows.extend(_vec_rows_from_result(res, rules, day_tables))
            done += P
            _report()
        return rows

    # generic path (iid / block / regime-shuffle methods): draws stay per path, evaluation is batched
    first = draw_one()
    ds0 = first[1] if first[1] is not None else base_ds
    if ds0 is None:
        raise ValueError("generic vector path needs a DayStructure")
    L0 = len(first[0])
    chunk = int(max(64, min(4096, _VEC_CELL_BUDGET_GENERIC // max(L0, 1))))
    day_tables = build_day_tables(ds0.day_dates)
    pending = [first]
    while done < n_sims:
        P = min(chunk, n_sims - done)
        draws = pending + [draw_one() for _ in range(P - len(pending))]
        pending = []
        lens = np.array([len(d[0]) for d in draws], dtype=np.int64)
        L = int(lens.max())
        pnl2 = np.zeros((L, P)); slot2 = np.full((L, P), -1, dtype=np.int64); last2 = np.zeros((L, P), dtype=bool)
        mae2 = mfe2 = risk2 = None
        has_exc = isinstance(draws[0][2], tuple) and len(draws[0][2]) == 3
        if has_exc:
            mae2 = np.zeros((L, P)); mfe2 = np.zeros((L, P))
            if draws[0][2][0] is not None:
                risk2 = np.zeros((L, P))
        elif draws[0][2] is not None:
            risk2 = np.zeros((L, P))
        for c, (pn, ds, rk) in enumerate(draws):
            ds = ds if ds is not None else base_ds
            n = len(pn)
            sl = np.asarray(ds.day_index_per_trade, dtype=np.int64)
            if n == 0 or sl[0] != 0 or (np.diff(sl) > 1).any() or (np.diff(sl) < 0).any() or len(ds.day_dates) != len(day_tables.cal0):
                raise ValueError("day structure not contiguous")
            pnl2[:n, c] = pn
            slot2[:n, c] = sl
            last2[:n, c] = np.asarray(ds.is_last_of_day, dtype=bool)
            if has_exc:
                r_, ma_, mf_ = rk
                mae2[:n, c] = ma_; mfe2[:n, c] = mf_
                if risk2 is not None:
                    risk2[:n, c] = r_
            elif rk is not None:
                risk2[:n, c] = rk

        def get_step(t, pnl2=pnl2, slot2=slot2, last2=last2, mae2=mae2, mfe2=mfe2, risk2=risk2):
            return (pnl2[t], None if mae2 is None else mae2[t], None if mfe2 is None else mfe2[t],
                    None if risk2 is None else risk2[t], slot2[t], last2[t])

        res = simulate_paths(rules, lens, get_step, L, day_tables)
        rows.extend(_vec_rows_from_result(res, rules, day_tables))
        done += P
        _report()
    return rows


# -- v9.13: cross-process path evaluation --------------------------------
# The resample DRAWS are the only part of a Monte Carlo run that must
# stay sequential (they consume one rng stream); given a drawn path,
# simulate_account is a pure function. So the parallel path draws every
# path in THIS process, in the original order, and farms only the
# simulation work out -- results are collected by sim index and fed to
# the exact same aggregation, in the exact same order, as the serial
# loop. Identical draws + identical per-path math + identical
# aggregation order = a bit-identical MonteCarloResult; this is pinned
# by tests/search/test_v913_progress_explorer_speed.py. Spawn (not fork) to
# match every other pool in this app, and any pool failure falls back
# to the serial loop with a freshly seeded rng (see run_monte_carlo).
_MC_PARALLEL_MIN_SIMS = 200

_MC_WORKER: dict = {}


def _mc_worker_init(base_dates, rules, day_structure, reset_on_breach) -> None:
    global _MC_WORKER
    _MC_WORKER = {
        "base_dates": base_dates, "rules": rules,
        "day_structure": day_structure, "reset_on_breach": reset_on_breach,
    }


def _mc_worker_task(payload: list) -> list:
    w = _MC_WORKER
    out = []
    for sim_pnls, sim_ds, sim_risks in payload:
        out.append(_mc_simulate_path(
            sim_pnls, w["base_dates"], w["rules"],
            sim_ds if sim_ds is not None else w["day_structure"],
            w["reset_on_breach"], sim_risks,
        ))
    return out


# v9.15 (WS-F): stateless chunk task for a CALLER-OWNED shared pool. The
# per-call pool above binds (base_dates, rules, day_structure) in the
# worker initializer, which forces a fresh process spawn for every
# run_monte_carlo call -- Full Pipeline alone calls it 5x per run
# (baseline, final, 3x seed-rescore), and on Windows each spawn
# re-imports the whole app in every worker. This task instead carries
# the (small) shared state per chunk, so ONE pool can serve every MC
# call in a pipeline run. The per-path math is _mc_simulate_path itself,
# draws still happen in the parent in sim order, and rows are collected
# by sim index -- bit-identical to the serial and per-call-pool paths.
def _mc_chunk_task(task):
    base_dates, rules, fallback_ds, reset_on_breach, items = task
    return [
        _mc_simulate_path(
            sim_pnls, base_dates, rules,
            sim_ds if sim_ds is not None else fallback_ds,
            reset_on_breach, sim_risks,
        )
        for sim_pnls, sim_ds, sim_risks in items
    ]


def _mc_rows_on_pool(pool, n_sims, draw_one, *, base_dates, rules, day_structure,
                     reset_on_breach, progress_cb, chunk_cap=256):
    """_mc_rows_parallel's contract (draw in sim order in THIS process,
    indexed collection, identical aggregation) against a caller-owned
    ProcessPoolExecutor instead of a fresh one. The pool is never shut
    down here -- its lifecycle belongs to the caller. Raises on any pool
    failure; the caller falls back exactly as _mc_rows_parallel's does."""
    from concurrent.futures import FIRST_COMPLETED
    from concurrent.futures import wait as _futures_wait

    workers = max(1, int(getattr(pool, "_max_workers", 2) or 2))
    rows: list = [None] * n_sims
    chunk = max(16, min(chunk_cap, n_sims // max(1, workers * 16)))
    done = 0
    next_i = 0
    pending: dict = {}

    def _submit_more() -> None:
        nonlocal next_i
        while next_i < n_sims and len(pending) < workers * 2:
            start = next_i
            payload = [draw_one() for _ in range(start, min(start + chunk, n_sims))]
            next_i = start + len(payload)
            task = (base_dates, rules, day_structure, reset_on_breach, payload)
            pending[pool.submit(_mc_chunk_task, task)] = start

    _submit_more()
    while pending:
        finished, _ = _futures_wait(list(pending), return_when=FIRST_COMPLETED)
        for fut in finished:
            start = pending.pop(fut)
            for off, row in enumerate(fut.result()):
                rows[start + off] = row
                done += 1
            if progress_cb is not None:
                try:
                    progress_cb(done, n_sims)
                except Exception:  # noqa: BLE001 -- progress must never sink the run
                    pass
        _submit_more()
    return rows


def _mc_rows_parallel(n_sims, draw_one, *, base_dates, rules, day_structure,
                      reset_on_breach, workers, progress_cb, chunk_cap=256):
    """Draws paths via draw_one() (in sim order, this process) and
    evaluates them across `workers` processes, returning _SimRows in sim
    order. Raises on any pool failure -- the caller falls back to the
    serial loop with a re-seeded rng, so a broken pool can never change
    the numbers, only the speed."""
    import multiprocessing
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
    from concurrent.futures import wait as _futures_wait

    rows: list = [None] * n_sims
    chunk = max(16, min(chunk_cap, n_sims // max(1, workers * 16)))
    done = 0
    next_i = 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx,
        initializer=_mc_worker_init,
        initargs=(base_dates, rules, day_structure, reset_on_breach),
    ) as pool:
        pending: dict = {}

        def _submit_more() -> None:
            nonlocal next_i
            while next_i < n_sims and len(pending) < workers * 2:
                start = next_i
                payload = [draw_one() for _ in range(start, min(start + chunk, n_sims))]
                next_i = start + len(payload)
                pending[pool.submit(_mc_worker_task, payload)] = start

        _submit_more()
        while pending:
            finished, _ = _futures_wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                start = pending.pop(fut)
                for off, row in enumerate(fut.result()):
                    rows[start + off] = row
                    done += 1
                if progress_cb is not None:
                    try:
                        progress_cb(done, n_sims)
                    except Exception:  # noqa: BLE001 -- progress must never sink the run
                        pass
            _submit_more()
    return rows


def run_monte_carlo(
    trades: list[Trade],
    rules: PropRules,
    cfg: MonteCarloConfig | None = None,
    selection_bias_caveat: bool = False,
    max_workers: int | None = None,
    progress_cb=None,
    pool=None,
) -> MonteCarloResult:
    """
    selection_bias_caveat: MC-004. Pass True when `trades` are known (or
    can't be ruled out) to have come from a search/optimization step that
    already selected them for scoring well on some metric -- e.g. a
    genome the walk-forward-aware GA just picked. Only changes the
    wording of the returned result's `methodology_note`; never changes
    any numeric output.

    max_workers (v9.13): when > 1 and n_simulations is large enough,
    the per-path simulations run across that many worker processes.
    Purely a speed change -- the resample draws stay in this process in
    the original order and results are aggregated in sim order, so the
    returned MonteCarloResult is bit-identical to the serial run (see
    _mc_rows_parallel). None (the default) keeps every existing caller
    on the exact serial path it has always used.

    pool (v9.15, WS-F): an optional caller-owned ProcessPoolExecutor.
    When given (and n_simulations clears the same floor), paths are
    evaluated on THAT pool via the stateless _mc_chunk_task instead of
    spawning a fresh pool per call -- one spawn can then serve every
    Monte Carlo call in a pipeline run. Same bit-identical contract;
    the pool is never shut down here (the caller owns its lifecycle),
    and any failure falls back to the max_workers/serial paths with a
    re-seeded rng, so a broken shared pool can never change the numbers.

    progress_cb: optional callable(done, total), invoked as paths
    complete (at most ~20 times). Purely observational; exceptions in
    it are swallowed.
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
    # 2026-10 audit: per-trade MAE/MFE (dollars), shifted by the same session-slippage delta
    # as the P&Ls so the excursions bracket the adjusted outcome, resampled with the SAME draw.
    base_mae = base_mfe = None
    excursion_set = None
    if cfg.use_excursions:
        from app.prop.excursions import trade_excursions
        excursion_set = trade_excursions(trades)
        _raw = np.array([float(getattr(t, "pnl", 0.0) or 0.0) for t in trades], dtype=float)
        _delta = np.asarray(base_pnls, dtype=float) - _raw
        base_mae = np.minimum.reduce([np.array(excursion_set.mae) + _delta, np.asarray(base_pnls, dtype=float), np.zeros(len(trades))])
        base_mfe = np.maximum.reduce([np.array(excursion_set.mfe) + _delta, np.asarray(base_pnls, dtype=float), np.zeros(len(trades))])
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
    use_regime_shuffle = cfg.method == "regime_shuffle"
    use_day_blocks = cfg.method == "day_block_bootstrap" or use_regime_shuffle
    _month_grp = None
    if use_day_blocks:
        _groups, _gdates = _day_groups(trades)
        _tables = _day_block_tables(_groups)
        n_source_days = len(_groups)
        if use_regime_shuffle:
            _month_grp = _month_groups(_gdates)
    else:
        n_source_days = n_trading_days

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
    first_bust_flags: list[bool] = []
    sum_attempts_failed = 0
    failed_days_all: list[int] = []          # calendar days lived by EVERY failed attempt (0 = unknown)
    sum_failed_before_payout = 0
    sum_funded_failed = 0

    # v9.13: the draw for each path happens HERE, in sim order, whether
    # the path itself is then simulated in this process or a worker --
    # that is what keeps the parallel path bit-identical (see
    # _mc_rows_parallel). _draw_one() is exactly the draw sequence the
    # old serial loop performed per iteration.
    def _draw_one():
        if use_regime_shuffle:
            sim_idx, sim_ds = _resample_month_blocks(rng, _groups, _gdates, _month_grp)
        elif use_day_blocks:
            sim_idx, sim_ds = _resample_day_blocks(rng, _groups, _gdates, cfg.day_block_days, _tables)
        else:
            sim_idx = _resample_indices(rng, len(base_pnls), cfg.method, eff_block_size)
            sim_ds = None
        sim_pnls = base_pnls[sim_idx]
        sim_pnls = _apply_slippage_stress(sim_pnls, cfg.slippage_stress_pct)
        sim_risks = base_risks[sim_idx] if has_risks else None
        if base_mae is not None:
            # slippage stress changes the path P&L; keep the excursions bracketing it
            _d = sim_pnls - base_pnls[sim_idx]
            sim_risks = (
                sim_risks,
                np.minimum(base_mae[sim_idx] + np.minimum(_d, 0.0), np.minimum(sim_pnls, 0.0)),
                np.maximum(base_mfe[sim_idx] + np.maximum(_d, 0.0), np.maximum(sim_pnls, 0.0)),
            )
        return sim_pnls, sim_ds, sim_risks

    n_sims = int(cfg.n_simulations)
    rows: list | None = None
    # 2026-10-15: vectorized-across-paths evaluation (see _mc_rows_vectorized). Tried first; anything the
    # kernel does not cover, or any failure, falls through to the original per-path paths below with a
    # re-seeded rng, so it can change the speed but never the numbers.
    from app.prop.vector_account import supports_vectorized, vectorize_enabled
    if (n_sims >= _VEC_MIN_SIMS and vectorize_enabled()
            and supports_vectorized(rules, cfg.reset_on_breach, have_excursions=base_mae is not None)):
        try:
            rows = _mc_rows_vectorized(
                n_sims=n_sims, rng=rng, cfg=cfg, rules=rules, base_pnls=base_pnls, base_risks=base_risks,
                has_risks=has_risks, base_mae=base_mae, base_mfe=base_mfe,
                use_day_blocks=use_day_blocks, use_regime_shuffle=use_regime_shuffle,
                groups=_groups if use_day_blocks else None, gdates=_gdates if use_day_blocks else None,
                tables=_tables if use_day_blocks else None, draw_one=_draw_one, progress_cb=progress_cb,
                base_ds=day_structure,
            )
        except Exception:  # noqa: BLE001 -- never let the fast path change the numbers
            rows = None
            rng = np.random.default_rng(cfg.random_seed)
    # v9.15 (WS-F): a caller-owned shared pool (pool=...) is tried first:
    # one spawn serves every MC call in a pipeline run instead of a
    # spawn/teardown per call. Draws stay in this process in sim order
    # either way, so all three paths (shared pool, per-call pool, serial)
    # produce bit-identical results; any pool failure re-seeds the rng
    # before the next attempt, exactly as the per-call path always has.
    _attempted_parallel = False
    if rows is None and pool is None and max_workers is not None and int(max_workers) > 1 and n_sims >= _MC_PARALLEL_MIN_SIMS:
        # 2026-10-15: no caller-owned pool -> use the process-wide persistent one instead of spawning a
        # fresh pool for this call (falls back to the per-call pool below if it cannot be created).
        from app.monte_carlo.shared_pool import get_shared_mc_pool
        pool = get_shared_mc_pool(int(max_workers))
    if rows is None and pool is not None and n_sims >= _MC_PARALLEL_MIN_SIMS:
        _attempted_parallel = True
        try:
            rows = _mc_rows_on_pool(
                pool, n_sims, _draw_one, base_dates=base_dates, rules=rules,
                day_structure=day_structure, reset_on_breach=cfg.reset_on_breach,
                progress_cb=progress_cb,
                chunk_cap=32 if use_day_blocks else 256,
            )
        except Exception:  # noqa: BLE001 -- a broken pool must never change the numbers
            rows = None
            rng = np.random.default_rng(cfg.random_seed)
    _want_parallel = (
        max_workers is not None and int(max_workers) > 1 and n_sims >= _MC_PARALLEL_MIN_SIMS
    )
    if rows is None and _want_parallel:
        _attempted_parallel = True
        try:
            rows = _mc_rows_parallel(
                n_sims, _draw_one, base_dates=base_dates, rules=rules,
                day_structure=day_structure, reset_on_breach=cfg.reset_on_breach,
                workers=int(max_workers), progress_cb=progress_cb,
                chunk_cap=32 if use_day_blocks else 256,
            )
        except Exception:  # noqa: BLE001 -- a broken pool must never change the numbers
            rows = None
    if rows is None:
        if _attempted_parallel:
            # The failed parallel attempt already consumed draws from
            # rng; re-seed so the serial loop below draws the identical
            # sequence a purely serial run would have drawn.
            rng = np.random.default_rng(cfg.random_seed)
        rows = []
        _report_every = max(1, n_sims // 20) if progress_cb is not None else 0
        for i in range(n_sims):
            sim_pnls, sim_ds, sim_risks = _draw_one()
            rows.append(_mc_simulate_path(
                sim_pnls, base_dates, rules,
                sim_ds if sim_ds is not None else day_structure,
                cfg.reset_on_breach, sim_risks,
            ))
            if progress_cb is not None and ((i + 1) % _report_every == 0 or i + 1 == n_sims):
                try:
                    progress_cb(i + 1, n_sims)
                except Exception:  # noqa: BLE001 -- progress must never sink the run
                    pass

    for row in rows:
        passed_flags.append(row.passed)
        first_payout_flags.append(row.reached_first_payout)
        failed_before_payout_flags.append(row.failed_before_payout)
        multiple_payout_flags.append(row.multiple_payouts)
        attempts_per_path.append(row.total_attempts)
        sum_attempts_passed += row.attempts_passed
        sum_attempts_reached_payout += row.attempts_reached_payout
        sum_total_attempts += row.total_attempts
        # v5: count EVERY account death, whatever the failure_reason
        # ("daily_loss_limit", "max_drawdown (...)", inactivity closure).
        # result.attempts always holds at least the attempt-#1 record, and
        # more than one record only when reset_on_breach rebuys into the
        # remaining history after a bust.
        death_flags.append(row.any_attempt_failed)
        first_bust_flags.append(row.first_attempt_bust)
        sum_attempts_failed += row.attempts_failed
        failed_days_all.extend(row.failed_calendar_days)
        sum_failed_before_payout += row.failed_before_payout_attempts
        sum_funded_failed += row.funded_failed

        if row.days_to_pass is not None:
            days_to_pass_list.append(row.days_to_pass)
        if row.first_payout_day_index is not None:
            days_to_first_payout_list.append(row.first_payout_day_index)

        return_pcts.append(row.return_pct)
        payout_amounts.append(row.total_payout_amount)
        total_withdrawals += row.total_payout_amount
        drawdown_pcts.append(row.max_drawdown_pct)
        losing_streaks.append(row.max_losing_streak)

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
    # ACCURACY OVERHAUL: ruin is now per purchased account (failed attempts
    # / attempts), not "any attempt in a rebuy chain died" -- the chain
    # definition approaches 100% for any strategy that can bust once and
    # says nothing about one account's odds.
    per_attempt_bust = (sum_attempts_failed / sum_total_attempts) if sum_total_attempts else 0.0
    attempts_arr = np.array(attempts_per_path)

    # 2026-10-15 review: TIME-BOUNDED ruin. Share of purchased attempts that died within H
    # calendar days of their own start. An attempt whose length is unknown (0) counts as inside
    # the window -- a bookkeeping gap must never make ruin look smaller. Because only the first
    # H days of each attempt can contribute, a longer source history can no longer raise ruin
    # just by giving accounts more time to bust after funding.
    _failed_days = np.asarray(failed_days_all, dtype=float)

    def _ruin_within(days: int) -> float:
        if not sum_total_attempts:
            return 0.0
        return float(np.count_nonzero(_failed_days <= float(days)) / sum_total_attempts)

    _horizon = max(1, int(cfg.ruin_horizon_days))
    ruin_horizon_frac = _ruin_within(_horizon)
    ruin_90_frac = _ruin_within(RUIN_WINDOW_SHORT_DAYS)
    ruin_180_frac = _ruin_within(RUIN_WINDOW_LONG_DAYS)
    ruin_arr = np.array([ruin_horizon_frac])
    _cal_span = (max(base_dates) - min(base_dates)).days + 1 if base_dates else 0
    _horizon_covered = _cal_span >= _horizon

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

    _notes: list[str] = []
    if len(trades) < MIN_TRADES_FOR_VERDICT:
        _notes.append(f"only {len(trades)} source trades (< {MIN_TRADES_FOR_VERDICT}): resampled probabilities are not meaningful.")
    if n_source_days < MIN_DAYS_FOR_VERDICT:
        _notes.append(f"only {n_source_days} source trading days (< {MIN_DAYS_FOR_VERDICT}): day-level rules (daily loss, consistency) are barely exercised.")
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
        risk_of_ruin_definition=(
            f"share of purchased accounts (attempts) that breached a firm rule within {_horizon} calendar days of starting"
        ),
        median_max_losing_streak=float(np.median(streak_arr)) if len(streak_arr) else 0.0,
        worst_max_losing_streak=int(streak_arr.max()) if len(streak_arr) else 0,
        return_percentiles={q: pct(return_arr, q) for q in (5, 25, 50, 75, 95)},
        drawdown_percentiles={q: pct(dd_arr, q) for q in (5, 25, 50, 75, 95)},
        days_to_payout_distribution=days_to_first_payout_list,
        return_distribution=return_arr.tolist(),
        drawdown_distribution=dd_arr.tolist(),
        methodology_note=_methodology_note(cfg, len(trades), selection_bias_caveat, n_trading_days=n_trading_days, excursion_set=excursion_set)
        + (
            ""
            if _horizon_covered
            else (
                f" RUIN WINDOW NOT COVERED: the source history spans only {_cal_span} calendar day(s), shorter than the "
                f"{_horizon}-day ruin window, so the reported risk of ruin is UNDERSTATED (accounts were never exposed for the "
                "full window); a READY verdict is capped at MARGINAL until the history covers the window."
            )
        ),
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
        bust_before_pass_probability=float(np.mean(first_bust_flags) * 100) if first_bust_flags else 0.0,
        per_attempt_bust_probability=float(per_attempt_bust * 100),
        expected_attempts_to_pass=(float(sum_total_attempts / sum_attempts_passed) if sum_attempts_passed else None),
        sample_ok=not _notes,
        sample_notes=_notes,
        n_source_trades=len(trades),
        n_source_days=int(n_source_days),
        ruin_horizon_days=_horizon,
        ruin_within_90d_pct=float(ruin_90_frac * 100),
        ruin_within_180d_pct=float(ruin_180_frac * 100),
        ruin_before_first_payout_pct=float(sum_failed_before_payout / sum_total_attempts * 100) if sum_total_attempts else 0.0,
        lifetime_ruin_pct=float(per_attempt_bust * 100),
        funded_bust_pct=float(sum_funded_failed / sum_attempts_passed * 100) if sum_attempts_passed else 0.0,
        n_source_calendar_days=int(_cal_span),
        ruin_horizon_covered=bool(_horizon_covered),
        per_attempt_paid_probability=float(sum_attempts_reached_payout / sum_total_attempts * 100) if sum_total_attempts else 0.0,
        per_attempt_paid_ci95=_wilson_score_interval(int(sum_attempts_reached_payout), int(sum_total_attempts)),
    )
    return result
