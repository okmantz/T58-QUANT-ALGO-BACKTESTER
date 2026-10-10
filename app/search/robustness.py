"""
Statistical robustness tools for the Search Lab.

These exist because of one specific risk: running thousands of independent
backtests does not just create more chances to find a real edge -- it
creates, overwhelmingly more, chances to find a FAKE one that looks
excellent purely by chance on this exact historical window. This is the
same trap already hit once by hand in this project (an IV-wall signal that
looked strong on 34 trades and vanished on 97 -- see
/areas/prop-firm-falsification-kit.md). At "thousands of variations" scale
that trap gets much easier to fall into, not harder, unless something
explicitly corrects for it. Every function here does exactly that:

  deflated_sharpe_ratio()            -- corrects the champion's Sharpe for
                                         HOW MANY candidates were tried
                                         before it was picked (Bailey &
                                         Lopez de Prado, "The Deflated
                                         Sharpe Ratio", 2014).
  run_walk_forward()                 -- checks the SAME config (no
                                         re-tuning) across several rolling
                                         train/test folds, not just one
                                         80/20 split.
  parameter_neighborhood_robustness() -- checks whether nearby parameter
                                         values still work. A real edge
                                         lives in a plateau; a value that
                                         only works at one exact parameter
                                         setting is fit to noise in this
                                         window, not to a tradeable effect.

None of these functions can tell you a strategy IS good. They can only
tell you when the evidence for "good" doesn't hold up to scrutiny -- which,
at this scale, is the more important of the two questions.
"""
from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Standard normal helpers (no scipy dependency, to match this repo's
# existing "keep external dependencies minimal" convention).
# ---------------------------------------------------------------------------

def _norm_ppf(p: float) -> float:
    """Inverse CDF of the standard normal (Acklam's rational approximation, ~1e-9 accurate)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Deflated / Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------

@dataclass
class DeflatedSharpeResult:
    observed_sharpe: float
    n_trials: int
    n_trade_returns: int
    benchmark_sharpe: float        # E[max Sharpe] expected by CHANCE across n_trials independent trials
    probabilistic_sharpe: float    # P(true Sharpe > benchmark_sharpe | observed sample), 0-1
    deflated_sharpe: float         # alias of probabilistic_sharpe -- the number to actually look at
    is_significant: bool
    significance_threshold: float
    note: str


def expected_max_sharpe(trial_sharpes: list[float], n_trials: int) -> float:
    """
    Bailey & Lopez de Prado's approximation of the best Sharpe ratio you'd
    expect to see by PURE CHANCE across `n_trials` independent backtests,
    given the actual observed spread (std dev) of Sharpe ratios across
    those trials. This -- not zero -- is the bar a search's champion
    candidate has to clear.
    """
    finite = [s for s in trial_sharpes if isinstance(s, (int, float)) and math.isfinite(s)]
    if n_trials <= 1 or len(finite) < 2:
        return 0.0
    sigma_sr = float(np.std(finite, ddof=1))
    if sigma_sr <= 0:
        return 0.0
    n = max(int(n_trials), 2)
    z1 = _norm_ppf(1 - 1.0 / n)
    z2 = _norm_ppf(1 - 1.0 / (n * math.e))
    return sigma_sr * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    trial_sharpes: list[float],
    n_trials: int,
    n_trade_returns: int,
    returns_skew: float = 0.0,
    returns_kurtosis: float = 3.0,
    significance_threshold: float = 0.95,
) -> DeflatedSharpeResult:
    """
    Deflates a candidate's raw Sharpe ratio for having been selected as the
    best of `n_trials` independent candidates, and returns a Probabilistic
    Sharpe Ratio (PSR) against that deflated benchmark rather than against
    zero. Interpretation: `probabilistic_sharpe` is the estimated
    probability that this candidate's TRUE (out-of-sample, infinite-data)
    Sharpe genuinely exceeds what you'd expect the best of `n_trials` random
    candidates to show by chance alone. Values near 1.0 are a real signal;
    values near 0.5 mean "indistinguishable from the best of the crowd you
    tried, which is exactly what pure luck would also produce."

    `returns_skew` / `returns_kurtosis` (of the trade-level P&L
    distribution, not of price returns) are worth passing when known --
    Sharpe ratios computed on fat-tailed, skewed trade distributions
    (typical for short-RR, high win-rate prop strategies) are noisier than
    the same Sharpe on a symmetric distribution, and the PSR formula
    accounts for that directly. Defaulting to 0/3 (normal) is conservative
    only in the sense of being the textbook default, not in the sense of
    understating risk -- pass real values when you have them.
    """
    benchmark = expected_max_sharpe(trial_sharpes, n_trials)
    n = max(int(n_trade_returns), 2)
    sr = float(observed_sharpe) if math.isfinite(observed_sharpe) else 0.0
    denom = math.sqrt(max(1 - returns_skew * sr + ((returns_kurtosis - 1) / 4.0) * sr * sr, 1e-9))
    z = (sr - benchmark) * math.sqrt(n - 1) / denom
    psr = _norm_cdf(z)
    return DeflatedSharpeResult(
        observed_sharpe=sr,
        n_trials=int(n_trials),
        n_trade_returns=n,
        benchmark_sharpe=benchmark,
        probabilistic_sharpe=psr,
        deflated_sharpe=psr,
        is_significant=psr >= significance_threshold,
        significance_threshold=significance_threshold,
        note=(
            f"Deflated against an expected best-of-{n_trials} chance Sharpe of "
            f"{benchmark:.3f}, derived from the spread of Sharpe ratios this search's own "
            f"Stage 1 pass actually produced (not an assumed/textbook spread)."
        ),
    )


# ---------------------------------------------------------------------------
# Walk-forward validation (multiple rolling folds, no re-tuning between them)
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    fold_index: int
    train_period: tuple
    test_period: tuple
    train_bars: int
    test_bars: int
    train_metric: float
    test_metric: float


@dataclass
class WalkForwardResult:
    folds: list          # list[WalkForwardFold]
    n_folds: int
    metric: str
    mean_train_metric: float
    mean_test_metric: float
    walk_forward_efficiency: float   # mean_test_metric / mean_train_metric, clipped to [-5, 5]
    is_stable: bool
    stability_threshold: float
    # VAL-005: bars actually skipped at the start of `df` because of
    # `embargo_start_bar` (0 if no embargo was requested, or if it was
    # requested but the whole dataset was already past that bar) -- see
    # run_walk_forward's own docstring for why this exists.
    embargo_applied_bars: int = 0
    warnings: list = field(default_factory=list)


def _metric_value(stats: dict, metric: str, trades: list | None = None, prop_rules=None, mc_cfg=None) -> float:
    """Reads a fold's score off its backtest stats dict, EXCEPT for
    "eval_pass_probability": that one isn't in the stats dict at all (it
    only exists after a Monte Carlo run), so when prop_rules is supplied
    this runs a small per-fold Monte Carlo instead. This is what lets
    walk-forward/CPCV fold scoring actually answer "probability of
    reaching the profit target before hitting a limit" per fold, rather
    than silently falling back to 0.0 for a key that was never there."""
    if metric == "eval_pass_probability" and prop_rules is not None:
        from app.monte_carlo.engine import eval_pass_probability_for_trades
        return eval_pass_probability_for_trades(trades or [], prop_rules, mc_cfg)
    v = stats.get(metric, 0.0)
    if v == float("inf"):
        return 10.0
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return 0.0
    return float(v)


# v9.15 (WS-F): worker state/task for parallel run_walk_forward. Folds
# are independent (fixed config, disjoint slice construction), so they
# can evaluate across a process pool. The builder lambda most callers
# pass is not picklable under spawn, so parallel execution is
# spec-driven (workers rebuild from `strategy_spec` per fold); results
# are collected by fold index, keeping the returned WalkForwardResult
# identical to the serial run. Any pool failure falls back to the
# serial builder loop for every fold.
_WF_WORKER: dict = {}


def _wf_worker_init(df, risk, metric, prop_rules, mc_cfg, strategy_spec, tmp_dir) -> None:
    global _WF_WORKER
    _WF_WORKER = {
        "df": df, "risk": risk, "metric": metric, "prop_rules": prop_rules,
        "mc_cfg": mc_cfg, "spec": strategy_spec, "tmp_dir": tmp_dir,
    }


def _wf_one_fold(fold_idx, train_df, test_df, strategy_builder, risk, metric, prop_rules, mc_cfg):
    """One walk-forward fold: two backtests + two metric reads. Returns
    (WalkForwardFold, train_val, test_val), or None for a thin fold --
    exactly the serial loop's `continue`. Pure function of its inputs."""
    from app.backtest.engine import run_backtest

    if len(train_df) < 10 or len(test_df) < 5:
        return None
    train_result = run_backtest(train_df, strategy_builder(), risk)
    test_result = run_backtest(test_df, strategy_builder(), risk)
    train_stats = train_result.statistics.to_dict()
    test_stats = test_result.statistics.to_dict()
    train_val = _metric_value(train_stats, metric, train_result.trades, prop_rules, mc_cfg)
    test_val = _metric_value(test_stats, metric, test_result.trades, prop_rules, mc_cfg)
    fold = WalkForwardFold(
        fold_index=fold_idx,
        train_period=(str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])),
        test_period=(str(test_df["timestamp"].iloc[0]), str(test_df["timestamp"].iloc[-1])),
        train_bars=len(train_df), test_bars=len(test_df),
        train_metric=train_val, test_metric=test_val,
    )
    return fold, train_val, test_val


def _wf_worker_task(task):
    fold_idx, train_end, test_end = task
    w = _WF_WORKER
    df = w["df"]
    train_df = df.iloc[:train_end].reset_index(drop=True)
    test_df = df.iloc[train_end:test_end].reset_index(drop=True)
    return fold_idx, _wf_one_fold(
        fold_idx, train_df, test_df,
        lambda: _build_spec_strategy(w["spec"], w["tmp_dir"]),
        w["risk"], w["metric"], w["prop_rules"], w["mc_cfg"],
    )


def _build_spec_strategy(spec, tmp_dir):
    from app.search.strategy_space import build_strategy_from_spec

    return build_strategy_from_spec(spec, tmp_dir)


def run_walk_forward(
    df: pd.DataFrame,
    strategy_builder,
    risk,
    n_folds: int = 4,
    metric: str = "eval_pass_probability",
    stability_threshold: float = 0.4,
    prop_rules=None,
    mc_cfg=None,
    embargo_start_bar: int | None = None,
    max_workers: int | None = None,
    strategy_spec=None,
    tmp_dir=None,
):
    """
    strategy_builder: a zero-argument callable returning a FRESH Strategy
    instance each call (cheap for ManualStrategy -- always build fresh per
    fold rather than reusing one instance, since some strategy sources
    cache internal state keyed to the data they last saw).

    Splits `df` into `n_folds` chronological, EXPANDING train windows each
    followed by a contiguous out-of-sample test slice -- fold i trains on
    bars [0, split_i) and tests on the next slice, walking split_i forward
    each fold. The exact same config is used for every fold with NO
    re-tuning in between (consistent with this app's existing
    run_holdout_comparison philosophy in app/backtest/engine.py) -- this
    answers "does this exact strategy, as found, keep working across
    several distinct historical stretches?", not "what's the best
    per-period configuration?" (that second question is what Stage 2's GA
    already answered, on the full dataset).

    embargo_start_bar: VAL-005 fix. When a GA (e.g.
    app.optimize.walkforward_ga) already searched/selected against this
    same `df` using its own chained-out-of-sample fold construction, this
    function's own EXPANDING fold construction used to overlap those same
    fold test windows almost entirely (verified: with this app's actual
    default settings, 3 of 4 "out-of-sample" folds here substantially or
    fully overlapped the GA's own fold test windows) -- meaning what this
    function reports as evidence the winning configuration "keeps working
    across several distinct historical stretches" was, for most of those
    stretches, re-confirming performance on data the GA already selected
    the winner FOR being good at, not independent evidence. Pass the
    furthest-forward bar the GA's own folds touched (see
    WalkforwardGAResult.max_test_bar_used) here to skip straight past it
    before building this function's OWN folds, so every fold this
    function reports is guaranteed to start on data the GA never saw.
    None (default) reproduces the original, unembargoed behavior exactly.

    Returns None (not a failure) when there isn't enough data to fold
    meaningfully -- callers should treat that as "unproven", not "failed".
    """
    from app.backtest.engine import run_backtest

    if metric == "eval_pass_probability" and prop_rules is None:
        # Can't run a per-fold Monte Carlo without prop rules -- fall
        # back to a reported metric rather than silently scoring every
        # fold 0.0. Callers doing prop-firm work should pass prop_rules.
        metric = "profit_factor"

    warnings: list[str] = []
    embargo_applied_bars = 0
    if embargo_start_bar is not None and embargo_start_bar > 0:
        original_len = len(df)
        embargo_applied_bars = min(embargo_start_bar, original_len)
        df = df.iloc[embargo_applied_bars:].reset_index(drop=True)
        if embargo_applied_bars >= original_len:
            warnings.append(
                f"Embargo of {embargo_start_bar} bar(s) (to avoid re-testing data the "
                "optimization search already used) left zero bars for this out-of-sample "
                "check -- the dataset isn't long enough for both a search AND a genuinely "
                "independent walk-forward check on top of it."
            )
        else:
            warnings.append(
                f"Embargoed the first {embargo_applied_bars:,} bar(s) of this dataset (already "
                "used by the optimization search's own out-of-sample folds) before building "
                f"these {n_folds} walk-forward fold(s), so every fold below starts on data the "
                "search never saw."
            )

    n = len(df)
    if n_folds < 2 or n < n_folds * 20:
        return None

    fold_size = n // (n_folds + 1)
    if fold_size < 10:
        return None

    # v9.15 (WS-F): fold spans first, then serial -- or, when a
    # candidate spec is supplied and max_workers > 1, spec-driven
    # parallel evaluation with indexed collection. `folds` and the two
    # metric lists end up in fold order either way, so the returned
    # WalkForwardResult is identical; any pool failure falls back to
    # the serial builder loop for every fold.
    fold_spans = [
        (i, fold_size * (i + 1), min(fold_size * (i + 2), n)) for i in range(n_folds)
    ]
    fold_results: list = [None] * len(fold_spans)
    _want_wf_parallel = (
        strategy_spec is not None and max_workers is not None
        and int(max_workers) > 1 and len(fold_spans) > 1
    )
    if _want_wf_parallel:
        try:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            from app.orchestration.resource_guard import safe_worker_count

            workers = safe_worker_count(
                df, requested=int(max_workers), max_candidates_in_flight=len(fold_spans))
            if workers > 1:
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=ctx,
                    initializer=_wf_worker_init,
                    initargs=(df, risk, metric, prop_rules, mc_cfg,
                              strategy_spec, str(tmp_dir) if tmp_dir is not None else None),
                ) as pool:
                    for _fi, _res in pool.map(_wf_worker_task, fold_spans):
                        fold_results[_fi] = _res
        except Exception:  # noqa: BLE001 -- serial below is the same math
            fold_results = [None] * len(fold_spans)

    folds: list[WalkForwardFold] = []
    train_metrics, test_metrics = [], []
    for i, train_end, test_end in fold_spans:
        _one = fold_results[i]
        if _one is None:
            train_df = df.iloc[:train_end].reset_index(drop=True)
            test_df = df.iloc[train_end:test_end].reset_index(drop=True)
            _one = _wf_one_fold(i, train_df, test_df, strategy_builder, risk, metric, prop_rules, mc_cfg)
        if _one is None:
            continue
        _fold, train_val, test_val = _one
        train_metrics.append(train_val)
        test_metrics.append(test_val)
        folds.append(_fold)

    if not folds:
        return None

    mean_train = float(np.mean(train_metrics))
    mean_test = float(np.mean(test_metrics))
    if mean_train == 0:
        wfe = 0.0
    else:
        wfe = max(min(mean_test / mean_train, 5.0), -5.0)

    return WalkForwardResult(
        folds=folds, n_folds=len(folds), metric=metric,
        mean_train_metric=mean_train, mean_test_metric=mean_test,
        walk_forward_efficiency=wfe,
        is_stable=wfe >= stability_threshold,
        stability_threshold=stability_threshold,
        embargo_applied_bars=embargo_applied_bars,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Parameter-neighborhood robustness
# ---------------------------------------------------------------------------

@dataclass
class RobustnessResult:
    n_neighbors_tested: int
    baseline_fitness: float
    neighbor_fitnesses: list
    mean_neighbor_fitness: float
    min_neighbor_fitness: float
    stability_ratio: float
    is_stable: bool
    stability_threshold: float


def parameter_neighborhood_robustness(
    candidate_spec: dict,
    df: pd.DataFrame,
    risk,
    prop_rules,
    mc_config,
    fitness_metric: str = "eval_pass_probability",
    perturbation_frac: float = 0.15,
    n_neighbors: int = 6,
    seed: int = 42,
    stability_threshold: float = 0.4,
    tmp_dir=None,
):
    """
    Perturbs every tunable numeric parameter in `candidate_spec` by up to
    +/- perturbation_frac of that parameter's normal search range and
    re-evaluates `n_neighbors` such perturbed neighbors. A real edge lives
    in a plateau -- nearby parameter values should perform similarly. A
    candidate whose score collapses a few percent off its exact winning
    parameters is very likely fit to noise in this specific historical
    window, not to a real, tradeable effect.

    `candidate_spec` is the same uniform candidate-spec dict used
    throughout the Search Lab (see app.search.strategy_space):
        {"source_type": "manual", "config": {...}}
        {"source_type": "python"/"pinescript"/"mql5", "code_text": "...", "code_extension": "..."}
    Manual parameters are discovered/applied via app.optimize.parameter_space
    (extract_genome/apply_genome); Python/PineScript/MQL5 parameters via
    app.optimize.code_parameter_space (discover_code_genes/
    materialize_code_strategy) -- the exact same machinery Step 6's
    Iterative Refinement GA already uses for each source type. `tmp_dir` is
    required when `candidate_spec` is a Python candidate (PythonStrategy
    only accepts a file path).

    Returns None (not a failure) if the strategy has no tunable numeric
    parameters to perturb -- callers should treat that as "unknown", not
    "unstable".
    """
    from app.backtest.engine import run_backtest
    from app.monte_carlo.engine import run_monte_carlo
    from app.optimize.code_parameter_space import discover_code_genes, materialize_code_strategy
    from app.optimize.parameter_space import apply_genome, extract_genome
    from app.optimize.refinement import compute_fitness
    from app.prop.simulator import simulate_account, summarize_single_run
    from app.search.strategy_space import build_strategy_from_spec
    from app.strategy.manual import ManualStrategy

    source_type = candidate_spec.get("source_type", "manual")
    base_strategy = build_strategy_from_spec(candidate_spec, tmp_dir)

    if source_type == "manual":
        genes = extract_genome(candidate_spec["config"])
    else:
        genes = discover_code_genes(base_strategy)
    if not genes:
        return None

    rng = _random.Random(seed)

    def _fitness_for(strategy) -> float:
        bt = run_backtest(df, strategy, risk)
        if not bt.trades:
            return float("-inf")
        pnls = [t.pnl for t in bt.trades]
        dates = [t.entry_time for t in bt.trades]
        single_run = simulate_account(pnls, dates, prop_rules)
        mc = run_monte_carlo(bt.trades, prop_rules, mc_config)
        prop_summary = summarize_single_run(single_run)
        fitness = compute_fitness(bt.statistics.to_dict(), prop_summary, mc, fitness_metric)
        return fitness if math.isfinite(fitness) else float("-inf")

    baseline_fitness = _fitness_for(build_strategy_from_spec(candidate_spec, tmp_dir))

    neighbor_fitnesses: list[float] = []
    for _ in range(n_neighbors):
        perturbed_genome = []
        for g in genes:
            span = g.hi - g.lo
            delta = rng.uniform(-perturbation_frac, perturbation_frac) * span
            v = min(max(g.base_value + delta, g.lo), g.hi)
            perturbed_genome.append(float(round(v)) if g.is_int else float(v))

        if source_type == "manual":
            neighbor_strategy = ManualStrategy(apply_genome(candidate_spec["config"], genes, perturbed_genome))
        else:
            neighbor_strategy = materialize_code_strategy(base_strategy, genes, perturbed_genome, tmp_dir)
        neighbor_fitnesses.append(_fitness_for(neighbor_strategy))

    finite_neighbors = [f for f in neighbor_fitnesses if math.isfinite(f)]
    if not finite_neighbors or not math.isfinite(baseline_fitness) or baseline_fitness <= 0:
        stability_ratio = 0.0
    else:
        stability_ratio = max(min(min(finite_neighbors) / baseline_fitness, 5.0), -5.0)

    return RobustnessResult(
        n_neighbors_tested=len(neighbor_fitnesses),
        baseline_fitness=baseline_fitness,
        neighbor_fitnesses=neighbor_fitnesses,
        mean_neighbor_fitness=float(np.mean(finite_neighbors)) if finite_neighbors else float("-inf"),
        min_neighbor_fitness=float(min(finite_neighbors)) if finite_neighbors else float("-inf"),
        stability_ratio=stability_ratio,
        is_stable=stability_ratio >= stability_threshold,
        stability_threshold=stability_threshold,
    )


# ---------------------------------------------------------------------------
# Gate primitives: honest trial counting + the DSR as a REJECT gate
# ---------------------------------------------------------------------------

def count_all_trials(
    baseline_count: int = 1,
    ga_total_evaluations: int = 0,
    extra_trial_counts: tuple[int, ...] = (),
) -> int:
    """The n_trials fix for deflated_sharpe_ratio: EVERY parameter
    configuration a pipeline actually evaluated counts as a trial -- not
    just the screening-stage candidates that survived onto a leaderboard.

    The classic undercount (this repo's own Search Lab Stage 4 did exactly
    this): ``n_trials = len(stage1_records)``, silently ignoring every
    genome the Stage-2 GA backtested in its inner loop while refining the
    survivors. Those inner-loop trials are still chances the search had to
    get lucky, so omitting them understates the expected best-of-N chance
    Sharpe -- and an understated chance benchmark makes the deflated Sharpe
    ratio OPTIMISTIC: the champion looks more significant than the search's
    actual multiple-testing burden justifies. Count everything.

    baseline_count: configurations evaluated outside any optimizer loop
        (usually 1: the supplied / baseline strategy itself).
    ga_total_evaluations: every genome the optimizer actually backtested --
        e.g. WalkforwardGAResult.total_evaluations /
        RefinementResult.total_evaluations. Pass the ACTUALLY-RAN count,
        not the configured population*(generations+1) budget: auto-shrink-
        on-low-trades and AI-assist can push the real number either side
        of the budget (see the Bonferroni-vs-actual-evaluations fix in
        app.orchestration.full_pipeline).
    extra_trial_counts: any other evaluated-configuration counts (AI-
        suggested candidates, manual spot-checks, ...) as ints.

    Returns at least 1 -- a Sharpe that was computed at all was selected
    from at least one trial.
    """
    total = max(int(baseline_count), 0) + max(int(ga_total_evaluations), 0)
    for c in extra_trial_counts:
        total += max(int(c), 0)
    return max(total, 1)


@dataclass
class DeflatedSharpeGateResult:
    passed: bool
    probabilistic_sharpe: float      # the champion's PSR vs the deflated benchmark, 0-1
    min_probabilistic_sharpe: float  # the bar it had to clear
    n_trials: int                    # the (honestly counted) trial count behind the deflation
    benchmark_sharpe: float          # E[best Sharpe by chance] across n_trials
    reason: str                      # one human-readable line, pass or fail
    dsr_result: "DeflatedSharpeResult | None" = None


def deflated_sharpe_gate(
    observed_sharpe: float,
    trial_sharpes: list[float],
    n_trials: int,
    n_trade_returns: int,
    returns_skew: float = 0.0,
    returns_kurtosis: float = 3.0,
    min_probabilistic_sharpe: float = 0.95,
) -> DeflatedSharpeGateResult:
    """The Deflated Sharpe Ratio as a REJECT gate, not a score penalty.

    deflated_sharpe_ratio() above only ever produced ``is_significant`` as
    a diagnostic flag -- callers folded it into a weighted score (Search
    Lab Stage 4: 25% of the composite), so a spectacularly overfit
    champion could still be crowned as long as its other numbers were
    loud enough. This wrapper turns the same Bailey & Lopez de Prado
    (2014) computation into a pass/fail verdict: below
    ``min_probabilistic_sharpe`` the candidate is REJECTED outright,
    because "indistinguishable from the best of N random tries" is not a
    weak signal to weigh -- it is the absence of evidence.

    ``n_trials`` should come from count_all_trials() above, not from a
    leaderboard length: the whole point of the deflation is the search's
    full multiple-testing burden, GA inner-loop trials included.

    min_probabilistic_sharpe default 0.95: Bailey & de Prado's recommended
    significance bar, and the same default deflated_sharpe_ratio() itself
    uses -- a gate should not be looser than the diagnostic it promotes.
    """
    dsr = deflated_sharpe_ratio(
        observed_sharpe,
        trial_sharpes,
        n_trials,
        n_trade_returns,
        returns_skew=returns_skew,
        returns_kurtosis=returns_kurtosis,
        significance_threshold=min_probabilistic_sharpe,
    )
    passed = dsr.deflated_sharpe >= min_probabilistic_sharpe
    if passed:
        reason = (
            f"Deflated Sharpe gate PASSED: probabilistic Sharpe {dsr.deflated_sharpe:.3f} >= "
            f"{min_probabilistic_sharpe:.2f} (deflated against an expected best-of-{dsr.n_trials} "
            f"chance Sharpe of {dsr.benchmark_sharpe:.3f})."
        )
    else:
        reason = (
            f"REJECTED by the Deflated Sharpe gate: probabilistic Sharpe {dsr.deflated_sharpe:.3f} < "
            f"{min_probabilistic_sharpe:.2f} -- after deflating for {dsr.n_trials} evaluated "
            f"configuration(s) (expected best-by-chance Sharpe {dsr.benchmark_sharpe:.3f}), this "
            f"champion's Sharpe of {dsr.observed_sharpe:.3f} is indistinguishable from what the "
            f"search would produce by luck alone."
        )
    return DeflatedSharpeGateResult(
        passed=passed,
        probabilistic_sharpe=dsr.deflated_sharpe,
        min_probabilistic_sharpe=min_probabilistic_sharpe,
        n_trials=dsr.n_trials,
        benchmark_sharpe=dsr.benchmark_sharpe,
        reason=reason,
        dsr_result=dsr,
    )
