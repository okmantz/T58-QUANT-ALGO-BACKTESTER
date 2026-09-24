"""
Backtest engine.

Orchestrates: Dataset + Strategy + Risk/Execution Configuration
           -> Trade List + Equity Curve + Statistics
"""
from __future__ import annotations

import warnings as _warnings_module
from dataclasses import dataclass, field

import pandas as pd

from app.backtest.adaptive_risk import AdaptiveRiskConfig
from app.backtest.execution import Trade, run_execution
from app.backtest.risk import RiskConfig, position_sizing_deviation_message
from app.backtest.statistics import BacktestStatistics, compute_statistics
from app.data.timeframe_resample import prepare_timeframe_aligned_data
from app.strategy.base import Strategy, StrategyResult, apply_days_of_week_exclusion, apply_regime_exclusion


@dataclass
class BacktestResult:
    strategy_name: str
    trades: list[Trade]
    equity_curve: pd.DataFrame
    statistics: BacktestStatistics
    initial_balance: float
    warnings: list[str] = field(default_factory=list)
    # ^ Execution-integrity warnings from app.backtest.execution.run_execution
    # (fallback stops, forced daily-limit closes, pip-size/instrument
    # mismatches, gap-through stop fills). run_execution raises these as
    # ordinary Python RuntimeWarnings; nothing upstream was ever catching
    # them, so on the desktop app (which doesn't surface stderr anywhere)
    # they were silently invisible no matter how serious. Capturing them
    # here, once, means every caller of run_backtest gets them for free.


def run_holdout_comparison(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    holdout_frac: float = 0.2,
) -> dict:
    """
    Chronological in-sample / out-of-sample split, run ONCE.

    Splits df at a single time point (default: the last 20% of bars becomes
    the holdout), runs the identical strategy + risk config independently on
    each half, and returns both statistics sets side by side. Each half is
    passed to the strategy as its own standalone DataFrame -- higher-timeframe
    context the strategy derives internally (e.g. resampling to 1H/4H) is
    therefore also correctly confined to that half, with no leakage of
    future (holdout-period) price action into the in-sample run or vice
    versa.

    This is deliberately NOT a walk-forward optimizer and does not re-tune
    any parameters between segments -- it answers a narrower, falsification-
    style question: "does this exact strategy, as written, keep working on
    data it has never touched?" A strategy whose edge is real should degrade
    gracefully, not evaporate or invert, on the holdout segment.
    """
    n = len(df)
    split_idx = int(n * (1 - holdout_frac))
    split_idx = max(1, min(split_idx, n - 1)) if n > 1 else n

    in_sample_df = df.iloc[:split_idx].reset_index(drop=True)
    holdout_df = df.iloc[split_idx:].reset_index(drop=True)

    in_sample_result = run_backtest(in_sample_df, strategy, risk) if len(in_sample_df) else None
    holdout_result = run_backtest(holdout_df, strategy, risk) if len(holdout_df) else None

    return {
        "holdout_frac": holdout_frac,
        "in_sample_period": (
            (str(in_sample_df["timestamp"].iloc[0]), str(in_sample_df["timestamp"].iloc[-1]))
            if len(in_sample_df) else (None, None)
        ),
        "holdout_period": (
            (str(holdout_df["timestamp"].iloc[0]), str(holdout_df["timestamp"].iloc[-1]))
            if len(holdout_df) else (None, None)
        ),
        "in_sample_bars": len(in_sample_df),
        "holdout_bars": len(holdout_df),
        "in_sample_statistics": in_sample_result.statistics.to_dict() if in_sample_result else None,
        "holdout_statistics": holdout_result.statistics.to_dict() if holdout_result else None,
    }


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    adaptive_risk: AdaptiveRiskConfig | None = None,
) -> BacktestResult:
    """
    df: standardized OHLCV DataFrame (see app.data.importer)
    strategy: any Strategy subclass instance (manual/python/pinescript/mql5)
    risk: RiskConfig describing sizing, costs, and execution assumptions
    adaptive_risk: optional declarative money-management overlay (see
        app.backtest.adaptive_risk) -- None/omitted runs exactly as before
        this parameter existed.
    """
    # FIX (MTF-STRATEGY-001): resample `df` to whatever timeframe(s)
    # `strategy` itself declares it needs (see app.data.timeframe_resample
    # for exactly how each source type declares this) BEFORE the strategy
    # ever sees the data -- this is the one chokepoint every tool in the
    # app funnels through (Run & Report, Full Pipeline, Quick Optimize,
    # Search Lab, Evolution Lab, Forge, Speed Run, CPCV, WFO/WFGA, Multi-
    # Objective, Ensemble, Portfolio, ...), so fixing it here fixes it
    # everywhere with no other caller needing to change. A strategy that
    # declares nothing gets `df` back completely unchanged -- byte-
    # identical to every run before this existed.
    df, timeframe_warnings = prepare_timeframe_aligned_data(df, strategy)

    strat_result: StrategyResult = strategy.generate(df)

    # UPGRADE (day-of-week trading restriction): forces the strategy's own
    # signal flat (0) on any weekday it has declared excluded (Manual's
    # config["filters"]["days_of_week"]["exclude"], Python's
    # EXCLUDE_DAYS_OF_WEEK, or PineScript/MQL5's `// T58_EXCLUDE_DAYS=`
    # directive -- see app.strategy.base.resolve_excluded_days_of_week for
    # the full convention). A strategy that declares nothing here gets its
    # signals back completely unchanged. Applied here, in the one
    # chokepoint every tool in the app funnels through, so it works
    # identically regardless of which tool ran the backtest.
    strat_result.signals = apply_days_of_week_exclusion(df, strat_result.signals, strategy)

    # UPGRADE (regime-conditional-trading primitive): same chokepoint,
    # same "declares nothing -> completely unchanged" convention as the
    # day-of-week exclusion just above -- see app.strategy.base.
    # apply_regime_exclusion/resolve_excluded_regimes for what a strategy
    # can declare and how app.validation.regime_matrix's own regime
    # detection is reused to enforce it.
    strat_result.signals = apply_regime_exclusion(df, strat_result.signals, strategy)

    condition_warnings: list[str] = []
    if getattr(strategy, "source_type", None) == "manual" and isinstance(getattr(strategy, "config", None), dict):
        # Catches a class of bug the GA-gene-bounds fix in
        # app.optimize.parameter_space now prevents going forward, but a
        # hand-typed, imported, or already-saved config can still contain:
        # a comparison against a bounded oscillator (RSI, Stochastic, ...)
        # whose threshold sits outside that oscillator's possible range,
        # silently disabling that branch of the strategy's logic. See
        # app.strategy.manual.validate_bounded_conditions for the full
        # explanation.
        from app.strategy.manual import validate_bounded_conditions
        condition_warnings = validate_bounded_conditions(strategy.config)

    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always", RuntimeWarning)
        trades, equity_curve = run_execution(
            df=df,
            signals=strat_result.signals,
            risk=risk,
            stop_loss_pips=strat_result.stop_loss_pips,
            take_profit_pips=strat_result.take_profit_pips,
            stop_loss_distance=strat_result.stop_loss_distance,
            take_profit_distance=strat_result.take_profit_distance,
            trailing_stop_distance=strat_result.trailing_stop_distance,
            breakeven_trigger_r=strat_result.breakeven_trigger_r,
            partial_exit_config=strat_result.partial_exit,
            adaptive_risk=adaptive_risk,
        )
    execution_warnings = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]

    stats = compute_statistics(trades, equity_curve, initial_balance=risk.initial_balance)

    # UPGRADE (buried-position-sizing-deviation): computed from `stats`,
    # which only exists AFTER the catch_warnings block above closes --
    # unlike condition_warnings/timeframe_warnings, this can't be raised
    # as a RuntimeWarning inside run_execution itself, so it's appended
    # here instead. See app.backtest.risk.position_sizing_deviation_
    # message for what triggers it; None (the common case) adds nothing.
    sizing_warning = position_sizing_deviation_message(stats.to_dict())

    return BacktestResult(
        strategy_name=strat_result.name,
        trades=trades,
        equity_curve=equity_curve,
        statistics=stats,
        initial_balance=risk.initial_balance,
        warnings=timeframe_warnings + condition_warnings + execution_warnings + ([sizing_warning] if sizing_warning else []),
    )
