"""
Execution desk -- the seam the article points at directly: "[NautilusTrader]
deliberately ships no 'rebalance to target weights' helper. That missing
piece is exactly where our glue code goes."

This module IS that glue code, except instead of wiring to a new
NautilusTrader install, it plays the same role this app's own
`app.backtest.engine` already plays for single-instrument strategies:
turn a decision (there: a -1/0/1 signal; here: a target weight vector)
into a discrete, cost-aware trade sequence and a walked-forward equity
curve.

Deliberately NOT built on top of app.backtest.engine.run_backtest, for
the same reason app.portfolio.portfolio.py already gives for not reusing
it: that engine's trade model is single-instrument, single-open-position,
signal-driven. A rebalance-cycle book is a genuinely different trade
model (N simultaneous target weights, walked forward and re-costed every
cycle) and forcing it through the signal engine would be the wrong kind
of code reuse -- it would obscure the model, not simplify it. Stats here
are therefore computed directly against the resulting equity curve rather
than via app.backtest.statistics.compute_statistics, which expects a
list of discrete long/flat/short Trade objects this book never produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.hedge_fund.black_litterman import BlackLittermanError, solve_posterior_weights, views_to_posterior
from app.hedge_fund.research import AssetView, EnsembleForecastConfig, ResearchError, generate_views
from app.quant_lab.portfolio_optimizer import PortfolioOptimizerError, build_inputs_from_prices
from app.strategy.base import Strategy


class RebalanceError(Exception):
    """Raised when a rebalance cycle backtest cannot proceed."""


@dataclass
class Order:
    timestamp: pd.Timestamp
    asset: str
    side: str            # "buy" | "sell"
    notional: float        # dollar value traded (always positive)
    price: float
    shares_delta: float     # signed change in shares held


@dataclass
class RebalanceCycle:
    timestamp: pd.Timestamp
    views: dict[str, AssetView]
    prior_mu: dict[str, float]
    posterior_mu: dict[str, float]
    target_weights: dict[str, float]
    turnover: float          # sum(|w_new - w_old|), 0..2
    orders: list[Order]
    warnings: list[str] = field(default_factory=list)


@dataclass
class RebalanceConfig:
    initial_balance: float = 100_000.0
    rebalance_every_bars: int = 5     # matches the article's weekly cadence on daily bars
    forecast: EnsembleForecastConfig = field(default_factory=EnsembleForecastConfig)
    confidence: float = 0.5             # the Black-Litterman "how much to trust the research desk" dial, 0..1
    long_only: bool = True
    max_turnover: float | None = 0.5      # cap on sum(|w_new - w_old|) per cycle; None = uncapped
    transaction_cost_bps: float = 10.0     # matches the article's "10 bps each way"
    min_trade_frac: float = 0.005            # skip rebalance trades smaller than this fraction of equity (dust)
    risk_free_rate: float = 0.0


@dataclass
class RebalanceStats:
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    num_rebalances: int
    avg_turnover_pct: float
    total_transaction_costs: float


@dataclass
class HedgeFundBacktestResult:
    equity_curve: pd.DataFrame               # columns: timestamp, equity
    cycles: list[RebalanceCycle]
    stats: RebalanceStats
    warnings: list[str] = field(default_factory=list)


def _align_price_data(price_data: dict[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """Inner-joins every asset's df on timestamp so the walk-forward loop
    can step one shared bar index at a time. Assets with gaps relative to
    the rest of the universe lose those bars from the whole book, same
    trade-off app.quant_lab.portfolio_optimizer.build_inputs_from_prices
    already documents and accepts for its own overlapping-dates join.
    """
    aligned = {}
    common_ts = None
    for asset, df in price_data.items():
        s = df.copy()
        s["timestamp"] = pd.to_datetime(s["timestamp"])
        s = s.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        aligned[asset] = s
        ts_index = pd.DatetimeIndex(s["timestamp"])
        common_ts = ts_index if common_ts is None else common_ts.intersection(ts_index)
    if common_ts is None or len(common_ts) < 30:
        raise RebalanceError(
            f"Only {0 if common_ts is None else len(common_ts)} overlapping bars across all assets -- "
            f"need at least 30 to run a rebalance backtest."
        )
    common_ts = common_ts.sort_values()
    out = {asset: df[df["timestamp"].isin(common_ts)].reset_index(drop=True) for asset, df in aligned.items()}
    return common_ts, out


def weights_to_orders(
    timestamp: pd.Timestamp, target_weights: dict[str, float], current_shares: dict[str, float],
    prices: dict[str, float], equity: float, min_trade_frac: float,
) -> tuple[list[Order], dict[str, float]]:
    """Compares target weights to current holdings and returns the orders
    needed to close the gap, plus the resulting share counts. This is
    exactly the diff-and-submit step the article's StackStrategy.on_event
    does inline against NautilusTrader's order factory -- here it's a
    pure function so it can be unit-tested without a running engine."""
    orders: list[Order] = []
    new_shares = dict(current_shares)
    min_trade_notional = equity * min_trade_frac
    for asset, target_w in target_weights.items():
        price = prices.get(asset)
        if price is None or price <= 0:
            continue
        target_shares = (target_w * equity) / price
        current = current_shares.get(asset, 0.0)
        delta_shares = target_shares - current
        notional = abs(delta_shares) * price
        if notional < min_trade_notional:
            continue
        orders.append(Order(
            timestamp=timestamp, asset=asset, side="buy" if delta_shares > 0 else "sell",
            notional=notional, price=price, shares_delta=delta_shares,
        ))
        new_shares[asset] = target_shares
    return orders, new_shares


def _mark_to_market(shares: dict[str, float], prices: dict[str, float], cash: float) -> float:
    return cash + sum(shares.get(a, 0.0) * prices.get(a, 0.0) for a in shares)


def run_rebalance_backtest(
    price_data: dict[str, pd.DataFrame],
    config: RebalanceConfig,
    strategies: dict[str, Strategy] | None = None,
) -> HedgeFundBacktestResult:
    """The whole four-desk cycle, walked forward across the backtest
    window -- "the whole system as a single Monday afternoon," repeated
    every `config.rebalance_every_bars` bars. Research -> Black-Litterman
    -> weights -> orders happens once per cycle; equity is marked to
    market on every bar in between.
    """
    if len(price_data) < 2:
        raise RebalanceError("Need at least 2 assets to run a hedge fund backtest (1 asset has no portfolio to construct).")

    common_ts, aligned = _align_price_data(price_data)
    n_bars = len(common_ts)
    lookback = config.forecast.lookback_bars
    if n_bars <= lookback + config.forecast.horizon_bars:
        raise RebalanceError(
            f"Only {n_bars} overlapping bars across all assets -- need more than lookback ({lookback}) + "
            f"horizon ({config.forecast.horizon_bars}) to run even one rebalance cycle."
        )

    assets = list(aligned.keys())
    close_lookup = {a: aligned[a]["close"].to_numpy() for a in assets}

    cash = config.initial_balance
    shares: dict[str, float] = {a: 0.0 for a in assets}
    weights_now: dict[str, float] = {a: 0.0 for a in assets}
    equity_rows = []
    cycles: list[RebalanceCycle] = []
    warnings: list[str] = []
    total_costs = 0.0

    for i in range(n_bars):
        prices_i = {a: float(close_lookup[a][i]) for a in assets}
        equity = _mark_to_market(shares, prices_i, cash)
        equity_rows.append((common_ts[i], equity))

        is_rebalance_bar = (i >= lookback) and ((i - lookback) % config.rebalance_every_bars == 0)
        if not is_rebalance_bar:
            continue

        as_of_slice = {a: aligned[a].iloc[: i + 1] for a in assets}
        try:
            views = generate_views(as_of_slice, config.forecast, strategies)
        except ResearchError as exc:
            warnings.append(f"{common_ts[i]}: research desk skipped this cycle -- {exc}")
            continue

        try:
            inputs = build_inputs_from_prices(as_of_slice)
        except PortfolioOptimizerError as exc:
            warnings.append(f"{common_ts[i]}: could not build covariance inputs -- {exc}")
            continue

        prior_mu, posterior_mu = views_to_posterior(inputs, views, config.confidence)
        try:
            allocation, solve_warnings = solve_posterior_weights(
                inputs, posterior_mu, long_only=config.long_only, previous_weights=weights_now,
                max_turnover=config.max_turnover, transaction_cost_bps=config.transaction_cost_bps,
                risk_free_rate=config.risk_free_rate,
            )
        except BlackLittermanError as exc:
            warnings.append(f"{common_ts[i]}: portfolio desk could not solve weights -- {exc}")
            continue

        orders, new_shares = weights_to_orders(
            common_ts[i], allocation.weights, shares, prices_i, equity, config.min_trade_frac,
        )
        cycle_cost = sum(o.notional for o in orders) * (config.transaction_cost_bps / 10_000.0)
        cash -= cycle_cost
        total_costs += cycle_cost
        # Cash funds/absorbs the notional delta of every order (weights
        # sum to <=1 so this book never uses margin beyond what's raised
        # by simultaneous sells).
        for o in orders:
            cash -= o.shares_delta * o.price
        shares = new_shares
        turnover = sum(abs(allocation.weights.get(a, 0.0) - weights_now.get(a, 0.0)) for a in assets)
        weights_now = dict(allocation.weights)

        cycles.append(RebalanceCycle(
            timestamp=common_ts[i], views=views, prior_mu=dict(zip(inputs.tickers, prior_mu.tolist())),
            posterior_mu=dict(zip(inputs.tickers, posterior_mu.tolist())), target_weights=allocation.weights,
            turnover=turnover, orders=orders, warnings=solve_warnings,
        ))

    equity_curve = pd.DataFrame(equity_rows, columns=["timestamp", "equity"])
    stats = _compute_stats(equity_curve, cycles, config, total_costs)
    return HedgeFundBacktestResult(equity_curve=equity_curve, cycles=cycles, stats=stats, warnings=warnings)


def _compute_stats(
    equity_curve: pd.DataFrame, cycles: list[RebalanceCycle], config: RebalanceConfig, total_costs: float,
) -> RebalanceStats:
    equity = equity_curve["equity"].to_numpy()
    if len(equity) < 2 or equity[0] <= 0:
        return RebalanceStats(0.0, 0.0, 0.0, 0.0, len(cycles), 0.0, total_costs)

    total_return_pct = (equity[-1] / equity[0] - 1.0) * 100.0
    n_days = max((equity_curve["timestamp"].iloc[-1] - equity_curve["timestamp"].iloc[0]).days, 1)
    years = n_days / 365.25
    cagr_pct = ((equity[-1] / equity[0]) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and equity[-1] > 0 else 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_drawdown_pct = float(drawdowns.min()) * 100.0

    daily_returns = np.diff(equity) / equity[:-1]
    sharpe_ratio = (
        float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(config.forecast.bars_per_year))
        if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0 else 0.0
    )

    avg_turnover_pct = float(np.mean([c.turnover for c in cycles])) * 100.0 if cycles else 0.0
    return RebalanceStats(
        total_return_pct=total_return_pct, cagr_pct=cagr_pct, max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio, num_rebalances=len(cycles), avg_turnover_pct=avg_turnover_pct,
        total_transaction_costs=total_costs,
    )
