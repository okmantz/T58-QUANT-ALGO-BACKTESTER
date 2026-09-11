"""
Research desk -- produces a per-asset forward VIEW (an expected return and
how much to trust it), the same job description the article gives Kronos:
"produces views with uncertainty attached, not commands."

We deliberately do NOT vendor a foundation model here. Three reasons, all
from the source article itself:
  1. An independent test scored Kronos at a Brier of 0.189 on 5-minute
     BTC against 0.188 for a plain Brownian-motion baseline -- i.e.
     statistically indistinguishable from a coin flip.
  2. Kronos ships with no pip package (repo clone only) and no official
     Windows-friendly install path, which is a real cost against this
     app's "one .exe, no manual setup" delivery goal.
  3. The architectural lesson that actually matters -- SAMPLE many
     forecasts, use the mean as the view and the DISPERSION across
     samples as the confidence fed to the optimizer -- is fully
     reproducible without a trained model at all.

So this module implements that lesson two ways:

  `bootstrap_view()`     -- the honest statistical baseline. Resamples
                             the asset's own realized bar-to-bar returns
                             (with replacement) into `n_samples` synthetic
                             horizon-length paths. This assumes NO edge
                             (it's a resample of history, not a forecast
                             of direction) -- its job is to supply a
                             calibrated Sigma (uncertainty) for the
                             optimizer even when there's no real view,
                             which is exactly what Black-Litterman needs
                             to safely fall back to the market-equilibrium
                             weights (see black_litterman.py).

  `strategy_signal_view()` -- the module that actually earns this app's
                             name. Runs one of YOUR existing Strategy
                             objects (manual/Python/PineScript/MQL5,
                             anything already in the Strategy Library) on
                             the lookback window, reads its current
                             signal, and turns the strategy's own recent
                             same-direction hit rate into a mu/sigma pair.
                             This is the difference between "some
                             forecaster" and "the research your T58
                             strategies already do."

Both return an AssetView with the SAME shape, so the portfolio desk never
needs to know which one produced it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.strategy.base import Strategy, StrategyError

DEFAULT_BARS_PER_YEAR = 252.0  # overridable for intraday data, matches app.quant_lab convention


class ResearchError(Exception):
    """Raised when a view cannot be produced from the data given."""


@dataclass
class AssetView:
    asset: str
    mu: float                 # annualized expected return
    sigma: float               # annualized standard deviation OF THE VIEW ITSELF (Omega input, not asset volatility)
    method: str                 # "bootstrap" | "strategy_signal"
    n_samples: int
    signal_direction: int = 0   # -1/0/1, only meaningful for strategy_signal views
    detail: str = ""            # one-line human-readable explanation, surfaced in the journal


@dataclass
class EnsembleForecastConfig:
    lookback_bars: int = 400       # matches the article's "last 400 daily bars" cache pull
    horizon_bars: int = 5           # matches the article's weekly (5-daily-bar) rebalance horizon
    n_samples: int = 32              # matches the article's 32 sampled Kronos futures
    bars_per_year: float = DEFAULT_BARS_PER_YEAR
    seed: int | None = None


def _log_returns(closes: pd.Series) -> np.ndarray:
    closes = closes.dropna()
    if len(closes) < 2:
        return np.array([])
    return np.log(closes.to_numpy()[1:] / closes.to_numpy()[:-1])


def bootstrap_view(df: pd.DataFrame, asset: str, config: EnsembleForecastConfig) -> AssetView:
    """Block-free bootstrap of the asset's own recent bar returns.

    Not a forecast of direction -- a calibrated measure of how much
    dispersion `horizon_bars` ahead genuinely contains, given nothing
    more than the asset's own recent volatility. Feeding this into
    Black-Litterman with mu close to zero and sigma reflecting real
    historical dispersion is what lets the optimizer safely fall back to
    market-equilibrium weights on an asset nobody has an edge on, rather
    than the optimizer inventing a confident view out of noise.
    """
    window = df["close"].tail(config.lookback_bars)
    returns = _log_returns(window)
    if len(returns) < 20:
        raise ResearchError(
            f"'{asset}': only {len(returns)} usable bars in the lookback window -- need at least 20 "
            f"to bootstrap a view."
        )

    rng = np.random.default_rng(config.seed)
    path_avg_returns = np.empty(config.n_samples)
    for i in range(config.n_samples):
        sampled = rng.choice(returns, size=config.horizon_bars, replace=True)
        path_avg_returns[i] = sampled.mean()  # average per-bar log return for this sampled path

    mu = float(path_avg_returns.mean()) * config.bars_per_year
    sigma = float(path_avg_returns.std(ddof=1)) * config.bars_per_year if config.n_samples > 1 else float(returns.std(ddof=1)) * config.bars_per_year
    return AssetView(
        asset=asset, mu=mu, sigma=max(sigma, 1e-6), method="bootstrap", n_samples=config.n_samples,
        detail=f"bootstrap of last {len(returns)} bars, {config.n_samples} paths x {config.horizon_bars} bars",
    )


def strategy_signal_view(
    df: pd.DataFrame, asset: str, strategy: Strategy, config: EnsembleForecastConfig,
) -> AssetView:
    """Turns an existing Strategy's own recent behavior into a view.

    Deliberately simple and honestly scoped: this is NOT a proper
    walk-forward predictive backtest (see app.backtest.engine for that).
    It runs the strategy once on the lookback window, reads whatever
    signal it holds on the LAST bar, and measures how that same signal
    direction has actually paid off historically inside this same
    window -- i.e. "if this strategy's current call has looked like this
    before, what happened next." Treat the resulting mu as a rough,
    directional prior, not a calibrated forecast; the sigma this produces
    (from the SPREAD of those historical forward returns, not a single
    number) is what keeps Black-Litterman honest about how much to trust
    it.
    """
    window = df.tail(config.lookback_bars).reset_index(drop=True)
    if len(window) < config.horizon_bars + 20:
        raise ResearchError(
            f"'{asset}': lookback window too short ({len(window)} bars) for a "
            f"{config.horizon_bars}-bar horizon strategy-signal view."
        )
    try:
        result = strategy.generate(window)
    except StrategyError as exc:
        raise ResearchError(f"'{asset}': strategy failed to generate signals: {exc}") from exc

    signals = result.signals.to_numpy()
    closes = window["close"].to_numpy()
    current_direction = int(np.sign(signals[-1])) if len(signals) else 0

    if current_direction == 0:
        return AssetView(
            asset=asset, mu=0.0, sigma=1.0, method="strategy_signal", n_samples=0, signal_direction=0,
            detail=f"'{strategy.source_type}' strategy is flat on the last bar -- no view, defers to market equilibrium",
        )

    # Forward `horizon_bars`-ahead returns following every historical bar
    # where the strategy held the SAME direction as it does right now.
    forward_returns = []
    for i in range(len(signals) - config.horizon_bars):
        if int(np.sign(signals[i])) == current_direction and closes[i] > 0:
            fwd = np.log(closes[i + config.horizon_bars] / closes[i])
            forward_returns.append(fwd if current_direction > 0 else -fwd)

    if len(forward_returns) < 5:
        return AssetView(
            asset=asset, mu=0.0, sigma=1.0, method="strategy_signal", n_samples=len(forward_returns),
            signal_direction=current_direction,
            detail=f"'{strategy.source_type}' signal is {['flat','long','short'][current_direction if current_direction>0 else 2]} "
                   f"but only {len(forward_returns)} historical same-direction instances -- too few to trust, defers to equilibrium",
        )

    forward_returns = np.array(forward_returns)
    per_bar_avg = forward_returns / config.horizon_bars
    mu = float(per_bar_avg.mean()) * config.bars_per_year
    sigma = float(per_bar_avg.std(ddof=1)) * config.bars_per_year
    hit_rate = float((forward_returns > 0).mean())
    direction_word = "long" if current_direction > 0 else "short"
    return AssetView(
        asset=asset, mu=mu, sigma=max(sigma, 1e-6), method="strategy_signal", n_samples=len(forward_returns),
        signal_direction=current_direction,
        detail=(f"'{strategy.source_type}' strategy currently {direction_word}; {len(forward_returns)} historical "
                f"same-direction instances, {hit_rate:.0%} forward hit rate over {config.horizon_bars} bars"),
    )


def generate_views(
    price_data: dict[str, pd.DataFrame],
    config: EnsembleForecastConfig,
    strategies: dict[str, Strategy] | None = None,
) -> dict[str, AssetView]:
    """One call per rebalance cycle: `price_data` is {asset: OHLCV df,
    ALREADY TRUNCATED to the as-of cutoff for this cycle} -- callers
    (rebalancer.py's walk-forward loop) are responsible for not leaking
    future bars in. Assets present in `strategies` use
    `strategy_signal_view`; everything else falls back to `bootstrap_view`.
    """
    strategies = strategies or {}
    views: dict[str, AssetView] = {}
    for asset, df in price_data.items():
        if asset in strategies:
            try:
                views[asset] = strategy_signal_view(df, asset, strategies[asset], config)
                continue
            except ResearchError:
                pass  # fall through to bootstrap rather than dropping the asset from the book
        views[asset] = bootstrap_view(df, asset, config)
    return views
