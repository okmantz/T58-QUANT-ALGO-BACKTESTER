"""
Pairs Trading (Statistical Arbitrage) -- the classic stat-arb recipe:
find two historically correlated instruments, watch the spread (here, the
price RATIO) between them for unusual divergence, and bet on it reverting
back to its historical mean.

This app's engine (app.backtest.engine.run_backtest) is single-instrument
by design (see app.portfolio.portfolio's and app.data.pairs's module
docstrings for the same architectural note). Rather than rewriting it,
this module uses the machinery this repo ALREADY has for exactly this
purpose: app.data.pairs.merge_pair_series() folds the second instrument's
close price into the first instrument's DataFrame as an ordinary extra
column, and app.strategy.indicators.pair_zscore() (already wired into
the Manual Strategy Builder's "pair_zscore" condition operand) computes
the rolling z-score of the two-instrument price ratio -- the standard
stat-arb entry/exit trigger. What this module adds on top:

  1. A SCREENER (`screen_pairs`) that pulls real historical bars for a
     universe of symbols via the Alpaca connector already in this app
     (app.data.alpaca_source), computes every pair's return correlation,
     and -- for pairs that clear a minimum correlation -- a simple,
     from-scratch Dickey-Fuller stationarity check on the price-ratio
     spread (does the spread actually mean-revert, or does the ratio
     just wander?), ranking candidates so you don't have to eyeball a
     correlation matrix by hand.
  2. A STRATEGY BUILDER (`build_pairs_strategy_config`) that turns a
     chosen pair + z-score thresholds into a ready-to-run Manual Strategy
     Builder config using pair_zscore, so it's immediately compatible
     with everything else in this app (Iterative Refinement, Evolution
     Lab, the Strategy Library, etc.) once saved.
  3. An end-to-end `run_pairs_backtest` that fetches both instruments via
     Alpaca, merges them, and runs the real backtest engine, so "find a
     pair, then see if trading the spread would have worked" is one
     function call.

Honest limitation: because the underlying engine is single-instrument,
this backtests trading instrument A ALONE, triggered by the two-
instrument ratio -- it is a directional bet on reversion, not a true
market-neutral dollar-matched long/short book (which would need
simultaneous, independently-sized positions in BOTH legs, and this app's
engine doesn't model two simultaneous open positions on one signal
series). This is the same honest scoping app.data.pairs's own docstring
already establishes; it is a legitimate and common simplified backtest of
the reversion signal itself, not a claim of modeling real dollar-neutral
P&L on both legs.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtest.engine import BacktestResult, run_backtest
from app.backtest.risk import RiskConfig
from app.data.alpaca_source import AlpacaFetchError, fetch_stock_bars
from app.data.pairs import DEFAULT_PAIR_COLUMN, merge_pair_series
from app.strategy.manual import ManualStrategy

# Approximate Dickey-Fuller critical values for a regression with a
# constant, no trend term, no augmentation lags (MacKinnon-style
# large-sample approximations, commonly cited in introductory stat-arb
# material). These are NOT exact finite-sample critical values -- treat
# `stationarity_verdict` as a useful heuristic bucket, not a rigorous
# hypothesis-test p-value.
_ADF_CRITICAL_VALUES = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


class PairsTradingError(Exception):
    """Raised when a screen/backtest cannot proceed (too few symbols,
    an Alpaca fetch failure, misaligned data)."""


@dataclass
class PairCandidate:
    symbol_a: str
    symbol_b: str
    correlation: float          # Pearson correlation of daily returns
    hedge_ratio: float          # OLS slope: close_a ~ hedge_ratio * close_b (+ intercept)
    adf_statistic: float        # Dickey-Fuller test statistic on the price ratio (more negative = more mean-reverting)
    stationarity_verdict: str   # "likely mean-reverting" | "borderline" | "likely a random walk"
    current_zscore: float       # the ratio's z-score as of the last bar in the data used

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _adf_statistic(series: pd.Series) -> float:
    """A basic (non-augmented) Dickey-Fuller test statistic for the
    regression: delta_y_t = alpha + beta * y_(t-1) + epsilon_t. beta's own
    t-statistic (computed via plain OLS, from scratch -- no statsmodels)
    is the ADF stat: the more negative, the more confidently y_(t-1) has
    been pulling the series back toward its mean rather than the series
    wandering freely."""
    y = series.dropna().to_numpy(dtype=float)
    if len(y) < 30:
        raise PairsTradingError("Need at least 30 overlapping bars to run a stationarity check.")
    y_lag = y[:-1]
    dy = np.diff(y)
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta_hat, residuals_ss, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    fitted = X @ beta_hat
    resid = dy - fitted
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(np.sum(resid ** 2)) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_beta = math.sqrt(sigma2 * xtx_inv[1, 1])
    if se_beta <= 0:
        return 0.0
    return float(beta_hat[1] / se_beta)


def _stationarity_verdict(adf_stat: float) -> str:
    if adf_stat <= _ADF_CRITICAL_VALUES["5%"]:
        return "likely mean-reverting"
    if adf_stat <= _ADF_CRITICAL_VALUES["10%"]:
        return "borderline"
    return "likely a random walk"


def _hedge_ratio_ols(close_a: pd.Series, close_b: pd.Series) -> float:
    """OLS slope of close_a regressed on close_b -- 'how many units of B
    per unit of A' to make the spread level-comparable. Plain closed-form
    simple linear regression, from scratch."""
    x = close_b.to_numpy(dtype=float)
    y = close_a.to_numpy(dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        raise PairsTradingError("Cannot compute a hedge ratio: the second instrument's price never moved.")
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _rolling_zscore(ratio: pd.Series, period: int) -> pd.Series:
    mean = ratio.rolling(period, min_periods=period).mean()
    std = ratio.rolling(period, min_periods=period).std(ddof=0)
    return ((ratio - mean) / std.replace(0, np.nan)).fillna(0.0)


def screen_pairs(
    price_data: dict[str, pd.DataFrame],
    min_correlation: float = 0.7,
    zscore_period: int = 50,
) -> list[PairCandidate]:
    """price_data: {symbol: OHLCV DataFrame} for every symbol in your
    universe (fetch these with app.data.alpaca_source.fetch_stock_bars,
    or app.quant_lab.pairs_trading.fetch_universe below). Returns every
    pair clearing `min_correlation`, ranked best-first by how negative
    (more mean-reverting) its ADF statistic is."""
    symbols = list(price_data.keys())
    if len(symbols) < 2:
        raise PairsTradingError("Need at least 2 symbols to screen for pairs.")

    aligned = {}
    for sym, df in price_data.items():
        s = df[["timestamp", "close"]].copy()
        s["timestamp"] = pd.to_datetime(s["timestamp"])
        aligned[sym] = s.set_index("timestamp")["close"]

    candidates: list[PairCandidate] = []
    for sym_a, sym_b in itertools.combinations(symbols, 2):
        merged = pd.concat([aligned[sym_a], aligned[sym_b]], axis=1, keys=[sym_a, sym_b]).dropna()
        if len(merged) < max(zscore_period + 10, 40):
            continue
        returns = merged.pct_change().dropna()
        if len(returns) < 10:
            continue
        corr = float(returns[sym_a].corr(returns[sym_b]))
        if not math.isfinite(corr) or corr < min_correlation:
            continue

        hedge_ratio = _hedge_ratio_ols(merged[sym_a], merged[sym_b])
        ratio = merged[sym_a] / merged[sym_b].replace(0, np.nan)
        adf_stat = _adf_statistic(ratio)
        zscore = _rolling_zscore(ratio, zscore_period)
        candidates.append(PairCandidate(
            symbol_a=sym_a, symbol_b=sym_b, correlation=corr, hedge_ratio=hedge_ratio,
            adf_statistic=adf_stat, stationarity_verdict=_stationarity_verdict(adf_stat),
            current_zscore=float(zscore.iloc[-1]),
        ))

    candidates.sort(key=lambda c: c.adf_statistic)
    return candidates


def fetch_universe(
    api_key: str, secret_key: str, symbols: list[str], timeframe_label: str, start: str, end: str,
    feed: str = "iex",
) -> dict[str, pd.DataFrame]:
    """Fetches OHLCV bars for every symbol in `symbols` via Alpaca. A
    symbol that fails to fetch (bad ticker, no data in range) is skipped
    with its reason returned in `skipped` rather than aborting the whole
    universe fetch."""
    data: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}
    for sym in symbols:
        try:
            data[sym] = fetch_stock_bars(api_key, secret_key, sym, timeframe_label, start, end, feed=feed)
        except AlpacaFetchError as exc:
            skipped[sym] = str(exc)
    if not data:
        raise PairsTradingError(f"Fetched no usable data for any symbol. Failures: {skipped}")
    return data


def build_pairs_strategy_config(
    symbol_a: str, symbol_b: str, zscore_period: int = 50, entry_z: float = 2.0, exit_z: float = 0.5,
    stop_loss_pips: float | None = None, take_profit_pips: float | None = None,
    pair_column: str = DEFAULT_PAIR_COLUMN,
) -> dict:
    """A ready-to-run Manual Strategy Builder config trading the mean-
    reversion of `symbol_a`/`symbol_b`'s price ratio: goes long `symbol_a`
    when the ratio's z-score has stretched below -entry_z (A is unusually
    cheap relative to B -- bet on it reverting up), goes short when it's
    stretched above +entry_z, and exits either side once the z-score has
    reverted back inside +/-exit_z."""
    config = {
        "name": f"Pairs Reversion: {symbol_a} / {symbol_b}",
        "entry_conditions": {
            "long": [{"left": {"type": "pair_zscore", "period": zscore_period, "field": pair_column},
                      "operator": "<", "right": -abs(entry_z)}],
            "short": [{"left": {"type": "pair_zscore", "period": zscore_period, "field": pair_column},
                       "operator": ">", "right": abs(entry_z)}],
        },
        "exit_conditions": {
            "long": [{"left": {"type": "pair_zscore", "period": zscore_period, "field": pair_column},
                      "operator": ">", "right": -abs(exit_z)}],
            "short": [{"left": {"type": "pair_zscore", "period": zscore_period, "field": pair_column},
                       "operator": "<", "right": abs(exit_z)}],
        },
        "risk_management": {},
    }
    if stop_loss_pips is not None:
        config["risk_management"]["stop_type"] = "fixed"
        config["risk_management"]["stop_value"] = stop_loss_pips
    if take_profit_pips is not None:
        config["risk_management"]["target_type"] = "fixed"
        config["risk_management"]["target_value"] = take_profit_pips
    return config


@dataclass
class PairsBacktestResult:
    pair: PairCandidate | None
    strategy_config: dict
    backtest: BacktestResult


def run_pairs_backtest(
    api_key: str, secret_key: str, symbol_a: str, symbol_b: str, timeframe_label: str,
    start: str, end: str, risk: RiskConfig | None = None, zscore_period: int = 50,
    entry_z: float = 2.0, exit_z: float = 0.5, feed: str = "iex",
) -> PairsBacktestResult:
    """End-to-end: fetch both instruments from Alpaca, merge the second
    into the first as `pair_close`, build the pair_zscore reversion
    strategy above, and run it through the real backtest engine."""
    df_a = fetch_stock_bars(api_key, secret_key, symbol_a, timeframe_label, start, end, feed=feed)
    df_b = fetch_stock_bars(api_key, secret_key, symbol_b, timeframe_label, start, end, feed=feed)
    merged_df = merge_pair_series(df_a, df_b)

    pair_info: PairCandidate | None = None
    try:
        aligned = merged_df[["close", DEFAULT_PAIR_COLUMN]].dropna()
        if len(aligned) >= max(zscore_period + 10, 40):
            returns = aligned.pct_change().dropna()
            corr = float(returns["close"].corr(returns[DEFAULT_PAIR_COLUMN]))
            hedge_ratio = _hedge_ratio_ols(aligned["close"], aligned[DEFAULT_PAIR_COLUMN])
            ratio = aligned["close"] / aligned[DEFAULT_PAIR_COLUMN].replace(0, np.nan)
            adf_stat = _adf_statistic(ratio)
            zscore = _rolling_zscore(ratio, zscore_period)
            pair_info = PairCandidate(
                symbol_a=symbol_a, symbol_b=symbol_b, correlation=corr, hedge_ratio=hedge_ratio,
                adf_statistic=adf_stat, stationarity_verdict=_stationarity_verdict(adf_stat),
                current_zscore=float(zscore.iloc[-1]),
            )
    except PairsTradingError:
        pair_info = None

    config = build_pairs_strategy_config(symbol_a, symbol_b, zscore_period, entry_z, exit_z)
    strategy = ManualStrategy(config)
    risk_cfg = risk or RiskConfig()
    bt = run_backtest(merged_df, strategy, risk_cfg)
    return PairsBacktestResult(pair=pair_info, strategy_config=config, backtest=bt)
