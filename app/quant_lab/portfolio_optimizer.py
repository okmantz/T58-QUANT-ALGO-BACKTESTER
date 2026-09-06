"""
Markowitz Mean-Variance Portfolio Optimizer -- built from scratch: given a
set of tickers, fetch historical returns (via the Alpaca connector already
in this app), estimate their annualized mean return vector and covariance
matrix, and solve for portfolio weights along the efficient frontier.

Three solutions, all closed-form linear algebra (Lagrange multipliers on
the quadratic "minimize w'*Sigma*w subject to w'*1=1 [and w'*mu=target]"
problem) -- no external optimization library needed for the UNCONSTRAINED
(short-selling allowed) case, which is the textbook Markowitz result:

  - `min_variance_portfolio`: the single lowest-volatility portfolio.
  - `max_sharpe_portfolio`: the tangency portfolio (highest reward per
    unit of risk, given a risk-free rate).
  - `efficient_frontier`: the full curve of minimum-variance portfolios
    across a range of target returns.

`optimize_for_risk_level(risk_level)` is the practical "I want an
allocation for MY risk tolerance" entry point: risk_level is a 0.0-1.0
dial that interpolates between the min-variance portfolio's own
volatility (risk_level=0) and a configurable maximum volatility
(risk_level=1, default: the single most volatile asset's own volatility),
then finds the efficient-frontier portfolio at that target volatility.

Long-only (no short positions) is a real inequality-constrained quadratic
program, which the closed-form Lagrange solution above does not enforce.
When `long_only=True`, this module uses `scipy.optimize.minimize` (SLSQP)
when scipy is installed (optional dependency, same "optional in spirit"
pattern as pyarrow/py7zr/pypdf elsewhere in this app's requirements.txt);
if scipy isn't installed, it falls back to the unconstrained closed-form
weights with negative positions clipped to zero and the remainder
renormalized to sum to 1 -- a documented APPROXIMATION, not a true
long-only optimum, since clip-and-renormalize does not re-solve for the
new (constrained) optimal allocation among the remaining assets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.data.alpaca_source import AlpacaFetchError, fetch_stock_bars

TRADING_DAYS_PER_YEAR = 252


class PortfolioOptimizerError(Exception):
    """Raised for invalid inputs (too few assets, a singular covariance
    matrix, a target return/volatility outside what's achievable)."""


@dataclass
class OptimizationInputs:
    tickers: list
    mean_returns: np.ndarray     # annualized, shape (n,)
    cov_matrix: np.ndarray       # annualized, shape (n, n)

    @property
    def n_assets(self) -> int:
        return len(self.tickers)


@dataclass
class PortfolioAllocation:
    weights: dict                 # {ticker: weight}
    expected_return: float         # annualized
    volatility: float               # annualized standard deviation
    sharpe_ratio: float | None      # None if no risk-free rate was supplied

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def render_summary(self) -> str:
        lines = [f"Expected return: {self.expected_return:.2%}   Volatility: {self.volatility:.2%}"]
        if self.sharpe_ratio is not None:
            lines.append(f"Sharpe ratio: {self.sharpe_ratio:.3f}")
        lines.append("Weights:")
        for ticker, w in sorted(self.weights.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {ticker:<8} {w:+.2%}")
        return "\n".join(lines)


def build_inputs_from_prices(price_data: dict[str, pd.DataFrame]) -> OptimizationInputs:
    """price_data: {ticker: OHLCV DataFrame}. Uses daily close-to-close
    log returns, aligned on overlapping dates only."""
    if len(price_data) < 2:
        raise PortfolioOptimizerError("Need at least 2 tickers to build a covariance matrix.")
    closes = {}
    for ticker, df in price_data.items():
        s = df[["timestamp", "close"]].copy()
        s["timestamp"] = pd.to_datetime(s["timestamp"])
        closes[ticker] = s.set_index("timestamp")["close"]
    prices = pd.concat(closes, axis=1).dropna()
    if len(prices) < 30:
        raise PortfolioOptimizerError(
            f"Only {len(prices)} overlapping trading days across all tickers -- need at least 30."
        )
    log_returns = np.log(prices / prices.shift(1)).dropna()
    tickers = list(price_data.keys())
    mean_returns = log_returns[tickers].mean().to_numpy() * TRADING_DAYS_PER_YEAR
    cov_matrix = log_returns[tickers].cov().to_numpy() * TRADING_DAYS_PER_YEAR
    return OptimizationInputs(tickers=tickers, mean_returns=mean_returns, cov_matrix=cov_matrix)


def fetch_and_build_inputs(
    api_key: str, secret_key: str, tickers: list[str], start: str, end: str,
    timeframe_label: str = "1Day", feed: str = "iex",
) -> OptimizationInputs:
    price_data = {}
    skipped = {}
    for ticker in tickers:
        try:
            price_data[ticker] = fetch_stock_bars(api_key, secret_key, ticker, timeframe_label, start, end, feed=feed)
        except AlpacaFetchError as exc:
            skipped[ticker] = str(exc)
    if skipped:
        raise PortfolioOptimizerError(f"Failed to fetch data for: {skipped}")
    return build_inputs_from_prices(price_data)


def _portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float | None) -> tuple[float, float, float | None]:
    ret = float(weights @ mean_returns)
    vol = float(np.sqrt(weights @ cov_matrix @ weights))
    sharpe = ((ret - risk_free_rate) / vol) if (risk_free_rate is not None and vol > 0) else None
    return ret, vol, sharpe


def _to_allocation(weights: np.ndarray, inputs: OptimizationInputs, risk_free_rate: float | None) -> PortfolioAllocation:
    ret, vol, sharpe = _portfolio_stats(weights, inputs.mean_returns, inputs.cov_matrix, risk_free_rate)
    return PortfolioAllocation(
        weights=dict(zip(inputs.tickers, (float(w) for w in weights))),
        expected_return=ret, volatility=vol, sharpe_ratio=sharpe,
    )


def _inv_cov(cov_matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError as exc:
        raise PortfolioOptimizerError(
            "Covariance matrix is singular (e.g. two tickers with identical or perfectly "
            "collinear returns) -- cannot solve the unconstrained optimizer."
        ) from exc


def min_variance_portfolio(inputs: OptimizationInputs, risk_free_rate: float | None = None) -> PortfolioAllocation:
    """Closed form: w = Sigma^-1 * 1 / (1' * Sigma^-1 * 1)."""
    ones = np.ones(inputs.n_assets)
    inv_cov = _inv_cov(inputs.cov_matrix)
    raw = inv_cov @ ones
    weights = raw / (ones @ raw)
    return _to_allocation(weights, inputs, risk_free_rate)


def max_sharpe_portfolio(inputs: OptimizationInputs, risk_free_rate: float = 0.0) -> PortfolioAllocation:
    """Closed form (unconstrained tangency portfolio):
    w ∝ Sigma^-1 * (mu - rf * 1), normalized to sum to 1."""
    ones = np.ones(inputs.n_assets)
    inv_cov = _inv_cov(inputs.cov_matrix)
    excess = inputs.mean_returns - risk_free_rate * ones
    raw = inv_cov @ excess
    denom = ones @ raw
    if abs(denom) < 1e-12:
        raise PortfolioOptimizerError("Tangency portfolio is undefined for these inputs (denominator ~= 0).")
    weights = raw / denom
    return _to_allocation(weights, inputs, risk_free_rate)


def _min_variance_weights_for_target_return(inputs: OptimizationInputs, target_return: float) -> np.ndarray:
    """Closed-form two-constraint Lagrangian solution: minimize w'*Sigma*w
    subject to w'*1=1 and w'*mu=target_return. Standard Markowitz result
    via the 2x2 system built from A = 1'*Sigma^-1*1, B = 1'*Sigma^-1*mu,
    C = mu'*Sigma^-1*mu."""
    ones = np.ones(inputs.n_assets)
    inv_cov = _inv_cov(inputs.cov_matrix)
    mu = inputs.mean_returns

    A = float(ones @ inv_cov @ ones)
    B = float(ones @ inv_cov @ mu)
    C = float(mu @ inv_cov @ mu)
    D = A * C - B * B
    if abs(D) < 1e-12:
        raise PortfolioOptimizerError("Efficient frontier is degenerate for these inputs (D ~= 0).")

    lam = (C - B * target_return) / D
    gam = (A * target_return - B) / D
    weights = inv_cov @ (lam * ones + gam * mu)
    return weights


def _long_only_weights_for_target_return(inputs: OptimizationInputs, target_return: float) -> np.ndarray:
    try:
        from scipy.optimize import minimize
    except ImportError:
        weights = _min_variance_weights_for_target_return(inputs, target_return)
        clipped = np.clip(weights, 0.0, None)
        total = clipped.sum()
        if total <= 0:
            raise PortfolioOptimizerError(
                "Long-only fallback failed: every closed-form weight was negative for this target "
                "return. Install scipy for a proper constrained solve (`pip install scipy`)."
            )
        return clipped / total

    n = inputs.n_assets
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ inputs.mean_returns) - target_return},
    ]
    result = minimize(
        lambda w: w @ inputs.cov_matrix @ w, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise PortfolioOptimizerError(f"Long-only optimizer did not converge: {result.message}")
    return result.x


def efficient_frontier(
    inputs: OptimizationInputs, n_points: int = 25, risk_free_rate: float | None = None, long_only: bool = False,
) -> list:
    """Returns `n_points` PortfolioAllocations spanning the TRUE efficient
    frontier: from the min-variance portfolio's own return up to the
    single highest-returning asset's return, each the minimum-variance
    portfolio for that target return. Target returns below the
    min-variance portfolio's own return are excluded on purpose -- the
    minimum-variance solution for a return below that point sits on the
    parabola's INEFFICIENT lower branch (same volatility achievable for a
    higher return exists), which is conventionally not called part of
    "the efficient frontier" at all.
    """
    min_var = min_variance_portfolio(inputs, risk_free_rate)
    lo = max(min_var.expected_return, float(inputs.mean_returns.min()))
    hi = float(inputs.mean_returns.max())
    if hi <= lo:
        raise PortfolioOptimizerError("All assets have the same expected return -- no frontier to trace.")
    targets = np.linspace(lo, hi, n_points)
    out = []
    for target in targets:
        try:
            weights = (_long_only_weights_for_target_return(inputs, target) if long_only
                       else _min_variance_weights_for_target_return(inputs, target))
        except PortfolioOptimizerError:
            continue
        out.append(_to_allocation(weights, inputs, risk_free_rate))
    if not out:
        raise PortfolioOptimizerError("Could not compute any point on the efficient frontier for these inputs.")
    return out


def optimize_for_risk_level(
    inputs: OptimizationInputs, risk_level: float, risk_free_rate: float | None = None,
    long_only: bool = False, max_volatility: float | None = None,
) -> PortfolioAllocation:
    """risk_level: 0.0 (as conservative as possible -- the min-variance
    portfolio) to 1.0 (as aggressive as this universe allows -- the
    target volatility given by `max_volatility`, default: the single most
    volatile individual asset's own volatility). Values in between
    linearly interpolate the TARGET VOLATILITY, then solve the efficient
    frontier for the allocation that achieves it."""
    if not 0.0 <= risk_level <= 1.0:
        raise PortfolioOptimizerError("risk_level must be between 0.0 and 1.0.")

    min_var = min_variance_portfolio(inputs, risk_free_rate)
    if max_volatility is None:
        max_volatility = float(np.sqrt(np.diag(inputs.cov_matrix)).max())
    if max_volatility <= min_var.volatility:
        return min_var

    target_vol = min_var.volatility + risk_level * (max_volatility - min_var.volatility)

    # Binary-search the target RETURN (monotonic with frontier volatility
    # above the min-variance point) that produces `target_vol`, since the
    # closed-form solver is parameterized by return, not volatility.
    lo_ret, hi_ret = min_var.expected_return, float(inputs.mean_returns.max())
    best = min_var
    for _ in range(40):
        mid_ret = (lo_ret + hi_ret) / 2.0
        try:
            weights = (_long_only_weights_for_target_return(inputs, mid_ret) if long_only
                       else _min_variance_weights_for_target_return(inputs, mid_ret))
        except PortfolioOptimizerError:
            break
        candidate = _to_allocation(weights, inputs, risk_free_rate)
        best = candidate
        if abs(candidate.volatility - target_vol) < 1e-5:
            break
        if candidate.volatility < target_vol:
            lo_ret = mid_ret
        else:
            hi_ret = mid_ret
    return best
