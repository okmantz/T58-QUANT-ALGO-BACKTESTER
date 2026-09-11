"""
Portfolio desk -- turns research-desk AssetViews into position weights.

This is the seam the article credits to skfolio: "BlackLitterman accepts
views as plain strings... exactly the shape a forecasting model can
emit." We don't take a new dependency on skfolio itself -- this app's own
app.quant_lab.portfolio_optimizer already implements closed-form
Markowitz (min-variance, max-Sharpe, efficient frontier) from scratch, in
this app's own style (closed-form first, scipy SLSQP as an optional
upgrade for constraints, honestly-labeled fallback if scipy is missing).
This module reuses that machinery for the covariance estimate and the
final weight solve, and adds exactly the piece it doesn't have: the
Black-Litterman blend of views with market equilibrium.

The formula (identical to the one in the source article):

    mu_post = pi + tau*Sigma*P'(P*tau*Sigma*P' + Omega)^-1 (Q - P*pi)

Omega (the variance assigned to each view) is THE most important number
in this whole module -- send it to zero and the optimizer takes the
forecast literally; send it to infinity and the posterior collapses back
to market equilibrium. `confidence_to_omega` and `confidence_sweep` below
exist specifically to make that dial visible and testable, matching the
article's own "safety valve" chart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.hedge_fund.research import AssetView
from app.quant_lab.portfolio_optimizer import (
    OptimizationInputs,
    PortfolioAllocation,
    PortfolioOptimizerError,
    max_sharpe_portfolio,
)

DEFAULT_RISK_AVERSION = 2.5   # a conventional textbook default for the reverse-optimization step
DEFAULT_TAU = 0.05             # standard "small number" scaling uncertainty of the PRIOR (not the views)


class BlackLittermanError(Exception):
    """Raised when a posterior/weight solve cannot proceed."""


@dataclass
class RebalanceSolveResult:
    allocation: PortfolioAllocation
    posterior_mu: dict[str, float]
    prior_mu: dict[str, float]
    warnings: list[str]


def market_implied_returns(
    cov_matrix: np.ndarray, market_weights: np.ndarray, risk_aversion: float = DEFAULT_RISK_AVERSION,
) -> np.ndarray:
    """Reverse optimization: pi = risk_aversion * Sigma * w_mkt.

    This app has no market-cap data source, so `market_weights` defaults
    (see `equal_weight_market_proxy`) to an EQUAL-weight proxy rather than
    true float-adjusted market-cap weights -- a documented approximation,
    same spirit as this app's other "optional in spirit" fallbacks. For a
    universe of correlated, similarly-sized instruments (the common case
    for a prop-account book) this is a reasonable stand-in; for a universe
    spanning wildly different asset classes it will understate how much
    the market actually leans on the biggest names.
    """
    return risk_aversion * (cov_matrix @ market_weights)


def equal_weight_market_proxy(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def confidence_to_omega(view_sigma: float, confidence: float) -> float:
    """Maps a 0..1 confidence dial to the view's Omega (variance term).

    confidence=1.0 -> Omega = view_sigma**2 (trust the research desk's OWN
        measured dispersion exactly -- not zero. A view is never asserted
        with perfect certainty here; its floor is however much dispersion
        the research desk actually observed across its samples).
    confidence=0.0 -> Omega -> a very large number (the view is
        effectively ignored and the posterior falls back to market
        equilibrium), matching the article's sweep exactly.
    """
    if not 0.0 <= confidence <= 1.0:
        raise BlackLittermanError("confidence must be between 0.0 and 1.0.")
    floor_var = max(view_sigma, 1e-6) ** 2
    if confidence <= 1e-6:
        return floor_var * 1e8
    return floor_var / confidence


def black_litterman_posterior(
    pi: np.ndarray, cov_matrix: np.ndarray, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray, tau: float = DEFAULT_TAU,
) -> np.ndarray:
    """Returns posterior_mu only (not posterior covariance) -- this app's
    weight solve uses the ORIGINAL sample covariance for risk, which is
    the standard simplification most practitioner implementations use
    (the full posterior-covariance update is a second-order correction;
    skipping it does not change which asset a view favors, only slightly
    understates post-view uncertainty)."""
    tau_sigma = tau * cov_matrix
    middle = P @ tau_sigma @ P.T + Omega
    try:
        middle_inv = np.linalg.inv(middle)
    except np.linalg.LinAlgError as exc:
        raise BlackLittermanError("Black-Litterman posterior is singular for these views/covariance.") from exc
    adjustment = tau_sigma @ P.T @ middle_inv @ (Q - P @ pi)
    return pi + adjustment


def views_to_posterior(
    inputs: OptimizationInputs,
    views: dict[str, AssetView],
    confidence: float,
    market_weights: dict[str, float] | None = None,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    tau: float = DEFAULT_TAU,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds pi, P, Q, Omega from the research desk's views and returns
    (prior_mu, posterior_mu), both aligned to inputs.tickers order."""
    n = inputs.n_assets
    if market_weights is None:
        w_mkt = equal_weight_market_proxy(n)
    else:
        w_mkt = np.array([market_weights.get(t, 0.0) for t in inputs.tickers])
        total = w_mkt.sum()
        if total <= 0:
            raise BlackLittermanError("market_weights sum to zero or less.")
        w_mkt = w_mkt / total

    pi = market_implied_returns(inputs.cov_matrix, w_mkt, risk_aversion)

    view_assets = [t for t in inputs.tickers if t in views]
    if not view_assets:
        return pi, pi.copy()

    k = len(view_assets)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)
    for row, asset in enumerate(view_assets):
        col = inputs.tickers.index(asset)
        P[row, col] = 1.0
        Q[row] = views[asset].mu
        omega_diag[row] = confidence_to_omega(views[asset].sigma, confidence)
    Omega = np.diag(omega_diag)

    posterior_mu = black_litterman_posterior(pi, inputs.cov_matrix, P, Q, Omega, tau)
    return pi, posterior_mu


def _solve_with_turnover_and_costs(
    inputs: OptimizationInputs, posterior_mu: np.ndarray, long_only: bool,
    previous_weights: np.ndarray | None, max_turnover: float | None, transaction_cost_bps: float,
    risk_aversion: float,
) -> tuple[np.ndarray, list[str], bool]:
    """Mean-variance utility maximization with turnover cap and
    transaction costs INSIDE the objective -- matching the article's
    explicit point that "the difference between a paper edge and a real
    one usually dies exactly [in costs discovered after the fact]."
    Requires scipy; caller falls back to the closed-form no-turnover
    solve (see `solve_posterior_weights`) if scipy isn't installed, the
    same "optional in spirit" pattern as app.quant_lab.portfolio_optimizer's
    own long-only solve.
    """
    from scipy.optimize import minimize

    n = inputs.n_assets
    prev = previous_weights if previous_weights is not None else np.zeros(n)
    cost_rate = transaction_cost_bps / 10_000.0

    def objective(w):
        turnover_cost = cost_rate * np.sum(np.abs(w - prev))
        utility = w @ posterior_mu - 0.5 * risk_aversion * (w @ inputs.cov_matrix @ w) - turnover_cost
        return -utility

    x0 = prev.copy() if previous_weights is not None else np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if max_turnover is not None:
        constraints.append({"type": "ineq", "fun": lambda w: max_turnover - np.sum(np.abs(w - prev))})

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-12})
    warnings: list[str] = []
    converged = result.success and abs(float(np.sum(result.x)) - 1.0) < 1e-4
    if not converged:
        warnings.append(f"Turnover/cost-constrained solve did not converge cleanly ({result.message}).")
    return result.x, warnings, converged


def solve_posterior_weights(
    inputs: OptimizationInputs,
    posterior_mu: np.ndarray,
    long_only: bool = True,
    previous_weights: dict[str, float] | None = None,
    max_turnover: float | None = None,
    transaction_cost_bps: float = 0.0,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    risk_free_rate: float = 0.0,
) -> tuple[PortfolioAllocation, list[str]]:
    """Solves for weights given the BL posterior mu. Fast path (no
    turnover cap, no transaction costs): reuses
    app.quant_lab.portfolio_optimizer.max_sharpe_portfolio directly on an
    OptimizationInputs built from the posterior mu -- exact same
    closed-form tangency-portfolio math this app already ships and tests
    elsewhere, just fed the BL-adjusted return vector instead of the raw
    historical mean. Long-only in the fast path is approximated by
    clip-and-renormalize (documented, same as the underlying function's
    own fallback) -- use a turnover cap or cost > 0 to route through the
    proper constrained solve below instead.
    """
    warnings: list[str] = []
    posterior_inputs = OptimizationInputs(tickers=inputs.tickers, mean_returns=posterior_mu, cov_matrix=inputs.cov_matrix)
    prev_arr = None
    if previous_weights is not None:
        prev_arr = np.array([previous_weights.get(t, 0.0) for t in inputs.tickers])

    # A turnover CAP only makes sense against an existing book -- with no
    # prior position (a fresh start, previous_weights is None/all-cash),
    # sum(|w - 0|) == sum(w) == 1 for ANY fully-invested long-only
    # allocation, so a cap below 1.0 combined with the sum-to-1 equality
    # constraint has NO feasible solution at all (this is what produced
    # the "positive directional derivative" non-convergence originally).
    # The correct fix is to not apply the cap to money going in for the
    # first time, not to relax the sum-to-1 constraint -- the cap exists
    # to bound CHURN of an existing book, and there is no churn yet.
    effective_max_turnover = max_turnover
    if max_turnover is not None and (prev_arr is None or float(np.sum(prev_arr)) <= 1e-9):
        effective_max_turnover = None

    needs_constrained_solve = (effective_max_turnover is not None) or (transaction_cost_bps > 0)
    if needs_constrained_solve:
        try:
            weights, solve_warnings, converged = _solve_with_turnover_and_costs(
                posterior_inputs, posterior_mu, long_only, prev_arr, effective_max_turnover, transaction_cost_bps, risk_aversion,
            )
            warnings.extend(solve_warnings)
            if not converged:
                # A non-converged point from a TURNOVER-constrained solve
                # can violate the very turnover cap it was asked to
                # respect (renormalizing it back to sum-to-1 does not fix
                # that -- it can push turnover higher, not lower). The
                # only allocation guaranteed not to breach the cap is
                # "don't trade this cycle": hold whatever was already
                # held (turnover exactly 0), or fall back to equal-weight
                # if this is the very first cycle and there's nothing to
                # hold. This favors an honest no-op over a broken order.
                if prev_arr is not None and float(np.sum(prev_arr)) > 1e-9:
                    weights = prev_arr.copy()
                    warnings.append("Held previous weights this cycle rather than act on a non-converged solve.")
                else:
                    # No real previous book to hold (either the first
                    # cycle, or `previous_weights` was itself all-cash) --
                    # "hold previous" would mean holding nothing, which
                    # isn't a safer choice than just picking a starting
                    # point. Equal-weight is the same equilibrium
                    # fallback Black-Litterman itself collapses to at
                    # zero confidence (see confidence_to_omega).
                    weights = np.full(inputs.n_assets, 1.0 / inputs.n_assets)
                    warnings.append("No real previous book to fall back to -- used equal-weight instead of a non-converged solve.")
        except ImportError:
            warnings.append(
                "scipy isn't installed -- turnover cap and transaction costs were IGNORED for this cycle "
                "(closed-form fallback below has no way to enforce them). `pip install scipy` to fix."
            )
            allocation = max_sharpe_portfolio(posterior_inputs, risk_free_rate)
            if long_only:
                clipped = np.clip(np.array(list(allocation.weights.values())), 0.0, None)
                clipped = clipped / clipped.sum() if clipped.sum() > 0 else np.full(inputs.n_assets, 1.0 / inputs.n_assets)
                weights = clipped
            else:
                weights = np.array(list(allocation.weights.values()))
    else:
        try:
            allocation = max_sharpe_portfolio(posterior_inputs, risk_free_rate)
            weights = np.array(list(allocation.weights.values()))
            if long_only and np.any(weights < 0):
                weights = np.clip(weights, 0.0, None)
                total = weights.sum()
                if total <= 0:
                    raise PortfolioOptimizerError("Every posterior weight was negative -- no long-only allocation exists.")
                weights = weights / total
                warnings.append("Unconstrained tangency portfolio held short positions -- clipped to long-only and renormalized (approximation; pass max_turnover or transaction_cost_bps>0 for a proper constrained solve).")
        except PortfolioOptimizerError as exc:
            raise BlackLittermanError(str(exc)) from exc

    # SLSQP occasionally returns an infeasible point when it can't find a
    # descent direction (observed message: "Positive directional
    # derivative for linesearch") -- it still reports weights, just ones
    # that don't sum to 1. Rather than hand a broken allocation to the
    # execution desk (which would silently size the WHOLE BOOK at half
    # intended notional, or worse), renormalize and say so loudly. This
    # is a safety net, not a fix for the underlying non-convergence --
    # a cycle that needed it is worth a second look if it happens often.
    total_weight = float(np.sum(weights))
    if abs(total_weight - 1.0) > 1e-6:
        warnings.append(
            f"Solver returned weights summing to {total_weight:.4f} instead of 1.0 (non-convergence) -- "
            f"renormalized. If this recurs often, loosen max_turnover or check for a degenerate posterior."
        )
        if long_only:
            weights = np.clip(weights, 0.0, None)
            total_weight = float(np.sum(weights))
        weights = weights / total_weight if total_weight > 1e-9 else np.full(inputs.n_assets, 1.0 / inputs.n_assets)

    ret = float(weights @ inputs.mean_returns)
    vol = float(np.sqrt(weights @ inputs.cov_matrix @ weights))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else None
    final_allocation = PortfolioAllocation(
        weights=dict(zip(inputs.tickers, (float(w) for w in weights))),
        expected_return=ret, volatility=vol, sharpe_ratio=sharpe,
    )
    return final_allocation, warnings


def confidence_sweep(
    inputs: OptimizationInputs, views: dict[str, AssetView], confidences: list[float] | None = None,
    market_weights: dict[str, float] | None = None, long_only: bool = True,
) -> list[dict]:
    """Reproduces the article's own diagnostic chart: sweep the
    confidence dial from confident to useless and show the resulting
    weights collapse toward the equilibrium (equal-weight, absent
    market-cap data) portfolio. Returns a list of
    {confidence, weights: {asset: weight}} for direct use in a UI chart.
    """
    confidences = confidences if confidences is not None else [1.0, 0.75, 0.5, 0.25, 0.1, 0.01]
    out = []
    for c in confidences:
        _, posterior_mu = views_to_posterior(inputs, views, c, market_weights)
        allocation, _ = solve_posterior_weights(inputs, posterior_mu, long_only=long_only)
        out.append({"confidence": c, "weights": allocation.weights})
    return out
