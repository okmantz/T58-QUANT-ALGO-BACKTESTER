from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from app.quant_lab.portfolio_optimizer import (
    OptimizationInputs,
    PortfolioOptimizerError,
    build_inputs_from_prices,
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    optimize_for_risk_level,
)


def _mk_price_series(mu, sigma, start, n, seed):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=n, freq="1D")
    r = rng.normal(mu, sigma, n)
    price = start * np.exp(np.cumsum(r))
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price * 1.001, "low": price * 0.999,
        "close": price, "volume": 1000.0,
    })


def _three_asset_inputs(n=600):
    price_data = {
        "LOWVOL": _mk_price_series(0.0004, 0.005, 50, n, seed=1),
        "MIDVOL": _mk_price_series(0.0006, 0.012, 100, n, seed=2),
        "HIVOL": _mk_price_series(0.0009, 0.025, 200, n, seed=3),
    }
    return build_inputs_from_prices(price_data)


def test_build_inputs_requires_at_least_two_tickers():
    with pytest.raises(PortfolioOptimizerError):
        build_inputs_from_prices({"A": _mk_price_series(0.001, 0.01, 100, 100, 1)})


def test_min_variance_weights_sum_to_one():
    inputs = _three_asset_inputs()
    result = min_variance_portfolio(inputs, risk_free_rate=0.02)
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.volatility > 0


def test_min_variance_has_lowest_volatility_on_the_frontier():
    inputs = _three_asset_inputs()
    min_var = min_variance_portfolio(inputs)
    frontier = efficient_frontier(inputs, n_points=15)
    assert all(min_var.volatility <= f.volatility + 1e-6 for f in frontier)


def test_efficient_frontier_volatility_is_monotonically_nondecreasing():
    inputs = _three_asset_inputs()
    frontier = efficient_frontier(inputs, n_points=15)
    vols = [f.volatility for f in frontier]
    assert all(b >= a - 1e-6 for a, b in zip(vols, vols[1:]))


def test_max_sharpe_weights_sum_to_one():
    inputs = _three_asset_inputs()
    result = max_sharpe_portfolio(inputs, risk_free_rate=0.02)
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.sharpe_ratio is not None


def test_optimize_for_risk_level_bounds():
    inputs = _three_asset_inputs()
    conservative = optimize_for_risk_level(inputs, 0.0, risk_free_rate=0.02)
    aggressive = optimize_for_risk_level(inputs, 1.0, risk_free_rate=0.02)
    assert conservative.volatility <= aggressive.volatility
    with pytest.raises(PortfolioOptimizerError):
        optimize_for_risk_level(inputs, 1.5)


def test_optimize_for_risk_level_is_monotonic_in_volatility():
    inputs = _three_asset_inputs()
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    vols = [optimize_for_risk_level(inputs, lvl, risk_free_rate=0.02).volatility for lvl in levels]
    assert all(b >= a - 1e-4 for a, b in zip(vols, vols[1:]))


def test_long_only_weights_are_never_negative_with_scipy():
    pytest.importorskip("scipy")
    inputs = _three_asset_inputs()
    alloc = optimize_for_risk_level(inputs, 0.6, risk_free_rate=0.02, long_only=True)
    assert all(w >= -1e-6 for w in alloc.weights.values())
    assert sum(alloc.weights.values()) == pytest.approx(1.0, abs=1e-4)


def test_long_only_fallback_without_scipy(monkeypatch):
    """Simulates scipy not being installed by making its import fail --
    the module must fall back to clip-and-renormalize rather than crash."""
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    inputs = _three_asset_inputs()
    alloc = optimize_for_risk_level(inputs, 0.6, risk_free_rate=0.02, long_only=True)
    assert all(w >= -1e-9 for w in alloc.weights.values())
    assert sum(alloc.weights.values()) == pytest.approx(1.0, abs=1e-6)
