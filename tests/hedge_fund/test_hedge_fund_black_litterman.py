import numpy as np
import pandas as pd
import pytest

from app.hedge_fund.black_litterman import (
    BlackLittermanError,
    confidence_sweep,
    confidence_to_omega,
    equal_weight_market_proxy,
    market_implied_returns,
    solve_posterior_weights,
    views_to_posterior,
)
from app.hedge_fund.research import AssetView
from app.quant_lab.portfolio_optimizer import build_inputs_from_prices


def _price_data(n=300, seed_offset=0):
    tickers = ["AAA", "BBB", "CCC"]
    rng_base = np.random.default_rng(1 + seed_offset)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D")
    data = {}
    for i, t in enumerate(tickers):
        rng = np.random.default_rng(10 + i + seed_offset)
        price = 100.0
        rows = []
        for k in range(n):
            step = 0.0002 * (i + 1) + rng.normal(0, 0.01)
            price *= (1 + step)
            rows.append((ts[k], price, price, price, price, 1000))
        data[t] = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return data


def test_confidence_to_omega_bounds():
    assert confidence_to_omega(0.1, 1.0) == pytest.approx(0.01)
    assert confidence_to_omega(0.1, 0.0) > confidence_to_omega(0.1, 0.01)
    with pytest.raises(BlackLittermanError):
        confidence_to_omega(0.1, 1.5)


def test_market_implied_returns_shape():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = equal_weight_market_proxy(2)
    pi = market_implied_returns(cov, w, risk_aversion=2.5)
    assert pi.shape == (2,)
    assert np.all(np.isfinite(pi))


def test_views_to_posterior_matches_prior_when_no_views():
    inputs = build_inputs_from_prices(_price_data())
    pi, posterior = views_to_posterior(inputs, {}, confidence=0.5)
    assert np.allclose(pi, posterior)


def test_low_confidence_collapses_toward_equal_weight_equilibrium():
    """Reproduces the article's own 'safety valve' result: as confidence
    falls, posterior weights should converge toward the equilibrium
    (equal-weight, absent market-cap data) portfolio."""
    price_data = _price_data()
    inputs = build_inputs_from_prices(price_data)
    views = {"AAA": AssetView(asset="AAA", mu=0.5, sigma=0.05, method="bootstrap", n_samples=32)}

    sweep = confidence_sweep(inputs, views, confidences=[1.0, 0.5, 0.01], long_only=True)
    high_conf_weights = sweep[0]["weights"]
    low_conf_weights = sweep[-1]["weights"]

    equal = 1.0 / inputs.n_assets
    high_conf_spread = max(abs(w - equal) for w in high_conf_weights.values())
    low_conf_spread = max(abs(w - equal) for w in low_conf_weights.values())
    assert low_conf_spread < high_conf_spread


def test_solve_posterior_weights_long_only_sums_to_one_and_nonnegative():
    inputs = build_inputs_from_prices(_price_data())
    _, posterior_mu = views_to_posterior(
        inputs, {"AAA": AssetView(asset="AAA", mu=0.3, sigma=0.1, method="bootstrap", n_samples=16)}, confidence=0.7,
    )
    allocation, warnings = solve_posterior_weights(inputs, posterior_mu, long_only=True)
    weights = np.array(list(allocation.weights.values()))
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(weights >= -1e-9)


def test_solve_posterior_weights_respects_turnover_cap():
    inputs = build_inputs_from_prices(_price_data())
    _, posterior_mu = views_to_posterior(
        inputs, {"AAA": AssetView(asset="AAA", mu=0.9, sigma=0.05, method="bootstrap", n_samples=32)}, confidence=1.0,
    )
    previous = {t: 1.0 / inputs.n_assets for t in inputs.tickers}
    allocation, warnings = solve_posterior_weights(
        inputs, posterior_mu, long_only=True, previous_weights=previous, max_turnover=0.10, transaction_cost_bps=10.0,
    )
    turnover = sum(abs(allocation.weights[t] - previous[t]) for t in inputs.tickers)
    assert turnover <= 0.10 + 1e-6
