from __future__ import annotations

import math

import pytest

from app.quant_lab.options_pricing import (
    OptionsPricingError,
    black_scholes_greeks,
    black_scholes_price,
    compare_to_market,
    implied_volatility,
    norm_cdf,
)


def test_norm_cdf_matches_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-9)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)


def test_call_price_matches_textbook_reference():
    # Classic Hull textbook example: S=100, K=100, T=1, r=5%, sigma=20% -> ~10.4506
    price = black_scholes_price(100, 100, 1, 0.05, 0.20, "call")
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity_holds():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    call = black_scholes_price(S, K, T, r, sigma, "call")
    put = black_scholes_price(S, K, T, r, sigma, "put")
    assert (call - put) == pytest.approx(S - K * math.exp(-r * T), abs=1e-8)


def test_deep_itm_call_delta_near_one():
    greeks = black_scholes_greeks(200, 100, 0.5, 0.05, 0.20, "call")
    assert greeks.delta > 0.95


def test_deep_otm_put_delta_near_zero():
    greeks = black_scholes_greeks(200, 100, 0.5, 0.05, 0.20, "put")
    assert greeks.delta > -0.05


def test_gamma_identical_for_call_and_put_at_same_strike():
    call_g = black_scholes_greeks(100, 100, 1, 0.05, 0.20, "call")
    put_g = black_scholes_greeks(100, 100, 1, 0.05, 0.20, "put")
    assert call_g.gamma == pytest.approx(put_g.gamma, rel=1e-9)
    assert call_g.vega == pytest.approx(put_g.vega, rel=1e-9)


def test_implied_volatility_recovers_the_input_sigma():
    true_sigma = 0.35
    price = black_scholes_price(150, 140, 0.75, 0.03, true_sigma, "put")
    recovered = implied_volatility(price, 150, 140, 0.75, 0.03, "put")
    assert recovered == pytest.approx(true_sigma, abs=1e-4)


def test_implied_volatility_below_intrinsic_raises():
    with pytest.raises(OptionsPricingError):
        implied_volatility(market_price=0.01, S=200, K=100, T=1, r=0.05, option_type="call")


def test_invalid_inputs_raise():
    with pytest.raises(OptionsPricingError):
        black_scholes_price(-100, 100, 1, 0.05, 0.2, "call")
    with pytest.raises(OptionsPricingError):
        black_scholes_price(100, 100, 0, 0.05, 0.2, "call")
    with pytest.raises(OptionsPricingError):
        black_scholes_price(100, 100, 1, 0.05, 0.2, "swap")


def test_compare_to_market_reports_both_price_and_vol_gap():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    model_price = black_scholes_price(S, K, T, r, sigma, "call")
    result = compare_to_market(model_price + 1.0, S, K, T, r, sigma, "call")
    assert result.model_price == pytest.approx(model_price)
    assert result.price_diff == pytest.approx(-1.0, abs=1e-6)
    assert result.implied_vol > sigma  # a higher market price implies higher vol
    assert "Model price" in result.render_summary()
