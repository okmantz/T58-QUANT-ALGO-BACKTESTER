import numpy as np
import pandas as pd
import pytest

from app.hedge_fund.research import (
    EnsembleForecastConfig,
    ResearchError,
    bootstrap_view,
    generate_views,
    strategy_signal_view,
)
from app.strategy.manual import ManualStrategy


def _df(n=300, seed=1, drift=0.0005, vol=0.01):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D")
    price = 100.0
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, vol)
        o = price
        c = o * (1 + step)
        h = max(o, c) * 1.001
        l = min(o, c) * 0.999
        rows.append((ts[i], o, h, l, c, 1000))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config():
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 20, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow", "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow", "short_exit": "sma_fast > sma_slow",
    }


def test_bootstrap_view_returns_finite_mu_sigma():
    config = EnsembleForecastConfig(lookback_bars=100, horizon_bars=5, n_samples=32, seed=7)
    view = bootstrap_view(_df(), "AAA", config)
    assert view.method == "bootstrap"
    assert np.isfinite(view.mu)
    assert view.sigma > 0


def test_bootstrap_view_is_deterministic_given_a_seed():
    config = EnsembleForecastConfig(lookback_bars=100, horizon_bars=5, n_samples=32, seed=99)
    df = _df()
    v1 = bootstrap_view(df, "AAA", config)
    v2 = bootstrap_view(df, "AAA", config)
    assert v1.mu == v2.mu
    assert v1.sigma == v2.sigma


def test_bootstrap_view_rejects_too_little_history():
    config = EnsembleForecastConfig(lookback_bars=100, horizon_bars=5, n_samples=8)
    with pytest.raises(ResearchError):
        bootstrap_view(_df(n=10), "AAA", config)


def test_strategy_signal_view_uses_current_direction_and_falls_back_when_thin():
    config = EnsembleForecastConfig(lookback_bars=200, horizon_bars=5, n_samples=8)
    strategy = ManualStrategy(_sma_config())
    view = strategy_signal_view(_df(n=300, drift=0.001, seed=3), "AAA", strategy, config)
    assert view.method == "strategy_signal"
    assert view.signal_direction in (-1, 0, 1)
    # a flat book (no trend at all) should legitimately produce a flat/low-confidence view
    flat_view = strategy_signal_view(_df(n=300, drift=0.0, vol=0.0001, seed=3), "AAA", strategy, config)
    assert flat_view.method == "strategy_signal"


def test_generate_views_falls_back_to_bootstrap_when_no_strategy_given():
    config = EnsembleForecastConfig(lookback_bars=100, horizon_bars=5, n_samples=8)
    price_data = {"AAA": _df(seed=1), "BBB": _df(seed=2)}
    views = generate_views(price_data, config)
    assert set(views.keys()) == {"AAA", "BBB"}
    assert all(v.method == "bootstrap" for v in views.values())


def test_generate_views_uses_strategy_signal_only_for_assigned_assets():
    config = EnsembleForecastConfig(lookback_bars=200, horizon_bars=5, n_samples=8)
    price_data = {"AAA": _df(n=300, seed=1), "BBB": _df(n=300, seed=2)}
    strategies = {"AAA": ManualStrategy(_sma_config())}
    views = generate_views(price_data, config, strategies)
    assert views["AAA"].method == "strategy_signal"
    assert views["BBB"].method == "bootstrap"
