from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.data.pairs import merge_pair_series
from app.quant_lab import pairs_trading
from app.quant_lab.pairs_trading import (
    PairsTradingError,
    build_pairs_strategy_config,
    run_pairs_backtest,
    screen_pairs,
)
from app.strategy.manual import ManualStrategy
from app.backtest.engine import run_backtest


def _mkdf(ts, price):
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.2, "low": price - 0.2,
        "close": price, "volume": 1000.0,
    })


def _cointegrated_universe(n=600, seed=5):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D")
    common = np.cumsum(rng.normal(0, 1, n))
    a = 100 + common + rng.normal(0, 0.5, n)          # A tracks the common factor tightly
    b = (a - rng.normal(0, 0.5, n)) / 2.0 + 25         # B is A's own linear transform + noise -> should cointegrate
    c = 80 + np.cumsum(rng.normal(0, 1, n))            # C is unrelated
    return {"A": _mkdf(ts, a), "B": _mkdf(ts, b), "C": _mkdf(ts, c)}


def test_screen_pairs_needs_at_least_two_symbols():
    with pytest.raises(PairsTradingError):
        screen_pairs({"A": _mkdf(pd.date_range("2023-01-01", periods=100), np.arange(100.0))})


def test_screen_pairs_finds_the_correlated_pair_and_ranks_it_first():
    universe = _cointegrated_universe()
    candidates = screen_pairs(universe, min_correlation=0.5, zscore_period=30)
    assert candidates, "expected at least one candidate pair above the correlation threshold"
    top = candidates[0]
    assert {top.symbol_a, top.symbol_b} == {"A", "B"}
    assert top.correlation > 0.5
    assert top.stationarity_verdict in ("likely mean-reverting", "borderline", "likely a random walk")


def test_screen_pairs_excludes_uncorrelated_pairs():
    universe = _cointegrated_universe()
    candidates = screen_pairs(universe, min_correlation=0.9, zscore_period=30)
    pairs_found = {frozenset((c.symbol_a, c.symbol_b)) for c in candidates}
    assert frozenset(("A", "C")) not in pairs_found
    assert frozenset(("B", "C")) not in pairs_found


def test_build_pairs_strategy_config_shape():
    config = build_pairs_strategy_config("AAPL", "MSFT", zscore_period=40, entry_z=2.0, exit_z=0.5,
                                          stop_loss_pips=20, take_profit_pips=40)
    assert config["entry_conditions"]["long"][0]["right"] == -2.0
    assert config["entry_conditions"]["short"][0]["right"] == 2.0
    assert config["risk_management"]["stop_value"] == 20
    # The config must be a real, runnable Manual Strategy Builder config.
    strat = ManualStrategy(config)
    assert strat.config["name"].startswith("Pairs Reversion")


def test_pairs_strategy_produces_a_runnable_backtest():
    universe = _cointegrated_universe()
    merged = merge_pair_series(universe["A"], universe["B"])
    config = build_pairs_strategy_config("A", "B", zscore_period=30, entry_z=1.0, exit_z=0.25)
    strat = ManualStrategy(config)
    bt = run_backtest(merged, strat, RiskConfig())
    assert bt.statistics.total_trades == len(bt.trades)


def test_run_pairs_backtest_end_to_end_with_mocked_alpaca(monkeypatch):
    universe = _cointegrated_universe(n=400, seed=9)

    def fake_fetch(api_key, secret_key, symbol, timeframe_label, start, end, feed="iex"):
        return universe[symbol]

    monkeypatch.setattr(pairs_trading, "fetch_stock_bars", fake_fetch)
    result = run_pairs_backtest(
        "fake_key", "fake_secret", "A", "B", "1Day", "2023-01-01", "2024-12-31",
        zscore_period=30, entry_z=1.0, exit_z=0.25,
    )
    assert result.pair is not None
    assert result.pair.symbol_a == "A" and result.pair.symbol_b == "B"
    assert result.backtest.statistics.total_trades == len(result.backtest.trades)
