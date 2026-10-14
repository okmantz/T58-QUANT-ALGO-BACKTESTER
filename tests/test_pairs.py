"""Tests for app.data.pairs.merge_pair_series."""
from __future__ import annotations

import pandas as pd
import pytest
from app.data.pairs import PairDataError, merge_pair_series
import numpy as np
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


# ---- from test_pairs.py ----

def _df(timestamps, closes):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "open": closes, "high": closes, "low": closes, "close": closes,
    })


def test_merge_is_backward_looking_and_forward_fills():
    df = _df(["2023-01-01 00:00", "2023-01-01 00:05", "2023-01-01 00:10", "2023-01-01 00:15"], [1, 2, 3, 4])
    pair_df = _df(["2023-01-01 00:00", "2023-01-01 00:10"], [100, 200])
    merged = merge_pair_series(df, pair_df)
    assert list(merged["pair_close"]) == [100, 100, 200, 200]


def test_bars_before_pair_data_starts_get_nan():
    df = _df(["2023-01-01 00:00", "2023-01-01 00:05", "2023-01-01 00:10"], [1, 2, 3])
    pair_df = _df(["2023-01-01 00:05"], [50])
    merged = merge_pair_series(df, pair_df)
    assert pd.isna(merged["pair_close"].iloc[0])
    assert merged["pair_close"].iloc[1] == 50
    assert merged["pair_close"].iloc[2] == 50


def test_original_row_order_and_columns_preserved():
    df = _df(["2023-01-01 00:00", "2023-01-01 00:05", "2023-01-01 00:10"], [1, 2, 3])
    pair_df = _df(["2023-01-01 00:00"], [10])
    merged = merge_pair_series(df, pair_df)
    assert list(merged["close"]) == [1, 2, 3]  # original data untouched, in original order
    assert list(merged.index) == list(df.index)


def test_custom_column_name():
    df = _df(["2023-01-01 00:00"], [1])
    pair_df = _df(["2023-01-01 00:00"], [99])
    merged = merge_pair_series(df, pair_df, column_name="dxy_close")
    assert "dxy_close" in merged.columns
    assert merged["dxy_close"].iloc[0] == 99


def test_missing_timestamp_column_raises():
    df = pd.DataFrame({"close": [1, 2, 3]})
    pair_df = _df(["2023-01-01 00:00"], [1])
    with pytest.raises(PairDataError):
        merge_pair_series(df, pair_df)


def test_missing_close_column_on_pair_df_raises():
    df = _df(["2023-01-01 00:00"], [1])
    pair_df = pd.DataFrame({"timestamp": pd.to_datetime(["2023-01-01 00:00"])})
    with pytest.raises(PairDataError):
        merge_pair_series(df, pair_df)


def test_original_df_not_mutated():
    df = _df(["2023-01-01 00:00", "2023-01-01 00:05"], [1, 2])
    pair_df = _df(["2023-01-01 00:00"], [10])
    merge_pair_series(df, pair_df)
    assert "pair_close" not in df.columns


# ---- from test_pairs_trading.py ----

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
