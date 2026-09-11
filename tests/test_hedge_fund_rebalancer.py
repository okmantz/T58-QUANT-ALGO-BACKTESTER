import numpy as np
import pandas as pd
import pytest

from app.hedge_fund.rebalancer import (
    RebalanceConfig,
    RebalanceError,
    run_rebalance_backtest,
    weights_to_orders,
)
from app.hedge_fund.research import EnsembleForecastConfig


def _df(n=500, seed=1, drift=0.0005):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D")
    price = 100.0
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.01)
        o = price
        c = o * (1 + step)
        rows.append((ts[i], o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1000))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _cfg(**overrides):
    base = dict(
        initial_balance=100_000.0, rebalance_every_bars=10,
        forecast=EnsembleForecastConfig(lookback_bars=100, horizon_bars=5, n_samples=16, seed=5),
        confidence=0.5, long_only=True, max_turnover=0.5, transaction_cost_bps=10.0,
    )
    base.update(overrides)
    return RebalanceConfig(**base)


def test_weights_to_orders_generates_buys_from_flat():
    orders, new_shares = weights_to_orders(
        pd.Timestamp("2023-01-01"), {"AAA": 0.6, "BBB": 0.4}, {}, {"AAA": 100.0, "BBB": 50.0},
        equity=10_000.0, min_trade_frac=0.001,
    )
    assert {o.asset for o in orders} == {"AAA", "BBB"}
    assert all(o.side == "buy" for o in orders)
    assert new_shares["AAA"] == pytest.approx(60.0)
    assert new_shares["BBB"] == pytest.approx(80.0)


def test_weights_to_orders_skips_dust_trades():
    orders, new_shares = weights_to_orders(
        pd.Timestamp("2023-01-01"), {"AAA": 0.5001}, {"AAA": 50.0}, {"AAA": 100.0},
        equity=10_000.0, min_trade_frac=0.05,  # 5% of equity = $500 minimum
    )
    assert orders == []  # the implied trade is $0.10 notional, well under the dust floor


def test_run_rebalance_backtest_requires_at_least_two_assets():
    with pytest.raises(RebalanceError):
        run_rebalance_backtest({"AAA": _df()}, _cfg())


def test_run_rebalance_backtest_requires_enough_overlapping_bars():
    with pytest.raises(RebalanceError):
        run_rebalance_backtest({"AAA": _df(n=10), "BBB": _df(n=10, seed=2)}, _cfg())


def test_run_rebalance_backtest_produces_full_length_equity_curve():
    price_data = {"AAA": _df(seed=1, drift=0.0008), "BBB": _df(seed=2, drift=0.0002), "CCC": _df(seed=3, drift=-0.0003)}
    result = run_rebalance_backtest(price_data, _cfg())
    assert len(result.equity_curve) == 500
    assert result.equity_curve["equity"].iloc[0] == pytest.approx(100_000.0)
    assert result.stats.num_rebalances == len(result.cycles)
    assert result.stats.num_rebalances > 0


def test_every_cycles_target_weights_sum_to_one_and_are_nonnegative_when_long_only():
    price_data = {"AAA": _df(seed=1), "BBB": _df(seed=2), "CCC": _df(seed=3)}
    result = run_rebalance_backtest(price_data, _cfg())
    for cycle in result.cycles:
        total = sum(cycle.target_weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)
        assert all(w >= -1e-9 for w in cycle.target_weights.values())


def test_higher_transaction_costs_reduce_or_equal_net_return():
    price_data = {"AAA": _df(seed=1, drift=0.001), "BBB": _df(seed=2, drift=0.0002), "CCC": _df(seed=3, drift=-0.0005)}
    cheap = run_rebalance_backtest(price_data, _cfg(transaction_cost_bps=0.0, rebalance_every_bars=5))
    expensive = run_rebalance_backtest(price_data, _cfg(transaction_cost_bps=200.0, rebalance_every_bars=5))
    assert expensive.stats.total_transaction_costs >= cheap.stats.total_transaction_costs
    assert expensive.equity_curve["equity"].iloc[-1] <= cheap.equity_curve["equity"].iloc[-1] + 1e-6


def test_max_turnover_cap_is_never_exceeded_once_the_book_is_established():
    """The turnover cap bounds CHURN of an existing book. It legitimately
    does not (and cannot) bind on the very first allocation out of cash
    -- sum(|w - 0|) == sum(w) == 1 for any fully-invested long-only
    allocation, so a cap below 1.0 would make the first cycle infeasible
    by construction, not merely expensive. Every cycle AFTER the first
    must respect it."""
    price_data = {"AAA": _df(seed=1, drift=0.002), "BBB": _df(seed=2, drift=-0.002), "CCC": _df(seed=3, drift=0.0)}
    result = run_rebalance_backtest(price_data, _cfg(max_turnover=0.10, transaction_cost_bps=5.0, rebalance_every_bars=5))
    assert len(result.cycles) > 1
    assert all(c.turnover <= 0.10 + 1e-6 for c in result.cycles[1:])
