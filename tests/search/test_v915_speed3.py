"""v9.15 WS-F (speed round 3): structural speedups must be bit-identical.

What this suite pins -- every change is a pure execution-strategy change
(shared Monte Carlo pool, parallel DSR trial backtests, initializer-held
frames and parallel paths in CPCV/PBO, parallel walk-forward folds) and
must produce byte-identical numbers to the serial/per-call paths it
replaces:

  * run_monte_carlo(pool=shared) == run_monte_carlo(max_workers=...) ==
    run_monte_carlo() serial, including ONE pool reused across two calls
    with different trade sets (the Full Pipeline baseline/final/rescore
    pattern);
  * run_cpcv(strategy_spec=..., max_workers=2) == serial builder run;
  * compute_pbo(max_workers=2) == compute_pbo(max_workers=1);
  * run_walk_forward(strategy_spec=..., max_workers=2) == serial builder run;
  * _trial_sharpes_for_specs(workers=2) == serial list (order + skips).

Nothing here touches a verdict threshold, seed, simulation count, or
statistical definition.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest


def _trending_df(n=1200, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 100.0
    rows = []
    for i in range(n):
        step = 0.15 * (1 if (i // 40) % 2 == 0 else -1) + rng.normal(0, 0.05)
        o, c = price, price + step
        rows.append((ts[i], o, max(o, c) + 0.02, min(o, c) - 0.02, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config(fast=5, slow=15):
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": fast, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": slow, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
    }


def _manual_spec(fast=5, slow=15):
    return {"source_type": "manual", "config": _sma_config(fast, slow)}


# ----------------------------------------------------------------------
# Shared Monte Carlo pool
# ----------------------------------------------------------------------

def _mc_trades(n=120, seed=7):
    from app.backtest.execution import Trade

    ts = pd.date_range("2026-01-05", periods=n, freq="1D")
    return [
        Trade(entry_time=ts[i], exit_time=ts[i] + pd.Timedelta(hours=2),
              direction=1 if i % 3 else -1, entry_price=100.0,
              exit_price=100.0 + float(((i * 37) % 11) - 5), size=10.0,
              pnl=float(((i * 37) % 11) - 5) * 40.0, pnl_pct=1.0,
              exit_reason="signal", commission=2.0, equity_after=50_000.0)
        for i in range(n)
    ]


def test_shared_mc_pool_matches_serial_and_per_call_pool():
    from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
    from app.orchestration.full_pipeline import _make_mc_pool
    from app.prop.simulator import PropRules

    trades = _mc_trades()
    rules = PropRules(account_size=50_000, evaluation_profit_target_pct=8,
                      daily_loss_limit_pct=5, max_drawdown_pct=10)
    cfg = MonteCarloConfig(n_simulations=240, random_seed=7, reset_on_breach=True)

    serial = run_monte_carlo(trades, rules, cfg)
    per_call = run_monte_carlo(trades, rules, cfg, max_workers=2)

    pool = _make_mc_pool(2)
    try:
        shared = run_monte_carlo(trades, rules, cfg, max_workers=2, pool=pool)
        # Reuse the SAME pool for a second call on a different trade set
        # (the pipeline baseline -> final -> rescore pattern).
        trades2 = _mc_trades(90, seed=11)
        shared2 = run_monte_carlo(trades2, rules, cfg, max_workers=2, pool=pool)
        serial2 = run_monte_carlo(trades2, rules, cfg)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    assert asdict(shared) == asdict(per_call) == asdict(serial)
    assert asdict(shared2) == asdict(serial2)


# ----------------------------------------------------------------------
# CPCV: spec-driven parallel == serial builder
# ----------------------------------------------------------------------

def test_cpcv_parallel_matches_serial():
    from app.backtest.risk import RiskConfig
    from app.prop.simulator import PropRules
    from app.strategy.manual import ManualStrategy
    from app.validation.cpcv import run_cpcv

    df = _trending_df()
    risk = RiskConfig(initial_balance=50_000, pip_size=1.0, contract_size=1.0)
    rules = PropRules(account_size=50_000)
    spec = _manual_spec()

    serial = run_cpcv(df, lambda: ManualStrategy(_sma_config()), risk,
                      n_groups=4, n_test_groups=2, metric="profit_factor",
                      prop_rules=rules, max_paths=6)
    parallel = run_cpcv(df, lambda: ManualStrategy(_sma_config()), risk,
                        n_groups=4, n_test_groups=2, metric="profit_factor",
                        prop_rules=rules, max_paths=6,
                        strategy_spec=spec, max_workers=2)
    assert parallel.to_dict() == serial.to_dict()


# ----------------------------------------------------------------------
# PBO: pooled (initializer frame) == serial
# ----------------------------------------------------------------------

def test_pbo_parallel_matches_serial():
    from app.backtest.risk import RiskConfig
    from app.prop.simulator import PropRules
    from app.validation.cpcv import compute_pbo

    df = _trending_df()
    risk = RiskConfig(initial_balance=50_000, pip_size=1.0, contract_size=1.0)
    rules = PropRules(account_size=50_000)
    specs = [_manual_spec(5, 15), _manual_spec(8, 20), _manual_spec(3, 10)]

    serial = compute_pbo(df, specs, risk, n_groups=4, n_test_groups=2,
                         metric="profit_factor", prop_rules=rules,
                         max_paths=6, max_workers=1)
    pooled = compute_pbo(df, specs, risk, n_groups=4, n_test_groups=2,
                         metric="profit_factor", prop_rules=rules,
                         max_paths=6, max_workers=2)
    assert pooled.to_dict() == serial.to_dict()


# ----------------------------------------------------------------------
# Walk-forward: spec-driven parallel == serial builder
# ----------------------------------------------------------------------

def test_walk_forward_parallel_matches_serial():
    from app.backtest.risk import RiskConfig
    from app.prop.simulator import PropRules
    from app.search.robustness import run_walk_forward
    from app.strategy.manual import ManualStrategy

    df = _trending_df()
    risk = RiskConfig(initial_balance=50_000, pip_size=1.0, contract_size=1.0)
    rules = PropRules(account_size=50_000)
    spec = _manual_spec()

    serial = run_walk_forward(df, lambda: ManualStrategy(_sma_config()), risk,
                              n_folds=3, metric="profit_factor", prop_rules=rules)
    parallel = run_walk_forward(df, lambda: ManualStrategy(_sma_config()), risk,
                                n_folds=3, metric="profit_factor", prop_rules=rules,
                                max_workers=2, strategy_spec=spec)
    assert serial is not None and parallel is not None
    assert asdict(parallel) == asdict(serial)


# ----------------------------------------------------------------------
# DSR trial Sharpes: parallel helper == serial loop
# ----------------------------------------------------------------------

def test_dsr_trial_sharpes_parallel_matches_serial():
    from app.backtest.risk import RiskConfig
    from app.orchestration.full_pipeline import _trial_sharpes_for_specs

    df = _trending_df()
    risk = RiskConfig(initial_balance=50_000, pip_size=1.0, contract_size=1.0)
    specs = [_manual_spec(f, s) for f, s in ((5, 15), (8, 20), (3, 10), (6, 18), (4, 12))]

    serial = _trial_sharpes_for_specs(specs, df, risk, None, workers=1)
    parallel = _trial_sharpes_for_specs(specs, df, risk, None, workers=2)
    assert len(serial) >= 4  # fixture must actually exercise the fan-out
    assert parallel == serial
