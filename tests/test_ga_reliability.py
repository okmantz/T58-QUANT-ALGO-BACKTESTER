from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.refinement import apply_ga_reliability_adjustments
from app.prop.simulator import PropRules


def _trades(n, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2022-01-03 15:00")
    out = []
    for i in range(n):
        d = t0 + pd.Timedelta(days=i // 2, hours=i % 2)
        p = float(rng.choice([750, -500]))
        out.append(Trade(entry_time=d, exit_time=d + pd.Timedelta(minutes=30), direction=1, entry_price=1, exit_price=1,
                         size=1, pnl=p, pnl_pct=0, exit_reason="x", commission=0, equity_after=0, initial_risk=1))
    return out


def test_few_trades_cannot_be_ranked_and_quiet_never_beats_active():
    assert apply_ga_reliability_adjustments(5.0, 14) == float("-inf")
    quiet = apply_ga_reliability_adjustments(1.0, 20)
    active = apply_ga_reliability_adjustments(0.5, 300)
    assert active > quiet


def test_widened_stop_skip_rate_is_penalised_and_negative_untouched():
    clean = apply_ga_reliability_adjustments(1.0, 200, skipped_for_sizing=0)
    widened = apply_ga_reliability_adjustments(1.0, 200, skipped_for_sizing=800)
    assert widened < 0.3 * clean
    assert apply_ga_reliability_adjustments(-1.0, 200, skipped_for_sizing=800) == -1.0


def test_fitness_noise_across_40_seeds_is_bounded_at_the_floor():
    rules = PropRules(account_size=50_000, max_drawdown_pct=4.0, drawdown_type="trailing", daily_loss_limit_pct=2.4,
                      evaluation_profit_target_pct=6.0)
    trades = _trades(120)
    vals = []
    for seed in range(40):
        r = run_monte_carlo(trades, rules, MonteCarloConfig(n_simulations=300, random_seed=seed))
        vals.append(r.per_attempt_pass_probability)
    vals = np.array(vals, dtype=float)
    assert (vals > 0).all()
    assert vals.std() < 0.10 * vals.mean()
    assert (vals <= 1.0 + 1e-9).all() or (vals <= 100.0).all()
