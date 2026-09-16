import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules


def _mock_trades(n=60, seed_pnls=None):
    trades = []
    base = pd.Timestamp("2024-01-01")
    pnls = seed_pnls or ([150, -80] * (n // 2))
    for i, pnl in enumerate(pnls[:n]):
        t = base + pd.Timedelta(days=i // 3)
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1, entry_price=1.1, exit_price=1.1,
            size=1000, pnl=pnl, pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        ))
    return trades


def test_monte_carlo_runs_and_produces_probabilities():
    trades = _mock_trades(90)
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5, daily_loss_limit_pct=50,
                       max_drawdown_pct=50, min_trading_days=1, consistency_rule_pct=None)
    cfg = MonteCarloConfig(n_simulations=200, method="bootstrap", random_seed=1)
    result = run_monte_carlo(trades, rules, cfg)
    assert 0 <= result.evaluation_pass_probability <= 100
    assert 0 <= result.first_payout_probability <= 100
    assert result.n_simulations == 200


def test_monte_carlo_shuffle_method():
    trades = _mock_trades(60)
    rules = PropRules()
    cfg = MonteCarloConfig(n_simulations=100, method="shuffle", random_seed=2)
    result = run_monte_carlo(trades, rules, cfg)
    assert result.n_simulations == 100


def test_monte_carlo_empty_trades_raises():
    with pytest.raises(ValueError):
        run_monte_carlo([], PropRules(), MonteCarloConfig(n_simulations=10))

def test_reset_on_breach_off_is_byte_identical_to_before():
    """MonteCarloConfig.reset_on_breach defaults to False -- every
    existing caller (which never sets it) must see identical output to
    before this field existed."""
    trades = _mock_trades(90, seed_pnls=[300, -700] * 45)
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5, daily_loss_limit_pct=50,
                       max_drawdown_pct=50, min_trading_days=1, consistency_rule_pct=None)
    cfg = MonteCarloConfig(n_simulations=150, method="bootstrap", random_seed=7)
    result = run_monte_carlo(trades, rules, cfg)
    assert result.reset_on_breach is False
    assert result.mean_attempts_per_path == 1.0
    assert result.median_attempts_per_path == 1.0
    assert "reset_on_breach was ON" not in result.methodology_note


def test_reset_on_breach_chains_within_each_resampled_path():
    """With reset_on_breach=True, each resampled path is walked through
    simulate_account's chain instead of stopping at the first bust, which
    can only raise (never lower) the any-attempt pass probability versus
    the single-attempt run on the identical resampling."""
    trades = _mock_trades(150, seed_pnls=[400, -900] * 75)
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5, daily_loss_limit_pct=20,
                       max_drawdown_pct=20, min_trading_days=1, consistency_rule_pct=None)

    single_cfg = MonteCarloConfig(n_simulations=300, method="block_bootstrap", block_size=8,
                                   reset_on_breach=False, random_seed=3)
    single = run_monte_carlo(trades, rules, single_cfg)

    chained_cfg = MonteCarloConfig(n_simulations=300, method="block_bootstrap", block_size=8,
                                    reset_on_breach=True, random_seed=3)
    chained = run_monte_carlo(trades, rules, chained_cfg)

    assert chained.reset_on_breach is True
    assert chained.mean_attempts_per_path >= 1.0
    assert chained.evaluation_pass_probability >= single.evaluation_pass_probability
    assert chained.any_attempt_pass_probability == chained.evaluation_pass_probability
    assert "reset_on_breach was ON" in chained.methodology_note
    assert len(chained.attempts_distribution) == 300
