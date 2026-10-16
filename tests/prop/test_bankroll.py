import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.monte_carlo.bankroll import BankrollConfig, simulate_bankroll_survival
from app.prop.simulator import PropRules
from app.prop.survival_engine import ResetEconomics


def _mock_trades(n=150, seed_pnls=None):
    trades = []
    base = pd.Timestamp("2024-01-01")
    pnls = seed_pnls or ([400, -900] * (n // 2))
    for i, pnl in enumerate(pnls[:n]):
        t = base + pd.Timedelta(days=i // 3)
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1, entry_price=1.1, exit_price=1.1,
            size=1000, pnl=pnl, pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        ))
    return trades


def _rules():
    return PropRules(account_size=10000, evaluation_profit_target_pct=5, daily_loss_limit_pct=20,
                      max_drawdown_pct=20, min_trading_days=1, consistency_rule_pct=None,
                      payout_frequency_days=0, payout_threshold_pct=0)


def test_bankroll_survival_runs_and_produces_probabilities():
    trades = _mock_trades()
    econ = ResetEconomics(evaluation_fee=100, reset_fee=100, profit_split_pct=80, max_attempts=10)
    cfg = BankrollConfig(starting_bankroll=500, reset_economics=econ, n_simulations=200, random_seed=1)
    result = simulate_bankroll_survival(trades, _rules(), cfg)
    assert result.n_simulations == 200
    assert 0 <= result.probability_reach_first_payout <= 100
    assert 0 <= result.probability_ruin_before_first_payout <= 100
    assert 0 <= result.probability_exhausted_attempts_without_payout <= 100
    # The three outcome buckets are mutually exclusive and exhaustive.
    total = (
        result.probability_reach_first_payout
        + result.probability_ruin_before_first_payout
        + result.probability_exhausted_attempts_without_payout
    )
    assert total == pytest.approx(100.0, abs=0.01)


def test_smaller_bankroll_never_beats_larger_bankroll_on_ruin_probability():
    """A trader with less money to fund resets can never have a LOWER
    probability of going broke before their first payout than a trader
    with more money, all else equal."""
    trades = _mock_trades()
    econ = ResetEconomics(evaluation_fee=200, reset_fee=200, profit_split_pct=80, max_attempts=15)

    small_cfg = BankrollConfig(starting_bankroll=200, reset_economics=econ, n_simulations=400, random_seed=5)
    big_cfg = BankrollConfig(starting_bankroll=5000, reset_economics=econ, n_simulations=400, random_seed=5)

    small = simulate_bankroll_survival(trades, _rules(), small_cfg)
    big = simulate_bankroll_survival(trades, _rules(), big_cfg)

    assert small.probability_ruin_before_first_payout >= big.probability_ruin_before_first_payout


def test_zero_fee_economics_never_produces_ruin():
    """With no fee to pay, a life can never go broke before a payout --
    the only ways to end a chain without reaching one are running out of
    attempts or running out of trade history."""
    trades = _mock_trades()
    econ = ResetEconomics(evaluation_fee=0, reset_fee=0, profit_split_pct=80, max_attempts=5)
    cfg = BankrollConfig(starting_bankroll=0, reset_economics=econ, n_simulations=200, random_seed=2)
    result = simulate_bankroll_survival(trades, _rules(), cfg)
    assert result.probability_ruin_before_first_payout == 0.0


def test_stop_after_first_payout_never_uses_more_attempts_than_continuing():
    trades = _mock_trades()
    econ = ResetEconomics(evaluation_fee=50, reset_fee=50, profit_split_pct=80, max_attempts=10)

    stop_cfg = BankrollConfig(starting_bankroll=1000, reset_economics=econ, n_simulations=300,
                               random_seed=9, stop_after_first_payout=True)
    keep_going_cfg = BankrollConfig(starting_bankroll=1000, reset_economics=econ, n_simulations=300,
                                     random_seed=9, stop_after_first_payout=False)

    stop_result = simulate_bankroll_survival(trades, _rules(), stop_cfg)
    keep_going_result = simulate_bankroll_survival(trades, _rules(), keep_going_cfg)

    assert stop_result.expected_attempts_used <= keep_going_result.expected_attempts_used


def test_zero_trades_raises():
    cfg = BankrollConfig(n_simulations=10)
    with pytest.raises(ValueError):
        simulate_bankroll_survival([], _rules(), cfg)
