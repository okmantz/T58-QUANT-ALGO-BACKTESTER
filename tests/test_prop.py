import pandas as pd

from app.prop.simulator import PropRules, simulate_account


def test_evaluation_passes_on_sufficient_profit():
    rules = PropRules(
        account_size=10000,
        evaluation_profit_target_pct=5,
        daily_loss_limit_pct=100,
        max_drawdown_pct=100,
        consistency_rule_pct=None,
        min_trading_days=2,
        payout_frequency_days=0,
        payout_threshold_pct=0,
    )
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    pnls = [300, 300, 0]  # 600 / 10000 = 6% > 5% target, over 3 trading days
    result = simulate_account(pnls, dates, rules)
    assert result.passed_evaluation
    assert result.days_to_pass is not None


def test_daily_loss_limit_triggers_failure():
    rules = PropRules(account_size=10000, daily_loss_limit_pct=5, max_drawdown_pct=100)
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")]
    pnls = [-300, -300]  # -600 in one day = -6% > 5% limit
    result = simulate_account(pnls, dates, rules)
    assert result.failed
    assert result.failure_reason == "daily_loss_limit"


def test_max_drawdown_triggers_failure():
    rules = PropRules(account_size=10000, daily_loss_limit_pct=100, max_drawdown_pct=5, drawdown_type="static")
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    pnls = [-200, -200, -200]  # cumulative -600 = -6% > 5% max dd, spread across days to avoid daily limit
    result = simulate_account(pnls, dates, rules)
    assert result.failed
    assert "max_drawdown" in result.failure_reason


def test_funded_account_reaches_payout():
    rules = PropRules(
        account_size=10000,
        evaluation_profit_target_pct=5,
        daily_loss_limit_pct=100,
        max_drawdown_pct=100,
        consistency_rule_pct=None,
        min_trading_days=1,
        payout_frequency_days=1,
        payout_threshold_pct=1,
        required_buffer_pct=0,
    )
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    pnls = [600, 200, 200]  # passes eval on day 1, then accrues funded profit for payout
    result = simulate_account(pnls, dates, rules)
    assert result.passed_evaluation
    assert result.reached_first_payout


def test_no_trades_returns_safe_default():
    result = simulate_account([], [], PropRules())
    assert not result.passed_evaluation
    assert not result.failed


def test_reset_on_breach_defaults_to_single_attempt_identical_to_before():
    """reset_on_breach=False (the default) must be byte-identical to this
    function's behavior before the parameter existed -- every existing
    caller passes nothing here and must see no change at all."""
    rules = PropRules(account_size=10000, daily_loss_limit_pct=5, max_drawdown_pct=100)
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    pnls = [-300, -300, 500]  # busts the daily loss limit on day 1; day 2's trade is never reached
    result = simulate_account(pnls, dates, rules)
    assert result.failed
    assert result.total_attempts == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].failed


def test_reset_on_breach_chains_through_remaining_trades():
    """A bust with reset_on_breach=True snaps the account back to
    account_size and keeps walking the SAME remaining trade sequence,
    instead of stopping -- see app.prop.simulator.simulate_account's
    docstring."""
    rules = PropRules(account_size=10000, daily_loss_limit_pct=5, max_drawdown_pct=100,
                       evaluation_profit_target_pct=100, min_trading_days=1)
    # Three separate days, each one busts the daily loss limit on its own.
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    pnls = [-600, -600, -600]
    result = simulate_account(pnls, dates, rules, reset_on_breach=True)
    assert result.failed
    assert result.total_attempts == 3
    assert all(a.failed and a.failure_reason == "daily_loss_limit" for a in result.attempts)
    assert result.attempts_passed == 0


def test_reset_on_breach_stops_once_an_attempt_survives_to_the_end():
    """Once an attempt rides out every remaining trade without busting,
    the chain stops there -- there's no more history to mechanically
    rebuy into (see the "still alive" case in the docstring)."""
    rules = PropRules(account_size=10000, daily_loss_limit_pct=5, max_drawdown_pct=100,
                       evaluation_profit_target_pct=100, min_trading_days=1, consistency_rule_pct=None)
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    pnls = [-600, 50, 50]  # attempt 1 busts on day 1; attempt 2 survives days 2-3 without busting
    result = simulate_account(pnls, dates, rules, reset_on_breach=True)
    assert result.total_attempts == 2
    assert result.attempts[0].failed
    assert not result.attempts[1].failed
    assert not result.failed  # trailing state reflects the last (surviving) attempt


def test_reset_on_breach_top_level_fields_report_first_success_in_chain():
    """The top-level passed_evaluation field describes the FIRST attempt
    in the chain that reached that milestone, so a caller that never
    looks at `attempts` still gets simple single-outcome semantics --
    even though that same attempt goes on to bust later (funded-stage
    failures don't retroactively un-pass an earlier evaluation pass)."""
    rules = PropRules(account_size=10000, daily_loss_limit_pct=5, max_drawdown_pct=100,
                       evaluation_profit_target_pct=5, min_trading_days=1, consistency_rule_pct=None,
                       payout_frequency_days=0, payout_threshold_pct=0)
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    # Attempt 1 (day 1) busts the daily-loss limit immediately.
    # Attempt 2 starts fresh on day 2: +600 passes evaluation (6% > 5%
    # target) in the same attempt, which then carries straight into the
    # funded stage for day 3's trade -- a -600 day-3 trade then busts
    # THAT SAME attempt's daily-loss limit, ending the chain (no trades
    # left to mechanically rebuy into).
    pnls = [-600, 600, -600]
    result = simulate_account(pnls, dates, rules, reset_on_breach=True)
    assert result.total_attempts == 2
    assert result.passed_evaluation  # from attempt 2, even though attempt 2 later busts
    assert result.attempts_passed == 1
    assert result.failed  # trailing state: attempt 2 (the last attempt) did bust
