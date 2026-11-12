"""Regression: a floating daily-loss breach under lock_day must BOOK the loss (liquidation at the limit).

Before the fix, on_trade_close returned DAY_LOCKED before ``balance += pnl`` so a stop-out beyond the
daily limit cost $0 and three -$1,314 days left lucid_50k at $50,000 with failed=False.
"""
import pandas as pd

from app.prop.presets import get_preset
from app.prop.simulator import simulate_account


def _run(n_days=3, pnl=-1314.0, mae=-1564.0):
    rules = get_preset("lucid_50k").to_prop_rules()
    days = pd.bdate_range("2026-01-05", periods=n_days)
    times = [d + pd.Timedelta(hours=10) for d in days]
    return rules, simulate_account(
        [pnl] * n_days, times, rules,
        trade_mae=[mae] * n_days, trade_mfe=[0.0] * n_days,
    )


def test_one_floating_breach_day_books_the_loss():
    rules, res = _run(n_days=1)
    assert res.final_balance < 50_000.0
    assert abs(res.final_balance - (50_000.0 - 1_200.0)) < 1e-6   # liquidated at the $1,200 limit


def test_three_losing_days_reduce_balance_and_trip_trailing_dd():
    rules, res = _run(n_days=3)
    assert res.final_balance < 50_000.0
    assert res.failed          # 3 x -$1,200 breaches the $2,000 trailing DD


def test_loss_inside_limit_is_booked_unchanged():
    rules, res = _run(n_days=1, pnl=-500.0, mae=-600.0)
    assert abs(res.final_balance - 49_500.0) < 1e-6
