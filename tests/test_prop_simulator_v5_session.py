"""v5 simulator tests: session-day grouping, adverse floating-DD mode,
portfolio correlation caps.

The session-day tests depend on `app.data.trading_day.trading_day` with the
spec'd signature trading_day(ts, *, tz="America/Chicago", roll_hour=17) ->
datetime.date (sibling worker's module). These tests only assume:
  - naive timestamps are treated as UTC,
  - a trade at 16:59 CT and a trade at 17:01 CT land on DIFFERENT days.
"""
import pandas as pd

from app.data.trading_day import trading_day
from app.prop.simulator import PropRules, precompute_day_structure, simulate_account


def _loose_rules(**over):
    kw = dict(
        account_size=10000,
        evaluation_profit_target_pct=1000.0,  # never passes; pure survival test
        daily_loss_limit_pct=100.0,           # effectively off
        max_drawdown_pct=100.0,               # effectively off
        consistency_rule_pct=None,
        min_trading_days=1,
        payout_frequency_days=0,
        payout_threshold_pct=0,
        winning_days_for_payout=1,
    )
    kw.update(over)
    return PropRules(**kw)


# ---------------------------------------------------------------------------
# 1. Session-day grouping: 17:00 CT roll boundary
# ---------------------------------------------------------------------------

def test_trading_day_splits_at_1700_ct_roll():
    # January: CT = UTC-6. Both naive-UTC stamps are the same UTC calendar
    # day; in CT they straddle the 17:00 roll.
    a = pd.Timestamp("2024-01-02 22:59")  # 16:59 CT
    b = pd.Timestamp("2024-01-02 23:01")  # 17:01 CT
    assert trading_day(a) != trading_day(b)


def test_day_grouping_uses_session_day_not_utc_midnight():
    rules = _loose_rules(daily_loss_limit_pct=5.0)  # $500/day on $10k
    dates = [pd.Timestamp("2024-01-02 22:59"), pd.Timestamp("2024-01-02 23:01")]
    ds = precompute_day_structure(dates)
    assert ds.n_days == 2, "naive-UTC-midnight bucketing would give 1 day"
    assert ds.day_index_per_trade == [0, 1]
    assert ds.is_last_of_day == [True, True]

    # Daily-loss attribution follows the session day: each session loses
    # $400 (< $500 limit) -> account survives. Old UTC-midnight bucketing
    # would put -$800 on one day and fail the account.
    result = simulate_account([-400.0, -400.0], dates, rules)
    assert not result.failed, f"unexpected failure: {result.failure_reason}"
    assert result.trading_days_count == 2


def test_daily_loss_still_fails_within_one_session():
    rules = _loose_rules(daily_loss_limit_pct=5.0)
    # Both trades well before the 17:00 CT roll -> same session day.
    dates = [pd.Timestamp("2024-01-02 20:00"), pd.Timestamp("2024-01-02 21:00")]  # 14:00/15:00 CT
    result = simulate_account([-300.0, -300.0], dates, rules)
    assert result.failed
    assert result.failure_reason == "daily_loss_limit"


# ---------------------------------------------------------------------------
# 2. Adverse floating-drawdown mode
# ---------------------------------------------------------------------------

def _adverse_scenario():
    # Static 5% max DD -> floor $9,500 on $10k. Two losses keep realized
    # balance above the floor, but trade 2's full initial risk ($300) would
    # have dragged floating equity to $9,400 intrabar.
    rules = _loose_rules(max_drawdown_pct=5.0, drawdown_type="static")
    dates = [pd.Timestamp("2024-01-02 20:00"), pd.Timestamp("2024-01-02 21:00")]
    pnls = [-300.0, -150.0]
    risks = [300.0, 300.0]
    return rules, dates, pnls, risks


def test_adverse_mode_fails_where_realized_passes():
    rules, dates, pnls, risks = _adverse_scenario()
    realized = simulate_account(pnls, dates, rules)  # default mode
    assert not realized.failed, "realized mode should pass this scenario"

    adverse_rules = _loose_rules(
        max_drawdown_pct=5.0, drawdown_type="static", floating_drawdown_mode="adverse"
    )
    adverse = simulate_account(pnls, dates, adverse_rules, trade_initial_risks=risks)
    assert adverse.failed
    assert "adverse-floating proxy" in adverse.failure_reason


def test_realized_mode_ignores_initial_risks():
    # Passing trade_initial_risks with the default ("realized") mode must be
    # byte-identical to not passing them at all.
    rules, dates, pnls, risks = _adverse_scenario()
    plain = simulate_account(pnls, dates, rules)
    with_risks = simulate_account(pnls, dates, rules, trade_initial_risks=risks)
    assert with_risks.failed == plain.failed
    assert with_risks.final_balance == plain.final_balance
    assert not with_risks.failed


def test_adverse_mode_without_risks_is_noop():
    rules, dates, pnls, _ = _adverse_scenario()
    adverse_rules = _loose_rules(
        max_drawdown_pct=5.0, drawdown_type="static", floating_drawdown_mode="adverse"
    )
    result = simulate_account(pnls, dates, adverse_rules)  # no risks -> proxy collapses to realized
    assert not result.failed


# ---------------------------------------------------------------------------
# 3. Portfolio correlation caps (astra port #5)
# ---------------------------------------------------------------------------

def _concurrent_trades(n, sides=None):
    """n trades, all open at once: entries 10:00..10:0n, exits 12:00..12:0n."""
    entries = [pd.Timestamp(f"2024-01-02 10:{i:02d}") for i in range(n)]
    exits = [pd.Timestamp(f"2024-01-02 12:{i:02d}") for i in range(n)]
    pnls = [100.0] * n
    return entries, exits, pnls, sides if sides is not None else [1] * n


def test_concurrent_position_cap_blocks_7th():
    entries, exits, pnls, sides = _concurrent_trades(7)
    rules = _loose_rules(account_size=100000, max_concurrent_positions=6)
    result = simulate_account(
        pnls, exits, rules, trade_entry_times=entries, trade_sides=sides
    )
    assert result.blocked_trades == 1
    assert result.final_balance == 100000 + 6 * 100.0
    assert not result.failed


def test_caps_off_by_default_is_byte_identical():
    entries, exits, pnls, sides = _concurrent_trades(7)
    rules = _loose_rules(account_size=100000)  # caps None -> overlay off
    result = simulate_account(
        pnls, exits, rules, trade_entry_times=entries, trade_sides=sides
    )
    assert result.blocked_trades == 0
    assert result.final_balance == 100000 + 7 * 100.0


def test_caps_configured_without_entry_times_stays_off():
    _, exits, pnls, _ = _concurrent_trades(7)
    rules = _loose_rules(account_size=100000, max_concurrent_positions=6)
    result = simulate_account(pnls, exits, rules)  # no entry times -> no overlap info -> off
    assert result.blocked_trades == 0
    assert result.final_balance == 100000 + 7 * 100.0


def test_same_direction_cap_blocks_correlated_longs():
    sides = [1, 1, 1, 1, -1, 1, 1]
    entries, exits, pnls, _ = _concurrent_trades(7, sides)
    rules = _loose_rules(account_size=100000, max_same_direction_exposure=4)
    result = simulate_account(
        pnls, exits, rules, trade_entry_times=entries, trade_sides=sides
    )
    # 4 longs taken; the short is fine (opposite direction); the 5th and 6th
    # longs are blocked as same-direction over-exposure.
    assert result.blocked_trades == 2
    assert result.final_balance == 100000 + 5 * 100.0


def test_blocked_trades_get_no_day_attribution():
    # All 7 trades share one session day; the blocked one must not create
    # PnL, but the day still exists for the other six.
    entries, exits, pnls, sides = _concurrent_trades(7)
    rules = _loose_rules(account_size=100000, max_concurrent_positions=6)
    result = simulate_account(
        pnls, exits, rules, trade_entry_times=entries, trade_sides=sides
    )
    assert result.trading_days_count == 1
    assert result.final_balance == 100600.0
