"""v6 W3 build tests: funded-stage consistency (B5), preset remap (B6-B9),
to_prop_rules pass-through (B10), constraint-utilization export (B12-sim),
daily-loss action (B13), eval time limit + payout caps (B14), and the
daily-loss base (B15).
"""
import pandas as pd
import pytest

from app.prop.presets import get_preset
from app.prop.simulator import PropRules, simulate_account, summarize_single_run


def _dates(n, start="2024-01-01"):
    return [pd.Timestamp(start) + pd.Timedelta(days=i) for i in range(n)]


def _funded_rules(**over):
    """Rules that pass the eval fast and pay out freely (except the gate
    under test)."""
    kw = dict(
        account_size=50_000,
        evaluation_profit_target_pct=1.0,   # $500 target
        daily_loss_limit_pct=100.0,          # effectively off
        max_drawdown_pct=100.0,             # effectively off
        consistency_rule_pct=None,
        min_trading_days=1,
        payout_frequency_days=0,
        payout_threshold_pct=0.0,
        winning_days_for_payout=0,
        min_winning_day_profit=0.0,
    )
    kw.update(over)
    return PropRules(**kw)


# ---------------------------------------------------------------------------
# B5: funded-stage consistency
# ---------------------------------------------------------------------------

def test_funded_consistency_blocks_payout_when_breached():
    # Day 1 passes the eval (+1000). Day 2 is one huge +9000 day: with a
    # 50% funded consistency rule the payout must be blocked (9000/10000 = 90%).
    dates = _dates(10)
    pnls = [1000.0, 9000.0] + [100.0] * 8
    r = simulate_account(pnls, dates, _funded_rules(funded_consistency_rule_pct=50.0))
    assert r.passed_evaluation
    assert not r.reached_first_payout, "huge-day payout should be blocked by funded consistency"
    assert r.payouts == []


def test_funded_consistency_allows_payout_when_clean():
    # Same shape but the post-pass profit is spread evenly: 1000/day for
    # 10 days -> best day 1000 / 10000 = 10% <= 50% -> payout allowed.
    dates = _dates(11)
    pnls = [1000.0] + [1000.0] * 10
    r = simulate_account(pnls, dates, _funded_rules(funded_consistency_rule_pct=50.0))
    assert r.passed_evaluation
    assert r.reached_first_payout
    assert len(r.payouts) >= 1


def test_funded_consistency_none_means_no_gate():
    dates = _dates(10)
    pnls = [1000.0, 9000.0] + [100.0] * 8
    r = simulate_account(pnls, dates, _funded_rules(funded_consistency_rule_pct=None))
    assert r.reached_first_payout


def test_funded_consistency_resets_window_after_payout():
    # The consistency window restarts at each payout baseline: day 2's
    # even profit earns payout 1, then a huge day 3 blocks payout 2.
    dates = _dates(7)
    pnls = [1000.0, 2000.0, 2000.0, 9000.0, 100.0, 100.0, 100.0]
    r = simulate_account(pnls, dates, _funded_rules(funded_consistency_rule_pct=50.0))
    assert r.passed_evaluation
    assert len(r.payouts) == 1, f"expected exactly 1 payout, got {len(r.payouts)}"
    assert r.payouts[0].amount == pytest.approx(4000.0)


# ---------------------------------------------------------------------------
# B10: to_prop_rules passes the 4 new fields (+ funded consistency)
# ---------------------------------------------------------------------------

def test_to_prop_rules_passes_new_fields():
    p = get_preset("topstep_50k")
    r = p.to_prop_rules()
    assert r.winning_days_for_payout == p.winning_days_for_payout == 5
    assert r.min_winning_day_profit == p.min_winning_day_profit == 150.0
    assert r.floating_drawdown_mode == p.floating_drawdown_mode == "realized"
    assert r.funded_consistency_rule_pct == p.funded_consistency_rule_pct == 40.0


def test_to_prop_rules_inactive_days_convention():
    # preset default 0 -> rules None (rule off); FTMO's 30 passes through.
    assert get_preset("the5ers_20k").to_prop_rules().max_inactive_days is None
    assert get_preset("ftmo_100k").to_prop_rules().max_inactive_days == 30


# ---------------------------------------------------------------------------
# B13: daily_loss_action
# ---------------------------------------------------------------------------

def test_daily_loss_action_fail_kills_account():
    dates = [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-01 11:00"),
             pd.Timestamp("2024-01-02 10:00")]
    rules = PropRules(account_size=10_000, evaluation_profit_target_pct=1000.0,
                      daily_loss_limit_pct=5.0, max_drawdown_pct=100.0,
                      consistency_rule_pct=None, min_trading_days=1,
                      payout_frequency_days=0, winning_days_for_payout=0,
                      daily_loss_action="fail")
    r = simulate_account([-600.0, 50.0, 50.0], dates, rules)
    assert r.failed and r.failure_reason == "daily_loss_limit"
    assert r.final_balance == 9400.0  # died at the breach; later trades never taken


def test_daily_loss_action_lock_day_resumes_next_day():
    dates = [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-01 11:00"),
             pd.Timestamp("2024-01-02 10:00"), pd.Timestamp("2024-01-02 11:00")]
    rules = PropRules(account_size=10_000, evaluation_profit_target_pct=1000.0,
                      daily_loss_limit_pct=5.0, max_drawdown_pct=100.0,
                      consistency_rule_pct=None, min_trading_days=1,
                      payout_frequency_days=0, winning_days_for_payout=0,
                      daily_loss_action="lock_day")
    # Day 1: -600 breaches the $500 DLL -> the +100 later that day is
    # skipped; day 2 trades (+100, +100) are taken normally.
    r = simulate_account([-600.0, 100.0, 100.0, 100.0], dates, rules)
    assert not r.failed, f"lock_day should survive, got {r.failure_reason}"
    assert r.final_balance == pytest.approx(10_000 - 600 + 200)


def test_daily_loss_action_rejects_unknown_values():
    with pytest.raises(ValueError):
        PropRules(daily_loss_action="explode")
    with pytest.raises(ValueError):
        PropRules(daily_loss_base="sometimes")


# ---------------------------------------------------------------------------
# B14: eval time limit + payout caps
# ---------------------------------------------------------------------------

def test_max_eval_calendar_days_fails_stale_eval():
    dates = _dates(40)  # 40 calendar days of small gains, never hitting target
    pnls = [10.0] * 40
    rules = PropRules(account_size=10_000, evaluation_profit_target_pct=50.0,
                      daily_loss_limit_pct=100.0, max_drawdown_pct=100.0,
                      consistency_rule_pct=None, min_trading_days=1,
                      payout_frequency_days=0, winning_days_for_payout=0,
                      max_eval_calendar_days=30)
    r = simulate_account(pnls, dates, rules)
    assert r.failed and r.failure_reason == "eval_time_limit"


def test_no_eval_time_limit_by_default():
    dates = _dates(40)
    pnls = [10.0] * 40
    rules = PropRules(account_size=10_000, evaluation_profit_target_pct=50.0,
                      daily_loss_limit_pct=100.0, max_drawdown_pct=100.0,
                      consistency_rule_pct=None, min_trading_days=1,
                      payout_frequency_days=0, winning_days_for_payout=0)
    r = simulate_account(pnls, dates, rules)
    assert not r.failed  # survives to the end of the data


def test_payout_cap_dollars_enforced():
    dates = _dates(6)
    pnls = [1000.0] + [2000.0] * 5  # pass day 1, then +2000/day
    r = simulate_account(pnls, dates, _funded_rules(payout_cap_dollars=500.0))
    assert r.reached_first_payout
    assert all(p.amount <= 500.0 + 1e-9 for p in r.payouts)
    assert r.payouts[0].amount == pytest.approx(500.0)


def test_max_payouts_caps_lifetime_payouts():
    dates = _dates(12)
    pnls = [1000.0] + [2000.0] * 11
    r = simulate_account(pnls, dates, _funded_rules(max_payouts=2))
    assert len(r.payouts) == 2


# ---------------------------------------------------------------------------
# B15: daily_loss_base
# ---------------------------------------------------------------------------

def test_daily_loss_base_initial_is_default_behavior():
    # Sanity: "initial" behaves exactly like the old DLL check.
    dates = [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-02 10:00")]
    rules = PropRules(account_size=10_000, evaluation_profit_target_pct=1000.0,
                      daily_loss_limit_pct=5.0, max_drawdown_pct=100.0,
                      consistency_rule_pct=None, min_trading_days=1,
                      payout_frequency_days=0, winning_days_for_payout=0,
                      daily_loss_base="initial")
    r = simulate_account([-600.0, 50.0], dates, rules)
    assert r.failed and r.failure_reason == "daily_loss_limit"


def test_daily_loss_base_ratchet_up_never_decreases():
    # Day 1 closes +1000 (base ratchets to 11000 -> $550 limit).
    # Day 2 closes -500 (base stays 11000 -- ratchet never decreases).
    # Day 3 loses -520: under "initial" the $500 limit fails; under
    # "ratchet_up" the $550 limit survives.
    dates = [pd.Timestamp("2024-01-01 10:00"),
             pd.Timestamp("2024-01-02 10:00"),
             pd.Timestamp("2024-01-03 10:00")]
    pnls = [1000.0, -500.0, -520.0]
    kw = dict(account_size=10_000, evaluation_profit_target_pct=1000.0,
              daily_loss_limit_pct=5.0, max_drawdown_pct=100.0,
              consistency_rule_pct=None, min_trading_days=1,
              payout_frequency_days=0, winning_days_for_payout=0)
    r_init = simulate_account(pnls, dates, PropRules(daily_loss_base="initial", **kw))
    r_ratchet = simulate_account(pnls, dates, PropRules(daily_loss_base="ratchet_up", **kw))
    assert r_init.failed and r_init.failure_reason == "daily_loss_limit"
    assert not r_ratchet.failed, f"ratchet base should survive -520: {r_ratchet.failure_reason}"


def test_daily_loss_base_prior_day_high():
    # Day 1: +400 then -100 -> peak 10400, close 10300.
    # Day 2 loses -510: "initial" ($500 limit) fails; "prior_day_high"
    # (peak 10400 -> $520 limit) survives.
    dates = [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-01 11:00"),
             pd.Timestamp("2024-01-02 10:00")]
    pnls = [400.0, -100.0, -510.0]
    kw = dict(account_size=10_000, evaluation_profit_target_pct=1000.0,
              daily_loss_limit_pct=5.0, max_drawdown_pct=100.0,
              consistency_rule_pct=None, min_trading_days=1,
              payout_frequency_days=0, winning_days_for_payout=0)
    r_init = simulate_account(pnls, dates, PropRules(daily_loss_base="initial", **kw))
    r_prior = simulate_account(pnls, dates, PropRules(daily_loss_base="prior_day_high", **kw))
    assert r_init.failed and r_init.failure_reason == "daily_loss_limit"
    assert not r_prior.failed, f"prior-day-high base should survive -510: {r_prior.failure_reason}"


# ---------------------------------------------------------------------------
# B12-sim: constraint-utilization export
# ---------------------------------------------------------------------------

def test_utilization_fields_exact_names_and_ratios():
    dates = _dates(6)
    pnls = [1000.0] + [500.0] * 5  # pass day 1, then steady gains
    rules = _funded_rules(consistency_rule_pct=None, funded_consistency_rule_pct=50.0,
                          daily_loss_limit_pct=5.0, max_drawdown_pct=10.0)
    r = simulate_account(pnls, dates, rules)
    s = summarize_single_run(r, rules)
    # best day 1000; total net profit = (final - 50000) + payouts.
    # breach amount = 50% x net profit -> ratio is a sane positive float.
    assert isinstance(s["best_day_profit_pct_of_limit"], float)
    assert s["best_day_profit_pct_of_limit"] > 0
    assert s["worst_daily_loss_pct_of_limit"] == pytest.approx(0.0)  # no losing days
    assert s["max_dd_pct_of_limit"] == pytest.approx(r.max_drawdown_pct_reached / 10.0)


def test_utilization_none_when_rules_disabled():
    dates = _dates(4)
    pnls = [100.0, -50.0, 100.0, 100.0]
    rules = _funded_rules(consistency_rule_pct=None, funded_consistency_rule_pct=None,
                          daily_loss_limit_pct=100.0,  # disabled convention
                          max_drawdown_pct=100.0)
    r = simulate_account(pnls, dates, rules)
    s = summarize_single_run(r, rules)
    assert s["best_day_profit_pct_of_limit"] is None
    assert s["worst_daily_loss_pct_of_limit"] is None
    assert s["max_dd_pct_of_limit"] == pytest.approx(r.max_drawdown_pct_reached / 100.0)


def test_utilization_ratio_hits_one_at_limit():
    # Worst daily loss exactly at the DLL: ratio == 1.0.
    dates = [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-02 10:00")]
    rules = PropRules(account_size=10_000, evaluation_profit_target_pct=1000.0,
                      daily_loss_limit_pct=5.0, max_drawdown_pct=100.0,
                      consistency_rule_pct=None, min_trading_days=1,
                      payout_frequency_days=0, winning_days_for_payout=0,
                      daily_loss_action="lock_day")  # survive the breach to get a summary
    r = simulate_account([-500.0, 100.0], dates, rules)
    s = summarize_single_run(r, rules)
    assert s["worst_daily_loss_pct_of_limit"] == pytest.approx(1.0)


def test_summarize_without_rules_keeps_old_callers_working():
    dates = _dates(3)
    r = simulate_account([100.0, 100.0, 100.0], dates, _funded_rules())
    s = summarize_single_run(r)  # no rules -> new fields are None
    assert s["best_day_profit_pct_of_limit"] is None
    assert s["worst_daily_loss_pct_of_limit"] is None
    assert s["max_dd_pct_of_limit"] is None
    assert s["evaluation_pass_pct"] in (0.0, 100.0)


# ---------------------------------------------------------------------------
# B6-B9: preset values match the spec
# ---------------------------------------------------------------------------

def test_apex_split_values():
    specs = {
        "apex_50k_eod":      (2.0, 4.0, "eod", 250.0),
        "apex_50k_intraday": (100.0, 4.0, "intrabar", 200.0),
        "apex_100k_eod":     (1.5, 3.0, "eod", 250.0),
        "apex_100k_intraday": (100.0, 3.0, "intrabar", 200.0),
    }
    for key, (dll, mdd, mode, gate) in specs.items():
        p = get_preset(key)
        assert p.daily_loss_limit_pct == dll, key
        assert p.max_drawdown_pct == mdd, key
        assert p.drawdown_type == "trailing", key
        assert p.drawdown_check_mode == mode, key
        assert p.winning_days_for_payout == 5, key
        assert p.min_winning_day_profit == gate, key
        assert p.funded_consistency_rule_pct == 50.0, key
        assert p.consistency_rule_pct is None and p.min_trading_days == 0, key


def test_ftmo_values():
    for key in ("ftmo_10k", "ftmo_100k", "ftmo_200k"):
        p = get_preset(key)
        assert p.drawdown_check_mode == "intrabar", key
        assert p.evaluation_profit_target_pct == 10.0, key
        assert p.daily_loss_limit_pct == 5.0, key
        assert p.max_drawdown_pct == 10.0, key
        assert p.drawdown_type == "static", key
        assert p.min_trading_days == 4, key


def test_fundednext_values():
    p25, p100 = get_preset("fundednext_25k"), get_preset("fundednext_100k")
    for p in (p25, p100):
        assert p.evaluation_profit_target_pct == 8.0
        assert p.payout_frequency_days == 14
        assert "21" in p.source_note  # 21-day first-payout caveat documented
        assert p.consistency_rule_pct == 40.0
        assert p.funded_consistency_rule_pct == 40.0
        assert p.min_trading_days == 5
        assert p.winning_days_for_payout == 5
    assert p25.min_winning_day_profit == 100.0
    assert p100.min_winning_day_profit == 200.0


def test_the5ers_high_stakes_relabel():
    for key in ("the5ers_20k", "the5ers_100k"):
        p = get_preset(key)
        assert "High-Stakes" in p.label
        assert "Bootcamp" not in p.label
        # numbers unchanged: 8% / 5% / 10%-static / 30% / 3d
        assert (p.evaluation_profit_target_pct, p.daily_loss_limit_pct,
                p.max_drawdown_pct, p.drawdown_type,
                p.consistency_rule_pct, p.min_trading_days) == (8.0, 5.0, 10.0, "static", 30.0, 3)
    note = get_preset("the5ers_20k").source_note
    assert "Bootcamp" in note and "NOT modeled" in note


def test_lucid_funded_consistency():
    for key in ("lucid_50k", "lucid_100k"):
        p = get_preset(key)
        assert p.funded_consistency_rule_pct == 40.0
        assert p.consistency_rule_pct is None  # still no eval-stage rule


def test_topstep_gate_and_funded_consistency():
    for key in ("topstep_50k", "topstep_100k"):
        p = get_preset(key)
        assert p.winning_days_for_payout == 5
        assert p.min_winning_day_profit == 150.0
        assert p.funded_consistency_rule_pct == 40.0
