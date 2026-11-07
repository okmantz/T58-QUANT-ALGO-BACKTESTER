"""Prop-account lifecycle simulation (app.prop.lifecycle): stage separation, censoring, clocks and the two objective modes."""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from app.prop.lifecycle import (
    DEFAULT_OBJECTIVE_MODE, OBJECTIVE_NET_PROFIT, OBJECTIVE_PROP, LifecycleConfig, canonical_trade_dates, evaluate_objective,
    kaplan_meier_survival, net_profit_objective, normalize_objective_mode, run_lifecycle, wilson_interval,
)
from app.prop.simulator import PropRules, precompute_day_structure


def _trades(win_rate=0.48, win=400.0, loss=-200.0, n=100, gap_days=1.4, seed=1, start="2026-01-05 14:30"):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    pnls = np.where(rng.random(n) < win_rate, win, loss)
    return [NS(pnl=float(p), entry_time=t0 + pd.Timedelta(days=int(i * gap_days)),
               exit_time=t0 + pd.Timedelta(days=int(i * gap_days), hours=2)) for i, p in enumerate(pnls)]


def _rules(**kw):
    base = dict(account_size=50_000, evaluation_profit_target_pct=6.0, daily_loss_limit_pct=2.0, max_drawdown_pct=4.0,
                drawdown_type="trailing", consistency_rule_pct=50.0, min_trading_days=5, payout_frequency_days=14)
    base.update(kw)
    return PropRules(**base)


def test_default_objective_mode_is_prop_and_unknown_values_fall_back_to_it():
    assert DEFAULT_OBJECTIVE_MODE == OBJECTIVE_PROP
    assert normalize_objective_mode(None) == OBJECTIVE_PROP
    assert normalize_objective_mode("garbage") == OBJECTIVE_PROP
    assert normalize_objective_mode("Net Profit") == OBJECTIVE_NET_PROFIT
    assert normalize_objective_mode("live") == OBJECTIVE_NET_PROFIT


def test_lifecycle_stages_are_separate_and_consistent():
    r = run_lifecycle(_trades(), _rules(), LifecycleConfig(n_simulations=400))
    e, f = r.evaluation, r.funded
    assert 0 <= e.p_pass <= 100 and 0 <= f.p_funded <= 100
    assert e.p_reach_target_before_breach >= e.p_pass - 1e-9            # you cannot pass without reaching the target first
    assert r.p_pass_and_first_payout <= e.p_pass + 1e-9                  # a payout needs a pass
    assert f.p_funded == pytest.approx(e.p_pass)                         # default: passing == funded
    # failure causes + passes + unresolved account for every simulated attempt
    total = e.p_pass + e.p_fail_daily_loss + e.p_fail_max_drawdown + e.p_fail_time_limit + e.p_fail_other + e.p_unresolved
    assert total == pytest.approx(100.0, abs=1e-6)
    # unconditional pass-within-N curve is monotone and bounded by P(pass)
    vals = [e.p_pass_within_days[k] for k in sorted(e.p_pass_within_days)]
    assert vals == sorted(vals) and vals[-1] <= e.p_pass + 1e-9


def test_time_to_target_reports_calendar_before_trading_days():
    r = run_lifecycle(_trades(gap_days=2.5, n=120), _rules(), LifecycleConfig(n_simulations=300))
    tt = r.evaluation.time_to_target
    assert tt["calendar"]["p50"] is not None and tt["trading"]["p50"] is not None
    # a strategy trading every ~2.5 days has far more calendar days than trading days: the "8 vs 180 days" discrepancy
    assert tt["calendar"]["p50"] > tt["trading"]["p50"]
    assert "CALENDAR" in r.clocks["days_to_pass"] and "TRADING" in r.clocks["legacy_days_to_pass"]


def test_funding_approval_probability_lowers_funded_but_not_pass():
    base = run_lifecycle(_trades(), _rules(), LifecycleConfig(n_simulations=400))
    gated = run_lifecycle(_trades(), _rules(), LifecycleConfig(n_simulations=400, funding_approval_probability=50.0))
    assert gated.evaluation.p_pass == pytest.approx(base.evaluation.p_pass)
    assert gated.funded.p_funded < base.evaluation.p_pass * 0.8


def test_losing_strategy_has_low_lifecycle_probability():
    bad = run_lifecycle(_trades(win_rate=0.30), _rules(), LifecycleConfig(n_simulations=300))
    good = run_lifecycle(_trades(win_rate=0.55), _rules(), LifecycleConfig(n_simulations=300))
    assert bad.p_pass_and_first_payout < good.p_pass_and_first_payout
    assert bad.evaluation.p_fail_max_drawdown > good.evaluation.p_fail_max_drawdown


def test_expected_net_includes_fee_and_split():
    free = run_lifecycle(_trades(), _rules(), LifecycleConfig(n_simulations=300, evaluation_fee=0.0, profit_split_pct=80.0))
    paid = run_lifecycle(_trades(), _rules(), LifecycleConfig(n_simulations=300, evaluation_fee=300.0, profit_split_pct=80.0))
    assert paid.expected_net_per_attempt == pytest.approx(free.expected_net_per_attempt - 300.0)
    assert paid.expected_attempts_to_first_payout is None or paid.expected_attempts_to_first_payout >= 1.0


def test_same_seed_is_reproducible():
    a = run_lifecycle(_trades(), _rules(), LifecycleConfig(n_simulations=200, random_seed=7))
    b = run_lifecycle(_trades(), _rules(), LifecycleConfig(n_simulations=200, random_seed=7))
    assert a.to_dict()["p_pass_and_first_payout"] == b.to_dict()["p_pass_and_first_payout"]


def test_kaplan_meier_treats_censored_accounts_as_unknown_not_alive():
    out = kaplan_meier_survival([10, 20, 30, 40], [True, False, True, False], [5, 15, 25, 35, 50])
    assert out[5] == (1.0, 4)
    assert out[15][0] == pytest.approx(0.75)
    assert out[35][0] == pytest.approx(0.375)
    assert out[50] == (None, 0)           # nobody observed that long: unknown, never 100%
    assert kaplan_meier_survival([], [], [30])[30] == (None, 0)


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(50, 100)
    assert 0 < lo < 50 < hi < 100
    assert wilson_interval(0, 0) == (0.0, 100.0)


def test_net_profit_mode_scores_without_prop_rules_and_prop_mode_falls_back_honestly():
    t = _trades(win_rate=0.6)
    net = evaluate_objective(t, None, "net_profit")
    assert net["mode"] == OBJECTIVE_NET_PROFIT and net["net_profit"] > 0 and net["n_trades"] == 100
    fb = evaluate_objective(t, None, "prop_lifecycle")
    assert "fallback" in fb                                      # no rules -> says so instead of inventing prop numbers
    prop = evaluate_objective(t, _rules(), "prop_lifecycle", LifecycleConfig(n_simulations=150))
    assert prop["mode"] == OBJECTIVE_PROP and "lifecycle" in prop and prop["objective"] > 0


def test_net_profit_objective_penalises_drawdown_and_handles_empty():
    assert net_profit_objective([])["objective"] == float("-inf")
    smooth = [NS(pnl=100.0, entry_time=pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)) for i in range(40)]
    lumpy = [NS(pnl=(1450.0 if i % 10 == 9 else -50.0), entry_time=pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)) for i in range(40)]
    assert net_profit_objective(smooth, 10_000)["net_profit"] == net_profit_objective(lumpy, 10_000)["net_profit"] == 4000.0
    assert net_profit_objective(smooth, 10_000)["max_drawdown"] < net_profit_objective(lumpy, 10_000)["max_drawdown"]


def test_canonical_dates_keep_the_time_of_day_so_session_days_match_the_direct_simulator():
    # 23:30 UTC is after the 17:00 CT roll -> belongs to the NEXT session day. Midnight-normalising (the old MC/survival behaviour)
    # put it on the previous one.
    t = [NS(entry_time=pd.Timestamp("2026-03-03 14:30"), pnl=1.0), NS(entry_time=pd.Timestamp("2026-03-03 23:30"), pnl=1.0)]
    rules = _rules()
    raw = precompute_day_structure(canonical_trade_dates(t), tz=rules.session_timezone, roll_hour=rules.session_roll_hour)
    old = precompute_day_structure([pd.Timestamp(x.entry_time).normalize() for x in t], tz=rules.session_timezone, roll_hour=rules.session_roll_hour)
    assert raw.n_days == 2 and old.n_days == 1


def test_zero_trades_raises():
    with pytest.raises(ValueError):
        run_lifecycle([], _rules())
