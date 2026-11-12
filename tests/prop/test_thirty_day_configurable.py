"""honest_eval configurability: no time limit, DD type, target, daily loss, min days, funded-phase payouts."""
import numpy as np
import pandas as pd
import pytest

from app.prop.presets import get_preset
from app.prop.thirty_day import ThirtyDayConfig, TradeLedger, apply_rule_overrides, evaluate_thirty_day
from tests.prop.test_thirty_day import _ledger, _rules, _span


def test_no_time_limit_passes_a_slow_winner_that_the_30_day_clock_rejects():
    led = _ledger([300.0] * 80, every_days=3)
    s, e = _span(led, 0)
    clock = evaluate_thirty_day(led, _rules(), data_start=s, data_end=e, with_sensitivity=False)
    free = evaluate_thirty_day(led, _rules(), data_start=s, data_end=e, with_sensitivity=False,
                               cfg=ThirtyDayConfig(no_time_limit=True, no_limit_horizon_days=120))
    assert clock.p_pass_30 < 0.2
    assert free.p_pass_30 > clock.p_pass_30


def test_overrides_convert_dollars_to_percent_and_set_dd_type():
    r = apply_rule_overrides(_rules(), ThirtyDayConfig(dd_type="static", target_usd=2500, max_dd_usd=3000,
                                                      daily_loss_usd=0, min_days=3))
    assert r.drawdown_type == "static"
    assert r.evaluation_profit_target_pct == pytest.approx(5.0)
    assert r.max_drawdown_pct == pytest.approx(6.0)
    assert r.daily_loss_limit_pct == 100.0
    assert r.min_trading_days == 3


def test_larger_drawdown_distance_helps_a_choppy_ledger():
    pn = [500.0, -450.0] * 60
    led = _ledger(pn)
    s, e = _span(led, 0)
    tight = evaluate_thirty_day(led, _rules(), data_start=s, data_end=e, with_sensitivity=False,
                                cfg=ThirtyDayConfig(max_dd_usd=500))
    wide = evaluate_thirty_day(led, _rules(), data_start=s, data_end=e, with_sensitivity=False,
                               cfg=ThirtyDayConfig(max_dd_usd=4000))
    assert wide.p_pass_30 >= tight.p_pass_30


def test_larger_dd_type_is_the_better_of_trailing_and_static():
    led = _ledger([400.0, -350.0] * 60)
    s, e = _span(led, 0)
    kw = dict(data_start=s, data_end=e, with_sensitivity=False)
    both = evaluate_thirty_day(led, _rules(), cfg=ThirtyDayConfig(dd_type="larger"), **kw)
    tr = evaluate_thirty_day(led, _rules(), cfg=ThirtyDayConfig(dd_type="trailing"), **kw)
    st = evaluate_thirty_day(led, _rules(), cfg=ThirtyDayConfig(dd_type="static"), **kw)
    assert both.p_pass_30 == max(tr.p_pass_30, st.p_pass_30)


def test_funded_phase_reports_payouts_for_a_steady_winner_and_none_for_a_loser():
    win = _ledger([400.0] * 200)
    s, e = _span(win, 0)
    r = evaluate_thirty_day(win, _rules(), data_start=s, data_end=e, with_sensitivity=False,
                            cfg=ThirtyDayConfig(funded_horizon_days=90))
    assert r.funded_horizon_days == 90
    assert r.p_paid > 0 and r.mean_payout_usd > 0
    lose = _ledger([-300.0] * 80)
    s, e = _span(lose, 0)
    r2 = evaluate_thirty_day(lose, _rules(), data_start=s, data_end=e, with_sensitivity=False)
    assert r2.p_paid == 0.0 and r2.mean_payout_usd == 0.0
