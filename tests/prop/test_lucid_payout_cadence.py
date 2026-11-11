"""Lucid's payout cadence is 5 trading days (owner-verified against Lucid's payout page, 2026-11-11); it was modelled as 3."""
import pytest

from app.prop.presets import get_preset


@pytest.mark.parametrize("key", ["lucid_25k", "lucid_50k", "lucid_100k", "lucid_flex_25k", "lucid_flex_50k", "lucid_flex_100k"])
def test_every_lucid_preset_pays_out_on_a_five_trading_day_cycle(key):
    p = get_preset(key)
    assert p.payout_frequency_days == 5
    assert p.to_prop_rules().payout_frequency_days == 5


def test_the_lucid_50k_account_state_machine_respects_the_five_day_gap():
    import pandas as pd
    from app.prop.simulator import simulate_account
    rules = get_preset("lucid_50k").to_prop_rules()
    days = pd.bdate_range("2026-01-05", periods=40)
    times = [d + pd.Timedelta(hours=15) for d in days]
    res = simulate_account([400.0] * len(times), times, rules)
    assert res.passed_evaluation
    gaps = [b.day_index - a.day_index for a, b in zip(res.payouts, res.payouts[1:]) if hasattr(a, "day_index")]
    assert all(g >= 5 for g in gaps)
