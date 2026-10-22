"""Defaults must charge real friction and agree on account size; explicit zeros stay honored."""
from __future__ import annotations

from datetime import date

from app.backtest.risk import (DEFAULT_ACCOUNT_SIZE, DEFAULT_COMMISSION_PER_TRADE, DEFAULT_SLIPPAGE_PIPS,
                               DEFAULT_SPREAD_PIPS, RiskConfig, commission_from_mapping)
from app.prop.presets import PROP_FIRM_PRESETS, get_preset, is_ready_eligible, rules_stamp_age_days
from app.prop.simulator import PropRules


def test_riskconfig_defaults_are_not_zero_friction():
    r = RiskConfig()
    assert r.commission_per_trade == DEFAULT_COMMISSION_PER_TRADE > 0
    assert r.spread_pips == DEFAULT_SPREAD_PIPS > 0
    assert r.slippage_pips == DEFAULT_SLIPPAGE_PIPS > 0


def test_account_size_is_consistent_between_risk_and_prop_rules():
    assert RiskConfig().initial_balance == PropRules().account_size == DEFAULT_ACCOUNT_SIZE


def test_explicit_zero_is_still_honored():
    r = RiskConfig(spread_pips=0.0, slippage_pips=0.0, commission_per_trade=0.0)
    assert (r.spread_pips, r.slippage_pips, r.commission_per_trade) == (0.0, 0.0, 0.0)


def test_commission_from_mapping():
    assert commission_from_mapping({}) == DEFAULT_COMMISSION_PER_TRADE
    assert commission_from_mapping({"commission": ""}) == DEFAULT_COMMISSION_PER_TRADE
    assert commission_from_mapping({"commission": "  "}) == DEFAULT_COMMISSION_PER_TRADE
    assert commission_from_mapping({"commission": "abc"}) == DEFAULT_COMMISSION_PER_TRADE
    assert commission_from_mapping({"commission": "0"}) == 0.0          # typed zero is honored
    assert commission_from_mapping({"commission": "4.5"}) == 4.5
    assert commission_from_mapping(None) == DEFAULT_COMMISSION_PER_TRADE


def test_verified_presets_carry_a_dated_primary_stamp():
    stamped = [p for p in PROP_FIRM_PRESETS if p.verified_on]
    assert len(stamped) >= 14
    today = date(2026, 10, 22)
    for p in stamped:
        assert p.source_url.startswith("https://")
        assert rules_stamp_age_days(p, today) == 0
        assert is_ready_eligible(p, today)
    assert not is_ready_eligible(get_preset("ftmo_100k"), date(2026, 12, 31)), "stale stamps must stop counting"


def test_unverified_presets_cannot_reach_ready():
    unstamped = [p for p in PROP_FIRM_PRESETS if not p.verified_on]
    assert unstamped, "the unverified rows are labelled honestly, not stamped"
    assert not any(is_ready_eligible(p, date(2026, 10, 22)) for p in unstamped)
