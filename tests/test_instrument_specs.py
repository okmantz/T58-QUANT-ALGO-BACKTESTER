"""Tests for app.data.instrument_specs."""
from __future__ import annotations

import pytest

from app.backtest.risk import RiskConfig
from app.data.instrument_specs import (
    KNOWN_INSTRUMENTS,
    apply_instrument_spec,
    get_instrument_spec,
    known_instrument_symbols,
)


def test_known_symbols_include_owens_examples():
    symbols = known_instrument_symbols()
    for sym in ("MES", "MNQ", "MGC", "ES", "NQ", "GC"):
        assert sym in symbols


def test_get_instrument_spec_case_insensitive():
    assert get_instrument_spec("mnq") is get_instrument_spec("MNQ")
    assert get_instrument_spec("MNQ").contract_size == 2.0


def test_get_instrument_spec_unknown_returns_none():
    assert get_instrument_spec("NOTAREALSYMBOL") is None
    assert get_instrument_spec("") is None
    assert get_instrument_spec(None) is None


def test_owens_point_values_match_exactly():
    assert KNOWN_INSTRUMENTS["MES"].contract_size == 5.0
    assert KNOWN_INSTRUMENTS["MNQ"].contract_size == 2.0
    assert KNOWN_INSTRUMENTS["MGC"].contract_size == 10.0


def test_apply_instrument_spec_only_touches_pip_size_and_contract_size():
    risk = RiskConfig(initial_balance=25_000.0, risk_value=0.5, commission_per_trade=2.5)
    out = apply_instrument_spec(risk, "MNQ")
    assert out.pip_size == 1.0
    assert out.contract_size == 2.0
    assert out.initial_balance == risk.initial_balance
    assert out.risk_value == risk.risk_value
    assert out.commission_per_trade == risk.commission_per_trade
    # Original is untouched (replace() returns a copy).
    assert risk.pip_size != out.pip_size or risk.contract_size != out.contract_size


def test_apply_instrument_spec_unknown_symbol_raises_with_helpful_message():
    with pytest.raises(KeyError) as exc_info:
        apply_instrument_spec(RiskConfig(), "NOTAREALSYMBOL")
    assert "MNQ" in str(exc_info.value)  # names known symbols


def test_apply_instrument_spec_fills_commission_when_still_at_zero_default():
    """The 2026-09-24 upgrade: picking a known futures instrument also
    fills a realistic commission_per_trade, but ONLY when the caller left
    it at RiskConfig's own 0.0 default -- this is the exact gap a $0-
    commission Full Pipeline report (on ES) was found to have."""
    risk = RiskConfig()  # commission_per_trade defaults to 0.0
    out = apply_instrument_spec(risk, "ES")
    assert out.commission_per_trade == KNOWN_INSTRUMENTS["ES"].default_commission_round_turn
    assert out.commission_per_trade > 0.0


def test_apply_instrument_spec_never_overwrites_an_explicit_commission():
    risk = RiskConfig(commission_per_trade=2.5)
    out = apply_instrument_spec(risk, "ES")
    assert out.commission_per_trade == 2.5


def test_every_known_instrument_has_a_positive_default_commission():
    for symbol, spec in KNOWN_INSTRUMENTS.items():
        assert spec.default_commission_round_turn > 0.0, symbol


def test_guess_instrument_symbol_from_common_label_shapes():
    from app.data.instrument_specs import guess_instrument_symbol
    assert guess_instrument_symbol("ES1!/futures_ES.F_1m.parquet") == "ES"
    assert guess_instrument_symbol("futures_MES.F_1m.parquet") == "MES"
    assert guess_instrument_symbol("NQ1!") == "NQ"
    assert guess_instrument_symbol("MNQ_1m.csv") == "MNQ"


def test_guess_instrument_symbol_does_not_shadow_micro_with_full_size():
    from app.data.instrument_specs import guess_instrument_symbol
    # A naive substring match would find "ES" inside "MES" or "NQ" inside
    # "MNQ" -- longest-symbol-first + word-boundary matching must not do that.
    assert guess_instrument_symbol("futures_MES.F_1m.parquet") != "ES"
    assert guess_instrument_symbol("MNQ_1m.csv") != "NQ"


def test_guess_instrument_symbol_returns_none_when_unrecognized():
    from app.data.instrument_specs import guess_instrument_symbol
    assert guess_instrument_symbol("EURUSD.csv") is None
    assert guess_instrument_symbol("") is None
    assert guess_instrument_symbol(None) is None
