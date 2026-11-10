"""TradingView bridge: the settings ID round-trips, catches truncation, and lands in the HTML report."""
from __future__ import annotations

import pytest

from app.reports import tv_bridge as tv

SETTINGS = {"n": "Opening Range", "s": "ES", "tf": "15m", "rm": "percent", "rv": 0.5, "mt": 3, "dl": 2, "dd": 4, "ab": 50000, "ar": "balanced"}


def test_round_trip_keeps_every_field():
    back = tv.decode(tv.encode(SETTINGS))
    assert back["n"] == "Opening Range" and back["s"] == "ES" and back["rv"] == 0.5 and back["ab"] == 50000.0 and back["ar"] == "balanced"


def test_truncated_or_edited_ids_are_rejected():
    sid = tv.encode(SETTINGS)
    with pytest.raises(tv.SettingsIdError):
        tv.decode(sid[:-8])
    with pytest.raises(tv.SettingsIdError):
        tv.decode(sid.replace("rv=0.5", "rv=5"))
    with pytest.raises(tv.SettingsIdError):
        tv.decode("hello")


def test_separator_characters_in_a_name_cannot_corrupt_the_id():
    sid = tv.encode({**SETTINGS, "n": "A|B=C"})
    assert tv.decode(sid)["n"] == "A B C"


def test_empty_values_are_skipped_and_describe_matches_encode():
    sid = tv.encode({"n": "X", "s": None, "tf": ""})
    assert "s=" not in sid and "tf=" not in sid
    assert tv.describe(tv.decode(sid)) == [("Strategy", "X")]


def test_pine_script_lists_every_field_key_and_the_version():
    pine = tv.pine_indicator()
    assert pine.startswith("//@version=5") and tv.VERSION in pine
    for key, (label, _n) in tv._FIELDS.items():
        assert f'"{key}"' in pine and f'"{label}"' in pine
    assert "__" not in pine


def test_settings_from_report_reads_the_report_blocks():
    report = {"strategy": {"name": "S", "instrument": "NQ", "timeframe": "5m"},
              "risk_config": {"risk_mode": "percent", "risk_value": 1.0, "max_trades_per_day": 2, "initial_balance": 100000},
              "prop_firm_rules": {"daily_loss_limit_pct": 3, "max_drawdown_pct": 6, "profit_split_pct": 90}}
    got = tv.decode(tv.settings_id_for_report(report, "accelerated"))
    assert got["s"] == "NQ" and got["ab"] == 100000.0 and got["dl"] == 3.0 and got["ar"] == "accelerated"
