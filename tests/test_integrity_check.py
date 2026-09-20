import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy
from app.validation.integrity_check import (
    STATUS_BLOCKED,
    STATUS_VALID,
    run_integrity_check,
)


def _clean_1m_df(n=3000, seed=1, tz="America/Chicago"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02", periods=n, freq="1min", tz=tz)
    price = 100.0
    rows = []
    for i in range(n):
        step = rng.normal(0, 0.05)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.02))
        l = min(o, c) - abs(rng.normal(0, 0.02))
        rows.append((ts[i], o, h, l, c, 10.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config():
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def test_empty_dataframe_is_blocked():
    report = run_integrity_check(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), None, RiskConfig())
    assert report.status == STATUS_BLOCKED
    assert "No usable market data" in report.blocking_reason
    assert "No performance results were produced" in report.render()


def test_clean_data_no_strategy_no_resample_is_valid():
    df = _clean_1m_df()
    report = run_integrity_check(df, None, RiskConfig(initial_balance=50_000), prop_rules=PropRules())
    assert report.status == STATUS_VALID
    assert report.sections["DATA"][2].detail == f"{len(df):,}"
    assert "STRATEGY" not in report.sections  # no strategy given
    rendered = report.render()
    assert "T58 BACKTEST INTEGRITY CHECK" in rendered
    assert "BACKTEST STATUS: VALID" in rendered


def test_duplicate_timestamps_are_flagged_but_not_blocking():
    df = _clean_1m_df(n=500)
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    report = run_integrity_check(df, None, RiskConfig())
    assert report.status == STATUS_VALID
    dup_line = next(l for l in report.sections["DATA"] if l.label == "Duplicate timestamps")
    assert dup_line.ok is False
    assert dup_line.detail == "1"


def test_heavily_corrupt_ohlc_is_blocked():
    df = _clean_1m_df(n=500)
    df.loc[: int(len(df) * 0.5), "high"] = df.loc[: int(len(df) * 0.5), "low"] - 1.0  # high < low for half the bars
    report = run_integrity_check(df, None, RiskConfig())
    assert report.status == STATUS_BLOCKED
    assert "Corrupt market data" in report.blocking_reason


def test_requesting_finer_timeframe_than_native_is_blocked():
    df = _clean_1m_df(n=500, seed=2)
    df["timestamp"] = pd.date_range("2024-01-02", periods=len(df), freq="5min", tz="America/Chicago")  # native = 5m
    report = run_integrity_check(df, None, RiskConfig(), requested_timeframe="1m")
    assert report.status == STATUS_BLOCKED
    assert "Resampling: FAILED" in report.blocking_detail


def test_requesting_coarser_timeframe_resamples_successfully():
    df = _clean_1m_df(n=3000)  # native = 1m, ~50 hours
    report = run_integrity_check(df, None, RiskConfig(), requested_timeframe="15m")
    assert report.status == STATUS_VALID
    tf_lines = {l.label: l for l in report.sections["TIMEFRAME"]}
    assert tf_lines["Requested"].detail == "15m"
    assert tf_lines["Source"].detail == "1m"
    assert int(tf_lines["15m bars created"].detail.replace(",", "")) > 0


def test_strategy_section_included_and_lookahead_pass_for_clean_strategy():
    df = _clean_1m_df(n=1000)
    strategy = ManualStrategy(_sma_config())
    report = run_integrity_check(df, strategy, RiskConfig())
    assert report.status == STATUS_VALID
    assert "STRATEGY" in report.sections
    lookahead_line = next(l for l in report.sections["STRATEGY"] if l.label == "Lookahead protection")
    assert lookahead_line.ok is True


def test_account_section_reflects_risk_and_prop_rules():
    df = _clean_1m_df(n=200)
    report = run_integrity_check(df, None, RiskConfig(initial_balance=25_000), prop_rules=PropRules())
    account = {l.label: l for l in report.sections["ACCOUNT"]}
    assert account["Starting balance"].detail == "$25,000"
    assert account["Prop rules"].detail == "Loaded"


def test_to_dict_round_trips_status_and_text():
    df = _clean_1m_df(n=200)
    report = run_integrity_check(df, None, RiskConfig())
    d = report.to_dict()
    assert d["status"] == STATUS_VALID
    assert "DATA" in d["sections"]
    assert "T58 BACKTEST INTEGRITY CHECK" in d["text"]
