"""Audit 2026-10-12 independent tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
from app.search.robustness import deflated_sharpe_gate


def _null_false_positive_rate(units: str, n_sims=300, n_trades=150, trades_per_year=600, n_trials=50, seed=1):
    """Best of n_trials null strategies (zero-mean trade returns). Fraction of
    sims where the best one passes the 0.95 DSR gate."""
    rng = np.random.default_rng(seed)
    passes = 0
    for _ in range(n_sims):
        r = rng.normal(0, 1, size=(n_trials, n_trades))
        sr = r.mean(1) / r.std(1, ddof=1)               # per-trade Sharpe
        if units == "annualized":
            sr = sr * np.sqrt(trades_per_year)          # legacy (buggy) units
        best = float(sr.max())
        g = deflated_sharpe_gate(best, [float(x) for x in sr], n_trials, n_trades)
        passes += bool(getattr(g, "passed", getattr(g, "is_significant", False)))
    return passes / n_sims


def test_dsr_per_trade_units_control_false_positives():
    fixed = _null_false_positive_rate("per_trade")
    legacy = _null_false_positive_rate("annualized")
    print("DSR null false-positive rate  fixed=%.3f legacy=%.3f" % (fixed, legacy))
    assert fixed <= 0.10
    assert legacy > fixed


# --------------------------------------------------------------------------
# F-FUNDED: funded stage must start fresh on Topstep presets
# --------------------------------------------------------------------------
from app.prop.account import PropAccount
from app.prop.presets import get_preset


def _run(rules, pnls):
    acct = PropAccount(rules)
    out = []
    for day, pnl in enumerate(pnls):
        res = acct.on_trade_close(pnl, day, mae=min(pnl, 0.0), mfe=max(pnl, 0.0))
        out.append(res)
    return acct, out


def test_topstep_funded_account_starts_fresh_and_floor_is_2000_below_start():
    rules = get_preset("topstep_50k").to_prop_rules()
    # +1400/day x3 days, each <55% of total profit -> passes the 6% ($3,000) target
    acct, _ = _run(rules, [1100, 1100, 1100])
    assert acct.passed_evaluation and acct.stage == "funded"
    assert acct.balance == 50_000          # fresh XFA, eval profit not carried
    assert acct.floor() == 48_000          # full $2,000 MLL below the start

    # -1,950 stays alive ; a further -100 breaches 48,000
    r1 = acct.on_trade_close(-1950, 3, mae=-1950, mfe=0)
    assert not acct.failed
    r2 = acct.on_trade_close(-100, 4, mae=-100, mfe=0)
    assert acct.failed


def test_legacy_rules_carry_eval_profit_into_funded():
    rules = get_preset("topstep_50k").to_prop_rules()
    rules.funded_starts_fresh = False
    acct, _ = _run(rules, [1100, 1100, 1100])
    assert acct.balance == 53_300


def test_topstep_floor_is_fixed_dollar_and_locks_at_start():
    rules = get_preset("topstep_50k").to_prop_rules()
    acct = PropAccount(rules)
    acct.trailing_peak = 51_000
    assert acct.floor() == 49_000          # peak - $2,000 (not 4% of peak = 48,960)
    acct.trailing_peak = 60_000
    assert acct.floor() == 50_000          # locked at starting balance


# --------------------------------------------------------------------------
# F-ROLL: root-symbol filenames are continuous futures
# --------------------------------------------------------------------------
import os
import pytest
from app.data.importer import _looks_like_continuous_future, import_csv


@pytest.mark.parametrize("name,expected", [
    ("MNQ_1min_20260120_20260415.csv", True), ("ES_5min.csv", True), ("gc_1min.csv", True),
    ("ES1!_5.csv", True), ("ESH6_1min.csv", False), ("NQ_H26.csv", False),
    ("EURUSD_1min.csv", False), ("BTCUSD.csv", False), ("esport_data.csv", False),
])
def test_continuous_future_filename_detection(name, expected):
    assert _looks_like_continuous_future(name) is expected


_MNQ = "data/raw/MNQ1!/MNQ_1min_20260120_20260415.csv"


@pytest.mark.skipif(not os.path.exists(_MNQ), reason="MNQ file not present")
def test_mnq_contract_flip_is_repaired_on_import():
    df = import_csv(_MNQ).dataframe
    assert df.attrs.get("roll_adjusted") is True
    # raw file jumps -215/+224 around 2026-03-18 21:00-22:00 UTC (contract flip)
    import pandas as pd
    ts = pd.to_datetime(df["timestamp"])
    w = df[(ts >= "2026-03-18 20:30") & (ts <= "2026-03-18 22:30")]
    assert w["close"].diff().abs().max() < 60


def test_timezone_warning_flags_shifted_futures_data():
    import pandas as pd
    from app.data.importer import _futures_timezone_warning

    def mk(shift):
        t = pd.date_range("2026-01-05", "2026-02-20", freq="1min")
        t = t[~(((t.hour - shift) % 24) == 22)]
        return pd.DataFrame({"timestamp": t})

    ok, bad = [], []
    _futures_timezone_warning(mk(0), "ES_1min.csv", ok)
    _futures_timezone_warning(mk(-5), "ES_1min.csv", bad)
    assert ok == [] and len(bad) == 1


# --------------------------------------------------------------------------
# Free-data providers (mock only; NOT verified live)
# --------------------------------------------------------------------------
def test_free_sources_parse_documented_schemas():
    from app.data import free_sources as fs
    fred = fs.fetch_fred_series("VIXCLS", "k", lambda u, p: {"observations": [
        {"date": "2026-01-02", "value": "14.5"}, {"date": "2026-01-03", "value": "."}, {"date": "2026-01-06", "value": "16"}]})
    assert list(fred.values) == [14.5, 16.0]
    td = fs.fetch_twelve_data_ohlcv("SPY", "1min", "k", http_get=lambda u, p: {"values": [
        {"datetime": "2026-01-02 14:31:00", "open": "2", "high": "3", "low": "1", "close": "2.5", "volume": "10"},
        {"datetime": "2026-01-02 14:30:00", "open": "1", "high": "2", "low": "1", "close": "2", "volume": "5"}]})
    assert td["timestamp"].is_monotonic_increasing and len(td) == 2
    pg = fs.fetch_polygon_ohlcv("SPY", 1, "minute", "2026-01-02", "2026-01-02", "k", http_get=lambda u, p: {"results": [
        {"t": 1767364200000, "o": 1, "h": 2, "l": 1, "c": 2, "v": 3}]})
    assert pg["timestamp"].iloc[0] == pd.Timestamp("2026-01-02 14:30:00")
    import pytest as _p
    with _p.raises(fs.DataSourceError):
        fs.fetch_fred_series("X", "")


def test_saving_other_keys_does_not_wipe_new_provider_keys(tmp_path):
    from app.accounts import api_keys as ak
    ak.save_settings(ak.ApiKeysSettings(fred_api_key="a", twelve_data_api_key="b", databento_api_key="c"))
    import dataclasses
    cur = ak.load_settings()
    ak.save_settings(dataclasses.replace(cur, openai_api_key="z"))
    again = ak.load_settings()
    assert (again.fred_api_key, again.twelve_data_api_key, again.databento_api_key, again.openai_api_key) == ("a", "b", "c", "z")


def test_web_api_keys_page_and_save_roundtrip():
    from app.web.server import app as flask_app
    from app.accounts import api_keys as ak
    c = flask_app.test_client()
    r = c.get("/settings/api-keys")
    assert r.status_code == 200 and b"Twelve Data" in r.data
    c.post("/settings/api-keys/save-data", data={"tiingo_api_key": "TT", "fred_api_key": "FF"})
    c.post("/settings/api-keys/save-ai", data={"openai_api_key": "OO"})
    s = ak.load_settings()
    assert (s.tiingo_api_key, s.fred_api_key, s.openai_api_key) == ("TT", "FF", "OO")
