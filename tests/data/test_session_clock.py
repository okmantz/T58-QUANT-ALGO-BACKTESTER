"""P0 regression tests (report 2026-10-20): the exchange clock.

Vendor bars are UTC; every session / opening-range / gap / VWAP / event operand means EXCHANGE time. These tests pin the
conversion on synthetic frames AND on the real bundled NQ / ES files, so a regression to "UTC read as exchange time"
(45% of NQ volume inside RTH instead of 77%) fails loudly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data import session_clock as sc
from app.data.event_calendar import FOMC_DECISION_DATES
from app.data.trading_day import trading_day

ROOT = Path(__file__).resolve().parents[2]
NQ_PARQUET = ROOT / "data" / "raw" / "NQ1!" / "futures_NQ.F_1m.parquet"
ES_PARQUET = ROOT / "data" / "raw" / "ES1!" / "futures_ES.F_1m.parquet"


def _load_stamped(path: Path, symbol: str) -> pd.DataFrame:
    pytest.importorskip("pyarrow")
    df = pd.read_parquet(path).rename(columns={"ts": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)     # naive UTC, as the importer produces
    df.attrs["source_timezone"] = "UTC"
    return sc.stamp_session_columns(df, symbol=symbol)


@pytest.fixture(scope="module")
def nq():
    if not NQ_PARQUET.exists():
        pytest.skip("bundled NQ parquet not present")
    return _load_stamped(NQ_PARQUET, "NQ")


@pytest.fixture(scope="module")
def es():
    if not ES_PARQUET.exists():
        pytest.skip("bundled ES parquet not present")
    return _load_stamped(ES_PARQUET, "ES")


# ------------------------------------------------------------------------------------------------ conversion units
def test_utc_to_exchange_is_dst_aware():
    t = pd.Series(pd.to_datetime(["2026-01-15 14:30", "2026-07-15 13:30", "2026-03-08 14:30", "2026-11-01 14:30"]))
    out = sc.to_exchange_wallclock(t, zone=sc.NY)
    assert out.iloc[0] == pd.Timestamp("2026-01-15 09:30")          # EST: UTC-5
    assert out.iloc[1] == pd.Timestamp("2026-07-15 09:30")          # EDT: UTC-4
    assert out.iloc[2] == pd.Timestamp("2026-03-08 10:30")          # US spring-forward day: 14:30Z is already EDT
    assert out.iloc[3] == pd.Timestamp("2026-11-01 09:30")          # fall-back day: 14:30Z is EST


def test_symbol_profiles():
    assert sc.session_profile("NQ1!").zone == sc.NY and sc.session_profile("NQ1!").kind == "equity_index"
    assert sc.session_profile("MES").roll_tz == sc.CHI and sc.session_profile("MES").roll_hour == 17
    assert sc.session_profile("GC").zone == sc.NY
    assert sc.session_profile("EURUSD").kind == "fx" and sc.session_profile("EURUSD").zone == sc.NY
    assert sc.session_profile("BITCOIN").zone == "UTC"
    assert sc.infer_symbol("data/raw/NASDAQ 100/x.csv") == "NAS100"
    assert sc.infer_symbol("data/raw/NQ1!/futures_NQ.F_1m.parquet") == "NQ"
    # unknown markets fall back to the New York convention, never to UTC-by-accident
    assert sc.session_profile("ZZZZ").zone == sc.NY


def test_trading_day_matches_the_existing_17ct_roll():
    ts = pd.to_datetime(["2026-04-15 21:59", "2026-04-15 22:00", "2026-01-15 22:59", "2026-01-15 23:00"])
    df = pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    sc.stamp_session_columns(df, symbol="ES")
    expect = [pd.Timestamp(trading_day(t)) for t in ts]
    assert list(df["trading_day"]) == expect
    assert df["trading_day"].iloc[0] == pd.Timestamp("2026-04-15") and df["trading_day"].iloc[1] == pd.Timestamp("2026-04-16")


def test_frames_without_provenance_keep_legacy_reading():
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-05 09:30", periods=3, freq="1min"), "close": 1.0})
    assert not sc.has_clock_provenance(df)
    assert list(sc.frame_exchange_time(df)) == list(df["timestamp"])          # naive == exchange wall-clock
    df.attrs["source_timezone"] = "UTC"
    assert sc.frame_exchange_time(df).iloc[0] == pd.Timestamp("2026-01-05 04:30")


def test_importer_stamps_exchange_columns_and_warns_on_bad_clock(tmp_path):
    from app.data.importer import import_csv
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2026-01-05", periods=40)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h, minutes=m) for d in days for h in range(0, 24) for m in (0, 15, 30, 45)])
    ex = sc.to_exchange_wallclock(pd.Series(idx), source_tz="UTC", zone=sc.NY)
    mod = (ex.dt.hour * 60 + ex.dt.minute).to_numpy()
    good_vol = np.where((mod >= 570) & (mod < 960), 4000, 300) + rng.integers(0, 50, len(idx))
    px = 15000 + np.cumsum(rng.normal(0, 3, len(idx)))

    def write(name: str, vol) -> Path:
        p = tmp_path / name
        pd.DataFrame({"timestamp": idx, "open": px, "high": px + 2, "low": px - 2, "close": px + 0.5, "volume": vol}).to_csv(p, index=False)
        return p

    res = import_csv(write("NQ_15m.csv", good_vol), drop_trailing_partial_bar=False)
    assert res.is_valid
    df = res.dataframe
    assert {"exchange_time", "trading_day"} <= set(df.columns)
    assert df.attrs["source_timezone"] == "UTC" and df.attrs["exchange_timezone"] == sc.NY
    assert not [i for i in res.issues if "Session clock" in i.message]

    # volume that is busiest when the UTC stamps read 09:30-16:00 means the file is NOT UTC -> import-time warning
    utc_mod = (idx.hour * 60 + idx.minute).to_numpy()
    bad_vol = np.where((utc_mod >= 570) & (utc_mod < 960), 4000, 300) + rng.integers(0, 50, len(idx))
    bad = import_csv(write("NQ_bad_15m.csv", bad_vol), drop_trailing_partial_bar=False)
    assert any("Session clock" in i.message and "09:30-16:00" in i.message for i in bad.issues)


# --------------------------------------------------------------------------- real bundled data (the report's three tests)
def test_nq_open_spike_lands_on_0930_et(nq):
    """The morning volume peak is the 09:30 ET open and dwarfs the bar before it (a 1h or DST misalignment would fail this).

    NQ's busiest single minute overall is the 15:59 close, so the open is tested inside the 04:00-12:00 ET window."""
    peak = sc.peak_volume_minute(nq, start_minute=4 * 60, end_minute=12 * 60)
    assert peak == 9 * 60 + 30, f"morning volume peak is at {peak // 60:02d}:{peak % 60:02d} ET, expected 09:30"
    assert sc.minute_volume(nq, 9 * 60 + 30) > 3 * sc.minute_volume(nq, 9 * 60 + 29)


def test_nq_open_spike_is_dst_stable(nq):
    """Winter and summer must both put the open spike on 09:30 (a fixed UTC offset drifts an hour twice a year)."""
    ex = nq["exchange_time"]
    for months in ({12, 1, 2}, {6, 7, 8}):
        sub = nq[ex.dt.month.isin(months)]
        assert sc.peak_volume_minute(sub, start_minute=4 * 60, end_minute=12 * 60) == 9 * 60 + 30, months


@pytest.mark.parametrize("fixture_name", ["nq", "es"])
def test_equity_index_rth_volume_share_above_65pct(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    share = sc.rth_volume_share(df)
    assert share is not None and share > sc.MIN_RTH_VOLUME_SHARE, share
    assert sc.clock_quality_issues(df, fixture_name.upper()) == []


def test_utc_misread_as_exchange_time_is_caught(nq):
    """The exact bug the report found: UTC stamps read as exchange wall-clock gives ~45% RTH share and a warning."""
    legacy = nq[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    legacy.attrs = {}                                    # no provenance: stamps are read as exchange wall-clock
    share = sc.rth_volume_share(legacy)
    assert share < sc.MIN_RTH_VOLUME_SHARE
    assert any("09:30-16:00" in m for _, m in sc.clock_quality_issues(legacy, "NQ"))
    with pytest.raises(ValueError):
        sc.assert_clock_ok(legacy, "NQ")


def test_fomc_1400_et_lands_on_the_1400_et_bar(nq):
    day = pd.Timestamp("2025-01-29")
    assert day.strftime("%Y-%m-%d") in FOMC_DECISION_DATES
    ev = pd.Timestamp("2025-01-29 14:00")
    ex = nq["exchange_time"]
    sub = nq[(ex >= "2025-01-29 12:00") & (ex <= "2025-01-29 16:00")]
    assert len(sub) > 100
    label = sc.event_bar_timestamps([ev], sub).iloc[0]
    assert sub.loc[label, "exchange_time"] == ev
    # and through the real operand: minutes_since_fomc is 0 on that bar and 1 on the next
    from app.strategy import indicators_v11 as iv
    f = sub.copy()
    f.attrs = dict(nq.attrs)
    msf = iv.V11_INDICATOR_BUILDERS["minutes_since_fomc"](f, 14, "close", 14)
    assert float(msf.loc[label]) == 0.0
    assert float(msf.loc[label + 1]) == 1.0


def test_session_operands_read_exchange_time(nq):
    """minutes_since_open is 0 at 09:30 ET on real UTC-stamped bars (it was +270/+330 on the old clock)."""
    from app.strategy import indicators_v11 as iv
    day = nq[(nq["exchange_time"] >= "2025-03-12 08:00") & (nq["exchange_time"] <= "2025-03-12 12:00")].copy()
    day.attrs = dict(nq.attrs)
    mso = iv.V11_INDICATOR_BUILDERS["minutes_since_open"](day, 14, "close", 14)
    at_open = mso[day["exchange_time"] == pd.Timestamp("2025-03-12 09:30")]
    assert len(at_open) == 1 and float(at_open.iloc[0]) == 0.0


def test_manual_time_of_day_defaults_to_exchange_zone(nq):
    """An unqualified 09:30-10:00 window on UTC-stamped NQ bars is 13:30-14:00 UTC in summer (EDT), 14:30-15:00 in winter."""
    from app.strategy.manual import ManualStrategy
    strat = ManualStrategy({"name": "t"})
    op = {"type": "time_of_day", "session_start": "09:30", "session_end": "10:00"}
    for day, first_utc in (("2025-07-15", "13:30"), ("2025-01-15", "14:30")):
        d = nq[(nq["exchange_time"] >= f"{day} 00:00") & (nq["exchange_time"] < f"{pd.Timestamp(day) + pd.Timedelta(days=1)}")].copy()
        d = d.reset_index(drop=True)
        d.attrs = dict(nq.attrs)
        ts = strat._operand_local_ts(d["timestamp"], op, d)
        t = ts.dt.time
        sel = (t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("10:00").time())
        hit = d.loc[sel.to_numpy(), "timestamp"]
        assert len(hit) == 31
        assert hit.iloc[0].strftime("%H:%M") == first_utc
    # an explicit operand timezone still wins ("UTC" restores the old literal reading)
    d.attrs = dict(nq.attrs)
    utc_ts = strat._operand_local_ts(d["timestamp"], {"timezone": "UTC"}, d)
    assert utc_ts.dt.hour.iloc[0] == d["timestamp"].dt.hour.iloc[0]
