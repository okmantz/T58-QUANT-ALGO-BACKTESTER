"""Regression tests for the Oct-2026 data-layer audit fix (see
app/data/continuous_contract.py module docstring).

Bug being locked down: the legacy detector looked only for a >=2h session break,
so on the vendor NQ/ES/GC files it MISSED every real contract splice (they sit on
the first bar of a new date with no break) and ERASED ~40 genuine Sunday-open
price gaps per instrument.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.continuous_contract import (
    back_adjust_continuous,
    detect_splices_and_interleaves,
)

REPO = Path(__file__).resolve().parents[2]


def _minute_frame(start: str, bars: int, seed: int = 0, base: float = 5000.0, step: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=bars, freq="1min", tz="UTC")
    close = base + np.cumsum(rng.normal(0, step, bars))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, step * 0.3, bars)   # opens near (not exactly at) the prior close
    high = np.maximum(open_, close) + 0.25
    low = np.minimum(open_, close) - 0.25
    return pd.DataFrame(dict(timestamp=ts, open=open_, high=high, low=low, close=close, volume=100))


def _shift_from(df: pd.DataFrame, k: int, amount: float) -> pd.DataFrame:
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        out.loc[k:, c] += amount
    return out


def test_splice_on_first_bar_of_new_date_is_detected_and_removed():
    df = _minute_frame("2024-12-16 12:00", 3 * 24 * 60)
    k = int(np.argmax(df["timestamp"] >= pd.Timestamp("2024-12-17 00:00", tz="UTC")))
    df = _shift_from(df, k, 60.0)                                   # contiguous 1-minute bars, +60 pt splice
    rep = detect_splices_and_interleaves(df, roll_months=(3, 6, 9, 12), roll_days=range(4, 22))
    assert [r.index for r in rep.splice_rolls] == [k]
    assert abs(rep.splice_rolls[0].gap - 60.0) < 3
    adj, _ = back_adjust_continuous(df, report=rep)
    jumps = (adj["open"] - adj["close"].shift()).abs()
    assert jumps.max() < 5                                          # no tradable-looking gap left
    assert adj["close"].iloc[-1] == pytest.approx(df["close"].iloc[-1])   # newest prices untouched


def test_weekend_open_gap_is_not_a_roll():
    """The legacy detector erased exactly this (a real Sunday-open gap)."""
    df = _minute_frame("2024-12-13 20:00", 90)                       # Fri evening
    wk = _minute_frame("2024-12-15 23:00", 90, seed=1)               # Sun evening (49h break)
    wk[["open", "high", "low", "close"]] += df["close"].iloc[-1] + 40.0 - wk["open"].iloc[0]
    full = pd.concat([df, wk], ignore_index=True)
    rep = detect_splices_and_interleaves(full, roll_months=(3, 6, 9, 12), roll_days=range(4, 22))
    assert rep.rolls == []
    assert rep.ambiguous_weekend_gaps                                # but it is reported, not silently dropped
    adj, _ = back_adjust_continuous(full, report=rep)
    assert (adj["close"].to_numpy() == full["close"].to_numpy()).all()


def test_roll_calendar_filter_rejects_splice_outside_roll_months():
    df = _minute_frame("2024-02-12 12:00", 3 * 24 * 60)
    k = int(np.argmax(df["timestamp"] >= pd.Timestamp("2024-02-13 00:00", tz="UTC")))
    df = _shift_from(df, k, 60.0)
    assert detect_splices_and_interleaves(df, roll_months=(3, 6, 9, 12), roll_days=range(4, 22)).rolls == []
    assert len(detect_splices_and_interleaves(df).splice_rolls) == 1  # metals-style: any month


def test_other_contract_teleport_bar_is_repaired():
    """A bar printed from the other contract month: jumps +33 in, -33 out, almost no own range."""
    df = _minute_frame("2026-05-25 04:00", 6000, base=4500.0, step=0.4)   # long enough for a real jump scale
    k = 3000
    for c in ("open", "high", "low", "close"):
        df.loc[k, c] = df.loc[k - 1, "close"] + 33.0                # tiny-range bar at the other contract's level
    rep = detect_splices_and_interleaves(df)
    assert len(rep.interleave_events) == 2
    adj, _ = back_adjust_continuous(df, report=rep)
    assert (adj["open"] - adj["close"].shift()).abs().max() < 5


def test_real_news_spike_with_wide_bar_range_is_not_touched():
    df = _minute_frame("2026-08-12 12:00", 120, base=7400.0)
    k = 33
    df.loc[k, "open"] = df.loc[k - 1, "close"] + 11.0
    df.loc[k, "high"] = df.loc[k, "open"] + 2.0
    df.loc[k, "low"] = df.loc[k - 1, "close"] - 1.0                  # the bar travelled through the whole move
    df.loc[k, "close"] = df.loc[k, "open"]
    df.loc[k + 1, "open"] = df.loc[k, "close"] - 10.5                # and reverts next bar
    rep = detect_splices_and_interleaves(df)
    assert rep.rolls == []


@pytest.mark.skipif(not (REPO / "data/raw/NQ1!/futures_NQ.F_1m.parquet").exists(), reason="vendor file not in checkout")
def test_real_nq_rolls_found_and_real_gaps_left_alone():
    from app.data.importer import import_csv

    res = import_csv(str(REPO / "data/raw/NQ1!/futures_NQ.F_1m.parquet"))
    df = res.dataframe
    off = df["adj_offset"].to_numpy()
    roll_idx = np.flatnonzero(df["roll_bar"].to_numpy())
    gaps = {str(df["timestamp"].iloc[i].date()): off[i - 1] - off[i] for i in roll_idx}
    assert abs(gaps["2024-12-17"] - 296.75) < 1.0                    # the real Dec-2024 roll (missed by the legacy detector)
    assert abs(gaps["2023-06-12"] - 178.25) < 1.0
    assert "2022-06-21" not in gaps                                  # +172.5 across a data hole is a REAL move, not a roll
    assert not any(d.endswith(("-03-08", "-03-15")) and d.startswith("2020") for d in gaps)   # COVID weekend gaps are real
    j = (df["open"] - df["close"].shift()).abs()
    contiguous = df["timestamp"].diff().dt.total_seconds() <= 120
    assert j[contiguous & (df["timestamp"].dt.hour == 0) & (df["timestamp"].dt.minute == 0)].max() < 30


@pytest.mark.skipif(not (REPO / "data/raw/GC1!/futures_GC.F_1m.parquet").exists(), reason="vendor file not in checkout")
def test_real_gc_interleaved_window_is_cleaned():
    from app.data.importer import import_csv

    df = import_csv(str(REPO / "data/raw/GC1!/futures_GC.F_1m.parquet")).dataframe
    w = df[(df["timestamp"] >= "2026-05-25") & (df["timestamp"] <= "2026-05-27 02:00")]
    assert (w["open"] - w["close"].shift()).abs().max() < 6          # was ~34 points per flip before the fix
