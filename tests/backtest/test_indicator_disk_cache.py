"""Shared indicator disk tier: keyed by a content hash of the data, readable by any process, safe against stale values."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy import indicator_cache as ic
from app.strategy.indicators import build_indicator_series


@pytest.fixture(autouse=True)
def disk(tmp_path, monkeypatch):
    monkeypatch.setenv("T58_INDICATOR_DISK_CACHE", "1")
    monkeypatch.setenv("T58_INDICATOR_CACHE_DIR", str(tmp_path / "ind"))
    monkeypatch.setattr(ic, "_DISK_MIN_ROWS", 100)
    ic.clear()
    yield tmp_path / "ind"
    ic.clear()


def _frame(n=500, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
                         "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 10.0})


def test_a_second_process_reads_what_the_first_wrote(disk):
    df = _frame()
    first = build_indicator_series(df, "ema", 20)
    assert ic.stats()["disk_writes"] == 1 and list(disk.rglob("*.npy"))
    ic.clear()                                   # a fresh process: empty memory tier, same disk
    again = build_indicator_series(df.copy(), "ema", 20)
    assert ic.stats()["disk_hits"] >= 1
    pd.testing.assert_series_equal(first, again)


def test_different_data_or_params_never_share_an_entry():
    a, b = _frame(seed=1), _frame(seed=2)
    ea, eb = build_indicator_series(a, "ema", 20), build_indicator_series(b, "ema", 20)
    ic.clear()
    assert not np.allclose(ea.to_numpy(), eb.to_numpy())
    pd.testing.assert_series_equal(build_indicator_series(b, "ema", 20), eb)
    assert not np.allclose(build_indicator_series(a, "ema", 50).to_numpy(), ea.to_numpy())


def test_a_prefix_slice_is_not_served_the_parents_values():
    df = _frame()
    full = build_indicator_series(df, "ema", 20)
    ic.clear()
    prefix = build_indicator_series(df.iloc[:300].reset_index(drop=True), "ema", 20)
    assert len(prefix) == 300
    np.testing.assert_allclose(prefix.to_numpy(), full.to_numpy()[:300])      # right answer, computed for the prefix itself


def test_small_frames_stay_off_disk(disk, monkeypatch):
    monkeypatch.setattr(ic, "_DISK_MIN_ROWS", 10_000)
    build_indicator_series(_frame(), "ema", 20)
    assert not list(disk.rglob("*.npy")) if disk.exists() else True


def test_disabled_switch_and_corrupt_files_are_just_misses(disk, monkeypatch):
    df = _frame()
    expected = build_indicator_series(df, "ema", 20)
    for f in disk.rglob("*.npy"):
        f.write_bytes(b"garbage")
    ic.clear()
    pd.testing.assert_series_equal(build_indicator_series(df, "ema", 20), expected)
    monkeypatch.setenv("T58_INDICATOR_DISK_CACHE", "0")
    ic.clear()
    before = ic.stats()["disk_writes"]
    build_indicator_series(_frame(seed=9), "ema", 20)
    assert ic.stats()["disk_writes"] == before
