import shutil

import pandas as pd
import pytest

from app.data import health, storage


@pytest.fixture(autouse=True)
def clean_raw_dir(tmp_path, monkeypatch):
    """Isolate every test in this file from the real data/raw/ directory,
    mirroring tests/test_storage.py's own fixture."""
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: tmp_path)
    raw_dir = storage.get_raw_data_dir()
    yield
    shutil.rmtree(raw_dir, ignore_errors=True)


def _write_csv(instrument: str, filename: str, rows: list[dict]) -> None:
    raw_dir = storage.get_raw_data_dir()
    d = raw_dir / instrument
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / filename, index=False)


def _clean_5m_rows(n=40):
    base = pd.Timestamp("2026-01-05 09:30:00")
    rows = []
    for i in range(n):
        ts = base + pd.Timedelta(minutes=5 * i)
        rows.append({"timestamp": ts, "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100 + i, "volume": 10})
    return rows


def test_clean_file_reports_healthy():
    _write_csv("ES", "es_5m.csv", _clean_5m_rows())
    report = health.compute_data_center()
    assert report["total_files"] == 1
    assert report["total_unhealthy"] == 0
    es = next(g for g in report["instruments"] if g["instrument"] == "ES")
    assert es["file_count"] == 1
    f = es["files"][0]
    assert f["healthy"] is True
    assert f["bar_count"] == 40
    assert f["duplicate_count"] == 0
    assert f["gap_count"] == 0


def test_duplicate_timestamps_flagged():
    rows = _clean_5m_rows(25)
    rows.append(dict(rows[0]))  # exact duplicate timestamp
    _write_csv("ES", "es_dupe.csv", rows)
    report = health.compute_data_center()
    f = report["instruments"][0]["files"][0]
    assert f["duplicate_count"] == 1
    assert f["healthy"] is False
    assert any("duplicate" in i for i in f["issues"])


def test_large_gap_flagged():
    rows = _clean_5m_rows(30)
    # Blow a huge hole in the middle -- everything after index 15 jumps
    # forward by 3 days instead of the usual 5 minutes.
    shifted = []
    for i, r in enumerate(rows):
        r = dict(r)
        if i >= 15:
            r["timestamp"] = pd.Timestamp(r["timestamp"]) + pd.Timedelta(days=3)
        shifted.append(r)
    _write_csv("NQ", "nq_gap.csv", shifted)
    report = health.compute_data_center()
    f = report["instruments"][0]["files"][0]
    assert f["gap_count"] >= 1
    assert f["healthy"] is False
    assert f["largest_gap"] is not None


def test_empty_placeholder_file_flagged():
    raw_dir = storage.get_raw_data_dir()
    d = raw_dir / "GC"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gc_empty.csv").write_bytes(b"")
    report = health.compute_data_center()
    gc = next(g for g in report["instruments"] if g["instrument"] == "GC")
    assert gc["unhealthy_count"] == 1


def test_timeframe_availability_aggregated_per_instrument():
    _write_csv("ES", "es_5m.csv", _clean_5m_rows(30))
    base = pd.Timestamp("2026-01-05 09:30:00")
    daily_rows = [
        {"timestamp": base + pd.Timedelta(days=i), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
        for i in range(30)
    ]
    _write_csv("ES", "es_1d.csv", daily_rows)
    report = health.compute_data_center()
    es = next(g for g in report["instruments"] if g["instrument"] == "ES")
    assert es["file_count"] == 2
    assert len(es["timeframes_available"]) >= 1  # at least one timeframe inferred without crashing


def test_no_files_returns_empty_report():
    report = health.compute_data_center()
    assert report["total_files"] == 0
    assert report["instruments"] == []
