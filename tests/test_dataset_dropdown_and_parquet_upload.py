"""
Regression tests for two related bugs reported against the web app:

1. The Search Lab (and, by omission of the same kwarg, most other tabs)
   never received `dataset_groups` from the Flask route, so their
   "Use a previously stored dataset" dropdown rendered with zero options
   even though data/raw/ had files in it. Evolution Lab had a flat-list
   fallback for the same missing kwarg, so it showed data but ungrouped.

2. Uploading a .parquet file through the web app's file input silently
   mis-read it as CSV, because import_csv_bytes() wrapped the upload in a
   bare io.BytesIO with no filename -- the importer's extension dispatch
   (app.data.importer._read_raw_file) needs a filename/suffix to know to
   use the parquet reader instead of the delimiter-guessing CSV reader.

These tests exercise the real Flask routes and the real importer, not
mocks, so they fail the same way the reported bugs did before the fix.
"""
from __future__ import annotations

import shutil

import pandas as pd
import pytest

from app.data import storage
from app.data.importer import import_csv_bytes
from app.web.server import app


@pytest.fixture(autouse=True)
def clean_raw_dir(tmp_path, monkeypatch):
    """Isolate every test here from the real data/raw/ directory (same
    pattern as tests/test_storage.py)."""
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: tmp_path)
    raw_dir = storage.get_raw_data_dir()
    yield
    shutil.rmtree(raw_dir, ignore_errors=True)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _seed_dataset(instrument: str, filename: str) -> None:
    """Writes a tiny valid OHLCV CSV under data/raw/<instrument>/<filename>,
    the same folder layout the desktop app and web importer both use to
    group datasets by instrument."""
    raw_dir = storage.get_raw_data_dir()
    folder = raw_dir / instrument
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01 00:00:00,1.1,1.2,1.0,1.15,100\n"
        "2024-01-01 00:05:00,1.15,1.25,1.05,1.2,120\n"
    )


# ---------------------------------------------------------------------------
# Bug 1: dataset_groups missing from route context
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "route,form_page",
    [
        ("/search", "search.html"),
        ("/evolution", "evolution.html"),
        ("/", "index.html"),
        ("/refine", "refine.html"),
        ("/full-pipeline", "full_pipeline.html"),
    ],
)
def test_market_data_dropdown_shows_grouped_dataset(client, route, form_page):
    _seed_dataset("EURUSD", "EURUSD5.csv")
    resp = client.get(route)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # The instrument name must appear as an <optgroup> label -- this is
    # exactly what was missing when dataset_groups wasn't passed to the
    # template (Search Lab showed nothing at all; other tabs showed a
    # flat, ungrouped list without the instrument name as a group header).
    assert 'optgroup label="EURUSD"' in body
    assert "EURUSD5.csv" in body


def test_evolution_multi_instrument_groups_checkboxes(client):
    _seed_dataset("XAUUSD", "XAUUSD15.csv")
    _seed_dataset("GBPUSD", "GBPUSD5.csv")
    resp = client.get("/evolution/multi-instrument")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "XAUUSD" in body
    assert "GBPUSD" in body
    assert "XAUUSD15.csv" in body


def test_search_multi_instrument_groups_checkboxes(client):
    _seed_dataset("XAUUSD", "XAUUSD15.csv")
    _seed_dataset("GBPUSD", "GBPUSD5.csv")
    resp = client.get("/search/multi-instrument")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'class="dataset-group-label"' in body
    assert "XAUUSD" in body and "GBPUSD" in body


# ---------------------------------------------------------------------------
# Bug 2: uploaded .parquet bytes lost their extension
# ---------------------------------------------------------------------------

def test_import_csv_bytes_reads_parquet_when_filename_given(tmp_path):
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:05:00"],
        "open": [1.1, 1.15],
        "high": [1.2, 1.25],
        "low": [1.0, 1.05],
        "close": [1.15, 1.2],
        "volume": [100, 120],
    })
    parquet_path = tmp_path / "sample.parquet"
    df.to_parquet(parquet_path)
    content = parquet_path.read_bytes()

    # Without a filename, this used to silently fall through to the CSV
    # reader on raw parquet bytes and fail (or produce garbage).
    result_no_name = import_csv_bytes(content)
    assert not result_no_name.is_valid

    # With the filename threaded through, it must dispatch to the parquet
    # reader and produce a valid 2-row OHLCV frame.
    result_with_name = import_csv_bytes(content, filename="sample.parquet")
    assert result_with_name.is_valid, result_with_name.errors
    assert len(result_with_name.dataframe) == 2


def test_parquet_upload_accepted_end_to_end(client):
    import numpy as np
    n = 300
    rng = np.random.default_rng(42)
    close = 1.10 + np.cumsum(rng.normal(0, 0.001, n))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open": close,
        "high": close + 0.001,
        "low": close - 0.001,
        "close": close,
        "volume": [100] * n,
    })
    import io as _io
    buf = _io.BytesIO()
    df.to_parquet(buf)
    buf.seek(0)

    resp = client.post(
        "/run",
        data={"csv_file": (buf, "EURUSD5.parquet")},
        content_type="multipart/form-data",
    )
    body = resp.get_data(as_text=True)
    # The point of this test is that the parquet bytes were correctly
    # recognized and parsed (not mis-read as CSV) -- a parse failure
    # surfaces as this specific notice regardless of what happens next
    # in the backtest itself.
    assert "Please upload at least one valid CSV" not in body
    assert "Could not parse" not in body
    assert "EURUSD5.parquet" in body


# ---------------------------------------------------------------------------
# Bug 1b: file picker accept attributes hid parquet/tsv/txt/archives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["/search", "/evolution", "/", "/refine"])
def test_upload_input_accepts_non_csv_market_data_formats(client, route):
    resp = client.get(route)
    body = resp.get_data(as_text=True)
    assert 'name="csv_file"' in body
    # Every occurrence of the market-data file input must advertise the
    # full set of importer-supported extensions, not just .csv.
    import re
    for m in re.finditer(r'name="csv_file"[^>]*accept="([^"]+)"', body):
        accept = m.group(1)
        for ext in (".csv", ".tsv", ".txt", ".parquet", ".zip", ".7z"):
            assert ext in accept, f"{route}: {ext} missing from accept={accept!r}"
