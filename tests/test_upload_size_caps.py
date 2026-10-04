"""Regression tests for the Data Center upload size caps (v4 Worker 3, Oct 2026).

Context: Owen hit the web app's ~240MB upload ceiling trying to ingest his
full ~755MB dataset through the SINGLE-FILE upload card. Investigation
showed the 256MB cap is per-request/per-file only -- every real data file
in data/raw/ at HEAD is under 32MB, and the folder importer POSTs one file
per request with no total cap -- so the fix keeps the 256MB cap but makes
it legible:

1. The single-file upload card now points large-dataset users at
   "Import folder" and documents the 256MB-per-file limit.
2. The single-file route (/data-center/import) now enforces the cap with
   the same two-layer checks as the folder endpoint (declared
   Content-Length, then a bounded read) instead of relying on Flask's bare
   HTML 413 page.
3. A global 413 handler (`_upload_too_large`) turns any remaining
   Flask-raised RequestEntityTooLarge into a clean redirect-with-notice
   (single-file form) or clean JSON 413 (everything else).

These tests exercise the real Flask routes. The 256MB cap is monkeypatched
down to 512KB so the tests stay fast -- the behavior under test is
identical, just at a smaller threshold.
"""
from __future__ import annotations

import io
import shutil

import pytest
from werkzeug.exceptions import RequestEntityTooLarge

from app.data import storage
from app.web import server as server_module
from app.web.server import app


TINY_CAP = 512 * 1024  # patched MAX_CONTENT_LENGTH for fast tests


@pytest.fixture(autouse=True)
def clean_raw_dir(tmp_path, monkeypatch):
    """Isolate every test here from the real data/raw/ directory (same
    pattern as tests/test_dataset_dropdown_and_parquet_upload.py)."""
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: tmp_path)
    raw_dir = storage.get_raw_data_dir()
    yield
    shutil.rmtree(raw_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def small_upload_cap(monkeypatch):
    """Shrink the upload cap to 512KB for the duration of each test so
    oversized/undersized behavior is exercisable without 256MB payloads."""
    monkeypatch.setitem(app.config, "MAX_CONTENT_LENGTH", TINY_CAP)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _csv_bytes(n_rows: int) -> bytes:
    lines = ["timestamp,open,high,low,close,volume"]
    for i in range(n_rows):
        lines.append(f"2024-01-01 00:{i // 60:02d}:{i % 60:02d},1.1,1.2,1.0,1.15,100")
    return ("\n".join(lines) + "\n").encode()


def _oversized_payload() -> bytes:
    """Deterministic payload just over the patched cap (plus slack for the
    multipart framing, which also counts toward Content-Length)."""
    return b"x" * (TINY_CAP + 64 * 1024)


# ---------------------------------------------------------------------------
# Folder import: file just under the cap must be accepted
# ---------------------------------------------------------------------------

def test_folder_import_accepts_file_just_under_cap(client):
    content = _csv_bytes(400)  # well under the 512KB patched cap
    assert len(content) < TINY_CAP
    resp = client.post(
        "/data-center/import-folder",
        data={"data_file": (io.BytesIO(content), "under_cap.csv"), "relpath": "fx/under_cap.csv"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True, data
    assert data["status"] == "imported", data
    # And it actually landed in data/raw/.
    assert (storage.get_raw_data_dir() / "under_cap.csv").exists()


# ---------------------------------------------------------------------------
# Folder import: oversized file -> clean JSON 413 (never a traceback page)
# ---------------------------------------------------------------------------

def test_folder_import_rejects_oversized_file_with_clean_413(client):
    resp = client.post(
        "/data-center/import-folder",
        data={"data_file": (io.BytesIO(_oversized_payload()), "too_big.csv"), "relpath": "fx/too_big.csv"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413
    data = resp.get_json()
    assert data is not None, "oversized upload must return JSON, not an HTML error page"
    assert data["ok"] is False
    assert "per-file limit" in data["error"]


# ---------------------------------------------------------------------------
# Single-file import: oversized file -> clean notice redirect (no bare 413)
# ---------------------------------------------------------------------------

def test_single_file_import_rejects_oversized_with_clean_notice(client):
    resp = client.post(
        "/data-center/import",
        data={"data_file": (io.BytesIO(_oversized_payload()), "too_big.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    body = resp.get_data(as_text=True)
    assert "Traceback" not in body
    assert "per-file limit" in body
    assert "Import folder" in body  # notice points at the folder path


def test_single_file_import_accepts_file_just_under_cap(client):
    content = _csv_bytes(400)
    assert len(content) < TINY_CAP
    resp = client.post(
        "/data-center/import",
        data={"data_file": (io.BytesIO(content), "ok.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    body = resp.get_data(as_text=True)
    assert "Traceback" not in body
    assert "Imported" in body
    assert (storage.get_raw_data_dir() / "ok.csv").exists()


# ---------------------------------------------------------------------------
# The 413 error handler itself: clean JSON for fetch endpoints, clean
# redirect for the single-file form -- covers the backstop path where
# Flask/Werkzeug raises before a route's own checks run.
# ---------------------------------------------------------------------------

def test_413_handler_redirects_single_file_form():
    with app.test_request_context("/data-center/import", method="POST"):
        resp = server_module._upload_too_large(RequestEntityTooLarge())
    assert resp.status_code == 302
    assert "/data-center?" in resp.headers["Location"]
    assert "notice=" in resp.headers["Location"]


def test_413_handler_returns_json_for_folder_endpoint():
    with app.test_request_context("/data-center/import-folder", method="POST"):
        resp, code = server_module._upload_too_large(RequestEntityTooLarge())
    assert code == 413
    payload = resp.get_json()
    assert payload["ok"] is False
    assert "per-file limit" in payload["error"]


def test_413_handler_returns_json_for_unknown_paths():
    with app.test_request_context("/some-other-route", method="POST"):
        resp, code = server_module._upload_too_large(RequestEntityTooLarge())
    assert code == 413
    assert resp.get_json()["ok"] is False
