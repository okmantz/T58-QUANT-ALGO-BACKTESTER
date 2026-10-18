"""Regression tests for the market-data import structure and the dataset dropdown grouping.

Reported (Oct 2026): importing the market data (a) flattened every file into data/raw/, so the
dropdowns showed a header per timeframe ("EURUSD1", "EURUSD5" ...) and lumped every "futures_*" file
under one "FUTURES" header -- ES / NQ / GC 1-minute parquets looked missing -- and (b) the NVIDIA
file was refused outright. Run:  python -m pytest tests/data/test_instrument_folders.py -q
"""
from __future__ import annotations

import pytest

import app.data.storage as storage
from app.data import folder_import
from app.data.instrument_folders import (
    dataset_sort_key, instrument_folder_for_import, instrument_from_filename, picker_label,
    split_import_relpath, timeframe_label,
)

CSV_BODY = (
    "timestamp,open,high,low,close,volume\n"
    "2024-01-01 00:00,100,101,99,100,1000\n"
    "2024-01-02 00:00,100,102,99,101,1100\n"
    "2024-01-03 00:00,101,103,100,102,900\n"
)


@pytest.mark.parametrize("name,expected", [
    ("EURUSD1.csv", "EURUSD"), ("EURUSD1440.csv", "EURUSD"), ("EURUSD_5M_sample.csv", "EURUSD"),
    ("fx_EUR_USD_1m.parquet", "EURUSD"), ("AUD_USD_1m.parquet", "AUDUSD"),
    ("futures_ES.F_1m.parquet", "ES1!"), ("futures_NQ.F_1m.parquet", "NQ1!"),
    ("futures_GC.F_1m.parquet", "GC1!"), ("futures_SI.F_1m.parquet", "SI1!"),
    ("ES_5min_20260120_20260415.csv", "ES1!"), ("MGC_1m_2023-2026.parquet", "MGC1!"),
    ("nvda-1999-2014.csv", "NVIDIA"), ("USATECHIDXUSD15.csv", "NASDAQ 100"),
    ("NAS100_USD_1m.parquet", "NASDAQ 100"), ("USA500IDXUSD5.csv", "S&P 500"),
    ("XAU_USD_1m.parquet", "XAUUSD (GOLD)"), ("BTCUSD1440.csv", "BITCOIN"),
    ("ES1!_5m.csv", "ES1!"),
])
def test_known_instruments_resolve_to_their_folder(name, expected):
    assert instrument_from_filename(name) == (expected, True)
    assert picker_label(name) == expected


def test_every_timeframe_of_one_instrument_shares_one_header():
    labels = {picker_label(f) for f in (
        "EURUSD1.csv", "EURUSD5.csv", "EURUSD15.csv", "EURUSD60.csv", "EURUSD240.csv", "EURUSD1440.csv",
        "fx_EUR_USD_1m.parquet", "EURUSD_5M_sample.csv")}
    assert labels == {"EURUSD"}


def test_futures_and_fx_prefixes_no_longer_collapse():
    labels = {picker_label(f) for f in (
        "futures_ES.F_1m.parquet", "futures_NQ.F_1m.parquet", "futures_GC.F_1m.parquet")}
    assert labels == {"ES1!", "NQ1!", "GC1!"}


def test_unknown_names_keep_legacy_behaviour():
    assert picker_label("AAA_5M.csv") == "AAA"
    assert picker_label("") == "(UNGROUPED)"
    assert instrument_folder_for_import("notes.csv") == ""  # loose stays loose


def test_pack_wrapper_is_stripped():
    assert split_import_relpath("T58-Market-Data/data/raw/EURUSD/EURUSD1.csv") == (["EURUSD"], "EURUSD1.csv")
    assert split_import_relpath("raw/ES1!/x.csv") == (["ES1!"], "x.csv")
    assert split_import_relpath("EURUSD/EURUSD1.csv") == (["EURUSD"], "EURUSD1.csv")
    assert instrument_folder_for_import("pack/data/raw/S&P 500/USA500IDXUSD5.csv") == "S&P 500"


def test_existing_folder_with_different_case_is_reused():
    assert instrument_folder_for_import("EURUSD/EURUSD5.csv", ["eurusd"]) == "eurusd"


def test_timeframe_labels_and_sort_order():
    assert [timeframe_label(f) for f in (
        "EURUSD1.csv", "EURUSD5.csv", "EURUSD60.csv", "EURUSD240.csv", "EURUSD1440.csv",
        "fx_EUR_USD_1m.parquet", "XCU_USD_15s.parquet", "ES_5min_x.csv")] == [
        "1m", "5m", "1h", "4h", "1D", "1m", "15s", "5m"]
    names = ["EURUSD1440.csv", "EURUSD5.csv", "EURUSD1.csv", "EURUSD240.csv", "EURUSD15.csv", "EURUSD60.csv"]
    assert sorted(names, key=dataset_sort_key) == [
        "EURUSD1.csv", "EURUSD5.csv", "EURUSD15.csv", "EURUSD60.csv", "EURUSD240.csv", "EURUSD1440.csv"]


@pytest.fixture()
def raw_dir(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(storage, "get_raw_data_dir", lambda: raw)
    monkeypatch.setattr(storage, "_ROW_COUNT_CACHE", None)
    monkeypatch.setattr(storage, "_ROW_COUNT_CACHE_DIRTY", False)
    return raw


def test_folder_import_keeps_instrument_folders(raw_dir):
    files = [
        ("pack/data/raw/EURUSD/EURUSD1.csv", CSV_BODY.encode()),
        ("pack/data/raw/EURUSD/EURUSD5.csv", CSV_BODY.encode()),
        ("pack/data/raw/ES1!/ES_5min_20260120_20260415.csv", CSV_BODY.encode()),
        ("fx_GBP_USD_1m.csv", CSV_BODY.encode()),  # loose file -> filed under its instrument
        ("pack/DATA_README.txt", b"read me"),
    ]
    report = folder_import.import_uploaded_files(files)
    assert len(report.imported) == 4 and not report.failed
    assert [o.name for o in report.skipped] == ["pack/DATA_README.txt"]
    assert (raw_dir / "EURUSD" / "EURUSD1.csv").is_file()
    assert (raw_dir / "EURUSD" / "EURUSD5.csv").is_file()
    assert (raw_dir / "ES1!" / "ES_5min_20260120_20260415.csv").is_file()
    assert (raw_dir / "GBPUSD" / "fx_GBP_USD_1m.csv").is_file()
    assert not list(raw_dir.glob("*.csv"))  # nothing flattened into data/raw/ itself
    groups = {g["instrument"]: [f["name"] for f in g["files"]] for g in storage.list_datasets_grouped_for_picker()}
    assert groups == {"ES1!": ["ES_5min_20260120_20260415.csv"], "EURUSD": ["EURUSD1.csv", "EURUSD5.csv"],
                      "GBPUSD": ["fx_GBP_USD_1m.csv"]}


def test_reimport_is_idempotent_and_never_overwrites(raw_dir):
    folder_import.import_uploaded_files([("EURUSD/EURUSD1.csv", CSV_BODY.encode())])
    folder_import.import_uploaded_files([("EURUSD/EURUSD1.csv", CSV_BODY.encode())])
    assert [p.name for p in (raw_dir / "EURUSD").iterdir()] == ["EURUSD1.csv"]
    changed = CSV_BODY.replace("1000", "1001").encode()
    folder_import.import_uploaded_files([("EURUSD/EURUSD1.csv", changed)])
    assert sorted(p.name for p in (raw_dir / "EURUSD").iterdir()) == ["EURUSD1 (2).csv", "EURUSD1.csv"]


def test_legacy_flat_files_are_grouped_by_instrument(raw_dir):
    for n in ("EURUSD1.csv", "EURUSD5.csv", "futures_ES.F_1m.parquet", "futures_NQ.F_1m.parquet", "nvda-1999-2014.csv"):
        (raw_dir / n).write_text(CSV_BODY, encoding="utf-8")
    groups = {g["instrument"]: [f["name"] for f in g["files"]] for g in storage.list_datasets_grouped_for_picker()}
    assert groups == {"ES1!": ["futures_ES.F_1m.parquet"], "EURUSD": ["EURUSD1.csv", "EURUSD5.csv"],
                      "NQ1!": ["futures_NQ.F_1m.parquet"], "NVIDIA": ["nvda-1999-2014.csv"]}


def test_yahoo_style_file_with_unadjusted_close_imports():
    """Open/High/Low are split-adjusted but Close is raw -- the NVIDIA file's shape. It used to be
    refused ("2,171 of 4,012 rows dropped"); the Adj Close column now repairs it."""
    from app.data.importer import import_csv_bytes
    rows = ["Date,Open,High,Low,Close,Adj Close,Volume"]
    for day in range(1, 29):
        o = 1.0 + day * 0.01
        rows.append(f"2000-02-{day:02d},{o:.4f},{o + 0.1:.4f},{o - 0.05:.4f},{o * 12:.4f},{o + 0.02:.4f},1000")
    result = import_csv_bytes("\n".join(rows).encode(), filename="nvda-test.csv")
    assert result.is_valid
    assert len(result.dataframe) == 28
    assert any("Adj Close" in i.message for i in result.issues)
