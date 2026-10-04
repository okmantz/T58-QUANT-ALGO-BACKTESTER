"""Regression test: the web app's dataset dropdown groups datasets by
instrument folder (one optgroup per data/raw/<INSTRUMENT>/), and every
grouped entry stays selectable.

v2 (Oct 2026, commit d1a7228) introduced resolve_stored_dataset(), which
sanitized the whole submitted name down to its basename. The dropdown
submits the POSIX-style relative path list_stored_datasets() reported
(e.g. "NQ1!/NQ1!_2024.csv"), so every dataset living in an instrument
subfolder -- i.e. everything the grouped dropdown shows -- resolved to
None and could never be selected. The basename-flattening ("flat")
pre-fix implementation fails the resolve assertions below; the fixed
segment-preserving resolver passes them.

Run:  python -m pytest tests/test_dataset_dropdown_grouping.py -q
"""
from __future__ import annotations

import pytest

import app.data.storage as storage

CSV_BODY = (
    "timestamp,open,high,low,close,volume\n"
    "2024-01-01 00:00,100,101,99,100,1000\n"
    "2024-01-02 00:00,100,102,99,101,1100\n"
)

# Mirrors the real bundle layout -- folder names carry characters like
# "!" that secure_filename() would strip, which is exactly what made the
# v2 read path unable to resolve them even segment-by-segment.
LAYOUT = {
    "NQ1!": ["NQ1!_2024-01.csv", "NQ1!_2024-02.csv"],
    "MGC1!": ["MGC1!_2024-01.csv"],
    "ES1!": ["ES1!_2024-01.csv"],
}


@pytest.fixture()
def instrument_tree(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    for instrument, files in LAYOUT.items():
        folder = raw / instrument
        folder.mkdir(parents=True)
        for fname in files:
            (folder / fname).write_text(CSV_BODY, encoding="utf-8")
    # One loose file at the top level -> the "(ungrouped)" bucket.
    (raw / "loose.csv").write_text(CSV_BODY, encoding="utf-8")
    monkeypatch.setattr(storage, "get_raw_data_dir", lambda: raw)
    # Isolate the module-global row-count cache from the real data dir.
    monkeypatch.setattr(storage, "_ROW_COUNT_CACHE", None)
    monkeypatch.setattr(storage, "_ROW_COUNT_CACHE_DIRTY", False)
    return raw


def test_list_stored_datasets_preserves_relative_paths(instrument_tree):
    """The lister must report POSIX-style paths relative to data/raw/
    ("NQ1!/NQ1!_2024-01.csv"), not bare filenames -- the dropdown's
    option values and the grouping both derive from these."""
    names = sorted(ds.name for ds in storage.list_stored_datasets())
    assert names == sorted(
        ["loose.csv"]
        + [f"{inst}/{f}" for inst, files in LAYOUT.items() for f in files]
    )


def test_list_datasets_by_instrument_groups_by_folder(instrument_tree):
    """The dropdown's optgroups come from here: one group per top-level
    data/raw/ subfolder, files sorted inside, loose files under
    "(ungrouped)". A flat implementation (single group / basename-only
    names) fails this."""
    groups = storage.list_datasets_by_instrument(count_rows=False)
    by_instrument = {g["instrument"]: g for g in groups}
    assert set(by_instrument) == set(LAYOUT) | {"(ungrouped)"}
    assert by_instrument["NQ1!"]["file_count"] == 2
    assert by_instrument["MGC1!"]["file_count"] == 1
    assert by_instrument["(ungrouped)"]["file_count"] == 1
    # full_name is what the <option value="..."> submits: it must keep
    # the instrument-folder prefix so the selection round-trips.
    for instrument, files in LAYOUT.items():
        full_names = {f["full_name"] for f in by_instrument[instrument]["files"]}
        assert full_names == {f"{instrument}/{f}" for f in files}
        assert {f["name"] for f in by_instrument[instrument]["files"]} == set(files)


def test_resolve_stored_dataset_keeps_instrument_folder(instrument_tree):
    """THE v2 regression: the dropdown submits "NQ1!/NQ1!_2024-01.csv";
    the resolver must return the real file, not None. The pre-fix
    basename-flattening implementation returns None for every one of
    these and fails this test."""
    raw = instrument_tree
    for instrument, files in LAYOUT.items():
        for fname in files:
            rel = f"{instrument}/{fname}"
            resolved = storage.resolve_stored_dataset(rel)
            assert resolved is not None, (
                f"{rel!r} should resolve to the on-disk file "
                f"(pre-fix basename flattening returns None)"
            )
            assert resolved == raw / instrument / fname
    # Top-level files keep resolving too.
    assert storage.resolve_stored_dataset("loose.csv") == raw / "loose.csv"


def test_resolve_stored_dataset_still_blocks_traversal(instrument_tree):
    """The security property v2 added must survive the fix: traversal
    payloads resolve to None ("no such dataset"), never to a path
    outside data/raw/."""
    assert storage.resolve_stored_dataset("../../etc/passwd") is None
    assert storage.resolve_stored_dataset("..") is None
    assert storage.resolve_stored_dataset("../loose.csv") is None
    assert storage.resolve_stored_dataset("NQ1!/../../etc/passwd") is None
    assert storage.resolve_stored_dataset("NQ1!\\..\\..\\etc\\passwd") is None
    assert storage.resolve_stored_dataset("/etc/passwd") is None
    assert storage.resolve_stored_dataset("C:/Windows/win.ini") is None
    assert storage.resolve_stored_dataset("") is None
    assert storage.resolve_stored_dataset("   ") is None
    # Nonexistent names are "no such dataset", not a path to open.
    assert storage.resolve_stored_dataset("NQ1!/nope.csv") is None
    assert storage.resolve_stored_dataset("nope.csv") is None
