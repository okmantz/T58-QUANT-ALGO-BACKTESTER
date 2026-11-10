"""The ES / NQ / GC starter data ships INSIDE the app build, so the first run needs no import."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load("t58_spec_common_starter", "build_tools/spec_common.py")


@pytest.mark.parametrize("web", [True, False])
def test_both_builds_bundle_es_nq_gc_and_the_pack_manifests(web):
    dests = {dest for _src, dest in sc.data_entries(REPO_ROOT, web=web)}
    for folder in ("data/raw/ES1!", "data/raw/NQ1!", "data/raw/GC1!"):
        assert folder in dests, f"{folder} must be bundled so the first run needs no import"
    assert "data" in dests  # data_pack.json + data_manifest.json land in <bundle>/data


def test_starter_files_in_the_repo_are_real_parquet_not_lfs_pointers():
    for symbol, folder in (("ES", "ES1!"), ("NQ", "NQ1!"), ("GC", "GC1!")):
        files = list((REPO_ROOT / "data" / "raw" / folder).glob("*.parquet"))
        assert files, f"no parquet for {symbol}"
        assert all(f.stat().st_size > 1_000_000 for f in files), f"{symbol} data looks like an LFS pointer"


def test_data_pack_manifest_covers_the_sealed_pack_symbols_and_gc_ships_as_a_plain_dataset():
    """The sealed-holdout manifest is frozen on ES/NQ (and their micros) on purpose; adding GC to it is a pack-version decision for
    the owner. GC is still bundled and importable as a normal dataset."""
    manifest = json.loads((REPO_ROOT / "data" / "data_pack.json").read_text(encoding="utf-8"))
    for symbol in ("ES", "NQ"):
        entry = manifest["symbols"][symbol]
        assert entry["available"] is True
        assert (REPO_ROOT / "data" / "raw" / entry["file"]).is_file()
    assert list((REPO_ROOT / "data" / "raw" / "GC1!").glob("*.parquet"))


def test_an_lfs_pointer_in_the_starter_data_fails_the_build(tmp_path):
    root = tmp_path
    for src, _dest in sc.COMMON_DATA + sc.STARTER_DATA_DIRS + sc.WEB_DATA:
        (root / src).mkdir(parents=True, exist_ok=True)
    for src, _dest in sc.STARTER_DATA_FILES:
        (root / src).parent.mkdir(parents=True, exist_ok=True)
        (root / src).write_text("{}", encoding="utf-8")
    (root / "data/raw/ES1!/futures_ES.F_1m.parquet").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="LFS pointer"):
        sc.data_entries(root, web=True)


# ---- first-run seeding ----------------------------------------------------------------------------------
def _fake_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "_MEIPASS"
    (bundle / "data" / "raw" / "ES1!").mkdir(parents=True)
    (bundle / "data" / "raw" / "ES1!" / "futures_ES.F_1m.parquet").write_bytes(b"es")
    (bundle / "data" / "raw" / "NQ1!").mkdir(parents=True)
    (bundle / "data" / "raw" / "NQ1!" / "futures_NQ.F_1m.parquet").write_bytes(b"nq")
    (bundle / "data" / "data_pack.json").write_text("{}", encoding="utf-8")
    return bundle


def test_seed_copies_starter_data_and_manifests_on_first_run(tmp_path):
    from app.data import storage

    bundle = _fake_bundle(tmp_path)
    raw = tmp_path / "app" / "data" / "raw"
    raw.mkdir(parents=True)
    copied = storage._seed_bundled_raw_data(raw, bundle_root=bundle)
    assert sorted(copied) == ["ES1!/futures_ES.F_1m.parquet", "NQ1!/futures_NQ.F_1m.parquet"]
    assert (raw / "ES1!" / "futures_ES.F_1m.parquet").read_bytes() == b"es"
    assert (raw.parent / "data_pack.json").is_file()


def test_a_dataset_the_person_deleted_stays_deleted_but_new_bundle_files_still_arrive(tmp_path):
    from app.data import storage

    bundle = _fake_bundle(tmp_path)
    raw = tmp_path / "app" / "data" / "raw"
    raw.mkdir(parents=True)
    storage._seed_bundled_raw_data(raw, bundle_root=bundle)
    (raw / "NQ1!" / "futures_NQ.F_1m.parquet").unlink()          # the person removes NQ
    (bundle / "data" / "raw" / "GC1!").mkdir()
    (bundle / "data" / "raw" / "GC1!" / "futures_GC.F_1m.parquet").write_bytes(b"gc")   # a newer build adds GC
    copied = storage._seed_bundled_raw_data(raw, bundle_root=bundle)
    assert copied == ["GC1!/futures_GC.F_1m.parquet"]
    assert not (raw / "NQ1!" / "futures_NQ.F_1m.parquet").exists()


def test_seeding_never_overwrites_a_file_the_person_already_has(tmp_path):
    from app.data import storage

    bundle = _fake_bundle(tmp_path)
    raw = tmp_path / "app" / "data" / "raw"
    (raw / "ES1!").mkdir(parents=True)
    (raw / "ES1!" / "futures_ES.F_1m.parquet").write_bytes(b"mine")
    storage._seed_bundled_raw_data(raw, bundle_root=bundle)
    assert (raw / "ES1!" / "futures_ES.F_1m.parquet").read_bytes() == b"mine"


def test_source_runs_do_not_seed(tmp_path):
    from app.data import storage

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    assert storage._seed_bundled_raw_data(raw) == []     # not frozen, no bundle root given
