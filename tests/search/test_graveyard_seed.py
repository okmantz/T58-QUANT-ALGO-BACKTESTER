"""The starter Strategy Graveyard seed: shipped file is sane, and installing it is idempotent and non-destructive."""
from __future__ import annotations

import json

import pytest

from app.search import graveyard as gy


@pytest.fixture()
def seed_home(tmp_path, monkeypatch):
    monkeypatch.setattr(gy, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(gy, "_SEED_CHECKED", False)
    yield tmp_path
    gy._SEED_CHECKED = False


def _write_seed(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _row(cid, inst="EURUSD/EURUSD60.csv", tf="1h"):
    return {"_seed_instrument": inst, "_seed_timeframe": tf, "candidate_id": cid, "family": "ma_cross",
            "reason": "no edge after costs", "param_signature": f"sig-{cid}", "stage_died": "not_ranked"}


def test_shipped_seed_has_500_plus_real_rows():
    assert gy.SEED_PATH.exists(), "seed file must ship with the app"
    rows = [json.loads(ln) for ln in gy.SEED_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) >= 500
    keys = [(r["_seed_instrument"], r["_seed_timeframe"], r["candidate_id"]) for r in rows]
    assert len(keys) == len(set(keys)), "candidate ids must be unique within each instrument/timeframe file"
    sigs = [(r["_seed_instrument"], r["_seed_timeframe"], r["param_signature"]) for r in rows]
    assert len(sigs) == len(set(sigs)), "no duplicate parameter neighbourhoods per file"
    for r in rows[:50]:
        assert r["_seed_instrument"] and r["_seed_timeframe"]
        assert r["reason"] and r["primary_failure"]
        assert any("starter seed" in n for n in r["notes"]), "every row records where it was measured"


def test_install_is_idempotent(seed_home, monkeypatch):
    seed = seed_home / "seed.jsonl"
    _write_seed(seed, [_row("a"), _row("b"), _row("c", inst="GBPUSD/GBPUSD240.csv", tf="4h")])
    monkeypatch.setattr(gy, "SEED_PATH", seed)
    assert gy.ensure_seed_installed(force=True) == 3
    assert (seed_home / "data" / "evolution" / ".graveyard_seed_installed").exists()
    gy._SEED_CHECKED = False
    assert gy.ensure_seed_installed() == 0            # marker matches -> nothing to do
    assert gy.ensure_seed_installed(force=True) == 0  # forced, but rows already present
    p = gy.graveyard_path_for("EURUSD/EURUSD60.csv", "1h")
    assert len(gy.load_graveyard(p, _install=False)) == 2


def test_rebuilt_seed_merges_only_new_rows_and_keeps_user_rows(seed_home, monkeypatch):
    seed = seed_home / "seed.jsonl"
    monkeypatch.setattr(gy, "SEED_PATH", seed)
    _write_seed(seed, [_row("a")])
    gy.ensure_seed_installed(force=True)
    p = gy.graveyard_path_for("EURUSD/EURUSD60.csv", "1h")
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"candidate_id": "mine", "family": "x", "reason": "user run"}) + "\n")
    _write_seed(seed, [_row("a"), _row("z")])
    gy._SEED_CHECKED = False
    assert gy.ensure_seed_installed() == 1
    ids = {r["candidate_id"] for r in gy.load_graveyard(p, _install=False)}
    assert ids == {"a", "mine", "z"}


def test_load_graveyard_triggers_install_and_never_raises(seed_home, monkeypatch):
    seed = seed_home / "seed.jsonl"
    _write_seed(seed, [_row("a"), _row("b")])
    monkeypatch.setattr(gy, "SEED_PATH", seed)
    p = gy.graveyard_path_for("EURUSD/EURUSD60.csv", "1h")
    assert len(gy.load_graveyard(p)) == 2
    seed.write_text("not json\n{broken", encoding="utf-8")
    gy._SEED_CHECKED = False
    assert gy.ensure_seed_installed(force=True) == 0   # bad lines skipped, no exception


def test_missing_seed_file_is_fine(seed_home, monkeypatch):
    monkeypatch.setattr(gy, "SEED_PATH", seed_home / "nope.jsonl")
    assert gy.ensure_seed_installed(force=True) == 0
