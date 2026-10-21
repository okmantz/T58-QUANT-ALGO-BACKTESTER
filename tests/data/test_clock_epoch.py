"""One-time clock-epoch reset (report 2026-10-20, P0 step 5): archive contaminated memory, keep the trial ledger."""
from __future__ import annotations

import json
import sqlite3

from app.data import clock_epoch as ce
from app.validation import trial_ledger


def _seed(root):
    (root / "reports" / "search").mkdir(parents=True)
    (root / "data" / "evolution" / "multi_instrument").mkdir(parents=True)
    (root / "reports" / "search" / "search_abc.db").write_bytes(b"x")
    (root / "data" / "evolution" / "tested_candidates.jsonl").write_text('{"family":"gap_and_go"}\n')
    (root / "data" / "evolution" / "strategy_graveyard__nq__15m.jsonl").write_text('{"family":"gap_and_go"}\n')
    (root / "reports" / "preregistered_results.json").write_text("[]")
    (root / "reports" / "preregistrations.json").write_text(json.dumps({"registrations": [{"fingerprint": "abc"}]}))


def test_apply_archives_memory_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(trial_ledger, "_PATH", tmp_path / "reports" / "trial_ledger.json")
    _seed(tmp_path)
    trial_ledger.record("family:gap_and_go", 40)
    before = trial_ledger.raw_total("family:gap_and_go")

    out = ce.apply_clock_epoch(root=tmp_path)
    assert out["applied_now"] is True
    # archived, recoverable, not deleted
    assert not (tmp_path / "reports" / "search" / "search_abc.db").exists()
    assert (tmp_path / "reports" / "search" / "search_abc.db.pre-reset").exists()
    assert (tmp_path / "data" / "evolution" / "strategy_graveyard__nq__15m.jsonl.pre-clock-fix").exists()
    assert (tmp_path / "reports" / "preregistered_results.json.pre-clock-fix").exists()
    # the registry (fingerprints) stays, annotated
    reg = json.loads((tmp_path / "reports" / "preregistrations.json").read_text())
    assert reg["registrations"][0]["fingerprint"] == "abc" and reg["clock_epochs"][0]["epoch"] == ce.CLOCK_EPOCH_ID
    # the trial ledger is KEPT (count unchanged) and records the epoch
    assert trial_ledger.raw_total("family:gap_and_go") == before == 40
    assert trial_ledger.clock_epochs()[0]["epoch"] == ce.CLOCK_EPOCH_ID
    assert ce.status(tmp_path)["applied"] is True

    again = ce.apply_clock_epoch(root=tmp_path)
    assert again["applied_now"] is False                                      # runs once per install
    assert len(trial_ledger.clock_epochs()) == 1


def test_dead_end_families_clear_after_reset(tmp_path, monkeypatch):
    """30 zero-pass tests mark a family dead; after the epoch reset the same family is 'no evidence yet' again."""
    from app.search.family_health import dead_end_families
    monkeypatch.setattr(trial_ledger, "_PATH", tmp_path / "reports" / "trial_ledger.json")
    evo = tmp_path / "data" / "evolution"
    evo.mkdir(parents=True)
    rows = "\n".join(json.dumps({"family": "gap_and_go", "stage": "prefilter", "passed": False}) for _ in range(40))
    (evo / "tested_candidates.jsonl").write_text(rows + "\n")
    assert "gap_and_go" in dead_end_families(tmp_path / "reports" / "search", evo)
    ce.apply_clock_epoch(root=tmp_path)
    assert "gap_and_go" not in dead_end_families(tmp_path / "reports" / "search", evo)


def test_skip_env_var_disables_startup_hook(monkeypatch):
    monkeypatch.setenv("T58_SKIP_CLOCK_EPOCH", "1")
    assert ce.ensure_clock_epoch() is None
