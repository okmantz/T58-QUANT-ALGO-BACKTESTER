"""Trial-ledger integrity (2026-10-31 audit, section 4): the ledger fails loudly, same-data trials never decay in Full Pipeline, and the
two trial counts carry names."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from app.orchestration import full_pipeline as fp
from app.search.robustness import deflated_sharpe_gate
from app.validation import trial_ledger as tl

DAY = 86400.0


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "reports" / "trial_ledger.json")
    return tl


def _corrupt(ledger, text="{ not json"):
    ledger._PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger._PATH.write_text(text, encoding="utf-8")


def test_missing_file_is_a_genuinely_empty_ledger(ledger):
    assert not ledger._PATH.exists()
    assert ledger.total("anything") == 0 and ledger.raw_total("anything") == 0


@pytest.mark.parametrize("op", [
    lambda t: t.total("s"), lambda t: t.raw_total("s"), lambda t: t.record("s", 3),
    lambda t: t.book_run("r", {"f": 5}), lambda t: t.family_prior("f"), lambda t: t.mark_clock_epoch("e", "n"),
])
def test_corrupt_file_raises_instead_of_reading_as_empty(ledger, op):
    _corrupt(ledger)
    with pytest.raises(tl.TrialLedgerError):
        op(ledger)


def test_corrupt_file_is_preserved_and_a_forensic_copy_saved(ledger):
    _corrupt(ledger, "{ truncated")
    with pytest.raises(tl.TrialLedgerError):
        ledger.total("s")
    assert ledger._PATH.read_text() == "{ truncated"            # original untouched: it keeps raising until a human repairs it
    copies = list(ledger._PATH.parent.glob("trial_ledger.corrupt-*.json"))
    assert len(copies) == 1 and copies[0].read_text() == "{ truncated"
    with pytest.raises(tl.TrialLedgerError):                      # second failure does not pile up copies
        ledger.total("s")
    assert len(list(ledger._PATH.parent.glob("trial_ledger.corrupt-*.json"))) == 1


def test_non_object_json_is_rejected(ledger):
    _corrupt(ledger, "[1, 2, 3]")
    with pytest.raises(tl.TrialLedgerError, match="JSON object"):
        ledger.total("s")


def test_repairing_the_file_restores_the_counts(ledger):
    ledger.record("s", 5)
    good = ledger._PATH.read_text()
    _corrupt(ledger)
    with pytest.raises(tl.TrialLedgerError):
        ledger.total("s")
    ledger._PATH.write_text(good)
    assert ledger.total("s") == 5


def test_failed_write_aborts_instead_of_reporting_a_total(ledger, monkeypatch):
    ledger.record("s", 2)
    def boom(*a, **k):
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr(os, "replace", boom)
        with pytest.raises(tl.TrialLedgerError, match="NOT recorded"):
            ledger.record("s", 10)
        with pytest.raises(tl.TrialLedgerError):
            ledger.book_run("run-x", {"fam": 4})
    assert ledger.total("s") == 2                                # nothing half-written
    assert not list(ledger._PATH.parent.glob(".trial_ledger.*"))  # temp file cleaned up


# --- same-data trials never decay ------------------------------------------------------------------------------------------

def test_two_same_data_runs_800_days_apart_give_equal_counts(ledger):
    now = 1_800_000_000.0
    ledger.record("S", 100, now=now - 800 * DAY, data_fp="FP1")
    ledger.record("S", 50, now=now, data_fp="FP1")
    assert ledger.total("S", now=now, current_fp="FP1") == 150
    assert ledger.total("S", now=now - 800 * DAY, current_fp="FP1") == 150    # same count whichever day it is asked
    assert ledger.total("S", now=now) < 150                                  # without the fingerprint the old trials still fade


def test_a_different_data_window_still_ages(ledger):
    now = 1_800_000_000.0
    ledger.record("S", 100, now=now - 730 * DAY, data_fp="OLD")
    assert ledger.total("S", now=now, current_fp="NEW") == 50                # one half-life


def test_full_pipeline_helpers_pass_the_data_fingerprint(ledger):
    strat = SimpleNamespace(name="S")
    old = __import__("time").time() - 800 * DAY
    ledger.record("S", 100, now=old, data_fp="FP1")
    ledger.book_run("search:1", {"famx": 200}, now=old, data_fp="FP1")
    cfg = SimpleNamespace(ledger_family="famx", library_ref=None)
    assert fp._ledger_prior_trials(strat, data_fp="FP1") == 100
    assert fp._ledger_family_prior(cfg, strat, data_fp="FP1") == 200
    assert fp._ledger_prior_trials(strat) < 100 and fp._ledger_family_prior(cfg, strat) < 200   # old behaviour without a fingerprint
    fp._ledger_record_trials(strat, 7, data_fp="FP1")
    assert fp._ledger_prior_trials(strat, data_fp="FP1") == 107
    entries = json.loads(ledger._PATH.read_text())["s"]
    assert all(e.get("fp") == "FP1" for e in entries)                        # every Full Pipeline booking carries the fingerprint


def test_full_pipeline_helpers_raise_on_a_corrupt_ledger(ledger):
    _corrupt(ledger)
    strat = SimpleNamespace(name="S")
    cfg = SimpleNamespace(ledger_family="f", library_ref=None)
    with pytest.raises(tl.TrialLedgerError):
        fp._ledger_prior_trials(strat)
    with pytest.raises(tl.TrialLedgerError):
        fp._ledger_family_prior(cfg, strat)
    with pytest.raises(tl.TrialLedgerError):
        fp._ledger_record_trials(strat, 1)


def test_decay_report_states_which_rule_fired(ledger):
    now = 1_800_000_000.0
    ledger.record("S", 100, now=now - 800 * DAY, data_fp="FP1")
    ledger.record("S", 40, now=now - 800 * DAY, data_fp="OTHER")
    r = ledger.decay_report("S", now=now, current_fp="FP1")
    assert r["raw"] == 140 and r["same_data"] == 100 and r["aged_by_half_life"] == 40
    assert "FP1" in r["rule"] and "do not decay" in r["rule"]
    assert "half-life" in ledger.decay_report("S", now=now, current_fp="ZZZ")["rule"]
    assert "no aging" in ledger.decay_report("S", half_life_days=None)["rule"]


# --- two named counts -------------------------------------------------------------------------------------------------------

def test_dsr_gate_result_carries_raw_and_effective_trial_counts():
    g = deflated_sharpe_gate(0.2, [0.05, 0.08, 0.1, 0.12], 72, 500, raw_trials=313, effective_trials=72,
                             trial_clustering="average-linkage at |corr| >= 0.9")
    assert (g.raw_trials, g.effective_trials, g.n_trials) == (313, 72, 72)
    assert "0.9" in g.trial_clustering
    legacy = deflated_sharpe_gate(0.2, [0.05, 0.08, 0.1, 0.12], 72, 500)
    assert legacy.raw_trials is None and legacy.effective_trials is None
