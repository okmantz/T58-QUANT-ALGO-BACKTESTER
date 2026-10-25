"""Family-level trial ledger (2026-10-24): Search Lab / Evolution Lab bookings, idempotence, aging, the ICIR Bonferroni
re-scale and the pipeline's use of the family prior."""
from __future__ import annotations

import json
import random
import time

import pandas as pd
import pytest

from app.validation import trial_ledger as tl
from app.validation.icir import bonferroni_adjusted_alpha, rescale_gate_for_n_tests, run_icir_gate
from tests.validation.test_icir import FakeTrade, _real_signal_trades


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "reports" / "trial_ledger.json")
    return tl


def test_book_run_books_each_family_once(ledger):
    out = ledger.book_run("search:abc", {"mean_reversion": 4200, "orb": 300})
    assert out == {"family:mean_reversion": 4200, "family:orb": 300}
    assert ledger.family_prior("mean_reversion") == 4200 and ledger.family_prior("family:orb") == 300
    # idempotent: a retried / resumed run never double counts
    assert ledger.book_run("search:abc", {"mean_reversion": 4200}) == {}
    assert ledger.family_prior("mean_reversion") == 4200
    # a different run adds
    ledger.book_run("search:def", {"mean_reversion": 100})
    assert ledger.family_prior("mean_reversion") == 4300
    assert ledger.family_prior("never_seen") == 0


def test_reserved_bookkeeping_key_never_leaks_into_totals(ledger):
    ledger.book_run("evolution:r1:0", {"x": 5})
    data = json.loads(ledger._PATH.read_text())
    assert ledger.BOOKED_KEY in data
    assert ledger.raw_total(ledger.BOOKED_KEY) == 0 or isinstance(data[ledger.BOOKED_KEY], dict)
    assert ledger.total("family:x", half_life_days=None) == 5


def test_stage1_kills_count_and_old_runs_age_out(ledger):
    now = time.time()
    ledger.book_run("search:old", {"f": 1000}, now=now - 730 * 86400)   # one half-life ago
    ledger.book_run("search:new", {"f": 1000}, now=now)
    assert ledger.family_prior("f", now=now) == pytest.approx(1500, abs=2)
    assert ledger.total("family:f", half_life_days=None) == 2000


def test_atomic_write_leaves_valid_json_and_no_temp_files(ledger):
    for i in range(20):
        ledger.book_run(f"r{i}", {"fam": i + 1})
    json.loads(ledger._PATH.read_text())
    assert not list(ledger._PATH.parent.glob(".trial_ledger.*"))


def test_existing_record_api_unchanged(ledger):
    assert ledger.record("MyStrategy", 7) == 7
    assert ledger.total("MyStrategy", half_life_days=None) == 7
    ledger.mark_clock_epoch("e1", "note")
    assert ledger.clock_epochs()[0]["epoch"] == "e1"
    assert ledger.total("MyStrategy", half_life_days=None) == 7


def test_run_search_books_every_stage_per_family(ledger):
    from app.search.batch_runner import _family_eval_counts
    s1 = [{"family": "a"}] * 10 + [{"family": "b"}] * 5          # Stage-1 kills included
    s2 = [{"family": "a", "ga_total_evaluations": 200}, {"family": "b", "ga_total_evaluations": 0}]
    s3 = [{"family": "a"}]
    assert _family_eval_counts(s1, s2, s3) == {"a": 10 + 200 + 1, "b": 5}


def _gate(n_tests):
    # Strong-ish signal: significant at n_tests=1, not at huge n_tests
    return run_icir_gate(_real_signal_trades(240, 1), _real_signal_trades(240, 3), n_tests=n_tests, period="W")


def test_rescale_gate_only_ever_tightens():
    g1 = _gate(1)
    # same p-value, bigger n_tests -> adjusted alpha shrinks, never loosens
    g_big = rescale_gate_for_n_tests(g1, 5000)
    assert g_big.significance.n_tests == 5000
    assert g_big.significance.adjusted_alpha == pytest.approx(bonferroni_adjusted_alpha(0.05, 5000))
    assert g_big.significance.p_value == g1.significance.p_value
    assert (not g_big.ok) or g1.ok
    # asking for fewer tests is ignored
    assert rescale_gate_for_n_tests(g_big, 10) is g_big
    # equals recomputing from scratch with the larger n
    fresh = _gate(5000)
    assert g_big.ok == fresh.ok
    assert g_big.significance.significant == fresh.significance.significant
    assert any("5000" in r for r in g_big.reasons)


def test_pipeline_family_prior_reads_cfg_then_library_sidecar(ledger, monkeypatch):
    from types import SimpleNamespace
    from app.orchestration import full_pipeline as fp
    ledger.book_run("search:z", {"orb": 777})
    assert fp._ledger_family_prior(SimpleNamespace(ledger_family="orb", library_ref=None)) == 777
    assert fp._ledger_family_prior(SimpleNamespace(ledger_family="", library_ref=None)) == 0
    import app.strategy.library as lib
    monkeypatch.setattr(lib, "load_strategy_metadata", lambda t, f: {"search_family": "orb"})
    assert fp._ledger_family_prior(SimpleNamespace(ledger_family="", library_ref=("manual", "x.json"))) == 777
    monkeypatch.setattr(lib, "load_strategy_metadata", lambda t, f: {})
    assert fp._ledger_family_prior(SimpleNamespace(ledger_family="", library_ref=("manual", "x.json"))) == 0
