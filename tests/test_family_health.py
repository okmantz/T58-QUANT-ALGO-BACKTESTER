"""Tests for app.search.family_health -- cross-run dead-end family
detection combining Search Lab's SQLite history and Evolution Lab's
tested-candidates log, used to auto-exclude families that have never once
produced a real survivor across every past run."""
from __future__ import annotations

import json

import pytest

from app.search.family_health import (
    apply_family_exclusions, compute_family_health, dead_end_families,
)
from app.search.results_db import ResultsDB


def _make_search_db(path, family, n_tested, n_passed, run_id="run1"):
    with ResultsDB(path) as db:
        db.create_run(run_id, mode="family", family=family, instrument="X", timeframe="Y",
                       total_candidates=n_tested, config={})
        for i in range(n_tested):
            db.insert_candidate(run_id, f"{family}-{i}", "stage1", {"family": family, "passed_stage1": True})
        for i in range(n_passed):
            db.insert_candidate(
                run_id, f"{family}-pass-{i}", "stage3",
                {"family": family, "passed_stage3_gate": True, "composite_score": 10.0},
            )
        db.finish_run(run_id)


def _write_evolution_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_never_tried_family_is_absent_not_flagged(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    health = compute_family_health(search_dir, evo_dir, min_samples=5)
    assert health == {}
    assert dead_end_families(search_dir, evo_dir, min_samples=5) == set()


def test_family_with_few_samples_is_not_dead_end_yet(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    _make_search_db(search_dir / "search_abc.db", "mean_reversion_band", n_tested=3, n_passed=0)
    dead = dead_end_families(search_dir, evo_dir, min_samples=30)
    assert dead == set()


def test_family_with_many_failures_and_zero_successes_is_dead_end(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    _make_search_db(search_dir / "search_abc.db", "mean_reversion_band", n_tested=40, n_passed=0)
    dead = dead_end_families(search_dir, evo_dir, min_samples=30)
    assert dead == {"mean_reversion_band"}


def test_family_with_many_samples_and_at_least_one_success_is_healthy(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    _make_search_db(search_dir / "search_abc.db", "trend_breakout", n_tested=100, n_passed=1)
    dead = dead_end_families(search_dir, evo_dir, min_samples=30)
    assert dead == set()


def test_counts_combine_across_multiple_search_dbs_and_runs(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    _make_search_db(search_dir / "search_a.db", "session_time_effect", n_tested=15, n_passed=0, run_id="run1")
    _make_search_db(search_dir / "search_b.db", "session_time_effect", n_tested=20, n_passed=0, run_id="run2")
    health = compute_family_health(search_dir, evo_dir, min_samples=30)
    assert health["session_time_effect"].n_tested == 35
    assert health["session_time_effect"].n_passed == 0
    assert health["session_time_effect"].is_dead_end


def test_search_db_found_in_nested_subfolder(tmp_path):
    """Multi-instrument search writes each instrument's db under a nested
    subfolder (multi_instrument/<job_id>/), not flat under SEARCH_DIR --
    the scan must find those too."""
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    nested = search_dir / "multi_instrument" / "job123" / "EURUSD_5m"
    _make_search_db(nested / "search_x.db", "vwap_reversion", n_tested=40, n_passed=0)
    dead = dead_end_families(search_dir, evo_dir, min_samples=30)
    assert dead == {"vwap_reversion"}


def test_evolution_lab_tested_log_counts(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    rows = [{"family": "rsi_extreme_reversion", "stage": "prefilter", "passed": False} for _ in range(35)]
    _write_evolution_log(evo_dir / "tested_candidates.jsonl", rows)
    dead = dead_end_families(search_dir, evo_dir, min_samples=30)
    assert dead == {"rsi_extreme_reversion"}


def test_evolution_lab_log_found_in_nested_multi_instrument_subfolder(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    rows = [{"family": "macd_cross_trend", "stage": "prefilter", "passed": False} for _ in range(31)]
    _write_evolution_log(evo_dir / "multi_instrument" / "grp1" / "EURUSD_5m" / "tested_candidates.jsonl", rows)
    dead = dead_end_families(search_dir, evo_dir, min_samples=30)
    assert dead == {"macd_cross_trend"}


def test_evolution_lab_one_pass_among_many_makes_it_healthy(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    rows = [{"family": "wma_ribbon_trend", "stage": "prefilter", "passed": False} for _ in range(40)]
    rows.append({"family": "wma_ribbon_trend", "stage": "prefilter", "passed": True})
    _write_evolution_log(evo_dir / "tested_candidates.jsonl", rows)
    dead = dead_end_families(search_dir, evo_dir, min_samples=30)
    assert dead == set()


def test_search_and_evolution_history_combine_for_the_same_family(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    _make_search_db(search_dir / "search_a.db", "overnight_gap_fade", n_tested=15, n_passed=0)
    rows = [{"family": "overnight_gap_fade", "stage": "prefilter", "passed": False} for _ in range(16)]
    _write_evolution_log(evo_dir / "tested_candidates.jsonl", rows)
    health = compute_family_health(search_dir, evo_dir, min_samples=30)
    assert health["overnight_gap_fade"].n_tested == 31
    assert health["overnight_gap_fade"].is_dead_end


def test_apply_family_exclusions_returns_none_when_nothing_dead(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    families, excluded = apply_family_exclusions(search_dir, evo_dir, min_samples=30)
    assert families is None
    assert excluded == []


def test_apply_family_exclusions_excludes_only_dead_families(tmp_path):
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    _make_search_db(search_dir / "search_a.db", "mean_reversion_band", n_tested=40, n_passed=0)
    _make_search_db(search_dir / "search_b.db", "trend_breakout", n_tested=40, n_passed=2, run_id="run2")
    families, excluded = apply_family_exclusions(search_dir, evo_dir, min_samples=30)
    assert families is not None
    assert "mean_reversion_band" not in families
    assert "trend_breakout" in families
    assert excluded == ["mean_reversion_band"]


def test_apply_family_exclusions_never_excludes_everything(tmp_path, monkeypatch):
    """If every registered family somehow qualifies as dead-end (only
    plausible with an aggressive min_samples on a small history), this
    must back off to searching everything rather than returning a search
    space with nothing in it."""
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    from app.search.strategy_space import list_families
    for fam in list_families():
        _make_search_db(search_dir / f"search_{fam}.db", fam, n_tested=30, n_passed=0, run_id=f"run_{fam}")
    families, excluded = apply_family_exclusions(search_dir, evo_dir, min_samples=30)
    assert families is None
    assert set(excluded) == set(list_families())


def test_apply_family_exclusions_never_collapses_to_a_handful_of_survivors(tmp_path):
    """Regression test for a real report: "every time I run the evolution
    lab, it creates the same three strategies." Root cause -- the OLD
    safety rule here only ever checked "would this leave zero families,"
    so on an instrument where most families legitimately fail most of
    the time, exclusions can accumulate across dozens of sessions until
    only a small handful of survivor families are ever left in play --
    which is exactly what happened. This asserts the NEW min_active_families
    floor kicks in well before that point: excluding all-but-3 of the
    registered families must back off to searching everything, not hand
    back a search space collapsed down to 3."""
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    from app.search.strategy_space import list_families
    all_families = list_families()
    assert len(all_families) > 6  # sanity: this test only means something if there ARE more than 6 to collapse from
    survivors_to_keep = set(list(all_families)[:3])
    for fam in all_families:
        if fam not in survivors_to_keep:
            _make_search_db(search_dir / f"search_{fam}.db", fam, n_tested=30, n_passed=0, run_id=f"run_{fam}")

    families, excluded = apply_family_exclusions(search_dir, evo_dir, min_samples=30, min_active_families=6)
    # Would have left only 3 families in play -- below the floor -- so
    # this must back off to searching everything instead.
    assert families is None
    assert set(excluded) == set(all_families) - survivors_to_keep


def test_apply_family_exclusions_allows_exclusion_when_plenty_of_survivors_remain(tmp_path):
    """The floor should NOT block a normal, healthy exclusion where
    plenty of diverse families remain -- only the collapse case."""
    search_dir = tmp_path / "search"
    evo_dir = tmp_path / "evolution"
    from app.search.strategy_space import list_families
    all_families = list(list_families())
    # Exclude just one family -- plenty of survivors remain either way.
    dead_family = all_families[0]
    _make_search_db(search_dir / "search_dead.db", dead_family, n_tested=30, n_passed=0)

    families, excluded = apply_family_exclusions(search_dir, evo_dir, min_samples=30, min_active_families=6)
    assert families is not None
    assert dead_family not in families
    assert excluded == [dead_family]
