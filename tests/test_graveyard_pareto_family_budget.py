"""Unit tests for the three new standalone modules delivered alongside
the Evolution Lab fixes: app.search.graveyard, app.scoring.pareto,
app.evolution.family_budget."""
from pathlib import Path

from app.evolution.family_budget import FamilyBudgetTracker
from app.scoring.pareto import compute_pareto_frontier, label_frontier, pareto_report
from app.search.graveyard import (
    GraveyardEntry, graveyard_path_for, is_known_dead_neighborhood, list_graveyard_files, load_graveyard,
    param_signature, record_rejection, record_rejections, render_graveyard_report, summarize_graveyard,
)


# ---------------------------------------------------------------------------
# app.search.graveyard
# ---------------------------------------------------------------------------

def test_record_and_load_graveyard_round_trip(tmp_path):
    path = tmp_path / "graveyard.jsonl"
    entry = GraveyardEntry(
        candidate_id="orb_breakout-v173", family="prev_day_range_breakout", generation=42,
        stage_died="stress", reason="High parameter sensitivity", oos_result="negative",
        neighbor_robustness_pct=12.0, monte_carlo_failure_pct=71.0, prop_sim_pass_pct=18.0,
    )
    record_rejection(entry, path)
    rows = load_graveyard(path)
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "orb_breakout-v173"
    assert rows[0]["neighbor_robustness_pct"] == 12.0


def test_graveyard_entry_render_matches_expected_shape():
    entry = GraveyardEntry(
        candidate_id="orb_breakout-v173", family="prev_day_range_breakout", generation=42,
        stage_died="stress", reason="High parameter sensitivity", oos_result="negative",
        neighbor_robustness_pct=12.0, monte_carlo_failure_pct=71.0, prop_sim_pass_pct=18.0,
    )
    text = entry.render()
    assert "FAILED" in text
    assert "High parameter sensitivity" in text
    assert "12%" in text
    assert "71%" in text
    assert "18%" in text


def test_param_signature_dedupes_near_identical_configs():
    sig_a = param_signature("rsi_extreme_reversion", {"lookback": 68, "target_r": 1.83})
    sig_b = param_signature("rsi_extreme_reversion", {"lookback": 68.04, "target_r": 1.81})
    sig_c = param_signature("rsi_extreme_reversion", {"lookback": 20, "target_r": 3.0})
    assert sig_a == sig_b, "near-identical mutations of the same neighborhood should collapse to one signature"
    assert sig_a != sig_c


def test_param_signature_handles_missing_config_gracefully():
    assert param_signature("mean_reversion_band", None) == "mean_reversion_band"
    assert param_signature("mean_reversion_band", {}) == "mean_reversion_band"


def test_summarize_graveyard_clusters_repeated_attempts_and_ranks_by_count(tmp_path):
    path = tmp_path / "graveyard.jsonl"
    entries = [
        GraveyardEntry(candidate_id=f"rsi-{i}", family="rsi_extreme_reversion", generation=i,
                        stage_died="stress", reason="Failed the stress test", prop_sim_pass_pct=20.0,
                        param_signature="rsi_extreme_reversion|lookback=68.0")
        for i in range(5)
    ] + [
        GraveyardEntry(candidate_id="orb-1", family="prev_day_range_breakout", generation=1,
                        stage_died="stress", reason="High parameter sensitivity", prop_sim_pass_pct=18.0,
                        param_signature="prev_day_range_breakout|lookback=20.0")
    ]
    record_rejections(entries, path)
    rows = load_graveyard(path)
    clusters = summarize_graveyard(rows)
    assert clusters[0].family == "rsi_extreme_reversion"
    assert clusters[0].n_tested == 5
    assert clusters[1].n_tested == 1

    report = render_graveyard_report(clusters)
    assert "rsi_extreme_reversion" in report
    assert "prev_day_range_breakout" in report


def test_summarize_graveyard_empty_is_honest():
    assert summarize_graveyard([]) == []
    assert "empty" in render_graveyard_report([])


def test_is_known_dead_neighborhood_respects_min_attempts():
    rows = [{"family": "rsi_extreme_reversion", "param_signature": "rsi_extreme_reversion|lookback=68.0"}] * 3
    is_dead, n = is_known_dead_neighborhood(
        "rsi_extreme_reversion", {"lookback": 68}, rows=rows, min_attempts=8,
    )
    assert is_dead is False
    assert n == 3

    rows_many = rows * 3  # 9 attempts
    is_dead, n = is_known_dead_neighborhood(
        "rsi_extreme_reversion", {"lookback": 68}, rows=rows_many, min_attempts=8,
    )
    assert is_dead is True


# ---------------------------------------------------------------------------
# Persistent, instrument-scoped graveyard paths (app.search.graveyard.
# graveyard_path_for / list_graveyard_files) -- regression coverage for the
# bug where the web app previously wrote each Forge run to a unique
# per-job-id file that no future run (or UI) ever read again, silently
# defeating the entire "map of dead strategy space accumulates over time"
# purpose.
# ---------------------------------------------------------------------------

def test_graveyard_path_for_is_stable_across_calls():
    p1 = graveyard_path_for("ES1!", "1m")
    p2 = graveyard_path_for("ES1!", "1m")
    assert p1 == p2, "the same instrument+timeframe must always resolve to the same file"


def test_graveyard_path_for_scopes_by_instrument():
    es = graveyard_path_for("ES1!", "1m")
    eu = graveyard_path_for("EURUSD", "1m")
    assert es != eu


def test_graveyard_path_for_sanitizes_unsafe_characters():
    p = graveyard_path_for("ES1!/futures ES.F", "1m")
    assert p.name.isascii()
    assert "/" not in p.name and " " not in p.name


def test_list_graveyard_files_finds_written_files(tmp_path, monkeypatch):
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    path = graveyard_path_for("ES1!", "1m")
    record_rejection(GraveyardEntry(
        candidate_id="c1", family="fam_a", generation=None, stage_died="cpcv", reason="died",
    ), path=path)
    files = list_graveyard_files()
    assert len(files) == 1
    assert files[0]["instrument"] == "es1-"
    assert files[0]["timeframe"] == "1m"
    assert files[0]["n_rows"] == 1


def test_list_graveyard_files_empty_when_nothing_written(tmp_path, monkeypatch):
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    assert list_graveyard_files() == []


# ---------------------------------------------------------------------------
# app.scoring.pareto
# ---------------------------------------------------------------------------

def test_pareto_frontier_identifies_non_dominated_points():
    points = [
        {"candidate_id": "A", "eval_pass": 85, "payout": 38},
        {"candidate_id": "B", "eval_pass": 72, "payout": 61},
        {"candidate_id": "C", "eval_pass": 67, "payout": 73},
        {"candidate_id": "D", "eval_pass": 50, "payout": 20},  # dominated by everything
    ]
    metrics = {"eval_pass": "max", "payout": "max"}
    frontier = compute_pareto_frontier(points, metrics, key_fn=lambda p: p["candidate_id"])
    by_key = {p.key: p for p in frontier}
    assert by_key["A"].dominated is False
    assert by_key["B"].dominated is False
    assert by_key["C"].dominated is False
    assert by_key["D"].dominated is True
    assert "A" in by_key["D"].dominated_by


def test_pareto_frontier_respects_minimize_direction():
    points = [
        {"candidate_id": "low_dd_low_return", "return_pct": 10, "max_dd": 2},
        {"candidate_id": "high_dd_high_return", "return_pct": 40, "max_dd": 20},
        {"candidate_id": "worse_than_both", "return_pct": 5, "max_dd": 25},
    ]
    metrics = {"return_pct": "max", "max_dd": "min"}
    frontier = compute_pareto_frontier(points, metrics, key_fn=lambda p: p["candidate_id"])
    by_key = {p.key: p for p in frontier}
    assert by_key["low_dd_low_return"].dominated is False
    assert by_key["high_dd_high_return"].dominated is False
    assert by_key["worse_than_both"].dominated is True


def test_label_frontier_conservative_balanced_aggressive():
    points = [
        {"candidate_id": "A", "eval_pass": 85, "payout": 38},
        {"candidate_id": "B", "eval_pass": 72, "payout": 61},
        {"candidate_id": "C", "eval_pass": 67, "payout": 73},
    ]
    metrics = {"eval_pass": "max", "payout": "max"}
    frontier = compute_pareto_frontier(points, metrics, key_fn=lambda p: p["candidate_id"])
    frontier = label_frontier(frontier, conservative_metric="eval_pass", aggressive_metric="payout")
    by_key = {p.key: p for p in frontier}
    assert by_key["A"].label == "Conservative"
    assert by_key["C"].label == "Aggressive"
    assert by_key["B"].label == "Balanced"

    report = pareto_report(frontier)
    assert "Conservative" in report
    assert "Aggressive" in report


def test_pareto_frontier_tolerates_missing_metric_values():
    points = [
        {"candidate_id": "A", "eval_pass": 80, "payout": None},
        {"candidate_id": "B", "eval_pass": 60, "payout": 40},
    ]
    metrics = {"eval_pass": "max", "payout": "max"}
    # must not raise despite the missing value
    frontier = compute_pareto_frontier(points, metrics, key_fn=lambda p: p["candidate_id"])
    assert len(frontier) == 2


def test_label_frontier_single_point_is_balanced():
    points = [{"candidate_id": "only", "eval_pass": 80, "payout": 40}]
    frontier = compute_pareto_frontier(points, {"eval_pass": "max", "payout": "max"}, key_fn=lambda p: p["candidate_id"])
    frontier = label_frontier(frontier, "eval_pass", "payout")
    assert frontier[0].label == "Balanced"


# ---------------------------------------------------------------------------
# app.evolution.family_budget
# ---------------------------------------------------------------------------

def test_family_budget_no_history_is_neutral():
    tracker = FamilyBudgetTracker()
    assert tracker.multiplier("never_tried_family") == 1.0


def test_family_budget_boosts_families_with_stress_survivors():
    tracker = FamilyBudgetTracker(window=10)
    tracker.record_generation({
        "failed_breakout": {"tested": 10, "prefilter_passed": 4, "stress_passed": 3},
        "fvg_imbalance_continuation": {"tested": 10, "prefilter_passed": 1, "stress_passed": 0},
    })
    boosted = tracker.multiplier("failed_breakout")
    shrunk = tracker.multiplier("fvg_imbalance_continuation")
    assert boosted > 1.0
    assert boosted > shrunk


def test_family_budget_shrinks_but_never_zeroes_a_consistently_dead_family():
    tracker = FamilyBudgetTracker(window=10, min_frac=0.4)
    for _ in range(5):
        tracker.record_generation({"dead_family": {"tested": 10, "prefilter_passed": 0, "stress_passed": 0}})
    mult = tracker.multiplier("dead_family")
    assert mult == tracker.min_frac
    assert mult > 0.0, "adaptive budget must never zero a family outright -- that is family_health's job"


def test_family_budget_multiplier_clamped_to_max_frac():
    tracker = FamilyBudgetTracker(window=10, max_frac=2.0)
    tracker.record_generation({"great_family": {"tested": 5, "prefilter_passed": 5, "stress_passed": 5}})
    assert tracker.multiplier("great_family") == 2.0


def test_family_budget_recovers_after_a_cold_stretch():
    """The rolling window (not a permanent cross-run record) means a
    family that starts producing survivors again should recover, unlike
    family_health's permanent exclusion."""
    tracker = FamilyBudgetTracker(window=3)
    for _ in range(3):
        tracker.record_generation({"recovering_family": {"tested": 10, "prefilter_passed": 0, "stress_passed": 0}})
    cold_mult = tracker.multiplier("recovering_family")
    for _ in range(3):
        tracker.record_generation({"recovering_family": {"tested": 10, "prefilter_passed": 8, "stress_passed": 2}})
    warm_mult = tracker.multiplier("recovering_family")
    assert warm_mult > cold_mult


def test_family_budget_status_reports_multiplier_table():
    tracker = FamilyBudgetTracker()
    tracker.record_generation({"family_a": {"tested": 10, "prefilter_passed": 5, "stress_passed": 1}})
    status = tracker.status()
    assert "family_a" in status
    assert status["family_a"]["tested"] == 10
    assert "multiplier" in status["family_a"]


def test_family_budget_multipliers_bulk_helper():
    tracker = FamilyBudgetTracker()
    tracker.record_generation({"a": {"tested": 5, "prefilter_passed": 5, "stress_passed": 1}})
    result = tracker.multipliers(["a", "b"])
    assert set(result.keys()) == {"a", "b"}
    assert result["b"] == 1.0  # no history
