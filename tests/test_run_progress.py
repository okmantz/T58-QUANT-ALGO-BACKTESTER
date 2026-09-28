"""Tests for app.orchestration.run_progress -- the live phase/counter tracker
behind the "Phase 2 · Breeding · 1,240 candidates · 38/s" banner.

Two kinds of test: (1) the tracker's behaviour on realistic log sequences, and
(2) ANCHOR tests that grep the three engines' source for the exact wording each
regex depends on, so rewording a log line fails a test here instead of the
banner silently freezing."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.orchestration.run_progress import ProgressTracker, format_banner
from app.web.job_manager import JobManager

ROOT = Path(__file__).resolve().parent.parent


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def _tracker(kind="search"):
    clock = Clock()
    return ProgressTracker(kind, clock=clock), clock


# ---- banner format -----------------------------------------------------------

def test_banner_matches_the_documented_shape():
    assert format_banner(2, "Breeding", 1240, 38.4) == "Phase 2 · Breeding · 1,240 candidates · 38/s"


def test_banner_singular_low_rate_generation_and_done():
    assert format_banner(1, "Proving Ground", 1, 0.5) == "Phase 1 · Proving Ground · 1 candidate · 0.5/s"
    assert format_banner(4, "Survivors", 900, 12, generation=7).startswith("Gen 7 · Phase 4")
    assert format_banner(4, "Survivors", 900, 12, done=True).endswith("complete")
    assert "/s" not in format_banner(4, "Survivors", 900, 12, done=True)
    assert format_banner(0, "Starting", 0, 0) == "Starting…"


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        ProgressTracker("nope")


# ---- Search Lab --------------------------------------------------------------

SEARCH_LINES = [
    "Search space ready: 1,200 candidate(s) (family, family=trend_breakout).",
    "  Stage 1: 3/12 batch(es) evaluated (300/1,200 candidates)...",
    "  Stage 1: 12/12 batch(es) evaluated (1,200/1,200 candidates)...",
    "Stage 1 complete: 40/1,200 candidate(s) survived the cheap filter and advance to Stage 2 (GA refinement).",
    "Stage 2/5: genetic-algorithm refinement on 40 surviving skeleton(s)...",
    "Stage 2 complete: 10 candidate(s) advance to the Stage 3 validation gate.",
    "  Stage 3: 4/10 candidate(s) validated...",
    "Stage 4/5: computing deflated Sharpe ratios and ranking the leaderboard...",
]


def test_search_lab_walks_the_four_phases_in_order():
    t, clock = _tracker("search")
    seen = []
    for line in SEARCH_LINES:
        clock.advance(5)
        t.feed(line)
        seen.append(t.snapshot().phase_index)
    assert seen == [1, 1, 1, 1, 2, 3, 3, 4]
    snap = t.snapshot()
    assert snap.candidates_done == 1200 and snap.candidates_total == 1200
    assert snap.survivors == 10 and snap.phase_label == "Survivors"


def test_search_counters_and_rate():
    t, clock = _tracker("search")
    t.feed(SEARCH_LINES[0]); clock.advance(10); t.feed(SEARCH_LINES[1])
    snap = t.snapshot()
    assert (snap.candidates_done, snap.candidates_total) == (300, 1200)
    assert snap.rate_per_sec == 30.0
    assert snap.banner == "Phase 1 · Proving Ground · 300 candidates · 30/s"


def test_no_rate_reported_in_the_first_second():
    t, clock = _tracker("search")
    t.feed(SEARCH_LINES[0]); clock.advance(0.01); t.feed(SEARCH_LINES[1])
    assert t.snapshot().rate_per_sec == 0.0


def test_search_phase_never_steps_backwards():
    t, _ = _tracker("search")
    for line in SEARCH_LINES:
        t.feed(line)
    t.feed("  Stage 1: 12/12 batch(es) evaluated (1,200/1,200 candidates)...")  # a stray late line
    assert t.snapshot().phase_index == 4


def test_loop_mode_round_resets_phase_and_banks_candidates():
    t, clock = _tracker("search")
    for line in SEARCH_LINES:
        t.feed(line)
    clock.advance(60)
    t.feed("Search space ready: 800 candidate(s) (family).")  # round 2 starts
    snap = t.snapshot()
    assert snap.phase_index == 1                       # reset, not stuck on Survivors
    assert snap.candidates_done == 1200                # round 1 banked
    assert snap.candidates_total == 2000               # 1200 banked + 800 this round
    t.feed("  Stage 1: 2/8 batch(es) evaluated (200/800 candidates)...")
    assert t.snapshot().candidates_done == 1400


# ---- Speed Run ---------------------------------------------------------------

SPEED_LINES = [
    "Phase 1/3: Wide discovery search across every strategy family (speed-tuned settings)...",
    "Search space ready: 300 candidate(s) (family).",
    "  Stage 1: 5/10 batch(es) evaluated (150/300 candidates)...",
    "Stage 2/5: genetic-algorithm refinement on 40 surviving skeleton(s)...",
    "Stage 4/5: computing deflated Sharpe ratios and ranking the leaderboard...",
    "Phase 1 complete in 12.0s: 300 candidate(s) -> 40 Stage 1 -> 10 Stage 2 -> 3 Stage 3 survivor(s).",
    "Phase 2/3: Validating top 3 candidate(s) through Full Pipeline (speed-tuned settings, up to 2 at once)...",
]


def test_speed_run_inner_search_never_claims_survivors_before_phase_3():
    t, _ = _tracker("speed_run")
    phases = []
    for line in SPEED_LINES:
        t.feed(line)
        phases.append(t.snapshot().phase_index)
    assert max(phases) == 3            # the inner "Stage 4/5" is capped
    assert phases[-1] == 3             # long Full Pipeline validation is still Out-of-Sample
    t.feed("\nPhase 3/3: Selecting the best validated candidate...")
    assert t.snapshot().phase_index == 4
    assert t.snapshot().survivors == 3


def test_speed_run_new_round_resets():
    t, _ = _tracker("speed_run")
    for line in SPEED_LINES:
        t.feed(line)
    t.feed("Phase 1/3: Wide discovery search across every strategy family (speed-tuned settings)...")
    assert t.snapshot().phase_index == 1


# ---- Evolution Lab -----------------------------------------------------------

def _generation(gen, population=100, survivors=12):
    return [
        f"===== GENERATION {gen} =====",
        f"GENERATE: {population} candidates.",
        f"PRE-FILTER + BACKTEST: {survivors}/{population} survived (took 4.0s).",
        f"ROBUSTNESS / OOS / MONTE CARLO / PROP SIMULATION: {survivors} candidates scored.",
        "CPCV / PBO: re-scored top 8 candidates.",
        "STRESS TEST (2x costs): 3/8 still fitness-positive.",
        f"Generation {gen} complete in 9.5s. Best fitness so far: 61.25",
    ]


def test_evolution_cycles_phases_every_generation_and_accumulates():
    t, clock = _tracker("evolution")
    phases = []
    for gen in (1, 2):
        for line in _generation(gen):
            clock.advance(2)
            t.feed(line)
            phases.append(t.snapshot().phase_index)
    assert phases[:7] == [2, 2, 1, 3, 3, 3, 4]
    assert phases[7] == 2 and phases[-1] == 4      # goes BACK to breeding for generation 2
    snap = t.snapshot()
    assert snap.generation == 2
    assert snap.candidates_done == 200             # 100 per generation, accumulated
    assert snap.best_fitness == 61.25


def test_evolution_complete_line_without_best_fitness():
    t, _ = _tracker("evolution")
    t.feed("Generation 3 complete in 2.0s.")
    snap = t.snapshot()
    assert snap.generation == 3 and snap.best_fitness is None


# ---- robustness --------------------------------------------------------------

def test_unrecognised_and_hostile_lines_are_ignored():
    t, _ = _tracker("search")
    for junk in ("", None, "hello", "Stage 1: x/y batch(es)", "\x00\xff" * 50, "Search space ready: many candidate"):
        t.feed(junk)
    assert t.snapshot().phase_index == 0
    assert t.snapshot().banner == "Starting…"


def test_finish_freezes_elapsed_and_marks_done():
    t, clock = _tracker("search")
    t.feed(SEARCH_LINES[0]); t.feed(SEARCH_LINES[1]); clock.advance(10)
    t.finish(); clock.advance(500)
    snap = t.snapshot()
    assert snap.done and snap.elapsed_seconds == 10.0
    assert snap.banner.endswith("complete")


def test_total_never_reads_lower_than_done():
    t, _ = _tracker("search")
    t.feed("  Stage 1: 1/2 batch(es) evaluated (50/40 candidates)...")
    snap = t.snapshot()
    assert snap.candidates_total >= snap.candidates_done


# ---- job manager integration -------------------------------------------------

def test_job_manager_tracks_progress_only_when_asked():
    jm = JobManager()
    plain = jm.create()
    assert jm.get_progress(plain) is None and jm.get_progress("nope") is None
    tracked = jm.create(progress_kind="search")
    jm.log(tracked, SEARCH_LINES[0]); jm.log(tracked, SEARCH_LINES[1])
    assert jm.get_progress(tracked)["candidates_done"] == 300
    assert jm.get(tracked)["log"] == SEARCH_LINES[:2]


def test_job_manager_counts_lines_seeded_through_create():
    jm = JobManager()
    job = jm.create(log=[SEARCH_LINES[0], SEARCH_LINES[1]], progress_kind="search")
    assert jm.get_progress(job)["candidates_done"] == 300


def test_job_manager_finish_and_fail_freeze_the_tracker_and_stamp_finished_at():
    jm = JobManager()
    a, b = jm.create(progress_kind="search"), jm.create(progress_kind="search")
    jm.finish(a); jm.fail(b, "boom")
    for jid in (a, b):
        assert jm.get_progress(jid)["done"] is True
        assert isinstance(jm.get(jid)["finished_at"], float)
    assert jm.get(b)["error"] == "boom" and jm.get(a)["done"] is True


def test_finish_keeps_a_caller_supplied_finished_at_out_of_the_way_of_done():
    jm = JobManager()
    job = jm.create()
    jm.finish(job, result={"x": 1}, done=False)   # done can never be overridden to False
    assert jm.get(job)["done"] is True and jm.get(job)["result"] == {"x": 1}


def test_bad_progress_kind_never_blocks_job_creation():
    jm = JobManager()
    job = jm.create(progress_kind="typo")
    assert jm.get(job) is not None and jm.get_progress(job) is None


def test_feed_progress_does_not_grow_the_job_log():
    jm = JobManager()
    job = jm.create(progress_kind="evolution")
    for line in _generation(1):
        jm.feed_progress(job, line)
    assert jm.get(job)["log"] == []
    assert jm.get_progress(job)["generation"] == 1
    jm.feed_progress("nope", "x")  # unknown job: no error


def test_job_dicts_stay_json_safe_for_the_activity_feed():
    jm = JobManager()
    job = jm.create(progress_kind="search", tool="Search Lab")
    listed = jm.list_jobs()[0]
    assert listed["tool"] == "Search Lab" and "_tracker" in listed  # present, but routes filter non-primitives


# ---- anchors: the engines must still say what the regexes expect -------------

ANCHORS = {
    "app/search/batch_runner.py": [
        "Search space ready:", "batch(es) evaluated (", "Stage 1 complete:", "Stage 2/5:",
        "Stage 2 complete:", "Stage 3: {done}/", "Stage 4/5:",
    ],
    "app/orchestration/speed_run.py": [
        "Phase 1/3:", "Phase 1 complete in", "Phase 2/3:", "Phase 3/3:", "Speed Run complete in",
        "Stage 3 survivor(s)",
    ],
    "app/evolution/engine.py": [
        "===== GENERATION {gen} =====", "GENERATE: {len(population)} candidates.", "PRE-FILTER + BACKTEST:",
        "ROBUSTNESS / OOS / MONTE CARLO / PROP SIMULATION:", "CPCV / PBO:", "STRESS TEST (",
        "Generation {gen} complete in", "Best fitness so far:",
    ],
}


@pytest.mark.parametrize("path,phrase", [(p, a) for p, phrases in ANCHORS.items() for a in phrases])
def test_engine_still_emits_the_log_line_the_tracker_parses(path, phrase):
    source = (ROOT / path).read_text(encoding="utf-8")
    assert phrase in source, (
        f"{path} no longer contains {phrase!r}. app/orchestration/run_progress.py parses that log line "
        f"for the live phase banner -- update its regex (and this anchor) together."
    )
