"""
Family Health -- has this named strategy family EVER produced a real
survivor, anywhere in this app's history, or has it just kept failing?

app.search.family_diversity.summarize_family_performance already answers
"which family did best in THIS ONE run" -- useful for a single completed
search, but it can't tell Evolution Lab or Search Lab "don't bother with
family X, it's failed every single time it's been tried across dozens of
runs on multiple instruments." This module answers that question instead,
aggregating EVERY past run this app has a record of:

  - Search Lab: every search_*.db under SEARCH_DIR (including its
    multi_instrument/ subfolder -- each instrument's own db), read via
    app.search.results_db.ResultsDB. Stage 1's "family" + "passed_stage1"
    columns give the tested/attempted count; Stage 3's "passed_stage3_gate"
    gives the success count.
  - Evolution Lab: data/evolution/tested_candidates.jsonl and every
    per-instrument copy under data/evolution/multi_instrument/ (see
    app.evolution.checkpoint) -- the PRE-FILTER stage's own durable "what
    was tested" log already records "family" + "passed" per row, so no
    new logging is needed to make this work.

A family only ever counts as a dead end once it has a genuinely large
sample (min_samples, default 30) across ALL of that combined history with
ZERO successes -- a family that's simply never been tried yet, or only a
handful of times, is "no evidence yet," not "proven dead." This
deliberately errs toward under-excluding: a false "dead end" wastes a
family's chance to redeem itself on a new instrument; a false "healthy"
just costs a bit more compute re-proving what's already known. Every
caller that uses this to skip families also always leaves at least one
family in play (see apply_family_exclusions' own docstring) -- flagging
EVERY family as dead-end (a real possibility on a small/synthetic
dataset) must never leave a search with nothing to search.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.data.storage import get_app_base_dir
from app.evolution import checkpoint as evo_checkpoint
from app.search.results_db import ResultsDB
from app.search.strategy_space import list_families


def default_search_dir() -> Path:
    """Where the web app's Search Lab writes its SQLite results dbs (see
    app.web.server.SEARCH_DIR) -- kept here too, independently, so this
    module (and app.evolution.engine, which must not import anything from
    app.web.server) can resolve a sensible default without a reverse
    dependency on the Flask app."""
    return get_app_base_dir() / "reports" / "search"


def default_evolution_base_dir() -> Path:
    return get_app_base_dir() / "data" / "evolution"


@dataclass
class FamilyHealth:
    family: str
    label: str
    n_tested: int
    n_passed: int
    min_samples: int = 30

    @property
    def is_dead_end(self) -> bool:
        return self.n_tested >= self.min_samples and self.n_passed == 0


def _search_lab_counts(search_dir: Path) -> dict[str, list[int]]:
    """{family: [n_tested, n_passed]} from every search_*.db this app has
    ever written, found anywhere under `search_dir` (including nested
    subfolders such as multi_instrument/<job_id>/ and champion/)."""
    counts: dict[str, list[int]] = {}
    if not search_dir.exists():
        return counts
    for db_path in search_dir.rglob("search_*.db"):
        try:
            with ResultsDB(db_path) as db:
                for run in db.list_runs(limit=10_000):
                    run_id = run.get("run_id")
                    if not run_id:
                        continue
                    for rec in db.leaderboard(run_id, stage="stage1", top_n=1_000_000):
                        fam = rec.get("family")
                        if not fam:
                            continue
                        if rec.get("error"):
                            # Same reasoning as the Evolution Lab prefilter log
                            # below: a build/backtest exception is a bug or a
                            # crash, not a verdict on the family's hypothesis.
                            continue
                        bucket = counts.setdefault(fam, [0, 0])
                        bucket[0] += 1
                    for rec in db.leaderboard(run_id, stage="stage3", top_n=1_000_000):
                        fam = rec.get("family")
                        if fam and rec.get("passed_stage3_gate"):
                            bucket = counts.setdefault(fam, [0, 0])
                            bucket[1] += 1
        except Exception:  # noqa: BLE001 -- a corrupt/partial/locked db must not break the whole scan
            continue
    return counts


def _evolution_lab_counts(evolution_base_dir: Path) -> dict[str, list[int]]:
    """{family: [n_tested, n_passed]} from every tested_candidates.jsonl
    this app has ever written, found anywhere under `evolution_base_dir`
    (the default single-instrument log plus every multi-instrument
    group's own per-instrument copy). 'passed' here means the PRE-FILTER
    stage's own pass/fail (a much cheaper bar than Search Lab's Stage 3
    gate) -- appropriate for Evolution Lab, where a family only reaches
    the expensive stages at all once it's cleared this one."""
    counts: dict[str, list[int]] = {}
    if not evolution_base_dir.exists():
        return counts
    for log_path in evolution_base_dir.rglob("tested_candidates.jsonl"):
        try:
            rows = evo_checkpoint.read_tested_rows(log_path, limit=10_000_000)
        except Exception:  # noqa: BLE001 -- a corrupt/partial log must not break the whole scan
            continue
        for row in rows:
            fam = row.get("family")
            if not fam or row.get("stage") != "prefilter":
                continue
            if row.get("error") or "build_or_backtest_error" in (row.get("reasons") or []):
                # A build/backtest exception (a bug in that family's config
                # builder, a since-fixed crash, an out-of-memory hiccup mid-run,
                # etc.) is evidence something went wrong running the candidate,
                # not evidence the family's trading hypothesis doesn't work.
                # Counting it toward n_tested would let a transient or
                # already-fixed implementation bug permanently blacklist a
                # family that was never actually given a fair trial -- skip
                # it entirely (neither tested nor passed) rather than let it
                # poison this family's health forever.
                continue
            bucket = counts.setdefault(fam, [0, 0])
            bucket[0] += 1
            if row.get("passed"):
                bucket[1] += 1
    return counts


def compute_family_health(
    search_dir: Path | str | None = None, evolution_base_dir: Path | str | None = None,
    min_samples: int = 30,
) -> dict[str, FamilyHealth]:
    """One FamilyHealth per named family with any history at all (a family
    never yet tried is simply absent -- "no evidence yet" -- rather than
    included with n_tested=0, so callers can distinguish "never tried"
    from "tried and clearly fine" without an extra check). Both directory
    arguments default to this app's real, persistent locations (see
    default_search_dir/default_evolution_base_dir) -- pass explicit paths
    only to scope the scan (e.g. tests, or a non-default --output dir).
    """
    search_dir = Path(search_dir) if search_dir is not None else default_search_dir()
    evolution_base_dir = Path(evolution_base_dir) if evolution_base_dir is not None else default_evolution_base_dir()

    combined: dict[str, list[int]] = {}
    for source in (_search_lab_counts(search_dir), _evolution_lab_counts(evolution_base_dir)):
        for fam, (tested, passed) in source.items():
            bucket = combined.setdefault(fam, [0, 0])
            bucket[0] += tested
            bucket[1] += passed

    labels = list_families()
    return {
        fam: FamilyHealth(family=fam, label=labels.get(fam, fam), n_tested=tested, n_passed=passed, min_samples=min_samples)
        for fam, (tested, passed) in combined.items()
    }


def dead_end_families(
    search_dir: Path | str | None = None, evolution_base_dir: Path | str | None = None,
    min_samples: int = 30,
) -> set[str]:
    """The set of named families (app.search.strategy_space.FAMILIES keys)
    that have been tested at least `min_samples` times across every past
    Search Lab and Evolution Lab run combined, with zero successes in all
    of them. Safe to use directly as an exclusion set -- if literally
    every registered family somehow qualifies (only plausible on a tiny
    synthetic/test dataset with an aggressively low min_samples), this
    still returns the full set; it is each CALLER's job to never end up
    excluding every family from an actual search (see apply_family_
    exclusions below, which does that safely)."""
    health = compute_family_health(search_dir, evolution_base_dir, min_samples=min_samples)
    return {fam for fam, h in health.items() if h.is_dead_end}


def reset_family_health(
    search_dir: Path | str | None = None, evolution_base_dir: Path | str | None = None,
) -> dict:
    """Wipes the durable history compute_family_health reads from, so
    every family starts fresh with "no evidence yet" instead of whatever
    dead-end verdicts accumulated from past runs. Use this after a bug
    fix (e.g. a crash or an OOM that produced a burst of spurious
    build_or_backtest_error rows before this module started excluding
    those from the tested count) or simply to give every family a clean
    slate on a new instrument/dataset. Renames rather than deletes each
    file (adds a '.pre-reset' suffix) so this is recoverable, not
    destructive. Returns {"search_dbs_reset": N, "evolution_logs_reset": N}.
    """
    search_dir = Path(search_dir) if search_dir is not None else default_search_dir()
    evolution_base_dir = Path(evolution_base_dir) if evolution_base_dir is not None else default_evolution_base_dir()

    def _archive(path: Path) -> bool:
        try:
            archived = path.with_name(path.name + ".pre-reset")
            if archived.exists():
                archived.unlink()
            path.rename(archived)
            return True
        except Exception:  # noqa: BLE001 -- a locked/missing file must not abort the whole reset
            return False

    n_search = 0
    if search_dir.exists():
        for db_path in search_dir.rglob("search_*.db"):
            if _archive(db_path):
                n_search += 1

    n_evo = 0
    if evolution_base_dir.exists():
        for log_path in evolution_base_dir.rglob("tested_candidates.jsonl"):
            if _archive(log_path):
                n_evo += 1

    return {"search_dbs_reset": n_search, "evolution_logs_reset": n_evo}


def apply_family_exclusions(
    search_dir: Path | str | None = None, evolution_base_dir: Path | str | None = None,
    min_samples: int = 30, min_active_families: int = 6,
) -> tuple[list[str] | None, list[str]]:
    """The actual safe-to-call-anywhere helper: computes the dead-end set
    and returns (families_to_search, excluded_families).

    families_to_search is None when nothing should be excluded (either
    nothing is dead-end yet, or excluding everything flagged would leave
    fewer than `min_active_families` families to search -- in which case
    this backs off to searching everything rather than returning a
    search space that's collapsed down to a small, stagnant handful).
    None is exactly the value app.search.strategy_space.generate_search_space
    and app.evolution.engine.EvolutionConfig already treat as "every
    family," so callers can pass this straight through unchanged.
    excluded_families is always the flagged set (possibly non-empty even
    when families_to_search is None, if excluding it all would have been
    unsafe) so callers can still log what was FOUND dead-end, distinct
    from what was actually excluded this run.

    min_active_families exists because "leaves at least one family" was
    the ONLY floor this used to enforce -- real report this fixes: on an
    instrument (gold) where most strategy families legitimately fail most
    of the time, it's entirely plausible for MOST registered families to
    individually cross the min_samples/zero-successes bar over dozens of
    sessions, leaving only 2-3 survivor families in play -- which is
    exactly "every time I run the evolution lab, it creates the same
    three strategies." A family that's been excluded is gone forever
    (this has no re-trial/decay mechanism yet -- see the module
    docstring), so as more families cross the bar over the app's
    lifetime, the search space only ever shrinks, never recovers. A
    non-empty-but-collapsed result was technically "safe" by the old
    zero-families check but not a search space anyone would actually
    want the GA restricted to; this raises the floor from 1 to a
    genuinely diverse minimum instead.
    """
    dead = dead_end_families(search_dir, evolution_base_dir, min_samples=min_samples)
    if not dead:
        return None, []
    survivors = [f for f in list_families() if f not in dead]
    if len(survivors) < min(min_active_families, len(list_families())):
        return None, sorted(dead)
    return survivors, sorted(dead)
