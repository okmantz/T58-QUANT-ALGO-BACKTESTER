"""
Live run telemetry: turns the log lines Search Lab, Speed Run and Evolution
Lab already emit into a small structured snapshot -- current phase, candidate
counters, generation, survivors and throughput -- so a job page or the
Project Chat "Activity" tab can show a header like

    Phase 2 · Breeding · 1,240 candidates · 38/s

instead of only a wall of log text.

Why parse log lines instead of adding structured callbacks to the engines:
the three engines (app.search.batch_runner, app.orchestration.speed_run,
app.evolution.engine) are performance-critical and heavily tested; every one
of them already funnels ALL progress through a single ``progress_cb(str)``.
Reading that one stream costs the engines nothing and can't change what they
compute. The trade-off is that this module is coupled to their log wording,
so tests/test_run_progress.py pins every pattern against the real wording AND
greps the engine source for each anchor phrase -- if someone rewords a log
line, a test fails instead of the banner silently freezing.

Four display phases, borrowed from the reel that inspired this feature:

    1  Proving Ground  -- generate + cheap backtest of every candidate
    2  Breeding        -- GA refinement / breeding the next generation
    3  Out-of-Sample   -- validation, robustness, walk-forward, Monte Carlo
    4  Survivors       -- ranking and the final leaderboard

Search Lab and Speed Run move through them once, in order. Evolution Lab
cycles through them every generation, so its phase can go 2 -> 1 -> 3 -> 4
and back; the tracker never assumes monotonic progress.

Never raises: an unrecognised line is simply ignored (it still lands in the
normal log), so a wording change degrades the banner, never the run.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

PHASES: tuple[tuple[str, str], ...] = (
    ("proving_ground", "Proving Ground"),
    ("breeding", "Breeding"),
    ("out_of_sample", "Out-of-Sample"),
    ("survivors", "Survivors"),
)

KINDS = ("search", "speed_run", "evolution")

# -- Search Lab ------------------------------------------------------------
_RE_SPACE_READY = re.compile(r"Search space ready:\s*([\d,]+)\s+candidate")
_RE_STAGE1_PROGRESS = re.compile(r"Stage 1:\s*\d+/\d+ batch\(es\) evaluated \(([\d,]+)/([\d,]+) candidates\)")
_RE_STAGE1_DONE = re.compile(r"Stage 1 complete:\s*([\d,]+)/([\d,]+) candidate")
_RE_STAGE2_START = re.compile(r"Stage 2/5:")
_RE_STAGE2_DONE = re.compile(r"Stage 2 complete:\s*([\d,]+) candidate")
_RE_STAGE3_PROGRESS = re.compile(r"Stage 3:\s*([\d,]+)/([\d,]+) candidate")
_RE_STAGE4_START = re.compile(r"Stage 4/5:")

# -- Speed Run -------------------------------------------------------------
_RE_SPEED_P1 = re.compile(r"Phase 1/3:")
_RE_SPEED_P1_DONE = re.compile(
    r"Phase 1 complete in [\d.]+s:\s*([\d,]+) candidate\(s\) -> ([\d,]+) Stage 1 -> "
    r"([\d,]+) Stage 2 -> ([\d,]+) Stage 3 survivor"
)
_RE_SPEED_P2 = re.compile(r"Phase 2/3:")
_RE_SPEED_P3 = re.compile(r"Phase 3/3:")
_RE_SPEED_DONE = re.compile(r"Speed Run complete in")

# -- Evolution Lab ---------------------------------------------------------
_RE_EVO_GEN = re.compile(r"={3,}\s*GENERATION\s+(\d+)\s*={3,}")
_RE_EVO_GENERATE = re.compile(r"GENERATE:\s*([\d,]+) candidates")
_RE_EVO_PREFILTER = re.compile(r"PRE-FILTER \+ BACKTEST:\s*([\d,]+)/([\d,]+) survived")
_RE_EVO_ROBUST = re.compile(r"ROBUSTNESS / OOS / MONTE CARLO / PROP SIMULATION:\s*([\d,]+) candidates scored")
_RE_EVO_CPCV = re.compile(r"CPCV / PBO:")
_RE_EVO_STRESS = re.compile(r"STRESS TEST \([^)]*\):\s*([\d,]+)/([\d,]+) still fitness-positive")
_RE_EVO_COMPLETE = re.compile(r"Generation\s+(\d+) complete in [\d.]+s\.(?: Best fitness so far:\s*(-?[\d.]+))?")


def _int(text: str) -> int:
    return int(text.replace(",", ""))


@dataclass
class ProgressSnapshot:
    kind: str
    phase_index: int          # 0 = not started, else 1..4
    phase_key: str
    phase_label: str
    candidates_done: int
    candidates_total: Optional[int]
    generation: Optional[int]
    survivors: Optional[int]
    best_fitness: Optional[float]
    rate_per_sec: float
    elapsed_seconds: float
    done: bool
    banner: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "phase_index": self.phase_index, "phase_key": self.phase_key,
            "phase_label": self.phase_label, "candidates_done": self.candidates_done,
            "candidates_total": self.candidates_total, "generation": self.generation,
            "survivors": self.survivors, "best_fitness": self.best_fitness,
            "rate_per_sec": self.rate_per_sec, "elapsed_seconds": self.elapsed_seconds,
            "done": self.done, "banner": self.banner,
        }


def format_banner(
    phase_index: int, phase_label: str, candidates_done: int, rate_per_sec: float,
    generation: Optional[int] = None, done: bool = False,
) -> str:
    """The one-line header, e.g. ``Phase 2 · Breeding · 1,240 candidates · 38/s``.
    Pure function so the format is unit-testable and shared by every surface
    (web job pages, Activity tab, desktop panel)."""
    if phase_index <= 0:
        return "Starting…"
    parts = []
    if generation is not None:
        parts.append(f"Gen {generation}")
    parts.append(f"Phase {phase_index} · {phase_label}")
    parts.append(f"{candidates_done:,} candidate{'s' if candidates_done != 1 else ''}")
    if not done and rate_per_sec > 0:
        parts.append(f"{rate_per_sec:,.0f}/s" if rate_per_sec >= 10 else f"{rate_per_sec:.1f}/s")
    if done:
        parts.append("complete")
    return " · ".join(parts)


class ProgressTracker:
    """Feed it every progress line with :meth:`feed`; read :meth:`snapshot`.
    Thread-safe (the engine thread feeds while a polling request reads)."""

    def __init__(self, kind: str = "search", clock: Callable[[], float] = time.time) -> None:
        if kind not in KINDS:
            raise ValueError(f"Unknown progress kind {kind!r}; expected one of {KINDS}")
        self.kind = kind
        self._clock = clock
        self._lock = threading.Lock()
        self._started_at = clock()
        self._finished_at: Optional[float] = None
        self._phase = 0
        self._banked = 0          # candidates finished in EARLIER rounds/runs (loop mode re-runs the engine)
        self._done_count = 0      # candidates finished in the CURRENT run
        self._total: Optional[int] = None
        self._generation: Optional[int] = None
        self._survivors: Optional[int] = None
        self._best_fitness: Optional[float] = None

    # -- feeding ---------------------------------------------------------
    def feed(self, line: str) -> None:
        """Consume one log line. Unknown lines are ignored; never raises."""
        try:
            with self._lock:
                self._feed_locked(line or "")
        except Exception:  # noqa: BLE001 -- telemetry must never break a run
            pass

    def finish(self) -> None:
        """Freeze elapsed time / rate once the job is over (job runners call
        this via JobManager.finish/fail so the rate stops decaying)."""
        with self._lock:
            if self._finished_at is None:
                self._finished_at = self._clock()

    def _set_phase(self, phase: int, reset: bool = False) -> None:
        """Search Lab and Speed Run move through the four phases once, so they
        never step backwards (Speed Run's inner search emits its own stage
        lines while the outer run is still validating; without this the
        banner would flick Survivors -> Out-of-Sample -> Survivors).
        ``reset`` is for a genuinely new run/round (loop mode). Evolution Lab
        cycles every generation, so it may move freely."""
        if self.kind == "evolution" or reset or phase >= self._phase:
            self._phase = phase

    def _search_phase(self, phase: int) -> int:
        """A Speed Run's inner search must not claim the final phase -- the
        long Full Pipeline validation that follows is still Out-of-Sample.
        Only Speed Run's own "Phase 3/3" line reaches Survivors."""
        return min(phase, 3) if self.kind == "speed_run" else phase

    def _feed_locked(self, line: str) -> None:
        m = _RE_SPACE_READY.search(line)
        if m:
            # A new search run is starting. In loop mode (or a Speed Run's next
            # round) the previous run's counters reset to zero inside the
            # engine, so bank them first -- the banner shows candidates across
            # the WHOLE job and the rate stays honest.
            self._banked += self._done_count
            self._done_count = 0
            self._total = _int(m.group(1))
            self._set_phase(1, reset=True)
            return
        m = _RE_STAGE1_PROGRESS.search(line)
        if m:
            self._done_count, self._total = _int(m.group(1)), _int(m.group(2))
            self._set_phase(1)
            return
        m = _RE_STAGE1_DONE.search(line)
        if m:
            self._survivors, self._done_count = _int(m.group(1)), _int(m.group(2))
            self._total = self._done_count
            self._set_phase(1)
            return
        if _RE_STAGE2_START.search(line):
            self._set_phase(self._search_phase(2))
            return
        m = _RE_STAGE2_DONE.search(line)
        if m:
            self._survivors = _int(m.group(1))
            self._set_phase(self._search_phase(3))
            return
        m = _RE_STAGE3_PROGRESS.search(line)
        if m:
            self._set_phase(self._search_phase(3))
            return
        if _RE_STAGE4_START.search(line):
            self._set_phase(self._search_phase(4))
            return

        if _RE_SPEED_P1.search(line):
            self._set_phase(1, reset=True)
            return
        m = _RE_SPEED_P1_DONE.search(line)
        if m:
            self._done_count = self._total = _int(m.group(1))
            self._survivors = _int(m.group(4))
            self._set_phase(3)
            return
        if _RE_SPEED_P2.search(line):
            self._set_phase(3)
            return
        if _RE_SPEED_P3.search(line):
            self._set_phase(4)
            return
        if _RE_SPEED_DONE.search(line):
            self._set_phase(4)
            return

        m = _RE_EVO_GEN.search(line)
        if m:
            self._generation, self._phase = int(m.group(1)), 2
            return
        if _RE_EVO_GENERATE.search(line):
            self._phase = 2
            return
        m = _RE_EVO_PREFILTER.search(line)
        if m:
            self._done_count += _int(m.group(2))  # everything generated this round has now been backtested
            self._survivors = _int(m.group(1))
            self._phase = 1
            return
        if _RE_EVO_ROBUST.search(line) or _RE_EVO_CPCV.search(line):
            self._phase = 3
            return
        m = _RE_EVO_STRESS.search(line)
        if m:
            self._survivors, self._phase = _int(m.group(1)), 3
            return
        m = _RE_EVO_COMPLETE.search(line)
        if m:
            self._generation, self._phase = int(m.group(1)), 4
            if m.group(2) is not None:
                try:
                    self._best_fitness = float(m.group(2))
                except ValueError:
                    pass

    # -- reading ---------------------------------------------------------
    def snapshot(self) -> ProgressSnapshot:
        with self._lock:
            end = self._finished_at if self._finished_at is not None else self._clock()
            elapsed = max(0.0, end - self._started_at)

            done = self._finished_at is not None
            done_count = self._banked + self._done_count
            # Under a second of data gives absurd rates (thousands/s from one
            # early line), so report none until there's a meaningful window.
            rate = (done_count / elapsed) if elapsed >= 1.0 and done_count else 0.0
            if self._phase <= 0:
                key, label = "starting", "Starting"
            else:
                key, label = PHASES[self._phase - 1]
            total = (self._banked + self._total) if self._total is not None else None
            if total is not None and total < done_count:
                total = done_count  # a stale/lower total must never read as "more done than exist"
            return ProgressSnapshot(
                kind=self.kind, phase_index=self._phase, phase_key=key, phase_label=label,
                candidates_done=done_count, candidates_total=total,
                generation=self._generation, survivors=self._survivors,
                best_fitness=self._best_fitness, rate_per_sec=round(rate, 2),
                elapsed_seconds=round(elapsed, 1), done=done,
                banner=format_banner(self._phase, label, done_count, rate, self._generation, done),
            )


# ---------------------------------------------------------------------------
# Full Pipeline weighted completion (v9.13)
#
# Owen: "Make the progress bar on the right show completion percentage
# for all the runs during the run." The pipeline's seven steps have
# wildly unequal costs, so a plain step counter would sit at "Step 2/7"
# for half the run; instead each step carries a weight approximating
# its typical share of a full run's wall clock. Calibrated on the two
# measured 471k-bar configurations in the v9.13 proof set (ES 5m
# momentum / Lucid 50k and GC 1h trend / Apex 100k -- their post-fix
# step shares averaged roughly 4/37/17/1/0/18/24 percent, with the
# fold/holdout steps kept slightly above their measured share because
# configurations whose winner differs from the baseline pay real
# backtests there). The weights are documented here and in
# CHANGES_SUMMARY.md; the bar is an honest estimate, not a promise --
# the within-step fractions (GA generation, Monte Carlo paths, null
# runs) are exact counts of work done.
PIPELINE_STEP_LABELS: tuple[str, ...] = (
    "Baseline",
    "Search (walk-forward GA)",
    "Final validation",
    "Out-of-sample folds",
    "Holdout",
    "Significance gates",
    "Evidence + report",
)
PIPELINE_STEP_WEIGHTS: tuple[float, ...] = (0.05, 0.38, 0.17, 0.03, 0.02, 0.15, 0.20)


def pipeline_percent(step: int, fraction: float) -> float:
    """Overall weighted completion (0-100) for a 1-based pipeline step
    at `fraction` (0-1) within that step. Pure function of the weights
    above so it is unit-testable without any job plumbing."""
    step = max(1, min(int(step), len(PIPELINE_STEP_WEIGHTS)))
    fraction = max(0.0, min(1.0, float(fraction)))
    base = sum(PIPELINE_STEP_WEIGHTS[: step - 1])
    return round(100.0 * (base + PIPELINE_STEP_WEIGHTS[step - 1] * fraction), 1)


class PipelineProgress:
    """Stateful tracker behind the Full Pipeline job page's right-rail
    percentage. The pipeline reports (step, fraction) via its
    progress_hook; this turns it into a monotonic overall percent.

    Multi-run jobs (the timeframe sweep runs the whole pipeline once
    per timeframe) pass run_count > 1: when a fresh run's Step 1
    arrives after a previous run already reported progress, the tracker
    rolls into the next run's slice of the bar instead of restarting
    at 0. Thread-safe; finish() pins the percent at 100."""

    def __init__(self, run_count: int = 1, clock: Callable[[], float] = time.time) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._started_at = clock()
        self._finished_at: Optional[float] = None
        self._run_count = max(1, int(run_count))
        self._run_index = 0
        self._run_pct = 0.0          # weighted percent within the current run
        self._percent = 0.0          # overall percent (monotonic)
        self._step: Optional[int] = None
        self._step_label = ""
        self._label = ""

    def set_run_count(self, run_count: int) -> None:
        with self._lock:
            self._run_count = max(1, int(run_count))

    def update(self, step: int, step_total: int, fraction: float, label: str = "") -> dict:
        with self._lock:
            step = int(step)
            if step == 1 and float(fraction) <= 0.0 and self._step is not None and self._finished_at is None:
                # A new run of a multi-run job is starting (timeframe
                # sweep): roll into the next run's slice of the bar.
                self._run_index = min(self._run_index + 1, self._run_count - 1)
                self._run_pct = 0.0
            run_pct = pipeline_percent(step, fraction)
            if run_pct > self._run_pct:
                self._run_pct = run_pct
            self._step = step
            if 1 <= step <= len(PIPELINE_STEP_LABELS):
                self._step_label = PIPELINE_STEP_LABELS[step - 1]
            if label:
                self._label = label
            overall = (self._run_index + self._run_pct / 100.0) / self._run_count * 100.0
            if overall > self._percent:
                self._percent = round(overall, 1)
            return self._payload_locked()

    def finish(self) -> dict:
        with self._lock:
            if self._finished_at is None:
                self._finished_at = self._clock()
            self._percent = 100.0
            self._run_pct = 100.0
            return self._payload_locked()

    def payload(self) -> dict:
        with self._lock:
            return self._payload_locked()

    def _payload_locked(self) -> dict:
        end = self._finished_at if self._finished_at is not None else self._clock()
        return {
            "percent": self._percent,
            "step": self._step,
            "step_total": len(PIPELINE_STEP_WEIGHTS),
            "step_label": self._step_label,
            "label": self._label,
            "elapsed_seconds": round(max(0.0, end - self._started_at), 1),
            "done": self._finished_at is not None,
        }
