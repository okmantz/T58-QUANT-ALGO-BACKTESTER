"""
Adaptive Family Budget -- Owen's "killer feature: adaptive search" ask:
"T58 shouldn't continue allocating equal resources [to every family].
It should say: failed-breakout + volatility-expansion are producing
substantially more viable candidates. Then allocate more computation
there."

app.search.family_health already does the coarse, cross-run version of
this: a family that's failed min_samples (default 30) times with ZERO
successes, ever, gets excluded entirely. That's a binary, permanent,
slow-to-trigger safety net -- appropriate for "this family is dead,"
not for "this family is currently converting immigrants into pre-filter
survivors at 3x the rate of that one, so give it more of THIS run's
population budget starting next generation."

This module is the graduated, intra-run version that sits alongside it:
every generation, EvolutionRunner reports how many candidates of each
family were generated / survived PRE-FILTER / survived STRESS, and this
tracker turns that rolling history into a per-family multiplier applied
to app.evolution.engine's per-family immigrant count. A family with no
history yet gets multiplier 1.0 (no opinion); a family that's been
producing pre-filter survivors gets boosted; a family that's been
tested a meaningful number of times with nothing surviving gets
shrunk -- but never to zero and never below the caller's own
min_immigrants_per_family floor, since going to zero IS family_health's
job (a permanent, deliberate decision), not this module's (a soft,
every-generation nudge that a later generation's results can reverse).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class _FamilyWindow:
    tested: deque = field(default_factory=lambda: deque(maxlen=1))
    prefilter_passed: deque = field(default_factory=lambda: deque(maxlen=1))
    stress_passed: deque = field(default_factory=lambda: deque(maxlen=1))


class FamilyBudgetTracker:
    def __init__(self, window: int = 10, min_frac: float = 0.4, max_frac: float = 2.5):
        """window: how many recent generations' worth of per-family
        counts to keep (older generations age out, so a family's budget
        can recover if it starts working again after a cold stretch --
        family_health's cross-run exclusion has no such recovery path by
        design; this one deliberately does). min_frac/max_frac: the
        multiplier floor/ceiling applied to a family's baseline
        per-family immigrant count."""
        self.window = max(1, window)
        self.min_frac = min_frac
        self.max_frac = max_frac
        self._data: dict[str, _FamilyWindow] = {}

    def _bucket(self, family: str) -> _FamilyWindow:
        b = self._data.get(family)
        if b is None or b.tested.maxlen != self.window:
            b = _FamilyWindow(
                tested=deque(maxlen=self.window),
                prefilter_passed=deque(maxlen=self.window),
                stress_passed=deque(maxlen=self.window),
            )
            self._data[family] = b
        return b

    def record_generation(self, family_counts: dict[str, dict]) -> None:
        """family_counts: {family: {"tested": N, "prefilter_passed": N,
        "stress_passed": N}} for ONE generation -- call once per
        generation with that generation's counts (not cumulative; this
        class does the rolling accumulation itself)."""
        for fam, counts in family_counts.items():
            bucket = self._bucket(fam)
            bucket.tested.append(int(counts.get("tested", 0)))
            bucket.prefilter_passed.append(int(counts.get("prefilter_passed", 0)))
            bucket.stress_passed.append(int(counts.get("stress_passed", 0)))

    def multiplier(self, family: str) -> float:
        """1.0 for a family with no recorded history yet (no opinion --
        never punish a family for not having been tried). Otherwise a
        weighted combination of pre-filter survival rate (cheap, high-
        volume signal) and stress survival rate (rare, but the real
        signal a candidate actually has a shot at surviving the full
        pipeline), clamped to [min_frac, max_frac]."""
        bucket = self._data.get(family)
        if bucket is None or sum(bucket.tested) == 0:
            return 1.0
        tested = sum(bucket.tested)
        prefilter_rate = sum(bucket.prefilter_passed) / tested if tested else 0.0
        stress_rate = sum(bucket.stress_passed) / tested if tested else 0.0
        # Stress survival is the rarer, more valuable signal -- weighted
        # heavier than raw pre-filter survival, which a family can rack
        # up on noise alone (see app.evolution.prop_fitness's own
        # discounting of tiny-sample-size "survivors"). A 1.0 baseline
        # plus a bounded bonus/penalty keeps a family with a handful of
        # early stress survivors from swinging all the way to max_frac
        # off a sample of one or two generations.
        raw = 1.0 + prefilter_rate * 1.5 + stress_rate * 6.0
        # Families tested a meaningful number of times (not just 1-2
        # immigrants) with truly zero signal get pulled down, not just
        # left at neutral -- this is the "shrink, don't just fail to
        # grow" half of the ask.
        if tested >= 3 * self.window and prefilter_rate == 0.0:
            raw = self.min_frac
        return max(self.min_frac, min(self.max_frac, raw))

    def multipliers(self, families: list[str]) -> dict[str, float]:
        return {fam: self.multiplier(fam) for fam in families}

    def status(self) -> dict:
        """Per-family rolling totals + resolved multiplier, for
        surfacing in a UI/status() dict -- exactly the table Owen's own
        note sketches ('Family | Tested | Survivors')."""
        out = {}
        for fam, bucket in self._data.items():
            tested = sum(bucket.tested)
            out[fam] = {
                "tested": tested,
                "prefilter_passed": sum(bucket.prefilter_passed),
                "stress_passed": sum(bucket.stress_passed),
                "multiplier": round(self.multiplier(fam), 3),
            }
        return out
