"""One honest trial count for every verdict.

Before this module each tool counted trials its own way: Quick Optimize Bonferroni-corrected for its own GA evaluations only,
Full Pipeline added the ledger, Search Lab Stage 4 counted Stage-1 records, Evolution Lab counted nothing. Here every caller asks
the same three questions:

  ``lifetime_trials``     how many candidates has this family/strategy EVER had evaluated on this data window (ledger, no decay
                          on an unchanged window; half-life decay only when the data window changed)
  ``effective_trials``    how many DISTINCT ideas is that (|corr| >= threshold clustering of daily P&L, scaled to the raw count)
  ``TrialCounter``       the per-run tally of EVERY evaluation, split by kind, booked once into the ledger at the end of the run

Kinds booked: ``prefilter_kills``, ``stage1``, ``ga_inner_loop``, ``grammar_mutations``, ``walk_forward``, ``seed_evals``,
``other``. Nothing that scored a candidate is free.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

DEFAULT_CLUSTER_CORR = 0.85
KINDS = ("prefilter_kills", "stage1", "ga_inner_loop", "grammar_mutations", "walk_forward", "seed_evals", "other")


class TrialCounter:
    """Thread-safe per-run tally of evaluations by (family, kind)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_family: dict[str, int] = {}
        self._by_kind: dict[str, int] = {}

    def add(self, family: str, n: int = 1, kind: str = "other") -> None:
        n = int(n)
        if n <= 0:
            return
        fam = str(family or "unknown")
        k = kind if kind in KINDS else "other"
        with self._lock:
            self._by_family[fam] = self._by_family.get(fam, 0) + n
            self._by_kind[k] = self._by_kind.get(k, 0) + n

    def merge_counts(self, family_counts: dict, kind: str = "other") -> None:
        for fam, n in (family_counts or {}).items():
            self.add(fam, int(n or 0), kind)

    @property
    def total(self) -> int:
        with self._lock:
            return int(sum(self._by_family.values()))

    def by_family(self) -> dict[str, int]:
        with self._lock:
            return dict(self._by_family)

    def by_kind(self) -> dict[str, int]:
        with self._lock:
            return dict(self._by_kind)

    def book(self, run_key: str, data_fp: str | None = None, now: float | None = None) -> dict:
        """Write the tally into the lifetime ledger exactly once for ``run_key``. Best-effort: never raises."""
        try:
            from app.validation import trial_ledger
            return trial_ledger.book_run(run_key, self.by_family(), now=now, data_fp=data_fp, kinds=self.by_kind())
        except Exception:  # noqa: BLE001
            return {}


def lifetime_trials(names, *, data_fp: str | None = None, half_life_days: float | None = None) -> int:
    """Ledger total over one or more family / strategy names on the CURRENT data window.

    ``half_life_days`` default of ``None`` here means "use the ledger's default half-life, but never decay entries recorded on
    ``data_fp``". Pass an explicit number to override. Unknown names count as 0."""
    from app.validation import trial_ledger
    hl = trial_ledger.DEFAULT_HALF_LIFE_DAYS if half_life_days is None else half_life_days
    if isinstance(names, str):
        names = [names]
    tot = 0
    for n in names or []:
        if not n:
            continue
        label = str(n)
        if not label.lower().startswith("family:"):
            tot += trial_ledger.total(label, half_life_days=hl, current_fp=data_fp)
            tot += trial_ledger.family_prior(label, half_life_days=hl, current_fp=data_fp)
        else:
            tot += trial_ledger.family_prior(label, half_life_days=hl, current_fp=data_fp)
    return int(tot)


def effective_trials(raw_trials: int, streams=None, corr_threshold: float = DEFAULT_CLUSTER_CORR) -> dict:
    """{"raw_trials", "effective_trials", "n_clusters", "n_streams"} -- clusters the sample ``streams`` (daily P&L Series) and
    scales the cluster ratio to ``raw_trials``. With fewer than 3 streams the raw count is returned unchanged (never loosens)."""
    raw = max(1, int(raw_trials))
    streams = [s for s in (streams or []) if s is not None and len(s)]
    if len(streams) < 3:
        return {"raw_trials": raw, "effective_trials": raw, "n_clusters": len(streams), "n_streams": len(streams), "clustered": False}
    from app.validation.effective_trials import scale_trial_count
    info = scale_trial_count(raw, streams, corr_threshold=corr_threshold)
    return {"raw_trials": raw, "effective_trials": int(info["effective_trials"]), "n_clusters": int(info["n_clusters"]),
            "n_streams": int(info["n_streams"]), "clustered": True}


@dataclass
class TrialVerdictInputs:
    """Everything a caller needs to hand a deflation or Bonferroni test the same N."""

    run_evaluations: int
    ledger_prior: int
    raw_trials: int
    effective_trials: int
    clustered: bool
    notes: list = field(default_factory=list)

    @property
    def n_tests(self) -> int:
        return max(1, int(self.effective_trials))

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def trial_inputs(run_evaluations: int, *, names=(), data_fp: str | None = None, streams=None,
                 corr_threshold: float = DEFAULT_CLUSTER_CORR, extra: int = 0) -> TrialVerdictInputs:
    """Combine this run's evaluations with the lifetime ledger and cluster to effective trials.

    This is the ONE function every Deflated-Sharpe / Bonferroni call goes through (Quick Optimize, Full Pipeline, Search Lab Stage 4,
    Evolution Lab). The ledger part is read BEFORE this run is booked, so a run never double counts itself."""
    prior = lifetime_trials(list(names or []), data_fp=data_fp)
    raw = max(1, int(run_evaluations) + int(prior) + int(extra))
    eff = effective_trials(raw, streams, corr_threshold)
    notes = []
    if prior:
        notes.append(f"{prior:,} earlier evaluation(s) from the lifetime ledger added")
    if eff["clustered"] and eff["effective_trials"] != raw:
        notes.append(f"clustered {raw:,} raw -> {eff['effective_trials']:,} effective (|corr| >= {corr_threshold:g}, {eff['n_clusters']} stream(s))")
    return TrialVerdictInputs(int(run_evaluations), int(prior), raw, int(eff["effective_trials"]), bool(eff["clustered"]), notes)


def quick_optimize_n_tests(ga_total_evaluations: int, mutated_by_ga: bool, strategy_names=(), *, data_fp: str | None = None,
                           streams=None) -> TrialVerdictInputs:
    """Quick Optimize's Bonferroni count: this run's GA evaluations (1 when the GA did not mutate the strategy) PLUS the lifetime
    ledger for the strategy's name / family, clustered when daily-P&L streams are available. Never below what Quick Optimize
    counted before (``max(1, total_evaluations)``)."""
    own = max(1, int(ga_total_evaluations)) if mutated_by_ga else 1
    return trial_inputs(own, names=strategy_names, data_fp=data_fp, streams=streams)
