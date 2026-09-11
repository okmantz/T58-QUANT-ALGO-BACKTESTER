"""
Strategy Graveyard -- Owen's ask: "every rejected strategy should be
stored with why it failed... eventually you build a map of dead
strategy space."

app.search.family_health already answers "has this whole FAMILY ever
produced a real survivor" (a coarse, binary, cross-run signal). This
module answers the finer-grained question underneath that: for a
candidate that got far enough to be genuinely interesting (it survived
the cheap pre-filter and reached robustness/OOS/Monte Carlo/CPCV/stress
scoring) but still didn't make the leaderboard, WHY not, specifically --
so the next run doesn't spend another four hours re-discovering that
"RSI-extreme-reversion with a 68-period lookback and 1.8R target is
71% likely to fail Monte Carlo" the hard way.

Storage: JSON Lines, one row per rejected candidate, append-only --
same pattern as app.evolution.checkpoint's tested_candidates.jsonl and
app.evolution.knowledge_graph's own log, deliberately NOT a new
database dependency. Cheap to write (one line, no locking needed
beyond append), cheap to read back in full for a single run's worth of
history (tens of thousands of lines at most).

This module does not duplicate the pre-filter's own rejection log
(app.evolution.checkpoint's "stage": "prefilter" rows already cover
"91/91 unprofitable" at near-zero cost per candidate) -- it exists
specifically for candidates that got EXPENSIVE (robustness, Monte
Carlo, CPCV, stress) before dying, where the "why" is worth a full
sentence, not just a rejection-reason keyword.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.data.storage import get_app_base_dir


def default_graveyard_path() -> Path:
    return get_app_base_dir() / "data" / "evolution" / "strategy_graveyard.jsonl"


@dataclass
class GraveyardEntry:
    candidate_id: str
    family: str
    generation: int | None
    stage_died: str                 # "cpcv" | "stress" | "not_ranked" -- where it was cut
    reason: str                     # one-sentence, human-readable primary reason
    oos_result: str | None = None                # "negative" / "positive" / None if not computed
    neighbor_robustness_pct: float | None = None  # parameter-neighborhood stability, 0-100
    monte_carlo_failure_pct: float | None = None  # 100 - eval pass probability
    prop_sim_pass_pct: float | None = None        # raw eval pass probability (in-sample MC)
    cpcv_oos_pass_pct: float | None = None        # honest held-out estimate, if computed
    pbo: float | None = None
    fitness_score: float | None = None
    param_signature: str | None = None            # coarse "same neighborhood" fingerprint
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        lines = [f"{self.candidate_id}", "FAILED", "", f"Reason:\n{self.reason}", ""]
        if self.oos_result is not None:
            lines += [f"OOS:\n{self.oos_result}", ""]
        if self.neighbor_robustness_pct is not None:
            lines += [f"Neighbor robustness:\n{self.neighbor_robustness_pct:.0f}%", ""]
        if self.monte_carlo_failure_pct is not None:
            lines += [f"Monte Carlo:\nFailure probability {self.monte_carlo_failure_pct:.0f}%", ""]
        if self.prop_sim_pass_pct is not None:
            lines += [f"Prop simulation:\nPASS probability {self.prop_sim_pass_pct:.0f}%", ""]
        return "\n".join(lines).rstrip()


def _round_sig(v: float, digits: int = 1) -> float:
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return 0.0


def param_signature(family: str, config: dict | None, keys: tuple[str, ...] = ()) -> str:
    """A coarse fingerprint for 'is this the same neighborhood as
    something already in the graveyard' -- rounds numeric params to 1
    significant change so that gen258/gen260/gen264's near-identical
    mutations of the same elite (see app.evolution.engine's mutation
    step) collapse to one signature instead of three separate entries.
    Not a hash of the full config (irrelevant fields like display names
    would fragment the signature); if `keys` is empty, uses every
    numeric top-level key found. Best-effort: a config shape this can't
    walk simply signs as the family name alone, which still dedupes
    exact-family repeats even without parameter-level granularity.
    """
    if not config:
        return family
    try:
        items = []
        source = config if not keys else {k: config.get(k) for k in keys if k in config}
        for k, v in sorted(source.items()):
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                items.append(f"{k}={_round_sig(v)}")
        return family + "|" + ",".join(items) if items else family
    except Exception:  # noqa: BLE001 -- signature is a best-effort dedupe key, never load-bearing
        return family


def record_rejection(entry: GraveyardEntry, path: Path | str | None = None) -> None:
    path = Path(path) if path else default_graveyard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict(), default=str) + "\n")


def record_rejections(entries: list[GraveyardEntry], path: Path | str | None = None) -> None:
    if not entries:
        return
    path = Path(path) if path else default_graveyard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict(), default=str) + "\n")


def load_graveyard(path: Path | str | None = None, limit: int = 50_000) -> list[dict]:
    path = Path(path) if path else default_graveyard_path()
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:  # noqa: BLE001 -- a corrupt/partial file must not crash a caller just reading history
        return []
    return rows[-limit:]


@dataclass
class GraveyardCluster:
    """One 'dead neighborhood' -- every rejection sharing a
    param_signature, collapsed into one entry with a count, so 40
    near-identical mutations of the same failed idea show up as
    ONE line ('tested 40 times, still dead'), not 40 lines."""
    signature: str
    family: str
    n_tested: int
    most_common_reason: str
    worst_stage_reached: str
    best_prop_sim_pass_pct: float | None
    example_candidate_ids: list

    def to_dict(self) -> dict:
        return dict(self.__dict__)


_STAGE_RANK = {"not_ranked": 0, "cpcv": 1, "stress": 2}


def summarize_graveyard(rows: list[dict], top_n: int = 25) -> list[GraveyardCluster]:
    """Groups graveyard rows by param_signature (falling back to family
    alone for rows with no signature) and returns the `top_n` clusters
    with the most repeated attempts first -- that ordering is exactly
    'don't waste another 4 hours here,' the biggest neighborhood T58 has
    already proven dead, first."""
    by_sig: dict[str, list[dict]] = {}
    for r in rows:
        sig = r.get("param_signature") or r.get("family") or "unknown"
        by_sig.setdefault(sig, []).append(r)

    clusters: list[GraveyardCluster] = []
    for sig, group in by_sig.items():
        reasons = [g.get("reason") for g in group if g.get("reason")]
        most_common = max(set(reasons), key=reasons.count) if reasons else "unknown"
        best_pct = max(
            (g.get("prop_sim_pass_pct") for g in group if g.get("prop_sim_pass_pct") is not None),
            default=None,
        )
        worst_stage = max((g.get("stage_died", "not_ranked") for g in group), key=lambda s: _STAGE_RANK.get(s, 0))
        clusters.append(GraveyardCluster(
            signature=sig, family=group[0].get("family", "?"), n_tested=len(group),
            most_common_reason=most_common, worst_stage_reached=worst_stage,
            best_prop_sim_pass_pct=best_pct,
            example_candidate_ids=[g.get("candidate_id") for g in group[:3]],
        ))
    clusters.sort(key=lambda c: c.n_tested, reverse=True)
    return clusters[:top_n]


def render_graveyard_report(clusters: list[GraveyardCluster]) -> str:
    lines = ["Strategy Graveyard -- dead neighborhoods, most-tested first", ""]
    if not clusters:
        lines.append("(empty -- nothing has reached full evaluation and failed yet)")
        return "\n".join(lines)
    header = f"{'Family':<28}{'Tested':>8}{'Worst stage':>14}{'Best pass %':>12}   Reason"
    lines.append(header)
    lines.append("-" * len(header))
    for c in clusters:
        best = f"{c.best_prop_sim_pass_pct:.0f}%" if c.best_prop_sim_pass_pct is not None else "--"
        lines.append(f"{c.family:<28}{c.n_tested:>8}{c.worst_stage_reached:>14}{best:>12}   {c.most_common_reason}")
    return "\n".join(lines)


def is_known_dead_neighborhood(
    family: str, config: dict | None, rows: list[dict] | None = None,
    min_attempts: int = 8, keys: tuple[str, ...] = (),
    path: Path | str | None = None,
) -> tuple[bool, int]:
    """Cheap check before spending a full generation's compute on a
    candidate: has something with this exact param_signature already
    been tried at least `min_attempts` times and always died? Returns
    (is_dead, n_previous_attempts). rows can be pre-loaded (e.g. once
    per generation) to avoid re-reading the file per-candidate; if not
    given, loads from `path`/the default location every call."""
    sig = param_signature(family, config, keys=keys)
    rows = rows if rows is not None else load_graveyard(path)
    matches = [r for r in rows if (r.get("param_signature") or r.get("family")) == sig]
    return (len(matches) >= min_attempts, len(matches))
