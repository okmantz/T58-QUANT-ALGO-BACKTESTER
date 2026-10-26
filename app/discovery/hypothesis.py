"""
HYPOTHESIS objects and their record (2026-10-07 discovery layer).

A Hypothesis is a falsifiable claim, not a strategy: "markets that break a
40-bar high keep going" with a mechanism, a validated rule spec, the markets and
timeframes it must hold on, and the conditions that would kill it. Every test
run against it is appended as an Experiment, win or lose, with the number of
variants tried (so later significance can be deflated for the search). The
store is a single JSON file; nothing is ever deleted, only superseded -- a dead
idea's record is what stops it being re-discovered and re-believed.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUSES = ("proposed", "testing", "survived", "broken", "inconclusive")


@dataclass
class Experiment:
    kind: str                       # "grid" | "break_battery" | "manual"
    started_at: float
    summary: dict
    n_variants: int = 1             # variants/parameter sets tried to get this result
    passed: bool | None = None
    notes: str = ""


@dataclass
class Hypothesis:
    idea: str
    spec: dict
    mechanism: str = ""
    expected_edge: str = ""
    markets: list = field(default_factory=list)
    timeframes: list = field(default_factory=list)
    falsifiers: list = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "proposed"
    created_at: float = field(default_factory=time.time)
    experiments: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    source: str = "manual"          # "manual" | "keyword" | "llm"

    def add_experiment(self, exp: Experiment) -> None:
        self.experiments.append(exp)

    def total_variants_tried(self) -> int:
        return sum(max(1, int(e.n_variants)) for e in self.experiments) or 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        d = dict(d)
        d["experiments"] = [Experiment(**e) if isinstance(e, dict) else e for e in d.get("experiments", [])]
        return cls(**d)


class HypothesisStore:
    def __init__(self, path: str | Path | None = None):
        if path is None:
            from app.data.storage import get_app_base_dir
            path = Path(get_app_base_dir()) / "data" / "discovery" / "hypotheses.json"
        self.path = Path(path)

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save(self, hyp: Hypothesis) -> None:
        data = self._load()
        data[hyp.id] = hyp.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, hid: str) -> Hypothesis | None:
        d = self._load().get(hid)
        return Hypothesis.from_dict(d) if d else None

    def all(self) -> list[Hypothesis]:
        return [Hypothesis.from_dict(d) for d in self._load().values()]

    def find_by_spec(self, spec: dict) -> list[Hypothesis]:
        key = json.dumps(spec, sort_keys=True)
        return [h for h in self.all() if json.dumps(h.spec, sort_keys=True) == key]


# ------------------------------------------------------------------------------------------------ v13 pre-registration
V13_PREREG_PREFIX = "v13-"


def _v13_id(number: int) -> str:
    return f"{V13_PREREG_PREFIX}{int(number):03d}"


def preregister_v13_families(store: "HypothesisStore | None" = None) -> dict:
    """Write the kill condition of each of the report's 100 families into the hypothesis store BEFORE any of them has run
    (report item 16). Idempotent: an existing record keeps its original ``created_at`` and its experiments. Returns
    {"registered": n_new, "existing": n_existing, "total": 100}. Blocked families are recorded too (status ``proposed``, with the
    reason they cannot be run) so the register is complete and a blocked idea is not silently forgotten."""
    from app.search import families_v13 as f13
    store = store or HypothesisStore()
    new = old = 0

    def put(number: int, name: str, meta, spec_extra: dict) -> None:
        nonlocal new, old
        hid = _v13_id(number)
        if store.get(hid) is not None:
            old += 1
            return
        h = Hypothesis(
            idea=f"[{number}] {meta.hypothesis}",
            spec={"family": name, "number": number, "group": meta.group, "needs": meta.needs, "status": meta.status,
                  "instrument_tied": bool(meta.tied), "rollout_batch": meta.batch, **spec_extra},
            mechanism=meta.hypothesis, expected_edge="unknown -- the graveyard holds 1,100+ measured failures; treat as a hypothesis, not a winner",
            falsifiers=[meta.kill], id=hid, source="preregistered")
        store.save(h)
        new += 1

    for name, meta in f13.V13_META.items():
        put(meta.number, name, meta, {"retired_into": f13.RETIRED_V13.get(name, ""), "aux": meta.aux})
    for number, meta in f13.V13_BLOCKED.items():
        put(number, meta.name, meta, {"blocked_reason": meta.reason})
    return {"registered": new, "existing": old, "total": new + old}


def is_preregistered(family: str, store: "HypothesisStore | None" = None) -> bool:
    """True when ``family`` (a v13 family name) has its kill condition on record. Non-v13 families are not subject to the rule."""
    from app.search import families_v13 as f13
    if family not in f13.V13_NUMBER:
        return True
    store = store or HypothesisStore()
    h = store.get(_v13_id(f13.V13_NUMBER[family]))
    return bool(h and h.falsifiers)
