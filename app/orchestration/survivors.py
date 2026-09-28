"""
"Proof of the survivors": how many strategies in the Strategy Library have
made it through each stage of the research pipeline.

Reads the SAME per-strategy progress the dashboard already shows
(app.strategy.library.compute_pipeline_progress: Create -> Test -> Optimize ->
Validate -> Champion Check -> Forward Test -> Deploy), so this strip can never
disagree with the per-strategy progress bars -- it just counts them.

Deliberately reports only what the library actually records. There is no
payout ledger in this app, so the strip stops at the "Deploy" stage flag; it
never invents a payout figure.

``build_funnel`` is pure (takes metadata dicts) so it is unit-tested with no
filesystem; ``load_library_funnel`` is the thin IO wrapper.
"""
from __future__ import annotations

from typing import Any, Iterable

from app.strategy.library import compute_pipeline_progress


def build_funnel(metadata_items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counts, per pipeline stage, how many strategies have completed it.

    Stages come straight from compute_pipeline_progress (in its order), so a
    stage added there appears here automatically. ``pct_of_total`` is each
    stage's count over the whole library (not over the previous stage --
    stages can be completed out of order, so a stage-over-stage ratio could
    exceed 100% and would mislead).
    """
    items = list(metadata_items)
    total = len(items)
    counts: dict[str, int] = {}
    titles: dict[str, str] = {}
    order: list[str] = []
    # Seed the stage list from an empty strategy so the strip always shows the
    # full pipeline, even for an empty library (otherwise it would be blank).
    for stage in compute_pipeline_progress({})["stages"]:
        counts[stage["key"]], titles[stage["key"]] = 0, stage["title"]
        order.append(stage["key"])
    ready_verdicts = 0
    for meta in items:
        progress = compute_pipeline_progress(meta or {})
        for stage in progress["stages"]:
            key = stage["key"]
            if key not in counts:
                counts[key], titles[key] = 0, stage["title"]
                order.append(key)
            if stage["done"]:
                counts[key] += 1
        if (progress.get("verdict") or "").upper() == "READY":
            ready_verdicts += 1
    stages = [
        {
            "key": key, "title": titles[key], "count": counts[key],
            "pct_of_total": round(100.0 * counts[key] / total, 1) if total else 0.0,
        }
        for key in order
    ]
    return {"total": total, "stages": stages, "ready_verdict_count": ready_verdicts}


def load_library_funnel() -> dict[str, Any]:
    """The funnel for every strategy currently saved in the library."""
    from app.strategy.library import list_saved_strategies

    return build_funnel(s.metadata for s in list_saved_strategies())
