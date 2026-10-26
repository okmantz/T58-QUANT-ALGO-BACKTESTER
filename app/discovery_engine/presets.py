"""Presets: the old lab names as configurations of the ONE Strategy Discovery Engine.

Search Lab, Evolution Lab, Forge, Speed Run and the app/discovery layer used to be five labs with five objectives, five trade
floors and five trial counts. They are now five *presets* of one engine: the same objective (``edge_lb_r``), the same entry floor,
one lifetime trial ledger, one promotion sequence. A preset only chooses HOW candidates are generated and how big the budget is.

``prop_floor_mode`` is the one deliberate difference from the old Forge/Speed Run behaviour: Stage-3 prop-survival numbers (eval pass
probability, payout odds) no longer REJECT a strategy that has a proven edge -- prop pass probability is carried through to the
final stage, where it sizes the position (``app.prop``), exactly as the report asks. ``STAGE3_PROP_RELAXED`` lists the
SearchStageConfig fields that implement that; they are applied only when the field exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Fields of SearchStageConfig / ForgeConfig that gate on PROP-pass numbers at Stage 3. Relaxed (not removed) so prop pass probability
# is reported and used for last-stage sizing but is no longer a reason to discard a candidate that has an edge.
STAGE3_PROP_RELAXED: dict = {
    "min_eval_pass_probability": 0.0,
    "stage3_min_profit_factor": 0.0,
    "locked_oos_min_pass_rate": 0.0,
}


@dataclass(frozen=True)
class Preset:
    name: str
    label: str
    backend: str                      # search | evolution | forge | speed_run | discovery
    description: str
    families: object = "all"          # "all" | "ready" | list[str]
    max_candidates: int = 2000        # search-style budgets
    top_n_promote: int = 3            # how many ranked candidates enter the promotion sequence
    loop: bool = False                # loop mode: stop at the target, then the MANDATORY sequence runs and a failed step fails the run
    prop_floor_mode: str = "sizing_only"
    overrides: dict = field(default_factory=dict)


PRESETS: dict[str, Preset] = {p.name: p for p in (
    Preset("search_lab", "Search Lab", "search",
           "Finite family-grid sweep: every enabled family's grid is evaluated once and ranked by edge_lb_r.",
           families="all", max_candidates=2000, top_n_promote=3),
    Preset("evolution_lab", "Evolution Lab", "evolution",
           "Population evolution (family immigrants + grammar, elites, mutation) on the same objective, floor and trial ledger.",
           families="all", top_n_promote=3),
    Preset("evolution_loop", "Evolution Lab (loop until target)", "evolution",
           "Evolution that stops at the 70% target -- the stop is a CANDIDATE: the mandatory promotion sequence still has to pass.",
           families="all", loop=True, top_n_promote=1),
    Preset("forge", "Forge Strategy", "forge",
           "One-button generate, screen, validate: family hypotheses, cheap screen, CPCV/regime, rolling prop evaluation, locked OOS.",
           families="all", max_candidates=10_000, top_n_promote=3),
    Preset("speed_run", "Speed Run", "speed_run",
           "Tight-budget run: speed-tuned discovery across every enabled family, then the full pipeline on the leaders.",
           families="all", max_candidates=800, top_n_promote=2),
    Preset("discovery", "Hypothesis-first discovery", "discovery",
           "Pre-registered hypotheses only: each family runs with its written kill condition and nothing else.",
           families="ready", max_candidates=400, top_n_promote=3),
)}


def get_preset(name: str) -> Preset:
    key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"search": "search_lab", "evolution": "evolution_lab", "speedrun": "speed_run", "forge_strategy": "forge"}
    key = aliases.get(key, key)
    if key not in PRESETS:
        raise KeyError(f"Unknown discovery preset '{name}'. Known: {', '.join(PRESETS)}")
    return PRESETS[key]


def apply_overrides(cfg, preset: Preset, extra: dict | None = None):
    """Return ``cfg`` (a dataclass) with the preset's overrides, the prop-floor relaxation and ``extra`` applied -- only for fields
    that exist, so a preset never invents a config key."""
    from dataclasses import fields, replace
    names = {f.name for f in fields(cfg)}
    changes = {k: v for k, v in {**preset.overrides, **(extra or {})}.items() if k in names}
    if preset.prop_floor_mode == "sizing_only":
        changes.update({k: v for k, v in STAGE3_PROP_RELAXED.items() if k in names and k not in (extra or {})})
    return replace(cfg, **changes) if changes else cfg
