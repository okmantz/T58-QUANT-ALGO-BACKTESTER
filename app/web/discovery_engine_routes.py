"""Strategy Discovery Engine page and status API (2026-10-25).

Search Lab, Evolution Lab, Forge, Speed Run and the hypothesis layer are ONE engine with five presets (app.discovery_engine). This
blueprint shows what the engine is, what is switched on, and why:

  GET  /discovery-engine                       the page (presets, families by status and rollout batch, gate, retest record)
  GET  /api/discovery-engine/status            the same as JSON
  POST /api/discovery-engine/preregister       write the kill conditions of the report's 100 families to the hypothesis store

It is read-only apart from pre-registration; runs are launched from the existing lab pages (each is a preset) or from
``scripts/run_discovery_engine.py``.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, render_template

discovery_engine_bp = Blueprint("discovery_engine", __name__)

PRESET_PAGES = {"search_lab": "/search", "evolution_lab": "/evolution", "evolution_loop": "/evolution", "forge": "/forge",
                "speed_run": "/speed-run", "discovery": "/discover"}


def engine_status() -> dict:
    from app.discovery_engine import objective as obj
    from app.discovery_engine.presets import PRESETS
    from app.discovery_engine.retest import lab_ranking
    from app.discovery_engine.surrogate import gate_status
    from app.search import families_v13 as f13
    from app.search.strategy_space import FAMILIES, RETIRED_FAMILIES

    gate = gate_status()
    by_status: dict = {}
    for m in f13.V13_META.values():
        by_status[m.status] = by_status.get(m.status, 0) + 1
    fam_rows = sorted(({"number": m.number, "name": n, "group": m.group, "needs": m.needs, "status": ("retired" if n in f13.RETIRED_V13 else m.status),
                        "batch": m.batch, "tied": m.tied, "kill": m.kill} for n, m in f13.V13_META.items()), key=lambda r: r["number"])
    fam_rows += [{"number": k, "name": m.name, "group": m.group, "needs": m.needs, "status": "blocked", "batch": 9, "tied": True, "kill": m.reason}
                 for k, m in sorted(f13.V13_BLOCKED.items())]
    fam_rows.sort(key=lambda r: r["number"])
    return {
        "name": "Strategy Discovery Engine",
        "objective": obj.EDGE_OBJECTIVE,
        "floor": {"min_trades": obj.DEFAULT_FLOOR.min_trades, "min_t_stat": obj.DEFAULT_FLOOR.min_t_stat, "max_cost_in_r": obj.DEFAULT_FLOOR.max_cost_in_r},
        "presets": [{"name": p.name, "label": p.label, "backend": p.backend, "description": p.description, "page": PRESET_PAGES.get(p.name, "")} for p in PRESETS.values()],
        "families_total": len(FAMILIES), "families_retired": len(RETIRED_FAMILIES),
        "v13": {"counts": by_status, "blocked": len(f13.V13_BLOCKED), "rows": fam_rows},
        "gate": {"open": gate["open"], "reasons": gate["reasons"],
                 "null_rate": (gate["report"] or {}).get("null_rate"), "n_null": (gate["report"] or {}).get("n_null"),
                 "power": (gate["report"] or {}).get("power")},
        "retest_ranking": lab_ranking()[:12],
    }


@discovery_engine_bp.route("/api/discovery-engine/status")
def status_json():
    return jsonify(engine_status())


@discovery_engine_bp.route("/api/discovery-engine/preregister", methods=["POST"])
def preregister():
    from app.discovery.hypothesis import preregister_v13_families
    return jsonify(preregister_v13_families())


@discovery_engine_bp.route("/discovery-engine")
def page():
    return render_template("discovery_engine.html", active_page="discovery_engine", st=engine_status())
