"""Strategy Discovery Engine -- ONE tool that finds strategies for you (2026-10-26 rebuild of the 2026-10-25 hub).

Search Lab, Evolution Lab, Forge, Speed Run and the hypothesis layer are ONE engine with several presets (app.discovery_engine). The page used
to be a hub of links to each lab; it is now the tool itself: pick the market data and a prop firm, choose how thoroughly to look, press Start.

  GET  /discovery-engine                         the start page (3 steps + one button) with the engine details tucked underneath
  POST /discovery-engine/start                   runs the engine as a background job and redirects to its progress page
  GET  /discovery-engine/job/<id>                live progress, then the results in plain language
  GET  /discovery-engine/job/<id>/status.json    what the progress page polls
  POST /discovery-engine/job/<id>/stop           cooperative cancel
  POST /discovery-engine/job/<id>/save/<n>       save champion n to the Strategy Library (then Test it)
  GET  /api/discovery-engine/status              engine facts as JSON (unchanged)
  POST /api/discovery-engine/preregister         write the kill conditions of the report's 100 families to the hypothesis store (the start
                                                 route also does this automatically, so nobody has to know it exists)
"""
from __future__ import annotations

import json
import re
import threading

from flask import Blueprint, jsonify, redirect, render_template, request, session

discovery_engine_bp = Blueprint("discovery_engine", __name__)

JOB_DISCOVERY_ENGINE = "Strategy Discovery Engine"

# How thoroughly to look -> (engine preset, headline). Plain words on the page; the preset names stay internal.
EFFORT = {
    "quick": ("speed_run", "Quick look", "A fast, narrow scan. Good for a first pass or a small machine."),
    "standard": ("search_lab", "Standard search", "Every strategy type is tried across its settings grid, then the best are checked hard."),
    "deep": ("forge", "Deep search", "The widest search: thousands of ideas, screened and validated for prop-firm survival. Takes the longest."),
}
METHODS = {  # advanced override
    "": "Let the engine choose",
    "search_lab": "Grid sweep (Search Lab method)",
    "evolution_lab": "Evolve a population (Evolution Lab method)",
    "forge": "Wide generate-and-screen (Forge method)",
    "speed_run": "Tight-budget scan (Speed Run method)",
    "discovery": "Pre-registered hypotheses only",
}
STEP_LABELS = {
    "reproduction": "Gives the same result when re-run",
    "sequence_deflated": "Still good after counting every strategy ever tried",
    "null_gate": "Beats random entry timing",
    "pooled_edge": "Works on other markets too",
    "sealed_slice": "Held up on data it never saw",
    "entry_floor": "Enough trades and a clear enough edge to judge",
    "not_promoted": "Not checked further",
}

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



def _family_title(name: str) -> str:
    return str(name or "?").replace("_", " ").strip().capitalize()


def _pool_frames(label: str, df, limit: int = 6, bars: int = 12000):
    """Other stored markets at the SAME bar size, for the 'works on other markets too' check. Best-effort and bounded."""
    frames, risks = {}, {}
    try:
        from app.data.importer import import_csv
        from app.data.storage import list_datasets_grouped_for_picker, resolve_stored_dataset
        m = re.search(r"(\d+)\.csv$", str(label), re.I)
        suffix = m.group(1) if m else None
        if not suffix:
            return frames, risks
        mine = _symbol_for(label)
        seen = {mine}
        for g in list_datasets_grouped_for_picker():
            for f in g["files"]:
                if len(frames) >= limit:
                    break
                fm = re.search(r"(\d+)\.csv$", f["name"], re.I)
                if not fm or fm.group(1) != suffix or f.get("empty"):
                    continue
                sym = _symbol_for(f["full_name"])
                if sym in seen:
                    continue
                path = resolve_stored_dataset(f["full_name"])
                if path is None:
                    continue
                res = import_csv(path)
                if not res.is_valid:
                    continue
                seen.add(sym)
                frames[sym] = res.dataframe.tail(bars).reset_index(drop=True)
                risks[sym] = _risk_for(f["full_name"], 100000.0, 0.5)
    except Exception:  # noqa: BLE001 -- the pooled check degrades to whatever markets were found; it never blocks a run
        pass
    return frames, risks


def _symbol_for(label: str) -> str:
    from app.data.instrument_specs import guess_any_instrument_symbol
    return (guess_any_instrument_symbol(label) or re.sub(r"\W+", "", str(label).split("/")[0]).upper() or "MARKET")


def _risk_for(label: str, balance: float, risk_pct: float):
    from app.backtest.risk import RiskConfig
    from app.data.instrument_specs import apply_any_instrument_spec, guess_any_instrument_symbol, micro_equivalent
    risk = RiskConfig(initial_balance=balance, risk_mode="percent", risk_value=risk_pct)
    sym = guess_any_instrument_symbol(label)
    if sym:
        m = micro_equivalent(sym)
        try:
            risk = apply_any_instrument_spec(risk, m.symbol if m else sym)
        except Exception:  # noqa: BLE001
            pass
    return risk


def _page_context(**extra) -> dict:
    from app.data.storage import list_datasets_grouped_for_picker
    try:
        groups = list_datasets_grouped_for_picker()
    except Exception:  # noqa: BLE001
        groups = []
    ctx = dict(active_page="discovery_engine", st=engine_status(), dataset_groups=groups, effort=EFFORT, methods=METHODS,
               csrf=session.get("csrf_token", ""), error=None, form={})
    ctx.update(extra)
    return ctx


@discovery_engine_bp.route("/discovery-engine")
def page():
    return render_template("discovery_engine.html", **_page_context(form={"effort": request.args.get("effort", "standard")}))


def _run_engine_job(job_id, req, held_guard: str):
    from app.orchestration.resource_guard import HEAVY_JOB_GUARD
    from app.reports.crash_log import log_crash
    from app.web.job_manager import JOB_MANAGER
    try:
        from app.discovery_engine.engine import DiscoveryEngine
        result = DiscoveryEngine().run(req)
        JOB_MANAGER.finish(job_id, engine_result=result, cancelled=bool(req.cancel_event and req.cancel_event.is_set()))
    except Exception as exc:  # noqa: BLE001 -- must surface on the progress page, not die silently in the thread
        log_crash("Strategy Discovery Engine (web)", exc=exc)
        JOB_MANAGER.fail(job_id, f"The engine stopped with an error: {exc}")
    finally:
        HEAVY_JOB_GUARD.release(held_guard)


@discovery_engine_bp.route("/discovery-engine/start", methods=["POST"])
def start():
    from app.data.importer import import_csv
    from app.data.storage import resolve_stored_dataset
    from app.data.timeframe_resample import infer_timeframe_label
    from app.discovery.hypothesis import preregister_v13_families
    from app.discovery_engine.engine import EngineRequest
    from app.orchestration.resource_guard import HEAVY_JOB_GUARD
    from app.prop.presets import get_preset as get_prop_preset
    from app.prop.simulator import PropRules
    from app.web.job_manager import JOB_MANAGER

    form = request.form
    fdict = {k: form.get(k, "") for k in ("dataset", "prop_preset", "effort", "method", "risk_pct", "max_bars", "families", "seed")}

    def back(msg, code=400):
        return render_template("discovery_engine.html", **_page_context(error=msg, form=fdict)), code

    label = (form.get("dataset") or "").strip()
    if not label:
        return back("Pick the market data to search on (step 1).")
    path = resolve_stored_dataset(label)
    if path is None:
        return back("That dataset couldn't be found. Pick another one, or import it from the Data section first.")
    res = import_csv(path)
    if not res.is_valid:
        return back("That dataset couldn't be read: " + "; ".join(res.errors)[:200])
    df = res.dataframe
    try:
        max_bars = int(form.get("max_bars") or 60000)
    except ValueError:
        max_bars = 60000
    if max_bars > 0 and len(df) > max_bars:
        df = df.tail(max_bars).reset_index(drop=True)
    if len(df) < 500:
        return back("That dataset has too few bars to search on (needs at least 500).")

    try:
        preset_row = get_prop_preset(form.get("prop_preset") or "ftmo_100k")
        rules = preset_row.to_prop_rules()
        balance = float(preset_row.account_size)
        prop_label = preset_row.label
    except Exception:  # noqa: BLE001
        rules, balance, prop_label = PropRules(), 100000.0, "default $100k rules"
    try:
        risk_pct = max(0.05, min(5.0, float(form.get("risk_pct") or 0.5)))
    except ValueError:
        risk_pct = 0.5
    risk = _risk_for(label, balance, risk_pct)

    effort = form.get("effort") if form.get("effort") in EFFORT else "standard"
    preset_name = (form.get("method") or "").strip() or EFFORT[effort][0]
    if preset_name not in METHODS:
        preset_name = EFFORT[effort][0]
    families = {"": None, "ready": "ready", "all": "all"}.get((form.get("families") or "").strip(), None)
    try:
        seed = int(form.get("seed") or 42)
    except ValueError:
        seed = 42

    if not HEAVY_JOB_GUARD.try_acquire(JOB_DISCOVERY_ENGINE):
        return back(f"{HEAVY_JOB_GUARD.active_name} is already running. Running two heavy jobs at once can exhaust memory -- "
                    "wait for it to finish, then start the engine.", 409)
    try:
        preregister_v13_families()      # the engine refuses families without a written kill condition; do it for the person
        tf = infer_timeframe_label(df)
        frames, risks = _pool_frames(label, df)
        sym = _symbol_for(label)
        frames[sym], risks[sym] = df, risk
        cancel = threading.Event()
        pool_note = (f"Also checking the best ideas on {len(frames) - 1} other market(s) at the same bar size." if len(frames) > 1
                     else "No other markets at this bar size are stored, so the 'works on other markets' check will be limited.")
        job_id = JOB_MANAGER.create(
            page_template="/discovery-engine/job/{job_id}", tool="Strategy Discovery Engine", instrument=label, cancel_event=cancel,
            log=[f"Loaded {len(df):,} bars of {label} ({tf}).", f"Looking: {EFFORT[effort][1]} ({METHODS.get(preset_name, preset_name)}).", pool_note],
            engine_result=None, cancelled=False, settings={"dataset": label, "timeframe": tf, "effort": effort, "prop": prop_label, "risk_pct": risk_pct})
        JOB_MANAGER.prune(max_age_seconds=6 * 3600)
        req = EngineRequest(preset=preset_name, df=df, risk=risk, prop_rules=rules, instrument=label, timeframe=tf, families=families, seed=seed,
                            frames=frames, risks=risks, progress_cb=lambda m: JOB_MANAGER.log(job_id, str(m)), cancel_event=cancel)
        threading.Thread(target=_run_engine_job, args=(job_id, req, JOB_DISCOVERY_ENGINE), daemon=True).start()
        return redirect(f"/discovery-engine/job/{job_id}")
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_DISCOVERY_ENGINE)
        from app.reports.crash_log import log_crash
        log_crash("Strategy Discovery Engine (web, start)", exc=exc)
        return back(f"Couldn't start the engine: {exc}", 500)


@discovery_engine_bp.route("/discovery-engine/job/<job_id>")
def job_page(job_id):
    from app.web.job_manager import JOB_MANAGER
    job = JOB_MANAGER.get(job_id)
    return render_template("discovery_engine_job.html", job_id=job_id, not_found=job is None, active_page="discovery_engine",
                           csrf=session.get("csrf_token", "")), (404 if job is None else 200)


def _champion_view(i, c, p):
    steps = []
    for s in (p or {}).get("steps") or []:
        steps.append({"label": STEP_LABELS.get(s.get("name"), s.get("name")), "state": "skipped" if s.get("skipped") else ("pass" if s.get("passed") else "fail"),
                      "detail": s.get("detail") or ""})
    failed = (p or {}).get("failed_step")
    edge = c.get("edge") or {}
    return {
        "index": i, "family": c.get("family"), "family_title": _family_title(c.get("family")), "candidate_id": c.get("candidate_id"),
        "passed": bool((p or {}).get("passed")), "failed_step": STEP_LABELS.get(failed, failed) if failed else None,
        "failed_reason": (p or {}).get("failed_reason") or "", "edge_lb_r": c.get("edge_lb_r"),
        "trades": edge.get("n_trades") or edge.get("trades"), "mean_r": edge.get("mean_r"),
        "steps": steps, "floor_reasons": c.get("floor_reasons") or [],
        "can_save": bool((c.get("spec") or {}).get("source_type", "manual") == "manual" and (c.get("spec") or {}).get("config")),
    }


@discovery_engine_bp.route("/discovery-engine/job/<job_id>/status.json")
def job_status(job_id):
    import time

    from app.web.job_manager import JOB_MANAGER
    job = JOB_MANAGER.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    out = {"found": True, "done": bool(job.get("done")), "error": job.get("error"), "cancelled": bool(job.get("cancelled")),
           "log": [str(x) for x in (job.get("log") or [])][-200:], "elapsed": round((job.get("finished_at") or time.time()) - job.get("started_at", time.time())),
           "settings": job.get("settings") or {}}
    res = job.get("engine_result")
    if job.get("done") and res is not None:
        champs = [_champion_view(i, c, p) for i, (c, p) in enumerate(zip(res.champions, res.promotions))]
        out["result"] = {
            "n_evaluations": res.n_evaluations, "families_run": len(res.families_run), "n_forward": len(res.passed),
            "champions": champs, "notes": res.notes, "elapsed_seconds": round(res.elapsed_seconds),
            "effective_trials": (res.trial_inputs or {}).get("n_tests"),
        }
    return jsonify(out)


@discovery_engine_bp.route("/discovery-engine/job/<job_id>/stop", methods=["POST"])
def job_stop(job_id):
    from app.web.job_manager import JOB_MANAGER
    job = JOB_MANAGER.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job not found."}), 404
    ev = job.get("cancel_event")
    if ev is not None and not job.get("done"):
        ev.set()
        JOB_MANAGER.log(job_id, "Stop requested -- finishing the step in progress...")
    return jsonify({"ok": True})


@discovery_engine_bp.route("/discovery-engine/job/<job_id>/save/<int:n>", methods=["POST"])
def job_save(job_id, n):
    from app.strategy.library import StrategyAlreadyExists, safe_filename_stem, save_strategy_text
    from app.web.job_manager import JOB_MANAGER
    job = JOB_MANAGER.get(job_id)
    res = (job or {}).get("engine_result")
    if res is None or n < 0 or n >= len(res.champions):
        return jsonify({"ok": False, "error": "That result is no longer available (the server may have restarted)."}), 404
    c = res.champions[n]
    spec = c.get("spec") or {}
    if spec.get("source_type", "manual") != "manual" or not spec.get("config"):
        return jsonify({"ok": False, "error": "This candidate isn't a saveable builder strategy."}), 400
    cfg = dict(spec["config"])
    base = safe_filename_stem(f"{_family_title(c.get('family'))} {str(job.get('instrument', '')).split('/')[-1].replace('.csv', '')} discovery", "discovery_strategy")
    cfg["name"] = base
    for k in range(0, 50):
        fn = f"{base}{'' if k == 0 else '_' + str(k + 1)}.json"
        try:
            path = save_strategy_text(json.dumps(cfg, indent=2, default=str), fn, "manual")
            return jsonify({"ok": True, "name": path.name, "test_url": f"/full-pipeline?load_strategy_type=manual&load_strategy_name={path.name}"})
        except StrategyAlreadyExists:
            continue
    return jsonify({"ok": False, "error": "Couldn't find a free file name."}), 409
