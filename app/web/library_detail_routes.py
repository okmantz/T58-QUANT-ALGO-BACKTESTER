"""Strategy Library detail panel: saved results + report downloads (2026-10-28).

    GET /api/library/detail?type=&filename=          one JSON for the Backtest / Results / Validation / Optimize tabs
    GET /library/download/<kind>?type=&filename=     kind = full_pipeline | validation ; fmt = html|json|txt ; id= (optional)
    GET /library/download-all.zip?type=&filename=    every saved report for that strategy in one zip

Everything is read from files already on disk (the strategy's .meta.json sidecar, its consolidated report folder and
the Validate run reports) -- opening a tab never re-runs anything. Flask-free helpers at the top are unit-testable.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from flask import Blueprint, abort, jsonify, request, send_file

from app.data.storage import get_app_base_dir

library_detail_bp = Blueprint("library_detail", __name__)

VALIDATION_LABELS = {"wfo": "Walk-forward", "cpcv": "CPCV / PBO", "sensitivity": "Sensitivity",
                     "wfga": "Walk-forward GA", "regime_matrix": "Regime matrix"}


def _full_pipeline_dir() -> Path:
    return get_app_base_dir() / "reports" / "full_pipeline"


def _validation_dir() -> Path:
    return get_app_base_dir() / "reports" / "validation_runs"


# --------------------------------------------------------------------------- helpers (pure)
def _meta(stype: str, fname: str) -> dict:
    from app.strategy.library import load_strategy_metadata
    try:
        return load_strategy_metadata(stype, fname) or {}
    except Exception:  # noqa: BLE001
        return {}


def _folder_entries(stype: str, fname: str, name: str) -> list[dict]:
    try:
        from app.reports import strategy_folder
        return strategy_folder.list_reports(name or Path(fname).stem, "", library_type=stype, library_filename=fname)
    except Exception:  # noqa: BLE001
        return []


def latest_report_entry(entries: list[dict], tools: tuple[str, ...]) -> dict | None:
    rows = [e for e in entries if e.get("tool") in tools and (e.get("files") or {})]
    return max(rows, key=lambda e: float(e.get("recorded_at", 0))) if rows else None


def validation_reports(stype: str, fname: str, name: str, limit: int = 20) -> list[dict]:
    """Saved Validate-run reports belonging to this library strategy, newest first."""
    from app.validation import report_export
    out = []
    for r in report_export.list_reports(_validation_dir(), limit=400):
        same = (r.get("library_type") == stype and r.get("library_file") == fname) or (
            not r.get("library_file") and name and r.get("strategy") in (name, Path(fname).stem))
        if same:
            out.append(r)
        if len(out) >= limit:
            break
    return out


def _fmt_time(ts) -> str:
    try:
        return time.strftime("%b %d, %Y %H:%M", time.localtime(float(ts)))
    except Exception:  # noqa: BLE001
        return ""


def build_optimize_rundown(stype: str, fname: str, meta: dict, snap: dict | None, val_checks: list[dict]) -> dict:
    """Plain-language 'what to optimize' list from what was actually measured, plus the optimizer link that carries it."""
    from app.optimize.time_filter import suggest_blocked_slots
    focus: list[dict] = []
    st = (snap or {}).get("stats") or {}
    verdict = (snap or {}).get("verdict")
    slots = {"text": "", "items": [], "recovered_pnl": 0.0, "caveat": ""}
    if snap and snap.get("heatmap"):
        slots = suggest_blocked_slots(snap["heatmap"], snap.get("heatmap_counts"))
    already = (meta.get("blocked_entry_slots") or "").strip()
    if slots["text"] and already:  # do not re-suggest what is already blocked
        from app.backtest.time_slots import parse_slots
        have = set(parse_slots(already))
        slots["items"] = [i for i in slots["items"] if tuple(parse_slots(i["slot"]))[0] not in have]
        slots["text"] = "\n".join(i["slot"] for i in slots["items"])

    if not snap:
        focus.append({"goal": "eval", "title": "Run the Full Pipeline first",
                      "why": "There is no saved test of this strategy yet, so there is nothing to diagnose. Run it once and this tab will say exactly what to fix."})
    else:
        pep = snap.get("eval_pass_probability")
        ror = snap.get("risk_of_ruin_pct")
        dd = st.get("max_drawdown_pct")
        if slots["items"]:
            worst = slots["items"][0]
            focus.append({"goal": "timefilter", "title": "Stop trading the worst times",
                          "why": f"{worst['why']} Blocking {len(slots['items'])} slot(s) would have saved about ${slots['recovered_pnl']:,.0f} (optimistic: chosen on the same trades).",
                          "slots": slots["text"]})
        if (ror is not None and ror >= 5) or (dd is not None and dd >= 10):
            bits = []
            if ror is not None and ror >= 5:
                bits.append(f"risk of ruin {ror:.1f}%")
            if dd is not None and dd >= 10:
                bits.append(f"max drawdown {dd:.1f}%")
            focus.append({"goal": "risk", "title": "Right-size the risk per trade",
                          "why": f"The ride is too rough ({', '.join(bits)}). Smaller or smarter sizing can fix this without touching the entries."})
        if pep is not None and pep < 70 and verdict != "READY":
            focus.append({"goal": "eval", "title": "Raise the evaluation pass chance",
                          "why": f"Eval pass chance is {pep:.0f}% (the READY bar is 70%). Tuning the settings is the main lever."})
        wr, pf = st.get("win_rate"), st.get("profit_factor")
        if wr is not None and pf is not None and wr >= 55 and pf < 1.3:
            focus.append({"goal": "exits", "title": "Fix the exits",
                          "why": f"It wins often ({wr:.0f}%) but earns little per dollar risked (profit factor {pf:.2f}): winners are being given back."})
        if verdict == "READY" and not focus:
            focus.append({"goal": "payout", "title": "Get to the payout faster",
                          "why": "Already READY. The only upside left is reaching the first payout sooner."})
    if any(c.get("status") == "fail" for c in val_checks):
        failed = ", ".join(c["label"] for c in val_checks if c.get("status") == "fail")
        focus.append({"goal": "robust", "title": "Make the edge more reliable",
                      "why": f"Failed validation: {failed}. This goal scores on a conservative edge estimate so lucky settings lose to steady ones."})
    if not focus:
        focus.append({"goal": "robust", "title": "Nothing obvious is wrong",
                      "why": "The saved results show no clear weakness. Optimizing further risks curve-fitting; validate instead."})

    top = focus[0]
    q: dict[str, Any] = {"strategy_pick": f"{stype}::{fname}", "goal": top["goal"], "why": top["title"] + ": " + top["why"][:160]}
    lv = meta.get("last_validation") or {}
    if lv.get("dataset"):
        q["existing_dataset"] = lv["dataset"]
    if lv.get("timeframe") and lv["timeframe"] != "native":
        q["timeframe"] = lv["timeframe"]
    if lv.get("prop_preset"):
        q["prop_preset"] = lv["prop_preset"]
    if top.get("slots"):
        q["blocked_slots"] = top["slots"]
    lo = meta.get("last_optimize") or {}
    return {"focus": focus, "slots": slots, "active_slots": already,
            "last_optimize": lo or None, "href": "/optimize-simple?" + urlencode(q)}


def build_detail(stype: str, fname: str) -> dict:
    from app.strategy.library import VALIDATION_STEP_METHODS  # noqa: F401  (documented vocabulary)
    meta = _meta(stype, fname)
    name = str(meta.get("name") or Path(fname).stem)
    snap = meta.get("pipeline_snapshot") or None
    entries = _folder_entries(stype, fname, name)
    q = urlencode({"type": stype, "filename": fname})

    # ---------------- backtest tab
    backtest: dict[str, Any] = {"has": False}
    last_run = meta.get("last_run") or {}
    if snap:
        backtest = {"has": True, "source": "full_pipeline", "recorded_at": snap.get("recorded_at"),
                    "recorded_display": _fmt_time(snap.get("recorded_at")), "instrument": snap.get("instrument"),
                    "period": snap.get("period"), "equity": snap.get("equity") or [], "stats": snap.get("stats") or {},
                    "verdict": snap.get("verdict"), "reasons": snap.get("reasons") or [],
                    "t58_score": snap.get("t58_score"), "t58_tier": snap.get("t58_tier"),
                    "eval_pass_probability": snap.get("eval_pass_probability"),
                    "first_payout_probability": snap.get("first_payout_probability"),
                    "risk_of_ruin_pct": snap.get("risk_of_ruin_pct"),
                    "heatmap": snap.get("heatmap"), "heatmap_counts": snap.get("heatmap_counts")}
    elif last_run:
        backtest = {"has": True, "source": "last_run", "recorded_display": "", "equity": [],
                    "stats": {"net_profit": last_run.get("net_profit"), "total_trades": last_run.get("trades"),
                              "win_rate": last_run.get("win_rate"), "max_drawdown_pct": last_run.get("max_dd")},
                    "verdict": last_run.get("verdict"), "eval_pass_probability": last_run.get("eval_pass_probability"),
                    "first_payout_probability": last_run.get("first_payout_probability"),
                    "risk_of_ruin_pct": last_run.get("risk_of_ruin_pct"), "t58_score": last_run.get("t58_score"),
                    "t58_tier": last_run.get("t58_tier"), "reasons": [], "note": "Older run: no equity curve was saved. Re-run Full Pipeline to see it."}
    else:  # run_history fallback: match the equity curve by the strategy's display name / stem
        try:
            from app.reports.run_history import load_runs
            stem = Path(fname).stem.lower()
            rows = [r for r in load_runs() if str(r.get("strategy_name", "")).lower().split(" -- ")[0].strip() in (stem, name.lower())]
            if rows:
                r = rows[-1]
                backtest = {"has": True, "source": "run_history", "recorded_display": str(r.get("timestamp", ""))[:16].replace("T", " "),
                            "instrument": r.get("instrument"), "equity": r.get("equity_curve") or [], "heatmap": r.get("heatmap"),
                            "stats": {"net_profit": r.get("net_profit"), "total_trades": r.get("trades"), "win_rate": r.get("win_rate"),
                                      "profit_factor": r.get("profit_factor"), "sharpe_ratio": r.get("sharpe_ratio"),
                                      "max_drawdown_pct": r.get("max_drawdown_pct")},
                            "eval_pass_probability": r.get("eval_pass_probability"), "risk_of_ruin_pct": r.get("risk_of_ruin_pct"),
                            "reasons": [], "verdict": None}
        except Exception:  # noqa: BLE001
            pass

    # ---------------- validation tab
    methods_run = meta.get("validation_methods_run") or {}
    methods = []
    for key, label in VALIDATION_LABELS.items():
        rec = methods_run.get(key) or {}
        methods.append({"key": key, "label": label, "ran": bool(rec),
                        "status": rec.get("status") or ("pass" if rec.get("is_robust") else None),
                        "headline": rec.get("headline") or "", "recorded_display": _fmt_time(rec.get("recorded_at"))})
    vreps = validation_reports(stype, fname, name)
    latest_val: dict[str, Any] | None = None
    checks: list[dict] = []
    if vreps:
        from app.validation import report_export
        doc = report_export.load_report(_validation_dir(), vreps[0]["report_id"]) or {}
        checks = [{"check": c.get("check"), "label": c.get("label"), "status": c.get("status"), "verdict": c.get("verdict"),
                   "headline": c.get("headline"), "metrics": (c.get("metrics") or [])[:4]} for c in doc.get("checks", [])]
        latest_val = {"id": vreps[0]["report_id"], "overall": vreps[0].get("overall"), "summary": vreps[0].get("summary"),
                      "generated_at": vreps[0].get("generated_at"), "dataset": vreps[0].get("dataset"),
                      "next_step": (doc.get("next_step") or {})}
    # a check recorded from an older run (no saved report) still shows its status
    status_checks = checks or [{"label": m["label"], "status": m["status"], "headline": m["headline"], "verdict": (m["status"] or "").upper()}
                               for m in methods if m["ran"]]
    validation = {"has": bool(vreps or any(m["ran"] for m in methods)), "methods": methods, "ran_count": sum(1 for m in methods if m["ran"]),
                  "total": len(methods), "latest": latest_val, "checks": status_checks, "reports": vreps[:8]}

    # ---------------- results tab
    champ = meta.get("last_champion_check") or {}
    results = {"has": bool(snap or champ or last_run or meta.get("last_search") or meta.get("lookahead")),
               "verdict": champ.get("verdict") or (snap or {}).get("verdict"), "t58_score": champ.get("t58_score") or (snap or {}).get("t58_score"),
               "t58_tier": champ.get("t58_tier") or (snap or {}).get("t58_tier"), "reasons": champ.get("reasons") or (snap or {}).get("reasons") or [],
               "lookahead": meta.get("lookahead") or None, "last_search": meta.get("last_search") or None,
               "last_optimize": meta.get("last_optimize") or None, "status": meta.get("status")}

    optimize = build_optimize_rundown(stype, fname, meta, snap, status_checks)

    # ---------------- downloads
    fp = latest_report_entry(entries, ("full_pipeline",))
    fp_files: dict[str, str] = {}
    if fp:
        fp_files = {k: f"/library/download/full_pipeline?{q}&fmt={k}" for k in (fp.get("files") or {}) if k in ("html", "json")}
    elif snap and (snap.get("reports") or {}):
        fp_files = {k: f"/library/download/full_pipeline?{q}&fmt={k}" for k in ("html", "json") if (snap["reports"] or {}).get(k)}
    val_files = {}
    if vreps:
        rid = vreps[0]["report_id"]
        val_files = {f: f"/library/download/validation?{q}&fmt={f}&id={rid}" for f in ("html", "json", "txt")}
    downloads = {"full_pipeline": fp_files, "validation": val_files,
                 "all_zip": f"/library/download-all.zip?{q}" if (fp_files or val_files or entries) else None}
    return {"ok": True, "type": stype, "filename": fname, "name": name, "backtest": backtest, "results": results,
            "validation": validation, "optimize": optimize, "downloads": downloads,
            "full_pipeline_href": "/full-pipeline?" + urlencode({"strategy_pick": f"{stype}::{fname}"}),
            "validate_href": "/validate-simple?" + urlencode({"strategy_pick": f"{stype}::{fname}"})}


def _resolve_pipeline_file(stype: str, fname: str, fmt: str) -> Path | None:
    meta = _meta(stype, fname)
    name = str(meta.get("name") or Path(fname).stem)
    entry = latest_report_entry(_folder_entries(stype, fname, name), ("full_pipeline",))
    if entry and fmt in (entry.get("files") or {}):
        from app.reports import strategy_folder
        slug = strategy_folder.strategy_slug(name, "", library_type=stype, library_filename=fname)
        p = strategy_folder.reports_root() / slug / entry["files"][fmt]
        if p.is_file():
            return p
    rep = ((meta.get("pipeline_snapshot") or {}).get("reports") or {}).get(fmt)
    if rep:
        p = _full_pipeline_dir() / Path(rep).name
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------- routes
def _ident() -> tuple[str, str]:
    stype = (request.args.get("type") or "").strip()
    fname = (request.args.get("filename") or "").strip()
    if not stype or not fname or ".." in fname or "/" in fname or "\\" in fname:
        abort(400)
    return stype, fname


@library_detail_bp.route("/api/library/detail")
def library_detail():
    stype, fname = _ident()
    try:
        return jsonify(build_detail(stype, fname))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500


@library_detail_bp.route("/library/download/<kind>")
def library_download(kind):
    stype, fname = _ident()
    fmt = (request.args.get("fmt") or "html").lower()
    if kind == "full_pipeline":
        if fmt not in ("html", "json"):
            abort(400)
        p = _resolve_pipeline_file(stype, fname, fmt)
        if p is None:
            abort(404, "No saved Full Pipeline report for this strategy yet -- run it once from the Backtest tab.")
    elif kind == "validation":
        if fmt not in ("html", "json", "txt"):
            abort(400)
        from app.validation import report_export
        meta = _meta(stype, fname)
        reps = validation_reports(stype, fname, str(meta.get("name") or Path(fname).stem), limit=100)
        rid = (request.args.get("id") or "").strip()
        if rid and rid not in {r["report_id"] for r in reps}:
            abort(404)
        rid = rid or (reps[0]["report_id"] if reps else "")
        p = report_export.report_path(_validation_dir(), rid, fmt) if rid else None
        if p is None:
            abort(404, "No saved validation report for this strategy yet -- run Validate once from the Validation tab.")
    else:
        abort(404)
    stem = Path(fname).stem
    return send_file(p, as_attachment=True, download_name=f"{stem}__{kind}__{p.name}" if kind == "validation" else f"{stem}__full_pipeline.{fmt}")


@library_detail_bp.route("/library/download-all.zip")
def library_download_all():
    stype, fname = _ident()
    meta = _meta(stype, fname)
    name = str(meta.get("name") or Path(fname).stem)
    stem = Path(fname).stem
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        try:
            from app.reports import strategy_folder
            slug = strategy_folder.strategy_slug(name, "", library_type=stype, library_filename=fname)
            folder = strategy_folder.reports_root() / slug
            if folder.is_dir():
                for p in sorted(folder.iterdir()):
                    if p.is_file():
                        z.write(p, f"{stem}/tests/{p.name}")
                        n += 1
        except Exception:  # noqa: BLE001
            pass
        from app.validation import report_export
        for r in validation_reports(stype, fname, name, limit=50):
            for f in ("html", "json", "txt"):
                p = report_export.report_path(_validation_dir(), r["report_id"], f)
                if p:
                    z.write(p, f"{stem}/validation/{p.name}")
                    n += 1
        for f in ("html", "json"):
            p = _resolve_pipeline_file(stype, fname, f)
            if p and not any(i.filename.endswith(p.name) for i in z.infolist()):
                z.write(p, f"{stem}/full_pipeline/{p.name}")
                n += 1
        if n == 0:
            z.writestr(f"{stem}/README.txt", "No saved reports for this strategy yet. Run the Full Pipeline and Validate, then download again.\n")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{stem}__all_reports.zip")
