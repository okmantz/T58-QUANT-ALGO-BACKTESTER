"""
Routes added in the 2026-10-17 release: Gamma Center, Trading Journal, clickable-metric definitions, the library
control-center feed and the prop-firm simulator presets. One Blueprint (`v9_bp`), registered in app/web/server.py.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.data import gamma_data as gd
from app.scoring import metric_defs
from app.web import journal_store

v9_bp = Blueprint("v9", __name__)


def _err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


# ----------------------------------------------------------------- Gamma Center
@v9_bp.route("/gamma")
def gamma_page():
    return render_template("gamma.html", active_page="gamma", instruments=gd.INSTRUMENT_MAP)


@v9_bp.route("/api/gamma/status")
def gamma_status():
    return jsonify({"ok": True, **gd.gamma_status(), "instruments_meta": gd.INSTRUMENT_MAP})


@v9_bp.route("/api/gamma/live/<instrument>")
def gamma_live(instrument: str):
    try:
        dte_raw = request.args.get("dte", "7")
        max_dte = None if dte_raw.lower() in ("all", "none", "") else max(0, min(int(dte_raw), 365))
        res = gd.live_gex(instrument, max_dte=max_dte)
    except gd.GammaDataError as exc:
        return _err(str(exc), 502)
    except ValueError:
        return _err("dte must be a whole number or 'all'")
    return jsonify({"ok": True, "gex": res.to_dict()})


@v9_bp.route("/api/gamma/snapshot/<instrument>", methods=["POST"])
def gamma_snapshot(instrument: str):
    try:
        res = gd.live_gex(instrument, max_dte=7)
        path = gd.record_snapshot(res)
    except gd.GammaDataError as exc:
        return _err(str(exc), 502)
    return jsonify({"ok": True, "saved": str(path.name), "levels": {
        "spot": res.spot, "gamma_flip": res.gamma_flip, "call_wall": res.call_wall, "put_wall": res.put_wall,
        "net_gex": res.net_gex}})


@v9_bp.route("/api/gamma/import/<instrument>", methods=["POST"])
def gamma_import(instrument: str):
    try:
        text = ""
        if request.files.get("file"):
            text = request.files["file"].read().decode("utf-8", "replace")
        else:
            text = (request.get_json(silent=True) or {}).get("csv") or request.form.get("csv", "")
        n = gd.import_levels_csv(text, instrument, source=(request.form.get("source") or "imported")[:40])
    except gd.GammaDataError as exc:
        return _err(str(exc))
    return jsonify({"ok": True, "imported": n})


@v9_bp.route("/api/gamma/history/refresh", methods=["POST"])
def gamma_history_refresh():
    try:
        df = gd.fetch_squeezemetrics_history()
    except gd.GammaDataError as exc:
        return _err(str(exc), 502)
    return jsonify({"ok": True, "rows": int(len(df)), "first": str(df["date"].iloc[0]), "last": str(df["date"].iloc[-1])})


# ------------------------------------------------------------------ Journal
@v9_bp.route("/journal")
def journal_page():
    return render_template("journal.html", active_page="journal")


@v9_bp.route("/api/journal")
def journal_get():
    data = journal_store.load()
    return jsonify({"ok": True, **data, "stats": journal_store.journal_stats(data["trades"], data["settings"])})


@v9_bp.route("/api/journal/settings", methods=["PUT", "POST"])
def journal_settings():
    try:
        cur = journal_store.save_settings(request.get_json(force=True, silent=True) or {})
    except ValueError as exc:
        return _err(str(exc))
    return jsonify({"ok": True, "settings": cur})


@v9_bp.route("/api/journal/trades", methods=["POST"])
def journal_trade_save():
    try:
        t = journal_store.upsert_trade(request.get_json(force=True, silent=True) or {})
    except ValueError as exc:
        return _err(str(exc))
    return jsonify({"ok": True, "trade": t})


@v9_bp.route("/api/journal/trades/<trade_id>", methods=["DELETE"])
def journal_trade_delete(trade_id: str):
    return jsonify({"ok": journal_store.delete_trade(trade_id)})


@v9_bp.route("/api/journal/reset", methods=["POST"])
def journal_reset():
    """Restores the preloaded playbook + tags (keeps trades)."""
    data = journal_store.load()
    journal_store.save_settings(journal_store.default_settings())
    return jsonify({"ok": True, "trades_kept": len(data["trades"])})


# ----------------------------------------------------------- clickable metrics
@v9_bp.route("/api/metrics")
def metrics_index():
    return jsonify({"ok": True, "labels": metric_defs.all_labels()})


@v9_bp.route("/api/metrics/<key>")
def metrics_definition(key: str):
    d = metric_defs.definition(key)
    if not d:
        return _err(f"No definition for '{key}'.", 404)
    return jsonify({"ok": True, "metric": d})


@v9_bp.route("/api/metrics/<key>/history")
def metrics_history(key: str):
    if not metric_defs.resolve(key):
        return _err(f"No definition for '{key}'.", 404)
    return jsonify({"ok": True, **metric_defs.history_for(key)})


# ----------------------------------------------------- library control center
@v9_bp.route("/api/library/control")
def library_control():
    from app.strategy import library_control as lc

    rows = lc.control_rows()
    return jsonify({"ok": True, "rows": rows, "summary": lc.summarize(rows)})


# --------------------------------------------------------- prop simulator data
@v9_bp.route("/api/prop-sim/presets")
def prop_sim_presets():
    from app.prop.presets import list_presets

    out = []
    for p in list_presets():
        out.append({
            "key": p.key, "firm": p.firm, "label": p.label, "account_size": p.account_size,
            "target_pct": p.evaluation_profit_target_pct, "daily_loss_pct": p.daily_loss_limit_pct,
            "max_dd_pct": p.max_drawdown_pct, "drawdown_type": p.drawdown_type,
            "consistency_pct": p.consistency_rule_pct, "min_days": p.min_trading_days,
            "payout_threshold_pct": p.payout_threshold_pct,
            "funded_consistency_pct": getattr(p, "funded_consistency_rule_pct", None),
            "payout_frequency_days": p.payout_frequency_days,
        })
    return jsonify({"ok": True, "presets": out})
