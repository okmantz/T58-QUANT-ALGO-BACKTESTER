"""
Extra web routes for this changeset -- Strategy Compare, native PDF
export, and the Dukascopy forex/CFD fetch button.

Deploy Live's account credentials and live order-placement endpoints are
DELIBERATELY NOT here -- see the "Deploy Live" section further down in
this file and INTEGRATION.md for why, and where those belong instead
(the desktop app, not this web server).

Registered as a Flask Blueprint (`extra_bp`), same pattern as
app/web/quant_lab_routes.py, app/web/ai_assistant_routes.py, etc: this
whole feature area lives in one file, and app/web/server.py's own change
is a single import + one `app.register_blueprint(extra_bp)` line (see
INTEGRATION.md).
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, send_file

from app.backtest.risk import RiskConfig
from app.data.importer import import_csv
from app.data.storage import get_raw_data_dir, list_stored_datasets
from app.live_deploy.broker_registry import SUPPORTED_PLATFORMS
from app.live_deploy.prop_firms import PROP_FIRMS
from app.prop.simulator import PropRules
from app.prop.presets import list_presets as list_prop_firm_presets
from app.strategy.compare import CompareError, compare_strategies, compare_to_dicts
from app.strategy.library import list_saved_strategies

extra_bp = Blueprint("extra", __name__)


def _prop_presets_json() -> str:
    """Same shape as app.web.server's own private _prop_presets_json() --
    duplicated here (rather than imported) purely to avoid a circular
    import between this blueprint module and server.py; see this
    function's twin there for the exact field list."""
    return json.dumps([p.to_dict() for p in list_prop_firm_presets()])


# ---------------------------------------------------------------------------
# Strategy Compare
# ---------------------------------------------------------------------------

def _prop_rules_from_form(form) -> PropRules:
    """Same field names as the existing Prop-Firm Rules form fragment
    (see app/web/templates/_prop_rules_extra_fields.html + INTEGRATION.md)
    plus the four new optional fields."""
    return PropRules(
        account_size=float(form.get("account_size", 100000) or 100000),
        evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
        daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
        max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        drawdown_type=form.get("dd_type", "trailing"),
        drawdown_check_mode=form.get("dd_check_mode", "intrabar"),
        consistency_rule_pct=float(form["consistency"]) if form.get("consistency") else None,
        min_trading_days=int(form.get("min_days", 5) or 5),
        payout_frequency_days=int(form.get("payout_freq", 14) or 14),
        payout_threshold_pct=float(form.get("payout_threshold", 0) or 0),
        required_buffer_pct=float(form.get("buffer", 0) or 0),
        payout_cap_pct=float(form["payout_cap"]) if form.get("payout_cap") else None,
        news_blackout_windows=form.get("news_blackout_windows", "") or "",
        weekend_hold_allowed=form.get("weekend_hold_allowed", "on") == "on",
        max_lot_size=float(form["max_lot_size"]) if form.get("max_lot_size") else None,
        hedging_allowed=form.get("hedging_allowed", "on") == "on",
    )


def _render_compare(strategies, stored_datasets, result=None, error=None):
    return render_template(
        "compare.html", active_page="compare",
        strategies=[{"type": s.strategy_type, "name": s.name} for s in strategies],
        stored_datasets=stored_datasets, result=result, error=error,
        prop_presets_json=_prop_presets_json(),
    )


@extra_bp.route("/compare", methods=["GET"])
def compare_page():
    return _render_compare(list_saved_strategies(), list_stored_datasets())


@extra_bp.route("/compare/run", methods=["POST"])
def compare_run():
    form = request.form
    picks_raw = form.getlist("strategy_pick")  # each item "type::filename"
    candidates = []
    for p in picks_raw:
        if "::" in p:
            stype, fname = p.split("::", 1)
            candidates.append((stype, fname))

    dataset_name = (form.get("existing_dataset") or "").strip()
    strategies = list_saved_strategies()
    stored_datasets = list_stored_datasets()

    if len(candidates) < 2:
        return _render_compare(strategies, stored_datasets, error="Pick at least 2 saved strategies to compare.")
    if not dataset_name:
        return _render_compare(strategies, stored_datasets, error="Choose a dataset to backtest all candidates against.")

    candidate = get_raw_data_dir() / dataset_name
    import_result = import_csv(candidate)
    if not import_result.is_valid:
        return _render_compare(strategies, stored_datasets,
                                error=f"Could not load dataset: {'; '.join(import_result.errors)}")

    risk = RiskConfig(
        initial_balance=float(form.get("account_size", 100000) or 100000),
        risk_value=float(form.get("risk_value", 1.0) or 1.0),
        pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
    )
    prop_rules = _prop_rules_from_form(form)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            results = compare_strategies(candidates[:4], import_result.dataframe, risk, prop_rules, tmp_dir=tmp_dir)
        except CompareError as exc:
            return _render_compare(strategies, stored_datasets, error=str(exc))

    return _render_compare(strategies, stored_datasets, result=compare_to_dicts(results))


# ---------------------------------------------------------------------------
# Native PDF export
# ---------------------------------------------------------------------------

@extra_bp.route("/export/pdf", methods=["POST"])
def export_pdf_route():
    """Accepts the same report dict app.reports.generator.build_report
    produces, as a JSON string in the `report_json` form field (add a
    hidden <input name="report_json"> populated by the report page's
    own already-rendered data -- see INTEGRATION.md), and returns a
    downloadable native PDF instead of relying on the browser's
    print-to-PDF dialog."""
    from app.reports.pdf_export import PdfExportError, export_pdf

    raw = request.form.get("report_json") or (request.get_json(silent=True) or {}).get("report_json")
    if not raw:
        return jsonify({"error": "Missing report_json."}), 400
    try:
        report = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Invalid report_json: {exc}"}), 400

    strategy_name = (report.get("strategy", {}) or {}).get("name", "strategy")
    safe_name = "".join(c for c in strategy_name if c.isalnum() or c in ("-", "_")) or "strategy"

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / f"{safe_name}_report.pdf"
        try:
            export_pdf(report, out_path)
        except PdfExportError as exc:
            return jsonify({"error": str(exc)}), 500
        buf = io.BytesIO(out_path.read_bytes())
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=f"{safe_name}_report.pdf")


# ---------------------------------------------------------------------------
# Dukascopy forex/CFD fetch
# ---------------------------------------------------------------------------

@extra_bp.route("/data/dukascopy/fetch", methods=["POST"])
def data_dukascopy_fetch():
    from app.data.dukascopy_source import DukascopyFetchError, fetch_ohlcv, known_symbols, save_bars_as_csv

    form = request.form
    symbol = (form.get("dukascopy_symbol") or "").strip().upper()
    timeframe = form.get("dukascopy_timeframe") or "1Hour"
    start = (form.get("dukascopy_start") or "").strip()
    end = (form.get("dukascopy_end") or "").strip()

    if not symbol or not start or not end:
        return jsonify({"error": "Symbol, start, and end date are all required."}), 400
    try:
        df = fetch_ohlcv(symbol, timeframe, start, end)
        dest = save_bars_as_csv(df, symbol, timeframe)
    except DukascopyFetchError as exc:
        return jsonify({"error": str(exc), "known_symbols": known_symbols()}), 400
    return jsonify({"ok": True, "saved_as": dest.relative_to(get_raw_data_dir()).as_posix(), "rows": len(df)})


# ---------------------------------------------------------------------------
# Deploy Live -- NOT wired into this web server. Read this before adding it.
# ---------------------------------------------------------------------------
#
# app/web/templates/deploy_live.html already documents, deliberately, why this
# app's live-money features stay OUT of the web server: it has no login and
# listens on every network interface, so anyone on the same network (or
# further, if a port is ever forwarded) could reach it. Adding broker-account
# credential storage and real order-placement endpoints here -- which is
# exactly what a web "Deploy Live" control panel needs -- would directly
# undo that decision.
#
# The new broker adapters (app/live_deploy/broker_*.py) and
# LiveExecutionSession (app/live_deploy/execution_engine.py) are still fully
# usable -- just from the desktop app (app/ui/main_window.py), the same
# place MT5 Forward Test already lives, not from here. See
# INTEGRATION.md's "Deploy Live" section for exactly where in
# main_window.py to wire a start/stop/flatten control panel using them,
# following the same pattern the existing Forward Test tab already uses
# for MT5Connector + ForwardTestSession.
#
# The one thing kept here is read-only, credential-free, and safe on an
# open network: which platforms/firms are connectable at all.

@extra_bp.route("/deploy-live/platforms", methods=["GET"])
def deploy_live_platforms():
    return jsonify({
        "platforms": SUPPORTED_PLATFORMS,
        "firms": [{"name": f.name, "platforms": f.platforms, "asset_focus": f.asset_focus,
                   "connectable_today": f.connectable_today, "notes": f.notes} for f in PROP_FIRMS],
    })
