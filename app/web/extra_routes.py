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
import math
import tempfile
import threading
from pathlib import Path

from flask import Blueprint, jsonify, redirect, render_template, request, send_file, url_for

from app.backtest.risk import RiskConfig, with_prop_safety_defaults
from app.data.importer import import_csv
from app.data.storage import get_raw_data_dir, list_datasets_by_instrument, list_stored_datasets
from app.live_deploy.broker_registry import SUPPORTED_PLATFORMS
from app.live_deploy.prop_firms import PROP_FIRMS
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import FITNESS_METRICS
from app.orchestration.resource_guard import HEAVY_JOB_GUARD
from app.prop.simulator import PropRules
from app.prop.presets import list_presets as list_prop_firm_presets
from app.strategy.compare import CompareError, compare_strategies, compare_to_dicts
from app.optimize.multi_market import AGGREGATION_METHODS
from app.search.cross_instrument import CrossInstrumentCancelled, run_cross_instrument_search
from app.search.strategy_space import (
    FAMILIES_REQUIRING_CALENDAR_DATA, FAMILIES_REQUIRING_PAIR_DATA, family_description, list_families,
)
from app.strategy.library import (
    StrategyAlreadyExists, list_saved_strategies, safe_filename_stem, save_strategy_metadata,
    save_strategy_text, set_strategy_status,
)
from app.web.alpaca_shared import alpaca_template_context
from app.web.job_manager import JOB_MANAGER

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


def _render_compare(strategies, stored_datasets, result=None, error=None, alpaca_notice=None, alpaca_notice_kind="info"):
    return render_template(
        "compare.html", active_page="compare",
        strategies=[{"type": s.strategy_type, "name": s.name} for s in strategies],
        stored_datasets=stored_datasets, result=result, error=error,
        prop_presets_json=_prop_presets_json(),
        alpaca_notice=alpaca_notice, alpaca_notice_kind=alpaca_notice_kind,
        **alpaca_template_context(),
    )


@extra_bp.route("/compare", methods=["GET"])
def compare_page():
    return _render_compare(
        list_saved_strategies(), list_stored_datasets(),
        alpaca_notice=request.args.get("alpaca_notice"),
        alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"),
    )


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
        commission_per_trade=float(form.get("commission", 0) or 0),
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



# ---------------------------------------------------------------------------
# Cross-Instrument Search -- pick several instruments, find the ONE strategy
# that works across all of them. See app.search.cross_instrument's module
# docstring for how this differs from Multi-Market (tunes one strategy you
# already have) and from the multi-instrument Search/Evolution Lab modes
# (independent per-instrument searches).
# ---------------------------------------------------------------------------

JOB_CROSS_INSTRUMENT = "Cross-Instrument Search"


def _finite(value, digits: int = 4):
    """JSON-safe number: NaN / +-inf become None (browsers reject the
    `Infinity` token jsonify would otherwise emit for a dead candidate)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _cross_instrument_form_context(**extra) -> dict:
    hidden = FAMILIES_REQUIRING_PAIR_DATA | FAMILIES_REQUIRING_CALENDAR_DATA
    families = [
        {"name": name, "description": family_description(name)}
        for name in list_families() if name not in hidden
    ]
    ctx = dict(
        active_page="cross_instrument",
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        families=families, fitness_metrics=FITNESS_METRICS, aggregation_methods=AGGREGATION_METHODS,
    )
    ctx.update(extra)
    return ctx


@extra_bp.route("/cross-instrument", methods=["GET"])
def cross_instrument_form():
    return render_template("cross_instrument.html", **_cross_instrument_form_context())


def _cross_instrument_error(message: str, status: int = 400):
    return render_template("cross_instrument.html", **_cross_instrument_form_context(error=message)), status


def _run_cross_instrument_job(job_id: str, kwargs: dict) -> None:
    def cancelled() -> bool:
        job = JOB_MANAGER.get(job_id)
        return bool(job and job.get("cancelled"))

    try:
        result = run_cross_instrument_search(
            **kwargs, progress_cb=lambda msg: JOB_MANAGER.log(job_id, msg), cancel_check=cancelled,
        )
        JOB_MANAGER.finish(job_id, result=result)
    except CrossInstrumentCancelled:
        JOB_MANAGER.fail(job_id, "Cancelled.")
    except RefinementError as exc:
        JOB_MANAGER.fail(job_id, str(exc))
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not die silently in the thread
        JOB_MANAGER.fail(job_id, f"Unexpected error: {exc}")
    finally:
        HEAVY_JOB_GUARD.release(JOB_CROSS_INSTRUMENT)


@extra_bp.route("/cross-instrument/start", methods=["POST"])
def cross_instrument_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_CROSS_INSTRUMENT):
        return _cross_instrument_error(
            f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Wait for it to finish first.", 409,
        )
    started = False
    try:
        selected = form.getlist("datasets")
        if len(selected) < 2:
            return _cross_instrument_error("Select at least 2 instruments to search across.")

        dfs, load_warnings = {}, []
        for name in selected:
            path = get_raw_data_dir() / name
            if not path.exists():
                load_warnings.append(f"{name}: file not found -- skipped.")
                continue
            imported = import_csv(path)
            if not imported.is_valid:
                load_warnings.append(f"{name}: could not be read as market data -- skipped.")
                continue
            dfs[Path(name).stem] = imported.dataframe
        if len(dfs) < 2:
            return _cross_instrument_error(
                "Fewer than 2 of the selected datasets could be loaded: " + "; ".join(load_warnings)
            )

        reset_on_breach = form.get("reset_on_breach", "on") == "on"
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent") or "percent",
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            max_trades_per_day=int(form.get("max_trades_day", 10) or 10),
            commission_per_trade=float(form.get("commission", 0) or 0),
            slippage_pips=float(form.get("slippage_pips", 0.5) or 0.5),
            spread_pips=float(form.get("spread_pips", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
            contract_size=(float(form.get("contract_size")) if form.get("contract_size") else None),
            reset_on_breach=reset_on_breach,
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        )
        risk = with_prop_safety_defaults(risk, rules)
        holdout_pct = max(0.0, min(40.0, float(form.get("holdout_pct", 20) or 0)))
        kwargs = dict(
            dfs=dfs, risk=risk, prop_rules=rules,
            mc_config=MonteCarloConfig(
                n_simulations=max(50, int(form.get("n_sims", 1000) or 1000)), reset_on_breach=reset_on_breach,
            ),
            families=form.getlist("families") or None,
            fitness_metric=form.get("fitness_metric", "eval_pass_probability") or "eval_pass_probability",
            aggregation=form.get("aggregation", "mean_minus_dispersion") or "mean_minus_dispersion",
            max_candidates=max(2, min(400, int(form.get("max_candidates", 60) or 60))),
            screen_mc_sims=max(30, int(form.get("screen_sims", 150) or 150)),
            finalists=max(1, min(15, int(form.get("finalists", 5) or 5))),
            holdout_frac=holdout_pct / 100.0,
            refine_winner=form.get("refine_winner") == "on",
            seed=int(form.get("random_seed", 42) or 42),
        )

        job_id = JOB_MANAGER.create(
            log=[f"Loaded {len(dfs)} instrument(s): {', '.join(dfs)}."] + load_warnings,
            markets=list(dfs), aggregation=kwargs["aggregation"], tool=JOB_CROSS_INSTRUMENT,
            cancelled=False,
        )
        JOB_MANAGER.prune(max_age_seconds=6 * 3600)
        threading.Thread(target=_run_cross_instrument_job, args=(job_id, kwargs), daemon=True).start()
        started = True
        return redirect(url_for("extra.cross_instrument_job", job_id=job_id))
    except (RefinementError, ValueError) as exc:
        return _cross_instrument_error(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _cross_instrument_error(f"Unexpected error: {exc}", 500)
    finally:
        if not started:
            HEAVY_JOB_GUARD.release(JOB_CROSS_INSTRUMENT)


@extra_bp.route("/cross-instrument/job/<job_id>")
def cross_instrument_job(job_id):
    job = JOB_MANAGER.get(job_id)
    return render_template("cross_instrument_job.html", job_id=job_id, not_found=job is None)


def _scores_json(scores) -> list:
    return [
        {"market": s.market, "fitness": _finite(s.fitness, 2), "trade_count": s.trade_count,
         "evaluated": getattr(s, "evaluated", True)}
        for s in (scores or [])
    ]


@extra_bp.route("/cross-instrument/job/<job_id>/status.json")
def cross_instrument_job_status(job_id):
    job = JOB_MANAGER.get(job_id)
    if job is None:
        return jsonify({"not_found": True}), 404
    payload = {
        "done": job["done"], "error": job.get("error"), "log": job["log"][-200:],
        "markets": job.get("markets", []),
    }
    result = job.get("result")
    if result is not None:
        payload["result"] = {
            "markets": result.markets, "aggregation": result.aggregation,
            "fitness_metric": result.fitness_metric, "families": result.families,
            "candidates_generated": result.candidates_generated,
            "candidates_viable": result.candidates_viable,
            "total_backtests": result.total_backtests, "holdout_frac": result.holdout_frac,
            "elapsed_seconds": round(result.elapsed_seconds, 1), "warnings": result.warnings,
            "leaderboard": [
                {
                    "rank": i + 1, "family": c.family, "candidate_id": c.candidate_id, "refined": c.refined,
                    "robustness_score": _finite(c.robustness_score), "mean_fitness": _finite(c.mean_fitness, 2),
                    "worst_case_fitness": _finite(c.worst_case_fitness, 2), "dispersion": _finite(c.dispersion, 2),
                    "per_market": _scores_json(c.per_market),
                    "holdout_robustness": _finite(c.holdout_robustness),
                    "holdout_per_market": _scores_json(c.holdout_per_market),
                    "holdout_verdict": c.holdout_verdict,
                    "params": c.params,
                }
                for i, c in enumerate(result.leaderboard)
            ],
        }
    return jsonify(payload)


@extra_bp.route("/cross-instrument/job/<job_id>/cancel", methods=["POST"])
def cross_instrument_job_cancel(job_id):
    if JOB_MANAGER.get(job_id) is None:
        return jsonify({"not_found": True}), 404
    JOB_MANAGER.update(job_id, cancelled=True)
    return jsonify({"ok": True})


@extra_bp.route("/cross-instrument/job/<job_id>/save/<int:rank>", methods=["POST"])
def cross_instrument_job_save(job_id, rank):
    """Saves leaderboard entry #`rank` (1-based) to the Strategy Library as a
    Manual-Builder JSON tagged 'cross-instrument', status 'draft'."""
    job = JOB_MANAGER.get(job_id)
    result = job.get("result") if job else None
    if result is None or not (1 <= rank <= len(result.leaderboard)):
        return jsonify({"error": "That result is no longer available (the server may have restarted)."}), 404
    cand = result.leaderboard[rank - 1]
    stem = safe_filename_stem(f"crossinst_{job_id[:8]}_{cand.family}_{rank}", fallback="crossinst")
    filename = f"{stem}.json"
    try:
        save_strategy_text(json.dumps(cand.config, indent=2), filename, "manual", overwrite=False)
    except StrategyAlreadyExists:
        return jsonify({"error": f"{filename} is already in the Strategy Library."}), 409
    set_strategy_status("manual", filename, "draft")
    save_strategy_metadata(
        "manual", filename,
        {
            "tags": ["cross-instrument"],
            "description": (
                f"Cross-Instrument Search {job_id[:8]}: family '{cand.family}', found across "
                f"{', '.join(result.markets)} ({result.aggregation})."
            ),
        },
        merge=True,
    )
    return jsonify({"ok": True, "filename": filename})
