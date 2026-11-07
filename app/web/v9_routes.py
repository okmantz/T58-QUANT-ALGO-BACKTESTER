"""
Routes added in the 2026-10-17 release: Gamma Center, Trading Journal, clickable-metric definitions, the library
control-center feed and the prop-firm simulator presets. One Blueprint (`v9_bp`), registered in app/web/server.py.
"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, render_template, request, send_file

from app.data import gamma_data as gd
from app.scoring import metric_defs
from app.web import journal_store

v9_bp = Blueprint("v9", __name__)


def _err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


# ----------------------------------------------------------------- Gamma Center
@v9_bp.route("/gamma")
def gamma_page():
    return render_template("gamma.html", active_page="quant_lab", instruments=gd.INSTRUMENT_MAP)


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


@v9_bp.route("/api/journal/screenshots/<name>")
def journal_screenshot(name: str):
    """Serves one trade's stored screenshot. Names are validated against the exact pattern the store generates, so
    nothing outside the screenshots folder can ever be read through this route."""
    path = journal_store.screenshot_path(name)
    if path is None:
        abort(404)
    resp = send_file(path, conditional=True)
    resp.headers["Cache-Control"] = "private, max-age=86400"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


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
@v9_bp.route("/api/prop-sim/datasets")
def prop_sim_datasets():
    """Stored datasets the prop-sim can measure a strategy on (plus the auto-pick for each library row)."""
    from app.data.storage import list_datasets_by_instrument
    from app.prop.edge_probe import suggest_dataset
    from app.strategy.library import STRATEGY_TYPES, list_saved_strategies

    groups = [{"instrument": g.get("instrument"),
               "files": [{"full_name": f.get("full_name"), "name": f.get("name")} for f in g.get("files", []) if not f.get("empty")]}
              for g in list_datasets_by_instrument(count_rows=False)]
    suggested = {}
    for t in STRATEGY_TYPES:
        for s in list_saved_strategies(t):
            try:
                suggested[f"{t}::{s.name}"] = suggest_dataset(t, s.name, s.metadata)
            except Exception:  # noqa: BLE001
                continue
    return jsonify({"ok": True, "groups": groups, "suggested": suggested})


@v9_bp.route("/api/prop-sim/strategy-edge", methods=["POST"])
def prop_sim_strategy_edge():
    """Backtest ONE saved strategy (any language) and return the simulator's edge inputs. Runs the same
    run_backtest every other tool uses, so the numbers match what a normal run would report."""
    from app.prop.edge_probe import measure_edge

    body = request.get_json(silent=True) or request.form
    stype = (body.get("type") or "").strip()
    name = (body.get("name") or "").strip()
    if not stype or not name:
        return _err("Pick a saved strategy first.")
    try:
        return jsonify(measure_edge(stype, name, (body.get("dataset") or "").strip() or None,
                                    use_cache=str(body.get("refresh", "")).lower() not in ("1", "true", "yes")))
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001 -- surface a readable message instead of a 500 page
        return _err(f"Could not measure this strategy: {type(exc).__name__}: {exc}", 500)


@v9_bp.route("/api/prop-sim/presets")
def prop_sim_presets():
    from app.prop.presets import get_fee_info, list_presets

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
            "dd_basis": getattr(p, "dd_basis", "legacy"), "trailing_lock": getattr(p, "trailing_lock", False),
            "daily_loss_action": getattr(p, "daily_loss_action", "fail"),
            "max_eval_days": getattr(p, "max_eval_calendar_days", None),
            "algo_allowed": getattr(p, "algo_trading_allowed", False),
            "fee": (lambda f: None if f is None else {"price": f.challenge_price, "reset": f.reset_price, "split_pct": f.profit_split_pct,
                                                      "activation": f.activation_fee, "monthly": f.monthly_subscription})(get_fee_info(p.key)),
        })
    return jsonify({"ok": True, "presets": out})


# ----------------------------------------------------------- payout forecast (2026-11-07)
@v9_bp.route("/api/payout-forecast/run", methods=["POST"])
def payout_forecast_run():
    """Will this saved strategy pass an evaluation AND get paid, at which firm, and is the fee worth it?

    Backtests the strategy on a stored dataset, then forecasts every algo-allowed firm preset from its real trades (Monte Carlo + real
    start-date replay), ranked by conservative expected payout per month. ``holdout_frac`` (0-0.5, default 0) forecasts only on the LAST
    share of the trades -- use it when the strategy was NOT tuned on that period; the result is then graded out-of-sample."""
    import pandas as pd
    from app.prop.edge_probe import backtest_saved_strategy
    from app.validation.payout_forecast import forecast_across_firms, summarize

    body = request.get_json(silent=True) or {}
    stype, name = (body.get("type") or "").strip(), (body.get("name") or "").strip()
    if not stype or not name:
        return _err("Pick a saved strategy first.")
    try:
        trades, info = backtest_saved_strategy(stype, name, (body.get("dataset") or "").strip() or None)
        frac = min(max(float(body.get("holdout_frac") or 0.0), 0.0), 0.5)
        oos = False
        if frac > 0:
            cut = pd.Timestamp(trades[0].entry_time) + (pd.Timestamp(trades[-1].entry_time) - pd.Timestamp(trades[0].entry_time)) * (1 - frac)
            later = [t for t in trades if pd.Timestamp(t.entry_time) > cut]
            if len(later) >= 30:
                trades, oos = later, True
            else:
                info["note"] = f"Only {len(later)} trades in the last {frac * 100:.0f}% of the history; forecast uses all trades (in-sample)."
        fs = forecast_across_firms(trades, source_account_size=info["source_account_size"], sims=int(body.get("sims") or 1000),
                                   max_firms=int(body.get("max_firms") or 12), out_of_sample=oos, algo_only=bool(body.get("algo_only", True)))
        return jsonify({"ok": True, "info": info, "summary": summarize(fs), "firms": [f.to_dict() for f in fs]})
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not build the forecast: {type(exc).__name__}: {exc}", 500)


@v9_bp.route("/api/payout-forecast/outcome", methods=["POST"])
def payout_forecast_outcome():
    """Record what really happened to a forecast (passed / paid) so future forecasts are calibrated against reality."""
    from app.validation.payout_forecast import calibration_summary, record_outcome

    b = request.get_json(silent=True) or {}
    try:
        rec = record_outcome(str(b.get("strategy") or ""), str(b.get("preset_key") or ""), float(b.get("forecast_p_pass")),
                             passed=b.get("passed"), paid=b.get("paid"),
                             forecast_p_paid=None if b.get("forecast_p_paid") is None else float(b.get("forecast_p_paid")),
                             note=str(b.get("note") or ""))
    except (TypeError, ValueError):
        return _err("forecast_p_pass is required (a fraction between 0 and 1).")
    return jsonify({"ok": True, "record": rec, "calibration": calibration_summary()})


@v9_bp.route("/api/payout-forecast/calibration")
def payout_forecast_calibration():
    from app.validation.payout_forecast import calibration_summary, read_ledger, shrink_factor

    return jsonify({"ok": True, "calibration": calibration_summary(), "shrink_factor": shrink_factor(), "records": read_ledger()[-50:]})
