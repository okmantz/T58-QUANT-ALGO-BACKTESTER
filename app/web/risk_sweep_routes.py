"""
Risk Sweep -- the missing UI for app.optimize.risk_sweep.run_risk_sweep.

That module was already fully implemented and covered by 6 passing tests
(tests/test_risk_sweep.py) -- it re-runs a strategy at 8 candidate
risk-per-trade levels (0.10% through 1.00%) and scores each one against
the full app.prop.survival_engine model (pass probability, first-payout
probability, median days to payout, drawdown recovery) to find which
risk level actually maximizes odds of reaching payout, not just which
one looks best on a single backtest. Its own docstring makes the point:
don't assume your current risk-per-trade is optimal, and don't optimize
for one trade's return.

The only thing missing was a way to reach it: run_risk_sweep was
imported once in app/web/server.py and never called from any route. This
file is that route, registered as a Flask Blueprint -- same pattern as
app/web/extra_routes.py, app/web/quant_lab_routes.py, etc: this whole
feature area lives in one file, and app/web/server.py's own change is a
single import + one `app.register_blueprint(risk_sweep_bp)` line.

Mirrors app/web/extra_routes.py's Compare Strategies flow (pick a saved
strategy + an existing stored dataset, backtest synchronously, render a
table) rather than the background-job/polling pattern used by Refine or
Search Lab: a sweep is 8 backtests + 8 survival analyses, the same order
of magnitude of work as Compare's "a handful of sequential backtests",
not the hundreds-to-thousands of runs that justify a job/thread/log-poll
page elsewhere in this app.
"""
from __future__ import annotations

import json
import tempfile

from flask import Blueprint, render_template, request

from app.backtest.risk import RiskConfig
from app.data.importer import import_csv
from app.data.storage import get_raw_data_dir, list_stored_datasets
from app.optimize.risk_sweep import DEFAULT_RISK_VALUES, run_risk_sweep
from app.prop.presets import list_presets as list_prop_firm_presets
from app.prop.simulator import PropRules
from app.prop.survival_engine import PropSurvivalConfig
from app.search.strategy_space import build_strategy_from_spec
from app.strategy.library import list_saved_strategies, load_strategy_text
from app.web.alpaca_shared import alpaca_template_context

risk_sweep_bp = Blueprint("risk_sweep", __name__)


def _prop_presets_json() -> str:
    """Same shape as app.web.server's private _prop_presets_json() --
    duplicated here (rather than imported) to avoid a circular import
    between this blueprint and server.py, same reasoning as the twin
    function in app/web/extra_routes.py."""
    return json.dumps([p.to_dict() for p in list_prop_firm_presets()])


def _prop_rules_from_form(form) -> PropRules:
    """Same base field names as the main Run &amp; Report form
    (app/web/server.py's run_pipeline) -- account_size, profit_target,
    daily_loss, max_dd, dd_type, dd_check_mode, consistency, min_days,
    payout_threshold, payout_freq, buffer, payout_cap."""
    payout_cap = (form.get("payout_cap") or "").strip()
    return PropRules(
        account_size=float(form.get("account_size", 100000) or 100000),
        evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
        daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
        max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        drawdown_type=form.get("dd_type", "trailing"),
        drawdown_check_mode=form.get("dd_check_mode", "intrabar"),
        consistency_rule_pct=float(form["consistency"]) if form.get("consistency") else None,
        min_trading_days=int(form.get("min_days", 5) or 5),
        payout_threshold_pct=float(form.get("payout_threshold", 0) or 0),
        payout_cap_pct=float(payout_cap) if payout_cap else None,
        payout_frequency_days=int(form.get("payout_freq", 14) or 14),
        required_buffer_pct=float(form.get("buffer", 0) or 0),
    )


def _parse_risk_values(raw: str) -> list[float] | None:
    """Comma-separated risk levels from the form, e.g. '0.1, 0.25, 0.5'.
    Blank -> None, which run_risk_sweep resolves to DEFAULT_RISK_VALUES
    (the exact 0.10%-1.00% list the Masterclass material suggests)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    values: list[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return values or None


def _render_form(*, error: str | None = None, result=None, form_values: dict | None = None,
                  alpaca_notice: str | None = None, alpaca_notice_kind: str = "info"):
    return render_template(
        "risk_sweep.html", active_page="risk_sweep",
        strategies=[{"type": s.strategy_type, "name": s.name} for s in list_saved_strategies()],
        stored_datasets=list_stored_datasets(),
        prop_presets_json=_prop_presets_json(),
        default_risk_values=", ".join(f"{v:.2f}" for v in DEFAULT_RISK_VALUES),
        error=error, result=result, form_values=form_values or {},
        alpaca_notice=alpaca_notice, alpaca_notice_kind=alpaca_notice_kind,
        **alpaca_template_context(),
    )


@risk_sweep_bp.route("/risk-sweep", methods=["GET"])
def risk_sweep_form():
    return _render_form(
        alpaca_notice=request.args.get("alpaca_notice"),
        alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"),
    )


@risk_sweep_bp.route("/risk-sweep/run", methods=["POST"])
def risk_sweep_run():
    form = request.form
    form_values = form.to_dict()

    pick = (form.get("strategy_pick") or "").strip()
    dataset_name = (form.get("existing_dataset") or "").strip()

    if "::" not in pick:
        return _render_form(error="Pick one saved strategy to sweep.", form_values=form_values)
    strategy_type, filename = pick.split("::", 1)

    if not dataset_name:
        return _render_form(error="Choose a dataset to backtest every risk level against.", form_values=form_values)

    try:
        candidate = get_raw_data_dir() / dataset_name
        import_result = import_csv(candidate)
    except Exception as exc:  # noqa: BLE001
        return _render_form(error=f"Could not load dataset: {exc}", form_values=form_values)
    if not import_result.is_valid:
        return _render_form(
            error=f"Could not load dataset: {'; '.join(import_result.errors)}", form_values=form_values,
        )

    try:
        risk_values = _parse_risk_values(form.get("risk_values", ""))
    except ValueError:
        return _render_form(error="Risk levels must be a comma-separated list of numbers, e.g. 0.1, 0.25, 0.5.",
                             form_values=form_values)

    try:
        base_risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode="percent",  # sweeping risk LEVELS only makes sense in percent mode -- see run_risk_sweep's docstring
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            max_trades_per_day=int(form.get("max_trades_day", 10) or 10),
            commission_per_trade=float(form.get("commission", 0) or 0),
            slippage_pips=float(form.get("slippage_pips", 0.5) or 0.5),
            spread_pips=float(form.get("spread_pips", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        prop_rules = _prop_rules_from_form(form)
    except ValueError as exc:
        return _render_form(error=f"Invalid input: {exc}", form_values=form_values)

    quick_mode = form.get("quick_mode") == "on"
    survival_cfg = PropSurvivalConfig(n_simulations=1000, life_simulations=250) if quick_mode else None

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            code_text = load_strategy_text(strategy_type, filename)
        except (FileNotFoundError, ValueError) as exc:
            return _render_form(error=f"Could not load saved strategy: {exc}", form_values=form_values)

        spec = {"source_type": strategy_type, "code_text": code_text}

        def strategy_builder():
            # Fresh instance per call, same "always build fresh" convention
            # run_risk_sweep's own docstring calls out (some strategy
            # sources cache internal state keyed to the data they last saw).
            return build_strategy_from_spec(spec, tmp_dir=tmp_dir)

        try:
            sweep = run_risk_sweep(
                import_result.dataframe, strategy_builder, base_risk, prop_rules,
                risk_values=risk_values, survival_cfg=survival_cfg,
            )
        except ValueError as exc:
            return _render_form(error=str(exc), form_values=form_values)
        except Exception as exc:  # noqa: BLE001 -- surface on the page, don't 500 on a bad candidate
            return _render_form(error=f"Risk Sweep failed: {exc}", form_values=form_values)

    return _render_form(
        result={
            "strategy_type": strategy_type, "filename": filename, "dataset": dataset_name,
            "quick_mode": quick_mode, "points": [p.to_dict() for p in sweep.points],
            "best_point": sweep.best_point.to_dict() if sweep.best_point else None,
            "notes": sweep.notes, "table": sweep.render_table(),
        },
        form_values=form_values,
    )
