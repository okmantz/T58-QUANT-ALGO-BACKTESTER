"""
Hedge Fund Manager web routes.

Registered as a Flask Blueprint (`hedge_fund_bp`), same pattern as
app.web.quant_lab_routes and app.web.ai_assistant_routes -- this whole
feature area lives in one file and app/web/server.py's own change is a
single import + one `app.register_blueprint(...)` line.

Scope note (documented, not silently skipped): the underlying pipeline
(app.hedge_fund.research.generate_views) supports per-asset "use one of
my saved Strategy Library strategies as this asset's view generator"
views, not just the statistical bootstrap. This tab's form does not yet
expose that picker -- it always uses the bootstrap view generator. Wiring
the same library-strategy-picker UI that app/web/templates/portfolio.html
already has (see its `leg{i}_library_mode` fields) into this form is a
natural next increment; in the meantime, the strategy-signal view path is
fully implemented, tested (tests/test_hedge_fund_research.py), and usable
directly from Python via `run_hedge_fund_manager(..., strategies={...})`.
"""
from __future__ import annotations

from flask import Blueprint, render_template, request

from app.data.importer import import_csv, import_csv_bytes
from app.data.storage import get_raw_data_dir, list_datasets_by_instrument, list_stored_datasets, store_csv_bytes
from app.hedge_fund.pipeline import HedgeFundManagerError, run_hedge_fund_manager
from app.hedge_fund.rebalancer import RebalanceConfig
from app.hedge_fund.research import EnsembleForecastConfig
from app.strategy.base import StrategyError

hedge_fund_bp = Blueprint("hedge_fund", __name__, url_prefix="/hedge-fund")

MAX_ASSETS = 6


def _resolve_asset_dataset(form, files, prefix: str):
    """Resolves ONE asset slot's dataset -- same shape as
    app.web.server._resolve_leg_dataset (Multi-Asset Portfolio's own
    per-leg resolver), duplicated here rather than imported so this
    blueprint has no import-time dependency on server.py (server.py is
    the one that imports blueprints, not the reverse). Returns
    (df, label) or (None, None) if this slot wasn't filled in."""
    uploaded = files.get(f"{prefix}_csv")
    if uploaded and uploaded.filename:
        content = uploaded.read()
        result = import_csv_bytes(content, filename=uploaded.filename)
        if result.is_valid:
            store_csv_bytes(content, uploaded.filename)
            return result.dataframe, uploaded.filename
        raise StrategyError(f"'{uploaded.filename}': {'; '.join(result.errors)}")
    existing_choice = (form.get(f"{prefix}_existing_dataset") or "").strip()
    if existing_choice:
        candidate = get_raw_data_dir() / existing_choice
        if candidate.exists():
            result = import_csv(candidate)
            if result.is_valid:
                return result.dataframe, existing_choice
    return None, None


def _asset_label(form, prefix: str, fallback: str) -> str:
    custom = (form.get(f"{prefix}_label") or "").strip()
    return custom or fallback


def _svg_line_chart(values: list[float], width: int = 680, height: int = 160) -> str:
    """A dependency-free inline SVG sparkline for the equity curve --
    this app already leans on lightweight-charts for candlestick data
    (see app/web/templates/live_market.html), but that's real overkill
    for a single equity line; plain SVG needs no extra script tag and
    renders identically on desktop and mobile browsers."""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pad = 8
    n = len(values)
    points = []
    for i, v in enumerate(values):
        x = pad + (i / (n - 1)) * (width - 2 * pad)
        y = height - pad - ((v - lo) / span) * (height - 2 * pad)
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    color = "#35e0b0" if values[-1] >= values[0] else "#ff6b6b"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}" /></svg>'
    )


@hedge_fund_bp.route("/")
def hedge_fund_form():
    return render_template(
        "hedge_fund.html", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        max_assets=MAX_ASSETS,
    )


@hedge_fund_bp.route("/run", methods=["POST"])
def hedge_fund_run():
    form = request.form
    ctx = lambda **kw: dict(
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        max_assets=MAX_ASSETS, **kw,
    )

    price_data = {}
    try:
        for i in range(1, MAX_ASSETS + 1):
            prefix = f"asset{i}"
            df, label = _resolve_asset_dataset(form, request.files, prefix)
            if df is None:
                continue
            label = _asset_label(form, prefix, label)
            if label in price_data:
                label = f"{label} ({i})"
            price_data[label] = df
    except StrategyError as exc:
        return render_template("hedge_fund.html", **ctx(error=str(exc))), 400

    if len(price_data) < 2:
        return render_template("hedge_fund.html", **ctx(
            error="A hedge fund book needs at least 2 assets -- fill in a market data file/dataset for at least 2 of the asset slots below.",
        )), 400

    try:
        forecast_config = EnsembleForecastConfig(
            lookback_bars=int(form.get("lookback_bars", 100) or 100),
            horizon_bars=int(form.get("horizon_bars", 5) or 5),
            n_samples=int(form.get("n_samples", 32) or 32),
        )
        config = RebalanceConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            rebalance_every_bars=int(form.get("rebalance_every_bars", 5) or 5),
            forecast=forecast_config,
            confidence=float(form.get("confidence", 0.5) or 0.5),
            long_only=form.get("long_only", "on") == "on",
            max_turnover=(float(form["max_turnover"]) if form.get("max_turnover") not in (None, "", "none") else None),
            transaction_cost_bps=float(form.get("transaction_cost_bps", 10.0) or 10.0),
            min_trade_frac=float(form.get("min_trade_frac", 0.005) or 0.005),
        )
    except (TypeError, ValueError) as exc:
        return render_template("hedge_fund.html", **ctx(error=f"Invalid setting: {exc}")), 400

    write_journal = form.get("write_journal", "on") == "on"

    try:
        outcome = run_hedge_fund_manager(price_data, config, strategies=None, write_journal=write_journal)
    except HedgeFundManagerError as exc:
        return render_template("hedge_fund.html", **ctx(error=str(exc))), 400

    equity_values = outcome.backtest.equity_curve["equity"].tolist()
    result = {
        "assets": list(price_data.keys()),
        "stats": outcome.backtest.stats,
        "equity_svg": _svg_line_chart(equity_values),
        "equity_start": equity_values[0],
        "equity_end": equity_values[-1],
        "num_cycles": len(outcome.backtest.cycles),
        "last_cycle": outcome.backtest.cycles[-1] if outcome.backtest.cycles else None,
        "run_warnings": outcome.backtest.warnings,
        "journal": outcome.journal,
        "journal_note": outcome.journal_note,
        "confidence_sweep": outcome.confidence_sweep_diagnostic,
    }
    return render_template("hedge_fund.html", **ctx(result=result))
