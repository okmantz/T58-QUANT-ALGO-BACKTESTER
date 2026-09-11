"""
Hedge Fund Manager web routes.

Registered as a Flask Blueprint (`hedge_fund_bp`), same pattern as
app.web.quant_lab_routes and app.web.ai_assistant_routes -- this whole
feature area lives in one file and app/web/server.py's own change is a
single import + one `app.register_blueprint(...)` line.

Per-asset Strategy Library picker: each asset slot can optionally load
one of your saved Python/PineScript/MQL5 strategies to BE that asset's
research-desk view generator (app.hedge_fund.research.strategy_signal_view)
instead of the statistical bootstrap -- same "load from library" pattern
app/web/templates/portfolio.html already uses per-leg (`leg{i}_library_mode`
/ `leg{i}_library_name`), renamed to `asset{i}_...` here. Strategy loading
and building is duplicated from app.web.server's `load_strategy_text` /
`build_strategy_from_code` rather than imported from server.py, for the
same reason `_resolve_asset_dataset` below duplicates
`_resolve_leg_dataset` -- server.py imports blueprints, not the reverse.
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

from flask import Blueprint, render_template, request

from app.data.importer import import_csv, import_csv_bytes
from app.data.storage import get_raw_data_dir, list_datasets_by_instrument, list_stored_datasets, store_csv_bytes
from app.hedge_fund.pipeline import HedgeFundManagerError, run_hedge_fund_manager
from app.hedge_fund.rebalancer import RebalanceConfig
from app.hedge_fund.research import EnsembleForecastConfig
from app.strategy.base import Strategy, StrategyError
from app.strategy.library import STRATEGY_TYPES, list_saved_strategies, load_strategy_text
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy

hedge_fund_bp = Blueprint("hedge_fund", __name__, url_prefix="/hedge-fund")

MAX_ASSETS = 6
LIBRARY_STRATEGY_TYPES = [t for t in STRATEGY_TYPES if t != "manual"]  # manual configs aren't Strategy-Library files with source text


def _saved_strategies_json() -> str:
    """{"python": [{"name", "description"}, ...], "pinescript": [...],
    "mql5": [...]} -- deliberately a smaller shape than
    app.web.server's own _saved_strategies_json (this picker only needs
    enough to label the dropdown, not the full filter/search metadata
    that page's JS uses)."""
    return json.dumps({
        t: [{"name": s.name, "description": s.metadata.get("description", "")} for s in list_saved_strategies(t)]
        for t in LIBRARY_STRATEGY_TYPES
    })


def _build_strategy_from_code(mode: str, code: str) -> Strategy:
    """Single source of truth for turning a loaded library file's source
    text into a Strategy object -- same job app.web.server's own
    build_strategy_from_code does for the main backtest/portfolio forms,
    duplicated (not imported) for the reason in this module's docstring."""
    if mode == "python":
        tmp = Path(tempfile.mkdtemp()) / f"strategy_{uuid.uuid4().hex}.py"
        tmp.write_text(code, encoding="utf-8")
        return PythonStrategy(tmp)
    if mode == "pinescript":
        return PineScriptStrategy(code)
    if mode == "mql5":
        return MQL5Strategy(code)
    raise StrategyError(f"Unknown strategy mode: {mode}")


def _resolve_asset_view_strategy(form, prefix: str) -> Strategy | None:
    """Resolves ONE asset slot's optional view-generator strategy from
    the Strategy Library. Returns None (defer to the bootstrap view) if
    the slot's library picker was left on "none"."""
    mode = (form.get(f"{prefix}_library_mode") or "").strip()
    name = (form.get(f"{prefix}_library_name") or "").strip()
    if not mode or not name:
        return None
    try:
        code = load_strategy_text(mode, name)
    except (FileNotFoundError, OSError) as exc:
        raise StrategyError(f"Could not load saved strategy '{name}' ({mode}): {exc}") from exc
    return _build_strategy_from_code(mode, code)


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
        max_assets=MAX_ASSETS, saved_strategies_json=_saved_strategies_json(),
    )


@hedge_fund_bp.route("/run", methods=["POST"])
def hedge_fund_run():
    form = request.form
    ctx = lambda **kw: dict(
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        max_assets=MAX_ASSETS, saved_strategies_json=_saved_strategies_json(), **kw,
    )

    price_data = {}
    strategies: dict[str, Strategy] = {}
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
            view_strategy = _resolve_asset_view_strategy(form, prefix)
            if view_strategy is not None:
                strategies[label] = view_strategy
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
        outcome = run_hedge_fund_manager(price_data, config, strategies=strategies or None, write_journal=write_journal)
    except HedgeFundManagerError as exc:
        return render_template("hedge_fund.html", **ctx(error=str(exc))), 400

    equity_values = outcome.backtest.equity_curve["equity"].tolist()
    last_cycle = outcome.backtest.cycles[-1] if outcome.backtest.cycles else None
    view_methods = {asset: v.method for asset, v in last_cycle.views.items()} if last_cycle else {}
    result = {
        "assets": list(price_data.keys()),
        "assets_using_library_strategy": sorted(strategies.keys()),
        "view_methods": view_methods,
        "stats": outcome.backtest.stats,
        "equity_svg": _svg_line_chart(equity_values),
        "equity_start": equity_values[0],
        "equity_end": equity_values[-1],
        "num_cycles": len(outcome.backtest.cycles),
        "last_cycle": last_cycle,
        "run_warnings": outcome.backtest.warnings,
        "journal": outcome.journal,
        "journal_note": outcome.journal_note,
        "confidence_sweep": outcome.confidence_sweep_diagnostic,
    }
    return render_template("hedge_fund.html", **ctx(result=result))
