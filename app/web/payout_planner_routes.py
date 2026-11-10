"""Payout-EV planner page: "payout faster" as a managed number instead of a hope.

Pick a saved strategy and a dataset, tick the accounts to compare (Lucid 50k and 100k are ticked by default), optionally enter an
evaluation price the fee table does not have, and the page backtests the strategy at EACH account's own size, runs the Monte Carlo under
that account's own rules and turns the pass / first-payout probabilities into: expected attempts to a first payout, expected fee
spend, expected calendar days, the budget for 90% confidence, and what running two or three evaluations in parallel does to the timeline.

All maths lives in app.prop.payout_planner (closed form, unit-tested); this file only gathers inputs and renders.
"""
from __future__ import annotations

import tempfile

from flask import Blueprint, render_template, request

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig, commission_from_mapping
from app.data.importer import import_csv
from app.data.storage import get_raw_data_dir, list_stored_datasets
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop import payout_planner as planner
from app.prop.presets import get_fee_info, get_preset, list_presets
from app.search.strategy_space import build_strategy_from_spec
from app.strategy.library import list_saved_strategies, load_strategy_text
from app.web.accuracy_form import harden_risk_config

payout_planner_bp = Blueprint("payout_planner", __name__)

DEFAULT_ACCOUNTS = ("lucid_50k", "lucid_100k")
MAX_ACCOUNTS = 6
MAX_LANES = 4


def _price_or_none(raw) -> float | None:
    raw = (raw or "").strip().replace("$", "").replace(",", "")
    if not raw:
        return None
    value = float(raw)
    if value < 0:
        raise ValueError("Prices cannot be negative.")
    return value


def _accounts_for_form() -> list[dict]:
    rows = []
    for preset in list_presets():
        fee = get_fee_info(preset.key)
        rows.append({
            "key": preset.key, "label": preset.label, "account_size": preset.account_size,
            "known_price": getattr(fee, "challenge_price", None), "default": preset.key in DEFAULT_ACCOUNTS,
        })
    # The two defaults first, then everything else in catalog order.
    rows.sort(key=lambda r: (not r["default"], DEFAULT_ACCOUNTS.index(r["key"]) if r["default"] else 0))
    return rows


def _render(*, error: str | None = None, result=None, form_values: dict | None = None):
    return render_template(
        "payout_planner.html", active_page="payout_planner", error=error, result=result, form_values=form_values or {},
        strategies=[{"type": s.strategy_type, "name": s.name} for s in list_saved_strategies()],
        stored_datasets=list_stored_datasets(), accounts=_accounts_for_form(), max_lanes=MAX_LANES,
    )


@payout_planner_bp.route("/payout-planner", methods=["GET"])
def payout_planner_form():
    return _render()


def _risk_for(form, account_size: float) -> RiskConfig:
    risk = RiskConfig(
        initial_balance=float(account_size), risk_mode="percent", risk_value=float(form.get("risk_value", 0.5) or 0.5),
        commission_per_trade=commission_from_mapping(form),
        pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        contract_size=(float(form.get("contract_size")) if form.get("contract_size") else None),
    )
    return risk


@payout_planner_bp.route("/payout-planner/run", methods=["POST"])
def payout_planner_run():
    form = request.form
    form_values = form.to_dict()
    pick = (form.get("strategy_pick") or "").strip()
    dataset_name = (form.get("existing_dataset") or "").strip()
    keys = [k for k in form.getlist("accounts") if k][:MAX_ACCOUNTS]
    if "::" not in pick:
        return _render(error="Step 1: pick a saved strategy.", form_values=form_values)
    if not dataset_name:
        return _render(error="Step 2: pick the market data to test it on.", form_values=form_values)
    if not keys:
        return _render(error="Step 3: tick at least one account to plan for.", form_values=form_values)

    try:
        import_result = import_csv(get_raw_data_dir() / dataset_name)
    except Exception as exc:  # noqa: BLE001
        return _render(error=f"Could not load the data: {exc}", form_values=form_values)
    if not import_result.is_valid:
        return _render(error=f"Could not load the data: {'; '.join(import_result.errors)}", form_values=form_values)

    try:
        lanes = min(max(int(form.get("lanes", 2) or 2), 1), MAX_LANES)
        n_sims = min(max(int(form.get("n_sims", 1500) or 1500), 200), 10_000)
        independence = min(max(float(form.get("independence", planner.DEFAULT_PARALLEL_INDEPENDENCE) or 0.5), 0.0), 1.0)
        prices = {k: _price_or_none(form.get(f"price_{k}")) for k in keys}
    except ValueError as exc:
        return _render(error=f"Check the numbers you entered: {exc}", form_values=form_values)

    strategy_type, filename = pick.split("::", 1)
    try:
        code_text = load_strategy_text(strategy_type, filename)
    except (FileNotFoundError, ValueError) as exc:
        return _render(error=f"Could not load the saved strategy: {exc}", form_values=form_values)
    spec = {"source_type": strategy_type, "code_text": code_text}

    inputs, skipped = [], []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for key in keys:
            try:
                preset = get_preset(key)
                rules = preset.to_prop_rules()
                risk = harden_risk_config(_risk_for(form, preset.account_size), prop_rules=rules, form=form)
                strategy = build_strategy_from_spec(spec, tmp_dir=tmp_dir)
                backtest = run_backtest(import_result.dataframe, strategy, risk)
                if not backtest.trades:
                    skipped.append(f"{preset.label}: the strategy made no trades at this size.")
                    continue
                mc = run_monte_carlo(backtest.trades, rules, MonteCarloConfig(n_simulations=n_sims))
            except Exception as exc:  # noqa: BLE001 -- one bad account must not sink the comparison
                skipped.append(f"{key}: {exc}")
                continue
            fee = get_fee_info(key)
            inp = planner.from_monte_carlo(preset.label, mc, fee=fee, preset_key=key, challenge_price=prices.get(key))
            if fee is not None and getattr(fee, "monthly_subscription", False):
                inp.notes.append("This firm bills monthly; the fee shown assumes one billing period per attempt.")
            inputs.append(inp)

    if not inputs:
        return _render(error="Nothing could be planned. " + " ".join(skipped), form_values=form_values)

    comparison = planner.compare_options(inputs)
    parallel = [{"label": inp.label, "plans": [planner.parallel_plan(inp, n, independence).to_dict() for n in range(1, lanes + 1)]}
                for inp in inputs]
    return _render(result={
        "strategy": filename, "dataset": dataset_name, "n_sims": n_sims, "lanes": lanes, "independence": independence,
        "comparison": comparison, "parallel": parallel, "skipped": skipped,
    }, form_values=form_values)
