"""Web-form support for the accuracy overhaul.

* sizing_mode / max_stop_dollars / fixed_contracts / intrabar_replay /
  account_model are read from the submitted form wherever a RiskConfig is
  built (server.py wraps RiskConfig with `risk_config_from_request`), so
  every existing tool page gets them just by including the fragment
  `_accuracy_fields.html`.
* `/api/accuracy/instrument/<symbol>` is the 'pick instrument first' source:
  pip size, $/point, tick costs, commission and round-turn cost, all from the
  instrument spec, so nothing is hand-typed.
* `check_form_mismatch` blocks a run whose hand-typed pip size disagrees with
  the chosen instrument.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

accuracy_bp = Blueprint("accuracy_form", __name__)

SIZING_MODES = ("skip", "fit_stop", "fixed_contracts", "micro_fallback")


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def accuracy_kwargs(form) -> dict:
    """RiskConfig kwargs derived from the accuracy fields of a form."""
    out: dict = {}
    mode = (form.get("sizing_mode") or "").strip()
    if mode in SIZING_MODES:
        out["sizing_mode"] = mode
    if _num(form.get("max_stop_dollars")) is not None:
        out["max_stop_dollars"] = _num(form.get("max_stop_dollars"))
    if mode == "fixed_contracts" and _num(form.get("fixed_contracts")):
        out["fixed_contracts"] = int(_num(form.get("fixed_contracts")))
    if form.get("intrabar_replay") in ("on", "1", "true", "True"):
        out["intrabar_replay"] = True
    if (form.get("account_model") or "") == "prop":
        out["account_model"] = "prop"
    return out


def risk_config_from_request(base_cls, *args, **kwargs):
    """Drop-in for RiskConfig(...) inside request handlers."""
    try:
        extra = accuracy_kwargs(request.form) if request else {}
    except RuntimeError:  # no request context (scripts, tests)
        extra = {}
    for k, v in extra.items():
        kwargs.setdefault(k, v)
    return base_cls(*args, **kwargs)


def check_form_mismatch(form) -> str | None:
    """None if fine; otherwise the plain-language reason to block the run."""
    sym_in = (form.get("acc_instrument") or "").strip()
    if not sym_in:
        return None
    from app.data.instrument_specs import get_any_instrument_spec as get_instrument_spec, guess_any_instrument_symbol as guess_instrument_symbol
    spec = get_instrument_spec(guess_instrument_symbol(sym_in) or sym_in)
    if spec is None:
        return None
    pip = _num(form.get("pip_size"))
    if pip is not None and abs(pip - spec.pip_size) > 1e-12:
        return (f"pip_size={pip:g} does not match {spec.symbol} ({spec.pip_size:g}). Every stop, target, cost and "
                f"position size would be scaled wrongly. Pick the instrument first and let the form fill pip size.")
    cs = _num(form.get("contract_size"))
    if cs is not None and abs(cs - spec.contract_size) > 1e-9:
        return f"contract size ${cs:g}/point does not match {spec.symbol} (${spec.contract_size:g}/point)."
    return None


@accuracy_bp.before_app_request
def _block_mismatch():
    if request.method != "POST" or not request.form:
        return None
    msg = check_form_mismatch(request.form)
    if msg:
        return jsonify({"ok": False, "error": msg}), 400
    return None


@accuracy_bp.route("/api/accuracy/instrument/<symbol>")
def instrument_defaults(symbol: str):
    from app.data.instrument_specs import get_any_instrument_spec as get_instrument_spec, guess_any_instrument_symbol as guess_instrument_symbol
    spec = get_instrument_spec(guess_instrument_symbol(symbol) or symbol)
    if spec is None:
        return jsonify({"ok": False, "error": f"unknown instrument {symbol!r}"}), 404
    return jsonify({
        "ok": True, "symbol": spec.symbol, "pip_size": spec.pip_size, "contract_size": spec.contract_size,
        "tick_size": spec.tick_size, "spread_ticks": spec.default_spread_ticks,
        "slippage_ticks": spec.default_slippage_ticks, "commission_per_contract": spec.default_commission_round_turn,
        "spread_pips": spec.default_spread_pips, "slippage_pips": spec.default_slippage_pips,
        "round_trip_cost_dollars": spec.round_trip_cost_dollars(),
    })


@accuracy_bp.route("/api/accuracy/preset/<key>")
def preset_defaults(key: str):
    from app.prop.presets import get_preset
    p = get_preset(key)
    if p is None:
        return jsonify({"ok": False, "error": "unknown preset"}), 404
    return jsonify({"ok": True, "preset": p.to_dict(), "checked": p.rules_checked_on or "not re-verified"})
