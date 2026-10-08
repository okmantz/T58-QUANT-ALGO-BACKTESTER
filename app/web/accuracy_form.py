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


def instrument_from_form(form) -> str | None:
    """Best-effort instrument symbol for a submitted form: the explicit
    accuracy picker wins; otherwise guess from the stored-dataset name."""
    if form is None:
        return None
    sym = (form.get("acc_instrument") or "").strip()
    if sym:
        return sym
    label = (form.get("existing_dataset") or form.get("dataset") or "").strip()
    if label:
        try:
            from app.data.instrument_specs import guess_any_instrument_symbol
            guessed = guess_any_instrument_symbol(label)
            if guessed:
                return guessed
        except Exception:  # noqa: BLE001 -- hardening must never break a run
            pass
    return None


def prop_rules_from_form(form):
    """PropRules built from the standard prop-rule form fields, or None.

    Returns None when the form carries no prop-rule fields at all (so a
    plain backtest form is never silently turned into a prop evaluation).
    Tolerant of blanks: the same defaults the /run handler uses apply.
    """
    if form is None:
        return None
    keys = ("account_size", "profit_target", "daily_loss", "max_dd", "prop_preset")
    if not any(form.get(k) not in (None, "") for k in keys):
        # No explicit prop fields: only harden with prop rules when the
        # accuracy fragment explicitly asked for the prop account model.
        if (form.get("account_model") or "") != "prop":
            return None
    try:
        from app.prop.simulator import PropRules
        preset_key = (form.get("prop_preset") or "").strip()
        if preset_key:
            try:
                from app.prop.presets import get_preset
                preset = get_preset(preset_key)
                if preset is not None:
                    return preset.to_prop_rules()
            except Exception:  # noqa: BLE001
                pass
        def _f(name, default):
            v = _num(form.get(name))
            return default if v is None else v
        payout_cap = _num(form.get("payout_cap"))
        consistency = _num(form.get("consistency"))
        return PropRules(
            account_size=_f("account_size", 100000.0),
            evaluation_profit_target_pct=_f("profit_target", 8.0),
            daily_loss_limit_pct=_f("daily_loss", 5.0),
            max_drawdown_pct=_f("max_dd", 10.0),
            drawdown_type=(form.get("dd_type") or "trailing"),
            drawdown_check_mode=(form.get("dd_check_mode") or "intrabar"),
            consistency_rule_pct=consistency,
            min_trading_days=int(_f("min_days", 5)),
            payout_threshold_pct=_f("payout_threshold", 0.0),
            payout_cap_pct=payout_cap,
            payout_frequency_days=int(_f("payout_freq", 14)),
            required_buffer_pct=_f("buffer", 0.0),
        )
    except Exception:  # noqa: BLE001 -- hardening must never break a run
        return None


def harden_risk_config(risk, prop_rules=None, instrument=None, form=None):
    """Single web chokepoint: harden ``risk`` via build_run_context.

    ``build_run_context(risk, prop_rules=None, instrument=None)`` is the
    backend-owned hardened builder (app.backtest.risk). When it is not
    importable yet, fall back to the equivalent composition here:
    instrument spec first, then with_prop_safety_defaults. Never raises.
    """
    if risk is None:
        return risk
    if form is not None:
        if prop_rules is None:
            prop_rules = prop_rules_from_form(form)
        if instrument is None:
            instrument = instrument_from_form(form)
    try:
        from app.backtest.risk import build_run_context
        return build_run_context(risk, prop_rules=prop_rules, instrument=instrument)
    except Exception:  # noqa: BLE001 -- fall back below
        pass
    try:
        hardened = risk
        if instrument:
            try:
                from app.data.instrument_specs import (
                    apply_any_instrument_spec,
                    guess_any_instrument_symbol,
                    get_any_instrument_spec,
                )
                sym = guess_any_instrument_symbol(str(instrument)) or str(instrument)
                if get_any_instrument_spec(sym) is not None:
                    hardened = apply_any_instrument_spec(hardened, sym)
            except Exception:  # noqa: BLE001
                pass
        if prop_rules is not None:
            try:
                from app.backtest.risk import with_prop_safety_defaults
                hardened = with_prop_safety_defaults(hardened, prop_rules)
            except Exception:  # noqa: BLE001
                pass
        return hardened
    except Exception:  # noqa: BLE001
        return risk


def describe_simulation_text(risk, prop_rules=None) -> str:
    """Plain-English 'How this was simulated' lines for web reports.

    Uses backend describe_simulation(risk, prop_rules=None) when present;
    otherwise composes the same facts locally. Never raises.
    """
    try:
        from app.backtest.risk import describe_simulation
        text = describe_simulation(risk, prop_rules=prop_rules)
        if text:
            return text if isinstance(text, str) else "\n".join(str(x) for x in text)
    except Exception:  # noqa: BLE001
        pass
    try:
        lines = [
            f"Sizing mode: {getattr(risk, 'sizing_mode', 'skip')} "
            f"(budget ${float(getattr(risk, 'initial_balance', 0) or 0):,.0f}, "
            f"risk {getattr(risk, 'risk_value', '?')} {getattr(risk, 'risk_mode', '')}).",
            f"Account model: {getattr(risk, 'account_model', 'legacy')}; "
            f"intrabar replay: {'on' if getattr(risk, 'intrabar_replay', False) else 'off'}.",
            f"Costs: pip size {getattr(risk, 'pip_size', '?')}, "
            f"contract ${getattr(risk, 'contract_size', '?')}/point, "
            f"spread {getattr(risk, 'spread_pips', 0)} + slippage {getattr(risk, 'slippage_pips', 0)} pips, "
            f"commission ${float(getattr(risk, 'commission_per_trade', 0) or 0):,.2f}/trade.",
        ]
        if prop_rules is not None:
            lines.append(
                f"Prop firm: ${float(getattr(prop_rules, 'account_size', 0) or 0):,.0f} account, "
                f"+{getattr(prop_rules, 'evaluation_profit_target_pct', '?')}% target, "
                f"-{getattr(prop_rules, 'max_drawdown_pct', '?')}% max drawdown "
                f"({getattr(prop_rules, 'drawdown_type', '?')})."
            )
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return "Simulation details unavailable."


def full_risk_kwargs(form) -> dict:
    """All RiskConfig fields the standard run forms collect.

    Mirror of what the /run handler reads, so handlers that historically
    read only balance/pip/contract/commission (Sensitivity, Parameter
    Robustness) can wire the full typed risk through instead of dropping
    it. Only fields actually present (non-blank) in the form are returned.
    """
    out: dict = {}
    def _put(name, caster, key=None):
        raw = form.get(name)
        if raw in (None, ""):
            return
        try:
            out[key or name] = caster(raw)
        except (TypeError, ValueError):
            return
    _put("initial_balance", float)
    _put("risk_mode", str)
    _put("risk_value", float)
    _put("max_trades_day", int, "max_trades_per_day")
    _put("pip_size", float)
    _put("contract_size", float)
    _put("spread_pips", float)
    _put("slippage_pips", float)
    _put("commission", float, "commission_per_trade")
    _put("commission_per_contract", float)
    _put("max_stop_dollars", float)
    # Accuracy-fragment fields ride along too (same read path).
    for k, v in accuracy_kwargs(form).items():
        out.setdefault(k, v)
    return out


def risk_config_from_request(base_cls, *args, **kwargs):
    """Drop-in for RiskConfig(...) inside request handlers.

    Builds the config (accuracy-fragment fields included), then hardens
    it through build_run_context with the form's prop rules + instrument,
    so every route using this wrapper gets prop-safe sizing/account
    semantics automatically. Outside a request context it is a plain
    construction (scripts/tests)."""
    try:
        form = request.form if request else None
    except RuntimeError:  # no request context (scripts, tests)
        form = None
    if form is not None:
        extra = accuracy_kwargs(form)
        for k, v in extra.items():
            kwargs.setdefault(k, v)
    risk = base_cls(*args, **kwargs)
    if form is not None:
        risk = harden_risk_config(risk, form=form)
    return risk


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
