"""
v7 (2026-10-05, worker B) -- per-instrument-leg risk resolution.

Fix #7 (pip_size UX) and fix #11 (multi-instrument legs) share one root
cause: a single RiskConfig -- whose pip_size defaults to the FX-scale
0.0001 -- gets stamped onto market data it was never meant for. The web
templates now auto-detect on dataset select (see
app.web.templates._pip_size_autodetect), but callers that bypass the form
(the desktop app, API-style callers, and every multi-instrument fan-out,
which only ever had ONE shared form for N instruments) need the same
protection at the engine boundary.

`resolve_leg_risk` is that engine-boundary backstop, called once per
(instrument, dataframe) leg -- from app.search.batch_runner.run_search
(which already receives the leg's `instrument` label) and from
app.evolution.multi_instrument.MultiInstrumentEvolutionGroup.

Policy -- explicit user values are NEVER overridden:

* pip_size: only the UNTOUCHED FX default (exactly 0.0001) is ever
  replaced. Replacement prefers the known-instrument spec when the leg's
  label names a known contract (app.data.instrument_specs.
  guess_instrument_symbol), and falls back to data-driven
  app.backtest.risk.suggest_pip_size otherwise. A pip_size the user typed
  themselves (anything != 0.0001) is kept verbatim on every leg.
* contract_size / commission_per_contract / spread_pips / slippage_pips:
  filled from the matched spec ONLY when still at their untouched
  defaults (None / 0.0) -- an explicit nonzero value is never touched.
  When no spec matches and the pip was data-detected, contract_size is
  cleared (a contract $/point typed for a different instrument would
  silently mis-size this leg) -- mirroring the long-standing semantics
  of app.data.instrument_specs.resolve_risk_per_market, which this
  helper delegates to for the untouched-default path.
* Never raises: a leg that can't be resolved keeps the base risk and gets
  a note saying so, so one weird dataset can't sink a whole
  multi-instrument run.

Every decision is returned as human-readable note lines; callers are
expected to log them LOUDLY (the whole point is that a silently-wrong
pip_size is the #1 mechanical footgun in this app).
"""

from __future__ import annotations

from dataclasses import replace

from app.backtest.risk import RiskConfig

# The form default every lab template ships with -- FX scale. Anything
# else in this field means a human typed it.
FX_DEFAULT_PIP_SIZE = 0.0001


def _is_untouched_fx_default(pip_size: float) -> bool:
    try:
        return abs(float(pip_size) - FX_DEFAULT_PIP_SIZE) < 1e-12
    except (TypeError, ValueError):
        return False


def resolve_leg_risk(
    base_risk: RiskConfig,
    df,
    instrument_label: str | None,
) -> tuple[RiskConfig, list[str]]:
    """Resolve one instrument leg's RiskConfig from its own data + label.

    Returns (resolved_risk, notes). See the module docstring for the
    never-override-explicit-values policy.
    """
    label = instrument_label or "unknown"
    notes: list[str] = []
    try:
        from app.data.instrument_specs import (
            get_instrument_spec,
            guess_instrument_symbol,
            resolve_risk_per_market,
        )

        if _is_untouched_fx_default(base_risk.pip_size):
            # Untouched default: full per-leg resolution (spec preferred,
            # data-detected fallback) -- this is the auto-fix path.
            risks, report = resolve_risk_per_market(
                {label: df}, base_risk, auto_detect=True
            )
            resolved = risks[label]
            detail = report[0].describe() if report else ""
            notes.append(
                f"!!! [v7 pip_size backstop] {detail} -- your form still had the "
                f"untouched FX default pip_size=0.0001, so this leg's risk was "
                f"resolved from the instrument/data instead. Set pip_size "
                f"explicitly to keep your own value."
            )
            return resolved, notes

        # Explicit pip_size: kept verbatim on every leg. Only FILL fields
        # the user left at their untouched defaults, from the matched
        # spec when there is one -- purely additive, never an override.
        symbol = guess_instrument_symbol(label)
        spec = get_instrument_spec(symbol) if symbol else None
        if spec is None:
            notes.append(
                f"[v7 pip_size backstop] {label}: kept your explicit "
                f"pip_size={base_risk.pip_size:g} (no known contract matched "
                f"the label, so contract/commission were left as set)."
            )
            return base_risk, notes
        updates: dict = {}
        filled: list[str] = []
        if base_risk.contract_size is None:
            updates["contract_size"] = spec.contract_size
            filled.append(f"contract ${spec.contract_size:g}/point")
        if base_risk.commission_per_contract == 0.0:
            updates["commission_per_contract"] = spec.default_commission_round_turn
            filled.append(f"commission ${spec.default_commission_round_turn:g}/contract")
        if base_risk.spread_pips == 0.0:
            updates["spread_pips"] = spec.default_spread_pips  # ticks -> pips via tick_size (2026-10-07 cost-unit fix)
            filled.append(f"spread {spec.default_spread_ticks} tick(s)")
        if base_risk.slippage_pips == 0.0:
            updates["slippage_pips"] = spec.default_slippage_pips  # ticks -> pips via tick_size (2026-10-07 cost-unit fix)
            filled.append(f"slippage {spec.default_slippage_ticks} tick(s)")
        resolved = replace(base_risk, **updates) if updates else base_risk
        notes.append(
            f"[v7 pip_size backstop] {label}: kept your explicit "
            f"pip_size={base_risk.pip_size:g}"
            + (f"; filled from the {spec.symbol} spec: " + ", ".join(filled) if filled
               else "; nothing else needed filling.")
        )
        return resolved, notes
    except Exception as exc:  # noqa: BLE001 -- never sink a run on resolution
        notes.append(
            f"[v7 pip_size backstop] {label}: per-leg risk resolution failed "
            f"({exc}); kept the shared risk settings unchanged."
        )
        return base_risk, notes
