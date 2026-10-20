"""Single cost model, denominated in TICKS (report 2026-10-20, section 4: "cost stress from a single cost model").

Before this module the app had three cost vocabularies: pips on web forms (legacy 1.0 + 0.5 pips, ~3x too high on NQ/ES),
percent-of-notional in the cost ladder, and ticks in the instrument registry. Everything now resolves through here:

    per side:   spread_ticks + slippage_ticks   (charged on entry AND exit)
    per trade:  commission_round_turn dollars per contract
    stress:     multiply the TICK components (not commission) by k -- the 2x cost-stress gate is `stress(2.0)`

A broker profile can override the registry defaults per instrument ("real fill costs" once the user has measured them from a
forward-test fill ledger -- see app.forward_test.fill_ledger). Nothing here changes what a run charges unless a caller asks.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from app.data.instrument_specs import InstrumentSpec, get_any_instrument_spec, guess_any_instrument_symbol


@dataclass(frozen=True)
class CostLine:
    symbol: str
    spread_ticks: float
    slippage_ticks: float
    commission_round_turn: float
    tick_size: float
    tick_value: float
    pip_size: float
    source: str = "registry default"

    @property
    def spread_pips(self) -> float:
        return self.spread_ticks * self.tick_size / self.pip_size

    @property
    def slippage_pips(self) -> float:
        return self.slippage_ticks * self.tick_size / self.pip_size

    def round_trip_dollars(self, contracts: float = 1.0) -> float:
        return (2.0 * (self.spread_ticks + self.slippage_ticks) * self.tick_value + self.commission_round_turn) * float(contracts)

    def stress(self, multiplier: float = 2.0) -> "CostLine":
        """Tick costs scaled by `multiplier`; commission is a posted fee and is NOT scaled."""
        m = float(multiplier)
        return replace(self, spread_ticks=self.spread_ticks * m, slippage_ticks=self.slippage_ticks * m, source=f"{self.source} x{m:g} stress")

    def breakeven_r_cost(self, stop_ticks: float, contracts: float = 1.0) -> float:
        """Round-trip cost expressed in R for a stop of `stop_ticks` ticks (the quantity that eats a small edge)."""
        risk = float(stop_ticks) * self.tick_value * float(contracts)
        return self.round_trip_dollars(contracts) / risk if risk > 0 else float("inf")


@dataclass
class BrokerProfile:
    """Measured or assumed costs per instrument. Unlisted instruments fall back to the registry defaults."""
    name: str = "registry"
    spread_ticks: dict = field(default_factory=dict)
    slippage_ticks: dict = field(default_factory=dict)
    commission_round_turn: dict = field(default_factory=dict)


# Deliberately conservative placeholders until the forward-test fill ledger supplies measured numbers.
PROFILES: dict[str, BrokerProfile] = {
    "registry": BrokerProfile("registry"),
    "conservative": BrokerProfile("conservative", slippage_ticks={"NQ": 2.0, "MNQ": 2.0, "ES": 1.5, "MES": 1.5, "GC": 2.0, "MGC": 2.0}),
}


def resolve_cost(symbol_or_label: str, profile: BrokerProfile | str | None = None) -> CostLine | None:
    sym = guess_any_instrument_symbol(symbol_or_label) or str(symbol_or_label or "").strip().upper()
    spec: InstrumentSpec | None = get_any_instrument_spec(sym)
    if spec is None:
        return None
    prof = PROFILES.get(profile or "registry") if isinstance(profile, (str, type(None))) else profile
    prof = prof or PROFILES["registry"]
    src = "registry default" if prof.name == "registry" else f"profile '{prof.name}'"
    return CostLine(
        symbol=spec.symbol,
        spread_ticks=float(prof.spread_ticks.get(spec.symbol, spec.default_spread_ticks)),
        slippage_ticks=float(prof.slippage_ticks.get(spec.symbol, spec.default_slippage_ticks)),
        commission_round_turn=float(prof.commission_round_turn.get(spec.symbol, spec.default_commission_round_turn)),
        tick_size=spec.tick_size, tick_value=spec.tick_value, pip_size=spec.pip_size, source=src,
    )


def apply_cost_line(risk, line: CostLine):
    """RiskConfig copy carrying `line` (spread/slippage in pips via the tick->pip conversion; commission per contract)."""
    return replace(risk, spread_pips=line.spread_pips, slippage_pips=line.slippage_pips,
                   commission_per_trade=0.0, commission_per_contract=line.commission_round_turn)


def compute_tick_cost_ladder(trades: Iterable, line: CostLine, multipliers: Iterable[float] = (1.0, 1.5, 2.0, 3.0)) -> list[dict]:
    """Re-cost the SAME trades with the tick components scaled by each multiplier (relative to costs the run already charged).

    Each trade is assumed to have been charged at multiplier 1.0 of `line`; rung k adds (k-1) x (spread+slippage) x 2 sides x
    tick_value x size. Net profit, profit factor and mean R per rung come back so the 2x stress and the cost ladder are one
    computation from one model rather than two."""
    import numpy as np
    ts = list(trades or [])
    pnl = np.array([float(getattr(t, "pnl", 0.0) or 0.0) for t in ts], dtype=float)
    size = np.array([float(getattr(t, "size", 0.0) or 0.0) for t in ts], dtype=float)
    risk = np.array([float(getattr(t, "initial_risk", 0.0) or 0.0) * float(getattr(t, "size", 0.0) or 0.0) for t in ts], dtype=float)
    per_k = 2.0 * (line.spread_ticks + line.slippage_ticks) * line.tick_value * size  # $ cost of ONE multiple of tick costs
    out = []
    for k in multipliers:
        p = pnl - (float(k) - 1.0) * per_k
        gp, gl = float(p[p > 0].sum()), float(p[p <= 0].sum())
        r = np.where(risk > 0, p / np.where(risk > 0, risk, 1.0), np.nan)
        out.append({
            "multiplier": float(k), "n": int(p.size), "net_profit": float(p.sum()),
            "profit_factor": (gp / abs(gl)) if gl != 0 else (float("inf") if gp > 0 else 0.0),
            "mean_r": float(np.nanmean(r)) if np.isfinite(r).any() else None,
        })
    return out
