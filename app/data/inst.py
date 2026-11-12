"""ONE cost table for every instrument (2026-11-11): the single source of truth for commission, tick and slippage.

Before this module the scanner (``opportunity_scanner.INSTRUMENTS``), the instrument registry (``instrument_specs``) and the research
helpers each carried their own friction numbers, and the scanner's were ~40% lighter than the registry's. Now each of them DERIVES its
numbers from ``INST`` below, so a commission change is made in exactly one place.

Per instrument (all per ONE contract of that symbol):
    pv          $ per 1.0 price unit (point)
    tick        minimum price increment
    comm_usd    ROUND-TURN all-in commission+fees in $ (broker + exchange + NFA), Alpha Futures schedule
    slip_ticks  stop-out (market order) slippage in ticks; a stop that gaps THROUGH its level fills at the open as well, on top of this
    entry_slip_ticks   market-entry slippage in ticks
    target_slip_ticks  resting limit-target slippage in ticks (a limit fills at its price: 0)
Slippage is conditional on the order type: entry 1 tick, stop-out 1 tick (+ gap-through), target 0. Commission is charged on every trade.

Commissions: Alpha Futures "Permitted Products and Fees" (NFA $0.02/side on every instrument; $0.99/side commission on minis, $0.29/side
on micros; the exchange fee varies by contract). ``ALPHA_COMMISSION_RT`` holds the whole schedule; ``INST`` holds the instruments the
backtester has price specs for. Other firms differ (e.g. Topstep ES/NQ $3.78, MES/MNQ $1.22, GC $4.32; Apex ES/NQ $3.10 Tradovate /
$3.98 Rithmic) -- use ``comm_override`` / ``with_commission`` to model them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# Round-trip commission per contract, Alpha Futures full schedule ($, NFA + commission + exchange, both sides).
ALPHA_COMMISSION_RT: dict[str, float] = {
    # CME equity index
    "ES": 4.78, "NQ": 4.78, "RTY": 4.78, "MES": 1.32, "MNQ": 1.32, "M2K": 1.32, "NKD": 6.32, "MBT": 2.92, "MET": 0.82,
    # NYMEX
    "CL": 5.02, "MCL": 1.62, "QM": 4.42, "PL": 5.22, "QG": 3.02, "RB": 5.02, "HO": 5.02, "NG": 5.18, "MNG": 1.82,
    # CBOT equity index
    "YM": 4.78, "MYM": 1.32,
    # COMEX
    "GC": 5.22, "MGC": 1.82, "SI": 5.22, "SIL": 2.62, "HG": 5.22, "MHG": 1.82,
    # FX
    "6A": 5.22, "6B": 5.22, "6C": 5.22, "6E": 5.22, "6J": 5.22, "6S": 5.22, "6M": 5.22, "6N": 5.22,
    "M6A": 1.10, "M6E": 1.10, "M6B": 1.10, "E7": 3.72,
    # rates
    "ZT": 3.32, "ZF": 3.32, "ZN": 3.58, "TN": 3.62, "ZB": 3.76, "UB": 3.92,
    # ags
    "HE": 6.22, "LE": 6.22, "ZC": 6.28, "ZW": 6.28, "ZS": 6.28, "ZM": 6.28, "ZL": 6.28,
}

# Other firms' round-trip schedules for the contracts the backtester prices (comparison / what-if only).
OTHER_FIRM_COMMISSION_RT: dict[str, dict[str, float]] = {
    "topstep": {"ES": 3.78, "NQ": 3.78, "MES": 1.22, "MNQ": 1.22, "GC": 4.32},
    "apex_tradovate": {"ES": 3.10, "NQ": 3.10},
    "apex_rithmic": {"ES": 3.98, "NQ": 3.98},
}


@dataclass(frozen=True)
class Inst:
    symbol: str
    pv: float                      # $ per 1.0 price unit, ONE contract
    tick: float
    comm_usd: float                # round-turn commission per contract
    slip_ticks: float = 1.0        # stop-out (market) slippage
    entry_slip_ticks: float = 1.0  # market entry slippage
    target_slip_ticks: float = 0.0 # limit target slippage
    micro: str | None = None       # the micro of this mini (None for a micro / no micro)
    mini: str | None = None        # the mini of this micro

    @property
    def tick_value(self) -> float:
        return self.pv * self.tick

    @property
    def comm_points(self) -> float:
        """Round-turn commission expressed in price points (per contract)."""
        return self.comm_usd / self.pv

    @property
    def slip_points(self) -> float:
        return self.slip_ticks * self.tick

    @property
    def entry_slip_points(self) -> float:
        return self.entry_slip_ticks * self.tick

    @property
    def target_slip_points(self) -> float:
        return self.target_slip_ticks * self.tick

    def with_commission(self, comm_usd: float) -> "Inst":
        return replace(self, comm_usd=float(comm_usd))

    def with_slip(self, ticks: float) -> "Inst":
        """Same instrument with market-order slippage (entry AND stop-out) set to ``ticks``; limit targets stay at 0."""
        return replace(self, slip_ticks=float(ticks), entry_slip_ticks=float(ticks))


def _mk(sym, pv, tick, micro=None, mini=None) -> Inst:
    return Inst(sym, float(pv), float(tick), ALPHA_COMMISSION_RT[sym], micro=micro, mini=mini)


INST: dict[str, Inst] = {i.symbol: i for i in (
    _mk("ES", 50.0, 0.25, micro="MES"), _mk("MES", 5.0, 0.25, mini="ES"),
    _mk("NQ", 20.0, 0.25, micro="MNQ"), _mk("MNQ", 2.0, 0.25, mini="NQ"),
    _mk("YM", 5.0, 1.0, micro="MYM"), _mk("MYM", 0.5, 1.0, mini="YM"),
    _mk("RTY", 50.0, 0.10, micro="M2K"), _mk("M2K", 5.0, 0.10, mini="RTY"),
    _mk("GC", 100.0, 0.10, micro="MGC"), _mk("MGC", 10.0, 0.10, mini="GC"),
    _mk("CL", 1000.0, 0.01, micro="MCL"), _mk("MCL", 100.0, 0.01, mini="CL"),
    _mk("ZN", 1000.0, 0.015625), _mk("ZB", 1000.0, 0.03125),
    _mk("6E", 125_000.0, 0.00005),
)}


def get_inst(symbol: str, *, comm_override: float | None = None, slip_ticks: float | None = None) -> Inst:
    """Look up ``symbol`` (KeyError if unknown), optionally overriding commission ($ round turn) and/or market slippage (ticks)."""
    i = INST[str(symbol).upper()]
    if comm_override is not None:
        i = i.with_commission(comm_override)
    if slip_ticks is not None:
        i = i.with_slip(slip_ticks)
    return i
