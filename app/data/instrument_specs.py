"""
Instrument metadata: point value / tick size for common futures contracts.

UPGRADE (instrument-metadata-belongs-with-the-data, not hand-derived per
RiskConfig): app.backtest.risk.RiskConfig has no concept of "which
instrument is this" -- only the two raw numbers app.backtest.execution's
position sizing actually reads:

    pip_size       -- the smallest meaningful price increment this engine
                      sizes stops/targets against (see RiskConfig.pip_size
                      and app.backtest.risk.suggest_pip_size).
    contract_size  -- how many "units" make up ONE real, whole contract
                      (see RiskConfig.contract_size's own docstring for
                      this engine's "1 unit = $1 per pip_size move"
                      convention). With pip_size=1.0, contract_size IS a
                      contract's dollar-per-point value -- e.g. NQ's
                      contract_size=20.0 literally means "$20/point".

Before this module existed, applying that convention to a REAL instrument
meant hand-deriving and hardcoding pip_size/contract_size, per instrument,
in every RiskConfig a user built by hand -- exactly the "I had to
hand-derive and hardcode MES=$5, MNQ=$2, MGC=$10 myself" complaint this
closes. This is a plain data module (no engine dependency, safe to import
from anywhere) so that metadata can live in ONE place, alongside the
instrument's own market data, instead of buried inside every config.

NOT an exhaustive or officially-maintained reference -- contract specs do
occasionally change (tick sizes, in particular, are set by the exchange
and have been revised before). Confirm against your broker/exchange
before trusting real size against it; this exists to remove the tedious,
error-prone part (looking these up and hand-deriving pip_size/
contract_size) for the handful of contracts this app's users most
commonly backtest, not to be the last word on any of them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from app.backtest.risk import RiskConfig


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    description: str
    exchange: str
    pip_size: float        # see RiskConfig.pip_size -- 1.0 for every instrument below (index/metal points)
    contract_size: float   # see RiskConfig.contract_size -- $ per 1.0-pip_size move for ONE whole contract
    tick_size: float        # exchange's real minimum price increment (may be finer than pip_size)
    tick_value: float       # $ value of one tick_size move, for ONE whole contract

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# Point value ($ per 1-point move) and tick size for the CME futures
# contracts this app's users most commonly backtest -- see this module's
# own docstring for what pip_size/contract_size mean here, and its
# "NOT an exhaustive/official reference" caveat.
_SPECS: tuple[InstrumentSpec, ...] = (
    InstrumentSpec("ES", "E-mini S&P 500", "CME", pip_size=1.0, contract_size=50.0, tick_size=0.25, tick_value=12.50),
    InstrumentSpec("MES", "Micro E-mini S&P 500", "CME", pip_size=1.0, contract_size=5.0, tick_size=0.25, tick_value=1.25),
    InstrumentSpec("NQ", "E-mini Nasdaq-100", "CME", pip_size=1.0, contract_size=20.0, tick_size=0.25, tick_value=5.00),
    InstrumentSpec("MNQ", "Micro E-mini Nasdaq-100", "CME", pip_size=1.0, contract_size=2.0, tick_size=0.25, tick_value=0.50),
    InstrumentSpec("YM", "E-mini Dow", "CBOT", pip_size=1.0, contract_size=5.0, tick_size=1.0, tick_value=5.00),
    InstrumentSpec("MYM", "Micro E-mini Dow", "CBOT", pip_size=1.0, contract_size=0.5, tick_size=1.0, tick_value=0.50),
    InstrumentSpec("RTY", "E-mini Russell 2000", "CME", pip_size=1.0, contract_size=50.0, tick_size=0.10, tick_value=5.00),
    InstrumentSpec("M2K", "Micro E-mini Russell 2000", "CME", pip_size=1.0, contract_size=5.0, tick_size=0.10, tick_value=0.50),
    InstrumentSpec("GC", "Gold futures", "COMEX", pip_size=1.0, contract_size=100.0, tick_size=0.10, tick_value=10.00),
    InstrumentSpec("MGC", "Micro Gold futures", "COMEX", pip_size=1.0, contract_size=10.0, tick_size=0.10, tick_value=1.00),
)

KNOWN_INSTRUMENTS: dict[str, InstrumentSpec] = {spec.symbol: spec for spec in _SPECS}


def known_instrument_symbols() -> list[str]:
    """Sorted symbols this module has a spec for -- e.g. for populating a
    UI dropdown or a helpful error message (see apply_instrument_spec)."""
    return sorted(KNOWN_INSTRUMENTS.keys())


def get_instrument_spec(symbol: str) -> InstrumentSpec | None:
    """Case-insensitive lookup by root symbol -- e.g. "mnq" or "MNQ" both
    match. Does not strip a trailing contract-month code (e.g. "MNQZ25");
    pass the bare root symbol. Returns None (never raises) when `symbol`
    isn't in the registry -- see apply_instrument_spec for the raising
    variant."""
    if not symbol:
        return None
    return KNOWN_INSTRUMENTS.get(str(symbol).strip().upper())


def apply_instrument_spec(risk: RiskConfig, symbol: str) -> RiskConfig:
    """Returns a copy of `risk` with pip_size/contract_size set from
    KNOWN_INSTRUMENTS[symbol] -- the exact two fields a user would
    otherwise have to hand-derive (see this module's own docstring).
    Every other RiskConfig field (risk_value, max_position_size,
    commission, ...) is left completely untouched.

    Raises KeyError (naming the known symbols) for a symbol not in the
    registry, rather than silently leaving `risk` unchanged -- a caller
    that mistypes a symbol should find out immediately, not ship a
    backtest still running on RiskConfig's own FX-scaled defaults."""
    spec = get_instrument_spec(symbol)
    if spec is None:
        raise KeyError(
            f"No known instrument spec for '{symbol}'. Known symbols: "
            f"{', '.join(known_instrument_symbols())}. Set RiskConfig.pip_size/"
            "contract_size directly for anything not in this list, or add a new "
            "InstrumentSpec to app.data.instrument_specs.KNOWN_INSTRUMENTS."
        )
    return replace(risk, pip_size=spec.pip_size, contract_size=spec.contract_size)
