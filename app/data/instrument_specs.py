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
    # ROUND-TURN-COMMISSION DEFAULT (2026-09-24): a realistic, deliberately
    # rough estimate of a typical retail futures commission for ONE whole
    # round-turn (one contract, in + out) on this contract -- exchange fee
    # + a typical broker's per-side rate, NOT a specific broker's actual
    # posted rate (those vary and change). Existed to close a real gap
    # found via an external comparison: RiskConfig.commission_per_trade
    # defaults to 0.0, and nothing previously nudged a user toward a
    # nonzero value for a futures instrument -- a strategy that's a loser
    # at any realistic commission can look like a solid winner at $0/trade
    # (see this module's own apply_instrument_spec and
    # app.validation.integrity_check's EXECUTION section, both of which
    # use this field). Confirm against your own broker before trusting it
    # for anything beyond "is my backtest even in the right ballpark."
    default_commission_round_turn: float = 4.20

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# Point value ($ per 1-point move) and tick size for the CME futures
# contracts this app's users most commonly backtest -- see this module's
# own docstring for what pip_size/contract_size mean here, and its
# "NOT an exhaustive/official reference" caveat.
_SPECS: tuple[InstrumentSpec, ...] = (
    InstrumentSpec("ES", "E-mini S&P 500", "CME", pip_size=1.0, contract_size=50.0, tick_size=0.25, tick_value=12.50, default_commission_round_turn=4.20),
    InstrumentSpec("MES", "Micro E-mini S&P 500", "CME", pip_size=1.0, contract_size=5.0, tick_size=0.25, tick_value=1.25, default_commission_round_turn=1.42),
    InstrumentSpec("NQ", "E-mini Nasdaq-100", "CME", pip_size=1.0, contract_size=20.0, tick_size=0.25, tick_value=5.00, default_commission_round_turn=4.20),
    InstrumentSpec("MNQ", "Micro E-mini Nasdaq-100", "CME", pip_size=1.0, contract_size=2.0, tick_size=0.25, tick_value=0.50, default_commission_round_turn=1.42),
    InstrumentSpec("YM", "E-mini Dow", "CBOT", pip_size=1.0, contract_size=5.0, tick_size=1.0, tick_value=5.00, default_commission_round_turn=4.20),
    InstrumentSpec("MYM", "Micro E-mini Dow", "CBOT", pip_size=1.0, contract_size=0.5, tick_size=1.0, tick_value=0.50, default_commission_round_turn=1.42),
    InstrumentSpec("RTY", "E-mini Russell 2000", "CME", pip_size=1.0, contract_size=50.0, tick_size=0.10, tick_value=5.00, default_commission_round_turn=4.20),
    InstrumentSpec("M2K", "Micro E-mini Russell 2000", "CME", pip_size=1.0, contract_size=5.0, tick_size=0.10, tick_value=0.50, default_commission_round_turn=1.42),
    InstrumentSpec("GC", "Gold futures", "COMEX", pip_size=1.0, contract_size=100.0, tick_size=0.10, tick_value=10.00, default_commission_round_turn=5.10),
    InstrumentSpec("MGC", "Micro Gold futures", "COMEX", pip_size=1.0, contract_size=10.0, tick_size=0.10, tick_value=1.00, default_commission_round_turn=1.60),
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
    Every other RiskConfig field (risk_value, max_position_size, ...) is
    left completely untouched, WITH ONE EXCEPTION: commission_per_contract
    is also filled in from the spec's default_commission_round_turn ($ per
    ONE contract round-turn, e.g. MGC = $1.60/contract), but ONLY when the
    caller's current commission_per_contract is still exactly 0.0
    (RiskConfig's own generic default) -- an explicit nonzero value the
    user already typed in (their own broker's real rate) is never
    overwritten. B2-4 (2026-10-04): the spec rate now lands on
    commission_per_contract (charged per contract at settle:
    commission_per_trade + commission_per_contract * contracts) instead of
    the old flat commission_per_trade fill -- the flat fill undercharged
    every multi-contract position (a 30-micro MGC position costs ~$48
    round-turn live, $1.60 in the old sim). commission_per_trade is left
    exactly as the caller set it.

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
    updates = {"pip_size": spec.pip_size, "contract_size": spec.contract_size}
    if risk.commission_per_contract == 0.0:
        updates["commission_per_contract"] = spec.default_commission_round_turn
    return replace(risk, **updates)


def guess_instrument_symbol(label: str | None) -> str | None:
    """Best-effort extraction of a known root symbol (see
    KNOWN_INSTRUMENTS) from a free-form dataset/instrument label such as
    a filename ("futures_ES.F_1m.parquet"), a TradingView-style ticker
    ("ES1!"), or a plain symbol ("MNQ"). Returns None (never raises) when
    nothing in the label matches a known symbol -- callers (e.g.
    app.validation.integrity_check) treat that as "can't tell", not "not
    a futures instrument", and simply skip the checks that need a symbol.

    Deliberately conservative: matches a known symbol only as a whole
    "word" (surrounded by start/end of string, '.', '_', '!', or a digit)
    so it can't mistake, say, "GC" inside an unrelated ticker for gold.
    Longer symbols are checked first so "MES" doesn't get shadowed by a
    same-prefix match against "ES".
    """
    if not label:
        return None
    import re
    text = str(label).upper()
    for symbol in sorted(KNOWN_INSTRUMENTS, key=len, reverse=True):
        if re.search(rf"(?<![A-Z]){re.escape(symbol)}(?![A-Z])", text):
            return symbol
    return None


@dataclass(frozen=True)
class MarketRiskResolution:
    """What resolve_risk_per_market decided for ONE market, in a form both
    the run log and the form's "Detect from data" button can show."""
    label: str
    pip_size: float
    contract_size: float | None
    commission_per_trade: float
    symbol: str | None          # known root symbol matched from the label, or None
    pip_source: str             # "instrument spec" | "detected from price data" | "shared setting"
    note: str = ""

    def describe(self) -> str:
        contract = f", ${self.contract_size:g}/point" if self.contract_size else ", not lot-rounded"
        sym = f" [{self.symbol}]" if self.symbol else ""
        extra = f" -- {self.note}" if self.note else ""
        return (f"{self.label}{sym}: pip size {self.pip_size:g}{contract}, commission "
                f"${self.commission_per_trade:g} ({self.pip_source}){extra}")


def resolve_risk_per_market(
    dfs: dict, base_risk: RiskConfig, auto_detect: bool = True,
) -> tuple[dict, list]:
    """Builds ONE RiskConfig per market instead of stamping a single
    pip_size/contract_size/commission on every instrument.

    Why: a multi-instrument run (ES + GC + MGC, or several FX pairs plus a
    JPY pair) shares exactly one RiskConfig, so at most one of the markets
    ever had the right price scale. The rest got a fixed-pips stop that was
    nonsense for their price level, or a contract size from a different
    contract entirely -- which then either sized positions to 0 contracts or
    made the whole run untrustworthy.

    auto_detect=True, per market:
      1. If the label names a known contract (see KNOWN_INSTRUMENTS), use
         that spec's pip_size / contract_size (and its default commission,
         but only when the shared commission is still exactly 0.0 -- an
         explicit nonzero commission is never overwritten).
      2. Otherwise detect pip_size from the market's own price data
         (app.backtest.risk.suggest_pip_size) and clear contract_size, since
         a contract size typed for a different instrument would be wrong here.
    auto_detect=False keeps `base_risk` for every market unchanged (the old
    behavior), still reported so the log says what was used.

    Returns ({label: RiskConfig}, [MarketRiskResolution, ...]). Never raises
    for a bad frame -- suggest_pip_size falls back to its own default.
    """
    from app.backtest.risk import suggest_pip_size

    risks: dict = {}
    report: list = []
    for label, df in dfs.items():
        if not auto_detect:
            risks[label] = base_risk
            report.append(MarketRiskResolution(
                label, base_risk.pip_size, base_risk.contract_size, base_risk.commission_per_trade,
                None, "shared setting"))
            continue
        symbol = guess_instrument_symbol(label)
        spec = get_instrument_spec(symbol) if symbol else None
        if spec is not None:
            risk = apply_instrument_spec(base_risk, spec.symbol)
            report.append(MarketRiskResolution(
                label, risk.pip_size, risk.contract_size, risk.commission_per_trade,
                spec.symbol, "instrument spec"))
        else:
            detected = suggest_pip_size(df)
            risk = replace(base_risk, pip_size=detected, contract_size=None)
            report.append(MarketRiskResolution(
                label, detected, None, risk.commission_per_trade, None, "detected from price data",
                note="no known contract in the name, so positions are not lot-rounded"))
        risks[label] = risk
    return risks, report
