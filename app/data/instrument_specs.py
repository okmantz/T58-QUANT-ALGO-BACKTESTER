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
    # SPREAD / SLIPPAGE DEFAULTS (2026-10-04): typical inside spread and
    # market-order slippage for this contract, in TICKS (tick_size units,
    # not pip_size units). Same honesty standard as
    # default_commission_round_turn above: these are typical-condition
    # estimates for liquid front-month contracts, NOT your broker's actual
    # fills -- every contract in this registry trades with a typical
    # inside spread of 1 tick, and a market order that doesn't sweep the
    # book slips ~1 tick on average. apply_instrument_spec fills
    # RiskConfig.spread_pips/slippage_pips from these (ticks == pips for
    # every spec here since pip_size is 1.0), but ONLY when the caller's
    # values are still exactly 0.0 (RiskConfig's own generic default) --
    # an explicit nonzero value is never overwritten. The ZERO FRICTION
    # warning in app.backtest.execution stays as the backstop: it still
    # fires whenever all four friction fields end up 0.0 (e.g. a
    # hand-built RiskConfig that never applied a spec).
    default_spread_ticks: float = 1.0
    default_slippage_ticks: float = 1.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def ticks_to_pips(self, ticks: float) -> float:
        """Converts a tick count into RiskConfig pip units.

        ACCURACY FIX (2026-10-07, Full Pipeline Audit "cost units"):
        spread/slippage defaults are denominated in TICKS (tick_size price
        units), but RiskConfig.spread_pips/slippage_pips are denominated in
        PIPS (pip_size price units). The old code copied the tick count
        straight into the pip field, which is only correct when
        tick_size == pip_size. Every spec in this registry has
        pip_size == 1.0 while tick_size is 0.25 (ES/NQ), 0.10 (RTY/GC) or
        1.0 (YM), so ES was charged 1.0 point per tick instead of 0.25 --
        4x too high per component, ~3x too high on the combined
        spread+slippage the audit measured against the intended cost.
        price = ticks x tick_size, pips = price / pip_size."""
        return float(ticks) * float(self.tick_size) / float(self.pip_size)

    @property
    def default_spread_pips(self) -> float:
        return self.ticks_to_pips(self.default_spread_ticks)

    @property
    def default_slippage_pips(self) -> float:
        return self.ticks_to_pips(self.default_slippage_ticks)

    def round_trip_cost_dollars(
        self, contracts: float = 1.0, spread_ticks: float | None = None,
        slippage_ticks: float | None = None, commission_round_turn: float | None = None,
    ) -> float:
        """Round-trip dollar cost of `contracts` whole contracts under the
        engine's cost model (spread + slippage are charged on BOTH the
        entry and the exit fill, plus commission once per round turn):

            2 x (spread_ticks + slippage_ticks) x tick_value x contracts
              + commission_round_turn x contracts

        e.g. ES with the registry defaults (1 tick + 1 tick, $4.20):
        2 x 2 x $12.50 + $4.20 = $54.20 per contract."""
        s = self.default_spread_ticks if spread_ticks is None else spread_ticks
        sl = self.default_slippage_ticks if slippage_ticks is None else slippage_ticks
        c = self.default_commission_round_turn if commission_round_turn is None else commission_round_turn
        return (2.0 * (float(s) + float(sl)) * self.tick_value + float(c)) * float(contracts)


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

# dataset-folder / vendor labels that do not contain the root symbol itself
_ALIASES: dict[str, str] = {"NASDAQ 100": "NAS100", "USATECHIDXUSD": "NAS100", "NAS100_USD": "NAS100",
                            "BITCOIN": "BTCUSD", "XAUUSD (GOLD)": "XAUUSD", "GOLD SPOT": "XAUUSD"}

KNOWN_INSTRUMENTS: dict[str, InstrumentSpec] = {spec.symbol: spec for spec in _SPECS}

# RETAIL SPOT/CFD instruments for cross-market tests (added 2026-10-07). Kept OUT of KNOWN_INSTRUMENTS on
# purpose: that registry means "known FUTURES contract" to the integrity check ($0-commission block,
# whole-contract warning) and to guess_instrument_symbol callers. Use the *_any_* helpers below to include these.
_CROSS_MARKET_SPECS: tuple[InstrumentSpec, ...] = (
    # One "contract" here is a small, divisible unit so whole-contract sizing is not absurdly coarse:
    # FX = 0.1 lot (10,000 units), gold = 0.1 lot (10 oz), NAS100 = 1 index unit, BTC = 0.1 coin.
    # Costs are TYPICAL retail ECN-style figures (spread/slippage in ticks, commission per round turn per
    # contract), not any broker's actual quotes -- replace with your broker's before trusting small edges.
    InstrumentSpec("EURUSD", "Euro / US dollar spot", "FX", pip_size=0.0001, contract_size=10_000.0, tick_size=0.00001, tick_value=0.10,
                   default_commission_round_turn=0.70, default_spread_ticks=6.0, default_slippage_ticks=2.0),
    InstrumentSpec("GBPUSD", "British pound / US dollar spot", "FX", pip_size=0.0001, contract_size=10_000.0, tick_size=0.00001, tick_value=0.10,
                   default_commission_round_turn=0.70, default_spread_ticks=10.0, default_slippage_ticks=3.0),
    InstrumentSpec("XAUUSD", "Gold spot (10 oz contract)", "OTC", pip_size=0.10, contract_size=10.0, tick_size=0.01, tick_value=0.10,
                   default_commission_round_turn=0.70, default_spread_ticks=25.0, default_slippage_ticks=10.0),
    InstrumentSpec("NAS100", "Nasdaq-100 cash index CFD (1 unit)", "OTC", pip_size=1.0, contract_size=1.0, tick_size=0.10, tick_value=0.10,
                   default_commission_round_turn=0.0, default_spread_ticks=10.0, default_slippage_ticks=5.0),
    InstrumentSpec("BTCUSD", "Bitcoin / US dollar spot (0.1 coin contract)", "CRYPTO", pip_size=1.0, contract_size=0.1, tick_size=1.0, tick_value=0.10,
                   default_commission_round_turn=0.0, default_spread_ticks=15.0, default_slippage_ticks=10.0),
)
CROSS_MARKET_INSTRUMENTS: dict[str, InstrumentSpec] = {spec.symbol: spec for spec in _CROSS_MARKET_SPECS}

# Full-size contract -> its micro equivalent. Used by RiskConfig's
# "micro_fallback" sizing mode: when a full-size contract cannot be sized
# inside the risk budget, trade the micro (1/10th the point value) instead
# of either skipping the signal or oversizing the risk.
MICRO_EQUIVALENT: dict[str, str] = {
    "ES": "MES", "NQ": "MNQ", "YM": "MYM", "RTY": "M2K", "GC": "MGC",
}


def micro_equivalent(symbol: str | None) -> "InstrumentSpec | None":
    """The micro contract spec for a full-size symbol, or None when the
    symbol has no micro in this registry (or is already a micro)."""
    if not symbol:
        return None
    micro = MICRO_EQUIVALENT.get(str(symbol).strip().upper())
    return KNOWN_INSTRUMENTS.get(micro) if micro else None


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
    # C5 + ACCURACY FIX (2026-10-07): the spec's spread/slippage defaults
    # are TICKS; RiskConfig's fields are PIPS. Convert through the tick
    # size (see InstrumentSpec.ticks_to_pips) instead of copying the tick
    # count across -- ES was being charged 4x its real per-tick cost.
    # Same only-when-still-0.0 rule as the commission fill above: an
    # explicit nonzero value the user already typed in is never
    # overwritten.
    if risk.spread_pips == 0.0:
        updates["spread_pips"] = spec.default_spread_pips
    if risk.slippage_pips == 0.0:
        updates["slippage_pips"] = spec.default_slippage_pips
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


# ---- helpers that ALSO see the retail spot/CFD instruments (cross-market tests, accuracy form, preflight) ----

def get_any_instrument_spec(symbol: str) -> InstrumentSpec | None:
    return get_instrument_spec(symbol) or CROSS_MARKET_INSTRUMENTS.get(str(symbol or "").strip().upper())


def guess_any_instrument_symbol(label: str | None) -> str | None:
    """guess_instrument_symbol, then aliases and the retail spot/CFD symbols."""
    sym = guess_instrument_symbol(label)
    if sym:
        return sym
    if not label:
        return None
    import re
    text = str(label).upper()
    for alias, symbol in _ALIASES.items():
        if alias in text:
            return symbol
    for symbol in sorted(CROSS_MARKET_INSTRUMENTS, key=len, reverse=True):
        if re.search(rf"(?<![A-Z]){re.escape(symbol)}(?![A-Z])", text):
            return symbol
    return None


def apply_any_instrument_spec(risk: RiskConfig, symbol: str) -> RiskConfig:
    """apply_instrument_spec that also accepts the retail spot/CFD symbols."""
    if symbol and str(symbol).strip().upper() in CROSS_MARKET_INSTRUMENTS:
        spec = CROSS_MARKET_INSTRUMENTS[str(symbol).strip().upper()]
        updates = {"pip_size": spec.pip_size, "contract_size": spec.contract_size}
        if risk.commission_per_contract == 0.0:
            updates["commission_per_contract"] = spec.default_commission_round_turn
        if risk.spread_pips == 0.0:
            updates["spread_pips"] = spec.default_spread_pips
        if risk.slippage_pips == 0.0:
            updates["slippage_pips"] = spec.default_slippage_pips
        return replace(risk, **updates)
    return apply_instrument_spec(risk, symbol)
