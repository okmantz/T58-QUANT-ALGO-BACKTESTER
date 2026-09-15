"""
Risk & execution configuration and position sizing.

Position sizing is expressed in generic "units" rather than broker-specific
lots: pnl = units * price_move. This keeps the engine instrument-agnostic
(FX, indices, crypto, etc.) while still allowing pip-based stop/target
distances via `pip_size`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    initial_balance: float = 10_000.0
    risk_mode: str = "percent"          # "percent" | "fixed"
    risk_value: float = 1.0             # % of equity, or fixed $ amount, per trade
    max_trades_per_day: int = 10
    commission_per_trade: float = 0.0   # flat $ per round-turn trade
    slippage_pips: float = 0.0
    spread_pips: float = 0.0
    pip_size: float = 0.0001            # price move that equals "1 pip" (e.g. 0.0001 for EURUSD)
    max_position_size: float | None = None  # cap on units, None = unlimited
    daily_loss_limit_pct: float | None = None  # % of initial_balance; once a day's REALIZED
    # pnl breaches -this, no new entries are taken for the rest of that calendar day.
    # None = disabled (no circuit breaker; this was the only behavior before this field
    # existed). This is the correct, supported way to give a strategy a daily-loss cutoff --
    # a strategy's own generate_signals() cannot implement this itself (see app/strategy/
    # python.py) because it never sees realized trade outcomes, only price data.
    reentry_cooldown_bars: int = 0
    # FIX (EXEC-002): after an open position is stopped out (or hits its
    # take-profit) intrabar, app.backtest.execution used to let a brand
    # new position open in the SAME direction on that exact same bar's
    # close, with no cooldown at all, as long as the strategy's signal
    # hadn't gone flat -- i.e. a whipsaw bar that clips a tight stop and
    # then closes back at a level the strategy still signals on produced
    # TWO trades and re-exposed the same risk within one bar. That
    # behavior is still the default (0 = no cooldown, byte-for-byte the
    # original behavior, so nothing changes for any existing caller/
    # saved config unless this is explicitly set), but it is now an
    # explicit, documented, configurable choice instead of a silent one.
    # Setting this to N>0 blocks any NEW entry for N bars (measured from
    # the bar the previous position closed on, inclusive) regardless of
    # what the strategy's signal says -- see app.backtest.execution's
    # module docstring for the exact accounting.

    # --- account-survivability hard caps ---------------------------------
    # A real prop/broker account has negative-balance protection and a hard
    # firm-level loss floor: no single trade can ever cost more than the
    # money actually in the account, and once the firm's max-drawdown floor
    # is breached the account is terminated -- it cannot keep incurring
    # further losses. Before these fields existed, a pip_size/instrument
    # mismatch or an honest gap-through fill could size a trade so far off
    # that a single loss exceeded the ENTIRE account (real observed cases:
    # -$50,000 and -$2,534,176 single-run net losses on nominal $50k
    # accounts), which cannot happen in live trading and silently poisoned
    # every downstream risk-of-ruin / eval-pass-probability number. These
    # two caps make that structurally impossible instead of just warning
    # about it after the fact.
    max_loss_per_trade_pct: float | None = None
    # Hard ceiling on any single trade's realized loss, as a % of
    # initial_balance. None (default) falls back to 3x the trade's own
    # intended risk_amount -- generous enough to still show a real gap-
    # through loss as materially worse than a normal stop-out, but bounded.
    max_account_drawdown_pct: float | None = None
    # % of initial_balance the account may lose (from its starting balance)
    # before the engine marks it BLOWN and stops opening new trades for the
    # rest of the run -- mirrors a prop firm terminating a failed account
    # rather than letting the simulation keep "trading" a dead account into
    # deeper and deeper negative equity. None = disabled (no such floor is
    # enforced beyond the per-trade cap above). Typically set from the
    # active PropRules.max_drawdown_pct so the raw backtest and the prop
    # simulation agree on where the account actually dies.

    def risk_amount(self, current_equity: float) -> float:
        # Floor equity at 0 for sizing purposes: a negative-equity account
        # is already blown (see max_account_drawdown_pct / account_blown
        # handling in app.backtest.execution) and must never be sized as if
        # it had negative money to risk, which previously flipped the sign
        # of "%-of-equity" sizing and could make losses compound instead of
        # halting.
        equity_for_sizing = max(current_equity, 0.0)
        if self.risk_mode == "fixed":
            return max(self.risk_value, 0.0)
        return max(equity_for_sizing * (self.risk_value / 100.0), 0.0)

    def position_size(self, current_equity: float, stop_loss_pips: float) -> float:
        """Units such that a full stop-out loses exactly `risk_amount`."""
        if not stop_loss_pips or stop_loss_pips <= 0:
            stop_loss_pips = 10.0  # sane fallback so sizing never divides by zero
        stop_distance = stop_loss_pips * self.pip_size
        risk_amt = self.risk_amount(current_equity)
        units = risk_amt / stop_distance if stop_distance > 0 else 0.0
        if self.max_position_size is not None:
            units = min(units, self.max_position_size)
        return max(units, 0.0)

    def max_trade_loss(self, equity_at_entry: float) -> float:
        """Hard dollar ceiling on how much a single trade may realistically
        lose, regardless of how it was sized or how far price gapped past
        its stop. Real prop firms/brokers cap the damage one trade can do
        (negative-balance protection, firm-level daily/overall loss
        floors) -- a simulated trade should never be able to blow past
        that on its own."""
        if self.max_loss_per_trade_pct is not None:
            return max(self.initial_balance * (self.max_loss_per_trade_pct / 100.0), 0.0)
        return self.risk_amount(equity_at_entry) * 3.0

    def account_blown_floor(self) -> float | None:
        """Equity level at/below which the account is BLOWN and must stop
        opening new trades. None if no such floor is configured."""
        if self.max_account_drawdown_pct is None:
            return None
        return self.initial_balance * (1.0 - self.max_account_drawdown_pct / 100.0)


def with_prop_safety_defaults(risk: "RiskConfig", prop_rules) -> "RiskConfig":
    """Returns a copy of `risk` with max_account_drawdown_pct, AND
    daily_loss_limit_pct filled in from `prop_rules` whenever the caller
    hasn't already set an explicit value of their own for that field, AND
    initial_balance forced to match `prop_rules.account_size`. This is
    what makes the account-blown circuit breaker AND the daily-loss
    circuit breaker (see app.backtest.execution) apply automatically
    during the RAW BACKTEST itself, against the SAME dollar account the
    post-hoc prop simulator (app.prop.simulator.simulate_account) checks
    the finished trade sequence against -- not just later, and not
    against a different balance. Never overrides a value the caller
    explicitly configured on either PERCENTAGE field (max_account_
    drawdown_pct / daily_loss_limit_pct); initial_balance is the one
    exception -- see the RISK-001 fix note below for why.

    FIX (2026-09-12): daily_loss_limit_pct used to be left out of this
    function entirely -- only max_account_drawdown_pct was wired through.
    That meant a strategy could freely keep opening new trades on a day
    that had already blown through the prop firm's daily loss limit
    during its OWN raw backtest (the stage every search/optimization loop
    scores fitness from), with the violation only surfacing later at the
    Monte Carlo / CPCV / prop-simulation stage -- wasting compute
    exploring candidates whose raw backtest was never a realistic
    account to begin with, and understating how early a real account
    would have been forced to stop trading that day. This is exactly the
    'hard, declarative prop-firm-realistic constraint layer at strategy-
    discovery time, not just at simulation time' this app was missing:
    every caller of this function (Speed Run, Full Pipeline, and now
    Evolution Lab -- see app.evolution.engine.EvolutionRunner.__init__)
    gets the fix automatically, with no other code path needing to
    change.

    FIX (RISK-001): `daily_loss_limit_pct` and `max_account_drawdown_pct`
    are PERCENTAGES, and they used to get applied against two different
    dollar bases depending on which layer checked them -- the raw
    backtest's intrabar circuit breaker used `risk.initial_balance`,
    while the post-hoc prop-firm verdict (simulate_account) used
    `prop_rules.account_size`. Both values default to different numbers
    (10,000 vs 100,000) and, worse, both were independently user-editable
    (two separate "Initial balance ($)" / "Account size ($)" fields in
    both the desktop and web UI, with no sync between them), so a
    strategy could cleanly survive the raw-backtest circuit breaker
    (checked against the wrong, too-generous dollar floor) and then be
    silently re-scored against a completely different floor at the final
    verdict, or vice versa. Since PropRules IS the definition of the
    account actually being evaluated, `prop_rules.account_size` is now
    treated as authoritative and always wins here -- this function is
    the one policy chokepoint already shared by every pipeline that
    matters for a research verdict, so fixing it here fixes the
    divergence everywhere this function is already called. Callers that
    construct a RiskConfig/PropRules pair directly (rather than through
    a pipeline that calls this function) should call
    account_size_mismatch_message() themselves first if they want to
    warn the person BEFORE the values get silently reconciled -- see its
    docstring."""
    from dataclasses import replace
    updates: dict = {}
    if risk.max_account_drawdown_pct is None:
        max_dd = getattr(prop_rules, "max_drawdown_pct", None)
        if max_dd is not None:
            updates["max_account_drawdown_pct"] = max_dd
    if risk.daily_loss_limit_pct is None:
        daily_loss = getattr(prop_rules, "daily_loss_limit_pct", None)
        if daily_loss is not None:
            updates["daily_loss_limit_pct"] = daily_loss
    account_size = getattr(prop_rules, "account_size", None)
    if account_size is not None and risk.initial_balance != account_size:
        updates["initial_balance"] = account_size
    if not updates:
        return risk
    return replace(risk, **updates)


def account_size_mismatch_message(initial_balance: float, account_size: float) -> str | None:
    """RISK-001: None if `initial_balance` (RiskConfig, drives position
    sizing and the raw backtest's own intrabar circuit breakers) and
    `account_size` (PropRules, what the final prop-firm verdict is
    actually computed against) already agree -- otherwise an actionable
    message explaining the mismatch and that with_prop_safety_defaults()
    will make `account_size` win. Callers that build a RiskConfig and a
    PropRules directly (web routes, the desktop UI, any script) should
    call this BEFORE calling with_prop_safety_defaults so the person
    sees why their numbers just changed, the same way
    instrument_scale_mismatch_message() is surfaced before its own
    silent-but-safe correction."""
    if initial_balance == account_size:
        return None
    return (
        f"Account-size mismatch: Risk config's 'Initial balance' (${initial_balance:,.2f}) does not "
        f"match the prop rules' 'Account size' (${account_size:,.2f}). These must be the same dollar "
        "account -- position sizing, the raw backtest's daily-loss/max-drawdown circuit breakers, AND "
        "the final prop-firm pass/fail verdict all need to agree on how big the account actually is, "
        "or a strategy can silently survive one stage's check and fail (or pass) the other's, on the "
        "exact same trades, purely because they used different dollar floors for the same percentage "
        f"rule. Using the prop rules' account size (${account_size:,.2f}) for this run -- update the "
        "'Initial balance' field to match if that wasn't intended."
    )


_ACCOUNT_SIZE_MISMATCH_MARKER = "Account-size mismatch:"


def has_account_size_mismatch(warnings: "list[str]") -> bool:
    """True if any warning in `warnings` is account_size_mismatch_message's
    warning -- same shared-detection pattern as has_instrument_scale_mismatch
    and has_impossible_condition, so every caller escalates this the same way."""
    return any(_ACCOUNT_SIZE_MISMATCH_MARKER in w for w in warnings)


def suggest_pip_size(df) -> float:
    """Suggests a starting pip_size from a loaded OHLCV DataFrame's actual
    price scale, purely by magnitude of the median close price. This is a
    starting point for the person to confirm, not an authoritative
    per-instrument lookup (it can't distinguish gold from a $2,000 index,
    for instance) -- it exists because leaving pip_size at its FX default
    (0.0001) against a non-FX-scaled instrument (stocks, indices, crypto,
    JPY pairs) is the single most common cause of a strategy's fixed-pips
    stop translating into a nonsensical position size (see the
    pip_scale_mismatch warning in app.backtest.execution).

    Rough bands, all "1 pip = smallest meaningful price increment" for
    that price level:
      >= 500          -> 1.0    (large-index / high-priced-crypto scale)
      >= 20            -> 0.01   (typical stock-in-dollars or JPY-pair scale)
      >= 5             -> 0.01   (lower-priced stocks; still cent-scale)
      < 5              -> 0.0001 (FX-major scale, e.g. EURUSD ~1.10)
    """
    if df is None or "close" not in getattr(df, "columns", []) or len(df) == 0:
        return 0.0001
    median_price = float(df["close"].abs().median())
    if not median_price or median_price != median_price:  # NaN guard
        return 0.0001
    if median_price >= 500:
        return 1.0
    if median_price >= 5:
        return 0.01
    return 0.0001


# The two exact substrings app.backtest.execution's run_backtest emits
# when a fixed-pips stop is an implausible fraction of either the
# instrument's raw price (pip_scale_mismatch) or its own recent ATR
# (atr_scale_mismatch) -- see that module for the full warning text.
# Both are checked (not just the price-ratio one) because a fixed-pips
# stop can pass the price-ratio check -- look like a perfectly ordinary
# fraction of price -- while still being tiny next to the instrument's
# own actual volatility; a high-priced but volatile instrument such as
# an equity index is the case that price-ratio alone misses.
_INSTRUMENT_MISMATCH_MARKERS = (
    "doesn't match the instrument actually being tested",
    "under 15% of this instrument's own recent ATR",
)


def has_instrument_scale_mismatch(warnings: "list[str]") -> bool:
    """True if any warning in `warnings` (e.g. a BacktestResult.warnings
    list) is app.backtest.execution's pip_scale_mismatch or
    atr_scale_mismatch warning -- the single most common cause of a
    strategy's numbers being unreliable (see suggest_pip_size above).
    Shared by app.orchestration.full_pipeline (which skips its GA search
    on this) and app.orchestration.quick_optimize (which surfaces it
    prominently instead) so both tools agree on exactly what counts."""
    return any(marker in w for w in warnings for marker in _INSTRUMENT_MISMATCH_MARKERS)


def has_impossible_condition(warnings: "list[str]") -> bool:
    """True if any warning in `warnings` is app.strategy.manual's
    validate_bounded_conditions warning -- a condition comparing a
    bounded oscillator (RSI, Stochastic, MFI, ...) to a threshold outside
    its possible range, which can never be satisfied and permanently
    disables that branch of the strategy's logic. Shared by Quick
    Optimize and Full Pipeline so both escalate it the same prominent
    way they already escalate has_instrument_scale_mismatch."""
    from app.strategy.manual import IMPOSSIBLE_CONDITION_MARKER
    return any(IMPOSSIBLE_CONDITION_MARKER in w for w in warnings)


def instrument_scale_mismatch_message(pip_size: float) -> str:
    """Shared, actionable explanation shown wherever
    has_instrument_scale_mismatch() is True -- one copy of the wording so
    Full Pipeline and Quick Optimize never drift into saying two
    different things about the same problem."""
    return (
        f"Pip-size/instrument-scale mismatch detected (current pip_size: {pip_size}). Every position "
        "size and stop distance this run computed is unreliable -- this almost always means "
        "risk.pip_size doesn't match the instrument actually being tested (e.g. an FX-calibrated "
        "0.0001 run against gold, an index, crypto, or a JPY pair). Set pip_size to match the real "
        "instrument (e.g. 0.01 for gold/JPY pairs, 1.0 for high-priced indices/stocks -- see "
        "suggest_pip_size, or click \"DETECT PIP SIZE FROM DATA\") before trusting this result."
    )
