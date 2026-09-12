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
    """Returns a copy of `risk` with max_account_drawdown_pct AND
    daily_loss_limit_pct filled in from `prop_rules` whenever the caller
    hasn't already set an explicit value of their own for that field.
    This is what makes the account-blown circuit breaker AND the
    daily-loss circuit breaker (see app.backtest.execution) apply
    automatically during the RAW BACKTEST itself -- not just later, when
    the post-hoc prop simulator (app.prop.simulator.simulate_account)
    checks the finished trade sequence against the same rules. Never
    overrides a value the caller explicitly configured on either field.

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
    change."""
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
    if not updates:
        return risk
    return replace(risk, **updates)


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
