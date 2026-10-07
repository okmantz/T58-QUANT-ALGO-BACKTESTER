"""
PropAccount -- the ONE prop-firm account rule engine.

ACCURACY OVERHAUL (2026-10-07, plan section "Prop rules, one implementation"):
before this module the repo enforced prop-firm rules in three places that
disagreed with each other on the same trades:

  * the bar engine (app.backtest.execution) -- static floor on realized
    equity, daily loss = forced close, equity "teleported" back to the
    starting balance on a breach;
  * the post-hoc simulator (app.prop.simulator.simulate_account) -- trailing
    high-water mark on closed trades only, daily loss = failed account;
  * the Monte Carlo (app.monte_carlo.engine) -- UTC-midnight day buckets.

PropAccount is a small state machine for ONE attempt (one purchased
evaluation/funded account). It owns every rule that decides whether that
account is alive, passed or failed:

  * trailing or static max drawdown, measured on the equity basis the firm
    actually uses (`PropRules.dd_basis`): realized balance, end-of-day
    balance, or floating (intraday) equity -- with an optional LOCK of the
    floor at the starting balance (`trailing_lock`);
  * the daily loss limit on the firm's SESSION day, optionally including
    open P&L (`daily_loss_basis="floating"`), with the firm's action
    (`daily_loss_action`: fail the account vs lock the day);
  * profit target + minimum trading days + consistency rule + evaluation
    time limit + inactivity rule;
  * the funded stage: payout eligibility (consistency, winning days,
    frequency, caps), with the withdrawal netted out of the trailing peak.

It is fed by EITHER side of the app, so the same trades produce the same
verdict everywhere:

  * trade-sequence callers (simulate_account, Monte Carlo, rolling
    evaluation) call `on_trade_close()` once per closed trade, optionally
    passing the trade's measured adverse/favorable excursion;
  * the bar engine additionally calls `on_equity_extremes()` every bar with
    the floating equity high/low, and `begin_day()` / `end_day()` as the
    session day rolls.

With every new PropRules field at its default (`dd_basis="legacy"`,
`daily_loss_basis="realized"`, no lock) the behavior is byte-for-byte the
behavior simulate_account had before it delegated here -- the differential
test in tests/test_account_parity.py replays thousands of random sequences
against the pre-refactor implementation to hold that line.

This module deliberately imports nothing from app.prop.simulator (which
imports THIS module); `rules` is duck-typed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd


@dataclass
class PayoutRecord:
    day_index: int
    amount: float
    balance_after: float


DD_BASES = ("legacy", "realized", "eod", "floating")
DAILY_LOSS_BASES = ("realized", "floating")

OK = "ok"
FAILED = "failed"
DAY_LOCKED = "day_locked"


def _days_between(later: Any, earlier: Any) -> int:
    return (pd.Timestamp(later) - pd.Timestamp(earlier)).days


class PropAccount:
    """State of ONE evaluation/funded attempt. See the module docstring."""

    def __init__(
        self,
        rules: Any,
        *,
        start_day_index: int = 0,
        day_dates: Sequence | None = None,
    ):
        self.rules = rules
        self.day_dates = day_dates
        self.start_day_index = start_day_index
        self.attempt_last_day_index = start_day_index
        self.last_day_index_reached = start_day_index

        dd_basis = getattr(rules, "dd_basis", "legacy") or "legacy"
        if dd_basis not in DD_BASES:
            raise ValueError(f"dd_basis must be one of {DD_BASES}, got {dd_basis!r}")
        self.dd_basis = dd_basis
        self._legacy = dd_basis == "legacy"
        self.daily_loss_basis = getattr(rules, "daily_loss_basis", "realized") or "realized"
        if self.daily_loss_basis not in DAILY_LOSS_BASES:
            raise ValueError(f"daily_loss_basis must be one of {DAILY_LOSS_BASES}, got {self.daily_loss_basis!r}")

        # -- money -------------------------------------------------------
        self.balance = float(rules.account_size)
        self.trailing_peak = float(rules.account_size)
        self.static_floor = rules.account_size * (1 - rules.max_drawdown_pct / 100.0)

        # -- outcome -----------------------------------------------------
        self.stage = "evaluation"
        self.passed_evaluation = False
        self.days_to_pass: int | None = None
        self.failed = False
        self.failure_reason: str | None = None
        self.failure_day_index: int | None = None
        self.max_dd_pct_reached = 0.0

        # -- per-day bookkeeping ----------------------------------------
        self.daily_pnl: dict = {}
        self.day_profit_history: dict = {}
        self.day_peak: dict = {}
        self.day_close: dict = {}
        self.day_start_balance: dict = {}
        self.locked_day: int | None = None
        self._prev_day_index: int | None = None
        self._current_day: int | None = None

        # -- eval / funded bookkeeping ----------------------------------
        self.total_profit_since_start = 0.0
        self.best_day_profit = 0.0
        self.best_day_since_baseline = 0.0
        self.payout_baseline_balance = float(rules.account_size)
        self.funded_start_day_index: int | None = None
        self.last_payout_day_index = -10 ** 9
        self.payouts: list[PayoutRecord] = []
        self.first_payout_day_index: int | None = None
        self.first_payout_amount: float | None = None
        self.n_trades = 0

    # ------------------------------------------------------------------
    # derived rule values
    # ------------------------------------------------------------------
    @property
    def trailing(self) -> bool:
        return self.rules.drawdown_type == "trailing"

    def _trail_distance_basis_peak(self) -> bool:
        return getattr(self.rules, "trailing_distance_basis", "peak") != "account"

    def floor(self) -> float:
        """Current equity level at/below which the account fails the max-
        drawdown rule. Trailing: follows the high-water mark (percent of
        the PEAK by default -- the repo's long-standing convention -- or a
        fixed distance from the starting balance with
        trailing_distance_basis="account", which is how most futures firms
        define it), optionally LOCKED so it never rises above the starting
        balance (+ offset)."""
        r = self.rules
        if not self.trailing:
            return self.static_floor
        if self._trail_distance_basis_peak():
            fl = self.trailing_peak * (1 - r.max_drawdown_pct / 100.0)
        else:
            fl = self.trailing_peak - r.account_size * (r.max_drawdown_pct / 100.0)
        if getattr(r, "trailing_lock", False):
            lock_level = r.account_size * (1 + float(getattr(r, "trailing_lock_offset_pct", 0.0) or 0.0) / 100.0)
            fl = min(fl, lock_level)
        return fl

    def daily_limit_dollars(self, day_index: int | None = None) -> float:
        r = self.rules
        day = self._current_day if day_index is None else day_index
        if r.daily_loss_base == "initial" or day is None:
            base = r.account_size
        elif r.daily_loss_base == "prior_day_high":
            base = self.day_peak.get(day - 1, r.account_size)
        else:  # ratchet_up
            prior = [c for d, c in self.day_close.items() if d < day]
            base = max([r.account_size] + prior)
        return base * (r.daily_loss_limit_pct / 100.0)

    def daily_floor_equity(self, day_index: int | None = None) -> float | None:
        """Equity level at which today's daily-loss limit is breached
        (floating basis: day-start balance minus the limit). None when the
        daily limit is effectively disabled."""
        r = self.rules
        if r.daily_loss_limit_pct is None or r.daily_loss_limit_pct >= 100.0:
            return None
        day = self._current_day if day_index is None else day_index
        start = self.day_start_balance.get(day, self.balance)
        return start - self.daily_limit_dollars(day)

    def breach_equity_level(self) -> float:
        """The equity level at which the account would be terminated right
        now by the drawdown rule (used by the bar engine to place the
        liquidation price)."""
        return self.floor()

    # ------------------------------------------------------------------
    def _fail(self, reason: str, day_index: int) -> str:
        self.failed = True
        self.failure_reason = reason
        self.failure_day_index = day_index
        return FAILED

    def begin_day(self, day_index: int) -> None:
        """Marks the start of a session day (idempotent). Records the
        balance the day started with, the base for floating daily loss."""
        self._current_day = day_index
        self.day_start_balance.setdefault(day_index, self.balance)

    def end_day(self, day_index: int, closing_equity: float | None = None) -> None:
        """Session day over: the end-of-day HWM update for dd_basis="eod"."""
        if self.dd_basis == "eod" and self.trailing:
            eq = self.balance if closing_equity is None else closing_equity
            self.trailing_peak = max(self.trailing_peak, eq)

    # ------------------------------------------------------------------
    # floating-equity checks (bar engine, or a trade's MAE/MFE)
    # ------------------------------------------------------------------
    def on_equity_extremes(
        self, day_index: int, equity_high: float, equity_low: float,
    ) -> str:
        """Feed the FLOATING equity range reached inside one tick/bar (or
        one trade, via its measured MAE/MFE). Applies:

          * dd_basis="floating": raises the trailing peak to equity_high
            FIRST (the conservative order -- the favorable excursion is
            assumed to come before the adverse one, which is the worst case
            for a trailing floor), then breaches if equity_low touches the
            floor;
          * dd_basis="eod": breaches if equity_low touches the floor (the
            floor itself only moves at end of day);
          * daily_loss_basis="floating": breaches/locks if equity_low is
            more than the daily limit below the day's starting balance.

        No-op (returns OK) for dd_basis in ("legacy","realized") with a
        realized daily-loss basis, so legacy callers are untouched."""
        if self.failed:
            return FAILED
        self.begin_day(day_index)
        r = self.rules
        floating_dd = self.dd_basis in ("floating", "eod")
        if floating_dd and self.trailing and self.dd_basis == "floating":
            self.trailing_peak = max(self.trailing_peak, equity_high)
        if floating_dd:
            fl = self.floor()
            if self.trailing:
                cur_dd = max(0.0, (self.trailing_peak - equity_low) / self.trailing_peak * 100.0) if self.trailing_peak else 0.0
            else:
                cur_dd = max(0.0, (r.account_size - equity_low) / r.account_size * 100.0) if r.account_size else 0.0
            self.max_dd_pct_reached = max(self.max_dd_pct_reached, cur_dd)
            if equity_low <= fl:
                return self._fail(f"max_drawdown ({r.drawdown_type}, floating)", day_index)
        if self.daily_loss_basis == "floating":
            dfloor = self.daily_floor_equity(day_index)
            if dfloor is not None and equity_low <= dfloor:
                if r.daily_loss_action == "lock_day":
                    self.locked_day = day_index
                    return DAY_LOCKED
                return self._fail("daily_loss_limit", day_index)
        return OK

    def projected_floors(self, equity_high: float | None = None) -> tuple[float | None, float | None]:
        """(drawdown_floor_equity, daily_floor_equity) as they would stand
        for a bar whose floating high is `equity_high`, WITHOUT mutating the
        account. dd floor is None unless dd_basis is floating/eod (those are
        the bases checked on floating equity); daily floor is None unless
        daily_loss_basis == "floating". Lets the bar engine decide whether a
        resting stop or the firm's liquidation level is reached first before
        it commits the excursion to the account."""
        dd_floor = None
        if self.dd_basis in ("floating", "eod"):
            if self.trailing and self.dd_basis == "floating" and equity_high is not None:
                saved = self.trailing_peak
                self.trailing_peak = max(self.trailing_peak, equity_high)
                dd_floor = self.floor()
                self.trailing_peak = saved
            else:
                dd_floor = self.floor()
        daily = None
        if self.daily_loss_basis == "floating":
            daily = self.daily_floor_equity()
        return dd_floor, daily

    def check_new_day(self, today_date: Any) -> str:
        """Clock-driven rules that need no trade: evaluation calendar limit
        and the inactivity rule. Call at the first bar of each session day
        the account is alive. `today_date` is the session date; the account
        measures from `start_date` (the date it was bought) and from its
        last trade day."""
        if self.failed:
            return FAILED
        r = self.rules
        start = getattr(self, "start_date", None)
        if start is None and self.day_dates is not None and self.start_day_index < len(self.day_dates):
            start = self.day_dates[self.start_day_index]
        if start is None:
            return OK
        day_index = self.last_day_index_reached
        if (self.stage == "evaluation" and r.max_eval_calendar_days is not None
                and _days_between(today_date, start) > r.max_eval_calendar_days):
            return self._fail("eval_time_limit", day_index)
        if getattr(r, "max_inactive_days", None) is not None:
            last = start
            if self._prev_day_index is not None and self.day_dates is not None:
                last = self.day_dates[self._prev_day_index]
            if _days_between(today_date, last) - 1 > r.max_inactive_days:
                return self._fail("inactivity", day_index)
        return OK

    # ------------------------------------------------------------------
    # realized accounting (trade-sequence callers AND the bar engine)
    # ------------------------------------------------------------------
    def on_trade_close(
        self,
        pnl: float,
        day_index: int,
        *,
        is_last_of_day: bool = True,
        trade_initial_risk: float | None = None,
        mae: float | None = None,
        mfe: float | None = None,
        floating_handled: bool = False,
    ) -> str:
        """One CLOSED trade's realized P&L. Returns OK, FAILED or DAY_LOCKED
        (the day's remaining trades must then be skipped -- see
        `locked_day`). mae/mfe are the trade's measured adverse (<=0) and
        favorable (>=0) excursion in dollars; when given and the rules use a
        floating basis, they are folded in as the trade's floating equity
        range BEFORE the realized checks (set floating_handled=True when the
        caller already fed per-bar extremes)."""
        if self.failed:
            # The account already failed on a floating excursion (bar engine);
            # the liquidation trade's realized P&L still belongs in the final
            # balance, but no further rule checks run.
            self.balance += pnl
            return FAILED
        r = self.rules
        cur = day_index
        self.attempt_last_day_index = cur
        self.last_day_index_reached = cur
        self.n_trades += 1

        # inactivity rule (calendar-day gap between two trade days)
        if (
            getattr(r, "max_inactive_days", None) is not None
            and self._prev_day_index is not None
            and self._prev_day_index != cur
            and self.day_dates is not None
        ):
            gap_days = _days_between(self.day_dates[cur], self.day_dates[self._prev_day_index])
            if gap_days - 1 > r.max_inactive_days:
                return self._fail("inactivity", cur)
        self._prev_day_index = cur

        self.begin_day(cur)
        balance_before = self.balance

        # floating pre-check from the trade's own excursion
        floating_basis_active = (self.dd_basis in ("floating", "eod")) or self.daily_loss_basis == "floating"
        if floating_basis_active and not floating_handled:
            adverse = min(mae, 0.0) if mae is not None else min(pnl, 0.0)
            favorable = max(mfe, 0.0) if mfe is not None else max(pnl, 0.0)
            res = self.on_equity_extremes(cur, balance_before + favorable, balance_before + adverse)
            if res == FAILED:
                return FAILED
            if res == DAY_LOCKED:
                return DAY_LOCKED

        self.balance += pnl
        self.daily_pnl[cur] = self.daily_pnl.get(cur, 0.0) + pnl
        self.day_profit_history[cur] = self.day_profit_history.get(cur, 0.0) + pnl
        self.total_profit_since_start += pnl
        self.best_day_profit = max(self.best_day_profit, self.day_profit_history[cur])

        # WHEN the realized checks run
        if self._legacy:
            check_now = (r.drawdown_check_mode == "intrabar") or is_last_of_day
        elif self.dd_basis == "eod":
            check_now = True  # realized floor check each close; HWM moves at day end only
        else:
            check_now = True
        if not check_now:
            return OK  # "eod" legacy mode: defer to the day's last trade

        self.day_peak[cur] = max(self.day_peak.get(cur, float("-inf")), self.balance)
        if is_last_of_day:
            self.day_close[cur] = self.balance

        # high-water mark
        if self.dd_basis == "eod":
            if is_last_of_day:
                self.trailing_peak = max(self.trailing_peak, self.balance)
        else:
            self.trailing_peak = max(self.trailing_peak, self.balance)
        if self.trailing:
            dd_floor = self.floor()
            current_dd_pct = max(0.0, (self.trailing_peak - self.balance) / self.trailing_peak * 100.0) if self.trailing_peak else 0.0
        else:
            dd_floor = self.static_floor
            current_dd_pct = max(0.0, (r.account_size - self.balance) / r.account_size * 100.0) if r.account_size else 0.0
        self.max_dd_pct_reached = max(self.max_dd_pct_reached, current_dd_pct)

        # daily loss on the realized day P&L
        dll_limit = self.daily_limit_dollars(cur)
        if self.daily_pnl[cur] <= -dll_limit:
            if r.daily_loss_action == "lock_day":
                self.day_close[cur] = self.balance
                self.locked_day = cur
                return DAY_LOCKED
            return self._fail("daily_loss_limit", cur)

        if self.balance <= dd_floor:
            return self._fail(f"max_drawdown ({r.drawdown_type})", cur)

        # legacy opt-in floating proxy: assume the trade drew down to its
        # full initial risk before closing.
        if self._legacy and r.floating_drawdown_mode == "adverse" and trade_initial_risk is not None:
            adverse_balance = self.balance - pnl - float(trade_initial_risk)
            if adverse_balance <= dd_floor:
                return self._fail(f"max_drawdown ({r.drawdown_type}, adverse-floating proxy)", cur)

        # evaluation pass / funded payout
        if self.stage == "evaluation":
            if r.max_eval_calendar_days is not None and self.day_dates is not None:
                eval_cal_days = _days_between(self.day_dates[cur], self.day_dates[self.start_day_index])
                if eval_cal_days > r.max_eval_calendar_days:
                    return self._fail("eval_time_limit", cur)
            target_balance = r.account_size * (1 + r.evaluation_profit_target_pct / 100.0)
            trading_days_so_far = cur - self.start_day_index + 1
            if self.balance >= target_balance and trading_days_so_far >= r.min_trading_days:
                consistency_ok = True
                if r.consistency_rule_pct is not None and self.total_profit_since_start > 0:
                    consistency_ok = (
                        self.best_day_profit / self.total_profit_since_start * 100.0
                    ) <= r.consistency_rule_pct
                if consistency_ok:
                    self.stage = "funded"
                    self.passed_evaluation = True
                    self.days_to_pass = trading_days_so_far
                    self.payout_baseline_balance = self.balance
                    self.best_day_since_baseline = 0.0
                    self.funded_start_day_index = cur
                    self.last_payout_day_index = cur
        elif self.stage == "funded":
            self._maybe_payout(cur)
        return OK

    # ------------------------------------------------------------------
    def _maybe_payout(self, cur: int) -> None:
        r = self.rules
        profit_since_baseline = self.balance - self.payout_baseline_balance
        required_profit = r.account_size * (r.payout_threshold_pct / 100.0) \
            + r.account_size * (r.required_buffer_pct / 100.0)
        self.best_day_since_baseline = max(
            self.best_day_since_baseline, self.day_profit_history.get(cur, 0.0)
        )
        funded_consistency_ok = True
        if r.funded_consistency_rule_pct is not None and profit_since_baseline > 0:
            funded_consistency_ok = (
                self.best_day_since_baseline / profit_since_baseline * 100.0
            ) <= r.funded_consistency_rule_pct
        days_since_last_payout = cur - self.last_payout_day_index
        funded_from = self.funded_start_day_index if self.funded_start_day_index is not None else cur
        winning_days = sum(
            1 for d, p in self.day_profit_history.items()
            if d >= funded_from and p >= r.min_winning_day_profit
        )
        if (profit_since_baseline >= required_profit
                and days_since_last_payout >= r.payout_frequency_days
                and winning_days >= r.winning_days_for_payout
                and funded_consistency_ok
                and (r.max_payouts is None or len(self.payouts) < r.max_payouts)):
            withdrawable = profit_since_baseline - r.account_size * (r.required_buffer_pct / 100.0)
            if r.payout_cap_pct is not None:
                withdrawable = min(withdrawable, profit_since_baseline * (r.payout_cap_pct / 100.0))
            if r.payout_cap_dollars is not None:
                withdrawable = min(withdrawable, r.payout_cap_dollars)
            withdrawable = max(withdrawable, 0.0)
            if withdrawable > 0:
                self.balance -= withdrawable
                self.payouts.append(PayoutRecord(day_index=cur, amount=withdrawable, balance_after=self.balance))
                if self.first_payout_day_index is None:
                    self.first_payout_day_index = cur
                    self.first_payout_amount = withdrawable
                self.payout_baseline_balance = self.balance
                self.last_payout_day_index = cur
                self.best_day_since_baseline = 0.0
                # a withdrawal nets out of the trailing peak so being paid
                # can never push the account toward its own floor.
                self.trailing_peak = max(self.trailing_peak - withdrawable, self.balance)

    # ------------------------------------------------------------------
    @property
    def reached_first_payout(self) -> bool:
        return self.first_payout_day_index is not None

    @property
    def total_payout_amount(self) -> float:
        return sum(p.amount for p in self.payouts)

    def status(self) -> str:
        if self.failed:
            return "failed"
        return "passed" if self.passed_evaluation else "active"

    def snapshot(self) -> dict:
        return {
            "status": self.status(), "stage": self.stage, "balance": self.balance,
            "trailing_peak": self.trailing_peak, "floor": self.floor(),
            "failure_reason": self.failure_reason, "passed_evaluation": self.passed_evaluation,
            "days_to_pass": self.days_to_pass, "n_trades": self.n_trades,
            "payouts": len(self.payouts), "max_dd_pct_reached": self.max_dd_pct_reached,
        }


def coerce_rules(rules: Any):
    """Accept PropRules, a dict of PropRules fields, or None."""
    if rules is None:
        return None
    if isinstance(rules, dict):
        from app.prop.simulator import PropRules
        return PropRules(**rules)
    return rules
