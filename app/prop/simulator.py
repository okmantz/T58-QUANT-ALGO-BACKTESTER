"""
Prop-firm evaluation + funded-account simulator.

Given a chronological sequence of trade P&Ls (plus their dates), walks the
account forward under a configurable set of prop-firm rules and determines:
  - whether/when the evaluation is passed
  - whether/when the account is failed (and which rule caused it)
  - whether/when the funded account reaches its first payout, and all
    subsequent payouts

This module is independent of *how* the trade sequence was produced, so the
exact same simulate_account() function is used both for the single
historical trade sequence (section 4 of the spec) and for every resampled
sequence inside the Monte Carlo engine (section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type

import pandas as pd


@dataclass
class PropRules:
    account_size: float = 100_000.0
    evaluation_profit_target_pct: float = 8.0     # % gain required to pass evaluation
    daily_loss_limit_pct: float = 5.0              # max loss in a single day (% of account_size)
    max_drawdown_pct: float = 10.0                 # overall max drawdown allowed
    drawdown_type: str = "trailing"                 # "trailing" | "static"
    drawdown_check_mode: str = "intrabar"           # "intrabar" | "eod" -- see note below
    consistency_rule_pct: float | None = 30.0       # best single day's profit <= this % of total profit
    min_trading_days: int = 5
    payout_threshold_pct: float = 0.0               # extra profit % (above account size) required before 1st payout eligibility, funded stage
    payout_cap_pct: float | None = None             # max % of available profit withdrawable per payout (None = 100%)
    payout_frequency_days: int = 14                 # min days between payouts
    required_buffer_pct: float = 0.0                # profit buffer that must be maintained above account_size before payout
    max_position_size: float | None = None          # informational cap on units (enforced in RiskConfig)

    # -- live-execution-only fields (added for Deploy Live / execution_engine.py) --------------
    # These four have NO effect on simulate_account() below or on any backtest/Monte Carlo/
    # Evolution Lab output -- they describe constraints on HOW a strategy is allowed to be
    # traded live, not on a fixed sequence of already-realized trade P&Ls, so there is nothing
    # for the historical-trade-sequence simulator to enforce. They are read directly by
    # app.live_deploy.execution_engine.LiveExecutionSession. Kept on this same dataclass
    # (rather than a separate one) so the Enter Prop-Firm Rules screen can present them as
    # ordinary optional fields alongside profit target / drawdown / consistency, and so a
    # single PropRules instance -- built by hand or from a preset -- is what both the
    # eval-pass-probability tools AND Deploy Live consume.
    news_blackout_windows: str = ""                 # one per line: "HH:MM-HH:MM" (daily) or "FRI 19:55-21:05" (specific weekday)
    weekend_hold_allowed: bool = True                # False = flatten all positions before the weekend and block new entries until Monday
    max_lot_size: float | None = None                # hard cap on live order volume, independent of RiskConfig.max_position_size
    hedging_allowed: bool = True                     # False = block opening a position opposite an already-open one, account-wide

    # drawdown_check_mode controls WHEN the daily-loss-limit and max-drawdown
    # failure checks are evaluated:
    #   "intrabar" (default, conservative): checked after every single trade
    #   as it closes, matching a firm that monitors floating equity in real
    #   time and can auto-liquidate mid-day the instant a floor is crossed.
    #   "eod": checked only once per calendar day, using that day's final
    #   cumulative balance after all of that day's trades -- matching firms
    #   (many real futures prop firms, including both evaluated in this
    #   codebase's research notes) that explicitly state an "EOD" drawdown
    #   type: you can be deep underwater intraday and still be fine as long
    #   as you close the day above the floor. Using "intrabar" against a
    #   firm that is actually "eod" understates your true pass probability;
    #   using "eod" against a firm that is actually real-time overstates it.
    #   Match this to what the specific firm's rules document actually say.


@dataclass
class PayoutEvent:
    day_index: int
    date: str
    amount: float
    balance_after: float


@dataclass
class AttemptRecord:
    """One "buy-in" inside a reset-on-breach chain (see simulate_account's
    `reset_on_breach` parameter): a single evaluation/funded run that starts
    fresh at `rules.account_size` and ends either because it busted a
    prop-firm rule or because the available trade history ran out while it
    was still alive. Only ever produced when `reset_on_breach=True` -- a
    normal (non-reset) call to simulate_account still returns exactly one
    of these (attempt #1) so callers can treat both modes uniformly, but
    the chain only continues past attempt #1 when reset_on_breach is set.
    """
    attempt_index: int              # 0-based: 0 is the first attempt
    start_day_index: int            # day index (in the full trade sequence) this attempt started on
    end_day_index: int              # day index this attempt ended on (failed or ran out of trades)
    days_used: int                  # trading days this attempt lasted
    passed_evaluation: bool
    failed: bool
    failure_reason: str | None
    reached_first_payout: bool
    payout_amount: float            # total $ paid out (gross, before any profit split) during this attempt
    ending_balance: float


@dataclass
class AccountSimResult:
    passed_evaluation: bool
    failed: bool
    failure_reason: str | None
    failure_day_index: int | None
    days_to_pass: int | None
    first_payout_day_index: int | None
    first_payout_amount: float | None
    payouts: list[PayoutEvent] = field(default_factory=list)
    final_balance: float = 0.0
    max_drawdown_pct_reached: float = 0.0
    trading_days_count: int = 0
    # -- reset-on-breach chain bookkeeping (see simulate_account's docstring) --
    # Populated for every call (attempts always has at least one record,
    # attempt #1); only ever has more than one entry when reset_on_breach
    # was True and an earlier attempt busted with trade history still left
    # to mechanically "rebuy" into. These four fields are pure summary
    # views over `attempts` -- nothing here changes any pre-existing field
    # above, and every field above still describes attempt #1 alone when
    # reset_on_breach=False, exactly as before this chain feature existed.
    reset_on_breach: bool = False
    attempts: list[AttemptRecord] = field(default_factory=list)
    total_attempts: int = 1
    attempts_passed: int = 0
    attempts_reached_payout: int = 0

    @property
    def reached_first_payout(self) -> bool:
        return self.first_payout_day_index is not None

    @property
    def total_payout_amount(self) -> float:
        return sum(p.amount for p in self.payouts)


@dataclass
class DayStructure:
    """The part of simulate_account's bookkeeping that depends ONLY on
    `trade_dates`, never on the P&L values themselves: which calendar day
    each trade in the sequence falls on, and which trades are the last of
    their day. The Monte Carlo engine resamples/shuffles P&L VALUES across
    thousands of simulations while reusing the exact same (fixed,
    historical) trade dates every time -- see app.monte_carlo.engine's
    run_monte_carlo -- so this structure is identical across every one of
    those simulations and only needs to be computed once, not
    re-derived (via per-trade pandas Timestamp parsing and dict lookups)
    on every single call. Precomputing it once and passing it in cut a
    meaningful amount of wall-clock time out of every Monte Carlo run,
    the walk-forward-aware GA, and Search Lab -- all of which call
    simulate_account many thousands of times per run -- without changing
    a single output value.
    """
    day_index_per_trade: list       # length == n_trades; which day (0-based, chronological) each trade belongs to
    is_last_of_day: list            # length == n_trades; True where a trade is the last one of its calendar day
    day_dates: list                 # length == n_days; the normalized date for each day index
    n_days: int


def precompute_day_structure(trade_dates: list) -> DayStructure:
    """Builds the (dates-only) bookkeeping simulate_account needs, once,
    so it can be reused across many calls that all share the same
    `trade_dates` but different `trade_pnls` (exactly the Monte Carlo
    engine's resampling pattern). See DayStructure's docstring."""
    dates_norm = [pd.Timestamp(d).normalize() for d in trade_dates]
    day_index_map: dict = {}
    day_order: list = []
    day_index_per_trade: list = []
    for d in dates_norm:
        idx = day_index_map.get(d)
        if idx is None:
            idx = len(day_order)
            day_index_map[d] = idx
            day_order.append(d)
        day_index_per_trade.append(idx)
    n = len(dates_norm)
    is_last_of_day = [
        (i == n - 1) or (dates_norm[i] != dates_norm[i + 1])
        for i in range(n)
    ]
    return DayStructure(
        day_index_per_trade=day_index_per_trade, is_last_of_day=is_last_of_day,
        day_dates=day_order, n_days=len(day_order),
    )


def simulate_account(
    trade_pnls: list[float],
    trade_dates: list,
    rules: PropRules,
    _day_structure: "DayStructure | None" = None,
    reset_on_breach: bool = False,
) -> AccountSimResult:
    """
    trade_pnls: P&L of each trade (account-currency $), in chronological order
    trade_dates: date (or datetime) of each trade, same order/length as trade_pnls

    _day_structure: internal fast-path for callers (the Monte Carlo engine)
    that invoke this function many times with the SAME trade_dates and only
    trade_pnls changing -- pass a DayStructure from precompute_day_structure()
    once, computed from trade_dates, to skip re-deriving it on every call.
    Every other caller can ignore this parameter entirely; it's derived
    from trade_dates automatically when omitted, with identical results.

    reset_on_breach: when False (the default -- byte-identical to this
    function's behavior before this parameter existed), the walk stops the
    instant the account busts a rule, exactly as before. When True, a bust
    does NOT end the simulation: the account snaps back to a fresh
    `rules.account_size` and a new "attempt" starts on the very next trade
    in the SAME sequence, continuing until either an attempt survives all
    the way to the end of `trade_pnls` (the chain stops there -- there's no
    more history to mechanically rebuy into) or the trades run out mid-
    attempt. This models "what if I mechanically bought a new evaluation
    every single time this one busted" against ONE fixed, already-ordered
    trade sequence -- e.g. one Monte Carlo simulation's resampled path --
    as opposed to app.prop.survival_engine.simulate_reset_chain, which
    draws a FRESH independent resample for every attempt. Every attempt is
    recorded in the returned AccountSimResult.attempts list; the top-level
    passed_evaluation/failed/first_payout_day_index/etc. fields describe
    the FIRST attempt only that reached that milestone (so a caller that
    never asks about `attempts` sees exactly the single-attempt semantics
    it always has), while total_attempts/attempts_passed/
    attempts_reached_payout summarize the whole chain. This function never
    applies attempt fees, profit splits, or a bankroll cap -- that
    trader-economics layer lives in app.monte_carlo.bankroll, which
    consumes `attempts` to answer "does a trader with $X survive this
    chain to a payout," precisely so this function stays a pure mechanical
    rules engine, not an economics one.
    """
    if len(trade_pnls) == 0:
        return AccountSimResult(
            passed_evaluation=False, failed=False, failure_reason="No trades generated.",
            failure_day_index=None, days_to_pass=None, first_payout_day_index=None,
            first_payout_amount=None, final_balance=rules.account_size,
            reset_on_breach=reset_on_breach, attempts=[], total_attempts=0,
        )

    day_structure = _day_structure if _day_structure is not None else precompute_day_structure(trade_dates)
    day_index_per_trade = day_structure.day_index_per_trade
    is_last_of_day = day_structure.is_last_of_day
    day_dates = day_structure.day_dates
    n = len(trade_pnls)
    static_floor = rules.account_size * (1 - rules.max_drawdown_pct / 100.0)

    attempts: list[AttemptRecord] = []
    overall_payouts: list[PayoutEvent] = []
    overall_passed_evaluation = False
    overall_days_to_pass = None
    overall_first_payout_day_index = None
    overall_first_payout_amount = None
    overall_max_dd_pct_reached = 0.0
    last_day_idx_reached = -1

    i = 0
    attempt_index = 0
    # `balance`/`failed`/`failure_reason`/`failure_day_index` below always
    # describe whichever attempt most recently ran, so that after the loop
    # exits (for any reason) they can be used directly as the overall
    # result's trailing state -- matching the single-attempt case exactly.
    balance = rules.account_size
    failed = False
    failure_reason = None
    failure_day_index = None

    while i < n:
        # --- fresh per-attempt state ---
        balance = rules.account_size
        trailing_peak = rules.account_size
        stage = "evaluation"
        passed_evaluation = False
        days_to_pass = None
        failed = False
        failure_reason = None
        failure_day_index = None
        attempt_start_day_idx = day_index_per_trade[i]
        attempt_last_day_idx = attempt_start_day_idx
        daily_pnl: dict = {}
        day_profit_history: dict = {}
        payouts_this_attempt: list[PayoutEvent] = []
        first_payout_day_index_attempt = None
        first_payout_amount_attempt = None
        last_payout_day_index = -10 ** 9
        payout_baseline_balance = rules.account_size
        total_profit_since_start = 0.0
        best_day_profit = 0.0

        j = i
        while j < n:
            pnl = trade_pnls[j]
            cur_day_idx = day_index_per_trade[j]
            attempt_last_day_idx = cur_day_idx
            last_day_idx_reached = cur_day_idx

            balance += pnl
            daily_pnl[cur_day_idx] = daily_pnl.get(cur_day_idx, 0.0) + pnl
            day_profit_history[cur_day_idx] = day_profit_history.get(cur_day_idx, 0.0) + pnl
            total_profit_since_start += pnl
            best_day_profit = max(best_day_profit, day_profit_history[cur_day_idx])

            check_now = (rules.drawdown_check_mode == "intrabar") or is_last_of_day[j]

            if not check_now:
                # "eod" mode: this trade isn't the day's last -- defer the
                # peak/drawdown/failure evaluation until the day is complete.
                j += 1
                continue

            trailing_peak = max(trailing_peak, balance)
            if rules.drawdown_type == "trailing":
                dd_floor = trailing_peak * (1 - rules.max_drawdown_pct / 100.0)
            else:
                dd_floor = static_floor
            current_dd_pct = max(0.0, (trailing_peak - balance) / trailing_peak * 100.0) if trailing_peak else 0.0
            overall_max_dd_pct_reached = max(overall_max_dd_pct_reached, current_dd_pct)

            # --- Failure checks (apply in both evaluation and funded stages) ---
            if daily_pnl[cur_day_idx] <= -rules.account_size * (rules.daily_loss_limit_pct / 100.0):
                failed = True
                failure_reason = "daily_loss_limit"
                failure_day_index = cur_day_idx
                break

            if balance <= dd_floor:
                failed = True
                failure_reason = f"max_drawdown ({rules.drawdown_type})"
                failure_day_index = cur_day_idx
                break

            # --- Evaluation pass check ---
            if stage == "evaluation":
                target_balance = rules.account_size * (1 + rules.evaluation_profit_target_pct / 100.0)
                trading_days_so_far = cur_day_idx - attempt_start_day_idx + 1
                if balance >= target_balance and trading_days_so_far >= rules.min_trading_days:
                    consistency_ok = True
                    if rules.consistency_rule_pct is not None and total_profit_since_start > 0:
                        consistency_ok = (
                            best_day_profit / total_profit_since_start * 100.0
                        ) <= rules.consistency_rule_pct
                    if consistency_ok:
                        stage = "funded"
                        passed_evaluation = True
                        days_to_pass = trading_days_so_far
                        payout_baseline_balance = balance
                        last_payout_day_index = cur_day_idx  # start payout clock from pass date

            # --- Funded stage payout check ---
            elif stage == "funded":
                profit_since_baseline = balance - payout_baseline_balance
                required_profit = rules.account_size * (rules.payout_threshold_pct / 100.0) \
                    + rules.account_size * (rules.required_buffer_pct / 100.0)
                days_since_last_payout = cur_day_idx - last_payout_day_index
                if profit_since_baseline > required_profit and days_since_last_payout >= rules.payout_frequency_days:
                    withdrawable = profit_since_baseline - rules.account_size * (rules.required_buffer_pct / 100.0)
                    if rules.payout_cap_pct is not None:
                        withdrawable = min(withdrawable, profit_since_baseline * (rules.payout_cap_pct / 100.0))
                    withdrawable = max(withdrawable, 0.0)
                    if withdrawable > 0:
                        balance -= withdrawable
                        payout = PayoutEvent(
                            day_index=cur_day_idx,
                            date=str(day_dates[cur_day_idx].date()),
                            amount=withdrawable,
                            balance_after=balance,
                        )
                        payouts_this_attempt.append(payout)
                        if first_payout_day_index_attempt is None:
                            first_payout_day_index_attempt = cur_day_idx
                            first_payout_amount_attempt = withdrawable
                        payout_baseline_balance = balance
                        last_payout_day_index = cur_day_idx
                        trailing_peak = max(trailing_peak, balance)

            j += 1

        # --- attempt finished: either it busted (break above) or it rode
        # out every remaining trade without failing (j == n) ---
        attempts.append(AttemptRecord(
            attempt_index=attempt_index,
            start_day_index=attempt_start_day_idx,
            end_day_index=attempt_last_day_idx,
            days_used=max(0, attempt_last_day_idx - attempt_start_day_idx + 1),
            passed_evaluation=passed_evaluation,
            failed=failed,
            failure_reason=failure_reason,
            reached_first_payout=first_payout_day_index_attempt is not None,
            payout_amount=sum(p.amount for p in payouts_this_attempt),
            ending_balance=balance,
        ))
        overall_payouts.extend(payouts_this_attempt)
        if not overall_passed_evaluation and passed_evaluation:
            overall_passed_evaluation = True
            overall_days_to_pass = days_to_pass
        if overall_first_payout_day_index is None and first_payout_day_index_attempt is not None:
            overall_first_payout_day_index = first_payout_day_index_attempt
            overall_first_payout_amount = first_payout_amount_attempt

        attempt_index += 1

        if not reset_on_breach or not failed:
            # Non-reset mode always stops after attempt #1 (matching this
            # function's behavior before reset_on_breach existed); reset
            # mode stops once an attempt survives to the end of the data
            # (nothing left to mechanically rebuy into).
            break

        i = j + 1  # failed at index j -- next attempt starts on the following trade

    return AccountSimResult(
        passed_evaluation=overall_passed_evaluation,
        failed=failed,
        failure_reason=failure_reason,
        failure_day_index=failure_day_index,
        days_to_pass=overall_days_to_pass,
        first_payout_day_index=overall_first_payout_day_index,
        first_payout_amount=overall_first_payout_amount,
        payouts=overall_payouts,
        final_balance=balance,
        max_drawdown_pct_reached=overall_max_dd_pct_reached,
        trading_days_count=last_day_idx_reached + 1,
        reset_on_breach=reset_on_breach,
        attempts=attempts,
        total_attempts=len(attempts),
        attempts_passed=sum(1 for a in attempts if a.passed_evaluation),
        attempts_reached_payout=sum(1 for a in attempts if a.reached_first_payout),
    )


def summarize_single_run(result: AccountSimResult) -> dict:
    """
    Section-4-style summary for the single deterministic historical trade
    sequence. Pass/fail/payout rates here are necessarily 0% or 100% since
    there is only one run -- statistical distributions over many possible
    trade sequences are the job of the Monte Carlo engine (section 5/6).
    """
    return {
        "evaluation_pass_pct": 100.0 if result.passed_evaluation else 0.0,
        "evaluation_failure_pct": 100.0 if result.failed and not result.passed_evaluation else 0.0,
        "first_payout_pct": 100.0 if result.reached_first_payout else 0.0,
        "days_to_pass": result.days_to_pass,
        "days_to_first_payout": result.first_payout_day_index,
        "first_payout_amount": result.first_payout_amount,
        "total_payouts": len(result.payouts),
        "total_payout_amount": result.total_payout_amount,
        "final_balance": result.final_balance,
        "max_drawdown_pct": result.max_drawdown_pct_reached,
        "failure_reason": result.failure_reason,
        "trading_days_count": result.trading_days_count,
    }
