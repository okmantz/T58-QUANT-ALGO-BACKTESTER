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

from app.data.trading_day import trading_day, trading_days
from app.prop.account import DAY_LOCKED, FAILED, PropAccount


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
    # P2-7: payout_frequency_days counts TRADE DAYS (days with >= 1 trade),
    # not calendar days -- day indices in simulate_account are ordinals
    # over trade days only, so a "14-day" frequency is 14 trading days.
    payout_frequency_days: int = 14                 # min trade-days between payouts
    required_buffer_pct: float = 0.0                # profit buffer that must be maintained above account_size before payout
    # P2-6 (Topstep-style payout gate): a payout additionally requires at
    # least `winning_days_for_payout` distinct funded-stage days each with
    # day-PnL >= `min_winning_day_profit`. Defaults (5, 0.0) reproduce
    # today's looser behavior in practice (any funded run that reaches
    # payout eligibility almost always has >= 5 non-negative days), while
    # letting a firm-accurate gate be configured (e.g. Topstep: 5 days of
    # $150+ -> winning_days_for_payout=5, min_winning_day_profit=150).
    # NOTE: not strictly byte-identical to the pre-gate code in the
    # pathological case (a payout-eligible run with < 5 non-negative
    # funded-stage days); see simulate_account's funded payout branch.
    winning_days_for_payout: int = 5
    min_winning_day_profit: float = 0.0
    # P2-6 (inactivity rule, e.g. FTMO's 30-day rule): fail the account
    # with reason "inactivity" when more than this many CALENDAR days with
    # no trades pass between two trade days. None (default) = rule off,
    # byte-identical to before this field existed.
    max_inactive_days: int | None = None
    # B5 (v6): funded-stage consistency rule -- checked at every funded
    # payout, mirroring the eval-stage rule's shape: the payout is gated
    # unless best_day_since_baseline / profit_since_baseline <= this
    # percentage. Guarded against divide-by-zero (skipped when
    # profit_since_baseline <= 0). None (default) = rule off,
    # byte-identical to before this field existed. Firms gate payouts
    # (not the eval pass) with this kind of rule -- Apex 50%, Lucid 40%,
    # TopStep 40%, FundedNext 40% (see app.prop.presets).
    funded_consistency_rule_pct: float | None = None
    # B13 (v6): what happens when a session day breaches the daily loss
    # limit. "fail" (default) = the account busts immediately, exactly as
    # before. "lock_day" = the day's remaining trades are skipped and the
    # sim resumes on the next session day (models firms/desks that lock
    # you out for the day instead of closing the account). Validated in
    # __post_init__ -- anything else raises ValueError.
    daily_loss_action: str = "fail"              # "fail" | "lock_day"
    # B15 (v6): WHAT balance the daily-loss-limit percentage is taken of.
    #   "initial" (default): today's behavior, byte-identical -- the limit
    #   is always account_size * daily_loss_limit_pct / 100.
    #   "prior_day_high": the limit is taken of the prior session day's
    #   realized-balance peak (the closest this realized-PnL-only sim can
    #   get to "max(prior session close, equity high at prior midnight)"
    #   -- peak >= close, so the max collapses to the peak).
    #   "ratchet_up": the limit is taken of max(account_size, all prior
    #   session-close balances) -- the base never decreases once it rises.
    # Day 1 (no prior session) always falls back to account_size.
    # Validated in __post_init__ -- anything else raises ValueError.
    daily_loss_base: str = "initial"             # "initial" | "prior_day_high" | "ratchet_up"
    # B14 (v6): evaluation time limit -- fail the account with reason
    # "eval_time_limit" once more than this many CALENDAR days have
    # elapsed since the eval started (attempt start day) without passing.
    # None (default) = no time limit, byte-identical to before.
    max_eval_calendar_days: int | None = None
    # B14 (v6): absolute dollar cap applied to every payout request
    # (on top of payout_cap_pct). None (default) = no dollar cap.
    payout_cap_dollars: float | None = None
    # B14 (v6): maximum number of payouts over the life of one attempt
    # (each attempt is a fresh account, so the cap is per attempt).
    # None (default) = unlimited.
    max_payouts: int | None = None

    def __post_init__(self):
        # B13/B15 (v6): reject misspelled/unknown enum values at
        # construction instead of silently misbehaving mid-simulation.
        if self.daily_loss_action not in ("fail", "lock_day"):
            raise ValueError(
                f"daily_loss_action must be 'fail' or 'lock_day', got {self.daily_loss_action!r}"
            )
        if self.daily_loss_base not in ("initial", "prior_day_high", "ratchet_up"):
            raise ValueError(
                f"daily_loss_base must be 'initial', 'prior_day_high' or 'ratchet_up', "
                f"got {self.daily_loss_base!r}"
            )
        if self.dd_basis not in ("legacy", "realized", "eod", "floating"):
            raise ValueError(
                f"dd_basis must be 'legacy', 'realized', 'eod' or 'floating', got {self.dd_basis!r}"
            )
        if self.daily_loss_basis not in ("realized", "floating"):
            raise ValueError(
                f"daily_loss_basis must be 'realized' or 'floating', got {self.daily_loss_basis!r}"
            )
        if self.trailing_distance_basis not in ("peak", "account"):
            raise ValueError(
                f"trailing_distance_basis must be 'peak' or 'account', got {self.trailing_distance_basis!r}"
            )
    # ------------------------------------------------------------------
    # ACCURACY OVERHAUL (2026-10-07): one account model for every mode.
    # All defaults reproduce the previous behavior exactly; see
    # app.prop.account.PropAccount for the semantics.
    # ------------------------------------------------------------------
    dd_basis: str = "legacy"
    # WHAT equity the drawdown floor trails and is checked against:
    #   "legacy"   previous behavior (drawdown_check_mode / floating_
    #              drawdown_mode decide), realized trade-close balance;
    #   "realized" high-water mark and breach on realized balance after
    #              every trade close;
    #   "eod"      high-water mark moves at END OF DAY only, breach checked
    #              against floating (intraday) equity -- the typical
    #              "end-of-day trailing" futures-firm rule;
    #   "floating" high-water mark follows the INTRADAY equity high, breach
    #              against floating equity -- the "intraday trailing" rule.
    daily_loss_basis: str = "realized"
    # "realized": the day's closed-trade P&L; "floating": includes open P&L
    # (what most futures firms monitor). Combined with daily_loss_action
    # ("fail" = account busts, "lock_day" = liquidate and stop for the day)
    # this is the firm's liquidate-or-fail choice.
    trailing_lock: bool = False
    # Trailing floors that stop rising once they reach the starting balance
    # (+ trailing_lock_offset_pct of it). False = the floor trails forever.
    trailing_lock_offset_pct: float = 0.0
    trailing_distance_basis: str = "peak"
    # "peak" (default, the repo's long-standing convention): floor =
    # peak x (1 - max_drawdown_pct). "account": floor = peak - account_size x
    # max_drawdown_pct, i.e. a fixed dollar distance, which is how most
    # futures firms state a trailing drawdown.
    max_contracts: int | None = None
    # Firm contract cap (informational for the sim; enforced by the engine
    # via RiskConfig.max_contracts when the preset is applied).
    rules_checked_on: str | None = None
    # ISO date the rules were last verified against the firm's rules page
    # (None = never verified). Presets store this so stale rules are visible.

    # P1-3: WHAT balance the drawdown failure checks trail on.
    #   "realized" (default): today's behavior, byte-identical -- checks
    #   run against realized trade-close balance only.
    #   "adverse": degrades the max-drawdown failure check with each
    #   trade's initial risk (see simulate_account's trade_initial_risks
    #   parameter) as a floating-drawdown proxy -- see the comment at the
    #   check itself for exactly what approximation this is.
    # ADAPTATION (audit asked for `drawdown_check_mode: str = "realized"`):
    # that name is already taken by the intrabar/eod WHEN-to-check field
    # above, so this orthogonal WHAT-to-trail axis gets its own name.
    floating_drawdown_mode: str = "realized"        # "realized" | "adverse"
    # -- session-day bucketing (v5) -------------------------------------------
    # All day grouping in this module (daily-loss attribution, winning-day
    # counts, trade-day ordinals, the inactivity gap) uses the shared
    # app.data.trading_day.trading_day() helper, NOT naive UTC midnight:
    # the trading day rolls at `session_roll_hour` (17:00 = 5 PM CT, the
    # Globex roll used by Alpha Futures and most US futures firms).
    # Behavior change vs older revisions: evening PnL between 17:00 and
    # 23:59 CT now attributes to the NEXT session day instead of the
    # calendar (UTC-midnight) day, which changes daily-loss-limit
    # attribution, the 5x$200 winning-day gate, and per-day consistency
    # numerators. Naive timestamps are assumed to be UTC (the state of
    # imported price data); tz-aware timestamps are converted properly.
    session_timezone: str = "America/Chicago"    # tz for session-day bucketing
    session_roll_hour: int = 17                  # hour of day (in session_timezone) when the trading day rolls

    # -- portfolio correlation caps (v5, astra port #5) ------------------------
    # Optional overlay for JOINT N-strategy sims (one simulate_account call
    # over the interleaved trades of several strategies): cap how many
    # positions may be open at once and how many of them may point the same
    # direction. Correlated same-direction exposure is the blow-up vector
    # per-strategy drawdown limits miss -- e.g. five "different" strategies
    # all long NQ into the same flush. Default None/None = overlay OFF:
    # single-strategy sims (and any caller that doesn't pass entry times)
    # are byte-identical to before these fields existed. When enabled, the
    # caller must also pass trade_entry_times (and trade_sides for the
    # direction cap) to simulate_account; a trade that would violate a cap
    # is SKIPPED as if it never happened (blocked live by the desk's risk
    # gate) and counted in AccountSimResult.blocked_trades.
    max_concurrent_positions: int | None = None    # e.g. 6 (astra risk_constants.py:48-50)
    max_same_direction_exposure: int | None = None # e.g. 4 (astra risk_constants.py:48-50)
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
    # P2-7: days_to_pass counts TRADE DAYS (days with >= 1 trade from the
    # attempt's first trade day), not calendar days.
    days_to_pass: int | None
    first_payout_day_index: int | None
    first_payout_amount: float | None
    payouts: list[PayoutEvent] = field(default_factory=list)
    final_balance: float = 0.0
    max_drawdown_pct_reached: float = 0.0
    trading_days_count: int = 0
    # -- portfolio correlation-cap overlay (v5) --------------------------------
    # Trades skipped because they would have violated
    # max_concurrent_positions / max_same_direction_exposure. 0 unless the
    # overlay is enabled (both the cap fields on PropRules AND entry times
    # passed to simulate_account); with the overlay off this is always 0
    # and every other field is byte-identical to before it existed.
    blocked_trades: int = 0
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
    # -- B12-sim (v6): constraint-utilization tracking -------------------------
    # Overall (across all attempts in the chain) best single-day PnL and
    # worst single-day PnL, consumed by summarize_single_run's
    # best_day_profit_pct_of_limit / worst_daily_loss_pct_of_limit fields.
    # 0.0 defaults = byte-neutral for any run with no positive/negative
    # days respectively; nothing here changes any pre-existing field.
    best_day_profit: float = 0.0
    worst_day_pnl: float = 0.0

    @property
    def reached_first_payout(self) -> bool:
        return self.first_payout_day_index is not None

    @property
    def total_payout_amount(self) -> float:
        return sum(p.amount for p in self.payouts)


@dataclass
class DayStructure:
    """The part of simulate_account's bookkeeping that depends ONLY on
    `trade_dates`, never on the P&L values themselves: which trading day
    each trade in the sequence falls on, and which trades are the last of
    their day. (v5: "day" = session day per app.data.trading_day with the
    tz/roll_hour precompute_day_structure was called with -- see the
    behavior-change note on PropRules.session_roll_hour.) The Monte Carlo engine resamples/shuffles P&L VALUES across
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


def precompute_day_structure(
    trade_dates: list,
    *,
    tz: str = "America/Chicago",
    roll_hour: int = 17,
) -> DayStructure:
    """Builds the (dates-only) bookkeeping simulate_account needs, once,
    so it can be reused across many calls that all share the same
    `trade_dates` but different `trade_pnls` (exactly the Monte Carlo
    engine's resampling pattern). See DayStructure's docstring.

    v5: day grouping uses the shared session-day helper
    app.data.trading_day.trading_day() -- the trading day rolls at
    `roll_hour` o'clock in `tz` (defaults 17:00 America/Chicago, the
    futures-session roll), NOT naive UTC midnight. Naive timestamps are
    assumed to be UTC; tz-aware timestamps convert properly.
    """
    # v9.13 speed: vectorized twin of the per-date trading_day() loop
    # (elementwise identical -- see app.data.trading_day.trading_days).
    dates_norm = list(pd.DatetimeIndex(trading_days(trade_dates, tz=tz, roll_hour=roll_hour)))
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
    trade_initial_risks: "list[float] | None" = None,
    trade_entry_times: "list | None" = None,
    trade_sides: "list | None" = None,
    trade_exit_times: "list | None" = None,
    trade_mae: "list[float] | None" = None,
    trade_mfe: "list[float] | None" = None,
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

    trade_initial_risks: optional per-trade initial dollar risk, same
    order/length as trade_pnls (e.g. Trade.intended_risk_dollars, or
    initial_risk * size). Only used when
    rules.floating_drawdown_mode == "adverse" (see P1-3 below); None
    (default) keeps the check a no-op, byte-identical to not passing it.

    trade_entry_times: optional per-trade entry datetimes, same
    order/length as trade_pnls. Only used by the portfolio correlation-cap
    overlay (rules.max_concurrent_positions /
    rules.max_same_direction_exposure); None (default) disables the
    overlay entirely, byte-identical to not passing it.
    trade_sides: optional per-trade side, same order/length as trade_pnls:
    +1/-1 (or "long"/"short" strings). Only used by the
    max_same_direction_exposure cap; None (default) disables that cap.
    trade_exit_times: optional per-trade exit datetimes overriding
    trade_dates for the concurrency-overlap computation. Defaults to
    trade_dates (i.e. trade_dates are the position close times).

    trade_mae / trade_mfe: optional per-trade MEASURED max adverse (<=0) and
    favorable (>=0) excursion in dollars (see Trade.worst_price/mfe_price).
    Only consulted when rules.dd_basis is "eod"/"floating" or
    rules.daily_loss_basis is "floating": they give the account the
    trade's real floating-equity range instead of just its closing P&L, so
    the intraday drawdown rule is checked against what actually happened
    mid-trade. None falls back to the realized P&L as the extreme.

    ACCURACY OVERHAUL (2026-10-07): the per-trade rule evaluation now lives
    in app.prop.account.PropAccount -- the SAME state machine the bar engine
    and the rolling attempt replay use -- and this function is the loop that
    feeds it trades (plus the correlation-cap overlay and the reset chain).
    """
    if len(trade_pnls) == 0:
        return AccountSimResult(
            passed_evaluation=False, failed=False, failure_reason="No trades generated.",
            failure_day_index=None, days_to_pass=None, first_payout_day_index=None,
            first_payout_amount=None, final_balance=rules.account_size,
            reset_on_breach=reset_on_breach, attempts=[], total_attempts=0,
        )

    day_structure = (
        _day_structure
        if _day_structure is not None
        else precompute_day_structure(
            trade_dates, tz=rules.session_timezone, roll_hour=rules.session_roll_hour
        )
    )
    day_index_per_trade = day_structure.day_index_per_trade
    is_last_of_day = day_structure.is_last_of_day
    day_dates = day_structure.day_dates
    n = len(trade_pnls)
    static_floor = rules.account_size * (1 - rules.max_drawdown_pct / 100.0)

    # --- portfolio correlation-cap overlay (v5, astra port #5) --------------
    # Off unless a cap is configured on `rules` AND entry times are given.
    # `_correlation_caps_active` is False in every single-strategy / legacy
    # call, so the inner trade loop below is byte-identical to before.
    _caps_active = (
        (rules.max_concurrent_positions is not None or rules.max_same_direction_exposure is not None)
        and trade_entry_times is not None
    )
    _blocked_trades = 0

    def _side_of(value) -> int | None:
        """Normalize a trade side to +1 (long) / -1 (short) / None (unknown)."""
        if isinstance(value, str):
            v = value.strip().lower()
            if v.startswith("long"):
                return 1
            if v.startswith("short"):
                return -1
            return None
        if value is None:
            return None
        if value > 0:
            return 1
        if value < 0:
            return -1
        return None

    # Positions already taken (and not skipped) in the current attempt:
    # list of (entry_time, exit_time, side) used for the overlap test.
    _open_pool: list = []

    attempts: list[AttemptRecord] = []
    overall_payouts: list[PayoutEvent] = []
    overall_passed_evaluation = False
    overall_days_to_pass = None
    overall_first_payout_day_index = None
    overall_first_payout_amount = None
    overall_max_dd_pct_reached = 0.0
    last_day_idx_reached = -1
    overall_best_day_profit = 0.0             # B12-sim (v6): run-wide best single-day PnL
    overall_worst_day_pnl = 0.0               # B12-sim (v6): run-wide worst single-day PnL

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
        # --- fresh account for this attempt ---
        attempt_start_day_idx = day_index_per_trade[i]
        acct = PropAccount(rules, start_day_index=attempt_start_day_idx, day_dates=day_dates)
        # correlation-cap overlay: each attempt is a fresh account, so the
        # open-position pool resets at the attempt boundary.
        _open_pool = []

        j = i
        while j < n:
            # --- correlation-cap overlay: a blocked trade never happened ---
            if _caps_active:
                entry_j = pd.Timestamp(trade_entry_times[j])
                exit_j = (
                    pd.Timestamp(trade_exit_times[j])
                    if trade_exit_times is not None
                    else pd.Timestamp(trade_dates[j])
                )
                open_sides = []
                for (e_k, x_k, s_k) in _open_pool:
                    if e_k <= entry_j < x_k:
                        open_sides.append(s_k)
                blocked = (
                    rules.max_concurrent_positions is not None
                    and len(open_sides) >= rules.max_concurrent_positions
                )
                if not blocked and rules.max_same_direction_exposure is not None and trade_sides is not None:
                    side_j = _side_of(trade_sides[j])
                    same_dir = sum(1 for s in open_sides if s is not None and s == side_j)
                    blocked = same_dir >= rules.max_same_direction_exposure
                if blocked:
                    # Live, the desk's risk gate would have rejected this
                    # entry -- the sim skips it wholesale: no balance move,
                    # no day attribution, no winning-day credit.
                    _blocked_trades += 1
                    j += 1
                    continue
                _open_pool.append(
                    (entry_j, exit_j, _side_of(trade_sides[j]) if trade_sides is not None else None)
                )

            cur_day_idx = day_index_per_trade[j]
            risk_j = None
            if trade_initial_risks is not None:
                risk_j = float(trade_initial_risks[j]) if j < len(trade_initial_risks) else 0.0
            res = acct.on_trade_close(
                trade_pnls[j], cur_day_idx,
                is_last_of_day=is_last_of_day[j],
                trade_initial_risk=risk_j,
                mae=(float(trade_mae[j]) if trade_mae is not None and j < len(trade_mae) else None),
                mfe=(float(trade_mfe[j]) if trade_mfe is not None and j < len(trade_mfe) else None),
            )
            if res == FAILED:
                break
            if res == DAY_LOCKED:
                # B13 (v6): skip the rest of this session day and resume on
                # the next one -- the account survives the breach. Skipped
                # trades are treated as never taken (no balance move, no
                # day attribution); the breached day's close is recorded so
                # DLL-base bookkeeping stays consistent.
                while j < n and day_index_per_trade[j] == cur_day_idx:
                    j += 1
                continue
            j += 1

        # --- attempt finished: either it busted (break above) or it rode
        # out every remaining trade without failing (j == n) ---
        if acct.n_trades > 0:
            last_day_idx_reached = acct.last_day_index_reached
        overall_max_dd_pct_reached = max(overall_max_dd_pct_reached, acct.max_dd_pct_reached)
        # B12-sim (v6): fold this attempt's day extremes into the
        # run-wide constraint-utilization tracking.
        if acct.daily_pnl:
            overall_best_day_profit = max(overall_best_day_profit, max(acct.daily_pnl.values()))
            overall_worst_day_pnl = min(overall_worst_day_pnl, min(acct.daily_pnl.values()))
        balance = acct.balance
        failed = acct.failed
        failure_reason = acct.failure_reason
        failure_day_index = acct.failure_day_index
        payouts_this_attempt = [
            PayoutEvent(
                day_index=p.day_index, date=str(day_dates[p.day_index].date()),
                amount=p.amount, balance_after=p.balance_after,
            )
            for p in acct.payouts
        ]
        attempts.append(AttemptRecord(
            attempt_index=attempt_index,
            start_day_index=attempt_start_day_idx,
            end_day_index=acct.attempt_last_day_index,
            days_used=max(0, acct.attempt_last_day_index - attempt_start_day_idx + 1),
            passed_evaluation=acct.passed_evaluation,
            failed=failed,
            failure_reason=failure_reason,
            reached_first_payout=acct.first_payout_day_index is not None,
            payout_amount=sum(p.amount for p in payouts_this_attempt),
            ending_balance=balance,
        ))
        overall_payouts.extend(payouts_this_attempt)
        if not overall_passed_evaluation and acct.passed_evaluation:
            overall_passed_evaluation = True
            overall_days_to_pass = acct.days_to_pass
        if overall_first_payout_day_index is None and acct.first_payout_day_index is not None:
            overall_first_payout_day_index = acct.first_payout_day_index
            overall_first_payout_amount = acct.first_payout_amount

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
        blocked_trades=_blocked_trades,
        reset_on_breach=reset_on_breach,
        attempts=attempts,
        total_attempts=len(attempts),
        attempts_passed=sum(1 for a in attempts if a.passed_evaluation),
        attempts_reached_payout=sum(1 for a in attempts if a.reached_first_payout),
        best_day_profit=overall_best_day_profit,
        worst_day_pnl=overall_worst_day_pnl,
    )


def summarize_single_run(result: AccountSimResult, rules: "PropRules | None" = None) -> dict:
    """
    Section-4-style summary for the single deterministic historical trade
    sequence. Pass/fail/payout rates here are necessarily 0% or 100% since
    there is only one run -- statistical distributions over many possible
    trade sequences are the job of the Monte Carlo engine (section 5/6).

    rules: the PropRules the run was simulated under. Optional -- when
    omitted, the three B12-sim constraint-utilization fields
    (best_day_profit_pct_of_limit, worst_daily_loss_pct_of_limit,
    max_dd_pct_of_limit) are None, since the limits they are ratios
    against come from the rules. Every pre-existing caller passes just
    `result` and keeps working unchanged.

    B12-sim (v6): the three utilization fields are float ratios where
    1.0 = exactly at the limit, None when the corresponding rule is
    disabled (or the ratio is undefined):
      - best_day_profit_pct_of_limit: best single-day profit / the
        day-profit amount that would breach consistency, where the breach
        amount = consistency% x the run's total net profit
        ((final_balance - account_size) + total_payout_amount). The eval
        rule (consistency_rule_pct) is preferred when set; otherwise the
        funded rule (funded_consistency_rule_pct). None when neither rule
        is set, or when total net profit <= 0 (the ratio is undefined).
      - worst_daily_loss_pct_of_limit: |worst single-day PnL| / the
        daily-loss-limit dollar amount, computed off the initial base
        (an approximation when daily_loss_base is "prior_day_high" or
        "ratchet_up", whose limit moves day to day). None when the DLL
        is effectively disabled (daily_loss_limit_pct >= 100, the
        presets' "no DLL" convention) or non-positive.
      - max_dd_pct_of_limit: max_drawdown_pct_reached / max_drawdown_pct.
        None when max_drawdown_pct <= 0.
    """
    summary = {
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

    best_day_profit_pct_of_limit = None
    worst_daily_loss_pct_of_limit = None
    max_dd_pct_of_limit = None
    if rules is not None:
        _consistency_pct = (
            rules.consistency_rule_pct
            if rules.consistency_rule_pct is not None
            else rules.funded_consistency_rule_pct
        )
        if _consistency_pct is not None:
            _total_net_profit = (result.final_balance - rules.account_size) + result.total_payout_amount
            if _total_net_profit > 0 and result.best_day_profit > 0:
                _breach_amount = _consistency_pct / 100.0 * _total_net_profit
                if _breach_amount > 0:
                    best_day_profit_pct_of_limit = result.best_day_profit / _breach_amount
        if 0.0 < rules.daily_loss_limit_pct < 100.0:
            _dll_dollars = rules.account_size * rules.daily_loss_limit_pct / 100.0
            if _dll_dollars > 0:
                worst_daily_loss_pct_of_limit = abs(min(result.worst_day_pnl, 0.0)) / _dll_dollars
        if rules.max_drawdown_pct > 0:
            max_dd_pct_of_limit = result.max_drawdown_pct_reached / rules.max_drawdown_pct

    summary["best_day_profit_pct_of_limit"] = best_day_profit_pct_of_limit
    summary["worst_daily_loss_pct_of_limit"] = worst_daily_loss_pct_of_limit
    summary["max_dd_pct_of_limit"] = max_dd_pct_of_limit
    return summary
