"""
Bar-by-bar trade execution simulator.

Consumes OHLCV data + a standardized signal series (-1/0/1) + risk config
and produces a discrete trade list. Entries occur on the bar the signal
changes, filled at open[i + risk.entry_fill_lag_bars] (default: the NEXT
bar's open -- no live trader can fill at the close that produced the
signal), adjusted for spread/slippage; each open trade is then walked
forward bar-by-bar checking for stop-loss / take-profit intrabar hits
(using high/low), a time-invalidation stop, or a signal-driven exit
(which also fills at open[i + entry_fill_lag_bars]).

This is intentionally a straightforward, transparent simulation appropriate
for an MVP -- no partial fills, no multi-leg positions, one open trade at a
time (consistent with the standardized long/flat/short signal model).

EXPLICIT EXECUTION ASSUMPTIONS (documented here so none of these are silent):
  - Stop-vs-target ordering on a bar whose range spans both levels always
    resolves to the stop, never the target (see _resolve_intrabar_exit) --
    a deliberate conservative bias, not a guess at which came first.
  - A resting stop the bar gapped straight through fills at the bar's
    open (worse than the stop price), never at the stop level itself.
  - Same-bar stop-out-then-reentry: if a position closes intrabar (stop,
    target, or forced daily-loss close) and the strategy's signal for
    that same bar is still non-flat, a fresh position in the same
    direction is allowed to open at that bar's own close. This was the
    ORIGINAL, default behavior (RiskConfig.reentry_cooldown_bars == 0),
    preserved exactly for backward compatibility -- but the default is
    now 1 (no same-bar reentry; earliest reentry is the next bar), and
    the original behavior is still available as an explicit,
    configurable choice: set reentry_cooldown_bars to 0 to opt back
    into it, or to N>0 to block any new entry for N bars after a
    position closes, regardless of what the signal says. See
    RiskConfig.reentry_cooldown_bars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskState, volatility_percentile_series
from app.backtest.risk import RiskConfig, SizingDecision

try:
    # B2-3 (session day): shared trading-day helper (created by a sibling
    # v5 worker). Guarded so this module still imports/runs if that module
    # hasn't been merged yet -- in that case run_execution falls back to
    # the previous wall-clock calendar-day grouping.
    from app.data.trading_day import trading_day
except ImportError:  # pragma: no cover - only until the sibling module lands
    trading_day = None


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int          # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str        # "stop_loss" | "take_profit" | "signal" | "end_of_data"
    commission: float
    equity_after: float
    initial_risk: float | None = None  # |entry - stop| in raw price units, at entry time
    adaptive_risk_multiplier: float = 1.0     # position-size multiplier in effect when this trade was OPENED
    adaptive_risk_rules_active: tuple = ()    # human-readable labels of whichever adaptive-risk rule(s) fired
    intended_risk_dollars: float | None = None
    # RISK-RECON: the raw dollar figure risk.risk_amount(equity_at_entry) targeted
    # for this trade -- i.e. literally "how much you told the system you're
    # willing to risk" (RiskConfig.risk_value, e.g. 0.5% of a $50k account =
    # $250), computed BEFORE any max_position_size cap or adaptive-risk
    # throttle shrinks the actual position. Compare against
    # `initial_risk * size` (the trade's ACTUAL dollar risk at its own stop,
    # given the size that was actually taken) to see whether -- and by how
    # much -- a cap/throttle/rounding made this trade risk less (or, via a
    # gap-through fill on the realized `pnl` itself, more) than what was
    # configured. See app.backtest.statistics.compute_risk_reconciliation
    # for the aggregated view surfaced in every report. None only for a
    # trade whose sizing produced no finite risk_amount (equity <= 0).
    worst_price: float | None = None
    # P2-5: the most ADVERSE price the trade's bar extremes ever reached
    # while the position was open (low for a long, high for a short) --
    # the mirror of the engine-internal best_price tracking. Feeds
    # per-trade MAE (max adverse excursion in currency) in
    # app.backtest.statistics.compute_statistics. None for trades built
    # by paths that don't track it (e.g. the vectorized Stage-1 fast
    # path, which settles without intrabar extremes).
    mfe_price: float | None = None
    # Part A port #1: the most FAVORABLE price the trade's bar extremes ever
    # reached while the position was open (high for a long, low for a
    # short) -- mirrors best_price (engine-internal favorable extreme) the
    # way worst_price above mirrors the adverse one. Feeds per-trade MFE
    # (max favorable excursion) in app.analysis.exit_quality. None for
    # trades built by paths that don't track it.
    exit_cause: str | None = None
    # Part A port #1: machine-verified exit taxonomy, refined from the raw
    # exit_reason at settle time -- one of "stop_loss" | "take_profit" |
    # "trailing_stop" | "breakeven" | "time_stop" | "signal_exit" |
    # "session_close" | "daily_loss_close" | "end_of_data". Lets
    # exit-quality analysis attribute results per exit mechanism instead of
    # lumping every stop fill together. None only for trades built by paths
    # that don't tag it.
    sized_above_risk_target: bool = False
    # Part C fix 2: True when this trade was opened by the dead-lock rescue
    # under RiskConfig.allow_single_contract_minimum at/above
    # initial_balance -- i.e. its 1-contract size may exceed the configured
    # risk_value % for that entry. False for every normally-sized trade.
    risk_at_stop_dollars: float | None = None
    # ACCURACY OVERHAUL (2026-10-07): the worst-case $ loss at THIS trade's
    # stop, exit-side spread/slippage and commission included -- exactly
    # what the sizing budget covered (RiskConfig.size_for_stop). A stopped
    # trade that loses about this much is on budget; compute_risk_
    # reconciliation measures overshoot against it instead of against the
    # bare stop distance (which made 100% of trades "overshoot").
    stop_capped: bool = False          # fit_stop shrank this trade's stop to fit the budget
    sizing_mode: str | None = None     # RiskConfig.sizing_mode in force when sized
    used_micro: bool = False           # micro_fallback swapped in the micro contract
    contracts: float | None = None     # whole contracts of the contract actually traded
    attempt_id: int = 0                # which purchased evaluation/funded account this trade belongs to
    fill_resolution: str = "bar"       # "bar" (stop/target from the bar's high/low) or "intrabar" (finer-bar replay)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_time"] = str(self.entry_time)
        d["exit_time"] = str(self.exit_time)
        return d


from app.backtest.risk import (  # noqa: E402  (grouped with the sizing constants on purpose)
    SKIP_COSTS_EXCEED_BUDGET, SKIP_NO_MICRO, SKIP_STOP_CAP_TOO_TIGHT,
    SKIP_STOP_TOO_WIDE, SKIP_TOO_WIDE_FOR_MICRO,
)

# Sizing skip reasons that mean "the budget cannot afford this signal" (as
# opposed to a degenerate stop / no money) -- they feed the sizing-halt
# diagnostic and the preflight's skip-rate gate.
_SIZING_SKIP_REASONS = frozenset({
    SKIP_STOP_TOO_WIDE, SKIP_COSTS_EXCEED_BUDGET, SKIP_STOP_CAP_TOO_TIGHT,
    SKIP_TOO_WIDE_FOR_MICRO, SKIP_NO_MICRO,
})

DEFAULT_STOP_PCT_OF_PRICE = 0.01  # 1% of entry price, used only when a strategy defines no stop at all

# C6 (market-impact guardrail): entries above this many whole contracts
# fill at the quoted price with zero market impact assumed -- the engine
# warns (RuntimeWarning) instead of silently pretending a 30+ lot sweep
# doesn't move the book. Deliberately a module constant, not a
# RiskConfig field: the threshold is visible at the warning site and
# there is no impact model to configure.
MAX_CONTRACTS_BEFORE_IMPACT_WARN = 30


def sizing_mode_is_contract_skip(risk) -> bool:
    """The bounded dead-lock guard applies only to the plain 'skip' mode;
    fit_stop / micro_fallback already keep a small account trading."""
    return getattr(risk, "sizing_mode", "skip") == "skip"


def run_execution(
    df: pd.DataFrame,
    signals: pd.Series,
    risk: RiskConfig,
    stop_loss_pips: float | None,
    take_profit_pips: float | None,
    stop_loss_distance: pd.Series | None = None,
    take_profit_distance: pd.Series | None = None,
    trailing_stop_distance: pd.Series | None = None,
    breakeven_trigger_r: float | None = None,
    partial_exit_config: dict | None = None,
    adaptive_risk: AdaptiveRiskConfig | None = None,
    intrabar_df: pd.DataFrame | None = None,
    attempt_mode: str = "chain",
    stop_on_pass: bool = False,
) -> tuple[list[Trade], pd.DataFrame]:
    """
    Returns (trades, equity_curve_df) where equity_curve_df has columns
    [timestamp, equity] for every bar in df.

    stop_loss_distance / take_profit_distance: optional per-bar distances
    in raw price units (e.g. an ATR-multiple stop). When provided, these
    take precedence over the fixed stop_loss_pips/take_profit_pips for
    that entry bar.

    trailing_stop_distance: optional per-bar distance in raw price units.
    The distance is captured once at trade entry and then used to ratchet
    the stop toward price as the trade moves favorably; it never widens.

    breakeven_trigger_r: once open profit reaches this multiple of the
    trade's initial risk (entry-to-stop distance), the stop is moved to
    the entry price (only ever tightened, never loosened).

    adaptive_risk: optional declarative money-management overlay (see
    app.backtest.adaptive_risk) -- scales the SIZE of each new entry by
    whatever multiplier its rules currently imply (consecutive losses,
    today's realized P&L, progress toward a profit target). Evaluated
    fresh at every entry decision using only trade outcomes ALREADY
    realized as of that bar, so it introduces no lookahead. None/disabled
    means every entry uses its full nominal size, unchanged from before
    this parameter existed.

    partial_exit_config: optional {"r_multiple", "fraction",
    "move_stop_to_breakeven"} dict (see StrategyResult.partial_exit's own
    docstring). None/omitted -- the default -- reproduces every backtest
    run before this parameter existed, byte for byte; this is purely
    additive and never changes behavior for a strategy that doesn't set it.

    intrabar_df: optional FINER-timeframe OHLC frame (e.g. 1-minute bars
    under a 1-hour strategy frame) with a `timestamp` column. Only used when
    risk.intrabar_replay is True: while a trade is open, each strategy bar's
    stop / target / break-even / trailing / floating-floor decisions are
    resolved by walking the finer bars inside that bar IN TIME ORDER, so a
    bar that touched both the stop and the target books whichever really
    traded first instead of always booking the loss.

    attempt_mode / stop_on_pass (only with risk.account_model == "prop"):
    "chain" (default) starts a fresh purchased account on the bar after one
    ends and keeps going to the last bar, labelling every trade and equity
    row with its attempt_id; "single" ends the run when the first attempt
    ends (pass, bust or time limit) -- what the rolling attempt replay
    uses. stop_on_pass ends an attempt the moment the evaluation target is
    met instead of running on into the funded stage.
    """
    n = len(df)
    equity = risk.initial_balance
    trades: list[Trade] = []
    adaptive_state = AdaptiveRiskState(initial_balance=risk.initial_balance)

    # Rolling ATR (14-bar, simple True Range mean) purely as a volatility
    # yardstick for the fixed-pips-vs-instrument sanity check below -- this
    # is intentionally independent of anything a strategy itself computes.
    # See the pip_scale_mismatch check for why a price-ratio band alone
    # isn't enough: a fixed-pips stop can land just inside that ratio band
    # (e.g. ~0.03% of price) yet still be a small fraction of a genuinely
    # volatile instrument's typical bar-to-bar range -- an index at
    # several thousand points per share with a triple-digit-point ATR is
    # the textbook case. Comparing the stop distance to the instrument's
    # own recent ATR catches that; comparing it only to price does not.
    # UPGRADE (speed): this used to build three full-length pandas Series,
    # pd.concat() them into a temporary (n, 3) DataFrame, then reduce with
    # .max(axis=1) -- on 2M+ rows that's several redundant Series/DataFrame
    # allocations plus a row-wise (not columnar) max, purely to compute a
    # single elementwise "biggest of 3 numbers" per bar. Plain numpy
    # elementwise np.maximum() does the identical calculation without ever
    # materializing an (n, 3) DataFrame, and the subsequent EWM-free simple
    # rolling mean is left on the (much cheaper, already a plain ndarray)
    # pandas Series just for its rolling-window implementation.
    _tr_high = df["high"].to_numpy(dtype=float)
    _tr_low = df["low"].to_numpy(dtype=float)
    _close_arr = df["close"].to_numpy(dtype=float)
    _tr_prev_close = np.empty_like(_tr_high)
    _tr_prev_close[0] = np.nan
    _tr_prev_close[1:] = _close_arr[:-1]
    _true_range = np.maximum(
        _tr_high - _tr_low,
        np.maximum(np.abs(_tr_high - _tr_prev_close), np.abs(_tr_low - _tr_prev_close)),
    )
    _atr_for_mismatch_check = (
        pd.Series(_true_range).rolling(14, min_periods=1).mean().to_numpy()
    )
    # Only computed when a rule actually needs it (see
    # AdaptiveRiskConfig.uses_volatility_trigger) -- reuses the SAME ATR
    # series above rather than computing a second one, so the volatility-
    # percentile throttle and the pip/ATR-scale-mismatch check are always
    # looking at identical numbers.
    _vol_percentile_arr = None
    if adaptive_risk is not None and adaptive_risk.uses_volatility_trigger():
        _vol_percentile_arr = volatility_percentile_series(
            _atr_for_mismatch_check, lookback_bars=adaptive_risk.volatility_lookback_bars,
        )

    open_trade: dict | None = None
    fallback_stop_count = 0
    zero_size_contract_floor_count = 0  # see RiskConfig.sizing_floored_to_zero_contracts
    pip_scale_mismatch_count = 0
    pip_scale_mismatch_worst_ratio = None  # smallest (stop_distance / entry_price) seen, for the warning
    atr_scale_mismatch_count = 0
    atr_scale_mismatch_worst_ratio = None  # smallest (stop_distance / ATR) seen, for the warning
    gap_loss_count = 0
    clamped_loss_count = 0
    account_blown = False
    account_blown_at = None
    reset_events: list[dict] = []  # populated when risk.reset_on_breach and/or risk.reset_on_target is True
    # kind="breach" entries: account crossed the loss floor. kind="payout"
    # entries: account crossed the profit target and had the excess
    # withdrawn (see risk.reset_on_target / risk.profit_target_pct
    # below). Sharing one list keeps drawdown segmentation
    # (app.backtest.statistics._reset_segment_ids) correct across BOTH
    # kinds of "this account's equity reference point just changed"
    # event with no separate code path; the "kind" field is what lets
    # reporting tell a good payout apart from a bad breach.
    payout_target_baseline = risk.initial_balance  # advances by profit_target_pct-of-initial_balance after each payout
    daily_limit_amount = (
        risk.initial_balance * (risk.daily_loss_limit_pct / 100.0)
        if risk.daily_loss_limit_pct is not None
        else None
    )
    blown_floor = risk.account_blown_floor()  # None if no floor configured
    if getattr(risk, "account_model", "legacy") == "prop":
        daily_limit_amount = None   # the PropAccount owns the daily-loss rule
        blown_floor = None          # ... and the drawdown rule
    # UPGRADE (2026-09-29, "trade until the data ends"): unless a caller
    # explicitly opts into the legacy permanent halt (RiskConfig.halt_on_breach),
    # a breach starts a fresh account and the run keeps trading to the last bar.
    halt_on_breach = bool(getattr(risk, "halt_on_breach", False))
    min_contract_rescue_count = 0
    attempt_id = 0                      # which purchased account the loop is currently trading
    # ---- PROP ACCOUNT MODE (risk.account_model == "prop") ----------------
    # The bar engine feeds ONE PropAccount per attempt (the same rule engine
    # simulate_account / the rolling replay / Monte Carlo use), so trailing /
    # static drawdown, the daily loss limit, the profit target, min days,
    # consistency, eval time limit, inactivity and payouts mean the same
    # thing everywhere. Every attempt is a fresh purchased account.
    prop_mode = getattr(risk, "account_model", "legacy") == "prop"
    acct = None
    prop_rules = None
    prop_day_map: dict = {}             # session-day idx -> active-day idx
    prop_day_dates: list = []           # active-day idx -> date (for calendar rules)
    prop_pending: str | None = None     # "failed" / "day_locked" raised inside _settle_exit
    prop_locked_bar_date = -1
    prop_attempts: list[dict] = []
    prop_payouts_seen = 0
    prop_done = False
    prop_attempt_start_time = None
    if prop_mode:
        from app.prop.account import PropAccount, coerce_rules, DAY_LOCKED as _DAY_LOCKED, FAILED as _FAILED
        prop_rules = coerce_rules(getattr(risk, "prop_account_rules", None))
        if prop_rules is None:
            raise ValueError("risk.account_model='prop' requires risk.prop_account_rules")
        if abs(float(prop_rules.account_size) - float(risk.initial_balance)) > 1e-6:
            import warnings
            warnings.warn(
                f"prop rules account_size ({prop_rules.account_size:,.2f}) != risk.initial_balance "
                f"({risk.initial_balance:,.2f}); the firm's account size is used for the run.",
                RuntimeWarning,
            )
            equity = float(prop_rules.account_size)
        if getattr(prop_rules, "max_contracts", None):
            from dataclasses import replace as _dc_replace
            _mc = int(prop_rules.max_contracts)
            if risk.max_contracts is not None:
                _mc = min(_mc, int(risk.max_contracts))
            risk = _dc_replace(risk, max_contracts=_mc)
    sizing_considered_count = 0         # entries that reached the sizing step
    sizing_skip_reasons: dict = {}      # skip_reason -> count (see risk.SKIP_*)
    stop_capped_count = 0               # fit_stop shrank the stop to fit the budget
    micro_fallback_count = 0            # micro_fallback traded the micro contract
    adaptive_probe_count = 0  # entries kept at the 1-contract minimum instead of being throttled below one lot (2026-10-06 stall fix)
    _adaptive_state_day = -1  # day_idx the adaptive state's daily accumulators were last rolled to (2026-10-06 stall fix)
    blocked_daily_limit_count = 0
    blocked_cooldown_count = 0
    blocked_max_trades_count = 0
    blocked_halted_count = 0
    blocked_adaptive_zero_count = 0
    blocked_news_blackout_count = 0  # P2-6: entries refused on a news-blackout date
    blocked_weekend_hold_count = 0   # P2-6: entries refused while the weekend-hold block is latched
    weekend_close_count = 0          # P2-6: positions force-closed on a week's last bar
    weekend_entry_block = False      # P2-6: latched at a week-end close, cleared on Monday
    last_entry_bar_idx = -1

    # EXEC-002: bar index the most recently CLOSED (fully closed, not
    # partial) position exited on -- -10**9 sentinel so the very first
    # entry of the run is never blocked by "no previous close yet". Only
    # `_settle_exit` (a full close) updates this; `_settle_partial_exit`
    # deliberately does not, since the position it's called from is
    # still open, not re-entered. With risk.reentry_cooldown_bars == 0,
    # `i - last_close_bar_idx >= 0` is true the instant a position
    # closes, reproducing the original (no-cooldown) behavior exactly --
    # the current default is 1 (blocks the same bar's close only); set 0
    # explicitly to opt back into the original unconditional behavior.
    last_close_bar_idx = -10 ** 9

    def _clamp_loss(pnl_value: float, equity_at_entry: float) -> float:
        """A single trade's loss can never realistically exceed what the
        account actually has to lose (negative-balance protection) or a
        configured hard per-trade ceiling -- see RiskConfig.max_trade_loss.
        Wins are never touched."""
        nonlocal clamped_loss_count
        if pnl_value >= 0 or not math.isfinite(pnl_value):
            return pnl_value
        cap = risk.max_trade_loss(equity_at_entry)
        if cap and -pnl_value > cap:
            clamped_loss_count += 1
            return -cap
        return pnl_value

    sig = signals.values
    ts = df["timestamp"].values
    # VAL-006 fix: `.values` on a timezone-AWARE column silently drops to a
    # bare numpy datetime64 array with no tz attached at all -- numpy has
    # no tz-aware datetime dtype, so this is a lossy conversion even
    # though the underlying instants stay correct. Every Trade.entry_time/
    # exit_time built from `ts[i]` below used to therefore come out
    # tz-NAIVE whenever the input df was tz-aware (true of most real
    # vendor/broker feeds), while other code re-deriving timestamps
    # straight from the original DataFrame (e.g.
    # app.validation.regime_matrix's date-to-trade lookup) stayed
    # tz-aware -- comparing the two raises "Cannot compare tz-naive and
    # tz-aware datetime-like objects" and silently disables that
    # diagnostic. `_ts_tz` captures the ORIGINAL column's tz (None for an
    # already-naive column, in which case _restore_tz below is a no-op)
    # so every timestamp this engine produces matches the input data's
    # tz-awareness instead of losing it on the way through this numpy
    # round-trip.
    _ts_tz = getattr(df["timestamp"].dtype, "tz", None)

    # ---- 1-minute (finer-bar) fill-order replay -------------------------
    # A strategy bar that touched BOTH the resting stop and the target does
    # not tell us which traded first. With risk.intrabar_replay and a finer
    # frame we walk the finer bars inside [ts[i], ts[i+1]) in time order.
    _ib_on = False
    _ib_ts = _ib_hi = _ib_lo = None
    if getattr(risk, "intrabar_replay", False) and intrabar_df is not None and len(intrabar_df) > 0:
        _ibd = intrabar_df.sort_values("timestamp")
        _ibt = pd.to_datetime(_ibd["timestamp"])
        if getattr(_ibt.dt, "tz", None) is not None:
            _ibt = _ibt.dt.tz_convert("UTC").dt.tz_localize(None)
        _ib_ts = _ibt.values.astype("datetime64[ns]")
        _ib_hi = _ibd["high"].to_numpy(dtype=np.float64)
        _ib_lo = _ibd["low"].to_numpy(dtype=np.float64)
        _ib_on = True
    _bar_span = None
    if _ib_on and n > 1:
        _d = np.diff(ts.astype("datetime64[ns]"))
        _bar_span = np.median(_d) if len(_d) else None

    def _intrabar_first_hit(direction_: int, stop_, take_, i_: int):
        """'stop' / 'take' / None: which resting level traded first inside
        bar i_ according to the finer frame. Same finer bar touching both
        resolves to the stop (conservative). None = finer data does not
        cover the bar (caller falls back to the bar-level rule)."""
        if not _ib_on or stop_ is None or take_ is None:
            return None
        t0 = ts[i_].astype("datetime64[ns]")
        t1 = ts[i_ + 1].astype("datetime64[ns]") if i_ + 1 < n else (t0 + _bar_span if _bar_span is not None else None)
        if t1 is None:
            return None
        a = int(np.searchsorted(_ib_ts, t0, side="left"))
        b = int(np.searchsorted(_ib_ts, t1, side="left"))
        if b <= a:
            return None
        hi = _ib_hi[a:b]
        lo = _ib_lo[a:b]
        if direction_ == 1:
            hs, ht = lo <= stop_, hi >= take_
        else:
            hs, ht = hi >= stop_, lo <= take_
        any_s, any_t = hs.any(), ht.any()
        if not any_s and not any_t:
            return None
        fs = int(np.argmax(hs)) if any_s else len(hs) + 1
        ft = int(np.argmax(ht)) if any_t else len(ht) + 1
        return "take" if ft < fs else "stop"

    def _restore_tz(ts_value) -> pd.Timestamp:
        """Rebuilds a `pd.Timestamp` from a raw `ts[i]` numpy datetime64
        value with the original column's tz reattached. A tz-aware pandas
        datetime64 array is always stored internally as UTC instants, so
        localizing to UTC first and then converting recovers the exact
        original instant and label -- this is not a guess, it is the
        inverse of the tz-stripping `.values` does above."""
        t = pd.Timestamp(ts_value)
        return t.tz_localize("UTC").tz_convert(_ts_tz) if _ts_tz is not None else t

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    # UPGRADE (2026-09-03, speed): this used to be
    # `bar_date = pd.Timestamp(ts[i]).normalize()` computed FRESH every
    # single iteration of the per-bar loop below -- constructing a
    # pd.Timestamp object and normalizing it is one of the more expensive
    # things you can do per-element in pandas, and this loop runs on
    # every backtest in the app (Run & Report, every Search Lab stage,
    # every GA generation, every walk-forward fold) -- often 1,000+ times
    # per Full Pipeline run alone. bar_date is only ever used below as a
    # dict key (pnl_today / trades_today), so a plain numpy datetime64
    # day-floor value works identically as a dict key and is computed
    # ONCE, vectorized, outside the loop instead of n times inside it.
    # P2-3 (UTC-day fix): `ts` above was tz-stripped to UTC, so
    # normalizing it directly keys max_trades_per_day / pnl_today_sum /
    # the daily-loss circuit breaker off UTC days (19:00-19:00 CT for
    # Chicago-aware data). Re-attach the input column's original tz
    # first so "a day" is the wall-clock calendar day the data was
    # recorded in -- the same day app.backtest.statistics'
    # _periodic_max_drawdown attributes P&L to.
    # B2-3 (session day, 2026-10-04): supersedes the P2-3 wall-clock day.
    # Naive timestamps are assumed UTC (see app.data.trading_day),
    # converted to America/Chicago, and the day rolls at 17:00 CT --
    # matching how prop firms (e.g. the standard 17:00 CT futures day roll) actually
    # attribute a trading day for the daily-loss limit, the 5x$200
    # winning-day count, and consistency numerators. Every per-day
    # structure below (trades_today_count, pnl_today_sum, the daily-loss
    # circuit breaker, news blackout, weekend hold) keys off this SESSION
    # day. Falls back to the P2-3 wall-clock calendar day if the
    # trading_day module isn't importable yet (guarded import at top).
    if trading_day is not None:
        _ts_for_days = pd.DatetimeIndex(ts)
        if _ts_tz is not None:
            _ts_for_days = _ts_for_days.tz_localize("UTC").tz_convert(_ts_tz)
        bar_dates = np.array(
            [trading_day(t, tz="America/Chicago", roll_hour=17) for t in _ts_for_days],
            dtype=object,
        )
    else:
        _ts_for_days = df["timestamp"]
        if _ts_tz is not None:
            _ts_for_days = pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert(_ts_tz)
        bar_dates = pd.DatetimeIndex(_ts_for_days).normalize().to_numpy()

    # UPGRADE (speed): trades_today / pnl_today used to be plain dicts keyed
    # by the numpy.datetime64 in bar_dates, with a hash + dict lookup (get(),
    # __contains__, __setitem__) on every single bar of the loop below --
    # for a 2M+ row 1-minute dataset that's several million Python-level
    # dict operations that exist purely to answer "which trading day is this
    # bar in, and what's this day's running count/P&L so far". Since every
    # bar already belongs to exactly one of a small, fixed number of
    # calendar days, np.unique() converts bar_dates into a dense integer
    # day-index per bar ONCE, up front, and the running per-day state
    # becomes plain numpy arrays indexed by that integer -- an array index
    # is materially cheaper than a dict lookup, and it also means these
    # never grow a Python dict entry-by-entry across a multi-year run.
    _unique_days, day_idx = np.unique(bar_dates, return_inverse=True)
    trades_today_count = np.zeros(len(_unique_days), dtype=np.int64)
    pnl_today_sum = np.zeros(len(_unique_days), dtype=np.float64)
    day_has_pnl = np.zeros(len(_unique_days), dtype=bool)

    # P2-6 (news blackout): path to a CSV with one YYYY-MM-DD date per
    # line (blank lines and #-comments ignored). When set, run_execution
    # blocks ALL new entries on those dates -- the backtest-path
    # equivalent of the news-blackout gate the live engine already
    # enforces. Parsed once, up front. None (default) = disabled,
    # byte-identical to every run before this field existed.
    # NOTE: bar_dates can be an object array of tz-aware Timestamps
    # (pandas 3.x), which np.isin(datetime64) silently misses -- compare
    # plain datetime.date objects instead, which is unambiguous.
    _news_blackout_bar = np.zeros(n, dtype=bool)
    _news_csv = getattr(risk, "news_blackout_csv", None)
    if _news_csv:
        _news_blackout_dates: set = set()
        try:
            with open(_news_csv, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _s = _line.strip()
                    if _s and not _s.startswith("#"):
                        _news_blackout_dates.add(pd.Timestamp(_s).date())
        except (OSError, ValueError) as _e:
            import warnings
            warnings.warn(
                f"news_blackout_csv '{_news_csv}' could not be read/parsed ({_e}); "
                "news blackout is DISABLED for this run -- no dates will be blocked.",
                RuntimeWarning,
            )
        if _news_blackout_dates:
            _bar_day_dates = pd.DatetimeIndex(bar_dates).date
            _news_blackout_bar = np.fromiter(
                (d in _news_blackout_dates for d in _bar_day_dates),
                dtype=bool, count=n,
            )

    # P2-6 (weekend hold): bar_dates above is wall-clock (see the P2-3
    # fix), so day-of-week here is the calendar day the data was recorded
    # in, not UTC. _week_ends_at[i] is True when the next bar belongs to
    # a new week -- Friday for normal market data (next bar Monday),
    # Thursday before a Friday holiday, Sunday for weekend-traded
    # instruments. False (default) = the whole feature is off,
    # byte-identical to every run before it existed.
    _weekend_hold = bool(getattr(risk, "block_weekend_hold", False))
    _dow = pd.DatetimeIndex(bar_dates).dayofweek.to_numpy()  # Mon=0..Sun=6
    _week_ends_at = np.zeros(n, dtype=bool)
    if n:
        _week_ends_at[-1] = True
        _week_ends_at[:-1] = _dow[1:] <= _dow[:-1]

    sl_dist_vals = stop_loss_distance.values if stop_loss_distance is not None else None
    tp_dist_vals = take_profit_distance.values if take_profit_distance is not None else None
    trail_dist_vals = trailing_stop_distance.values if trailing_stop_distance is not None else None

    spread_price = risk.spread_pips * risk.pip_size
    slip_price = risk.slippage_pips * risk.pip_size

    # B2-2 (fill honesty): market entries and signal-driven exits fill at
    # open[i + fill_lag_bars], never at the signal bar's own close -- no
    # live trader can fill at the close that produced the signal. Default
    # 1 (next bar's open); 0 = the signal bar's own open. THIS CHANGES ALL
    # BACKTEST NUMBERS vs the old default.
    fill_lag_bars = max(int(risk.entry_fill_lag_bars or 0), 0)

    force_closed_count = 0

    # ------------------------------------------------------------------
    # ORDER/BROKER-STYLE FILL SETTLEMENT (single code path)
    # ------------------------------------------------------------------
    # Before this refactor, three separate call sites -- the normal
    # stop/take/signal exit, the daily-loss-limit forced close, and the
    # final end-of-data close -- each carried their OWN hand-copied
    # version of "apply spread/slippage, deduct commission, clamp the
    # loss, update equity, build the Trade, update adaptive-risk/day
    # bookkeeping". That triplication is exactly how a fix (e.g. the
    # asymmetric-cost and stop-fill-honesty bugs previously found here)
    # can get applied to one or two of the three paths and silently
    # missed on the third. `_settle_exit` is now the ONE place any exit,
    # of any kind, turns into a settled trade -- mirroring a
    # broker/order-fill abstraction rather than three independent ad hoc
    # blocks. This also fixes one such latent inconsistency: a
    # non-finite pnl on the daily-loss-forced-close or end-of-data path
    # used to be silently zeroed without the "_invalid_pnl_skipped"
    # suffix the normal exit path already applied -- now every path
    # gets the same treatment.
    def _exit_cause_for(reason: str, open_pos: dict) -> str:
        """Maps the engine-internal exit `reason` to the machine-verified
        exit_cause taxonomy stored on Trade.exit_cause (Part A port #1).
        The interesting refinement is "stop_loss": a stop resting AT the
        entry price (moved there by breakeven_trigger_r or a partial
        exit's move_stop_to_breakeven) is an exit-quality "breakeven", not
        a loss, and a stop the trailing ratchet moved away from its
        original level is a "trailing_stop" -- lumping those in with plain
        stop losses is exactly the "entries fine, exits broken" blindness
        this taxonomy closes. account_blown_forced_close has no closer
        taxonomy value than "session_close" (that account's trading
        session ended via termination); partial_take_profit is a
        take-profit mechanism, so it maps to "take_profit"."""
        reason = reason.removesuffix("_invalid_pnl_skipped")
        if reason == "stop_loss":
            stop_now = open_pos.get("stop_price")
            entry = open_pos.get("entry_price")
            direction_ = open_pos.get("direction")
            init_risk = open_pos.get("initial_risk")
            if (
                stop_now is not None and entry is not None
                and math.isclose(stop_now, entry, rel_tol=1e-9, abs_tol=1e-12)
            ):
                return "breakeven"
            if init_risk and direction_ and stop_now is not None and entry is not None:
                orig_stop = entry - direction_ * init_risk
                if (
                    open_pos.get("trailing_distance")
                    and not math.isclose(stop_now, orig_stop, rel_tol=1e-9, abs_tol=1e-12)
                ):
                    return "trailing_stop"
            return "stop_loss"
        return {
            "take_profit": "take_profit",
            "signal": "signal_exit",
            "time_stop": "time_stop",
            "daily_loss_limit_forced_close": "daily_loss_close",
            "weekend_hold_forced_close": "session_close",
            "account_blown_forced_close": "session_close",
            "partial_take_profit": "take_profit",
            "end_of_data": "end_of_data",
        }.get(reason, reason)

    def _prop_day(bar_date_: int) -> int:
        """Active-day index (days the account actually traded or held a
        position) for a session-day index; creates it on first use."""
        d = prop_day_map.get(bar_date_)
        if d is None:
            d = len(prop_day_dates)
            prop_day_map[bar_date_] = d
            prop_day_dates.append(_unique_days[bar_date_])
        return d

    def _prop_feed_close(pnl_: float, bar_date_: int, risk_dollars, i_: int) -> None:
        """Feed one realized close to the attempt's PropAccount and sync
        the engine's equity with the account balance (payouts withdraw)."""
        nonlocal equity, prop_pending, prop_locked_bar_date, prop_payouts_seen
        pday = _prop_day(bar_date_)
        res = acct.on_trade_close(
            pnl_, pday, is_last_of_day=True, trade_initial_risk=risk_dollars,
            floating_handled=True,
        )
        if res == _FAILED:
            prop_pending = "failed"
        elif res == _DAY_LOCKED:
            prop_locked_bar_date = bar_date_
        if len(acct.payouts) > prop_payouts_seen:
            for rec in acct.payouts[prop_payouts_seen:]:
                reset_events.append({
                    "kind": "payout", "reset_at": _restore_tz(ts[i_]),
                    "attempt_id": attempt_id, "payout_amount": rec.amount,
                    "equity_before_payout": rec.balance_after + rec.amount,
                    "equity_after_payout": rec.balance_after,
                    "message": (
                        f"{_restore_tz(ts[i_])}: PROFIT TARGET REACHED - "
                        f"attempt {attempt_id}: payout ${rec.amount:,.2f} banked; "
                        f"account balance ${rec.balance_after:,.2f}; trading continues."
                    ),
                })
            prop_payouts_seen = len(acct.payouts)
        equity = acct.balance

    def _settle_exit(open_pos: dict, raw_exit_price: float, reason: str, direction_: int, i: int) -> float:
        nonlocal equity, gap_loss_count, last_close_bar_idx
        last_close_bar_idx = i  # EXEC-002: this is a FULL close -- see reentry_cooldown_bars above
        filled_exit_price = raw_exit_price - (spread_price + slip_price) * direction_
        pnl = (filled_exit_price - open_pos["entry_price"]) * open_pos["size"] * direction_
        # B2-4: commission = flat per-trade + per-contract x contracts
        # actually closed (open_pos["size"] is post-partial, in sizing
        # units; contract_size converts to whole contracts). 0.0 defaults
        # keep this byte-identical to the old flat-only charge.
        _pos_cs = open_pos.get("contract_size", risk.contract_size)
        _pos_cpc = open_pos.get("commission_per_contract", risk.commission_per_contract)
        _contracts_closed = open_pos["size"] / _pos_cs if _pos_cs else 0.0
        _commission_charged = risk.commission_per_trade + _pos_cpc * _contracts_closed
        pnl -= _commission_charged
        if not math.isfinite(pnl):
            # Guard against a runaway/degenerate trade (e.g. an entry
            # sized off a near-zero ATR-based stop distance) ever
            # corrupting the equity curve with NaN/inf.
            pnl = 0.0
            reason = f"{reason}_invalid_pnl_skipped"
        if (
            reason == "stop_loss"
            and open_pos["initial_risk"]
            and abs(pnl) > 3 * open_pos["initial_risk"] * open_pos["size"]
        ):
            # The stop fired, but the fill was still several times worse
            # than the risk this trade was sized for -- almost always a
            # genuine price gap jumping straight past the resting stop
            # in one bar, not a bug. Surfaced as a warning below so a
            # handful of outsized losses in an otherwise-sane backtest
            # don't get mistaken for a broken engine.
            gap_loss_count += 1
        pnl = _clamp_loss(pnl, open_pos["equity_at_entry"])
        equity += pnl
        trades.append(Trade(
            entry_time=open_pos["entry_time"],
            exit_time=_restore_tz(ts[i]),
            direction=direction_,
            entry_price=open_pos["entry_price"],
            exit_price=filled_exit_price,
            size=open_pos["size"],
            pnl=pnl,
            pnl_pct=(pnl / open_pos["equity_at_entry"]) * 100 if open_pos["equity_at_entry"] else 0.0,
            exit_reason=reason,
            commission=_commission_charged,
            equity_after=equity,
            initial_risk=open_pos["initial_risk"],
            adaptive_risk_multiplier=open_pos["adaptive_multiplier"],
            adaptive_risk_rules_active=tuple(open_pos["adaptive_rules_active"]),
            intended_risk_dollars=open_pos.get("intended_risk_dollars"),
            worst_price=open_pos.get("worst_price"),
            mfe_price=open_pos.get("mfe_price"),
            exit_cause=_exit_cause_for(reason, open_pos),
            sized_above_risk_target=bool(open_pos.get("sized_above_risk_target", False)),
            risk_at_stop_dollars=open_pos.get("risk_at_stop_dollars"),
            stop_capped=bool(open_pos.get("stop_capped", False)),
            sizing_mode=open_pos.get("sizing_mode"),
            used_micro=bool(open_pos.get("used_micro", False)),
            contracts=open_pos.get("contracts"),
            attempt_id=attempt_id,
            fill_resolution=open_pos.get("fill_resolution", "bar"),
        ))
        bar_date_ = day_idx[i]
        adaptive_state.record_trade_close(pnl, is_new_day=not day_has_pnl[bar_date_])
        pnl_today_sum[bar_date_] += pnl
        day_has_pnl[bar_date_] = True
        if prop_mode:
            _prop_feed_close(pnl, bar_date_, open_pos.get("risk_at_stop_dollars"), i)
        return pnl

    partial_r_multiple = float(partial_exit_config["r_multiple"]) if partial_exit_config else None
    partial_fraction = float(partial_exit_config["fraction"]) if partial_exit_config else None
    partial_move_to_breakeven = bool(partial_exit_config.get("move_stop_to_breakeven", True)) if partial_exit_config else False

    def _settle_partial_exit(open_pos: dict, fill_price: float, direction_: int, i: int) -> float:
        """Closes `partial_fraction` of open_pos's ORIGINAL size at
        fill_price -- a real, independently-settled trade of its own
        (exit_reason='partial_take_profit'), NOT a full close: open_pos
        itself keeps running afterward with its size reduced by the same
        amount. Mirrors _settle_exit's cost/clamp/bookkeeping treatment
        so a partial exit is held to the same honesty standard (spread,
        slippage, commission, loss-clamping) as any other settled trade;
        commission is charged pro-rata to the fraction closed rather than
        a full extra round-turn, since only a fraction of the position is
        actually being closed out."""
        nonlocal equity
        partial_size = open_pos["initial_size"] * partial_fraction
        filled_exit_price = fill_price - (spread_price + slip_price) * direction_
        pnl = (filled_exit_price - open_pos["entry_price"]) * partial_size * direction_
        # B2-4: same per-trade + per-contract charge as a full settle, scaled
        # pro-rata to the fraction actually closed (contracts of the
        # ORIGINAL size x the fraction closed).
        _pos_cs = open_pos.get("contract_size", risk.contract_size)
        _pos_cpc = open_pos.get("commission_per_contract", risk.commission_per_contract)
        _contracts_total = open_pos["initial_size"] / _pos_cs if _pos_cs else 0.0
        _commission_charged = (
            risk.commission_per_trade + _pos_cpc * _contracts_total
        ) * partial_fraction
        pnl -= _commission_charged
        if not math.isfinite(pnl):
            pnl = 0.0
        pnl = _clamp_loss(pnl, open_pos["equity_at_entry"])
        equity += pnl
        trades.append(Trade(
            entry_time=open_pos["entry_time"],
            exit_time=_restore_tz(ts[i]),
            direction=direction_,
            entry_price=open_pos["entry_price"],
            exit_price=filled_exit_price,
            size=partial_size,
            pnl=pnl,
            pnl_pct=(pnl / open_pos["equity_at_entry"]) * 100 if open_pos["equity_at_entry"] else 0.0,
            exit_reason="partial_take_profit",
            commission=_commission_charged,
            equity_after=equity,
            initial_risk=open_pos["initial_risk"],
            adaptive_risk_multiplier=open_pos["adaptive_multiplier"],
            adaptive_risk_rules_active=tuple(open_pos["adaptive_rules_active"]),
            intended_risk_dollars=open_pos.get("intended_risk_dollars"),
            worst_price=open_pos.get("worst_price"),
            mfe_price=open_pos.get("mfe_price"),
            exit_cause="take_profit",  # partial scale-out is a take-profit mechanism
            sized_above_risk_target=bool(open_pos.get("sized_above_risk_target", False)),
            risk_at_stop_dollars=open_pos.get("risk_at_stop_dollars"),
            stop_capped=bool(open_pos.get("stop_capped", False)),
            sizing_mode=open_pos.get("sizing_mode"),
            used_micro=bool(open_pos.get("used_micro", False)),
            contracts=open_pos.get("contracts"),
            attempt_id=attempt_id,
            fill_resolution=open_pos.get("fill_resolution", "bar"),
        ))
        open_pos["size"] -= partial_size
        bar_date_ = day_idx[i]
        pnl_today_sum[bar_date_] += pnl
        day_has_pnl[bar_date_] = True
        if prop_mode:
            _prop_feed_close(pnl, bar_date_, None, i)
        # Deliberately NOT calling adaptive_state.record_trade_close here --
        # the position this partial belongs to is still open, so this isn't
        # a trade CLOSE for adaptive-risk purposes (consecutive-loss/streak
        # tracking is keyed to whether a position was closed, not to every
        # settled P&L event within one).
        return pnl

    def _resolve_intrabar_exit(
        direction_: int, stop: float | None, take: float | None,
        low: float, high: float, open_: float,
    ) -> tuple[float | None, str | None]:
        """Pure fill-resolution logic, extracted so it's independently
        testable: does this bar's high/low trigger the resting stop or
        target, and if so, at what honest price? A resting stop that the
        bar gapped straight through does NOT fill at the stop level --
        it fills at the open, which is worse. Filling every stop at its
        exact level is one of the most common sources of a fake
        backtest edge."""
        if direction_ == 1:
            if stop is not None and low <= stop:
                return min(stop, open_), "stop_loss"
            if take is not None and high >= take:
                return take, "take_profit"
        else:
            if stop is not None and high >= stop:
                return max(stop, open_), "stop_loss"
            if take is not None and low <= take:
                return take, "take_profit"
        return None, None

    # UPGRADE (speed): equity_curve used to be a plain Python list that
    # every one of n bars appended a (timestamp, equity) tuple to, then fed
    # to pd.DataFrame(list_of_tuples, columns=[...]) at the end --
    # constructing a DataFrame from a list of 2M+ Python tuples is a
    # row-wise conversion (pandas has to inspect and transpose every tuple),
    # which is far slower than building the two columns as numpy arrays
    # directly and handing pandas already-columnar data. Preallocating (vs.
    # letting the list grow) also avoids 2M+ individual tuple allocations.
    equity_arr = np.empty(n, dtype=np.float64)
    attempt_arr = np.zeros(n, dtype=np.int32)
    prop_last_bar_date = None

    def _prop_new_account(i_: int) -> None:
        nonlocal acct, prop_payouts_seen, prop_pending, prop_attempt_start_time
        acct = PropAccount(prop_rules, start_day_index=len(prop_day_dates), day_dates=prop_day_dates)
        acct.start_date = _unique_days[day_idx[min(i_, n - 1)]]
        prop_payouts_seen = 0
        prop_pending = None
        prop_attempt_start_time = _restore_tz(ts[min(i_, n - 1)])

    def _prop_end_attempt(i_: int, fill_price=None) -> None:
        """Closes the current attempt (bust or pass): flat the book, record
        it, and -- in chain mode -- buy a fresh account."""
        nonlocal open_trade, equity, attempt_id, prop_pending, prop_done, payout_target_baseline
        if open_trade is not None:
            px = closes[i_] if fill_price is None else fill_price
            _settle_exit(
                open_trade, px,
                "account_blown_forced_close" if acct.failed else "attempt_end_forced_close",
                open_trade["direction"], i_,
            )
            open_trade = None
        failed = acct.failed
        end_t = _restore_tz(ts[i_])
        prop_attempts.append({
            "attempt_id": attempt_id, "start_time": prop_attempt_start_time, "end_time": end_t,
            "outcome": "failed" if failed else "passed",
            "failure_reason": acct.failure_reason, "passed_evaluation": acct.passed_evaluation,
            "days_to_pass": acct.days_to_pass, "end_balance": acct.balance,
            "n_trades": acct.n_trades, "total_payout": acct.total_payout_amount,
            "max_dd_pct": acct.max_dd_pct_reached,
        })
        reset_events.append({
            "kind": "breach" if failed else "attempt_pass",
            "reset_at": end_t, "attempt_id": attempt_id,
            "equity_before_reset": acct.balance, "equity_after_forced_close": acct.balance,
            "drawdown_pct": ((prop_rules.account_size - acct.balance) / prop_rules.account_size * 100.0) if failed else None,
            "message": (
                f"{end_t}: attempt {attempt_id} " + (
                    f"FAILED ({acct.failure_reason}) at balance ${acct.balance:,.2f}"
                    if failed else f"PASSED the evaluation at balance ${acct.balance:,.2f}")
                + ("; fresh account started." if attempt_mode != "single" else ".")
            ),
        })
        if attempt_mode == "single":
            prop_done = True
            return
        attempt_id += 1
        equity = float(prop_rules.account_size)
        payout_target_baseline = equity
        adaptive_state.reset()
        _prop_new_account(i_)

    if prop_mode:
        _prop_new_account(0)

    for i in range(n):
        bar_date = day_idx[i]

        if prop_mode:
            if bar_date != prop_last_bar_date:
                if prop_last_bar_date is not None and prop_last_bar_date in prop_day_map and i > 0:
                    acct.end_day(prop_day_map[prop_last_bar_date], float(equity_arr[i - 1]))
                prop_last_bar_date = bar_date
                if acct.check_new_day(_unique_days[bar_date]) == _FAILED:
                    _prop_end_attempt(i, opens[i])
                    if prop_done:
                        equity_arr[i:] = equity
                        attempt_arr[i:] = attempt_id
                        break

        # STALL FIX (2026-10-06): roll the adaptive-risk daily
        # accumulators over WITH THE CLOCK. This used to happen only
        # inside record_trade_close (a new day was noticed when the next
        # trade CLOSED), so on any stretch where entries were throttled
        # to zero and no trade ever closed, "today" stayed frozen on the
        # last active day -- and a daily profit lock (multiplier 0.0)
        # tripped that day blocked every entry for the rest of the
        # dataset (Owen's 2020-2026 ES run: last entry 2021-09-20, then
        # 55,795 later signals blocked by "adaptive risk zero size").
        # Daily triggers now always see the CURRENT day's realized P&L.
        if adaptive_risk is not None and adaptive_risk.enabled and bar_date != _adaptive_state_day:
            adaptive_state.begin_new_day()
            _adaptive_state_day = bar_date

        # P2-6 (weekend hold): a new week clears the weekend entry block
        # (latched when a position was force-closed on the week's last
        # bar). Saturday/Sunday bars stay blocked via weekend_blocked_today
        # below even though the latch is only set at week-end.
        if _weekend_hold and _dow[i] == 0:
            weekend_entry_block = False

        # --- manage open trade: trailing stop / break-even, then stop/take intrabar ---
        if open_trade is not None:
            direction = open_trade["direction"]

            favorable_extreme = highs[i] if direction == 1 else lows[i]
            if direction == 1:
                open_trade["best_price"] = max(open_trade["best_price"], favorable_extreme)
            else:
                open_trade["best_price"] = min(open_trade["best_price"], favorable_extreme)

            # Part A port #1: track the most FAVORABLE price reached (MFE)
            # exactly like best_price above -- best_price doubles as the
            # trailing-stop ratchet input, while mfe_price is the clean
            # per-trade favorable extreme for exit-quality analysis.
            if direction == 1:
                open_trade["mfe_price"] = max(open_trade["mfe_price"], favorable_extreme)
            else:
                open_trade["mfe_price"] = min(open_trade["mfe_price"], favorable_extreme)

            # P2-5: mirror best_price (favorable extreme) with worst_price
            # (adverse extreme) so max adverse excursion (MAE) is
            # computable per trade in app.backtest.statistics. Defined
            # here, once, so the daily-loss check below reuses it.
            adverse_extreme = lows[i] if direction == 1 else highs[i]
            if direction == 1:
                open_trade["worst_price"] = min(open_trade["worst_price"], adverse_extreme)
            else:
                open_trade["worst_price"] = max(open_trade["worst_price"], adverse_extreme)

            # PROP MODE: floating drawdown / floating daily-loss. The firm
            # liquidates the position the instant FLOATING equity touches the
            # applicable floor. Order matters: a resting stop that is closer
            # to entry than the liquidation level fills first (the account
            # survives with the stop loss); only if the floor is reached
            # BEFORE the stop does the firm close the trade. Gaps fill at
            # the open, never better.
            if prop_mode and (acct.dd_basis in ("floating", "eod") or acct.daily_loss_basis == "floating"):
                _ent = open_trade["entry_price"]
                _sz = open_trade["size"]
                _stop_now = open_trade["stop_price"]
                _adv_eff = adverse_extreme
                if _stop_now is not None:
                    if direction == 1 and lows[i] <= _stop_now:
                        _adv_eff = max(adverse_extreme, min(_stop_now, opens[i]))
                    elif direction == -1 and highs[i] >= _stop_now:
                        _adv_eff = min(adverse_extreme, max(_stop_now, opens[i]))
                _eq_hi = equity + (favorable_extreme - _ent) * _sz * direction
                _eq_lo = equity + (_adv_eff - _ent) * _sz * direction
                _dd_fl, _dl_fl = acct.projected_floors(_eq_hi)
                _lvl = None
                _kind = None
                if _dd_fl is not None and _eq_lo <= _dd_fl:
                    _lvl, _kind = _dd_fl, "prop_floor_liquidation"
                if _dl_fl is not None and _eq_lo <= _dl_fl and (_lvl is None or _dl_fl > _lvl):
                    _lvl, _kind = _dl_fl, "daily_loss_limit_forced_close"
                _pd = _prop_day(bar_date)
                if _lvl is None:
                    acct.on_equity_extremes(_pd, _eq_hi, _eq_lo)
                else:
                    _res = acct.on_equity_extremes(_pd, _eq_hi, _eq_lo)
                    _liq = _ent + direction * (_lvl - equity) / _sz if _sz else adverse_extreme
                    if direction == 1:
                        _liq = min(max(_liq, adverse_extreme), opens[i])
                    else:
                        _liq = max(min(_liq, adverse_extreme), opens[i])
                    _settle_exit(open_trade, _liq, _kind, direction, i)
                    open_trade = None
                    force_closed_count += 1
                    if _res == _DAY_LOCKED:
                        prop_locked_bar_date = bar_date
                    equity_arr[i] = equity
                    if prop_pending == "failed" or acct.failed:
                        _prop_end_attempt(i)
                        attempt_arr[i] = attempt_id
                        equity_arr[i] = equity
                        if prop_done:
                            equity_arr[i:] = equity
                            attempt_arr[i:] = attempt_id
                            break
                    continue

            # Mark-to-market daily-loss check, using the ADVERSE intrabar
            # extreme (low for a long, high for a short) rather than the
            # close. A prop firm's daily-loss floor is monitored on
            # floating equity in real time, not just on realized P&L at
            # the moment a trade happens to close — a trade that dips
            # deep underwater and recovers by the close of the bar can
            # still have breached (and been auto-liquidated at) the daily
            # floor intrabar. Checking only realized same-day P&L (the
            # old behavior) silently let strategies "survive" daily-loss
            # breaches that a real funded account would have been
            # stopped out of. (adverse_extreme is defined once, next to
            # the best_price/worst_price tracking above.)
            floating_adverse_pnl = (adverse_extreme - open_trade["entry_price"]) * open_trade["size"] * direction
            day_realized_so_far = pnl_today_sum[bar_date]
            if (
                daily_limit_amount is not None
                and (day_realized_so_far + floating_adverse_pnl) <= -daily_limit_amount
            ):
                # C3 (liquidate AT the floor): the old code settled the
                # forced close at the bar's adverse extreme -- but the
                # real account is liquidated the FIRST moment cumulative
                # day P&L hits the daily floor, which is strictly before
                # (or at) the adverse extreme on the bar that triggers
                # this. Solve for the price where the floor is hit:
                #   day_realized_so_far + (liq_price - entry)*size*direction
                #       == -daily_limit_amount
                # i.e. liq_price = entry - direction*(daily_limit_amount
                # + day_realized_so_far)/size. (Note: day_realized_so_far
                # is a SIGNED sum -- negative on a losing day -- so the
                # remaining room to the floor is daily_limit_amount +
                # day_realized_so_far, e.g. $1000 + (-$200) = $800 left.
                # The minus-sign variant would liquidate $400 PAST the
                # floor here, and -- via the clamp below -- degenerate to
                # the old adverse-extreme behavior whenever the day was
                # already red.) Clamped to not exceed the bar's adverse
                # extreme (for a long the liquidation price can never be
                # MORE adverse than the bar's own low -- the clamp only
                # guards float noise at the boundary) and settled with
                # the same round-turn cost as any other exit via
                # _settle_exit.
                _liq_size = open_trade["size"]
                if _liq_size and _liq_size > 0:
                    _liq_dist = (daily_limit_amount + day_realized_so_far) / _liq_size
                    _liq_price = open_trade["entry_price"] - direction * _liq_dist
                else:
                    _liq_price = adverse_extreme
                if direction == 1:
                    _liq_price = max(_liq_price, adverse_extreme)
                else:
                    _liq_price = min(_liq_price, adverse_extreme)
                _settle_exit(open_trade, _liq_price, "daily_loss_limit_forced_close", direction, i)
                open_trade = None
                force_closed_count += 1
                if getattr(risk, "prop_daily_loss_is_breach", False):
                    # P1-2: most real prop firms TERMINATE the account on
                    # a daily-loss breach -- not just "no new entries for
                    # the rest of the day". Mark it blown (no new trades
                    # for the rest of the run, exactly like hitting
                    # max_account_drawdown_pct) instead of merely
                    # blocking entries. False (default) = byte-identical
                    # to the historical force-close-and-keep-trading
                    # behavior.
                    account_blown = True
                    account_blown_at = _restore_tz(ts[i])
                equity_arr[i] = equity
                continue

            stop = open_trade["stop_price"]

            # C4 (intrabar ordering: stop before tightening). The
            # tightening blocks below (breakeven / trailing / partial
            # scale-out) only ever move the resting stop CLOSER to price.
            # On a bar that touches BOTH the profit trigger and the
            # RESTING stop level, the live engine fills the resting stop
            # first -- it was the order in the book while the bar traded
            # there. Resolving the tightening first would let the same bar
            # exit at breakeven (a scratch) instead of the full stop loss
            # it actually hit. So the pre-tightening stop level is
            # resolved BEFORE any tightening is applied; a hit here exits
            # at the resting stop with the conservative gap-through fill,
            # consistent with stop-beats-target ordering in
            # _resolve_intrabar_exit (the take-profit is deliberately NOT
            # pre-checked -- take handling is unchanged).
            if _ib_on and open_trade["take_price"] is not None and stop is not None:
                _both_s, _ = _resolve_intrabar_exit(direction, stop, None, lows[i], highs[i], opens[i])
                _both_t, _ = _resolve_intrabar_exit(direction, None, open_trade["take_price"], lows[i], highs[i], opens[i])
                if _both_s is not None and _both_t is not None:
                    _first = _intrabar_first_hit(direction, stop, open_trade["take_price"], i)
                    open_trade["fill_resolution"] = "intrabar"
                    if _first == "take" and not (
                        (direction == 1 and opens[i] <= stop) or (direction == -1 and opens[i] >= stop)
                    ):
                        _settle_exit(open_trade, open_trade["take_price"], "take_profit", direction, i)
                        open_trade = None
                        equity_arr[i] = equity
                        continue
            _pre_tighten_exit, _pre_tighten_reason = _resolve_intrabar_exit(
                direction, stop, None, lows[i], highs[i], opens[i]
            )
            if _pre_tighten_exit is not None:
                # open_trade["stop_price"] still holds the pre-tightening
                # (resting) level here, so _exit_cause_for's
                # breakeven/trailing taxonomy classifies correctly.
                _settle_exit(open_trade, _pre_tighten_exit, _pre_tighten_reason, direction, i)
                open_trade = None
                equity_arr[i] = equity
                continue

            # Break-even: once profit reaches the configured R multiple,
            # move the stop to entry (only ever tightens the stop).
            if (
                breakeven_trigger_r is not None
                and open_trade["initial_risk"]
                and not open_trade["breakeven_done"]
            ):
                profit_dist = (open_trade["best_price"] - open_trade["entry_price"]) * direction
                if profit_dist >= breakeven_trigger_r * open_trade["initial_risk"]:
                    candidate = open_trade["entry_price"]
                    if stop is None or (direction == 1 and candidate > stop) or (direction == -1 and candidate < stop):
                        stop = candidate
                    open_trade["breakeven_done"] = True

            # Trailing stop: ratchet toward price, never away from it.
            if open_trade.get("trailing_distance"):
                candidate = open_trade["best_price"] - direction * open_trade["trailing_distance"]
                if stop is None or (direction == 1 and candidate > stop) or (direction == -1 and candidate < stop):
                    stop = candidate

            # Partial exit / scale-out: once open profit reaches the
            # configured R multiple, close `fraction` of the ORIGINAL
            # size at that level (a real settled trade of its own -- see
            # _settle_partial_exit) and optionally tighten the remaining
            # position's stop to breakeven. Fires at most once per trade
            # (partial_taken), and only when there's still a genuine
            # initial_risk to measure R against. Checked via the same
            # intrabar high/low the stop/take logic below uses, filled at
            # the exact target level (matching how a take-profit already
            # fills here -- see _resolve_intrabar_exit's docstring for why
            # STOPS, not targets, get the conservative gap-through fill).
            if (
                partial_exit_config is not None
                and not open_trade["partial_taken"]
                and open_trade["initial_risk"]
            ):
                partial_target = open_trade["entry_price"] + direction * partial_r_multiple * open_trade["initial_risk"]
                partial_hit = (highs[i] >= partial_target) if direction == 1 else (lows[i] <= partial_target)
                # If the strategy's own take-profit is at or before the
                # partial target, the position fully closes there anyway
                # before ever reaching the partial level -- skip so the
                # normal full-exit path below is the one that fires.
                take_before_partial = (
                    open_trade["take_price"] is not None
                    and (
                        (direction == 1 and open_trade["take_price"] <= partial_target)
                        or (direction == -1 and open_trade["take_price"] >= partial_target)
                    )
                )
                if partial_hit and not take_before_partial:
                    _settle_partial_exit(open_trade, partial_target, direction, i)
                    open_trade["partial_taken"] = True
                    if partial_move_to_breakeven:
                        candidate = open_trade["entry_price"]
                        if stop is None or (direction == 1 and candidate > stop) or (direction == -1 and candidate < stop):
                            stop = candidate

            open_trade["stop_price"] = stop
            take = open_trade["take_price"]
            exit_price, reason = _resolve_intrabar_exit(direction, stop, take, lows[i], highs[i], opens[i])

            # signal-driven exit (flat or reversal): a market order filled at
            # the next bar's open (see B2-2) if no SL/TP hit intrabar
            if exit_price is None and sig[i] != direction:
                _sig_fill_idx = i + fill_lag_bars
                if _sig_fill_idx < n:
                    # B2-2: a signal exit is a market order -- it fills at
                    # the next bar's open, not at the close that produced
                    # the signal. The trade settles on the FILL bar (exit
                    # time, cooldown, and day-PnL all key off it).
                    exit_price, reason = opens[_sig_fill_idx], "signal"
                    _settle_exit(open_trade, exit_price, reason, direction, _sig_fill_idx)
                    # Cooldown keys off the TRIGGER bar (i), not the fill
                    # bar: reentry_cooldown_bars is documented as "measured
                    # from the bar the previous position closed on", and its
                    # contract is about the information bar -- a signal on
                    # the very next bar is new information and must stay
                    # tradable (cooldown=1 blocks the same bar's decision
                    # only). Keying it off the fill bar would silently
                    # extend every cooldown by the fill lag.
                    last_close_bar_idx = i
                    open_trade = None
                # else: no future bar to fill on -- leave the position open;
                # the end-of-data close below settles it honestly.

            # Part A port #4 (time-invalidation stop): at the bar's close,
            # if the position is older than time_stop_hours AND price has
            # barely moved from the entry (|close - entry| still inside
            # time_stop_atr_band * ATR), the capital is dead -- close it as
            # stagnant. Absolute (either-direction) excursion: a flat trade
            # is stagnant whether it flatlined a touch above or below the
            # entry; a trade that actually went somewhere (winner OR slow
            # loser past the band) is left to its stop/target/signal logic.
            # Checked after the signal exit so a strategy-driven exit keeps
            # its own cause; fills at the close like any bar-close decision.
            if exit_price is None and open_trade is not None and risk.time_stop_hours:
                _age_hours = (ts[i] - ts[open_trade["entry_bar_idx"]]) / np.timedelta64(1, "h")
                _atr_now = float(_atr_for_mismatch_check[i])
                _stagnation = abs(closes[i] - open_trade["entry_price"])
                if (
                    _age_hours > risk.time_stop_hours
                    and _atr_now > 0
                    and _stagnation <= risk.time_stop_atr_band * _atr_now
                ):
                    exit_price, reason = closes[i], "time_stop"

            if exit_price is not None and open_trade is not None:
                # Every exit is a real transaction and pays the same
                # round-turn cost the entry did — crediting a stop/take/
                # signal exit at its exact quoted level (with no spread or
                # slippage) flatters every single trade by that amount.
                # See _settle_exit for the shared fill/cost/bookkeeping
                # logic every exit path (this one, the daily-loss forced
                # close, and the end-of-data close) now goes through.
                _settle_exit(open_trade, exit_price, reason, direction, i)
                open_trade = None

            # P2-6 (weekend hold): still open at the close of the week's
            # last bar -> force-close at the close, mirroring the live
            # engine's weekend-hold ban. The intrabar stop/take/signal
            # exits just above take precedence -- they really happened
            # first. Latches weekend_entry_block so no new position opens
            # until Monday (cleared at the top of the loop).
            if open_trade is not None and _weekend_hold and _week_ends_at[i]:
                _settle_exit(open_trade, closes[i], "weekend_hold_forced_close", direction, i)
                open_trade = None
                weekend_close_count += 1
                weekend_entry_block = True

        # --- mark-to-market equity curve point for this bar ---
        # Realized equity plus the floating P&L of any still-open
        # position, valued at this bar's close. This is what feeds the
        # drawdown statistics (app.backtest.statistics) -- using realized
        # equity alone understated true intrabar/multi-day drawdown
        # whenever a trade was open across the peak.
        if open_trade is not None:
            floating_close_pnl = (
                (closes[i] - open_trade["entry_price"]) * open_trade["size"] * open_trade["direction"]
            )
            mtm_equity = equity + floating_close_pnl
        else:
            mtm_equity = equity
        equity_arr[i] = mtm_equity
        attempt_arr[i] = attempt_id

        if prop_mode:
            if prop_pending == "failed" or acct.failed:
                _prop_end_attempt(i)
                equity_arr[i] = equity
                attempt_arr[i] = attempt_id if not prop_done else attempt_arr[i]
            elif stop_on_pass and acct.passed_evaluation:
                _prop_end_attempt(i)
                equity_arr[i] = equity
            if prop_done:
                equity_arr[i:] = equity
                attempt_arr[i:] = attempt_id
                break

        # --- account-blown circuit breaker ---
        # A real prop/broker account is terminated the instant it breaches
        # the firm's max-drawdown floor -- it does not keep "trading"
        # itself into ever-deeper negative equity. Once realized equity
        # crosses that floor (or the account has literally run out of
        # money), the default (risk.reset_on_breach == False) is to stop
        # opening any new trades for the remainder of the run; this is
        # what makes a single misconfigured/gapped trade (see
        # max_trade_loss above) unable to cascade into the kind of
        # -$50,000-on-a-$50,000-account or -$2.5M results this was built
        # to prevent. Any already-open trade still manages normally
        # (stop/target/signal exits) -- only NEW entries are blocked.
        if not prop_mode and not account_blown and (equity <= 0 or (blown_floor is not None and equity <= blown_floor)):
            if not halt_on_breach:
                # RESET-ON-BREACH (now the engine default, see RiskConfig.reset_on_breach): a
                # prop-firm evaluator who "doesn't care about blowing
                # accounts, as long as I can make money before" mechanically
                # buys a new account and keeps going rather than treating
                # one blown account as the end of the story. Mirror that
                # here instead of halting forever: force-close any position
                # still open (a terminated account cannot keep holding
                # one -- same settlement path, spread/slippage/commission/
                # loss-clamp included, as any other exit), record the
                # event, and reset equity to initial_balance so entries
                # resume on the very next eligible signal. account_blown
                # is deliberately never set True on this path.
                breach_equity = equity
                breach_at = _restore_tz(ts[i])
                if open_trade is not None:
                    direction = open_trade["direction"]
                    _settle_exit(open_trade, closes[i], "account_blown_forced_close", direction, i)
                    open_trade = None
                _floor_txt = f"${blown_floor:,.2f}" if blown_floor is not None else "$0"
                reset_events.append({
                    "kind": "breach",
                    "reset_at": breach_at,
                    "equity_before_reset": breach_equity,
                    "equity_after_forced_close": equity,
                    "drawdown_pct": (
                        (risk.initial_balance - breach_equity) / risk.initial_balance * 100.0
                        if risk.initial_balance else None
                    ),
                    "message": (
                        f"{breach_at}: ACCOUNT BREACH #{sum(1 for e in reset_events if e.get('kind') == 'breach') + 1} - "
                        f"equity ${breach_equity:,.2f} fell to the loss floor ({_floor_txt}). "
                        f"Fresh ${risk.initial_balance:,.2f} account started; trading continues."
                    ),
                })
                equity = risk.initial_balance
                equity_arr[i] = equity
                payout_target_baseline = risk.initial_balance
                adaptive_state.reset()  # a new account must not inherit the old one's throttle state
            else:
                account_blown = True
                account_blown_at = _restore_tz(ts[i])

        # --- profit-target payout (see RiskConfig.reset_on_target) ---
        # The breach circuit breaker above handles the LOSS side of "this
        # account's story just ended, but the run should keep going
        # anyway" -- this is the same idea for the PROFIT side, which
        # previously didn't exist at all: nothing in this engine ever
        # recognized "the account hit its profit target" as an event,
        # so a strategy that reached target simply kept accumulating
        # equity on the same never-reset number for the rest of the
        # dataset (this is why an equity curve could visibly go flat --
        # or, in the mark-to-market display, just keep silently climbing
        # -- the moment a big early winning streak was through: nothing
        # was WRONG, there was just no concept of "bank the win and keep
        # trading" on this side). Checked against REALIZED equity only
        # (never the mark-to-market value computed above) so an open
        # position's still-floating profit is never withdrawn out from
        # under it -- only money that has actually settled through
        # _settle_exit.
        if (
            not prop_mode and risk.reset_on_target and risk.profit_target_pct and risk.profit_target_pct > 0 and not account_blown
            and equity >= payout_target_baseline * (1 + risk.profit_target_pct / 100.0)
        ):
            payout_amount = equity - payout_target_baseline
            reset_events.append({
                "kind": "payout",
                "reset_at": _restore_tz(ts[i]),
                "equity_before_payout": equity,
                "payout_amount": payout_amount,
                "equity_after_payout": payout_target_baseline,
                "message": (
                    f"{_restore_tz(ts[i])}: PROFIT TARGET REACHED #{sum(1 for e in reset_events if e.get('kind') == 'payout') + 1} - "
                    f"equity ${equity:,.2f} hit {risk.profit_target_pct:g}% target; "
                    f"${payout_amount:,.2f} banked as a payout. Account reset to "
                    f"${payout_target_baseline:,.2f}; trading continues."
                ),
            })
            equity = payout_target_baseline
            equity_arr[i] = equity
            adaptive_state.reset()  # "progress to target" / drawdown triggers start over

        # --- consider new entry ---
        day_realized_pnl = pnl_today_sum[bar_date]
        daily_limit_breached = (
            daily_limit_amount is not None and day_realized_pnl <= -daily_limit_amount
        ) or (prop_mode and prop_locked_bar_date == bar_date)
        # EXEC-002: without this, a position stopped out (or hit its
        # target) intrabar on bar `i` could immediately reopen a fresh
        # position in the SAME direction at that SAME bar's close, as
        # long as the strategy's signal hadn't gone flat -- a whipsaw bar
        # that clips a tight stop and then closes back at a level the
        # strategy still signals on produced two trades and re-exposed
        # the same risk within one bar, with no cooldown and no way to
        # configure one. reentry_cooldown_bars defaults to 1, under which
        # the same bar's close is blocked but the next bar is allowed;
        # set it explicitly to 0 for byte-identical original
        # (undocumented, unconditional) same-bar reentry, or to N>0 to
        # block any new entry until N bars after the previous close,
        # regardless of signal.
        cooldown_active = (i - last_close_bar_idx) < risk.reentry_cooldown_bars
        # P2-6 gates: no new entries on news-blackout dates, and none
        # while the weekend-hold block is active (weekend bars themselves
        # or the latch set by a week-end force-close, until Monday).
        news_blackout_today = bool(_news_blackout_bar[i])
        weekend_blocked_today = _weekend_hold and (_dow[i] >= 5 or weekend_entry_block)
        if open_trade is None and sig[i] != 0 and not daily_limit_breached and not account_blown and not cooldown_active and not news_blackout_today and not weekend_blocked_today:
            # B2-2: a market entry fills at open[i + fill_lag_bars], not at
            # the signal bar's close. No future bar to fill on (signal on
            # the last bar(s)) means no entry -- an honest engine cannot
            # fill an order after the data ends.
            fill_idx = i + fill_lag_bars
            if fill_idx >= n:
                continue
            # The trade opens on the FILL bar: max-trades/day counting and
            # the stall diagnostic's last-entry marker key off it.
            fill_bar_date = day_idx[fill_idx]
            n_today = trades_today_count[fill_bar_date]
            if n_today >= risk.max_trades_per_day:
                blocked_max_trades_count += 1
            if n_today < risk.max_trades_per_day:
                direction = int(sig[i])
                raw_price = opens[fill_idx]
                entry_price = raw_price + (spread_price + slip_price) * direction

                # UPGRADE (speed): pd.isna() on a single numpy float64 scalar
                # dispatches through pandas' general-purpose type checking,
                # which is materially slower than a direct NaN check for a
                # value already known to be a plain float. math.isnan() does
                # the identical check for these purely-numeric arrays.
                bar_sl_distance = float(sl_dist_vals[i]) if sl_dist_vals is not None and not math.isnan(sl_dist_vals[i]) else None
                bar_tp_distance = float(tp_dist_vals[i]) if tp_dist_vals is not None and not math.isnan(tp_dist_vals[i]) else None
                bar_trail_distance = float(trail_dist_vals[i]) if trail_dist_vals is not None and not math.isnan(trail_dist_vals[i]) else None

                used_fallback_stop = False
                if not bar_sl_distance and not stop_loss_pips:
                    # The strategy defined no stop loss at all (no ATR-based
                    # distance and no fixed pips). Sizing off a hardcoded
                    # small pip count here would be wrong for any
                    # non-FX-scaled instrument (e.g. gold, indices, crypto)
                    # and would also leave the position completely
                    # unprotected. Fall back to a stop sized as a percentage
                    # of the entry price instead — this scales correctly
                    # regardless of instrument or pip size.
                    bar_sl_distance = abs(raw_price) * DEFAULT_STOP_PCT_OF_PRICE
                    used_fallback_stop = True

                if bar_sl_distance:
                    sl_price_dist = bar_sl_distance
                else:
                    sl_price_dist = (stop_loss_pips or 0) * risk.pip_size
                sizing_pips = (sl_price_dist / risk.pip_size) if risk.pip_size else 0
                intended_risk_dollars = risk.risk_amount(equity)
                # ACCURACY OVERHAUL (2026-10-07): ONE sizing routine for every
                # mode (see risk.SIZING_MODES / RiskConfig.size_for_stop). The
                # decision carries the contracts, the (possibly fit-to-budget)
                # stop, whether the micro contract was swapped in, and -- when
                # the entry is skipped -- WHY. Costs are inside the budget.
                # The old dead-lock "1-contract rescue" that opened one
                # contract ABOVE the risk budget is gone: the only way to
                # exceed the budget now is the explicit, tagged
                # allow_single_contract_minimum opt-in below.
                decision = risk.size_for_stop(equity, sl_price_dist)
                size = decision.units
                pos_contract_size = decision.contract_size if decision.contract_size else risk.contract_size
                pos_commission_pc = decision.commission_per_contract
                sizing_considered_count += 1
                rescue_above_risk_target = bool(decision.above_budget)
                # Dead-lock guard (bounded): a config that a FRESH account can
                # afford (>= 1 whole contract at initial_balance) must not stall
                # forever just because a small loss pulled equity a hair under
                # the one-contract threshold -- with no trades there is no way
                # to climb back. One contract is allowed only while its
                # worst-case loss stays within 1.5x the current budget; the
                # trade is tagged sized_above_risk_target and counted as a
                # rescue. allow_single_contract_minimum keeps its older,
                # unbounded meaning.
                _deadlock_guard = False
                if size <= 0 and risk.contract_size and equity > 0 and decision.skip_reason is not None \
                        and sizing_mode_is_contract_skip(risk):
                    _fresh = risk.size_for_stop(risk.initial_balance, sl_price_dist)
                    if _fresh.units > 0 and _fresh.contracts >= 1:
                        _one = risk.worst_case_loss(
                            float(risk.contract_size), sl_price_dist, risk.contract_size, risk.commission_per_contract,
                        )
                        _deadlock_guard = _one <= 1.5 * max(intended_risk_dollars, 1e-9)
                if (
                    size <= 0 and (risk.allow_single_contract_minimum or _deadlock_guard) and risk.contract_size and equity > 0
                ):
                    size = float(risk.contract_size)
                    pos_contract_size = risk.contract_size
                    pos_commission_pc = risk.commission_per_contract
                    min_contract_rescue_count += 1
                    rescue_above_risk_target = True

                adaptive_multiplier = 1.0
                adaptive_rules_active: list[str] = []
                if adaptive_risk is not None and adaptive_risk.enabled:
                    current_vol_pct = None
                    if _vol_percentile_arr is not None:
                        _v = _vol_percentile_arr[i]
                        current_vol_pct = float(_v) if not math.isnan(_v) else None
                    adaptive_multiplier = adaptive_state.active_multiplier(adaptive_risk, current_vol_pct)
                    adaptive_rules_active = adaptive_state.active_rule_labels(adaptive_risk, current_vol_pct)
                    _size_before_adaptive = size
                    size *= adaptive_multiplier
                    # STALL FIX (2026-10-06): position_size() floored to whole
                    # contracts BEFORE this multiplier, so multiplying after
                    # the fact produced fractional contracts (e.g. 0.39 of an
                    # ES contract -- not a real, placeable size) and, whenever
                    # the stacked limit-aware multipliers got small enough
                    # (0.25 daily-loss x 0.25 drawdown x 0.125 volatility =
                    # 0.008x), shrunk size to effectively nothing for as long
                    # as conditions stayed rough -- months or years of a
                    # flat trade chart. Adaptive throttling may shrink size
                    # TOWARD the 1-contract minimum, never past it: re-floor
                    # to whole contracts, and if flooring would erase a trade
                    # the UNTHROTTLED sizing could afford (>= 1 contract),
                    # open at the 1-contract minimum instead. A multiplier of
                    # exactly 0.0 (a deliberate hard lock, e.g. the daily
                    # profit lock) still blocks outright -- that is the rule
                    # doing its job for the rest of THAT day, and the new
                    # clock-driven day rollover above releases it tomorrow.
                    if pos_contract_size and size > 0 and adaptive_multiplier > 0:
                        _lots = math.floor(size / pos_contract_size + 1e-9)
                        if _lots <= 0 and _size_before_adaptive >= pos_contract_size:
                            _lots = 1
                            adaptive_probe_count += 1
                        size = _lots * pos_contract_size

                if not math.isfinite(size) or size <= 0:
                    # Degenerate sizing (e.g. an ATR-based stop distance
                    # that rounds to ~0 for this bar) — skip this entry
                    # rather than opening a trade with an invalid size.
                    if adaptive_multiplier <= 0:
                        blocked_adaptive_zero_count += 1
                    if risk.contract_size and decision.skip_reason in _SIZING_SKIP_REASONS:
                        zero_size_contract_floor_count += 1
                        sizing_skip_reasons[decision.skip_reason] = sizing_skip_reasons.get(decision.skip_reason, 0) + 1
                    elif decision.skip_reason is not None:
                        sizing_skip_reasons[decision.skip_reason] = sizing_skip_reasons.get(decision.skip_reason, 0) + 1
                    pass  # n_today unchanged; no-op, kept for readability
                else:
                    if used_fallback_stop:
                        fallback_stop_count += 1
                    stop_price = None
                    take_price = None
                    if bar_sl_distance:
                        stop_price = entry_price - direction * bar_sl_distance
                    elif stop_loss_pips:
                        fixed_stop_distance = stop_loss_pips * risk.pip_size
                        stop_price = entry_price - direction * fixed_stop_distance
                        # Sanity-check the fixed pip stop against this bar's
                        # actual price level. A strategy's STOP_LOSS_PIPS is
                        # only meaningful if risk.pip_size matches the
                        # instrument it's being tested against (e.g. 0.0001
                        # for 4-decimal FX pairs) -- testing an FX-calibrated
                        # strategy against gold, an index, crypto, or a
                        # differently-quoted instrument while pip_size stays
                        # at its FX default silently turns a "25 pip" stop
                        # into a wildly wrong fraction of price, producing
                        # position sizes (and eventual losses) that bear no
                        # relation to the intended risk. This never changes
                        # behavior -- it only counts how often it happens so
                        # a clear warning can be raised below.
                        if raw_price:
                            distance_ratio = abs(fixed_stop_distance) / abs(raw_price)
                            if distance_ratio < 0.0002 or distance_ratio > 0.25:
                                pip_scale_mismatch_count += 1
                                if pip_scale_mismatch_worst_ratio is None or distance_ratio < pip_scale_mismatch_worst_ratio:
                                    pip_scale_mismatch_worst_ratio = distance_ratio
                        # Second, independent check: the same fixed-pips
                        # stop compared to this instrument's own recent ATR
                        # rather than to its raw price level. A stop can be
                        # a perfectly ordinary-looking fraction of price
                        # (comfortably inside the 0.02%-25% band above) and
                        # still be tiny next to how much this instrument
                        # actually moves bar to bar -- that's still a real
                        # pip_size/instrument mismatch, just one the
                        # price-ratio check alone can miss.
                        bar_atr = _atr_for_mismatch_check[i]
                        if bar_atr and bar_atr > 0:
                            atr_ratio = abs(fixed_stop_distance) / bar_atr
                            if atr_ratio < 0.15:
                                atr_scale_mismatch_count += 1
                                if atr_scale_mismatch_worst_ratio is None or atr_ratio < atr_scale_mismatch_worst_ratio:
                                    atr_scale_mismatch_worst_ratio = atr_ratio
                    if decision.stop_capped:
                        # fit_stop: 1 contract, stop shrunk to the distance
                        # whose worst case equals the budget. The target
                        # follows per risk.fit_stop_target (scale / keep /
                        # fixed_r); a trailing distance shrinks with the
                        # stop so the strategy's shape is preserved.
                        stop_price = entry_price - direction * decision.stop_distance
                        stop_capped_count += 1
                        _base_tp = bar_tp_distance if bar_tp_distance else (
                            take_profit_pips * risk.pip_size if take_profit_pips else None
                        )
                        if risk.fit_stop_target == "fixed_r":
                            bar_tp_distance = risk.fit_stop_target_r * decision.stop_distance
                        elif _base_tp and risk.fit_stop_target == "scale":
                            bar_tp_distance = _base_tp * decision.stop_scale
                        elif _base_tp:
                            bar_tp_distance = _base_tp
                        if bar_trail_distance:
                            bar_trail_distance = bar_trail_distance * decision.stop_scale
                    if decision.used_micro:
                        micro_fallback_count += 1
                    if bar_tp_distance:
                        take_price = entry_price + direction * bar_tp_distance
                    elif take_profit_pips:
                        take_price = entry_price + direction * take_profit_pips * risk.pip_size

                    initial_risk = abs(entry_price - stop_price) if stop_price is not None else None

                    # C6 (market-impact guardrail): a fill of this many
                    # whole contracts is assumed to land at the quoted
                    # price with zero market impact -- on real books a
                    # 30+ lot sweep walks the order book. Docs-only +
                    # warning: there is deliberately NO impact model here,
                    # because any particular impact curve would be a
                    # guess; the warning tells the user the fill assumed
                    # infinite liquidity. MAX_CONTRACTS_BEFORE_IMPACT_WARN
                    # is a module constant (not a RiskConfig field) so the
                    # threshold is visible at the warning site.
                    if pos_contract_size:
                        _whole_contracts = size / pos_contract_size
                        if _whole_contracts > MAX_CONTRACTS_BEFORE_IMPACT_WARN:
                            import warnings
                            warnings.warn(
                                f"MARKET IMPACT: entry of {_whole_contracts:.0f} whole contracts "
                                f"({_restore_tz(ts[fill_idx])}) filled at the quoted price with "
                                f"zero market impact assumed (above the {MAX_CONTRACTS_BEFORE_IMPACT_WARN}-contract "
                                f"guardrail). Real fills of this size move the book; this backtest does not model that.",
                                RuntimeWarning,
                            )

                    open_trade = {
                        "entry_time": _restore_tz(ts[fill_idx]),
                        "entry_bar_idx": fill_idx,
                        "direction": direction,
                        "entry_price": entry_price,
                        "size": size,
                        "initial_size": size,
                        "stop_price": stop_price,
                        "take_price": take_price,
                        "equity_at_entry": equity,
                        "best_price": entry_price,
                        "worst_price": entry_price,
                        "mfe_price": entry_price,
                        "initial_risk": initial_risk,
                        "intended_risk_dollars": intended_risk_dollars,
                        "breakeven_done": False,
                        "partial_taken": False,
                        "trailing_distance": bar_trail_distance,
                        "adaptive_multiplier": adaptive_multiplier,
                        "adaptive_rules_active": adaptive_rules_active,
                        "sized_above_risk_target": rescue_above_risk_target,
                        "contract_size": pos_contract_size,
                        "commission_per_contract": pos_commission_pc,
                        "risk_at_stop_dollars": risk.worst_case_loss(
                            size, abs(entry_price - stop_price) if stop_price is not None else sl_price_dist,
                            pos_contract_size, pos_commission_pc,
                        ),
                        "stop_capped": bool(decision.stop_capped),
                        "sizing_mode": risk.sizing_mode,
                        "used_micro": bool(decision.used_micro),
                        "contracts": (size / pos_contract_size) if pos_contract_size else None,
                        "fill_resolution": "bar",
                    }
                    trades_today_count[fill_bar_date] = n_today + 1
                    last_entry_bar_idx = fill_idx

        elif open_trade is None and sig[i] != 0:
            if account_blown:
                blocked_halted_count += 1
            elif daily_limit_breached:
                blocked_daily_limit_count += 1
            elif cooldown_active:
                blocked_cooldown_count += 1
            elif news_blackout_today:
                blocked_news_blackout_count += 1
            elif weekend_blocked_today:
                blocked_weekend_hold_count += 1

    # close any still-open trade at final bar close
    if open_trade is not None:
        i = n - 1
        direction = open_trade["direction"]
        _settle_exit(open_trade, closes[i], "end_of_data", direction, i)

    if prop_mode and not prop_done:
        prop_attempts.append({
            "attempt_id": attempt_id, "start_time": prop_attempt_start_time,
            "end_time": _restore_tz(ts[n - 1]) if n else None,
            "outcome": "passed_funded" if acct.passed_evaluation else "open",
            "failure_reason": None, "passed_evaluation": acct.passed_evaluation,
            "days_to_pass": acct.days_to_pass, "end_balance": acct.balance,
            "n_trades": acct.n_trades, "total_payout": acct.total_payout_amount,
            "max_dd_pct": acct.max_dd_pct_reached,
        })

    # UPGRADE (speed): building the DataFrame straight from the two
    # already-columnar numpy arrays (the original `ts` array + the
    # preallocated equity_arr) instead of a list of per-bar tuples avoids
    # pandas' row-wise tuple-unpacking path entirely -- see equity_arr's
    # definition above for why that path was expensive at 2M+ rows.
    # VAL-006: use the ORIGINAL (tz-aware, if the input was) timestamp
    # column here rather than the numpy-stripped `ts` array -- same root
    # cause as the Trade.entry_time/exit_time fix above. df's row order
    # matches equity_arr's build order exactly (one entry per bar, built
    # in a single forward pass over df), so reset_index(drop=True) is
    # only a defensive guard against a caller passing a non-default index.
    equity_df = pd.DataFrame({
        "timestamp": df["timestamp"].reset_index(drop=True),
        "equity": equity_arr,
    })
    # Additive, non-breaking: reset_on_breach details for anything downstream
    # that wants them (e.g. the report generator). Always present (empty
    # list when reset_on_breach is False or no reset ever fired) so callers
    # don't need to guard against a missing key.
    if prop_mode:
        equity_df["attempt_id"] = attempt_arr
        equity_df.attrs["prop_attempts"] = prop_attempts
        equity_df.attrs["prop_rules"] = prop_rules
    equity_df.attrs["account_reset_events"] = reset_events
    equity_df.attrs["payout_events"] = [ev for ev in reset_events if ev.get("kind") == "payout"]
    equity_df.attrs["breach_events"] = [ev for ev in reset_events if ev.get("kind", "breach") == "breach"]
    equity_df.attrs["zero_size_contract_floor_count"] = zero_size_contract_floor_count
    equity_df.attrs["min_contract_rescue_count"] = min_contract_rescue_count
    equity_df.attrs["sizing_summary"] = {
        "mode": risk.sizing_mode,
        "entries_considered": sizing_considered_count,
        "entries_taken": len([t for t in trades if t.exit_reason != "partial_take_profit"]),
        "skipped_for_sizing": zero_size_contract_floor_count,
        "skip_rate": (zero_size_contract_floor_count / sizing_considered_count) if sizing_considered_count else 0.0,
        "skip_reasons": dict(sizing_skip_reasons),
        "stop_capped": stop_capped_count,
        "micro_fallback": micro_fallback_count,
    }
    equity_df.attrs["adaptive_probe_count"] = adaptive_probe_count
    equity_df.attrs["entry_block_counts"] = {
        "engine_halted": blocked_halted_count,
        "daily_loss_limit": blocked_daily_limit_count,
        "reentry_cooldown": blocked_cooldown_count,
        "max_trades_per_day": blocked_max_trades_count,
        "zero_contract_sizing": zero_size_contract_floor_count,
        "adaptive_risk_zero_size": blocked_adaptive_zero_count,
        "news_blackout": blocked_news_blackout_count,
        "weekend_hold": blocked_weekend_hold_count,
    }
    equity_df.attrs["weekend_hold_close_count"] = weekend_close_count
    # Plain-language, chronological log of every account reset / payout, so a
    # report can show exactly when and why the account restarted.
    equity_df.attrs["account_reset_log"] = [ev["message"] for ev in reset_events if ev.get("message")]

    _breach_events = equity_df.attrs["breach_events"]
    _payout_events = equity_df.attrs["payout_events"]
    if _breach_events:
        import warnings
        warnings.warn(
            f"{len(_breach_events)} account reset(s) occurred (reset_on_breach=True): "
            "the account crossed the configured account-survivability floor "
            f"{len(_breach_events)} time(s) and was mechanically restarted at "
            "initial_balance each time, exactly like buying a new funded "
            "account, rather than being halted for the remainder of the run. "
            "Every trade after the first reset belongs to a DIFFERENT "
            "simulated account than the one before it -- see "
            "equity_df.attrs['account_reset_log'] for when each reset "
            "happened and what the account's equity was at that point. First: "
            + " | ".join(ev["message"] for ev in _breach_events[:3] if ev.get("message")),
            RuntimeWarning,
        )
    if _payout_events:
        total_payout = sum(ev["payout_amount"] for ev in _payout_events)
        import warnings
        warnings.warn(
            f"{len(_payout_events)} payout(s) taken (reset_on_target=True): the account crossed "
            f"its configured profit target {len(_payout_events)} time(s) and had the profit above "
            f"baseline withdrawn each time (${total_payout:,.2f} total across all payouts) rather "
            "than left to accumulate on an ever-growing equity number -- these are NOT breaches or "
            "losses. See equity_df.attrs['payout_events'] for when each payout happened and how much "
            "was withdrawn.",
            RuntimeWarning,
        )

    # STALL DIAGNOSTIC (2026-09-29): the engine is meant to trade to the last
    # bar. If it opened its last position in the first three quarters of the
    # data even though the strategy kept signalling afterwards, say so and
    # say WHY, instead of leaving a silent multi-year gap in the trade chart.
    if n > 20:
        _sig_arr = np.asarray(sig)
        _late_signal_bars = int(np.count_nonzero(_sig_arr[last_entry_bar_idx + 1:]))
        _gap_days = float((ts[n - 1] - ts[max(last_entry_bar_idx, 0)]) / np.timedelta64(1, "D"))
        # A real stall is a LONG stretch of calendar time with no entries; a
        # short dataset where the strategy just held a position is not one.
        if _late_signal_bars > 0 and last_entry_bar_idx < int(n * 0.75) and _gap_days >= 14:
            _blocks = ", ".join(
                f"{k.replace('_', ' ')}={v:,}" for k, v in equity_df.attrs["entry_block_counts"].items() if v
            ) or "none of the engine's own entry blocks fired -- the strategy's signal itself was the limit"
            import warnings
            _when = _restore_tz(ts[last_entry_bar_idx]) if last_entry_bar_idx >= 0 else "never"
            warnings.warn(
                f"TRADING STALL: the last entry was at {_when} "
                f"({(last_entry_bar_idx + 1) / n * 100:.0f}% of the way through the data) although the strategy "
                f"signalled on {_late_signal_bars:,} later bar(s). Entries blocked by: {_blocks}.",
                RuntimeWarning,
            )

    if account_blown:
        import warnings
        warnings.warn(
            f"Account BLOWN at {account_blown_at}: equity crossed the configured "
            f"account-survivability floor (max_account_drawdown_pct) or hit zero. "
            "No new trades were opened for the remainder of this run -- exactly "
            "like a real prop/broker account being terminated rather than being "
            "allowed to keep 'trading' itself into deeper negative equity. If you "
            "want to know what happens after a re-funded account, run a fresh "
            "backtest starting from that point rather than reading further trades "
            "on this run as real.",
            RuntimeWarning,
        )

    if clamped_loss_count:
        import warnings
        warnings.warn(
            f"{clamped_loss_count} trade(s) had their simulated loss capped at "
            "RiskConfig.max_trade_loss (a hard per-trade ceiling, default 3x the "
            "trade's own intended risk) instead of being allowed to realize the "
            "full computed loss. This models real negative-balance protection / "
            "firm loss floors -- without it, a pip_size mismatch or an extreme "
            "gap-through fill could report a single trade losing more than the "
            "entire account, which cannot happen live. If you see this often, "
            "the underlying cause (usually pip_scale_mismatch, see any warning "
            "above) is still worth fixing -- capping the number doesn't fix the "
            "sizing bug that produced it.",
            RuntimeWarning,
        )

    if force_closed_count:
        import warnings
        warnings.warn(
            f"{force_closed_count} trade(s) were force-closed intrabar because "
            "floating losses breached the configured daily_loss_limit_pct before "
            "the trade's own stop/target/signal exit would have fired. This "
            "mirrors a real prop firm auto-liquidating a funded account the "
            "moment equity crosses the daily floor, which realized-P&L-only "
            "accounting previously missed.",
            RuntimeWarning,
        )

    if fallback_stop_count:
        import warnings
        warnings.warn(
            f"{fallback_stop_count} trade(s) had no stop loss defined by the "
            f"strategy at all (no fixed pips, no ATR-based distance) — a "
            f"{DEFAULT_STOP_PCT_OF_PRICE * 100:.0f}%-of-price protective stop "
            "was used instead purely for sane position sizing and account "
            "protection. Add a real stop loss / STOP_LOSS_PIPS to the "
            "strategy for accurate results.",
            RuntimeWarning,
        )

    if min_contract_rescue_count:
        import warnings
        warnings.warn(
            f"{min_contract_rescue_count:,} entr{'y' if min_contract_rescue_count == 1 else 'ies'} were opened at the "
            f"1-contract minimum (contract_size={risk.contract_size:g}) under allow_single_contract_minimum=True even "
            "though ONE contract at the strategy's stop exceeds your risk budget. Those trades carry MORE than the "
            "configured risk per trade (tagged sized_above_risk_target). This is an explicit opt-in -- the default "
            "sizing modes never exceed the budget. Use sizing_mode='fit_stop' (shrink the stop to the budget) or "
            "'micro_fallback' (trade the micro contract) to stay on budget instead.",
            RuntimeWarning,
        )

    if adaptive_probe_count:
        import warnings
        warnings.warn(
            f"{adaptive_probe_count:,} entr{'y' if adaptive_probe_count == 1 else 'ies'} traded at the "
            f"1-contract minimum (contract_size={risk.contract_size:g}) because adaptive-risk "
            "throttling would otherwise have shrunk them below one whole contract. Adaptive risk "
            "scales size DOWN toward the minimum real position; it no longer reduces a trade to a "
            "fractional contract or to zero (a multiplier of exactly 0.0 -- e.g. the daily profit "
            "lock -- still blocks entries outright for the rest of that day). These trades carry "
            "more per-trade risk than the throttled target but never more than your unthrottled "
            "configured risk; raise risk_value or reduce the stop distance if you want the "
            "throttled sizes to stay above one contract on their own.",
            RuntimeWarning,
        )

    if zero_size_contract_floor_count:
        import warnings
        pct_of_signals = (
            f" ({zero_size_contract_floor_count / max(len(trades) + zero_size_contract_floor_count, 1) * 100:.0f}% "
            "of all potential entries)" if trades else " (every potential entry this run)"
        )
        warnings.warn(
            f"{zero_size_contract_floor_count} potential entr{'y' if zero_size_contract_floor_count == 1 else 'ies'}"
            f"{pct_of_signals} were skipped because position sizing rounded DOWN to 0 whole "
            f"contracts of this instrument (RiskConfig.contract_size={risk.contract_size:g}) at your "
            f"configured risk -- risk_value is too small to afford even ONE contract given this "
            "strategy's stop distance. This is the single most common reason a real, previously-"
            "profitable strategy suddenly shows 0 (or near-0) trades: the same strategy tested "
            "WITHOUT contract_size set (continuous/fractional sizing) can look completely normal, "
            "since fractional contracts aren't real but aren't floored to zero either. Fix by "
            "raising risk_value, switching to that instrument's micro contract (e.g. MGC instead of "
            "GC, MES instead of ES -- see app.data.instrument_specs), increasing account size, or "
            "tightening the strategy's stop distance.",
            RuntimeWarning,
        )

    # Part C (2026-10-04, silent sizing halt): the engine is CORRECT to skip
    # entries it can't afford at whole-contract size -- the bug was that a
    # run could go permanently quiet (96% of signals skipped, equity flat
    # for months on a profitable account) with the cause buried in one
    # warning line. Flag it machine-readably so the report generator can
    # banner it unmissably. Always present (not just when skips happened)
    # so downstream code never has to guard on the key.
    total_signals = len(trades) + zero_size_contract_floor_count
    halt_ratio = (zero_size_contract_floor_count / total_signals) if total_signals else 0.0
    equity_df.attrs["sizing_halt"] = {
        "halted": halt_ratio > 0.5,
        "skipped": zero_size_contract_floor_count,
        "skip_ratio": halt_ratio,
        "last_trade_exit": trades[-1].exit_time if trades else None,
        "risk_value": risk.risk_value,
        "contract_size": risk.contract_size,
        "sizing_mode": risk.sizing_mode,
        "skip_reasons": dict(sizing_skip_reasons),
    }

    if risk.commission_per_trade == 0.0 and risk.commission_per_contract == 0.0 and risk.slippage_pips == 0.0 and risk.spread_pips == 0.0:
        import warnings
        warnings.warn(
            "ZERO FRICTION: commission, slippage, AND spread are all 0.0 -- every trade is filled "
            "at the exact quoted price. Real fills are never free; use apply_instrument_spec() or set "
            "these explicitly before trusting net profit.",
            RuntimeWarning,
        )

    if pip_scale_mismatch_count:
        import warnings
        warnings.warn(
            f"{pip_scale_mismatch_count} trade(s) had a fixed-pips stop "
            f"whose price distance was an implausible fraction of this "
            f"instrument's price (as small as {pip_scale_mismatch_worst_ratio:.5%} "
            f"of price on the worst one). This almost always means the "
            f"configured pip_size ({risk.pip_size}) doesn't match the "
            "instrument actually being tested — e.g. an FX-calibrated "
            "strategy (pip_size 0.0001) run against gold, an index, crypto, "
            "or a JPY pair. Position sizing and every stop distance for "
            "these trades is unreliable. Set pip_size to match the real "
            "instrument (e.g. 0.01 for gold/JPY pairs) in the Risk "
            "configuration before trusting this result.",
            RuntimeWarning,
        )

    if atr_scale_mismatch_count:
        import warnings
        warnings.warn(
            f"{atr_scale_mismatch_count} trade(s) had a fixed-pips stop "
            f"under 15% of this instrument's own recent ATR (as little as "
            f"{atr_scale_mismatch_worst_ratio:.1%} of one bar's typical "
            f"range on the worst one) — a stop that tight gets taken out "
            f"by ordinary noise almost every time, regardless of whether "
            f"the strategy's entry logic is any good. This can happen even "
            f"when the stop looks like a normal fraction of price (see "
            f"pip_scale_mismatch above, if also shown): a high-priced but "
            f"volatile instrument, such as an equity index priced in the "
            f"thousands, can still move far more per bar than a fixed pip "
            f"count sized for a lower-volatility instrument like gold or "
            f"FX ever assumed. Set pip_size to match the real instrument "
            "(or click \"detect pip size from data\" on the Data tab) "
            "before trusting this result, or use an ATR-based dynamic stop "
            "instead of a fixed pip count.",
            RuntimeWarning,
        )

    if gap_loss_count:
        import warnings
        warnings.warn(
            f"{gap_loss_count} trade(s) hit their stop loss but filled at a "
            "price more than 3x further away than the intended risk -- the "
            "market gapped straight past the resting stop within a single "
            "bar. This is a real, honestly-simulated gap-through fill (see "
            "the 'honest fill' comment above), not an engine bug, but it "
            "means a handful of trades lost far more than the strategy's "
            "nominal per-trade risk. Worth checking whether this instrument/"
            "timeframe is prone to large single-bar gaps (news events, "
            "weekend opens, thin low-timeframe data) before trusting the "
            "risk-of-ruin numbers.",
            RuntimeWarning,
        )

    return trades, equity_df
