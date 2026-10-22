"""
Unified portfolio engine: ONE account, N instruments, several open positions, equity marked to market every bar,
and the account rules enforced on the COMBINED equity.

Why this exists
---------------
``run_backtest`` / ``run_execution`` simulate one instrument and one open position. ``app.portfolio.portfolio``
(the original portfolio) worked around that by backtesting each leg on its OWN account and then adding the trade
P&Ls in close-time order -- fine for realised P&L, but blind to everything that only exists when positions
overlap: floating losses that add up across instruments, a trailing drawdown that is hit by the SUM of two
open trades, a daily-loss limit shared by every leg, a contract/margin cap on the whole book.

This engine steps a master clock (``app.data.portfolio_clock``) and carries per-instrument position state:

  * every leg is prepared by ``app.backtest.engine.prepare_run`` -- the exact preparation ``run_backtest`` does
    (timeframe alignment, signal filters, behaviour rules), so a leg is set up identically to a standalone run;
  * every fill, cost, stop/target resolution, partial exit and position size comes from ``app.backtest.fills`` --
    the same functions ``run_execution`` calls (they were extracted from it, not re-implemented);
  * account equity = realised balance + the floating P&L of ALL open positions at each step;
  * the account rules -- legacy daily-loss limit, loss-floor breach/reset, profit-target payouts, session
    flatten, weekend hold, news blackout, and the full ``PropAccount`` (trailing / static / end-of-day /
    floating drawdown, floating daily loss, profit target, payouts) -- run on that combined equity;
  * a portfolio-level cap on open positions, total contracts, margin and open risk.

Parity contract (tests/backtest/test_portfolio_engine.py): a one-leg portfolio reproduces ``run_backtest``
exactly -- same trades, same equity curve, same account events -- and two legs whose trades never overlap
reproduce the original ``run_portfolio_backtest`` combination of independent legs.

Documented modelling choices (they only matter when several positions are open at once)
---------------------------------------------------------------------------------------
  * Worst-case intrabar ordering. Within one step, the floating-equity test assumes EVERY open leg is at its
    adverse extreme at the same instant (and, for the trailing-drawdown high-water mark, at its favourable
    extreme at the same instant). Bars carry no intra-bar ordering across instruments, so this is the
    conservative bound -- it can liquidate a book that a tick-by-tick replay would have saved, never the reverse.
  * Liquidation. When floating equity touches a drawdown / daily-loss floor the account is liquidated: every open
    position is closed at the price implied by scaling each leg's adverse move until the combined loss equals the
    floor (clamped to the bar's range), the same rule ``run_execution`` applies to one position.
  * Capital priority. At one step, legs are considered for entry in the order they were supplied; earlier legs
    claim position / contract / margin capacity first.
  * A closed market. A leg with no bar at a step cannot trigger anything; its open position is carried at its
    last close. (See app.data.portfolio_clock.)
  * Account checks after a stop-out. ``run_execution`` skips its end-of-bar account checks (prop failure,
    breach, payout) on a bar where a position was stopped out intrabar and runs them on the next bar; this engine
    keeps that rule (it is what makes a one-leg portfolio identical to ``run_backtest``). Set
    ``PortfolioEngineConfig.immediate_account_checks=True`` to run them on the same step instead.

Unsupported (raises ``PortfolioEngineError`` rather than silently approximating): resting-order / zone
strategies (``StrategyResult.entry_orders``), which fill on limit prices rather than bar signals.
"""
from __future__ import annotations

import math
import warnings as _warnings
from dataclasses import dataclass, field, replace as _dc_replace

import numpy as np
import pandas as pd

from app.backtest import fills as _fills
from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskState, volatility_percentile_series
from app.backtest.engine import PreparedRun, prepare_run
from app.backtest.execution import Trade
from app.backtest.risk import RiskConfig
from app.backtest.statistics import BacktestStatistics, compute_statistics
from app.data.portfolio_clock import PortfolioClock, PortfolioClockError, _utc_ns, build_portfolio_clock
from app.strategy.base import Strategy

try:
    from app.data.trading_day import trading_days
except ImportError:  # pragma: no cover
    trading_days = None


class PortfolioEngineError(Exception):
    """Raised when a portfolio cannot be simulated (bad inputs or an unsupported strategy type)."""


@dataclass
class EngineLeg:
    """One instrument/strategy in the book.

    ``risk`` carries the INSTRUMENT-level settings (pip size, contract size, spread/slippage/commission, sizing
    mode and risk per trade, fill lag, cooldown, max trades per day, time stop). The account-level settings
    (balance, daily loss, drawdown floor, prop rules, ...) come from ``PortfolioEngineConfig.account``.
    ``weight`` scales the leg's ``risk_value`` (1.0 = as configured)."""
    name: str
    df: pd.DataFrame
    strategy: Strategy
    risk: RiskConfig
    weight: float = 1.0
    adaptive_risk: AdaptiveRiskConfig | None = None
    intrabar_df: pd.DataFrame | None = None


@dataclass
class PortfolioEngineConfig:
    account: RiskConfig | None = None            # account-level rules; None -> the first leg's RiskConfig
    window: str = "union"                        # "union" | "overlap" (see build_portfolio_clock)
    size_basis: str = "account"                  # "account": size every leg off the shared balance;
                                                 # "leg": size each leg off its own balance track (initial + its own
                                                 #        realised P&L) -- reproduces independently-run legs
    # --- portfolio-level capacity (None = unlimited) -----------------------------------------------------
    max_open_positions: int | None = None
    max_total_contracts: float | None = None     # whole contracts summed across open positions
    max_open_risk_pct: float | None = None       # sum of every open position's $ risk at its stop, % of balance
    margin_per_contract: dict | None = None      # {leg name: $ margin per contract}
    max_margin_pct: float | None = None          # margin in use, % of balance
    max_trades_per_day_total: int | None = None  # across all legs, per session day
    cap_policy: str = "clip"                     # "clip": shrink an entry to the capacity left; "skip": refuse it
    immediate_account_checks: bool = False       # see module docstring
    record_leg_curves: bool = True               # per-leg cumulative P&L curves (memory: legs x steps x 8 bytes)
    compute_stats: bool = True
    lookahead_check: str = "auto"                # forwarded to prepare_run ("skip" for inner search loops)


@dataclass
class LegSummary:
    name: str
    trades: int
    net_profit: float
    win_rate: float | None
    entries_considered: int
    entries_skipped_for_sizing: int
    blocked_by_portfolio_caps: int
    clipped_by_portfolio_caps: int
    bars: int
    coverage: float
    warnings: list = field(default_factory=list)


@dataclass
class PortfolioEngineResult:
    trades: list                                 # list[Trade], in settlement order (Trade.instrument names the leg)
    equity_curve: pd.DataFrame                   # master clock, mark-to-market combined equity (+attempt_id in prop mode)
    statistics: BacktestStatistics | None
    initial_balance: float
    legs: list                                   # list[LegSummary]
    clock: PortfolioClock
    portfolio_stats: dict                        # concurrency / exposure / contribution (see _portfolio_stats)
    leg_curves: dict | None = None               # {name: DataFrame[timestamp, pnl]} cumulative realised+floating P&L per leg
    warnings: list = field(default_factory=list)
    lookahead_check_failed: bool = False

    @property
    def combined_trades(self) -> list:
        """Trades in chronological (exit-time) order; stable, so ties keep settlement order."""
        return sorted(self.trades, key=lambda t: t.exit_time)

    def trades_for(self, name: str) -> list:
        return [t for t in self.trades if t.instrument == name]


# ======================================================================================================================
class _Leg:
    """Per-instrument state while the engine runs."""
    __slots__ = (
        "k", "name", "risk", "prep", "adaptive_risk", "n", "ts", "tz", "opens", "highs", "lows", "closes", "sig",
        "sl_vals", "tp_vals", "trail_vals", "stop_loss_pips", "take_profit_pips", "breakeven_r", "partial_cfg",
        "partial_r", "partial_frac", "partial_be", "atr", "vol_pct", "fill_lag", "open", "last_close_bar_idx",
        "last_entry_bar_idx", "trades_today", "leg_equity", "ib_on", "ib_ts", "ib_hi", "ib_lo", "bar_span",
        "considered", "floor_skips", "skip_reasons", "stop_capped", "micro", "fallback_stops", "pip_mismatch",
        "pip_worst", "atr_mismatch", "atr_worst", "rescue", "probe", "adaptive_zero", "blocked_cap", "clipped_cap",
        "blocked_daily", "blocked_cooldown", "blocked_maxtrades", "blocked_news", "blocked_weekend", "blocked_halt",
        "impact", "gap_losses", "net", "n_trades", "wins", "step", "bar_at", "last_bar", "warn",
    )


def _atr_yardstick(df: pd.DataFrame) -> np.ndarray:
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    prev = np.empty_like(hi)
    prev[0] = np.nan
    prev[1:] = cl[:-1]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev), np.abs(lo - prev)))
    return pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()


def run_portfolio_engine(legs: list, config: PortfolioEngineConfig | None = None) -> PortfolioEngineResult:
    """Simulate ``legs`` on one account. See the module docstring."""
    return _PortfolioEngine(list(legs), config or PortfolioEngineConfig()).run()


class _PortfolioEngine:
    def __init__(self, legs: list, cfg: PortfolioEngineConfig):
        if not legs:
            raise PortfolioEngineError("A portfolio needs at least one leg.")
        names = [leg.name for leg in legs]
        if len(set(names)) != len(names):
            raise PortfolioEngineError("Leg names must be unique.")
        if cfg.size_basis not in ("account", "leg"):
            raise PortfolioEngineError("size_basis must be 'account' or 'leg'.")
        if cfg.cap_policy not in ("clip", "skip"):
            raise PortfolioEngineError("cap_policy must be 'clip' or 'skip'.")
        self.cfg = cfg
        self.input_legs = legs

    # ------------------------------------------------------------------------------------------------------------------
    def run(self) -> PortfolioEngineResult:
        cfg = self.cfg
        in_legs = self.input_legs
        account: RiskConfig = cfg.account if cfg.account is not None else in_legs[0].risk
        K = len(in_legs)

        # ---- prepare every leg exactly as run_backtest would -------------------------------------------------------
        preps: list[PreparedRun] = []
        for leg in in_legs:
            leg_risk = leg.risk
            if leg.weight != 1.0:
                leg_risk = _dc_replace(leg_risk, risk_value=leg_risk.risk_value * leg.weight)
            leg_risk = _dc_replace(leg_risk, initial_balance=account.initial_balance)
            try:
                preps.append(prepare_run(leg.df, leg.strategy, leg_risk, leg.adaptive_risk, leg.intrabar_df,
                                         cfg.lookahead_check))
            except Exception as exc:  # noqa: BLE001
                raise PortfolioEngineError(f"Leg '{leg.name}' could not be prepared: {type(exc).__name__}: {exc}") from exc
            if getattr(preps[-1].strat_result, "entry_orders", None) is not None:
                raise PortfolioEngineError(
                    f"Leg '{leg.name}' is a resting-order (zone/limit) strategy; the portfolio engine only supports "
                    "bar-signal strategies for now. Run it standalone or convert it to a signal strategy.")

        try:
            clock = build_portfolio_clock([p.df for p in preps], [leg.name for leg in in_legs], cfg.window)
        except PortfolioClockError as exc:
            raise PortfolioEngineError(str(exc)) from exc
        M = len(clock)
        master = clock.master

        # ---- account-level setup (mirrors run_execution's preamble) ------------------------------------------------
        equity = float(account.initial_balance)
        adaptive_state = AdaptiveRiskState(initial_balance=account.initial_balance)
        trades: list[Trade] = []
        warn_msgs: list[str] = []

        daily_limit_amount = (
            account.initial_balance * (account.daily_loss_limit_pct / 100.0)
            if account.daily_loss_limit_pct is not None else None
        )
        blown_floor = account.account_blown_floor()
        prop_mode = getattr(account, "account_model", "legacy") == "prop"
        if prop_mode:
            daily_limit_amount = None
            blown_floor = None
        halt_on_breach = bool(getattr(account, "halt_on_breach", False))
        payout_target_baseline = account.initial_balance
        account_blown = False
        account_blown_at = None
        reset_events: list[dict] = []
        attempt_id = 0
        force_closed_count = 0
        clamped_loss_count = 0
        weekend_close_count = 0
        flatten_close_count = 0
        weekend_entry_block = False
        total_trades_today_blocked = 0

        # ---- session days / calendar arrays on the MASTER clock -----------------------------------------------------
        _ts_days = pd.DatetimeIndex(master)
        if trading_days is not None:
            bar_dates = trading_days(_ts_days, tz="America/Chicago", roll_hour=17)
        else:  # pragma: no cover
            bar_dates = _ts_days.normalize().to_numpy()
        unique_days, day_idx = np.unique(bar_dates, return_inverse=True)
        n_days = len(unique_days)
        pnl_today_sum = np.zeros(n_days, dtype=np.float64)
        day_has_pnl = np.zeros(n_days, dtype=bool)
        total_trades_today = np.zeros(n_days, dtype=np.int64)

        news_blackout_bar = np.zeros(M, dtype=bool)
        news_csv = getattr(account, "news_blackout_csv", None)
        if news_csv:
            dates_set: set = set()
            try:
                with open(news_csv, "r", encoding="utf-8") as fh:
                    for line in fh:
                        s = line.strip()
                        if s and not s.startswith("#"):
                            dates_set.add(pd.Timestamp(s).date())
            except (OSError, ValueError) as exc:
                warn_msgs.append(f"news_blackout_csv '{news_csv}' could not be read/parsed ({exc}); news blackout is DISABLED for this run.")
            if dates_set:
                bd = pd.DatetimeIndex(bar_dates).date
                news_blackout_bar = np.fromiter((d in dates_set for d in bd), dtype=bool, count=M)
        nbw = getattr(account, "news_blackout_windows", None)
        if nbw and M:
            try:
                from app.backtest.behavior_rules import parse_windows, window_mask
                news_blackout_bar = news_blackout_bar | window_mask(master, parse_windows(nbw))
            except Exception as exc:  # noqa: BLE001
                warn_msgs.append(f"news_blackout_windows could not be applied ({exc}); window blackout is DISABLED for this run.")

        weekend_hold = bool(getattr(account, "block_weekend_hold", False))
        dow = pd.DatetimeIndex(bar_dates).dayofweek.to_numpy()
        week_ends_at = np.zeros(M, dtype=bool)
        if M:
            week_ends_at[-1] = True
            week_ends_at[:-1] = dow[1:] <= dow[:-1]

        # ---- prop account -------------------------------------------------------------------------------------------
        acct = None
        prop_rules = None
        prop_day_map: dict = {}
        prop_day_dates: list = []
        prop_pending = None
        prop_locked_bar_date = -1
        prop_attempts: list[dict] = []
        prop_payouts_seen = 0
        prop_first_payout_time = None
        prop_done = False
        prop_attempt_start_time = None
        funded_reset_logged: list = []
        DAY_LOCKED = FAILED = None
        PropAccount = None
        prop_total_contract_cap = None
        if prop_mode:
            from app.prop.account import PropAccount, coerce_rules, DAY_LOCKED, FAILED
            prop_rules = coerce_rules(getattr(account, "prop_account_rules", None))
            if prop_rules is None:
                raise PortfolioEngineError("account.account_model='prop' requires account.prop_account_rules")
            if abs(float(prop_rules.account_size) - float(account.initial_balance)) > 1e-6:
                warn_msgs.append(
                    f"prop rules account_size ({prop_rules.account_size:,.2f}) != account.initial_balance "
                    f"({account.initial_balance:,.2f}); the firm's account size is used for the run.")
                equity = float(prop_rules.account_size)
            if getattr(prop_rules, "max_contracts", None):
                prop_total_contract_cap = int(prop_rules.max_contracts)

        flat_zone = np.zeros(M, dtype=bool)
        flatten_str = getattr(prop_rules, "flatten_time_ct", None) if prop_mode else None
        risk_flat = getattr(account, "flatten_time_ct", None)
        if risk_flat:
            try:
                rfm = int(str(risk_flat).split(":")[0]) * 60 + int(str(risk_flat).split(":")[1])
                pfm = (int(str(flatten_str).split(":")[0]) * 60 + int(str(flatten_str).split(":")[1])) if flatten_str else None
                if pfm is None or rfm < pfm:
                    flatten_str = str(risk_flat)
            except Exception:  # noqa: BLE001
                pass
        if flatten_str and M:
            try:
                fh_, fm_ = (int(x) for x in str(flatten_str).split(":"))
                ct = pd.DatetimeIndex(master).tz_localize("UTC").tz_convert("America/Chicago")
                mod = (ct.hour * 60 + ct.minute).to_numpy()
                flat_zone = (mod >= fh_ * 60 + fm_) & (mod < 17 * 60)
            except Exception:  # noqa: BLE001
                flat_zone = np.zeros(M, dtype=bool)

        acct_tz = getattr(preps[0].df["timestamp"].dtype, "tz", None)

        def restore_tz(ts_value, tz) -> pd.Timestamp:
            t = pd.Timestamp(ts_value)
            return t.tz_localize("UTC").tz_convert(tz) if tz is not None else t

        def at_time(m_: int) -> pd.Timestamp:
            return restore_tz(master[m_], acct_tz)

        # ---- per-leg state -------------------------------------------------------------------------------------------
        L: list[_Leg] = []
        for k, (leg_in, prep) in enumerate(zip(in_legs, preps)):
            lg = _Leg()
            lg.k, lg.name, lg.prep = k, leg_in.name, prep
            risk = prep.risk
            if prop_total_contract_cap is not None:
                _mc = prop_total_contract_cap
                if risk.max_contracts is not None:
                    _mc = min(_mc, int(risk.max_contracts))
                risk = _dc_replace(risk, max_contracts=_mc)
            lg.risk = risk
            lg.adaptive_risk = prep.adaptive_risk
            df = prep.df
            lg.n = len(df)
            lg.ts = _utc_ns(df["timestamp"])
            lg.tz = getattr(df["timestamp"].dtype, "tz", None)
            lg.opens = df["open"].to_numpy(dtype=float)
            lg.highs = df["high"].to_numpy(dtype=float)
            lg.lows = df["low"].to_numpy(dtype=float)
            lg.closes = df["close"].to_numpy(dtype=float)
            sr = prep.strat_result
            lg.sig = sr.signals.values
            lg.sl_vals = sr.stop_loss_distance.values if sr.stop_loss_distance is not None else None
            lg.tp_vals = sr.take_profit_distance.values if sr.take_profit_distance is not None else None
            lg.trail_vals = sr.trailing_stop_distance.values if sr.trailing_stop_distance is not None else None
            lg.stop_loss_pips, lg.take_profit_pips = sr.stop_loss_pips, sr.take_profit_pips
            lg.breakeven_r = sr.breakeven_trigger_r
            lg.partial_cfg = sr.partial_exit
            lg.partial_r = float(sr.partial_exit["r_multiple"]) if sr.partial_exit else None
            lg.partial_frac = float(sr.partial_exit["fraction"]) if sr.partial_exit else None
            lg.partial_be = bool(sr.partial_exit.get("move_stop_to_breakeven", True)) if sr.partial_exit else False
            lg.atr = _atr_yardstick(df)
            lg.vol_pct = None
            if lg.adaptive_risk is not None and lg.adaptive_risk.uses_volatility_trigger():
                lg.vol_pct = volatility_percentile_series(lg.atr, lookback_bars=lg.adaptive_risk.volatility_lookback_bars)
            lg.fill_lag = max(int(risk.entry_fill_lag_bars or 0), 0)
            lg.open = None
            lg.last_close_bar_idx = -10 ** 9
            lg.last_entry_bar_idx = -1
            lg.trades_today = np.zeros(n_days, dtype=np.int64)
            lg.leg_equity = float(account.initial_balance)
            lg.step = clock.leg_step[k]
            lg.bar_at = clock.bar_at[k]
            lg.last_bar = clock.last_bar[k]
            # intrabar (finer-bar) fill-order replay
            lg.ib_on = False
            lg.ib_ts = lg.ib_hi = lg.ib_lo = None
            lg.bar_span = None
            if getattr(risk, "intrabar_replay", False) and prep.intrabar_df is not None and len(prep.intrabar_df) > 0:
                ibd = prep.intrabar_df.sort_values("timestamp")
                ibt = pd.to_datetime(ibd["timestamp"])
                if getattr(ibt.dt, "tz", None) is not None:
                    ibt = ibt.dt.tz_convert("UTC").dt.tz_localize(None)
                lg.ib_ts = ibt.values.astype("datetime64[ns]")
                lg.ib_hi = ibd["high"].to_numpy(dtype=np.float64)
                lg.ib_lo = ibd["low"].to_numpy(dtype=np.float64)
                lg.ib_on = True
                if lg.n > 1:
                    d = np.diff(lg.ts.astype("datetime64[ns]"))
                    lg.bar_span = np.median(d) if len(d) else None
            for a in ("considered", "floor_skips", "stop_capped", "micro", "fallback_stops", "pip_mismatch", "atr_mismatch",
                      "rescue", "probe", "adaptive_zero", "blocked_cap", "clipped_cap", "blocked_daily", "blocked_cooldown",
                      "blocked_maxtrades", "blocked_news", "blocked_weekend", "blocked_halt", "gap_losses", "n_trades", "wins"):
                setattr(lg, a, 0)
            lg.net = 0.0
            lg.skip_reasons = {}
            lg.pip_worst = lg.atr_worst = None
            lg.impact = []
            lg.warn = list(prep.timeframe_warnings) + list(prep.condition_warnings)
            if prep.lookahead_result is not None and getattr(prep.lookahead_result, "bug_detected", False):
                lg.warn.append("LOOKAHEAD BIAS DETECTED: " + prep.lookahead_result.summary())
            if prep.lookahead_error is not None:
                lg.warn.append("LOOKAHEAD CHECK CRASHED: " + prep.lookahead_error)
            # warm-up fills etc. are handled inside prepare_run; nothing else to do per leg here
            L.append(lg)
        any_lookahead_failed = any(p.lookahead_error is not None for p in preps)

        # ---- outputs -------------------------------------------------------------------------------------------------
        equity_arr = np.empty(M, dtype=np.float64)
        attempt_arr = np.zeros(M, dtype=np.int32)
        n_open_arr = np.zeros(M, dtype=np.int16)
        gross_contracts_arr = np.zeros(M, dtype=np.float32)
        margin_arr = np.zeros(M, dtype=np.float32) if cfg.margin_per_contract else None
        track_curves = bool(cfg.record_leg_curves) and (M * K) <= 12_000_000
        leg_pnl_arr = np.zeros((K, M), dtype=np.float64) if track_curves else None
        leg_realised = np.zeros(K, dtype=np.float64)
        prop_last_bar_date = None
        sizing_considered_total = 0
        immediate = bool(cfg.immediate_account_checks)

        # ---------------------------------------------------------------------------------------------------------------
        def exit_cause(reason, open_pos):
            return _fills.exit_cause_for(reason, open_pos)

        def prop_day(bar_date_: int) -> int:
            d = prop_day_map.get(bar_date_)
            if d is None:
                d = len(prop_day_dates)
                prop_day_map[bar_date_] = d
                prop_day_dates.append(unique_days[bar_date_])
            return d

        def prop_new_account(m_: int) -> None:
            nonlocal acct, prop_payouts_seen, prop_pending, prop_attempt_start_time, prop_first_payout_time
            acct = PropAccount(prop_rules, start_day_index=len(prop_day_dates), day_dates=prop_day_dates)
            acct.start_date = unique_days[day_idx[min(m_, M - 1)]]
            prop_payouts_seen = 0
            prop_first_payout_time = None
            prop_pending = None
            prop_attempt_start_time = at_time(min(m_, M - 1))

        def prop_feed_close(pnl_: float, bar_date_: int, risk_dollars, m_: int) -> None:
            nonlocal equity, prop_pending, prop_locked_bar_date, prop_payouts_seen, prop_first_payout_time
            pday = prop_day(bar_date_)
            res = acct.on_trade_close(
                pnl_, pday, is_last_of_day=(getattr(acct, "dd_basis", None) != "eod"),
                trade_initial_risk=risk_dollars, floating_handled=True,
            )
            if res == FAILED:
                prop_pending = "failed"
            elif res == DAY_LOCKED:
                prop_locked_bar_date = bar_date_
            if len(acct.payouts) > prop_payouts_seen:
                if prop_first_payout_time is None:
                    prop_first_payout_time = at_time(m_)
                for rec in acct.payouts[prop_payouts_seen:]:
                    reset_events.append({
                        "kind": "payout", "reset_at": at_time(m_), "attempt_id": attempt_id,
                        "payout_amount": rec.amount, "equity_before_payout": rec.balance_after + rec.amount,
                        "equity_after_payout": rec.balance_after,
                        "message": (f"{at_time(m_)}: PROFIT TARGET REACHED - attempt {attempt_id}: payout "
                                    f"${rec.amount:,.2f} banked; account balance ${rec.balance_after:,.2f}; trading continues."),
                    })
                prop_payouts_seen = len(acct.payouts)
            if (getattr(prop_rules, "funded_starts_fresh", False) and acct.passed_evaluation
                    and attempt_id not in funded_reset_logged):
                funded_reset_logged.append(attempt_id)
                reset_events.append({
                    "kind": "funded_reset", "reset_at": at_time(m_), "attempt_id": attempt_id,
                    "equity_before_reset": equity + pnl_, "equity_after_reset": acct.balance,
                    "message": (f"{at_time(m_)}: EVALUATION PASSED - attempt {attempt_id}: funded account starts "
                                f"fresh at ${acct.balance:,.2f} (evaluation profit is not carried over)."),
                })
            equity = acct.balance

        def record_trade(lg: _Leg, trade: Trade, pnl: float) -> None:
            trades.append(trade)
            lg.net += pnl
            lg.n_trades += 1
            if pnl > 0:
                lg.wins += 1

        def settle_exit(lg: _Leg, pos: dict, raw_exit_price: float, reason: str, direction_: int,
                        bar_idx: int, m_fill: int) -> float:
            """The one place any exit becomes a settled trade. ``bar_idx`` is the leg's own bar index (cooldown),
            ``m_fill`` the master step the fill happens on (exit time, session day)."""
            nonlocal equity, clamped_loss_count
            lg.last_close_bar_idx = bar_idx
            sm = _fills.settle_math(lg.risk, pos, raw_exit_price, reason, direction_)
            filled_exit_price, pnl, commission, reason = sm.filled_exit_price, sm.pnl, sm.commission, sm.reason
            if sm.gap_loss:
                lg.gap_losses += 1
            pnl, clamped = _fills.clamp_trade_loss(lg.risk, pnl, pos["equity_at_entry"])
            if clamped:
                clamped_loss_count += 1
            equity += pnl
            lg.leg_equity += pnl
            leg_realised[lg.k] += pnl
            record_trade(lg, Trade(
                entry_time=pos["entry_time"], exit_time=at_time(m_fill) if lg.tz == acct_tz else restore_tz(master[m_fill], lg.tz),
                direction=direction_, entry_price=pos["entry_price"], exit_price=filled_exit_price, size=pos["size"],
                pnl=pnl, pnl_pct=(pnl / pos["equity_at_entry"]) * 100 if pos["equity_at_entry"] else 0.0,
                exit_reason=reason, commission=commission, equity_after=equity,
                initial_risk=pos["initial_risk"], adaptive_risk_multiplier=pos["adaptive_multiplier"],
                adaptive_risk_rules_active=tuple(pos["adaptive_rules_active"]),
                intended_risk_dollars=pos.get("intended_risk_dollars"), worst_price=pos.get("worst_price"),
                mfe_price=pos.get("mfe_price"), exit_cause=exit_cause(reason, pos),
                sized_above_risk_target=bool(pos.get("sized_above_risk_target", False)),
                risk_at_stop_dollars=pos.get("risk_at_stop_dollars"), stop_capped=bool(pos.get("stop_capped", False)),
                sizing_mode=pos.get("sizing_mode"), used_micro=bool(pos.get("used_micro", False)),
                contracts=pos.get("contracts"), attempt_id=attempt_id, fill_resolution=pos.get("fill_resolution", "bar"),
                instrument=lg.name,
            ), pnl)
            bd = day_idx[m_fill]
            adaptive_state.record_trade_close(pnl, is_new_day=not day_has_pnl[bd])
            pnl_today_sum[bd] += pnl
            day_has_pnl[bd] = True
            if prop_mode:
                prop_feed_close(pnl, bd, pos.get("risk_at_stop_dollars"), m_fill)
            return pnl

        def settle_partial(lg: _Leg, pos: dict, fill_price: float, direction_: int, m_fill: int) -> float:
            nonlocal equity, clamped_loss_count
            pm = _fills.partial_settle_math(lg.risk, pos, fill_price, direction_, lg.partial_frac)
            pnl, clamped = _fills.clamp_trade_loss(lg.risk, pm.pnl, pos["equity_at_entry"])
            if clamped:
                clamped_loss_count += 1
            equity += pnl
            lg.leg_equity += pnl
            leg_realised[lg.k] += pnl
            record_trade(lg, Trade(
                entry_time=pos["entry_time"], exit_time=at_time(m_fill) if lg.tz == acct_tz else restore_tz(master[m_fill], lg.tz),
                direction=direction_, entry_price=pos["entry_price"], exit_price=pm.filled_exit_price, size=pm.partial_size,
                pnl=pnl, pnl_pct=(pnl / pos["equity_at_entry"]) * 100 if pos["equity_at_entry"] else 0.0,
                exit_reason="partial_take_profit", commission=pm.commission, equity_after=equity,
                initial_risk=pos["initial_risk"], adaptive_risk_multiplier=pos["adaptive_multiplier"],
                adaptive_risk_rules_active=tuple(pos["adaptive_rules_active"]),
                intended_risk_dollars=pos.get("intended_risk_dollars"), worst_price=pos.get("worst_price"),
                mfe_price=pos.get("mfe_price"), exit_cause="take_profit",
                sized_above_risk_target=bool(pos.get("sized_above_risk_target", False)),
                risk_at_stop_dollars=pos.get("risk_at_stop_dollars"), stop_capped=bool(pos.get("stop_capped", False)),
                sizing_mode=pos.get("sizing_mode"), used_micro=bool(pos.get("used_micro", False)),
                contracts=pos.get("contracts"), attempt_id=attempt_id, fill_resolution=pos.get("fill_resolution", "bar"),
                instrument=lg.name,
            ), pnl)
            pos["size"] -= pm.partial_size
            bd = day_idx[m_fill]
            pnl_today_sum[bd] += pnl
            day_has_pnl[bd] = True
            if prop_mode:
                prop_feed_close(pnl, bd, None, m_fill)
            return pnl

        def intrabar_first_hit(lg: _Leg, direction_: int, stop_, take_, i_: int):
            if not lg.ib_on or stop_ is None or take_ is None:
                return None
            t0 = lg.ts[i_].astype("datetime64[ns]")
            t1 = lg.ts[i_ + 1].astype("datetime64[ns]") if i_ + 1 < lg.n else (t0 + lg.bar_span if lg.bar_span is not None else None)
            if t1 is None:
                return None
            a = int(np.searchsorted(lg.ib_ts, t0, side="left"))
            b = int(np.searchsorted(lg.ib_ts, t1, side="left"))
            if b <= a:
                return None
            hi = lg.ib_hi[a:b]
            lo = lg.ib_lo[a:b]
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

        def mark_price(lg: _Leg, m_: int) -> float:
            """Last known close of the leg at or before step m_ (the as-of mark)."""
            return lg.closes[lg.last_bar[m_]]

        def close_leg_here(lg: _Leg, m_: int, reason: str, price_if_bar: str = "open"):
            """Force-close the leg's position at step m_: at that bar's open/close if the leg has a bar here,
            else at its last close. Returns nothing; clears lg.open."""
            pos = lg.open
            i_ = lg.bar_at[m_]
            if i_ >= 0:
                px = lg.opens[i_] if price_if_bar == "open" else lg.closes[i_]
                bi = i_
            else:
                bi = lg.last_bar[m_]
                px = lg.closes[bi]
            settle_exit(lg, pos, px, reason, pos["direction"], bi, m_)
            lg.open = None

        def open_legs():
            return [lg for lg in L if lg.open is not None]

        def prop_end_attempt(m_: int, use_open: bool = False) -> None:
            """Closes the current attempt (bust or pass): flat the book, record it, and buy a fresh account."""
            nonlocal equity, attempt_id, prop_pending, prop_done, payout_target_baseline
            for lg in L:
                if lg.open is not None:
                    pos = lg.open
                    i_ = lg.bar_at[m_]
                    if i_ >= 0:
                        px = lg.opens[i_] if use_open else lg.closes[i_]
                        bi = i_
                    else:
                        bi = lg.last_bar[m_]
                        px = lg.closes[bi]
                    settle_exit(lg, pos, px, "account_blown_forced_close" if acct.failed else "attempt_end_forced_close",
                                pos["direction"], bi, m_)
                    lg.open = None
            failed = acct.failed
            end_t = at_time(m_)
            prop_attempts.append({
                "attempt_id": attempt_id, "start_time": prop_attempt_start_time, "end_time": end_t,
                "outcome": "failed" if failed else "passed", "failure_reason": acct.failure_reason,
                "passed_evaluation": acct.passed_evaluation, "days_to_pass": acct.days_to_pass,
                "end_balance": acct.balance, "n_trades": acct.n_trades, "total_payout": acct.total_payout_amount,
                "max_dd_pct": acct.max_dd_pct_reached, "first_payout_time": prop_first_payout_time,
            })
            reset_events.append({
                "kind": "breach" if failed else "attempt_pass", "reset_at": end_t, "attempt_id": attempt_id,
                "equity_before_reset": acct.balance, "equity_after_forced_close": acct.balance,
                "drawdown_pct": ((prop_rules.account_size - acct.balance) / prop_rules.account_size * 100.0) if failed else None,
                "message": (f"{end_t}: attempt {attempt_id} " + (
                    f"FAILED ({acct.failure_reason}) at balance ${acct.balance:,.2f}" if failed
                    else f"PASSED the evaluation at balance ${acct.balance:,.2f}") + "; fresh account started."),
            })
            attempt_id += 1
            equity = float(prop_rules.account_size)
            payout_target_baseline = equity
            adaptive_state.reset()
            for lg2 in L:
                lg2.leg_equity = equity
            prop_new_account(m_)

        def capacity_check(lg: _Leg, plan_pos: dict):
            """Portfolio-level capacity. Returns (allowed_fraction in [0,1], reason). 1.0 = fits as sized."""
            if (cfg.max_open_positions is None and cfg.max_total_contracts is None and prop_total_contract_cap is None
                    and cfg.max_open_risk_pct is None and not (cfg.max_margin_pct and cfg.margin_per_contract)):
                return 1.0, None
            open_now = open_legs()
            if cfg.max_open_positions is not None and len(open_now) >= cfg.max_open_positions:
                return 0.0, "max_open_positions"
            frac = 1.0
            reason = None
            size = plan_pos["size"]
            cs = plan_pos["contract_size"]
            new_contracts = (size / cs) if cs else size
            cap_contracts = [c for c in (cfg.max_total_contracts, prop_total_contract_cap) if c is not None]
            if cap_contracts:
                cap = min(cap_contracts)
                used = sum((p.open["size"] / p.open["contract_size"]) if p.open.get("contract_size") else p.open["size"]
                           for p in open_now)
                room = cap - used
                if new_contracts > room + 1e-9:
                    f = max(room, 0.0) / new_contracts if new_contracts else 0.0
                    if f < frac:
                        frac, reason = f, "max_total_contracts"
            if cfg.max_open_risk_pct is not None and plan_pos.get("risk_at_stop_dollars"):
                bal = equity if cfg.size_basis == "account" else lg.leg_equity
                cap_d = bal * cfg.max_open_risk_pct / 100.0
                used = sum((p.open.get("risk_at_stop_dollars") or 0.0) * (p.open["size"] / p.open["initial_size"])
                           for p in open_now)
                room = cap_d - used
                if plan_pos["risk_at_stop_dollars"] > room + 1e-9:
                    f = max(room, 0.0) / plan_pos["risk_at_stop_dollars"]
                    if f < frac:
                        frac, reason = f, "max_open_risk_pct"
            if cfg.max_margin_pct and cfg.margin_per_contract:
                mpc = cfg.margin_per_contract
                used = sum(mpc.get(p.name, 0.0) * ((p.open["size"] / p.open["contract_size"]) if p.open.get("contract_size") else p.open["size"])
                           for p in open_now)
                need = mpc.get(lg.name, 0.0) * new_contracts
                room = equity * cfg.max_margin_pct / 100.0 - used
                if need > room + 1e-9 and need > 0:
                    f = max(room, 0.0) / need
                    if f < frac:
                        frac, reason = f, "max_margin_pct"
            return frac, reason

        def clip_position(lg: _Leg, pos: dict, frac: float):
            """Shrink a planned position to ``frac`` of its size, to whole contracts where the leg trades them.
            Returns None when nothing is left."""
            cs = pos["contract_size"]
            if cs:
                contracts = math.floor(pos["size"] / cs * frac + 1e-9)
                if contracts <= 0:
                    return None
                new_size = contracts * cs
            else:
                new_size = pos["size"] * frac
                if new_size <= 0:
                    return None
            pos = dict(pos)
            pos["size"] = new_size
            pos["initial_size"] = new_size
            pos["contracts"] = (new_size / cs) if cs else None
            sd = abs(pos["entry_price"] - pos["stop_price"]) if pos["stop_price"] is not None else None
            if sd is not None:
                pos["risk_at_stop_dollars"] = lg.risk.worst_case_loss(new_size, sd, cs, pos["commission_per_contract"])
            else:
                pos["risk_at_stop_dollars"] = (pos["risk_at_stop_dollars"] or 0.0) * (new_size / pos["size"]) if pos["size"] else 0.0
            return pos

        # ======================================================================================================================
        if prop_mode:
            prop_new_account(0)

        deferred_checks = False         # a stop-out last step deferred this step's account checks (see docstring)
        broke_out = False
        for m in range(M):
            bar_date = day_idx[m]

            # ---- prop: new session day --------------------------------------------------------------------------------
            if prop_mode:
                if bar_date != prop_last_bar_date:
                    if prop_last_bar_date is not None and prop_last_bar_date in prop_day_map and m > 0:
                        acct.end_day(prop_day_map[prop_last_bar_date], float(equity_arr[m - 1]))
                    prop_last_bar_date = bar_date
                    if acct.check_new_day(unique_days[bar_date]) == FAILED:
                        prop_end_attempt(m, use_open=True)
                        # single-attempt handling is not offered at portfolio level: attempts chain to the last bar

            # adaptive-risk daily accumulators roll with the clock
            if bar_date != getattr(self, "_adaptive_day", -1):
                if any(lg.adaptive_risk is not None and lg.adaptive_risk.enabled for lg in L):
                    adaptive_state.begin_new_day()
                self._adaptive_day = bar_date

            if weekend_hold and dow[m] == 0:
                weekend_entry_block = False

            # ---- session flatten ---------------------------------------------------------------------------------------
            if flat_zone[m]:
                for lg in L:
                    if lg.open is not None:
                        close_leg_here(lg, m, "session_flatten_forced_close", "open")
                        flatten_close_count += 1

            step_continue = False       # a `continue` path of run_execution was taken somewhere this step
            skip_entry = [False] * K
            fav = [0.0] * K
            adv = [0.0] * K
            has_bar = [False] * K

            # ---- manage open positions: extremes -----------------------------------------------------------------------
            for lg in L:
                pos = lg.open
                if pos is None:
                    continue
                i_ = lg.bar_at[m]
                if i_ >= 0:
                    has_bar[lg.k] = True
                    d_ = pos["direction"]
                    fe = lg.highs[i_] if d_ == 1 else lg.lows[i_]
                    if d_ == 1:
                        pos["best_price"] = max(pos["best_price"], fe)
                        pos["mfe_price"] = max(pos["mfe_price"], fe)
                    else:
                        pos["best_price"] = min(pos["best_price"], fe)
                        pos["mfe_price"] = min(pos["mfe_price"], fe)
                    ae = lg.lows[i_] if d_ == 1 else lg.highs[i_]
                    if d_ == 1:
                        pos["worst_price"] = min(pos["worst_price"], ae)
                    else:
                        pos["worst_price"] = max(pos["worst_price"], ae)
                    fav[lg.k], adv[lg.k] = fe, ae
                else:
                    px = mark_price(lg, m)
                    fav[lg.k] = adv[lg.k] = px

            open_now = open_legs()
            liquidated = False

            # ---- PROP: floating drawdown / floating daily loss on the COMBINED book -------------------------------------
            if open_now and prop_mode and (acct.dd_basis in ("floating", "eod") or acct.daily_loss_basis == "floating"):
                hi_f = 0.0
                lo_f = 0.0
                adv_eff_by = {}
                for lg in open_now:
                    pos = lg.open
                    d_ = pos["direction"]
                    ent, sz, stop_now = pos["entry_price"], pos["size"], pos["stop_price"]
                    ae = adv[lg.k]
                    i_ = lg.bar_at[m]
                    if i_ >= 0 and stop_now is not None:
                        if d_ == 1 and lg.lows[i_] <= stop_now:
                            ae = max(adv[lg.k], min(stop_now, lg.opens[i_]))
                        elif d_ == -1 and lg.highs[i_] >= stop_now:
                            ae = min(adv[lg.k], max(stop_now, lg.opens[i_]))
                    adv_eff_by[lg.k] = ae
                    hi_f += (fav[lg.k] - ent) * sz * d_
                    lo_f += (ae - ent) * sz * d_
                eq_hi = equity + hi_f
                eq_lo = equity + lo_f
                dd_fl, dl_fl = acct.projected_floors(eq_hi)
                lvl = None
                kind = None
                if dd_fl is not None and eq_lo <= dd_fl:
                    lvl, kind = dd_fl, "prop_floor_liquidation"
                if dl_fl is not None and eq_lo <= dl_fl and (lvl is None or dl_fl > lvl):
                    lvl, kind = dl_fl, "daily_loss_limit_forced_close"
                pd_ = prop_day(bar_date)
                res = acct.on_equity_extremes(pd_, eq_hi, eq_lo)
                if lvl is not None:
                    s = None
                    if len(open_now) > 1 and lo_f < 0:
                        s = (lvl - equity) / lo_f
                    for lg in open_now:
                        pos = lg.open
                        d_ = pos["direction"]
                        ent, sz = pos["entry_price"], pos["size"]
                        if len(open_now) == 1 or s is None:
                            liq = ent + d_ * (lvl - equity) / sz if sz else adv[lg.k]
                        else:
                            liq = ent + d_ * ((adv_eff_by[lg.k] - ent) * d_) * s
                        i_ = lg.bar_at[m]
                        o_ = lg.opens[i_] if i_ >= 0 else adv[lg.k]
                        if d_ == 1:
                            liq = min(max(liq, adv[lg.k]), o_)
                        else:
                            liq = max(min(liq, adv[lg.k]), o_)
                        bi = i_ if i_ >= 0 else lg.last_bar[m]
                        settle_exit(lg, pos, liq, kind, d_, bi, m)
                        lg.open = None
                        force_closed_count += 1
                    if res == DAY_LOCKED:
                        prop_locked_bar_date = bar_date
                    equity_arr[m] = equity
                    liquidated = True
                    if prop_pending == "failed" or acct.failed:
                        prop_end_attempt(m)
                        attempt_arr[m] = attempt_id
                        equity_arr[m] = equity
                    attempt_arr[m] = attempt_id
                    if leg_pnl_arr is not None:
                        leg_pnl_arr[:, m] = leg_realised
                    continue

            # ---- legacy daily-loss circuit breaker on the combined book ----------------------------------------------------
            if open_now and daily_limit_amount is not None:
                floating_adverse = 0.0
                for lg in open_now:
                    pos = lg.open
                    floating_adverse += (adv[lg.k] - pos["entry_price"]) * pos["size"] * pos["direction"]
                day_realized_so_far = pnl_today_sum[bar_date]
                if (day_realized_so_far + floating_adverse) <= -daily_limit_amount:
                    s = None
                    if len(open_now) > 1 and floating_adverse < 0:
                        s = (-daily_limit_amount - day_realized_so_far) / floating_adverse
                    for lg in open_now:
                        pos = lg.open
                        d_ = pos["direction"]
                        liq_size = pos["size"]
                        if len(open_now) == 1 or s is None:
                            if liq_size and liq_size > 0:
                                liq_dist = (daily_limit_amount + day_realized_so_far) / liq_size
                                liq_price = pos["entry_price"] - d_ * liq_dist
                            else:
                                liq_price = adv[lg.k]
                        else:
                            liq_price = pos["entry_price"] + d_ * ((adv[lg.k] - pos["entry_price"]) * d_) * s
                        if d_ == 1:
                            liq_price = max(liq_price, adv[lg.k])
                        else:
                            liq_price = min(liq_price, adv[lg.k])
                        i_ = lg.bar_at[m]
                        bi = i_ if i_ >= 0 else lg.last_bar[m]
                        settle_exit(lg, pos, liq_price, "daily_loss_limit_forced_close", d_, bi, m)
                        lg.open = None
                        force_closed_count += 1
                    if getattr(account, "prop_daily_loss_is_breach", False):
                        account_blown = True
                        account_blown_at = at_time(m)
                    equity_arr[m] = equity
                    attempt_arr[m] = attempt_id
                    if leg_pnl_arr is not None:
                        leg_pnl_arr[:, m] = leg_realised
                    continue

            # ---- per-leg: stops, targets, tightening, partials, signal exits -----------------------------------------------
            for lg in L:
                pos = lg.open
                if pos is None:
                    continue
                i_ = lg.bar_at[m]
                if i_ < 0:
                    # market closed for this leg: nothing can trigger; weekend hold still flattens at the last close
                    if weekend_hold and week_ends_at[m]:
                        d_ = pos["direction"]
                        bi = lg.last_bar[m]
                        settle_exit(lg, pos, lg.closes[bi], "weekend_hold_forced_close", d_, bi, m)
                        lg.open = None
                        weekend_close_count += 1
                        weekend_entry_block = True
                    continue
                direction = pos["direction"]
                n_ = lg.n
                opens_, highs_, lows_, closes_ = lg.opens, lg.highs, lg.lows, lg.closes
                stop = pos["stop_price"]

                if lg.ib_on and pos["take_price"] is not None and stop is not None:
                    both_s, _ = _fills.resolve_intrabar_exit(direction, stop, None, lows_[i_], highs_[i_], opens_[i_])
                    both_t, _ = _fills.resolve_intrabar_exit(direction, None, pos["take_price"], lows_[i_], highs_[i_], opens_[i_])
                    if both_s is not None and both_t is not None:
                        first = intrabar_first_hit(lg, direction, stop, pos["take_price"], i_)
                        pos["fill_resolution"] = "intrabar"
                        if first == "take" and not ((direction == 1 and opens_[i_] <= stop) or (direction == -1 and opens_[i_] >= stop)):
                            settle_exit(lg, pos, pos["take_price"], "take_profit", direction, i_, m)
                            lg.open = None
                            step_continue = True
                            skip_entry[lg.k] = True
                            continue
                pre_exit, pre_reason = _fills.resolve_intrabar_exit(direction, stop, None, lows_[i_], highs_[i_], opens_[i_])
                if pre_exit is not None:
                    settle_exit(lg, pos, pre_exit, pre_reason, direction, i_, m)
                    lg.open = None
                    step_continue = True
                    skip_entry[lg.k] = True
                    continue

                # break-even
                if lg.breakeven_r is not None and pos["initial_risk"] and not pos["breakeven_done"]:
                    profit_dist = (pos["best_price"] - pos["entry_price"]) * direction
                    if profit_dist >= lg.breakeven_r * pos["initial_risk"]:
                        cand = pos["entry_price"]
                        if stop is None or (direction == 1 and cand > stop) or (direction == -1 and cand < stop):
                            stop = cand
                        pos["breakeven_done"] = True
                # trailing stop
                if pos.get("trailing_distance"):
                    cand = pos["best_price"] - direction * pos["trailing_distance"]
                    if stop is None or (direction == 1 and cand > stop) or (direction == -1 and cand < stop):
                        stop = cand
                # partial exit
                if lg.partial_cfg is not None and not pos["partial_taken"] and pos["initial_risk"]:
                    partial_target = pos["entry_price"] + direction * lg.partial_r * pos["initial_risk"]
                    partial_hit = (highs_[i_] >= partial_target) if direction == 1 else (lows_[i_] <= partial_target)
                    take_before_partial = (
                        pos["take_price"] is not None
                        and ((direction == 1 and pos["take_price"] <= partial_target)
                             or (direction == -1 and pos["take_price"] >= partial_target))
                    )
                    if partial_hit and not take_before_partial:
                        settle_partial(lg, pos, partial_target, direction, m)
                        pos["partial_taken"] = True
                        if lg.partial_be:
                            cand = pos["entry_price"]
                            if stop is None or (direction == 1 and cand > stop) or (direction == -1 and cand < stop):
                                stop = cand

                pos["stop_price"] = stop
                take = pos["take_price"]
                exit_price, reason = _fills.resolve_intrabar_exit(direction, stop, take, lows_[i_], highs_[i_], opens_[i_])

                # signal exit (flat / reversal): market order at the fill bar's open
                if exit_price is None and lg.sig[i_] != direction:
                    fill_idx = i_ + lg.fill_lag
                    if fill_idx < n_ and lg.step[fill_idx] >= 0:
                        exit_price, reason = opens_[fill_idx], "signal"
                        settle_exit(lg, pos, exit_price, reason, direction, fill_idx, int(lg.step[fill_idx]))
                        lg.last_close_bar_idx = i_
                        lg.open = None

                # time-invalidation stop
                if exit_price is None and lg.open is not None and lg.risk.time_stop_hours:
                    age_h = (lg.ts[i_] - lg.ts[pos["entry_bar_idx"]]) / np.timedelta64(1, "h")
                    atr_now = float(lg.atr[i_])
                    stagnation = abs(closes_[i_] - pos["entry_price"])
                    if (age_h > lg.risk.time_stop_hours and atr_now > 0
                            and stagnation <= lg.risk.time_stop_atr_band * atr_now):
                        exit_price, reason = closes_[i_], "time_stop"

                if exit_price is not None and lg.open is not None:
                    settle_exit(lg, pos, exit_price, reason, direction, i_, m)
                    lg.open = None

                # weekend hold at the week's last bar
                if lg.open is not None and weekend_hold and week_ends_at[m]:
                    settle_exit(lg, pos, closes_[i_], "weekend_hold_forced_close", direction, i_, m)
                    lg.open = None
                    weekend_close_count += 1
                    weekend_entry_block = True

            # ---- mark-to-market equity point --------------------------------------------------------------------------
            floating_total = 0.0
            n_open = 0
            gross_c = 0.0
            marg = 0.0
            for lg in L:
                pos = lg.open
                if pos is None:
                    continue
                i_ = lg.bar_at[m]
                px = lg.closes[i_] if i_ >= 0 else mark_price(lg, m)
                floating_total += (px - pos["entry_price"]) * pos["size"] * pos["direction"]
                n_open += 1
                c_ = (pos["size"] / pos["contract_size"]) if pos.get("contract_size") else pos["size"]
                gross_c += c_
                if margin_arr is not None:
                    marg += (cfg.margin_per_contract or {}).get(lg.name, 0.0) * c_
            mtm_equity = equity + floating_total if n_open else equity
            equity_arr[m] = mtm_equity
            attempt_arr[m] = attempt_id
            n_open_arr[m] = n_open
            gross_contracts_arr[m] = gross_c
            if margin_arr is not None:
                margin_arr[m] = marg
            if leg_pnl_arr is not None:
                leg_pnl_arr[:, m] = leg_realised
                if n_open:
                    for lg in L:
                        pos = lg.open
                        if pos is not None:
                            i_ = lg.bar_at[m]
                            px = lg.closes[i_] if i_ >= 0 else mark_price(lg, m)
                            leg_pnl_arr[lg.k, m] += (px - pos["entry_price"]) * pos["size"] * pos["direction"]

            account_checks_now = (not step_continue) or immediate
            if deferred_checks and not immediate:
                account_checks_now = not step_continue
            # run_execution skips the end-of-bar account block on a `continue` bar; the checks then happen next bar
            if not account_checks_now:
                deferred_checks = True
                # entries are skipped for legs that stopped out; other legs may still enter below
            else:
                deferred_checks = False
                # ---- prop: attempt end ------------------------------------------------------------------------------------
                if prop_mode:
                    if prop_pending == "failed" or acct.failed:
                        prop_end_attempt(m)
                        equity_arr[m] = equity
                        attempt_arr[m] = attempt_id
                # ---- legacy breach / reset -------------------------------------------------------------------------------
                if not prop_mode and not account_blown and (equity <= 0 or (blown_floor is not None and equity <= blown_floor)):
                    if not halt_on_breach:
                        breach_equity = equity
                        breach_at = at_time(m)
                        for lg in L:
                            if lg.open is not None:
                                close_leg_here(lg, m, "account_blown_forced_close", "close")
                        floor_txt = f"${blown_floor:,.2f}" if blown_floor is not None else "$0"
                        reset_events.append({
                            "kind": "breach", "reset_at": breach_at, "equity_before_reset": breach_equity,
                            "equity_after_forced_close": equity,
                            "drawdown_pct": ((account.initial_balance - breach_equity) / account.initial_balance * 100.0)
                            if account.initial_balance else None,
                            "message": (f"{breach_at}: ACCOUNT BREACH #{sum(1 for e in reset_events if e.get('kind') == 'breach') + 1} - "
                                        f"equity ${breach_equity:,.2f} fell to the loss floor ({floor_txt}). "
                                        f"Fresh ${account.initial_balance:,.2f} account started; trading continues."),
                        })
                        equity = account.initial_balance
                        equity_arr[m] = equity
                        payout_target_baseline = account.initial_balance
                        adaptive_state.reset()
                        for lg in L:
                            lg.leg_equity = equity
                    else:
                        account_blown = True
                        account_blown_at = at_time(m)
                # ---- profit-target payout --------------------------------------------------------------------------------
                if (not prop_mode and account.reset_on_target and account.profit_target_pct and account.profit_target_pct > 0
                        and not account_blown and equity >= payout_target_baseline * (1 + account.profit_target_pct / 100.0)):
                    payout_amount = equity - payout_target_baseline
                    reset_events.append({
                        "kind": "payout", "reset_at": at_time(m), "equity_before_payout": equity,
                        "payout_amount": payout_amount, "equity_after_payout": payout_target_baseline,
                        "message": (f"{at_time(m)}: PROFIT TARGET REACHED #{sum(1 for e in reset_events if e.get('kind') == 'payout') + 1} - "
                                    f"equity ${equity:,.2f} hit {account.profit_target_pct:g}% target; "
                                    f"${payout_amount:,.2f} banked as a payout. Account reset to "
                                    f"${payout_target_baseline:,.2f}; trading continues."),
                    })
                    equity = payout_target_baseline
                    equity_arr[m] = equity
                    adaptive_state.reset()

            # ---- entries ------------------------------------------------------------------------------------------------
            day_realized_pnl = pnl_today_sum[bar_date]
            daily_limit_breached = (
                daily_limit_amount is not None and day_realized_pnl <= -daily_limit_amount
            ) or (prop_mode and prop_locked_bar_date == bar_date)
            news_today = bool(news_blackout_bar[m])
            weekend_blocked = (weekend_hold and (dow[m] >= 5 or weekend_entry_block)) or bool(flat_zone[m])
            for lg in L:
                i_ = lg.bar_at[m]
                if i_ < 0:
                    continue
                if lg.open is not None or lg.sig[i_] == 0:
                    continue
                if skip_entry[lg.k]:
                    continue
                risk = lg.risk
                cooldown_active = (i_ - lg.last_close_bar_idx) < risk.reentry_cooldown_bars
                if (not daily_limit_breached and not account_blown and not cooldown_active and not news_today
                        and not weekend_blocked):
                    fill_idx = i_ + lg.fill_lag
                    if fill_idx >= lg.n or lg.step[fill_idx] < 0:
                        continue
                    m_fill = int(lg.step[fill_idx])
                    fill_bar_date = day_idx[m_fill]
                    n_today = lg.trades_today[fill_bar_date]
                    if n_today >= risk.max_trades_per_day:
                        lg.blocked_maxtrades += 1
                    total_cap = cfg.max_trades_per_day_total
                    if total_cap is not None and total_trades_today[fill_bar_date] >= total_cap:
                        total_trades_today_blocked += 1
                        lg.blocked_cap += 1
                        continue
                    if n_today < risk.max_trades_per_day:
                        direction = int(lg.sig[i_])
                        raw_price = lg.opens[fill_idx]
                        bar_sl = float(lg.sl_vals[i_]) if lg.sl_vals is not None and not math.isnan(lg.sl_vals[i_]) else None
                        bar_tp = float(lg.tp_vals[i_]) if lg.tp_vals is not None and not math.isnan(lg.tp_vals[i_]) else None
                        bar_tr = float(lg.trail_vals[i_]) if lg.trail_vals is not None and not math.isnan(lg.trail_vals[i_]) else None
                        vol_pct = None
                        if lg.adaptive_risk is not None and lg.adaptive_risk.enabled and lg.vol_pct is not None:
                            v = lg.vol_pct[i_]
                            vol_pct = float(v) if not math.isnan(v) else None
                        size_equity = equity if cfg.size_basis == "account" else lg.leg_equity
                        plan = _fills.plan_entry(
                            risk=risk, equity=size_equity, direction=direction, raw_price=raw_price,
                            bar_sl_distance=bar_sl, bar_tp_distance=bar_tp, bar_trail_distance=bar_tr,
                            stop_loss_pips=lg.stop_loss_pips, take_profit_pips=lg.take_profit_pips,
                            adaptive_risk=lg.adaptive_risk, adaptive_state=adaptive_state, vol_pct=vol_pct,
                            atr_now=lg.atr[i_],
                        )
                        lg.considered += 1
                        sizing_considered_total += 1
                        if plan.min_contract_rescue:
                            lg.rescue += 1
                        if plan.adaptive_probe:
                            lg.probe += 1
                        if plan.skipped:
                            if plan.adaptive_blocked:
                                lg.adaptive_zero += 1
                            if plan.skip_is_floor:
                                lg.floor_skips += 1
                                lg.skip_reasons[plan.skip_reason] = lg.skip_reasons.get(plan.skip_reason, 0) + 1
                            elif plan.skip_reason is not None:
                                lg.skip_reasons[plan.skip_reason] = lg.skip_reasons.get(plan.skip_reason, 0) + 1
                            continue
                        position = plan.position
                        # portfolio-level capacity
                        frac, why = capacity_check(lg, position)
                        if frac < 1.0 - 1e-12:
                            if cfg.cap_policy == "skip" or frac <= 0:
                                lg.blocked_cap += 1
                                lg.skip_reasons[why] = lg.skip_reasons.get(why, 0) + 1
                                continue
                            clipped = clip_position(lg, position, frac)
                            if clipped is None:
                                lg.blocked_cap += 1
                                lg.skip_reasons[why] = lg.skip_reasons.get(why, 0) + 1
                                continue
                            position = clipped
                            lg.clipped_cap += 1
                        if plan.used_fallback_stop:
                            lg.fallback_stops += 1
                        if plan.pip_mismatch_ratio is not None:
                            lg.pip_mismatch += 1
                            if lg.pip_worst is None or plan.pip_mismatch_ratio < lg.pip_worst:
                                lg.pip_worst = plan.pip_mismatch_ratio
                        if plan.atr_mismatch_ratio is not None:
                            lg.atr_mismatch += 1
                            if lg.atr_worst is None or plan.atr_mismatch_ratio < lg.atr_worst:
                                lg.atr_worst = plan.atr_mismatch_ratio
                        if plan.stop_capped:
                            lg.stop_capped += 1
                        if plan.used_micro:
                            lg.micro += 1
                        if plan.impact_contracts is not None:
                            lg.impact.append(plan.impact_contracts)
                        lg.open = {
                            "entry_time": restore_tz(lg.ts[fill_idx], lg.tz),
                            "entry_bar_idx": fill_idx,
                            **position,
                        }
                        lg.trades_today[fill_bar_date] = n_today + 1
                        total_trades_today[fill_bar_date] += 1
                        lg.last_entry_bar_idx = fill_idx
                else:
                    if account_blown:
                        lg.blocked_halt += 1
                    elif daily_limit_breached:
                        lg.blocked_daily += 1
                    elif cooldown_active:
                        lg.blocked_cooldown += 1
                    elif news_today:
                        lg.blocked_news += 1
                    elif weekend_blocked:
                        lg.blocked_weekend += 1

        # ---- end of data: close whatever is still open at each leg's final close ------------------------------------------
        for lg in L:
            if lg.open is not None:
                bi = int(lg.last_bar[M - 1])
                m_fill = int(lg.step[bi]) if lg.step[bi] >= 0 else M - 1
                settle_exit(lg, lg.open, lg.closes[bi], "end_of_data", lg.open["direction"], bi, m_fill)
                lg.open = None
        if prop_mode:
            prop_attempts.append({
                "attempt_id": attempt_id, "start_time": prop_attempt_start_time,
                "end_time": at_time(M - 1) if M else None,
                "outcome": "passed_funded" if acct.passed_evaluation else "open", "failure_reason": None,
                "passed_evaluation": acct.passed_evaluation, "days_to_pass": acct.days_to_pass,
                "end_balance": acct.balance, "n_trades": acct.n_trades, "total_payout": acct.total_payout_amount,
                "max_dd_pct": acct.max_dd_pct_reached, "first_payout_time": prop_first_payout_time,
            })

        # ---- result assembly ------------------------------------------------------------------------------------------------
        if len(L) == 1 and clock.window == "union":
            ts_col = preps[0].df["timestamp"].reset_index(drop=True)
        else:
            ts_col = pd.Series(master.astype("datetime64[ns]"))
            if acct_tz is not None:
                ts_col = ts_col.dt.tz_localize("UTC").dt.tz_convert(acct_tz)
        equity_df = pd.DataFrame({"timestamp": ts_col, "equity": equity_arr})
        if prop_mode:
            equity_df["attempt_id"] = attempt_arr
            equity_df.attrs["prop_attempts"] = prop_attempts
            equity_df.attrs["prop_rules"] = prop_rules
        floor_total = sum(lg.floor_skips for lg in L)
        considered_total = sum(lg.considered for lg in L)
        skip_reasons_total: dict = {}
        for lg in L:
            for kk, vv in lg.skip_reasons.items():
                skip_reasons_total[kk] = skip_reasons_total.get(kk, 0) + vv
        equity_df.attrs["account_reset_events"] = reset_events
        equity_df.attrs["payout_events"] = [ev for ev in reset_events if ev.get("kind") == "payout"]
        equity_df.attrs["breach_events"] = [ev for ev in reset_events if ev.get("kind", "breach") == "breach"]
        equity_df.attrs["zero_size_contract_floor_count"] = floor_total
        equity_df.attrs["min_contract_rescue_count"] = sum(lg.rescue for lg in L)
        equity_df.attrs["sizing_summary"] = {
            "mode": L[0].risk.sizing_mode, "entries_considered": considered_total,
            "entries_taken": len([t for t in trades if t.exit_reason != "partial_take_profit"]),
            "skipped_for_sizing": floor_total, "skip_rate": (floor_total / considered_total) if considered_total else 0.0,
            "skip_reasons": dict(skip_reasons_total), "stop_capped": sum(lg.stop_capped for lg in L),
            "micro_fallback": sum(lg.micro for lg in L),
        }
        equity_df.attrs["adaptive_probe_count"] = sum(lg.probe for lg in L)
        equity_df.attrs["entry_block_counts"] = {
            "engine_halted": sum(lg.blocked_halt for lg in L), "daily_loss_limit": sum(lg.blocked_daily for lg in L),
            "reentry_cooldown": sum(lg.blocked_cooldown for lg in L), "max_trades_per_day": sum(lg.blocked_maxtrades for lg in L),
            "zero_contract_sizing": floor_total, "adaptive_risk_zero_size": sum(lg.adaptive_zero for lg in L),
            "news_blackout": sum(lg.blocked_news for lg in L), "weekend_hold": sum(lg.blocked_weekend for lg in L),
            "portfolio_capacity": sum(lg.blocked_cap for lg in L),
        }
        equity_df.attrs["weekend_hold_close_count"] = weekend_close_count
        equity_df.attrs["session_flatten_close_count"] = flatten_close_count
        equity_df.attrs["account_reset_log"] = [ev["message"] for ev in reset_events if ev.get("message")]
        halt_total = len(trades) + floor_total
        halt_ratio = (floor_total / halt_total) if halt_total else 0.0
        equity_df.attrs["sizing_halt"] = {
            "halted": halt_ratio > 0.5, "skipped": floor_total, "skip_ratio": halt_ratio,
            "last_trade_exit": trades[-1].exit_time if trades else None, "risk_value": L[0].risk.risk_value,
            "contract_size": L[0].risk.contract_size, "sizing_mode": L[0].risk.sizing_mode,
            "skip_reasons": dict(skip_reasons_total),
        }
        equity_df.attrs["instruments"] = [lg.name for lg in L]

        # warnings (condensed twin of run_execution's end-of-run notices, aggregated across legs)
        for note in clock.notes:
            warn_msgs.append(note)
        for lg in L:
            for w in lg.warn:
                warn_msgs.append(f"[{lg.name}] {w}")
            if lg.fallback_stops:
                warn_msgs.append(f"[{lg.name}] {lg.fallback_stops} trade(s) had no stop loss at all; a "
                                 f"{_fills.DEFAULT_STOP_PCT_OF_PRICE * 100:.0f}%-of-price protective stop was used for sizing.")
            if lg.pip_mismatch:
                warn_msgs.append(f"[{lg.name}] {lg.pip_mismatch} trade(s) had a fixed-pips stop that is an implausible "
                                 f"fraction of this instrument's price (as small as {lg.pip_worst:.5%}); check pip_size ({lg.risk.pip_size}).")
            if lg.atr_mismatch and lg.atr_mismatch >= max(3, int(0.05 * max(lg.n_trades, 1))):
                warn_msgs.append(f"[{lg.name}] {lg.atr_mismatch} trade(s) had a fixed-pips stop under 15% of the instrument's "
                                 f"recent ATR (as little as {lg.atr_worst:.1%}); set pip_size to match the instrument or use an ATR stop.")
            if lg.gap_losses:
                warn_msgs.append(f"[{lg.name}] {lg.gap_losses} trade(s) hit their stop but filled more than 3x further away than "
                                 "the intended risk (a gap straight through the resting stop).")
            if lg.impact:
                warn_msgs.append(f"[{lg.name}] MARKET IMPACT: {len(lg.impact)} entr{'y' if len(lg.impact) == 1 else 'ies'} exceeded "
                                 f"the {_fills.MAX_CONTRACTS_BEFORE_IMPACT_WARN}-contract guardrail (largest {max(lg.impact):.0f}); "
                                 "filled at the quoted price with no market impact modelled.")
            if lg.floor_skips:
                warn_msgs.append(f"[{lg.name}] {lg.floor_skips} potential entr{'y' if lg.floor_skips == 1 else 'ies'} were skipped "
                                 "because sizing rounded down to 0 whole contracts at the configured risk.")
            if lg.blocked_cap or lg.clipped_cap:
                warn_msgs.append(f"[{lg.name}] portfolio capacity: {lg.blocked_cap} entr{'y' if lg.blocked_cap == 1 else 'ies'} refused, "
                                 f"{lg.clipped_cap} shrunk to fit.")
        breaches = equity_df.attrs["breach_events"]
        payouts = equity_df.attrs["payout_events"]
        if breaches:
            warn_msgs.append(f"{len(breaches)} account reset(s): the combined account crossed its loss floor and was restarted at "
                             f"initial balance each time. First: " + " | ".join(ev["message"] for ev in breaches[:3] if ev.get("message")))
        if payouts:
            warn_msgs.append(f"{len(payouts)} payout(s) taken (${sum(ev['payout_amount'] for ev in payouts):,.2f} total).")
        if account_blown:
            warn_msgs.append(f"Account BLOWN at {account_blown_at}: no new trades were opened for the rest of the run.")
        if force_closed_count:
            warn_msgs.append(f"{force_closed_count} position(s) were force-closed intrabar because FLOATING equity across the whole book "
                             "breached a daily-loss / drawdown floor.")
        if clamped_loss_count:
            warn_msgs.append(f"{clamped_loss_count} trade(s) had their simulated loss capped at RiskConfig.max_trade_loss.")
        if (account.commission_per_trade == 0.0 and all(lg.risk.commission_per_trade == 0.0 and lg.risk.commission_per_contract == 0.0
                                                          and lg.risk.slippage_pips == 0.0 and lg.risk.spread_pips == 0.0 for lg in L)):
            warn_msgs.append("ZERO FRICTION: commission, slippage AND spread are all 0.0 for every leg; fills are free.")

        # statistics
        stats = None
        if cfg.compute_stats:
            stats = compute_statistics(trades, equity_df, initial_balance=float(account.initial_balance))

        # per-leg summaries and portfolio stats
        leg_summaries = []
        for lg, cov in zip(L, clock.coverage):
            leg_summaries.append(LegSummary(
                name=lg.name, trades=lg.n_trades, net_profit=float(lg.net),
                win_rate=(lg.wins / lg.n_trades) if lg.n_trades else None, entries_considered=lg.considered,
                entries_skipped_for_sizing=lg.floor_skips, blocked_by_portfolio_caps=lg.blocked_cap,
                clipped_by_portfolio_caps=lg.clipped_cap, bars=lg.n, coverage=cov.coverage, warnings=list(lg.warn)))
        leg_curves = None
        if leg_pnl_arr is not None:
            leg_curves = {lg.name: pd.DataFrame({"timestamp": ts_col, "pnl": leg_pnl_arr[lg.k]}) for lg in L}
        pstats = _portfolio_stats(trades, equity_df, n_open_arr, gross_contracts_arr, margin_arr, L, leg_pnl_arr, float(account.initial_balance),
                                  skip_reasons_total, cfg)
        return PortfolioEngineResult(
            trades=trades, equity_curve=equity_df, statistics=stats, initial_balance=float(account.initial_balance),
            legs=leg_summaries, clock=clock, portfolio_stats=pstats, leg_curves=leg_curves, warnings=warn_msgs,
            lookahead_check_failed=any_lookahead_failed,
        )


# ======================================================================================================================
def _portfolio_stats(trades, equity_df, n_open, gross_contracts, margin_arr, legs, leg_pnl, initial_balance, skip_reasons, cfg) -> dict:
    """Quantities that only exist for a book: concurrency, exposure, drawdown on combined equity vs realised-only,
    and each leg's contribution."""
    eq = equity_df["equity"].to_numpy(dtype=float)
    out: dict = {}
    M = len(eq)
    out["steps"] = int(M)
    out["max_concurrent_positions"] = int(n_open.max()) if M else 0
    out["avg_open_positions"] = float(n_open.mean()) if M else 0.0
    out["pct_time_in_market"] = float((n_open > 0).mean() * 100.0) if M else 0.0
    out["pct_time_multiple_positions"] = float((n_open > 1).mean() * 100.0) if M else 0.0
    out["peak_gross_contracts"] = float(gross_contracts.max()) if M else 0.0
    if margin_arr is not None and M:
        out["peak_margin_used"] = float(margin_arr.max())
        out["peak_margin_pct_of_initial"] = float(margin_arr.max() / initial_balance * 100.0) if initial_balance else None
    if M:
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        out["max_drawdown_dollars"] = float(dd.max())
        out["max_drawdown_pct_of_peak"] = float((dd / np.where(peak > 0, peak, np.nan)).max() * 100.0)
    # per-leg contribution
    contrib = {}
    total = sum(float(lg.net) for lg in legs)
    for lg in legs:
        contrib[lg.name] = {
            "net_profit": float(lg.net), "trades": int(lg.n_trades),
            "share_of_net_profit": (float(lg.net) / total) if total else None,
            "win_rate": (lg.wins / lg.n_trades) if lg.n_trades else None,
        }
    out["contribution"] = contrib
    out["entries_refused_or_shrunk_by_capacity"] = {
        "refused": int(sum(lg.blocked_cap for lg in legs)), "shrunk": int(sum(lg.clipped_cap for lg in legs)),
        "reasons": {k: v for k, v in skip_reasons.items() if k in ("max_open_positions", "max_total_contracts", "max_open_risk_pct", "max_margin_pct")},
    }
    # leg correlation of daily P&L (from the leg curves, if recorded)
    if leg_pnl is not None and len(legs) > 1 and M > 2:
        try:
            ts = pd.to_datetime(equity_df["timestamp"])
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            frame = pd.DataFrame({lg.name: leg_pnl[lg.k] for lg in legs})
            frame.index = ts
            daily = frame.resample("1D").last().ffill().diff().dropna(how="all")
            if len(daily) >= 3:
                c = daily.corr().fillna(0.0)
                out["daily_pnl_correlation"] = {a: {b: float(c.loc[a, b]) for b in c.columns} for a in c.index}
        except Exception:  # noqa: BLE001 -- a diagnostic, never worth failing a run over
            pass
    return out
