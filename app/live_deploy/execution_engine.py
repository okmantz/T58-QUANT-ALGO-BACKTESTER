"""
LiveExecutionSession -- the real backend behind Deploy Live.

Before this file existed, Deploy Live (app/web/templates/deploy_live.html
+ its server.py route) was informational only: it explained which prop
firms were connectable and let you save account credentials
(live_settings.py), but nothing in the app actually read a strategy's
signals and placed a live order. This is the missing piece.

Deliberately modeled on app.forward_test.engine.ForwardTestSession (same
poll -> signal -> size -> order loop, same journal, same drift check) but
generalized in two ways:
  1. Works against ANY BrokerAdapter (app.live_deploy.broker_base), not
     only MT5Connector -- so it's the same class whether the account is
     MT4/MT5, cTrader, Tradovate, TradeLocker, or DXtrade.
  2. Enforces PropRuleGuard checks (news-blackout windows, weekend-hold
     restriction, max lot size, hedging) before every new entry, which
     ForwardTestSession never needed against a demo account but which
     matter a great deal against a real funded evaluation.

Reuses app.forward_test.journal.ForwardTestJournal as-is for trade/event
storage -- the schema (session/trades/events) is already generic enough
for a live account, and giving live sessions their own duplicate journal
implementation for no functional gain wasn't worth the maintenance cost.
The `mt5_ticket`/`mt5_login` column names are a pre-existing naming
artifact, not a live-session bug: any platform's ticket/login string is
stored there just fine (SQLite has no real column typing).

THE SAME CAUTION THAT APPLIES TO EVERY NEW BROKER ADAPTER IN THIS PACKAGE
APPLIES DOUBLY HERE: this is the code path that puts real orders on a
real funded account. Run it against a demo/practice account on whichever
platform you're using for at least a few full poll cycles before trusting
it with a funded evaluation, and start with the smallest position size
your risk config allows.
"""
from __future__ import annotations

import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Callable, Optional

import pandas as pd

from app.backtest.execution import DEFAULT_STOP_PCT_OF_PRICE
from app.backtest.risk import RiskConfig
from app.forward_test.journal import ForwardTestJournal
from app.live_deploy.broker_base import BrokerAdapter
from app.prop.simulator import PropRules
from app.strategy.base import Strategy

LogCallback = Callable[[str, str], None]

_WEEKDAY_NAMES = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
_WINDOW_RE = re.compile(
    r"^\s*(?:(MON|TUE|WED|THU|FRI|SAT|SUN)\s+)?(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$", re.IGNORECASE,
)


@dataclass
class BlackoutWindow:
    weekday: Optional[int]   # None = applies every day
    start: dt_time
    end: dt_time

    def contains(self, ts: pd.Timestamp) -> bool:
        if self.weekday is not None and ts.weekday() != self.weekday:
            return False
        t = ts.time()
        if self.start <= self.end:
            return self.start <= t <= self.end
        return t >= self.start or t <= self.end  # window crosses midnight


def parse_blackout_windows(raw: str) -> list[BlackoutWindow]:
    """Parses the free-text "news blackout window" field: one window per
    line, either `HH:MM-HH:MM` (applies every day, account/server time)
    or `FRI 19:55-21:05` (a specific weekday). Silently skips lines that
    don't match rather than raising -- this feeds a live trading loop, and
    a typo in one line should degrade to "that one line is ignored" not
    "the whole session refuses to start"; the Deploy Live UI should
    surface unparsed lines back to the user separately, before saving.
    """
    windows: list[BlackoutWindow] = []
    for line in (raw or "").splitlines():
        m = _WINDOW_RE.match(line)
        if not m:
            continue
        day_str, sh, sm, eh, em = m.groups()
        weekday = _WEEKDAY_NAMES.get(day_str.upper()) if day_str else None
        windows.append(BlackoutWindow(
            weekday=weekday, start=dt_time(int(sh), int(sm)), end=dt_time(int(eh), int(em)),
        ))
    return windows


@dataclass
class LiveExecutionStatus:
    running: bool = False
    connected: bool = False
    platform: str = ""
    last_bar_time: Optional[pd.Timestamp] = None
    last_signal: int = 0
    open_position_ticket: Optional[str] = None
    balance: Optional[float] = None
    equity: Optional[float] = None
    n_trades_closed: int = 0
    win_rate: Optional[float] = None
    net_pnl: float = 0.0
    halted_reason: Optional[str] = None
    baseline_win_rate: Optional[float] = None
    drift_flag: Optional[str] = None


@dataclass
class LiveExecutionConfig:
    symbol: str
    timeframe_minutes: int
    risk: RiskConfig
    prop_rules: PropRules            # supplies max_drawdown/daily-loss context AND the new
                                      # news_blackout_windows / weekend_hold_allowed / hedging_allowed fields
    poll_seconds: int = 20
    history_bars: int = 1500
    min_drift_sample: int = 20
    drift_tolerance_pts: float = 20.0
    baseline_win_rate: Optional[float] = None
    weekend_flatten_time: dt_time = dt_time(20, 45)  # Friday server-time cutoff when weekend_hold_allowed=False


class LiveExecutionSession:
    def __init__(
        self,
        strategy: Strategy,
        strategy_type: str,
        strategy_filename: str,
        broker: BrokerAdapter,
        journal: ForwardTestJournal,
        config: LiveExecutionConfig,
        on_log: Optional[LogCallback] = None,
        on_status: Optional[Callable[[LiveExecutionStatus], None]] = None,
    ):
        self.strategy = strategy
        self.strategy_type = strategy_type
        self.strategy_filename = strategy_filename
        self.broker = broker
        self.journal = journal
        self.cfg = config
        self._blackout_windows = parse_blackout_windows(
            getattr(config.prop_rules, "news_blackout_windows", "") or ""
        )
        self._on_log = on_log or (lambda level, msg: None)
        self._on_status = on_status or (lambda status: None)

        self.status = LiveExecutionStatus(platform=broker.platform_name)
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._session_id: Optional[int] = None
        self._open_trade_row_id: Optional[int] = None
        self._daily_realized_pnl = 0.0
        self._daily_key: Optional[str] = None
        self._link_down = False
        self._weekend_flattened_this_week = False

    # -- public controls ------------------------------------------------

    def start(self) -> tuple[bool, str]:
        if self.status.running:
            return False, "Already running."
        conn = self.broker.connect()
        if not conn.ok:
            self._log("error", conn.message)
            return False, conn.message
        self.status.connected = True
        self.status.balance = conn.balance
        self.status.equity = conn.equity

        self._session_id = self.journal.start_session(
            self.strategy_type, self.strategy_filename, self.cfg.symbol,
            self.cfg.timeframe_minutes, conn.account_login or "", conn.account_server or self.broker.platform_name,
        )
        self.status.baseline_win_rate = self.cfg.baseline_win_rate
        self._reconcile_existing_position()

        self._stop_flag.clear()
        self.status.running = True
        self.status.halted_reason = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._log("info", f"LIVE session started: {self.strategy_filename} on {self.cfg.symbol} "
                           f"({self.cfg.timeframe_minutes}m), {self.broker.platform_name} account "
                           f"{conn.account_login}@{conn.account_server}. Real capital is at risk.")
        return True, "Started."

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.poll_seconds + 10)
        self.status.running = False
        if self._session_id is not None:
            self.journal.end_session(self._session_id)
        self.broker.disconnect()
        self.status.connected = False
        self._log("info", "Live session stopped.")

    def flatten_all_and_stop(self) -> None:
        self._log("warn", "Kill switch pressed -- flattening all open positions.")
        try:
            for r in self.broker.close_all(self.cfg.symbol):
                self._log("info" if r.ok else "error", r.message)
        except Exception as exc:  # noqa: BLE001
            self._log("error", f"Error while flattening positions: {exc}")
        self.stop()

    # -- internals --------------------------------------------------------

    def _log(self, level: str, message: str) -> None:
        if self._session_id is not None:
            try:
                self.journal.log_event(self._session_id, level, message)
            except Exception:
                pass
        self._on_log(level, message)

    def _reconcile_existing_position(self) -> None:
        try:
            positions = self.broker.get_open_positions(self.cfg.symbol)
        except Exception as exc:  # noqa: BLE001
            self._log("warn", f"Could not check for existing open positions: {exc}")
            return
        if not positions:
            return
        p = positions[0]
        self.status.open_position_ticket = p.ticket
        self.status.last_signal = p.direction
        self._log("info", f"Found an existing open position on {p.symbol} (ticket {p.ticket}) "
                           "-- adopting it instead of opening a duplicate.")

    def _today_key(self, ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%d")

    def _reset_daily_counter_if_needed(self, ts: pd.Timestamp) -> None:
        key = self._today_key(ts)
        if key != self._daily_key:
            self._daily_key = key
            self._daily_realized_pnl = 0.0
            self.status.halted_reason = None
            if ts.weekday() != 4:  # not Friday -- reset the once-per-week flatten flag
                self._weekend_flattened_this_week = False

    def _daily_loss_breached(self) -> bool:
        if self.cfg.risk.daily_loss_limit_pct is None:
            return False
        limit_amount = self.cfg.risk.initial_balance * (self.cfg.risk.daily_loss_limit_pct / 100.0)
        return self._daily_realized_pnl <= -abs(limit_amount)

    def _in_news_blackout(self, ts: pd.Timestamp) -> bool:
        return any(w.contains(ts) for w in self._blackout_windows)

    def _hedging_conflict(self, signal: int) -> bool:
        """True if opening `signal` would create a hedge (a position
        opposite an already-open one) while prop_rules.hedging_allowed is
        False. Checked account-wide, not just this symbol, since most
        prop-firm "no hedging" rules are written at the account level."""
        if getattr(self.cfg.prop_rules, "hedging_allowed", True):
            return False
        try:
            existing = self.broker.get_open_positions()
        except Exception:
            return False
        return any(p.direction != signal for p in existing)

    def _weekend_guard_active(self, ts: pd.Timestamp) -> bool:
        """When weekend_hold_allowed is False: blocks new entries and
        flattens everything once, at/after weekend_flatten_time on
        Friday (server/broker time, matching whatever timezone
        fetch_completed_bars' timestamps use)."""
        if getattr(self.cfg.prop_rules, "weekend_hold_allowed", True):
            return False
        if ts.weekday() != 4:  # only Friday matters -- Sat/Sun have no bars on most FX/CFD feeds anyway
            return False
        return ts.time() >= self.cfg.weekend_flatten_time

    def _resolve_stop_distance(self, result, price: float) -> float:
        if result.stop_loss_distance is not None:
            val = float(result.stop_loss_distance.iloc[-1])
            if val > 0:
                return val
        if result.stop_loss_pips:
            return float(result.stop_loss_pips) * self.cfg.risk.pip_size
        return abs(price) * DEFAULT_STOP_PCT_OF_PRICE

    def _resolve_target_distance(self, result, price: float) -> Optional[float]:
        if result.take_profit_distance is not None:
            val = float(result.take_profit_distance.iloc[-1])
            if val > 0:
                return val
        if result.take_profit_pips:
            return float(result.take_profit_pips) * self.cfg.risk.pip_size
        return None

    def _check_drift(self) -> None:
        stats = self.journal.closed_trade_stats(self._session_id)
        n = stats["n_trades"]
        self.status.n_trades_closed = n
        self.status.win_rate = stats["win_rate"]
        self.status.net_pnl = stats["net_pnl"]
        if self.cfg.baseline_win_rate is None or n < self.cfg.min_drift_sample or stats["win_rate"] is None:
            return
        deviation = stats["win_rate"] - self.cfg.baseline_win_rate
        if abs(deviation) >= self.cfg.drift_tolerance_pts:
            direction = "below" if deviation < 0 else "above"
            msg = (f"Live win rate ({stats['win_rate']:.1f}%, n={n}) is {abs(deviation):.1f} "
                   f"points {direction} the backtest's ({self.cfg.baseline_win_rate:.1f}%) -- "
                   "worth a closer look before trusting this strategy further.")
            self.status.drift_flag = msg
            self._log("warn", msg)

    def _run_loop(self) -> None:
        try:
            while not self._stop_flag.is_set():
                try:
                    self._poll_once()
                except Exception as exc:  # noqa: BLE001
                    self._log("error", f"Poll error: {exc}\n{traceback.format_exc(limit=3)}")
                self._on_status(self.status)
                self._stop_flag.wait(self.cfg.poll_seconds)
        finally:
            self.status.running = False
            self._on_status(self.status)

    def _poll_once(self) -> None:
        was_down = self._link_down
        reconnect = self.broker.ensure_connected()
        if not reconnect.ok:
            self._link_down = True
            self._log("error", f"{self.broker.platform_name} connection lost, reconnect failed: {reconnect.message}")
            return
        self._link_down = False
        if was_down:
            self._log("info", f"{self.broker.platform_name} connection recovered -- resuming polling.")
        summary = self.broker.account_summary()
        if summary:
            self.status.balance = summary.get("balance")
            self.status.equity = summary.get("equity")

        df = self.broker.fetch_completed_bars(self.cfg.symbol, self.cfg.timeframe_minutes, self.cfg.history_bars)
        if df.empty:
            return
        latest_bar_time = df["timestamp"].iloc[-1]
        self._reset_daily_counter_if_needed(latest_bar_time)

        if self.status.last_bar_time is not None and latest_bar_time <= self.status.last_bar_time:
            return
        self.status.last_bar_time = latest_bar_time

        self._reconcile_closed_trade()

        # -- weekend-hold restriction: flatten once, then block new entries --
        if self._weekend_guard_active(latest_bar_time):
            if not self._weekend_flattened_this_week and self.status.open_position_ticket is not None:
                self._log("warn", "Weekend-hold restriction: flattening before the weekend close.")
                self._close_current_trade("weekend flatten")
            self._weekend_flattened_this_week = True
            reason = "Weekend-hold restriction active -- no new entries until next week."
            self.status.halted_reason = reason
            self._check_drift()
            return

        if self._daily_loss_breached():
            reason = f"Daily loss limit reached ({self.cfg.risk.daily_loss_limit_pct}% of balance) -- no new entries today."
            if self.status.halted_reason != reason:
                self._log("warn", reason)
            self.status.halted_reason = reason
            self._check_drift()
            return

        if self._in_news_blackout(latest_bar_time):
            reason = "News-blackout window active -- no new entries."
            if self.status.halted_reason != reason:
                self._log("info", reason)
            self.status.halted_reason = reason
            self._check_drift()
            return
        if self.status.halted_reason and "blackout" in self.status.halted_reason:
            self.status.halted_reason = None  # window ended

        result = self.strategy.generate(df)
        signal = int(result.signals.iloc[-1])
        price = float(df["close"].iloc[-1])
        self.status.last_signal = signal

        if self.status.open_position_ticket is None and signal != 0:
            if self._hedging_conflict(signal):
                self._log("warn", f"Hedging not permitted for this account -- skipping {('LONG' if signal == 1 else 'SHORT')} entry "
                                   "while an opposite-direction position is open elsewhere on the account.")
            else:
                self._open_new_trade(signal, result, price)
        elif self.status.open_position_ticket is not None and signal == 0:
            self._close_current_trade("signal flat")
        elif self.status.open_position_ticket is not None and signal != 0 and signal != self._last_trade_direction():
            self._close_current_trade("signal reversed")
            if not self._hedging_conflict(signal):
                self._open_new_trade(signal, result, price)

        self._check_drift()

    def _last_trade_direction(self) -> int:
        open_trades = self.journal.open_trades(self._session_id)
        return open_trades[0].direction if open_trades else 0

    def _open_new_trade(self, signal: int, result, price: float) -> None:
        stop_distance = self._resolve_stop_distance(result, price)
        target_distance = self._resolve_target_distance(result, price)
        stop_pips = stop_distance / self.cfg.risk.pip_size if self.cfg.risk.pip_size else 0
        equity = self.status.equity or self.cfg.risk.initial_balance
        volume = self.cfg.risk.position_size(equity, stop_pips)

        max_lot = getattr(self.cfg.prop_rules, "max_lot_size", None)
        if max_lot:
            volume = min(volume, max_lot)

        if volume <= 0:
            self._log("warn", "Computed position size was zero -- skipping entry.")
            return

        sl_price = price - signal * stop_distance
        tp_price = price + signal * target_distance if target_distance else None

        order = self.broker.place_market_order(self.cfg.symbol, signal, volume, sl_price=sl_price, tp_price=tp_price)
        if not order.ok:
            self._log("error", f"Order failed: {order.message}")
            return

        self.status.open_position_ticket = order.ticket
        self._open_trade_row_id = self.journal.record_open(
            self._session_id, order.ticket, signal, order.volume or volume,
            order.price or price, sl_price, tp_price,
        )
        self._log("info", f"Opened {'LONG' if signal == 1 else 'SHORT'} {order.volume or volume:.2f} "
                           f"{self.cfg.symbol} @ {order.price or price:.5f} (ticket {order.ticket}).")

    def _close_current_trade(self, reason: str) -> None:
        ticket = self.status.open_position_ticket
        if ticket is None:
            return
        result = self.broker.close_position(ticket)
        if not result.ok:
            self._log("error", f"Close failed for ticket {ticket}: {result.message}")
            return
        self._finalize_closed_trade(result.price)
        self._log("info", f"Closed position (ticket {ticket}) -- {reason}.")

    def _reconcile_closed_trade(self) -> None:
        ticket = self.status.open_position_ticket
        if ticket is None:
            return
        still_open = any(p.ticket == ticket for p in self.broker.get_open_positions(self.cfg.symbol))
        if still_open:
            return
        self._finalize_closed_trade(exit_price=None)
        self._log("info", f"Position (ticket {ticket}) closed on the broker side (SL/TP hit) since last check.")

    def _finalize_closed_trade(self, exit_price: Optional[float]) -> None:
        if self._open_trade_row_id is None:
            self.status.open_position_ticket = None
            return
        open_trades = self.journal.open_trades(self._session_id)
        row = next((t for t in open_trades if t.id == self._open_trade_row_id), None)
        pnl = 0.0
        if row is not None:
            exit_p = exit_price if exit_price is not None else row.entry_price
            pnl = (exit_p - row.entry_price) * row.direction * row.volume
            self.journal.record_close(row.id, exit_p, pnl)
        self._daily_realized_pnl += pnl
        self.status.open_position_ticket = None
        self._open_trade_row_id = None
