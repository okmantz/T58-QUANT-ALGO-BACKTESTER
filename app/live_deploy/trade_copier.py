"""
Trade copier: run ONE strategy and mirror every trade it takes onto several
broker accounts (for example 3 Lucid accounts and 2 FTMO accounts).

    leader account  -- the account LiveExecutionSession trades. It is the single
                       source of truth: followers only ever copy trades the
                       leader actually placed, and close when the leader closes.
    follower legs   -- every other account. Each has its own broker connection,
                       its own sizing (your risk % on THAT account's equity, a
                       multiple of the leader's dollar exposure, or a fixed
                       number of contracts/lots), its own symbol, and its own
                       prop guardrails (max drawdown, daily loss, news blackout,
                       weekend flatten, no hedging, lot cap).

Failure isolation (the part that matters on real accounts):
  * A follower that loses its connection 3 polls in a row, or breaches ITS OWN
    max-drawdown floor, flattens itself and halts. The rest of the group keeps
    trading.
  * If the leader emergency-halts (drawdown breach, lost connection), every
    follower is flattened too (``halt_group_on_leader_halt``, default True):
    follower positions are mirrors of the leader's, and a mirror with no leader
    is an unmanaged position.
  * Starting is all-or-nothing: if any selected account fails to connect, nothing
    trades and everything that did connect is disconnected again.
  * A follower position is never opened late (no chasing) and never touched when
    the copier did not open it.

Concurrency: orders and closes fan out to all legs in parallel (so the accounts
fill within a moment of each other) and join before the leader's loop continues.
Each leg has its own broker adapter; the shared SQLite journal is guarded by one
lock.

This module has no Flask or Tk imports. ``validate_group`` is pure.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import time as dt_time
from typing import Any, Callable, Optional

import pandas as pd

from app.backtest.risk import RiskConfig
from app.forward_test.journal import ForwardTestJournal
from app.live_deploy.broker_base import BrokerAdapter
from app.live_deploy.execution_engine import parse_blackout_windows
from app.live_deploy.guardrails import (
    contract_size_for_symbol, futures_prop_safe_rules, resolve_units_per_lot,
)
from app.live_deploy.live_settings import LiveAccount, normalize_platform
from app.prop.simulator import PropRules

SIZE_RISK = "risk"
SIZE_MULTIPLIER = "multiplier"
SIZE_FIXED = "fixed"
SIZE_MODES = (SIZE_RISK, SIZE_MULTIPLIER, SIZE_FIXED)

MAX_FOLLOWERS = 12
_MAX_CONSECUTIVE_FAILURES = 3
_MAX_CLOSE_FAILURES_BEFORE_ALERT = 3
_FANOUT_TIMEOUT_SECONDS = 90.0
_MT5 = "MT4/MT5"
# Platforms whose login can own several accounts and whose adapter needs an explicit pick.
_MULTI_ACCOUNT_PLATFORMS = ("Tradovate", "TradeLocker", "DXtrade")

LogFn = Callable[[str, str], None]


# --------------------------------------------------------------------------- specs

@dataclass
class FollowerSpec:
    """What the person chose for one follower account."""
    account_id: str
    symbol: str = ""            # "" = same symbol as the leader
    size_mode: str = SIZE_RISK
    # risk: risk per trade in the SAME unit as the leader's risk mode (% of equity or fixed $); 0 = use the leader's value.
    # multiplier: x the leader's dollar exposure (ES->MES maps correctly). fixed: contracts/lots in this platform's units.
    size_value: float = 0.0
    account_size: float = 0.0   # 0 = same as the leader's account size
    max_dd_pct: float = 0.0     # 0 = same as the leader's
    daily_loss_pct: float = 0.0  # 0 = same as the leader's
    contract_size: Optional[float] = None  # None = looked up from the symbol (ES 50, MES 5, ...)

    @classmethod
    def from_dict(cls, d: dict) -> "FollowerSpec":
        def num(key: str, default: float = 0.0) -> float:
            raw = d.get(key)
            if raw in (None, ""):
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"'{key}' must be a number (got {raw!r}).") from None

        mode = str(d.get("size_mode") or SIZE_RISK).strip().lower()
        if mode not in SIZE_MODES:
            raise ValueError(f"size_mode must be one of {', '.join(SIZE_MODES)} (got {mode!r}).")
        cs_raw = d.get("contract_size")
        return cls(
            account_id=str(d.get("account_id") or "").strip(),
            symbol=str(d.get("symbol") or "").strip(),
            size_mode=mode, size_value=num("size_value"), account_size=num("account_size"),
            max_dd_pct=num("max_dd_pct"), daily_loss_pct=num("daily_loss_pct"),
            contract_size=(float(cs_raw) if cs_raw not in (None, "") else None),
        )


def parse_follower_specs(raw: Any) -> list[FollowerSpec]:
    """JSON list (already decoded) -> specs. Raises ValueError with a plain message."""
    if raw in (None, "", []):
        return []
    if not isinstance(raw, list):
        raise ValueError("Followers must be a list.")
    return [FollowerSpec.from_dict(item if isinstance(item, dict) else {}) for item in raw]


# --------------------------------------------------------------------------- pure sizing

def units_per_broker_qty(contract_size: Optional[float], units_per_lot: Optional[float]) -> float:
    """Sizing units in ONE broker-native quantity (a contract on Tradovate, a lot elsewhere)."""
    return float(units_per_lot or contract_size or 0.0)


def compute_follower_units(
    *, mode: str, value: float, risk: RiskConfig, equity: float, stop_distance: float,
    leader_units: float, contract_size: Optional[float], units_per_lot: Optional[float],
) -> float:
    """Sizing units for a follower, before the broker's own flooring.

    risk        -- this account's own equity and risk setting against the strategy's stop.
    multiplier  -- value x the leader's EXECUTED dollar exposure (units), so a leader trading
                   1 ES (50 units) mirrored x1 onto an MES account buys 10 MES.
    fixed       -- value contracts/lots in the follower platform's native quantity.
    """
    if mode == SIZE_RISK:
        pip = risk.pip_size or 1.0
        return float(risk.position_size(equity, stop_distance / pip))
    if mode == SIZE_MULTIPLIER:
        return max(float(leader_units) * float(value), 0.0)
    if mode == SIZE_FIXED:
        return max(float(value), 0.0) * units_per_broker_qty(contract_size, units_per_lot)
    raise ValueError(f"Unknown size mode {mode!r}")


# --------------------------------------------------------------------------- preflight

def validate_group(
    leader: Optional[LiveAccount], specs: list[FollowerSpec], accounts_by_id: dict[str, LiveAccount],
    *, leader_symbol: str, adapter_class_for: Callable[[str], Optional[type]],
) -> dict:
    """Pre-start checks for a copier group. Pure: no connections, no I/O.

    ``adapter_class_for(platform)`` returns the BrokerAdapter subclass for a
    platform (its class-level capability flags are read, nothing is built).
    Returns {"items": [...], "can_start": bool, "needs_ack": bool,
    "feed_account_id": str|None}. Item levels: ok | warn | ack | block.
    """
    items: list[dict] = []

    def add(key: str, level: str, title: str, detail: str = "") -> None:
        items.append({"key": key, "level": level, "title": title, "detail": detail})

    if leader is None:
        add("leader", "block", "No leader account selected", "Pick the account that will trade the strategy (the leader).")
        return {"items": items, "can_start": False, "needs_ack": False, "feed_account_id": None}
    if not specs:
        add("followers", "block", "No follower accounts selected", "Tick at least one account to copy trades onto.")
        return {"items": items, "can_start": False, "needs_ack": False, "feed_account_id": None}
    if len(specs) > MAX_FOLLOWERS:
        add("followers", "block", f"Too many followers ({len(specs)})", f"The limit is {MAX_FOLLOWERS} follower accounts per group.")

    members: list[tuple[str, LiveAccount, Optional[FollowerSpec]]] = [("leader", leader, None)]
    seen_ids = {leader.id}
    for spec in specs:
        acct = accounts_by_id.get(spec.account_id)
        if acct is None:
            add("follower_missing", "block", "A selected follower account no longer exists", f"Account id {spec.account_id or '(blank)'} is not saved.")
            continue
        if spec.account_id in seen_ids:
            add("follower_dup", "block", f"{acct.nickname or acct.login} is selected twice",
                "An account can be the leader or a follower, once.")
            continue
        seen_ids.add(spec.account_id)
        members.append(("follower", acct, spec))

        if spec.size_mode in (SIZE_MULTIPLIER, SIZE_FIXED) and spec.size_value <= 0:
            add("size", "block", f"{acct.nickname or acct.login}: size value is required",
                "Multiplier and fixed-size modes need a number greater than zero.")
        if spec.size_value < 0 or spec.account_size < 0 or spec.max_dd_pct < 0 or spec.daily_loss_pct < 0:
            add("size", "block", f"{acct.nickname or acct.login}: negative numbers are not allowed", "")

    # -- every member must be a usable, connectable account ----------------------------------
    for role, acct, _spec in members:
        label = f"{acct.nickname or acct.login or 'Account'} ({'leader' if role == 'leader' else 'follower'})"
        plat = normalize_platform(acct.platform)
        if adapter_class_for(plat) is None:
            add("platform", "block", f"{label}: {acct.platform} cannot be connected", "No connector exists for this platform.")
        elif not acct.is_usable:
            add("usable", "block", f"{label}: account details incomplete", "Open the account and fill in the missing credentials.")

    # -- the same underlying trading account listed twice -----------------------------------
    keys: dict[tuple, str] = {}
    for _role, acct, _spec in members:
        plat = normalize_platform(acct.platform)
        extra = acct.extra_credentials or {}
        ident = (plat, (acct.login or "").strip().lower(), (acct.server or "").strip().lower(),
                 str(extra.get("account_id") or extra.get("ctid_trader_account_id") or "").strip().lower(),
                 str(extra.get("base_url") or "").strip().lower())
        name = acct.nickname or acct.login
        if ident in keys and keys[ident] != name:
            add("same_account", "block", f"'{keys[ident]}' and '{name}' are the same trading account",
                "Two saved entries point at one real account; copying onto it would double the position.")
        keys.setdefault(ident, name)

    # -- several accounts under one login need an explicit account id -------------------------
    by_login: dict[tuple, list[LiveAccount]] = {}
    for _role, acct, _spec in members:
        plat = normalize_platform(acct.platform)
        if plat in _MULTI_ACCOUNT_PLATFORMS:
            by_login.setdefault((plat, (acct.login or "").strip().lower()), []).append(acct)
    for (plat, _login), accts in by_login.items():
        if len(accts) > 1:
            missing = [a.nickname or a.login for a in accts if not str((a.extra_credentials or {}).get("account_id") or "").strip()]
            if missing:
                add("account_pick", "block", f"{plat}: several entries share one login without an account ID",
                    f"Without an 'Account ID or name' each of these would trade the login's FIRST account: {', '.join(missing)}. "
                    "Open each in 'Connect / manage accounts' and fill in the Account ID or name.")

    # -- MT5's Python package is one global connection per process ----------------------------
    mt5_members = [a for _r, a, _s in members if normalize_platform(a.platform) == _MT5]
    if len(mt5_members) > 1:
        add("mt5_single", "block", "Only one MT4/MT5 account can be in a copier group",
            "The MetaTrader5 Python package holds a single terminal login per process, so two MT5 accounts cannot trade together here: "
            + ", ".join(a.nickname or a.login for a in mt5_members)
            + ". Use cTrader or DXtrade (or TradeLocker) for the other accounts where your firm offers them.")

    # -- who supplies price bars -------------------------------------------------------------
    feed_id: Optional[str] = None
    for _role, acct, _spec in members:  # leader first, then followers in order
        cls = adapter_class_for(normalize_platform(acct.platform))
        if cls is not None and getattr(cls, "supports_bars", True):
            feed_id = acct.id
            break
    if feed_id is None:
        add("feed", "block", "No account in this group can supply price bars",
            "Tradovate has no bar history in this app. Add an account on a platform that does (MT5, cTrader, TradeLocker or DXtrade) "
            "to the group; it will supply the price feed even if it trades a different instrument.")
    elif feed_id != leader.id:
        feed_acct = accounts_by_id.get(feed_id)
        add("feed", "warn", f"Price bars will come from {feed_acct.nickname or feed_acct.login if feed_acct else 'a follower'}, not the leader",
            "The leader's platform has no bar feed here, so the strategy reads another account's prices. Make sure that feed's instrument "
            "(its futures contract or CFD) matches what the strategy was tested on; CFD and futures prices, volume and session hours differ.")
    else:
        add("feed", "ok", "Leader supplies the price bars", "")

    # -- capability warnings -------------------------------------------------------------------
    no_stop = [a.nickname or a.login for _r, a, _s in members
               if not getattr(adapter_class_for(normalize_platform(a.platform)) or object, "supports_broker_stops", True)]
    if no_stop:
        add("stops", "warn", "No broker-side stop on: " + ", ".join(no_stop),
            "The Tradovate connector sends a plain market order. Those accounts are protected only by this app: the strategy's own exit, "
            "the max-drawdown halt, and the kill switch. If the app stops running, nothing closes the position. Set a stop in the platform "
            "yourself or keep size small.")

    # -- follower sizing resolvable ------------------------------------------------------------
    for _role, acct, spec in members[1:]:
        sym = (spec.symbol or leader_symbol).strip()
        plat = normalize_platform(acct.platform)
        cs = spec.contract_size or contract_size_for_symbol(sym)
        if plat == "Tradovate" and not cs:
            add("contract_size", "block", f"{acct.nickname or acct.login}: contract size unknown for {sym}",
                "Tradovate orders are whole contracts, so the app needs the contract size (ES 50, MES 5, NQ 20, MNQ 2). "
                "Fill in the contract size for this follower.")

    # -- copy-trading permission (never assume it) -----------------------------------------------
    firms = sorted({(a.firm_name or "").strip() for _r, a, _s in members if (a.firm_name or "").strip()})
    add("permission", "ack", "Confirm each firm allows copying trades across these accounts",
        "Firms differ on copy trading between accounts (some forbid it, some limit it to accounts you own at the same firm, some "
        "treat identical trades across accounts as a violation). Check the rules for: " + (", ".join(firms) or "each firm in the group")
        + ". A violation can get funded accounts terminated whatever the performance.")

    can_start = not any(i["level"] == "block" for i in items)
    return {"items": items, "can_start": can_start, "needs_ack": True, "feed_account_id": feed_id}


# --------------------------------------------------------------------------- follower leg

@dataclass
class LegStatus:
    account_id: str = ""
    nickname: str = ""
    platform: str = ""
    symbol: str = ""
    size_mode: str = SIZE_RISK
    size_value: float = 0.0
    running: bool = False
    connected: bool = False
    balance: Optional[float] = None
    equity: Optional[float] = None
    n_trades_closed: int = 0
    net_pnl: float = 0.0
    open_position_ticket: Optional[str] = None
    last_qty: Optional[float] = None
    halted_reason: Optional[str] = None
    alert: Optional[str] = None
    blocked_reason: Optional[str] = None
    last_error: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class FollowerLeg:
    """One follower account: its broker, sizing, guardrails, journal session and halt state."""

    def __init__(
        self, *, account: LiveAccount, spec: FollowerSpec, broker: BrokerAdapter, symbol: str, timeframe_minutes: int,
        risk: RiskConfig, prop_rules: PropRules, contract_size: Optional[float], units_per_lot: Optional[float],
        journal: ForwardTestJournal, journal_lock: threading.Lock, strategy_type: str, strategy_filename: str,
        log: LogFn, weekend_flatten_time: dt_time = dt_time(20, 45),
    ):
        self.account = account
        self.spec = spec
        self.broker = broker
        self.symbol = symbol
        self.timeframe_minutes = timeframe_minutes
        self.risk = risk
        self.rules = prop_rules
        self.contract_size = contract_size
        self.units_per_lot = units_per_lot
        self.journal = journal
        self._jlock = journal_lock
        self.strategy_type = strategy_type
        self.strategy_filename = strategy_filename
        self._log_out = log
        self.weekend_flatten_time = weekend_flatten_time
        self._blackout = parse_blackout_windows(getattr(prop_rules, "news_blackout_windows", "") or "")
        self.status = LegStatus(
            account_id=account.id or "", nickname=account.nickname or account.login or "Account",
            platform=broker.platform_name, symbol=symbol, size_mode=spec.size_mode, size_value=spec.size_value,
        )
        self._lock = threading.RLock()        # one worker at a time per leg
        self._session_id: Optional[int] = None
        self._trade_row_id: Optional[int] = None
        self._open_direction = 0
        self._failures = 0
        self._close_failures = 0
        self._dd_peak: Optional[float] = None
        self._daily_key: Optional[str] = None
        self._daily_pnl = 0.0

    # ---- logging -----------------------------------------------------------------
    def log(self, level: str, message: str) -> None:
        text = f"[{self.status.nickname}] {message}"
        if self._session_id is not None:
            try:
                with self._jlock:
                    self.journal.log_event(self._session_id, level, message)
            except Exception:  # noqa: BLE001
                pass
        self._log_out(level, text)

    # ---- lifecycle ------------------------------------------------------------------
    def connect(self) -> tuple[bool, str]:
        with self._lock:
            conn = self.broker.connect()
            if not conn.ok:
                self.status.last_error = conn.message
                return False, f"{self.status.nickname}: {conn.message}"
            self.status.connected = True
            self.status.balance, self.status.equity = conn.balance, conn.equity
            self._dd_peak = conn.equity or conn.balance or self.rules.account_size
            # Journal under the saved-account id so the portfolio overview can tell several accounts on one login apart.
            with self._jlock:
                self._session_id = self.journal.start_session(
                    self.strategy_type, self.strategy_filename, self.symbol, self.timeframe_minutes,
                    self.account.id or conn.account_login or "", conn.account_server or self.broker.platform_name,
                )
            return True, ""

    def adopt_or_block(self, leader_direction: int) -> None:
        """At start: an existing position on this account is adopted ONLY when the leader holds the same
        direction; otherwise the account is parked (never touched, never added to) until it is flat."""
        with self._lock:
            try:
                existing = self.broker.get_open_positions(self.symbol)
            except Exception as exc:  # noqa: BLE001
                self.log("warn", f"Could not check for existing positions: {exc}")
                return
            if not existing:
                return
            p = existing[0]
            if leader_direction and p.direction == leader_direction:
                self.status.open_position_ticket = p.ticket
                self._open_direction = p.direction
                self.log("info", f"Adopted the existing {self.symbol} position (ticket {p.ticket}) that matches the leader.")
            else:
                self.status.blocked_reason = (f"This account already holds a {self.symbol} position the copier did not open; "
                                              "it is left alone and no copied trades are placed here until it is flat.")
                self.log("warn", self.status.blocked_reason)

    def disconnect(self) -> None:
        with self._lock:
            if self._session_id is not None:
                try:
                    with self._jlock:
                        self.journal.end_session(self._session_id)
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.broker.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self.status.connected = False
            self.status.running = False

    # ---- guard checks -----------------------------------------------------------------
    def _roll_day(self, ts: pd.Timestamp) -> None:
        key = ts.strftime("%Y-%m-%d")
        if key != self._daily_key:
            self._daily_key = key
            self._daily_pnl = 0.0

    def _entry_blocked(self, signal: int, ts: pd.Timestamp) -> Optional[str]:
        s = self.status
        if s.alert or s.halted_reason:
            return f"halted ({s.alert or s.halted_reason})"
        if s.blocked_reason:
            return "parked (existing position not opened by the copier)"
        if s.open_position_ticket is not None:
            return "already in a copied trade"
        self._roll_day(ts)
        if self.risk.daily_loss_limit_pct is not None:
            limit = self.rules.account_size * (self.risk.daily_loss_limit_pct / 100.0)
            if self._daily_pnl <= -abs(limit):
                return f"daily loss limit reached ({self.risk.daily_loss_limit_pct}%)"
        if not getattr(self.rules, "weekend_hold_allowed", True) and ts.weekday() == 4 and ts.time() >= self.weekend_flatten_time:
            return "weekend-hold restriction"
        if any(w.contains(ts) for w in self._blackout):
            return "news blackout"
        if not getattr(self.rules, "hedging_allowed", True):
            try:
                if any(p.direction != signal for p in self.broker.get_open_positions()):
                    return "an opposite-direction position is open on this account (hedging not allowed)"
            except Exception:  # noqa: BLE001
                pass
        return None

    def _reference_price(self, leader_price: float, leader_symbol: str) -> Optional[float]:
        """Price to anchor this account's stop/target on. Own latest bar when the platform serves bars, else the
        leader's price when the instrument is the same. None means 'no safe reference' (entry refused)."""
        if not self.broker.supports_broker_stops:
            return leader_price                      # the adapter sends no stop; nothing is anchored
        if self.broker.supports_bars:
            try:
                df = self.broker.fetch_completed_bars(self.symbol, self.timeframe_minutes, 2)
                if df is not None and not df.empty:
                    px = float(df["close"].iloc[-1])
                    if px > 0:
                        return px
            except Exception:  # noqa: BLE001
                pass
        if self.symbol.strip().upper() == (leader_symbol or "").strip().upper():
            return leader_price
        return None

    # ---- mirror actions ------------------------------------------------------------------
    def open(self, *, signal: int, stop_distance: float, target_distance: Optional[float], leader_units: float,
             leader_price: float, leader_symbol: str, ts: pd.Timestamp) -> None:
        with self._lock:
            why = self._entry_blocked(signal, ts)
            if why:
                self.log("warn", f"Skipped copying the {'LONG' if signal == 1 else 'SHORT'} entry: {why}.")
                return
            equity = self.status.equity or self.status.balance or self.rules.account_size
            units = compute_follower_units(
                mode=self.spec.size_mode, value=self.spec.size_value, risk=self.risk, equity=float(equity),
                stop_distance=stop_distance, leader_units=leader_units,
                contract_size=self.contract_size, units_per_lot=self.units_per_lot,
            )
            qty = self.broker.to_broker_qty(units, contract_size=self.contract_size, units_per_lot=self.units_per_lot)
            if qty <= 0:
                self.log("warn", f"Skipped copying the entry: size ({units:.2f} sizing units, mode {self.spec.size_mode}) converts to "
                                 f"zero {self.broker.platform_name} quantity (below one whole contract/lot, or the contract size is missing).")
                return
            max_lot = getattr(self.rules, "max_lot_size", None)
            if max_lot and qty > max_lot:
                self.log("warn", f"Quantity capped by max_lot_size: {qty:.2f} -> {float(max_lot):.2f}.")
                qty = float(max_lot)
            ref = self._reference_price(leader_price, leader_symbol)
            if ref is None:
                self.log("error", f"Skipped copying the entry: no reference price for {self.symbol} on {self.broker.platform_name}, "
                                  "so a protective stop could not be placed safely.")
                return
            sl = ref - signal * stop_distance if self.broker.supports_broker_stops else None
            tp = (ref + signal * target_distance) if (target_distance and self.broker.supports_broker_stops) else None
            order = self.broker.place_market_order(self.symbol, signal, qty, sl_price=sl, tp_price=tp,
                                                   comment=f"T58 copy {self.strategy_filename}"[:31])
            if not order.ok:
                self.status.last_error = order.message
                self.log("error", f"Copied order failed: {order.message}")
                return
            fill = order.price or ref
            executed_units = qty * units_per_broker_qty(self.contract_size, self.units_per_lot)
            with self._jlock:
                self._trade_row_id = self.journal.record_open(self._session_id, order.ticket, signal, executed_units, fill, sl, tp)
            self.status.open_position_ticket = order.ticket
            self.status.last_qty = qty
            self._open_direction = signal
            self.status.last_error = None
            self.log("info", f"Copied {'LONG' if signal == 1 else 'SHORT'} {qty:g} {self.broker.platform_name} units of {self.symbol} "
                             f"@ {fill:.5f} (ticket {order.ticket}).")

    def close(self, reason: str) -> bool:
        """Close this leg's copied position. True when flat afterwards."""
        with self._lock:
            ticket = self.status.open_position_ticket
            if ticket is None:
                return True
            res = self.broker.close_position(ticket)
            if not res.ok:
                self._close_failures += 1
                self.status.last_error = res.message
                self.log("error", f"Close failed (ticket {ticket}): {res.message}")
                if self._close_failures >= _MAX_CLOSE_FAILURES_BEFORE_ALERT:
                    self.status.alert = (f"CLOSE FAILED {self._close_failures} times for ticket {ticket}: {res.message}. "
                                         "Verify and close this position on the broker.")
                return False
            self._close_failures = 0
            self._finalize(res.price)
            self.log("info", f"Closed copied position (ticket {ticket}) -- {reason}.")
            return True

    def _finalize(self, exit_price: Optional[float]) -> None:
        pnl = 0.0
        if self._trade_row_id is not None:
            with self._jlock:
                row = next((t for t in self.journal.open_trades(self._session_id) if t.id == self._trade_row_id), None)
                if row is not None:
                    exit_p = exit_price if exit_price is not None else row.entry_price
                    pnl = (exit_p - row.entry_price) * row.direction * row.volume
                    self.journal.record_close(row.id, exit_p, pnl)
        self._daily_pnl += pnl
        self.status.open_position_ticket = None
        self._trade_row_id = None
        self._open_direction = 0
        self._refresh_stats()

    def _refresh_stats(self) -> None:
        try:
            with self._jlock:
                stats = self.journal.closed_trade_stats(self._session_id)
            self.status.n_trades_closed = stats["n_trades"]
            self.status.net_pnl = stats["net_pnl"]
        except Exception:  # noqa: BLE001
            pass

    def flatten(self, reason: str) -> None:
        """Best-effort close of EVERYTHING on this account for the symbol (kill switch / group halt)."""
        with self._lock:
            try:
                for r in self.broker.close_all(self.symbol):
                    self.log("info" if r.ok else "error", f"Flatten ({reason}): {r.message}")
            except Exception as exc:  # noqa: BLE001
                self.log("error", f"Flatten failed ({reason}): {exc}")
            if self.status.open_position_ticket is not None and self._trade_row_id is not None:
                self._finalize(None)

    def emergency_halt(self, reason: str) -> None:
        self.log("error", reason)
        self.flatten("emergency halt")
        self.status.alert = reason
        self.status.halted_reason = reason
        self.status.open_position_ticket = None
        self.status.running = False

    # ---- per-poll health -----------------------------------------------------------------
    def _drawdown_floor(self) -> Optional[float]:
        pct = getattr(self.rules, "max_drawdown_pct", 0) or 0
        if pct <= 0:
            return None
        base = getattr(self.rules, "account_size", 0) or self.risk.initial_balance
        amount = base * pct / 100.0
        if getattr(self.rules, "drawdown_type", "trailing") == "static":
            return base - amount
        return (self._dd_peak if self._dd_peak is not None else base) - amount

    def poll(self) -> None:
        """Connection, equity, drawdown and broker-side-close reconciliation. Never raises."""
        with self._lock:
            if self.status.alert or self.status.halted_reason:
                return
            try:
                re = self.broker.ensure_connected()
                if not re.ok:
                    self._failures += 1
                    self.status.connected = False
                    self.log("error", f"Connection lost, reconnect failed: {re.message} "
                                      f"(consecutive failures: {self._failures}/{_MAX_CONSECUTIVE_FAILURES})")
                    if self._failures >= _MAX_CONSECUTIVE_FAILURES:
                        self.emergency_halt(f"CONNECTION LOST: {_MAX_CONSECUTIVE_FAILURES} consecutive poll failures. "
                                            "This account was flattened and halted; the rest of the group keeps trading. "
                                            "Verify its positions on the broker before restarting.")
                    return
                self._failures = 0
                self.status.connected = True
                summary = self.broker.account_summary()
                if summary:
                    self.status.balance = summary.get("balance")
                    self.status.equity = summary.get("equity")
                eq = self.status.equity if self.status.equity is not None else self.status.balance
                if eq is not None:
                    if self._dd_peak is None or eq > self._dd_peak:
                        self._dd_peak = eq
                    floor = self._drawdown_floor()
                    if floor is not None and eq <= floor:
                        self.emergency_halt(
                            f"MAX DRAWDOWN BREACHED: equity ${eq:,.2f} is at/below the "
                            f"{getattr(self.rules, 'drawdown_type', 'trailing')} floor ${floor:,.2f} "
                            f"({getattr(self.rules, 'max_drawdown_pct', 0)}% rule). This account was flattened and halted; "
                            "the rest of the group keeps trading.")
                        return
                ticket = self.status.open_position_ticket
                if ticket is not None and not any(p.ticket == ticket for p in self.broker.get_open_positions(self.symbol)):
                    self._finalize(None)
                    self.log("info", f"Position (ticket {ticket}) closed on the broker side (stop/target) since the last check.")
            except Exception as exc:  # noqa: BLE001
                self.status.last_error = str(exc)
                self.log("error", f"Poll error: {exc}")


# --------------------------------------------------------------------------- the copier

class TradeCopier:
    """Owns the follower legs and the fan-out. LiveExecutionSession calls the ``mirror_*`` / ``poll`` hooks."""

    def __init__(self, legs: list[FollowerLeg], *, on_log: Optional[LogFn] = None,
                 halt_group_on_leader_halt: bool = True, connect_stagger_seconds: float = 0.6):
        self.legs = legs
        self._on_log = on_log or (lambda level, msg: None)
        self.halt_group_on_leader_halt = halt_group_on_leader_halt
        self._stagger = connect_stagger_seconds
        self.leader_open = False

    # ---- helpers ---------------------------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        self._on_log(level, message)

    def _fanout(self, fn: Callable[[FollowerLeg], Any], legs: Optional[list[FollowerLeg]] = None) -> None:
        targets = [leg for leg in (legs if legs is not None else self.legs)]
        if not targets:
            return

        def run(leg: FollowerLeg) -> None:
            try:
                fn(leg)
            except Exception as exc:  # noqa: BLE001 -- one leg's failure must never stop the others
                leg.status.last_error = str(exc)
                leg.log("error", f"Unexpected error: {exc}\n{traceback.format_exc(limit=3)}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(targets), 8), thread_name_prefix="t58-copy") as pool:
            futures = [pool.submit(run, leg) for leg in targets]
            done, not_done = concurrent.futures.wait(futures, timeout=_FANOUT_TIMEOUT_SECONDS)
            if not_done:
                self._log("error", f"{len(not_done)} follower account(s) did not answer within {_FANOUT_TIMEOUT_SECONDS:.0f}s; "
                                   "check them on the broker.")

    # ---- lifecycle -----------------------------------------------------------------------
    def start(self, leader_direction: int = 0) -> tuple[bool, str]:
        """Connect every follower. All-or-nothing: one failure disconnects the rest and refuses to start."""
        connected: list[FollowerLeg] = []
        for i, leg in enumerate(self.legs):
            if i and self._stagger > 0:
                time.sleep(self._stagger)   # eases per-login auth rate limits (e.g. several Tradovate accounts)
            ok, msg = leg.connect()
            if not ok:
                for c in connected:
                    c.disconnect()
                return False, f"Follower account failed to connect, so nothing was started: {msg}"
            connected.append(leg)
        for leg in self.legs:
            leg.adopt_or_block(leader_direction)
            leg.status.running = True
        self.leader_open = bool(leader_direction)
        self._log("info", f"Trade copier armed: {len(self.legs)} follower account(s) connected "
                          f"({', '.join(l.status.nickname for l in self.legs)}).")
        return True, ""

    def stop(self) -> None:
        """Graceful: positions stay open (same as stopping the leader); connections close."""
        self._fanout(lambda leg: leg.disconnect())

    def flatten_all(self, reason: str) -> None:
        self._fanout(lambda leg: leg.flatten(reason))

    # ---- hooks called by LiveExecutionSession ---------------------------------------------------
    def poll(self) -> None:
        """Every poll cycle: follower health first, then retry any close that is still owed."""
        self._fanout(lambda leg: leg.poll())
        if not self.leader_open:
            owed = [leg for leg in self.legs if leg.status.open_position_ticket is not None]
            if owed:
                self._fanout(lambda leg: leg.close("leader is flat (retry)"), owed)

    def mirror_open(self, *, signal: int, stop_distance: float, target_distance: Optional[float], leader_units: float,
                    leader_price: float, leader_symbol: str, ts: pd.Timestamp) -> None:
        self.leader_open = True
        self._fanout(lambda leg: leg.open(signal=signal, stop_distance=stop_distance, target_distance=target_distance,
                                          leader_units=leader_units, leader_price=leader_price,
                                          leader_symbol=leader_symbol, ts=ts))

    def mirror_close(self, reason: str) -> None:
        self.leader_open = False
        owed = [leg for leg in self.legs if leg.status.open_position_ticket is not None]
        self._fanout(lambda leg: leg.close(reason), owed)

    def on_leader_emergency(self, reason: str) -> None:
        self.leader_open = False
        if self.halt_group_on_leader_halt:
            self._log("error", f"Leader halted; flattening and halting every follower. Reason: {reason}")
            self._fanout(lambda leg: leg.emergency_halt(f"LEADER HALTED: {reason}"))
        else:
            self._log("warn", "Leader halted; followers were left as they are (halt_group_on_leader_halt is off).")

    def statuses(self) -> list[dict]:
        return [leg.status.to_dict() for leg in self.legs]


# --------------------------------------------------------------------------- builder

def build_copier(
    *, specs: list[FollowerSpec], accounts_by_id: dict[str, LiveAccount], adapter_factory: Callable[[LiveAccount], BrokerAdapter],
    leader_symbol: str, timeframe_minutes: int, leader_risk: RiskConfig, leader_rules: PropRules,
    journal: ForwardTestJournal, strategy_type: str, strategy_filename: str, on_log: Optional[LogFn] = None,
    halt_group_on_leader_halt: bool = True, connect_stagger_seconds: float = 0.6,
) -> TradeCopier:
    """Turn follower specs into legs. Per-follower numbers left at 0 inherit the leader's."""
    jlock = threading.Lock()
    log = on_log or (lambda level, msg: None)
    legs: list[FollowerLeg] = []
    for spec in specs:
        account = accounts_by_id[spec.account_id]
        symbol = spec.symbol or leader_symbol
        contract_size = spec.contract_size or contract_size_for_symbol(symbol) or leader_risk.contract_size
        units_per_lot = resolve_units_per_lot(contract_size)
        acct_size = spec.account_size or leader_rules.account_size
        max_dd = spec.max_dd_pct or leader_rules.max_drawdown_pct
        daily = spec.daily_loss_pct or leader_rules.daily_loss_limit_pct
        rules = futures_prop_safe_rules(
            account_size=acct_size, news_blackout_windows=leader_rules.news_blackout_windows,
            weekend_hold_allowed=leader_rules.weekend_hold_allowed, hedging_allowed=leader_rules.hedging_allowed,
            max_lot_size=leader_rules.max_lot_size, daily_loss_limit_pct=daily, max_drawdown_pct=max_dd,
        )
        rules.drawdown_type = leader_rules.drawdown_type
        risk = dataclasses.replace(
            leader_risk, initial_balance=acct_size, contract_size=contract_size, daily_loss_limit_pct=daily,
            risk_value=(spec.size_value if (spec.size_mode == SIZE_RISK and spec.size_value > 0) else leader_risk.risk_value),
        )
        legs.append(FollowerLeg(
            account=account, spec=spec, broker=adapter_factory(account), symbol=symbol, timeframe_minutes=timeframe_minutes,
            risk=risk, prop_rules=rules, contract_size=contract_size, units_per_lot=units_per_lot, journal=journal,
            journal_lock=jlock, strategy_type=strategy_type, strategy_filename=strategy_filename, log=log,
        ))
    return TradeCopier(legs, on_log=log, halt_group_on_leader_halt=halt_group_on_leader_halt,
                       connect_stagger_seconds=connect_stagger_seconds)
