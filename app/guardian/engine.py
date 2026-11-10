"""
Evaluation Guardian engine.

Given a firm's rules and the account's CURRENT state (from the broker, not from a backtest), it reports where every rule stands and
the one action the account should take now:

    Daily loss limit       Maximum drawdown        Trailing drawdown (the floor moves with the equity high)
    Profit target          Consistency rule        Minimum trading days
    Maximum contracts      Position exposure       News restrictions        Trading hours / session flatten

Escalation ladder, driven by the share of a limit that is used (thresholds come from the Conservative / Balanced / Accelerated
preset in ``app.prop.payout_optimizer``)::

    ok  ->  warn  ->  reduce (smaller size)  ->  block (no new entries)  ->  flatten (close everything, halt the strategy for the day)

``evaluate`` is pure. ``GuardianTracker`` keeps the few things a live session has to remember between polls (start-of-day balance,
equity high-water mark, trading days, best day) and is what ``LiveExecutionSession`` calls. All dollar figures here are the firm's
rules applied to the account's own numbers; the values on the Guardian page's demo are hypothetical and do not describe any firm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from app.guardian import news as gnews
from app.prop import payout_optimizer as po

LEVELS = ("ok", "warn", "reduce", "block", "flatten")
_RANK = {k: i for i, k in enumerate(LEVELS)}


def _worse(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


# ------------------------------------------------------------------------------------------------ rules and state
@dataclass
class GuardianRules:
    account_size: float
    daily_loss_limit: float | None = None          # dollars
    max_drawdown: float | None = None              # dollars
    drawdown_type: str = "trailing"                # trailing | static
    trailing_lock_floor: float | None = None       # a trailing floor stops rising at this equity level (None = never locks)
    profit_target: float | None = None             # dollars above the starting balance (evaluation only)
    consistency_pct: float | None = None           # best day must be <= this % of total profit
    min_trading_days: int = 0
    max_contracts: int | None = None               # the firm's hard cap
    max_exposure: float | None = None              # your own cap on open contracts across symbols (never above max_contracts)
    news_restricted: bool = False
    news_before_min: int = 5
    news_after_min: int = 5
    trading_start: str | None = None               # "HH:MM" America/Chicago; None = no window
    trading_end: str | None = None
    flatten_time: str | None = None                # "HH:MM" America/Chicago: must be flat by then
    stage: str = "evaluation"                      # evaluation | funded
    firm: str = ""
    preset_key: str = ""
    source: str = "custom"

    @classmethod
    def from_prop_rules(cls, rules, *, stage: str = "evaluation", firm: str = "", preset_key: str = "",
                        news_restricted: bool = False, news_minutes: int = 5) -> "GuardianRules":
        a = float(rules.account_size)
        dd = float(rules.max_drawdown_pct or 0) / 100.0 * a
        lock = None
        if getattr(rules, "trailing_lock", False):
            lock = a + float(getattr(rules, "trailing_lock_offset_pct", 0.0) or 0.0) / 100.0 * a
        flat = getattr(rules, "flatten_time_ct", None)
        return cls(
            account_size=a,
            daily_loss_limit=(float(rules.daily_loss_limit_pct) / 100.0 * a) if (rules.daily_loss_limit_pct or 0) > 0 else None,
            max_drawdown=dd if dd > 0 else None, drawdown_type=str(getattr(rules, "drawdown_type", "trailing")),
            trailing_lock_floor=lock,
            profit_target=(float(rules.evaluation_profit_target_pct) / 100.0 * a) if stage == "evaluation" and (rules.evaluation_profit_target_pct or 0) > 0 else None,
            consistency_pct=(getattr(rules, "consistency_rule_pct", None) if stage == "evaluation"
                             else getattr(rules, "funded_consistency_rule_pct", None)),
            min_trading_days=int(getattr(rules, "min_trading_days", 0) or 0) if stage == "evaluation" else 0,
            max_contracts=int(rules.max_contracts) if getattr(rules, "max_contracts", None) else None,
            news_restricted=bool(news_restricted), news_before_min=news_minutes, news_after_min=news_minutes,
            flatten_time=flat, stage=stage, firm=firm, preset_key=preset_key or str(getattr(rules, "preset_key", "") or ""),
            source="preset" if (preset_key or getattr(rules, "preset_key", None)) else "custom")

    @classmethod
    def from_preset(cls, key: str, *, stage: str = "evaluation", news_restricted: bool = False) -> "GuardianRules":
        from app.prop.presets import get_preset
        p = get_preset(key)
        return cls.from_prop_rules(p.to_prop_rules(), stage=stage, firm=p.firm, preset_key=key, news_restricted=news_restricted)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AccountState:
    balance: float
    equity: float | None = None                    # includes open P&L; falls back to balance
    start_of_day_balance: float | None = None
    peak_equity: float | None = None               # high-water mark for a trailing floor
    open_contracts: float = 0.0
    open_risk: float = 0.0                         # dollars lost if every open position hit its stop
    trading_days: int = 0
    best_day_profit: float = 0.0
    now: datetime | None = None

    @property
    def eq(self) -> float:
        return float(self.balance if self.equity is None else self.equity)


@dataclass
class Check:
    key: str
    label: str
    level: str = "ok"
    used: float | None = None                      # share of the limit used, 0..1+
    limit: float | None = None
    remaining: float | None = None
    message: str = ""
    kind: str = "limit"                            # limit | goal | info

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class GuardianReport:
    level: str
    headline: str
    metrics: dict
    checks: list
    action: dict
    preset: str
    rules: dict
    news: dict = field(default_factory=dict)
    disclaimer: str = ("Rules are applied from the firm preset or the values you entered, against the account's own balance. Confirm "
                       "them on the firm's rules page: firms change their rules, and this tool does not replace the firm's own risk engine.")

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["checks"] = [c.to_dict() if hasattr(c, "to_dict") else c for c in self.checks]
        return d


# ------------------------------------------------------------------------------------------------ helpers
def _ladder(used: float, th: dict) -> str:
    if used >= 1.0:
        return "flatten"
    if used >= th["block"]:
        return "block"
    if used >= th["reduce"]:
        return "reduce"
    if used >= th["warn"]:
        return "warn"
    return "ok"


def _hhmm(s: str | None):
    try:
        h, m = str(s).split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def _chicago_minutes(now: datetime | None) -> int | None:
    if now is None:
        return None
    ts = pd.Timestamp(now)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    c = ts.tz_convert("America/Chicago")
    return c.hour * 60 + c.minute


def drawdown_floor(rules: GuardianRules, state: AccountState) -> float | None:
    """Equity level the account may not touch. Trailing floors follow the equity high; a lock stops them at ``trailing_lock_floor``."""
    if not rules.max_drawdown:
        return None
    if rules.drawdown_type == "static":
        return rules.account_size - rules.max_drawdown
    peak = max(state.peak_equity if state.peak_equity is not None else rules.account_size, state.eq, rules.account_size)
    floor = peak - rules.max_drawdown
    if rules.trailing_lock_floor is not None:
        floor = min(floor, rules.trailing_lock_floor - rules.max_drawdown) if False else min(floor, rules.trailing_lock_floor)
    return floor


# ------------------------------------------------------------------------------------------------ the evaluation
def evaluate(rules: GuardianRules, state: AccountState, *, preset: str | None = None, events: pd.DataFrame | None = None,
             now: datetime | None = None) -> GuardianReport:
    """Where every rule stands and what to do now. Pure: the same inputs always give the same report."""
    pname = po.preset_name(preset)
    th = po.preset(pname)
    now = now or state.now
    eq = state.eq
    sod = state.start_of_day_balance if state.start_of_day_balance is not None else state.balance
    checks: list[Check] = []
    size_mult = 1.0

    # 1. Daily loss limit (floating P&L against the day's opening balance, as most futures firms monitor it)
    if rules.daily_loss_limit:
        loss = max(0.0, sod - eq)
        used = loss / rules.daily_loss_limit
        lvl = _ladder(used, th)
        left = max(0.0, rules.daily_loss_limit - loss)
        msg = {"ok": f"${left:,.0f} of daily loss left.", "warn": f"{used:.0%} of the daily loss limit used -- ${left:,.0f} left.",
               "reduce": f"{used:.0%} of the daily loss limit used -- trading at reduced size.",
               "block": f"{used:.0%} of the daily loss limit used -- no new entries today.",
               "flatten": "Daily loss limit reached -- flatten and stop for the day."}[lvl]
        checks.append(Check("daily_loss", "Daily loss limit", lvl, used, rules.daily_loss_limit, left, msg))

    # 2/3. Maximum drawdown and trailing drawdown
    floor = drawdown_floor(rules, state)
    if floor is not None and rules.max_drawdown:
        cushion = eq - floor
        used = max(0.0, 1.0 - cushion / rules.max_drawdown) if rules.drawdown_type == "static" or rules.trailing_lock_floor is None else max(0.0, 1.0 - cushion / rules.max_drawdown)
        used = max(used, 1.0) if cushion <= 0 else used
        lvl = _ladder(used, th)
        kind = "Trailing drawdown" if rules.drawdown_type != "static" else "Maximum drawdown"
        msg = {"ok": f"${max(cushion, 0):,.0f} above the ${floor:,.0f} floor.",
               "warn": f"${max(cushion, 0):,.0f} above the ${floor:,.0f} floor -- {used:.0%} of the drawdown used.",
               "reduce": f"Only ${max(cushion, 0):,.0f} above the floor -- trading at reduced size.",
               "block": f"Only ${max(cushion, 0):,.0f} above the floor -- no new entries.",
               "flatten": f"At or below the ${floor:,.0f} floor -- flatten everything and halt."}[lvl]
        if rules.drawdown_type != "static" and rules.trailing_lock_floor is not None and floor >= rules.trailing_lock_floor - 1e-9:
            msg += " The trailing floor has locked."
        checks.append(Check("max_drawdown", kind, lvl, used, rules.max_drawdown, max(cushion, 0.0), msg))

    # reduce-level checks set the size multiplier: 0.5 at the reduce line, falling to 0.25 at the block line
    for c in checks:
        if c.level == "reduce" and c.used is not None:
            span = max(th["block"] - th["reduce"], 1e-9)
            size_mult = min(size_mult, 0.5 - 0.25 * min(1.0, (c.used - th["reduce"]) / span))

    # 4. Profit target (and the consistency rule that gates it)
    total_profit = state.balance - rules.account_size
    if rules.profit_target:
        remaining = rules.profit_target - total_profit
        done = remaining <= 0
        msg = "Profit target reached." if done else f"${remaining:,.0f} to go."
        checks.append(Check("profit_target", "Profit target", "ok", min(1.0, max(0.0, total_profit / rules.profit_target)),
                            rules.profit_target, max(0.0, remaining), msg, kind="goal"))
    if rules.consistency_pct and total_profit > 0:
        share = state.best_day_profit / total_profit * 100.0 if total_profit else 0.0
        ok = share <= rules.consistency_pct + 1e-9
        need_total = state.best_day_profit / (rules.consistency_pct / 100.0)
        msg = (f"Best day is {share:.0f}% of total profit (limit {rules.consistency_pct:.0f}%)." if ok else
               f"Best day is {share:.0f}% of total profit, over the {rules.consistency_pct:.0f}% limit -- about ${max(0.0, need_total - total_profit):,.0f} more profit, spread over normal days, fixes it.")
        checks.append(Check("consistency", "Consistency rule", "ok" if ok else "warn", share / rules.consistency_pct, rules.consistency_pct,
                            None, msg, kind="goal"))
    if rules.min_trading_days:
        left_days = max(0, rules.min_trading_days - state.trading_days)
        checks.append(Check("min_days", "Minimum trading days", "ok", min(1.0, state.trading_days / rules.min_trading_days),
                            float(rules.min_trading_days), float(left_days),
                            "Minimum trading days met." if not left_days else f"{left_days} more trading day{'s' if left_days != 1 else ''} needed.", kind="goal"))

    # 5. Contracts and exposure
    cap = rules.max_contracts
    if rules.max_exposure:
        cap = min(cap, rules.max_exposure) if cap else rules.max_exposure
    if cap:
        open_c = float(state.open_contracts)
        if open_c > cap + 1e-9:
            lvl, msg = "flatten", f"{open_c:g} contracts open, over the {cap:g} cap -- reduce immediately."
        elif open_c >= cap - 1e-9:
            lvl, msg = "block", f"At the {cap:g}-contract cap -- no more size."
        else:
            lvl, msg = "ok", f"{cap - open_c:g} of {cap:g} contracts free."
        checks.append(Check("contracts", "Maximum contracts", lvl, open_c / cap, float(cap), max(0.0, cap - open_c), msg))
    if state.open_risk > 0 and rules.daily_loss_limit:
        room = max(0.0, rules.daily_loss_limit - max(0.0, sod - eq))
        lvl = "block" if state.open_risk >= room else ("warn" if state.open_risk >= 0.5 * room else "ok")
        msg = (f"If every open stop is hit you lose ${state.open_risk:,.0f}; ${room:,.0f} of daily loss is left."
               + (" That would breach it." if lvl == "block" else ""))
        checks.append(Check("exposure", "Position exposure", lvl, state.open_risk / room if room else 1.0, room, max(0.0, room - state.open_risk), msg))

    # 6. News restrictions (economic calendar)
    ns = {}
    if rules.news_restricted:
        ns = gnews.news_state(now, before_min=rules.news_before_min, after_min=rules.news_after_min, events=events)
        if ns["in_window"]:
            checks.append(Check("news", "News restriction", "block", None, None, None,
                                f"{ns['event']} window -- no new entries until {pd.Timestamp(ns['ends']):%H:%M} ET.", kind="limit"))
        elif ns["minutes_to_next"] is not None and ns["minutes_to_next"] <= 15:
            checks.append(Check("news", "News restriction", "warn", None, None, None,
                                f"{ns['next_event']} in {ns['minutes_to_next']:.0f} min -- entries pause {rules.news_before_min} min before.", kind="limit"))
        else:
            nxt = f" Next: {ns['next_event']} at {pd.Timestamp(ns['next_at']):%H:%M} ET." if ns["next_event"] else ""
            checks.append(Check("news", "News restriction", "ok", None, None, None, "No restricted window right now." + nxt, kind="info"))

    # 7. Trading hours and the session flatten
    cm = _chicago_minutes(now)
    if cm is not None and (rules.trading_start or rules.trading_end or rules.flatten_time):
        lvl, msg = "ok", "Inside trading hours."
        s, e, f = _hhmm(rules.trading_start), _hhmm(rules.trading_end), _hhmm(rules.flatten_time)
        if f is not None and cm >= f and cm < 17 * 60:
            lvl, msg = ("flatten" if state.open_contracts else "block"), f"Past the {rules.flatten_time} CT flatten time."
        elif f is not None and 0 <= f - cm <= 15:
            lvl, msg = "warn", f"{f - cm} min to the {rules.flatten_time} CT flatten time."
        elif s is not None and e is not None and not (s <= cm < e):
            lvl, msg = "block", f"Outside the {rules.trading_start}-{rules.trading_end} CT trading window."
        checks.append(Check("hours", "Trading hours", lvl, None, None, None, msg, kind="limit"))

    # roll the checks up into one action
    level = "ok"
    for c in checks:
        level = _worse(level, c.level)
    drew_flatten = [c for c in checks if c.level == "flatten"]
    halt_account = any(c.key == "max_drawdown" for c in drew_flatten)
    halt_day = any(c.key in ("daily_loss", "max_drawdown") for c in drew_flatten)
    reasons = [f"{c.label}: {c.message}" for c in checks if c.level != "ok"]
    can_open = level in ("ok", "warn", "reduce")
    action = {"level": level, "size_multiplier": round(size_mult, 2) if can_open else 0.0, "can_open": can_open,
              "flatten": level == "flatten", "halt_day": halt_day, "halt_account": halt_account, "reasons": reasons,
              "label": {"ok": "All clear", "warn": "Warning", "reduce": "Reduce size", "block": "No new entries",
                        "flatten": "Flatten and halt for the day" if halt_day else "Flatten"}[level]}

    daily_left = next((c.remaining for c in checks if c.key == "daily_loss"), None)
    cushion = next((c.remaining for c in checks if c.key == "max_drawdown"), None)
    target_left = next((c.remaining for c in checks if c.key == "profit_target"), None)
    metrics = {"balance": round(state.balance, 2), "equity": round(eq, 2),
               "daily_loss_remaining": None if daily_left is None else round(daily_left, 2),
               "drawdown_cushion": None if cushion is None else round(cushion, 2),
               "profit_target_remaining": None if target_left is None else round(target_left, 2),
               "drawdown_floor": None if floor is None else round(floor, 2),
               "open_contracts": state.open_contracts, "trading_days": state.trading_days}
    headline = action["label"] if not reasons else f"{action['label']} -- {reasons[0]}"
    return GuardianReport(level=level, headline=headline, metrics=metrics, checks=checks, action=action, preset=pname,
                          rules=rules.to_dict(), news=ns)


# ------------------------------------------------------------------------------------------------ live session memory
class GuardianTracker:
    """What a live session must remember between polls: the day's opening balance, the equity high-water mark, trading days and the
    best day. ``update`` takes the broker's own balance / equity every poll; the day rolls on the shared 17:00 CT session clock."""

    def __init__(self, rules: GuardianRules, preset: str | None = None, *, events: pd.DataFrame | None = None):
        self.rules, self.preset, self.events = rules, po.preset_name(preset), events
        self._day = None
        self._sod: float | None = None
        self._peak: float | None = None
        self._days_traded: set = set()
        self._day_start_balance: dict = {}
        self._best_day = 0.0
        self._last_balance: float | None = None
        self.last: GuardianReport | None = None

    def update(self, balance: float | None, equity: float | None, now, *, open_contracts: float = 0.0, open_risk: float = 0.0,
               traded_today: bool = False) -> GuardianReport | None:
        if balance is None:
            return self.last
        from app.data.trading_day import trading_day
        ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        day = trading_day(ts)
        if day != self._day:
            if self._day is not None and self._sod is not None and self._last_balance is not None:
                self._best_day = max(self._best_day, self._last_balance - self._sod)
            self._day, self._sod = day, float(balance)
        if traded_today:
            self._days_traded.add(day)
        eq = float(equity if equity is not None else balance)
        self._peak = max(self._peak if self._peak is not None else self.rules.account_size, eq)
        self._last_balance = float(balance)
        today_gain = float(balance) - (self._sod or float(balance))
        st = AccountState(balance=float(balance), equity=eq, start_of_day_balance=self._sod, peak_equity=self._peak,
                          open_contracts=open_contracts, open_risk=open_risk, trading_days=len(self._days_traded),
                          best_day_profit=max(self._best_day, today_gain), now=ts)
        self.last = evaluate(self.rules, st, preset=self.preset, events=self.events, now=ts)
        save_latest(self.last)
        return self.last


# ------------------------------------------------------------------------------------------------ latest report (for the Mission page)
def _latest_path():
    from app.data.storage import get_app_base_dir
    d = get_app_base_dir() / "data" / "guardian"
    d.mkdir(parents=True, exist_ok=True)
    return d / "latest.json"


def save_latest(report: GuardianReport) -> None:
    """Remember the newest live report so the Mission page can say what the account is doing right now. Best effort."""
    import json
    import time
    try:
        _latest_path().write_text(json.dumps({"saved_at": time.time(), "report": report.to_dict()}, default=str), encoding="utf-8")
    except OSError:
        pass


def load_latest(max_age_seconds: float = 900.0) -> dict | None:
    """The newest live report, or None when there is none or it is older than ``max_age_seconds`` (a stale report is never shown as live)."""
    import json
    import time
    try:
        d = json.loads(_latest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - float(d.get("saved_at", 0)) > max_age_seconds:
        return None
    return d.get("report")
