"""Behaviour rules the optimizer can search (2026-10 audit, build item 5).

Four trader-behaviour rules that decide whether a prop account SURVIVES long enough to be paid, none of which can
live inside a stateless signal function (it cannot see its own P&L or the clock):

    stop-for-the-day after N losses   BEHAVIOR_STOP_AFTER_LOSSES        int   0 = off
    size down near the target         BEHAVIOR_SIZE_DOWN_PROGRESS_PCT   float 0 = off; x BEHAVIOR_SIZE_DOWN_MULT once
                                                                              that % of the profit target is banked
    no entries inside news windows    BEHAVIOR_NEWS_BLOCK_MIN           int   0 = off; no entries within +/-N minutes
                                                                              of the recurring release times in
                                                                              NEWS_ANCHORS_CT (08:30, 10:00, 13:00 CT)
    flatten before the close          BEHAVIOR_FLATTEN_BEFORE_CLOSE_MIN int   0 = off; flat by 17:00 CT - N minutes

A strategy declares them as top-level constants (Python: ``BEHAVIOR_STOP_AFTER_LOSSES = 2``; Manual config:
``"behavior": {...}`` with the same keys). Because they are plain top-level numerics, the code-gene discovery in
app.optimize.code_parameter_space finds them and the GA searches them with the domain bounds in BEHAVIOR_BOUNDS --
no new optimizer is needed. run_backtest calls ``apply_strategy_behavior`` next to ``apply_strategy_trade_cap`` and the
ENGINE enforces them (they are overlays on entries/size, never on stops/targets):

    stop-for-the-day  -> AdaptiveRiskRule("daily_losses", N, 0.0)   (new trigger; releases next session day)
    size down         -> AdaptiveRiskRule("progress_to_target_pct", X, MULT)  (needs the profit target amount)
    news windows      -> RiskConfig.news_blackout_windows (entry mask on recurring clock windows, CT)
    flatten           -> RiskConfig.flatten_time_ct (existing session-flatten machinery, now usable without prop mode)

Honest limits: the news windows are RECURRING daily clock windows around the usual US release times (they also block
quiet days -- a blunt tool); date-specific events still go through RiskConfig.news_blackout_csv.
"""
from __future__ import annotations

import re
from dataclasses import replace
from datetime import time as dt_time

import numpy as np
import pandas as pd

BEHAVIOR_DEFAULTS = {
    "BEHAVIOR_STOP_AFTER_LOSSES": 0,
    "BEHAVIOR_SIZE_DOWN_PROGRESS_PCT": 0.0,
    "BEHAVIOR_SIZE_DOWN_MULT": 0.5,
    "BEHAVIOR_NEWS_BLOCK_MIN": 0,
    "BEHAVIOR_FLATTEN_BEFORE_CLOSE_MIN": 0,
}
# (lo, hi) the GA may use for each constant (code_parameter_space reads this table)
BEHAVIOR_BOUNDS = {
    "BEHAVIOR_STOP_AFTER_LOSSES": (0.0, 6.0),
    "BEHAVIOR_SIZE_DOWN_PROGRESS_PCT": (0.0, 95.0),
    "BEHAVIOR_SIZE_DOWN_MULT": (0.2, 0.9),
    "BEHAVIOR_NEWS_BLOCK_MIN": (0.0, 30.0),
    "BEHAVIOR_FLATTEN_BEFORE_CLOSE_MIN": (0.0, 90.0),
}
NEWS_ANCHORS_CT = ("08:30", "10:00", "13:00")   # CPI/jobs, ISM/consumer confidence, FOMC statement (CT wall clock)
SESSION_ROLL_CT_MINUTES = 17 * 60


def resolve_behavior(strategy) -> dict:
    """The strategy's declared behaviour constants merged over BEHAVIOR_DEFAULTS ({} values = all off)."""
    out = dict(BEHAVIOR_DEFAULTS)
    declared: dict = {}
    st = getattr(strategy, "source_type", None)
    if st == "python":
        ma = getattr(strategy, "module_attr", None)
        if ma is not None:
            for k in BEHAVIOR_DEFAULTS:
                v = ma(k, None)
                if v is not None:
                    declared[k] = v
    elif st == "manual":
        cfg = getattr(strategy, "config", None)
        b = cfg.get("behavior") if isinstance(cfg, dict) else None
        if isinstance(b, dict):
            for k, v in b.items():
                key = str(k).upper()
                key = key if key.startswith("BEHAVIOR_") else "BEHAVIOR_" + key
                if key in BEHAVIOR_DEFAULTS:
                    declared[key] = v
    for k, v in declared.items():
        try:
            out[k] = float(v) if isinstance(BEHAVIOR_DEFAULTS[k], float) else int(round(float(v)))
        except (TypeError, ValueError):
            continue
    return out


def any_active(b: dict) -> bool:
    return bool(b.get("BEHAVIOR_STOP_AFTER_LOSSES", 0) >= 1 or b.get("BEHAVIOR_SIZE_DOWN_PROGRESS_PCT", 0) > 0
                or b.get("BEHAVIOR_NEWS_BLOCK_MIN", 0) >= 1 or b.get("BEHAVIOR_FLATTEN_BEFORE_CLOSE_MIN", 0) >= 1)


def news_windows_text(block_minutes: int, anchors=NEWS_ANCHORS_CT) -> str:
    """One window per line, 'HH:MM-HH:MM' (CT), +/- block_minutes around every anchor."""
    n = int(block_minutes)
    lines = []
    for a in anchors:
        h, m = (int(x) for x in a.split(":"))
        c = h * 60 + m
        lo, hi = max(0, c - n), min(24 * 60 - 1, c + n)
        lines.append(f"{lo // 60:02d}:{lo % 60:02d}-{hi // 60:02d}:{hi % 60:02d}")
    return "\n".join(lines)


def flatten_time_text(before_close_min: int) -> str:
    t = SESSION_ROLL_CT_MINUTES - int(before_close_min)
    return f"{t // 60:02d}:{t % 60:02d}"


_WINDOW_RE = re.compile(r"^\s*(?:(MON|TUE|WED|THU|FRI|SAT|SUN)\s+)?(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$", re.I)
_DAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def parse_windows(raw: str | None) -> list:
    """Same line grammar as the live engine's parse_blackout_windows ('HH:MM-HH:MM' or 'FRI 07:30-07:45').
    Returns [(weekday|None, start_minute, end_minute)]. Bad lines are skipped (a typo disables that line only)."""
    out = []
    for line in (raw or "").splitlines():
        m = _WINDOW_RE.match(line)
        if not m:
            continue
        d, sh, sm, eh, em = m.groups()
        out.append((_DAYS.get(d.upper()) if d else None, int(sh) * 60 + int(sm), int(eh) * 60 + int(em)))
    return out


def window_mask(ts_naive_utc, windows: list, tz: str = "America/Chicago") -> np.ndarray:
    """Boolean mask over naive-UTC timestamps: True where the bar's CT wall-clock time sits inside any window.
    Windows crossing midnight (start > end) wrap."""
    idx = pd.DatetimeIndex(ts_naive_utc)
    if not windows or len(idx) == 0:
        return np.zeros(len(idx), dtype=bool)
    ct = idx.tz_localize("UTC").tz_convert(tz)
    mod = (ct.hour * 60 + ct.minute).to_numpy()
    dow = ct.dayofweek.to_numpy()
    mask = np.zeros(len(idx), dtype=bool)
    for wd, s, e in windows:
        inside = (mod >= s) & (mod <= e) if s <= e else (mod >= s) | (mod <= e)
        if wd is not None:
            inside &= dow == wd
        mask |= inside
    return mask


def _profit_target_amount(risk) -> float | None:
    rules = getattr(risk, "prop_account_rules", None)
    if rules is not None:
        try:
            from app.prop.account import coerce_rules
            r = coerce_rules(rules)
            return float(r.account_size) * float(r.evaluation_profit_target_pct) / 100.0
        except Exception:  # noqa: BLE001
            pass
    pct = getattr(risk, "profit_target_pct", None)
    if pct:
        return float(risk.initial_balance) * float(pct) / 100.0
    return None


def apply_strategy_behavior(risk, adaptive_risk, strategy):
    """(risk, adaptive_risk) with the strategy's declared behaviour rules layered on. Unchanged objects when the strategy
    declares nothing (byte-identical to before). Rules only ever ADD restrictions to what the caller passed in."""
    b = resolve_behavior(strategy)
    if not any_active(b):
        return risk, adaptive_risk
    from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskRule
    rules = list(getattr(adaptive_risk, "rules", []) or [])
    target_amt = getattr(adaptive_risk, "profit_target_amount", None)
    n = int(b["BEHAVIOR_STOP_AFTER_LOSSES"])
    if n >= 1:
        rules.append(AdaptiveRiskRule("daily_losses", n, 0.0, label=f"behavior: stop for the day after {n} loss(es)"))
    pp = float(b["BEHAVIOR_SIZE_DOWN_PROGRESS_PCT"])
    if pp > 0:
        mult = min(max(float(b["BEHAVIOR_SIZE_DOWN_MULT"]), 0.05), 1.0)
        rules.append(AdaptiveRiskRule("progress_to_target_pct", pp, mult, label=f"behavior: x{mult:g} size once {pp:g}% of target is banked"))
        if not target_amt:
            target_amt = _profit_target_amount(risk)
    new_adaptive = adaptive_risk
    if rules != list(getattr(adaptive_risk, "rules", []) or []):
        base = adaptive_risk or AdaptiveRiskConfig()
        new_adaptive = replace(base, enabled=True, rules=rules, profit_target_amount=target_amt)
    kw = {}
    nb = int(b["BEHAVIOR_NEWS_BLOCK_MIN"])
    if nb >= 1:
        existing = getattr(risk, "news_blackout_windows", None) or ""
        kw["news_blackout_windows"] = (existing + "\n" if existing else "") + news_windows_text(nb)
    fl = int(b["BEHAVIOR_FLATTEN_BEFORE_CLOSE_MIN"])
    if fl >= 1:
        cand = flatten_time_text(fl)
        cur = getattr(risk, "flatten_time_ct", None)
        # an earlier flatten time is the stricter one; never loosen what the caller set
        if not cur or _to_min(cand) < _to_min(cur):
            kw["flatten_time_ct"] = cand
    return (replace(risk, **kw) if kw else risk), new_adaptive


def _to_min(s: str) -> int:
    h, m = (int(x) for x in str(s).split(":"))
    return h * 60 + m
