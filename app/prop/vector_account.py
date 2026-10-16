"""Vectorized prop-account kernel for Monte Carlo (2026-10-15 audit, speed item).

app.prop.account.PropAccount runs ONE simulated path trade by trade in pure
Python; Monte Carlo ran 10,000 of them one after another. This module mirrors
the SAME state machine across P paths in lock-step: it loops over the TRADE
INDEX (not over simulations) and keeps one numpy state array per rule value
(balance, trailing peak, day P&L, ...). The arithmetic is the same IEEE-754
double operations in the same order as PropAccount, so results are bit-identical
(pinned by tests/test_vector_account_parity.py, which differential-tests it
against simulate_account on random rule sets and random paths).

Scope. The kernel covers one attempt per path (``reset_on_breach=False``, the
Monte Carlo default) with every PropRules feature the Monte Carlo can reach:
dd_basis legacy/realized/eod/floating, daily_loss_basis realized/floating,
daily_loss_action fail/lock_day, daily_loss_base initial/prior_day_high/
ratchet_up, trailing/static, trailing lock + account-distance basis,
funded_starts_fresh, eval consistency, min trading days, eval calendar limit,
inactivity limit, funded payouts (frequency, buffer, caps, winning days,
consistency, max payouts). ``supports_vectorized`` returns False for anything
else and the caller falls back to the original per-path loop, so a rule the
kernel does not know can never silently change a number.

No new dependency: numpy only. ``T58_MC_VECTORIZE=0`` switches it off.
"""
from __future__ import annotations

import os
from typing import Any, NamedTuple, Optional

import numpy as np

_NEG_INF = float("-inf")


def vectorize_enabled() -> bool:
    return (os.environ.get("T58_MC_VECTORIZE", "1").strip().lower() not in ("0", "false", "no", "off"))


def supports_vectorized(rules: Any, reset_on_breach: bool, *, have_excursions: bool,
                        caps_active: bool = False) -> bool:
    """True when the kernel reproduces PropAccount exactly for these rules."""
    if reset_on_breach or caps_active:
        return False
    dd_basis = getattr(rules, "dd_basis", "legacy") or "legacy"
    dl_basis = getattr(rules, "daily_loss_basis", "realized") or "realized"
    if dd_basis not in ("legacy", "realized", "eod", "floating") or dl_basis not in ("realized", "floating"):
        return False
    if (dd_basis in ("eod", "floating") or dl_basis == "floating") and not have_excursions:
        return False  # legacy path owns the missing-excursion policy (warn / raise)
    if getattr(rules, "daily_loss_action", "fail") not in ("fail", "lock_day"):
        return False
    if getattr(rules, "daily_loss_base", "initial") not in ("initial", "prior_day_high", "ratchet_up"):
        return False
    if getattr(rules, "drawdown_type", "trailing") not in ("trailing", "static"):
        return False
    if getattr(rules, "daily_loss_limit_pct", None) is None:
        return False
    return True


class DayTables(NamedTuple):
    """Per-run (not per-path) calendar tables indexed by day SLOT."""
    cal0: np.ndarray      # calendar days from slot 0 to slot i
    gap: np.ndarray       # calendar days from slot i-1 to slot i (gap[0] = 0)


def build_day_tables(day_dates) -> DayTables:
    import pandas as pd
    n = len(day_dates)
    ts = [pd.Timestamp(d) for d in day_dates]
    cal0 = np.fromiter(((t - ts[0]).days for t in ts), dtype=np.int64, count=n) if n else np.zeros(0, np.int64)
    gap = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        gap[i] = (ts[i] - ts[i - 1]).days
    return DayTables(cal0, gap)


class VecResult(NamedTuple):
    passed: np.ndarray
    failed: np.ndarray
    days_to_pass: np.ndarray       # -1 = never
    first_payout_day: np.ndarray   # -1 = never
    n_payouts: np.ndarray
    total_payout: np.ndarray
    final_balance: np.ndarray
    max_dd_pct: np.ndarray
    last_slot: np.ndarray
    max_losing_streak: np.ndarray


def simulate_paths(
    rules: Any,
    path_len: np.ndarray,
    get_step,
    n_steps: int,
    tables: DayTables,
) -> VecResult:
    """Run P paths in lock-step.

    ``get_step(t)`` returns ``(pnl, mae, mfe, risk, slot, is_last)`` for trade
    position ``t`` across all P paths (1-D arrays of length P; ``mae``/``mfe``/
    ``risk`` may be None). Positions >= ``path_len[p]`` are padding and are
    ignored for path p. Day slots are contiguous per path starting at 0.
    """
    P = int(path_len.shape[0])
    r = rules
    acct = float(r.account_size)
    f64 = np.float64

    dd_basis = getattr(r, "dd_basis", "legacy") or "legacy"
    dl_basis = getattr(r, "daily_loss_basis", "realized") or "realized"
    legacy = dd_basis == "legacy"
    trailing = r.drawdown_type == "trailing"
    peak_basis = getattr(r, "trailing_distance_basis", "peak") != "account"
    t_lock = bool(getattr(r, "trailing_lock", False))
    lock_level = acct * (1 + float(getattr(r, "trailing_lock_offset_pct", 0.0) or 0.0) / 100.0)
    static_floor = acct * (1 - r.max_drawdown_pct / 100.0)
    dd_mult = (1 - r.max_drawdown_pct / 100.0)
    dd_dist = acct * (r.max_drawdown_pct / 100.0)
    floating_dd = dd_basis in ("floating", "eod")
    floating_dl = dl_basis == "floating"
    floating_active = floating_dd or floating_dl
    lock_action = r.daily_loss_action == "lock_day"
    dl_base = r.daily_loss_base
    dl_frac = r.daily_loss_limit_pct / 100.0
    dl_disabled_float = (r.daily_loss_limit_pct >= 100.0)     # daily_floor_equity() -> None
    thr = getattr(r, "min_winning_day_profit", 0.0)
    max_inactive = getattr(r, "max_inactive_days", None)
    eval_cal_limit = r.max_eval_calendar_days
    fresh = bool(getattr(r, "funded_starts_fresh", False))
    target_balance = acct * (1 + r.evaluation_profit_target_pct / 100.0)
    required_profit = acct * (r.payout_threshold_pct / 100.0) + acct * (r.required_buffer_pct / 100.0)
    buffer_dollars = acct * (r.required_buffer_pct / 100.0)
    adverse_mode = legacy and getattr(r, "floating_drawdown_mode", "realized") == "adverse"

    bal = np.full(P, acct, dtype=f64)
    peak = np.full(P, acct, dtype=f64)
    failed = np.zeros(P, dtype=bool)
    funded = np.zeros(P, dtype=bool)
    passed = np.zeros(P, dtype=bool)
    days_to_pass = np.full(P, -1, dtype=np.int64)
    locked = np.full(P, -1, dtype=np.int64)
    cur_slot = np.full(P, -1, dtype=np.int64)
    last_slot = np.zeros(P, dtype=np.int64)
    day_pnl = np.zeros(P, dtype=f64)
    day_start = np.zeros(P, dtype=f64)
    best_day = np.zeros(P, dtype=f64)
    total_profit = np.zeros(P, dtype=f64)
    wd_total = np.zeros(P, dtype=np.int64)
    wd_before = np.zeros(P, dtype=np.int64)
    best_since = np.zeros(P, dtype=f64)
    baseline = np.full(P, acct, dtype=f64)
    last_payout = np.full(P, -10 ** 9, dtype=np.int64)
    n_pay = np.zeros(P, dtype=np.int64)
    first_pay = np.full(P, -1, dtype=np.int64)
    total_pay = np.zeros(P, dtype=f64)
    max_dd = np.zeros(P, dtype=f64)
    day_peak_cur = np.full(P, _NEG_INF, dtype=f64)
    day_peak_set = np.zeros(P, dtype=bool)
    day_close_cur = np.zeros(P, dtype=f64)
    day_close_set = np.zeros(P, dtype=bool)
    prev_day_peak_base = np.full(P, acct, dtype=f64)   # day_peak.get(cur-1, account_size)
    max_close_prior = np.full(P, acct, dtype=f64)      # max([account_size] + closes of earlier days)
    streak_run = np.zeros(P, dtype=np.int64)
    streak_best = np.zeros(P, dtype=np.int64)

    cal0, gap = tables.cal0, tables.gap

    def floor_of(pk):
        if not trailing:
            return np.full(P, static_floor, dtype=f64)
        fl = pk * dd_mult if peak_basis else pk - dd_dist
        if t_lock:
            fl = np.minimum(fl, lock_level)
        return fl

    def dd_pct(pk, level):
        if trailing:
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.maximum(0.0, (pk - level) / pk * 100.0)
            return np.where(pk != 0, v, 0.0)
        return np.maximum(0.0, (acct - level) / acct * 100.0) if acct else np.zeros(P, dtype=f64)

    def daily_limit():
        if dl_base == "initial":
            base = acct
        elif dl_base == "prior_day_high":
            base = prev_day_peak_base
        else:
            base = max_close_prior
        return base * dl_frac

    for t in range(n_steps):
        valid = path_len > t
        pnl, mae, mfe, risk, slot, is_last = get_step(t)
        # max losing streak runs over EVERY resampled trade (as _max_losing_streak does)
        loss = (pnl <= 0) & valid
        streak_run = (streak_run + 1) * loss
        streak_best = np.maximum(streak_best, streak_run)

        act = valid & ~failed & (locked != slot)
        if not act.any():
            continue
        sl = slot
        # ---- new session day: roll the per-day trackers ---------------------------------
        new_day = act & (cur_slot != sl)
        if new_day.any():
            started = new_day & (cur_slot >= 0)
            prev_day_peak_base = np.where(new_day, np.where(day_peak_set, day_peak_cur, acct), prev_day_peak_base)
            max_close_prior = np.where(new_day & day_close_set, np.maximum(max_close_prior, day_close_cur), max_close_prior)
        else:
            started = new_day
        last_slot = np.where(act, sl, last_slot)
        go = act.copy()
        # ---- inactivity rule (calendar gap between two trade days) ----------------------
        if max_inactive is not None:
            f = go & started & ((gap[np.clip(sl, 0, len(gap) - 1)] - 1) > max_inactive)
            if f.any():
                failed |= f
                go &= ~f
        # ---- begin_day ------------------------------------------------------------------
        day_start = np.where(new_day, bal, day_start)
        day_pnl = np.where(new_day, 0.0, day_pnl)
        day_peak_cur = np.where(new_day, _NEG_INF, day_peak_cur)
        day_peak_set = np.where(new_day, False, day_peak_set)
        day_close_set = np.where(new_day, False, day_close_set)
        cur_slot = np.where(new_day, sl, cur_slot)
        # ---- floating pre-check from the trade's own excursion ---------------------------
        if floating_active:
            bb = bal
            adv = np.minimum(mae, 0.0)
            fav = np.maximum(mfe, 0.0)
            high = bb + fav
            low = bb + adv
            if floating_dd and trailing and dd_basis == "floating":
                peak = np.where(go, np.maximum(peak, high), peak)
            if floating_dd:
                fl = floor_of(peak)
                cdd = dd_pct(peak, low)
                max_dd = np.where(go, np.maximum(max_dd, cdd), max_dd)
                f = go & (low <= fl)
                failed |= f
                go &= ~f
            if floating_dl and not dl_disabled_float:
                dfloor = day_start - daily_limit()
                hit = go & (low <= dfloor)
                if hit.any():
                    if lock_action:
                        locked = np.where(hit, sl, locked)
                    else:
                        failed |= hit
                    go &= ~hit
        if not go.any():
            continue
        # ---- realized accounting ----------------------------------------------------------
        was_funded = funded.copy()
        old_day = day_pnl
        bal = np.where(go, bal + pnl, bal)
        day_pnl = np.where(go, day_pnl + pnl, day_pnl)
        was_win = (~new_day) & (old_day >= thr)
        is_win = day_pnl >= thr
        wd_total = wd_total + np.where(go & is_win & ~was_win, 1, 0) - np.where(go & was_win & ~is_win, 1, 0)
        total_profit = np.where(go, total_profit + pnl, total_profit)
        best_day = np.where(go, np.maximum(best_day, day_pnl), best_day)

        if legacy:
            cn = go if r.drawdown_check_mode == "intrabar" else (go & is_last)
        else:
            cn = go
        if not cn.any():
            continue
        day_peak_cur = np.where(cn, np.maximum(day_peak_cur, bal), day_peak_cur)
        day_peak_set = day_peak_set | cn
        m = cn & is_last
        day_close_cur = np.where(m, bal, day_close_cur)
        day_close_set = day_close_set | m
        if dd_basis == "eod":
            peak = np.where(m, np.maximum(peak, bal), peak)
        else:
            peak = np.where(cn, np.maximum(peak, bal), peak)
        if trailing:
            dd_floor = floor_of(peak)
        else:
            dd_floor = np.full(P, static_floor, dtype=f64)
        max_dd = np.where(cn, np.maximum(max_dd, dd_pct(peak, bal)), max_dd)
        # daily loss on the realized day P&L
        dll = daily_limit()
        hit = cn & (day_pnl <= -dll)
        if hit.any():
            if lock_action:
                day_close_cur = np.where(hit, bal, day_close_cur)
                day_close_set = day_close_set | hit
                locked = np.where(hit, sl, locked)
            else:
                failed |= hit
            cn = cn & ~hit
        f = cn & (bal <= dd_floor)
        failed |= f
        cn = cn & ~f
        if adverse_mode and risk is not None:
            f = cn & (((bal - pnl) - risk) <= dd_floor)
            failed |= f
            cn = cn & ~f
        if not cn.any():
            continue
        # ---- evaluation pass ---------------------------------------------------------------
        ev = cn & ~was_funded
        if eval_cal_limit is not None:
            f = ev & (cal0[np.clip(sl, 0, len(cal0) - 1)] > eval_cal_limit)
            failed |= f
            ev = ev & ~f
        tds = sl + 1
        cond = ev & (bal >= target_balance) & (tds >= r.min_trading_days)
        if r.consistency_rule_pct is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = best_day / total_profit * 100.0
            ok = np.where(total_profit > 0, ratio <= r.consistency_rule_pct, True)
            cond = cond & ok
        if cond.any():
            funded = funded | cond
            passed = passed | cond
            days_to_pass = np.where(cond, tds, days_to_pass)
            if fresh:
                bal = np.where(cond, acct, bal)
                peak = np.where(cond, acct, peak)
            baseline = np.where(cond, bal, baseline)
            best_since = np.where(cond, 0.0, best_since)
            last_payout = np.where(cond, sl, last_payout)
            cur_win = (day_pnl >= thr).astype(np.int64)
            wd_before = np.where(cond, wd_total if fresh else wd_total - cur_win, wd_before)
        # ---- funded payouts (only for accounts that were ALREADY funded before this trade) -
        fu = cn & was_funded
        if fu.any():
            profit_since = bal - baseline
            best_since = np.where(fu, np.maximum(best_since, day_pnl), best_since)
            fcons = np.ones(P, dtype=bool)
            if r.funded_consistency_rule_pct is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    fr = best_since / profit_since * 100.0
                fcons = np.where(profit_since > 0, fr <= r.funded_consistency_rule_pct, True)
            ok = (fu & (profit_since >= required_profit)
                  & ((sl - last_payout) >= r.payout_frequency_days)
                  & ((wd_total - wd_before) >= r.winning_days_for_payout)
                  & fcons)
            if r.max_payouts is not None:
                ok = ok & (n_pay < r.max_payouts)
            if ok.any():
                w = profit_since - buffer_dollars
                if r.payout_cap_pct is not None:
                    w = np.minimum(w, profit_since * (r.payout_cap_pct / 100.0))
                if r.payout_cap_dollars is not None:
                    w = np.minimum(w, r.payout_cap_dollars)
                w = np.maximum(w, 0.0)
                pay = ok & (w > 0)
                if pay.any():
                    bal = np.where(pay, bal - w, bal)
                    n_pay = n_pay + pay
                    first_pay = np.where(pay & (first_pay < 0), sl, first_pay)
                    total_pay = np.where(pay, total_pay + w, total_pay)
                    baseline = np.where(pay, bal, baseline)
                    last_payout = np.where(pay, sl, last_payout)
                    best_since = np.where(pay, 0.0, best_since)
                    peak = np.where(pay, np.maximum(peak - w, bal), peak)
    return VecResult(
        passed=passed, failed=failed, days_to_pass=days_to_pass, first_payout_day=first_pay,
        n_payouts=n_pay, total_payout=total_pay, final_balance=bal, max_dd_pct=max_dd,
        last_slot=last_slot, max_losing_streak=streak_best,
    )
