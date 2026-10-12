"""Scripted-day rulebook replays per firm preset (2026-10 audit).

``simulate_account`` / ``PropAccount`` are only as good as their agreement with each preset's DECLARED rules. This module
replays short, hand-built days whose correct outcome follows directly from a preset's own fields and checks the account
produces it:

    clean_pass         evenly spread winning days that reach the target -> evaluation passed, no failure
    daily_loss_breach  one trade losing 105% of the daily limit            -> failed (action 'fail') or day locked, account alive ('lock_day')
    max_dd_breach      small daily losses summing to 102% of the max drawdown -> failed 'max_drawdown'
    trailing_vs_static a gain, then losses of 102% of the drawdown FROM THE PEAK (net still above the static floor)
                       -> fails iff the drawdown is trailing (and the lock has not frozen the floor)
    intraday_excursion a trade that closes green but whose MAE exceeds the limits -> fails iff the preset watches floating equity
                     (dd_basis floating or eod -- 'eod' trails the floor at end of day but breaches intraday -- or a floating
                     daily-loss basis with action 'fail')

It checks the SIMULATOR against the PRESET. It cannot check the preset against the FIRM -- that is what
PropFirmPreset.verified_on / source_url (app.prop.presets) record. Run:  python -m app.prop.rulebook_tests
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from app.prop.excursions import ExcursionSet  # noqa: F401  (documented dependency of the scripted days)
from app.prop.simulator import simulate_account


@dataclass
class RulebookCase:
    preset_key: str
    name: str
    expected: str
    got: str
    ok: bool


def _days(n: int, start: str = "2026-03-02") -> list:
    return list(pd.bdate_range(start, periods=n))


def _run(rules, pnls, dates, mae=None, mfe=None):
    mae = mae if mae is not None else [min(p, 0.0) for p in pnls]
    mfe = mfe if mfe is not None else [max(p, 0.0) for p in pnls]
    return simulate_account(list(pnls), list(dates), rules, reset_on_breach=False, trade_mae=list(mae), trade_mfe=list(mfe))


def _fmt(res) -> str:
    return f"passed={res.passed_evaluation} failed={res.failed} reason={res.failure_reason}"


def run_rulebook_tests(preset_key: str) -> list:
    from app.prop.presets import get_preset
    p = get_preset(preset_key)
    r = p.to_prop_rules()
    size = float(r.account_size)
    target = size * r.evaluation_profit_target_pct / 100.0
    day_lim = size * r.daily_loss_limit_pct / 100.0
    dd = size * r.max_drawdown_pct / 100.0
    out: list[RulebookCase] = []

    def add(name, expected, res, ok):
        out.append(RulebookCase(preset_key, name, expected, _fmt(res), bool(ok)))

    # 1. clean pass ---------------------------------------------------------------------------------------------
    k = max(int(r.min_trading_days or 1), 6, int(math.ceil(100.0 / r.consistency_rule_pct)) + 2 if r.consistency_rule_pct else 6)
    per_day = target * 1.03 / k
    if per_day < day_lim * 0.9:
        res = _run(r, [per_day] * k, _days(k))
        add("clean_pass", "evaluation passed, not failed", res, res.passed_evaluation and not res.failed)

    # 2. daily-loss breach ------------------------------------------------------------------------------------------
    loss = day_lim * 1.05
    if loss < dd * 0.98:
        res = _run(r, [-loss], _days(1))
        if r.daily_loss_action == "fail":
            add("daily_loss_breach", "account FAILED (daily loss, action=fail)", res, res.failed)
        else:
            add("daily_loss_breach", "account alive, day locked (action=lock_day)", res, not res.failed)

    # 3. max drawdown breach by small daily losses -----------------------------------------------------------------
    chunk = day_lim * 0.7
    n = int(math.ceil(dd * 1.02 / chunk))
    res = _run(r, [-(dd * 1.02 / n)] * n, _days(n))
    add("max_dd_breach", "account FAILED (max drawdown)", res, res.failed)

    # 4. trailing vs static -----------------------------------------------------------------------------------------
    gain = min(target * 0.6, day_lim * 0.9)
    lock_floor_hit = bool(r.trailing_lock) and gain >= dd + size * (r.trailing_lock_offset_pct or 0.0) / 100.0
    # trailing distance: a fixed dollar amount ('account') or a percentage of the peak ('peak', the legacy convention)
    dd_trail = dd if getattr(r, "trailing_distance_basis", "peak") == "account" else (size + gain) * r.max_drawdown_pct / 100.0
    n2 = int(math.ceil(dd_trail * 1.02 / chunk))
    pnls = [gain] + [-(dd_trail * 1.02 / n2)] * n2
    net_above_static = gain - dd_trail * 1.02 > -dd
    if net_above_static and not lock_floor_hit:
        res = _run(r, pnls, _days(len(pnls)))
        trailing = r.drawdown_type == "trailing"
        add("trailing_vs_static", "FAILED" if trailing else "NOT failed (static floor not reached)", res, res.failed == trailing)

    # 5. intraday excursion ------------------------------------------------------------------------------------------
    mae = -max(day_lim, dd) * 1.1
    res = _run(r, [25.0], _days(1), mae=[mae], mfe=[25.0])
    # 'eod' trails the FLOOR on end-of-day balance but still breaches on intraday equity touching it (Apex-EOD style);
    # only 'realized'/'legacy' ignore the open P&L.
    floating_dd = r.dd_basis in ("floating", "eod")
    floating_dl_fail = r.daily_loss_basis == "floating" and r.daily_loss_action == "fail"
    fail_expected = floating_dd or floating_dl_fail
    add("intraday_excursion", "FAILED (floating basis)" if fail_expected else "NOT failed (closed/eod basis)", res, res.failed == fail_expected)
    return out


def run_all() -> list:
    from app.prop.presets import list_presets
    rows: list[RulebookCase] = []
    for p in list_presets():
        rows.extend(run_rulebook_tests(p.key))
    return rows


if __name__ == "__main__":  # pragma: no cover
    rows = run_all()
    bad = [c for c in rows if not c.ok]
    for c in rows:
        print(("ok   " if c.ok else "FAIL ") + f"{c.preset_key:28s} {c.name:20s} expected: {c.expected:48s} got: {c.got}")
    print(f"\n{len(rows) - len(bad)}/{len(rows)} scripted-day checks agree with the declared rules")
    raise SystemExit(1 if bad else 0)
