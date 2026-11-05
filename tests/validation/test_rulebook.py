"""The Edge Rulebook's published numbers, reproduced by app.validation.rulebook."""
import statistics
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from app.validation import rulebook as R


def test_z_bar_table():
    assert [round(R.z_bar(n), 2) for n in (1, 5, 10, 25, 50)] == [1.96, 2.58, 2.81, 3.09, 3.29]


def test_trades_needed_table_within_one_trade_of_rulebook():
    book = {1: 271, 5: 807, 10: 1038, 25: 1328, 50: 1539}
    for n, want in book.items():
        assert abs(R.trades_needed(0.10, n) - want) <= 1, n


def test_required_lift_table():
    assert [r["lift_pts"] for r in R.lift_table()] == [10.0, 6.7, 5.0, 4.0]


def test_edge_and_breakeven():
    assert R.edge_per_trade(0.40, 2, 0.05) == pytest.approx(0.15)
    assert R.breakeven_win_rate(2, 0.05) == pytest.approx(0.35)


def test_effective_bets_examples():
    assert round(R.effective_bets(5, 0.3), 1) == 2.3
    assert round(R.effective_bets(4, 0.8), 1) == 1.2


def test_frame_blocks_high_lift_and_missing_mechanism():
    ok = R.Frame(mechanism="yields lead equity futures", sign="+", payoff_b=2).check()
    assert ok["passed"]
    bad = R.Frame(mechanism="", sign="", payoff_b=1).check()
    assert not bad["passed"] and len(bad["problems"]) == 3


def test_llr_kill_dead_edge_killed_healthy_mostly_survives():
    rng = np.random.default_rng(5)
    w, l = 1.95, -1.05

    def kill_at(edge):
        p = (edge - l) / (w - l)
        k = R.LLRKillSwitch()
        for v in np.where(rng.random(3000) < p, w, l):
            k.add(v)
            if k.killed_at_trade:
                break
        return k.killed_at_trade

    dead = [kill_at(0.0) for _ in range(300)]
    assert sum(d is not None for d in dead) / 300 > 0.95
    assert 220 <= statistics.median([d for d in dead if d]) <= 400       # Rulebook: median 309
    healthy = [kill_at(0.15) for _ in range(300)]
    assert sum(h is not None for h in healthy) / 300 < 0.25               # Rulebook: 14% of 3,000-trade paths


def test_sprt_monitor_surfaces_rulebook_kill():
    from app.forward_test.sprt_monitor import SprtMonitor
    m = SprtMonitor(h1_r=0.15)
    st = None
    for _ in range(700):
        st = m.add(-0.4)
    assert st["rulebook_llr"]["killed"] and st["halted"]


def test_alpha_ledger_budget_and_refusal():
    rows = []
    d = date(2026, 11, 3)
    for i in range(10):
        r = R.add_hypothesis(f"idea {i}", "next 30 min", today=d, ledger_rows=rows, persist=False)
        assert r["ok"]
    assert rows[-1]["z_bar"] == 2.81 and rows[-1]["id"] == "H10"
    assert not R.add_hypothesis("one too many", "x", today=d, ledger_rows=rows, persist=False)["ok"]
    # a new quarter has a fresh budget
    assert R.add_hypothesis("later", "x", today=date(2027, 1, 5), ledger_rows=rows, persist=False)["ok"]


def _trade(direction, entry, exit_, pnl, risk=100.0):
    return SimpleNamespace(direction=direction, entry_price=entry, exit_price=exit_, pnl=pnl, intended_risk_dollars=risk, size=1, initial_risk=1)


def test_binary_screen_one_sided_is_not_testable_never_a_pass():
    ts = [_trade(1, 100, 101, 50) for _ in range(60)]
    out = R.binary_screen(ts)
    assert out["testable"] is False and out["passed"] is False and "one side" in out["reason"]


def test_binary_screen_real_lift_passes_and_noise_fails():
    rng = np.random.default_rng(2)
    ts = []
    for i in range(800):
        side = 1 if i % 2 == 0 else -1
        up = rng.random() < (0.62 if side == 1 else 0.38)        # the signal really predicts direction
        move = 1.0 if up else -1.0
        pnl = side * move * 100 * (2 if side * move > 0 else 1) * (1 if side * move > 0 else 1) - (5 if True else 0)
        ts.append(_trade(side, 100, 100 + move, pnl))
    good = R.binary_screen(ts, n_looks=1)
    assert good["testable"] and good["lift"] > 0.15
    noise = []
    for i in range(800):
        side = 1 if i % 2 == 0 else -1
        move = 1.0 if rng.random() < 0.5 else -1.0
        noise.append(_trade(side, 100, 100 + move, side * move * 100 - 5))
    assert not R.binary_screen(noise, n_looks=10)["passed"]
