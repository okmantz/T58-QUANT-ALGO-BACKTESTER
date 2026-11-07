"""Net Profit Tester (live accounts): verdicts, cost stress, resampling sanity."""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from app.validation.net_profit_tester import (
    VERDICT_FAIL, VERDICT_INSUFFICIENT, VERDICT_MARGINAL, VERDICT_PASS, NetProfitTestConfig, run_net_profit_test,
)


def _trades(n=300, win=0.55, w=120.0, l=-100.0, comm=2.0, seed=3, per_day=1):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-01-06 10:00")
    out = []
    for i in range(n):
        day = t0 + pd.Timedelta(days=i // per_day)
        out.append(NS(pnl=float(w if rng.random() < win else l), entry_time=day + pd.Timedelta(minutes=i % per_day), commission=comm))
    return out


CFG = NetProfitTestConfig(n_simulations=400, starting_equity=10_000.0)


def test_good_strategy_passes_and_reports_every_section():
    r = run_net_profit_test(_trades(), CFG)
    assert r.verdict == VERDICT_PASS, r.reasons
    for key in ("n_trades", "net_profit", "profit_factor", "max_drawdown_pct", "win_rate_pct"):
        assert key in r.baseline
    assert r.edge["t_stat"] > 1.5 and r.edge["ci_excludes_zero"]
    assert r.monte_carlo["p_profit_pct"] > 75 and r.monte_carlo["n_simulations"] == 400
    assert set(r.cost_stress) == {1.0, 1.5, 2.0}
    assert r.stability["n_months"] >= 6
    assert "PASS" in r.headline()


def test_losing_strategy_fails():
    r = run_net_profit_test(_trades(win=0.35), CFG)
    assert r.verdict == VERDICT_FAIL and any("net profit" in x for x in r.reasons)


def test_too_few_trades_is_insufficient_not_a_pass():
    r = run_net_profit_test(_trades(n=40), CFG)
    assert r.verdict in (VERDICT_INSUFFICIENT, VERDICT_FAIL)
    assert any("too few" in x for x in r.reasons)


def test_edge_that_vanishes_at_1_5x_costs_is_rejected():
    # tiny edge, big commission: net positive at 1x, negative once costs are scaled
    r = run_net_profit_test(_trades(w=101.0, l=-100.0, win=0.5, comm=3.0, n=400), NetProfitTestConfig(n_simulations=200))
    assert r.cost_stress[1.5]["net_profit"] < r.cost_stress[1.0]["net_profit"]
    if r.baseline["net_profit"] > 0 and r.cost_stress[1.5]["net_profit"] <= 0:
        assert r.verdict == VERDICT_FAIL


def test_restress_callback_is_used_when_provided():
    base = _trades()
    called = []

    def restress(m):
        called.append(m)
        return [NS(pnl=t.pnl - (m - 1) * 10.0, entry_time=t.entry_time, commission=t.commission) for t in base]

    r = run_net_profit_test(base, CFG, restress=restress)
    assert sorted(called) == [1.5, 2.0]
    assert "re-run backtest" in r.cost_stress[1.5]["method"]
    assert r.cost_stress[2.0]["net_profit"] == pytest.approx(sum(t.pnl for t in base) - 10.0 * len(base))


def test_same_seed_reproduces_and_empty_input_is_handled():
    a = run_net_profit_test(_trades(), CFG).to_dict()["monte_carlo"]
    b = run_net_profit_test(_trades(), CFG).to_dict()["monte_carlo"]
    assert a == b
    assert run_net_profit_test([], CFG).verdict == VERDICT_INSUFFICIENT


def test_drawdown_tolerance_is_the_traders_own_not_a_firm_rule():
    wild = run_net_profit_test(_trades(w=900.0, l=-700.0, win=0.5, n=300), NetProfitTestConfig(n_simulations=300, max_p95_drawdown_pct=5.0))
    assert any("tolerance" in x for x in wild.reasons)
