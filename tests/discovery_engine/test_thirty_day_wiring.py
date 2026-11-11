"""2026-11-11: the 30-day prop evaluator (``evaluate_thirty_day``) is wired into the Discovery Engine's champion scoring."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.discovery_engine.engine import BackendOutput, DiscoveryEngine, EngineRequest
from app.discovery_engine import promotion as pr
from app.prop.presets import get_preset
from app.validation import trial_ledger as tl


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "reports" / "trial_ledger.json")


def _risk():
    return RiskConfig(initial_balance=50000, spread_pips=0.0, slippage_pips=0.0, commission_per_trade=0.0)


def _trades(n, mean_r, size, seed):
    from app.backtest.execution import Trade
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2022-01-03 15:00")
    out = []
    for i in range(n):
        r = float(mean_r + 0.8 * rng.standard_normal())
        d = t0 + pd.Timedelta(days=i)
        out.append(Trade(entry_time=d, exit_time=d + pd.Timedelta(minutes=30), direction=1, entry_price=100.0, exit_price=100.0 + r, size=size,
                         pnl=r * size, pnl_pct=r / 100.0, exit_reason="take_profit" if r > 0 else "stop_loss", commission=0.0,
                         equity_after=50000.0 + r * size, initial_risk=1.0))
    return out


def _df():
    ts = pd.date_range("2022-01-03", periods=1200, freq="D")
    c = np.linspace(100, 110, len(ts))
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0})


def _backend(specs):
    def backend(req, preset, families, df):
        return BackendOutput(native="x", champions=[{"spec": {"config": {"id": i}}, "family": "trend_breakout", "trades": t} for i, t in specs], evaluations=300)
    return backend


def _steps(**kw):
    return pr.Steps(**{n: (lambda c, ctx, n=n: pr.StepResult(n, True, detail="fake")) for n in pr.STEP_ORDER})


def _run(rules):
    specs = [("weak", _trades(300, 0.05, 150, 2)), ("good", _trades(300, 0.45, 150, 1))]
    eng = DiscoveryEngine(backends={"search": _backend(specs)}, steps_factory=_steps)
    return eng.run(EngineRequest(preset="search_lab", df=_df(), risk=_risk(), prop_rules=rules, families=["trend_breakout"], register_retests=False,
                                 sealed_frac=0.1))


def test_champions_carry_a_thirty_day_record_under_the_requests_prop_rules():
    res = _run(get_preset("lucid_50k").to_prop_rules())
    assert res.champions
    for c in res.champions:
        td = c["thirty_day"]
        assert td is not None and td["n_windows"] > 0 and 0.0 <= td["lower_bound"] <= td["p_pass_30"] <= 1.0
        assert td["target_probability"] == 0.65 and isinstance(td["meets_target_lower_bound"], bool)
    json.dumps(res.to_dict(), default=str)


def test_prop_mode_ranks_on_the_thirty_day_lower_bound():
    res = _run(get_preset("lucid_50k").to_prop_rules())
    lbs = [c["thirty_day"]["lower_bound"] for c in res.champions]
    assert lbs == sorted(lbs, reverse=True)


def test_without_prop_rules_the_record_is_absent_and_ranking_is_unchanged():
    res = _run(None)
    assert all(c["thirty_day"] is None for c in res.champions)
    assert [c["spec"]["config"]["id"] for c in res.champions][0] == "good"


def test_the_net_profit_objective_never_scores_prop_rules():
    specs = [("good", _trades(300, 0.45, 150, 1))]
    eng = DiscoveryEngine(backends={"search": _backend(specs)}, steps_factory=_steps)
    res = eng.run(EngineRequest(preset="search_lab", df=_df(), risk=_risk(), prop_rules=get_preset("lucid_50k").to_prop_rules(),
                                families=["trend_breakout"], register_retests=False, objective_mode="net_profit"))
    assert all(c["thirty_day"] is None for c in res.champions)


def test_the_cli_builds_the_real_lucid_preset_not_a_bare_PropRules():
    from scripts.run_discovery_engine import _prop
    rules, balance = _prop("lucid_50k")
    assert balance == 50000.0 and rules.account_size == 50000 and rules.payout_frequency_days == 5 and rules.max_contracts == 4
    with pytest.raises(SystemExit):
        _prop("no_such_firm")
