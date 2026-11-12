"""Replay window must not manufacture a '0% passed' that contradicts the Monte Carlo; Validate/Optimize reuse the saved setup."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.reports import basis


def _daily_df(n=1500):
    ts = pd.date_range("2020-01-01", periods=n, freq="1D")
    price = 5000.0 + np.cumsum(np.full(n, 0.35))
    return pd.DataFrame({"timestamp": ts, "open": price, "high": price + 0.6, "low": price - 0.6, "close": price + 0.2, "volume": 100.0})


def test_replay_window_widens_until_accounts_can_resolve(monkeypatch):
    from app.orchestration import full_pipeline as fp
    from app.strategy.manual import ManualStrategy
    import app.prop.attempt_replay as ar

    strat = ManualStrategy({
        "name": "sma cross",
        "indicators": [{"type": "sma", "period": 5, "column": "close", "as": "f"}, {"type": "sma", "period": 15, "column": "close", "as": "s"}],
        "long_entry": "f > s", "long_exit": "f < s", "short_entry": "f < s", "short_exit": "f > s",
        "stop_loss_pips": 20, "take_profit_pips": 40,
    })
    df = _daily_df()
    risk = RiskConfig(initial_balance=50_000.0, pip_size=1.0, contract_size=50.0)
    bt = run_backtest(df, strat, risk, lookahead_check="skip")
    rules = replace(PropRules(account_size=50_000.0, evaluation_profit_target_pct=6.0, daily_loss_limit_pct=2.0, max_drawdown_pct=10.0),
                    max_eval_calendar_days=None)
    horizons = []

    def fake_replay(*a, horizon_days=None, min_horizon_coverage=0.25, **kw):
        hz = int(horizon_days or 120)
        horizons.append((hz, min_horizon_coverage))
        open_ = 48 if hz < 500 else 2          # accounts need ~500 days to reach the target
        return SimpleNamespace(horizon_days=hz, n_attempts=48, n_open=open_, n_passed=48 - open_, n_failed=0,
                               pass_rate=(48 - open_) / 48, sample_ok=True, render=lambda: "fake replay")

    monkeypatch.setattr(ar, "run_attempt_replay", fake_replay)
    cfg = SimpleNamespace(extra_gates_enabled=True, attempt_replay_starts=48, null_n_seeds=2, break_cost_mult=2.0,
                          replay_window_multiple=2.0, replay_horizon_cap_days=365)
    replay, _null, _cs = fp._replay_and_null_evidence(strat, df, risk, rules, bt, cfg, lambda m: None, run_null=False)
    assert replay.n_open / replay.n_attempts <= 0.30, "the replay kept a window too short to ever pass"
    assert horizons[0][0] == 120 and horizons[-1][0] >= 500
    assert horizons[-1][1] == 1.0          # widened runs count only attempts whose whole window was observed


def test_reproducible_engine_requires_pipeline_basis_and_same_preset():
    eng = {"risk_value": 1.0, "sizing_mode": "rr_planned", "planned_target_dollars": 375.0, "mc": {"sims": 2000, "seed": 42}}
    b = basis.make_basis("full_pipeline", prop_preset="topstep_100k", risk_pct=0.19, engine=eng)
    assert basis.reproducible_engine({"basis": b}, "topstep_100k")["engine"]["planned_target_dollars"] == 375.0
    assert basis.reproducible_engine({"basis": b}, "apex_50k_eod") is None
    assert basis.reproducible_engine({"basis": basis.make_basis("validate", risk_pct=1)}, "") is None
    assert basis.reproducible_engine({}, "") is None


def test_annotate_matching_numbers_says_engines_agree():
    saved = {"eval_pass_probability": 97.7, "basis": basis.make_basis("full_pipeline", prop_preset="p", risk_pct=0.19)}
    item = {"detail": "", "metrics": [{"label": "Eval pass chance", "value": "97.7%"}], "guidance": {"steps": []}}
    basis.annotate_mc_item(item, basis.make_basis("validate", prop_preset="p", risk_pct=0.19), saved)
    assert "engines agree" in item["detail"]


def test_saved_setup_rebuilds_the_pipeline_risk_and_mc_config(monkeypatch):
    from app.web import server
    eng = basis.engine_from_run(
        RiskConfig(risk_mode="percent", risk_value=1.0, sizing_mode="rr_planned", planned_target_dollars=375.0, initial_balance=100000.0),
        adaptive={"enabled": False}, mc={"sims": 2000, "seed": 42, "reset_on_breach": True}, holdout_frac=0.2)
    b = basis.make_basis("full_pipeline", prop_preset="topstep_100k", risk_pct=0.188, sizing_mode="rr_planned", engine=eng)
    monkeypatch.setattr(server, "load_strategy_metadata", lambda *a, **k: {"last_run": {"basis": b, "eval_pass_probability": 97.7}})
    rules = SimpleNamespace(preset_key="topstep_100k", account_size=100000.0)
    risk, setup = server._vs_saved_setup({"prop_preset": "topstep_100k"}, ("python", "x.py"), rules, RiskConfig())
    assert risk.sizing_mode == "rr_planned" and risk.planned_target_dollars == 375.0 and risk.reset_on_breach is True
    assert setup["mc_cfg"].n_simulations == 2000 and setup["mc_cfg"].random_seed == 42 and setup["holdout_frac"] == 0.2
    base = RiskConfig()
    risk2, none_setup = server._vs_saved_setup({"prop_preset": "topstep_100k", "setup_mode": "custom"}, ("python", "x.py"), rules, base)
    assert none_setup is None and risk2 is base
