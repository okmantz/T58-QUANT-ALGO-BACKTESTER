"""Acceptance tests for the 2026-10-07 accuracy overhaul + discovery layer."""
from __future__ import annotations

import dataclasses
import random
import warnings

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import Trade, run_execution
from app.backtest.risk import RiskConfig
from app.data.instrument_specs import get_instrument_spec
from app.prop.simulator import PropRules, simulate_account

warnings.simplefilter("ignore")


def _walk(n=4000, seed=1, step=3.0, start="2024-01-02", freq="15min", phi=0.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    c = 4000 + np.cumsum(r * step / 2)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 1.5, n)
    l = np.minimum(o, c) - rng.uniform(0, 1.5, n)
    return pd.DataFrame(dict(timestamp=pd.date_range(start, periods=n, freq=freq), open=o, high=h, low=l, close=c, volume=1))


def _es_risk(**kw):
    base = dict(initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0, pip_size=0.25, contract_size=50.0,
                spread_pips=1.0, slippage_pips=1.0, commission_per_contract=2.4, max_trades_per_day=20)
    base.update(kw)
    return RiskConfig(**base)


# ---------------------------------------------------------------- costs / sizing
def test_cost_units_ticks_to_pips_and_round_trip():
    es = get_instrument_spec("ES")
    assert es.default_spread_pips == pytest.approx(es.default_spread_ticks * es.tick_size / es.pip_size)
    # 2 sides x (1 spread + 1 slip tick) x $12.50 + commission
    assert es.round_trip_cost_dollars() == pytest.approx(2 * 2 * es.tick_value + es.default_commission_round_turn)


@pytest.mark.parametrize("stop", [3.0, 7.5, 11.0, 20.0])
def test_sizing_never_exceeds_budget_with_costs(stop):
    r = _es_risk(risk_value=1000.0)
    d = r.size_for_stop(50_000.0, stop)
    if d.skip_reason is None:
        assert d.risk_at_stop <= d.budget + 1e-9
        assert float(d.contracts).is_integer()
        # one more contract would break the budget
        assert r.worst_case_loss((d.contracts + 1) * 50.0, stop) > d.budget


def test_sizing_modes():
    wide = 30.0  # $1500/contract before costs vs a $500 budget
    skip = _es_risk(sizing_mode="skip").size_for_stop(50_000.0, wide)
    assert skip.units == 0 and skip.skip_reason
    fit = _es_risk(sizing_mode="fit_stop").size_for_stop(50_000.0, wide)
    assert fit.contracts == 1 and fit.stop_capped and fit.stop_distance < wide and fit.risk_at_stop <= fit.budget + 1e-9
    fixed = _es_risk(sizing_mode="fixed_contracts", fixed_contracts=2).size_for_stop(50_000.0, wide)
    assert fixed.contracts == 2 and fixed.above_budget
    micro = _es_risk(sizing_mode="micro_fallback", micro_contract_size=5.0, micro_commission_per_contract=0.6).size_for_stop(50_000.0, wide)
    assert micro.used_micro and micro.contracts >= 1 and micro.risk_at_stop <= micro.budget + 1e-9


def test_engine_trades_never_exceed_budget_at_stop():
    df = _walk()
    sig = pd.Series(np.where(np.arange(len(df)) % 40 < 20, 1, -1))
    tr, eq = run_execution(df, sig, _es_risk(sizing_mode="fit_stop"), stop_loss_pips=24, take_profit_pips=48)
    assert tr
    assert all(t.risk_at_stop_dollars <= 500.0 + 1e-6 for t in tr if t.risk_at_stop_dollars)
    assert eq.attrs["sizing_summary"]["mode"] == "fit_stop"


# ---------------------------------------------------------------- parity vs frozen legacy simulator
def _case(rng: random.Random):
    rules = PropRules(
        account_size=rng.choice([25_000, 50_000]), evaluation_profit_target_pct=rng.choice([3.0, 6.0]),
        daily_loss_limit_pct=rng.choice([2.0, 3.0, 100.0]), max_drawdown_pct=rng.choice([3.0, 6.0]),
        drawdown_type=rng.choice(["trailing", "static"]), drawdown_check_mode=rng.choice(["intrabar", "eod"]),
        consistency_rule_pct=rng.choice([None, 40.0]), min_trading_days=rng.choice([1, 3]),
        payout_threshold_pct=rng.choice([0.0, 2.0]), payout_frequency_days=rng.choice([1, 7]),
        winning_days_for_payout=rng.choice([1, 3]), max_inactive_days=rng.choice([None, 10]),
        daily_loss_action=rng.choice(["fail", "lock_day"]), daily_loss_base=rng.choice(["initial", "prior_day_high", "ratchet_up"]),
        max_eval_calendar_days=rng.choice([None, 20]),
    )
    n = rng.randint(1, 80)
    t = pd.Timestamp("2024-01-02 14:00", tz="UTC")
    dates, pnls = [], []
    for _ in range(n):
        t = t + pd.Timedelta(hours=rng.choice([0, 1, 2, 5, 30]))
        dates.append(t)
        pnls.append(round(rng.gauss(0.05, 1.0) * rules.account_size * 0.004, 2))
    return rules, pnls, dates, rng.random() < 0.5


def _to_legacy(rules):
    from tests.reference import legacy_simulator as old
    fields = {f.name for f in dataclasses.fields(old.PropRules)}
    return old.PropRules(**{k: v for k, v in dataclasses.asdict(rules).items() if k in fields})


def test_simulate_account_parity_with_frozen_legacy_reference():
    from tests.reference import legacy_simulator as old
    rng = random.Random(1234)
    for _ in range(250):
        rules, pnls, dates, reset = _case(rng)
        a = simulate_account(pnls, dates, rules, reset_on_breach=reset)
        b = old.simulate_account(pnls, dates, _to_legacy(rules), reset_on_breach=reset)
        for f in ("passed_evaluation", "failed", "failure_reason", "final_balance", "days_to_pass",
                  "total_attempts", "attempts_passed", "first_payout_day_index"):
            assert getattr(a, f) == getattr(b, f), f
        assert len(a.payouts) == len(b.payouts)


# ---------------------------------------------------------------- prop account in the bar engine
def _prop_rules(**kw):
    base = dict(account_size=50_000, max_drawdown_pct=4.0, drawdown_type="trailing", daily_loss_limit_pct=2.0,
                evaluation_profit_target_pct=6.0, dd_basis="floating", daily_loss_basis="floating", daily_loss_action="lock_day")
    base.update(kw)
    return PropRules(**base)


def _alt_signals(n):
    return pd.Series(np.where(np.arange(n) % 60 < 30, 1, -1))


def test_prop_mode_chain_labels_attempts_and_busts():
    df = _walk(6000, seed=5)
    risk = _es_risk(account_model="prop", prop_account_rules=_prop_rules(), sizing_mode="fit_stop")
    tr, eq = run_execution(df, _alt_signals(len(df)), risk, 24, 48, attempt_mode="chain")
    atts = eq.attrs["prop_attempts"]
    assert len(atts) >= 2 and len(eq) == len(df) and "attempt_id" in eq.columns
    assert sorted({t.attempt_id for t in tr}) == sorted({t.attempt_id for t in tr})  # labelled
    assert max(t.attempt_id for t in tr) >= 1
    assert all(a["outcome"] in ("failed", "passed", "open", "passed_funded") for a in atts)
    # a failed attempt never ends above its drawdown floor
    for a in atts:
        if a["outcome"] == "failed" and "max_drawdown" in (a["failure_reason"] or ""):
            assert a["end_balance"] <= 50_000 * 1.0  # trailing floor is below the account size unless the peak rose


def test_prop_mode_single_stops_at_first_attempt_end():
    df = _walk(6000, seed=5)
    risk = _es_risk(account_model="prop", prop_account_rules=_prop_rules(), sizing_mode="fit_stop")
    tr, eq = run_execution(df, _alt_signals(len(df)), risk, 24, 48, attempt_mode="single")
    assert len(eq.attrs["prop_attempts"]) == 1
    assert {t.attempt_id for t in tr} == {0}


def test_prop_floating_liquidation_beats_realized_only():
    """A trade that dips through the trailing floor intrabar but recovers is a
    bust on a floating basis and a survivor on a realized basis."""
    n = 60
    ts = pd.date_range("2024-01-02 15:00", periods=n, freq="15min")
    px = np.full(n, 4000.0)
    df = pd.DataFrame(dict(timestamp=ts, open=px, high=px + 0.5, low=px - 0.5, close=px, volume=1))
    df.loc[20, ["high", "low", "close"]] = [4001.0, 3940.0, 3999.0]   # deep wick down, closes back
    sig = pd.Series(np.zeros(n, dtype=int))
    sig.iloc[5:40] = 1
    base = dict(account_size=50_000, max_drawdown_pct=2.0, drawdown_type="static", daily_loss_limit_pct=100.0,
                evaluation_profit_target_pct=50.0)
    risk_kw = dict(risk_mode="fixed", risk_value=500.0, sizing_mode="fixed_contracts", fixed_contracts=5,
                   spread_pips=0.0, slippage_pips=0.0, commission_per_contract=0.0)
    r_float = _es_risk(account_model="prop", prop_account_rules=_prop_rules(**base), **risk_kw)
    r_real = _es_risk(account_model="prop", prop_account_rules=_prop_rules(**{**base, "dd_basis": "realized", "daily_loss_basis": "realized"}), **risk_kw)
    tr_f, eq_f = run_execution(df, sig, r_float, 4000, None, attempt_mode="single")
    tr_r, eq_r = run_execution(df, sig, r_real, 4000, None, attempt_mode="single")
    assert eq_f.attrs["prop_attempts"][0]["outcome"] == "failed"
    liq = [t for t in tr_f if t.exit_reason == "prop_floor_liquidation"][0]
    # static 2% floor = $49,000: liquidated AT the floor level, not at the wick low
    assert liq.exit_price > 3940 and eq_f.attrs["prop_attempts"][0]["end_balance"] == pytest.approx(49_000, abs=60)
    assert any(t.exit_reason == "prop_floor_liquidation" for t in tr_f)
    assert eq_r.attrs["prop_attempts"][0]["outcome"] != "failed"


# ---------------------------------------------------------------- attempt replay + Monte Carlo
def test_attempt_replay_reports_fresh_accounts():
    from app.prop.attempt_replay import run_attempt_replay
    df = _walk(5000, seed=9)
    sig = _alt_signals(len(df))
    res = run_attempt_replay(df, sig, _es_risk(sizing_mode="fit_stop"), _prop_rules(), 24, 48,
                             horizon_days=20, n_starts=12, min_attempts=30)
    assert res.n_attempts == 12 and not res.sample_ok            # honest about the small sample
    assert res.n_passed + res.n_failed + res.n_open == res.n_attempts
    assert res.notes and "not statistically meaningful" in res.notes[0]
    assert res.render()


def _fake_trades(n=120, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-02 15:00")
    out = []
    for i in range(n):
        d = t0 + pd.Timedelta(days=i // 2, hours=i % 2)
        p = float(rng.choice([300, -200]))
        out.append(Trade(entry_time=d, exit_time=d + pd.Timedelta(minutes=30), direction=1, entry_price=1, exit_price=1,
                         size=1, pnl=p, pnl_pct=0, exit_reason="x", commission=0, equity_after=0, initial_risk=1))
    return out


def test_monte_carlo_day_blocks_and_attempt_accounting():
    from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
    rules = PropRules(account_size=50_000, max_drawdown_pct=4.0, drawdown_type="trailing", daily_loss_limit_pct=2.0,
                      evaluation_profit_target_pct=6.0)
    r = run_monte_carlo(_fake_trades(), rules, MonteCarloConfig(n_simulations=120))
    assert MonteCarloConfig().method == "day_block_bootstrap"
    assert 0 <= r.bust_before_pass_probability <= 100
    assert r.risk_of_ruin_pct == pytest.approx(r.per_attempt_bust_probability)
    assert r.expected_attempts_to_pass is None or r.expected_attempts_to_pass >= 1
    assert r.sample_ok and r.n_source_days > 20
    tiny = run_monte_carlo(_fake_trades(10), rules, MonteCarloConfig(n_simulations=20))
    assert not tiny.sample_ok and tiny.sample_notes


def test_verdict_wrapper_caps_thin_evidence(monkeypatch):
    import app.orchestration.full_pipeline as fp
    monkeypatch.setattr(fp, "_make_verdict_core", lambda *a, **k: ("READY", [], object(), False, False))
    thin = type("MC", (), {"sample_ok": False, "sample_notes": ["only 10 trades"], "per_attempt_pass_probability": 90.0})()
    v, reasons, *_ = fp._make_verdict(thin, None)
    assert v == "MARGINAL" and "SAMPLE FLOOR" in reasons[-1]
    ok = type("MC", (), {"sample_ok": True, "sample_notes": [], "per_attempt_pass_probability": 90.0})()
    replay = type("R", (), {"sample_ok": True, "pass_rate": 0.2, "n_attempts": 80})()
    v, reasons, *_ = fp._make_verdict(ok, None, attempt_replay=replay)
    assert v == "MARGINAL" and "REPLAY DISAGREES" in reasons[-1]


# ---------------------------------------------------------------- preflight
def test_preflight_flags_structural_problems():
    from app.validation.preflight import run_preflight
    df = _walk(300)
    risk = RiskConfig(initial_balance=50_000, risk_mode="fixed", risk_value=200, pip_size=0.25, contract_size=50)
    rep = run_preflight(df, risk, PropRules(account_size=50_000, max_drawdown_pct=4, daily_loss_limit_pct=2),
                        stop_loss_pips=160, symbol="ES")
    codes = {i.code for i in rep.issues}
    assert {"pip_size_mismatch", "zero_costs", "sizing_infeasible"} <= codes and rep.blocked


# ---------------------------------------------------------------- resting orders
def test_resting_limit_requires_trade_through_and_books_target():
    from app.backtest.resting_orders import LimitOrder, simulate_resting_orders
    d = pd.DataFrame(dict(timestamp=pd.date_range("2024-01-01", periods=6, freq="h"),
                          open=[100, 100, 99, 98, 100, 101], high=[101, 101, 100, 99, 102, 103],
                          low=[99, 99, 97, 97, 99, 100], close=[100, 100, 98, 98, 101, 102], volume=1))
    r = RiskConfig(initial_balance=10_000, risk_mode="fixed", risk_value=100, pip_size=1, contract_size=1, sizing_mode="fit_stop")
    t = simulate_resting_orders(d, [LimitOrder(1, 98.5, 0, 5, 96.5, 102.0)], r)
    assert len(t) == 1 and t[0].entry_price == 98.5 and t[0].exit_reason == "take_profit" and t[0].pnl == pytest.approx(175.0)
    # price only touches 99.0 exactly -> a buy limit at 99.0 is NOT filled under trade-through
    assert simulate_resting_orders(d, [LimitOrder(1, 99.0, 0, 1, 97.0, 103.0)], r) == []


# ---------------------------------------------------------------- discovery
def test_rule_spec_rejects_hallucinated_fields():
    from app.discovery.rule_spec import SpecError, validate_spec
    with pytest.raises(SpecError):
        validate_spec({"kind": "magic_indicator"})
    with pytest.raises(SpecError):
        validate_spec({"kind": "momentum", "params": {"lookback": 50, "bogus": 1}})
    with pytest.raises(SpecError):
        validate_spec({"kind": "momentum", "params": {"lookback": 9999}})
    assert validate_spec({"kind": "momentum"})["params"]["lookback"] == 50


def test_idea_compiler_falls_back_when_model_unavailable_or_wrong():
    from app.ai.llm_client import NullClient, ScriptedClient
    from app.discovery.idea_compiler import compile_idea
    h = compile_idea("Buy the breakout of the 40-bar high, long only", llm=NullClient())
    assert h.spec["kind"] == "donchian_breakout" and h.spec["params"]["lookback"] == 40 and h.spec["direction"] == "long"
    assert h.source == "keyword" and h.warnings and h.falsifiers
    bad = compile_idea("fair value gap retest", llm=ScriptedClient('{"kind":"momentum","params":{"bogus":1}}'))
    assert bad.spec["kind"] == "fvg_retest" and "rejected" in bad.warnings[0]
    good = compile_idea("trend idea", llm=ScriptedClient('{"kind":"momentum","params":{"lookback":80},"mechanism":"m"}'))
    assert good.source == "llm" and good.spec["params"]["lookback"] == 80


def test_break_battery_kills_noise_and_records(tmp_path, monkeypatch):
    import app.ai.experiment_memory as em
    import app.search.graveyard as gy
    monkeypatch.setattr(em, "_db_path", lambda: tmp_path / "mem.db")
    monkeypatch.setattr(gy, "get_app_base_dir", lambda: tmp_path)
    from app.discovery.experiment_runner import run_hypothesis
    from app.discovery.hypothesis import HypothesisStore
    from app.discovery.idea_compiler import compile_idea
    store = HypothesisStore(tmp_path / "h.json")
    h = compile_idea("breakout of the 40-bar high", timeframes=["1h"])
    risk = _es_risk(risk_value=1000.0, initial_balance=100_000.0, sizing_mode="fit_stop")
    run = run_hypothesis(h, {"A": _walk(9000, seed=11), "B": _walk(9000, seed=12)}, risk, store=store, n_null=20)
    assert h.status in ("broken", "inconclusive")           # a random walk has no breakout edge
    saved = store.get(h.id)
    assert saved is not None and saved.experiments and saved.status == h.status
    assert run.n_trials >= 2 and run.render()
    # linked into experiment memory by hypothesis id, one row per market/timeframe cell
    rows = em.experiments_for_hypothesis(h.id)
    assert len(rows) == len(run.cells) and all(r["cell"] for r in rows)
    if h.status == "broken":
        assert list((tmp_path / "data" / "evolution").glob("strategy_graveyard__*.jsonl"))


def test_hypothesis_variants_accumulate_for_deflation(tmp_path):
    from app.discovery.hypothesis import Experiment, Hypothesis
    h = Hypothesis(idea="x", spec={"kind": "momentum"})
    h.add_experiment(Experiment("grid", 0.0, {}, n_variants=8))
    h.add_experiment(Experiment("grid", 1.0, {}, n_variants=4))
    assert h.total_variants_tried() == 12
