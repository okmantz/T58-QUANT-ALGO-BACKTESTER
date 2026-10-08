"""v9.6 run-context hardening: every run type passes its RiskConfig
through build_run_context so a silent 'skip' sizing mode / legacy
account model can no longer produce results that are not comparable
across tabs, and the preflight block explains itself in dollars."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backtest.risk import RiskConfig, build_run_context, describe_simulation  # noqa: E402
from app.prop.simulator import PropRules  # noqa: E402


def _rules(account_size=50_000.0):
    return PropRules(
        account_size=account_size,
        evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=2.0,
        max_drawdown_pct=10.0,
    )


def _df(n=120):
    px = [100.0 + i * 0.01 for i in range(n)]
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": px, "high": [p + 0.5 for p in px], "low": [p - 0.5 for p in px],
        "close": px, "volume": [100] * n,
    })


# (a) sizing upgrade, with and without prop rules; explicit modes preserved
def test_build_run_context_upgrades_skip_without_prop_rules():
    risk = RiskConfig(initial_balance=10_000.0)  # sizing_mode defaults to "skip"
    out = build_run_context(risk)
    assert out is not risk
    assert out.sizing_mode == "fit_stop"
    assert out.account_model == "legacy"  # no prop rules = plain backtest, untouched
    assert risk.sizing_mode == "skip"  # caller's object never mutated


def test_build_run_context_upgrades_skip_with_prop_rules_and_preserves_fixed():
    out = build_run_context(RiskConfig(), _rules())
    assert out.sizing_mode == "fit_stop"

    fixed = RiskConfig(sizing_mode="fixed_contracts", fixed_contracts=2)
    out_fixed = build_run_context(fixed)
    assert out_fixed.sizing_mode == "fixed_contracts"
    assert out_fixed.fixed_contracts == 2
    assert fixed.sizing_mode == "fixed_contracts"

    micro = RiskConfig(sizing_mode="micro_fallback")
    assert build_run_context(micro, _rules()).sizing_mode == "micro_fallback"
    assert build_run_context(micro).sizing_mode == "micro_fallback"


def test_build_run_context_applies_instrument_spec():
    out = build_run_context(RiskConfig(pip_size=0.0001), instrument="ES")
    assert out.pip_size == 1.0
    assert out.contract_size == 50.0
    # Unknown instruments must not raise or invent a scale.
    out_unknown = build_run_context(RiskConfig(pip_size=0.01), instrument="NOT_A_REAL_SYMBOL_XYZ")
    assert out_unknown.pip_size == 0.01


# (b) prop rules: prop account model, intrabar on, balance synced
def test_build_run_context_with_prop_rules():
    risk = RiskConfig(initial_balance=10_000.0)
    out = build_run_context(risk, _rules(50_000.0))
    assert out.account_model == "prop"
    assert out.intrabar_replay is True
    assert out.initial_balance == 50_000.0
    assert risk.initial_balance == 10_000.0
    assert risk.account_model == "legacy"


# (c) wired entry points actually harden (captured via monkeypatched run_backtest)
def test_wired_entry_points_harden_risk(monkeypatch):
    captured: dict[str, list] = {"cpcv": [], "sensitivity": [], "risk_sweep": [], "regime": []}

    class _Stats:
        def to_dict(self):
            return {"profit_factor": 1.0, "net_profit": 0.0}

    def _fake_bt(name):
        def _run(df, strategy, risk, **kwargs):
            captured[name].append(risk)
            return SimpleNamespace(trades=[], statistics=_Stats(), equity_curve=pd.DataFrame({"timestamp": [], "equity": []}))
        return _run

    df = _df(120)

    # CPCV
    import app.validation.cpcv as cpcv_mod
    monkeypatch.setattr(cpcv_mod, "run_backtest", _fake_bt("cpcv"))
    cpcv_mod.run_cpcv(df, lambda: object(), RiskConfig(), n_groups=3, n_test_groups=1, max_paths=1, metric="profit_factor")
    assert captured["cpcv"] and all(r.sizing_mode != "skip" for r in captured["cpcv"])

    # Sensitivity (1D)
    import app.validation.sensitivity as sens_mod
    from app.monte_carlo.engine import MonteCarloConfig
    from app.strategy.manual import ManualStrategy

    monkeypatch.setattr(sens_mod, "run_backtest", _fake_bt("sensitivity"))
    config = {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow", "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow", "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20, "take_profit_pips": 40,
    }
    sens_mod.compute_1d_sensitivity(
        df, ManualStrategy(config), RiskConfig(), _rules(), MonteCarloConfig(n_simulations=10),
        metric="profit_factor", n_steps=3, max_params=1,
    )
    assert captured["sensitivity"] and all(r.sizing_mode != "skip" for r in captured["sensitivity"])

    # Risk Sweep
    import app.optimize.risk_sweep as sweep_mod
    monkeypatch.setattr(sweep_mod, "run_backtest", _fake_bt("risk_sweep"))
    sweep_mod.run_risk_sweep(df, lambda: object(), RiskConfig(), _rules(), risk_values=[1.0])
    assert captured["risk_sweep"] and all(r.sizing_mode != "skip" for r in captured["risk_sweep"])
    # ...and with prop rules the sweep's risk is a prop account, synced.
    assert captured["risk_sweep"][0].account_model == "prop"
    assert captured["risk_sweep"][0].initial_balance == 50_000.0

    # Regime Matrix
    import app.validation.regime_matrix as regime_mod
    monkeypatch.setattr(regime_mod, "run_backtest", _fake_bt("regime"))
    try:
        regime_mod.run_regime_matrix(df, object(), RiskConfig())
    except Exception:
        # build_regime_matrix may decline on tiny synthetic data; the
        # hardening assertion is about the risk that reached run_backtest.
        pass
    assert captured["regime"] and all(r.sizing_mode != "skip" for r in captured["regime"])


# (d) preflight sizing_skips explains itself in dollars, naming the micro
def test_preflight_sizing_skips_message_has_dollars_and_micro():
    from app.validation.preflight import baseline_preflight

    trades = [SimpleNamespace(direction=1, risk_at_stop_dollars=None) for _ in range(6)]
    trades += [SimpleNamespace(direction=-1, risk_at_stop_dollars=None) for _ in range(6)]
    eq = pd.DataFrame({"timestamp": [], "equity": []})
    eq.attrs["sizing_halt"] = {"skip_ratio": 0.35, "skipped": 7, "skip_reasons": {"stop_too_wide_for_budget": 7}}
    bt = SimpleNamespace(trades=trades, equity_curve=eq)
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=500, freq="1h")})
    risk = RiskConfig(
        initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0,
        pip_size=1.0, contract_size=50.0, spread_pips=1.0, slippage_pips=1.0,
        commission_per_contract=2.4,
    )
    rep = baseline_preflight(bt, df, risk, symbol="ES", stop_loss_pips=20, min_trades=1)
    blocks = [i for i in rep.issues if i.code == "sizing_skips"]
    assert blocks and blocks[0].severity == "block"
    msg = blocks[0].message
    assert "$" in msg
    assert "micro" in msg.lower()
    assert "would need" in msg  # stop is computable here, so the needed budget is stated


# (e) describe_simulation names the simulation choices
def test_describe_simulation_mentions_sizing_and_intrabar():
    text = describe_simulation(RiskConfig())
    assert "sizing" in text.lower()
    assert "intrabar" in text.lower()

    prop_text = describe_simulation(build_run_context(RiskConfig(), _rules()), _rules())
    assert "70%" in prop_text
    assert "100 trades" in prop_text
