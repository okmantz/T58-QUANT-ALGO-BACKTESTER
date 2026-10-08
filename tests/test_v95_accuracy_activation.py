"""v9.5 accuracy activation: the Oct 2026 audit found the corrected
machinery (unified PropAccount, fit_stop sizing, roll adjustment,
per-attempt Monte Carlo, discovery grid + break battery) implemented
but not switched on. These tests pin the ACTIVATION, not just the
parts:

* every prop workflow's shared chokepoint (with_prop_safety_defaults)
  must hand the engine prop account semantics + "risk UP TO the
  budget" sizing (fit_stop) -- Owen's screenshot trade ($450 risk,
  1 ES contract, stop capped to the budget) must trade instead of
  sizing to zero;
* Owen's account lifecycle example must score exactly as he stated it:
  pass + funded blow, then pass + payout = 100% eval, 50% first
  payout, with the chain running to the end of the dataset;
* continuous futures are back-adjusted at import;
* no preset may ride the silent "legacy" rule basis;
* a manual strategy file missing its .json extension still appears
  in the library (the "one ES strategy doesn't show up" bug);
* the plain refinement path applies the GA reliability adjustments;
* the discovery grid tests parameter variants and counts every run;
* Full Pipeline's evidence step returns the 2x cost-stress break-it
  result the verdict gates on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backtest.engine import run_backtest  # noqa: E402
from app.backtest.risk import RiskConfig, with_prop_safety_defaults  # noqa: E402
from app.prop.simulator import PropRules, simulate_account  # noqa: E402


def _dates(n: int):
    start = pd.Timestamp("2024-01-01")
    return [start + pd.Timedelta(days=i) for i in range(n)]


def _owen_rules() -> PropRules:
    """50k account: +$3,000 (6%) passes the eval; a -$2,000 day breaks
    the $1,000 (2%) daily loss limit and blows the funded account;
    first payout at +$2,000 (4%) of funded profit, one winning day."""
    return PropRules(
        account_size=50_000.0,
        evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=2.0,
        max_drawdown_pct=10.0,
        drawdown_type="static",
        drawdown_check_mode="eod",
        consistency_rule_pct=None,
        funded_consistency_rule_pct=None,
        min_trading_days=1,
        payout_threshold_pct=4.0,
        payout_frequency_days=1,
        winning_days_for_payout=1,
        min_winning_day_profit=0.0,
    )


# ---------------------------------------------------------------------------
# Owen's lifecycle example, exactly as he stated it (2026-10-08)
# ---------------------------------------------------------------------------

def test_owen_lifecycle_example_scores_100_and_50():
    # Buy 50k, +$3,000 passes; funded -$2,000 blows it. Buy another,
    # pass, funded +$2,000 pays out.
    pnls = [3000.0, -2000.0, 3000.0, 2000.0]
    result = simulate_account(pnls, _dates(len(pnls)), _owen_rules(), reset_on_breach=True)

    assert result.total_attempts == 2
    assert result.attempts_passed == 2
    assert result.attempts_reached_payout == 1

    eval_pct = result.attempts_passed / result.total_attempts * 100
    payout_pct = result.attempts_reached_payout / result.attempts_passed * 100
    assert eval_pct == 100.0
    assert payout_pct == 50.0

    first, second = result.attempts
    assert first.passed_evaluation and first.failed and not first.reached_first_payout
    assert second.passed_evaluation and second.reached_first_payout
    assert second.payout_amount == pytest.approx(2000.0)


def test_lifecycle_chain_runs_to_end_of_dataset():
    # After the payout the chain keeps trading: a later bust starts a
    # fresh eval, and the last attempt runs until the trades run out.
    pnls = [3000.0, -2000.0, 3000.0, 2000.0, -6000.0, 3000.0, 500.0]
    result = simulate_account(pnls, _dates(len(pnls)), _owen_rules(), reset_on_breach=True)

    assert result.total_attempts == 3
    assert result.attempts_passed == 3
    assert result.attempts_reached_payout == 1
    # attempts tile the whole trade sequence with no gap and no overlap
    assert result.attempts[0].start_day_index == 0
    for prev, nxt in zip(result.attempts, result.attempts[1:]):
        assert nxt.start_day_index == prev.end_day_index + 1
    assert result.attempts[-1].end_day_index == len(pnls) - 1


# ---------------------------------------------------------------------------
# Prop workflows actually run under prop semantics ("risk UP TO" sizing)
# ---------------------------------------------------------------------------

def test_prop_safety_defaults_activate_prop_account_and_fit_stop():
    risk = with_prop_safety_defaults(RiskConfig(), _owen_rules())
    assert risk.account_model == "prop"
    assert risk.prop_account_rules is not None
    assert risk.sizing_mode == "fit_stop"
    assert risk.intrabar_replay is True
    assert risk.initial_balance == 50_000.0


def test_prop_safety_defaults_preserve_explicit_sizing_choice():
    risk = with_prop_safety_defaults(
        RiskConfig(sizing_mode="fixed_contracts", fixed_contracts=2), _owen_rules()
    )
    assert risk.sizing_mode == "fixed_contracts"
    risk = with_prop_safety_defaults(RiskConfig(sizing_mode="micro_fallback"), _owen_rules())
    assert risk.sizing_mode == "micro_fallback"


class _OneShotStrategy:
    """One long trade (bars 10-30) with a fixed 12-point stop/target --
    wider than a $450 budget at $50/point ($600), like a strategy
    whose own stop does not fit the account's per-trade risk."""

    source_type = "python"
    name = "one-shot"

    def generate(self, df):
        from app.strategy.base import StrategyResult

        signals = pd.Series(0, index=df.index, dtype=int)
        signals.iloc[10:30] = 1
        dist = pd.Series(12.0, index=df.index)
        return StrategyResult(
            name=self.name, source_type=self.source_type, signals=signals,
            stop_loss_distance=dist, take_profit_distance=dist.copy(),
        )


def _es_like_df(n=80):
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    price = 5000.0 + np.cumsum(np.full(n, 0.35))
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.6,
        "low": price - 0.6, "close": price + 0.2, "volume": 100.0,
    })


def test_fit_stop_trades_the_trade_skip_mode_starves():
    """Owen's screenshot: 1 ES contract, risk capped at $450. Under the
    old default ("skip") a strategy stop worth $600/contract sizes to
    ZERO contracts -- no error, just no trades. fit_stop caps the stop
    at the budget and takes the trade, exactly like placing the stop
    by hand."""
    df = _es_like_df()
    base = dict(initial_balance=50_000.0, risk_mode="fixed", risk_value=450.0,
                contract_size=50.0, pip_size=1.0, commission_per_contract=3.5)

    skipped = run_backtest(df, _OneShotStrategy(), RiskConfig(sizing_mode="skip", **base),
                           lookahead_check="skip")
    assert skipped.statistics.total_trades == 0

    fitted = run_backtest(df, _OneShotStrategy(), RiskConfig(sizing_mode="fit_stop", **base),
                          lookahead_check="skip")
    assert fitted.statistics.total_trades == 1
    trade = fitted.trades[0]
    assert trade.contracts == 1
    assert trade.stop_capped is True
    # the capped stop (costs included) never risks more than the budget
    assert trade.risk_at_stop_dollars is not None
    assert trade.risk_at_stop_dollars <= 450.0 + 1e-6


# ---------------------------------------------------------------------------
# Roll adjustment wired into data loading
# ---------------------------------------------------------------------------

def _continuous_csv(path: Path, name: str):
    rows = []
    price = 5000.0
    day = pd.Timestamp("2024-03-04")  # Monday
    days = 0
    while days < 9:
        if day.weekday() < 5:
            for hour in range(7, 16):
                ts = day + pd.Timedelta(hours=hour)
                o = price
                if day == pd.Timestamp("2024-03-11") and hour == 7:
                    o = price + 6.0  # the roll gap: not tradable, must be removed
                c = o + 0.1
                rows.append((ts, o, o + 0.3, o - 0.3, c, 100.0))
                price = c
            days += 1
        day += pd.Timedelta(days=1)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.to_csv(path / name, index=False)


def test_continuous_future_back_adjusted_at_import(tmp_path):
    from app.data.importer import import_csv

    _continuous_csv(tmp_path, "ES1!_2024.csv")
    res = import_csv(str(tmp_path / "ES1!_2024.csv"), historical_complete=True)
    assert res.is_valid
    df = res.dataframe
    assert "roll_bar" in df.columns and "adj_offset" in df.columns
    assert df.attrs.get("roll_adjusted") is True
    assert int(df["roll_bar"].sum()) >= 1
    # after adjustment the overnight gap at the roll is ordinary-sized
    gaps = (df["open"].iloc[1:].to_numpy() - df["close"].iloc[:-1].to_numpy())
    assert np.max(np.abs(gaps)) < 3.0


def test_single_contract_series_not_adjusted(tmp_path):
    from app.data.importer import import_csv

    _continuous_csv(tmp_path, "ESH24_2024.csv")
    res = import_csv(str(tmp_path / "ESH24_2024.csv"), historical_complete=True)
    assert res.is_valid
    assert "roll_bar" not in res.dataframe.columns


# ---------------------------------------------------------------------------
# Presets: no silent legacy rule basis
# ---------------------------------------------------------------------------

def test_no_preset_uses_the_legacy_rule_basis():
    from app.prop.presets import list_presets

    presets = list_presets()
    assert len(presets) >= 15
    for p in presets:
        assert p.dd_basis != "legacy", f"{p.key} still rides the silent legacy basis"
        assert p.dd_basis in {"realized", "eod", "floating"}
        if p.rules_checked_on:
            continue
        assert "NOT re-verified" in p.source_note, f"{p.key} basis is explicit but unverified-looking"


# ---------------------------------------------------------------------------
# Manual strategy saved without its .json extension still appears
# ---------------------------------------------------------------------------

def test_extensionless_manual_strategy_is_listed(tmp_path, monkeypatch):
    import app.strategy.library as library

    manual_dir = tmp_path / "strategies" / "manual"
    manual_dir.mkdir(parents=True)
    (manual_dir / "ES1! Momentum Continuation 5m").write_text(
        '{"name": "Momentum Continuation", "timeframe": "5m",'
        ' "entry_conditions": {"long": [], "short": []},'
        ' "exit_conditions": {}, "risk_management": {}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    names = [s.name for s in library.list_saved_strategies("manual")]
    assert "ES1! Momentum Continuation 5m" in names


def test_the_momentum_continuation_file_itself_is_listed():
    import app.strategy.library as library

    names = [s.name for s in library.list_saved_strategies("manual")]
    assert any(n.startswith("ES1! Momentum Continuation 5m") for n in names)


# ---------------------------------------------------------------------------
# GA reliability adjustments on the plain refinement path
# ---------------------------------------------------------------------------

def test_ga_reliability_adjustments_floors_and_penalties():
    from app.optimize.refinement import apply_ga_reliability_adjustments

    assert apply_ga_reliability_adjustments(1.0, 14) == float("-inf")   # below hard floor: unrankable
    assert apply_ga_reliability_adjustments(1.0, 50) == pytest.approx(0.5)   # scaled by n/100
    assert apply_ga_reliability_adjustments(1.0, 500) == pytest.approx(1.0)  # full confidence
    # 100 of 300 signals skipped for sizing -> tradable share 2/3
    assert apply_ga_reliability_adjustments(1.0, 200, skipped_for_sizing=100) == pytest.approx(2 / 3)
    assert apply_ga_reliability_adjustments(-2.0, 50) == pytest.approx(-2.0)  # bad stays bad


# ---------------------------------------------------------------------------
# Discovery grid: parameter variants are tested AND counted
# ---------------------------------------------------------------------------

def test_discovery_grid_counts_parameter_variants():
    from app.discovery.experiment_runner import run_hypothesis
    from app.discovery.hypothesis import Hypothesis

    rng = np.random.default_rng(7)
    frames = {}
    for mk in ("AAA", "BBB"):
        price = 100.0 + np.cumsum(rng.normal(0, 1, 400))
        frames[mk] = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=400, freq="D"),
            "open": price, "high": price + 1.0, "low": price - 1.0,
            "close": price + 0.1, "volume": 1000.0,
        })
    spec = {"kind": "momentum", "params": {"lookback": 50, "threshold_atr": 1.0,
                                           "stop_atr": 2.5, "target_r": 2.0},
            "direction": "both"}
    hyp = Hypothesis(idea="momentum keeps going", spec=spec, markets=["AAA", "BBB"], timeframes=[])
    run = run_hypothesis(hyp, frames, RiskConfig(initial_balance=50_000.0),
                         n_null=5, min_trades=20)
    # one base-spec cell per market, but trials counted = variants x cells
    assert len(run.cells) == 2
    assert run.n_trials > len(run.cells)
    grid = [e for e in hyp.experiments if e.kind == "grid"]
    assert grid and grid[0].n_variants == run.n_trials


# ---------------------------------------------------------------------------
# Full Pipeline evidence: 2x cost-stress break-it is computed
# ---------------------------------------------------------------------------

def test_full_pipeline_evidence_returns_cost_stress():
    from types import SimpleNamespace

    from app.orchestration.full_pipeline import _replay_and_null_evidence
    from app.strategy.manual import ManualStrategy

    df = _es_like_df(300)
    strat = ManualStrategy({
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow", "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow", "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20, "take_profit_pips": 40,
    })
    risk = RiskConfig(initial_balance=50_000.0, pip_size=1.0, contract_size=50.0)
    final_bt = run_backtest(df, strat, risk, lookahead_check="skip")
    cfg = SimpleNamespace(extra_gates_enabled=True, attempt_replay_starts=3,
                          null_n_seeds=3, break_cost_mult=2.0)
    replay, null, cost_stress = _replay_and_null_evidence(
        strat, df, risk, _owen_rules(), final_bt, cfg, lambda msg: None
    )
    assert cost_stress is not None
    assert cost_stress["multiplier"] == 2.0
    assert "net_profit" in cost_stress and "base_net_profit" in cost_stress
