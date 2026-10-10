"""v9.14 WS-4: Full Pipeline accuracy audit -- adversarial checks.

These tests attack the pipeline the way a skeptic would, rather than
re-asserting its happy path:

  1. HOLDOUT INTEGRITY -- spy on every dataframe the pipeline hands
     to Steps 1-4 machinery (backtest, GA, walk-forward) and prove
     none of them ever includes a bar past the holdout split; then
     swap the holdout for a canary (prices x40) and prove Steps 1-4
     outputs are bit-identical while the Step-5 holdout result moves.
  2. LOOKAHEAD -- a planted strategy that reads FUTURE bars must be
     hard-failed NOT READY; a clean causal control must not be.
  3. VERDICT BASIS -- chain-level MC pass probability can be 96%
     while per-attempt odds are weak: the verdict must still be
     NOT READY (regression, at the full _make_verdict level).
  4. COST STRESS -- the 2x leg must really charge 2x: trade-level
     commission doubles exactly and every trade's net falls.
  5. DETERMINISM -- same config + seed twice: identical numbers.
  6. GATE BATTERY -- the 70% per-attempt bar, ruin cap, null p limit,
     DSR 0.95, PBO 0.5, ICIR Bonferroni alpha are untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration import full_pipeline as fp
from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline
from app.prop.simulator import PropRules
from app.strategy import library
from app.strategy.manual import ManualStrategy


@pytest.fixture(autouse=True)
def clean_library_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    yield


def _trending_df(n=2400, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift * (1 if (i // 40) % 2 == 0 else -1) + rng.normal(0, 0.00006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config(fast=5, slow=15):
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": fast, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": slow, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
    }


def _cfg(**overrides):
    base = dict(
        n_folds=3, ga_population=4, ga_generations=1, ga_search_mc_sims=20,
        final_mc_sims=200, oos_check_folds=3, holdout_frac=0.2, save_to_library=False,
    )
    base.update(overrides)
    return FullPipelineConfig(**base)


# ----------------------------------------------------------------------
# 1. Holdout integrity
# ----------------------------------------------------------------------

def _make_spies(monkeypatch):
    """Patch the frame-consuming entry points ONCE (a second patch
    would wrap the first spy and double-count). Returns (seen, flip)
    where flip() starts recording into a fresh dict."""
    state = {"seen": None}

    def fresh():
        state["seen"] = {"backtest_max_ts": [], "backtest_callers": [], "ga_max_ts": [],
                         "wf_max_ts": [], "holdout_input_max_ts": []}
        return state["seen"]

    real_bt = fp.run_backtest
    real_ga = fp.run_walkforward_aware_refinement
    real_wf = fp.run_walk_forward
    real_ho = fp.run_holdout_comparison

    def spy_bt(d, *a, **k):
        import traceback
        seen = state["seen"]
        seen["backtest_max_ts"].append(pd.to_datetime(d["timestamp"]).max())
        stack = traceback.extract_stack()
        seen["backtest_callers"].append(stack[-2].name if len(stack) >= 2 else "?")
        return real_bt(d, *a, **k)

    def spy_ga(d, *a, **k):
        state["seen"]["ga_max_ts"].append(pd.to_datetime(d["timestamp"]).max())
        return real_ga(d, *a, **k)

    def spy_wf(d, *a, **k):
        state["seen"]["wf_max_ts"].append(pd.to_datetime(d["timestamp"]).max())
        return real_wf(d, *a, **k)

    def spy_ho(d, *a, **k):
        state["seen"]["holdout_input_max_ts"].append(pd.to_datetime(d["timestamp"]).max())
        return real_ho(d, *a, **k)

    monkeypatch.setattr(fp, "run_backtest", spy_bt)
    monkeypatch.setattr(fp, "run_walkforward_aware_refinement", spy_ga)
    monkeypatch.setattr(fp, "run_walk_forward", spy_wf)
    monkeypatch.setattr(fp, "run_holdout_comparison", spy_ho)
    return fresh


@pytest.mark.timeout(600)
def test_holdout_canary_steps_1_to_4_never_see_it(tmp_path, monkeypatch):
    base = _trending_df()
    split_idx = int(len(base) * 0.8)
    split_ts = base["timestamp"].iloc[split_idx - 1]

    fresh = _make_spies(monkeypatch)
    seen_a = fresh()
    result_a = run_full_pipeline(
        base, ManualStrategy(_sma_config()), RiskConfig(), PropRules(), tmp_path / "a", _cfg())

    canary = base.copy()
    canary.loc[canary.index[split_idx:], ["open", "high", "low", "close"]] *= 40.0
    seen_b = fresh()
    result_b = run_full_pipeline(
        canary, ManualStrategy(_sma_config()), RiskConfig(), PropRules(), tmp_path / "b", _cfg())

    for seen in (seen_a, seen_b):
        assert seen["backtest_max_ts"], "spy must have observed backtests"
        # Exactly ONE documented exception exists: after the final
        # strategy is frozen, the pipeline re-runs it once on the full
        # frame so the HTML report's trade chart spans the whole price
        # axis (see the comment block above `full_history_bt` in
        # run_full_pipeline). It feeds nothing decision-relevant --
        # final_bt / final_mc stay dev-only -- and the canary
        # assertions below prove outcomes can't depend on it.
        over = [(ts, c) for ts, c in zip(seen["backtest_max_ts"], seen["backtest_callers"]) if ts > split_ts]
        assert len(over) <= 1, f"unexpected post-holdout backtests: {over}"
        assert all(ts <= split_ts for ts in seen["ga_max_ts"])
        assert all(ts <= split_ts for ts in seen["wf_max_ts"])
        assert seen["holdout_input_max_ts"], "holdout comparison must have run"
        assert all(ts == base["timestamp"].max() for ts in seen["holdout_input_max_ts"]), (
            "Step 5 must receive the ORIGINAL full frame so its holdout is genuinely unseen")

    # Steps 1-4 outputs are identical despite a wildly different holdout...
    assert result_a.baseline_bt.statistics.net_profit == result_b.baseline_bt.statistics.net_profit
    assert result_a.final_bt.statistics.net_profit == result_b.final_bt.statistics.net_profit
    assert len(result_a.final_bt.trades) == len(result_b.final_bt.trades)
    assert result_a.verdict == result_b.verdict
    assert result_a.final_mc.per_attempt_pass_probability == result_b.final_mc.per_attempt_pass_probability
    # ...while the Step-5 holdout numbers actually moved (the canary was seen there).
    ho_a = (result_a.final_holdout or {}).get("holdout_statistics") or {}
    ho_b = (result_b.final_holdout or {}).get("holdout_statistics") or {}
    assert ho_a.get("net_profit") != ho_b.get("net_profit")


# ----------------------------------------------------------------------
# 2. Planted lookahead
# ----------------------------------------------------------------------

_CHEAT = '''
import pandas as pd


def generate_signals(df):
    fut = df["close"].shift(-5)
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[fut > df["close"] * 1.0005] = 1.0
    sig[fut < df["close"] * 0.9995] = -1.0
    return sig
'''

_CLEAN = '''
import pandas as pd


def generate_signals(df):
    past = df["close"].shift(5)
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[past < df["close"] * 0.9995] = 1.0
    sig[past > df["close"] * 1.0005] = -1.0
    return sig
'''


@pytest.mark.timeout(600)
def test_planted_lookahead_hard_fails_and_clean_control_passes(tmp_path):
    from app.strategy.lookahead_check import check_for_lookahead
    from app.strategy.python import PythonStrategy

    df = _trending_df()
    cheat_path = tmp_path / "cheat.py"
    cheat_path.write_text(_CHEAT, encoding="utf-8")
    clean_path = tmp_path / "clean.py"
    clean_path.write_text(_CLEAN, encoding="utf-8")

    cheat = check_for_lookahead(PythonStrategy(cheat_path), df, max_signal_checkpoints=8)
    assert cheat.bug_detected, "the planted future-reading strategy must be caught"

    clean = check_for_lookahead(PythonStrategy(clean_path), df, max_signal_checkpoints=8)
    assert not clean.bug_detected, "the causal control must not be flagged"

    result = run_full_pipeline(
        df, PythonStrategy(cheat_path), RiskConfig(), PropRules(), tmp_path / "run", _cfg(),
    )
    assert result.verdict == "NOT READY"
    assert result.lookahead_hard_fail is True
    assert any("lookahead" in r.lower() for r in result.verdict_reasons)


# ----------------------------------------------------------------------
# 3. Verdict basis (per-attempt, never chain)
# ----------------------------------------------------------------------

def test_verdict_not_ready_on_weak_per_attempt_despite_strong_chain():
    from app.monte_carlo.engine import MonteCarloResult
    from app.orchestration.full_pipeline import _make_verdict

    mc = MonteCarloResult(
        n_simulations=1000, evaluation_pass_probability=96.0, first_payout_probability=80.0,
        failure_before_payout_probability=5.0, multiple_payout_probability=40.0,
        median_days_to_pass=10.0, median_days_to_first_payout=20.0, average_days_to_first_payout=22.0,
        median_return_pct=10.0, mean_return_pct=11.0, expected_payout=500.0, median_payout=450.0,
        total_simulated_withdrawals=1000.0, median_drawdown_pct=3.0, p95_drawdown_pct=6.0,
        worst_drawdown_pct=9.0, risk_of_ruin_pct=5.0, median_max_losing_streak=4.0, worst_max_losing_streak=8,
        return_percentiles={25: 8.0, 50: 10.0, 75: 12.0},
        per_attempt_pass_ci95=(20.0, 45.0), per_attempt_pass_probability=33.0,
    )
    verdict, reasons, *_ = _make_verdict(mc, None)
    assert verdict == "NOT READY"
    assert any("per-attempt eval pass odds too low" in r for r in reasons)

    # And the mirror: weak CHAIN (40%) with strong per-attempt odds must
    # not trip the per-attempt gate (the verdict may say many things,
    # but not THAT).
    mc2 = MonteCarloResult(
        n_simulations=1000, evaluation_pass_probability=40.0, first_payout_probability=80.0,
        failure_before_payout_probability=5.0, multiple_payout_probability=40.0,
        median_days_to_pass=10.0, median_days_to_first_payout=20.0, average_days_to_first_payout=22.0,
        median_return_pct=10.0, mean_return_pct=11.0, expected_payout=500.0, median_payout=450.0,
        total_simulated_withdrawals=1000.0, median_drawdown_pct=3.0, p95_drawdown_pct=6.0,
        worst_drawdown_pct=9.0, risk_of_ruin_pct=5.0, median_max_losing_streak=4.0, worst_max_losing_streak=8,
        return_percentiles={25: 8.0, 50: 10.0, 75: 12.0},
        per_attempt_pass_ci95=(75.0, 92.0), per_attempt_pass_probability=85.0,
    )
    _v2, reasons2, *_ = _make_verdict(mc2, None)
    assert not any("per-attempt eval pass odds too low" in r for r in reasons2)


# ----------------------------------------------------------------------
# 4. Cost stress (trade level)
# ----------------------------------------------------------------------

def test_cost_stress_charges_double_at_trade_level():
    from dataclasses import replace

    from app.backtest.engine import run_backtest

    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig(
        initial_balance=100_000.0, risk_mode="fixed", risk_value=100.0,
        sizing_mode="fixed_contracts", fixed_contracts=1,
        commission_per_contract=2.0, commission_per_trade=0.0,
        spread_pips=1.0, slippage_pips=0.5, pip_size=1.0, contract_size=1.0,
    )
    stressed = replace(
        risk, spread_pips=risk.spread_pips * 2.0, slippage_pips=risk.slippage_pips * 2.0,
        commission_per_contract=risk.commission_per_contract * 2.0,
        commission_per_trade=risk.commission_per_trade * 2.0,
    )
    base = run_backtest(df, strategy, risk)
    stress = run_backtest(df, strategy, stressed)
    assert len(base.trades) > 5
    assert len(stress.trades) == len(base.trades)
    assert [t.entry_time for t in stress.trades] == [t.entry_time for t in base.trades]
    base_comm = sum(t.commission for t in base.trades)
    stress_comm = sum(t.commission for t in stress.trades)
    assert base_comm > 0
    assert stress_comm == pytest.approx(2.0 * base_comm, rel=1e-9)
    for tb, ts_ in zip(base.trades, stress.trades):
        assert ts_.pnl <= tb.pnl + 1e-9
    assert stress.statistics.net_profit < base.statistics.net_profit


# ----------------------------------------------------------------------
# 5. Determinism
# ----------------------------------------------------------------------

@pytest.mark.timeout(600)
def test_pipeline_deterministic_same_seed(tmp_path):
    df = _trending_df()
    r1 = run_full_pipeline(df, ManualStrategy(_sma_config()), RiskConfig(), PropRules(), tmp_path / "r1", _cfg())
    r2 = run_full_pipeline(df, ManualStrategy(_sma_config()), RiskConfig(), PropRules(), tmp_path / "r2", _cfg())
    assert r1.verdict == r2.verdict
    assert r1.baseline_bt.statistics.net_profit == r2.baseline_bt.statistics.net_profit
    assert r1.final_bt.statistics.net_profit == r2.final_bt.statistics.net_profit
    assert len(r1.final_bt.trades) == len(r2.final_bt.trades)
    assert r1.final_mc.per_attempt_pass_probability == r2.final_mc.per_attempt_pass_probability


# ----------------------------------------------------------------------
# 6. Gate battery intact
# ----------------------------------------------------------------------

def test_gate_thresholds_unchanged():
    from app.search.robustness import deflated_sharpe_gate
    from app.validation.cpcv import pbo_gate
    from app.validation.icir import DEFAULT_ALPHA, bonferroni_adjusted_alpha

    cfg = FullPipelineConfig()
    assert cfg.min_per_attempt_pass_pct == 70.0
    assert cfg.risk_of_ruin_cap == 20.0
    assert cfg.null_p_max == 0.10
    import inspect
    assert inspect.signature(deflated_sharpe_gate).parameters["min_probabilistic_sharpe"].default == 0.95
    assert inspect.signature(pbo_gate).parameters["max_pbo"].default == 0.5
    assert DEFAULT_ALPHA == 0.05
    assert bonferroni_adjusted_alpha(0.05, 4) == pytest.approx(0.0125)
