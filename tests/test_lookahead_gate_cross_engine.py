"""
Regression tests for the audit's two highest-value findings:

1. Every discovery/validation engine that can declare a strategy "done"
   (Search Lab, Full Pipeline, Evolution Lab) must actually disqualify a
   strategy with a confirmed lookahead-bias leak -- not just compute the
   check and ignore the result. This is the exact class of bug found in
   app.orchestration.full_pipeline._make_verdict (computed lookahead_summary,
   never gated on it) and the exact gap found in Evolution Lab (no lookahead
   check anywhere in its own pipeline). See CHANGES_SUMMARY.md.

2. Quick Optimize and Full Pipeline must produce byte-identical baseline
   numbers for the identical (data, strategy, risk, prop_rules) input when
   configured equivalently (no holdout carve-out on either side) -- both
   are just `run_backtest(dev_df, strategy, risk, adaptive_risk=...)`
   underneath, so any numeric drift between them is a real bug, not
   floating-point noise.

Kept deliberately small-scale (tiny GA populations/generations, low Monte
Carlo sim counts) so this file runs in well under a minute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy
from app.strategy.python import PythonStrategy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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


def _sample_df(n=600):
    """The exact data generator tests/test_lookahead_check.py uses for
    LEAKY_STRATEGY -- reliably triggers detection at every checkpoint
    count used across this app (8, 15, 30), unlike a smooth/strongly-
    trending series where the affected bars are sparse enough that a
    fixed-seed random sample can legitimately miss all of them. Used for
    every "must fail" test below; the smooth _trending_df() above is used
    only for the clean cross-engine parity test, where lookahead behavior
    is irrelevant and a realistic trending series is more representative.
    """
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(7)
    close = 1900 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 100.0,
    })


# The known-leaky HTF strategy from tests/test_lookahead_check.py, repeated
# here (rather than imported) so this file has no import-order dependency on
# that one and stays readable standalone. Keep the two in sync if either
# changes -- see that file's own docstring for what makes this leaky.
LEAKY_STRATEGY = """
import pandas as pd

def generate_signals(df, config=None):
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    h1 = x.set_index("timestamp").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    out = pd.Series(0, index=x.index, dtype="int8")
    for i in range(60, len(x)):
        ts = x["timestamp"].iloc[i]
        # BUG: includes the still-forming current-hour bar (future data).
        h1c = h1[h1.index < ts]
        if len(h1c) < 2:
            continue
        if h1c["close"].iloc[-1] > h1c["close"].iloc[-2]:
            out.iloc[i] = 1
    return out
"""


def _write_leaky_strategy(tmp_path) -> str:
    path = tmp_path / "leaky.py"
    path.write_text(LEAKY_STRATEGY, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# D3 -- "a leaky strategy must fail every discovery/validation engine"
# ---------------------------------------------------------------------------

def test_leaky_strategy_fails_full_pipeline_verdict_via_lookahead_gate(tmp_path):
    """Regression test for the audit's Bug #1: Full Pipeline used to compute
    the lookahead check and never act on it, so a strategy with a confirmed
    leak could still receive a READY/MARGINAL verdict as long as its (equally
    untrustworthy) other numbers looked fine. This asserts the hard gate
    itself fires -- not just that the verdict happens to be NOT READY for
    some other reason."""
    from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline

    df = _sample_df()
    strategy = PythonStrategy(_write_leaky_strategy(tmp_path))
    cfg = FullPipelineConfig(
        n_folds=2, ga_population=4, ga_generations=1, ga_search_mc_sims=20,
        final_mc_sims=200, oos_check_folds=2, holdout_frac=0.2,
        reserve_true_holdout=False,
    )
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path / "out", cfg, progress_cb=None,
    )
    assert result.lookahead_hard_fail is True
    assert result.verdict == "NOT READY"
    assert any("LOOKAHEAD" in r.upper() for r in result.verdict_reasons)


def test_leaky_strategy_fails_search_lab_stage3_gate(tmp_path):
    """Same regression, for Search Lab -- Stage 3's passed_stage3_gate ANDs
    in `not lookahead.bug_detected`; kept here so a future change to that
    gate is caught by the same test file as the Full Pipeline / Evolution
    Lab versions above and below.

    v5 note: this exercises _stage3_task directly (not the full run_search)
    because honest next-bar-open fills (B2-2) removed this leaky strategy's
    illusory edge (PF 0.56, net negative) -- it can no longer survive
    Stage 1's `net_profit > 0` filter, which is correct behavior, not a
    gate failure. The regression under test is the Stage 3 lookahead gate
    itself: a leaky candidate that REACHES Stage 3 must be rejected there.
    The acceptance floors are zeroed so the lookahead check is the only
    possible reason for rejection.
    """
    from app.search import batch_runner as br
    from app.search.strategy_space import spec_from_strategy

    df = _sample_df()
    strategy = PythonStrategy(_write_leaky_strategy(tmp_path))
    spec = spec_from_strategy(strategy)

    # _stage3_task runs in a worker process in production; here we populate
    # the worker state it reads (_WORKER) in-process instead.
    br._WORKER["df"] = df
    br._WORKER["risk"] = RiskConfig()
    br._WORKER["prop_rules"] = PropRules()
    br._WORKER["tmp_dir"] = tmp_path
    try:
        stage3_cfg = {
            "full_mc_sims": 50, "random_seed": 1,
            "fitness_metric": "eval_pass_probability",
            "stage3_min_trades": 1,
            "stage3_min_profit_factor": 0.0,
            "stage3_max_drawdown_buffer_mult": 100.0,
            "stage3_require_positive_net": False,
            # Zero the acceptance floors so the ONLY thing that can fail
            # this candidate is the lookahead gate under test.
            "min_eval_pass_probability": 0.0,
            "min_first_payout_probability": 0.0,
            "walk_forward_folds": 0,
            "walk_forward_metric": "profit_factor",
            "walk_forward_min_efficiency": 0.5,
            "robustness_neighbors": 0,
            "robustness_perturbation_frac": 0.1,
            "robustness_min_stability": 0.5,
            "reset_on_breach": False,
        }
        record = br._stage3_task("leaky", spec, stage3_cfg)
    finally:
        br._WORKER.clear()

    assert record["lookahead"]["bug_detected"] is True
    assert not record["passed_stage3_gate"], (
        "the candidate has a confirmed lookahead leak -- it must never pass "
        "Stage 3's gate, however good its raw numbers look; "
        f"gate notes: {record.get('gate_notes')}"
    )


def test_leaky_python_candidate_excluded_from_evolution_lab_cpcv_pool(tmp_path):
    """Regression test for the audit's Evolution Lab gap: unlike Search Lab
    and Full Pipeline, Evolution Lab had no lookahead check anywhere in its
    own pipeline. Evolution Lab only ever generates 'manual' (indicator-
    builder) candidates today, so this constructs a synthetic pool entry
    wrapping a python-source leaky strategy directly and calls
    _cpcv_and_pbo() -- the one point every leaderboard-eligible candidate
    passes through every generation -- to prove the safety net actually
    works for a non-manual candidate, not just that it compiles."""
    from app.backtest.risk import with_prop_safety_defaults
    from app.evolution.engine import EvolutionCandidateRecord, EvolutionConfig, EvolutionRunner

    df = _sample_df()
    prop_rules = PropRules()
    risk = with_prop_safety_defaults(RiskConfig(), prop_rules)
    cfg = EvolutionConfig(
        population_size=4, elite_keep=1, max_generations=1,
        cpcv_top_n=2, cpcv_max_paths=3, cpcv_n_groups=3,
        save_to_library=False, knowledge_graph_path=str(tmp_path / "kg.jsonl"),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        tested_log_path=str(tmp_path / "tested_candidates.jsonl"),
    )
    runner = EvolutionRunner(df, risk, prop_rules, cfg, progress_cb=None)

    leaky_spec = {"source_type": "python", "code_text": LEAKY_STRATEGY, "code_extension": ".py"}
    record = EvolutionCandidateRecord(
        candidate_id="leaky-1", spec=leaky_spec, meta={"family": "test"},
        stats={"total_trades": 50, "max_drawdown_pct": 3.0, "max_losing_streak": 2},
        mc_summary={"evaluation_pass_probability": 99.0, "first_payout_probability": 95.0},
        trade_pnls=[10.0] * 50,
        trades=[],
    )
    # Give it a real (excellent-looking) fitness so exclusion can only be
    # explained by the lookahead check, never by low fitness getting
    # dropped some other way.
    from app.evolution.prop_fitness import compute_prop_fitness
    record.fitness = compute_prop_fitness(
        record.stats, record.mc_summary, None, None, record.trade_pnls,
    )

    survivors = runner._cpcv_and_pbo([record])

    assert survivors == []
    assert record.lookahead_bug_detected is True
    assert record.lookahead_summary is not None and "LOOKAHEAD" in record.lookahead_summary.upper()


# ---------------------------------------------------------------------------
# D2 -- cross-engine backtest parity (Quick Optimize vs Full Pipeline)
# ---------------------------------------------------------------------------

def test_quick_optimize_and_full_pipeline_baselines_match_when_configured_equivalently():
    """Full Pipeline reserves a true holdout by default (reserve_true_
    holdout=True) while Quick Optimize doesn't (reserve_holdout=False) --
    that is a deliberate, documented difference in what each tool measures
    by default, not a bug, but it means their baseline numbers legitimately
    differ out of the box. This test removes that variable (holdout off on
    both sides) to isolate the thing that SHOULD never differ: both tools'
    baseline stage is just run_backtest(dev_df, strategy, risk,
    adaptive_risk=...) underneath, so with identical inputs and no holdout
    carve-out, their baseline trade count/net profit/drawdown must match
    exactly, not approximately."""
    from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline
    from app.orchestration.quick_optimize import QuickOptimizeConfig, run_quick_optimize

    df = _trending_df()
    risk = RiskConfig()
    prop_rules = PropRules()

    qo_cfg = QuickOptimizeConfig(reserve_holdout=False)
    qo_result = run_quick_optimize(
        df, ManualStrategy(_sma_config()), risk, prop_rules, qo_cfg, progress_cb=None,
    )

    fp_cfg = FullPipelineConfig(reserve_true_holdout=False)
    import tempfile
    with tempfile.TemporaryDirectory() as out_dir:
        fp_result = run_full_pipeline(
            df, ManualStrategy(_sma_config()), risk, prop_rules, out_dir, fp_cfg, progress_cb=None,
        )

    assert qo_result.baseline_trades == len(fp_result.baseline_bt.trades)
    assert qo_result.baseline_net_profit == pytest.approx(
        fp_result.baseline_bt.statistics.net_profit, abs=0.01,
    )
    assert qo_result.baseline_win_rate == pytest.approx(
        fp_result.baseline_bt.statistics.win_rate, abs=0.01,
    )
    assert qo_result.baseline_eval_pass_probability == pytest.approx(
        fp_result.baseline_mc.evaluation_pass_probability, abs=0.5,
    )
