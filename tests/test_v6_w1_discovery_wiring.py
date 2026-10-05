"""v6 W1 (discovery wiring) tests: astra terminal registration, candidate_source
thread-through, evo unfundable pre-screen, structural breeding budget,
threshold bounds sanity, and Stage-3 eval-count metadata.

Run: ~/workspace/t58-audit-ws-a/venv/bin/python -m pytest tests/test_v6_w1_discovery_wiring.py -q
"""
from __future__ import annotations

import inspect
import random

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.evolution import structural
from app.evolution.engine import EvolutionConfig, EvolutionRunner, _evo_prefilter_task, _EVO_WORKER
from app.prop.simulator import PropRules
from app.search import grammar
from app.search.grammar import (
    lint_warnings,
    random_operand,
    random_threshold,
    validate,
)

ASTRA_NUMERIC = (
    "rsi_regime", "rsi_divergence",
    "session_vwap", "vwap_sigma", "vwap_vah", "vwap_val",
    "vwap_upper_2", "vwap_lower_2", "vwap_poc", "vwap_zscore",
)
ASTRA_BOOLEAN = (
    "rsi_zone_buy", "rsi_zone_sell",
    "vwap_above", "vwap_below", "vwap_outside_value_area",
)
ASTRA_ALL = ASTRA_NUMERIC + ASTRA_BOOLEAN


def _trending_df(n=400, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.linspace(0, 40, n)
    noise = np.cumsum(rng.normal(0, 0.4, n))
    price = 1900 + drift + noise
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.3, "low": price - 0.3,
        "close": price, "volume": 100.0,
    })


def _elite_specs(n=4, seed=9):
    from app.search.strategy_space import FAMILIES
    spec = FAMILIES["trend_breakout"]
    combos = spec.combinations()
    out = []
    for i in range(n):
        cfg = spec.build(combos[(seed + i) % len(combos)])
        out.append(({"source_type": "manual", "config": cfg},
                    {"family": "trend_breakout", "params": {}}))
    return out


def _single_condition_config(operand: dict, boolean: bool) -> dict:
    if boolean:
        cond = {"left": operand, "operator": "is true", "right": {"type": "value", "value": 1}}
    else:
        cond = {"left": operand, "operator": ">",
                "right": random_threshold(random.Random(5), operand["type"])}
    return {
        "name": "w1 sweep",
        "entry_conditions": {"long": [cond], "short": []},
        "risk_management": {"stop_type": "atr", "stop_value": 2.0,
                            "target_type": "atr", "target_value": 3.0,
                            "opposite_signal_exit": True},
        "market": {"direction": "Both"},
    }


# ---------------------------------------------------------------------------
# A1: astra terminal registration + draw sweep
# ---------------------------------------------------------------------------

def test_astra_terminals_registered_with_builders():
    for kind in ASTRA_ALL:
        assert kind in grammar.OPERAND_BUILDERS, f"{kind} has no registered builder"
        assert kind in grammar.ALL_OPERAND_KINDS, f"{kind} missing from ALL_OPERAND_KINDS"
    for kind in ASTRA_NUMERIC:
        assert kind in grammar.NUMERIC_KINDS, f"{kind} missing from NUMERIC_KINDS"
    for kind in ASTRA_BOOLEAN:
        assert kind in grammar.BOOLEAN_KINDS, f"{kind} missing from BOOLEAN_KINDS"


def test_astra_builders_sample_documented_knobs():
    rng = random.Random(20261004)
    for _ in range(10):
        rsi_op = grammar.OPERAND_BUILDERS["rsi_divergence"](rng)
        assert rsi_op["rsi_period"] in (7, 14, 21)
        assert rsi_op["div_lookback"] in (10, 20, 30)
        assert rsi_op["trend_left"] == 5 and rsi_op["trend_right"] == 5
        vwap_op = grammar.OPERAND_BUILDERS["vwap_poc"](rng)
        assert vwap_op["roll_hour"] == 17
        assert vwap_op["value_area_sigma"] == 1.0
        assert vwap_op["extreme_sigma"] == 2.0
        assert vwap_op["poc_buckets"] == 30


def test_astra_draw_sweep_50():
    """50 draws per category: every drawn astra operand validates through
    the real Manual builder, and at least one of each category is drawn."""
    rng = random.Random(20261004)
    seen_numeric, seen_boolean = set(), set()
    for _ in range(50):
        op = random_operand(rng, "numeric")
        if op["type"] in ASTRA_NUMERIC:
            seen_numeric.add(op["type"])
            assert not validate(_single_condition_config(op, boolean=False)), \
                f"astra numeric {op['type']} failed validate"
    for _ in range(50):
        op = random_operand(rng, "boolean")
        if op["type"] in ASTRA_BOOLEAN:
            seen_boolean.add(op["type"])
            assert not validate(_single_condition_config(op, boolean=True)), \
                f"astra boolean {op['type']} failed validate"
    assert seen_numeric, "no astra numeric terminal drawn in 50 draws"
    assert seen_boolean, "no astra boolean terminal drawn in 50 draws"


def test_every_astra_kind_validates():
    rng = random.Random(99)
    for kind in ASTRA_NUMERIC:
        op = grammar.OPERAND_BUILDERS[kind](rng)
        assert not validate(_single_condition_config(op, boolean=False)), kind
    for kind in ASTRA_BOOLEAN:
        op = grammar.OPERAND_BUILDERS[kind](rng)
        assert not validate(_single_condition_config(op, boolean=True)), kind


def test_atr_expansion_contraction_terminals():
    # Registered as boolean terminals and dispatch to the right legs of
    # the tristate atr_regime flag (without changing "is true" semantics).
    assert "atr_expansion" in grammar.BOOLEAN_KINDS
    assert "atr_contraction" in grammar.BOOLEAN_KINDS
    rng = random.Random(3)
    for kind in ("atr_expansion", "atr_contraction"):
        op = grammar.OPERAND_BUILDERS[kind](rng)
        assert op["period"] in (10, 14, 20)
        assert not validate(_single_condition_config(op, boolean=True)), kind
    df = grammar._synthetic_ohlcv()
    from app.strategy.manual import ManualStrategy
    strat = ManualStrategy({"entry_conditions": {"long": [], "short": []}})
    exp = strat._series_from_operand(df, {"type": "atr_expansion", "period": 14}, "left")
    con = strat._series_from_operand(df, {"type": "atr_contraction", "period": 14}, "left")
    reg = strat._series_from_operand(df, {"type": "atr_regime", "period": 14}, "left")
    assert set(exp.unique()) <= {0, 1} and set(con.unique()) <= {0, 1}
    assert bool(((exp == 1) == (reg == 1)).all())
    assert bool(((con == 1) == (reg == -1)).all())


# ---------------------------------------------------------------------------
# A2: candidate_source thread-through (StrategyLabSpec -> SearchStageConfig)
# ---------------------------------------------------------------------------

def test_strategy_lab_spec_candidate_source_default():
    from app.lab.strategy_lab import StrategyLabSpec
    assert StrategyLabSpec().candidate_source == "grammar"
    with pytest.raises(ValueError):
        StrategyLabSpec(candidate_source="gramar")


def test_candidate_source_threaded_into_search_stage_config():
    from app.lab import strategy_lab
    from app.lab.strategy_lab import StrategyLabSpec
    from app.search.batch_runner import SearchStageConfig
    # The default spec value must survive the handoff run_strategy_lab
    # performs when it builds its SearchStageConfig.
    src = inspect.getsource(strategy_lab.run_strategy_lab)
    assert "candidate_source=spec.candidate_source" in src
    stage_cfg = SearchStageConfig(candidate_source=StrategyLabSpec().candidate_source)
    assert stage_cfg.candidate_source == "grammar"


# ---------------------------------------------------------------------------
# A4: unfundable pre-screen in the evolution pre-filter
# ---------------------------------------------------------------------------

def _unfundable_spec() -> dict:
    return {
        "source_type": "manual",
        "config": {
            "name": "unfundable test",
            "entry_conditions": {
                "long": [{"left": {"type": "close"}, "operator": ">",
                          "right": {"type": "value", "value": 0}}],
                "short": [],
            },
            "risk_management": {"stop_type": "fixed", "stop_value": 1000.0,
                                "target_type": "fixed", "target_value": 2000.0,
                                "opposite_signal_exit": True},
            "market": {"direction": "Both"},
        },
    }


def _unfundable_risk() -> RiskConfig:
    # 1% of $10k = $100 risk vs a 1000-pip fixed stop at pip_size 0.01
    # ($10 stop distance) -> 10 units < contract_size 50 -> unfundable.
    return RiskConfig(initial_balance=10_000.0, risk_value=1.0,
                      pip_size=0.01, contract_size=50.0)


def test_evo_prefilter_task_skips_unfundable():
    df = _trending_df()
    _EVO_WORKER["df"], _EVO_WORKER["risk"] = df, _unfundable_risk()
    _EVO_WORKER["adaptive_risk"] = None
    cid, spec, meta, bt, reasons, error, stats = _evo_prefilter_task(
        "t-unfundable", _unfundable_spec(), {"family": "test"},
        1, 1.0, 100.0, 10.0, None,
    )
    assert reasons == ["unfundable"]
    assert bt is None and stats is None
    assert error and "unfundable" in error


def test_evo_prefilter_task_passes_through_without_contract_size():
    # Same spec, default risk (no contract_size) -> pre-screen is a no-op.
    df = _trending_df()
    _EVO_WORKER["df"], _EVO_WORKER["risk"] = df, RiskConfig()
    _EVO_WORKER["adaptive_risk"] = None
    cid, spec, meta, bt, reasons, error, stats = _evo_prefilter_task(
        "t-ok", _unfundable_spec(), {"family": "test"},
        1, 0.5, 100.0, 10.0, None,
    )
    assert reasons != ["unfundable"]


def test_evo_prefilter_counts_unfundable_separately(tmp_path):
    cfg = EvolutionConfig(
        population_size=4, elite_keep=2, max_generations=1,
        min_trades=1, min_profit_factor=0.5, max_drawdown_buffer_mult=10.0,
        parallel_workers=1,  # serial path -- seeds _EVO_WORKER, same task fn
        save_to_library=False,
        knowledge_graph_path=str(tmp_path / "kg.jsonl"),
        checkpoint_path=str(tmp_path / "cp.json"),
        tested_log_path=str(tmp_path / "tested.jsonl"),
    )
    runner = EvolutionRunner(_trending_df(), _unfundable_risk(),
                             # Small prop account: with_prop_safety_defaults
                             # forces risk.initial_balance to match, keeping
                             # the 1000-pip stop unfundable (10 units < 50).
                             PropRules(account_size=10_000.0), cfg, progress_cb=None)
    pop = [("t-unfundable", _unfundable_spec(), {"family": "test"})]
    survivors, rejection_counts, _ = runner._prefilter(pop, gen=0)
    assert survivors == []
    assert rejection_counts["unfundable"] == 1
    assert rejection_counts["build_or_backtest_error"] == 0


# ---------------------------------------------------------------------------
# A5: structural breeding budget
# ---------------------------------------------------------------------------

def _budget_runner(tmp_path, **overrides):
    overrides.setdefault("use_structural_operators", True)
    overrides.setdefault("population_size", 24)
    cfg = EvolutionConfig(
        elite_keep=4, random_seed=11,
        knowledge_graph_path=str(tmp_path / "kg.json"),
        **overrides,
    )
    return EvolutionRunner(_trending_df(n=500), RiskConfig(), PropRules(), cfg)


def test_structural_budget_children_count(tmp_path):
    runner = _budget_runner(tmp_path)
    pop = runner._generate_population(3, _elite_specs())
    budget = [(cid, spec, meta) for cid, spec, meta in pop if meta.get("structural_budget")]
    # max(2, int(24 * 0.15)) == 3, drawn even though the immigrant floor
    # fills the whole population (n_children == 0 here).
    assert len(budget) == max(2, int(24 * 0.15)) == 3
    assert {m["structural_op"] for _, _, m in budget} <= set(structural.STRUCTURAL_OPERATORS)
    for _, spec, _ in budget:
        assert not validate(spec["config"])


def test_structural_mutation_frac_still_respected(tmp_path):
    # Small family set so the regular children loop actually breeds;
    # structural_mutation_frac=0 must keep THAT loop numeric-only while
    # the dedicated budget still breeds structural children.
    runner = _budget_runner(
        tmp_path, population_size=30, families=["trend_breakout"],
        min_immigrants_per_family=1, structural_mutation_frac=0.0,
    )
    pop = runner._generate_population(3, _elite_specs())
    budget = [m for _, _, m in pop if m.get("structural_budget")]
    regular_structural = [m for _, _, m in pop
                          if m.get("structural") and not m.get("structural_budget")]
    assert len(budget) == max(2, int(30 * 0.15))
    assert not regular_structural, "frac=0 must keep the regular children loop numeric-only"


def test_no_structural_budget_when_flag_off(tmp_path):
    runner = _budget_runner(tmp_path, use_structural_operators=False)
    pop = runner._generate_population(3, _elite_specs())
    assert not [m for _, _, m in pop if m.get("structural")]


# ---------------------------------------------------------------------------
# A6: threshold bounds sanity + constancy lint
# ---------------------------------------------------------------------------

def test_threshold_bounds_map_contents():
    assert grammar.THRESHOLD_BOUNDS["adx"] == (0.0, 100.0)
    assert grammar.THRESHOLD_BOUNDS["aroon_up"] == (0.0, 100.0)
    assert grammar.THRESHOLD_BOUNDS["aroon_down"] == (0.0, 100.0)
    assert grammar.THRESHOLD_BOUNDS["cmf"] == (-1.0, 1.0)
    assert grammar.THRESHOLD_BOUNDS["rsi_regime"] == (-1.0, 1.0)
    assert grammar.THRESHOLD_BOUNDS["rsi_divergence"] == (-1.0, 1.0)


def test_threshold_bounds_sanity_draws():
    rng = random.Random(11)
    for kind in ("adx", "aroon_up", "aroon_down"):
        for _ in range(20):
            v = random_threshold(rng, kind)["value"]
            assert 0.0 <= v <= 100.0, (kind, v)
    for _ in range(20):
        assert -1.0 <= random_threshold(rng, "cmf")["value"] <= 1.0
    for kind in ("rsi_regime", "rsi_divergence"):
        for _ in range(20):
            assert -1.0 <= random_threshold(rng, kind)["value"] <= 1.0


def test_dead_code_thresholds_rejected():
    for kind, bad_value in (("adx", 150), ("aroon_up", 120), ("cmf", 2.0)):
        cfg = {"entry_conditions": {"long": [
            {"left": {"type": kind, "period": 14}, "operator": ">",
             "right": {"type": "value", "value": bad_value}}], "short": []}}
        assert validate(cfg), f"{kind} > {bad_value} must be rejected as dead code"


def test_price_scale_thresholds():
    rng = random.Random(11)
    vals = [t["value"] for t in (random_threshold(rng, "session_vwap") for _ in range(40))
            if t.get("type") == "value"]
    assert vals, "expected some constant draws"
    assert all(v > 100 for v in vals), vals  # ~2000-scale, not 0.1-3.0


def test_constancy_lint_warns():
    cfg = {"name": "lint", "entry_conditions": {"long": [
        {"left": {"type": "time_of_day", "session_start": "00:00", "session_end": "23:59"},
         "operator": "is true", "right": {"type": "value", "value": 1}},
    ], "short": []}, "market": {"direction": "Both"}}
    warnings = lint_warnings(cfg)
    assert any("constant" in w and "time_of_day" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# A3: grammar immigrants seeded from the building-block pool
# ---------------------------------------------------------------------------

def test_grammar_block_pool_built_once_per_runner(tmp_path):
    runner = _budget_runner(tmp_path)
    assert getattr(runner, "_grammar_block_pool", None) is None
    pop = runner._generate_population(0, [])
    pool = runner._grammar_block_pool
    assert isinstance(pool, dict) and pool.get("entry_blocks"), "pool must be built and non-empty"
    # Second generation reuses the same pool object (once per runner).
    runner._generate_population(1, [])
    assert runner._grammar_block_pool is pool


def test_generate_random_with_real_block_pool_never_asserts():
    pool = grammar.building_block_pool()
    rng = random.Random(20261004)
    for _ in range(30):
        cfg = grammar.generate_random(rng=rng, block_pool=pool)
        assert not validate(cfg)


# ---------------------------------------------------------------------------
# D4(c): stage_eval_counts on validated records
# ---------------------------------------------------------------------------

def test_stage2_task_reports_ga_total_evaluations():
    from app.search import batch_runner as br
    import inspect as _inspect
    src = _inspect.getsource(br._stage2_task)
    assert "ga_total_evaluations" in src


def test_stage_eval_counts_helper():
    from app.search.batch_runner import _stage_eval_counts
    stage1 = [{"candidate_id": f"c{i}"} for i in range(10)]
    stage2 = [{"candidate_id": "a", "ga_total_evaluations": 44},
              {"candidate_id": "b", "ga_total_evaluations": 40},
              {"candidate_id": "c"}]  # missing key -> 0, not a crash
    stage3 = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    counts = _stage_eval_counts(stage1, stage2, stage3)
    assert counts == {
        "stage1_backtests": 10,
        "stage2_ga_evaluations": 84,
        "stage3_backtests": 2,
        "total_evaluations": 96,
    }


def test_stage_eval_counts_wired_into_stage4():
    from app.search import batch_runner as br
    import inspect as _inspect
    src = _inspect.getsource(br.run_search)
    assert "_stage_eval_counts(stage1_records, stage2_records, stage3_records)" in src
    assert 'rec["stage_eval_counts"]' in src
