"""v5 B1-1/B1-3 tests: compositional grammar, structural operators,
evolution engine wiring (structural flag + holdout gate + 70% default),
and the Search Lab Stage 2 grammar hook's config surface.

Run: ~/workspace/t58-audit-ws-a/venv/bin/python -m pytest tests/test_structural_grammar_v5.py -q
"""
from __future__ import annotations

import copy
import random
import sys

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.evolution.engine import (
    EvolutionCandidateRecord,
    EvolutionConfig,
    EvolutionRunner,
)
from app.prop.simulator import PropRules
from app.search import grammar
from app.search.grammar import building_block_pool, generate_random, validate
from app.evolution import structural


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _always_long_config() -> dict:
    """A deterministic config that always holds long -- guaranteed trades
    on any df, for the holdout-gate integration test."""
    return {
        "name": "always-long test",
        "entry_conditions": {
            "long": [{"left": {"type": "close"}, "operator": ">",
                      "right": {"type": "value", "value": 0}}],
            "short": [],
        },
        "risk_management": {"stop_type": "atr", "stop_value": 2.0,
                            "target_type": "atr", "target_value": 3.0,
                            "opposite_signal_exit": True},
        "market": {"direction": "Both"},
    }


class _Fit:
    def __init__(self, score: float):
        self.final_score = score


def _gated_record(cid: str, score: float, config: dict | None = None) -> EvolutionCandidateRecord:
    return EvolutionCandidateRecord(
        candidate_id=cid, spec={"source_type": "manual", "config": config or _always_long_config()},
        meta={"family": "grammar"}, fitness=_Fit(score),
    )


def _runner(tmp_path, **overrides) -> EvolutionRunner:
    cfg = EvolutionConfig(
        knowledge_graph_path=str(tmp_path / "kg.jsonl"),
        checkpoint_path=str(tmp_path / "cp.json"),
        tested_log_path=str(tmp_path / "tested.jsonl"),
        mc_sims=20,
        **overrides,
    )
    return EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), cfg, progress_cb=None)


# ---------------------------------------------------------------------------
# 1. Grammar: generate + validate round-trip (50 random configs)
# ---------------------------------------------------------------------------

def test_grammar_generate_validate_round_trip_50():
    rng = random.Random(20261004)
    for i in range(50):
        cfg = generate_random(rng=rng)
        errors = validate(cfg)
        assert not errors, f"config {i} invalid: {errors[:2]}"


def test_grammar_rejects_invalid_configs():
    # Impossible bounded-oscillator threshold.
    bad_thr = {"entry_conditions": {"long": [
        {"left": {"type": "rsi", "period": 14}, "operator": ">",
         "right": {"type": "value", "value": 150}}], "short": []}}
    assert validate(bad_thr), "rsi > 150 must be rejected"
    # Unknown operand kind.
    bad_kind = {"entry_conditions": {"long": [
        {"left": {"type": "no_such_kind"}, "operator": ">",
         "right": {"type": "value", "value": 1}}], "short": []}}
    assert validate(bad_kind), "unknown kind must be rejected"
    # No conditions at all.
    assert validate({"entry_conditions": {"long": [], "short": []}})
    # Bad connector count.
    bad_conn = {"entry_conditions": {
        "long": [
            {"left": {"type": "close"}, "operator": ">", "right": {"type": "sma", "period": 20}},
            {"left": {"type": "rsi", "period": 14}, "operator": "<", "right": {"type": "value", "value": 30}},
        ],
        "long_connectors": ["AND", "OR", "AND"],
        "short": [],
    }}
    assert validate(bad_conn), "connector count mismatch must be rejected"
    # Bad operator.
    bad_op = {"entry_conditions": {"long": [
        {"left": {"type": "close"}, "operator": "~~>",
         "right": {"type": "value", "value": 1}}], "short": []}}
    assert validate(bad_op), "unknown operator must be rejected"


def test_grammar_extension_point_register_operand_kind():
    def _my_terminal(rng: random.Random) -> dict:
        return {"type": "sma", "period": int(rng.choice([10, 20])), "field": "close"}

    grammar.register_operand_kind("test_sma_alias", _my_terminal)
    assert "test_sma_alias" in grammar.OPERAND_BUILDERS
    op = grammar.random_operand(random.Random(1), "any")
    assert isinstance(op, dict) and "type" in op
    # Cleanup: don't leak the test terminal into other tests' draws.
    del grammar.OPERAND_BUILDERS["test_sma_alias"]


# ---------------------------------------------------------------------------
# 2. Building-block pool: decomposes the 84 templates, doesn't delete them
# ---------------------------------------------------------------------------

def test_building_block_pool_decomposes_templates():
    from app.search.strategy_space import FAMILIES
    # v7 (2026-10-05): 84 frozen templates + 10 new v7 families
    # (app/search/families_v7.py). v8 (2026-10-05): + 8 new v8 families
    # (app/search/families_v8.py). The original 84 must still exist,
    # untouched -- the count check below enforces all three.
    try:
        from app.search.families_v7 import V7_FAMILY_NAMES
        v7_names = set(V7_FAMILY_NAMES)
    except ImportError:
        v7_names = set()
    try:
        from app.search.families_v8 import V8_FAMILY_NAMES
        v8_names = set(V8_FAMILY_NAMES)
    except ImportError:
        v8_names = set()
    assert len(FAMILIES) == 84 + len(v7_names) + len(v8_names), (
        f"expected 84 frozen templates + {len(v7_names)} v7 families + {len(v8_names)} v8 families"
    )
    original = set(FAMILIES) - v7_names - v8_names
    assert len(original) == 84, "the 84 frozen templates must still exist, untouched"
    pool = building_block_pool()
    assert pool["families_used"], "no families decomposed"
    assert pool["entry_blocks"], "no entry blocks extracted"
    assert pool["risk_blocks"], "no risk blocks extracted"
    # Every extracted block must itself be grammar-valid in context.
    rng = random.Random(5)
    for _ in range(10):
        block = grammar.random_block(pool, "entry_blocks", rng)
        cfg = {"name": "pool-block-test",
               "entry_conditions": {"long": block["conditions"],
                                    "long_connectors": block["connectors"],
                                    "short": []},
               "risk_management": grammar.random_block(pool, "risk_blocks", rng)}
        assert not validate(cfg), f"pool block invalid: {validate(cfg)[:2]}"
    # Templates untouched: FAMILIES registry still has all 84 originals
    # after pooling (v7: plus the 10 new families; v8: plus 8 more).
    assert len(FAMILIES) - len(v7_names) - len(v8_names) == 84


# ---------------------------------------------------------------------------
# 3. Structural operators: validity (+ graft specifics)
# ---------------------------------------------------------------------------

_OPERATORS = ["add_condition", "remove_condition", "swap_operand_kind",
              "flip_connector", "mutate_filter", "mutate_risk_block"]


@pytest.mark.parametrize("op_name", _OPERATORS)
def test_structural_operator_preserves_validity(op_name):
    base = generate_random(rng=random.Random(77))
    op = getattr(structural, op_name)
    for seed in range(8):
        out = op(copy.deepcopy(base), random.Random(1000 + seed))
        errors = validate(out)
        assert not errors, f"{op_name} seed {seed}: {errors[:2]}"


def test_structural_operators_safe_on_degenerate_config():
    mini = {"name": "mini",
            "entry_conditions": {"long": [
                {"left": {"type": "close"}, "operator": ">",
                 "right": {"type": "sma", "period": 20, "field": "close"}}],
                "short": []}}
    assert not validate(mini)
    for op_name in _OPERATORS:
        out = getattr(structural, op_name)(copy.deepcopy(mini), random.Random(3))
        assert not validate(out), f"{op_name} broke the degenerate config"


def test_graft_subtree_produces_valid_children_and_preserves_parents():
    parent_a = generate_random(rng=random.Random(11))
    parent_b = generate_random(rng=random.Random(22))
    a0, b0 = copy.deepcopy(parent_a), copy.deepcopy(parent_b)
    children = [structural.graft_subtree(parent_a, parent_b, random.Random(300 + i)) for i in range(10)]
    for child in children:
        assert not validate(child), f"graft child invalid: {validate(child)[:2]}"
    assert parent_a == a0 and parent_b == b0, "graft must not mutate its parents"
    # At least one child must actually differ from parent A (a real
    # crossover happened, not a silent identity return).
    assert any(c != parent_a for c in children), "graft never changed anything in 10 tries"


def test_random_operator_never_returns_invalid():
    base = generate_random(rng=random.Random(9))
    seen_ops = set()
    for seed in range(15):
        name, out = structural.random_operator(copy.deepcopy(base), random.Random(seed))
        assert not validate(out), f"random_operator({name}) invalid"
        seen_ops.add(name)
    assert len(seen_ops) > 1, f"random_operator only ever picked {seen_ops}"


# ---------------------------------------------------------------------------
# 4. Holdout gate (B1-3)
# ---------------------------------------------------------------------------

def _fake_holdout(pass_prob: float):
    def _fake(self, candidate=None, mc_sims=None):
        return {
            "n_bars": 80, "n_trades": 12,
            "per_attempt_pass_probability": pass_prob,
            "per_attempt_payout_probability": pass_prob / 2,
        }
    return _fake


def test_holdout_gate_rejects_failing_champion(tmp_path, monkeypatch):
    runner = _runner(tmp_path, target_eval_pass_pct=70.0, elite_keep=3)
    monkeypatch.setattr(EvolutionRunner, "evaluate_champion_on_locked_holdout",
                        _fake_holdout(10.0))  # everyone fails
    bad = _gated_record("bad", 99.0)
    worse = _gated_record("worse", 50.0)
    runner._update_leaderboard([bad, worse])
    assert runner.leaderboard == [], "failing champions must be rejected, not crowned"
    assert bad.holdout_check["passed"] is False
    assert worse.holdout_check["passed"] is False


def test_holdout_gate_promotes_next_passing_candidate(tmp_path, monkeypatch):
    runner = _runner(tmp_path, target_eval_pass_pct=70.0, elite_keep=3)
    calls = []

    def _fake(self, candidate=None, mc_sims=None):
        calls.append(candidate.candidate_id)
        prob = 85.0 if candidate.candidate_id == "second" else 10.0
        return {"n_bars": 80, "n_trades": 12,
                "per_attempt_pass_probability": prob,
                "per_attempt_payout_probability": prob / 2}

    monkeypatch.setattr(EvolutionRunner, "evaluate_champion_on_locked_holdout", _fake)
    first = _gated_record("first", 99.0)
    second = _gated_record("second", 80.0)
    third = _gated_record("third", 10.0)
    runner._update_leaderboard([first, second, third])
    assert [r.candidate_id for r in runner.leaderboard] == ["second", "third"]
    assert first.holdout_check["passed"] is False
    assert second.holdout_check["passed"] is True
    # Lazy evaluation: "third" never needed the expensive holdout call.
    assert calls == ["first", "second"]
    assert third.holdout_check is None


def test_holdout_gate_disabled_when_target_none(tmp_path, monkeypatch):
    runner = _runner(tmp_path, target_eval_pass_pct=None, elite_keep=3)

    def _boom(self, candidate=None, mc_sims=None):
        raise AssertionError("gate must not evaluate when disabled")

    monkeypatch.setattr(EvolutionRunner, "evaluate_champion_on_locked_holdout", _boom)
    recs = [_gated_record("a", 99.0), _gated_record("b", 50.0)]
    runner._update_leaderboard(recs)
    assert [r.candidate_id for r in runner.leaderboard] == ["a", "b"]
    assert all(r.holdout_check is None for r in runner.leaderboard)


def test_holdout_gate_fails_open_on_eval_error(tmp_path, monkeypatch):
    runner = _runner(tmp_path, target_eval_pass_pct=70.0, elite_keep=3)

    def _raise(self, candidate=None, mc_sims=None):
        raise RuntimeError("simulated holdout blowup")

    monkeypatch.setattr(EvolutionRunner, "evaluate_champion_on_locked_holdout", _raise)
    rec = _gated_record("fragile", 99.0)
    runner._update_leaderboard([rec])
    # Inconclusive != failure: kept, flagged, never silently crowned-or-killed.
    assert [r.candidate_id for r in runner.leaderboard] == ["fragile"]
    assert rec.holdout_check["passed"] is None
    assert "simulated holdout blowup" in rec.holdout_check["error"]


def test_holdout_gate_caches_verdict_across_generations(tmp_path, monkeypatch):
    runner = _runner(tmp_path, target_eval_pass_pct=70.0, elite_keep=3)
    calls = []

    def _fake(self, candidate=None, mc_sims=None):
        calls.append(candidate.candidate_id)
        return {"n_bars": 80, "n_trades": 12,
                "per_attempt_pass_probability": 90.0,
                "per_attempt_payout_probability": 40.0}

    monkeypatch.setattr(EvolutionRunner, "evaluate_champion_on_locked_holdout", _fake)
    rec = _gated_record("steady", 99.0)
    runner._update_leaderboard([rec])
    runner._update_leaderboard([rec])  # second generation, same candidate
    assert calls == ["steady"], "each candidate must be holdout-evaluated at most once"


def test_holdout_gate_integration_real_backtest(tmp_path):
    """End-to-end through the REAL evaluate_champion_on_locked_holdout
    (backtest + MC on the locked slice). An impossible bar (101%) must
    reject; a trivial bar (0%) must pass -- deterministic either way."""
    runner = _runner(tmp_path, target_eval_pass_pct=101.0, elite_keep=2)
    rec = _gated_record("real", 99.0)
    runner._update_leaderboard([rec])
    assert runner.leaderboard == []
    assert rec.holdout_check["passed"] is False
    assert rec.holdout_check["n_trades"] > 0, "the always-long config must trade on the holdout"

    runner2 = _runner(tmp_path, target_eval_pass_pct=0.0, elite_keep=2)
    rec2 = _gated_record("real2", 99.0)
    runner2._update_leaderboard([rec2])
    assert [r.candidate_id for r in runner2.leaderboard] == ["real2"]
    assert rec2.holdout_check["passed"] is True


# ---------------------------------------------------------------------------
# 5. Config defaults (B1-3: 70.0) and the structural reproducibility flag
# ---------------------------------------------------------------------------

def test_target_eval_pass_pct_defaults_to_70():
    assert EvolutionConfig().target_eval_pass_pct == 70.0


def test_use_structural_operators_defaults_on():
    assert EvolutionConfig().use_structural_operators is True


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


def test_numeric_only_flag_is_reproducible(tmp_path):
    """use_structural_operators=False must give byte-identical populations
    for the same seed (determinism), regardless of the new code existing."""
    def _pop(**kw):
        cfg = EvolutionConfig(population_size=24, elite_keep=4, random_seed=11,
                              knowledge_graph_path=str(tmp_path / "kg.json"),
                              use_structural_operators=False, **kw)
        r = EvolutionRunner(_trending_df(n=500), RiskConfig(), PropRules(), cfg)
        return [(cid, spec, meta) for cid, spec, meta in r._generate_population(3, _elite_specs())]

    p1, p2 = _pop(), _pop()
    assert [(c, m) for c, _, m in p1] == [(c, m) for c, _, m in p2]
    assert [s["config"] for _, s, _ in p1] == [s["config"] for _, s, _ in p2]


def test_numeric_only_flag_never_touches_new_modules(tmp_path, monkeypatch):
    """The flag must GENUINELY restore old behavior: with it off, the
    population path must succeed even if app.search.grammar /
    app.evolution.structural blow up on import."""
    import app.evolution.structural  # noqa: F401  (ensure real modules exist first)

    class _Boom:
        def __getattr__(self, name):
            raise ImportError("structural/grammar must not be imported in numeric-only mode")

    monkeypatch.setitem(sys.modules, "app.search.grammar", _Boom())
    monkeypatch.setitem(sys.modules, "app.evolution.structural", _Boom())
    cfg = EvolutionConfig(population_size=24, elite_keep=4, random_seed=11,
                          knowledge_graph_path=str(tmp_path / "kg.json"),
                          use_structural_operators=False)
    runner = EvolutionRunner(_trending_df(n=500), RiskConfig(), PropRules(), cfg)
    pop = runner._generate_population(3, _elite_specs())
    assert len(pop) > 0
    assert all("structural" not in meta for _, _, meta in pop)


def test_structural_mode_breeds_structural_children(tmp_path):
    cfg = EvolutionConfig(population_size=24, elite_keep=4, random_seed=11,
                          knowledge_graph_path=str(tmp_path / "kg.json"),
                          use_structural_operators=True,
                          structural_mutation_frac=1.0, structural_graft_frac=0.0)
    runner = EvolutionRunner(_trending_df(n=500), RiskConfig(), PropRules(), cfg)
    pop = runner._generate_population(3, _elite_specs())
    structural_kids = [meta for _, _, meta in pop if meta.get("structural")]
    assert structural_kids, "structural mode produced zero structural children"
    assert {m["structural_op"] for m in structural_kids} <= set(structural.STRUCTURAL_OPERATORS)
    # Every population member's config must be grammar-valid.
    for _, spec, _ in pop:
        cfg_d = spec.get("config")
        if cfg_d:
            assert not validate(cfg_d), f"invalid config in population: {validate(cfg_d)[:1]}"


def test_structural_mode_adds_grammar_immigrants(tmp_path):
    cfg = EvolutionConfig(population_size=30, elite_keep=4, random_seed=11,
                          min_immigrants_per_family=2,
                          knowledge_graph_path=str(tmp_path / "kg.json"),
                          use_structural_operators=True)
    runner = EvolutionRunner(_trending_df(n=500), RiskConfig(), PropRules(), cfg)
    pop = runner._generate_population(0, [])
    fams = {meta.get("family") for _, _, meta in pop}
    assert "grammar" in fams, "grammar pseudo-family missing from stratified draw"
    n_grammar = sum(1 for _, _, meta in pop if meta.get("family") == "grammar")
    assert n_grammar >= cfg.min_immigrants_per_family


# ---------------------------------------------------------------------------
# 6. batch_runner Stage 2 hook: config surface
# ---------------------------------------------------------------------------

def test_batch_runner_candidate_source_config():
    from app.search.batch_runner import SearchStageConfig
    cfg = SearchStageConfig()
    assert cfg.candidate_source == "templates"
    assert cfg.grammar_candidates_per_survivor >= 0
    custom = SearchStageConfig(candidate_source="grammar", grammar_candidates_per_survivor=3)
    assert custom.candidate_source == "grammar"
    assert custom.grammar_candidates_per_survivor == 3


# ---------------------------------------------------------------------------
# 7. End-to-end: one serial generation with the real holdout gate
# ---------------------------------------------------------------------------

def test_full_generation_serial_with_real_holdout_gate(tmp_path):
    """End-to-end single generation (serial workers, small data) with the
    REAL locked-holdout gate active: whatever the funnel crowns as
    champion must carry a passing holdout_check evaluated on the locked
    slice -- the gate from B1-3 working inside a real _run_one_generation,
    not just against monkeypatched fakes."""
    cfg = EvolutionConfig(
        population_size=6, elite_keep=2, max_generations=1,
        min_immigrants_per_family=1,
        min_trades=8, min_profit_factor=1.1, max_drawdown_buffer_mult=20.0,
        mc_sims=10, robustness_neighbors=1, walk_forward_folds=2,
        cpcv_top_n=2, cpcv_max_paths=2, cpcv_n_groups=3,
        parallel_workers=1,  # serial: the funnel without the process pool
        save_to_library=False,
        knowledge_graph_path=str(tmp_path / "kg.jsonl"),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        tested_log_path=str(tmp_path / "tested.jsonl"),
        target_eval_pass_pct=0.0,  # trivially passing bar -- exercises the gate, not the bar
    )
    runner = EvolutionRunner(_trending_df(n=300), RiskConfig(), PropRules(), cfg, progress_cb=None)
    runner._run_one_generation(0)
    assert len(runner.journal) == 1
    if runner.leaderboard:
        champ = runner.leaderboard[0]
        assert champ.holdout_check is not None, "champion must have been holdout-evaluated"
        assert champ.holdout_check["passed"] in (True, None)
        assert champ.holdout_check["target_eval_pass_pct"] == 0.0
        assert champ.holdout_check["n_bars"] > 0
