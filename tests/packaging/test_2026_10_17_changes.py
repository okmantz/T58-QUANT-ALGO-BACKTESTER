"""2026-10-17 changes: READY threshold, shared stage pool, genome memo keys, trade-list PBO, firm fees, roll-gap gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.optimize.genome_cache import MemoTable, genome_key
from app.orchestration import stage_pool as sp
from app.prop.presets import FIRM_FEES, PROP_FIRM_PRESETS
from app.scoring.t58_scorecard import ELITE_MIN_SCORE, READY_MIN_SCORE, tier_for_score


# ---- READY means "cleared the gates with score >= 75"; Elite (>= 85) is only a label ---------------------------------
def test_ready_threshold_is_75_and_elite_is_85():
    assert READY_MIN_SCORE == 75.0 and ELITE_MIN_SCORE == 85.0
    assert tier_for_score(75.0) == "Strong" and tier_for_score(84.9) == "Strong"
    assert tier_for_score(85.0) == "Elite"
    assert tier_for_score(74.9) == "Promising" and tier_for_score(64.9) == "Reject"


# ---- every preset now has a fee row (None price = unknown, never a guess) --------------------------------------------
def test_every_preset_has_a_fee_row_and_unknown_prices_are_none():
    missing = [p.key for p in PROP_FIRM_PRESETS if p.key not in FIRM_FEES]
    assert not missing
    for key, row in FIRM_FEES.items():
        assert row.preset_key == key and row.date_checked and row.source_url
        assert row.challenge_price is None or row.challenge_price > 0
        assert row.written_confirmation is False                  # never auto-confirmed
    assert FIRM_FEES["topstep_100k"].monthly_subscription is True


# ---- memo keys: integer genes that round equal share one backtest ------------------------------------------------------
class _Gene:
    def __init__(self, is_int):
        self.is_int = is_int


def test_genome_key_rounds_integer_genes_only():
    genes = [_Gene(True), _Gene(False)]
    assert genome_key(genes, [12.2, 0.5]) == genome_key(genes, [11.8, 0.5])
    assert genome_key(genes, [12.2, 0.5]) != genome_key(genes, [12.2, 0.6])
    assert genome_key(None, [1, 2]) == (1.0, 2.0)


def test_memo_table_computes_once_per_key():
    memo, calls = MemoTable(), []
    for _ in range(3):
        assert memo.get_or_compute("k", lambda: calls.append(1) or 7) == 7
    assert len(calls) == 1 and memo.hits == 2 and memo.misses == 1


# ---- shared stage pool ---------------------------------------------------------------------------------------------------
_STATE: dict = {}


def _init_df(df, tag):
    _STATE["df"], _STATE["tag"] = df, tag


def _task_df(i):
    return _STATE["tag"], int(_STATE["df"]["x"].iloc[i])


def _init_tag(tag):
    _STATE.clear()
    _STATE["tag"] = tag


def _task_tag(i):
    return _STATE["tag"], i


def test_stage_pool_reuses_workers_across_stages_and_spills_frames():
    try:
        df = pd.DataFrame({"x": np.arange(50)})
        h1 = sp.acquire_stage(2, _init_df, (df, "A"))
        assert h1 is not None
        assert sorted(h1.map(_task_df, range(4))) == [("A", 0), ("A", 1), ("A", 2), ("A", 3)]
        pids = set(sp._pool._processes)
        h1.shutdown()
        h2 = sp.acquire_stage(2, _init_tag, ("B",))
        assert sorted(h2.map(_task_tag, range(3))) == [("B", 0), ("B", 1), ("B", 2)]
        assert set(sp._pool._processes) == pids                        # same worker processes, new stage
        h2.shutdown(cancel_futures=True)                              # the Stop path discards the pool
        assert sp._pool is None
    finally:
        sp.shutdown_stage_pool(wait=True)


def test_stage_pool_is_none_for_a_single_worker():
    assert sp.acquire_stage(1, _init_tag, ("x",)) is None and sp.acquire_stage(None, _init_tag, ("x",)) is None


# ---- PBO from precomputed trade lists agrees with the original per-path backtests --------------------------------------
def _trend_df(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    c = 100 + np.cumsum(rng.normal(0.002, 0.08, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"timestamp": ts, "open": o, "high": np.maximum(o, c) + 0.02, "low": np.minimum(o, c) - 0.02,
                         "close": c, "volume": 100.0})


def _sma(f, s, sl, tp):
    return {"source_type": "manual", "config": {
        "name": f"sma {f}/{s}",
        "indicators": [{"type": "sma", "period": f, "column": "close", "as": "sma_fast"},
                       {"type": "sma", "period": s, "column": "close", "as": "sma_slow"}],
        "long_entry": "sma_fast > sma_slow", "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow", "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": sl, "take_profit_pips": tp}}


def test_pbo_from_trade_lists_matches_the_per_path_method_in_shape():
    from app.backtest.engine import run_backtest
    from app.backtest.risk import RiskConfig, build_run_context
    from app.monte_carlo.engine import MonteCarloConfig
    from app.prop.simulator import PropRules
    from app.search.strategy_space import build_strategy_from_spec
    from app.validation.cpcv import compute_pbo

    df = _trend_df()
    specs = [_sma(5, 15, 100, 200), _sma(10, 30, 150, 300), _sma(12, 40, 300, 600)]
    rules = PropRules(evaluation_profit_target_pct=3.0, min_trading_days=1, consistency_rule_pct=None,
                      max_drawdown_pct=6.0, daily_loss_limit_pct=3.0)
    risk = build_run_context(RiskConfig(), rules)
    mc = MonteCarloConfig(n_simulations=20, random_seed=3)
    trades = [run_backtest(df, build_strategy_from_spec(s, None), risk).trades for s in specs]
    kw = dict(n_groups=6, n_test_groups=2, max_paths=5, metric="eval_pass_probability", prop_rules=rules, mc_cfg=mc)
    fast = compute_pbo(df, specs, risk, precomputed_trades=trades, **kw)
    assert fast.n_paths == 5 and fast.n_candidates == 3 and 0.0 <= fast.pbo <= 1.0
    assert "trade lists" in fast.note
    # a missing trade list silently falls back to the original method
    slow = compute_pbo(df, specs, risk, precomputed_trades=[trades[0], None, trades[2]], max_workers=1, **kw)
    assert "trade lists" not in slow.note
