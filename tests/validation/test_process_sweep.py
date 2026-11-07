"""Audit finding 2 (2026-11-06): validation sweeps evaluate independent backtests in worker PROCESSES, not threads.

The contract these tests pin: processes give EXACTLY the numbers the serial path gives (same strategy, same data, same
order), the work really goes to a pool when asked, and anything that breaks falls back to serial instead of failing.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.refinement import RefinementConfig
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy
from app.validation import process_sweep
from app.validation.process_sweep import GenomeEvaluator, spec_from_strategy
from app.validation.sensitivity import compute_1d_sensitivity, compute_2d_heatmap
from app.validation.walk_forward_opt import _optimize_on_window
from tests.validation.test_sensitivity import _sma_config, _trending_df


def _inputs():
    return _trending_df(n=1500), ManualStrategy(_sma_config()), RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=30)


def test_decision_rules(monkeypatch):
    monkeypatch.delenv(process_sweep.ENV_FLAG, raising=False)
    assert not process_sweep._decide_parallel(1, 10_000_000)            # one item never needs a pool
    assert not process_sweep._decide_parallel(8, 1_000)                 # tiny job: spawn cost would dominate
    assert process_sweep._decide_parallel(49, 100_000)                  # a 7x7 heatmap on 100k bars
    monkeypatch.setenv(process_sweep.ENV_FLAG, "0")
    assert not process_sweep._decide_parallel(49, 100_000)
    monkeypatch.setenv(process_sweep.ENV_FLAG, "force")
    assert process_sweep._decide_parallel(2, 10)


def test_spec_round_trips_manual_and_rejects_unknown():
    s = spec_from_strategy(ManualStrategy(_sma_config()))
    assert s is not None and s.kind == "manual" and s.manual_config["name"] == "sma cross"

    class Weird:
        source_type = "something_else"

    assert spec_from_strategy(Weird()) is None


def test_failed_pool_falls_back_to_serial(monkeypatch):
    """If the pool dies mid-map, the same genomes are evaluated serially and the answer is unchanged."""
    monkeypatch.setenv(process_sweep.ENV_FLAG, "force")
    df, strategy, risk, rules, mc = _inputs()

    class _Boom:
        def submit(self, *a, **k):
            raise RuntimeError("pool exploded")

        def shutdown(self, *a, **k):
            pass

    ev = GenomeEvaluator(df=df, strategy=strategy, genes=[], risk=risk, prop_rules=rules, mc_config=mc,
                         local_fn=lambda g: sum(g), task="metric", task_args=("net_profit",), n_items=4)
    ev._handle, ev.parallel = _Boom(), True
    assert ev.map([[1, 2], [3, 4], [5, 6]]) == [3, 7, 11]
    assert ev.parallel is False
    ev.close()


def test_sensitivity_1d_processes_match_serial(monkeypatch):
    df, strategy, risk, rules, mc = _inputs()
    monkeypatch.setenv(process_sweep.ENV_FLAG, "0")
    serial = compute_1d_sensitivity(df, strategy, risk, rules, mc, metric="net_profit", n_steps=5, max_params=3)

    monkeypatch.setenv(process_sweep.ENV_FLAG, "force")
    used = []
    real_close = GenomeEvaluator.close

    def spy(self):
        used.append(self.parallel)        # still True here only if EVERY task ran in the pool (no serial fallback happened)
        real_close(self)

    monkeypatch.setattr(GenomeEvaluator, "close", spy)
    progress = []
    par = compute_1d_sensitivity(df, strategy, risk, rules, mc, metric="net_profit", n_steps=5, max_params=3,
                                 progress_cb=progress.append)
    assert used and used[-1] is True, "the forced run must actually have used the worker pool"
    assert [r.gene_label for r in serial] == [r.gene_label for r in par]
    for a, b in zip(serial, par):
        assert a.values == b.values
        assert a.metric_values == pytest.approx(b.metric_values, nan_ok=True)
    assert len(progress) == len(par)                 # one progress line per parameter, still


def test_heatmap_processes_match_serial(monkeypatch):
    df, strategy, risk, rules, mc = _inputs()
    labels = [g for g in __import__("app.validation.sensitivity", fromlist=["x"]).list_tunable_parameters(strategy)]
    a, b = labels[0], labels[1]
    monkeypatch.setenv(process_sweep.ENV_FLAG, "0")
    serial = compute_2d_heatmap(df, strategy, risk, rules, mc, a, b, metric="net_profit", n_steps=3)
    monkeypatch.setenv(process_sweep.ENV_FLAG, "force")
    rows = []
    _states = []
    _real_close = GenomeEvaluator.close
    monkeypatch.setattr(GenomeEvaluator, "close", lambda self: (_states.append(self.parallel), _real_close(self))[1])
    par = compute_2d_heatmap(df, strategy, risk, rules, mc, a, b, metric="net_profit", n_steps=3, progress_cb=rows.append)
    assert serial.a_values == par.a_values and serial.b_values == par.b_values
    for ra, rb in zip(serial.grid, par.grid):
        assert ra == pytest.approx(rb, nan_ok=True)
    assert len(rows) == len(par.a_values)            # one progress line per row
    assert _states == [True], "heatmap cells must have run in the worker pool"


def test_walk_forward_window_processes_match_serial(monkeypatch):
    df, strategy, risk, rules, mc = _inputs()
    cfg = RefinementConfig(enabled=True, fitness_metric="net_profit", population_size=6, generations=2,
                           search_monte_carlo_sims=20)

    def run():
        return _optimize_on_window(df, strategy, risk, rules, mc, cfg, None, random.Random(7))

    monkeypatch.setenv(process_sweep.ENV_FLAG, "0")
    g_serial, bases_serial, w1 = run()
    monkeypatch.setenv(process_sweep.ENV_FLAG, "force")
    g_par, bases_par, w2 = run()
    assert (w1, w2) == (None, None)
    assert g_serial == g_par and bases_serial == bases_par
