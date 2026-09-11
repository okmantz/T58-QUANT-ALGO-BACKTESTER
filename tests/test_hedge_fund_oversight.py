import numpy as np
import pandas as pd

from app.ai.ollama_settings import OllamaSettings
from app.hedge_fund.oversight import build_deterministic_summary, generate_journal
from app.hedge_fund.rebalancer import RebalanceConfig, run_rebalance_backtest
from app.hedge_fund.research import EnsembleForecastConfig


def _df(n=200, seed=1, drift=0.0005):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D")
    price = 100.0
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.01)
        o = price
        c = o * (1 + step)
        rows.append((ts[i], o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1000))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _result():
    price_data = {"AAA": _df(seed=1), "BBB": _df(seed=2)}
    config = RebalanceConfig(
        rebalance_every_bars=10, forecast=EnsembleForecastConfig(lookback_bars=60, horizon_bars=5, n_samples=8, seed=1),
    )
    return run_rebalance_backtest(price_data, config), config


def test_deterministic_summary_contains_headline_stats_only_from_result():
    result, config = _result()
    summary = build_deterministic_summary(result, config)
    assert f"{result.stats.num_rebalances}" in summary
    assert "%" in summary


def test_generate_journal_falls_back_to_deterministic_when_ollama_not_configured():
    result, config = _result()
    disabled_settings = OllamaSettings(enabled=False, host="", model="", api_key="")
    journal, note = generate_journal(result, config, settings=disabled_settings)
    assert journal == build_deterministic_summary(result, config)
    assert note is not None and "Ollama isn't configured" in note


def test_oversight_module_has_no_order_generation_capability():
    """Guardrail test: the oversight module must never IMPORT anything
    that can PRODUCE orders/weights, only consume an already-finished
    result. Checked via the actual import graph (ast), not a substring
    search of the source -- the module's own docstring names these
    functions on purpose, to document the boundary, which would trip a
    naive text search. If this starts failing because someone added a
    real import of one of these, that is the bug -- not the test."""
    import ast
    import inspect

    import app.hedge_fund.oversight as oversight_module

    forbidden = {"weights_to_orders", "solve_posterior_weights", "views_to_posterior", "run_rebalance_backtest"}
    imported_names = set()
    tree = ast.parse(inspect.getsource(oversight_module))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)

    assert not (imported_names & forbidden), f"oversight.py imported order-capable names: {imported_names & forbidden}"
