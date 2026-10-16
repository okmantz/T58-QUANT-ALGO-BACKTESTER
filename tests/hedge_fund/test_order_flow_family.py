from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.search.strategy_space import FAMILIES, build_strategy_from_spec, generate_search_space
from app.strategy.family_taxonomy import FAMILY_GROUPS, classify_family
from app.strategy.manual import ManualStrategy


def _volume_spiky_df(n=1500, seed=4):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    price = 100 + np.cumsum(rng.normal(0, 0.3, n))
    volume = np.abs(rng.normal(1000, 400, n))
    spike_idx = rng.choice(n, size=100, replace=False)
    volume[spike_idx] *= 4
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.2, "low": price - 0.2,
        "close": price, "volume": volume,
    })


def test_order_flow_family_registered():
    assert "order_flow_absorption" in FAMILIES
    spec = FAMILIES["order_flow_absorption"]
    assert len(spec.combinations()) > 0
    assert "order flow" in spec.description.lower() or "order-flow" in spec.description.lower()


def test_order_flow_family_classifies_to_its_own_taxonomy_group():
    assert "order_flow" in FAMILY_GROUPS
    assert classify_family(skeleton_family="order_flow_absorption") == "order_flow"


def test_order_flow_family_generates_valid_search_space():
    space = generate_search_space(mode="family", family="order_flow_absorption", max_candidates=5)
    assert len(space.candidates) == 5
    for cand in space.candidates.values():
        strat = build_strategy_from_spec(cand)
        assert strat.source_type == "manual"


def test_order_flow_family_builds_a_runnable_strategy_with_both_conditions():
    params = FAMILIES["order_flow_absorption"].combinations()[0]
    config = FAMILIES["order_flow_absorption"].build(params)
    long_entry = config["entry_conditions"]["long"]
    assert any(c["left"]["type"] == "relative_volume" for c in long_entry)
    assert any(c["left"]["type"] == "volume_delta" for c in long_entry)
    assert config["entry_conditions"]["long_connectors"] == ["AND"]

    strat = ManualStrategy(config)
    df = _volume_spiky_df()
    bt = run_backtest(df, strat, RiskConfig())
    assert bt.statistics.total_trades == len(bt.trades)


def test_order_flow_family_degrades_gracefully_without_volume():
    params = FAMILIES["order_flow_absorption"].combinations()[0]
    config = FAMILIES["order_flow_absorption"].build(params)
    strat = ManualStrategy(config)
    df = _volume_spiky_df().drop(columns=["volume"])
    bt = run_backtest(df, strat, RiskConfig())
    # No real volume column -> relative_volume/volume_delta fall back to
    # flat default series -> the AND-combined threshold conditions should
    # never both fire -> zero trades, not an error.
    assert len(bt.trades) == 0
