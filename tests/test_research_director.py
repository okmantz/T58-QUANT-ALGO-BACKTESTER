import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
import app.research.director as d


def _synthetic_df(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0006, n))
    return pd.DataFrame({
        "timestamp": idx,
        "open": close + rng.normal(0, 0.0001, n),
        "high": close + np.abs(rng.normal(0, 0.0004, n)),
        "low": close - np.abs(rng.normal(0, 0.0004, n)),
        "close": close,
        "volume": rng.integers(100, 1000, n),
    })


def _manual_spec():
    return {
        "source_type": "manual",
        "config": {
            "entry_conditions": {
                "long": [
                    {"left": {"type": "time_of_day", "session_start": "08:30", "session_end": "11:00"}, "operator": "==", "right": 1},
                    {"left": {"type": "rsi", "period": 14}, "operator": "<", "right": 30},
                ],
                "long_connectors": ["AND"],
                "short": [],
            },
            "exit_conditions": {"long": [], "short": []},
            "stop_loss_pips": 20,
            "take_profit_pips": 30,
        },
    }


@pytest.fixture
def env():
    df = _synthetic_df()
    spec = _manual_spec()
    risk = RiskConfig(initial_balance=100_000, pip_size=0.0001)
    rules = PropRules(account_size=100_000)
    bt = d._run_spec(spec, df, risk)
    assert bt is not None and bt.trades, "fixture strategy should generate at least one trade"
    return df, spec, risk, rules, bt


def test_row_shape(env):
    df, spec, risk, rules, bt = env
    row = d._row("x", bt, rules, 20)
    for key in ("label", "total_trades", "expectancy_r", "profit_factor", "net_profit", "max_drawdown_pct", "pass_rate_pct"):
        assert key in row
    assert row["total_trades"] == len(bt.trades)


def test_edge_decomposition_baseline_never_empty(env):
    df, spec, risk, rules, bt = env
    result = d.edge_decomposition(spec, df, risk, rules, window_trading_days=20)
    steps = result["steps"]
    assert len(steps) >= 2
    # Baseline step must have generated trades -- the setup-role fallback
    # (treat the first declared condition as "setup" when nothing was
    # classified that way) exists specifically to prevent this being 0.
    assert steps[0]["row"]["total_trades"] > 0
    assert isinstance(result["verdict"], str) and result["verdict"]


def test_edge_decomposition_requires_manual_or_variants():
    df = _synthetic_df(n=200)
    risk, rules = RiskConfig(), PropRules()
    with pytest.raises(ValueError):
        d.edge_decomposition({"source_type": "python", "code_text": "x"}, df, risk, rules)


def test_ablation_returns_full_strategy_row_first(env):
    df, spec, risk, rules, bt = env
    result = d.ablation_test(spec, df, risk, rules, window_trading_days=20)
    assert result["rows"][0]["label"] == "Full strategy"
    assert len(result["rows"]) > 1  # at least one condition was ablated


def test_null_baselines_covers_all_builtins(env):
    df, spec, risk, rules, bt = env
    target_row = d._row("target", bt, rules, 20)
    result = d.null_baselines(df, risk, rules, target_row, window_trading_days=20)
    labels = {r["label"] for r in result["baselines"]}
    assert labels == set(d._NULL_BUILDERS.keys())
    assert result["edge_contribution"] in {None, "LOW", "MODERATE", "HIGH"}


def test_signal_degradation_baseline_matches_normal_run(env):
    df, spec, risk, rules, bt = env
    result = d.signal_degradation(spec, df, risk, rules, window_trading_days=20)
    assert result["baseline"]["total_trades"] == len(bt.trades)
    assert len(result["stress_tests"]) == 6
    assert result["execution_fragility_score"] is None or 0 <= result["execution_fragility_score"] <= 100


def test_trade_contribution_flags_concentration():
    from app.backtest.execution import Trade
    trades = []
    base_time = pd.Timestamp("2024-01-01")
    # 9 tiny losers + 1 huge winner -> concentration flag should fire.
    for i in range(9):
        trades.append(Trade(entry_time=base_time + pd.Timedelta(days=i), exit_time=base_time + pd.Timedelta(days=i, hours=1),
                             direction=1, entry_price=1.0, exit_price=0.999, size=1, pnl=-10, pnl_pct=-0.1,
                             exit_reason="stop_loss", commission=0, equity_after=100_000, initial_risk=0.001))
    trades.append(Trade(entry_time=base_time + pd.Timedelta(days=20), exit_time=base_time + pd.Timedelta(days=20, hours=1),
                         direction=1, entry_price=1.0, exit_price=1.05, size=1, pnl=2000, pnl_pct=5.0,
                         exit_reason="take_profit", commission=0, equity_after=102_000, initial_risk=0.001))
    result = d.trade_contribution(trades, initial_balance=100_000)
    assert result["n_trades"] == 10
    assert any("concentration" in f.lower() or "best" in f.lower() for f in result["flags"])


def test_conditional_expectancy_buckets(env):
    df, spec, risk, rules, bt = env
    result = d.conditional_expectancy(bt.trades, df)
    assert "by_hour" in result and isinstance(result["by_hour"], list)
    total_hour_trades = sum(row["n_trades"] for row in result["by_hour"])
    assert total_hour_trades == len(bt.trades)


def test_regime_discovery_runs_holdout_check(env):
    df, spec, risk, rules, bt = env
    holdout = df.iloc[-500:].reset_index(drop=True)
    result = d.regime_discovery(bt.trades, df, risk, rules, spec, holdout, window_trading_days=20)
    assert "next_step" in result
    assert "hypothesis" in result


def test_research_director_report_synthesizes_families():
    rng = np.random.default_rng(3)
    candidates = [
        {"family": "fam_strong" if i % 4 == 0 else "fam_weak", "passed_stage3_gate": (i % 4 == 0), "composite_score": rng.random(), "max_drawdown_pct": rng.random() * 10}
        for i in range(60)
    ]
    report = d.research_director_report(candidates)
    assert report["n_candidates"] == 60
    assert len(report["family_stats"]) == 2
    assert isinstance(report["summary"], str) and report["summary"]


def test_research_director_report_empty_input():
    assert "note" in d.research_director_report([])
