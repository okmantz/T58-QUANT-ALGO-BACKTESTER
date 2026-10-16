import pandas as pd

from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig
from app.prop.presets import get_preset
from app.prop.recommender import recommend_prop_firms, render_recommendation_table
from app.prop.simulator import PropRules


def _make_trades(pnls, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(pnls), freq="D")
    trades = []
    for d, pnl in zip(dates, pnls):
        trades.append(Trade(
            entry_time=d, exit_time=d, direction=1, entry_price=100.0, exit_price=100.0 + pnl,
            size=1.0, pnl=pnl, pnl_pct=pnl / 10000.0, exit_reason="signal", commission=0.0,
            equity_after=10000.0 + pnl,
        ))
    return trades


def test_recommend_returns_empty_for_no_trades():
    assert recommend_prop_firms([]) == []


def test_recommend_scores_every_preset_by_default():
    trades = _make_trades([50, 60, -20, 80, 40, -10, 70, 30] * 5)
    results = recommend_prop_firms(trades, mc_cfg=MonteCarloConfig(n_simulations=100, random_seed=1))
    assert len(results) > 0
    # sorted best-first
    scores = [r.composite_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_accepts_custom_candidates():
    trades = _make_trades([100, 100, 100, -50] * 5)
    custom = [("My Custom Firm", PropRules(account_size=25_000, evaluation_profit_target_pct=5, daily_loss_limit_pct=100, max_drawdown_pct=100))]
    results = recommend_prop_firms(trades, candidates=custom, mc_cfg=MonteCarloConfig(n_simulations=50, random_seed=1))
    assert len(results) == 1
    assert results[0].preset is None
    assert results[0].label == "My Custom Firm"


def test_recommend_preserves_preset_reference():
    trades = _make_trades([50, -10, 60, 20] * 5)
    preset = get_preset("ftmo_100k")
    results = recommend_prop_firms(trades, candidates=[preset], mc_cfg=MonteCarloConfig(n_simulations=50, random_seed=1))
    assert len(results) == 1
    assert results[0].preset.key == "ftmo_100k"


def test_render_recommendation_table_empty():
    assert "No candidates" in render_recommendation_table([])


def test_render_recommendation_table_nonempty():
    trades = _make_trades([50, -10, 60, 20] * 5)
    results = recommend_prop_firms(trades, mc_cfg=MonteCarloConfig(n_simulations=50, random_seed=1))
    table = render_recommendation_table(results)
    assert "Score" in table
    assert results[0].label in table
