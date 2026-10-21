import numpy as np
import pandas as pd

from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchStageConfig, _mean_r_t_stat, _stage1_score
from app.search.run_controls import control_log_lines, random_entry_control


def _walk(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.1, n))
    ts = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 100})


def test_stage1_defaults_and_t_stat():
    cfg = SearchStageConfig()
    assert cfg.min_trades == 100 and cfg.stage1_min_t_stat > 0 and cfg.stage1_max_cost_in_r > 0
    assert _mean_r_t_stat({"total_trades": 100, "average_r": 0.13}) > 0.9
    assert _mean_r_t_stat({"total_trades": 1, "average_r": 1.0}) == float("-inf")


def test_random_entries_rarely_pass_the_stage1_rule():
    df = _walk()
    risk = RiskConfig(initial_balance=50000, risk_value=0.5, pip_size=0.01)
    filters = {"min_trades": 100, "min_profit_factor": 1.05, "max_drawdown_buffer_mult": 1.5, "min_t_stat": 1.0}
    ctl = random_entry_control(df, risk, PropRules(), stop_pips=30, target_pips=60, hold_bars=12, n_entries=150, filters=filters, n_draws=12)
    assert ctl["n_draws"] > 0
    assert ctl["random_pass_rate"] is not None and ctl["random_pass_rate"] <= 0.5
    assert control_log_lines(ctl)[0].startswith("Random-entry control")
