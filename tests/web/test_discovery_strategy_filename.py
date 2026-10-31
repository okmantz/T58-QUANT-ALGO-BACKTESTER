"""Discovery saves are named '[instrument] [strategy] [timeframe]' using the bar size the strategy works on, not the data file's bar size."""
from app.discovery_lab.codegen import tf_text
from app.discovery_lab.engine import plan_bar_sizes
from app.web.discovery_engine_routes import strategy_file_base


def test_name_uses_working_timeframe_not_dataset_timeframe():
    assert strategy_file_base("data/NQ1!/NQ1! 1m.csv", "VWAP Mean Reversion", 15) == "NQ1! VWAP Mean Reversion 15m"
    assert strategy_file_base("ES1!_5m.csv", "Trend", 60) == "ES1! Trend 1h"
    assert strategy_file_base("NQ1! 1m", "Trend", None, "4h") == "NQ1! Trend 4h"


def test_tf_text():
    assert [tf_text(m) for m in (1, 15, 60, 240, 1440, 90)] == ["1m", "15m", "1h", "4h", "1d", "90m"]


def test_plan_bar_sizes_spreads_across_ladder():
    import numpy as np
    import pandas as pd
    n = 200_000
    ts = pd.date_range("2022-01-03", periods=n, freq="1min")
    rng = np.random.default_rng(1)
    c = 15000 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1})
    frames, _ = plan_bar_sizes(df, lambda d: 0.0)
    mins = [m for m, _, _ in frames]
    assert len(mins) <= 4 and mins == sorted(mins) and len(set(mins)) == len(mins)
    assert len(mins) >= 3 and mins[0] < mins[-1]
