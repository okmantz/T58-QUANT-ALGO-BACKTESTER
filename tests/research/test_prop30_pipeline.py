import numpy as np
import pandas as pd

from app.research.prop30_pipeline import Prop30Config, run, to_markdown


def _synth(days=420, seed=11):
    rng = np.random.default_rng(seed)
    parts = [pd.date_range(d + pd.Timedelta(hours=13), periods=390, freq="1min") for d in pd.bdate_range("2022-01-03", periods=days)]
    ts = parts[0].append(parts[1:])
    c = 15000 + np.cumsum(rng.normal(0, 1.2, len(ts)))
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 0.8, "low": c - 0.8, "close": c, "volume": 1.0})


def test_pipeline_runs_and_cannot_confirm_on_noise():
    cfg = Prop30Config(discovery=("2022-01-01", "2022-12-01"), development=("2022-12-01", "2023-05-01"), holdout=("2023-07-01", "2023-08-20"))
    res = run({"NQ": _synth()}, cfg)
    assert res["verdict"] != "CONFIRMED"
    assert "feasibility_frontier" in res and res["scan"]["n_trials"] > 0
    md = to_markdown(res)
    assert "Honesty notes" in md and "Eventual (chain) Monte Carlo" in md
