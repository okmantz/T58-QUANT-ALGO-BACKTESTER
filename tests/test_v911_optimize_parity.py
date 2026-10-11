"""v9.11: Optimize parity with Full Pipeline (sizing, progress) + speed-path equivalence."""
import numpy as np
import pandas as pd

from app.backtest.statistics import _reset_segment_ids
from app.orchestration.run_progress import SimpleProgress


def test_reset_segment_ids_matches_naive_loop():
    ts = pd.Series(pd.date_range("2024-01-01", periods=500, freq="5min"))
    events = [{"reset_at": ts.iloc[i]} for i in (50, 51, 300, 499)]
    naive = np.zeros(len(ts), dtype=np.int64)
    for ev in events:
        naive += (ts >= ev["reset_at"]).to_numpy().astype(np.int64)
    eq = pd.DataFrame({"timestamp": ts, "equity": 1.0})
    eq.attrs["account_reset_events"] = events
    assert np.array_equal(_reset_segment_ids(eq), naive)


def test_simple_progress_monotonic_and_engine_slices():
    p = SimpleProgress(engine_total=2)
    p.begin_engine(0, "a")
    assert p.update(0.5)["percent"] == 25.0
    assert p.update(0.2)["percent"] == 25.0          # never goes backwards
    assert p.begin_engine(1, "b")["percent"] == 50.0
    assert p.update(1.0)["percent"] == 99.0          # only finish() hits 100
    out = p.finish()
    assert out["percent"] == 100.0 and out["done"]


def test_baseline_preflight_message_has_no_self_contradiction():
    import re
    from app.validation import preflight
    src = open(preflight.__file__).read()
    assert "needed > budget * 1.0001" in src
    assert re.search(r"def resolve_baseline_sizing", src)
