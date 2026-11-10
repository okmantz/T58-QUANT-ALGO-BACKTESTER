import numpy as np
import pandas as pd

from app.portfolio.prop_portfolio import (allocate_weights, apply_shared_risk, correlation_matrix, daily_pnl_matrix, regime_gate)
from app.prop.presets import get_preset
from app.prop.thirty_day import TradeLedger


def _led(times, pnls, hold_min=30, name="s", risk=400.0):
    et = np.array(pd.to_datetime(times), dtype="datetime64[ns]")
    xt = et + np.timedelta64(hold_min, "m")
    p = np.asarray(pnls, float)
    z = np.zeros(len(p))
    return TradeLedger(et, xt, p, np.minimum(p, 0), np.maximum(p, 0), z + 4, z + risk, np.ones(len(p), int),
                       np.array([name] * len(p), dtype=object), np.array(["MNQ"] * len(p), dtype=object))


def test_concurrency_cap_skips_overlapping_trades():
    led = _led(["2024-01-02 15:00", "2024-01-02 15:05", "2024-01-02 15:10", "2024-01-02 17:00"], [100, 100, 100, 100], hold_min=60)
    out, st = apply_shared_risk(led, max_concurrent=2)
    assert len(out) == 3 and st.skipped_concurrency == 1


def test_open_risk_cap():
    led = _led(["2024-01-02 15:00", "2024-01-02 15:05"], [100, 100], hold_min=60, risk=600)
    out, st = apply_shared_risk(led, max_concurrent=5, max_open_risk_usd=1000)
    assert len(out) == 1 and st.skipped_open_risk == 1


def test_daily_stop_uses_only_exited_trades():
    led = _led(["2024-01-02 15:00", "2024-01-02 16:00", "2024-01-02 17:00"], [-900, 50, 50], hold_min=10)
    out, st = apply_shared_risk(led, max_concurrent=5, daily_stop_usd=800)
    assert len(out) == 1 and st.skipped_daily_stop == 2


def test_regime_gate_is_causal():
    led = _led(["2024-01-03 15:00", "2024-01-04 15:00"], [10, 10])
    labels = pd.Series(["hi", "lo"], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))    # day-before label drives each entry
    out = regime_gate(led, labels, ["hi"])
    assert len(out) == 1 and pd.Timestamp(out.entry_time[0]).day == 3
    same_day = pd.Series(["hi"], index=pd.to_datetime(["2024-01-03"]))
    kept = regime_gate(led, same_day, ["hi"])                # the 01-03 label is unknown to a 01-03 entry, known to the 01-04 one
    assert len(kept) == 1 and pd.Timestamp(kept.entry_time[0]).day == 4


def test_correlation_and_allocation_run():
    d = pd.bdate_range("2024-01-02", periods=80)
    a = _led([x + pd.Timedelta(hours=15) for x in d], [150 if i % 3 else -100 for i in range(80)], name="a")
    b = _led([x + pd.Timedelta(hours=16) for x in d], [120 if i % 4 else -90 for i in range(80)], name="b")
    m = daily_pnl_matrix({"a": a, "b": b})
    assert m.shape[1] == 2
    assert correlation_matrix({"a": a, "b": b}).shape == (2, 2)
    rep = allocate_weights({"a": a, "b": b}, get_preset("lucid_50k").to_prop_rules(), grid=(0.0, 1.0),
                           data_start=d[0] - pd.Timedelta(days=1), data_end=d[-1] + pd.Timedelta(days=70))
    assert rep.n_trials == 3 and rep.best() is not None and rep.notes
