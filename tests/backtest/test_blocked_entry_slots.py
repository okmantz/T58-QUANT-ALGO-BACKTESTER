import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.backtest.time_slots import parse_slots, merge_slot_text, slot_mask_for_timestamps
from tests.backtest.test_backtest import _sma_strategy


def _df(n=2500, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    price = 1.1
    rows = []
    for i in range(n):
        o = price
        c = o + rng.normal(0, 0.0004)
        rows.append((ts[i], o, max(o, c) + 0.0002, min(o, c) - 0.0002, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_parse_and_merge():
    assert parse_slots("THU 11\nFRI *, * 15, MON 9-10") == [(3, (11, 11)), (4, None), (None, (15, 15)), (0, (9, 10))]
    assert merge_slot_text("THU 11", "thu 11\nFRI *") == "THU 11\nFRI *"


def test_mask_hits_expected_bars():
    df = _df(400)
    m = slot_mask_for_timestamps(df["timestamp"], "WED 10")
    ts = pd.DatetimeIndex(df["timestamp"])
    assert m.sum() > 0
    assert all((ts[m].dayofweek == 2) & (ts[m].hour == 10))


def test_engine_never_enters_in_blocked_slot():
    df = _df()
    base = run_backtest(df, _sma_strategy(), RiskConfig())
    trades = base.trades if hasattr(base, "trades") else base["trades"]
    assert len(trades) > 20
    # pick the busiest weekday/hour bucket and block it
    from collections import Counter
    c = Counter((pd.Timestamp(t.entry_time).dayofweek, pd.Timestamp(t.entry_time).hour) for t in trades)
    (d, h), n_in = c.most_common(1)[0]
    names = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    blocked = run_backtest(df, _sma_strategy(), RiskConfig(blocked_entry_slots=f"{names[d]} {h}"))
    bt = blocked.trades if hasattr(blocked, "trades") else blocked["trades"]
    assert n_in > 0
    assert all(not (pd.Timestamp(t.entry_time).dayofweek == d and pd.Timestamp(t.entry_time).hour == h) for t in bt)
