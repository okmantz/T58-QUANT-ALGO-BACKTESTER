"""Optimizer: sparse-strategy time filter + zero-trade baseline handling + automatic window widening."""
from types import SimpleNamespace

import pandas as pd

from app.optimize import time_filter as tf


def test_suggest_blocked_slots_sparse_cells_with_adaptive_minimum():
    pnl = [[0.0] * 24 for _ in range(7)]
    cnt = [[0] * 24 for _ in range(7)]
    pnl[4][13], cnt[4][13] = -500.0, 3
    pnl[1][10], cnt[1][10] = 50.0, 3
    assert tf.suggest_blocked_slots(pnl, cnt, min_trades=5)["slots"] == []
    assert tf.suggest_blocked_slots(pnl, cnt, min_trades=3)["text"] == "FRI 13"


def test_zero_trade_baseline_explains_instead_of_silent_zero(monkeypatch):
    import app.backtest.engine as eng
    monkeypatch.setattr(eng, "run_backtest", lambda df, s, r: SimpleNamespace(trades=[], statistics=None))
    out = tf.run_time_filter_optimization(pd.DataFrame({"x": range(10)}), object(), SimpleNamespace(blocked_entry_slots=None))
    assert out["no_baseline_trades"] is True and out["improved"] is False
    assert "no trades" in out["note"] and "Widen" in out["note"]


def test_widen_window_grows_until_enough_trades(monkeypatch):
    from app.web import server as srv
    full = pd.DataFrame({"x": range(1000)})
    small = full.tail(100)
    monkeypatch.setattr(srv, "_baseline_trade_count", lambda d, s, r: 0 if len(d) < 400 else 50)
    logs = []
    got = srv._widen_optimize_window(full, small, None, None, logs.append)
    assert len(got) >= 400 and logs


def test_widen_window_noop_when_already_enough(monkeypatch):
    from app.web import server as srv
    full = pd.DataFrame({"x": range(1000)})
    small = full.tail(100)
    monkeypatch.setattr(srv, "_baseline_trade_count", lambda d, s, r: 99)
    assert srv._widen_optimize_window(full, small, None, None, lambda m: None) is small
