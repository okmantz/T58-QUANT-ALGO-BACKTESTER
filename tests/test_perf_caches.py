"""2026-10-15: persistent lookahead cache, loaded-module cache, shared Monte Carlo pool."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest import engine as bt_engine
from app.strategy import lookahead_check as lc
from app.strategy.python import PythonStrategy, clear_module_cache

SRC = "import pandas as pd\n\ndef generate_signals(df):\n    return (df['close'].diff() > 0).astype(int)\n"


def _df(n=3000, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"), "open": 1.0,
        "high": r.random(n) + 2, "low": r.random(n), "close": np.cumsum(r.standard_normal(n)) + 100, "volume": 1,
    })


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("T58_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("T58_LOOKAHEAD_DISK_CACHE", "1")
    bt_engine._LOOKAHEAD_CACHE.clear()
    return tmp_path


def test_lookahead_disk_cache_round_trips_and_is_keyed_on_content_and_data(cache_dir):
    f = cache_dir / "s.py"
    f.write_text(SRC)
    st, df = PythonStrategy(f), _df()
    assert bt_engine._lookahead_disk_load(st, df) is None
    res = lc.check_for_lookahead(st, df)
    assert res.checked
    bt_engine._lookahead_disk_store(st, df, res)
    assert bt_engine._lookahead_disk_load(st, df) == res
    f.write_text(SRC + "# edited\n")                       # same path, different CONTENT -> miss
    assert bt_engine._lookahead_disk_load(PythonStrategy(f), df) is None
    df2 = df.copy()
    df2.loc[5, "close"] += 1.0                             # same length/first/last timestamp, different data -> miss
    assert bt_engine._lookahead_disk_load(st, df2) is None


def test_lookahead_disk_cache_never_stores_skips_or_crashes(cache_dir):
    f = cache_dir / "s.py"
    f.write_text(SRC)
    st, df = PythonStrategy(f), _df()
    bt_engine._lookahead_disk_store(st, df, lc.LookaheadCheckResult(checked=False, bug_detected=False, skip_reason="x"))
    assert bt_engine._lookahead_disk_load(st, df) is None


def test_run_backtest_uses_the_disk_cache_across_a_cold_memory_cache(cache_dir, monkeypatch):
    f = cache_dir / "s.py"
    f.write_text(SRC)
    df = _df()
    from app.backtest.risk import RiskConfig
    calls = []
    real = lc.check_for_lookahead
    monkeypatch.setattr(lc, "check_for_lookahead", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    risk = RiskConfig()
    bt_engine.run_backtest(df, PythonStrategy(f), risk)
    assert len(calls) == 1
    bt_engine._LOOKAHEAD_CACHE.clear()                     # simulate a new process
    bt_engine.run_backtest(df, PythonStrategy(f), risk)
    assert len(calls) == 1, "second 'process' should have read the verdict from disk"


def test_module_is_loaded_once_per_file_version(tmp_path):
    f = tmp_path / "strat.py"
    f.write_text("LOADS = []\nLOADS.append(1)\n" + SRC)
    clear_module_cache()
    st = PythonStrategy(f)
    m1, m2 = st._load_module(), st._load_module()
    assert m1 is m2 and len(m1.LOADS) == 1
    st.module_flag("X"), st.module_attr("Y")
    assert st._load_module() is m1
    f.write_text("LOADS = []\nLOADS.append(1)\n" + SRC + "# edit\n")
    assert st._load_module() is not m1                     # edited file -> fresh module


def test_stateful_modules_and_the_kill_switch_still_reload(tmp_path, monkeypatch):
    f = tmp_path / "strat.py"
    f.write_text("T58_STATEFUL = True\n" + SRC)
    clear_module_cache()
    st = PythonStrategy(f)
    assert st._load_module() is not st._load_module()
    g = tmp_path / "plain.py"
    g.write_text(SRC)
    monkeypatch.setenv("T58_MODULE_CACHE", "0")
    assert PythonStrategy(g)._load_module() is not PythonStrategy(g)._load_module()


def test_shared_pool_is_reused_and_recreated_on_size_change_or_shutdown():
    from app.monte_carlo import shared_pool as sp
    assert sp.get_shared_mc_pool(1) is None and sp.get_shared_mc_pool(None) is None
    p1 = sp.get_shared_mc_pool(2)
    assert p1 is not None and sp.get_shared_mc_pool(2) is p1
    p2 = sp.get_shared_mc_pool(3)
    assert p2 is not p1
    sp.shutdown_shared_mc_pool()
    assert sp.get_shared_mc_pool(3) is not p2
    sp.shutdown_shared_mc_pool()
