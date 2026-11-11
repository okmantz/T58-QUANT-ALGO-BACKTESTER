"""'Uploaded strategy raised an exception: index 401 is out of bounds for axis 1 with size 401' -- a cached module holding state sized for a
previous (different-length) DataFrame. The adapter evicts the cached module and retries once on a fresh import."""
import numpy as np
import pandas as pd
import pytest

from app.strategy.base import StrategyError
from app.strategy.python import PythonStrategy, clear_module_cache  # noqa: F401

STATEFUL = """
import numpy as np
_buf = None

def generate_signals(df):
    global _buf
    n = len(df)
    if _buf is None:
        _buf = np.zeros((1, n))           # sized once, on the first call: stale for any other length
    out = np.zeros(n)
    for i in range(n):
        _buf[0, i] = df['close'].iloc[i]  # IndexError once n exceeds the first size
    out[-1] = 1
    return __import__('pandas').Series(out, index=df.index)
"""


def _df(n):
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    c = pd.Series(np.linspace(100, 110, n))
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0})


def test_a_second_run_on_a_longer_frame_survives_stale_module_state(tmp_path, monkeypatch):
    monkeypatch.delenv("T58_MODULE_CACHE", raising=False)
    p = tmp_path / "s.py"
    p.write_text(STATEFUL)
    s = PythonStrategy(p)
    s.generate(_df(50))
    out = s.generate(_df(401 + 20))          # the cached module would raise IndexError here
    assert len(out.signals) == 421


def test_a_genuinely_broken_strategy_still_reports_its_own_error(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("def generate_signals(df):\n    raise ValueError('boom')\n")
    with pytest.raises(StrategyError, match="boom"):
        PythonStrategy(p).generate(_df(30))
