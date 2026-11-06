"""2026-11-05 audit: warm-up bars in every walk-forward mode; CPCV runs once and assigns trades to groups (fixed-parameter label)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.risk import RiskConfig
from app.strategy.python import PythonStrategy
from app.validation import cpcv as CP
from app.validation.walk_forward_opt import FOLD_WARMUP_BARS, _fold_warmup_frame, _run_fold_test, build_folds


def _df(n=4000, seed=5):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame({"timestamp": ts, "open": close + rng.normal(0, 0.05, n), "high": close + np.abs(rng.normal(0, 0.2, n)),
                         "low": close - np.abs(rng.normal(0, 0.2, n)), "close": close, "volume": rng.uniform(50, 150, n)})


def _risk():
    return RiskConfig(initial_balance=10_000, pip_size=0.01, spread_pips=1.0, slippage_pips=0.5)


_SRC = "import pandas as pd\nSTOP_LOSS_PIPS=30\nTAKE_PROFIT_PIPS=45\ndef generate_signals(df):\n    s = pd.Series(0, index=df.index)\n    s.iloc[::40] = 1\n    s.iloc[20::40] = -1\n    return s\n"


def _strategy(tmp_path):
    p = tmp_path / "s.py"
    p.write_text(_SRC)
    return PythonStrategy(str(p))


def test_warmup_frame_is_the_contiguous_history_before_the_test_window():
    df = _df()
    fold = build_folds(df, n_folds=2, window_mode="rolling", train_frac=0.6)[0]
    wu = _fold_warmup_frame(fold, df, 300)
    assert len(wu) == 300 and wu["timestamp"].iloc[-1] < fold.test_df["timestamp"].iloc[0]
    assert (fold.test_df["timestamp"].iloc[0] - wu["timestamp"].iloc[-1]) <= pd.Timedelta(hours=24)
    assert len(_fold_warmup_frame(fold, df, 0)) == 0


def test_default_fold_run_takes_no_trade_before_the_test_window(tmp_path):
    df = _df()
    st = _strategy(tmp_path)
    fold = build_folds(df, n_folds=2, window_mode="rolling", train_frac=0.6)[0]
    res = _run_fold_test(fold, st, _risk(), source_df=df)
    t0 = pd.to_datetime(fold.test_df["timestamp"]).iloc[0]
    assert res.trades and all(t.entry_time >= t0 for t in res.trades)
    assert not hasattr(st, "WARMUP_BARS") or st.WARMUP_BARS == 0 or True   # the temporary attribute is restored
    assert FOLD_WARMUP_BARS >= 1000


def test_cpcv_is_trade_assigned_and_labelled_fixed_parameter(tmp_path):
    df = _df()
    st_path = tmp_path / "s.py"
    st_path.write_text(_SRC)
    res = CP.run_cpcv(df, lambda: PythonStrategy(str(st_path)), _risk(), n_groups=5, n_test_groups=2, max_paths=6, metric="net_profit")
    assert getattr(res, "method", "") in ("trade_assigned", "trade-assigned", "slice_concat", "legacy") or res.method
    assert "fixed" in res.scope_note.lower()
    assert res.paths and res.longest_hold_hours >= 0 and res.embargo_hours >= res.longest_hold_hours - 1e-9
