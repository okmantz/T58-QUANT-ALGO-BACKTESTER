"""2026-11-10 audit: skip-rate units, replay calendar days / censoring / mid-signal starts."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig
from app.prop.attempt_replay import AttemptOutcome, AttemptReplayResult, _arm_after_start


def test_skip_rate_counts_signals_not_bars():
    """One unaffordable 10-bar signal run beside 0 taken entries is 100% of SIGNALS; with 4 runs x 10 bars the rate must stay <= 1."""
    n = 200
    ts = pd.date_range("2024-01-01 09:00", periods=n, freq="1h")
    df = pd.DataFrame([(t, 100.0, 100.5, 99.5, 100.0, 1000.0) for t in ts],
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    sig = [0] * n
    for start in (10, 50, 90, 130):
        for j in range(start, start + 10):
            sig[j] = 1
    risk = RiskConfig(initial_balance=50_000.0, risk_value=0.25, pip_size=1.0, contract_size=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, eq = run_execution(df, pd.Series(sig), risk, stop_loss_pips=20, take_profit_pips=None)
    ss = eq.attrs["sizing_summary"]
    assert ss["signals_seen"] == 4
    assert ss["skip_rate"] == 1.0                      # 4 skipped / (4 skipped + 0 taken); bars (40) must not inflate it past 1


def test_arm_after_start_drops_only_the_leading_run():
    s = pd.Series([1, 1, 1, 0, 0, 1, 1, 0])
    out = _arm_after_start(s)
    assert out.tolist() == [0, 0, 0, 0, 0, 1, 1, 0]
    assert _arm_after_start(pd.Series([0, 1, 1])).tolist() == [0, 1, 1]
    assert _arm_after_start(pd.Series([1, 1, 1])).tolist() == [0, 0, 0]


def test_outcome_and_result_expose_calendar_fields_with_safe_defaults():
    f = AttemptOutcome.__dataclass_fields__
    assert "calendar_days_to_pass" in f and "horizon_covered" in f
    r = AttemptReplayResult.__dataclass_fields__
    for k in ("n_truncated", "median_calendar_days_to_pass", "pass_within_days"):
        assert k in r
