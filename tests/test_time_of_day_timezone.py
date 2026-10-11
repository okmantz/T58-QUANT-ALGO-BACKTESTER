"""time_of_day windows: default UTC (back-compat); optional exchange-time via operand['timezone'] incl. DST."""
import pandas as pd

from app.strategy.manual import ManualStrategy


def _frame():
    # 14:30 UTC is 09:30 EST (winter) / 10:30 EDT (summer)
    ts = pd.to_datetime(["2024-01-16 14:30", "2024-07-16 14:30", "2024-07-16 13:30"], utc=True)
    return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})


def _mask(**extra):
    op = {"type": "time_of_day", "session_start": "09:30", "session_end": "09:30", **extra}
    return ManualStrategy({})._series_from_operand(_frame(), op).tolist()


def test_default_is_utc():
    assert _mask() == [0, 0, 0]


def test_new_york_clock_handles_dst():
    assert _mask(timezone="America/New_York") == [1, 0, 1]
