"""Session-aware trading-day helper (B2-3, Oct 2026).

CME futures trade nearly around the clock, and the exchange session
day rolls at 17:00 CT (America/Chicago): a bar printed at 18:00 CT on
Monday belongs to *Tuesday's* trading day, and a bar at 16:59 CT on
Monday still belongs to Monday. Any code that groups bars or PnL by
"day" (daily loss limits, winning-day counts, consistency rules) must
use this helper instead of ``Timestamp.date()`` / UTC-midnight day,
or evening-session PnL lands on the wrong day.

The bundled market data is naive UTC (verified in the Oct 2026
analysis), so naive timestamps are assumed UTC unless the importer
stamped something else in ``df.attrs["source_timezone"]``.
"""

from __future__ import annotations

import datetime

import pandas as pd


def trading_day(ts: pd.Timestamp, *, tz: str = "America/Chicago", roll_hour: int = 17) -> datetime.date:
    """Return the trading day a bar timestamp belongs to.

    A trading day rolls at ``roll_hour`` (default 17:00) in ``tz``:
    any bar at or after the roll hour counts toward the *next*
    calendar day's session.

    - Naive timestamps are assumed UTC (the bundled data is naive
      UTC -- see ``app/data/importer.py``'s ``source_timezone`` stamp).
    - tz-aware timestamps are converted to ``tz`` first.

    Examples
    --------
    >>> trading_day(pd.Timestamp("2026-04-15 22:00:00"))  # 17:00 CT, naive->UTC
    datetime.date(2026, 4, 16)
    >>> trading_day(pd.Timestamp("2026-04-15 21:59:00"))  # 16:59 CT, naive->UTC
    datetime.date(2026, 4, 15)

    (22:00 UTC = 17:00 CT in April (CDT, UTC-5), so that bar belongs to
    the April-16 session; 21:59 UTC = 16:59 CT, still April 15.)
    """
    if not 0 <= roll_hour < 24:
        raise ValueError(f"roll_hour must be in [0, 24), got {roll_hour}")

    if ts.tzinfo is None:
        # Naive input is assumed UTC (bundled data is naive UTC).
        localized = ts.tz_localize("UTC")
    else:
        localized = ts

    local = localized.tz_convert(tz)
    # Shift the session boundary to midnight: a bar at roll_hour maps
    # to 00:00 of the following calendar day, so .date() lands on the
    # trading day. E.g. roll_hour=17, 18:00 CT -> +7h -> 01:00 next day.
    return (local + pd.Timedelta(hours=(24 - roll_hour) % 24)).date()


def trading_days(index, *, tz: str = "America/Chicago", roll_hour: int = 17):
    """Vectorized twin of :func:`trading_day` over a whole timestamp
    sequence (v9.13 speed): returns an object ndarray of ``datetime.date``
    elementwise IDENTICAL to ``[trading_day(t, tz=tz, roll_hour=roll_hour)
    for t in index]`` -- same naive-means-UTC assumption, same tz
    conversion, same roll shift, applied column-wise instead of one
    ``pd.Timestamp`` construction (plus a pytz zone lookup) per bar.

    Exists because the bar engine and the Monte Carlo day-structure
    precompute used to call the scalar helper once per bar/trade: on a
    75k-bar frame that loop alone was roughly half the wall-clock of a
    single backtest, and every Full Pipeline step pays it again. The two
    implementations are pinned elementwise-equal by
    tests/test_v913_progress_explorer_speed.py over naive, tz-aware, and
    DST-boundary inputs, so this can never quietly drift from the scalar
    semantics the rest of the app (and its older tests) rely on."""
    if not 0 <= roll_hour < 24:
        raise ValueError(f"roll_hour must be in [0, 24), got {roll_hour}")
    import numpy as np

    idx = index if isinstance(index, pd.DatetimeIndex) else pd.DatetimeIndex(index)
    if idx.tz is None:
        # Naive input is assumed UTC -- same convention as trading_day().
        local = idx.tz_localize("UTC").tz_convert(tz)
    else:
        local = idx.tz_convert(tz)
    return np.asarray((local + pd.Timedelta(hours=(24 - roll_hour) % 24)).date, dtype=object)
