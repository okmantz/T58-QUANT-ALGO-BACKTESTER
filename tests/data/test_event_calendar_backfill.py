import pandas as pd

from app.data.event_calendar import FOMC_DECISION_DATES, builtin_events


def test_fomc_backfilled_2010_2026_structure():
    d = pd.to_datetime(list(FOMC_DECISION_DATES))
    assert d.min().year == 2010 and d.max().year == 2026
    assert d.is_monotonic_increasing and d.is_unique
    assert (d.dayofweek < 5).all()
    per_year = pd.Series(d.year).value_counts()
    assert (per_year.drop(2020) == 8).all() and per_year[2020] == 7   # 2020: two emergency actions excluded


def test_builtin_events_default_covers_2010():
    ev = builtin_events()
    assert ev["timestamp"].min().year == 2010
    assert set(ev["event"]) == {"FOMC", "NFP"}
    assert (ev[ev.event == "FOMC"]["timestamp"].dt.hour == 14).all()
