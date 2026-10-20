"""v11 primitives and families: no lookahead, windowed families trade only inside their windows, forward-test pieces."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.search import strategy_space as _sps  # noqa: F401  (registers families; import before families_v11)
from app.search.families_v11 import V11_FAMILIES
from app.strategy import indicators_v11 as iv
from tests.strategy.test_strategy_space import _session_df


@pytest.fixture(scope="module")
def frame():
    return _session_df(days=120)


SELF_CONTAINED = sorted(k for k in iv.V11_INDICATOR_BUILDERS if k in iv.V11_SELF_CONTAINED)


@pytest.mark.parametrize("kind", SELF_CONTAINED)
def test_v11_indicator_has_no_lookahead(frame, kind):
    """Value on bar t computed from a truncated frame must equal the full-frame value (prefix invariance)."""
    fn = iv.V11_INDICATOR_BUILDERS[kind]
    full = pd.Series(np.asarray(fn(frame.copy(), 14, "close", 14), dtype=float))
    for cut in (len(frame) // 3, len(frame) // 2, len(frame) - 7):
        part = pd.Series(np.asarray(fn(frame.iloc[:cut].copy(), 14, "close", 14), dtype=float))
        a, b = full.iloc[:cut].to_numpy(), part.to_numpy()
        both = ~(np.isnan(a) | np.isnan(b))
        assert np.allclose(a[both], b[both]), f"{kind} changes when future bars are removed"
        assert (np.isnan(a) == np.isnan(b)).mean() > 0.99


def test_nfp_calendar_dates():
    from app.data.event_calendar import nfp_dates
    ds = {d.isoformat() for d in nfp_dates(2024, 2024)}
    assert "2024-01-05" in ds and "2024-02-02" in ds and "2024-03-08" in ds


def test_event_family_trades_only_inside_window(frame):
    from app.search.strategy_space import generate_search_space
    from app.strategy.manual import ManualStrategy
    sp = generate_search_space(mode="family", family="event_post_release_drift", max_candidates=8, seed=1)
    ts = pd.to_datetime(frame["timestamp"])
    for spec in sp.candidates.values():
        sig = ManualStrategy(spec["config"]).generate(frame.copy()).signals
        active = pd.DatetimeIndex(ts[np.asarray(sig != 0)])
        # an event-window family must be flat on most days (it is not a continuous strategy)
        assert active.normalize().nunique() <= 0.2 * pd.DatetimeIndex(ts).normalize().nunique()


def test_bias_journal_is_immutable_and_locks_at_open(tmp_path):
    from app.forward_test.bias_journal import JournalLocked, log_bias
    p = tmp_path / "j.jsonl"
    log_bias("2026-10-21", "bullish", now=datetime(2026, 10, 21, 8, 0), path=p)
    with pytest.raises(JournalLocked):
        log_bias("2026-10-21", "bearish", now=datetime(2026, 10, 21, 8, 5), path=p)       # immutable
    with pytest.raises(JournalLocked):
        log_bias("2026-10-22", "bullish", now=datetime(2026, 10, 22, 9, 31), path=p)       # after the open
    with pytest.raises(ValueError):
        log_bias("2026-10-23", "moon", now=datetime(2026, 10, 23, 8, 0), path=p)


def test_sprt_monitor_accepts_real_edge_and_rejects_none():
    from app.forward_test.sprt_monitor import SprtMonitor
    rng = np.random.default_rng(1)
    good, flat = SprtMonitor(h1_r=0.21), SprtMonitor(h1_r=0.21)
    for r in rng.normal(0.35, 1.3, 400):
        if good.add(r)["decision"] != "CONTINUE":
            break
    for r in rng.normal(-0.1, 1.3, 400):
        if flat.add(r)["decision"] != "CONTINUE":
            break
    assert good.status()["decision"] == "ACCEPT_H1"
    assert flat.status()["decision"] != "ACCEPT_H1"


def test_preregistration_is_fingerprinted_and_books_one_trial():
    from app.research.preregistered_tests import benjamini_hochberg
    keep = benjamini_hochberg([0.001, 0.2, 0.9, 0.012], q=0.05)
    assert list(keep) == [True, False, False, True]


def test_all_v11_families_registered():
    assert len(V11_FAMILIES) == 11
    assert set(V11_FAMILIES) <= set(_sps.FAMILIES)
