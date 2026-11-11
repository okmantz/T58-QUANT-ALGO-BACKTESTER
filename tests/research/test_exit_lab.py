import numpy as np
import pandas as pd

from app.prop.presets import get_preset
from app.research.exit_lab import ExitPolicy, FrozenEntry, atr_series, run_exit_lab, simulate_exit


def _bars(rows):
    o, h, l, c = (np.array(x, float) for x in zip(*rows))
    return o, h, l, c, atr_series(h, l, c)


def test_stop_wins_when_stop_and_target_in_same_bar():
    o, h, l, c, a = _bars([(100, 100, 100, 100), (100, 103, 98, 100)])      # long entry 100, stop 99, target 2R=102: both touched
    r = simulate_exit(o, h, l, c, a, FrozenEntry(0, 1, 100.0, 1.0), ExitPolicy("f", target_r=2.0))
    assert r.reason == "stop" and r.r < 0


def test_gap_through_stop_fills_at_open_not_stop():
    o, h, l, c, a = _bars([(100, 100, 100, 100), (97, 98, 96, 97)])         # opens 3 below a stop 1 below entry
    r = simulate_exit(o, h, l, c, a, FrozenEntry(0, 1, 100.0, 1.0), ExitPolicy("f", target_r=2.0))
    assert abs(r.r - (-3.0)) < 1e-9


def test_slippage_and_cost_reduce_r():
    o, h, l, c, a = _bars([(100, 100, 100, 100), (100, 103, 100, 102)])
    base = simulate_exit(o, h, l, c, a, FrozenEntry(0, 1, 100.0, 1.0), ExitPolicy("f", target_r=2.0))
    paid = simulate_exit(o, h, l, c, a, FrozenEntry(0, 1, 100.0, 1.0), ExitPolicy("f", target_r=2.0), cost_points=0.2)
    assert abs(base.r - 2.0) < 1e-9 and abs(paid.r - 1.8) < 1e-9


def test_breakeven_applies_next_bar_only():
    # bar1 reaches +1R and falls back through entry in the same bar: BE must NOT have protected it (no same-bar look-ahead)
    o, h, l, c, a = _bars([(100, 100, 100, 100), (100, 101.2, 99.5, 100.0), (100, 100, 99.0, 99.0)])
    r = simulate_exit(o, h, l, c, a, FrozenEntry(0, 1, 100.0, 1.0), ExitPolicy("be", "breakeven", target_r=3.0, be_trigger_r=1.0))
    assert r.exit_bar == 2 and r.reason == "stop" and r.r == 0.0


def test_short_side_mirrors_long():
    o, h, l, c, a = _bars([(100, 100, 100, 100), (100, 100, 97.5, 98)])
    r = simulate_exit(o, h, l, c, a, FrozenEntry(0, -1, 100.0, 1.0), ExitPolicy("f", target_r=2.0))
    assert r.reason == "target" and abs(r.r - 2.0) < 1e-9


def test_lab_freezes_entries_and_counts_trials():
    rng = np.random.default_rng(3)
    n = 6000
    ts = pd.date_range("2024-01-02 14:30", periods=n, freq="1min")
    c = 100 + np.cumsum(rng.normal(0, 0.05, n))
    df = pd.DataFrame({"timestamp": ts, "open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": 1.0})
    entries = [FrozenEntry(i, 1, float(c[i]), 0.5, fills_at_open=False) for i in range(100, 5500, 60)]   # priced at the bar's close
    snapshot = list(entries)
    rep = run_exit_lab(df, entries, get_preset("lucid_50k").to_prop_rules(), policies=[ExitPolicy("a", target_r=1.0), ExitPolicy("b", target_r=2.0)])
    assert entries == snapshot and rep.n_trials == 2 and rep.entries_frozen == len(entries)
    few = run_exit_lab(df, entries[:5], get_preset("lucid_50k").to_prop_rules())
    assert few.candidates == [] and few.notes


def test_the_entry_bar_itself_is_traded_for_open_filled_entries():
    """2026-11-11: an entry filled at the open of bar 0 is exposed to that bar's own range; it used to be skipped (loop started at bar + 1)."""
    o, h, l, c, a = _bars([(100, 100.5, 98, 99), (99, 99.5, 98.5, 99)])            # bar 0 trades down through a stop 1 below the 100 fill
    r = simulate_exit(o, h, l, c, a, FrozenEntry(0, 1, 100.0, 1.0), ExitPolicy("f", target_r=2.0))
    assert r.reason == "stop" and r.exit_bar == 0 and abs(r.r - (-1.0)) < 1e-9
    r2 = simulate_exit(o, h, l, c, a, FrozenEntry(0, 1, 100.0, 1.0, fills_at_open=False), ExitPolicy("f", target_r=2.0))
    assert r2.exit_bar == 1                                                           # close-priced entries start on the NEXT bar


def test_entry_bar_target_is_also_credited():
    o, h, l, c, a = _bars([(100, 102.5, 99.8, 102), (102, 102, 101, 101)])
    r = simulate_exit(o, h, l, c, a, FrozenEntry(0, 1, 100.0, 1.0), ExitPolicy("f", target_r=2.0))
    assert r.reason == "target" and r.exit_bar == 0 and abs(r.r - 2.0) < 1e-9
