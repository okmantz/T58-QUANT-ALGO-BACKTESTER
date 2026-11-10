import numpy as np
import pandas as pd

from app.discovery.opportunity_scanner import (INSTRUMENTS, build_ledger, resample_ohlc, scan, signals_orb, _with_ny_time)
from app.prop.presets import get_preset
from app.research.exit_lab import ExitPolicy, FrozenEntry, atr_series


def _synth(days=260, seed=5, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range("2022-01-03", periods=days):
        idx.append(pd.date_range(d + pd.Timedelta(hours=13), periods=390, freq="1min"))   # 13:00-19:30 UTC covers NY open in all seasons
    ts = idx[0].append(idx[1:]) if len(idx) > 1 else idx[0]
    c = 15000 + np.cumsum(rng.normal(drift, 1.2, len(ts)))
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 0.8, "low": c - 0.8, "close": c, "volume": 1.0})


def test_resample_is_causal_bar_open_labels():
    df = _synth(2)
    b = resample_ohlc(df, 5)
    assert b["timestamp"].is_monotonic_increasing and (b["high"] >= b["low"]).all()


def test_orb_entries_fill_after_the_signal_bar():
    df = _synth(30)
    b = resample_ohlc(df, 5)
    atr = atr_series(b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy())
    ents = signals_orb(_with_ny_time(b), atr, session="ny_open", or_minutes=30)
    assert ents and all(e.stop_dist > 0 and e.bar >= 1 for e in ents)


def test_sizing_skips_when_one_micro_exceeds_risk():
    df = _synth(5)
    b = resample_ohlc(df, 5)
    atr = atr_series(b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy())
    huge = [FrozenEntry(50, 1, float(b["open"].iloc[50]), 5000.0)]                 # 5000 pts x $2 = $10,000 per micro > $500 risk
    led, skipped, _ = build_ledger(b, atr, huge, ExitPolicy("t", target_r=1.0), "NQ", risk_usd=500, max_contracts=4, label="x")
    assert len(led) == 0 and skipped == 1


def test_contract_cap_binds_and_actual_risk_is_below_budget():
    df = _synth(5)
    b = resample_ohlc(df, 5)
    atr = atr_series(b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy())
    e = [FrozenEntry(50, 1, float(b["open"].iloc[50]), 2.0)]                       # 500/(2*2)=125 micros wanted, cap 4
    led, skipped, _ = build_ledger(b, atr, e, ExitPolicy("t", target_r=1.0), "NQ", risk_usd=500, max_contracts=4, label="x")
    assert len(led) == 1 and abs(led.risk[0] - 4 * 2.0 * INSTRUMENTS["NQ"]["micro_pv"]) < 1e-9


def test_scan_never_reads_past_discovery_end_and_counts_trials():
    df = _synth(260)
    cut = pd.Timestamp("2022-09-01")
    rep = scan({"NQ": df}, get_preset("lucid_50k").to_prop_rules(), discovery_end=cut, sessions={"NQ": ["ny_open"]},
               families=[("orb", {"or_minutes": 30})], exits=[ExitPolicy("t1R", target_r=1.0, max_bars=60)], feasibility_sims=200)
    assert rep.n_trials >= 1 and rep.notes and "ES" in rep.skipped_instruments
    assert rep.discovery_end.startswith("2022-09-01")
