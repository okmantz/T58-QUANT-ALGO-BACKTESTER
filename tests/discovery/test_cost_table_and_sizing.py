"""One INST cost table; commission vs conditional slippage; mini/micro sizing; roll sessions; stress mode."""
import numpy as np
import pandas as pd
import pytest

from app.data.inst import ALPHA_COMMISSION_RT, INST, get_inst
from app.data.instrument_specs import KNOWN_INSTRUMENTS
from app.data.roll_handling import back_adjust, detect_roll_bars, roll_session_dates, session_ordinals
from app.discovery.opportunity_scanner import INSTRUMENTS, build_ledger, resample_ohlc, scan
from app.prop.presets import get_preset
from app.research.exit_lab import ExitPolicy, FrozenEntry, atr_series, simulate_exit


def test_commissions_match_the_alpha_schedule():
    for sym, rt in {"NQ": 4.78, "ES": 4.78, "MGC": 1.82, "MNQ": 1.32, "MES": 1.32, "GC": 5.22, "CL": 5.02, "ZN": 3.58, "6E": 5.22}.items():
        assert INST[sym].comm_usd == rt
    assert ALPHA_COMMISSION_RT["NKD"] == 6.32 and ALPHA_COMMISSION_RT["UB"] == 3.92 and ALPHA_COMMISSION_RT["ZC"] == 6.28
    assert get_inst("NQ", comm_override=3.78).comm_usd == 3.78 and INST["NQ"].comm_usd == 4.78


def test_registry_and_scanner_derive_from_the_one_table():
    for sym, i in INST.items():
        spec = KNOWN_INSTRUMENTS[sym]
        assert spec.default_commission_round_turn == i.comm_usd
        assert spec.default_slippage_ticks == i.slip_ticks
        assert spec.tick_size == pytest.approx(i.tick) and spec.tick_value == pytest.approx(i.tick_value)
    nq = INSTRUMENTS["NQ"]
    assert nq["cost_points"] == pytest.approx(INST["MNQ"].comm_usd / INST["MNQ"].pv)
    assert nq["mini_cost_points"] == pytest.approx(INST["NQ"].comm_usd / INST["NQ"].pv)
    assert nq["slip_points"] == pytest.approx(0.25) and nq["target_slip_points"] == 0.0


def _bars(path):
    o = np.array(path, float)
    return o, o + 0.3, o - 0.3, o


def test_slippage_is_conditional_on_the_order_type():
    # long from 100, stop 1.0, target 2R: price rises to the target
    o = np.array([100.0, 100.0, 102.5]); h = np.array([100.1, 100.5, 102.6]); l = np.array([99.9, 99.9, 102.0]); c = o
    e = FrozenEntry(0, 1, 100.0, 1.0)
    pol = ExitPolicy("f", target_r=2.0)
    base = simulate_exit(o, h, l, c, np.ones(3), e, pol)
    win = simulate_exit(o, h, l, c, np.ones(3), e, pol, cost_points=0.2, slippage_points=0.25, entry_slip_points=0.25, target_slip_points=0.0)
    assert base.reason == win.reason == "target"
    assert win.r == pytest.approx(base.r - 0.2 - 0.25)           # commission + ENTRY slippage only: no exit slippage on a limit target
    # same entry, stopped out
    h2 = np.array([100.1, 100.2, 100.2]); l2 = np.array([99.9, 99.9, 98.5]); o2 = np.array([100.0, 100.0, 100.0]); c2 = o2
    lose0 = simulate_exit(o2, h2, l2, c2, np.ones(3), e, pol)
    lose = simulate_exit(o2, h2, l2, c2, np.ones(3), e, pol, cost_points=0.2, slippage_points=0.25, entry_slip_points=0.25)
    assert lose0.reason == lose.reason == "stop"
    assert lose.r == pytest.approx(lose0.r - 0.2 - 0.25 - 0.25)  # + stop-out slippage
    assert lose.cost_r == pytest.approx(0.2 + 0.25 + 0.25)
    assert win.cost_r == pytest.approx(0.2 + 0.25)


def _flat_bars(n=80, base=15000.0):
    ts = pd.date_range("2022-01-04 14:30", periods=n, freq="5min")
    c = base + np.zeros(n)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 0.5, "low": c - 0.5, "close": c})


def test_mini_when_a_whole_mini_fits_the_risk_else_micros():
    b = _flat_bars(); atr = atr_series(b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy())
    pol = ExitPolicy("t", target_r=1.0)
    tight = build_ledger(b, atr, [FrozenEntry(10, 1, 15000.0, 10.0)], pol, "NQ", risk_usd=250, max_contracts=4, label="x")[0]
    assert len(tight) == 1 and tight.risk[0] == pytest.approx(1 * 10.0 * 20.0)         # 1 NQ mini: $200 of the $250 risk
    wide = build_ledger(b, atr, [FrozenEntry(10, 1, 15000.0, 20.0)], pol, "NQ", risk_usd=250, max_contracts=4, label="x")[0]
    assert len(wide) == 1 and wide.risk[0] == pytest.approx(6 * 20.0 * 2.0)            # 6 MNQ micros
    micro = build_ledger(b, atr, [FrozenEntry(10, 1, 15000.0, 10.0)], pol, "NQ", risk_usd=250, max_contracts=4, label="x", sizing="micro")[0]
    assert micro.risk[0] == pytest.approx(12 * 10.0 * 2.0)


def _roll_frame():
    """5-min bars across the March-2022 roll (3rd Friday = Mar 18) with a +40pt splice gap on Mar 17's evening session."""
    days = pd.bdate_range("2022-03-01", "2022-03-31")
    parts = []
    lvl = 15000.0
    for d in days:
        ts = pd.date_range(d + pd.Timedelta(hours=14), periods=60, freq="5min")
        if d == pd.Timestamp("2022-03-17"):
            lvl += 40.0
        parts.append(pd.DataFrame({"timestamp": ts, "open": lvl + np.sin(np.arange(60)) * 0.2, "high": lvl + 1.0, "low": lvl - 1.0,
                                   "close": lvl + np.sin(np.arange(60)) * 0.2}))
        lvl += 0.0
    return pd.concat(parts, ignore_index=True)


def test_roll_gap_is_detected_excluded_and_back_adjustable():
    f = _roll_frame()
    rb = detect_roll_bars(f, "NQ")
    assert len(rb) == 1 and pd.Timestamp(f["timestamp"].iloc[rb[0]]).date() == pd.Timestamp("2022-03-17").date()
    days = roll_session_dates(f, rb)
    atr = atr_series(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy())
    sess = session_ordinals(f)
    j_roll = int(rb[0]) + 5
    j_ok = 5
    ents = [FrozenEntry(j_ok, 1, float(f["open"].iloc[j_ok]), 10.0), FrozenEntry(j_roll, 1, float(f["open"].iloc[j_roll]), 10.0)]
    pol = ExitPolicy("t", target_r=1.0, max_bars=10)
    led, skipped, _ = build_ledger(f, atr, ents, pol, "NQ", risk_usd=250, max_contracts=4, label="x")
    assert int(sess[j_roll]) in days and len(led) == 1 and skipped == 1
    led2, skipped2, _ = build_ledger(f, atr, ents, pol, "NQ", risk_usd=250, max_contracts=4, label="x", roll_policy="none")
    assert len(led2) == 2 and skipped2 == 0
    adj = back_adjust(f, rb)
    assert abs(adj["open"].iloc[rb[0]] - adj["close"].iloc[rb[0] - 1]) < 1e-6


def _synth(days=260, seed=5):
    rng = np.random.default_rng(seed)
    idx = [pd.date_range(d + pd.Timedelta(hours=13), periods=390, freq="1min") for d in pd.bdate_range("2022-01-03", periods=days)]
    ts = idx[0].append(idx[1:])
    c = 15000 + np.cumsum(rng.normal(0.0, 1.2, len(ts)))
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 0.8, "low": c - 0.8, "close": c, "volume": 1.0})


def test_stress_mode_runs_every_surviving_cell_at_1_and_2_ticks_and_trusts_only_if_both_pass():
    from app.prop.thirty_day import ThirtyDayConfig
    rep = scan({"NQ": _synth(260)}, get_preset("lucid_50k").to_prop_rules(), discovery_end=pd.Timestamp("2022-12-31"),
               sessions={"NQ": ["ny_open"]}, families=[("orb", {"or_minutes": 30})], exits=[ExitPolicy("t1R", target_r=1.0, max_bars=60)],
               feasibility_sims=100, cfg=ThirtyDayConfig(), stress_ticks=(1.0, 2.0), stress_min_lower_bound=0.0)
    survivors = [r for r in rep.results if r.result is not None]
    for r in survivors:
        assert set(r.stress) == {1.0, 2.0}
        assert r.trusted == all(v["passed"] for v in r.stress.values())
        assert r.stress[2.0]["net_expectancy_r"] <= r.stress[1.0]["net_expectancy_r"] + 1e-9
    assert all(r.trusted is False for r in rep.results if r.result is None)
