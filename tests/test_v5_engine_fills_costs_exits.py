"""v5 engine tests: Part C halt attrs, fill lag, per-contract commission,
time stop, MFE + exit_cause taxonomy, allow_single_contract_minimum.

Covers the 2026-10-04 analysis items implemented in app/backtest/risk.py
(new RiskConfig fields) and app/backtest/execution.py.
"""
import pandas as pd
import pytest

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig
from app.data.instrument_specs import apply_instrument_spec


def _df(rows, start="2024-01-01 09:00", freq="1h"):
    ts = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(
        [(t, o, h, l, c, 1000.0) for t, (o, h, l, c) in zip(ts, rows)],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


# ----------------------------------------------------------------------
# Part C: silent-halt attrs
# ----------------------------------------------------------------------

def test_sizing_halt_attrs_flag_silent_halt():
    """Part C signature: risk_value too small to afford 1 contract, so
    every signal floors to zero. The run must say so machine-readably."""
    rows = [(100.0, 100.5, 99.5, 100.0)] * 60
    df = _df(rows, freq="1d")
    signals = pd.Series([1] * 60)
    risk = RiskConfig(initial_balance=50_000.0, risk_value=0.25, pip_size=1.0,
                      contract_size=10.0)
    # $125/trade budget vs a $200 one-contract stop -> 0 contracts, always.
    trades, equity_df = run_execution(df, signals, risk,
                                      stop_loss_pips=20, take_profit_pips=None)
    assert trades == []
    halt = equity_df.attrs["sizing_halt"]
    assert halt["halted"] is True
    assert halt["skipped"] > 30
    assert halt["skip_ratio"] == pytest.approx(1.0)
    assert halt["last_trade_exit"] is None
    assert halt["risk_value"] == 0.25
    assert halt["contract_size"] == 10.0


def test_sizing_halt_attrs_present_but_not_halted_when_trading():
    """A healthy run still carries the key (halted=False) so downstream
    code never has to guard on its existence."""
    rows = [
        (100.0, 100.5, 99.5, 100.2),
        (100.2, 100.8, 99.8, 100.5),
        (100.5, 101.0, 100.0, 100.8),
        (100.8, 101.2, 100.3, 101.0),
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 0, 0])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0)
    trades, equity_df = run_execution(df, signals, risk,
                                      stop_loss_pips=50, take_profit_pips=None)
    assert len(trades) == 1
    halt = equity_df.attrs["sizing_halt"]
    assert halt["halted"] is False
    assert halt["skipped"] == 0
    assert halt["skip_ratio"] == pytest.approx(0.0)
    assert str(halt["last_trade_exit"]) == str(trades[-1].exit_time)


# ----------------------------------------------------------------------
# B2-2: fill lag
# ----------------------------------------------------------------------

def test_entry_fills_at_next_bar_open_by_default():
    rows = [
        (100.0, 100.5, 99.5, 100.2),
        (101.0, 101.5, 100.5, 101.2),
        (101.2, 101.8, 100.8, 101.5),
        (101.5, 102.0, 101.0, 101.8),
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 1, 1])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=50, take_profit_pips=None)
    assert len(trades) == 1
    t = trades[0]
    # NOT the signal bar's close (100.2) -- the next bar's open (101.0).
    assert t.entry_price == pytest.approx(101.0)
    assert t.entry_time == df["timestamp"][1]


def test_entry_fill_lag_zero_uses_signal_bar_open():
    rows = [
        (100.0, 100.5, 99.5, 100.2),
        (101.0, 101.5, 100.5, 101.2),
        (101.2, 101.8, 100.8, 101.5),
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 1])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
                      entry_fill_lag_bars=0)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=50, take_profit_pips=None)
    assert len(trades) == 1
    assert trades[0].entry_price == pytest.approx(100.0)  # open[0]
    assert trades[0].entry_time == df["timestamp"][0]


def test_signal_exit_fills_at_next_bar_open():
    rows = [
        (100.0, 100.5, 99.5, 100.2),
        (100.2, 100.8, 99.8, 100.5),
        (100.5, 101.0, 100.0, 100.8),
        (100.8, 101.2, 100.3, 101.0),
        (101.0, 101.5, 100.5, 101.2),
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 1, 0, 0])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=50, take_profit_pips=None)
    assert len(trades) == 1
    t = trades[0]
    # Signal flips on bar 3 -> market exit fills at bar 4's open (101.0),
    # not bar 3's close (101.0 is the close too -- assert the TIME as well).
    assert t.exit_price == pytest.approx(101.0)
    assert t.exit_time == df["timestamp"][4]
    assert t.exit_reason == "signal"
    assert t.exit_cause == "signal_exit"


def test_signal_on_last_bar_opens_no_trade_with_default_lag():
    rows = [
        (100.0, 100.5, 99.5, 100.2),
        (100.2, 100.8, 99.8, 100.5),
        (100.5, 101.0, 100.0, 100.8),
    ]
    df = _df(rows)
    signals = pd.Series([0, 0, 1])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=50, take_profit_pips=None)
    # No future bar exists to fill on -- an honest engine opens nothing.
    assert trades == []


# ----------------------------------------------------------------------
# B2-4: per-contract commission
# ----------------------------------------------------------------------

def test_per_contract_commission_charged_at_settle():
    rows = [
        (100.0, 100.5, 99.5, 100.2),
        (100.2, 100.8, 99.8, 100.5),
        (100.5, 101.0, 100.0, 100.8),
        (100.8, 101.2, 100.3, 101.0),
        (101.0, 101.5, 100.5, 101.2),
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 1, 0, 0])
    risk = RiskConfig(initial_balance=50_000.0, risk_value=2.0, pip_size=1.0,
                      contract_size=10.0,
                      commission_per_trade=2.0, commission_per_contract=1.60)
    # $1000 budget / $200 one-contract stop = 5 contracts.
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=20, take_profit_pips=None)
    assert len(trades) == 1
    t = trades[0]
    assert t.size == pytest.approx(50.0)  # 5 contracts x 10 units
    assert t.commission == pytest.approx(2.0 + 1.60 * 5)
    # And the P&L actually paid it.
    gross = (t.exit_price - t.entry_price) * t.size
    assert t.pnl == pytest.approx(gross - t.commission)


def test_mgc_spec_fills_per_contract_commission_end_to_end():
    """apply_instrument_spec('MGC') -> $1.60/contract; a 2-contract trade
    is charged $3.20 at settle, not the old flat $1.60."""
    risk = apply_instrument_spec(
        RiskConfig(initial_balance=50_000.0, risk_value=1.0), "MGC")
    assert risk.commission_per_contract == pytest.approx(1.60)
    assert risk.commission_per_trade == 0.0
    rows = [
        (2000.0, 2001.0, 1999.0, 2000.5),
        (2000.5, 2002.0, 1999.5, 2001.0),
        (2001.0, 2003.0, 2000.0, 2002.0),
        (2002.0, 2004.0, 2001.0, 2003.0),
        (2003.0, 2005.0, 2002.0, 2004.0),
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 1, 0, 0])
    # $500 budget / $200 one-contract stop (20 pts x $10) = 2 contracts.
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=20, take_profit_pips=None)
    assert len(trades) == 1
    assert trades[0].commission == pytest.approx(1.60 * 2)


# ----------------------------------------------------------------------
# Part A port #4: time stop
# ----------------------------------------------------------------------

def test_time_stop_closes_stagnant_position():
    # Hourly bars pinned at 100 -- the trade goes nowhere.
    rows = [(100.0, 100.05, 99.95, 100.0)] * 8
    df = _df(rows, freq="1h")
    signals = pd.Series([1, 1, 1, 1, 1, 0, 0, 0])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
                      time_stop_hours=2.0, time_stop_atr_band=0.15)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=500, take_profit_pips=None)
    assert len(trades) == 1
    t = trades[0]
    # Entry fills at bar 1's open (10:00); age exceeds 2h at bar 4 (13:00).
    assert t.exit_reason == "time_stop"
    assert t.exit_cause == "time_stop"
    assert t.exit_time == df["timestamp"][4]


def test_time_stop_spares_position_that_moved():
    # Same clock, but price trends up well beyond the stagnation band.
    rows = [(100.0 + i * 0.5, 100.3 + i * 0.5, 99.7 + i * 0.5, 100.0 + i * 0.5)
            for i in range(8)]
    df = _df(rows, freq="1h")
    signals = pd.Series([1] * 8)
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
                      time_stop_hours=2.0, time_stop_atr_band=0.15)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=500, take_profit_pips=None)
    assert len(trades) == 1
    # Still open at the end of data -- the time stop never fired.
    assert trades[0].exit_reason == "end_of_data"
    assert trades[0].exit_cause == "end_of_data"


def test_time_stop_disabled_by_default():
    rows = [(100.0, 100.05, 99.95, 100.0)] * 8
    df = _df(rows, freq="1h")
    signals = pd.Series([1] * 8)
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0)
    assert risk.time_stop_hours is None
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=500, take_profit_pips=None)
    assert len(trades) == 1
    assert trades[0].exit_reason == "end_of_data"


# ----------------------------------------------------------------------
# Part A port #1: MFE + exit_cause taxonomy
# ----------------------------------------------------------------------

def test_mfe_price_tracks_favorable_extreme():
    rows = [
        (100.0, 100.5, 99.5, 100.2),   # bar 0: signal
        (100.2, 102.0, 100.0, 101.0),  # bar 1: entry @ 100.2 open, high 102
        (101.0, 103.5, 100.5, 102.0),  # bar 2: high 103.5 <- MFE
        (102.0, 102.5, 101.0, 101.5),  # bar 3
        (101.5, 102.0, 100.0, 100.5),  # bar 4
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 1, 1, 1])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=500, take_profit_pips=None)
    assert len(trades) == 1
    assert trades[0].mfe_price == pytest.approx(103.5)


def test_exit_cause_stop_loss_and_take_profit():
    # Long: stop 2 points below entry, target 3 above. Bar 2 tanks through
    # the stop; a second trade then rides to the target.
    rows = [
        (100.0, 100.5, 99.5, 100.2),   # 0: signal -> fill @ 100.5 open[1]
        (100.5, 101.0, 100.0, 100.8),  # 1: entry 100.5, stop 98.5, tp 103.5
        (100.8, 101.0, 97.0, 98.0),    # 2: low 97 <= stop -> stop_loss
        (98.0, 99.0, 97.5, 98.5),      # 3: signal -> fill @ 99.0 open[4]
        (99.0, 104.0, 98.5, 103.0),    # 4: entry 99.0, high 104 >= tp 102 -> take_profit
        (103.0, 103.5, 102.5, 103.2),  # 5
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 1, 1, 0, 0])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
                      entry_fill_lag_bars=1, reentry_cooldown_bars=0)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=2, take_profit_pips=3)
    assert len(trades) == 2
    assert trades[0].exit_reason == "stop_loss"
    assert trades[0].exit_cause == "stop_loss"
    assert trades[1].exit_reason == "take_profit"
    assert trades[1].exit_cause == "take_profit"


def test_exit_cause_breakeven():
    # breakeven_trigger_r=1.0 moves the stop to entry once +1R is reached;
    # price then falls back and tags the breakeven stop.
    rows = [
        (100.0, 100.5, 99.5, 100.2),   # 0: signal -> fill @ 100.5 open[1]
        (100.5, 101.0, 100.0, 100.8),  # 1: entry 100.5, stop 99.5 (1R = 1.0)
        (100.8, 102.0, 100.5, 101.5),  # 2: best 102 -> +1.5R -> stop -> 100.5
        (101.5, 101.8, 100.4, 101.0),  # 3: low 100.4 <= 100.5 -> breakeven stop
        (101.0, 101.5, 100.5, 101.2),  # 4
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 0, 0, 0])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
                      reentry_cooldown_bars=0)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=1, take_profit_pips=None,
                              breakeven_trigger_r=1.0)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "stop_loss"  # raw engine reason unchanged
    assert t.exit_cause == "breakeven"   # taxonomy sees the moved stop


def test_exit_cause_trailing_stop():
    # A 1-point trailing distance ratchets the stop up; price then falls
    # back onto the TRAILED stop (above the original).
    rows = [
        (100.0, 100.5, 99.5, 100.2),   # 0: signal -> fill @ 100.5 open[1]
        (100.5, 101.0, 100.2, 100.8),  # 1: entry 100.5, stop 98.5 -> trails to 100.0, survives
        (100.8, 103.0, 101.0, 102.5),  # 2: best 103 -> trail stop -> 102.0; low 101 tags it
        (102.5, 102.8, 102.0, 102.2),  # 3
        (102.2, 102.5, 101.8, 102.0),  # 4
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 0, 0, 0])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
                      reentry_cooldown_bars=0)
    trades, _ = run_execution(
        df, signals, risk, stop_loss_pips=2, take_profit_pips=None,
        trailing_stop_distance=pd.Series([1.0] * len(df)),
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.exit_cause == "trailing_stop"


def test_exit_cause_daily_loss_close():
    # Daily loss limit $200: the position's floating loss breaches it
    # intrabar on bar 2 -> forced close tagged daily_loss_close.
    rows = [
        (100.0, 100.5, 99.5, 100.2),   # 0: signal -> fill @ 100.5 open[1]
        (100.5, 101.0, 100.0, 100.8),  # 1: entry 100.5, size = 100/20 = 5
        (100.8, 101.0, 60.0, 61.0),    # 2: low 60 -> floating -$200+ -> forced
        (61.0, 62.0, 60.0, 61.5),      # 3
    ]
    df = _df(rows)
    signals = pd.Series([1, 1, 1, 1])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
                      daily_loss_limit_pct=2.0, reentry_cooldown_bars=0)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=20, take_profit_pips=None)
    assert len(trades) >= 1
    t = trades[0]
    assert t.exit_reason == "daily_loss_limit_forced_close"
    assert t.exit_cause == "daily_loss_close"


def test_exit_cause_session_close_on_weekend_hold():
    # block_weekend_hold=True: a position open into the week's last bar is
    # force-closed -> session_close.
    rows = [
        (100.0, 100.5, 99.5, 100.2),   # Mon 2024-01-01 09:00
        (100.2, 100.8, 99.8, 100.5),   # Tue
        (100.5, 101.0, 100.0, 100.8),  # Wed
        (100.8, 101.2, 100.3, 101.0),  # Thu
        (101.0, 101.5, 100.5, 101.2),  # Fri -> week ends here
    ]
    df = _df(rows, start="2024-01-01 09:00", freq="1d")
    signals = pd.Series([1, 1, 1, 1, 1])
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
                      block_weekend_hold=True, reentry_cooldown_bars=0)
    trades, _ = run_execution(df, signals, risk,
                              stop_loss_pips=500, take_profit_pips=None)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "weekend_hold_forced_close"
    assert t.exit_cause == "session_close"


# ----------------------------------------------------------------------
# Part C fix 2: allow_single_contract_minimum
# ----------------------------------------------------------------------

def test_allow_single_contract_minimum_rescues_profitable_account():
    """Stops too wide for the budget on a profitable (equity >= initial)
    account: default OFF skips everything (silent halt); opt-in takes the
    1-contract minimum and tags it sized_above_risk_target=True."""
    rows = [(2000.0, 2001.0, 1999.0, 2000.5)] * 30
    df = _df(rows, freq="1d")
    signals = pd.Series([1] * 30)

    def _run(**kw):
        risk = RiskConfig(initial_balance=50_000.0, risk_value=0.25,
                          pip_size=1.0, contract_size=10.0, **kw)
        # $125 budget vs $200 one-contract stop -> floors to 0 every time.
        return run_execution(df, signals, risk,
                             stop_loss_pips=20, take_profit_pips=None)

    trades_off, eq_off = _run()
    assert trades_off == []
    assert eq_off.attrs["sizing_halt"]["halted"] is True

    trades_on, _ = _run(allow_single_contract_minimum=True)
    assert len(trades_on) > 0
    assert all(t.sized_above_risk_target for t in trades_on)
    assert all(t.size == pytest.approx(10.0) for t in trades_on)  # 1 contract


def test_allow_single_contract_minimum_defaults_off():
    assert RiskConfig().allow_single_contract_minimum is False
