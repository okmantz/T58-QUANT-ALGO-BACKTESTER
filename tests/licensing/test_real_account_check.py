import pandas as pd

from app.validation.real_account_check import compare_trade_lists


def _rows(shift_min=0, pnl_add=0.0, drop=0):
    t0 = pd.Timestamp("2026-09-01 14:30")
    rows = []
    for i in range(24):
        rows.append(dict(entry_time=t0 + pd.Timedelta(hours=6 * i, minutes=shift_min), exit_time=t0 + pd.Timedelta(hours=6 * i + 1),
                         direction="long" if i % 2 else "short", entry_price=5000 + i, exit_price=5004 + i if i % 2 else 4996 + i, pnl=200.0 + pnl_add))
    return pd.DataFrame(rows).iloc[drop:]


def test_same_trades_within_costs_agree():
    r = compare_trade_lists(_rows(), _rows(shift_min=3, pnl_add=-25.0))
    assert r.agrees and r.n_matched == 24 and "AGREES" in r.render()


def test_missing_and_mismatched_trades_do_not_agree():
    r = compare_trade_lists(_rows(), _rows(drop=10))
    assert not r.agrees and len(r.unmatched_real) == 10
    r2 = compare_trade_lists(_rows(), _rows(pnl_add=-400.0))
    assert not r2.agrees
