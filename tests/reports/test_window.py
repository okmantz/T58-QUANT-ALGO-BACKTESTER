"""Every stored number says which window it covers; reconcile() explains 390 vs 463 instead of leaving two 'different' answers."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from app.reports import window as w
from app.strategy.library_control import build_row


def _bt(n, net, win, sharpe, start, end):
    st = SimpleNamespace(total_trades=n, net_profit=net, win_rate=win, profit_factor=1.6, sharpe_ratio=sharpe, max_drawdown_pct=2.6)
    return SimpleNamespace(statistics=st, equity_curve=pd.DataFrame({"timestamp": pd.to_datetime([start, end]), "equity": [50000.0, 50000.0 + net]}))


def test_window_info_counts_the_holdout():
    dev, full = _bt(390, 12416.0, 58.2, 0.95, "2020-01-01", "2024-12-03"), _bt(463, 13761.0, 57.0, 0.90, "2020-01-01", "2025-09-30")
    wi = w.window_info(dev, full)
    assert wi["scored_on"] == "dev" and wi["holdout_trades"] == 73 and wi["same_window"] is False
    assert wi["dev"]["stats"]["total_trades"] == 390 and wi["full"]["stats"]["total_trades"] == 463
    assert wi["dev"]["period"] == ["2020-01-01", "2024-12-03"]


def test_window_info_same_window_when_no_holdout():
    dev = _bt(100, 10.0, 50.0, 1.0, "2020-01-01", "2021-01-01")
    wi = w.window_info(dev, None)
    assert wi["same_window"] is True and wi["holdout_trades"] == 0


def test_reconcile_explains_the_gap():
    dev, full = _bt(390, 12416.0, 58.2, 0.95, "2020-01-01", "2024-12-03"), _bt(463, 13761.0, 57.0, 0.90, "2020-01-01", "2025-09-30")
    rc = w.reconcile({"pipeline_snapshot": {"window_info": w.window_info(dev, full)}})
    assert rc["has"] and rc["agree"] is False and [r["key"] for r in rc["rows"]] == ["dev", "full"]
    text = " ".join(rc["lines"])
    assert "390 trades" in text and "73" in text and "463" in text and "0.95" in text and "0.90" in text
    assert "Dashboard cards" in rc["rows"][1]["surfaces"]


def test_reconcile_older_result_uses_run_history_row():
    meta = {"pipeline_snapshot": {"stats": {"total_trades": 390, "net_profit": 12416.0}}}
    rc = w.reconcile(meta, {"trades": 463, "net_profit": 13761.0, "sharpe_ratio": 0.9})
    assert rc["has"] and len(rc["rows"]) == 2 and rc["agree"] is False


def test_reconcile_nothing_to_compare():
    assert w.reconcile({}) == {"has": False}


def test_library_card_does_not_mix_windows():
    """Net profit and trades are the development window's; profit factor must come from the same result, not run history's 463-trade row."""
    meta = {"last_run": {"trades": 390, "net_profit": 12416.0, "profit_factor": 1.66, "window": "dev", "eval_pass_probability": 98.0}}
    run = {"trades": 463, "net_profit": 13761.0, "profit_factor": 1.59}
    row = build_row("python", "NQ.py", meta, "tested_failed", run)
    assert row["profit_factor"] == 1.66 and row["trades"] == 390 and row["window"] == "dev"
    old = build_row("python", "NQ.py", {"last_run": {"trades": 390, "net_profit": 12416.0}}, "tested_failed", run)
    assert old["profit_factor"] is None and old["trades"] == 390                       # never 1.59 beside 390 trades
    only_run = build_row("python", "NQ.py", {}, "tested_failed", run)
    assert only_run["profit_factor"] == 1.59 and only_run["window"] == "full"
