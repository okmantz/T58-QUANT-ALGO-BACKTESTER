"""Champion award -> live P&L panel; demo account rules sheet + projection; deploy page shows the demo."""
from __future__ import annotations

import pytest

from app import achievements
from app.demo_account import account as demo
from app.web.server import app


@pytest.fixture
def client():
    return app.test_client()


def _acct_with_trades(n=20):
    a = demo._fresh(50_000.0)
    for i in range(n):
        p = 120.0 if i % 3 else -60.0
        a.balance += p
        a.trades.append({"n": i + 1, "closed_at": f"2024-03-{1 + i // 4:02d}T12:00:00+00:00", "pnl": p, "balance": a.balance, "result": "win" if p > 0 else "loss"})
    return a


def test_rules_sheet_lists_every_rule_in_plain_words():
    names = [r["name"] for r in demo.rules_sheet(demo._fresh(50_000.0))]
    for want in ("Profit target", "Maximum loss (drawdown)", "Daily loss limit (soft)", "Consistency (best-day rule)", "Minimum trading days"):
        assert want in names


def test_projection_needs_15_trades_then_gives_a_chance_and_days():
    assert demo.projection(_acct_with_trades(10)) is None
    pr = demo.projection(_acct_with_trades(20))
    assert 0 <= pr["pass_pct"] <= 100 and pr["basis_trades"] == 20


def test_to_dict_carries_everything_the_panel_needs():
    d = _acct_with_trades().to_dict()
    for k in ("rules_sheet", "equity", "daily_room", "projection", "stage", "trading_days"):
        assert k in d


def test_first_champion_award_unlocks_the_live_pnl_panel(monkeypatch):
    st = achievements.unlock_state()["features"]["live_pnl_panel"]
    assert "champ-1" in achievements.CATALOGUE and "First Champion" in st["text"]
    monkeypatch.setattr(achievements, "_read_unlocked", lambda: ({"champ-1": {"at": 1, "seen": True}}, 1))
    assert achievements.unlock_state()["features"]["live_pnl_panel"]["unlocked"] is True


def test_deploy_page_offers_the_demo(client):
    html = client.get("/deploy-live").get_data(as_text=True)
    assert "T58 prop-firm demo" in html and "/demo-account" in html


def test_demo_page_shows_the_rules_table(client):
    html = client.get("/demo-account").get_data(as_text=True)
    assert "The rules of this account" in html and "Maximum loss (drawdown)" in html
