"""Structure checks for the redesigned lifecycle pages: vertical steps, one mode selector, no instrument picker, embed mode."""
from __future__ import annotations

from datetime import date

import pytest

from app.prop import payout_calendar
from app.web.server import app


@pytest.fixture
def client():
    return app.test_client()


@pytest.mark.parametrize("path,steps", [("/full-pipeline", 3), ("/optimize-simple", 4), ("/validate-simple", 4)])
def test_lifecycle_pages_have_vertical_steps_and_the_mode_selector(client, path, steps):
    body = client.get(path).get_data(as_text=True)
    assert body.count('class="t58-step-n"') >= steps
    assert 'name="objective_mode"' in body and 'value="prop_lifecycle"' in body and 'value="net_profit"' in body
    assert "Pick known instrument" not in body and "Pick a known instrument" not in body
    assert "Individual tools" not in body


def test_prop_firm_is_the_default_mode(client):
    body = client.get("/full-pipeline").get_data(as_text=True)
    i = body.index('value="prop_lifecycle"')
    assert "checked" in body[i - 120:i + 120]


def test_test_page_drops_previous_runs(client):
    assert "Previous runs" not in client.get("/full-pipeline").get_data(as_text=True)


def test_optimize_page_drops_the_blocked_times_card(client):
    assert "Don't trade at these times" not in client.get("/optimize-simple").get_data(as_text=True)


def test_replay_is_four_plain_steps(client):
    body = client.get("/replay").get_data(as_text=True)
    for label in ("Select strategy", "Select market data", "Select prop firm", "Start replay"):
        assert label in body


def test_embed_mode_hides_the_sidebar_but_keeps_the_page(client):
    plain = client.get("/payout-planner").get_data(as_text=True)
    emb = client.get("/payout-planner?embed=1").get_data(as_text=True)
    assert "t58-sidebar" in plain and "t58-sidebar" not in emb
    assert "t58h" in emb


def test_deploy_has_three_tabs_and_ordered_tools(client):
    body = client.get("/forward-test").get_data(as_text=True)
    for tab in ("dv-tab-demo", "dv-tab-mt5", "dv-tab-live"):
        assert tab in body
    tools = body[body.index("More deploy tools"):]
    order = [tools.index(x) for x in ("Monitor (Live Market)", "Interactive Replay", "TradingView bridge", "Webhook Relay (TradingView)")]
    assert order == sorted(order)
    assert "Trade log" not in body
    assert 'class="tname">Deploy Live' not in tools and 'class="tname">T58 Demo account' not in tools


def test_account_sections_are_collapsible(client):
    body = client.get("/settings/account").get_data(as_text=True)
    for sec in ("sec-profile", "sec-security", "sec-subscription", "saved-runs", "pz-sec-accent", "pz-sec-workspace"):
        assert sec in body
    assert "Accent colour" not in body and "colour" not in body.lower().replace("colourful", "")


def test_prop_journey_has_the_calendar_and_planner_tabs(client):
    body = client.get("/accounts").get_data(as_text=True)
    assert "planner" in body and "recommender" in body and "cal-" in body


def test_calendar_feed_is_flat_dated_rows():
    rows = payout_calendar.feed({"accounts": [], "sim_runs": []}, today=date(2026, 11, 10))
    assert rows == []
