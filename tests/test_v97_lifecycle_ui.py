"""v9.7 lifecycle UI: preview-matching step pages, linked sidebar
section headers, prop-firm rules panel, and the dashboard/champion-board
next-step fix (untested -> Run Full Pipeline, never Open Validate).

Headless like the v9.6 web tests: pages render through the Flask test
client; the champion board is stubbed at server.champion_board.
"""
from __future__ import annotations

import pytest

import app.web.server as server


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def _row(*, untested: bool, can_promote: bool = False):
    if untested:
        requirements = [
            {"key": "backtested", "label": "Has been backtested", "met": False,
             "detail": "No run on record."},
            {"key": "validated_cpcv", "label": "Validated by CPCV", "met": False,
             "detail": "No CPCV result on record."},
        ]
    else:
        requirements = [
            {"key": "backtested", "label": "Has been backtested", "met": True,
             "detail": "Last run 2026-10-07."},
            {"key": "validated_cpcv", "label": "Validated by CPCV", "met": False,
             "detail": "No CPCV result on record."},
        ]
    return {
        "strategy_type": "python",
        "filename": "demo_strategy.py",
        "display_name": "Demo Strategy",
        "status": "DEVELOPING",
        "verdict": None if untested else "MARGINAL",
        "eval_pct": None if untested else 55.0,
        "payout_pct": None if untested else 40.0,
        "oos_pct": None if untested else 48.0,
        "t58_score": None,
        "strength": None,
        "promotion": {
            "stage": "candidate",
            "stage_title": "Candidate",
            "stage_index": 0,
            "stage_pct": 0.0,
            "next_stage": "validated",
            "next_stage_title": "Validated",
            "can_promote": can_promote,
            "requirements": requirements,
            "promoted_at": None,
        },
        "modified": 0.0,
    }


# ------------------------------------------------------- sidebar headers

def test_sidebar_lifecycle_headers_link_to_step_pages(client):
    body = client.get("/dashboard").get_data(as_text=True)
    for href in ("/start-here/create", "/full-pipeline", "/optimize-simple",
                 "/validate-simple", "/champion-simple", "/forward-test",
                 "/graveyard"):
        assert f'href="{href}"' in body, href
    # The summary still carries the numbered labels the v9.6 tests pin.
    assert "&#9312;</span> Create" in body
    assert "&#9318;</span> Strategy Graveyard" in body


# ------------------------------------------------------------ step pages

@pytest.mark.parametrize("path", [
    "/full-pipeline", "/validate-simple", "/optimize-simple",
    "/champion-simple", "/forward-test", "/start-here/create",
])
def test_lifecycle_pages_render_stepper_and_shell(client, path):
    body = client.get(path).get_data(as_text=True)
    assert "lc-stepper" in body, path


@pytest.mark.parametrize("path", [
    "/full-pipeline", "/validate-simple", "/optimize-simple",
    "/champion-simple",
])
def test_selector_pages_have_prop_firm_rules_panel(client, path):
    body = client.get(path).get_data(as_text=True)
    assert "lc-rules-body" in body, path          # the rail rules panel
    assert "LC_PROP_PRESETS" in body or "PROP_PRESETS" in body, path


def test_full_pipeline_page_is_preview_shaped(client):
    body = client.get("/full-pipeline").get_data(as_text=True)
    assert "Test it properly" in body
    assert "RUN FULL PIPELINE" in body
    assert "Run log — live" in body
    assert "lc-will-run" in body


def test_validate_page_keeps_all_check_boxes(client):
    body = client.get("/validate-simple").get_data(as_text=True)
    for name in ("check_monte_carlo", "check_cpcv", "check_pbo",
                 "check_walk_forward", "check_sensitivity",
                 "check_robustness", "check_regime"):
        assert f'name="{name}"' in body, name


# ------------------------------------------- next-step logic (the fix)

def test_champion_board_untested_row_never_points_at_validate(client, monkeypatch):
    monkeypatch.setattr(server.champion_board, "list_board",
                        lambda: [_row(untested=True)])
    body = client.get("/champion-simple").get_data(as_text=True)
    assert "Run Full Pipeline" in body
    assert "Open Validate hub" not in body
    assert "/full-pipeline?load_strategy_type=python" in body


def test_champion_board_tested_row_points_at_validate_checks(client, monkeypatch):
    monkeypatch.setattr(server.champion_board, "list_board",
                        lambda: [_row(untested=False)])
    body = client.get("/champion-simple").get_data(as_text=True)
    assert "Open Validate (pick checks)" in body
    assert "Open Validate hub" not in body


def test_dashboard_untested_row_never_points_at_validate(client, monkeypatch):
    monkeypatch.setattr(server.champion_board, "list_board",
                        lambda: [_row(untested=True)])
    body = client.get("/dashboard").get_data(as_text=True)
    assert "Open Validate hub" not in body
    assert "Run Full Pipeline" in body


def test_dashboard_tested_row_points_at_validate_checks(client, monkeypatch):
    monkeypatch.setattr(server.champion_board, "list_board",
                        lambda: [_row(untested=False)])
    body = client.get("/dashboard").get_data(as_text=True)
    assert "Open Validate (pick checks)" in body
    assert "Open Validate hub" not in body


# ---------------------------------------------- champion_board href logic

def test_next_action_href_untested_is_full_pipeline():
    href = server.champion_board._next_action_href(_row(untested=True))
    assert href == "/full-pipeline"


def test_next_action_href_tested_is_validate_simple():
    href = server.champion_board._next_action_href(_row(untested=False))
    assert href == "/validate-simple"
