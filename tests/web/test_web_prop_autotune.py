"""Tests for /api/suggest-loop-config (app.orchestration.prop_autotune's
web wiring)."""
from __future__ import annotations

from app.web.server import app


def test_suggest_loop_config_returns_a_suggestion():
    client = app.test_client()
    r = client.get("/api/suggest-loop-config?account_size=100000&profit_target=8&daily_loss=5&max_dd=10")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    s = data["suggestion"]
    assert 0.1 <= s["risk_value_pct"] <= 2.0
    assert 35.0 <= s["target_eval_pass_pct"] <= 75.0
    assert s["tightness_label"] in {"tight", "moderate", "loose"}
    assert len(s["preferred_families"]) > 0


def test_suggest_loop_config_uses_defaults_when_params_missing():
    client = app.test_client()
    r = client.get("/api/suggest-loop-config")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_suggest_loop_config_rejects_invalid_input():
    client = app.test_client()
    r = client.get("/api/suggest-loop-config?profit_target=not-a-number")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_search_form_still_loads_with_the_new_suggest_button():
    client = app.test_client()
    r = client.get("/search")
    assert r.status_code == 200
    assert b"Suggest settings from my prop rules" in r.data
