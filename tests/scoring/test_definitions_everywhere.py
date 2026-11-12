"""Every metric/check a user can see must be clickable and defined: what it is, how it is calculated, what passes."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.scoring import metric_defs as md
from app.validation.simple_report import CHECK_LABELS
from app.web.server import app

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("label", ["Eval pass chance", "Payout chance", "Single-run pass rate", "Best Sharpe", "Risk of ruin", "Win rate", "Sharpe",
                                   "Net profit", "Profit factor", "Max drawdown", "Trades", "T58 score", "OOS", "Champion Check", "Pooled OOS"])
def test_headline_labels_resolve(label):
    assert md.resolve(label), label


@pytest.mark.parametrize("key,label", list(CHECK_LABELS.items()))
def test_every_validate_check_label_is_defined(key, label):
    k = md.resolve(label)
    assert k, f"{label!r} has no definition"
    d = md.definition(k)
    assert d["meaning"] and d["formula"]


@pytest.mark.parametrize("key", ["walk_forward", "wfga", "sensitivity", "parameter_robustness", "regime_matrix", "cpcv", "pbo", "eval_pass_probability",
                                 "first_payout_probability", "pooled_oos"])
def test_checks_explain_how_they_are_calculated_and_what_passes(key):
    d = md.definition(key)
    assert d["how"] and len(d["how"]) >= 3, key
    assert d["bar"] or key in ("eval_pass_probability",), key
    assert d.get("fix"), key


def test_library_tile_labels_are_explicit_metrics():
    html = (ROOT / "app/web/templates/library.html").read_text(encoding="utf-8")
    for key in ("eval_pass_probability", "first_payout_probability", "sharpe_ratio", "win_rate", "risk_of_ruin", "profit_factor", "net_profit"):
        assert f"'{key}')" in html, key


def test_api_serves_new_definitions():
    c = app.test_client()
    for key in ("walk_forward", "sensitivity", "regime_matrix", "wfga"):
        j = c.get(f"/api/metrics/{key}").get_json()
        assert j["ok"] and j["metric"]["how"] and j["metric"]["bar"]
    labels = c.get("/api/metrics").get_json()["labels"]
    assert labels["eval pass chance"] == "eval_pass_probability" and labels["walk-forward"] == "walk_forward"


def test_popup_js_renders_how_bar_and_fix():
    js = (ROOT / "app/web/static/t58-metrics.js").read_text(encoding="utf-8")
    for needle in ("How it is calculated", "What counts as a pass", "If it is not passing", "t58-i"):
        assert needle in js


def test_no_alias_collides_with_verdict_pills():
    """Aliases that equal a verdict pill would hijack the pill's own 'why' popup (the metrics click handler runs first)."""
    for pill in ("ready", "not ready", "marginal", "verdict"):
        assert md.resolve(pill) is None, pill
