"""Research Labs, Pine Indicator Creator, memory endpoints, Quant Lab cards and Mobile Access tabs (2026-10-18 release)."""
from __future__ import annotations

import io
import json
import re

import numpy as np
import pandas as pd
import pytest

from app.web.server import app

LAB_SLUGS = ["volume-profile", "order-flow", "order-book", "lead-lag", "stat-arb", "event-study", "seasonality", "hypothesis"]


@pytest.fixture()
def client():
    return app.test_client()


def _csv(seed: int, n: int = 3000) -> tuple[io.BytesIO, str]:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.r_[c[0], c[:-1]]
    ts = pd.date_range("2024-01-02", periods=n, freq="5min")
    df = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "open": o, "high": np.maximum(o, c) * 1.0005,
                       "low": np.minimum(o, c) * 0.9995, "close": c, "volume": rng.integers(100, 900, n)})
    return io.BytesIO(df.to_csv(index=False).encode()), f"m{seed}.csv"


def _token(client, url="/research-labs/hypothesis") -> str:
    return re.search(r'data-csrf="([^"]+)"', client.get(url).get_data(as_text=True)).group(1)


def test_hub_lists_all_eight_labs(client):
    html = client.get("/research-labs/").get_data(as_text=True)
    for slug in LAB_SLUGS:
        assert f'href="/research-labs/{slug}"' in html


@pytest.mark.parametrize("slug", LAB_SLUGS)
def test_each_lab_page_renders(client, slug):
    assert client.get(f"/research-labs/{slug}").status_code == 200


def test_unknown_lab_is_404(client):
    assert client.get("/research-labs/nope").status_code == 404


def test_lab_run_needs_csrf_token(client):
    r = client.post("/research-labs/volume-profile/run", data={})
    assert r.status_code in (400, 403)


def test_volume_profile_run_over_http(client):
    tok = _token(client)
    r = client.post("/research-labs/volume-profile/run", data={"_csrf_token": tok, "uploads": [_csv(1)]},
                    content_type="multipart/form-data", headers={"X-CSRF-Token": tok})
    j = r.get_json()
    assert r.status_code == 200 and j["ok"] and j["record"]["assumptions"] and j["record"]["spec_hash"]


def test_order_book_without_level2_says_it_needs_data(client):
    tok = _token(client)
    r = client.post("/research-labs/order-book/run", data={"_csrf_token": tok}, content_type="multipart/form-data", headers={"X-CSRF-Token": tok})
    assert r.status_code == 200 and r.get_json()["record"]["status"] == "needs_data"


def test_lead_lag_with_one_market_is_a_clear_400(client):
    tok = _token(client)
    r = client.post("/research-labs/lead-lag/run", data={"_csrf_token": tok, "uploads": [_csv(1)]}, content_type="multipart/form-data",
                    headers={"X-CSRF-Token": tok})
    assert r.status_code == 400 and "two" in r.get_json()["error"].lower()


def test_hypothesis_bad_json_is_a_clear_400(client):
    tok = _token(client)
    r = client.post("/research-labs/hypothesis/run", data={"_csrf_token": tok, "spec": "{nope", "uploads": [_csv(1, 4000)]},
                    content_type="multipart/form-data", headers={"X-CSRF-Token": tok})
    assert r.status_code == 400 and "JSON" in r.get_json()["error"]


def test_sidebar_and_quant_lab_link_the_labs_and_gamma_center(client):
    side = client.get("/dashboard").get_data(as_text=True)
    assert 'href="/research-labs"' in side and 'id="t58-install-slot"' not in side
    ql = client.get("/quant-lab/").get_data(as_text=True)
    assert 'href="/gamma"' in ql and "Gamma Center" in ql and 'href="/research-labs"' in ql


def test_pine_indicator_page_and_api(client):
    html = client.get("/pine-indicator/").get_data(as_text=True)
    assert "Pine Indicator Creator" in html
    tok = re.search(r'data-csrf="([^"]+)"', html).group(1)
    r = client.post("/pine-indicator/api/create", json={"mode": "idea", "text": "EMA 21 and Bollinger bands", "use_ai": False},
                    headers={"X-CSRF-Token": tok})
    j = r.get_json()
    assert r.status_code == 200 and j["ok"] and j["result"]["ok"] and "//@version=5" in j["result"]["code"]
    r = client.post("/pine-indicator/api/create", json={"mode": "idea", "text": ""}, headers={"X-CSRF-Token": tok})
    assert r.status_code == 400


def test_pine_tool_is_on_the_create_page(client):
    assert 'href="/pine-indicator"' in client.get("/start-here/create").get_data(as_text=True)


def test_mobile_access_has_install_tab_and_sidebar_has_no_install_slot(client):
    html = client.get("/mobile-access").get_data(as_text=True)
    assert 'id="tab-install"' in html and 'id="panel-install"' in html and 'id="t58-install-slot"' in html
    assert 'id="install-card"' in html.split('id="panel-install"', 1)[1]          # the HTTPS/Tailscale install help moved with it
    assert html.count("<div") == html.count("</div>")


def test_memory_endpoints(client):
    st = client.get("/api/memory").get_json()
    assert "pid" in st
    tok = _token(client)
    r = client.post("/api/memory/free", json={"unload_ollama": False}, headers={"X-CSRF-Token": tok})
    assert r.status_code in (200, 409)


def test_surviving_findings_become_journal_tags(client, tmp_path, monkeypatch):
    from app.research_labs import journal_bridge
    from app.web import journal_store

    monkeypatch.setattr(journal_store, "journal_path", lambda: tmp_path / "journal.json")
    rec = {"lab": "volume_profile", "results": {"groups": [{"group": "below prior VAL", "verdict": "promising: significant after correction AND same sign out-of-sample"},
                                                              {"group": "inside value", "verdict": "no reliable difference"}]}}
    assert journal_bridge.tags_from_record(rec) == ["VP: below prior VAL"]
    assert journal_store.add_tags("Research findings", ["VP: below prior VAL"]) == ["VP: below prior VAL"]
    assert journal_store.add_tags("Research findings", ["VP: below prior VAL"]) == []          # idempotent
    groups = {g["name"]: g for g in journal_store.load()["settings"]["tagGroups"]}
    assert [i["name"] for i in groups["Research findings"]["items"]] == ["VP: below prior VAL"]
