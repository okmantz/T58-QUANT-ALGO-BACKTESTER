"""2026-10-17: running jobs stay reachable, PWA installability pieces, Premium Subscription dropdown."""
from __future__ import annotations

import json

import pytest

from app.accounts import subscription as sub
from app.web import job_manager as jm
from app.web.server import app


@pytest.fixture()
def client():
    return app.test_client()


def test_track_external_appears_and_closes():
    mgr = jm.JobManager()
    state = {"alive": True}
    jid = mgr.track_external(tool="Research Loop", page_url="/research-loop/job/abc", is_running=lambda: state["alive"], instrument="ES")
    job = mgr.get(jid)
    assert job["tool"] == "Research Loop" and job["page_url"] == "/research-loop/job/abc" and not job.get("done")
    assert any(j.get("job_id", j.get("id")) == jid and not j.get("done") for j in mgr.list_jobs())
    state["alive"] = False
    mgr.list_jobs()  # reapers run whenever the feed is read
    assert mgr.get(jid).get("done")


def test_running_jobs_feed_shape(client):
    d = client.get("/api/running-jobs.json").get_json()
    assert {"running", "recent", "server_time"} <= set(d)


def test_runs_page_and_nav(client):
    html = client.get("/runs").get_data(as_text=True)
    assert "Running now" in html and "recent" in html.lower()


def test_manifest_is_installable(client):
    m = client.get("/manifest.json").get_json()
    assert m["display"] == "standalone" and m["start_url"] and m["scope"] == "/"
    sizes = {(i["sizes"], i.get("purpose", "any")) for i in m["icons"]}
    assert ("192x192", "any") in sizes and ("512x512", "any") in sizes and ("512x512", "maskable") in sizes
    for i in m["icons"]:
        assert client.get(i["src"]).status_code == 200


def test_service_worker_controls_root_and_has_fetch_handler(client):
    r = client.get("/sw.js")
    body = r.get_data(as_text=True)
    assert r.headers["Service-Worker-Allowed"] == "/" and "addEventListener(\"fetch\"" in body
    assert "javascript" in r.content_type


def test_head_links_manifest_on_every_page(client):
    for url in ("/dashboard", "/journal", "/library", "/gamma"):
        html = client.get(url).get_data(as_text=True)
        assert 'rel="manifest"' in html and "t58-pwa.js" in html, url


def test_subscription_is_premium_and_active_for_signed_in_user():
    eff = sub.effective_subscription(sub.SubscriptionInfo(plan="free text", license_key="", license_status="", renewal_date=""))
    assert eff.plan == "Premium Subscription" and eff.license_status == "active"
    expired = sub.effective_subscription(sub.SubscriptionInfo(plan="x", license_key="", license_status="expired", renewal_date=""))
    assert expired.license_status == "expired"
    assert sub.PREMIUM_PLAN in sub.PLAN_CHOICES


def test_account_settings_shows_premium_dropdown(client):
    html = client.get("/settings/account").get_data(as_text=True)
    assert "<select" in html and "Premium Subscription" in html
    assert 'name="plan"' in html
