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


def _plan_select(html: str) -> str:
    start = html.index('<select id="plan"')
    return html[start:html.index("</select>", start)]


def _state(**over):
    from app.licensing.client import LicenseState

    base = dict(email="a@b.co", license_key="T58-AAAAA-BBBBB-CCCCC-DDDDD", status="active", plan="Pro Annual", expires_at=None)
    base.update(over)
    return LicenseState(**base)


def test_subscription_comes_from_the_license_not_from_anything_typed():
    eff = sub.effective_subscription(_state(plan="Pro Monthly", expires_at="2027-01-15T00:00:00.000Z"))
    assert eff.plan == "Pro Monthly" and eff.license_status == "active"
    assert eff.license_key == "T58-AAAAA-BBBBB-CCCCC-DDDDD" and eff.renewal_date == "2027-01-15"
    assert sub.effective_subscription(_state(plan="Lifetime")).renewal_date == sub.NO_EXPIRY_TEXT
    assert sub.effective_subscription(_state(status="revoked")).license_status == "revoked"
    assert sub.effective_subscription(_state(status="expired")).license_status == "expired"
    assert sub.effective_subscription(_state(status="trial")).license_status == "active"  # reaching Account means the gate passed
    none = sub.effective_subscription(_state(license_key=""))
    assert none.plan == "" and none.license_status == "unset"


def test_license_without_a_plan_falls_back_to_the_generic_label():
    assert sub.effective_subscription(_state(plan="")).plan == sub.PREMIUM_PLAN


def test_master_key_reads_as_lifetime(monkeypatch):
    import hashlib

    from app.licensing import client as lic

    monkeypatch.setenv("T58_MASTER_LICENSE_KEY_HASH", hashlib.sha256(b"T58-OWNER").hexdigest())
    eff = sub.effective_subscription(_state(license_key="T58-OWNER", plan=""))
    assert eff.plan == lic.MASTER_PLAN == "Lifetime"


def test_plan_choices_are_the_license_sheet_plans():
    assert sub.PLAN_CHOICES == ("Pro Monthly", "Pro Annual", "Lifetime", "Trial", "Complimentary")


def test_account_settings_shows_a_locked_plan_dropdown(client):
    # the suite's activation is the master key, which is always Lifetime
    html = client.get("/settings/account").get_data(as_text=True)
    assert 'id="plan"' in html and 'name="plan"' in html
    plan_tag = _plan_select(html)
    assert " disabled" in plan_tag and "Lifetime" in plan_tag
    assert 'id="license_key"' in html and "readonly" in html
    assert "Save subscription info" not in html and "Check key format" not in html


def test_plan_cannot_be_changed_from_the_form(client):
    r = client.post("/settings/account/subscription/refresh", data={"plan": "Complimentary", "license_status": "cancelled"})
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    plan_tag = _plan_select(html)
    assert "Lifetime" in plan_tag and "Complimentary" not in plan_tag
    assert client.post("/settings/account/subscription/save", data={"plan": "Complimentary"}).status_code in (404, 405)
