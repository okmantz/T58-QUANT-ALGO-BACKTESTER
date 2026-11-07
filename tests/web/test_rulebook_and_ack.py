"""Rulebook page + in-app prop-rules acknowledgement endpoint."""
import json

import pytest

import app.web.server as server
from app.prop import presets as P


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def test_rulebook_page_and_calc(client, tmp_path, monkeypatch):
    from app.validation import rulebook as rb
    monkeypatch.setattr(rb, "_register_path", lambda: tmp_path / "register.json")
    monkeypatch.setattr(rb, "_sheet_path", lambda: tmp_path / "sheet.json")
    r = client.get("/rulebook")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    for s in ("Prop Firm Rulebook v2", "G0", "G7", "Rules sheet", "Capped paid attempts", "0.112", "trial register"):
        assert s in body, s
    for gone in ("Binary screen", "Alpha budget", "Edge Rulebook"):
        assert gone not in body, gone
    j = client.get("/api/rulebook/calc?n=400&looks=3&edge=0.15").get_json()
    assert j["ok"] and abs(j["required_observed_edge"] - 0.112) < 0.001 and j["e_plan"] > 0.05
    j = client.get("/api/rulebook/attempts?outcomes=1,1").get_json()
    assert j["ok"] and j["data"]["decision"] == "SCALE"


def test_register_trial_route_requires_g2(client, tmp_path, monkeypatch):
    from app.validation import rulebook as rb
    monkeypatch.setattr(rb, "_register_path", lambda: tmp_path / "register.json")
    monkeypatch.setattr(rb, "_sheet_path", lambda: tmp_path / "sheet.json")
    r = client.post("/rulebook/trial", data={"family": "f", "mechanism": "", "counterparty": "", "sign": "long", "instruments": "MNQ", "mechanisms": "1", "variants": "1"})
    assert r.status_code == 302 and "ok=0" in r.headers["Location"]
    r = client.post("/rulebook/trial", data={"family": "opening-drive", "mechanism": "late fade", "counterparty": "reversal traders", "sign": "long",
                                             "instruments": "MNQ, MES, M2K", "mechanisms": "1", "variants": "1", "seal_date": "2026-03-01", "code": "x"})
    assert "ok=1" in r.headers["Location"] and rb.load_register()["rows"][0]["id"] == "T1"
    assert rb.load_register()["rows"][0]["code_hash"] and rb.load_register()["seal_dates"]["MNQ"] == "2026-03-01"


def test_rules_sheet_route_marks_g0_open_until_complete(client, tmp_path, monkeypatch):
    from app.validation import rulebook as rb
    monkeypatch.setattr(rb, "_register_path", lambda: tmp_path / "register.json")
    monkeypatch.setattr(rb, "_sheet_path", lambda: tmp_path / "sheet.json")
    r = client.post("/rulebook/rules-sheet/preset", data={"preset": "lucid_50k"})
    assert r.status_code == 302 and "ok=1" in r.headers["Location"]
    assert rb.load_rules_sheet()["max_loss"]["value"] in (2000.0, "2000.0")
    r = client.post("/rulebook/rules-sheet", data={"fee": "172"})
    assert "ok=0" in r.headers["Location"]          # most of the sheet is still blank


def test_acknowledge_endpoint_stamps_for_30_days(client, tmp_path, monkeypatch):
    f = tmp_path / "preset_stamps.json"
    monkeypatch.setattr(P, "_stamp_file", lambda: f)

    def loader():
        raw = json.loads(f.read_text()) if f.exists() else {}
        return {k: (v["date"], v.get("url", ""), v.get("tier", "primary")) for k, v in raw.items()}
    monkeypatch.setattr(P, "_load_stamp_overrides", loader)
    P._refresh_stamps()
    try:
        assert not P.is_ready_eligible_key("lucid_25k")
        assert client.post("/api/prop-presets/lucid_25k/acknowledge", json={}).status_code == 400   # box not ticked
        r = client.post("/api/prop-presets/lucid_25k/acknowledge", json={"confirm": True})
        assert r.status_code == 200 and r.get_json()["status"]["needs_ack"] is False
        assert P.is_ready_eligible_key("lucid_25k")
        assert client.post("/api/prop-presets/nope/acknowledge", json={"confirm": True}).status_code == 404
    finally:
        monkeypatch.undo()
        P._refresh_stamps()


def test_25k_presets_exist_and_start_unverified():
    keys = {p.key for p in P.list_presets() if p.account_size == 25_000}
    assert {"ftmo_25k", "apex_25k_eod", "apex_25k_intraday", "lucid_25k", "fundednext_25k"} <= keys
    for k in ("ftmo_25k", "apex_25k_eod", "apex_25k_intraday", "lucid_25k"):
        assert P.ack_status(k)["state"] == "unverified"
