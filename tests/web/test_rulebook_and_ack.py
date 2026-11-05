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
    monkeypatch.setattr(rb, "_ledger_path", lambda: tmp_path / "alpha.json")
    r = client.get("/rulebook")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    for s in ("Edge Rulebook", "Alpha budget", "Binary screen", "Forward test", "2.81"):
        assert s in body, s
    j = client.get("/api/rulebook/calc?sr=0.10&looks=10").get_json()
    assert j["ok"] and j["z_bar"] == 2.807 and abs(j["trades_needed"] - 1038) <= 1


def test_log_hypothesis_requires_frame(client, tmp_path, monkeypatch):
    from app.validation import rulebook as rb
    monkeypatch.setattr(rb, "_ledger_path", lambda: tmp_path / "alpha.json")
    r = client.post("/rulebook/hypothesis", data={"mechanism": "", "segment": "x", "sign": "+", "payoff": "2", "cost_r": "0.05"})
    assert r.status_code == 302 and "ok=0" in r.headers["Location"]
    r = client.post("/rulebook/hypothesis", data={"mechanism": "yields lead futures", "segment": "next 30m", "sign": "+", "payoff": "2", "cost_r": "0.05"})
    assert "ok=1" in r.headers["Location"] and rb.load_ledger()[0]["id"] == "H1"


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
