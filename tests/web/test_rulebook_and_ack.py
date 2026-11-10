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


def test_rulebook_is_not_a_user_facing_page(client):
    """The Rulebook v2 gates run inside the pipeline and the Discovery Engine; there is no page or form for them."""
    for path in ("/rulebook", "/api/rulebook/calc", "/api/rulebook/attempts"):
        assert client.get(path).status_code == 404, path
    for path in ("/rulebook/trial", "/rulebook/rules-sheet", "/rulebook/rules-sheet/preset"):
        assert client.post(path, data={}).status_code in (404, 405), path
    assert b"Prop Firm Rulebook" not in client.get("/dashboard").data


def test_rulebook_math_still_available_to_the_engine():
    from app.validation import rulebook as rb
    assert abs(rb.required_observed_edge(400, 3) - 0.112) < 0.001
    assert rb.g7_capped_attempts([1, 1]).data["decision"] == "SCALE"


def test_register_trial_requires_g2(tmp_path, monkeypatch):
    from app.validation import rulebook as rb
    monkeypatch.setattr(rb, "_register_path", lambda: tmp_path / "register.json")
    monkeypatch.setattr(rb, "_sheet_path", lambda: tmp_path / "sheet.json")
    bad = rb.register_trial(family="f", mechanism="", counterparty="", sign="long", instruments=["MNQ"])
    assert not bad["ok"]
    good = rb.register_trial(family="opening-drive", mechanism="late fade", counterparty="reversal traders", sign="long",
                             instruments=["MNQ", "MES", "M2K"], sealed={"MNQ": "2026-03-01", "MES": "2026-03-01", "M2K": "2026-03-01"}, code_text="x")
    assert good["ok"] and rb.load_register()["rows"][0]["id"] == "T1" and rb.load_register()["rows"][0]["code_hash"]


def test_rules_sheet_from_preset_marks_g0_open_until_complete(tmp_path, monkeypatch):
    from app.validation import rulebook as rb
    monkeypatch.setattr(rb, "_register_path", lambda: tmp_path / "register.json")
    monkeypatch.setattr(rb, "_sheet_path", lambda: tmp_path / "sheet.json")
    sheet = rb.rules_sheet_from_preset("lucid_50k")
    assert sheet["max_loss"]["value"] in (2000.0, "2000.0")
    assert not rb.g0_rules_sheet({"fee": {"value": 172}}).passed          # most of the sheet is still blank


def test_discovery_engine_notes_carry_rulebook_context(tmp_path, monkeypatch):
    from app.validation import rulebook as rb
    from app.discovery_engine.engine import _rulebook_notes
    monkeypatch.setattr(rb, "_register_path", lambda: tmp_path / "register.json")
    monkeypatch.setattr(rb, "_sheet_path", lambda: tmp_path / "sheet.json")
    notes = _rulebook_notes()
    assert any("Rulebook v2" in n for n in notes) and any("G0" in n for n in notes)


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
