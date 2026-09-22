"""Tests for the Support page and the Account tab's profile-picture
feature."""
from __future__ import annotations

import io

from app.web.server import app


def _client():
    app.config["TESTING"] = True
    return app.test_client()


def test_support_page_renders_with_discord_link():
    r = _client().get("/support")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "discord.gg/3MbKm3S2zG" in body
    assert "/education" in body


def test_report_issue_requires_some_content():
    r = _client().post("/support/report-issue", data={"title": "", "body": ""})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_report_issue_saves_and_appends():
    client = _client()
    r1 = client.post("/support/report-issue", data={"title": "bug one", "body": "details one"})
    assert r1.get_json()["ok"] is True
    r2 = client.post("/support/report-issue", data={"title": "bug two", "body": "details two"})
    assert r2.get_json()["ok"] is True

    from app.data.storage import get_app_base_dir
    log_path = get_app_base_dir() / "data" / "config" / "issue_reports.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("bug one" in line for line in lines)
    assert any("bug two" in line for line in lines)


def test_profile_picture_upload_serve_and_remove_round_trip():
    client = _client()
    fake_png = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    r1 = client.post(
        "/settings/account/profile-picture",
        data={"profile_picture": (fake_png, "me.png")},
        content_type="multipart/form-data",
    )
    assert r1.status_code == 200
    assert "Remove picture" in r1.get_data(as_text=True)

    r2 = client.get("/settings/account/profile-picture/file")
    assert r2.status_code == 200
    assert r2.content_type == "image/png"

    r3 = client.post("/settings/account/profile-picture/remove")
    assert r3.status_code == 200
    assert "Remove picture" not in r3.get_data(as_text=True)

    r4 = client.get("/settings/account/profile-picture/file")
    assert r4.status_code == 404


def test_profile_picture_rejects_non_image_extension():
    client = _client()
    fake_file = io.BytesIO(b"not an image")
    r = client.post(
        "/settings/account/profile-picture",
        data={"profile_picture": (fake_file, "malware.exe")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert "must be a PNG" in r.get_data(as_text=True)


def test_uploading_a_new_picture_removes_the_old_extension_file():
    """Switching from a .png to a .jpg shouldn't leave the old .png
    sitting around still servable/ambiguous about which is current."""
    client = _client()
    png = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    client.post("/settings/account/profile-picture", data={"profile_picture": (png, "me.png")}, content_type="multipart/form-data")

    from app.data.storage import get_app_base_dir
    config_dir = get_app_base_dir() / "data" / "config"
    assert (config_dir / "profile_picture.png").exists()

    jpg = io.BytesIO(b"\xff\xd8\xff" + b"0" * 20)
    client.post("/settings/account/profile-picture", data={"profile_picture": (jpg, "me.jpg")}, content_type="multipart/form-data")
    assert not (config_dir / "profile_picture.png").exists()
    assert (config_dir / "profile_picture.jpg").exists()

    client.post("/settings/account/profile-picture/remove")
