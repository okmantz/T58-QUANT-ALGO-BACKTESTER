"""Library > Research papers: bundled PDFs, reading-list links, like/read state."""
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.web import pages_routes, server
    root = tmp_path / "app"
    (root / "research" / "papers" / "01_Test_Cat").mkdir(parents=True)
    (root / "research" / "papers" / "01_Test_Cat" / "The Deflated Sharpe Ratio.pdf").write_bytes(b"%PDF-1.4 x")
    (root / "research" / "01_Reading").mkdir(parents=True)
    (root / "research" / "01_Reading" / "Reading_List.md").write_text(
        '1. A - "Linked Paper"\n   - Role: why\n   - Link: https://example.org/p\n2. B - "Have It" [IN REPO: x]\n', encoding="utf-8")
    monkeypatch.setattr(pages_routes, "_base_dir", lambda: root)
    server._license_ok_cached = True
    server.app.config["TESTING"] = True
    c = server.app.test_client()
    with c.session_transaction() as s:
        s["csrf_token"] = "tok"
    return c


def test_lists_bundled_and_linked(client):
    cats = client.get("/api/library/papers").get_json()["categories"]
    lib = next(c for c in cats if c["group"] == "library")
    assert lib["papers"][0]["pdf"] and lib["papers"][0]["authors"].startswith("Bailey")
    rd = next(c for c in cats if c["group"] == "reading")
    assert [p["title"] for p in rd["papers"]] == ["Linked Paper"]  # the [IN REPO] entry is skipped (it is bundled)
    assert client.get("/library-files/papers/papers/01_Test_Cat/The Deflated Sharpe Ratio.pdf").status_code == 200


def test_like_and_read_persist(client):
    pid = "papers/01_Test_Cat/The Deflated Sharpe Ratio"
    r = client.post("/api/library/paper-state", json={"id": pid, "liked": True, "read": True}, headers={"X-CSRF-Token": "tok"})
    assert r.get_json()["ok"]
    p = client.get("/api/library/papers").get_json()["categories"][0]["papers"][0]
    assert p["liked"] and p["read"]
    client.post("/api/library/paper-state", json={"id": pid, "liked": False, "read": False}, headers={"X-CSRF-Token": "tok"})
    p = client.get("/api/library/papers").get_json()["categories"][0]["papers"][0]
    assert not p["liked"] and not p["read"]
