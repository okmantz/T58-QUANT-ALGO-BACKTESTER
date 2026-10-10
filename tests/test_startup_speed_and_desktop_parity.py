"""Regression tests for the Oct 2026 desktop startup/parity work:
row-count cache, license fast-start, shared Start Here content, thread-safe Tk layer."""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ------------------------------------------------------------------ row-count cache
@pytest.fixture
def storage(tmp_path, monkeypatch):
    from app.data import storage
    monkeypatch.setattr(storage, "_row_cache_path", lambda: tmp_path / "row_count_cache.json")
    monkeypatch.setattr(storage, "_ROW_COUNT_CACHE", None)
    monkeypatch.setattr(storage, "_ROW_COUNT_CACHE_DIRTY", False)
    return storage


@pytest.mark.parametrize("text", ["h\n1\n2\n3\n", "h\n1\n2\n3", "h\n", "h", ""])
def test_quick_row_count_matches_old_line_iteration(storage, tmp_path, text):
    f = tmp_path / "d.csv"
    f.write_bytes(text.encode())
    old = max(sum(1 for _ in open(f, "rb")) - 1, 0)
    assert storage._quick_row_count(f) == old


def test_row_count_is_cached_until_file_changes(storage, tmp_path, monkeypatch):
    f = tmp_path / "d.csv"
    f.write_text("h\n1\n2\n")
    calls = []
    real = storage._count_newlines
    monkeypatch.setattr(storage, "_count_newlines", lambda p: (calls.append(1), real(p))[1])
    assert storage._quick_row_count(f) == 2
    assert storage._quick_row_count(f) == 2
    assert len(calls) == 1                      # second call served from the cache
    f.write_text("h\n1\n2\n3\n4\n")             # file changed -> recount
    assert storage._quick_row_count(f) == 4
    assert len(calls) == 2


def test_row_cache_persists_across_sessions(storage, tmp_path, monkeypatch):
    f = tmp_path / "d.csv"
    f.write_text("h\n1\n2\n")
    storage._quick_row_count(f)
    storage.flush_row_count_cache()
    monkeypatch.setattr(storage, "_ROW_COUNT_CACHE", None)   # simulate a fresh process
    monkeypatch.setattr(storage, "_count_newlines", lambda p: pytest.fail("should come from the on-disk cache"))
    assert storage._quick_row_count(f) == 2


def test_picker_listing_never_reads_files(storage, tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "ES"
    raw.mkdir(parents=True)
    (raw / "a.csv").write_text("timestamp,open,high,low,close,volume\n2026-01-01,1,2,0.5,1.5,10\n")
    monkeypatch.setattr(storage, "get_raw_data_dir", lambda: tmp_path / "raw")
    monkeypatch.setattr(storage, "_count_newlines", lambda p: pytest.fail("pickers must not count rows"))
    groups = storage.list_datasets_by_instrument(count_rows=False)
    assert groups[0]["file_count"] == 1
    assert groups[0]["files"][0]["rows"] == storage.ROWS_NOT_COUNTED
    assert groups[0]["files"][0]["empty"] is False


# ------------------------------------------------------------------ license fast start
@pytest.fixture
def lic(monkeypatch):
    from app.licensing import client
    state = client.LicenseState(
        email="a@b.co", license_key="T58-AAAA-BBBB-CCCC-DDDD", device_id="d", status="active",
        expires_at=None, last_validated_at=datetime.now(timezone.utc).isoformat(),
    )
    monkeypatch.setattr(client, "load_state", lambda: state)
    monkeypatch.setattr(client, "_is_master_key", lambda k: False)
    return client, state


def test_fast_start_does_not_block_on_network_when_recently_verified(lic, monkeypatch):
    client, _state = lic
    started = threading.Event()
    monkeypatch.setattr(client, "validate", lambda: (started.set(), time.sleep(2), (True, "x"))[2])
    t0 = time.time()
    ok, _msg = client.validate_fast_start()
    assert ok and time.time() - t0 < 0.5           # returned immediately...
    assert started.wait(2)                         # ...while a background re-check really started


def test_fast_start_does_a_full_validate_when_check_is_stale(lic, monkeypatch):
    client, state = lic
    state.last_validated_at = (datetime.now(timezone.utc) - timedelta(hours=client.FAST_START_MAX_AGE_HOURS + 1)).isoformat()
    monkeypatch.setattr(client, "validate", lambda: (False, "revoked"))
    assert client.validate_fast_start() == (False, "revoked")


def test_fast_start_never_trusts_a_non_active_license(lic, monkeypatch):
    client, state = lic
    state.status = "revoked"
    monkeypatch.setattr(client, "validate", lambda: (False, "revoked"))
    assert client.validate_fast_start() == (False, "revoked")


def test_ensure_licensed_does_not_revalidate_after_activation(monkeypatch):
    from app.licensing import client, gate
    monkeypatch.setattr(client, "validate_fast_start", lambda: (False, "Not activated."))
    monkeypatch.setattr(client, "load_state", lambda: client.LicenseState(email=""))
    monkeypatch.setattr(gate, "show_activation_window", lambda **kw: True)
    monkeypatch.setattr(client, "validate", lambda: pytest.fail("a second network validate must not run after activate()"))
    assert gate.ensure_licensed(interactive=True) is True


def test_main_checks_the_license_exactly_once():
    src = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
    assert len(re.findall(r"\bensure_licensed\(", src)) == 1


# ------------------------------------------------------------------ shared Start Here content
def test_every_start_here_tool_has_a_desktop_page():
    from app.orchestration.section_guides import DESKTOP_NAV_FOR_HREF, DESKTOP_SECTIONS, SECTION_GUIDES
    for section in DESKTOP_SECTIONS:
        assert section in SECTION_GUIDES
        for tool in SECTION_GUIDES[section]["tools"]:
            assert tool["href"] in DESKTOP_NAV_FOR_HREF, (section, tool)


def test_web_and_desktop_share_one_source_of_start_here_text():
    from app.orchestration.section_guides import SECTION_GUIDES
    from app.web import server
    assert server._SECTION_START_HERE is SECTION_GUIDES


def test_web_start_here_pages_still_render():
    from app.orchestration.section_guides import SECTION_GUIDES
    from app.web import server
    client = server.app.test_client()
    # v9.15 (Owen): the Quant Lab and Account Start Here pages were
    # deliberately removed from the WEB app (Quant Lab's explainer moved
    # onto the Quant Lab page itself). Their shared SECTION_GUIDES data
    # stays for the DESKTOP app, which still renders its own Start Here
    # pages for them -- so on web they must 404, not 200. Every other
    # section still renders.
    web_removed = {"quantlab", "account"}
    for section in SECTION_GUIDES:
        resp = client.get(f"/start-here/{section}")
        if section in web_removed:
            assert resp.status_code == 404, section
        else:
            assert resp.status_code == 200, section


# ------------------------------------------------------------------ Tk thread-safety layer
def _tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        return tk, tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")


def test_worker_thread_widget_updates_are_applied_on_the_main_thread():
    tk, root = _tk_root()
    from app.ui import tk_safety
    tk_safety.install(root)
    label, var, text = tk.Label(root, text="a"), tk.StringVar(master=root), tk.Text(root)
    t = threading.Thread(target=lambda: (label.config(text="from-thread"), var.set("v"), text.insert("end", "x\n")))
    t.start()
    t.join()
    end = time.time() + 3
    while time.time() < end and var.get() != "v":
        root.update()
        time.sleep(0.01)
    assert label.cget("text") == "from-thread" and var.get() == "v" and text.get("1.0", "end").strip() == "x"
    root.destroy()


def test_update_to_destroyed_widget_is_harmless():
    tk, root = _tk_root()
    from app.ui import tk_safety
    tk_safety.install(root)
    w = tk.Label(root)
    t = threading.Thread(target=lambda: w.config(text="late"))
    t.start()
    t.join()
    w.destroy()
    for _ in range(20):
        root.update()
    root.destroy()


def test_log_text_widgets_are_bounded():
    tk, root = _tk_root()
    from app.ui import tk_safety
    tk_safety.install(root)
    text = tk.Text(root)
    for i in range(tk_safety.MAX_TEXT_LINES + 2000):
        text.insert("end", f"line {i}\n")
    assert int(text.index("end-1c").split(".")[0]) <= tk_safety.MAX_TEXT_LINES + tk_safety._TRIM_CHECK_EVERY
    assert text.get("end-2l", "end-1c").strip().startswith("line")
    root.destroy()
