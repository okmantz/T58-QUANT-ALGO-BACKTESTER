"""Oct 2026 desktop work: Start Here pages, Strategy Library tab, Interactive Replay tab,
first-launch tour wiring, redesigned activation window, startup splash, Data Center cache.

GUI tests need a display (they skip cleanly without one -- CI runs them under xvfb)."""
from __future__ import annotations

import re
import threading
import time
import types
from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")

ROOT = Path(__file__).resolve().parents[1]


def _tk_root():
    try:
        return tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")


@pytest.fixture(scope="module")
def window():
    """One real MainWindow for the whole module (building it takes ~2s)."""
    root = _tk_root()
    root.withdraw()
    from app.ui import main_window as mw
    from app.ui import tk_safety
    tk_safety.install(root)
    win = mw.MainWindow(root)
    root.update()
    # Captured NOW, before any test opens pages (the Data Center scan starts when that tab is first shown).
    win._startup_scan_state = (win._datacenter_scanned, win._datacenter_scanning)
    yield win
    try:
        root.destroy()
    except Exception:
        pass
    mw.apply_theme("dark")


_FAKE_REPORT = {"instruments": [], "total_files": 0, "total_unhealthy": 0, "total_bars": 0, "raw_dir": ""}


@pytest.fixture
def no_real_scan(monkeypatch):
    """Visiting the Data Center page starts a REAL health scan of everything under data/raw
    (minutes for hundreds of MB). Tests that only care about navigation / threading use a stub."""
    from app.data import health
    monkeypatch.setattr(health, "compute_data_center", lambda: dict(_FAKE_REPORT))


# ------------------------------------------------------------------ navigation / parity
def test_every_start_here_target_is_a_real_desktop_page(window):
    from app.orchestration.section_guides import DESKTOP_NAV_FOR_HREF, DESKTOP_SECTIONS
    nav_keys = {k for k, *_ in window._nav_items if k}
    assert set(DESKTOP_NAV_FOR_HREF.values()) <= nav_keys
    for section in DESKTOP_SECTIONS:
        assert f"starthere_{section}" in nav_keys


def test_each_section_group_starts_with_start_here(window):
    items = window._nav_items
    for section, header_word in (("create", "CREATE"), ("test", "TEST"), ("champion", "CHAMPION"),
                                 ("deployment", "DEPLOYMENT"), ("graveyard", "GRAVEYARD"),
                                 ("quantlab", "QUANT LAB"), ("account", "ACCOUNT")):
        idx = next(i for i, it in enumerate(items) if it[0] is None and it[1] is None and header_word in it[2])
        assert items[idx + 1][0] == f"starthere_{section}", section


def test_strategy_library_is_in_create_and_replay_is_under_monitor(window):
    keys = [k for k, *_ in window._nav_items]
    assert keys.index("starthere_create") < keys.index("stratlibrary") < keys.index("strategy")
    assert keys.index("livemarket") + 1 == keys.index("replay")


def test_tour_steps_only_visit_real_pages(window):
    from app.ui.onboarding_tour import STEPS
    nav_keys = {k for k, *_ in window._nav_items if k}
    assert {key for key, _title, _body in STEPS} <= nav_keys


def test_every_page_can_be_shown(window, no_real_scan):
    for key in [k for k, *_ in window._nav_items if k]:
        window._show_page(key)
        window.root.update()
    assert window.active_page == [k for k, *_ in window._nav_items if k][-1]


def test_stage_stepper_covers_new_pages(window):
    for key, stage in (("validatehub", "Validate"), ("optimizehub", "Optimize"), ("starthere_test", "Test"),
                       ("stratlibrary", "Create"), ("replay", "Monitor"), ("starthere_deployment", "Forward Test")):
        window.active_page = key
        assert window._current_stage_name() == stage, key
    window.active_page = "dashboard"


# ------------------------------------------------------------------ startup speed
def test_market_data_library_is_gone_from_the_dashboard(window):
    assert not hasattr(window, "_dash_library_frame")
    assert not hasattr(window, "_paint_data_library")


def test_data_center_does_not_scan_at_startup(window):
    assert window._startup_scan_state == (False, False)


def test_data_center_scans_in_background_when_opened(window, no_real_scan):
    window._show_page("datacenter")
    end = time.time() + 30
    while time.time() < end and not window._datacenter_scanned:
        window.root.update()
        time.sleep(0.05)
    assert window._datacenter_scanned


def test_health_results_are_cached_per_file(tmp_path, monkeypatch):
    from app.data import health
    csv = tmp_path / "ES" / "a.csv"
    csv.parent.mkdir()
    csv.write_text("timestamp,open,high,low,close,volume\n2026-01-01 00:00,1,2,0.5,1.5,10\n")
    ds = types.SimpleNamespace(name="ES/a.csv", path=csv, size_bytes=csv.stat().st_size)
    monkeypatch.setattr(health, "list_stored_datasets", lambda: [ds])
    monkeypatch.setattr(health, "get_raw_data_dir", lambda: tmp_path)
    monkeypatch.setattr(health, "_health_cache_path", lambda: tmp_path / "health_cache.json")
    calls = []
    real = health.compute_file_health
    monkeypatch.setattr(health, "compute_file_health", lambda p, n: (calls.append(n), real(p, n))[1])
    first = health.compute_data_center()
    second = health.compute_data_center()
    assert len(calls) == 1 and first == second          # second scan came from the cache
    csv.write_text(csv.read_text() + "2026-01-01 00:05,1,2,0.5,1.5,10\n")   # file changed -> rescanned
    health.compute_data_center()
    assert len(calls) == 2


# ------------------------------------------------------------------ Strategy Library tab
def test_strategy_library_lists_and_shows_a_selection(window):
    from app.ui import extra_tabs
    window._show_page("stratlibrary")
    window.root.update()
    extra_tabs._lib_refresh(window)
    rows = window._lib2_tree.get_children()
    if not rows:
        pytest.skip("library is empty in this environment")
    window._lib2_tree.selection_set(rows[0])
    window.root.update()
    assert window._lib2_title.cget("text") == window._lib2["items"][rows[0]].name
    assert window._lib2_code.get("1.0", "end").strip()
    window._lib2_search.set("zzz-no-such-strategy")
    extra_tabs._lib_refresh(window)
    assert not window._lib2_tree.get_children()
    window._lib2_search.set("")
    extra_tabs._lib_refresh(window)


# ------------------------------------------------------------------ Replay
def _fake_result(df, trades):
    import pandas as pd
    eq = pd.DataFrame({"timestamp": df["timestamp"], "equity": [10_000.0 + i for i in range(len(df))]})
    return types.SimpleNamespace(trades=trades, equity_curve=eq)


def test_replay_pack_maps_trades_to_bars_and_tracks_prop_state():
    import numpy as np
    import pandas as pd
    from app.ui import extra_tabs
    n = 50
    ts = pd.date_range("2026-01-01 09:00", periods=n, freq="1h")
    df = pd.DataFrame({"timestamp": ts, "open": np.linspace(100, 110, n), "high": np.linspace(101, 111, n),
                       "low": np.linspace(99, 109, n), "close": np.linspace(100.5, 110.5, n), "volume": 1.0})
    trade = types.SimpleNamespace(
        entry_time=ts[10], exit_time=ts[20], direction=1, entry_price=102.0, exit_price=104.0, pnl=398.0,
        commission=2.0, exit_reason="take_profit", equity_after=10_398.0, initial_risk=1.0)
    d = extra_tabs._rp_pack(df, _fake_result(df, [trade]), 10_000.0, 10.0, 5.0, 10.0, "label")
    t = d["trades"][0]
    assert (t["ei"], t["xi"], t["dir"]) == (10, 20, 1)
    assert t["sl"] == pytest.approx(101.0)                      # entry - dir * initial_risk
    assert t["per_unit"] == pytest.approx(200.0)                # (pnl + commission) / favourable move
    assert d["n"] == n and len(d["eq"]) == n and d["eq"][5] == pytest.approx(10_005.0)
    assert d["day_start"][0] == pytest.approx(10_000.0)         # first bar of day 1 starts at the account size
    assert d["peak"][-1] == pytest.approx(max(d["eq"]))


def test_replay_price_format_scales_with_instrument():
    from app.ui.extra_tabs import _px
    assert _px(1.17436) == "1.17436" and _px(150.1234) == "150.123" and _px(5012.5) == "5,012.50"


def test_replay_playback_steps_and_stops_at_the_end(window):
    import numpy as np
    import pandas as pd
    from app.ui import extra_tabs
    n = 80
    ts = pd.date_range("2026-01-01", periods=n, freq="1h")
    df = pd.DataFrame({"timestamp": ts, "open": np.linspace(100, 101, n), "high": np.linspace(101, 102, n),
                       "low": np.linspace(99, 100, n), "close": np.linspace(100, 101, n), "volume": 1.0})
    data = extra_tabs._rp_pack(df, _fake_result(df, []), 10_000.0, 10.0, 5.0, 10.0, "synthetic")
    window._show_page("replay")
    extra_tabs._rp_ready(window, data)
    window.root.update()
    assert window._rp["i"] == 60
    extra_tabs._rp_step(window, -1)
    assert window._rp["i"] == 59
    extra_tabs._rp_step(window, "start")
    assert window._rp["i"] == 0
    window._rp_speed.set("50x")
    extra_tabs._rp_toggle(window)
    end = time.time() + 5
    while time.time() < end and window._rp["playing"]:
        window.root.update()
        time.sleep(0.02)
    assert window._rp["i"] == n - 1 and not window._rp["playing"]
    window._show_page("dashboard")


# ------------------------------------------------------------------ launch / tour / splash / activation
def test_launch_starts_the_tour_and_installs_the_stability_layer():
    src = (ROOT / "app" / "ui" / "main_window.py").read_text(encoding="utf-8")
    launch = src[src.index("def launch("):]
    assert "tk_safety.install(root)" in launch
    assert "maybe_start_tour(window)" in launch


def test_main_launches_through_one_shared_root_and_splash():
    src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "_launch_gui()" in src and "StartupSplash(root)" in src
    assert "launch(root=root, splash=splash)" in src
    assert len(re.findall(r"\bensure_licensed\(", src)) == 1


def test_startup_splash_shows_updates_and_closes():
    root = _tk_root()
    root.withdraw()
    from app.ui.splash import StartupSplash
    splash = StartupSplash(root)
    splash.set_status("Loading Search Lab...")
    assert splash._status.cget("text") == "Loading Search Lab..."
    before = splash._pos
    splash.set_status("Building navigation...")
    assert splash._pos > before
    splash.close()
    assert not splash.win.winfo_exists()
    root.destroy()


def _drive_activation_window(monkeypatch, drive):
    """Runs show_activation_window with `drive(window)` scheduled once its mainloop starts."""
    orig = tk.Misc.mainloop

    def patched(self, n=0):
        if not hasattr(self, "email_var"):
            return orig(self, n)
        # Own bounded loop instead of Tk's mainloop(): that only returns once EVERY Tk window in
        # the process is gone, and the module-scoped MainWindow fixture keeps one alive.
        self.after(50, lambda: drive(self))
        end = time.time() + 30
        try:
            while time.time() < end and self.winfo_exists():
                self.update()
                time.sleep(0.01)
        except tk.TclError:
            pass  # the window was destroyed -- that is the normal way out

    monkeypatch.setattr(tk.Misc, "mainloop", patched)
    from app.licensing import gate
    try:
        return gate.show_activation_window()
    except tk.TclError:
        pytest.skip("no display available")


def test_activation_window_activates_on_a_worker_thread_and_closes(monkeypatch):
    from app.licensing import client
    seen = {}

    def fake_activate(email, key, remember=True):
        seen["thread"] = threading.current_thread()
        seen["args"] = (email, key, remember)
        return True, "Activated."

    monkeypatch.setattr(client, "activate", fake_activate)
    monkeypatch.setattr(client, "validate", lambda: pytest.fail("no network validate after activating"))

    def drive(win):
        win.email_var.set("me@example.com")
        win.key_entry.focus_force()
        win.key_var.set("t58-aaaa-bbbb-cccc-dddd")
        win.key_entry.configure(fg="#E7EBF2")
        win._on_activate()

    assert _drive_activation_window(monkeypatch, drive) is True
    assert seen["thread"] is not threading.main_thread()           # the window never freezes on the network
    assert seen["args"][0] == "me@example.com" and seen["args"][2] is True


def test_activation_window_shows_errors_and_stays_open(monkeypatch):
    from app.licensing import client
    monkeypatch.setattr(client, "activate", lambda e, k, remember=True: (False, "That license key wasn't found."))
    out = {}

    def drive(win):
        win.email_var.set("me@example.com")
        win.key_var.set("T58-AAAA-BBBB-CCCC-DDDD")
        win._on_activate()
        end = time.time() + 5
        while time.time() < end and not win.status_var.get():
            win.update()
            time.sleep(0.02)
        out["status"], out["btn"], out["state"] = win.status_var.get(), win.activate_btn.cget("text"), str(win.activate_btn.cget("state"))
        win.destroy()

    assert _drive_activation_window(monkeypatch, drive) is False
    assert "wasn't found" in out["status"] and out["btn"] == "Activate" and out["state"] == "normal"


def test_activation_window_requires_both_fields(monkeypatch):
    from app.licensing import client
    monkeypatch.setattr(client, "activate", lambda *a, **k: pytest.fail("must not call the server with empty fields"))
    out = {}

    def drive(win):
        win._on_activate()           # email empty, key is the placeholder
        out["status"] = win.status_var.get()
        win.destroy()

    _drive_activation_window(monkeypatch, drive)
    assert "both" in out["status"].lower()
