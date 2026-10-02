"""GUI tests for the Oct 2026 desktop pass: new tabs (Start Here pages, Strategy Library,
Interactive Replay), the first-run tour, the stability layer, and the dashboard cleanup.
(Startup cache / license fast-start / shared Start Here content are covered in
test_startup_speed_and_desktop_parity.py.)

These need a display and skip cleanly on a headless box -- on Linux run them with
`xvfb-run -a pytest tests/test_desktop_gui_tabs.py`."""
from __future__ import annotations

import threading
import time

import pytest

# ---------------------------------------------------------------- GUI (needs a display)
tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="module")
def gui():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    from app.ui import main_window as mw
    from app.ui import tk_safety
    tk_safety.install(root)
    errors = []
    root.report_callback_exception = lambda *a: errors.append(a)
    win = mw.MainWindow(root)
    root.geometry("1300x850+0+0")
    root.update()
    yield win, root, errors
    mw.apply_theme("dark")
    root.destroy()


@pytest.fixture(autouse=True)
def _no_real_data_scan(monkeypatch):
    """Opening the Data Center page starts a real health scan of all of data/raw (minutes of
    pandas work); these tests are about the UI, so stub it."""
    from app.data import health
    monkeypatch.setattr(health, "compute_data_center", lambda: {
        "instruments": [], "total_files": 0, "total_unhealthy": 0, "total_bars": 0, "raw_dir": ""})


def test_new_pages_are_in_the_sidebar_and_every_page_opens(gui):
    win, root, errors = gui
    keys = [k for k, *_ in win._nav_items if k]
    for expected in ("stratlibrary", "replay", "starthere_create", "starthere_test", "starthere_champion",
                     "starthere_deployment", "starthere_graveyard", "starthere_quantlab", "starthere_account"):
        assert expected in keys, expected
    for k in keys:
        win._show_page(k)
        root.update()
    assert not errors, errors[:1]


def test_dashboard_has_no_market_data_library(gui):
    win, root, _ = gui
    assert not hasattr(win, "_dash_library_frame")
    assert not hasattr(win, "_paint_data_library")


def test_start_here_pages_render_their_guide_content(gui):
    from app.orchestration.section_guides import SECTION_GUIDES
    win, root, _ = gui
    win._show_page("starthere_deployment")
    root.update()

    def texts(w):
        out = []
        for c in w.winfo_children():
            if c.winfo_class() == "Label":
                out.append(c.cget("text"))
            out += texts(c)
        return out

    shown = " ".join(texts(win._starthere_frames["deployment"]))
    assert "Deployment \u2014 Start Here" in shown
    assert SECTION_GUIDES["deployment"]["tagline"] in shown


def test_worker_thread_widget_updates_are_applied_on_the_main_thread(gui):
    win, root, _ = gui
    lbl = tk.Label(root, text="a")
    txt = tk.Text(root)
    var = tk.StringVar(value="x")

    def work():
        lbl.config(text="changed")
        for i in range(8000):
            txt.insert(tk.END, f"line {i}\n")
        var.set("set")

    t = threading.Thread(target=work)
    t.start()
    deadline = time.time() + 20
    while time.time() < deadline and (t.is_alive() or not _pending_empty()):
        root.update()
        time.sleep(0.005)
    for _ in range(40):
        root.update()
    assert lbl.cget("text") == "changed" and var.get() == "set"
    lines = int(txt.index("end-1c").split(".")[0])
    from app.ui import tk_safety
    assert lines <= tk_safety.MAX_TEXT_LINES
    assert "line 7999" in txt.get("end-3l", "end")
    lbl.destroy(); txt.destroy()


def _pending_empty():
    from app.ui import tk_safety
    return tk_safety._pending.empty()


def test_update_to_a_destroyed_widget_is_harmless(gui):
    win, root, errors = gui
    w = tk.Label(root)
    t = threading.Thread(target=lambda: (time.sleep(0.05), w.config(text="late")))
    t.start()
    w.destroy()
    for _ in range(40):
        root.update()
        time.sleep(0.01)
    t.join()
    assert not errors


def test_tour_walks_every_step_and_only_shows_once(gui, tmp_path, monkeypatch):
    from app.ui import onboarding_tour as ot
    win, root, _ = gui
    monkeypatch.setattr(ot, "_flag_path", lambda: tmp_path / "tour.json")
    nav_keys = {k for k, *_ in win._nav_items if k}
    assert all(step[0] in nav_keys for step in ot.STEPS)
    assert ot.tour_seen() is False
    tour = ot.maybe_start_tour(win)
    assert tour is not None and tour.card is not None
    for i in range(len(ot.STEPS)):
        tour.go(i)
        root.update()
        assert win.active_page == ot.STEPS[i][0]
    tour.finish("done")
    assert ot.tour_seen() is True and win.active_page == "dashboard"
    assert ot.maybe_start_tour(win) is None  # not shown again


def test_strategy_library_tab_lists_filters_and_shows_details(gui):
    win, root, _ = gui
    win._show_page("stratlibrary")
    root.update()
    from app.ui import extra_tabs
    from app.strategy.library import list_saved_strategies
    extra_tabs._lib_refresh(win)
    total = len(list_saved_strategies())
    assert len(win._lib2_tree.get_children()) == total
    if total:
        first = win._lib2_tree.get_children()[0]
        win._lib2_tree.selection_set(first)
        root.update()
        assert win._lib2_title.cget("text") == first.split("::", 1)[1]
        win._lib2_search.set("zzz-no-such-strategy-zzz")
        extra_tabs._lib_refresh(win)
        assert len(win._lib2_tree.get_children()) == 0
        win._lib2_search.set("")
        extra_tabs._lib_refresh(win)


def test_replay_prepares_plays_and_pauses(gui, tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from app.ui import extra_tabs
    from app.strategy.library import save_strategy_text
    win, root, errors = gui
    n = 600
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1000.0,
    })
    # Pack a real backtest result directly (the exact path _rp_prepare uses after run_backtest).
    from app.backtest.engine import run_backtest
    from app.backtest.risk import RiskConfig
    from app.strategy.manual import ManualStrategy
    strat = ManualStrategy({
        "name": "t", "market": {"timeframe": "15m"},
        "indicators": [{"type": "sma", "period": 5, "column": "close", "as": "f"},
                       {"type": "sma", "period": 20, "column": "close", "as": "s"}],
        "long_entry": "f > s", "long_exit": "f < s", "short_entry": "f < s", "short_exit": "f > s",
        "stop_loss_pips": 30, "take_profit_pips": 60,
    })
    result = run_backtest(df, strat, RiskConfig(initial_balance=50_000, pip_size=0.01))
    data = extra_tabs._rp_pack(df, result, 50_000.0, 10.0, 5.0, 10.0, "unit test")
    assert data["n"] == n and len(data["trades"]) == len(result.trades) > 0
    win._show_page("replay")
    extra_tabs._rp_ready(win, data)
    root.update()
    start = win._rp["i"]
    extra_tabs._rp_step(win, 1)
    assert win._rp["i"] == start + 1
    extra_tabs._rp_step(win, "trade")
    assert any(t["ei"] == win._rp["i"] for t in data["trades"]) or win._rp["i"] == n - 1
    win._rp_speed.set("10x")
    pos = win._rp["i"]
    extra_tabs._rp_toggle(win)
    for _ in range(15):
        root.update()
        time.sleep(0.1)
    assert win._rp["i"] > pos
    extra_tabs._rp_toggle(win)
    assert win._rp["playing"] is False
    win._show_page("dashboard")  # leaving the page also stops playback
    assert not errors
