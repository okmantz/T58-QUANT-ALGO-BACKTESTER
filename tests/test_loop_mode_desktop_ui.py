"""Real-Tk smoke tests for this round's desktop Loop Mode additions --
Search Lab's new "Loop mode" section and Evolution Lab's new loop-mode
fields. Follows the same pattern as tests/test_batch_results_ui.py: a
minimal fake window (just `self.root` plus whatever bare attributes the
tab-builder method needs before it reaches the new section) with the real,
unbound MainWindow methods bound onto it, driving a real Tk root (via
Xvfb), rather than either mocking Tkinter or constructing the entire
~15,000-line app.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("tkinter")
tk = pytest.importorskip("tkinter")

try:
    _probe = tk.Tk()
    _probe.destroy()
except Exception as exc:  # noqa: BLE001 -- no display available in this environment
    pytest.skip(f"no Tk display available: {exc}", allow_module_level=True)

from app.ui import main_window as mw  # noqa: E402


class _FakeMainWindow:
    """Minimal stand-in for MainWindow -- just enough state for
    _build_search_tab / _build_evolution_lab_tab (and the handful of
    helper methods they call) to run for real against a real Tk root."""

    def __init__(self, root):
        self.root = root
        self.csv_paths: list = []
        self._bulk_strategy_paths: list = []
        self._search_pair_csv_path = None
        self._active_library_strategy = None
        self._last_search_summary = None
        self._last_search_space = None
        self._last_search_html_path = None
        self._last_champion_html_path = None
        self._last_search_db_path = None
        self._last_search_df = None
        self._last_search_risk = None
        self._last_search_rules = None
        self._selected_library_items: list = []
        self._SEARCH_MODE_LABELS = mw.MainWindow._SEARCH_MODE_LABELS
        self._STRATEGY_TYPE_FILEDIALOG = mw.MainWindow._STRATEGY_TYPE_FILEDIALOG


def _bind_real_methods(fake, *names):
    for name in names:
        setattr(fake, name, types.MethodType(getattr(mw.MainWindow, name), fake))


@pytest.fixture
def root():
    r = tk.Tk()
    yield r
    try:
        r.destroy()
    except Exception:
        pass


_SHARED_HELPER_METHODS = (
    "_page_header", "_button", "_scrollable", "_section",
    "_bind_isolated_wheel", "_bind_isolated_wheel_tree",
)


def test_search_tab_builds_with_loop_mode_widgets(root):
    fake = _FakeMainWindow(root)
    fake.tab_search = tk.Frame(root)
    _bind_real_methods(
        fake, *_SHARED_HELPER_METHODS,
        "_build_search_tab", "_search_choose_pair_csv", "_search_clear_pair_csv",
        "_search_run_clicked", "_search_stop_clicked", "_promote_search_champion_clicked",
        "_open_search_report", "_open_champion_report", "_bulk_add_files", "_bulk_add_files_by_type",
        "_on_search_mode_changed", "_bulk_add_from_library", "_bulk_clear_files", "_bulk_remove_selected",
    )
    fake._build_search_tab()  # must not raise

    # The new Loop Mode widgets exist, with the expected defaults.
    assert fake.search_loop_mode.get() is False
    assert fake.search_loop_target.get_float() == 60.0
    assert fake.search_loop_max_rounds.get_int() == 20
    assert fake.search_loop_time_budget_hours.get_str() == ""
    assert fake.search_loop_stall_rounds.get_int() == 2

    # Toggling the checkbox and changing values works like any other
    # LabeledEntry/LabeledCheckbox on this tab.
    fake.search_loop_mode.var.set(True)
    assert fake.search_loop_mode.get() is True
    fake.search_loop_target.var.set("75")
    assert fake.search_loop_target.get_float() == 75.0


def test_search_run_clicked_warns_instead_of_starting_loop_mode_in_single_mode(root, monkeypatch):
    """Loop Mode only supports the named-family/all-families search mode --
    picking it together with Single-strategy mode must show a warning and
    NOT start a background job (mirrors the analogous web-side guard)."""
    fake = _FakeMainWindow(root)
    fake.tab_search = tk.Frame(root)
    _bind_real_methods(
        fake, *_SHARED_HELPER_METHODS,
        "_build_search_tab", "_search_choose_pair_csv", "_search_clear_pair_csv",
        "_search_run_clicked", "_search_stop_clicked", "_promote_search_champion_clicked",
        "_open_search_report", "_open_champion_report", "_bulk_add_files", "_bulk_add_files_by_type",
        "_on_search_mode_changed", "_bulk_add_from_library", "_bulk_clear_files", "_bulk_remove_selected",
    )
    fake._build_search_tab()
    fake.csv_paths = ["dummy.csv"]
    fake.search_loop_mode.var.set(True)
    fake.search_mode.var.set(next(k for k, v in fake._SEARCH_MODE_LABELS.items() if v == "single"))

    started = {"loop": False, "normal": False}
    fake._try_start_heavy_job = lambda name: True
    fake._release_heavy_job = lambda name: None
    monkeypatch.setattr(mw.threading, "Thread", lambda target, daemon=True: types.SimpleNamespace(
        start=lambda: started.__setitem__(
            "loop" if target == fake._search_run_loop_pipeline else "normal", True
        )
    ))
    warned = {"n": 0}
    monkeypatch.setattr(mw.messagebox, "showwarning", lambda *a, **k: warned.__setitem__("n", warned["n"] + 1))

    fake._search_run_clicked()

    assert warned["n"] == 1
    assert started["loop"] is False
    assert started["normal"] is False


def test_evolution_lab_tab_builds_with_loop_mode_widgets(root, monkeypatch):
    """_build_evolution_lab_tab shares several StringVars with OTHER tabs
    (03 Prop Rules / 04 Risk -- see the "shared live" comment right above
    account_section in main_window.py), so building it fully standalone
    would mean faking most of the rest of the app's tabs too. Instead this
    drives the two new widgets directly (same LabeledEntry/LabeledCombo
    classes, same construction the real tab uses) and the new
    _poll_evolution_status banner logic against a real Label widget --
    real Tkinter throughout, just without the full ~15,000-line tab tree."""
    section = tk.Frame(root, bg=mw.PANEL)
    evo_loop_target = mw.LabeledEntry(section, "Loop mode target %", "")
    evo_loop_target_metric = mw.LabeledCombo(
        section, "Loop mode target metric",
        ["CPCV out-of-sample eval-pass % (honest, recommended)", "Raw in-sample eval-pass % (Monte Carlo only)"],
        "CPCV out-of-sample eval-pass % (honest, recommended)",
    )
    assert evo_loop_target.get_str() == ""
    assert evo_loop_target_metric.get_str().startswith("CPCV")
    evo_loop_target.var.set("72.5")
    assert evo_loop_target.get_float() == 72.5

    fake = types.SimpleNamespace(root=root)
    fake.evo_status_label = tk.Label(root)
    fake.evo_loop_status_label = tk.Label(root)
    fake._evo_guide_shown = True  # skip the after-stop guide-text branch, unrelated to this test
    fake._release_heavy_job = lambda name: None
    fake._refresh_evo_leaderboard_listbox = lambda rows: None
    fake._evolution_runner = types.SimpleNamespace(
        status=lambda: {
            "running": False, "generation": 5, "leaderboard_size": 3, "resumed": False,
            "family_health": None, "target_eval_pass_pct": 60.0,
            "target_reached": True, "target_reached_candidate_id": "abc123",
        },
        leaderboard=[],
    )
    mw.MainWindow._poll_evolution_status(fake)
    assert "abc123" in fake.evo_loop_status_label.cget("text")
    assert "60.0" in fake.evo_loop_status_label.cget("text")

    # And the "active but not yet reached" phrasing when target_reached is False.
    fake._evolution_runner.status = lambda: {
        "running": True, "generation": 2, "leaderboard_size": 1, "resumed": False,
        "family_health": None, "target_eval_pass_pct": 60.0,
        "target_reached": False, "target_reached_candidate_id": None,
    }
    mw.MainWindow._poll_evolution_status(fake)
    assert "60.0" in fake.evo_loop_status_label.cget("text")
    assert "abc123" not in fake.evo_loop_status_label.cget("text")

    # And blank when no target is configured at all (the old/default behavior).
    fake._evolution_runner.status = lambda: {
        "running": False, "generation": 1, "leaderboard_size": 0, "resumed": False,
        "family_health": None, "target_eval_pass_pct": None,
        "target_reached": False, "target_reached_candidate_id": None,
    }
    mw.MainWindow._poll_evolution_status(fake)
    assert fake.evo_loop_status_label.cget("text") == ""


def test_forge_tab_builds_with_loop_mode_widgets(root):
    fake = _FakeMainWindow(root)
    fake.tab_forge = tk.Frame(root)
    fake._forge_cancel_event = None
    fake._last_forge_graveyard_path = None
    fake._last_forge_result = None
    _bind_real_methods(
        fake, *_SHARED_HELPER_METHODS,
        "_build_forge_tab", "_forge_run_clicked", "_forge_stop_clicked", "_open_forge_graveyard",
    )
    fake._build_forge_tab()  # must not raise

    assert fake.forge_loop_mode.get() is False
    assert fake.forge_loop_target.get_float() == 60.0
    assert fake.forge_loop_max_rounds.get_int() == 10
    assert fake.forge_loop_time_budget_hours.get_str() == ""
    assert fake.forge_loop_stall_rounds.get_int() == 2
    assert fake.forge_loop_require_locked_oos.get() is True

    fake.forge_loop_mode.var.set(True)
    assert fake.forge_loop_mode.get() is True


def test_forge_run_clicked_starts_the_loop_pipeline_thread_when_enabled(root, monkeypatch):
    fake = _FakeMainWindow(root)
    fake.tab_forge = tk.Frame(root)
    fake._forge_cancel_event = __import__("threading").Event()
    fake._last_forge_graveyard_path = None
    fake._last_forge_result = None
    _bind_real_methods(
        fake, *_SHARED_HELPER_METHODS,
        "_build_forge_tab", "_forge_run_clicked", "_forge_stop_clicked", "_open_forge_graveyard",
        "_forge_run_loop_pipeline",
    )
    fake._build_forge_tab()
    fake.csv_paths = ["dummy.csv"]
    fake.forge_loop_mode.var.set(True)

    started = {"loop": False, "normal": False}
    fake._try_start_heavy_job = lambda name: True
    fake._release_heavy_job = lambda name: None
    monkeypatch.setattr(mw.threading, "Thread", lambda target, daemon=True: types.SimpleNamespace(
        start=lambda: started.__setitem__(
            "loop" if target == fake._forge_run_loop_pipeline else "normal", True
        )
    ))
    fake._forge_run_clicked()

    assert started["loop"] is True
    assert started["normal"] is False


def test_speedrun_tab_builds_with_loop_mode_widgets(root):
    fake = _FakeMainWindow(root)
    fake.tab_speedrun = tk.Frame(root)
    _bind_real_methods(
        fake, *_SHARED_HELPER_METHODS,
        "_build_speedrun_tab", "_speedrun_run_clicked", "_speedrun_stop_clicked",
        "_open_speedrun_winner_report", "_open_speedrun_selected_candidate_report",
        "_view_speedrun_selected_candidate_code", "_save_speedrun_selected_candidate",
    )
    fake._build_speedrun_tab()  # must not raise

    assert fake.sr_loop_mode.get() is False
    assert fake.sr_loop_max_rounds.get_int() == 10
    assert fake.sr_loop_time_budget_hours.get_str() == ""
    assert fake.sr_loop_stall_rounds.get_int() == 2

    fake.sr_loop_mode.var.set(True)
    assert fake.sr_loop_mode.get() is True


def test_speedrun_run_clicked_starts_the_loop_pipeline_thread_when_enabled(root, monkeypatch):
    fake = _FakeMainWindow(root)
    fake.tab_speedrun = tk.Frame(root)
    _bind_real_methods(
        fake, *_SHARED_HELPER_METHODS,
        "_build_speedrun_tab", "_speedrun_run_clicked", "_speedrun_stop_clicked",
        "_open_speedrun_winner_report", "_open_speedrun_selected_candidate_report",
        "_view_speedrun_selected_candidate_code", "_save_speedrun_selected_candidate",
        "_speedrun_run_loop_pipeline",
    )
    fake._build_speedrun_tab()
    fake.csv_paths = ["dummy.csv"]
    fake.sr_loop_mode.var.set(True)

    started = {"loop": False, "normal": False}
    fake._try_start_heavy_job = lambda name: True
    fake._release_heavy_job = lambda name: None
    monkeypatch.setattr(mw.threading, "Thread", lambda target, daemon=True: types.SimpleNamespace(
        start=lambda: started.__setitem__(
            "loop" if target == fake._speedrun_run_loop_pipeline else "normal", True
        )
    ))
    fake._speedrun_run_clicked()

    assert started["loop"] is True
    assert started["normal"] is False
