"""Tests for app.ui.main_window's theme system -- covers the pure-logic
pieces (apply_theme, persistence) without needing a real Tk display."""
from __future__ import annotations

import json
import threading
from unittest import mock

import pytest

from app.ui import main_window as mw


@pytest.fixture(autouse=True)
def _restore_dark_theme_after_each_test():
    """Every test in this module mutates module-global color constants --
    always leave them back on 'dark' afterward so other test modules that
    import main_window aren't affected by test order."""
    yield
    mw.apply_theme("dark")


def test_both_themes_define_every_color_key():
    dark_keys = set(mw.THEMES["dark"].keys())
    light_keys = set(mw.THEMES["light"].keys())
    assert dark_keys == light_keys


def test_apply_theme_updates_module_globals():
    mw.apply_theme("dark")
    assert mw.BG == mw.THEMES["dark"]["BG"]
    mw.apply_theme("light")
    assert mw.BG == mw.THEMES["light"]["BG"]
    assert mw.CURRENT_THEME == "light"


def test_apply_theme_ignores_unknown_name():
    mw.apply_theme("dark")
    before = mw.BG
    mw.apply_theme("not_a_real_theme")
    assert mw.BG == before
    assert mw.CURRENT_THEME == "dark"


def test_refresh_dashboard_reschedules_via_after_off_the_main_thread():
    """Reproduces the reported intermittent freeze: Search Lab, Full
    Pipeline, Speed Run, and the single-strategy backtest run all call
    _refresh_dashboard() straight from their own background worker
    thread once they finish. Tkinter widgets are only safe to touch from
    the main thread -- this must detect that and reschedule itself via
    root.after instead of touching any widget, exactly like Batch Test's
    own (already-correct) _refresh_after_run pattern."""
    fake_self = mock.MagicMock()
    outcome = {}

    def _call_from_worker_thread():
        try:
            mw.MainWindow._refresh_dashboard(fake_self)
            outcome["ran"] = True
        except Exception as exc:  # pragma: no cover -- would fail the test below
            outcome["exc"] = exc

    t = threading.Thread(target=_call_from_worker_thread)
    t.start()
    t.join(timeout=5)

    assert outcome.get("ran") is True, outcome.get("exc")
    fake_self.root.after.assert_called_once_with(0, fake_self._refresh_dashboard)
    # The guard must return before touching any dashboard widget.
    fake_self._dash_stats_row.winfo_children.assert_not_called()


def test_refresh_strategy_library_reschedules_via_after_off_the_main_thread():
    """Same cross-thread hazard, same fix, for the other shared refresh
    helper several pipelines also call directly after finishing."""
    fake_self = mock.MagicMock()
    outcome = {}

    def _call_from_worker_thread():
        try:
            mw.MainWindow._refresh_strategy_library(fake_self)
            outcome["ran"] = True
        except Exception as exc:  # pragma: no cover
            outcome["exc"] = exc

    t = threading.Thread(target=_call_from_worker_thread)
    t.start()
    t.join(timeout=5)

    assert outcome.get("ran") is True, outcome.get("exc")
    fake_self.root.after.assert_called_once_with(0, fake_self._refresh_strategy_library)
    fake_self.strategy_library_listbox.delete.assert_not_called()


def test_apply_theme_persists_choice(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.storage.get_app_base_dir", lambda: tmp_path)
    mw.apply_theme("light")
    saved = json.loads((tmp_path / "data" / "config" / "ui_theme.json").read_text())
    assert saved["theme"] == "light"
    mw.apply_theme("dark")
    saved = json.loads((tmp_path / "data" / "config" / "ui_theme.json").read_text())
    assert saved["theme"] == "dark"


def test_load_theme_name_defaults_to_dark_when_nothing_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.storage.get_app_base_dir", lambda: tmp_path)
    assert mw._load_theme_name() == "dark"


def test_load_theme_name_reads_persisted_choice(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.storage.get_app_base_dir", lambda: tmp_path)
    config_dir = tmp_path / "data" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "ui_theme.json").write_text(json.dumps({"theme": "light"}))
    assert mw._load_theme_name() == "light"


def test_load_theme_name_ignores_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.storage.get_app_base_dir", lambda: tmp_path)
    config_dir = tmp_path / "data" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "ui_theme.json").write_text("{not valid json")
    assert mw._load_theme_name() == "dark"


def test_light_theme_status_colors_differ_from_dark():
    """The semantic pass/fail/info/warning colors must actually change
    between themes, not just the backgrounds -- a light theme reusing the
    neon-bright dark-theme green/red would be unreadable on a white panel."""
    assert mw.THEMES["dark"]["GREEN"] != mw.THEMES["light"]["GREEN"]
    assert mw.THEMES["dark"]["RED"] != mw.THEMES["light"]["RED"]
