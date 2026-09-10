"""
Tests for app.ui.main_window._open_progress_window's per-strategy RESULTS
table -- the desktop-side fix for "when I ran multiple strategies through
Full Pipeline, it spit out one metric and one report". Before this, RUN
FULL PIPELINE (BATCH) (CHECKED) and RUN BATCH TEST (CHECKED) both wrote a
real, individual report per strategy to disk (see
app.orchestration.full_pipeline.run_full_pipeline_batch /
app.orchestration.batch_test.run_batch_test), but the progress window only
ever showed a plain text log -- there was no way to open any strategy's
own report except whatever the single-strategy Run & Report / Full
Pipeline tab last produced, which read exactly like the whole batch only
ever tested and reported on one strategy.

These tests drive the real Tkinter widgets (via Xvfb / a real Tk root,
skipped cleanly if tkinter or a display isn't available) rather than
mocking them, since the bug was specifically about what the actual UI
shows and lets you click.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("tkinter")
tk = pytest.importorskip("tkinter")
from tkinter import ttk  # noqa: E402

try:
    _probe = tk.Tk()
    _probe.destroy()
except Exception as exc:  # noqa: BLE001 -- no display available in this environment
    pytest.skip(f"no Tk display available: {exc}", allow_module_level=True)

from app.ui import main_window as mw  # noqa: E402


class _FakeMainWindow:
    """Minimal stand-in for MainWindow: just enough state
    (`self.root`) for the real, unbound `_open_progress_window` /
    `_button` / `_bind_isolated_wheel` / `_generic_text_wheel` methods to
    run against a real Tk root, without constructing the entire ~14,000
    line app UI."""

    def __init__(self, root):
        self.root = root


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


@pytest.fixture
def fake_main(root):
    fake = _FakeMainWindow(root)
    _bind_real_methods(fake, "_open_progress_window", "_button", "_bind_isolated_wheel", "_generic_text_wheel")
    return fake


class _Outcome:
    """Matches the shape of both BatchTestOutcome and
    FullPipelineBatchOutcome closely enough for show_results()'s getattr-
    based field access."""

    def __init__(self, label, ok, verdict=None, eval_pass_probability=None,
                 net_profit=None, trades=None, report_html=None, reason=None):
        self.label = label
        self.ok = ok
        self.verdict = verdict
        self.eval_pass_probability = eval_pass_probability
        self.net_profit = net_profit
        self.trades = trades
        self.report_html = report_html
        self.reason = reason


def _find_widgets(widget, cls):
    found = []
    for child in widget.winfo_children():
        if isinstance(child, cls):
            found.append(child)
        found.extend(_find_widgets(child, cls))
    return found


def _find_treeview(win):
    trees = _find_widgets(win, ttk.Treeview)
    assert trees, "expected a results ttk.Treeview inside the progress window"
    return trees[0]


def _find_button_by_text(win, text):
    for btn in _find_widgets(win, tk.Button):
        if btn.cget("text") == text:
            return btn
    raise AssertionError(f"no button with text {text!r} found")


def test_open_progress_window_returns_append_and_show_results(fake_main):
    win, append, show_results = fake_main._open_progress_window("Test window")
    try:
        assert callable(append)
        assert callable(show_results)
        # Results table exists but starts unpopulated/hidden.
        tree = _find_treeview(win)
        assert tree.get_children() == ()
    finally:
        win.destroy()


def test_show_results_populates_one_row_per_strategy_with_its_own_report(fake_main, tmp_path):
    win, append, show_results = fake_main._open_progress_window("Full Pipeline: 3 strategy(ies)...")
    try:
        report_a = tmp_path / "report_a.html"
        report_b = tmp_path / "report_b.html"
        report_a.write_text("<html>A</html>")
        report_b.write_text("<html>B</html>")

        outcomes = [
            _Outcome("StratA", ok=True, verdict="READY", eval_pass_probability=61.5,
                     net_profit=1234.5, trades=40, report_html=report_a),
            _Outcome("StratB", ok=True, verdict="MARGINAL", eval_pass_probability=22.0,
                     net_profit=-50.0, trades=12, report_html=report_b),
            _Outcome("StratC", ok=False, reason="zero trades generated"),
        ]
        show_results(outcomes)
        fake_main.root.update()  # flush the root.after(0, ...) callback

        tree = _find_treeview(win)
        rows = tree.get_children()
        # One row per strategy -- this is the actual regression check:
        # every strategy tested gets its own row/report, not one
        # aggregated row for the whole batch.
        assert len(rows) == 3

        labels = [tree.item(r, "values")[0] for r in rows]
        assert set(labels) == {"StratA", "StratB", "StratC"}

        # Successes are ranked ahead of the failure, best eval-pass first.
        assert labels[0] == "StratA"
        assert labels[1] == "StratB"
        assert labels[2] == "StratC"

        failed_row = rows[2]
        assert "FAILED" in tree.item(failed_row, "values")[1]
        assert "zero trades generated" in tree.item(failed_row, "values")[1]
    finally:
        win.destroy()


def test_open_report_button_opens_the_selected_rows_own_report(fake_main, tmp_path, monkeypatch):
    """The actual bug: confirms each row opens ITS OWN report, not
    whichever report happened to be generated last."""
    win, append, show_results = fake_main._open_progress_window("Full Pipeline: 2 strategy(ies)...")
    try:
        report_a = tmp_path / "report_a.html"
        report_b = tmp_path / "report_b.html"
        report_a.write_text("<html>A</html>")
        report_b.write_text("<html>B</html>")

        outcomes = [
            _Outcome("StratA", ok=True, verdict="READY", eval_pass_probability=80.0,
                     net_profit=900.0, trades=30, report_html=report_a),
            _Outcome("StratB", ok=True, verdict="READY", eval_pass_probability=70.0,
                     net_profit=800.0, trades=25, report_html=report_b),
        ]
        show_results(outcomes)
        fake_main.root.update()

        tree = _find_treeview(win)
        rows = tree.get_children()
        row_by_label = {tree.item(r, "values")[0]: r for r in rows}

        opened_urls = []
        monkeypatch.setattr(mw.webbrowser, "open", lambda url: opened_urls.append(url))

        open_report_btn = _find_button_by_text(win, "OPEN REPORT")

        # Select StratB specifically (not the top-ranked row) and confirm
        # its OWN report opens -- not StratA's, and not nothing.
        tree.selection_set(row_by_label["StratB"])
        open_report_btn.invoke()
        assert len(opened_urls) == 1
        assert "report_b.html" in opened_urls[0]

        opened_urls.clear()
        tree.selection_set(row_by_label["StratA"])
        open_report_btn.invoke()
        assert len(opened_urls) == 1
        assert "report_a.html" in opened_urls[0]
    finally:
        win.destroy()


def test_open_all_reports_opens_every_successful_strategys_report(fake_main, tmp_path, monkeypatch):
    win, append, show_results = fake_main._open_progress_window("Full Pipeline: 3 strategy(ies)...")
    try:
        report_a = tmp_path / "report_a.html"
        report_b = tmp_path / "report_b.html"
        report_a.write_text("<html>A</html>")
        report_b.write_text("<html>B</html>")

        outcomes = [
            _Outcome("StratA", ok=True, verdict="READY", eval_pass_probability=80.0,
                     net_profit=900.0, trades=30, report_html=report_a),
            _Outcome("StratB", ok=True, verdict="MARGINAL", eval_pass_probability=40.0,
                     net_profit=100.0, trades=15, report_html=report_b),
            _Outcome("StratC", ok=False, reason="strategy failed to load"),
        ]
        show_results(outcomes)
        fake_main.root.update()

        opened_urls = []
        monkeypatch.setattr(mw.webbrowser, "open", lambda url: opened_urls.append(url))

        open_all_btn = _find_button_by_text(win, "OPEN ALL REPORTS")
        open_all_btn.invoke()

        # Exactly the two successful strategies' reports open -- the
        # failed one (no report) is silently skipped, not an error.
        assert len(opened_urls) == 2
        assert any("report_a.html" in u for u in opened_urls)
        assert any("report_b.html" in u for u in opened_urls)
    finally:
        win.destroy()
