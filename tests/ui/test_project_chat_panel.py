"""Tests for the desktop Project Chat (app.ui.project_chat_panel).

The controller holds all the logic and needs no display. The Tk view is
exercised only when tkinter AND a display (e.g. xvfb) are available."""
from __future__ import annotations

import os
import threading
import time

import pytest

from app.orchestration import projects
from app.ui.project_chat_panel import ProjectChatController


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(projects, "get_app_base_dir", lambda: tmp_path)


class FakeClient:
    def __init__(self, reply="pong", error=None, delay=0.0):
        self.reply, self.error, self.delay, self.calls = reply, error, delay, []

    def chat(self, name, message, history=None, activity_lines=None):
        self.calls.append((name, message, list(history or [])))
        time.sleep(self.delay)
        return ("" if self.error else self.reply), self.error


def _send(ctl, text):
    done = threading.Event(); box = {}
    def cb(reply, error): box.update(reply=reply, error=error); done.set()
    started = ctl.send(text, cb)
    if started:
        assert done.wait(5), "reply never arrived"
    return started, box


def test_send_persists_both_turns_and_passes_prior_history():
    client = FakeClient()
    ctl = ProjectChatController(lambda: client)
    ctl.create("Desk")
    assert _send(ctl, "one")[1]["reply"] == "pong"
    _send(ctl, "two")
    assert [t["content"] for t in ctl.history()] == ["one", "pong", "two", "pong"]
    assert client.calls[0] == ("Desk", "one", [])
    assert [t["content"] for t in client.calls[1][2]] == ["one", "pong"]


def test_error_keeps_the_user_turn_and_saves_no_reply():
    ctl = ProjectChatController(lambda: FakeClient(error="Ollama down"))
    ctl.create("Desk")
    started, box = _send(ctl, "hello")
    assert started and box["error"] == "Ollama down"
    assert [t["role"] for t in ctl.history()] == ["user"]
    assert ctl.busy is False


def test_empty_message_and_no_project_do_nothing():
    ctl = ProjectChatController(lambda: FakeClient())
    assert ctl.send("hi", lambda *a: None) is False        # no active project
    ctl.create("P")
    assert ctl.send("   ", lambda *a: None) is False


def test_second_send_while_busy_is_refused_and_busy_clears():
    ctl = ProjectChatController(lambda: FakeClient(delay=0.3))
    ctl.create("P")
    done = threading.Event()
    assert ctl.send("first", lambda *a: done.set()) is True
    assert ctl.busy is True and ctl.send("second", lambda *a: None) is False
    assert done.wait(5)
    time.sleep(0.05)
    assert ctl.busy is False
    assert len(ctl.history()) == 2                            # the refused message was never stored


def test_project_deleted_mid_flight_reports_an_error_instead_of_crashing():
    ctl = ProjectChatController(lambda: FakeClient())
    project = ctl.create("Gone")
    projects.delete_project(project["id"])                    # e.g. deleted from the web app
    started, box = _send(ctl, "hi")
    assert started and box["error"] and ctl.busy is False


def test_client_exception_is_reported_not_raised():
    class Boom:
        def chat(self, *a, **k): raise RuntimeError("kaboom")
    ctl = ProjectChatController(lambda: Boom())
    ctl.create("P")
    _, box = _send(ctl, "hi")
    assert "kaboom" in box["error"] and ctl.busy is False


def test_project_management_and_web_parity():
    ctl = ProjectChatController(lambda: FakeClient())
    a = ctl.create("Alpha")
    projects.create_project("Made on the web")                # shared storage: visible to the desktop
    assert {p["name"] for p in ctl.list_projects()} == {"Alpha", "Made on the web"}
    assert ctl.rename("Alpha 2")["name"] == "Alpha 2"
    ctl.delete()
    assert ctl.active_id is None and all(p["id"] != a["id"] for p in ctl.list_projects())
    with pytest.raises(projects.ProjectNotFound):
        ctl.history()
    with pytest.raises(projects.ProjectNotFound):
        ctl.select("nope")


try:
    import tkinter
    _HAS_TK = True
except ImportError:
    _HAS_TK = False


@pytest.mark.skipif(not _HAS_TK or (not os.environ.get("DISPLAY") and os.name != "nt"), reason="needs tkinter and a display")
def test_panel_round_trip_under_tk():
    from app.ui.project_chat_panel import ProjectChatPanel
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("no usable display")
    try:
        ctl = ProjectChatController(lambda: FakeClient(reply="echo"))
        panel = ProjectChatPanel(root, ctl)
        panel.open(); root.update()
        assert panel.empty.winfo_ismapped()
        p = ctl.create("Tk"); panel._reload_projects(select_id=p["id"]); root.update()
        panel.entry.insert(0, "hello"); panel._send()
        end = time.time() + 5
        while time.time() < end and "echo" not in panel.log.get("1.0", "end"):
            root.update(); time.sleep(0.02)
        text = panel.log.get("1.0", "end")
        assert "You: hello" in text and "echo" in text
        projects.create_project("Tk"); panel._reload_projects()
        assert len(panel._names) == 2                         # duplicate names stay selectable
    finally:
        root.destroy()
