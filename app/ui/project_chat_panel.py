"""
Desktop (Tk) Project Chat: the same floating, always-present chat the web
app has, sharing its data.

Projects and chat history live in app.orchestration.projects (one JSON file
per project under data/projects/), so a project created on the web shows up
here and vice-versa. The AI side is app.ai.project_chat.ProjectChatClient,
using the Ollama host/model already configured for AI Assist.

Scope, stated plainly: this is the CHAT + PROJECTS half. The web-only pieces
(Activity feed, live phase banner, equity swarm, Universe Map) read the web
server's in-memory JobManager, which the desktop tools don't use, so they
have no data to show here and are not faked.

Split in two so the logic is testable without a display:
  ProjectChatController -- no tkinter; projects + sending + threading.
  ProjectChatPanel      -- thin Tk view (floating button + panel).
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

from app.ai.ollama_settings import load_settings as load_ollama_settings
from app.ai.project_chat import ProjectChatClient
from app.orchestration import projects

# Palette copied (not imported) from app.ui.condition_builder / main_window to
# avoid a circular import -- same convention condition_builder.py documents.
_BG, _PANEL, _PANEL_3 = "#0D1017", "#171B25", "#1E232E"
_BORDER, _TEXT, _DIM = "#3D4453", "#E9EBEF", "#8A93A6"
_TEAL, _RED = "#35E0B0", "#F0596A"
_FONT = "Segoe UI"


class ProjectChatController:
    """All behaviour, no widgets. ``send`` runs Ollama on a daemon thread and
    reports through ``on_done(reply, error)`` -- the view is responsible for
    hopping back onto the Tk thread (see ProjectChatPanel._post)."""

    def __init__(self, client_factory: Optional[Callable[[], ProjectChatClient]] = None) -> None:
        self._client_factory = client_factory or (lambda: ProjectChatClient(load_ollama_settings()))
        self.active_id: Optional[str] = None
        self._busy = False
        self._lock = threading.Lock()

    # -- projects ---------------------------------------------------------
    def list_projects(self) -> list[dict]:
        return projects.list_projects()

    def create(self, name: str) -> dict:
        project = projects.create_project(name)
        self.active_id = project["id"]
        return project

    def select(self, project_id: str) -> dict:
        project = projects.get_project(project_id)  # raises ProjectNotFound
        self.active_id = project_id
        return project

    def rename(self, name: str) -> dict:
        self._require_active()
        return projects.rename_project(self.active_id, name)

    def delete(self) -> None:
        self._require_active()
        projects.delete_project(self.active_id)
        self.active_id = None

    def history(self) -> list[dict]:
        return projects.get_project(self._require_active())["chat_history"]

    # -- chat -------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._busy

    def send(self, message: str, on_done: Callable[[str, Optional[str]], None]) -> bool:
        """Returns False (and does nothing) for an empty message, no active
        project, or a reply already in flight. Otherwise persists the user's
        turn, calls Ollama off-thread, persists the reply, and calls
        ``on_done(reply, error)`` exactly once."""
        message = (message or "").strip()
        if not message or self.active_id is None:
            return False
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        project_id = self.active_id
        try:
            project = projects.get_project(project_id)
            projects.append_chat_message(project_id, "user", message)
        except Exception as exc:  # noqa: BLE001 -- e.g. project deleted from the web meanwhile
            self._busy = False
            on_done("", f"{type(exc).__name__}: {exc}")
            return True
        prior = project.get("chat_history", [])

        def worker() -> None:
            reply, error = "", None
            try:
                reply, error = self._client_factory().chat(project["name"], message, history=prior)
                if not error:
                    projects.append_chat_message(project_id, "assistant", reply)
            except Exception as exc:  # noqa: BLE001
                reply, error = "", f"{type(exc).__name__}: {exc}"
            finally:
                self._busy = False
            on_done(reply, error)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _require_active(self) -> str:
        if self.active_id is None:
            raise projects.ProjectNotFound("no active project")
        return self.active_id


class ProjectChatPanel:
    """Floating button in the window's bottom-right; click to open/close the
    chat panel. Never raises out of __init__ paths the main window depends on
    (the caller wraps construction in try/except as well)."""

    def __init__(self, root, controller: Optional[ProjectChatController] = None) -> None:
        from tkinter import Button, Frame, Label, Text, Entry, StringVar, ttk

        self.root = root
        self.ctl = controller or ProjectChatController()
        self._open = False
        self._names: dict[str, str] = {}  # combobox label -> project id
        # Worker threads never touch Tk: they enqueue callbacks here and the
        # main thread drains the queue (see _pump). Tk is not thread-safe.
        self._queue: "queue.Queue[Callable[[], None]]" = queue.Queue()

        self.button = Button(
            root, text="\U0001F4AC", bg=_TEAL, fg="#04120E", activebackground=_TEAL, relief="flat", bd=0,
            font=(_FONT, 15), cursor="hand2", command=self.toggle,
        )
        self.button.place(relx=1.0, rely=1.0, anchor="se", x=-18, y=-40, width=46, height=46)

        self.panel = Frame(root, bg=_PANEL, highlightthickness=1, highlightbackground=_BORDER)
        top = Frame(self.panel, bg=_PANEL_3)
        top.pack(fill="x")
        self.combo_var = StringVar()
        self.combo = ttk.Combobox(top, textvariable=self.combo_var, state="readonly")
        self.combo.pack(side="left", fill="x", expand=True, padx=6, pady=6)
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self._on_select())
        for text, cmd in (("+", self._new), ("\u270E", self._rename), ("\U0001F5D1", self._delete), ("\u00D7", self.close)):
            Button(top, text=text, command=cmd, bg=_PANEL, fg=_TEXT, relief="flat", bd=0, width=3,
                   font=(_FONT, 10), cursor="hand2").pack(side="left", padx=(0, 4), pady=6)

        self.empty = Label(self.panel, text="No project yet. Click + to start one.", bg=_PANEL, fg=_DIM,
                           font=(_FONT, 10), wraplength=300)
        self.log = Text(self.panel, bg=_BG, fg=_TEXT, wrap="word", state="disabled", relief="flat",
                        font=(_FONT, 10), padx=8, pady=8, height=16)
        self.log.tag_configure("user", foreground=_TEAL)
        self.log.tag_configure("assistant", foreground=_TEXT)
        self.log.tag_configure("error", foreground=_RED)
        self.status = Label(self.panel, text="", bg=_PANEL, fg=_DIM, font=(_FONT, 9), anchor="w")
        self.entry = Entry(self.panel, bg=_PANEL_3, fg=_TEXT, insertbackground=_TEXT, relief="flat", font=(_FONT, 10))
        self.entry.bind("<Return>", lambda _e: self._send())
        self.send_btn = Button(self.panel, text="Send", command=self._send, bg=_TEAL, fg="#04120E",
                               relief="flat", bd=0, font=(_FONT, 10, "bold"), cursor="hand2")

        self.status.pack(side="bottom", fill="x", padx=8)
        row = Frame(self.panel, bg=_PANEL)
        row.pack(side="bottom", fill="x", padx=6, pady=6)
        self.entry.pack(in_=row, side="left", fill="x", expand=True, ipady=5)
        self.send_btn.pack(in_=row, side="left", padx=(6, 0))
        self.log.pack(fill="both", expand=True)
        self._keep_on_top()
        self._pump()

    # -- window plumbing --------------------------------------------------
    def toggle(self) -> None:
        self.close() if self._open else self.open()

    def open(self) -> None:
        self._open = True
        self.panel.place(relx=1.0, rely=1.0, anchor="se", x=-18, y=-94, width=380, height=460)
        self._reload_projects()
        self.panel.lift(); self.button.lift()

    def close(self) -> None:
        self._open = False
        self.panel.place_forget()

    def _keep_on_top(self) -> None:
        """Pages are placed over the content area as they're shown; re-lifting
        keeps the floating button above them. Cheap, and stops with the window."""
        try:
            self.button.lift()
            if self._open:
                self.panel.lift()
            self.root.after(1500, self._keep_on_top)
        except Exception:  # noqa: BLE001 -- window closed
            pass

    def _post(self, fn: Callable[[], None]) -> None:
        """Queue ``fn`` to run on the Tk thread. Safe to call from any thread."""
        self._queue.put(fn)

    def _pump(self) -> None:
        """Main-thread loop: run whatever worker threads queued, then re-arm."""
        try:
            while True:
                try:
                    fn = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    fn()
                except Exception:  # noqa: BLE001 -- a widget destroyed mid-reply must not kill the pump
                    pass
            self.root.after(100, self._pump)
        except Exception:  # noqa: BLE001 -- window closed
            pass

    # -- data <-> widgets -------------------------------------------------
    def _reload_projects(self, select_id: Optional[str] = None) -> None:
        items = self.ctl.list_projects()
        self._names = {}
        labels = []
        for p in items:
            label = p["name"]
            if label in self._names:  # two projects with one name must stay selectable
                label = f"{label} ({p['id'][:4]})"
            self._names[label] = p["id"]
            labels.append(label)
        self.combo["values"] = labels
        target = select_id or self.ctl.active_id
        if target not in self._names.values():
            target = None
        if target is None and items:
            target = items[0]["id"]
        if target is None:
            self.ctl.active_id = None
            self.combo_var.set("")
            self._render_history([])
            self.empty.pack(before=self.log)
            return
        self.empty.pack_forget()
        label = next(l for l, i in self._names.items() if i == target)
        self.combo_var.set(label)
        try:
            self.ctl.select(target)
        except projects.ProjectNotFound:
            self._reload_projects()
            return
        self._render_history(self.ctl.history())

    def _render_history(self, history: list[dict]) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        for turn in history:
            self._append(turn.get("role", "assistant"), turn.get("content", ""), _raw=True)
        self.log.configure(state="disabled")
        self.log.see("end")

    def _append(self, role: str, text: str, _raw: bool = False) -> None:
        if not _raw:
            self.log.configure(state="normal")
        prefix = "You: " if role == "user" else ("" if role == "assistant" else "! ")
        self.log.insert("end", prefix + text + "\n\n", role)
        if not _raw:
            self.log.configure(state="disabled")
            self.log.see("end")

    def _on_select(self) -> None:
        pid = self._names.get(self.combo_var.get())
        if pid:
            self._reload_projects(select_id=pid)

    def _new(self) -> None:
        from tkinter import simpledialog
        name = simpledialog.askstring("New project", "Project name:", parent=self.root)
        if name is None:
            return
        project = self.ctl.create(name)
        self._reload_projects(select_id=project["id"])

    def _rename(self) -> None:
        if self.ctl.active_id is None:
            return
        from tkinter import simpledialog
        name = simpledialog.askstring("Rename project", "New name:", parent=self.root)
        if name and name.strip():
            self.ctl.rename(name)
            self._reload_projects()

    def _delete(self) -> None:
        if self.ctl.active_id is None:
            return
        from tkinter import messagebox
        if messagebox.askyesno("Delete project", "Delete this project and its chat history?", parent=self.root):
            self.ctl.delete()
            self._reload_projects()

    def _send(self) -> None:
        text = self.entry.get().strip()
        if not text or self.ctl.active_id is None or self.ctl.busy:
            return
        self.entry.delete(0, "end")
        self._append("user", text)
        self.status.configure(text="Thinking\u2026")
        self.send_btn.configure(state="disabled")

        def done(reply: str, error: Optional[str]) -> None:
            def apply() -> None:
                self.send_btn.configure(state="normal")
                self.status.configure(text="")
                self._append("error" if error else "assistant", error or reply)
            self._post(apply)

        if not self.ctl.send(text, done):
            self.send_btn.configure(state="normal")
            self.status.configure(text="")
