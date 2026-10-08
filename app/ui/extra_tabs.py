"""
Desktop tabs that bring the web app's remaining pages to the desktop build:

  * Start Here pages (one per sidebar section)      -> build_start_here_tabs
  * Strategy Library (CREATE section)               -> build_strategy_library_tab
  * Interactive Replay (DEPLOYMENT section)         -> build_replay_tab

They live in their own module so main_window.py (21k lines) only needs small
hooks. Colors are always read as `mw.<NAME>` at call time (never imported by
value) so a Dark/Light theme toggle -- which rebuilds every tab -- picks up the
new palette.
"""
from __future__ import annotations

import json
import threading
import time
from tkinter import (
    BooleanVar, Button, Canvas, END, Entry, Frame, Label, StringVar, Text, filedialog, messagebox, ttk,
)

# ======================================================================
# Start Here pages
# ======================================================================
def _section_accent(mw, section: str) -> str:
    return {
        "create": mw.NEON_VIOLET, "test": mw.NEON_CYAN, "champion": mw.NEON_MAGENTA,
        "deployment": mw.NEON_LIME, "graveyard": mw.METAL_BRIGHT, "quantlab": mw.NEON_CYAN,
        "account": mw.METAL_BRIGHT,
    }.get(section, mw.GREEN)


def build_start_here_tabs(win) -> None:
    """One 'Start Here' page per section, rendered from the SAME shared
    content the web app's /start-here/<section> pages use."""
    from app.orchestration.section_guides import DESKTOP_NAV_FOR_HREF, SECTION_GUIDES

    mw = _mw()
    for section, frame in win._starthere_frames.items():
        guide = SECTION_GUIDES.get(section)
        if not guide:
            continue
        accent = _section_accent(mw, section)
        f = win._scrollable(frame)
        win._page_header(f, "", f"{guide['title']} \u2014 Start Here", guide["tagline"])

        card = win._section(f, "What this section is for")
        Label(
            card, text=guide["description"], bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(10),
            wraplength=900, justify="left", anchor="w",
        ).pack(anchor="w", padx=14, pady=(0, 8))
        for n, step in enumerate(guide.get("roadmap", []), 1):
            row = Frame(card, bg=mw.PANEL)
            row.pack(fill="x", padx=14, pady=2)
            Label(row, text=f"{n}.", bg=mw.PANEL, fg=accent, font=mw._safe_font(10, "bold"), width=3,
                  anchor="ne").pack(side="left", anchor="n")
            Label(row, text=step, bg=mw.PANEL, fg=mw.TEXT, font=mw._safe_font(10), wraplength=860,
                  justify="left", anchor="w").pack(side="left", fill="x", expand=True)
        buttons = guide.get("primary_buttons", [])
        if buttons:
            brow = Frame(card, bg=mw.PANEL)
            brow.pack(anchor="w", padx=14, pady=(12, 12))
            for b in buttons:
                key = DESKTOP_NAV_FOR_HREF.get(b["href"])
                if not key:
                    continue
                btn = win._button(brow, b["label"], lambda k=key: win._show_page(k), primary=True)
                hover = mw._blend_hex(accent, "#FFFFFF", 0.18)
                btn.configure(bg=accent, activebackground=hover, fg=mw.ACCENT_INK, activeforeground=mw.ACCENT_INK)
                btn.bind("<Enter>", lambda _e, bt=btn, h=hover: bt.configure(bg=h))
                btn.bind("<Leave>", lambda _e, bt=btn, a=accent: bt.configure(bg=a))
                btn.pack(side="left", padx=(0, 8))

        grid_card = win._section(f, f"Everything in {guide['title']}")
        cards = [
            (t["name"], t["desc"], DESKTOP_NAV_FOR_HREF[t["href"]])
            for t in guide.get("tools", []) if t["href"] in DESKTOP_NAV_FOR_HREF
        ]
        win._link_card_grid(grid_card, cards)


# ======================================================================
# Strategy Library
# ======================================================================
def build_strategy_library_tab(win) -> None:
    mw = _mw()
    f = win._scrollable(win.tab_stratlibrary)
    win._page_header(
        f, "", "Strategy Library",
        "Every saved strategy in one place -- search, see how far each one has got through the pipeline, "
        "edit its code, update its details, or send it straight into Full Pipeline / Quick Optimize.",
    )
    win._lib2 = {"items": [], "selected": None, "after": None}

    # ---------------- filters
    fcard = win._section(f, "Find a strategy")
    frow = Frame(fcard, bg=mw.PANEL)
    frow.pack(fill="x", padx=14, pady=(0, 12))
    win._lib2_search = StringVar()
    win._lib2_type = StringVar(value="All types")
    win._lib2_status = StringVar(value="All statuses")
    win._lib2_market = StringVar(value="All markets")

    def _label(text):
        return Label(frow, text=text, bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9))

    _label("Search").pack(side="left")
    entry = Entry(
        frow, textvariable=win._lib2_search, width=28, bg=mw.PANEL_3, fg=mw.TEXT, insertbackground=mw.TEXT,
        relief="flat", highlightthickness=1, highlightbackground=mw.BORDER_LIGHT, highlightcolor=mw.ACCENT,
        font=mw._safe_font(10),
    )
    entry.pack(side="left", padx=(6, 14), ipady=4)
    entry.bind("<KeyRelease>", lambda _e: _lib_debounce(win))
    combos = {}
    for label, var, values, width in (
        ("Type", win._lib2_type, ["All types", *mw.STRATEGY_TYPES], 12),
        ("Status", win._lib2_status, ["All statuses", *mw.STATUS_LABELS_ORDERED], 20),
        ("Market", win._lib2_market, ["All markets"], 14),
    ):
        _label(label).pack(side="left")
        cb = ttk.Combobox(frow, textvariable=var, values=values, state="readonly", width=width,
                          style="T58.TCombobox", font=mw._safe_font(9))
        cb.pack(side="left", padx=(6, 14))
        cb.bind("<<ComboboxSelected>>", lambda _e: _lib_refresh(win))
        combos[label] = cb
    win._lib2_market_combo = combos["Market"]
    win._button(frow, "REFRESH", lambda: _lib_refresh(win)).pack(side="right")

    # ---------------- list
    lcard = win._section(f, "Strategies")
    win._lib2_count = Label(lcard, text="", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9))
    win._lib2_count.pack(anchor="w", padx=14, pady=(0, 6))
    tree_wrap = Frame(lcard, bg=mw.PANEL)
    tree_wrap.pack(fill="x", padx=14, pady=(0, 12))
    cols = ("name", "type", "status", "stage", "progress", "market", "modified")
    win._lib2_tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", style="T58.Treeview",
                                  height=9, selectmode="browse")
    for col, text, width in (
        ("name", "Strategy", 280), ("type", "Type", 80), ("status", "Status", 150), ("stage", "Stage", 110),
        ("progress", "Progress", 80), ("market", "Market", 100), ("modified", "Modified", 120),
    ):
        win._lib2_tree.heading(col, text=text, anchor="w")
        win._lib2_tree.column(col, width=width, anchor="w")
    sb = ttk.Scrollbar(tree_wrap, orient="vertical", command=win._lib2_tree.yview, style="T58.Vertical.TScrollbar")
    win._lib2_tree.configure(yscrollcommand=sb.set)
    win._lib2_tree.pack(side="left", fill="x", expand=True)
    sb.pack(side="right", fill="y")
    win._bind_isolated_wheel(win._lib2_tree)
    win._lib2_tree.bind("<<TreeviewSelect>>", lambda _e: _lib_on_select(win))

    # ---------------- detail
    dcard = win._section(f, "Selected strategy", emphasize=True)
    win._lib2_title = Label(dcard, text="Select a strategy above.", bg=mw.PANEL, fg=mw.TEXT,
                            font=mw._safe_font(13, "bold"), anchor="w")
    win._lib2_title.pack(anchor="w", padx=14, pady=(0, 2))
    win._lib2_meta = Label(dcard, text="", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9), anchor="w")
    win._lib2_meta.pack(anchor="w", padx=14)
    win._lib2_stage_row = Frame(dcard, bg=mw.PANEL)
    win._lib2_stage_row.pack(anchor="w", padx=14, pady=(10, 2))
    win._lib2_next = Label(dcard, text="", bg=mw.PANEL, fg=mw.GREEN, font=mw._safe_font(9), anchor="w")
    win._lib2_next.pack(anchor="w", padx=14, pady=(0, 8))

    arow = Frame(dcard, bg=mw.PANEL)
    arow.pack(anchor="w", padx=14, pady=(0, 10))
    win._lib2_actions = [
        win._button(arow, "OPEN IN FULL PIPELINE \u2192", lambda: _lib_send(win, "fullpipeline"), primary=True),
        win._button(arow, "OPEN IN QUICK OPTIMIZE \u2192", lambda: _lib_send(win, "quickoptimize")),
        win._button(arow, "USE AS ACTIVE STRATEGY", lambda: _lib_send(win, None)),
        win._button(arow, "SET AS CURRENT (VALIDATE)", lambda: _lib_set_current(win)),
    ]
    for b in win._lib2_actions:
        b.pack(side="left", padx=(0, 8))

    # Code
    Label(dcard, text="CODE", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9, "bold")).pack(anchor="w", padx=14)
    code_wrap = Frame(dcard, bg=mw.PANEL)
    code_wrap.pack(fill="x", padx=14, pady=(4, 6))
    win._lib2_code = Text(
        code_wrap, height=12, wrap="none", bg=mw.LOG_BG, fg=mw.TEXT, insertbackground=mw.TEXT, relief="flat",
        highlightthickness=1, highlightbackground=mw.BORDER, font=(mw.MONO, 9), undo=True,
    )
    csb = ttk.Scrollbar(code_wrap, orient="vertical", command=win._lib2_code.yview, style="T58.Vertical.TScrollbar")
    win._lib2_code.configure(yscrollcommand=csb.set)
    win._lib2_code.pack(side="left", fill="x", expand=True)
    csb.pack(side="right", fill="y")
    win._bind_isolated_wheel(win._lib2_code)
    crow = Frame(dcard, bg=mw.PANEL)
    crow.pack(anchor="w", padx=14, pady=(0, 10))
    win._button(crow, "SAVE CHANGES", lambda: _lib_save_code(win)).pack(side="left")
    win._lib2_code_note = Label(crow, text="", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9))
    win._lib2_code_note.pack(side="left", padx=10)

    # Results
    Label(dcard, text="RESULTS", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9, "bold")).pack(anchor="w", padx=14)
    win._lib2_results = Label(dcard, text="", bg=mw.PANEL, fg=mw.TEXT, font=mw._safe_font(9), anchor="w",
                              justify="left", wraplength=880)
    win._lib2_results.pack(anchor="w", padx=14, pady=(4, 10))

    # Settings
    Label(dcard, text="SETTINGS", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9, "bold")).pack(anchor="w", padx=14)
    win._lib2_desc = StringVar()
    win._lib2_mkt = StringVar()
    win._lib2_tags = StringVar()
    win._lib2_set_status = StringVar()
    win._lib2_rename = StringVar()
    grid = Frame(dcard, bg=mw.PANEL)
    grid.pack(fill="x", padx=14, pady=(4, 4))
    grid.columnconfigure(1, weight=1)
    for r, (label, var) in enumerate((("Description", win._lib2_desc), ("Market", win._lib2_mkt),
                                       ("Tags (comma-separated)", win._lib2_tags))):
        Label(grid, text=label, bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9), width=22,
              anchor="w").grid(row=r, column=0, sticky="w", pady=3)
        Entry(grid, textvariable=var, bg=mw.PANEL_3, fg=mw.TEXT, insertbackground=mw.TEXT, relief="flat",
              highlightthickness=1, highlightbackground=mw.BORDER_LIGHT, highlightcolor=mw.ACCENT,
              font=mw._safe_font(10)).grid(row=r, column=1, sticky="ew", pady=3, ipady=4)
    Label(grid, text="Status", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9), width=22,
          anchor="w").grid(row=3, column=0, sticky="w", pady=3)
    ttk.Combobox(grid, textvariable=win._lib2_set_status, values=mw.STATUS_LABELS_ORDERED, state="readonly",
                 width=26, style="T58.TCombobox", font=mw._safe_font(9)).grid(row=3, column=1, sticky="w", pady=3)
    Label(grid, text="Rename file to", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9), width=22,
          anchor="w").grid(row=4, column=0, sticky="w", pady=3)
    Entry(grid, textvariable=win._lib2_rename, bg=mw.PANEL_3, fg=mw.TEXT, insertbackground=mw.TEXT, relief="flat",
          highlightthickness=1, highlightbackground=mw.BORDER_LIGHT, highlightcolor=mw.ACCENT,
          font=mw._safe_font(10)).grid(row=4, column=1, sticky="ew", pady=3, ipady=4)
    brow = Frame(dcard, bg=mw.PANEL)
    brow.pack(anchor="w", padx=14, pady=(6, 14))
    win._button(brow, "SAVE INFO", lambda: _lib_save_info(win), primary=True).pack(side="left", padx=(0, 8))
    win._button(brow, "RENAME", lambda: _lib_rename(win)).pack(side="left", padx=(0, 8))
    delete = win._button(brow, "DELETE STRATEGY", lambda: _lib_delete(win))
    delete.configure(fg=mw.RED)
    delete.pack(side="left")
    win._lib2_status_note = Label(dcard, text="", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9))
    win._lib2_status_note.pack(anchor="w", padx=14, pady=(0, 12))

    _lib_refresh(win)


def _lib_debounce(win) -> None:
    if win._lib2.get("after"):
        try:
            win.root.after_cancel(win._lib2["after"])
        except Exception:
            pass
    win._lib2["after"] = win.root.after(250, lambda: _lib_refresh(win))


def _lib_refresh(win) -> None:
    """Re-reads the library (cheap: small .meta.json sidecars) and repaints
    the list, keeping the current selection when it still exists."""
    mw = _mw()
    if not hasattr(win, "_lib2_tree"):
        return
    try:
        all_items = mw.list_saved_strategies()
    except Exception as exc:  # noqa: BLE001
        win._lib2_count.config(text=f"Could not read the library: {exc}", fg=mw.RED)
        return
    markets = sorted({i.metadata.get("market", "") for i in all_items if i.metadata.get("market")})
    win._lib2_market_combo.configure(values=["All markets", *markets])
    if win._lib2_market.get() not in ("All markets", *markets):
        win._lib2_market.set("All markets")

    q = win._lib2_search.get().strip().lower()
    t = win._lib2_type.get()
    s = win._lib2_status.get()
    m = win._lib2_market.get()
    items = []
    for i in all_items:
        if t != "All types" and i.strategy_type != t:
            continue
        if s != "All statuses" and i.status_display != s:
            continue
        if m != "All markets" and i.metadata.get("market", "") != m:
            continue
        if q and q not in " ".join([
            i.name, str(i.metadata.get("description", "")), str(i.metadata.get("market", "")), " ".join(i.tags),
        ]).lower():
            continue
        items.append(i)

    prev = win._lib2.get("selected")
    win._lib2["items"] = {f"{i.strategy_type}::{i.name}": i for i in items}
    tree = win._lib2_tree
    tree.delete(*tree.get_children())
    for iid, i in win._lib2["items"].items():
        pp = i.pipeline_progress
        stage = next((st["title"] for st in pp["stages"] if st["key"] == pp["current_stage"]), "Create")
        tree.insert("", "end", iid=iid, values=(
            i.name, i.strategy_type, i.status_display, stage, f"{pp['progress_pct']:.0f}%",
            i.metadata.get("market", ""), time.strftime("%Y-%m-%d %H:%M", time.localtime(i.modified)),
        ))
    win._lib2_count.config(
        text=f"{len(items)} of {len(all_items)} strategies" if len(items) != len(all_items) else f"{len(items)} strategies",
        fg=mw.TEXT_MUTED,
    )
    if prev in win._lib2["items"]:
        tree.selection_set(prev)
    else:
        win._lib2["selected"] = None
        _lib_show(win, None)


def _lib_on_select(win) -> None:
    sel = win._lib2_tree.selection()
    key = sel[0] if sel else None
    win._lib2["selected"] = key
    _lib_show(win, win._lib2["items"].get(key) if key else None)


def _lib_show(win, item) -> None:
    mw = _mw()
    for child in win._lib2_stage_row.winfo_children():
        child.destroy()
    win._lib2_code.delete("1.0", END)
    win._lib2_code_note.config(text="")
    win._lib2_status_note.config(text="")
    if item is None:
        win._lib2_title.config(text="Select a strategy above.")
        for v in (win._lib2_meta, win._lib2_next, win._lib2_results):
            v.config(text="")
        for v in (win._lib2_desc, win._lib2_mkt, win._lib2_tags, win._lib2_set_status, win._lib2_rename):
            v.set("")
        return
    win._lib2_title.config(text=item.name)
    win._lib2_meta.config(
        text=f"{item.strategy_type}  \u00b7  {item.status_display}  \u00b7  {item.size_bytes / 1024:.1f} KB  \u00b7  "
             f"modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(item.modified))}")
    pp = item.pipeline_progress
    for n, st in enumerate(pp["stages"]):
        done = st["done"]
        Label(
            win._lib2_stage_row, text=("\u2713 " if done else "\u25cb ") + st["title"],
            bg=mw._blend_hex(mw.PANEL, mw.GREEN, 0.14) if done else mw.PANEL_3,
            fg=mw.GREEN if done else mw.TEXT_DIM, font=mw._safe_font(9, "bold"), padx=9, pady=4,
        ).pack(side="left", padx=(0, 4))
    nxt = next((st["title"] for st in pp["stages"] if st["key"] == pp["next_stage"]), None) if pp["next_stage"] else None
    note = f"{pp['progress_pct']:.0f}% through the pipeline"
    if nxt:
        note += f"  \u00b7  next: {nxt}"
    if pp.get("verdict"):
        note += f"  \u00b7  verdict: {pp['verdict']}"
    win._lib2_next.config(text=note)
    try:
        win._lib2_code.insert("1.0", mw.load_strategy_text(item.strategy_type, item.name))
        win._lib2_code.edit_reset()
    except Exception as exc:  # noqa: BLE001
        win._lib2_code.insert("1.0", f"Could not load code: {exc}")

    lines = []
    for key, label in (("last_run", "Last backtest"), ("lookahead", "Lookahead check"), ("last_search", "Last search"),
                       ("last_optimize", "Last optimize"), ("last_validation", "Last validation")):
        val = item.metadata.get(key)
        if val:
            lines.append(f"{label}: " + "  \u2022  ".join(f"{k}: {v}" for k, v in val.items()))
    win._lib2_results.config(text="\n".join(lines) if lines else "No backtest, search, or validation recorded yet.")
    win._lib2_desc.set(item.metadata.get("description", ""))
    win._lib2_mkt.set(item.metadata.get("market", ""))
    win._lib2_tags.set(", ".join(item.tags))
    win._lib2_set_status.set(item.status_display)
    win._lib2_rename.set(item.name)


def _lib_selected(win):
    key = win._lib2.get("selected")
    return win._lib2["items"].get(key) if key else None


def _lib_send(win, page) -> None:
    item = _lib_selected(win)
    if item is None:
        messagebox.showinfo("No selection", "Select a strategy from the list first.")
        return
    try:
        win._load_library_item_into_active_slot(item)
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("Could not load strategy", str(exc))
        return
    if page:
        win._show_page(page)
    else:
        win._lib2_status_note.config(text=f"'{item.name}' is now the active strategy for Run & Report, Full Pipeline and the other tabs.",
                                     fg=_mw().GREEN)


def _lib_set_current(win) -> None:
    mw = _mw()
    item = _lib_selected(win)
    if item is None:
        messagebox.showinfo("No selection", "Select a strategy from the list first.")
        return
    try:
        from app.reports import strategy_state
        stem = item.name.rsplit(".", 1)[0]
        strategy_state.set_current_strategy(
            stem, item.metadata.get("market", "") or "", item.metadata.get("timeframe", "") or "",
            library_type=item.strategy_type, library_filename=item.name,
        )
        win._lib2_status_note.config(text=f"'{item.name}' is now the current strategy -- see Validate \u2192 Start Here.", fg=mw.GREEN)
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("Could not set current strategy", str(exc))


def _lib_save_code(win) -> None:
    mw = _mw()
    item = _lib_selected(win)
    if item is None:
        return
    try:
        mw.save_strategy_text(win._lib2_code.get("1.0", "end-1c"), item.name, item.strategy_type, overwrite=True)
        win._lib2_code_note.config(text="Saved.", fg=mw.GREEN)
    except Exception as exc:  # noqa: BLE001
        win._lib2_code_note.config(text=f"Could not save: {exc}", fg=mw.RED)


def _lib_save_info(win) -> None:
    mw = _mw()
    item = _lib_selected(win)
    if item is None:
        return
    try:
        mw.save_strategy_metadata(item.strategy_type, item.name, {
            "description": win._lib2_desc.get().strip(), "market": win._lib2_mkt.get().strip(),
        })
        raw = win._lib2_tags.get().strip()
        mw.set_strategy_tags(item.strategy_type, item.name, [t.strip() for t in raw.split(",") if t.strip()])
        label = win._lib2_set_status.get()
        mw.set_strategy_status(item.strategy_type, item.name, mw.STATUS_LABEL_TO_KEY.get(label, label))
        win._lib2_status_note.config(text="Saved.", fg=mw.GREEN)
        _lib_refresh(win)
        win._refresh_strategy_library()
    except Exception as exc:  # noqa: BLE001
        win._lib2_status_note.config(text=f"Could not save: {exc}", fg=mw.RED)


def _lib_rename(win) -> None:
    mw = _mw()
    item = _lib_selected(win)
    new = win._lib2_rename.get().strip()
    if item is None or not new or new == item.name:
        return
    try:
        path = mw.rename_saved_strategy(item.strategy_type, item.name, new)
        win._lib2["selected"] = f"{item.strategy_type}::{path.name}"
        win._lib2_status_note.config(text=f"Renamed to {path.name}.", fg=mw.GREEN)
        _lib_refresh(win)
        win._refresh_strategy_library()
    except Exception as exc:  # noqa: BLE001
        win._lib2_status_note.config(text=f"Could not rename: {exc}", fg=mw.RED)


def _lib_delete(win) -> None:
    mw = _mw()
    item = _lib_selected(win)
    if item is None:
        return
    if not messagebox.askyesno("Delete strategy", f"Permanently delete '{item.name}' from the {item.strategy_type} library?"):
        return
    try:
        mw.delete_saved_strategy(item.strategy_type, item.name)
        win._lib2["selected"] = None
        _lib_refresh(win)
        win._refresh_strategy_library()
    except Exception as exc:  # noqa: BLE001
        win._lib2_status_note.config(text=f"Could not delete: {exc}", fg=mw.RED)


# ======================================================================
# Interactive Replay
# ======================================================================
SPEEDS = {"1x": 1, "2x": 2, "5x": 5, "10x": 10, "25x": 25, "50x": 50}
VISIBLE_BARS = 110


def build_replay_tab(win) -> None:
    mw = _mw()
    f = win._scrollable(win.tab_replay)
    win._page_header(
        f, "", "Interactive Replay",
        "Step bar-by-bar through a historical session. This replays the exact trades one normal backtest "
        "produced -- the same engine as Run & Report -- so it can never show a different result; it just lets "
        "you watch it unfold.",
    )
    win._rp = {"data": None, "i": 0, "playing": False, "job": None, "preparing": False}

    setup = win._section(f, "Replay setup", emphasize=True)
    grid = Frame(setup, bg=mw.PANEL)
    grid.pack(fill="x", padx=14, pady=(0, 6))
    grid.columnconfigure(1, weight=1)
    grid.columnconfigure(3, weight=1)
    win._rp_dataset = StringVar()
    win._rp_strategy = StringVar()
    fields = {
        "account": StringVar(value="50000"), "risk_mode": StringVar(value="percent"), "risk": StringVar(value="1.0"),
        "pip": StringVar(value="auto"), "commission": StringVar(value="0"), "max_bars": StringVar(value="2000"),
        "target": StringVar(value="10"), "daily": StringVar(value="5"), "dd": StringVar(value="10"),
    }
    win._rp_fields = fields

    def _entry(parent, var, width=14):
        return Entry(parent, textvariable=var, width=width, bg=mw.PANEL_3, fg=mw.TEXT, insertbackground=mw.TEXT,
                     relief="flat", highlightthickness=1, highlightbackground=mw.BORDER_LIGHT,
                     highlightcolor=mw.ACCENT, font=mw._safe_font(10))

    def _lbl(r, c, text):
        Label(grid, text=text, bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9), anchor="w").grid(
            row=r, column=c, sticky="w", pady=4, padx=(0, 8))

    _lbl(0, 0, "Dataset")
    win._rp_dataset_combo = ttk.Combobox(grid, textvariable=win._rp_dataset, state="readonly", width=44,
                                         style="T58.TCombobox", font=mw._safe_font(9))
    win._rp_dataset_combo.grid(row=0, column=1, sticky="w", pady=4)
    _lbl(0, 2, "Strategy (from library)")
    win._rp_strategy_combo = ttk.Combobox(grid, textvariable=win._rp_strategy, state="readonly", width=44,
                                          style="T58.TCombobox", font=mw._safe_font(9))
    win._rp_strategy_combo.grid(row=0, column=3, sticky="w", pady=4)
    _lbl(1, 0, "Account size ($)")
    _entry(grid, fields["account"]).grid(row=1, column=1, sticky="w", pady=4)
    _lbl(1, 2, "Risk per trade")
    rr = Frame(grid, bg=mw.PANEL)
    rr.grid(row=1, column=3, sticky="w", pady=4)
    ttk.Combobox(rr, textvariable=fields["risk_mode"], values=["percent", "fixed"], state="readonly", width=8,
                 style="T58.TCombobox", font=mw._safe_font(9)).pack(side="left", padx=(0, 6))
    _entry(rr, fields["risk"], 8).pack(side="left")
    _lbl(2, 0, "Pip size (or 'auto')")
    _entry(grid, fields["pip"]).grid(row=2, column=1, sticky="w", pady=4)
    _lbl(2, 2, "Commission per trade ($)")
    _entry(grid, fields["commission"]).grid(row=2, column=3, sticky="w", pady=4)
    _lbl(3, 0, "Profit target (%) — simulated")
    _entry(grid, fields["target"]).grid(row=3, column=1, sticky="w", pady=4)
    _lbl(3, 2, "Daily loss limit (%) — simulated")
    _entry(grid, fields["daily"]).grid(row=3, column=3, sticky="w", pady=4)
    _lbl(4, 0, "Max drawdown (%) — simulated")
    _entry(grid, fields["dd"]).grid(row=4, column=1, sticky="w", pady=4)
    _lbl(4, 2, "Bars to replay (latest N)")
    _entry(grid, fields["max_bars"]).grid(row=4, column=3, sticky="w", pady=4)

    brow = Frame(setup, bg=mw.PANEL)
    brow.pack(anchor="w", padx=14, pady=(4, 12))
    win._rp_prepare_btn = win._button(brow, "PREPARE REPLAY", lambda: _rp_prepare(win), primary=True)
    win._rp_prepare_btn.pack(side="left")
    win._button(brow, "REFRESH LISTS", lambda: _rp_load_lists(win)).pack(side="left", padx=8)
    win._rp_status = Label(brow, text="", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9))
    win._rp_status.pack(side="left", padx=10)

    # ---------------- player
    win._rp_view = Frame(f, bg=mw.BG)
    win._rp_view.pack(fill="x")
    _rp_build_player(win)
    _rp_load_lists(win)


def _rp_load_lists(win) -> None:
    mw = _mw()
    try:
        win._rp_datasets = {}
        for g in mw.list_datasets_by_instrument(count_rows=False):
            for fi in g["files"]:
                if not fi["empty"]:
                    win._rp_datasets[f"{g['instrument']}  /  {fi['name']}"] = fi["full_name"]
        names = list(win._rp_datasets)
        win._rp_dataset_combo.configure(values=names)
        if names and win._rp_dataset.get() not in names:
            win._rp_dataset.set(names[0])
        items = mw.list_saved_strategies()
        win._rp_strategies = {f"{i.name}   [{i.strategy_type}]": i for i in items}
        snames = list(win._rp_strategies)
        win._rp_strategy_combo.configure(values=snames)
        if snames and win._rp_strategy.get() not in snames:
            win._rp_strategy.set(snames[0])
    except Exception as exc:  # noqa: BLE001
        win._rp_status.config(text=f"Could not list data/strategies: {exc}", fg=mw.RED)


def _rp_float(var, default, name):
    raw = var.get().strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"'{name}' must be a number.")


def _rp_prepare(win) -> None:
    mw = _mw()
    st = win._rp
    if st["preparing"]:
        return
    ds_label, st_label = win._rp_dataset.get(), win._rp_strategy.get()
    if not ds_label or not st_label:
        messagebox.showinfo("Replay", "Pick a dataset and a strategy first (import data on the Market Data tab; "
                                      "create or save a strategy in the Strategy Library).")
        return
    try:
        f = win._rp_fields
        account = _rp_float(f["account"], 50000.0, "Account size")
        risk_value = _rp_float(f["risk"], 1.0, "Risk per trade")
        commission = _rp_float(f["commission"], 0.0, "Commission")
        max_bars = int(_rp_float(f["max_bars"], 2000, "Bars to replay"))
        target = _rp_float(f["target"], None, "Profit target")
        daily = _rp_float(f["daily"], None, "Daily loss limit")
        dd = _rp_float(f["dd"], None, "Max drawdown")
        pip_raw = f["pip"].get().strip().lower()
        pip = None if pip_raw in ("", "auto") else _rp_float(f["pip"], None, "Pip size")
        risk_mode = f["risk_mode"].get()
    except ValueError as exc:
        messagebox.showerror("Replay", str(exc))
        return
    path = mw.get_raw_data_dir() / win._rp_datasets[ds_label]
    item = win._rp_strategies[st_label]
    st["preparing"] = True
    win._rp_prepare_btn.config(state="disabled")
    win._rp_status.config(text="Running the backtest...", fg=mw.TEXT_MUTED)
    _rp_stop(win)

    from app.ui import tk_safety

    def work():
        try:
            from app.backtest.engine import run_backtest
            from app.backtest.risk import RiskConfig, build_run_context, suggest_pip_size
            from app.prop.simulator import PropRules
            from app.data.importer import import_csv
            from app.data.timeframe_resample import prepare_timeframe_aligned_data
            from app.strategy.library_loader import load_strategy_object

            res = import_csv(path)
            if not res.is_valid:
                raise ValueError("; ".join(res.errors) or "the dataset has no usable rows")
            df = res.dataframe
            strategy = load_strategy_object(item)
            df, _warn = prepare_timeframe_aligned_data(df, strategy)
            if len(df) > max_bars:
                df = df.tail(max_bars).reset_index(drop=True)
            base_risk = RiskConfig(
                initial_balance=account, risk_mode=risk_mode, risk_value=risk_value,
                pip_size=pip if pip else suggest_pip_size(df), commission_per_trade=commission,
            )
            # Profit target / daily loss / max drawdown are SIMULATED, not
            # display-only: build a PropRules from the three fields and
            # route the risk through build_run_context so the raw engine
            # itself enforces them (prop account model, balance synced).
            _prop_kwargs = {"account_size": account}
            if target is not None:
                _prop_kwargs["evaluation_profit_target_pct"] = target
            if daily is not None:
                _prop_kwargs["daily_loss_limit_pct"] = daily
            if dd is not None:
                _prop_kwargs["max_drawdown_pct"] = dd
            prop_rules = PropRules(**_prop_kwargs)
            risk = build_run_context(base_risk, prop_rules)
            result = run_backtest(df, strategy, risk)
            data = _rp_pack(df, result, account, target, daily, dd, f"{ds_label}  \u00b7  {item.name}")
            tk_safety.call_soon(lambda: _rp_ready(win, data))
        except Exception as exc:  # noqa: BLE001
            from app.reports.crash_log import log_crash
            log_crash("Interactive Replay prepare", exc)
            tk_safety.call_soon(lambda e=exc: _rp_failed(win, e))

    threading.Thread(target=work, daemon=True).start()


def _rp_failed(win, exc) -> None:
    mw = _mw()
    win._rp["preparing"] = False
    try:
        win._rp_prepare_btn.config(state="normal")
        win._rp_status.config(text=f"Could not prepare the replay: {exc}", fg=mw.RED)
    except Exception:
        pass


def _rp_pack(df, result, account, target, daily, dd, label) -> dict:
    """Everything the player needs, as plain numpy arrays / lists -- computed
    once so stepping through bars is just indexing."""
    import numpy as np
    import pandas as pd

    ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
    tsv = ts.values
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(df)

    eq = np.full(n, float(account))
    ec = result.equity_curve
    if ec is not None and len(ec) and "timestamp" in ec.columns and "equity" in ec.columns:
        s = pd.Series(ec["equity"].to_numpy(dtype=float), index=pd.to_datetime(ec["timestamp"]))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        eq = s.reindex(ts, method="ffill").fillna(float(account)).to_numpy(dtype=float)

    trades = []
    for t in result.trades:
        ei = int(np.searchsorted(tsv, np.datetime64(pd.Timestamp(t.entry_time)), side="left"))
        xi = int(np.searchsorted(tsv, np.datetime64(pd.Timestamp(t.exit_time)), side="left"))
        ei, xi = min(max(ei, 0), n - 1), min(max(xi, 0), n - 1)
        move = (t.exit_price - t.entry_price) * t.direction
        per_unit = ((t.pnl + t.commission) / move) if move else None  # $ per 1.0 of favourable price move
        trades.append({
            "ei": ei, "xi": max(xi, ei), "dir": t.direction, "entry": t.entry_price, "exit": t.exit_price,
            "pnl": t.pnl, "reason": t.exit_reason, "equity": t.equity_after, "per_unit": per_unit,
            "sl": (t.entry_price - t.direction * t.initial_risk) if t.initial_risk else None,
        })
    day = ts.dt.normalize().values
    day_start = np.empty(n)
    cur_day, start_eq = None, float(account)
    for k in range(n):
        if day[k] != cur_day:
            cur_day, start_eq = day[k], (eq[k - 1] if k else float(account))
        day_start[k] = start_eq
    return {
        "n": n, "ts": ts, "o": o, "h": h, "l": l, "c": c, "eq": eq, "trades": trades, "account": float(account),
        "target": target, "daily": daily, "dd": dd, "peak": np.maximum.accumulate(eq), "day_start": day_start,
        "label": label,
    }


def _rp_ready(win, data) -> None:
    mw = _mw()
    st = win._rp
    st["preparing"] = False
    st["data"] = data
    win._rp_trees_shown = -1
    st["i"] = min(data["n"] - 1, max(0, min(60, data["n"] - 1)))
    win._rp_prepare_btn.config(state="normal")
    win._rp_status.config(
        text=f"Ready: {data['n']:,} bars, {len(data['trades'])} trades. Press \u25b6 to play.", fg=mw.GREEN)
    win._rp_scale.configure(to=max(1, data["n"] - 1))
    _rp_render(win)


def _rp_build_player(win) -> None:
    mw = _mw()
    v = win._rp_view
    card = win._section(v, "Replay")
    win._rp_title = Label(card, text="Prepare a replay above to begin.", bg=mw.PANEL, fg=mw.TEXT_MUTED,
                          font=mw._safe_font(9), anchor="w")
    win._rp_title.pack(anchor="w", padx=14, pady=(0, 6))

    bar = Frame(card, bg=mw.PANEL)
    bar.pack(fill="x", padx=14, pady=(0, 8))
    win._rp_play_btn = win._button(bar, "\u25b6  PLAY", lambda: _rp_toggle(win), primary=True)
    win._rp_play_btn.pack(side="left")
    for text, delta in (("\u23ee", "start"), ("\u25c0", -1), ("\u25b6\u2223", 1), ("NEXT TRADE \u25b6\u25b6", "trade")):
        win._button(bar, text, lambda d=delta: _rp_step(win, d)).pack(side="left", padx=(6, 0))
    Label(bar, text="Speed", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9)).pack(side="left", padx=(16, 4))
    win._rp_speed = StringVar(value="5x")
    ttk.Combobox(bar, textvariable=win._rp_speed, values=list(SPEEDS), state="readonly", width=5,
                 style="T58.TCombobox", font=mw._safe_font(9)).pack(side="left")
    win._rp_pos = Label(bar, text="", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(9))
    win._rp_pos.pack(side="right")

    win._rp_scale = ttk.Scale(card, from_=0, to=1, orient="horizontal", style="T58.Horizontal.TScale",
                              command=lambda val: _rp_seek(win, val))
    win._rp_scale.pack(fill="x", padx=14, pady=(0, 8))

    body = Frame(card, bg=mw.PANEL)
    body.pack(fill="x", padx=14, pady=(0, 12))
    left = Frame(body, bg=mw.PANEL)
    left.pack(side="left", fill="both", expand=True)
    win._rp_chart = Canvas(left, height=360, bg=mw.LOG_BG, highlightthickness=1, highlightbackground=mw.BORDER)
    win._rp_chart.pack(fill="x")
    win._rp_eqc = Canvas(left, height=90, bg=mw.LOG_BG, highlightthickness=1, highlightbackground=mw.BORDER)
    win._rp_eqc.pack(fill="x", pady=(8, 0))
    for cv in (win._rp_chart, win._rp_eqc):
        cv.bind("<Configure>", lambda _e: _rp_render(win))

    right = Frame(body, bg=mw.PANEL, width=270)
    right.pack(side="left", fill="y", padx=(12, 0))
    right.pack_propagate(False)

    def _panel(title):
        Label(right, text=title, bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(8, "bold")).pack(anchor="w", pady=(0, 3))
        box = Frame(right, bg=mw.PANEL_3, highlightthickness=1, highlightbackground=mw.BORDER)
        box.pack(fill="x", pady=(0, 10))
        return box

    acct = _panel("ACCOUNT")
    win._rp_bal = Label(acct, text="--", bg=mw.PANEL_3, fg=mw.TEXT, font=mw._safe_font(18, "bold"), anchor="w")
    win._rp_bal.pack(anchor="w", padx=10, pady=(8, 0))
    win._rp_pl = Label(acct, text="", bg=mw.PANEL_3, fg=mw.TEXT_MUTED, font=mw._safe_font(9), anchor="w")
    win._rp_pl.pack(anchor="w", padx=10, pady=(0, 8))
    pos = _panel("OPEN POSITION")
    win._rp_open = Label(pos, text="Flat", bg=mw.PANEL_3, fg=mw.TEXT_MUTED, font=mw._safe_font(9), anchor="w",
                         justify="left", wraplength=240)
    win._rp_open.pack(anchor="w", padx=10, pady=8)
    prop = _panel("PROP ACCOUNT")
    win._rp_prop = Canvas(prop, height=96, bg=mw.PANEL_3, highlightthickness=0)
    win._rp_prop.pack(fill="x", padx=10, pady=8)

    Label(card, text="TRADES SO FAR", bg=mw.PANEL, fg=mw.TEXT_MUTED, font=mw._safe_font(8, "bold")).pack(anchor="w", padx=14)
    twrap = Frame(card, bg=mw.PANEL)
    twrap.pack(fill="x", padx=14, pady=(3, 14))
    cols = ("n", "dir", "entry", "exit", "pnl", "reason", "bal")
    win._rp_tree = ttk.Treeview(twrap, columns=cols, show="headings", style="T58.Treeview", height=6)
    for col, text, w in (("n", "#", 40), ("dir", "Side", 70), ("entry", "Entry", 110), ("exit", "Exit", 110),
                         ("pnl", "P/L", 110), ("reason", "Exit reason", 130), ("bal", "Balance", 120)):
        win._rp_tree.heading(col, text=text, anchor="w")
        win._rp_tree.column(col, width=w, anchor="w")
    tsb = ttk.Scrollbar(twrap, orient="vertical", command=win._rp_tree.yview, style="T58.Vertical.TScrollbar")
    win._rp_tree.configure(yscrollcommand=tsb.set)
    win._rp_tree.pack(side="left", fill="x", expand=True)
    tsb.pack(side="right", fill="y")
    win._bind_isolated_wheel(win._rp_tree)
    win._rp_tree.tag_configure("win", foreground=mw.GREEN)
    win._rp_tree.tag_configure("loss", foreground=mw.RED)
    win._rp_trees_shown = -1


# ---- playback
def _rp_toggle(win) -> None:
    st = win._rp
    if st["data"] is None:
        return
    if st["playing"]:
        _rp_stop(win)
    else:
        if st["i"] >= st["data"]["n"] - 1:
            st["i"] = 0
        st["playing"] = True
        win._rp_play_btn.config(text="\u23f8  PAUSE")
        _rp_tick(win)


def _rp_stop(win) -> None:
    st = win._rp
    st["playing"] = False
    if st.get("job"):
        try:
            win.root.after_cancel(st["job"])
        except Exception:
            pass
        st["job"] = None
    try:
        win._rp_play_btn.config(text="\u25b6  PLAY")
    except Exception:
        pass


def _rp_tick(win) -> None:
    st = win._rp
    if not st["playing"] or st["data"] is None:
        return
    step = SPEEDS.get(win._rp_speed.get(), 1)
    st["i"] = min(st["data"]["n"] - 1, st["i"] + step)
    _rp_render(win)
    if st["i"] >= st["data"]["n"] - 1:
        _rp_stop(win)
        return
    st["job"] = win.root.after(90, lambda: _rp_tick(win))


def _rp_step(win, delta) -> None:
    st = win._rp
    d = st["data"]
    if d is None:
        return
    _rp_stop(win)
    if delta == "start":
        st["i"] = 0
    elif delta == "trade":
        nxt = next((t["ei"] for t in d["trades"] if t["ei"] > st["i"]), None)
        st["i"] = nxt if nxt is not None else d["n"] - 1
    else:
        st["i"] = min(d["n"] - 1, max(0, st["i"] + delta))
    _rp_render(win)


def _rp_seek(win, value) -> None:
    st = win._rp
    if st["data"] is None or getattr(win, "_rp_seeking", False):
        return
    st["i"] = int(float(value))
    _rp_render(win, from_scale=True)


# ---- rendering
def _rp_render(win, from_scale: bool = False) -> None:
    mw = _mw()
    st = win._rp
    d = st["data"]
    if d is None or not hasattr(win, "_rp_chart"):
        return
    i = st["i"]
    if not from_scale:
        win._rp_seeking = True
        try:
            win._rp_scale.set(i)
        finally:
            win._rp_seeking = False
    win._rp_title.config(text=d["label"], fg=mw.TEXT)
    win._rp_pos.config(text=f"Bar {i + 1:,} / {d['n']:,}   \u00b7   {d['ts'].iloc[i]:%Y-%m-%d %H:%M}")
    _rp_draw_chart(win, d, i)
    _rp_draw_equity(win, d, i)
    _rp_update_panels(win, d, i)


def _rp_draw_chart(win, d, i) -> None:
    mw = _mw()
    cv = win._rp_chart
    cv.delete("all")
    w, h = max(cv.winfo_width(), 300), max(cv.winfo_height(), 200)
    lo_i = max(0, i - VISIBLE_BARS + 1)
    idx = range(lo_i, i + 1)
    hi_p = max(d["h"][lo_i:i + 1]); lo_p = min(d["l"][lo_i:i + 1])
    for t in d["trades"]:  # keep entry/stop/exit markers inside the view
        if t["ei"] <= i and t["xi"] >= lo_i:
            for p in (t["entry"],) + ((t["sl"],) if t["sl"] else ()):
                hi_p, lo_p = max(hi_p, p), min(lo_p, p)
    pad = (hi_p - lo_p) * 0.06 or 1.0
    hi_p, lo_p = hi_p + pad, lo_p - pad
    left, right, top, bottom = 8, 64, 10, 18
    cw, ch = w - left - right, h - top - bottom
    slot = cw / VISIBLE_BARS
    x_of = lambda k: left + (k - lo_i + 0.5) * slot
    y_of = lambda p: top + (hi_p - p) / (hi_p - lo_p) * ch

    for g in range(5):
        p = lo_p + (hi_p - lo_p) * g / 4
        y = y_of(p)
        cv.create_line(left, y, w - right, y, fill=mw.BORDER)
        cv.create_text(w - right + 6, y, text=_px(p), anchor="w", fill=mw.TEXT_DIM, font=mw._safe_font(8))
    up, down = mw.GREEN, mw.RED
    bw = max(1.0, slot * 0.6)
    for k in idx:
        o, hh, ll, c = d["o"][k], d["h"][k], d["l"][k], d["c"][k]
        col = up if c >= o else down
        x = x_of(k)
        cv.create_line(x, y_of(hh), x, y_of(ll), fill=col)
        y1, y2 = y_of(max(o, c)), y_of(min(o, c))
        cv.create_rectangle(x - bw / 2, y1, x + bw / 2, max(y2, y1 + 1), fill=col, outline=col)

    for n, t in enumerate(d["trades"]):
        if t["ei"] > i or t["xi"] < lo_i:
            continue
        long_ = t["dir"] > 0
        ex = x_of(t["ei"])
        ey = y_of(t["entry"])
        col = mw.GREEN if long_ else mw.RED
        tri = [ex, ey - 8, ex - 5, ey + 3, ex + 5, ey + 3] if long_ else [ex, ey + 8, ex - 5, ey - 3, ex + 5, ey - 3]
        cv.create_polygon(tri, fill=col, outline="")
        closed = t["xi"] <= i
        if closed:
            xx, xy = x_of(t["xi"]), y_of(t["exit"])
            ccol = mw.GREEN if t["pnl"] >= 0 else mw.RED
            cv.create_line(ex, ey, xx, xy, fill=ccol, dash=(3, 3))
            cv.create_text(xx, xy, text="\u00d7", fill=ccol, font=mw._safe_font(11, "bold"))
        else:  # open: entry + stop levels
            cv.create_line(ex, ey, w - right, ey, fill=mw.BLUE, dash=(4, 3))
            if t["sl"]:
                sy = y_of(t["sl"])
                cv.create_line(ex, sy, w - right, sy, fill=mw.RED, dash=(4, 3))
                cv.create_text(w - right + 6, sy, text="SL", anchor="w", fill=mw.RED, font=mw._safe_font(8, "bold"))
            cv.create_text(w - right + 6, ey, text="ENTRY", anchor="w", fill=mw.BLUE, font=mw._safe_font(8, "bold"))
    last = d["c"][i]
    ly = y_of(last)
    cv.create_line(left, ly, w - right, ly, fill=mw.TEXT_DIM, dash=(2, 4))
    cv.create_rectangle(w - right + 2, ly - 8, w - 2, ly + 8, fill=mw.PANEL_3, outline=mw.BORDER_LIGHT)
    cv.create_text(w - right + 6, ly, text=_px(last), anchor="w", fill=mw.TEXT, font=mw._safe_font(8, "bold"))


def _rp_draw_equity(win, d, i) -> None:
    mw = _mw()
    cv = win._rp_eqc
    cv.delete("all")
    w, h = max(cv.winfo_width(), 300), max(cv.winfo_height(), 60)
    eq = d["eq"][: i + 1]
    lo, hi = min(d["eq"]), max(d["eq"])
    span = (hi - lo) or 1.0
    n = d["n"]
    base = h - 8 - (d["account"] - lo) / span * (h - 16)
    cv.create_line(6, base, w - 6, base, fill=mw.BORDER, dash=(3, 3))
    step = max(1, len(eq) // 600)
    pts = []
    for k in range(0, len(eq), step):
        pts += [6 + k / max(n - 1, 1) * (w - 12), h - 8 - (eq[k] - lo) / span * (h - 16)]
    if len(pts) >= 4:
        cv.create_line(*pts, fill=mw.GREEN if eq[-1] >= d["account"] else mw.RED, width=2)
    cv.create_text(8, 6, text="EQUITY", anchor="nw", fill=mw.TEXT_DIM, font=mw._safe_font(7, "bold"))


def _rp_update_panels(win, d, i) -> None:
    mw = _mw()
    bal = float(d["eq"][i])
    pl = bal - d["account"]
    win._rp_bal.config(text=f"${bal:,.2f}")
    win._rp_pl.config(text=f"{'+' if pl >= 0 else ''}{pl:,.2f}  ({pl / d['account'] * 100:+.2f}%)",
                      fg=mw.GREEN if pl >= 0 else mw.RED)
    open_t = next((t for t in d["trades"] if t["ei"] <= i < t["xi"]), None)
    if open_t is None:
        win._rp_open.config(text="Flat", fg=mw.TEXT_MUTED)
    else:
        side = "LONG" if open_t["dir"] > 0 else "SHORT"
        text = f"{side} from {_px(open_t['entry'])}"
        if open_t["sl"]:
            text += f"\nStop {_px(open_t['sl'])}"
        colour = mw.TEXT
        if open_t["per_unit"]:
            unreal = (d["c"][i] - open_t["entry"]) * open_t["dir"] * open_t["per_unit"]
            text += f"\nUnrealized (est.): {'+' if unreal >= 0 else ''}{unreal:,.2f}"
            colour = mw.GREEN if unreal >= 0 else mw.RED  # colour by profit/loss, not by side
        win._rp_open.config(text=text, fg=colour)

    # prop bars
    cv = win._rp_prop
    cv.delete("all")
    w = max(cv.winfo_width(), 200)
    acct = d["account"]
    rows = []
    if d["target"]:
        rows.append(("Profit target", max(0.0, pl) / (acct * d["target"] / 100), mw.GREEN,
                     f"{max(0.0, pl) / acct * 100:.1f}% / {d['target']:g}%"))
    if d["daily"]:
        used = max(0.0, d["day_start"][i] - bal)
        rows.append(("Daily loss", used / (acct * d["daily"] / 100), mw.AMBER, f"{used / acct * 100:.1f}% / {d['daily']:g}%"))
    if d["dd"]:
        ddv = max(0.0, d["peak"][i] - bal)
        rows.append(("Max drawdown", ddv / (acct * d["dd"] / 100), mw.RED, f"{ddv / acct * 100:.1f}% / {d['dd']:g}%"))
    if not rows:
        cv.create_text(4, 4, text="Add prop limits in setup\nto track them here.", anchor="nw",
                       fill=mw.TEXT_DIM, font=mw._safe_font(8))
    for r, (name, frac, col, text) in enumerate(rows):
        y = 4 + r * 30
        cv.create_text(0, y, text=name.upper(), anchor="nw", fill=mw.TEXT_MUTED, font=mw._safe_font(7, "bold"))
        cv.create_text(w, y, text=text, anchor="ne", fill=mw.TEXT, font=mw._safe_font(8))
        cv.create_rectangle(0, y + 13, w, y + 19, fill=mw.BORDER, outline="")
        cv.create_rectangle(0, y + 13, w * min(max(frac, 0.0), 1.0), y + 19, fill=col, outline="")

    # trades table (only repainted when the number of closed trades changes)
    closed = [t for t in d["trades"] if t["xi"] <= i]
    if len(closed) != win._rp_trees_shown:
        win._rp_trees_shown = len(closed)
        tree = win._rp_tree
        tree.delete(*tree.get_children())
        for n, t in enumerate(closed, 1):
            tree.insert("", "end", tags=("win" if t["pnl"] >= 0 else "loss",), values=(
                n, "LONG" if t["dir"] > 0 else "SHORT", _px(t["entry"]), _px(t["exit"]),
                f"{t['pnl']:+,.2f}", t["reason"], f"{t['equity']:,.2f}"))
        kids = tree.get_children()
        if kids:
            tree.see(kids[-1])


def _px(p: float) -> str:
    """Price text with a sensible number of decimals for the instrument's scale
    (FX ~1.17436 needs 5, JPY pairs ~150.123 need 3, indices/futures ~5000.25 need 2)."""
    a = abs(p)
    return f"{p:,.5f}" if a < 10 else (f"{p:,.3f}" if a < 1000 else f"{p:,.2f}")


def _mw():
    from app.ui import main_window
    return main_window
