"""Shared widget/layout helpers used by every tab (styles, buttons, scrollables, sections)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ui.condition_builder import ConditionList
    import pandas as pd

import os
from app.data.importer import import_csv
from app.data.multi_timeframe import merge_multi_timeframe
from app.monte_carlo.engine import MonteCarloConfig
from tkinter import Button, Canvas, Frame, Label, Text, ttk
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import _blend_hex
from app.ui.widgets.cards import RoundedCard, _card
from app.ui.widgets.dialogs import _ChoiceListDialog
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
PANEL_2 = ""
PANEL_3 = ""
PANEL_HOVER = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
LOG_BG = ""
GREEN = ""
ACCENT = ""
ACCENT_HOVER = ""
ACCENT_INK = ""
theme.bind(globals())


class HelpersMixin:
    """Shared widget/layout helpers used by every tab (styles, buttons, scrollables, sections)."""


    def _configure_styles(self):
        style = ttk.Style(self.root)

        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "T58.TNotebook",
            background=BG,
            borderwidth=0,
            tabmargins=[0, 0, 0, 0],
        )
        style.configure(
            "T58.TNotebook.Tab",
            background=PANEL,
            foreground=TEXT_MUTED,
            padding=[20, 12],
            borderwidth=0,
            font=_safe_font(9, "bold"),
        )
        style.map(
            "T58.TNotebook.Tab",
            background=[
                ("selected", PANEL_2),
                ("active", PANEL_3),
            ],
            foreground=[
                ("selected", ACCENT_HOVER),
                ("active", TEXT),
            ],
        )

        # Slim, arrowless scrollbar -- the web app's minimal dark scrollbar
        # rather than clam's chunky arrow-button default.
        style.layout(
            "T58.Vertical.TScrollbar",
            [("Vertical.Scrollbar.trough", {
                "sticky": "ns",
                "children": [("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})],
            })],
        )
        style.configure(
            "T58.Vertical.TScrollbar",
            background=BORDER_LIGHT,
            troughcolor=BG,
            bordercolor=BG,
            lightcolor=BORDER_LIGHT,
            darkcolor=BORDER_LIGHT,
            gripcount=0,
            width=10,
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "T58.Vertical.TScrollbar",
            background=[("pressed", TEXT_MUTED), ("active", TEXT_DIM)],
            lightcolor=[("pressed", TEXT_MUTED), ("active", TEXT_DIM)],
            darkcolor=[("pressed", TEXT_MUTED), ("active", TEXT_DIM)],
        )

        style.configure(
            "T58.Horizontal.TScale", background=ACCENT, troughcolor=PANEL_3, bordercolor=BORDER,
            lightcolor=ACCENT, darkcolor=ACCENT, borderwidth=0, sliderlength=18,
        )
        style.configure(
            "T58.TCombobox",
            fieldbackground=PANEL_3,
            background=PANEL_3,
            foreground=TEXT,
            arrowcolor=TEXT_MUTED,
            bordercolor=BORDER_LIGHT,
            lightcolor=PANEL_3,
            darkcolor=PANEL_3,
            selectbackground=PANEL_3,
            selectforeground=TEXT,
            padding=4,
        )
        style.map(
            "T58.TCombobox",
            fieldbackground=[("readonly", PANEL_3)],
            foreground=[("readonly", TEXT)],
            background=[("readonly", PANEL_3)],
            bordercolor=[("focus", ACCENT)],
        )
        self.root.option_add("*TCombobox*Listbox.background", PANEL_3)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", BORDER_LIGHT)

        style.configure(
            "T58.Horizontal.TProgressbar",
            background=ACCENT,
            troughcolor=PANEL_3,
            bordercolor=PANEL_3,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )

        # Tables -- the web's flat table look: no outer frame, flat muted
        # header, roomy rows, accent-tinted selection. Covers the two custom
        # names the Dashboard uses plus the stock "Treeview" so any tab that
        # never named a style still matches.
        sel_bg = _blend_hex(PANEL, ACCENT, 0.16)
        for name in ("Treeview", "T58.Treeview", "T58Board.Treeview"):
            style.configure(
                name, background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                rowheight=28, borderwidth=0, relief="flat",
                bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL, font=_safe_font(9),
            )
            style.map(
                name,
                background=[("selected", sel_bg)],
                foreground=[("selected", TEXT)],
            )
            style.layout(name, [("Treeview.treearea", {"sticky": "nswe"})])  # drop clam's outer border
            style.configure(
                f"{name}.Heading", background=PANEL, foreground=TEXT_MUTED, relief="flat",
                borderwidth=0, bordercolor=BORDER, lightcolor=PANEL, darkcolor=PANEL,
                padding=(8, 8), font=_safe_font(9, "bold"),
            )
            style.map(f"{name}.Heading", background=[("active", PANEL_3)])

    def _page_header(self, parent, eyebrow, title, description=""):
        """Web .t58-page-header: a plain 18px title over a 12px muted
        description -- no eyebrow line, no rule (the stage stepper already
        says where you are). `eyebrow` is accepted for call-site
        compatibility but deliberately not rendered."""
        box = Frame(parent, bg=BG)
        box.pack(fill="x", padx=16, pady=(16, 10))

        Label(
            box, text=title, bg=BG, fg=TEXT, font=_safe_font(14, "bold"), anchor="w", justify="left",
        ).pack(anchor="w")

        if description:
            Label(
                box, text=description, bg=BG, fg=TEXT_MUTED, font=_safe_font(9),
                wraplength=1000, justify="left", anchor="w",
            ).pack(anchor="w", pady=(4, 0))

    def _condition_list_buttons(self, parent, clist: "ConditionList", pady):
        """+ Add Condition / Undo / Redo row for one ConditionList -- shared
        by all four (long/short entry, long/short exit) condition builders
        so add/remove/reload mistakes can be reversed in one click."""
        row = Frame(parent, bg=PANEL)
        row.pack(anchor="w", padx=18, pady=pady)
        self._button(row, "+ Add Condition", clist.add_row).pack(side="left")
        self._button(row, "Undo", clist.undo, width=8).pack(side="left", padx=(6, 0))
        self._button(row, "Redo", clist.redo, width=8).pack(side="left", padx=(6, 0))
        return row

    def _button(self, parent, text, command, primary: bool = False, width: int | None = None) -> Button:
        """Builds one themed Button with hand-rolled hover recoloring -- Tk's
        native Button only recolors on click, not on hover. Styled after the
        web's .t58-btn: primary = solid teal with near-black ink; default =
        panel-3 fill with a --border-light outline. Shared by every tab in
        the app (~190 call sites)."""
        kwargs = {
            "text": text,
            "command": command,
            "font": _safe_font(9, "bold"),
            "relief": "flat",
            "bd": 0,
            "cursor": "hand2",
            "padx": 16,
            "pady": 8,
            "disabledforeground": TEXT_DIM,
        }

        if primary:
            kwargs.update(
                bg=ACCENT,
                fg=ACCENT_INK,
                activebackground=ACCENT_HOVER,
                activeforeground=ACCENT_INK,
                highlightthickness=0,
            )
            hover_bg, idle_bg = ACCENT_HOVER, ACCENT
        else:
            kwargs.update(
                bg=PANEL_3,
                fg=TEXT,
                activebackground=PANEL_HOVER,
                activeforeground=TEXT,
                highlightthickness=1,
                highlightbackground=BORDER_LIGHT,
                highlightcolor=ACCENT,
            )
            hover_bg, idle_bg = PANEL_HOVER, PANEL_3

        if width:
            kwargs["width"] = width

        btn = Button(parent, **kwargs)

        def _on_enter(_e, b=btn, c=hover_bg):
            if str(b["state"]) != "disabled":
                b.configure(bg=c)

        def _on_leave(_e, b=btn, c=idle_bg):
            if str(b["state"]) != "disabled":
                b.configure(bg=c)

        btn.bind("<Enter>", _on_enter)
        btn.bind("<Leave>", _on_leave)

        return btn

    def _scrollable(self, parent) -> Frame:
        """Wraps `parent` in a mouse-wheel-scrollable canvas and returns an
        inner Frame to build content into. Used for every tab in the app.

        Every tab frame (self.tab_dashboard, self.tab_data, ...) is
        `.place()`-d on top of every other one at the exact same (x=0, y=0,
        relwidth=1, relheight=1) rectangle within self.content, and
        `_show_page` switches between them with `.lift()` -- so a hidden
        tab's canvas is still `winfo_ismapped()` and reports the exact same
        on-screen bounding box as whichever tab actually happens to be on
        top. A wheel handler that dispatches by "which registered canvas's
        bounding box contains the cursor" therefore always resolves to
        whichever canvas was registered FIRST, regardless of which page is
        actually visible -- it can never correctly reach any tab but that
        one. Keying the dispatch off `self.active_page` (which `_show_page`
        always keeps current) instead of screen geometry is correct for
        this app's actual place()+lift() tab-switching, and also means the
        cursor doesn't need to be over any particular widget for the wheel
        to work -- anywhere over the content area scrolls the active tab.
        """
        outer = Frame(parent, bg=BG)
        outer.pack(fill="both", expand=True)

        canvas = Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, style="T58.Vertical.TScrollbar")
        inner = Frame(canvas, bg=BG)

        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        if not hasattr(self, "_scroll_canvas_by_tab"):
            self._scroll_canvas_by_tab = {}
        self._scroll_canvas_by_tab[parent] = canvas

        def _dispatch_wheel(_event, delta):
            frame = self._tab_frame_by_key.get(self.active_page)
            target = self._scroll_canvas_by_tab.get(frame) if frame is not None else None
            if target is not None:
                try:
                    target.yview_scroll(int(delta), "units")
                except Exception:
                    pass

        if not getattr(self, "_wheel_bound", False):
            self._wheel_bound = True
            self.root.bind_all(
                "<MouseWheel>",
                lambda e: _dispatch_wheel(e, -1 * (e.delta // 120)),
            )
            self.root.bind_all("<Button-4>", lambda e: _dispatch_wheel(e, -1))
            self.root.bind_all("<Button-5>", lambda e: _dispatch_wheel(e, 1))

        return inner

    def _section(self, parent, title, subtitle="", emphasize=False):
        """A rounded web-style card (.t58-card: panel fill, 1px border).
        `emphasize=True` marks the one primary action card on a tab with a
        teal-tinted border (the web's .t58-champion treatment) so each
        screen still has one clear focal point."""
        wrap = Frame(parent, bg=BG)
        wrap.pack(fill="x", padx=16, pady=7)

        card = RoundedCard(wrap, fill=PANEL, border=(_blend_hex(BORDER, GREEN, 0.55) if emphasize else BORDER))
        card.pack(fill="both", expand=True)
        box = card.body

        title_row = Frame(box, bg=PANEL)
        title_row.pack(fill="x", padx=14, pady=(10, 2))

        Label(
            title_row,
            text=title.upper(),
            bg=PANEL,
            fg=GREEN if emphasize else TEXT_MUTED,
            font=_safe_font(9, "bold"),
        ).pack(side="left")

        if subtitle:
            Label(
                box,
                text=subtitle,
                bg=PANEL,
                fg=TEXT_MUTED,
                font=_safe_font(9),
                wraplength=920,
                justify="left",
            ).pack(anchor="w", padx=14, pady=(0, 8))

        return box

    # -----------------------------------------------------------------------
    # Dashboard — live stats across every strategy run through the app
    # -----------------------------------------------------------------------

    @staticmethod
    def _blend(color_hex: str, toward_hex: str, t: float) -> str:
        """Blend color_hex toward toward_hex by fraction t (0=color, 1=toward).
        Tkinter's Canvas has no real alpha compositing, so this is how the
        glow halos fake a soft falloff against the known dark background."""
        c = color_hex.lstrip("#")
        b = toward_hex.lstrip("#")
        cr, cg, cb = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        br, bg_, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
        r = round(cr + (br - cr) * t)
        g = round(cg + (bg_ - cg) * t)
        bl = round(cb + (bb - cb) * t)
        return f"#{r:02x}{g:02x}{bl:02x}"

    def _glow_line(self, canvas: Canvas, points, color, width=2):
        if len(points) < 4:
            return
        for halo_width, t in ((7, 0.75), (5, 0.55), (3.2, 0.3)):
            canvas.create_line(*points, fill=self._blend(color, PANEL_2, t), width=halo_width, smooth=True)
        canvas.create_line(*points, fill=color, width=width, smooth=True)

    def _glow_dot(self, canvas: Canvas, x, y, r, color, ring_color=None):
        for dr, t in ((r * 2.2, 0.8), (r * 1.6, 0.55)):
            blended = self._blend(color, PANEL_2, t)
            canvas.create_oval(x - dr, y - dr, x + dr, y + dr, fill=blended, outline="")
        canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=ring_color or color, width=1.5)

    def _generic_text_wheel(self, text_widget, event):
        delta = -1 if getattr(event, "delta", 0) > 0 else 1
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        text_widget.yview_scroll(delta * 3, "units")
        return "break"

    def _bind_isolated_wheel(self, widget):
        """Binds mouse-wheel scrolling directly on `widget` (any Listbox or
        Text that has its own scrollbar) and returns "break" from the
        handler (see _generic_text_wheel) so the event stops right there.

        Every tab's content lives inside a page-level scrollable canvas
        built by _scrollable(), which registers ONE wheel handler via
        self.root.bind_all(...). bind_all fires on the "all" bindtag, which
        Tk checks last -- after the widget's own bindings -- but only if
        nothing earlier in the chain returned "break". Listbox/Text already
        have built-in class-level scroll bindings that do NOT return
        "break", so without this, every wheel tick over one of these small
        boxes was doing BOTH things at once: scrolling the box itself via
        its own default binding, AND scrolling the entire page underneath
        it via the bind_all dispatcher -- which is exactly the "small boxes
        can't scroll separately from the whole page" behavior Owen reported
        (Evolution Lab's tested-strategies list, leaderboard, log boxes,
        every Step's output console, etc). Binding directly on the widget
        and returning "break" here intercepts the event before it can ever
        reach the page-level dispatcher, so this box scrolls on its own and
        the page underneath it stays put.
        """
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(seq, lambda e, w=widget: self._generic_text_wheel(w, e))

    def _bind_isolated_wheel_tree(self, root_widget, scroll_widget=None):
        """Like _bind_isolated_wheel, but also walks every CURRENT child
        (and grandchild, etc.) of root_widget and binds each one the same
        way, redirecting the scroll to `scroll_widget` (defaults to
        root_widget itself).

        _bind_isolated_wheel alone only intercepts the wheel while the
        cursor is directly over the container widget -- e.g. the empty
        margin of a canvas. A canvas fully packed with child widgets (a
        checkbox per row, a label per row, etc., as in Evolution Lab's
        Families-to-include list) means the cursor is almost always over
        one of those CHILDREN instead, which have no such binding and no
        "break", so the event falls through to the page-level bind_all
        dispatcher and scrolls the whole page underneath instead of the
        list the cursor is actually sitting on. Binding every child closes
        that gap. Only covers children that already exist at call time --
        fine for a list built once up front (like Families-to-include),
        not meant for a box whose children keep changing afterward.
        """
        target = scroll_widget if scroll_widget is not None else root_widget

        def _apply(widget):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(seq, lambda e, w=target: self._generic_text_wheel(w, e))
            for child in widget.winfo_children():
                _apply(child)

        _apply(root_widget)

    def _ask_choice_from_list(self, title: str, prompt: str, choices: list[str]) -> str | None:
        """Shows a small modal single-select list (see _ChoiceListDialog)
        and returns the chosen string, or None if cancelled / nothing
        picked."""
        return _ChoiceListDialog.ask(self.root, title, prompt, choices)


    # -----------------------------------------------------------------------
    # Validation Lab — shared helpers used by all six tabs below
    # -----------------------------------------------------------------------

    def _load_df_for_page(self, log_fn) -> "pd.DataFrame | None":
        """Loads (and, if multiple files are selected on Step 2, merges) the
        currently-selected market data the exact same way every other tab
        does. Returns None (after logging why) if nothing usable is
        selected -- callers should stop rather than proceed with no data."""
        if not self.csv_paths:
            log_fn("Please select a market data CSV in Step 2 (Market Data) first.")
            return None
        per_file_results = []
        for p in self.csv_paths:
            result = import_csv(p)
            if not result.is_valid:
                log_fn(f"Import errors ({os.path.basename(p)}):\n" + "\n".join(result.errors))
                return None
            per_file_results.append((p, result))
        if len(per_file_results) == 1:
            df = per_file_results[0][1].dataframe
        else:
            df, _labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
        log_fn(f"Loaded {len(df)} bars.")
        return df

    def _validation_mc_config(self, n_simulations: int | None = None) -> MonteCarloConfig:
        n_sims = n_simulations if n_simulations is not None else self.mc_sims.get_int(10000)
        method = self.mc_method.get_str().strip() or "bootstrap"
        return MonteCarloConfig(n_simulations=n_sims, method=method)

    # -----------------------------------------------------------------------
    # v9.6 "pick your ..." pages -- Optimize / Validate / Champion, one
    # clear thing to do each. All three reuse the app's existing patterns
    # wholesale: a RunContextPanel for this-run data/prop/risk, the shared
    # Step 01 strategy builder (_build_strategy), risk hardened through
    # build_run_context, the app's own module-level runners, and a plain
    # text progress log. Anything sophisticated lives under ONE collapsed
    # "Advanced options" expander per page -- nothing here invents a new
    # engine, store, or verdict.
    # -----------------------------------------------------------------------

    def _advanced_expander(self, parent, title="Advanced options"):
        """A collapsed-by-default frame: a toggle button plus a body frame
        that starts hidden and is packed/forgotten on toggle. Returns the
        body frame to build the advanced fields into."""
        wrap = Frame(parent, bg=BG)
        wrap.pack(fill="x", padx=16, pady=7)
        body = Frame(wrap, bg=BG)
        state = {"open": False}

        def _toggle():
            state["open"] = not state["open"]
            if state["open"]:
                body.pack(fill="x")
                btn.config(text=f"\u25be {title}")
            else:
                body.pack_forget()
                btn.config(text=f"\u25b8 {title}")

        btn = self._button(wrap, f"\u25b8 {title}", _toggle)
        btn.pack(anchor="w")
        return body

    def _log_box(self, parent, height=16):
        """The standard output-log widget (Text + scrollbar in a panel
        frame), factored out of the copy every tab used to repeat."""
        frame = Frame(parent, bg=PANEL)
        text = Text(
            frame, height=height, wrap="word", bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview, style="T58.Vertical.TScrollbar")
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(text)
        return text

    def _link_card_grid(self, parent, cards, columns: int = 3):
        """The web's .t58-method-grid: a grid of clickable cards (title +
        one-line description). Hover turns the border teal like the web's
        .t58-method-card:hover. `cards` is [(title, description, nav_key)]."""
        grid = Frame(parent, bg=PANEL)
        grid.pack(fill="x", padx=10, pady=(2, 10))
        for c in range(columns):
            grid.columnconfigure(c, weight=1, uniform="linkcards")
        for i, (title, desc, nav_key) in enumerate(cards):
            body = _card(grid, border=BORDER, fill=PANEL_2)
            body.grid(row=i // columns, column=i % columns, sticky="nsew", padx=5, pady=5)
            card = body._t58_card
            head = Label(
                body, text=title, bg=PANEL_2, fg=TEXT, font=_safe_font(10, "bold"),
                anchor="w", justify="left", wraplength=230, cursor="hand2",
            )
            head.pack(anchor="w", padx=10, pady=(10, 2))
            sub = Label(
                body, text=desc, bg=PANEL_2, fg=TEXT_MUTED, font=_safe_font(9),
                anchor="w", justify="left", wraplength=230, cursor="hand2",
            )
            sub.pack(anchor="w", padx=10, pady=(0, 10))
            for w in (body, head, sub):
                w.bind("<Button-1>", lambda _e, k=nav_key: self._show_page(k))
                w.bind("<Enter>", lambda _e, cd=card: cd.set_border(GREEN))
                w.bind("<Leave>", lambda _e, cd=card: cd.set_border(BORDER))
        return grid
