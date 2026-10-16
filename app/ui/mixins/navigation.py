"""Sidebar navigation, page switching, stage stepper, settings popup and global search."""
from __future__ import annotations

import math
import re
import webbrowser
from app.search.global_search import global_search
from app.strategy.library import get_strategy_library_dir
from pathlib import Path
from tkinter import Canvas, END, Entry, Frame, Label, Listbox, StringVar, Toplevel, messagebox, ttk
from app.ui.common import OUTPUT_DIR, _safe_font
from app.ui.platform_win import _apply_dark_titlebar
from app.ui.theme import MONO, SIDEBAR_WIDTH
from app.ui.widgets.basic import _blend_hex
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
SIDEBAR = ""
PANEL_2 = ""
PANEL_3 = ""
PANEL_HOVER = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
GREEN = ""
ACCENT = ""
VIOLET = ""
NEON_CYAN = ""
CURRENT_THEME = "dark"
theme.bind(globals())


class NavigationMixin:
    """Sidebar navigation, page switching, stage stepper, settings popup and global search."""


    # Per-row glyphs -- the same codepoints the web sidebar (_sidebar.html)
    # uses for the same tools, so the two builds read identically.
    _NAV_ICONS = {
        "dashboard": "\u2302", "aiassistant": "\u2609", "manual": "\U0001F4D6",
        "genstrat": "\u2726", "researchagent": "\u2609", "researchdirector": "\U0001F50D",
        "researchloop": "\u21bb", "speedrun": "\u26a1", "speedrunmulti": "\u26a1", "forge": "\u26a1",
        "strategy": "\U0001F4DA",
        "strategyconfig": "\u2699", "data": "\u25a4", "prop": "\u2696", "risk": "\u26a0",
        "run": "\u25b6", "payout": "\u2696", "propfirmrec": "\U0001F3C6",
        "optimizehub": "\U0001F4A1", "fullpipeline": "\u26a1", "quickoptimize": "\u26a1",
        "search": "\u25a4", "evolution": "\u2694", "multiobj": "\u2726", "searchmulti": "\u25a4",
        "evolutionmulti": "\u2694", "refine": "\u267b", "risksweep": "\u2696",
        "validatehub": "\U0001F4A1", "wfo": "\u21bb", "wfga": "\u21bb", "cpcv": "\u22a2",
        "pbo": "\u22a2", "sensitivity": "\u22a2", "paramrobustness": "\u22a2",
        "regimematrix": "\u2609", "montecarlo": "\u2696",
        "leaderboard": "\U0001F3C6",
        "familydiversity": "\u2726", "portfolio": "\u25eb", "ensemble": "\u25eb",
        "autopilot_pointer": "\u26a1", "compare": "\u2696", "strathealth_pointer": "\U0001F4C8",
        "optimize_simple": "\u2726", "validate_simple": "\u2726", "champion_simple": "\u2726",
        "fullpipeline_test_pointer": "\u26a1", "youridea_validate_pointer": "\U0001F4A1",
        "notifications": "\u2699",
        "forwardtest": "\u25ef", "deploylive": "\u26a0", "livemarket": "\u25cf",
        "graveyard": "\U0001F480",
        "quantlab": "\u2696", "optionsoutlook": "\u25eb", "hedgefund": "\u2696",
        "datacenter": "\U0001F4CA", "account": "\u2699", "apikeys": "\U0001F511",
        "support": "\U0001F6DF", "education": "\U0001F393", "resources": "\U0001F393",
        "stratlibrary": "\U0001F4DA", "replay": "\u23ef",
        "starthere_create": "\U0001F4A1", "starthere_test": "\U0001F4A1", "starthere_champion": "\U0001F4A1",
        "starthere_deployment": "\U0001F4A1", "starthere_graveyard": "\U0001F4A1",
        "starthere_quantlab": "\U0001F4A1", "starthere_account": "\U0001F4A1",
    }
    # Plain top-level rows (Dashboard / AI Assistant / User Manual): bold, teal,
    # and -- like the web's un-accented .t58-nav-item -- no resting left bar.
    _NAV_TOP_LEVEL = {"dashboard", "aiassistant", "manual"}
    # v9.7 (Owen): clicking a lifecycle section header's numeral/name jumps
    # straight to that step's main page (the group's "home"), mirroring the
    # web sidebar's linked summaries; the chevron keeps expand/collapse.
    # Keyed by the header's display name (numerals repeat across super-
    # headers -- (1) exists under Lifecycle, Quant Lab AND Account).
    _SECTION_HOME_KEYS = {
        "CREATE": "starthere_create",
        "TEST": "fullpipeline_test_pointer",
        "OPTIMIZE": "optimize_simple",
        "VALIDATE": "validate_simple",
        "FINAL SELECTION": "leaderboard",
        "CHAMPION": "champion_simple",
        "DEPLOYMENT": "forwardtest",
        "STRATEGY GRAVEYARD": "graveyard",
    }
    # Leading glyph or "1  " step-number that older labels carried inline; the
    # icon column now shows the glyph, and the web sidebar doesn't number rows.
    _NAV_LABEL_PREFIX_RE = re.compile(r"^(?:[\u2190-\u2bff\u2600-\u27bf\U0001F000-\U0001FAFF]+\s*|\d+\s{2,})")

    def _nav_hover_bg(self) -> str:
        return PANEL_2 if CURRENT_THEME == "dark" else PANEL_3

    def _build_sidebar_nav(self):
        def _wheel(event):
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            self._sidebar_canvas.yview_scroll(delta, "units")
            return "break"  # see the matching note on _sidebar_wheel above

        # Every collapsible group starts CLOSED (matches the web sidebar,
        # which also launches fully condensed); groups auto-open when a page
        # inside them is shown (see _show_page).
        if not hasattr(self, "_collapsed_groups"):
            self._collapsed_groups = {
                lbl_text for k, icon, lbl_text, _frame, _color in self._nav_items
                if k is None and icon not in ("SUBHEADER", "SUPERHEADER") and lbl_text != "OVERVIEW"
            }

        # Rebuilding from scratch on every toggle is simple and cheap here
        # (~30 widgets total).
        for w in list(self._sidebar_inner.winfo_children()):
            w.destroy()
        self._nav_buttons: dict[str, tuple] = {}

        section_collapsed = False
        for key, _icon, label, frame, color in self._nav_items:
            if key is None and _icon == "SUPERHEADER":
                # Web: .t58-nav-divider + .t58-nav-section-label ("Strategy Lab" etc.)
                wrap = Frame(self._sidebar_inner, bg=SIDEBAR)
                wrap.pack(fill="x", pady=(6, 0))
                Frame(wrap, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 8))
                Label(
                    wrap, text=label.upper(), bg=SIDEBAR, fg=TEXT_DIM,
                    font=_safe_font(8, "bold"), anchor="w", padx=10,
                ).pack(fill="x")
                continue
            if key is None and _icon == "SUBHEADER":
                # Web: .t58-nav-subheader -- tiny label inside a group; hides with it.
                if section_collapsed:
                    continue
                Label(
                    self._sidebar_inner, text=label.upper(), bg=SIDEBAR, fg=TEXT_DIM,
                    font=_safe_font(7, "bold"), anchor="w", padx=14,
                ).pack(fill="x", padx=6, pady=(8, 2))
                continue
            if key is None:
                collapsible = label != "OVERVIEW"
                section_collapsed = collapsible and label in self._collapsed_groups
                if not collapsible:
                    continue  # the plain top-level rows have no header on the web either
                # Web: <summary> = numeral (teal) | NAME | chevron, space-between.
                header_row = Frame(self._sidebar_inner, bg=SIDEBAR, cursor="hand2")
                header_row.pack(fill="x", padx=6, pady=(4, 0))
                numeral, _sep, rest = label.partition(" ")
                has_numeral = len(numeral) == 1 and "\u2460" <= numeral <= "\u2473"
                name_text = (rest if has_numeral else label).upper()
                left = Label(
                    header_row, text=(numeral if has_numeral else ""), bg=SIDEBAR, fg=GREEN,
                    font=_safe_font(10), width=2, anchor="w",
                )
                left.pack(side="left", padx=(8, 0), pady=(8, 6))
                right = Label(
                    header_row, text=("\u25b8" if section_collapsed else "\u25be"), bg=SIDEBAR, fg=TEXT_DIM,
                    font=_safe_font(8), width=2, anchor="e",
                )
                right.pack(side="right", padx=(0, 8))
                mid = Label(
                    header_row, text=name_text, bg=SIDEBAR, fg=TEXT_DIM,
                    font=_safe_font(8, "bold"), anchor="center",
                )
                mid.pack(side="left", fill="x", expand=True)

                def _toggle(_e=None, name=label):
                    if name in self._collapsed_groups:
                        self._collapsed_groups.discard(name)
                    else:
                        self._collapsed_groups.add(name)
                    self._build_sidebar_nav()

                # v9.7: numeral + name jump straight to the step's
                # main page (auto-expanding its group via _show_page);
                # the chevron alone keeps expand/collapse. Sections
                # without a home key keep the old toggle-everywhere
                # behavior.
                home_key = self._SECTION_HOME_KEYS.get(name_text)

                def _go_home(_e=None, key=home_key):
                    if key:
                        self._show_page(key)

                def _hdr_hover(entering, widgets=(header_row, left, mid, right)):
                    for w in widgets:
                        w.configure(bg=(self._nav_hover_bg() if entering else SIDEBAR))

                for w in (header_row, left, mid):
                    w.bind("<Button-1>", _go_home if home_key else _toggle)
                    w.bind("<Enter>", lambda _e, f=_hdr_hover: f(True))
                    w.bind("<Leave>", lambda _e, f=_hdr_hover: f(False))
                    for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                        w.bind(seq, _wheel)
                right.bind("<Button-1>", _toggle)
                right.bind("<Enter>", lambda _e, f=_hdr_hover: f(True))
                right.bind("<Leave>", lambda _e, f=_hdr_hover: f(False))
                for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    right.bind(seq, _wheel)
                continue
            if section_collapsed:
                continue

            top = key in self._NAV_TOP_LEVEL
            # Web .t58-nav-item: [3px accent bar][icon][label], 1px ring that
            # only shows when active.
            row = Frame(
                self._sidebar_inner, bg=SIDEBAR, cursor="hand2",
                highlightthickness=1, highlightbackground=SIDEBAR,
            )
            row.pack(fill="x", padx=6, pady=1)
            # height=1 is deliberate: a Canvas with no explicit height
            # defaults to a large platform size and would balloon every row.
            accent = Canvas(row, bg=SIDEBAR, width=3, height=1, highlightthickness=0)
            accent.pack(side="left", fill="y")
            accent.bind("<Configure>", lambda _e, k=key: self._draw_nav_accent(k))
            icon_lbl = Label(
                row, text=self._NAV_ICONS.get(key, ""), bg=SIDEBAR, fg=color,
                font=_safe_font(10), width=2, anchor="center",
            )
            icon_lbl.pack(side="left", padx=(8, 0))
            row._t58_icon = icon_lbl
            text = self._NAV_LABEL_PREFIX_RE.sub("", label)
            lbl = Label(
                row, text=text, bg=SIDEBAR, fg=TEXT_MUTED,
                font=_safe_font(10, "bold" if (top or text == "Start Here") else "normal"),
                anchor="w", justify="left", padx=6, pady=6,
                wraplength=SIDEBAR_WIDTH - 82,
            )
            lbl.pack(side="left", fill="x", expand=True)
            for widget in (row, accent, icon_lbl, lbl):
                widget.bind("<Button-1>", lambda _e, k=key: self._show_page(k))
                widget.bind("<Enter>", lambda _e, k=key: self._on_nav_hover(k, True))
                widget.bind("<Leave>", lambda _e, k=key: self._on_nav_hover(k, False))
                for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    widget.bind(seq, _wheel)
            self._nav_buttons[key] = (row, accent, lbl, color)
            self._paint_nav_row(key, "active" if key == getattr(self, "active_page", "dashboard") else "idle")

    def _paint_nav_row(self, key: str, state: str):
        """One sidebar row's colors for idle / hover / active -- the web
        sidebar's .t58-nav-item states: hover = panel tint; active = accent-
        tinted fill + accent ring + accent text; resting accent rows keep a
        faint left bar in their section color."""
        if key not in self._nav_buttons:
            return
        row, accent, lbl, color = self._nav_buttons[key]
        icon = getattr(row, "_t58_icon", None)
        if state == "active":
            bg = _blend_hex(SIDEBAR, color, 0.10)
            ring = _blend_hex(SIDEBAR, color, 0.35)
            fg = color
        elif state == "hover":
            bg = ring = self._nav_hover_bg()
            fg = TEXT
        else:
            bg = ring = SIDEBAR
            fg = TEXT_MUTED
        try:
            row.configure(bg=bg, highlightbackground=ring)
            lbl.configure(bg=bg, fg=fg)
            accent.configure(bg=bg)
            if icon is not None:
                icon.configure(bg=bg, fg=color)
        except Exception:
            return
        self._draw_nav_accent(key, state)

    def _draw_nav_accent(self, key: str, state: str | None = None):
        """Paints one row's 3px left accent bar. `state` is "idle" / "hover"
        / "active"; omitted (e.g. on a <Configure> resize) means "whatever
        this row's current state already is"."""
        if key not in self._nav_buttons:
            return
        row, accent, _lbl, color = self._nav_buttons[key]
        if state is None:
            state = "active" if key == getattr(self, "active_page", "dashboard") else "idle"
        accent.delete("all")
        h = max(accent.winfo_height(), 20)
        bg = str(row.cget("bg"))
        accent.configure(bg=bg)
        if key in self._NAV_TOP_LEVEL and state != "active":
            return  # plain top-level rows have no resting bar (web: no data-accent)
        fill = color if state == "active" else _blend_hex(bg, color, 0.55)
        accent.create_rectangle(0, 0, 3, h, fill=fill, outline="")

    def _on_nav_hover(self, key, entering):
        if key == self.active_page:
            return
        self._paint_nav_row(key, "hover" if entering else "idle")

    def _show_page(self, key: str):
        group = self._group_header_for_key(key)
        if group and group in getattr(self, "_collapsed_groups", set()):
            self._collapsed_groups.discard(group)
            self._build_sidebar_nav()
        if getattr(self, "active_page", None) == "replay" and key != "replay":
            try:
                from app.ui import extra_tabs
                extra_tabs._rp_stop(self)
            except Exception:
                pass
        self.active_page = key
        for k in list(self._nav_buttons):
            self._paint_nav_row(k, "active" if k == key else "idle")
        self._update_stage_stepper()
        target = self._tab_frame_by_key.get(key)
        if target is not None:
            for other in self._all_tab_frames:
                if other is not target and other.winfo_manager():
                    other.place_forget()
            target.place(in_=self.content, x=0, y=0, relwidth=1, relheight=1)
            target.lift()
            # Geometry for a frame that was never on screen is only computed once it is
            # mapped -- do it now so charts/canvases sized in the deferred refresh below
            # (after(1, ...)) see real widths instead of Tk's 1px default.
            self.content.update_idletasks()
        if key == "dashboard":
            # Deferred one tick so the tab switch paints immediately and the
            # heavier data load/repaint happens right after (see the
            # 2026-09-03 smoothness note this replaces in git history).
            self.root.after(1, self._refresh_dashboard)
        elif key == "datacenter" and not getattr(self, "_datacenter_scanned", True):
            self.root.after(50, self._refresh_data_center)
        elif key == "stratlibrary":
            from app.ui import extra_tabs
            self.root.after(1, lambda: extra_tabs._lib_refresh(self))
        elif key == "replay":
            from app.ui import extra_tabs
            self.root.after(1, lambda: extra_tabs._rp_load_lists(self))
        elif key == "validatehub":
            # The Validate Start Here checklist is built from results other
            # tabs record -- rebuild on every visit so it is never stale.
            self.root.after(1, self._refresh_validate_hub)

    # Global top bar -- one persistent strip above the sidebar/content
    # split (every tab, every page). Requested layout: brand + tagline on
    # the left, a global search box in the middle (strategies / reports /
    # datasets / runs -- see app.search.global_search), and Engine Online
    # status + theme toggle + settings gear on the right, matching the
    # web app's own top-of-page status/branding so the two builds read as
    # the same product rather than two different apps.
    _SEARCH_PLACEHOLDER = "Search strategies, datasets, reports, runs..."

    def _draw_brand_mark(self, cv: Canvas):
        """The web sidebar's .t58-brand .mark: a rounded cyan->violet
        gradient tile with "T58" in near-black. Tk has no gradients, so it is
        painted column by column, with the corner columns shortened to round
        the tile."""
        w, h, r = 34, 28, 8
        x0, y0 = 1, 1
        for i in range(w):
            t = i / (w - 1)
            color = _blend_hex(NEON_CYAN, VIOLET, t)
            edge = min(i, w - 1 - i)
            inset = 0.0 if edge >= r else r - math.sqrt(max(r * r - (r - edge - 0.5) ** 2, 0.0))
            cv.create_line(x0 + i, y0 + inset, x0 + i, y0 + h - inset, fill=color)
        cv.create_text(x0 + w / 2, y0 + h / 2, text="T58", fill="#04120E", font=_safe_font(8, "bold"))

    def _build_sidebar_chrome(self, sidebar):
        """Brand block + search box (top of the sidebar) and the engine
        status / theme / settings footer (bottom) -- what used to live in a
        separate tall header bar, moved to where the web app keeps it."""
        # -- Brand ---------------------------------------------------------
        brand = Frame(sidebar, bg=SIDEBAR)
        brand.pack(fill="x", padx=10, pady=(14, 8))
        mark = Canvas(brand, width=36, height=30, bg=SIDEBAR, highlightthickness=0)
        mark.pack(side="left", padx=(8, 10))
        self._draw_brand_mark(mark)
        txt = Frame(brand, bg=SIDEBAR)
        txt.pack(side="left", fill="x", expand=True)
        Label(
            txt, text="T58 Quant Algo Backtester", bg=SIDEBAR, fg=TEXT, font=_safe_font(10, "bold"),
            anchor="w", justify="left", wraplength=150,
        ).pack(anchor="w")
        Label(
            txt, text="T58 TRADING", bg=SIDEBAR, fg=TEXT_DIM, font=_safe_font(7, "bold"), anchor="w",
        ).pack(anchor="w", pady=(1, 0))

        # -- Global search (strategies / reports / datasets / runs) --------
        search_wrap = Frame(sidebar, bg=SIDEBAR)
        search_wrap.pack(fill="x", padx=16, pady=(2, 10))
        self._global_search_var = StringVar(value=self._SEARCH_PLACEHOLDER)
        search_entry = Entry(
            search_wrap, textvariable=self._global_search_var, bg=PANEL_3, fg=TEXT_DIM,
            insertbackground=TEXT, relief="flat", font=_safe_font(9),
            highlightthickness=1, highlightbackground=BORDER_LIGHT, highlightcolor=ACCENT,
        )
        search_entry.pack(fill="x", ipady=6)
        self._global_search_entry = search_entry

        def _on_focus_in(_e):
            if self._global_search_var.get() == self._SEARCH_PLACEHOLDER:
                self._global_search_var.set("")
                search_entry.configure(fg=TEXT)

        def _on_focus_out(_e):
            if not self._global_search_var.get().strip():
                self._global_search_var.set(self._SEARCH_PLACEHOLDER)
                search_entry.configure(fg=TEXT_DIM)

        search_entry.bind("<FocusIn>", _on_focus_in)
        search_entry.bind("<FocusOut>", _on_focus_out)
        search_entry.bind("<Return>", lambda _e: self._run_global_search())

        # -- Footer: engine status, theme toggle, settings (packed to the
        # bottom BEFORE the nav canvas so they never get squeezed out) ------
        footer = Frame(sidebar, bg=SIDEBAR)
        footer.pack(side="bottom", fill="x", padx=12, pady=(6, 12))

        def _footer_button(text, command):
            btn = Label(
                footer, text=text, bg=PANEL_3, fg=TEXT_MUTED, font=_safe_font(9, "bold"),
                padx=10, pady=7, anchor="w", cursor="hand2",
                highlightthickness=1, highlightbackground=BORDER_LIGHT,
            )
            btn.pack(fill="x", pady=(4, 0))
            btn.bind("<Button-1>", lambda _e: command())
            btn.bind("<Enter>", lambda _e: btn.configure(bg=PANEL_HOVER, fg=TEXT))
            btn.bind("<Leave>", lambda _e: btn.configure(bg=PANEL_3, fg=TEXT_MUTED))
            return btn

        theme_icon = "\u263D" if CURRENT_THEME == "dark" else "\u2600"
        theme_name = "Dark" if CURRENT_THEME == "dark" else "Light"
        _footer_button("\u2699  Settings", self._open_settings_popup)
        _footer_button(f"{theme_icon}  {theme_name} theme", self._toggle_theme)

        status_row = Frame(footer, bg=SIDEBAR)
        status_row.pack(fill="x", pady=(0, 4))
        Label(status_row, text="\u25CF", bg=SIDEBAR, fg=GREEN, font=_safe_font(8)).pack(side="left", padx=(2, 5))
        Label(
            status_row, text="Engine Online", bg=SIDEBAR, fg=TEXT_MUTED, font=_safe_font(8, "bold"),
        ).pack(side="left")

    def _build_header(self, parent):
        """The main column's top strip: just the persistent stage stepper.
        (Brand, search, engine status and the theme/settings buttons moved
        into the sidebar -- see _build_sidebar_chrome -- to match the web
        layout.)"""
        self._build_stage_stepper(parent)

    # The 8-stage journey (matches the web app's own t58-chrome.js
    # STAGES exactly): Create -> Test -> Optimize -> Validate ->
    # Champion -> Forward Test -> Deploy -> Monitor. `landing_key` is
    # which nav key clicking the stage name jumps to; `keys` is every
    # nav key that counts as "in" this stage, used to highlight whichever
    # stage the currently active page belongs to. A wayfinding aid only --
    # like the web version, it doesn't claim per-strategy completion.
    _STAGE_DEFS = [
        ("Create", "speedrun", {"genstrat", "researchagent", "researchdirector", "researchloop", "youridea",
                                 "autopilot_pointer",
                                 "speedrun", "speedrunmulti", "forge", "strategy", "stratlibrary", "starthere_create"}),
        ("Test", "run", {"starthere_test", "strategyconfig", "data", "prop", "risk", "run", "payout", "propfirmrec",
                          "fullpipeline_test_pointer"}),
        ("Optimize", "fullpipeline", {"optimizehub", "fullpipeline", "quickoptimize", "search", "searchmulti", "evolution",
                                     "evolutionmulti", "multiobj", "refine", "risksweep", "optimize_simple"}),
        ("Validate", "wfo", {"validatehub", "wfo", "wfga", "cpcv", "pbo", "sensitivity", "paramrobustness",
                              "regimematrix", "montecarlo", "validate_simple", "youridea_validate_pointer"}),
        ("Champion", "familydiversity", {"starthere_champion", "familydiversity", "portfolio", "ensemble", "leaderboard",
                                          "champion_simple", "compare", "strathealth_pointer"}),
        ("Forward Test", "forwardtest", {"starthere_deployment", "forwardtest"}),
        ("Deploy", "deploylive", {"deploylive"}),
        ("Monitor", "livemarket", {"livemarket", "replay"}),
    ]

    def _build_stage_stepper(self, parent):
        """Desktop counterpart to the web app's persistent Create-Test-
        Optimize-...-Monitor sequence banner (t58-chrome.js buildStepper) --
        same 8 stages, same wayfinding-not-checklist framing, just a Tk
        row of clickable pills instead of injected HTML."""
        wrap = Frame(parent, bg=BG)
        wrap.pack(fill="x")
        bar = Frame(wrap, bg=BG)
        bar.pack(fill="x", padx=16, pady=(8, 8))
        self._stage_stepper_labels: dict[str, Label] = {}
        for i, (name, landing_key, _keys) in enumerate(self._STAGE_DEFS):
            lbl = Label(
                bar, text=name.upper(), bg=BG, fg=TEXT_DIM, font=_safe_font(8, "bold"),
                cursor="hand2", padx=8, pady=3,
            )
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda _e, k=landing_key: self._show_page(k))
            lbl.bind(
                "<Enter>",
                lambda _e, l=lbl, n=name: l.configure(fg=TEXT_MUTED, bg=PANEL_2) if self._current_stage_name() != n else None,
            )
            lbl.bind("<Leave>", lambda _e: self._update_stage_stepper())
            self._stage_stepper_labels[name] = lbl
            if i < len(self._STAGE_DEFS) - 1:
                Label(bar, text="\u2192", bg=BG, fg=BORDER_LIGHT, font=_safe_font(8)).pack(side="left")
        Frame(wrap, bg=BORDER, height=1).pack(fill="x")  # border under the stepper (web: .t58-stepper)
        self._update_stage_stepper()

    def _current_stage_name(self) -> str | None:
        active = getattr(self, "active_page", "dashboard")
        for name, _landing_key, keys in self._STAGE_DEFS:
            if active in keys:
                return name
        return None

    def _update_stage_stepper(self):
        """Repaints the stage stepper's highlight to match self.active_page --
        called from _show_page (same place the sidebar's own active-row
        highlight is repainted) so the two never disagree about which
        stage is current. Current stage = teal text on a faint teal pill,
        like the web's .t58-stepper a.current."""
        labels = getattr(self, "_stage_stepper_labels", None)
        if not labels:
            return
        current = self._current_stage_name()
        for name, lbl in labels.items():
            if name == current:
                lbl.configure(fg=GREEN, bg=_blend_hex(BG, GREEN, 0.10))
            else:
                lbl.configure(fg=TEXT_DIM, bg=BG)

    def _open_settings_popup(self):
        """Minimal settings surface behind the top bar's gear icon.
        Deliberately small -- this app's real per-feature settings (Ollama
        host/model/API key, prop rules, risk defaults, etc.) already live
        on their own tabs; this is just the handful of things that are
        genuinely global (theme, where things are saved) plus a jump-off
        point to the rest."""
        win = Toplevel(self.root)
        win.title("Settings")
        win.configure(bg=BG)
        win.geometry("420x320")
        try:
            _apply_dark_titlebar(win)
        except Exception:
            pass

        Label(win, text="SETTINGS", bg=BG, fg=TEXT, font=_safe_font(12, "bold")).pack(anchor="w", padx=18, pady=(16, 10))

        row = Frame(win, bg=BG)
        row.pack(fill="x", padx=18, pady=6)
        Label(row, text="Theme", bg=BG, fg=TEXT_MUTED, font=_safe_font(9)).pack(side="left")
        theme_label = "Light" if CURRENT_THEME == "dark" else "Dark"
        self._button(row, f"Switch to {theme_label}", lambda: (self._toggle_theme(), win.destroy())).pack(side="right")

        row2 = Frame(win, bg=BG)
        row2.pack(fill="x", padx=18, pady=6)
        Label(row2, text="Strategy Library folder", bg=BG, fg=TEXT_MUTED, font=_safe_font(9)).pack(anchor="w")
        try:
            lib_dir = str(get_strategy_library_dir("python").parent)
        except Exception:
            lib_dir = "unknown"
        Label(
            row2, text=lib_dir, bg=BG, fg=TEXT_DIM, font=(MONO, 8), wraplength=380, justify="left",
        ).pack(anchor="w", pady=(2, 0))

        row3 = Frame(win, bg=BG)
        row3.pack(fill="x", padx=18, pady=6)
        Label(row3, text="Reports folder", bg=BG, fg=TEXT_MUTED, font=_safe_font(9)).pack(anchor="w")
        Label(
            row3, text=str(OUTPUT_DIR), bg=BG, fg=TEXT_DIM, font=(MONO, 8), wraplength=380, justify="left",
        ).pack(anchor="w", pady=(2, 0))

        Label(
            win,
            text="AI Assist (Ollama), prop rules, and risk defaults are configured on "
            "their own tabs (Research Agent / Prop Rules / Risk).",
            bg=BG, fg=TEXT_DIM, font=_safe_font(8), wraplength=380, justify="left",
        ).pack(anchor="w", padx=18, pady=(14, 6))

        self._button(win, "CLOSE", win.destroy).pack(anchor="e", padx=18, pady=12)

    def _run_global_search(self):
        query = self._global_search_var.get().strip()
        if not query or query == self._SEARCH_PLACEHOLDER:
            return
        try:
            results = global_search(query, max_per_kind=8, reports_dir=OUTPUT_DIR)
        except Exception as exc:  # noqa: BLE001 -- search is a convenience, never worth crashing over
            messagebox.showerror("Search failed", str(exc))
            return
        self._show_global_search_results(query, results)

    def _show_global_search_results(self, query: str, results: list):
        win = Toplevel(self.root)
        win.title(f"Search: {query}")
        win.configure(bg=BG)
        win.geometry("760x520")
        try:
            _apply_dark_titlebar(win)
        except Exception:
            pass

        Label(
            win, text=f"SEARCH RESULTS -- \u201c{query}\u201d", bg=BG, fg=TEXT, font=_safe_font(12, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 2))

        if not results:
            Label(
                win, text="No strategies, reports, datasets, or runs matched.",
                bg=BG, fg=TEXT_DIM, font=_safe_font(9),
            ).pack(anchor="w", padx=16, pady=(4, 12))
            self._button(win, "CLOSE", win.destroy).pack(anchor="e", padx=16, pady=12)
            return

        Label(
            win, text=f"{len(results)} match(es), grouped by kind. Double-click a row to open it.",
            bg=BG, fg=TEXT_DIM, font=_safe_font(8),
        ).pack(anchor="w", padx=16, pady=(0, 8))

        list_frame = Frame(win, bg=PANEL)
        list_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        box = Listbox(
            list_frame, exportselection=False, bg=PANEL_3, fg=TEXT, activestyle="none",
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=box.yview, style="T58.Vertical.TScrollbar")
        box.config(yscrollcommand=scroll.set)
        self._bind_isolated_wheel(box)
        box.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        kind_labels = {"strategy": "STRATEGY", "dataset": "DATASET", "report": "REPORT", "run": "RUN"}
        for r in results:
            box.insert(END, f"[{kind_labels.get(r.kind, r.kind.upper()):9s}] {r.title}  -- {r.subtitle}")

        def _open_selected(_e=None):
            sel = box.curselection()
            if not sel:
                return
            self._open_global_search_result(results[sel[0]])

        box.bind("<Double-Button-1>", _open_selected)

        btn_row = Frame(win, bg=BG)
        btn_row.pack(fill="x", padx=16, pady=(0, 14))
        self._button(btn_row, "OPEN", _open_selected).pack(side="left")
        self._button(btn_row, "CLOSE", win.destroy).pack(side="left", padx=8)

    def _open_global_search_result(self, result) -> None:
        try:
            if result.kind == "report" and result.path:
                webbrowser.open(f"file://{Path(result.path).resolve()}")
            elif result.kind == "strategy" and result.path:
                text = Path(result.path).read_text(encoding="utf-8")
                self._show_text_viewer(result.title, text)
            elif result.kind == "dataset" and result.path:
                self._show_text_viewer(
                    result.title,
                    f"Dataset file: {result.path}\n\nUse Step 2 (Data) to load this file for a run.",
                )
            elif result.kind == "run" and result.experiment_id:
                from app.ai.experiment_memory import get_experiment_by_id

                exp = get_experiment_by_id(result.experiment_id)
                if exp is not None:
                    body = "\n".join(f"{k}: {v}" for k, v in exp.to_dict().items() if k != "config")
                    self._show_text_viewer(f"Run -- {exp.strategy_name}", body)
        except Exception as exc:  # noqa: BLE001 -- opening a result is best-effort
            messagebox.showerror("Couldn't open result", str(exc))
