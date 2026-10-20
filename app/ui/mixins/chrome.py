"""Window chrome: theme toggle, custom title bar, resize grip, heavy-job guard, root UI build."""
from __future__ import annotations

from app.orchestration.resource_guard import HEAVY_JOB_GUARD
from tkinter import Canvas, Frame, Label, PhotoImage, messagebox
from app.ui.common import _asset_path, _safe_font
from app.ui.platform_win import _apply_dark_titlebar, _get_work_area
from app.ui.theme import SIDEBAR_WIDTH, apply_theme
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
SIDEBAR = ""
PANEL_2 = ""
PANEL_HOVER = ""
BORDER = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
METAL_BRIGHT = ""
GREEN = ""
RED = ""
BLUE = ""
NEON_CYAN = ""
NEON_MAGENTA = ""
NEON_LIME = ""
NEON_VIOLET = ""
NEON_AMBER = ""
CURRENT_THEME = "dark"
theme.bind(globals())


class ChromeMixin:
    """Window chrome: theme toggle, custom title bar, resize grip, heavy-job guard, root UI build."""


    def _toggle_theme(self):
        """Dark/Light switch in the header. Repaints the ENTIRE app: color
        constants are plain module globals referenced by name throughout
        this file (not threaded through as a theme object), so the only
        reliable way to make thousands of already-built widgets and
        hand-drawn Canvas charts pick up a new palette is to update those
        globals via apply_theme() and then throw away and rebuild every
        widget from scratch -- reconfiguring each one in place would mean
        separately re-deriving the right color for every Label/Canvas/
        chart/glow effect in the app, which is exactly what building them
        fresh already does correctly. Whichever tab was open stays open
        across the rebuild."""
        new_theme = "light" if CURRENT_THEME == "dark" else "dark"
        current_page = getattr(self, "active_page", "dashboard")
        apply_theme(new_theme)
        try:
            _apply_dark_titlebar(self.root)
        except Exception:
            pass
        for child in self.root.winfo_children():
            child.destroy()
        self._build_ui()
        self._show_page(current_page)

    def _build_custom_titlebar(self):
        """Replaces Windows' own native title bar -- the bright OS-accent
        blue bar repeatedly reported as "ugly" -- with this app's own,
        drawn in its own dark theme. Works together with
        `self.root.overrideredirect(True)` (see __init__): once the
        native caption is gone there is nothing left for Windows to paint
        blue, so unlike the three earlier _apply_dark_titlebar() rounds,
        this doesn't depend on a particular Windows build's DWM support
        actually taking effect. Only ever called when
        self._custom_titlebar_active is True (win32 + overrideredirect
        succeeded) -- macOS/Linux, or a machine where overrideredirect
        itself fails for any reason, keep the native title bar untouched.
        """
        bar = Frame(self.root, bg=PANEL_2, height=32)
        bar.pack(side="top", fill="x")
        bar.pack_propagate(False)
        self._titlebar = bar

        icon_shown = False
        try:
            icon_path = _asset_path("t58_mark_medium.png")
            if icon_path.exists():
                self._titlebar_icon = PhotoImage(file=str(icon_path))
                factor = max(1, self._titlebar_icon.width() // 18)
                if factor > 1:
                    self._titlebar_icon = self._titlebar_icon.subsample(factor, factor)
                Label(bar, image=self._titlebar_icon, bg=PANEL_2).pack(side="left", padx=(10, 6))
                icon_shown = True
        except Exception:
            icon_shown = False

        title_lbl = Label(
            bar, text="T58 Trading — Quant Algo Backtester", bg=PANEL_2, fg=TEXT_MUTED,
            font=_safe_font(9, "bold"),
        )
        title_lbl.pack(side="left", padx=(0 if icon_shown else 10, 0))

        btn_row = Frame(bar, bg=PANEL_2)
        btn_row.pack(side="right")

        def _make_btn(symbol, hover_bg, hover_fg, command):
            b = Label(
                btn_row, text=symbol, bg=PANEL_2, fg=TEXT_MUTED, font=_safe_font(11),
                width=4, cursor="hand2",
            )
            b.pack(side="left", fill="y")
            b.bind("<Button-1>", lambda _e: command())
            b.bind("<Enter>", lambda _e, w=b, hb=hover_bg, hf=hover_fg: w.configure(bg=hb, fg=hf))
            b.bind("<Leave>", lambda _e, w=b: w.configure(bg=PANEL_2, fg=TEXT_MUTED))
            return b

        _make_btn("\u2013", PANEL_HOVER, TEXT, self._titlebar_minimize)
        self._maximize_btn = _make_btn("\u25A1", PANEL_HOVER, TEXT, self._titlebar_toggle_maximize)
        _make_btn("\u2715", RED, "#FFFFFF", self._titlebar_close)

        # Dragging the bar (or its title text) moves the whole window --
        # the one native-caption behavior a borderless window has to
        # reimplement by hand.
        def _start_drag(event):
            self._drag_offset = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

        def _do_drag(event):
            if self._maximized or self._drag_offset is None:
                return
            ox, oy = self._drag_offset
            self.root.geometry(f"+{event.x_root - ox}+{event.y_root - oy}")

        for widget in (bar, title_lbl):
            widget.bind("<Button-1>", _start_drag)
            widget.bind("<B1-Motion>", _do_drag)
            widget.bind("<Double-Button-1>", lambda _e: self._titlebar_toggle_maximize())

    def _titlebar_minimize(self):
        """iconify() on an overrideredirect(True) window doesn't behave
        like a normal minimize on Windows -- the window can vanish with
        no reliable way to bring it back. Briefly restoring the native
        frame for the iconify/deiconify round-trip, then reapplying
        overrideredirect once Windows reports the window "normal" again
        (via the <Map> event a restore fires), is the standard
        workaround."""
        try:
            self.root.overrideredirect(False)
            self.root.iconify()
        except Exception:
            return

        def _watch(_event=None):
            try:
                if self.root.state() == "normal":
                    self.root.overrideredirect(True)
                    self.root.unbind("<Map>", bind_id)
            except Exception:
                pass

        bind_id = self.root.bind("<Map>", _watch)

    def _titlebar_toggle_maximize(self):
        if self._maximized:
            if self._normal_geometry:
                self.root.geometry(self._normal_geometry)
            self._maximized = False
            if hasattr(self, "_maximize_btn"):
                self._maximize_btn.configure(text="\u25A1")
            return
        self._normal_geometry = self.root.geometry()
        area = _get_work_area()
        if area:
            x, y, w, h = area
            self.root.geometry(f"{w}x{h}+{x}+{y}")
        else:
            self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")
        self._maximized = True
        if hasattr(self, "_maximize_btn"):
            self._maximize_btn.configure(text="\u25A3")

    def _titlebar_close(self):
        self.root.destroy()

    def _build_resize_grip(self):
        """A borderless window (overrideredirect(True)) has no native
        resize handles on its edges either -- this puts one small,
        diagonal-cursor grip in the bottom-right corner (the same
        minimalist approach apps like Discord/Spotify use for their own
        custom title bars) rather than reimplementing all four edges plus
        four corners for a desktop app that's rarely resized by dragging
        in the first place."""
        grip = Label(self.root, text="\u2599", bg=BG, fg=TEXT_DIM, cursor="size_nw_se", font=_safe_font(10))
        grip.place(relx=1.0, rely=1.0, anchor="se", x=-2, y=-2)
        grip.lift()
        self._resize_grip = grip
        self._resize_start = None

        def _start(event):
            self._resize_start = (event.x_root, event.y_root, self.root.winfo_width(), self.root.winfo_height())

        def _do(event):
            if self._resize_start is None or self._maximized:
                return
            sx, sy, sw, sh = self._resize_start
            nw = max(900, sw + (event.x_root - sx))
            nh = max(680, sh + (event.y_root - sy))
            self.root.geometry(f"{nw}x{nh}")

        grip.bind("<Button-1>", _start)
        grip.bind("<B1-Motion>", _do)

    def _try_start_heavy_job(self, name: str) -> bool:
        """Acquires app.orchestration.resource_guard.HEAVY_JOB_GUARD for
        one of the multi-worker-process jobs (Search Lab, Evolution Lab,
        Full Pipeline, Speed Run) before starting it. Each of these
        independently spawns up to os.cpu_count() worker processes, each
        holding its own full copy of the loaded market data in memory --
        running more than one of them at the same time (e.g. leaving
        Evolution Lab running overnight, then also starting Full Pipeline
        and Speed Run on top of it) is exactly the kind of memory pileup
        that can freeze/crash the whole machine, not just this app. Returns
        False (and shows a clear warning naming the job already running)
        if another heavy job is already active; the caller should not
        proceed in that case."""
        if HEAVY_JOB_GUARD.try_acquire(name):
            return True
        messagebox.showwarning(
            "Another heavy job is running",
            f"{HEAVY_JOB_GUARD.active_name} is already running. Running more than one of "
            f"Search Lab / Evolution Lab / Full Pipeline / Speed Run at the same time can "
            f"exhaust your machine's memory (each spawns its own worker processes, each "
            f"holding a full copy of the loaded data) and has been known to freeze the whole "
            f"desktop. Please wait for it to finish, or click its STOP button, before "
            f"starting {name}.",
        )
        return False

    def _release_heavy_job(self, name: str) -> None:
        HEAVY_JOB_GUARD.release(name)

    def _build_ui(self):
        self.root.configure(bg=BG)
        self._configure_styles()
        self._pump_splash("Configuring theme...")

        if self._custom_titlebar_active:
            self._build_custom_titlebar()

        # Main application shell -- mirrors the web app's layout (theme.css
        # .t58-shell): a full-height sidebar on the left (brand, search, nav,
        # footer controls), and a main column on the right (stage stepper on
        # top, page content below). Previously this was a tall header bar
        # over a padded sidebar+content split, which is what made the
        # desktop read as a different product from the web build.
        shell = Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True)

        self.sidebar = Frame(shell, bg=SIDEBAR, width=SIDEBAR_WIDTH)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        Frame(shell, bg=BORDER, width=1).pack(side="left", fill="y")  # sidebar border-right

        main_col = Frame(shell, bg=BG)
        main_col.pack(side="left", fill="both", expand=True)

        # Brand + search (top) and theme/settings footer (bottom) are packed
        # into the sidebar BEFORE the scrolling nav canvas, so they keep
        # their space and the nav list takes whatever is left.
        self._build_sidebar_chrome(self.sidebar)
        self._pump_splash("Building navigation...")
        self._build_header(main_col)  # persistent stage stepper

        # The nav list itself scrolls: with 18+ tabs + section dividers it is
        # taller than the sidebar on a small window, and every tab must stay
        # reachable at any window size.
        self._sidebar_canvas = Canvas(self.sidebar, bg=SIDEBAR, highlightthickness=0)
        self._sidebar_canvas.pack(side="left", fill="both", expand=True)
        self._sidebar_inner = Frame(self._sidebar_canvas, bg=SIDEBAR)
        self._sidebar_window_id = self._sidebar_canvas.create_window(
            (0, 0), window=self._sidebar_inner, anchor="nw",
        )
        self._sidebar_inner.bind(
            "<Configure>", lambda _e: self._sidebar_canvas.configure(scrollregion=self._sidebar_canvas.bbox("all")),
        )
        self._sidebar_canvas.bind(
            "<Configure>", lambda e: self._sidebar_canvas.itemconfig(self._sidebar_window_id, width=e.width),
        )

        def _sidebar_wheel(event):
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            self._sidebar_canvas.yview_scroll(delta, "units")
            # "break" stops the event propagating to the global content-area
            # wheel dispatcher bound in _scrollable(), so scrolling the
            # sidebar doesn't also scroll whichever page is underneath.
            return "break"

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._sidebar_canvas.bind(seq, _sidebar_wheel)
            self._sidebar_inner.bind(seq, _sidebar_wheel)

        body = Frame(main_col, bg=BG)
        body.pack(fill="both", expand=True)
        self.content = Frame(body, bg=BG)
        self.content.pack(side="left", fill="both", expand=True)

        self._all_tab_frames: list = []
        self.tab_dashboard = Frame(self.content, bg=BG)
        self.tab_ai_assistant = Frame(self.content, bg=BG)
        self.tab_manual = Frame(self.content, bg=BG)
        self.tab_strategyconfig = Frame(self.content, bg=BG)
        self.tab_data = Frame(self.content, bg=BG)
        self.tab_strategy = Frame(self.content, bg=BG)
        self.tab_prop = Frame(self.content, bg=BG)
        self.tab_risk = Frame(self.content, bg=BG)
        self.tab_run = Frame(self.content, bg=BG)
        self.tab_payout = Frame(self.content, bg=BG)
        self.tab_prop_recommender = Frame(self.content, bg=BG)
        self.tab_refine = Frame(self.content, bg=BG)
        self.tab_search = Frame(self.content, bg=BG)
        self.tab_search_multi = Frame(self.content, bg=BG)
        self.tab_wfo = Frame(self.content, bg=BG)
        self.tab_cpcv = Frame(self.content, bg=BG)
        self.tab_pbo = Frame(self.content, bg=BG)
        self.tab_sensitivity = Frame(self.content, bg=BG)
        self.tab_param_robustness = Frame(self.content, bg=BG)
        self.tab_portfolio = Frame(self.content, bg=BG)
        self.tab_multiobj = Frame(self.content, bg=BG)
        self.tab_wfga = Frame(self.content, bg=BG)
        self.tab_ensemble = Frame(self.content, bg=BG)
        self.tab_fullpipeline = Frame(self.content, bg=BG)
        self.tab_quickoptimize = Frame(self.content, bg=BG)
        self.tab_evolution_multi = Frame(self.content, bg=BG)
        self.tab_risksweep = Frame(self.content, bg=BG)
        self.tab_compare = Frame(self.content, bg=BG)
        self.tab_optimize_simple = Frame(self.content, bg=BG)
        self.tab_validate_simple = Frame(self.content, bg=BG)
        self.tab_champion_simple = Frame(self.content, bg=BG)
        self.tab_optimize_hub = Frame(self.content, bg=BG)
        self.tab_validate_hub = Frame(self.content, bg=BG)
        self.tab_forge = Frame(self.content, bg=BG)
        self.tab_research_director = Frame(self.content, bg=BG)
        self.tab_speedrun = Frame(self.content, bg=BG)
        self.tab_speedrun_multi = Frame(self.content, bg=BG)
        self.tab_research_loop = Frame(self.content, bg=BG)
        self.tab_forwardtest = Frame(self.content, bg=BG)
        self.tab_deploylive = Frame(self.content, bg=BG)
        self.tab_livemarket = Frame(self.content, bg=BG)
        self.tab_genstrat = Frame(self.content, bg=BG)
        self.tab_evolution = Frame(self.content, bg=BG)
        self.tab_researchagent = Frame(self.content, bg=BG)
        self.tab_regime_matrix = Frame(self.content, bg=BG)
        self.tab_family_diversity = Frame(self.content, bg=BG)
        self.tab_quantlab = Frame(self.content, bg=BG)
        self.tab_options_outlook = Frame(self.content, bg=BG)
        self.tab_resources = Frame(self.content, bg=BG)
        self.tab_education = Frame(self.content, bg=BG)
        self.tab_graveyard = Frame(self.content, bg=BG)
        self.tab_datacenter = Frame(self.content, bg=BG)
        self.tab_account = Frame(self.content, bg=BG)
        self.tab_api_keys = Frame(self.content, bg=BG)
        self.tab_support = Frame(self.content, bg=BG)
        self.tab_hedge_fund = Frame(self.content, bg=BG)
        self.tab_leaderboard = Frame(self.content, bg=BG)
        self.tab_stratlibrary = Frame(self.content, bg=BG)
        self.tab_replay = Frame(self.content, bg=BG)
        self.tab_youridea = Frame(self.content, bg=BG)
        # One 'Start Here' page per section (same content as the web app's /start-here/<section>).
        from app.orchestration.section_guides import DESKTOP_SECTIONS
        self._starthere_frames = {sec: Frame(self.content, bg=BG) for sec in DESKTOP_SECTIONS}

        for f in (
            self.tab_dashboard, self.tab_ai_assistant, self.tab_manual, self.tab_resources, self.tab_education, self.tab_strategyconfig, self.tab_data, self.tab_strategy, self.tab_prop,
            self.tab_risk, self.tab_run, self.tab_payout, self.tab_prop_recommender, self.tab_refine, self.tab_search,
            self.tab_wfo, self.tab_cpcv, self.tab_pbo, self.tab_sensitivity, self.tab_param_robustness, self.tab_portfolio,
            self.tab_multiobj, self.tab_wfga, self.tab_ensemble, self.tab_fullpipeline,
            self.tab_quickoptimize, self.tab_search_multi, self.tab_evolution_multi, self.tab_risksweep,
            self.tab_compare, self.tab_optimize_simple, self.tab_validate_simple, self.tab_champion_simple,
            self.tab_optimize_hub, self.tab_validate_hub,
            self.tab_forge, self.tab_research_director,
            self.tab_speedrun, self.tab_speedrun_multi, self.tab_research_loop,
            self.tab_forwardtest, self.tab_deploylive, self.tab_livemarket, self.tab_genstrat,
            self.tab_evolution, self.tab_researchagent, self.tab_regime_matrix, self.tab_family_diversity,
            self.tab_quantlab, self.tab_options_outlook,
            self.tab_graveyard, self.tab_account, self.tab_api_keys, self.tab_support, self.tab_hedge_fund, self.tab_leaderboard,
            self.tab_stratlibrary, self.tab_replay, self.tab_youridea, *self._starthere_frames.values(),
        ):
            # PERF: only the ACTIVE tab is ever mapped (see _show_page). They used to all be
            # placed on top of each other and stay mapped, so every window resize / repaint
            # laid out and redrew ~8,800 widgets across all 65 tabs instead of just the
            # visible one. Frames are still built fully (their widgets exist), just not
            # on screen until their page is shown.
            self._all_tab_frames.append(f)

        # Every entry is (key, icon, label, frame, color). `icon` is kept
        # in the tuple shape for backward compatibility but no longer
        # rendered (see _build_sidebar_nav) -- a leading glyph AND a
        # leading number on every row read as cluttered, so this sidebar
        # picks one signal (the workflow number, where one exists) rather
        # than both. A divider entry has key=None; its `label` slot holds
        # the section header text shown above that group (小-caps style,
        # letter-spaced) instead of a bare line -- every group is named
        # rather than just visually separated, which is what actually
        # makes a long list like this read as organized instead of messy.
        #
        # UPGRADE (Sep 2026 UI pass): regrouped around the user's journey
        # (Create -> Test -> Optimize -> Validate -> Champion -> Forward
        # Test -> Deploy -> Monitor) instead of one flat "WORKFLOW" list --
        # same idea as the web app's sidebar regroup, same 8 stage names,
        # every existing key/frame/builder untouched (this is purely which
        # section a tab is listed under, its label text, and its accent
        # color -- nothing about what a tab does changed).
        self._nav_items = [
            (None, None, "OVERVIEW", None, None),
            ("dashboard", "", "Dashboard", self.tab_dashboard, GREEN),
            ("aiassistant", "", "AI Assistant", self.tab_ai_assistant, GREEN),
            ("stratlibrary", "", "Strategy Library", self.tab_stratlibrary, NEON_VIOLET),
            ("manual", "", "User Manual", self.tab_manual, GREEN),

            (None, "SUPERHEADER", "Lifecycle", None, None),
            (None, None, "\u2460 CREATE", None, NEON_VIOLET),
            ("starthere_create", "", "Start Here", self._starthere_frames["create"], NEON_VIOLET),
            ("genstrat", "", "Generate Strategies (AI)", self.tab_genstrat, NEON_VIOLET),
            ("researchagent", "", "Research Agent", self.tab_researchagent, NEON_VIOLET),
            ("researchdirector", "", "\U0001F50D Research Director", self.tab_research_director, NEON_VIOLET),
            ("researchloop", "", "\u21bb Research Loop (Background)", self.tab_research_loop, NEON_VIOLET),
            ("youridea", "", "\U0001F4A1 Your Idea", self.tab_youridea, NEON_VIOLET),
            # Moved here from DEPLOYMENT (v9.6 lifecycle pass): Overnight
            # Autopilot *creates* a strategy run overnight, so it belongs
            # with the other create/discovery tools. Still just a pointer
            # at Speed Run's own tab/frame, exactly as before.
            ("autopilot_pointer", "", "\u26a1 Overnight Autopilot", self.tab_speedrun, NEON_VIOLET),
            ("speedrun", "", "\u26a1 Speed Run", self.tab_speedrun, NEON_VIOLET),
            ("speedrunmulti", "", "\u26a1 Multi-Instrument Speed Run", self.tab_speedrun_multi, NEON_VIOLET),
            ("forge", "", "\u26a1 Forge Strategy", self.tab_forge, NEON_LIME),
            ("strategy", "", "Strategy Builder", self.tab_strategy, NEON_VIOLET),

            (None, None, "\u2461 TEST", None, NEON_CYAN),
            ("starthere_test", "", "Start Here", self._starthere_frames["test"], NEON_CYAN),
            ("strategyconfig", "", "1  Strategy Configuration", self.tab_strategyconfig, NEON_CYAN),
            ("data", "", "2  Market Data", self.tab_data, NEON_CYAN),
            ("prop", "", "3  Prop-Firm Rules", self.tab_prop, NEON_CYAN),
            ("risk", "", "4  Risk & Execution", self.tab_risk, NEON_CYAN),
            ("run", "", "5  Run & Report", self.tab_run, NEON_CYAN),
            ("fullpipeline_test_pointer", "", "Full Pipeline (all-in-one)", self.tab_fullpipeline, NEON_CYAN),
            ("payout", "", "6  Payout Probability", self.tab_payout, NEON_CYAN),
            ("propfirmrec", "", "7  Prop-Firm Recommender", self.tab_prop_recommender, NEON_CYAN),

            (None, None, "\u2462 OPTIMIZE", None, BLUE),
            ("optimizehub", "", "Start Here", self.tab_optimize_hub, BLUE),
            ("optimize_simple", "", "\u2726 Optimize (pick your engines)", self.tab_optimize_simple, BLUE),
            ("fullpipeline", "", "Full Pipeline (all-in-one)", self.tab_fullpipeline, BLUE),
            ("quickoptimize", "", "\u26a1 Quick Optimize", self.tab_quickoptimize, BLUE),
            ("search", "", "Search Lab", self.tab_search, BLUE),
            ("evolution", "", "Evolution Lab (GA)", self.tab_evolution, BLUE),
            ("multiobj", "", "Multi-Objective Optimization", self.tab_multiobj, BLUE),
            ("searchmulti", "", "Multi-Instrument Search Lab", self.tab_search_multi, BLUE),
            ("evolutionmulti", "", "Multi-Instrument Evolution Lab", self.tab_evolution_multi, BLUE),
            ("refine", "", "Iterative Refinement", self.tab_refine, BLUE),
            ("risksweep", "", "\U0001F4CA Risk Sweep", self.tab_risksweep, BLUE),

            (None, None, "\u2463 VALIDATE", None, NEON_AMBER),
            ("validatehub", "", "Start Here", self.tab_validate_hub, NEON_AMBER),
            ("validate_simple", "", "\u2726 Validate (pick your checks)", self.tab_validate_simple, NEON_AMBER),
            ("youridea_validate_pointer", "", "\U0001F4A1 Your Idea", self.tab_youridea, NEON_AMBER),
            ("wfo", "", "Walk-Forward Optimization", self.tab_wfo, NEON_AMBER),
            ("wfga", "", "Walk-Forward GA", self.tab_wfga, NEON_AMBER),
            ("cpcv", "", "CPCV", self.tab_cpcv, NEON_AMBER),
            ("pbo", "", "PBO", self.tab_pbo, NEON_AMBER),
            ("sensitivity", "", "Sensitivity", self.tab_sensitivity, NEON_AMBER),
            ("paramrobustness", "", "Parameter Stability / Robustness Map", self.tab_param_robustness, NEON_AMBER),
            ("regimematrix", "", "Regime Survival Matrix", self.tab_regime_matrix, NEON_AMBER),
            ("montecarlo", "", "Monte Carlo", self.tab_payout, NEON_AMBER),

            (None, None, "FINAL SELECTION", None, None),
            ("leaderboard", "", "\U0001F3C6 Final Selection Leaderboard", self.tab_leaderboard, NEON_LIME),

            (None, None, "\u2464 CHAMPION", None, NEON_MAGENTA),
            ("starthere_champion", "", "Start Here", self._starthere_frames["champion"], NEON_MAGENTA),
            ("champion_simple", "", "\u2726 Champion Board (promote)", self.tab_champion_simple, NEON_MAGENTA),
            ("familydiversity", "", "Family Diversity", self.tab_family_diversity, NEON_MAGENTA),
            ("portfolio", "", "Multi-Asset Portfolio", self.tab_portfolio, NEON_MAGENTA),
            ("ensemble", "", "Multi-Strategy Ensemble", self.tab_ensemble, NEON_MAGENTA),
            # Moved here from DEPLOYMENT (v9.6 lifecycle pass): Compare and
            # Strategy Health are champion-selection chores, not deployment.
            # Same keys/frames as before -- only their section changed.
            # "Strategy Health" still points at the Quant Lab screen those
            # controls already live on.
            ("compare", "", "\u2696 Compare Strategies", self.tab_compare, NEON_MAGENTA),
            ("strathealth_pointer", "", "\U0001F4C8 Strategy Health / Auto Re-tune", self.tab_quantlab, NEON_MAGENTA),

            (None, None, "\u2465 DEPLOYMENT", None, NEON_LIME),
            ("starthere_deployment", "", "Start Here", self._starthere_frames["deployment"], NEON_LIME),
            (None, "SUBHEADER", "Live Markets", None, None),
            ("forwardtest", "", "Forward Test (MT5)", self.tab_forwardtest, NEON_LIME),
            ("deploylive", "", "Deploy Live", self.tab_deploylive, RED),
            ("livemarket", "", "Monitor (Live Market)", self.tab_livemarket, NEON_CYAN),
            ("replay", "", "Interactive Replay", self.tab_replay, NEON_CYAN),

            # "SUPERHEADER" (icon field) marks a subtle, always-visible,
            # non-collapsible umbrella label ABOVE a run of ordinary
            # (still individually collapsible) section headers -- e.g.
            # everything from CREATE through STRATEGY GRAVEYARD reads as
            # "Strategy Lab, steps 1-7" without changing how any of those
            # groups collapse/expand on their own. See _build_sidebar_nav.
            (None, None, "\u2466 STRATEGY GRAVEYARD", None, METAL_BRIGHT),
            ("starthere_graveyard", "", "Start Here", self._starthere_frames["graveyard"], METAL_BRIGHT),
            ("graveyard", "", "\U0001F480 Strategy Graveyard", self.tab_graveyard, METAL_BRIGHT),

            (None, "SUPERHEADER", "Quant Lab", None, None),
            (None, None, "\u2460 QUANT LAB", None, NEON_CYAN),
            ("starthere_quantlab", "", "Start Here", self._starthere_frames["quantlab"], NEON_CYAN),
            ("quantlab", "", "Quant Lab", self.tab_quantlab, METAL_BRIGHT),

            (None, None, "\u2461 OPTIONS", None, NEON_LIME),
            ("optionsoutlook", "", "Options Outlook (calls & puts)", self.tab_options_outlook, NEON_LIME),

            (None, None, "\u2462 HEDGE FUND MANAGER", None, NEON_MAGENTA),
            ("hedgefund", "", "\U0001F3E6 Hedge Fund Manager", self.tab_hedge_fund, NEON_MAGENTA),

            (None, "SUPERHEADER", "Account", None, None),
            (None, None, "\u2460 ACCOUNT", None, METAL_BRIGHT),
            ("starthere_account", "", "Start Here", self._starthere_frames["account"], METAL_BRIGHT),
            ("account", "", "Account Settings", self.tab_account, METAL_BRIGHT),
            # Notification Settings live inside the Account tab (no separate
            # frame exists) -- this entry is a second pointer at that frame,
            # same pattern as montecarlo -> tab_payout.
            ("notifications", "", "Notification Settings", self.tab_account, METAL_BRIGHT),
            ("apikeys", "", "\U0001F511 API Keys", self.tab_api_keys, METAL_BRIGHT),
            ("datacenter", "", "\U0001F4CA Data Center", self.tab_datacenter, METAL_BRIGHT),
            ("support", "", "\U0001F6DF Support", self.tab_support, METAL_BRIGHT),

            (None, None, "\u2461 EDUCATION", None, METAL_BRIGHT),
            ("education", "", "\U0001F393 Education (course)", self.tab_education, METAL_BRIGHT),
            ("resources", "", "\U0001F393 Resources", self.tab_resources, METAL_BRIGHT),
        ]
        self._tab_frame_by_key = {k: frame for k, _icon, _label, frame, _color in self._nav_items if k}
        self._nav_buttons: dict[str, Label] = {}
        self._build_sidebar_nav()
        self.active_page = "dashboard"

        for label, builder in (
            ("Dashboard", self._build_dashboard_tab),
            ("Forge Strategy", self._build_forge_tab),
            ("Research Director", self._build_research_director_tab),
            ("Your idea", self._build_youridea_tab),
            ("AI Assistant", self._build_ai_assistant_tab),
            ("Manual builder", self._build_manual_tab),
            ("Resources", self._build_resources_tab),
            ("Education", self._build_education_tab),
            ("Speed Run", self._build_speedrun_tab),
            ("Multi-Instrument Speed Run", self._build_speedrun_multi_tab),
            ("Research Loop", self._build_research_loop_tab),
            ("Strategy Configuration", self._build_strategy_config_tab),
            ("Data", self._build_data_tab),
            ("Strategy", self._build_strategy_tab),
            ("Prop rules", self._build_prop_tab),
            ("Risk", self._build_risk_tab),
            ("Run", self._build_run_tab),
            ("Payout Probability", self._build_payout_probability_tab),
            ("Prop-Firm Recommender", self._build_prop_recommender_tab),
            ("Refinement", self._build_refine_tab),
            ("Search Lab", self._build_search_tab),
            ("Multi-Instrument Search Lab", self._build_search_multi_tab),
            ("Walk-forward", self._build_wfo_tab),
            ("CPCV", self._build_cpcv_tab),
            ("PBO", self._build_pbo_tab),
            ("Sensitivity", self._build_sensitivity_tab),
            ("Parameter Stability / Robustness Map", self._build_param_robustness_tab),
            ("Portfolio", self._build_portfolio_tab),
            ("Multi-objective", self._build_multiobj_tab),
            ("Walk-forward GA", self._build_wfga_tab),
            ("Ensemble", self._build_ensemble_tab),
            ("Regime Survival Matrix", self._build_regime_matrix_tab),
            ("Family Diversity", self._build_family_diversity_tab),
            ("Strategy generator", self._build_generate_strategies_tab),
            ("Evolution Lab", self._build_evolution_lab_tab),
            ("Multi-Instrument Evolution Lab", self._build_evolution_multi_tab),
            ("Full Pipeline", self._build_full_pipeline_tab),
            ("Quick Optimize", self._build_quick_optimize_tab),
            ("Optimize (pick your engines)", self._build_optimize_simple_tab),
            ("Validate (pick your checks)", self._build_validate_simple_tab),
            ("Champion Board (promote)", self._build_champion_simple_tab),
            ("Risk Sweep", self._build_risk_sweep_tab),
            ("Compare Strategies", self._build_compare_tab),
            ("Optimize Overview", self._build_optimize_hub_tab),
            ("Validate Overview", self._build_validate_hub_tab),
            ("Forward Test", self._build_forward_test_tab),
            ("Deploy Live", self._build_deploy_live_tab),
            ("Live Market", self._build_live_market_tab),
            ("Research Agent", self._build_research_agent_tab),
            ("Quant Lab", self._build_quant_lab_tab),
            ("Options Outlook", self._build_options_outlook_tab),
            ("Strategy Graveyard", self._build_graveyard_tab),
            ("Final Selection Leaderboard", self._build_leaderboard_tab),
            ("Hedge Fund Manager", self._build_hedge_fund_tab),
            ("Data Center", self._build_data_center_tab),
            ("Account", self._build_account_tab),
            ("API Keys", self._build_api_keys_tab),
            ("Support", self._build_support_tab),
            ("Start Here pages", self._build_start_here_tabs),
            ("Strategy Library", self._build_strategy_library_tab),
            ("Interactive Replay", self._build_replay_tab),
        ):
            self._pump_splash(f"Loading {label}...")
            builder()

        self._show_page("dashboard")

        if self._custom_titlebar_active:
            self._build_resize_grip()

        # Floating Project Chat (same projects/history as the web widget --
        # see app/ui/project_chat_panel.py). Optional: a failure here must
        # never stop the app from starting.
        try:
            from app.ui.project_chat_panel import ProjectChatPanel
            self._project_chat = ProjectChatPanel(self.root)
        except Exception:  # noqa: BLE001
            self._project_chat = None

    def _pump_splash(self, status: str) -> None:
        """Best-effort: updates the boot splash's status text and pumps
        the Tk event loop once, so the splash's glow animation actually
        animates through _build_ui()'s otherwise fully synchronous,
        several-second widget construction instead of freezing on its
        first frame and only reappearing once everything is already
        built. Silently does nothing if there's no splash (e.g. this
        MainWindow wasn't created via launch(), such as in tests)."""
        splash = getattr(self.root, "_t58_splash", None)
        if splash is None:
            return
        try:
            splash.set_status(status)
            self.root.update()
        except Exception:
            pass

    def _group_header_for_key(self, key: str) -> str | None:
        """Which section-divider label a given nav key falls under, e.g.
        "cpcv" -> "\u2463 VALIDATE". Used so switching to a tab always
        auto-expands the group it lives in, even if that group is
        currently collapsed."""
        current_header = None
        for k, icon, lbl_text, _frame, _color in self._nav_items:
            if k is None and icon in ("SUBHEADER", "SUPERHEADER"):
                continue  # a sub-/super-header, not a real (collapsible) group
            if k is None:
                current_header = lbl_text
            elif k == key:
                return current_header
        return None
