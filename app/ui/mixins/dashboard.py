"""Dashboard tab: KPI cards, hero equity, universe and heat-map painters."""
from __future__ import annotations

import math
import random
import threading
from app.data.storage import list_stored_datasets
from app.orchestration import pipeline_guide
from app.reports import run_history, strategy_state
from app.scoring import champion_board
from tkinter import Canvas, Frame, Label, messagebox, ttk
from app.ui.common import _safe_font
from app.ui.widgets.basic import _blend_hex
from app.ui.widgets.cards import GlowCard, _card
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
PANEL_2 = ""
BORDER = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
METAL = ""
METAL_BRIGHT = ""
GREEN = ""
RED = ""
BLUE = ""
AMBER = ""
ACCENT = ""
NEON_CYAN = ""
NEON_MAGENTA = ""
NEON_LIME = ""
NEON_VIOLET = ""
NEON_AMBER = ""
theme.bind(globals())


class DashboardMixin:
    """Dashboard tab: KPI cards, hero equity, universe and heat-map painters."""


    def _draw_neural_background(self, canvas, width, height):
        """A soft, glowing node-link pattern -- purely decorative, in the
        spirit of a knowledge-graph view (Obsidian's graph view, etc).
        Deliberately muted (every color blended heavily toward BG) so it
        reads as ambient texture behind/around the Dashboard's real
        content rather than competing with it. Uses a fixed seed so the
        pattern is stable across resizes and relaunches rather than
        jittering into a new random layout every time.
        """
        canvas.delete("neural_bg")
        if width < 20 or height < 20:
            return
        rng = random.Random(20260830)
        palette = [NEON_CYAN, NEON_VIOLET, NEON_MAGENTA, NEON_LIME, BLUE]

        nodes: list[tuple[float, float, str, bool]] = []
        edges: list[tuple[float, float, float, float, str]] = []
        n_clusters = max(5, int((width * height) / 40000))
        for _ in range(n_clusters):
            cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            color = rng.choice(palette)
            nodes.append((cx, cy, color, True))
            for _ in range(rng.randint(3, 9)):
                angle = rng.uniform(0, 2 * math.pi)
                dist = rng.uniform(26, 100)
                lx, ly = cx + dist * math.cos(angle), cy + dist * math.sin(angle)
                nodes.append((lx, ly, color, False))
                edges.append((cx, cy, lx, ly, color))
                if rng.random() < 0.3:
                    angle2 = rng.uniform(0, 2 * math.pi)
                    dist2 = rng.uniform(14, 40)
                    lx2, ly2 = lx + dist2 * math.cos(angle2), ly + dist2 * math.sin(angle2)
                    nodes.append((lx2, ly2, color, False))
                    edges.append((lx, ly, lx2, ly2, color))
        for _ in range(n_clusters * 5):
            nodes.append((rng.uniform(0, width), rng.uniform(0, height), rng.choice(palette), False))

        for x1, y1, x2, y2, color in edges:
            canvas.create_line(x1, y1, x2, y2, fill=_blend_hex(BG, color, 0.11), width=1, tags="neural_bg")
        for x, y, color, is_hub in nodes:
            if is_hub:
                for dr, t in ((5.5, 0.85), (3.6, 0.6)):
                    canvas.create_oval(
                        x - dr, y - dr, x + dr, y + dr, fill=_blend_hex(BG, color, 1 - t), outline="",
                        tags="neural_bg",
                    )
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=_blend_hex(BG, color, 0.75), outline="", tags="neural_bg")
            else:
                r = rng.uniform(1.1, 2.2)
                canvas.create_oval(
                    x - r, y - r, x + r, y + r, fill=_blend_hex(BG, color, 0.42), outline="", tags="neural_bg",
                )
        canvas.tag_lower("neural_bg")

    def _build_dashboard_scrollable(self, parent) -> Frame:
        """Same scroll-and-wheel contract as _scrollable() (registers into
        the shared self._scroll_canvas_by_tab dispatch table, so the
        existing global mouse-wheel binding just works here too).

        UPGRADE (2026-09, web-parity pass): this used to also paint a
        soft glowing "neural network" node-link backdrop behind the whole
        tab (see the now-unused _draw_neural_background) -- purely
        decorative, in the spirit of a knowledge-graph view. Owen asked
        for the desktop dashboard to match the web app's actual look,
        which has no such background; removed here rather than just left
        unused, since "make the desktop UI look exactly like the web app"
        specifically named this pattern. _draw_neural_background itself is
        left in place (harmless, unreferenced) rather than deleted, in
        case a future tab wants the same effect deliberately.
        """
        MARGIN = 16  # vertical breathing room only; cards supply their own 16px side padding
        outer = Frame(parent, bg=BG)
        outer.pack(fill="both", expand=True)

        canvas = Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, style="T58.Vertical.TScrollbar")
        inner = Frame(canvas, bg=BG)

        window_id = canvas.create_window((0, MARGIN), window=inner, anchor="nw")

        def _sync_scrollregion(_e=None):
            content_h = inner.winfo_reqheight()
            canvas_w = max(canvas.winfo_width(), 200)
            canvas.configure(scrollregion=(0, 0, canvas_w, content_h + MARGIN * 2))

        def _on_canvas_configure(e):
            canvas.itemconfig(window_id, width=max(1, e.width))
            _sync_scrollregion()

        inner.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        if not hasattr(self, "_scroll_canvas_by_tab"):
            self._scroll_canvas_by_tab = {}
        self._scroll_canvas_by_tab[parent] = canvas
        return inner

    def _build_dashboard_tab(self):
        f = self.tab_dashboard
        self._page_header(
            f, "OVERVIEW", "Dashboard",
            "Live stats across every strategy that has been run through the app -- "
            "desktop, mobile web, and Search Lab all feed this automatically.",
        )

        tour_row = Frame(f, bg=BG)
        tour_row.pack(fill="x", padx=16, pady=(0, 6))
        self._button(tour_row, "\U0001F393  Take the 60-second tour", self._replay_tour).pack(side="left")

        # First-run welcome -- shown only for a genuinely fresh install (no
        # market data ever loaded AND no backtest ever run, per
        # app.orchestration.pipeline_guide.should_show_first_run_welcome --
        # the SAME predicate the web app's /dashboard route uses, so the
        # two can never disagree about when someone still needs this).
        # Recomputed fresh every time this tab is built rather than tracked
        # as a "seen it" flag -- it naturally stops appearing the moment
        # either becomes true, with nothing to dismiss-and-forget.
        show_welcome = pipeline_guide.should_show_first_run_welcome(
            has_stored_datasets=bool(list_stored_datasets()),
            has_run_history=bool(run_history.dashboard_data().get("total_runs")),
        )
        if show_welcome:
            welcome_box = self._section(
                f, "👋 New here? Start here",
                "So you never have to guess the order -- three steps to your first real result.",
                emphasize=True,
            )
            Label(
                welcome_box, text=pipeline_guide.first_run_welcome(), bg=PANEL, fg=TEXT,
                font=_safe_font(9), wraplength=880, justify="left",
            ).pack(anchor="w", padx=18, pady=(0, 10))
            welcome_btn_row = Frame(welcome_box, bg=PANEL)
            welcome_btn_row.pack(anchor="w", padx=18, pady=(0, 14))
            self._button(
                welcome_btn_row, "1. GO TO STRATEGY CONFIGURATION", lambda: self._show_page("strategyconfig"),
                primary=True,
            ).pack(side="left")
            self._button(welcome_btn_row, "OR TRY SEARCH LAB", lambda: self._show_page("search")).pack(
                side="left", padx=8,
            )
            self._button(welcome_btn_row, "READ THE FULL WALKTHROUGH", lambda: self._show_page("manual")).pack(
                side="left", padx=8,
            )

        # A dedicated variant of _scrollable() -- no decorative background
        # (see _build_dashboard_scrollable's docstring: the neural-graph
        # backdrop this used to also paint has been removed to match the
        # web app's actual look).
        scroll_frame = self._build_dashboard_scrollable(f)

        # ---- Five questions T58 should always answer first, plus the
        # Champion Board -- replaces the old four flat KPI cards
        # (Strategies tested / Eval pass rate / Best Sharpe / Leader).
        # Uses GREEN (== web --teal, the web app's actual glow accent) via
        # GlowCard rather than the decorative NEON_* set below, since this
        # is the dashboard's headline panel, same role as web's
        # .t58-champion block. ----
        self._dash_five_q_card = GlowCard(scroll_frame, accent=GREEN, height=210)
        self._dash_five_q_card.pack(fill="x", padx=16, pady=(4, 14))
        self._dash_five_q_frame = self._dash_five_q_card.body

        board_wrap = _card(scroll_frame, border=BORDER)
        board_wrap.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        Label(board_wrap, text="CHAMPION BOARD", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=14, pady=(10, 2)
        )
        Label(
            board_wrap,
            text="Every saved strategy's Eval / Payout / OOS standing, and how far it's been explicitly "
                 "promoted -- promotion is manual, never automatic from a raw metric.",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=900, justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 8))

        # Table styles: see _configure_styles.

        board_columns = ("strategy", "status", "eval", "payout", "oos", "stage")
        board_tree_frame = Frame(board_wrap, bg=PANEL)
        board_tree_frame.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        self._dash_board_tree = ttk.Treeview(
            board_tree_frame, columns=board_columns, show="headings", style="T58Board.Treeview", height=6,
        )
        board_headings = {
            "strategy": "Strategy", "status": "Status", "eval": "Eval",
            "payout": "Payout", "oos": "OOS", "stage": "Promotion stage",
        }
        for col, text in board_headings.items():
            self._dash_board_tree.heading(col, text=text, anchor="w")
            self._dash_board_tree.column(col, width=(230 if col == "strategy" else 110), anchor="w")
        self._dash_board_tree.pack(side="left", fill="both", expand=True)
        board_tree_scrollbar = ttk.Scrollbar(
            board_tree_frame, orient="vertical", command=self._dash_board_tree.yview, style="T58.Vertical.TScrollbar",
        )
        board_tree_scrollbar.pack(side="right", fill="y")
        self._dash_board_tree.configure(yscrollcommand=board_tree_scrollbar.set)
        self._bind_isolated_wheel(self._dash_board_tree)

        board_btn_row = Frame(board_wrap, bg=PANEL)
        board_btn_row.pack(anchor="w", padx=14, pady=(0, 12))
        self._button(board_btn_row, "PROMOTE SELECTED", self._promote_selected_board_row, primary=True).pack(side="left")
        self._dash_board_status = Label(board_btn_row, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=700, justify="left")
        self._dash_board_status.pack(side="left", padx=10)

        hero_wrap = _card(scroll_frame, border=BORDER)
        hero_wrap.pack(fill="x", padx=16, pady=(0, 14))
        Label(hero_wrap, text="PORTFOLIO EQUITY — BEST STRATEGY", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=14, pady=(10, 4)
        )
        self._dash_hero_canvas = Canvas(hero_wrap, bg=PANEL, height=200, highlightthickness=0)
        self._dash_hero_canvas.pack(fill="x", padx=14, pady=(0, 14))

        universe_wrap = _card(scroll_frame, border=BORDER)
        universe_wrap.pack(fill="x", padx=16, pady=(0, 14))
        Label(universe_wrap, text="STRATEGY UNIVERSE", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=14, pady=(10, 4)
        )
        self._dash_universe_canvas = Canvas(universe_wrap, bg=PANEL, height=220, highlightthickness=0)
        self._dash_universe_canvas.pack(fill="x", padx=14, pady=(0, 14))

        charts_row = Frame(scroll_frame, bg=BG)
        charts_row.pack(fill="x", padx=16, pady=(0, 14))

        equity_wrap = _card(charts_row, border=BORDER)
        equity_wrap.pack(side="left", fill="both", expand=True, padx=(0, 7))
        Label(equity_wrap, text="EQUITY CURVES — TOP STRATEGIES", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=14, pady=(10, 4)
        )
        self._dash_equity_canvas = Canvas(equity_wrap, bg=PANEL, height=180, highlightthickness=0)
        self._dash_equity_canvas.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        heatmap_wrap = _card(charts_row, border=BORDER)
        heatmap_wrap.pack(side="left", fill="both", expand=True, padx=(7, 0))
        Label(heatmap_wrap, text="WEEKDAY x HOUR PNL (ALL RUNS)", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=14, pady=(10, 4)
        )
        self._dash_heatmap_canvas = Canvas(heatmap_wrap, bg=PANEL, height=180, highlightthickness=0)
        self._dash_heatmap_canvas.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        table_wrap = _card(scroll_frame, border=BORDER)
        table_wrap.pack(fill="both", expand=True, padx=16, pady=(0, 20))
        Label(table_wrap, text="STRATEGY SCORECARD", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=14, pady=(10, 6)
        )


        columns = ("strategy", "instrument", "trades", "net", "win", "sharpe", "dd", "runs", "result")
        tree_frame = Frame(table_wrap, bg=PANEL)
        tree_frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self._dash_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", style="T58.Treeview", height=8)
        headings = {
            "strategy": "Strategy", "instrument": "Instrument", "trades": "Trades", "net": "Net P/L",
            "win": "Win %", "sharpe": "Sharpe", "dd": "Max DD", "runs": "Runs", "result": "Result",
        }
        for col, text in headings.items():
            self._dash_tree.heading(col, text=text, anchor="w")
            self._dash_tree.column(col, width=(230 if col == "strategy" else 100), anchor="w")
        self._dash_tree.pack(side="left", fill="both", expand=True)
        # Scorecard often holds more strategies than the fixed 8-row height
        # shows at once -- without its own scrollbar, anything past row 8
        # was simply unreachable (the page's own scroll just moves the
        # whole card, it doesn't reveal more tree rows). A dedicated
        # vertical scrollbar tied to the treeview's yview fixes that.
        dash_tree_scrollbar = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self._dash_tree.yview, style="T58.Vertical.TScrollbar",
        )
        dash_tree_scrollbar.pack(side="right", fill="y")
        self._dash_tree.configure(yscrollcommand=dash_tree_scrollbar.set)
        self._bind_isolated_wheel(self._dash_tree)

        self._refresh_dashboard()

    def _refresh_dashboard(self):
        """Reloads run_history and repaints every dashboard widget. Called
        when the tab is opened, and right after any run/search completes.

        Thread-safety: Batch Test's own pipeline already learned the hard
        way that Tkinter widgets are only safe to touch from the main
        thread (see its `_refresh_after_run` closure) and routes its own
        calls through `root.after` -- but Search Lab, Full Pipeline,
        Speed Run, and the single-strategy backtest run each call this
        method directly from their own background worker thread once
        they finish, with no such wrapping. Touching Tk widgets off the
        main thread doesn't always crash outright; it can also just
        stall/freeze the whole window ("app not responding") depending on
        timing, which matches exactly what gets reported after a long
        run finishes. Rather than relying on every call site to
        remember to wrap itself, this method now guards itself: if it's
        not running on the main thread, it re-schedules itself onto the
        mainloop and returns immediately without touching any widget.
        """
        if threading.current_thread() is not threading.main_thread():
            try:
                self.root.after(0, self._refresh_dashboard)
            except Exception:
                pass
            return
        try:
            data = run_history.dashboard_data()
        except Exception:
            data = {
                "total_strategies": 0, "total_runs": 0, "pass_rate": 0.0, "best": None,
                "strategies": [], "graph": {"nodes": [], "edges": [], "instruments": []},
                "heatmap": [[0.0] * 24 for _ in range(7)], "equity_series": [],
            }

        for child in self._dash_five_q_frame.winfo_children():
            child.destroy()

        try:
            board_rows = champion_board.list_board()
        except Exception:
            board_rows = []
        try:
            snapshot = champion_board.five_question_snapshot(strategy_state.get_current_strategy(), board_rows)
        except Exception:
            snapshot = None
        self._dash_board_rows_by_key = {(r["strategy_type"], r["filename"]): r for r in board_rows}
        self._paint_five_questions(snapshot)

        for row_id in self._dash_board_tree.get_children():
            self._dash_board_tree.delete(row_id)
        for r in board_rows:
            self._dash_board_tree.insert(
                "", "end", iid=f"{r['strategy_type']}::{r['filename']}",
                values=(
                    r["display_name"], r["status"],
                    f"{r['eval_pct']:.0f}%" if r["eval_pct"] is not None else "--",
                    f"{r['payout_pct']:.0f}%" if r["payout_pct"] is not None else "--",
                    f"{r['oos_pct']:.0f}%" if r["oos_pct"] is not None else "--",
                    r["promotion"]["stage_title"],
                ),
            )
        self._dash_board_status.config(text="")


        self.root.after(30, lambda: self._paint_universe(data["graph"]))
        self.root.after(30, lambda: self._paint_hero_equity(data["equity_series"]))
        self.root.after(30, lambda: self._paint_equity(data["equity_series"]))
        self.root.after(30, lambda: self._paint_heatmap(data["heatmap"]))

        for row_id in self._dash_tree.get_children():
            self._dash_tree.delete(row_id)
        for s in data["strategies"]:
            self._dash_tree.insert("", "end", values=(
                s["strategy_name"], s["instrument"], s["trades"],
                f"${s['net_profit']:,.0f}", f"{s['win_rate']:.1f}%",
                f"{s['sharpe_ratio']:.2f}", f"{s['max_drawdown_pct']:.1f}%",
                s["run_count"], "PASS" if s["single_run_passed"] else "FAIL",
            ))

    def _paint_five_questions(self, snapshot):
        """Populates the dashboard's headline GlowCard with the five
        questions T58 should always answer first: what am I working on,
        where is it in the process, is it working, why, and what should I
        do next. See app.scoring.champion_board.five_question_snapshot."""
        parent = self._dash_five_q_frame
        if snapshot is None:
            Label(
                parent, text="No strategy tested yet -- run your first backtest and T58 will tell you "
                             "what's working, why, and what to do next.",
                bg=PANEL_2, fg=TEXT_DIM, font=_safe_font(10), wraplength=880, justify="left",
            ).pack(anchor="w", padx=16, pady=16)
            return

        title_row = Frame(parent, bg=PANEL_2)
        title_row.pack(fill="x", padx=16, pady=(14, 6))
        Label(title_row, text=snapshot["what"], bg=PANEL_2, fg=TEXT, font=_safe_font(15, "bold")).pack(side="left")
        status_color = {"READY": GREEN, "MARGINAL": AMBER, "NOT READY": RED}.get(snapshot["is_it_working"], METAL)
        Label(
            title_row, text=snapshot["is_it_working"], bg=PANEL_2, fg=status_color, font=_safe_font(9, "bold"),
        ).pack(side="left", padx=(12, 0))

        grid = Frame(parent, bg=PANEL_2)
        grid.pack(fill="x", padx=16, pady=(4, 10))
        for col in range(4):
            grid.grid_columnconfigure(col, weight=1, uniform="fiveq")

        qa = [
            ("WHERE IS IT IN THE PROCESS?", f"{snapshot['where']} — {snapshot['where_pct']:.0f}%"),
            ("IS IT WORKING?", snapshot["is_it_working"]),
            ("WHY?", snapshot["why"]),
            ("WHAT SHOULD I DO NEXT?", snapshot["next_action"]),
        ]
        for i, (label, value) in enumerate(qa):
            cell = Frame(grid, bg=PANEL_2, highlightthickness=0)
            cell.grid(row=0, column=i, sticky="nw", padx=(0 if i == 0 else 10, 0))
            Frame(cell, bg=GREEN, width=2).pack(side="left", fill="y")
            inner = Frame(cell, bg=PANEL_2)
            inner.pack(side="left", fill="both", padx=(8, 0))
            Label(inner, text=label, bg=PANEL_2, fg=TEXT_MUTED, font=_safe_font(7, "bold")).pack(anchor="w")
            Label(
                inner, text=value, bg=PANEL_2, fg=(GREEN if i == 3 else TEXT), font=_safe_font(9, "bold"),
                wraplength=190, justify="left",
            ).pack(anchor="w", pady=(2, 0))

        btn_row = Frame(parent, bg=PANEL_2)
        btn_row.pack(anchor="w", padx=16, pady=(0, 14))
        row = snapshot["row"]
        # Lifecycle-aware primary action (v9.6): this used to always say
        # "OPEN VALIDATE" whenever promotion wasn't available -- including
        # for a strategy that has never been tested at all, where the
        # honest next step is a first real run, not a validation check.
        # Order matters: a live/forward-testing stage wins over any
        # stale verdict, then READY, then the failure/empty cases.
        stage = (row.get("promotion") or {}).get("stage", "")
        verdict = (row.get("verdict") or "").upper()
        if stage in ("forward_testing", "production_ready"):
            self._button(btn_row, "MONITOR", lambda: self._show_page("livemarket")).pack(side="left")
        elif verdict == "READY":
            self._button(btn_row, "CHAMPION", lambda: self._show_page("champion_simple"), primary=True).pack(side="left")
        elif verdict == "NOT READY":
            self._button(btn_row, "OPTIMIZE", lambda: self._show_page("optimize_simple"), primary=True).pack(side="left")
        elif row["promotion"]["can_promote"]:
            self._button(
                btn_row, f"PROMOTE TO {row['promotion']['next_stage_title'].upper()}",
                lambda: self._promote_board_row(row["strategy_type"], row["filename"]), primary=True,
            ).pack(side="left")
        else:
            # v9.7: same logic order as the web dashboard -- an untested
            # strategy goes to the Full Pipeline, never Validate; a
            # tested strategy with checks still missing goes to the
            # pick-your-checks Validate page.
            reqs = row["promotion"].get("requirements") or []
            bt_req = next((r for r in reqs if r.get("key") == "backtested"), None)
            if bt_req is not None and not bt_req.get("met"):
                self._button(btn_row, "\u25b6 RUN FULL PIPELINE", lambda: self._show_page("fullpipeline"), primary=True).pack(side="left")
            elif any(not r.get("met") for r in reqs):
                self._button(btn_row, "OPEN VALIDATE (PICK CHECKS)", lambda: self._show_page("validate_simple"), primary=True).pack(side="left")
            else:
                self._button(btn_row, "\u25b6 RUN FULL PIPELINE", lambda: self._show_page("fullpipeline"), primary=True).pack(side="left")

    def _promote_board_row(self, strategy_type: str, filename: str):
        try:
            ok, message, _new_stage = champion_board.promote_strategy(strategy_type, filename)
        except Exception as exc:  # pragma: no cover - defensive
            ok, message = False, f"Unexpected error: {exc}"
        self._dash_board_status.config(text=message, fg=(GREEN if ok else AMBER))
        self._refresh_dashboard()

    def _promote_selected_board_row(self):
        selection = self._dash_board_tree.selection()
        if not selection:
            messagebox.showinfo("Promote", "Select a strategy in the Champion Board first.")
            return
        strategy_type, filename = selection[0].split("::", 1)
        self._promote_board_row(strategy_type, filename)

    def _paint_universe(self, graph):
        c = self._dash_universe_canvas
        c.delete("all")
        w = max(c.winfo_width(), 400)
        h = max(c.winfo_height(), 200)
        nodes = graph.get("nodes", [])
        if not nodes:
            c.create_text(w / 2, h / 2, text="No strategies run yet.", fill=TEXT_DIM, font=_safe_font(9))
            return

        instruments = graph.get("instruments", [])
        n_clusters = max(len(instruments), 1)
        palette = [GREEN, ACCENT, RED, AMBER, BLUE]
        import math
        centers = []
        for i in range(n_clusters):
            angle = (i / n_clusters) * 2 * math.pi
            r = min(w, h) * 0.28 if n_clusters > 1 else 0
            centers.append((w / 2 + r * math.cos(angle), h / 2 + r * math.sin(angle)))

        pos = []
        for i, n in enumerate(nodes):
            cx, cy = centers[n["cluster"]] if n["cluster"] < len(centers) else (w / 2, h / 2)
            angle = (i / max(len(nodes), 1)) * 2 * math.pi
            pos.append([cx + 24 * math.cos(angle), cy + 24 * math.sin(angle)])

        edges = graph.get("edges", [])
        for _ in range(50):
            forces = [[0.0, 0.0] for _ in nodes]
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i == j:
                        continue
                    dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
                    d2 = max(dx * dx + dy * dy, 20)
                    forces[i][0] += dx / d2 * 300
                    forces[i][1] += dy / d2 * 300
                cx, cy = centers[nodes[i]["cluster"]] if nodes[i]["cluster"] < len(centers) else (w / 2, h / 2)
                forces[i][0] += (cx - pos[i][0]) * 0.02
                forces[i][1] += (cy - pos[i][1]) * 0.02
            for e in edges:
                a, b = e["source"], e["target"]
                dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
                pull = e["weight"] * 0.02
                forces[a][0] += dx * pull
                forces[a][1] += dy * pull
                forces[b][0] -= dx * pull
                forces[b][1] -= dy * pull
            for i in range(len(nodes)):
                pos[i][0] = min(w - 16, max(16, pos[i][0] + forces[i][0] * 0.3))
                pos[i][1] = min(h - 16, max(16, pos[i][1] + forces[i][1] * 0.3))

        for e in edges:
            a, b = pos[e["source"]], pos[e["target"]]
            width = 0.5 + e["weight"] * 2
            c.create_line(a[0], a[1], b[0], b[1], fill=self._blend(ACCENT, PANEL, 0.35), width=width)

        for i, n in enumerate(nodes):
            color = palette[n["cluster"] % len(palette)]
            r = 5 + min(max(n["sharpe"], 0), 3) * 2
            ring = GREEN if n["passed"] else RED
            self._glow_dot(c, pos[i][0], pos[i][1], r, color, ring_color=ring)

        legend_x = 10
        for i, name in enumerate(instruments):
            color = palette[i % len(palette)]
            c.create_oval(legend_x, h - 16, legend_x + 8, h - 8, fill=color, outline="")
            c.create_text(legend_x + 14, h - 12, text=name, fill=TEXT_MUTED, font=_safe_font(7), anchor="w")
            legend_x += 14 + len(name) * 6 + 14

    def _catmull_rom_points(self, xs: list[float], ys: list[float], samples_per_seg: int = 10) -> list[float]:
        """Densifies a polyline into a smooth Catmull-Rom spline through
        the same original points -- genuinely curved between data points
        rather than relying on Tkinter's own `smooth=True` (a quadratic
        bezier approximation that still visibly kinks with few, unevenly-
        spaced points, which equity curves from real trade data usually
        are). Falls back to the raw points unchanged for fewer than 3
        points, where a spline isn't meaningful anyway."""
        n = len(xs)
        if n < 3:
            out = []
            for x, y in zip(xs, ys):
                out.extend([x, y])
            return out

        def pt(i):
            i = max(0, min(n - 1, i))
            return xs[i], ys[i]

        out = []
        for i in range(n - 1):
            p0, p1, p2, p3 = pt(i - 1), pt(i), pt(i + 1), pt(i + 2)
            for s in range(samples_per_seg):
                t = s / samples_per_seg
                t2, t3 = t * t, t * t * t
                x = 0.5 * (
                    (2 * p1[0]) + (-p0[0] + p2[0]) * t
                    + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    (2 * p1[1]) + (-p0[1] + p2[1]) * t
                    + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                out.extend([x, y])
        out.extend([xs[-1], ys[-1]])
        return out

    def _paint_hero_equity(self, series):
        """The dashboard's featured chart: the single best strategy's
        equity curve, drawn full-width and taller than the small
        multi-strategy comparison chart below, with a genuinely smooth
        Catmull-Rom curve, a soft glow, and a gradient fill under the
        line -- the "make it glow and look smooth" upgrade on top of the
        existing (still useful, more compact) multi-strategy comparison."""
        c = self._dash_hero_canvas
        c.delete("all")
        w = max(c.winfo_width(), 400)
        h = max(c.winfo_height(), 160)
        pad_x, pad_top, pad_bottom = 20, 16, 28
        if not series:
            c.create_text(w / 2, h / 2, text="No completed runs yet -- run a backtest to populate this chart.",
                           fill=TEXT_DIM, font=_safe_font(9))
            return

        # Feature whichever series is passing (or, failing that, the
        # first available) -- this chart tells one clear story rather
        # than overlaying everything the smaller comparison chart below
        # already shows.
        best_series = next((s for s in series if s.get("passed")), series[0])
        values = best_series["values"]
        if len(values) < 2:
            c.create_text(w / 2, h / 2, text="Not enough closed trades yet for an equity curve.",
                           fill=TEXT_DIM, font=_safe_font(9))
            return

        lo, hi = min(values), max(values)
        rng = (hi - lo) or 1
        n = len(values)
        xs = [pad_x + (i / (n - 1)) * (w - 2 * pad_x) for i in range(n)]
        ys = [h - pad_bottom - ((v - lo) / rng) * (h - pad_top - pad_bottom) for v in values]

        # Gridlines first, underneath everything.
        for frac in (0.0, 0.5, 1.0):
            gy = pad_top + frac * (h - pad_top - pad_bottom)
            c.create_line(pad_x, gy, w - pad_x, gy, fill=BORDER, dash=(2, 3))
        c.create_text(pad_x, pad_top - 8, text=f"${hi:,.0f}", fill=TEXT_DIM, font=_safe_font(7), anchor="sw")
        c.create_text(pad_x, h - pad_bottom + 16, text=f"${lo:,.0f}", fill=TEXT_DIM, font=_safe_font(7), anchor="nw")

        smooth_points = self._catmull_rom_points(xs, ys, samples_per_seg=12)
        color = GREEN if best_series.get("passed") else RED

        # Gradient area fill under the curve -- several stacked bands
        # fading from a visible tint near the line down to nearly nothing
        # at the baseline, faking a vertical gradient the same way every
        # other glow effect in this app fakes soft alpha falloff (Tkinter
        # has no real alpha compositing).
        baseline_y = h - pad_bottom
        n_bands = 10
        for band in range(n_bands, 0, -1):
            frac = band / n_bands
            band_alpha = 0.22 * frac
            poly = []
            for i in range(0, len(smooth_points), 2):
                poly.extend([smooth_points[i], smooth_points[i + 1]])
            top_of_band_scale = frac
            scaled = []
            for i in range(0, len(poly), 2):
                x, y = poly[i], poly[i + 1]
                y_scaled = baseline_y - (baseline_y - y) * top_of_band_scale
                scaled.extend([x, y_scaled])
            polygon_points = scaled + [smooth_points[-2], baseline_y, smooth_points[0], baseline_y]
            c.create_polygon(polygon_points, fill=_blend_hex(PANEL, color, band_alpha), outline="")

        self._glow_line(c, smooth_points, color, width=2.2)

        # A glowing dot on the final (most recent) point -- draws the eye
        # to "where this strategy stands right now."
        self._glow_dot(c, xs[-1], ys[-1], 4.5, color, ring_color=METAL_BRIGHT)

        label = f"{best_series['name']} — {'PASSING' if best_series.get('passed') else 'FAILING'}"
        c.create_oval(w - pad_x - 90, pad_top - 4, w - pad_x - 84, pad_top + 2, fill=color, outline="")
        c.create_text(w - pad_x - 78, pad_top - 1, text=label, fill=TEXT_MUTED, font=_safe_font(7, "bold"), anchor="w")

    def _paint_equity(self, series):
        c = self._dash_equity_canvas
        c.delete("all")
        w = max(c.winfo_width(), 300)
        h = max(c.winfo_height(), 140)
        pad = 16
        if not series:
            c.create_text(w / 2, h / 2, text="No completed runs yet.", fill=TEXT_DIM, font=_safe_font(9))
            return

        lo = min(v for s in series for v in s["values"]) if series else 0
        hi = max(v for s in series for v in s["values"]) if series else 1
        rng = (hi - lo) or 1
        max_len = max(len(s["values"]) for s in series)

        # Each strategy gets its own distinct hue from this app's
        # decorative neon palette, so multiple lines on the same chart
        # stay visually distinguishable. Coloring every line strictly by
        # pass/fail (GREEN/RED) meant every failing strategy rendered as
        # the exact same solid red -- indistinguishable from one another
        # whenever most (or, for a fresh batch, all) of the ranked
        # strategies happened to be failing, which is the common case.
        # Pass/fail is still shown -- just via the legend's status text/
        # color instead of collapsing every line to one of two colors.
        palette = [NEON_CYAN, NEON_AMBER, NEON_VIOLET, NEON_LIME, NEON_MAGENTA, BLUE]

        c.create_line(pad, h - pad, w - pad, h - pad, fill=BORDER)
        for idx, s in enumerate(series):
            color = palette[idx % len(palette)]
            n = len(s["values"])
            if n < 2:
                continue
            points = []
            for i, v in enumerate(s["values"]):
                x = pad + (i / max(max_len - 1, 1)) * (w - 2 * pad)
                y = h - pad - ((v - lo) / rng) * (h - 2 * pad)
                points.extend([x, y])
            self._glow_line(c, points, color, width=1.6)

        legend_y = 6
        for idx, s in enumerate(series):
            color = palette[idx % len(palette)]
            status_color = GREEN if s["passed"] else RED
            status_text = "PASS" if s["passed"] else "FAIL"
            c.create_oval(6, legend_y, 12, legend_y + 6, fill=color, outline="")
            c.create_text(18, legend_y + 3, text=s["name"], fill=TEXT_MUTED, font=_safe_font(7), anchor="w")
            c.create_text(w - 8, legend_y + 3, text=status_text, fill=status_color, font=_safe_font(7, "bold"), anchor="e")
            legend_y += 13

    def _paint_heatmap(self, grid):
        c = self._dash_heatmap_canvas
        c.delete("all")
        w = max(c.winfo_width(), 300)
        h = max(c.winfo_height(), 140)
        left_pad = 26
        cell_w = (w - left_pad - 4) / 24
        cell_h = (h - 4) / 7
        max_abs = max((abs(v) for row in grid for v in row), default=0.0001) or 0.0001
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for d, row in enumerate(grid):
            c.create_text(4, 2 + d * cell_h + cell_h / 2, text=days[d], fill=TEXT_DIM, font=_safe_font(6), anchor="w")
            for hr, v in enumerate(row):
                intensity = min(abs(v) / max_abs, 1)
                base = GREEN if v >= 0 else RED
                color = self._blend(base, PANEL, 1 - (0.15 + intensity * 0.8))
                x0 = left_pad + hr * cell_w
                y0 = 2 + d * cell_h
                c.create_rectangle(x0, y0, x0 + cell_w - 1, y0 + cell_h - 1, fill=color, outline="")
