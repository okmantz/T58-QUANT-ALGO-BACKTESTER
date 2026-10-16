"""Strategy configuration tab and strategy builder."""
from __future__ import annotations

import json
from app.strategy.base import StrategyError
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy
from app.ui.condition_builder import ConditionList
from pathlib import Path
from tkinter import Canvas, EXTENDED, Frame, Label, Listbox, ttk
from app.ui.common import STATUS_KEY_TO_LABEL, STATUS_LABELS_ORDERED, _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
PANEL = ""
PANEL_3 = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_DIM = ""
METAL = ""
METAL_BRIGHT = ""
GREEN = ""
RED = ""
theme.bind(globals())


class StrategyConfigMixin:
    """Strategy configuration tab and strategy builder."""


    # -----------------------------------------------------------------------
    # Tab 2 — Strategy
    # -----------------------------------------------------------------------

    def _build_strategy_config_tab(self):
        f = self._scrollable(self.tab_strategyconfig)

        self._page_header(
            f,
            "TEST / Strategy Configuration",
            "Strategy Configuration",
            "Choose your strategy source -- Manual, Python, PineScript, or MQL5 -- manage "
            "the strategy library, and stage strategies for batch testing. Manual mode is "
            "built out visually on the Strategy Builder tab (CREATE section); Python / "
            "PineScript / MQL5 files are browsed in below.",
        )

        section = self._section(
            f,
            "Strategy source",
            "Select one of the supported strategy formats.",
        )

        modes = Frame(section, bg=PANEL)
        modes.pack(anchor="w", padx=18, pady=(2, 8))

        for val, text in [
            ("manual", "MANUAL"),
            ("python", "PYTHON"),
            ("pinescript", "PINESCRIPT"),
            ("mql5", "MQL5"),
        ]:
            self._button(
                modes,
                text,
                lambda v=val: self._set_strategy_mode(v),
            ).pack(side="left", padx=(0, 7))

        self.strategy_mode_label = Label(
            section,
            text="SELECTED  •  MANUAL STRATEGY BUILDER",
            bg=PANEL,
            fg=GREEN,
            font=_safe_font(9, "bold"),
        )
        self.strategy_mode_label.pack(anchor="w", padx=18, pady=(4, 9))

        self._button(
            section,
            "BROWSE STRATEGY FILE",
            self._browse_strategy_file,
        ).pack(anchor="w", padx=18, pady=(0, 5))

        self.strategy_file_status = Label(
            section,
            text="Only needed for Python / PineScript / MQL5 modes.",
            bg=PANEL,
            fg=TEXT_DIM,
            font=_safe_font(8),
        )
        self.strategy_file_status.pack(anchor="w", padx=18, pady=(0, 14))

        # ------------------------------------------------------------
        # Strategy library — save/load strategies inside the app itself
        # ------------------------------------------------------------
        library_section = self._section(
            f,
            "Strategy library",
            "Strategies live here inside the app instead of only on your computer. "
            "Importing a file above automatically saves a copy into the library; "
            "you can also load, rename, or delete a previously saved one below. "
            "Ctrl/Cmd-click or Shift-click to select more than one for bulk delete/export. "
            "Note: a packaged .exe's library lives next to the .exe, not in your "
            "git repo — use EXPORT LIBRARY AS ZIP below and unzip it into the "
            "repo's strategies/ folder to keep them in sync.",
        )

        self.strategy_search = LabeledEntry(library_section, "Search (name / market / tags)", "")
        self.strategy_search.entry.bind("<KeyRelease>", lambda _e: self._refresh_strategy_library())

        filter_row = Frame(library_section, bg=PANEL)
        filter_row.pack(fill="x", padx=0, pady=(0, 2))

        self.strategy_filter_market = LabeledCombo(filter_row, "Browse market", ["All markets"], default="All markets")
        self.strategy_filter_market.combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_strategy_library())

        self.strategy_filter_tag = LabeledCombo(filter_row, "Browse tag", ["All tags"], default="All tags")
        self.strategy_filter_tag.combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_strategy_library())

        self.strategy_filter_status = LabeledCombo(
            filter_row, "Browse status", ["All statuses", *STATUS_LABELS_ORDERED], default="All statuses"
        )
        self.strategy_filter_status.combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_strategy_library())

        lib_list_frame = Frame(library_section, bg=PANEL)
        lib_list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.strategy_library_listbox = Listbox(
            lib_list_frame,
            height=6,
            selectmode=EXTENDED,
            exportselection=False,
            bg=PANEL_3,
            fg=TEXT,
            selectbackground=BORDER_LIGHT,
            selectforeground=METAL_BRIGHT,
            activestyle="none",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.strategy_library_listbox.pack(side="left", fill="both", expand=True)

        lib_scrollbar = ttk.Scrollbar(
            lib_list_frame,
            orient="vertical",
            command=self.strategy_library_listbox.yview,
            style="T58.Vertical.TScrollbar",
        )
        lib_scrollbar.pack(side="right", fill="y")
        self.strategy_library_listbox.config(yscrollcommand=lib_scrollbar.set)
        self._bind_isolated_wheel(self.strategy_library_listbox)
        # Double-click now opens the same kind of stats+code detail popup
        # the Evolution Lab leaderboard already has (VIEW DETAILS there),
        # instead of immediately loading the strategy into the active
        # slot -- that's the "same capability the leaderboard has" for
        # every saved strategy Owen asked for. LOAD SELECTED (button
        # below) still does the load.
        self.strategy_library_listbox.bind(
            "<Double-Button-1>", lambda _e: self._view_selected_strategy_detail()
        )
        self.strategy_library_listbox.bind(
            "<<ListboxSelect>>", lambda _e: self._on_library_selection_changed()
        )

        lib_btn_row = Frame(library_section, bg=PANEL)
        lib_btn_row.pack(anchor="w", padx=18, pady=(0, 6))

        self._button(
            lib_btn_row, "LOAD SELECTED", self._load_selected_library_strategy, primary=True
        ).pack(side="left")
        self._button(
            lib_btn_row, "VIEW DETAILS", self._view_selected_strategy_detail
        ).pack(side="left", padx=8)
        self._button(
            lib_btn_row, "RENAME SELECTED", self._rename_selected_library_strategy
        ).pack(side="left", padx=8)
        self._button(
            lib_btn_row, "DELETE SELECTED", self._delete_selected_library_strategy
        ).pack(side="left")
        self._button(
            lib_btn_row, "REFRESH LIBRARY", self._refresh_strategy_library
        ).pack(side="left", padx=8)

        lib_btn_row_3 = Frame(library_section, bg=PANEL)
        lib_btn_row_3.pack(anchor="w", padx=18, pady=(0, 6))

        self._button(
            lib_btn_row_3, "VIEW CODE / CONFIG", self._view_selected_strategy_code, primary=True
        ).pack(side="left")
        self._button(
            lib_btn_row_3, "OPTIMIZE SELECTED", self._optimize_selected_library_strategies
        ).pack(side="left", padx=8)

        self.lib_optimize_adaptive_risk = LabeledCheckbox(
            library_section,
            "OPTIMIZE SELECTED: enable adaptive, limit-aware position sizing on the search "
            "(same engine as 05 Run & Report's Adaptive Risk section)",
            False,
        )

        # 2026-09-17 fix, point (5): mirrors the web app's "Reserve a
        # holdout slice" checkbox on the Quick Optimize page -- optional,
        # off by default (byte-identical behavior to before this existed),
        # carves off the trailing 20% of whatever data OPTIMIZE SELECTED
        # loads and reports the winner's performance on it separately. See
        # QuickOptimizeConfig.reserve_holdout's own docstring for exactly
        # what this does and does not check.
        self.lib_optimize_reserve_holdout = LabeledCheckbox(
            library_section,
            "OPTIMIZE SELECTED: reserve a holdout slice (lighter, optional check -- still not a "
            "substitute for Full Pipeline; every result stays labeled NOT OOS VALIDATED either way)",
            True,
        )

        lib_btn_row_4 = Frame(library_section, bg=PANEL)
        lib_btn_row_4.pack(anchor="w", padx=18, pady=(0, 6))

        self._button(
            lib_btn_row_4, "VIEW DNA", self._view_selected_strategy_dna
        ).pack(side="left")
        self._button(
            lib_btn_row_4, "FIND COMMON PATTERNS (ALL)", self._find_common_dna_patterns_library, primary=True
        ).pack(side="left", padx=8)

        Label(
            library_section,
            text="VIEW DNA breaks the selected strategy down into ENTRY / EXIT / RISK genes (market "
            "structure, liquidity, momentum, volatility, time filter / fixed RR, ATR, structure, time "
            "exit / fixed %, adaptive, daily governor) -- see app.strategy.dna. FIND COMMON PATTERNS "
            "(ALL) compares every saved strategy's DNA against its own recorded last-run result and "
            "reports which gene combinations are disproportionately common among the ones that passed "
            "their evaluation versus the ones that didn't -- the 'what do the winners actually have in "
            "common' answer, not just which single strategy scored highest.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        Label(
            library_section,
            text="VIEW CODE / CONFIG shows the saved source for a selected Python/PineScript/MQL5 "
            "strategy, or the built config for whatever's currently set up in Manual mode. OPTIMIZE "
            "SELECTED runs the same walk-forward-aware GA Full Pipeline uses (Step 2) against just the "
            "highlighted strategy(ies) above -- quicker than a full Full Pipeline run, and saves each "
            "winning configuration into the library as a new '<name>_optimized' file (status: draft) "
            "without touching the original. For batch-testing several strategies at once, check them "
            "off in the Batch selection panel below instead of using this list's own selection.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        # ------------------------------------------------------------
        # Batch selection -- checkbox style (mirrors the web app's
        # section 6 "batch checkbox list"), replacing the old two-step
        # "highlight rows -> ADD SELECTED TO BATCH QUEUE -> manage a
        # separate queue listbox" flow. Every saved strategy across all
        # four types shows up here as its own checkbox; whichever boxes
        # are checked at the moment you click RUN BATCH TEST (CHECKED) or
        # RUN FULL PIPELINE (BATCH) (CHECKED) is exactly what runs -- no
        # separate staging step, and nothing to forget to add or remove.
        # ------------------------------------------------------------
        batch_section = self._section(
            f, "Batch selection",
            "Check off any strategies below (any mix of Python / PineScript / MQL5 / Manual) "
            "then click RUN BATCH TEST (CHECKED) or RUN FULL PIPELINE (BATCH) (CHECKED) -- each "
            "runs the same backtest/prop-sim/Monte Carlo pipeline as Run & Report or Full Pipeline "
            "using whatever Prop Rules (03) and Risk (04) are set at the moment you click, one saved "
            "report per strategy, all showing up on the Dashboard afterward. Checked state persists "
            "if you filter/search the library list above or switch strategy-source tabs.",
        )
        batch_toolbar = Frame(batch_section, bg=PANEL)
        batch_toolbar.pack(fill="x", padx=18, pady=(2, 4))
        self._button(batch_toolbar, "SELECT ALL", self._batch_checklist_select_all).pack(side="left")
        self._button(batch_toolbar, "CLEAR ALL", self._batch_checklist_clear_all).pack(side="left", padx=(6, 0))
        self._button(batch_toolbar, "REFRESH LIST", self._refresh_batch_checklist).pack(side="left", padx=(6, 0))

        batch_list_outer = Frame(batch_section, bg=PANEL_3, highlightthickness=1, highlightbackground=BORDER)
        batch_list_outer.pack(fill="both", expand=True, padx=18, pady=(4, 0))
        batch_canvas = Canvas(batch_list_outer, bg=PANEL_3, highlightthickness=0, height=220)
        batch_scroll = ttk.Scrollbar(
            batch_list_outer, orient="vertical", command=batch_canvas.yview, style="T58.Vertical.TScrollbar",
        )
        batch_canvas.configure(yscrollcommand=batch_scroll.set)
        batch_canvas.pack(side="left", fill="both", expand=True)
        batch_scroll.pack(side="right", fill="y")

        self._batch_checklist_inner = Frame(batch_canvas, bg=PANEL_3)
        batch_window_id = batch_canvas.create_window((0, 0), window=self._batch_checklist_inner, anchor="nw")
        self._batch_checklist_inner.bind(
            "<Configure>", lambda _e: batch_canvas.configure(scrollregion=batch_canvas.bbox("all")),
        )
        batch_canvas.bind(
            "<Configure>", lambda e: batch_canvas.itemconfig(batch_window_id, width=e.width),
        )
        self._bind_isolated_wheel(batch_canvas)
        self._bind_isolated_wheel(self._batch_checklist_inner)

        batch_btn_row = Frame(batch_section, bg=PANEL)
        batch_btn_row.pack(anchor="w", padx=18, pady=(8, 4))
        self._button(
            batch_btn_row, "RUN BATCH TEST (CHECKED)", self._run_batch_checked_clicked, primary=True
        ).pack(side="left")
        self._button(
            batch_btn_row, "RUN FULL PIPELINE (BATCH) (CHECKED)", self._run_full_pipeline_checked_clicked, primary=True
        ).pack(side="left", padx=8)

        Label(
            batch_section, text="RUN BATCH TEST (CHECKED) runs every checked strategy through the plain "
            "backtest -> prop-sim -> Monte Carlo pipeline (fast). RUN FULL PIPELINE (BATCH) (CHECKED) runs "
            "every checked strategy through the FULL 15 Full Pipeline instead (baseline -> "
            "walk-forward-aware GA search -> re-validated Monte Carlo -> out-of-sample check -> "
            "holdout check -> ICIR gate -> verdict) -- slower per strategy since it includes the "
            "GA search, but this is how you batch-test dozens of checked strategies through "
            "the full validation ladder without opening and running Full Pipeline on each one by "
            "hand. Either way: one saved report per strategy, using whatever settings are "
            "currently configured on 03 Prop Rules / 04 Risk (and, for the Full Pipeline batch, "
            "15 Full Pipeline's own GA/AI Assist settings) at the moment you click.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(4, 0))

        self.batch_checklist_status = Label(
            batch_section, text="No saved strategies yet.", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        )
        self.batch_checklist_status.pack(anchor="w", padx=18, pady=(6, 10))

        lib_btn_row_2 = Frame(library_section, bg=PANEL)
        lib_btn_row_2.pack(anchor="w", padx=18, pady=(0, 10))

        self._button(
            lib_btn_row_2, "OPEN LIBRARY FOLDER", self._open_strategy_library_folder
        ).pack(side="left")
        self._button(
            lib_btn_row_2, "EXPORT SELECTED AS ZIP", self._export_selected_library_strategies
        ).pack(side="left", padx=8)
        self._button(
            lib_btn_row_2, "EXPORT LIBRARY AS ZIP", self._export_strategy_library
        ).pack(side="left")

        Label(
            library_section,
            text="Tip: OPEN LIBRARY FOLDER is only for copying/dropping files in Explorer "
            "(or Finder). To actually use a saved strategy, select it in the list above and "
            "click LOAD SELECTED (or double-click it) -- double-clicking a .py/.pine/.mq5 "
            "file itself in Explorer won't work, since Windows has no default app registered "
            "for those extensions and will say so.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        # ---- Metadata for the selected saved strategy -------------
        meta_frame = Frame(library_section, bg=PANEL)
        meta_frame.pack(fill="x", padx=0, pady=(0, 4))

        self.strategy_meta_description = LabeledEntry(meta_frame, "Description", "")
        self.strategy_meta_market = LabeledEntry(meta_frame, "Market / timeframe", "")
        self.strategy_meta_tags = LabeledEntry(meta_frame, "Tags (comma-separated)", "")
        self.strategy_meta_status = LabeledCombo(
            meta_frame, "Status", STATUS_LABELS_ORDERED, default=STATUS_KEY_TO_LABEL["draft"]
        )

        self._button(
            meta_frame, "SAVE INFO TO SELECTED", self._save_selected_library_metadata
        ).pack(anchor="w", padx=18, pady=(2, 4))

        self.strategy_meta_last_run = Label(
            library_section,
            text="",
            bg=PANEL,
            fg=TEXT_DIM,
            font=_safe_font(8),
            justify="left",
        )
        self.strategy_meta_last_run.pack(anchor="w", padx=18, pady=(0, 8))

        self.strategy_library_status = Label(
            library_section,
            text="",
            bg=PANEL,
            fg=TEXT_DIM,
            font=_safe_font(8),
        )
        self.strategy_library_status.pack(anchor="w", padx=18, pady=(0, 14))

        self._strategy_library_items: list = []
        self._refresh_strategy_library()

    def _build_strategy_tab(self):
        f = self._scrollable(self.tab_strategy)

        self._page_header(
            f,
            "CREATE / Strategy Builder",
            "Strategy Builder",
            "Build a complete strategy visually -- no code required. Define indicators and "
            "entry/exit rules below; when you're ready to test it, head to Strategy "
            "Configuration (Step 1, TEST section) and make sure MANUAL is the selected "
            "strategy source there. Bringing your own Python / PineScript / MQL5 file? "
            "Skip this tab entirely -- that's all handled on Strategy Configuration too.",
        )

        # ------------------------------------------------------------
        # 24.1  Strategy information
        # ------------------------------------------------------------
        info = self._section(
            f,
            "Strategy information",
            "Descriptive info + the market this strategy is meant to trade. None of this "
            "is required to run a backtest — it's just kept with the strategy for your own records.",
        )
        self.s_name = LabeledEntry(info, "Strategy name", "My Strategy")
        self.s_description = LabeledEntry(info, "Description", "")
        self.s_author = LabeledEntry(info, "Author", "")
        self.s_version = LabeledEntry(info, "Version", "1.0")
        self.s_instrument = LabeledEntry(info, "Instrument", "")
        self.s_timeframe = LabeledEntry(info, "Timeframe", "5m")
        self.s_session_start = LabeledEntry(info, "Session start (HH:MM, 24h)", "08:30")
        self.s_session_end = LabeledEntry(info, "Session end (HH:MM, 24h)", "15:00")
        Label(
            info, text="Session Start/End is also used automatically by any Session High/Low "
                       "or Opening Range condition below.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 8))
        self.s_direction = LabeledCombo(info, "Trade direction", ["Both", "Long", "Short"], "Both")

        # ------------------------------------------------------------
        # 24.2  Entry conditions
        # ------------------------------------------------------------
        entry_section = self._section(
            f,
            "Entry conditions",
            "Build one or more rules using AND / OR. A trade only enters once every "
            "condition in the chain evaluates true. Example: Close > EMA(50) AND RSI(14) > 55.",
            emphasize=True,
        )

        Label(entry_section, text="LONG ENTRY", bg=PANEL, fg=GREEN, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(4, 2)
        )
        long_entry_container = Frame(entry_section, bg=PANEL)
        long_entry_container.pack(fill="x", padx=18, pady=(0, 4))
        self.long_entry_conditions = ConditionList(long_entry_container, get_session=self._current_session)
        self._condition_list_buttons(entry_section, self.long_entry_conditions, (0, 14))

        Label(entry_section, text="SHORT ENTRY", bg=PANEL, fg=RED, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(4, 2)
        )
        short_entry_container = Frame(entry_section, bg=PANEL)
        short_entry_container.pack(fill="x", padx=18, pady=(0, 4))
        self.short_entry_conditions = ConditionList(short_entry_container, get_session=self._current_session)
        self._condition_list_buttons(entry_section, self.short_entry_conditions, (0, 14))

        # ------------------------------------------------------------
        # 24.3  Exit conditions / risk management
        # ------------------------------------------------------------
        exit_section = self._section(
            f,
            "Exit conditions",
            "Every field here is optional — leave anything you don't want at its default "
            "(None / unchecked / blank) and it's simply ignored.",
        )

        Label(exit_section, text="STOP LOSS", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(4, 2)
        )
        self.stop_type = LabeledCombo(exit_section, "Stop loss type", ["None", "Fixed (pips)", "ATR Multiple"], "Fixed (pips)")
        self.stop_value = LabeledEntry(exit_section, "Stop loss value (pips, or ATR multiple e.g. 1.0)", 20)
        self.stop_atr_period = LabeledEntry(exit_section, "Stop loss ATR period", 14)

        Label(exit_section, text="TAKE PROFIT", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        self.target_type = LabeledCombo(exit_section, "Take profit type", ["None", "Fixed (pips)", "ATR Multiple"], "Fixed (pips)")
        self.target_value = LabeledEntry(exit_section, "Take profit value (pips, or ATR multiple e.g. 2.0)", 40)
        self.target_atr_period = LabeledEntry(exit_section, "Take profit ATR period", 14)

        Label(exit_section, text="TRAILING STOP  (ATR-based)", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        self.trailing_enabled = LabeledCheckbox(exit_section, "Enable trailing stop", False)
        self.trailing_value = LabeledEntry(exit_section, "Trailing distance (ATR multiple, e.g. 1.5)", 1.5)
        self.trailing_atr_period = LabeledEntry(exit_section, "Trailing stop ATR period", 14)

        Label(exit_section, text="BREAK EVEN", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        self.breakeven_enabled = LabeledCheckbox(exit_section, "Move stop to break-even once in profit", False)
        self.breakeven_trigger = LabeledEntry(exit_section, "Trigger, in multiples of initial risk (e.g. 1.0 = +1R)", 1.0)

        Label(exit_section, text="TIME-BASED EXIT", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        self.time_exit_enabled = LabeledCheckbox(exit_section, "Flatten any open trade at a fixed time", False)
        self.time_exit_time = LabeledEntry(exit_section, "Exit time (HH:MM, 24h)", "15:55")

        Label(exit_section, text="OTHER EXIT RULES", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        self.max_bars = LabeledEntry(exit_section, "Maximum bars in trade (blank = unlimited)", "")
        self.opposite_signal_exit = LabeledCheckbox(
            exit_section, "Opposite Signal Exit — an opposite entry signal closes/reverses an open trade", True,
        )

        Label(exit_section, text="INDICATOR EXIT", bg=PANEL, fg=METAL, font=_safe_font(9, "bold")).pack(
            anchor="w", padx=18, pady=(10, 2)
        )
        Label(
            exit_section, text="Optional extra exit rule(s), built the same way as entry conditions above.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        ).pack(anchor="w", padx=18, pady=(0, 4))

        Label(exit_section, text="Long exit", bg=PANEL, fg=GREEN, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=18, pady=(4, 2)
        )
        long_exit_container = Frame(exit_section, bg=PANEL)
        long_exit_container.pack(fill="x", padx=18, pady=(0, 4))
        self.long_exit_conditions = ConditionList(long_exit_container, get_session=self._current_session)
        self._condition_list_buttons(exit_section, self.long_exit_conditions, (0, 10))

        Label(exit_section, text="Short exit", bg=PANEL, fg=RED, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=18, pady=(4, 2)
        )
        short_exit_container = Frame(exit_section, bg=PANEL)
        short_exit_container.pack(fill="x", padx=18, pady=(0, 4))
        self.short_exit_conditions = ConditionList(short_exit_container, get_session=self._current_session)
        self._condition_list_buttons(exit_section, self.short_exit_conditions, (0, 16))

    def _stop_target_block(self, type_widget, value_widget, atr_period_widget) -> tuple[str, float | None, int]:
        label = type_widget.get_str()
        kind = {"None": "none", "Fixed (pips)": "fixed", "ATR Multiple": "atr"}.get(label, "none")
        value = value_widget.get_float(0) if kind != "none" else None
        period = atr_period_widget.get_int(14)
        return kind, value, period

    def _build_strategy(self):
        mode = self.strategy_mode.get()

        if mode == "manual":
            # A manual strategy LOADED FROM THE LIBRARY (via LOAD SELECTED /
            # the Batch queue / a promoted Evolution Lab leader) must run
            # AS SAVED, not get silently replaced by whatever is currently
            # sitting in the interactive builder form below -- before this
            # check, _load_library_item_into_active_slot() only ever set
            # strategy_py_path/_active_library_strategy and switched the
            # mode; every actual strategy build still read the live form
            # fields, so loading a library item did nothing for manual
            # mode and Full Pipeline / Refinement / etc. always tested
            # whatever was left in the form instead of the loaded config.
            active = getattr(self, "_active_library_strategy", None)
            if (
                active is not None and active[0] == "manual"
                and self.strategy_py_path and Path(self.strategy_py_path).name == active[1]
            ):
                try:
                    cfg = json.loads(Path(self.strategy_py_path).read_text(encoding="utf-8"))
                except Exception as exc:
                    raise StrategyError(f"Could not load manual strategy '{active[1]}': {exc}") from exc
                return ManualStrategy(cfg)

            long_entry, long_entry_conn = self.long_entry_conditions.to_condition_list()
            short_entry, short_entry_conn = self.short_entry_conditions.to_condition_list()
            long_exit, long_exit_conn = self.long_exit_conditions.to_condition_list()
            short_exit, short_exit_conn = self.short_exit_conditions.to_condition_list()

            if not long_entry and not short_entry:
                raise StrategyError(
                    "Add at least one Long Entry or Short Entry condition in Step 2 before running."
                )

            stop_kind, stop_value, stop_period = self._stop_target_block(
                self.stop_type, self.stop_value, self.stop_atr_period
            )
            target_kind, target_value, target_period = self._stop_target_block(
                self.target_type, self.target_value, self.target_atr_period
            )

            max_bars_raw = self.max_bars.get_str().strip()

            cfg = {
                "name": self.s_name.get_str().strip() or "Manual Strategy",
                "description": self.s_description.get_str(),
                "author": self.s_author.get_str(),
                "version": self.s_version.get_str(),
                "market": {
                    "instrument": self.s_instrument.get_str(),
                    "timeframe": self.s_timeframe.get_str(),
                    "session_start": self.s_session_start.get_str().strip() or "08:30",
                    "session_end": self.s_session_end.get_str().strip() or "15:00",
                    "direction": self.s_direction.get_str(),
                },
                "entry_conditions": {
                    "long": long_entry, "long_connectors": long_entry_conn,
                    "short": short_entry, "short_connectors": short_entry_conn,
                },
                "exit_conditions": {
                    "long": long_exit, "long_connectors": long_exit_conn,
                    "short": short_exit, "short_connectors": short_exit_conn,
                },
                "risk_management": {
                    "stop_type": stop_kind,
                    "stop_value": stop_value,
                    "stop_atr_period": stop_period,
                    "target_type": target_kind,
                    "target_value": target_value,
                    "target_atr_period": target_period,
                    "trailing_stop": {
                        "enabled": self.trailing_enabled.get(),
                        "value": self.trailing_value.get_float(1.5),
                        "atr_period": self.trailing_atr_period.get_int(14),
                    },
                    "break_even": {
                        "enabled": self.breakeven_enabled.get(),
                        "trigger_r": self.breakeven_trigger.get_float(1.0),
                    },
                    "time_based_exit": {
                        "enabled": self.time_exit_enabled.get(),
                        "time": self.time_exit_time.get_str().strip(),
                    },
                    "max_bars_in_trade": int(max_bars_raw) if max_bars_raw else None,
                    "opposite_signal_exit": self.opposite_signal_exit.get(),
                },
            }
            return ManualStrategy(cfg)

        if not self.strategy_py_path:
            raise StrategyError(
                f"No file selected for '{mode}' strategy mode."
            )

        if mode == "python":
            return PythonStrategy(self.strategy_py_path)
        if mode == "pinescript":
            return PineScriptStrategy(self.strategy_py_path)
        if mode == "mql5":
            return MQL5Strategy(self.strategy_py_path)

        raise StrategyError(f"Unknown strategy mode: {mode}")
