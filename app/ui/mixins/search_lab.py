"""Search Lab, bulk backtests, multi-instrument search and auto-ensemble."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
import traceback
import webbrowser
from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.statistics import net_profit_reset_note
from app.data.importer import import_csv
from app.data.multi_timeframe import merge_multi_timeframe
from app.data.pairs import PairDataError, merge_pair_series
from app.data.storage import store_csv_path
from app.data.timeframe_resample import infer_timeframe_label as _infer_timeframe_label
from app.ensemble.auto_builder import AutoEnsembleError, build_diversified_ensemble
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.refinement import FITNESS_METRICS, OPTIMIZER_MODES
from app.orchestration import pipeline_guide
from app.orchestration.loop_runner import SearchLoopConfig, run_search_loop
from app.orchestration.multi_instrument_search import InstrumentJob
from app.orchestration.resource_guard import JOB_MULTI_INSTRUMENT_SEARCH, JOB_SEARCH_LAB
from app.prop.simulator import simulate_account
from app.reports.crash_log import log_crash
from app.reports.generator import generate_full_report
from app.search.batch_runner import SearchCancelled, SearchStageConfig, promote_champion, run_search
from app.search.search_report import generate_search_report
from app.search.strategy_space import StrategySpaceError, generate_search_space, list_families
from app.strategy.base import StrategyError
from app.strategy.library import record_search_result
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from tkinter import END, EXTENDED, Frame, Label, Listbox, Text, Toplevel, filedialog, messagebox, ttk
from app.ui.common import OUTPUT_DIR, _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui.widgets.cards import NeuralProgress
from app.ui.widgets.run_context import RunContextPanel
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
PANEL_3 = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
METAL_BRIGHT = ""
LOG_BG = ""
GREEN = ""
RED = ""
theme.bind(globals())


class SearchLabMixin:
    """Search Lab, bulk backtests, multi-instrument search and auto-ensemble."""


    # -----------------------------------------------------------------------
    # Tab 7 — Search Lab (Stages 1-5: cheap filter -> GA refinement ->
    # validation gate -> leaderboard -> champion promotion)
    # -----------------------------------------------------------------------

    _SEARCH_MODE_LABELS = {
        "Family -- named hypothesis grid (Manual strategies only)": "family_named",
        "Family -- grid around my current Strategy tab config (any strategy type)": "family_grid",
        "Single -- re-validate my current Strategy tab config exactly as configured": "single",
        "Bulk backtest -- upload multiple Python / PineScript / MQL5 files and run each one": "bulk_upload",
    }

    def _build_search_tab(self):
        f = self._scrollable(self.tab_search)

        self._page_header(
            f,
            "OPTIMIZE / Search Lab",
            "Search Lab (Stages 1-5)",
            "Generates and tests many strategy variations at once instead of one at a "
            "time: a fast filter narrows thousands of candidates down to a shortlist, "
            "the same Iterative Refinement engine from Step 6 tunes each shortlisted "
            "candidate, and a strict validation gate (walk-forward, lookahead check, "
            "parameter-neighborhood robustness, a deflated Sharpe ratio that corrects "
            "for how many candidates were tried) decides what actually survives. "
            "Works with Manual, Python, PineScript, and MQL5 strategies alike. "
            "Completely separate from the normal Run & Report pipeline and from Step 6 "
            "-- nothing here changes unless you click Run below. Its own market data, "
            "prop-firm rules, and risk settings below are self-contained to this tab, "
            "just like the web app -- independent of Steps 02-04 and every other tab.",
        )

        self.search_context = RunContextPanel(self, "Search Lab")
        self.search_context.build(f)
        self._button(f, "CHECK DATA INTEGRITY", lambda c=self.search_context: self._check_data_integrity(c)).pack(anchor="w", padx=18, pady=(0, 8))

        mode_section = self._section(
            f, "What to search",
            "\u2022 Named hypothesis grid expands one of this app's built-in trading hypotheses "
            "(Manual strategies only).\n"
            "\u2022 Grid around my current Strategy tab config discovers whatever's currently "
            "configured on Step 2's own tunable numeric parameters (indicator periods, stop/target "
            "values, or -- for Python/PineScript/MQL5 -- SCREAMING_SNAKE_CASE constants, "
            "input.int()/input.float() values, or iMA()/iRSI() periods) and grid-searches around "
            "them. Works for any of the 4 strategy types.\n"
            "\u2022 Single re-validates that one exact strategy through the same 5-stage funnel, "
            "with no parameter search at all -- an independent stress test.",
        )
        self.search_mode = LabeledCombo(
            mode_section, "Mode", list(self._SEARCH_MODE_LABELS.keys()),
            "Family -- named hypothesis grid (Manual strategies only)",
        )
        self.search_mode.combo.bind("<<ComboboxSelected>>", lambda _e: self._on_search_mode_changed())

        family_labels = ["All families (search every hypothesis together)"] + [
            f"{label} [{name}]" for name, label in list_families().items()
        ]
        self._search_family_label_to_key = {"All families (search every hypothesis together)": "all"}
        for name, label in list_families().items():
            self._search_family_label_to_key[f"{label} [{name}]"] = name
        self.search_family = LabeledCombo(
            mode_section, "Named hypothesis family (Manual-only mode above)", family_labels, family_labels[0],
        )
        self.search_grid_points = LabeledEntry(
            mode_section, "Grid points per parameter (grid-around-config mode above)", 3,
        )

        bulk_section = self._section(
            f, "Strategies to upload (Bulk backtest mode above)",
            "Runs every file added here through the exact same pipeline as Run & Report -- "
            "full historical backtest, prop-firm simulation, Monte Carlo, and a saved HTML "
            "report each -- reusing this tab's own Prop Rules and Risk & Execution settings "
            "and the dataset(s) selected above, so every strategy in the batch is judged on "
            "the same terms. Mixing Python (.py), PineScript (.pine/.txt), and "
            "MQL5 (.mq5) files in the same batch is fine -- each is detected by extension. "
            "Every result is recorded automatically and shows up on the Dashboard afterward.",
        )
        bulk_list_frame = Frame(bulk_section, bg=PANEL)
        bulk_list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.bulk_strategy_listbox = Listbox(
            bulk_list_frame, height=6, selectmode=EXTENDED, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
        )
        self.bulk_strategy_listbox.pack(side="left", fill="both", expand=True)
        bulk_scroll = ttk.Scrollbar(
            bulk_list_frame, orient="vertical", command=self.bulk_strategy_listbox.yview,
            style="T58.Vertical.TScrollbar",
        )
        bulk_scroll.pack(side="right", fill="y")
        self.bulk_strategy_listbox.config(yscrollcommand=bulk_scroll.set)
        self._bind_isolated_wheel(self.bulk_strategy_listbox)

        bulk_btn_row = Frame(bulk_section, bg=BG)
        bulk_btn_row.pack(fill="x", padx=18, pady=(4, 2))
        Label(
            bulk_btn_row, text="Add by type:", bg=BG, fg=TEXT_DIM, font=_safe_font(8, "bold"),
        ).pack(side="left", padx=(0, 8))
        self._button(bulk_btn_row, "MANUAL", lambda: self._bulk_add_files_by_type("manual"), primary=True).pack(side="left")
        self._button(bulk_btn_row, "PYTHON", lambda: self._bulk_add_files_by_type("python"), primary=True).pack(side="left", padx=8)
        self._button(bulk_btn_row, "PINESCRIPT", lambda: self._bulk_add_files_by_type("pinescript"), primary=True).pack(side="left")
        self._button(bulk_btn_row, "MQL5", lambda: self._bulk_add_files_by_type("mql5"), primary=True).pack(side="left", padx=8)

        bulk_btn_row_2 = Frame(bulk_section, bg=BG)
        bulk_btn_row_2.pack(fill="x", padx=18, pady=(0, 12))
        self._button(bulk_btn_row_2, "ADD MIXED FILES...", self._bulk_add_files).pack(side="left")
        self._button(bulk_btn_row_2, "ADD FROM STRATEGY LIBRARY", self._bulk_add_from_library).pack(side="left", padx=8)
        self._button(bulk_btn_row_2, "REMOVE SELECTED", self._bulk_remove_selected).pack(side="left", padx=8)
        self._button(bulk_btn_row_2, "CLEAR ALL", self._bulk_clear_files).pack(side="left")
        Label(
            bulk_section,
            text="MANUAL picks a saved strategy config (.json) -- easiest is ADD FROM STRATEGY "
            "LIBRARY instead, which lists every saved Manual/Python/PineScript/MQL5 strategy by "
            "name. PYTHON/PINESCRIPT/MQL5 browse directly to a source file of that type. ADD "
            "MIXED FILES lets you multi-select across types in one dialog.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 4))
        self.bulk_upload_status = Label(
            bulk_section, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        )
        self.bulk_upload_status.pack(anchor="w", padx=18, pady=(0, 10))

        self._bulk_strategy_paths: list[Path] = []

        space_section = self._section(
            f, "Search space size",
            "How many generated candidates Stage 1 actually evaluates. If the full grid "
            "for a family is larger than this, a random (reproducible) sample is taken "
            "instead of just the first N -- see the random seed below.",
        )
        self.search_max_candidates = LabeledEntry(space_section, "Max candidates (Family mode)", 500)
        self.search_workers = LabeledEntry(space_section, "Parallel workers (blank = all CPU cores)", "")
        self.search_seed = LabeledEntry(space_section, "Random seed", 42)

        pair_section = self._section(
            f, "Pair instrument (Statistical Pairs / Relative Value family only)",
            "Merges a second instrument's close price in as a 'pair_close' column so the "
            "'stat_pairs' family can be searched. Leave blank to search every OTHER family "
            "as usual (a family=All search simply skips stat_pairs without this).",
        )
        pair_row = Frame(pair_section, bg=PANEL)
        pair_row.pack(fill="x", padx=18, pady=(2, 10))
        self.search_pair_csv_label = Label(
            pair_row, text="No pair instrument selected.", bg=PANEL, fg=TEXT_MUTED,
            font=_safe_font(9), anchor="w",
        )
        self.search_pair_csv_label.pack(side="left", fill="x", expand=True)
        self._button(pair_row, "CHOOSE PAIR CSV...", self._search_choose_pair_csv).pack(side="right")
        self._button(pair_row, "CLEAR", self._search_clear_pair_csv).pack(side="right", padx=(0, 8))
        self._search_pair_csv_path: str | None = None

        stage1_section = self._section(
            f, "Stage 1 -- cheap filter",
            "One fast backtest per candidate, no Monte Carlo. Kills the vast majority "
            "of candidates in minutes, not hours. Loosen these if a run reports zero "
            "Stage 1 survivors.",
        )
        self.search_min_trades = LabeledEntry(stage1_section, "Minimum trades to survive", 20)
        self.search_min_pf = LabeledEntry(stage1_section, "Minimum profit factor to survive", 1.05)
        self.search_stage1_top_n = LabeledEntry(stage1_section, "Survivors that advance to Stage 2 (GA)", 40)
        self.search_reset_on_breach = LabeledCheckbox(
            stage1_section,
            "Reset-on-breach: score every candidate on the basis that a blown account gets "
            "a fresh eval and keeps going, not a dead end (recommended -- uncheck to "
            "optimize on a single account with no reset)",
            True,
        )

        stage2_section = self._section(
            f, "Stage 2 -- GA refinement",
            "The same genetic-algorithm engine as Step 6, applied to every Stage 1 "
            "survivor instead of to one hand-picked strategy.",
        )
        self.search_ga_population = LabeledEntry(stage2_section, "GA population size", 10)
        self.search_ga_generations = LabeledEntry(stage2_section, "GA generations", 4)
        self.search_stage2_top_n = LabeledEntry(stage2_section, "Survivors that advance to Stage 3 (validation)", 10)
        self.search_cost_stress_enabled = LabeledCheckbox(stage2_section, "Enable cost-stress penalty in the GA fitness", True)
        self.search_cost_stress_multiplier = LabeledEntry(stage2_section, "Cost multiplier for the stressed re-run", 2.0)
        self.search_cost_stress_weight = LabeledEntry(stage2_section, "Penalty weight (0=ignore, 1=full)", 0.35)

        stage3_section = self._section(
            f, "Stage 3 -- validation gate",
            "Full-fidelity Monte Carlo, multi-fold walk-forward, the lookahead-bias "
            "detector, and parameter-neighborhood robustness. A candidate must clear "
            "every gate to become the champion -- this is deliberately strict.",
        )
        self.search_full_mc_sims = LabeledEntry(stage3_section, "Monte Carlo simulations (full fidelity)", 3000)
        self.search_walk_forward_folds = LabeledEntry(stage3_section, "Walk-forward folds (0 disables)", 4)
        self.search_robustness_neighbors = LabeledEntry(
            stage3_section, "Parameter-neighborhood samples (0 disables)", 6,
        )
        self._search_metric_labels = list(FITNESS_METRICS.values())
        self.search_metric = LabeledCombo(
            stage3_section, "Fitness metric (what \u201cbest\u201d means)", self._search_metric_labels,
            FITNESS_METRICS["eval_pass_probability"],
        )
        # Independent of _build_refine_tab's own copy of these two maps --
        # never relies on tab build order.
        self._optimizer_mode_labels = list(OPTIMIZER_MODES.values())
        self._optimizer_mode_label_to_key = {v: k for k, v in OPTIMIZER_MODES.items()}
        self.search_optimizer_mode = LabeledCombo(
            stage3_section, "Optimizer mode (Stage 2 refinement)", self._optimizer_mode_labels,
            OPTIMIZER_MODES["genetic"],
        )

        loop_section = self._section(
            f, "Loop mode",
            "Only applies to a named-family/all-families search (not Single-strategy or Family "
            "Grid mode). Repeats Search Lab rounds automatically instead of stopping after one: "
            "if a round makes no progress for a few rounds in a row, the next round broadens to "
            "every family (dropping any family Family Health has flagged as a dead end) and "
            "raises the candidate cap. Stops the moment a candidate clears your target, when the "
            "round or time budget runs out, or when you hit STOP.",
        )
        self.search_loop_mode = LabeledCheckbox(loop_section, "Enable Loop Mode", False)
        self.search_loop_target = LabeledEntry(loop_section, "Stop once a candidate's eval-pass % reaches", 60)
        self.search_loop_max_rounds = LabeledEntry(loop_section, "Max rounds", 20)
        self.search_loop_time_budget_hours = LabeledEntry(
            loop_section, "Time budget in hours (blank = no limit, rely on rounds/STOP)", "",
        )
        self.search_loop_stall_rounds = LabeledEntry(loop_section, "Rounds with no improvement before widening", 2)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)

        self._button(button_row, "RUN SEARCH LAB", self._search_run_clicked, primary=True).pack(side="left")

        self.stop_search_btn = self._button(button_row, "STOP", self._search_stop_clicked)
        self.stop_search_btn.config(state="disabled")
        self.stop_search_btn.pack(side="left", padx=8)

        self.open_search_report_btn = self._button(
            button_row, "OPEN LEADERBOARD", self._open_search_report,
        )
        self.open_search_report_btn.config(state="disabled")
        self.open_search_report_btn.pack(side="left", padx=8)

        self.promote_champion_btn = self._button(
            button_row, "PROMOTE CHAMPION TO FULL REPORT", self._promote_search_champion_clicked,
        )
        self.promote_champion_btn.config(state="disabled")
        self.promote_champion_btn.pack(side="left", padx=8)

        self.open_champion_report_btn = self._button(
            button_row, "OPEN CHAMPION REPORT", self._open_champion_report,
        )
        self.open_champion_report_btn.config(state="disabled")
        self.open_champion_report_btn.pack(side="left", padx=8)

        # UPGRADE (ensemble-builder button): app.ensemble.auto_builder.
        # build_diversified_ensemble already existed and is already wired
        # on the web app's own Search Lab job page -- this is the same
        # one-click "turn this finished run's own leaderboard into a
        # diversified 3-5-leg basket" action, using the exact same
        # instrument/timeframe/risk/rules this Search Lab run just used.
        self.auto_ensemble_btn = self._button(
            button_row, "AUTO-BUILD ENSEMBLE FROM LEADERBOARD", self._search_auto_build_ensemble,
        )
        self.auto_ensemble_btn.config(state="disabled")
        self.auto_ensemble_btn.pack(side="left", padx=8)

        self.search_progress = NeuralProgress(f)
        self.search_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Search Lab output", "Live funnel log.")
        _search_output_frame = Frame(output_section, bg=PANEL)
        self.search_output = Text(
            _search_output_frame, height=18, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _search_output_scroll = ttk.Scrollbar(
            _search_output_frame, orient="vertical", command=self.search_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.search_output.configure(yscrollcommand=_search_output_scroll.set)
        self.search_output.pack(side="left", fill="both", expand=True)
        _search_output_scroll.pack(side="right", fill="y")
        _search_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.search_output)

        self._last_search_summary = None
        self._last_search_space = None
        self._last_search_html_path = None
        self._last_search_db_path = None
        self._last_search_df = None
        self._last_search_risk = None
        self._last_search_rules = None
        self._last_champion_html_path = None
        self._search_cancel_event = threading.Event()

    def _on_search_mode_changed(self):
        mode = self._SEARCH_MODE_LABELS.get(self.search_mode.get_str())
        # Family-only settings are visually left enabled either way (Tk
        # combobox rebuilding is more churn than it's worth for a disabled
        # look) -- they're simply ignored by _search_run_pipeline in Single
        # mode.

    def _bulk_add_files(self):
        paths = filedialog.askopenfilenames(
            title="Add strategy files (Python / PineScript / MQL5)",
            filetypes=[
                ("Supported strategy files", "*.py *.pine *.pinescript *.mq5 *.mqh *.txt"),
                ("Python strategies", "*.py"),
                ("PineScript strategies", "*.pine *.pinescript *.txt"),
                ("MQL5 strategies", "*.mq5 *.mqh"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        existing = {str(p) for p in self._bulk_strategy_paths}
        for p in paths:
            if p not in existing:
                self._bulk_strategy_paths.append(Path(p))
                self.bulk_strategy_listbox.insert(END, f"  {Path(p).name}")

    # Per-type file dialog filters shared by the Search Lab bulk uploader
    # and the Portfolio / Ensemble "add leg/member by type" buttons -- one
    # definition instead of four near-duplicate filedialog calls scattered
    # across those three features.
    _STRATEGY_TYPE_FILEDIALOG = {
        "manual": ("Add a Manual Strategy Builder config", [("Manual strategy config", "*.json"), ("All files", "*.*")]),
        "python": ("Add a Python strategy", [("Python strategies", "*.py"), ("All files", "*.*")]),
        "pinescript": ("Add a PineScript strategy", [("PineScript strategies", "*.pine *.pinescript *.txt"), ("All files", "*.*")]),
        "mql5": ("Add an MQL5 strategy", [("MQL5 strategies", "*.mq5 *.mqh"), ("All files", "*.*")]),
    }

    def _bulk_add_files_by_type(self, strategy_type: str):
        """MANUAL / PYTHON / PINESCRIPT / MQL5 quick-add buttons -- same
        destination list as ADD MIXED FILES, just pre-filtered to one
        format so a Manual .json isn't lost in a generic 'All files' pick."""
        title, filetypes = self._STRATEGY_TYPE_FILEDIALOG[strategy_type]
        paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
        if not paths:
            return
        existing = {str(p) for p in self._bulk_strategy_paths}
        added = 0
        for p in paths:
            if p not in existing:
                self._bulk_strategy_paths.append(Path(p))
                self.bulk_strategy_listbox.insert(END, f"  [{strategy_type}] {Path(p).name}")
                added += 1
        if added:
            self.bulk_upload_status.config(
                text=f"Added {added} {strategy_type} file(s).", fg=TEXT_DIM,
            )

    def _bulk_add_from_library(self):
        """Adds the currently-selected Strategy Library row(s) (Step 1
        Strategy Configuration) straight into the bulk-upload list -- lets
        Manual Strategy Builder configs (which normally aren't loose files
        on disk) join the same batch as Python/PineScript/MQL5 files."""
        items = self._selected_library_items()
        if not items:
            messagebox.showinfo(
                "No selection",
                "Select one or more saved strategies in the Strategy Library "
                "(01 Strategy Configuration) first, then click ADD FROM STRATEGY LIBRARY.",
            )
            return
        existing = {str(p) for p in self._bulk_strategy_paths}
        added = 0
        for item in items:
            p = str(item.path)
            if p not in existing:
                self._bulk_strategy_paths.append(Path(p))
                self.bulk_strategy_listbox.insert(END, f"  [{item.strategy_type}] {item.name}")
                existing.add(p)
                added += 1
        self.bulk_upload_status.config(
            text=f"Added {added} strategy(ies) from the library.", fg=TEXT_DIM,
        )

    def _bulk_remove_selected(self):
        sel = list(self.bulk_strategy_listbox.curselection())
        for i in reversed(sel):
            self.bulk_strategy_listbox.delete(i)
            del self._bulk_strategy_paths[i]

    def _bulk_clear_files(self):
        self.bulk_strategy_listbox.delete(0, END)
        self._bulk_strategy_paths = []

    @staticmethod
    def _load_bulk_strategy(path: Path):
        """Detect a strategy's source type by extension and construct it --
        mirrors what the Strategy tab does per-type, just driven by a file
        list instead of the tab's radio selection."""
        suffix = path.suffix.lower()
        if suffix == ".py":
            return PythonStrategy(path)
        if suffix in (".pine", ".pinescript"):
            return PineScriptStrategy(path)
        if suffix in (".mq5", ".mqh"):
            return MQL5Strategy(path)
        if suffix == ".json":
            # Manual Strategy Builder configs saved to the library (or
            # promoted from the Evolution Lab leaderboard) are JSON, not
            # code -- this is what lets a manual strategy sit in the
            # Batch selection checklist / OPTIMIZE SELECTED / the bulk multi-
            # strategy list alongside Python/PineScript/MQL5 files instead
            # of being the one type those features silently couldn't load.
            cfg = json.loads(path.read_text(encoding="utf-8"))
            return ManualStrategy(cfg)
        if suffix == ".txt":
            # Ambiguous extension -- sniff for PineScript's declaration
            # syntax before falling back to treating it as one.
            try:
                head = path.read_text(encoding="utf-8", errors="ignore")[:400]
            except OSError:
                head = ""
            if "//@version" in head or "strategy(" in head or "indicator(" in head:
                return PineScriptStrategy(path)
            return PineScriptStrategy(path)
        raise StrategyError(f"Unsupported strategy file type: {path.name}")

    def _log_search(self, msg: str):
        self.search_output.insert(END, msg + "\n")
        self.search_output.see(END)
        self.root.update_idletasks()

    def _open_search_report(self):
        if self._last_search_html_path:
            webbrowser.open(f"file://{self._last_search_html_path.resolve()}")

    def _open_champion_report(self):
        if self._last_champion_html_path:
            webbrowser.open(f"file://{self._last_champion_html_path.resolve()}")

    def _search_choose_pair_csv(self):
        path = filedialog.askopenfilename(
            title="Select the second instrument's market data CSV (for the stat_pairs family)",
            filetypes=[("Market data", "*.csv *.tsv *.txt *.parquet *.zip *.7z"), ("All files", "*.*")],
        )
        if not path:
            return
        self._search_pair_csv_path = path
        self.search_pair_csv_label.config(text=os.path.basename(path), fg=TEXT)

    def _search_clear_pair_csv(self):
        self._search_pair_csv_path = None
        self.search_pair_csv_label.config(text="No pair instrument selected.", fg=TEXT_MUTED)

    def _build_search_stage_config(self) -> SearchStageConfig:
        metric_label = self.search_metric.get_str()
        metric_key = self._refine_metric_label_to_key.get(metric_label, "eval_pass_probability")
        optimizer_label = self.search_optimizer_mode.get_str()
        optimizer_key = self._optimizer_mode_label_to_key.get(optimizer_label, "genetic")
        workers_raw = self.search_workers.get_str().strip()
        workers = int(workers_raw) if workers_raw else None
        return SearchStageConfig(
            min_trades=self.search_min_trades.get_int(20),
            min_profit_factor=self.search_min_pf.get_float(1.05),
            stage1_top_n=self.search_stage1_top_n.get_int(40),
            ga_population=self.search_ga_population.get_int(10),
            ga_generations=self.search_ga_generations.get_int(4),
            stage2_top_n=self.search_stage2_top_n.get_int(10),
            full_mc_sims=self.search_full_mc_sims.get_int(3000),
            walk_forward_folds=self.search_walk_forward_folds.get_int(4),
            robustness_neighbors=self.search_robustness_neighbors.get_int(6),
            fitness_metric=metric_key,
            workers=workers,
            random_seed=self.search_seed.get_int(42),
            cost_stress_enabled=self.search_cost_stress_enabled.get(),
            cost_stress_multiplier=self.search_cost_stress_multiplier.get_float(2.0),
            cost_stress_penalty_weight=self.search_cost_stress_weight.get_float(0.35),
            reset_on_breach=self.search_reset_on_breach.var.get(),
            optimizer_mode=optimizer_key,
        )

    def _search_run_clicked(self):
        if not self.search_context.csv_paths:
            messagebox.showwarning(
                "Missing data",
                "Please select a market data CSV above (this tab's own Market Data section).",
            )
            return

        mode_key = self._SEARCH_MODE_LABELS.get(self.search_mode.get_str(), "family_named")
        if mode_key == "bulk_upload":
            if not self._bulk_strategy_paths:
                messagebox.showwarning(
                    "No strategy files added",
                    "Add at least one Python (.py), PineScript (.pine), or MQL5 (.mq5) "
                    "file above before running a bulk backtest.",
                )
                return
            self.search_output.delete("1.0", END)
            self.open_search_report_btn.config(state="disabled")
            self.promote_champion_btn.config(state="disabled")
            self.open_champion_report_btn.config(state="disabled")
            self.auto_ensemble_btn.config(state="disabled")
            self.search_progress.start(10)
            # The bulk backtest loop isn't cancellable mid-strategy (each one
            # is quick), so STOP stays disabled for this mode -- it's only
            # meaningful for the long-running Stage 1-5 GA search below.
            threading.Thread(target=self._run_bulk_backtest_pipeline, daemon=True).start()
            return

        if not self._try_start_heavy_job(JOB_SEARCH_LAB):
            return
        self.search_output.delete("1.0", END)
        self.open_search_report_btn.config(state="disabled")
        self.promote_champion_btn.config(state="disabled")
        self.open_champion_report_btn.config(state="disabled")
        self.auto_ensemble_btn.config(state="disabled")
        self._search_cancel_event.clear()
        self.stop_search_btn.config(state="normal")
        self.search_progress.start(10)

        if self.search_loop_mode.get():
            if mode_key not in ("family_named",):
                self._release_heavy_job(JOB_SEARCH_LAB)
                self.stop_search_btn.config(state="disabled")
                self.search_progress.stop()
                messagebox.showwarning(
                    "Loop Mode needs a family search",
                    "Loop Mode repeats a named-family/all-families search across rounds -- it "
                    "doesn't apply to Single-strategy or Family Grid mode (each of those is "
                    "already searching one strategy's own grid, with no other family to widen "
                    "into on a stall). Switch Mode to a named family or All, or turn off Loop "
                    "Mode, then run again.",
                )
                return
            threading.Thread(target=self._search_run_loop_pipeline, daemon=True).start()
            return

        threading.Thread(target=self._search_run_pipeline, daemon=True).start()

    def _search_stop_clicked(self):
        """STOP for the Stage 1-5 GA search. Sets the shared cancel event,
        which run_search() checks between candidates in every stage -- it
        doesn't kill mid-flight worker processes, but it stops the run from
        starting any further candidates and shuts the pool down instead of
        grinding through everything still queued. Any candidates already
        scored before the stop are kept in the results DB, not thrown away."""
        self._search_cancel_event.set()
        self.stop_search_btn.config(state="disabled")
        self._log_search("\nStopping... (finishing the candidates already in flight, no new ones will start)")

    def _run_bulk_backtest_pipeline(self):
        """Runs every uploaded strategy file through the exact same
        backtest -> prop-firm sim -> Monte Carlo -> report pipeline as
        Run & Report, one after another, reusing the currently configured
        Prop Rules / Risk / Monte Carlo settings so every strategy is
        judged on the same terms. Each report funnels through
        generate_full_report(), so every result is automatically recorded
        into run_history and shows up on the Dashboard afterward -- no
        separate wiring needed here."""
        try:
            self._log_search(f"Loading {len(self.search_context.csv_paths)} market data file(s)...")
            per_file_results = []
            for p in self.search_context.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    self._log_search(
                        f"Import errors ({os.path.basename(p)}):\n" + "\n".join(result.errors)
                    )
                    return
                per_file_results.append((p, result))

            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
            else:
                df, _labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
            self._log_search(f"Loaded {len(df)} bars.\n")

            risk = self.search_context.build_risk_config()
            rules = self.search_context.build_prop_rules()
            n_sims = self.mc_sims.get_int(10000)
            method = self.mc_method.get_str().strip() or "bootstrap"
            instrument = self.search_context.instrument_label()
            period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))

            total = len(self._bulk_strategy_paths)
            results_summary = []

            for i, path in enumerate(self._bulk_strategy_paths, start=1):
                self._log_search(f"[{i}/{total}] {path.name}")
                try:
                    strategy = self._load_bulk_strategy(path)
                except (StrategyError, Exception) as exc:
                    self._log_search(f"  Skipped -- could not load strategy: {exc}\n")
                    continue

                try:
                    bt_result = run_backtest(df, strategy, risk)
                except Exception as exc:
                    self._log_search(f"  Skipped -- backtest error: {exc}\n")
                    continue

                if not bt_result.trades:
                    self._log_search("  Skipped -- 0 trades generated on this data.\n")
                    continue

                self._log_search(
                    f"  Trades: {len(bt_result.trades)}  "
                    f"Net profit: ${bt_result.statistics.net_profit:,.2f}  "
                    f"Win rate: {bt_result.statistics.win_rate:.1f}%"
                    + net_profit_reset_note(bt_result.statistics)
                )

                trade_pnls = [t.pnl for t in bt_result.trades]
                trade_dates = [t.entry_time for t in bt_result.trades]
                single_run = simulate_account(trade_pnls, trade_dates, rules)

                mc_cfg = MonteCarloConfig(n_simulations=n_sims, method=method)
                mc_result = run_monte_carlo(bt_result.trades, rules, mc_cfg)

                try:
                    holdout_comparison = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2)
                except Exception:
                    holdout_comparison = None

                paths = generate_full_report(
                    output_dir=OUTPUT_DIR,
                    basename=f"bulk_{i:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', path.stem)}",
                    strategy_name=path.stem,
                    strategy_source_type=strategy.source_type,
                    instrument=instrument,
                    timeframe=_infer_timeframe_label(df),
                    backtest_period=period,
                    backtest_result=bt_result,
                    prop_rules=rules,
                    prop_single_run=single_run,
                    monte_carlo_result=mc_result,
                    holdout_comparison=holdout_comparison,
                    risk_config=risk,
                    price_df=df,
                )

                self._log_search(
                    f"  Eval pass probability: {mc_result.evaluation_pass_probability:.1f}%  "
                    f"Report: {paths['html'].name}\n"
                )
                results_summary.append({
                    "name": path.stem,
                    "net_profit": bt_result.statistics.net_profit,
                    "eval_pass": mc_result.evaluation_pass_probability,
                    "html": paths["html"],
                })

            self._log_search(f"\nDone. {len(results_summary)}/{total} strategies produced a report.")
            if results_summary:
                ranked = sorted(results_summary, key=lambda r: r["eval_pass"], reverse=True)
                self._log_search("\nRanked by eval pass probability:")
                for r in ranked:
                    self._log_search(
                        f"  {r['eval_pass']:5.1f}%  ${r['net_profit']:>12,.2f}   {r['name']}"
                    )
                self._last_search_html_path = ranked[0]["html"]
                self.open_search_report_btn.config(state="normal")
                self._log_search(
                    "\nOpen Leaderboard above opens the top-ranked strategy's report. "
                    "Full results for all strategies are on the Dashboard tab."
                )
            try:
                self._refresh_dashboard()
            except Exception:
                pass

        except Exception:
            self._log_search("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.search_progress.stop()
            self.stop_search_btn.config(state="disabled")

    def _search_run_pipeline(self):
        try:
            self._log_search("Importing market data...")
            per_file_results = []
            for p in self.search_context.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    self._log_search(
                        f"Import errors ({os.path.basename(p)}):\n" + "\n".join(result.errors)
                    )
                    return
                per_file_results.append((p, result))

            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
            else:
                df, _labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
            self._log_search(f"Loaded {len(df)} bars.")

            has_pair_data = False
            if self._search_pair_csv_path:
                pair_result = import_csv(str(store_csv_path(self._search_pair_csv_path)))
                if not pair_result.is_valid:
                    self._log_search(
                        "Pair CSV import errors:\n" + "\n".join(pair_result.errors)
                    )
                    return
                df = merge_pair_series(df, pair_result.dataframe)
                has_pair_data = True
                self._log_search(
                    f"Merged pair instrument from {os.path.basename(self._search_pair_csv_path)} "
                    "(enables the 'stat_pairs' family)."
                )

            risk = self.search_context.build_risk_config()
            rules = self.search_context.build_prop_rules()

            mode_key = self._SEARCH_MODE_LABELS.get(self.search_mode.get_str(), "family_named")
            if mode_key == "single":
                strategy = self._build_strategy()  # any of the 4 types, whatever's configured
                space = generate_search_space(mode="single", strategy=strategy)
            elif mode_key == "family_grid":
                strategy = self._build_strategy()
                space = generate_search_space(
                    mode="family", strategy=strategy,
                    grid_points_per_gene=self.search_grid_points.get_int(3),
                    max_candidates=self.search_max_candidates.get_int(500),
                    seed=self.search_seed.get_int(42),
                )
            else:  # family_named
                family_label = self.search_family.get_str()
                family_key = self._search_family_label_to_key.get(family_label, "all")
                space = generate_search_space(
                    mode="family", family=family_key,
                    max_candidates=self.search_max_candidates.get_int(500),
                    seed=self.search_seed.get_int(42),
                    has_pair_data=has_pair_data,
                )

            # (run_search() itself logs the "Search space ready..." line via
            # progress_cb once it starts -- not duplicated here.)
            stage_cfg = self._build_search_stage_config()
            instrument = self.search_context.instrument_label()
            db_path = str(OUTPUT_DIR / "search" / "search.db")

            # timeframe is now the real inferred value (see
            # app.data.timeframe_resample.infer_timeframe_label) --
            # app.web.server's own /search/start route was updated to
            # compute the SAME value from the same df, so a desktop run
            # and a web run against equivalent data still land in the
            # same graveyard_path_for(instrument, timeframe) file; only
            # runs against genuinely irregular/malformed timestamp data
            # (where inference itself fails) fall back to "unknown".
            summary = run_search(
                df, risk, rules, space, stage_cfg, db_path=db_path,
                instrument=instrument, timeframe=_infer_timeframe_label(df), progress_cb=self._log_search,
                cancel_event=self._search_cancel_event,
            )

            report_paths = generate_search_report(
                output_dir=str(OUTPUT_DIR / "search"), summary=summary, space=space,
                instrument=instrument, timeframe=_infer_timeframe_label(df),
            )

            active_lib_strategy = getattr(self, "_active_library_strategy", None)
            if mode_key in ("single", "family_grid") and active_lib_strategy and summary.leaderboard:
                try:
                    record_search_result(*active_lib_strategy, {
                        "candidates_tested": summary.total_candidates,
                        "best_fitness": round(summary.leaderboard[0].get("fitness", 0), 4),
                        "fitness_metric": stage_cfg.fitness_metric,
                        "report_html": str(report_paths["html"]),
                    })
                except (FileNotFoundError, ValueError):
                    pass  # base strategy renamed/deleted mid-search -- not worth failing the run over

            self._last_search_summary = summary
            self._last_search_space = space
            try:
                self._refresh_dashboard()
            except Exception:
                pass
            self._last_search_html_path = report_paths["html"]
            self._last_search_db_path = db_path
            self._last_search_df = df
            self._last_search_risk = risk
            self._last_search_rules = rules

            self.open_search_report_btn.config(state="normal")
            self.auto_ensemble_btn.config(state="normal")
            if summary.champion_candidate_id:
                self.promote_champion_btn.config(state="normal")

            self._log_search("\nDone. Search leaderboard written to:")
            for k, p in report_paths.items():
                self._log_search(f"  {k}: {p}")
            self._log_search(
                "\n" + pipeline_guide.after_search_complete(
                    summary.champion_candidate_id, len(summary.leaderboard or []),
                    total_candidates=summary.total_candidates,
                )
            )

        except SearchCancelled:
            self._log_search(
                "\nSearch Lab stopped. Candidates already scored before the stop are still "
                "saved in the results DB, even though no leaderboard report was generated "
                "for this (incomplete) run."
            )
        except StrategySpaceError as exc:
            self._log_search(f"\nSearch space error: {exc}")
        except PairDataError as exc:
            self._log_search(f"\nPair CSV error: {exc}")
        except StrategyError as exc:
            self._log_search(f"\nStrategy error: {exc}")
        except BrokenProcessPool:
            self._log_search(
                "\nSearch Lab crashed: a worker process was terminated abruptly "
                "(BrokenProcessPool).\n\n"
                "If you're running the built .exe: this is almost always caused "
                "by a .exe built before this app called "
                "multiprocessing.freeze_support() at startup -- without it, every "
                "worker process a packaged .exe spawns re-launches the whole app "
                "instead of running as a plain worker, and immediately dies. "
                "Rebuild/redownload the .exe (run_app.py now calls "
                "freeze_support() first thing) and try again.\n\n"
                "If you're running from source (python launchers/run_app.py) and still see "
                "this, it usually means a worker genuinely ran out of memory or "
                "crashed hard -- try lowering 'Parallel workers' on this tab, or "
                "reducing the candidate pool / population size."
            )
            log_crash("Search Lab (BrokenProcessPool)")
        except Exception as exc:
            self._log_search("\nUnexpected error:\n" + traceback.format_exc())
            log_crash("Search Lab", exc=exc)
        finally:
            self.search_progress.stop()
            self._release_heavy_job(JOB_SEARCH_LAB)

    def _search_run_loop_pipeline(self):
        """Loop Mode's desktop counterpart to _search_run_pipeline above --
        same data-loading/risk/rules setup, but drives
        app.orchestration.loop_runner.run_search_loop (repeated rounds,
        widening on stall) instead of a single run_search() call. Reuses
        the same log box / progress bar / STOP button / heavy-job guard as
        the normal Stage 1-5 run; the only genuinely new UI state is
        self._last_search_summary/_space, which are updated after EVERY
        round (via on_round) so OPEN LEADERBOARD / PROMOTE CHAMPION work
        against the latest round even while the loop keeps running."""
        try:
            self._log_search("Importing market data...")
            per_file_results = []
            for p in self.search_context.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    self._log_search(
                        f"Import errors ({os.path.basename(p)}):\n" + "\n".join(result.errors)
                    )
                    return
                per_file_results.append((p, result))

            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
            else:
                df, _labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
            self._log_search(f"Loaded {len(df)} bars.")

            has_pair_data = False
            if self._search_pair_csv_path:
                pair_result = import_csv(str(store_csv_path(self._search_pair_csv_path)))
                if not pair_result.is_valid:
                    self._log_search(
                        "Pair CSV import errors:\n" + "\n".join(pair_result.errors)
                    )
                    return
                df = merge_pair_series(df, pair_result.dataframe)
                has_pair_data = True
                self._log_search(
                    f"Merged pair instrument from {os.path.basename(self._search_pair_csv_path)} "
                    "(enables the 'stat_pairs' family)."
                )

            risk = self.search_context.build_risk_config()
            rules = self.search_context.build_prop_rules()
            stage_cfg = self._build_search_stage_config()
            instrument = self.search_context.instrument_label()

            family_label = self.search_family.get_str()
            family_key = self._search_family_label_to_key.get(family_label, "all")
            time_budget_raw = self.search_loop_time_budget_hours.get_str().strip()
            loop_cfg = SearchLoopConfig(
                target_eval_pass_pct=self.search_loop_target.get_float(60.0),
                max_rounds=self.search_loop_max_rounds.get_int(20),
                time_budget_seconds=(float(time_budget_raw) * 3600.0) if time_budget_raw else None,
                stall_rounds_before_widen=self.search_loop_stall_rounds.get_int(2),
                starting_family=(None if family_key in (None, "all") else family_key),
                starting_max_candidates=self.search_max_candidates.get_int(500),
                seed=self.search_seed.get_int(42),
            )
            self._log_search(
                f"Loop mode ON -- will repeat Search Lab rounds (widening on stall) until a "
                f"candidate clears {loop_cfg.target_eval_pass_pct:.0f}%, {loop_cfg.max_rounds} "
                f"rounds run, or the time budget is used up. has_pair_data={has_pair_data}."
            )

            loop_dir = str(OUTPUT_DIR / "search" / "loop")

            def on_round(round_result) -> None:
                # Same "directly touch Tk state from the worker thread"
                # convention _log_search already uses throughout this file
                # -- not officially thread-safe per Tkinter's own docs, but
                # consistent with every other background job in this app.
                self._last_search_summary = round_result.summary
                self._last_search_space = round_result.space
                self._last_search_db_path = round_result.summary.db_path
                self._last_search_df = df
                self._last_search_risk = risk
                self._last_search_rules = rules
                self.open_search_report_btn.config(state="normal")
                if round_result.summary.champion_candidate_id:
                    self.promote_champion_btn.config(state="normal")
                try:
                    self._refresh_dashboard()
                except Exception:
                    pass

            # timeframe is now the real inferred value, same coordinated
            # fix as plain Search Lab's run_search call above (see its
            # comment) -- web's /search/loop route was updated to match.
            result = run_search_loop(
                df, risk, rules, stage_cfg, db_dir=loop_dir, loop_cfg=loop_cfg,
                instrument=instrument, timeframe=_infer_timeframe_label(df), progress_cb=self._log_search,
                cancel_event=self._search_cancel_event, on_round=on_round,
                family_health_search_dir=str(OUTPUT_DIR / "search"),
                family_health_evolution_dir=str(OUTPUT_DIR / "evolution"),
            )

            if result.stopped_reason == "error":
                self._log_search(f"\nLoop Mode stopped: a round crashed -- {result.error}")
                log_crash("Search Lab Loop Mode", exc=RuntimeError(result.error or "unknown"))
                return
            if result.stopped_reason == "cancelled":
                self._log_search(
                    "\nLoop Mode stopped by user. The latest completed round's results are "
                    "still on the leaderboard above."
                )
                return

            reason_text = {
                "target_reached": f"Target reached ({loop_cfg.target_eval_pass_pct:.0f}%) -- stopped automatically.",
                "max_rounds": f"Stopped: all {loop_cfg.max_rounds} rounds completed with no candidate clearing the target.",
                "time_budget": "Stopped: time budget used up.",
            }.get(result.stopped_reason, result.stopped_reason)
            self._log_search(f"\n{reason_text}")

            winner = result.winner_round
            if winner is not None:
                self._log_search(f"Winning candidate: {result.winner_candidate_id}")
                report_paths = generate_search_report(
                    output_dir=str(OUTPUT_DIR / "search"), summary=winner.summary, space=winner.space,
                    instrument=instrument, timeframe=_infer_timeframe_label(df),
                )
                self._last_search_html_path = report_paths["html"]
                self._log_search("\nWinning round's leaderboard written to:")
                for k, p in report_paths.items():
                    self._log_search(f"  {k}: {p}")
            elif result.rounds:
                last = result.rounds[-1]
                self._log_search(
                    "No round produced a candidate clearing the target -- the leaderboard above "
                    "still shows the closest calls from the most recent round."
                )
                if last.summary.leaderboard:
                    report_paths = generate_search_report(
                        output_dir=str(OUTPUT_DIR / "search"), summary=last.summary, space=last.space,
                        instrument=instrument, timeframe=_infer_timeframe_label(df),
                    )
                    self._last_search_html_path = report_paths["html"]

        except SearchCancelled:
            self._log_search(
                "\nSearch Lab Loop Mode stopped. Candidates already scored before the stop are "
                "still saved in the results DB, even though no leaderboard report was generated "
                "for the in-progress round."
            )
        except StrategySpaceError as exc:
            self._log_search(f"\nSearch space error: {exc}")
        except PairDataError as exc:
            self._log_search(f"\nPair CSV error: {exc}")
        except Exception as exc:
            self._log_search("\nUnexpected error:\n" + traceback.format_exc())
            log_crash("Search Lab Loop Mode", exc=exc)
        finally:
            self.search_progress.stop()
            self.stop_search_btn.config(state="disabled")
            self._release_heavy_job(JOB_SEARCH_LAB)

    def _promote_search_champion_clicked(self):
        summary = self._last_search_summary
        if not summary or not summary.champion_candidate_id:
            messagebox.showinfo(
                "No champion",
                "Run Search Lab first -- promotion is only available when a run has "
                "at least one candidate that passed every Stage 3 gate.",
            )
            return
        self.promote_champion_btn.config(state="disabled")
        self.search_progress.start(10)
        threading.Thread(target=self._promote_search_champion_pipeline, daemon=True).start()

    def _promote_search_champion_pipeline(self):
        try:
            summary = self._last_search_summary
            self._log_search(
                f"\nPromoting champion candidate {summary.champion_candidate_id} to a full report..."
            )
            promo = promote_champion(
                self._last_search_db_path, summary.run_id, summary.champion_candidate_id,
                self._last_search_df, self._last_search_risk, self._last_search_rules,
                output_dir=str(OUTPUT_DIR / "search" / "champion"),
            )
            self._last_champion_html_path = promo["report_paths"]["html"]
            self.open_champion_report_btn.config(state="normal")
            self._log_search("Champion report written to:")
            for k, p in promo["report_paths"].items():
                self._log_search(f"  {k}: {p}")
        except Exception:
            self._log_search("\nChampion promotion failed:\n" + traceback.format_exc())
        finally:
            self.promote_champion_btn.config(state="normal")
            self.search_progress.stop()

    def _search_auto_build_ensemble(self):
        """Desktop parity for the web app's Search Lab job page
        "Auto-Build Ensemble from this leaderboard" button -- see
        app.web.server.search_job_auto_ensemble's own comment for why
        this exists (app.ensemble.auto_builder.build_diversified_ensemble
        had no caller anywhere in the app before that). Same call, using
        this desktop tab's own self._last_search_* state instead of a web
        job dict."""
        if self._last_search_summary is None or self._last_search_db_path is None:
            messagebox.showwarning("Auto-Build Ensemble", "Run a search on this tab first -- no finished run to build from yet.")
            return
        self.auto_ensemble_btn.config(state="disabled", text="BUILDING ENSEMBLE...")
        self.root.update_idletasks()

        def run():
            try:
                from app.search.results_db import ResultsDB

                with ResultsDB(self._last_search_db_path) as db:
                    records = db.leaderboard(self._last_search_summary.run_id, stage="stage3", top_n=50, only_passed=False)
                if not records:
                    raise AutoEnsembleError("No stage-3 candidates on this run's leaderboard to build an ensemble from.")
                result = build_diversified_ensemble(
                    self._last_search_df, records, self._last_search_risk, prop_rules=self._last_search_rules,
                    mc_config=MonteCarloConfig(n_simulations=3000),
                    min_legs=3, max_legs=5, initial_balance=self._last_search_risk.initial_balance,
                )
                summary_dict = result.to_summary_dict()
                ok, message = True, summary_dict
            except AutoEnsembleError as exc:
                ok, message = False, str(exc)
            except Exception as exc:  # noqa: BLE001
                ok, message = False, f"Unexpected error: {exc}"

            def _finish():
                self.auto_ensemble_btn.config(state="normal", text="AUTO-BUILD ENSEMBLE FROM LEADERBOARD")
                if not ok:
                    messagebox.showerror("Auto-Build Ensemble", message)
                    return
                win = Toplevel(self.root)
                win.title("Auto-Built Ensemble")
                win.configure(bg=PANEL)
                win.geometry("640x480")
                text = Text(win, bg=PANEL_3, fg=TEXT, font=(MONO, 10), relief="flat", bd=0, wrap="word")
                text.pack(fill="both", expand=True, padx=12, pady=12)
                legs = message.get("basket_names", [])
                families = message.get("basket_families", [])
                lines = [f"Status: {message.get('status', '?')}\n", f"{len(legs)}-leg diversified ensemble\n", "-" * 40 + "\n"]
                for name, fam in zip(legs, families):
                    lines.append(f"  {name}  ({fam})\n")
                if message.get("notes"):
                    lines.append("\nNotes:\n")
                    for note in message["notes"]:
                        lines.append(f"  - {note}\n")
                if message.get("portfolio"):
                    lines.append("\nPortfolio result:\n")
                    for k, v in message["portfolio"].items():
                        lines.append(f"  {k}: {v}\n")
                text.insert("1.0", "".join(lines))
                text.config(state="disabled")
            # Direct call rather than self.root.after(0, _finish) -- this
            # codebase's own established convention for background-thread
            # UI updates elsewhere (e.g. RunContextPanel._set_alpaca_status)
            # already does this, and self.root.after() from a non-main
            # thread turned out to be genuinely flaky in testing (Python's
            # tkinter raises "main thread is not in main loop" from a
            # background thread under some timing conditions, silently
            # swallowed by a bare except here, which would leave a button
            # stuck on its "in progress" text forever with no popup/result
            # and no error shown -- confirmed and reproduced this exact
            # failure directly before switching every new background-
            # thread callback in this session to this direct-call fix).
            _finish()

        threading.Thread(target=run, daemon=True).start()

    # -----------------------------------------------------------------------
    # Multi-Instrument Search Lab -- desktop parity with the web app's
    # /search/multi-instrument page: the SAME search space (single-
    # strategy mode, built from the strategy configured on Step 01) run
    # CONCURRENTLY across several instrument/timeframe datasets, via
    # app.orchestration.multi_instrument_search.run_multi_instrument_search.
    # -----------------------------------------------------------------------

    def _build_search_multi_tab(self):
        f = self._scrollable(self.tab_search_multi)

        self._page_header(
            f,
            "OPTIMIZE / Multi-Instrument Search Lab",
            "\u25a3 Multi-Instrument Search Lab",
            "Runs Search Lab's own discover-then-validate pipeline CONCURRENTLY across "
            "several instrument/timeframe datasets -- select 2 or more below (Ctrl/Cmd-"
            "click or Shift-click). Uses single-strategy mode (the exact strategy "
            "configured on Step 01, no family/grid expansion) against each dataset; each "
            "job gets its own results database under data/evolution/search/multi_instrument/ "
            "and its own leaderboard.",
        )

        self.searchmulti_context = RunContextPanel(self, "Multi-Instrument Search Lab")
        self.searchmulti_context.build(f)

        settings = self._section(
            f, "Search settings (applied identically to every instrument)",
            emphasize=True,
        )
        self.searchmulti_workers = LabeledEntry(settings, "Total worker budget (split across concurrent instruments)", 4)
        self.searchmulti_max_concurrent_instruments = LabeledEntry(settings, "Max instruments run at once", 2)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN ACROSS ALL SELECTED INSTRUMENTS", self._search_multi_run_clicked, primary=True).pack(side="left")

        self.searchmulti_progress = NeuralProgress(f)
        self.searchmulti_progress.pack(fill="x", padx=24, pady=(2, 10))

        results_section = self._section(f, "Per-instrument results", "One line per instrument/timeframe.")
        self.searchmulti_results_listbox = Listbox(
            results_section, height=8, exportselection=False, bg=PANEL_3, fg=TEXT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.searchmulti_results_listbox.pack(fill="x", padx=18, pady=(2, 12))
        self._bind_isolated_wheel(self.searchmulti_results_listbox)

        best_section = self._section(f, "Best across all instruments", "Filled in once the run completes.")
        self.searchmulti_best_label = Label(
            best_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=900, anchor="w",
        )
        self.searchmulti_best_label.pack(anchor="w", fill="x", padx=18, pady=(2, 12))

        output_section = self._section(f, "Multi-Instrument Search Lab output", "Live progress log, one line per instrument.")
        _sm_output_frame = Frame(output_section, bg=PANEL)
        self.searchmulti_output = Text(
            _sm_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _sm_output_scroll = ttk.Scrollbar(
            _sm_output_frame, orient="vertical", command=self.searchmulti_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.searchmulti_output.configure(yscrollcommand=_sm_output_scroll.set)
        self.searchmulti_output.pack(side="left", fill="both", expand=True)
        _sm_output_scroll.pack(side="right", fill="y")
        _sm_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.searchmulti_output)

    def _log_search_multi(self, label: str, msg: str):
        def _write():
            self.searchmulti_output.insert(END, f"[{label}] {msg}\n")
            self.searchmulti_output.see(END)
        try:
            self.root.after(0, _write)
        except Exception:
            pass

    def _search_multi_run_clicked(self):
        if len(self.searchmulti_context.csv_paths) < 2:
            messagebox.showwarning(
                "Select at least 2 datasets",
                "Select 2 or more datasets above (Ctrl/Cmd-click or Shift-click) -- with only "
                "1 selected, use the regular Search Lab tab instead.",
            )
            return
        if not self._try_start_heavy_job(JOB_MULTI_INSTRUMENT_SEARCH):
            return
        self.searchmulti_output.delete("1.0", END)
        self.searchmulti_results_listbox.delete(0, END)
        self.searchmulti_best_label.config(text="Running...", fg=TEXT_DIM)
        self.searchmulti_progress.start(10)
        threading.Thread(target=self._search_multi_run_pipeline, daemon=True).start()

    def _search_multi_run_pipeline(self):
        from app.orchestration.multi_instrument_search import (
            best_result_across_instruments, run_multi_instrument_search,
        )

        try:
            jobs = []
            for path in self.searchmulti_context.csv_paths:
                instrument, timeframe = self._misr_instrument_label_for_path(path)
                jobs.append(InstrumentJob(instrument=instrument, timeframe=timeframe, csv_path=path))

            risk = self.searchmulti_context.build_risk_config()
            rules = self.searchmulti_context.build_prop_rules()
            strategy = self._build_strategy()
            space = generate_search_space(mode="single", strategy=strategy)
            stage_cfg = self._build_search_stage_config()
            stage_cfg = dataclasses.replace(stage_cfg, workers=self.searchmulti_workers.get_int(4))
            max_concurrent = self.searchmulti_max_concurrent_instruments.get_int(2)

            self._log_search_multi(
                "multi-instrument",
                f"Starting Search Lab on {len(jobs)} instrument/timeframe target(s): "
                + ", ".join(f"{j.instrument}/{j.timeframe}" for j in jobs),
            )
            results = run_multi_instrument_search(
                jobs, space, risk, rules, stage_cfg, OUTPUT_DIR / "search" / "multi_instrument",
                max_concurrent_instruments=max_concurrent, progress_cb=self._log_search_multi,
            )

            self.searchmulti_results_listbox.delete(0, END)
            for label, res in sorted(results.items()):
                if res.error:
                    self.searchmulti_results_listbox.insert(END, f"[FAILED] {label} -- {res.error.splitlines()[0]}")
                    continue
                s = res.summary
                self.searchmulti_results_listbox.insert(
                    END, f"[OK] {label} -- {s.stage3_survivors} Stage 3 survivor(s), "
                         f"{s.stage1_survivors} Stage 1 survivor(s)",
                )

            best = best_result_across_instruments(results)
            if best is not None and best.summary is not None and best.summary.leaderboard:
                top = best.summary.leaderboard[0]
                self.searchmulti_best_label.config(
                    text=f"BEST: {best.label} -- {top.candidate_id}  eval pass {top.eval_pass_probability:.1f}%",
                    fg=GREEN,
                )
            else:
                self.searchmulti_best_label.config(text="No instrument produced a leaderboard entry.", fg=RED)

            n_ok = sum(1 for r in results.values() if r.summary is not None)
            self._log_search_multi("multi-instrument", f"\nDone -- {n_ok}/{len(jobs)} instrument(s) succeeded.")
        except Exception as exc:
            self._log_search_multi("multi-instrument", "\nUnexpected error:\n" + traceback.format_exc())
            self.searchmulti_best_label.config(text="Failed -- see log.", fg=RED)
            log_crash("Multi-Instrument Search Lab", exc=exc)
        finally:
            self.searchmulti_progress.stop()
            self._release_heavy_job(JOB_MULTI_INSTRUMENT_SEARCH)
