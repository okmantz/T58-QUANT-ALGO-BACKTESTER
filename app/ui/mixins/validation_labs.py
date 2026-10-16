"""Regime matrix, family diversity, walk-forward, CPCV, PBO, sensitivity, parameter robustness."""
from __future__ import annotations

import random
import threading
import traceback
import webbrowser
from app.optimize.parameter_space import RefinementError, apply_genome, extract_genome
from app.optimize.refinement import FITNESS_METRICS, RefinementConfig
from app.orchestration.resource_guard import (
    JOB_CPCV,
    JOB_PARAMETER_ROBUSTNESS,
    JOB_REGIME_MATRIX,
    JOB_SENSITIVITY,
    JOB_WFO,
)
from app.reports import strategy_state
from app.reports.validation_reports import (
    generate_cpcv_report,
    generate_pbo_report,
    generate_sensitivity_report,
    generate_walk_forward_report,
)
from app.strategy.base import StrategyError
from app.strategy.library import record_validation_result
from app.validation.cpcv import CPCVError, compute_pbo, run_cpcv
from app.validation.parameter_robustness import compute_parameter_robustness
from app.validation.regime_matrix import run_regime_matrix
from app.validation.sensitivity import compute_1d_sensitivity, compute_2d_heatmap, list_tunable_parameters
from app.validation.walk_forward_opt import run_walk_forward_optimization
from pathlib import Path
from tkinter import END, Frame, Label, Text, messagebox, ttk
from app.ui.common import OUTPUT_DIR, _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCombo, LabeledEntry
from app.ui.widgets.cards import NeuralProgress
from app.ui.widgets.run_context import RunContextPanel
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
BORDER = ""
TEXT = ""
TEXT_DIM = ""
LOG_BG = ""
GREEN = ""
RED = ""
AMBER = ""
theme.bind(globals())


class ValidationLabsMixin:
    """Regime matrix, family diversity, walk-forward, CPCV, PBO, sensitivity, parameter robustness."""


    # -----------------------------------------------------------------------
    # Regime Survival Matrix -- classifies every bar on trend/volatility/
    # session/environment and attributes the current strategy's own trades
    # to whichever regime was active at entry (see
    # app.validation.regime_matrix). Reuses the SAME data/strategy/risk this
    # strategy would run with on the Run & Report tab, via the Validation
    # Lab's shared _load_df_for_page/_build_strategy/_build_risk_config.
    # -----------------------------------------------------------------------

    def _build_regime_matrix_tab(self):
        f = self._scrollable(self.tab_regime_matrix)

        self._page_header(
            f,
            "VALIDATE / Regime Survival Matrix",
            "Regime Survival Matrix",
            "Classifies every bar on trend, volatility, session, and market environment, "
            "then attributes THIS strategy's own trades to whichever regime was active at "
            "entry -- so you can see exactly which market conditions to gate the strategy "
            "off in, not just one aggregate backtest number. Uses the strategy configured "
            "on Step 01 -- but its own market data and risk settings below are "
            "self-contained to this tab, just like the web app.",
        )

        self.regime_context = RunContextPanel(self, "Regime Survival Matrix")
        self.regime_context.build(f)

        section = self._section(
            f, "Matrix dimensions to cross",
            "Trend, volatility, session, and environment are all classified either way -- "
            "the other two always show up as single-dimension breakdowns below the main table.",
            emphasize=True,
        )
        self.rm_dimension_a = LabeledCombo(
            section, "Dimension A", ["volatility", "trend", "session", "environment"], default="volatility",
        )
        self.rm_dimension_b = LabeledCombo(
            section, "Dimension B", ["environment", "trend", "volatility", "session"], default="environment",
        )

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN REGIME SURVIVAL MATRIX", self._regime_matrix_clicked, primary=True).pack(side="left")

        output_section = self._section(f, "Result", "Plain-text matrix, single-dimension breakdowns, and any recommended OFF regimes.")
        _rm_output_frame = Frame(output_section, bg=PANEL)
        self.rm_output = Text(
            _rm_output_frame, height=26, wrap="word", bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        _rm_output_scroll = ttk.Scrollbar(
            _rm_output_frame, orient="vertical", command=self.rm_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.rm_output.configure(yscrollcommand=_rm_output_scroll.set)
        self.rm_output.pack(side="left", fill="both", expand=True)
        _rm_output_scroll.pack(side="right", fill="y")
        _rm_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.rm_output)

    def _regime_matrix_clicked(self):
        if not self._try_start_heavy_job(JOB_REGIME_MATRIX):
            return
        self.rm_output.delete("1.0", END)
        threading.Thread(target=self._run_regime_matrix_pipeline, daemon=True).start()

    def _run_regime_matrix_pipeline(self):
        def log(msg):
            self.rm_output.insert(END, msg + "\n")
            self.rm_output.see(END)
            self.root.update_idletasks()

        try:
            dim_a = self.rm_dimension_a.get_str()
            dim_b = self.rm_dimension_b.get_str()
            if dim_a == dim_b:
                log("Pick two DIFFERENT dimensions to cross for the primary matrix.")
                return

            df = self.regime_context.load_dataframe(log)
            if df is None:
                return

            log("Building strategy...")
            strategy = self._build_strategy()
            risk = self.regime_context.build_risk_config()

            log(f"Classifying regimes and attributing trades ({dim_a} x {dim_b})...")
            result = run_regime_matrix(df, strategy, risk, dimensions=(dim_a, dim_b))
            if result is None:
                log("Not enough bars in this dataset to classify regimes reliably -- try a longer history.")
                return

            log("")
            log(result.render_table())
            for dim_name, cells in result.single_dimension.items():
                log("")
                log(f"-- {dim_name.title()} breakdown (single dimension) --")
                for c in cells:
                    log(f"  {c.label:<20} trades={c.n_trades:<6} win%={c.win_rate:>5.1f} pf={c.profit_factor:>5.2f}")
            for n in result.notes:
                log("")
                log(f"Note: {n}")
            # Matches app.web.server's own /regime-matrix route's record_validation call.
            _active_lib = getattr(self, "_active_library_strategy", None)
            strategy_state.record_validation(
                getattr(strategy, "name", "Strategy"), self.regime_context.instrument_label(), "regime_matrix",
                passed=None, summary=f"{len(result.cells)} regime cell(s) analyzed",
                library_type=(_active_lib[0] if _active_lib else ""), library_filename=(_active_lib[1] if _active_lib else ""),
            )
        except StrategyError as exc:
            log(f"Strategy error: {exc}")
        except Exception:
            log("Unexpected error:\n" + traceback.format_exc())
        finally:
            self._release_heavy_job(JOB_REGIME_MATRIX)

    # -----------------------------------------------------------------------
    # Strategy Family Diversity -- reads the most recently completed Search
    # Lab run (this session) and ranks its candidates' hypothesis FAMILIES,
    # not just individual candidates, per app.search.family_diversity.
    # -----------------------------------------------------------------------

    def _build_family_diversity_tab(self):
        f = self._scrollable(self.tab_family_diversity)

        self._page_header(
            f,
            "CHAMPION / Family Diversity",
            "Strategy Family Diversity",
            "Groups the most recent Search Lab run's candidates by hypothesis family "
            "(trend following, breakout, mean reversion, liquidity sweep, momentum, VWAP, "
            "opening range, market structure, volatility expansion/contraction, pullback, "
            "statistical arbitrage, relative strength, regime switching) and ranks the "
            "FAMILIES themselves -- run a search on the Search Lab tab first.",
        )

        section = self._section(f, "Which stage to summarize", "", emphasize=True)
        self.fd_stage = LabeledCombo(section, "Stage", ["stage1", "stage2", "stage3"], default="stage1")

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "SHOW FAMILY BREAKDOWN", self._family_diversity_clicked, primary=True).pack(side="left")

        output_section = self._section(f, "Result", "")
        _fd_output_frame = Frame(output_section, bg=PANEL)
        self.fd_output = Text(
            _fd_output_frame, height=22, wrap="word", bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        _fd_output_scroll = ttk.Scrollbar(
            _fd_output_frame, orient="vertical", command=self.fd_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.fd_output.configure(yscrollcommand=_fd_output_scroll.set)
        self.fd_output.pack(side="left", fill="both", expand=True)
        _fd_output_scroll.pack(side="right", fill="y")
        _fd_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.fd_output)

    def _family_diversity_clicked(self):
        self.fd_output.delete("1.0", END)
        summary = getattr(self, "_last_search_summary", None)
        db_path = getattr(self, "_last_search_db_path", None)
        if not summary or not db_path:
            self.fd_output.insert(END, "Run a search on the Search Lab tab first -- there's no completed run to summarize yet.\n")
            return
        try:
            from app.search.family_diversity import render_family_report, summarize_family_performance
            from app.search.results_db import ResultsDB

            stage = self.fd_stage.get_str()
            with ResultsDB(db_path) as db:
                records = db.leaderboard(summary.run_id, stage=stage, top_n=5000)
            if not records:
                self.fd_output.insert(END, f"No '{stage}' candidates found for the most recent run -- try a different stage.\n")
                return
            summaries = summarize_family_performance(records)
            self.fd_output.insert(END, render_family_report(summaries))
        except Exception:
            self.fd_output.insert(END, "Unexpected error:\n" + traceback.format_exc())

    # -----------------------------------------------------------------------
    # Tab 8 — Walk-Forward Optimization
    # -----------------------------------------------------------------------

    def _build_wfo_tab(self):
        f = self._scrollable(self.tab_wfo)

        self._page_header(
            f,
            "VALIDATE / Walk-Forward Opt",
            "Walk-Forward Optimization",
            "Re-optimizes this strategy fresh on each rolling or anchored fold's "
            "training window using a small GA search, applies the winning "
            "configuration UNCHANGED to that fold's held-out test window, and "
            "chains every fold's out-of-sample trades into one continuous equity "
            "curve. This is the number to trust over a single in-sample backtest "
            "-- it never lets a fold's optimizer see the data it will be judged on. "
            "Uses the strategy configured on Step 01 -- but its own market data, "
            "prop rules, and risk settings below are self-contained to this tab, "
            "just like the web app.",
        )

        self.wfo_context = RunContextPanel(self, "Walk-Forward Opt")
        self.wfo_context.build(f)

        settings = self._section(
            f, "Fold settings",
            "'Rolling' slides a fixed-size train window forward each fold (better if "
            "you suspect a regime-dependent edge). 'Anchored' always starts training "
            "at bar 0 and grows it (better if you believe the edge is stable "
            "over time, so more data only helps).",
            emphasize=True,
        )
        self.wfo_window_mode = LabeledCombo(settings, "Window mode", ["rolling", "anchored"], "rolling")
        self.wfo_folds = LabeledEntry(settings, "Number of folds", 5)
        self.wfo_train_frac = LabeledEntry(settings, "Train fraction per fold (rolling mode)", 0.6)

        ga_settings = self._section(
            f, "Per-fold GA search settings",
            "A fresh, smaller GA search runs once per fold -- keep these modest, "
            "since the total work is population x generations x folds.",
        )
        self._wfo_metric_labels = list(FITNESS_METRICS.values())
        self._wfo_metric_label_to_key = {v: k for k, v in FITNESS_METRICS.items()}
        self.wfo_metric = LabeledCombo(
            ga_settings, "Fitness metric", self._wfo_metric_labels, FITNESS_METRICS["eval_pass_probability"],
        )
        self.wfo_population = LabeledEntry(ga_settings, "Population size per fold", 8)
        self.wfo_generations = LabeledEntry(ga_settings, "Generations per fold", 3)
        self.wfo_seed = LabeledEntry(ga_settings, "Random seed", 42)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN WALK-FORWARD OPTIMIZATION", self._wfo_run_clicked, primary=True).pack(side="left")
        self.open_wfo_report_btn = self._button(button_row, "OPEN REPORT", self._open_wfo_report)
        self.open_wfo_report_btn.config(state="disabled")
        self.open_wfo_report_btn.pack(side="left", padx=8)

        self.wfo_progress = NeuralProgress(f)
        self.wfo_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Walk-forward output", "Live progress log.")
        _wfo_output_frame = Frame(output_section, bg=PANEL)
        self.wfo_output = Text(
            _wfo_output_frame, height=18, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _wfo_output_scroll = ttk.Scrollbar(
            _wfo_output_frame, orient="vertical", command=self.wfo_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.wfo_output.configure(yscrollcommand=_wfo_output_scroll.set)
        self.wfo_output.pack(side="left", fill="both", expand=True)
        _wfo_output_scroll.pack(side="right", fill="y")
        _wfo_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.wfo_output)

        self._last_wfo_html_path = None

    def _log_wfo(self, msg: str):
        self.wfo_output.insert(END, msg + "\n")
        self.wfo_output.see(END)
        self.root.update_idletasks()

    def _open_wfo_report(self):
        if self._last_wfo_html_path:
            webbrowser.open(f"file://{self._last_wfo_html_path.resolve()}")

    def _wfo_run_clicked(self):
        if not self.wfo_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        if not self._try_start_heavy_job(JOB_WFO):
            return
        self.wfo_output.delete("1.0", END)
        self.wfo_progress.start(10)
        threading.Thread(target=self._wfo_run_pipeline, daemon=True).start()

    def _wfo_run_pipeline(self):
        try:
            df = self.wfo_context.load_dataframe(self._log_wfo)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self.wfo_context.build_risk_config()
            rules = self.wfo_context.build_prop_rules()
            mc_cfg = self._validation_mc_config()

            metric_key = self._wfo_metric_label_to_key.get(self.wfo_metric.get_str(), "eval_pass_probability")
            refine_cfg = RefinementConfig(
                population_size=self.wfo_population.get_int(8),
                generations=self.wfo_generations.get_int(3),
                fitness_metric=metric_key,
                random_seed=self.wfo_seed.get_int(42),
            )

            self._log_wfo("Starting walk-forward optimization...")
            result = run_walk_forward_optimization(
                df, strategy, risk, rules, mc_cfg,
                n_folds=self.wfo_folds.get_int(5),
                window_mode=self.wfo_window_mode.get_str(),
                train_frac=self.wfo_train_frac.get_float(0.6),
                refine_cfg=refine_cfg,
                random_seed=self.wfo_seed.get_int(42),
                progress_cb=self._log_wfo,
            )
            paths = generate_walk_forward_report(OUTPUT_DIR / "walk_forward", result)
            self._last_wfo_html_path = paths["html"]
            self.open_wfo_report_btn.config(state="normal")
            self._log_wfo("\nDone. Walk-forward optimization report written to:")
            for k, p in paths.items():
                self._log_wfo(f"  {k}: {p}")
            eff = getattr(result, "out_of_sample_efficiency", None)
            summary = f"OOS efficiency {eff:.2f}" if eff is not None else f"{self.wfo_folds.get_int(5)} folds completed"
            # No strict pass/fail verdict is computed by this tool -- passed=None
            # records that it *ran*, matching app.web.server's own /walk-forward-opt route.
            _active_lib = getattr(self, "_active_library_strategy", None)
            strategy_state.record_validation(
                getattr(strategy, "name", "Strategy"), self.wfo_context.instrument_label(), "wfo",
                passed=None, summary=summary, report_html=f"file://{self._last_wfo_html_path.resolve()}",
                library_type=(_active_lib[0] if _active_lib else ""), library_filename=(_active_lib[1] if _active_lib else ""),
            )
        except StrategyError as exc:
            self._log_wfo(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_wfo(f"\nWalk-forward optimization error: {exc}")
        except Exception:
            self._log_wfo("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.wfo_progress.stop()
            self._release_heavy_job(JOB_WFO)

    # -----------------------------------------------------------------------
    # Tab 9 — CPCV / PBO
    # -----------------------------------------------------------------------

    def _build_cpcv_tab(self):
        f = self._scrollable(self.tab_cpcv)

        self._page_header(
            f,
            "VALIDATE / CPCV",
            "Combinatorial Purged Cross-Validation",
            "CPCV stress-tests this strategy across many different combinatorial "
            "train/test partitions of the same data, instead of just one holdout "
            "split. Uses the strategy configured on Step 01 -- but its own market "
            "data, prop rules, and risk settings below are self-contained to this "
            "tab, just like the web app. See VALIDATE / PBO (its own tab) for "
            "Probability of Backtest Overfitting, which checks a small POOL of "
            "candidate configurations instead of validating just this one.",
        )

        self.cpcv_context = RunContextPanel(self, "CPCV")
        self.cpcv_context.build(f)

        cpcv_settings = self._section(
            f, "CPCV settings (single strategy)",
            "Splits the data into N groups and evaluates every combination of "
            "k groups as the test set, with the rest as train.",
            emphasize=True,
        )
        self.cpcv_n_groups = LabeledEntry(cpcv_settings, "Number of groups (N)", 6)
        self.cpcv_n_test_groups = LabeledEntry(cpcv_settings, "Test groups per path (k)", 2)
        self.cpcv_metric = LabeledEntry(cpcv_settings, "Metric (eval_pass_probability recommended; or profit_factor, sharpe_ratio, net_profit)", "eval_pass_probability")
        self.cpcv_max_paths = LabeledEntry(cpcv_settings, "Max combinatorial paths to evaluate", 30)

        cpcv_btn_row = Frame(f, bg=BG)
        cpcv_btn_row.pack(fill="x", padx=24, pady=(4, 10))
        self._button(cpcv_btn_row, "RUN CPCV", self._cpcv_run_clicked, primary=True).pack(side="left")
        self.open_cpcv_report_btn = self._button(cpcv_btn_row, "OPEN CPCV REPORT", self._open_cpcv_report)
        self.open_cpcv_report_btn.config(state="disabled")
        self.open_cpcv_report_btn.pack(side="left", padx=8)

        self.cpcv_progress = NeuralProgress(f)
        self.cpcv_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "CPCV output", "Live progress log.")
        _cpcv_output_frame = Frame(output_section, bg=PANEL)
        self.cpcv_output = Text(
            _cpcv_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _cpcv_output_scroll = ttk.Scrollbar(
            _cpcv_output_frame, orient="vertical", command=self.cpcv_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.cpcv_output.configure(yscrollcommand=_cpcv_output_scroll.set)
        self.cpcv_output.pack(side="left", fill="both", expand=True)
        _cpcv_output_scroll.pack(side="right", fill="y")
        _cpcv_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.cpcv_output)

        self._last_cpcv_html_path = None

    # -----------------------------------------------------------------------
    # PBO (Probability of Backtest Overfitting) -- split out from the old
    # combined "CPCV / PBO" tab into its own screen (Sep 2026 VALIDATE
    # reorder), with its own RunContextPanel, progress bar, and output log
    # instead of sharing CPCV's. Nothing about compute_pbo() or the report
    # generator changed -- only which widgets this UI reads from/writes to.
    # -----------------------------------------------------------------------

    def _build_pbo_tab(self):
        f = self._scrollable(self.tab_pbo)

        self._page_header(
            f,
            "VALIDATE / PBO",
            "Probability of Backtest Overfitting",
            "Checks a small POOL of candidate configurations (this strategy plus "
            "a few automatically perturbed variants) and reports the probability "
            "that whichever one looks best in-sample is really just noise. Uses "
            "the strategy configured on Step 01 -- but its own market data, prop "
            "rules, and risk settings below are self-contained to this tab, just "
            "like the web app. See VALIDATE / CPCV (its own tab) for a single-"
            "strategy combinatorial train/test stress test instead.",
        )

        self.pbo_context = RunContextPanel(self, "PBO")
        self.pbo_context.build(f)

        pbo_settings = self._section(
            f, "PBO settings (candidate pool)",
            "The pool is this strategy's own configuration plus N-1 variants "
            "with its numeric parameters randomly perturbed -- a quick, "
            "self-contained way to check whether the search process itself is "
            "trustworthy. For a genuine multi-strategy PBO, use the CLI "
            "(--pbo) or call app.validation.cpcv.compute_pbo() directly with a "
            "Search Lab leaderboard slice.",
            emphasize=True,
        )
        self.pbo_n_groups = LabeledEntry(pbo_settings, "Number of groups (N)", 6)
        self.pbo_n_test_groups = LabeledEntry(pbo_settings, "Test groups per path (k)", 2)
        self.pbo_metric = LabeledEntry(pbo_settings, "Metric to rank candidates by", "sharpe_ratio")
        self.pbo_max_paths = LabeledEntry(pbo_settings, "Max combinatorial paths to evaluate", 30)
        self.pbo_n_candidates = LabeledEntry(pbo_settings, "Number of candidates (baseline + perturbed)", 5)
        self.pbo_seed = LabeledEntry(pbo_settings, "Random seed (candidate perturbation)", 42)

        pbo_btn_row = Frame(f, bg=BG)
        pbo_btn_row.pack(fill="x", padx=24, pady=(4, 10))
        self._button(pbo_btn_row, "RUN PBO", self._pbo_run_clicked, primary=True).pack(side="left")
        self.open_pbo_report_btn = self._button(pbo_btn_row, "OPEN PBO REPORT", self._open_pbo_report)
        self.open_pbo_report_btn.config(state="disabled")
        self.open_pbo_report_btn.pack(side="left", padx=8)

        self.pbo_progress = NeuralProgress(f)
        self.pbo_progress.pack(fill="x", padx=24, pady=(2, 10))

        pbo_output_section = self._section(f, "PBO output", "Live progress log.")
        _pbo_output_frame = Frame(pbo_output_section, bg=PANEL)
        self.pbo_output = Text(
            _pbo_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _pbo_output_scroll = ttk.Scrollbar(
            _pbo_output_frame, orient="vertical", command=self.pbo_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.pbo_output.configure(yscrollcommand=_pbo_output_scroll.set)
        self.pbo_output.pack(side="left", fill="both", expand=True)
        _pbo_output_scroll.pack(side="right", fill="y")
        _pbo_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.pbo_output)

        self._last_pbo_html_path = None

    def _log_cpcv(self, msg: str):
        self.cpcv_output.insert(END, msg + "\n")
        self.cpcv_output.see(END)
        self.root.update_idletasks()

    def _log_pbo(self, msg: str):
        self.pbo_output.insert(END, msg + "\n")
        self.pbo_output.see(END)
        self.root.update_idletasks()

    def _open_cpcv_report(self):
        if self._last_cpcv_html_path:
            webbrowser.open(f"file://{self._last_cpcv_html_path.resolve()}")

    def _open_pbo_report(self):
        if self._last_pbo_html_path:
            webbrowser.open(f"file://{self._last_pbo_html_path.resolve()}")

    def _cpcv_run_clicked(self):
        if not self.cpcv_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        if not self._try_start_heavy_job(JOB_CPCV):
            return
        self.cpcv_output.delete("1.0", END)
        self.cpcv_progress.start(10)
        threading.Thread(target=self._cpcv_run_pipeline, daemon=True).start()

    def _cpcv_run_pipeline(self):
        try:
            df = self.cpcv_context.load_dataframe(self._log_cpcv)
            if df is None:
                return
            risk = self.cpcv_context.build_risk_config()
            prop_rules = self.cpcv_context.build_prop_rules()
            # A fresh Strategy instance per call, since some strategy sources
            # cache state keyed to the data they last saw.
            strategy_builder = self._build_strategy

            self._log_cpcv("Running CPCV...")
            result = run_cpcv(
                df, strategy_builder, risk,
                n_groups=self.cpcv_n_groups.get_int(6),
                n_test_groups=self.cpcv_n_test_groups.get_int(2),
                metric=self.cpcv_metric.get_str().strip() or "eval_pass_probability",
                max_paths=self.cpcv_max_paths.get_int(30),
                prop_rules=prop_rules,
            )
            paths = generate_cpcv_report(OUTPUT_DIR / "cpcv", result)
            self._last_cpcv_html_path = paths["html"]
            self.open_cpcv_report_btn.config(state="normal")
            self._log_cpcv(f"  Mean OOS {result.metric}: {result.mean_oos_metric:.3f}  (robust: {result.is_robust})")
            self._log_cpcv("\nDone. CPCV report written to:")
            for k, p in paths.items():
                self._log_cpcv(f"  {k}: {p}")
            # Matches app.web.server's own /cpcv route's record_validation call.
            _active_lib = getattr(self, "_active_library_strategy", None)
            strategy_state.record_validation(
                getattr(strategy_builder(), "name", "Strategy"), self.cpcv_context.instrument_label(), "cpcv",
                passed=bool(result.is_robust), summary=f"{result.n_paths} paths evaluated",
                report_html=f"file://{self._last_cpcv_html_path.resolve()}",
                library_type=(_active_lib[0] if _active_lib else ""), library_filename=(_active_lib[1] if _active_lib else ""),
            )
            # UPGRADE (library_ref): the web app's own /cpcv route also
            # stamps last_validation onto the Strategy Library entry (see
            # app.web.server._run_cpcv_job), which is what
            # compute_pipeline_progress uses to advance a strategy's
            # Create/Test/Optimize/Validate/Champion Check/Ready funnel to
            # "Validate". This desktop tab previously only recorded into
            # the separate validation-checklist store above -- a strategy
            # run through this tab never showed "Validate" done in the
            # Strategy Library, even though the exact same CPCV run did
            # when triggered from the web app. self._active_library_strategy
            # is the same (strategy_type, filename) pointer Run & Report's
            # own record_backtest_result call already relies on (set
            # whenever the currently-configured Step 01 strategy was
            # loaded from the library) -- None for a one-off/uploaded
            # strategy, exactly like every other library_ref call site.
            active_lib_strategy = getattr(self, "_active_library_strategy", None)
            if active_lib_strategy:
                lib_mode, lib_filename = active_lib_strategy
                try:
                    record_validation_result(lib_mode, lib_filename, {
                        "method": "cpcv", "n_paths": result.n_paths,
                        "is_robust": bool(result.is_robust),
                        "efficiency": getattr(result, "mean_oos_metric", None),
                        "report_html": f"file://{self._last_cpcv_html_path.resolve()}",
                    })
                except (FileNotFoundError, ValueError):
                    pass  # strategy was renamed/deleted mid-run -- not worth failing the run over
        except StrategyError as exc:
            self._log_cpcv(f"\nStrategy error: {exc}")
        except CPCVError as exc:
            self._log_cpcv(f"\nCPCV error: {exc}")
        except Exception:
            self._log_cpcv("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.cpcv_progress.stop()
            self._release_heavy_job(JOB_CPCV)

    def _pbo_run_clicked(self):
        if not self.pbo_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        if not self._try_start_heavy_job(JOB_CPCV):
            return
        self.pbo_output.delete("1.0", END)
        self.pbo_progress.start(10)
        threading.Thread(target=self._pbo_run_pipeline, daemon=True).start()

    def _pbo_run_pipeline(self):
        try:
            df = self.pbo_context.load_dataframe(self._log_pbo)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self.pbo_context.build_risk_config()

            n_candidates = self.pbo_n_candidates.get_int(5)
            specs = [{"source_type": strategy.source_type,
                      "config": dict(strategy.config)} if strategy.source_type == "manual"
                     else {"source_type": strategy.source_type, "code_text": Path(strategy.file_path).read_text(encoding="utf-8"),
                           "code_extension": Path(strategy.file_path).suffix}]

            if strategy.source_type == "manual":
                genes = extract_genome(strategy.config)
                rng = random.Random(self.pbo_seed.get_int(42))
                for _ in range(max(n_candidates - 1, 0)):
                    if not genes:
                        break
                    genome = [max(min(g.base_value + rng.uniform(-0.3, 0.3) * (g.hi - g.lo), g.hi), g.lo) for g in genes]
                    specs.append({"source_type": "manual", "config": apply_genome(strategy.config, genes, genome)})
                if not genes:
                    self._log_pbo(
                        "This strategy has no tunable numeric parameters -- PBO will run "
                        "with a single candidate, which is a degenerate (but still valid) case."
                    )
            else:
                self._log_pbo(
                    f"'{strategy.source_type}' candidate perturbation isn't wired up in the "
                    "desktop UI yet -- running PBO with just this one strategy as the pool "
                    "(degenerate case). Use the CLI or compute_pbo() directly for a real "
                    "multi-candidate code-strategy pool."
                )

            self._log_pbo(f"Running PBO across {len(specs)} candidate(s)...")
            result = compute_pbo(
                df, specs, risk,
                n_groups=self.pbo_n_groups.get_int(6),
                n_test_groups=self.pbo_n_test_groups.get_int(2),
                metric=self.pbo_metric.get_str().strip() or "sharpe_ratio",
                max_paths=self.pbo_max_paths.get_int(30),
            )
            paths = generate_pbo_report(OUTPUT_DIR / "pbo", result)
            self._last_pbo_html_path = paths["html"]
            self.open_pbo_report_btn.config(state="normal")
            self._log_pbo(f"  PBO: {result.pbo * 100:.1f}%")
            self._log_pbo("\nDone. PBO report written to:")
            for k, p in paths.items():
                self._log_pbo(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_pbo(f"\nStrategy error: {exc}")
        except CPCVError as exc:
            self._log_pbo(f"\nPBO error: {exc}")
        except Exception:
            self._log_pbo("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.pbo_progress.stop()
            self._release_heavy_job(JOB_CPCV)

    # -----------------------------------------------------------------------
    # Tab 10 — Parameter Sensitivity
    # -----------------------------------------------------------------------

    def _build_sensitivity_tab(self):
        f = self._scrollable(self.tab_sensitivity)

        self._page_header(
            f,
            "VALIDATE / Sensitivity",
            "Parameter Sensitivity",
            "Sweeps every tunable numeric parameter across +/- a percentage of its "
            "current value, holding everything else fixed, and flags a 'cliff' "
            "wherever the metric drops sharply between adjacent steps -- the sign "
            "of a knife-edge parameter rather than a real, stable plateau. "
            "Optionally also produces a 2D heatmap for a chosen pair of parameters, "
            "since two parameters can interact even when each looks fine alone. Uses "
            "the strategy configured on Step 01 -- but its own market data, prop "
            "rules, and risk settings below are self-contained to this tab, just "
            "like the web app.",
        )

        self.sensitivity_context = RunContextPanel(self, "Sensitivity")
        self.sensitivity_context.build(f)

        settings = self._section(
            f, "Sweep settings",
            "Applies to every tunable numeric parameter this strategy has.",
            emphasize=True,
        )
        self.sens_metric = LabeledEntry(settings, "Metric (e.g. profit_factor, net_profit, sharpe_ratio)", "profit_factor")
        self.sens_pct_range = LabeledEntry(settings, "Sweep range (+/- fraction of current value)", 0.5)
        self.sens_steps = LabeledEntry(settings, "Steps per 1D sweep", 9)

        heatmap_settings = self._section(
            f, "Optional 2D heatmap",
            "Click LIST PARAMETERS to discover this strategy's tunable parameter "
            "labels, then pick two for a heatmap. Leave blank to skip the heatmap "
            "and only run the 1D sweeps above.",
        )
        list_row = Frame(heatmap_settings, bg=PANEL)
        list_row.pack(fill="x", padx=18, pady=(0, 6))
        self._button(list_row, "LIST PARAMETERS", self._sens_list_parameters).pack(side="left")
        self.sens_param_status = Label(
            list_row, text="No parameters listed yet.", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        )
        self.sens_param_status.pack(side="left", padx=10)

        self.sens_heatmap_a = LabeledCombo(heatmap_settings, "Heatmap parameter A", [], "")
        self.sens_heatmap_b = LabeledCombo(heatmap_settings, "Heatmap parameter B", [], "")

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN SENSITIVITY", self._sens_run_clicked, primary=True).pack(side="left")
        self.open_sens_report_btn = self._button(button_row, "OPEN REPORT", self._open_sens_report)
        self.open_sens_report_btn.config(state="disabled")
        self.open_sens_report_btn.pack(side="left", padx=8)

        self.sens_progress = NeuralProgress(f)
        self.sens_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Sensitivity output", "Live progress log.")
        _sens_output_frame = Frame(output_section, bg=PANEL)
        self.sens_output = Text(
            _sens_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _sens_output_scroll = ttk.Scrollbar(
            _sens_output_frame, orient="vertical", command=self.sens_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.sens_output.configure(yscrollcommand=_sens_output_scroll.set)
        self.sens_output.pack(side="left", fill="both", expand=True)
        _sens_output_scroll.pack(side="right", fill="y")
        _sens_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.sens_output)

        self._last_sens_html_path = None

    def _log_sens(self, msg: str):
        self.sens_output.insert(END, msg + "\n")
        self.sens_output.see(END)
        self.root.update_idletasks()

    def _open_sens_report(self):
        if self._last_sens_html_path:
            webbrowser.open(f"file://{self._last_sens_html_path.resolve()}")

    def _sens_list_parameters(self):
        try:
            strategy = self._build_strategy()
            labels = list_tunable_parameters(strategy)
            self.sens_heatmap_a.combo.config(values=labels)
            self.sens_heatmap_b.combo.config(values=labels)
            if labels:
                self.sens_param_status.config(
                    text=f"{len(labels)} parameter(s) found.", fg=GREEN,
                )
            else:
                self.sens_param_status.config(
                    text="This strategy has no tunable numeric parameters.", fg=AMBER,
                )
        except StrategyError as exc:
            messagebox.showerror("Strategy error", str(exc))
        except Exception:
            messagebox.showerror("Error", traceback.format_exc())

    def _sens_run_clicked(self):
        if not self.sensitivity_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        if not self._try_start_heavy_job(JOB_SENSITIVITY):
            return
        self.sens_output.delete("1.0", END)
        self.sens_progress.start(10)
        threading.Thread(target=self._sens_run_pipeline, daemon=True).start()

    def _sens_run_pipeline(self):
        try:
            df = self.sensitivity_context.load_dataframe(self._log_sens)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self.sensitivity_context.build_risk_config()
            rules = self.sensitivity_context.build_prop_rules()
            mc_cfg = self._validation_mc_config(n_simulations=min(self.mc_sims.get_int(10000), 1000))
            metric = self.sens_metric.get_str().strip() or "profit_factor"

            self._log_sens("Running 1D sensitivity sweeps...")
            sweeps = compute_1d_sensitivity(
                df, strategy, risk, rules, mc_cfg, metric=metric,
                pct_range=self.sens_pct_range.get_float(0.5), n_steps=self.sens_steps.get_int(9),
                progress_cb=self._log_sens,
            )
            for r in sweeps:
                flag = " <-- CLIFF" if r.cliff_detected else ""
                self._log_sens(f"  {r.gene_label}: max adjacent-step drop {r.max_pct_drop_between_adjacent_steps:.0f}%{flag}")

            heatmap = None
            a_label, b_label = self.sens_heatmap_a.get_str().strip(), self.sens_heatmap_b.get_str().strip()
            if a_label and b_label:
                self._log_sens(f"Running 2D heatmap for {a_label} x {b_label}...")
                heatmap = compute_2d_heatmap(df, strategy, risk, rules, mc_cfg, a_label, b_label, metric=metric, progress_cb=self._log_sens)

            paths = generate_sensitivity_report(OUTPUT_DIR / "sensitivity", sweeps, heatmap)
            self._last_sens_html_path = paths["html"]
            self.open_sens_report_btn.config(state="normal")
            self._log_sens("\nDone. Sensitivity report written to:")
            for k, p in paths.items():
                self._log_sens(f"  {k}: {p}")
            # This tool is diagnostic, not pass/fail -- passed=None records that it
            # ran, matching app.web.server's own /sensitivity route.
            _active_lib = getattr(self, "_active_library_strategy", None)
            strategy_state.record_validation(
                getattr(strategy, "name", "Strategy"), self.sensitivity_context.instrument_label(), "sensitivity",
                passed=None, summary=f"{len(sweeps)} parameter(s) swept",
                report_html=f"file://{self._last_sens_html_path.resolve()}",
                library_type=(_active_lib[0] if _active_lib else ""), library_filename=(_active_lib[1] if _active_lib else ""),
            )
        except StrategyError as exc:
            self._log_sens(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_sens(f"\nSensitivity error: {exc}")
        except Exception:
            self._log_sens("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.sens_progress.stop()
            self._release_heavy_job(JOB_SENSITIVITY)

    # -----------------------------------------------------------------------
    # Tab 10b — Parameter Stability / Robustness Map
    # -----------------------------------------------------------------------

    def _build_param_robustness_tab(self):
        f = self._scrollable(self.tab_param_robustness)

        self._page_header(
            f,
            "VALIDATE / Robustness Map",
            "Parameter Stability / Robustness Map",
            "Answers \"did I discover a robust edge, or the exact historical combination that "
            "happened to work?\" Sweeps every tunable parameter around its current value, "
            "heatmaps the most sensitive pair, and rolls it all into one 0-100 Parameter "
            "Robustness Score with a ROBUST / WATCH / FRAGILE verdict -- reuses the exact same "
            "app.validation.parameter_robustness engine as the web app's own Robustness Map page. "
            "Uses the strategy configured on Step 01 -- but its own market data, prop rules, "
            "and risk settings below are self-contained to this tab, just like the web app.",
        )

        self.paramrob_context = RunContextPanel(self, "Parameter Robustness")
        self.paramrob_context.build(f)

        settings = self._section(
            f, "Robustness map settings",
            "Every tunable numeric parameter is swept, then the most sensitive pair gets a "
            "full 2D heatmap so you can see whether the current setting sits in a mountain "
            "or on a needle.",
            emphasize=True,
        )
        self.pr_metric = LabeledEntry(settings, "Metric (default: prop eval pass probability)", "eval_pass_probability")
        self.pr_pass_threshold_pct = LabeledEntry(settings, "Pass threshold (on the metric's own scale, e.g. 50)", 50.0)
        self.pr_pct_range = LabeledEntry(settings, "Percent range (+/- around each current value)", 0.5)
        self.pr_n_steps_1d = LabeledEntry(settings, "Steps per parameter (1D sweep)", 9)
        self.pr_n_steps_2d = LabeledEntry(settings, "Grid size per side (2D heatmap)", 7)
        self.pr_max_params = LabeledEntry(settings, "Max parameters to check", 6)
        self.pr_n_heatmap_pairs = LabeledEntry(settings, "Number of parameter pairs to heatmap", 1)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN ROBUSTNESS MAP", self._param_robustness_run_clicked, primary=True).pack(side="left")

        self.pr_progress = NeuralProgress(f)
        self.pr_progress.pack(fill="x", padx=24, pady=(2, 10))

        score_section = self._section(f, "Parameter Robustness Score", "0-100, with a plain verdict.")
        score_row = Frame(score_section, bg=PANEL)
        score_row.pack(fill="x", padx=18, pady=(0, 10))
        self.pr_score_label = Label(
            score_row, text="Not run yet.", bg=PANEL, fg=TEXT_DIM, font=_safe_font(20, "bold"),
        )
        self.pr_score_label.pack(side="left")
        self.pr_verdict_label = Label(
            score_row, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(12, "bold"),
        )
        self.pr_verdict_label.pack(side="left", padx=(14, 0))

        output_section = self._section(f, "Robustness map output", "Per-parameter sweep results, heatmap grid(s), and notes.")
        _pr_output_frame = Frame(output_section, bg=PANEL)
        self.pr_output = Text(
            _pr_output_frame, height=22, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _pr_output_scroll = ttk.Scrollbar(
            _pr_output_frame, orient="vertical", command=self.pr_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.pr_output.configure(yscrollcommand=_pr_output_scroll.set)
        self.pr_output.pack(side="left", fill="both", expand=True)
        _pr_output_scroll.pack(side="right", fill="y")
        _pr_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.pr_output)

    def _log_pr(self, msg: str):
        self.pr_output.insert(END, msg + "\n")
        self.pr_output.see(END)
        self.root.update_idletasks()

    @staticmethod
    def _pr_verdict_for(score: float) -> tuple[str, str]:
        """Same 70/40 thresholds and verdict labels as the web app's own
        parameter_robustness_job.html JS, so the two UIs never disagree
        about what a given score means. Returns (verdict, color-name)."""
        if score >= 70:
            return "ROBUST", "GREEN"
        if score >= 40:
            return "WATCH", "AMBER"
        return "FRAGILE", "RED"

    def _param_robustness_run_clicked(self):
        if not self.paramrob_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        if not self._try_start_heavy_job(JOB_PARAMETER_ROBUSTNESS):
            return
        self.pr_output.delete("1.0", END)
        self.pr_score_label.config(text="Running...", fg=TEXT_DIM)
        self.pr_verdict_label.config(text="")
        self.pr_progress.start(10)
        threading.Thread(target=self._param_robustness_run_pipeline, daemon=True).start()

    def _param_robustness_run_pipeline(self):
        try:
            df = self.paramrob_context.load_dataframe(self._log_pr)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self.paramrob_context.build_risk_config()
            rules = self.paramrob_context.build_prop_rules()
            mc_cfg = self._validation_mc_config(n_simulations=min(self.mc_sims.get_int(10000), 1000))

            self._log_pr(
                f"Sweeping up to {self.pr_max_params.get_int(6)} tunable parameter(s), then "
                f"heatmapping the {self.pr_n_heatmap_pairs.get_int(1)} most sensitive pair(s)..."
            )
            result = compute_parameter_robustness(
                df, strategy, risk, rules, mc_cfg,
                metric=self.pr_metric.get_str().strip() or "eval_pass_probability",
                pass_threshold_pct=self.pr_pass_threshold_pct.get_float(50.0),
                max_params=self.pr_max_params.get_int(6),
                pct_range=self.pr_pct_range.get_float(0.5),
                n_steps_1d=self.pr_n_steps_1d.get_int(9),
                n_steps_2d=self.pr_n_steps_2d.get_int(7),
                n_heatmap_pairs=self.pr_n_heatmap_pairs.get_int(1),
                progress_cb=self._log_pr,
            )

            verdict, color_name = self._pr_verdict_for(result.parameter_robustness_score)
            color = {"GREEN": GREEN, "AMBER": AMBER, "RED": RED}[color_name]
            self.pr_score_label.config(text=f"{result.parameter_robustness_score:.1f} / 100", fg=color)
            self.pr_verdict_label.config(text=verdict, fg=color)

            self._log_pr(
                f"\n{result.n_parameters_checked} parameter(s) checked against {result.metric}, "
                f"pass threshold {result.pass_threshold_pct:g}%. {result.n_cliffs_detected} cliff(s) detected."
            )
            for note in result.notes:
                self._log_pr(f"  note: {note}")

            self._log_pr("\nPer-parameter 1D sweep:")
            for r in result.per_parameter:
                flag = " <-- CLIFF" if r.cliff_detected else ""
                self._log_pr(
                    f"  {r.gene_label}: current={r.base_value:.4g}, current metric={r.base_metric:.2f}, "
                    f"max adjacent-step drop {r.max_pct_drop_between_adjacent_steps:.0f}%{flag}"
                )

            if result.pair_heatmaps:
                self._log_pr("\nNeighborhood heatmap(s) -- the highlighted [*] cell is the current setting:")
            for pair in result.pair_heatmaps:
                hm = pair.heatmap
                tag = "NEEDLE (cliff)" if pair.is_cliff else ("PLATEAU" if pair.is_plateau else "mixed")
                self._log_pr(
                    f"\n  {hm.gene_a_label} x {hm.gene_b_label} -- grid pass rate "
                    f"{pair.fraction_of_grid_passing:.0f}% -- {tag}"
                )
                a_vals, b_vals, grid = hm.a_values, hm.b_values, hm.grid
                base_i, base_j = len(a_vals) // 2, len(b_vals) // 2
                header = "         " + "".join(f"{b:>9.3g}" for b in b_vals)
                self._log_pr(f"  {header}")
                for i, a in enumerate(a_vals):
                    row_cells = []
                    for j, _b in enumerate(b_vals):
                        val = grid[i][j] if grid and i < len(grid) and j < len(grid[i]) else None
                        cell = "     n/a" if val is None else f"{val:>7.0f}"
                        marker = "*" if (i == base_i and j == base_j) else " "
                        row_cells.append(f"{cell}{marker}")
                    self._log_pr(f"  {a:>7.3g}  " + "".join(row_cells))

            self._log_pr("\nDone.")
        except StrategyError as exc:
            self._log_pr(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_pr(f"\nRobustness map error: {exc}")
        except Exception:
            self._log_pr("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.pr_progress.stop()
            self._release_heavy_job(JOB_PARAMETER_ROBUSTNESS)
