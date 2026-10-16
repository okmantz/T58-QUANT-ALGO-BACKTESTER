"""Refinement and single-run (Run & Report) tabs."""
from __future__ import annotations

import dataclasses
import os
import threading
import traceback
import webbrowser
from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import with_prop_safety_defaults
from app.backtest.statistics import net_profit_reset_note
from app.data.importer import import_csv
from app.data.multi_timeframe import merge_multi_timeframe
from app.data.timeframe_resample import infer_timeframe_label as _infer_timeframe_label
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import FITNESS_METRICS, OPTIMIZER_MODES, RefinementConfig, run_iterative_refinement
from app.orchestration import pipeline_guide
from app.prop.simulator import simulate_account
from app.reports.generator import generate_full_report
from app.reports.refinement_report import generate_refinement_report
from app.strategy.base import StrategyError
from app.strategy.library import record_backtest_result, record_lookahead_result
from app.strategy.lookahead_check import check_for_lookahead
from pathlib import Path
from tkinter import END, Frame, Label, Text, messagebox, ttk
from app.ui.common import OUTPUT_DIR, _safe_font, _strategy_display_name
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui.widgets.cards import NeuralProgress
from app.ui.widgets.run_context import RunContextPanel
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
BORDER = ""
TEXT = ""
TEXT_MUTED = ""
LOG_BG = ""
GREEN = ""
AMBER = ""
theme.bind(globals())


class RefineRunMixin:
    """Refinement and single-run (Run & Report) tabs."""


    # -----------------------------------------------------------------------
    # Iterative Refinement — shared execution helper (used by both the
    # standalone button on Tab 6 and the optional auto-run from Tab 5)
    # -----------------------------------------------------------------------

    def _build_refine_config(self) -> RefinementConfig:
        metric_label = self.refine_metric.get_str()
        metric_key = self._refine_metric_label_to_key.get(metric_label, "eval_pass_probability")
        optimizer_label = self.refine_optimizer_mode.get_str()
        optimizer_key = self._optimizer_mode_label_to_key.get(optimizer_label, "genetic")
        return RefinementConfig(
            fitness_metric=metric_key,
            population_size=self.refine_population.get_int(10),
            generations=self.refine_generations.get_int(5),
            elite_count=self.refine_elite.get_int(2),
            mutation_rate=self.refine_mutation_rate.get_float(0.35),
            mutation_strength=self.refine_mutation_strength.get_float(0.25),
            random_immigrants_frac=self.refine_immigrants.get_float(0.15),
            search_monte_carlo_sims=self.refine_search_sims.get_int(500),
            random_seed=self.refine_seed.get_int(42),
            cost_stress_enabled=self.refine_cost_stress_enabled.get(),
            cost_stress_multiplier=self.refine_cost_stress_multiplier.get_float(2.0),
            cost_stress_penalty_weight=self.refine_cost_stress_weight.get_float(0.35),
            optimizer_mode=optimizer_key,
        )

    def _execute_refinement(self, df, strategy, risk, rules, mc_cfg, log_fn) -> dict:
        """
        Runs Iterative Refinement against an already-built strategy/df/risk/
        rules/mc_cfg (the exact same objects the normal pipeline would use)
        and writes a second, separate report. Works for Manual, Python,
        PineScript, and MQL5 strategies alike. Raises RefinementError if the
        strategy has no tunable parameters for its source type.
        """
        refine_cfg = self._build_refine_config()
        result = run_iterative_refinement(
            df, strategy, risk, rules, mc_cfg, refine_cfg, progress_cb=log_fn,
        )

        period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
        instrument = (
            os.path.basename(self.csv_paths[0]) if len(self.csv_paths) == 1
            else " + ".join(os.path.basename(p) for p in self.csv_paths)
        )
        paths = generate_refinement_report(
            output_dir=OUTPUT_DIR,
            result=result,
            strategy_name=_strategy_display_name(strategy),
            instrument=instrument,
            timeframe=_infer_timeframe_label(df),
            backtest_period=period,
            price_df=df,
        )

        self._last_refinement_result = result
        self._last_refinement_html_path = paths["html"]
        self._last_refinement_best_strategy_path = paths.get("best_strategy_file")
        self.open_refine_report_btn.config(state="normal")
        self.apply_best_config_btn.config(state="normal")
        return paths

    # -----------------------------------------------------------------------
    # Tab 6 — Iterative Refinement (optional)
    # -----------------------------------------------------------------------

    def _build_refine_tab(self):
        f = self._scrollable(self.tab_refine)

        self._page_header(
            f,
            "OPTIMIZE / Iterative Refinement",
            "Iterative Refinement (Optional)",
            "Genetic-algorithm-style parameter search: re-runs this strategy many times "
            "with mutated parameters on the SAME historical data, keeps the "
            "best-performing configurations each round, and converges toward the "
            "best-scoring configuration it can find. Produces its own, separate report "
            "-- the normal Run & Report tab and report.html are completely unaffected "
            "unless you enable this below. Its own market data, prop rules, and risk "
            "settings below are self-contained to this tab, just like the web app.",
        )

        section = self._section(
            f, "Enable for this run",
            "Off by default, and reset per session -- nothing changes about the normal "
            "pipeline unless this is checked. When ON, clicking RUN FULL PIPELINE in "
            "Step 5 will also run Iterative Refinement afterward and produce a second "
            "report. You can also run it on its own with the button below at any time, "
            "regardless of this setting.",
        )
        self.refine_enabled = LabeledCheckbox(
            section, "Enable Iterative Refinement when running the full pipeline (Step 5)", False,
        )
        Label(
            section,
            text="Works with Manual Strategy Builder, Python, PineScript, and MQL5 strategies. "
                 "For Python it searches every top-level SCREAMING_SNAKE_CASE numeric constant "
                 "(e.g. EMA_FAST, STOP_LOSS_PIPS); for PineScript, every input.int()/input.float() "
                 "value; for MQL5, every iMA()/iRSI() period -- plus the T58_SL_PIPS/T58_TP_PIPS "
                 "directives for all three. A strategy with no such parameters will say so clearly "
                 "rather than run a meaningless search.",
            bg=PANEL, fg=AMBER, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        self.refine_context = RunContextPanel(self, "Iterative Refinement")
        self.refine_context.build(f)

        settings = self._section(
            f, "Search settings",
            "Every setting below has a reasonable default -- you don't need to touch any "
            "of them to run a first search.",
        )

        self._refine_metric_labels = list(FITNESS_METRICS.values())
        self._refine_metric_label_to_key = {v: k for k, v in FITNESS_METRICS.items()}
        self.refine_metric = LabeledCombo(
            settings, "Fitness metric (what \u201cbest\u201d means)", self._refine_metric_labels,
            FITNESS_METRICS["eval_pass_probability"],
        )
        self._optimizer_mode_labels = list(OPTIMIZER_MODES.values())
        self._optimizer_mode_label_to_key = {v: k for k, v in OPTIMIZER_MODES.items()}
        self.refine_optimizer_mode = LabeledCombo(
            settings, "Optimizer mode", self._optimizer_mode_labels, OPTIMIZER_MODES["genetic"],
        )
        self.refine_population = LabeledEntry(settings, "Population size (configs per generation)", 10)
        self.refine_generations = LabeledEntry(settings, "Generations (rounds)", 5)
        self.refine_elite = LabeledEntry(settings, "Elite count (top configs carried forward unchanged)", 2)
        self.refine_mutation_rate = LabeledEntry(settings, "Mutation rate (0-1, chance per parameter per child)", 0.35)
        self.refine_mutation_strength = LabeledEntry(settings, "Mutation strength (0-1, fraction of each parameter's search range)", 0.25)
        self.refine_immigrants = LabeledEntry(settings, "Random immigrants fraction (0-1, per generation)", 0.15)
        self.refine_search_sims = LabeledEntry(settings, "Monte Carlo simulations per candidate during search", 500)
        self.refine_seed = LabeledEntry(settings, "Random seed", 42)

        cost_stress_section = self._section(
            f, "Cost-stress penalty (on by default)",
            "ALSO re-backtests every candidate at spread/slippage/commission multiplied "
            "by the factor below, and blends that stressed-cost result into the fitness "
            "the GA actually selects on -- biasing the search toward strategies whose "
            "edge survives worse execution, not just strategies that look best under the "
            "default cost assumptions. Reported statistics always stay nominal (un-"
            "stressed); only the scalar the GA breeds toward is adjusted.",
        )
        self.refine_cost_stress_enabled = LabeledCheckbox(cost_stress_section, "Enable cost-stress penalty", True)
        self.refine_cost_stress_multiplier = LabeledEntry(cost_stress_section, "Cost multiplier for the stressed re-run", 2.0)
        self.refine_cost_stress_weight = LabeledEntry(cost_stress_section, "Penalty weight (0=ignore, 1=full)", 0.35)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)

        self._button(
            button_row, "RUN ITERATIVE REFINEMENT", self._refine_run_clicked, primary=True,
        ).pack(side="left")

        self.open_refine_report_btn = self._button(
            button_row, "OPEN REFINEMENT REPORT", self._open_refine_report,
        )
        self.open_refine_report_btn.config(state="disabled")
        self.open_refine_report_btn.pack(side="left", padx=8)

        self.apply_best_config_btn = self._button(
            button_row, "APPLY BEST CONFIG TO STRATEGY TAB", self._apply_best_configuration,
        )
        self.apply_best_config_btn.config(state="disabled")
        self.apply_best_config_btn.pack(side="left", padx=8)

        self.refine_progress = NeuralProgress(f)
        self.refine_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Refinement output", "Live search log.")
        _refine_output_frame = Frame(output_section, bg=PANEL)
        self.refine_output = Text(
            _refine_output_frame, height=18, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _refine_output_scroll = ttk.Scrollbar(
            _refine_output_frame, orient="vertical", command=self.refine_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.refine_output.configure(yscrollcommand=_refine_output_scroll.set)
        self.refine_output.pack(side="left", fill="both", expand=True)
        _refine_output_scroll.pack(side="right", fill="y")
        _refine_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.refine_output)

        self._last_refinement_result = None
        self._last_refinement_html_path = None
        self._last_refinement_best_strategy_path = None

    def _log_refine(self, msg: str):
        self.refine_output.insert(END, msg + "\n")
        self.refine_output.see(END)
        self.root.update_idletasks()

    def _open_refine_report(self):
        if self._last_refinement_html_path:
            webbrowser.open(f"file://{self._last_refinement_html_path.resolve()}")

    def _refine_run_clicked(self):
        if not self.refine_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        self.refine_output.delete("1.0", END)
        self.refine_progress.start(10)
        threading.Thread(target=self._refine_run_pipeline, daemon=True).start()

    def _refine_run_pipeline(self):
        try:
            self._log_refine("Importing market data...")
            per_file_results = []
            for p in self.refine_context.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    self._log_refine(
                        f"Import errors ({os.path.basename(p)}):\n"
                        + "\n".join(result.errors)
                    )
                    return
                per_file_results.append((p, result))

            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
            else:
                df, labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
            self._log_refine(f"Loaded {len(df)} bars.")

            self._log_refine("Building strategy...")
            strategy = self._build_strategy()
            risk = self.refine_context.build_risk_config()
            rules = self.refine_context.build_prop_rules()

            n_sims = self.mc_sims.get_int(10000)
            method = self.mc_method.get_str().strip() or "bootstrap"
            mc_cfg = MonteCarloConfig(n_simulations=n_sims, method=method)

            self._log_refine("Starting Iterative Refinement search...")
            paths = self._execute_refinement(df, strategy, risk, rules, mc_cfg, self._log_refine)

            self._log_refine("\nDone. Iterative Refinement report written to:")
            for k, p in paths.items():
                self._log_refine(f"  {k}: {p}")

        except StrategyError as exc:
            self._log_refine(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_refine(f"\nIterative Refinement error: {exc}")
        except Exception:
            self._log_refine("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.refine_progress.stop()

    def _apply_best_configuration(self):
        if not getattr(self, "_last_refinement_result", None):
            messagebox.showinfo(
                "No refinement result",
                "Run Iterative Refinement first, then apply its best configuration.",
            )
            return

        result = self._last_refinement_result
        source_type = result.source_type

        if source_type != "manual":
            self._apply_best_code_strategy(result, source_type)
            return

        if self.strategy_mode.get() != "manual":
            messagebox.showwarning(
                "Mode mismatch",
                "This refinement result was optimized for a Manual Strategy Builder "
                "strategy. Switch to MANUAL mode on the Strategy Configuration tab "
                "(Step 1) to apply it, or re-run Iterative Refinement against whatever "
                "strategy is currently selected.",
            )
            return

        cfg = result.best.config
        entries = cfg.get("entry_conditions", {}) or {}
        exits = cfg.get("exit_conditions", {}) or {}

        self.long_entry_conditions.set_from_conditions(entries.get("long", []), entries.get("long_connectors"))
        self.short_entry_conditions.set_from_conditions(entries.get("short", []), entries.get("short_connectors"))
        self.long_exit_conditions.set_from_conditions(exits.get("long", []), exits.get("long_connectors"))
        self.short_exit_conditions.set_from_conditions(exits.get("short", []), exits.get("short_connectors"))

        rm = cfg.get("risk_management", {}) or {}
        if rm.get("stop_value") is not None:
            self.stop_value.var.set(str(rm["stop_value"]))
        if rm.get("stop_atr_period") is not None:
            self.stop_atr_period.var.set(str(rm["stop_atr_period"]))
        if rm.get("target_value") is not None:
            self.target_value.var.set(str(rm["target_value"]))
        if rm.get("target_atr_period") is not None:
            self.target_atr_period.var.set(str(rm["target_atr_period"]))
        trailing = rm.get("trailing_stop", {}) or {}
        if trailing.get("value") is not None:
            self.trailing_value.var.set(str(trailing["value"]))
        if trailing.get("atr_period") is not None:
            self.trailing_atr_period.var.set(str(trailing["atr_period"]))
        be = rm.get("break_even", {}) or {}
        if be.get("trigger_r") is not None:
            self.breakeven_trigger.var.set(str(be["trigger_r"]))
        if rm.get("max_bars_in_trade") is not None:
            self.max_bars.var.set(str(rm["max_bars_in_trade"]))

        # Legacy top-level fields, in case this config came from a
        # non-visual-builder ManualStrategy (e.g. loaded from an older
        # config or the CLI default strategy).
        if cfg.get("stop_loss_pips") is not None and not entries:
            self.stop_value.var.set(str(cfg["stop_loss_pips"]))
        if cfg.get("take_profit_pips") is not None and not entries:
            self.target_value.var.set(str(cfg["take_profit_pips"]))

        messagebox.showinfo(
            "Applied",
            "The optimized parameters have been loaded into the Strategy Builder tab. "
            "Switch tabs to review them, then re-run the normal pipeline (Step 5) to "
            "confirm the result with a fresh, non-search report.",
        )

    def _apply_best_code_strategy(self, result, source_type: str):
        """
        For Python/PineScript/MQL5 strategies, "applying" the winner means
        pointing the Strategy Configuration tab's file selector at the
        already-written patched source file (its logic is identical to the
        original -- only the numeric parameter values changed), so the next
        run uses it.
        """
        path = getattr(self, "_last_refinement_best_strategy_path", None)
        if not path:
            messagebox.showinfo(
                "File not available",
                "The optimized strategy file wasn't written for this run -- try "
                "running Iterative Refinement again.",
            )
            return

        expected_mode = source_type  # "python" | "pinescript" | "mql5"
        if self.strategy_mode.get() != expected_mode:
            self._set_strategy_mode(expected_mode)

        self.strategy_py_path = str(path)
        self.strategy_file_status.config(text=f"Selected: {os.path.basename(str(path))}", fg=GREEN)

        messagebox.showinfo(
            "Applied",
            f"The optimized {FITNESS_METRICS.get(result.fitness_metric, result.fitness_metric)}-tuned "
            f"strategy file has been selected on the Strategy Configuration tab (Step 1):\n\n{path}\n\n"
            "Re-run the normal pipeline (Step 5) to confirm the result with a fresh, "
            "non-search report.",
        )

    # -----------------------------------------------------------------------
    # Tab 5 — Run & report
    # -----------------------------------------------------------------------

    def _build_run_tab(self):
        f = self._scrollable(self.tab_run)

        self._page_header(
            f,
            "TEST / Run & Report",
            "Run Backtest & Report",
            "Backtest → Prop Simulation → Monte Carlo → Report. This runs ONE strategy, once, "
            "with no GA search -- for batch-testing several queued strategies at once, or for the "
            "full walk-forward-aware GA search (baseline -> GA -> re-validated Monte Carlo -> "
            "out-of-sample check -> holdout check -> verdict), use RUN FULL PIPELINE (BATCH) on "
            "01 Strategy Configuration's Batch selection panel, or 15 Full Pipeline for a single strategy.",
        )

        section = self._section(
            f,
            "Simulation",
            "Configure the Monte Carlo run before starting.",
            emphasize=True,
        )

        self.mc_sims = LabeledEntry(
            section, "Monte Carlo simulations", 10000
        )
        self.mc_method = LabeledEntry(
            section,
            "Method (bootstrap/shuffle/block_bootstrap)",
            "bootstrap",
        )
        self.mc_reset_on_breach = LabeledCheckbox(
            section,
            "Reset-on-breach: chain mechanical rebuys within each simulated path instead of "
            "stopping at the first bust (see 'Full-Lifecycle Survival Simulator' for the version "
            "priced with real fees/bankroll)",
            default=False,
        )

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)

        self._button(
            button_row,
            "RUN BACKTEST & REPORT",
            self._run_clicked,
            primary=True,
        ).pack(side="left")

        self.open_report_btn = self._button(
            button_row,
            "OPEN HTML REPORT",
            self._open_report,
        )
        self.open_report_btn.config(state="disabled")
        self.open_report_btn.pack(side="left", padx=8)

        batch_row = Frame(f, bg=BG)
        batch_row.pack(fill="x", padx=24, pady=(0, 10))
        Label(
            batch_row, text="Testing more than one strategy at once?", bg=BG, fg=TEXT_MUTED,
            font=_safe_font(8),
        ).pack(side="left")
        self._button(
            batch_row, "GO TO BATCH SELECTION (01 Strategy Configuration)",
            lambda: self._show_page("strategyconfig"),
        ).pack(side="left", padx=8)

        self.progress = NeuralProgress(f)
        self.progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(
            f,
            "Pipeline output",
            "Live execution log.",
        )

        _output_frame = Frame(output_section, bg=PANEL)
        self.output = Text(
            _output_frame, height=18,
            wrap="word",
            bg=LOG_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            font=(MONO, 9),
        )
        _output_scroll = ttk.Scrollbar(
            _output_frame, orient="vertical", command=self.output.yview, style="T58.Vertical.TScrollbar",
        )
        self.output.configure(yscrollcommand=_output_scroll.set)
        self.output.pack(side="left", fill="both", expand=True)
        _output_scroll.pack(side="right", fill="y")
        _output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.output)

        self._last_html_path: Path | None = None

    def _log(self, msg: str):
        self.output.insert(END, msg + "\n")
        self.output.see(END)
        self.root.update_idletasks()

    def _run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning(
                "Missing data",
                "Please select a market data CSV in Step 2.",
            )
            return

        self.output.delete("1.0", END)
        self.progress.start(10)
        threading.Thread(
            target=self._run_pipeline,
            daemon=True,
        ).start()

    def _open_report(self):
        if self._last_html_path:
            webbrowser.open(
                f"file://{self._last_html_path.resolve()}"
            )

    def _run_pipeline(self):
        try:
            self._log("Importing market data...")

            per_file_results = []
            for p in self.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    self._log(
                        f"Import errors ({os.path.basename(p)}):\n"
                        + "\n".join(result.errors)
                    )
                    return
                per_file_results.append((p, result))
                for w in result.warnings:
                    self._log(f"  [warning] {os.path.basename(p)}: {w}")

            if len(per_file_results) == 1:
                df = per_file_results[0][1].dataframe
                self._log(f"Loaded {len(df)} bars.")
            else:
                df, labels = merge_multi_timeframe(
                    [r.dataframe for _, r in per_file_results]
                )
                self._log(
                    f"Loaded {len(per_file_results)} timeframes and merged them "
                    f"on the finest timeframe: {' + '.join(labels)}  "
                    f"({len(df)} base bars)."
                )

            self._log("Building strategy...")
            strategy = self._build_strategy()

            self._log("Configuring risk & prop rules...")
            risk = self._build_risk_config()
            rules = self._build_prop_rules()
            # FIX (audit): reconcile risk against the active prop rules
            # before backtesting -- see app.backtest.risk.with_prop_safety_
            # defaults' own docstring and Quick Optimize/Full Pipeline/
            # Evolution Lab/Speed Run, which already do this. Without it, this
            # tab could silently backtest against a different account balance
            # (and without the account-blown/daily-loss circuit breakers those
            # other tools' backtests enforce) than the other tools would for
            # the exact same strategy/data/settings.
            risk = with_prop_safety_defaults(risk, rules)
            # FIX (2026-09-18): read this here (BEFORE the backtest below
            # runs) and set it on `risk` itself, not just on mc_cfg further
            # down -- see RiskConfig.reset_on_breach's own docstring.
            # Previously this was read only after bt_result already
            # existed, and only ever reached the post-hoc Monte Carlo
            # layer, so checking it never actually kept the raw backtest
            # itself trading past the first blown account.
            reset_on_breach = self.mc_reset_on_breach.get()
            risk = dataclasses.replace(risk, reset_on_breach=reset_on_breach)
            adaptive_risk = self._build_adaptive_risk_config()
            if adaptive_risk is not None:
                self._log(f"Adaptive risk enabled: {len(adaptive_risk.rules)} rule(s)")

            self._log("Running historical backtest...")
            bt_result = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)

            if strategy.source_type == "python":
                self._log("Checking for lookahead bias...")
                try:
                    lookahead_result = check_for_lookahead(strategy, df, max_signal_checkpoints=8)
                    self._log(f"  {lookahead_result.summary()}")
                    active_lib_strategy = getattr(self, "_active_library_strategy", None)
                    if active_lib_strategy:
                        try:
                            record_lookahead_result(*active_lib_strategy, {
                                "clean": not lookahead_result.bug_detected,
                                "summary": lookahead_result.summary(),
                            })
                        except (FileNotFoundError, ValueError):
                            pass  # strategy renamed/deleted mid-run -- not worth failing the run over
                except Exception:
                    # This check is a best-effort audit, not part of the
                    # core pipeline -- never let it take down a run that
                    # would otherwise succeed.
                    self._log("  Lookahead check failed to run (skipped).")

            self._log(
                f"  Trades: {len(bt_result.trades)}  "
                f"Net profit: ${bt_result.statistics.net_profit:,.2f}  "
                f"Win rate: {bt_result.statistics.win_rate:.1f}%  "
                f"Max DD: {bt_result.statistics.max_drawdown_pct:.2f}%"
                + net_profit_reset_note(bt_result.statistics)
            )

            self._log(
                "Running prop-firm simulation on historical sequence..."
            )

            trade_pnls = [t.pnl for t in bt_result.trades]
            trade_dates = [t.entry_time for t in bt_result.trades]

            single_run = simulate_account(
                trade_pnls,
                trade_dates,
                rules,
                reset_on_breach=reset_on_breach,
            )

            self._log(
                f"  Passed evaluation: {single_run.passed_evaluation}  "
                f"Reached payout: {single_run.reached_first_payout}  "
                f"Failed: {single_run.failed} "
                f"({single_run.failure_reason})"
            )

            if not bt_result.trades:
                # A strategy that produces zero signals over the given data
                # has nothing for the prop simulation or Monte Carlo to
                # resample -- both are fundamentally undefined here, not
                # just numerically awkward. Previously this fell through to
                # run_monte_carlo(), which correctly raises ValueError, but
                # that surfaced to the user as an unhandled traceback rather
                # than a clear, actionable message. Stop here instead.
                self._log("\n" + pipeline_guide.after_first_backtest({"trade_count": 0}))
                return

            n_sims = self.mc_sims.get_int(10000)
            method = self.mc_method.get_str().strip() or "bootstrap"

            self._log(
                f"Running Monte Carlo simulation "
                f"({n_sims:,} runs, method={method}"
                f"{', reset-on-breach chaining ON' if reset_on_breach else ''})..."
            )

            mc_cfg = MonteCarloConfig(
                n_simulations=n_sims,
                method=method,
                reset_on_breach=reset_on_breach,
            )

            mc_result = run_monte_carlo(
                bt_result.trades,
                rules,
                mc_cfg,
            )

            self._log(
                f"  Evaluation pass probability: "
                f"{mc_result.evaluation_pass_probability:.1f}%"
            )
            self._log(
                f"  First payout probability: "
                f"{mc_result.first_payout_probability:.1f}%"
            )
            self._log(
                f"  Expected payout: "
                f"${mc_result.expected_payout:,.2f}"
            )
            self._log(
                f"  Risk of ruin: "
                f"{mc_result.risk_of_ruin_pct:.1f}%"
            )
            if reset_on_breach:
                self._log(
                    f"  Mean mechanical rebuys per simulated path: "
                    f"{mc_result.mean_attempts_per_path:.1f} "
                    f"(median {mc_result.median_attempts_per_path:.0f}, "
                    f"p95 {mc_result.p95_attempts_per_path:.0f}) -- see "
                    "the methodology note in the report for what this changes about the "
                    "probabilities above; use 15 Payout Probability's Bankroll/EV section "
                    "to price this against a real starting bankroll and reset fees."
                )

            self._log("Running out-of-sample holdout check...")
            try:
                holdout_comparison = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2)
            except Exception:
                self._log("  Holdout check skipped (not enough data to split).")
                holdout_comparison = None

            self._log("Generating report...")

            period = (
                str(df["timestamp"].iloc[0]),
                str(df["timestamp"].iloc[-1]),
            )

            paths = generate_full_report(
                output_dir=OUTPUT_DIR,
                strategy_name=bt_result.strategy_name,
                strategy_source_type=strategy.source_type,
                instrument=(
                    os.path.basename(self.csv_paths[0])
                    if len(self.csv_paths) == 1
                    else " + ".join(os.path.basename(p) for p in self.csv_paths)
                ),
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

            self._last_html_path = paths["html"]
            self.open_report_btn.config(state="normal")
            try:
                self._refresh_dashboard()
            except Exception:
                pass

            active_lib_strategy = getattr(self, "_active_library_strategy", None)
            if active_lib_strategy:
                lib_mode, lib_filename = active_lib_strategy
                try:
                    record_backtest_result(lib_mode, lib_filename, {
                        "trades": len(bt_result.trades),
                        "net_profit": round(bt_result.statistics.net_profit, 2),
                        "win_rate": round(bt_result.statistics.win_rate, 1),
                        "max_dd": round(bt_result.statistics.max_drawdown_pct, 2),
                        "passed_evaluation": single_run.passed_evaluation,
                        "report_html": str(paths["html"]),
                    })
                except (FileNotFoundError, ValueError):
                    pass  # strategy was renamed/deleted mid-run -- not worth failing the run over

            self._log("\nDone. Report written to:")

            for k, p in paths.items():
                self._log(f"  {k}: {p}")

            self._log(
                "\n" + pipeline_guide.after_first_backtest(
                    bt_result.statistics.to_dict(), passed_evaluation=single_run.passed_evaluation,
                )
            )

            if getattr(self, "refine_enabled", None) and self.refine_enabled.get():
                self._log("\n--- Iterative Refinement (optional feature enabled on Step 6) ---")
                try:
                    refine_paths = self._execute_refinement(df, strategy, risk, rules, mc_cfg, self._log)
                    self._log("\nIterative Refinement report written to:")
                    for k, p in refine_paths.items():
                        self._log(f"  {k}: {p}")
                except RefinementError as exc:
                    self._log(f"\nIterative Refinement skipped: {exc}")
                except Exception:
                    self._log("\nIterative Refinement failed:\n" + traceback.format_exc())

        except StrategyError as exc:
            self._log(f"\nStrategy error: {exc}")

        except Exception:
            self._log(
                "\nUnexpected error:\n"
                + traceback.format_exc()
            )

        finally:
            self.progress.stop()
