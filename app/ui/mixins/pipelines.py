"""Full Pipeline, Quick Optimize, Optimize Simple, Champion Simple, Autopilot and hub tabs."""
from __future__ import annotations

import os
import threading
import traceback
import webbrowser
from app.backtest.risk import build_run_context
from app.optimize.multi_objective import MultiObjectiveConfig, run_multi_objective_refinement
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import FITNESS_METRICS, RefinementConfig, run_iterative_refinement
from app.orchestration import pipeline_guide
from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline
from app.orchestration.overnight_autopilot import AutopilotConfig, run_overnight_autopilot
from app.orchestration.resource_guard import JOB_FULL_PIPELINE, JOB_SPEED_RUN
from app.orchestration.speed_run import SpeedRunConfig
from app.reports.crash_log import log_crash
from app.scoring import champion_board
from app.strategy.base import StrategyError
from app.strategy.library import STRATEGY_STATUSES, list_saved_strategies, record_validation_result, status_label
from app.strategy.library_loader import load_strategy_object
from pathlib import Path
from tkinter import BooleanVar, Checkbutton, END, Frame, Label, Text, messagebox, ttk
from app.ui.common import OUTPUT_DIR, _safe_font, _strategy_display_name
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
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
LOG_BG = ""
GREEN = ""
RED = ""
AMBER = ""
theme.bind(globals())


class PipelinesMixin:
    """Full Pipeline, Quick Optimize, Optimize Simple, Champion Simple, Autopilot and hub tabs."""


    def _build_full_pipeline_tab(self):
        f = self._scrollable(self.tab_fullpipeline)

        self._page_header(
            f,
            "OPTIMIZE / Full Pipeline",
            "Full Pipeline",
            "One button, the whole workflow: backtests the strategy as given, runs "
            "app.optimize.walkforward_ga to search for a configuration that generalizes "
            "(scored ONLY on out-of-sample fold data, never in-sample -- so it isn't just "
            "curve-fit harder), re-validates the winner with a fresh full-fidelity Monte "
            "Carlo, checks it holds up across several distinct historical stretches with "
            "no further re-tuning, and produces one final report with a plain READY / "
            "MARGINAL / NOT READY verdict. For Python/PineScript/MQL5 strategies, the "
            "winning source is also saved straight into the Strategy Library, auto-tagged "
            "TESTED / PASSED, VALIDATED, or TESTED / FAILED based on that verdict (or "
            "whatever fixed status you pick below), ready to use. Uses the strategy "
            "configured on Step 01 -- but its own market data, prop-firm rules, and risk "
            "settings below are self-contained to this tab, just like the web app: picking "
            "a dataset or changing the account size here never touches any other tab.",
        )

        self.fp_context = RunContextPanel(self, "Full Pipeline")
        self.fp_context.build(f)
        self._button(f, "CHECK DATA INTEGRITY", lambda c=self.fp_context: self._check_data_integrity(c)).pack(anchor="w", padx=18, pady=(0, 8))

        settings = self._section(
            f, "Pipeline settings",
            "Sensible defaults for a single run -- raise population/generations for a more "
            "thorough (slower) search once you know a strategy is worth the time.",
            emphasize=True,
        )
        self.fp_window_mode = LabeledCombo(settings, "GA fold window mode", ["rolling", "anchored"], "rolling")
        self.fp_folds = LabeledEntry(settings, "Number of folds (GA + OOS check)", 4)
        self._fp_metric_labels = list(FITNESS_METRICS.values())
        self._fp_metric_label_to_key = {v: k for k, v in FITNESS_METRICS.items()}
        self.fp_metric = LabeledCombo(
            settings, "Fitness metric", self._fp_metric_labels, FITNESS_METRICS["eval_pass_probability"],
        )
        self.fp_population = LabeledEntry(settings, "GA population size", 12)
        self.fp_generations = LabeledEntry(settings, "GA generations", 6)
        self.fp_search_mc_sims = LabeledEntry(settings, "Monte Carlo sims during search", 200)
        self.fp_final_mc_sims = LabeledEntry(settings, "Monte Carlo sims for final report", 10000)
        self.fp_holdout_frac = LabeledEntry(settings, "Final holdout fraction", 0.2)
        self.fp_seed = LabeledEntry(settings, "Random seed", 42)
        self.fp_adaptive_risk_enabled = LabeledCheckbox(
            settings,
            "Enable adaptive, limit-aware position sizing (throttles risk as the account "
            "nears its daily-loss/max-drawdown limits; same engine as 05 Run & Report's "
            "Adaptive Risk section, applied automatically to every backtest this pipeline runs)",
            False,
        )
        self.fp_reset_on_breach = LabeledCheckbox(
            settings,
            "Reset-on-breach: score on the basis that a blown account gets a fresh eval and "
            "keeps going, not a dead end (recommended -- uncheck to optimize on a single "
            "account with no reset)",
            True,
        )

        library_section = self._section(
            f, "Strategy Library",
            "Works for code strategies (saved as a new .py/.pine/.mq5 file) and Manual Strategy "
            "Builder / Search Lab configs alike (saved as JSON) -- either way it's written under "
            "a new filename, never overwriting the strategy you started from.",
        )
        self.fp_save_to_library = LabeledCheckbox(
            library_section, "Save the winning strategy to the Strategy Library when finished", True,
        )
        _AUTO_STATUS_LABEL = "Auto (based on READY / MARGINAL / NOT READY verdict)"
        self._fp_status_label_to_key = {_AUTO_STATUS_LABEL: None}
        self._fp_status_label_to_key.update({status_label(s): s for s in STRATEGY_STATUSES})
        self.fp_library_status = LabeledCombo(
            library_section, "Status to tag it with",
            [_AUTO_STATUS_LABEL] + [status_label(s) for s in STRATEGY_STATUSES],
            _AUTO_STATUS_LABEL,
        )

        self._build_ai_assist_section(f)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN FULL PIPELINE", self._fullpipeline_run_clicked, primary=True).pack(side="left")
        self.open_fullpipeline_report_btn = self._button(button_row, "OPEN REPORT", self._open_fullpipeline_report)
        self.open_fullpipeline_report_btn.config(state="disabled")
        self.open_fullpipeline_report_btn.pack(side="left", padx=8)

        self.fullpipeline_progress = NeuralProgress(f)
        self.fullpipeline_progress.pack(fill="x", padx=24, pady=(2, 10))

        verdict_section = self._section(f, "Verdict", "Filled in once a run completes.")
        self.fullpipeline_verdict_label = Label(
            verdict_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=820, anchor="w",
        )
        self.fullpipeline_verdict_label.pack(anchor="w", fill="x", padx=18, pady=(2, 10))

        output_section = self._section(f, "Full Pipeline output", "Live progress log.")
        _fullpipeline_output_frame = Frame(output_section, bg=PANEL)
        self.fullpipeline_output = Text(
            _fullpipeline_output_frame, height=20, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _fullpipeline_output_scroll = ttk.Scrollbar(
            _fullpipeline_output_frame, orient="vertical", command=self.fullpipeline_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.fullpipeline_output.configure(yscrollcommand=_fullpipeline_output_scroll.set)
        self.fullpipeline_output.pack(side="left", fill="both", expand=True)
        _fullpipeline_output_scroll.pack(side="right", fill="y")
        _fullpipeline_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.fullpipeline_output)

        self._last_fullpipeline_html_path = None

    def _log_fullpipeline(self, msg: str):
        self.fullpipeline_output.insert(END, msg + "\n")
        self.fullpipeline_output.see(END)
        self.root.update_idletasks()

    def _open_fullpipeline_report(self):
        if self._last_fullpipeline_html_path:
            webbrowser.open(f"file://{self._last_fullpipeline_html_path.resolve()}")

    def _fullpipeline_run_clicked(self):
        if not self.fp_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        if not self._try_start_heavy_job(JOB_FULL_PIPELINE):
            return
        self.fullpipeline_output.delete("1.0", END)
        self.fullpipeline_verdict_label.config(text="Running...", fg=TEXT_DIM)
        self.fullpipeline_progress.start(10)
        threading.Thread(target=self._fullpipeline_run_pipeline, daemon=True).start()

    def _fullpipeline_run_pipeline(self):
        try:
            df = self.fp_context.load_dataframe(self._log_fullpipeline)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self.fp_context.build_risk_config()
            rules = self.fp_context.build_prop_rules()

            metric_key = self._fp_metric_label_to_key.get(self.fp_metric.get_str(), "eval_pass_probability")
            cfg = FullPipelineConfig(
                n_folds=self.fp_folds.get_int(4),
                window_mode=self.fp_window_mode.get_str(),
                ga_population=self.fp_population.get_int(12),
                ga_generations=self.fp_generations.get_int(6),
                ga_search_mc_sims=self.fp_search_mc_sims.get_int(200),
                fitness_metric=metric_key,
                final_mc_sims=self.fp_final_mc_sims.get_int(10000),
                holdout_frac=self.fp_holdout_frac.get_float(0.2),
                oos_check_folds=self.fp_folds.get_int(4),
                random_seed=self.fp_seed.get_int(42),
                save_to_library=self.fp_save_to_library.var.get(),
                library_status=self._fp_status_label_to_key.get(self.fp_library_status.get_str()),
                adaptive_risk_enabled=self.fp_adaptive_risk_enabled.var.get(),
                reset_on_breach=self.fp_reset_on_breach.var.get(),
            )

            self._log_fullpipeline(f"Starting Full Pipeline for '{_strategy_display_name(strategy)}'...\n")
            instrument = self.fp_context.instrument_label()
            ollama_settings = self._build_ollama_settings()
            result = run_full_pipeline(
                df, strategy, risk, rules, OUTPUT_DIR / "full_pipeline", cfg,
                progress_cb=self._log_fullpipeline, instrument=instrument,
                ollama_settings=ollama_settings,
            )

            self._last_fullpipeline_html_path = result.report_paths["html"]
            self.open_fullpipeline_report_btn.config(state="normal")

            verdict_color = {"READY": GREEN, "MARGINAL": AMBER, "NOT READY": RED}.get(result.verdict, TEXT_DIM)
            self.fullpipeline_verdict_label.config(
                text=f"{result.verdict}\n" + "\n".join(f"  \u2022 {r}" for r in result.verdict_reasons),
                fg=verdict_color,
            )

            self._log_fullpipeline(
                f"\nBaseline -> Final:  "
                f"trades {len(result.baseline_bt.trades)} -> {len(result.final_bt.trades)}  |  "
                f"net ${result.baseline_bt.statistics.net_profit:,.2f} -> ${result.final_bt.statistics.net_profit:,.2f}  |  "
                f"eval pass {result.baseline_mc.evaluation_pass_probability:.1f}% -> {result.final_mc.evaluation_pass_probability:.1f}%  |  "
                f"payout {result.baseline_mc.first_payout_probability:.1f}% -> {result.final_mc.first_payout_probability:.1f}%"
            )
            if result.refinement_skip_reason:
                self._log_fullpipeline(f"\nOptimization was skipped: {result.refinement_skip_reason}")
            if result.saved_library_note:
                self._log_fullpipeline(f"\n{result.saved_library_note}")
            for w in result.warnings:
                self._log_fullpipeline(f"WARNING: {w}")

            self._log_fullpipeline(
                "\n" + pipeline_guide.after_full_pipeline(result.verdict, bool(result.saved_library_note), result=result)
            )

            try:
                self._refresh_dashboard()
            except Exception:
                pass

            self._log_fullpipeline(f"\nVerdict: {result.verdict}")
            self._log_fullpipeline("\nDone. Full Pipeline report written to:")
            for k, p in result.report_paths.items():
                self._log_fullpipeline(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_fullpipeline(f"\nStrategy error: {exc}")
            self.fullpipeline_verdict_label.config(text="Failed -- see log.", fg=RED)
        except RefinementError as exc:
            self._log_fullpipeline(f"\nFull Pipeline error: {exc}")
            self.fullpipeline_verdict_label.config(text="Failed -- see log.", fg=RED)
        except Exception as exc:
            self._log_fullpipeline("\nUnexpected error:\n" + traceback.format_exc())
            self.fullpipeline_verdict_label.config(text="Failed -- see log.", fg=RED)
            log_crash("Full Pipeline", exc=exc)
        finally:
            self.fullpipeline_progress.stop()
            self._release_heavy_job(JOB_FULL_PIPELINE)

    # -----------------------------------------------------------------------
    # Quick Optimize -- split out into its own OPTIMIZE tab (Sep 2026
    # reorder) for desktop/web parity. Was previously only reachable via
    # the Strategy Library's "OPTIMIZE SELECTED" button (still there,
    # unchanged, for optimizing several saved strategies at once); this
    # tab is the single-strategy version using the strategy configured on
    # Step 01, matching every other "own RunContextPanel" tool tab.
    # -----------------------------------------------------------------------

    def _build_quick_optimize_tab(self):
        f = self._scrollable(self.tab_quickoptimize)

        self._page_header(
            f,
            "OPTIMIZE / Quick Optimize",
            "\u26a1 Quick Optimize",
            "\"Select a strategy, click Optimize.\" Runs Full Pipeline's own walk-forward-"
            "aware GA (app.orchestration.quick_optimize) in isolation against the strategy "
            "configured on Step 01, and hands back a before/after comparison -- without the "
            "OOS holdout check or the ICIR/Bonferroni significance gate Full Pipeline runs "
            "afterward specifically to catch a GA that got lucky. \u26a0 NOT OOS VALIDATED -- "
            "treat a strong result here as \"worth a real Full Pipeline run,\" not as a "
            "finished answer. To optimize several saved strategies at once instead of one, "
            "use the Strategy Library's own OPTIMIZE SELECTED button.",
        )

        self.qopt_context = RunContextPanel(self, "Quick Optimize")
        self.qopt_context.build(f)
        self._button(f, "CHECK DATA INTEGRITY", lambda c=self.qopt_context: self._check_data_integrity(c)).pack(anchor="w", padx=18, pady=(0, 8))

        settings = self._section(
            f, "Quick Optimize settings",
            "The same walk-forward-aware GA engine Full Pipeline's Step 2 uses.",
            emphasize=True,
        )
        self.qopt_ga_population = LabeledEntry(settings, "GA population", 16)
        self.qopt_ga_generations = LabeledEntry(settings, "GA generations", 8)
        self.qopt_n_folds = LabeledEntry(settings, "Walk-forward folds", 4)
        self.qopt_final_mc_sims = LabeledEntry(settings, "Final Monte Carlo sims", 1000)
        self.qopt_reset_on_breach = LabeledCheckbox(settings, "Reset-on-breach (prop-firm realistic) search basis", True)
        self.qopt_adaptive_risk = LabeledCheckbox(settings, "Enable adaptive, limit-aware position sizing", False)
        self.qopt_reserve_holdout = LabeledCheckbox(settings, "Reserve a holdout slice (required; unchecking makes Quick Optimize refuse to rank)", True)
        self.qopt_save_to_library = LabeledCheckbox(settings, "Save the result to the Strategy Library (tagged \"draft\")", True)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN QUICK OPTIMIZE", self._quick_optimize_run_clicked, primary=True).pack(side="left")

        self.qopt_progress = NeuralProgress(f)
        self.qopt_progress.pack(fill="x", padx=24, pady=(2, 10))

        result_section = self._section(f, "Before / after", "\u26a0 NOT OOS VALIDATED -- see Full Pipeline for that.")
        self.qopt_result_label = Label(
            result_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=900, anchor="w",
        )
        self.qopt_result_label.pack(anchor="w", fill="x", padx=18, pady=(2, 12))

        output_section = self._section(f, "Quick Optimize output", "Live progress log.")
        _qopt_output_frame = Frame(output_section, bg=PANEL)
        self.qopt_output = Text(
            _qopt_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _qopt_output_scroll = ttk.Scrollbar(
            _qopt_output_frame, orient="vertical", command=self.qopt_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.qopt_output.configure(yscrollcommand=_qopt_output_scroll.set)
        self.qopt_output.pack(side="left", fill="both", expand=True)
        _qopt_output_scroll.pack(side="right", fill="y")
        _qopt_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.qopt_output)

    def _log_qopt(self, msg: str):
        self.qopt_output.insert(END, msg + "\n")
        self.qopt_output.see(END)
        self.root.update_idletasks()

    def _quick_optimize_run_clicked(self):
        if not self.qopt_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        self.qopt_output.delete("1.0", END)
        self.qopt_result_label.config(text="Running...", fg=TEXT_DIM)
        self.qopt_progress.start(10)
        threading.Thread(target=self._quick_optimize_run_pipeline, daemon=True).start()

    def _quick_optimize_run_pipeline(self):
        from app.orchestration.quick_optimize import QuickOptimizeConfig, run_quick_optimize

        try:
            df = self.qopt_context.load_dataframe(self._log_qopt)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self.qopt_context.build_risk_config()
            rules = self.qopt_context.build_prop_rules()
            cfg = QuickOptimizeConfig(
                ga_population=self.qopt_ga_population.get_int(16),
                ga_generations=self.qopt_ga_generations.get_int(8),
                n_folds=self.qopt_n_folds.get_int(4),
                final_mc_sims=self.qopt_final_mc_sims.get_int(1000),
                reset_on_breach=self.qopt_reset_on_breach.get(),
                adaptive_risk_enabled=self.qopt_adaptive_risk.get(),
                reserve_holdout=self.qopt_reserve_holdout.get(),
                save_to_library=self.qopt_save_to_library.get(),
            )
            res = run_quick_optimize(df, strategy, risk, rules, cfg, progress_cb=self._log_qopt, instrument_label=os.path.basename(self.qopt_context.csv_paths[0]) if getattr(self.qopt_context, "csv_paths", None) else "")
            if res.instrument_mismatch_warning:
                self._log_qopt(f"\n!!! {res.instrument_mismatch_warning}")
            marker = "IMPROVED" if res.improved else "no improvement"
            self.qopt_result_label.config(
                text=(
                    "\u26a0 NOT OOS VALIDATED  --  "
                    f"{res.baseline_eval_pass_probability:.1f}% -> {res.optimized_eval_pass_probability:.1f}%  ({marker})"
                ),
                fg=GREEN if res.improved else AMBER,
            )
            self._log_qopt(f"\nDone in {res.elapsed_seconds:.1f}s.")
            if res.saved_library_path:
                self._log_qopt(f"Saved to library: {res.saved_library_path}")
            try:
                self._refresh_strategy_library()
            except Exception:
                pass
        except Exception as exc:
            self._log_qopt("\nUnexpected error:\n" + traceback.format_exc())
            self.qopt_result_label.config(text="Failed -- see log.", fg=RED)
            log_crash("Quick Optimize", exc=exc)
        finally:
            self.qopt_progress.stop()

    # -----------------------------------------------------------------------
    # Optimize -- pick your engines
    # -----------------------------------------------------------------------

    def _build_optimize_simple_tab(self):
        f = self._scrollable(self.tab_optimize_simple)

        self._page_header(
            f,
            "OPTIMIZE / Optimize (pick your engines)",
            "\u2726 Optimize \u2014 pick your engines",
            "One clear thing to do: the strategy comes from CREATE \u2192 Strategy Builder "
            "(Step 01, same as every other Optimize tab), the dataset / prop firm / risk "
            "below are this tab's own copy, then tick the engines you want and hit RUN. "
            "Ticked engines run one after another on the same hardened settings, so their "
            "results are comparable. Search Lab and Evolution Lab keep their full settings "
            "on their own tabs -- ticking them here runs the wired engines first, then "
            "takes you there.",
        )

        self.opt_simple_context = RunContextPanel(self, "Optimize (pick your engines)")
        self.opt_simple_context.build(f)

        picker_section = self._section(
            f, "What will run",
            "Strategy: Step 01 builder. Dataset / preset / risk: the panel above. "
            "Risk is hardened through the prop safety defaults before any engine sees it.",
        )
        self.opt_simple_summary = Label(
            picker_section, text="Pick a dataset above, tick engines below, then RUN.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(9), wraplength=900, justify="left", anchor="w",
        )
        self.opt_simple_summary.pack(anchor="w", fill="x", padx=18, pady=(2, 12))

        engines = self._section(
            f, "Engines", "Ticked engines run sequentially, in this order. Everything runs -- nothing is implied.",
            emphasize=True,
        )
        self.opt_eng_quick = LabeledCheckbox(engines, "Quick Optimize \u2014 walk-forward-aware GA, before/after comparison (\u26a0 not OOS validated)", True)
        self.opt_eng_refine = LabeledCheckbox(engines, "Iterative Refinement \u2014 plain GA parameter refinement", False)
        self.opt_eng_risksweep = LabeledCheckbox(engines, "Risk Sweep \u2014 which risk-per-trade level survives best", False)
        self.opt_eng_multiobj = LabeledCheckbox(engines, "Multi-Objective \u2014 NSGA-II Pareto front", False)
        self.opt_eng_search = LabeledCheckbox(engines, "Search Lab \u2014 full settings live on its own tab (link-out below)", False)
        self.opt_eng_evolution = LabeledCheckbox(engines, "Evolution Lab (GA) \u2014 checkpointed runner lives on its own tab (link-out below)", False)

        advanced = self._advanced_expander(f)
        adv_section = self._section(
            advanced, "Advanced options",
            "Defaults are sensible \u2014 skip this unless you know why you're changing it. "
            "Mirrors Quick Optimize's own fitness / GA fields.",
        )
        self.opt_simple_fitness = LabeledCombo(adv_section, "Fitness metric (what 'best' means)", list(FITNESS_METRICS), default="eval_pass_probability")
        self.opt_simple_ga_population = LabeledEntry(adv_section, "GA population", 16)
        self.opt_simple_ga_generations = LabeledEntry(adv_section, "GA generations", 8)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN TICKED ENGINES", self._optimize_simple_run_clicked, primary=True).pack(side="left")

        self.opt_simple_progress = NeuralProgress(f)
        self.opt_simple_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Optimize output", "Every engine that ran is logged here, in order \u2014 including the ones that only link out.")
        self.opt_simple_output = self._log_box(output_section, height=18)

    def _log_opt_simple(self, msg: str):
        self.opt_simple_output.insert(END, msg + "\n")
        self.opt_simple_output.see(END)
        self.root.update_idletasks()

    def _optimize_simple_run_clicked(self):
        if not self.opt_simple_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        engines = []
        if self.opt_eng_quick.get():
            engines.append("quick")
        if self.opt_eng_refine.get():
            engines.append("refine")
        if self.opt_eng_risksweep.get():
            engines.append("risksweep")
        if self.opt_eng_multiobj.get():
            engines.append("multiobj")
        link_outs = []
        if self.opt_eng_search.get():
            link_outs.append("search")
        if self.opt_eng_evolution.get():
            link_outs.append("evolution")
        if not engines and not link_outs:
            messagebox.showwarning("Nothing ticked", "Tick at least one engine first.")
            return
        self.opt_simple_summary.config(
            text=f"Dataset: {self.opt_simple_context.instrument_label()}  \u00b7  "
                 f"Preset: {self.opt_simple_context.p_preset_combo.get_str()}  \u00b7  "
                 f"Engines: {', '.join(engines + link_outs)}",
            fg=TEXT,
        )
        self.opt_simple_output.delete("1.0", END)
        self.opt_simple_progress.start(10)
        threading.Thread(
            target=self._optimize_simple_run_pipeline, args=(engines, link_outs), daemon=True,
        ).start()

    def _optimize_simple_run_pipeline(self, engines: list, link_outs: list):
        log = self._log_opt_simple
        try:
            df = self.opt_simple_context.load_dataframe(log)
            if df is None:
                return
            rules = self.opt_simple_context.build_prop_rules()
            # build_risk_config() already routes through build_run_context;
            # pass explicitly again so this page's contract is visible at
            # the call site (the second pass is a no-op by construction).
            risk = build_run_context(self.opt_simple_context.build_risk_config(), rules)
            mc_config = self._validation_mc_config()
            fitness = self.opt_simple_fitness.get_str().strip() or "eval_pass_probability"
            population = self.opt_simple_ga_population.get_int(16)
            generations = self.opt_simple_ga_generations.get_int(8)

            if "quick" in engines:
                log("\n=== Quick Optimize ===")
                try:
                    from app.orchestration.quick_optimize import QuickOptimizeConfig, run_quick_optimize

                    strategy = self._build_strategy()
                    cfg = QuickOptimizeConfig(
                        ga_population=population, ga_generations=generations, fitness_metric=fitness,
                    )
                    res = run_quick_optimize(df, strategy, risk, rules, cfg, progress_cb=log)
                    if res.instrument_mismatch_warning:
                        log(f"!!! {res.instrument_mismatch_warning}")
                    marker = "IMPROVED" if res.improved else "no improvement"
                    log(f"Quick Optimize done in {res.elapsed_seconds:.1f}s: eval pass "
                        f"{res.baseline_eval_pass_probability:.1f}% -> {res.optimized_eval_pass_probability:.1f}% ({marker}). "
                        "\u26a0 NOT OOS VALIDATED.")
                except Exception:
                    log("Quick Optimize failed:\n" + traceback.format_exc())

            if "refine" in engines:
                log("\n=== Iterative Refinement ===")
                try:
                    strategy = self._build_strategy()
                    refine_cfg = RefinementConfig(
                        fitness_metric=fitness, population_size=population, generations=generations,
                    )
                    res = run_iterative_refinement(df, strategy, risk, rules, mc_config, refine_cfg, progress_cb=log)
                    log(f"Iterative Refinement done in {res.elapsed_seconds:.1f}s: best {res.fitness_metric} "
                        f"fitness {res.best.fitness:.4g} (baseline {res.baseline.fitness:.4g}).")
                except Exception:
                    log("Iterative Refinement failed:\n" + traceback.format_exc())

            if "risksweep" in engines:
                log("\n=== Risk Sweep ===")
                try:
                    from app.optimize.risk_sweep import run_risk_sweep

                    res = run_risk_sweep(df, self._build_strategy, risk, rules)
                    log(res.render_table())
                except Exception:
                    log("Risk Sweep failed:\n" + traceback.format_exc())

            if "multiobj" in engines:
                log("\n=== Multi-Objective ===")
                try:
                    strategy = self._build_strategy()
                    mo_cfg = MultiObjectiveConfig(population_size=population, generations=generations)
                    res = run_multi_objective_refinement(df, strategy, risk, rules, mc_config, mo_cfg, progress_cb=log)
                    log(f"Multi-Objective done in {res.elapsed_seconds:.1f}s: "
                        f"{len(res.pareto_front)} Pareto-optimal candidate(s) on the final front.")
                    for w in res.warnings:
                        log(f"  warning: {w}")
                except Exception:
                    log("Multi-Objective failed:\n" + traceback.format_exc())

            log("\nAll wired engines finished.")
        except StrategyError as exc:
            log(f"\nStrategy error: {exc}")
        except Exception as exc:
            log("\nUnexpected error:\n" + traceback.format_exc())
            log_crash("Optimize (pick your engines)", exc=exc)
        finally:
            self.opt_simple_progress.stop()

        # Link-outs run last, only after the wired engines above -- the
        # status line is logged BEFORE navigating so the log (this page's
        # honest record) says exactly what did and didn't run here.
        if "search" in link_outs:
            log("\u2192 open the Search Lab tab to run (settings carried) \u2014 Search Lab's hypothesis/family "
                "settings live there, so it isn't run from here.")
            self.root.after(0, lambda: self._show_page("search"))
        if "evolution" in link_outs:
            log("\u2192 open the Evolution Lab tab to run (settings carried) \u2014 Evolution Lab's checkpointed "
                "runner lives there, so it isn't run from here.")
            self.root.after(0, lambda: self._show_page("evolution"))

    # -----------------------------------------------------------------------
    # Champion Board -- promote
    # -----------------------------------------------------------------------

    def _build_champion_simple_tab(self):
        f = self._scrollable(self.tab_champion_simple)

        self._page_header(
            f,
            "CHAMPION / Champion Board (promote)",
            "\u2726 Champion Board \u2014 promote",
            "One clear thing to do: pick a saved strategy below, optionally re-run the "
            "validation checks you care about against it, then promote it. The table is "
            "the same Champion Board the Dashboard shows (app.scoring.champion_board). "
            "PROMOTE uses that board's own gated mechanism \u2014 it advances exactly one "
            "stage, and only when every requirement for that stage is already met; "
            "otherwise it names what's missing. Nothing is promoted automatically.",
        )

        board_section = self._section(
            f, "Champion Board",
            "Every saved strategy's verdict and promotion stage. Re-running checks below "
            "uses this tab's own dataset / prop / risk panel, further down.",
            emphasize=True,
        )
        board_frame = Frame(board_section, bg=PANEL)
        board_frame.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        self.champ_simple_tree = ttk.Treeview(
            board_frame, columns=("strategy", "verdict", "notes"), show="headings",
            style="T58Board.Treeview", height=10,
        )
        self.champ_simple_tree.heading("strategy", text="Strategy", anchor="w")
        self.champ_simple_tree.heading("verdict", text="Verdict", anchor="w")
        self.champ_simple_tree.heading("notes", text="Notes", anchor="w")
        self.champ_simple_tree.column("strategy", width=240, anchor="w")
        self.champ_simple_tree.column("verdict", width=120, anchor="w")
        self.champ_simple_tree.column("notes", width=520, anchor="w")
        self.champ_simple_tree.pack(side="left", fill="both", expand=True)
        board_scroll = ttk.Scrollbar(
            board_frame, orient="vertical", command=self.champ_simple_tree.yview, style="T58.Vertical.TScrollbar",
        )
        board_scroll.pack(side="right", fill="y")
        self.champ_simple_tree.configure(yscrollcommand=board_scroll.set)
        self._bind_isolated_wheel(self.champ_simple_tree)
        self._champ_rows_by_iid: dict = {}

        board_btn_row = Frame(board_section, bg=PANEL)
        board_btn_row.pack(anchor="w", padx=14, pady=(0, 12))
        self._button(board_btn_row, "REFRESH", self._champion_simple_refresh).pack(side="left")
        self._button(board_btn_row, "PROMOTE TO CHAMPION", self._champion_simple_promote_clicked, primary=True).pack(side="left", padx=8)
        self.champ_simple_status = Label(
            board_btn_row, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=640, justify="left",
        )
        self.champ_simple_status.pack(side="left", padx=10)

        self.champ_simple_context = RunContextPanel(self, "Champion Board (promote)")
        self.champ_simple_context.build(f)

        checks_section = self._section(
            f, "Re-run checks on the selected strategy",
            "Same checks, same dispatch as VALIDATE \u2192 Validate (pick your checks) \u2014 "
            "run against the board row selected above, on the dataset picked in this tab's panel.",
            emphasize=True,
        )
        checks_row = Frame(checks_section, bg=PANEL)
        checks_row.pack(fill="x", padx=14, pady=(2, 6))
        # Compact row: plain Checkbuttons (not LabeledCheckbox, which is a
        # full-width row widget) so all seven fit on one line.
        self.champ_chk_vars: dict = {}
        for key, label in (
            ("montecarlo", "Monte Carlo"), ("cpcv", "CPCV"), ("pbo", "PBO"),
            ("walkforward", "Walk-Forward"), ("sensitivity", "Sensitivity"),
            ("robustness", "Robustness"), ("regime", "Regime"),
        ):
            var = BooleanVar(value=(key == "cpcv"))
            Checkbutton(
                checks_row, text=label, variable=var, bg=PANEL, fg=TEXT,
                selectcolor=PANEL_3, activebackground=PANEL, highlightthickness=0,
                font=_safe_font(9),
            ).pack(side="left", padx=(0, 10))
            self.champ_chk_vars[key] = var

        advanced = self._advanced_expander(f)
        adv_section = self._section(
            advanced, "Advanced options",
            "Defaults are sensible \u2014 skip this unless you know why you're changing it.",
        )
        self.champ_simple_mc_sims = LabeledEntry(adv_section, "Monte Carlo sims", 2000)
        self.champ_simple_cpcv_groups = LabeledEntry(adv_section, "CPCV groups (N)", 6)
        self.champ_simple_cpcv_test_groups = LabeledEntry(adv_section, "CPCV test groups (k)", 2)
        self.champ_simple_sens_pct = LabeledEntry(adv_section, "Sensitivity range (\u00b1 fraction, e.g. 0.5 = \u00b150%)", 0.5)
        self.champ_simple_sens_steps = LabeledEntry(adv_section, "Sensitivity steps", 9)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RE-RUN TICKED CHECKS", self._champion_simple_rerun_clicked, primary=True).pack(side="left")

        self.champ_simple_progress = NeuralProgress(f)
        self.champ_simple_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Champion output", "Re-run results for the selected strategy, in the same format as the Validate page.")
        self.champ_simple_output = self._log_box(output_section, height=18)

        self._champion_simple_refresh()

    def _log_champ_simple(self, msg: str):
        self.champ_simple_output.insert(END, msg + "\n")
        self.champ_simple_output.see(END)
        self.root.update_idletasks()

    def _champion_simple_refresh(self):
        try:
            rows = champion_board.list_board()
        except Exception:
            rows = []
        self._champ_rows_by_iid = {}
        for iid in self.champ_simple_tree.get_children():
            self.champ_simple_tree.delete(iid)
        for r in rows:
            iid = f"{r['strategy_type']}::{r['filename']}"
            self._champ_rows_by_iid[iid] = r
            notes = r["promotion"]["stage_title"]
            if r["promotion"]["can_promote"] and r["promotion"]["next_stage_title"]:
                notes += f" \u00b7 ready to promote to {r['promotion']['next_stage_title']}"
            elif r["promotion"]["next_stage_title"]:
                # v9.7: name the honest next ACTION -- an untested row's
                # next step is the Full Pipeline, never Validate.
                reqs = r["promotion"].get("requirements") or []
                bt_req = next((x for x in reqs if x.get("key") == "backtested"), None)
                if bt_req is not None and not bt_req.get("met"):
                    notes += " \u00b7 next step: \u25b6 run the Full Pipeline first (not tested yet)"
                else:
                    notes += f" \u00b7 next stage: {r['promotion']['next_stage_title']} \u2014 run the missing checks in Validate (pick checks)"
            if r["eval_pct"] is not None:
                notes += f" \u00b7 Eval {r['eval_pct']:.0f}%"
            if r["oos_pct"] is not None:
                notes += f" \u00b7 OOS {r['oos_pct']:.0f}%"
            self.champ_simple_tree.insert(
                "", "end", iid=iid,
                values=(r["display_name"], r["verdict"] or "no verdict yet", notes),
            )

    def _champion_simple_selected_row(self):
        selection = self.champ_simple_tree.selection()
        if not selection:
            messagebox.showinfo("Champion Board", "Select a strategy in the board above first.")
            return None
        return self._champ_rows_by_iid.get(selection[0])

    def _champion_simple_promote_clicked(self):
        row = self._champion_simple_selected_row()
        if row is None:
            return
        # The board's own real promotion mechanism -- the same one the
        # Dashboard's PROMOTE button calls. It advances one gated stage
        # and refuses (naming the unmet requirements) when the strategy
        # hasn't earned it; there is no separate "champion" marker to fake.
        try:
            ok, message, _new_stage = champion_board.promote_strategy(row["strategy_type"], row["filename"])
        except Exception as exc:  # pragma: no cover - defensive
            ok, message = False, f"Unexpected error: {exc}"
        self.champ_simple_status.config(text=message, fg=(GREEN if ok else AMBER))
        self._champion_simple_refresh()

    def _champion_simple_rerun_clicked(self):
        row = self._champion_simple_selected_row()
        if row is None:
            return
        if not self.champ_simple_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in this tab's Market Data section first.")
            return
        checks = {k for k, v in self.champ_chk_vars.items() if v.get()}
        if not checks:
            messagebox.showwarning("Nothing ticked", "Tick at least one check first.")
            return
        self.champ_simple_output.delete("1.0", END)
        self.champ_simple_progress.start(10)
        threading.Thread(
            target=self._champion_simple_rerun_pipeline, args=(row, checks), daemon=True,
        ).start()

    def _champion_simple_rerun_pipeline(self, row: dict, checks: set):
        log = self._log_champ_simple
        try:
            stored = next(
                (s for s in list_saved_strategies(row["strategy_type"]) if s.name == row["filename"]), None,
            )
            if stored is None:
                log(f"'{row['filename']}' is no longer in the Strategy Library -- refresh the board.")
                return

            def strategy_factory():
                return load_strategy_object(stored)

            df = self.champ_simple_context.load_dataframe(log)
            if df is None:
                return
            rules = self.champ_simple_context.build_prop_rules()
            risk = build_run_context(self.champ_simple_context.build_risk_config(), rules)
            mc_config = self._validation_mc_config(n_simulations=self.champ_simple_mc_sims.get_int(2000))
            log(f"Re-running {', '.join(sorted(checks))} for {row['display_name']} "
                f"on {self.champ_simple_context.instrument_label()}...")
            results = self._run_validation_checks(
                checks, strategy_factory, df, risk, rules, mc_config, log,
                cpcv_n_groups=self.champ_simple_cpcv_groups.get_int(6),
                cpcv_n_test_groups=self.champ_simple_cpcv_test_groups.get_int(2),
                sens_pct_range=self.champ_simple_sens_pct.get_float(0.5),
                sens_steps=self.champ_simple_sens_steps.get_int(9),
            )
            # Mirror the dedicated CPCV tab's library record so a re-run
            # here moves the board's OOS column exactly like a CPCV tab
            # run does. (The other checks' official records live on their
            # own tabs; this quick re-run reports them in place.)
            cpcv_res = results.get("cpcv")
            if cpcv_res is not None:
                try:
                    record_validation_result(row["strategy_type"], row["filename"], {
                        "method": "cpcv", "n_paths": cpcv_res.n_paths,
                        "is_robust": bool(cpcv_res.is_robust),
                        "efficiency": getattr(cpcv_res, "mean_oos_metric", None),
                    })
                    log("\nRecorded the CPCV result on the library entry (board refreshed).")
                except (FileNotFoundError, ValueError):
                    pass
            self.root.after(0, self._champion_simple_refresh)
            log("Done.")
        except StrategyError as exc:
            log(f"\nStrategy error: {exc}")
        except Exception as exc:
            log("\nUnexpected error:\n" + traceback.format_exc())
            log_crash("Champion Board (promote)", exc=exc)
        finally:
            self.champ_simple_progress.stop()

    # -----------------------------------------------------------------------
    # Optimize Overview / Picker + Validate Overview / Checklist -- desktop
    # parity with the web app's lightweight /optimize and /validate hub
    # pages (app.reports.strategy_state): a guided picker for OPTIMIZE
    # (every method underneath is still its own full tab, this just helps
    # decide which one) and an aggregated checklist for VALIDATE, built
    # from whatever WFO/WFGA/CPCV/Sensitivity/Regime Matrix has already
    # recorded against the strategy marked "current".
    #
    # RESOLVED (2026-09): strategy_state.record_validation() is now called
    # from desktop's own WFO/WFGA/CPCV/Sensitivity/Regime Matrix tabs at
    # the end of each successful run (_wfo_run_pipeline, _cpcv_run_pipeline,
    # _sens_run_pipeline, _wfga_run_pipeline, _run_regime_matrix_pipeline),
    # mirroring the summary/passed conventions of app.web.server's own five
    # routes. Since data/config/current_strategy.json and
    # validation_checklist.json are shared by both apps on the same machine
    # (get_app_base_dir()), this Validate Overview tab now reflects
    # checklist progress from EITHER a web or a desktop-only run.
    # -----------------------------------------------------------------------

    def _build_optimize_hub_tab(self):
        f = self._scrollable(self.tab_optimize_hub)
        self._page_header(
            f,
            "OPTIMIZE / Overview \u2022 Picker",
            "\u2261 Which OPTIMIZE tool should I use?",
            "Every method below is still its own full tab -- nothing is lost by using this "
            "picker first, it just answers \"which one\" before sending you on.",
        )
        picks = [
            ("Full Pipeline", "fullpipeline", "One button, the whole workflow -- backtest, GA search, re-validate, holdout check. Start here if you're not sure."),
            ("Quick Optimize", "quickoptimize", "Just the GA search step, fast, with no OOS holdout check. Good for a quick \"is this worth pursuing\" signal -- NOT a finished answer."),
            ("Search Lab", "search", "Wide discovery across many candidate configurations (single strategy, a parameter grid, or a whole named family)."),
            ("Evolution Lab (GA)", "evolution", "A long-running, checkpointed genetic search -- start it and let it run in the background for hours."),
            ("Multi-Objective Optimization", "multiobj", "Search Lab, but optimizing several competing objectives (e.g. return AND drawdown) at once instead of one fitness metric."),
            ("Multi-Instrument Search Lab", "searchmulti", "The same Search Lab space run concurrently across several instrument/timeframe datasets."),
            ("Multi-Instrument Evolution Lab", "evolutionmulti", "The same Evolution Lab run concurrently across several instrument/timeframe datasets, each with its own checkpoint."),
            ("Iterative Refinement", "refine", "Small, targeted nudges to an already-decent strategy's parameters -- not a wide search."),
            ("Risk Sweep", "risksweep", "Not a strategy-parameter search at all -- finds which RISK-PER-TRADE level maximizes prop-survival odds for the strategy as configured."),
        ]
        for label, key, desc in picks:
            row = self._section(f, label, desc)
            self._button(row, f"OPEN {label.upper()}", lambda k=key: self._show_page(k)).pack(anchor="w", padx=18, pady=(2, 12))

    def _open_autopilot_report(self):
        if self._last_autopilot_report_path:
            webbrowser.open(f"file://{Path(self._last_autopilot_report_path).resolve()}")

    def _autopilot_run_clicked(self):
        if not self.sr_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        if not self._try_start_heavy_job(JOB_SPEED_RUN):
            return
        self.speedrun_output.delete("1.0", END)
        self.speedrun_candidates_listbox.delete(0, END)
        self._speedrun_candidates_cache = []
        self.autopilot_open_report_btn.config(state="disabled")
        self.speedrun_verdict_label.config(text="Running Overnight Autopilot...", fg=TEXT_DIM)
        self._speedrun_cancel_event.clear()
        self.speedrun_progress.start(10)
        threading.Thread(target=self._autopilot_run_pipeline, daemon=True).start()

    def _autopilot_run_pipeline(self):
        try:
            df = self.sr_context.load_dataframe(self._log_speedrun)
            if df is None:
                return
            risk = self.sr_context.build_risk_config()
            rules = self.sr_context.build_prop_rules()

            metric_key = self._sr_metric_label_to_key.get(self.sr_metric.get_str(), "eval_pass_probability")
            speed_run_cfg = SpeedRunConfig(
                max_candidates=self.sr_max_candidates.get_int(1200),
                stage1_top_n=self.sr_stage1_top_n.get_int(24),
                ga_population=self.sr_ga_population.get_int(8),
                ga_generations=self.sr_ga_generations.get_int(3),
                top_k_to_validate=self.sr_top_k.get_int(3),
                max_concurrent_validations=self.sr_max_concurrent.get_int(2),
                validation_folds=self.sr_validation_folds.get_int(3),
                validation_final_mc_sims=self.sr_validation_final_mc_sims.get_int(3000),
                fitness_metric=metric_key,
                save_winner_to_library=self.sr_save_to_library.get(),
                random_seed=self.sr_seed.get_int(42),
                reset_on_breach=True,
            )
            autopilot_cfg = AutopilotConfig(
                speed_run_cfg=speed_run_cfg,
                auto_forward_test=self.autopilot_auto_forward_test.get(),
                forward_test_risk_value_pct=self.autopilot_ft_risk_pct.get_float(1.0),
                forward_test_max_trades_per_day=self.autopilot_ft_max_trades_per_day.get_int(10),
                report_dir=OUTPUT_DIR / "autopilot",
            )

            instrument = (
                os.path.basename(self.sr_context.csv_paths[0])
                if len(self.sr_context.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.sr_context.csv_paths)
            )
            self._log_speedrun(f"Starting Overnight Autopilot on {instrument} ({len(df)} bars)...\n")
            result = run_overnight_autopilot(
                df, risk, rules, OUTPUT_DIR / "speed_run", autopilot_cfg,
                progress_cb=self._log_speedrun, instrument=instrument,
                cancel_event=self._speedrun_cancel_event,
            )

            self._render_speedrun_candidates(result.speed_run)
            self._last_autopilot_report_path = result.report_path
            self.autopilot_open_report_btn.config(state="normal")

            verdict_color = {"READY": GREEN, "MARGINAL": AMBER}.get(result.winner_verdict, TEXT_DIM if not result.winner_verdict else RED)
            self.speedrun_verdict_label.config(
                text=(
                    f"Autopilot done -- verdict: {result.winner_verdict or 'no winner'}\n"
                    f"Forward test: {result.forward_test_message}"
                ),
                fg=verdict_color,
            )
            if result.speed_run is not None and result.speed_run.winner is not None and result.speed_run.winner.pipeline_result is not None:
                pr = result.speed_run.winner.pipeline_result
                self._last_speedrun_html_path = pr.report_paths.get("html")
                self.open_speedrun_report_btn.config(state="normal" if self._last_speedrun_html_path else "disabled")

            self._log_speedrun(f"\nAutopilot report: {result.report_path}")
            self._log_speedrun(f"Done in {result.elapsed_seconds / 60:.1f} minute(s).")
        except Exception as exc:
            self._log_speedrun("\nUnexpected error:\n" + traceback.format_exc())
            self.speedrun_verdict_label.config(text="Autopilot failed -- see log.", fg=RED)
            log_crash("Overnight Autopilot", exc=exc)
        finally:
            self.speedrun_progress.stop()
            self._release_heavy_job(JOB_SPEED_RUN)
