"""Batch operations over library strategies (progress window, batch tests, exports)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ui.widgets.run_context import RunContextPanel

import os
import sys
import threading
import traceback
import webbrowser
from app.orchestration import pipeline_guide
from app.orchestration.batch_test import BatchTestItem, run_batch_test
from app.orchestration.full_pipeline import FullPipelineBatchItem, FullPipelineConfig, run_full_pipeline_batch
from app.strategy.library import STRATEGY_TYPES, export_library_zip, get_strategy_library_dir, list_saved_strategies
from pathlib import Path
from tkinter import BooleanVar, Checkbutton, END, Frame, Label, Text, Toplevel, filedialog, messagebox, ttk
from app.ui.common import OUTPUT_DIR, _safe_font
from app.ui.platform_win import _apply_dark_titlebar
from app.ui.theme import MONO
from app.ui.widgets.run_context import RunContextDialog
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
PANEL_2 = ""
PANEL_3 = ""
BORDER = ""
TEXT = ""
TEXT_DIM = ""
GREEN = ""
theme.bind(globals())


class LibraryBatchMixin:
    """Batch operations over library strategies (progress window, batch tests, exports)."""


    def _open_progress_window(self, title: str):
        """A small Toplevel with a scrolling log -- used for TEST SELECTED
        (BATCH), OPTIMIZE SELECTED, and RUN FULL PIPELINE (BATCH) so a
        multi-strategy run has somewhere to show progress without
        borrowing the Search Lab tab's own console.

        Also builds (but keeps hidden until populated) a per-strategy
        RESULTS table with an OPEN REPORT button/double-click and an OPEN
        ALL REPORTS button. Before this, a batch run of several strategies
        only ever showed a plain text log -- each strategy's own report
        WAS written to disk (run_batch_test / run_full_pipeline_batch
        always wrote one full report per strategy, see
        app.orchestration.batch_test / app.orchestration.full_pipeline),
        but there was no way to open any of them except the single
        strategy's report from the last time you happened to run ONE
        strategy through 05 Run & Report or 15 Full Pipeline directly --
        which reads exactly like "it only tested one and gave one report"
        even though every strategy in the batch really was tested
        individually the whole time. Call the returned `show_results()`
        with the batch's list of outcome objects/dicts (anything with
        `label`/`ok`/`verdict`/`eval_pass_probability`/`net_profit`/
        `trades`/`report_html` attributes or keys -- BatchTestOutcome and
        FullPipelineBatchOutcome both already match this shape) once the
        run finishes to reveal it."""
        win = Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        win.geometry("820x680")
        try:
            _apply_dark_titlebar(win)
        except Exception:
            pass  # cosmetic only -- see _apply_dark_titlebar's own docstring
        Label(win, text=title, bg=BG, fg=TEXT, font=_safe_font(11, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        text_frame = Frame(win, bg=BG)
        text_frame.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        txt = Text(text_frame, wrap="word", bg=PANEL_3, fg=TEXT, font=(MONO, 9), relief="flat", bd=0, height=14)
        vs = ttk.Scrollbar(text_frame, orient="vertical", command=txt.yview, style="T58.Vertical.TScrollbar")
        txt.config(yscrollcommand=vs.set)
        self._bind_isolated_wheel(txt)
        txt.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")

        # Results table -- built now, packed later (by show_results, once
        # there's at least one outcome to show) so a run that's still in
        # progress doesn't show an empty table above the CLOSE button.
        results_frame = Frame(win, bg=BG)
        Label(
            results_frame, text="RESULTS -- one row per strategy tested, each with its own report",
            bg=BG, fg=TEXT_DIM, font=_safe_font(8, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        tree_wrap = Frame(results_frame, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        tree_wrap.pack(fill="both", expand=True)
        columns = ("strategy", "verdict", "eval_pass", "net_profit", "trades")
        results_tree = ttk.Treeview(
            tree_wrap, columns=columns, show="headings", style="T58.Treeview", height=7,
        )
        for col, text, width in (
            ("strategy", "Strategy", 220), ("verdict", "Verdict / status", 200),
            ("eval_pass", "Eval Pass %", 90), ("net_profit", "Net Profit", 110), ("trades", "Trades", 70),
        ):
            results_tree.heading(col, text=text)
            results_tree.column(col, width=width, anchor="w")
        results_tree.pack(side="left", fill="both", expand=True)
        results_tree_scroll = ttk.Scrollbar(
            tree_wrap, orient="vertical", command=results_tree.yview, style="T58.Vertical.TScrollbar",
        )
        results_tree_scroll.pack(side="right", fill="y")
        results_tree.configure(yscrollcommand=results_tree_scroll.set)
        self._bind_isolated_wheel(results_tree)

        _report_paths_by_row: dict[str, object] = {}

        def _open_selected_report(_evt=None):
            sel = results_tree.selection()
            if not sel:
                messagebox.showinfo("No row selected", "Select a strategy's row above first.")
                return
            path = _report_paths_by_row.get(sel[0])
            if not path:
                messagebox.showinfo(
                    "No report", "This strategy didn't produce a report (it failed, was skipped, or "
                    "generated zero trades).",
                )
                return
            webbrowser.open(f"file://{Path(path).resolve()}")

        results_tree.bind("<Double-1>", _open_selected_report)

        results_btn_row = Frame(results_frame, bg=BG)
        results_btn_row.pack(fill="x", pady=(6, 0))
        self._button(results_btn_row, "OPEN REPORT", _open_selected_report, primary=True).pack(side="left")

        def _open_all_reports():
            paths = [p for p in _report_paths_by_row.values() if p]
            if not paths:
                messagebox.showinfo("No reports", "None of these strategies produced a report to open.")
                return
            for path in paths:
                webbrowser.open(f"file://{Path(path).resolve()}")

        self._button(results_btn_row, "OPEN ALL REPORTS", _open_all_reports).pack(side="left", padx=8)

        btn_row = Frame(win, bg=BG)
        btn_row.pack(fill="x", padx=14, pady=(0, 12))
        self._button(btn_row, "CLOSE", win.destroy).pack(side="left")

        def append(msg: str) -> None:
            def _do():
                txt.insert(END, msg + "\n")
                txt.see(END)
            try:
                self.root.after(0, _do)
            except Exception:
                pass

        def show_results(outcomes) -> None:
            """Populates and reveals the results table -- one row per
            strategy in `outcomes`, sorted successes-first by eval-pass
            probability. Safe to call from a background thread (routes
            through root.after like `append` does)."""
            def _get(o, name, default=None):
                return o.get(name, default) if isinstance(o, dict) else getattr(o, name, default)

            def _do():
                for row in results_tree.get_children():
                    results_tree.delete(row)
                _report_paths_by_row.clear()

                ranked = sorted(
                    outcomes,
                    key=lambda o: (not _get(o, "ok", True), -(_get(o, "eval_pass_probability") or 0)),
                )
                for o in ranked:
                    label = _get(o, "label", "?")
                    if _get(o, "ok", True):
                        verdict = _get(o, "verdict") or "OK"
                        eval_pass = _get(o, "eval_pass_probability")
                        net_profit = _get(o, "net_profit")
                        trades = _get(o, "trades")
                        values = (
                            label, verdict,
                            f"{eval_pass:.1f}" if eval_pass is not None else "--",
                            f"${net_profit:,.2f}" if net_profit is not None else "--",
                            trades if trades is not None else "--",
                        )
                    else:
                        reason = _get(o, "reason") or "failed"
                        values = (label, f"FAILED -- {reason}", "--", "--", "--")
                    iid = results_tree.insert("", END, values=values)
                    _report_paths_by_row[iid] = _get(o, "report_html")

                if outcomes and not results_frame.winfo_ismapped():
                    results_frame.pack(fill="both", expand=True, padx=14, pady=(0, 8), before=btn_row)

            try:
                self.root.after(0, _do)
            except Exception:
                pass

        return win, append, show_results

    def _refresh_batch_checklist(self):
        """Rebuilds the Batch selection checkbox panel from every saved
        strategy across all 4 types (python/pinescript/mql5/manual) --
        the desktop counterpart to the web app's flat section-6 checkbox
        list. Existing BooleanVars are re-used by (type, name) key so a
        checked box survives a refresh (e.g. after a save/delete/rename
        elsewhere in the library) instead of resetting to unchecked."""
        if threading.current_thread() is not threading.main_thread():
            try:
                self.root.after(0, self._refresh_batch_checklist)
            except Exception:
                pass
            return
        for child in self._batch_checklist_inner.winfo_children():
            child.destroy()
        self._batch_checklist_items = {}
        old_vars = self._batch_checked_vars
        self._batch_checked_vars = {}
        total = 0
        for stype in STRATEGY_TYPES:
            for item in list_saved_strategies(stype):
                key = (stype, item.name)
                self._batch_checklist_items[key] = item
                var = old_vars.get(key) or BooleanVar(value=False)
                self._batch_checked_vars[key] = var
                Checkbutton(
                    self._batch_checklist_inner,
                    text=f"[{stype}] [{item.status_display}] {item.name}",
                    variable=var, bg=PANEL_3, fg=self._status_color(item.status),
                    selectcolor=PANEL_2, activebackground=PANEL_3, activeforeground=TEXT,
                    font=(MONO, 9), anchor="w", relief="flat", highlightthickness=0, padx=6, bd=0,
                    cursor="hand2", command=self._refresh_batch_checklist_status,
                ).pack(fill="x", anchor="w")
                total += 1
        if total == 0:
            Label(
                self._batch_checklist_inner,
                text="No saved strategies yet -- save one from 01 Strategy Configuration "
                "(or run Evolution Lab / Search Lab) before using batch selection.",
                bg=PANEL_3, fg=TEXT_DIM, font=_safe_font(9), wraplength=760, justify="left",
            ).pack(anchor="w", padx=6, pady=6)
        self._refresh_batch_checklist_status()

    def _refresh_batch_checklist_status(self):
        checked = sum(1 for v in self._batch_checked_vars.values() if v.get())
        total = len(self._batch_checked_vars)
        if total == 0:
            self.batch_checklist_status.config(text="No saved strategies yet.", fg=TEXT_DIM)
        elif checked == 0:
            self.batch_checklist_status.config(
                text=f"0 of {total} checked -- check the strategies you want, then click a RUN button below.",
                fg=TEXT_DIM,
            )
        else:
            self.batch_checklist_status.config(
                text=f"{checked} of {total} checked -- click RUN BATCH TEST (CHECKED) or "
                "RUN FULL PIPELINE (BATCH) (CHECKED) when ready.",
                fg=GREEN,
            )

    def _batch_checklist_select_all(self):
        for var in self._batch_checked_vars.values():
            var.set(True)
        self._refresh_batch_checklist_status()

    def _batch_checklist_clear_all(self):
        for var in self._batch_checked_vars.values():
            var.set(False)
        self._refresh_batch_checklist_status()

    def _get_checked_batch_items(self) -> list:
        return [
            self._batch_checklist_items[key]
            for key, var in self._batch_checked_vars.items()
            if var.get() and key in self._batch_checklist_items
        ]

    def _run_batch_checked_clicked(self):
        items = self._get_checked_batch_items()
        if not items:
            messagebox.showinfo(
                "Nothing checked",
                "Check one or more strategies in the Batch selection panel first.",
            )
            return
        context = RunContextDialog.ask(self.root, self, "Test Checked (Batch)")
        if context is None:
            return
        win, append, show_results = self._open_progress_window(f"Testing {len(items)} strategy(ies)...")
        threading.Thread(
            target=self._run_library_batch_test_pipeline, args=(items, append, show_results, context), daemon=True,
        ).start()

    def _run_library_batch_test_pipeline(self, items, log, show_results=None, context: "RunContextPanel" = None):
        """Runs every checked Strategy Library item through
        app.orchestration.batch_test.run_batch_test -- the same pipeline
        Bulk Backtest already uses, just sourced from the Batch selection
        checklist instead of a fresh file upload, and recording each result back onto
        that strategy's own library metadata. Market data, Prop Rules, and Risk are
        resolved from the RunContextDialog confirmed when TEST CHECKED (BATCH) was
        clicked -- self-contained to this run, independent of Steps 02-04."""
        try:
            df = context.load_dataframe(log)
            if df is None:
                return

            risk = context.build_risk_config()
            rules = context.build_prop_rules()
            n_sims = self.mc_sims.get_int(10000)
            method = self.mc_method.get_str().strip() or "bootstrap"
            instrument = context.instrument_label()

            batch_items = []
            for item in items:
                try:
                    strategy = self._load_bulk_strategy(item.path)
                except Exception as exc:
                    log(f"  Skipped {item.name} -- could not load: {exc}")
                    continue
                batch_items.append(
                    BatchTestItem(label=item.name, strategy=strategy, library_ref=(item.strategy_type, item.name))
                )

            if not batch_items:
                log("\nNothing to test -- every queued strategy failed to load.")
                return

            modes_present = {item.strategy_type for item in items}
            prefix_mode = modes_present.pop() if len(modes_present) == 1 else "mixed"

            summary = run_batch_test(
                df, batch_items, risk, rules, OUTPUT_DIR,
                instrument=instrument, mc_sims=n_sims, mc_method=method,
                basename_prefix=f"library_{prefix_mode}", progress_cb=log,
            )
            if summary.succeeded:
                ranked = sorted(summary.succeeded, key=lambda o: o.eval_pass_probability, reverse=True)
                log("\nRanked by eval pass probability:")
                for o in ranked:
                    log(f"  {o.eval_pass_probability:5.1f}%  ${o.net_profit:>12,.2f}   {o.label}")
            if show_results is not None:
                show_results(summary.outcomes)
            # This whole method runs on a background thread (see
            # _run_batch_checked_clicked) -- Tkinter widgets can only safely
            # be touched from the main thread, so both refreshes are handed
            # to the mainloop via root.after instead of being called here
            # directly. Calling them straight from a worker thread is what
            # produces the occasional glitchy/frozen UI after a batch run.
            def _refresh_after_run():
                try:
                    self._refresh_dashboard()
                except Exception:
                    pass
                try:
                    self._refresh_strategy_library()
                except Exception:
                    pass
            try:
                self.root.after(0, _refresh_after_run)
            except Exception:
                pass
        except Exception:
            log("\nUnexpected error:\n" + traceback.format_exc())

    def _optimize_selected_library_strategies(self):
        """OPTIMIZE SELECTED -- runs app.orchestration.quick_optimize against
        every currently-highlighted Strategy Library row. Unlike the Batch
        test queue, this acts directly on whatever's selected in the list
        right now (no separate queue/stage step), since it's meant as a
        quick "is this worth a real Full Pipeline run" check.

        Self-contained like the web app's Quick Optimize page: confirming
        the RunContextDialog below resolves market data + prop rules +
        risk settings just for THIS click, independent of Steps 02-04."""
        items = self._selected_library_items()
        if not items:
            messagebox.showinfo(
                "No selection",
                "Select one or more saved strategies from the list above first "
                "(Ctrl/Cmd-click or Shift-click for more than one).",
            )
            return
        context = RunContextDialog.ask(self.root, self, "Quick Optimize")
        if context is None:
            return
        win, append, _show_results = self._open_progress_window(f"Optimizing {len(items)} strategy(ies)...")
        threading.Thread(
            target=self._run_library_quick_optimize, args=(items, append, context), daemon=True,
        ).start()

    def _run_library_quick_optimize(self, items, log, context: "RunContextPanel"):
        from app.orchestration.quick_optimize import QuickOptimizeConfig, run_quick_optimize

        try:
            df = context.load_dataframe(log)
            if df is None:
                return

            risk = context.build_risk_config()
            rules = context.build_prop_rules()
            cfg = QuickOptimizeConfig(
                adaptive_risk_enabled=self.lib_optimize_adaptive_risk.var.get(),
                reset_on_breach=True,
                reserve_holdout=self.lib_optimize_reserve_holdout.var.get(),
            )
            results = []
            for i, item in enumerate(items, start=1):
                log(f"===== [{i}/{len(items)}] Optimizing: {item.name} =====")
                try:
                    strategy = self._load_bulk_strategy(item.path)
                except Exception as exc:
                    log(f"  Skipped -- could not load: {exc}\n")
                    continue
                try:
                    res = run_quick_optimize(df, strategy, risk, rules, cfg, progress_cb=lambda m: log(f"  {m}"), instrument_label=os.path.basename(context.csv_paths[0]) if getattr(context, "csv_paths", None) else "")
                    results.append((item.name, res))
                    if res.instrument_mismatch_warning:
                        log(f"  !!! WARNING for {item.name}: {res.instrument_mismatch_warning}")
                except Exception as exc:
                    log(f"  Optimize failed: {exc}\n")
                    continue
                log("")

            if results:
                # 2026-09-17 fix, points (1) and (2): this summary line is
                # exactly the kind of clean-looking percentage that made
                # the original Quick-Optimize-vs-Full-Pipeline discrepancy
                # so easy to misread as a finished result -- the per-item
                # log above already prints the full "NOT OOS VALIDATED"
                # banner and (when relevant) the too-few-trades warning,
                # but this rolled-up summary line needs its own reminder
                # too, since it's the line most likely to be read on its
                # own after a long batch scrolls past.
                log("Summary (eval-pass probability, before -> after) -- ⚠️ NOT OOS VALIDATED, see Full Pipeline:")
                for name, res in sorted(results, key=lambda t: t[1].optimized_eval_pass_probability, reverse=True):
                    marker = "IMPROVED" if res.improved else "no improvement"
                    mismatch_flag = "  [!!! PIP-SIZE MISMATCH -- UNRELIABLE, SEE ABOVE]" if res.instrument_mismatch_warning else ""
                    trade_count_flag = "  [!!! TOO FEW OOS TRADES -- SEE ABOVE]" if not res.min_trade_count_met else ""
                    log(
                        f"  {name}: {res.baseline_eval_pass_probability:.1f}% -> "
                        f"{res.optimized_eval_pass_probability:.1f}%  ({marker}){mismatch_flag}{trade_count_flag}"
                    )

            def _refresh_after_run():
                try:
                    self._refresh_strategy_library()
                except Exception:
                    pass
            try:
                self.root.after(0, _refresh_after_run)
            except Exception:
                pass
        except Exception:
            log("\nUnexpected error:\n" + traceback.format_exc())

    def _run_full_pipeline_checked_clicked(self):
        items = self._get_checked_batch_items()
        if not items:
            messagebox.showinfo(
                "Nothing checked",
                "Check one or more strategies in the Batch selection panel first.",
            )
            return
        if not self.fp_context.csv_paths:
            messagebox.showwarning(
                "Missing data",
                "This reads its market data/prop rules/risk from the Full Pipeline tab's own "
                "Market Data section -- open the Full Pipeline tab and select a dataset there first.",
            )
            return
        proceed = messagebox.askokcancel(
            "Run Full Pipeline on multiple strategies?",
            f"This runs the FULL Full Pipeline (baseline, walk-forward-aware GA search, "
            f"re-validated Monte Carlo, out-of-sample check, holdout check, ICIR gate) for all "
            f"{len(items)} checked strategy(ies), one after another. This is much slower per "
            f"strategy than RUN BATCH TEST -- for a large batch (dozens to hundreds) this can "
            f"run for a long time in the background. Continue?",
        )
        if not proceed:
            return
        win, append, show_results = self._open_progress_window(f"Full Pipeline: {len(items)} strategy(ies)...")
        threading.Thread(
            target=self._run_library_full_pipeline_batch, args=(items, append, show_results), daemon=True,
        ).start()

    def _run_library_full_pipeline_batch(self, items, log, show_results=None):
        """Runs every checked Strategy Library item through
        app.orchestration.full_pipeline.run_full_pipeline_batch -- the same
        FULL 7-step pipeline the 15 Full Pipeline tab's single-strategy
        RUN FULL PIPELINE button uses, just looped over every strategy
        staged in the Batch selection checklist instead of one loaded strategy.
        Market data, Prop Rules, Risk, GA population/generations/etc., and AI
        Assist settings are all read fresh from the Full Pipeline tab's own
        self-contained RunContextPanel/widgets, so this batch action runs
        with exactly whatever that tab is currently configured with."""
        try:
            df = self.fp_context.load_dataframe(log)
            if df is None:
                return

            risk = self.fp_context.build_risk_config()
            rules = self.fp_context.build_prop_rules()
            instrument = self.fp_context.instrument_label()

            # Same settings the 15 Full Pipeline tab's own RUN FULL PIPELINE
            # button reads -- see _fullpipeline_run_pipeline. Falls back to
            # FullPipelineConfig()'s defaults if that tab was never opened
            # this session (its widgets are only created once the tab is
            # built, which happens unconditionally at startup, so this
            # should always be available in practice).
            try:
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
                ollama_settings = self._build_ollama_settings()
            except Exception:
                cfg = FullPipelineConfig(reset_on_breach=True)
                ollama_settings = None
            log(
                f"Using Full Pipeline settings from the 15 Full Pipeline tab: "
                f"population={cfg.ga_population}, generations={cfg.ga_generations}, "
                f"folds={cfg.n_folds}.\n"
            )

            batch_items = []
            for item in items:
                try:
                    strategy = self._load_bulk_strategy(item.path)
                except Exception as exc:
                    log(f"  Skipped {item.name} -- could not load: {exc}")
                    continue
                batch_items.append(
                    FullPipelineBatchItem(label=item.name, strategy=strategy, library_ref=(item.strategy_type, item.name))
                )

            if not batch_items:
                log("\nNothing to test -- every queued strategy failed to load.")
                return

            # Deliberately serial (max_parallel_strategies=1), not auto-
            # scaled off the core count. This used to run several
            # strategies' pipelines concurrently via ProcessPoolExecutor,
            # which cut wall-clock time but cost two things Owen actually
            # relies on for this specific button: (1) run_full_pipeline_batch
            # can't pass a live progress_cb across a process boundary, so the
            # parallel path only ever logs a start/finish line per strategy
            # instead of every step (baseline, GA generations, OOS/holdout
            # checks, ...) -- which is exactly the "it only showed the final
            # verdict, not all the steps" regression this restores. (2) a
            # worker process dying mid-run (OOM, a native crash inside
            # numpy/pandas under concurrent load, etc.) is a much larger,
            # harder-to-diagnose failure surface than a single-process loop
            # where one bad strategy's exception is caught right where it
            # happens -- the most likely explanation for a batch crashing
            # partway through a run of several strategies. Step 2's GA
            # search inside EACH strategy's own pipeline still parallelizes
            # internally (see cfg.parallel_search_max_workers) -- this only
            # removes the outer, less safe layer of parallelism ACROSS
            # strategies. Pass max_parallel_strategies>1 to
            # run_full_pipeline_batch directly (e.g. from a script) if raw
            # throughput matters more than per-step visibility for a given
            # run.
            summary = run_full_pipeline_batch(
                df, batch_items, risk, rules, OUTPUT_DIR / "full_pipeline",
                cfg=cfg, instrument=instrument, ollama_settings=ollama_settings, progress_cb=log,
                max_parallel_strategies=1,
            )
            if summary.succeeded:
                ranked = sorted(summary.succeeded, key=lambda o: o.eval_pass_probability, reverse=True)
                log("\nRanked by eval pass probability:")
                for o in ranked:
                    log(f"  {o.eval_pass_probability:5.1f}%  ${o.net_profit:>12,.2f}   [{o.verdict}]   {o.label}")
            if summary.failed:
                log("\nFailed / skipped:")
                for o in summary.failed:
                    log(f"  {o.label}: {o.reason}")

            if summary.outcomes:
                log(
                    "\n" + pipeline_guide.after_full_pipeline_batch([
                        {"label": o.label, "ok": o.ok, "verdict": o.verdict,
                         "eval_pass_probability": o.eval_pass_probability}
                        for o in summary.outcomes
                    ])
                )
            if show_results is not None:
                show_results(summary.outcomes)

            # Background thread -- see _run_library_batch_test_pipeline for
            # why these refreshes are routed through root.after instead of
            # called directly.
            def _refresh_after_run():
                try:
                    self._refresh_dashboard()
                except Exception:
                    pass
                try:
                    self._refresh_strategy_library()
                except Exception:
                    pass
            try:
                self.root.after(0, _refresh_after_run)
            except Exception:
                pass
        except Exception:
            log("\nUnexpected error:\n" + traceback.format_exc())

    def _open_strategy_library_folder(self):
        # This folder-open button is a frequent source of confusion: people
        # naturally double-click a .py/.pine/.mq5 file once Explorer/Finder
        # is open, which fails with an OS-level "no program configured to
        # open this file" error (Windows has no default app registered for
        # those extensions). That error comes from Windows itself, not this
        # app, and it can't be silently fixed here -- so we head it off with
        # an explicit heads-up *before* opening the folder, rather than
        # relying on the small print label under the button, which is easy
        # to miss or scroll past. Cancelling here skips opening the folder
        # entirely, so a person who reads the warning never sees the error.
        proceed = messagebox.askokcancel(
            "Before you open this folder",
            "This opens the raw strategy files on disk -- useful for backing "
            "them up or copying them to another device.\n\n"
            "Don't double-click a strategy file from there: Windows has no "
            "program registered for .py / .pine / .mq5 files and will show "
            "\"no program configured to open this file.\" That's expected "
            "and isn't something this app can fix.\n\n"
            "To actually use a saved strategy, come back to this screen, "
            "select it in the list above, and click LOAD SELECTED (or "
            "double-click it there instead).\n\n"
            "Click OK to open the folder anyway, or Cancel to go back.",
        )
        if not proceed:
            return
        d = get_strategy_library_dir(self.strategy_mode.get()) \
            if self.strategy_mode.get() in STRATEGY_TYPES else get_strategy_library_dir()
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)  # noqa: S606 -- opening a known local folder, not user input
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(d)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(d)])
        except OSError as exc:
            messagebox.showinfo("Strategy library folder", f"{d}\n\n(Couldn't open it automatically: {exc})")

    def _export_strategy_library(self):
        dest = filedialog.asksaveasfilename(
            title="Export strategy library",
            defaultextension=".zip",
            initialfile="t58_strategy_library_backup.zip",
            filetypes=[("Zip archive", "*.zip")],
        )
        if not dest:
            return
        try:
            export_library_zip(dest)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo(
            "Export complete",
            f"Exported the full strategy library to:\n{dest}\n\n"
            "Unzip it into your repo's strategies/ folder to keep a packaged "
            ".exe's saved strategies in sync with GitHub.",
        )

    def _export_selected_library_strategies(self):
        mode = self.strategy_mode.get()
        items = self._selected_library_items()
        if not items:
            messagebox.showinfo("No selection", "Select one or more saved strategies from the list first.")
            return
        dest = filedialog.asksaveasfilename(
            title="Export selected strategies",
            defaultextension=".zip",
            initialfile="t58_strategy_selection.zip",
            filetypes=[("Zip archive", "*.zip")],
        )
        if not dest:
            return
        try:
            export_library_zip(dest, selection=[(mode, i.name) for i in items])
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Export complete", f"Exported {len(items)} strategy(ies) to:\n{dest}")
