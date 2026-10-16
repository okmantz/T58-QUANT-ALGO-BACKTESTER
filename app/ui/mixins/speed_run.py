"""Speed Run (single, multi-instrument and loop) tabs."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.orchestration.speed_run import SpeedRunCandidateResult
    from app.orchestration.multi_instrument_speed_run import MultiInstrumentSpeedRunResult

import json
import os
import threading
import traceback
import webbrowser
from app.data.storage import get_raw_data_dir
from app.optimize.refinement import FITNESS_METRICS
from app.orchestration.loop_runner import SpeedRunLoopConfig, run_speed_run_loop
from app.orchestration.multi_instrument_search import InstrumentJob
from app.orchestration.multi_instrument_speed_run import (
    best_speed_run_across_instruments,
    run_multi_instrument_speed_run,
)
from app.orchestration.resource_guard import JOB_SPEED_RUN
from app.orchestration.speed_run import SpeedRunConfig, SpeedRunResult, _rank_key as _speedrun_rank_key, run_speed_run
from app.reports.crash_log import log_crash
from app.strategy.library import (
    StrategyAlreadyExists,
    record_backtest_result,
    save_strategy_text,
    set_strategy_status,
)
from pathlib import Path
from tkinter import END, Frame, Label, Listbox, Text, messagebox, ttk
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
TEXT = ""
TEXT_DIM = ""
LOG_BG = ""
GREEN = ""
RED = ""
AMBER = ""
theme.bind(globals())


class SpeedRunMixin:
    """Speed Run (single, multi-instrument and loop) tabs."""


    # -----------------------------------------------------------------------
    # Speed Run — "I have almost no time left, find me anything that
    # works." One button, no strategy to bring in: chains a wide
    # multi-family discovery search (app.search.batch_runner via
    # app.orchestration.speed_run) straight into concurrent Full Pipeline
    # validation of the top survivors, then reports whichever one is best.
    # Every setting this module exposes is a speed/thoroughness tradeoff,
    # never a correctness one -- see app.orchestration.speed_run's module
    # docstring. This is the one tab in the app that discovers its OWN
    # strategy rather than testing one you already configured on Step 01,
    # so it only needs Steps 02 (Data), 03 (Prop Rules), and 04 (Risk).
    # -----------------------------------------------------------------------

    def _build_speedrun_tab(self):
        f = self._scrollable(self.tab_speedrun)

        self._page_header(
            f,
            "CREATE / Speed Run",
            "Speed Run",
            "One button: searches EVERY registered strategy family at once for something "
            "that shows an edge, validates the best few candidates through Full Pipeline "
            "(walk-forward-aware GA + fresh Monte Carlo + out-of-sample check) running "
            "concurrently, and hands back whichever one is best -- READY, MARGINAL, or "
            "(if nothing cleared the bar) an honest 'no winner found'. This does NOT skip "
            "any check Full Pipeline or Search Lab would otherwise run; it only asks each "
            "of them to spend less time per candidate, and every existing safeguard "
            "(lookahead detection, pip-scale mismatch, stop-fill honesty, the account-blown "
            "circuit breaker) still runs exactly as it does everywhere else in this app. "
            "It finds its own strategy, so Step 01 (Strategy Configuration) is skipped "
            "entirely -- and its own market data, prop rules, and risk settings below are "
            "self-contained to this tab, just like the web app. Depending on your "
            "data size and machine, a run typically takes anywhere from several minutes to "
            "an hour or more; use the settings below to trade thoroughness for speed.",
        )

        self.sr_context = RunContextPanel(self, "Speed Run")
        self.sr_context.build(f)

        settings = self._section(
            f, "Speed Run settings",
            "Defaults are already tuned toward speed over thoroughness -- lower them further "
            "for a quicker, rougher pass, or raise them if you have more time to spend before "
            "the deadline.",
            emphasize=True,
        )
        self.sr_max_candidates = LabeledEntry(settings, "Discovery: max candidates sampled (all families combined)", 1200)
        self.sr_stage1_top_n = LabeledEntry(settings, "Discovery: Stage 1 survivors kept", 24)
        self.sr_ga_population = LabeledEntry(settings, "Discovery: GA population", 8)
        self.sr_ga_generations = LabeledEntry(settings, "Discovery: GA generations", 3)
        self.sr_top_k = LabeledEntry(settings, "Candidates to validate through Full Pipeline", 3)
        self.sr_max_concurrent = LabeledEntry(settings, "Max concurrent validations", 2)
        self.sr_validation_folds = LabeledEntry(settings, "Validation: walk-forward/OOS folds", 3)
        self.sr_validation_final_mc_sims = LabeledEntry(settings, "Validation: final Monte Carlo sims", 3000)
        self._sr_metric_labels = list(FITNESS_METRICS.values())
        self._sr_metric_label_to_key = {v: k for k, v in FITNESS_METRICS.items()}
        self.sr_metric = LabeledCombo(
            settings, "Fitness metric", self._sr_metric_labels, FITNESS_METRICS["eval_pass_probability"],
        )
        self.sr_seed = LabeledEntry(settings, "Random seed", 42)

        library_section = self._section(
            f, "Strategy Library",
            "Every validated candidate -- whether it's a Manual Strategy Builder / Search Lab "
            "config or an actual Python/PineScript/MQL5 file -- can be saved to the Strategy "
            "Library, either automatically below or on demand from the Validated Candidates list.",
        )
        self.sr_save_to_library = LabeledCheckbox(
            library_section, "Save every validated candidate to the Strategy Library when finished", True,
        )

        loop_section = self._section(
            f, "Loop mode",
            "Repeats Speed Run automatically: if a round finds no winner, the next round raises "
            "the candidate cap and validates more discoveries instead of stopping. Stops the "
            "moment a round actually finds a winner, when the round or time budget runs out, or "
            "when you hit STOP.",
        )
        self.sr_loop_mode = LabeledCheckbox(loop_section, "Enable Loop Mode", False)
        self.sr_loop_max_rounds = LabeledEntry(loop_section, "Max rounds", 10)
        self.sr_loop_time_budget_hours = LabeledEntry(
            loop_section, "Time budget in hours (blank = no limit, rely on rounds/STOP)", "",
        )
        self.sr_loop_stall_rounds = LabeledEntry(loop_section, "Rounds with no winner before widening", 2)

        autopilot_section = self._section(
            f, "Overnight Autopilot",
            "One button: runs Speed Run (using the settings above), and if it lands on a "
            "READY/MARGINAL winner, automatically starts forward-testing it live on your saved "
            "MT5 demo account (set up on the Forward Test tab first) -- then writes one dated "
            "report summarizing both. This can't finish a full forward-test evaluation overnight "
            "(forward testing is a live, ongoing process) -- what you get by morning is a "
            "completed search AND, if it qualified, a winner already live and trading on your "
            "demo account, instead of needing to notice the search finished and start the forward "
            "test yourself. Only Python/PineScript/MQL5 strategies can be forward-tested (same "
            "restriction as the Forward Test tab); a Manual-config winner still gets a full report, "
            "just without an auto-started forward test.",
        )
        self.autopilot_auto_forward_test = LabeledCheckbox(
            autopilot_section, "Auto-start a live MT5 demo forward test on a READY/MARGINAL winner", True,
        )
        self.autopilot_ft_risk_pct = LabeledEntry(autopilot_section, "Forward test risk per trade (%)", 1.0)
        self.autopilot_ft_max_trades_per_day = LabeledEntry(autopilot_section, "Forward test max trades/day", 10)
        autopilot_btn_row = Frame(autopilot_section, bg=PANEL)
        autopilot_btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.autopilot_run_btn = self._button(autopilot_btn_row, "RUN OVERNIGHT AUTOPILOT", self._autopilot_run_clicked, primary=True)
        self.autopilot_run_btn.pack(side="left")
        self.autopilot_open_report_btn = self._button(autopilot_btn_row, "OPEN AUTOPILOT REPORT", self._open_autopilot_report)
        self.autopilot_open_report_btn.config(state="disabled")
        self.autopilot_open_report_btn.pack(side="left", padx=8)
        self._last_autopilot_report_path = None

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "FIND ME A WINNER", self._speedrun_run_clicked, primary=True).pack(side="left")
        self.stop_speedrun_btn = self._button(button_row, "STOP", self._speedrun_stop_clicked)
        self.stop_speedrun_btn.config(state="disabled")
        self.stop_speedrun_btn.pack(side="left", padx=8)
        self.open_speedrun_report_btn = self._button(button_row, "OPEN WINNER REPORT", self._open_speedrun_winner_report)
        self.open_speedrun_report_btn.config(state="disabled")
        self.open_speedrun_report_btn.pack(side="left", padx=8)

        self.speedrun_progress = NeuralProgress(f)
        self.speedrun_progress.pack(fill="x", padx=24, pady=(2, 10))

        verdict_section = self._section(f, "Winner", "Filled in once a run completes.")
        self.speedrun_verdict_label = Label(
            verdict_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=820, anchor="w",
        )
        self.speedrun_verdict_label.pack(anchor="w", fill="x", padx=18, pady=(2, 10))

        candidates_section = self._section(
            f, "Validated candidates",
            "Every candidate that made it to Full Pipeline validation, ranked best first "
            "(READY beats MARGINAL beats anything else, tied-broken by eval pass probability). "
            "Double-click a row (or select it and click OPEN REPORT) to open that candidate's "
            "own full HTML report.",
        )
        cand_list_frame = Frame(candidates_section, bg=PANEL)
        cand_list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 6))
        cand_list_frame.columnconfigure(0, weight=1)
        cand_list_frame.rowconfigure(0, weight=1)
        self.speedrun_candidates_listbox = Listbox(
            cand_list_frame, height=8, exportselection=False, bg=PANEL_3, fg=TEXT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        sr_cand_scrollbar = ttk.Scrollbar(
            cand_list_frame, orient="vertical", command=self.speedrun_candidates_listbox.yview,
            style="T58.Vertical.TScrollbar",
        )
        self.speedrun_candidates_listbox.config(yscrollcommand=sr_cand_scrollbar.set)
        self._bind_isolated_wheel(self.speedrun_candidates_listbox)
        self.speedrun_candidates_listbox.grid(row=0, column=0, sticky="nsew")
        sr_cand_scrollbar.grid(row=0, column=1, sticky="ns")
        self.speedrun_candidates_listbox.bind("<Double-Button-1>", lambda e: self._open_speedrun_selected_candidate_report())

        cand_btn_row = Frame(candidates_section, bg=PANEL)
        cand_btn_row.pack(anchor="w", padx=18, pady=(0, 12))
        self._button(cand_btn_row, "OPEN REPORT", self._open_speedrun_selected_candidate_report).pack(side="left")
        self._button(cand_btn_row, "VIEW CODE", self._view_speedrun_selected_candidate_code).pack(side="left", padx=8)
        self._button(cand_btn_row, "SAVE TO LIBRARY", self._save_speedrun_selected_candidate).pack(side="left")

        self._speedrun_candidates_cache: list[dict] = []

        output_section = self._section(f, "Speed Run output", "Live progress log (Phase 1 discovery -> Phase 2 validation -> Phase 3 winner selection).")
        _speedrun_output_frame = Frame(output_section, bg=PANEL)
        self.speedrun_output = Text(
            _speedrun_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _speedrun_output_scroll = ttk.Scrollbar(
            _speedrun_output_frame, orient="vertical", command=self.speedrun_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.speedrun_output.configure(yscrollcommand=_speedrun_output_scroll.set)
        self.speedrun_output.pack(side="left", fill="both", expand=True)
        _speedrun_output_scroll.pack(side="right", fill="y")
        _speedrun_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.speedrun_output)

        self._last_speedrun_html_path = None
        self._speedrun_cancel_event = threading.Event()

    def _log_speedrun(self, msg: str):
        # Unlike every other tab's progress log, Speed Run's own Phase 2
        # can call this from SEVERAL concurrent validation worker threads
        # at once (run_speed_run serializes them with its own lock, but
        # they're still not the single background thread every other
        # tab's log callback assumes). Tkinter widgets are only safe to
        # touch from the main thread, so marshal onto it via root.after()
        # instead of writing directly -- the same pattern already used by
        # _test_ollama_connection's background-thread callback.
        def _write():
            self.speedrun_output.insert(END, msg + "\n")
            self.speedrun_output.see(END)
        try:
            self.root.after(0, _write)
        except Exception:
            pass

    def _open_speedrun_winner_report(self):
        if self._last_speedrun_html_path:
            webbrowser.open(f"file://{self._last_speedrun_html_path.resolve()}")

    def _open_speedrun_selected_candidate_report(self):
        sel = self.speedrun_candidates_listbox.curselection()
        if not sel or not self._speedrun_candidates_cache:
            return
        row = self._speedrun_candidates_cache[sel[0]]
        html_path = row.get("html_path")
        if html_path:
            webbrowser.open(f"file://{html_path.resolve()}")

    def _selected_speedrun_candidate(self) -> "SpeedRunCandidateResult | None":
        sel = self.speedrun_candidates_listbox.curselection()
        if not sel or not self._speedrun_candidates_cache:
            return None
        return self._speedrun_candidates_cache[sel[0]].get("row")

    def _view_speedrun_selected_candidate_code(self):
        """VIEW CODE for a Speed Run candidate. Full Pipeline's own
        final_code_text now covers python/pinescript/mql5 winners (their
        real patched source) AND manual-builder/Search-Lab winners (their
        config serialized as JSON -- see the full_pipeline.py fix that
        stopped manual configs from being treated as having 'nothing to
        show'), so this works for every candidate that made it to
        validation, not just code-strategy ones."""
        r = self._selected_speedrun_candidate()
        if r is None:
            messagebox.showinfo("Select a candidate", "Select a validated candidate from the list above first.")
            return
        pr = r.pipeline_result
        if pr is None or not pr.final_code_text:
            messagebox.showinfo(
                "No code available",
                "This candidate has no code or configuration to show (it failed validation, "
                "or produced no final configuration).",
            )
            return
        kind_label = {
            "python": "Python", "pinescript": "PineScript", "mql5": "MQL5",
            "manual": "Manual Strategy Builder config (JSON)",
        }.get(pr.final_source_type, pr.final_source_type)
        self._show_text_viewer(f"{r.candidate_id} -- {kind_label}", pr.final_code_text)

    def _save_speedrun_selected_candidate(self):
        """SAVE TO LIBRARY for a Speed Run candidate, on demand. Full
        Pipeline already auto-saves every validated candidate when 'Save
        every validated candidate to the Strategy Library' is checked
        above, so this is mainly for: that checkbox was off during the
        run, or you want to re-confirm/re-save a specific candidate
        without re-running Speed Run."""
        r = self._selected_speedrun_candidate()
        if r is None:
            messagebox.showinfo("Select a candidate", "Select a validated candidate from the list above first.")
            return
        pr = r.pipeline_result
        if pr is None:
            messagebox.showwarning("Nothing to save", "This candidate failed validation, so there's nothing to save.")
            return
        if pr.saved_library_path:
            messagebox.showinfo(
                "Already saved",
                f"This candidate was already saved to the Strategy Library as "
                f"'{Path(pr.saved_library_path).name}'.",
            )
            return
        source_type = pr.final_source_type
        if source_type in ("python", "pinescript", "mql5") and pr.final_code_text:
            ext = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}[source_type]
            text = pr.final_code_text
        elif source_type == "manual" and (pr.final_code_text or pr.final_config):
            ext = ".json"
            text = pr.final_code_text or json.dumps(pr.final_config, indent=2)
        else:
            messagebox.showwarning("Nothing to save", "This candidate has no code or configuration to save.")
            return
        base_name = f"speedrun_{r.candidate_id}".replace(" ", "_")
        filename = f"{base_name}{ext}"
        try:
            try:
                path = save_strategy_text(text, filename, source_type, overwrite=False)
            except StrategyAlreadyExists:
                if not messagebox.askyesno(
                    "Already in library", f"'{filename}' already exists in the Strategy Library. Overwrite it?",
                ):
                    return
                path = save_strategy_text(text, filename, source_type, overwrite=True)
            set_strategy_status(
                source_type, filename, "tested_passed" if pr.verdict in ("READY", "MARGINAL") else "tested_failed",
            )
            record_backtest_result(source_type, filename, {
                "trades": len(pr.final_bt.trades),
                "net_profit": round(pr.final_bt.statistics.net_profit, 2),
                "win_rate": round(pr.final_bt.statistics.win_rate, 1),
                "max_dd": round(pr.final_bt.statistics.max_drawdown_pct, 2),
                "eval_pass_probability": round(pr.final_mc.evaluation_pass_probability, 1),
                "first_payout_probability": round(pr.final_mc.first_payout_probability, 1),
                "verdict": pr.verdict,
                "report_html": str(pr.report_paths.get("html", "")),
            })
            messagebox.showinfo("Saved", f"Saved to the Strategy Library as '{filename}' (in {path.parent}).")
            self._refresh_strategy_library()
        except Exception as exc:  # noqa: BLE001 -- surface it, don't crash the tab
            messagebox.showerror("Save failed", str(exc))

    def _render_speedrun_candidates(self, result: SpeedRunResult):
        self.speedrun_candidates_listbox.delete(0, END)
        self._speedrun_candidates_cache = []
        ranked = sorted(result.candidates, key=_speedrun_rank_key)
        for r in ranked:
            if r.pipeline_result is not None:
                pr = r.pipeline_result
                html_path = pr.report_paths.get("html")
                line = (
                    f"[{pr.verdict:9s}]  {r.candidate_id}  ({r.family or 'unknown family'})  "
                    f"eval pass {pr.final_mc.evaluation_pass_probability:5.1f}%  "
                    f"payout {pr.final_mc.first_payout_probability:5.1f}%"
                )
            else:
                html_path = None
                line = f"[FAILED   ]  {r.candidate_id}  ({r.family or 'unknown family'})  -- {r.error}"
            self.speedrun_candidates_listbox.insert(END, line)
            self._speedrun_candidates_cache.append({"row": r, "html_path": html_path})

    def _speedrun_run_clicked(self):
        if not self.sr_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        if not self._try_start_heavy_job(JOB_SPEED_RUN):
            return
        self.speedrun_output.delete("1.0", END)
        self.speedrun_candidates_listbox.delete(0, END)
        self._speedrun_candidates_cache = []
        self.open_speedrun_report_btn.config(state="disabled")
        self.speedrun_verdict_label.config(text="Running...", fg=TEXT_DIM)
        self._speedrun_cancel_event.clear()
        self.stop_speedrun_btn.config(state="normal")
        self.speedrun_progress.start(10)
        if self.sr_loop_mode.get():
            threading.Thread(target=self._speedrun_run_loop_pipeline, daemon=True).start()
        else:
            threading.Thread(target=self._speedrun_run_pipeline, daemon=True).start()

    def _speedrun_stop_clicked(self):
        """Sets the shared cancel event, which run_speed_run() checks between
        discovery and validation -- an in-flight validation still finishes
        (Full Pipeline itself isn't interruptible mid-fold), but no further
        candidates are started and whatever already finished is still
        reported rather than thrown away."""
        self._speedrun_cancel_event.set()
        self.stop_speedrun_btn.config(state="disabled")
        self._log_speedrun("\nStopping... (finishing whatever's already in flight, no new candidates will start)")

    def _speedrun_run_pipeline(self):
        try:
            df = self.sr_context.load_dataframe(self._log_speedrun)
            if df is None:
                return
            risk = self.sr_context.build_risk_config()
            rules = self.sr_context.build_prop_rules()

            metric_key = self._sr_metric_label_to_key.get(self.sr_metric.get_str(), "eval_pass_probability")
            cfg = SpeedRunConfig(
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

            instrument = (
                os.path.basename(self.sr_context.csv_paths[0])
                if len(self.sr_context.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.sr_context.csv_paths)
            )
            self._log_speedrun(f"Starting Speed Run on {instrument} ({len(df)} bars)...\n")
            result = run_speed_run(
                df, risk, rules, OUTPUT_DIR / "speed_run", cfg,
                progress_cb=self._log_speedrun, instrument=instrument,
                cancel_event=self._speedrun_cancel_event,
            )

            self._render_speedrun_candidates(result)

            if result.winner is not None and result.winner.pipeline_result is not None:
                pr = result.winner.pipeline_result
                self._last_speedrun_html_path = pr.report_paths.get("html")
                self.open_speedrun_report_btn.config(state="normal" if self._last_speedrun_html_path else "disabled")
                verdict_color = {"READY": GREEN, "MARGINAL": AMBER}.get(pr.verdict, TEXT_DIM)
                self.speedrun_verdict_label.config(
                    text=(
                        f"WINNER: {result.winner.candidate_id} ({result.winner.family or 'unknown family'})\n"
                        f"  {pr.verdict}  --  eval pass {pr.final_mc.evaluation_pass_probability:.1f}%  --  "
                        f"payout {pr.final_mc.first_payout_probability:.1f}%"
                    ),
                    fg=verdict_color,
                )
                if pr.saved_library_note:
                    self._log_speedrun(f"\n{pr.saved_library_note}")
            else:
                self._last_speedrun_html_path = None
                self.open_speedrun_report_btn.config(state="disabled")
                hint = "  See the log below for what to try next." if result.guidance else ""
                self.speedrun_verdict_label.config(text=f"No winner. {result.winner_reason}{hint}", fg=RED)

            try:
                self._refresh_dashboard()
            except Exception:
                pass

            self._log_speedrun(f"\n{result.winner_reason}")
            self._log_speedrun(f"\nDone in {result.elapsed_seconds / 60:.1f} minute(s).")
        except Exception as exc:
            self._log_speedrun("\nUnexpected error:\n" + traceback.format_exc())
            self.speedrun_verdict_label.config(text="Failed -- see log.", fg=RED)
            log_crash("Speed Run", exc=exc)
        finally:
            self.speedrun_progress.stop()
            self.stop_speedrun_btn.config(state="disabled")
            self._release_heavy_job(JOB_SPEED_RUN)

    # -----------------------------------------------------------------------
    # Multi-Instrument Speed Run -- desktop parity with the web app's
    # /speed-run/multi-instrument page: the SAME SpeedRunConfig run
    # CONCURRENTLY across several instrument/timeframe datasets (a thin
    # wrapper over app.orchestration.multi_instrument_speed_run.
    # run_multi_instrument_speed_run -- no discovery/validation logic is
    # reimplemented here). Uses its own RunContextPanel's multi-select
    # dataset list, but unlike every other tab's RunContextPanel usage,
    # each selected CSV becomes its OWN separate job instead of being
    # stitched into one combined dataframe.
    # -----------------------------------------------------------------------

    def _misr_instrument_label_for_path(self, path: str) -> tuple[str, str]:
        """(instrument, timeframe) labels for one selected CSV, same
        convention as list_datasets_by_instrument: the top-level data/raw/
        subfolder is the instrument, the filename stem is the timeframe.
        Falls back to the bare filename stem for a file outside data/raw/
        (e.g. picked via Alpaca fetch or an arbitrary import)."""
        p = Path(path)
        try:
            rel = p.relative_to(get_raw_data_dir())
            parts = rel.parts
            if len(parts) > 1:
                return parts[0], rel.stem
        except ValueError:
            pass
        return p.stem, p.stem

    def _build_speedrun_multi_tab(self):
        f = self._scrollable(self.tab_speedrun_multi)

        self._page_header(
            f,
            "CREATE / Multi-Instrument Speed Run",
            "\u26a1 Multi-Instrument Speed Run",
            "Runs the SAME Speed Run (wide discovery -> validate the leaders -> pick a winner) "
            "CONCURRENTLY across several instrument/timeframe datasets -- a real edge is often "
            "instrument-dependent, so one overnight session across several instruments covers more "
            "ground than guessing which one to try next. Select 2 or more datasets below (Ctrl/Cmd-"
            "click or Shift-click); each gets its own Speed Run using the SAME settings, its own "
            "output directory, and its own winner (or honest 'no winner'). Every existing safeguard "
            "(lookahead detection, pip-scale mismatch, stop-fill honesty, the account-blown circuit "
            "breaker) still runs exactly as it does everywhere else in this app.",
        )

        self.misr_context = RunContextPanel(self, "Multi-Instrument Speed Run")
        self.misr_context.build(f)

        settings = self._section(
            f, "Speed Run settings (applied identically to every instrument)",
            "Same knobs as the regular Speed Run tab -- see that tab's own settings section for what "
            "each one does.",
            emphasize=True,
        )
        self.misr_max_candidates = LabeledEntry(settings, "Discovery: max candidates sampled (all families combined)", 1200)
        self.misr_stage1_top_n = LabeledEntry(settings, "Discovery: Stage 1 survivors kept", 24)
        self.misr_ga_population = LabeledEntry(settings, "Discovery: GA population", 8)
        self.misr_ga_generations = LabeledEntry(settings, "Discovery: GA generations", 3)
        self.misr_top_k = LabeledEntry(settings, "Candidates to validate through Full Pipeline", 3)
        self.misr_max_concurrent = LabeledEntry(settings, "Max concurrent validations (per instrument)", 2)
        self.misr_validation_folds = LabeledEntry(settings, "Validation: walk-forward/OOS folds", 3)
        self.misr_validation_final_mc_sims = LabeledEntry(settings, "Validation: final Monte Carlo sims", 3000)
        self._misr_metric_labels = list(FITNESS_METRICS.values())
        self._misr_metric_label_to_key = {v: k for k, v in FITNESS_METRICS.items()}
        self.misr_metric = LabeledCombo(
            settings, "Fitness metric", self._misr_metric_labels, FITNESS_METRICS["eval_pass_probability"],
        )
        self.misr_seed = LabeledEntry(settings, "Random seed", 42)
        self.misr_save_to_library = LabeledCheckbox(
            settings, "Save every validated candidate to the Strategy Library when finished", True,
        )
        self.misr_max_concurrent_instruments = LabeledEntry(
            settings, "Max instruments run at once (splits worker/concurrency budget across them)", 2,
        )

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN ACROSS ALL SELECTED INSTRUMENTS", self._speedrun_multi_run_clicked, primary=True).pack(side="left")

        self.misr_progress = NeuralProgress(f)
        self.misr_progress.pack(fill="x", padx=24, pady=(2, 10))

        results_section = self._section(
            f, "Per-instrument results",
            "One line per instrument/timeframe -- double-click (or select + OPEN REPORT) to open "
            "that instrument's own winner report.",
        )
        misr_list_frame = Frame(results_section, bg=PANEL)
        misr_list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 6))
        misr_list_frame.columnconfigure(0, weight=1)
        misr_list_frame.rowconfigure(0, weight=1)
        self.misr_results_listbox = Listbox(
            misr_list_frame, height=8, exportselection=False, bg=PANEL_3, fg=TEXT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        misr_scrollbar = ttk.Scrollbar(
            misr_list_frame, orient="vertical", command=self.misr_results_listbox.yview, style="T58.Vertical.TScrollbar",
        )
        self.misr_results_listbox.config(yscrollcommand=misr_scrollbar.set)
        self._bind_isolated_wheel(self.misr_results_listbox)
        self.misr_results_listbox.grid(row=0, column=0, sticky="nsew")
        misr_scrollbar.grid(row=0, column=1, sticky="ns")
        self.misr_results_listbox.bind("<Double-Button-1>", lambda e: self._open_misr_selected_report())

        misr_btn_row = Frame(results_section, bg=PANEL)
        misr_btn_row.pack(anchor="w", padx=18, pady=(0, 12))
        self._button(misr_btn_row, "OPEN REPORT", self._open_misr_selected_report).pack(side="left")

        self._misr_results_cache: list[dict] = []

        best_section = self._section(f, "Best across all instruments", "Filled in once the run completes.")
        self.misr_best_label = Label(
            best_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=900, anchor="w",
        )
        self.misr_best_label.pack(anchor="w", fill="x", padx=18, pady=(2, 12))

        output_section = self._section(f, "Multi-Instrument Speed Run output", "Live progress log, one line per instrument.")
        _misr_output_frame = Frame(output_section, bg=PANEL)
        self.misr_output = Text(
            _misr_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _misr_output_scroll = ttk.Scrollbar(
            _misr_output_frame, orient="vertical", command=self.misr_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.misr_output.configure(yscrollcommand=_misr_output_scroll.set)
        self.misr_output.pack(side="left", fill="both", expand=True)
        _misr_output_scroll.pack(side="right", fill="y")
        _misr_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.misr_output)

    def _log_misr(self, label: str, msg: str):
        # Called from several concurrent instrument-job threads at once
        # (same reasoning as _log_speedrun) -- marshal onto the main
        # thread rather than touching the Text widget directly.
        def _write():
            self.misr_output.insert(END, f"[{label}] {msg}\n")
            self.misr_output.see(END)
        try:
            self.root.after(0, _write)
        except Exception:
            pass

    def _open_misr_selected_report(self):
        sel = self.misr_results_listbox.curselection()
        if not sel or not self._misr_results_cache:
            return
        row = self._misr_results_cache[sel[0]]
        html_path = row.get("html_path")
        if html_path:
            webbrowser.open(f"file://{Path(html_path).resolve()}")
        else:
            messagebox.showinfo("No report", "This instrument didn't produce a winner report.")

    def _render_misr_results(self, results: dict[str, "MultiInstrumentSpeedRunResult"]):
        self.misr_results_listbox.delete(0, END)
        self._misr_results_cache = []
        for label, res in sorted(results.items()):
            if res.error:
                line = f"[FAILED   ]  {label}  -- {res.error.splitlines()[0]}"
                self.misr_results_listbox.insert(END, line)
                self._misr_results_cache.append({"html_path": None})
                continue
            r = res.result
            if r.winner is not None and r.winner.pipeline_result is not None:
                pr = r.winner.pipeline_result
                html_path = pr.report_paths.get("html")
                line = (
                    f"[{pr.verdict:9s}]  {label}  eval pass {pr.final_mc.evaluation_pass_probability:5.1f}%  "
                    f"payout {pr.final_mc.first_payout_probability:5.1f}%"
                )
            else:
                html_path = None
                line = f"[NO WINNER]  {label}  -- {r.winner_reason}"
            self.misr_results_listbox.insert(END, line)
            self._misr_results_cache.append({"html_path": html_path})

    def _speedrun_multi_run_clicked(self):
        if len(self.misr_context.csv_paths) < 2:
            messagebox.showwarning(
                "Select at least 2 datasets",
                "Select 2 or more datasets above (Ctrl/Cmd-click or Shift-click) -- with only 1 "
                "selected, use the regular Speed Run tab instead.",
            )
            return
        if not self._try_start_heavy_job(JOB_SPEED_RUN):
            return
        self.misr_output.delete("1.0", END)
        self.misr_results_listbox.delete(0, END)
        self._misr_results_cache = []
        self.misr_best_label.config(text="Running...", fg=TEXT_DIM)
        self.misr_progress.start(10)
        threading.Thread(target=self._speedrun_multi_run_pipeline, daemon=True).start()

    def _speedrun_multi_run_pipeline(self):
        try:
            jobs = []
            for path in self.misr_context.csv_paths:
                instrument, timeframe = self._misr_instrument_label_for_path(path)
                jobs.append(InstrumentJob(instrument=instrument, timeframe=timeframe, csv_path=path))

            risk = self.misr_context.build_risk_config()
            rules = self.misr_context.build_prop_rules()
            metric_key = self._misr_metric_label_to_key.get(self.misr_metric.get_str(), "eval_pass_probability")
            cfg = SpeedRunConfig(
                max_candidates=self.misr_max_candidates.get_int(1200),
                stage1_top_n=self.misr_stage1_top_n.get_int(24),
                ga_population=self.misr_ga_population.get_int(8),
                ga_generations=self.misr_ga_generations.get_int(3),
                top_k_to_validate=self.misr_top_k.get_int(3),
                max_concurrent_validations=self.misr_max_concurrent.get_int(2),
                validation_folds=self.misr_validation_folds.get_int(3),
                validation_final_mc_sims=self.misr_validation_final_mc_sims.get_int(3000),
                fitness_metric=metric_key,
                save_winner_to_library=self.misr_save_to_library.get(),
                random_seed=self.misr_seed.get_int(42),
                reset_on_breach=True,
            )
            max_concurrent_instruments = self.misr_max_concurrent_instruments.get_int(2)

            self._log_misr(
                "multi-instrument",
                f"Starting Speed Run on {len(jobs)} instrument/timeframe target(s): "
                + ", ".join(f"{j.instrument}/{j.timeframe}" for j in jobs),
            )
            results = run_multi_instrument_speed_run(
                jobs, risk, rules, cfg, OUTPUT_DIR / "speed_run" / "multi_instrument",
                max_concurrent_instruments=max_concurrent_instruments,
                progress_cb=self._log_misr,
            )

            self._render_misr_results(results)

            best = best_speed_run_across_instruments(results)
            if best is not None and best.result.winner is not None and best.result.winner.pipeline_result is not None:
                pr = best.result.winner.pipeline_result
                verdict_color = {"READY": GREEN, "MARGINAL": AMBER}.get(pr.verdict, TEXT_DIM)
                self.misr_best_label.config(
                    text=(
                        f"BEST: {best.label} -- {best.result.winner.candidate_id} "
                        f"({best.result.winner.family or 'unknown family'})\n"
                        f"  {pr.verdict}  --  eval pass {pr.final_mc.evaluation_pass_probability:.1f}%  --  "
                        f"payout {pr.final_mc.first_payout_probability:.1f}%"
                    ),
                    fg=verdict_color,
                )
            else:
                self.misr_best_label.config(text="No instrument produced a winner. See the log below.", fg=RED)

            try:
                self._refresh_dashboard()
            except Exception:
                pass

            n_winners = sum(1 for r in results.values() if r.has_winner)
            self._log_misr("multi-instrument", f"\nDone -- {n_winners}/{len(jobs)} instrument(s) produced a winner.")
        except Exception as exc:
            self._log_misr("multi-instrument", "\nUnexpected error:\n" + traceback.format_exc())
            self.misr_best_label.config(text="Failed -- see log.", fg=RED)
            log_crash("Multi-Instrument Speed Run", exc=exc)
        finally:
            self.misr_progress.stop()
            self._release_heavy_job(JOB_SPEED_RUN)

    def _speedrun_run_loop_pipeline(self):
        """Loop Mode's desktop counterpart to _speedrun_run_pipeline above
        -- same data-loading/config/risk/rules setup, but drives
        app.orchestration.loop_runner.run_speed_run_loop (repeated rounds
        until a winner is found) instead of a single run_speed_run call.
        Repaints the candidates list/verdict after EVERY round (via
        on_round), so the tab shows live progress across a loop that can
        run for a long time."""
        try:
            df = self.sr_context.load_dataframe(self._log_speedrun)
            if df is None:
                return
            risk = self.sr_context.build_risk_config()
            rules = self.sr_context.build_prop_rules()

            metric_key = self._sr_metric_label_to_key.get(self.sr_metric.get_str(), "eval_pass_probability")
            base_config = SpeedRunConfig(
                stage1_top_n=self.sr_stage1_top_n.get_int(24),
                ga_population=self.sr_ga_population.get_int(8),
                ga_generations=self.sr_ga_generations.get_int(3),
                max_concurrent_validations=self.sr_max_concurrent.get_int(2),
                validation_folds=self.sr_validation_folds.get_int(3),
                validation_final_mc_sims=self.sr_validation_final_mc_sims.get_int(3000),
                fitness_metric=metric_key,
                save_winner_to_library=self.sr_save_to_library.get(),
                reset_on_breach=True,
            )
            instrument = (
                os.path.basename(self.sr_context.csv_paths[0])
                if len(self.sr_context.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.sr_context.csv_paths)
            )
            time_budget_raw = self.sr_loop_time_budget_hours.get_str().strip()
            loop_cfg = SpeedRunLoopConfig(
                max_rounds=self.sr_loop_max_rounds.get_int(10),
                time_budget_seconds=(float(time_budget_raw) * 3600.0) if time_budget_raw else None,
                stall_rounds_before_widen=self.sr_loop_stall_rounds.get_int(2),
                starting_max_candidates=self.sr_max_candidates.get_int(1200),
                starting_top_k_to_validate=self.sr_top_k.get_int(3),
                seed=self.sr_seed.get_int(42),
                base_config=base_config,
            )
            self._log_speedrun(
                f"Loop mode ON -- will repeat Speed Run rounds (raising the candidate cap and "
                f"validation width on a stall) until a round finds a winner, {loop_cfg.max_rounds} "
                f"rounds run, or the time budget is used up.\n"
            )

            def on_round(round_result) -> None:
                result = round_result.result

                def _paint():
                    self._render_speedrun_candidates(result)
                    if result.winner is not None and result.winner.pipeline_result is not None:
                        pr = result.winner.pipeline_result
                        self._last_speedrun_html_path = pr.report_paths.get("html")
                        self.open_speedrun_report_btn.config(state="normal" if self._last_speedrun_html_path else "disabled")
                        verdict_color = {"READY": GREEN, "MARGINAL": AMBER}.get(pr.verdict, TEXT_DIM)
                        self.speedrun_verdict_label.config(
                            text=(
                                f"WINNER: {result.winner.candidate_id} ({result.winner.family or 'unknown family'})\n"
                                f"  {pr.verdict}  --  eval pass {pr.final_mc.evaluation_pass_probability:.1f}%  --  "
                                f"payout {pr.final_mc.first_payout_probability:.1f}%"
                            ),
                            fg=verdict_color,
                        )
                    else:
                        self.speedrun_verdict_label.config(
                            text=f"Loop round {round_result.round_index} -- no winner yet.", fg=AMBER,
                        )
                self.root.after(0, _paint)

            result = run_speed_run_loop(
                df, risk, rules, output_dir=OUTPUT_DIR / "speed_run" / "loop", loop_cfg=loop_cfg,
                instrument=instrument, progress_cb=self._log_speedrun,
                cancel_event=self._speedrun_cancel_event,
            )

            if result.stopped_reason == "error":
                self._log_speedrun(f"\nLoop Mode stopped: a round crashed -- {result.error}")
                self.speedrun_verdict_label.config(text="Failed -- see log.", fg=RED)
                log_crash("Speed Run Loop Mode (desktop)", exc=RuntimeError(result.error or "unknown"))
                return

            reason_text = {
                "target_reached": "A round found a winner -- stopped automatically.",
                "max_rounds": f"Stopped: all {loop_cfg.max_rounds} rounds completed with no winner.",
                "time_budget": "Stopped: time budget used up.",
                "cancelled": "Stopped by user.",
            }.get(result.stopped_reason, result.stopped_reason)
            self._log_speedrun(f"\n{reason_text}")
            self._log_speedrun(f"\nDone in {result.total_elapsed_seconds / 60:.1f} minute(s) across {len(result.rounds)} round(s).")

            try:
                self._refresh_dashboard()
            except Exception:
                pass
        except Exception as exc:
            self._log_speedrun("\nUnexpected error:\n" + traceback.format_exc())
            self.speedrun_verdict_label.config(text="Failed -- see log.", fg=RED)
            log_crash("Speed Run Loop Mode (desktop)", exc=exc)
        finally:
            self.speedrun_progress.stop()
            self.stop_speedrun_btn.config(state="disabled")
            self._release_heavy_job(JOB_SPEED_RUN)
