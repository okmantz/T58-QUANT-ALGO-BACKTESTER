"""Evolution Lab (single and multi-instrument) and family-health views."""
from __future__ import annotations

import app.evolution.checkpoint as evo_checkpoint
import json
import os
import threading
import uuid
from app.data.importer import import_csv
from app.data.multi_timeframe import merge_multi_timeframe
from app.optimize.refinement import OPTIMIZER_MODES
from app.orchestration import pipeline_guide
from app.orchestration.resource_guard import JOB_EVOLUTION_LAB, JOB_MULTI_INSTRUMENT_EVOLUTION
from app.search.strategy_space import list_families
from app.strategy.library import StrategyAlreadyExists, save_strategy_text, set_strategy_status
from tkinter import BooleanVar, Canvas, Checkbutton, END, Frame, Label, Listbox, Text, Toplevel, messagebox, ttk
from app.ui.common import _safe_font
from app.ui.platform_win import _apply_dark_titlebar
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui.widgets.run_context import RunContextPanel
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
PANEL_2 = ""
PANEL_3 = ""
BORDER = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
LOG_BG = ""
GREEN = ""
AMBER = ""
theme.bind(globals())


class EvolutionLabMixin:
    """Evolution Lab (single and multi-instrument) and family-health views."""


    def _evo_families_select_all(self):
        for var in getattr(self, "evo_family_vars", {}).values():
            var.set(True)

    def _evo_families_clear_all(self):
        for var in getattr(self, "evo_family_vars", {}).values():
            var.set(False)

    def _view_family_health(self):
        """Shows every family's cross-run tested/passed tally (Search Lab
        + Evolution Lab combined, across every instrument this app has
        ever run) and flags which ones are currently excluded as dead
        ends -- the SAME accumulated history that silently narrows which
        families a NEW run is even allowed to pick from (see
        EvolutionRunner._apply_family_health_exclusions). This is exactly
        why a run can keep landing on the same 2-3 families: once enough
        OTHER families have racked up 30+ zero-success attempts (which
        adds up fast across a long history on one instrument, or one
        whose default prefilter thresholds don't suit a family's grid),
        they're quietly dropped from the pool before population
        generation even starts, not by GA selection pressure within any
        one run. RESET FAMILY HEALTH (next to this button) clears that
        accumulated history so every family gets a clean slate."""
        try:
            from app.search.family_health import compute_family_health
            health = compute_family_health()
        except Exception as exc:
            messagebox.showerror("Family Health", f"Couldn't compute family health: {exc}")
            return
        win = Toplevel(self.root)
        win.title("Family Health -- cross-run history")
        win.configure(bg=PANEL)
        win.geometry("640x480")
        Label(
            win, text="Tested/passed across EVERY Search Lab + Evolution Lab run this app has "
            "on disk, for every instrument -- not just the current run. A family only shows "
            "DEAD END once it has 30+ tested with zero passes; a family with no rows at all "
            "here has simply never been tried yet and isn't shown.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(9), wraplength=600, justify="left",
        ).pack(anchor="w", padx=12, pady=(12, 8))
        text = Text(win, bg=PANEL_3, fg=TEXT, font=(MONO, 10), relief="flat", bd=0, wrap="none")
        text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        if not health:
            text.insert("1.0", "No history yet -- every family is still \"no evidence yet.\"")
        else:
            rows = sorted(health.values(), key=lambda h: (-h.n_tested))
            text.insert("1.0", f"{'FAMILY':38s}{'TESTED':>8s}{'PASSED':>8s}   STATUS\n")
            text.insert(END, "-" * 70 + "\n")
            for h in rows:
                status = "DEAD END" if h.is_dead_end else "healthy"
                text.insert(END, f"{h.family:38s}{h.n_tested:8d}{h.n_passed:8d}   {status}\n")
        text.config(state="disabled")

    def _reset_family_health(self):
        proceed = messagebox.askokcancel(
            "Reset family health?",
            "This archives (renames, does not delete) every Search Lab result database and "
            "Evolution Lab tested-candidates log this app has ever written, so every family's "
            "dead-end tally starts fresh. Use this if you suspect a run keeps landing on the "
            "same few families because most others were auto-excluded from a long accumulated "
            "history (see VIEW FAMILY HEALTH), or after a bug fix that could have produced a "
            "burst of spurious failures. This is recoverable -- old files are renamed with a "
            "'.pre-reset' suffix, not deleted.",
        )
        if not proceed:
            return
        try:
            from app.search.family_health import reset_family_health
            result = reset_family_health()
        except Exception as exc:
            messagebox.showerror("Reset family health", f"Couldn't reset: {exc}")
            return
        messagebox.showinfo(
            "Family health reset",
            f"Archived {result['search_dbs_reset']} Search Lab result database(s) and "
            f"{result['evolution_logs_reset']} Evolution Lab tested-candidate log(s). "
            f"Every family starts fresh again on your next run.",
        )

    # -----------------------------------------------------------------------
    # Tab 15 — Full Pipeline (run everything, hand back the champion)
    # -----------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Evolution Lab -- natural-selection-based strategy discovery.
    # See app.evolution.engine for the actual generate -> pre-filter ->
    # robustness/OOS/Monte Carlo/prop-sim -> CPCV/PBO -> stress -> cluster
    # -> keep top N -> mutate -> repeat loop this tab drives.
    # ------------------------------------------------------------------
    def _build_evolution_lab_tab(self):
        f = self._scrollable(self.tab_evolution)
        Label(
            f, text="EVOLUTION LAB", bg=BG, fg=TEXT, font=_safe_font(16, "bold"),
        ).pack(anchor="w", padx=24, pady=(20, 4))
        Label(
            f,
            text=(
                "Natural-selection-based strategy discovery: each generation generates a fresh "
                "population of strategies across every family (07 Search Lab's own generator), "
                "runs them through pre-filter -> robustness -> walk-forward -> Monte Carlo -> prop "
                "simulation -> real CPCV / PBO -> a stress test at higher execution costs -> a "
                "correlation-based cluster dedupe, keeps the top N by PROP FITNESS (not raw profit -- "
                "see app/evolution/prop_fitness.py), mutates them, and repeats. Click START and it runs "
                "in the background -- safe to leave running for hours while you work in other tabs. "
                "Every candidate evaluated (not just the winners) is logged to the knowledge graph, so "
                "later generations' journal entries can say what's historically worked before, not just "
                "what this generation found."
            ),
            bg=BG, fg=TEXT_DIM, font=_safe_font(9), wraplength=900, justify="left",
        ).pack(anchor="w", padx=24, pady=(0, 10))

        # Evolution Lab now has its OWN self-contained market data + prop
        # account + risk settings via RunContextPanel below -- independent
        # of Steps 02-04 and every other tab, same as Full Pipeline / Search
        # Lab / the other converted tabs. (This used to intentionally SHARE
        # StringVars with 03 Prop Rules / 04 Risk so an overnight run and
        # Full Pipeline's later re-check couldn't drift apart -- but now
        # that Full Pipeline is itself self-contained rather than reading
        # 03/04, that coupling reason no longer applies, and genuine
        # independence is what was actually asked for.)
        self.evo_context = RunContextPanel(self, "Evolution Lab (GA)")
        self.evo_context.build(f)
        self._button(f, "CHECK DATA INTEGRITY", lambda c=self.evo_context: self._check_data_integrity(c)).pack(anchor="w", padx=18, pady=(0, 8))

        cfg_section = self._section(
            f, "Run configuration",
            "Uses the market data and prop account / risk settings above (this tab's own, "
            "self-contained) at the moment you click START -- changing any of those after "
            "starting has no effect on an already-running Evolution Lab run.",
            emphasize=True,
        )
        self.evo_population = LabeledEntry(cfg_section, "Population size per generation", "60")
        self.evo_elite_keep = LabeledEntry(cfg_section, "Elite keep (top N)", "10")
        self.evo_max_generations = LabeledEntry(cfg_section, "Max generations (blank = run until stopped)", "")
        self.evo_loop_target = LabeledEntry(
            cfg_section,
            "Loop mode: stop automatically once a candidate clears this eval-pass % (blank = disabled)",
            "",
        )
        self.evo_loop_target_metric = LabeledCombo(
            cfg_section, "Loop mode target metric",
            ["CPCV out-of-sample eval-pass % (honest, recommended)", "Raw in-sample eval-pass % (Monte Carlo only)"],
            "CPCV out-of-sample eval-pass % (honest, recommended)",
        )
        self.evo_min_trades = LabeledEntry(cfg_section, "Min trades (pre-filter)", "20")
        self.evo_mc_sims = LabeledEntry(cfg_section, "Monte Carlo sims per candidate", "1000")
        self._optimizer_mode_labels = list(OPTIMIZER_MODES.values())
        self._optimizer_mode_label_to_key = {v: k for k, v in OPTIMIZER_MODES.items()}
        self.evo_optimizer_mode = LabeledCombo(
            cfg_section, "Optimizer mode (per-family child proposals)", self._optimizer_mode_labels,
            OPTIMIZER_MODES["genetic"],
        )
        self.evo_cpcv_top_n = LabeledEntry(cfg_section, "CPCV / PBO pool size (most expensive stage)", "8")
        self.evo_stress_mult = LabeledEntry(cfg_section, "Stress test cost multiplier", "2.0")
        self.evo_prefilter_max_bars = LabeledEntry(
            cfg_section,
            "Pre-filter bars cap (blank = auto; 0 = never cap; large 1-min "
            "feeds auto-cap the CHEAP pre-filter pass only -- every survivor "
            "still gets evaluated on the full dataset)",
            "",
        )
        self.evo_adaptive_risk_enabled = LabeledCheckbox(
            cfg_section,
            "Enable adaptive, limit-aware position sizing for every candidate this run "
            "evaluates (same engine as 05 Run & Report's Adaptive Risk section)",
            False,
        )
        self.evo_reset_on_breach = LabeledCheckbox(
            cfg_section,
            "Reset-on-breach: score every candidate on the basis that a blown account gets "
            "a fresh eval and keeps going, not a dead end (recommended -- uncheck to "
            "optimize on a single account with no reset)",
            True,
        )
        self.evo_sealed_final_frac = LabeledEntry(
            cfg_section,
            "Sealed final test slice (fraction of data, never touched until the end; 0 = off)",
            0.10,
        )

        families_frame = Frame(cfg_section, bg=PANEL)
        families_frame.pack(fill="x", padx=18, pady=(4, 4))
        Label(
            families_frame, text="Families to include (none checked = every family)",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
        ).pack(anchor="w")
        # UPGRADE (family deselect fix): this used to be a Listbox in
        # EXTENDED select mode -- a plain click on ANY row, with no Ctrl
        # held, replaces the entire selection with just that one row. That
        # is exactly the reported "I clicked one by accident and lost my
        # whole selection" trap: there was no way to remove a single row
        # without also knowing the (undiscoverable, unlabeled) Ctrl+click
        # toggle gesture. A checkbox per family has no such trap -- clicking
        # one only ever flips that one family, on or off, leaving every
        # other checkbox exactly as it was. SELECT ALL / CLEAR ALL below
        # cover the "toggle everything at once" case the old Listbox's
        # Ctrl+A / click-drag range-select used to.
        families_toolbar = Frame(families_frame, bg=PANEL)
        families_toolbar.pack(fill="x", pady=(2, 0))
        self._button(families_toolbar, "SELECT ALL", self._evo_families_select_all).pack(side="left")
        self._button(families_toolbar, "CLEAR ALL", self._evo_families_clear_all).pack(side="left", padx=(6, 0))
        self._button(families_toolbar, "VIEW FAMILY HEALTH", self._view_family_health).pack(side="left", padx=(6, 0))
        self._button(families_toolbar, "RESET FAMILY HEALTH", self._reset_family_health).pack(side="left", padx=(6, 0))

        # height=6 -> 10-ish rows visible with a real scrollbar: the family
        # list grew from 11 to 34 (see app.search.strategy_space's
        # "expansion round"s), so a short fixed box with no scrollbar at
        # all used to hide most of the list with no way to reach it.
        families_list_frame = Frame(families_frame, bg=PANEL_3, highlightthickness=1, highlightbackground=BORDER)
        families_list_frame.pack(fill="both", expand=True, pady=(4, 0))
        families_canvas = Canvas(families_list_frame, bg=PANEL_3, highlightthickness=0, height=200)
        families_scroll = ttk.Scrollbar(
            families_list_frame, orient="vertical", command=families_canvas.yview,
            style="T58.Vertical.TScrollbar",
        )
        families_canvas.configure(yscrollcommand=families_scroll.set)
        families_canvas.pack(side="left", fill="both", expand=True)
        families_scroll.pack(side="right", fill="y")

        families_inner = Frame(families_canvas, bg=PANEL_3)
        families_window_id = families_canvas.create_window((0, 0), window=families_inner, anchor="nw")
        families_inner.bind(
            "<Configure>", lambda _e: families_canvas.configure(scrollregion=families_canvas.bbox("all")),
        )
        families_canvas.bind(
            "<Configure>", lambda e: families_canvas.itemconfig(families_window_id, width=e.width),
        )
        self._bind_isolated_wheel(families_canvas)
        self._bind_isolated_wheel(families_inner)

        self.evo_family_vars: dict[str, BooleanVar] = {}
        try:
            for fam in sorted(list_families().keys()):
                var = BooleanVar(value=False)
                self.evo_family_vars[fam] = var
                Checkbutton(
                    families_inner, text=fam, variable=var, bg=PANEL_3, fg=TEXT,
                    selectcolor=PANEL_2, activebackground=PANEL_3, activeforeground=TEXT,
                    font=(MONO, 9), anchor="w", relief="flat", highlightthickness=0, padx=6, bd=0,
                    cursor="hand2",
                ).pack(fill="x", anchor="w")
        except Exception:
            pass
        # _bind_isolated_wheel above only catches the wheel over the
        # canvas's own empty margin -- with every row now packed full of
        # Checkbuttons, the cursor is almost always over one of THOSE
        # instead, which had no such binding and let the scroll fall
        # through to the whole page underneath. Binding every checkbutton
        # too (now that they all exist) is what actually makes this list
        # scroll independently of the page.
        self._bind_isolated_wheel_tree(families_inner, families_canvas)

        btn_row = Frame(cfg_section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(10, 6))
        self._button(btn_row, "START EVOLUTION LAB", self._start_evolution_lab, primary=True).pack(side="left")
        self._button(btn_row, "STOP", self._stop_evolution_lab).pack(side="left", padx=8)
        self._button(btn_row, "RESET (discard saved progress)", self._reset_evolution_lab).pack(side="left", padx=8)

        self.evo_status_label = Label(
            cfg_section, text="Not running.", bg=PANEL, fg=TEXT_DIM, font=_safe_font(9),
        )
        self.evo_status_label.pack(anchor="w", padx=18, pady=(0, 2))
        self.evo_loop_status_label = Label(
            cfg_section, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(9), wraplength=900, justify="left",
        )
        self.evo_loop_status_label.pack(anchor="w", padx=18, pady=(0, 2))
        Label(
            cfg_section,
            text=(
                "Progress (generation, leaderboard, journal) is saved to disk after every generation. "
                "STOP then START AGAIN resumes from exactly where it left off, as long as the same market "
                "data is still loaded -- click RESET first if you want a genuinely fresh run instead."
            ),
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=900, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        lb_section = self._section(
            f, "Leaderboard (all-time best seen)",
            "Refreshed after every generation. Sorted by PROP FITNESS, which already accounts for "
            "pass probability, payout probability, robustness, OOS consistency, drawdown, and the "
            "penalties described above -- not just net profit. The OOS% column (when present) is "
            "the honest, held-out CPCV estimate -- a big gap below the raw pass probability shown in "
            "VIEW DETAILS means the GA likely fit this dataset's noise rather than a real edge, which "
            "is exactly the pattern behind a strategy looking great here and coming back much weaker "
            "out of 15 Full Pipeline's stricter validation; PROMOTE will warn you about it directly "
            "when the gap is large. Double-click (or select + VIEW DETAILS) to see the full stat "
            "breakdown and generated code/config for any leader, and PROMOTE it straight into the "
            "Strategy Library to run it through 15 Full Pipeline.",
        )
        lb_frame = Frame(lb_section, bg=PANEL)
        # fill="both", expand=True (not the old fill="x") so this frame --
        # and the listbox inside it -- actually grows to use the space the
        # window gives it instead of clipping at a fixed pixel height.
        lb_frame.pack(fill="both", expand=True, padx=18, pady=(2, 6))
        lb_frame.columnconfigure(0, weight=1)
        lb_frame.rowconfigure(0, weight=1)
        self.evo_leaderboard_listbox = Listbox(
            # height=16 -> 30: with population_size defaulting to 60 and
            # elite_keep=10 (but the all-time leaderboard can hold more
            # than one generation's elites), 16 visible rows was cramped
            # enough that Owen couldn't work with it without scrolling
            # constantly. This is still just the *starting* height --
            # rowconfigure(weight=1) above plus fill="both" means it also
            # grows further if the window itself is resized taller.
            lb_frame, height=30, exportselection=False, bg=PANEL_3, fg=TEXT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        evo_lb_scrollbar = ttk.Scrollbar(
            lb_frame, orient="vertical", command=self.evo_leaderboard_listbox.yview, style="T58.Vertical.TScrollbar",
        )
        self.evo_leaderboard_listbox.config(yscrollcommand=evo_lb_scrollbar.set)
        self._bind_isolated_wheel(self.evo_leaderboard_listbox)
        self.evo_leaderboard_listbox.grid(row=0, column=0, sticky="nsew")
        evo_lb_scrollbar.grid(row=0, column=1, sticky="ns")
        self.evo_leaderboard_listbox.bind("<Double-Button-1>", lambda e: self._view_evolution_leader_detail())

        lb_btn_row = Frame(lb_section, bg=PANEL)
        lb_btn_row.pack(anchor="w", padx=18, pady=(0, 12))
        self._button(lb_btn_row, "VIEW DETAILS", self._view_evolution_leader_detail, primary=True).pack(side="left")
        self._button(lb_btn_row, "PROMOTE TO STRATEGY LIBRARY", self._promote_selected_evolution_leader).pack(side="left", padx=8)
        self._button(lb_btn_row, "FINAL TEST (SEALED)", self._final_test_selected_evolution_leader).pack(side="left", padx=8)
        self._button(lb_btn_row, "REFRESH FROM DISK", self._load_evolution_leaderboard_from_disk).pack(side="left", padx=8)

        self._evo_leaderboard_cache: list[dict] = []  # index-aligned with evo_leaderboard_listbox rows

        tested_section = self._section(
            f, "Tested strategies (most recent first)",
            "Every candidate the PRE-FILTER stage has actually backtested this run -- pass or fail, "
            "with the reason it was rejected if it was, and how far it got if it passed. Persists "
            "across STOP/START and survives closing the app; click REFRESH to pull the latest.",
        )
        tested_btn_row = Frame(tested_section, bg=PANEL)
        tested_btn_row.pack(anchor="w", padx=18, pady=(2, 4))
        self._button(tested_btn_row, "REFRESH", self._refresh_evolution_tested).pack(side="left")
        tested_list_frame = Frame(tested_section, bg=PANEL)
        tested_list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 12))
        tested_list_frame.columnconfigure(0, weight=1)
        tested_list_frame.rowconfigure(0, weight=1)
        self.evo_tested_listbox = Listbox(
            tested_list_frame, height=10, exportselection=False, bg=PANEL_3, fg=TEXT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        evo_tested_scroll = ttk.Scrollbar(
            tested_list_frame, orient="vertical", command=self.evo_tested_listbox.yview,
            style="T58.Vertical.TScrollbar",
        )
        self.evo_tested_listbox.config(yscrollcommand=evo_tested_scroll.set)
        self.evo_tested_listbox.grid(row=0, column=0, sticky="nsew")
        evo_tested_scroll.grid(row=0, column=1, sticky="ns")
        self._bind_isolated_wheel(self.evo_tested_listbox)

        log_section = self._section(
            f, "Live log + hypothesis journal",
            "Each generation ends with a numbered HYPOTHESIS entry (test/result/OOS/CPCV/stress/"
            "winner/confidence) plus a knowledge-graph similarity readout for that generation's winner.",
        )
        log_frame = Frame(log_section, bg=PANEL)
        log_frame.pack(fill="both", expand=True, padx=18, pady=(2, 14))
        self.evo_log_text = Text(
            log_frame, wrap="word", bg=PANEL_3, fg=TEXT, font=(MONO, 9), relief="flat", bd=0, height=18,
        )
        evo_log_scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.evo_log_text.yview, style="T58.Vertical.TScrollbar",
        )
        self.evo_log_text.config(yscrollcommand=evo_log_scrollbar.set)
        self._bind_isolated_wheel(self.evo_log_text)
        self.evo_log_text.pack(side="left", fill="both", expand=True)
        evo_log_scrollbar.pack(side="right", fill="y")

        self._evolution_runner = None
        try:
            self._load_evolution_leaderboard_from_disk()
            self._refresh_evolution_tested()
        except Exception:
            pass

    # An Evolution Lab run is meant to be left going overnight across many
    # generations, and every log line from every stage (PRE-FILTER,
    # ROBUSTNESS, MONTE CARLO, ...) for every candidate went into this one
    # Tk Text widget with nothing ever removed. A Text widget's own insert
    # cost and memory footprint both grow with how much it's already
    # holding, so a genuinely long run (the kind Owen described leaving
    # running "for a while") could accumulate tens of thousands of lines
    # and get progressively slower/heavier until something -- this
    # widget's own redraw, or just system memory pressure -- gave out.
    # 4000 lines is generous scrollback for actually reading recent
    # activity while keeping the widget's size bounded for the life of a
    # long run.
    _EVO_LOG_MAX_LINES = 4000

    def _evo_log(self, msg: str) -> None:
        def _do():
            try:
                self.evo_log_text.insert(END, msg + "\n")
                # int(...) on the Text index's line component; cheap
                # relative to the insert itself.
                n_lines = int(self.evo_log_text.index("end-1c").split(".")[0])
                if n_lines > self._EVO_LOG_MAX_LINES:
                    excess = n_lines - self._EVO_LOG_MAX_LINES
                    self.evo_log_text.delete("1.0", f"{excess + 1}.0")
                self.evo_log_text.see(END)
            except Exception:
                pass
        try:
            self.root.after(0, _do)
        except Exception:
            pass

    def _start_evolution_lab(self):
        if self._evolution_runner is not None and self._evolution_runner.is_running:
            messagebox.showinfo(
                "Already running",
                "Evolution Lab is already running. Click STOP first if you want to change settings "
                "and start a fresh run.",
            )
            return
        if not self.evo_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data file in 02 Data before starting the Evolution Lab.")
            return
        if not self._try_start_heavy_job(JOB_EVOLUTION_LAB):
            return

        from app.evolution.engine import EvolutionConfig, EvolutionRunner

        try:
            per_file_results = []
            for p in self.evo_context.csv_paths:
                result = import_csv(p)
                if not result.is_valid:
                    messagebox.showerror("Import error", f"{os.path.basename(p)}:\n" + "\n".join(result.errors))
                    self._release_heavy_job(JOB_EVOLUTION_LAB)
                    return
                per_file_results.append(result)
            if len(per_file_results) == 1:
                df = per_file_results[0].dataframe
            else:
                df, _labels = merge_multi_timeframe([r.dataframe for r in per_file_results])
        except Exception as exc:
            messagebox.showerror("Could not load data", str(exc))
            self._release_heavy_job(JOB_EVOLUTION_LAB)
            return

        risk = self.evo_context.build_risk_config()
        rules = self.evo_context.build_prop_rules()
        selected_families = [fam for fam, var in self.evo_family_vars.items() if var.get()] or None
        max_gen_raw = self.evo_max_generations.get_str().strip()
        max_generations = int(max_gen_raw) if max_gen_raw.isdigit() else None
        prefilter_bars_raw = self.evo_prefilter_max_bars.get_str().strip()
        prefilter_max_bars = int(prefilter_bars_raw) if prefilter_bars_raw.isdigit() else None
        loop_target_raw = self.evo_loop_target.get_str().strip()
        loop_target = float(loop_target_raw) if loop_target_raw else None
        loop_metric_key = (
            "eval_pass_probability"
            if self.evo_loop_target_metric.get_str().startswith("Raw in-sample")
            else "cpcv_oos_eval_pass_probability"
        )

        _evo_instrument = (
            os.path.basename(self.evo_context.csv_paths[0]) if len(self.evo_context.csv_paths) == 1
            else " + ".join(os.path.basename(p) for p in self.evo_context.csv_paths)
        ) if getattr(self, "csv_paths", None) else "unknown"
        cfg = EvolutionConfig(
            population_size=self.evo_population.get_int(60),
            elite_keep=self.evo_elite_keep.get_int(10),
            instrument=_evo_instrument,
            families=selected_families,
            min_trades=self.evo_min_trades.get_int(20),
            mc_sims=self.evo_mc_sims.get_int(1000),
            cpcv_top_n=self.evo_cpcv_top_n.get_int(8),
            stress_cost_multiplier=self.evo_stress_mult.get_float(2.0),
            max_generations=max_generations,
            adaptive_risk_enabled=self.evo_adaptive_risk_enabled.var.get(),
            prefilter_max_bars=prefilter_max_bars,
            target_eval_pass_pct=loop_target,
            target_metric=loop_metric_key,
            reset_on_breach=self.evo_reset_on_breach.var.get(),
            optimizer_mode=self._optimizer_mode_label_to_key.get(self.evo_optimizer_mode.get_str(), "genetic"),
            sealed_final_frac=max(0.0, min(self.evo_sealed_final_frac.get_float(0.10), 0.4)),
        )
        self.evo_log_text.delete("1.0", END)
        self._evo_guide_shown = False
        self._evolution_runner = EvolutionRunner(df, risk, rules, cfg, progress_cb=self._evo_log)
        self._evolution_runner.start()
        self._evo_log(
            f"Evolution Lab started -- population {cfg.population_size}, elite keep {cfg.elite_keep}, "
            f"{'unlimited generations' if cfg.max_generations is None else f'max {cfg.max_generations} generations'}."
        )
        if loop_target is not None:
            self._evo_log(
                f"Loop mode ON -- will stop automatically once a candidate clears {loop_target:.1f}% "
                f"on {loop_metric_key}."
            )
        self._poll_evolution_status()
        self._refresh_evolution_tested()

    def _stop_evolution_lab(self):
        if self._evolution_runner is None or not self._evolution_runner.is_running:
            messagebox.showinfo("Not running", "Evolution Lab isn't currently running.")
            return
        self._evolution_runner.stop()
        self._evo_log("Stop requested -- finishing the current generation, then stopping. Progress up to "
                       "that point is saved -- START again to resume.")

    def _reset_evolution_lab(self):
        if self._evolution_runner is not None and self._evolution_runner.is_running:
            messagebox.showwarning(
                "Still running", "Click STOP first, then RESET once it's finished the current generation."
            )
            return
        if not messagebox.askyesno(
            "Reset Evolution Lab",
            "This discards the saved generation/leaderboard/journal checkpoint and the tested-strategies "
            "log, so the next START begins a completely fresh run. This cannot be undone. Continue?",
        ):
            return
        if self._evolution_runner is None:
            evo_checkpoint.clear_checkpoint()
            evo_checkpoint.clear_tested_log()
        else:
            self._evolution_runner.reset()
        self.evo_log_text.delete("1.0", END)
        self.evo_leaderboard_listbox.delete(0, END)
        self.evo_tested_listbox.delete(0, END)
        self._evo_leaderboard_cache = []
        self.evo_status_label.config(text="Not running. (Saved progress cleared.)", fg=TEXT_DIM)
        self._evo_log("Evolution Lab progress reset -- the next START begins a fresh run.")

    def _refresh_evolution_tested(self):
        try:
            self.evo_tested_listbox.delete(0, END)
            if self._evolution_runner is not None:
                rows = self._evolution_runner.tested_candidates(limit=300)
            else:
                rows = evo_checkpoint.read_tested_rows(limit=300)
            for row in reversed(rows):
                gen = row.get("generation", "?")
                fam = str(row.get("family", "?"))[:16]
                stage = row.get("stage", "?")
                if row.get("passed"):
                    if stage == "full_eval":
                        score = row.get("fitness_score")
                        detail = f"PASSED prefilter, fitness {score:.2f}" if isinstance(score, (int, float)) else "PASSED prefilter"
                    else:
                        detail = "PASSED prefilter"
                else:
                    reasons = ", ".join(row.get("reasons") or []) or (row.get("error") or "unknown")
                    detail = f"rejected: {reasons}"
                trades = row.get("n_trades")
                trades_str = f"{trades} trades" if trades is not None else ""
                self.evo_tested_listbox.insert(
                    END, f"  gen {gen:>3}  {fam:16s}  {trades_str:12s}  {detail}"
                )
        except Exception:
            pass

    def _refresh_evo_leaderboard_listbox(self, records: list[dict]) -> None:
        """`records` is a list of checkpoint-shaped dicts (candidate_id,
        spec, meta, stats, mc_summary, fitness, ...) -- the same shape
        whether they came from a live runner (via to_checkpoint_dict())
        or straight off disk. Keeps `self._evo_leaderboard_cache`
        index-aligned with the listbox rows so VIEW DETAILS / PROMOTE
        can look up the right one from the current selection."""
        self.evo_leaderboard_listbox.delete(0, END)
        self._evo_leaderboard_cache = records
        for r in records:
            fitness = r.get("fitness") or {}
            score = fitness.get("final_score", float("nan"))
            fam = str((r.get("meta") or {}).get("family", "?"))[:18]
            stats = r.get("stats") or {}
            wr = stats.get("win_rate")
            wr_str = f"{wr:5.1f}% WR" if isinstance(wr, (int, float)) else "   n/a WR"
            trades = stats.get("total_trades")
            trades_str = f"{trades:4d} trades" if isinstance(trades, int) else "   ? trades"
            # cpcv_oos_eval_pass_probability is the honest, held-out
            # estimate from engine.py's _cpcv_and_pbo -- shown right on
            # the leaderboard row (not just buried in VIEW DETAILS) so a
            # big gap against the raw in-sample number is visible before
            # PROMOTE is even clicked, not just as a one-time confirmation
            # dialog at promote time. Real report this fixes: strategies
            # promoted at a raw "40% pass / 30% payout" coming back 2%/1%
            # out of 15 Full Pipeline -- this is the number that would
            # have flagged that in advance.
            oos_pct = r.get("cpcv_oos_eval_pass_probability")
            oos_str = f"  OOS:{oos_pct:5.1f}%" if isinstance(oos_pct, (int, float)) else ""
            self.evo_leaderboard_listbox.insert(
                END,
                f"  {score:8.2f}   {fam:18s}  {trades_str}  {wr_str}{oos_str}  {r.get('candidate_id', '?')}",
            )

    def _load_evolution_leaderboard_from_disk(self) -> None:
        """Populates the leaderboard from the saved checkpoint even when
        no EvolutionRunner is currently alive in this process (e.g. the
        app was closed and reopened after an overnight run) -- Evolution
        Lab saves progress to disk after every generation specifically so
        this works."""
        if self._evolution_runner is not None:
            try:
                self._refresh_evo_leaderboard_listbox([r.to_checkpoint_dict() for r in self._evolution_runner.leaderboard])
                return
            except Exception:
                pass
        try:
            checkpoint = evo_checkpoint.load_checkpoint()
        except Exception:
            checkpoint = None
        if checkpoint is None:
            return
        try:
            self._refresh_evo_leaderboard_listbox(list(checkpoint.leaderboard))
        except Exception:
            pass

    def _selected_evolution_leader(self) -> dict | None:
        sel = self.evo_leaderboard_listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Select a strategy from the Leaderboard first.")
            return None
        idx = sel[0]
        if idx >= len(self._evo_leaderboard_cache):
            return None
        return self._evo_leaderboard_cache[idx]

    def _view_evolution_leader_detail(self) -> None:
        record = self._selected_evolution_leader()
        if record is None:
            return
        stats = record.get("stats") or {}
        mc = record.get("mc_summary") or {}
        fitness = record.get("fitness") or {}
        robustness = record.get("robustness") or {}
        walk_forward = record.get("walk_forward") or {}

        def pct(v):
            # Fractions (0..1), e.g. PropFitnessBreakdown.pass_probability.
            return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"

        def pct_already(v):
            # Already-a-percentage values (0..100), e.g.
            # MonteCarloResult.evaluation_pass_probability /
            # first_payout_probability. Passing these through pct() above
            # multiplied an already-0..100 number by 100 again -- a real
            # 5.5% pass probability rendered as "550.0%", which is where
            # Evolution Lab leaderboard details showing >100% (and the
            # impossible-looking "550% pass / 0% payout" combination)
            # came from. These two Monte Carlo fields never get *100.
            return f"{v:.1f}%" if isinstance(v, (int, float)) else "n/a"

        def num(v, fmt="{:.2f}"):
            return fmt.format(v) if isinstance(v, (int, float)) else "n/a"

        lines = [
            f"Candidate: {record.get('candidate_id', '?')}",
            f"Family: {(record.get('meta') or {}).get('family', '?')}",
            "",
            "-- Prop fitness --",
            f"  Final score:          {num(fitness.get('final_score'))}",
            f"  Pass probability:     {pct(fitness.get('pass_probability'))}",
            f"  Payout probability:   {pct(fitness.get('payout_probability'))}",
            f"  Robustness:           {num(fitness.get('robustness'))}",
            f"  OOS consistency:      {num(fitness.get('oos_consistency'))}",
            "",
            "-- Backtest stats --",
            f"  Total trades:         {stats.get('total_trades', 'n/a')}",
            f"  Win rate:             {(num(stats.get('win_rate'), '{:.1f}') + '%') if isinstance(stats.get('win_rate'), (int, float)) else 'n/a'}",
            f"  Net profit:           ${num(stats.get('net_profit'))}",
            f"  Profit factor:        {num(stats.get('profit_factor'))}",
            f"  Sharpe ratio:         {num(stats.get('sharpe_ratio'))}",
            f"  Sortino ratio:        {num(stats.get('sortino_ratio'))}",
            f"  Max drawdown:         {num(stats.get('max_drawdown_pct'), '{:.1f}')}%",
            f"  Expectancy:           {num(stats.get('expectancy'))}",
            "",
            "-- Monte Carlo / eval simulation (IN-SAMPLE -- same data the GA searched) --",
            f"  Evaluation pass probability:  {pct_already(mc.get('evaluation_pass_probability'))}",
            f"  First payout probability:     {pct_already(mc.get('first_payout_probability'))}",
            "",
            "-- Robustness / walk-forward --",
            f"  Parameter-neighborhood stable:  {robustness.get('is_stable', 'n/a')}  "
            f"(stability ratio {num(robustness.get('stability_ratio'))})",
            f"  Walk-forward stable:            {walk_forward.get('is_stable', 'n/a')}  "
            f"(WF efficiency {num(walk_forward.get('walk_forward_efficiency'))})",
        ]
        if record.get("pbo") is not None:
            lines.append(f"  PBO (probability of backtest overfitting): {num(record.get('pbo'))}")
        if record.get("cpcv_degradation") is not None:
            lines.append(f"  CPCV degradation: {num(record.get('cpcv_degradation'))}")
        oos_pct = record.get("cpcv_oos_eval_pass_probability")
        if oos_pct is not None:
            raw_pct = mc.get("evaluation_pass_probability")
            gap_note = ""
            if isinstance(raw_pct, (int, float)) and isinstance(oos_pct, (int, float)) and (raw_pct - oos_pct) > 15:
                gap_note = "  <-- big gap vs. raw in-sample number above; likely overfit, verify with Full Pipeline"
            lines.append(f"  CPCV out-of-sample eval pass probability: {num(oos_pct, '{:.1f}')}%{gap_note}")

        win = Toplevel(self.root)
        win.title(f"Leaderboard detail -- {record.get('candidate_id', '?')}")
        win.configure(bg=BG)
        win.geometry("760x680")
        try:
            _apply_dark_titlebar(win)
        except Exception:
            pass  # cosmetic only -- see _apply_dark_titlebar's own docstring
        Label(
            win, text="LEADERBOARD DETAIL", bg=BG, fg=TEXT, font=_safe_font(13, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 6))

        text_frame = Frame(win, bg=BG)
        text_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        txt = Text(
            text_frame, wrap="word", bg=PANEL_3, fg=TEXT, font=(MONO, 10), relief="flat", bd=0,
        )
        vs = ttk.Scrollbar(text_frame, orient="vertical", command=txt.yview, style="T58.Vertical.TScrollbar")
        txt.config(yscrollcommand=vs.set)
        self._bind_isolated_wheel(txt)
        txt.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        txt.insert("1.0", "\n".join(lines))
        txt.config(state="disabled")

        btn_row = Frame(win, bg=BG)
        btn_row.pack(fill="x", padx=16, pady=(0, 14))
        config = (record.get("spec") or {}).get("config")

        def _view_code():
            if not config:
                messagebox.showinfo("No config", "This candidate has no saved manual-builder config to show.")
                return
            self._show_text_viewer(
                f"Config -- {record.get('candidate_id', '?')}", json.dumps(config, indent=2),
            )

        def _promote():
            self._promote_evolution_leader_record(record)

        self._button(btn_row, "VIEW CODE / CONFIG", _view_code, primary=True).pack(side="left")
        self._button(btn_row, "PROMOTE TO STRATEGY LIBRARY", _promote).pack(side="left", padx=8)
        self._button(btn_row, "CLOSE", win.destroy).pack(side="left", padx=8)

    def _final_test_selected_evolution_leader(self) -> None:
        """One-shot sealed FINAL test -- same as the web Evolution Lab's
        "Final test" button (POST /evolution/final-test)."""
        record = self._selected_evolution_leader()
        if record is None:
            return
        runner = self._evolution_runner
        if runner is None:
            messagebox.showinfo("No live run", "The sealed final test needs this session's run -- start (and then stop) the Evolution Lab first.")
            return
        rec = next((r for r in runner.leaderboard if r.candidate_id == record.get("candidate_id")), None)
        if rec is None:
            messagebox.showinfo("Not on the leaderboard", "That candidate is not on this session's live leaderboard.")
            return
        try:
            r = runner.evaluate_champion_on_sealed_final(rec)
        except ValueError as exc:
            messagebox.showinfo("Final test unavailable", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Final test failed", str(exc))
            return
        messagebox.showinfo(
            "Sealed final test",
            f"{r.get('n_bars', 0)} unseen bars, {r.get('n_trades', 0)} trades\n"
            f"Per-attempt eval pass: {r.get('per_attempt_pass_probability', 0):.1f}%\n"
            f"Per-attempt payout: {r.get('per_attempt_payout_probability', 0):.1f}%\n"
            f"Risk of ruin: {r.get('risk_of_ruin_pct', 0):.1f}%\n\n{r.get('note', '')}"
            + ("\n\n(Stored result from the first look -- not re-scored.)" if r.get("already_evaluated") else ""),
        )

    def _promote_selected_evolution_leader(self) -> None:
        record = self._selected_evolution_leader()
        if record is None:
            return
        self._promote_evolution_leader_record(record)

    def _promote_evolution_leader_record(self, record: dict) -> None:
        """Saves a specific leaderboard entry into the Strategy Library
        (manual-builder JSON, same format Evolution Lab's own
        save_to_library option writes) tagged 'promoted', independent of
        whether auto-save-to-library is enabled for the run -- this is
        the on-demand version so a leader found overnight can be pushed
        into the library and run through 15 Full Pipeline without
        needing to re-run the whole generation with that option on."""
        config = (record.get("spec") or {}).get("config")
        if not config:
            messagebox.showwarning(
                "Nothing to promote",
                "This candidate has no manual-builder config attached, so there's nothing to save "
                "to the Strategy Library.",
            )
            return
        # Real-report root cause: a strategy promoted straight off the raw
        # leaderboard number (e.g. 40% pass / 30% payout) can come back
        # from 15 Full Pipeline at 2%/1% -- because everything upstream of
        # engine.py's _cpcv_and_pbo evaluates each candidate against the
        # SAME data the GA searched and selected it against for
        # potentially hundreds of generations, so a high raw number there
        # is exactly as likely to be "this genome fit this dataset's noise
        # well" as a real edge. _cpcv_and_pbo already computes an honest,
        # held-out estimate (cpcv_oos_eval_pass_probability) for exactly
        # this reason and logs a warning when the gap is large -- but that
        # console line was easy to miss, and PROMOTE never checked it
        # before this. Now it does, right before the save actually
        # happens, using the same >15-point gap threshold the log warning
        # already uses.
        raw_pct = (record.get("mc_summary") or {}).get("evaluation_pass_probability")
        oos_pct = record.get("cpcv_oos_eval_pass_probability")
        if isinstance(raw_pct, (int, float)) and isinstance(oos_pct, (int, float)) and (raw_pct - oos_pct) > 15:
            proceed = messagebox.askyesno(
                "Large in-sample / out-of-sample gap",
                f"This candidate's raw (in-sample) eval pass probability is {raw_pct:.1f}%, but the "
                f"honest, held-out CPCV estimate is only {oos_pct:.1f}% -- a {raw_pct - oos_pct:.1f}-point "
                f"gap. That pattern (looks great in-sample, much weaker out-of-sample) is the classic "
                f"signature of the GA having fit this dataset's noise rather than a real edge, and is "
                f"exactly what tends to come back as a much lower pass/payout rate out of 15 Full "
                f"Pipeline's stricter walk-forward + holdout validation.\n\n"
                f"Promote it anyway? (15 Full Pipeline will still re-validate it properly either way -- "
                f"this is just an early warning so you don't have to wait for that run to find out.)",
            )
            if not proceed:
                return
        family = (record.get("meta") or {}).get("family", "strategy")
        cid = record.get("candidate_id", "unknown")
        filename = f"evolab_promoted_{family}_{cid[-8:]}.json"
        text = json.dumps(config, indent=2)
        try:
            try:
                save_strategy_text(text, filename, "manual", overwrite=False)
            except StrategyAlreadyExists:
                if not messagebox.askyesno(
                    "Already promoted", f"'{filename}' is already in the Strategy Library. Overwrite it?",
                ):
                    return
                save_strategy_text(text, filename, "manual", overwrite=True)
            # "promoted" is not one of STRATEGY_STATUSES (draft /
            # tested_failed / tested_passed / validated / ready_for_demo /
            # ready_for_live) -- that call used to raise "Unknown status
            # 'promoted'" immediately after the "manual" type fix above,
            # a second promote-flow bug behind the first. "validated" is
            # the correct existing status here, not a new one invented
            # for this: an Evolution Lab leaderboard candidate has
            # already cleared walk-forward AND CPCV/PBO (see engine.py's
            # pipeline) before it can appear on the leaderboard at all,
            # which is exactly what STRATEGY_STATUSES documents
            # "validated" as meaning.
            set_strategy_status("manual", filename, "validated")
        except Exception as exc:
            messagebox.showerror("Could not promote", str(exc))
            return
        messagebox.showinfo(
            "Promoted",
            f"Saved to Strategy Library as '{filename}' (manual, status: validated). "
            "Find it in 06 Strategy Library / 15 Full Pipeline's Batch selection panel to run it through "
            "the full validation pipeline.",
        )
        try:
            self._refresh_strategy_library()
        except Exception:
            pass

    def _poll_evolution_status(self):
        runner = self._evolution_runner
        if runner is None:
            return
        status = runner.status()
        resumed_note = " (resumed)" if status.get("resumed") else ""
        fam_health = status.get("family_health") or {}
        fam_note = ""
        if fam_health.get("applied") and fam_health.get("excluded"):
            excluded_n = len(fam_health["excluded"])
            active_n = fam_health.get("active_family_count")
            if active_n is not None and active_n >= len(list_families()):
                fam_note = (
                    f"  |  {excluded_n} famil{'y' if excluded_n == 1 else 'ies'} flagged dead-end but NOT "
                    f"excluded (would collapse below the active-family floor) -- searching all {active_n}"
                )
            else:
                fam_note = (
                    f"  |  {active_n} famil{'y' if active_n == 1 else 'ies'} active "
                    f"({excluded_n} excluded as dead-end)"
                )
        self.evo_status_label.config(
            text=(
                f"{'RUNNING' if status['running'] else 'STOPPED'} -- generation {status['generation']}, "
                f"leaderboard size {status['leaderboard_size']}{resumed_note}{fam_note}"
            ),
            fg=GREEN if status["running"] else TEXT_DIM,
        )
        if status.get("target_reached"):
            self.evo_loop_status_label.config(
                text=(
                    f"Loop mode target reached: candidate {status.get('target_reached_candidate_id')} "
                    f"cleared {status.get('target_eval_pass_pct')}% -- Evolution Lab stopped itself. "
                    "Review the leaderboard and promote it."
                ),
                fg=GREEN,
            )
        elif status.get("target_eval_pass_pct") is not None:
            self.evo_loop_status_label.config(
                text=f"Loop mode active -- will stop automatically once a candidate clears {status['target_eval_pass_pct']}%.",
                fg=TEXT_DIM,
            )
        else:
            self.evo_loop_status_label.config(text="")
        try:
            self._refresh_evo_leaderboard_listbox([r.to_checkpoint_dict() for r in runner.leaderboard])
        except Exception:
            pass
        if status["running"]:
            try:
                self.root.after(2000, self._poll_evolution_status)
                self.root.after(4000, self._refresh_evolution_tested)
            except Exception:
                pass
        else:
            # Covers both a deliberate STOP and the runner's own thread
            # exiting on its own (finished, or crashed -- see engine.py's
            # _run_loop finally block) -- either way the heavy-job slot
            # must free up so another tab can start.
            self._release_heavy_job(JOB_EVOLUTION_LAB)
            if not getattr(self, "_evo_guide_shown", False):
                self._evo_guide_shown = True
                self._evo_log(
                    "\n" + pipeline_guide.after_evolution_stop(
                        status["leaderboard_size"],
                        total_evaluated=status["generation"] * runner.cfg.population_size,
                    )
                )

    # -----------------------------------------------------------------------
    # Multi-Instrument Evolution Lab -- desktop parity with the web app's
    # /evolution/multi-instrument page: a MultiInstrumentEvolutionGroup
    # (app.evolution.multi_instrument) owning one independent EvolutionRunner
    # per selected instrument/timeframe, each on its own background thread
    # with its own fully-isolated checkpoint/tested-log/knowledge-graph
    # files. Unlike Speed Run/Search Lab's multi-instrument tabs (a single
    # run-to-completion call), this is start/stop + poll, same shape as the
    # single-instrument Evolution Lab tab.
    # -----------------------------------------------------------------------

    def _build_evolution_multi_tab(self):
        f = self._scrollable(self.tab_evolution_multi)

        self._page_header(
            f,
            "OPTIMIZE / Multi-Instrument Evolution Lab",
            "\u2620 Multi-Instrument Evolution Lab",
            "A named group of independent Evolution Lab runners, one per selected "
            "instrument/timeframe (select 2 or more below), each with its OWN checkpoint/ "
            "tested-log/knowledge-graph files -- never shared, so two instruments never "
            "pollute each other's search history. Every runner shares the same population/"
            "generation/family settings below; each stops itself independently once ITS OWN "
            "leaderboard clears any loop-mode target, so one instrument finishing early "
            "doesn't stop the others.",
        )

        self.evomulti_context = RunContextPanel(self, "Multi-Instrument Evolution Lab")
        self.evomulti_context.build(f)

        settings = self._section(f, "Evolution settings (applied identically to every instrument)", emphasize=True)
        self.evomulti_population = LabeledEntry(settings, "Population size", 40)
        self.evomulti_elite_keep = LabeledEntry(settings, "Elite keep", 8)
        self.evomulti_min_trades = LabeledEntry(settings, "Minimum trades to be scoreable", 20)
        self.evomulti_mc_sims = LabeledEntry(settings, "Monte Carlo sims per candidate", 500)
        self._optimizer_mode_labels = list(OPTIMIZER_MODES.values())
        self._optimizer_mode_label_to_key = {v: k for k, v in OPTIMIZER_MODES.items()}
        self.evomulti_optimizer_mode = LabeledCombo(
            settings, "Optimizer mode (per-family child proposals)", self._optimizer_mode_labels,
            OPTIMIZER_MODES["genetic"],
        )
        self.evomulti_max_generations = LabeledEntry(settings, "Max generations (blank = unlimited)", "")

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "START GROUP", self._evo_multi_start_clicked, primary=True).pack(side="left")
        self._button(button_row, "STOP GROUP", self._evo_multi_stop_clicked).pack(side="left", padx=8)

        self.evomulti_status_label = Label(f, text="Not started.", bg=BG, fg=TEXT_DIM, font=_safe_font(10, "bold"))
        self.evomulti_status_label.pack(anchor="w", padx=24, pady=(2, 10))

        results_section = self._section(f, "Per-instrument status", "Refreshes automatically every 2 seconds while running.")
        self.evomulti_results_listbox = Listbox(
            results_section, height=8, exportselection=False, bg=PANEL_3, fg=TEXT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.evomulti_results_listbox.pack(fill="x", padx=18, pady=(2, 12))
        self._bind_isolated_wheel(self.evomulti_results_listbox)

        output_section = self._section(f, "Multi-Instrument Evolution Lab log", "Combined log across every instrument in the group.")
        _em_output_frame = Frame(output_section, bg=PANEL)
        self.evomulti_output = Text(
            _em_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _em_output_scroll = ttk.Scrollbar(
            _em_output_frame, orient="vertical", command=self.evomulti_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.evomulti_output.configure(yscrollcommand=_em_output_scroll.set)
        self.evomulti_output.pack(side="left", fill="both", expand=True)
        _em_output_scroll.pack(side="right", fill="y")
        _em_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.evomulti_output)

        self._evo_multi_group = None
        self._evo_multi_seen_log_counts: dict[str, int] = {}

    def _evo_multi_start_clicked(self):
        if self._evo_multi_group is not None and self._evo_multi_group.is_running:
            messagebox.showinfo("Already running", "Stop the current group first if you want to change settings and start fresh.")
            return
        if len(self.evomulti_context.csv_paths) < 2:
            messagebox.showwarning(
                "Select at least 2 datasets",
                "Select 2 or more datasets above (Ctrl/Cmd-click or Shift-click) -- with only "
                "1 selected, use the regular Evolution Lab tab instead.",
            )
            return

        from app.evolution.engine import EvolutionConfig
        from app.evolution.multi_instrument import EvolutionInstrumentJob, MultiInstrumentEvolutionGroup

        jobs = []
        for path in self.evomulti_context.csv_paths:
            instrument, timeframe = self._misr_instrument_label_for_path(path)
            jobs.append(EvolutionInstrumentJob(instrument=instrument, timeframe=timeframe, csv_path=path))

        risk = self.evomulti_context.build_risk_config()
        rules = self.evomulti_context.build_prop_rules()
        max_gen_raw = self.evomulti_max_generations.get_str().strip()
        base_cfg = EvolutionConfig(
            population_size=self.evomulti_population.get_int(40),
            elite_keep=self.evomulti_elite_keep.get_int(8),
            min_trades=self.evomulti_min_trades.get_int(20),
            mc_sims=self.evomulti_mc_sims.get_int(500),
            max_generations=int(max_gen_raw) if max_gen_raw.isdigit() else None,
            optimizer_mode=self._optimizer_mode_label_to_key.get(self.evomulti_optimizer_mode.get_str(), "genetic"),
        )

        if not self._try_start_heavy_job(JOB_MULTI_INSTRUMENT_EVOLUTION):
            return
        group_id = uuid.uuid4().hex[:10]
        self.evomulti_output.delete("1.0", END)
        self._evo_multi_seen_log_counts = {}
        try:
            self._evo_multi_group = MultiInstrumentEvolutionGroup(group_id, jobs, risk, rules, base_cfg)
        except Exception as exc:
            self._release_heavy_job(JOB_MULTI_INSTRUMENT_EVOLUTION)
            messagebox.showerror("Could not start group", str(exc))
            return
        for label, err in self._evo_multi_group.errors.items():
            self.evomulti_output.insert(END, f"[{label}] SKIPPED: {err}\n")
        try:
            self._evo_multi_group.start_all()
        except Exception as exc:
            self._release_heavy_job(JOB_MULTI_INSTRUMENT_EVOLUTION)
            messagebox.showerror("Could not start group", str(exc))
            return
        self.evomulti_output.insert(
            END, f"Started group {group_id} on {len(self._evo_multi_group.runners)} instrument(s).\n",
        )
        self._poll_evo_multi_status()

    def _evo_multi_stop_clicked(self):
        if self._evo_multi_group is None:
            return
        self.evomulti_status_label.config(text="Stopping...", fg=AMBER)
        threading.Thread(target=self._evo_multi_group.stop_all, daemon=True).start()

    def _poll_evo_multi_status(self):
        group = self._evo_multi_group
        if group is None:
            return
        status = group.status()
        self.evomulti_status_label.config(
            text=f"{'RUNNING' if status['running'] else 'STOPPED'} -- {len(status['instruments'])} instrument(s) in group",
            fg=GREEN if status["running"] else TEXT_DIM,
        )
        self.evomulti_results_listbox.delete(0, END)
        for label, inst in sorted(status["instruments"].items()):
            if inst.get("error"):
                self.evomulti_results_listbox.insert(END, f"[FAILED] {label} -- {inst['error']}")
                continue
            leader = inst["leaderboard"][0] if inst["leaderboard"] else None
            leader_note = f"  best {leader['candidate_id']}" if leader else "  no leaderboard entry yet"
            state = "RUNNING" if inst["running"] else "stopped"
            self.evomulti_results_listbox.insert(
                END, f"[{state:7s}] {label} -- gen {inst['generation']}, "
                     f"leaderboard {inst['leaderboard_size']}{leader_note}",
            )
            new_lines = inst["log"][self._evo_multi_seen_log_counts.get(label, 0):]
            for line in new_lines:
                self.evomulti_output.insert(END, f"[{label}] {line}\n")
            self._evo_multi_seen_log_counts[label] = len(inst["log"])
        self.evomulti_output.see(END)
        if status["running"]:
            try:
                self.root.after(2000, self._poll_evo_multi_status)
            except Exception:
                pass
        else:
            # Group finished or was stopped -- free the heavy-job slot,
            # same as the single-instrument Evolution Lab poll above.
            self._release_heavy_job(JOB_MULTI_INSTRUMENT_EVOLUTION)
