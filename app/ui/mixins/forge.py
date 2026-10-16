"""Forge tab and forge pipelines."""
from __future__ import annotations

import os
import threading
import traceback
from app.data.timeframe_resample import infer_timeframe_label as _infer_timeframe_label
from app.orchestration.forge import ForgeConfig, run_forge
from app.orchestration.loop_runner import ForgeLoopConfig, run_forge_loop
from app.orchestration.resource_guard import JOB_FORGE
from app.reports.crash_log import log_crash
from app.search.batch_runner import SearchCancelled
from app.search.graveyard import graveyard_path_for, list_graveyard_files, load_graveyard, summarize_graveyard
from pathlib import Path
from tkinter import END, Frame, Label, Text, Toplevel, messagebox, ttk
from app.ui.common import OUTPUT_DIR, _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledEntry
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
AMBER = ""
theme.bind(globals())


class ForgeMixin:
    """Forge tab and forge pipelines."""


    # ------------------------------------------------------------------
    # Forge Strategy -- desktop parity for the web app's /forge. Same
    # thin orchestrator (app.orchestration.forge.run_forge), same single-
    # button generate -> screen -> validate -> leaderboard funnel, same
    # persistent, instrument-scoped Strategy Graveyard feedback loop.
    # Uses whatever's already configured on Steps 02 (Data) / 03 (Prop
    # Rules) / 04 (Risk) -- same convention as Full Pipeline above, not a
    # duplicated set of account/risk fields.
    # ------------------------------------------------------------------

    def _build_forge_tab(self):
        f = self._scrollable(self.tab_forge)

        self._page_header(
            f,
            "FORGE STRATEGY",
            "Forge Strategy",
            "One button: generates thousands of distinct, named market-hypothesis strategies "
            "(every family in app.search.strategy_space, not random parameter mutation), screens "
            "them hard and cheaply, and only spends expensive validation compute (CPCV/PBO, regime "
            "testing, deeper Monte Carlo, rolling prop evaluation, a locked out-of-sample holdout) "
            "on the survivors -- ranked the whole way through by PROP SURVIVAL, not raw profit. "
            "Every rejection expensive enough to be worth explaining is diagnosed and logged to the "
            "Strategy Graveyard, and a hypothesis whose parameter neighborhood a PRIOR run already "
            "proved dead is skipped automatically before a single backtest runs. Its own market "
            "data, prop rules, and risk settings below are self-contained to this tab, just like "
            "the web app.",
        )

        self.forge_context = RunContextPanel(self, "Forge Strategy")
        self.forge_context.build(f)

        settings = self._section(
            f, "Forge settings",
            "Sensible defaults for a real overnight-scale run. Lower n_hypotheses / stage sizes for "
            "a much faster (less thorough) smoke test first.",
            emphasize=True,
        )
        self.forge_n_hypotheses = LabeledEntry(settings, "Hypotheses to generate", 10_000)
        self.forge_seed = LabeledEntry(settings, "Random seed", 42)
        self.forge_workers = LabeledEntry(settings, "Parallel workers (blank = all CPU cores)", "")

        stage1_section = self._section(
            f, "Stage 1 -- fast screen",
            "Auto-relaxes twice on its own if the strict defaults find nothing (see the run log for "
            "exactly what it loosened to) before concluding the search space itself has no edge.",
        )
        self.forge_min_trades = LabeledEntry(stage1_section, "Minimum trades to survive", 20)
        self.forge_min_pf = LabeledEntry(stage1_section, "Minimum profit factor to survive", 1.05)
        self.forge_stage1_top_n = LabeledEntry(stage1_section, "Survivors that advance to Stage 2 (GA)", 1000)

        stage2_section = self._section(
            f, "Stage 2 -- prop survival screen (GA)",
            "Fitness = eval_pass_probability -- optimized for surviving a prop eval, not raw profit.",
        )
        self.forge_ga_population = LabeledEntry(stage2_section, "GA population size", 12)
        self.forge_ga_generations = LabeledEntry(stage2_section, "GA generations", 4)
        self.forge_stage2_top_n = LabeledEntry(stage2_section, "Survivors that advance to Stage 3", 200)

        stage3_section = self._section(
            f, "Stage 3 onward -- validation depth",
            "Neighbor robustness + walk-forward + Monte Carlo (batch_runner's own Stage 3), then "
            "Forge's own CPCV/PBO + regime testing, a deeper final Monte Carlo, rolling prop "
            "evaluation across every real historical starting day, and a locked OOS holdout "
            "reserved before hypothesis generation even started.",
        )
        self.forge_stage3_mc_sims = LabeledEntry(stage3_section, "Stage 3 Monte Carlo sims", 2000)
        self.forge_cpcv_pool_size = LabeledEntry(stage3_section, "Stage 3 survivors sent to CPCV/regime testing", 30)
        self.forge_cpcv_survivors = LabeledEntry(stage3_section, "CPCV/regime survivors kept", 10)
        self.forge_final_mc_sims = LabeledEntry(stage3_section, "Final Monte Carlo sims (deeper)", 10_000)
        self.forge_mc_survivors = LabeledEntry(stage3_section, "Final Monte Carlo survivors kept", 5)
        self.forge_eval_window_days = LabeledEntry(stage3_section, "Rolling evaluation window (trading days)", 60)
        self.forge_rolling_survivors = LabeledEntry(stage3_section, "Rolling evaluation survivors kept", 2)
        self.forge_locked_holdout_frac = LabeledEntry(stage3_section, "Locked OOS holdout fraction", 0.15)

        graveyard_section = self._section(
            f, "Strategy Graveyard feedback loop",
            "Skips a hypothesis before it costs a single backtest if a PRIOR Forge run against this "
            "same instrument/timeframe already proved that exact parameter neighborhood dead this "
            "many times or more. Turn off to force a completely fresh sweep of the whole space.",
        )
        self.forge_graveyard_skip = LabeledCheckbox(
            graveyard_section, "Skip hypotheses in known-dead parameter neighborhoods", True,
        )
        self.forge_graveyard_min_attempts = LabeledEntry(graveyard_section, "Minimum prior failures before skipping", 8)

        loop_section = self._section(
            f, "Loop mode",
            "Repeats Forge Strategy automatically: if a round makes no progress for a few rounds "
            "in a row, the next round searches DEEPER (more hypotheses, more survivors carried "
            "forward) instead of stopping. Stops the moment a champion clears your target pass "
            "rate, when the round or time budget runs out, or when you hit STOP.",
        )
        self.forge_loop_mode = LabeledCheckbox(loop_section, "Enable Loop Mode", False)
        self.forge_loop_target = LabeledEntry(loop_section, "Stop once a champion's pass rate reaches", 60)
        self.forge_loop_max_rounds = LabeledEntry(loop_section, "Max rounds", 10)
        self.forge_loop_time_budget_hours = LabeledEntry(
            loop_section, "Time budget in hours (blank = no limit, rely on rounds/STOP)", "",
        )
        self.forge_loop_stall_rounds = LabeledEntry(loop_section, "Rounds with no improvement before searching deeper", 2)
        self.forge_loop_require_locked_oos = LabeledCheckbox(
            loop_section, "Only count a champion that also passed the locked holdout check", True,
        )

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN FORGE", self._forge_run_clicked, primary=True).pack(side="left")
        self.stop_forge_btn = self._button(button_row, "STOP", self._forge_stop_clicked)
        self.stop_forge_btn.config(state="disabled")
        self.stop_forge_btn.pack(side="left", padx=8)
        self._button(button_row, "OPEN STRATEGY GRAVEYARD", self._open_forge_graveyard).pack(side="left", padx=8)

        self.forge_progress = NeuralProgress(f)
        self.forge_progress.pack(fill="x", padx=24, pady=(2, 10))

        champion_section = self._section(f, "Champion", "Filled in once a run completes with at least one PASSED locked-OOS survivor.")
        self.forge_champion_label = Label(
            champion_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=900, anchor="w",
        )
        self.forge_champion_label.pack(anchor="w", fill="x", padx=18, pady=(2, 10))

        funnel_section = self._section(f, "Funnel", "How many candidates survived each stage.")
        funnel_tree_frame = Frame(funnel_section, bg=PANEL)
        funnel_tree_frame.pack(fill="both", padx=18, pady=(2, 12))
        self.forge_funnel_tree = ttk.Treeview(
            funnel_tree_frame, columns=("stage", "in", "out"), show="headings",
            style="T58.Treeview", height=9,
        )
        for col, text, w in (("stage", "Stage", 380), ("in", "In", 90), ("out", "Out", 90)):
            self.forge_funnel_tree.heading(col, text=text)
            self.forge_funnel_tree.column(col, width=w, anchor="w")
        self.forge_funnel_tree.pack(side="left", fill="both", expand=True)
        self._bind_isolated_wheel(self.forge_funnel_tree)

        leaderboard_section = self._section(f, "Leaderboard", "Ranked by prop survival score, not raw profit.")
        lb_tree_frame = Frame(leaderboard_section, bg=PANEL)
        lb_tree_frame.pack(fill="both", padx=18, pady=(2, 12))
        lb_columns = (
            "candidate", "family", "pass_rate", "payout_rate", "survival", "locked_oos",
            "max_dd", "net_profit", "trades",
        )
        self.forge_leaderboard_tree = ttk.Treeview(
            lb_tree_frame, columns=lb_columns, show="headings", style="T58.Treeview", height=8,
        )
        lb_headings = {
            "candidate": "Candidate", "family": "Family", "pass_rate": "Pass %", "payout_rate": "Payout %",
            "survival": "Survival score", "locked_oos": "Locked OOS", "max_dd": "Max DD %",
            "net_profit": "Net P/L", "trades": "Trades",
        }
        for col, text in lb_headings.items():
            self.forge_leaderboard_tree.heading(col, text=text)
            self.forge_leaderboard_tree.column(col, width=110, anchor="w")
        self.forge_leaderboard_tree.pack(side="left", fill="both", expand=True)
        lb_scroll = ttk.Scrollbar(
            lb_tree_frame, orient="vertical", command=self.forge_leaderboard_tree.yview, style="T58.Vertical.TScrollbar",
        )
        lb_scroll.pack(side="right", fill="y")
        self.forge_leaderboard_tree.configure(yscrollcommand=lb_scroll.set)
        self._bind_isolated_wheel(self.forge_leaderboard_tree)

        output_section = self._section(f, "Forge output", "Live progress log -- every stage, every auto-relax, every graveyard skip.")
        _forge_output_frame = Frame(output_section, bg=PANEL)
        self.forge_output = Text(
            _forge_output_frame, height=20, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _forge_output_scroll = ttk.Scrollbar(
            _forge_output_frame, orient="vertical", command=self.forge_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.forge_output.configure(yscrollcommand=_forge_output_scroll.set)
        self.forge_output.pack(side="left", fill="both", expand=True)
        _forge_output_scroll.pack(side="right", fill="y")
        _forge_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.forge_output)

        self._forge_cancel_event = threading.Event()
        self._last_forge_result = None
        self._last_forge_graveyard_path = None

    def _log_forge(self, msg: str) -> None:
        def _do():
            try:
                self.forge_output.insert(END, msg + "\n")
                self.forge_output.see(END)
            except Exception:
                pass
        try:
            self.root.after(0, _do)
        except Exception:
            pass

    def _forge_run_clicked(self):
        if not self.forge_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 2 (Market Data).")
            return
        if not self._try_start_heavy_job(JOB_FORGE):
            return
        self.forge_output.delete("1.0", END)
        for tree in (self.forge_funnel_tree, self.forge_leaderboard_tree):
            for row in tree.get_children():
                tree.delete(row)
        self.forge_champion_label.config(text="Running...", fg=TEXT_DIM)
        self._forge_cancel_event.clear()
        self.stop_forge_btn.config(state="normal")
        self.forge_progress.start(10)
        if self.forge_loop_mode.get():
            threading.Thread(target=self._forge_run_loop_pipeline, daemon=True).start()
        else:
            threading.Thread(target=self._forge_run_pipeline, daemon=True).start()

    def _forge_stop_clicked(self):
        self._forge_cancel_event.set()
        self.stop_forge_btn.config(state="disabled")
        self._log_forge("\nStopping... (finishing the current candidate, no new stages will start)")

    def _forge_run_pipeline(self):
        try:
            df = self.forge_context.load_dataframe(self._log_forge)
            if df is None:
                return
            risk = self.forge_context.build_risk_config()
            rules = self.forge_context.build_prop_rules()

            config = ForgeConfig(
                n_hypotheses=self.forge_n_hypotheses.get_int(10_000),
                seed=self.forge_seed.get_int(42),
                min_trades=self.forge_min_trades.get_int(20),
                min_profit_factor=self.forge_min_pf.get_float(1.05),
                stage1_top_n=self.forge_stage1_top_n.get_int(1000),
                ga_population=self.forge_ga_population.get_int(12),
                ga_generations=self.forge_ga_generations.get_int(4),
                stage2_top_n=self.forge_stage2_top_n.get_int(200),
                stage3_mc_sims=self.forge_stage3_mc_sims.get_int(2000),
                cpcv_pool_size=self.forge_cpcv_pool_size.get_int(30),
                cpcv_survivors=self.forge_cpcv_survivors.get_int(10),
                final_mc_sims=self.forge_final_mc_sims.get_int(10_000),
                mc_survivors=self.forge_mc_survivors.get_int(5),
                eval_window_days=self.forge_eval_window_days.get_int(60),
                rolling_survivors=self.forge_rolling_survivors.get_int(2),
                locked_holdout_frac=self.forge_locked_holdout_frac.get_float(0.15),
                graveyard_skip_known_dead=self.forge_graveyard_skip.var.get(),
                graveyard_min_attempts=self.forge_graveyard_min_attempts.get_int(8),
                workers=int(self.forge_workers.get_str().strip()) if self.forge_workers.get_str().strip() else None,
                random_seed=self.forge_seed.get_int(42),
                reset_on_breach=True,
            )
            instrument = (
                os.path.basename(self.forge_context.csv_paths[0]) if len(self.forge_context.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.forge_context.csv_paths)
            )
            # Same persistent, instrument-scoped path the web app uses (see
            # app.search.graveyard.graveyard_path_for) -- a rejection from a
            # desktop run and a web run against the same instrument/
            # timeframe land in the SAME file, so the graveyard-feedback
            # loop benefits from every run against this data regardless of
            # which UI produced it. timeframe is the real inferred value
            # (app.web.server's own Forge route computes the same thing
            # from the same df), not the literal string "unknown".
            timeframe_label = _infer_timeframe_label(df)
            graveyard_path = graveyard_path_for(instrument, timeframe_label)
            self._last_forge_graveyard_path = graveyard_path
            db_path = str(OUTPUT_DIR / "forge" / "forge.db")

            result = run_forge(
                df, risk, rules, config, db_path=db_path,
                instrument=instrument, timeframe=timeframe_label, graveyard_path=graveyard_path,
                progress_cb=self._log_forge, cancel_event=self._forge_cancel_event,
            )
            self._last_forge_result = result

            def _paint():
                for stage in result.funnel:
                    self.forge_funnel_tree.insert("", END, values=(stage.name, stage.n_in, stage.n_out))
                for row in result.leaderboard:
                    self.forge_leaderboard_tree.insert("", END, values=(
                        row.candidate_id, row.family_label, f"{row.pass_rate_pct:.1f}",
                        f"{row.payout_rate_pct:.1f}", f"{row.prop_survival_score:.1f}",
                        row.locked_oos_status, f"{row.max_drawdown_pct:.2f}",
                        f"${row.net_profit:,.2f}", row.total_trades,
                    ))
                if result.champion_candidate_id:
                    champ = next((r for r in result.leaderboard if r.candidate_id == result.champion_candidate_id), None)
                    if champ:
                        self.forge_champion_label.config(
                            text=(
                                f"{champ.candidate_id}  ({champ.family_label})\n"
                                f"Prop survival score: {champ.prop_survival_score:.1f}   "
                                f"Locked OOS: {champ.locked_oos_status} ({champ.locked_oos_pass_rate_pct or 0:.0f}% pass)   "
                                f"Rolling pass rate: {champ.pass_rate_pct:.1f}%   Payout rate: {champ.payout_rate_pct:.1f}%"
                            ),
                            fg=GREEN,
                        )
                else:
                    self.forge_champion_label.config(
                        text="No candidate reached a PASSED locked-OOS holdout this run -- see the log above "
                             "for exactly why each stage cut what it cut, and check the Strategy Graveyard for "
                             "accumulated dead-neighborhood history on this instrument.",
                        fg=AMBER,
                    )
            self.root.after(0, _paint)

            self._log_forge(f"\nDone. {len(result.leaderboard)} strategy(ies) reached the leaderboard.")
            self._log_forge(f"Strategy Graveyard for this instrument: {graveyard_path}")
            try:
                self._refresh_dashboard()
            except Exception:
                pass

        except SearchCancelled:
            self._log_forge(
                "\nForge Strategy stopped. Whatever reached the results DB / graveyard before the stop "
                "is kept -- no leaderboard is finalized for this incomplete run."
            )
        except Exception as exc:
            self._log_forge("\nUnexpected error:\n" + traceback.format_exc())
            log_crash("Forge Strategy (desktop)", exc=exc)
        finally:
            self.forge_progress.stop()
            self.stop_forge_btn.config(state="disabled")
            self._release_heavy_job(JOB_FORGE)

    def _forge_run_loop_pipeline(self):
        """Loop Mode's desktop counterpart to _forge_run_pipeline above --
        same data-loading/config/risk/rules setup, but drives
        app.orchestration.loop_runner.run_forge_loop (repeated rounds,
        searching deeper on stall) instead of a single run_forge call.
        Repaints the funnel/leaderboard/champion display after EVERY
        round (via on_round), so the tab shows live progress across a
        loop that can run for a long time, not just a frozen "Running..."
        state until the whole thing finishes."""
        try:
            df = self.forge_context.load_dataframe(self._log_forge)
            if df is None:
                return
            risk = self.forge_context.build_risk_config()
            rules = self.forge_context.build_prop_rules()

            base_config = ForgeConfig(
                min_trades=self.forge_min_trades.get_int(20),
                min_profit_factor=self.forge_min_pf.get_float(1.05),
                ga_population=self.forge_ga_population.get_int(12),
                ga_generations=self.forge_ga_generations.get_int(4),
                stage3_mc_sims=self.forge_stage3_mc_sims.get_int(2000),
                cpcv_pool_size=self.forge_cpcv_pool_size.get_int(30),
                cpcv_survivors=self.forge_cpcv_survivors.get_int(10),
                final_mc_sims=self.forge_final_mc_sims.get_int(10_000),
                mc_survivors=self.forge_mc_survivors.get_int(5),
                eval_window_days=self.forge_eval_window_days.get_int(60),
                rolling_survivors=self.forge_rolling_survivors.get_int(2),
                locked_holdout_frac=self.forge_locked_holdout_frac.get_float(0.15),
                graveyard_skip_known_dead=self.forge_graveyard_skip.var.get(),
                graveyard_min_attempts=self.forge_graveyard_min_attempts.get_int(8),
                workers=int(self.forge_workers.get_str().strip()) if self.forge_workers.get_str().strip() else None,
                reset_on_breach=True,
            )
            instrument = (
                os.path.basename(self.forge_context.csv_paths[0]) if len(self.forge_context.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.forge_context.csv_paths)
            )
            graveyard_path = graveyard_path_for(instrument, _infer_timeframe_label(df))
            self._last_forge_graveyard_path = graveyard_path

            time_budget_raw = self.forge_loop_time_budget_hours.get_str().strip()
            loop_cfg = ForgeLoopConfig(
                target_pass_rate_pct=self.forge_loop_target.get_float(60.0),
                require_locked_oos_passed=self.forge_loop_require_locked_oos.get(),
                max_rounds=self.forge_loop_max_rounds.get_int(10),
                time_budget_seconds=(float(time_budget_raw) * 3600.0) if time_budget_raw else None,
                stall_rounds_before_widen=self.forge_loop_stall_rounds.get_int(2),
                starting_n_hypotheses=self.forge_n_hypotheses.get_int(10_000),
                starting_stage1_top_n=self.forge_stage1_top_n.get_int(1000),
                starting_stage2_top_n=self.forge_stage2_top_n.get_int(200),
                seed=self.forge_seed.get_int(42),
                base_config=base_config,
            )
            self._log_forge(
                f"Loop mode ON -- will repeat Forge rounds (searching deeper on stall) until a "
                f"champion clears {loop_cfg.target_pass_rate_pct:.0f}%, {loop_cfg.max_rounds} "
                f"rounds run, or the time budget is used up."
            )

            def on_round(round_result) -> None:
                result = round_result.result

                def _paint():
                    for tree in (self.forge_funnel_tree, self.forge_leaderboard_tree):
                        for row in tree.get_children():
                            tree.delete(row)
                    for stage in result.funnel:
                        self.forge_funnel_tree.insert("", END, values=(stage.name, stage.n_in, stage.n_out))
                    for row in result.leaderboard:
                        self.forge_leaderboard_tree.insert("", END, values=(
                            row.candidate_id, row.family_label, f"{row.pass_rate_pct:.1f}",
                            f"{row.payout_rate_pct:.1f}", f"{row.prop_survival_score:.1f}",
                            row.locked_oos_status, f"{row.max_drawdown_pct:.2f}",
                            f"${row.net_profit:,.2f}", row.total_trades,
                        ))
                    self.forge_champion_label.config(
                        text=(
                            f"Loop round {round_result.round_index} -- best champion pass rate so far: "
                            f"{round_result.champion_pass_rate_pct:.1f}%"
                            if round_result.champion_pass_rate_pct is not None else
                            f"Loop round {round_result.round_index} -- no candidate reached a PASSED "
                            "locked-OOS holdout yet."
                        ),
                        fg=GREEN if round_result.champion_pass_rate_pct is not None else AMBER,
                    )
                self.root.after(0, _paint)
                self._last_forge_result = result

            # timeframe is now the real inferred value -- app.web.server's
            # own Forge Loop call was updated to compute the same value
            # from the same df, so the two stay pointed at the same
            # graveyard_path_for(instrument, timeframe) file.
            result = run_forge_loop(
                df, risk, rules, db_dir=str(OUTPUT_DIR / "forge" / "loop"), loop_cfg=loop_cfg,
                instrument=instrument, timeframe=_infer_timeframe_label(df),
                progress_cb=self._log_forge, cancel_event=self._forge_cancel_event, on_round=on_round,
            )

            if result.stopped_reason == "error":
                self._log_forge(f"\nLoop Mode stopped: a round crashed -- {result.error}")
                log_crash("Forge Strategy Loop Mode (desktop)", exc=RuntimeError(result.error or "unknown"))
                return
            if result.stopped_reason == "cancelled":
                self._log_forge(
                    "\nLoop Mode stopped by user. The latest completed round's results are still "
                    "shown above."
                )
                return

            reason_text = {
                "target_reached": f"Target reached ({loop_cfg.target_pass_rate_pct:.0f}%) -- stopped automatically.",
                "max_rounds": f"Stopped: all {loop_cfg.max_rounds} rounds completed with no champion clearing the target.",
                "time_budget": "Stopped: time budget used up.",
            }.get(result.stopped_reason, result.stopped_reason)
            self._log_forge(f"\n{reason_text}")
            self._log_forge(f"Strategy Graveyard for this instrument: {graveyard_path}")
            if result.winner_round:
                self._log_forge(f"Winning candidate: {result.winner_candidate_id}")
            try:
                self._refresh_dashboard()
            except Exception:
                pass

        except SearchCancelled:
            self._log_forge(
                "\nForge Strategy Loop Mode stopped. Whatever reached the results DB / graveyard "
                "before the stop is kept."
            )
        except Exception as exc:
            self._log_forge("\nUnexpected error:\n" + traceback.format_exc())
            log_crash("Forge Strategy Loop Mode (desktop)", exc=exc)
        finally:
            self.forge_progress.stop()
            self.stop_forge_btn.config(state="disabled")
            self._release_heavy_job(JOB_FORGE)

    def _open_forge_graveyard(self):
        """Opens the Strategy Graveyard for whichever instrument this tab's
        Forge run most recently used, falling back to whatever's most
        recently modified if no run has happened yet this session -- same
        underlying files the web app's /graveyard page reads (persistent,
        instrument-scoped; see app.search.graveyard)."""
        path = self._last_forge_graveyard_path
        if path is None:
            files = list_graveyard_files()
            if not files:
                messagebox.showinfo(
                    "No graveyard yet",
                    "No Strategy Graveyard file exists yet -- run Forge Strategy at least once and "
                    "rejections that reach CPCV/stress-testing will start accumulating here.",
                )
                return
            path = files[0]["path"]
        rows = load_graveyard(path)
        clusters = summarize_graveyard(rows, top_n=50)
        win = Toplevel(self.root)
        win.title(f"Strategy Graveyard -- {Path(path).stem}")
        win.configure(bg=BG)
        win.geometry("900x500")
        text = Text(win, bg=LOG_BG, fg=TEXT, font=(MONO, 9), wrap="word")
        text.pack(fill="both", expand=True, padx=10, pady=10)
        if not clusters:
            text.insert(END, f"({path})\n\nNo rejections recorded yet for this instrument.")
        else:
            header = f"{'Family':<28}{'Tested':>8}{'Worst stage':>14}{'Best pass %':>12}   Reason\n"
            text.insert(END, f"({path})\n\n" + header + "-" * len(header) + "\n")
            for c in clusters:
                best = f"{c.best_prop_sim_pass_pct:.0f}%" if c.best_prop_sim_pass_pct is not None else "--"
                text.insert(
                    END,
                    f"{c.family:<28}{c.n_tested:>8}{c.worst_stage_reached:>14}{best:>12}   {c.most_common_reason}\n",
                )
        text.config(state="disabled")
