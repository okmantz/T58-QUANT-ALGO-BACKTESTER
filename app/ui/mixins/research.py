"""Research Director, Research Agent and Research Loop tabs."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ai.ollama_settings import OllamaSettings

import app.ai.experiment_memory as experiment_memory_module
import app.ai.research_library as research_library_module
import json
import os
import threading
import traceback
from app.ai.research_agent import ResearchAgent, ResearchAgentContext
from app.ai.research_loop import ResearchLoopConfig, run_research_loop
from app.reports.crash_log import log_crash
from app.research import director as research_director
from app.strategy.base import StrategyError
from app.strategy.library import list_saved_strategies, load_strategy_text
from pathlib import Path
from tkinter import END, Frame, Label, Text, filedialog, messagebox, ttk
from app.ui.common import _safe_font, _strategy_display_name
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
NEON_LIME = ""
theme.bind(globals())


class ResearchMixin:
    """Research Director, Research Agent and Research Loop tabs."""


    # ------------------------------------------------------------------
    # Research Director -- desktop parity for the web app's /research.
    # Same six analyses (edge decomposition, ablation, null-strategy
    # benchmarks, signal degradation, trade contribution, conditional
    # expectancy, regime discovery) run against ONE already-saved Manual
    # Strategy Builder strategy and the currently loaded data -- this is
    # explanatory ("why does/doesn't this edge work"), not a search, so
    # it's fast enough to run synchronously in a background thread with
    # no Stop button or HEAVY_JOB_GUARD, matching the web route's own
    # single-request shape.
    # ------------------------------------------------------------------

    def _build_research_director_tab(self):
        f = self._scrollable(self.tab_research_director)

        self._page_header(
            f,
            "RESEARCH DIRECTOR",
            "Research Director",
            "Runs a battery of falsification-style tests against ONE saved Manual Strategy Builder "
            "strategy to explain WHY it does or doesn't have a real edge: edge decomposition (which "
            "individual conditions actually contribute), ablation testing (what happens if each "
            "condition is removed), null-strategy benchmarks (random / coin-flip / buy-and-hold / "
            "session-only / breakout / mean-reversion / previous-bar baselines on the SAME data), "
            "signal degradation (execution-fragility stress), trade contribution + conditional "
            "expectancy analysis, and regime discovery. Its own market data, prop rules, and "
            "risk settings below are self-contained to this tab, just like the web app.",
        )

        self.resdir_context = RunContextPanel(self, "Research Director")
        self.resdir_context.build(f)

        strategy_section = self._section(
            f, "Strategy to analyze",
            "Only Manual Strategy Builder strategies (saved JSON configs) -- Research Director's "
            "analyses (ablation, null baselines, etc.) need to inspect and rebuild individual "
            "conditions, which code strategies don't expose the same way.",
        )
        self._rd_strategy_names = [s.name for s in list_saved_strategies("manual")]
        self.rd_strategy = LabeledCombo(
            strategy_section, "Saved Manual strategy",
            self._rd_strategy_names or ["(none saved yet)"],
            self._rd_strategy_names[0] if self._rd_strategy_names else "(none saved yet)",
        )
        self._button(strategy_section, "REFRESH LIST", self._rd_refresh_strategy_list).pack(anchor="w", padx=18, pady=(0, 10))

        window_section = self._section(
            f, "Settings",
            "window_trading_days controls the rolling-evaluation window used inside these analyses; "
            "regime discovery reserves the final 15% of the loaded data as its own holdout slice.",
        )
        self.rd_window_days = LabeledEntry(window_section, "Window trading days", 30)

        checks_section = self._section(f, "Which analyses to run", "All six are independent -- uncheck any you don't need for a faster run.")
        self.rd_run_decomposition = LabeledCheckbox(checks_section, "Edge Decomposition", True)
        self.rd_run_ablation = LabeledCheckbox(checks_section, "Ablation Testing", True)
        self.rd_run_null = LabeledCheckbox(checks_section, "Null Strategy Benchmarks", True)
        self.rd_run_degradation = LabeledCheckbox(checks_section, "Execution Fragility / Signal Degradation", True)
        self.rd_run_contribution = LabeledCheckbox(checks_section, "Trade Contribution Analysis", True)
        self.rd_run_conditional = LabeledCheckbox(checks_section, "Conditional Expectancy Maps", True)
        self.rd_run_regime = LabeledCheckbox(checks_section, "Regime Discovery", True)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN RESEARCH DIRECTOR", self._rd_run_clicked, primary=True).pack(side="left")

        self.rd_progress = NeuralProgress(f)
        self.rd_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Research Director output", "Full results, same structure as the web app's report.")
        _rd_output_frame = Frame(output_section, bg=PANEL)
        self.rd_output = Text(
            _rd_output_frame, height=24, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _rd_output_scroll = ttk.Scrollbar(
            _rd_output_frame, orient="vertical", command=self.rd_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.rd_output.configure(yscrollcommand=_rd_output_scroll.set)
        self.rd_output.pack(side="left", fill="both", expand=True)
        _rd_output_scroll.pack(side="right", fill="y")
        _rd_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.rd_output)

    def _rd_refresh_strategy_list(self):
        self._rd_strategy_names = [s.name for s in list_saved_strategies("manual")]
        self.rd_strategy.combo.config(values=self._rd_strategy_names or ["(none saved yet)"])
        if self._rd_strategy_names:
            self.rd_strategy.var.set(self._rd_strategy_names[0])

    def _rd_log(self, msg: str) -> None:
        def _do():
            try:
                self.rd_output.insert(END, msg + "\n")
                self.rd_output.see(END)
            except Exception:
                pass
        try:
            self.root.after(0, _do)
        except Exception:
            pass

    def _rd_run_clicked(self):
        strategy_name = self.rd_strategy.get_str().strip()
        if not strategy_name or strategy_name == "(none saved yet)":
            messagebox.showwarning(
                "No strategy selected",
                "Save a strategy in Manual Strategy Builder first, then pick it here.",
            )
            return
        if not self.resdir_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 2 (Market Data).")
            return
        self.rd_output.delete("1.0", END)
        self.rd_progress.start(10)
        threading.Thread(target=self._rd_run_pipeline, args=(strategy_name,), daemon=True).start()

    def _rd_run_pipeline(self, strategy_name: str):
        try:
            df = self.resdir_context.load_dataframe(self._rd_log)
            if df is None:
                return
            try:
                spec = {"source_type": "manual", "config": json.loads(load_strategy_text("manual", strategy_name))}
            except Exception as exc:
                self._rd_log(f"Couldn't load strategy '{strategy_name}': {exc}")
                return

            risk = self.resdir_context.build_risk_config()
            rules = self.resdir_context.build_prop_rules()
            window_trading_days = self.rd_window_days.get_int(30)

            full_bt = research_director._run_spec(spec, df, risk)
            if full_bt is None or not full_bt.trades:
                self._rd_log(
                    "This strategy produced zero trades on the selected dataset with the current "
                    "risk/prop settings -- nothing to analyze. Check the strategy's conditions and "
                    "the pip size on Step 4 (Risk & Execution)."
                )
                return
            target_row = research_director._row(strategy_name, full_bt, rules, window_trading_days)
            self._rd_log(f"Base backtest: {len(full_bt.trades)} trades, net ${full_bt.statistics.net_profit:,.2f}.\n")

            if self.rd_run_decomposition.var.get():
                self._rd_log("=== Edge Decomposition ===")
                try:
                    result = research_director.edge_decomposition(spec, df, risk, rules, window_trading_days)
                    for step in result["steps"]:
                        self._rd_log(f"  {step.get('label', '?')}: {step}")
                    self._rd_log(f"Verdict: {result.get('verdict')}\n")
                except Exception as exc:
                    self._rd_log(f"  Couldn't run: {exc}\n")

            if self.rd_run_ablation.var.get():
                self._rd_log("=== Ablation Testing ===")
                try:
                    result = research_director.ablation_test(spec, df, risk, rules, window_trading_days)
                    for row in result["rows"]:
                        self._rd_log(f"  {row}")
                    self._rd_log(f"Verdict: {result.get('verdict')}\n")
                except Exception as exc:
                    self._rd_log(f"  Couldn't run: {exc}\n")

            if self.rd_run_null.var.get():
                self._rd_log("=== Null Strategy Benchmarks ===")
                try:
                    result = research_director.null_baselines(df, risk, rules, target_row, window_trading_days)
                    for b in result["baselines"]:
                        self._rd_log(f"  {b}")
                    self._rd_log(f"Verdict: {result.get('verdict')}\n")
                except Exception as exc:
                    self._rd_log(f"  Couldn't run: {exc}\n")

            if self.rd_run_degradation.var.get():
                self._rd_log("=== Execution Fragility / Signal Degradation ===")
                try:
                    result = research_director.signal_degradation(spec, df, risk, rules, window_trading_days)
                    for t in result["stress_tests"]:
                        self._rd_log(f"  {t}")
                    self._rd_log(f"Verdict: {result.get('verdict')}\n")
                except Exception as exc:
                    self._rd_log(f"  Couldn't run: {exc}\n")

            if self.rd_run_contribution.var.get():
                self._rd_log("=== Trade Contribution Analysis ===")
                result = research_director.trade_contribution(full_bt.trades, risk.initial_balance)
                self._rd_log(f"  {result}\n")

            if self.rd_run_conditional.var.get():
                self._rd_log("=== Conditional Expectancy Maps ===")
                result = research_director.conditional_expectancy(full_bt.trades, df)
                self._rd_log(f"  {result}\n")

            if self.rd_run_regime.var.get():
                self._rd_log("=== Regime Discovery ===")
                try:
                    n = len(df)
                    holdout_df = df.iloc[int(n * 0.85):].reset_index(drop=True)
                    result = research_director.regime_discovery(
                        full_bt.trades, df, risk, rules, spec, holdout_df, window_trading_days,
                    )
                    self._rd_log(f"  Hypothesis: {result.get('hypothesis')}")
                    self._rd_log(f"  Next step: {result.get('next_step')}\n")
                except Exception as exc:
                    self._rd_log(f"  Couldn't run: {exc}\n")

            self._rd_log("Done.")
        except Exception as exc:
            self._rd_log("\nUnexpected error:\n" + traceback.format_exc())
            log_crash("Research Director (desktop)", exc=exc)
        finally:
            self.rd_progress.stop()

    # -----------------------------------------------------------------------
    # Tab 16 — AI Research Agent (T58 AI Research Engine)
    #
    # The "research analyst" upgrade to AI Assist: instead of a single
    # request/response call (numeric parameter suggestions, or a one-shot
    # generated strategy file), this hands a local Ollama model a fixed
    # toolbox of READ-ONLY analysis actions -- backtest, prop-simulation,
    # Monte Carlo, walk-forward, regime analysis, parameter sensitivity,
    # cost stress, plus the research/ paper library and T58's own memory
    # of every past experiment -- and lets it reason across several steps
    # before answering. See app.ai.research_agent's module docstring for
    # the full safety rationale: every tool call runs this app's OWN
    # already-validated engine, so the model can propose a next step or a
    # diagnosis, but it can never invent a number or write/change code.
    # Uses the strategy, data, prop rules, and risk settings already
    # configured in Steps 01-04, exactly like 15 Full Pipeline does.
    # -----------------------------------------------------------------------

    def _build_research_agent_tab(self):
        f = self._scrollable(self.tab_researchagent)
        self._page_header(
            f,
            "CREATE / Research Agent",
            "AI Research Agent",
            "Ask a local Ollama model to investigate the strategy configured in Steps 01-04. "
            "It can call run_backtest, run_prop_simulation, run_monte_carlo, run_walk_forward, "
            "run_regime_analysis, run_parameter_sensitivity, run_cost_stress, compare_strategies, "
            "search_research (your research/ paper library), and search_experiments (T58's own "
            "memory of every past strategy test) -- each one runs this app's real, already-"
            "validated engine, never a guess. The model can recommend a next step (e.g. a "
            "parameter worth testing), but cannot apply it itself -- take any recommendation "
            "into 09 Refinement / Quick Optimize / 15 Full Pipeline to actually test it. "
            "Its own market data, prop rules, and risk settings below are self-contained "
            "to this tab (shared by the Closed Research Loop section further down), just "
            "like the web app.",
        )

        self.resagent_context = RunContextPanel(self, "Research Agent")
        self.resagent_context.build(f)

        question_section = self._section(
            f, "Research question",
            "What do you want the agent to investigate? Be specific -- \"Is this strategy "
            "robust enough for a 50K prop account?\" gets a more useful investigation than "
            "\"is this good\".",
            emphasize=True,
        )
        _ra_question_frame = Frame(question_section, bg=PANEL)
        self.ra_question = Text(
            _ra_question_frame, height=4, wrap="word", bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        _ra_question_scroll = ttk.Scrollbar(
            _ra_question_frame, orient="vertical", command=self.ra_question.yview, style="T58.Vertical.TScrollbar",
        )
        self.ra_question.configure(yscrollcommand=_ra_question_scroll.set)
        self.ra_question.pack(side="left", fill="both", expand=True)
        _ra_question_scroll.pack(side="right", fill="y")
        self.ra_question.insert(
            "1.0",
            "Is this strategy robust, or does its edge depend on a fragile parameter or a "
            "specific market regime? What's the most promising next thing to test?",
        )
        _ra_question_frame.pack(fill="x", padx=18, pady=(4, 12))
        self._bind_isolated_wheel(self.ra_question)
        self.ra_max_steps = LabeledEntry(question_section, "Max tool-calling steps", 6)

        reports_section = self._section(
            f, "Optional: upload backtest reports / screenshots",
            "HTML/CSV/PDF reports (from this app or elsewhere) are parsed for real metrics. "
            "Screenshots (.png/.jpg) are read by the vision model below on a best-effort basis and "
            "flagged as unverified. The agent can call read_uploaded_report on any of these while "
            "investigating your question.",
        )
        self._ra_uploaded_reports: list[str] = []
        ra_reports_btn_row = Frame(reports_section, bg=PANEL)
        ra_reports_btn_row.pack(anchor="w", padx=18, pady=(2, 4))
        self._button(ra_reports_btn_row, "IMPORT REPORT / SCREENSHOT", self._ra_import_report_clicked).pack(side="left")
        self._button(ra_reports_btn_row, "CLEAR", self._ra_clear_reports_clicked).pack(side="left", padx=8)
        self.ra_reports_status = Label(
            reports_section, text="No files attached.", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
            wraplength=900, justify="left",
        )
        self.ra_reports_status.pack(anchor="w", padx=18, pady=(0, 12))

        self._build_ai_assist_section(f, prefix="ra_ai")

        memory_section = self._section(
            f, "Research Library + T58 Research Memory",
            "The research/ paper library (RAG) and the durable record of every strategy this "
            "app has tested. Embedding the library is optional (needs an Ollama embedding "
            "model, e.g. `ollama pull nomic-embed-text`) -- without it, search_research still "
            "works via plain keyword matching.",
        )
        btn_row = Frame(memory_section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(2, 6))
        self._button(btn_row, "EMBED RESEARCH LIBRARY", self._ra_embed_library_clicked, primary=True).pack(side="left")
        self._button(btn_row, "REFRESH MEMORY SUMMARY", self._ra_refresh_memory_clicked).pack(side="left", padx=8)
        self._button(btn_row, "VIEW LEADERBOARD", self._ra_view_leaderboard_clicked).pack(side="left", padx=8)
        self.ra_memory_status = Label(
            memory_section, text="Click REFRESH MEMORY SUMMARY to see how many experiments T58 has recorded.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=900, justify="left",
        )
        self.ra_memory_status.pack(anchor="w", padx=18, pady=(0, 12))

        loop_pointer_section = self._section(
            f, "Closed Research Loop (Ollama)",
            "Moved to its own tab -- CREATE / Research Loop (Background) -- so it can run as an "
            "independent, self-contained background companion (own market data, own AI Assist "
            "settings) instead of sharing this tab's setup.",
        )
        self._button(
            loop_pointer_section, "OPEN RESEARCH LOOP (BACKGROUND)",
            lambda: self._show_page("researchloop"),
        ).pack(anchor="w", padx=18, pady=(2, 12))

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self.ra_run_btn = self._button(button_row, "RUN RESEARCH AGENT", self._ra_run_clicked, primary=True)
        self.ra_run_btn.pack(side="left")

        self.ra_progress = NeuralProgress(f)
        self.ra_progress.pack(fill="x", padx=24, pady=(2, 10))

        answer_section = self._section(f, "Final Answer", "Filled in once the agent finishes.")
        self.ra_answer_label = Label(
            answer_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=900, anchor="w",
        )
        self.ra_answer_label.pack(anchor="w", fill="x", padx=18, pady=(2, 12))

        output_section = self._section(f, "Agent transcript", "Thought / Action / Observation, one step at a time.")
        _ra_output_frame = Frame(output_section, bg=PANEL)
        self.ra_output = Text(
            _ra_output_frame, height=22, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _ra_output_scroll = ttk.Scrollbar(
            _ra_output_frame, orient="vertical", command=self.ra_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.ra_output.configure(yscrollcommand=_ra_output_scroll.set)
        self.ra_output.pack(side="left", fill="both", expand=True)
        _ra_output_scroll.pack(side="right", fill="y")
        _ra_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.ra_output)

    # -----------------------------------------------------------------------
    # Research Loop (Background) -- split out from Research Agent into its
    # own tab (Sep 2026 CREATE reorder) so it's a fully independent,
    # self-contained background companion: own market data / prop rules /
    # risk settings (own RunContextPanel) and its own AI Assist (Ollama)
    # settings, matching the web app's standalone /research-loop page
    # instead of quietly sharing Research Agent's setup.
    # -----------------------------------------------------------------------

    def _build_research_loop_tab(self):
        f = self._scrollable(self.tab_research_loop)
        self._page_header(
            f,
            "CREATE / Research Loop (Background)",
            "\u21bb Closed Research Loop (Ollama)",
            "The full 'Research Papers -> Ollama -> hypothesis -> generated strategy -> backtest -> "
            "Monte Carlo -> prop-firm survival simulation -> analyze WHY it failed -> Ollama proposes "
            "an improved hypothesis -> repeat' loop from the AI Research Engine plan -- see "
            "app.ai.research_loop. Every KEEP/DISCARD verdict comes from the real Prop Survival Score "
            "(app.prop.survival_engine), never Ollama's own judgment; the failure diagnosis each round "
            "is a computed number (e.g. '73% of losses happened in low-volatility regimes'), not a "
            "guess. Every round is recorded into T58 Research Memory, and this loop refuses to "
            "re-run a strategy whose Strategy DNA exactly matches a pattern that already failed earlier "
            "in the SAME run. Its own market data, prop rules, risk settings, and AI Assist settings "
            "below are self-contained to this tab -- leave iterations blank/large and let it run as a "
            "background companion alongside whatever else you're doing.",
        )

        self.researchloop_context = RunContextPanel(self, "Research Loop")
        self.researchloop_context.build(f)

        self._build_ai_assist_section(f, prefix="loop_ai")

        loop_section = self._section(
            f, "Loop settings",
            "Leave iterations high (or rely on STOP) to let this run for a long time as a background "
            "companion -- e.g. alongside a Multi-Instrument Speed Run.",
            emphasize=True,
        )
        _loop_initial_idea_frame = Frame(loop_section, bg=PANEL)
        self.loop_initial_idea = Text(
            _loop_initial_idea_frame, height=3, wrap="word", bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        _loop_initial_idea_scroll = ttk.Scrollbar(
            _loop_initial_idea_frame, orient="vertical", command=self.loop_initial_idea.yview, style="T58.Vertical.TScrollbar",
        )
        self.loop_initial_idea.configure(yscrollcommand=_loop_initial_idea_scroll.set)
        self.loop_initial_idea.pack(side="left", fill="both", expand=True)
        _loop_initial_idea_scroll.pack(side="right", fill="y")
        Label(loop_section, text="Starting hypothesis", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9)).pack(
            anchor="w", padx=18, pady=(2, 0),
        )
        self.loop_initial_idea.insert(
            "1.0", "A trend-following strategy using a higher-timeframe bias filter and a volatility-based stop.",
        )
        _loop_initial_idea_frame.pack(fill="x", padx=18, pady=(2, 10))
        self._bind_isolated_wheel(self.loop_initial_idea)
        self.loop_iterations = LabeledEntry(loop_section, "Iterations", 5)
        self.loop_keep_threshold = LabeledEntry(loop_section, "Prop Survival Score needed to KEEP", 40)
        self.loop_mc_sims = LabeledEntry(loop_section, "Monte Carlo sims per iteration", 500)
        self.loop_survival_sims = LabeledEntry(loop_section, "Survival-simulation sims per iteration", 1000)

        loop_btn_row = Frame(f, bg=BG)
        loop_btn_row.pack(fill="x", padx=24, pady=10)
        self._button(loop_btn_row, "RUN RESEARCH LOOP", self._research_loop_run_clicked, primary=True).pack(side="left")

        self.loop_progress = NeuralProgress(f)
        self.loop_progress.pack(fill="x", padx=24, pady=(2, 10))

        loop_summary_section = self._section(f, "Best Iteration", "Filled in once the loop finishes.")
        self.loop_summary_label = Label(
            loop_summary_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=900, anchor="w",
        )
        self.loop_summary_label.pack(anchor="w", fill="x", padx=18, pady=(2, 12))

        loop_output_section = self._section(f, "Research Loop log", "Iteration-by-iteration progress.")
        _loop_output_frame = Frame(loop_output_section, bg=PANEL)
        self.loop_output = Text(
            _loop_output_frame, height=22, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _loop_output_scroll = ttk.Scrollbar(
            _loop_output_frame, orient="vertical", command=self.loop_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.loop_output.configure(yscrollcommand=_loop_output_scroll.set)
        self.loop_output.pack(side="left", fill="both", expand=True)
        _loop_output_scroll.pack(side="right", fill="y")
        _loop_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.loop_output)

    def _log_research_loop(self, msg: str):
        self.loop_output.insert(END, msg + "\n")
        self.loop_output.see(END)
        self.root.update_idletasks()

    def _research_loop_run_clicked(self):
        if not self.researchloop_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        self.loop_output.delete("1.0", END)
        self.loop_summary_label.config(text="Running...", fg=TEXT_DIM)
        self.loop_progress.start(10)
        threading.Thread(target=self._research_loop_run_pipeline, daemon=True).start()

    def _research_loop_run_pipeline(self):
        try:
            df = self.researchloop_context.load_dataframe(self._log_research_loop)
            if df is None:
                return
            risk = self.researchloop_context.build_risk_config()
            rules = self.researchloop_context.build_prop_rules()
            settings = self._build_ollama_settings(prefix="loop_ai")
            if not settings.is_usable:
                self._log_research_loop(
                    "Ollama isn't enabled -- turn on AI Assist above (in this tab) with a valid host/model first."
                )
                return

            cfg = ResearchLoopConfig(
                n_iterations=self.loop_iterations.get_int(5),
                initial_idea=self.loop_initial_idea.get("1.0", END).strip(),
                mc_sims=self.loop_mc_sims.get_int(500),
                survival_sims=self.loop_survival_sims.get_int(1000),
                keep_score_threshold=self.loop_keep_threshold.get_float(40),
            )
            result = run_research_loop(df, risk, rules, settings, cfg, progress_cb=self._log_research_loop)

            self._log_research_loop(f"\nStopped: {result.stopped_reason} ({len(result.iterations)} iteration(s) ran)")
            if result.best_iteration:
                b = result.best_iteration
                self.loop_summary_label.config(
                    text=(
                        f"Best: iteration {b.iteration} ({b.strategy_name}) -- "
                        f"Prop Survival Score {b.prop_survival_score:.1f}/100, verdict {b.verdict}, "
                        f"{b.trades} trades, net profit ${b.net_profit:,.2f}"
                    ),
                    fg=NEON_LIME if b.verdict == "KEEP" else TEXT_DIM,
                )
            else:
                self.loop_summary_label.config(text="No iteration produced a usable result.", fg=TEXT_DIM)
        except Exception:
            self._log_research_loop("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.loop_progress.stop()

    def _log_research_agent(self, msg: str):
        self.ra_output.insert(END, msg + "\n")
        self.ra_output.see(END)
        self.root.update_idletasks()

    def _ra_embed_library_clicked(self):
        settings = self._build_ollama_settings(prefix="ra_ai")
        if not settings.is_usable:
            messagebox.showwarning(
                "Ollama not enabled",
                "Turn on 'Enable AI Assist for this run' above (and confirm TEST CONNECTION works) first.",
            )
            return
        self.ra_memory_status.config(text="Embedding research library...", fg=AMBER)
        self.root.update_idletasks()

        def run():
            try:
                stats = research_library_module.embed_index(settings)
                if stats.error:
                    text = f"Embedded {stats.chunks_embedded} new chunk(s); stopped early: {stats.error}"
                    color = AMBER
                else:
                    text = (
                        f"Research library ready: {stats.chunks_embedded} newly embedded, "
                        f"{stats.chunks_already_current} already current, {stats.total_chunks} total chunks."
                    )
                    color = GREEN
            except Exception as exc:
                text, color = f"Embedding failed: {exc}", RED

            def _finish():
                self.ra_memory_status.config(text=text, fg=color)

            try:
                self.root.after(0, _finish)
            except Exception:
                pass

        threading.Thread(target=run, daemon=True).start()

    def _ra_refresh_memory_clicked(self):
        try:
            counts = experiment_memory_module.get_summary_counts()
        except Exception as exc:
            self.ra_memory_status.config(text=f"Could not read Research Memory: {exc}", fg=RED)
            return
        if counts["total"] == 0:
            text = "No experiments recorded yet -- run 15 Full Pipeline, Quick Optimize, or a Batch Test first."
        else:
            breakdown = ", ".join(f"{v}: {n}" for v, n in counts["by_verdict"].items())
            text = f"T58 Research Memory: {counts['total']} experiments recorded. By verdict -- {breakdown}."
        self.ra_memory_status.config(text=text, fg=TEXT_DIM)

    def _ra_view_leaderboard_clicked(self):
        """Shows every recorded experiment (across Full Pipeline, Quick
        Optimize, Batch Test, and the Research Loop), ranked by
        evaluation-pass probability, each as a KEEP/DISCARD card with a
        DNA diff against its parent where one is recorded -- the
        'Experiment #1842 / Changes / Results / Compared with parent /
        Verdict' leaderboard from the AI Research Engine plan."""
        try:
            board = experiment_memory_module.get_leaderboard(limit=25)
        except Exception as exc:
            messagebox.showerror("Could not read Research Memory", str(exc))
            return
        if not board:
            messagebox.showinfo(
                "No experiments recorded yet",
                "Run 15 Full Pipeline, Quick Optimize, a Batch Test, or the Research Loop below first.",
            )
            return
        cards = []
        for exp in board:
            parent = None
            parent_id = exp.config.get("parent_experiment") if isinstance(exp.config, dict) else None
            if parent_id:
                parent = experiment_memory_module.get_experiment_by_id(parent_id)
            cards.append(experiment_memory_module.render_leaderboard_card(exp, parent=parent))
        body = f"Top {len(board)} experiments by evaluation-pass probability:\n\n" + \
            ("\n\n" + "-" * 60 + "\n\n").join(cards)
        self._show_text_viewer("T58 Research Memory -- Leaderboard", body)

    def _ra_import_report_clicked(self):
        paths = filedialog.askopenfilenames(
            title="Import backtest report(s) or screenshot(s)",
            filetypes=[
                ("Backtest reports & screenshots", "*.html *.htm *.csv *.pdf *.png *.jpg *.jpeg *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self._ra_uploaded_reports.extend(paths)
        names = ", ".join(os.path.basename(p) for p in self._ra_uploaded_reports)
        self.ra_reports_status.config(text=f"{len(self._ra_uploaded_reports)} file(s) attached: {names}")

    def _ra_clear_reports_clicked(self):
        self._ra_uploaded_reports = []
        self.ra_reports_status.config(text="No files attached.")

    def _ra_run_clicked(self):
        question = self.ra_question.get("1.0", END).strip()
        if not question:
            messagebox.showinfo("Ask a question first", "Type a research question in the box above.")
            return
        if not self.resagent_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        settings = self._build_ollama_settings(prefix="ra_ai")
        if not settings.is_usable:
            messagebox.showwarning(
                "Ollama not enabled",
                "Turn on 'Enable AI Assist for this run' above (and confirm TEST CONNECTION works) first.",
            )
            return
        self.ra_output.delete("1.0", END)
        self.ra_answer_label.config(text="Running...", fg=TEXT_DIM)
        self.ra_run_btn.config(state="disabled")
        self.ra_progress.start(10)
        threading.Thread(target=self._ra_run_agent, args=(question, settings), daemon=True).start()

    def _ra_run_agent(self, question: str, settings: "OllamaSettings"):
        try:
            df = self.resagent_context.load_dataframe(self._log_research_agent)
            if df is None:
                return
            risk = self.resagent_context.build_risk_config()
            rules = self.resagent_context.build_prop_rules()
            max_steps = self.ra_max_steps.get_int(6)
            instrument = (
                os.path.basename(self.resagent_context.csv_paths[0])
                if len(self.resagent_context.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.resagent_context.csv_paths)
            )

            # Zero-arg builder consistent with every other tab's "always
            # build fresh" convention (see app.search.robustness /
            # app.validation.regime_testing) -- rereads current Step 01
            # strategy config on every call, exactly like Full Pipeline's
            # own strategy_builder does.
            strategy_snapshot = self._build_strategy()
            uploaded_reports = [Path(p) for p in getattr(self, "_ra_uploaded_reports", [])]
            ctx = ResearchAgentContext(
                df=df, strategy_builder=self._build_strategy,
                strategy_name=_strategy_display_name(strategy_snapshot),
                source_type=strategy_snapshot.source_type,
                risk=risk, prop_rules=rules, instrument=instrument,
                uploaded_reports=uploaded_reports,
            )

            self._log_research_agent(f"Investigating '{ctx.strategy_name}' on {instrument}...\n")
            if uploaded_reports:
                self._log_research_agent(
                    f"Attached {len(uploaded_reports)} report/screenshot file(s): "
                    + ", ".join(p.name for p in uploaded_reports) + "\n"
                )
            agent = ResearchAgent(settings, max_steps=max_steps)
            result = agent.run(question, ctx, progress_cb=self._log_research_agent)

            for step in result.steps:
                self._log_research_agent(f"\n--- Step {step.step_index} ---")
                if step.thought:
                    self._log_research_agent(f"Thought: {step.thought}")
                if step.action:
                    self._log_research_agent(f"Action: {step.action}({json.dumps(step.action_input or {})})")
                if step.observation is not None:
                    self._log_research_agent(f"Observation: {json.dumps(step.observation, indent=2)[:2000]}")
                if step.note:
                    self._log_research_agent(f"Note: {step.note}")

            if result.final_answer:
                self.ra_answer_label.config(text=result.final_answer, fg=GREEN)
                self._log_research_agent(f"\nFinal Answer: {result.final_answer}")
            elif result.error:
                self.ra_answer_label.config(text=f"Stopped: {result.error}", fg=RED)
                self._log_research_agent(f"\nStopped: {result.error}")
            else:
                self.ra_answer_label.config(text=f"Stopped without a final answer: {result.stopped_reason}", fg=AMBER)
                self._log_research_agent(f"\nStopped without a final answer: {result.stopped_reason}")

            # Best-effort: record this investigation in T58 Research Memory too,
            # using whatever the agent's own run_backtest/run_monte_carlo tool
            # calls already computed (never a fresh computation just for this).
            try:
                bt = ctx.cache_get("__baseline_bt__")
                if bt is not None and bt.trades:
                    experiment_memory_module.record_experiment(
                        origin="research_agent", strategy_name=ctx.strategy_name,
                        source_type=ctx.source_type, instrument=instrument,
                        verdict="INVESTIGATED", trades=len(bt.trades),
                        net_profit=bt.statistics.net_profit, win_rate=bt.statistics.win_rate,
                        profit_factor=bt.statistics.profit_factor,
                        max_drawdown_pct=bt.statistics.max_drawdown_pct,
                        lesson=(result.final_answer or "")[:500], settings=settings,
                    )
            except Exception:
                pass
        except StrategyError as exc:
            self._log_research_agent(f"\nStrategy error: {exc}")
            self.ra_answer_label.config(text="Failed -- see log.", fg=RED)
        except Exception:
            self._log_research_agent("\nUnexpected error:\n" + traceback.format_exc())
            self.ra_answer_label.config(text="Failed -- see log.", fg=RED)
        finally:
            self.ra_progress.stop()
            self.ra_run_btn.config(state="normal")
