"""Graveyard, Your-Idea, leaderboard and Generate-Strategies tabs."""
from __future__ import annotations

import app.ai.strategy_generator as strategy_generator_module
import re
import threading
import traceback
from app.scoring.leaderboard import build_leaderboard, render_leaderboard_table
from app.search.graveyard import list_graveyard_files, load_graveyard, render_graveyard_report, summarize_graveyard
from app.strategy.library import STRATEGY_TYPES, save_strategy_metadata, save_strategy_text, set_strategy_status
from pathlib import Path
from tkinter import (
    BooleanVar,
    Checkbutton,
    END,
    EXTENDED,
    Entry,
    Frame,
    Label,
    Listbox,
    StringVar,
    Text,
    messagebox,
    ttk,
)
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCombo, LabeledEntry
from app.ui.widgets.cards import NeuralProgress
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


class DiscoveryTabsMixin:
    """Graveyard, Your-Idea, leaderboard and Generate-Strategies tabs."""


    # -----------------------------------------------------------------------
    # Strategy Graveyard -- desktop read-only view onto the same shared,
    # persistent, instrument-scoped graveyard files the web app's /graveyard
    # page reads (see app.search.graveyard.graveyard_path_for /
    # list_graveyard_files). A rejection recorded by Forge, Search Lab, or
    # Evolution Lab from EITHER the desktop app or the web app lands in the
    # same file, so this always shows the full picture regardless of which
    # UI produced any given run.
    # -----------------------------------------------------------------------
    def _build_graveyard_tab(self):
        f = self._scrollable(self.tab_graveyard)

        self._page_header(
            f,
            "STRATEGY GRAVEYARD",
            "Strategy Graveyard",
            "Dead neighborhoods -- parameter regions that reached full evaluation and CPCV but "
            "failed, most-tested first, so you don't waste hours re-discovering the same dead end. "
            "Shared with the web app: a rejection from Forge, Search Lab, or Evolution Lab on either "
            "UI shows up here.",
        )

        section = self._section(f, "Which file to browse", "", emphasize=True)
        self.gy_file_combo = LabeledCombo(section, "Instrument / timeframe", [], default="")
        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "REFRESH LIST", self._graveyard_refresh_list, primary=False).pack(side="left", padx=(0, 8))
        self._button(button_row, "SHOW GRAVEYARD", self._graveyard_show_clicked, primary=True).pack(side="left")

        output_section = self._section(f, "Result", "")
        _gy_output_frame = Frame(output_section, bg=PANEL)
        self.gy_output = Text(
            _gy_output_frame, height=24, wrap="word", bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        _gy_output_scroll = ttk.Scrollbar(
            _gy_output_frame, orient="vertical", command=self.gy_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.gy_output.configure(yscrollcommand=_gy_output_scroll.set)
        self.gy_output.pack(side="left", fill="both", expand=True)
        _gy_output_scroll.pack(side="right", fill="y")
        _gy_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.gy_output)

        self._graveyard_files_by_label: dict[str, str] = {}
        self._graveyard_refresh_list()

    def _graveyard_refresh_list(self):
        """(Re)populates the instrument/timeframe dropdown from every
        persistent graveyard file currently on disk (see
        app.search.graveyard.list_graveyard_files) -- call this after
        running Forge / Search Lab / Evolution Lab to pick up a file that
        didn't exist yet when this tab was first built."""
        try:
            files = list_graveyard_files()
        except Exception:
            files = []
        self._graveyard_files_by_label = {}
        labels = []
        for entry in files:
            label = f"{entry['instrument']} / {entry['timeframe']} ({entry['n_rows']} rows)"
            self._graveyard_files_by_label[label] = entry["path"]
            labels.append(label)
        self.gy_file_combo.set_options(labels)
        if labels:
            self.gy_file_combo.set_str(labels[0])

    def _graveyard_show_clicked(self):
        self.gy_output.delete("1.0", END)
        label = self.gy_file_combo.get_str()
        path = self._graveyard_files_by_label.get(label)
        if not path:
            self.gy_output.insert(
                END,
                "No graveyard file selected -- click REFRESH LIST after running Forge, Search Lab, "
                "or Evolution Lab at least once (a graveyard file is only created once something has "
                "reached full evaluation/CPCV and then failed).\n",
            )
            return
        try:
            rows = load_graveyard(Path(path))
            clusters = summarize_graveyard(rows, top_n=100)
            self.gy_output.insert(END, render_graveyard_report(clusters))
        except Exception:
            self.gy_output.insert(END, "Unexpected error:\n" + traceback.format_exc())

    # -----------------------------------------------------------------------
    # Tab — Your Idea (discovery)
    # -----------------------------------------------------------------------

    def _build_youridea_tab(self):
        """Desktop twin of the web /discover screen (v9.5): type an idea,
        it compiles to a validated rule spec, runs a market x timeframe
        grid (parameter variants included, every run counted for
        deflated-Sharpe), then the strongest cell is attacked by the
        break-it battery. Hypotheses persist in the discovery store, so
        desktop and web see the same research memory."""
        f = self._scrollable(self.tab_youridea)

        self._page_header(
            f,
            "DISCOVERY",
            "Your Idea",
            "Describe a trading idea in plain words. It becomes a validated rule, runs across the "
            "datasets and timeframes you pick (parameter variants included -- every run counts against "
            "the deflation), then the strongest cell is attacked: random-entry null, held-out data, "
            "2x costs, volatility regimes, parameter neighbours, time folds, other markets.",
        )

        idea_section = self._section(f, "The idea", "", emphasize=True)
        self.yi_idea_text = Text(
            idea_section, height=4, wrap="word", bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=_safe_font(10),
        )
        self.yi_idea_text.pack(fill="x", padx=18, pady=(3, 10))
        self.yi_idea_text.insert(
            END,
            "buy the first retrace into a fair value gap, stop beyond the far edge, target 2R",
        )

        pick_section = self._section(f, "Datasets and timeframes", "")
        Label(
            pick_section, text="Datasets (Ctrl/Cmd-click for several markets)",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), anchor="w",
        ).pack(fill="x", padx=18, pady=(6, 2))
        self.yi_dataset_list = Listbox(
            pick_section, selectmode=EXTENDED, height=8, bg=LOG_BG, fg=TEXT,
            selectbackground="#2b3a55", relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=_safe_font(9), exportselection=False,
        )
        self.yi_dataset_list.pack(fill="x", padx=18, pady=(0, 8))
        self.yi_dataset_names: list[str] = []

        self.yi_tf_var = StringVar(value="")
        self.yi_bars_var = StringVar(value="60000")
        self.yi_null_var = StringVar(value="200")
        for label, var, hint in (
            ("Timeframes (comma separated; empty = data's own)", self.yi_tf_var, "e.g. 15min, 1h"),
            ("Max bars per dataset (most recent)", self.yi_bars_var, ""),
            ("Random-entry null draws", self.yi_null_var, ""),
        ):
            row = Frame(pick_section, bg=PANEL)
            row.pack(fill="x", padx=18, pady=3)
            Label(row, text=label, width=52, anchor="w", bg=PANEL, fg=TEXT_MUTED,
                  font=_safe_font(9)).pack(side="left")
            Entry(row, textvariable=var, width=28, bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
                  relief="flat", highlightthickness=1, highlightbackground=BORDER,
                  font=_safe_font(9)).pack(side="left", padx=(4, 0))
            if hint:
                Label(row, text=hint, bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8)).pack(side="left", padx=(8, 0))

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN THE IDEA", self._youridea_run_clicked, primary=True).pack(side="left", padx=(0, 8))
        self._button(button_row, "REFRESH", self._youridea_refresh, primary=False).pack(side="left", padx=(0, 8))
        self._button(button_row, "MINE RESEARCH LIBRARY", self._youridea_mine_clicked, primary=False).pack(side="left")
        self.yi_status_var = StringVar(value="")
        Label(button_row, textvariable=self.yi_status_var, bg=BG, fg=TEXT_MUTED,
              font=_safe_font(9)).pack(side="left", padx=(12, 0))

        output_section = self._section(f, "Result", "")
        _yi_out_frame = Frame(output_section, bg=PANEL)
        self.yi_output = Text(
            _yi_out_frame, height=26, wrap="word", bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        _yi_out_scroll = ttk.Scrollbar(
            _yi_out_frame, orient="vertical", command=self.yi_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.yi_output.configure(yscrollcommand=_yi_out_scroll.set)
        self.yi_output.pack(side="left", fill="both", expand=True)
        _yi_out_scroll.pack(side="right", fill="y")
        _yi_out_frame.pack(fill="both", expand=True, padx=18, pady=(3, 10))
        self._bind_isolated_wheel(self.yi_output)

        hyp_section = self._section(f, "Hypotheses on file", "")
        _yi_hyp_frame = Frame(hyp_section, bg=PANEL)
        self.yi_hypotheses = Text(
            _yi_hyp_frame, height=10, wrap="word", bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        _yi_hyp_scroll = ttk.Scrollbar(
            _yi_hyp_frame, orient="vertical", command=self.yi_hypotheses.yview, style="T58.Vertical.TScrollbar",
        )
        self.yi_hypotheses.configure(yscrollcommand=_yi_hyp_scroll.set)
        self.yi_hypotheses.pack(side="left", fill="both", expand=True)
        _yi_hyp_scroll.pack(side="right", fill="y")
        _yi_hyp_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.yi_hypotheses)

        self._youridea_refresh()

    def _youridea_refresh(self):
        """Repopulate the dataset listbox and the stored-hypotheses view."""
        try:
            from app.data.storage import list_stored_datasets

            self.yi_dataset_names = [d.name for d in list_stored_datasets()]
        except Exception:
            self.yi_dataset_names = []
        self.yi_dataset_list.delete(0, END)
        for name in self.yi_dataset_names:
            self.yi_dataset_list.insert(END, name)
        if self.yi_dataset_names:
            self.yi_dataset_list.selection_set(0)

        lines = []
        try:
            from app.discovery.hypothesis import HypothesisStore

            for h in HypothesisStore().all()[-25:][::-1]:
                lines.append(
                    f"[{h.status}] {h.idea[:110]}  |  markets: {', '.join(map(str, h.markets))}  "
                    f"|  timeframes: {', '.join(map(str, h.timeframes))}  |  experiments: {len(h.experiments)}  |  via {h.source}"
                )
        except Exception:
            lines = []
        self.yi_hypotheses.delete("1.0", END)
        self.yi_hypotheses.insert(
            END, "\n".join(lines) if lines else "No hypotheses stored yet -- run an idea above, or mine the research library."
        )

    def _youridea_run_clicked(self):
        idea = self.yi_idea_text.get("1.0", END).strip()
        names = [self.yi_dataset_names[i] for i in self.yi_dataset_list.curselection()]
        if not idea or not names:
            self.yi_output.delete("1.0", END)
            self.yi_output.insert(END, "Describe an idea and select at least one dataset first.\n")
            return
        tfs = [t.strip() for t in self.yi_tf_var.get().split(",") if t.strip()]
        try:
            max_bars = int(self.yi_bars_var.get() or 60000)
        except ValueError:
            max_bars = 60000
        try:
            n_null = int(self.yi_null_var.get() or 200)
        except ValueError:
            n_null = 200
        self.yi_status_var.set("Running...")
        self.yi_output.delete("1.0", END)
        self.yi_output.insert(END, "Compiling the idea and running the grid -- this can take several minutes.\n")
        threading.Thread(
            target=self._youridea_worker, daemon=True, args=(idea, names, tfs, max_bars, n_null)
        ).start()

    def _youridea_worker(self, idea, names, tfs, max_bars, n_null):
        def _show(text):
            def _apply():
                self.yi_output.delete("1.0", END)
                self.yi_output.insert(END, text)
                self.yi_status_var.set("")
                self._youridea_refresh()

            self.root.after(0, _apply)

        try:
            from app.ai.llm_client import preferred_client
            from app.backtest.risk import RiskConfig
            from app.data.importer import import_csv
            from app.data.instrument_specs import (
                apply_any_instrument_spec as apply_instrument_spec,
                guess_any_instrument_symbol as guess_instrument_symbol,
            )
            from app.data.storage import resolve_stored_dataset
            from app.discovery.experiment_runner import run_hypothesis
            from app.discovery.hypothesis import HypothesisStore
            from app.discovery.idea_compiler import compile_idea

            client = preferred_client()
            hyp = compile_idea(
                idea, llm=None if client.__class__.__name__ == "NullClient" else client,
                markets=names, timeframes=tfs,
            )
            datasets = {}
            for n in names:
                path = resolve_stored_dataset(n)
                if path is None:
                    raise ValueError(f"unknown dataset {n!r}")
                res = import_csv(str(path))
                if not res.is_valid:
                    raise ValueError(f"could not read {n!r}")
                df = res.dataframe
                datasets[n] = df.iloc[-max_bars:].reset_index(drop=True) if max_bars and len(df) > max_bars else df
            sym = guess_instrument_symbol(names[0]) if names else None
            base = RiskConfig(
                initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0,
                sizing_mode="fit_stop", max_trades_per_day=20,
            )
            risk = apply_instrument_spec(base, sym) if sym else base
            run = run_hypothesis(hyp, datasets, risk, store=HypothesisStore(), n_null=n_null)
            text = run.render()  # includes the break-it battery render
            _show(text + f"\n\nStatus: {hyp.status}. Stored as hypothesis {hyp.id}.\n")
        except Exception:
            _show("The run failed:\n" + traceback.format_exc())

    def _youridea_mine_clicked(self):
        self.yi_status_var.set("Mining research library...")
        threading.Thread(target=self._youridea_mine_worker, daemon=True).start()

    def _youridea_mine_worker(self):
        def _done(msg):
            def _apply():
                self.yi_status_var.set(msg)
                self._youridea_refresh()

            self.root.after(0, _apply)

        try:
            from app.ai.llm_client import preferred_client
            from app.discovery.hypothesis import HypothesisStore
            from app.discovery.paper_hypotheses import extract_from_library

            client = preferred_client()
            found = extract_from_library(
                llm=None if client.__class__.__name__ == "NullClient" else client,
                store=HypothesisStore(),
            )
            _done(f"Mined {len(found)} new hypothesis(es) from the research library.")
        except Exception:
            _done("Mining failed:\n" + traceback.format_exc())

    # -----------------------------------------------------------------------
    # Tab — Final Selection Leaderboard
    # -----------------------------------------------------------------------

    def _build_leaderboard_tab(self):
        f = self._scrollable(self.tab_leaderboard)

        self._page_header(
            f,
            "FINAL SELECTION",
            "Final Selection Leaderboard",
            "One ranked view across every strategy that has been through Full Pipeline -- Forge, "
            "Evolution Lab, Search Lab, and Full Pipeline all converge here once a strategy has a "
            "T58 Score, replacing the separate leaderboards each of those tools used to keep on its "
            "own. Ranked by T58 Score (app.scoring.t58_scorecard); anything that failed the risk-of-"
            "ruin hard safety gate is excluded by default.",
        )

        section = self._section(f, "Filters", "", emphasize=True)
        self.lb_type_combo = LabeledCombo(section, "Strategy type", ["All"] + list(STRATEGY_TYPES), default="All")
        self.lb_top_n_combo = LabeledCombo(section, "Show top", ["10", "20", "50", "100"], default="20")
        self.lb_include_ruin_fail_var = BooleanVar(value=False)
        Checkbutton(
            section, text="Include strategies that failed the risk-of-ruin hard gate",
            variable=self.lb_include_ruin_fail_var, bg=PANEL, fg=TEXT, selectcolor=PANEL,
            activebackground=PANEL, highlightthickness=0,
        ).pack(anchor="w", padx=14, pady=(4, 10))

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "REFRESH LEADERBOARD", self._leaderboard_refresh_clicked, primary=True).pack(side="left")

        output_section = self._section(f, "Ranking", "")
        _lb_output_frame = Frame(output_section, bg=PANEL)
        self.lb_output = Text(
            _lb_output_frame, height=28, wrap="none", bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        _lb_output_scroll = ttk.Scrollbar(
            _lb_output_frame, orient="vertical", command=self.lb_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.lb_output.configure(yscrollcommand=_lb_output_scroll.set)
        self.lb_output.pack(side="left", fill="both", expand=True)
        _lb_output_scroll.pack(side="right", fill="y")
        _lb_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.lb_output)

        self._leaderboard_refresh_clicked()

    def _leaderboard_refresh_clicked(self):
        self.lb_output.delete("1.0", END)
        type_choice = self.lb_type_combo.get_str()
        strategy_type = None if type_choice in ("All", "") else type_choice
        try:
            top_n = int(self.lb_top_n_combo.get_str())
        except (TypeError, ValueError):
            top_n = 20
        try:
            entries = build_leaderboard(
                strategy_type=strategy_type, top_n=top_n,
                exclude_ruin_hard_fail=not self.lb_include_ruin_fail_var.get(),
            )
            self.lb_output.insert(END, render_leaderboard_table(entries))
        except Exception:
            self.lb_output.insert(END, "Unexpected error:\n" + traceback.format_exc())

    # -----------------------------------------------------------------------
    # Generate Strategies (AI) -- draft new strategy code from a plain-
    # language idea via a local Ollama model, grounded in research/ papers
    # and your own best existing strategies. See
    # app.ai.strategy_generator for the full safety rationale: every
    # result lands in the library tagged DRAFT, never higher, and nothing
    # here ever runs generated code automatically.
    # -----------------------------------------------------------------------

    def _build_generate_strategies_tab(self):
        f = self._scrollable(self.tab_genstrat)
        self._page_header(
            f,
            "CREATE / Generate Strategies (AI)",
            "Generate Strategies",
            "Drafts a NEW strategy's source code from a plain-language idea, using a local "
            "Ollama model -- grounded in whatever papers you've dropped into the research/ "
            "folder and in your own best-performing saved strategies of the same language, "
            "so it matches this app's actual code style and constraints instead of guessing. "
            "Every strategy this produces is saved tagged DRAFT and has to earn its way "
            "through the normal backtest -> validation ladder like anything else -- nothing "
            "is ever run automatically.",
        )

        warn = Frame(f, bg="#3A2E0E", highlightthickness=1, highlightbackground=AMBER)
        warn.pack(fill="x", padx=24, pady=(0, 16))
        Label(
            warn, text="Read before using: a local model can write code that LOOKS reasonable "
            "but has a lookahead bug, an unsupported PineScript/MQL5 construct, or exit logic "
            "that silently never fires. Treat every generated strategy as a rough first draft, "
            "not a finished one -- run it through 05 Run & Report and ideally 15 Full Pipeline "
            "before trusting any of its numbers, exactly like a stranger's strategy you found "
            "online.",
            bg="#3A2E0E", fg=AMBER, font=_safe_font(9), wraplength=900, justify="left",
        ).pack(anchor="w", padx=16, pady=12)

        idea_section = self._section(
            f, "Strategy idea",
            "Describe entry/exit logic, indicators, market/timeframe, and anything else you "
            "want reflected -- the more specific, the better the draft. Research excerpts and "
            "your own prior strategies (if any exist yet for the chosen language) are added "
            "automatically as grounding.",
            emphasize=True,
        )
        # Deliberately NOT list(STRATEGY_TYPES): the AI strategy generator
        # (app.ai.strategy_generator.LANGUAGE_EXTENSIONS) only ever emits
        # source code in these three languages -- "manual" configs aren't
        # code text it generates, so it must not appear as a choice here
        # even though "manual" is now a real Strategy Library type.
        self.genstrat_language = LabeledCombo(idea_section, "Language", ["python", "pinescript", "mql5"], "python")
        _genstrat_idea_frame = Frame(idea_section, bg=PANEL)
        self.genstrat_idea = Text(
            _genstrat_idea_frame, height=6, wrap="word", bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        _genstrat_idea_scroll = ttk.Scrollbar(
            _genstrat_idea_frame, orient="vertical", command=self.genstrat_idea.yview, style="T58.Vertical.TScrollbar",
        )
        self.genstrat_idea.configure(yscrollcommand=_genstrat_idea_scroll.set)
        self.genstrat_idea.pack(side="left", fill="both", expand=True)
        _genstrat_idea_scroll.pack(side="right", fill="y")
        _genstrat_idea_frame.pack(fill="x", padx=18, pady=(4, 12))
        self._bind_isolated_wheel(self.genstrat_idea)

        self._build_ai_assist_section(f, prefix="genstrat_ai")

        resource_section = self._section(
            f, "Resource usage (local Ollama)",
            "Keeps this one-off code-generation call lean on a local machine -- a smaller "
            "context window and output cap mean less RAM/VRAM used and a faster response, "
            "without materially limiting a strategy file (a few hundred lines fits "
            "comfortably within these defaults). The request also streams its response "
            "instead of waiting for one giant reply, so a slow model that's still actively "
            "producing tokens won't get cut off just for taking a while -- only a model that "
            "goes fully quiet mid-response, or runs past the hard time cap below, does. Raise "
            "these only if generation keeps visibly getting cut off mid-file.",
        )
        self.genstrat_num_ctx = LabeledEntry(resource_section, "Context window (tokens)", strategy_generator_module.DEFAULT_NUM_CTX)
        self.genstrat_num_predict = LabeledEntry(resource_section, "Max output length (tokens)", strategy_generator_module.DEFAULT_NUM_PREDICT)
        self.genstrat_stall_timeout = LabeledEntry(
            resource_section, "Stall timeout -- seconds of total silence before giving up", strategy_generator_module.DEFAULT_TIMEOUT_SECONDS,
        )
        self.genstrat_max_total = LabeledEntry(
            resource_section, "Hard time cap (seconds, even if still generating)", strategy_generator_module.DEFAULT_MAX_TOTAL_SECONDS,
        )

        btn_row = Frame(f, bg=BG)
        btn_row.pack(fill="x", padx=24, pady=(2, 10))
        self.genstrat_run_btn = self._button(btn_row, "GENERATE DRAFT", self._genstrat_generate_clicked, primary=True)
        self.genstrat_run_btn.pack(side="left")
        self.genstrat_progress = NeuralProgress(f)
        self.genstrat_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Generated code", "Nothing has been saved yet -- review it below first.")
        self.genstrat_filename = LabeledEntry(output_section, "Filename (no extension)", "")
        _genstrat_output_frame = Frame(output_section, bg=PANEL)
        self.genstrat_output = Text(
            _genstrat_output_frame, height=22, wrap="none", bg=LOG_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        _genstrat_output_scroll = ttk.Scrollbar(
            _genstrat_output_frame, orient="vertical", command=self.genstrat_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.genstrat_output.configure(yscrollcommand=_genstrat_output_scroll.set)
        self.genstrat_output.pack(side="left", fill="both", expand=True)
        _genstrat_output_scroll.pack(side="right", fill="y")
        _genstrat_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 10))
        self._bind_isolated_wheel(self.genstrat_output)

        out_btn_row = Frame(output_section, bg=PANEL)
        out_btn_row.pack(anchor="w", padx=18, pady=(0, 6))
        self.genstrat_save_btn = self._button(
            out_btn_row, "SAVE TO LIBRARY AS DRAFT", self._genstrat_save_clicked, primary=True
        )
        self.genstrat_save_btn.config(state="disabled")
        self.genstrat_save_btn.pack(side="left")
        Label(
            output_section, text="Saved strategies land in the Strategy Library (01 Strategy Configuration) tagged "
            "DRAFT -- open that tab, LOAD SELECTED (or check it in Batch selection), and run it through "
            "05 Run & Report / 15 Full Pipeline like anything else before trusting it.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=900, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        self.genstrat_status = Label(
            f, text="", bg=BG, fg=TEXT_DIM, font=_safe_font(9), wraplength=900, justify="left",
        )
        self.genstrat_status.pack(anchor="w", padx=26, pady=(0, 20))

        self._genstrat_last_result = None

    def _genstrat_generate_clicked(self):
        idea = self.genstrat_idea.get("1.0", END).strip()
        if not idea:
            messagebox.showinfo("Describe the idea first", "Type a strategy idea in the box above first.")
            return
        language = self.genstrat_language.get_str()
        settings = self._build_ollama_settings(prefix="genstrat_ai")
        if not settings.is_usable:
            messagebox.showwarning(
                "Ollama not enabled",
                "Turn on 'Enable AI Assist for this run' above (and confirm TEST CONNECTION works) first.",
            )
            return
        self.genstrat_output.delete("1.0", END)
        self.genstrat_filename.var.set("")
        self.genstrat_save_btn.config(state="disabled")
        self.genstrat_status.config(text="Generating... this can take a while on a local model.", fg=AMBER)
        self.genstrat_run_btn.config(state="disabled")
        self.genstrat_progress.start(10)
        num_ctx = self.genstrat_num_ctx.get_int(strategy_generator_module.DEFAULT_NUM_CTX)
        num_predict = self.genstrat_num_predict.get_int(strategy_generator_module.DEFAULT_NUM_PREDICT)
        stall_timeout = self.genstrat_stall_timeout.get_int(strategy_generator_module.DEFAULT_TIMEOUT_SECONDS)
        max_total = self.genstrat_max_total.get_int(strategy_generator_module.DEFAULT_MAX_TOTAL_SECONDS)
        threading.Thread(
            target=self._genstrat_run,
            args=(settings, language, idea, num_ctx, num_predict, stall_timeout, max_total),
            daemon=True,
        ).start()

    def _genstrat_progress(self, tokens: int, elapsed: float) -> None:
        def _update():
            self.genstrat_status.config(
                text=f"Generating... {tokens} token(s) received so far, {elapsed:.0f}s elapsed.", fg=AMBER,
            )
        try:
            self.root.after(0, _update)
        except Exception:
            pass

    def _genstrat_run(self, settings, language, idea, num_ctx, num_predict, stall_timeout, max_total):
        from app.ai.strategy_generator import generate_strategy

        try:
            result = generate_strategy(
                settings, language, idea,
                timeout=stall_timeout, max_total_seconds=max_total,
                num_ctx=num_ctx, num_predict=num_predict,
                progress_cb=self._genstrat_progress,
            )
        except Exception as exc:
            result = None
            error = f"Unexpected error: {exc}"
        else:
            error = result.error

        def _finish():
            self.genstrat_progress.stop()
            self.genstrat_run_btn.config(state="normal")
            if result is None or result.code is None:
                self.genstrat_status.config(text=error or "Generation failed.", fg=RED)
                return
            self._genstrat_last_result = result
            self.genstrat_output.insert("1.0", result.code)
            self.genstrat_filename.var.set(result.filename_hint or "generated_strategy")
            self.genstrat_save_btn.config(state="normal")
            self.genstrat_status.config(
                text="Draft generated -- review the code below (it is NOT saved or tested yet), "
                "then SAVE TO LIBRARY AS DRAFT when you're ready.",
                fg=GREEN,
            )

        try:
            self.root.after(0, _finish)
        except Exception:
            pass

    def _genstrat_save_clicked(self):
        if self._genstrat_last_result is None or not self._genstrat_last_result.code:
            return
        language = self.genstrat_language.get_str()
        filename_stem = self.genstrat_filename.get_str().strip() or self._genstrat_last_result.filename_hint or "generated_strategy"
        filename_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", filename_stem).strip("_") or "generated_strategy"
        ext = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}[language]
        filename = f"{filename_stem}{ext}"
        # Whatever's currently in the text box wins -- lets you hand-edit
        # the draft before saving without re-generating.
        code_text = self.genstrat_output.get("1.0", END).rstrip("\n")
        try:
            saved_path = self._save_to_library_with_overwrite_prompt(
                lambda overwrite: save_strategy_text(code_text, filename, language, overwrite=overwrite),
                fallback_path=Path(filename),
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        if saved_path is None:
            return
        # Always DRAFT, regardless of anything else -- see
        # app.ai.strategy_generator's module docstring for why an
        # AI-generated strategy is never allowed to start higher than this.
        try:
            set_strategy_status(language, saved_path.name, "draft")
            save_strategy_metadata(language, saved_path.name, {
                "description": f"AI-drafted from idea: {self.genstrat_idea.get('1.0', END).strip()[:200]}",
            })
        except Exception:
            pass
        self.genstrat_status.config(
            text=f"Saved as {saved_path.name}  [DRAFT]. Open 01 Strategy Configuration to load and test it.", fg=GREEN,
        )
        self._refresh_strategy_library()
