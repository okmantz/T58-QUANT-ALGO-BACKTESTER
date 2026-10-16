"""Portfolio, multi-objective, walk-forward GA and ensemble tabs."""
from __future__ import annotations

import dataclasses
import os
import threading
import traceback
import webbrowser
from app.backtest.risk import RiskConfig, with_prop_safety_defaults
from app.data.importer import import_csv
from app.data.storage import store_csv_path
from app.data.timeframe_resample import infer_timeframe_label as _infer_timeframe_label
from app.ensemble.ensemble import EnsembleError, EnsembleVoteConfig, run_ensemble_blend, run_ensemble_vote
from app.monte_carlo.engine import run_monte_carlo
from app.optimize.multi_objective import (
    DEFAULT_OBJECTIVES,
    MultiObjectiveConfig,
    OBJECTIVE_DIRECTIONS,
    run_multi_objective_refinement,
)
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import FITNESS_METRICS, RefinementConfig
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.orchestration.resource_guard import JOB_MULTI_OBJECTIVE, JOB_WFGA
from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig, PortfolioError, run_portfolio_backtest
from app.prop.simulator import PropRules, simulate_account
from app.reports import strategy_state
from app.reports.generator import generate_full_report
from app.reports.validation_reports import (
    generate_multi_objective_report,
    generate_portfolio_report,
    generate_walkforward_ga_report,
)
from app.strategy.base import StrategyError
from pathlib import Path
from tkinter import (
    BooleanVar,
    Checkbutton,
    END,
    EXTENDED,
    Frame,
    Label,
    Listbox,
    SINGLE,
    Text,
    filedialog,
    messagebox,
    simpledialog,
    ttk,
)
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
theme.bind(globals())


class PortfolioTabsMixin:
    """Portfolio, multi-objective, walk-forward GA and ensemble tabs."""


    # -----------------------------------------------------------------------
    # Tab 11 — Multi-Asset Portfolio
    # -----------------------------------------------------------------------

    def _build_portfolio_tab(self):
        f = self._scrollable(self.tab_portfolio)

        self._page_header(
            f,
            "CHAMPION / Multi-Asset Portfolio",
            "Multi-Asset Portfolio",
            "Applies a strategy to every instrument listed below, computes the correlation "
            "matrix of their daily returns, re-weights each instrument's risk (correlated "
            "instruments get sized down), and merges every instrument's trades into one "
            "shared account equity curve -- the way trading a portfolio out of one prop "
            "account actually works. Each leg picks its own market data file directly -- "
            "and this tab's own Prop Rules and Risk settings below (for the COMBINED "
            "portfolio-level simulation) are self-contained to this tab, just like the "
            "web app.",
        )

        legs_section = self._section(
            f, "Instrument legs (at least 2 required)",
            "Each leg can use a DIFFERENT strategy. Use one of the ADD LEG: MANUAL / PYTHON / "
            "PINESCRIPT / MQL5 buttons to pick a strategy of that type directly (a Manual pick "
            "opens the Strategy Library filtered to saved Manual configs, since Manual "
            "strategies aren't loose files); ADD LEG FROM LIBRARY works the same way but lets "
            "you multi-select across saved strategies of any type at once; ADD LEG (CSV, current config) reuses "
            "whatever strategy is currently configured on Strategy Configuration / Strategy "
            "Builder (Step 1) instead. Running several genuinely different, uncorrelated "
            "strategies together as one portfolio raises the combined account's own probability "
            "of passing more reliably than pushing any single strategy's pass probability higher "
            "alone -- see PORTFOLIO PASS PROBABILITY below once you run it.",
            emphasize=True,
        )
        legs_frame = Frame(legs_section, bg=PANEL)
        legs_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.portfolio_leg_listbox = Listbox(
            legs_frame, height=6, selectmode=SINGLE, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.portfolio_leg_listbox.pack(side="left", fill="both", expand=True)
        leg_scroll = ttk.Scrollbar(legs_frame, orient="vertical", command=self.portfolio_leg_listbox.yview, style="T58.Vertical.TScrollbar")
        leg_scroll.pack(side="right", fill="y")
        self.portfolio_leg_listbox.config(yscrollcommand=leg_scroll.set)
        self._bind_isolated_wheel(self.portfolio_leg_listbox)

        leg_type_row = Frame(legs_section, bg=PANEL)
        leg_type_row.pack(anchor="w", padx=18, pady=(0, 4))
        Label(
            leg_type_row, text="Add leg by type:", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8, "bold"),
        ).pack(side="left", padx=(0, 8))
        self._button(leg_type_row, "MANUAL", lambda: self._portfolio_add_leg_by_type("manual"), primary=True).pack(side="left")
        self._button(leg_type_row, "PYTHON", lambda: self._portfolio_add_leg_by_type("python"), primary=True).pack(side="left", padx=8)
        self._button(leg_type_row, "PINESCRIPT", lambda: self._portfolio_add_leg_by_type("pinescript"), primary=True).pack(side="left")
        self._button(leg_type_row, "MQL5", lambda: self._portfolio_add_leg_by_type("mql5"), primary=True).pack(side="left", padx=8)

        leg_btn_row = Frame(legs_section, bg=PANEL)
        leg_btn_row.pack(anchor="w", padx=18, pady=(0, 12))
        self._button(leg_btn_row, "ADD LEG FROM LIBRARY", self._portfolio_add_leg_from_library, primary=True).pack(side="left")
        self._button(leg_btn_row, "ADD LEG (CSV, current config)", self._portfolio_add_leg).pack(side="left", padx=8)
        self._button(leg_btn_row, "REMOVE SELECTED LEG", self._portfolio_remove_leg).pack(side="left", padx=8)

        self._portfolio_legs: list[dict] = []


        settings = self._section(f, "Portfolio settings", "")
        self.portfolio_balance = LabeledEntry(settings, "Shared initial account balance ($)", 100000)
        self.portfolio_corr_strength = LabeledEntry(
            settings, "Correlation penalty strength (0=ignore, 1=full re-weighting)", 0.6,
        )
        self.portfolio_compute_pass_prob = LabeledCheckbox(
            settings, "Also compute the COMBINED portfolio's own probability of passing (uses this tab's own Prop Rules + Monte Carlo sims)", True,
        )

        self.portfolio_context = RunContextPanel(self, "Multi-Asset Portfolio")
        self.portfolio_context.build(f, include_data=False)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN PORTFOLIO BACKTEST", self._portfolio_run_clicked, primary=True).pack(side="left")
        self.open_portfolio_report_btn = self._button(button_row, "OPEN REPORT", self._open_portfolio_report)
        self.open_portfolio_report_btn.config(state="disabled")
        self.open_portfolio_report_btn.pack(side="left", padx=8)

        self.portfolio_progress = NeuralProgress(f)
        self.portfolio_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Portfolio output", "Live progress log.")
        _portfolio_output_frame = Frame(output_section, bg=PANEL)
        self.portfolio_output = Text(
            _portfolio_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _portfolio_output_scroll = ttk.Scrollbar(
            _portfolio_output_frame, orient="vertical", command=self.portfolio_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.portfolio_output.configure(yscrollcommand=_portfolio_output_scroll.set)
        self.portfolio_output.pack(side="left", fill="both", expand=True)
        _portfolio_output_scroll.pack(side="right", fill="y")
        _portfolio_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.portfolio_output)

        self._last_portfolio_html_path = None

    def _log_portfolio(self, msg: str):
        self.portfolio_output.insert(END, msg + "\n")
        self.portfolio_output.see(END)
        self.root.update_idletasks()

    def _open_portfolio_report(self):
        if self._last_portfolio_html_path:
            webbrowser.open(f"file://{self._last_portfolio_html_path.resolve()}")

    def _portfolio_add_leg(self):
        path = filedialog.askopenfilename(
            title="Select market data CSV for this instrument",
            filetypes=[("Market data", "*.csv *.tsv *.txt *.parquet *.zip *.7z"), ("All files", "*.*")],
        )
        if not path:
            return
        weight = simpledialog.askfloat(
            "Instrument weight",
            f"Nominal (pre-correlation) risk weight for {os.path.basename(path)}:",
            initialvalue=1.0, minvalue=0.01, maxvalue=10.0,
        )
        if weight is None:
            return
        self._portfolio_legs.append({"path": path, "weight": weight, "library_item": None, "strategy_path": None})
        self.portfolio_leg_listbox.insert(END, f"{os.path.basename(path)}  (weight={weight:g}, strategy=current Strategy Configuration)")

    def _portfolio_add_leg_by_type(self, strategy_type: str):
        """ADD LEG: MANUAL / PYTHON / PINESCRIPT / MQL5 -- picks a strategy of
        exactly that type directly, then a market-data CSV to pair it with,
        without first needing to select anything in the Strategy Library
        listbox (that's what ADD LEG FROM LIBRARY is for). MANUAL strategies
        aren't loose files on disk, so that case is delegated to the same
        library-selection flow filtered down to Manual-type saved configs.
        """
        if strategy_type == "manual":
            manual_items = [
                i for i in getattr(self, "_strategy_library_items", []) if i.strategy_type == "manual"
            ]
            if not manual_items:
                messagebox.showinfo(
                    "No saved Manual strategies",
                    "No Manual Strategy Builder configs are saved to the library yet. Build one "
                    "on the Strategy Builder tab, save it from Strategy Configuration (Step 1), "
                    "then come back here.",
                )
                return
            names = [i.name for i in manual_items]
            choice = self._ask_choice_from_list(
                "Choose Manual strategy", "Select a saved Manual strategy for this leg:", names,
            )
            if choice is None:
                return
            item = manual_items[names.index(choice)]
            self._portfolio_add_leg_for_library_item(item)
            return

        title, filetypes = self._STRATEGY_TYPE_FILEDIALOG[strategy_type]
        strategy_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if not strategy_path:
            return
        csv_path = filedialog.askopenfilename(
            title=f"Select market data CSV for '{os.path.basename(strategy_path)}'",
            filetypes=[("Market data", "*.csv *.tsv *.txt *.parquet *.zip *.7z"), ("All files", "*.*")],
        )
        if not csv_path:
            return
        weight = simpledialog.askfloat(
            "Instrument weight",
            f"Nominal (pre-correlation) risk weight for {os.path.basename(csv_path)}:",
            initialvalue=1.0, minvalue=0.01, maxvalue=10.0,
        )
        if weight is None:
            return
        self._portfolio_legs.append({
            "path": csv_path, "weight": weight, "library_item": None, "strategy_path": strategy_path,
        })
        self.portfolio_leg_listbox.insert(
            END,
            f"{os.path.basename(csv_path)}  (weight={weight:g}, strategy=[{strategy_type}] "
            f"{os.path.basename(strategy_path)})",
        )

    def _portfolio_add_leg_for_library_item(self, item):
        """Shared tail end of adding a leg once a specific Strategy Library
        item has been chosen -- used by both ADD LEG FROM LIBRARY (multi
        select) and the ADD LEG: MANUAL quick button (single pick)."""
        path = filedialog.askopenfilename(
            title=f"Select market data CSV for '{item.name}'",
            filetypes=[("Market data", "*.csv *.tsv *.txt *.parquet *.zip *.7z"), ("All files", "*.*")],
        )
        if not path:
            return
        weight = simpledialog.askfloat(
            "Instrument weight",
            f"Nominal (pre-correlation) risk weight for '{item.name}' on {os.path.basename(path)}:",
            initialvalue=1.0, minvalue=0.01, maxvalue=10.0,
        )
        if weight is None:
            return
        self._portfolio_legs.append({"path": path, "weight": weight, "library_item": item, "strategy_path": None})
        self.portfolio_leg_listbox.insert(
            END, f"{os.path.basename(path)}  (weight={weight:g}, strategy=[{item.strategy_type}] {item.name})",
        )

    def _portfolio_add_leg_from_library(self):
        """Adds one leg per currently-selected Strategy Library row (Step 1
        Strategy Configuration), each paired with a market-data file the
        user picks right after -- this is what lets a portfolio combine
        genuinely DIFFERENT strategies (e.g. a trend-follower and a
        mean-reversion strategy) into one shared-account book, instead of
        every leg being forced to reuse whatever strategy happens to be
        configured on Strategy Configuration / Strategy Builder."""
        items = self._selected_library_items()
        if not items:
            messagebox.showinfo(
                "No selection",
                "Select one or more saved strategies in the Strategy Library "
                "(01 Strategy Configuration) first, then click ADD LEG FROM LIBRARY.",
            )
            return
        for item in items:
            self._portfolio_add_leg_for_library_item(item)

    def _portfolio_remove_leg(self):
        sel = self.portfolio_leg_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.portfolio_leg_listbox.delete(idx)
        del self._portfolio_legs[idx]

    def _portfolio_run_clicked(self):
        if len(self._portfolio_legs) < 2:
            messagebox.showwarning(
                "Not enough legs",
                "Portfolio backtesting requires at least 2 instrument legs -- use one of the "
                "ADD LEG: MANUAL/PYTHON/PINESCRIPT/MQL5 buttons, ADD LEG FROM LIBRARY, or "
                "ADD LEG (CSV, current config) above.",
            )
            return
        self.portfolio_output.delete("1.0", END)
        self.portfolio_progress.start(10)
        threading.Thread(target=self._portfolio_run_pipeline, daemon=True).start()

    def _portfolio_run_pipeline(self):
        try:
            legs = []
            for leg_spec in self._portfolio_legs:
                stored_path = str(store_csv_path(leg_spec["path"]))
                result = import_csv(stored_path)
                if not result.is_valid:
                    self._log_portfolio(f"Import errors ({os.path.basename(stored_path)}):\n" + "\n".join(result.errors))
                    return
                self._log_portfolio(f"Loaded {len(result.dataframe)} bars from {os.path.basename(stored_path)}")

                library_item = leg_spec.get("library_item")
                strategy_path = leg_spec.get("strategy_path")
                if library_item is not None:
                    try:
                        leg_strategy = self._load_bulk_strategy(library_item.path)
                    except Exception as exc:
                        self._log_portfolio(f"  Could not load strategy '{library_item.name}': {exc}")
                        return
                    leg_name = library_item.name
                elif strategy_path:
                    try:
                        leg_strategy = self._load_bulk_strategy(Path(strategy_path))
                    except Exception as exc:
                        self._log_portfolio(f"  Could not load strategy '{os.path.basename(strategy_path)}': {exc}")
                        return
                    leg_name = Path(strategy_path).stem
                else:
                    leg_strategy = self._build_strategy()
                    leg_name = Path(stored_path).stem

                legs.append(InstrumentLeg(
                    name=leg_name, df=result.dataframe,
                    strategy=leg_strategy, risk=self.portfolio_context.build_risk_config(),
                    weight=leg_spec["weight"],
                ))

            self._log_portfolio(f"Running portfolio backtest across {len(legs)} leg(s)...")
            initial_balance = self.portfolio_balance.get_float(100000)
            prop_rules = None
            mc_cfg = None
            if self.portfolio_compute_pass_prob.get():
                try:
                    prop_rules = self.portfolio_context.build_prop_rules()
                    if abs(prop_rules.account_size - initial_balance) > 0.01:
                        self._log_portfolio(
                            f"  NOTE: Step 3's Prop Rules account size (${prop_rules.account_size:,.0f}) differs "
                            f"from this portfolio's shared initial balance (${initial_balance:,.0f}) -- "
                            f"the pass-probability check below uses Step 3's account size."
                        )
                    mc_cfg = self._validation_mc_config()
                except Exception as exc:
                    self._log_portfolio(f"  Could not build Prop Rules for the combined pass-probability check ({exc}) -- skipping that step.")
            config = PortfolioConfig(
                initial_balance=initial_balance,
                correlation_penalty_strength=self.portfolio_corr_strength.get_float(0.6),
                prop_rules=prop_rules, mc_config=mc_cfg,
            )
            result = run_portfolio_backtest(legs, config)
            paths = generate_portfolio_report(OUTPUT_DIR / "portfolio", result)
            self._last_portfolio_html_path = paths["html"]
            self.open_portfolio_report_btn.config(state="normal")
            self._log_portfolio(f"  Combined net profit: ${result.combined_statistics.net_profit:,.2f}")
            if result.mc_result is not None:
                self._log_portfolio(
                    f"  COMBINED PORTFOLIO pass probability: {result.mc_result.evaluation_pass_probability:.1f}%  "
                    f"(first payout: {result.mc_result.first_payout_probability:.1f}%) -- this is the number "
                    f"that answers whether diversifying actually helped."
                )
            for w in result.warnings:
                self._log_portfolio(f"  WARNING: {w}")
            self._log_portfolio("\nDone. Portfolio report written to:")
            for k, p in paths.items():
                self._log_portfolio(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_portfolio(f"\nStrategy error: {exc}")
        except PortfolioError as exc:
            self._log_portfolio(f"\nPortfolio error: {exc}")
        except Exception:
            self._log_portfolio("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.portfolio_progress.stop()

    # -----------------------------------------------------------------------
    # Tab 12 — Multi-Objective Optimization
    # -----------------------------------------------------------------------

    def _build_multiobj_tab(self):
        f = self._scrollable(self.tab_multiobj)

        self._page_header(
            f,
            "OPTIMIZE / Multi-Objective",
            "Multi-Objective Optimization",
            "Runs a real NSGA-II search across several objectives at once (e.g. "
            "Sharpe, max drawdown, eval-pass probability) instead of collapsing "
            "them into one weighted score the way Iterative Refinement's GA does. "
            "Produces a Pareto front -- a set of candidates where none is "
            "strictly worse than any other on the front. Picking a final winner "
            "from that list is left as your call. Uses the strategy configured "
            "on Step 01 -- but its own market data, prop rules, and risk "
            "settings below are self-contained to this tab, just like the web app.",
        )

        self.multiobj_context = RunContextPanel(self, "Multi-Objective")
        self.multiobj_context.build(f)

        obj_section = self._section(
            f, "Objectives (pick at least 2)",
            "Direction (higher/lower is better) is handled automatically for each.",
            emphasize=True,
        )
        self._multiobj_vars: dict[str, BooleanVar] = {}
        obj_grid = Frame(obj_section, bg=PANEL)
        obj_grid.pack(fill="x", padx=18, pady=(0, 10))
        for i, (obj_name, direction) in enumerate(OBJECTIVE_DIRECTIONS.items()):
            var = BooleanVar(value=obj_name in DEFAULT_OBJECTIVES)
            self._multiobj_vars[obj_name] = var
            cb = Checkbutton(
                obj_grid, text=f"{obj_name}  ({direction})", variable=var,
                bg=PANEL, fg=TEXT_MUTED, activebackground=PANEL, selectcolor=PANEL_3,
                highlightthickness=0, bd=0, font=_safe_font(9), anchor="w",
            )
            cb.grid(row=i // 2, column=i % 2, sticky="w", padx=6, pady=3)

        settings = self._section(f, "Search settings", "")
        self.mo_population = LabeledEntry(settings, "Population size", 20)
        self.mo_generations = LabeledEntry(settings, "Generations", 8)
        self.mo_seed = LabeledEntry(settings, "Random seed", 42)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN MULTI-OBJECTIVE SEARCH", self._multiobj_run_clicked, primary=True).pack(side="left")
        self.open_multiobj_report_btn = self._button(button_row, "OPEN REPORT", self._open_multiobj_report)
        self.open_multiobj_report_btn.config(state="disabled")
        self.open_multiobj_report_btn.pack(side="left", padx=8)

        self.multiobj_progress = NeuralProgress(f)
        self.multiobj_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Multi-objective output", "Live progress log.")
        _multiobj_output_frame = Frame(output_section, bg=PANEL)
        self.multiobj_output = Text(
            _multiobj_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _multiobj_output_scroll = ttk.Scrollbar(
            _multiobj_output_frame, orient="vertical", command=self.multiobj_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.multiobj_output.configure(yscrollcommand=_multiobj_output_scroll.set)
        self.multiobj_output.pack(side="left", fill="both", expand=True)
        _multiobj_output_scroll.pack(side="right", fill="y")
        _multiobj_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.multiobj_output)

        self._last_multiobj_html_path = None

    def _log_multiobj(self, msg: str):
        self.multiobj_output.insert(END, msg + "\n")
        self.multiobj_output.see(END)
        self.root.update_idletasks()

    def _open_multiobj_report(self):
        if self._last_multiobj_html_path:
            webbrowser.open(f"file://{self._last_multiobj_html_path.resolve()}")

    def _multiobj_run_clicked(self):
        if not self.multiobj_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        selected = [name for name, var in self._multiobj_vars.items() if var.get()]
        if len(selected) < 2:
            messagebox.showwarning("Not enough objectives", "Select at least 2 objectives.")
            return
        if not self._try_start_heavy_job(JOB_MULTI_OBJECTIVE):
            return
        self.multiobj_output.delete("1.0", END)
        self.multiobj_progress.start(10)
        threading.Thread(target=self._multiobj_run_pipeline, args=(selected,), daemon=True).start()

    def _multiobj_run_pipeline(self, selected_objectives: list[str]):
        try:
            df = self.multiobj_context.load_dataframe(self._log_multiobj)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self.multiobj_context.build_risk_config()
            rules = self.multiobj_context.build_prop_rules()
            mc_cfg = self._validation_mc_config()

            mo_cfg = MultiObjectiveConfig(
                objectives=selected_objectives,
                population_size=self.mo_population.get_int(20),
                generations=self.mo_generations.get_int(8),
                random_seed=self.mo_seed.get_int(42),
                reset_on_breach=True,
            )
            self._log_multiobj(f"Running multi-objective search for: {selected_objectives}...")
            result = run_multi_objective_refinement(df, strategy, risk, rules, mc_cfg, mo_cfg, progress_cb=self._log_multiobj)
            paths = generate_multi_objective_report(OUTPUT_DIR / "multi_objective", result)
            self._last_multiobj_html_path = paths["html"]
            self.open_multiobj_report_btn.config(state="normal")
            self._log_multiobj(f"  Final Pareto front size: {len(result.pareto_front)}")
            self._log_multiobj("\nDone. Multi-objective report written to:")
            for k, p in paths.items():
                self._log_multiobj(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_multiobj(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_multiobj(f"\nMulti-objective error: {exc}")
        except Exception:
            self._log_multiobj("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.multiobj_progress.stop()
            self._release_heavy_job(JOB_MULTI_OBJECTIVE)

    # -----------------------------------------------------------------------
    # Tab 13 — Walk-Forward-Aware GA
    # -----------------------------------------------------------------------

    def _build_wfga_tab(self):
        f = self._scrollable(self.tab_wfga)

        self._page_header(
            f,
            "VALIDATE / Walk-Forward GA",
            "Walk-Forward-Aware GA",
            "Same crossover/mutation/tournament-selection GA as Iterative Refinement "
            "(Step 6), but every candidate's fitness is scored ONLY on chained "
            "out-of-sample fold data -- never on the training windows or the full "
            "dataset. A genome that only fits one historical stretch simply scores "
            "lower here and gets selected against, generation over generation. "
            "Uses the strategy configured on Step 01 -- but its own market data, "
            "prop rules, and risk settings below are self-contained to this tab, "
            "just like the web app.",
        )

        self.wfga_context = RunContextPanel(self, "Walk-Forward GA")
        self.wfga_context.build(f)

        settings = self._section(
            f, "Fold + search settings",
            "'Rolling' or 'anchored' fold windows, same meaning as Walk-Forward Optimization (Step 8).",
            emphasize=True,
        )
        self.wfga_window_mode = LabeledCombo(settings, "Window mode", ["rolling", "anchored"], "rolling")
        self.wfga_folds = LabeledEntry(settings, "Number of folds", 4)
        self._wfga_metric_labels = list(FITNESS_METRICS.values())
        self._wfga_metric_label_to_key = {v: k for k, v in FITNESS_METRICS.items()}
        self.wfga_metric = LabeledCombo(
            settings, "Fitness metric", self._wfga_metric_labels, FITNESS_METRICS["eval_pass_probability"],
        )
        self.wfga_population = LabeledEntry(settings, "Population size", 12)
        self.wfga_generations = LabeledEntry(settings, "Generations", 6)
        self.wfga_seed = LabeledEntry(settings, "Random seed", 42)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN WALK-FORWARD-AWARE GA", self._wfga_run_clicked, primary=True).pack(side="left")
        self.open_wfga_report_btn = self._button(button_row, "OPEN REPORT", self._open_wfga_report)
        self.open_wfga_report_btn.config(state="disabled")
        self.open_wfga_report_btn.pack(side="left", padx=8)

        self.wfga_progress = NeuralProgress(f)
        self.wfga_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Walk-forward-aware GA output", "Live progress log.")
        _wfga_output_frame = Frame(output_section, bg=PANEL)
        self.wfga_output = Text(
            _wfga_output_frame, height=18, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _wfga_output_scroll = ttk.Scrollbar(
            _wfga_output_frame, orient="vertical", command=self.wfga_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.wfga_output.configure(yscrollcommand=_wfga_output_scroll.set)
        self.wfga_output.pack(side="left", fill="both", expand=True)
        _wfga_output_scroll.pack(side="right", fill="y")
        _wfga_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.wfga_output)

        self._last_wfga_html_path = None

    def _log_wfga(self, msg: str):
        self.wfga_output.insert(END, msg + "\n")
        self.wfga_output.see(END)
        self.root.update_idletasks()

    def _open_wfga_report(self):
        if self._last_wfga_html_path:
            webbrowser.open(f"file://{self._last_wfga_html_path.resolve()}")

    def _wfga_run_clicked(self):
        if not self.wfga_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        if not self._try_start_heavy_job(JOB_WFGA):
            return
        self.wfga_output.delete("1.0", END)
        self.wfga_progress.start(10)
        threading.Thread(target=self._wfga_run_pipeline, daemon=True).start()

    def _wfga_run_pipeline(self):
        try:
            df = self.wfga_context.load_dataframe(self._log_wfga)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self.wfga_context.build_risk_config()
            rules = self.wfga_context.build_prop_rules()
            # FIX (audit, Sep 2026): Walk-Forward GA never hardened its
            # RiskConfig against PropRules here either -- see
            # app.backtest.risk.with_prop_safety_defaults' own docstring.
            risk = with_prop_safety_defaults(risk, rules)
            # FIX (2026-09-18): see RiskConfig.reset_on_breach's docstring
            # -- this was already threaded into mc_cfg below but never into
            # `risk`, which every fold's own run_backtest() call actually
            # scores fitness from.
            risk = dataclasses.replace(risk, reset_on_breach=True)
            mc_cfg = dataclasses.replace(self._validation_mc_config(), reset_on_breach=True)

            metric_key = self._wfga_metric_label_to_key.get(self.wfga_metric.get_str(), "eval_pass_probability")
            refine_cfg = RefinementConfig(
                population_size=self.wfga_population.get_int(12),
                generations=self.wfga_generations.get_int(6),
                fitness_metric=metric_key,
                random_seed=self.wfga_seed.get_int(42),
            )

            self._log_wfga("Starting walk-forward-aware GA...")
            result = run_walkforward_aware_refinement(
                df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg,
                n_folds=self.wfga_folds.get_int(4), window_mode=self.wfga_window_mode.get_str(),
                progress_cb=self._log_wfga,
            )
            paths = generate_walkforward_ga_report(OUTPUT_DIR / "walkforward_ga", result)
            self._last_wfga_html_path = paths["html"]
            self.open_wfga_report_btn.config(state="normal")
            self._log_wfga(f"  Best chained-OOS fitness: {result.best.fitness:.3f}  (overfitting gap: {result.overfitting_gap})")
            for w in result.warnings:
                self._log_wfga(f"  WARNING: {w}")
            self._log_wfga("\nDone. Walk-forward-aware GA report written to:")
            for k, p in paths.items():
                self._log_wfga(f"  {k}: {p}")
            gap = getattr(result, "overfitting_gap", None)
            summary = f"overfitting gap {gap:.2f}" if gap is not None else f"{self.wfga_folds.get_int(4)} folds, walk-forward-aware GA"
            # Matches app.web.server's own /walk-forward-ga route's record_validation call.
            _active_lib = getattr(self, "_active_library_strategy", None)
            strategy_state.record_validation(
                getattr(strategy, "name", "Strategy"), self.wfga_context.instrument_label(), "wfga",
                passed=None, summary=summary, report_html=f"file://{self._last_wfga_html_path.resolve()}",
                library_type=(_active_lib[0] if _active_lib else ""), library_filename=(_active_lib[1] if _active_lib else ""),
            )
        except StrategyError as exc:
            self._log_wfga(f"\nStrategy error: {exc}")
        except RefinementError as exc:
            self._log_wfga(f"\nWalk-forward-aware GA error: {exc}")
        except Exception:
            self._log_wfga("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.wfga_progress.stop()
            self._release_heavy_job(JOB_WFGA)


    # -----------------------------------------------------------------------
    # Tab 14 — Multi-Strategy Ensemble (blend or vote)
    # -----------------------------------------------------------------------

    _ENSEMBLE_MODE_LABELS = {
        "Blend (each strategy trades independently, correlation-adjusted risk)": "blend",
        "Vote (one combined signal, entered only once enough strategies agree)": "vote",
    }

    def _build_ensemble_tab(self):
        f = self._scrollable(self.tab_ensemble)

        self._page_header(
            f,
            "CHAMPION / Multi-Strategy Ensemble",
            "Multi-Strategy Ensemble",
            "The mirror case of Step 11's Multi-Asset Portfolio: several DIFFERENT "
            "strategies combined on the SAME instrument, instead of one strategy across "
            "several instruments. Two modes -- Blend keeps every strategy trading "
            "independently at a correlation-adjusted share of the account's risk budget "
            "(reuses the Portfolio feature's own math); Vote combines every strategy's "
            "signal into one entry, taken only once enough of them agree on direction. "
            "Its own market data below is self-contained to this tab, just like the web app "
            "(account balance and risk are configured directly in Ensemble settings below).",
        )

        self.ensemble_context = RunContextPanel(self, "Multi-Strategy Ensemble")
        self.ensemble_context._build_data_section(f)

        legs_section = self._section(
            f, "Strategy legs (at least 2 required)",
            "Manual Strategy Builder configs, Python (.py), PineScript (.pine), or MQL5 "
            "(.mq5) strategies -- mixing types in the same ensemble is fine, each is "
            "detected by extension. Use the ADD LEG: MANUAL/PYTHON/PINESCRIPT/MQL5 buttons "
            "to pick one strategy of that type at a time (MANUAL opens the Strategy Library "
            "filtered to saved Manual configs), ADD FROM STRATEGY LIBRARY to multi-select "
            "across any saved type at once, or ADD FILES... to browse multiple Python / "
            "PineScript / MQL5 files in one dialog. Each leg trades the SAME instrument, "
            "using the SAME base risk settings from Step 4.",
            emphasize=True,
        )
        legs_frame = Frame(legs_section, bg=PANEL)
        legs_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.ensemble_leg_listbox = Listbox(
            legs_frame, height=6, selectmode=EXTENDED, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.ensemble_leg_listbox.pack(side="left", fill="both", expand=True)
        leg_scroll = ttk.Scrollbar(legs_frame, orient="vertical", command=self.ensemble_leg_listbox.yview, style="T58.Vertical.TScrollbar")
        leg_scroll.pack(side="right", fill="y")
        self.ensemble_leg_listbox.config(yscrollcommand=leg_scroll.set)
        self._bind_isolated_wheel(self.ensemble_leg_listbox)

        leg_type_row = Frame(legs_section, bg=PANEL)
        leg_type_row.pack(anchor="w", padx=18, pady=(0, 4))
        Label(
            leg_type_row, text="Add leg by type:", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8, "bold"),
        ).pack(side="left", padx=(0, 8))
        self._button(leg_type_row, "MANUAL", lambda: self._ensemble_add_leg_by_type("manual"), primary=True).pack(side="left")
        self._button(leg_type_row, "PYTHON", lambda: self._ensemble_add_leg_by_type("python"), primary=True).pack(side="left", padx=8)
        self._button(leg_type_row, "PINESCRIPT", lambda: self._ensemble_add_leg_by_type("pinescript"), primary=True).pack(side="left")
        self._button(leg_type_row, "MQL5", lambda: self._ensemble_add_leg_by_type("mql5"), primary=True).pack(side="left", padx=8)

        leg_btn_row = Frame(legs_section, bg=PANEL)
        leg_btn_row.pack(anchor="w", padx=18, pady=(0, 12))
        self._button(leg_btn_row, "ADD FROM STRATEGY LIBRARY", self._ensemble_add_from_library, primary=True).pack(side="left")
        self._button(leg_btn_row, "ADD FILES...", self._ensemble_add_legs).pack(side="left", padx=8)
        self._button(leg_btn_row, "REMOVE SELECTED", self._ensemble_remove_legs).pack(side="left", padx=8)
        self._button(leg_btn_row, "CLEAR ALL", self._ensemble_clear_legs).pack(side="left")

        self._ensemble_leg_paths: list[Path] = []


        settings = self._section(f, "Ensemble settings", "")
        self.ensemble_mode = LabeledCombo(
            settings, "Combination mode", list(self._ENSEMBLE_MODE_LABELS.keys()),
            "Blend (each strategy trades independently, correlation-adjusted risk)",
        )
        self.ensemble_min_agreement = LabeledEntry(settings, "Min agreement (Vote mode only)", 2)
        self.ensemble_balance = LabeledEntry(settings, "Shared initial account balance ($)", 100000)
        self.ensemble_corr_strength = LabeledEntry(
            settings, "Correlation penalty strength (Blend mode, 0=ignore, 1=full re-weighting)", 0.6,
        )

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN ENSEMBLE BACKTEST", self._ensemble_run_clicked, primary=True).pack(side="left")
        self.open_ensemble_report_btn = self._button(button_row, "OPEN REPORT", self._open_ensemble_report)
        self.open_ensemble_report_btn.config(state="disabled")
        self.open_ensemble_report_btn.pack(side="left", padx=8)

        self.ensemble_progress = NeuralProgress(f)
        self.ensemble_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Ensemble output", "Live progress log.")
        _ensemble_output_frame = Frame(output_section, bg=PANEL)
        self.ensemble_output = Text(
            _ensemble_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _ensemble_output_scroll = ttk.Scrollbar(
            _ensemble_output_frame, orient="vertical", command=self.ensemble_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.ensemble_output.configure(yscrollcommand=_ensemble_output_scroll.set)
        self.ensemble_output.pack(side="left", fill="both", expand=True)
        _ensemble_output_scroll.pack(side="right", fill="y")
        _ensemble_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.ensemble_output)

        self._last_ensemble_html_path = None

    def _log_ensemble(self, msg: str):
        self.ensemble_output.insert(END, msg + "\n")
        self.ensemble_output.see(END)
        self.root.update_idletasks()

    def _open_ensemble_report(self):
        if self._last_ensemble_html_path:
            webbrowser.open(f"file://{self._last_ensemble_html_path.resolve()}")

    def _ensemble_add_legs(self):
        paths = filedialog.askopenfilenames(
            title="Add strategy files for this ensemble (Python / PineScript / MQL5)",
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
        existing = {str(p) for p in self._ensemble_leg_paths}
        for p in paths:
            if p not in existing:
                self._ensemble_leg_paths.append(Path(p))
                self.ensemble_leg_listbox.insert(END, f"  {Path(p).name}")

    def _ensemble_add_leg_by_type(self, strategy_type: str):
        """ADD LEG: MANUAL / PYTHON / PINESCRIPT / MQL5 -- mirrors Multi-Asset
        Portfolio's per-type quick-add. MANUAL strategies aren't loose files,
        so that case picks from the Strategy Library instead of a file
        dialog; the other three browse directly to a source file."""
        if strategy_type == "manual":
            manual_items = [
                i for i in getattr(self, "_strategy_library_items", []) if i.strategy_type == "manual"
            ]
            if not manual_items:
                messagebox.showinfo(
                    "No saved Manual strategies",
                    "No Manual Strategy Builder configs are saved to the library yet. Build one "
                    "on the Strategy Builder tab, save it from Strategy Configuration (Step 1), "
                    "then come back here.",
                )
                return
            names = [i.name for i in manual_items]
            choice = self._ask_choice_from_list(
                "Choose Manual strategy", "Select a saved Manual strategy for this leg:", names,
            )
            if choice is None:
                return
            item = manual_items[names.index(choice)]
            self._ensemble_add_leg_path(Path(item.path))
            return

        title, filetypes = self._STRATEGY_TYPE_FILEDIALOG[strategy_type]
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if not path:
            return
        self._ensemble_add_leg_path(Path(path))

    def _ensemble_add_from_library(self):
        """Adds one leg per currently-selected Strategy Library row (Step 1
        Strategy Configuration) -- lets an ensemble combine already-saved
        strategies of any mix of types without re-browsing for their files."""
        items = self._selected_library_items()
        if not items:
            messagebox.showinfo(
                "No selection",
                "Select one or more saved strategies in the Strategy Library "
                "(01 Strategy Configuration) first, then click ADD FROM STRATEGY LIBRARY.",
            )
            return
        for item in items:
            self._ensemble_add_leg_path(Path(item.path))

    def _ensemble_add_leg_path(self, path: Path):
        if str(path) in {str(p) for p in self._ensemble_leg_paths}:
            return
        self._ensemble_leg_paths.append(path)
        self.ensemble_leg_listbox.insert(END, f"  {path.name}")

    def _ensemble_remove_legs(self):
        sel = list(self.ensemble_leg_listbox.curselection())
        for i in reversed(sel):
            self.ensemble_leg_listbox.delete(i)
            del self._ensemble_leg_paths[i]

    def _ensemble_clear_legs(self):
        self.ensemble_leg_listbox.delete(0, END)
        self._ensemble_leg_paths = []

    def _ensemble_run_clicked(self):
        if len(self._ensemble_leg_paths) < 2:
            messagebox.showwarning(
                "Not enough legs",
                "An ensemble requires at least 2 strategy legs -- use one of the ADD LEG: "
                "MANUAL/PYTHON/PINESCRIPT/MQL5 buttons, ADD FROM STRATEGY LIBRARY, or ADD FILES... above.",
            )
            return
        if not self.ensemble_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        self.ensemble_output.delete("1.0", END)
        self.open_ensemble_report_btn.config(state="disabled")
        self.ensemble_progress.start(10)
        threading.Thread(target=self._ensemble_run_pipeline, daemon=True).start()

    def _ensemble_run_pipeline(self):
        try:
            df = self.ensemble_context.load_dataframe(self._log_ensemble)
            if df is None:
                return

            strategies, names = [], []
            for path in self._ensemble_leg_paths:
                try:
                    strategies.append(self._load_bulk_strategy(path))
                except (StrategyError, Exception) as exc:
                    self._log_ensemble(f"Skipped '{path.name}' -- could not load strategy: {exc}")
                    continue
                names.append(path.stem)
            if len(strategies) < 2:
                self._log_ensemble("\nFewer than 2 strategy legs loaded successfully -- nothing to combine.")
                return
            self._log_ensemble(f"Loaded {len(strategies)} strategy leg(s): {', '.join(names)}")

            mode_label = self.ensemble_mode.get_str()
            mode = self._ENSEMBLE_MODE_LABELS.get(mode_label, "blend")
            balance = self.ensemble_balance.get_float(100000)
            risk = RiskConfig(initial_balance=balance)
            instrument = (
                os.path.basename(self.ensemble_context.csv_paths[0]) if len(self.ensemble_context.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.ensemble_context.csv_paths)
            )

            if mode == "blend":
                self._log_ensemble(f"Running blend ensemble across {len(strategies)} strategies...")
                config = PortfolioConfig(
                    initial_balance=balance,
                    correlation_penalty_strength=self.ensemble_corr_strength.get_float(0.6),
                )
                result = run_ensemble_blend(df, strategies, risk, names=names, config=config)
                paths = generate_portfolio_report(OUTPUT_DIR / "ensemble", result)
                self._last_ensemble_html_path = paths["html"]
                self.open_ensemble_report_btn.config(state="normal")
                self._log_ensemble(f"  Combined net profit: ${result.combined_statistics.net_profit:,.2f}")
                for w in result.warnings:
                    self._log_ensemble(f"  WARNING: {w}")
                self._log_ensemble("\nDone. Ensemble (blend) report written to:")
                for k, p in paths.items():
                    self._log_ensemble(f"  {k}: {p}")

            elif mode == "vote":
                min_agreement = self.ensemble_min_agreement.get_int(2)
                self._log_ensemble(
                    f"Running vote ensemble across {len(strategies)} strategies "
                    f"(min_agreement={min_agreement})..."
                )
                bt_result = run_ensemble_vote(
                    df, strategies, risk, names=names,
                    vote_config=EnsembleVoteConfig(min_agreement=min_agreement),
                )
                self._log_ensemble(
                    f"  Trades: {len(bt_result.trades)}  Net profit: ${bt_result.statistics.net_profit:,.2f}"
                )
                if not bt_result.trades:
                    self._log_ensemble("\nNo trades were generated by this vote ensemble -- nothing further to report.")
                    return
                rules = PropRules(account_size=balance)
                period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
                trade_pnls = [t.pnl for t in bt_result.trades]
                trade_dates = [t.entry_time for t in bt_result.trades]
                single_run = simulate_account(trade_pnls, trade_dates, rules)
                mc_cfg = self._validation_mc_config()
                mc_result = run_monte_carlo(bt_result.trades, rules, mc_cfg)
                paths = generate_full_report(
                    output_dir=OUTPUT_DIR / "ensemble", strategy_name=bt_result.strategy_name,
                    strategy_source_type="ensemble_vote", instrument=instrument, timeframe=_infer_timeframe_label(df),
                    backtest_period=period, backtest_result=bt_result, prop_rules=rules,
                    prop_single_run=single_run, monte_carlo_result=mc_result, holdout_comparison=None,
                    risk_config=risk, price_df=df,
                )
                self._last_ensemble_html_path = paths["html"]
                self.open_ensemble_report_btn.config(state="normal")
                self._log_ensemble(f"  Evaluation pass probability: {mc_result.evaluation_pass_probability:.1f}%")
                self._log_ensemble("\nDone. Ensemble (vote) report written to:")
                for k, p in paths.items():
                    self._log_ensemble(f"  {k}: {p}")
            else:
                self._log_ensemble(f"\nUnknown ensemble mode '{mode_label}'.")
        except EnsembleError as exc:
            self._log_ensemble(f"\nEnsemble error: {exc}")
        except StrategyError as exc:
            self._log_ensemble(f"\nStrategy error: {exc}")
        except Exception:
            self._log_ensemble("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.ensemble_progress.stop()
