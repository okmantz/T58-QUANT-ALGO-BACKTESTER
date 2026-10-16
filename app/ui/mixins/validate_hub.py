"""Validate (simple/hub), risk sweep, compare, replay/start-here hooks and autopilot hubs."""
from __future__ import annotations

import random
import threading
import traceback
from app.backtest.engine import run_backtest
from app.backtest.risk import build_run_context
from app.monte_carlo.engine import run_monte_carlo
from app.optimize.parameter_space import apply_genome, extract_genome
from app.prop.survival_engine import PropSurvivalConfig
from app.reports import strategy_state
from app.reports.crash_log import log_crash
from app.strategy.base import StrategyError
from app.strategy.library import list_saved_strategies
from app.validation.cpcv import compute_pbo, run_cpcv
from app.validation.parameter_robustness import compute_parameter_robustness
from app.validation.regime_matrix import run_regime_matrix
from app.validation.sensitivity import compute_1d_sensitivity
from app.validation.walk_forward_opt import run_walk_forward_optimization
from pathlib import Path
from tkinter import END, Frame, Label, Listbox, Text, messagebox, ttk
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledEntry
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
VIOLET = ""
theme.bind(globals())


class ValidateHubMixin:
    """Validate (simple/hub), risk sweep, compare, replay/start-here hooks and autopilot hubs."""


    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Validate -- pick your checks (shared dispatch)
    # -----------------------------------------------------------------------

    def _run_validation_checks(self, checks, strategy_factory, df, risk, rules, mc_config, log, *,
                               cpcv_n_groups=6, cpcv_n_test_groups=2, sens_pct_range=0.5, sens_steps=9):
        """The one dispatch behind "Validate (pick your checks)" AND the
        Champion Board's re-run button, so both surfaces run identical
        checks. Runs ONE baseline backtest first (risk already hardened by
        the caller), then feeds that run's trades / fresh strategy
        instances to each ticked check's own module-level runner.
        `checks` is a set drawn from {"montecarlo", "cpcv", "pbo",
        "walkforward", "sensitivity", "robustness", "regime"}. Every
        check is attempted independently -- one failing never blocks the
        rest. Returns {check_name: result_object} for the checks that
        produced one; all human reporting goes through `log`, using only
        fields the returned objects actually carry (no invented
        thresholds -- PASS/FAIL wording only where the module itself
        computes a verdict: CPCV's is_robust)."""
        results: dict = {}
        log("Running baseline backtest (hardened risk)...")
        bt = run_backtest(df, strategy_factory(), risk)
        stats = bt.statistics
        log(f"Baseline: {len(bt.trades)} trade(s), net ${stats.net_profit:,.2f}, "
            f"win rate {stats.win_rate:.1f}%, profit factor {stats.profit_factor:.2f}, "
            f"max drawdown {stats.max_drawdown_pct:.1f}%.")

        if "montecarlo" in checks:
            log("\n--- Monte Carlo ---")
            try:
                if not bt.trades:
                    log("Monte Carlo: skipped -- the baseline produced zero trades.")
                else:
                    mc = run_monte_carlo(bt.trades, rules, mc_config)
                    results["montecarlo"] = mc
                    log(f"Monte Carlo ({mc_config.n_simulations:,} sims): eval pass "
                        f"{mc.evaluation_pass_probability:.1f}%, first payout "
                        f"{mc.first_payout_probability:.1f}%, risk of ruin {mc.risk_of_ruin_pct:.1f}%.")
            except Exception:
                log("Monte Carlo failed:\n" + traceback.format_exc())

        if "cpcv" in checks:
            log("\n--- CPCV ---")
            try:
                res = run_cpcv(
                    df, strategy_factory, risk,
                    n_groups=cpcv_n_groups, n_test_groups=cpcv_n_test_groups,
                    prop_rules=rules, mc_cfg=mc_config,
                )
                results["cpcv"] = res
                log(f"CPCV: {res.n_paths} path(s), mean OOS {res.metric} {res.mean_oos_metric:.3f} "
                    f"vs in-sample {res.mean_is_metric:.3f} -- CPCV verdict: "
                    f"{'ROBUST' if res.is_robust else 'NOT ROBUST'} "
                    f"(OOS below in-sample on {res.pct_paths_oos_below_is:.0f}% of paths).")
            except Exception:
                log("CPCV failed:\n" + traceback.format_exc())

        if "pbo" in checks:
            log("\n--- PBO ---")
            try:
                strategy = strategy_factory()
                if strategy.source_type == "manual":
                    specs = [{"source_type": "manual", "config": dict(strategy.config)}]
                    genes = extract_genome(strategy.config)
                    rng = random.Random(42)
                    for _ in range(4):
                        if not genes:
                            break
                        genome = [max(min(g.base_value + rng.uniform(-0.3, 0.3) * (g.hi - g.lo), g.hi), g.lo) for g in genes]
                        specs.append({"source_type": "manual", "config": apply_genome(strategy.config, genes, genome)})
                    if not genes:
                        log("No tunable parameters -- PBO runs with a single candidate (degenerate case).")
                else:
                    specs = [{
                        "source_type": strategy.source_type,
                        "code_text": Path(strategy.file_path).read_text(encoding="utf-8"),
                        "code_extension": Path(strategy.file_path).suffix,
                    }]
                    log(f"'{strategy.source_type}' candidate perturbation isn't built here -- "
                        "PBO runs with this one strategy as the pool (degenerate case).")
                res = compute_pbo(
                    df, specs, risk,
                    n_groups=cpcv_n_groups, n_test_groups=cpcv_n_test_groups,
                    prop_rules=rules,
                )
                results["pbo"] = res
                log(f"PBO: {res.pbo:.2f} across {res.n_candidates} candidate(s) on {res.n_paths} path(s) "
                    f"-- the probability the in-sample winner is a backtest-overfitting artifact (lower is better).")
            except Exception:
                log("PBO failed:\n" + traceback.format_exc())

        if "walkforward" in checks:
            log("\n--- Walk-Forward ---")
            try:
                res = run_walk_forward_optimization(
                    df, strategy_factory(), risk, rules, mc_config, progress_cb=log,
                )
                results["walkforward"] = res
                eff = res.out_of_sample_efficiency
                eff_text = f"{eff:.2f}" if eff is not None else "n/a"
                log(f"Walk-Forward: {len(res.folds)} fold(s) completed, OOS efficiency {eff_text}, "
                    f"{len(res.combined_trades)} combined out-of-sample trade(s).")
            except Exception:
                log("Walk-Forward failed:\n" + traceback.format_exc())

        if "sensitivity" in checks:
            log("\n--- Sensitivity ---")
            try:
                sweeps = compute_1d_sensitivity(
                    df, strategy_factory(), risk, rules, mc_config, metric="profit_factor",
                    pct_range=sens_pct_range, n_steps=sens_steps, progress_cb=log,
                )
                results["sensitivity"] = sweeps
                for r in sweeps:
                    flag = "  <-- CLIFF" if r.cliff_detected else ""
                    log(f"  {r.gene_label}: max adjacent-step drop "
                        f"{r.max_pct_drop_between_adjacent_steps:.0f}%{flag}")
            except Exception:
                log("Sensitivity failed:\n" + traceback.format_exc())

        if "robustness" in checks:
            log("\n--- Robustness ---")
            try:
                res = compute_parameter_robustness(
                    df, strategy_factory(), risk, rules, mc_config,
                    pct_range=sens_pct_range, n_steps_1d=sens_steps, progress_cb=log,
                )
                results["robustness"] = res
                log(f"Robustness: score {res.parameter_robustness_score:.0f}/100 across "
                    f"{res.n_parameters_checked} parameter(s); {res.n_cliffs_detected} cliff(s) detected.")
            except Exception:
                log("Robustness failed:\n" + traceback.format_exc())

        if "regime" in checks:
            log("\n--- Regime ---")
            try:
                res = run_regime_matrix(df, strategy_factory(), risk)
                if res is None:
                    log("Regime: not enough bars in this dataset to classify regimes reliably.")
                else:
                    results["regime"] = res
                    disabled = res.disable_regimes()
                    log(f"Regime: {len(res.cells)} cell(s) analysed across "
                        f"{' x '.join(res.primary_dimensions)}; {len(disabled)} regime cell(s) "
                        "flagged as gate-OFF candidates (enough trades, losing profit factor).")
                    for n in res.notes:
                        log(f"  Note: {n}")
            except Exception:
                log("Regime failed:\n" + traceback.format_exc())

        return results

    def _build_validate_simple_tab(self):
        f = self._scrollable(self.tab_validate_simple)

        self._page_header(
            f,
            "VALIDATE / Validate (pick your checks)",
            "\u2726 Validate \u2014 pick your checks",
            "One clear thing to do: the strategy comes from CREATE \u2192 Strategy Builder "
            "(Step 01, same as every other Validate tab), the dataset / prop firm / risk "
            "below are this tab's own copy, then tick the checks you want and hit RUN. "
            "One baseline backtest runs first on hardened settings; every ticked check "
            "feeds off that same run, so the numbers describe the same strategy on the "
            "same data. Each dedicated Validate tab still exists for its full settings "
            "and official reports.",
        )

        self.val_simple_context = RunContextPanel(self, "Validate (pick your checks)")
        self.val_simple_context.build(f)

        checks_section = self._section(
            f, "Checks", "Ticked checks run in this order against the one baseline backtest.", emphasize=True,
        )
        self.val_chk_montecarlo = LabeledCheckbox(checks_section, "Monte Carlo \u2014 resampled pass / payout / ruin probabilities", True)
        self.val_chk_cpcv = LabeledCheckbox(checks_section, "CPCV \u2014 combinatorial purged cross-validation (ROBUST / NOT ROBUST)", False)
        self.val_chk_pbo = LabeledCheckbox(checks_section, "PBO \u2014 probability the winner is a backtest-overfitting artifact", False)
        self.val_chk_walkforward = LabeledCheckbox(checks_section, "Walk-Forward \u2014 re-optimise per fold, test out-of-sample", False)
        self.val_chk_wfga = LabeledCheckbox(checks_section, "Walk-Forward GA \u2014 genetic search scored only on unseen folds (slowest)", False)
        self.val_chk_sensitivity = LabeledCheckbox(checks_section, "Sensitivity \u2014 1D parameter sweeps with cliff detection", False)
        self.val_chk_robustness = LabeledCheckbox(checks_section, "Robustness \u2014 parameter stability score (0\u2013100)", False)
        self.val_chk_regime = LabeledCheckbox(checks_section, "Regime \u2014 which market regimes to gate the strategy off in", False)

        advanced = self._advanced_expander(f)
        adv_section = self._section(
            advanced, "Advanced options",
            "Defaults are sensible \u2014 skip this unless you know why you're changing it.",
        )
        self.val_simple_mc_sims = LabeledEntry(adv_section, "Monte Carlo sims", 2000)
        self.val_simple_cpcv_groups = LabeledEntry(adv_section, "CPCV groups (N)", 6)
        self.val_simple_cpcv_test_groups = LabeledEntry(adv_section, "CPCV test groups (k)", 2)
        self.val_simple_sens_pct = LabeledEntry(adv_section, "Sensitivity range (\u00b1 fraction, e.g. 0.5 = \u00b150%)", 0.5)
        self.val_simple_sens_steps = LabeledEntry(adv_section, "Sensitivity steps", 9)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN TICKED CHECKS", self._validate_simple_run_clicked, primary=True).pack(side="left")

        self.val_simple_progress = NeuralProgress(f)
        self.val_simple_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Validate output", "Baseline first, then each ticked check's own numbers \u2014 PASS/FAIL wording only where the check itself computes a verdict.")
        self.val_simple_output = self._log_box(output_section, height=20)

    def _log_val_simple(self, msg: str):
        self.val_simple_output.insert(END, msg + "\n")
        self.val_simple_output.see(END)
        self.root.update_idletasks()

    def _ticked_validation_checks(self, prefix: str) -> set:
        """Reads one page's seven validation checkboxes by attribute
        prefix ("val_chk_" / "champ_chk_") into the dispatch's check-key
        set -- shared so Validate-simple and Champion-simple can never
        drift apart in what a tick means."""
        checks = set()
        if getattr(self, f"{prefix}montecarlo").get():
            checks.add("montecarlo")
        if getattr(self, f"{prefix}cpcv").get():
            checks.add("cpcv")
        if getattr(self, f"{prefix}pbo").get():
            checks.add("pbo")
        if getattr(self, f"{prefix}walkforward").get():
            checks.add("walkforward")
        _wfga_box = getattr(self, f"{prefix}wfga", None)
        if _wfga_box is not None and _wfga_box.get():
            checks.add("wfga")
        if getattr(self, f"{prefix}sensitivity").get():
            checks.add("sensitivity")
        if getattr(self, f"{prefix}robustness").get():
            checks.add("robustness")
        if getattr(self, f"{prefix}regime").get():
            checks.add("regime")
        return checks

    def _validate_simple_run_clicked(self):
        if not self.val_simple_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        checks = self._ticked_validation_checks("val_chk_")
        if not checks:
            messagebox.showwarning("Nothing ticked", "Tick at least one check first.")
            return
        self.val_simple_output.delete("1.0", END)
        self.val_simple_progress.start(10)
        threading.Thread(target=self._validate_simple_run_pipeline, args=(checks,), daemon=True).start()

    # Desktop tick-name -> the shared (web) check key
    _VAL_DESKTOP_TO_SHARED = {
        "montecarlo": "monte_carlo", "cpcv": "cpcv", "pbo": "pbo", "walkforward": "walk_forward",
        "sensitivity": "sensitivity", "robustness": "robustness", "regime": "regime",
    }

    def _validate_simple_run_pipeline(self, checks: set):
        """Runs the SAME shared dispatcher (app.validation.runner) and the
        SAME plain-language report (app.validation.simple_report) the web
        Validate page uses, so the desktop and web apps give identical
        verdicts, identical failure explanations, and the same "what to do
        next" guidance."""
        log = self._log_val_simple
        try:
            df = self.val_simple_context.load_dataframe(log)
            if df is None:
                return
            from app.validation import simple_report
            from app.validation.runner import dispatch_validation_checks
            rules = self.val_simple_context.build_prop_rules()
            risk = build_run_context(self.val_simple_context.build_risk_config(), rules)
            strategy = self._build_strategy()
            shared_checks = [c for c in simple_report.CHECK_ORDER
                             if c in {self._VAL_DESKTOP_TO_SHARED.get(x, x) for x in checks}]
            form = {
                "mc_sims": self.val_simple_mc_sims.get_int(2000),
                "cpcv_groups": self.val_simple_cpcv_groups.get_int(6),
                "cpcv_test_groups": self.val_simple_cpcv_test_groups.get_int(2),
                "pct_range": self.val_simple_sens_pct.get_float(0.5),
                "n_steps": self.val_simple_sens_steps.get_int(9),
            }
            log("Running baseline backtest (hardened risk)...")
            baseline = run_backtest(df, strategy, risk)
            bsum = {
                "trades": len(baseline.trades), "net_profit": baseline.statistics.net_profit,
                "win_rate": baseline.statistics.win_rate, "max_drawdown_pct": baseline.statistics.max_drawdown_pct,
            }
            log(f"Baseline: {bsum['trades']} trades, net {bsum['net_profit']:.2f}.")
            if bsum["trades"] == 0:
                log("Warning: the baseline opened ZERO trades on this data, so most checks cannot run. "
                    "Each failed check below explains why and what to change.")

            def _on_start(check):
                log(f"{simple_report.CHECK_LABELS.get(check, check)}: running...")

            def _on_done(check, result):
                item = simple_report.interpret_check(result)
                log(f"{item['label']}: {item['status'].upper()} -- {item['headline']}")

            results = dispatch_validation_checks(
                df, strategy, risk, rules, shared_checks, form, baseline_trades=baseline.trades,
                on_start=_on_start, on_done=_on_done, log=lambda m: log("  " + str(m)),
            )
            report = simple_report.build_report(results, context={})
            log("")
            log(simple_report.report_to_text(report, bsum))
        except StrategyError as exc:
            log(f"\nStrategy error: {exc}")
        except Exception as exc:
            log("\nUnexpected error:\n" + traceback.format_exc())
            log_crash("Validate (pick your checks)", exc=exc)
        finally:
            self.val_simple_progress.stop()

    # Risk Sweep -- "Risk Is a Strategy Variable" (T58 Quant Trading
    # Masterclass, Lesson 7): re-runs the strategy at several candidate
    # risk-per-trade levels and scores each with the same full-lifecycle
    # prop-survival model the Search Lab/Research Loop already trust
    # (app.optimize.risk_sweep -- a thin desktop wrapper, no numerical
    # logic re-implemented here).
    # -----------------------------------------------------------------------

    def _build_risk_sweep_tab(self):
        f = self._scrollable(self.tab_risksweep)

        self._page_header(
            f,
            "OPTIMIZE / Risk Sweep",
            "\U0001F4CA Risk Sweep",
            "Don't assume 1% risk-per-trade is optimal, and don't optimize for the return "
            "of any ONE trade -- optimize P(reach payout target before hitting a loss "
            "constraint), which a SMALLER risk level can sometimes dramatically improve at "
            "the cost of taking longer. Re-runs the strategy configured on Step 01 at each "
            "candidate risk level (position sizing changes every trade's dollar P&L, so this "
            "can't be approximated by rescaling one backtest's numbers after the fact) and "
            "scores each with the T58 Prop Survival Score.",
        )

        self.risksweep_context = RunContextPanel(self, "Risk Sweep")
        self.risksweep_context.build(f)

        settings = self._section(
            f, "Risk Sweep settings",
            "Leave the default risk levels blank to use the standard sweep "
            "(0.10%, 0.15%, 0.20%, 0.25%, 0.35%, 0.50%, 0.75%, 1.00%).",
            emphasize=True,
        )
        self.risksweep_values = LabeledEntry(
            settings, "Risk levels to test (comma-separated %, blank = default)", "",
        )
        self.risksweep_survival_sims = LabeledEntry(settings, "Survival-simulation sims per level", 1000)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN RISK SWEEP", self._risk_sweep_run_clicked, primary=True).pack(side="left")

        self.risksweep_progress = NeuralProgress(f)
        self.risksweep_progress.pack(fill="x", padx=24, pady=(2, 10))

        result_section = self._section(f, "Best risk level", "Filled in once the sweep finishes.")
        self.risksweep_result_label = Label(
            result_section, text="No run yet.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(11, "bold"), justify="left", wraplength=900, anchor="w",
        )
        self.risksweep_result_label.pack(anchor="w", fill="x", padx=18, pady=(2, 12))

        output_section = self._section(f, "Risk Sweep output", "The full table renders here once the sweep finishes.")
        _risksweep_output_frame = Frame(output_section, bg=PANEL)
        self.risksweep_output = Text(
            _risksweep_output_frame, height=20, wrap="none", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _risksweep_output_scroll = ttk.Scrollbar(
            _risksweep_output_frame, orient="vertical", command=self.risksweep_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.risksweep_output.configure(yscrollcommand=_risksweep_output_scroll.set)
        self.risksweep_output.pack(side="left", fill="both", expand=True)
        _risksweep_output_scroll.pack(side="right", fill="y")
        _risksweep_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.risksweep_output)

    def _risk_sweep_run_clicked(self):
        if not self.risksweep_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        self.risksweep_output.delete("1.0", END)
        self.risksweep_result_label.config(text="Running...", fg=TEXT_DIM)
        self.risksweep_progress.start(10)
        threading.Thread(target=self._risk_sweep_run_pipeline, daemon=True).start()

    def _risk_sweep_run_pipeline(self):
        from app.optimize.risk_sweep import run_risk_sweep

        def _log(msg: str):
            self.risksweep_output.insert(END, msg + "\n")
            self.risksweep_output.see(END)
            self.root.update_idletasks()

        try:
            df = self.risksweep_context.load_dataframe(_log)
            if df is None:
                return
            risk = self.risksweep_context.build_risk_config()
            rules = self.risksweep_context.build_prop_rules()
            strategy_builder = self._build_strategy

            raw_levels = self.risksweep_values.get_str().strip()
            risk_values = None
            if raw_levels:
                try:
                    risk_values = [float(v.strip()) for v in raw_levels.split(",") if v.strip()]
                except ValueError:
                    _log("Couldn't parse the risk levels list -- using the default sweep instead.")
                    risk_values = None

            survival_cfg = PropSurvivalConfig(n_simulations=self.risksweep_survival_sims.get_int(1000))

            _log("Running Risk Sweep...")
            result = run_risk_sweep(df, strategy_builder, risk, rules, risk_values=risk_values, survival_cfg=survival_cfg)
            self.risksweep_output.delete("1.0", END)
            self.risksweep_output.insert(END, result.render_table())
            for note in result.notes:
                self.risksweep_output.insert(END, f"\n\nNote: {note}")

            if result.best_point:
                self.risksweep_result_label.config(
                    text=(
                        f"Best: {result.best_point.risk_value:.2f}% per trade  --  "
                        f"T58 Prop Survival Score {result.best_point.prop_survival_score:.1f}/100  --  "
                        f"pass {result.best_point.probability_pass_evaluation:.1f}%  --  "
                        f"payout {result.best_point.probability_first_payout:.1f}%"
                    ),
                    fg=GREEN,
                )
            else:
                self.risksweep_result_label.config(text="No risk level produced any trades.", fg=RED)
        except Exception as exc:
            _log("\nUnexpected error:\n" + traceback.format_exc())
            self.risksweep_result_label.config(text="Failed -- see log.", fg=RED)
            log_crash("Risk Sweep", exc=exc)
        finally:
            self.risksweep_progress.stop()

    # -----------------------------------------------------------------------
    # Compare Strategies -- the direct, visual side-by-side comparison
    # screen (app.strategy.compare): pick 2-4 saved CODE strategies
    # (python/pinescript/mql5 -- not "manual" configs, which have no
    # library file to load), backtest each fresh against the same
    # dataset/risk config, and see net profit / eval-pass probability /
    # Prop Fitness in one place.
    # -----------------------------------------------------------------------

    def _build_compare_tab(self):
        f = self._scrollable(self.tab_compare)

        self._page_header(
            f,
            "DEPLOYMENT / Compare Strategies",
            "\u2696 Compare Strategies",
            "Pick 2-4 saved strategies (Python, PineScript, or MQL5 -- not a \"manual\" "
            "config, which has no library file to load) and see them backtested fresh "
            "against the SAME dataset/risk config side by side: net profit, Monte Carlo "
            "eval-pass probability, and a simplified Prop Fitness score. That Prop Fitness "
            "number is directly comparable to the others on THIS screen (all computed the "
            "same simplified way) but will read differently from one out of a full "
            "Evolution Lab run, which also factors in parameter-neighborhood robustness and "
            "walk-forward consistency -- this quick view deliberately skips both.",
        )

        self.compare_context = RunContextPanel(self, "Compare Strategies")
        self.compare_context.build(f)

        picker_section = self._section(
            f, "Pick 2-4 saved strategies",
            "Ctrl/Cmd-click or Shift-click to select several. Refresh if you just saved a new one.",
            emphasize=True,
        )
        picker_btn_row = Frame(picker_section, bg=PANEL)
        picker_btn_row.pack(anchor="w", padx=18, pady=(2, 6))
        self._button(picker_btn_row, "REFRESH LIST", self._refresh_compare_list).pack(side="left")

        compare_list_frame = Frame(picker_section, bg=PANEL)
        compare_list_frame.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        compare_list_frame.columnconfigure(0, weight=1)
        self.compare_listbox = Listbox(
            compare_list_frame, height=10, selectmode="extended", exportselection=False,
            bg=PANEL_3, fg=TEXT, activestyle="none", relief="flat", bd=0,
            highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9),
        )
        compare_scrollbar = ttk.Scrollbar(
            compare_list_frame, orient="vertical", command=self.compare_listbox.yview, style="T58.Vertical.TScrollbar",
        )
        self.compare_listbox.config(yscrollcommand=compare_scrollbar.set)
        self._bind_isolated_wheel(self.compare_listbox)
        self.compare_listbox.grid(row=0, column=0, sticky="nsew")
        compare_scrollbar.grid(row=0, column=1, sticky="ns")
        self._compare_listbox_items: list[tuple[str, str]] = []  # [(strategy_type, filename), ...] parallel to listbox rows

        settings = self._section(f, "Compare settings", emphasize=True)
        self.compare_mc_sims = LabeledEntry(settings, "Monte Carlo sims per candidate", 1000)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN COMPARE", self._compare_run_clicked, primary=True).pack(side="left")

        self.compare_progress = NeuralProgress(f)
        self.compare_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Comparison", "Each candidate's own summary, in the order selected.")
        _compare_output_frame = Frame(output_section, bg=PANEL)
        self.compare_output = Text(
            _compare_output_frame, height=20, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _compare_output_scroll = ttk.Scrollbar(
            _compare_output_frame, orient="vertical", command=self.compare_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.compare_output.configure(yscrollcommand=_compare_output_scroll.set)
        self.compare_output.pack(side="left", fill="both", expand=True)
        _compare_output_scroll.pack(side="right", fill="y")
        _compare_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.compare_output)

        self._refresh_compare_list()

    def _refresh_compare_list(self):
        self.compare_listbox.delete(0, END)
        self._compare_listbox_items = []
        try:
            strategies = list_saved_strategies()
        except Exception:
            strategies = []
        # Code strategies only (python/pinescript/mql5) -- "manual" configs
        # have no library file for compare_strategies() to load.
        for s in strategies:
            if s.strategy_type not in ("python", "pinescript", "mql5"):
                continue
            self.compare_listbox.insert(END, f"[{s.strategy_type:10s}] {s.name}  ({s.status_display})")
            self._compare_listbox_items.append((s.strategy_type, s.name))

    def _compare_run_clicked(self):
        sel = self.compare_listbox.curselection()
        if not (2 <= len(sel) <= 4):
            messagebox.showwarning("Select 2-4 strategies", "Select 2 to 4 strategies above (Ctrl/Cmd-click or Shift-click) to compare.")
            return
        if not self.compare_context.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV above (this tab's own Market Data section).")
            return
        self.compare_output.delete("1.0", END)
        self.compare_progress.start(10)
        threading.Thread(target=self._compare_run_pipeline, args=(list(sel),), daemon=True).start()

    def _compare_run_pipeline(self, selected_indices: list[int]):
        from app.strategy.compare import CompareError, compare_strategies

        def _log(msg: str):
            self.compare_output.insert(END, msg + "\n")
            self.compare_output.see(END)
            self.root.update_idletasks()

        try:
            df = self.compare_context.load_dataframe(_log)
            if df is None:
                return
            risk = self.compare_context.build_risk_config()
            rules = self.compare_context.build_prop_rules()
            candidates = [self._compare_listbox_items[i] for i in selected_indices]

            _log(f"Comparing {len(candidates)} strategies against {len(df)} bars...")
            results = compare_strategies(
                candidates, df, risk, rules,
                mc_n_simulations=self.compare_mc_sims.get_int(1000),
            )
            self.compare_output.delete("1.0", END)
            for r in results:
                self.compare_output.insert(END, f"=== [{r.strategy_type}] {r.filename} ===\n")
                if not r.ok:
                    self.compare_output.insert(END, f"  FAILED: {r.error}\n\n")
                    continue
                self.compare_output.insert(
                    END,
                    f"  Net profit: {r.stats.get('net_profit', 0):,.2f}   "
                    f"Trades: {r.stats.get('total_trades', '?')}   "
                    f"Win rate: {r.stats.get('win_rate', 0):.1f}%\n"
                    f"  Eval-pass probability: {r.eval_pass_probability_pct:.1f}%   "
                    f"First-payout probability: {r.first_payout_probability_pct:.1f}%\n"
                    f"  Prop Fitness (simplified{'*' if r.prop_fitness_is_simplified else ''}): "
                    f"{r.prop_fitness_score:.3f}\n\n"
                )
            self.compare_output.insert(
                END,
                "\n* Simplified: doesn't factor in parameter-neighborhood robustness or "
                "walk-forward OOS consistency -- comparable to the other rows here only, not "
                "to an Evolution Lab leaderboard score.",
            )
        except CompareError as exc:
            _log(f"\n{exc}")
        except Exception as exc:
            _log("\nUnexpected error:\n" + traceback.format_exc())
            log_crash("Compare Strategies", exc=exc)
        finally:
            self.compare_progress.stop()

    def _build_validate_hub_tab(self):
        f = self._scrollable(self.tab_validate_hub)
        self._validate_hub_inner = f
        self._populate_validate_hub(f)

    def _replay_tour(self):
        """Dashboard 'Take the tour' button -- starts the guided tour on demand."""
        try:
            from app.ui.onboarding_tour import OnboardingTour
            existing = getattr(self, "_tour", None)
            if existing is not None and getattr(existing, "card", None) is not None:
                return
            self._tour = OnboardingTour(self)
            self._tour.start()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Tour", f"Could not start the tour: {exc}")

    def _build_start_here_tabs(self):
        from app.ui import extra_tabs
        extra_tabs.build_start_here_tabs(self)

    def _build_strategy_library_tab(self):
        from app.ui import extra_tabs
        extra_tabs.build_strategy_library_tab(self)

    def _build_replay_tab(self):
        from app.ui import extra_tabs
        extra_tabs.build_replay_tab(self)

    def _refresh_validate_hub(self):
        """Rebuild the Validate Start Here content from what the validation
        tabs have recorded so far -- called every time the page is shown, so
        the checklist is never a stale startup snapshot."""
        f = getattr(self, "_validate_hub_inner", None)
        if f is None:
            return
        try:
            for child in list(f.winfo_children()):
                child.destroy()
            self._populate_validate_hub(f)
        except Exception:
            pass

    def _populate_validate_hub(self, f):
        """Desktop twin of the web /validate page (validate_hub.html): intro
        card, the current strategy's checks-run summary, the validation
        checklist with Run / Re-run buttons, the "once validated" actions, and
        the "Everything in Validate" card grid. Every card here opens its
        real tab via _show_page -- there are no URL links to go stale."""
        self._page_header(
            f, "", "Validate \u2014 Start Here",
            "Is this strategy actually robust, or did it just get lucky on one backtest? This checklist is "
            "built entirely from validation tools you've already run -- nothing here is re-computed or guessed.",
        )

        intro = self._section(f, "What this section is for")
        Label(
            intro,
            text=(
                "Validate exists to catch a strategy that only looked good because of luck on one backtest. "
                "The checklist below aggregates results from WFO, WFGA, CPCV, PBO, Sensitivity, and the Regime "
                "Survival Matrix for whichever strategy is marked current -- it never re-runs or guesses at "
                "those results, so run the checks you need first if the checklist below shows them as pending."
            ),
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), wraplength=900, justify="left", anchor="w",
        ).pack(anchor="w", padx=14, pady=(0, 10))


        current = None
        try:
            current = strategy_state.get_current_strategy()
        except Exception:
            pass

        # Which nav tab each checklist kind opens (web: VALIDATION_HREFS).
        nav_for_kind = {
            "wfo": "wfo", "wfga": "wfga", "cpcv": "cpcv",
            "sensitivity": "sensitivity", "regime_matrix": "regimematrix",
        }

        if not current:
            empty = self._section(f, "Current strategy", emphasize=True)
            Label(
                empty,
                text="No strategy is marked current yet. Go to the Dashboard, find a strategy in the scorecard, "
                     "and click \"Set as current\" to start tracking its validation checklist here.",
                bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), wraplength=880, justify="left", anchor="w",
            ).pack(anchor="w", padx=14, pady=(2, 8))
            self._button(empty, "OPEN DASHBOARD", lambda: self._show_page("dashboard")).pack(anchor="w", padx=14, pady=(0, 12))
        else:
            lib_kwargs = {
                "library_type": current.get("library_type", ""),
                "library_filename": current.get("library_filename", ""),
            }
            checklist = {}
            score = {"ran_count": 0, "total_count": len(strategy_state.VALIDATION_KINDS), "decided_count": 0}
            try:
                checklist = strategy_state.get_checklist(current["strategy_name"], current["instrument"], **lib_kwargs)
                score = strategy_state.robustness_score(current["strategy_name"], current["instrument"], **lib_kwargs)
            except Exception:
                pass

            hero = self._section(f, "Validating", emphasize=True)
            Label(
                hero, text=f"{current['strategy_name']}  \u00b7  {current['instrument']}",
                bg=PANEL, fg=TEXT, font=_safe_font(13, "bold"), anchor="w",
            ).pack(anchor="w", padx=14, pady=(2, 8))
            stats = Frame(hero, bg=PANEL)
            stats.pack(anchor="w", padx=14, pady=(0, 12))
            left = Frame(stats, bg=PANEL)
            left.pack(side="left", padx=(0, 28))
            Label(left, text=f"{score.get('ran_count', 0)}/{score.get('total_count', 0)}", bg=PANEL, fg=GREEN,
                  font=_safe_font(20, "bold")).pack(anchor="w")
            Label(left, text="CHECKS RUN", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8, "bold")).pack(anchor="w")
            if score.get("decided_count"):
                pct = score.get("pct_passed_of_decided", 0)
                right = Frame(stats, bg=PANEL)
                right.pack(side="left")
                Label(right, text=f"{pct}%", bg=PANEL, fg=(GREEN if pct >= 50 else RED),
                      font=_safe_font(20, "bold")).pack(anchor="w")
                Label(right, text=f"PASSED (OF {score['decided_count']} WITH A VERDICT)", bg=PANEL, fg=TEXT_MUTED,
                      font=_safe_font(8, "bold")).pack(anchor="w")

            cl = self._section(f, "Validation checklist")
            for kind in strategy_state.VALIDATION_KINDS:
                check = checklist.get(kind, {"ran": False})
                ran, passed = bool(check.get("ran")), check.get("passed")
                if not ran:
                    dot, dot_color, detail = "\u25cb", TEXT_DIM, "Not run yet for this strategy."
                elif passed is None:
                    dot, dot_color = "\u25cf", VIOLET
                    detail = f"Ran \u2014 {check.get('summary') or 'no strict pass/fail verdict for this tool'}."
                elif passed:
                    dot, dot_color, detail = "\u2713", GREEN, f"Passed \u2014 {check.get('summary')}."
                else:
                    dot, dot_color, detail = "\u2715", RED, f"Did not pass \u2014 {check.get('summary')}."
                row = Frame(cl, bg=PANEL)
                row.pack(fill="x", padx=14, pady=(2, 2))
                Frame(cl, bg=BORDER, height=1).pack(fill="x", padx=14)
                Label(row, text=dot, bg=PANEL, fg=dot_color, font=_safe_font(11, "bold"), width=3).pack(side="left")
                textcol = Frame(row, bg=PANEL)
                textcol.pack(side="left", fill="x", expand=True, padx=(4, 8), pady=8)
                Label(textcol, text=strategy_state.VALIDATION_LABELS.get(kind, kind), bg=PANEL, fg=TEXT,
                      font=_safe_font(10, "bold"), anchor="w").pack(anchor="w")
                Label(textcol, text=detail, bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), anchor="w",
                      justify="left", wraplength=620).pack(anchor="w", pady=(2, 0))
                self._button(
                    row, "Re-run" if ran else "Run", lambda k=nav_for_kind.get(kind, "validatehub"): self._show_page(k),
                    primary=not ran,
                ).pack(side="right", pady=6)

            done = self._section(f, "Once validated")
            actions = Frame(done, bg=PANEL)
            actions.pack(anchor="w", padx=14, pady=(2, 12))
            self._button(actions, "Multi-Asset Portfolio", lambda: self._show_page("portfolio")).pack(side="left", padx=(0, 8))
            self._button(actions, "Multi-Strategy Ensemble", lambda: self._show_page("ensemble")).pack(side="left", padx=(0, 8))
            self._button(actions, "Forward Test \u2192", lambda: self._show_page("forwardtest")).pack(side="left")

        everything = self._section(f, "Everything in Validate")
        self._link_card_grid(everything, [
            ("Start Here", "This page.", "validatehub"),
            ("Walk-Forward Optimization", "Trains and tests on rolling, non-overlapping windows through time.", "wfo"),
            ("Walk-Forward GA", "Walk-forward validation using the genetic-algorithm search.", "wfga"),
            ("CPCV", "Combinatorial purged cross-validation -- many train/test path combinations.", "cpcv"),
            ("PBO", "Probability of backtest overfitting.", "pbo"),
            ("Sensitivity", "How much results change with small parameter perturbations.", "sensitivity"),
            ("Parameter Stability / Robustness Map",
             "Visualizes the whole neighborhood around your parameters, not just one point.", "paramrobustness"),
            ("Regime Survival Matrix",
             "Does the strategy survive across different market regimes, not just one period?", "regimematrix"),
            ("Monte Carlo / Payout Probability", "Simulated odds of actually reaching a funded payout.", "montecarlo"),
        ])
