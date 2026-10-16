"""Payout probability, bankroll survival and prop-firm recommender tabs."""
from __future__ import annotations

import os
import threading
import traceback
import webbrowser
from app.backtest.engine import run_backtest
from app.backtest.risk import with_prop_safety_defaults
from app.monte_carlo.bankroll import BankrollConfig, simulate_bankroll_survival
from app.monte_carlo.engine import MonteCarloConfig
from app.prop.presets import list_presets as list_prop_firm_presets
from app.prop.recommender import recommend_prop_firms, render_recommendation_table
from app.prop.scaling import ScalingPlan, run_scaling_stress_test
from app.prop.survival_engine import PropSurvivalConfig, ResetEconomics, run_prop_survival_analysis
from app.reports.survival_report import generate_survival_report
from app.strategy.base import StrategyError
from tkinter import BooleanVar, END, Frame, Text, messagebox, ttk
from app.ui.common import OUTPUT_DIR, _strategy_display_name
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui.widgets.cards import NeuralProgress
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
BORDER = ""
TEXT = ""
LOG_BG = ""
theme.bind(globals())


class PayoutMixin:
    """Payout probability, bankroll survival and prop-firm recommender tabs."""


    # -----------------------------------------------------------------------
    # Tab -- Payout Probability (full-lifecycle prop survival simulator)
    # -----------------------------------------------------------------------

    def _build_payout_probability_tab(self):
        f = self._scrollable(self.tab_payout)

        self._page_header(
            f,
            "TEST / Payout Probability",
            "Full-Lifecycle Survival Simulator",
            "Goes beyond a single pass-probability number: simulates thousands of complete account "
            "lifecycles (Start -> Evaluation -> Funded -> Payout #1 -> Payout #2 -> ... -> "
            "Reset/Continue) and reports the actual funnel -- how many of those simulated accounts "
            "made it through EACH stage, not just whether they passed the eval. Uses the strategy, "
            "data, prop rules, and risk settings already configured in Steps 01-04. Backed entirely "
            "by app.prop.survival_engine -- the same resampling machinery as 05 Run & Report's own "
            "Monte Carlo simulation, just reshaped into the lifecycle view a trader actually thinks in.",
        )

        settings = self._section(
            f, "Simulation settings",
            "Reset/retry economics model what it actually costs, out of pocket, to keep re-attempting "
            "the evaluation after a failure -- and what you actually keep after the firm's profit split.",
            emphasize=True,
        )
        self.payout_n_sims = LabeledEntry(settings, "Number of simulated account lifetimes", 5000)
        self.payout_max_tracked = LabeledEntry(settings, "Payout milestones to track (e.g. 5 = through Payout #5)", 5)
        self.payout_funding_approval_pct = LabeledEntry(
            settings, "Funding approval probability (%) -- 100 = every eval pass becomes funded immediately", 100,
        )
        self.payout_eval_fee = LabeledEntry(settings, "Evaluation fee ($)", 0)
        self.payout_reset_fee = LabeledEntry(settings, "Reset fee ($, blank = same as evaluation fee)", 0)
        self.payout_profit_split = LabeledEntry(settings, "Trader's profit split (%)", 80)
        self.payout_max_attempts = LabeledEntry(settings, "Max lifetime attempts to fund", 3)

        scaling_section = self._section(
            f, "Account-scaling stress test (optional)",
            "Models a funded account scaling up after a run of consecutive payouts -- a common shape "
            "at several modern prop firms (e.g. +25% account size every 4 payouts, capped at 4x). "
            "Additive only: leave unchecked to keep the plain (no-scaling) result above; see "
            "app.prop.scaling's module docstring for the modeling assumptions.",
        )
        self.payout_scaling_enabled = LabeledCheckbox(scaling_section, "Enable account-scaling stress test", default=False)
        self.payout_scale_payouts_per = LabeledEntry(scaling_section, "Consecutive payouts per scale-up", 4)
        self.payout_scale_multiplier = LabeledEntry(scaling_section, "Scale multiplier (e.g. 1.25 = +25%)", 1.25)
        self.payout_scale_max_multiple = LabeledEntry(scaling_section, "Max scale multiple (cap, e.g. 4 = never exceed 4x)", 4.0)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN PAYOUT PROBABILITY SIMULATION", self._payout_run_clicked, primary=True).pack(side="left")
        self.open_payout_report_btn = self._button(button_row, "OPEN REPORT", self._open_payout_report)
        self.open_payout_report_btn.config(state="disabled")
        self.open_payout_report_btn.pack(side="left", padx=8)

        self.payout_progress = NeuralProgress(f)
        self.payout_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Payout Probability output", "Live progress + the funnel table.")
        _payout_output_frame = Frame(output_section, bg=PANEL)
        self.payout_output = Text(
            _payout_output_frame, height=24, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _payout_output_scroll = ttk.Scrollbar(
            _payout_output_frame, orient="vertical", command=self.payout_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.payout_output.configure(yscrollcommand=_payout_output_scroll.set)
        self.payout_output.pack(side="left", fill="both", expand=True)
        _payout_output_scroll.pack(side="right", fill="y")
        _payout_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.payout_output)

        self._last_payout_html_path = None

        # -----------------------------------------------------------------
        # Bankroll / EV Survival -- a different question than the funnel
        # above: given a REAL, finite amount of money set aside for buying
        # evaluations/resets, what fraction of plausible futures reach a
        # payout before it runs out? Backed by app.monte_carlo.bankroll,
        # which reuses this tab's reset economics (fee/split/max attempts
        # above) but chains ALL of a simulated life's attempts within ONE
        # resampled trade ordering (ideally block bootstrap, so a real bad
        # stretch of trades stays clustered together), instead of a fresh
        # independent resample per attempt the way the funnel above does.
        # -----------------------------------------------------------------
        bankroll_section = self._section(
            f, "Bankroll / EV Survival (chained reset-on-breach, priced against a real bankroll)",
            "Reuses the evaluation/reset fee and profit split above, but answers a sharper "
            "question: with $X actually set aside for buying attempts, what's the probability "
            "of reaching a payout before going broke -- and what's the expected net profit? "
            "Each simulated life draws ONE resampled trade ordering and chains every mechanical "
            "rebuy within that SAME ordering (app.prop.simulator.simulate_account's "
            "reset_on_breach), so a real bad stretch of trades stays clustered together the way "
            "it would in one continuous plausible future, rather than being redrawn fresh (and "
            "so smeared away) on every attempt.",
            emphasize=True,
        )
        self.bankroll_starting_amount = LabeledEntry(bankroll_section, "Starting bankroll ($)", 1000)
        self.bankroll_n_sims = LabeledEntry(bankroll_section, "Number of simulated bankroll lifetimes", 5000)
        self.bankroll_method = LabeledCombo(
            bankroll_section, "Resampling method",
            ["block_bootstrap", "bootstrap", "shuffle"], default="block_bootstrap",
        )
        self.bankroll_block_size = LabeledEntry(bankroll_section, "Block size (block_bootstrap only)", 5)
        self.bankroll_stop_after_first_payout = LabeledCheckbox(
            bankroll_section,
            "Stop each simulated life at its first payout (measures 'ever get paid before going "
            "broke', not lifetime return -- leave unchecked for the fuller lifetime-return question)",
            default=False,
        )

        bankroll_button_row = Frame(f, bg=BG)
        bankroll_button_row.pack(fill="x", padx=24, pady=10)
        self._button(
            bankroll_button_row, "RUN BANKROLL / EV SURVIVAL", self._bankroll_run_clicked, primary=True,
        ).pack(side="left")

        self.bankroll_progress = NeuralProgress(f)
        self.bankroll_progress.pack(fill="x", padx=24, pady=(2, 10))

        bankroll_output_section = self._section(
            f, "Bankroll / EV Survival output", "Live progress + the ruin/payout/EV summary.",
        )
        _bankroll_output_frame = Frame(bankroll_output_section, bg=PANEL)
        self.bankroll_output = Text(
            _bankroll_output_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _bankroll_output_scroll = ttk.Scrollbar(
            _bankroll_output_frame, orient="vertical", command=self.bankroll_output.yview,
            style="T58.Vertical.TScrollbar",
        )
        self.bankroll_output.configure(yscrollcommand=_bankroll_output_scroll.set)
        self.bankroll_output.pack(side="left", fill="both", expand=True)
        _bankroll_output_scroll.pack(side="right", fill="y")
        _bankroll_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.bankroll_output)

    def _log_payout(self, msg: str):
        self.payout_output.insert(END, msg + "\n")
        self.payout_output.see(END)

    def _open_payout_report(self):
        if self._last_payout_html_path:
            webbrowser.open(f"file://{self._last_payout_html_path.resolve()}")

    def _payout_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 2.")
            return
        self.payout_output.delete("1.0", END)
        self.payout_progress.start(10)
        threading.Thread(target=self._payout_run_pipeline, daemon=True).start()

    def _payout_run_pipeline(self):
        try:
            df = self._load_df_for_page(self._log_payout)
            if df is None:
                return
            strategy = self._build_strategy()
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
            adaptive_risk = self._build_adaptive_risk_config()

            self._log_payout("Running historical backtest to obtain a real trade sequence...")
            bt_result = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)
            if not bt_result.trades:
                self._log_payout(
                    "\nNo trades were generated by this strategy over the given data -- there is "
                    "nothing to run a lifecycle simulation on."
                )
                return
            self._log_payout(f"  {len(bt_result.trades)} trades.")

            reset_fee_raw = self.payout_reset_fee.get_float(0)
            econ = ResetEconomics(
                evaluation_fee=self.payout_eval_fee.get_float(0),
                reset_fee=(reset_fee_raw if reset_fee_raw > 0 else None),
                profit_split_pct=self.payout_profit_split.get_float(80),
                max_attempts=self.payout_max_attempts.get_int(3),
            )
            cfg = PropSurvivalConfig(
                n_simulations=self.payout_n_sims.get_int(5000),
                max_payouts_tracked=self.payout_max_tracked.get_int(5),
                funding_approval_probability=self.payout_funding_approval_pct.get_float(100),
                reset_economics=econ,
            )

            self._log_payout(
                f"Simulating {cfg.n_simulations:,} account lifetimes "
                f"(tracking payouts #1-#{cfg.max_payouts_tracked})..."
            )
            result = run_prop_survival_analysis(bt_result.trades, rules, cfg)

            self._log_payout("")
            self._log_payout(result.funnel.render_table(_strategy_display_name(strategy)))
            self._log_payout("")
            self._log_payout(f"Prop Survival Score: {result.prop_survival_score:.0f} / 100")
            self._log_payout(
                f"Net positive after resets: {result.reset_economics.probability_net_positive_after_resets:.1f}% "
                f"(expected: ${result.reset_economics.expected_net_profit_after_resets:,.2f})"
            )
            for note in result.notes:
                self._log_payout(f"\nNOTE: {note}")

            if self.payout_scaling_enabled.get():
                try:
                    plan = ScalingPlan(
                        payouts_per_scale=self.payout_scale_payouts_per.get_int(4),
                        scale_multiplier=self.payout_scale_multiplier.get_float(1.25),
                        max_scale_multiple=self.payout_scale_max_multiple.get_float(4.0),
                    )
                    self._log_payout("\nRunning account-scaling stress test...")
                    scaling_result = run_scaling_stress_test(bt_result.trades, rules, plan)
                    self._log_payout(
                        f"  Probability of any scale-up: {scaling_result.probability_any_scale_up:.1f}%\n"
                        f"  Median final account size: ${scaling_result.median_final_account_size:,.0f}\n"
                        f"  Expected total payout -- no scaling:   ${scaling_result.expected_total_payout_unscaled:,.0f}\n"
                        f"  Expected total payout -- with scaling: ${scaling_result.expected_total_payout_scaled:,.0f}\n"
                        f"  Expected payout uplift: {scaling_result.payout_uplift_pct:.1f}%"
                    )
                except ValueError as exc:
                    self._log_payout(f"\nAccount-scaling stress test skipped: {exc}")

            instrument = (
                os.path.basename(self.csv_paths[0]) if len(self.csv_paths) == 1
                else " + ".join(os.path.basename(p) for p in self.csv_paths)
            )
            paths = generate_survival_report(
                OUTPUT_DIR / "payout_probability", result,
                _strategy_display_name(strategy), instrument, rules, cfg,
            )
            self._last_payout_html_path = paths["html"]
            self.open_payout_report_btn.config(state="normal")
            self._log_payout("\nDone. Report written to:")
            for k, p in paths.items():
                self._log_payout(f"  {k}: {p}")
        except StrategyError as exc:
            self._log_payout(f"\nStrategy error: {exc}")
        except Exception:
            self._log_payout("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.payout_progress.stop()

    # -----------------------------------------------------------------------
    # Bankroll / EV Survival (see the section built in _build_payout_probability_tab)
    # -----------------------------------------------------------------------

    def _bankroll_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 2.")
            return
        self.bankroll_output.delete("1.0", END)
        self.bankroll_progress.start(10)
        threading.Thread(target=self._bankroll_run_pipeline, daemon=True).start()

    def _log_bankroll(self, msg: str):
        self.bankroll_output.insert(END, msg + "\n")
        self.bankroll_output.see(END)

    def _bankroll_run_pipeline(self):
        try:
            df = self._load_df_for_page(self._log_bankroll)
            if df is None:
                return
            strategy = self._build_strategy()
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
            adaptive_risk = self._build_adaptive_risk_config()

            self._log_bankroll("Running historical backtest to obtain a real trade sequence...")
            bt_result = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)
            if not bt_result.trades:
                self._log_bankroll(
                    "\nNo trades were generated by this strategy over the given data -- there is "
                    "nothing to run a bankroll simulation on."
                )
                return
            self._log_bankroll(f"  {len(bt_result.trades)} trades.")

            # Reuses this SAME tab's reset-economics fields (fee/split/max
            # attempts) so the two sections above/below never show two
            # different prices for the same reset.
            reset_fee_raw = self.payout_reset_fee.get_float(0)
            econ = ResetEconomics(
                evaluation_fee=self.payout_eval_fee.get_float(0),
                reset_fee=(reset_fee_raw if reset_fee_raw > 0 else None),
                profit_split_pct=self.payout_profit_split.get_float(80),
                max_attempts=self.payout_max_attempts.get_int(3),
            )
            cfg = BankrollConfig(
                starting_bankroll=self.bankroll_starting_amount.get_float(1000),
                reset_economics=econ,
                n_simulations=self.bankroll_n_sims.get_int(5000),
                method=self.bankroll_method.get_str().strip() or "block_bootstrap",
                block_size=self.bankroll_block_size.get_int(5),
                stop_after_first_payout=self.bankroll_stop_after_first_payout.get(),
            )

            self._log_bankroll(
                f"Simulating {cfg.n_simulations:,} bankroll lifetimes starting from "
                f"${cfg.starting_bankroll:,.0f} (method={cfg.method})..."
            )
            result = simulate_bankroll_survival(bt_result.trades, rules, cfg)

            self._log_bankroll("")
            self._log_bankroll(f"  Probability of reaching a payout before ruin: {result.probability_reach_first_payout:.1f}%")
            self._log_bankroll(f"  Probability of ruin before any payout:        {result.probability_ruin_before_first_payout:.1f}%")
            self._log_bankroll(f"  Probability of exhausting attempts, no payout: {result.probability_exhausted_attempts_without_payout:.1f}%")
            self._log_bankroll("")
            self._log_bankroll(f"  Expected net profit:  ${result.expected_net_profit:,.2f}")
            self._log_bankroll(f"  Median net profit:    ${result.median_net_profit:,.2f}")
            self._log_bankroll(f"  Probability net positive: {result.probability_net_positive:.1f}%")
            self._log_bankroll("")
            self._log_bankroll(f"  Expected attempts used: {result.expected_attempts_used:.1f} (median {result.median_attempts_used:.0f})")
            self._log_bankroll(f"  Expected fees paid:     ${result.expected_fees_paid:,.2f}")
            self._log_bankroll(f"  Expected gross payouts: ${result.expected_gross_payouts:,.2f}")
            self._log_bankroll(f"  Median bankroll low point: ${result.median_bankroll_low_point:,.2f}")
            self._log_bankroll(f"  Worst bankroll low point:  ${result.worst_bankroll_low_point:,.2f}")
            self._log_bankroll("")
            self._log_bankroll(result.methodology_note)
        except StrategyError as exc:
            self._log_bankroll(f"\nStrategy error: {exc}")
        except ValueError as exc:
            self._log_bankroll(f"\n{exc}")
        except Exception:
            self._log_bankroll("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.bankroll_progress.stop()

    # -----------------------------------------------------------------------
    # Tab -- Prop-Firm Recommender (reverse of Payout Probability: given a
    # strategy's own trade sequence, score it against EVERY prop-firm
    # preset and rank them)
    # -----------------------------------------------------------------------

    def _build_prop_recommender_tab(self):
        f = self._scrollable(self.tab_prop_recommender)

        self._page_header(
            f,
            "TEST / Prop-Firm Recommender",
            "Prop-Firm Recommender",
            "The reverse of Payout Probability: runs your strategy's own trade sequence against "
            "EVERY prop-firm preset (or a chosen subset) and returns a ranked \"which firm's rules "
            "does this strategy actually fit best\" table, instead of checking one firm's rules by "
            "hand, one at a time. Uses the strategy, data, and risk settings already configured in "
            "Steps 01/02/04 -- prop rules come from the firm checklist below, not from Step 03.",
        )

        settings = self._section(
            f, "Firms to score against",
            "Leave every box unchecked to score against the FULL preset catalog. Check specific "
            "firms to narrow the comparison. See each preset's \"as of\" date -- verify against the "
            "firm's own current rules page before relying on this for a real evaluation.",
            emphasize=True,
        )
        self.prop_rec_firm_vars: dict[str, BooleanVar] = {}
        for p in list_prop_firm_presets():
            cb = LabeledCheckbox(settings, f"{p.label}  (as of {p.as_of})", default=False)
            self.prop_rec_firm_vars[p.key] = cb.var

        sim_section = self._section(f, "Simulation settings", "")
        self.prop_rec_n_sims = LabeledEntry(sim_section, "Simulations per firm", 2000)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN PROP-FIRM RECOMMENDER", self._prop_rec_run_clicked, primary=True).pack(side="left")

        self.prop_rec_progress = NeuralProgress(f)
        self.prop_rec_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Prop-Firm Recommender output", "Live progress + the ranked table.")
        _prop_rec_output_frame = Frame(output_section, bg=PANEL)
        self.prop_rec_output = Text(
            _prop_rec_output_frame, height=24, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _prop_rec_output_scroll = ttk.Scrollbar(
            _prop_rec_output_frame, orient="vertical", command=self.prop_rec_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.prop_rec_output.configure(yscrollcommand=_prop_rec_output_scroll.set)
        self.prop_rec_output.pack(side="left", fill="both", expand=True)
        _prop_rec_output_scroll.pack(side="right", fill="y")
        _prop_rec_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.prop_rec_output)

    def _log_prop_rec(self, msg: str):
        self.prop_rec_output.insert(END, msg + "\n")
        self.prop_rec_output.see(END)

    def _prop_rec_run_clicked(self):
        if not self.csv_paths:
            messagebox.showwarning("Missing data", "Please select a market data CSV in Step 2.")
            return
        self.prop_rec_output.delete("1.0", END)
        self.prop_rec_progress.start(10)
        threading.Thread(target=self._prop_rec_run_pipeline, daemon=True).start()

    def _prop_rec_run_pipeline(self):
        try:
            df = self._load_df_for_page(self._log_prop_rec)
            if df is None:
                return
            strategy = self._build_strategy()
            risk = self._build_risk_config()
            adaptive_risk = self._build_adaptive_risk_config()

            self._log_prop_rec("Running historical backtest to obtain a real trade sequence...")
            bt_result = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)
            if not bt_result.trades:
                self._log_prop_rec(
                    "\nNo trades were generated by this strategy over the given data -- there is "
                    "nothing to score against prop-firm rule sets."
                )
                return
            self._log_prop_rec(f"  {len(bt_result.trades)} trades.")

            selected_keys = [key for key, var in self.prop_rec_firm_vars.items() if var.get()]
            candidates = None
            if selected_keys:
                all_presets = {p.key: p for p in list_prop_firm_presets()}
                candidates = [all_presets[k] for k in selected_keys if k in all_presets]

            n_sims = self.prop_rec_n_sims.get_int(2000)
            self._log_prop_rec(
                f"\nScoring against {len(candidates) if candidates else len(list_prop_firm_presets())} "
                f"firm preset(s), {n_sims:,} simulations each..."
            )
            recommendations = recommend_prop_firms(
                bt_result.trades, candidates=candidates,
                mc_cfg=MonteCarloConfig(n_simulations=n_sims),
                progress_cb=self._log_prop_rec,
            )
            self._log_prop_rec("")
            self._log_prop_rec(render_recommendation_table(recommendations))
        except StrategyError as exc:
            self._log_prop_rec(f"\nStrategy error: {exc}")
        except Exception:
            self._log_prop_rec("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.prop_rec_progress.stop()
