"""Quant Lab tab and tools."""
from __future__ import annotations

import json
import threading
from app.data import alpaca_credentials
from app.forward_test.journal import ForwardTestJournal
from app.monitoring.strategy_health import check_strategy_health
from app.monte_carlo.engine import MonteCarloResult
from app.orchestration.auto_retune import maybe_trigger_retune
from app.portfolio.composer import compose_portfolio
from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig
from app.quant_lab.factor_model import (
    compute_factor_exposures,
    compute_returns_from_prices,
    fetch_fama_french_factors,
)
from app.quant_lab.market_structure import (
    calculate_hh_ll_structure,
    calculate_wyckoff_events,
    summarize_market_structure,
)
from app.quant_lab.options_pricing import black_scholes_greeks, black_scholes_price, compare_to_market
from app.quant_lab.order_book import LimitOrderBook
from app.quant_lab.pairs_trading import fetch_universe, run_pairs_backtest, screen_pairs
from app.quant_lab.portfolio_optimizer import fetch_and_build_inputs, optimize_for_risk_level
from app.quant_lab.sentiment_price import (
    aggregate_sentiment,
    correlate_sentiment_with_price,
    fetch_headlines,
    score_headlines,
)
from app.quant_lab.vol_surface import build_iv_surface, export_surface_html, fetch_option_chain, synthetic_demo_chain
from app.strategy.auto_regime_selector import select_regime_strategies
from app.strategy.composite_thresholds import cross_level, hold_level, in_range, mix_thresholds, preview_signal
from app.strategy.library import list_saved_strategies, load_strategy_text
from app.strategy.library_loader import load_strategy_object
from app.strategy.translator import translate_strategy
from pathlib import Path
from tkinter import END, EXTENDED, Frame, Label, Listbox, Text, filedialog, messagebox
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui.widgets.run_context import RunContextPanel
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
PANEL = ""
PANEL_3 = ""
BORDER = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
theme.bind(globals())


class QuantLabMixin:
    """Quant Lab tab and tools."""


    # -----------------------------------------------------------------------
    # Quant Lab -- desktop wiring for every backend module added across this
    # project's recent upgrade rounds (Universal Strategy Translator, Auto
    # Regime Selector, Strategy Health monitor, Portfolio Composer, and the
    # whole app/quant_lab/ toolkit). Every tool here is a THIN wrapper --
    # same principle as cli.py and the web app's quant_lab_routes.py: parse
    # the form, call straight into the already-tested library function,
    # show the result. No business logic lives in this method.
    #
    # Shared UI pattern for all 12 tools: a _section() card with
    # LabeledEntry/LabeledCombo/LabeledCheckbox inputs, a RUN button that
    # disables itself and runs the real work on a background thread (via
    # _quant_lab_run_async below) so a slow Alpaca fetch or backtest never
    # freezes the UI, and the result shown in the same read-only
    # _show_text_viewer popup already used elsewhere in this app (e.g.
    # VIEW CODE) -- one shared result-display widget instead of 12
    # bespoke ones.
    # -----------------------------------------------------------------------

    def _quant_lab_run_async(self, button, work_fn, title):
        """Runs work_fn() (a zero-arg callable that returns the result
        text, or raises) on a background thread, disabling `button` while
        it runs and showing the result (or a clear error) back on the
        Tkinter main thread when done."""
        original_text = button.cget("text")
        button.config(state="disabled", text="RUNNING...")

        def _worker():
            try:
                result_text = work_fn()
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
                def _err(exc=exc):
                    button.config(state="normal", text=original_text)
                    messagebox.showerror(title, str(exc))
                self.root.after(0, _err)
                return

            def _ok():
                button.config(state="normal", text=original_text)
                self._show_text_viewer(title, result_text)
            self.root.after(0, _ok)

        threading.Thread(target=_worker, daemon=True).start()

    def _ql_alpaca_fields(self, section):
        """Two LabeledEntry fields (key/secret) pre-filled from saved
        credentials when available, plus the matching resolver used by
        every Alpaca-backed tool below."""
        saved = alpaca_credentials.load_credentials()
        note = "Using your saved Alpaca keys (Data tab) if left blank." if saved else "No saved Alpaca keys -- enter one below, or save one on the Data tab first."
        Label(section, text=note, bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=760, justify="left").pack(anchor="w", padx=18, pady=(2, 0))
        key_entry = LabeledEntry(section, "Alpaca API key (optional if saved)")
        secret_entry = LabeledEntry(section, "Alpaca API secret (optional if saved)", secret=True)
        return key_entry, secret_entry

    def _ql_resolve_alpaca_keys(self, key_entry, secret_entry) -> tuple[str, str]:
        key = key_entry.get_str().strip()
        secret = secret_entry.get_str().strip()
        if key and secret:
            return key, secret
        creds = alpaca_credentials.load_credentials()
        if creds is None or not creds.is_usable:
            raise ValueError("No Alpaca API key/secret given, and none saved on the Data tab.")
        return creds.api_key, creds.secret_key

    def _ql_validated_names(self) -> list[str]:
        # Same reasoning as above, for the Auto Regime Selector/Portfolio
        # Composer: every saved strategy is a candidate except one
        # explicitly marked failed, not only ones marked "validated".
        names = [s.name for s in list_saved_strategies() if s.status != "tested_failed"]
        return names or ["(none found)"]

    def _ql_all_translator_options(self) -> list[str]:
        """Every saved strategy across all four types (manual/python/
        pinescript/mql5), regardless of status -- translation is a
        pre-validation authoring/porting tool, not a live-trading gate.
        Mirrors app.web.quant_lab_routes._all_strategy_options() exactly
        so the desktop and web translators offer the same source list."""
        options = []
        for t in ("manual", "python", "pinescript", "mql5"):
            for s in list_saved_strategies(strategy_type=t):
                options.append(f"{t}:{s.name}")
        return options or ["(none found)"]

    def _build_quant_lab_tab(self):
        f = self._scrollable(self.tab_quantlab)
        self._page_header(
            f,
            "QUANT LAB",
            "Quant Lab",
            "Standalone quant-finance tools added across recent upgrade rounds -- the Universal "
            "Strategy Translator, Auto Regime Selector, Strategy Health monitor, Portfolio "
            "Composer, and a full toolkit (pairs trading, options pricing, an order book "
            "simulator, sentiment-price correlation, Markowitz portfolio optimization, an "
            "implied-vol surface, and a Fama-French factor model). Each is a real, tested "
            "backend module -- this tab is a thin form wrapper over it, same as the CLI's and "
            "web app's own Quant Lab wiring. The tools below that need market data, prop rules, "
            "or risk settings share one self-contained context (independent of Steps 02-04 and "
            "every other tab) rather than each duplicating its own copy, since they're all part "
            "of the same page.",
        )

        self.qlab_context = RunContextPanel(self, "Quant Lab")
        self.qlab_context.build(f)

        # -- 1. Universal Strategy Translator --------------------------------
        sec = self._section(
            f, "Universal Strategy Translator",
            "Converts ANY saved strategy -- Manual Strategy Builder config, Python, PineScript, or "
            "MQL5 -- into any of the other three. A file this tool generated (or one hand-annotated "
            "with its round-trip T58_MANUAL_CONFIG_JSON directive) translates perfectly; a "
            "hand-written PineScript/MQL5 file translates via a real parse of its supported subset "
            "(sma/ema/wma/rsi, crossover/crossunder, basic comparisons); hand-written Python has no "
            "safe general reduction back to structured conditions and can only be translated if it "
            "carries that same directive.",
        )
        self.ql_tr_strategy = LabeledCombo(sec, "Source strategy", self._ql_all_translator_options())
        self.ql_tr_target = LabeledCombo(sec, "Target language", ["pinescript", "mql5", "python"], default="pinescript")
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_tr_btn = self._button(btn_row, "TRANSLATE", self._ql_translator_clicked, primary=True)
        self.ql_tr_btn.pack(side="left")

        # -- 2. Auto Regime Selector ------------------------------------------
        sec = self._section(
            f, "Auto Regime Selector",
            "Backtests every validated Strategy Library entry, attributes trades per regime, and "
            "auto-assigns the best one per regime into a ready-to-run router. Uses the market "
            "data currently selected on Step 2 (Market Data).",
        )
        self.ql_rs_dimension = LabeledCombo(sec, "Regime dimension", ["trend", "volatility", "session", "environment"], default="trend")
        self.ql_rs_min_trades = LabeledEntry(sec, "Minimum trades per regime cell", 20)
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_rs_btn = self._button(btn_row, "RUN AUTO REGIME SELECT", self._ql_regime_selector_clicked, primary=True)
        self.ql_rs_btn.pack(side="left")

        # -- 3. Strategy Health / Drift Monitor -------------------------------
        sec = self._section(
            f, "Strategy Health / Drift Monitor",
            "Compares a forward-test session's realized results against its strategy's own "
            "predicted Monte Carlo distribution, flagging return/drawdown/losing-streak drift.",
        )
        self.ql_sh_journal = LabeledEntry(sec, "Forward-test journal .db path")
        journal_btn_row = Frame(sec, bg=PANEL); journal_btn_row.pack(anchor="w", padx=18)
        self._button(journal_btn_row, "BROWSE...", lambda: self._ql_browse_into(self.ql_sh_journal, [("SQLite DB", "*.db")])).pack(side="left")
        self.ql_sh_session_id = LabeledEntry(sec, "Session ID", 1)
        self.ql_sh_strategy_label = LabeledEntry(sec, "Strategy label")
        self.ql_sh_mc_json = LabeledEntry(sec, "Predicted Monte Carlo result (.json path)")
        mc_btn_row = Frame(sec, bg=PANEL); mc_btn_row.pack(anchor="w", padx=18)
        self._button(mc_btn_row, "BROWSE...", lambda: self._ql_browse_into(self.ql_sh_mc_json, [("JSON", "*.json")])).pack(side="left")
        self.ql_sh_balance = LabeledEntry(sec, "Account balance", 10000)
        retune_note = Label(
            sec, bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=760, justify="left",
            text="Optional: pick a Strategy Library entry below to automatically re-tune it with "
                 "Quick Optimize if this health check finds drift at or above the chosen severity -- "
                 "uses the market data and risk settings from Steps 02/04. Leave the picker on "
                 "'(health check only)' to skip this and just see the health report.",
        )
        retune_note.pack(anchor="w", padx=18, pady=(6, 2))
        self.ql_sh_retune_strategy = LabeledCombo(
            sec, "Strategy to auto re-tune (optional)", ["(health check only)"] + self._ql_validated_names(),
            default="(health check only)",
        )
        self.ql_sh_retune_threshold = LabeledCombo(
            sec, "Auto re-tune threshold", ["watch", "warning", "critical"], default="warning",
        )
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_sh_btn = self._button(btn_row, "CHECK STRATEGY HEALTH", self._ql_strategy_health_clicked, primary=True)
        self.ql_sh_btn.pack(side="left")

        # -- 4. Automated Portfolio Composer -----------------------------------
        sec = self._section(
            f, "Automated Portfolio Composer",
            "Searches your validated Strategy Library for the N-strategy combination maximizing "
            "combined performance, using the market data currently selected on Step 2 for every leg.",
        )
        Label(sec, text="Strategies to consider (2+):", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9)).pack(anchor="w", padx=18, pady=(6, 2))
        list_frame = Frame(sec, bg=PANEL); list_frame.pack(anchor="w", padx=18, pady=(0, 6), fill="x")
        self.ql_pc_listbox = Listbox(list_frame, selectmode=EXTENDED, height=6, bg=PANEL_3, fg=TEXT, font=_safe_font(9), highlightthickness=1, highlightbackground=BORDER)
        for name in self._ql_validated_names():
            self.ql_pc_listbox.insert(END, name)
        self.ql_pc_listbox.pack(fill="x")
        self.ql_pc_min_legs = LabeledEntry(sec, "Min legs", 2)
        self.ql_pc_max_legs = LabeledEntry(sec, "Max legs", 4)
        self.ql_pc_max_evals = LabeledEntry(sec, "Max evaluations", 60)
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_pc_btn = self._button(btn_row, "COMPOSE PORTFOLIO", self._ql_portfolio_composer_clicked, primary=True)
        self.ql_pc_btn.pack(side="left")

        # -- 5/6. Pairs Trading -------------------------------------------------
        sec = self._section(f, "Pairs Screener", "Screens a symbol universe (via Alpaca) for correlated, mean-reverting pairs.")
        self.ql_ps_key, self.ql_ps_secret = self._ql_alpaca_fields(sec)
        self.ql_ps_symbols = LabeledEntry(sec, "Symbols (comma-separated)", "AAPL,MSFT,GOOG")
        self.ql_ps_timeframe = LabeledEntry(sec, "Timeframe", "1Day")
        self.ql_ps_start = LabeledEntry(sec, "Start date", "2023-01-01")
        self.ql_ps_end = LabeledEntry(sec, "End date", "2024-01-01")
        self.ql_ps_min_corr = LabeledEntry(sec, "Minimum correlation", 0.7)
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_ps_btn = self._button(btn_row, "SCREEN PAIRS", self._ql_pairs_screen_clicked, primary=True)
        self.ql_ps_btn.pack(side="left")

        sec = self._section(f, "Pairs Backtest", "Fetches two instruments, builds the mean-reversion strategy, and runs the real backtest engine.")
        self.ql_pb_key, self.ql_pb_secret = self._ql_alpaca_fields(sec)
        self.ql_pb_symbol_a = LabeledEntry(sec, "Symbol A", "AAPL")
        self.ql_pb_symbol_b = LabeledEntry(sec, "Symbol B", "MSFT")
        self.ql_pb_timeframe = LabeledEntry(sec, "Timeframe", "1Day")
        self.ql_pb_start = LabeledEntry(sec, "Start date", "2023-01-01")
        self.ql_pb_end = LabeledEntry(sec, "End date", "2024-01-01")
        self.ql_pb_entry_z = LabeledEntry(sec, "Entry z-score", 2.0)
        self.ql_pb_exit_z = LabeledEntry(sec, "Exit z-score", 0.5)
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_pb_btn = self._button(btn_row, "BACKTEST PAIR", self._ql_pairs_backtest_clicked, primary=True)
        self.ql_pb_btn.pack(side="left")

        # -- 7. Options Pricing Calculator --------------------------------------
        sec = self._section(f, "Options Pricing Calculator", "Black-Scholes price, Greeks, implied volatility, and market comparison -- all from scratch.")
        self.ql_op_spot = LabeledEntry(sec, "Spot price", 100)
        self.ql_op_strike = LabeledEntry(sec, "Strike", 100)
        self.ql_op_expiry = LabeledEntry(sec, "Time to expiry (years)", 1.0)
        self.ql_op_rate = LabeledEntry(sec, "Risk-free rate", 0.05)
        self.ql_op_vol = LabeledEntry(sec, "Volatility", 0.20)
        self.ql_op_type = LabeledCombo(sec, "Type", ["call", "put"], default="call")
        self.ql_op_market_price = LabeledEntry(sec, "Market price (optional, for comparison)")
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_op_btn = self._button(btn_row, "CALCULATE", self._ql_options_pricing_clicked, primary=True)
        self.ql_op_btn.pack(side="left")

        # -- 8. Order Book Simulator ---------------------------------------------
        sec = self._section(f, "Order Book Simulator", "Replays a scripted sequence of orders through a real price-time-priority matching engine.")
        Label(sec, text="Orders (JSON list of {side, type, price?, quantity}):", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9)).pack(anchor="w", padx=18, pady=(6, 2))
        ob_frame = Frame(sec, bg=PANEL); ob_frame.pack(anchor="w", padx=18, pady=(0, 6), fill="x")
        self.ql_ob_text = Text(ob_frame, height=6, wrap="none", bg=PANEL_3, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, font=(MONO, 9))
        self.ql_ob_text.pack(fill="x")
        self.ql_ob_text.insert("1.0", (
            '[\n  {"side": "buy", "type": "limit", "price": 100.0, "quantity": 10},\n'
            '  {"side": "sell", "type": "limit", "price": 101.0, "quantity": 10},\n'
            '  {"side": "sell", "type": "limit", "price": 100.0, "quantity": 4},\n'
            '  {"side": "buy", "type": "market", "quantity": 20}\n]'
        ))
        self._bind_isolated_wheel(self.ql_ob_text)
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_ob_btn = self._button(btn_row, "REPLAY ORDERS", self._ql_order_book_clicked, primary=True)
        self.ql_ob_btn.pack(side="left")

        # -- 9. Sentiment-Price Correlation ---------------------------------------
        sec = self._section(f, "Sentiment-Price Correlation", "Scrapes financial headlines, scores sentiment with a from-scratch lexicon model, and correlates against price movement (Step 2's market data, if loaded).")
        self.ql_sp_query = LabeledEntry(sec, "Headline search query", "AAPL Apple stock")
        self.ql_sp_max_results = LabeledEntry(sec, "Max headlines", 50)
        self.ql_sp_use_price = LabeledCheckbox(sec, "Correlate against Step 2's currently loaded market data")
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_sp_btn = self._button(btn_row, "FETCH + SCORE HEADLINES", self._ql_sentiment_clicked, primary=True)
        self.ql_sp_btn.pack(side="left")

        # -- 10. Portfolio Optimizer (Markowitz) -----------------------------------
        sec = self._section(f, "Portfolio Optimizer", "Markowitz mean-variance optimization -- input tickers and a risk level, get the optimal allocation.")
        self.ql_po_key, self.ql_po_secret = self._ql_alpaca_fields(sec)
        self.ql_po_tickers = LabeledEntry(sec, "Tickers (comma-separated)", "AAPL,MSFT,GOOG")
        self.ql_po_start = LabeledEntry(sec, "Start date", "2022-01-01")
        self.ql_po_end = LabeledEntry(sec, "End date", "2024-01-01")
        self.ql_po_risk_level = LabeledEntry(sec, "Risk level (0.0 conservative -- 1.0 aggressive)", 0.5)
        self.ql_po_rf = LabeledEntry(sec, "Risk-free rate", 0.02)
        self.ql_po_long_only = LabeledCheckbox(sec, "Long-only (no short positions)")
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_po_btn = self._button(btn_row, "OPTIMIZE PORTFOLIO", self._ql_portfolio_optimizer_clicked, primary=True)
        self.ql_po_btn.pack(side="left")

        # -- 11. Volatility Surface --------------------------------------------------
        sec = self._section(f, "Volatility Surface", "Builds an interactive 3D implied-volatility surface across strikes and expirations, exported as an HTML file you can open in any browser.")
        self.ql_vs_demo = LabeledCheckbox(sec, "Use synthetic demo chain (no Alpaca options entitlement needed)", default=True)
        self.ql_vs_key, self.ql_vs_secret = self._ql_alpaca_fields(sec)
        self.ql_vs_symbol = LabeledEntry(sec, "Underlying symbol (if not using demo)")
        self.ql_vs_spot = LabeledEntry(sec, "Spot price (for demo mode)", 100)
        self.ql_vs_rate = LabeledEntry(sec, "Risk-free rate", 0.04)
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_vs_btn = self._button(btn_row, "BUILD + SAVE SURFACE", self._ql_vol_surface_clicked, primary=True)
        self.ql_vs_btn.pack(side="left")

        # -- 12. Factor Model -------------------------------------------------------
        sec = self._section(f, "Factor Model", "Fama-French 3-factor decomposition of Step 2's currently loaded market data -- tests whether apparent outperformance is real alpha or just factor exposure.")
        self.ql_fm_frequency = LabeledCombo(sec, "Factor data frequency", ["daily", "monthly"], default="daily")
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_fm_btn = self._button(btn_row, "RUN FACTOR MODEL", self._ql_factor_model_clicked, primary=True)
        self.ql_fm_btn.pack(side="left")

        # -- 13. Market Structure (Wyckoff + BOS/ChoCH) ------------------------------
        sec = self._section(
            f, "Market Structure (Wyckoff + BOS/ChoCH)",
            "Deterministic swing structure (HH/HL/LH/LL, break of structure / change of character) and Wyckoff "
            "spring/upthrust/SOS/SOW + phase detection on Step 2's currently loaded market data -- the same "
            "facts app.ai.market_intelligence now feeds T58 AI instead of the old EMA-only proxy.",
        )
        self.ql_ms_swing_left = LabeledEntry(sec, "Fractal swing bars, left", 5)
        self.ql_ms_swing_right = LabeledEntry(sec, "Fractal swing bars, right", 5)
        self.ql_ms_wyckoff_window = LabeledEntry(sec, "Wyckoff consolidation window (bars)", 40)
        self.ql_ms_wyckoff_width = LabeledEntry(sec, "Wyckoff max range width (fraction)", 0.15)
        self.ql_ms_wyckoff_lookforward = LabeledEntry(sec, "Wyckoff lookforward (bars)", 30)
        self.ql_ms_wyckoff_vol_mult = LabeledEntry(sec, "Wyckoff SOS/SOW volume multiple", 1.2)
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_ms_btn = self._button(btn_row, "ANALYZE MARKET STRUCTURE", self._ql_market_structure_clicked, primary=True)
        self.ql_ms_btn.pack(side="left")

        # -- 14. Composite Signal Builder --------------------------------------------
        sec = self._section(
            f, "Composite Signal Builder",
            "Combine two threshold signals (crossLevel, inRange, holdLevel) via AND/OR into one first-class "
            "composite signal on Step 2's currently loaded market data -- e.g. 'RSI oversold AND price above "
            "the 200 EMA' as a single signal, instead of hand-rolling the combination per strategy.",
        )
        SIGNAL_TYPES = ["cross_level_above", "cross_level_below", "in_range", "hold_level_above", "hold_level_below"]
        self.ql_cs_a_kind = LabeledEntry(sec, "Signal A: indicator", "rsi")
        self.ql_cs_a_period = LabeledEntry(sec, "Signal A: period", 14)
        self.ql_cs_a_type = LabeledCombo(sec, "Signal A: type", SIGNAL_TYPES, default="cross_level_above")
        self.ql_cs_a_level = LabeledEntry(sec, "Signal A: level / lower bound", 30)
        self.ql_cs_a_level2 = LabeledEntry(sec, "Signal A: upper bound (in_range only)", 70)
        self.ql_cs_a_min_bars = LabeledEntry(sec, "Signal A: min consecutive bars (hold_level only)", 3)
        self.ql_cs_b_kind = LabeledEntry(sec, "Signal B: indicator", "ema")
        self.ql_cs_b_period = LabeledEntry(sec, "Signal B: period", 200)
        self.ql_cs_b_type = LabeledCombo(sec, "Signal B: type", SIGNAL_TYPES, default="in_range")
        self.ql_cs_b_level = LabeledEntry(sec, "Signal B: level / lower bound", 0)
        self.ql_cs_b_level2 = LabeledEntry(sec, "Signal B: upper bound (in_range only)", 1000000)
        self.ql_cs_b_min_bars = LabeledEntry(sec, "Signal B: min consecutive bars (hold_level only)", 3)
        self.ql_cs_mode = LabeledCombo(sec, "Combine mode", ["and", "or"], default="and")
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.ql_cs_btn = self._button(btn_row, "BUILD COMPOSITE SIGNAL", self._ql_composite_signal_clicked, primary=True)
        self.ql_cs_btn.pack(side="left")

    def _ql_browse_into(self, entry: "LabeledEntry", filetypes) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes + [("All files", "*.*")])
        if path:
            entry.var.set(path)

    # -- handlers: each gathers form values on the main thread, then hands
    # a zero-arg closure to _quant_lab_run_async to do the real work off-thread.

    def _ql_translator_clicked(self):
        choice = self.ql_tr_strategy.get_str()
        target = self.ql_tr_target.get_str()

        def work():
            if choice == "(none found)" or ":" not in choice:
                raise ValueError(
                    "No saved strategies found -- save one first (Manual Strategy Builder, "
                    "Python, PineScript, or MQL5)."
                )
            source_type, name = choice.split(":", 1)
            text = load_strategy_text(source_type, name)
            source = json.loads(text) if source_type == "manual" else text
            return translate_strategy(source_type, source, target)
        self._quant_lab_run_async(self.ql_tr_btn, work, f"Translated: {choice} -> {target}")

    def _ql_regime_selector_clicked(self):
        dimension = self.ql_rs_dimension.get_str()
        min_trades = self.ql_rs_min_trades.get_int(20)

        def work():
            log_lines = []
            df = self.qlab_context.load_dataframe(log_lines.append)
            if df is None:
                raise ValueError("\n".join(log_lines) or "No market data loaded.")
            # Every saved strategy is a candidate except one explicitly
            # marked failed -- see _ql_validated_names's own note above;
            # load_validated_candidates()'s default (status="validated"
            # only) meant this was empty unless a strategy had been
            # promoted through the full pipeline.
            candidates = {}
            for s in list_saved_strategies():
                if s.status == "tested_failed":
                    continue
                try:
                    candidates[s.name] = load_strategy_object(s)
                except Exception:
                    continue
            if not candidates:
                raise ValueError(
                    "No saved Strategy Library entries found to consider -- save a strategy first."
                )
            result = select_regime_strategies(df, candidates, dimension, min_trades_per_cell=min_trades)
            return result.render_table()
        self._quant_lab_run_async(self.ql_rs_btn, work, "Auto Regime Selector")

    def _ql_strategy_health_clicked(self):
        journal_path = self.ql_sh_journal.get_str().strip()
        session_id = self.ql_sh_session_id.get_int(0)
        strategy_label = self.ql_sh_strategy_label.get_str().strip() or "strategy"
        mc_json_path = self.ql_sh_mc_json.get_str().strip()
        balance = self.ql_sh_balance.get_float(0.0)
        retune_choice = self.ql_sh_retune_strategy.get_str().strip()
        retune_threshold = self.ql_sh_retune_threshold.get_str().strip() or "warning"

        def work():
            if not journal_path or not Path(journal_path).exists():
                raise ValueError("Choose a valid forward-test journal .db file.")
            if not mc_json_path or not Path(mc_json_path).exists():
                raise ValueError("Choose a valid Monte Carlo result .json file.")
            with open(mc_json_path, "r", encoding="utf-8") as f:
                mc_data = json.load(f)
            predicted = MonteCarloResult(**mc_data)
            journal = ForwardTestJournal(db_path=Path(journal_path))

            if not retune_choice or retune_choice == "(health check only)":
                result = check_strategy_health(journal, session_id, strategy_label, predicted, account_balance=balance)
                return result.render_table()

            by_name = {s.name: s for s in list_saved_strategies() if s.status != "tested_failed"}
            stored = by_name.get(retune_choice)
            if stored is None:
                raise ValueError(f"Could not find saved strategy '{retune_choice}' -- re-select it and try again.")
            strategy = load_strategy_object(stored)
            log_lines: list[str] = []
            df = self.qlab_context.load_dataframe(log_lines.append)
            if df is None:
                raise ValueError("\n".join(log_lines) or "No market data loaded.")
            risk = self.qlab_context.build_risk_config()
            rules = self.qlab_context.build_prop_rules()

            outcome = maybe_trigger_retune(
                journal, session_id, strategy_label, predicted, account_balance=balance,
                df=df, strategy=strategy, risk=risk, prop_rules=rules,
                severity_threshold=retune_threshold,
            )
            lines = [outcome.health.render_table()]
            if not outcome.triggered:
                lines.append(f"\nNo auto re-tune triggered (severity below '{retune_threshold}').")
            elif outcome.error:
                lines.append(f"\nAuto re-tune triggered ({outcome.trigger_reason}) but failed: {outcome.error}")
            else:
                r = outcome.retune_result
                lines.append(f"\nAuto re-tune triggered ({outcome.trigger_reason}).")
                lines.append(
                    f"  Baseline: eval pass {r.baseline_eval_pass_probability:.1f}%, "
                    f"payout {r.baseline_payout_probability:.1f}%, win rate {r.baseline_win_rate:.1f}%"
                )
                lines.append(
                    f"  Re-tuned: eval pass {r.optimized_eval_pass_probability:.1f}%, "
                    f"payout {r.optimized_payout_probability:.1f}%, win rate {r.optimized_win_rate:.1f}%"
                )
                lines.append(f"  Improved: {'yes' if r.improved else 'no'}")
                if r.saved_library_note:
                    lines.append(f"  {r.saved_library_note}")
            return "\n".join(lines)
        self._quant_lab_run_async(self.ql_sh_btn, work, "Strategy Health")

    def _ql_portfolio_composer_clicked(self):
        selected_idx = self.ql_pc_listbox.curselection()
        names = [self.ql_pc_listbox.get(i) for i in selected_idx]
        min_legs = self.ql_pc_min_legs.get_int(2)
        max_legs = self.ql_pc_max_legs.get_int(4)
        max_evals = self.ql_pc_max_evals.get_int(60)

        def work():
            if len(names) < 2:
                raise ValueError("Select at least 2 strategies from the list (ctrl/shift-click for multiple).")
            log_lines = []
            df = self.qlab_context.load_dataframe(log_lines.append)
            if df is None:
                raise ValueError("\n".join(log_lines) or "No market data loaded.")
            risk = self.qlab_context.build_risk_config()
            # Same broadened pool as the listbox itself (_ql_validated_names) --
            # everything except a strategy explicitly marked failed.
            by_name = {s.name: s for s in list_saved_strategies() if s.status != "tested_failed"}
            legs = [InstrumentLeg(name=n, df=df, strategy=load_strategy_object(by_name[n]), risk=risk) for n in names]
            result = compose_portfolio(legs, min_legs=min_legs, max_legs=max_legs, max_evaluations=max_evals,
                                        portfolio_config=PortfolioConfig())
            return result.render_table()
        self._quant_lab_run_async(self.ql_pc_btn, work, "Portfolio Composer")

    def _ql_pairs_screen_clicked(self):
        symbols_str = self.ql_ps_symbols.get_str()
        timeframe = self.ql_ps_timeframe.get_str()
        start, end = self.ql_ps_start.get_str(), self.ql_ps_end.get_str()
        min_corr = self.ql_ps_min_corr.get_float(0.7)
        key_entry, secret_entry = self.ql_ps_key, self.ql_ps_secret

        def work():
            api_key, secret_key = self._ql_resolve_alpaca_keys(key_entry, secret_entry)
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
            universe = fetch_universe(api_key, secret_key, symbols, timeframe, start, end)
            candidates = screen_pairs(universe, min_correlation=min_corr)
            if not candidates:
                return "No pairs cleared the correlation threshold."
            return "\n".join(
                f"{c.symbol_a}/{c.symbol_b}: corr={c.correlation:.3f} adf={c.adf_statistic:+.2f} "
                f"({c.stationarity_verdict}) z={c.current_zscore:+.2f}" for c in candidates
            )
        self._quant_lab_run_async(self.ql_ps_btn, work, "Pairs Screener")

    def _ql_pairs_backtest_clicked(self):
        symbol_a, symbol_b = self.ql_pb_symbol_a.get_str(), self.ql_pb_symbol_b.get_str()
        timeframe = self.ql_pb_timeframe.get_str()
        start, end = self.ql_pb_start.get_str(), self.ql_pb_end.get_str()
        entry_z, exit_z = self.ql_pb_entry_z.get_float(2.0), self.ql_pb_exit_z.get_float(0.5)
        key_entry, secret_entry = self.ql_pb_key, self.ql_pb_secret

        def work():
            api_key, secret_key = self._ql_resolve_alpaca_keys(key_entry, secret_entry)
            result = run_pairs_backtest(api_key, secret_key, symbol_a, symbol_b, timeframe, start, end,
                                         entry_z=entry_z, exit_z=exit_z)
            stats = result.backtest.statistics
            verdict = result.pair.stationarity_verdict if result.pair else "n/a"
            return (f"Pair: {symbol_a}/{symbol_b}\n{verdict}\n"
                    f"Trades: {stats.total_trades}  Net profit: {stats.net_profit:+.2f}  Win rate: {stats.win_rate:.1f}%")
        self._quant_lab_run_async(self.ql_pb_btn, work, "Pairs Backtest")

    def _ql_options_pricing_clicked(self):
        spot, strike = self.ql_op_spot.get_float(100), self.ql_op_strike.get_float(100)
        T, r, vol = self.ql_op_expiry.get_float(1.0), self.ql_op_rate.get_float(0.05), self.ql_op_vol.get_float(0.2)
        opt_type = self.ql_op_type.get_str()
        market_price_str = self.ql_op_market_price.get_str().strip()

        def work():
            if market_price_str:
                cmp = compare_to_market(float(market_price_str), spot, strike, T, r, vol, opt_type)
                return cmp.render_summary()
            price = black_scholes_price(spot, strike, T, r, vol, opt_type)
            greeks = black_scholes_greeks(spot, strike, T, r, vol, opt_type)
            return f"Price: {price:.4f}\n" + "\n".join(f"{k}: {v:+.4f}" for k, v in greeks.to_dict().items())
        self._quant_lab_run_async(self.ql_op_btn, work, "Options Pricing")

    def _ql_order_book_clicked(self):
        orders_text = self.ql_ob_text.get("1.0", END)

        def work():
            orders = json.loads(orders_text)
            book = LimitOrderBook()
            lines, total_trades = [], 0
            for order in orders:
                if order["type"] == "limit":
                    res = book.submit_limit_order(order["side"], order["price"], order["quantity"])
                else:
                    res = book.submit_market_order(order["side"], order["quantity"])
                total_trades += len(res.trades)
                lines.append(f"{order} -> {len(res.trades)} trade(s), resting={res.resting}")
            lines.append(f"\nTotal trades printed: {total_trades}")
            lines.append(f"Best bid/ask: {book.best_bid()} / {book.best_ask()}   Spread: {book.spread()}")
            lines.append(f"Depth: {book.depth_snapshot()}")
            return "\n".join(lines)
        self._quant_lab_run_async(self.ql_ob_btn, work, "Order Book Simulator")

    def _ql_sentiment_clicked(self):
        query = self.ql_sp_query.get_str()
        max_results = self.ql_sp_max_results.get_int(50)
        use_price = self.ql_sp_use_price.get()

        def work():
            headlines = fetch_headlines(query, max_results=max_results)
            sentiment_df = score_headlines(headlines)
            overall = aggregate_sentiment(sentiment_df)
            lines = [overall.render_summary(), ""]
            lines += [f"{row.label:<9} {row.sentiment:+.2f}  {row.title}" for row in sentiment_df.itertuples()]
            if use_price:
                log_lines = []
                df = self.qlab_context.load_dataframe(log_lines.append)
                if df is not None:
                    # Correlation is an optional add-on -- if it can't run
                    # (e.g. the price data doesn't overlap the recent
                    # headline window), keep the sentiment result and say
                    # why, instead of failing the whole job. Mirrors the
                    # web route's handling in app/web/quant_lab_routes.py.
                    try:
                        corr = correlate_sentiment_with_price(sentiment_df, df)
                        lines.append("")
                        lines.append(corr.render_summary())
                        lines += [f"Note: {w}" for w in corr.warnings]
                    except Exception as exc:  # noqa: BLE001 -- degrade, don't discard
                        lines.append("")
                        lines.append(f"Price correlation unavailable: {exc}")
                else:
                    lines.append("")
                    lines.append("(Could not correlate: " + "; ".join(log_lines) + ")")
            return "\n".join(lines)
        self._quant_lab_run_async(self.ql_sp_btn, work, "Sentiment-Price Correlation")

    def _ql_portfolio_optimizer_clicked(self):
        tickers_str = self.ql_po_tickers.get_str()
        start, end = self.ql_po_start.get_str(), self.ql_po_end.get_str()
        risk_level = self.ql_po_risk_level.get_float(0.5)
        rf = self.ql_po_rf.get_float(0.02)
        long_only = self.ql_po_long_only.get()
        key_entry, secret_entry = self.ql_po_key, self.ql_po_secret

        def work():
            api_key, secret_key = self._ql_resolve_alpaca_keys(key_entry, secret_entry)
            tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
            inputs = fetch_and_build_inputs(api_key, secret_key, tickers, start, end)
            allocation = optimize_for_risk_level(inputs, risk_level, risk_free_rate=rf, long_only=long_only)
            return allocation.render_summary()
        self._quant_lab_run_async(self.ql_po_btn, work, "Portfolio Optimizer")

    def _ql_vol_surface_clicked(self):
        demo = self.ql_vs_demo.get()
        symbol = self.ql_vs_symbol.get_str().strip()
        spot = self.ql_vs_spot.get_float(100)
        rate = self.ql_vs_rate.get_float(0.04)
        key_entry, secret_entry = self.ql_vs_key, self.ql_vs_secret

        # The save-location dialog is a native modal window and MUST be
        # opened on the main thread (same reason every other file/save
        # dialog in this app is triggered from a button click, never from
        # inside a background worker) -- ask for it here, before handing
        # the actual (slow, network-bound) work off to a thread.
        save_path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML", "*.html")],
                                                   initialfile="vol_surface.html")
        if not save_path:
            return

        def work():
            if demo:
                chain = synthetic_demo_chain(spot=spot)
            else:
                if not symbol:
                    raise ValueError("Enter an underlying symbol, or check 'Use synthetic demo chain'.")
                api_key, secret_key = self._ql_resolve_alpaca_keys(key_entry, secret_entry)
                chain = fetch_option_chain(api_key, secret_key, symbol)
            surface = build_iv_surface(chain, r=rate)
            out = export_surface_html(surface, save_path, title="Implied Volatility Surface")
            return f"Wrote {out} ({len(surface)} points). Open it in any browser to view the interactive 3D surface."
        self._quant_lab_run_async(self.ql_vs_btn, work, "Volatility Surface")

    def _ql_factor_model_clicked(self):
        frequency = self.ql_fm_frequency.get_str()

        def work():
            log_lines = []
            df = self.qlab_context.load_dataframe(log_lines.append)
            if df is None:
                raise ValueError("\n".join(log_lines) or "No market data loaded.")
            returns = compute_returns_from_prices(df)
            factors = fetch_fama_french_factors(frequency=frequency)
            result = compute_factor_exposures(returns, factors, periods_per_year=252 if frequency == "daily" else 12)
            return result.render_summary()
        self._quant_lab_run_async(self.ql_fm_btn, work, "Factor Model")

    def _ql_market_structure_clicked(self):
        swing_left = self.ql_ms_swing_left.get_int(5)
        swing_right = self.ql_ms_swing_right.get_int(5)
        wyckoff_window = self.ql_ms_wyckoff_window.get_int(40)
        wyckoff_width = self.ql_ms_wyckoff_width.get_float(0.15)
        wyckoff_lookforward = self.ql_ms_wyckoff_lookforward.get_int(30)
        wyckoff_vol_mult = self.ql_ms_wyckoff_vol_mult.get_float(1.2)

        def work():
            log_lines = []
            df = self.qlab_context.load_dataframe(log_lines.append)
            if df is None:
                raise ValueError("\n".join(log_lines) or "No market data loaded.")
            summary = summarize_market_structure(
                df, swing_left=swing_left, swing_right=swing_right, wyckoff_window=wyckoff_window,
                wyckoff_max_width_pct=wyckoff_width, wyckoff_lookforward=wyckoff_lookforward,
                wyckoff_volume_mult=wyckoff_vol_mult,
            )
            lines = [summary.render_summary(), ""]

            hh_ll = calculate_hh_ll_structure(df, left=swing_left, right=swing_right)
            lines.append(f"Swing structure: {len(hh_ll)} swing(s) found.")
            events_only = hh_ll[hh_ll["event"].notna()].tail(20)
            if not events_only.empty:
                lines.append("Most recent BOS/ChoCH events:")
                lines += [
                    f"  {row.timestamp}  {row.event.upper():<5} ({row.structure}, trend={row.trend}, price={row.price:.5f})"
                    for row in events_only.itertuples()
                ]

            wyckoff = calculate_wyckoff_events(
                df, window=wyckoff_window, max_width_pct=wyckoff_width,
                lookforward=wyckoff_lookforward, volume_mult=wyckoff_vol_mult,
            )
            lines.append("")
            lines.append(f"Wyckoff events: {len(wyckoff)} found.")
            if not wyckoff.empty:
                lines.append("Most recent Wyckoff events:")
                lines += [
                    f"  {row.timestamp}  {row.event.upper():<9} phase={row.phase:<13} price={row.price:.5f}"
                    for row in wyckoff.tail(20).itertuples()
                ]
            return "\n".join(lines)
        self._quant_lab_run_async(self.ql_ms_btn, work, "Market Structure")

    def _ql_build_signal(self, df, kind, period, sig_type, level, level2, min_bars):
        if sig_type == "cross_level_above":
            return cross_level(df, kind, period, level, direction="above")
        if sig_type == "cross_level_below":
            return cross_level(df, kind, period, level, direction="below")
        if sig_type == "in_range":
            return in_range(df, kind, period, level, level2)
        if sig_type == "hold_level_above":
            return hold_level(df, kind, period, level, direction="above", min_bars=min_bars)
        if sig_type == "hold_level_below":
            return hold_level(df, kind, period, level, direction="below", min_bars=min_bars)
        raise ValueError(f"Unknown signal type '{sig_type}'.")

    def _ql_composite_signal_clicked(self):
        a_kind = self.ql_cs_a_kind.get_str().strip().lower()
        a_period = self.ql_cs_a_period.get_int(14)
        a_type = self.ql_cs_a_type.get_str()
        a_level = self.ql_cs_a_level.get_float(30)
        a_level2 = self.ql_cs_a_level2.get_float(70)
        a_min_bars = self.ql_cs_a_min_bars.get_int(3)
        b_kind = self.ql_cs_b_kind.get_str().strip().lower()
        b_period = self.ql_cs_b_period.get_int(200)
        b_type = self.ql_cs_b_type.get_str()
        b_level = self.ql_cs_b_level.get_float(0)
        b_level2 = self.ql_cs_b_level2.get_float(1000000)
        b_min_bars = self.ql_cs_b_min_bars.get_int(3)
        mode = self.ql_cs_mode.get_str() or "and"

        def work():
            log_lines = []
            df = self.qlab_context.load_dataframe(log_lines.append)
            if df is None:
                raise ValueError("\n".join(log_lines) or "No market data loaded.")
            signal_a = self._ql_build_signal(df, a_kind, a_period, a_type, a_level, a_level2, a_min_bars)
            signal_b = self._ql_build_signal(df, b_kind, b_period, b_type, b_level, b_level2, b_min_bars)
            combo = mix_thresholds([signal_a, signal_b], mode=mode)
            lines = [
                "== Signal A ==", preview_signal(df, signal_a).render_summary(), "",
                "== Signal B ==", preview_signal(df, signal_b).render_summary(), "",
                f"== Combined ({mode.upper()}) ==", preview_signal(df, combo).render_summary(),
            ]
            return "\n".join(lines)
        self._quant_lab_run_async(self.ql_cs_btn, work, "Composite Signal Builder")
