"""Forward Test (MT5 demo) tab."""
from __future__ import annotations

import app.forward_test.mt5_settings as mt5_settings_module
import threading
from app.backtest.risk import RiskConfig
from app.forward_test import mt5_connector as mt5_connector_module
from app.forward_test.engine import ForwardTestConfig, ForwardTestSession
from app.forward_test.journal import ForwardTestJournal
from app.forward_test.mt5_settings import MT5Settings
from app.strategy.base import StrategyError
from app.strategy.library import list_saved_strategies
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy
from tkinter import Button, END, Frame, Label, StringVar, Text, filedialog, messagebox, ttk
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCombo, LabeledEntry
from app.ui.widgets.cards import NeuralProgress
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
PANEL = ""
BORDER = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
LOG_BG = ""
GREEN = ""
RED = ""
AMBER = ""
theme.bind(globals())


class ForwardTestMixin:
    """Forward Test (MT5 demo) tab."""


    # -----------------------------------------------------------------------
    # Tab 16 — Live Demo Test (MT5 Demo)
    # -----------------------------------------------------------------------

    def _build_forward_test_tab(self):
        f = self._scrollable(self.tab_forwardtest)
        self._ft_session: ForwardTestSession | None = None
        self._ft_journal = None

        self._page_header(
            f,
            "FORWARD TEST / Forward Test (MT5)",
            "Live Demo Test (MT5 Demo)",
            "Deploy any Strategy Library strategy to a free MetaTrader 5 demo account and "
            "watch it trade forward, bar by bar, against real broker prices instead of a "
            "CSV. Uses the exact same signal engine and position-sizing math as the "
            "backtester -- no separate re-implementation to drift out of sync. This is a "
            "demo account only; there is no live/funded order path here.",
        )

        if not mt5_connector_module.is_available():
            notice = self._section(f, "MetaTrader 5 not detected", "", emphasize=True)
            Label(
                notice, text=mt5_connector_module.unavailable_reason(), bg=PANEL, fg=AMBER,
                font=_safe_font(9), wraplength=820, justify="left",
            ).pack(anchor="w", padx=18, pady=(2, 6))
            Label(
                notice,
                text="To use this tab: have an MT5 terminal installed and logged into a demo account "
                     "(any MT5 broker's site offers a free demo account signup). On Windows make sure "
                     "`pip install MetaTrader5` succeeded; on Linux/macOS run the terminal under Wine and "
                     "start the mt5linux bridge (`python scripts/mt5_bridge_server.py` prints the steps). The "
                     "rest of this tab still works for entering settings -- Start Live Demo Test "
                     "will just fail with a clear message until MT5 is reachable.",
                bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
            ).pack(anchor="w", padx=18, pady=(0, 12))

        strat_section = self._section(
            f, "Strategy",
            "Pulled from your Strategy Library (Step 01) -- Python, PineScript, and MQL5 "
            "strategies all work here since they share the same signal interface.",
            emphasize=True,
        )
        picker_row = Frame(strat_section, bg=PANEL)
        picker_row.pack(fill="x", padx=18, pady=(2, 4))
        Label(
            picker_row, text="Strategy", width=31, anchor="w",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
        ).pack(side="left")
        self.ft_strategy_var = StringVar(value="")
        self.ft_strategy_combo = ttk.Combobox(
            picker_row, textvariable=self.ft_strategy_var, values=[], state="readonly",
            width=52, font=_safe_font(9), style="T58.TCombobox",
        )
        self.ft_strategy_combo.pack(side="left", padx=(4, 8))
        self._button(picker_row, "REFRESH", self._ft_refresh_strategies).pack(side="left")
        self._ft_strategy_items: list = []
        self._ft_refresh_strategies()

        market_section = self._section(
            f, "Market",
            "Symbol must match this exact string on your MT5 account/broker (e.g. 'XAUUSD', "
            "'EURUSD', 'US30' -- some brokers append a suffix like 'XAUUSD.m', check your "
            "MT5 Market Watch if the exact symbol isn't found).",
        )
        self.ft_symbol = LabeledEntry(market_section, "Symbol", mt5_settings_module.DEFAULT_SYMBOL)
        self.ft_timeframe = LabeledCombo(
            market_section, "Timeframe",
            ["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour", "4 hours", "1 day"],
            "15 minutes",
        )

        account_section = self._section(
            f, "MT5 Demo Account",
            "Saved locally (password via your OS keyring where available). Get a free demo "
            "login from any MT5-supporting broker's website, or your prop firm's own demo/"
            "trial account if they offer one.",
        )
        saved_mt5 = mt5_settings_module.load_settings()
        self.ft_login = LabeledEntry(account_section, "Login (account number)", saved_mt5.login)
        self.ft_server = LabeledEntry(account_section, "Server", saved_mt5.server)
        self.ft_password = LabeledEntry(account_section, "Password", saved_mt5.password, secret=True)
        self.ft_terminal_path = LabeledEntry(
            account_section, "Terminal path (optional, only if auto-detect fails)", saved_mt5.terminal_path,
        )
        terminal_btn_row = Frame(account_section, bg=PANEL)
        terminal_btn_row.pack(anchor="w", padx=18, pady=(0, 8))
        Label(
            terminal_btn_row, text="", width=31, bg=PANEL,
        ).pack(side="left")  # spacer matching LabeledEntry's label column so buttons line up under the field
        self._button(terminal_btn_row, "AUTO-DETECT", self._ft_auto_detect_terminal).pack(side="left")
        self._button(terminal_btn_row, "BROWSE...", self._ft_browse_terminal).pack(side="left", padx=(6, 0))
        conn_row = Frame(account_section, bg=PANEL)
        conn_row.pack(anchor="w", padx=18, pady=(2, 10))
        self.ft_test_conn_btn = self._button(conn_row, "SAVE & TEST CONNECTION", self._ft_test_connection, primary=True)
        self.ft_test_conn_btn.pack(side="left")
        self.ft_conn_status = Label(conn_row, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=760, justify="left")
        self.ft_conn_status.pack(side="left", padx=(12, 0))

        risk_section = self._section(
            f, "Risk (same fields as Step 04, applied to this live demo account)",
            "Position sizing calls the exact same RiskConfig.position_size(...) the "
            "backtester uses -- if a strategy defines its own dynamic stop, that's honored "
            "first; otherwise a fixed-pip or 1%-of-price fallback applies, same precedence "
            "as a backtest run.",
        )
        self.ft_risk_pct = LabeledEntry(risk_section, "Risk per trade (% of equity)", 1.0)
        self.ft_daily_loss_limit = LabeledEntry(risk_section, "Daily loss limit (% of balance, halts new entries)", 5.0)
        self.ft_max_trades_per_day = LabeledEntry(risk_section, "Max trades per day", 10)
        self.ft_pip_size = LabeledEntry(risk_section, "Pip size (leave blank to auto-detect from MT5 symbol info)", "")
        self.ft_baseline_win_rate = LabeledEntry(
            risk_section, "Backtest win rate %, for drift comparison (optional)", "",
        )

        control_section = self._section(f, "Session control", "")
        btn_row = Frame(control_section, bg=PANEL)
        btn_row.pack(fill="x", padx=18, pady=(2, 4))
        self.ft_start_btn = self._button(btn_row, "START LIVE DEMO TEST", self._ft_start_clicked, primary=True)
        self.ft_start_btn.pack(side="left")
        self.ft_stop_btn = self._button(btn_row, "STOP", self._ft_stop_clicked)
        self.ft_stop_btn.pack(side="left", padx=8)
        self.ft_stop_btn.config(state="disabled")
        self.ft_kill_btn = Button(
            btn_row, text="KILL SWITCH — FLATTEN & STOP", command=self._ft_kill_clicked,
            bg=RED, fg="#1a0a0d", activebackground="#c94656", activeforeground="#1a0a0d",
            relief="flat", bd=0, font=_safe_font(9, "bold"), padx=14, pady=8, cursor="hand2",
        )
        self.ft_kill_btn.pack(side="left", padx=8)
        self.ft_kill_btn.config(state="disabled")

        self.ft_progress = NeuralProgress(control_section)
        self.ft_progress.pack(fill="x", padx=18, pady=(6, 10))

        status_row = Frame(control_section, bg=PANEL)
        status_row.pack(fill="x", padx=18, pady=(0, 12))
        self.ft_status_label = Label(
            status_row, text="Not running.", bg=PANEL, fg=TEXT_DIM,
            font=_safe_font(10, "bold"), justify="left", wraplength=820,
        )
        self.ft_status_label.pack(anchor="w")

        journal_section = self._section(f, "Trade journal (this session)", "")
        columns = ("time", "direction", "volume", "entry", "exit", "pnl", "status")
        self.ft_journal_tree = ttk.Treeview(journal_section, columns=columns, show="headings", height=8)
        for col, label, w in (
            ("time", "Entry Time", 140), ("direction", "Dir", 50), ("volume", "Size", 70),
            ("entry", "Entry", 90), ("exit", "Exit", 90), ("pnl", "P&L", 80), ("status", "Status", 70),
        ):
            self.ft_journal_tree.heading(col, text=label)
            self.ft_journal_tree.column(col, width=w, anchor="center")
        self.ft_journal_tree.pack(fill="x", padx=18, pady=(2, 12))

        log_section = self._section(f, "Live log", "")
        _ft_log_frame = Frame(log_section, bg=PANEL)
        self.ft_log = Text(
            _ft_log_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _ft_log_scroll = ttk.Scrollbar(
            _ft_log_frame, orient="vertical", command=self.ft_log.yview, style="T58.Vertical.TScrollbar",
        )
        self.ft_log.configure(yscrollcommand=_ft_log_scroll.set)
        self.ft_log.pack(side="left", fill="both", expand=True)
        _ft_log_scroll.pack(side="right", fill="y")
        _ft_log_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.ft_log)

    def _ft_refresh_strategies(self):
        items = list_saved_strategies(None)
        self._ft_strategy_items = items
        labels = [f"[{it.strategy_type}] {it.name}" for it in items]
        self.ft_strategy_combo["values"] = labels
        if labels and not self.ft_strategy_var.get():
            self.ft_strategy_var.set(labels[0])

    def _ft_selected_strategy_item(self):
        label = self.ft_strategy_var.get()
        for it in self._ft_strategy_items:
            if f"[{it.strategy_type}] {it.name}" == label:
                return it
        return None

    def _ft_build_strategy_instance(self, item):
        if item.strategy_type == "python":
            return PythonStrategy(item.path)
        text = item.path.read_text(encoding="utf-8")
        if item.strategy_type == "pinescript":
            return PineScriptStrategy(text)
        if item.strategy_type == "mql5":
            return MQL5Strategy(text)
        raise StrategyError(f"Unsupported strategy type for live demo test: {item.strategy_type}")

    def _ft_timeframe_minutes(self) -> int:
        mapping = {
            "1 minute": 1, "5 minutes": 5, "15 minutes": 15, "30 minutes": 30,
            "1 hour": 60, "4 hours": 240, "1 day": 1440,
        }
        return mapping.get(self.ft_timeframe.get_str(), 15)

    def _ft_log_line(self, level: str, message: str):
        color = {"error": RED, "warn": AMBER, "info": TEXT}.get(level, TEXT)
        tag = f"lvl_{level}"
        self.ft_log.tag_config(tag, foreground=color)
        self.ft_log.insert(END, f"[{level.upper()}] {message}\n", tag)
        self.ft_log.see(END)
        try:
            self.root.after(0, self.root.update_idletasks)
        except Exception:
            pass

    def _ft_browse_terminal(self):
        path = filedialog.askopenfilename(
            title="Locate your MT5 terminal64.exe (Windows, or inside your Wine prefix)",
            filetypes=[("MT5 terminal", "terminal64.exe"), ("All files", "*.*")],
        )
        if path:
            self.ft_terminal_path.var.set(path)
            self.ft_conn_status.config(
                text=f"Terminal path set to {path}. Click Save & Test Connection to verify it.", fg=TEXT_MUTED,
            )

    def _ft_auto_detect_terminal(self):
        if not mt5_connector_module.is_available():
            self.ft_conn_status.config(text=mt5_connector_module.unavailable_reason(), fg=AMBER)
            return
        self.ft_conn_status.config(text="Searching common install locations for terminal64.exe...", fg=AMBER)
        self.root.update_idletasks()
        candidates = mt5_connector_module.find_terminal_candidates()
        if not candidates:
            self.ft_conn_status.config(
                text="Couldn't find terminal64.exe in any common install location (Program Files, "
                     "AppData\\MetaQuotes). If MT5 is installed somewhere else, use Browse... to "
                     "point at it directly -- or install it fresh from your broker's site if it "
                     "isn't installed at all.",
                fg=AMBER,
            )
            return
        self.ft_terminal_path.var.set(candidates[0])
        extra = f" ({len(candidates) - 1} other install(s) also found -- this is the most recently used.)" if len(candidates) > 1 else ""
        self.ft_conn_status.config(
            text=f"Found: {candidates[0]}{extra} Click Save & Test Connection to verify it.", fg=GREEN,
        )

    def _ft_save_mt5_settings(self) -> MT5Settings:
        settings = MT5Settings(
            login=self.ft_login.get_str().strip(),
            server=self.ft_server.get_str().strip(),
            password=self.ft_password.get_str(),
            symbol=self.ft_symbol.get_str().strip() or mt5_settings_module.DEFAULT_SYMBOL,
            timeframe_minutes=self._ft_timeframe_minutes(),
            terminal_path=self.ft_terminal_path.get_str().strip(),
        )
        mt5_settings_module.save_settings(settings)
        return settings

    def _ft_test_connection(self):
        settings = self._ft_save_mt5_settings()
        if not mt5_connector_module.is_available():
            self.ft_conn_status.config(text=mt5_connector_module.unavailable_reason(), fg=AMBER)
            return
        self.ft_test_conn_btn.config(state="disabled")
        self.ft_conn_status.config(text="Connecting...", fg=AMBER)

        def run():
            from app.forward_test.mt5_connector import MT5Connector
            connector = MT5Connector(settings.login, settings.password, settings.server, settings.terminal_path)
            result = connector.connect()
            if result.ok:
                msg = (f"Connected: account {result.account_login} @ {result.account_server} — "
                       f"balance {result.balance:,.2f} {result.currency}, equity {result.equity:,.2f}.")
                # Auto-detection may have found a working terminal path this
                # session even though the field was left blank -- persist
                # it now so future connections (and the Forward Test
                # session itself) don't have to rediscover it every time.
                if result.resolved_terminal_path and not settings.terminal_path:
                    settings.terminal_path = result.resolved_terminal_path
                    mt5_settings_module.save_settings(settings)
                    self.root.after(0, lambda: self.ft_terminal_path.var.set(result.resolved_terminal_path))
                    msg += f" (auto-detected and saved terminal path: {result.resolved_terminal_path})"
                connector.disconnect()
            else:
                msg = result.message
            self.root.after(0, lambda: self.ft_conn_status.config(text=msg, fg=GREEN if result.ok else RED))
            self.root.after(0, lambda: self.ft_test_conn_btn.config(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    def _ft_start_clicked(self):
        item = self._ft_selected_strategy_item()
        if item is None:
            messagebox.showwarning("No strategy selected", "Choose a strategy from the Strategy Library first.")
            return
        if not mt5_connector_module.is_available():
            messagebox.showwarning("MT5 not available", mt5_connector_module.unavailable_reason())
            return

        settings = self._ft_save_mt5_settings()
        if not settings.is_usable:
            messagebox.showwarning("Missing credentials", "Enter your MT5 demo login, server, and password.")
            return

        try:
            strategy = self._ft_build_strategy_instance(item)
        except Exception as exc:
            messagebox.showerror("Strategy error", f"Could not load strategy: {exc}")
            return

        from app.forward_test.mt5_connector import MT5Connector

        connector = MT5Connector(settings.login, settings.password, settings.server, settings.terminal_path)
        pip_size_str = self.ft_pip_size.get_str().strip()
        pip_size = None
        if pip_size_str:
            try:
                pip_size = float(pip_size_str)
            except ValueError:
                pip_size = None

        def resolve_pip_and_start():
            nonlocal pip_size
            probe = MT5Connector(settings.login, settings.password, settings.server, settings.terminal_path)
            conn = probe.connect()
            if not conn.ok:
                self.root.after(0, lambda: self._ft_log_line("error", conn.message))
                return
            if pip_size is None:
                try:
                    pip_size = probe.symbol_point(settings.symbol)
                except Exception as exc:
                    self.root.after(0, lambda exc=exc: self._ft_log_line(
                        "warn", f"Could not auto-detect pip size ({exc}); falling back to 0.0001."))
                    pip_size = 0.0001
            probe.disconnect()

            risk = RiskConfig(
                initial_balance=conn.balance or 10_000.0,
                risk_mode="percent",
                risk_value=self.ft_risk_pct.get_float(1.0),
                max_trades_per_day=self.ft_max_trades_per_day.get_int(10),
                pip_size=pip_size,
                daily_loss_limit_pct=self.ft_daily_loss_limit.get_float(5.0) or None,
            )
            baseline_str = self.ft_baseline_win_rate.get_str().strip()
            baseline_win_rate = float(baseline_str) if baseline_str else None

            cfg = ForwardTestConfig(
                symbol=settings.symbol, timeframe_minutes=settings.timeframe_minutes,
                risk=risk, baseline_win_rate=baseline_win_rate,
            )
            self._ft_journal = ForwardTestJournal()
            session = ForwardTestSession(
                strategy=strategy, strategy_type=item.strategy_type, strategy_filename=item.name,
                connector=connector, journal=self._ft_journal, config=cfg,
                on_log=lambda level, msg: self.root.after(0, lambda: self._ft_log_line(level, msg)),
                on_status=lambda status: self.root.after(0, lambda: self._ft_update_status(status)),
            )
            ok, msg = session.start()
            if not ok:
                self.root.after(0, lambda: self._ft_log_line("error", msg))
                return
            self._ft_session = session
            self.root.after(0, self._ft_on_started)

        threading.Thread(target=resolve_pip_and_start, daemon=True).start()
        self.ft_start_btn.config(state="disabled")
        self.ft_status_label.config(text="Connecting...", fg=AMBER)

    def _ft_on_started(self):
        self.ft_stop_btn.config(state="normal")
        self.ft_kill_btn.config(state="normal")
        self.ft_progress.start()
        self.ft_status_label.config(text="Running.", fg=GREEN)

    def _ft_update_status(self, status):
        parts = [f"Running: {status.running}", f"Last signal: {status.last_signal}"]
        if status.balance is not None:
            parts.append(f"Balance ${status.balance:,.2f}")
        if status.equity is not None:
            parts.append(f"Equity ${status.equity:,.2f}")
        if status.n_trades_closed:
            parts.append(f"Closed trades: {status.n_trades_closed}")
        if status.win_rate is not None:
            parts.append(f"Win rate {status.win_rate:.1f}%")
        if status.halted_reason:
            parts.append(f"HALTED: {status.halted_reason}")
        color = RED if status.halted_reason else (GREEN if status.running else TEXT_DIM)
        if status.drift_flag:
            parts.append(f"⚠ {status.drift_flag}")
            color = AMBER
        self.ft_status_label.config(text="  |  ".join(parts), fg=color)
        self._ft_refresh_journal_view()

    def _ft_refresh_journal_view(self):
        if self._ft_journal is None or self._ft_session is None or self._ft_session._session_id is None:
            return
        for row in self.ft_journal_tree.get_children():
            self.ft_journal_tree.delete(row)
        trades = self._ft_journal.all_trades(self._ft_session._session_id)
        for t in trades[:100]:
            import datetime
            entry_time = datetime.datetime.fromtimestamp(t.entry_time).strftime("%m-%d %H:%M")
            direction = "LONG" if t.direction == 1 else "SHORT"
            exit_price = f"{t.exit_price:.5f}" if t.exit_price else ""
            pnl = f"{t.pnl:,.2f}" if t.pnl is not None else ""
            self.ft_journal_tree.insert("", END, values=(
                entry_time, direction, f"{t.volume:.2f}", f"{t.entry_price:.5f}", exit_price, pnl, t.status,
            ))

    def _ft_stop_clicked(self):
        if self._ft_session is None:
            return
        self.ft_status_label.config(text="Stopping...", fg=AMBER)
        threading.Thread(target=self._ft_session.stop, daemon=True).start()
        self.ft_start_btn.config(state="normal")
        self.ft_stop_btn.config(state="disabled")
        self.ft_kill_btn.config(state="disabled")
        self.ft_progress.stop()

    def _ft_kill_clicked(self):
        if self._ft_session is None:
            return
        if not messagebox.askyesno(
            "Kill switch",
            "This closes every open position on this symbol immediately and stops the "
            "session. Continue?",
        ):
            return
        self.ft_status_label.config(text="Flattening and stopping...", fg=RED)
        threading.Thread(target=self._ft_session.flatten_all_and_stop, daemon=True).start()
        self.ft_start_btn.config(state="normal")
        self.ft_stop_btn.config(state="disabled")
        self.ft_kill_btn.config(state="disabled")
        self.ft_progress.stop()
