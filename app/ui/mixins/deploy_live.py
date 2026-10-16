"""Deploy Live tab."""
from __future__ import annotations

import threading
from app.backtest.risk import RiskConfig
from app.forward_test import mt5_connector as mt5_connector_module
from app.forward_test.journal import ForwardTestJournal
from app.live_deploy import (
    connect_flow as live_connect_flow,
    live_settings as live_deploy_settings,
    prop_firms as live_deploy_prop_firms,
)
from app.live_deploy.broker_registry import build_adapter
from app.live_deploy.execution_engine import LiveExecutionConfig, LiveExecutionSession
from app.strategy.library import list_saved_strategies
from tkinter import (
    BooleanVar,
    Button,
    Checkbutton,
    END,
    Frame,
    Label,
    Listbox,
    SINGLE,
    Text,
    filedialog,
    messagebox,
    ttk,
)
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCombo, LabeledEntry
from app.ui.widgets.cards import NeuralProgress
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
PANEL = ""
PANEL_3 = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
METAL_BRIGHT = ""
LOG_BG = ""
GREEN = ""
RED = ""
AMBER = ""
theme.bind(globals())


class DeployLiveMixin:
    """Deploy Live tab."""


    # -----------------------------------------------------------------------
    # Tab 18 — Deploy Live
    # -----------------------------------------------------------------------

    def _build_deploy_live_tab(self):
        f = self._scrollable(self.tab_deploylive)
        self._page_header(
            f,
            "DEPLOY / Deploy Live",
            "Deploy Live",
            "Connect a validated strategy directly to a real, funded prop-firm account for "
            "automated live trading -- no manual clicking required once it's running.",
        )

        warn_wrap = Frame(f, bg="#3A0E14", highlightthickness=1, highlightbackground=RED)
        warn_wrap.pack(fill="x", padx=24, pady=(0, 16))
        Label(
            warn_wrap, text="⚠  THIS TRADES REAL MONEY -- READ BEFORE CONNECTING ANY ACCOUNT",
            bg="#3A0E14", fg="#FF8FA0", font=_safe_font(10, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 4))
        Label(
            warn_wrap,
            text="This is not the same as Live Demo Test. An account connected here places real "
                 "orders against real capital in a live or funded prop-firm account. Before using this:\n"
                 "  1) Confirm your prop firm's rules actually PERMIT automated/EA trading on this "
                 "account -- many firms restrict or ban certain automation, and violating that can get "
                 "a funded account terminated regardless of performance.\n"
                 "  2) Validate the strategy thoroughly first (15 FULL PIPELINE) and run it on 16 LIVE "
                 "DEMO TEST for a meaningful stretch before ever pointing this at real capital.\n"
                 "  3) Never share account credentials anywhere else -- this app stores the password "
                 "locally via your OS's secure credential store (the same mechanism as the Live Demo "
                 "Test tab), the same way, for this reason.\n"
                 "  4) Start with the smallest account/position size your firm allows.",
            bg="#3A0E14", fg="#F0C7CC", font=_safe_font(8), wraplength=900, justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 14))

        # ---- How this actually connects to a prop firm ---------------------
        info_section = self._section(
            f, "How this connects to a prop firm -- read this first",
            "",
            emphasize=True,
        )
        Label(
            info_section,
            text="Any prop firm works as long as it gives you a login on one of these platforms -- "
                 "pick your firm below (or \"Other\") and the platform it gave you:\n\n"
                 "  •  MT4 / MT5 -- login, password and server from your firm's email (Windows + the MT5 terminal).\n"
                 "  •  cTrader -- Open API client id/secret + refresh token.\n"
                 "  •  Tradovate -- username/password + API app id/secret, cid and sec.\n"
                 "  •  TradeLocker and DXtrade -- login (and server / base URL).\n\n"
                 "Not connectable: Rithmic, NinjaTrader and TopStepX/ProjectX (the firm note explains why and "
                 "what to use instead). Only MT5 has been tested against a live account by the developer; "
                 "the other connectors follow each platform's public API docs, so TEST CONNECTION on a "
                 "demo/practice account first -- if it fails, the app tells you what to fix.",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), wraplength=900, justify="left",
        ).pack(anchor="w", padx=18, pady=(2, 14))

        # ---- Saved live accounts --------------------------------------------
        accounts_section = self._section(
            f, "Live prop-firm accounts",
            "Each saved account remembers its prop firm, platform, and login -- the password is "
            "stored the same secure way as the Live Demo Test tab (OS keyring), never in plain text.",
            emphasize=True,
        )

        list_frame = Frame(accounts_section, bg=PANEL)
        list_frame.pack(fill="both", padx=18, pady=(2, 8))
        self.dl_accounts_listbox = Listbox(
            list_frame, height=6, selectmode=SINGLE, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.dl_accounts_listbox.pack(side="left", fill="both", expand=True)
        dl_scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.dl_accounts_listbox.yview, style="T58.Vertical.TScrollbar",
        )
        dl_scrollbar.pack(side="right", fill="y")
        self.dl_accounts_listbox.config(yscrollcommand=dl_scrollbar.set)
        self._bind_isolated_wheel(self.dl_accounts_listbox)
        self.dl_accounts_listbox.bind("<<ListboxSelect>>", self._dl_on_account_selected)

        list_btn_row = Frame(accounts_section, bg=PANEL)
        list_btn_row.pack(anchor="w", padx=18, pady=(0, 14))
        self._button(list_btn_row, "DELETE SELECTED", self._dl_delete_account).pack(side="left")

        # ---- Add / edit an account -------------------------------------------
        add_section = self._section(
            f, "Add / edit an account",
            "Pick your prop firm (or Other) and platform, then enter the login details from your "
            "firm's account-issued email or dashboard.",
            emphasize=True,
        )

        firm_names = [firm.name for firm in live_deploy_prop_firms.PROP_FIRMS]
        self.dl_firm = LabeledCombo(add_section, "Prop firm", firm_names, firm_names[0] if firm_names else "Other")
        self.dl_firm.combo.bind("<<ComboboxSelected>>", lambda _e: self._dl_on_firm_changed())
        self.dl_platform = LabeledCombo(
            add_section, "Platform your firm gave you", ["MT4/MT5"], "MT4/MT5",
        )
        self.dl_platform.combo.bind("<<ComboboxSelected>>", lambda _e: self._dl_render_fields())
        self.dl_nickname = LabeledEntry(add_section, "Account nickname (yours, e.g. 'FTMO 100k #1')", "")
        # Platform-specific credential fields are built from the SAME spec the
        # web app uses (app.live_deploy.connect_flow.PLATFORM_FIELDS).
        self.dl_fields_frame = Frame(add_section, bg=PANEL)
        self.dl_fields_frame.pack(fill="x")
        self._dl_field_widgets: dict = {}

        firm_note_row = Frame(add_section, bg=PANEL)
        firm_note_row.pack(anchor="w", padx=18, pady=(0, 8))
        self.dl_firm_note = Label(
            firm_note_row, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        )
        self.dl_firm_note.pack(anchor="w")

        confirm_row = Frame(add_section, bg=PANEL)
        confirm_row.pack(anchor="w", padx=18, pady=(2, 4))
        self.dl_confirm_var = BooleanVar(value=False)
        Checkbutton(
            confirm_row, variable=self.dl_confirm_var, bg=PANEL, activebackground=PANEL,
            selectcolor=PANEL_3, highlightthickness=0,
            text="I understand this account will trade with real capital, and I've confirmed my prop "
                 "firm permits automated trading on it.",
            fg=TEXT_MUTED, font=_safe_font(8), wraplength=780, justify="left", anchor="w",
        ).pack(anchor="w")

        add_btn_row = Frame(add_section, bg=PANEL)
        add_btn_row.pack(anchor="w", padx=18, pady=(8, 14))
        self._button(add_btn_row, "SAVE ACCOUNT", self._dl_save_account, primary=True).pack(side="left")
        self._button(add_btn_row, "TEST THESE DETAILS", self._dl_test_form).pack(side="left", padx=8)
        self._button(add_btn_row, "AUTO-DETECT TERMINAL", self._dl_auto_detect_terminal).pack(side="left", padx=8)
        self._button(add_btn_row, "BROWSE...", self._dl_browse_terminal).pack(side="left")

        # ---- Market + risk (same shape as Live Demo Test's own fields) -----
        market_section = self._section(
            f, "Market",
            "Which symbol and timeframe the strategy trades on this account.",
        )
        self.dl_symbol = LabeledEntry(market_section, "Symbol", "XAUUSD")
        self.dl_timeframe = LabeledCombo(
            market_section, "Timeframe",
            ["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour", "4 hours", "1 day"],
            "15 minutes",
        )

        risk_section = self._section(
            f, "Risk (same fields as Step 04, applied to this live account)",
            "Position sizing calls the exact same RiskConfig.position_size(...) the backtester "
            "and Live Demo Test use -- if a strategy defines its own dynamic stop, that's "
            "honored first; otherwise a fixed-pip or 1%-of-price fallback applies, same "
            "precedence as everywhere else in this app.",
        )
        self.dl_risk_pct = LabeledEntry(risk_section, "Risk per trade (% of equity)", 1.0)
        self.dl_daily_loss_limit = LabeledEntry(risk_section, "Daily loss limit (% of balance, halts new entries)", 5.0)
        self.dl_max_trades_per_day = LabeledEntry(risk_section, "Max trades per day", 10)
        self.dl_pip_size = LabeledEntry(risk_section, "Pip size (leave blank to auto-detect from MT5 symbol info)", "")
        self.dl_baseline_win_rate = LabeledEntry(
            risk_section, "Backtest win rate %, for drift comparison (optional)", "",
        )

        # ---- Prop guardrails (v7 P0-2: enforced on every live session) ------
        # Before v7 these shipped DISABLED (bare PropRules defaults) with an
        # inline comment admitting the widgets didn't exist yet. They are
        # real widgets now, defaulting to futures-prop-safe values; the
        # engine additionally halts the session (flattening everything) on
        # a max-drawdown breach, which no widget can turn off.
        guard_section = self._section(
            f, "Prop guardrails (enforced)",
            "A live session can blow an evaluation on a pure rule technicality (news trade, "
            "weekend gap, hedge, oversize ticket) even with perfect strategy performance. "
            "These ship ON with futures-prop-safe defaults -- the notice below states exactly "
            "what the engine enforces.",
        )
        from app.live_deploy.guardrails import (
            DEFAULT_MAX_LOT_SIZE, default_news_blackout_entry_text,
        )
        self.dl_contract_size = LabeledEntry(
            guard_section,
            "Contract size -- sizing units per whole contract (ES=50, MES=5, NQ=20; blank = auto from futures spec)",
            "",
        )
        self.dl_units_per_lot = LabeledEntry(
            guard_section,
            "Units per broker lot (blank = auto: contract size for futures, 100,000 FX standard)",
            "",
        )
        self.dl_news_blackout = LabeledEntry(
            guard_section,
            "News blackout windows -- HH:MM-HH:MM, comma-separated (server/broker time)",
            default_news_blackout_entry_text(),
        )
        self.dl_weekend_hold_var = BooleanVar(value=False)
        Checkbutton(
            guard_section, variable=self.dl_weekend_hold_var, bg=PANEL, activebackground=PANEL,
            selectcolor=PANEL_3, highlightthickness=0,
            text="Allow holding positions over the weekend (leave OFF = flatten Friday, prop-safe)",
            fg=TEXT_MUTED, font=_safe_font(9), anchor="w",
        ).pack(anchor="w", padx=18, pady=(2, 2))
        self.dl_hedging_var = BooleanVar(value=False)
        Checkbutton(
            guard_section, variable=self.dl_hedging_var, bg=PANEL, activebackground=PANEL,
            selectcolor=PANEL_3, highlightthickness=0,
            text="Allow hedging / opposite-direction positions (leave OFF = prop-safe)",
            fg=TEXT_MUTED, font=_safe_font(9), anchor="w",
        ).pack(anchor="w", padx=18, pady=(2, 2))
        self.dl_max_lot = LabeledEntry(
            guard_section,
            "Max contracts/lots per single order (broker-native, enforced after sizing conversion)",
            DEFAULT_MAX_LOT_SIZE,
        )
        # v7 P0-2: on-screen notice stating the enforced values.
        Label(
            guard_section,
            text="ENFORCED ON EVERY SESSION -- news blackout ON "
                 f"({default_news_blackout_entry_text()}; server/broker time, editable above); "
                 "weekend holds BLOCKED (flattened Friday ~20:45 server time) unless checked above; "
                 "hedging BLOCKED unless checked above; "
                 f"max {DEFAULT_MAX_LOT_SIZE:g} contracts/lots per single order; "
                 "max-drawdown halt ARMED from your firm's rule -- a breach flattens ALL positions "
                 "and stops the session; 3 consecutive poll failures also flatten and halt.",
            bg="#1a2b1a", fg="#9fe6a0", font=_safe_font(8, "bold"),
            wraplength=900, justify="left",
        ).pack(anchor="w", padx=18, pady=(8, 14))

        # ---- Connect + deploy -------------------------------------------------
        deploy_section = self._section(
            f, "Connect and deploy",
            "Same underlying engine as Live Demo Test -- pick a saved account above, pick a "
            "validated strategy from the Strategy Library, and start. The kill switch works "
            "identically.",
            emphasize=True,
        )
        self.dl_strategy_combo = LabeledCombo(deploy_section, "Strategy (from Strategy Library)", [""], "")
        deploy_btn_row = Frame(deploy_section, bg=PANEL)
        deploy_btn_row.pack(anchor="w", padx=18, pady=(6, 6))
        self._button(deploy_btn_row, "TEST CONNECTION", self._dl_test_connection).pack(side="left")
        self.dl_start_btn = self._button(deploy_btn_row, "START LIVE TRADING", self._dl_start_clicked, primary=True)
        self.dl_start_btn.pack(side="left", padx=8)
        self.dl_start_btn.config(state="disabled")
        self.dl_stop_btn = self._button(deploy_btn_row, "STOP", self._dl_stop_clicked)
        self.dl_stop_btn.pack(side="left", padx=8)
        self.dl_stop_btn.config(state="disabled")
        self.dl_kill_btn = Button(
            deploy_btn_row, text="KILL SWITCH — FLATTEN & STOP", command=self._dl_kill_clicked,
            bg=RED, fg="#1a0a0d", activebackground="#c94656", activeforeground="#1a0a0d",
            relief="flat", bd=0, font=_safe_font(9, "bold"), padx=14, pady=8, cursor="hand2",
        )
        self.dl_kill_btn.pack(side="left", padx=8)
        self.dl_kill_btn.config(state="disabled")

        self.dl_progress = NeuralProgress(deploy_section)
        self.dl_progress.pack(fill="x", padx=18, pady=(6, 10))

        self.dl_status = Label(
            deploy_section, text="Select a saved account above, then Test Connection.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(10, "bold"), wraplength=900, justify="left",
        )
        self.dl_status.pack(anchor="w", padx=18, pady=(2, 16))

        journal_section = self._section(f, "Trade journal (this session)", "")
        columns = ("time", "direction", "volume", "entry", "exit", "pnl", "status")
        self.dl_journal_tree = ttk.Treeview(journal_section, columns=columns, show="headings", height=8)
        for col, label, w in (
            ("time", "Entry Time", 140), ("direction", "Dir", 50), ("volume", "Size", 70),
            ("entry", "Entry", 90), ("exit", "Exit", 90), ("pnl", "P&L", 80), ("status", "Status", 70),
        ):
            self.dl_journal_tree.heading(col, text=label)
            self.dl_journal_tree.column(col, width=w, anchor="center")
        self.dl_journal_tree.pack(fill="x", padx=18, pady=(2, 12))

        log_section = self._section(f, "Live log", "")
        _dl_log_frame = Frame(log_section, bg=PANEL)
        self.dl_log = Text(
            _dl_log_frame, height=16, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _dl_log_scroll = ttk.Scrollbar(
            _dl_log_frame, orient="vertical", command=self.dl_log.yview, style="T58.Vertical.TScrollbar",
        )
        self.dl_log.configure(yscrollcommand=_dl_log_scroll.set)
        self.dl_log.pack(side="left", fill="both", expand=True)
        _dl_log_scroll.pack(side="right", fill="y")
        _dl_log_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.dl_log)

        self._dl_accounts: list = []
        self._dl_editing_id: str | None = None
        self._dl_session = None
        self._dl_journal = None
        self._dl_selected_account = None
        self._dl_strategy_items: list = []
        self._dl_refresh_accounts()
        self._dl_on_firm_changed()
        self._dl_refresh_strategy_list()

    def _dl_render_fields(self, values: dict | None = None):
        """(Re)build the credential inputs for the selected platform."""
        for w in list(self.dl_fields_frame.winfo_children()):
            w.destroy()
        self._dl_field_widgets = {}
        values = values or {}
        plat = live_connect_flow.normalize_platform(self.dl_platform.get_str())
        for fld in live_connect_flow.fields_for(plat):
            label = fld["label"] + (" *" if fld.get("required") else "")
            default = values.get(fld["key"], fld.get("default", ""))
            if fld["kind"] == "select":
                labels = fld.get("option_labels", {})
                disp = [labels.get(o, o) for o in fld["options"]]
                cur = labels.get(str(default), str(default))
                w = LabeledCombo(self.dl_fields_frame, label, disp, cur)
                w._opts = {labels.get(o, o): o for o in fld["options"]}
            else:
                w = LabeledEntry(self.dl_fields_frame, label, "" if fld["kind"] == "secret" else default,
                                 secret=(fld["kind"] == "secret"))
            w.pack(anchor="w", padx=18, pady=2)
            self._dl_field_widgets[fld["key"]] = (fld, w)
            if fld.get("help"):
                Label(self.dl_fields_frame, text=fld["help"], bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
                      wraplength=820, justify="left").pack(anchor="w", padx=18)

    def _dl_form_values(self) -> dict:
        out = {}
        for key, (fld, w) in self._dl_field_widgets.items():
            raw = w.get_str()
            if fld["kind"] == "select":
                raw = getattr(w, "_opts", {}).get(raw, raw)
            out[key] = raw
        return out

    def _dl_form_account(self):
        plat = live_connect_flow.normalize_platform(self.dl_platform.get_str())
        values = self._dl_form_values()
        errs = live_connect_flow.validate_values(plat, values, editing=bool(self._dl_editing_id))
        if errs:
            return None, errs
        acct = live_connect_flow.account_from_values(
            platform=plat, firm_name=self.dl_firm.get_str(), nickname=self.dl_nickname.get_str(),
            values=values, account_id=self._dl_editing_id,
        )
        if self._dl_editing_id:
            old = next((a for a in self._dl_accounts if a.id == self._dl_editing_id), None)
            acct = live_connect_flow.merge_saved_secrets(acct, old)
        return acct, []

    def _dl_show_test_result(self, res: dict):
        if res.get("ok"):
            live = res.get("live")
            env = "LIVE environment" if live is True else ("demo/practice environment" if live is False else "environment: check your platform")
            bal = f" balance {res['balance']:,.2f}" if res.get("balance") is not None else ""
            self.dl_status.config(text=f"Connected ({env}).{bal} {res.get('message', '')}", fg=GREEN)
            self.dl_start_btn.config(state="normal" if self._dl_session is None else "disabled")
        else:
            steps = "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(res.get("fix", [])))
            self.dl_status.config(text=f"Could not connect: {res.get('message', 'unknown error')}\nWhat to do:\n{steps}", fg=RED)

    def _dl_test_form(self):
        acct, errs = self._dl_form_account()
        if acct is None:
            self.dl_status.config(text="Fill in the missing fields first:\n" + "\n".join(f"  - {e}" for e in errs), fg=AMBER)
            return
        self.dl_status.config(text="Connecting...", fg=AMBER)

        def run():
            res = live_connect_flow.test_account(acct)
            self.root.after(0, lambda: self._dl_show_test_result(res))
        threading.Thread(target=run, daemon=True).start()

    def _dl_timeframe_minutes(self) -> int:
        mapping = {
            "1 minute": 1, "5 minutes": 5, "15 minutes": 15, "30 minutes": 30,
            "1 hour": 60, "4 hours": 240, "1 day": 1440,
        }
        return mapping.get(self.dl_timeframe.get_str(), 15)

    def _dl_log_line(self, level: str, message: str):
        color = {"error": RED, "warn": AMBER, "info": TEXT}.get(level, TEXT)
        tag = f"lvl_{level}"
        self.dl_log.tag_config(tag, foreground=color)
        self.dl_log.insert(END, f"[{level.upper()}] {message}\n", tag)
        self.dl_log.see(END)
        try:
            self.root.after(0, self.root.update_idletasks)
        except Exception:
            pass

    def _dl_selected_strategy_item(self):
        label = self.dl_strategy_combo.get_str()
        for it in self._dl_strategy_items:
            if f"[{it.strategy_type}] {it.name}" == label:
                return it
        return None

    def _dl_refresh_journal_view(self):
        for row in self.dl_journal_tree.get_children():
            self.dl_journal_tree.delete(row)
        if self._dl_journal is None or self._dl_session is None or self._dl_session._session_id is None:
            return
        trades = self._dl_journal.all_trades(self._dl_session._session_id)
        for t in trades[:100]:
            import datetime

            entry_time = datetime.datetime.fromtimestamp(t.entry_time).strftime("%m-%d %H:%M")
            direction = "LONG" if t.direction == 1 else "SHORT"
            exit_price = f"{t.exit_price:.5f}" if t.exit_price else ""
            pnl = f"{t.pnl:,.2f}" if t.pnl is not None else ""
            self.dl_journal_tree.insert("", END, values=(
                entry_time, direction, f"{t.volume:.2f}", f"{t.entry_price:.5f}", exit_price, pnl, t.status,
            ))

    def _dl_update_status(self, status):
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
        self.dl_status.config(text="  |  ".join(parts), fg=color)
        self._dl_refresh_journal_view()

    def _dl_on_started(self):
        self.dl_start_btn.config(state="disabled")
        self.dl_stop_btn.config(state="normal")
        self.dl_kill_btn.config(state="normal")
        self.dl_progress.start()
        self.dl_status.config(text="Running -- placing real orders on this account.", fg=GREEN)

    def _dl_on_firm_changed(self):
        name = self.dl_firm.get_str()
        firm = live_deploy_prop_firms.find(name)
        if firm is None:
            self.dl_firm_note.config(text="")
            return
        connectable_note = (
            "" if firm.connectable_today else
            " ⚠ None of this firm's platforms are supported by this app yet -- you can save the "
            "account details below, but connecting/trading won't work until that platform is added."
        )
        self.dl_firm_note.config(
            text=f"Platform(s): {', '.join(firm.platforms)}. {firm.notes}{connectable_note}"
                 + "".join(f"\n{n['platform']}: {n['why']}" for n in next((f for f in live_connect_flow.firms_catalog() if f['name'] == firm.name), {}).get('not_connectable', []))
                 + f" Always "
                 f"confirm the exact server name from your own account-issued email -- server names "
                 f"change and vary by region/account type even within one firm.",
            fg=TEXT_DIM if firm.connectable_today else AMBER,
        )
        plats = live_connect_flow.firm_platforms(firm.name) or ["MT4/MT5"]
        self.dl_platform.combo.config(values=plats)
        if self.dl_platform.get_str() not in plats:
            self.dl_platform.var.set(plats[0])
        self._dl_render_fields()

    def _dl_refresh_accounts(self):
        self._dl_accounts = live_deploy_settings.load_accounts()
        self.dl_accounts_listbox.delete(0, END)
        for acct in self._dl_accounts:
            self.dl_accounts_listbox.insert(
                END, f"{acct.nickname}  —  {acct.firm_name} ({acct.platform})  —  login {acct.login}",
            )

    def _dl_on_account_selected(self, _event=None):
        sel = self.dl_accounts_listbox.curselection()
        if not sel:
            return
        acct = self._dl_accounts[sel[0]]
        self._dl_editing_id = acct.id
        self._dl_selected_account = acct
        self.dl_firm.var.set(acct.firm_name)
        plat = live_connect_flow.normalize_platform(acct.platform)
        plats = live_connect_flow.firm_platforms(acct.firm_name) or ["MT4/MT5"]
        if plat not in plats:
            plats = plats + [plat]
        self.dl_platform.combo.config(values=plats)
        self.dl_platform.var.set(plat)
        self.dl_nickname.var.set(acct.nickname)
        self._dl_render_fields(live_connect_flow.public_values(acct))
        self.dl_firm_note.config(text="Editing this saved account. Password is never re-displayed -- leave it "
                                       "blank to keep the existing one, or enter a new one to replace it. Same "
                                       "for any secret extra-credential fields (client secret, refresh token, "
                                       "etc.) -- they're omitted above too; only re-enter them if you're "
                                       "changing that value.")
        connectable = live_connect_flow.normalize_platform(acct.platform) in live_connect_flow.PLATFORM_FIELDS
        self.dl_start_btn.config(state="normal" if connectable and self._dl_session is None else "disabled")
        self.dl_status.config(
            text=f"Selected '{acct.nickname}'. Click TEST CONNECTION, then START LIVE TRADING when ready."
            if connectable else
            f"'{acct.nickname}' uses {acct.platform}, which isn't connectable yet -- see the note above.",
            fg=TEXT_DIM if connectable else AMBER,
        )

    def _dl_delete_account(self):
        sel = self.dl_accounts_listbox.curselection()
        if not sel:
            messagebox.showinfo("Delete account", "Select an account in the list first.")
            return
        acct = self._dl_accounts[sel[0]]
        if not messagebox.askyesno("Delete account", f"Remove the saved connection for '{acct.nickname}'? This does not affect the account at your prop firm itself."):
            return
        live_deploy_settings.delete_account(acct.id)
        self._dl_refresh_accounts()

    def _dl_set_terminal_path(self, path: str):
        entry = self._dl_field_widgets.get("terminal_path")
        if entry is None:
            messagebox.showinfo("Terminal path", "Terminal path only applies to MT4/MT5 accounts -- pick MT4/MT5 as the platform.")
            return
        entry[1].var.set(path)

    def _dl_browse_terminal(self):
        path = filedialog.askopenfilename(
            title="Locate your MT5/MT4 terminal executable",
            filetypes=[("Terminal executable", "terminal64.exe terminal.exe"), ("All files", "*.*")],
        )
        if path:
            self._dl_set_terminal_path(path)

    def _dl_auto_detect_terminal(self):
        candidates = mt5_connector_module.find_terminal_candidates()
        if not candidates:
            messagebox.showinfo("Auto-detect", "Couldn't find a terminal in any common install location. Use Browse... instead.")
            return
        self._dl_set_terminal_path(candidates[0])

    def _dl_save_account(self):
        if not self.dl_confirm_var.get():
            messagebox.showwarning(
                "Confirmation required",
                "Check the confirmation box above before saving a live account -- this is a real-money "
                "account, not a demo.",
            )
            return
        acct, errs = self._dl_form_account()
        if acct is None:
            messagebox.showwarning("Missing fields", "\n".join(errs))
            return
        live_deploy_settings.save_account(acct)
        self._dl_editing_id = None
        self.dl_confirm_var.set(False)
        self._dl_refresh_accounts()
        self._dl_render_fields()
        messagebox.showinfo("Saved", "Account saved. Select it in the list above, then Test Connection.")


    def _dl_refresh_strategy_list(self):
        try:
            items = list_saved_strategies(None)
        except Exception:
            items = []
        self._dl_strategy_items = items
        labels = [f"[{it.strategy_type}] {it.name}" for it in items]
        self.dl_strategy_combo.combo.config(values=labels or [""])
        if labels and not self.dl_strategy_combo.get_str():
            self.dl_strategy_combo.var.set(labels[0])

    def _dl_test_connection(self):
        sel = self.dl_accounts_listbox.curselection()
        if not sel:
            messagebox.showinfo("Test connection", "Select a saved account first.")
            return
        acct = self._dl_accounts[sel[0]]
        self.dl_status.config(text="Connecting...", fg=AMBER)
        self.root.update_idletasks()

        def run():
            res = live_connect_flow.test_account(acct)
            self.root.after(0, lambda: self._dl_show_test_result(res))

        threading.Thread(target=run, daemon=True).start()

    def _dl_preflight_dialog(self, acct, item) -> bool:
        """Same pre-deploy checklist as the web page. Returns True to proceed."""
        meta = None
        try:
            from app.strategy.library import load_strategy_metadata
            meta = load_strategy_metadata(item.strategy_type, item.name)
        except Exception:
            meta = None
        pf = live_connect_flow.preflight_checklist(acct, strategy_meta=meta)
        icon = {"ok": "[OK]   ", "warn": "[CHECK]", "ack": "[STOP] ", "block": "[BLOCK]"}
        text = "\n\n".join(f"{icon[i['level']]} {i['title']}" + (f"\n        {i['detail']}" if i["detail"] else "") for i in pf["items"])
        if not pf["can_deploy"]:
            messagebox.showerror("Cannot deploy yet", text)
            return False
        if pf["needs_ack"]:
            return messagebox.askyesno(
                "This firm does not clearly permit automation",
                text + "\n\nDo you have the firm's WRITTEN permission to run this automation, and accept the risk "
                       "of account termination?", icon="warning")
        return messagebox.askyesno("Pre-deploy checklist", text + "\n\nContinue to the final confirmation?")


    def _dl_start_clicked(self):
        sel = self.dl_accounts_listbox.curselection()
        if not sel:
            messagebox.showwarning("No account selected", "Select a saved live account first.")
            return
        acct = self._dl_accounts[sel[0]]
        if live_connect_flow.normalize_platform(acct.platform) not in live_connect_flow.PLATFORM_FIELDS:
            messagebox.showwarning("Not supported yet", f"{acct.platform} isn't wired up yet -- see the note above.")
            return
        if live_connect_flow.normalize_platform(acct.platform) == "MT4/MT5" and not mt5_connector_module.is_available():
            messagebox.showwarning("MT5 not available", mt5_connector_module.unavailable_reason())
            return

        item = self._dl_selected_strategy_item()
        if item is None:
            messagebox.showwarning("No strategy selected", "Choose a strategy from the Strategy Library first.")
            return
        if not self._dl_preflight_dialog(acct, item):
            return

        symbol = self.dl_symbol.get_str().strip()
        if not symbol:
            messagebox.showwarning("Missing symbol", "Enter the symbol to trade (must match your broker's Market Watch name).")
            return

        if not messagebox.askyesno(
            "Confirm live trading",
            f"This will place REAL orders with REAL money on account '{acct.nickname}' "
            f"({acct.firm_name}, login {acct.login}) trading strategy '{item.name}' on {symbol}.\n\n"
            "Have you validated this strategy (15 FULL PIPELINE) and run it on Live Demo Test first, "
            "and confirmed your prop firm permits automated trading on this account?\n\n"
            "This cannot be undone once trades are placed. Continue?",
            icon="warning",
        ):
            return

        try:
            strategy = self._ft_build_strategy_instance(item)
        except Exception as exc:
            messagebox.showerror("Strategy error", f"Could not load strategy: {exc}")
            return

        try:
            broker = build_adapter(acct)
        except Exception as exc:
            messagebox.showerror("Connection error", str(exc))
            return
        pip_size_str = self.dl_pip_size.get_str().strip()
        pip_size = None
        if pip_size_str:
            try:
                pip_size = float(pip_size_str)
            except ValueError:
                pip_size = None

        def resolve_pip_and_start():
            nonlocal pip_size
            try:
                probe = build_adapter(acct)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._dl_log_line("error", str(exc)))
                self.root.after(0, lambda: self.dl_start_btn.config(state="normal"))
                return
            conn = probe.connect()
            if not conn.ok:
                self.root.after(0, lambda: self._dl_log_line("error", conn.message))
                self.root.after(0, lambda: self.dl_start_btn.config(state="normal"))
                return
            if pip_size is None:
                # symbol_point() is an MT5-specific convenience (reads the broker's own
                # minimum price increment) -- not part of the common BrokerAdapter interface,
                # since not every platform exposes an equivalent single call. Auto-detect
                # where available; otherwise fall back the same way MT5 itself does when
                # detection fails.
                if hasattr(probe, "symbol_point"):
                    try:
                        pip_size = probe.symbol_point(symbol)
                    except Exception as exc:
                        self.root.after(0, lambda exc=exc: self._dl_log_line(
                            "warn", f"Could not auto-detect pip size ({exc}); falling back to 0.0001."))
                        pip_size = 0.0001
                else:
                    self.root.after(0, lambda: self._dl_log_line(
                        "info", f"{acct.platform} has no pip-size auto-detect yet -- using 0.0001. "
                                "Set the Pip size field above explicitly if that's wrong for this symbol."))
                    pip_size = 0.0001
            probe.disconnect()

            # v7 P0-1: resolve the sizing-unit -> broker-quantity conversion
            # factors. contract_size: explicit field first, else auto from
            # the futures instrument spec (ESZ25 -> ES). units_per_lot:
            # explicit field, else contract_size for futures, else the FX
            # standard 100,000. Refuse to start when Tradovate (whole
            # contracts) has no contract_size -- guessing here is how 50x
            # orders happen; every other adapter refuses inside
            # to_broker_qty instead of guessing.
            from app.live_deploy.guardrails import (
                DEFAULT_MAX_LOT_SIZE, contract_size_for_symbol,
                enforced_guardrails_summary, futures_prop_safe_rules,
                normalize_blackout_text, resolve_units_per_lot,
            )
            contract_size = None
            cs_str = self.dl_contract_size.get_str().strip()
            if cs_str:
                try:
                    contract_size = float(cs_str)
                except ValueError:
                    contract_size = None
            if contract_size is None:
                contract_size = contract_size_for_symbol(symbol)
            if contract_size is not None and contract_size <= 0:
                contract_size = None
            if contract_size is None and live_connect_flow.normalize_platform(acct.platform) == "Tradovate":
                self.root.after(0, lambda: messagebox.showerror(
                    "Missing contract size",
                    f"Tradovate orders are whole contracts and '{symbol}' is not a known futures symbol -- "
                    "enter the contract size explicitly (sizing units per whole contract, e.g. ES=50, MES=5)."))
                self.root.after(0, lambda: self.dl_start_btn.config(state="normal"))
                return
            upl_str = self.dl_units_per_lot.get_str().strip()
            units_per_lot = None
            if upl_str:
                try:
                    units_per_lot = float(upl_str)
                except ValueError:
                    units_per_lot = None
            units_per_lot = resolve_units_per_lot(contract_size, units_per_lot)

            risk = RiskConfig(
                initial_balance=conn.balance or 10_000.0,
                risk_mode="percent",
                risk_value=self.dl_risk_pct.get_float(1.0),
                max_trades_per_day=self.dl_max_trades_per_day.get_int(10),
                pip_size=pip_size,
                contract_size=contract_size,
                daily_loss_limit_pct=self.dl_daily_loss_limit.get_float(5.0) or None,
            )
            baseline_str = self.dl_baseline_win_rate.get_str().strip()
            baseline_win_rate = float(baseline_str) if baseline_str else None

            # v7 P0-2: the tab's own guardrail widgets, defaulting to the
            # futures-prop-safe values (blackout ON, no weekend hold, no
            # hedging, per-order lot cap). The engine additionally enforces
            # the max-drawdown halt on every poll -- no widget disables it.
            try:
                max_lot = float(self.dl_max_lot.get_str().strip())
            except ValueError:
                max_lot = DEFAULT_MAX_LOT_SIZE
            if max_lot <= 0:
                max_lot = None
            prop_rules = futures_prop_safe_rules(
                account_size=conn.balance or 10_000.0,
                news_blackout_windows=normalize_blackout_text(self.dl_news_blackout.get_str()),
                weekend_hold_allowed=bool(self.dl_weekend_hold_var.get()),
                hedging_allowed=bool(self.dl_hedging_var.get()),
                max_lot_size=max_lot,
                daily_loss_limit_pct=self.dl_daily_loss_limit.get_float(5.0) or None,
            )

            cfg = LiveExecutionConfig(
                symbol=symbol, timeframe_minutes=self._dl_timeframe_minutes(),
                risk=risk, prop_rules=prop_rules, baseline_win_rate=baseline_win_rate,
                contract_size=contract_size, units_per_lot=units_per_lot,
            )
            self._dl_journal = ForwardTestJournal()
            session = LiveExecutionSession(
                strategy=strategy, strategy_type=item.strategy_type, strategy_filename=item.name,
                broker=broker, journal=self._dl_journal, config=cfg,
                on_log=lambda level, msg: self.root.after(0, lambda: self._dl_log_line(level, msg)),
                on_status=lambda status: self.root.after(0, lambda: self._dl_update_status(status)),
            )
            ok, msg = session.start()
            if not ok:
                self.root.after(0, lambda: self._dl_log_line("error", msg))
                self.root.after(0, lambda: self.dl_start_btn.config(state="normal"))
                return
            self._dl_session = session
            self.root.after(0, lambda: self._dl_log_line("warn", enforced_guardrails_summary(prop_rules)))
            self.root.after(0, self._dl_on_started)

        threading.Thread(target=resolve_pip_and_start, daemon=True).start()
        self.dl_start_btn.config(state="disabled")
        self.dl_status.config(text="Connecting...", fg=AMBER)
        self._dl_log_line("info", f"Starting live trading on '{acct.nickname}' ({symbol}, {item.name})...")

    def _dl_stop_clicked(self):
        if self._dl_session is None:
            return
        self.dl_status.config(text="Stopping...", fg=AMBER)
        threading.Thread(target=self._dl_session.stop, daemon=True).start()
        self.dl_start_btn.config(state="normal")
        self.dl_stop_btn.config(state="disabled")
        self.dl_kill_btn.config(state="disabled")
        self.dl_progress.stop()
        self._dl_session = None

    def _dl_kill_clicked(self):
        if self._dl_session is None:
            return
        if not messagebox.askyesno(
            "Kill switch",
            "This closes every open REAL position on this account immediately and stops the "
            "session. Continue?",
        ):
            return
        self.dl_status.config(text="Flattening and stopping...", fg=RED)
        session = self._dl_session
        threading.Thread(target=session.flatten_all_and_stop, daemon=True).start()
        self.dl_start_btn.config(state="normal")
        self.dl_stop_btn.config(state="disabled")
        self.dl_kill_btn.config(state="disabled")
        self.dl_progress.stop()
        self._dl_session = None
