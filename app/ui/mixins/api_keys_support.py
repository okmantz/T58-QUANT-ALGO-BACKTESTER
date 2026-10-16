"""API keys settings and Support tabs."""
from __future__ import annotations

import json
import threading
import webbrowser
from app.data.alpaca_source import test_connection
from tkinter import Frame, Label, Text
from app.ui.common import _safe_font
from app.ui.widgets.basic import LabeledEntry
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
PANEL_3 = ""
BORDER = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
GREEN = ""
RED = ""
BLUE = ""
AMBER = ""
NEON_CYAN = ""
NEON_LIME = ""
NEON_VIOLET = ""
theme.bind(globals())


class ApiKeysSupportMixin:
    """API keys settings and Support tabs."""


    # -----------------------------------------------------------------------
    # API Keys tab -- desktop parity with the web app's own consolidated
    # /settings/api-keys page: every integration's credentials in one
    # place instead of configured separately wherever each is used.
    # Same four categories, same backend modules, so a key saved on
    # either build shows up on the other (all local JSON/keyring under
    # this computer's app data folder -- see each module's own file).
    # -----------------------------------------------------------------------
    def _build_api_keys_tab(self):
        from app.accounts.api_keys import load_settings as load_api_keys_settings
        from app.data.alpaca_credentials import load_credentials as load_alpaca_credentials

        f = self._scrollable(self.tab_api_keys)
        self._page_header(
            f,
            "ACCOUNT / API Keys",
            "\U0001F511 API Keys",
            "Every integration's credentials, in one place, used across every tab that needs them -- "
            "instead of configuring the same key separately wherever it's used. Desktop parity with the "
            "web app's own /settings/api-keys page -- both read/write the exact same local files, so a "
            "key saved on either build shows up on the other.",
        )

        saved_keys = load_api_keys_settings()
        alpaca_creds = load_alpaca_credentials()

        # ================= AI =================
        Label(
            f, text="\U0001F9E0  AI", bg=BG, fg=NEON_VIOLET, font=_safe_font(11, "bold"),
        ).pack(anchor="w", padx=24, pady=(4, 2))
        self._build_ai_assist_section(f, prefix="apikeys")
        ai_section = self._section(f, "OpenAI & Claude")
        self.apikeys_openai = LabeledEntry(ai_section, "OpenAI API key", saved_keys.openai_api_key, secret=True, width=44)
        self.apikeys_claude = LabeledEntry(ai_section, "Claude / Anthropic API key", saved_keys.claude_api_key, secret=True, width=44)
        ai_btn_row = Frame(ai_section, bg=PANEL)
        ai_btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self._button(ai_btn_row, "SAVE AI KEYS", self._save_apikeys_ai, primary=True).pack(side="left")
        self._button(ai_btn_row, "TEST OPENAI", lambda: self._test_apikeys_simple_key("openai")).pack(side="left", padx=8)
        self._button(ai_btn_row, "TEST CLAUDE", lambda: self._test_apikeys_simple_key("claude")).pack(side="left")
        self.apikeys_ai_status = Label(ai_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=820, justify="left")
        self.apikeys_ai_status.pack(anchor="w", padx=18, pady=(4, 12))

        # ================= TRADING =================
        Label(
            f, text="\U0001F4C8  Trading", bg=BG, fg=NEON_CYAN, font=_safe_font(11, "bold"),
        ).pack(anchor="w", padx=24, pady=(14, 2))
        trading_section = self._section(
            f, "Alpaca (market data)",
            "Used for fetching market data across the app -- not for live order execution. For "
            "MT5/cTrader/Tradovate/TradeLocker/DXtrade live or demo trading accounts, see Brokers / "
            "Prop Firms below.",
        )
        self.apikeys_alpaca_key = LabeledEntry(
            trading_section, "Alpaca API key", (alpaca_creds.api_key if alpaca_creds else ""), secret=True, width=44,
        )
        self.apikeys_alpaca_secret = LabeledEntry(
            trading_section, "Alpaca secret key", (alpaca_creds.secret_key if alpaca_creds else ""), secret=True, width=44,
        )
        trading_btn_row = Frame(trading_section, bg=PANEL)
        trading_btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self._button(trading_btn_row, "SAVE", self._save_apikeys_alpaca, primary=True).pack(side="left")
        self._button(trading_btn_row, "TEST CONNECTION", self._test_apikeys_alpaca).pack(side="left", padx=8)
        self.apikeys_trading_status = Label(trading_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=820, justify="left")
        self.apikeys_trading_status.pack(anchor="w", padx=18, pady=(4, 12))

        # ================= DATA =================
        Label(
            f, text="\U0001F4CA  Data", bg=BG, fg=BLUE, font=_safe_font(11, "bold"),
        ).pack(anchor="w", padx=24, pady=(14, 2))
        data_section = self._section(f, "FRED (economic data)", "Used by the AI Assistant tab to pull economic news and events.")
        self.apikeys_fred = LabeledEntry(data_section, "FRED API key", saved_keys.fred_api_key, secret=True, width=44)
        fred_btn_row = Frame(data_section, bg=PANEL)
        fred_btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self._button(fred_btn_row, "SAVE", self._save_apikeys_data, primary=True).pack(side="left")
        self._button(fred_btn_row, "TEST CONNECTION", self._test_apikeys_fred).pack(side="left", padx=8)
        self.apikeys_fred_status = Label(data_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=820, justify="left")
        self.apikeys_fred_status.pack(anchor="w", padx=18, pady=(4, 4))

        lse_section = self._section(
            f, "London Strategic Edge",
            "Market data (stocks, forex, crypto, commodities, indices, ETFs, futures -- one key covers "
            "all of them). Used by 2 Market Data's \"Fetch & Save\" flow, same as Alpaca.",
        )
        self.apikeys_lse = LabeledEntry(lse_section, "API key", saved_keys.london_strategic_edge_key, secret=True, width=44)
        lse_btn_row = Frame(lse_section, bg=PANEL)
        lse_btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self._button(lse_btn_row, "SAVE", self._save_apikeys_data, primary=True).pack(side="left")
        self._button(lse_btn_row, "TEST CONNECTION", self._test_apikeys_lse).pack(side="left", padx=8)
        self.apikeys_lse_status = Label(lse_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=820, justify="left")
        self.apikeys_lse_status.pack(anchor="w", padx=18, pady=(4, 12))

        # ================= BROKERS / PROP FIRMS =================
        Label(
            f, text="\U0001F3DB  Brokers / Prop Firms", bg=BG, fg=NEON_LIME, font=_safe_font(11, "bold"),
        ).pack(anchor="w", padx=24, pady=(14, 2))
        brokers_section = self._section(
            f, "Saved accounts",
            "Add, edit, and test accounts (MT4/MT5, cTrader, Tradovate, TradeLocker, DXtrade) on the "
            "Deploy Live tab -- save as many as you have, e.g. two Tradovate accounts and two "
            "TradeLocker accounts at once. Shown here read-only for a one-place overview.",
        )
        self.apikeys_brokers_list_frame = Frame(brokers_section, bg=PANEL)
        self.apikeys_brokers_list_frame.pack(fill="x", padx=18, pady=(4, 8))
        self._refresh_apikeys_broker_list()
        self._button(
            brokers_section, "OPEN DEPLOY LIVE TO ADD/EDIT ACCOUNTS", lambda: self._show_page("deploylive"),
        ).pack(anchor="w", padx=18, pady=(0, 12))

    def _refresh_apikeys_broker_list(self):
        from app.live_deploy.live_settings import load_accounts

        for child in list(self.apikeys_brokers_list_frame.winfo_children()):
            child.destroy()
        accounts = load_accounts()
        if not accounts:
            Label(
                self.apikeys_brokers_list_frame, text="No broker/prop-firm accounts saved yet.",
                bg=PANEL, fg=TEXT_DIM, font=_safe_font(9),
            ).pack(anchor="w")
            return
        for acct in accounts:
            row = Frame(self.apikeys_brokers_list_frame, bg=PANEL_3)
            row.pack(fill="x", pady=2)
            Label(
                row, text=f"  {acct.nickname or acct.login}  \u2014  {acct.platform}"
                + (f"  ({acct.firm_name})" if acct.firm_name else ""),
                bg=PANEL_3, fg=TEXT, font=_safe_font(9), anchor="w",
            ).pack(side="left", fill="x", expand=True, ipady=4)

    def _save_apikeys_ai(self):
        from app.accounts.api_keys import load_settings, save_settings

        self._build_ollama_settings("apikeys")  # saves Ollama fields via the shared section
        existing = load_settings()
        import dataclasses as _dc
        save_settings(_dc.replace(
            existing,
            openai_api_key=self.apikeys_openai.get_str().strip(),
            claude_api_key=self.apikeys_claude.get_str().strip(),
        ))
        self.apikeys_ai_status.config(text="\u25cf  AI keys saved.", fg=GREEN)

    def _save_apikeys_alpaca(self):
        from app.data.alpaca_credentials import save_credentials

        api_key = self.apikeys_alpaca_key.get_str().strip()
        secret_key = self.apikeys_alpaca_secret.get_str().strip()
        if not api_key or not secret_key:
            self.apikeys_trading_status.config(text="Enter both an API key and a secret key first.", fg=AMBER)
            return
        save_credentials(api_key, secret_key)
        self.apikeys_trading_status.config(text="\u25cf  Alpaca keys saved.", fg=GREEN)

    def _save_apikeys_data(self):
        from app.accounts.api_keys import load_settings, save_settings

        existing = load_settings()
        import dataclasses as _dc
        save_settings(_dc.replace(
            existing,
            fred_api_key=self.apikeys_fred.get_str().strip(),
            london_strategic_edge_key=self.apikeys_lse.get_str().strip(),
        ))
        self.apikeys_fred_status.config(text="\u25cf  Saved.", fg=GREEN)
        self.apikeys_lse_status.config(text="\u25cf  Saved.", fg=GREEN)

    def _test_apikeys_alpaca(self):

        api_key = self.apikeys_alpaca_key.get_str().strip()
        secret_key = self.apikeys_alpaca_secret.get_str().strip()
        if not api_key or not secret_key:
            self.apikeys_trading_status.config(text="No Alpaca credentials saved yet.", fg=AMBER)
            return
        self.apikeys_trading_status.config(text="Testing connection...", fg=AMBER)
        self.root.update_idletasks()

        def run():
            try:
                message = test_connection(api_key, secret_key)
                ok = True
            except Exception as exc:  # noqa: BLE001
                message, ok = str(exc), False

            def _finish():
                self.apikeys_trading_status.config(text=message, fg=(GREEN if ok else RED))
            # Direct call rather than self.root.after(0, _finish) --
            # see the identical fix (and full explanation) in
            # _search_auto_build_ensemble's own comment above.
            _finish()

        threading.Thread(target=run, daemon=True).start()

    def _test_apikeys_fred(self):
        key = self.apikeys_fred.get_str().strip()
        if not key:
            self.apikeys_fred_status.config(text="No FRED API key saved yet.", fg=AMBER)
            return
        self.apikeys_fred_status.config(text="Testing connection...", fg=AMBER)
        self.root.update_idletasks()

        def run():
            try:
                import urllib.request
                url = f"https://api.stlouisfed.org/fred/series?series_id=GDP&api_key={key}&file_type=json"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    ok = resp.status == 200
                    message = "FRED API key is valid." if ok else f"Unexpected status {resp.status}."
            except Exception as exc:  # noqa: BLE001
                ok, message = False, f"FRED rejected the key or is unreachable: {exc}"

            def _finish():
                self.apikeys_fred_status.config(text=message, fg=(GREEN if ok else RED))
            # Direct call rather than self.root.after(0, _finish) --
            # see the identical fix (and full explanation) in
            # _search_auto_build_ensemble's own comment above.
            _finish()

        threading.Thread(target=run, daemon=True).start()

    def _test_apikeys_lse(self):
        key = self.apikeys_lse.get_str().strip()
        if not key:
            self.apikeys_lse_status.config(text="No London Strategic Edge API key saved yet.", fg=AMBER)
            return
        self.apikeys_lse_status.config(text="Testing connection...", fg=AMBER)
        self.root.update_idletasks()

        def run():
            try:
                from app.data.london_strategic_edge_source import LSEFetchError, LSEImportError, test_connection
                message = test_connection(key)
                ok = True
            except (LSEImportError, LSEFetchError) as exc:
                ok, message = False, str(exc)
            except Exception as exc:  # noqa: BLE001
                ok, message = False, f"Could not connect to London Strategic Edge: {exc}"

            def _finish():
                self.apikeys_lse_status.config(text=message, fg=(GREEN if ok else RED))
            # Direct call rather than self.root.after(0, _finish) -- see
            # the identical fix (and full explanation) in
            # _search_auto_build_ensemble's own comment above.
            _finish()

        threading.Thread(target=run, daemon=True).start()

    def _test_apikeys_simple_key(self, service: str):
        key = (self.apikeys_openai.get_str() if service == "openai" else self.apikeys_claude.get_str()).strip()
        if not key:
            self.apikeys_ai_status.config(text=f"No {service} API key saved yet.", fg=AMBER)
            return
        self.apikeys_ai_status.config(text="Testing connection...", fg=AMBER)
        self.root.update_idletasks()

        def run():
            try:
                import urllib.request
                if service == "openai":
                    req = urllib.request.Request(
                        "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"},
                    )
                else:
                    req = urllib.request.Request(
                        "https://api.anthropic.com/v1/models",
                        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                    )
                urllib.request.urlopen(req, timeout=10)
                ok, message = True, f"{service.title()} key is valid."
            except Exception as exc:  # noqa: BLE001
                ok, message = False, f"{service.title()} rejected the key or is unreachable: {exc}"

            def _finish():
                self.apikeys_ai_status.config(text=message, fg=(GREEN if ok else RED))
            # Direct call rather than self.root.after(0, _finish) --
            # see the identical fix (and full explanation) in
            # _search_auto_build_ensemble's own comment above.
            _finish()

        threading.Thread(target=run, daemon=True).start()

    # -----------------------------------------------------------------------
    # Support tab -- desktop parity with the web app's own /support page:
    # a link to Education, the Discord community link, an FAQ, and a
    # local issue-report log (this app has no real support backend on
    # either build -- see the web page's own copy for why).
    # -----------------------------------------------------------------------
    def _build_support_tab(self):
        f = self._scrollable(self.tab_support)
        self._page_header(
            f,
            "ACCOUNT / Support",
            "\U0001F6DF Support",
            "Get help from the community, learn the app, or log an issue.",
        )

        edu_section = self._section(
            f, "Learn the app",
            "The Education tab walks through building your first strategy and reading a backtest "
            "report, step by step.",
            emphasize=True,
        )
        self._button(edu_section, "GO TO EDUCATION", lambda: self._show_page("education"), primary=True).pack(
            anchor="w", padx=18, pady=(0, 12),
        )

        community_section = self._section(f, "Community", "Join the Discord for help, discussion, and updates.", emphasize=True)
        self._button(community_section, "JOIN DISCORD", self._support_open_discord, primary=True).pack(
            anchor="w", padx=18, pady=(0, 12),
        )

        faq_section = self._section(f, "FAQ", emphasize=True)
        faq_items = [
            ("How do I create a strategy?",
             "Use Forge Strategy under Create to build one from scratch (Manual, Python, PineScript, "
             "or MQL5), or start from a template in the Strategy Library."),
            ("What's the difference between Quick Optimize and Full Pipeline?",
             "Quick Optimize runs the walk-forward-aware GA search on its own for a fast iteration "
             "loop. Full Pipeline runs the complete 15-step process -- baseline, search, final Monte "
             "Carlo, out-of-sample checks, and a READY/MARGINAL/NOT READY verdict -- and is what you'd "
             "run before trusting a result."),
            ("Why does a search say a candidate has zero trades?",
             "The strategy's entry conditions never fired on that dataset/timeframe combination -- "
             "widen the parameter ranges, check the timeframe matches what the strategy expects, or "
             "try a different instrument."),
            ("What do Search Lab, Evolution Lab, and Iterative Refinement each do differently?",
             "Search Lab tries many different STRATEGY FAMILIES/hypotheses. Evolution Lab runs a "
             "longer, multi-generation search within families. Iterative Refinement takes a strategy "
             "you already have and tunes its existing numeric parameters via a genetic algorithm (or "
             "TPE/CMA-ES)."),
            ("Can I use MT5, cTrader, Tradovate, TradeLocker, or DXtrade?",
             "Yes -- add an account under API Keys \u2192 Brokers/Prop Firms, then use Forward Test "
             "(demo) or Deploy Live."),
        ]
        for question, answer in faq_items:
            Label(
                faq_section, text=f"\u2022 {question}", bg=PANEL, fg=TEXT, font=_safe_font(9, "bold"),
                anchor="w", wraplength=820, justify="left",
            ).pack(anchor="w", padx=18, pady=(6, 0))
            Label(
                faq_section, text=answer, bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8),
                anchor="w", wraplength=820, justify="left",
            ).pack(anchor="w", padx=30, pady=(2, 4))

        issue_section = self._section(
            f, "Report an issue",
            "This app has no external support backend -- there's no ticket system on the other end of "
            "this form. What you type here is saved to a local file (data/config/issue_reports.jsonl) "
            "on this machine, so you have a running log of things to fix or follow up on yourself, or "
            "to paste elsewhere (an email, a GitHub issue, a message to whoever's helping you build "
            "this) when you're ready.",
            emphasize=True,
        )
        self.support_issue_title = LabeledEntry(issue_section, "Short summary", "", width=48)
        Label(issue_section, text="Details", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9)).pack(anchor="w", padx=18, pady=(8, 2))
        body_frame = Frame(issue_section, bg=PANEL)
        body_frame.pack(fill="x", padx=18, pady=(0, 8))
        self.support_issue_body = Text(
            body_frame, height=6, bg=PANEL_3, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0,
            font=_safe_font(9), wrap="word", highlightthickness=1, highlightbackground=BORDER,
        )
        self.support_issue_body.pack(fill="x")
        self._button(issue_section, "SAVE REPORT", self._support_save_issue, primary=True).pack(anchor="w", padx=18)
        self.support_issue_status = Label(issue_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8))
        self.support_issue_status.pack(anchor="w", padx=18, pady=(6, 12))

    def _support_open_discord(self):

        webbrowser.open("https://discord.gg/3MbKm3S2zG")

    def _support_save_issue(self):
        from datetime import datetime, timezone

        from app.data.storage import get_app_base_dir

        title = self.support_issue_title.get_str().strip()
        body = self.support_issue_body.get("1.0", "end").strip()
        if not title and not body:
            self.support_issue_status.config(text="Nothing to save.", fg=AMBER)
            return
        log_dir = get_app_base_dir() / "data" / "config"
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "title": title, "body": body}
        with open(log_dir / "issue_reports.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        self.support_issue_title.var.set("")
        self.support_issue_body.delete("1.0", "end")
        self.support_issue_status.config(text="\u25cf  Saved locally.", fg=GREEN)
