"""AI Assistant, Ollama settings and Options Outlook tabs."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ai.trading_assistant import TradingAssistantClient

import app.ai.ollama_settings as ollama_settings_module
import threading
from app.ai.ollama_settings import OllamaSettings
from pathlib import Path
from tkinter import END, Frame, Label, Listbox, StringVar, Text, filedialog, messagebox, ttk
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
PANEL = ""
PANEL_2 = ""
PANEL_3 = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
LOG_BG = ""
GREEN = ""
RED = ""
AMBER = ""
ACCENT = ""
theme.bind(globals())


class AiAssistantMixin:
    """AI Assistant, Ollama settings and Options Outlook tabs."""


    def _build_ai_assist_section(self, parent, prefix: str = "ai"):
        """Optional local-Ollama AI assistant (see app.ai.ollama_client):
        off by default and everywhere. When enabled, Step 2's search asks
        a local Ollama model for candidate parameter values once per
        generation, seeded into the population alongside the normal
        random/bred candidates -- the model only ever proposes numbers
        for the strategy's already-discovered tunable parameters, never
        code, and every suggestion still goes through the exact same
        backtest/prop-sim/Monte Carlo evaluation as any other candidate.
        Any failure to reach Ollama (not running, wrong host, model not
        pulled) degrades silently to the search running exactly as if
        this were disabled.

        `prefix` namespaces the widget attributes (e.g. "genstrat_ai" ->
        self.genstrat_ai_enabled) so more than one tab can each have its
        own independent copy of this section without colliding -- the
        Generate Strategies tab uses this for actual code generation
        (see app.ai.strategy_generator), a materially different, riskier
        use of the same underlying Ollama connection than the numeric-only
        parameter suggestions this section was originally written for."""
        section = self._section(
            parent, "AI Assist (optional, local Ollama)",
            "Off by default. When enabled, a local Ollama model suggests parameter values "
            "to try during Step 2's search, once per generation, alongside the normal "
            "random search -- every suggestion still has to pass the exact same "
            "backtest/prop-simulation/Monte Carlo checks as anything else. Requires "
            "Ollama installed and running locally (https://ollama.com) -- nothing is sent "
            "anywhere except to the host you configure below.",
        )
        saved = ollama_settings_module.load_settings()
        setattr(self, f"{prefix}_enabled", LabeledCheckbox(section, "Enable AI Assist for this run", saved.enabled))
        setattr(self, f"{prefix}_host", LabeledEntry(section, "Ollama host", saved.host))
        setattr(self, f"{prefix}_model", LabeledEntry(
            section, "Model (must already be pulled, e.g. `ollama pull llama3.1`)", saved.model,
        ))
        setattr(self, f"{prefix}_api_key", LabeledEntry(
            section, "API key (optional -- only for a remote/proxied Ollama)", saved.api_key, secret=True,
        ))

        btn_row = Frame(section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(2, 8))
        test_btn = self._button(btn_row, "TEST CONNECTION", lambda: self._test_ollama_connection(prefix), primary=True)
        test_btn.pack(side="left")
        setattr(self, f"{prefix}_test_btn", test_btn)
        status = Label(section, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left")
        status.pack(anchor="w", padx=18, pady=(0, 12))
        setattr(self, f"{prefix}_status", status)

    def _build_ollama_settings(self, prefix: str = "ai") -> "OllamaSettings":
        settings = OllamaSettings(
            enabled=getattr(self, f"{prefix}_enabled").get(),
            host=getattr(self, f"{prefix}_host").get_str().strip() or ollama_settings_module.DEFAULT_HOST,
            model=getattr(self, f"{prefix}_model").get_str().strip() or ollama_settings_module.DEFAULT_MODEL,
            api_key=getattr(self, f"{prefix}_api_key").get_str().strip(),
        )
        ollama_settings_module.save_settings(settings)
        return settings

    def _test_ollama_connection(self, prefix: str = "ai"):
        from app.ai.ollama_client import OllamaClient

        settings = self._build_ollama_settings(prefix)
        test_btn = getattr(self, f"{prefix}_test_btn")
        status = getattr(self, f"{prefix}_status")
        test_btn.config(state="disabled")
        status.config(text="Testing connection...", fg=AMBER)

        def run():
            try:
                ok, message = OllamaClient(settings).test_connection()
                if ok:
                    # Best-effort, same background thread -- if this fails
                    # or times out, the connection test itself still
                    # succeeded and is reported as such; the first real
                    # request just pays the normal load cost as before.
                    try:
                        OllamaClient(settings).warm_up()
                    except Exception:
                        pass
            except Exception as exc:
                ok, message = False, f"Unexpected error: {exc}"

            def _finish():
                status.config(text=message, fg=GREEN if ok else RED)
                test_btn.config(state="normal")

            # Runs on a background thread -- Tkinter widgets are only safe
            # to touch from the main thread, so hand the update back to it.
            try:
                self.root.after(0, _finish)
            except Exception:
                pass

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # AI Assistant tab -- chat + chart/trade screenshot analysis, backed
    # by app.ai.trading_assistant.TradingAssistantClient (the same
    # Ollama-based client the web app's /assistant page uses) reasoning
    # over Owen's exact strategy documents (see PERSONAL_STRATEGY_PROMPT /
    # PERSONAL_ASSISTANT_PROMPT in that module).
    # ------------------------------------------------------------------

    def _build_ai_assistant_tab(self):
        f = self._scrollable(self.tab_ai_assistant)
        self._page_header(
            f, "AI ASSISTANT", "T58 AI",
            "Chat against Owen's exact strategy hierarchy (macro -> HTF structure -> 50/200 EMA -> "
            "location -> liquidity -> sweep -> supply/demand -> premium/discount -> M15 confirmation), "
            "or upload a chart/trade screenshot below for an image-based read. Requires a local Ollama "
            "install (https://ollama.com). Plain chat works with any pulled text model; screenshot "
            "analysis needs a vision-capable model pulled separately (e.g. `ollama pull llava` or "
            "`ollama pull llama3.2-vision`).",
        )

        self._build_ai_assist_section(f, prefix="aiassistant")

        # ------------------------------------------------------------
        # Market Intelligence -- News window + Best Trades window, plus
        # one "GENERATE BASIC OUTLOOK" button that refreshes both and
        # writes a combined macro + best-forex/crypto/futures summary
        # below. Backed by app.ai.market_intelligence (ForexFactory's
        # public calendar feed + the same MT5/Alpaca-driven scanner the
        # web app's dashboard already uses) -- see that module's
        # docstring. Works with Ollama off (deterministic summary from
        # app.ai.trading_assistant.build_deterministic_outlook); if
        # Ollama IS on, the model's narrative read is appended after it.
        # ------------------------------------------------------------
        mi_section = self._section(
            f, "Market Intelligence",
            "News from ForexFactory's calendar feed, and the same forex/crypto/futures 'what "
            "moved best' scanner the web dashboard uses. REFRESH pulls fresh data (needs MT5 "
            "connected on the Forward Test tab, or saved Alpaca keys on the Data tab, for the "
            "Best Trades side -- News works with neither).",
            emphasize=True,
        )
        mi_btn_row = Frame(mi_section, bg=PANEL)
        mi_btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self.aiassistant_outlook_btn = self._button(
            mi_btn_row, "GENERATE BASIC OUTLOOK", self._ai_generate_outlook, primary=True,
        )
        self.aiassistant_outlook_btn.pack(side="left")
        self.aiassistant_refresh_news_btn = self._button(mi_btn_row, "REFRESH NEWS", self._ai_refresh_news)
        self.aiassistant_refresh_news_btn.pack(side="left", padx=(8, 0))
        self.aiassistant_refresh_trades_btn = self._button(mi_btn_row, "REFRESH BEST TRADES", self._ai_refresh_rankings)
        self.aiassistant_refresh_trades_btn.pack(side="left", padx=(8, 0))

        # Analyze a Symbol: pick ANY symbol (not limited to the fixed
        # Best Trades universe or the Trade-of-the-Day MES/MNQ/MGC set)
        # and get the exact same T58 checklist + entry/stop/target -- web
        # equivalent is analyzeSymbol() in ai_assistant.html.
        symbol_picker_row = Frame(mi_section, bg=PANEL)
        symbol_picker_row.pack(anchor="w", fill="x", padx=18, pady=(0, 4))
        self.aiassistant_symbol_picker = LabeledEntry(symbol_picker_row, "Analyze a Symbol", "")
        self.aiassistant_symbol_picker.pack(side="left", padx=(0, 8))
        self.aiassistant_symbol_picker_btn = self._button(
            symbol_picker_row, "ANALYZE", self._ai_analyze_symbol,
        )
        self.aiassistant_symbol_picker_btn.pack(side="left")


        mi_windows_row = Frame(mi_section, bg=PANEL)
        mi_windows_row.pack(fill="both", expand=True, padx=18, pady=(4, 14))

        news_win = Frame(mi_windows_row, bg=PANEL_2, highlightthickness=1, highlightbackground=BORDER)
        news_win.pack(side="left", fill="both", expand=True, padx=(0, 6))
        Label(news_win, text="NEWS", bg=PANEL_2, fg=TEXT_MUTED, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=10, pady=(8, 2)
        )
        news_list_frame = Frame(news_win, bg=PANEL_2)
        news_list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.aiassistant_news_list = Listbox(
            news_list_frame, height=10, bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.aiassistant_news_list.pack(side="left", fill="both", expand=True)
        news_scroll = ttk.Scrollbar(
            news_list_frame, orient="vertical", command=self.aiassistant_news_list.yview, style="T58.Vertical.TScrollbar",
        )
        news_scroll.pack(side="right", fill="y")
        self.aiassistant_news_list.configure(yscrollcommand=news_scroll.set)
        self._bind_isolated_wheel(self.aiassistant_news_list)
        # Not auto-fetched at launch on purpose -- every tab in this app is
        # built eagerly at startup (see the builder loop near _build_ui),
        # and this needs a live network call to ForexFactory's calendar
        # feed. Auto-running it here would add unpredictable network
        # latency to every app launch; REFRESH NEWS / GENERATE BASIC
        # OUTLOOK fetch it on demand instead, same as every other
        # network-backed tool in this app.
        self.aiassistant_news_list.insert(END, "Click REFRESH NEWS or GENERATE BASIC OUTLOOK to load.")

        trades_win = Frame(mi_windows_row, bg=PANEL_2, highlightthickness=1, highlightbackground=BORDER)
        trades_win.pack(side="left", fill="both", expand=True, padx=(6, 0))
        Label(trades_win, text="BEST TRADES (FOREX / CRYPTO / FUTURES)", bg=PANEL_2, fg=TEXT_MUTED, font=_safe_font(8, "bold")).pack(
            anchor="w", padx=10, pady=(8, 2)
        )
        trades_tree_frame = Frame(trades_win, bg=PANEL_2)
        trades_tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        trades_columns = ("symbol", "class", "status", "direction", "move", "score")
        self.aiassistant_trades_tree = ttk.Treeview(
            trades_tree_frame, columns=trades_columns, show="headings", style="T58.Treeview", height=10,
        )
        trades_headings = {
            "symbol": "Symbol", "class": "Class", "status": "Status",
            "direction": "Dir", "move": "Move %", "score": "Score",
        }
        for col, text in trades_headings.items():
            self.aiassistant_trades_tree.heading(col, text=text)
            self.aiassistant_trades_tree.column(col, width=80, anchor="w")
        self.aiassistant_trades_tree.pack(side="left", fill="both", expand=True)
        trades_scroll = ttk.Scrollbar(
            trades_tree_frame, orient="vertical", command=self.aiassistant_trades_tree.yview, style="T58.Vertical.TScrollbar",
        )
        trades_scroll.pack(side="right", fill="y")
        self.aiassistant_trades_tree.configure(yscrollcommand=trades_scroll.set)
        self._bind_isolated_wheel(self.aiassistant_trades_tree)

        self.aiassistant_outlook_status = Label(
            mi_section, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), anchor="w",
        )
        self.aiassistant_outlook_status.pack(anchor="w", padx=18, pady=(0, 10))

        self._ai_last_rankings = None

        # ------------------------------------------------------------
        # AI Director -- portfolio-level view across the WHOLE Strategy
        # Library (every .py/.pine/.mq5/.json strategy saved, at every
        # pipeline stage), backed by app.ai.ai_director. Desktop parity
        # for the web app's AI Director panel (app/web/templates/
        # ai_assistant.html) -- same deterministic-priority-list-first,
        # Ollama-narrates-on-top posture as Basic Outlook above: the
        # ranking itself never depends on Ollama being on.
        # ------------------------------------------------------------
        director_section = self._section(
            f, "AI Director",
            "Surveys every strategy in the Strategy Library (pipeline stage, verdict, how long it's "
            "sat idle) plus today's Best Trades scan, and ranks what's actually worth doing next -- "
            "computed by the app, not guessed by the model. REFRESH PRIORITIES recomputes the table "
            "only; GENERATE DIRECTOR'S BRIEFING also writes a narrative summary to the log below.",
            emphasize=True,
        )
        director_btn_row = Frame(director_section, bg=PANEL)
        director_btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self.aidirector_refresh_btn = self._button(director_btn_row, "REFRESH PRIORITIES", self._ai_refresh_director)
        self.aidirector_refresh_btn.pack(side="left")
        self.aidirector_briefing_btn = self._button(
            director_btn_row, "GENERATE DIRECTOR'S BRIEFING", self._ai_generate_director_briefing, primary=True,
        )
        self.aidirector_briefing_btn.pack(side="left", padx=(8, 0))

        director_tree_frame = Frame(director_section, bg=PANEL)
        director_tree_frame.pack(fill="both", expand=True, padx=18, pady=(4, 4))
        director_columns = ("rank", "strategy", "type", "action", "priority", "reason")
        self.aidirector_tree = ttk.Treeview(
            director_tree_frame, columns=director_columns, show="headings", style="T58.Treeview", height=8,
        )
        director_headings = {
            "rank": "#", "strategy": "Strategy", "type": "Type", "action": "Action",
            "priority": "Priority", "reason": "Why",
        }
        director_widths = {"rank": 30, "strategy": 180, "type": 70, "action": 130, "priority": 60, "reason": 340}
        for col, text in director_headings.items():
            self.aidirector_tree.heading(col, text=text)
            self.aidirector_tree.column(col, width=director_widths.get(col, 100), anchor="w")
        self.aidirector_tree.pack(side="left", fill="both", expand=True)
        director_scroll = ttk.Scrollbar(
            director_tree_frame, orient="vertical", command=self.aidirector_tree.yview, style="T58.Vertical.TScrollbar",
        )
        director_scroll.pack(side="right", fill="y")
        self.aidirector_tree.configure(yscrollcommand=director_scroll.set)
        self._bind_isolated_wheel(self.aidirector_tree)

        self.aidirector_status = Label(
            director_section, text="Click REFRESH PRIORITIES to scan the Strategy Library.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), anchor="w",
        )
        self.aidirector_status.pack(anchor="w", padx=18, pady=(0, 14))

        self._ai_last_directives = None

        vision_section = self._section(
            f, "Vision model (screenshot analysis only)",
            "Separate from the chat model above -- screenshot analysis needs a multimodal model.",
        )
        saved = ollama_settings_module.load_settings()
        self.aiassistant_vision_model = LabeledEntry(
            vision_section, "Vision model (must already be pulled)", getattr(saved, "vision_model", "llava"),
        )

        shot_section = self._section(
            f, "Screenshot Analysis",
            "Upload a chart screenshot for an exact trading plan in Owen's format, or a trade/PnL "
            "screenshot for a Session-Review-style breakdown of what went right or wrong.",
        )
        self.aiassistant_shot_notes = LabeledEntry(
            shot_section, "Optional notes (symbol, timeframe, what happened)", "",
        )
        shot_btn_row = Frame(shot_section, bg=PANEL)
        shot_btn_row.pack(anchor="w", padx=18, pady=(4, 14))
        self.aiassistant_chart_btn = self._button(
            shot_btn_row, "UPLOAD CHART SCREENSHOT -> TRADING PLAN", self._ai_analyze_chart_screenshot, primary=True,
        )
        self.aiassistant_chart_btn.pack(side="left")
        self.aiassistant_trade_btn = self._button(
            shot_btn_row, "UPLOAD TRADE SCREENSHOT -> SESSION REVIEW", self._ai_analyze_trade_screenshot,
        )
        self.aiassistant_trade_btn.pack(side="left", padx=(8, 0))

        chat_section = self._section(f, "Chat", "Ask about a market, request the daily brief, or discuss a trade.")
        mode_row = Frame(chat_section, bg=PANEL)
        mode_row.pack(anchor="w", padx=18, pady=(0, 8))
        Label(mode_row, text="Mode:", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8, "bold")).pack(side="left", padx=(0, 8))
        self.aiassistant_chat_mode = StringVar(value="personal")
        self.aiassistant_mode_personal_btn = self._button(
            mode_row, "PERSONAL", lambda: self._ai_set_chat_mode("personal"), primary=True,
        )
        self.aiassistant_mode_personal_btn.pack(side="left")
        self.aiassistant_mode_t58_btn = self._button(mode_row, "T58 GROUP", lambda: self._ai_set_chat_mode("t58"))
        self.aiassistant_mode_t58_btn.pack(side="left", padx=(6, 0))
        chat_out_frame = Frame(chat_section, bg=PANEL)
        self.aiassistant_output = Text(
            chat_out_frame, height=22, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        chat_out_scroll = ttk.Scrollbar(
            chat_out_frame, orient="vertical", command=self.aiassistant_output.yview, style="T58.Vertical.TScrollbar",
        )
        self.aiassistant_output.configure(yscrollcommand=chat_out_scroll.set)
        self.aiassistant_output.pack(side="left", fill="both", expand=True)
        chat_out_scroll.pack(side="right", fill="y")
        chat_out_frame.pack(fill="both", expand=True, padx=18, pady=(3, 10))
        self._bind_isolated_wheel(self.aiassistant_output)
        self.aiassistant_output.configure(state="disabled")

        self.aiassistant_question = LabeledEntry(chat_section, "Ask T58 AI", "")
        btn_row2 = Frame(chat_section, bg=PANEL)
        btn_row2.pack(anchor="w", padx=18, pady=(0, 16))
        self.aiassistant_send_btn = self._button(btn_row2, "SEND", self._ai_send_chat, primary=True)
        self.aiassistant_send_btn.pack(side="left")
        self._button(btn_row2, "DAILY BRIEF", self._ai_daily_brief).pack(side="left", padx=(8, 0))
        self._button(btn_row2, "WATCHLIST", self._ai_watchlist).pack(side="left", padx=(8, 0))

        self._ai_chat_history: list[dict] = []
        self._ai_set_chat_mode("personal")
        self._ai_warm_up_in_background()

    def _ai_warm_up_in_background(self) -> None:
        """Best-effort, non-blocking: if Ollama's already enabled and
        configured, load the model into memory now (on a daemon thread)
        instead of waiting for the person's first real chat/screenshot/
        outlook request to pay that cost. Silently does nothing if Ollama
        is off, unconfigured, or unreachable -- this is purely a speed
        optimization, never a requirement."""
        saved = ollama_settings_module.load_settings()
        if not saved.is_usable:
            return

        def run():
            try:
                from app.ai.ollama_client import OllamaClient
                OllamaClient(saved).warm_up()
            except Exception:
                pass

        threading.Thread(target=run, daemon=True).start()

    def _ai_append_output(self, label: str, text: str) -> None:
        self.aiassistant_output.configure(state="normal")
        self.aiassistant_output.insert(END, f"[{label}]\n{text}\n\n")
        self.aiassistant_output.see(END)
        self.aiassistant_output.configure(state="disabled")
        self.root.update_idletasks()

    def _ai_client(self) -> "TradingAssistantClient":
        from app.ai.trading_assistant import TradingAssistantClient
        settings = self._build_ollama_settings("aiassistant")
        # vision_model isn't part of _build_ollama_settings' saved fields
        # (that helper mirrors the "AI Assist" section's own save/load,
        # shared by every tab that embeds it) -- overlay it here from this
        # tab's own field before use, and persist it alongside the rest.
        settings.vision_model = self.aiassistant_vision_model.get_str().strip() or settings.vision_model
        ollama_settings_module.save_settings(settings)
        return TradingAssistantClient(settings)

    def _ai_market_context(self, include_rankings: bool = True) -> dict:
        """Best-effort market intelligence (rankings + news) for chat/
        screenshot context -- degrades to an empty context (no rankings,
        no news) rather than failing the whole request if MT5/Alpaca/the
        news feed aren't reachable from the desktop app.

        BUGFIX (Sep 2026): this used to hardcode rankings=[] -- see
        app.ai.market_intelligence's module docstring for why that meant
        Daily Brief/chat never saw "best markets" data on desktop even
        though the identical web app feature already did. `include_rankings
        =False` is used by _ai_send_chat/_ai_daily_brief when the News/Best
        Trades panels on this tab already have fresh data cached, so a
        chat message doesn't also re-scan every symbol synchronously."""
        from app.ai import market_intelligence, trading_assistant as ta_module
        try:
            news_result = market_intelligence.compute_news()
            events = news_result.events if not news_result.error else []
        except Exception:
            news_result = None
            events = []
        rankings = []
        if include_rankings:
            cached = getattr(self, "_ai_last_rankings", None)
            if cached is not None:
                rankings = cached
            else:
                try:
                    rankings, _errors = market_intelligence.compute_rankings(news_result=news_result)
                except Exception:
                    rankings = []
        structure_notes = {}
        if rankings:
            try:
                top_symbols = [r.symbol for r in rankings[:5]]
                structure_notes = market_intelligence.compute_market_structure_notes(top_symbols)
            except Exception:
                structure_notes = {}
        return ta_module.build_context(rankings=rankings, news_events=events, market_structure_by_symbol=structure_notes)

    def _ai_run_async(self, button, work_fn, on_success):
        original_text = button.cget("text")
        button.config(state="disabled", text="WORKING...")

        def _worker():
            try:
                result = work_fn()
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user
                def _err(exc=exc):
                    button.config(state="normal", text=original_text)
                    messagebox.showerror("AI Assistant", str(exc))
                self.root.after(0, _err)
                return

            def _ok():
                button.config(state="normal", text=original_text)
                on_success(result)
            self.root.after(0, _ok)

        threading.Thread(target=_worker, daemon=True).start()

    def _ai_set_chat_mode(self, mode: str) -> None:
        """Switches the AI Assistant chat between "Personal" (Owen's own
        strategy hierarchy + assistant persona) and "T58 Group" (market/
        strategy content only, no personal assistant framing) -- mirrors
        the web app's mode-toggle buttons in ai_assistant.html."""
        self.aiassistant_chat_mode.set(mode)
        personal_active = mode == "personal"
        self.aiassistant_mode_personal_btn.config(bg=ACCENT if personal_active else PANEL_3)
        self.aiassistant_mode_t58_btn.config(bg=ACCENT if not personal_active else PANEL_3)

    def _ai_send_chat(self):
        question = self.aiassistant_question.get_str().strip()
        if not question:
            return
        self.aiassistant_question.var.set("")
        self._ai_append_output("Owen", question)
        mode = self.aiassistant_chat_mode.get()

        def work():
            client = self._ai_client()
            context = self._ai_market_context()
            reply, error = client.ask(question, context, mode=mode, history=self._ai_chat_history)
            if error:
                raise RuntimeError(error)
            return reply

        def done(reply):
            self._ai_append_output("T58 AI", reply)
            self._ai_chat_history.append({"role": "user", "content": question})
            self._ai_chat_history.append({"role": "assistant", "content": reply})

        self._ai_run_async(self.aiassistant_send_btn, work, done)

    def _ai_daily_brief(self):
        self._ai_append_output("Owen", "[Daily Brief]")

        def work():
            client = self._ai_client()
            context = self._ai_market_context()
            reply, error = client.daily_brief(context)
            if error:
                raise RuntimeError(error)
            return reply

        self._ai_run_async(self.aiassistant_send_btn, work, lambda reply: self._ai_append_output("T58 AI", reply))

    def _ai_watchlist(self):
        """Generates a watchlist from the top symbols in the Best Trades
        panel (or a fresh scan if that panel hasn't been refreshed yet
        this session) -- desktop equivalent of the web app's WATCHLIST
        button (see runWatchlist() in ai_assistant.html)."""
        cached = getattr(self, "_ai_last_rankings", None)
        if cached is None:
            try:
                from app.ai import market_intelligence
                cached, _errors = market_intelligence.compute_rankings()
                self._ai_last_rankings = cached
                self._ai_populate_trades_tree(cached)
            except Exception:
                cached = []
        from app.ai.market_scanner import ranking_to_dict
        symbols = [ranking_to_dict(r)["symbol"] for r in (cached or [])[:6]]
        self._ai_append_output("Owen", f"[Watchlist: {', '.join(symbols)}]")

        def work():
            client = self._ai_client()
            context = self._ai_market_context(include_rankings=False)
            context["watchlist_symbols"] = symbols
            reply, error = client.watchlist(context)
            if error:
                raise RuntimeError(error)
            return reply

        self._ai_run_async(self.aiassistant_send_btn, work, lambda reply: self._ai_append_output("T58 AI", reply))

    # -- Market Intelligence: News window + Best Trades window + Basic
    # Outlook button (see the section built in _build_ai_assistant_tab). --

    # ForexFactory folder colors: red = High, orange = Medium, yellow = Low,
    # gray = Holiday/non-economic -- a small colored square stands in for
    # the folder icon in this plain-text Listbox (Tkinter Listbox items are
    # text-only, so per-item foreground color via itemconfig is how the
    # folder-color cue gets applied here, matching the web app's colored
    # folder icon in app/web/templates/ai_assistant.html).
    _NEWS_FOLDER_COLORS = {
        "High": "#ff3b3b",
        "Medium": "#ff9f1c",
        "Low": "#f2d024",
        "Holiday": "#9098a8",
    }

    def _ai_populate_news_list(self, events) -> None:
        self.aiassistant_news_list.delete(0, END)
        if not events:
            self.aiassistant_news_list.insert(END, "No upcoming events (calendar unavailable or nothing scheduled).")
            return
        for i, e in enumerate(events[:60]):
            when = e.when.strftime("%a %H:%M UTC") if getattr(e, "when", None) else "TBD"
            self.aiassistant_news_list.insert(END, f"\u25a0 [{e.impact:<6}] {e.currency:<3} {when}  {e.title}")
            color = self._NEWS_FOLDER_COLORS.get(e.impact)
            if color:
                self.aiassistant_news_list.itemconfig(i, fg=color)

    def _ai_populate_trades_tree(self, rankings) -> None:
        from app.ai import market_scanner
        for row in self.aiassistant_trades_tree.get_children():
            self.aiassistant_trades_tree.delete(row)
        if not rankings:
            return
        for r in rankings[:40]:
            d = market_scanner.ranking_to_dict(r)
            self.aiassistant_trades_tree.insert("", END, values=(
                d["symbol"], d["asset_class"], d["status"], d["direction"],
                f"{d['momentum_pct']:+.2f}", d["score"],
            ))

    def _ai_refresh_news(self):
        def work():
            from app.ai import market_intelligence
            result = market_intelligence.compute_news()
            if result.error:
                raise RuntimeError(result.error)
            return result.events

        self._ai_run_async(self.aiassistant_refresh_news_btn, work, self._ai_populate_news_list)

    def _ai_refresh_rankings(self):
        def work():
            from app.ai import market_intelligence
            rankings, errors = market_intelligence.compute_rankings()
            self._ai_last_rankings = rankings
            if not rankings and errors:
                raise RuntimeError("No symbols scanned: " + "; ".join(errors[:3]))
            return rankings

        self._ai_run_async(self.aiassistant_refresh_trades_btn, work, self._ai_populate_trades_tree)

    def _ai_generate_outlook(self):
        """Refreshes both Market Intelligence windows, then writes a
        combined macro + best-forex/crypto/futures-trades summary to the
        chat log below. Works with Ollama off (falls back to
        build_deterministic_outlook -- every figure it uses was already
        computed by the app) or on (appends the model's narrative read
        after the deterministic facts)."""
        self.aiassistant_outlook_status.config(text="Fetching news + scanning markets...", fg=TEXT_DIM)
        self._ai_append_output("Owen", "[Generate Basic Outlook]")

        def work():
            from app.ai import market_intelligence, trading_assistant as ta_module

            news_result = market_intelligence.compute_news()
            events = news_result.events if not news_result.error else []
            rankings, errors = market_intelligence.compute_rankings(news_result=news_result)
            self._ai_last_rankings = rankings
            try:
                structure_notes = market_intelligence.compute_market_structure_notes([r.symbol for r in rankings[:5]])
            except Exception:
                structure_notes = {}
            context = ta_module.build_context(rankings=rankings, news_events=events, market_structure_by_symbol=structure_notes)

            deterministic = ta_module.build_deterministic_outlook(context)
            settings = self._build_ollama_settings("aiassistant")
            if not settings.is_usable:
                note = "\n\n(Ollama isn't enabled -- showing deterministic data only. Turn on AI Assist above for a narrative read too.)"
                return context, events, rankings, deterministic + note
            client = ta_module.TradingAssistantClient(settings)
            reply, error = client.market_outlook(context)
            if error:
                return context, events, rankings, deterministic + f"\n\n(Ollama narrative unavailable: {error})"
            return context, events, rankings, deterministic + "\n\n--- T58 AI's read ---\n" + reply

        def done(result):
            context, events, rankings, text = result
            self._ai_populate_news_list(events)
            self._ai_populate_trades_tree(rankings)
            self._ai_append_output("T58 AI -- Basic Outlook", text)
            self.aiassistant_outlook_status.config(text="", fg=TEXT_DIM)

        self._ai_run_async(self.aiassistant_outlook_btn, work, done)

    def _ai_analyze_symbol(self):
        """Handler for the ANALYZE button next to the Analyze a Symbol
        entry -- desktop equivalent of api_analyze_symbol() in
        app.web.ai_assistant_routes / analyzeSymbol() in ai_assistant.html.
        Works for any symbol the connected data feed knows, not just the
        fixed Best Trades / Trade-of-the-Day universes."""
        symbol = self.aiassistant_symbol_picker.get_str().strip()
        if not symbol:
            messagebox.showinfo("Analyze a Symbol", "Type a symbol first (e.g. ES1!, XAUUSD, BTCUSD).")
            return
        self._ai_append_output("Owen", f"[Analyze Symbol: {symbol}]")

        def work():
            from app.ai import market_intelligence, market_scanner, news_forexfactory
            from app.ai import t58_strategy_engine as t58
            from app.ai import trading_assistant as ta_module

            h1 = market_intelligence.bar_fetcher(symbol, 60, 300)
            m15 = market_intelligence.bar_fetcher(symbol, 15, 200)
            if h1 is None or h1.empty or len(h1) < 60:
                raise RuntimeError(
                    f"No usable H1 data for '{symbol}' (need at least 60 bars). Check the symbol name "
                    f"matches your data feed/MT5 exactly, and that MT5/Alpaca is connected."
                )
            news_result = market_intelligence.compute_news()
            macro_bias = market_intelligence.daily_trend_bias(symbol)
            news_risk = news_forexfactory.news_risk_for_symbol(news_result, symbol)
            snapshot = t58.build_market_snapshot(
                symbol=symbol, h1_frame=h1, m15_frame=m15, macro_bias=macro_bias, news_risk=news_risk,
            )
            assessment = t58.assess(snapshot)
            try:
                currency_bias = news_forexfactory.recent_data_surprise_bias_by_currency(news_result)
                fundamental_bias = market_intelligence.fundamental_bias_for_symbol(symbol, currency_bias)
            except Exception:
                fundamental_bias = "neutral"
            ranking_dict = {
                "symbol": symbol, "asset_class": "custom", "score": assessment.score,
                "status": assessment.status, "direction": assessment.direction,
                "momentum_pct": round(market_scanner._momentum_pct(h1), 3),
                "atr_normalized_move": round(market_scanner._atr_normalized_move(h1), 2),
                "zone": snapshot.location.zone, "ema_alignment": snapshot.ema.alignment,
                "target": assessment.target, "missing": assessment.missing,
                "news_risk": snapshot.news_risk, "fundamental_bias": fundamental_bias,
                "entry_price": assessment.entry_price, "stop_price": assessment.stop_price,
                "target_price": assessment.target_price,
            }
            deterministic = ta_module.build_deterministic_symbol_report(ranking_dict)
            settings = self._build_ollama_settings("aiassistant")
            if not settings.is_usable:
                return deterministic + "\n\n(Ollama isn't enabled -- showing deterministic data only.)"
            rankings, errors = market_intelligence.compute_rankings(news_result=news_result)
            context = ta_module.build_context(
                rankings=rankings, news_events=news_result.events, symbol_assessment=ranking_dict,
            )
            client = ta_module.TradingAssistantClient(settings)
            reply, error = client.analyze_symbol(context, symbol)
            if error:
                return deterministic + f"\n\n(Ollama narrative unavailable: {error})"
            return deterministic + "\n\n--- T58 AI's read ---\n" + reply

        self._ai_run_async(
            self.aiassistant_symbol_picker_btn, work,
            lambda text: self._ai_append_output(f"T58 AI -- {symbol} Analysis", text),
        )

    # -- AI Director: portfolio-level priority list across the whole
    # Strategy Library (see app.ai.ai_director, and the section built in
    # _build_ai_assistant_tab above). --

    def _ai_compute_director_directives(self):
        """Shared by REFRESH PRIORITIES and GENERATE DIRECTOR'S BRIEFING.
        Reuses the Best Trades panel's cached rankings if already scanned
        this session (same convention as _ai_watchlist) rather than
        forcing a second network scan; falls back to a fresh scan if
        nothing's cached yet. Never raises: a library or market-data
        failure just means an empty/partial directive list, same
        fail-soft posture as every other AI Assistant tab action."""
        from app.ai import ai_director, experiment_memory, market_scanner
        from app.strategy import library

        try:
            strategies = library.list_saved_strategies()
        except Exception:
            strategies = []

        rankings = getattr(self, "_ai_last_rankings", None)
        if rankings is None:
            try:
                from app.ai import market_intelligence
                rankings, _errors = market_intelligence.compute_rankings()
                self._ai_last_rankings = rankings
                self._ai_populate_trades_tree(rankings)
            except Exception:
                rankings = []
        ranking_dicts = [market_scanner.ranking_to_dict(r) for r in (rankings or [])]

        try:
            memory_counts = experiment_memory.get_summary_counts()
        except Exception:
            memory_counts = {"total": 0, "by_verdict": {}, "top_strategies": {}}

        directives = ai_director.compute_directives(strategies, rankings=ranking_dicts)
        self._ai_last_directives = directives
        return directives, memory_counts

    def _ai_populate_director_tree(self, directives) -> None:
        for row in self.aidirector_tree.get_children():
            self.aidirector_tree.delete(row)
        if not directives:
            self.aidirector_status.config(
                text="Nothing in the Strategy Library needs action right now -- start a new idea via "
                     "Strategy Generator, Search Lab, or the Research Loop.",
                fg=TEXT_DIM,
            )
            return
        for i, d in enumerate(directives, start=1):
            self.aidirector_tree.insert("", END, values=(
                i, d.strategy_name, d.strategy_type, d.action, f"{d.priority:.0f}", d.reason,
            ))
        self.aidirector_status.config(text=f"{len(directives)} item(s) ranked by priority.", fg=TEXT_DIM)

    def _ai_refresh_director(self):
        def work():
            directives, _memory_counts = self._ai_compute_director_directives()
            return directives

        self._ai_run_async(self.aidirector_refresh_btn, work, self._ai_populate_director_tree)

    def _ai_generate_director_briefing(self):
        """Refreshes the priority table, then writes a combined
        deterministic-list + (if Ollama's on) narrative briefing to the
        chat log below -- same "deterministic always, narrative on top"
        pattern as _ai_generate_outlook."""
        self.aidirector_status.config(text="Surveying the Strategy Library and today's markets...", fg=TEXT_DIM)
        self._ai_append_output("Owen", "[Generate Director's Briefing]")

        def work():
            from app.ai import ai_director, trading_assistant as ta_module

            directives, memory_counts = self._ai_compute_director_directives()
            deterministic = ai_director.build_deterministic_briefing(directives, memory_counts)
            settings = self._build_ollama_settings("aiassistant")
            if not settings.is_usable:
                note = "\n\n(Ollama isn't enabled -- showing the deterministic priority list only. Turn on AI Assist above for a narrative briefing too.)"
                return directives, deterministic + note
            client = ta_module.TradingAssistantClient(settings)
            user_message = ai_director.build_director_prompt(directives, memory_counts)
            reply, error = client.director_briefing(user_message)
            if error:
                return directives, deterministic + f"\n\n(Ollama narrative unavailable: {error})"
            return directives, deterministic + "\n\n--- T58 AI Director's briefing ---\n" + reply

        def done(result):
            directives, text = result
            self._ai_populate_director_tree(directives)
            self._ai_append_output("T58 AI Director", text)

        self._ai_run_async(self.aidirector_briefing_btn, work, done)

    def _ai_analyze_chart_screenshot(self):
        path = filedialog.askopenfilename(
            title="Select a chart screenshot",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        notes = self.aiassistant_shot_notes.get_str().strip()
        self._ai_append_output("Owen", f"[Chart screenshot uploaded: {Path(path).name}]" + (f" notes: {notes}" if notes else ""))

        def work():
            import base64
            client = self._ai_client()
            context = self._ai_market_context()
            image_b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            reply, error = client.analyze_chart_screenshot(image_b64, context=context, extra_notes=notes)
            if error:
                raise RuntimeError(error)
            return reply

        self._ai_run_async(self.aiassistant_chart_btn, work, lambda reply: self._ai_append_output("T58 AI -- Trading Plan", reply))

    def _ai_analyze_trade_screenshot(self):
        path = filedialog.askopenfilename(
            title="Select a trade/PnL screenshot",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        notes = self.aiassistant_shot_notes.get_str().strip()
        self._ai_append_output("Owen", f"[Trade screenshot uploaded: {Path(path).name}]" + (f" notes: {notes}" if notes else ""))

        def work():
            import base64
            client = self._ai_client()
            image_b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            reply, error = client.analyze_trade_screenshot(image_b64, extra_notes=notes)
            if error:
                raise RuntimeError(error)
            return reply

        self._ai_run_async(self.aiassistant_trade_btn, work, lambda reply: self._ai_append_output("T58 AI -- Session Review", reply))

    # ------------------------------------------------------------------
    # Options Outlook tab -- deterministic call/put candidates (Black-
    # Scholes, app.ai.options_outlook) ranked and explained by Ollama
    # (app.ai.trading_assistant.options_outlook). Every strike/premium/
    # delta shown is computed by the app; the model only ranks/explains.
    # ------------------------------------------------------------------

    def _build_options_outlook_tab(self):
        f = self._scrollable(self.tab_options_outlook)
        self._page_header(
            f, "OPTIONS", "Options Outlook",
            "Deterministic call/put candidates around target deltas (~50/25/10), priced with the "
            "same from-scratch Black-Scholes engine as the Options Pricing Calculator, then ranked "
            "and explained by T58 AI for the horizon you pick. Every strike, premium, and delta is "
            "computed by the app -- the model never invents a number.",
        )

        sec = self._section(
            f, "Scenario",
            "Enter the underlying's spot price and an implied/estimated volatility (annualized, "
            "decimal -- e.g. 0.25 for 25%). Use the Options Pricing Calculator's implied-vol solver "
            "on the Quant Lab tab if you have a market option price to back it out from.",
        )
        self.oo_symbol = LabeledEntry(sec, "Symbol", "SPY")
        self.oo_spot = LabeledEntry(sec, "Spot price", 100)
        self.oo_iv = LabeledEntry(sec, "Implied/estimated volatility (decimal)", 0.25)
        self.oo_rate = LabeledEntry(sec, "Risk-free rate (decimal)", 0.045)
        self.oo_horizon = LabeledCombo(sec, "Horizon", ["today", "this_week"], default="today")
        self.oo_notes = LabeledEntry(sec, "Directional/context notes (optional)", "")
        btn_row = Frame(sec, bg=PANEL); btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self.oo_btn = self._button(btn_row, "GENERATE OPTIONS OUTLOOK", self._options_outlook_clicked, primary=True)
        self.oo_btn.pack(side="left")

    def _options_outlook_clicked(self):
        symbol = self.oo_symbol.get_str().strip() or "SYMBOL"
        spot = self.oo_spot.get_float()
        iv = self.oo_iv.get_float()
        rate = self.oo_rate.get_float()
        horizon = self.oo_horizon.get_str()
        notes = self.oo_notes.get_str().strip()

        def work():
            from app.ai import options_outlook as options_outlook_module
            from app.ai.trading_assistant import TradingAssistantClient

            dte_days = options_outlook_module.HORIZON_PRESETS.get(horizon, 1)
            candidates = options_outlook_module.build_candidates(spot=spot, iv=iv, dte_days=dte_days, r=rate)
            candidate_dicts = options_outlook_module.candidates_to_dicts(candidates)

            client = TradingAssistantClient(ollama_settings_module.load_settings())
            reply, error = client.options_outlook(symbol, horizon, candidate_dicts, notes=notes)
            if error:
                raise RuntimeError(error)

            lines = [f"OPTIONS OUTLOOK -- {symbol} -- {horizon}", "", reply, "", "--- Raw candidate data ---"]
            for c in candidate_dicts:
                lines.append(
                    f"{c['option_type'].upper():4} K={c['strike']:<10} delta={c['delta']:<8} "
                    f"premium={c['premium']:<8} breakeven={c['breakeven']:<10} "
                    f"theta/day={c['theta_per_day']:<8} dte={c['dte_days']}"
                )
            return "\n".join(lines)

        self._quant_lab_run_async(self.oo_btn, work, f"Options Outlook -- {symbol}")
