"""Live Market tab."""
from __future__ import annotations

import app.forward_test.mt5_settings as mt5_settings_module
import re
import threading
import urllib.parse
import webbrowser
from app.data import alpaca_credentials
from app.forward_test import mt5_connector as mt5_connector_module
from app.web import live_market
from tkinter import Frame, Label
from app.ui.common import _safe_font
from app.ui.widgets.basic import LabeledCombo, LabeledEntry
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
TEXT_MUTED = ""
TEXT_DIM = ""
GREEN = ""
RED = ""
AMBER = ""
CURRENT_THEME = "dark"
theme.bind(globals())


class LiveMarketMixin:
    """Live Market tab."""


    # -----------------------------------------------------------------------
    # Tab 17 — Live Market
    # -----------------------------------------------------------------------

    def _build_live_market_tab(self):
        f = self._scrollable(self.tab_livemarket)
        self._page_header(
            f,
            "MONITOR / Live Market",
            "Live Market",
            "Two ways to see a live-style chart for a symbol. OPEN TRADINGVIEW CHART (recommended) "
            "just opens the real tradingview.com chart for the symbol below in your browser -- no "
            "local server, no MT5/Alpaca connection needed, works the same on Windows/Mac/Linux. "
            "OPEN LIVE CHART is this app's own built-in chart (candlesticks, volume, EMA 20/50, "
            "VWAP, trade markers from your Live Demo Test sessions) fed from MT5/Alpaca/a local "
            "CSV replay -- it starts a small local server on your own machine and opens the chart "
            "in its own window (or your default browser if a native chart window isn't available).",
        )

        try:
            saved_mt5 = mt5_settings_module.load_settings()
        except Exception:
            saved_mt5 = None

        tv_section = self._section(
            f, "TradingView chart (recommended)",
            "Opens the real TradingView website for this symbol in your default browser. "
            "Simplest and most reliable option -- no local server, no broker connection required, "
            "and it always shows genuinely live data straight from TradingView.",
            emphasize=True,
        )
        self.lm_tv_symbol = LabeledEntry(
            tv_section, "Symbol", saved_mt5.symbol if (saved_mt5 is not None and saved_mt5.is_usable) else "XAUUSD",
        )
        tv_btn_row = Frame(tv_section, bg=PANEL)
        tv_btn_row.pack(anchor="w", padx=18, pady=(0, 4))
        self._button(tv_btn_row, "OPEN TRADINGVIEW CHART", self._lm_open_tradingview, primary=True).pack(side="left")
        Label(
            tv_section, text="Common FX/metals/index/crypto tickers (XAUUSD, EURUSD, US30, "
            "BTCUSD, ...) are mapped to a sensible TradingView exchange automatically -- for "
            "anything else, type whatever symbol/ticker TradingView itself would recognize.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        section = self._section(
            f, "Built-in chart data source (advanced)",
            "Live (MT5) is used automatically when your Live Demo Test MT5 account is "
            "configured; Alpaca is used if you've saved API keys on the Data tab; with neither, "
            "the chart replays an already-imported local CSV bar by bar so there's always "
            "something real on screen. Requires this app's own local server; if OPEN LIVE CHART "
            "ever errors or the native window doesn't appear (most common on Linux without the "
            "optional GTK/QT webview packages installed), use OPEN TRADINGVIEW CHART above instead.",
        )

        # Pick a sane default source instead of always defaulting to "mt5"
        # -- this mirrors the exact fallback the Flask page itself uses
        # (see live_market_page() in app/web/server.py), which used to
        # only ever kick in when NO source was passed in the URL at all.
        # Since OPEN LIVE CHART always sent an explicit source= param, an
        # un-configured MT5 (the common case -- MT5 needs Windows + a
        # running terminal) meant every click silently requested MT5 data,
        # which fails with an empty bar list and a permanently blank chart
        # (status badge stuck on UNAVAILABLE) -- with no error, indistin-
        # guishable from the chart "just not opening". (saved_mt5 was
        # already loaded above, for the TradingView symbol field's default.)
        mt5_usable = bool(
            saved_mt5 is not None and saved_mt5.is_usable and mt5_connector_module.is_available()
        )
        try:
            has_alpaca = bool(alpaca_credentials.load_credentials())
        except Exception:
            has_alpaca = False
        default_source = "mt5" if mt5_usable else ("alpaca" if has_alpaca else "replay")

        self.lm_source = LabeledCombo(section, "Source", ["mt5", "alpaca", "replay"], default_source)
        self.lm_source.combo.bind("<<ComboboxSelected>>", lambda _e: self._lm_sync_source_fields())
        self.lm_symbol = LabeledEntry(section, "Symbol (MT5 / Alpaca)", "XAUUSD")
        if saved_mt5 is not None and saved_mt5.is_usable:
            self.lm_symbol.var.set(saved_mt5.symbol)
        self.lm_timeframe = LabeledCombo(
            section, "Timeframe (minutes)", [str(m) for m in (1, 5, 15, 30, 60, 240, 1440)], "15",
        )
        self.lm_dataset = LabeledCombo(section, "Dataset (Replay only)", [""], "")

        status_lines = []
        if mt5_connector_module.is_available():
            status_lines.append("MT5 package: available.")
        else:
            status_lines.append("MT5 package: not available on this platform (Windows + MT5 terminal required).")
        status_lines.append(f"Alpaca keys saved: {'yes' if has_alpaca else 'no (set them on the Market Data tab)'}.")
        status_lines.append(f"Source defaulted to '{default_source}' based on what's actually configured above -- "
                             "switch it manually any time.")
        Label(
            section, text="  •  ".join(status_lines), bg=PANEL, fg=TEXT_MUTED,
            font=_safe_font(8), wraplength=820, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        btn_row = Frame(section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(0, 14))
        self._button(btn_row, "REFRESH DATASETS", self._lm_refresh_datasets).pack(side="left")
        self.lm_open_btn = self._button(btn_row, "OPEN LIVE CHART", self._lm_open_chart, primary=True)
        self.lm_open_btn.pack(side="left", padx=8)

        self.lm_status = Label(
            f, text="Not started yet -- click OPEN LIVE CHART.", bg=BG, fg=TEXT_MUTED,
            font=_safe_font(9), wraplength=900, justify="left",
        )
        self.lm_status.pack(anchor="w", padx=26, pady=(4, 20))

        self._lm_refresh_datasets()
        self._lm_sync_source_fields()

    def _lm_sync_source_fields(self):
        is_replay = self.lm_source.get_str() == "replay"
        state = "disabled" if is_replay else "normal"
        self.lm_symbol.entry.config(state=state)

    def _lm_refresh_datasets(self):
        try:
            names = live_market.list_replay_datasets()
        except Exception:
            names = []
        self.lm_dataset.combo.config(values=names or [""])
        if names:
            self.lm_dataset.var.set(names[0])

    def _lm_ensure_server(self) -> int:
        if getattr(self, "_lm_port", None):
            return self._lm_port
        import socket

        from werkzeug.serving import make_server

        from app.web.server import app as flask_app

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server = make_server("127.0.0.1", port, flask_app, threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self._lm_port = port
        self._lm_server = server
        return port

    def _lm_open_in_native_window(self, url: str) -> bool:
        """Best-effort: opens the chart in a chromeless native window (via
        the optional `pywebview` package) so it feels like part of the app
        rather than a separate browser tab. Tkinter itself has no way to
        run real JavaScript/Canvas content -- what Lightweight Charts
        needs -- so this is the closest thing to "embedded" achievable
        without a much heavier dependency (bundling a full Chromium
        build). Returns False on any failure (not installed, no supported
        native webview backend on this machine, etc.) so the caller can
        fall back to the system's default browser -- still fully
        functional either way.

        On Linux, pywebview needs an actual GTK or QT webview backend
        (e.g. `pip install pywebview[gtk]` plus the system
        `python3-gi`/`gir1.2-webkit2-*` packages) -- when that's missing,
        `webview.start()` can raise from inside the background thread
        started below rather than at import time, which previously left
        the person staring at "Opened in a live chart window" with no
        window and no explanation. That failure is now caught and folded
        into a browser fallback instead."""
        try:
            import webview
        except ImportError:
            return False
        try:
            existing = getattr(self, "_lm_webview_window", None)
            if existing is not None:
                existing.load_url(url)
                return True

            def run():
                try:
                    window = webview.create_window(
                        "Live Market — T58", url, width=1300, height=880, min_size=(900, 600),
                    )
                    self._lm_webview_window = window
                    webview.start()
                except Exception as exc:
                    # The native webview backend genuinely isn't usable on
                    # this machine (common on Linux without the GTK/QT
                    # webview system packages installed) -- fall back to
                    # the default browser instead of leaving the status
                    # line claiming a window opened that never appeared.
                    def _fallback(exc=exc):
                        opened = webbrowser.open(url)
                        self.lm_status.config(
                            text=(f"Native chart window isn't available on this machine ({exc}) -- "
                                  f"opened in your browser instead." if opened else
                                  f"Native chart window isn't available on this machine ({exc}), and "
                                  f"no browser could be auto-launched either -- open this URL "
                                  f"manually: {url}"),
                            fg=AMBER if opened else RED,
                        )
                    try:
                        self.root.after(0, _fallback)
                    except Exception:
                        pass
                finally:
                    self._lm_webview_window = None  # the window was closed (or never opened)

            threading.Thread(target=run, daemon=True).start()
            return True
        except Exception:
            return False

    # Rough symbol -> TradingView-ticker mapping for the most common cases
    # this app already deals with (FX majors/gold/silver traded through a
    # prop-firm-style broker, plus indices) -- TradingView needs an
    # EXCHANGE:TICKER pair for its best match, and guessing a reasonable
    # one here means OPEN TRADINGVIEW CHART works immediately for the
    # common case without the person needing to know TradingView's symbol
    # syntax. Anything not recognized is passed through as-is (TradingView
    # itself still does a best-effort search on a bare ticker), and the
    # symbol field can always just be hand-edited to whatever TradingView
    # expects.
    _TRADINGVIEW_FX_METALS = {
        "XAUUSD": "OANDA:XAUUSD", "XAGUSD": "OANDA:XAGUSD",
        "EURUSD": "OANDA:EURUSD", "GBPUSD": "OANDA:GBPUSD", "USDJPY": "OANDA:USDJPY",
        "USDCHF": "OANDA:USDCHF", "USDCAD": "OANDA:USDCAD", "AUDUSD": "OANDA:AUDUSD",
        "NZDUSD": "OANDA:NZDUSD", "EURJPY": "OANDA:EURJPY", "GBPJPY": "OANDA:GBPJPY",
        "EURGBP": "OANDA:EURGBP",
        "US30": "OANDA:US30USD", "NAS100": "OANDA:NAS100USD", "SPX500": "OANDA:SPX500USD",
        "USOIL": "OANDA:WTICOUSD", "BTCUSD": "COINBASE:BTCUSD", "ETHUSD": "COINBASE:ETHUSD",
    }

    @classmethod
    def _tradingview_symbol_for(cls, symbol: str) -> str:
        s = re.sub(r"[^A-Za-z0-9]", "", (symbol or "")).upper()
        return cls._TRADINGVIEW_FX_METALS.get(s, s or "OANDA:XAUUSD")

    def _lm_open_tradingview(self):
        """Opens the real tradingview.com chart for the current symbol in
        the default browser -- no local Flask server, no MT5/Alpaca
        connection, no pywebview dependency, so this always works
        (including on Linux, and even with none of the optional live-data
        sources configured) as long as there's internet access and a
        browser installed. webbrowser.open() uses xdg-open/$BROWSER under
        the hood on Linux, exactly like _open_strategy_library_folder's
        folder-opening already relies on for that platform."""
        raw_symbol = self.lm_tv_symbol.get_str().strip() or "XAUUSD"
        tv_symbol = self._tradingview_symbol_for(raw_symbol)
        url = f"https://www.tradingview.com/chart/?symbol={urllib.parse.quote(tv_symbol)}"
        opened = webbrowser.open(url)
        if opened:
            self.lm_status.config(text=f"Opened TradingView chart for {tv_symbol} in your browser.", fg=GREEN)
        else:
            self.lm_status.config(
                text=f"Couldn't auto-launch a browser -- open this URL manually: {url}", fg=AMBER,
            )

    def _lm_open_chart(self):
        self.lm_status.config(text="Starting local server...", fg=AMBER)
        self.root.update_idletasks()
        try:
            port = self._lm_ensure_server()
        except Exception as exc:
            self.lm_status.config(text=f"Couldn't start the local server: {exc}", fg=RED)
            return

        source = self.lm_source.get_str()
        symbol = self.lm_dataset.get_str() if source == "replay" else self.lm_symbol.get_str().strip()
        if not symbol:
            self.lm_status.config(text="Enter a symbol (or pick a dataset for Replay) first.", fg=AMBER)
            return
        params = urllib.parse.urlencode({
            "source": source, "symbol": symbol, "timeframe": self.lm_timeframe.get_str(),
            "theme": CURRENT_THEME,
        })
        url = f"http://127.0.0.1:{port}/live-market?{params}"
        if self._lm_open_in_native_window(url):
            self.lm_status.config(text="Opened in a live chart window.", fg=GREEN)
        else:
            opened = webbrowser.open(url)
            if opened:
                self.lm_status.config(text=f"Opened in your browser: {url}", fg=GREEN)
            else:
                # webbrowser.open() returning False means it couldn't find/launch
                # anything -- previously this branch showed "Opened in your
                # browser" regardless, which looked exactly like "nothing
                # happened" with no indication of why. Surface the URL instead
                # so it can be opened by hand.
                self.lm_status.config(
                    text=f"Couldn't auto-launch a browser -- open this URL manually: {url}", fg=AMBER,
                )
