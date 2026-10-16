"""Data tab (Alpaca / London Strategic Edge fetch, CSV import) and Data Center."""
from __future__ import annotations

import os
import threading
from app.accounts.api_keys import load_settings as load_apikeys_settings, save_settings as save_apikeys_settings
from app.data import alpaca_credentials
from app.data.alpaca_source import (
    ADJUSTMENT_CHOICES,
    ASSET_CLASSES,
    AlpacaFetchError,
    AlpacaImportError,
    FEED_CHOICES,
    TIMEFRAME_LABELS,
    fetch_bars,
    save_bars_as_csv,
    test_connection,
)
from app.data.importer import import_csv
from app.data.london_strategic_edge_source import (
    LSEFetchError,
    LSEImportError,
    TIMEFRAME_CHOICES as LSE_TIMEFRAME_CHOICES,
    fetch_candles as lse_fetch_candles,
    save_bars_as_csv as lse_save_bars_as_csv,
    test_connection as lse_test_connection,
)
from app.data.multi_timeframe import merge_multi_timeframe
from app.data.storage import list_datasets_by_instrument, list_stored_datasets, store_csv_path
from pathlib import Path
from tkinter import END, EXTENDED, Frame, Label, Listbox, filedialog, messagebox, ttk

# Optional OS drag-and-drop for market-data folders: tkinterdnd2 wires Tk
# up as a native file-drop target, but it is NOT a hard dependency --
# when it isn't installed the DnD setup (_maybe_enable_folder_dnd) is a
# silent no-op and the IMPORT FOLDER buttons remain the way in.
try:
    import tkinterdnd2
except ImportError:  # pragma: no cover - optional dependency
    tkinterdnd2 = None
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
PANEL_2 = ""
PANEL_3 = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
METAL_BRIGHT = ""
GREEN = ""
RED = ""
AMBER = ""
theme.bind(globals())


class DataTabMixin:
    """Data tab (Alpaca / London Strategic Edge fetch, CSV import) and Data Center."""


    # -----------------------------------------------------------------------
    # Tab 1 — Market Data
    # -----------------------------------------------------------------------

    def _build_data_tab(self):
        f = self._scrollable(self.tab_data)

        self._page_header(
            f,
            "TEST / Market Data",
            "Market Data",
            "Select historical OHLC data for the backtest. Built-in datasets are "
            "loaded automatically; imported CSVs are stored locally for future sessions.",
        )

        section = self._section(
            f,
            "Available datasets",
            "DATA/RAW • Ctrl/Cmd-click or Shift-click to select more than one timeframe "
            "for multi-timeframe analysis (e.g. 60m for bias, 15m for zone, 5m for entry). "
            "The finest timeframe selected becomes the base/entry timeframe; the others are "
            "merged in as 'tfNN_open/high/low/close/volume' context columns.",
            emphasize=True,
        )

        list_frame = Frame(section, bg=PANEL)
        list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 12))

        self.dataset_listbox = Listbox(
            list_frame,
            height=9,
            selectmode=EXTENDED,
            exportselection=False,
            bg=PANEL_3,
            fg=TEXT,
            selectbackground=BORDER_LIGHT,
            selectforeground=METAL_BRIGHT,
            activestyle="none",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            font=(MONO, 9),
        )
        self.dataset_listbox.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.dataset_listbox.yview,
            style="T58.Vertical.TScrollbar",
        )
        scrollbar.pack(side="right", fill="y")
        self.dataset_listbox.config(yscrollcommand=scrollbar.set)
        self._bind_isolated_wheel(self.dataset_listbox)
        self.dataset_listbox.bind("<<ListboxSelect>>", self._on_dataset_selected)

        btn_row = Frame(section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(0, 14))

        self._button(
            btn_row, "IMPORT CSV(S)", self._browse_csv, primary=True
        ).pack(side="left")

        self._button(
            btn_row, "IMPORT FOLDER...", lambda: self._import_folder_clicked("data")
        ).pack(side="left", padx=8)

        self._button(
            btn_row, "REFRESH LIST", self._refresh_dataset_list
        ).pack(side="left", padx=8)

        self.data_status = Label(
            f,
            text="●  No dataset selected.",
            bg=BG,
            fg=TEXT_MUTED,
            font=_safe_font(9),
        )
        self.data_status.pack(anchor="w", padx=26, pady=(8, 2))

        Label(
            f,
            text="Tip: you can also place CSV files directly in data/raw/ and press REFRESH LIST.",
            bg=BG,
            fg=TEXT_DIM,
            font=_safe_font(8),
        ).pack(anchor="w", padx=26)

        self._build_alpaca_section(f)
        self._build_lse_section(f)

        self._maybe_enable_folder_dnd()

        self._refresh_dataset_list()

    # -----------------------------------------------------------------------
    # Tab 1 — Market Data — Alpaca API fetch
    # -----------------------------------------------------------------------

    def _build_alpaca_section(self, parent):
        """An alternative to picking local files above: fetch bars directly
        from Alpaca using saved (or freshly entered) API keys. Fetched data
        is written into data/raw/<SYMBOL>/ as a normal CSV, so it shows up
        in the 'Available datasets' list above and can be picked -- alone
        or Ctrl/Cmd-clicked together with other files -- exactly like any
        manually imported CSV. This mirrors how the falsification-kit
        scripts (fetch_5m.py / fetch_cache.py) pull bars via alpaca-py,
        except keys are entered once here and saved for reuse instead of
        being read from ALPACA_API_KEY/ALPACA_SECRET_KEY environment
        variables every run."""
        section = self._section(
            parent,
            "Fetch data from Alpaca",
            "Alternative to local files above — pulls bars via the Alpaca API "
            "(stocks + crypto only; no forex/futures/CFD feed) and saves them into "
            "data/raw/ so they join the dataset list above.",
        )

        saved = alpaca_credentials.load_credentials()
        prefill_key = saved.api_key if saved else ""
        prefill_secret = saved.secret_key if saved else ""

        self.alp_api_key = LabeledEntry(section, "Alpaca API key", prefill_key, secret=True, width=32)
        self.alp_secret_key = LabeledEntry(section, "Alpaca secret key", prefill_secret, secret=True, width=32)
        self.alp_save_keys = LabeledCheckbox(
            # Defaults to checked even on a first-ever entry (not just
            # bool(saved)) -- an unchecked-by-default box meant the very
            # first time someone typed their key here, it silently wasn't
            # saved, so every later fetch AND every Quant Lab tool that
            # falls back to the saved key (see _ql_resolve_alpaca_keys)
            # prompted for it all over again.
            section, "Save these keys on this computer for next time", default=True
        )

        self.alp_asset_class = LabeledCombo(section, "Asset class", ASSET_CLASSES, default=ASSET_CLASSES[0])
        self.alp_asset_class.combo.bind("<<ComboboxSelected>>", lambda _e: self._on_alpaca_asset_class_changed())
        self.alp_symbols = LabeledEntry(
            section, "Symbol(s), comma-separated", "AAPL", width=32
        )
        self.alp_timeframe = LabeledCombo(
            section, "Timeframe", TIMEFRAME_LABELS, default="1Day"
        )
        self.alp_start = LabeledEntry(section, "Start date (YYYY-MM-DD)", "2024-01-01")
        self.alp_end = LabeledEntry(section, "End date (YYYY-MM-DD)", "2026-01-01")
        self.alp_feed = LabeledCombo(section, "Feed (stocks only)", FEED_CHOICES, default="iex")
        self.alp_adjustment = LabeledCombo(
            section, "Adjustment (stocks only)", ADJUSTMENT_CHOICES, default="raw"
        )

        btn_row = Frame(section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(4, 4))

        self.alp_test_btn = self._button(btn_row, "TEST CONNECTION", self._test_alpaca_connection)
        self.alp_test_btn.pack(side="left")

        self.alp_fetch_btn = self._button(btn_row, "FETCH & SAVE", self._fetch_alpaca_clicked, primary=True)
        self.alp_fetch_btn.pack(side="left", padx=8)

        self.alp_forget_btn = self._button(btn_row, "FORGET SAVED KEYS", self._forget_alpaca_keys)
        self.alp_forget_btn.pack(side="left")

        self.alpaca_status = Label(
            section,
            text="●  A free Alpaca paper-trading account provides API keys for data access.",
            bg=PANEL,
            fg=TEXT_MUTED,
            font=_safe_font(9),
            wraplength=760,
            justify="left",
        )
        self.alpaca_status.pack(anchor="w", padx=18, pady=(2, 14))

    def _on_alpaca_asset_class_changed(self):
        # Feed/adjustment only apply to stock bars on Alpaca; crypto has
        # neither concept, so grey them out rather than let them silently
        # do nothing when Crypto is selected.
        is_stock = self.alp_asset_class.get_str() == "Stock"
        state = "readonly" if is_stock else "disabled"
        self.alp_feed.combo.config(state=state)
        self.alp_adjustment.combo.config(state=state)

    def _set_alpaca_status(self, text, color=None):
        self.alpaca_status.config(text=f"●  {text}", fg=color or TEXT_MUTED)
        self.root.update_idletasks()

    def _set_alpaca_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.alp_test_btn.config(state=state)
        self.alp_fetch_btn.config(state=state)
        self.alp_forget_btn.config(state=state)

    def _maybe_save_alpaca_keys(self, api_key: str, secret_key: str):
        if self.alp_save_keys.get():
            alpaca_credentials.save_credentials(api_key, secret_key)

    def _forget_alpaca_keys(self):
        alpaca_credentials.clear_credentials()
        self.alp_api_key.var.set("")
        self.alp_secret_key.var.set("")
        self.alp_save_keys.var.set(False)
        self._set_alpaca_status("Saved keys removed from this computer.", GREEN)

    def _test_alpaca_connection(self):
        api_key = self.alp_api_key.get_str().strip()
        secret_key = self.alp_secret_key.get_str().strip()
        if not api_key or not secret_key:
            messagebox.showwarning("Missing keys", "Enter both an API key and a secret key first.")
            return

        self._set_alpaca_buttons_enabled(False)
        self._set_alpaca_status("Testing connection...", AMBER)

        def worker():
            try:
                message = test_connection(api_key, secret_key)
                self._maybe_save_alpaca_keys(api_key, secret_key)
                self._set_alpaca_status(message, GREEN)
            except (AlpacaImportError, AlpacaFetchError) as exc:
                self._set_alpaca_status(str(exc), RED)
            except Exception as exc:  # pragma: no cover - defensive
                self._set_alpaca_status(f"Unexpected error: {exc}", RED)
            finally:
                self._set_alpaca_buttons_enabled(True)

        threading.Thread(target=worker, daemon=True).start()

    def _fetch_alpaca_clicked(self):
        api_key = self.alp_api_key.get_str().strip()
        secret_key = self.alp_secret_key.get_str().strip()
        symbols = [s.strip() for s in self.alp_symbols.get_str().split(",") if s.strip()]
        asset_class = self.alp_asset_class.get_str()
        timeframe_label = self.alp_timeframe.get_str()
        start = self.alp_start.get_str().strip()
        end = self.alp_end.get_str().strip()
        feed = self.alp_feed.get_str()
        adjustment = self.alp_adjustment.get_str()

        if not api_key or not secret_key:
            messagebox.showwarning("Missing keys", "Enter both an API key and a secret key first.")
            return
        if not symbols:
            messagebox.showwarning("Missing symbol", "Enter at least one symbol (comma-separated for more than one).")
            return

        self._set_alpaca_buttons_enabled(False)
        threading.Thread(
            target=self._fetch_alpaca_pipeline,
            args=(api_key, secret_key, symbols, asset_class, timeframe_label, start, end, feed, adjustment),
            daemon=True,
        ).start()

    def _fetch_alpaca_pipeline(self, api_key, secret_key, symbols, asset_class, timeframe_label, start, end, feed, adjustment):
        try:
            self._maybe_save_alpaca_keys(api_key, secret_key)
            saved_paths = []
            for i, symbol in enumerate(symbols, start=1):
                self._set_alpaca_status(f"Fetching {symbol} ({i}/{len(symbols)})...", AMBER)
                df = fetch_bars(
                    api_key, secret_key, symbol, asset_class, timeframe_label, start, end,
                    feed=feed, adjustment=adjustment,
                )
                dest = save_bars_as_csv(df, symbol, timeframe_label)
                saved_paths.append(dest)
                self._set_alpaca_status(f"Saved {symbol}: {len(df):,} bars -> {dest.name}", GREEN)

            self._refresh_dataset_list()
            if len(saved_paths) == 1:
                self._select_datasets(saved_paths, silent=True)
                self._set_alpaca_status(
                    f"Done. {saved_paths[0].name} is now the active dataset.", GREEN
                )
            else:
                names = ", ".join(p.name for p in saved_paths)
                self._set_alpaca_status(
                    f"Done. Saved {len(saved_paths)} file(s): {names}. "
                    "Ctrl/Cmd-click them in the list above to combine as multi-timeframe, "
                    "or pick just one.",
                    GREEN,
                )
        except (AlpacaImportError, AlpacaFetchError) as exc:
            self._set_alpaca_status(str(exc), RED)
        except Exception as exc:  # pragma: no cover - defensive
            self._set_alpaca_status(f"Unexpected error: {exc}", RED)
        finally:
            self._set_alpaca_buttons_enabled(True)

    # -----------------------------------------------------------------------
    # Tab 1 — Market Data — London Strategic Edge (LSE) API fetch
    # -----------------------------------------------------------------------
    # Mirrors _build_alpaca_section above field-for-field (same saved-key
    # checkbox, TEST CONNECTION / FETCH & SAVE / FORGET SAVED KEYS button
    # row, same worker-thread + status-label pattern) so the two data
    # sources feel identical from the UI's point of view -- only the
    # underlying module and its options differ (LSE is a single-API-key
    # service with no separate secret key, no feed/adjustment concept, and
    # its own asset-class/timeframe vocabulary; see
    # app.data.london_strategic_edge_source). Only a single-page
    # fetch_candles() pull is exposed here, matching the web app's own
    # /data/lse/fetch route (fetch_history()'s deeper vault-export pull
    # isn't wired into either UI yet). Previously only TEST CONNECTION
    # existed for LSE on desktop (on the API Keys tab); this adds the
    # actual FETCH & SAVE flow the Alpaca section has always had.

    def _build_lse_section(self, parent):
        section = self._section(
            parent,
            "Fetch data from London Strategic Edge",
            "Another alternative to local files above — a single API key covers stocks, forex, "
            "crypto, commodities, indices, ETFs and futures, and saves bars into data/raw/ so they "
            "join the dataset list above, same as an Alpaca fetch.",
        )

        saved_keys = load_apikeys_settings()
        prefill_key = saved_keys.london_strategic_edge_key or ""

        self.lse_api_key = LabeledEntry(section, "London Strategic Edge API key", prefill_key, secret=True, width=32)
        self.lse_save_key = LabeledCheckbox(
            section, "Save this key on this computer for next time", default=True
        )

        self.lse_symbols = LabeledEntry(
            section, "Symbol(s), comma-separated (e.g. AAPL, EUR/USD, GC)", "AAPL", width=32
        )
        self.lse_timeframe = LabeledCombo(section, "Timeframe", LSE_TIMEFRAME_CHOICES, default="1d")
        self.lse_start = LabeledEntry(section, "Start date (YYYY-MM-DD)", "2024-01-01")
        self.lse_end = LabeledEntry(section, "End date (YYYY-MM-DD)", "2026-01-01")

        btn_row = Frame(section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(4, 4))

        self.lse_test_btn = self._button(btn_row, "TEST CONNECTION", self._test_lse_connection)
        self.lse_test_btn.pack(side="left")

        self.lse_fetch_btn = self._button(btn_row, "FETCH & SAVE", self._fetch_lse_clicked, primary=True)
        self.lse_fetch_btn.pack(side="left", padx=8)

        self.lse_forget_btn = self._button(btn_row, "FORGET SAVED KEY", self._forget_lse_key)
        self.lse_forget_btn.pack(side="left")

        self.lse_status = Label(
            section,
            text="●  A London Strategic Edge account provides a single API key for data across every asset class.",
            bg=PANEL,
            fg=TEXT_MUTED,
            font=_safe_font(9),
            wraplength=760,
            justify="left",
        )
        self.lse_status.pack(anchor="w", padx=18, pady=(2, 14))

    def _set_lse_status(self, text, color=None):
        self.lse_status.config(text=f"●  {text}", fg=color or TEXT_MUTED)
        self.root.update_idletasks()

    def _set_lse_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.lse_test_btn.config(state=state)
        self.lse_fetch_btn.config(state=state)
        self.lse_forget_btn.config(state=state)

    def _maybe_save_lse_key(self, api_key: str):
        if not self.lse_save_key.get():
            return
        settings = load_apikeys_settings()
        settings.london_strategic_edge_key = api_key
        save_apikeys_settings(settings)

    def _forget_lse_key(self):
        settings = load_apikeys_settings()
        settings.london_strategic_edge_key = ""
        save_apikeys_settings(settings)
        self.lse_api_key.var.set("")
        self.lse_save_key.var.set(False)
        self._set_lse_status("Saved key removed from this computer.", GREEN)

    def _test_lse_connection(self):
        api_key = self.lse_api_key.get_str().strip()
        if not api_key:
            messagebox.showwarning("Missing key", "Enter a London Strategic Edge API key first.")
            return

        self._set_lse_buttons_enabled(False)
        self._set_lse_status("Testing connection...", AMBER)

        def worker():
            try:
                message = lse_test_connection(api_key)
                self._maybe_save_lse_key(api_key)
                self._set_lse_status(message, GREEN)
            except (LSEImportError, LSEFetchError) as exc:
                self._set_lse_status(str(exc), RED)
            except Exception as exc:  # pragma: no cover - defensive
                self._set_lse_status(f"Unexpected error: {exc}", RED)
            finally:
                self._set_lse_buttons_enabled(True)

        threading.Thread(target=worker, daemon=True).start()

    def _fetch_lse_clicked(self):
        api_key = self.lse_api_key.get_str().strip()
        symbols = [s.strip() for s in self.lse_symbols.get_str().split(",") if s.strip()]
        timeframe_label = self.lse_timeframe.get_str()
        start = self.lse_start.get_str().strip()
        end = self.lse_end.get_str().strip()

        if not api_key:
            messagebox.showwarning("Missing key", "Enter a London Strategic Edge API key first.")
            return
        if not symbols:
            messagebox.showwarning("Missing symbol", "Enter at least one symbol (comma-separated for more than one).")
            return

        self._set_lse_buttons_enabled(False)
        threading.Thread(
            target=self._fetch_lse_pipeline,
            args=(api_key, symbols, timeframe_label, start, end),
            daemon=True,
        ).start()

    def _fetch_lse_pipeline(self, api_key, symbols, timeframe_label, start, end):
        try:
            self._maybe_save_lse_key(api_key)
            saved_paths = []
            for i, symbol in enumerate(symbols, start=1):
                self._set_lse_status(f"Fetching {symbol} ({i}/{len(symbols)})...", AMBER)
                df = lse_fetch_candles(api_key, symbol, timeframe_label, start, end)
                dest = lse_save_bars_as_csv(df, symbol, timeframe_label)
                saved_paths.append(dest)
                self._set_lse_status(f"Saved {symbol}: {len(df):,} bars -> {dest.name}", GREEN)

            self._refresh_dataset_list()
            if len(saved_paths) == 1:
                self._select_datasets(saved_paths, silent=True)
                self._set_lse_status(
                    f"Done. {saved_paths[0].name} is now the active dataset.", GREEN
                )
            else:
                names = ", ".join(p.name for p in saved_paths)
                self._set_lse_status(
                    f"Done. Saved {len(saved_paths)} file(s): {names}. "
                    "Ctrl/Cmd-click them in the list above to combine as multi-timeframe, "
                    "or pick just one.",
                    GREEN,
                )
        except (LSEImportError, LSEFetchError) as exc:
            self._set_lse_status(str(exc), RED)
        except Exception as exc:  # pragma: no cover - defensive
            self._set_lse_status(f"Unexpected error: {exc}", RED)
        finally:
            self._set_lse_buttons_enabled(True)

    def _refresh_dataset_list(self):
        self.dataset_listbox.delete(0, END)
        self._stored_datasets = list_stored_datasets()
        by_name = {ds.name: i for i, ds in enumerate(self._stored_datasets)}

        # Grouped by instrument folder (e.g. data/raw/EURUSD/...) rather than
        # a flat mtime-sorted list, so opening the Data tab actually shows
        # "the different instrument folders" instead of one long file list.
        groups = list_datasets_by_instrument(count_rows=False)  # PERF: pickers never need row counts
        self._dataset_row_map: dict[int, int] = {}      # listbox row -> _stored_datasets index
        self._dataset_index_to_row: dict[int, int] = {}  # _stored_datasets index -> listbox row
        row = 0
        first_nonempty_row = None

        for g in groups:
            self.dataset_listbox.insert(END, f"\u2500\u2500 {g['instrument']} \u2500\u2500")
            self.dataset_listbox.itemconfig(row, fg=TEXT_MUTED, selectforeground=TEXT_MUTED, selectbackground=PANEL)
            row += 1
            for file_info in g["files"]:
                idx = by_name.get(file_info["full_name"])
                if idx is None:
                    continue
                label = f"    {file_info['name']}"
                if file_info["empty"]:
                    label += "   (empty)"
                elif file_info["rows"] == -1:
                    # -1 is _quick_row_count's "unknown, it's an archive"
                    # sentinel (see storage.py) -- flagged here so it
                    # doesn't just look like a file whose row count failed
                    # to load. Selecting it still works exactly like any
                    # other dataset; the archive is opened and its data
                    # member extracted at import time.
                    label += "   (archive — extracted on import)"
                self.dataset_listbox.insert(END, label)
                if file_info["empty"]:
                    self.dataset_listbox.itemconfig(row, fg=TEXT_DIM)
                elif first_nonempty_row is None:
                    first_nonempty_row = row
                self._dataset_row_map[row] = idx
                self._dataset_index_to_row[idx] = row
                row += 1

        if self._stored_datasets and not self.csv_paths:
            # Auto-select the first dataset with real data in it, not just
            # the most-recently-modified file — with data organized into
            # per-instrument folders it's common for some timeframes to be
            # empty placeholder exports (0 rows), and mtimes on a fresh
            # checkout/extraction can tie or favor one of those. Landing on
            # an empty file here used to silently leave "no data" active
            # (or pop a blocking error dialog) the moment the app opened.
            target_row = first_nonempty_row if first_nonempty_row is not None else next(iter(self._dataset_row_map), None)
            if target_row is not None:
                chosen = self._stored_datasets[self._dataset_row_map[target_row]]
                self.dataset_listbox.selection_set(target_row)
                self.dataset_listbox.see(target_row)
                self._select_datasets([chosen.path], silent=True)

    def _on_dataset_selected(self, _event):
        sel = self.dataset_listbox.curselection()
        paths = [self._stored_datasets[self._dataset_row_map[i]].path for i in sel if i in self._dataset_row_map]
        if not paths:
            return
        self._select_datasets(paths)

    def _select_datasets(self, paths: list[Path], silent: bool = False):
        """Load and validate one or more selected CSVs. Multiple files are
        treated as multiple timeframes for a multi-timeframe backtest.

        silent=True (used only for the automatic startup pick above) never
        pops a blocking messagebox — an empty/invalid file just leaves the
        status line red with an explanation, so the app can never freeze
        on launch waiting for a dialog no one is there to dismiss."""
        results = []
        for path in paths:
            result = import_csv(path)
            if not result.is_valid:
                detail = "; ".join(result.errors) or "no rows found"
                if silent:
                    self.data_status.config(
                        text=f"●  {path.name}: {detail} — pick another dataset from the list.",
                        fg=AMBER,
                    )
                else:
                    messagebox.showerror(
                        "Import failed",
                        f"{path.name}:\n" + "\n".join(result.errors),
                    )
                    self.data_status.config(text=f"●  {path.name}: import failed.", fg=RED)
                return
            results.append((path, result))

        self.csv_paths = [str(p) for p, _ in results]
        self.csv_path = self.csv_paths[0]

        total_warn = sum(len(r.warnings) for _, r in results)
        warn = f"  •  {total_warn} warning(s)" if total_warn else ""

        if len(results) == 1:
            path, result = results[0]
            n = len(result.dataframe)
            self.data_status.config(
                text=f"●  ACTIVE  {path.name}  •  {n:,} bars{warn}",
                fg=GREEN,
            )
        else:
            _, labels = merge_multi_timeframe([r.dataframe for _, r in results])
            names = ", ".join(p.name for p, _ in results)
            self.data_status.config(
                text=f"●  ACTIVE (multi-timeframe)  {names}  •  {' + '.join(labels)}{warn}",
                fg=GREEN,
            )

    # Backward-compatible alias used elsewhere in this file.
    def _select_dataset(self, path: Path):
        self._select_datasets([path])

    def _browse_csv(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Market data", "*.csv *.tsv *.txt *.parquet *.zip *.7z"), ("All files", "*.*")]
        )
        if not paths:
            return

        imported, failed = [], []

        for p in paths:
            result = import_csv(p)

            if not result.is_valid:
                failed.append(
                    (os.path.basename(p), "; ".join(result.errors))
                )
                continue

            stored_path = store_csv_path(p)
            imported.append(stored_path)

        self._refresh_dataset_list()

        if imported:
            self._select_dataset(imported[-1])

            for i, ds in enumerate(self._stored_datasets):
                if ds.path == imported[-1]:
                    row = self._dataset_index_to_row.get(i)
                    if row is not None:
                        self.dataset_listbox.selection_clear(0, END)
                        self.dataset_listbox.selection_set(row)
                        self.dataset_listbox.see(row)
                    break

        if failed:
            detail = "\n".join(
                f"- {name}: {err}" for name, err in failed
            )
            messagebox.showwarning(
                "Some files failed to import",
                f"{len(imported)} file(s) imported successfully.\n\n"
                f"{len(failed)} file(s) failed:\n{detail}",
            )
        elif imported:
            messagebox.showinfo(
                "Import complete",
                f"Imported and stored {len(imported)} file(s) in data/raw/.",
            )

    # -----------------------------------------------------------------------
    # Folder import — shared by the Market Data ("data") and Data Center
    # ("datacenter") tabs
    # -----------------------------------------------------------------------

    def _import_folder_clicked(self, tab: str):
        """IMPORT FOLDER... button: pick a folder and import every
        recognized file under it (recursively) through the shared
        app.data.folder_import pipeline -- CSV, TSV, TXT, and parquet land
        in data/raw/ exactly as if each had been imported individually."""
        folder = filedialog.askdirectory(title="Import market-data folder")
        if not folder:
            return
        self._run_folder_import_async([folder], [], tab)

    def _folder_import_status_label(self, tab: str):
        """The status label the import summary lands on -- the tab that
        started the import owns it. getattr-defensive because the tabs
        build in sequence."""
        label = self.data_status if tab == "data" else getattr(self, "datacenter_import_status", None)
        return label if label is not None else getattr(self, "data_status", None)

    def _run_folder_import_async(self, folders: list[str], files: list[str], tab: str):
        """Runs the folder/file import off the UI thread (a folder can be
        hundreds of MB) and paints the result on the Tk main thread via
        tk_safety.call_soon -- the same worker-thread pattern the Data
        Center health scan uses in _refresh_data_center."""
        if getattr(self, "_folder_import_running", False):
            messagebox.showinfo(
                "Import in progress",
                "A folder import is already running -- wait for it to finish.",
            )
            return
        self._folder_import_running = True
        status = self._folder_import_status_label(tab)
        if status is not None:
            status.config(text="●  Importing folder(s)...", fg=AMBER)

        def _work():
            try:
                # Lazy import: app.data.folder_import pulls pandas-level
                # weight only when an import actually runs.
                from app.data.folder_import import (
                    RECOGNIZED_SUFFIXES, FileImportOutcome, FolderImportReport,
                    import_file_bytes, import_folder_tree,
                )
                report = FolderImportReport()
                for folder in folders:
                    report.outcomes.extend(import_folder_tree(folder).outcomes)
                for path in files:
                    name = os.path.basename(path)
                    if Path(name).suffix.lower() not in RECOGNIZED_SUFFIXES:
                        report.outcomes.append(
                            FileImportOutcome(name, "skipped",
                                              f"unrecognized type '{Path(name).suffix or '(none)'}'")
                        )
                        continue
                    try:
                        content = Path(path).read_bytes()
                    except Exception as exc:  # noqa: BLE001 -- report, don't crash the batch
                        report.outcomes.append(FileImportOutcome(name, "failed", f"read error: {exc}"))
                        continue
                    report.outcomes.append(import_file_bytes(name, content))
            except Exception as exc:  # pragma: no cover - defensive
                from app.data.folder_import import FileImportOutcome, FolderImportReport
                report = FolderImportReport(
                    [FileImportOutcome("(import)", "failed", f"importer crashed: {exc}")]
                )
            try:
                from app.ui import tk_safety
                tk_safety.call_soon(lambda: self._finish_folder_import(report, tab))
            except Exception:
                self._folder_import_running = False  # window closed mid-import

        threading.Thread(target=_work, daemon=True).start()

    def _finish_folder_import(self, report, tab: str):
        """Runs on the Tk main thread: paints the summary on the owning
        tab's status label, shows the per-file breakdown, and refreshes
        both dataset lists so the new files appear without a restart."""
        self._folder_import_running = False
        summary = report.summary_line()
        status = self._folder_import_status_label(tab)
        color = GREEN if not report.failed else (AMBER if report.imported else RED)
        if status is not None:
            try:
                status.config(text=f"●  Folder import: {summary}", fg=color)
            except Exception:
                pass  # widgets gone (e.g. theme rebuild) -- nothing to paint into

        detail = report.detail_lines(max_lines=40)
        body = summary + ("\n\n" + "\n".join(detail) if detail else "")
        if not report.outcomes:
            messagebox.showinfo("Folder import", "No files found -- nothing was imported.")
        elif len(report.outcomes) > 12:
            # Many files: the scrollable read-only Text dialog (the same
            # viewer the strategy tabs use for code/config), not a giant
            # messagebox.
            self._show_text_viewer("Folder import — results", body)
        else:
            messagebox.showinfo("Folder import complete", body)

        # New files land in data/raw/ -- refresh both pickers. Guarded like
        # _data_center_import_clicked: either list may not exist yet.
        for refresh in (self._refresh_dataset_list, self._refresh_data_center):
            try:
                refresh()
            except Exception:
                pass

    def _on_data_files_dropped(self, event, tab: str):
        """<<Drop>> callback for the optional tkinterdnd2 drop targets on
        the data tabs. Folders go through the same background folder-import
        handler as the IMPORT FOLDER buttons; loose files go through the
        same per-file pipeline (validate + store into data/raw/)."""
        raw = getattr(event, "data", "") or ""
        try:
            paths = list(self.root.tk.splitlist(raw))
        except Exception:
            paths = [p for p in raw.replace("{", "").replace("}", "").split() if p]
        folders = [p for p in paths if os.path.isdir(p)]
        files = [p for p in paths if os.path.isfile(p)]
        if folders or files:
            self._run_folder_import_async(folders, files, tab)
        return "copy"  # DnD callback contract: report the action taken

    def _maybe_enable_folder_dnd(self):
        """Optional OS drag-and-drop: drop market-data folders (or loose
        data files) onto either data tab to import them. tkinterdnd2 is NOT
        a hard dependency -- importing it monkeypatches every Tk widget
        with drop_target_register/dnd_bind, and require() loads the native
        tkdnd package into this interpreter. When the package isn't
        installed this is a silent no-op and the IMPORT FOLDER buttons
        remain the way in."""
        if getattr(self, "_folder_dnd_enabled", False):
            return
        if tkinterdnd2 is None:
            return
        try:
            tkinterdnd2.require(self.root)
        except Exception:
            return  # native tkdnd unavailable on this platform -- skip silently
        ok = True
        for frame, tab in ((self.tab_data, "data"), (self.tab_datacenter, "datacenter")):
            try:
                frame.drop_target_register(tkinterdnd2.DND_FILES)
                frame.dnd_bind("<<Drop>>", lambda event, t=tab: self._on_data_files_dropped(event, t))
            except Exception:
                ok = False  # keep whatever did register; the buttons still work
        self._folder_dnd_enabled = ok

    # -----------------------------------------------------------------------
    # Data Center -- desktop parity for the web app's Data Center page
    # (app.web.server.data_center / templates/data_center.html): dataset
    # coverage, timeframe availability, gaps, duplicates, timezone,
    # sessions, and bar counts for everything under data/raw/, plus a CSV/
    # parquet import button. Shares app.data.health.compute_data_center
    # with the web app so the two can never disagree about a file's health.
    # -----------------------------------------------------------------------

    def _build_data_center_tab(self):
        f = self._scrollable(self.tab_datacenter)
        self._page_header(
            f, "ACCOUNT", "Data Center",
            "Dataset coverage, timeframe availability, gaps, duplicates, timezone, sessions, and bar "
            "counts for everything under data/raw/ -- plus a place to import new CSV or parquet files.",
        )

        import_section = self._section(
            f, "Import data", "CSV, TSV, or parquet -- saved into data/raw/ as its own file, same as "
            "fetching from Alpaca or London Strategic Edge.",
        )
        import_btn_row = Frame(import_section, bg=PANEL)
        import_btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self._button(import_btn_row, "IMPORT CSV / PARQUET...", self._data_center_import_clicked, primary=True).pack(side="left")
        self._button(import_btn_row, "IMPORT FOLDER...", lambda: self._import_folder_clicked("datacenter")).pack(side="left", padx=(8, 0))
        self.datacenter_import_status = Label(import_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=820, justify="left")
        self.datacenter_import_status.pack(anchor="w", padx=18, pady=(4, 12))

        summary_section = self._section(f, "Overview", "")
        self.datacenter_summary_frame = Frame(summary_section, bg=PANEL)
        self.datacenter_summary_frame.pack(fill="x", padx=18, pady=(0, 12))

        self.datacenter_groups_frame = Frame(f, bg=BG)
        self.datacenter_groups_frame.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        refresh_row = Frame(f, bg=BG)
        refresh_row.pack(fill="x", padx=24, pady=(0, 10))
        self._button(refresh_row, "REFRESH", self._refresh_data_center).pack(side="left")

        # PERF: the health scan parses every dataset with pandas (tens of
        # seconds for a few hundred MB). It no longer runs at startup -- it
        # runs in the background the first time this tab is opened (see
        # _show_page) and on REFRESH.
        self._datacenter_scanned = False
        self._datacenter_scanning = False
        self._maybe_enable_folder_dnd()
        Label(
            self.datacenter_groups_frame, text="Open this tab to scan your datasets.",
            bg=BG, fg=TEXT_DIM, font=_safe_font(9),
        ).pack(anchor="w", pady=8)

    def _data_center_import_clicked(self):
        path = filedialog.askopenfilename(
            title="Import market data",
            filetypes=[("Market data", "*.csv *.tsv *.txt *.parquet"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            dest = store_csv_path(path)
            self.datacenter_import_status.config(text=f"Imported '{Path(path).name}' -> {dest.name}.", fg=GREEN)
        except Exception as exc:  # pragma: no cover - defensive
            self.datacenter_import_status.config(text=f"Could not import '{Path(path).name}': {exc}", fg=RED)
        self._refresh_data_center()
        try:
            self._refresh_dataset_list()
        except Exception:
            pass

    def _refresh_data_center(self):
        """Scans datasets on a worker thread (never blocks the UI), then
        paints on the Tk main thread."""
        if getattr(self, "_datacenter_scanning", False):
            return
        self._datacenter_scanning = True
        for child in self.datacenter_summary_frame.winfo_children():
            child.destroy()
        for child in self.datacenter_groups_frame.winfo_children():
            child.destroy()
        Label(
            self.datacenter_groups_frame, text="Scanning datasets...", bg=BG, fg=TEXT_MUTED, font=_safe_font(9),
        ).pack(anchor="w", pady=8)

        def _work():
            try:
                from app.data.health import compute_data_center
                report, error = compute_data_center(), None
            except Exception as exc:  # pragma: no cover - defensive
                report, error = None, exc
            try:
                from app.ui import tk_safety
                tk_safety.call_soon(lambda: self._paint_data_center(report, error))
            except Exception:
                self._datacenter_scanning = False  # window closed mid-scan

        threading.Thread(target=_work, daemon=True).start()

    def _paint_data_center(self, report, error=None):
        self._datacenter_scanning = False
        self._datacenter_scanned = True
        try:
            for child in self.datacenter_summary_frame.winfo_children():
                child.destroy()
            for child in self.datacenter_groups_frame.winfo_children():
                child.destroy()
        except Exception:
            return  # widgets gone (e.g. theme rebuild) -- nothing to paint into

        if error is not None or report is None:
            Label(
                self.datacenter_groups_frame, text=f"Could not compute data health: {error}",
                bg=BG, fg=RED, font=_safe_font(9),
            ).pack(anchor="w")
            return

        stats = [
            ("Files", str(report["total_files"])),
            ("Total bars", f"{report['total_bars']:,}"),
            ("Files needing attention", str(report["total_unhealthy"])),
            ("Instruments", str(len(report["instruments"]))),
        ]
        for i, (label, value) in enumerate(stats):
            cell = Frame(self.datacenter_summary_frame, bg=PANEL)
            cell.pack(side="left", padx=(0 if i == 0 else 24, 0))
            color = RED if label == "Files needing attention" and value != "0" else TEXT
            Label(cell, text=value, bg=PANEL, fg=color, font=_safe_font(16, "bold")).pack(anchor="w")
            Label(cell, text=label.upper(), bg=PANEL, fg=TEXT_MUTED, font=_safe_font(7, "bold")).pack(anchor="w")

        if not report["instruments"]:
            Label(
                self.datacenter_groups_frame,
                text="No datasets found under data/raw/ yet -- import one above, or fetch data from Alpaca/"
                     "London Strategic Edge on the Market Data tab.",
                bg=BG, fg=TEXT_DIM, font=_safe_font(9), wraplength=880, justify="left",
            ).pack(anchor="w", pady=8)
            return

        for group in report["instruments"]:
            group_wrap = Frame(self.datacenter_groups_frame, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
            group_wrap.pack(fill="x", pady=(0, 10))
            header = Frame(group_wrap, bg=PANEL)
            header.pack(fill="x", padx=14, pady=(10, 4))
            Label(header, text=group["instrument"], bg=PANEL, fg=TEXT, font=_safe_font(11, "bold")).pack(side="left")
            tf_text = ", ".join(group["timeframes_available"]) if group["timeframes_available"] else "unknown"
            Label(
                header, text=f"  {group['file_count']} file(s) · {group['total_bars']:,} bars · timeframes: {tf_text}",
                bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8),
            ).pack(side="left")
            if group["unhealthy_count"]:
                Label(
                    header, text=f"  ⚠ {group['unhealthy_count']} need attention", bg=PANEL, fg=AMBER, font=_safe_font(8, "bold"),
                ).pack(side="left")

            for file_info in group["files"]:
                row = Frame(group_wrap, bg=PANEL_2)
                row.pack(fill="x", padx=14, pady=1)
                dot_color = GREEN if file_info["healthy"] else RED
                Label(row, text="●", bg=PANEL_2, fg=dot_color, font=_safe_font(9)).pack(side="left", padx=(0, 6))
                Label(row, text=file_info["name"], bg=PANEL_2, fg=TEXT, font=_safe_font(9)).pack(side="left")
                detail_bits = [f"{file_info['bar_count']:,} bars"]
                if file_info.get("timeframe"):
                    detail_bits.append(file_info["timeframe"])
                if file_info.get("timezone"):
                    detail_bits.append(file_info["timezone"])
                if file_info.get("session_coverage_pct"):
                    detail_bits.append(f"{file_info['session_coverage_pct']:.0f}% of hours covered")
                Label(
                    row, text="  —  " + " · ".join(detail_bits), bg=PANEL_2, fg=TEXT_MUTED, font=_safe_font(8),
                ).pack(side="left")
                if file_info.get("issues"):
                    Label(
                        row, text="  " + "; ".join(file_info["issues"]), bg=PANEL_2, fg=AMBER, font=_safe_font(8),
                        wraplength=500, justify="left",
                    ).pack(side="left")
                if file_info.get("error"):
                    Label(row, text="  " + file_info["error"], bg=PANEL_2, fg=RED, font=_safe_font(8)).pack(side="left")
