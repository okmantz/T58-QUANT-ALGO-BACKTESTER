"""Run-context panel and dialog."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ui.main_window import MainWindow

import os
import threading
from app.backtest.risk import RiskConfig, build_run_context, suggest_pip_size
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
from app.data.instrument_specs import get_instrument_spec, known_instrument_symbols
from app.data.multi_timeframe import merge_multi_timeframe
from app.data.storage import list_datasets_by_instrument, list_stored_datasets, store_csv_path
from app.prop.presets import list_presets as list_prop_firm_presets
from app.prop.simulator import PropRules
from tkinter import Canvas, END, EXTENDED, Frame, Label, Listbox, Toplevel, filedialog, messagebox, ttk
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
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
GREEN = ""
RED = ""
AMBER = ""
theme.bind(globals())


class RunContextPanel:
    """A fully self-contained Market Data + Prop Account + Risk & Execution
    configuration block, embeddable directly inside any single tab.

    This is the desktop-side answer to a gap from the web app: over there,
    Full Pipeline / Search Lab / Quick Optimize / etc. each have their OWN
    copy of "which dataset, which prop-firm rules, which risk settings" --
    nothing is read from some other page, so opening any one of those tools
    is a complete, self-sufficient ecosystem. The desktop app instead grew
    ONE shared Market Data tab (Step 02) and ONE shared Prop Rules / Risk
    tab pair (Steps 03-04) that every other tab quietly read from via
    self.csv_paths / self._build_prop_rules() / self._build_risk_config().
    That's convenient when you want one setup to apply everywhere, but it
    means changing the dataset or the account size for a Search Lab run
    also silently changes what Full Pipeline (or anything else) would use
    next -- tabs aren't independent the way they are on the web.

    RunContextPanel gives a tab its own private copy of exactly those three
    things -- an independent dataset picker (with the same local-file +
    Alpaca-fetch options as Step 02), its own Prop Account fields (same as
    Step 03, presets included), and its own Risk & Execution fields (same
    as Step 04, minus the Adaptive Risk sub-builder, which is niche enough
    that tabs needing it keep their own toggle inline as they already do).
    Every instance keeps its state on itself (self.csv_paths, self.p_*,
    self.r_*, self.alp_*) -- never on the MainWindow -- so two tabs each
    holding a RunContextPanel never see or affect each other's settings.

    `owner` is the MainWindow instance this panel is embedded in; only used
    for the shared, state-free widget builders (_button, _section,
    _bind_isolated_wheel) and .root, never for reading or writing owner's
    own self.csv_paths / self.p_* / self.r_* fields.
    """

    def __init__(self, owner: "MainWindow", key: str, default_account_size: float = 100000):
        self.owner = owner
        self.key = key  # unique per embedding tab, used only in status text
        self.csv_paths: list[str] = []
        self._stored_datasets = []
        self._dataset_row_map: dict[int, int] = {}
        self._dataset_index_to_row: dict[int, int] = {}
        self._default_account_size = default_account_size

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, parent, include_data: bool = True) -> None:
        if include_data:
            self._build_data_section(parent)
        self._build_prop_section(parent)
        self._build_risk_section(parent)

    def _build_data_section(self, parent):
        section = self.owner._section(
            parent, "Market data (this run only)",
            "Independent of every other tab's data selection -- pick or fetch a dataset "
            "just for this run. Ctrl/Cmd-click or Shift-click for multi-timeframe.",
            emphasize=True,
        )
        list_frame = Frame(section, bg=PANEL)
        list_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.dataset_listbox = Listbox(
            list_frame, height=6, selectmode=EXTENDED, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.dataset_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.dataset_listbox.yview, style="T58.Vertical.TScrollbar",
        )
        scrollbar.pack(side="right", fill="y")
        self.dataset_listbox.config(yscrollcommand=scrollbar.set)
        self.owner._bind_isolated_wheel(self.dataset_listbox)
        self.dataset_listbox.bind("<<ListboxSelect>>", self._on_dataset_selected)

        btn_row = Frame(section, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(0, 10))
        self.owner._button(btn_row, "IMPORT CSV(S)", self._browse_csv, primary=True).pack(side="left")
        self.owner._button(btn_row, "REFRESH LIST", self._refresh_dataset_list).pack(side="left", padx=8)

        self.data_status = Label(
            section, text="●  No dataset selected for this run.", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
        )
        self.data_status.pack(anchor="w", padx=18, pady=(0, 10))

        self._build_alpaca_section(section)
        self._refresh_dataset_list()

    def _build_alpaca_section(self, parent):
        """Same fetch-from-Alpaca flow as Step 02's own copy (see
        MainWindow._build_alpaca_section), duplicated here with entirely
        instance-local widgets/state so it saves into (and this panel's
        list refreshes from) the same data/raw/ folder without touching
        any other tab's selection."""
        sub = self.owner._section(
            parent, "Fetch data from Alpaca",
            "Pulls bars via the Alpaca API (stocks + crypto only) straight into data/raw/, "
            "then selects the result for this run automatically.",
        )
        saved = alpaca_credentials.load_credentials()
        prefill_key = saved.api_key if saved else ""
        prefill_secret = saved.secret_key if saved else ""

        self.alp_api_key = LabeledEntry(sub, "Alpaca API key", prefill_key, secret=True, width=32)
        self.alp_secret_key = LabeledEntry(sub, "Alpaca secret key", prefill_secret, secret=True, width=32)
        self.alp_save_keys = LabeledCheckbox(sub, "Save these keys on this computer for next time", default=True)
        self.alp_asset_class = LabeledCombo(sub, "Asset class", ASSET_CLASSES, default=ASSET_CLASSES[0])
        self.alp_asset_class.combo.bind("<<ComboboxSelected>>", lambda _e: self._on_alpaca_asset_class_changed())
        self.alp_symbols = LabeledEntry(sub, "Symbol(s), comma-separated", "AAPL", width=32)
        self.alp_timeframe = LabeledCombo(sub, "Timeframe", TIMEFRAME_LABELS, default="1Day")
        self.alp_start = LabeledEntry(sub, "Start date (YYYY-MM-DD)", "2024-01-01")
        self.alp_end = LabeledEntry(sub, "End date (YYYY-MM-DD)", "2026-01-01")
        self.alp_feed = LabeledCombo(sub, "Feed (stocks only)", FEED_CHOICES, default="iex")
        self.alp_adjustment = LabeledCombo(sub, "Adjustment (stocks only)", ADJUSTMENT_CHOICES, default="raw")

        btn_row = Frame(sub, bg=PANEL)
        btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self.alp_test_btn = self.owner._button(btn_row, "TEST CONNECTION", self._test_alpaca_connection)
        self.alp_test_btn.pack(side="left")
        self.alp_fetch_btn = self.owner._button(btn_row, "FETCH & SAVE", self._fetch_alpaca_clicked, primary=True)
        self.alp_fetch_btn.pack(side="left", padx=8)
        self.alp_forget_btn = self.owner._button(btn_row, "FORGET SAVED KEYS", self._forget_alpaca_keys)
        self.alp_forget_btn.pack(side="left")

        self.alpaca_status = Label(
            sub, text="●  A free Alpaca paper-trading account provides API keys for data access.",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), wraplength=760, justify="left",
        )
        self.alpaca_status.pack(anchor="w", padx=18, pady=(2, 12))

    def _build_prop_section(self, parent):
        section = self.owner._section(
            parent, "Prop-firm rules (this run only)",
            "Own copy of account size, drawdown, consistency, and payout rules -- "
            "changing these never affects any other tab.",
        )
        preset_labels = ["(Choose a preset...)"] + [p.label for p in list_prop_firm_presets()]
        self.p_preset_combo = LabeledCombo(section, "Prop-firm preset", preset_labels, default=preset_labels[0])
        self.p_preset_note = Label(
            section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), anchor="w", justify="left", wraplength=560,
        )
        self.p_preset_note.pack(fill="x", padx=18, pady=(0, 6))
        self.p_preset_combo.combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_prop_firm_preset())

        self.p_account_size = LabeledEntry(section, "Account size ($)", self._default_account_size)
        self.p_profit_target = LabeledEntry(section, "Evaluation profit target (%)", 8)
        self.p_daily_loss = LabeledEntry(section, "Daily loss limit (%)", 5)
        self.p_max_dd = LabeledEntry(section, "Maximum drawdown (%)", 10)
        self.p_dd_type = LabeledEntry(section, "Drawdown type (trailing/static)", "trailing")
        self.p_dd_check_mode = LabeledEntry(section, "Drawdown check mode (intrabar/eod)", "intrabar")
        self.p_consistency = LabeledEntry(section, "Consistency rule (% best day of total profit)", 30)
        self.p_min_days = LabeledEntry(section, "Minimum trading days", 5)
        self.p_payout_threshold = LabeledEntry(section, "Payout threshold (extra % profit)", 0)
        self.p_payout_cap = LabeledEntry(section, "Payout cap (% of profit, blank=100)", 100)
        self.p_payout_freq = LabeledEntry(section, "Payout frequency (days)", 14)
        self.p_buffer = LabeledEntry(section, "Required buffer (%)", 0)
        self.p_max_pos = LabeledEntry(section, "Max position size (units, blank=unlimited)", "")

    def _build_risk_section(self, parent):
        section = self.owner._section(
            parent, "Risk & execution (this run only)",
            "Own copy of position risk, trading frequency, transaction costs, and pip size.",
        )
        # RISK-001 fix: same reasoning as the main Tab 3/4 pair -- these
        # two fields represent the same "this run only" account, so they
        # share one StringVar (self.p_account_size.var, built just above
        # in _build_prop_section) rather than being two independently-
        # editable numbers that can silently disagree.
        self.r_initial_balance = LabeledEntry(
            section, "Initial balance ($) -- kept in sync with Account size above",
            variable=self.p_account_size.var,
        )
        self.r_risk_mode = LabeledEntry(section, "Risk mode (percent/fixed)", "percent")
        self.r_risk_value = LabeledEntry(section, "Risk per trade (% or $)", 1.0)
        self.r_max_trades_day = LabeledEntry(section, "Max trades/day", 10)
        self.r_commission = LabeledEntry(section, "Commission per trade ($)", 3)
        self.r_slippage = LabeledEntry(section, "Slippage (pips)", 0.5)
        self.r_spread = LabeledEntry(section, "Spread (pips)", 1.0)
        self.r_pip_size = LabeledEntry(section, "Pip size (e.g. 0.0001 FX)", 0.0001)
        # UPGRADE (instrument-metadata-had-no-ui): same wiring as the main
        # Step 04 Risk & Execution tab's matching dropdown (see
        # MainWindow._apply_instrument_spec's docstring) -- this run's own
        # independent copy, since RunContextPanel's whole point is that a
        # tab's settings never leak into or out of another tab's.
        self.r_instrument = LabeledCombo(
            section, "Or pick a known instrument (auto-fills pip size + $/point)",
            ["(none -- set manually)"] + known_instrument_symbols(),
            default="(none -- set manually)",
        )
        self.r_instrument.combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_instrument_spec())
        self.r_contract_size = LabeledEntry(
            section, "Contract $/point (blank = not lot-rounded; auto-filled by instrument pick above)", "",
        )

        pip_row = Frame(section, bg=PANEL)
        pip_row.pack(anchor="w", padx=18, pady=(0, 10))
        self.owner._button(pip_row, "DETECT PIP SIZE FROM DATA", self._detect_pip_size_from_data).pack(side="left")
        self.pip_detect_status = Label(pip_row, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8))
        self.pip_detect_status.pack(side="left", padx=(10, 0))

    # ------------------------------------------------------------------
    # Dataset selection (instance-local -- mirrors MainWindow's Step 02
    # methods exactly, but never touches self.owner.csv_paths)
    # ------------------------------------------------------------------

    def _refresh_dataset_list(self):
        self.dataset_listbox.delete(0, END)
        self._stored_datasets = list_stored_datasets()
        by_name = {ds.name: i for i, ds in enumerate(self._stored_datasets)}
        groups = list_datasets_by_instrument(count_rows=False)  # PERF: pickers never need row counts
        self._dataset_row_map = {}
        self._dataset_index_to_row = {}
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
                    label += "   (archive — extracted on import)"
                self.dataset_listbox.insert(END, label)
                if file_info["empty"]:
                    self.dataset_listbox.itemconfig(row, fg=TEXT_DIM)
                elif first_nonempty_row is None:
                    first_nonempty_row = row
                self._dataset_row_map[row] = idx
                self._dataset_index_to_row[idx] = row
                row += 1
        # Unlike Step 02, this panel does NOT auto-select a dataset on
        # first build -- an empty selection here is exactly the correct,
        # obvious starting state for a run-local picker (Step 02 auto-
        # selects because it's the app's one persistent default).

    def _on_dataset_selected(self, _event):
        sel = self.dataset_listbox.curselection()
        paths = [self._stored_datasets[self._dataset_row_map[i]].path for i in sel if i in self._dataset_row_map]
        if not paths:
            return
        self._select_datasets(paths)

    def _select_datasets(self, paths, silent: bool = False):
        results = []
        for path in paths:
            result = import_csv(path)
            if not result.is_valid:
                detail = "; ".join(result.errors) or "no rows found"
                if silent:
                    self.data_status.config(text=f"●  {path.name}: {detail} — pick another dataset.", fg=AMBER)
                else:
                    messagebox.showerror("Import failed", f"{path.name}:\n" + "\n".join(result.errors))
                    self.data_status.config(text=f"●  {path.name}: import failed.", fg=RED)
                return
            results.append((path, result))

        self.csv_paths = [str(p) for p, _ in results]

        # Auto-suggest pip size the moment a dataset is chosen -- removes
        # the silent-default trap where r_pip_size quietly sits at its FX
        # default (0.0001) against a non-FX instrument until someone
        # remembers to click "DETECT PIP SIZE FROM DATA" by hand (see
        # CIRCULARITY_AUDIT.md-adjacent diagnosis: this is exactly how two
        # independent RunContextPanel instances ended up with different
        # pip_size for what the person believed was "the same run").
        # Every OTHER risk/prop field (account size, spread, commission,
        # slippage, drawdown rules, ...) has no equivalent way to derive a
        # value from OHLCV data alone -- pip_size is the one field this
        # codebase has an actual, tested heuristic for
        # (app.backtest.risk.suggest_pip_size, keyed to the data's own
        # price scale), so it's the one auto-filled here. Guarded with
        # hasattr since _build_risk_section (which creates r_pip_size)
        # runs after _build_data_section during .build() -- this only
        # ever fires from real user interaction after the panel is fully
        # built, but stays defensive rather than assuming that ordering.
        if hasattr(self, "r_pip_size"):
            try:
                combined_df = results[0][1].dataframe if len(results) == 1 else \
                    merge_multi_timeframe([r.dataframe for _, r in results])[0]
                suggested = suggest_pip_size(combined_df)
                self.r_pip_size.var.set(str(suggested))
                if hasattr(self, "pip_detect_status"):
                    self.pip_detect_status.config(
                        text=f"Auto-suggested {suggested} from {os.path.basename(str(paths[0]))} "
                             f"-- confirm this matches the instrument.",
                        fg=GREEN,
                    )
            except Exception:
                pass  # best-effort -- never let a suggestion failure block picking a dataset

        total_warn = sum(len(r.warnings) for _, r in results)
        warn = f"  •  {total_warn} warning(s)" if total_warn else ""
        if len(results) == 1:
            path, result = results[0]
            n = len(result.dataframe)
            self.data_status.config(text=f"●  ACTIVE  {path.name}  •  {n:,} bars{warn}", fg=GREEN)
        else:
            _, labels = merge_multi_timeframe([r.dataframe for _, r in results])
            names = ", ".join(p.name for p, _ in results)
            self.data_status.config(text=f"●  ACTIVE (multi-timeframe)  {names}  •  {' + '.join(labels)}{warn}", fg=GREEN)

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
                failed.append((os.path.basename(p), "; ".join(result.errors)))
                continue
            imported.append(store_csv_path(p))
        self._refresh_dataset_list()
        if imported:
            self._select_datasets([imported[-1]], silent=True)
            for i, ds in enumerate(self._stored_datasets):
                if ds.path == imported[-1]:
                    row = self._dataset_index_to_row.get(i)
                    if row is not None:
                        self.dataset_listbox.selection_clear(0, END)
                        self.dataset_listbox.selection_set(row)
                        self.dataset_listbox.see(row)
                    break
        if failed:
            detail = "\n".join(f"- {name}: {err}" for name, err in failed)
            messagebox.showwarning(
                "Some files failed to import",
                f"{len(imported)} file(s) imported successfully.\n\n{len(failed)} file(s) failed:\n{detail}",
            )
        elif imported:
            messagebox.showinfo("Import complete", f"Imported and stored {len(imported)} file(s) in data/raw/.")

    # ------------------------------------------------------------------
    # Alpaca fetch (instance-local)
    # ------------------------------------------------------------------

    def _on_alpaca_asset_class_changed(self):
        is_stock = self.alp_asset_class.get_str() == "Stock"
        state = "readonly" if is_stock else "disabled"
        self.alp_feed.combo.config(state=state)
        self.alp_adjustment.combo.config(state=state)

    def _set_alpaca_status(self, text, color=None):
        self.alpaca_status.config(text=f"●  {text}", fg=color or TEXT_MUTED)
        self.owner.root.update_idletasks()

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
                self._set_alpaca_status(f"Done. {saved_paths[0].name} is now this run's active dataset.", GREEN)
            else:
                names = ", ".join(p.name for p in saved_paths)
                self._set_alpaca_status(
                    f"Done. Saved {len(saved_paths)} file(s): {names}. "
                    "Ctrl/Cmd-click them in the list above to combine as multi-timeframe, or pick just one.",
                    GREEN,
                )
        except (AlpacaImportError, AlpacaFetchError) as exc:
            self._set_alpaca_status(str(exc), RED)
        except Exception as exc:  # pragma: no cover - defensive
            self._set_alpaca_status(f"Unexpected error: {exc}", RED)
        finally:
            self._set_alpaca_buttons_enabled(True)

    # ------------------------------------------------------------------
    # Prop rules / risk config / dataframe resolution
    # ------------------------------------------------------------------

    def _apply_prop_firm_preset(self):
        idx = self.p_preset_combo.combo.current()
        presets = list_prop_firm_presets()
        if idx <= 0 or idx - 1 >= len(presets):
            self.p_preset_note.config(text="")
            return
        p = presets[idx - 1]
        self.p_account_size.var.set(str(p.account_size))
        self.p_profit_target.var.set(str(p.evaluation_profit_target_pct))
        self.p_daily_loss.var.set(str(p.daily_loss_limit_pct))
        self.p_max_dd.var.set(str(p.max_drawdown_pct))
        self.p_dd_type.var.set(p.drawdown_type)
        self.p_dd_check_mode.var.set(p.drawdown_check_mode)
        self.p_consistency.var.set(str(p.consistency_rule_pct) if p.consistency_rule_pct is not None else "")
        self.p_min_days.var.set(str(p.min_trading_days))
        self.p_payout_threshold.var.set(str(p.payout_threshold_pct))
        self.p_payout_cap.var.set(str(p.payout_cap_pct) if p.payout_cap_pct is not None else "100")
        self.p_payout_freq.var.set(str(p.payout_frequency_days))
        self.p_buffer.var.set(str(p.required_buffer_pct))
        self.p_preset_note.config(text=f"As of {p.as_of}: {p.source_note}")

    def build_prop_rules(self) -> PropRules:
        cap = self.p_payout_cap.get_str().strip()
        max_pos = self.p_max_pos.get_str().strip()
        return PropRules(
            account_size=self.p_account_size.get_float(self._default_account_size),
            evaluation_profit_target_pct=self.p_profit_target.get_float(8),
            daily_loss_limit_pct=self.p_daily_loss.get_float(5),
            max_drawdown_pct=self.p_max_dd.get_float(10),
            drawdown_type=self.p_dd_type.get_str().strip() or "trailing",
            drawdown_check_mode=self.p_dd_check_mode.get_str().strip() or "intrabar",
            consistency_rule_pct=self.p_consistency.get_float(30),
            min_trading_days=self.p_min_days.get_int(5),
            payout_threshold_pct=self.p_payout_threshold.get_float(0),
            payout_cap_pct=float(cap) if cap else None,
            payout_frequency_days=self.p_payout_freq.get_int(14),
            required_buffer_pct=self.p_buffer.get_float(0),
            max_position_size=float(max_pos) if max_pos else None,
        )

    def build_risk_config(self) -> RiskConfig:
        contract_size_str = self.r_contract_size.get_str().strip()
        risk = RiskConfig(
            initial_balance=self.r_initial_balance.get_float(self._default_account_size),
            risk_mode=self.r_risk_mode.get_str().strip() or "percent",
            risk_value=self.r_risk_value.get_float(1.0),
            max_trades_per_day=self.r_max_trades_day.get_int(10),
            commission_per_trade=self.r_commission.get_float(0),
            slippage_pips=self.r_slippage.get_float(0.5),
            spread_pips=self.r_spread.get_float(1.0),
            pip_size=self.r_pip_size.get_float(0.0001),
            contract_size=float(contract_size_str) if contract_size_str else None,
        )
        # HARDENING (v9.6, audit finding A2): every RunContextPanel tab used
        # to start from RiskConfig's raw defaults (sizing_mode="skip",
        # account_model="legacy", no intrabar replay), so results from
        # WFO/CPCV/Sensitivity/etc. weren't comparable to Full Pipeline's.
        # build_run_context is the single backend chokepoint that applies
        # the prop safety defaults against THIS panel's own prop rules --
        # route every panel-built risk through it before returning.
        return build_run_context(risk, self.build_prop_rules())

    def _apply_instrument_spec(self):
        symbol = self.r_instrument.var.get()
        spec = get_instrument_spec(symbol)
        if spec is None:
            return
        self.r_pip_size.var.set(str(spec.pip_size))
        self.r_contract_size.var.set(str(spec.contract_size))
        # COMMISSION-DEFAULT-UPGRADE (2026-09-24): only fills in a realistic
        # commission when the field is still at its 0 default -- never
        # overwrites a rate the user already typed in. Mirrors app.data.
        # instrument_specs.apply_instrument_spec's own rule for the web/
        # backend path, so a strategy picked from this dropdown can no
        # longer end up silently backtested at $0 commission.
        commission_note = ""
        if self.r_commission.get_float(0) in (0.0, 3.0):
            self.r_commission.var.set(str(spec.default_commission_round_turn))
            commission_note = f", commission=${spec.default_commission_round_turn:g}/trade"
        self.pip_detect_status.config(
            text=f"Applied {spec.symbol} ({spec.description}, {spec.exchange}): "
                 f"pip_size={spec.pip_size}, ${spec.contract_size}/point per contract{commission_note}.",
            fg=GREEN,
        )

    def _detect_pip_size_from_data(self):
        if not self.csv_paths:
            self.pip_detect_status.config(text="Select a market data CSV above first.", fg=AMBER)
            return
        try:
            result = import_csv(self.csv_paths[0])
            if not result.is_valid:
                self.pip_detect_status.config(text="Couldn't read that CSV.", fg=RED)
                return
            suggested = suggest_pip_size(result.dataframe)
            self.r_pip_size.var.set(str(suggested))
            self.pip_detect_status.config(
                text=f"Suggested {suggested} from {os.path.basename(self.csv_paths[0])} "
                     f"-- confirm this matches the instrument.",
                fg=GREEN,
            )
        except Exception as exc:
            self.pip_detect_status.config(text=f"Couldn't detect: {exc}", fg=RED)

    def load_dataframe(self, log_fn):
        """Same shape/contract as MainWindow._load_df_for_page, but reads
        this panel's own self.csv_paths instead of the shared Step 02
        selection."""
        if not self.csv_paths:
            log_fn(f"Please select a market data CSV above for this {self.key} run first.")
            return None
        per_file_results = []
        for p in self.csv_paths:
            result = import_csv(p)
            if not result.is_valid:
                log_fn(f"Import errors ({os.path.basename(p)}):\n" + "\n".join(result.errors))
                return None
            per_file_results.append((p, result))
        if len(per_file_results) == 1:
            df = per_file_results[0][1].dataframe
        else:
            df, _labels = merge_multi_timeframe([r.dataframe for _, r in per_file_results])
        log_fn(f"Loaded {len(df)} bars.")
        return df

    def instrument_label(self) -> str:
        if not self.csv_paths:
            return "unknown"
        return (
            os.path.basename(self.csv_paths[0]) if len(self.csv_paths) == 1
            else " + ".join(os.path.basename(p) for p in self.csv_paths)
        )


class RunContextDialog(Toplevel):
    """Modal, one-off version of RunContextPanel for actions that aren't a
    whole tab of their own -- e.g. OPTIMIZE SELECTED in the Strategy
    Library. Confirming this dialog resolves this ONE click's market data /
    prop rules / risk settings entirely on its own (nothing shared with the
    Data/Prop Rules/Risk tabs or any other tab), matching the same
    self-contained-ecosystem idea as a full RunContextPanel tab, just
    scoped to a single action instead of a persistent page."""

    def __init__(self, parent, owner: "MainWindow", action_label: str):
        super().__init__(parent)
        self.title(f"{action_label} — market data & prop settings")
        self.configure(bg=BG)
        self.geometry("720x680")
        self.confirmed = False

        outer = Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)
        canvas = Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, style="T58.Vertical.TScrollbar")
        inner = Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta // 120)), "units"))

        Label(
            inner, text=f"Configure THIS {action_label} run",
            bg=BG, fg=TEXT, font=_safe_font(12, "bold"),
        ).pack(anchor="w", padx=24, pady=(16, 2))
        Label(
            inner,
            text="Self-contained to this run only -- picking a dataset or changing account/risk "
            "settings here never touches Steps 02-04 or any other tab.",
            bg=BG, fg=TEXT_DIM, font=_safe_font(8), wraplength=660, justify="left",
        ).pack(anchor="w", padx=24, pady=(0, 8))

        self.panel = RunContextPanel(owner, action_label)
        self.panel.build(inner)

        btn_row = Frame(self, bg=BG)
        btn_row.pack(fill="x", padx=24, pady=10)
        owner._button(btn_row, "RUN", self._confirm, primary=True).pack(side="right")
        owner._button(btn_row, "CANCEL", self._cancel).pack(side="right", padx=8)

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _confirm(self):
        if not self.panel.csv_paths:
            messagebox.showwarning("Missing data", "Select or fetch a market data file above first.")
            return
        self.confirmed = True
        self.destroy()

    def _cancel(self):
        self.confirmed = False
        self.destroy()

    @classmethod
    def ask(cls, parent, owner: "MainWindow", action_label: str) -> "RunContextPanel | None":
        dialog = cls(parent, owner, action_label)
        parent.wait_window(dialog)
        return dialog.panel if dialog.confirmed else None
