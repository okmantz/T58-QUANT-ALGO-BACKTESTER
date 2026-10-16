"""Hedge-fund manager tab."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import os
import threading
import traceback
from app.data.importer import import_csv
from app.data.storage import list_stored_datasets, store_csv_path
from tkinter import END, Frame, Label, Text, filedialog, messagebox, ttk
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui.widgets.cards import NeuralProgress
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
BORDER = ""
TEXT = ""
TEXT_MUTED = ""
LOG_BG = ""
theme.bind(globals())


class HedgeFundMixin:
    """Hedge-fund manager tab."""


    # -----------------------------------------------------------------------
    # Hedge Fund Manager -- desktop parity for the web app's /hedge-fund.
    # Self-contained like every RunContextPanel-based tab: its own asset
    # slots (each an independent dataset pick, nothing shared with Steps
    # 02-04 or any other tab) and its own rebalance/forecast settings.
    # Thin wrapper over app.hedge_fund.pipeline.run_hedge_fund_manager --
    # same entry point app.web.hedge_fund_routes uses, so a book run here
    # and a book run on the web app are scored identically.
    # -----------------------------------------------------------------------

    _HF_MAX_ASSETS = 6

    def _build_hedge_fund_tab(self):
        f = self._scrollable(self.tab_hedge_fund)

        self._page_header(
            f,
            "CHAMPION / Hedge Fund Manager",
            "\U0001F3E6 Hedge Fund Manager",
            "Research -> Portfolio -> Execution -> Oversight over a multi-asset book: each "
            "asset gets an ensemble return forecast (bootstrap, or -- if you point a slot at "
            "a saved Strategy Library entry -- that strategy's own signal), the forecasts "
            "become Black-Litterman views, and the book rebalances on a fixed cadence with "
            "transaction costs and a turnover cap. Needs at least 2 assets. Every setting "
            "on this page is self-contained to this tab, same as the web app's version.",
        )

        assets_section = self._section(
            f, "Book assets (pick at least 2)",
            "Each slot is independent -- its own dataset pick and its own optional label. "
            "Leave a slot's dataset on '(none)' to skip it.",
            emphasize=True,
        )

        self._hf_asset_rows = []
        for i in range(1, self._HF_MAX_ASSETS + 1):
            row = Frame(assets_section, bg=PANEL)
            row.pack(fill="x", padx=18, pady=4)
            Label(row, text=f"Asset {i}", width=8, anchor="w", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9)).pack(side="left")
            dataset_combo = LabeledCombo(row, "Dataset", ["(none)"], default="(none)")
            label_entry = LabeledEntry(row, "Label (optional)", "", width=14)
            import_btn = self._button(row, "IMPORT CSV...", lambda idx=i: self._hf_import_csv_for_slot(idx))
            import_btn.pack(side="left", padx=(6, 0))
            self._hf_asset_rows.append({"combo": dataset_combo, "label": label_entry})

        refresh_row = Frame(assets_section, bg=PANEL)
        refresh_row.pack(anchor="w", padx=18, pady=(2, 12))
        self._button(refresh_row, "REFRESH DATASET LISTS", self._hf_refresh_dataset_combos).pack(side="left")
        self._hf_refresh_dataset_combos()

        config_section = self._section(
            f, "Rebalance & forecast settings",
            "Own copy of book-level settings -- nothing here is read from any other tab.",
        )
        self.hf_initial_balance = LabeledEntry(config_section, "Initial balance ($)", 100000)
        self.hf_rebalance_every_bars = LabeledEntry(config_section, "Rebalance every N bars", 5)
        self.hf_lookback_bars = LabeledEntry(config_section, "Forecast lookback (bars)", 100)
        self.hf_horizon_bars = LabeledEntry(config_section, "Forecast horizon (bars)", 5)
        self.hf_n_samples = LabeledEntry(config_section, "Forecast samples", 32)
        self.hf_confidence = LabeledEntry(config_section, "Black-Litterman confidence (0-1)", 0.5)
        self.hf_long_only = LabeledCheckbox(config_section, "Long-only (no short positions)", True)
        self.hf_max_turnover = LabeledEntry(config_section, "Max turnover per cycle (blank=uncapped)", 0.5)
        self.hf_transaction_cost_bps = LabeledEntry(config_section, "Transaction cost (bps each way)", 10.0)
        self.hf_min_trade_frac = LabeledEntry(config_section, "Min trade size (fraction of equity, skips dust)", 0.005)
        self.hf_write_journal = LabeledCheckbox(config_section, "Write an AI research-desk journal for this run", True)

        button_row = Frame(f, bg=BG)
        button_row.pack(fill="x", padx=24, pady=10)
        self._button(button_row, "RUN HEDGE FUND MANAGER", self._hf_run_clicked, primary=True).pack(side="left")

        self.hf_progress = NeuralProgress(f)
        self.hf_progress.pack(fill="x", padx=24, pady=(2, 10))

        output_section = self._section(f, "Output", "Stats, journal, and per-cycle detail from the most recent run.")
        _hf_output_frame = Frame(output_section, bg=PANEL)
        self.hf_output = Text(
            _hf_output_frame, height=22, wrap="word", bg=LOG_BG, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        _hf_output_scroll = ttk.Scrollbar(_hf_output_frame, orient="vertical", command=self.hf_output.yview, style="T58.Vertical.TScrollbar")
        self.hf_output.configure(yscrollcommand=_hf_output_scroll.set)
        self.hf_output.pack(side="left", fill="both", expand=True)
        _hf_output_scroll.pack(side="right", fill="y")
        _hf_output_frame.pack(fill="both", expand=True, padx=18, pady=(3, 16))
        self._bind_isolated_wheel(self.hf_output)

    def _hf_refresh_dataset_combos(self):
        names = ["(none)"] + [ds.name for ds in list_stored_datasets()]
        for row in getattr(self, "_hf_asset_rows", []):
            row["combo"].set_options(names)

    def _hf_import_csv_for_slot(self, slot_index: int):
        paths = filedialog.askopenfilenames(
            filetypes=[("Market data", "*.csv *.tsv *.txt *.parquet *.zip *.7z"), ("All files", "*.*")]
        )
        if not paths:
            return
        result = import_csv(paths[0])
        if not result.is_valid:
            messagebox.showerror("Import failed", f"{os.path.basename(paths[0])}:\n" + "\n".join(result.errors))
            return
        stored_path = store_csv_path(paths[0])
        self._hf_refresh_dataset_combos()
        self._hf_asset_rows[slot_index - 1]["combo"].set_str(stored_path.name)

    def _hf_log(self, msg: str):
        self.hf_output.insert(END, msg + "\n")
        self.hf_output.see(END)
        self.root.update_idletasks()

    def _hf_run_clicked(self):
        price_data: dict[str, "pd.DataFrame"] = {}
        for i, row in enumerate(self._hf_asset_rows, start=1):
            name = row["combo"].get_str()
            if not name or name == "(none)":
                continue
            match = next((ds for ds in list_stored_datasets() if ds.name == name), None)
            if match is None:
                continue
            result = import_csv(match.path)
            if not result.is_valid:
                messagebox.showerror("Import failed", f"{name}:\n" + "\n".join(result.errors))
                return
            label = row["label"].get_str().strip() or name
            if label in price_data:
                label = f"{label} ({i})"
            price_data[label] = result.dataframe

        if len(price_data) < 2:
            messagebox.showwarning(
                "Not enough assets",
                "A hedge fund book needs at least 2 assets -- pick a dataset for at least 2 of the asset slots above.",
            )
            return

        self.hf_output.delete("1.0", END)
        self.hf_progress.start(10)
        threading.Thread(target=self._hf_run_pipeline, args=(price_data,), daemon=True).start()

    def _hf_run_pipeline(self, price_data):
        from app.hedge_fund.pipeline import HedgeFundManagerError, run_hedge_fund_manager
        from app.hedge_fund.rebalancer import RebalanceConfig
        from app.hedge_fund.research import EnsembleForecastConfig

        try:
            max_turnover_raw = self.hf_max_turnover.get_str().strip()
            forecast_config = EnsembleForecastConfig(
                lookback_bars=self.hf_lookback_bars.get_int(100),
                horizon_bars=self.hf_horizon_bars.get_int(5),
                n_samples=self.hf_n_samples.get_int(32),
            )
            config = RebalanceConfig(
                initial_balance=self.hf_initial_balance.get_float(100000),
                rebalance_every_bars=self.hf_rebalance_every_bars.get_int(5),
                forecast=forecast_config,
                confidence=self.hf_confidence.get_float(0.5),
                long_only=self.hf_long_only.get(),
                max_turnover=(float(max_turnover_raw) if max_turnover_raw else None),
                transaction_cost_bps=self.hf_transaction_cost_bps.get_float(10.0),
                min_trade_frac=self.hf_min_trade_frac.get_float(0.005),
            )

            self._hf_log(f"Running Hedge Fund Manager over {len(price_data)} assets: {', '.join(price_data.keys())}\n")
            outcome = run_hedge_fund_manager(price_data, config, write_journal=self.hf_write_journal.get())

            stats = outcome.backtest.stats
            self._hf_log(
                f"Total return: {stats.total_return_pct:.2f}%   |   CAGR: {stats.cagr_pct:.2f}%   |   "
                f"Max drawdown: {stats.max_drawdown_pct:.2f}%   |   Sharpe: {stats.sharpe_ratio:.2f}"
            )
            self._hf_log(
                f"Rebalances: {stats.num_rebalances}   |   Avg turnover: {stats.avg_turnover_pct:.2f}%   |   "
                f"Total transaction costs: ${stats.total_transaction_costs:,.2f}"
            )
            equity = outcome.backtest.equity_curve["equity"]
            self._hf_log(f"Equity: ${equity.iloc[0]:,.2f} -> ${equity.iloc[-1]:,.2f}\n")

            for w in outcome.backtest.warnings:
                self._hf_log(f"WARNING: {w}")

            if outcome.backtest.cycles:
                last_cycle = outcome.backtest.cycles[-1]
                self._hf_log("Most recent cycle's target weights:")
                for asset, weight in last_cycle.target_weights.items():
                    self._hf_log(f"  {asset}: {weight:.1%}")

            if outcome.journal:
                self._hf_log("\n--- Research desk journal ---")
                self._hf_log(outcome.journal)
                if outcome.journal_note:
                    self._hf_log(f"({outcome.journal_note})")
        except HedgeFundManagerError as exc:
            self._hf_log(f"\nHedge Fund Manager error: {exc}")
        except Exception:
            self._hf_log("\nUnexpected error:\n" + traceback.format_exc())
        finally:
            self.hf_progress.stop()
