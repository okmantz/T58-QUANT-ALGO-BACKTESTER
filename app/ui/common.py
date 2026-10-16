"""Shared constants and small helpers used across the desktop UI (extracted from main_window.py)."""
from __future__ import annotations

import sys
from app.strategy.library import STRATEGY_STATUSES, status_label
from pathlib import Path
from app.ui.theme import FONT

# Display <-> storage-slug maps for the strategy status lifecycle (draft,
# tested_failed, tested_passed, validated, ready_for_demo, ready_for_live)
# -- built once from STRATEGY_STATUSES so every dropdown/listbox/badge
# shows the same human-readable label ("TESTED / FAILED") while the file
# on disk keeps the plain slug ("tested_failed").
STATUS_KEY_TO_LABEL: dict[str, str] = {s: status_label(s) for s in STRATEGY_STATUSES}


STATUS_LABEL_TO_KEY: dict[str, str] = {v: k for k, v in STATUS_KEY_TO_LABEL.items()}


STATUS_LABELS_ORDERED: list[str] = [STATUS_KEY_TO_LABEL[s] for s in STRATEGY_STATUSES]


OUTPUT_DIR = Path.cwd() / "reports"


def _strategy_display_name(strategy) -> str:
    if strategy.source_type == "manual":
        return strategy.config.get("name", "Manual Strategy")
    if strategy.source_type == "python":
        return Path(strategy.file_path).stem
    if strategy.source_type == "pinescript":
        return "PineScript Strategy"
    if strategy.source_type == "mql5":
        return "MQL5 Strategy"
    return "Strategy"


DEFAULT_MANUAL_STRATEGY = {
    "name": "SMA 20/50 Cross",
    "description": "Trend-following moving-average crossover strategy.",
    "author": "",
    "version": "1.0",
    "market": {"instrument": "", "timeframe": "5m", "session": "All", "direction": "Both"},
    "indicators": [
        {"type": "sma", "period": 20, "column": "close", "as": "sma_fast"},
        {"type": "sma", "period": 50, "column": "close", "as": "sma_slow"},
    ],
    "long_entry": "sma_fast > sma_slow",
    "long_exit": "sma_fast < sma_slow",
    "short_entry": "sma_fast < sma_slow",
    "short_exit": "sma_fast > sma_slow",
    "stop_loss_pips": 20,
    "take_profit_pips": 40,
}


# Kill-switch for the custom borderless title bar (see MainWindow.__init__
# and _build_custom_titlebar) -- flip to False to fall back to the native
# Windows title bar + the older DWM-recolor attempt (_apply_dark_titlebar)
# if the custom one ever misbehaves on a particular machine/Windows build.
USE_CUSTOM_TITLEBAR = True


# NOTE: the condition-row vocabulary (sources/operators/kind mapping) used
# to live here, but now lives in app.ui.condition_builder alongside the
# widget that uses it, so there's a single source of truth.


def _asset_path(filename: str) -> Path:
    """Resolves a bundled UI asset both in dev mode and inside a
    PyInstaller-frozen .exe (where files added via --add-data land under
    sys._MEIPASS instead of next to this source file)."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / "app" / "ui" / "assets" / filename
    return Path(__file__).resolve().parent / "assets" / filename


def _safe_font(size=10, weight="normal"):
    return (FONT, size, weight)


_ADAPTIVE_RISK_TRIGGERS = [
    "consecutive_losses", "daily_loss_pct", "daily_profit_pct", "progress_to_target_pct", "drawdown_pct",
]


_ADAPTIVE_RISK_TRIGGER_HELP = {
    "consecutive_losses": "threshold = number of losing trades in a row (today)",
    "daily_loss_pct": "threshold = % of initial balance realized-lost so far today",
    "daily_profit_pct": "threshold = % of initial balance realized-gained so far today",
    "progress_to_target_pct": "threshold = % of the way from 0 to the profit target, all-time",
    "drawdown_pct": "threshold = % of initial balance down from the account's own realized-equity peak, all-time",
}
