"""
T58 Quant Algo Backtester — Desktop GUI.

Dark T58 interface with a full visual Manual Strategy Builder. The builder
creates a structured JSON-like strategy configuration consumed by
app.strategy.manual.ManualStrategy; no strategy code is required.

The original data, strategy-file, prop-rules, risk, run, Monte Carlo and
report workflows are preserved.
"""
# ruff: noqa: F401
# NOTE: this module used to hold the whole desktop UI (23k lines). The code now lives in app/ui/theme.py,
# app/ui/common.py, app/ui/platform_win.py, app/ui/widgets/ and app/ui/mixins/. Every name that used to be
# importable from here (colors, widgets, helper functions, third-party names other modules read as
# ``mw.<name>``) is re-exported below, so existing imports keep working.
from __future__ import annotations


import dataclasses
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
import webbrowser
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, Text, END,
    filedialog, messagebox, simpledialog, ttk, Listbox, SINGLE, EXTENDED, BooleanVar, Canvas,
    Checkbutton, PhotoImage, Toplevel,
)

# Optional OS drag-and-drop for market-data folders: tkinterdnd2 wires Tk
# up as a native file-drop target, but it is NOT a hard dependency --
# when it isn't installed the DnD setup (_maybe_enable_folder_dnd) is a
# silent no-op and the IMPORT FOLDER buttons remain the way in.
try:
    import tkinterdnd2
except ImportError:  # pragma: no cover - optional dependency
    tkinterdnd2 = None

import app.evolution.checkpoint as evo_checkpoint
from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskError, AdaptiveRiskRule
from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig, build_run_context, suggest_pip_size, with_prop_safety_defaults
from app.data.instrument_specs import known_instrument_symbols, get_instrument_spec
from app.backtest.statistics import net_profit_reset_note
import app.ai.ollama_settings as ollama_settings_module
from app.ai.ollama_settings import OllamaSettings
import app.ai.strategy_generator as strategy_generator_module
import app.ai.research_library as research_library_module
import app.ai.experiment_memory as experiment_memory_module
from app.ai.research_agent import ResearchAgentContext, ResearchAgent
from app.ai.research_loop import ResearchLoopConfig, run_research_loop
from app.data import alpaca_credentials
from app.data.alpaca_source import (
    ASSET_CLASSES, ADJUSTMENT_CHOICES, FEED_CHOICES, TIMEFRAME_LABELS,
    AlpacaFetchError, AlpacaImportError, fetch_bars, save_bars_as_csv, test_connection,
)
from app.data.london_strategic_edge_source import (
    TIMEFRAME_CHOICES as LSE_TIMEFRAME_CHOICES,
    LSEFetchError, LSEImportError,
    fetch_candles as lse_fetch_candles, save_bars_as_csv as lse_save_bars_as_csv,
    test_connection as lse_test_connection,
)
from app.accounts.api_keys import load_settings as load_apikeys_settings, save_settings as save_apikeys_settings
from app.data.importer import import_csv
from app.data.multi_timeframe import merge_multi_timeframe
from app.data.timeframe_resample import infer_timeframe_label as _infer_timeframe_label
from app.data.pairs import PairDataError, merge_pair_series
from app.data.storage import EMPTY_DATASET_BYTES, get_raw_data_dir, list_datasets_by_instrument, list_stored_datasets, store_csv_path
from app.ensemble.ensemble import EnsembleError, EnsembleVoteConfig, run_ensemble_blend, run_ensemble_vote
import app.forward_test.mt5_settings as mt5_settings_module
from app.forward_test.mt5_settings import MT5Settings
from app.forward_test import mt5_connector as mt5_connector_module
from app.forward_test.journal import ForwardTestJournal
from app.forward_test.engine import ForwardTestConfig, ForwardTestSession
from app.live_deploy import prop_firms as live_deploy_prop_firms
from app.live_deploy import live_settings as live_deploy_settings
from app.live_deploy import connect_flow as live_connect_flow
from app.live_deploy.broker_registry import build_adapter, SUPPORTED_PLATFORMS
from app.live_deploy.execution_engine import LiveExecutionSession, LiveExecutionConfig
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from app.monte_carlo.bankroll import BankrollConfig, BankrollSurvivalResult, simulate_bankroll_survival
from app.optimize.parameter_space import RefinementError, apply_genome, extract_genome
from app.optimize.refinement import FITNESS_METRICS, OPTIMIZER_MODES, RefinementConfig, run_iterative_refinement
from app.optimize.multi_objective import DEFAULT_OBJECTIVES, MultiObjectiveConfig, OBJECTIVE_DIRECTIONS, run_multi_objective_refinement
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.orchestration import pipeline_guide
from app.orchestration.batch_test import BatchTestItem, run_batch_test
from app.orchestration.forge import ForgeConfig, run_forge
from app.orchestration.loop_runner import (
    ForgeLoopConfig, SearchLoopConfig, SpeedRunLoopConfig, run_forge_loop, run_search_loop, run_speed_run_loop,
)
from app.orchestration.resource_guard import (
    HEAVY_JOB_GUARD, JOB_EVOLUTION_LAB, JOB_FORGE, JOB_FULL_PIPELINE, JOB_SEARCH_LAB, JOB_SPEED_RUN,
    JOB_WFO, JOB_WFGA, JOB_CPCV, JOB_SENSITIVITY, JOB_MULTI_OBJECTIVE, JOB_REGIME_MATRIX,
    JOB_PARAMETER_ROBUSTNESS, JOB_MULTI_INSTRUMENT_EVOLUTION, JOB_MULTI_INSTRUMENT_SEARCH,
)
from app.orchestration.speed_run import SpeedRunConfig, SpeedRunResult, run_speed_run
from app.orchestration.speed_run import _rank_key as _speedrun_rank_key
from app.orchestration.multi_instrument_search import InstrumentJob
from app.orchestration.multi_instrument_speed_run import (
    MultiInstrumentSpeedRunResult,
    best_speed_run_across_instruments,
    run_multi_instrument_speed_run,
)
from app.orchestration.overnight_autopilot import AutopilotConfig, run_overnight_autopilot
from app.orchestration.auto_retune import maybe_trigger_retune
from app.orchestration.full_pipeline import (
    FullPipelineBatchItem, FullPipelineConfig, run_full_pipeline, run_full_pipeline_batch,
)
from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig, PortfolioError, run_portfolio_backtest
from app.prop.simulator import PropRules, simulate_account
from app.prop.presets import list_presets as list_prop_firm_presets
from app.prop.recommender import recommend_prop_firms, render_recommendation_table
from app.prop.scaling import ScalingPlan, run_scaling_stress_test
from app.prop.survival_engine import PropSurvivalConfig, ResetEconomics, run_prop_survival_analysis
from app.reports.generator import generate_full_report
from app.reports.crash_log import log_crash
from app.reports import run_history
from app.reports import strategy_state
from app.scoring import champion_board
from app.reports.refinement_report import generate_refinement_report
from app.reports.survival_report import generate_survival_report
from app.reports.validation_reports import (
    generate_cpcv_report, generate_multi_objective_report, generate_pbo_report,
    generate_portfolio_report, generate_sensitivity_report, generate_walk_forward_report,
    generate_walkforward_ga_report,
)
from app.scoring.leaderboard import build_leaderboard, render_leaderboard_table
from app.search.batch_runner import SearchCancelled, SearchStageConfig, promote_champion, run_search
from app.search.global_search import global_search
from app.search.graveyard import graveyard_path_for, list_graveyard_files, load_graveyard, render_graveyard_report, summarize_graveyard
from app.search.search_report import generate_search_report
from app.search.strategy_space import (
    StrategySpaceError, generate_search_space, hypothesis_question, list_families,
)
from app.research import director as research_director
from app.strategy.base import StrategyError
from app.strategy.dna import extract_dna, find_common_patterns
from app.strategy.library import (
    STRATEGY_STATUSES, STRATEGY_TYPES, StrategyAlreadyExists, delete_many,
    delete_saved_strategy, export_library_zip, get_strategy_library_dir,
    list_all_markets, list_all_tags, list_misplaced_files, list_saved_strategies,
    load_strategy_text, record_backtest_result, record_lookahead_result, record_search_result,
    record_optimize_result, record_validation_result, record_champion_check_result,
    compute_pipeline_progress,
    rename_saved_strategy, save_strategy_bytes, save_strategy_metadata,
    save_strategy_path, save_strategy_text, set_strategy_status, set_strategy_tags, status_label,
)
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy
from app.strategy.translator import SUPPORTED_LANGUAGES as TRANSLATOR_LANGUAGES, TranslationError, translate_strategy
from app.strategy.auto_regime_selector import select_regime_strategies
from app.strategy.library_loader import load_validated_candidates, load_strategy_object
from app.monitoring.strategy_health import check_strategy_health
from app.portfolio.composer import compose_portfolio
from app.quant_lab.order_book import LimitOrderBook
from app.quant_lab.options_pricing import black_scholes_greeks, black_scholes_price, compare_to_market
from app.quant_lab.pairs_trading import fetch_universe, run_pairs_backtest, screen_pairs
from app.quant_lab.portfolio_optimizer import fetch_and_build_inputs, optimize_for_risk_level
from app.quant_lab.sentiment_price import (
    aggregate_sentiment, correlate_sentiment_with_price, fetch_headlines, score_headlines,
)
from app.quant_lab.vol_surface import build_iv_surface, export_surface_html, fetch_option_chain, synthetic_demo_chain
from app.quant_lab.factor_model import compute_factor_exposures, compute_returns_from_prices, fetch_fama_french_factors
from app.quant_lab.market_structure import calculate_hh_ll_structure, calculate_wyckoff_events, summarize_market_structure
from app.strategy.composite_thresholds import cross_level, hold_level, in_range, mix_thresholds, preview_signal
from app.ui.condition_builder import ConditionList
from app.validation.cpcv import CPCVError, compute_pbo, run_cpcv
from app.ensemble.auto_builder import AutoEnsembleError, build_diversified_ensemble
from app.validation.integrity_check import run_integrity_check
from app.validation.regime_matrix import run_regime_matrix
from app.validation.sensitivity import compute_1d_sensitivity, compute_2d_heatmap, list_tunable_parameters
from app.validation.parameter_robustness import compute_parameter_robustness
from app.validation.walk_forward_opt import run_walk_forward_optimization
from app.web import live_market
import random

from app.ui import theme
from app.ui.theme import THEMES, apply_theme, _load_theme_name, _save_theme_name, _THEME_SETTINGS_FILENAME, FONT, MONO, SIDEBAR_WIDTH

# Palette names (kept current by theme.bind / apply_theme).
BG = ""
PANEL = ""
SIDEBAR = ""
PANEL_2 = ""
PANEL_3 = ""
PANEL_HOVER = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
METAL = ""
METAL_BRIGHT = ""
LOG_BG = ""
GREEN = ""
RED = ""
BLUE = ""
AMBER = ""
ACCENT = ""
ACCENT_HOVER = ""
ACCENT_DIM = ""
ACCENT_INK = ""
VIOLET = ""
NEON_CYAN = ""
NEON_MAGENTA = ""
NEON_LIME = ""
NEON_VIOLET = ""
NEON_AMBER = ""
CURRENT_THEME = "dark"
theme.bind(globals())

from app.ui.common import (
    DEFAULT_MANUAL_STRATEGY,
    OUTPUT_DIR,
    STATUS_KEY_TO_LABEL,
    STATUS_LABELS_ORDERED,
    STATUS_LABEL_TO_KEY,
    USE_CUSTOM_TITLEBAR,
    _ADAPTIVE_RISK_TRIGGERS,
    _ADAPTIVE_RISK_TRIGGER_HELP,
    _asset_path,
    _safe_font,
    _strategy_display_name,
)
from app.ui.platform_win import (
    _apply_dark_titlebar,
    _apply_tk_scaling,
    _force_dwm_composition,
    _get_work_area,
    _hex_to_colorref,
    _keep_taskbar_icon,
    _make_dpi_aware,
    _patch_messagebox_for_overrideredirect_focus,
)
from app.ui.theme import (
    FONT,
    MONO,
    SIDEBAR_WIDTH,
    THEMES,
    _THEME_SETTINGS_FILENAME,
    _load_theme_name,
    _save_theme_name,
    apply_theme,
)
from app.ui.widgets.basic import (
    LabeledCheckbox,
    LabeledCombo,
    LabeledEntry,
    _blend_hex,
)
from app.ui.widgets.cards import (
    GlowCard,
    NeuralProgress,
    RingProgress,
    RoundedCard,
    _card,
)
from app.ui.widgets.dialogs import (
    _AdaptiveRuleDialog,
    _ChoiceListDialog,
)
from app.ui.widgets.run_context import (
    RunContextDialog,
    RunContextPanel,
)
from app.ui.mixins.account import AccountMixin
from app.ui.mixins.ai_assistant import AiAssistantMixin
from app.ui.mixins.api_keys_support import ApiKeysSupportMixin
from app.ui.mixins.chrome import ChromeMixin
from app.ui.mixins.dashboard import DashboardMixin
from app.ui.mixins.data_tab import DataTabMixin
from app.ui.mixins.deploy_live import DeployLiveMixin
from app.ui.mixins.discovery_tabs import DiscoveryTabsMixin
from app.ui.mixins.evolution_lab import EvolutionLabMixin
from app.ui.mixins.forge import ForgeMixin
from app.ui.mixins.forward_test import ForwardTestMixin
from app.ui.mixins.hedge_fund import HedgeFundMixin
from app.ui.mixins.helpers import HelpersMixin
from app.ui.mixins.learn import LearnMixin
from app.ui.mixins.library_batch import LibraryBatchMixin
from app.ui.mixins.live_market import LiveMarketMixin
from app.ui.mixins.manual_builder import ManualBuilderMixin
from app.ui.mixins.navigation import NavigationMixin
from app.ui.mixins.payout import PayoutMixin
from app.ui.mixins.pipelines import PipelinesMixin
from app.ui.mixins.portfolio_tabs import PortfolioTabsMixin
from app.ui.mixins.prop_risk import PropRiskMixin
from app.ui.mixins.quant_lab import QuantLabMixin
from app.ui.mixins.refine_run import RefineRunMixin
from app.ui.mixins.research import ResearchMixin
from app.ui.mixins.search_lab import SearchLabMixin
from app.ui.mixins.speed_run import SpeedRunMixin
from app.ui.mixins.strategy_config import StrategyConfigMixin
from app.ui.mixins.strategy_library import StrategyLibraryMixin
from app.ui.mixins.validate_hub import ValidateHubMixin
from app.ui.mixins.validation_labs import ValidationLabsMixin


class MainWindow(
    AccountMixin,
    AiAssistantMixin,
    ApiKeysSupportMixin,
    ChromeMixin,
    DashboardMixin,
    DataTabMixin,
    DeployLiveMixin,
    DiscoveryTabsMixin,
    EvolutionLabMixin,
    ForgeMixin,
    ForwardTestMixin,
    HedgeFundMixin,
    HelpersMixin,
    LearnMixin,
    LibraryBatchMixin,
    LiveMarketMixin,
    ManualBuilderMixin,
    NavigationMixin,
    PayoutMixin,
    PipelinesMixin,
    PortfolioTabsMixin,
    PropRiskMixin,
    QuantLabMixin,
    RefineRunMixin,
    ResearchMixin,
    SearchLabMixin,
    SpeedRunMixin,
    StrategyConfigMixin,
    StrategyLibraryMixin,
    ValidateHubMixin,
    ValidationLabsMixin,
):
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("T58 Trading — Quant Algo Backtester")
        try:
            icon_path = _asset_path("t58_mark_medium.png")
            if icon_path.exists():
                self._icon_image = PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, self._icon_image)
        except Exception:
            pass
        # Web-parity layout has a 240px sidebar (like the web app's), so the
        # old 1000x760 default left only ~760px for content and clipped the
        # dashboard cards -- open larger, but never bigger than the screen.
        try:
            _sw, _sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            _w = min(1440, max(1100, _sw - 120))
            _h = min(900, max(700, _sh - 140))
        except Exception:
            _w, _h = 1280, 820
        self.root.geometry(f"{_w}x{_h}")
        self.root.minsize(1000, 680)

        # UPGRADE (Sep 2026, "ugly blue banner" fix, final attempt): three
        # earlier rounds of _apply_dark_titlebar() tried to recolor
        # Windows' own native title bar via DWM and never actually stuck
        # on this machine, despite running with no error every time --
        # some Windows builds just don't honor DWMWA_USE_IMMERSIVE_DARK_MODE
        # reliably, and there's no way to detect that failure from here to
        # fall back cleanly. Removing the native title bar entirely and
        # drawing this app's own (see _build_custom_titlebar) sidesteps the
        # whole problem: there is no native caption left for Windows to
        # paint blue. USE_CUSTOM_TITLEBAR is a single kill-switch (module
        # constant, top of file) in case this ever needs to be turned off
        # on a machine where it misbehaves -- everything downstream checks
        # self._custom_titlebar_active, never the raw platform check alone.
        self._custom_titlebar_active = False
        if USE_CUSTOM_TITLEBAR and sys.platform == "win32":
            try:
                self.root.overrideredirect(True)
                _keep_taskbar_icon(self.root)
                self._custom_titlebar_active = True
            except Exception:
                self._custom_titlebar_active = False
        self._drag_offset: tuple[int, int] | None = None
        self._maximized = False
        self._normal_geometry: str | None = None

        # State that must survive a theme toggle's full widget rebuild
        # (see _toggle_theme) lives directly on self, set once here --
        # _build_ui() only ever CREATES widgets, it never resets this.
        self.csv_path: str | None = None
        self.csv_paths: list[str] = []
        self.strategy_py_path: str | None = None
        self._active_library_strategy: tuple[str, str] | None = None
        # Checkbox-style batch selection (mirrors the web app's flat
        # checkbox list of every saved strategy across all types) --
        # keyed by (strategy_type, name) -> tk.BooleanVar so the checked
        # state survives a REFRESH LIST re-population. RUN BATCH TEST
        # (CHECKED) and RUN FULL PIPELINE (BATCH) (CHECKED) both read
        # straight off whichever boxes are checked at click time --
        # there's no separate "add to queue" step to remember.
        self._batch_checked_vars: dict[tuple[str, str], "BooleanVar"] = {}
        self._batch_checklist_items: dict[tuple[str, str], object] = {}
        self.strategy_mode = StringVar(value="manual")

        self._build_ui()


def launch(root: "Tk | None" = None, splash=None):
    """Starts the desktop app. `root`/`splash` are passed by app.main so the
    startup splash is already on screen (and stays up, with live status)
    until this window is fully built; both default to None for callers that
    just want the app (tests, other entry points)."""
    _make_dpi_aware()
    if root is None:
        root = Tk()
    if splash is not None:
        root._t58_splash = splash  # consumed by MainWindow._pump_splash
    # Stability layer (thread-safe widget updates, bounded log widgets, crash capture)
    # -- installed before anything else touches Tk. See app/ui/tk_safety.py.
    try:
        from app.ui import tk_safety
        tk_safety.install(root)
    except Exception:
        pass
    _apply_tk_scaling(root)
    _force_dwm_composition(root)
    root.withdraw()  # hidden while the real window builds -- avoids showing a half-built window

    window = MainWindow(root)
    if splash is not None:
        splash.close()
        root._t58_splash = None

    root.deiconify()
    try:
        root.lift()
        root.focus_force()
    except Exception:
        pass
    # First launch: the person lands on the Dashboard and the guided tour
    # starts immediately (shown once per install; replay it any time with
    # "Take the tour" on the Dashboard).
    try:
        from app.ui.onboarding_tour import maybe_start_tour
        root.after(250, lambda: maybe_start_tour(window))
    except Exception:
        pass
    # UPGRADE (Sep 2026 UI pass, round 3): this used to run BEFORE
    # root.withdraw() above -- i.e. on a window that wasn't even visible
    # yet, then got hidden immediately after. DWM had nothing on screen to
    # recolor at that point, and deiconify() later doesn't retroactively
    # apply it. Recoloring the title bar only works reliably once the
    # window is actually shown, so this now runs after deiconify()/lift(),
    # on the real, visible window -- that's what was actually behind the
    # title bar staying Windows' default blue despite this code running
    # with no error.
    try:
        _apply_dark_titlebar(root)
    except Exception:
        pass
    root.mainloop()
