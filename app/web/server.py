"""
T58 Quant Algo Backtester — Mobile Web App.

A thin Flask front end over the exact same engine used by the desktop GUI
and the --cli entry point (app/backtest, app/prop, app/monte_carlo,
app/strategy, app/reports) -- no logic is duplicated here.

This is what makes the tool usable "as an app" on a phone: any phone
browser on the same network as the machine running this server can open
it, and "Add to Home Screen" installs it as an installable PWA (manifest +
service worker included) with its own icon and a standalone window, no App
Store / Play Store submission required.

Run:
    python -m app.web.server
    -> serves on http://0.0.0.0:5000
    -> on your phone (same Wi-Fi), open http://<your-computer's-LAN-IP>:5000

To make it reachable from anywhere (not just local Wi-Fi), deploy this
Flask app to any small host (Render, Railway, Fly.io, a VPS, etc.) -- see
README.md for notes. Running a persistent Python server directly on a
phone (as opposed to browsing to one) is out of scope for this MVP.
"""
from __future__ import annotations

import json
import os
import random
import re
import tempfile
import threading
from dataclasses import replace
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import (
    Flask, Response, jsonify, redirect, render_template, request, send_from_directory, session, url_for,
)

from app.ai.ollama_settings import OllamaSettings
from app.education import content as education_content
from app.web.network_info import lan_url, print_startup_banner, qr_code_data_uri, qr_code_file, tailscale_url
from app.web.notifications import (
    NotificationSettings, load_notification_settings, notify_job_finished, save_notification_settings,
    send_job_notification, _send_email_notification,
)
from app.web.quant_lab_routes import quant_lab_bp
from app.web.extra_routes import extra_bp
from app.web.ai_assistant_routes import ai_assistant_bp
from app.web.options_outlook_routes import options_outlook_bp
from app.web.hedge_fund_routes import hedge_fund_bp
from app.web.risk_sweep_routes import risk_sweep_bp
from app.ai.ollama_settings import load_settings as load_ollama_settings
from app.ai.ollama_settings import save_settings as save_ollama_settings
from app.ai.research_agent import ResearchAgentContext, ResearchAgent
from app.ai.research_loop import ResearchLoopConfig, ResearchLoopRunner
from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig, suggest_pip_size, with_prop_safety_defaults
from app.data import alpaca_credentials
from app.data.alpaca_source import (
    ASSET_CLASSES, ADJUSTMENT_CHOICES, FEED_CHOICES, TIMEFRAME_LABELS,
    AlpacaFetchError, AlpacaImportError, fetch_bars, save_bars_as_csv,
)
from app.data.london_strategic_edge_source import (
    TIMEFRAME_CHOICES as TIMEFRAME_CHOICES_LSE,
    LSEFetchError, LSEImportError, fetch_candles as lse_fetch_candles,
    save_bars_as_csv as lse_save_bars_as_csv, test_connection as lse_test_connection,
)
from app.data.importer import import_csv, import_csv_bytes
from app.data.timeframe_resample import infer_timeframe_label
from app.web.alpaca_shared import alpaca_template_context
from app.web.lse_shared import lse_template_context
from app.data.storage import get_app_base_dir, get_raw_data_dir, list_datasets_by_instrument, list_stored_datasets, store_csv_bytes
from app.ensemble.auto_builder import AutoEnsembleError, build_diversified_ensemble
from app.ensemble.ensemble import EnsembleError, EnsembleVoteConfig, run_ensemble_blend, run_ensemble_vote
from app.evolution import checkpoint as evo_checkpoint
from app.evolution.engine import EvolutionConfig, EvolutionRunner, evolution_stats_metadata
from app.orchestration import pipeline_guide
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.monte_carlo.bankroll import BankrollConfig, simulate_bankroll_survival
from app.optimize.risk_sweep import DEFAULT_RISK_VALUES, run_risk_sweep
from app.optimize.multi_objective import (    DEFAULT_OBJECTIVES, MultiObjectiveConfig, OBJECTIVE_DIRECTIONS, run_multi_objective_refinement,
    run_multi_objective_sweep,
)
from app.backtest.adaptive_risk import build_limit_aware_preset
from app.optimize.distribution_summary import compute_distribution_summary
from app.optimize.refinement import FITNESS_METRICS, OPTIMIZER_MODES, RefinementConfig, RefinementError, run_iterative_refinement
from app.optimize.multi_market import AGGREGATION_METHODS, run_multi_market_search
from app.optimize.walkforward_ga import WalkforwardGACancelled, run_walkforward_aware_refinement
from app.orchestration.batch_test import BatchTestItem, run_batch_test
from app.orchestration.full_pipeline import (
    FullPipelineBatchCancelled, FullPipelineBatchItem, FullPipelineCancelled, FullPipelineConfig,
    run_full_pipeline, run_full_pipeline_batch,
)
from app.orchestration.quick_optimize import QuickOptimizeConfig, run_quick_optimize, run_quick_optimize_sweep
from app.orchestration.resource_guard import (
    HEAVY_JOB_GUARD, JOB_EVOLUTION_LAB, JOB_FORGE, JOB_FULL_PIPELINE, JOB_SEARCH_LAB, JOB_SPEED_RUN,
    JOB_WFO, JOB_WFGA, JOB_CPCV, JOB_SENSITIVITY, JOB_MULTI_OBJECTIVE, JOB_REGIME_MATRIX, JOB_PBO,
    JOB_PARAMETER_ROBUSTNESS, JOB_MULTI_MARKET,
    JOB_MULTI_INSTRUMENT_SEARCH, JOB_MULTI_INSTRUMENT_SPEED_RUN, JOB_MULTI_INSTRUMENT_EVOLUTION,
)
from app.orchestration.forge import ForgeConfig, run_forge
from app.research import director as research_director
from app.evolution.multi_instrument import EvolutionInstrumentJob, MultiInstrumentEvolutionGroup
from app.data.timeframe_sweep import DEFAULT_SWEEP_TIMEFRAMES, parse_sweep_timeframes
from app.orchestration.multi_timeframe_jobs import (
    describe_skipped, evolution_jobs_from_expansion, expand_dataset_across_timeframes,
    search_jobs_from_expansion,
)
from app.orchestration.multi_instrument_search import (
    InstrumentJob, best_result_across_instruments, run_multi_instrument_search,
    run_multi_instrument_search_loop,
)
from app.orchestration.multi_instrument_speed_run import (
    best_speed_run_across_instruments, run_multi_instrument_speed_run,
)
from app.orchestration.speed_run import SpeedRunConfig, SpeedRunResult, run_speed_run
from app.orchestration.speed_run import _rank_key as _speedrun_rank_key
from app.orchestration.overnight_autopilot import AutopilotConfig, run_overnight_autopilot
from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig, PortfolioError, run_portfolio_backtest
from app.prop.simulator import PropRules, simulate_account
from app.prop.presets import get_preset as get_prop_firm_preset, list_presets as list_prop_firm_presets
from app.prop.recommender import recommend_prop_firms, render_recommendation_table
from app.prop.scaling import ScalingPlan, run_scaling_stress_test
from app.prop.survival_engine import PropSurvivalConfig, ResetEconomics, run_prop_survival_analysis
from app.reports.generator import generate_full_report
from app.reports.crash_log import install_thread_excepthook, log_crash
from app.reports.refinement_report import generate_refinement_report
from app.reports.survival_report import generate_survival_report
from app.reports.validation_reports import (
    generate_cpcv_report, generate_multi_objective_report, generate_pbo_report, generate_portfolio_report,
    generate_sensitivity_report, generate_walk_forward_report, generate_walkforward_ga_report,
)
from app.reports import run_history
from app.reports import strategy_state
from app.scoring.t58_scorecard import score_from_results
from app.search.batch_runner import SearchCancelled, SearchStageConfig, promote_champion, run_search
from app.orchestration.loop_runner import (
    ForgeLoopConfig, SearchLoopConfig, SpeedRunLoopConfig, run_forge_loop, run_search_loop, run_speed_run_loop,
)
from app.orchestration.prop_autotune import suggest_from_prop_rules
from app.search.family_diversity import render_family_report, summarize_family_performance
from app.search.graveyard import (
    graveyard_path_for, list_graveyard_files, load_graveyard, render_graveyard_report, summarize_graveyard,
)
from app.search.search_report import generate_search_report
from app.search.strategy_space import (
    StrategySpaceError, family_description, generate_search_space, hypothesis_question, list_families,
    spec_from_strategy,
)
from app.search.results_db import ResultsDB
from app.strategy.base import StrategyError
from app.data.timeframe_resample import prepare_timeframe_aligned_data, describe_resolved_timeframe
from app.validation.cpcv import CPCVError, compute_pbo, run_cpcv
from app.validation.sensitivity import compute_2d_heatmap
from app.optimize.parameter_space import apply_genome, extract_genome
from app.optimize.code_parameter_space import apply_code_genome, discover_code_genes
from app.strategy.library_loader import load_strategy_object
from app.validation.parameter_robustness import compute_parameter_robustness
from app.validation.regime_matrix import run_regime_matrix
from app.validation.sensitivity import compute_1d_sensitivity
from app.validation.walk_forward_opt import run_walk_forward_optimization
from app.strategy.library import (
    STRATEGY_STATUSES, STRATEGY_TYPES, StrategyAlreadyExists, delete_many,
    delete_saved_strategy, export_library_zip_bytes, list_all_markets, list_all_tags,
    list_saved_strategies, load_strategy_text, record_backtest_result, record_lookahead_result,
    record_search_result, record_optimize_result, record_validation_result, record_champion_check_result,
    rename_saved_strategy, save_strategy_bytes, save_strategy_metadata,
    save_strategy_text, set_strategy_status, set_strategy_tags,
)
from app.strategy.lookahead_check import check_for_lookahead
from app.validation.integrity_check import run_integrity_check
from app.strategy.manual import ManualStrategy
from app.scoring import champion_board
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy
from app.web import live_market

# get_app_base_dir() already knows how to find a persistent, writable
# folder next to the running .exe when frozen (see app/data/storage.py),
# vs. the repo root during normal development. Reusing it here (instead
# of the old `Path(__file__).resolve().parent.parent.parent`) matters
# specifically for the packaged web-app exe: PyInstaller --onefile runs
# code from a temporary extraction folder that's deleted on exit, so the
# old path would silently drop every report the moment the app closed.
BASE_DIR = get_app_base_dir()
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_DIR = BASE_DIR / "reports" / "search"
SEARCH_DIR.mkdir(parents=True, exist_ok=True)
FORGE_DIR = BASE_DIR / "reports" / "forge"
FORGE_DIR.mkdir(parents=True, exist_ok=True)
REFINEMENT_DIR = BASE_DIR / "reports" / "refinement"
REFINEMENT_DIR.mkdir(parents=True, exist_ok=True)
FULL_PIPELINE_DIR = BASE_DIR / "reports" / "full_pipeline"
FULL_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
WFO_DIR = BASE_DIR / "reports" / "walk_forward_opt"
WFO_DIR.mkdir(parents=True, exist_ok=True)
MULTI_OBJ_DIR = BASE_DIR / "reports" / "multi_objective"
MULTI_OBJ_DIR.mkdir(parents=True, exist_ok=True)
WFGA_DIR = BASE_DIR / "reports" / "walkforward_ga"
WFGA_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_DIR = BASE_DIR / "reports" / "portfolio"
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR = BASE_DIR / "reports" / "ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
CPCV_DIR = BASE_DIR / "reports" / "cpcv"
PBO_DIR = BASE_DIR / "reports" / "pbo"
CPCV_DIR.mkdir(parents=True, exist_ok=True)
PBO_DIR.mkdir(parents=True, exist_ok=True)
SENSITIVITY_DIR = BASE_DIR / "reports" / "sensitivity"
SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)
QUICK_OPT_DIR = BASE_DIR / "reports" / "quick_optimize"
QUICK_OPT_DIR.mkdir(parents=True, exist_ok=True)
PAYOUT_DIR = BASE_DIR / "reports" / "payout_probability"
PAYOUT_DIR.mkdir(parents=True, exist_ok=True)
REGIME_DIR = BASE_DIR / "reports" / "regime_matrix"
REGIME_DIR.mkdir(parents=True, exist_ok=True)
SPEEDRUN_DIR = BASE_DIR / "reports" / "speed_run"
SPEEDRUN_DIR.mkdir(parents=True, exist_ok=True)

# Maps each known report directory to the URL prefix that actually serves
# it (see the @app.route("/..._reports/<path:filename>") handlers spread
# throughout this file). Used by _dashboard_report_url below -- see that
# function's docstring for the bug this fixes.
_REPORT_DIR_URL_PREFIXES: list[tuple[Path, str]] = [
    (FULL_PIPELINE_DIR, "full_pipeline_reports"),
    (SEARCH_DIR, "search_reports"),
    (REFINEMENT_DIR, "refinement_reports"),
    (WFO_DIR, "wfo_reports"),
    (MULTI_OBJ_DIR, "mo_reports"),
    (WFGA_DIR, "wfga_reports"),
    (PORTFOLIO_DIR, "portfolio_reports"),
    (ENSEMBLE_DIR, "ensemble_reports"),
    (CPCV_DIR, "cpcv_reports"),
    (PBO_DIR, "pbo_reports"),
    (SENSITIVITY_DIR, "sensitivity_reports"),
    (PAYOUT_DIR, "payout_reports"),
    (SPEEDRUN_DIR, "speed_run_reports"),
    # REPORTS_DIR (the bare "reports/" root, served at /reports/<file>) is
    # deliberately listed LAST: it's an ancestor of every directory above,
    # so it must only match once none of the more specific ones did.
    (REPORTS_DIR, "reports"),
]


def _dashboard_report_url(raw: str | None) -> str | None:
    """Turns whatever app.reports.run_history / app.strategy.library
    stored for a run's report_html field into a URL this web app can
    actually serve.

    Historically, most callers of record_backtest_result()/record_run()
    stored a ready-to-use relative URL (e.g. "/search_reports/x.html"),
    matching the specific route that serves that tool's own report
    directory. But app.orchestration.full_pipeline (and, through it,
    app.reports.run_history.record_run's shared "generate_full_report()"
    path used by every tool) stores the raw ABSOLUTE FILESYSTEM PATH
    instead (e.g. ".../reports/full_pipeline/x.html") -- correct for the
    desktop app opening a local file directly, but meaningless as a web
    URL. The Dashboard used to naively assume every report lives flat
    under /reports/<filename>, which is only true for the plain "Run &
    Report" tool -- clicking a Full-Pipeline-sourced report from the
    Dashboard (including after restarting the server, since this is read
    back from the persistent run-history file) 404'd because the file
    actually lives under /full_pipeline_reports/<filename> instead.

    This resolves it generically: check whether `raw`'s parent directory
    is literally one of this app's known report directories (compared as
    real filesystem paths, not by string prefix) -- if so, it's a raw
    absolute path from one of the buggy callers, and we can build the
    correct URL for that specific route. If no known directory matches,
    `raw` was already a proper relative URL to begin with (a leading "/"
    alone can't distinguish "already a URL" from "an absolute filesystem
    path this list doesn't know about yet" on POSIX, so an unmatched
    value is always returned unchanged rather than guessed at).
    """
    if not raw:
        return None
    try:
        p = Path(raw)
    except Exception:  # noqa: BLE001 -- never let a bad stored value break the dashboard
        return raw
    for report_dir, url_prefix in _REPORT_DIR_URL_PREFIXES:
        try:
            if p.parent == report_dir:
                return f"/{url_prefix}/{p.name}"
        except Exception:  # noqa: BLE001
            continue
    # No known report directory matched -- on POSIX, a value like
    # "/search_reports/x.html" (already a correct relative URL from a
    # caller that did this right) is indistinguishable from a real
    # absolute filesystem path by looking at leading "/" alone, so the
    # only safe move when nothing matches is to return it unchanged
    # rather than guessing a prefix that could just as easily be wrong.
    return raw
# run_speed_run() itself writes each validated candidate's Full Pipeline
# report into output_dir / "speed_run" -- see app.orchestration.speed_run.
SPEEDRUN_REPORTS_DIR = SPEEDRUN_DIR / "speed_run"
SPEEDRUN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")

# UPGRADE (Evolution Lab optimizer_mode): a Jinja GLOBAL rather than
# passing optimizer_modes=OPTIMIZER_MODES through every one of Evolution
# Lab's ~13 render_template("evolution.html"/"evolution_multi_instrument
# .html", ...) call sites (every error path, every re-render, both single
# and multi-instrument) -- Full Pipeline/Quick Optimize/Refinement thread
# it through their own render_template kwargs explicitly instead, but
# those routes each only have a handful of call sites; Evolution Lab's
# many scattered call sites make that same approach error-prone (easy to
# miss one and get a silent "optimizer_modes is undefined" in only some
# code paths). A global is available to every template unconditionally,
# and a route that ALSO passes optimizer_modes explicitly simply
# overrides this for that one render -- the two approaches coexist fine.
app.jinja_env.globals["optimizer_modes"] = OPTIMIZER_MODES

# Session cookie signing key -- only needed for the OPTIONAL local account
# lock (see app.accounts.settings' docstring: set a password on the
# Account tab and this app asks for it once per browser before showing
# any page; leave it blank and none of this is ever touched). Persisted
# alongside the other local JSON config files so an already-unlocked
# browser tab doesn't get re-locked every time the app restarts, while
# still being generated locally rather than hardcoded.
def _load_or_create_flask_secret_key() -> bytes:
    from app.data.storage import get_app_base_dir
    path = get_app_base_dir() / ".flask_secret_key"
    try:
        if path.exists():
            existing = path.read_bytes()
            if existing:
                return existing
    except Exception:  # noqa: BLE001
        pass
    key = os.urandom(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
    except Exception:  # noqa: BLE001
        pass  # Worst case: a fresh random key every restart, which just means re-entering the lock password.
    return key


app.secret_key = _load_or_create_flask_secret_key()

# UPGRADE (licensing, web app): the desktop build has always gated behind
# app.licensing.gate.ensure_licensed() (see app/main.py) -- this build,
# launched via run_web.py / the built T58-Web-App.exe, had no equivalent
# at all, so anyone who received the zip/exe could open and use the full
# app with no email/license-key prompt whatsoever. This reuses the exact
# same app.licensing.client module the desktop activation window calls
# (per-email, per-device, server-verified, revoke/suspend/approve all
# already supported by license_server/ and admin_cli.py) -- just a web
# form (below) in place of gate.py's Tkinter window, since this process
# has no GUI toolkit loaded at all.
#
# Validated ONCE per server process (cached in `_license_ok_cached`)
# rather than on every request, matching the desktop build's own
# convention exactly (app.licensing.gate.ensure_licensed() is likewise
# only ever called once, at launch -- see that module's docstring). A
# revoked/expired license taking effect for an already-running server
# means restarting it, same as it would for an already-running desktop
# app instance.
_license_ok_cached: bool | None = None

# Paths the license gate below never blocks -- the activation page/form
# itself (else nobody could ever activate), and the same static/PWA
# assets the lock gate exempts (so the activation page can render with
# its icon/manifest/theme).
_LICENSE_GATE_EXEMPT_PREFIXES = ("/activate", "/static/", "/manifest.json", "/favicon.ico")


@app.before_request
def _license_gate():
    """Runs before the account-lock gate below (registration order = the
    order Flask calls before_request hooks in), so an unlicensed copy
    never even reaches the optional password lock -- license comes
    first, exactly like the desktop build's ensure_licensed() runs
    before main_window.launch()."""
    global _license_ok_cached
    path = request.path
    if any(path == p or path.startswith(p) for p in _LICENSE_GATE_EXEMPT_PREFIXES):
        return None
    if _license_ok_cached:
        return None

    from app.licensing import client as license_client
    ok, message = license_client.validate()
    _license_ok_cached = ok
    if ok:
        return None
    if request.method == "GET":
        return redirect(url_for("activate_form", next=path))
    return jsonify({"error": "not_licensed", "message": message}), 401


@app.route("/activate", methods=["GET"])
def activate_form():
    from app.licensing import client as license_client
    if _license_ok_cached:
        return redirect(request.args.get("next") or url_for("dashboard"))
    state = license_client.load_state()
    return render_template(
        "activate.html", next=request.args.get("next") or "/dashboard",
        error=None, email=state.email,
    )


@app.route("/activate/submit", methods=["POST"])
def activate_submit():
    global _license_ok_cached
    from app.licensing import client as license_client
    next_path = request.form.get("next") or "/dashboard"
    email = request.form.get("email", "")
    license_key = request.form.get("license_key", "")
    remember = request.form.get("remember") == "1"
    ok, message = license_client.activate(email, license_key, remember=remember)
    if ok:
        _license_ok_cached = True
        return redirect(next_path)
    return render_template("activate.html", next=next_path, error=message, email=email), 401


# Paths the account lock gate below never blocks, even when a password is
# set and the browser hasn't unlocked yet: the lock page itself (else
# nobody could ever unlock), static assets/PWA files (so the lock screen
# itself can render with its icon/manifest/theme), and the Account
# Settings page + its POST endpoints (so a password can be SET or CLEARED
# even from a not-yet-unlocked browser on first run -- see
# _account_lock_gate's own comment for why this one exemption is safe).
_LOCK_GATE_EXEMPT_PREFIXES = (
    "/lock", "/static/", "/manifest.json", "/favicon.ico",
    "/settings/account",
)


@app.before_request
def _account_lock_gate():
    """Optional local app lock -- see app.accounts.settings' module
    docstring. No-op (returns None, request proceeds normally) whenever
    no password is set, the request already unlocked this session, or the
    path is one of the always-exempt ones above. Exempting the whole
    /settings/account* family is deliberate: it is the ONLY place a
    password can be set/changed/cleared, so it must stay reachable even
    pre-unlock, and it reveals nothing itself (no strategy/account data,
    just the profile form) -- everything else on the site stays gated.
    """
    path = request.path
    if any(path == p or path.startswith(p) for p in _LOCK_GATE_EXEMPT_PREFIXES):
        return None
    try:
        from app.accounts.settings import load_account_settings
        settings = load_account_settings()
    except Exception:  # noqa: BLE001 -- a settings-load failure must never lock someone out entirely
        return None
    if not settings.has_password or session.get("t58_unlocked"):
        return None
    if request.method == "GET":
        return redirect(url_for("account_lock_form", next=path))
    # Non-GET (POST/AJAX/job-poll) requests get a plain 401 instead of a
    # redirect, so any existing JS fetch() call fails visibly in the
    # console rather than silently receiving the lock page's HTML back
    # where JSON was expected.
    return jsonify({"error": "locked", "message": "This app is locked. Open it in a browser tab to unlock."}), 401


@app.route("/lock", methods=["GET"])
def account_lock_form():
    from app.accounts.settings import load_account_settings
    if not load_account_settings().has_password or session.get("t58_unlocked"):
        return redirect(request.args.get("next") or url_for("dashboard"))
    return render_template("lock.html", next=request.args.get("next") or "/dashboard", error=None)


@app.route("/lock/unlock", methods=["POST"])
def account_lock_unlock():
    from app.accounts.settings import load_account_settings, verify_password
    next_path = request.form.get("next") or "/dashboard"
    settings = load_account_settings()
    if not settings.has_password:
        session["t58_unlocked"] = True
        return redirect(next_path)
    if verify_password(request.form.get("password", ""), settings.password_hash):
        session["t58_unlocked"] = True
        return redirect(next_path)
    return render_template("lock.html", next=next_path, error="Incorrect password."), 401


# Quant Lab (translator, auto regime selector, strategy health, portfolio
# composer, and the app/quant_lab/ toolkit) lives in its own Blueprint --
# see app/web/quant_lab_routes.py's module docstring for why.
app.register_blueprint(quant_lab_bp)
# T58 AI Assistant (chat + chart/trade screenshot analysis) -- was defined
# in app/web/ai_assistant_routes.py but never actually registered here, so
# /assistant was unreachable from the web app; wiring it in now, alongside
# the sidebar link added in _sidebar.html.
app.register_blueprint(ai_assistant_bp)
# Options Outlook (daily/weekly calls & puts, Black-Scholes-computed) --
# see app/web/options_outlook_routes.py.
app.register_blueprint(options_outlook_bp)
# Hedge Fund Manager (Research -> Black-Litterman portfolio -> execution ->
# read-only Ollama oversight journal) -- see app/web/hedge_fund_routes.py.
app.register_blueprint(hedge_fund_bp)
# Strategy Compare, native PDF export, and the Dukascopy forex/CFD fetch button --
# see app/web/extra_routes.py's module docstring (Deploy Live's live-money
# routes are deliberately NOT here; see that file for why).
app.register_blueprint(extra_bp)
# Risk Sweep -- run_risk_sweep (app/optimize/risk_sweep.py) was already
# fully implemented and tested but had no route calling it; see
# app/web/risk_sweep_routes.py's module docstring.
app.register_blueprint(risk_sweep_bp)
# Belt-and-suspenders alongside run_web.py's own call (this module can also
# be run directly via `python -m app.web.server`, which never goes through
# run_web.py) -- idempotent either way. See app.reports.crash_log.
install_thread_excepthook()


def resolve_strategy_source(mode: str, form, files) -> tuple[str, str]:
    """Single source of truth for "where did this strategy's code come
    from" -- an uploaded file, a saved-library pick, or pasted text.
    Returns (code, suggested_save_name). Any future entry point that needs
    a strategy's source text (not just /run and /search/start) should call
    this instead of re-implementing the precedence rules.

    Precedence: a freshly uploaded file wins over pasted text (the page's
    JS also mirrors an upload into the textarea, so this only matters if
    JS didn't run), and pasted/uploaded text wins over a saved-library pick
    -- matches _resolve_dataset's "most recent upload wins" pattern."""
    code = (form.get("strategy_code") or "").strip()
    existing_choice = (form.get(f"existing_strategy_{mode}") or "").strip()
    uploaded = files.get("strategy_file")
    save_name = (form.get("strategy_save_name") or "").strip()

    if uploaded and uploaded.filename:
        code = uploaded.read().decode("utf-8", errors="replace")
        if not save_name:
            save_name = uploaded.filename

    library_ref = None
    if not code and existing_choice:
        code = load_strategy_text(mode, existing_choice)
        library_ref = existing_choice

    if not code:
        raise StrategyError(
            f"Paste your {mode} strategy code, upload a file, or choose a saved strategy from the library."
        )
    return code, (save_name or library_ref or "strategy")


def maybe_save_strategy_to_library(mode: str, code: str, save_name: str, form) -> str | None:
    """Single source of truth for "save this strategy's code to the
    library", including the overwrite-vs-duplicate prompt and optional
    description/market/tags. Returns the saved filename, or None if the
    "save to library" checkbox wasn't checked. Any future entry point that
    wants to offer a save-to-library option should call this instead of
    re-implementing it."""
    if not form.get("strategy_save_to_library"):
        return None

    overwrite = bool(form.get("strategy_overwrite"))
    try:
        saved_path = save_strategy_text(code, save_name, mode, overwrite=overwrite)
    except StrategyAlreadyExists as exc:
        raise StrategyError(
            f"'{exc.filename}' is already in the {mode} strategy library. "
            'Check "Overwrite existing" and run again to replace it, or change "Save as" to a new name.'
        ) from exc

    description = (form.get("strategy_save_description") or "").strip()
    market = (form.get("strategy_save_market") or "").strip()
    tags_raw = (form.get("strategy_save_tags") or "").strip()
    meta_update = {}
    if description:
        meta_update["description"] = description
    if market:
        meta_update["market"] = market
    if meta_update:
        save_strategy_metadata(mode, saved_path.name, meta_update)
    if tags_raw:
        set_strategy_tags(mode, saved_path.name, [t.strip() for t in tags_raw.split(",")])
    return saved_path.name


def build_strategy_from_code(mode: str, code: str):
    """Single source of truth for turning resolved source text into a
    Strategy object, once code is known (see resolve_strategy_source)."""
    if mode == "python":
        tmp = Path(tempfile.mkdtemp()) / f"strategy_{uuid.uuid4().hex}.py"
        tmp.write_text(code, encoding="utf-8")
        return PythonStrategy(tmp)
    if mode == "pinescript":
        return PineScriptStrategy(code)
    if mode == "mql5":
        return MQL5Strategy(code)
    raise StrategyError(f"Unknown strategy mode: {mode}")


def _try_acquire_heavy_job(job_name: str, template_name: str, **template_ctx):
    """Shared guard check for every long-running background tool (not just
    the four ProcessPoolExecutor-based ones) -- see resource_guard.py's
    module docstring for why WFO/WFGA/CPCV/Sensitivity/Multi-Objective/
    Regime Matrix are included here too. Returns None if the job may
    proceed; otherwise a ready-to-return (response, 409) tuple refusing it,
    rendered with the same template/context the caller's own error paths
    already use, so the person sees an ordinary in-page error rather than a
    generic 409 page."""
    if HEAVY_JOB_GUARD.try_acquire(job_name):
        return None
    msg = (
        f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than one "
        f"long, heavy job at the same time can exhaust available memory or CPU and is a common "
        f"cause of the app becoming unresponsive or crashing. Wait for it to finish (or stop it) "
        f"before starting {job_name}."
    )
    return render_template(template_name, error=msg, **template_ctx), 409


def _build_strategy(mode: str, form, files):
    """Thin orchestrator over resolve_strategy_source / build_strategy_from_code
    / maybe_save_strategy_to_library -- kept as the one call /run and
    /search/start both use, but each piece is independently reusable (see
    their own docstrings) so a future entry point isn't stuck copy-pasting
    this. Returns (strategy, library_ref) where library_ref is
    (mode, filename) if the strategy is tied to a saved library entry
    (loaded from it, or just saved to it) -- used to stamp lookahead/search
    results back onto that entry's metadata -- or None for manual/one-off
    pasted strategies with no library tie."""
    if mode == "manual":
        # A saved-library pick (or an uploaded/pasted manual JSON config)
        # now takes precedence, exactly like the other three strategy
        # types (see resolve_strategy_source) -- this is what was
        # missing before: manual mode ignored the library entirely and
        # always rebuilt a fresh hardcoded SMA-crossover strategy from
        # the quick-builder fields below, regardless of anything picked
        # from a "load from library" dropdown (which didn't even exist
        # for manual before this fix). That made every manually-saved
        # strategy -- including every one the Evolution Lab auto-saves
        # or lets you promote off its leaderboard, see
        # app.evolution.engine._maybe_save_to_library -- completely
        # unusable anywhere in the web app that runs a strategy (Full
        # Pipeline, Search Lab, WFO, WFGA, CPCV, Sensitivity,
        # Multi-Objective, Portfolio, Payout Probability, Research
        # Agent, Regime Matrix, and the main Run page).
        #
        # Falls back to the quick-builder fields (unchanged) only when
        # no library pick, upload, or pasted JSON was given at all, so
        # "just type SMA numbers and run" keeps working exactly as
        # before with zero library interaction required.
        pasted = (form.get("strategy_code") or "").strip()
        existing_choice = (form.get("existing_strategy_manual") or "").strip()
        uploaded = files.get("strategy_file")
        save_name = (form.get("strategy_save_name") or "").strip()

        code = pasted
        if uploaded and uploaded.filename:
            code = uploaded.read().decode("utf-8", errors="replace")
            if not save_name:
                save_name = uploaded.filename

        library_ref = None
        if not code and existing_choice:
            code = load_strategy_text("manual", existing_choice)
            library_ref = ("manual", existing_choice)

        if code:
            try:
                cfg = json.loads(code)
            except json.JSONDecodeError as exc:
                raise StrategyError(
                    f"Manual strategy JSON is invalid: {exc}. Manual strategies are saved/loaded as a "
                    "JSON config (indicators + entry/exit rules), not free-form code."
                ) from exc
            if not isinstance(cfg, dict):
                raise StrategyError("Manual strategy JSON must be an object (a JSON dictionary), not a list or scalar.")

            saved_name = maybe_save_strategy_to_library(
                "manual", code, save_name or (library_ref[1] if library_ref else "manual_strategy.json"), form,
            )
            if saved_name:
                library_ref = ("manual", saved_name)
            return ManualStrategy(cfg), library_ref

        cfg = {
            "name": "Manual Strategy (web)",
            "indicators": [
                {"type": "sma", "period": int(form.get("sma_fast", 20)), "column": "close", "as": "sma_fast"},
                {"type": "sma", "period": int(form.get("sma_slow", 50)), "column": "close", "as": "sma_slow"},
            ],
            "long_entry": "sma_fast > sma_slow",
            "long_exit": "sma_fast < sma_slow",
            "short_entry": "sma_fast < sma_slow",
            "short_exit": "sma_fast > sma_slow",
            "stop_loss_pips": float(form.get("sl_pips", 20)),
            "take_profit_pips": float(form.get("tp_pips", 40)),
        }
        return ManualStrategy(cfg), None

    code, save_name = resolve_strategy_source(mode, form, files)
    existing_choice = (form.get(f"existing_strategy_{mode}") or "").strip()

    saved_name = maybe_save_strategy_to_library(mode, code, save_name, form)
    library_ref = (mode, saved_name) if saved_name else ((mode, existing_choice) if existing_choice else None)

    return build_strategy_from_code(mode, code), library_ref


def _resolve_dataset(form, files):
    """
    Shared by /run and /search/start: picks the active DataFrame from
    either newly-uploaded CSV file(s) or a previously-stored dataset,
    exactly the same precedence /run has always used (most recently
    uploaded valid file wins over a selected stored dataset). Returns
    (df, label, import_note, error_message) -- df is None and
    error_message is set if nothing usable was provided.
    """
    uploaded_files = [f for f in files.getlist("csv_file") if f and f.filename]
    existing_choice = (form.get("existing_dataset") or "").strip()

    imported_names, failed = [], []
    active_df = None
    active_label = None

    for f in uploaded_files:
        content = f.read()
        result = import_csv_bytes(content, filename=f.filename)
        if not result.is_valid:
            failed.append((f.filename, "; ".join(result.errors)))
            continue
        store_csv_bytes(content, f.filename)
        imported_names.append(f.filename)
        active_df = result.dataframe  # most recently imported valid file becomes active
        active_label = f.filename

    if active_df is None and existing_choice:
        candidate = get_raw_data_dir() / existing_choice
        if candidate.exists():
            result = import_csv(candidate)
            if result.is_valid:
                active_df = result.dataframe
                active_label = existing_choice

    if active_df is None:
        msg = "Please upload at least one valid CSV, or choose a previously stored dataset."
        if failed:
            msg += " Failed upload(s): " + "; ".join(f"{n} ({e})" for n, e in failed)
        return None, None, None, msg

    import_note = None
    if imported_names:
        import_note = f"Stored {len(imported_names)} file(s) in data/raw/: {', '.join(imported_names)}."
        if failed:
            import_note += f" {len(failed)} file(s) failed and were skipped: " + \
                "; ".join(f"{n} ({e})" for n, e in failed)

    return active_df, active_label, import_note, None




def _resolve_family_exclusions(log_lines: list | None = None) -> "set[str] | None":
    """Shared by /search/start and /search/multi-instrument/start: computes
    which named families app.search.family_health flags as dead ends
    (30+ tests across every past run, zero successes), for passing
    straight into generate_search_space's exclude_families -- which
    itself only ever applies this to an explicit "all families" request,
    never a single named family. Appends a one-line note to `log_lines`
    (if given) when anything is actually flagged, so it's visible on the
    run's own progress log, not just silently different candidate counts.
    Never raises -- a family-health scan failing must never block a
    search from starting."""
    try:
        from app.search.family_health import apply_family_exclusions
        survivors, excluded = apply_family_exclusions()
    except Exception:  # noqa: BLE001
        return None
    if excluded and log_lines is not None:
        log_lines.append(
            f"Auto-excluding {len(excluded)} dead-end famil{'y' if len(excluded) == 1 else 'ies'} "
            f"(tested 30+ times across past runs with zero successes): {', '.join(excluded)}."
        )
    return set(excluded) if survivors is not None else None


def _prop_presets_json() -> str:
    """[{"key","label","firm","account_size","evaluation_profit_target_pct",
    "daily_loss_limit_pct","max_drawdown_pct","drawdown_type",
    "drawdown_check_mode","consistency_rule_pct","min_trading_days",
    "payout_threshold_pct","payout_cap_pct","payout_frequency_days",
    "required_buffer_pct","as_of","source_note"}, ...] for any page's JS
    to build a "quick-fill from a prop-firm preset" dropdown that
    populates that page's own prop-rule form fields on selection. See
    app.prop.presets' module docstring for why these numbers are
    approximate and dated.
    """
    return json.dumps([p.to_dict() for p in list_prop_firm_presets()])


def _saved_strategies_json() -> str:
    """{"python": [{"name", "description", "market", "tags", "status",
    "last_run", "lookahead", "last_search", "evolution"}, ...], "pinescript": [...],
    "mql5": [...]} for the page's JS to build the "load from library"
    dropdowns, the client-side search/tag/market/status filters, and to
    prefill fields when a saved strategy is picked. "evolution" is only
    present for strategies the Evolution Lab produced (auto-saved elites
    or ones promoted off its leaderboard -- see
    app.evolution.engine.evolution_stats_metadata) -- None for anything
    hand-built, uploaded, or AI-generated.
    """
    return json.dumps({
        t: [
            {
                "name": s.name,
                "description": s.metadata.get("description", ""),
                "market": s.metadata.get("market", ""),
                "tags": s.tags,
                "status": s.status,
                "last_run": s.metadata.get("last_run"),
                "lookahead": s.metadata.get("lookahead"),
                "last_search": s.metadata.get("last_search"),
                "evolution": s.metadata.get("evolution"),
            }
            for s in list_saved_strategies(t)
        ]
        for t in STRATEGY_TYPES
    })


def _alpaca_template_context() -> dict:
    """Thin wrapper over app.web.alpaca_shared.alpaca_template_context so
    every existing call site in this file (index.html, full_pipeline.html,
    search.html, quick_optimize.html, and the batch of Optimize/Validate/
    Champion/Research pages added alongside it) keeps working unchanged.
    Blueprint modules (extra_routes.py, hedge_fund_routes.py,
    risk_sweep_routes.py) import alpaca_template_context directly from
    alpaca_shared instead, since they can't import this module."""
    return alpaca_template_context()


def _lse_template_context() -> dict:
    """Thin wrapper over app.web.lse_shared.lse_template_context, mirroring
    _alpaca_template_context() above."""
    return lse_template_context()


# Every page with its own "Or fetch data from Alpaca" section (see
# _alpaca_fetch_section.html) submits to the SAME two routes below rather
# than each having its own copy of this handler -- self-contained per page
# for the SETTINGS (each page's own dataset pick/prop rules/risk config,
# per the desktop app's RunContextPanel equivalent), but there's no reason
# for "talk to the Alpaca API and drop a file in data/raw/" to be
# duplicated per page since it has no per-page state of its own. Each
# page's Alpaca form includes a hidden alpaca_return_to field naming its
# own GET endpoint, so the redirect-with-notice lands back on whichever
# page the person actually submitted from instead of always bouncing to
# the Run & Report page.
_ALPACA_RETURN_ENDPOINTS = {
    "index", "full_pipeline_form", "search_form", "quickopt_form",
    "cpcv_form", "ensemble_form", "evolution_form", "forge_form", "mo_form",
    "overnight_autopilot_form", "parameter_robustness_form", "payout_probability_form",
    "pbo_form", "portfolio_form", "prop_firm_recommender_form", "refine_form",
    "regime_matrix_form", "research_form", "research_agent_form", "research_loop_form",
    "sensitivity_form", "speed_run_form", "wfga_form", "wfo_form",
    # These three live in their own Blueprints (extra_bp / hedge_fund_bp /
    # risk_sweep_bp), so url_for() needs the blueprint-qualified endpoint name.
    "extra.compare_page", "hedge_fund.hedge_fund_form", "risk_sweep.risk_sweep_form",
}


def _alpaca_redirect(notice: str, kind: str):
    endpoint = request.form.get("alpaca_return_to") or "index"
    if endpoint not in _ALPACA_RETURN_ENDPOINTS:
        endpoint = "index"
    return redirect(url_for(endpoint, alpaca_notice=notice, alpaca_notice_kind=kind))


@app.route("/")
def index():
    return render_template(
        "index.html",
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(),
        strategy_notice=request.args.get("strategy_notice"),
        alpaca_notice=request.args.get("alpaca_notice"),
        alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"),
        lse_notice=request.args.get("lse_notice"),
        lse_notice_kind=request.args.get("lse_notice_kind", "info"),
        strategy_statuses=STRATEGY_STATUSES,
        **_alpaca_template_context(), **_lse_template_context(),
    )


def _lse_redirect(notice: str, kind: str):
    endpoint = request.form.get("lse_return_to") or "index"
    if endpoint not in _ALPACA_RETURN_ENDPOINTS:
        endpoint = "index"
    return redirect(url_for(endpoint, lse_notice=notice, lse_notice_kind=kind))


@app.route("/data/lse/fetch", methods=["POST"])
def data_lse_fetch():
    """Fetches candles from London Strategic Edge and saves them into
    data/raw/<SYMBOL>/, same convention as data_alpaca_fetch() above --
    one key instead of two, otherwise the identical shape (plain form
    POST, redirect-with-notice back to whichever page submitted it)."""
    from app.accounts.api_keys import load_settings as load_api_keys_settings, save_settings as save_api_keys_settings

    form = request.form
    api_key = (form.get("lse_api_key") or "").strip()
    save_key = form.get("lse_save_key") == "on"

    existing_keys = load_api_keys_settings()
    # A blank, masked key field means "keep using the saved one" -- the
    # page never echoes a real saved key back into the HTML.
    if not api_key:
        api_key = existing_keys.london_strategic_edge_key

    symbols = [s.strip() for s in (form.get("lse_symbols") or "").split(",") if s.strip()]
    timeframe_label = form.get("lse_timeframe") or TIMEFRAME_CHOICES_LSE[0]
    start = (form.get("lse_start") or "").strip()
    end = (form.get("lse_end") or "").strip()

    if not api_key:
        return _lse_redirect("Enter a London Strategic Edge API key.", "error")
    if not symbols:
        return _lse_redirect("Enter at least one symbol.", "error")

    if save_key:
        existing_keys.london_strategic_edge_key = api_key
        save_api_keys_settings(existing_keys)

    saved_names, errors = [], []
    for symbol in symbols:
        try:
            df = lse_fetch_candles(api_key, symbol, timeframe_label, start, end)
            dest = lse_save_bars_as_csv(df, symbol, timeframe_label)
            saved_names.append(dest.relative_to(get_raw_data_dir()).as_posix())
        except (LSEImportError, LSEFetchError) as exc:
            errors.append(f"{symbol}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{symbol}: unexpected error ({exc})")

    if saved_names and not errors:
        notice, kind = f"Saved {len(saved_names)} file(s): {', '.join(saved_names)}.", "success"
    elif saved_names and errors:
        notice = f"Saved {len(saved_names)} file(s): {', '.join(saved_names)}. Failed: {'; '.join(errors)}"
        kind = "warning"
    else:
        notice, kind = f"Fetch failed: {'; '.join(errors)}", "error"

    return _lse_redirect(notice, kind)


@app.route("/data/lse/forget", methods=["POST"])
def data_lse_forget():
    from app.accounts.api_keys import load_settings as load_api_keys_settings, save_settings as save_api_keys_settings

    existing_keys = load_api_keys_settings()
    existing_keys.london_strategic_edge_key = ""
    save_api_keys_settings(existing_keys)
    return _lse_redirect("Saved London Strategic Edge key removed from this computer.", "success")


@app.route("/data/alpaca/fetch", methods=["POST"])
def data_alpaca_fetch():
    """Fetches bars from Alpaca and saves them into data/raw/<SYMBOL>/,
    same as the desktop app's FETCH & SAVE button. A plain form POST (not
    AJAX) to stay consistent with the rest of this page and to keep
    working with JS disabled; redirects back to whichever page submitted
    the form (see _alpaca_redirect above) with a short notice."""
    form = request.form
    api_key = (form.get("alpaca_api_key") or "").strip()
    secret_key = (form.get("alpaca_secret_key") or "").strip()
    save_keys = form.get("alpaca_save_keys") == "on"

    # A blank, masked secret field means "keep using the saved one" -- the
    # page never echoes a real saved secret back into the HTML.
    if not api_key or not secret_key:
        saved = alpaca_credentials.load_credentials()
        if saved:
            api_key = api_key or saved.api_key
            secret_key = secret_key or saved.secret_key

    symbols = [s.strip() for s in (form.get("alpaca_symbols") or "").split(",") if s.strip()]
    asset_class = form.get("alpaca_asset_class") or ASSET_CLASSES[0]
    timeframe_label = form.get("alpaca_timeframe") or TIMEFRAME_LABELS[0]
    start = (form.get("alpaca_start") or "").strip()
    end = (form.get("alpaca_end") or "").strip()
    feed = form.get("alpaca_feed") or FEED_CHOICES[0]
    adjustment = form.get("alpaca_adjustment") or ADJUSTMENT_CHOICES[0]

    if not api_key or not secret_key:
        return _alpaca_redirect("Enter both an API key and a secret key.", "error")
    if not symbols:
        return _alpaca_redirect("Enter at least one symbol.", "error")

    if save_keys:
        alpaca_credentials.save_credentials(api_key, secret_key)

    saved_names, errors = [], []
    for symbol in symbols:
        try:
            df = fetch_bars(
                api_key, secret_key, symbol, asset_class, timeframe_label, start, end,
                feed=feed, adjustment=adjustment,
            )
            dest = save_bars_as_csv(df, symbol, timeframe_label)
            saved_names.append(dest.relative_to(get_raw_data_dir()).as_posix())
        except (AlpacaImportError, AlpacaFetchError) as exc:
            errors.append(f"{symbol}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{symbol}: unexpected error ({exc})")

    if saved_names and not errors:
        notice, kind = f"Saved {len(saved_names)} file(s): {', '.join(saved_names)}.", "success"
    elif saved_names and errors:
        notice = f"Saved {len(saved_names)} file(s): {', '.join(saved_names)}. Failed: {'; '.join(errors)}"
        kind = "warning"
    else:
        notice, kind = f"Fetch failed: {'; '.join(errors)}", "error"

    return _alpaca_redirect(notice, kind)


@app.route("/data/alpaca/forget", methods=["POST"])
def data_alpaca_forget():
    alpaca_credentials.clear_credentials()
    return _alpaca_redirect("Saved Alpaca keys removed from this computer.", "success")


@app.route("/data/detect-pip-size", methods=["POST"])
def data_detect_pip_size():
    """AJAX counterpart to the desktop app's "DETECT PIP SIZE FROM DATA"
    button (Step 4, Risk & Execution). Reuses the exact same dataset
    resolution as /run (newly-uploaded file(s) win over a selected stored
    dataset) so a freshly-picked-but-not-yet-run CSV can still be detected
    against, then app.backtest.risk.suggest_pip_size gives one suggested
    starting value -- never applied automatically, just returned for the
    page's JS to drop into the Pip size field for the person to confirm."""
    try:
        df, label, _note, err = _resolve_dataset(request.form, request.files)
        if err:
            return jsonify({"error": err}), 400
        suggested = suggest_pip_size(df)
        return jsonify({
            "pip_size": suggested,
            "message": f"Suggested {suggested} from {label} -- confirm this matches the instrument before running a backtest.",
        })
    except Exception as exc:
        return jsonify({"error": f"Couldn't detect: {exc}"}), 400


@app.route("/mobile-access")
def mobile_access():
    """Shows the same LAN address + QR code the console banner prints
    (see app.web.network_info), but IN the running app itself -- so it's
    reachable from a browser tab on the PC (e.g. after starting via
    `python -m app.web.server`, which used to show none of this), not
    only from the separate run_web.py launcher's popped-open image."""
    url = lan_url()
    qr_data_uri = qr_code_data_uri(url)
    # Tailscale address is optional -- reachable from anywhere (not just
    # this Wi-Fi), shown as a second card only when Tailscale is actually
    # installed and signed in on this machine. See app.web.network_info.
    ts_url = tailscale_url()
    ts_qr_data_uri = qr_code_data_uri(ts_url) if ts_url else None
    return render_template(
        "mobile_access.html", active_page="mobile_access", url=url, qr_data_uri=qr_data_uri,
        tailscale_url=ts_url, tailscale_qr_data_uri=ts_qr_data_uri,
    )


@app.route("/resources")
def resources():
    """A static, no-engine-dependency page for the T58 30-Day Trading
    Quickstart Guide (Owen's own Google Doc) plus a short list of other
    free, well-known beginner trading resources -- purely educational,
    for anyone using this app who wants a running start on trading
    fundamentals before backtesting their first strategy. No form, no
    job, no report; just links out.
    """
    return render_template("resources.html", active_page="resources")


@app.route("/education")
def education():
    """Quant/algo-trading concept lessons -- what a number or a tab
    actually MEANS, as opposed to /user-manual (which button to click)
    and /resources (external trading-fundamentals links). Content lives
    in app.education.content as one shared source both this route and
    the desktop Education tab (app.ui.main_window._build_education_tab)
    render from, so the two can never drift apart. No form, no job, no
    report; just a reading page, same as Resources and User Manual."""
    return render_template("education.html", active_page="education", sections=education_content.list_sections())


@app.route("/user-manual")
def user_manual():
    """Web equivalent of the desktop app's User Manual tab (see
    app.ui.main_window._build_manual_tab) -- a start-to-finish workflow
    walkthrough (Create / Test / Optimize / Validate / Champion /
    Deployment), rewritten for the web app's own page names and URLs
    rather than a verbatim port of the desktop tab's Tkinter text
    (the two apps' navigation labels differ enough that a literal copy
    would point at things that don't exist here). No form, no job, no
    report; just a reading page, same as Resources."""
    return render_template("user_manual.html", active_page="user_manual")


@app.route("/dashboard")
def dashboard():
    current = strategy_state.get_current_strategy()
    checklist = None
    score = None
    current_metrics = None
    if current:
        checklist = strategy_state.get_checklist(current["strategy_name"], current["instrument"])
        score = strategy_state.robustness_score(current["strategy_name"], current["instrument"])
        current_metrics = run_history.latest_run_for(current["strategy_name"], current["instrument"])
    dashboard_stats = run_history.dashboard_data()
    # Fix up report_html links before they reach the template -- see
    # _dashboard_report_url's docstring for why the raw stored value
    # isn't always a usable URL as-is (Full Pipeline runs store an
    # absolute filesystem path, not a "/xxx_reports/file" URL, which
    # 404'd when the template guessed "/reports/<filename>" for every
    # row regardless of which tool actually produced it).
    if dashboard_stats.get("best") is not None:
        dashboard_stats["best"]["report_html"] = _dashboard_report_url(dashboard_stats["best"].get("report_html"))
    for _s in dashboard_stats.get("strategies") or []:
        _s["report_html"] = _dashboard_report_url(_s.get("report_html"))
    show_welcome = pipeline_guide.should_show_first_run_welcome(
        has_stored_datasets=bool(list_stored_datasets()),
        has_run_history=bool(dashboard_stats.get("total_runs")),
    )
    # Champion Board: "strongest validated candidate under your defined
    # research criteria" -- replaces the old highest-Sharpe "best" card as
    # the dashboard's headline. See app.scoring.champion_board's module
    # docstring for why raw Sharpe alone never crowns a champion here.
    board_rows = champion_board.list_board()
    champion = champion_board.strongest_validated_candidate(board_rows)
    snapshot = champion_board.five_question_snapshot(current, board_rows)
    return render_template(
        "dashboard.html",
        data=dashboard_stats,
        dataset_groups=list_datasets_by_instrument(),
        current_strategy=current,
        current_metrics=current_metrics,
        checklist=checklist,
        robustness=score,
        validation_labels=strategy_state.VALIDATION_LABELS,
        validation_hrefs=strategy_state.VALIDATION_HREFS,
        validation_kinds=strategy_state.VALIDATION_KINDS,
        show_welcome=show_welcome,
        welcome_message=pipeline_guide.first_run_welcome() if show_welcome else None,
        board_rows=board_rows,
        champion=champion,
        snapshot=snapshot,
    )


@app.route("/champion/promote", methods=["POST"])
def champion_promote():
    """Explicit, manual promotion action for one Champion Board row -- see
    app.scoring.champion_board.promote_strategy's docstring for why this
    is never automatic. Fails loudly (as a query-string notice) with the
    specific unmet requirement(s) rather than silently no-opping."""
    strategy_type = (request.form.get("strategy_type") or "").strip()
    filename = (request.form.get("filename") or "").strip()
    if strategy_type and filename:
        ok, message, _new_stage = champion_board.promote_strategy(strategy_type, filename)
        return redirect(url_for("dashboard", promote_notice=message, promote_ok="1" if ok else "0"))
    return redirect(url_for("dashboard"))


@app.route("/api/dashboard-data")
def api_dashboard_data():
    """JSON feed the dashboard page polls to refresh live, without a full
    page reload, whenever a run finishes (desktop, web, or CLI)."""
    return jsonify(run_history.dashboard_data())


@app.route("/api/prop-firm-presets")
def api_prop_firm_presets():
    """Same preset list as _prop_presets_json(), served as a standalone
    JSON endpoint so app/web/static/prop-presets.js can fetch it once
    and wire up a "prop firm preset" dropdown on ANY page -- including
    pages/render paths that don't (or forget to) pass prop_presets_json
    into their own render_template() call. This is what lets every tab
    with prop-firm-rule fields (account size, profit target, daily loss,
    max drawdown, ...) get a working preset shortcut without every one
    of ~60 render_template() call sites across this file needing to
    remember to thread the data through by hand."""
    return jsonify([p.to_dict() for p in list_prop_firm_presets()])


@app.route("/current-strategy/set", methods=["POST"])
def set_current_strategy():
    """Explicit only -- the person picks a strategy off the Dashboard
    scorecard (or the Champion card) and marks it current. Nothing here
    infers a current strategy from just running a backtest, so a stray
    one-off test never silently hijacks the checklist."""
    form = request.form
    name = (form.get("strategy_name") or "").strip()
    instrument = (form.get("instrument") or "").strip()
    timeframe = (form.get("timeframe") or "").strip()
    if name and instrument:
        strategy_state.set_current_strategy(name, instrument, timeframe)
    return redirect(url_for("dashboard"))


@app.route("/current-strategy/clear", methods=["POST"])
def clear_current_strategy():
    strategy_state.clear_current_strategy()
    return redirect(url_for("dashboard"))


@app.route("/optimize")
def optimize_hub():
    """Guided picker for the OPTIMIZE stage -- every method underneath is
    still its own full page (nothing lost), this just answers "which one
    should I use" before sending the person on to it."""
    current = strategy_state.get_current_strategy()
    return render_template("optimize_hub.html", active_page="optimize_hub", current_strategy=current)


@app.route("/validate")
def validate_hub():
    """Guided checklist for the VALIDATE stage, built entirely from data
    already recorded by CPCV / WFO / WFGA / Sensitivity / Regime Matrix --
    nothing here re-runs or duplicates those tools, it just aggregates what
    they've already found for whichever strategy is marked current."""
    current = strategy_state.get_current_strategy()
    checklist = None
    score = None
    if current:
        checklist = strategy_state.get_checklist(current["strategy_name"], current["instrument"])
        score = strategy_state.robustness_score(current["strategy_name"], current["instrument"])
    return render_template(
        "validate_hub.html", active_page="validate_hub", current_strategy=current,
        checklist=checklist, robustness=score,
        validation_labels=strategy_state.VALIDATION_LABELS,
        validation_hrefs=strategy_state.VALIDATION_HREFS,
        validation_kinds=strategy_state.VALIDATION_KINDS,
    )


@app.route("/strategies/<strategy_type>/<path:filename>")
def get_saved_strategy(strategy_type, filename):
    """Raw source text of one saved strategy, for the page's JS to pull into
    the strategy_code textarea when the person picks it from the library."""
    try:
        text = load_strategy_text(strategy_type, filename)
    except (ValueError, FileNotFoundError) as exc:
        return Response(str(exc), status=404, mimetype="text/plain")
    return Response(text, mimetype="text/plain")


def _redirect_target(form) -> str:
    if form.get("return_to") == "search":
        return "search_form"
    if form.get("return_to") == "library":
        return "strategy_library_page"
    return "index"


_REPLAY_RESULTS: dict = {}
_REPLAY_RESULTS_LOCK = threading.Lock()
# Separate from _REPLAY_RESULTS on purpose: that dict is returned verbatim
# by /replay/data/<id>.json via jsonify(), so it can only ever hold plain
# JSON-safe values. This one holds the actual df/Strategy/RiskConfig
# objects behind a prepared replay, purely so /replay/rerun/<id> (the
# "configure while in replay" feature) can re-run the SAME strategy
# against the SAME dataset with new risk/prop settings without the user
# re-uploading or re-selecting anything.
_REPLAY_SOURCES: dict = {}
_REPLAY_SOURCES_LOCK = threading.Lock()


@app.route("/replay")
def replay_form():
    return render_template(
        "replay.html", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context(),
    )


def _describe_strategy_for_replay(strategy, risk: RiskConfig, result) -> dict:
    """A small, JSON-safe summary of the selected strategy and its current
    risk settings for the Interactive Replay sidebar's Strategy panel --
    deliberately not a full config dump, just enough to confirm at a
    glance which strategy and settings actually produced this replay's
    trades."""
    return {
        "name": result.strategy_name,
        "source_type": getattr(strategy, "source_type", "unknown"),
        "timeframe": describe_resolved_timeframe(strategy),
        "risk_mode": risk.risk_mode,
        "risk_value": risk.risk_value,
        "pip_size": risk.pip_size,
        "initial_balance": risk.initial_balance,
    }


def _build_replay_payload(df, strategy, risk: RiskConfig, prop_account_size: float,
                           prop_rules_form: dict, active_label: str, result) -> dict:
    """Builds the exact JSON-safe dict /replay/data/<id>.json serves, from
    an already-run BacktestResult. Shared by /replay/prepare (the first
    run) and /replay/rerun/<id> (re-running the same strategy/dataset
    with different risk/prop settings from the replay sidebar) so both
    produce byte-identical payload shapes."""
    import pandas as pd
    bars = [
        {
            "time": int(pd.Timestamp(row.timestamp).timestamp()),
            "open": float(row.open), "high": float(row.high), "low": float(row.low), "close": float(row.close),
            "volume": float(row.volume) if hasattr(row, "volume") else 0.0,
        }
        for row in df.itertuples(index=False)
    ]
    trades = [
        {
            "entry_time": int(pd.Timestamp(t.entry_time).timestamp()),
            "exit_time": int(pd.Timestamp(t.exit_time).timestamp()),
            "direction": t.direction, "entry_price": t.entry_price, "exit_price": t.exit_price,
            "pnl": t.pnl, "exit_reason": t.exit_reason,
            # UPGRADE (Interactive Replay: TP/SL fields, running balance):
            # equity_after is already tracked per-trade by the backtest
            # engine itself (app.backtest.execution.Trade) -- exactly the
            # "new account balance after this trade" the right-hand panel
            # needs, not something computed here. stop_loss_price is
            # derived from initial_risk (|entry - stop| in price units,
            # also already tracked per-trade) -- always available whenever
            # the trade was risk-sized off a stop. take_profit_price is
            # only ever knowable when a take-profit was the actual reason
            # the trade closed (exit_price IS that price then) -- for any
            # other exit reason the strategy's TP target (if it even had
            # one) was never reached, so there is no real number to show;
            # left null rather than guessed.
            "equity_after": t.equity_after,
            "stop_loss_price": (
                round(t.entry_price - t.direction * t.initial_risk, 6) if t.initial_risk else None
            ),
            "take_profit_price": (t.exit_price if t.exit_reason == "take_profit" else None),
        }
        for t in result.trades
    ]
    equity = [
        {"time": int(pd.Timestamp(ts).timestamp()), "equity": float(eq)}
        for ts, eq in zip(result.equity_curve["timestamp"], result.equity_curve["equity"])
    ] if "timestamp" in result.equity_curve.columns and "equity" in result.equity_curve.columns else []

    return {
        "bars": bars, "trades": trades, "equity": equity,
        "label": active_label, "initial_balance": risk.initial_balance,
        "prop_account_size": prop_account_size,
        "prop_rules": prop_rules_form,
        "strategy_info": _describe_strategy_for_replay(strategy, risk, result),
        "statistics": result.statistics.to_dict() if hasattr(result.statistics, "to_dict") else {},
    }


def _parse_replay_prop_rules(form) -> dict:
    """The Prop Account section of both replay.html (initial setup) and
    replay_view.html's sidebar (re-run) posts these same four fields.
    Previously only account_size ever made it past this route -- profit
    target / daily loss / max drawdown were collected by the form and
    silently discarded, so Replay's Prop Account panel had nothing real
    to show or check against. Returns plain floats/None, never raises --
    a blank or unparsable field just means that limit isn't shown/checked,
    not a failed request."""
    def _f(name, default=None):
        raw = (form.get(name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default
    return {
        "account_size": _f("account_size"),
        "profit_target_pct": _f("profit_target"),
        "daily_loss_pct": _f("daily_loss"),
        "max_drawdown_pct": _f("max_dd"),
    }


@app.route("/replay/prepare", methods=["POST"])
def replay_prepare():
    """UPGRADE (Interactive Replay): runs the SAME run_backtest every
    other tool in this app uses -- once, up front -- then hands the
    browser the full bar/trade/equity history to scrub and play through
    client-side. This is pure visualization of an already-computed,
    already-validated backtest: no new execution or fill logic exists
    here, so there is no way for this feature to produce a different
    number than Run & Report would for the exact same inputs (worth
    stating plainly, since tick-level fills / L2 order-book simulation --
    the other two items in this phase -- are a fundamentally different,
    much riskier kind of change that this is NOT)."""
    form = request.form
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template(
                "replay.html", error=dataset_error, stored_datasets=list_stored_datasets(),
                dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
                **_alpaca_template_context(),
            ), 400

        strategy_mode = form.get("strategy_mode", "manual")
        if strategy_mode == "library":
            library_type = form.get("library_strategy_type", "")
            library_name = form.get("library_strategy_name", "")
            if not library_type or not library_name:
                raise StrategyError("Pick a saved strategy from the library first.")
            strategy = _load_library_strategy_for_batch(library_type, library_name)
        else:
            strategy, _library_ref = _build_strategy(strategy_mode, form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        # UPGRADE (Interactive Replay: prop-firm presets): optional -- an
        # account-size/risk panel for the same right-hand running-balance
        # display below, using the same preset dropdown (app.web.static.
        # prop-presets.js) and PropRules fields every other tool's Prop
        # Account section already uses. Not passed into run_backtest --
        # Replay is pure visualization of a plain backtest's own trade
        # sequence and equity curve (see this route's own top comment),
        # not a prop-firm evaluation/payout simulation -- account_size
        # only sets what "the account" starts at for the balance panel.
        prop_account_size = float(form.get("account_size", risk.initial_balance) or risk.initial_balance)
        prop_rules_form = _parse_replay_prop_rules(form)
    except (StrategyError, RefinementError) as exc:
        return render_template(
            "replay.html", error=str(exc), stored_datasets=list_stored_datasets(),
            dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
            **_alpaca_template_context(),
        ), 400

    # FIX (Interactive Replay taking zero trades): this used to trim to
    # max_bars BEFORE the strategy's own declared execution timeframe was
    # resolved (see app.data.timeframe_resample) -- a strategy that
    # declares e.g. TIMEFRAME="15m" against 1-minute data got its lookback
    # window trimmed on the WRONG (native, pre-resample) bar count, then
    # resampled what was left, silently handing it a much shorter history
    # than the same strategy would see through Run & Report (which never
    # trims at all). Resolving the timeframe alignment first and trimming
    # the ALREADY-aligned frame -- exactly what run_backtest itself will
    # see -- means Replay's displayed window matches what actually
    # generated the trades, instead of a mismatched pre-resample slice.
    df, _tf_warnings = prepare_timeframe_aligned_data(df, strategy)
    max_bars = int(form.get("max_bars", 2000) or 2000)
    if len(df) > max_bars:
        df = df.tail(max_bars).reset_index(drop=True)

    result = run_backtest(df, strategy, risk)
    payload = _build_replay_payload(df, strategy, risk, prop_account_size, prop_rules_form, active_label, result)

    replay_id = uuid.uuid4().hex[:12]
    with _REPLAY_RESULTS_LOCK:
        _REPLAY_RESULTS[replay_id] = payload
    with _REPLAY_SOURCES_LOCK:
        _REPLAY_SOURCES[replay_id] = {"df": df, "strategy": strategy, "risk": risk, "active_label": active_label}
    return redirect(url_for("replay_view", replay_id=replay_id))


@app.route("/replay/rerun/<replay_id>", methods=["POST"])
def replay_rerun(replay_id):
    """UPGRADE (Interactive Replay: configure while in replay): re-runs
    the SAME strategy against the SAME already-loaded dataset with new
    risk/prop settings from the replay sidebar's own form -- no re-upload
    or re-selection needed. Produces a NEW replay_id (the old one is left
    intact and still viewable) so a browser back button still works.
    404s -- with a plain explanatory page rather than a raw error -- if
    the source strategy/dataset for `replay_id` isn't resident in memory
    any more (a server restart since it was prepared, same limitation
    /replay/data/<id>.json already has)."""
    with _REPLAY_SOURCES_LOCK:
        source = _REPLAY_SOURCES.get(replay_id)
    if source is None:
        return render_template(
            "replay_view.html", replay_id=replay_id, not_found=True,
            rerun_expired=True,
        ), 404

    form = request.form
    old_risk = source["risk"]
    new_risk = RiskConfig(
        initial_balance=float(form.get("initial_balance") or old_risk.initial_balance),
        risk_mode=form.get("risk_mode") or old_risk.risk_mode,
        risk_value=float(form.get("risk_value") or old_risk.risk_value),
        pip_size=float(form.get("pip_size") or old_risk.pip_size),
    )
    prop_account_size = float(form.get("account_size") or new_risk.initial_balance)
    prop_rules_form = _parse_replay_prop_rules(form)

    df, strategy, active_label = source["df"], source["strategy"], source["active_label"]
    result = run_backtest(df, strategy, new_risk)
    payload = _build_replay_payload(df, strategy, new_risk, prop_account_size, prop_rules_form, active_label, result)

    new_replay_id = uuid.uuid4().hex[:12]
    with _REPLAY_RESULTS_LOCK:
        _REPLAY_RESULTS[new_replay_id] = payload
    with _REPLAY_SOURCES_LOCK:
        _REPLAY_SOURCES[new_replay_id] = {"df": df, "strategy": strategy, "risk": new_risk, "active_label": active_label}
    return redirect(url_for("replay_view", replay_id=new_replay_id))


@app.route("/replay/view/<replay_id>")
def replay_view(replay_id):
    with _REPLAY_RESULTS_LOCK:
        exists = replay_id in _REPLAY_RESULTS
    return render_template("replay_view.html", replay_id=replay_id, not_found=not exists)


@app.route("/replay/data/<replay_id>.json")
def replay_data(replay_id):
    with _REPLAY_RESULTS_LOCK:
        data = _REPLAY_RESULTS.get(replay_id)
    if data is None:
        return jsonify({"error": "Replay data not found (the server may have restarted since it was prepared)."}), 404
    return jsonify(data)


@app.route("/library")
def strategy_library_page():
    from app.strategy.library import list_saved_strategies, list_all_tags, list_all_markets, STRATEGY_TYPES

    all_strategies = []
    for t in STRATEGY_TYPES:
        for s in list_saved_strategies(t):
            all_strategies.append({
                "type": t, "name": s.name, "description": s.metadata.get("description", ""),
                "market": s.metadata.get("market", ""), "tags": s.tags, "status": s.status,
                "status_display": s.status_display, "last_run": s.metadata.get("last_run"),
                "last_search": s.metadata.get("last_search"), "size_bytes": s.size_bytes,
                "modified": s.modified, "pipeline_progress": s.pipeline_progress,
            })
    all_strategies.sort(key=lambda s: s["modified"], reverse=True)

    all_tags = sorted({tag for t in STRATEGY_TYPES for tag in list_all_tags(t)})
    all_markets = sorted({m for t in STRATEGY_TYPES for m in list_all_markets(t) if m})

    return render_template(
        "library.html", active_page="strategy_library",
        strategies_json=json.dumps(all_strategies), all_tags=all_tags, all_markets=all_markets,
        strategy_statuses=STRATEGY_STATUSES,
        notice=request.args.get("strategy_notice"),
    )


@app.route("/strategies/save-code", methods=["POST"])
def save_strategy_code_route():
    """Saves an edit made in the Strategy Library's Code tab back to the
    same file -- always overwrite=True, since this is editing an EXISTING
    saved strategy in place (renaming to a different file is what
    /strategies/rename is for)."""
    strategy_type = request.form.get("strategy_type", "")
    filename = request.form.get("filename", "")
    code = request.form.get("code", "")
    try:
        save_strategy_text(code, filename, strategy_type, overwrite=True)
        return jsonify({"ok": True})
    except (ValueError, StrategyError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/strategies/delete", methods=["POST"])
def delete_saved_strategy_route():
    strategy_type = request.form.get("strategy_type", "")
    filename = request.form.get("filename", "")
    notice = f"Deleted '{filename}' from the {strategy_type} library."
    try:
        delete_saved_strategy(strategy_type, filename)
    except (ValueError, FileNotFoundError) as exc:
        notice = str(exc)
    return redirect(url_for(_redirect_target(request.form), strategy_notice=notice))


@app.route("/strategies/bulk-delete", methods=["POST"])
def bulk_delete_saved_strategies_route():
    """Delete every "type::filename" item checked in the library's bulk-
    manage panel in one request, instead of one delete round-trip each."""
    items = _parse_bulk_items(request.form.getlist("items"))
    target = _redirect_target(request.form)
    if not items:
        return redirect(url_for(target, strategy_notice="No strategies were selected to delete."))
    deleted, failed = delete_many(items)
    notice = f"Deleted {len(deleted)} strategy(ies)."
    if failed:
        notice += f" {len(failed)} failed: " + "; ".join(failed)
    return redirect(url_for(target, strategy_notice=notice))


def _parse_bulk_items(raw_items: list[str]) -> list[tuple[str, str]]:
    """Bulk checkboxes post as "type::filename" strings (see the
    bulk-manage checkboxes in index.html/search.html) -- parse and drop
    anything malformed rather than letting one bad value 500 the request."""
    items = []
    for raw in raw_items:
        parts = raw.split("::", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            items.append((parts[0], parts[1]))
    return items


@app.route("/strategies/bulk-export")
def bulk_export_saved_strategies_route():
    """Download only the checked "type::filename" items as a zip (bulk
    export a subset), via query string (?items=python::a.py&items=...) so
    it can be a plain link like the full-library export."""
    items = _parse_bulk_items(request.args.getlist("items"))
    if not items:
        return Response("No strategies were selected to export.", status=400, mimetype="text/plain")
    try:
        data = export_library_zip_bytes(selection=items)
    except FileNotFoundError as exc:
        return Response(str(exc), status=404, mimetype="text/plain")
    return Response(
        data,
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=t58_strategy_library_selection.zip"},
    )


@app.route("/strategies/view-code")
def view_saved_strategy_code_route():
    """Plain-text view of one saved strategy's source -- the web
    equivalent of the desktop app's VIEW CODE / CONFIG button. Opens in a
    new tab from the library panel rather than a modal, since a phone
    browser handles a plain page far better than in-page JS overlay text."""
    strategy_type = request.args.get("strategy_type", "")
    filename = request.args.get("filename", "")
    try:
        text = load_strategy_text(strategy_type, filename)
    except (ValueError, FileNotFoundError) as exc:
        return Response(str(exc), status=404, mimetype="text/plain")
    return Response(text, mimetype="text/plain")


@app.route("/strategies/batch-test", methods=["POST"])
def batch_test_saved_strategies_route():
    """Runs every checked "type::filename" library item through the same
    backtest -> prop-sim -> Monte Carlo -> report pipeline /run uses, one
    after another -- the web equivalent of the desktop app's TEST
    SELECTED (BATCH) button. Reuses the SAME market-data / risk / prop
    rule fields the main Run & Report form already carries (the page's JS
    clones them into this request), with one exception: a freshly
    uploaded CSV can't be cloned this way by a browser for security
    reasons, so batch-testing from the web currently requires picking an
    already-stored dataset rather than uploading a brand new file in the
    same click -- upload it once via Run & Report first, then batch-test
    against it."""
    form = request.form
    items = _parse_bulk_items(form.getlist("batch_items"))
    if not items:
        return redirect(url_for("index", strategy_notice="No strategies were checked to batch-test."))

    df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
    if dataset_error:
        return redirect(url_for("index", strategy_notice=dataset_error))

    risk = RiskConfig(
        initial_balance=float(form.get("initial_balance", 100000)),
        risk_mode=form.get("risk_mode", "percent"),
        risk_value=float(form.get("risk_value", 1.0)),
        max_trades_per_day=int(form.get("max_trades_day", 10)),
        commission_per_trade=float(form.get("commission", 0)),
        slippage_pips=float(form.get("slippage_pips", 0.5)),
        spread_pips=float(form.get("spread_pips", 1.0)),
        pip_size=float(form.get("pip_size", 0.0001)),
    )
    payout_cap = form.get("payout_cap", "").strip()
    rules = PropRules(
        account_size=float(form.get("account_size", 100000)),
        evaluation_profit_target_pct=float(form.get("profit_target", 8)),
        daily_loss_limit_pct=float(form.get("daily_loss", 5)),
        max_drawdown_pct=float(form.get("max_dd", 10)),
        drawdown_type=form.get("dd_type", "trailing"),
        drawdown_check_mode=form.get("dd_check_mode", "intrabar"),
        consistency_rule_pct=float(form.get("consistency", 30)) if form.get("consistency") else None,
        min_trading_days=int(form.get("min_days", 5)),
        payout_threshold_pct=float(form.get("payout_threshold", 0)),
        payout_cap_pct=float(payout_cap) if payout_cap else None,
        payout_frequency_days=int(form.get("payout_freq", 14)),
        required_buffer_pct=float(form.get("buffer", 0)),
    )
    n_sims = int(form.get("n_sims", 5000))
    mc_method = form.get("mc_method", "bootstrap")

    batch_items = []
    load_errors = []
    for strategy_type, filename in items:
        try:
            code = load_strategy_text(strategy_type, filename)
            strategy = build_strategy_from_code(strategy_type, code)
        except Exception as exc:  # noqa: BLE001 -- one bad strategy must not stop the batch
            load_errors.append(f"{filename}: {exc}")
            continue
        batch_items.append(BatchTestItem(label=filename, strategy=strategy, library_ref=(strategy_type, filename)))

    if not batch_items:
        return redirect(url_for("index", strategy_notice="Every checked strategy failed to load: " + "; ".join(load_errors)))

    run_id = uuid.uuid4().hex[:8]
    summary = run_batch_test(
        df, batch_items, risk, rules, REPORTS_DIR,
        instrument=active_label, mc_sims=min(n_sims, 50_000), mc_method=mc_method,
        basename_prefix=f"webbatch_{run_id}",
    )

    return render_template(
        "index.html",
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(),
        **_alpaca_template_context(),
        batch_result={
            "active_dataset": active_label,
            "import_note": import_note,
            "outcomes": [
                {
                    "label": o.label, "ok": o.ok, "reason": o.reason, "trades": o.trades,
                    "net_profit": o.net_profit, "eval_pass_probability": o.eval_pass_probability,
                    "report_html": f"/reports/{o.report_html.name}" if o.report_html else None,
                }
                for o in summary.outcomes
            ],
            "load_errors": load_errors,
        },
    )


@app.route("/strategies/rename", methods=["POST"])
def rename_saved_strategy_route():
    strategy_type = request.form.get("strategy_type", "")
    old_filename = request.form.get("old_filename", "")
    new_filename = (request.form.get("new_filename") or "").strip()
    overwrite = bool(request.form.get("overwrite"))
    target = _redirect_target(request.form)

    if not new_filename:
        return redirect(url_for(target, strategy_notice="Enter a new filename to rename to."))
    try:
        new_path = rename_saved_strategy(strategy_type, old_filename, new_filename, overwrite=overwrite)
        notice = f"Renamed '{old_filename}' to '{new_path.name}'."
    except StrategyAlreadyExists as exc:
        notice = (
            f"'{exc.filename}' already exists in the {strategy_type} library. "
            'Check "Overwrite" and rename again to replace it.'
        )
    except (ValueError, FileNotFoundError) as exc:
        notice = str(exc)
    return redirect(url_for(target, strategy_notice=notice))


@app.route("/strategies/metadata", methods=["POST"])
def save_strategy_metadata_route():
    strategy_type = request.form.get("strategy_type", "")
    filename = request.form.get("filename", "")
    description = (request.form.get("description") or "").strip()
    market = (request.form.get("market") or "").strip()
    tags_raw = (request.form.get("tags") or "").strip()
    target = _redirect_target(request.form)
    try:
        save_strategy_metadata(strategy_type, filename, {"description": description, "market": market})
        set_strategy_tags(strategy_type, filename, [t.strip() for t in tags_raw.split(",")] if tags_raw else [])
        notice = f"Saved info for '{filename}'."
    except ValueError as exc:
        notice = str(exc)
    return redirect(url_for(target, strategy_notice=notice))


@app.route("/strategies/status", methods=["POST"])
def set_strategy_status_route():
    strategy_type = request.form.get("strategy_type", "")
    filename = request.form.get("filename", "")
    status = request.form.get("status", "")
    target = _redirect_target(request.form)
    try:
        set_strategy_status(strategy_type, filename, status)
        notice = f"'{filename}' marked {status}."
    except ValueError as exc:
        notice = str(exc)
    return redirect(url_for(target, strategy_notice=notice))


@app.route("/strategies/export")
def export_strategy_library_route():
    """Download every saved strategy (all three languages, plus their
    metadata sidecars) as one zip -- the backup button, and also how a
    packaged .exe's library (which lives next to the .exe, not in the git
    repo) gets synced back into the repo: download, unzip into
    strategies/, commit."""
    data = export_library_zip_bytes()
    return Response(
        data,
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=t58_strategy_library_backup.zip"},
    )


@app.route("/manifest.json")
def manifest():
    return send_from_directory(app.static_folder, "manifest.json", mimetype="application/manifest+json")


@app.route("/run", methods=["POST"])
def run_pipeline():
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(request.form, request.files)
        if dataset_error:
            return render_template("index.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400

        form = request.form
        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)

        # FIX (2026-09-18): read reset_on_breach here (BEFORE the backtest
        # below runs) and set it on `risk` itself, not just on the
        # MonteCarloConfig further down -- see RiskConfig.reset_on_breach's
        # own docstring. Previously this checkbox was parsed only after
        # bt_result already existed, and only ever reached the post-hoc
        # Monte Carlo layer, so checking it never actually kept the raw
        # backtest itself trading past the first blown account.
        reset_on_breach = form.get("reset_on_breach") == "on"
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            max_trades_per_day=int(form.get("max_trades_day", 10)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
            reset_on_breach=reset_on_breach,
        )

        payout_cap = form.get("payout_cap", "").strip()
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
            drawdown_type=form.get("dd_type", "trailing"),
            drawdown_check_mode=form.get("dd_check_mode", "intrabar"),
            consistency_rule_pct=float(form.get("consistency", 30)) if form.get("consistency") else None,
            min_trading_days=int(form.get("min_days", 5)),
            payout_threshold_pct=float(form.get("payout_threshold", 0)),
            payout_cap_pct=float(payout_cap) if payout_cap else None,
            payout_frequency_days=int(form.get("payout_freq", 14)),
            required_buffer_pct=float(form.get("buffer", 0)),
        )

        # FIX (audit): unlike Quick Optimize/Full Pipeline/Evolution Lab/
        # Speed Run/Walk-Forward GA, Run & Report never reconciled its
        # RiskConfig against the active PropRules before backtesting --
        # see app.backtest.risk.with_prop_safety_defaults' own docstring.
        # Without this, the SAME strategy/data/settings could silently
        # backtest against a different account balance here than in Quick
        # Optimize or Full Pipeline (whenever initial_balance and
        # account_size aren't already identical), and this route's raw
        # backtest never enforced the account-blown/daily-loss circuit
        # breakers those other tools' backtests do -- producing a
        # genuinely different trade sequence, not just a differently
        # labeled one, for what looked like an apples-to-apples comparison.
        risk = with_prop_safety_defaults(risk, rules)        # Same "Enable adaptive, limit-aware position sizing" overlay Quick
        # Optimize/Full Pipeline/Evolution Lab already offer (see
        # app.backtest.adaptive_risk) -- Run & Report was previously the
        # only place in the app with no way to turn this on at all, which
        # made a Run & Report result silently non-comparable to a Quick
        # Optimize/Full Pipeline run made against the same strategy with
        # this enabled (see the 2026-09-16 Quick-Optimize-vs-Full-Pipeline
        # diagnosis this closes).
        adaptive_risk = build_limit_aware_preset(rules) if form.get("adaptive_risk_enabled") == "on" else None

        # T58 BACKTEST INTEGRITY CHECK -- pre-flight gate, BEFORE the backtest
        # itself runs (see app.validation.integrity_check's own docstring for
        # why: catches a corrupt dataset, a timeframe the data can't actually
        # support, or a confirmed lookahead leak up front, instead of only
        # ever being discoverable after the fact from a suspicious-looking
        # result). A BLOCKED verdict refuses to run the backtest at all.
        integrity_report = run_integrity_check(
            df, strategy, risk, prop_rules=rules,
            requested_timeframe=form.get("timeframe") or None,
            data_label=active_label,
        )
        if integrity_report.status == "BLOCKED":
            return render_template(
                "index.html", error=integrity_report.render(),
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context(),
            ), 400

        bt_result = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)
        if adaptive_risk is not None:
            # Surfaced in the report's own warnings section (not just the
            # server log) so the setting that produced these numbers is
            # visible on the report itself, not just reconstructible from
            # a log line someone has to remember to check -- exactly the
            # gap that made an earlier Full Pipeline report look
            # unreproducible (see the 2026-09-16 Quick-Optimize-vs-Full-
            # Pipeline diagnosis).
            bt_result.warnings.append(
                f"Adaptive risk was ENABLED for this run: {len(adaptive_risk.rules)} limit-aware "
                f"throttle rule(s) applied. A run of this same strategy/data with adaptive risk OFF "
                f"is not an apples-to-apples comparison against this report."
            )

        lookahead_warning = None
        if strategy.source_type == "python":
            try:
                lookahead_result = check_for_lookahead(strategy, df, max_signal_checkpoints=8)
                if lookahead_result.bug_detected:
                    lookahead_warning = lookahead_result.summary()
                if library_ref:
                    record_lookahead_result(*library_ref, {
                        "clean": not lookahead_result.bug_detected,
                        "summary": lookahead_result.summary(),
                    })
            except Exception:
                # Best-effort audit -- never let it block a run that would
                # otherwise succeed.
                pass

        trade_pnls = [t.pnl for t in bt_result.trades]
        trade_dates = [t.entry_time for t in bt_result.trades]
        single_run = simulate_account(trade_pnls, trade_dates, rules, reset_on_breach=reset_on_breach)

        if not bt_result.trades:
            msg = pipeline_guide.after_first_backtest({"trade_count": 0})
            return render_template("index.html", error=msg, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400

        n_sims = int(form.get("n_sims", 5000))
        mc_cfg = MonteCarloConfig(
            n_simulations=min(n_sims, 50_000), method=form.get("mc_method", "bootstrap"),
            reset_on_breach=reset_on_breach,
        )
        mc_result = run_monte_carlo(bt_result.trades, rules, mc_cfg)

        try:
            holdout_comparison = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2)
        except Exception:
            holdout_comparison = None

        run_id = uuid.uuid4().hex[:10]
        period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
        paths = generate_full_report(
            output_dir=REPORTS_DIR,
            strategy_name=bt_result.strategy_name,
            strategy_source_type=strategy.source_type,
            instrument=active_label,
            timeframe=infer_timeframe_label(df),
            backtest_period=period,
            backtest_result=bt_result,
            prop_rules=rules,
            prop_single_run=single_run,
            monte_carlo_result=mc_result,
            basename=f"report_{run_id}",
            holdout_comparison=holdout_comparison,
            risk_config=risk,
            price_df=df,
        )

        if library_ref:
            try:
                record_backtest_result(*library_ref, {
                    "trades": len(bt_result.trades),
                    "net_profit": round(bt_result.statistics.net_profit, 2),
                    "win_rate": round(bt_result.statistics.win_rate, 1),
                    "max_dd": round(bt_result.statistics.max_drawdown_pct, 2),
                    "passed_evaluation": single_run.passed_evaluation,
                    "report_html": f"/reports/{paths['html'].name}",
                })
            except (FileNotFoundError, ValueError):
                pass  # strategy was renamed/deleted mid-run -- not worth failing the response over

        return render_template(
            "index.html",
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            saved_strategies_json=_saved_strategies_json(),
            **_alpaca_template_context(),
            result={
                "active_dataset": active_label,
                "import_note": import_note,
                "lookahead_warning": lookahead_warning,
                "integrity_check": integrity_report.render(),
                "integrity_check_status": integrity_report.status,
                "trades": len(bt_result.trades),
                "net_profit": bt_result.statistics.net_profit,
                "win_rate": bt_result.statistics.win_rate,
                "max_dd": bt_result.statistics.max_drawdown_pct,
                "passed_eval": single_run.passed_evaluation,
                "reached_payout": single_run.reached_first_payout,
                "eval_pass_prob": mc_result.evaluation_pass_probability,
                "first_payout_prob": mc_result.first_payout_probability,
                "risk_of_ruin": mc_result.risk_of_ruin_pct,
                "expected_payout": mc_result.expected_payout,
                "n_sims": mc_result.n_simulations,
                "reset_on_breach": mc_result.reset_on_breach,
                "mean_attempts_per_path": mc_result.mean_attempts_per_path,
                "median_attempts_per_path": mc_result.median_attempts_per_path,
                "report_html": f"/reports/{paths['html'].name}",
                "report_json": f"/reports/{paths['json'].name}",
                "report_csv": f"/reports/{paths['summary_csv'].name}",
                "next_step": pipeline_guide.after_first_backtest(
                    bt_result.statistics.to_dict(), passed_evaluation=single_run.passed_evaluation,
                ),
            },
        )
    except StrategyError as exc:
        return render_template("index.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("index.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 500


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/api/global-search")
def api_global_search():
    """Backs the web app's own top-bar search box -- same shared
    app.search.global_search fan-out the desktop app's search box already
    uses (strategies, datasets, reports, runs), just returned as JSON
    instead of populating a Tkinter Listbox. Read-only/best-effort: a
    query error surfaces as an empty result list, never a 500, since
    search is a convenience feature layered on top of four existing
    sources.
    """
    from app.search.global_search import global_search

    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"query": query, "results": []})
    try:
        results = global_search(query, max_per_kind=8, reports_dir=REPORTS_DIR)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"query": query, "results": [], "error": str(exc)})

    out = []
    for r in results:
        url = None
        if r.kind == "report" and r.path:
            try:
                rel = Path(r.path).resolve().relative_to(REPORTS_DIR.resolve())
                url = f"/reports/{rel.as_posix()}"
            except ValueError:
                url = None
        out.append({
            "kind": r.kind, "title": r.title, "subtitle": r.subtitle, "url": url,
        })
    return jsonify({"query": query, "results": out})


def _account_settings_page_context(**extra):
    from app.accounts.app_info import APP_VERSION
    from app.accounts.settings import load_account_settings
    from app.accounts.subscription import LICENSE_STATUS_CHOICES, load_subscription
    ctx = dict(
        settings=load_account_settings(),
        subscription=load_subscription(),
        license_status_choices=LICENSE_STATUS_CHOICES,
        app_version=APP_VERSION,
        active_page="account_settings",
    )
    ctx.update(extra)
    return ctx


@app.route("/data-center")
def data_center():
    """Dataset coverage, timeframe availability, gaps, duplicates,
    timezone, sessions, bar counts, and per-file data health for
    everything under data/raw/ -- see app.data.health.compute_data_center.
    Placed in the Account nav (before Settings) per Owen's own request."""
    from app.data.health import compute_data_center
    report = compute_data_center()
    return render_template(
        "data_center.html", active_page="data_center", report=report,
        notice=request.args.get("notice"), notice_kind=request.args.get("notice_kind", "info"),
    )


@app.route("/data-center/import", methods=["POST"])
def data_center_import():
    """Same CSV/parquet import path every other upload form in this app
    already uses (app.data.importer.import_csv via store_csv_bytes) --
    the Data Center just gives it its own entry point so importing and
    reviewing data health live on the same page."""
    file = request.files.get("data_file")
    if not file or not file.filename:
        return redirect(url_for("data_center", notice="Choose a CSV or parquet file first.", notice_kind="error"))
    try:
        content = file.read()
        result = import_csv_bytes(content, filename=file.filename)
        if not result.is_valid:
            return redirect(url_for(
                "data_center", notice=f"Could not import '{file.filename}': {'; '.join(result.errors)}",
                notice_kind="error",
            ))
        store_csv_bytes(content, file.filename)
        return redirect(url_for("data_center", notice=f"Imported '{file.filename}'.", notice_kind="success"))
    except Exception as exc:  # noqa: BLE001
        return redirect(url_for("data_center", notice=f"Unexpected error importing '{file.filename}': {exc}", notice_kind="error"))


@app.route("/settings/account")
def account_settings_form():
    return render_template("account_settings.html", **_account_settings_page_context())


@app.route("/settings/api-keys")
def api_keys_settings_form():
    from app.ai.ollama_settings import load_settings as load_ollama_settings
    from app.data.alpaca_credentials import load_credentials as load_alpaca_credentials
    from app.accounts.api_keys import load_settings as load_api_keys_settings
    from app.live_deploy.live_settings import load_accounts
    from app.live_deploy.broker_registry import SUPPORTED_PLATFORMS

    alpaca_creds = load_alpaca_credentials()
    return render_template(
        "api_keys_settings.html",
        active_page="api_keys_settings",
        ollama=load_ollama_settings(),
        alpaca_api_key=(alpaca_creds.api_key if alpaca_creds else ""),
        alpaca_secret_key=(alpaca_creds.secret_key if alpaca_creds else ""),
        api_keys=load_api_keys_settings(),
        broker_accounts=load_accounts(),
        broker_platforms=SUPPORTED_PLATFORMS,
    )


@app.route("/settings/api-keys/save-ai", methods=["POST"])
def api_keys_save_ai():
    from app.ai.ollama_settings import OllamaSettings, save_settings as save_ollama_settings, load_settings as load_ollama_settings
    from app.accounts.api_keys import save_settings as save_api_keys_settings, load_settings as load_api_keys_settings
    form = request.form
    existing_ollama = load_ollama_settings()
    save_ollama_settings(OllamaSettings(
        enabled=form.get("ollama_enabled") == "on",
        host=(form.get("ollama_host") or existing_ollama.host).strip(),
        model=(form.get("ollama_model") or existing_ollama.model).strip(),
        api_key=(form.get("ollama_api_key") or existing_ollama.api_key),
        vision_model=(form.get("ollama_vision_model") or existing_ollama.vision_model).strip(),
    ))
    existing_keys = load_api_keys_settings()
    save_api_keys_settings(existing_keys.__class__(
        fred_api_key=existing_keys.fred_api_key,
        openai_api_key=(form.get("openai_api_key") or existing_keys.openai_api_key),
        claude_api_key=(form.get("claude_api_key") or existing_keys.claude_api_key),
        london_strategic_edge_key=existing_keys.london_strategic_edge_key,
    ))
    return redirect(url_for("api_keys_settings_form"))


@app.route("/settings/api-keys/save-trading", methods=["POST"])
def api_keys_save_trading():
    from app.data.alpaca_credentials import save_credentials as save_alpaca_credentials, load_credentials as load_alpaca_credentials
    form = request.form
    existing = load_alpaca_credentials()
    api_key = (form.get("alpaca_api_key") or (existing.api_key if existing else "")).strip()
    secret_key = (form.get("alpaca_secret_key") or (existing.secret_key if existing else "")).strip()
    if api_key and secret_key:
        save_alpaca_credentials(api_key, secret_key)
    return redirect(url_for("api_keys_settings_form"))


@app.route("/settings/api-keys/save-data", methods=["POST"])
def api_keys_save_data():
    from app.accounts.api_keys import save_settings as save_api_keys_settings, load_settings as load_api_keys_settings
    form = request.form
    existing = load_api_keys_settings()
    save_api_keys_settings(existing.__class__(
        fred_api_key=(form.get("fred_api_key") or existing.fred_api_key),
        openai_api_key=existing.openai_api_key,
        claude_api_key=existing.claude_api_key,
        london_strategic_edge_key=(form.get("london_strategic_edge_key") or existing.london_strategic_edge_key),
    ))
    return redirect(url_for("api_keys_settings_form"))


@app.route("/settings/api-keys/brokers/save", methods=["POST"])
def api_keys_save_broker():
    from app.live_deploy.live_settings import LiveAccount, save_account
    form = request.form
    extra_keys = ["client_id", "client_secret", "refresh_token", "ctid_trader_account_id", "host",
                  "app_id", "app_secret", "cid", "sec", "is_live", "base_url", "environment"]
    extra = {k: form.get(k, "").strip() for k in extra_keys if form.get(k, "").strip()}
    account = LiveAccount(
        id=form.get("account_id") or None,
        nickname=(form.get("nickname") or "").strip(),
        firm_name=(form.get("firm_name") or "").strip(),
        platform=form.get("platform", "MT4/MT5"),
        login=(form.get("login") or "").strip(),
        server=(form.get("server") or "").strip(),
        password=(form.get("password") or "").strip(),
        terminal_path=(form.get("terminal_path") or "").strip(),
        extra_credentials=extra,
    )
    save_account(account)
    return redirect(url_for("api_keys_settings_form"))


@app.route("/settings/api-keys/brokers/delete", methods=["POST"])
def api_keys_delete_broker():
    from app.live_deploy.live_settings import delete_account
    account_id = request.form.get("account_id", "")
    if account_id:
        delete_account(account_id)
    return redirect(url_for("api_keys_settings_form"))


@app.route("/settings/api-keys/test", methods=["POST"])
def api_keys_test_connection():
    """Test Connection for every category. Prefers whatever is CURRENTLY
    TYPED in the form (the page now submits its own field values alongside
    `service` -- see testService()/testBroker() in api_keys_settings.html),
    falling back to whatever is already SAVED for any field left blank --
    a blank password field means "keep the saved secret" everywhere else
    on this page, so Test Connection honors that same convention rather
    than treating blank as "no key". This lets a freshly pasted key/host
    be tested immediately, without requiring Save first, while a bare
    {"service": ...} request (no other fields present) still falls back
    to the saved settings exactly as before."""
    service = request.form.get("service", "")

    if service == "ollama":
        from app.ai.ollama_client import OllamaClient
        from app.ai.ollama_settings import OllamaSettings
        from app.ai.ollama_settings import load_settings as load_ollama_settings
        saved = load_ollama_settings()
        if "ollama_host" in request.form:
            settings = OllamaSettings(
                enabled="ollama_enabled" in request.form,
                host=request.form.get("ollama_host", "").strip() or saved.host,
                model=request.form.get("ollama_model", "").strip() or saved.model,
                api_key=request.form.get("ollama_api_key", "").strip() or saved.api_key,
                vision_model=saved.vision_model,
            )
        else:
            settings = saved
        if not settings.is_usable:
            return jsonify({"ok": False, "error": "Ollama isn't enabled, or no host is set."})
        ok, message = OllamaClient(settings).test_connection()
        return jsonify({"ok": ok, "message": message} if ok else {"ok": False, "error": message})

    if service == "alpaca":
        from app.data.alpaca_source import test_connection as test_alpaca
        from app.data.alpaca_credentials import load_credentials as load_alpaca_credentials
        saved = load_alpaca_credentials()
        api_key = request.form.get("alpaca_api_key", "").strip() or (saved.api_key if saved else "")
        secret_key = request.form.get("alpaca_secret_key", "").strip() or (saved.secret_key if saved else "")
        if not api_key or not secret_key:
            return jsonify({"ok": False, "error": "No Alpaca credentials saved yet."})
        try:
            message = test_alpaca(api_key, secret_key)
            return jsonify({"ok": True, "message": message})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)})

    if service == "fred":
        from app.accounts.api_keys import load_settings as load_api_keys_settings
        key = request.form.get("fred_api_key", "").strip() or load_api_keys_settings().fred_api_key
        if not key:
            return jsonify({"ok": False, "error": "No FRED API key saved yet."})
        try:
            import urllib.request
            url = f"https://api.stlouisfed.org/fred/series?series_id=GDP&api_key={key}&file_type=json"
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status == 200:
                    return jsonify({"ok": True, "message": "FRED API key is valid."})
            return jsonify({"ok": False, "error": f"Unexpected status {resp.status}."})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"FRED rejected the key or is unreachable: {exc}"})

    if service in ("openai", "claude"):
        from app.accounts.api_keys import load_settings as load_api_keys_settings
        keys = load_api_keys_settings()
        saved_key = keys.openai_api_key if service == "openai" else keys.claude_api_key
        key = request.form.get(f"{service}_api_key", "").strip() or saved_key
        if not key:
            return jsonify({"ok": False, "error": f"No {service} API key saved yet."})
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
            with urllib.request.urlopen(req, timeout=10) as resp:
                return jsonify({"ok": True, "message": f"{service.title()} key is valid."})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"{service.title()} rejected the key or is unreachable: {exc}"})

    if service == "london_strategic_edge":
        from app.accounts.api_keys import load_settings as load_api_keys_settings
        keys = load_api_keys_settings()
        key = request.form.get("london_strategic_edge_key", "").strip() or keys.london_strategic_edge_key
        if not key:
            return jsonify({"ok": False, "error": "No London Strategic Edge API key saved yet."})
        try:
            message = lse_test_connection(key)
            return jsonify({"ok": True, "message": message})
        except (LSEImportError, LSEFetchError) as exc:
            return jsonify({"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"Could not connect to London Strategic Edge: {exc}"})

    if service == "broker":
        from app.live_deploy.live_settings import load_accounts
        from app.live_deploy.broker_registry import build_adapter
        account_id = request.form.get("account_id", "")
        account = next((a for a in load_accounts() if a.id == account_id), None)
        if account is None:
            return jsonify({"ok": False, "error": "Account not found."})
        try:
            adapter = build_adapter(account)
            result = adapter.connect()
            if result.ok:
                adapter.disconnect()
            return jsonify({"ok": result.ok, "message": result.message} if result.ok else {"ok": False, "error": result.message})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)})

    return jsonify({"ok": False, "error": f"Unknown service '{service}'."})


@app.route("/support")
def support_page():
    return render_template("support.html", active_page="support")


# UPGRADE (Sep 2026 Start Here pass): one dedicated "Start Here" landing
# page per top-level sidebar section, replacing the old model of a small
# collapsible card embedded at the top of ONE tool's own page (Forge,
# Search Lab, Evolution Lab, Full Pipeline) with a section-wide page that
# actually orients someone to everything in that group of tabs, not just
# whichever one tool happened to have the card. Optimize and Validate
# already had their own hub/picker pages (see optimize_hub_form /
# validate_hub_form below) -- those now open with the same "what this
# section is for" framing rather than getting a second, separate page.
_SECTION_START_HERE = {
    "create": {
        "title": "Create", "accent_color": "#a78bfa",
        "tagline": "Everything that produces a new strategy, from a blank page to a candidate worth testing.",
        "description": (
            "Create is where a strategy comes from in the first place -- either generated broadly by the "
            "system (Forge, Speed Run, Research Agent/Director/Loop, Generate Strategies) when you don't yet "
            "have a specific idea, or built directly (Strategy Library) when you do. Nothing here is validated "
            "yet -- a strategy that comes out of Create still needs to go through Test and Optimize before "
            "it's trustworthy."
        ),
        "roadmap": [
            "No idea yet? Start with Forge Strategy -- one button, generates and screens thousands of "
            "hypotheses, and validates survivors for prop-firm survival.",
            "Want it faster and narrower? Speed Run does a quicker single-pass version of the same idea.",
            "Have a specific idea already? Build or edit it directly in the Strategy Library.",
            "Want an LLM to reason about market structure and propose ideas? Try Research Agent, Research "
            "Director, or the background Research Loop.",
            "Once you have a survivor you like, send it to Search Lab or Evolution Lab (under Optimize) to "
            "sharpen its parameters, or straight to Full Pipeline for a validated verdict.",
        ],
        "primary_buttons": [{"label": "Forge Strategy", "href": "/forge"}, {"label": "Strategy Library", "href": "/library"}],
        "tools": [
            {"name": "Forge Strategy", "href": "/forge", "desc": "One-button broad generate + screen + prop-survival check."},
            {"name": "Generate Strategies (AI)", "href": "/generate-strategies", "desc": "LLM-assisted strategy code generation."},
            {"name": "Research Agent", "href": "/research-agent", "desc": "LLM reasons over market structure for ideas."},
            {"name": "Research Director", "href": "/research", "desc": "Directs a deeper, multi-step research pass."},
            {"name": "Research Loop (background)", "href": "/research-loop", "desc": "Keeps researching unattended, on a schedule."},
            {"name": "Speed Run", "href": "/speed-run", "desc": "Faster, narrower single-pass version of Forge."},
            {"name": "Multi-Instrument Speed Run", "href": "/speed-run/multi-instrument", "desc": "Speed Run across several datasets at once."},
            {"name": "Strategy Library", "href": "/library", "desc": "Build, edit, or browse strategies directly."},
        ],
    },
    "test": {
        "title": "Test", "accent_color": "#22d3ee",
        "tagline": "Run one specific strategy against one dataset and read the report.",
        "description": (
            "Test is the manual, single-run path: you already have a strategy and a dataset in mind, and you "
            "want to run a backtest and read the resulting report -- trades, equity curve, drawdown, prop-firm "
            "pass/payout probability -- without any search or optimization happening. Use Optimize instead once "
            "you want the system to search for better parameters rather than just report on the ones you gave it."
        ),
        "roadmap": [
            "Pick a strategy (from the Strategy Library, or configure one directly) and a dataset.",
            "Set your prop-firm rules and risk/execution settings.",
            "Run & Report gives you the full backtest report for that exact configuration.",
            "Payout Probability adds Monte Carlo-based odds of actually reaching a funded payout.",
            "Not sure which prop firm's rules to test against? Prop-Firm Recommender narrows it down.",
        ],
        "primary_buttons": [{"label": "Run & Report", "href": "/"}],
        "tools": [
            {"name": "Run & Report", "href": "/", "desc": "One strategy, one dataset, one full backtest report."},
            {"name": "Payout Probability", "href": "/payout-probability", "desc": "Monte Carlo odds of reaching a funded payout."},
            {"name": "Prop-Firm Recommender", "href": "/prop-firm-recommender", "desc": "Find which prop firm's rules fit a strategy best."},
        ],
    },
    "champion": {
        "title": "Champion", "accent_color": "#e879f9",
        "tagline": "Once you have strong individual strategies, this is where they're combined into something sturdier.",
        "description": (
            "Champion is about combining strength, not finding it from scratch: it takes strategies that already "
            "did well in Optimize/Validate and asks whether running several together (an ensemble), across "
            "several assets (a portfolio), or from genuinely different families (diversity) produces something "
            "more robust than any single strategy alone."
        ),
        "roadmap": [
            "Check Family Diversity first -- confirms your surviving strategies aren't all secretly the same idea.",
            "Multi-Asset Portfolio combines strategies across different instruments.",
            "Multi-Strategy Ensemble combines several strategies on the same instrument into one blended system.",
        ],
        "primary_buttons": [{"label": "Multi-Strategy Ensemble", "href": "/ensemble"}],
        "tools": [
            {"name": "Family Diversity", "href": "/family-diversity", "desc": "Confirms survivors aren't all the same underlying idea."},
            {"name": "Multi-Asset Portfolio", "href": "/portfolio", "desc": "Combine strategies across different instruments."},
            {"name": "Multi-Strategy Ensemble", "href": "/ensemble", "desc": "Blend several strategies into one combined system."},
        ],
    },
    "deployment": {
        "title": "Deployment", "accent_color": "#a3e635",
        "tagline": "Champion checks, then forward-testing and live monitoring once you're ready to actually trade a strategy.",
        "description": (
            "Deployment covers everything after a strategy is validated: ongoing champion checks (Overnight "
            "Autopilot, Compare, Strategy Health) that keep watching a strategy over time, and the live-markets "
            "path (Forward Test, Deploy Live, Monitor, Interactive Replay) for actually running it against a "
            "real or demo broker account. Deploy Live risks real money -- read its own page carefully before "
            "using it."
        ),
        "roadmap": [
            "Overnight Autopilot and Compare Strategies keep validating a champion over time as new data arrives.",
            "Add a broker/prop-firm account under Settings \u2192 API Keys \u2192 Brokers/Prop Firms.",
            "Forward Test runs a strategy against a demo/paper account first -- always do this before Deploy Live.",
            "Deploy Live trades real money -- read the warnings on that page first.",
            "Monitor (Live Market) and Interactive Replay let you watch what a running or historical session actually did.",
        ],
        "primary_buttons": [{"label": "Forward Test (MT5)", "href": "/forward-test"}],
        "tools": [
            {"name": "Overnight Autopilot", "href": "/overnight-autopilot", "desc": "Keeps re-checking a champion strategy unattended."},
            {"name": "Compare Strategies", "href": "/compare", "desc": "Side-by-side comparison of two or more strategies."},
            {"name": "Strategy Health / Auto Re-tune", "href": "/quant-lab/strategy-health", "desc": "Watches for a strategy's edge decaying over time."},
            {"name": "Forward Test (MT5)", "href": "/forward-test", "desc": "Demo/paper-account forward test -- do this before Deploy Live."},
            {"name": "Deploy Live", "href": "/deploy-live", "desc": "Real-money live trading -- read the warnings first."},
            {"name": "Monitor (Live Market)", "href": "/live-market", "desc": "Watch a running live/forward-test session."},
            {"name": "Interactive Replay", "href": "/replay", "desc": "Step bar-by-bar through a historical session."},
        ],
    },
    "graveyard": {
        "title": "Strategy Graveyard", "accent_color": "#c9d1d9",
        "tagline": "Every rejected strategy, and exactly why it failed -- so you don't re-test the same dead idea twice.",
        "description": (
            "Every strategy Forge, Search Lab, or Evolution Lab rejects gets recorded here with the specific "
            "reason it failed, per instrument -- not raw profit, but pass/payout probability and prop-firm "
            "survival. Worth checking before a new search: if a whole family keeps dying the same way on a "
            "given instrument, that's a pattern worth knowing before you spend more search budget on it."
        ),
        "roadmap": [
            "After a Forge/Search Lab/Evolution Lab run, check here for what got rejected and why.",
            "Look for repeated failure patterns within one family/instrument before re-running a similar search.",
        ],
        "primary_buttons": [{"label": "Open Strategy Graveyard", "href": "/graveyard"}],
        "tools": [
            {"name": "Strategy Graveyard", "href": "/graveyard", "desc": "Dead neighborhoods, and exactly why they failed."},
        ],
    },
    "quantlab": {
        "title": "Quant Lab", "accent_color": "#67e8f9",
        "tagline": "Standalone quant tools that sit outside the main Create \u2192 Deployment strategy pipeline.",
        "description": (
            "Quant Lab, Options Outlook, and Hedge Fund Manager are each self-contained -- a strategy-code "
            "translator, statistical-arbitrage and options tooling, an options calls/puts outlook, and a "
            "higher-level research \u2192 portfolio \u2192 execution \u2192 oversight workflow. None of them require "
            "having gone through Create/Test/Optimize/Validate first."
        ),
        "roadmap": [
            "Quant Lab covers strategy translation, statistical arbitrage, and other one-off quant tools.",
            "Options Outlook gives a calls/puts read on an instrument.",
            "Hedge Fund Manager is a separate, higher-level research/portfolio/execution/oversight workflow.",
        ],
        "primary_buttons": [{"label": "Open Quant Lab", "href": "/quant-lab"}],
        "tools": [
            {"name": "Quant Lab", "href": "/quant-lab", "desc": "Strategy translator, stat arb, and other quant tools."},
            {"name": "Options Outlook", "href": "/options-outlook", "desc": "Calls/puts outlook for an instrument."},
            {"name": "Hedge Fund Manager", "href": "/hedge-fund", "desc": "Research \u2192 portfolio \u2192 execution \u2192 oversight."},
        ],
    },
    "account": {
        "title": "Account", "accent_color": "#c9d1d9",
        "tagline": "Your profile, every integration's API keys in one place, notifications, and support.",
        "description": (
            "Account settings, one centralized place for every integration's credentials (AI, Trading, Data, "
            "and Broker/Prop-Firm accounts), where job-finished notifications go, and where to get help or "
            "report an issue -- all local to this app, no cloud account required."
        ),
        "roadmap": [
            "Set your name/email/company under Account Settings.",
            "Add every API key/broker account you'll use anywhere in the app under API Keys, once.",
            "Turn on email/Discord/Telegram notifications so a long-running job can tell you when it's done.",
            "Stuck on something? Support has the FAQ, Discord link, and an issue-report form.",
        ],
        "primary_buttons": [{"label": "API Keys", "href": "/settings/api-keys"}, {"label": "Account Settings", "href": "/settings/account"}],
        "tools": [
            {"name": "Account Settings", "href": "/settings/account", "desc": "Name, username, email, company, security."},
            {"name": "API Keys", "href": "/settings/api-keys", "desc": "Every integration's credentials in one place."},
            {"name": "Notification Settings", "href": "/settings/notifications", "desc": "Email, Discord, Telegram job-finished alerts."},
            {"name": "Support", "href": "/support", "desc": "FAQ, Discord, and issue reporting."},
        ],
    },
}


@app.route("/start-here/<section>")
def section_start_here(section):
    data = _SECTION_START_HERE.get(section)
    if data is None:
        return render_template("support.html", active_page="support"), 404
    return render_template(
        "section_start_here.html",
        active_page_key=f"start_here_{section}",
        section_title=data["title"], tagline=data["tagline"], accent_color=data["accent_color"],
        description=data["description"], roadmap=data.get("roadmap", []),
        primary_buttons=data.get("primary_buttons", []), tools=data.get("tools", []),
    )


@app.route("/support/report-issue", methods=["POST"])
def support_report_issue():
    """No external support backend exists for this app (see support.html's
    own copy) -- this just appends to a local JSONL log so an issue typed
    here isn't lost, for the person to paste elsewhere themselves later."""
    title = (request.form.get("title") or "").strip()
    body = (request.form.get("body") or "").strip()
    if not title and not body:
        return jsonify({"ok": False, "error": "Nothing to save."}), 400
    log_dir = get_app_base_dir() / "data" / "config"
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "title": title, "body": body,
    }
    with open(log_dir / "issue_reports.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return jsonify({"ok": True})


@app.route("/settings/account/save", methods=["POST"])
def account_settings_save():
    from app.accounts.settings import load_account_settings, save_account_settings
    form = request.form
    settings = load_account_settings()  # keep password_hash untouched -- Change Password below is the only way to alter it
    settings.display_name = (form.get("display_name") or "").strip()
    settings.username = (form.get("username") or "").strip()
    settings.email = (form.get("email") or "").strip()
    settings.company = (form.get("company") or "").strip()
    save_account_settings(settings)
    return render_template("account_settings.html", **_account_settings_page_context(saved="profile"))


@app.route("/settings/account/profile-picture", methods=["POST"])
def account_settings_profile_picture():
    from app.accounts.settings import load_account_settings, save_account_settings
    file = request.files.get("profile_picture")
    if file is None or not file.filename:
        return redirect(url_for("account_settings_form"))
    ext = Path(file.filename).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        return render_template(
            "account_settings.html",
            **_account_settings_page_context(error="Profile picture must be a PNG, JPG, GIF, or WEBP image."),
        ), 400

    settings = load_account_settings()
    # One fixed filename per extension slot -- overwrite in place rather
    # than accumulate a new file per upload, and remove any OTHER
    # extension's leftover file so switching from a .png to a .jpg doesn't
    # leave both being served depending on which route someone hits.
    pic_dir = get_app_base_dir() / "data" / "config"
    pic_dir.mkdir(parents=True, exist_ok=True)
    for other_ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        (pic_dir / f"profile_picture{other_ext}").unlink(missing_ok=True)
    filename = f"profile_picture{ext}"
    file.save(pic_dir / filename)
    settings.profile_picture_filename = filename
    save_account_settings(settings)
    return render_template("account_settings.html", **_account_settings_page_context(saved="profile"))


@app.route("/settings/account/profile-picture/remove", methods=["POST"])
def account_settings_profile_picture_remove():
    from app.accounts.settings import load_account_settings, save_account_settings
    settings = load_account_settings()
    if settings.profile_picture_filename:
        (get_app_base_dir() / "data" / "config" / settings.profile_picture_filename).unlink(missing_ok=True)
        settings.profile_picture_filename = ""
        save_account_settings(settings)
    return render_template("account_settings.html", **_account_settings_page_context(saved="profile"))


@app.route("/settings/account/profile-picture/file")
def account_settings_profile_picture_file():
    from app.accounts.settings import load_account_settings
    settings = load_account_settings()
    if not settings.profile_picture_filename:
        return "", 404
    return send_from_directory(get_app_base_dir() / "data" / "config", settings.profile_picture_filename)


@app.route("/settings/account/change_password", methods=["POST"])
def account_settings_change_password():
    from app.accounts.settings import hash_password, load_account_settings, save_account_settings
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")
    if new_password != confirm_password:
        return render_template("account_settings.html", **_account_settings_page_context(
            password_error="New password and confirmation didn't match -- nothing was changed.",
        )), 400
    settings = load_account_settings()
    settings.password_hash = hash_password(new_password)  # empty new_password -> hash_password("") -> "" -> lock removed
    save_account_settings(settings)
    session["t58_unlocked"] = True  # whoever just set/changed it is, by definition, already "in"
    message = "Local app lock password set." if new_password else "Local app lock password cleared -- this app no longer asks for one."
    return render_template("account_settings.html", **_account_settings_page_context(saved="password", password_message=message))


@app.route("/settings/account/logout", methods=["POST"])
def account_settings_logout():
    session.pop("t58_unlocked", None)
    from app.accounts.settings import load_account_settings
    if load_account_settings().has_password:
        return redirect(url_for("account_lock_form"))
    return render_template("account_settings.html", **_account_settings_page_context(saved="logout"))


@app.route("/settings/account/delete", methods=["POST"])
def account_settings_delete():
    """Clears all locally stored profile/lock fields (Account Settings'
    own JSON file) back to defaults. Deliberately does NOT touch
    subscription.json, strategies, datasets, or reports -- \"delete
    account\" here means \"forget my local profile + lock\", the only
    thing this app actually has an \"account\" for (see this module's own
    docstring on the app having no server-side account system)."""
    from app.accounts.settings import AccountSettings, save_account_settings
    save_account_settings(AccountSettings())
    session.pop("t58_unlocked", None)
    return render_template("account_settings.html", **_account_settings_page_context(saved="deleted"))


@app.route("/settings/account/subscription/save", methods=["POST"])
def account_settings_subscription_save():
    from app.accounts.subscription import SubscriptionInfo, LICENSE_STATUS_CHOICES, save_subscription
    form = request.form
    status = (form.get("license_status") or "unset").strip()
    if status not in LICENSE_STATUS_CHOICES:
        status = "unset"
    info = SubscriptionInfo(
        plan=(form.get("plan") or "").strip(),
        license_key=(form.get("license_key") or "").strip(),
        license_status=status,
        renewal_date=(form.get("renewal_date") or "").strip(),
    )
    save_subscription(info)
    return render_template("account_settings.html", **_account_settings_page_context(saved="subscription"))


@app.route("/settings/account/subscription/check_key", methods=["POST"])
def account_settings_check_license_key():
    from app.accounts.subscription import check_license_key_format
    looks_valid, message = check_license_key_format(request.form.get("license_key", ""))
    return render_template("account_settings.html", **_account_settings_page_context(license_check_message=message, license_check_ok=looks_valid))


@app.route("/settings/account/check_updates", methods=["POST"])
def account_settings_check_updates():
    from app.accounts.app_info import check_for_updates
    return render_template("account_settings.html", **_account_settings_page_context(update_check=check_for_updates()))


@app.route("/settings/notifications")
def notification_settings_form():
    return render_template("notification_settings.html", settings=load_notification_settings(), active_page="notification_settings")


@app.route("/settings/notifications/save", methods=["POST"])
def notification_settings_save():
    form = request.form
    existing = load_notification_settings()
    settings = NotificationSettings(
        notify_email=(form.get("notify_email") or "").strip(),
        notify_phone=(form.get("notify_phone") or "").strip(),
        smtp_host=(form.get("smtp_host") or "").strip(),
        smtp_port=int(form.get("smtp_port", 587) or 587),
        smtp_username=(form.get("smtp_username") or "").strip(),
        # Blank password on save means "keep the existing one" -- so
        # re-saving the email address doesn't force retyping the SMTP
        # password (e.g. a Gmail app password) every time.
        smtp_password=(form.get("smtp_password") or existing.smtp_password),
        smtp_from=(form.get("smtp_from") or "").strip(),
        email_enabled=form.get("email_enabled") == "on",
        discord_webhook_url=(form.get("discord_webhook_url") or "").strip(),
        # Same "blank means keep the existing one" rule as smtp_password --
        # a bot token is a secret, re-typing it on every unrelated save
        # (e.g. just adding an email address) would be needless friction.
        telegram_bot_token=(form.get("telegram_bot_token") or existing.telegram_bot_token),
        telegram_chat_id=(form.get("telegram_chat_id") or "").strip(),
    )
    save_notification_settings(settings)
    return render_template(
        "notification_settings.html", settings=settings, active_page="notification_settings",
        saved=True,
    )


@app.route("/settings/notifications/test", methods=["POST"])
def notification_settings_test():
    """Test Connection for Discord/Telegram -- fires an actual test
    message through send_job_notification/_send_telegram_notification's
    exact same code path a real job completion uses, using whatever is
    CURRENTLY SAVED (not the unsaved form fields, since a webhook POST or
    bot-API call needs the real secret, and a bare 'test' click shouldn't
    require Save first if nothing changed). Reports success only as
    'request sent' -- these channels are deliberately fire-and-forget
    (see their own docstrings), so this can't confirm delivery, only that
    the request didn't fail synchronously."""
    channel = request.form.get("channel", "")
    settings = load_notification_settings()
    if channel == "discord":
        if not settings.discord_is_usable:
            return jsonify({"ok": False, "error": "No Discord webhook URL saved yet."})
        try:
            import urllib.request
            payload = json.dumps({"content": "T58 -- test notification. If you see this, Discord is connected."}).encode("utf-8")
            req = urllib.request.Request(
                settings.discord_webhook_url.strip(), data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            return jsonify({"ok": True, "message": "Test message sent to Discord."})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"Discord webhook request failed: {exc}"})
    elif channel == "telegram":
        if not settings.telegram_is_usable:
            return jsonify({"ok": False, "error": "Telegram needs both a bot token and a chat ID saved."})
        try:
            import urllib.request
            import urllib.parse
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token.strip()}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": settings.telegram_chat_id.strip(),
                "text": "T58 -- test notification. If you see this, Telegram is connected.",
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            resp = urllib.request.urlopen(req, timeout=10)
            body = json.loads(resp.read().decode("utf-8"))
            if not body.get("ok"):
                return jsonify({"ok": False, "error": f"Telegram API rejected the request: {body.get('description', 'unknown error')}"})
            return jsonify({"ok": True, "message": "Test message sent to Telegram."})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"Telegram request failed: {exc}"})
    elif channel == "email":
        if not settings.is_usable:
            return jsonify({"ok": False, "error": "Email notifications aren't fully configured yet."})
        _send_email_notification(settings, "Test Notification", "connection test", None)
        return jsonify({"ok": True, "message": "Test email queued -- check your inbox in a moment."})
    return jsonify({"ok": False, "error": f"Unknown channel '{channel}'."})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Live Market -- a real-time (or best-available) candlestick chart backed by
# TradingView's Lightweight Charts library, fed from whichever data source
# is actually available: a connected MT5 demo terminal, saved Alpaca keys,
# or (with neither) a steady bar-by-bar replay of an already-imported CSV so
# the page always has something real to draw.
# ---------------------------------------------------------------------------

@app.route("/live-market")
def live_market_page():
    try:
        mt5 = live_market.mt5_status()
        has_alpaca = bool(alpaca_credentials.load_credentials())
        replay_datasets = live_market.list_replay_datasets()
        return render_template(
            "live_market.html",
            mt5_status=mt5,
            has_alpaca=has_alpaca,
            replay_datasets=replay_datasets,
            # Grouped by instrument (NQ1!, ES1!, ...) same as every other
            # "stored dataset" picker in this app (Replay, Search, etc.) --
            # see list_datasets_by_instrument() in app/data/storage.py.
            dataset_groups=list_datasets_by_instrument(),
            timeframe_choices=live_market.TIMEFRAME_CHOICES_MINUTES,
            theme=request.args.get("theme", "dark"),
            initial_symbol=request.args.get("symbol") or mt5["default_symbol"] or "XAUUSD",
            initial_timeframe=request.args.get("timeframe", type=int) or mt5["default_timeframe_minutes"] or 15,
            initial_source=request.args.get("source") or ("mt5" if mt5["configured"] else ("alpaca" if has_alpaca else "replay")),
        )
    except Exception as exc:  # noqa: BLE001 -- a broken data source must never take the whole page down
        return render_template(
            "live_market.html",
            mt5_status={"available": False, "configured": False, "connected": False,
                        "default_symbol": "XAUUSD", "default_timeframe_minutes": 15},
            has_alpaca=False, replay_datasets=[], dataset_groups=[], timeframe_choices=live_market.TIMEFRAME_CHOICES_MINUTES,
            theme=request.args.get("theme", "dark"), initial_symbol="XAUUSD", initial_timeframe=15,
            initial_source="replay", load_error=str(exc),
        )


@app.route("/api/live-market/status")
def api_live_market_status():
    try:
        return jsonify({
            "mt5": live_market.mt5_status(),
            "alpaca_configured": bool(alpaca_credentials.load_credentials()),
            "replay_datasets": live_market.list_replay_datasets(),
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"mt5": {"available": False, "configured": False, "connected": False},
                         "alpaca_configured": False, "replay_datasets": [], "error": str(exc)}), 200


@app.route("/api/live-market/bars")
def api_live_market_bars():
    source = request.args.get("source", "replay")
    symbol = request.args.get("symbol", "")
    timeframe = request.args.get("timeframe", 15, type=int) or 15
    seed = request.args.get("seed", "0") == "1"  # first request for this symbol/timeframe this page-load

    try:
        if source == "mt5":
            bars = live_market.fetch_mt5_bars(symbol, timeframe)
            status = "live" if bars else ("connecting" if live_market.mt5_status()["configured"] else "unavailable")
        elif source == "alpaca":
            asset_class = request.args.get("asset_class", "Stock")
            bars = live_market.fetch_alpaca_bars(symbol, asset_class, timeframe)
            status = "delayed" if bars else "unavailable"
        else:
            bars, finished = live_market.fetch_replay_bars(symbol, advance=not seed)
            status = "replay-finished" if finished else "replay"
        return jsonify({"bars": bars, "status": status})
    except Exception as exc:  # noqa: BLE001 -- any data-source hiccup must surface as UNAVAILABLE, never a 500
        return jsonify({"bars": [], "status": "unavailable", "error": str(exc)}), 200


@app.route("/api/live-market/trades")
def api_live_market_trades():
    symbol = request.args.get("symbol", "")
    try:
        return jsonify({"markers": live_market.recent_trade_markers(symbol)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"markers": [], "error": str(exc)}), 200


# ---------------------------------------------------------------------------
# Search Lab (Stages 1-5) -- same engine as the desktop app's Search Lab tab.
#
# A search run can take anywhere from ~10 seconds to several minutes
# (hundreds of candidates x GA refinement x full Monte Carlo x walk-forward
# x robustness), which is too long to hold open a single HTTP request/response
# for. So this runs the search in a background thread and hands the browser
# a job_id immediately; the job status page polls a small JSON endpoint
# every couple of seconds and renders the live log + leaderboard once done --
# the same "kick off a background job, poll for status" shape any long job
# on the web needs, just backed by a plain in-memory dict since this app is
# a single-user local/LAN tool (same trust model as the rest of this server).
# ---------------------------------------------------------------------------

_SEARCH_JOBS: dict[str, dict] = {}
_SEARCH_JOBS_LOCK = threading.Lock()


def _job_log(job_id: str, msg: str) -> None:
    with _SEARCH_JOBS_LOCK:
        job = _SEARCH_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_search_job(
    job_id: str, df, risk: RiskConfig, rules: PropRules, space, stage_cfg: SearchStageConfig,
    instrument: str, db_path: str, library_ref: tuple[str, str] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    try:
        summary = run_search(
            df, risk, rules, space, stage_cfg, db_path=db_path,
            instrument=instrument, timeframe=infer_timeframe_label(df),
            progress_cb=lambda msg: _job_log(job_id, msg),
            cancel_event=cancel_event,
        )
        report_paths = generate_search_report(
            output_dir=str(SEARCH_DIR), summary=summary, space=space,
            instrument=instrument, timeframe=infer_timeframe_label(df),
        )
        if library_ref and summary.leaderboard:
            try:
                record_search_result(*library_ref, {
                    "candidates_tested": summary.total_candidates,
                    "best_fitness": round(summary.leaderboard[0].get("fitness", 0), 4),
                    "fitness_metric": stage_cfg.fitness_metric,
                    "report_html": f"/search_reports/{report_paths['html'].name}",
                })
            except (FileNotFoundError, ValueError):
                pass  # base strategy was renamed/deleted mid-search -- don't fail the job over it
        with _SEARCH_JOBS_LOCK:
            job = _SEARCH_JOBS[job_id]
            job["done"] = True
            job["summary"] = summary
            job["db_path"] = db_path
            job["df"] = df
            job["risk"] = risk
            job["rules"] = rules
            job["report_html"] = f"/search_reports/{report_paths['html'].name}"
            job["report_json"] = f"/search_reports/{report_paths['json'].name}"
    except SearchCancelled:
        # A deliberate STOP, not a crash -- surface it as a clean "stopped"
        # state on the job status page instead of the red error banner.
        with _SEARCH_JOBS_LOCK:
            job = _SEARCH_JOBS[job_id]
            job["done"] = True
            job["cancelled"] = True
            job["log"].append("Search Lab run stopped by user.")
    except Exception as exc:  # noqa: BLE001 -- a search job must fail visibly on the status page, not crash a thread silently
        log_crash("Search Lab (web)", exc=exc)
        with _SEARCH_JOBS_LOCK:
            job = _SEARCH_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    finally:
        HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)


def _run_search_loop_job(
    job_id: str, df, risk: RiskConfig, rules: PropRules, stage_cfg: SearchStageConfig,
    instrument: str, loop_dir: str, loop_cfg: SearchLoopConfig,
    cancel_event: threading.Event | None = None, family_health_dir: str | None = None,
) -> None:
    """Search Lab's "Loop Mode" job -- the same background-job/poll-for-
    status shape as _run_search_job above, but driving
    app.orchestration.loop_runner.run_search_loop (repeated rounds) instead
    of a single run_search() call. Reuses the SAME _SEARCH_JOBS dict/status
    page/stop button as a normal Search Lab job so the UI doesn't need a
    second job-tracking system -- job["loop_result"] and job["loop_rounds"]
    are the loop-specific additions; job["summary"] is kept updated with
    the MOST RECENT round's SearchSummary (via on_round) so the existing
    leaderboard rendering on the status page works unchanged while a loop
    is still running across many rounds, not just once it fully finishes.
    """
    def on_round(round_result) -> None:
        with _SEARCH_JOBS_LOCK:
            job = _SEARCH_JOBS.get(job_id)
            if job is None:
                return
            job["summary"] = round_result.summary
            job["loop_rounds"] = job.get("loop_rounds", 0) + 1
            job["loop_last_round"] = {
                "round_index": round_result.round_index,
                "family": round_result.family or "all",
                "max_candidates": round_result.max_candidates,
                "excluded_families": round_result.excluded_families,
                "best_value": round_result.best_value,
                "best_candidate_id": round_result.best_candidate_id,
                "widened_after_this_round": round_result.widened_after_this_round,
            }

    try:
        result = run_search_loop(
            df, risk, rules, stage_cfg, db_dir=loop_dir, loop_cfg=loop_cfg,
            instrument=instrument, timeframe=infer_timeframe_label(df),
            progress_cb=lambda msg: _job_log(job_id, msg),
            cancel_event=cancel_event, on_round=on_round,
            # Scoped to this instrument's own search directory (same one
            # Search Lab's own runs and this loop's own past rounds write
            # to) rather than app.search.family_health's real machine-wide
            # default dirs -- keeps this route's family-exclusion decisions
            # scoped to data this app itself produced, and (as a side
            # effect) keeps tests that redirect SEARCH_DIR fully hermetic.
            family_health_search_dir=family_health_dir, family_health_evolution_dir=family_health_dir,
        )
        with _SEARCH_JOBS_LOCK:
            job = _SEARCH_JOBS[job_id]
            job["done"] = True
            job["loop_result"] = result
            job["db_path"] = (
                result.winner_round.summary.db_path if result.winner_round else
                (result.rounds[-1].summary.db_path if result.rounds else None)
            )
            job["df"] = df
            job["risk"] = risk
            job["rules"] = rules
            if result.stopped_reason == "cancelled":
                job["cancelled"] = True
            elif result.stopped_reason == "error":
                job["error"] = result.error
    except Exception as exc:  # noqa: BLE001 -- a loop job must fail visibly on the status page, not crash a thread silently
        log_crash("Search Lab Loop Mode (web)", exc=exc)
        with _SEARCH_JOBS_LOCK:
            job = _SEARCH_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    finally:
        HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)



# status" shape as Search Lab above, since a multi-generation GA run over a
# few hundred backtests is too slow for a single request/response cycle.
# ---------------------------------------------------------------------------

_REFINEMENT_JOBS: dict[str, dict] = {}
_REFINEMENT_JOBS_LOCK = threading.Lock()


def _refinement_job_log(job_id: str, msg: str) -> None:
    with _REFINEMENT_JOBS_LOCK:
        job = _REFINEMENT_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_refinement_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules,
    mc_cfg: MonteCarloConfig, cfg: RefinementConfig, active_label: str,
    library_ref: tuple[str, str] | None = None,
    adaptive_risk=None,
) -> None:
    try:
        result = run_iterative_refinement(
            df, strategy, risk, rules, mc_cfg, cfg,
            progress_cb=lambda msg: _refinement_job_log(job_id, msg),
            adaptive_risk=adaptive_risk,
        )
        period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
        paths = generate_refinement_report(
            output_dir=REFINEMENT_DIR, result=result,
            strategy_name=getattr(strategy, "name", "Strategy"),
            instrument=active_label, timeframe=infer_timeframe_label(df), backtest_period=period,
            basename=f"refinement_{job_id}", price_df=df,
        )
        if library_ref:
            try:
                record_backtest_result(*library_ref, {
                    "note": "iterative refinement run",
                    "best_fitness": round(result.best.fitness, 4),
                    "generations": cfg.generations,
                    "report_html": f"/refinement_reports/{paths['html'].name}",
                })
            except (FileNotFoundError, ValueError):
                pass
        with _REFINEMENT_JOBS_LOCK:
            job = _REFINEMENT_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = f"/refinement_reports/{paths['html'].name}"
            job["report_json"] = f"/refinement_reports/{paths['json'].name}"
            best_file_key = "best_config_json" if "best_config_json" in paths else "best_strategy_file"
            job["best_file"] = f"/refinement_reports/{paths[best_file_key].name}"
    except RefinementError as exc:
        with _REFINEMENT_JOBS_LOCK:
            job = _REFINEMENT_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        with _REFINEMENT_JOBS_LOCK:
            job = _REFINEMENT_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"


@app.route("/refine")
def refine_form():
    return render_template(
        "refine.html",
        alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, **_alpaca_template_context())


@app.route("/refine/start", methods=["POST"])
def refine_start():
    form = request.form
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("refine.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400

        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            max_trades_per_day=int(form.get("max_trades_day", 10)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("n_sims", 2000) or 2000))

        # Same "Enable adaptive, limit-aware position sizing" overlay Quick
        # Optimize/Full Pipeline/Evolution Lab already offer (see
        # app.backtest.adaptive_risk) -- Iterative Refinement was previously
        # the only search tool with no way to turn this on, which made its
        # baseline/GA/final numbers silently non-comparable to a Quick
        # Optimize or Full Pipeline run made against the same strategy with
        # this enabled (see the 2026-09-16 Quick-Optimize-vs-Full-Pipeline
        # diagnosis this closes).
        adaptive_risk = build_limit_aware_preset(rules) if form.get("adaptive_risk_enabled") == "on" else None

        cfg = RefinementConfig(
            enabled=True,
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            population_size=int(form.get("population_size", 10) or 10),
            generations=int(form.get("generations", 5) or 5),
            elite_count=int(form.get("elite_count", 2) or 2),
            mutation_rate=float(form.get("mutation_rate", 0.35) or 0.35),
            mutation_strength=float(form.get("mutation_strength", 0.25) or 0.25),
            random_immigrants_frac=float(form.get("random_immigrants_frac", 0.15) or 0.15),
            search_monte_carlo_sims=int(form.get("search_mc_sims", 500) or 500),
            cost_stress_enabled=form.get("cost_stress_enabled") == "on",
            cost_stress_multiplier=float(form.get("cost_stress_multiplier", 2.0) or 2.0),
            # UPGRADE (optimizer core): explicit optimizer mode -- "genetic"
            # (default, unchanged) or the optional TPE/CMA-ES samplers (see
            # app.optimize.refinement.OPTIMIZER_MODES). Every downstream
            # step (plateau-robust selection, cost-stress penalty, the
            # final full-fidelity re-run, the holdout check) runs
            # identically regardless of which one is picked.
            optimizer_mode=form.get("optimizer_mode", "genetic") or "genetic",
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _REFINEMENT_JOBS_LOCK:
            _REFINEMENT_JOBS[job_id] = {
                "log": initial_log, "done": False, "error": None, "result": None,
                "started_at": time.time(), "instrument": active_label,
            }
        thread = threading.Thread(
            target=_run_refinement_job,
            args=(job_id, df, strategy, risk, rules, mc_cfg, cfg, active_label, library_ref),
            kwargs={"adaptive_risk": adaptive_risk},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("refine_job", job_id=job_id))

    except (StrategyError, RefinementError) as exc:
        return render_template("refine.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("refine.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 500


@app.route("/refine/job/<job_id>")
def refine_job(job_id):
    with _REFINEMENT_JOBS_LOCK:
        job = _REFINEMENT_JOBS.get(job_id)
    if job is None:
        return render_template("refine_job.html", job_id=job_id, not_found=True), 404
    return render_template("refine_job.html", job_id=job_id, not_found=False)


@app.route("/refine/job/<job_id>/status.json")
def refine_job_status(job_id):
    with _REFINEMENT_JOBS_LOCK:
        job = _REFINEMENT_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404

    result = job.get("result")
    generations = None
    if result is not None:
        generations = [
            {
                "generation": g.generation, "best_fitness": g.best_fitness,
                "mean_fitness": g.mean_fitness, "diversity": g.diversity,
            }
            for g in (result.generation_history or [])
        ]

    return jsonify({
        "found": True,
        "done": job["done"],
        "error": job["error"],
        "log": job["log"],
        "instrument": job.get("instrument"),
        "summary": None if result is None else {
            "baseline_fitness": result.baseline.fitness,
            "best_fitness": result.best.fitness,
            "best_generation": result.best.generation,
            "improvement_pct": (
                None if not result.baseline.fitness else
                round(100 * (result.best.fitness - result.baseline.fitness) / abs(result.baseline.fitness), 1)
            ),
            "generations_run": len(generations or []),
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
            "best_file": job.get("best_file"),
            # UPGRADE (distribution-based results reporting): this was
            # already computed by run_iterative_refinement (see
            # app.optimize.refinement._compute_distribution_summary's own
            # docstring for exactly what it does and doesn't claim) but
            # never actually reached the UI -- surfaced here so the job
            # page can show "Raw Optimum vs Robust Candidate" instead of
            # only ever reporting the single best genome found.
            "distribution_summary": result.distribution_summary,
        },
        "generations": generations,
    })


@app.route("/refinement_reports/<path:filename>")
def serve_refinement_report(filename):
    return send_from_directory(REFINEMENT_DIR, filename)


# ---------------------------------------------------------------------------
# Multi-Market Aggregate Search -- see app.optimize.multi_market's module
# docstring. Same background-job/poll shape as Iterative Refinement above,
# except the strategy is scored against SEVERAL markets at once (mean/
# worst-case/mean-minus-dispersion aggregate), not one.
# ---------------------------------------------------------------------------

_MULTI_MARKET_JOBS: dict[str, dict] = {}
_MULTI_MARKET_JOBS_LOCK = threading.Lock()


def _multi_market_job_log(job_id: str, msg: str) -> None:
    with _MULTI_MARKET_JOBS_LOCK:
        job = _MULTI_MARKET_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_multi_market_job(
    job_id: str, dfs: dict, strategy, risk: RiskConfig, rules: PropRules,
    mc_cfg: MonteCarloConfig, cfg: RefinementConfig, aggregation: str,
) -> None:
    try:
        result = run_multi_market_search(
            dfs, strategy, risk, rules, mc_cfg, cfg, aggregation=aggregation,
            progress_cb=lambda msg: _multi_market_job_log(job_id, msg),
        )
        with _MULTI_MARKET_JOBS_LOCK:
            job = _MULTI_MARKET_JOBS[job_id]
            job["done"] = True
            job["result"] = result
    except RefinementError as exc:
        with _MULTI_MARKET_JOBS_LOCK:
            job = _MULTI_MARKET_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        log_crash("Multi-Market Aggregate Search (web)", exc=exc)
        with _MULTI_MARKET_JOBS_LOCK:
            job = _MULTI_MARKET_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_MULTI_MARKET)


@app.route("/multi-market")
def multi_market_form():
    return render_template(
        "multi_market.html",
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(),
        fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
        aggregation_methods=AGGREGATION_METHODS, **_alpaca_template_context(),
    )


@app.route("/multi-market/start", methods=["POST"])
def multi_market_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_MULTI_MARKET):
        return render_template(
            "multi_market.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Wait for it to "
                f"finish first."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS,
            optimizer_modes=OPTIMIZER_MODES, aggregation_methods=AGGREGATION_METHODS,
            **_alpaca_template_context(),
        ), 409
    try:
        selected = form.getlist("datasets")
        if len(selected) < 2:
            HEAVY_JOB_GUARD.release(JOB_MULTI_MARKET)
            return render_template(
                "multi_market.html", error="Select at least 2 markets to score candidates against.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS,
                optimizer_modes=OPTIMIZER_MODES, aggregation_methods=AGGREGATION_METHODS,
                **_alpaca_template_context(),
            ), 400

        dfs: dict[str, pd.DataFrame] = {}
        load_warnings: list[str] = []
        for name in selected:
            candidate_path = get_raw_data_dir() / name
            if not candidate_path.exists():
                load_warnings.append(f"{name}: file not found -- skipped.")
                continue
            import_result = import_csv(candidate_path)
            if not import_result.is_valid:
                load_warnings.append(f"{name}: could not be read as market data -- skipped.")
                continue
            label = Path(name).stem
            dfs[label] = import_result.dataframe

        if len(dfs) < 2:
            HEAVY_JOB_GUARD.release(JOB_MULTI_MARKET)
            return render_template(
                "multi_market.html",
                error="Fewer than 2 of the selected datasets could actually be loaded: "
                      + "; ".join(load_warnings),
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS,
                optimizer_modes=OPTIMIZER_MODES, aggregation_methods=AGGREGATION_METHODS,
                **_alpaca_template_context(),
            ), 400

        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        )
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("n_sims", 2000) or 2000))
        cfg = RefinementConfig(
            enabled=True,
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            optimizer_mode=form.get("optimizer_mode", "genetic") or "genetic",
            population_size=int(form.get("population_size", 10) or 10),
            generations=int(form.get("generations", 5) or 5),
            elite_count=int(form.get("elite_count", 2) or 2),
            search_monte_carlo_sims=int(form.get("search_mc_sims", 300) or 300),
            random_seed=int(form.get("random_seed", 42) or 42),
        )
        aggregation = form.get("aggregation", "mean_minus_dispersion") or "mean_minus_dispersion"

        # T58 BACKTEST INTEGRITY CHECK -- same pre-flight gate as every
        # other search tool (see app.validation.integrity_check). Runs
        # against the FIRST selected market only, on the same reasoning
        # Search Lab's family_named mode uses for strategy=None: a check
        # against one representative dataset catches corrupt data or a
        # timeframe mismatch before spending compute across all of them,
        # without needing a per-market integrity report (none of the
        # other multi-instrument tools have one either).
        first_df = next(iter(dfs.values()))
        integrity_report = run_integrity_check(
            first_df, strategy, risk, prop_rules=rules,
            requested_timeframe=form.get("timeframe") or None,
            data_label=next(iter(dfs.keys())),
        )
        if integrity_report.status == "BLOCKED":
            HEAVY_JOB_GUARD.release(JOB_MULTI_MARKET)
            return render_template(
                "multi_market.html", error=integrity_report.render(),
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS,
                optimizer_modes=OPTIMIZER_MODES, aggregation_methods=AGGREGATION_METHODS,
                **_alpaca_template_context(),
            ), 400

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(dfs)} market(s): {', '.join(dfs.keys())}."]
        if load_warnings:
            initial_log.extend(load_warnings)
        with _MULTI_MARKET_JOBS_LOCK:
            _MULTI_MARKET_JOBS[job_id] = {
                "log": initial_log, "done": False, "error": None, "result": None,
                "started_at": time.time(), "markets": list(dfs.keys()), "aggregation": aggregation,
            }
        thread = threading.Thread(
            target=_run_multi_market_job,
            args=(job_id, dfs, strategy, risk, rules, mc_cfg, cfg, aggregation),
            daemon=True,
        )
        thread.start()
        return redirect(url_for("multi_market_job", job_id=job_id))

    except (StrategyError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_MULTI_MARKET)
        return render_template(
            "multi_market.html", error=str(exc),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS,
            optimizer_modes=OPTIMIZER_MODES, aggregation_methods=AGGREGATION_METHODS,
            **_alpaca_template_context(),
        ), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_MULTI_MARKET)
        return render_template(
            "multi_market.html", error=f"Unexpected error: {exc}",
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS,
            optimizer_modes=OPTIMIZER_MODES, aggregation_methods=AGGREGATION_METHODS,
            **_alpaca_template_context(),
        ), 500


@app.route("/multi-market/job/<job_id>")
def multi_market_job(job_id):
    with _MULTI_MARKET_JOBS_LOCK:
        job = _MULTI_MARKET_JOBS.get(job_id)
    return render_template("multi_market_job.html", job_id=job_id, not_found=job is None)


@app.route("/multi-market/job/<job_id>/status.json")
def multi_market_job_status(job_id):
    with _MULTI_MARKET_JOBS_LOCK:
        job = _MULTI_MARKET_JOBS.get(job_id)
    if job is None:
        return jsonify({"not_found": True}), 404

    result = job.get("result")
    payload = {
        "done": job["done"], "error": job.get("error"), "log": job["log"][-200:],
        "markets": job.get("markets", []), "aggregation": job.get("aggregation"),
    }
    if result is not None:
        payload["result"] = {
            "aggregation": result.aggregation,
            "fitness_metric": result.fitness_metric,
            "markets": result.markets,
            "total_evaluations": result.total_evaluations,
            "elapsed_seconds": result.elapsed_seconds,
            "warnings": result.warnings,
            "baseline": {
                "robustness_score": result.baseline.robustness_score,
                "mean_fitness": result.baseline.mean_fitness,
                "worst_case_fitness": result.baseline.worst_case_fitness,
                "dispersion": result.baseline.dispersion,
                "per_market": [
                    {"market": p.market, "fitness": p.fitness, "trade_count": p.trade_count}
                    for p in result.baseline.per_market
                ],
            },
            "best": {
                "robustness_score": result.best.robustness_score,
                "mean_fitness": result.best.mean_fitness,
                "worst_case_fitness": result.best.worst_case_fitness,
                "dispersion": result.best.dispersion,
                "config": result.best.config,
                "per_market": [
                    {"market": p.market, "fitness": p.fitness, "trade_count": p.trade_count}
                    for p in result.best.per_market
                ],
            },
            "generation_history": [
                {"generation": g.generation, "best_robustness_score": g.best_robustness_score,
                 "mean_robustness_score": g.mean_robustness_score}
                for g in result.generation_history
            ],
            "leaderboard": [
                {"robustness_score": c.robustness_score, "mean_fitness": c.mean_fitness,
                 "worst_case_fitness": c.worst_case_fitness,
                 "per_market": [{"market": p.market, "fitness": p.fitness} for p in c.per_market]}
                for c in result.leaderboard[:25]
            ],
        }
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Full Pipeline (Step 15 on desktop) -- the "run everything" button:
# baseline -> walk-forward-aware GA search -> re-validated final Monte Carlo
# -> OOS fold check -> holdout check -> READY/MARGINAL/NOT READY verdict.
# Same background-job/poll shape as Search Lab and Iterative Refinement --
# this is the single slowest thing the app can run.
# ---------------------------------------------------------------------------

_FULLPIPELINE_JOBS: dict[str, dict] = {}
_FULLPIPELINE_JOBS_LOCK = threading.Lock()


def _fullpipeline_job_log(job_id: str, msg: str) -> None:
    with _FULLPIPELINE_JOBS_LOCK:
        job = _FULLPIPELINE_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_fullpipeline_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules,
    cfg: FullPipelineConfig, active_label: str, ollama_settings: OllamaSettings | None,
    cancel_event: threading.Event | None = None, notify_webhook_url: str | None = None,
) -> None:
    try:
        result = run_full_pipeline(
            df, strategy, risk, rules, FULL_PIPELINE_DIR, cfg,
            progress_cb=lambda msg: _fullpipeline_job_log(job_id, msg),
            instrument=active_label, ollama_settings=ollama_settings,
            report_basename=f"full_pipeline_{job_id}",
            cancel_event=cancel_event,
        )
        with _FULLPIPELINE_JOBS_LOCK:
            job = _FULLPIPELINE_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = f"/full_pipeline_reports/{Path(result.report_paths['html']).name}"
            job["report_json"] = f"/full_pipeline_reports/{Path(result.report_paths['json']).name}"
        notify_job_finished(
            notify_webhook_url, "Full Pipeline",
            f"verdict={getattr(result, 'verdict', '?')}, instrument={active_label}",
            job_url=f"/full-pipeline/job/{job_id}",
        )
    except FullPipelineCancelled:
        with _FULLPIPELINE_JOBS_LOCK:
            job = _FULLPIPELINE_JOBS[job_id]
            job["done"] = True
            job["cancelled"] = True
        notify_job_finished(notify_webhook_url, "Full Pipeline", "Stopped by request", job_url=f"/full-pipeline/job/{job_id}")
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        log_crash("Full Pipeline (web)", exc=exc)
        with _FULLPIPELINE_JOBS_LOCK:
            job = _FULLPIPELINE_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
        notify_job_finished(notify_webhook_url, "Full Pipeline", f"FAILED -- {exc}", job_url=f"/full-pipeline/job/{job_id}")
    finally:
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)


_FULLPIPELINE_BATCH_JOBS: dict[str, dict] = {}
_FULLPIPELINE_BATCH_JOBS_LOCK = threading.Lock()


def _fullpipeline_batch_job_log(job_id: str, msg: str) -> None:
    with _FULLPIPELINE_BATCH_JOBS_LOCK:
        job = _FULLPIPELINE_BATCH_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _load_library_strategy_for_batch(mode: str, name: str):
    """Loads one Strategy Library entry by (type, filename) into a runnable
    Strategy object -- the manual-JSON case needs its own branch (it isn't
    source code, so build_strategy_from_code doesn't handle it), matching
    how the desktop's Strategy Library batch queue loads items."""
    code = load_strategy_text(mode, name)
    if mode == "manual":
        return ManualStrategy(json.loads(code))
    return build_strategy_from_code(mode, code)


def _run_fullpipeline_batch_job(
    job_id: str, df, batch_items, risk: RiskConfig, rules: PropRules,
    cfg: FullPipelineConfig, active_label: str, ollama_settings: OllamaSettings | None,
    cancel_event: threading.Event | None = None, notify_webhook_url: str | None = None,
) -> None:
    try:
        summary = run_full_pipeline_batch(
            df, batch_items, risk, rules, FULL_PIPELINE_DIR, cfg=cfg,
            instrument=active_label, ollama_settings=ollama_settings,
            progress_cb=lambda msg: _fullpipeline_batch_job_log(job_id, msg),
            max_parallel_strategies=1,
            cancel_event=cancel_event,
        )
        outcomes = [
            {
                "label": o.label, "ok": o.ok, "reason": o.reason, "verdict": o.verdict,
                "trades": o.trades, "net_profit": o.net_profit,
                "eval_pass_probability": o.eval_pass_probability,
                "report_html": (
                    f"/full_pipeline_reports/{Path(o.report_html).name}" if o.report_html else None
                ),
            }
            for o in summary.outcomes
        ]
        with _FULLPIPELINE_BATCH_JOBS_LOCK:
            job = _FULLPIPELINE_BATCH_JOBS[job_id]
            job["done"] = True
            job["outcomes"] = outcomes
            job["elapsed_seconds"] = summary.elapsed_seconds
        n_ready = sum(1 for o in outcomes if o["ok"])
        notify_job_finished(
            notify_webhook_url, "Full Pipeline (batch)",
            f"{len(outcomes)} strategies run, {n_ready} came back ready, instrument={active_label}",
            job_url=f"/full-pipeline/batch-job/{job_id}",
        )
    except FullPipelineBatchCancelled:
        with _FULLPIPELINE_BATCH_JOBS_LOCK:
            job = _FULLPIPELINE_BATCH_JOBS[job_id]
            job["done"] = True
            job["cancelled"] = True
        notify_job_finished(notify_webhook_url, "Full Pipeline (batch)", "Cancelled by user.", job_url=f"/full-pipeline/batch-job/{job_id}")
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        log_crash("Full Pipeline batch (web)", exc=exc)
        with _FULLPIPELINE_BATCH_JOBS_LOCK:
            job = _FULLPIPELINE_BATCH_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
        notify_job_finished(notify_webhook_url, "Full Pipeline (batch)", f"FAILED -- {exc}", job_url=f"/full-pipeline/batch-job/{job_id}")
    finally:
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)


@app.route("/full-pipeline/start-batch", methods=["POST"])
def full_pipeline_start_batch():
    """Runs the FULL 7-step Full Pipeline (not the lighter batch_test) on
    every Strategy Library item the user checked, one after another --
    the web equivalent of the desktop's "RUN FULL PIPELINE (BATCH)"
    button (see app.orchestration.full_pipeline.run_full_pipeline_batch).
    Shares the same dataset/risk/prop-rules/GA-config fields as the
    single-strategy form above (submitted via this button's `formaction`
    on the same <form>) -- only the strategy selection differs."""
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_FULL_PIPELINE):
        return render_template(
            "full_pipeline.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job (Search Lab / Evolution Lab / Full Pipeline / Speed Run) at the same "
                f"time can exhaust available memory. Wait for it to finish before starting Full Pipeline."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
            fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
            return render_template("full_pipeline.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400

        selected = [s for s in form.getlist("batch_items") if s.strip()]
        if not selected:
            HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
            return render_template(
                "full_pipeline.html",
                error="No strategies were selected for the batch. Check one or more strategies in the "
                      "\"Run on multiple saved strategies\" list before clicking RUN FULL PIPELINE (BATCH).",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
                fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400

        batch_items = []
        load_errors = []
        for ref in selected:
            mode, _, name = ref.partition("::")
            try:
                strategy = _load_library_strategy_for_batch(mode, name)
            except Exception as exc:  # noqa: BLE001 -- one bad library entry must not block the rest
                load_errors.append(f"{name} ({mode}): {exc}")
                continue
            batch_items.append(FullPipelineBatchItem(label=name, strategy=strategy, library_ref=(mode, name)))

        if not batch_items:
            HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
            return render_template(
                "full_pipeline.html",
                error="Every selected strategy failed to load: " + "; ".join(load_errors),
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
                fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            max_trades_per_day=int(form.get("max_trades_day", 10)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        library_status_raw = (form.get("library_status") or "").strip()
        cfg = FullPipelineConfig(
            n_folds=int(form.get("n_folds", 4) or 4),
            window_mode=form.get("window_mode", "rolling"),
            ga_population=int(form.get("ga_population", 12) or 12),
            ga_generations=int(form.get("ga_generations", 6) or 6),
            ga_search_mc_sims=int(form.get("ga_search_mc_sims", 200) or 200),
            optimizer_mode=form.get("optimizer_mode", "genetic") or "genetic",
            adaptive_risk_enabled=form.get("adaptive_risk_enabled") == "on",
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            final_mc_sims=int(form.get("final_mc_sims", 10000) or 10000),
            baseline_mc_sims=int(form.get("baseline_mc_sims", 2000) or 2000),
            holdout_frac=float(form.get("holdout_frac", 0.2) or 0.2),
            oos_check_folds=int(form.get("oos_check_folds", 4) or 4),
            random_seed=int(form.get("random_seed", 42) or 42),
            save_to_library=form.get("save_to_library") == "on",
            library_status=library_status_raw or None,
            parallel_search=form.get("parallel_search", "on") == "on",
            reset_on_breach=form.get("reset_on_breach", "on") == "on",
        )

        ollama_settings = None
        if form.get("ai_enabled") == "on":
            ollama_settings = OllamaSettings(
                enabled=True,
                host=form.get("ai_host", "http://localhost:11434") or "http://localhost:11434",
                model=form.get("ai_model", "llama3.1") or "llama3.1",
            )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}.", f"Queued {len(batch_items)} strateg{'y' if len(batch_items) == 1 else 'ies'} for the Full Pipeline batch."]
        if import_note:
            initial_log.append(import_note)
        if load_errors:
            initial_log.append(f"{len(load_errors)} selected strateg{'y' if len(load_errors) == 1 else 'ies'} failed to load and were skipped: " + "; ".join(load_errors))
        with _FULLPIPELINE_BATCH_JOBS_LOCK:
            cancel_event = threading.Event()
            _FULLPIPELINE_BATCH_JOBS[job_id] = {
                "log": initial_log, "done": False, "error": None, "outcomes": None,
                "started_at": time.time(), "instrument": active_label, "total": len(batch_items),
                "cancel_event": cancel_event, "cancelled": False,
            }
        thread = threading.Thread(
            target=_run_fullpipeline_batch_job,
            args=(job_id, df, batch_items, risk, rules, cfg, active_label, ollama_settings, cancel_event),
            kwargs={"notify_webhook_url": form.get("notify_webhook_url")},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("full_pipeline_batch_job", job_id=job_id))

    except StrategyError as exc:
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
        return render_template("full_pipeline.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
        log_crash("Full Pipeline (web, start-batch)", exc=exc)
        return render_template("full_pipeline.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 500


# ---------------------------------------------------------------------------
# Overnight Scheduler -- delay-starts a Full Pipeline batch run at a chosen
# clock time, so a Windows Task Scheduler-free "queue this before bed,
# check the result in the morning" workflow doesn't require actually being
# awake at the moment the run should start. This does NOT attempt a general
# cron system across every job type (Evolution Lab/Search Lab have their
# own target/max-generations auto-stop already) -- Full Pipeline batch is
# the one long-running, no-native-auto-stop job explicitly built for
# running many strategies unattended, so it's the one this schedules.
#
# The dataset, strategy selection, and every config field are all resolved
# and validated IMMEDIATELY at schedule time (identical validation to the
# immediate /full-pipeline/start-batch path) -- only the actual HEAVY_JOB_GUARD
# acquisition and thread start are deferred, so a bad dataset or an empty
# strategy selection fails right away with a clear error instead of silently
# failing at 2 AM with no one watching.
# ---------------------------------------------------------------------------

_SCHEDULED_JOBS: dict[str, dict] = {}
_SCHEDULED_JOBS_LOCK = threading.Lock()


def _seconds_until(target_hour: int, target_minute: int) -> float:
    """Seconds from now until the next occurrence of HH:MM local time
    (today if it hasn't passed yet, otherwise tomorrow)."""
    now = datetime.now()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _run_scheduled_fullpipeline_batch(schedule_id: str, delay_seconds: float, launch_kwargs: dict) -> None:
    with _SCHEDULED_JOBS_LOCK:
        entry = _SCHEDULED_JOBS.get(schedule_id)
        if entry is None:
            return
        entry["status"] = "waiting"
    # Sleep in short increments so a cancellation request is honored
    # promptly instead of only after the full delay elapses.
    waited = 0.0
    while waited < delay_seconds:
        with _SCHEDULED_JOBS_LOCK:
            entry = _SCHEDULED_JOBS.get(schedule_id)
            if entry is None or entry.get("cancelled"):
                return
        time.sleep(min(5.0, delay_seconds - waited))
        waited += 5.0

    # Wait for the resource guard too (another heavy job may still be
    # running right at the scheduled moment) -- retries for up to an hour
    # rather than failing the whole scheduled run over ordinary timing.
    max_guard_wait = 3600.0
    guard_waited = 0.0
    while not HEAVY_JOB_GUARD.try_acquire(JOB_FULL_PIPELINE):
        with _SCHEDULED_JOBS_LOCK:
            entry = _SCHEDULED_JOBS.get(schedule_id)
            if entry is None or entry.get("cancelled"):
                return
        if guard_waited >= max_guard_wait:
            with _SCHEDULED_JOBS_LOCK:
                entry = _SCHEDULED_JOBS.get(schedule_id)
                if entry is not None:
                    entry["status"] = "failed"
                    entry["error"] = f"{HEAVY_JOB_GUARD.active_name} was still running an hour after the scheduled start time -- gave up."
            return
        time.sleep(10.0)
        guard_waited += 10.0

    job_id = uuid.uuid4().hex[:12]
    with _FULLPIPELINE_BATCH_JOBS_LOCK:
        cancel_event = threading.Event()
        _FULLPIPELINE_BATCH_JOBS[job_id] = {
            "log": launch_kwargs["initial_log"], "done": False, "error": None, "outcomes": None,
            "started_at": time.time(), "instrument": launch_kwargs["active_label"], "total": len(launch_kwargs["batch_items"]),
            "cancel_event": cancel_event, "cancelled": False,
        }
    with _SCHEDULED_JOBS_LOCK:
        entry = _SCHEDULED_JOBS.get(schedule_id)
        if entry is not None:
            entry["status"] = "started"
            entry["job_id"] = job_id
    _run_fullpipeline_batch_job(
        job_id, launch_kwargs["df"], launch_kwargs["batch_items"], launch_kwargs["risk"], launch_kwargs["rules"],
        launch_kwargs["cfg"], launch_kwargs["active_label"], launch_kwargs["ollama_settings"], cancel_event,
        notify_webhook_url=launch_kwargs.get("notify_webhook_url"),
    )


@app.route("/full-pipeline/schedule-batch", methods=["POST"])
def full_pipeline_schedule_batch():
    """Same validation/resolution as /full-pipeline/start-batch, but
    defers the actual run to a chosen clock time instead of starting it
    immediately -- see the module comment above."""
    form = request.form
    try:
        start_at = (form.get("schedule_start_at") or "").strip()
        if not start_at or ":" not in start_at:
            return render_template("full_pipeline.html", error="Give a start time (HH:MM) to schedule the batch run.", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400
        hour_str, minute_str = start_at.split(":")[:2]
        target_hour, target_minute = int(hour_str), int(minute_str)

        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("full_pipeline.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400

        selected = [s for s in form.getlist("batch_items") if s.strip()]
        if not selected:
            return render_template(
                "full_pipeline.html",
                error="No strategies were selected to schedule. Check one or more strategies in the "
                      "\"Run on multiple saved strategies\" list first.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
                fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400

        batch_items = []
        load_errors = []
        for ref in selected:
            mode, _, name = ref.partition("::")
            try:
                strategy = _load_library_strategy_for_batch(mode, name)
            except Exception as exc:  # noqa: BLE001
                load_errors.append(f"{name} ({mode}): {exc}")
                continue
            batch_items.append(FullPipelineBatchItem(label=name, strategy=strategy, library_ref=(mode, name)))
        if not batch_items:
            return render_template(
                "full_pipeline.html", error="Every selected strategy failed to load: " + "; ".join(load_errors),
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
                fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            max_trades_per_day=int(form.get("max_trades_day", 10)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        library_status_raw = (form.get("library_status") or "").strip()
        cfg = FullPipelineConfig(
            n_folds=int(form.get("n_folds", 4) or 4),
            window_mode=form.get("window_mode", "rolling"),
            ga_population=int(form.get("ga_population", 12) or 12),
            ga_generations=int(form.get("ga_generations", 6) or 6),
            ga_search_mc_sims=int(form.get("ga_search_mc_sims", 200) or 200),
            optimizer_mode=form.get("optimizer_mode", "genetic") or "genetic",
            adaptive_risk_enabled=form.get("adaptive_risk_enabled") == "on",
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            final_mc_sims=int(form.get("final_mc_sims", 10000) or 10000),
            baseline_mc_sims=int(form.get("baseline_mc_sims", 2000) or 2000),
            holdout_frac=float(form.get("holdout_frac", 0.2) or 0.2),
            oos_check_folds=int(form.get("oos_check_folds", 4) or 4),
            random_seed=int(form.get("random_seed", 42) or 42),
            save_to_library=form.get("save_to_library") == "on",
            library_status=library_status_raw or None,
            parallel_search=form.get("parallel_search", "on") == "on",
            reset_on_breach=form.get("reset_on_breach", "on") == "on",
        )
        ollama_settings = None
        if form.get("ai_enabled") == "on":
            ollama_settings = OllamaSettings(
                enabled=True,
                host=form.get("ai_host", "http://localhost:11434") or "http://localhost:11434",
                model=form.get("ai_model", "llama3.1") or "llama3.1",
            )

        delay_seconds = _seconds_until(target_hour, target_minute)
        schedule_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}.", f"Queued {len(batch_items)} strateg{'y' if len(batch_items) == 1 else 'ies'} for the Full Pipeline batch."]
        if import_note:
            initial_log.append(import_note)
        if load_errors:
            initial_log.append(f"{len(load_errors)} selected strateg{'y' if len(load_errors) == 1 else 'ies'} failed to load and were skipped: " + "; ".join(load_errors))

        with _SCHEDULED_JOBS_LOCK:
            _SCHEDULED_JOBS[schedule_id] = {
                "status": "scheduled", "start_at": f"{target_hour:02d}:{target_minute:02d}",
                "scheduled_for": (datetime.now() + timedelta(seconds=delay_seconds)).isoformat(),
                "created_at": time.time(), "cancelled": False, "job_id": None, "error": None,
                "n_strategies": len(batch_items), "instrument": active_label,
            }
        thread = threading.Thread(
            target=_run_scheduled_fullpipeline_batch,
            args=(schedule_id, delay_seconds, {
                "df": df, "batch_items": batch_items, "risk": risk, "rules": rules, "cfg": cfg,
                "active_label": active_label, "ollama_settings": ollama_settings, "initial_log": initial_log,
                "notify_webhook_url": form.get("notify_webhook_url"),
            }),
            daemon=True,
        )
        thread.start()
        return redirect(url_for("full_pipeline_schedule_status", schedule_id=schedule_id))
    except StrategyError as exc:
        return render_template("full_pipeline.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        log_crash("Full Pipeline (web, schedule-batch)", exc=exc)
        return render_template("full_pipeline.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 500


@app.route("/full-pipeline/schedule/<schedule_id>")
def full_pipeline_schedule_status(schedule_id):
    with _SCHEDULED_JOBS_LOCK:
        entry = _SCHEDULED_JOBS.get(schedule_id)
    if entry is None:
        return render_template("full_pipeline_schedule.html", schedule_id=schedule_id, not_found=True), 404
    return render_template("full_pipeline_schedule.html", schedule_id=schedule_id, not_found=False)


@app.route("/full-pipeline/schedule/<schedule_id>/status.json")
def full_pipeline_schedule_status_json(schedule_id):
    with _SCHEDULED_JOBS_LOCK:
        entry = _SCHEDULED_JOBS.get(schedule_id)
    if entry is None:
        return jsonify({"found": False}), 404
    return jsonify({"found": True, **{k: v for k, v in entry.items()}})


@app.route("/full-pipeline/schedule/<schedule_id>/cancel", methods=["POST"])
def full_pipeline_schedule_cancel(schedule_id):
    with _SCHEDULED_JOBS_LOCK:
        entry = _SCHEDULED_JOBS.get(schedule_id)
        if entry is None:
            return jsonify({"ok": False, "error": "Not found"}), 404
        if entry["status"] in ("started", "failed"):
            return jsonify({"ok": False, "error": "Already started (or failed) -- use the batch job's own Stop button instead."}), 400
        entry["cancelled"] = True
        entry["status"] = "cancelled"
    return jsonify({"ok": True})


@app.route("/full-pipeline/batch-job/<job_id>")
def full_pipeline_batch_job(job_id):
    with _FULLPIPELINE_BATCH_JOBS_LOCK:
        job = _FULLPIPELINE_BATCH_JOBS.get(job_id)
    if job is None:
        return render_template("full_pipeline_batch_job.html", job_id=job_id, not_found=True), 404
    return render_template("full_pipeline_batch_job.html", job_id=job_id, not_found=False, total=job["total"])


@app.route("/full-pipeline/batch-job/<job_id>/stop", methods=["POST"])
def full_pipeline_batch_job_stop(job_id):
    """Signals a running Full Pipeline batch job to stop -- previously
    there was no way to stop one at all once started (see
    app.orchestration.full_pipeline.run_full_pipeline_batch's
    cancel_event / FullPipelineBatchCancelled). Mirrors Search Lab's own
    /search/job/<id>/stop: a no-op, not an error, if the job is already
    done or was never found. Between-item in serial mode, or within
    roughly a second in the parallel pool path (see
    _drain_batch_pool_futures) -- whichever the batch happens to be
    running in."""
    with _FULLPIPELINE_BATCH_JOBS_LOCK:
        job = _FULLPIPELINE_BATCH_JOBS.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Job not found."}), 404
        if job.get("done"):
            return jsonify({"ok": True, "already_done": True})
        cancel_event = job.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    return jsonify({"ok": True})


@app.route("/full-pipeline/batch-job/<job_id>/status.json")
def full_pipeline_batch_job_status(job_id):
    with _FULLPIPELINE_BATCH_JOBS_LOCK:
        job = _FULLPIPELINE_BATCH_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    return jsonify({
        "found": True,
        "done": job["done"],
        "error": job["error"],
        "cancelled": job.get("cancelled", False),
        "log": job["log"],
        "total": job["total"],
        "outcomes": job.get("outcomes"),
        "elapsed_seconds": job.get("elapsed_seconds"),
        "next_step": pipeline_guide.after_full_pipeline_batch(job["outcomes"]) if job.get("outcomes") else None,
    })


@app.route("/full-pipeline")
def full_pipeline_form():
    saved_ai = load_ollama_settings()
    return render_template(
        "full_pipeline.html",
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES,
        alpaca_notice=request.args.get("alpaca_notice"),
        alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"),
        fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
        prop_presets_json=_prop_presets_json(),
        ai_enabled=saved_ai.enabled,
        ai_host=saved_ai.host,
        ai_model=saved_ai.model, **_alpaca_template_context())


@app.route("/full-pipeline/start", methods=["POST"])
def full_pipeline_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_FULL_PIPELINE):
        return render_template(
            "full_pipeline.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job (Search Lab / Evolution Lab / Full Pipeline / Speed Run) at the same "
                f"time can exhaust available memory. Wait for it to finish before starting Full Pipeline."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
            fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
            return render_template("full_pipeline.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400

        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            max_trades_per_day=int(form.get("max_trades_day", 10)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )

        # T58 BACKTEST INTEGRITY CHECK -- same pre-flight gate as Run &
        # Report's /run route and Quick Optimize (see
        # app.validation.integrity_check's own docstring): catches corrupt
        # data, an unsupportable timeframe, or a confirmed lookahead leak
        # BEFORE the 15-step pipeline spends any compute on it.
        integrity_report = run_integrity_check(
            df, strategy, risk, prop_rules=rules,
            requested_timeframe=form.get("timeframe") or None,
            data_label=active_label,
        )
        if integrity_report.status == "BLOCKED":
            HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
            return render_template(
                "full_pipeline.html", error=integrity_report.render(),
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
                prop_presets_json=_prop_presets_json(), **_alpaca_template_context(),
            ), 400

        library_status_raw = (form.get("library_status") or "").strip()
        cfg = FullPipelineConfig(
            n_folds=int(form.get("n_folds", 4) or 4),
            window_mode=form.get("window_mode", "rolling"),
            ga_population=int(form.get("ga_population", 12) or 12),
            ga_generations=int(form.get("ga_generations", 6) or 6),
            ga_search_mc_sims=int(form.get("ga_search_mc_sims", 200) or 200),
            optimizer_mode=form.get("optimizer_mode", "genetic") or "genetic",
            adaptive_risk_enabled=form.get("adaptive_risk_enabled") == "on",
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            final_mc_sims=int(form.get("final_mc_sims", 10000) or 10000),
            baseline_mc_sims=int(form.get("baseline_mc_sims", 2000) or 2000),
            holdout_frac=float(form.get("holdout_frac", 0.2) or 0.2),
            oos_check_folds=int(form.get("oos_check_folds", 4) or 4),
            random_seed=int(form.get("random_seed", 42) or 42),
            save_to_library=form.get("save_to_library") == "on",
            library_status=library_status_raw or None,
            parallel_search=form.get("parallel_search", "on") == "on",
            reset_on_breach=form.get("reset_on_breach", "on") == "on",
        )

        ollama_settings = None
        if form.get("ai_enabled") == "on":
            ollama_settings = OllamaSettings(
                enabled=True,
                host=form.get("ai_host", "http://localhost:11434") or "http://localhost:11434",
                model=form.get("ai_model", "llama3.1") or "llama3.1",
            )
            try:
                save_ollama_settings(ollama_settings)  # persists, same as the desktop tab's own checkbox
            except Exception:
                pass  # best-effort -- a save failure shouldn't block the run itself

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        cancel_event = threading.Event()
        with _FULLPIPELINE_JOBS_LOCK:
            _FULLPIPELINE_JOBS[job_id] = {
                "log": initial_log, "done": False, "error": None, "result": None,
                "started_at": time.time(), "instrument": active_label,
                "cancel_event": cancel_event, "cancelled": False,
            }
        thread = threading.Thread(
            target=_run_fullpipeline_job,
            args=(job_id, df, strategy, risk, rules, cfg, active_label, ollama_settings),
            kwargs={"cancel_event": cancel_event, "notify_webhook_url": form.get("notify_webhook_url")},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("full_pipeline_job", job_id=job_id))

    except StrategyError as exc:
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
        return render_template("full_pipeline.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
        log_crash("Full Pipeline (web, start)", exc=exc)
        return render_template("full_pipeline.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 500


@app.route("/full-pipeline/job/<job_id>")
def full_pipeline_job(job_id):
    with _FULLPIPELINE_JOBS_LOCK:
        job = _FULLPIPELINE_JOBS.get(job_id)
    if job is None:
        return render_template("full_pipeline_job.html", job_id=job_id, not_found=True), 404
    return render_template("full_pipeline_job.html", job_id=job_id, not_found=False)


@app.route("/full-pipeline/job/<job_id>/stop", methods=["POST"])
def full_pipeline_job_stop(job_id):
    """Signals cancellation to a running single-strategy Full Pipeline job
    -- see run_full_pipeline's cancel_event param. Checked between each of
    the 7 steps, so this stops the run at the next step boundary rather
    than instantly, same tradeoff the batch job's stop button already
    makes."""
    with _FULLPIPELINE_JOBS_LOCK:
        job = _FULLPIPELINE_JOBS.get(job_id)
        if job is None:
            return jsonify({"found": False}), 404
        cancel_event = job.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    return jsonify({"found": True, "stopping": True})


@app.route("/full-pipeline/job/<job_id>/status.json")
def full_pipeline_job_status(job_id):
    with _FULLPIPELINE_JOBS_LOCK:
        job = _FULLPIPELINE_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404

    result = job.get("result")
    summary = None
    if result is not None:
        summary = {
            "verdict": result.verdict,
            "verdict_reasons": result.verdict_reasons,
            "baseline_trades": len(result.baseline_bt.trades),
            "baseline_net_profit": result.baseline_bt.statistics.net_profit,
            "final_trades": len(result.final_bt.trades),
            "final_net_profit": result.final_bt.statistics.net_profit,
            "final_win_rate": result.final_bt.statistics.win_rate,
            "final_max_dd": result.final_bt.statistics.max_drawdown_pct,
            "eval_pass_probability": result.final_mc.evaluation_pass_probability,
            "first_payout_probability": result.final_mc.first_payout_probability,
            "risk_of_ruin_pct": result.final_mc.risk_of_ruin_pct,
            "refinement_ran": result.refinement_ran,
            "refinement_skip_reason": result.refinement_skip_reason,
            "oos_skip_reason": result.oos_validation_skip_reason,
            "saved_library_note": result.saved_library_note,
            "elapsed_seconds": result.elapsed_seconds,
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
            "warnings": result.warnings,
        }

    return jsonify({
        "found": True,
        "done": job["done"],
        "error": job["error"],
        "cancelled": job.get("cancelled", False),
        "log": job["log"],
        "instrument": job.get("instrument"),
        "summary": summary,
        "next_step": (
            pipeline_guide.after_full_pipeline(result.verdict, bool(result.saved_library_note), result=result)
            if result is not None else None
        ),
    })


@app.route("/full_pipeline_reports/<path:filename>")
def serve_fullpipeline_report(filename):
    return send_from_directory(FULL_PIPELINE_DIR, filename)


# ---------------------------------------------------------------------------
# Step 08: Walk-Forward Optimization -- re-optimizes on each fold's train
# window, applies the winner UNCHANGED to that fold's held-out test window,
# and chains every fold's OOS trades into one continuous result. Same
# background-job/poll shape as the other slow tabs above.
# ---------------------------------------------------------------------------

_WFO_JOBS: dict[str, dict] = {}
_WFO_JOBS_LOCK = threading.Lock()


def _wfo_job_log(job_id: str, msg: str) -> None:
    with _WFO_JOBS_LOCK:
        job = _WFO_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_wfo_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig,
    n_folds: int, window_mode: str, train_frac: float, embargo_bars: int, refine_cfg: RefinementConfig,
    strategy_name: str = "", instrument: str = "",
) -> None:
    try:
        result = run_walk_forward_optimization(
            df, strategy, risk, rules, mc_cfg, n_folds=n_folds, window_mode=window_mode,
            train_frac=train_frac, embargo_bars=embargo_bars, refine_cfg=refine_cfg,
            progress_cb=lambda msg: _wfo_job_log(job_id, msg),
        )
        paths = generate_walk_forward_report(WFO_DIR, result, basename=f"walk_forward_opt_{job_id}")
        report_html = f"/wfo_reports/{Path(paths['html']).name}"
        with _WFO_JOBS_LOCK:
            job = _WFO_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = report_html
            job["report_json"] = f"/wfo_reports/{Path(paths['json']).name}"
        eff = getattr(result, "out_of_sample_efficiency", None)
        summary = f"OOS efficiency {eff:.2f}" if eff is not None else f"{n_folds} folds completed"
        # No strict pass/fail verdict is computed by this tool -- passed=None
        # records that it *ran*, without inventing a threshold it doesn't set.
        strategy_state.record_validation(strategy_name, instrument, "wfo", passed=None, summary=summary, report_html=report_html)
    except RefinementError as exc:
        with _WFO_JOBS_LOCK:
            job = _WFO_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _WFO_JOBS_LOCK:
            job = _WFO_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_WFO)


@app.route("/walk-forward-opt")
def wfo_form():
    return render_template(
        "wfo.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context())


@app.route("/walk-forward-opt/start", methods=["POST"])
def wfo_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_WFO, "wfo.html", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
        fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_WFO)
            return render_template("wfo.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400

        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("n_sims", 1000) or 1000))
        refine_cfg = RefinementConfig(
            population_size=int(form.get("population_size", 8) or 8),
            generations=int(form.get("generations", 3) or 3),
            search_monte_carlo_sims=int(form.get("search_mc_sims", 200) or 200),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _WFO_JOBS_LOCK:
            _WFO_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(
            target=_run_wfo_job,
            args=(
                job_id, df, strategy, risk, rules, mc_cfg,
                int(form.get("n_folds", 5) or 5), form.get("window_mode", "rolling"),
                float(form.get("train_frac", 0.6) or 0.6), int(form.get("embargo_bars", 0) or 0), refine_cfg,
            ),
            kwargs={"strategy_name": getattr(strategy, "name", "Strategy"), "instrument": active_label},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("wfo_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_WFO)
        return render_template("wfo.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_WFO)
        return render_template("wfo.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 500


@app.route("/walk-forward-opt/job/<job_id>")
def wfo_job(job_id):
    with _WFO_JOBS_LOCK:
        job = _WFO_JOBS.get(job_id)
    if job is None:
        return render_template("wfo_job.html", job_id=job_id, not_found=True), 404
    return render_template("wfo_job.html", job_id=job_id, not_found=False)


@app.route("/walk-forward-opt/job/<job_id>/status.json")
def wfo_job_status(job_id):
    with _WFO_JOBS_LOCK:
        job = _WFO_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        d = result.to_summary_dict()
        d["report_html"] = job.get("report_html")
        d["report_json"] = job.get("report_json")
        summary = d
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


@app.route("/wfo_reports/<path:filename>")
def serve_wfo_report(filename):
    return send_from_directory(WFO_DIR, filename)


# ---------------------------------------------------------------------------
# Step 09: Multi-Objective (NSGA-II) Optimization -- searches for a Pareto
# front across 2+ objectives at once (e.g. Sharpe vs. max drawdown vs. eval
# pass probability) instead of collapsing everything into one fitness
# number. Same background-job/poll shape as the other slow tabs.
# ---------------------------------------------------------------------------

_MO_JOBS: dict[str, dict] = {}
_MO_JOBS_LOCK = threading.Lock()


def _mo_job_log(job_id: str, msg: str) -> None:
    with _MO_JOBS_LOCK:
        job = _MO_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_mo_job(job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig, mo_cfg: MultiObjectiveConfig) -> None:
    try:
        result = run_multi_objective_refinement(
            df, strategy, risk, rules, mc_cfg, mo_cfg,
            progress_cb=lambda msg: _mo_job_log(job_id, msg),
        )
        paths = generate_multi_objective_report(MULTI_OBJ_DIR, result, basename=f"multi_objective_{job_id}")
        with _MO_JOBS_LOCK:
            job = _MO_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = f"/mo_reports/{Path(paths['html']).name}"
            job["report_json"] = f"/mo_reports/{Path(paths['json']).name}"
    except RefinementError as exc:
        with _MO_JOBS_LOCK:
            job = _MO_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _MO_JOBS_LOCK:
            job = _MO_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"


def _run_mo_sweep_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig,
    mo_cfg: MultiObjectiveConfig, timeframes: list[str],
) -> None:
    """Same shape as _run_mo_job, but drives
    app.optimize.multi_objective.run_multi_objective_sweep -- one full
    NSGA-II run per timeframe, each with its own report. `job["result"]`
    is set to the FIRST timeframe's own MultiObjectiveResult so
    mo_job_status's existing single-result summary keeps working
    unchanged; `job["sweep_results"]` carries every timeframe's own
    summary + report link for the job page's per-timeframe breakdown.
    """
    try:
        sweep = run_multi_objective_sweep(
            df, strategy, risk, rules, mc_cfg, timeframes, mo_cfg,
            progress_cb=lambda msg: _mo_job_log(job_id, msg),
        )
        sweep_results = {}
        first_result = None
        for label, result in sweep.per_timeframe.items():
            paths = generate_multi_objective_report(
                MULTI_OBJ_DIR, result, basename=f"multi_objective_{job_id}_{label}",
            )
            report_html = f"/mo_reports/{Path(paths['html']).name}"
            if first_result is None:
                first_result = result
            sweep_results[label] = {
                "objectives": result.config.objectives,
                "pareto_front": [
                    {"objective_values": dict(zip(result.config.objectives, c.objective_values)), "feasible": c.feasible}
                    for c in result.pareto_front
                ],
                "generations_run": len(result.generation_history or []),
                "elapsed_seconds": result.elapsed_seconds,
                "report_html": report_html,
            }
        with _MO_JOBS_LOCK:
            job = _MO_JOBS[job_id]
            job["done"] = True
            job["result"] = first_result
            job["report_html"] = sweep_results.get(next(iter(sweep_results), ""), {}).get("report_html")
            job["sweep_results"] = sweep_results
            if sweep.skipped:
                job["log"].append(
                    "Skipped from the timeframe sweep: "
                    + "; ".join(f"{s.requested_label} ({s.reason})" for s in sweep.skipped)
                )
            for label, err in sweep.errors.items():
                job["log"].append(f"[{label}] could not be searched: {err}")
    except RefinementError as exc:
        with _MO_JOBS_LOCK:
            job = _MO_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _MO_JOBS_LOCK:
            job = _MO_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_MULTI_OBJECTIVE)


@app.route("/multi-objective")
def mo_form():
    return render_template(
        "multi_objective.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, all_objectives=sorted(OBJECTIVE_DIRECTIONS), default_objectives=DEFAULT_OBJECTIVES, **_alpaca_template_context())


@app.route("/multi-objective/start", methods=["POST"])
def mo_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_MULTI_OBJECTIVE, "multi_objective.html", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(), all_objectives=sorted(OBJECTIVE_DIRECTIONS),
        default_objectives=DEFAULT_OBJECTIVES,
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_MULTI_OBJECTIVE)
            return render_template("multi_objective.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), all_objectives=sorted(OBJECTIVE_DIRECTIONS), default_objectives=DEFAULT_OBJECTIVES, **_alpaca_template_context()), 400

        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("n_sims", 1000) or 1000))

        objectives = form.getlist("objectives") or list(DEFAULT_OBJECTIVES)
        mo_cfg = MultiObjectiveConfig(
            objectives=objectives,
            population_size=int(form.get("population_size", 20) or 20),
            generations=int(form.get("generations", 8) or 8),
            search_monte_carlo_sims=int(form.get("search_mc_sims", 300) or 300),
            reset_on_breach=form.get("reset_on_breach") == "on",
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _MO_JOBS_LOCK:
            _MO_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        # FIX (multi-timeframe sweep): "Timeframes to test" runs the SAME
        # NSGA-II search once per requested timeframe (df resampled per
        # timeframe -- see app.data.timeframe_sweep) and reports every
        # timeframe's own Pareto front side by side rather than merging
        # them -- see app.optimize.multi_objective.run_multi_objective_sweep
        # for why that merge is deliberately NOT attempted.
        expand_labels = parse_sweep_timeframes(form.get("expand_timeframes", ""))
        if expand_labels:
            thread = threading.Thread(
                target=_run_mo_sweep_job, args=(job_id, df, strategy, risk, rules, mc_cfg, mo_cfg, expand_labels),
                daemon=True,
            )
        else:
            thread = threading.Thread(target=_run_mo_job, args=(job_id, df, strategy, risk, rules, mc_cfg, mo_cfg), daemon=True)
        thread.start()
        return redirect(url_for("mo_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_MULTI_OBJECTIVE)
        return render_template("multi_objective.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), all_objectives=sorted(OBJECTIVE_DIRECTIONS), default_objectives=DEFAULT_OBJECTIVES, **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_MULTI_OBJECTIVE)
        return render_template("multi_objective.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), all_objectives=sorted(OBJECTIVE_DIRECTIONS), default_objectives=DEFAULT_OBJECTIVES, **_alpaca_template_context()), 500


@app.route("/multi-objective/job/<job_id>")
def mo_job(job_id):
    with _MO_JOBS_LOCK:
        job = _MO_JOBS.get(job_id)
    if job is None:
        return render_template("multi_objective_job.html", job_id=job_id, not_found=True), 404
    return render_template("multi_objective_job.html", job_id=job_id, not_found=False)


@app.route("/multi-objective/job/<job_id>/status.json")
def mo_job_status(job_id):
    with _MO_JOBS_LOCK:
        job = _MO_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = {
            "objectives": result.config.objectives,
            "pareto_front": [
                {"objective_values": dict(zip(result.config.objectives, c.objective_values)), "feasible": c.feasible}
                for c in result.pareto_front
            ],
            "generations_run": len(result.generation_history or []),
            "elapsed_seconds": result.elapsed_seconds,
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
        }
    return jsonify({
        "found": True, "done": job["done"], "error": job["error"], "log": job["log"],
        "instrument": job.get("instrument"), "summary": summary, "sweep_results": job.get("sweep_results"),
    })


@app.route("/mo_reports/<path:filename>")
def serve_mo_report(filename):
    return send_from_directory(MULTI_OBJ_DIR, filename)


# ---------------------------------------------------------------------------
# Step 10: Walk-Forward-Aware GA -- the same GA engine as Iterative
# Refinement (Step 06), but every candidate's fitness is scored ONLY on
# chained out-of-sample fold performance instead of a single in-sample run.
# Same background-job/poll shape as the other slow tabs.
# ---------------------------------------------------------------------------

_WFGA_JOBS: dict[str, dict] = {}
_WFGA_JOBS_LOCK = threading.Lock()


def _wfga_job_log(job_id: str, msg: str) -> None:
    with _WFGA_JOBS_LOCK:
        job = _WFGA_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_wfga_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig,
    refine_cfg: RefinementConfig, n_folds: int, window_mode: str, train_frac: float,
    strategy_name: str = "", instrument: str = "",
) -> None:
    try:
        result = run_walkforward_aware_refinement(
            df, strategy, risk, rules, mc_cfg, refine_cfg, n_folds=n_folds, window_mode=window_mode,
            train_frac=train_frac, progress_cb=lambda msg: _wfga_job_log(job_id, msg),
        )
        paths = generate_walkforward_ga_report(WFGA_DIR, result, basename=f"walkforward_ga_{job_id}")
        report_html = f"/wfga_reports/{Path(paths['html']).name}"
        with _WFGA_JOBS_LOCK:
            job = _WFGA_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = report_html
            job["report_json"] = f"/wfga_reports/{Path(paths['json']).name}"
        gap = getattr(result, "overfitting_gap", None)
        summary = f"overfitting gap {gap:.2f}" if gap is not None else f"{n_folds} folds, walk-forward-aware GA"
        strategy_state.record_validation(strategy_name, instrument, "wfga", passed=None, summary=summary, report_html=report_html)
    except RefinementError as exc:
        with _WFGA_JOBS_LOCK:
            job = _WFGA_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _WFGA_JOBS_LOCK:
            job = _WFGA_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_WFGA)


@app.route("/walk-forward-ga")
def wfga_form():
    return render_template(
        "wfga.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context())


@app.route("/walk-forward-ga/start", methods=["POST"])
def wfga_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_WFGA, "wfga.html", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
        fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_WFGA)
            return render_template("wfga.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400

        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        # FIX (audit, Sep 2026): Walk-Forward GA is one of the nine
        # JOB_WFGA-class heavy jobs (see app.orchestration.resource_guard)
        # but, unlike Evolution Lab/Quick Optimize/Full Pipeline/Speed
        # Run, never hardened its RiskConfig against the active PropRules
        # -- see app.backtest.risk.with_prop_safety_defaults' own
        # docstring. Without this, every fold backtest below could keep
        # opening new trades straight through a blown account or a
        # breached daily-loss limit.
        risk = with_prop_safety_defaults(risk, rules)
        # Default OFF when the field is absent (matches every other "on"
        # checkbox in this app, e.g. loop_mode/save_to_library, and keeps
        # a caller that posts without this field byte-identical to before
        # this existed) -- the web form itself renders the checkbox
        # pre-CHECKED, so a user who never touches it still gets "on"
        # submitted; only an explicit uncheck (or an old/scripted POST
        # that never sends the field) resolves to False here.
        reset_on_breach = form.get("reset_on_breach") == "on"
        # FIX (2026-09-18): see RiskConfig.reset_on_breach's docstring --
        # this was already threaded into mc_cfg below but never into
        # `risk`, which every fold's own run_backtest() call actually
        # scores fitness from.
        risk = replace(risk, reset_on_breach=reset_on_breach)
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("n_sims", 1000) or 1000), reset_on_breach=reset_on_breach)
        refine_cfg = RefinementConfig(
            population_size=int(form.get("population_size", 10) or 10),
            generations=int(form.get("generations", 5) or 5),
            search_monte_carlo_sims=int(form.get("search_mc_sims", 200) or 200),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _WFGA_JOBS_LOCK:
            _WFGA_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(
            target=_run_wfga_job,
            args=(
                job_id, df, strategy, risk, rules, mc_cfg, refine_cfg,
                int(form.get("n_folds", 4) or 4), form.get("window_mode", "rolling"), float(form.get("train_frac", 0.6) or 0.6),
            ),
            kwargs={"strategy_name": getattr(strategy, "name", "Strategy"), "instrument": active_label},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("wfga_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_WFGA)
        return render_template("wfga.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_WFGA)
        return render_template("wfga.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 500


@app.route("/walk-forward-ga/job/<job_id>")
def wfga_job(job_id):
    with _WFGA_JOBS_LOCK:
        job = _WFGA_JOBS.get(job_id)
    if job is None:
        return render_template("wfga_job.html", job_id=job_id, not_found=True), 404
    return render_template("wfga_job.html", job_id=job_id, not_found=False)


@app.route("/walk-forward-ga/job/<job_id>/status.json")
def wfga_job_status(job_id):
    with _WFGA_JOBS_LOCK:
        job = _WFGA_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = {
            "best_fitness": result.best.fitness,
            "best_in_sample_fitness": result.best.in_sample_fitness,
            "overfitting_gap": result.overfitting_gap,
            "oos_trade_count": result.best.oos_trade_count,
            "n_folds": result.n_folds,
            "window_mode": result.window_mode,
            "generations_run": len(result.generation_history or []),
            "elapsed_seconds": result.elapsed_seconds,
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
        }
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


@app.route("/wfga_reports/<path:filename>")
def serve_wfga_report(filename):
    return send_from_directory(WFGA_DIR, filename)


def _resolve_leg_dataset(form, files, prefix: str):
    """Resolves ONE instrument leg's dataset for Portfolio -- unlike
    _resolve_dataset (which picks a single active df for the whole page),
    Portfolio needs N independent DataFrames at once, one per leg, so each
    leg gets its own uploaded-file/stored-dataset pair under a distinct
    form-field prefix (leg1_csv/leg1_existing_dataset, leg2_..., etc.).
    Returns (df, label) or (None, None) if this leg wasn't filled in."""
    uploaded = files.get(f"{prefix}_csv")
    if uploaded and uploaded.filename:
        content = uploaded.read()
        result = import_csv_bytes(content, filename=uploaded.filename)
        if result.is_valid:
            store_csv_bytes(content, uploaded.filename)
            return result.dataframe, uploaded.filename
        raise StrategyError(f"'{uploaded.filename}': {'; '.join(result.errors)}")
    existing_choice = (form.get(f"{prefix}_existing_dataset") or "").strip()
    if existing_choice:
        candidate = get_raw_data_dir() / existing_choice
        if candidate.exists():
            # Read via the real path (not raw bytes) so the importer's
            # extension dispatch sees the actual .parquet/.tsv/etc suffix
            # instead of losing it the way a bare BytesIO would.
            result = import_csv(candidate)
            if result.is_valid:
                return result.dataframe, existing_choice
    return None, None


def _mode_from_filename(filename: str) -> str | None:
    """Extension-based strategy-type detection for Ensemble's multi-file
    leg upload, where (unlike every other form on this site) the mode
    isn't already known from which tab/tab-button the person is on."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in (".pine", ".pinescript", ".txt"):
        return "pinescript"
    if suffix in (".mq5", ".mqh"):
        return "mql5"
    return None


# ---------------------------------------------------------------------------
# Step 11: Multi-Asset Portfolio -- the SAME strategy config + SAME base
# risk settings applied to N DIFFERENT instruments at once, correlation-
# aware re-weighted and chained into one combined equity curve. Fast
# enough (a handful of plain backtests, no GA) to run synchronously like
# /run does, rather than the background-job/poll pattern the slower tabs
# above need.
# ---------------------------------------------------------------------------

@app.route("/portfolio")
def portfolio_form():
    return render_template(
        "portfolio.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, **_alpaca_template_context())


@app.route("/portfolio/run", methods=["POST"])
def portfolio_run():
    form = request.form
    ctx = lambda **kw: dict(stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **kw)
    try:
        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )

        legs: list[InstrumentLeg] = []
        leg_labels = []
        for i in range(1, 5):
            prefix = f"leg{i}"
            df, label = _resolve_leg_dataset(form, request.files, prefix)
            if df is None:
                continue
            weight = float(form.get(f"{prefix}_weight", 1.0) or 1.0)

            # Library-based leg picker: each leg can override the shared
            # Step-1 strategy with a specific saved strategy from the
            # library (mirrors the desktop app's "ADD LEG FROM LIBRARY"
            # button), so a portfolio can combine genuinely DIFFERENT
            # strategies -- not just one strategy across instruments.
            leg_mode = (form.get(f"{prefix}_library_mode") or "").strip()
            leg_name = (form.get(f"{prefix}_library_name") or "").strip()
            if leg_mode and leg_name:
                try:
                    leg_code = load_strategy_text(leg_mode, leg_name)
                    leg_strategy = build_strategy_from_code(leg_mode, leg_code)
                except (StrategyError, FileNotFoundError, OSError) as exc:
                    return render_template("portfolio.html", **ctx(
                        error=f"Could not load library strategy '{leg_name}' for leg {i}: {exc}"
                    ), **_alpaca_template_context()), 400
                leg_label = f"{label} ({leg_name})"
            else:
                leg_strategy = strategy
                leg_label = label

            legs.append(InstrumentLeg(name=leg_label, df=df, strategy=leg_strategy, risk=risk, weight=weight))
            leg_labels.append(leg_label)

        if len(legs) < 2:
            return render_template("portfolio.html", **ctx(error="A portfolio needs at least 2 instrument legs -- fill in a market data file/dataset for at least 2 of the leg slots below."), **_alpaca_template_context()), 400

        config = PortfolioConfig(
            initial_balance=risk.initial_balance,
            correlation_penalty_strength=float(form.get("correlation_penalty_strength", 0.6) or 0.6),
            max_instrument_weight_frac=float(form.get("max_instrument_weight_frac", 0.5) or 0.5),
            min_weight_frac=float(form.get("min_weight_frac", 0.15) or 0.15),
        )
        result = run_portfolio_backtest(legs, config)
        run_id = uuid.uuid4().hex[:10]
        paths = generate_portfolio_report(PORTFOLIO_DIR, result, basename=f"portfolio_{run_id}")

        return render_template("portfolio.html", **ctx(result={
            "legs": leg_labels,
            "trades": result.combined_statistics.total_trades,
            "net_profit": result.combined_statistics.net_profit,
            "max_dd": result.combined_statistics.max_drawdown_pct,
            "sharpe": result.combined_statistics.sharpe_ratio,
            "diversification_ratio": result.diversification_ratio,
            "warnings": result.warnings,
            "report_html": f"/portfolio_reports/{Path(paths['html']).name}",
            "report_json": f"/portfolio_reports/{Path(paths['json']).name}",
        }), **_alpaca_template_context())
    except (StrategyError, PortfolioError) as exc:
        return render_template("portfolio.html", **ctx(error=str(exc)), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("portfolio.html", **ctx(error=f"Unexpected error: {exc}"), **_alpaca_template_context()), 500


@app.route("/portfolio_reports/<path:filename>")
def serve_portfolio_report(filename):
    return send_from_directory(PORTFOLIO_DIR, filename)


# ---------------------------------------------------------------------------
# Regime Survival Matrix
# ---------------------------------------------------------------------------

_REGIME_DIMENSIONS = ("trend", "volatility", "session", "environment")


@app.route("/regime-matrix")
def regime_matrix_form():
    return render_template(
        "regime_matrix.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context())


@app.route("/regime-matrix/run", methods=["POST"])
def regime_matrix_run():
    form = request.form
    ctx = lambda **kw: dict(stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **kw)
    guard_resp = _try_acquire_heavy_job(JOB_REGIME_MATRIX, "regime_matrix.html", **ctx())
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("regime_matrix.html", **ctx(error=dataset_error), **_alpaca_template_context()), 400

        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )

        dim_a = form.get("dimension_a", "volatility")
        dim_b = form.get("dimension_b", "environment")
        if dim_a == dim_b:
            return render_template("regime_matrix.html", **ctx(
                error="Pick two DIFFERENT dimensions to cross for the primary matrix.",
            ), **_alpaca_template_context()), 400

        result = run_regime_matrix(df, strategy, risk, dimensions=(dim_a, dim_b))
        if result is None:
            return render_template("regime_matrix.html", **ctx(
                error="Not enough bars in this dataset to classify regimes reliably -- try a longer history.",
            ), **_alpaca_template_context()), 400

        strategy_state.record_validation(
            getattr(strategy, "name", "Strategy"), active_label, "regime_matrix",
            passed=None, summary=f"{len(result.cells)} regime cell(s) analyzed",
        )

        return render_template("regime_matrix.html", **ctx(result={
            "dataset": active_label,
            "dimensions": list(result.primary_dimensions),
            "cells": sorted([c.to_dict() for c in result.cells], key=lambda c: c["net_profit"], reverse=True),
            "single_dimension": {k: [c.to_dict() for c in v] for k, v in result.single_dimension.items()},
            "disable_regimes": [c.to_dict() for c in result.disable_regimes()],
            "notes": result.notes,
            "table_text": result.render_table(),
        }), **_alpaca_template_context())
    except StrategyError as exc:
        return render_template("regime_matrix.html", **ctx(error=str(exc)), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("regime_matrix.html", **ctx(error=f"Unexpected error: {exc}"), **_alpaca_template_context()), 500
    finally:
        HEAVY_JOB_GUARD.release(JOB_REGIME_MATRIX)


# ---------------------------------------------------------------------------
# Strategy Family Diversity -- reads an already-completed Search Lab run
# (any search_*.db under SEARCH_DIR) and reports per-family performance.
# ---------------------------------------------------------------------------

def _available_search_runs() -> list[dict]:
    """Scans every search_*.db under SEARCH_DIR (one per Search Lab job --
    see run_search's db_path convention) and lists their runs, most
    recent first, for the family-diversity page's run picker."""
    runs = []
    for db_file in sorted(SEARCH_DIR.glob("search_*.db"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with ResultsDB(db_file) as db:
                for row in db.list_runs(limit=10):
                    runs.append({
                        "db_path": str(db_file), "run_id": row.get("run_id"),
                        "mode": row.get("mode"), "family": row.get("family"),
                        "instrument": row.get("instrument"), "created_at": row.get("created_at"),
                    })
        except Exception:  # noqa: BLE001 -- a corrupt/partial db file must not break the whole picker
            continue
    return runs


@app.route("/family-diversity")
def family_diversity_form():
    db_path = request.args.get("db_path", "")
    run_id = request.args.get("run_id", "")
    stage = request.args.get("stage", "stage1")
    runs = _available_search_runs()

    result_ctx = None
    error = None
    if db_path and run_id:
        try:
            with ResultsDB(Path(db_path)) as db:
                records = db.leaderboard(run_id, stage=stage, top_n=5000)
            if not records:
                error = f"No '{stage}' candidates found for that run -- try a different stage."
            else:
                summaries = summarize_family_performance(records)
                result_ctx = {
                    "run_id": run_id, "stage": stage, "n_records": len(records),
                    "summaries": [s.to_dict() for s in summaries],
                    "report_text": render_family_report(summaries),
                }
        except Exception as exc:  # noqa: BLE001
            error = f"Could not load that run: {exc}"

    return render_template(
        "family_diversity.html", runs=runs, selected_db_path=db_path, selected_run_id=run_id,
        selected_stage=stage, result=result_ctx, error=error,
    )


@app.route("/payout-probability")
def payout_probability_form():
    return render_template(
        "payout_probability.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(), prop_presets_json=_prop_presets_json(), **_alpaca_template_context())


@app.route("/payout-probability/run", methods=["POST"])
def payout_probability_run():
    form = request.form
    ctx = lambda **kw: dict(stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), prop_presets_json=_prop_presets_json(), **kw)
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("payout_probability.html", **ctx(error=dataset_error), **_alpaca_template_context()), 400

        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )

        bt_result = run_backtest(df, strategy, risk)
        if not bt_result.trades:
            return render_template("payout_probability.html", **ctx(
                error="No trades were generated by this strategy over the given data -- there is "
                      "nothing to run a lifecycle simulation on."
            ), **_alpaca_template_context()), 400

        reset_fee_raw = (form.get("reset_fee") or "").strip()
        econ = ResetEconomics(
            evaluation_fee=float(form.get("evaluation_fee", 0) or 0),
            reset_fee=(float(reset_fee_raw) if reset_fee_raw else None),
            profit_split_pct=float(form.get("profit_split", 80) or 80),
            max_attempts=int(form.get("max_attempts", 3) or 3),
        )
        cfg = PropSurvivalConfig(
            n_simulations=min(int(form.get("n_sims", 5000) or 5000), 50_000),
            max_payouts_tracked=int(form.get("max_payouts_tracked", 5) or 5),
            funding_approval_probability=float(form.get("funding_approval_pct", 100) or 100),
            reset_economics=econ,
        )
        result = run_prop_survival_analysis(bt_result.trades, rules, cfg)

        run_id = uuid.uuid4().hex[:10]
        paths = generate_survival_report(
            PAYOUT_DIR, result, bt_result.strategy_name, active_label, rules, cfg,
            basename=f"payout_{run_id}",
        )

        # Optional account-scaling stress test (Owen's ask: "does the
        # expected-payout number get more realistic for firms that scale
        # a funded account up after a run of payouts") -- entirely
        # additive; the plain (non-scaling) numbers above are unaffected
        # either way. See app.prop.scaling's module docstring for the
        # modeling assumptions.
        scaling_result = None
        if form.get("enable_scaling") == "on":
            plan = ScalingPlan(
                payouts_per_scale=int(form.get("scale_payouts_per", 4) or 4),
                scale_multiplier=float(form.get("scale_multiplier", 1.25) or 1.25),
                max_scale_multiple=float(form.get("scale_max_multiple", 4.0) or 4.0),
            )
            try:
                scaling_result = run_scaling_stress_test(
                    bt_result.trades, rules, plan,
                    mc_cfg=None,
                ).to_dict()
            except ValueError:
                scaling_result = None

        # Optional bankroll / EV survival (Owen's ask: "given $X set aside
        # for buying attempts, what fraction of simulated futures let me
        # reach a first payout before running out of money"). Entirely
        # additive; the plain funnel numbers above are unaffected either
        # way. Reuses this SAME request's reset economics (econ, above)
        # so the two sections never show two different prices for the
        # same reset. See app.monte_carlo.bankroll's module docstring.
        bankroll_result = None
        if form.get("enable_bankroll") == "on":
            try:
                bankroll_cfg = BankrollConfig(
                    starting_bankroll=float(form.get("bankroll_amount", 1000) or 1000),
                    reset_economics=econ,
                    n_simulations=min(int(form.get("bankroll_n_sims", 5000) or 5000), 50_000),
                    method=form.get("bankroll_method", "block_bootstrap"),
                    block_size=int(form.get("bankroll_block_size", 5) or 5),
                    stop_after_first_payout=form.get("bankroll_stop_after_first_payout") == "on",
                )
                bankroll_result = simulate_bankroll_survival(bt_result.trades, rules, bankroll_cfg).to_dict()
            except ValueError:
                bankroll_result = None

        return render_template("payout_probability.html", **ctx(result={
            "strategy_name": bt_result.strategy_name,
            "instrument": active_label,
            "score": result.prop_survival_score,
            "funnel": result.funnel.to_dict(),
            "net_positive_after_resets": result.reset_economics.probability_net_positive_after_resets,
            "expected_net_profit_after_resets": result.reset_economics.expected_net_profit_after_resets,
            "notes": result.notes,
            "report_html": f"/payout_reports/{Path(paths['html']).name}",
            "report_json": f"/payout_reports/{Path(paths['json']).name}",
            "scaling": scaling_result,
            "bankroll": bankroll_result,
        }), **_alpaca_template_context())
    except StrategyError as exc:
        return render_template("payout_probability.html", **ctx(error=str(exc)), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("payout_probability.html", **ctx(error=f"Unexpected error: {exc}"), **_alpaca_template_context()), 500


@app.route("/payout_reports/<path:filename>")
def serve_payout_report(filename):
    return send_from_directory(PAYOUT_DIR, filename)


# ---------------------------------------------------------------------------
# Prop-Firm Recommender -- reverse of Payout Probability: given a strategy's
# own trade sequence, score it against EVERY preset firm's rules and rank
# them, instead of checking one firm's rules at a time by hand.
# ---------------------------------------------------------------------------

@app.route("/prop-firm-recommender")
def prop_firm_recommender_form():
    return render_template(
        "prop_firm_recommender.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(), all_presets=list_prop_firm_presets(), **_alpaca_template_context())


@app.route("/prop-firm-recommender/run", methods=["POST"])
def prop_firm_recommender_run():
    form = request.form
    ctx = lambda **kw: dict(
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(), all_presets=list_prop_firm_presets(), **kw,
    )
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("prop_firm_recommender.html", **ctx(error=dataset_error), **_alpaca_template_context()), 400

        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )

        bt_result = run_backtest(df, strategy, risk)
        if not bt_result.trades:
            return render_template("prop_firm_recommender.html", **ctx(
                error="No trades were generated by this strategy over the given data -- there is "
                      "nothing to score against prop-firm rule sets."
            ), **_alpaca_template_context()), 400

        n_sims = min(int(form.get("n_sims", 2000) or 2000), 20_000)
        selected_firms = form.getlist("firms")
        candidates = None
        if selected_firms:
            candidates = [get_prop_firm_preset(key) for key in selected_firms if key]

        recommendations = recommend_prop_firms(
            bt_result.trades, candidates=candidates,
            mc_cfg=MonteCarloConfig(n_simulations=n_sims),
        )

        return render_template("prop_firm_recommender.html", **ctx(result={
            "strategy_name": bt_result.strategy_name,
            "instrument": active_label,
            "recommendations": [r.to_dict() for r in recommendations],
        }), **_alpaca_template_context())
    except StrategyError as exc:
        return render_template("prop_firm_recommender.html", **ctx(error=str(exc)), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("prop_firm_recommender.html", **ctx(error=f"Unexpected error: {exc}"), **_alpaca_template_context()), 500


# ---------------------------------------------------------------------------
# Step 12: Multi-Strategy Ensemble -- the mirror case of Portfolio: N
# DIFFERENT strategies combined on the SAME instrument. "Blend" mode
# reuses run_portfolio_backtest under the hood (fast, synchronous, same
# report template). "Vote" mode is a single combined backtest + Monte
# Carlo, same shape as /run.
# ---------------------------------------------------------------------------

@app.route("/ensemble")
def ensemble_form():
    return render_template(
        "ensemble.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, **_alpaca_template_context())


@app.route("/ensemble/run", methods=["POST"])
def ensemble_run():
    form = request.form
    ctx = lambda **kw: dict(stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **kw)
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("ensemble.html", **ctx(error=dataset_error), **_alpaca_template_context()), 400

        strategies, names = [], []
        for i in range(1, 5):
            f = request.files.get(f"leg{i}_file")
            if not f or not f.filename:
                continue
            mode = _mode_from_filename(f.filename)
            if mode is None:
                return render_template("ensemble.html", **ctx(error=f"'{f.filename}': unrecognized strategy file type (expected .py, .pine, or .mq5)."), **_alpaca_template_context()), 400
            code = f.read().decode("utf-8", errors="replace")
            try:
                strategies.append(build_strategy_from_code(mode, code))
            except StrategyError as exc:
                return render_template("ensemble.html", **ctx(error=f"'{f.filename}': {exc}"), **_alpaca_template_context()), 400
            names.append(Path(f.filename).stem)

        if len(strategies) < 2:
            return render_template("ensemble.html", **ctx(error="An ensemble needs at least 2 strategy legs -- upload at least 2 strategy files below (Python/PineScript/MQL5, mixing types is fine)."), **_alpaca_template_context()), 400

        balance = float(form.get("initial_balance", 100000) or 100000)
        risk = RiskConfig(initial_balance=balance)
        mode = form.get("ensemble_mode", "blend")

        if mode == "vote":
            min_agreement = int(form.get("min_agreement", 2) or 2)
            bt_result = run_ensemble_vote(df, strategies, risk, names=names, vote_config=EnsembleVoteConfig(min_agreement=min_agreement))
            if not bt_result.trades:
                return render_template("ensemble.html", **ctx(error="This vote ensemble produced zero trades on the given data -- nothing to report."), **_alpaca_template_context()), 400
            rules = PropRules(account_size=balance)
            period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
            pnls = [t.pnl for t in bt_result.trades]
            dates = [t.entry_time for t in bt_result.trades]
            single_run = simulate_account(pnls, dates, rules)
            mc_result = run_monte_carlo(bt_result.trades, rules, MonteCarloConfig(n_simulations=int(form.get("n_sims", 2000) or 2000)))
            run_id = uuid.uuid4().hex[:10]
            paths = generate_full_report(
                output_dir=ENSEMBLE_DIR, strategy_name=bt_result.strategy_name, strategy_source_type="ensemble_vote",
                instrument=active_label, timeframe=infer_timeframe_label(df), backtest_period=period, backtest_result=bt_result,
                prop_rules=rules, prop_single_run=single_run, monte_carlo_result=mc_result, basename=f"ensemble_vote_{run_id}",
                risk_config=risk, price_df=df,
            )
            return render_template("ensemble.html", **ctx(result={
                "mode": "vote", "legs": names, "trades": len(bt_result.trades),
                "net_profit": bt_result.statistics.net_profit, "max_dd": bt_result.statistics.max_drawdown_pct,
                "eval_pass_probability": mc_result.evaluation_pass_probability,
                "report_html": f"/ensemble_reports/{Path(paths['html']).name}",
                "report_json": f"/ensemble_reports/{Path(paths['json']).name}",
            }), **_alpaca_template_context())

        config = PortfolioConfig(initial_balance=balance, correlation_penalty_strength=float(form.get("correlation_penalty_strength", 0.6) or 0.6))
        result = run_ensemble_blend(df, strategies, risk, names=names, config=config)
        run_id = uuid.uuid4().hex[:10]
        paths = generate_portfolio_report(ENSEMBLE_DIR, result, basename=f"ensemble_blend_{run_id}")
        return render_template("ensemble.html", **ctx(result={
            "mode": "blend", "legs": names, "trades": result.combined_statistics.total_trades,
            "net_profit": result.combined_statistics.net_profit, "max_dd": result.combined_statistics.max_drawdown_pct,
            "sharpe": result.combined_statistics.sharpe_ratio, "diversification_ratio": result.diversification_ratio,
            "warnings": result.warnings,
            "report_html": f"/ensemble_reports/{Path(paths['html']).name}",
            "report_json": f"/ensemble_reports/{Path(paths['json']).name}",
        }), **_alpaca_template_context())
    except (StrategyError, EnsembleError, PortfolioError) as exc:
        return render_template("ensemble.html", **ctx(error=str(exc)), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("ensemble.html", **ctx(error=f"Unexpected error: {exc}"), **_alpaca_template_context()), 500


@app.route("/ensemble_reports/<path:filename>")
def serve_ensemble_report(filename):
    return send_from_directory(ENSEMBLE_DIR, filename)


# ---------------------------------------------------------------------------
# Step 13: CPCV (Combinatorial Purged Cross-Validation) -- re-backtests one
# strategy across every combinatorial train/test partition of purged,
# embargoed groups, instead of trusting a single train/test split. Genuine
# multi-candidate PBO (Probability of Backtest Overfitting) needs a POOL of
# already-tried candidates (e.g. a Search Lab leaderboard or a Refinement
# run's final generation) as input rather than a single strategy config, so
# it isn't wired up here yet -- see WEB_PARITY_ROADMAP.md.
# ---------------------------------------------------------------------------

_CPCV_JOBS: dict[str, dict] = {}
_CPCV_JOBS_LOCK = threading.Lock()


def _cpcv_job_log(job_id: str, msg: str) -> None:
    with _CPCV_JOBS_LOCK:
        job = _CPCV_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_cpcv_job(job_id: str, df, strategy, risk: RiskConfig, n_groups: int, n_test_groups: int, embargo_frac: float, metric: str, robustness_threshold: float, max_paths: int, prop_rules=None, strategy_name: str = "", instrument: str = "", library_ref: tuple[str, str] | None = None) -> None:
    try:
        _cpcv_job_log(job_id, f"Running CPCV: {n_groups} groups, {n_test_groups} held out per path, metric={metric}...")
        result = run_cpcv(
            df, lambda: strategy, risk, n_groups=n_groups, n_test_groups=n_test_groups,
            embargo_frac=embargo_frac, metric=metric, robustness_threshold=robustness_threshold, max_paths=max_paths,
            prop_rules=prop_rules,
        )
        _cpcv_job_log(job_id, f"Done: {result.n_paths} paths evaluated.")
        paths = generate_cpcv_report(CPCV_DIR, result, basename=f"cpcv_{job_id}")
        report_html = f"/cpcv_reports/{Path(paths['html']).name}"
        with _CPCV_JOBS_LOCK:
            job = _CPCV_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = report_html
            job["report_json"] = f"/cpcv_reports/{Path(paths['json']).name}"
        strategy_state.record_validation(
            strategy_name, instrument, "cpcv",
            passed=bool(result.is_robust), summary=f"{result.n_paths} paths evaluated", report_html=report_html,
        )
        # Pipeline-progress tracker: only when this strategy came straight
        # from the Strategy Library (library_ref set), so a standalone/
        # uploaded-for-this-run-only strategy with no library entry has
        # nothing to stamp onto -- same guard every other library_ref
        # call site in this file already uses.
        if library_ref:
            try:
                record_validation_result(*library_ref, {
                    "method": "cpcv", "n_paths": result.n_paths,
                    "is_robust": bool(result.is_robust), "report_html": report_html,
                })
            except Exception:  # noqa: BLE001 -- recording to the library is a convenience, not core output
                pass
    except CPCVError as exc:
        with _CPCV_JOBS_LOCK:
            job = _CPCV_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _CPCV_JOBS_LOCK:
            job = _CPCV_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_CPCV)


@app.route("/cpcv")
def cpcv_form():
    return render_template("cpcv.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), strategy_statuses=STRATEGY_STATUSES, **_alpaca_template_context())


@app.route("/cpcv/start", methods=["POST"])
def cpcv_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_CPCV, "cpcv.html", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_CPCV)
            return render_template("cpcv.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        prop_rules = PropRules(account_size=float(form.get("initial_balance", 100000)))
        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _CPCV_JOBS_LOCK:
            _CPCV_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(
            target=_run_cpcv_job,
            args=(
                job_id, df, strategy, risk,
                int(form.get("n_groups", 6) or 6), int(form.get("n_test_groups", 2) or 2),
                float(form.get("embargo_frac", 0.01) or 0.01), form.get("metric", "eval_pass_probability"),
                float(form.get("robustness_threshold", 0.5) or 0.5), int(form.get("max_paths", 30) or 30),
                prop_rules,
            ),
            kwargs={"strategy_name": getattr(strategy, "name", "Strategy"), "instrument": active_label, "library_ref": library_ref},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("cpcv_job", job_id=job_id))
    except (StrategyError, CPCVError) as exc:
        HEAVY_JOB_GUARD.release(JOB_CPCV)
        return render_template("cpcv.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_CPCV)
        return render_template("cpcv.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 500


@app.route("/cpcv/job/<job_id>")
def cpcv_job(job_id):
    with _CPCV_JOBS_LOCK:
        job = _CPCV_JOBS.get(job_id)
    if job is None:
        return render_template("cpcv_job.html", job_id=job_id, not_found=True), 404
    return render_template("cpcv_job.html", job_id=job_id, not_found=False)


@app.route("/cpcv/job/<job_id>/status.json")
def cpcv_job_status(job_id):
    with _CPCV_JOBS_LOCK:
        job = _CPCV_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = result.to_dict()
        summary["report_html"] = job.get("report_html")
        summary["report_json"] = job.get("report_json")
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


@app.route("/cpcv_reports/<path:filename>")
def serve_cpcv_report(filename):
    return send_from_directory(CPCV_DIR, filename)


# ---------------------------------------------------------------------------
# PBO (Probability of Backtest Overfitting) -- candidate-pool picker.
# README's own acknowledged web/desktop-parity gap: compute_pbo() and
# generate_pbo_report() already existed (used by the --pbo CLI flag), but
# no web route/UI ever called them. Reuses the exact same job-thread/guard
# pattern as CPCV above. The candidate pool is built from THREE sources,
# combined:
#   1. the strategy configured in the form itself (always candidate 0),
#   2. any strategies the user checks from the Strategy Library (any of
#      the 4 source types, loaded via app.strategy.library_loader and
#      converted to a uniform spec via app.search.strategy_space.spec_from_strategy),
#   3. N random perturbations of the form strategy's own tunable numeric
#      parameters (manual-config strategies only -- mirrors app.main.py's
#      run_pbo_cli, which only ever perturbed the default manual config).
# PBO is only meaningful for 2+ candidates (see compute_pbo's own
# docstring), so the route requires at least one pool strategy or
# perturbed variant on top of the form strategy.
# ---------------------------------------------------------------------------

_PBO_JOBS: dict[str, dict] = {}
_PBO_JOBS_LOCK = threading.Lock()


def _pbo_job_log(job_id: str, msg: str) -> None:
    with _PBO_JOBS_LOCK:
        job = _PBO_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_pbo_job(job_id: str, df, specs: list[dict], risk: RiskConfig, n_groups: int, n_test_groups: int, embargo_frac: float, metric: str, max_paths: int, prop_rules=None, strategy_name: str = "", instrument: str = "") -> None:
    try:
        _pbo_job_log(job_id, f"Running PBO across {len(specs)} candidate(s): {n_groups} groups, {n_test_groups} held out per path, metric={metric}...")
        result = compute_pbo(
            df, specs, risk, n_groups=n_groups, n_test_groups=n_test_groups,
            embargo_frac=embargo_frac, metric=metric, max_paths=max_paths, prop_rules=prop_rules,
        )
        _pbo_job_log(job_id, f"Done: {result.n_paths} paths evaluated, PBO = {result.pbo * 100:.1f}%.")
        paths = generate_pbo_report(PBO_DIR, result, basename=f"pbo_{job_id}")
        report_html = f"/pbo_reports/{Path(paths['html']).name}"
        with _PBO_JOBS_LOCK:
            job = _PBO_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = report_html
            job["report_json"] = f"/pbo_reports/{Path(paths['json']).name}"
        # PBO is diagnostic (how likely is picking a winner among these
        # candidates to be noise), not itself a pass/fail gate -- a LOW
        # pbo is the good outcome, so "passed" tracks that directly.
        strategy_state.record_validation(
            strategy_name, instrument, "pbo",
            passed=bool(result.pbo < 0.5), summary=f"PBO {result.pbo * 100:.1f}% across {result.n_candidates} candidates", report_html=report_html,
        )
    except CPCVError as exc:
        with _PBO_JOBS_LOCK:
            job = _PBO_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _PBO_JOBS_LOCK:
            job = _PBO_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_PBO)


def _pool_strategy_specs(pool_refs: list[str]) -> tuple[list[dict], list[str]]:
    """pool_refs: 'type:name' strings from the form's checked Strategy
    Library entries. Returns (specs, warnings) -- a library entry that
    fails to load (deleted/corrupt file) is skipped with a warning rather
    than failing the whole PBO run."""
    specs, warnings = [], []
    for ref in pool_refs:
        if ":" not in ref:
            continue
        strategy_type, name = ref.split(":", 1)
        try:
            matches = [s for s in list_saved_strategies(strategy_type) if s.name == name]
            if not matches:
                warnings.append(f"Skipped '{name}': no longer in the {strategy_type} library.")
                continue
            strategy = load_strategy_object(matches[0])
            specs.append(spec_from_strategy(strategy))
        except (StrategyError, OSError, ValueError) as exc:
            warnings.append(f"Skipped '{name}': {exc}")
    return specs, warnings


def _perturbed_variant_specs(strategy, n_variants: int, seed: int) -> list[dict]:
    """N random perturbations of `strategy`'s own tunable numeric
    parameters, +/-30% of each gene's own range around its base value --
    identical perturbation logic to app.main.py's run_pbo_cli, generalized
    from manual-only to any of the 4 source types via the same gene
    discovery/apply machinery Iterative Refinement's GA already uses."""
    if n_variants <= 0:
        return []
    rng = random.Random(seed)
    out = []
    if strategy.source_type == "manual":
        genes = extract_genome(strategy.config)
        if not genes:
            return []
        for _ in range(n_variants):
            genome = [max(min(g.base_value + rng.uniform(-0.3, 0.3) * (g.hi - g.lo), g.hi), g.lo) for g in genes]
            out.append({"source_type": "manual", "config": apply_genome(strategy.config, genes, genome)})
    else:
        genes = discover_code_genes(strategy)
        if not genes:
            return []
        base_spec = spec_from_strategy(strategy)
        for _ in range(n_variants):
            genome = [max(min(g.base_value + rng.uniform(-0.3, 0.3) * (g.hi - g.lo), g.hi), g.lo) for g in genes]
            code_text = apply_code_genome(base_spec["code_text"], genes, genome)
            out.append({"source_type": strategy.source_type, "code_text": code_text, "code_extension": base_spec["code_extension"]})
    return out


@app.route("/pbo")
def pbo_form():
    return render_template(
        "pbo.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        saved_strategies_json=_saved_strategies_json(), strategy_statuses=STRATEGY_STATUSES, **_alpaca_template_context())


@app.route("/pbo/start", methods=["POST"])
def pbo_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_PBO, "pbo.html", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_PBO)
            return render_template("pbo.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)

        pool_refs = [r for r in form.getlist("pool_strategy") if r]
        n_variants = int(form.get("n_perturbed_variants", 0) or 0)
        seed = int(form.get("seed", 42) or 42)
        pool_specs, pool_warnings = _pool_strategy_specs(pool_refs)
        variant_specs = _perturbed_variant_specs(strategy, n_variants, seed)
        specs = [spec_from_strategy(strategy)] + pool_specs + variant_specs

        if len(specs) < 2:
            HEAVY_JOB_GUARD.release(JOB_PBO)
            return render_template(
                "pbo.html", error="PBO needs at least 2 candidates -- check one or more Strategy Library entries and/or set 'perturbed variants' above 0.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        prop_rules = PropRules(account_size=float(form.get("initial_balance", 100000)))
        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}.", f"Candidate pool: {len(specs)} ({1} form strategy + {len(pool_specs)} library + {len(variant_specs)} perturbed)."]
        if import_note:
            initial_log.append(import_note)
        initial_log.extend(pool_warnings)
        with _PBO_JOBS_LOCK:
            _PBO_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(
            target=_run_pbo_job,
            args=(
                job_id, df, specs, risk,
                int(form.get("n_groups", 6) or 6), int(form.get("n_test_groups", 2) or 2),
                float(form.get("embargo_frac", 0.01) or 0.01), form.get("metric", "sharpe_ratio"),
                int(form.get("max_paths", 30) or 30), prop_rules,
            ),
            kwargs={"strategy_name": getattr(strategy, "name", "Strategy"), "instrument": active_label},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("pbo_job", job_id=job_id))
    except (StrategyError, CPCVError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_PBO)
        return render_template("pbo.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_PBO)
        return render_template("pbo.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 500


@app.route("/pbo/job/<job_id>")
def pbo_job(job_id):
    with _PBO_JOBS_LOCK:
        job = _PBO_JOBS.get(job_id)
    if job is None:
        return render_template("pbo_job.html", job_id=job_id, not_found=True), 404
    return render_template("pbo_job.html", job_id=job_id, not_found=False)


@app.route("/pbo/job/<job_id>/status.json")
def pbo_job_status(job_id):
    with _PBO_JOBS_LOCK:
        job = _PBO_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = result.to_dict()
        summary["report_html"] = job.get("report_html")
        summary["report_json"] = job.get("report_json")
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


@app.route("/pbo_reports/<path:filename>")
def serve_pbo_report(filename):
    return send_from_directory(PBO_DIR, filename)


# ---------------------------------------------------------------------------
# Step 14: Parameter Sensitivity -- sweeps every tunable numeric parameter
# independently across +/- a percent range, holding others fixed, and flags
# any "cliff" (a narrow-edge parameter rather than a stable plateau). Also
# supports an on-demand 2D heatmap for any two parameter labels the 1D sweep
# discovered -- a genuine two-step UI (discover names, then pick 2) rather
# than the auto-picked-pairs shortcut Parameter Robustness already offers.
# ---------------------------------------------------------------------------

_SENS_JOBS: dict[str, dict] = {}
_SENS_JOBS_LOCK = threading.Lock()


def _sens_job_log(job_id: str, msg: str) -> None:
    with _SENS_JOBS_LOCK:
        job = _SENS_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_sensitivity_job(job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig, metric: str, pct_range: float, n_steps: int, max_params: int, strategy_name: str = "", instrument: str = "") -> None:
    try:
        _sens_job_log(job_id, f"Sweeping up to {max_params} tunable parameter(s), {n_steps} steps each, metric={metric}...")
        results = compute_1d_sensitivity(df, strategy, risk, rules, mc_cfg, metric=metric, pct_range=pct_range, n_steps=n_steps, max_params=max_params)
        _sens_job_log(job_id, f"Done: swept {len(results)} parameter(s).")
        paths = generate_sensitivity_report(SENSITIVITY_DIR, results, basename=f"sensitivity_{job_id}")
        report_html = f"/sensitivity_reports/{Path(paths['html']).name}"
        with _SENS_JOBS_LOCK:
            job = _SENS_JOBS[job_id]
            job["done"] = True
            job["results"] = results
            job["report_html"] = report_html
            job["report_json"] = f"/sensitivity_reports/{Path(paths['json']).name}"
            # Kept for an on-demand 2D heatmap requested from the job page --
            # see _run_sensitivity_heatmap_job below. Not put in the JSON
            # status payload (df/strategy objects aren't serializable).
            job["_ctx"] = {"df": df, "strategy": strategy, "risk": risk, "rules": rules, "mc_cfg": mc_cfg, "metric": metric}
        # This tool is diagnostic (flags cliffs vs. stable plateaus per
        # parameter) rather than pass/fail -- passed=None records that it ran.
        strategy_state.record_validation(
            strategy_name, instrument, "sensitivity",
            passed=None, summary=f"{len(results)} parameter(s) swept", report_html=report_html,
        )
    except RefinementError as exc:
        with _SENS_JOBS_LOCK:
            job = _SENS_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _SENS_JOBS_LOCK:
            job = _SENS_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_SENSITIVITY)


def _run_sensitivity_heatmap_job(job_id: str, param_a: str, param_b: str, pct_range: float, n_steps: int) -> None:
    with _SENS_JOBS_LOCK:
        job = _SENS_JOBS.get(job_id)
        ctx = job.get("_ctx") if job else None
    if job is None or ctx is None:
        return
    try:
        _sens_job_log(job_id, f"Running 2D heatmap for {param_a} x {param_b}...")
        heatmap = compute_2d_heatmap(
            ctx["df"], ctx["strategy"], ctx["risk"], ctx["rules"], ctx["mc_cfg"],
            param_a, param_b, metric=ctx["metric"], pct_range=pct_range, n_steps=n_steps,
        )
        results = job.get("results") or []
        paths = generate_sensitivity_report(SENSITIVITY_DIR, results, heatmap, basename=f"sensitivity_{job_id}")
        with _SENS_JOBS_LOCK:
            job = _SENS_JOBS[job_id]
            job["heatmap_done"] = True
            job["heatmap_running"] = False
            job["heatmap_error"] = None
            job["heatmap"] = heatmap
            job["report_html"] = f"/sensitivity_reports/{Path(paths['html']).name}"
        _sens_job_log(job_id, "2D heatmap done.")
    except RefinementError as exc:
        with _SENS_JOBS_LOCK:
            job = _SENS_JOBS[job_id]
            job["heatmap_done"] = True
            job["heatmap_running"] = False
            job["heatmap_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _SENS_JOBS_LOCK:
            job = _SENS_JOBS[job_id]
            job["heatmap_done"] = True
            job["heatmap_running"] = False
            job["heatmap_error"] = f"Unexpected error: {exc}"


@app.route("/sensitivity")
def sensitivity_form():
    return render_template("sensitivity.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), strategy_statuses=STRATEGY_STATUSES, **_alpaca_template_context())


@app.route("/sensitivity/start", methods=["POST"])
def sensitivity_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_SENSITIVITY, "sensitivity.html", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_SENSITIVITY)
            return render_template("sensitivity.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(initial_balance=float(form.get("initial_balance", 100000)), pip_size=float(form.get("pip_size", 0.0001)))
        rules = PropRules(account_size=float(form.get("account_size", 100000)))
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("mc_sims", 500) or 500))

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _SENS_JOBS_LOCK:
            _SENS_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "results": None, "started_at": time.time(), "instrument": active_label, "heatmap_done": False, "heatmap_error": None, "heatmap": None, "_ctx": None}
        thread = threading.Thread(
            target=_run_sensitivity_job,
            args=(
                job_id, df, strategy, risk, rules, mc_cfg, form.get("metric", "profit_factor"),
                float(form.get("pct_range", 0.5) or 0.5), int(form.get("n_steps", 9) or 9), int(form.get("max_params", 8) or 8),
            ),
            kwargs={"strategy_name": getattr(strategy, "name", "Strategy"), "instrument": active_label},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("sensitivity_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_SENSITIVITY)
        return render_template("sensitivity.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_SENSITIVITY)
        return render_template("sensitivity.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 500


@app.route("/sensitivity/job/<job_id>")
def sensitivity_job(job_id):
    with _SENS_JOBS_LOCK:
        job = _SENS_JOBS.get(job_id)
    if job is None:
        return render_template("sensitivity_job.html", job_id=job_id, not_found=True), 404
    return render_template("sensitivity_job.html", job_id=job_id, not_found=False)


@app.route("/sensitivity/job/<job_id>/status.json")
def sensitivity_job_status(job_id):
    with _SENS_JOBS_LOCK:
        job = _SENS_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    results = job.get("results")
    summary = None
    if results is not None:
        summary = {
            "sweeps": [r.to_dict() for r in results],
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
            # Parameter labels the 1D sweep actually discovered -- the web
            # UI's heatmap picker only ever offers a pair from this list, so
            # it can never request a label compute_2d_heatmap doesn't know.
            "available_params": [r.gene_label for r in results],
        }
    heatmap = job.get("heatmap")
    return jsonify({
        "found": True, "done": job["done"], "error": job["error"], "log": job["log"],
        "instrument": job.get("instrument"), "summary": summary,
        "heatmap_available": job.get("_ctx") is not None,
        "heatmap_running": bool(job.get("heatmap_running")),
        "heatmap_done": job.get("heatmap_done", False),
        "heatmap_error": job.get("heatmap_error"),
        "heatmap": heatmap.to_dict() if heatmap is not None else None,
    })


@app.route("/sensitivity/job/<job_id>/heatmap", methods=["POST"])
def sensitivity_job_heatmap(job_id):
    with _SENS_JOBS_LOCK:
        job = _SENS_JOBS.get(job_id)
        if job is None or job.get("_ctx") is None:
            return jsonify({"ok": False, "error": "This job has no data available for a 2D heatmap (still running, or it failed)."}), 400
        available = {r.gene_label for r in (job.get("results") or [])}
    form = request.form
    param_a, param_b = form.get("param_a", ""), form.get("param_b", "")
    if not param_a or not param_b or param_a == param_b:
        return jsonify({"ok": False, "error": "Pick two different parameters."}), 400
    if param_a not in available or param_b not in available:
        return jsonify({"ok": False, "error": "Unknown parameter label -- pick from the discovered list."}), 400
    with _SENS_JOBS_LOCK:
        job["heatmap_done"] = False
        job["heatmap_error"] = None
        job["heatmap"] = None
        job["heatmap_running"] = True
    thread = threading.Thread(
        target=_run_sensitivity_heatmap_job,
        args=(job_id, param_a, param_b, float(form.get("pct_range", 0.5) or 0.5), int(form.get("n_steps", 7) or 7)),
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True})


@app.route("/sensitivity_reports/<path:filename>")
def serve_sensitivity_report(filename):
    return send_from_directory(SENSITIVITY_DIR, filename)


# ---------------------------------------------------------------------------
# Parameter Stability / Robustness Map -- "did I discover a robust edge, or
# the exact historical combination that happened to work?" Reuses
# app.validation.parameter_robustness.compute_parameter_robustness
# (which itself reuses compute_1d_sensitivity + compute_2d_heatmap
# UNMODIFIED) rather than recomputing any sweep logic here -- this route is
# purely the job-queue/rendering wiring, same shape as Sensitivity above.
# ---------------------------------------------------------------------------

_PARAM_ROBUSTNESS_JOBS: dict[str, dict] = {}
_PARAM_ROBUSTNESS_JOBS_LOCK = threading.Lock()


def _param_robustness_job_log(job_id: str, msg: str) -> None:
    with _PARAM_ROBUSTNESS_JOBS_LOCK:
        job = _PARAM_ROBUSTNESS_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_param_robustness_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig,
    metric: str, pass_threshold_pct: float, pct_range: float, n_steps_1d: int, n_steps_2d: int,
    max_params: int, n_heatmap_pairs: int, strategy_name: str = "", instrument: str = "",
) -> None:
    try:
        _param_robustness_job_log(
            job_id,
            f"Sweeping up to {max_params} tunable parameter(s) individually, then heatmapping the "
            f"{n_heatmap_pairs} most sensitive pair(s), metric={metric}, pass threshold={pass_threshold_pct:g}%...",
        )
        result = compute_parameter_robustness(
            df, strategy, risk, rules, mc_cfg, metric=metric, pass_threshold_pct=pass_threshold_pct,
            max_params=max_params, pct_range=pct_range, n_steps_1d=n_steps_1d, n_steps_2d=n_steps_2d,
            n_heatmap_pairs=n_heatmap_pairs,
        )
        _param_robustness_job_log(
            job_id,
            f"Done: {result.n_parameters_checked} parameter(s) checked, {result.n_cliffs_detected} "
            f"cliff(s) detected. Parameter Robustness Score: {result.parameter_robustness_score:.1f}/100.",
        )
        with _PARAM_ROBUSTNESS_JOBS_LOCK:
            job = _PARAM_ROBUSTNESS_JOBS[job_id]
            job["done"] = True
            job["result"] = result
        # Diagnostic, not pass/fail on its own -- passed=None records that it ran, same convention
        # as Sensitivity's own record_validation call above.
        strategy_state.record_validation(
            strategy_name, instrument, "parameter_robustness",
            passed=None, summary=f"Parameter Robustness Score {result.parameter_robustness_score:.1f}/100",
        )
    except RefinementError as exc:
        with _PARAM_ROBUSTNESS_JOBS_LOCK:
            job = _PARAM_ROBUSTNESS_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _PARAM_ROBUSTNESS_JOBS_LOCK:
            job = _PARAM_ROBUSTNESS_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_PARAMETER_ROBUSTNESS)


@app.route("/parameter-robustness")
def parameter_robustness_form():
    return render_template(
        "parameter_robustness.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(),
        dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, **_alpaca_template_context())


@app.route("/parameter-robustness/start", methods=["POST"])
def parameter_robustness_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_PARAMETER_ROBUSTNESS, "parameter_robustness.html", stored_datasets=list_stored_datasets(),
        dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_PARAMETER_ROBUSTNESS)
            return render_template(
                "parameter_robustness.html", error=dataset_error, stored_datasets=list_stored_datasets(),
                dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(initial_balance=float(form.get("initial_balance", 100000)), pip_size=float(form.get("pip_size", 0.0001)))
        rules = PropRules(account_size=float(form.get("account_size", 100000)))
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("mc_sims", 500) or 500))

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _PARAM_ROBUSTNESS_JOBS_LOCK:
            _PARAM_ROBUSTNESS_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(
            target=_run_param_robustness_job,
            args=(
                job_id, df, strategy, risk, rules, mc_cfg, form.get("metric", "eval_pass_probability"),
                float(form.get("pass_threshold_pct", 50.0) or 50.0), float(form.get("pct_range", 0.5) or 0.5),
                int(form.get("n_steps_1d", 9) or 9), int(form.get("n_steps_2d", 7) or 7),
                int(form.get("max_params", 6) or 6), int(form.get("n_heatmap_pairs", 1) or 1),
            ),
            kwargs={"strategy_name": getattr(strategy, "name", "Strategy"), "instrument": active_label},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("parameter_robustness_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_PARAMETER_ROBUSTNESS)
        return render_template(
            "parameter_robustness.html", error=str(exc), stored_datasets=list_stored_datasets(),
            dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_PARAMETER_ROBUSTNESS)
        return render_template(
            "parameter_robustness.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(),
            dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 500


@app.route("/parameter-robustness/job/<job_id>")
def parameter_robustness_job(job_id):
    with _PARAM_ROBUSTNESS_JOBS_LOCK:
        job = _PARAM_ROBUSTNESS_JOBS.get(job_id)
    if job is None:
        return render_template("parameter_robustness_job.html", job_id=job_id, not_found=True), 404
    return render_template("parameter_robustness_job.html", job_id=job_id, not_found=False)


@app.route("/parameter-robustness/job/<job_id>/status.json")
def parameter_robustness_job_status(job_id):
    with _PARAM_ROBUSTNESS_JOBS_LOCK:
        job = _PARAM_ROBUSTNESS_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = result.to_dict() if result is not None else None
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


# ---------------------------------------------------------------------------
# Quick Optimize -- one-click single-strategy GA tune from the Strategy
# Library (reuses the same walk-forward-aware GA as Full Pipeline/Step 06,
# saves the winner back into the library tagged "draft"). Same background-
# job/poll pattern as the other GA-driven tabs.
# ---------------------------------------------------------------------------

_QUICKOPT_JOBS: dict[str, dict] = {}
_QUICKOPT_JOBS_LOCK = threading.Lock()


def _quickopt_job_log(job_id: str, msg: str) -> None:
    with _QUICKOPT_JOBS_LOCK:
        job = _QUICKOPT_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_quickopt_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, cfg: QuickOptimizeConfig,
    cancel_event: threading.Event | None = None,
) -> None:
    try:
        result = run_quick_optimize(
            df, strategy, risk, rules, cfg, progress_cb=lambda msg: _quickopt_job_log(job_id, msg),
            cancel_event=cancel_event,
        )
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["result"] = result
    except WalkforwardGACancelled:
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["cancelled"] = True
    except RefinementError as exc:
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"


def _run_quickopt_sweep_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, cfg: QuickOptimizeConfig,
    timeframes: list[str], cancel_event: threading.Event | None = None,
) -> None:
    """Same shape as _run_quickopt_job, but drives
    app.orchestration.quick_optimize.run_quick_optimize_sweep instead of a
    single run. `job["result"]` ends up as the WINNING timeframe's own
    QuickOptimizeResult -- so quickopt_job_status's existing summary-
    building code below needs no changes at all to render it; this only
    adds `job["sweep_timeframes"]`, a plain label -> headline-numbers dict
    for the extra per-timeframe comparison table on the job page.
    """
    try:
        sweep = run_quick_optimize_sweep(
            df, strategy, risk, rules, timeframes, cfg,
            progress_cb=lambda msg: _quickopt_job_log(job_id, msg), cancel_event=cancel_event,
        )
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["result"] = sweep.best_result
            job["best_timeframe"] = sweep.best_timeframe
            job["sweep_timeframes"] = {
                label: {
                    "optimized_eval_pass_probability": res.optimized_eval_pass_probability,
                    "optimized_win_rate": res.optimized_win_rate,
                    "optimized_net_profit": res.optimized_net_profit,
                    "optimized_trades": res.optimized_trades,
                    "is_best": label == sweep.best_timeframe,
                }
                for label, res in sweep.per_timeframe.items()
            }
            if sweep.skipped:
                job["log"].append(
                    "Skipped from the timeframe sweep: "
                    + "; ".join(f"{s.requested_label} ({s.reason})" for s in sweep.skipped)
                )
            for label, err in sweep.errors.items():
                job["log"].append(f"[{label}] could not be optimized: {err}")
    except WalkforwardGACancelled:
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["cancelled"] = True
    except RefinementError as exc:
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"


@app.route("/quick-optimize")
def quickopt_form():
    return render_template("quick_optimize.html", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), strategy_statuses=STRATEGY_STATUSES, fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), **_alpaca_template_context())


@app.route("/quick-optimize/start", methods=["POST"])
def quickopt_start():
    form = request.form
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("quick_optimize.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400
        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(initial_balance=float(form.get("initial_balance", 100000)), pip_size=float(form.get("pip_size", 0.0001)))
        rules = PropRules(account_size=float(form.get("account_size", 100000)))

        # T58 BACKTEST INTEGRITY CHECK -- same pre-flight gate as Run &
        # Report's /run route (see app.validation.integrity_check's own
        # docstring): catches corrupt data, an unsupportable timeframe, or
        # a confirmed lookahead leak BEFORE spending GA compute on it.
        integrity_report = run_integrity_check(
            df, strategy, risk, prop_rules=rules,
            requested_timeframe=form.get("timeframe") or None,
            data_label=active_label,
        )
        if integrity_report.status == "BLOCKED":
            return render_template(
                "quick_optimize.html", error=integrity_report.render(),
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
                **_alpaca_template_context(),
            ), 400

        cfg = QuickOptimizeConfig(
            ga_population=int(form.get("ga_population", 16) or 16),
            ga_generations=int(form.get("ga_generations", 8) or 8),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            optimizer_mode=form.get("optimizer_mode", "genetic") or "genetic",
            n_folds=int(form.get("n_folds", 4) or 4),
            save_to_library=form.get("save_to_library", "on") == "on",
            adaptive_risk_enabled=form.get("adaptive_risk_enabled") == "on",
            reset_on_breach=form.get("reset_on_breach", "on") == "on",
            # Point (5) of the 2026-09-17 fix: opt-in holdout carve-out --
            # unchecked by default, so an existing bookmark/saved form that
            # doesn't send this field keeps today's fully-in-sample
            # behavior. See QuickOptimizeConfig.reserve_holdout's docstring.
            reserve_holdout=form.get("reserve_holdout") == "on",
            holdout_frac=float(form.get("holdout_frac", 0.2) or 0.2),
        )
        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        cancel_event = threading.Event()
        with _QUICKOPT_JOBS_LOCK:
            _QUICKOPT_JOBS[job_id] = {
                "log": initial_log, "done": False, "error": None, "result": None,
                "started_at": time.time(), "instrument": active_label,
                "cancel_event": cancel_event, "cancelled": False,
            }
        # FIX (multi-timeframe sweep): "Timeframes to test" runs the SAME
        # GA once per requested timeframe (df resampled per timeframe --
        # see app.data.timeframe_sweep) and keeps the winner, exactly like
        # a single Quick Optimize run otherwise -- see
        # app.orchestration.quick_optimize.run_quick_optimize_sweep.
        expand_labels = parse_sweep_timeframes(form.get("expand_timeframes", ""))
        if expand_labels:
            thread = threading.Thread(
                target=_run_quickopt_sweep_job, args=(job_id, df, strategy, risk, rules, cfg, expand_labels, cancel_event),
                daemon=True,
            )
        else:
            thread = threading.Thread(target=_run_quickopt_job, args=(job_id, df, strategy, risk, rules, cfg, cancel_event), daemon=True)
        thread.start()
        return redirect(url_for("quickopt_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        return render_template("quick_optimize.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("quick_optimize.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 500


@app.route("/quick-optimize/job/<job_id>")
def quickopt_job(job_id):
    with _QUICKOPT_JOBS_LOCK:
        job = _QUICKOPT_JOBS.get(job_id)
    if job is None:
        return render_template("quick_optimize_job.html", job_id=job_id, not_found=True), 404
    return render_template("quick_optimize_job.html", job_id=job_id, not_found=False)


@app.route("/quick-optimize/job/<job_id>/stop", methods=["POST"])
def quickopt_job_stop(job_id):
    """Signals cancellation to a running Quick Optimize job -- checked
    once per GA generation (see run_walkforward_aware_refinement's
    cancel_event param), so this stops at the next generation boundary."""
    with _QUICKOPT_JOBS_LOCK:
        job = _QUICKOPT_JOBS.get(job_id)
        if job is None:
            return jsonify({"found": False}), 404
        cancel_event = job.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    return jsonify({"found": True, "stopping": True})


@app.route("/quick-optimize/job/<job_id>/status.json")
def quickopt_job_status(job_id):
    with _QUICKOPT_JOBS_LOCK:
        job = _QUICKOPT_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = {
            "strategy_display_name": result.strategy_display_name,
            "baseline_trades": result.baseline_trades, "baseline_net_profit": result.baseline_net_profit,
            "baseline_win_rate": result.baseline_win_rate, "baseline_eval_pass_probability": result.baseline_eval_pass_probability,
            "optimized_trades": result.optimized_trades, "optimized_net_profit": result.optimized_net_profit,
            "optimized_win_rate": result.optimized_win_rate, "optimized_eval_pass_probability": result.optimized_eval_pass_probability,
            "improved": result.improved,
            "saved_library_note": result.saved_library_note,
            "elapsed_seconds": result.elapsed_seconds,
            "warnings": result.warnings,
            # 2026-09-17 Quick-Optimize-vs-Full-Pipeline follow-up fields --
            # see app.orchestration.quick_optimize.QuickOptimizeResult for
            # what each of these means; the template renders all of them
            # so this tool can never again look like a finished, validated
            # answer purely because the UI didn't ask for the caveat.
            "validated": result.validated,
            "result_banner": result.result_banner,
            "result_banner_detail": result.result_banner_detail,
            "oos_trade_count": result.oos_trade_count,
            "min_trade_count_met": result.min_trade_count_met,
            "trade_count_warning": result.trade_count_warning,
            "significance_note": result.significance_note,
            "icir_gate_skip_reason": result.icir_gate_skip_reason,
            "icir_gate_ok": (result.icir_gate.ok if result.icir_gate is not None else None),
            "icir_gate_reasons": (result.icir_gate.reasons if result.icir_gate is not None else []),
            "parsimony_note": result.parsimony_note,
            "parsimony_score": (result.parsimony_result.score if result.parsimony_result is not None else None),
            "holdout_enabled": result.holdout_enabled,
            "holdout_trades": result.holdout_trades,
            "holdout_net_profit": result.holdout_net_profit,
            "holdout_win_rate": result.holdout_win_rate,
            "holdout_eval_pass_probability": result.holdout_eval_pass_probability,
            "holdout_payout_probability": result.holdout_payout_probability,
            "holdout_note": result.holdout_note,
        }
    return jsonify({
        "found": True, "done": job["done"], "error": job["error"], "cancelled": job.get("cancelled", False),
        "log": job["log"], "instrument": job.get("instrument"), "summary": summary,
        "best_timeframe": job.get("best_timeframe"), "sweep_timeframes": job.get("sweep_timeframes"),
    })


# ---------------------------------------------------------------------------
# Evolution Lab -- an open-ended, resumable multi-family GA search that
# runs generation after generation until STOP is clicked (or the process
# restarts, in which case its own on-disk checkpoint resumes it). Unlike
# every other tab above, this isn't a "start a job, wait for it to finish"
# shape -- app.evolution.engine.EvolutionRunner already owns its own
# background thread and start()/stop()/status() control surface, so the
# web app holds ONE global runner instance (matches this app's existing
# single-user/LAN trust model) and drives it directly rather than
# reimplementing job management around it.
# ---------------------------------------------------------------------------

_EVOLUTION_RUNNER: EvolutionRunner | None = None
_EVOLUTION_LOCK = threading.Lock()
_EVOLUTION_LOG: list[str] = []
_EVOLUTION_LOG_MAX = 500


def _evolution_log(msg: str) -> None:
    _EVOLUTION_LOG.append(msg)
    del _EVOLUTION_LOG[:-_EVOLUTION_LOG_MAX]


# Lets HEAVY_JOB_GUARD self-heal if Evolution Lab's slot ever gets stuck
# held (its normal release path is the /evolution/status.json poll below
# noticing is_running went False -- if nothing polls it, or its thread
# gets wedged, every other heavy job used to be refused forever). See
# HeavyJobGuard.register_health_check's docstring for the full reasoning.
HEAVY_JOB_GUARD.register_health_check(
    JOB_EVOLUTION_LAB,
    lambda: _EVOLUTION_RUNNER is not None and _EVOLUTION_RUNNER.is_running,
)


@app.route("/evolution")
def evolution_form():
    return render_template(
        "evolution.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        families=[{"name": n, "description": family_description(n)} for n in list_families()],
        running=(_EVOLUTION_RUNNER is not None and _EVOLUTION_RUNNER.is_running),
        optimizer_modes=OPTIMIZER_MODES,
        prop_presets_json=_prop_presets_json(), **_alpaca_template_context())


@app.route("/evolution/start", methods=["POST"])
def evolution_start():
    global _EVOLUTION_RUNNER
    form = request.form
    with _EVOLUTION_LOCK:
        if _EVOLUTION_RUNNER is not None and _EVOLUTION_RUNNER.is_running:
            return redirect(url_for("evolution_form"))
    if not HEAVY_JOB_GUARD.try_acquire(JOB_EVOLUTION_LAB):
        return render_template(
            "evolution.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one of Search Lab / Evolution Lab / Full Pipeline / Speed Run at the same time can "
                f"exhaust available memory (each spawns its own worker processes, each holding a full "
                f"copy of the loaded data). Wait for it to finish, or stop it, before starting Evolution Lab."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            running=False,
            prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
            return render_template("evolution.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), families=[{"name": n, "description": family_description(n)} for n in list_families()], running=False, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400

        # UPGRADE (Evolution Lab account/risk fields): this used to be
        # `RiskConfig(initial_balance=...)` / `PropRules(account_size=...)`
        # ONLY -- every other prop/risk field (profit target, daily loss
        # limit, max drawdown, risk mode/value, pip size, spread,
        # slippage, commission, max trades/day) silently fell back to
        # RiskConfig/PropRules' own dataclass defaults (100k account, 8%
        # target, 5%/10% drawdown, 1% risk, 0.0001 FX pip size) no matter
        # what the person actually runs their real prop evaluation under.
        # Full Pipeline's own /full-pipeline/start route (below) already
        # reads all of these from its form -- so a person who filled in
        # their real numbers there but not here got Evolution Lab
        # "winners" chosen and scored against a completely different,
        # invisible rule set, then had Full Pipeline correctly reject most
        # of them under the REAL rules. Reading every field here, with the
        # same field names Full Pipeline / the Run page use, is what
        # fixes that -- see evolution.html's "Prop account & risk" card.
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent") or "percent",
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            max_trades_per_day=int(form.get("max_trades_day", 10) or 10),
            commission_per_trade=float(form.get("commission", 0) or 0),
            slippage_pips=float(form.get("slippage_pips", 0.5) or 0.5),
            spread_pips=float(form.get("spread_pips", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        )

        # T58 BACKTEST INTEGRITY CHECK -- same pre-flight gate as Run &
        # Report / Quick Optimize / Full Pipeline / Search Lab. Evolution
        # Lab evolves within a family (no single fixed strategy to check),
        # so `strategy=None` here -- only DATA/TIMEFRAME/ACCOUNT sections
        # run, which is still enough to catch corrupt data or an
        # unsupportable timeframe before a whole generational run starts.
        integrity_report = run_integrity_check(
            df, None, risk, prop_rules=rules,
            requested_timeframe=form.get("timeframe") or None,
            data_label=active_label,
        )
        if integrity_report.status == "BLOCKED":
            HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
            return render_template(
                "evolution.html", error=integrity_report.render(),
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
                running=False, prop_presets_json=_prop_presets_json(), **_alpaca_template_context(),
            ), 400

        families_selected = form.getlist("families") or None
        cfg = EvolutionConfig(
            population_size=int(form.get("population_size", 60) or 60),
            elite_keep=int(form.get("elite_keep", 10) or 10),
            instrument=active_label,
            families=families_selected,
            mc_sims=int(form.get("mc_sims", 1000) or 1000),
            max_generations=(int(form["max_generations"]) if form.get("max_generations") else None),
            save_to_library=form.get("save_to_library", "on") == "on",
            resume_from_checkpoint=form.get("resume_from_checkpoint", "on") == "on",
            fitness_goal=_parse_fitness_goal_form(form),
            # Loop mode -- see EvolutionConfig.target_eval_pass_pct's own
            # comment. Leaving the "Loop mode" field blank on the form
            # keeps the exact old "run forever / to max_generations
            # regardless of the leaderboard" behavior.
            target_eval_pass_pct=(
                float(form["target_eval_pass_pct"]) if form.get("target_eval_pass_pct") else None
            ),
            target_metric=form.get("target_metric", "cpcv_oos_eval_pass_probability") or "cpcv_oos_eval_pass_probability",
            reset_on_breach=form.get("reset_on_breach", "on") == "on",
            optimizer_mode=form.get("optimizer_mode", "genetic") or "genetic",
        )
        _EVOLUTION_LOG.clear()
        _EVOLUTION_LOG.append(f"Loaded {len(df)} bars from {active_label}.")

        # FIX (multi-timeframe sweep): "Timeframes to test" -- same
        # mechanism as Search Lab's own copy of this (see that route's
        # comment). `active_label` is resampled into every requested
        # timeframe and each becomes its own independent evolution runner
        # in a Multi-Instrument Evolution group, reusing this exact `cfg`
        # as every runner's base config (instrument gets overridden per
        # job -- see MultiInstrumentEvolutionGroup.__init__). Results open
        # on the Multi-Instrument Evolution job page instead of this
        # page's own single-runner status.
        expand_labels = parse_sweep_timeframes(form.get("expand_timeframes", ""))
        if expand_labels:
            HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
            if not HEAVY_JOB_GUARD.try_acquire(JOB_MULTI_INSTRUMENT_EVOLUTION):
                return render_template(
                    "evolution.html",
                    error=f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Wait for it "
                          f"to finish before starting a timeframe sweep.",
                    stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                    families=[{"name": n, "description": family_description(n)} for n in list_families()],
                    running=False, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 409
            expansion = expand_dataset_across_timeframes(df, active_label, expand_labels)
            sweep_jobs = evolution_jobs_from_expansion(expansion)
            if not sweep_jobs:
                HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_EVOLUTION)
                reasons = "; ".join(f"{s.requested_label} ({s.reason})" for s in expansion.skipped)
                return render_template(
                    "evolution.html",
                    error=f"Could not resample '{active_label}' into any of the requested timeframes -- {reasons}",
                    stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                    families=[{"name": n, "description": family_description(n)} for n in list_families()],
                    running=False, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400
            group_id = uuid.uuid4().hex[:12]
            group = MultiInstrumentEvolutionGroup(group_id, sweep_jobs, risk, rules, cfg)
            if expansion.skipped:
                group.errors["(timeframe sweep)"] = "; ".join(describe_skipped(expansion, active_label))
            group.start_all()
            with _MULTI_EVOLUTION_LOCK:
                _MULTI_EVOLUTION_GROUPS[group_id] = group
            return redirect(url_for("evolution_multi_instrument_job", group_id=group_id))

        with _EVOLUTION_LOCK:
            _EVOLUTION_RUNNER = EvolutionRunner(df, risk, rules, cfg, progress_cb=_evolution_log)
            _EVOLUTION_RUNNER.start()
        return redirect(url_for("evolution_form"))
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
        log_crash("Evolution Lab (web)", exc=exc)
        return render_template("evolution.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), families=[{"name": n, "description": family_description(n)} for n in list_families()], running=False, prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 500


@app.route("/evolution/stop", methods=["POST"])
def evolution_stop():
    with _EVOLUTION_LOCK:
        runner = _EVOLUTION_RUNNER
    if runner is not None:
        # Blocks up to 5s for the run loop to actually exit (now realistic
        # -- see EvolutionRunner._drain_futures -- instead of the old
        # as_completed()-with-no-timeout loop that could hang indefinitely
        # on one slow/wedged candidate and make this button look dead).
        # Releasing the guard here too, not just via the status.json poll,
        # means the very next page load already reflects STOPPED.
        if runner.stop_and_wait(timeout=5.0):
            HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
    return redirect(url_for("evolution_form"))


def _parse_fitness_goal_form(form) -> "dict | str":
    """Shared by single- and multi-instrument Evolution Lab start routes:
    "fitness_goal_preset" is one of app.evolution.prop_fitness.FITNESS_GOAL_PRESETS'
    keys, or "custom" to build a weight dict from the goal_w_* fields
    instead (see evolution.html / evolution_multi_instrument.html's
    "Optimization goal" section for what each maps to)."""
    goal_preset = (form.get("fitness_goal_preset") or "balanced").strip()
    if goal_preset != "custom":
        return goal_preset
    return {
        "pass_probability": float(form.get("goal_w_pass_probability", 1.0) or 1.0),
        "payout_probability": float(form.get("goal_w_payout_probability", 1.0) or 1.0),
        "robustness": float(form.get("goal_w_robustness", 1.0) or 1.0),
        "oos_consistency": float(form.get("goal_w_oos_consistency", 1.0) or 1.0),
        "drawdown": float(form.get("goal_w_drawdown", 1.0) or 1.0),
        "net_profit": float(form.get("goal_w_net_profit", 0.0) or 0.0),
    }


@app.route("/evolution/reset", methods=["POST"])
def evolution_reset():
    with _EVOLUTION_LOCK:
        if _EVOLUTION_RUNNER is not None and not _EVOLUTION_RUNNER.is_running:
            _EVOLUTION_RUNNER.reset()
            _EVOLUTION_LOG.clear()
    return redirect(url_for("evolution_form"))


@app.route("/evolution/reset-family-health", methods=["POST"])
def evolution_reset_family_health():
    """Archives (renames, does not delete) every search_*.db and
    tested_candidates.jsonl this app has ever written, so
    app.search.family_health's dead-end blacklist starts clean. Useful
    after a bug fix (a burst of build_or_backtest_error rows from a crash
    could have already tipped a family over its min_samples threshold
    before this was excluded from the count -- see family_health.py) or
    just to give a fresh instrument/dataset an unbiased first run."""
    from app.search.family_health import reset_family_health
    result = reset_family_health()
    _EVOLUTION_LOG.append(
        f"Family-health history reset: archived {result['search_dbs_reset']} Search Lab result "
        f"database(s) and {result['evolution_logs_reset']} Evolution Lab tested-candidate log(s). "
        f"Every family starts fresh again."
    )
    return redirect(url_for("evolution_form"))


@app.route("/evolution/promote", methods=["POST"])
def evolution_promote():
    """Web equivalent of the desktop app's PROMOTE TO STRATEGY LIBRARY
    button: saves one leaderboard candidate's manual-builder config into
    the Strategy Library (as a "manual" type -- see
    app.strategy.library.STRATEGY_TYPES, which didn't recognize "manual"
    at all until this fix, so this exact action always failed with
    "Unknown strategy type 'manual'" on both desktop and web). Looks in
    the live runner first, then falls back to the on-disk checkpoint so
    this also works after a server restart, matching the desktop app's
    _load_evolution_leaderboard_from_disk fallback.
    """
    candidate_id = (request.form.get("candidate_id") or "").strip()
    if not candidate_id:
        return jsonify({"ok": False, "error": "No candidate_id given."}), 400

    record = None
    with _EVOLUTION_LOCK:
        runner = _EVOLUTION_RUNNER
    if runner is not None:
        for r in runner.leaderboard:
            d = r.to_checkpoint_dict()
            if d.get("candidate_id") == candidate_id:
                record = d
                break
    if record is None:
        try:
            checkpoint = evo_checkpoint.load_checkpoint()
            if checkpoint is not None:
                for d in checkpoint.leaderboard:
                    if d.get("candidate_id") == candidate_id:
                        record = d
                        break
        except Exception:
            pass
    if record is None:
        return jsonify({"ok": False, "error": f"Candidate '{candidate_id}' not found on the leaderboard."}), 404

    config = (record.get("spec") or {}).get("config")
    if not config:
        return jsonify({"ok": False, "error": "This candidate has no manual-builder config to promote."}), 400

    # Same overfitting-gap check as the desktop app's PROMOTE confirmation
    # dialog (see main_window.py's _promote_evolution_leader_record) --
    # real report this addresses: a strategy promoted at a raw "40% pass /
    # 30% payout" coming back 2%/1% out of Full Pipeline, because
    # everything upstream of engine.py's _cpcv_and_pbo scores each
    # candidate against the SAME data the GA searched against. The web UI
    # already shows a client-side confirm() for this using the leaderboard
    # data it already has in hand; this is the server-side backstop for
    # any other caller (curl, another client) that skips the UI.
    raw_pct = (record.get("mc_summary") or {}).get("evaluation_pass_probability")
    oos_pct = record.get("cpcv_oos_eval_pass_probability")
    force = (request.form.get("force") or "").strip() == "1"
    if (
        isinstance(raw_pct, (int, float)) and isinstance(oos_pct, (int, float))
        and (raw_pct - oos_pct) > 15 and not force
    ):
        return jsonify({
            "ok": False,
            "needs_confirmation": True,
            "raw_pass_probability": raw_pct,
            "cpcv_oos_eval_pass_probability": oos_pct,
            "error": (
                f"Raw in-sample eval pass probability is {raw_pct:.1f}% but the honest, held-out CPCV "
                f"estimate is only {oos_pct:.1f}% -- likely overfit. Resend with force=1 to promote anyway."
            ),
        }), 409

    family = (record.get("meta") or {}).get("family", "strategy")
    filename = f"evolab_promoted_{family}_{candidate_id[-8:]}.json"
    text = json.dumps(config, indent=2)
    fitness = record.get("fitness") or {}
    try:
        try:
            save_strategy_text(text, filename, "manual", overwrite=False)
        except StrategyAlreadyExists:
            save_strategy_text(text, filename, "manual", overwrite=True)
        set_strategy_status("manual", filename, "validated")  # see main_window.py's matching note
        # Same lab-stats sidecar the auto-save-every-generation path writes
        # (app.evolution.engine._maybe_save_to_library) -- a strategy
        # promoted by hand from the leaderboard deserves the exact same
        # Dashboard/Strategy Library visibility as one the lab auto-saved,
        # not a bare config file with no fitness/MC/robustness context.
        save_strategy_metadata(
            "manual", filename,
            {
                "tags": ["evolution-lab", "promoted"],
                "description": (
                    f"Promoted from Evolution Lab leaderboard, family '{family}' -- "
                    f"PROP FITNESS {fitness.get('final_score'):.2f}" if fitness.get("final_score") is not None else
                    f"Promoted from Evolution Lab leaderboard, family '{family}'"
                ),
                "evolution": evolution_stats_metadata(record),
            },
            merge=True,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "filename": filename, "next_step": pipeline_guide.after_promote_to_library(filename)})


@app.route("/evolution/status.json")
def evolution_status():
    with _EVOLUTION_LOCK:
        runner = _EVOLUTION_RUNNER
    if runner is None:
        return jsonify({"running": False, "started": False, "log": [], "leaderboard": [], "journal": []})
    status = runner.status()
    if not status["running"]:
        # Covers both a deliberate stop and the runner's own thread exiting
        # on its own (finished, or crashed) -- either way the heavy-job
        # slot must free up so another tab/route can start. Harmless if
        # some other job already holds/released it (release() is a no-op
        # unless this name is the current holder).
        HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
    leaderboard = [r.to_checkpoint_dict() for r in runner.leaderboard]
    return jsonify({
        "started": True,
        "running": status["running"],
        "generation": status["generation"],
        "leaderboard_size": status["leaderboard_size"],
        "resumed": status["resumed"],
        "family_health": status.get("family_health"),
        "log": list(_EVOLUTION_LOG),
        "leaderboard": leaderboard,
        "journal": runner.journal[-30:],
        # Loop mode -- see EvolutionConfig.target_eval_pass_pct.
        "target_eval_pass_pct": status.get("target_eval_pass_pct"),
        "target_reached": status.get("target_reached", False),
        "target_reached_candidate_id": status.get("target_reached_candidate_id"),
        # UPGRADE (distribution-based results reporting): median vs. best
        # across this run's own leaderboard -- see
        # app.optimize.distribution_summary's own module docstring.
        "distribution_summary": compute_distribution_summary(
            [
                {"fitness": (r.fitness.final_score if r.fitness else None), "mc_summary": r.mc_summary, "statistics": r.stats}
                for r in runner.leaderboard
            ],
            total_tested=status["generation"] * runner.cfg.population_size,
        ),
        "next_step": None if status["running"] else pipeline_guide.after_evolution_stop(
            status["leaderboard_size"], total_evaluated=status["generation"] * runner.cfg.population_size,
        ),
    })


# ---------------------------------------------------------------------------
# Multi-Instrument Evolution Lab -- runs several independent Evolution Lab
# runners CONCURRENTLY, one per instrument/timeframe (see
# app.evolution.multi_instrument.MultiInstrumentEvolutionGroup for the
# actual manager this wires up). Same open-ended, resumable, run-until-
# stopped shape as single-instrument Evolution Lab above, just as a GROUP
# instead of one global runner -- kept as a fully separate route tree
# (own dict of groups, own guard slot) rather than generalizing the
# single-instrument routes above, so this addition can't regress the
# already-hardened single-instrument stop/hang behavior.
# ---------------------------------------------------------------------------

_MULTI_EVOLUTION_GROUPS: dict[str, MultiInstrumentEvolutionGroup] = {}
_MULTI_EVOLUTION_LOCK = threading.Lock()

HEAVY_JOB_GUARD.register_health_check(
    JOB_MULTI_INSTRUMENT_EVOLUTION,
    lambda: any(g.is_running for g in _MULTI_EVOLUTION_GROUPS.values()),
)


@app.route("/evolution/multi-instrument")
def evolution_multi_instrument_form():
    with _MULTI_EVOLUTION_LOCK:
        groups = list(_MULTI_EVOLUTION_GROUPS.items())
    return render_template(
        "evolution_multi_instrument.html",
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        families=[{"name": n, "description": family_description(n)} for n in list_families()],
        active_groups=[{"group_id": gid, "running": g.is_running, "labels": [j.label for j in g.jobs]} for gid, g in groups],
        optimizer_modes=OPTIMIZER_MODES,
    )


@app.route("/evolution/multi-instrument/start", methods=["POST"])
def evolution_multi_instrument_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_MULTI_INSTRUMENT_EVOLUTION):
        return render_template(
            "evolution_multi_instrument.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job at the same time can exhaust available memory. Wait for it to finish, "
                f"or stop it, before starting Multi-Instrument Evolution Lab."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            active_groups=[],
        ), 409
    try:
        selected = form.getlist("datasets")
        expand_labels = parse_sweep_timeframes(form.get("expand_timeframes", ""))
        if not selected:
            HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_EVOLUTION)
            return render_template(
                "evolution_multi_instrument.html",
                error="Select at least 1 dataset.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
                active_groups=[],
            ), 400
        if not expand_labels and len(selected) < 2:
            HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_EVOLUTION)
            return render_template(
                "evolution_multi_instrument.html",
                error="Select at least 2 datasets to run across, or fill in \"Timeframes to auto-generate\" "
                      "to expand a single dataset into several timeframes instead.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
                active_groups=[],
            ), 400

        # FIX (multi-timeframe sweep): see the identical comment in
        # search_multi_instrument_start above -- same mechanism, this
        # engine's own EvolutionInstrumentJob shape instead of InstrumentJob.
        jobs: list[EvolutionInstrumentJob] = []
        sweep_warnings: list[str] = []
        for name in selected:
            candidate_path = get_raw_data_dir() / name
            if not candidate_path.exists():
                continue
            stem = Path(name).stem
            instrument = name.split("/")[0] if "/" in name else stem
            if expand_labels:
                import_result = import_csv(candidate_path)
                if not import_result.is_valid:
                    sweep_warnings.append(f"{name}: could not read this dataset to expand it -- skipped.")
                    continue
                expansion = expand_dataset_across_timeframes(import_result.dataframe, instrument, expand_labels)
                sweep_warnings.extend(describe_skipped(expansion, name))
                jobs.extend(evolution_jobs_from_expansion(expansion))
            else:
                jobs.append(EvolutionInstrumentJob(instrument=instrument, timeframe=stem, csv_path=str(candidate_path)))

        if len(jobs) < 2:
            HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_EVOLUTION)
            return render_template(
                "evolution_multi_instrument.html",
                error="Could not resolve at least 2 of the selected datasets to real files on disk.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
                active_groups=[],
            ), 400

        # UPGRADE (Evolution Lab account/risk fields) -- same fix as the
        # single-instrument /evolution/start route above: read every
        # prop/risk field the form now sends instead of defaulting almost
        # all of them silently. See that route's comment for why this
        # matters (Full Pipeline checking against different, invisible
        # rules than Evolution Lab actually searched under).
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent") or "percent",
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            max_trades_per_day=int(form.get("max_trades_day", 10) or 10),
            commission_per_trade=float(form.get("commission", 0) or 0),
            slippage_pips=float(form.get("slippage_pips", 0.5) or 0.5),
            spread_pips=float(form.get("spread_pips", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        )
        families_selected = form.getlist("families") or None
        base_cfg = EvolutionConfig(
            population_size=int(form.get("population_size", 60) or 60),
            elite_keep=int(form.get("elite_keep", 10) or 10),
            families=families_selected,
            mc_sims=int(form.get("mc_sims", 1000) or 1000),
            max_generations=(int(form["max_generations"]) if form.get("max_generations") else None),
            save_to_library=form.get("save_to_library", "on") == "on",
            resume_from_checkpoint=form.get("resume_from_checkpoint", "on") == "on",
            fitness_goal=_parse_fitness_goal_form(form),
            # Loop mode -- see EvolutionConfig.target_eval_pass_pct. Every
            # runner in the group shares this same target/metric (they only
            # differ in checkpoint/tested-log/knowledge-graph paths), so
            # each instrument stops itself independently the moment ITS OWN
            # leaderboard clears it.
            target_eval_pass_pct=(
                float(form["target_eval_pass_pct"]) if form.get("target_eval_pass_pct") else None
            ),
            target_metric=form.get("target_metric", "cpcv_oos_eval_pass_probability") or "cpcv_oos_eval_pass_probability",
            reset_on_breach=form.get("reset_on_breach", "on") == "on",
            optimizer_mode=form.get("optimizer_mode", "genetic") or "genetic",
        )

        # UPGRADE (budget field): app.search.budget_allocator was wired
        # into MultiInstrumentEvolutionGroup as a backward-compatible
        # opt-in param last session, but no form field ever set it --
        # every multi-instrument run silently used the OLD N-times-the-
        # compute behavior. Blank keeps that exact old behavior; a value
        # here spreads the same total compute across instruments instead
        # (see MultiInstrumentEvolutionGroup's own docstring).
        budget_raw = (form.get("total_evaluation_budget") or "").strip()
        total_evaluation_budget = int(budget_raw) if budget_raw else None

        group_id = uuid.uuid4().hex[:12]
        group = MultiInstrumentEvolutionGroup(
            group_id, jobs, risk, rules, base_cfg,
            total_evaluation_budget=total_evaluation_budget,
        )
        if sweep_warnings:
            group.errors["(timeframe sweep)"] = "; ".join(sweep_warnings)
        group.start_all()
        with _MULTI_EVOLUTION_LOCK:
            _MULTI_EVOLUTION_GROUPS[group_id] = group
        return redirect(url_for("evolution_multi_instrument_job", group_id=group_id))

    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_EVOLUTION)
        log_crash("Multi-Instrument Evolution Lab (web, start)", exc=exc)
        return render_template(
            "evolution_multi_instrument.html", error=f"Unexpected error: {exc}",
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            active_groups=[],
        ), 500


@app.route("/evolution/multi-instrument/job/<group_id>")
def evolution_multi_instrument_job(group_id):
    with _MULTI_EVOLUTION_LOCK:
        group = _MULTI_EVOLUTION_GROUPS.get(group_id)
    if group is None:
        return render_template("evolution_multi_instrument_job.html", group_id=group_id, not_found=True), 404
    return render_template("evolution_multi_instrument_job.html", group_id=group_id, not_found=False)


@app.route("/evolution/multi-instrument/job/<group_id>/status.json")
def evolution_multi_instrument_job_status(group_id):
    with _MULTI_EVOLUTION_LOCK:
        group = _MULTI_EVOLUTION_GROUPS.get(group_id)
    if group is None:
        return jsonify({"found": False}), 404
    status = group.status()
    if not status["running"]:
        # Same self-heal-on-poll reasoning as single-instrument Evolution
        # Lab's own /evolution/status.json -- see that route's comment.
        HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_EVOLUTION)
    return jsonify({"found": True, **status})


@app.route("/evolution/multi-instrument/job/<group_id>/stop", methods=["POST"])
def evolution_multi_instrument_stop(group_id):
    with _MULTI_EVOLUTION_LOCK:
        group = _MULTI_EVOLUTION_GROUPS.get(group_id)
    if group is not None:
        # Bounded wait so the very next page load already reflects
        # STOPPED, same reasoning as single-instrument Evolution Lab's
        # own stop_and_wait()-backed /evolution/stop route.
        if group.stop_all(timeout=10.0):
            HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_EVOLUTION)
    return redirect(url_for("evolution_multi_instrument_job", group_id=group_id))


@app.route("/evolution/multi-instrument/job/<group_id>/promote", methods=["POST"])
def evolution_multi_instrument_promote(group_id):
    with _MULTI_EVOLUTION_LOCK:
        group = _MULTI_EVOLUTION_GROUPS.get(group_id)
    if group is None:
        return jsonify({"ok": False, "error": "Group not found."}), 404

    label = (request.form.get("label") or "").strip()
    candidate_id = (request.form.get("candidate_id") or "").strip()
    if not label or not candidate_id:
        return jsonify({"ok": False, "error": "label and candidate_id are required."}), 400

    record = group.promote(label, candidate_id)
    if record is None:
        return jsonify({"ok": False, "error": f"Candidate '{candidate_id}' not found on {label}'s leaderboard."}), 404

    config = (record.get("spec") or {}).get("config")
    if not config:
        return jsonify({"ok": False, "error": "This candidate has no manual-builder config to promote."}), 400

    family = (record.get("meta") or {}).get("family", "strategy")
    filename = f"evolab_multi_{label.replace('/', '_')}_{family}_{candidate_id[-8:]}.json"
    text = json.dumps(config, indent=2)
    fitness = record.get("fitness") or {}
    try:
        try:
            save_strategy_text(text, filename, "manual", overwrite=False)
        except StrategyAlreadyExists:
            save_strategy_text(text, filename, "manual", overwrite=True)
        set_strategy_status("manual", filename, "validated")
        save_strategy_metadata(
            "manual", filename,
            {
                "tags": ["evolution-lab", "promoted"],
                "description": (
                    f"Promoted from Multi-Instrument Evolution Lab ({label}), family '{family}' -- "
                    f"PROP FITNESS {fitness.get('final_score'):.2f}" if fitness.get("final_score") is not None else
                    f"Promoted from Multi-Instrument Evolution Lab ({label}), family '{family}'"
                ),
                "evolution": evolution_stats_metadata(record),
            },
            merge=True,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "filename": filename, "next_step": pipeline_guide.after_promote_to_library(filename)})


# ---------------------------------------------------------------------------
# 18. Research Agent -- a ReAct-style tool-calling agent whose tools are
# 100% read-only calls into the real backtest/prop-sim/Monte Carlo/walk-
# forward/regime/sensitivity/cost-stress engine (see app.ai.research_agent
# -- no code-editing tool exists, so it can only recommend, never apply).
# Needs a local Ollama reachable from wherever this server runs -- see the
# AI Assist note on the Full Pipeline page for what that means (point it
# at whatever machine on your LAN is running Ollama).
# ---------------------------------------------------------------------------

_AGENT_JOBS: dict[str, dict] = {}
_AGENT_JOBS_LOCK = threading.Lock()


def _agent_job_log(job_id: str, msg: str) -> None:
    with _AGENT_JOBS_LOCK:
        job = _AGENT_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_agent_job(job_id: str, question: str, ctx: ResearchAgentContext, settings: OllamaSettings) -> None:
    try:
        agent = ResearchAgent(settings)
        result = agent.run(question, ctx, progress_cb=lambda msg: _agent_job_log(job_id, msg))
        with _AGENT_JOBS_LOCK:
            job = _AGENT_JOBS[job_id]
            job["done"] = True
            job["result"] = result
    except Exception as exc:  # noqa: BLE001
        with _AGENT_JOBS_LOCK:
            job = _AGENT_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"


@app.route("/research-agent")
def research_agent_form():
    saved_ai = load_ollama_settings()
    return render_template(
        "research_agent.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, ai_enabled=saved_ai.enabled, ai_host=saved_ai.host, ai_model=saved_ai.model, **_alpaca_template_context())


@app.route("/research-agent/start", methods=["POST"])
def research_agent_start():
    form = request.form
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("research_agent.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), ai_enabled=False, ai_host="", ai_model="", **_alpaca_template_context()), 400

        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(initial_balance=float(form.get("initial_balance", 100000) or 100000), pip_size=float(form.get("pip_size", 0.0001) or 0.0001))
        rules = PropRules(account_size=float(form.get("account_size", 100000) or 100000))
        question = (form.get("question") or "").strip()
        if not question:
            return render_template("research_agent.html", error="Enter a question for the agent to investigate.", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), ai_enabled=False, ai_host="", ai_model="", **_alpaca_template_context()), 400

        settings = OllamaSettings(enabled=True, host=form.get("ai_host", "http://localhost:11434") or "http://localhost:11434", model=form.get("ai_model", "llama3.1") or "llama3.1")
        try:
            save_ollama_settings(settings)
        except Exception:
            pass

        # Optional: HTML/CSV/PDF backtest reports or screenshots uploaded
        # alongside the question -- see app.ai.report_import and the
        # read_uploaded_report tool in app.ai.research_agent. Saved to a
        # per-job temp dir (never the working data dir) since these are
        # evidence for the agent to read, not strategies/datasets this app
        # manages long-term; nothing here is cleaned up automatically, same
        # as compute_pbo's own tmp_dir convention elsewhere in this file.
        uploaded_reports: list[Path] = []
        report_uploads = request.files.getlist("report_files")
        if report_uploads and any(f.filename for f in report_uploads):
            reports_dir = Path(tempfile.mkdtemp(prefix="t58_agent_reports_"))
            for f in report_uploads:
                if not f.filename:
                    continue
                suffix = Path(f.filename).suffix.lower()
                if suffix not in (".html", ".htm", ".csv", ".pdf", ".png", ".jpg", ".jpeg", ".webp"):
                    continue
                dest = reports_dir / Path(f.filename).name
                f.save(dest)
                uploaded_reports.append(dest)

        ctx = ResearchAgentContext(
            df=df, strategy_builder=(lambda s=strategy: s), strategy_name=getattr(strategy, "name", "Strategy"),
            source_type=strategy.source_type, risk=risk, prop_rules=rules, instrument=active_label,
            uploaded_reports=uploaded_reports,
        )

        job_id = uuid.uuid4().hex[:12]
        job_log = [f"Loaded {len(df)} bars from {active_label}.", f"Question: {question}"]
        if uploaded_reports:
            job_log.append(f"Uploaded {len(uploaded_reports)} report/screenshot file(s): " + ", ".join(p.name for p in uploaded_reports))
        with _AGENT_JOBS_LOCK:
            _AGENT_JOBS[job_id] = {"log": job_log, "done": False, "error": None, "result": None, "started_at": time.time()}
        thread = threading.Thread(target=_run_agent_job, args=(job_id, question, ctx, settings), daemon=True)
        thread.start()
        return redirect(url_for("research_agent_job", job_id=job_id))
    except StrategyError as exc:
        return render_template("research_agent.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), ai_enabled=False, ai_host="", ai_model="", **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("research_agent.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), saved_strategies_json=_saved_strategies_json(), ai_enabled=False, ai_host="", ai_model="", **_alpaca_template_context()), 500


@app.route("/research-agent/job/<job_id>")
def research_agent_job(job_id):
    with _AGENT_JOBS_LOCK:
        job = _AGENT_JOBS.get(job_id)
    if job is None:
        return render_template("research_agent_job.html", job_id=job_id, not_found=True), 404
    return render_template("research_agent_job.html", job_id=job_id, not_found=False)


@app.route("/research-agent/job/<job_id>/status.json")
def research_agent_job_status(job_id):
    with _AGENT_JOBS_LOCK:
        job = _AGENT_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = {
            "final_answer": result.final_answer,
            "error": result.error,
            "stopped_reason": result.stopped_reason,
            "steps": [
                {"step_index": s.step_index, "thought": s.thought, "action": s.action, "action_input": s.action_input, "observation": s.observation, "note": s.note}
                for s in result.steps
            ],
        }
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "summary": summary})


# ---------------------------------------------------------------------------
# Research Loop -- the closed hypothesis -> generate -> backtest -> Monte
# Carlo -> prop-survival -> real computed failure diagnosis -> Ollama-
# refined hypothesis -> repeat loop (see app.ai.research_loop). Unlike the
# Research Agent above (answers ONE question, then stops), this is meant
# to run as a background/overnight companion -- start it, leave it running
# indefinitely (n_iterations left blank) alongside a multi-instrument
# search, and check back on its leaderboard of KEEP-verdict strategies
# later. Deliberately NOT gated by HEAVY_JOB_GUARD: it runs one strategy
# at a time in-process (no worker-process pool like Search Lab/Evolution
# Lab/Full Pipeline/Speed Run spin up), so it's meant to run CONCURRENTLY
# alongside those, not compete with them for the same exclusive slot.
# ---------------------------------------------------------------------------

_RESEARCH_LOOP_JOBS: dict[str, dict] = {}
_RESEARCH_LOOP_JOBS_LOCK = threading.Lock()


def _research_loop_log(job_id: str, msg: str) -> None:
    with _RESEARCH_LOOP_JOBS_LOCK:
        job = _RESEARCH_LOOP_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)
            del job["log"][:-500]


@app.route("/research-loop")
def research_loop_form():
    saved_ai = load_ollama_settings()
    with _RESEARCH_LOOP_JOBS_LOCK:
        active_jobs = [
            {"job_id": jid, "running": j["runner"].is_running, "n_iterations_run": len(j["runner"].iterations)}
            for jid, j in _RESEARCH_LOOP_JOBS.items()
        ]
    return render_template(
        "research_loop.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        ai_enabled=saved_ai.enabled, ai_host=saved_ai.host, ai_model=saved_ai.model,
        active_jobs=active_jobs, **_alpaca_template_context())


@app.route("/research-loop/start", methods=["POST"])
def research_loop_start():
    form = request.form
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template(
                "research_loop.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                ai_enabled=False, ai_host="", ai_model="", active_jobs=[], **_alpaca_template_context()), 400

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(account_size=float(form.get("account_size", 100000) or 100000))
        settings = OllamaSettings(
            enabled=True,
            host=form.get("ai_host", "http://localhost:11434") or "http://localhost:11434",
            model=form.get("ai_model", "llama3.1") or "llama3.1",
        )
        try:
            save_ollama_settings(settings)
        except Exception:
            pass

        n_iterations_raw = (form.get("n_iterations") or "").strip()
        cfg = ResearchLoopConfig(
            n_iterations=(int(n_iterations_raw) if n_iterations_raw else None),
            initial_idea=(form.get("initial_idea") or "").strip(),
            mc_sims=int(form.get("mc_sims", 500) or 500),
            survival_sims=int(form.get("survival_sims", 1000) or 1000),
            keep_score_threshold=float(form.get("keep_score_threshold", 40.0) or 40.0),
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        runner = ResearchLoopRunner(df, risk, rules, settings, cfg, progress_cb=lambda msg: _research_loop_log(job_id, msg))
        with _RESEARCH_LOOP_JOBS_LOCK:
            _RESEARCH_LOOP_JOBS[job_id] = {"log": initial_log, "runner": runner, "instrument": active_label}
        runner.start()
        return redirect(url_for("research_loop_job", job_id=job_id))
    except Exception as exc:  # noqa: BLE001
        log_crash("Research Loop (web, start)", exc=exc)
        return render_template(
            "research_loop.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            ai_enabled=False, ai_host="", ai_model="", active_jobs=[], **_alpaca_template_context()), 500


@app.route("/research-loop/job/<job_id>")
def research_loop_job(job_id):
    with _RESEARCH_LOOP_JOBS_LOCK:
        job = _RESEARCH_LOOP_JOBS.get(job_id)
    if job is None:
        return render_template("research_loop_job.html", job_id=job_id, not_found=True), 404
    return render_template("research_loop_job.html", job_id=job_id, not_found=False)


@app.route("/research-loop/job/<job_id>/status.json")
def research_loop_job_status(job_id):
    with _RESEARCH_LOOP_JOBS_LOCK:
        job = _RESEARCH_LOOP_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    runner: ResearchLoopRunner = job["runner"]

    def _brief(it) -> dict:
        d = it.to_dict()
        d.pop("code", None)  # keep the poll payload bounded over a long overnight run
        return d

    recent = runner.iterations[-50:]
    return jsonify({
        "found": True,
        "running": runner.is_running,
        "stopped_reason": runner.stopped_reason,
        "log": job["log"][-200:],
        "n_iterations_run": len(runner.iterations),
        "iterations": [_brief(it) for it in recent],
        "best_iteration": runner.best_iteration.to_dict() if runner.best_iteration else None,
    })


@app.route("/research-loop/job/<job_id>/stop", methods=["POST"])
def research_loop_stop(job_id):
    with _RESEARCH_LOOP_JOBS_LOCK:
        job = _RESEARCH_LOOP_JOBS.get(job_id)
    if job is not None:
        job["runner"].stop_and_wait(timeout=15.0)
    return redirect(url_for("research_loop_job", job_id=job_id))


_FORWARD_TEST_SESSION: dict = {"session": None, "log": [], "id": None}
_FORWARD_TEST_LOCK = threading.Lock()
_LIVE_DEPLOY_SESSION: dict = {"session": None, "log": [], "id": None}
_LIVE_DEPLOY_LOCK = threading.Lock()


@app.route("/forward-test")
def forward_test_info():
    from app.live_deploy.live_settings import load_accounts
    mt5_accounts = [a for a in load_accounts() if a.platform == "MT4/MT5"]
    with _FORWARD_TEST_LOCK:
        running = _FORWARD_TEST_SESSION["session"] is not None and _FORWARD_TEST_SESSION["session"].status.running
    return render_template("forward_test.html", mt5_accounts=mt5_accounts, running=running, **_alpaca_template_context())


@app.route("/forward-test/start", methods=["POST"])
def forward_test_start():
    from app.forward_test.engine import ForwardTestConfig, ForwardTestSession
    from app.forward_test.journal import ForwardTestJournal
    from app.forward_test.mt5_connector import MT5Connector
    from app.live_deploy.live_settings import load_accounts

    with _FORWARD_TEST_LOCK:
        if _FORWARD_TEST_SESSION["session"] is not None and _FORWARD_TEST_SESSION["session"].status.running:
            return jsonify({"ok": False, "error": "A forward test is already running. Stop it first."}), 409

        form = request.form
        account_id = form.get("account_id", "")
        accounts = {a.id: a for a in load_accounts() if a.platform == "MT4/MT5"}
        account = accounts.get(account_id)
        if account is None:
            return jsonify({"ok": False, "error": "Select a saved MT5 account first (add one under Settings -> API Keys)."}), 400

        try:
            strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        except (StrategyError, RefinementError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        connector = MT5Connector(
            login=account.login, password=account.password, server=account.server,
            terminal_path=account.terminal_path,
        )
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        cfg = ForwardTestConfig(
            symbol=form.get("symbol", "").strip() or "EURUSD",
            timeframe_minutes=int(form.get("timeframe_minutes", 5) or 5),
            risk=risk,
            poll_seconds=int(form.get("poll_seconds", 20) or 20),
        )
        log = []
        session = ForwardTestSession(
            strategy=strategy, strategy_type=form.get("strategy_mode", "manual"),
            strategy_filename=form.get("strategy_mode", "manual"),
            connector=connector, journal=ForwardTestJournal(), config=cfg,
            on_log=lambda level, msg: log.append(f"[{level}] {msg}"),
        )
        ok, message = session.start()
        if not ok:
            return jsonify({"ok": False, "error": message}), 400
        _FORWARD_TEST_SESSION["session"] = session
        _FORWARD_TEST_SESSION["log"] = log
        return jsonify({"ok": True, "message": message})


@app.route("/forward-test/stop", methods=["POST"])
def forward_test_stop():
    with _FORWARD_TEST_LOCK:
        session = _FORWARD_TEST_SESSION["session"]
        if session is None:
            return jsonify({"ok": False, "error": "Nothing is running."})
        session.stop()
        return jsonify({"ok": True})


@app.route("/forward-test/status.json")
def forward_test_status():
    with _FORWARD_TEST_LOCK:
        session = _FORWARD_TEST_SESSION["session"]
        if session is None:
            return jsonify({"running": False, "log": []})
        s = session.status
        return jsonify({
            "running": s.running, "connected": s.connected, "balance": s.balance, "equity": s.equity,
            "n_trades_closed": s.n_trades_closed, "win_rate": s.win_rate, "net_pnl": s.net_pnl,
            "halted_reason": s.halted_reason, "drift_flag": s.drift_flag,
            "log": _FORWARD_TEST_SESSION["log"][-100:],
        })


@app.route("/deploy-live")
def deploy_live_info():
    from app.live_deploy.live_settings import load_accounts
    with _LIVE_DEPLOY_LOCK:
        running = _LIVE_DEPLOY_SESSION["session"] is not None and _LIVE_DEPLOY_SESSION["session"].status.running
    return render_template("deploy_live.html", broker_accounts=load_accounts(), running=running, **_alpaca_template_context())


@app.route("/deploy-live/start", methods=["POST"])
def deploy_live_start():
    """UPGRADE (MT5/prop-firm web deploy): this connects to a REAL,
    FUNDED account and can place REAL trades -- see deploy_live.html's own
    prominent warning. The typed confirmation phrase below is a real
    guard, not decoration: a stray/accidental POST to this endpoint
    (a browser back-button resubmit, a bookmarked/cached form, a CSRF
    attempt from a page that doesn't know the exact phrase) cannot start
    live trading without it. This does NOT add authentication to the
    server itself -- see deploy_live.html for why that's a decision for
    the person running this server to make about their own network
    exposure, not something to silently bolt on here."""
    from app.live_deploy.execution_engine import LiveExecutionConfig, LiveExecutionSession
    from app.live_deploy.broker_registry import build_adapter
    from app.live_deploy.live_settings import load_accounts
    from app.forward_test.journal import ForwardTestJournal

    with _LIVE_DEPLOY_LOCK:
        if _LIVE_DEPLOY_SESSION["session"] is not None and _LIVE_DEPLOY_SESSION["session"].status.running:
            return jsonify({"ok": False, "error": "A live session is already running. Stop it first."}), 409

        form = request.form
        if (form.get("confirm_text") or "").strip() != "DEPLOY LIVE":
            return jsonify({"ok": False, "error": "Type DEPLOY LIVE exactly, in capitals, to confirm you understand this trades real money."}), 400

        account_id = form.get("account_id", "")
        accounts = {a.id: a for a in load_accounts()}
        account = accounts.get(account_id)
        if account is None:
            return jsonify({"ok": False, "error": "Select a saved broker account first (add one under Settings -> API Keys)."}), 400

        try:
            strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        except (StrategyError, RefinementError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        try:
            broker = build_adapter(account)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"Could not build a broker connection: {exc}"}), 400

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        )
        cfg = LiveExecutionConfig(
            symbol=form.get("symbol", "").strip() or "EURUSD",
            timeframe_minutes=int(form.get("timeframe_minutes", 5) or 5),
            risk=risk, prop_rules=rules,
            poll_seconds=int(form.get("poll_seconds", 20) or 20),
        )
        log = []
        session = LiveExecutionSession(
            strategy=strategy, strategy_type=form.get("strategy_mode", "manual"),
            strategy_filename=form.get("strategy_mode", "manual"),
            broker=broker, journal=ForwardTestJournal(), config=cfg,
            on_log=lambda level, msg: log.append(f"[{level}] {msg}"),
        )
        ok, message = session.start()
        if not ok:
            return jsonify({"ok": False, "error": message}), 400
        _LIVE_DEPLOY_SESSION["session"] = session
        _LIVE_DEPLOY_SESSION["log"] = log
        return jsonify({"ok": True, "message": message})


@app.route("/deploy-live/stop", methods=["POST"])
def deploy_live_stop():
    with _LIVE_DEPLOY_LOCK:
        session = _LIVE_DEPLOY_SESSION["session"]
        if session is None:
            return jsonify({"ok": False, "error": "Nothing is running."})
        session.stop()
        return jsonify({"ok": True})


@app.route("/deploy-live/status.json")
def deploy_live_status():
    with _LIVE_DEPLOY_LOCK:
        session = _LIVE_DEPLOY_SESSION["session"]
        if session is None:
            return jsonify({"running": False, "log": []})
        s = session.status
        return jsonify({
            "running": s.running, "connected": s.connected, "platform": s.platform,
            "balance": s.balance, "equity": s.equity, "n_trades_closed": s.n_trades_closed,
            "win_rate": s.win_rate, "net_pnl": s.net_pnl, "halted_reason": s.halted_reason,
            "drift_flag": s.drift_flag, "log": _LIVE_DEPLOY_SESSION["log"][-100:],
        })


@app.route("/api/suggest-loop-config")
def api_suggest_loop_config():
    """Prop-parameter auto-tuning -- reads the prop rule query params a
    form's own fields already hold (account_size/profit_target/daily_loss/
    max_dd) and returns app.orchestration.prop_autotune's suggested Loop
    Mode / risk settings as JSON, for a page's own JS to pre-fill its
    Loop Mode fields with. See that module's own docstring: this is a
    heuristic starting point, not an authoritative answer."""
    try:
        rules = PropRules(
            account_size=float(request.args.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(request.args.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(request.args.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(request.args.get("max_dd", 10) or 10),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": f"Invalid prop rule value: {exc}"}), 400
    suggestion = suggest_from_prop_rules(rules)
    return jsonify({"ok": True, "suggestion": suggestion.to_dict()})


@app.route("/search")
def search_form():
    return render_template(
        "search.html",
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        families=[{"name": n, "description": family_description(n)} for n in list_families()],
        saved_strategies_json=_saved_strategies_json(),
        strategy_notice=request.args.get("strategy_notice"),
        strategy_statuses=STRATEGY_STATUSES,
        alpaca_notice=request.args.get("alpaca_notice"),
        alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"),
        optimizer_modes=OPTIMIZER_MODES,
        prop_presets_json=_prop_presets_json(), **_alpaca_template_context())


@app.route("/search/start", methods=["POST"])
def search_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_SEARCH_LAB):
        return render_template(
            "search.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job (Search Lab / Evolution Lab / Full Pipeline / Speed Run) at the same "
                f"time can exhaust available memory. Wait for it to finish before starting Search Lab."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            saved_strategies_json=_saved_strategies_json(),
            prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
            return render_template(
                "search.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
                saved_strategies_json=_saved_strategies_json(),
                prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400

        mode_key = form.get("search_mode", "family_named")
        seed = int(form.get("seed", 42) or 42)
        max_candidates = int(form.get("max_candidates", 200) or 200)
        library_ref = None
        strategy = None  # only "single"/"family_grid" build one concrete Strategy below;
        # "family_named" searches many candidates at once and has no single
        # strategy to check -- run_integrity_check accepts None and simply
        # omits its STRATEGY section in that case.
        _family_exclusion_log: list = []

        if mode_key == "single":
            strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
            space = generate_search_space(mode="single", strategy=strategy)
        elif mode_key == "family_grid":
            strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
            space = generate_search_space(
                mode="family", strategy=strategy,
                grid_points_per_gene=int(form.get("grid_points", 3) or 3),
                max_candidates=max_candidates, seed=seed,
            )
        else:
            family_key = form.get("family", "all") or "all"
            exclude_families = _resolve_family_exclusions(_family_exclusion_log)
            space = generate_search_space(
                mode="family", family=family_key, max_candidates=max_candidates, seed=seed,
                exclude_families=exclude_families,
            )

        workers_raw = (form.get("workers") or "").strip()
        stage_cfg = SearchStageConfig(
            min_trades=int(form.get("min_trades", 20) or 20),
            min_profit_factor=float(form.get("min_profit_factor", 1.05) or 1.05),
            stage1_top_n=int(form.get("stage1_top_n", 40) or 40),
            ga_population=int(form.get("ga_population", 10) or 10),
            ga_generations=int(form.get("ga_generations", 4) or 4),
            stage2_top_n=int(form.get("stage2_top_n", 10) or 10),
            full_mc_sims=int(form.get("full_mc_sims", 3000) or 3000),
            walk_forward_folds=int(form.get("walk_forward_folds", 4) or 4),
            robustness_neighbors=int(form.get("robustness_neighbors", 6) or 6),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            workers=int(workers_raw) if workers_raw else None,
            random_seed=seed,
            reset_on_breach=form.get("reset_on_breach") == "on",
            optimizer_mode=form.get("optimizer_mode", "genetic") or "genetic",
        )
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        )

        # T58 BACKTEST INTEGRITY CHECK -- same pre-flight gate as Run &
        # Report / Quick Optimize / Full Pipeline. `strategy` is None for
        # "family_named" mode (no single strategy to check yet), in which
        # case only the DATA/TIMEFRAME/ACCOUNT sections run -- still
        # enough to catch corrupt data or an unsupportable timeframe
        # before spending compute across the whole family.
        integrity_report = run_integrity_check(
            df, strategy, risk, prop_rules=rules,
            requested_timeframe=form.get("timeframe") or None,
            data_label=active_label,
        )
        if integrity_report.status == "BLOCKED":
            HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
            return render_template(
                "search.html", error=integrity_report.render(),
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
                saved_strategies_json=_saved_strategies_json(),
                prop_presets_json=_prop_presets_json(), **_alpaca_template_context(),
            ), 400

        job_id = uuid.uuid4().hex[:12]
        db_path = str(SEARCH_DIR / f"search_{job_id}.db")
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        initial_log.extend(_family_exclusion_log)
        cancel_event = threading.Event()

        # FIX (multi-timeframe sweep): "Timeframes to test" lets this single-
        # dataset page do a real multi-timeframe run without sending anyone
        # over to the separate Multi-Instrument Search page -- `active_label`
        # is resampled into every requested timeframe (see
        # app.data.timeframe_sweep) and each one becomes its own job, reusing
        # the exact same `space`/`stage_cfg`/`risk`/`rules` already built
        # above from this form. Results render on the Multi-Instrument
        # Search job page (built for exactly this "several targets, one
        # search space" shape) rather than duplicating that page's report
        # here.
        expand_labels = parse_sweep_timeframes(form.get("expand_timeframes", ""))
        if expand_labels:
            HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
            if not HEAVY_JOB_GUARD.try_acquire(JOB_MULTI_INSTRUMENT_SEARCH):
                return render_template(
                    "search.html",
                    error=(
                        f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Wait for it to "
                        f"finish before starting a timeframe sweep."
                    ),
                    stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                    families=[{"name": n, "description": family_description(n)} for n in list_families()],
                    saved_strategies_json=_saved_strategies_json(),
                    prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 409
            expansion = expand_dataset_across_timeframes(df, active_label, expand_labels)
            sweep_jobs = search_jobs_from_expansion(expansion)
            if not sweep_jobs:
                HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SEARCH)
                reasons = "; ".join(f"{s.requested_label} ({s.reason})" for s in expansion.skipped)
                return render_template(
                    "search.html",
                    error=f"Could not resample '{active_label}' into any of the requested timeframes -- {reasons}",
                    stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                    families=[{"name": n, "description": family_description(n)} for n in list_families()],
                    saved_strategies_json=_saved_strategies_json(),
                    prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400
            sweep_job_id = uuid.uuid4().hex[:12]
            sweep_log = initial_log + describe_skipped(expansion, active_label) + [
                f"Timeframe sweep of {active_label}: searching {len(sweep_jobs)} timeframe target(s): "
                + ", ".join(f"{j.instrument}/{j.timeframe}" for j in sweep_jobs),
            ]
            with _MULTI_SEARCH_JOBS_LOCK:
                _MULTI_SEARCH_JOBS[sweep_job_id] = {
                    "log": sweep_log, "done": False, "error": None, "results": None,
                    "best_label": None, "champion_report": None,
                    "labels": [f"{j.instrument}/{j.timeframe}" for j in sweep_jobs], "loop_mode": False,
                }
            thread = threading.Thread(
                target=_run_multi_search_job,
                args=(sweep_job_id, sweep_jobs, space, risk, rules, stage_cfg, min(len(sweep_jobs), 3)),
                daemon=True,
            )
            thread.start()
            return redirect(url_for("search_multi_instrument_job", job_id=sweep_job_id))

        loop_mode_on = form.get("loop_mode") == "on"
        if loop_mode_on:
            time_budget_raw = (form.get("loop_time_budget_hours") or "").strip()
            loop_cfg = SearchLoopConfig(
                target_eval_pass_pct=float(form.get("loop_target_eval_pass_pct", 60) or 60),
                max_rounds=int(form.get("loop_max_rounds", 20) or 20),
                time_budget_seconds=(float(time_budget_raw) * 3600.0) if time_budget_raw else None,
                stall_rounds_before_widen=int(form.get("loop_stall_rounds", 2) or 2),
                starting_family=(None if family_key in (None, "all") else family_key) if mode_key == "family_named" else None,
                starting_max_candidates=max_candidates,
                seed=seed,
            )
            loop_dir = str(SEARCH_DIR / f"loop_{job_id}")
            initial_log.append(
                f"Loop mode ON -- will repeat Search Lab rounds (widening on stall) until a candidate "
                f"clears {loop_cfg.target_eval_pass_pct:.0f}%, {loop_cfg.max_rounds} rounds run, or the "
                f"time budget is used up."
            )
            with _SEARCH_JOBS_LOCK:
                _SEARCH_JOBS[job_id] = {
                    "log": initial_log,
                    "done": False, "error": None, "summary": None, "cancelled": False,
                    "started_at": time.time(), "instrument": active_label, "mode": mode_key,
                    "cancel_event": cancel_event, "loop_mode": True, "loop_rounds": 0,
                    "loop_last_round": None, "loop_result": None,
                }
            thread = threading.Thread(
                target=_run_search_loop_job,
                args=(job_id, df, risk, rules, stage_cfg, active_label, loop_dir, loop_cfg, cancel_event),
                kwargs={"family_health_dir": str(SEARCH_DIR)},
                daemon=True,
            )
            thread.start()
            return redirect(url_for("search_job", job_id=job_id))

        with _SEARCH_JOBS_LOCK:
            _SEARCH_JOBS[job_id] = {
                "log": initial_log,
                "done": False, "error": None, "summary": None, "cancelled": False,
                "started_at": time.time(), "instrument": active_label, "mode": mode_key,
                "cancel_event": cancel_event, "loop_mode": False,
            }
        thread = threading.Thread(
            target=_run_search_job,
            args=(job_id, df, risk, rules, space, stage_cfg, active_label, db_path, library_ref, cancel_event),
            daemon=True,
        )
        thread.start()
        return redirect(url_for("search_job", job_id=job_id))

    except StrategySpaceError as exc:
        HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
        return render_template(
            "search.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            saved_strategies_json=_saved_strategies_json(),
            prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400
    except StrategyError as exc:
        HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
        return render_template(
            "search.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            saved_strategies_json=_saved_strategies_json(),
            prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
        log_crash("Search Lab (web, start)", exc=exc)
        return render_template(
            "search.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            saved_strategies_json=_saved_strategies_json(),
            prop_presets_json=_prop_presets_json(), **_alpaca_template_context()), 500


@app.route("/search/job/<job_id>")
def search_job(job_id):
    with _SEARCH_JOBS_LOCK:
        job = _SEARCH_JOBS.get(job_id)
    if job is None:
        return render_template("search_job.html", job_id=job_id, not_found=True), 404
    return render_template("search_job.html", job_id=job_id, not_found=False)


@app.route("/search/job/<job_id>/stop", methods=["POST"])
def search_job_stop(job_id):
    """Signals the background Search Lab job to stop at its next
    between-candidate check (see app.search.batch_runner's own
    _drain_futures/check_cancelled -- this can take up to ~1s to be
    noticed, same latency as Evolution Lab's stop button). A no-op,
    not an error, if the job is already done or was never found."""
    with _SEARCH_JOBS_LOCK:
        job = _SEARCH_JOBS.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Job not found."}), 404
        if job.get("done"):
            return jsonify({"ok": True, "already_done": True})
        cancel_event = job.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    return jsonify({"ok": True})


@app.route("/search/job/<job_id>/status.json")
def search_job_status(job_id):
    with _SEARCH_JOBS_LOCK:
        job = _SEARCH_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404

    summary = job.get("summary")
    leaderboard = None
    if summary is not None:
        leaderboard = [
            {
                "candidate_id": row.get("candidate_id"),
                "source_type": row.get("source_type", "manual"),
                "family": row.get("family"),
                "composite_score": row.get("composite_score"),
                "psr": (row.get("deflated_sharpe") or {}).get("probabilistic_sharpe"),
                "net_profit": (row.get("statistics") or {}).get("net_profit"),
                "profit_factor": (row.get("statistics") or {}).get("profit_factor"),
                "win_rate": (row.get("statistics") or {}).get("win_rate"),
                "total_trades": (row.get("statistics") or {}).get("total_trades"),
                "eval_pass_pct": (row.get("mc_summary") or {}).get("evaluation_pass_probability"),
                "passed_gate": bool(row.get("passed_stage3_gate")),
                "gate_notes": row.get("gate_notes") or "",
            }
            for row in (summary.leaderboard or [])
        ]

    return jsonify({
        "found": True,
        "done": job["done"],
        "error": job["error"],
        "cancelled": job.get("cancelled", False),
        "log": job["log"],
        "instrument": job.get("instrument"),
        "summary": None if summary is None else {
            "mode": summary.mode, "family": summary.family,
            "total_candidates": summary.total_candidates,
            "stage1_survivors": summary.stage1_survivors,
            "stage2_survivors": summary.stage2_survivors,
            "stage3_survivors": summary.stage3_survivors,
            "champion_candidate_id": summary.champion_candidate_id,
            "elapsed_seconds": summary.elapsed_seconds,
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
            # UPGRADE (distribution-based results reporting): median vs.
            # best across this run's own stage3 leaderboard -- see
            # app.optimize.distribution_summary's own module docstring.
            # "candidates_tested" reports the total funnel size
            # (total_candidates), not just how many reached stage3, so
            # the funnel narrowing itself stays visible.
            "distribution_summary": compute_distribution_summary(
                [{**row, "fitness": row.get("composite_score")} for row in (summary.leaderboard or [])],
                total_tested=summary.total_candidates,
            ),
        },
        "leaderboard": leaderboard,
        "loop_mode": job.get("loop_mode", False),
        "loop_rounds": job.get("loop_rounds", 0),
        "loop_last_round": job.get("loop_last_round"),
        "loop_result": (
            None if job.get("loop_result") is None else {
                "stopped_reason": job["loop_result"].stopped_reason,
                "winner_candidate_id": job["loop_result"].winner_candidate_id,
                "total_elapsed_seconds": job["loop_result"].total_elapsed_seconds,
                "graveyard_path": job["loop_result"].graveyard_path,
                "n_rounds": len(job["loop_result"].rounds),
            }
        ),
        "next_step": (
            pipeline_guide.after_search_complete(
                summary.champion_candidate_id, len(summary.leaderboard or []),
                total_candidates=summary.total_candidates,
            )
            if (job["done"] and summary is not None) else None
        ),
    })


@app.route("/search/job/<job_id>/promote", methods=["POST"])
def search_job_promote(job_id):
    with _SEARCH_JOBS_LOCK:
        job = _SEARCH_JOBS.get(job_id)
    if job is None or not job.get("done") or job.get("summary") is None:
        return jsonify({"ok": False, "error": "Job not found, not finished, or produced no results."}), 400

    candidate_id = request.form.get("candidate_id")
    if not candidate_id and request.is_json:
        candidate_id = (request.get_json(silent=True) or {}).get("candidate_id")
    if not candidate_id:
        return jsonify({"ok": False, "error": "candidate_id is required."}), 400

    try:
        result = promote_champion(
            job["db_path"], job["summary"].run_id, candidate_id,
            job["df"], job["risk"], job["rules"],
            output_dir=str(SEARCH_DIR / "champion" / job_id),
        )
        with _SEARCH_JOBS_LOCK:
            job.setdefault("promoted", {})[candidate_id] = {
                "html": f"/search_reports_champion/{job_id}/{result['report_paths']['html'].name}",
                "json": f"/search_reports_champion/{job_id}/{result['report_paths']['json'].name}",
            }
        return jsonify({
            "ok": True,
            "report_html": job["promoted"][candidate_id]["html"],
            "next_step": (
                "Champion report generated above. Next step: if you want a walk-forward-optimized "
                "re-validation with a READY/MARGINAL/NOT READY verdict, save this candidate to the "
                "Strategy Library (Manual Builder / Strategy Library tab) and run it through Full Pipeline."
            ),
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/search/job/<job_id>/auto_ensemble", methods=["POST"])
def search_job_auto_ensemble(job_id):
    """UPGRADE (ensemble-builder UI button): app.ensemble.auto_builder.
    build_diversified_ensemble has existed since the ensemble/budget/
    integrity-check delivery but had no caller anywhere in the app --
    Owen had to hand-pick legs on the Ensemble tab instead. This turns a
    finished Search Lab run's own leaderboard straight into a diversified
    3-5-leg basket with one click, reusing the exact same instrument/
    timeframe/risk/rules the search itself just ran under."""
    with _SEARCH_JOBS_LOCK:
        job = _SEARCH_JOBS.get(job_id)
    if job is None or not job.get("done") or job.get("summary") is None:
        return jsonify({"ok": False, "error": "Job not found, not finished, or produced no results."}), 400

    form = request.form
    min_legs = int(form.get("min_legs", 3) or 3)
    max_legs = int(form.get("max_legs", 5) or 5)
    top_n = int(form.get("top_n", 50) or 50)

    try:
        with ResultsDB(job["db_path"]) as db:
            records = db.leaderboard(job["summary"].run_id, stage="stage3", top_n=top_n, only_passed=False)
        if not records:
            return jsonify({"ok": False, "error": "No stage-3 candidates on this run's leaderboard to build an ensemble from."}), 400

        result = build_diversified_ensemble(
            job["df"], records, job["risk"], prop_rules=job["rules"],
            mc_config=MonteCarloConfig(n_simulations=3000),
            min_legs=min_legs, max_legs=max_legs,
            initial_balance=job["risk"].initial_balance,
        )
        return jsonify({"ok": True, "result": result.to_summary_dict()})
    except AutoEnsembleError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        log_crash("Search Lab auto-ensemble (web)", exc=exc)
        return jsonify({"ok": False, "error": f"Unexpected error: {exc}"}), 500


@app.route("/search_reports/<path:filename>")
def serve_search_report(filename):
    return send_from_directory(SEARCH_DIR, filename)


@app.route("/search_reports_champion/<job_id>/<path:filename>")
def serve_search_champion_report(job_id, filename):
    return send_from_directory(SEARCH_DIR / "champion" / job_id, filename)


# ---------------------------------------------------------------------------
# Forge Strategy -- the literal one-button "generate, screen, validate"
# tab. See app.orchestration.forge for the actual funnel this wires up
# (hypothesis generation across every named market-hypothesis family ->
# fast screen -> prop survival screen -> neighbor testing -> walk-forward
# -> CPCV/PBO + regime testing -> deeper Monte Carlo -> rolling prop
# evaluation -> locked OOS holdout). Same background-job/poll shape as
# Search Lab above.
# ---------------------------------------------------------------------------

_FORGE_JOBS: dict[str, dict] = {}
_FORGE_JOBS_LOCK = threading.Lock()


def _forge_job_log(job_id: str, msg: str) -> None:
    with _FORGE_JOBS_LOCK:
        job = _FORGE_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_forge_job(
    job_id: str, df, risk: RiskConfig, rules: PropRules, config: ForgeConfig,
    instrument: str, db_path: str, graveyard_path: str,
    cancel_event: threading.Event | None = None,
) -> None:
    try:
        result = run_forge(
            df, risk, rules, config, db_path=db_path,
            instrument=instrument, timeframe=infer_timeframe_label(df), graveyard_path=graveyard_path,
            progress_cb=lambda msg: _forge_job_log(job_id, msg),
            cancel_event=cancel_event,
        )
        with _FORGE_JOBS_LOCK:
            job = _FORGE_JOBS[job_id]
            job["done"] = True
            job["result"] = result
    except SearchCancelled:
        with _FORGE_JOBS_LOCK:
            job = _FORGE_JOBS[job_id]
            job["done"] = True
            job["cancelled"] = True
            job["log"].append("Forge Strategy run stopped by user.")
    except Exception as exc:  # noqa: BLE001 -- must fail visibly on the status page, not crash the thread silently
        log_crash("Forge Strategy (web)", exc=exc)
        with _FORGE_JOBS_LOCK:
            job = _FORGE_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    finally:
        HEAVY_JOB_GUARD.release(JOB_FORGE)


def _run_forge_loop_job(
    job_id: str, df, risk: RiskConfig, rules: PropRules,
    instrument: str, loop_dir: str, loop_cfg: ForgeLoopConfig,
    cancel_event: threading.Event | None = None,
) -> None:
    """Forge's Loop Mode job -- same background-job/poll-for-status shape
    as _run_forge_job above, but driving app.orchestration.loop_runner.
    run_forge_loop (repeated rounds, searching deeper on stall) instead of
    a single run_forge call. job["result"] is kept updated with the MOST
    RECENT round's ForgeResult (via on_round) so the existing funnel/
    leaderboard rendering on the status page works unchanged while a loop
    is still running across many rounds."""
    def on_round(round_result) -> None:
        with _FORGE_JOBS_LOCK:
            job = _FORGE_JOBS.get(job_id)
            if job is None:
                return
            job["result"] = round_result.result
            job["loop_rounds"] = job.get("loop_rounds", 0) + 1
            job["loop_last_round"] = {
                "round_index": round_result.round_index,
                "n_hypotheses": round_result.n_hypotheses,
                "excluded_families": round_result.excluded_families,
                "champion_pass_rate_pct": round_result.champion_pass_rate_pct,
                "champion_candidate_id": round_result.champion_candidate_id,
                "widened_after_this_round": round_result.widened_after_this_round,
            }

    try:
        result = run_forge_loop(
            df, risk, rules, db_dir=loop_dir, loop_cfg=loop_cfg,
            instrument=instrument, timeframe=infer_timeframe_label(df),
            progress_cb=lambda msg: _forge_job_log(job_id, msg),
            cancel_event=cancel_event, on_round=on_round,
            family_health_search_dir=str(SEARCH_DIR), family_health_evolution_dir=str(SEARCH_DIR),
        )
        with _FORGE_JOBS_LOCK:
            job = _FORGE_JOBS[job_id]
            job["done"] = True
            job["loop_result"] = result
            # Prefer the WINNING round's result on the status page once the
            # loop finishes; on_round already kept job["result"] updated to
            # the latest round throughout the run, so this only matters
            # when a later (non-winning) round ran after the winner -- it
            # never does today (the loop breaks immediately on a winner),
            # but is the more correct choice either way.
            chosen = result.winner_round or (result.rounds[-1] if result.rounds else None)
            job["result"] = chosen.result if chosen else job.get("result")
            job["loop_rounds"] = len(result.rounds)
            if result.stopped_reason == "cancelled":
                job["cancelled"] = True
            elif result.stopped_reason == "error":
                job["error"] = result.error
    except Exception as exc:  # noqa: BLE001 -- must fail visibly on the status page, not crash the thread silently
        log_crash("Forge Strategy Loop Mode (web)", exc=exc)
        with _FORGE_JOBS_LOCK:
            job = _FORGE_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    finally:
        HEAVY_JOB_GUARD.release(JOB_FORGE)


@app.route("/research")
def research_form():
    return render_template(
        "research.html",
        alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        manual_strategies=list_saved_strategies("manual"),
        active_page="research", **_alpaca_template_context())


@app.route("/research/run", methods=["POST"])
def research_run():
    form = request.form

    def _rerender(error, status=400):
        return render_template(
            "research.html", error=error,
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            manual_strategies=list_saved_strategies("manual"),
            selected_dataset=(form.get("existing_dataset") or ""),
            selected_strategy=(form.get("strategy_file") or ""),
            active_page="research", **_alpaca_template_context()), status

    df, active_label, _import_note, dataset_error = _resolve_dataset(form, request.files)
    if dataset_error:
        return _rerender(dataset_error)

    strategy_file = (form.get("strategy_file") or "").strip()
    if not strategy_file:
        return _rerender("Pick a saved Manual Strategy Builder strategy in Step 2 first.")
    try:
        spec = {"source_type": "manual", "config": json.loads(load_strategy_text("manual", strategy_file))}
    except Exception as exc:
        return _rerender(f"Couldn't load strategy '{strategy_file}': {exc}")

    try:
        risk = RiskConfig(
            initial_balance=float(form.get("account_size", 100000) or 100000),
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        )
        window_trading_days = int(form.get("window_trading_days", 30) or 30)

        full_bt = research_director._run_spec(spec, df, risk)
        if full_bt is None or not full_bt.trades:
            return _rerender(
                "This strategy produced zero trades on the selected dataset with the current risk/prop "
                "settings -- nothing to analyze. Check the strategy's conditions and the pip size above."
            )
        target_row = research_director._row(strategy_file, full_bt, rules, window_trading_days)

        results = {}
        if form.get("run_decomposition") == "on":
            try:
                results["decomposition"] = research_director.edge_decomposition(spec, df, risk, rules, window_trading_days)
            except Exception as exc:
                results["decomposition"] = {"steps": [], "verdict": f"Couldn't run: {exc}"}
        if form.get("run_ablation") == "on":
            try:
                results["ablation"] = research_director.ablation_test(spec, df, risk, rules, window_trading_days)
            except Exception as exc:
                results["ablation"] = {"rows": [], "verdict": f"Couldn't run: {exc}"}
        if form.get("run_null") == "on":
            try:
                results["null_baselines"] = research_director.null_baselines(df, risk, rules, target_row, window_trading_days)
            except Exception as exc:
                results["null_baselines"] = {"baselines": [], "target": target_row, "verdict": f"Couldn't run: {exc}"}
        if form.get("run_degradation") == "on":
            try:
                results["degradation"] = research_director.signal_degradation(spec, df, risk, rules, window_trading_days)
            except Exception as exc:
                results["degradation"] = {"baseline": target_row, "stress_tests": [], "verdict": f"Couldn't run: {exc}"}
        if form.get("run_contribution") == "on":
            results["contribution"] = research_director.trade_contribution(full_bt.trades, risk.initial_balance)
        if form.get("run_conditional") == "on":
            results["conditional"] = research_director.conditional_expectancy(full_bt.trades, df)
        if form.get("run_regime") == "on":
            n = len(df)
            holdout_df = df.iloc[int(n * 0.85):].reset_index(drop=True)
            try:
                results["regime"] = research_director.regime_discovery(full_bt.trades, df, risk, rules, spec, holdout_df, window_trading_days)
            except Exception as exc:
                results["regime"] = {"hypothesis": None, "next_step": f"Couldn't run: {exc}"}

        return render_template(
            "research.html", results=results,
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            manual_strategies=list_saved_strategies("manual"),
            selected_dataset=(form.get("existing_dataset") or ""),
            selected_strategy=strategy_file,
            active_page="research", **_alpaca_template_context())
    except Exception as exc:  # noqa: BLE001
        log_crash("Research Director (web)", exc=exc)
        return _rerender(f"Unexpected error: {exc}", status=500)


@app.route("/forge")
def forge_form():
    return render_template(
        "forge.html",
        alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        families=[{"name": n, "hypothesis": hypothesis_question(n)} for n in list_families()],
        active_page="forge", **_alpaca_template_context())


@app.route("/forge/start", methods=["POST"])
def forge_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_FORGE):
        return render_template(
            "forge.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job at the same time can exhaust available memory. Wait for it to finish "
                f"before starting Forge Strategy."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            active_page="forge", **_alpaca_template_context()), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_FORGE)
            return render_template(
                "forge.html", error=dataset_error,
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                active_page="forge", **_alpaca_template_context()), 400

        seed = int(form.get("seed", 42) or 42)
        workers_raw = (form.get("workers") or "").strip()
        advanced = (form.get("advanced_mode") or "") == "on"

        config = ForgeConfig(
            n_hypotheses=int(form.get("n_hypotheses", 10_000) or 10_000),
            seed=seed,
            min_trades=int(form.get("min_trades", 20) or 20) if advanced else 20,
            min_profit_factor=float(form.get("min_profit_factor", 1.05) or 1.05) if advanced else 1.05,
            stage1_top_n=int(form.get("stage1_top_n", 1000) or 1000) if advanced else 1000,
            ga_population=int(form.get("ga_population", 12) or 12) if advanced else 12,
            ga_generations=int(form.get("ga_generations", 4) or 4) if advanced else 4,
            stage2_top_n=int(form.get("stage2_top_n", 200) or 200) if advanced else 200,
            stage3_mc_sims=int(form.get("stage3_mc_sims", 2000) or 2000) if advanced else 2000,
            cpcv_pool_size=int(form.get("cpcv_pool_size", 30) or 30) if advanced else 30,
            cpcv_survivors=int(form.get("cpcv_survivors", 10) or 10) if advanced else 10,
            final_mc_sims=int(form.get("final_mc_sims", 10_000) or 10_000) if advanced else 10_000,
            mc_survivors=int(form.get("mc_survivors", 5) or 5) if advanced else 5,
            eval_window_days=int(form.get("eval_window_days", 60) or 60) if advanced else 60,
            rolling_survivors=int(form.get("rolling_survivors", 2) or 2) if advanced else 2,
            locked_holdout_frac=float(form.get("locked_holdout_frac", 0.15) or 0.15) if advanced else 0.15,
            workers=int(workers_raw) if workers_raw else None,
            random_seed=seed,
            reset_on_breach=form.get("reset_on_breach", "on") == "on",
        )
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        )

        job_id = uuid.uuid4().hex[:12]
        db_path = str(FORGE_DIR / f"forge_{job_id}.db")
        # Shared, persistent graveyard path -- deliberately NOT one unique
        # file per job. A per-job graveyard file was the bug that made the
        # "map of dead strategy space" pointless: every run wrote its
        # rejections somewhere no future run (or the /graveyard page) would
        # ever read again, so nothing ever accumulated and Forge's own
        # graveyard-feedback pre-filter (app.orchestration.forge, Stage 0)
        # never had anything to find. One shared file per instrument+
        # timeframe means a rejection from Monday's run is still visible
        # (and still skippable) in Friday's.
        graveyard_path = str(graveyard_path_for(active_label, infer_timeframe_label(df)))
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        cancel_event = threading.Event()

        loop_mode_on = form.get("loop_mode") == "on"
        if loop_mode_on:
            time_budget_raw = (form.get("loop_time_budget_hours") or "").strip()
            loop_cfg = ForgeLoopConfig(
                target_pass_rate_pct=float(form.get("loop_target_pass_rate_pct", 60) or 60),
                require_locked_oos_passed=form.get("loop_require_locked_oos", "on") == "on",
                max_rounds=int(form.get("loop_max_rounds", 10) or 10),
                time_budget_seconds=(float(time_budget_raw) * 3600.0) if time_budget_raw else None,
                stall_rounds_before_widen=int(form.get("loop_stall_rounds", 2) or 2),
                starting_n_hypotheses=config.n_hypotheses,
                starting_stage1_top_n=config.stage1_top_n,
                starting_stage2_top_n=config.stage2_top_n,
                seed=seed,
                base_config=config,
            )
            loop_dir = str(FORGE_DIR / f"loop_{job_id}")
            initial_log.append(
                f"Loop mode ON -- will repeat Forge rounds (searching deeper on stall) until a "
                f"champion clears {loop_cfg.target_pass_rate_pct:.0f}%, {loop_cfg.max_rounds} rounds "
                f"run, or the time budget is used up."
            )
            with _FORGE_JOBS_LOCK:
                _FORGE_JOBS[job_id] = {
                    "log": initial_log,
                    "done": False, "error": None, "result": None, "cancelled": False,
                    "started_at": time.time(), "instrument": active_label,
                    "cancel_event": cancel_event, "graveyard_path": graveyard_path,
                    "loop_mode": True, "loop_rounds": 0, "loop_last_round": None, "loop_result": None,
                }
            thread = threading.Thread(
                target=_run_forge_loop_job,
                args=(job_id, df, risk, rules, active_label, loop_dir, loop_cfg, cancel_event),
                daemon=True,
            )
            thread.start()
            return redirect(url_for("forge_job", job_id=job_id))

        with _FORGE_JOBS_LOCK:
            _FORGE_JOBS[job_id] = {
                "log": initial_log,
                "done": False, "error": None, "result": None, "cancelled": False,
                "started_at": time.time(), "instrument": active_label,
                "cancel_event": cancel_event, "graveyard_path": graveyard_path, "loop_mode": False,
            }
        thread = threading.Thread(
            target=_run_forge_job,
            args=(job_id, df, risk, rules, config, active_label, db_path, graveyard_path, cancel_event),
            daemon=True,
        )
        thread.start()
        return redirect(url_for("forge_job", job_id=job_id))

    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_FORGE)
        log_crash("Forge Strategy (web, start)", exc=exc)
        return render_template(
            "forge.html", error=f"Unexpected error: {exc}",
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            active_page="forge", **_alpaca_template_context()), 500


@app.route("/forge/job/<job_id>")
def forge_job(job_id):
    with _FORGE_JOBS_LOCK:
        job = _FORGE_JOBS.get(job_id)
    if job is None:
        return render_template("forge_job.html", job_id=job_id, not_found=True), 404
    return render_template("forge_job.html", job_id=job_id, not_found=False)


@app.route("/forge/job/<job_id>/stop", methods=["POST"])
def forge_job_stop(job_id):
    with _FORGE_JOBS_LOCK:
        job = _FORGE_JOBS.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Job not found."}), 404
        cancel_event = job.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    return jsonify({"ok": True})


@app.route("/forge/job/<job_id>/status.json")
def forge_job_status(job_id):
    with _FORGE_JOBS_LOCK:
        job = _FORGE_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404

    result = job.get("result")
    return jsonify({
        "found": True,
        "done": job["done"],
        "error": job["error"],
        "cancelled": job.get("cancelled", False),
        "log": job["log"],
        "instrument": job.get("instrument"),
        "funnel": [s.to_dict() for s in result.funnel] if result else None,
        "leaderboard": [r.to_dict() for r in result.leaderboard] if result else None,
        "diagnoses": [d.to_dict() for d in result.diagnoses] if result else None,
        "champion_candidate_id": result.champion_candidate_id if result else None,
        "cohort_pbo": result.cohort_pbo if result else None,
        "elapsed_seconds": result.elapsed_seconds if result else None,
        "loop_mode": job.get("loop_mode", False),
        "loop_rounds": job.get("loop_rounds", 0),
        "loop_last_round": job.get("loop_last_round"),
        "loop_result": (
            None if job.get("loop_result") is None else {
                "stopped_reason": job["loop_result"].stopped_reason,
                "winner_candidate_id": job["loop_result"].winner_candidate_id,
                "total_elapsed_seconds": job["loop_result"].total_elapsed_seconds,
                "n_rounds": len(job["loop_result"].rounds),
            }
        ),
    })


@app.route("/graveyard")
def graveyard_view():
    """Strategy Graveyard browser -- Owen's ask: 'this should be open
    source to the user to see exactly what failed and then fed back into
    the machine to improve subsequent runs.' The feeding-back-in half
    already happens automatically (see app.orchestration.forge, Stage 0:
    known-dead parameter neighborhoods are skipped before a single
    backtest runs); this route is the missing other half -- actually
    being able to SEE it. Every Forge run against a given instrument+
    timeframe writes to the same persistent file (app.search.graveyard.
    graveyard_path_for), so this accumulates across runs instead of
    resetting every time."""
    files = list_graveyard_files()
    selected_path = (request.args.get("path") or "").strip()
    if not selected_path and files:
        selected_path = files[0]["path"]
    rows: list[dict] = []
    clusters = []
    if selected_path:
        rows = load_graveyard(selected_path)
        clusters = [c.to_dict() for c in summarize_graveyard(rows, top_n=100)]
    family_filter = (request.args.get("family") or "").strip()
    if family_filter:
        clusters = [c for c in clusters if c["family"] == family_filter]
    all_families = sorted({r.get("family", "?") for r in rows}) if rows else []
    return render_template(
        "graveyard.html",
        files=files, selected_path=selected_path, clusters=clusters,
        total_rows=len(rows), family_filter=family_filter, all_families=all_families,
        active_page="graveyard",
    )


# ---------------------------------------------------------------------------
# Multi-Instrument Search -- runs the SAME family/grid search space
# CONCURRENTLY against several instrument/timeframe datasets instead of one
# at a time (see app.orchestration.multi_instrument_search for the actual
# orchestration this wires up). A real edge is often instrument- and
# timeframe-dependent, so this covers more ground per unit wall-clock time
# than repeated single-instrument Search Lab runs. Same background-job/poll
# shape as Search Lab above, just fanned out across N datasets picked via a
# multi-select instead of one dataset picker.
# ---------------------------------------------------------------------------

_MULTI_SEARCH_JOBS: dict[str, dict] = {}
_MULTI_SEARCH_JOBS_LOCK = threading.Lock()


def _multi_job_log(job_id: str, label: str, msg: str) -> None:
    with _MULTI_SEARCH_JOBS_LOCK:
        job = _MULTI_SEARCH_JOBS.get(job_id)
        if job is not None:
            job["log"].append(f"[{label}] {msg}")


def _run_multi_search_job(
    job_id: str, jobs: list[InstrumentJob], space, risk: RiskConfig, rules: PropRules,
    stage_cfg: SearchStageConfig, max_concurrent: int,
) -> None:
    try:
        db_dir = SEARCH_DIR / "multi_instrument" / job_id
        results = run_multi_instrument_search(
            jobs, space, risk, rules, stage_cfg, db_dir,
            max_concurrent_instruments=max_concurrent,
            progress_cb=lambda label, msg: _multi_job_log(job_id, label, msg),
        )
        per_instrument = {}
        for label, res in results.items():
            if res.error:
                per_instrument[label] = {"error": res.error.splitlines()[0]}
                continue
            s = res.summary
            report_paths = generate_search_report(
                output_dir=str(db_dir / label.replace("/", "_")), summary=s, space=space,
                instrument=res.job.instrument, timeframe=res.job.timeframe,
            )
            per_instrument[label] = {
                "error": None,
                "total_candidates": s.total_candidates,
                "stage1_survivors": s.stage1_survivors,
                "stage2_survivors": s.stage2_survivors,
                "stage3_survivors": s.stage3_survivors,
                "champion_candidate_id": s.champion_candidate_id,
                "report_html": f"/search_reports_multi/{job_id}/{label.replace('/', '_')}/{report_paths['html'].name}",
            }

        best = best_result_across_instruments(results)
        best_label = None
        champion_report = None
        if best is not None:
            best_label = best.label
            try:
                import_result = import_csv(best.job.csv_path)
                promo = promote_champion(
                    best.summary.db_path, best.summary.run_id, best.summary.champion_candidate_id,
                    import_result.dataframe, risk, rules,
                    output_dir=str(db_dir / best_label.replace("/", "_") / "champion"),
                )
                champion_report = f"/search_reports_multi/{job_id}/{best_label.replace('/', '_')}/champion/{promo['report_paths']['html'].name}"
            except Exception:  # noqa: BLE001 -- a champion-promotion hiccup must not hide the otherwise-successful search results
                pass

        with _MULTI_SEARCH_JOBS_LOCK:
            job = _MULTI_SEARCH_JOBS[job_id]
            job["done"] = True
            job["results"] = per_instrument
            job["best_label"] = best_label
            job["champion_report"] = champion_report
    except Exception as exc:  # noqa: BLE001
        log_crash("Multi-Instrument Search (web)", exc=exc)
        with _MULTI_SEARCH_JOBS_LOCK:
            job = _MULTI_SEARCH_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    finally:
        HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SEARCH)


def _run_multi_search_loop_job(
    job_id: str, jobs: list[InstrumentJob], loop_cfg: SearchLoopConfig, risk: RiskConfig,
    rules: PropRules, stage_cfg: SearchStageConfig, max_concurrent: int,
    cancel_event: threading.Event | None = None,
) -> None:
    """Loop Mode's counterpart to _run_multi_search_job above -- drives
    run_multi_instrument_search_loop (an independent Loop Mode run per
    instrument, concurrently) instead of a single run_multi_instrument_search
    call. Deliberately simpler than the non-loop job in one respect: no
    automatic "promote the best instrument's champion" step -- each
    instrument's own winning round already has its own report link below,
    and picking a single best-across-instruments champion the way
    best_result_across_instruments does would need a loop-aware version of
    that ranking; left as a manual next step (open that instrument's own
    report, use the regular Search Lab promote flow) rather than building
    that ranking helper for this round."""
    try:
        db_dir = SEARCH_DIR / "multi_instrument_loop" / job_id
        results = run_multi_instrument_search_loop(
            jobs, risk, rules, stage_cfg, loop_cfg, db_dir,
            max_concurrent_instruments=max_concurrent,
            progress_cb=lambda label, msg: _multi_job_log(job_id, label, msg),
            cancel_event=cancel_event,
        )
        per_instrument = {}
        for label, res in results.items():
            if res.error:
                per_instrument[label] = {"error": res.error.splitlines()[0]}
                continue
            lr = res.loop_result
            chosen = lr.winner_round or (lr.rounds[-1] if lr.rounds else None)
            report_html = None
            if chosen is not None and chosen.summary.leaderboard:
                try:
                    report_paths = generate_search_report(
                        output_dir=str(db_dir / label.replace("/", "_")), summary=chosen.summary, space=chosen.space,
                        instrument=res.job.instrument, timeframe=res.job.timeframe,
                    )
                    report_html = f"/search_reports_multi_loop/{job_id}/{label.replace('/', '_')}/{report_paths['html'].name}"
                except Exception:  # noqa: BLE001 -- a report-generation hiccup must not hide the otherwise-successful loop result
                    pass
            per_instrument[label] = {
                "error": None,
                "stopped_reason": lr.stopped_reason,
                "n_rounds": len(lr.rounds),
                "winner_candidate_id": lr.winner_candidate_id,
                "total_candidates": chosen.summary.total_candidates if chosen else None,
                "stage3_survivors": chosen.summary.stage3_survivors if chosen else None,
                "report_html": report_html,
            }

        with _MULTI_SEARCH_JOBS_LOCK:
            job = _MULTI_SEARCH_JOBS[job_id]
            job["done"] = True
            job["results"] = per_instrument
            job["cancelled"] = any(
                r.loop_result is not None and r.loop_result.stopped_reason == "cancelled"
                for r in results.values()
            )
    except Exception as exc:  # noqa: BLE001
        log_crash("Multi-Instrument Search Loop Mode (web)", exc=exc)
        with _MULTI_SEARCH_JOBS_LOCK:
            job = _MULTI_SEARCH_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    finally:
        HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SEARCH)


@app.route("/search_reports_multi_loop/<job_id>/<path:filename>")
def serve_search_report_multi_loop(job_id, filename):
    return send_from_directory(SEARCH_DIR / "multi_instrument_loop" / job_id, filename)


@app.route("/search/multi-instrument")
def search_multi_instrument_form():
    return render_template(
        "search_multi_instrument.html",
        stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        families=[{"name": n, "description": family_description(n)} for n in list_families()],
        optimizer_modes=OPTIMIZER_MODES,
    )


@app.route("/search/multi-instrument/start", methods=["POST"])
def search_multi_instrument_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_MULTI_INSTRUMENT_SEARCH):
        return render_template(
            "search_multi_instrument.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job (Search Lab / Multi-Instrument Search / Evolution Lab / Full Pipeline / "
                f"Speed Run) at the same time can exhaust available memory. Wait for it to finish first."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
        ), 409
    try:
        selected = form.getlist("datasets")
        expand_labels = parse_sweep_timeframes(form.get("expand_timeframes", ""))
        if not selected:
            HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SEARCH)
            return render_template(
                "search_multi_instrument.html",
                error="Select at least 1 dataset.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
            ), 400
        if not expand_labels and len(selected) < 2:
            HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SEARCH)
            return render_template(
                "search_multi_instrument.html",
                error="Select at least 2 datasets to search across, or fill in \"Timeframes to auto-generate\" "
                      "to expand a single dataset into several timeframes instead.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
            ), 400

        # FIX (multi-timeframe sweep): "Timeframes to auto-generate" lets one
        # (or several) selected dataset(s) stand in for N separate files --
        # each is resampled into every requested timeframe (see
        # app.data.timeframe_sweep) and written to data/raw/ as its own real
        # CSV, so the loop below builds jobs from those exactly like it
        # always has from a person's own separately-uploaded files. Every
        # dataset x every requested timeframe becomes its own job.
        jobs: list[InstrumentJob] = []
        sweep_warnings: list[str] = []
        for name in selected:
            candidate_path = get_raw_data_dir() / name
            if not candidate_path.exists():
                continue
            # Dataset names are stored as "<instrument-folder>/<file>.csv" or a
            # bare filename (see StoredDataset.name's own docstring) -- split
            # on the LAST path separator so a nested "EURUSD/EURUSD5.csv" name
            # reads as instrument="EURUSD", timeframe=the filename, while a
            # flat "XAUUSD15.csv" just uses the filename for both labels
            # (still unique, just less pretty) rather than raising here.
            stem = Path(name).stem
            instrument = name.split("/")[0] if "/" in name else stem

            if expand_labels:
                import_result = import_csv(candidate_path)
                if not import_result.is_valid:
                    sweep_warnings.append(f"{name}: could not read this dataset to expand it -- skipped.")
                    continue
                expansion = expand_dataset_across_timeframes(import_result.dataframe, instrument, expand_labels)
                sweep_warnings.extend(describe_skipped(expansion, name))
                jobs.extend(search_jobs_from_expansion(expansion))
            else:
                jobs.append(InstrumentJob(instrument=instrument, timeframe=stem, csv_path=str(candidate_path)))

        if len(jobs) < 2:
            HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SEARCH)
            return render_template(
                "search_multi_instrument.html",
                error="Could not resolve at least 2 of the selected datasets to real files on disk.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
            ), 400

        family_key = form.get("family", "all") or "all"
        _family_exclusion_log: list = []
        exclude_families = _resolve_family_exclusions(_family_exclusion_log)
        space = generate_search_space(
            mode="family", family=family_key,
            max_candidates=int(form.get("max_candidates", 300) or 300),
            seed=int(form.get("seed", 42) or 42),
            exclude_families=exclude_families,
        )
        stage_cfg = SearchStageConfig(
            min_trades=int(form.get("min_trades", 20) or 20),
            min_profit_factor=float(form.get("min_profit_factor", 1.05) or 1.05),
            stage1_top_n=int(form.get("stage1_top_n", 40) or 40),
            ga_population=int(form.get("ga_population", 10) or 10),
            ga_generations=int(form.get("ga_generations", 4) or 4),
            stage2_top_n=int(form.get("stage2_top_n", 10) or 10),
            full_mc_sims=int(form.get("full_mc_sims", 3000) or 3000),
            walk_forward_folds=int(form.get("walk_forward_folds", 4) or 4),
            robustness_neighbors=int(form.get("robustness_neighbors", 6) or 6),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            workers=None, random_seed=int(form.get("seed", 42) or 42),
            optimizer_mode=form.get("optimizer_mode", "genetic") or "genetic",
        )
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(
            account_size=float(form.get("initial_balance", 100000) or 100000),
        )
        max_concurrent = int(form.get("max_concurrent", 2) or 2)

        job_id = uuid.uuid4().hex[:12]
        loop_mode_on = form.get("loop_mode") == "on"
        if loop_mode_on:
            time_budget_raw = (form.get("loop_time_budget_hours") or "").strip()
            loop_cfg = SearchLoopConfig(
                target_eval_pass_pct=float(form.get("loop_target_eval_pass_pct", 60) or 60),
                max_rounds=int(form.get("loop_max_rounds", 20) or 20),
                time_budget_seconds=(float(time_budget_raw) * 3600.0) if time_budget_raw else None,
                stall_rounds_before_widen=int(form.get("loop_stall_rounds", 2) or 2),
                starting_family=(None if family_key in (None, "all") else family_key),
                starting_max_candidates=int(form.get("max_candidates", 300) or 300),
                seed=int(form.get("seed", 42) or 42),
            )
            cancel_event = threading.Event()
            with _MULTI_SEARCH_JOBS_LOCK:
                _MULTI_SEARCH_JOBS[job_id] = {
                    "log": [f"Loop mode ON -- searching {len(jobs)} instrument/timeframe target(s) "
                            f"independently until each clears {loop_cfg.target_eval_pass_pct:.0f}%: " +
                            ", ".join(f"{j.instrument}/{j.timeframe}" for j in jobs)]
                           + sweep_warnings + _family_exclusion_log,
                    "done": False, "error": None, "results": None,
                    "best_label": None, "champion_report": None,
                    "labels": [f"{j.instrument}/{j.timeframe}" for j in jobs],
                    "cancel_event": cancel_event, "loop_mode": True, "cancelled": False,
                }
            thread = threading.Thread(
                target=_run_multi_search_loop_job,
                args=(job_id, jobs, loop_cfg, risk, rules, stage_cfg, max_concurrent, cancel_event),
                daemon=True,
            )
            thread.start()
            return redirect(url_for("search_multi_instrument_job", job_id=job_id))

        with _MULTI_SEARCH_JOBS_LOCK:
            _MULTI_SEARCH_JOBS[job_id] = {
                "log": [f"Searching {len(jobs)} instrument/timeframe target(s): " +
                        ", ".join(f"{j.instrument}/{j.timeframe}" for j in jobs)]
                       + sweep_warnings + _family_exclusion_log,
                "done": False, "error": None, "results": None,
                "best_label": None, "champion_report": None,
                "labels": [f"{j.instrument}/{j.timeframe}" for j in jobs],
                "loop_mode": False,
            }
        thread = threading.Thread(
            target=_run_multi_search_job,
            args=(job_id, jobs, space, risk, rules, stage_cfg, max_concurrent),
            daemon=True,
        )
        thread.start()
        return redirect(url_for("search_multi_instrument_job", job_id=job_id))

    except StrategySpaceError as exc:
        HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SEARCH)
        return render_template(
            "search_multi_instrument.html", error=str(exc), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
        ), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SEARCH)
        log_crash("Multi-Instrument Search (web, start)", exc=exc)
        return render_template(
            "search_multi_instrument.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
        ), 500


@app.route("/search/multi-instrument/job/<job_id>")
def search_multi_instrument_job(job_id):
    with _MULTI_SEARCH_JOBS_LOCK:
        job = _MULTI_SEARCH_JOBS.get(job_id)
    if job is None:
        return render_template("search_multi_instrument_job.html", job_id=job_id, not_found=True), 404
    return render_template("search_multi_instrument_job.html", job_id=job_id, not_found=False)


@app.route("/search/multi-instrument/job/<job_id>/stop", methods=["POST"])
def search_multi_instrument_job_stop(job_id):
    """Only meaningful for a Loop Mode job -- the non-loop multi-instrument
    search path has no cancel_event (each instrument's single run_search
    call has always run to completion). A no-op, not an error, otherwise."""
    with _MULTI_SEARCH_JOBS_LOCK:
        job = _MULTI_SEARCH_JOBS.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Job not found."}), 404
        if job.get("done"):
            return jsonify({"ok": True, "already_done": True})
        cancel_event = job.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    return jsonify({"ok": True})


@app.route("/search/multi-instrument/job/<job_id>/status.json")
def search_multi_instrument_job_status(job_id):
    with _MULTI_SEARCH_JOBS_LOCK:
        job = _MULTI_SEARCH_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    return jsonify({
        "found": True,
        "done": job["done"],
        "error": job["error"],
        "log": job["log"][-200:],
        "labels": job.get("labels", []),
        "results": job.get("results"),
        "loop_mode": job.get("loop_mode", False),
        "cancelled": job.get("cancelled", False),
        "best_label": job.get("best_label"),
        "champion_report": job.get("champion_report"),
    })


@app.route("/search_reports_multi/<job_id>/<path:filename>")
def serve_search_report_multi(job_id, filename):
    return send_from_directory(SEARCH_DIR / "multi_instrument" / job_id, filename)


# ---------------------------------------------------------------------------
# Speed Run -- "I have almost no time left, find me anything that works."
# One button, no strategy to bring in: chains a wide multi-family discovery
# search straight into concurrent Full Pipeline validation of the top
# survivors, then reports whichever one is best. Same background-job/poll
# shape as Full Pipeline above -- see app.orchestration.speed_run for the
# actual chained-phases logic, which this route just wires up to the web UI
# the same way the desktop Speed Run tab wires it up to Tkinter.
# ---------------------------------------------------------------------------

_SPEEDRUN_JOBS: dict[str, dict] = {}
_SPEEDRUN_JOBS_LOCK = threading.Lock()


def _speedrun_job_log(job_id: str, msg: str) -> None:
    with _SPEEDRUN_JOBS_LOCK:
        job = _SPEEDRUN_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_speedrun_job(
    job_id: str, df, risk: RiskConfig, rules: PropRules, cfg: SpeedRunConfig, active_label: str,
) -> None:
    try:
        result = run_speed_run(
            df, risk, rules, SPEEDRUN_DIR, cfg,
            progress_cb=lambda msg: _speedrun_job_log(job_id, msg),
            instrument=active_label,
        )
        with _SPEEDRUN_JOBS_LOCK:
            job = _SPEEDRUN_JOBS[job_id]
            job["done"] = True
            job["result"] = result
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        log_crash("Speed Run (web)", exc=exc)
        with _SPEEDRUN_JOBS_LOCK:
            job = _SPEEDRUN_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)


def _run_speedrun_loop_job(
    job_id: str, df, risk: RiskConfig, rules: PropRules, active_label: str,
    loop_dir: str, loop_cfg: SpeedRunLoopConfig, cancel_event: threading.Event | None = None,
) -> None:
    """Speed Run's Loop Mode job -- same background-job/poll-for-status
    shape as _run_speedrun_job above, but driving app.orchestration.
    loop_runner.run_speed_run_loop (repeated rounds until a winner is
    found) instead of a single run_speed_run call. Unlike the single-shot
    Speed Run job above, this DOES support cancellation (see the new
    /speed-run/job/<id>/stop route) -- a loop can run for a long time
    across many rounds, so being able to stop it partway through matters
    here in a way it didn't for one bounded discover-then-validate pass."""
    try:
        result = run_speed_run_loop(
            df, risk, rules, output_dir=loop_dir, loop_cfg=loop_cfg,
            instrument=active_label, progress_cb=lambda msg: _speedrun_job_log(job_id, msg),
            cancel_event=cancel_event,
        )
        chosen = result.winner_round or (result.rounds[-1] if result.rounds else None)
        with _SPEEDRUN_JOBS_LOCK:
            job = _SPEEDRUN_JOBS[job_id]
            job["done"] = True
            job["loop_result"] = result
            job["result"] = chosen.result if chosen else None
            job["loop_rounds"] = len(result.rounds)
            if result.stopped_reason == "cancelled":
                job["cancelled"] = True
            elif result.stopped_reason == "error":
                job["error"] = result.error
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        log_crash("Speed Run Loop Mode (web)", exc=exc)
        with _SPEEDRUN_JOBS_LOCK:
            job = _SPEEDRUN_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)


@app.route("/speed-run")
def speed_run_form():
    return render_template(
        "speed_run.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context())


@app.route("/speed-run/start", methods=["POST"])
def speed_run_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_SPEED_RUN):
        return render_template(
            "speed_run.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job (Search Lab / Evolution Lab / Full Pipeline / Speed Run) at the same "
                f"time can exhaust available memory -- this is the same failure mode that can freeze "
                f"or crash the desktop app. Wait for it to finish before starting Speed Run."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)
            return render_template(
                "speed_run.html", error=dataset_error, stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
                fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 400

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        cfg = SpeedRunConfig(
            max_candidates=int(form.get("max_candidates", 1200) or 1200),
            stage1_top_n=int(form.get("stage1_top_n", 24) or 24),
            ga_population=int(form.get("ga_population", 8) or 8),
            ga_generations=int(form.get("ga_generations", 3) or 3),
            top_k_to_validate=int(form.get("top_k_to_validate", 3) or 3),
            max_concurrent_validations=int(form.get("max_concurrent_validations", 2) or 2),
            validation_folds=int(form.get("validation_folds", 3) or 3),
            validation_final_mc_sims=int(form.get("validation_final_mc_sims", 3000) or 3000),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            save_winner_to_library=form.get("save_to_library") == "on",
            random_seed=int(form.get("random_seed", 42) or 42),
            reset_on_breach=form.get("reset_on_breach", "on") == "on",
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)

        loop_mode_on = form.get("loop_mode") == "on"
        if loop_mode_on:
            time_budget_raw = (form.get("loop_time_budget_hours") or "").strip()
            loop_cfg = SpeedRunLoopConfig(
                max_rounds=int(form.get("loop_max_rounds", 10) or 10),
                time_budget_seconds=(float(time_budget_raw) * 3600.0) if time_budget_raw else None,
                stall_rounds_before_widen=int(form.get("loop_stall_rounds", 2) or 2),
                starting_max_candidates=cfg.max_candidates,
                starting_top_k_to_validate=cfg.top_k_to_validate,
                seed=cfg.random_seed,
                base_config=cfg,
            )
            loop_dir = str(SPEEDRUN_DIR / f"loop_{job_id}")
            initial_log.append(
                f"Loop mode ON -- will repeat Speed Run rounds (raising the candidate cap and "
                f"validation width on a stall) until a round finds a winner, {loop_cfg.max_rounds} "
                f"rounds run, or the time budget is used up."
            )
            cancel_event = threading.Event()
            with _SPEEDRUN_JOBS_LOCK:
                _SPEEDRUN_JOBS[job_id] = {
                    "log": initial_log, "done": False, "error": None, "result": None,
                    "started_at": time.time(), "instrument": active_label,
                    "cancel_event": cancel_event, "loop_mode": True, "loop_rounds": 0,
                    "loop_result": None, "cancelled": False,
                }
            thread = threading.Thread(
                target=_run_speedrun_loop_job,
                args=(job_id, df, risk, rules, active_label, loop_dir, loop_cfg, cancel_event),
                daemon=True,
            )
            thread.start()
            return redirect(url_for("speed_run_job", job_id=job_id))

        with _SPEEDRUN_JOBS_LOCK:
            _SPEEDRUN_JOBS[job_id] = {
                "log": initial_log, "done": False, "error": None, "result": None,
                "started_at": time.time(), "instrument": active_label, "loop_mode": False,
            }
        thread = threading.Thread(
            target=_run_speedrun_job, args=(job_id, df, risk, rules, cfg, active_label), daemon=True,
        )
        thread.start()
        return redirect(url_for("speed_run_job", job_id=job_id))

    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)
        log_crash("Speed Run (web, start)", exc=exc)
        return render_template(
            "speed_run.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
            fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES, **_alpaca_template_context()), 500


@app.route("/speed-run/job/<job_id>")
def speed_run_job(job_id):
    with _SPEEDRUN_JOBS_LOCK:
        job = _SPEEDRUN_JOBS.get(job_id)
    if job is None:
        return render_template("speed_run_job.html", job_id=job_id, not_found=True), 404
    return render_template("speed_run_job.html", job_id=job_id, not_found=False)


@app.route("/speed-run/job/<job_id>/stop", methods=["POST"])
def speed_run_job_stop(job_id):
    """Only meaningful for a Loop Mode job -- the single-shot Speed Run
    path has no cancel_event at all (one bounded discover-then-validate
    pass has always run to completion; see _run_speedrun_job above). A
    no-op, not an error, for a non-loop job or one already finished."""
    with _SPEEDRUN_JOBS_LOCK:
        job = _SPEEDRUN_JOBS.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Job not found."}), 404
        if job.get("done"):
            return jsonify({"ok": True, "already_done": True})
        cancel_event = job.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    return jsonify({"ok": True})


@app.route("/speed-run/job/<job_id>/status.json")
def speed_run_job_status(job_id):
    with _SPEEDRUN_JOBS_LOCK:
        job = _SPEEDRUN_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404

    result: SpeedRunResult | None = job.get("result")
    is_loop = job.get("loop_mode", False)

    def _report_url(report_paths: dict | None) -> str | None:
        """A loop-mode round's reports live under
        SPEEDRUN_DIR/loop_<job_id>/round_NNN/speed_run/..., not the flat
        SPEEDRUN_REPORTS_DIR a single-shot run's own report always lands
        in (see run_speed_run's own `output_dir / \"speed_run\"` call) --
        so a loop-mode job needs a different serving route
        (serve_speedrun_report_loop, scoped to that job's own
        SPEEDRUN_DIR/loop_<job_id> subtree) with a path RELATIVE to it,
        not just the bare filename the flat route uses."""
        if not report_paths or not report_paths.get("html"):
            return None
        html_path = Path(report_paths["html"])
        if not is_loop:
            return f"/speed_run_reports/{html_path.name}"
        loop_root = SPEEDRUN_DIR / f"loop_{job_id}"
        try:
            rel = html_path.relative_to(loop_root)
        except ValueError:
            return None
        return f"/speed_run_reports_loop/{job_id}/{rel.as_posix()}"

    summary = None
    if result is not None:
        winner = None
        if result.winner is not None and result.winner.pipeline_result is not None:
            pr = result.winner.pipeline_result
            winner = {
                "candidate_id": result.winner.candidate_id,
                "family": result.winner.family,
                "verdict": pr.verdict,
                "eval_pass_probability": pr.final_mc.evaluation_pass_probability,
                "first_payout_probability": pr.final_mc.first_payout_probability,
                "saved_library_note": pr.saved_library_note,
                "report_html": _report_url(pr.report_paths),
            }
        candidates = []
        for r in sorted(result.candidates, key=_speedrun_rank_key):
            if r.pipeline_result is not None:
                pr = r.pipeline_result
                candidates.append({
                    "candidate_id": r.candidate_id, "family": r.family, "verdict": pr.verdict,
                    "eval_pass_probability": pr.final_mc.evaluation_pass_probability,
                    "first_payout_probability": pr.final_mc.first_payout_probability,
                    "report_html": _report_url(pr.report_paths),
                })
            else:
                candidates.append({
                    "candidate_id": r.candidate_id, "family": r.family, "verdict": None,
                    "eval_pass_probability": None, "first_payout_probability": None, "report_html": None,
                })
        summary = {
            "winner": winner,
            "winner_reason": result.winner_reason,
            "elapsed_seconds": result.elapsed_seconds,
            "candidates": candidates,
            "guidance": result.guidance,
        }

    return jsonify({
        "found": True, "done": job["done"], "error": job["error"], "log": job["log"],
        "instrument": job.get("instrument"), "summary": summary,
        "cancelled": job.get("cancelled", False),
        "loop_mode": is_loop,
        "loop_rounds": job.get("loop_rounds", 0),
        "loop_result": (
            None if job.get("loop_result") is None else {
                "stopped_reason": job["loop_result"].stopped_reason,
                "total_elapsed_seconds": job["loop_result"].total_elapsed_seconds,
                "n_rounds": len(job["loop_result"].rounds),
            }
        ),
    })


@app.route("/speed_run_reports/<path:filename>")
def serve_speedrun_report(filename):
    return send_from_directory(SPEEDRUN_REPORTS_DIR, filename)


@app.route("/speed_run_reports_loop/<job_id>/<path:filename>")
def serve_speedrun_report_loop(job_id, filename):
    """Loop Mode's counterpart to serve_speedrun_report above -- a loop
    round's own reports live under SPEEDRUN_DIR/loop_<job_id>/round_NNN/
    speed_run/..., never the flat SPEEDRUN_REPORTS_DIR a single-shot run's
    report lands in, so this needs its own per-job serving root (see
    speed_run_job_status's own _report_url helper, which builds the
    matching relative path)."""
    return send_from_directory(SPEEDRUN_DIR / f"loop_{job_id}", filename)


# ---------------------------------------------------------------------------
# Overnight Autopilot -- chains Speed Run discovery straight into a live
# MT5 demo forward test of the winner (see
# app.orchestration.overnight_autopilot's module docstring for the full
# design). Same background-job/poll-for-status shape as Speed Run above,
# reusing HEAVY_JOB_GUARD/JOB_SPEED_RUN as the concurrency guard since this
# wraps run_speed_run internally. The forward-test-start step naturally
# no-ops on a server with no MT5 terminal available -- same restriction the
# desktop Forward Test tab already documents -- so on the web app this is
# mainly "Speed Run plus one written report," with the live-forward-test
# half only actually kicking in when this Flask process happens to be
# running on the same Windows machine as a logged-in MT5 demo terminal.
# ---------------------------------------------------------------------------

_AUTOPILOT_JOBS: dict[str, dict] = {}
_AUTOPILOT_JOBS_LOCK = threading.Lock()
AUTOPILOT_DIR = BASE_DIR / "reports" / "autopilot"
AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)


def _autopilot_job_log(job_id: str, msg: str) -> None:
    with _AUTOPILOT_JOBS_LOCK:
        job = _AUTOPILOT_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_autopilot_job(job_id: str, df, risk: RiskConfig, rules: PropRules, active_label: str, cfg: AutopilotConfig) -> None:
    try:
        result = run_overnight_autopilot(
            df, risk, rules, SPEEDRUN_DIR, cfg,
            progress_cb=lambda msg: _autopilot_job_log(job_id, msg), instrument=active_label,
        )
        with _AUTOPILOT_JOBS_LOCK:
            job = _AUTOPILOT_JOBS[job_id]
            job["done"] = True
            job["result"] = result
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        log_crash("Overnight Autopilot (web)", exc=exc)
        with _AUTOPILOT_JOBS_LOCK:
            job = _AUTOPILOT_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)


@app.route("/overnight-autopilot")
def overnight_autopilot_form():
    return render_template(
        "overnight_autopilot.html", alpaca_notice=request.args.get("alpaca_notice"), alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"), stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), **_alpaca_template_context())


@app.route("/overnight-autopilot/start", methods=["POST"])
def overnight_autopilot_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_SPEED_RUN):
        return render_template(
            "overnight_autopilot.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job at the same time can exhaust available memory. Wait for it to finish "
                f"before starting Overnight Autopilot."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), **_alpaca_template_context()), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)
            return render_template(
                "overnight_autopilot.html", error=dataset_error,
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), **_alpaca_template_context()), 400

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        speed_run_cfg = SpeedRunConfig(
            max_candidates=int(form.get("max_candidates", 1200) or 1200),
            stage1_top_n=int(form.get("stage1_top_n", 24) or 24),
            ga_population=int(form.get("ga_population", 8) or 8),
            ga_generations=int(form.get("ga_generations", 3) or 3),
            top_k_to_validate=int(form.get("top_k_to_validate", 3) or 3),
            max_concurrent_validations=int(form.get("max_concurrent_validations", 2) or 2),
            validation_folds=int(form.get("validation_folds", 3) or 3),
            validation_final_mc_sims=int(form.get("validation_final_mc_sims", 3000) or 3000),
            save_winner_to_library=form.get("save_to_library") == "on",
            reset_on_breach=form.get("reset_on_breach", "on") == "on",
        )
        autopilot_cfg = AutopilotConfig(
            speed_run_cfg=speed_run_cfg,
            auto_forward_test=form.get("auto_forward_test") == "on",
            forward_test_risk_value_pct=float(form.get("ft_risk_pct", 1.0) or 1.0),
            forward_test_max_trades_per_day=int(form.get("ft_max_trades_per_day", 10) or 10),
            report_dir=AUTOPILOT_DIR,
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _AUTOPILOT_JOBS_LOCK:
            _AUTOPILOT_JOBS[job_id] = {
                "log": initial_log, "done": False, "error": None, "result": None,
                "started_at": time.time(), "instrument": active_label,
            }
        thread = threading.Thread(
            target=_run_autopilot_job, args=(job_id, df, risk, rules, active_label, autopilot_cfg), daemon=True,
        )
        thread.start()
        return redirect(url_for("overnight_autopilot_job", job_id=job_id))
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)
        log_crash("Overnight Autopilot (web, start)", exc=exc)
        return render_template(
            "overnight_autopilot.html", error=f"Unexpected error: {exc}",
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), **_alpaca_template_context()), 500


@app.route("/overnight-autopilot/job/<job_id>")
def overnight_autopilot_job(job_id):
    with _AUTOPILOT_JOBS_LOCK:
        job = _AUTOPILOT_JOBS.get(job_id)
    if job is None:
        return render_template("overnight_autopilot_job.html", job_id=job_id, not_found=True), 404
    return render_template("overnight_autopilot_job.html", job_id=job_id, not_found=False)


@app.route("/overnight-autopilot/job/<job_id>/status.json")
def overnight_autopilot_job_status(job_id):
    with _AUTOPILOT_JOBS_LOCK:
        job = _AUTOPILOT_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404

    result = job.get("result")
    summary = None
    if result is not None:
        winner = None
        sr = result.speed_run
        if sr is not None and sr.winner is not None and sr.winner.pipeline_result is not None:
            pr = sr.winner.pipeline_result
            report_html = None
            if pr.report_paths.get("html"):
                report_html = f"/speed_run_reports/{Path(pr.report_paths['html']).name}"
            winner = {
                "candidate_id": sr.winner.candidate_id, "family": sr.winner.family,
                "verdict": pr.verdict,
                "eval_pass_probability": pr.final_mc.evaluation_pass_probability,
                "first_payout_probability": pr.final_mc.first_payout_probability,
                "report_html": report_html,
            }
        summary = {
            "winner": winner,
            "winner_verdict": result.winner_verdict,
            "forward_test_started": result.forward_test_started,
            "forward_test_message": result.forward_test_message,
            "elapsed_seconds": result.elapsed_seconds,
            "report_path": str(result.report_path),
        }

    return jsonify({
        "found": True, "done": job["done"], "error": job["error"], "log": job["log"],
        "instrument": job.get("instrument"), "summary": summary,
    })


@app.route("/autopilot_reports/<path:filename>")
def serve_autopilot_report(filename):
    return send_from_directory(AUTOPILOT_DIR, filename)


# ---------------------------------------------------------------------------
# Multi-Instrument Speed Run -- the same "run the SAME config CONCURRENTLY
# across several instrument/timeframe datasets" idea as Multi-Instrument
# Search above, applied to Speed Run instead of plain Search Lab (see
# app.orchestration.multi_instrument_speed_run for the actual orchestration
# this wires up). Same background-job/poll shape as Speed Run above, just
# fanned out across a dataset multi-select instead of one dataset picker.
# ---------------------------------------------------------------------------

_MULTI_SPEEDRUN_JOBS: dict[str, dict] = {}
_MULTI_SPEEDRUN_JOBS_LOCK = threading.Lock()
MULTI_SPEEDRUN_DIR = BASE_DIR / "reports" / "speed_run" / "multi_instrument"


def _multi_speedrun_log(job_id: str, label: str, msg: str) -> None:
    with _MULTI_SPEEDRUN_JOBS_LOCK:
        job = _MULTI_SPEEDRUN_JOBS.get(job_id)
        if job is not None:
            job["log"].append(f"[{label}] {msg}")


def _run_multi_speedrun_job(
    job_id: str, jobs: list[InstrumentJob], risk: RiskConfig, rules: PropRules,
    cfg: SpeedRunConfig, max_concurrent: int,
) -> None:
    try:
        job_dir = MULTI_SPEEDRUN_DIR / job_id
        results = run_multi_instrument_speed_run(
            jobs, risk, rules, cfg, job_dir, max_concurrent_instruments=max_concurrent,
            progress_cb=lambda label, msg: _multi_speedrun_log(job_id, label, msg),
        )
        per_instrument = {}
        for label, res in results.items():
            if res.error:
                per_instrument[label] = {"error": res.error.splitlines()[0], "has_winner": False}
                continue
            r = res.result
            winner_ctx = None
            if r.winner is not None and r.winner.pipeline_result is not None:
                pr = r.winner.pipeline_result
                winner_ctx = {
                    "candidate_id": r.winner.candidate_id, "family": r.winner.family,
                    "verdict": pr.verdict,
                    "eval_pass_probability": pr.final_mc.evaluation_pass_probability,
                    "first_payout_probability": pr.final_mc.first_payout_probability,
                    "report_html": (
                        f"/speed_run_reports_multi/{job_id}/{label.replace('/', '_')}/"
                        f"{Path(pr.report_paths['html']).name}"
                        if pr.report_paths.get("html") else None
                    ),
                }
            per_instrument[label] = {
                "error": None, "has_winner": winner_ctx is not None, "winner": winner_ctx,
                "winner_reason": r.winner_reason, "elapsed_seconds": r.elapsed_seconds,
                "guidance": r.guidance,
            }

        best = best_speed_run_across_instruments(results)
        with _MULTI_SPEEDRUN_JOBS_LOCK:
            job = _MULTI_SPEEDRUN_JOBS[job_id]
            job["done"] = True
            job["results"] = per_instrument
            job["best_label"] = best.label if best is not None else None
    except Exception as exc:  # noqa: BLE001
        log_crash("Multi-Instrument Speed Run (web)", exc=exc)
        with _MULTI_SPEEDRUN_JOBS_LOCK:
            job = _MULTI_SPEEDRUN_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    finally:
        HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SPEED_RUN)


@app.route("/speed-run/multi-instrument")
def speed_run_multi_instrument_form():
    return render_template(
        "speed_run_multi_instrument.html", stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(),
        fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
    )


@app.route("/speed-run/multi-instrument/start", methods=["POST"])
def speed_run_multi_instrument_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_MULTI_INSTRUMENT_SPEED_RUN):
        return render_template(
            "speed_run_multi_instrument.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job at the same time can exhaust available memory. Wait for it to finish first."
            ),
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
        ), 409
    try:
        selected = form.getlist("datasets")
        if len(selected) < 2:
            HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SPEED_RUN)
            return render_template(
                "speed_run_multi_instrument.html",
                error="Select at least 2 datasets to run across -- with only 1 selected, use the "
                      "regular Speed Run page instead.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
            ), 400

        jobs: list[InstrumentJob] = []
        for name in selected:
            candidate_path = get_raw_data_dir() / name
            if not candidate_path.exists():
                continue
            stem = Path(name).stem
            instrument = name.split("/")[0] if "/" in name else stem
            jobs.append(InstrumentJob(instrument=instrument, timeframe=stem, csv_path=str(candidate_path)))

        if len(jobs) < 2:
            HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SPEED_RUN)
            return render_template(
                "speed_run_multi_instrument.html",
                error="Could not resolve at least 2 of the selected datasets to real files on disk.",
                stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
            ), 400

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        )
        cfg = SpeedRunConfig(
            max_candidates=int(form.get("max_candidates", 1200) or 1200),
            stage1_top_n=int(form.get("stage1_top_n", 24) or 24),
            ga_population=int(form.get("ga_population", 8) or 8),
            ga_generations=int(form.get("ga_generations", 3) or 3),
            top_k_to_validate=int(form.get("top_k_to_validate", 3) or 3),
            max_concurrent_validations=int(form.get("max_concurrent_validations", 2) or 2),
            validation_folds=int(form.get("validation_folds", 3) or 3),
            validation_final_mc_sims=int(form.get("validation_final_mc_sims", 3000) or 3000),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            save_winner_to_library=form.get("save_to_library") == "on",
            random_seed=int(form.get("random_seed", 42) or 42),
            reset_on_breach=form.get("reset_on_breach", "on") == "on",
        )
        max_concurrent = int(form.get("max_concurrent_instruments", 2) or 2)

        job_id = uuid.uuid4().hex[:12]
        with _MULTI_SPEEDRUN_JOBS_LOCK:
            _MULTI_SPEEDRUN_JOBS[job_id] = {
                "log": [f"Running Speed Run on {len(jobs)} instrument/timeframe target(s): " +
                        ", ".join(f"{j.instrument}/{j.timeframe}" for j in jobs)],
                "done": False, "error": None, "results": None, "best_label": None,
                "labels": [f"{j.instrument}/{j.timeframe}" for j in jobs],
            }
        thread = threading.Thread(
            target=_run_multi_speedrun_job, args=(job_id, jobs, risk, rules, cfg, max_concurrent),
            daemon=True,
        )
        thread.start()
        return redirect(url_for("speed_run_multi_instrument_job", job_id=job_id))

    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SPEED_RUN)
        log_crash("Multi-Instrument Speed Run (web, start)", exc=exc)
        return render_template(
            "speed_run_multi_instrument.html", error=f"Unexpected error: {exc}",
            stored_datasets=list_stored_datasets(), dataset_groups=list_datasets_by_instrument(), fitness_metrics=FITNESS_METRICS, optimizer_modes=OPTIMIZER_MODES,
        ), 500


@app.route("/speed-run/multi-instrument/job/<job_id>")
def speed_run_multi_instrument_job(job_id):
    with _MULTI_SPEEDRUN_JOBS_LOCK:
        job = _MULTI_SPEEDRUN_JOBS.get(job_id)
    if job is None:
        return render_template("speed_run_multi_instrument_job.html", job_id=job_id, not_found=True), 404
    return render_template("speed_run_multi_instrument_job.html", job_id=job_id, not_found=False)


@app.route("/speed-run/multi-instrument/job/<job_id>/status.json")
def speed_run_multi_instrument_job_status(job_id):
    with _MULTI_SPEEDRUN_JOBS_LOCK:
        job = _MULTI_SPEEDRUN_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    return jsonify({
        "found": True, "done": job["done"], "error": job["error"], "log": job["log"][-200:],
        "labels": job.get("labels", []), "results": job.get("results"), "best_label": job.get("best_label"),
    })


@app.route("/speed_run_reports_multi/<job_id>/<path:filename>")
def serve_speedrun_report_multi(job_id, filename):
    return send_from_directory(MULTI_SPEEDRUN_DIR / job_id, filename)


# ---------------------------------------------------------------------------
# Generate Strategies (AI) -- drafts a NEW strategy's source code from a
# plain-language idea via a local Ollama model, grounded in research/ papers
# and your own best existing strategies. See app.ai.strategy_generator's
# module docstring for the full safety rationale: every result is saved
# tagged DRAFT and nothing here ever runs generated code automatically.
# Needs no market data at all, unlike everything else in this file -- it
# only drafts source code, it doesn't backtest it.
# ---------------------------------------------------------------------------

_GENSTRAT_JOBS: dict[str, dict] = {}
_GENSTRAT_JOBS_LOCK = threading.Lock()


def _genstrat_job_progress(job_id: str, tokens: int, elapsed: float) -> None:
    with _GENSTRAT_JOBS_LOCK:
        job = _GENSTRAT_JOBS.get(job_id)
        if job is not None:
            job["tokens"] = tokens
            job["elapsed"] = elapsed


def _run_genstrat_job(
    job_id: str, settings: OllamaSettings, language: str, idea: str,
    num_ctx: int, num_predict: int, stall_timeout: int, max_total: int,
) -> None:
    from app.ai.strategy_generator import generate_strategy

    try:
        result = generate_strategy(
            settings, language, idea,
            timeout=stall_timeout, max_total_seconds=max_total,
            num_ctx=num_ctx, num_predict=num_predict,
            progress_cb=lambda tokens, elapsed: _genstrat_job_progress(job_id, tokens, elapsed),
        )
        with _GENSTRAT_JOBS_LOCK:
            job = _GENSTRAT_JOBS[job_id]
            job["done"] = True
            if result.code is None:
                job["error"] = result.error or "Generation failed."
            else:
                job["code"] = result.code
                job["filename_hint"] = result.filename_hint
                job["language"] = language
                job["idea"] = idea
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        log_crash("Generate Strategies (web)", exc=exc)
        with _GENSTRAT_JOBS_LOCK:
            job = _GENSTRAT_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"


@app.route("/generate-strategies")
def generate_strategies_form():
    saved_ai = load_ollama_settings()
    return render_template(
        "generate_strategies.html", ai_enabled=saved_ai.enabled, ai_host=saved_ai.host, ai_model=saved_ai.model,
    )


@app.route("/generate-strategies/start", methods=["POST"])
def generate_strategies_start():
    form = request.form
    idea = (form.get("idea") or "").strip()
    if not idea:
        saved_ai = load_ollama_settings()
        return render_template(
            "generate_strategies.html", error="Describe the strategy idea first.",
            ai_enabled=saved_ai.enabled, ai_host=saved_ai.host, ai_model=saved_ai.model,
        ), 400

    language = form.get("language", "python")
    settings = OllamaSettings(
        enabled=True, host=form.get("ai_host", "http://localhost:11434") or "http://localhost:11434",
        model=form.get("ai_model", "llama3.1") or "llama3.1",
    )
    try:
        save_ollama_settings(settings)  # persists, same as the desktop tab's own checkbox
    except Exception:
        pass  # best-effort -- a save failure shouldn't block the run itself

    from app.ai.strategy_generator import DEFAULT_MAX_TOTAL_SECONDS, DEFAULT_NUM_CTX, DEFAULT_NUM_PREDICT, DEFAULT_TIMEOUT_SECONDS

    num_ctx = int(form.get("num_ctx", DEFAULT_NUM_CTX) or DEFAULT_NUM_CTX)
    num_predict = int(form.get("num_predict", DEFAULT_NUM_PREDICT) or DEFAULT_NUM_PREDICT)
    stall_timeout = int(form.get("stall_timeout", DEFAULT_TIMEOUT_SECONDS) or DEFAULT_TIMEOUT_SECONDS)
    max_total = int(form.get("max_total", DEFAULT_MAX_TOTAL_SECONDS) or DEFAULT_MAX_TOTAL_SECONDS)

    job_id = uuid.uuid4().hex[:12]
    with _GENSTRAT_JOBS_LOCK:
        _GENSTRAT_JOBS[job_id] = {
            "done": False, "error": None, "code": None, "filename_hint": None, "language": language,
            "idea": idea, "tokens": 0, "elapsed": 0.0, "started_at": time.time(),
        }
    thread = threading.Thread(
        target=_run_genstrat_job,
        args=(job_id, settings, language, idea, num_ctx, num_predict, stall_timeout, max_total),
        daemon=True,
    )
    thread.start()
    return redirect(url_for("generate_strategies_job", job_id=job_id))


@app.route("/generate-strategies/job/<job_id>")
def generate_strategies_job(job_id):
    with _GENSTRAT_JOBS_LOCK:
        job = _GENSTRAT_JOBS.get(job_id)
    if job is None:
        return render_template("generate_strategies_job.html", job_id=job_id, not_found=True), 404
    return render_template("generate_strategies_job.html", job_id=job_id, not_found=False)


@app.route("/generate-strategies/job/<job_id>/status.json")
def generate_strategies_job_status(job_id):
    with _GENSTRAT_JOBS_LOCK:
        job = _GENSTRAT_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    return jsonify({
        "found": True, "done": job["done"], "error": job["error"], "tokens": job["tokens"], "elapsed": job["elapsed"],
        "code": job["code"], "filename_hint": job["filename_hint"], "language": job["language"], "idea": job["idea"],
    })


@app.route("/generate-strategies/save", methods=["POST"])
def generate_strategies_save():
    """AJAX save -- mirrors the desktop tab's SAVE TO LIBRARY AS DRAFT
    button. Takes whatever's currently in the code editor on the page
    (lets you hand-edit the draft before saving, same as desktop), not
    whatever's stored in the job dict, so edits made after generation
    finished are respected."""
    payload = request.get_json(force=True, silent=True) or {}
    language = payload.get("language", "python")
    code_text = (payload.get("code") or "").rstrip("\n")
    idea = payload.get("idea", "")
    filename_stem = (payload.get("filename") or "").strip() or "generated_strategy"
    filename_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", filename_stem).strip("_") or "generated_strategy"
    ext = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}.get(language, ".py")
    filename = f"{filename_stem}{ext}"
    if not code_text:
        return jsonify({"ok": False, "error": "No code to save."}), 400
    try:
        try:
            save_strategy_text(code_text, filename, language, overwrite=False)
        except StrategyAlreadyExists:
            return jsonify({"ok": False, "error": f"'{filename}' already exists in the library. Rename it and try again."}), 409
        # Always DRAFT, regardless of anything else -- see
        # app.ai.strategy_generator's module docstring for why an
        # AI-generated strategy is never allowed to start higher than this.
        set_strategy_status(language, filename, "draft")
        save_strategy_metadata(language, filename, {"description": f"AI-drafted from idea: {idea[:200]}"})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "filename": filename})


def main():
    # UPGRADE (Sep 2026, QR-code/phone-reachability fix): this used to be
    # a bare `app.run(host="0.0.0.0", port=5000, ...)` with no banner at
    # all -- see app.web.network_info's module docstring for why that was
    # the actual root cause of "the QR code still doesn't generate" (there
    # was never a QR code generated on THIS entry point to begin with, only
    # on the separate `run_web.py` launcher). Now both entry points print
    # the identical LAN-address-and-QR-code banner.
    url = lan_url()
    qr_path = qr_code_file(url)
    print_startup_banner(url, qr_path)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
