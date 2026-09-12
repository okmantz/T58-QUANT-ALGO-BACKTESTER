"""
Overnight Autopilot -- chains the pieces that already exist (Speed Run
discovery, the Full Pipeline verdict Speed Run's own validation phase
already produces, and MT5 demo Forward Test) into ONE top-level run, so
"start this before bed, read one report in the morning" is an actual
button instead of babysitting three separate tabs across a work session.

This module reimplements NONE of the underlying engines -- it only
sequences them and writes one combined report:

  1. Discovery: run_speed_run() (wide family search -> Full Pipeline
     validation of the leaders -> a ranked winner with a READY/MARGINAL/
     NOT READY verdict). This is unchanged; Overnight Autopilot is a
     caller of it, not a replacement for it.
  2. Gate: only a READY or MARGINAL verdict (configurable) proceeds to
     forward testing. A NOT READY / no-winner outcome stops here, and
     the report includes Speed Run's own "what to try next" guidance.
  3. Forward test: connects to the configured MT5 demo account and
     STARTS a live ForwardTestSession for the winner, exactly like
     starting one by hand from the Forward Test tab (same connector,
     same pip-size resolution, same RiskConfig-from-live-balance
     pattern). This does NOT, and cannot, complete a meaningful forward-
     test evaluation overnight -- forward testing is a live, ongoing
     process (see app.forward_test.engine's module docstring) -- so
     "morning report" here means "your search finished, here's the
     winner, and it's already live on your MT5 demo, trading," not "here
     is a finished evaluation." The returned session is left running in
     the background; stop it the same way a manually-started one is
     stopped (.stop() / .flatten_all_and_stop()).
     Only python / pinescript / mql5 strategies can be forward-tested
     (same restriction the Forward Test tab already has -- manual/JSON
     strategies aren't supported there); a manual-config winner still
     gets a full discovery report, just without an auto-started forward
     test, and the report says so explicitly.
  4. Report: one dated Markdown file summarizing what was searched, the
     winner and its verdict, forward-test start status, and what to do
     once real trades have closed (Strategy Health / Drift Monitor --
     see app.orchestration.auto_retune for closing THAT loop too).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.backtest.risk import RiskConfig
from app.forward_test.engine import ForwardTestConfig, ForwardTestSession
from app.forward_test.journal import ForwardTestJournal
from app.forward_test.mt5_connector import MT5Connector, is_available as mt5_is_available, unavailable_reason as mt5_unavailable_reason
from app.forward_test.mt5_settings import MT5Settings, load_settings as load_mt5_settings
from app.orchestration.speed_run import SpeedRunConfig, SpeedRunResult, run_speed_run
from app.prop.simulator import PropRules
from app.strategy.library import list_saved_strategies
from app.strategy.library_loader import load_strategy_object

FORWARD_TESTABLE_TYPES = {"python", "pinescript", "mql5"}
ACCEPTABLE_VERDICTS_DEFAULT = frozenset({"READY", "MARGINAL"})


@dataclass
class AutopilotConfig:
    speed_run_cfg: SpeedRunConfig = field(default_factory=SpeedRunConfig)
    acceptable_verdicts: frozenset = ACCEPTABLE_VERDICTS_DEFAULT
    auto_forward_test: bool = True
    mt5_settings: MT5Settings | None = None       # None = load the app's saved settings
    forward_test_risk_value_pct: float = 1.0
    forward_test_max_trades_per_day: int = 10
    forward_test_poll_seconds: int = 20
    report_dir: str | Path = "reports/autopilot"


@dataclass
class AutopilotResult:
    speed_run: SpeedRunResult | None
    winner_verdict: str | None
    forward_test_started: bool
    forward_test_message: str
    forward_test_session: ForwardTestSession | None
    report_path: Path
    elapsed_seconds: float


def _find_winner_stored_strategy(saved_library_path):
    if not saved_library_path:
        return None
    saved_library_path = Path(saved_library_path)
    for stored in list_saved_strategies():
        try:
            if Path(stored.path) == saved_library_path:
                return stored
        except Exception:  # noqa: BLE001 -- a bad path on one entry must never abort the scan
            continue
    return None


def _start_forward_test(stored, cfg: AutopilotConfig, baseline_win_rate: float | None, log) -> tuple[bool, str, ForwardTestSession | None]:
    if stored is None:
        return False, "Skipped -- couldn't find the winner's saved library entry to load a Strategy object from.", None
    if stored.strategy_type not in FORWARD_TESTABLE_TYPES:
        return False, (
            f"Skipped -- the winner is a '{stored.strategy_type}' strategy; only "
            f"{sorted(FORWARD_TESTABLE_TYPES)} strategies can be forward-tested on MT5 "
            "(same restriction as the Forward Test tab). The winner is still saved to your "
            "library and ready to trade or refine by hand."
        ), None
    if not mt5_is_available():
        return False, f"Skipped -- MT5 is not available on this machine: {mt5_unavailable_reason()}", None

    mt5_settings = cfg.mt5_settings or load_mt5_settings()
    if not mt5_settings or not mt5_settings.is_usable:
        return False, "Skipped -- no saved MT5 demo account credentials (set them up in the Forward Test tab first).", None

    try:
        strategy = load_strategy_object(stored)
    except Exception as exc:  # noqa: BLE001
        return False, f"Skipped -- could not load the winner as a Strategy object: {exc}", None

    probe = MT5Connector(mt5_settings.login, mt5_settings.password, mt5_settings.server, mt5_settings.terminal_path)
    conn = probe.connect()
    if not conn.ok:
        return False, f"Skipped -- could not connect to the MT5 demo account: {conn.message}", None
    try:
        pip_size = probe.symbol_point(mt5_settings.symbol)
    except Exception:
        pip_size = 0.0001
    probe.disconnect()

    risk = RiskConfig(
        initial_balance=conn.balance or 10_000.0,
        risk_mode="percent",
        risk_value=cfg.forward_test_risk_value_pct,
        max_trades_per_day=cfg.forward_test_max_trades_per_day,
        pip_size=pip_size,
    )
    ft_cfg = ForwardTestConfig(
        symbol=mt5_settings.symbol, timeframe_minutes=mt5_settings.timeframe_minutes,
        risk=risk, poll_seconds=cfg.forward_test_poll_seconds, baseline_win_rate=baseline_win_rate,
    )
    connector = MT5Connector(mt5_settings.login, mt5_settings.password, mt5_settings.server, mt5_settings.terminal_path)
    journal = ForwardTestJournal()
    session = ForwardTestSession(
        strategy=strategy, strategy_type=stored.strategy_type, strategy_filename=stored.name,
        connector=connector, journal=journal, config=ft_cfg,
        on_log=lambda level, msg: log(f"[forward-test:{level}] {msg}"),
    )
    ok, msg = session.start()
    if not ok:
        return False, f"Failed to start: {msg}", None
    return True, (
        f"Started -- {stored.name} is now live on MT5 demo account {mt5_settings.login}@{mt5_settings.server}, "
        f"symbol {mt5_settings.symbol}, balance ${conn.balance:,.2f}."
    ), session


def _write_report(path: Path, speed_run: SpeedRunResult | None, verdict, fwd_started, fwd_message, elapsed) -> None:
    lines = [f"# Overnight Autopilot Report -- {datetime.now():%Y-%m-%d %H:%M}", ""]
    lines.append(f"Total elapsed: {elapsed / 60:.1f} minutes.")
    lines.append("")
    lines.append("## Discovery (Speed Run)")
    if speed_run is None:
        lines.append("Discovery did not run.")
    elif speed_run.winner is None:
        lines.append(f"No winner produced. Reason: {speed_run.winner_reason}")
        if speed_run.guidance:
            lines.append("")
            lines.append("Suggestions for the next run:")
            lines.extend(f"- {g}" for g in speed_run.guidance)
    else:
        w = speed_run.winner
        pr = w.pipeline_result
        lines.append(f"Winner: candidate `{w.candidate_id}` (family: {w.family or 'n/a'}) -- verdict **{pr.verdict if pr else 'N/A'}**.")
        if pr is not None:
            lines.append(f"- Eval-pass probability (final): {pr.final_mc.evaluation_pass_probability:.1f}%")
            lines.append(f"- First-payout probability: {pr.final_mc.first_payout_probability:.1f}%")
            lines.append(f"- Strategy display name: {pr.strategy_display_name}")
            if pr.saved_library_path:
                lines.append(f"- Saved to library: `{pr.saved_library_path}`")
            if pr.verdict_reasons:
                lines.append(f"- Verdict reasons: {'; '.join(pr.verdict_reasons)}")
    lines.append("")
    lines.append("## Forward test (MT5 demo)")
    lines.append(fwd_message)
    lines.append("")
    lines.append("## Next steps")
    if fwd_started:
        lines.append(
            "- The forward test is running live in the background. Once it has closed enough "
            "trades, run Strategy Health / Drift Monitor from the Forward Test tab -- or call "
            "app.orchestration.auto_retune.maybe_trigger_retune to check it and automatically "
            "re-tune the strategy for you the next time this autopilot (or a scheduled task) runs."
        )
    else:
        lines.append("- Address the reason above, then re-run the autopilot or the relevant tool directly.")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_overnight_autopilot(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    output_dir: str | Path,
    cfg: AutopilotConfig | None = None,
    progress_cb=None,
    instrument: str = "unknown",
    cancel_event: threading.Event | None = None,
) -> AutopilotResult:
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    cfg = cfg or AutopilotConfig()
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    log("Overnight Autopilot: Step 1/2 -- running Speed Run discovery (wide search -> Full Pipeline validation)...")
    speed_run = run_speed_run(
        df, risk, prop_rules, output_dir, cfg.speed_run_cfg,
        progress_cb=log, instrument=instrument, cancel_event=cancel_event,
    )

    verdict = None
    if speed_run.winner is not None and speed_run.winner.pipeline_result is not None:
        verdict = speed_run.winner.pipeline_result.verdict

    fwd_started, fwd_message, fwd_session = False, "Not attempted.", None
    if verdict not in cfg.acceptable_verdicts:
        fwd_message = (
            f"Skipped -- winner verdict was {verdict!r}, not in the acceptable set {sorted(cfg.acceptable_verdicts)}."
            if verdict else "Skipped -- no winner was produced by discovery."
        )
        log(f"Step 2/2: {fwd_message}")
    elif not cfg.auto_forward_test:
        fwd_message = "Skipped -- auto_forward_test is disabled in this autopilot's config."
        log(f"Step 2/2: {fwd_message}")
    elif cancel_event is not None and cancel_event.is_set():
        fwd_message = "Skipped -- cancelled before forward-test start."
    else:
        log("Step 2/2: winner is READY/MARGINAL -- starting a live MT5 demo forward test...")
        pr = speed_run.winner.pipeline_result
        stored = _find_winner_stored_strategy(pr.saved_library_path)
        baseline_win_rate = pr.final_bt.statistics.win_rate if pr.final_bt else None
        fwd_started, fwd_message, fwd_session = _start_forward_test(stored, cfg, baseline_win_rate, log)
        log(f"Step 2/2: {fwd_message}")

    elapsed = time.time() - t0
    report_path = report_dir / f"autopilot_{int(t0)}.md"
    _write_report(report_path, speed_run, verdict, fwd_started, fwd_message, elapsed)
    log(f"Overnight Autopilot finished in {elapsed / 60:.1f} min. Report: {report_path}")

    return AutopilotResult(
        speed_run=speed_run, winner_verdict=verdict,
        forward_test_started=fwd_started, forward_test_message=fwd_message,
        forward_test_session=fwd_session, report_path=report_path, elapsed_seconds=elapsed,
    )
