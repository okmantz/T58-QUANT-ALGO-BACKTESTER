from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration import overnight_autopilot as autopilot_mod
from app.orchestration.overnight_autopilot import (
    AutopilotConfig, run_overnight_autopilot,
)
from app.orchestration.speed_run import SpeedRunCandidateResult, SpeedRunResult
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchSummary


def _search_summary(leaderboard=None):
    return SearchSummary(
        run_id="test-run", mode="family", family="all", total_candidates=10,
        stage1_survivors=5, stage2_survivors=2, stage3_survivors=1,
        champion_candidate_id="cand-1", elapsed_seconds=1.0, db_path="x.db",
        leaderboard=leaderboard or [],
    )


def _fake_pipeline_result(verdict="READY", saved_library_path=None, win_rate=55.0):
    final_mc = types.SimpleNamespace(evaluation_pass_probability=72.0, first_payout_probability=60.0)
    final_bt = types.SimpleNamespace(statistics=types.SimpleNamespace(win_rate=win_rate))
    return types.SimpleNamespace(
        verdict=verdict, verdict_reasons=[], final_mc=final_mc, final_bt=final_bt,
        strategy_display_name="test_strategy", saved_library_path=saved_library_path,
    )


def _speed_run_result(verdict="READY", winner_present=True, saved_library_path=None):
    if not winner_present:
        return SpeedRunResult(
            search_summary=_search_summary(), candidates=[], winner=None,
            winner_reason="No candidate survived Stage 3 of discovery.",
            elapsed_seconds=1.0, guidance=["Try a different family.", "Loosen the filters."],
        )
    candidate = SpeedRunCandidateResult(
        candidate_id="cand-1", family="mean_reversion",
        pipeline_result=_fake_pipeline_result(verdict=verdict, saved_library_path=saved_library_path),
    )
    return SpeedRunResult(
        search_summary=_search_summary(), candidates=[candidate], winner=candidate,
        winner_reason="Best candidate.", elapsed_seconds=1.0,
    )


def _base_args(tmp_path):
    return dict(
        df=pd.DataFrame({"close": [1.0, 2.0, 3.0]}),
        risk=RiskConfig(initial_balance=10_000.0),
        prop_rules=PropRules(account_size=10_000.0),
        output_dir=tmp_path / "out",
    )


def test_no_winner_skips_forward_test_and_reports_guidance(tmp_path, monkeypatch):
    monkeypatch.setattr(autopilot_mod, "run_speed_run", lambda *a, **k: _speed_run_result(winner_present=False))
    cfg = AutopilotConfig(report_dir=tmp_path / "reports")

    result = run_overnight_autopilot(**_base_args(tmp_path), cfg=cfg)

    assert result.winner_verdict is None
    assert not result.forward_test_started
    assert "no winner" in result.forward_test_message.lower() or "no winner" in result.forward_test_message.lower()
    assert result.report_path.exists()
    text = result.report_path.read_text()
    assert "No winner produced" in text
    assert "Try a different family" in text


def test_not_ready_verdict_skips_forward_test(tmp_path, monkeypatch):
    monkeypatch.setattr(autopilot_mod, "run_speed_run", lambda *a, **k: _speed_run_result(verdict="NOT READY"))
    cfg = AutopilotConfig(report_dir=tmp_path / "reports")

    result = run_overnight_autopilot(**_base_args(tmp_path), cfg=cfg)

    assert result.winner_verdict == "NOT READY"
    assert not result.forward_test_started
    assert "not in the acceptable set" in result.forward_test_message


def test_auto_forward_test_disabled_skips_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(autopilot_mod, "run_speed_run", lambda *a, **k: _speed_run_result(verdict="READY"))
    cfg = AutopilotConfig(report_dir=tmp_path / "reports", auto_forward_test=False)

    result = run_overnight_autopilot(**_base_args(tmp_path), cfg=cfg)

    assert result.winner_verdict == "READY"
    assert not result.forward_test_started
    assert "disabled" in result.forward_test_message


def test_ready_verdict_but_no_saved_library_entry_skips_gracefully(tmp_path, monkeypatch):
    # verdict is READY/MARGINAL and auto_forward_test is on, but no
    # saved_library_path means there's no Strategy object to load --
    # must fail closed with a clear message, never raise.
    monkeypatch.setattr(autopilot_mod, "run_speed_run", lambda *a, **k: _speed_run_result(verdict="MARGINAL", saved_library_path=None))
    cfg = AutopilotConfig(report_dir=tmp_path / "reports", auto_forward_test=True)

    result = run_overnight_autopilot(**_base_args(tmp_path), cfg=cfg)

    assert result.winner_verdict == "MARGINAL"
    assert not result.forward_test_started
    assert "couldn't find" in result.forward_test_message.lower()


def test_report_written_even_when_forward_test_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(autopilot_mod, "run_speed_run", lambda *a, **k: _speed_run_result(verdict="READY"))
    cfg = AutopilotConfig(report_dir=tmp_path / "reports", auto_forward_test=False)

    result = run_overnight_autopilot(**_base_args(tmp_path), cfg=cfg)
    assert result.report_path.parent == (tmp_path / "reports")
    assert result.report_path.exists()
    text = result.report_path.read_text()
    assert "Overnight Autopilot Report" in text
    assert "READY" in text
