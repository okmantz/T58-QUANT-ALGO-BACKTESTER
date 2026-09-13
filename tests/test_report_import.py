"""Tests for app.ai.report_import -- parsing uploaded HTML/CSV/PDF backtest
reports and screenshots into a summary the Research Agent can read as a
tool Observation (see app.ai.research_agent's read_uploaded_report tool).

Screenshot (vision) parsing needs a live Ollama call, so it's only tested
here for its fail-safe behavior (no settings / unreachable host) -- not
mocked end-to-end, matching this test file's sibling test_research_agent.py
which does the same for anything Ollama-dependent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ai import report_import
from app.ai.ollama_settings import OllamaSettings


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def test_import_html_finds_inline_labeled_metrics(tmp_path: Path):
    html = """
    <html><body>
    <h1>Backtest Report</h1>
    <p>Total Trades: 142</p>
    <p>Profit Factor: 1.85</p>
    <p>Max Drawdown: 12.4%</p>
    <p>Win Rate: 58.3%</p>
    </body></html>
    """
    path = tmp_path / "report.html"
    path.write_text(html, encoding="utf-8")

    result = report_import.import_report_file(path)

    assert result.kind == "html"
    assert result.metrics["trade_count"] == 142
    assert result.metrics["profit_factor"] == 1.85
    assert result.metrics["max_drawdown_pct"] == 12.4
    assert result.metrics["win_rate"] == 58.3
    assert not result.is_ai_estimated


def test_import_html_finds_table_metrics(tmp_path: Path):
    html = """
    <html><body>
    <table>
      <tr><td>Profit Factor</td><td>2.1</td></tr>
      <tr><td>Sharpe Ratio</td><td>1.4</td></tr>
    </table>
    </body></html>
    """
    path = tmp_path / "report_table.html"
    path.write_text(html, encoding="utf-8")

    result = report_import.import_report_file(path)

    assert result.metrics.get("profit_factor") == 2.1
    assert result.metrics.get("sharpe_ratio") == 1.4


def test_import_html_with_no_known_metrics_still_returns_summary(tmp_path: Path):
    path = tmp_path / "empty.html"
    path.write_text("<html><body><p>Just some unrelated text.</p></body></html>", encoding="utf-8")

    result = report_import.import_report_file(path)

    assert result.metrics == {}
    assert "unrelated text" in result.summary_text or "none of this app's known metric labels" in result.summary_text.lower()


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def test_import_csv_summary_columns(tmp_path: Path):
    df = pd.DataFrame([{
        "profit_factor": 1.72, "sharpe_ratio": 1.1, "max_drawdown_pct": 9.5,
        "win_rate": 61.0, "total_trades": 88,
    }])
    path = tmp_path / "summary.csv"
    df.to_csv(path, index=False)

    result = report_import.import_report_file(path)

    assert result.kind == "csv"
    assert result.metrics["profit_factor"] == 1.72
    assert result.metrics["trade_count"] == 88
    assert result.metrics["sharpe_ratio"] == 1.1


def test_import_csv_trade_log_computes_headline_stats(tmp_path: Path):
    rng = np.random.default_rng(0)
    pnl = list(rng.normal(5, 20, size=60))
    df = pd.DataFrame({"pnl": pnl})
    path = tmp_path / "trades.csv"
    df.to_csv(path, index=False)

    result = report_import.import_report_file(path)

    assert result.kind == "csv"
    assert result.metrics["trade_count"] == 60
    assert "profit_factor" in result.metrics
    assert "win_rate" in result.metrics
    assert any("COMPUTED from a raw per-trade PnL column" in w for w in result.warnings)


def test_import_csv_with_no_recognizable_columns_warns(tmp_path: Path):
    df = pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})
    path = tmp_path / "unrelated.csv"
    df.to_csv(path, index=False)

    result = report_import.import_report_file(path)

    assert result.metrics == {}
    assert result.warnings


# ---------------------------------------------------------------------------
# PDF (only exercised if pypdf is installed; otherwise checks the
# graceful "not installed" message, matching app.ai.research_library's
# own optional-dependency convention)
# ---------------------------------------------------------------------------

def test_import_pdf_missing_pypdf_message(tmp_path: Path, monkeypatch):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4 not a real pdf")

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = report_import.import_report_file(path)
    assert result.kind == "pdf"
    assert "pypdf" in result.summary_text.lower()


# ---------------------------------------------------------------------------
# Screenshots -- fail-safe paths only (no live Ollama in tests)
# ---------------------------------------------------------------------------

def test_import_screenshot_without_settings_fails_safe(tmp_path: Path):
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot a real png")

    result = report_import.import_report_file(path, settings=None)

    assert result.kind == "screenshot"
    assert result.is_ai_estimated is True
    assert "ollama" in result.summary_text.lower()


def test_import_screenshot_with_disabled_settings_fails_safe(tmp_path: Path):
    path = tmp_path / "shot.jpg"
    path.write_bytes(b"\xff\xd8\xff not a real jpg")

    result = report_import.import_report_file(path, settings=OllamaSettings(enabled=False))

    assert result.kind == "screenshot"
    assert result.is_ai_estimated is True


def test_unsupported_extension_returns_clear_message(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("just some notes", encoding="utf-8")

    result = report_import.import_report_file(path)

    assert result.kind == "unknown"
    assert "unsupported" in result.summary_text.lower()


# ---------------------------------------------------------------------------
# to_observation() shape -- what the Research Agent tool actually sees
# ---------------------------------------------------------------------------

def test_to_observation_flags_ai_estimated_reports_with_caution():
    report = report_import.ImportedReport(
        source_path="/tmp/shot.png", kind="screenshot", metrics={"profit_factor": 1.5},
        summary_text="Vision-read summary", is_ai_estimated=True,
    )
    obs = report.to_observation()
    assert "caution" in obs
    assert obs["metrics_found"]["profit_factor"] == 1.5


def test_to_observation_omits_caution_for_deterministic_reports():
    report = report_import.ImportedReport(
        source_path="/tmp/report.csv", kind="csv", metrics={"profit_factor": 1.5}, summary_text="Parsed CSV",
    )
    obs = report.to_observation()
    assert "caution" not in obs


# ---------------------------------------------------------------------------
# Wiring into the Research Agent's tool registry
# ---------------------------------------------------------------------------

def _agent_ctx(uploaded_reports=None):
    from app.ai import research_agent
    from app.backtest.risk import RiskConfig
    from app.prop.simulator import PropRules
    from app.strategy.manual import ManualStrategy

    def builder():
        return ManualStrategy({
            "long_entry": "sma_fast > sma_slow", "long_exit": "sma_fast < sma_slow",
            "short_entry": "sma_fast < sma_slow", "short_exit": "sma_fast > sma_slow",
            "stop_loss_pips": 20, "take_profit_pips": 40,
        })

    n = 120
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "timestamp": ts, "open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1, "volume": 100,
    })
    return research_agent.ResearchAgentContext(
        df=df, strategy_builder=builder, strategy_name="sma cross", source_type="manual",
        risk=RiskConfig(initial_balance=10_000, pip_size=0.0001), prop_rules=PropRules(account_size=10_000),
        instrument="EURUSD", uploaded_reports=uploaded_reports or [],
    )


def test_read_uploaded_report_tool_errors_when_nothing_uploaded():
    from app.ai import research_agent

    ctx = _agent_ctx()
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    result = tools["read_uploaded_report"].fn({})
    assert "error" in result


def test_read_uploaded_report_tool_reads_a_real_file(tmp_path: Path):
    from app.ai import research_agent

    path = tmp_path / "report.html"
    path.write_text("<p>Profit Factor: 1.9</p>", encoding="utf-8")
    ctx = _agent_ctx(uploaded_reports=[path])
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    result = tools["read_uploaded_report"].fn({"filename": "report.html"})
    assert result["metrics_found"]["profit_factor"] == 1.9


def test_read_uploaded_report_tool_unknown_filename_errors(tmp_path: Path):
    from app.ai import research_agent

    path = tmp_path / "report.html"
    path.write_text("<p>Profit Factor: 1.9</p>", encoding="utf-8")
    ctx = _agent_ctx(uploaded_reports=[path])
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    result = tools["read_uploaded_report"].fn({"filename": "does_not_exist.html"})
    assert "error" in result


def test_read_uploaded_report_tool_description_lists_filenames(tmp_path: Path):
    from app.ai import research_agent

    path = tmp_path / "my_report.csv"
    ctx = _agent_ctx(uploaded_reports=[path])
    tools = research_agent.build_tool_registry(ctx, OllamaSettings(enabled=False))
    assert "my_report.csv" in tools["read_uploaded_report"].description
