"""
Native PDF report export -- closes the gap where the only way to get a
PDF out of this app was the browser's own print-to-PDF dialog on the
HTML report (app.reports.generator.export_html). That's fine for reading
on screen, awkward for a clean, consistently-formatted, emailable PDF
(headers/footers vary by browser, margins are whatever the print dialog
defaults to, and it requires a browser to be open at all -- not scriptable
from, say, a scheduled Full Pipeline batch run).

Consumes the exact same `report` dict app.reports.generator.build_report
already produces -- this is a fourth sibling to export_json/export_html/
export_summary_csv, not a parallel report-building pipeline, so a PDF
export can never drift out of sync with what the HTML/JSON report says.

Requires the optional `reportlab` package (pure-Python PDF generation,
no headless browser/Chromium dependency at all -- see config/
requirements.txt). Import is deferred so the rest of the app still works
without it installed; callers should catch PdfExportError and show it as
a plain message rather than a stack trace, matching this codebase's
existing pattern for every other optional dependency (pyarrow, py7zr,
scipy, scikit-learn).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class PdfExportError(RuntimeError):
    """Raised when reportlab isn't installed, or report data can't be rendered."""


def _require_reportlab():
    try:
        import reportlab  # noqa: F401
    except ImportError as exc:
        raise PdfExportError(
            "The 'reportlab' package isn't installed. Run `pip install reportlab` to enable "
            "native PDF export (this is separate from the HTML report's browser print-to-PDF, "
            "which needs no extra package)."
        ) from exc


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def export_pdf(report: dict, path: str | Path) -> Path:
    """Renders `report` (the dict returned by app.reports.generator.build_report)
    as a native PDF at `path`. Returns the written path."""
    _require_reportlab()
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("T58H1", parent=styles["Heading1"], spaceAfter=6)
    h2 = ParagraphStyle("T58H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = styles["BodyText"]

    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title=f"T58 Report - {report.get('strategy', {}).get('name', 'Strategy')}",
    )
    story: list = []

    strat = report.get("strategy", {}) or {}
    story.append(Paragraph(f"T58 Strategy Report: {strat.get('name', 'Unnamed strategy')}", h1))
    story.append(Paragraph(
        f"{strat.get('instrument', '-')} &middot; {strat.get('timeframe', '-')} &middot; "
        f"{strat.get('backtest_period_start', '-')} to {strat.get('backtest_period_end', '-')} "
        f"&middot; source: {strat.get('source_type', '-')}",
        body,
    ))
    generated_at = report.get("generated_at")
    if generated_at:
        story.append(Paragraph(f"Generated: {generated_at}", styles["Italic"]))
    story.append(Spacer(1, 10))

    verdict = report.get("verdict")
    if verdict:
        verdict_color = {"READY": colors.HexColor("#1a7f37"), "MARGINAL": colors.HexColor("#b08900"),
                          "NOT READY": colors.HexColor("#c0392b")}.get(verdict, colors.black)
        verdict_style = ParagraphStyle("T58Verdict", parent=styles["Heading2"], textColor=verdict_color)
        story.append(Paragraph(f"Verdict: {verdict}", verdict_style))
        for reason in report.get("verdict_reasons") or []:
            story.append(Paragraph(f"&bull; {reason}", body))
        story.append(Spacer(1, 8))

    def _table(rows: list[tuple[str, str]], col_widths=(2.6 * inch, 3.4 * inch)) -> Table:
        t = Table([[k, v] for k, v in rows], colWidths=list(col_widths))
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f5f5f5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    hist = report.get("historical_backtest", {}) or {}
    stats = hist.get("statistics", {}) or {}
    story.append(Paragraph("Historical Backtest", h2))
    story.append(_table([
        ("Total trades", _fmt(hist.get("total_trades"), 0)),
        ("Initial balance", f"${_fmt(hist.get('initial_balance'))}"),
        ("Final equity", f"${_fmt(hist.get('final_equity'))}"),
        ("Net profit", f"${_fmt(stats.get('net_profit'))}"),
        ("Win rate", f"{_fmt(stats.get('win_rate'))}%"),
        ("Profit factor", _fmt(stats.get("profit_factor"))),
        ("Max drawdown", f"{_fmt(stats.get('max_drawdown_pct'))}%"),
        ("Sharpe ratio", _fmt(stats.get("sharpe_ratio"))),
    ]))

    warnings = report.get("execution_warnings") or []
    if warnings:
        story.append(Paragraph("Execution-Integrity Warnings", h2))
        for w in warnings:
            story.append(Paragraph(f"&bull; {w}", body))

    rules = report.get("prop_firm_rules", {}) or {}
    story.append(Paragraph("Prop-Firm Rules", h2))
    story.append(_table([
        ("Account size", f"${_fmt(rules.get('account_size'))}"),
        ("Profit target", f"{_fmt(rules.get('evaluation_profit_target_pct'))}%"),
        ("Daily loss limit", f"{_fmt(rules.get('daily_loss_limit_pct'))}%"),
        ("Max drawdown", f"{_fmt(rules.get('max_drawdown_pct'))}% ({rules.get('drawdown_type', '-')})"),
        ("Consistency rule", f"{_fmt(rules.get('consistency_rule_pct'))}%" if rules.get("consistency_rule_pct") else "None"),
        ("Min trading days", _fmt(rules.get("min_trading_days"), 0)),
        ("News blackout windows", rules.get("news_blackout_windows") or "None"),
        ("Weekend hold allowed", _fmt(rules.get("weekend_hold_allowed"))),
        ("Max lot size", _fmt(rules.get("max_lot_size")) if rules.get("max_lot_size") else "No cap"),
        ("Hedging allowed", _fmt(rules.get("hedging_allowed"))),
    ]))

    single_run = report.get("prop_firm_single_run", {}) or {}
    if single_run:
        story.append(Paragraph("Single Historical Run vs. Prop Rules", h2))
        story.append(_table([(str(k).replace("_", " ").title(), _fmt(v)) for k, v in single_run.items()]))

    mc = report.get("monte_carlo", {}) or {}
    if mc:
        story.append(Paragraph("Monte Carlo Simulation", h2))
        story.append(_table([
            ("Simulations run", _fmt(mc.get("n_simulations"), 0)),
            ("Evaluation pass probability", f"{_fmt(mc.get('evaluation_pass_probability'))}%"),
            ("First payout probability", f"{_fmt(mc.get('first_payout_probability'))}%"),
            ("Failure before payout probability", f"{_fmt(mc.get('failure_before_payout_probability'))}%"),
            ("Median return", f"{_fmt(mc.get('median_return_pct'))}%"),
            ("Median drawdown", f"{_fmt(mc.get('median_drawdown_pct'))}%"),
            ("Worst-case drawdown (max sim)", f"{_fmt(mc.get('worst_drawdown_pct'))}%"),
            ("Risk of ruin", f"{_fmt(mc.get('risk_of_ruin_pct'))}%"),
        ]))

    holdout = report.get("holdout_comparison")
    if holdout:
        story.append(Paragraph("Holdout Comparison", h2))
        story.append(_table([(str(k).replace("_", " ").title(), _fmt(v)) for k, v in holdout.items()
                              if not isinstance(v, (list, dict))]))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Generated by T58 Prop Algo Backtester. This report reflects the assumptions in "
        "risk_config and prop_firm_rules above -- always read alongside those, not as a "
        "standalone guarantee of live results.",
        styles["Italic"],
    ))

    try:
        doc.build(story)
    except Exception as exc:  # noqa: BLE001
        raise PdfExportError(f"Failed to render PDF: {exc}") from exc
    return path
