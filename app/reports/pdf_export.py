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
    as a native PDF at `path`. Returns the written path.

    Layout mirrors the HTML report: title band, verdict hero (badge + T58 score + sorted reasons), per-attempt
    headline tiles, then grouped, humanised tables. Every figure the previous version showed is still here."""
    _require_reportlab()
    from xml.sax.saxutils import escape as _esc

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from app.reports import theme
    from app.reports.mc_headline import monte_carlo_headline

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    INK, MUTED, LINE = colors.HexColor("#111827"), colors.HexColor("#6b7280"), colors.HexColor("#e5e7eb")
    GOOD, WARN, BAD = colors.HexColor("#12a05c"), colors.HexColor("#c98200"), colors.HexColor("#d92d20")
    NAVY, PANEL = colors.HexColor("#0b1020"), colors.HexColor("#f8f9fb")

    styles = getSampleStyleSheet()
    base = ParagraphStyle("T58Body", parent=styles["BodyText"], fontSize=9, leading=12.5, textColor=INK, alignment=TA_LEFT)
    small = ParagraphStyle("T58Small", parent=base, fontSize=8, leading=10.5, textColor=MUTED)
    h2 = ParagraphStyle("T58H2", parent=base, fontName="Helvetica-Bold", fontSize=10.5, spaceBefore=16, spaceAfter=6, textColor=INK)
    h3 = ParagraphStyle("T58H3", parent=base, fontName="Helvetica-Bold", fontSize=9, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#374151"))
    title = ParagraphStyle("T58Title", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=colors.white)
    subtitle = ParagraphStyle("T58Sub", parent=base, fontSize=8.5, textColor=colors.HexColor("#9fb0d0"))
    cell = ParagraphStyle("T58Cell", parent=base, fontSize=8.6, leading=11)
    cellb = ParagraphStyle("T58CellB", parent=cell, fontName="Helvetica-Bold")
    hdr = ParagraphStyle("T58Hdr", parent=cell, fontName="Helvetica-Bold", fontSize=7.8, textColor=colors.HexColor("#374151"))

    strat = report.get("strategy", {}) or {}
    page_w = letter[0] - 1.5 * inch

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.75 * inch, 0.45 * inch, f"T58 Quant Algo Backtester  |  {strat.get('name', 'Strategy')}")
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.45 * inch, f"Page {doc_.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.6 * inch, bottomMargin=0.8 * inch,
        title=f"T58 Report - {strat.get('name', 'Strategy')}",
    )
    story: list = []

    # ---- title band --------------------------------------------------------------------------
    band = Table(
        [[Paragraph("Quant Algo Backtester - Strategy Report", title)],
         [Paragraph(f"{_esc(str(strat.get('name', 'Unnamed strategy')))}", ParagraphStyle("t2", parent=subtitle, fontSize=11, textColor=colors.white))],
         [Paragraph(
             f"{_esc(str(strat.get('instrument', '-')))} &middot; {_esc(str(strat.get('timeframe', '-')))} &middot; "
             f"{_esc(str(strat.get('backtest_period_start', '-')))} to {_esc(str(strat.get('backtest_period_end', '-')))} "
             f"&middot; source: {_esc(str(strat.get('source_type', '-')))}"
             + (f" &middot; generated {_esc(str(report.get('generated_at')))}" if report.get("generated_at") else ""), subtitle)]],
        colWidths=[page_w],
    )
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (0, 0), 12), ("BOTTOMPADDING", (0, -1), (-1, -1), 12), ("TOPPADDING", (0, 1), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -2), 1),
    ]))
    story += [band, Spacer(1, 10)]

    def _kv_rows(items):
        return [[Paragraph(_esc(str(k)), cell), Paragraph(_esc(str(v)), cellb)] for k, v in items]

    def _table(rows, col_widths=None, header=None) -> Table:
        """Two-column (label / value) table, or a headed multi-column table when `header` is given."""
        if header:
            data = [[Paragraph(_esc(h), hdr) for h in header]] + [[Paragraph(_esc(str(c)), cell) for c in r] for r in rows]
        else:
            data = _kv_rows(rows)
        ncol = len(data[0])
        widths = col_widths or ([page_w * 0.55, page_w * 0.45] if ncol == 2 else [page_w / ncol] * ncol)
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        st = [
            ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
        ]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), PANEL))
        t.setStyle(TableStyle(st))
        return t

    def _metric_table(d: dict) -> Table:
        rows = [(theme.humanize(k), _plain_value(k, v)) for k, v in d.items() if not isinstance(v, (dict, list))]
        return _table(rows)

    def _plain_value(k, v) -> str:
        import re as _re
        html_val = theme.fmt_value(k, v, escape=False)
        txt = _re.sub(r"<[^>]+>", "", html_val)
        return txt.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") or "-"

    # ---- verdict hero ------------------------------------------------------------------------
    verdict = report.get("verdict")
    if verdict:
        tone = {"READY": GOOD, "MARGINAL": WARN}.get(verdict, BAD)
        score, comps, blockers, caps, notes = theme.classify_reasons(report.get("verdict_reasons"))
        head = [[Paragraph(f'<font color="#{tone.hexval()[2:]}" size="22"><b>{_esc(verdict)}</b></font>', base),
                 Paragraph((f'<font size="18"><b>{score[0]:.1f}</b></font><font size="8" color="#6b7280"> / 100 &middot; {_esc(score[1])}</font><br/>'
                            f'<font size="7.5" color="#6b7280">{score[2]}/{score[3]} score components measured</font>') if score else "", base)]]
        ht = Table(head, colWidths=[page_w * 0.6, page_w * 0.4])
        ht.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, tone), ("LINEBEFORE", (0, 0), (0, -1), 4, tone),
                                 ("BACKGROUND", (0, 0), (-1, -1), PANEL), ("TOPPADDING", (0, 0), (-1, -1), 9),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story += [Paragraph("Full Pipeline verdict", h3), ht]
        if comps:
            rows = [(theme._COMPONENT_LABELS.get(n, theme.humanize(n)), f"{v:.0f} / 100", f"weight {w}") for n, v, w in comps]
            story += [Spacer(1, 5), _table(rows, header=("T58 score component", "Score", "Weight"),
                                           col_widths=[page_w * 0.6, page_w * 0.2, page_w * 0.2])]
        for label, items, col in (("Blocking", blockers, BAD), ("Capping / limiting", caps, WARN), ("Notes & supporting detail", notes, MUTED)):
            if items:
                story.append(Paragraph(f'<font color="#{col.hexval()[2:]}"><b>{label} ({len(items)})</b></font>', h3))
                for r in items:
                    for part in theme.split_reason(r):
                        story.append(Paragraph(f"&bull; {_esc(part)}", base))
        story.append(Spacer(1, 6))

    # ---- headline tiles (per attempt) --------------------------------------------------------
    mc = report.get("monte_carlo", {}) or {}
    if mc:
        try:
            mch = monte_carlo_headline(mc)
            fail = (mc.get("per_attempt_failure_before_payout_probability", mc.get("failure_before_payout_probability"))
                    if mc.get("reset_on_breach") else mc.get("failure_before_payout_probability"))
            med = mc.get("median_days_to_first_payout")
            tiles = [
                ("Evaluation pass", f"{mch['eval_pass_probability']:.1f}%"), ("First payout", f"{mch['first_payout_probability']:.1f}%"),
                ("Failure before payout", f"{_fmt(fail, 1)}%"), ("Median days to payout", "N/A" if med is None else str(med)),
                ("Expected payout", f"${_fmt(mc.get('expected_payout'), 0)}"), ("Risk of ruin", f"{_fmt(mc.get('risk_of_ruin_pct'), 1)}%"),
            ]
            cells = [[Paragraph(f'<font size="7" color="#6b7280">{_esc(l.upper())}</font><br/><font size="14"><b>{_esc(v)}</b></font>', base) for l, v in tiles[i:i + 3]]
                     for i in (0, 3)]
            tt = Table(cells, colWidths=[page_w / 3] * 3)
            tt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, LINE), ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                                     ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
            story += [Paragraph("Headline numbers (per account attempt)", h2), tt]
        except Exception:  # noqa: BLE001 - never lose the PDF over the tiles
            pass

    # ---- historical backtest -----------------------------------------------------------------
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

    # FIX (RESET-ACCT-001): same clarification the HTML export shows in its "Reset-on-breach was used" banner.
    reset_count = stats.get("account_reset_count", 0)
    if reset_count:
        story.append(Spacer(1, 5))
        story.append(Paragraph(
            f"<b>Reset-on-breach was used:</b> {reset_count} account reset(s) occurred. "
            f"\"Net profit\" above is CUMULATIVE P&amp;L across {reset_count + 1} simulated "
            f"accounts, not one account's result -- the account still standing at the end of "
            f"this run made ${_fmt(stats.get('final_segment_net_profit'))} of its own, over its "
            f"{_fmt(stats.get('final_segment_trade_count'), 0)} trades.",
            base,
        ))
    full_stats = {k: v for k, v in stats.items() if not isinstance(v, (dict, list))}
    if full_stats:
        story.append(Paragraph("All backtest statistics", h3))
        story.append(_metric_table(full_stats))

    avg_intended = stats.get("avg_intended_risk_dollars", 0.0)
    if avg_intended:
        story.append(Paragraph("Risk Reconciliation", h2))
        story.append(_table([
            ("Configured target risk/trade (avg)", f"${_fmt(avg_intended)}"),
            ("Actual risk at stop, given size taken (avg)", f"${_fmt(stats.get('avg_actual_stop_risk_dollars'))}"),
            ("Realized loss on losers (avg)", f"${_fmt(stats.get('avg_realized_loss_on_losers'))}"),
            ("% trades sized below target (cap/throttle)", f"{_fmt(stats.get('pct_trades_position_capped'))}%"),
            ("% trades with realized loss over their stop", f"{_fmt(stats.get('pct_trades_risk_overshoot'))}%"),
        ]))

    warnings = report.get("execution_warnings") or []
    if warnings:
        story.append(Paragraph("Execution-Integrity Warnings", h2))
        for w in warnings:
            story.append(Paragraph(f"&bull; {_esc(str(w))}", base))

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
        story.append(_metric_table({k: v for k, v in single_run.items() if k != "attempts"}))

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
        if mc.get("reset_on_breach"):
            story.append(Spacer(1, 5))
            story.append(Paragraph(
                "<b>Reset-on-breach Monte Carlo:</b> \"Evaluation pass probability\"/\"First payout "
                "probability\" above mean \"did at least one attempt anywhere in a mechanically-"
                f"rebought chain (avg {_fmt(mc.get('mean_attempts_per_path'))} attempts/path) "
                "eventually pass/get paid\" -- not one account's real odds. The per-attempt rate, "
                f"pooled across all {_fmt(mc.get('total_independent_attempts'), 0)} independent "
                f"attempts, is what answers \"if I buy ONE account\": pass "
                f"{_fmt(mc.get('per_attempt_pass_probability'))}%, payout "
                f"{_fmt(mc.get('per_attempt_payout_probability'))}%.",
                base,
            ))
        note = mc.get("methodology_note")
        if note:
            story.append(Paragraph("How these Monte Carlo numbers were produced", h3))
            import re as _re
            for part in [x.strip() for x in _re.split(r"(?<=[.;])\s+(?=[A-Z])", str(note)) if x.strip()]:
                story.append(Paragraph(f"&bull; {_esc(part)}", small))

    holdout = report.get("holdout_comparison")
    if holdout:
        story.append(Paragraph("Holdout Comparison", h2))
        story.append(_metric_table({k: v for k, v in holdout.items() if not isinstance(v, (list, dict))}))
        ins, outs = holdout.get("in_sample_statistics") or {}, holdout.get("holdout_statistics") or {}
        if ins or outs:
            keys = [("total_trades", "Trades"), ("net_profit", "Net profit"), ("profit_factor", "Profit factor"),
                    ("win_rate", "Win rate"), ("max_drawdown_pct", "Max drawdown"), ("sharpe_ratio", "Sharpe ratio")]
            rows = [(lbl, _plain_value(k, ins.get(k)), _plain_value(k, outs.get(k))) for k, lbl in keys]
            story += [Spacer(1, 5), _table(rows, header=("Metric", "In-sample", "Holdout (never re-tuned on)"))]

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Generated by T58 Quant Algo Backtester. This report reflects the assumptions in "
        "risk_config and prop_firm_rules above -- always read alongside those, not as a "
        "standalone guarantee of live results.",
        small,
    ))

    try:
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    except Exception as exc:  # noqa: BLE001
        raise PdfExportError(f"Failed to render PDF: {exc}") from exc
    return path
