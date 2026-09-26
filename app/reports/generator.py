"""
Final comprehensive report generator.

Combines strategy info, historical backtest results, single-run prop-firm
results, and Monte Carlo results into one report object, then exports it as
JSON, CSV (flattened key metrics), and a self-contained HTML file (which can
be printed/saved to PDF from any browser -- avoids pulling in a heavy PDF
rendering dependency for the MVP).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.backtest.engine import BacktestResult
from app.backtest.risk import RiskConfig
from app.backtest.statistics import compute_concentration_stats, compute_cost_ladder
from app.reports._assets import T58_LOGO_BASE64
from app.monte_carlo.engine import MonteCarloResult
from app.prop.simulator import AccountSimResult, PropRules, summarize_single_run
from app.reports.charts import svg_histogram, svg_line_chart
from app.reports.trade_chart import build_trade_chart_html
from app.reports import run_history


def _headline_risk_flags(
    concentration: dict, verdict_reasons: list[str] | None, statistics: "BacktestStatistics | None" = None,
) -> list[str]:
    """Promotes two specific findings out of the report's diagnostic
    tables/verdict_reasons list into short, impossible-to-miss strings,
    meant to render as a prominent banner (see _headline_warnings_banner)
    rather than requiring someone to read the full concentration-check
    table or scroll through every verdict_reasons line.

    Added 2026-09-24 after an external RoboQuant comparison: a Full
    Pipeline report had BOTH of these findings on record (one trade was
    31.8% of gross profit and net profit went negative without it; the
    ICIR/Bonferroni significance gate couldn't even run) but neither was
    visible anywhere except by reading the raw JSON closely -- the
    verdict banner and headline metrics looked like an ordinary pass.
    """
    flags: list[str] = []

    best_pct = concentration.get("best_trade_pct_of_gross_profit") or 0.0
    net_excl_trade = concentration.get("net_profit_excluding_best_trade")
    if net_excl_trade is not None and net_excl_trade < 0:
        flags.append(
            f"⚠ Single-trade concentration: the best trade is {best_pct:.0f}% of all gross profit -- "
            f"remove it and net profit goes NEGATIVE (${net_excl_trade:,.0f}). This result is not a "
            "repeatable process; it is one trade."
        )
    elif best_pct >= 25.0:
        flags.append(
            f"⚠ Single-trade concentration: the best trade is {best_pct:.0f}% of all gross profit. "
            "Treat this result as fragile until it holds up on more trades."
        )

    if verdict_reasons:
        icir_failed = any(
            "did NOT pass the ICIR" in r or "not enough" in r.lower() and "icir" in r.lower()
            for r in verdict_reasons
        )
        icir_unavailable = any("ICIR / signal-decay" in r and "couldn't run" in r for r in verdict_reasons)
        if icir_failed or icir_unavailable:
            flags.append(
                "⚠ Signal significance UNPROVEN: too few trades/distinct periods to run the "
                "ICIR / signal-decay / Bonferroni-corrected significance gate. Treat this strategy's "
                "edge as unverified, not merely 'not yet tested'."
            )

    total_trades = getattr(statistics, "total_trades", None) if statistics is not None else None
    if total_trades is not None and total_trades < 50:
        flags.append(
            f"⚠ Small sample: only {total_trades} trade(s) in this backtest. Headline win rate, "
            "profit factor, and Monte Carlo results all inherit this same thin sample."
        )

    return flags


def _headline_warnings_banner(flags: list[str]) -> str:
    if not flags:
        return ""
    items = "".join(f"<li>{f}</li>" for f in flags)
    # Deliberately its own CSS class, not verdict-banner: this fires for
    # ANY report (concentration/small-sample checks need no verdict at
    # all), so it must never look like -- or be mistaken by a test/reader
    # for -- the Full-Pipeline-only verdict banner right below it.
    return (
        '<div class="risk-flags-banner">'
        '<div class="verdict-title">⚠ Headline risk flags</div>'
        f"<ul>{items}</ul>"
        "</div>"
    )


def build_report(
    strategy_name: str,
    strategy_source_type: str,
    instrument: str,
    timeframe: str,
    backtest_period: tuple[str, str],
    backtest_result: BacktestResult,
    prop_rules: PropRules,
    prop_single_run: AccountSimResult,
    monte_carlo_result: MonteCarloResult,
    holdout_comparison: dict | None = None,
    risk_config: RiskConfig | None = None,
    verdict: str | None = None,
    verdict_reasons: list[str] | None = None,
    final_parameters: dict[str, str] | None = None,
    baseline_parameters: dict[str, str] | None = None,
) -> dict[str, Any]:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": {
            "name": strategy_name,
            "source_type": strategy_source_type,
            "instrument": instrument,
            "timeframe": timeframe,
            "backtest_period_start": backtest_period[0],
            "backtest_period_end": backtest_period[1],
        },
        # Only ever populated by the Full Pipeline (see
        # app.orchestration.full_pipeline) -- the READY/MARGINAL/NOT READY
        # verdict and the reasons behind it used to be returned to the
        # caller and logged to the live run console only, never written
        # into the saved report itself. Someone reading the report after
        # the fact (the artifact people actually keep and share) had no
        # way to tell whether "this" was the strategy that passed.
        "verdict": verdict,
        "verdict_reasons": list(verdict_reasons) if verdict_reasons else None,
        "final_parameters": dict(final_parameters) if final_parameters else None,
        # PARAMETER-FIDELITY FIX (2026-09-24): the pre-GA values for the
        # same gene set as final_parameters, so a reader can see exactly
        # which parameters the search moved and by how much, instead of
        # only ever seeing the (possibly very different) end result under
        # a strategy name that still describes the ORIGINAL values. None
        # whenever final_parameters itself is None, or the GA never ran.
        "baseline_parameters": dict(baseline_parameters) if baseline_parameters else None,
        # Every dollar figure in this report is a direct function of this
        # config (risk % per trade, spread/slippage/commission assumptions,
        # initial balance, max trades/day). Without it recorded here, a
        # report cannot be reproduced or audited later -- this was a real
        # gap found during an external verification pass (the report had
        # no way to say what config produced its own numbers).
        "risk_config": asdict(risk_config) if risk_config is not None else None,
        "historical_backtest": {
            # NOTE: computed over the FULL, uninterrupted trade sequence --
            # prop-firm rules (daily loss limit, max drawdown, etc.) are NOT
            # enforced here, so a strategy can show a healthy net_profit in
            # this section while still failing outright in the
            # "prop_firm_single_run" section below, which walks the exact
            # same trades forward and stops the account the moment a rule
            # is breached. Always read the two together.
            "statistics": backtest_result.statistics.to_dict(),
            "total_trades": len(backtest_result.trades),
            "initial_balance": backtest_result.initial_balance,
            "final_equity": float(backtest_result.equity_curve["equity"].iloc[-1]) if len(backtest_result.equity_curve) else backtest_result.initial_balance,
        },
        # Execution-integrity warnings from app.backtest.execution.run_execution
        # (fallback stops, pip-size/instrument-scale mismatches, gap-through
        # stop fills). These used to only ever reach a live run console --
        # if the console wasn't open or scrolled past by the time someone
        # read the saved report, a warning that explained a wildly-off
        # result was permanently lost. Surfacing it in the report itself
        # (see _warnings_section in export_html) means the artifact people
        # actually keep and share always carries its own explanation.
        "execution_warnings": list(getattr(backtest_result, "warnings", []) or []),
        "concentration_check": compute_concentration_stats(backtest_result.trades),
        "cost_ladder": compute_cost_ladder(backtest_result.trades),
        "holdout_comparison": holdout_comparison,
        "prop_firm_rules": asdict(prop_rules),
        "prop_firm_single_run": summarize_single_run(prop_single_run),
        "monte_carlo": monte_carlo_result.to_dict(),
    }
    # HEADLINE-RISK-FLAGS (2026-09-24): see _headline_risk_flags's own
    # docstring. Computed once, here, from data already in `report`/
    # `verdict_reasons`/`backtest_result.statistics` so every current and
    # future caller/consumer of this dict (HTML report, JSON, any web/
    # desktop summary card that reads this report) gets it for free,
    # rather than each one needing its own re-derivation of "is this
    # fragile" from the raw tables.
    report["headline_warnings"] = _headline_risk_flags(
        report["concentration_check"], verdict_reasons, getattr(backtest_result, "statistics", None),
    )
    return report


def export_json(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def export_trades_csv(backtest_result: BacktestResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [t.to_dict() for t in backtest_result.trades]
    if not rows:
        path.write_text("no trades generated\n")
        return path
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_summary_csv(report: dict, path: str | Path) -> Path:
    """Flat key -> value CSV of the headline metrics for quick spreadsheet review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat: dict[str, Any] = {}
    flat["strategy_name"] = report["strategy"]["name"]
    flat["instrument"] = report["strategy"]["instrument"]
    flat["timeframe"] = report["strategy"]["timeframe"]

    for k, v in report["historical_backtest"]["statistics"].items():
        flat[f"backtest_{k}"] = v

    for k, v in report.get("concentration_check", {}).items():
        flat[f"concentration_{k}"] = v

    holdout = report.get("holdout_comparison")
    if holdout:
        for k, v in (holdout.get("in_sample_statistics") or {}).items():
            flat[f"holdout_in_sample_{k}"] = v
        for k, v in (holdout.get("holdout_statistics") or {}).items():
            flat[f"holdout_out_of_sample_{k}"] = v

    for k, v in report["prop_firm_single_run"].items():
        flat[f"prop_single_run_{k}"] = v

    mc = report["monte_carlo"]
    for k, v in mc.items():
        if isinstance(v, (dict, list)):
            continue
        flat[f"monte_carlo_{k}"] = v

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in flat.items():
            writer.writerow([k, v])
    return path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>T58 Quant Algo Backtester — Report: {strategy_name}</title>
<style>
  :root {{
    --ink: #14161a; --muted: #6b7280; --line: #e6e8eb; --panel: #ffffff;
    --bg: #f7f8fa; --accent: #2f6fed; --accent-dark: #111827;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    margin: 0; color: var(--ink); background: var(--bg);
  }}
  .masthead {{
    background: linear-gradient(135deg, #0b0d10 0%, #14171c 100%);
    color: #e7e9ec; padding: 22px 40px; display:flex; align-items:center; gap:16px;
  }}
  .masthead img {{ height: 40px; display:block; }}
  .masthead .titles h1 {{ margin:0; font-size:18px; font-weight:700; letter-spacing:.01em; border:none; padding:0; color:#f2f3f5;}}
  .masthead .titles .sub {{ margin-top:3px; font-size:11px; color:#9aa1ac; letter-spacing:.03em; text-transform:uppercase; }}
  .content {{ max-width: 1040px; margin: 0 auto; padding: 28px 40px 60px; }}
  .meta {{ color: var(--muted); font-size: 12.5px; margin: 2px 0 0; }}
  h2 {{
    font-size: 13px; margin-top: 34px; margin-bottom: 10px; color: var(--accent-dark);
    text-transform: uppercase; letter-spacing: .06em; font-weight: 700;
    border-bottom: 2px solid var(--accent-dark); padding-bottom: 6px;
  }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 4px; background: var(--panel);
            box-shadow: 0 1px 2px rgba(16,24,40,0.04); border-radius: 6px; overflow: hidden; }}
  td, th {{ border-bottom: 1px solid var(--line); padding: 8px 12px; font-size: 13px; text-align: left; }}
  tr:last-child td {{ border-bottom: none; }}
  th {{ background: #f1f2f5; font-weight: 600; color: #374151; }}
  .headline {{ display:flex; gap: 14px; flex-wrap: wrap; margin-top: 12px; }}
  .card {{
    border: 1px solid var(--line); background: var(--panel); padding: 16px 20px;
    min-width: 180px; flex: 1 1 180px; border-radius: 8px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.05);
  }}
  .card .label {{ font-size:10.5px; color: var(--muted); text-transform:uppercase; letter-spacing:.05em; font-weight:600;}}
  .card .value {{ font-size:26px; font-weight:700; margin-top:6px; color: var(--accent-dark); }}
  .muted {{ color: var(--muted); font-size: 12px; }}
  .chart {{ border: 1px solid var(--line); background: var(--panel); padding: 12px;
             margin-top: 8px; border-radius: 8px; box-shadow: 0 1px 3px rgba(16,24,40,0.05); }}
  .chart-row {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .chart-row .chart {{ flex: 1 1 340px; }}
  .chart svg {{ width: 100%; height: auto; display:block; }}
  .footer-note {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line); }}
  .tabs {{ display:flex; gap:4px; margin-top:20px; border-bottom: 2px solid var(--line); }}
  .tab-btn {{
    background:none; border:none; padding:10px 18px; font-size:12.5px; font-weight:700;
    color: var(--muted); cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-2px;
    text-transform:uppercase; letter-spacing:.04em; font-family:inherit;
  }}
  .tab-btn:hover {{ color: var(--accent-dark); }}
  .tab-btn.active {{ color: var(--accent-dark); border-bottom-color: var(--accent); }}
  .tab-panel {{ display:none; }}
  .tab-panel.active {{ display:block; }}
  .warning-banner {{
    background: #fff4e5; border: 1px solid #f0b429; border-left: 4px solid #f0b429;
    border-radius: 6px; padding: 14px 18px; margin: 16px 0 20px;
  }}
  .info-banner {{
    background: #eff6ff; border: 1px solid #2f6fed; border-left: 4px solid #2f6fed;
    border-radius: 6px; padding: 14px 18px; margin: 16px 0 20px;
  }}
  .info-banner .info-title {{
    font-weight: 700; font-size: 12.5px; text-transform: uppercase; letter-spacing: .04em;
    color: #1e3a8a; margin-bottom: 8px;
  }}
  .info-banner p {{ font-size: 13px; color: #1e3a8a; margin: 0 0 8px; }}
  .info-banner p:last-child {{ margin-bottom: 0; }}
  .warning-banner .warning-title {{
    font-weight: 700; font-size: 12.5px; text-transform: uppercase; letter-spacing: .04em;
    color: #92400e; margin-bottom: 8px;
  }}
  .warning-banner ul {{ margin: 0; padding-left: 18px; }}
  .warning-banner li {{ font-size: 13px; color: #78350f; margin-bottom: 6px; }}
  .warning-banner li:last-child {{ margin-bottom: 0; }}
  .verdict-banner {{
    border-radius: 6px; padding: 14px 18px; margin: 16px 0 20px; border-left: 4px solid;
  }}
  .verdict-banner.verdict-ready {{ background: #ecfdf3; border-color: #12b76a; }}
  .verdict-banner.verdict-marginal {{ background: #fff4e5; border-color: #f0b429; }}
  .verdict-banner.verdict-not-ready {{ background: #fef3f2; border-color: #f04438; }}
  .risk-flags-banner {{
    border-radius: 6px; padding: 14px 18px; margin: 16px 0 20px; border-left: 4px solid #f04438;
    background: #fef3f2;
  }}
  .risk-flags-banner .verdict-title {{
    font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: .04em;
    margin-bottom: 8px; color: #b42318;
  }}
  .risk-flags-banner ul {{ margin: 0; padding-left: 18px; }}
  .risk-flags-banner li {{ font-size: 13px; margin-bottom: 4px; }}
  .verdict-banner .verdict-title {{
    font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: .04em;
    margin-bottom: 8px;
  }}
  .verdict-banner ul {{ margin: 0; padding-left: 18px; }}
  .verdict-banner li {{ font-size: 13px; margin-bottom: 4px; }}
  @media print {{ .masthead {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}
</style>
</head>
<body>
<div class="masthead">
  <img src="data:image/png;base64,{logo_base64}" alt="T58"/>
  <div class="titles">
    <h1>Quant Algo Backtester — Strategy Report</h1>
    <div class="sub">Precision-tested. Falsification-checked. No shortcuts.</div>
  </div>
</div>
<div class="content">
<p class="meta">Generated {generated_at} &middot; Strategy: <b>{strategy_name}</b> ({source_type}) &middot;
Instrument: {instrument} &middot; Timeframe: {timeframe} &middot; Period: {period_start} → {period_end}</p>

{warnings_section}

{headline_warnings_section}

{verdict_section}

<div class="tabs">
  <button class="tab-btn active" data-tab="overview" type="button">Overview</button>
  <button class="tab-btn" data-tab="tradechart" type="button">Trade Visualization</button>
</div>

<div class="tab-panel active" id="tab-overview">
<h2>Backtest Configuration</h2>
<p class="muted">The exact execution/risk assumptions used to produce every dollar figure below. Recorded here so this report can be reproduced or audited later.</p>
{risk_config_table}

{risk_reconciliation_section}

<h2>The Number That Matters Most</h2>
<div class="headline">
  <div class="card"><div class="label">Evaluation Pass Probability</div><div class="value">{eval_pass:.1f}%</div></div>
  <div class="card"><div class="label">First Payout Probability</div><div class="value">{first_payout:.1f}%</div></div>
  <div class="card"><div class="label">Failure Before Payout</div><div class="value">{failure_before_payout:.1f}%</div></div>
  <div class="card"><div class="label">Median Days to Payout</div><div class="value">{median_days_payout}</div></div>
  <div class="card"><div class="label">Expected Payout</div><div class="value">${expected_payout:,.0f}</div></div>
  <div class="card"><div class="label">Risk of Ruin</div><div class="value">{risk_of_ruin:.1f}%</div></div>
</div>
<p class="muted">"Risk of Ruin" is the probability that max drawdown is breached at <b>some point</b> across the full simulated path -- including after passing the evaluation and collecting payou[...]

{reset_chain_headline_section}

{final_parameters_section}

<h2>Historical Backtest Statistics</h2>
<p class="muted">Computed over the full, uninterrupted trade sequence with prop-firm rules (daily loss limit, max drawdown, etc.) <b>not</b> enforced. Compare against "Prop-Firm Single-Run Result[...]
{reset_chain_banner}
{backtest_table}

<h2>Concentration Check</h2>
<p class="muted">Is one lucky trade or one lucky day carrying the whole result? A repeatable edge shouldn't evaporate when its single best outcome is removed.</p>
{concentration_table}

<h2>Cost Ladder</h2>
<p class="muted">The same trade sequence above, re-costed at increasing added round-turn friction. A real edge should degrade gracefully as costs rise; an edge that only exists at 0% added cost i[...]
{cost_ladder_table}

<h2>Equity Curve (Historical Backtest)</h2>
<div class="chart">{equity_chart}</div>

{holdout_section}

<h2>Prop-Firm Rules Used</h2>
{rules_table}

<h2>Prop-Firm Single-Run Result (Historical Sequence)</h2>
<p class="muted">Same trades as above, but the account stops the instant a prop-firm rule is breached (this run may therefore reflect fewer effective trading days than the historical stats above)[...]
{single_run_table}

<h2>Monte Carlo Simulation ({n_sims:,} simulated accounts)</h2>
<p class="muted">{mc_methodology_note}</p>
{monte_carlo_table}

<div class="chart-row">
  <div class="chart">{return_chart}</div>
  <div class="chart">{drawdown_chart}</div>
</div>

</div><!-- /tab-overview -->

<div class="tab-panel" id="tab-tradechart">
<h2>Interactive Trade Chart</h2>
{trade_chart_html}
</div><!-- /tab-tradechart -->

<p class="footer-note muted">Report generated by T58 Trading — Quant Algo Backtester. All figures are simulated estimates based on historical data and resampling; past performance and simulated[...]
</div>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<script>
(function() {{
  var tradeChartRendered = false;

  function renderTradeChartIfNeeded() {{
    var el = document.getElementById("t58-trade-chart");
    if (!el || typeof Plotly === "undefined") return;
    if (tradeChartRendered) {{
      Plotly.Plots.resize(el);
      return;
    }}
    var payload = window.__t58TradeChartPayload;
    if (!payload) return;
    Plotly.newPlot(el, payload.data, payload.layout, {{responsive: true, displaylogo: false, scrollZoom: true}});
    tradeChartRendered = true;
  }}

  document.querySelectorAll(".tab-btn").forEach(function(btn) {{
    btn.addEventListener("click", function() {{
      document.querySelectorAll(".tab-btn").forEach(function(b) {{ b.classList.remove("active"); }});
      document.querySelectorAll(".tab-panel").forEach(function(p) {{ p.classList.remove("active"); }});
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "tradechart") {{
        renderTradeChartIfNeeded();
      }}
    }});
  }});
}})();
</script>
</body>
</html>
"""


def _dict_to_table(d: dict) -> str:
    rows = "\n".join(
        f"<tr><td>{k}</td><td>{v:,.4f}</td></tr>" if isinstance(v, float)
        else f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in d.items()
    )
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


def _cost_ladder_table(ladder: list[dict]) -> str:
    if not ladder:
        return "<p>No trades to re-cost.</p>"
    header = "<tr><th>Added round-turn cost</th><th>Net Profit</th><th>Profit Factor</th><th>Win Rate</th></tr>"
    rows = []
    for rung in ladder:
        pf = rung["profit_factor"]
        pf_str = "∞" if pf == float("inf") else f"{pf:,.2f}"
        rows.append(
            f"<tr><td>+{rung['extra_cost_pct_per_trade']:.2f}%</td>"
            f"<td>${rung['net_profit']:,.2f}</td>"
            f"<td>{pf_str}</td>"
            f"<td>{rung['win_rate']:.1f}%</td></tr>"
        )
    return f"<table>{header}{''.join(rows)}</table>"


def _verdict_section(verdict: str | None, verdict_reasons: list[str] | None) -> str:
    """Renders the Full Pipeline's READY/MARGINAL/NOT READY verdict (and
    the reasons behind it) prominently in the saved report -- this used to
    only ever reach the live run console and the return value handed back
    to the UI, so the report itself (the thing people actually keep and
    share) had no way to say whether the strategy in front of you passed.
    Renders nothing for reports that never had a verdict (every non-Full-
    Pipeline report), so this is fully backward compatible."""
    if not verdict:
        return ""
    css_class = {"READY": "verdict-ready", "MARGINAL": "verdict-marginal"}.get(verdict, "verdict-not-ready")
    reasons_html = "".join(f"<li>{r}</li>" for r in (verdict_reasons or []))
    reasons_block = f"<ul>{reasons_html}</ul>" if reasons_html else ""
    return (
        f'<div class="verdict-banner {css_class}">'
        f'<div class="verdict-title">Full Pipeline Verdict: {verdict}</div>'
        f"{reasons_block}"
        "</div>"
    )


def _final_parameters_section(final_parameters: dict[str, str] | None, baseline_parameters: dict[str, str] | None = None) -> str:
    """Renders the exact parameter values the Full Pipeline's search
    settled on (SL/TP pips, indicator periods, etc.) next to the metrics
    that show whether that specific configuration passes -- without this,
    the report showed final performance numbers with no way to see what
    configuration actually produced them.

    When `baseline_parameters` is also given (2026-09-24 PARAMETER-
    FIDELITY FIX -- see app.orchestration.full_pipeline._finish's own
    comment for the confusion this closes), renders a Baseline vs Final
    comparison instead of Final alone, with changed rows visibly flagged,
    so a strategy whose name/label describes one set of parameters but
    was actually backtested against GA-mutated ones can't be mistaken for
    "the same strategy" elsewhere (e.g. a comparison run in another tool)
    without that being obvious from the report itself."""
    if not final_parameters:
        return ""
    if not baseline_parameters:
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in final_parameters.items())
        return (
            "<h2>Final Parameters (Full Pipeline Search Result)</h2>"
            '<p class="muted">The exact tunable values this report\'s numbers were produced with, after the walk-forward-aware search.</p>'
            f"<table><tr><th>Parameter</th><th>Value</th></tr>{rows}</table>"
        )
    any_changed = False
    rows_parts = []
    for k, final_v in final_parameters.items():
        base_v = baseline_parameters.get(k)
        changed = base_v is not None and base_v != final_v
        any_changed = any_changed or changed
        row_class = ' class="param-changed"' if changed else ""
        rows_parts.append(f"<tr{row_class}><td>{k}</td><td>{base_v if base_v is not None else '–'}</td><td>{final_v}</td></tr>")
    rows = "".join(rows_parts)
    banner = (
        '<p class="muted param-changed-banner"><strong>The GA search changed one or more parameters '
        "from what was originally supplied</strong> -- this report's backtest ran against the FINAL "
        "column below, not the Baseline column. If you're comparing this result against a backtest run "
        "elsewhere (another tool, a manual re-check) using the strategy's ORIGINAL/supplied parameters, "
        "you are comparing two different rules, not the same one at a different account size.</p>"
        if any_changed else
        '<p class="muted">The GA search did not move any parameter away from what was originally supplied.</p>'
    )
    return (
        "<h2>Final Parameters (Full Pipeline Search Result)</h2>"
        '<p class="muted">The exact tunable values this report\'s numbers were produced with, after the walk-forward-aware search, next to what was originally supplied.</p>'
        f"{banner}"
        f"<table><tr><th>Parameter</th><th>Baseline (supplied)</th><th>Final (backtested)</th></tr>{rows}</table>"
    )


def _warnings_section(warnings: list[str] | None) -> str:
    """Renders execution-integrity warnings (fallback stops, pip-size/
    instrument mismatches, gap-through stop fills) as a prominent banner
    right under the report's header -- empty string (nothing rendered) when
    there are none, so a clean run's report is unchanged."""
    if not warnings:
        return ""
    items = "".join(f"<li>{w}</li>" for w in warnings)
    return (
        '<div class="warning-banner">'
        '<div class="warning-title">⚠ Execution warnings -- read before trusting the numbers below</div>'
        f"<ul>{items}</ul>"
        "</div>"
    )


def _concentration_table(c: dict) -> str:
    if not c:
        return "<p>No trades to check.</p>"
    rows = (
        f"<tr><td>Best single trade</td><td>${c['best_trade_pnl']:,.2f}</td>"
        f"<td>{c['best_trade_pct_of_gross_profit']:.1f}% of gross profit</td>"
        f"<td>Net profit excl. it: ${c['net_profit_excluding_best_trade']:,.2f}</td></tr>"
        f"<tr><td>Best single day</td><td>${c['best_day_pnl']:,.2f}</td>"
        f"<td>{c['best_day_pct_of_gross_profit']:.1f}% of gross profit</td>"
        f"<td>Net profit excl. it: ${c['net_profit_excluding_best_day']:,.2f}</td></tr>"
    )
    return (
        "<table><tr><th></th><th>P&amp;L</th><th>Share of gross profit</th>"
        f"<th>Result if removed</th></tr>{rows}</table>"
    )


def _holdout_section(holdout: dict | None) -> str:
    if not holdout:
        return ""
    in_s = holdout.get("in_sample_statistics") or {}
    out_s = holdout.get("holdout_statistics") or {}
    frac = holdout.get("holdout_frac", 0.0) * 100
    in_period = holdout.get("in_sample_period", (None, None))
    out_period = holdout.get("holdout_period", (None, None))

    def row(label, key, fmt="{:,.2f}"):
        a = in_s.get(key)
        b = out_s.get(key)
        a_str = fmt.format(a) if isinstance(a, (int, float)) else "n/a"
        b_str = fmt.format(b) if isinstance(b, (int, float)) else "n/a"
        return f"<tr><td>{label}</td><td>{a_str}</td><td>{b_str}</td></tr>"

    table = (
        "<table><tr><th>Metric</th><th>In-Sample</th><th>Holdout (never re-tuned on)</th></tr>"
        + row("Trades", "total_trades", "{:,.0f}")
        + row("Net profit", "net_profit", "${:,.2f}")
        + row("Profit factor", "profit_factor", "{:,.2f}")
        + row("Win rate (%)", "win_rate", "{:.1f}")
        + row("Max drawdown (%)", "max_drawdown_pct", "{:.2f}")
        + row("Sharpe ratio", "sharpe_ratio", "{:.2f}")
        + "</table>"
    )
    return f"""<h2>Out-of-Sample Holdout Check</h2>
<p class="muted">The final {frac:.0f}% of bars chronologically ({out_period[0]} &rarr; {out_period[1]}) were withheld and run through the exact same strategy and risk settings as the earlier {100 - frac:.0f}% ({in_period[0]} &rarr; {in_period[1]}).</p>
{table}"""


def _downsample(values: list[float], max_points: int = 400) -> list[float]:
    if len(values) <= max_points:
        return values
    step = len(values) / max_points
    return [values[int(i * step)] for i in range(max_points)]


def _risk_config_table(risk_config: dict | None) -> str:
    if not risk_config:
        return "<p>No risk configuration was recorded for this run.</p>"
    return _dict_to_table(risk_config)


def _risk_reconciliation_section(stats: dict) -> str:
    """"How much you told the system you're willing to risk" vs. "how much
    it actually risked" -- see app.backtest.execution's Trade.
    intended_risk_dollars / app.backtest.statistics.compute_risk_
    reconciliation. Returns "" when there's nothing to reconcile (no
    trade in the run carried an intended_risk_dollars figure at all,
    e.g. every trade was skipped or the run had zero trades)."""
    intended = stats.get("avg_intended_risk_dollars", 0.0)
    actual = stats.get("avg_actual_stop_risk_dollars", 0.0)
    if not intended and not actual:
        return ""
    realized_loss = stats.get("avg_realized_loss_on_losers", 0.0)
    pct_capped = stats.get("pct_trades_position_capped", 0.0)
    pct_overshoot = stats.get("pct_trades_risk_overshoot", 0.0)
    table = _dict_to_table({
        "Configured target risk per trade (avg)": intended,
        "Actual risk at stop, given size taken (avg)": actual,
        "Realized loss on losing trades (avg)": realized_loss,
        "% of trades sized below the configured target (cap/throttle engaged)": pct_capped,
        "% of trades whose realized loss exceeded their own stop risk (gap-through)": pct_overshoot,
    })
    cap_note = ""
    if pct_capped >= 5.0:
        cap_note = (
            f"<p class='muted'>{pct_capped:.0f}% of trades risked meaningfully LESS than the "
            "configured target -- a max-position-size cap or adaptive-risk throttle is engaging "
            "(the strategy's own stop distance combined with your risk % is calling for a "
            "position the cap won't allow). This is why the account can realize a smaller "
            "average loss than \"risk value % x account size\" alone would suggest.</p>"
        )
    overshoot_note = ""
    if pct_overshoot >= 5.0:
        overshoot_note = (
            f"<p class='muted'>{pct_overshoot:.0f}% of LOSING trades realized MORE loss than "
            "their own stop was sized for -- almost always a gap-through fill (price crossed the "
            "resting stop within one bar), not a bug. See the execution warnings above if this "
            "share is large.</p>"
        )
    return f"""<h2>Risk Reconciliation</h2>
<p class="muted">Reconciles the risk % you configured against what actually happened to it on a trade-by-trade basis -- two different, independent gaps to watch for: sizing that comes in UNDER yo[...]
{table}
{cap_note}{overshoot_note}"""


def _reset_chain_banner(stats: dict) -> str:
    """Prepended to the Historical Backtest Statistics table whenever this
    run used reset_on_breach and at least one reset actually occurred
    (stats['account_reset_count'] > 0) -- see BacktestStatistics.
    is_reset_chain's docstring for the full reasoning. `net_profit` inside
    the table immediately below this banner is the raw, unqualified
    cumulative figure across every simulated account in the chain; this
    banner is what tells the reader that BEFORE they get there, plus
    gives them the one number (final_segment_net_profit) that describes
    just the currently-standing account. Returns "" for any run with no
    resets -- the default, far more common case."""
    reset_count = stats.get("account_reset_count", 0)
    if not reset_count:
        return ""
    final_pnl = stats.get("final_segment_net_profit", 0.0)
    final_trades = stats.get("final_segment_trade_count", 0)
    net_profit = stats.get("net_profit", 0.0)
    return f"""<div class="info-banner">
<div class="info-title">Reset-on-breach was used -- read "Net Profit" below carefully</div>
<p>This backtest mechanically "bought a new account" {reset_count} time(s) whenever the configured
account-survivability floor was crossed (see RiskConfig.reset_on_breach) -- every trade after each
reset belongs to a DIFFERENT simulated account than the one before it, not the same account taking
a deeper loss.</p>
<p><b>"Net Profit" in the table below (${net_profit:,.2f}) is CUMULATIVE P&amp;L pooled across all
{reset_count + 1} of those simulated accounts</b> -- it is not, and should not be read as, one
account's result. The account still standing at the end of this run made
<b>${final_pnl:,.2f}</b> of its own, over its own {final_trades} trades.</p>
</div>"""


def _reset_chain_headline_section(mc: dict, is_reset_chain: bool) -> str:
    """A second row of headline cards, shown ONLY when this run's Monte
    Carlo used reset_on_breach, giving the true per-account-attempt
    pass/payout rate (MonteCarloResult.per_attempt_pass_probability et
    al.) right next to the chain-level "Evaluation Pass Probability" card
    above -- which, under reset_on_breach, answers a different question
    ("did at least one attempt anywhere in the chain pass") that can look
    far stronger than any single attempt's real odds once a chain runs
    many attempts (see mean_attempts_per_path). Returns "" for the
    default, non-reset case -- the existing headline already answers the
    single-attempt question correctly there."""
    if not is_reset_chain or not mc.get("reset_on_breach"):
        return ""
    per_attempt_pass = mc.get("per_attempt_pass_probability", 0.0)
    per_attempt_payout = mc.get("per_attempt_payout_probability", 0.0)
    mean_attempts = mc.get("mean_attempts_per_path", 0.0)
    total_attempts = mc.get("total_independent_attempts", 0)
    return f"""<div class="info-banner">
<div class="info-title">Reset-on-breach Monte Carlo -- the cards above answer a different question</div>
<p>With reset_on_breach on, each simulated path mechanically "rebought" an average of
{mean_attempts:,.1f} times (see "Attempts per path" in the Monte Carlo table below) before the
simulated history ran out. "Evaluation Pass Probability" / "First Payout Probability" above mean
<b>"did at least one attempt anywhere in that chain eventually pass / get paid"</b> -- with hundreds
of attempts per chain, that can look strong even when any ONE account's real odds are modest.</p>
<div class="headline">
  <div class="card"><div class="label">Per-Attempt Pass Probability</div><div class="value">{per_attempt_pass:.1f}%</div></div>
  <div class="card"><div class="label">Per-Attempt Payout Probability</div><div class="value">{per_attempt_payout:.1f}%</div></div>
</div>
<p>These two are pooled across all {total_attempts:,} independent account attempts this Monte Carlo
run represents, and directly answer <b>"if I buy ONE account, what's the probability it passes /
gets paid"</b> -- the number to trust for a single real-money account decision.</p>
</div>"""


def export_html(
    report: dict,
    path: str | Path,
    backtest_result: BacktestResult | None = None,
    price_df: pd.DataFrame | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mc = report["monte_carlo"]
    single = report["prop_firm_single_run"]

    if backtest_result is not None:
        trade_chart_html = build_trade_chart_html(
            price_df=price_df,
            trades=backtest_result.trades,
            equity_curve=backtest_result.equity_curve,
            instrument=report["strategy"]["instrument"],
        )
    else:
        trade_chart_html = '<p class="muted">No backtest result was available to plot.</p>'

    if backtest_result is not None and len(backtest_result.equity_curve):
        equity_values = _downsample(backtest_result.equity_curve["equity"].tolist())
        equity_chart = svg_line_chart(
            equity_values, title="Account Equity Over the Historical Backtest", y_label="Equity ($)",
        )
    else:
        equity_chart = "<p>No trades were generated, so no equity curve is available.</p>"

    return_chart = svg_histogram(
        mc.get("return_distribution", []),
        title="Monte Carlo: Distribution of Simulated Account Returns",
        x_label="Return (%)",
        color="#2f6fed",
        markers={
            "P5": mc["return_percentiles"].get(5),
            "Median": mc["return_percentiles"].get(50),
            "P95": mc["return_percentiles"].get(95),
        },
    )
    drawdown_chart = svg_histogram(
        mc.get("drawdown_distribution", []),
        title="Monte Carlo: Distribution of Simulated Max Drawdown",
        x_label="Max Drawdown (%)",
        color="#F05B63",
        markers={
            "Median": mc["drawdown_percentiles"].get(50),
            "P95": mc["drawdown_percentiles"].get(95),
        },
    )

    html = _HTML_TEMPLATE.format(
        logo_base64=T58_LOGO_BASE64,
        strategy_name=report["strategy"]["name"],
        source_type=report["strategy"]["source_type"],
        instrument=report["strategy"]["instrument"],
        timeframe=report["strategy"]["timeframe"],
        period_start=report["strategy"]["backtest_period_start"],
        period_end=report["strategy"]["backtest_period_end"],
        generated_at=report["generated_at"],
        warnings_section=_warnings_section(report.get("execution_warnings")),
        verdict_section=_verdict_section(report.get("verdict"), report.get("verdict_reasons")),
        headline_warnings_section=_headline_warnings_banner(report.get("headline_warnings") or []),
        final_parameters_section=_final_parameters_section(report.get("final_parameters"), report.get("baseline_parameters")),
        risk_config_table=_risk_config_table(report.get("risk_config")),
        risk_reconciliation_section=_risk_reconciliation_section(report["historical_backtest"]["statistics"]),
        eval_pass=mc["evaluation_pass_probability"],
        first_payout=mc["first_payout_probability"],
        failure_before_payout=mc["failure_before_payout_probability"],
        median_days_payout=mc["median_days_to_first_payout"] if mc["median_days_to_first_payout"] is not None else "N/A",
        expected_payout=mc["expected_payout"],
        risk_of_ruin=mc["risk_of_ruin_pct"],
        reset_chain_headline_section=_reset_chain_headline_section(
            mc, report["historical_backtest"]["statistics"].get("is_reset_chain", False)
        ),
        reset_chain_banner=_reset_chain_banner(report["historical_backtest"]["statistics"]),
        backtest_table=_dict_to_table(report["historical_backtest"]["statistics"]),
        concentration_table=_concentration_table(report.get("concentration_check", {})),
        cost_ladder_table=_cost_ladder_table(report.get("cost_ladder", [])),
        equity_chart=equity_chart,
        holdout_section=_holdout_section(report.get("holdout_comparison")),
        rules_table=_dict_to_table(report["prop_firm_rules"]),
        single_run_table=_dict_to_table(single),
        monte_carlo_table=_dict_to_table({
            k: v for k, v in mc.items()
            if not isinstance(v, (dict, list)) and k != "methodology_note"
        }),
        mc_methodology_note=mc.get("methodology_note", ""),
        return_chart=return_chart,
        drawdown_chart=drawdown_chart,
        n_sims=mc["n_simulations"],
        trade_chart_html=trade_chart_html,
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    # Native PDF export button, added alongside the browser's own
    # print-to-PDF (still works via Ctrl/Cmd+P as before) -- see
    # app/reports/pdf_export.py and app/web/extra_routes.py's /export/pdf
    # route. Injected as a small standalone block rather than a
    # _HTML_TEMPLATE placeholder so this whole feature is one self-contained,
    # easily-reviewed diff instead of touching the (large) template string.
    pdf_button_html = f"""
<div style="position:fixed;bottom:18px;right:18px;z-index:999;">
  <form method="post" action="/export/pdf" style="margin:0;">
    <input type="hidden" name="report_json" value='{json.dumps(report).replace("'", "&#39;")}'>
    <button type="submit" style="padding:10px 16px;border-radius:8px;border:1px solid #2f6fed;
      background:#2f6fed;color:#fff;font-size:13px;font-weight:600;cursor:pointer;
      box-shadow:0 2px 10px rgba(0,0,0,.25);">&#11015; Export PDF</button>
  </form>
</div>
"""
    with open(path, "r+", encoding="utf-8") as f:
        content = f.read()
        content = content.replace("</body>", pdf_button_html + "</body>") if "</body>" in content else content + pdf_button_html
        f.seek(0)
        f.write(content)
        f.truncate()
    return path


def generate_full_report(
    output_dir: str | Path,
    strategy_name: str,
    strategy_source_type: str,
    instrument: str,
    timeframe: str,
    backtest_period: tuple[str, str],
    backtest_result: BacktestResult,
    prop_rules: PropRules,
    prop_single_run: AccountSimResult,
    monte_carlo_result: MonteCarloResult,
    basename: str = "report",
    holdout_comparison: dict | None = None,
    risk_config: RiskConfig | None = None,
    price_df: pd.DataFrame | None = None,
    verdict: str | None = None,
    verdict_reasons: list[str] | None = None,
    final_parameters: dict[str, str] | None = None,
    baseline_parameters: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Builds the report dict and writes JSON + summary CSV + trades CSV + HTML to output_dir.

    price_df, if provided, is the standardized OHLCV DataFrame the backtest
    was run on (see app.data.importer) -- it powers the interactive "Trade
    Visualization" tab in the HTML report. Omitting it still produces a
    complete report; that tab just falls back to a short explanatory note
    instead of a chart.

    verdict / verdict_reasons / final_parameters / baseline_parameters are
    only ever populated by the Full Pipeline (see
    app.orchestration.full_pipeline) -- every other caller omits them and
    gets exactly the same report as before.
    """
    report = build_report(
        strategy_name, strategy_source_type, instrument, timeframe, backtest_period,
        backtest_result, prop_rules, prop_single_run, monte_carlo_result,
        holdout_comparison=holdout_comparison, risk_config=risk_config,
        verdict=verdict, verdict_reasons=verdict_reasons, final_parameters=final_parameters,
        baseline_parameters=baseline_parameters,
    )
    output_dir = Path(output_dir)
    paths = {
        "json": export_json(report, output_dir / f"{basename}.json"),
        "summary_csv": export_summary_csv(report, output_dir / f"{basename}_summary.csv"),
        "trades_csv": export_trades_csv(backtest_result, output_dir / f"{basename}_trades.csv"),
        "html": export_html(report, output_dir / f"{basename}.html", backtest_result=backtest_result, price_df=price_df),
    }
    # Best-effort: powers the Dashboard tab (desktop + web). Never allowed
    # to turn a successful report into a failed run.
    run_history.record_run(report, paths, backtest_result=backtest_result)
    return paths
