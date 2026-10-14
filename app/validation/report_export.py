"""One saved, downloadable report per Validate run.

Before this module the Validate page showed its pass/fail cards and then threw them away: the report
lived in an in-memory job for a few hours and nothing could be downloaded. This module turns a finished
run into ONE consolidated document -- every check that was run, whether it passed, failed or could not
run, WHY (the bar it was judged against, the numbers, the plain-language explanation and the fix steps),
plus the raw numbers behind each score (CPCV paths, PBO, walk-forward folds, regime cells, ...).

It writes three files side by side under one report id:

    <id>.html   self-contained, printable (the one to read / share)
    <id>.json   the complete machine-readable record (also what Auto-Fix reloads its settings from)
    <id>.txt    plain text (the same text the desktop app shows)

Flask-free and pandas-free on purpose, so it is unit-testable in isolation and shared by the web app
and the desktop app.
"""
from __future__ import annotations

import html
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.validation import simple_report

SCHEMA = "t58.validation_report.v1"

# Form keys that must never be written to disk inside a report's settings snapshot.
_SECRET_HINTS = ("key", "token", "secret", "password", "passwd", "credential")

# Settings worth echoing in the report so a reader knows exactly how each check was run.
_SETTING_KEYS = (
    "mc_sims", "validation_metric", "cpcv_groups", "cpcv_test_groups", "cpcv_max_paths", "pbo_variants",
    "wf_folds", "wf_window_mode", "wfga_population", "wfga_generations", "fitness_metric",
    "pct_range", "n_steps", "max_params", "robustness_metric", "regime_dim_a", "regime_dim_b",
)

_REPORT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,150}$")

_STATUS_WORD = {"pass": "PASS", "caution": "BORDERLINE", "fail": "FAIL", "error": "DID NOT RUN", "info": "RAN"}


# ---------------------------------------------------------------------------
# JSON-safety + ids
# ---------------------------------------------------------------------------

def clean_json(obj: Any) -> Any:
    """JSON-safe deep copy: numpy scalars -> Python, NaN/inf -> None, tuples/sets -> lists."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_json(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def safe_report_id(report_id: str) -> Optional[str]:
    """The id itself if it is a plain file-safe token, else None (blocks path traversal)."""
    rid = (report_id or "").strip()
    return rid if _REPORT_ID_RE.match(rid) and ".." not in rid else None


def new_report_id(strategy_name: str, when: Optional[float] = None) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", str(strategy_name or "strategy")).strip("-")[:48] or "strategy"
    stamp = datetime.fromtimestamp(when if when is not None else time.time()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{stem}"


def sanitize_form(form: Any) -> dict:
    """A plain dict snapshot of a posted form -- secrets dropped, values stringified, file uploads
    skipped. This is what lets Auto-Fix re-run the very same settings later (even after a restart)."""
    out: dict = {}
    try:
        items = list(form.items())
    except Exception:  # noqa: BLE001
        items = []
    for k, v in items:
        key = str(k)
        if any(h in key.lower() for h in _SECRET_HINTS):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = "" if v is None else str(v)
    return out


# ---------------------------------------------------------------------------
# Building the document
# ---------------------------------------------------------------------------

def build_report_document(
    report: dict, *, raw: Optional[dict] = None, baseline: Optional[dict] = None,
    meta: Optional[dict] = None, form: Optional[Any] = None, requested_checks: Optional[list] = None,
    simulation_text: str = "", report_id: Optional[str] = None, generated_at: Optional[float] = None,
    auto_fix: Optional[dict] = None,
) -> dict:
    """The complete, JSON-safe record of one Validate run.

    report   -- simple_report.build_report(...) output (items with status/headline/guidance).
    raw      -- {check: full summary dict} straight from the dispatcher (all the numbers).
    meta     -- {strategy, library_type, library_file, dataset, timeframe, prop_preset, ...}.
    form     -- the posted form (sanitised into a settings snapshot).
    requested_checks -- check keys the person ticked, so checks that never produced a result are listed.
    """
    raw = raw or {}
    meta = dict(meta or {})
    ts = generated_at if generated_at is not None else time.time()
    rid = safe_report_id(report_id or "") or new_report_id(meta.get("strategy") or report.get("strategy") or "strategy", ts)
    snapshot = sanitize_form(form) if form is not None else {}

    checks: list[dict] = []
    for it in report.get("items", []):
        g = it.get("guidance") or {}
        status = it.get("status", "info")
        checks.append({
            "check": it.get("check"),
            "label": it.get("label"),
            "status": status,
            "verdict": _STATUS_WORD.get(status, status.upper()),
            "headline": it.get("headline", ""),
            "detail": it.get("detail", ""),
            "bar": simple_report.CHECK_BARS.get(it.get("check", "")) or g.get("bar") or "",
            "metrics": [{"label": m.get("label"), "value": m.get("value")} for m in (it.get("metrics") or [])],
            "why_not_pass": g.get("why", "") if status in ("fail", "caution", "error") else "",
            "steps": list(g.get("steps") or []) if status in ("fail", "caution", "error") else [],
            "fix_kind": g.get("fix_kind"),
            "raw": raw.get(it.get("check")),
        })

    ran = {c["check"] for c in checks}
    not_run = [simple_report.CHECK_LABELS.get(c, c) for c in (requested_checks or []) if c not in ran]

    doc = {
        "schema": SCHEMA,
        "report_id": rid,
        "generated_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        "strategy": {
            "name": meta.get("strategy") or report.get("strategy") or "",
            "library_type": meta.get("library_type") or report.get("library_type") or "",
            "library_file": meta.get("library_file") or "",
        },
        "dataset": meta.get("dataset") or "",
        "timeframe": meta.get("timeframe") or "native",
        "prop_preset": meta.get("prop_preset") or "",
        "settings": {
            "checks_requested": list(requested_checks or []),
            "parameters": {k: snapshot[k] for k in _SETTING_KEYS if k in snapshot},
            "simulation_text": simulation_text or "",
        },
        "baseline": baseline or None,
        "overall": {
            "key": report.get("overall"), "label": report.get("overall_label"),
            "summary": report.get("summary_sentence"), "counts": report.get("counts") or {},
        },
        "checks": checks,
        "checks_not_run": not_run,
        "gating_missing": list(report.get("gating_missing_labels") or []),
        "next_step": {k: (report.get("next_step") or {}).get(k) for k in ("title", "why", "label", "key")},
        "form_snapshot": snapshot,
    }
    if auto_fix:
        doc["auto_fix"] = auto_fix
    return clean_json(doc)


# ---------------------------------------------------------------------------
# Rendering: HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14181f;--mute:#5d6675;--line:#dde1e8;--pass:#0a7d4f;--warn:#a86400;--fail:#c0262d;--grey:#6b7280}
*{box-sizing:border-box}body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:28px}
.wrap{max-width:980px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:26px 0 10px}h3{font-size:14px;margin:0}
.sub{color:var(--mute);font-size:12.5px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0}
.pill{display:inline-block;border-radius:999px;padding:2px 10px;font-size:11.5px;font-weight:700;color:#fff}
.pill.pass{background:var(--pass)}.pill.caution{background:var(--warn)}.pill.fail{background:var(--fail)}.pill.error,.pill.info{background:var(--grey)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600;font-size:12px}.num{text-align:right;font-variant-numeric:tabular-nums}
.check{border-left:5px solid var(--grey)}.check.pass{border-left-color:var(--pass)}.check.caution{border-left-color:var(--warn)}.check.fail{border-left-color:var(--fail)}
.head{display:flex;justify-content:space-between;align-items:center;gap:10px}.bar{color:var(--mute);font-size:12.5px;margin-top:6px}
.why{background:#fff7ed;border:1px solid #f3d9b1;border-radius:8px;padding:10px 12px;margin-top:10px;font-size:13px}
ol{margin:6px 0 0 18px;padding:0}details{margin-top:10px}summary{cursor:pointer;color:var(--mute);font-size:12.5px}
pre{background:#f1f3f6;border-radius:6px;padding:10px;overflow:auto;font-size:11.5px}.kv td:first-child{color:var(--mute);width:34%}
.overall{font-size:15px}.small{font-size:12px;color:var(--mute)}@media print{body{background:#fff;padding:0}.card{break-inside:avoid}}
"""


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.2f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return _e(v)


def _render_value(v: Any, depth: int = 0) -> str:
    """Generic, escaped rendering of a nested summary: dicts -> key/value tables, lists of dicts ->
    real tables, everything else inline. Deep enough for CPCV/WFO/regime numbers, never fancy."""
    if isinstance(v, dict):
        if not v:
            return '<span class="small">(none)</span>'
        rows = "".join(f"<tr><td>{_e(k)}</td><td>{_render_value(val, depth + 1)}</td></tr>" for k, val in v.items())
        return f'<table class="kv">{rows}</table>'
    if isinstance(v, list):
        if not v:
            return '<span class="small">(none)</span>'
        if all(isinstance(x, dict) for x in v):
            cols: list[str] = []
            for x in v:
                for k in x:
                    if k not in cols:
                        cols.append(k)
            head = "".join(f"<th>{_e(c)}</th>" for c in cols)
            body = "".join("<tr>" + "".join(f"<td>{_render_value(x.get(c), depth + 1)}</td>" for c in cols) + "</tr>" for x in v)
            return f"<table><tr>{head}</tr>{body}</table>"
        return ", ".join(_fmt(x) for x in v)
    return _fmt(v)


def _render_autofix_html(af: Optional[dict]) -> str:
    """The 'what Auto-Fix changed and whether re-running the checks confirmed it' section."""
    if not af:
        return ""
    verdict = "Kept" if af.get("accepted") else "Not recommended"
    stages = ""
    for st in af.get("stages") or []:
        reasons = "".join(f"<li>{_e(r)}</li>" for r in st.get("reasons") or [])
        stages += (f'<div class="card"><div class="head"><h3>{_e(st.get("label"))}</h3>'
                   f'<span class="pill {"pass" if st.get("accepted") else "caution"}">{"KEPT" if st.get("accepted") else "NOT KEPT"}</span></div>'
                   f'<div style="margin-top:6px">{_e(st.get("headline"))}</div>'
                   + (f"<ol>{reasons}</ol>" if reasons else "")
                   + f'<details><summary>Numbers behind this</summary>{_render_value(st.get("details"))}</details></div>')
    before = {r.get("check"): r for r in af.get("before") or []}
    rows = ""
    for a in af.get("after") or []:
        b = before.get(a.get("check")) or {}
        rows += (f'<tr><td><b>{_e(a.get("label"))}</b></td><td>{_e(_STATUS_WORD.get(b.get("status"), b.get("status") or "—"))}</td>'
                 f'<td>{_e(_STATUS_WORD.get(a.get("status"), a.get("status")))}</td><td>{_e(b.get("headline") or "")}</td><td>{_e(a.get("headline"))}</td></tr>')
    table = ('<table><tr><th>Check</th><th>Before</th><th>After</th><th>Before: what it found</th><th>After: what it found</th></tr>'
             + rows + "</table>") if rows else ""
    notes = "".join(f"<p class='small'>{_e(n)}</p>" for n in af.get("notes") or [])
    src = af.get("source_report_id")
    return (f'<h2>Auto-Fix: what was changed, and was it confirmed</h2><div class="card"><b>{_e(verdict)}.</b> {_e(af.get("headline"))}'
            + (f'<div class="small" style="margin-top:4px">Repaired from the strategy in report {_e(src)}. The original was not modified.</div>' if src else "")
            + notes + "</div>" + stages + (f'<div class="card"><h3>Same checks, before vs after</h3>{table}</div>' if table else ""))


def render_html(doc: dict) -> str:
    st = doc.get("strategy") or {}
    ov = doc.get("overall") or {}
    counts = ov.get("counts") or {}
    checks = doc.get("checks") or []
    base = doc.get("baseline")
    nx = doc.get("next_step") or {}

    rows = "".join(
        f'<tr><td><b>{_e(c["label"])}</b></td><td><span class="pill {_e(c["status"])}">{_e(c["verdict"])}</span></td>'
        f'<td>{_e(c["headline"])}</td><td class="small">{_e(c["bar"])}</td></tr>' for c in checks
    )
    scorecard = (
        '<table><tr><th>Check</th><th>Result</th><th>What it found</th><th>Bar to pass</th></tr>' + rows + "</table>"
        if rows else '<p class="small">No check produced a result.</p>'
    )

    cards = []
    for c in checks:
        metrics = "".join(f'<tr><td>{_e(m["label"])}</td><td>{_e(m["value"])}</td></tr>' for m in c.get("metrics") or [])
        why = ""
        if c.get("why_not_pass"):
            steps = "".join(f"<li>{_e(s)}</li>" for s in c.get("steps") or [])
            title = "Why it could not run" if c["status"] == "error" else "Why this is not a pass"
            why = (f'<div class="why"><b>{title}.</b> {_e(c["why_not_pass"])}'
                   + (f"<div style='margin-top:6px'><b>What to do</b></div><ol>{steps}</ol>" if steps else "") + "</div>")
        raw = (f"<details><summary>Raw numbers</summary>{_render_value(c['raw'])}</details>" if c.get("raw") else "")
        cards.append(
            f'<div class="card check {_e(c["status"])}"><div class="head"><h3>{_e(c["label"])}</h3>'
            f'<span class="pill {_e(c["status"])}">{_e(c["verdict"])}</span></div>'
            f'<div style="margin-top:6px"><b>{_e(c["headline"])}</b></div>'
            + (f'<div class="small">{_e(c["detail"])}</div>' if c.get("detail") else "")
            + f'<div class="bar"><b>Bar to pass:</b> {_e(c["bar"])}</div>'
            + (f'<table style="margin-top:8px">{metrics}</table>' if metrics else "")
            + why + raw + "</div>"
        )

    baseline_html = ""
    if base:
        baseline_html = '<h2>Baseline backtest</h2><div class="card">' + _render_value(
            {k: base.get(k) for k in ("trades", "net_profit", "win_rate", "max_drawdown_pct", "dataset") if k in base}) + "</div>"

    settings = doc.get("settings") or {}
    params = settings.get("parameters") or {}
    settings_html = (
        '<h2>How it was run</h2><div class="card">'
        + _render_value({
            "Dataset": doc.get("dataset"), "Timeframe": doc.get("timeframe"),
            "Prop-firm preset": doc.get("prop_preset") or "(typed rules)",
            "Checks requested": ", ".join(simple_report.CHECK_LABELS.get(c, c) for c in settings.get("checks_requested") or []),
            **{k: v for k, v in params.items()},
        })
        + (f'<details><summary>Simulation settings</summary><pre>{_e(settings.get("simulation_text"))}</pre></details>'
           if settings.get("simulation_text") else "") + "</div>"
    )

    not_run = ""
    if doc.get("checks_not_run"):
        not_run = '<div class="card"><b>Requested but produced no result:</b> ' + _e(", ".join(doc["checks_not_run"])) + "</div>"
    missing = ""
    if doc.get("gating_missing"):
        missing = f'<p class="small">Core checks still missing for promotion: {_e(", ".join(doc["gating_missing"]))}.</p>'

    autofix_html = _render_autofix_html(doc.get("auto_fix"))
    count_line = " · ".join(f"{counts.get(k, 0)} {w}" for k, w in (("pass", "passed"), ("caution", "borderline"), ("fail", "failed"), ("error", "did not run")))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Validation report — {_e(st.get("name"))}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Validation report — {_e(st.get("name") or "strategy")}</h1>
<div class="sub">T58 Quant Algo Backtester · generated {_e(doc.get("generated_at"))} · report id {_e(doc.get("report_id"))}</div>
<div class="card overall"><b>{_e(ov.get("label"))}</b> — {_e(ov.get("summary"))}<div class="small" style="margin-top:4px">{_e(count_line)}</div>
{('<div style="margin-top:8px"><b>Next step:</b> ' + _e(nx.get("title")) + ' — ' + _e(nx.get("why")) + '</div>') if nx.get("title") else ''}{missing}</div>
{autofix_html}<h2>Scorecard</h2><div class="card">{scorecard}</div>{not_run}
<h2>Every check in detail</h2>{''.join(cards)}
{baseline_html}{settings_html}
<p class="small">Pass/fail thresholds are fixed and stated next to each check. A pass is evidence, not a guarantee: all of these tests use historical data.</p>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Rendering: plain text
# ---------------------------------------------------------------------------

def render_text(doc: dict) -> str:
    st = doc.get("strategy") or {}
    ov = doc.get("overall") or {}
    lines = [
        "=" * 70,
        f"VALIDATION REPORT -- {st.get('name') or 'strategy'}",
        f"{ov.get('label', '')}: {ov.get('summary', '')}",
        f"Generated {doc.get('generated_at')}   (id {doc.get('report_id')})",
        f"Dataset: {doc.get('dataset')}   Timeframe: {doc.get('timeframe')}   Prop preset: {doc.get('prop_preset') or '(typed rules)'}",
        "=" * 70, "",
        "SCORECARD",
    ]
    for c in doc.get("checks", []):
        lines.append(f"  [{c['verdict']:<11}] {c['label']}: {c['headline']}")
    if doc.get("checks_not_run"):
        lines.append("  Requested but produced no result: " + ", ".join(doc["checks_not_run"]))
    lines.append("")
    af = doc.get("auto_fix")
    if af:
        lines += ["AUTO-FIX: " + ("KEPT" if af.get("accepted") else "NOT RECOMMENDED") + " -- " + str(af.get("headline", ""))]
        for st in af.get("stages") or []:
            lines.append(f"  [{'KEPT' if st.get('accepted') else 'not kept'}] {st.get('label')}: {st.get('headline')}")
            for r in st.get("reasons") or []:
                lines.append(f"      - {r}")
        for n in af.get("notes") or []:
            lines.append(f"  NOTE: {n}")
        lines.append("")
    for c in doc.get("checks", []):
        lines.append("-" * 70)
        lines.append(f"{c['label']} -- {c['verdict']}")
        lines.append(f"  {c['headline']}")
        if c.get("detail"):
            lines.append(f"  {c['detail']}")
        lines.append(f"  Bar to pass: {c['bar']}")
        for m in c.get("metrics", []):
            lines.append(f"    - {m['label']}: {m['value']}")
        if c.get("why_not_pass"):
            lines.append(f"  WHY: {c['why_not_pass']}")
            for n, step in enumerate(c.get("steps", []), 1):
                lines.append(f"    {n}. {step}")
        lines.append("")
    base = doc.get("baseline")
    if base:
        lines.append(f"Baseline backtest: {base.get('trades', 0)} trades, net {base.get('net_profit', 0)}, "
                     f"win rate {base.get('win_rate', 0)}, max drawdown {base.get('max_drawdown_pct', 0)}%.")
    nx = doc.get("next_step") or {}
    if nx.get("title"):
        lines += ["", f"NEXT STEP: {nx['title']}", f"  {nx.get('why', '')}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Saving / loading / listing
# ---------------------------------------------------------------------------

def save_report(directory: Path | str, doc: dict) -> dict:
    """Write <id>.json/.html/.txt under `directory`. Returns {format: Path}. Never raises for a single
    format failing to render -- the JSON (the complete record) is written first and always."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    rid = safe_report_id(doc.get("report_id", "")) or new_report_id((doc.get("strategy") or {}).get("name", ""))
    doc["report_id"] = rid
    paths: dict[str, Path] = {}
    jp = d / f"{rid}.json"
    jp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    paths["json"] = jp
    for ext, fn in (("html", render_html), ("txt", render_text)):
        try:
            p = d / f"{rid}.{ext}"
            p.write_text(fn(doc), encoding="utf-8")
            paths[ext] = p
        except Exception:  # noqa: BLE001
            pass
    return paths


def report_path(directory: Path | str, report_id: str, fmt: str = "html") -> Optional[Path]:
    rid = safe_report_id(report_id)
    if rid is None or fmt not in ("html", "json", "txt"):
        return None
    p = Path(directory) / f"{rid}.{fmt}"
    return p if p.is_file() else None


def load_report(directory: Path | str, report_id: str) -> Optional[dict]:
    p = report_path(directory, report_id, "json")
    if p is None:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def list_reports(directory: Path | str, strategy: Optional[str] = None, limit: int = 100) -> list[dict]:
    """Newest-first index of saved reports (reads each JSON header only once)."""
    d = Path(directory)
    if not d.is_dir():
        return []
    rows = []
    for p in sorted(d.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if doc.get("schema") != SCHEMA:
            continue
        name = (doc.get("strategy") or {}).get("name", "")
        if strategy and name != strategy:
            continue
        rows.append({
            "report_id": doc.get("report_id") or p.stem, "strategy": name,
            "generated_at": doc.get("generated_at"), "overall": (doc.get("overall") or {}).get("label"),
            "summary": (doc.get("overall") or {}).get("summary"), "dataset": doc.get("dataset"),
            "has_html": (d / f"{p.stem}.html").is_file(),
        })
        if len(rows) >= limit:
            break
    return rows
