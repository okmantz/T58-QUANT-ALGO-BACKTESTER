"""Shared look-and-feel for every generated HTML report (full pipeline, validation, refinement, survival, search).

One stylesheet + a handful of small renderers, so all reports read the same way:

    * ``BASE_CSS``          design tokens and component styles (print-friendly, no external fonts or scripts)
    * ``humanize(key)``     ``max_drawdown_pct`` -> "Max drawdown"
    * ``fmt_value(k, v)``   unit-aware value formatting ("$1,234.50", "12.3%", "Yes", "-")
    * ``kv_table(d)``       a clean two-column metric table (the raw key stays visible, muted, under the label)
    * ``stat_groups(d)``    the same, but split into titled, collapsible groups
    * ``verdict_hero(...)`` the verdict badge, T58 score gauge, component bars and the sorted reasons
    * ``kpi_cards(...)``    big-number tiles
    * ``page(...)``         a complete standalone HTML page around a body

Nothing here removes information: every key a table used to show is still shown, only labelled and formatted.
"""
from __future__ import annotations

import html as _html
import math
import re
from typing import Any, Iterable

# --------------------------------------------------------------------------------------------------
# Labels and value formatting
# --------------------------------------------------------------------------------------------------
_ACRONYMS = {
    "pct": "%", "dd": "DD", "mc": "MC", "pnl": "P&L", "r": "R", "mfe": "MFE", "mae": "MAE", "atr": "ATR",
    "ci": "CI", "ci95": "95% CI", "wfo": "WFO", "cpcv": "CPCV", "pbo": "PBO", "dsr": "DSR", "icir": "ICIR",
    "ga": "GA", "oos": "OOS", "is": "Is", "usd": "USD", "csv": "CSV", "id": "ID", "eod": "EOD", "tz": "TZ",
    "ct": "CT", "pf": "PF", "sl": "SL", "tp": "TP", "ev": "EV", "t58": "T58", "rr": "RR", "utc": "UTC",
    "sim": "sim", "sims": "sims", "vs": "vs", "p5": "P5", "p50": "P50", "p95": "P95",
}

# Hand-written labels where the generic rule reads badly.
LABELS = {
    "net_profit": "Net profit", "total_trades": "Total trades", "win_rate": "Win rate",
    "profit_factor": "Profit factor", "sharpe_ratio": "Sharpe ratio", "sortino_ratio": "Sortino ratio",
    "max_drawdown_pct": "Max drawdown", "max_drawdown": "Max drawdown (dollars)",
    "evaluation_pass_probability": "Evaluation pass probability (chain)",
    "first_payout_probability": "First payout probability (chain)",
    "risk_of_ruin_pct": "Risk of ruin",
    "initial_balance": "Starting balance", "risk_value": "Risk per trade value",
    "risk_mode": "Risk mode", "prop_account_rules": "Prop account rules (as simulated)",
    "reset_on_breach": "Buy a new account after a breach", "reset_on_target": "Buy a new account after the target",
    "n_simulations": "Simulated accounts", "median_days_to_first_payout": "Median days to first payout",
    "expected_payout": "Expected payout",
}

_DOLLAR_HINTS = ("avg_trade", "average_trade", "average_win", "average_los", "avg_loser", "avg_winner", "max_drawdown", "realized_loss", "profit", "loss", "balance", "equity", "payout", "dollars", "_usd", "pnl", "commission_per",
                 "expectancy", "avg_win", "avg_loss", "largest_", "gross_", "fee", "price", "account_size",
                 "cost", "drawdown_dollars", "risk_dollars")
_PCT_SUFFIXES = ("_pct", "_percent", "_probability", "_rate", "_prob")
_NO_UNIT_HINTS = ("ratio", "factor", "count", "trades", "days", "bars", "index", "hours", "pips", "size_mode",
                  "contracts", "attempts", "n_", "_n", "seed", "sims", "folds", "windows")


def humanize(key: Any) -> str:
    k = str(key)
    if k in LABELS:
        return LABELS[k]
    words = re.split(r"[_\s]+", k.strip("_"))
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in _ACRONYMS:
            out.append(_ACRONYMS[lw])
        else:
            out.append(w.lower() if i else w.capitalize())
    if out and out[-1] == "%":      # "..._pct": the value itself carries the % sign
        out = out[:-1]
    label = " ".join(out)
    return label[:1].upper() + label[1:]


def _num(v: float, key: str) -> str:
    lk = key.lower()
    if isinstance(v, float) and (math.isnan(v)):
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "∞" if v > 0 else "-∞"
    is_pct = lk.endswith(_PCT_SUFFIXES) or "_pct" in lk or lk.startswith("pct_") or lk in ("win_rate", "loss_rate")
    is_money = (not is_pct) and any(h in lk for h in _DOLLAR_HINTS) and not any(h in lk for h in ("ratio", "factor", "count", "trades", "days", "probability", "rate")) and not lk.startswith("n_") and not lk.endswith("_n")
    if is_pct:
        return f"{v:,.1f}%" if abs(v) < 1e6 else f"{v:,.0f}%"
    if is_money:
        sign = "-" if v < 0 else ""
        return f"{sign}${abs(v):,.2f}"
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) < 1e12):
        return f"{int(v):,}"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.1f}"
    if a >= 10:
        return f"{v:,.2f}"
    return f"{v:,.3f}".rstrip("0").rstrip(".") if a < 1 and a != 0 else f"{v:,.2f}"


def fmt_value(key: Any, v: Any, escape: bool = True) -> str:
    """HTML for one value, unit-aware. Dicts become inline chip lists, lists comma lists, bools Yes/No chips."""
    k = str(key)
    if v is None:
        return '<span class="na">–</span>'
    if isinstance(v, bool):
        return f'<span class="chip {"chip-yes" if v else "chip-no"}">{"Yes" if v else "No"}</span>'
    if isinstance(v, (int, float)):
        return f'<span class="num">{_num(v, k)}</span>'
    if isinstance(v, dict):
        if not v:
            return '<span class="na">–</span>'
        items = "".join(
            f'<span class="kv"><span class="kv-k">{_html.escape(humanize(a))}</span>'
            f'<span class="kv-v">{fmt_value(a, b, escape)}</span></span>' for a, b in v.items()
        )
        return f'<div class="kv-wrap">{items}</div>'
    if isinstance(v, (list, tuple)):
        if not v:
            return '<span class="na">–</span>'
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v) and len(v) <= 8:
            return '<span class="num">' + ", ".join(_num(x, k) for x in v) + "</span>"
        return _html.escape(", ".join(str(x) for x in v)) if escape else ", ".join(str(x) for x in v)
    s = str(v)
    return _html.escape(s) if escape else s


# --------------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------------
def kv_table(d: dict, key_header: str = "Metric", value_header: str = "Value", show_key: bool = True) -> str:
    if not d:
        return '<p class="muted">Nothing recorded.</p>'
    rows = []
    for k, v in d.items():
        raw = f'<span class="rawkey">{_html.escape(str(k))}</span>' if show_key and humanize(k).lower().replace(" ", "_") != str(k).lower() else ""
        rows.append(f'<tr><td class="lbl">{_html.escape(humanize(k))}{raw}</td><td class="val">{fmt_value(k, v)}</td></tr>')
    return (f'<table class="kv-table"><thead><tr><th>{_html.escape(key_header)}</th><th>{_html.escape(value_header)}</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


STAT_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Returns & profit", ("average_win", "average_los", "avg_trade", "average_trade", "average_r", "avg_r", "profit", "return", "expectancy", "payoff", "cagr", "avg_win", "avg_loss", "largest", "equity", "gross", "net_")),
    ("Risk & drawdown", ("risk_reward", "drawdown", "ruin", "ulcer", "var", "cvar", "sharpe", "sortino", "calmar", "volatility", "std", "recovery", "underwater")),
    ("Trade quality", ("win", "loss", "trades", "streak", "consec", "holding", "duration", "mfe", "mae", "r_multiple", "hit", "long", "short")),
    ("Costs, sizing & execution", ("commission", "spread", "slippage", "cost", "intended", "actual", "position", "contract", "size", "capped", "overshoot", "skipped")),
    ("Account resets & chains", ("reset", "segment", "chain", "attempt", "breach")),
]


def stat_groups(d: dict, groups: list[tuple[str, tuple[str, ...]]] | None = None, open_first: bool = True) -> str:
    """Split a flat metric dict into titled, collapsible groups (first matching group wins; leftovers go to 'Other')."""
    groups = groups or STAT_GROUPS
    buckets: dict[str, dict] = {g: {} for g, _ in groups}
    other: dict = {}
    for k, v in d.items():
        lk = str(k).lower()
        for g, hints in groups:
            if any(h in lk for h in hints):
                buckets[g][k] = v
                break
        else:
            other[k] = v
    parts = []
    first = True
    for g, _ in groups:
        if buckets[g]:
            parts.append(_details(g, kv_table(buckets[g]), open_=(first and open_first), count=len(buckets[g])))
            first = False
    if other:
        parts.append(_details("Other metrics", kv_table(other), open_=False, count=len(other)))
    return "".join(parts)


def _details(title: str, inner: str, open_: bool = False, count: int | None = None) -> str:
    badge = f'<span class="count">{count}</span>' if count is not None else ""
    return (f'<details class="fold"{" open" if open_ else ""}><summary><span class="fold-title">{_html.escape(title)}</span>{badge}</summary>'
            f'<div class="fold-body">{inner}</div></details>')


def details(title: str, inner: str, open_: bool = False, count: int | None = None) -> str:
    return _details(title, inner, open_, count)


def table(headers: Iterable[str], rows: Iterable[Iterable[Any]], num_cols: Iterable[int] = (), raw: bool = False) -> str:
    """Generic data table; ``rows`` cells are escaped unless ``raw``. ``num_cols`` right-aligns columns by index."""
    nc = set(num_cols)
    th = "".join(f'<th class="{"r" if i in nc else ""}">{_html.escape(str(h))}</th>' for i, h in enumerate(headers))
    body = []
    for r in rows:
        tds = "".join(f'<td class="{"r" if i in nc else ""}">{c if raw else _html.escape(str(c))}</td>' for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    return f'<table class="data"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


# --------------------------------------------------------------------------------------------------
# KPI tiles, gauge, bars
# --------------------------------------------------------------------------------------------------
def kpi_cards(items: list[tuple[str, str, str]]) -> str:
    """items = [(label, value_text, sub_text)] -> a responsive tile row. Values are trusted (already formatted)."""
    tiles = "".join(
        f'<div class="kpi"><div class="kpi-label">{_html.escape(l)}</div><div class="kpi-value">{v}</div>'
        + (f'<div class="kpi-sub">{s}</div>' if s else "") + "</div>"
        for l, v, s in items
    )
    return f'<div class="kpis">{tiles}</div>'


def tone_for(verdict: str | None) -> str:
    return {"READY": "good", "MARGINAL": "warn"}.get(str(verdict or "").upper(), "bad")


def score_gauge(score: float | None, tier: str = "", size: int = 150) -> str:
    """Semicircle 0-100 gauge as inline SVG."""
    if score is None:
        return ""
    s = max(0.0, min(100.0, float(score)))
    r = 60
    cx, cy = 75, 78
    circ = math.pi * r
    dash = circ * s / 100.0
    col = "var(--good)" if s >= 85 else "var(--warn)" if s >= 65 else "var(--bad)"
    return (
        f'<svg class="gauge" viewBox="0 0 150 100" width="{size}" height="{int(size * 100 / 150)}" role="img" aria-label="T58 Score {s:.1f} of 100">'
        f'<path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}" fill="none" stroke="var(--line)" stroke-width="12" stroke-linecap="round"/>'
        f'<path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}" fill="none" stroke="{col}" stroke-width="12" stroke-linecap="round" '
        f'stroke-dasharray="{dash:.1f} {circ:.1f}"/>'
        f'<text x="{cx}" y="{cy - 8}" text-anchor="middle" class="gauge-num">{s:.1f}</text>'
        f'<text x="{cx}" y="{cy + 10}" text-anchor="middle" class="gauge-sub">{_html.escape(tier or "T58 Score")}</text></svg>'
    )


def bar_row(label: str, value: float, weight: str = "", note: str = "") -> str:
    v = max(0.0, min(100.0, float(value)))
    col = "var(--good)" if v >= 75 else "var(--warn)" if v >= 50 else "var(--bad)"
    return (f'<div class="bar-row"><div class="bar-label">{_html.escape(label)}<span class="bar-w">{_html.escape(weight)}</span></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{v:.1f}%;background:{col}"></div></div>'
            f'<div class="bar-val">{v:.0f}<span>/100</span></div></div>')


# --------------------------------------------------------------------------------------------------
# Verdict hero
# --------------------------------------------------------------------------------------------------
_SCORE_RE = re.compile(r"T58 Score:\s*([\d.]+)/100\s*--\s*([A-Za-z ]+?)\s*\((\d+)/(\d+) checks available\)")
_COMP_RE = re.compile(r"^\s{2}([a-z0-9_]+):\s*([\d.]+)/100\s*\(weight\s*([+-]?\d+)\)\s*$")
_BLOCKER_PAT = re.compile(r"HARD (SAFETY|VALIDATION) GATE FAILED|FAILED:|DID NOT PASS", re.I)
_CAP_PAT = re.compile(r"CAPPED AT|DATA GATE|UNVERIFIED|UNUSABLE|NOT ALLOWED|not allowed|cannot support READY", re.I)

_COMPONENT_LABELS = {
    "pass_probability": "Evaluation pass probability", "first_payout_probability": "First payout probability",
    "risk_of_ruin": "Risk of ruin (inverted: higher = safer)", "walk_forward_stability": "Walk-forward stability",
    "monte_carlo_robustness": "Monte Carlo robustness", "parameter_stability": "Parameter stability",
    "expectancy": "Expectancy", "drawdown": "Drawdown control", "parsimony": "Parsimony (few parameters)",
    "cpcv_supporting": "CPCV supporting evidence",
}


def classify_reasons(reasons: list[str] | None):
    """Split a verdict's reasons into (score_match, components, blockers, caps, notes). Every input line lands in
    exactly one output (the score line and component lines are re-rendered as gauge/bars; all others keep their text)."""
    score = None
    comps: list[tuple[str, float, str]] = []
    blockers: list[str] = []
    caps: list[str] = []
    notes: list[str] = []
    for r in reasons or []:
        r = str(r)
        m = _SCORE_RE.search(r)
        if m and not r.lstrip().startswith("For reference"):
            score = (float(m.group(1)), m.group(2).strip(), int(m.group(3)), int(m.group(4)))
            continue
        cm = _COMP_RE.match(r)
        if cm:
            comps.append((cm.group(1), float(cm.group(2)), cm.group(3)))
            continue
        if _BLOCKER_PAT.search(r):
            blockers.append(r)
        elif _CAP_PAT.search(r):
            caps.append(r)
        else:
            notes.append(r)
            if m and score is None:  # "For reference, T58 Score ..." keeps its text but also feeds the gauge
                score = (float(m.group(1)), m.group(2).strip(), int(m.group(3)), int(m.group(4)))
    return score, comps, blockers, caps, notes


_PIPE_OUTSIDE_PARENS = re.compile(r"\s\|\s(?![^(]*\))")


def split_reason(text: str) -> list[str]:
    """Split a reason on ' | ' separators, but not on pipes inside parentheses (e.g. '(paid | passed)')."""
    return [p.strip(" -") for p in _PIPE_OUTSIDE_PARENS.split(str(text)) if p.strip(" -")]


def _reason_html(text: str) -> str:
    """Escape a reason and break ' | ' separated sub-parts onto their own lines (the EV verdict packs several into one string)."""
    return "<br>".join(_html.escape(p) for p in split_reason(text))


def _reason_list(items: list[str], cls: str) -> str:
    return f'<ul class="reasons {cls}">' + "".join(f"<li>{_reason_html(i)}</li>" for i in items) + "</ul>"


def verdict_hero(verdict: str | None, reasons: list[str] | None, title: str = "Full Pipeline verdict") -> str:
    """Badge + gauge + component bars + reasons grouped as 'Blocking', 'Capping', 'Notes'. Renders nothing without a verdict."""
    if not verdict:
        return ""
    tone = tone_for(verdict)
    score, comps, blockers, caps, notes = classify_reasons(reasons)
    meaning = {
        "READY": "Cleared every hard gate and scored in the Strong/Elite tier. Next step: forward-test on a demo, not a purchase.",
        "MARGINAL": "No hard failure, but the evidence is thinner than READY requires or a cap applied. See the amber items.",
        "NOT READY": "At least one hard gate failed, or the score is below the passing tiers. See the red items first.",
    }.get(str(verdict).upper(), "")
    gauge = score_gauge(score[0], score[1]) if score else ""
    coverage = f'<div class="muted small">{score[2]}/{score[3]} score components measured</div>' if score else ""
    bars = ""
    if comps:
        bars = '<div class="bars">' + "".join(
            bar_row(_COMPONENT_LABELS.get(n, humanize(n)), v, f"weight {w}") for n, v, w in comps
        ) + "</div>"
    body = ""
    if blockers:
        body += f'<h4 class="rh bad">Blocking ({len(blockers)})</h4>' + _reason_list(blockers, "bad")
    if caps:
        body += f'<h4 class="rh warn">Capping / limiting ({len(caps)})</h4>' + _reason_list(caps, "warn")
    if notes:
        body += f'<h4 class="rh">Notes &amp; supporting detail ({len(notes)})</h4>' + _reason_list(notes, "")
    return (
        f'<section class="hero hero-{tone}">'
        f'<div class="hero-top"><div class="hero-main"><div class="eyebrow">{_html.escape(title)}</div>'
        f'<div class="hero-verdict">{_html.escape(str(verdict))}</div><p class="hero-meaning">{_html.escape(meaning)}</p></div>'
        f'<div class="hero-gauge">{gauge}{coverage}</div></div>'
        f'{bars}<div class="hero-reasons">{body}</div></section>'
    )


def banner(kind: str, title: str, inner_html: str) -> str:
    """kind: info | warn | bad | good. ``inner_html`` is trusted markup."""
    return f'<div class="banner banner-{kind}"><div class="banner-title">{_html.escape(title)}</div>{inner_html}</div>'


def section(title: str, intro: str = "", inner: str = "", anchor: str = "") -> str:
    a = f' id="{anchor}"' if anchor else ""
    intro_html = f'<p class="muted intro">{intro}</p>' if intro else ""
    return f'<section class="sec"{a}><h2>{_html.escape(title)}</h2>{intro_html}{inner}</section>'


# --------------------------------------------------------------------------------------------------
# Stylesheet + page shell
# --------------------------------------------------------------------------------------------------
BASE_CSS = """
:root{--ink:#111827;--ink2:#374151;--muted:#6b7280;--line:#e5e7eb;--line2:#f1f2f4;--panel:#fff;--bg:#f5f6f8;
--brand:#1f4fd8;--brand-soft:#eaf0ff;--good:#12a05c;--good-bg:#ecfdf3;--warn:#c98200;--warn-bg:#fff7e6;--bad:#d92d20;--bad-bg:#fef3f2;
--info:#1f4fd8;--info-bg:#eff4ff;--radius:12px;--shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.06)}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
-webkit-font-smoothing:antialiased}
.masthead{background:linear-gradient(135deg,#0b1020 0%,#16203c 100%);color:#e8ecf5;padding:20px 40px;display:flex;align-items:center;gap:16px}
.masthead img{height:38px;display:block}
.masthead h1{margin:0;font-size:17px;font-weight:700;letter-spacing:.01em;color:#fff;border:0;padding:0}
.masthead .sub{margin-top:2px;font-size:11px;color:#9fb0d0;letter-spacing:.06em;text-transform:uppercase}
.content,.wrap{max-width:1080px;margin:0 auto;padding:26px 32px 64px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:12px;margin:0 0 12px;color:var(--ink2);text-transform:uppercase;letter-spacing:.09em;font-weight:700;display:flex;align-items:center;gap:10px}
h2:after{content:"";flex:1;height:1px;background:var(--line)}
h3{font-size:14px;margin:18px 0 8px;color:var(--ink)}
h4{margin:12px 0 6px;font-size:12px}
.sec{margin-top:34px;break-inside:auto}
.muted{color:var(--muted)} .small{font-size:12px} .intro{margin:-2px 0 12px;font-size:13px;max-width:78ch}
.na{color:#9ca3af}
.sub-meta{color:var(--muted);font-size:12.5px;margin:6px 0 0}
.titlebar{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline;margin-bottom:6px}
.titlebar .name{font-size:22px;font-weight:750;letter-spacing:-.01em}
.pills{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 2px}
.pill{background:#fff;border:1px solid var(--line);border-radius:999px;padding:3px 11px;font-size:12px;color:var(--ink2)}
.pill b{color:var(--ink)}
/* hero */
.hero{border-radius:16px;padding:22px 24px;margin:18px 0 6px;background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);border-left:6px solid var(--muted)}
.hero-good{border-left-color:var(--good);background:linear-gradient(180deg,var(--good-bg),#fff 70%)}
.hero-warn{border-left-color:var(--warn);background:linear-gradient(180deg,var(--warn-bg),#fff 70%)}
.hero-bad{border-left-color:var(--bad);background:linear-gradient(180deg,var(--bad-bg),#fff 70%)}
.hero-top{display:flex;gap:24px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:700}
.hero-verdict{font-size:38px;font-weight:800;letter-spacing:-.02em;line-height:1.1;margin-top:2px}
.hero-good .hero-verdict{color:var(--good)} .hero-warn .hero-verdict{color:var(--warn)} .hero-bad .hero-verdict{color:var(--bad)}
.hero-meaning{margin:6px 0 0;color:var(--ink2);max-width:60ch}
.hero-gauge{text-align:center}
.gauge-num{font-size:26px;font-weight:800;fill:var(--ink)} .gauge-sub{font-size:9px;fill:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.bars{margin-top:18px;display:grid;grid-template-columns:1fr 1fr;gap:6px 28px}
.bar-row{display:grid;grid-template-columns:minmax(120px,210px) 1fr 52px;gap:10px;align-items:center;font-size:12.5px}
.bar-label{color:var(--ink2)} .bar-w{display:block;color:#9ca3af;font-size:10.5px}
.bar-track{height:8px;border-radius:99px;background:var(--line2);overflow:hidden}
.bar-fill{height:100%;border-radius:99px}
.bar-val{text-align:right;font-weight:700} .bar-val span{font-weight:400;color:var(--muted);font-size:10.5px}
.hero-reasons{margin-top:14px}
.rh{text-transform:uppercase;letter-spacing:.07em;margin:14px 0 6px;color:var(--ink2)}
.rh.bad{color:var(--bad)} .rh.warn{color:var(--warn)}
ul.reasons{margin:0;padding:0;list-style:none}
ul.reasons li{padding:8px 12px;margin:0 0 6px;border-radius:8px;background:#fff;border:1px solid var(--line);font-size:13px;overflow-wrap:anywhere}
ul.reasons.bad li{border-left:4px solid var(--bad);background:var(--bad-bg)}
ul.reasons.warn li{border-left:4px solid var(--warn);background:var(--warn-bg)}
ul.reasons:not(.bad):not(.warn) li{border-left:4px solid var(--line)}
/* kpis */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:14px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;box-shadow:var(--shadow)}
.kpi-label{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.kpi-value{font-size:27px;font-weight:780;margin-top:4px;letter-spacing:-.01em}
.kpi-value.good{color:var(--good)} .kpi-value.warn{color:var(--warn)} .kpi-value.bad{color:var(--bad)}
.kpi-sub{font-size:11.5px;color:var(--muted);margin-top:2px}
/* tables */
table{border-collapse:collapse;width:100%;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow);margin-top:6px}
th,td{padding:8px 14px;text-align:left;border-bottom:1px solid var(--line2);font-size:13px;vertical-align:top}
th{background:#f8f9fb;color:var(--ink2);font-weight:650;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--line)}
tbody tr:nth-child(even) td{background:#fcfcfd}
tr:last-child td{border-bottom:0}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
.kv-table td.lbl{width:46%;color:var(--ink2)}
.kv-table td.val{font-variant-numeric:tabular-nums;font-weight:550}
.rawkey{display:block;font:10.5px ui-monospace,SFMono-Regular,Menlo,monospace;color:#9ca3af;margin-top:1px;font-weight:400}
.num{font-variant-numeric:tabular-nums}
.chip{display:inline-block;padding:1px 9px;border-radius:99px;font-size:11.5px;font-weight:650}
.chip-yes{background:var(--good-bg);color:var(--good)} .chip-no{background:var(--line2);color:var(--muted)}
.kv-wrap{display:flex;flex-wrap:wrap;gap:5px}
.kv{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden;font-size:11.5px;background:#fff}
.kv-k{background:#f8f9fb;color:var(--muted);padding:2px 8px;border-right:1px solid var(--line)} .kv-v{padding:2px 8px;font-weight:600}
/* folds */
details.fold{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);margin:10px 0;box-shadow:var(--shadow);overflow:hidden}
details.fold>summary{cursor:pointer;list-style:none;padding:11px 16px;display:flex;align-items:center;gap:10px;font-weight:650;background:#fafbfc;user-select:none}
details.fold>summary::-webkit-details-marker{display:none}
details.fold>summary:before{content:"▸";color:var(--muted);transition:transform .15s}
details.fold[open]>summary:before{transform:rotate(90deg)}
details.fold[open]>summary{border-bottom:1px solid var(--line)}
.count{margin-left:auto;background:var(--line2);color:var(--muted);border-radius:99px;font-size:11px;padding:1px 9px;font-weight:600}
.fold-body{padding:0} .fold-body table{border:0;border-radius:0;box-shadow:none;margin:0} .fold-body>p,.fold-body>ul{margin:12px 16px}
ul.method{margin:0;padding:12px 18px 12px 34px} ul.method li{margin:5px 0;color:var(--ink2);font-size:13px}
/* banners */
.banner{border-radius:var(--radius);padding:13px 16px;margin:14px 0;border:1px solid var(--line);border-left:5px solid var(--muted);background:#fff}
.banner-title{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.banner ul{margin:0;padding-left:18px} .banner li{margin:4px 0;font-size:13px} .banner p{margin:0 0 7px;font-size:13px} .banner p:last-child{margin:0}
.banner-info{background:var(--info-bg);border-left-color:var(--info)} .banner-info .banner-title{color:#1e3a8a}
.banner-warn{background:var(--warn-bg);border-left-color:var(--warn)} .banner-warn .banner-title{color:#92400e}
.banner-bad{background:var(--bad-bg);border-left-color:var(--bad)} .banner-bad .banner-title{color:#b42318}
.banner-good{background:var(--good-bg);border-left-color:var(--good)} .banner-good .banner-title{color:#067647}
/* charts, tabs */
.chart{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:12px;margin-top:8px;box-shadow:var(--shadow)}
.chart-row{display:flex;gap:14px;flex-wrap:wrap} .chart-row .chart{flex:1 1 340px} .chart svg{width:100%;height:auto;display:block}
.tabs{display:flex;gap:4px;margin-top:26px;border-bottom:2px solid var(--line)}
.tab-btn{background:none;border:0;padding:10px 18px;font:700 12.5px inherit;font-family:inherit;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;text-transform:uppercase;letter-spacing:.05em}
.tab-btn:hover{color:var(--ink)} .tab-btn.active{color:var(--ink);border-bottom-color:var(--brand)}
.tab-panel{display:none} .tab-panel.active{display:block}
.toc{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0 0}
.toc a{font-size:12px;text-decoration:none;color:var(--ink2);background:#fff;border:1px solid var(--line);border-radius:99px;padding:3px 12px}
.toc a:hover{border-color:var(--brand);color:var(--brand)}
.param-changed td{background:var(--warn-bg)!important}
.footer-note{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);font-size:12px}
/* legacy classes kept so older report fragments still look right */
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 18px;box-shadow:var(--shadow)}
span.pass:not(.pill),td.pass,span.ok,td.ok{color:var(--good);font-weight:650} span.fail:not(.pill),td.fail,span.bad,td.bad{color:var(--bad);font-weight:650}
.warning-banner{background:var(--warn-bg);border:1px solid #f0b429;border-left:5px solid #f0b429;border-radius:var(--radius);padding:13px 16px;margin:14px 0}
.warning-banner .warning-title{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#92400e;margin-bottom:6px}
.warning-banner ul{margin:0;padding-left:18px}
.info-banner{background:var(--info-bg);border:1px solid #bcd0ff;border-left:5px solid var(--info);border-radius:var(--radius);padding:13px 16px;margin:14px 0}
.info-banner .info-title{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#1e3a8a;margin-bottom:6px}
.info-banner p{font-size:13px;margin:0 0 7px}
.risk-flags-banner{border-radius:var(--radius);padding:13px 16px;margin:14px 0;border:1px solid #fecdca;border-left:5px solid var(--bad);background:var(--bad-bg)}
.risk-flags-banner .verdict-title{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#b42318;margin-bottom:6px}
.risk-flags-banner ul{margin:0;padding-left:18px}
.floating-equity-badge{border-radius:var(--radius);padding:13px 16px;margin:14px 0;border:1px solid #fedf89;border-left:5px solid #dc6803;background:#fffaeb}
.floating-equity-badge .verdict-title{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#93370d;margin-bottom:6px}
.floating-equity-badge p{margin:0;font-size:13px;color:#53390f}
.sizing-halt-banner{border-radius:var(--radius);padding:16px 18px;margin:14px 0;border:2px solid var(--bad);background:var(--bad-bg)}
.sizing-halt-banner .halt-title{font-weight:800;font-size:15px;text-transform:uppercase;letter-spacing:.04em;color:#b42318;margin-bottom:6px}
.sizing-halt-banner p{margin:0;color:#7a271a}

/* compat for older report templates (survival / refinement / search) */
.headline{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:12px}
.card .label{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.card .value{font-size:27px;font-weight:780;margin-top:4px;letter-spacing:-.01em}
.card.win .value{color:var(--good)}
.warning{background:var(--warn-bg);border:1px solid #f0c36d;border-left:5px solid #f0b429;color:#78350f;border-radius:var(--radius);padding:13px 18px;margin-top:14px;font-size:13px;line-height:1.5}
.warning b{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.warning ul{margin:6px 0 0;padding-left:18px}
div.warn{background:var(--warn-bg);border:1px solid #f0b429;border-left:5px solid #f0b429;color:#78350f;border-radius:var(--radius);padding:10px 14px;font-size:13px;margin:0 0 12px}
pre.config{background:#0f172a;color:#d6e3ff;padding:16px;border-radius:var(--radius);overflow-x:auto;font-size:12px;line-height:1.5;max-height:420px;overflow-y:auto}
.meta{color:var(--muted);font-size:12.5px;margin:2px 0 0}
.funnel{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0 22px}
.stage-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;box-shadow:var(--shadow)}
.stage-card .n{font-size:26px;font-weight:780;color:var(--brand)} .stage-card .label{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:700}
td.notes{color:var(--muted);font-size:11.5px;max-width:320px}
tr.pass td.pass-badge{color:var(--good);font-weight:700} tr.fail td.fail-badge{color:var(--bad);font-weight:700}
.svgwrap{background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px;overflow-x:auto}
@media (max-width:760px){.content,.wrap{padding:18px 14px 48px}.masthead{padding:16px 16px}.bars{grid-template-columns:1fr}.hero-verdict{font-size:30px}}
@media print{
 body{background:#fff;font-size:12px} .masthead,.hero,.banner,.kpi,.bar-fill,.chip,th{-webkit-print-color-adjust:exact;print-color-adjust:exact}
 .content,.wrap{max-width:none;padding:0 6mm} .tabs,.toc,#pdf-btn{display:none!important}
 details.fold{break-inside:avoid} details.fold>summary:before{content:""} .tab-panel{display:block!important}
 table,.kpi,.chart,.hero{box-shadow:none} tr{break-inside:avoid} h2{break-after:avoid}
}
"""

# Open every fold when printing / exporting so nothing is hidden on paper.
PRINT_JS = "<script>window.addEventListener('beforeprint',function(){document.querySelectorAll('details.fold').forEach(function(d){d.open=true})});</script>"


def masthead(title: str, sub: str = "", logo_b64: str | None = None) -> str:
    img = f'<img src="data:image/png;base64,{logo_b64}" alt="T58"/>' if logo_b64 else ""
    return (f'<div class="masthead">{img}<div class="titles"><h1>{_html.escape(title)}</h1>'
            + (f'<div class="sub">{_html.escape(sub)}</div>' if sub else "") + "</div></div>")


def page(title: str, body: str, subtitle: str = "", logo_b64: str | None = None, extra_head: str = "", extra_end: str = "") -> str:
    """A complete standalone page: masthead + content + shared CSS. ``body`` is trusted markup."""
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_html.escape(title)}</title><style>{BASE_CSS}</style>{extra_head}</head><body>"
        f'{masthead(title, subtitle or "Precision-tested. Falsification-checked. No shortcuts.", logo_b64)}'
        f'<div class="content">{body}'
        '<p class="footer-note muted">Generated by T58 Quant Algo Backtester. All figures are simulated estimates from historical data and '
        "resampling; they are not predictions or investment advice.</p></div>"
        f"{PRINT_JS}{extra_end}</body></html>"
    )


# For old .format()-style templates: the same CSS with braces doubled so it can be embedded in them.
TEMPLATE_CSS = BASE_CSS.replace("{", "{{").replace("}", "}}")
