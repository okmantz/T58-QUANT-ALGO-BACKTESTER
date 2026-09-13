"""Turns an uploaded backtest report -- an HTML/CSV report this app (or a
prior version of it) generated, a PDF export, or a plain screenshot of a
results screen -- into a plain-text summary the Research Agent can reason
over as an Observation, the same way it reasons over a real run_backtest()
result.

Why this exists: every other Research Agent tool (see app.ai.research_agent)
calls this app's OWN engine, so its numbers are always trustworthy. An
uploaded report is different -- it's evidence from OUTSIDE this run (a
different session, a different machine, a screenshot someone took on their
phone), so it can never be as authoritative as a fresh run_backtest() on
the currently-loaded data. This module is upfront about that distinction:

  - .csv / .html / .pdf are parsed DETERMINISTICALLY (pandas / regex over
    extracted text, no model involved) whenever the file contains
    recognizable metric labels or columns. This is the strong case: the
    numbers came out of the file exactly as written, no guessing.
  - .png / .jpg / .jpeg / .webp screenshots have no structured text to
    parse, so they go through a local Ollama vision call as a best-effort
    OCR/description step. The result is explicitly labeled AI-READ,
    UNVERIFIED in the summary text and in `is_ai_estimated`, exactly like
    app.ai.ollama_client's own "propose numbers, never trust blindly"
    convention -- the agent (and the person reading its transcript) should
    treat a screenshot-derived profit factor as a claim to sanity-check
    against a real run_backtest() on the same idea, never as ground truth.

Optional dependency: PDF text extraction reuses `pypdf` (already a
project dependency for the Research Library -- see
app.ai.research_library._extract_pdf_text), imported lazily so a report
with no PDFs still works with nothing extra installed.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = (".html", ".htm", ".csv", ".pdf", ".png", ".jpg", ".jpeg", ".webp")

# Label -> canonical metric name. Matches the vocabulary this app's own
# report generators use (see app.search.search_report, app.reports.generator,
# app.reports.validation_reports) plus a few common synonyms, so a report
# this app produced (or one from a similar tool) round-trips cleanly.
_METRIC_PATTERNS: dict[str, list[str]] = {
    "trade_count": [r"total\s*trades", r"trade\s*count", r"number\s*of\s*trades", r"n\s*trades"],
    "profit_factor": [r"profit\s*factor"],
    "sharpe_ratio": [r"sharpe\s*ratio", r"sharpe"],
    "sortino_ratio": [r"sortino\s*ratio", r"sortino"],
    "win_rate": [r"win\s*rate", r"win\s*%"],
    "max_drawdown_pct": [r"max(?:imum)?\s*drawdown", r"max\s*dd"],
    "net_profit": [r"net\s*profit", r"total\s*p&?l", r"total\s*profit"],
    "return_pct": [r"return\s*%", r"total\s*return", r"return\s*\(%\)"],
    "expectancy": [r"expectancy"],
    "eval_pass_probability": [r"eval(?:uation)?\s*pass\s*probab", r"evaluation\s*pass\s*%"],
}

_NUMBER = r"([\-+]?\$?\d[\d,]*\.?\d*)\s*%?"


@dataclass
class ImportedReport:
    source_path: str
    kind: str  # "html" | "csv" | "pdf" | "screenshot"
    metrics: dict = field(default_factory=dict)   # canonical metric name -> parsed float
    summary_text: str = ""
    is_ai_estimated: bool = False   # True only for screenshot vision-read results
    warnings: list = field(default_factory=list)

    def to_observation(self) -> dict:
        """Shape returned to the Research Agent's ReAct loop as a tool
        Observation -- see app.ai.research_agent._tool_read_uploaded_report."""
        out = {
            "source_file": Path(self.source_path).name,
            "kind": self.kind,
            "metrics_found": self.metrics,
            "summary": self.summary_text,
        }
        if self.is_ai_estimated:
            out["caution"] = (
                "These numbers were read from a screenshot by a vision model, not parsed from "
                "structured data -- treat them as an unverified claim, not ground truth. Sanity-check "
                "anything important with a real run_backtest() on an equivalent strategy/data if it "
                "matters to your conclusion."
            )
        if self.warnings:
            out["warnings"] = self.warnings
        return out


def _clean_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").replace("$", ""))
    except (ValueError, AttributeError):
        return None


def _scan_labeled_metrics(text: str) -> dict:
    """Regex-scans free text (HTML with tags stripped, PDF extracted text,
    or a flat CSV dump) for '<label> ... <number>' patterns using this
    app's own metric vocabulary. Deliberately simple/conservative -- a
    false negative (a metric this doesn't recognize) is fine, since the
    summary_text still carries the raw excerpt for the agent/person to
    read; a false positive (attaching the wrong number to a label) would
    be worse, so each pattern only looks at the ~40 chars right after the
    label, not the whole document."""
    found: dict[str, float] = {}
    for metric, patterns in _METRIC_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat + r"[^0-9\-+]{0,20}" + _NUMBER, text, re.IGNORECASE)
            if m:
                val = _clean_number(m.group(1))
                if val is not None:
                    found[metric] = val
                    break
    return found


def _strip_html_tags(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html)


def _import_html(path: Path) -> ImportedReport:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _strip_html_tags(raw)
    metrics = _scan_labeled_metrics(text)
    warnings = []

    # Also try structured <table> extraction (pandas) for reports that lay
    # metrics out as rows rather than inline "Label: value" text -- e.g.
    # this app's own Search Lab / Full Pipeline HTML reports.
    try:
        import pandas as pd

        tables = pd.read_html(str(path))
        for table in tables:
            if table.shape[1] < 2:
                continue
            for _, row in table.iterrows():
                label = str(row.iloc[0])
                for metric, patterns in _METRIC_PATTERNS.items():
                    if metric in metrics:
                        continue
                    if any(re.search(pat, label, re.IGNORECASE) for pat in patterns):
                        val = _clean_number(str(row.iloc[1]))
                        if val is not None:
                            metrics[metric] = val
    except ImportError:
        warnings.append("pandas.read_html unavailable -- only inline text was scanned, not <table> markup.")
    except Exception:
        pass  # no parseable tables; the inline-text regex scan above still stands

    summary = (
        f"Parsed HTML report '{path.name}': found {len(metrics)} recognizable metric(s): "
        f"{metrics}." if metrics else
        f"Parsed HTML report '{path.name}', but none of this app's known metric labels "
        f"(profit factor, Sharpe, max drawdown, trade count, win rate, ...) were found. "
        f"First 500 chars of extracted text: {text[:500]!r}"
    )
    return ImportedReport(source_path=str(path), kind="html", metrics=metrics, summary_text=summary, warnings=warnings)


def _import_csv(path: Path) -> ImportedReport:
    import pandas as pd

    warnings: list[str] = []
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return ImportedReport(
            source_path=str(path), kind="csv", summary_text=f"Could not parse '{path.name}' as CSV: {exc}",
            warnings=[str(exc)],
        )

    metrics: dict[str, float] = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
    # Case A: a "summary" CSV -- one row (or a label/value pair of columns)
    # matching this app's own stats field names directly.
    direct_names = {
        "profit_factor": "profit_factor", "sharpe_ratio": "sharpe_ratio", "sortino_ratio": "sortino_ratio",
        "win_rate": "win_rate", "max_drawdown_pct": "max_drawdown_pct", "net_profit": "net_profit",
        "return_pct": "return_pct", "expectancy": "expectancy",
        "total_trades": "trade_count", "trade_count": "trade_count",
    }
    for col_lower, canonical in direct_names.items():
        if col_lower in cols_lower:
            series = pd.to_numeric(df[cols_lower[col_lower]], errors="coerce").dropna()
            if len(series):
                metrics[canonical] = float(series.iloc[0] if len(series) == 1 else series.mean())

    # Case B: a raw TRADE LOG (one row per trade, a pnl-like column) --
    # compute the same headline numbers this app's own engine would, so a
    # trade log exported from elsewhere is just as usable as a summary file.
    pnl_col = next((cols_lower[c] for c in ("pnl", "profit", "net_pnl", "p&l", "profit_loss") if c in cols_lower), None)
    if pnl_col is not None and "trade_count" not in metrics:
        pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
        if len(pnl) > 0:
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            metrics["trade_count"] = float(len(pnl))
            metrics["net_profit"] = float(pnl.sum())
            metrics["win_rate"] = float(len(wins) / len(pnl) * 100) if len(pnl) else 0.0
            gross_loss = -losses.sum()
            metrics["profit_factor"] = float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf")
            equity = pnl.cumsum()
            running_max = equity.cummax()
            dd = (running_max - equity)
            metrics["max_drawdown_pct"] = float(dd.max())
            warnings.append(
                "profit_factor/max_drawdown_pct here were COMPUTED from a raw per-trade PnL column, not "
                "read from a pre-computed summary -- max_drawdown_pct is in raw PnL units, not %, unless "
                "the source column already was a % equity curve."
            )

    if not metrics:
        warnings.append(f"No recognizable metric columns found. Columns present: {list(df.columns)[:20]}")
    summary = f"Parsed CSV '{path.name}' ({len(df)} row(s)): {metrics}" if metrics else \
        f"Parsed CSV '{path.name}' ({len(df)} row(s)), but found no recognizable metrics or PnL column."
    return ImportedReport(source_path=str(path), kind="csv", metrics=metrics, summary_text=summary, warnings=warnings)


def _import_pdf(path: Path) -> ImportedReport:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ImportedReport(
            source_path=str(path), kind="pdf",
            summary_text=(
                "The 'pypdf' package isn't installed, so this PDF's text couldn't be extracted. "
                "Run `pip install pypdf` (it's already listed in config/requirements.txt) and re-upload."
            ),
            warnings=["pypdf not installed"],
        )
    try:
        reader = PdfReader(str(path))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        return ImportedReport(
            source_path=str(path), kind="pdf", summary_text=f"Could not extract text from '{path.name}': {exc}",
            warnings=[str(exc)],
        )
    metrics = _scan_labeled_metrics(text)
    summary = (
        f"Parsed PDF report '{path.name}' ({len(reader.pages)} page(s)): found {len(metrics)} recognizable "
        f"metric(s): {metrics}." if metrics else
        f"Parsed PDF report '{path.name}' ({len(reader.pages)} page(s)), but no known metric labels were "
        f"found. First 500 chars of extracted text: {text[:500]!r}"
    )
    return ImportedReport(source_path=str(path), kind="pdf", metrics=metrics, summary_text=summary)


def _import_screenshot(path: Path, settings=None) -> ImportedReport:
    """Best-effort vision read via local Ollama. Fails safe: any error
    (no settings, unreachable host, non-vision model, bad response) comes
    back as a summary explaining what happened, never an exception --
    same convention as app.ai.ollama_client and app.ai.research_agent."""
    if settings is None or not getattr(settings, "is_usable", False):
        return ImportedReport(
            source_path=str(path), kind="screenshot", is_ai_estimated=True,
            summary_text=(
                f"'{path.name}' is an image, which needs a local Ollama vision model to read (Ollama "
                "AI Assist isn't currently enabled/configured). Enable it above, using a vision-capable "
                "model (e.g. `ollama pull llama3.2-vision` or `ollama pull llava`), then re-run."
            ),
            warnings=["Ollama not configured/enabled"],
        )
    import requests

    try:
        image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception as exc:
        return ImportedReport(
            source_path=str(path), kind="screenshot", is_ai_estimated=True,
            summary_text=f"Could not read image file '{path.name}': {exc}", warnings=[str(exc)],
        )

    prompt = (
        "This image is a screenshot of a trading strategy backtest report. Transcribe every performance "
        "metric you can actually see (trade count, profit factor, Sharpe ratio, win rate, max drawdown %, "
        "net profit, return %, evaluation pass probability, etc.) as plain 'Label: value' lines. Only "
        "report numbers that are visibly printed in the image -- never estimate or infer a number that "
        "isn't shown. If you can't read the image or it isn't a backtest report, say so plainly."
    )
    host = (settings.host or "").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if getattr(settings, "api_key", None):
        headers["Authorization"] = f"Bearer {settings.api_key}"
    try:
        resp = requests.post(
            f"{host}/api/generate",
            headers=headers,
            json={
                "model": settings.model, "prompt": prompt, "images": [image_b64], "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw_text = resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return ImportedReport(
            source_path=str(path), kind="screenshot", is_ai_estimated=True,
            summary_text=f"Couldn't reach Ollama at {host} to read '{path.name}'.",
            warnings=[f"connection error: {host}"],
        )
    except Exception as exc:
        return ImportedReport(
            source_path=str(path), kind="screenshot", is_ai_estimated=True,
            summary_text=(
                f"Ollama request to read '{path.name}' failed: {exc}. If '{settings.model}' isn't a "
                "vision-capable model, pull one (e.g. `ollama pull llama3.2-vision`) and select it above."
            ),
            warnings=[str(exc)],
        )

    metrics = _scan_labeled_metrics(raw_text)
    return ImportedReport(
        source_path=str(path), kind="screenshot", metrics=metrics, is_ai_estimated=True,
        summary_text=f"Vision-read of screenshot '{path.name}' (UNVERIFIED, AI-read): {raw_text[:1000]}",
    )


def import_report_file(path: Path, settings=None) -> ImportedReport:
    """Single entry point -- dispatches by file extension. `settings`
    (an app.ai.ollama_settings.OllamaSettings) is only needed for image
    files; pass None for html/csv/pdf and it's simply ignored."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        return _import_html(path)
    if suffix == ".csv":
        return _import_csv(path)
    if suffix == ".pdf":
        return _import_pdf(path)
    if suffix in (".png", ".jpg", ".jpeg", ".webp"):
        return _import_screenshot(path, settings=settings)
    return ImportedReport(
        source_path=str(path), kind="unknown",
        summary_text=f"Unsupported file type '{suffix}' for '{path.name}'. Supported: {SUPPORTED_EXTENSIONS}.",
        warnings=[f"unsupported extension: {suffix}"],
    )
