"""Trader profile, per-workflow memory, pre-flight checks, time-to-answer and run comparison (2026-11-05 audit, "make the pipeline easier to run well").

Everything here is a convenience layered on top of the existing forms: it pre-fills fields, warns before a launch, and compares finished runs. None of it
changes what a backtest computes, and every route degrades to "no help" rather than an error (a broken helper must never block a run).

Storage (all under ``data/config``, which the backup vault already includes):
  trader_profile.json   who is trading what: prop firm preset + account size, instruments, session hours, typical hold, risk per trade
  workflow_memory.json  the last form values submitted to each workflow (server-side, so it follows the install, not the browser)
  run_durations.json    measured seconds per workflow (a rolling window), behind the time-to-answer estimate
"""
from __future__ import annotations

import json
import math
import re
import shutil
import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request

workflow_bp = Blueprint("workflow_support", __name__)
_LOCK = threading.Lock()

HOLD_CHOICES = {
    "scalp": "Scalp (minutes)",
    "intraday": "Intraday (hours, flat by the close)",
    "swing": "Swing (days)",
}
# Form-field names the profile can pre-fill, per its own key. A workflow's form only receives the names it actually has.
_PROFILE_TO_FIELDS = {
    "account_size": ("account_size", "initial_balance"),
    "risk_per_trade_pct": ("risk_per_trade_pct", "risk_pct"),
}
_SECRET_FIELD = re.compile(r"(key|token|secret|password|passwd|webhook|api|credential|email)", re.I)
_MAX_FIELDS = 80
_MAX_VALUE_LEN = 400
_DURATION_WINDOW = 12

# Seconds per 10,000 bars when no run of this workflow has been timed yet (conservative priors; replaced by measured values after the first run).
_PRIOR_SEC_PER_10K_BARS = {
    "full_pipeline": 160.0, "quick_optimize": 70.0, "discovery_engine": 240.0, "evolution": 300.0, "search": 180.0, "multi_market": 220.0,
}
_DEFAULT_PRIOR = 120.0


def _config_dir() -> Path:
    from app.data.storage import get_app_base_dir
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read(name: str, default):
    try:
        p = _config_dir() / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
    except Exception:  # noqa: BLE001
        return default


def _write(name: str, data) -> bool:
    try:
        p = _config_dir() / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------------------------------- trader profile
def load_profile() -> dict:
    p = _read("trader_profile.json", {})
    return p if isinstance(p, dict) else {}


def clean_profile(raw: dict) -> dict:
    """Validate/normalise a submitted profile; unknown keys are dropped, bad numbers are omitted rather than guessed."""
    out: dict = {}

    def num(key, lo, hi):
        try:
            v = float(str(raw.get(key, "")).replace(",", "").replace("$", "").strip())
        except ValueError:
            return
        if math.isfinite(v) and lo <= v <= hi:
            out[key] = v

    preset = str(raw.get("preset_key") or "").strip()
    if preset:
        try:
            from app.prop.presets import get_preset
            get_preset(preset)
            out["preset_key"] = preset
        except Exception:  # noqa: BLE001
            pass
    num("account_size", 500, 50_000_000)
    num("risk_per_trade_pct", 0.01, 10)
    num("session_start_hour", 0, 23)
    num("session_end_hour", 0, 24)
    inst = raw.get("instruments")
    if isinstance(inst, str):
        inst = [x for x in re.split(r"[,\s]+", inst) if x]
    if isinstance(inst, list):
        out["instruments"] = [re.sub(r"[^A-Za-z0-9._/-]", "", str(x))[:20].upper() for x in inst if str(x).strip()][:12]
    hold = str(raw.get("typical_hold") or "").strip().lower()
    if hold in HOLD_CHOICES:
        out["typical_hold"] = hold
    tz = str(raw.get("session_timezone") or "").strip()
    if tz:
        out["session_timezone"] = tz[:40]
    return out


def profile_defaults() -> dict:
    """Flat {form field name: value} the page script can pre-fill. Prop rules come from the profile's preset; an explicit account size wins."""
    prof = load_profile()
    out: dict = {}
    if prof.get("preset_key"):
        try:
            from app.prop.presets import get_preset
            pr = get_preset(prof["preset_key"])
            out.update(account_size=pr.account_size, profit_target=pr.evaluation_profit_target_pct,
                       daily_loss=pr.daily_loss_limit_pct, max_dd=pr.max_drawdown_pct)
            out["prop_preset_key"] = pr.key
        except Exception:  # noqa: BLE001
            pass
    for k, names in _PROFILE_TO_FIELDS.items():
        if prof.get(k) is not None:
            for n in names:
                out[n] = prof[k]
    if prof.get("session_start_hour") is not None:
        out["session_start_hour"] = int(prof["session_start_hour"])
    if prof.get("session_end_hour") is not None:
        out["session_end_hour"] = int(prof["session_end_hour"])
    return out


# --------------------------------------------------------------------------------------------------- workflow memory
def _safe_workflow(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", str(name or "").lower())[:40]


def remember(workflow: str, values: dict) -> bool:
    wf = _safe_workflow(workflow)
    if not wf or not isinstance(values, dict):
        return False
    keep = {}
    for k, v in values.items():
        k = str(k)[:60]
        if _SECRET_FIELD.search(k) or isinstance(v, (dict, list)) or len(keep) >= _MAX_FIELDS:
            continue
        s = str(v)
        if len(s) <= _MAX_VALUE_LEN:
            keep[k] = s
    if not keep:
        return False
    with _LOCK:
        mem = _read("workflow_memory.json", {})
        if not isinstance(mem, dict):
            mem = {}
        mem[wf] = {"saved_at": time.time(), "values": keep}
        return _write("workflow_memory.json", mem)


def recall(workflow: str) -> dict:
    mem = _read("workflow_memory.json", {})
    row = mem.get(_safe_workflow(workflow)) if isinstance(mem, dict) else None
    return row if isinstance(row, dict) else {}


# --------------------------------------------------------------------------------------------------- time-to-answer
def record_duration(workflow: str, bars: int | None, seconds: float) -> None:
    wf = _safe_workflow(workflow)
    if not wf or not seconds or seconds <= 0:
        return
    with _LOCK:
        d = _read("run_durations.json", {})
        if not isinstance(d, dict):
            d = {}
        rows = list(d.get(wf, []))
        rows.append({"bars": int(bars or 0), "seconds": round(float(seconds), 1), "at": time.time()})
        d[wf] = rows[-_DURATION_WINDOW:]
        _write("run_durations.json", d)


def estimate_seconds(workflow: str, bars: int | None) -> dict:
    """Median of measured seconds-per-10k-bars for this workflow, scaled to ``bars``; the prior until at least one run is timed."""
    wf = _safe_workflow(workflow)
    rows = [r for r in (_read("run_durations.json", {}) or {}).get(wf, []) if r.get("bars", 0) > 0 and r.get("seconds", 0) > 0]
    n = max(int(bars or 0), 1)
    timed = [r["seconds"] for r in (_read("run_durations.json", {}) or {}).get(wf, []) if r.get("seconds", 0) > 0]
    if not rows and timed:
        mid_s = sorted(timed)[len(timed) // 2]
        return {"seconds": round(mid_s), "low": round(mid_s * 0.6), "high": round(mid_s * 1.6),
                "basis": f"your last {len(timed)} run(s) of this workflow", "text": _fmt_range(mid_s * 0.6, mid_s * 1.6)}
    if rows:
        per10k = sorted(r["seconds"] / (r["bars"] / 10_000.0) for r in rows)
        mid = per10k[len(per10k) // 2]
        basis = f"your last {len(rows)} run(s) of this workflow"
    else:
        mid = _PRIOR_SEC_PER_10K_BARS.get(wf, _DEFAULT_PRIOR)
        basis = "a typical machine (no run of this workflow has been timed yet)"
    secs = max(5.0, mid * n / 10_000.0)
    return {"seconds": round(secs), "low": round(secs * 0.6), "high": round(secs * 1.6), "basis": basis, "text": _fmt_range(secs * 0.6, secs * 1.6)}


def _fmt_dur(s: float) -> str:
    s = max(int(round(s)), 1)
    if s < 90:
        return f"{s} sec"
    if s < 5400:
        return f"{int(round(s / 60))} min"
    return f"{s / 3600:.1f} h"


def _fmt_range(lo: float, hi: float) -> str:
    a, b = _fmt_dur(lo), _fmt_dur(hi)
    return a if a == b else f"{a} to {b}"


# --------------------------------------------------------------------------------------------------- pre-flight
def preflight(workflow: str, dataset: str | None = None, preset_key: str | None = None, adaptive: bool | None = None,
              optimizer_mode: str | None = None) -> list[dict]:
    """Checks to run BEFORE a long job starts. Each: {id, status: ok|warn|fail, text}. ``fail`` means the run cannot work; ``warn`` means it can
    but the answer will be weaker or may be capped."""
    checks: list[dict] = []

    def add(i, status, text):
        checks.append({"id": i, "status": status, "text": text})

    bars = None
    if dataset:
        try:
            from app.data.storage import list_stored_datasets
            hit = next((d for d in list_stored_datasets() if d.name == dataset or str(getattr(d, "path", "")).endswith(dataset)), None)
            if hit is None:
                add("dataset", "fail", f"Dataset '{dataset}' was not found in your data folder.")
            else:
                size = int(getattr(hit, "size_bytes", 0) or 0)
                if size < 4096:
                    add("dataset", "fail", f"'{hit.name}' is only {size} bytes: it looks like a placeholder, not market data.")
                else:
                    bars = int(size / 48)   # rough: ~48 bytes per CSV bar; only used for the time estimate
                    add("dataset", "ok", f"Dataset '{hit.name}' found ({size / 1e6:.1f} MB).")
        except Exception as exc:  # noqa: BLE001
            add("dataset", "warn", f"Could not check the dataset: {exc}")
    if preset_key:
        try:
            from app.prop.presets import ack_status
            st = ack_status(preset_key)
            if st.get("needs_ack"):
                days = st.get("age_days")
                add("rules", "warn", (f"The '{st.get('label')}' rules were last verified {days} days ago" if days is not None
                                      else f"The '{st.get('label')}' rules have never been verified")
                    + " (older than 30 days): a READY verdict is not possible until you re-check them against the firm's page.")
            else:
                add("rules", "ok", f"Prop rules for '{st.get('label')}' verified {st.get('age_days')} day(s) ago.")
        except KeyError:
            add("rules", "warn", "No prop-firm preset selected: the verdict cannot reach READY without verified prop rules.")
        except Exception:  # noqa: BLE001
            pass
    if adaptive is False:
        add("adaptive", "warn", "Adaptive risk is off. It is on by default; with it off, results are not comparable with runs that used it.")
    elif adaptive:
        add("adaptive", "ok", "Adaptive risk is on.")
    if optimizer_mode in ("tpe", "cma_es", "auto"):
        import importlib.util as iu
        need = {"tpe": "optuna", "cma_es": "cma", "auto": "optuna"}[optimizer_mode]
        if iu.find_spec(need) is None:
            add("optimizer", "warn" if optimizer_mode == "auto" else "fail",
                f"The '{need}' package is not installed" + (": Auto will fall back to the genetic search." if optimizer_mode == "auto"
                                                            else f", so '{optimizer_mode}' cannot run. Install it or pick another optimizer."))
    try:
        free = shutil.disk_usage(str(_config_dir())).free
        if free < 500 * 1024 * 1024:
            add("disk", "fail" if free < 100 * 1024 * 1024 else "warn", f"Only {free / 1e6:.0f} MB of disk space is free; reports and caches may not save.")
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.orchestration.resource_guard import HEAVY_JOB_GUARD
        busy = getattr(HEAVY_JOB_GUARD, "active_names", None)
        names = busy() if callable(busy) else None
        if names:
            add("busy", "warn", f"Another heavy job is already running ({', '.join(map(str, names))}); this one will be refused or slowed until it finishes.")
    except Exception:  # noqa: BLE001
        pass
    if workflow:
        est = estimate_seconds(workflow, bars)
        add("time", "ok", f"Expected time to an answer: {est['text']} (based on {est['basis']}).")
    return checks


# --------------------------------------------------------------------------------------------------- run comparison
_COMPARE_ROWS = (
    ("Verdict stage", "ladder"), ("Net profit", "net_profit"), ("Trades", "trades"), ("Win rate %", "win_rate"), ("Profit factor", "profit_factor"),
    ("Sharpe", "sharpe_ratio"), ("Max drawdown %", "max_drawdown_pct"), ("Eval pass %", "eval_pass_probability"),
    ("First payout %", "first_payout_probability"), ("Risk of ruin %", "risk_of_ruin_pct"),
)


def comparison(run_ids: list[str]) -> dict:
    from app.reports import run_history
    by_id = {r.get("run_id"): r for r in run_history.load_runs()}
    cols = [by_id[i] for i in run_ids if i in by_id][:4]
    rows = []
    for label, key in _COMPARE_ROWS:
        vals = [c.get(key) for c in cols]
        nums = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        best = None
        if key not in ("ladder",) and len(nums) > 1:
            lower_better = key in ("max_drawdown_pct", "risk_of_ruin_pct")
            best = (min if lower_better else max)(nums)
        rows.append({"label": label, "key": key, "values": vals, "best": best})
    return {"columns": cols, "rows": rows}


# --------------------------------------------------------------------------------------------------- routes
@workflow_bp.route("/api/profile/defaults")
def api_profile_defaults():
    return jsonify({"profile": load_profile(), "defaults": profile_defaults()})


@workflow_bp.route("/api/workflow-memory/<workflow>", methods=["GET"])
def api_recall(workflow):
    return jsonify(recall(workflow))


@workflow_bp.route("/api/workflow-memory/<workflow>", methods=["POST"])
def api_remember(workflow):
    data = request.get_json(silent=True) or request.form.to_dict()
    return jsonify({"saved": remember(workflow, data)})


@workflow_bp.route("/api/preflight")
def api_preflight():
    a = request.args
    adaptive = None if "adaptive" not in a else a.get("adaptive") in ("1", "true", "on")
    return jsonify({"checks": preflight(a.get("workflow", ""), a.get("dataset"), a.get("preset"), adaptive, a.get("optimizer_mode"))})


@workflow_bp.route("/api/time-estimate")
def api_time_estimate():
    try:
        bars = int(float(request.args.get("bars", "0")))
    except ValueError:
        bars = 0
    return jsonify(estimate_seconds(request.args.get("workflow", ""), bars))


@workflow_bp.route("/runs/compare")
def runs_compare():
    ids = [x for x in re.split(r"[,\s]+", request.args.get("ids", "")) if x]
    return render_template("runs_compare.html", active_page="runs", cmp=comparison(ids), ids=ids)


@workflow_bp.route("/api/runs/recent")
def api_runs_recent():
    from app.reports import run_history
    rows = run_history.load_runs()[-40:][::-1]
    keep = ("run_id", "timestamp", "strategy_name", "instrument", "timeframe", "trades", "net_profit", "eval_pass_probability")
    return jsonify([{k: r.get(k) for k in keep} for r in rows])
