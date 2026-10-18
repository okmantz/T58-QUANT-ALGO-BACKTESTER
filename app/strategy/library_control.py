"""
Strategy Library "Control Center" data layer.

Turns every saved strategy into one compact card row:
    instrument (ES / NQ / GC / other), status (Research / Developing / Ready / Forward Testing),
    profit factor, expectancy, max drawdown %, prop-firm pass rate, payout-simulation rate, robustness score,
    latest validation date, a mini equity curve, family, timeframe and validation status.

Numbers come from what the app already recorded (the strategy's .meta.json sidecar -- last_run / last_validation /
validation_methods_run / last_champion_check / promotion -- and app.reports.run_history rows matched by strategy name).
A value nobody has measured is None (rendered "--"), never a made-up 0.

Status mapping (the four control-center lanes):
    Forward Testing  promotion stage forward_testing/production_ready, a recorded forward test, or status ready_for_demo
    Ready            champion-check verdict READY, or status validated / ready_for_live, or promotion >= champion_candidate
    Developing       the strategy has been run or searched, but is not yet Ready
    Research         everything else (drafts, failed tests, untested ideas)

Robustness score (0-100) = mean of whatever is on record of: best walk-forward/CPCV efficiency, 100 - PBO%, and the T58
score from the last champion check. None if none of the three exist.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

STATUS_LANES = ("Research", "Developing", "Ready", "Forward Testing")
INSTRUMENT_PATTERNS = (
    ("ES", re.compile(r"(?<![A-Z])(ES|MES|SPX|SP500|S&P)(?![A-Z])", re.I)),
    ("NQ", re.compile(r"(?<![A-Z])(NQ|MNQ|NAS100|NDX|NASDAQ)(?![A-Z])", re.I)),
    ("GC", re.compile(r"(?<![A-Z])(GC|MGC|XAUUSD|XAU|GOLD)(?![A-Z])", re.I)),
)
_TF_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,3}\s?(?:s|m|min|h|d)|M\d{1,2}|H\d{1,2}|D1|W1)(?![A-Za-z0-9])", re.I)


def normalize_instrument(*texts: Any) -> str:
    for t in texts:
        s = str(t or "")
        for code, pat in INSTRUMENT_PATTERNS:
            if pat.search(s):
                return code
    for t in texts:
        s = str(t or "").strip()
        if re.fullmatch(r"[A-Za-z0-9!]{2,8}", s) and s.lower() not in ("unknown", "none", "manual", "python"):
            return s.upper()
    return "OTHER"


def normalize_timeframe(*texts: Any) -> str:
    for t in texts:
        m = _TF_RE.search(str(t or ""))
        if m:
            v = m.group(1).replace(" ", "").lower()
            if v.startswith(("m", "h")) and v[1:].isdigit():
                return v[1:] + v[0] if v[0] == "h" else v[1:] + "m"
            return {"min": "m"}.get(v, v).replace("min", "m")
    return "unknown"


def lane_for(meta: dict, status: str) -> str:
    promo = (meta.get("promotion") or {}).get("stage")
    if promo in ("forward_testing", "production_ready") or meta.get("last_forward_test") or status == "ready_for_demo":
        return "Forward Testing"
    verdict = str((meta.get("last_champion_check") or {}).get("verdict") or "").upper()
    if verdict == "READY" or status in ("validated", "ready_for_live") or promo in ("champion_candidate", "validated"):
        return "Ready"
    if meta.get("last_run") or meta.get("last_search") or meta.get("last_optimize") or status in ("tested_passed", "tested_failed"):
        return "Developing"
    return "Research"


def _f(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if x == x and abs(x) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _pct(v: Any) -> Optional[float]:
    """Accepts 0-1 or 0-100 probabilities, returns 0-100."""
    x = _f(v)
    if x is None:
        return None
    return round(x * 100.0 if 0 < x <= 1.0 else x, 1)


def robustness_inputs(meta: dict) -> dict:
    eff, pbo, t58 = [], None, None
    for m in (meta.get("validation_methods_run") or {}).values():
        e = _f((m or {}).get("efficiency"))
        if e is not None:
            eff.append(e)
        p = _f((m or {}).get("pbo"))
        if p is not None:
            pbo = p if pbo is None else min(pbo, p)
    lv = meta.get("last_validation") or {}
    if _f(lv.get("efficiency")) is not None:
        eff.append(_f(lv["efficiency"]))
    if pbo is None and _f(lv.get("pbo")) is not None:
        pbo = _f(lv["pbo"])
    t58 = _f((meta.get("last_champion_check") or {}).get("t58_score"))
    return {"best_efficiency": max(eff) if eff else None, "pbo": pbo, "t58_score": t58}


def robustness_score(meta: dict) -> Optional[float]:
    r = robustness_inputs(meta)
    parts = []
    if r["best_efficiency"] is not None:
        parts.append(max(0.0, min(100.0, r["best_efficiency"])))
    if r["pbo"] is not None:
        parts.append(max(0.0, min(100.0, 100.0 - r["pbo"])))
    if r["t58_score"] is not None:
        parts.append(max(0.0, min(100.0, r["t58_score"])))
    return round(sum(parts) / len(parts), 1) if parts else None


def validation_date(meta: dict) -> Optional[str]:
    stamps: list[float] = []
    for m in (meta.get("validation_methods_run") or {}).values():
        ts = _f((m or {}).get("recorded_at"))
        if ts:
            stamps.append(ts)
    for key in ("last_validation", "last_champion_check", "last_validate_run"):
        blk = meta.get(key) or {}
        for k in ("recorded_at", "timestamp", "time", "ts"):
            v = blk.get(k)
            if isinstance(v, (int, float)) and v > 1e9:
                stamps.append(float(v))
            elif isinstance(v, str):
                try:
                    stamps.append(datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    pass
    if not stamps:
        return None
    return datetime.fromtimestamp(max(stamps), tz=timezone.utc).date().isoformat()


def _curve(vals: Any, n: int = 40) -> list[float]:
    try:
        v = [float(x) for x in (vals or [])]
    except (TypeError, ValueError):
        return []
    if len(v) <= n:
        return [round(x, 2) for x in v]
    step = (len(v) - 1) / (n - 1)
    return [round(v[int(round(i * step))], 2) for i in range(n)]


def build_row(strategy_type: str, name: str, meta: dict, status: str, run: Optional[dict]) -> dict:
    last = meta.get("last_run") or {}
    run = run or {}
    instrument = normalize_instrument(meta.get("market"), run.get("instrument"), last.get("instrument"), name)
    timeframe = normalize_timeframe(meta.get("timeframe"), run.get("timeframe"), last.get("timeframe"), name)
    tags = list(meta.get("tags") or [])
    family = (meta.get("family") or (meta.get("last_search") or {}).get("family") or
              next((t for t in tags if t.lower() not in ("manual", "es", "nq", "gc")), None) or "uncategorised")
    pf = _f(last.get("profit_factor"))
    if pf is None:
        pf = _f(run.get("profit_factor"))
    trades = _f(last.get("trades") if last.get("trades") is not None else run.get("trades"))
    net = _f(last.get("net_profit") if last.get("net_profit") is not None else run.get("net_profit"))
    expectancy = _f(last.get("expectancy"))
    if expectancy is None and trades and net is not None and trades > 0:
        expectancy = round(net / trades, 2)
    dd = _f(last.get("max_dd_pct"))
    if dd is None:
        dd = _f(last.get("max_dd"))
    if dd is None:
        dd = _f(run.get("max_drawdown_pct"))
    pass_rate = _pct(last.get("eval_pass_probability") if last.get("eval_pass_probability") is not None else run.get("eval_pass_probability"))
    payout = _pct(last.get("first_payout_probability") if last.get("first_payout_probability") is not None else run.get("first_payout_probability"))
    vdate = validation_date(meta)
    lane = lane_for(meta, status)
    return {
        "id": f"{strategy_type}::{name}", "type": strategy_type, "name": name,
        "instrument": instrument, "timeframe": timeframe, "family": str(family), "status": lane,
        "lifecycle": status, "profit_factor": None if pf is None else round(pf, 2),
        "win_rate": _pct(last.get("win_rate") if last.get("win_rate") is not None else run.get("win_rate")),
        "expectancy": expectancy, "max_drawdown_pct": None if dd is None else round(dd, 2),
        "pass_rate": pass_rate, "payout_rate": payout,
        "robustness": robustness_score(meta), "robustness_inputs": robustness_inputs(meta),
        "validated_on": vdate, "validation_status": "Validated" if vdate else "Not validated",
        "trades": None if trades is None else int(trades), "net_profit": net,
        "curve": _curve(run.get("equity_curve") or last.get("equity_curve")),
        "t58_score": _f((meta.get("last_champion_check") or {}).get("t58_score")),
        "verdict": (meta.get("last_champion_check") or {}).get("verdict"),
    }


def control_rows() -> list[dict]:
    from app.reports import run_history
    from app.strategy.library import STRATEGY_TYPES, list_saved_strategies

    latest: dict[str, dict] = {}
    try:
        for r in run_history.load_runs():
            k = str(r.get("strategy_name", "")).strip().lower()
            if k and (k not in latest or r.get("timestamp", "") >= latest[k].get("timestamp", "")):
                latest[k] = r
    except Exception:  # noqa: BLE001
        pass
    rows = []
    for t in STRATEGY_TYPES:
        for s in list_saved_strategies(t):
            stem = s.name.rsplit(".", 1)[0].strip().lower()
            run = latest.get(stem) or latest.get(s.name.strip().lower()) or latest.get(str(s.metadata.get("display_name", "")).strip().lower())
            rows.append(build_row(t, s.name, s.metadata, s.status, run))
    return rows


def summarize(rows: list[dict]) -> dict:
    lanes = {k: 0 for k in STATUS_LANES}
    for r in rows:
        lanes[r["status"]] = lanes.get(r["status"], 0) + 1
    return {"total": len(rows), "lanes": lanes,
            "instruments": sorted({r["instrument"] for r in rows}),
            "families": sorted({r["family"] for r in rows}),
            "timeframes": sorted({r["timeframe"] for r in rows}),
            "validation": ["Validated", "Not validated"]}
