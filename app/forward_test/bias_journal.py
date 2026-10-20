"""Discretionary-bias journal (report 2026-10-20, section 5: macro analysis as a rules bridge, not a standalone predictor).

Before the open each day, log ONE line: bias (bullish / neutral / bearish) and an event-risk flag (0/1). Entries are append-only and
immutable: a date can be written once, and only before that day's cash open (``lock_hour_et`` default 09:30). After 100-150 logged
days ``compare`` answers the only question that matters: does the strategy do better when its trade direction AGREES with your
bias than when it conflicts or you were neutral? If not, the bias is a story, not an edge, and stays out of the rules.
"""
from __future__ import annotations

import json
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "reports" / "bias_journal.jsonl"
BIASES = {"bullish": 1, "neutral": 0, "bearish": -1}


class JournalLocked(Exception):
    pass


def _rows(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def log_bias(day, bias: str, event_risk: bool = False, note: str = "", *, now: datetime | None = None, lock_time: dtime = dtime(9, 30),
             path: Path | None = None) -> dict:
    """Write one day's entry. ``now`` is US/Eastern wall-clock (defaults to the machine clock - run it on Eastern time or pass it)."""
    b = str(bias).strip().lower()
    if b not in BIASES:
        raise ValueError(f"bias must be one of {sorted(BIASES)}")
    d = pd.Timestamp(day).normalize()
    now = now or datetime.now()
    if pd.Timestamp(now).normalize() > d or (pd.Timestamp(now).normalize() == d and now.time() >= lock_time):
        raise JournalLocked(f"{d.date()} can no longer be logged: entries must be written before {lock_time.strftime('%H:%M')} ET that day")
    p = Path(path) if path else DEFAULT_PATH
    if any(r["date"] == d.date().isoformat() for r in _rows(p)):
        raise JournalLocked(f"{d.date()} is already logged and entries are immutable")
    row = {"date": d.date().isoformat(), "bias": b, "event_risk": int(bool(event_risk)), "note": note, "logged_at": pd.Timestamp(now).isoformat()}
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load_journal(path: Path | None = None) -> pd.DataFrame:
    rows = _rows(Path(path) if path else DEFAULT_PATH)
    if not rows:
        return pd.DataFrame(columns=["date", "bias", "event_risk", "bias_sign"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["bias_sign"] = df["bias"].map(BIASES)
    return df


def compare(trades, journal: pd.DataFrame | None = None, min_days: int = 100, n_boot: int = 2000, seed: int = 31) -> dict:
    """Mean R when the trade direction agrees with the logged bias vs. conflicts/neutral, with a day-block bootstrap of the difference."""
    from app.search.robustness import _r_and_days
    j = load_journal() if journal is None else journal
    if len(j) < min_days:
        return {"status": "INSUFFICIENT", "logged_days": int(len(j)), "min_days": min_days,
                "message": f"{len(j)} logged day(s); need {min_days}+ before comparing (target 100-150)"}
    rows = []
    rs, days = _r_and_days(trades)
    dir_by_trade = [int(getattr(t, "direction", 0)) for t in trades if getattr(t, "initial_risk", None) and getattr(t, "size", None) and getattr(t, "pnl", None) is not None and t.initial_risk > 0]
    for r, d, direction in zip(rs, days, dir_by_trade):
        rows.append((pd.Timestamp(d).normalize(), r, direction))
    t = pd.DataFrame(rows, columns=["date", "r", "dir"]).merge(j[["date", "bias_sign", "event_risk"]], on="date", how="inner")
    if t.empty:
        return {"status": "NO_OVERLAP", "logged_days": int(len(j))}
    t["agree"] = (t["dir"] * t["bias_sign"] > 0)
    a, c = t[t["agree"]], t[~t["agree"]]
    if len(a) < 10 or len(c) < 10:
        return {"status": "INSUFFICIENT", "agree_trades": int(len(a)), "other_trades": int(len(c))}
    rng = np.random.default_rng(seed)
    udays = t["date"].unique()
    idx = {d: np.where(t["date"].to_numpy() == d)[0] for d in udays}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(udays), len(udays))
        ii = np.concatenate([idx[udays[k]] for k in pick])
        sub = t.iloc[ii]
        sa, sc = sub[sub["agree"]]["r"], sub[~sub["agree"]]["r"]
        if len(sa) and len(sc):
            diffs.append(sa.mean() - sc.mean())
    lb, ub = (float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))) if diffs else (None, None)
    helpful = lb is not None and lb > 0
    return {"status": "OK", "logged_days": int(len(j)), "agree_trades": int(len(a)), "other_trades": int(len(c)),
            "mean_r_agree": float(a["r"].mean()), "mean_r_other": float(c["r"].mean()), "diff": float(a["r"].mean() - c["r"].mean()),
            "diff_lb90": lb, "diff_ub90": ub,
            "conclusion": ("Agreement with your bias carries a statistically positive edge: it may be bridged into a rule." if helpful
                           else "No proven edge from your bias: keep it out of the rules."),
            "event_risk_days": int(j["event_risk"].sum())}
