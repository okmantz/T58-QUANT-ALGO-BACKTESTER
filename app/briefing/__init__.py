"""T58 Morning Briefing -- a short research digest built ONLY from results the app actually saved.

Every line in the briefing is computed from one of these records, and each section lists the evidence it came from:

  * data/run_history.json                         one row per backtest / Full Pipeline run
  * reports/by_strategy/*/index.json              one row per validation / optimize / pipeline report (tool, passed, summary, time)
  * Champion Board                                promotion stage, verdict and eval/payout/OOS per saved strategy
  * data/evolution/strategy_graveyard__*.jsonl    rejected strategies (counts and their recorded primary failure)
  * data/data_manifest.json                       data-quality flags per dataset file

A section with no supporting record is simply left out -- the briefing never pads, guesses or invents. If nothing at all has
happened, it says so in one line.

Sessions: a new session starts when the app is opened after 3+ hours of silence. "Since your last session" diffs the current state
against a snapshot taken at the end of the previous one. Frequency (daily / weekly / on startup / off) only decides when the
briefing pops open by itself; the Brief tab always shows the latest.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

SESSION_GAP_S = 3 * 3600
FREQUENCIES = ("daily", "weekly", "startup", "off")
DEFAULTS = {"frequency": "daily"}


# ------------------------------------------------------------------------------------------------------------------ state
def _base() -> Path:
    from app.data.storage import get_app_base_dir
    return get_app_base_dir()


def _state_path() -> Path:
    return _base() / "data" / "briefing.json"


def load_state() -> dict:
    try:
        d = json.loads(_state_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_state(st: dict) -> None:
    try:
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
        tmp.replace(p)
    except Exception:  # noqa: BLE001 -- the briefing is a convenience; it must never raise into a page
        pass


def get_settings(st: Optional[dict] = None) -> dict:
    st = st if st is not None else load_state()
    s = {**DEFAULTS, **(st.get("settings") or {})}
    if s["frequency"] not in FREQUENCIES:
        s["frequency"] = DEFAULTS["frequency"]
    return s


def set_frequency(freq: str) -> dict:
    st = load_state()
    if freq in FREQUENCIES:
        st["settings"] = {**(st.get("settings") or {}), "frequency": freq}
        if freq != "off":
            st.pop("last_popped", None)      # a fresh choice shows the briefing on the next open
        save_state(st)
    return get_settings(st)


# ------------------------------------------------------------------------------------------------------------------ facts
def _ts(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def _index_entries() -> list[dict]:
    root = _base() / "reports" / "by_strategy"
    out: list[dict] = []
    if not root.exists():
        return out
    for idx in root.glob("*/index.json"):
        try:
            rows = json.loads(idx.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for e in rows if isinstance(rows, list) else []:
            if isinstance(e, dict) and e.get("recorded_at"):
                out.append(e)
    return out


def _graveyard_counts() -> dict[str, int]:
    d = _base() / "data" / "evolution"
    out: dict[str, int] = {}
    if d.exists():
        for p in d.glob("strategy_graveyard__*__*.jsonl"):
            n = 0
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s and not s.startswith('{"candidate_id": "seed-'):
                            n += 1
            except Exception:  # noqa: BLE001
                continue
            out[p.name] = n
    return out


def _graveyard_failures() -> dict[str, int]:
    d = _base() / "data" / "evolution"
    out: dict[str, int] = {}
    if d.exists():
        for p in d.glob("strategy_graveyard__*__*.jsonl"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith('{"candidate_id": "seed-'):
                            continue
                        try:
                            pf = json.loads(s).get("primary_failure")
                        except Exception:  # noqa: BLE001
                            continue
                        if pf:
                            out[pf] = out.get(pf, 0) + 1
            except Exception:  # noqa: BLE001
                continue
    return out


def snapshot() -> dict:
    """Compact picture of the saved state, used to say what changed between sessions."""
    from app.reports.run_history import load_runs
    from app.scoring import champion_board
    runs = load_runs()
    rows = champion_board.list_board()
    return {
        "at": time.time(),
        "runs": len(runs),
        "experiments": len(_index_entries()),
        "stages": {f"{r['strategy_type']}::{r['filename']}": r["promotion"]["stage_title"] for r in rows},
        "graveyard": sum(_graveyard_counts().values()),
        "library": len(rows),
    }


# ------------------------------------------------------------------------------------------------------------------ build
_TOOL_LABEL = {"full_pipeline": "Full Pipeline", "run_and_report": "backtest", "cpcv": "CPCV", "wfo": "walk-forward", "walk_forward": "walk-forward",
               "walk_forward_opt": "walk-forward", "sensitivity": "sensitivity", "wfga": "walk-forward GA", "regime_matrix": "regime matrix",
               "quick_optimize": "optimize", "refine": "refinement", "risk_sweep": "risk sweep"}


def _tool_label(t: str) -> str:
    return _TOOL_LABEL.get(str(t), str(t).replace("_", " "))


def _plural(n: int, one: str, many: Optional[str] = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def build(*, now: Optional[float] = None, since: Optional[float] = None, prev: Optional[dict] = None) -> dict:
    """The briefing body. ``since`` = start of the 'overnight' window (previous session's last activity; default 24 h)."""
    from app.guidance import next_step
    from app.reports.run_history import load_runs
    from app.scoring import champion_board

    now = now or time.time()
    since = since or (now - 24 * 3600)
    sections: list[dict] = []

    entries = _index_entries()
    runs = load_runs()
    recent_e = [e for e in entries if since <= float(e.get("recorded_at") or 0) <= now]
    recent_runs = [r for r in runs if since <= _ts(r.get("timestamp")) <= now]
    # a Full Pipeline / backtest run is mirrored into both stores; count each experiment once (entries carry the tool)
    done = recent_e if recent_e else []
    extra_runs = max(0, len(recent_runs) - sum(1 for e in recent_e if e.get("tool") in ("full_pipeline", "run_and_report")))
    n_exp = len(done) + extra_runs

    # ---- 1. experiments completed since the last session
    if n_exp:
        by_tool: dict[str, list] = {}
        for e in done:
            by_tool.setdefault(_tool_label(e.get("tool")), []).append(e)
        if extra_runs:
            by_tool.setdefault("backtest", []).extend([{"passed": r.get("single_run_passed")} for r in recent_runs[:extra_runs]])
        lines = []
        for t, es in sorted(by_tool.items(), key=lambda kv: -len(kv[1])):
            judged = [x for x in es if x.get("passed") is not None]
            p = sum(1 for x in judged if x.get("passed"))
            lines.append(f"{_plural(len(es), t)}" + (f" ({p} passed, {len(judged) - p} failed)" if judged else ""))
        names = sorted({str(e.get("strategy_name") or "").split(" -- ")[0] for e in done if e.get("strategy_name")})
        sections.append({"id": "experiments", "title": "Experiments completed",
                         "headline": f"{_plural(n_exp, 'experiment')} finished since your last session",
                         "lines": lines[:6] + ([f"Strategies touched: {', '.join(names[:4])}" + (f" +{len(names) - 4} more" if len(names) > 4 else "")] if names else []),
                         "evidence": f"{len(recent_e)} saved report(s) and {len(recent_runs)} run(s) in run history between {datetime.fromtimestamp(since):%b %d %H:%M} and now"})

    # ---- 2. recurring weakness: only when the saved record repeats itself
    weak = _recurring_weakness(entries, now)
    if weak:
        sections.append(weak)

    # ---- 3. data-quality issues recorded for the datasets in use
    dq = _data_quality(entries + [{"instrument": r.get("instrument")} for r in runs[-60:]])
    if dq:
        sections.append(dq)

    # ---- 4. suggested next step, from the board
    rows = champion_board.list_board()
    nxt = _suggest(rows, weak)
    if nxt:
        sections.append(nxt)

    # ---- 5. what changed since the last session (snapshot diff)
    chg = _changes(prev, rows)
    if chg:
        sections.append(chg)

    today0 = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    today_n = sum(1 for e in entries if float(e.get("recorded_at") or 0) >= today0) + \
        max(0, sum(1 for r in runs if _ts(r.get("timestamp")) >= today0) - sum(1 for e in entries if float(e.get("recorded_at") or 0) >= today0 and e.get("tool") in ("full_pipeline", "run_and_report")))
    return {"generated_at": now, "sections": sections, "empty": not sections, "today_experiments": today_n,
            "empty_line": ("Nothing new in your saved results since the last session. Run a test or start a Discovery search and it will show up here." if not sections else ""),
            "window_start": since}


def _recurring_weakness(entries: list[dict], now: float) -> Optional[dict]:
    from app.scoring import champion_board
    week = [e for e in entries if float(e.get("recorded_at") or 0) >= now - 14 * 86400 and e.get("passed") is not None]
    fails: dict[str, list] = {}
    for e in week:
        if e.get("passed") is False:
            fails.setdefault(_tool_label(e.get("tool")), []).append(e)
    cands: list[tuple[float, dict]] = []
    for tool, fs in fails.items():
        total = sum(1 for e in week if _tool_label(e.get("tool")) == tool)
        if len(fs) >= 3 and len(fs) / total >= 0.5:
            cands.append((len(fs), {"title": f"{tool} keeps failing", "headline": f"{tool}: failed {len(fs)} of {total} runs in the last 14 days",
                                    "lines": [f"Most recent: {str(fs[-1].get('strategy_name') or '')[:60]} -- {str(fs[-1].get('summary') or '')[:80]}"],
                                    "evidence": f"{total} saved {tool} report(s) from the last 14 days"}))
    # weakest dimension across the board
    rows = champion_board.list_board()
    dims = {"eval_pct": ("Eval pass chance", 60.0), "payout_pct": ("Payout chance", 50.0), "oos_pct": ("Out-of-sample validation", 50.0)}
    for key, (label, thr) in dims.items():
        known = [r for r in rows if r.get(key) is not None]
        low = [r for r in known if r[key] < thr]
        if len(known) >= 3 and len(low) >= 3 and len(low) / len(known) >= 0.5:
            cands.append((len(low) + 0.5, {"title": f"{label} is the weak spot", "headline": f"{label} is below {thr:.0f}% for {len(low)} of {len(known)} scored strategies",
                                           "lines": [f"Lowest: {min(low, key=lambda r: r[key])['display_name'][:50]} at {min(r[key] for r in low):.0f}%"],
                                           "evidence": f"{len(known)} strategies with a saved {label.lower()} on the Champion Board", "goal": {"eval_pct": "eval", "payout_pct": "payout", "oos_pct": "robust"}[key]}))
    gf = _graveyard_failures()
    tot = sum(gf.values())
    if tot >= 5:
        top, n = max(gf.items(), key=lambda kv: kv[1])
        if n / tot >= 0.4 and n >= 3:
            cands.append((n, {"title": "Most rejections share a cause", "headline": f"{n} of {tot} recent rejections: {top}",
                              "lines": [], "evidence": f"{tot} rejected strategies in the Graveyard with a recorded cause"}))
    if not cands:
        return None
    best = max(cands, key=lambda c: c[0])[1]
    best["id"] = "weakness"
    return best


def _data_quality(items: list[dict]) -> Optional[dict]:
    try:
        man = json.loads((_base() / "data" / "data_manifest.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    used = {str(i.get("instrument") or "").strip().lower().replace(".csv", "") for i in items if i.get("instrument")}
    used = {u.split("/")[-1] for u in used if u}
    min_years = float(man.get("min_years_for_ready") or 0)
    bad = []
    for f in man.get("files") or []:
        stem = str(f.get("file") or "").split("/")[-1].rsplit(".", 1)[0].lower()
        if not any(u and (u == stem or u in stem or stem in u) for u in used):
            continue
        flags = str(f.get("flags") or "ok")
        why = []
        if flags not in ("ok", ""):
            why.append(flags)
        if f.get("synthetic"):
            why.append("synthetic data")
        yrs = f.get("years")
        if isinstance(yrs, (int, float)) and yrs == yrs and min_years and yrs < min_years:
            why.append(f"only {yrs:.1f} years (READY needs {min_years:g})")
        if why:
            bad.append((str(f.get("file")), "; ".join(why)))
    if not bad:
        return None
    return {"id": "dataquality", "title": "Data-quality flags", "headline": f"{_plural(len(bad), 'dataset')} you have tested on carry a data warning",
            "lines": [f"{n}: {w}" for n, w in bad[:4]], "evidence": "data/data_manifest.json, matched to datasets used in your saved runs"}


def _suggest(rows: list[dict], weak: Optional[dict]) -> Optional[dict]:
    from app.guidance import next_step
    live = [r for r in rows if r["promotion"]["next_stage"] is not None]
    if not live:
        return None
    from app.scoring.champion_board import PROMOTION_STAGES
    best = max(live, key=lambda r: (PROMOTION_STAGES.index(r["promotion"]["stage"]), r["strength"] if r.get("strength") is not None else -1.0))
    plan = next_step.resolve(best, None)
    lines = [plan["why"]] + plan["steps"][:2]
    if weak and weak.get("goal") and plan["stage"] in ("weak", "failed"):
        lines.append(f"This matches the recurring weakness above ({weak['title'].lower()}).")
    return {"id": "next", "title": "Suggested next step", "headline": f"{best['display_name']}: {plan['headline']}", "lines": lines,
            "action": plan["primary"], "evidence": f"Champion Board record for {best['display_name']} (stage: {best['promotion']['stage_title']})"}


def _changes(prev: Optional[dict], rows: list[dict]) -> Optional[dict]:
    if not prev:
        return None
    cur = {f"{r['strategy_type']}::{r['filename']}": r["promotion"]["stage_title"] for r in rows}
    names = {f"{r['strategy_type']}::{r['filename']}": r["display_name"] for r in rows}
    lines = []
    for k, st in cur.items():
        old = (prev.get("stages") or {}).get(k)
        if old is None:
            lines.append(f"New in library: {names[k][:50]} ({st})")
        elif old != st:
            lines.append(f"{names[k][:50]}: {old} -> {st}")
    for k in (prev.get("stages") or {}):
        if k not in cur:
            lines.append(f"Removed from library: {k.split('::')[-1][:50]}")
    g_now = sum(_graveyard_counts().values())
    if g_now > int(prev.get("graveyard") or 0):
        lines.append(f"{_plural(g_now - int(prev.get('graveyard') or 0), 'strategy', 'strategies')} added to the Graveyard")
    if not lines:
        return None
    return {"id": "changes", "title": "Since your last session", "headline": "What changed in your library", "lines": lines[:8],
            "evidence": "comparison with the snapshot taken at the end of your last session"}


# ------------------------------------------------------------------------------------------------------------------ public
def get_briefing(*, now: Optional[float] = None, touch: bool = True) -> dict:
    """The briefing payload for the widget, rolling the session forward when the gap since last activity is >= SESSION_GAP_S."""
    now = now or time.time()
    st = load_state()
    settings = get_settings(st)
    last = float(st.get("last_activity") or 0)
    if not st.get("session_start") or (last and now - last >= SESSION_GAP_S):
        st["prev_snapshot"] = st.get("last_snapshot")          # the state when the previous session ended
        st["prev_activity"] = last or None
        st["session_start"] = now
        st["new_session"] = True
    since = float(st.get("prev_activity") or 0) or (now - 24 * 3600)
    body = build(now=now, since=since, prev=st.get("prev_snapshot"))
    freq = settings["frequency"]
    popped = float(st.get("last_popped") or 0)
    due = False
    if freq == "startup":
        due = bool(st.get("new_session"))
    elif freq == "daily":
        due = not popped or datetime.fromtimestamp(popped).date() < datetime.fromtimestamp(now).date()
    elif freq == "weekly":
        due = not popped or datetime.fromtimestamp(popped).isocalendar()[:2] != datetime.fromtimestamp(now).isocalendar()[:2]
    if body["empty"] and not st.get("prev_snapshot"):
        due = False                                              # first ever open: nothing to brief on yet
    if touch:
        st["last_activity"] = now
        try:
            st["last_snapshot"] = snapshot()
        except Exception:  # noqa: BLE001
            pass
        save_state(st)
    return {"enabled": freq != "off", "frequency": freq, "due": bool(due and freq != "off"), **body,
            "session_started": st.get("session_start")}


def mark_popped() -> None:
    st = load_state()
    st["last_popped"] = time.time()
    st["new_session"] = False
    save_state(st)
