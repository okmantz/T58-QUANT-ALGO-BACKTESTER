"""Achievements: friendly milestones driven ONLY by real activity inside the app.

Nothing here claims a strategy is profitable. Every counter is something the person actually did: backtests run,
instruments analysed, validation (robustness) reports produced, strategies saved, graveyard entries written, journal
trades logged, datasets imported. Counters are recomputed from the app's own files each time (run history, strategy
library, per-strategy report folders, graveyard files, journal, data/raw), so there is no separate event stream to
drift out of sync, and everything works offline.

State that cannot be recomputed (when each badge was first earned, whether the person has seen the unlock toast) lives in
data/achievements.json. Unlocked badges are never taken away, even if the person later deletes history.

XP: every badge gives XP (easy 10 / medium 25 / hard 60 / elite 150). XP -> level is a gentle curve with a title per level.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from app.data.storage import get_app_base_dir

_LOCK = threading.Lock()
XP_BY_TIER = {"easy": 10, "medium": 25, "hard": 60, "elite": 150}
TIER_LABEL = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "elite": "Elite"}

LEVEL_TITLES = ["Newcomer", "Apprentice", "Tinkerer", "Analyst", "Researcher", "Quant", "Strategist", "Veteran", "Principal", "Architect", "Legend"]


def level_for(xp: int) -> dict[str, Any]:
    """Level n starts at 40*n*(n+1)/2... i.e. 0, 40, 120, 240, 400, 600, ... (each level asks a little more)."""
    lvl = 0
    while 20 * (lvl + 1) * (lvl + 2) <= xp:
        lvl += 1
    start = 20 * lvl * (lvl + 1)
    nxt = 20 * (lvl + 1) * (lvl + 2)
    return {"level": lvl + 1, "title": LEVEL_TITLES[min(lvl, len(LEVEL_TITLES) - 1)], "xp": xp, "level_start": start,
            "next_at": nxt, "into": xp - start, "span": nxt - start, "pct": round(100.0 * (xp - start) / max(1, nxt - start), 1)}


# ---- catalogue ---------------------------------------------------------------------------------------------------------
# (id, category, tier, title, description, counter, target, link)
# Counters (see compute_counters): runs, strategies, instruments, timeframes, pipelines, passes, validations, validation_tools,
# library, graveyard, journal, datasets, active_days, optimizes, discoveries
_CATALOGUE: list[tuple[str, str, str, str, str, str, int, str]] = []


def _add(cat: str, counter: str, link: str, rows: list[tuple[str, str, int, str, str]]) -> None:
    for tier, title, target, desc, aid in rows:
        _CATALOGUE.append((aid, cat, tier, title, desc, counter, target, link))


_add("Testing", "runs", "/full-pipeline", [
    ("easy", "First Backtest", 1, "Run your first backtest.", "runs-1"),
    ("easy", "Warming Up", 5, "Run 5 backtests.", "runs-5"),
    ("medium", "Getting Serious", 25, "Run 25 backtests.", "runs-25"),
    ("medium", "Century Club", 100, "Run 100 backtests.", "runs-100"),
    ("hard", "Research Machine", 250, "Run 250 backtests.", "runs-250"),
    ("elite", "Thousand Tests", 1000, "Run 1,000 backtests.", "runs-1000"),
])
_add("Testing", "strategies", "/full-pipeline", [
    ("easy", "Two Ideas", 2, "Test 2 different strategies.", "strat-2"),
    ("medium", "Ten Ideas", 10, "Test 10 different strategies.", "strat-10"),
    ("medium", "Idea Factory", 50, "Test 50 different strategies.", "strat-50"),
    ("hard", "Hundred Strategies", 100, "Test 100 different strategies.", "strat-100"),
    ("elite", "Strategy Archaeologist", 500, "Test 500 different strategies.", "strat-500"),
])
_add("Markets", "instruments", "/full-pipeline", [
    ("easy", "Second Market", 2, "Analyse 2 different instruments.", "inst-2"),
    ("easy", "Five Markets", 5, "Analyse 5 different instruments.", "inst-5"),
    ("medium", "Ten Instruments", 10, "Analyse 10 different instruments.", "inst-10"),
    ("hard", "Market Explorer", 25, "Analyse 25 different instruments.", "inst-25"),
])
_add("Markets", "timeframes", "/full-pipeline", [
    ("easy", "Two Timeframes", 2, "Test on 2 different timeframes.", "tf-2"),
    ("medium", "Timeframe Sweep", 5, "Test on 5 different timeframes.", "tf-5"),
])
_add("Markets", "datasets", "/data-center", [
    ("easy", "Bring Your Own Data", 1, "Have at least 1 dataset in the Data Center.", "data-1"),
    ("medium", "Data Librarian", 10, "Have 10 datasets in the Data Center.", "data-10"),
    ("hard", "Data Hoarder", 40, "Have 40 datasets in the Data Center.", "data-40"),
])
_add("Robustness", "validations", "/validate-simple", [
    ("easy", "Trust, But Verify", 1, "Produce your first validation report.", "val-1"),
    ("medium", "Ten Robustness Checks", 10, "Produce 10 validation reports.", "val-10"),
    ("medium", "Stress Tester", 50, "Run 50 robustness checks.", "val-50"),
    ("hard", "Skeptic", 150, "Run 150 robustness checks.", "val-150"),
])
_add("Robustness", "validation_tools", "/validate-simple", [
    ("medium", "Toolbox", 3, "Use 3 different validation tools.", "vtool-3"),
    ("hard", "Full Toolbox", 6, "Use 6 different validation tools.", "vtool-6"),
])
_add("Pipeline", "pipelines", "/full-pipeline", [
    ("easy", "Full Pipeline", 1, "Send a strategy through the Full Pipeline.", "pipe-1"),
    ("medium", "Pipeline Regular", 10, "Run the Full Pipeline 10 times.", "pipe-10"),
    ("hard", "Pipeline Veteran", 50, "Run the Full Pipeline 50 times.", "pipe-50"),
])
_add("Pipeline", "passes", "/champion-simple", [
    ("easy", "First Passing Run", 1, "Get a run that passes a prop-firm evaluation.", "pass-1"),
    ("medium", "Five Passing Runs", 5, "Get 5 runs that pass a prop-firm evaluation.", "pass-5"),
    ("hard", "Consistent Passer", 25, "Get 25 runs that pass a prop-firm evaluation.", "pass-25"),
])
_add("Library", "library", "/library", [
    ("easy", "First Save", 1, "Save a strategy to your Strategy Library.", "lib-1"),
    ("medium", "Ten in the Library", 10, "Keep 10 strategies in your library.", "lib-10"),
    ("hard", "Curator", 40, "Keep 40 strategies in your library.", "lib-40"),
])
_add("Library", "graveyard", "/graveyard", [
    ("easy", "First Burial", 1, "Let the lab bury its first rejected strategy.", "grave-1"),
    ("medium", "Learning From Failure", 100, "Have 100 rejected strategies recorded (each one teaches the search what to avoid).", "grave-100"),
    ("hard", "Graveyard Shift", 1000, "Have 1,000 rejected strategies recorded.", "grave-1000"),
])
_add("Discovery", "discoveries", "/discovery-engine", [
    ("easy", "First Discovery Run", 1, "Run the Discovery Engine once.", "disc-1"),
    ("medium", "Prospector", 10, "Run the Discovery Engine 10 times.", "disc-10"),
])
_add("Habits", "journal", "/journal", [
    ("easy", "Dear Journal", 1, "Log your first trade in the Trading Journal.", "jrnl-1"),
    ("medium", "Ten Entries", 10, "Log 10 trades in the journal.", "jrnl-10"),
    ("hard", "Chronicler", 100, "Log 100 trades in the journal.", "jrnl-100"),
])
_add("Habits", "active_days", "/dashboard", [
    ("easy", "Three Days", 3, "Work in the lab on 3 different days.", "days-3"),
    ("medium", "A Month of Work", 30, "Work in the lab on 30 different days (they do not need to be in a row).", "days-30"),
])

# Late-game badges: the hard / elite rows that make the gold and platinum collectibles genuinely rare.
_add("Discovery", "discoveries", "/discovery-engine", [
    ("hard", "Discovery Veteran", 50, "Run the Discovery Engine 50 times.", "disc-50"),
    ("elite", "Discovery Legend", 200, "Run the Discovery Engine 200 times.", "disc-200"),
])
_add("Pipeline", "pipelines", "/full-pipeline", [("elite", "Pipeline Legend", 250, "Run the Full Pipeline 250 times.", "pipe-250")])
_add("Pipeline", "passes", "/champion-simple", [("elite", "Proven Passer", 100, "Get 100 runs that pass a prop-firm evaluation.", "pass-100")])
_add("Robustness", "validations", "/validate-simple", [("elite", "Relentless Skeptic", 500, "Produce 500 validation reports.", "val-500")])
_add("Library", "library", "/library", [("elite", "Grand Curator", 100, "Keep 100 strategies in your library.", "lib-100")])
_add("Habits", "journal", "/journal", [("elite", "Master Chronicler", 500, "Log 500 trades in the journal.", "jrnl-500")])
_add("Habits", "active_days", "/dashboard", [
    ("hard", "Hundred Days", 100, "Work in the lab on 100 different days.", "days-100"),
    ("elite", "A Year of Research", 365, "Work in the lab on 365 different days.", "days-365"),
])
_add("Markets", "instruments", "/full-pipeline", [("elite", "Global Explorer", 50, "Analyse 50 different instruments.", "inst-50")])

CATALOGUE = {a[0]: a for a in _CATALOGUE}


# ---- counters ----------------------------------------------------------------------------------------------------------
def _safe(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001 -- a counter that cannot be read is simply 0, never an error page
        return default


def _base_name(name: str) -> str:
    n = (name or "").strip()
    for suffix in (" (Full Pipeline)",):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.lower()


def compute_counters() -> dict[str, int]:
    base = get_app_base_dir()
    c: dict[str, int] = {k: 0 for k in ("runs", "strategies", "instruments", "timeframes", "pipelines", "passes", "validations",
                                        "validation_tools", "library", "graveyard", "journal", "datasets", "active_days",
                                        "discoveries")}

    def runs():
        from app.reports.run_history import load_runs
        rs = load_runs()
        c["runs"] = len(rs)
        c["strategies"] = len({_base_name(r.get("strategy_name", "")) for r in rs if r.get("strategy_name")})
        c["instruments"] = len({(r.get("instrument") or "").strip().lower() for r in rs if (r.get("instrument") or "Unknown") != "Unknown"})
        c["timeframes"] = len({r.get("timeframe") for r in rs if r.get("timeframe") not in (None, "", "unknown")})
        c["pipelines"] = sum(1 for r in rs if "(Full Pipeline)" in (r.get("strategy_name") or ""))
        c["passes"] = sum(1 for r in rs if r.get("single_run_passed"))
        c["active_days"] = len({str(r.get("timestamp", ""))[:10] for r in rs if r.get("timestamp")})
    _safe(runs, None)

    def val():
        root = base / "reports" / "by_strategy"
        n, tools = 0, set()
        if root.exists():
            for idx in root.glob("*/index.json"):
                try:
                    entries = json.loads(idx.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                for e in entries if isinstance(entries, list) else []:
                    t = str(e.get("tool") or "")
                    if t and t not in ("run_and_report", "full_pipeline"):
                        n += 1
                        tools.add(t)
        c["validations"], c["validation_tools"] = n, len(tools)
    _safe(val, None)

    def lib():
        from app.strategy.library import list_saved_strategies
        c["library"] = len(list_saved_strategies())
    _safe(lib, None)

    def grave():
        d = base / "data" / "evolution"
        n = 0
        if d.exists():
            for p in d.glob("strategy_graveyard__*__*.jsonl"):
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s and not s.startswith('{"candidate_id": "seed-'):
                            n += 1
        c["graveyard"] = n
    _safe(grave, None)

    def jr():
        from app.web import journal_store
        c["journal"] = len(journal_store.load().get("trades", []))
    _safe(jr, None)

    def ds():
        raw = base / "data" / "raw"
        n = 0
        if raw.exists():
            for p in raw.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".csv", ".parquet"):
                    n += 1
        c["datasets"] = n
    _safe(ds, None)

    def disc():
        d = base / "data" / "discovery_engine"
        n = 0
        if d.exists():
            n = sum(1 for p in d.glob("run_*.json"))
        c["discoveries"] = max(n, int(_load().get("counters", {}).get("discoveries", 0)))
    _safe(disc, None)
    return c


# ---- persistence --------------------------------------------------------------------------------------------------------
def state_path() -> Path:
    p = get_app_base_dir() / "data" / "achievements.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load() -> dict[str, Any]:
    try:
        d = json.loads(state_path().read_text(encoding="utf-8"))
        if isinstance(d, dict):
            d.setdefault("unlocked", {})
            d.setdefault("counters", {})
            return d
    except Exception:  # noqa: BLE001
        pass
    return {"unlocked": {}, "counters": {}}


def _save(d: dict[str, Any]) -> None:
    p = state_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    tmp.replace(p)


def bump(counter: str, n: int = 1) -> None:
    """Record an activity that leaves no file of its own (e.g. a Discovery Engine run). Best-effort, never raises."""
    try:
        with _LOCK:
            d = _load()
            d["counters"][counter] = int(d["counters"].get(counter, 0)) + n
            _save(d)
    except Exception:  # noqa: BLE001
        pass


def _owner_mode() -> bool:
    """True when this device is signed in with the owner's master key or verified owner account: every reward is shown unlocked so backgrounds, themes,
    titles and badges can all be tried. It is an overlay only -- nothing is written to achievements.json."""
    try:
        from app.licensing import client
        return bool(client.full_access_active())
    except Exception:  # noqa: BLE001
        return False


def evaluate(persist: bool = True) -> dict[str, Any]:
    """Recompute counters, record new unlocks, and return the whole page model."""
    counters = compute_counters()
    with _LOCK:
        state = _load()
        for k, v in state.get("counters", {}).items():
            if k in counters and k == "discoveries":
                counters[k] = max(counters[k], int(v))
        now = time.time()
        newly = []
        for aid, (_i, cat, tier, title, desc, counter, target, link) in CATALOGUE.items():
            if counters.get(counter, 0) >= target and aid not in state["unlocked"]:
                state["unlocked"][aid] = {"at": now, "seen": False}
                newly.append(aid)
        if persist and newly:
            _save(state)
        if _owner_mode():
            state = {**state, "unlocked": {aid: {"at": (state["unlocked"].get(aid) or {}).get("at", now), "seen": True}
                                           for aid in CATALOGUE}}
    items = []
    xp = 0
    for aid, (_i, cat, tier, title, desc, counter, target, link) in CATALOGUE.items():
        have = counters.get(counter, 0)
        u = state["unlocked"].get(aid)
        if u:
            xp += XP_BY_TIER[tier]
        items.append({"id": aid, "category": cat, "tier": tier, "tier_label": TIER_LABEL[tier], "title": title, "description": desc,
                      "counter": counter, "have": min(have, target) if not u else target, "raw": have, "target": target,
                      "pct": 100.0 if u else round(100.0 * min(have, target) / target, 1), "unlocked": bool(u),
                      "unlocked_at": (u or {}).get("at"), "new": bool(u and not u.get("seen")), "xp": XP_BY_TIER[tier], "link": link})
    cats: dict[str, dict[str, int]] = {}
    for it in items:
        c = cats.setdefault(it["category"], {"done": 0, "total": 0})
        c["total"] += 1
        c["done"] += 1 if it["unlocked"] else 0
    # "Next up": the closest unfinished badge per category (highest pct first) -> the task cards
    todo = [i for i in items if not i["unlocked"]]
    next_up: list[dict[str, Any]] = []
    seen_cat: set[str] = set()
    for it in sorted(todo, key=lambda i: (-i["pct"], i["target"])):
        if it["category"] in seen_cat:
            continue
        seen_cat.add(it["category"])
        next_up.append(it)
    next_up.sort(key=lambda i: -i["pct"])
    return {"items": items, "counters": counters, "xp": xp, "level": level_for(xp), "categories": cats,
            "unlocked": sum(1 for i in items if i["unlocked"]), "total": len(items), "next_up": next_up[:6],
            "new": [i for i in items if i["new"]]}


def acknowledge(ids: list[str] | None = None) -> None:
    with _LOCK:
        d = _load()
        for aid, u in d["unlocked"].items():
            if ids is None or aid in ids:
                u["seen"] = True
        _save(d)


# ---- research rank, titles, collectible badges, unlockables --------------------------------------------------------------
# Rank = the level's metal. It is shown in Account settings (and on shared cards / certificates), nowhere else.
RANK_METALS = [(0, "bronze", "Bronze"), (2, "silver", "Silver"), (4, "gold", "Gold"), (6, "platinum", "Platinum"), (9, "obsidian", "Obsidian")]
TIER_METAL = {"easy": "bronze", "medium": "silver", "hard": "gold", "elite": "platinum"}
METAL_LABEL = {"bronze": "Bronze", "silver": "Silver", "gold": "Gold", "platinum": "Platinum", "obsidian": "Obsidian"}


def rank_for(level_index: int) -> dict[str, str]:
    """level_index is 0-based (level 1 -> 0)."""
    metal, label = RANK_METALS[0][1], RANK_METALS[0][2]
    for at, m, lab in RANK_METALS:
        if level_index >= at:
            metal, label = m, lab
    return {"metal": metal, "metal_label": label}


# Special titles: each is earned by mastering one feature (one achievement id). (id, title, achievement id, how it reads on the lock)
TITLES: list[tuple[str, str, str]] = [
    ("pipeline-master", "Pipeline Master", "pipe-50"),
    ("pipeline-legend", "Gatekeeper of the Pipeline", "pipe-250"),
    ("validator-prime", "Validator Prime", "vtool-6"),
    ("skeptic", "The Skeptic", "val-150"),
    ("graveyard-keeper", "Graveyard Keeper", "grave-1000"),
    ("data-alchemist", "Data Alchemist", "data-40"),
    ("prospector-general", "Prospector General", "disc-50"),
    ("curator", "Grand Curator", "lib-40"),
    ("chronicler", "Chronicler of the Tape", "jrnl-100"),
    ("market-explorer", "Cartographer of Markets", "inst-25"),
    ("consistent-passer", "Eval Slayer", "pass-25"),
    ("thousand-tests", "Thousand-Test Veteran", "runs-1000"),
]
TITLE_BY_ID = {t[0]: t for t in TITLES}

# Unlockable backgrounds and whole-interface themes. "ach" = every listed badge must be unlocked; "level" = minimum level.
BACKGROUND_UNLOCKS: dict[str, dict[str, Any]] = {
    "obsidian-gold": {"label": "Obsidian Gold", "ach": ["runs-250", "val-50"]},
    "sakura-reverie": {"label": "Sakura Reverie", "ach": ["jrnl-10", "lib-10"]},
    "abyssal-blue": {"label": "Abyssal Blue", "ach": ["grave-100", "disc-10"]},
    # 2026-10-30
    "endless-sands": {"label": "Endless Sands", "ach": ["days-30", "pipe-50"]},
    "coastal-italy": {"label": "Coastal Italy", "ach": ["inst-25", "pass-5"]},
    "bioluminescence": {"label": "Bioluminescence", "ach": ["val-150", "strat-100"]},
}
# 2026-10-30: Workspace presets (Account -> Personalize -> Workspace presets): whole-workspace looks, each an earned reward.
WORKSPACE_UNLOCKS: dict[str, dict[str, Any]] = {
    "obsidian": {"label": "Obsidian", "ach": ["strat-10", "runs-25"]},
    "evergreen": {"label": "Evergreen", "ach": ["pipe-10", "lib-10"]},
    "nordic": {"label": "Nordic", "ach": ["jrnl-100", "lib-40"]},
    "cyberpunk": {"label": "Cyberpunk", "ach": ["disc-50", "pass-5"]},
    "sovereign": {"label": "Sovereign", "ach": ["pass-25", "val-150"]},
}
THEME_UNLOCKS: dict[str, dict[str, Any]] = {
    "obsidian-gold": {"label": "Obsidian Gold", "level": 0, "ach": []},     # free for everyone from the first launch (a preview of what the rest look like)
    "midnight-azure": {"label": "Midnight Azure", "ach": ["vtool-3", "val-10"]},
    "terminal-green": {"label": "Terminal Green", "ach": ["pipe-10", "pass-1"]},
    "deep-space": {"label": "Deep Space", "ach": ["inst-10", "tf-5"]},
    "crimson-protocol": {"label": "Crimson Protocol", "ach": ["pass-5", "strat-50"]},
}


def _requirement(spec: dict[str, Any], unlocked: dict[str, Any], level: int) -> dict[str, Any]:
    parts, ok = [], True
    for aid in spec.get("ach", []):
        row = CATALOGUE.get(aid)
        have = aid in unlocked
        ok = ok and have
        if row:
            parts.append({"id": aid, "title": row[3], "description": row[4], "done": have})
    need_level = int(spec.get("level") or 0)
    lvl_ok = level >= need_level
    ok = ok and lvl_ok
    bits = [f"{p['title']} ({p['description'].rstrip('.')})" for p in parts]
    if need_level:
        bits.append(f"reach level {need_level} ({LEVEL_TITLES[min(need_level - 1, len(LEVEL_TITLES) - 1)]})")
    return {"unlocked": ok, "parts": parts, "level_needed": need_level, "level_ok": lvl_ok,
            "text": "To unlock: " + (" and ".join(bits) if bits else "available now") + "."}


def _read_unlocked() -> tuple[dict[str, Any], int]:
    """Cheap: reads the saved unlock record only (no file scans), for hot paths like the page-head bootstrap."""
    if _owner_mode():
        u = {aid: {"at": 0, "seen": True} for aid in CATALOGUE}
    else:
        u = _load().get("unlocked", {})
    xp = sum(XP_BY_TIER[CATALOGUE[a][2]] for a in u if a in CATALOGUE)
    return u, level_for(xp)["level"]


def unlock_state(fresh: bool = False) -> dict[str, Any]:
    if fresh:
        evaluate()
    unlocked, level = _read_unlocked()
    bgs = {k: {"label": v["label"], **_requirement(v, unlocked, level)} for k, v in BACKGROUND_UNLOCKS.items()}
    themes = {k: {"label": v["label"], **_requirement(v, unlocked, level)} for k, v in THEME_UNLOCKS.items()}
    workspaces = {k: {"label": v["label"], **_requirement(v, unlocked, level)} for k, v in WORKSPACE_UNLOCKS.items()}
    titles = {}
    for tid, name, aid in TITLES:
        row = CATALOGUE.get(aid)
        titles[tid] = {"title": name, "unlocked": aid in unlocked, "needs": (row[3] + " (" + row[4].rstrip(".") + ")") if row else aid}
    return {"backgrounds": bgs, "themes": themes, "workspaces": workspaces, "titles": titles, "level": level,
            "locked_workspaces": [k for k, v in workspaces.items() if not v["unlocked"]],
            "locked_backgrounds": [k for k, v in bgs.items() if not v["unlocked"]],
            "locked_themes": [k for k, v in themes.items() if not v["unlocked"]]}


def profile(model: dict[str, Any] | None = None, prefs: dict[str, str] | None = None) -> dict[str, Any]:
    """Everything the Account settings 'Research profile' card shows: rank, equipped title, showcase, collectibles, unlocks."""
    model = model or evaluate()
    if prefs is None:
        try:
            from app.persistence import ui_prefs
            prefs = ui_prefs.load()
        except Exception:  # noqa: BLE001
            prefs = {}
    L = model["level"]
    rank = rank_for(L["level"] - 1)
    items = {i["id"]: i for i in model["items"]}
    st = unlock_state()
    equipped = prefs.get("t58-title", "")
    if equipped not in st["titles"] or not st["titles"][equipped]["unlocked"]:
        equipped = ""
    try:
        picked = json.loads(prefs.get("t58-showcase", "[]"))
    except Exception:  # noqa: BLE001
        picked = []
    picked = [a for a in picked if a in items and items[a]["unlocked"]][:3]
    if not picked:       # default showcase: the three rarest unlocked badges, best first
        order = {"elite": 3, "hard": 2, "medium": 1, "easy": 0}
        best = sorted((i for i in items.values() if i["unlocked"]), key=lambda i: (-order[i["tier"]], -(i.get("unlocked_at") or 0)))
        picked = [i["id"] for i in best[:3]]
    def badge(i: dict[str, Any]) -> dict[str, Any]:
        return {"id": i["id"], "title": i["title"], "description": i["description"], "tier": i["tier"], "metal": TIER_METAL[i["tier"]],
                "metal_label": METAL_LABEL[TIER_METAL[i["tier"]]], "rare": i["tier"] == "elite", "category": i["category"],
                "unlocked": i["unlocked"], "unlocked_at": i.get("unlocked_at")}
    collection: dict[str, int] = {}
    for i in items.values():
        if i["unlocked"]:
            m = TIER_METAL[i["tier"]]
            collection[m] = collection.get(m, 0) + 1
    return {"level": L, "rank": rank, "xp": model["xp"], "title": st["titles"].get(equipped, {}).get("title", ""), "title_id": equipped,
            "titles": [{"id": k, **v} for k, v in st["titles"].items()],
            "showcase": [badge(items[a]) for a in picked], "showcase_ids": picked,
            "collection": collection, "badges": [badge(i) for i in items.values() if i["unlocked"]],
            "unlocked": model["unlocked"], "total": model["total"], "unlocks": st}
