"""
Strategy universe map: one dot per strategy, clustered by canonical family
(trend_following, mean_reversion, breakout, ...), with symbol / timeframe /
profit / drawdown on each dot's hover card.

Data comes from three places the app already keeps:
  * Search Lab result DBs (``search_*.db`` -- every candidate at every stage),
  * the Evolution Lab tested-candidates log,
  * the Strategy Library (saved strategies + their pipeline status).

Families come from app.strategy.family_taxonomy (the same vocabulary the
Family Diversity page uses), so the map and the rest of the app agree.

Each dot carries a ``stage`` so the map shows survival, not just volume:
    rejected  -- died before the leaderboard / failed
    tested    -- passed the cheap filter
    validated -- passed deeper validation
    survivor  -- made it all the way (Stage 3 gate / ready-for-demo or live)

Layout is deterministic and computed here (not in the browser) so it is unit
testable and identical on every device: family cluster centres sit on a ring
ordered by size, and inside a cluster dots follow a golden-angle spiral with
the best-performing strategies nearest the centre.

The adapters are pure (rows in, dots out); only ``load_universe`` touches disk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from app.strategy.family_taxonomy import classify_family, classify_record, family_label

STAGES = ("rejected", "tested", "validated", "survivor")
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
MAX_DOTS = 3000


@dataclass
class Dot:
    id: str
    family: str
    symbol: str
    timeframe: str
    profit: Optional[float]
    drawdown_pct: Optional[float]
    stage: str
    source: str            # "search" | "evolution" | "library"
    score: float = 0.0     # used only to order dots inside a cluster

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "family": self.family, "family_label": family_label(self.family),
            "symbol": self.symbol, "timeframe": self.timeframe, "profit": self.profit,
            "drawdown_pct": self.drawdown_pct, "stage": self.stage, "source": self.source,
        }


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _safe_family(record: dict) -> str:
    try:
        return classify_record(record)
    except Exception:  # noqa: BLE001 -- a strange record must never break the map
        return "uncategorized"


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def dots_from_search_rows(rows: Iterable[dict], symbol: str = "", timeframe: str = "") -> list[Dot]:
    """Rows from ResultsDB.live_leaderboard (best-known row per candidate)."""
    dots: list[Dot] = []
    for row in rows:
        stats = row.get("statistics") or {}
        stage_name = str(row.get("stage") or "")
        if row.get("passed_stage3_gate"):
            stage = "survivor"
        elif stage_name == "stage3" or row.get("passed_stage2"):
            stage = "validated"
        elif row.get("passed_stage1"):
            stage = "tested"
        else:
            stage = "rejected"
        score = _num(row.get("composite_score"))
        if score is None:
            score = _num(row.get("fitness"))
        if score is None:
            score = _num(row.get("quick_score")) or 0.0
        dots.append(Dot(
            id=str(row.get("candidate_id", "")), family=_safe_family(row),
            symbol=symbol or "?", timeframe=timeframe or "?",
            profit=_num(stats.get("net_profit")), drawdown_pct=_num(stats.get("max_drawdown_pct")),
            stage=stage, source="search", score=score,
        ))
    return dots


def dots_from_evolution_rows(rows: Iterable[dict], symbol: str = "", timeframe: str = "") -> list[Dot]:
    """Rows from app.evolution.checkpoint.read_tested_rows."""
    dots: list[Dot] = []
    for row in rows:
        family_name = str(row.get("family") or "")
        try:
            family = classify_family(skeleton_family=family_name)
        except Exception:  # noqa: BLE001
            family = "uncategorized"
        if not row.get("passed"):
            stage = "rejected"
        elif row.get("stage") == "full_eval":
            stage = "validated"
        else:
            stage = "tested"
        dots.append(Dot(
            id=str(row.get("candidate_id", "")), family=family,
            symbol=symbol or "?", timeframe=timeframe or "?",
            profit=_num(row.get("net_profit")), drawdown_pct=_num(row.get("max_drawdown_pct")),
            stage=stage, source="evolution", score=_num(row.get("fitness_score")) or 0.0,
        ))
    return dots


_LIBRARY_STAGE = {
    "draft": "tested", "tested_failed": "rejected", "tested_passed": "validated",
    "validated": "validated", "ready_for_demo": "survivor", "ready_for_live": "survivor",
}


def dots_from_library(strategies: Iterable[Any]) -> list[Dot]:
    """Objects shaped like app.strategy.library.StoredStrategy
    (``name``, ``status``, ``metadata``)."""
    dots: list[Dot] = []
    for s in strategies:
        meta = getattr(s, "metadata", None) or {}
        last_run = meta.get("last_run") or {}
        text = " ".join([str(getattr(s, "name", "")), str(meta.get("description", "")), " ".join(meta.get("tags") or [])])
        try:
            family = classify_family(raw_text=text)
        except Exception:  # noqa: BLE001
            family = "uncategorized"
        dots.append(Dot(
            id=str(getattr(s, "name", "")), family=family,
            symbol=str(meta.get("market") or "?"), timeframe=str(meta.get("timeframe") or "?"),
            profit=_num(last_run.get("net_profit")), drawdown_pct=_num(last_run.get("max_drawdown_pct")),
            stage=_LIBRARY_STAGE.get(str(getattr(s, "status", "draft")), "tested"),
            source="library", score=_num(last_run.get("net_profit")) or 0.0,
        ))
    return dots


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout_universe(dots: list[Dot], max_dots: int = MAX_DOTS) -> dict[str, Any]:
    """Positions every dot in a unit square (x, y in [-1, 1]).

    If there are more than ``max_dots`` the most interesting are kept: higher
    stage first, then higher score -- so a huge history never turns the map
    into an unreadable, slow blob (``truncated`` tells the UI it happened).
    """
    total = len(dots)
    stage_rank = {s: i for i, s in enumerate(STAGES)}
    kept = sorted(dots, key=lambda d: (stage_rank.get(d.stage, 0), d.score), reverse=True)[:max_dots]

    by_family: dict[str, list[Dot]] = {}
    for d in kept:
        by_family.setdefault(d.family, []).append(d)
    families = sorted(by_family, key=lambda f: (-len(by_family[f]), f))

    n_fam = len(families)
    ring = 0.0 if n_fam <= 1 else 0.62
    clusters: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    largest = max((len(v) for v in by_family.values()), default=1)
    # spiral spacing chosen so the largest cluster stays inside its radius
    # Two neighbouring clusters on the ring must not overlap: their centres are
    # 2*ring*sin(pi/n) apart, so cap each radius just under half of that. Without
    # the cap, 16 populated families (the whole taxonomy) collided.
    max_cluster_radius = min(0.34, 0.85 * ring * math.sin(math.pi / n_fam)) if n_fam > 1 else 0.9
    for i, fam in enumerate(families):
        angle = 2 * math.pi * i / n_fam - math.pi / 2 if n_fam else 0.0
        cx, cy = ring * math.cos(angle), ring * math.sin(angle)
        members = sorted(by_family[fam], key=lambda d: d.score, reverse=True)
        radius = max_cluster_radius * math.sqrt(len(members) / largest)
        radius = max(radius, min(0.06, max_cluster_radius))
        step = radius / math.sqrt(max(len(members), 1))
        for j, d in enumerate(members):
            r = step * math.sqrt(j + 0.5)
            theta = j * GOLDEN_ANGLE
            item = d.to_dict()
            item["x"], item["y"] = round(cx + r * math.cos(theta), 4), round(cy + r * math.sin(theta), 4)
            out.append(item)
        clusters.append({
            "family": fam, "label": family_label(fam), "count": len(members),
            "survivors": sum(1 for d in members if d.stage == "survivor"),
            "cx": round(cx, 4), "cy": round(cy, 4), "r": round(radius, 4),
        })
    return {"dots": out, "clusters": clusters, "total": total, "shown": len(out), "truncated": total > len(out)}


# ---------------------------------------------------------------------------
# Loading (IO)
# ---------------------------------------------------------------------------

def load_search_dots(search_dir: Path, max_runs: int = 6, per_run: int = 1500) -> list[Dot]:
    """Dots from the most recent Search Lab result DBs under ``search_dir``.
    A corrupt or half-written DB is skipped, never fatal."""
    from app.search.results_db import ResultsDB

    dots: list[Dot] = []
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:  # deleted (e.g. pruned) between the glob and the stat
            return 0.0

    files = sorted(Path(search_dir).glob("search_*.db"), key=_mtime, reverse=True)
    for db_file in files[:max_runs]:
        try:
            with ResultsDB(db_file) as db:
                for run in db.list_runs(limit=1):
                    rows = db.live_leaderboard(run["run_id"], top_n=per_run)
                    dots.extend(dots_from_search_rows(
                        rows, symbol=str(run.get("instrument") or ""), timeframe=str(run.get("timeframe") or ""),
                    ))
        except Exception:  # noqa: BLE001
            continue
    return dots


def load_evolution_dots(limit: int = 1500) -> list[Dot]:
    from app.evolution import checkpoint as evo_checkpoint

    try:
        return dots_from_evolution_rows(evo_checkpoint.read_tested_rows(limit=limit))
    except Exception:  # noqa: BLE001
        return []


def load_library_dots() -> list[Dot]:
    from app.strategy.library import list_saved_strategies

    try:
        return dots_from_library(list_saved_strategies())
    except Exception:  # noqa: BLE001
        return []


def load_universe(search_dir: Path, sources: Iterable[str] = ("search", "evolution", "library")) -> dict[str, Any]:
    wanted = set(sources)
    dots: list[Dot] = []
    if "search" in wanted:
        dots += load_search_dots(search_dir)
    if "evolution" in wanted:
        dots += load_evolution_dots()
    if "library" in wanted:
        dots += load_library_dots()
    result = layout_universe(dots)
    result["sources"] = sorted(wanted)
    return result
