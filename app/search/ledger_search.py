"""Searched-variant memory for the family-grid search backends (Search Lab, Speed Run, Discovery methods that call generate_search_space).

``generate_search_space`` draws a seeded random sample of each family's parameter grid, so the same seed re-draws the same sample and the
search lands on the same strategy again. ``fresh_space`` fixes that using the ledger (app.search.variant_ledger):

  1. over-sample the grid, then drop every variant an earlier run on this market already tested  -> never re-test the same variant
  2. top up to ~20% of the budget with ONE-STEP NEIGHBOURS of the best past variants               -> "find better variants" around what worked
  3. cut to the requested budget

and ``record_outcomes`` writes what this run tested (and how far each variant got) back to the ledger.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from app.search.variant_ledger import VariantLedger, stable_key

WARM_SHARE = 0.20


def scope_for(label: str) -> str:
    """The market a dataset label refers to ('ES1!', 'NQ', ...), same rule the Discovery Lab page uses."""
    try:
        from app.data.instrument_specs import guess_any_instrument_symbol
        sym = guess_any_instrument_symbol(label)
    except Exception:  # noqa: BLE001
        sym = None
    return sym or re.sub(r"\W+", "", str(label).split("/")[0]).upper() or "MARKET"


def engine_key(timeframe: str, family: str, candidate_id: str) -> str:
    return stable_key("engine", str(timeframe), family, candidate_id)


def _digest(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _neighbour_params(spec, params: dict) -> list[dict]:
    out = []
    for k, vals in spec.param_grid.items():
        if k not in params or params[k] not in vals or len(vals) < 2:
            continue
        i = vals.index(params[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                q = dict(params)
                q[k] = vals[j]
                try:
                    if spec.valid(q):
                        out.append(q)
                except Exception:  # noqa: BLE001
                    continue
    return out


def fresh_space(space, led: VariantLedger, scope: str, timeframe: str, want: int, *, warm_share: float = WARM_SHARE) -> dict:
    """Edit ``space`` in place to hold at most ``want`` candidates, none of them already in the ledger. Returns an info dict."""
    from app.search.strategy_space import FAMILIES
    before = len(space.candidates)
    fresh_ids = [cid for cid in space.candidates
                 if not led.is_known(scope, engine_key(timeframe, (space.meta.get(cid) or {}).get("family", ""), cid))]
    skipped = before - len(fresh_ids)
    led.skipped_known += skipped

    # warm start: one-step neighbours of the best past variants that are not known yet and not already drawn
    warm_ids, warm_built = [], {}
    quota = int(want * warm_share)
    if quota > 0:
        fams = {(space.meta.get(c) or {}).get("family") for c in space.candidates} - {None}
        have = set(space.candidates)
        for row in led.top_performers(scope, families=list(fams) or None, limit=30):
            if row.get("source") != "engine":
                continue
            sp = row.get("spec") or {}
            fam, params = sp.get("family"), sp.get("params")
            spec = FAMILIES.get(fam)
            if not spec or not isinstance(params, dict):
                continue
            for q in _neighbour_params(spec, params):
                cid = f"{fam}-{_digest(q)}"
                if cid in have or cid in warm_built or led.is_known(scope, engine_key(timeframe, fam, cid)):
                    continue
                try:
                    cfg = spec.build(q)
                except Exception:  # noqa: BLE001
                    continue
                warm_built[cid] = ({"source_type": "manual", "config": cfg}, {"family": fam, "params": q})
                warm_ids.append(cid)
                if len(warm_ids) >= quota:
                    break
            if len(warm_ids) >= quota:
                break
    led.warm_started += len(warm_ids)

    keep = warm_ids + [c for c in fresh_ids if c not in warm_built][: max(0, want - len(warm_ids))]
    cands, meta = {}, {}
    for cid in keep:
        if cid in warm_built:
            cands[cid], meta[cid] = warm_built[cid]
        else:
            cands[cid], meta[cid] = space.candidates[cid], space.meta[cid]
    space.candidates, space.meta = cands, meta
    return {"drawn": before, "skipped_known": skipped, "warm_neighbours": len(warm_ids), "kept": len(cands)}


def record_outcomes(led: VariantLedger, scope: str, timeframe: str, db_path, run_id: str, space) -> int:
    """Write every candidate the run evaluated to the ledger with how far it got. Returns rows recorded."""
    n = 0
    try:
        con = sqlite3.connect(str(db_path))
        try:
            rows = con.execute(
                "SELECT candidate_id, MAX(passed_stage1), MAX(passed_stage2), MAX(passed_stage3_gate), MAX(fitness), MAX(composite_score) "
                "FROM candidates WHERE run_id=? GROUP BY candidate_id", (run_id,)).fetchall()
        finally:
            con.close()
    except Exception:  # noqa: BLE001 -- bookkeeping must never fail a search
        return 0
    for cid, p1, p2, p3, fit, comp in rows:
        m = space.meta.get(cid) or {}
        fam = m.get("family", "")
        stage = "validated" if (p2 or p3) else ("train_pass" if p1 else "searched")
        score = fit if fit is not None else comp
        led.record(scope, engine_key(timeframe, fam, cid), source="engine", family=fam, bar_minutes=0,
                   spec={"family": fam, "params": m.get("params")}, label=f"{fam} {json.dumps(m.get('params'), sort_keys=True)}"[:300],
                   stage=stage, fit=(float(score) if score is not None else None), sig=engine_key(timeframe, fam, cid))
        n += 1
    return n
