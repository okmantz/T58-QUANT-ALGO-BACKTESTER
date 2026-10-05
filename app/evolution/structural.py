"""
Structural evolution operators (v5, B1-1).

The missing half of the search: every optimizer in this app (GA, TPE,
CMA-ES -- see app.optimize.parameter_space.GENE_KEY_RULES and
app.evolution.refinement._mutate) only ever perturbs NUMERIC leaves on
frozen templates. These operators mutate STRUCTURE -- add/remove/swap
conditions, flip AND<->OR, graft a subtree from a second parent (real
crossover, not numeric mixing), mutate filter and risk blocks.

Contract: every operator takes a Manual config dict (graft_subtree takes
two) and returns a Manual config dict that passes
app.search.grammar.validate(). Validity is re-checked after the mutation;
if the mutated config is invalid after a few retries, the operator
returns the input unchanged (never a broken config).
"""
from __future__ import annotations

import copy
import random
from typing import Any, Callable

from app.search import grammar
from app.search.grammar import (
    BOOLEAN_KINDS,
    CONNECTORS,
    NUMERIC_KINDS,
    validate,
)

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_SIDES = ("long", "short")

_MAX_VALIDATE_RETRIES = 5


def _rng(rng: random.Random | None) -> random.Random:
    return rng if rng is not None else random.Random()


def _sides_with_conditions(config: dict, block: str = "entry_conditions") -> list[str]:
    entry = config.get(block) or {}
    return [s for s in _SIDES if entry.get(s)]


def _conditions(config: dict, side: str, block: str = "entry_conditions") -> list[dict]:
    return list((config.get(block) or {}).get(side) or [])


def _set_conditions(config: dict, side: str, conditions: list[dict],
                    connectors: list[str] | None, block: str = "entry_conditions") -> None:
    entry = config.setdefault(block, {})
    entry[side] = conditions
    key = f"{side}_connectors"
    if len(conditions) >= 2:
        if connectors and len(connectors) == len(conditions) - 1:
            entry[key] = list(connectors)
        else:
            # Repair: keep the head of the old connector list, pad with AND.
            old = list(entry.get(key) or [])
            entry[key] = (old[: len(conditions) - 1]
                          + ["AND"] * max(0, len(conditions) - 1 - len(old)))
    elif key in entry:
        del entry[key]


def _try_mutations(config: dict, rng: random.Random,
                   mutate: Callable[[dict, random.Random], None]) -> dict:
    """Apply `mutate` to deep copies until one validates (else the input)."""
    for _ in range(_MAX_VALIDATE_RETRIES):
        candidate = copy.deepcopy(config)
        try:
            mutate(candidate, rng)
        except Exception:  # noqa: BLE001 -- a failed mutation attempt is just a miss
            continue
        if not validate(candidate):
            return candidate
    return config


def _numeric_kinds() -> list[str]:
    return [k for k in NUMERIC_KINDS if k in grammar.OPERAND_BUILDERS]


def _boolean_kinds() -> list[str]:
    return [k for k in BOOLEAN_KINDS if k in grammar.OPERAND_BUILDERS]


# ---------------------------------------------------------------------------
# Operators -- each takes/returns a Manual config dict.
# ---------------------------------------------------------------------------

def add_condition(config: dict, rng: random.Random | None = None) -> dict:
    """Append one random condition (grammar terminal or template-pool block
    condition) to a random side's entry list. The connector list grows by
    one random AND/OR."""
    rng = _rng(rng)

    def _mutate(cfg: dict, r: random.Random) -> None:
        side = r.choice(_SIDES)
        conds = _conditions(cfg, side)
        entry = (cfg.get("entry_conditions") or {})
        conns = list(entry.get(f"{side}_connectors") or [])
        conds.append(grammar.random_condition(r))
        conns.append(r.choice(CONNECTORS))
        _set_conditions(cfg, side, conds, conns)

    return _try_mutations(config, rng, _mutate)


def remove_condition(config: dict, rng: random.Random | None = None) -> dict:
    """Remove one random condition from a random side that has at least
    two (never empties a side's last condition -- use flip/mutate for
    shrinking to a single-condition strategy... actually a single side may
    still be emptied only if the OTHER side is non-empty, keeping the
    config valid)."""
    rng = _rng(rng)

    def _mutate(cfg: dict, r: random.Random) -> None:
        sides = _sides_with_conditions(cfg)
        r.shuffle(sides)
        for side in sides:
            conds = _conditions(cfg, side)
            other = _SIDES[1] if side == _SIDES[0] else _SIDES[0]
            if len(conds) >= 2 or (len(conds) == 1 and _conditions(cfg, other)):
                entry = (cfg.get("entry_conditions") or {})
                conns = list(entry.get(f"{side}_connectors") or [])
                idx = r.randrange(len(conds))
                del conds[idx]
                # Drop the connector adjacent to the removed condition.
                drop = min(idx, len(conns) - 1) if conns else None
                if drop is not None:
                    del conns[drop]
                _set_conditions(cfg, side, conds, conns)
                return
        raise ValueError("nothing removable")

    return _try_mutations(config, rng, _mutate)


def add_exit_condition(config: dict, rng: random.Random | None = None) -> dict:
    """Append one random condition (grammar terminal) to a random side's
    EXIT list -- the exit-logic invention the frozen templates can never
    do (their exit blocks come pre-shaped from the SkeletonSpec). The
    connector list grows by one random AND/OR. Mirrors add_condition
    exactly, against the exit_conditions block instead of
    entry_conditions."""
    rng = _rng(rng)

    def _mutate(cfg: dict, r: random.Random) -> None:
        side = r.choice(_SIDES)
        conds = _conditions(cfg, side, block="exit_conditions")
        block = (cfg.get("exit_conditions") or {})
        conns = list(block.get(f"{side}_connectors") or [])
        conds.append(grammar.random_condition(r))
        conns.append(r.choice(CONNECTORS))
        _set_conditions(cfg, side, conds, conns, block="exit_conditions")

    return _try_mutations(config, rng, _mutate)


def remove_exit_condition(config: dict, rng: random.Random | None = None) -> dict:
    """Remove one random condition from a random side's EXIT list.
    Unlike remove_condition there is no keep-one-side-non-empty guard --
    an empty exit block is valid (the builder treats it as "no
    signal-based exit"; risk_management still exits the trade), so a
    side may be emptied freely."""
    rng = _rng(rng)

    def _mutate(cfg: dict, r: random.Random) -> None:
        sides = _sides_with_conditions(cfg, block="exit_conditions")
        if not sides:
            raise ValueError("no exit conditions to remove")
        side = r.choice(sides)
        conds = _conditions(cfg, side, block="exit_conditions")
        block = (cfg.get("exit_conditions") or {})
        conns = list(block.get(f"{side}_connectors") or [])
        idx = r.randrange(len(conds))
        del conds[idx]
        # Drop the connector adjacent to the removed condition.
        drop = min(idx, len(conns) - 1) if conns else None
        if drop is not None:
            del conns[drop]
        _set_conditions(cfg, side, conds, conns, block="exit_conditions")

    return _try_mutations(config, rng, _mutate)


def swap_operand_kind(config: dict, rng: random.Random | None = None) -> dict:
    """Pick a random condition and swap one of its operands' KIND for
    another kind from the same category (numeric<->numeric,
    boolean<->boolean) -- the "indicator swap" the frozen templates can
    never do. Knobs (period/lookback/direction) are re-sampled for the
    new kind; the comparator and the other side are kept."""
    rng = _rng(rng)

    def _mutate(cfg: dict, r: random.Random) -> None:
        sides = _sides_with_conditions(cfg)
        if not sides:
            raise ValueError("no conditions")
        side = r.choice(sides)
        conds = _conditions(cfg, side)
        cond = r.choice(conds)
        for operand_key in r.sample(["left", "right"], 2):
            operand = cond.get(operand_key)
            if not isinstance(operand, dict):
                continue
            kind = str(operand.get("type", "")).lower().strip()
            if kind in _numeric_kinds():
                pool = [k for k in _numeric_kinds() if k != kind]
                category_ok = True
            elif kind in _boolean_kinds():
                pool = [k for k in _boolean_kinds() if k != kind]
                category_ok = True
            else:
                continue  # constant operand -- nothing sensible to swap to
            if not pool or not category_ok:
                continue
            new_kind = r.choice(pool)
            new_operand = grammar.OPERAND_BUILDERS[new_kind](r)
            # Keep a comparator-compatible shape: a boolean condition's
            # "is true"/"is false" only survives a boolean<->boolean swap
            # (guaranteed by the category match above); a numeric swap
            # keeps the numeric comparator.
            cond[operand_key] = new_operand
            entry = (cfg.get("entry_conditions") or {})
            _set_conditions(cfg, side, conds, list(entry.get(f"{side}_connectors") or []))
            return
        raise ValueError("no swappable operand found")

    return _try_mutations(config, rng, _mutate)


def flip_connector(config: dict, rng: random.Random | None = None) -> dict:
    """Flip one random AND<->OR connector on a random side that has at
    least two conditions."""
    rng = _rng(rng)

    def _mutate(cfg: dict, r: random.Random) -> None:
        entry = cfg.get("entry_conditions") or {}
        candidates = [
            s for s in _SIDES
            if len((entry.get(s) or [])) >= 2
        ]
        if not candidates:
            raise ValueError("no side with >= 2 conditions")
        side = r.choice(candidates)
        conds = _conditions(cfg, side)
        conns = list(entry.get(f"{side}_connectors") or [])
        # Repair path: a missing connector list is itself a mutation
        # (defaults to AND in the builder) -- materialize then flip.
        if len(conns) != len(conds) - 1:
            conns = ["AND"] * (len(conds) - 1)
        idx = r.randrange(len(conns))
        conns[idx] = "OR" if conns[idx].upper() == "AND" else "AND"
        _set_conditions(cfg, side, conds, conns)

    return _try_mutations(config, rng, _mutate)


def graft_subtree(parent_a: dict, parent_b: dict,
                  rng: random.Random | None = None) -> dict:
    """Real structural crossover: copy parent A, then replace ONE of its
    entry subtrees (a whole side's condition list, or a contiguous slice
    of it) with the corresponding subtree from parent B. Returns the
    child (a new dict; both parents untouched)."""
    rng = _rng(rng)
    child = copy.deepcopy(parent_a)

    def _mutate(cfg: dict, r: random.Random) -> None:
        sides_b = _sides_with_conditions(parent_b)
        if not sides_b:
            raise ValueError("parent B has no conditions to graft")
        side = r.choice(sides_b)
        donor_conds = copy.deepcopy(_conditions(parent_b, side))
        donor_conns = list((parent_b.get("entry_conditions") or {}).get(f"{side}_connectors") or [])
        host_conds = _conditions(cfg, side)
        if host_conds and donor_conds and r.random() < 0.5 and len(host_conds) >= 2:
            # Splice: replace a random contiguous slice of the host's
            # list with a random contiguous slice of the donor's.
            h_start = r.randrange(len(host_conds))
            h_end = r.randint(h_start + 1, len(host_conds))
            d_start = r.randrange(len(donor_conds))
            d_end = r.randint(d_start + 1, len(donor_conds))
            new_conds = host_conds[:h_start] + donor_conds[d_start:d_end] + host_conds[h_end:]
            new_conns = [r.choice(CONNECTORS) for _ in range(max(0, len(new_conds) - 1))]
        else:
            # Whole-subtree transplant: the side's full condition list
            # (and its connectors) comes from parent B.
            new_conds = donor_conds
            new_conns = donor_conns
        _set_conditions(cfg, side, new_conds, new_conns if new_conds else [])

    result = _try_mutations(child, rng, _mutate)
    # Tag provenance (harmless extra key -- the builder ignores it).
    if result is not child:
        meta = result.setdefault("evolution_meta", {})
        meta["graft_from"] = str((parent_b.get("name") or "?"))[:80]
    return result


def mutate_filter(config: dict, rng: random.Random | None = None) -> dict:
    """Add, change, or remove one filter: session time window (via a
    time_of_day entry condition -- the builder has no session filter
    block, so time gating lives in conditions), days-of-week exclusion,
    or a regime-exclusion cell."""
    rng = _rng(rng)

    def _mutate(cfg: dict, r: random.Random) -> None:
        choice = r.random()
        if choice < 0.35:
            # Toggle a session gate: add or remove a time_of_day condition.
            side = r.choice(_SIDES)
            conds = _conditions(cfg, side)
            entry = (cfg.get("entry_conditions") or {})
            conns = list(entry.get(f"{side}_connectors") or [])
            is_session_gate = [
                i for i, c in enumerate(conds)
                if isinstance(c, dict) and isinstance(c.get("left"), dict)
                and str(c["left"].get("type", "")).lower() == "time_of_day"
            ]
            if is_session_gate and r.random() < 0.5:
                idx = r.choice(is_session_gate)
                del conds[idx]
                drop = min(idx, len(conns) - 1) if conns else None
                if drop is not None:
                    del conns[drop]
            else:
                gate = {
                    "left": grammar.OPERAND_BUILDERS["time_of_day"](r),
                    "operator": "is true",
                    "right": {"type": "value", "value": 1},
                }
                conds.append(gate)
                conns.append("AND")  # a gate must conjoin, never widen
            _set_conditions(cfg, side, conds, conns)
        elif choice < 0.7:
            filters = cfg.setdefault("filters", {})
            dow = filters.setdefault("days_of_week", {})
            if dow.get("exclude") and r.random() < 0.4:
                dow["exclude"] = []
                if not dow["exclude"]:
                    del filters["days_of_week"]
                    if not filters:
                        del cfg["filters"]
            else:
                days = [5, 6] if r.random() < 0.7 else sorted(r.sample(range(7), r.randint(1, 2)))
                dow["exclude"] = days
        else:
            filters = cfg.setdefault("filters", {})
            cells = filters.setdefault("regime_exclude", [])
            if cells and r.random() < 0.4:
                del cells[r.randrange(len(cells))]
                if not cells:
                    del filters["regime_exclude"]
                    if not filters:
                        del cfg["filters"]
            else:
                dim = r.choice(list(grammar._REGIME_DIM_VALUES))
                cells.append({dim: r.choice(grammar._REGIME_DIM_VALUES[dim])})

    return _try_mutations(config, rng, _mutate)


def mutate_risk_block(config: dict, rng: random.Random | None = None) -> dict:
    """Perturb one element of the risk_management block: stop/target
    size, ATR period, trailing-stop / break-even / partial-exit toggles,
    max bars in trade, or opposite-signal-exit. Never deletes the block
    outright -- a missing block is valid, but the search learns more
    from varied risk than from absent risk."""
    rng = _rng(rng)

    def _mutate(cfg: dict, r: random.Random) -> None:
        rm = cfg.setdefault("risk_management", {})
        choice = r.random()
        if choice < 0.25:
            key = r.choice(["stop_value", "target_value"])
            cur = rm.get(key)
            try:
                cur_f = float(cur) if cur not in (None, "") else None
            except (TypeError, ValueError):
                cur_f = None
            base = cur_f if cur_f else (1.5 if key == "stop_value" else 2.5)
            rm[key] = round(base * r.uniform(0.7, 1.4), 2)
            if key == "stop_value" and not rm.get("stop_type"):
                rm["stop_type"] = "atr"
            if key == "target_value" and not rm.get("target_type"):
                rm["target_type"] = "atr"
        elif choice < 0.4:
            key = r.choice(["stop_atr_period", "target_atr_period"])
            rm[key] = r.choice([10, 14, 20, 21])
        elif choice < 0.55:
            ts = rm.setdefault("trailing_stop", {})
            if ts.get("enabled"):
                ts["enabled"] = False
            else:
                ts.update({"enabled": True, "value": round(r.uniform(1.0, 2.5), 2),
                           "atr_period": r.choice([10, 14, 20])})
        elif choice < 0.7:
            be = rm.setdefault("break_even", {})
            if be.get("enabled"):
                be["enabled"] = False
            else:
                be.update({"enabled": True, "trigger_r": round(r.uniform(0.5, 2.0), 2)})
        elif choice < 0.8:
            pe = rm.setdefault("partial_exit", {})
            if pe.get("enabled"):
                pe["enabled"] = False
            else:
                pe.update({"enabled": True, "r_multiple": round(r.uniform(1.0, 2.5), 2),
                           "fraction": round(r.uniform(0.25, 0.75), 2),
                           "move_stop_to_breakeven": True})
        elif choice < 0.9:
            if rm.get("max_bars_in_trade") and r.random() < 0.3:
                del rm["max_bars_in_trade"]
            else:
                rm["max_bars_in_trade"] = r.choice([24, 48, 96, 192])
        else:
            rm["opposite_signal_exit"] = not bool(rm.get("opposite_signal_exit", True))

    return _try_mutations(config, rng, _mutate)


# ---------------------------------------------------------------------------
# Random operator choice (used by the evolution engine's structural child
# path). Graft needs a second parent, so it is NOT in this table -- the
# engine calls graft_subtree() directly when it has two elites.
# ---------------------------------------------------------------------------

STRUCTURAL_OPERATORS: dict[str, Callable[[dict, random.Random | None], dict]] = {
    "add_condition": add_condition,
    "remove_condition": remove_condition,
    "add_exit_condition": add_exit_condition,
    "remove_exit_condition": remove_exit_condition,
    "swap_operand_kind": swap_operand_kind,
    "flip_connector": flip_connector,
    "mutate_filter": mutate_filter,
    "mutate_risk_block": mutate_risk_block,
}


def random_operator(config: dict, rng: random.Random | None = None) -> tuple[str, dict]:
    """Apply one uniformly-random structural operator. Returns
    (operator_name, mutated_config). The config is guaranteed valid --
    on total failure the input is returned unchanged (named
    "identity_fallback", which the engine treats as a miss and falls
    back to numeric mutation for that child slot)."""
    rng = _rng(rng)
    names = list(STRUCTURAL_OPERATORS)
    rng.shuffle(names)
    for name in names:
        try:
            out = STRUCTURAL_OPERATORS[name](config, rng)
        except Exception:  # noqa: BLE001
            continue
        if out is not config and not validate(out):
            return name, out
    return "identity_fallback", config
