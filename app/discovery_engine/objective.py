"""One objective, one entry floor -- the single definition of "winning" for the Strategy Discovery Engine.

Before 2026-10-25 the two discovery labs disagreed on what a winner was:

    Search Lab     ranked on ``edge_lb_r`` -- the day-block-bootstrap LOWER BOUND of mean R per trade after 2x costs,
                   year-stability weighted -- and refused to look at anything with fewer than 100 trades / t < 1.
    Evolution Lab  ranked on Monte-Carlo prop-pass probability x payout x robustness / drawdown and used a 20-trade,
                   profit-factor-1.05 prefilter -- exactly the objective and floor Search Lab retired on 2026-10-20 as the
                   noisiest and most overfittable available.

Everything in this module is imported by BOTH labs (and by the engine), so the two can never drift apart again:

    ``EntryFloor`` / ``entry_floor_reasons``   the shared Stage-1 / prefilter rule (100+ trades, mean-R t >= 1, cost < 0.15R)
    ``edge_lb_r``                              the shared ranking score (a thin, tested wrapper over app.optimize.two_stage)
    ``complexity_penalty``                     parsimony charge applied inside the fitness (conditions, parameters, depth)

Prop-pass probability is NOT removed -- it moves to the LAST stage, where it answers a sizing question ("how hard may I lean on
an edge that has already been proven?") rather than choosing the edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# The ONE objective name. Both labs, the engine and the UI use this string.
EDGE_OBJECTIVE = "edge_lb_r"
# Metric used inside walk-forward / CPCV / robustness (``run_walk_forward`` already supports it).
EDGE_FOLD_METRIC = "mean_r"
LEGACY_OBJECTIVE = "eval_pass_probability"


@dataclass(frozen=True)
class EntryFloor:
    """The shared "is this worth a full evaluation?" rule.

    A mean-R t-statistic of 1 on fewer than ~100 trades is statistically noise (20 trades at sd 1.3R cannot separate +0.3R from
    zero); the cost cap rejects strategies whose round-trip cost eats more than 0.15R of the stop distance -- no signal can rescue
    those. Defaults equal ``SearchStageConfig``'s so Search Lab's behaviour is unchanged.
    """

    min_trades: int = 100
    min_t_stat: float = 1.0
    max_cost_in_r: float = 0.15
    min_profit_factor: float = 1.05
    max_drawdown_buffer_mult: float = 1.5
    require_positive_net: bool = True

    def as_filters(self) -> dict:
        """The dict shape ``app.search.batch_runner`` workers already consume."""
        return {
            "min_trades": int(self.min_trades), "min_profit_factor": float(self.min_profit_factor),
            "max_drawdown_buffer_mult": float(self.max_drawdown_buffer_mult),
            "min_t_stat": float(self.min_t_stat), "max_cost_in_r": float(self.max_cost_in_r),
        }


DEFAULT_FLOOR = EntryFloor()
# Reproduces the pre-2026-10-20 behaviour for anyone who needs to compare against old runs.
LEGACY_FLOOR = EntryFloor(min_trades=20, min_t_stat=0.0, max_cost_in_r=0.0, min_profit_factor=1.05)


def mean_r_t_stat(stats: dict) -> float:
    """Approximate t-stat of mean R from summary statistics: ``average_r * sqrt(n) / sd_r`` with sd_r the report's 1.3R.

    -inf when it cannot be measured (fewer than 2 trades / non-finite average); +inf on an unexpected stats shape so a
    surprise can never silently reject a candidate (same convention as Search Lab's original helper)."""
    try:
        from app.validation.edge_stats import DEFAULT_TRADE_SD_R
        n = float(stats.get("total_trades", 0) or 0)
        ar = float(stats.get("average_r", 0.0) or 0.0)
        if n < 2 or not math.isfinite(ar):
            return float("-inf")
        return ar * math.sqrt(n) / DEFAULT_TRADE_SD_R
    except Exception:  # noqa: BLE001
        return float("inf")


def _pf_value(stats: dict) -> float:
    pf = stats.get("profit_factor", 0.0)
    return 10.0 if pf == float("inf") else float(pf or 0.0)


def entry_floor_reasons(stats: dict, floor: EntryFloor = DEFAULT_FLOOR, *, max_drawdown_pct: float | None = None) -> list[str]:
    """Why ``stats`` fails the shared entry floor ([] = passes). Reason codes match Evolution's prefilter vocabulary
    (``min_trades`` / ``profit_factor`` / ``max_drawdown`` / ``unprofitable``) plus ``t_stat`` so triage logs stay comparable."""
    reasons: list[str] = []
    if not stats:
        return ["no_stats"]
    n = int(stats.get("total_trades", 0) or 0)
    if n < floor.min_trades:
        reasons.append("min_trades")
    if _pf_value(stats) < floor.min_profit_factor:
        reasons.append("profit_factor")
    if max_drawdown_pct is not None:
        dd = float(stats.get("max_drawdown_pct", 0.0) or 0.0)
        if dd > float(max_drawdown_pct) * floor.max_drawdown_buffer_mult:
            reasons.append("max_drawdown")
    if floor.require_positive_net and float(stats.get("net_profit", 0.0) or 0.0) <= 0:
        reasons.append("unprofitable")
    if floor.min_t_stat and mean_r_t_stat(stats) < floor.min_t_stat:
        reasons.append("t_stat")
    return reasons


def passes_entry_floor(stats: dict, floor: EntryFloor = DEFAULT_FLOOR, *, max_drawdown_pct: float | None = None) -> bool:
    return not entry_floor_reasons(stats, floor, max_drawdown_pct=max_drawdown_pct)


def edge_lb_r(trades, min_trades: int = 30, conf: float = 0.95, n_boot: int = 400) -> float:
    """The shared ranking score: lower bound of mean R per trade (day-block bootstrap, year-stability weighted).

    ``-inf`` when there are too few measurable trades. Thin wrapper so Evolution Lab, Search Lab and the engine all call one
    function (the implementation lives in ``app.optimize.two_stage`` where the GA's inner loop already uses it)."""
    from app.optimize.two_stage import edge_fitness_from_trades
    return edge_fitness_from_trades(trades, min_trades=min_trades, conf=conf, n_boot=n_boot)


# --------------------------------------------------------------------------------------------------- complexity charge
@dataclass(frozen=True)
class Complexity:
    n_conditions: int = 0
    n_parameters: int = 0
    depth: int = 0           # nesting depth of the config (a proxy for structural depth)
    n_exit_conditions: int = 0

    @property
    def score(self) -> float:
        """One number: conditions dominate (each is a structural degree of freedom), parameters count a third, depth a half."""
        return float(self.n_conditions + self.n_exit_conditions + self.n_parameters / 3.0 + self.depth / 2.0)


def _count_numeric_leaves(node, _depth: int = 0) -> tuple[int, int]:
    """(numeric leaves, max depth) of a config tree; booleans and strings are not parameters."""
    if isinstance(node, dict):
        n, d = 0, _depth
        for k, v in node.items():
            if k in ("name", "type", "operator", "direction", "field"):
                continue
            a, b = _count_numeric_leaves(v, _depth + 1)
            n, d = n + a, max(d, b)
        return n, d
    if isinstance(node, (list, tuple)):
        n, d = 0, _depth
        for v in node:
            a, b = _count_numeric_leaves(v, _depth + 1)
            n, d = n + a, max(d, b)
        return n, d
    if isinstance(node, bool):
        return 0, _depth
    if isinstance(node, (int, float)):
        return 1, _depth
    return 0, _depth


def complexity_of_spec(spec: dict | None) -> Complexity:
    """Structural complexity of a candidate spec (Manual config only; code strategies report zeros -- no structure to charge)."""
    cfg = (spec or {}).get("config") if isinstance(spec, dict) and "config" in spec else spec
    if not isinstance(cfg, dict):
        return Complexity()
    entry = cfg.get("entry_conditions") or {}
    exits = cfg.get("exit_conditions") or {}
    n_cond = sum(len(entry.get(s) or []) for s in ("long", "short"))
    n_exit = sum(len(exits.get(s) or []) for s in ("long", "short"))
    n_par, depth = _count_numeric_leaves(cfg)
    return Complexity(n_conditions=n_cond, n_parameters=n_par, depth=depth, n_exit_conditions=n_exit)


def complexity_penalty(spec: dict | None, weight: float = 0.002, free_units: float = 8.0) -> float:
    """Charge, in R, subtracted from the edge lower bound.

    ``free_units`` complexity points are free (a plain two-condition rule with a stop and a target sits below it);
    each point above costs ``weight`` R. With the default weight a 20-point monster pays ~0.024R -- small enough not to
    punish a real edge, large enough to break ties toward the simpler of two equal rules, which is what "prefer a stable,
    simple winner" asks for. Set ``weight=0`` to disable."""
    if weight <= 0:
        return 0.0
    extra = max(0.0, complexity_of_spec(spec).score - float(free_units))
    return float(weight) * extra


def penalised_edge(trades, spec: dict | None = None, weight: float = 0.002, **kw) -> float:
    """``edge_lb_r`` minus the complexity charge; -inf stays -inf (an unmeasurable candidate never improves by being simple)."""
    lb = edge_lb_r(trades, **kw)
    if not math.isfinite(lb):
        return lb
    return float(lb - complexity_penalty(spec, weight))


@dataclass
class EdgeScore:
    """Everything the engine records about one candidate's edge, so rankings are auditable."""

    edge_lb_r: float = float("-inf")
    penalised: float = float("-inf")
    mean_r: float | None = None
    n_trades: int = 0
    t_stat: float | None = None
    complexity: float = 0.0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def score_edge(trades, spec: dict | None = None, *, complexity_weight: float = 0.002) -> EdgeScore:
    """Full edge scorecard for one candidate's trades."""
    from app.optimize.two_stage import trade_r_and_days
    rs, days = trade_r_and_days(trades)
    out = EdgeScore(n_trades=len(rs), complexity=complexity_of_spec(spec).score)
    if not rs:
        out.notes.append("no measurable trades")
        return out
    out.mean_r = float(sum(rs) / len(rs))
    try:
        from app.validation.pooled_edge import cluster_t_stat
        out.t_stat = cluster_t_stat(list(rs), days)
    except Exception:  # noqa: BLE001
        out.t_stat = None
    out.edge_lb_r = edge_lb_r(trades)
    out.penalised = out.edge_lb_r if not math.isfinite(out.edge_lb_r) else float(out.edge_lb_r - complexity_penalty(spec, complexity_weight))
    return out
