"""v9 — Prescriptive failure-recovery engine for the strategy funnel.

Owen's complaint that motivated this module: when a strategy fails Full
Pipeline (or any earlier stage), the app used to say "go back to the
drawing board / create a new strategy" with no indication of WHAT to
change. This module replaces that dead end with a data-driven plan:
which gate failed, by how much, which sub-metric is weakest, and the
concrete next actions IN ORDER — each carrying the exact parameters the
web UI needs to one-click into Quick Optimize / a new Search / a new
Evolution run.

Design rules (hard):
- NEVER prescribe lowering the 70% per-attempt validation gate.
- Every prescription must cite actual numbers from the run. If the data
  needed for a diagnosis is missing, the plan says so instead of
  guessing ("insufficient data to diagnose -- run X first").
- Generic "try different parameters" is a bug, not a feature: each
  action names exact settings.
- The module is pure logic (no Flask imports): it produces RecoveryPlan
  dataclasses; app/web/server.py turns actions into buttons/links.

Failure classes:
  SEARCH_NO_CANDIDATES   search space was empty (config problem)
  SEARCH_STAGE1_WIPEOUT  0 survivors at the cheap filters
  SEARCH_STAGE2_WIPEOUT  survivors at Stage 1, none refined past Stage 2
  SEARCH_STAGE3_WIPEOUT  refined candidates, none cleared the MC gate
  SEARCH_WEAK_CHAMPION   a champion exists but below the 70% gate
  EVO_EMPTY              evolution leaderboard empty
  EVO_STAGNANT           leaderboard exists but best fitness is weak/stalled
  QO_NO_IMPROVEMENT      Quick Optimize did not improve the candidate
  QO_BASELINE_BROKEN     Quick Optimize couldn't run (no trades / no tunables)
  PIPE_NOT_READY         Full Pipeline verdict NOT READY (decomposed below)
  PIPE_MARGINAL          Full Pipeline verdict MARGINAL
  PIPE_READY             Full Pipeline verdict READY (pass path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# The one gate this module will never suggest moving.
EVAL_PASS_GATE = 70.0          # per-attempt pass probability, 0-100 scale
PAYOUT_GATE = 50.0             # per-attempt first-payout probability, 0-100 scale


@dataclass
class RecoveryAction:
    """One concrete next step. `kind` selects the web handler; `params`
    carries everything that handler needs (no further user input)."""
    title: str
    rationale: str                       # cites actual numbers
    kind: str                            # see ACTION_KINDS
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"title": self.title, "rationale": self.rationale,
                "kind": self.kind, "params": self.params}


ACTION_KINDS = (
    "quick_optimize",     # one-click: start a Quick Optimize job, pre-filled
    "new_search",         # pre-filled Search Lab start (params -> form/POST)
    "new_evolution",      # pre-filled Evolution Lab start
    "rerun_pipeline",     # re-run Full Pipeline (after a settings change)
    "validate",           # go to the Validate hub with this strategy
    "promote",            # promote to champion / save to library
    "deploy_checklist",   # broker shakedown + live guardrails checklist
    "download_data",      # open the market-data downloader
    "manual_fix",         # text instructions only (e.g. fix a lookahead leak)
)


@dataclass
class RecoveryPlan:
    stage: str                 # "search" | "evolution" | "quick_optimize" | "pipeline"
    failure_class: str
    passed: bool               # True -> actions are the pass path
    headline: str              # one line: what happened, with numbers
    gate_name: Optional[str] = None
    gate_required: Optional[float] = None
    gate_actual: Optional[float] = None
    margin: Optional[float] = None      # shortfall in gate units (positive = missed by)
    weakest_metrics: list = field(default_factory=list)  # [(name, value, why)]
    actions: list = field(default_factory=list)           # [RecoveryAction], ordered

    def to_dict(self) -> dict:
        d = {
            "stage": self.stage, "failure_class": self.failure_class,
            "passed": self.passed, "headline": self.headline,
            "gate_name": self.gate_name, "gate_required": self.gate_required,
            "gate_actual": self.gate_actual, "margin": self.margin,
            "weakest_metrics": self.weakest_metrics,
            "actions": [a.to_dict() for a in self.actions],
        }
        return d


# ---------------------------------------------------------------------------
# DB-backed diagnostics (data-driven, not guessed)
# ---------------------------------------------------------------------------

def db_stage1_diagnostics(db_path: str | None, run_id: str | None) -> dict:
    """Query the search results DB for Stage-1 filter diagnostics.

    Returns median profit factor / trade count across Stage-1 candidates and
    the share failing on each filter, so the Stage-1 wipeout prescription
    can name the actual killer instead of guessing. Returns {} when the DB
    is unavailable -- callers treat missing diagnostics as "unknown", never
    as zero.
    """
    if not db_path or not run_id:
        return {}
    try:
        import sqlite3
        import json as _json
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT statistics_json FROM candidates WHERE run_id = ? AND stage = 'stage1'",
                (run_id,),
            ).fetchall()
        finally:
            con.close()
    except Exception:  # noqa: BLE001 -- diagnostics must never break the page
        return {}
    pfs, trs = [], []
    for (js,) in rows:
        try:
            st = _json.loads(js) if js else {}
        except Exception:  # noqa: BLE001
            continue
        pf = _num(st.get("profit_factor"))
        tr = _num(st.get("total_trades"))
        if pf is not None:
            pfs.append(pf)
        if tr is not None:
            trs.append(tr)
    if not pfs and not trs:
        return {}
    import statistics as _stats
    out = {}
    if pfs:
        out["stage1_median_pf"] = _stats.median(pfs)
        out["pf_fail_share"] = sum(1 for v in pfs if v < 1.05) / len(pfs)
    if trs:
        out["stage1_median_trades"] = _stats.median(trs)
        out["trades_fail_share"] = sum(1 for v in trs if v < 20) / len(trs)
    return out


def load_candidate_spec(db_path: str | None, run_id: str | None,
                        candidate_id: str | None) -> dict | None:
    """Load a runnable candidate spec dict from the search results DB.

    Returns a spec dict suitable for build_strategy_from_spec(), or None
    when the candidate can't be found. Used by the one-click
    POST /recovery/quick-optimize handler.
    """
    if not db_path or not run_id or not candidate_id:
        return None
    try:
        import sqlite3
        import json as _json
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = con.execute(
                """SELECT source_type, config_json, code_text, code_extension
                   FROM candidates WHERE run_id = ? AND candidate_id = ?
                   ORDER BY row_id DESC LIMIT 1""",
                (run_id, candidate_id),
            ).fetchone()
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return None
    if not row:
        return None
    source_type, config_json, code_text, code_extension = row
    try:
        import json as _json
        if (source_type or "manual") == "manual":
            return {"source_type": "manual",
                    "config": _json.loads(config_json) if config_json else {}}
        return {"source_type": source_type, "code_text": code_text or "",
                "code_extension": code_extension or ".py"}
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _num(x, default=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return v


def _fmt(x, digits=1):
    v = _num(x)
    return "n/a" if v is None else f"{v:.{digits}f}"


def _qopt_action(title: str, rationale: str, strategy_ref: dict,
                 dataset_label: str, qo_cfg: dict,
                 risk: dict | None = None, rules: dict | None = None) -> RecoveryAction:
    """One-click Quick Optimize: the web POST /recovery/quick-optimize
    takes {strategy_ref, dataset_label, qo_cfg, risk, rules} and starts
    the job directly."""
    return RecoveryAction(
        title=title, rationale=rationale, kind="quick_optimize",
        params={"strategy_ref": strategy_ref, "dataset_label": dataset_label,
                "qo_cfg": qo_cfg, "risk": risk or {}, "rules": rules or {}},
    )


def _search_action(title: str, rationale: str, search_cfg: dict) -> RecoveryAction:
    return RecoveryAction(title=title, rationale=rationale,
                          kind="new_search", params={"search_cfg": search_cfg})


def _search_cfg_base(dataset_label: str = "", risk: dict | None = None,
                     rules: dict | None = None, timeframe: str = "",
                     **extra) -> dict:
    """v9.16: the canonical prefill payload for a new_search action -- the
    finished run's real settings (market data, prop firm / risk, timeframe),
    so the Search Lab opens with them pre-filled instead of blank. Extra
    keys (family, max_candidates, seed, ...) ride along untouched."""
    cfg = {"dataset_label": dataset_label or "", "risk": risk or {},
           "rules": rules or {}, "timeframe": timeframe or ""}
    cfg.update(extra)
    return cfg


def _attach_context(plan: RecoveryPlan, risk: dict, rules: dict) -> RecoveryPlan:
    """Stamp the run's risk/rules onto every quick_optimize action so the
    one-click POST /recovery/quick-optimize has everything it needs."""
    for a in plan.actions:
        if a.kind == "quick_optimize":
            a.params["risk"] = risk or {}
            a.params["rules"] = rules or {}
    return plan


# ---------------------------------------------------------------------------
# SEARCH LAB diagnosis
# ---------------------------------------------------------------------------

def _diagnose_search_impl(summary, stage_cfg=None, dataset_label: str = "",
                    risk: dict | None = None, rules: dict | None = None,
                    db_diagnostics: dict | None = None) -> RecoveryPlan:
    """Diagnose a finished Search Lab run.

    summary: SearchSummary (or a dict with the same fields).
    db_diagnostics: optional precomputed stats from the results DB, e.g.
        {"stage1_median_pf": 0.91, "stage1_median_trades": 14,
         "pf_fail_share": 0.73, "trades_fail_share": 0.21,
         "best_family": "macd_trend", "n_families_tried": 94}
    """
    s = summary if isinstance(summary, dict) else summary.__dict__
    total = s.get("total_candidates", 0) or 0
    s1 = s.get("stage1_survivors", 0) or 0
    s2 = s.get("stage2_survivors", 0) or 0
    s3 = s.get("stage3_survivors", 0) or 0
    champ = s.get("champion_candidate_id")
    lb = s.get("leaderboard") or []
    diag = db_diagnostics or {}
    risk = risk or {}
    rules = rules or {}

    base_search = {
        "dataset_label": dataset_label, "risk": risk, "rules": rules,
        "seed": 1234,  # different seed from the failed run on purpose
    }

    # --- no candidates at all: configuration problem ---------------------
    if total == 0:
        return RecoveryPlan(
            stage="search", failure_class="SEARCH_NO_CANDIDATES", passed=False,
            headline="Search Lab generated 0 candidates -- the search space itself was empty.",
            actions=[
                RecoveryAction(
                    "Check the search configuration",
                    "0 candidates means every family was excluded or the grid collapsed "
                    "(e.g. family exclusions removed everything, or a single-family grid "
                    "had no valid parameter combinations). Re-run with family='all' and no exclusions.",
                    "new_search", {"search_cfg": {**base_search, "family": "all",
                                                 "max_candidates": 300,
                                                 "exclude_families": []}}),
            ],
        )

    # --- Stage 1 wipeout --------------------------------------------------
    if s1 == 0:
        min_pf = getattr(stage_cfg, "min_profit_factor", 1.05) if stage_cfg else 1.05
        min_tr = getattr(stage_cfg, "min_trades", 20) if stage_cfg else 20
        med_pf = _num(diag.get("stage1_median_pf"))
        med_tr = _num(diag.get("stage1_median_trades"))
        pf_share = _num(diag.get("pf_fail_share"))
        tr_share = _num(diag.get("trades_fail_share"))
        parts = [f"0 of {total} candidates survived Stage 1 "
                 f"(profit factor >= {min_pf}, trades >= {min_tr})"]
        if med_pf is not None:
            parts.append(f"median profit factor across all {total} candidates was {_fmt(med_pf, 2)}")
        if med_tr is not None:
            parts.append(f"median trade count was {_fmt(med_tr, 0)}")
        killer = ""
        if pf_share is not None and tr_share is not None:
            if pf_share >= tr_share:
                killer = (f"profit factor was the bigger killer "
                          f"({pf_share:.0%} failed on PF vs {tr_share:.0%} on trade count)")
            else:
                killer = (f"trade count was the bigger killer "
                          f"({tr_share:.0%} failed on trades vs {pf_share:.0%} on PF)")
        headline = ". ".join(parts) + (". " + killer if killer else "")
        actions = []
        if tr_share is not None and tr_share > 0.5:
            actions.append(_search_action(
                "Re-run on higher-frequency families",
                "Most candidates failed for too FEW trades, not bad profitability -- the "
                "families in this space don't trade often enough on this data/timeframe. "
                "Re-running with mean-reversion / breakout families (higher base trade "
                "frequency) attacks the actual failure instead of loosening filters.",
                {**base_search, "family": "all", "max_candidates": 400,
                 "notes": "prefer high-frequency families"}))
        actions.append(_search_action(
            "Re-run with 2x candidates and grammar invention ON",
            f"Median PF {_fmt(med_pf, 2)} vs {min_pf} needed is a near-miss across the space, "
            "not a broken setup -- more draws plus grammar-invented structures give the "
            "search more chances to find the right tail.",
            {**base_search, "family": "all", "max_candidates": min(total * 2, 2000),
             "candidate_source": "grammar"}))
        actions.append(RecoveryAction(
            "Check costs and data before re-running",
            "If commission/slippage are set high relative to the timeframe's typical move, "
            "Stage 1's profit-factor filter kills everything fairly. Verify the cost settings "
            "match the real broker, and that the dataset has enough bars (thin data = noisy PF).",
            "manual_fix", {}))
        return RecoveryPlan(
            stage="search", failure_class="SEARCH_STAGE1_WIPEOUT", passed=False,
            headline=headline,
            gate_name="Stage 1 filters", gate_required=min_pf, gate_actual=med_pf,
            margin=(min_pf - med_pf) if med_pf is not None else None,
            weakest_metrics=[("median profit factor", med_pf,
                               f"Stage 1 needs >= {min_pf}")],
            actions=actions)

    # --- Stage 2 wipeout --------------------------------------------------
    if s2 == 0:
        # Distinguish a genuine GA failure from the stall-watchdog killing
        # slow batches: if the summary reports stall-skipped batches, the
        # fix is SMALLER/faster batches (fewer Stage-1 survivors, leaner
        # GA), not a bigger GA budget -- doubling the budget on a slow box
        # just trips the watchdog again.
        stalled = s.get("stage2_stalled_skipped", 0) or 0
        if stalled and stalled >= s1 / 2:
            return RecoveryPlan(
                stage="search", failure_class="SEARCH_STAGE2_WIPEOUT", passed=False,
                headline=(f"{s1} candidate(s) cleared Stage 1 but {stalled} Stage-2 "
                          "refinement batches were killed by the stall watchdog (240s "
                          "with no completion) -- the machine is too slow for the "
                          "configured GA budget, so batches were skipped, not scored."),
                actions=[
                    _search_action(
                        "Re-run with a leaner Stage 2 (fewer survivors, smaller GA)",
                        f"{stalled} of {s1} batches never finished -- they were too slow, "
                        "not too weak. Cut Stage-1 survivors to 10 and the GA to 8x4 so "
                        "each batch finishes well inside the 240s stall timeout.",
                        {**base_search, "family": "all",
                         "max_candidates": total, "ga_population": 8,
                         "ga_generations": 4, "stage1_top_n": 10}),
                    _search_action(
                        "Re-run on fewer workers",
                        "Counter-intuitive but effective on small boxes: 2 workers "
                        "thrashing on huge batches complete nothing; 1 worker finishing "
                        "lean batches beats 2 workers stalling on fat ones.",
                        {**base_search, "family": "all",
                         "max_candidates": total, "stage1_top_n": 10,
                         "ga_population": 8, "ga_generations": 4}),
                ])
        return RecoveryPlan(
            stage="search", failure_class="SEARCH_STAGE2_WIPEOUT", passed=False,
            headline=(f"{s1} candidate(s) cleared Stage 1 but none survived Stage 2 "
                      "refinement -- the GA could not turn promising skeletons into "
                      "robust candidates."),
            actions=[
                _search_action(
                    "Re-run with a bigger Stage 2 GA budget",
                    f"{s1} skeletons entered refinement and all died there -- the GA budget "
                    "(population x generations) was likely too small to climb out of mediocre "
                    "basins. Double it and keep everything else identical for a clean comparison.",
                    {**base_search, "family": "all",
                     "max_candidates": total, "ga_population": 80,
                     "ga_generations": 30, "stage1_top_n": min(s1, 40)}),
                _search_action(
                    "Re-run with grammar invention ON",
                    "Template-only refinement explores near known structures. Grammar-drawn "
                    "candidates invent new structures, which helps exactly when every "
                    "template skeleton stalls in refinement.",
                    {**base_search, "family": "all", "max_candidates": total,
                     "candidate_source": "grammar",
                     "grammar_candidates_per_survivor": 4}),
            ])

    # --- Stage 3 wipeout --------------------------------------------------
    if s3 == 0:
        # Look at the best Stage-3 row's MC numbers to name the shortfall.
        best = lb[0] if lb else {}
        mc = best.get("mc_summary") or {}
        per_attempt = _num(mc.get("per_attempt_pass_probability"))
        stats = best.get("statistics") or {}
        pf = _num(stats.get("profit_factor"))
        wr = _num(stats.get("win_rate"))
        family = best.get("family", "unknown")
        margin = (EVAL_PASS_GATE - per_attempt) if per_attempt is not None else None
        actions = []
        strat_ref = {"candidate_id": best.get("candidate_id"),
                     "family": family, "db_path": s.get("db_path"),
                     "run_id": s.get("run_id")}
        if per_attempt is not None:
            actions.append(_qopt_action(
                f"Quick Optimize the best Stage-3 candidate ({family})",
                f"It reached {_fmt(per_attempt)}% per-attempt pass probability vs the "
                f"{EVAL_PASS_GATE:.0f}% gate -- short by {_fmt(margin)} points. Quick Optimize "
                "searches nearby parameter values for a stronger combination; it is the "
                "fastest targeted attempt before a full re-search.",
                strat_ref, dataset_label,
                {"fitness_metric": "eval_pass_probability",
                 "ga_population": 32, "ga_generations": 12, "n_folds": 4}))
        actions.append(_search_action(
            "Re-search with cost-stress raised",
            "Stage 3's Monte Carlo gate is where cost-fragile candidates die. Raising the "
            "cost-stress multiplier biases the whole search toward candidates that survive "
            "worse execution instead of ones that only look good under default costs.",
            {**base_search, "family": "all", "max_candidates": total,
             "cost_stress_multiplier": 3.0}))
        return RecoveryPlan(
            stage="search", failure_class="SEARCH_STAGE3_WIPEOUT", passed=False,
            headline=(f"{s2} candidate(s) reached Stage 3 but none cleared the Monte Carlo "
                      f"gate. Best reached {_fmt(per_attempt)}% per-attempt pass probability "
                      f"vs {EVAL_PASS_GATE:.0f}% needed (short by {_fmt(margin)} points)."),
            gate_name="Stage 3 MC gate (per-attempt pass probability)",
            gate_required=EVAL_PASS_GATE, gate_actual=per_attempt, margin=margin,
            weakest_metrics=[("per-attempt pass probability", per_attempt,
                               f"gate is {EVAL_PASS_GATE:.0f}%"),
                              ("profit factor", pf, "net of costs"),
                              ("win rate", wr, None)],
            actions=actions)

    # --- weak champion ----------------------------------------------------
    if champ and lb:
        best = lb[0]
        mc = best.get("mc_summary") or {}
        per_attempt = _num(mc.get("per_attempt_pass_probability"))
        if per_attempt is not None and per_attempt < EVAL_PASS_GATE:
            margin = EVAL_PASS_GATE - per_attempt
            strat_ref = {"candidate_id": best.get("candidate_id"),
                         "family": best.get("family"), "db_path": s.get("db_path"),
                         "run_id": s.get("run_id")}
            return RecoveryPlan(
                stage="search", failure_class="SEARCH_WEAK_CHAMPION", passed=False,
                headline=(f"Champion {best.get('candidate_id')} ({best.get('family')}) scores "
                          f"{_fmt(per_attempt)}% per-attempt pass probability -- "
                          f"{_fmt(margin)} points short of the {EVAL_PASS_GATE:.0f}% gate."),
                gate_name="per-attempt pass probability", gate_required=EVAL_PASS_GATE,
                gate_actual=per_attempt, margin=margin,
                actions=[
                    _qopt_action(
                        "Quick Optimize the champion",
                        f"Short by {_fmt(margin)} points -- a targeted GA pass over this exact "
                        "candidate's parameters is the highest-probability next step.",
                        strat_ref, dataset_label,
                        {"fitness_metric": "eval_pass_probability",
                         "ga_population": 32, "ga_generations": 12, "n_folds": 4}),
                    RecoveryAction(
                        "Promote anyway and run Full Pipeline for the full diagnosis",
                        "If you want the complete verdict (walk-forward, holdout, scorecard), "
                        "promote the champion and run Full Pipeline -- its recovery panel will "
                        "decompose exactly which sub-metric is dragging.",
                        "promote", {"candidate_id": best.get("candidate_id"),
                                    "db_path": s.get("db_path"),
                                    "run_id": s.get("run_id")}),
                ])
    # --- pass path --------------------------------------------------------
    if champ:
        return RecoveryPlan(
            stage="search", failure_class="", passed=True,
            headline=(f"Champion {champ} cleared the {EVAL_PASS_GATE:.0f}% gate."),
            actions=[
                RecoveryAction(
                    "Promote to champion and run Full Pipeline",
                    "The search gate is passed; Full Pipeline is the independent re-validation "
                    "(walk-forward, holdout, Monte Carlo, prop rules) before any real money.",
                    "promote", {"candidate_id": champ, "db_path": s.get("db_path"),
                                "run_id": s.get("run_id")}),
            ])
    return RecoveryPlan(
        stage="search", failure_class="SEARCH_UNKNOWN", passed=False,
        headline="Search finished in an unrecognized state -- insufficient data to diagnose.",
        actions=[RecoveryAction("Re-run the search",
                                 "No usable summary was produced; re-running is the only sane step.",
                                 "new_search", {"search_cfg": base_search})])


# ---------------------------------------------------------------------------
# FULL PIPELINE diagnosis
# ---------------------------------------------------------------------------

def _decompose_mc_shortfall(mc, stats: dict) -> list:
    """Break a per-attempt pass-probability shortfall into weakest sub-metrics."""
    out = []
    pf = _num(stats.get("profit_factor"))
    pf_gross = _num(stats.get("profit_factor_gross") or stats.get("gross_profit_factor"))
    wr = _num(stats.get("win_rate"))
    trades = _num(stats.get("total_trades"))
    exp = _num(stats.get("expectancy") or stats.get("avg_trade_net"))
    mc = mc or {}
    ruin = _num(mc.get("risk_of_ruin_pct"))
    p95dd = _num(mc.get("p95_drawdown_pct"))
    meddd = _num(mc.get("median_drawdown_pct"))
    if pf is not None and pf_gross is not None and pf_gross - pf > 0.25:
        out.append(("cost drag", pf,
                    f"gross PF {_fmt(pf_gross, 2)} collapses to net PF {_fmt(pf, 2)} -- "
                    "costs are eating the edge"))
    if pf is not None and pf < 1.2:
        out.append(("profit factor (net)", pf, "below 1.2 -- the edge itself is thin"))
    if wr is not None and wr < 0.4:
        out.append(("win rate", wr, "below 40% -- needs a much bigger payoff ratio to compensate"))
    if trades is not None and trades < 100:
        out.append(("trade count", trades, "under 100 -- statistics are noisy; results may not replicate"))
    if ruin is not None and ruin > 20:
        out.append(("risk of ruin", ruin, "above the 20% cap -- drawdowns threaten the account"))
    if p95dd is not None and p95dd > 8:
        out.append(("p95 drawdown", p95dd, "deep tail drawdowns fail prop gates"))
    if exp is not None and exp <= 0:
        out.append(("expectancy", exp, "non-positive per-trade expectancy"))
    return out


def _diagnose_pipeline_impl(result, dataset_label: str = "",
                      risk: dict | None = None, rules: dict | None = None,
                      timeframe: str = "") -> RecoveryPlan:
    """Diagnose a FullPipelineResult. Decomposes NOT READY / MARGINAL into
    the failing gate + margin + concrete next actions."""
    verdict = (getattr(result, "verdict", "") or "").upper()
    risk, rules = risk or {}, rules or {}
    timeframe = timeframe or ""
    mc = getattr(result, "final_mc", None)
    mc_d = {"risk_of_ruin_pct": getattr(mc, "risk_of_ruin_pct", None),
            "p95_drawdown_pct": getattr(mc, "p95_drawdown_pct", None),
            "median_drawdown_pct": getattr(mc, "median_drawdown_pct", None),
            "per_attempt_pass_probability": getattr(mc, "per_attempt_pass_probability", None),
            "per_attempt_payout_probability": getattr(mc, "per_attempt_payout_probability", None)}
    stats = getattr(getattr(result, "final_bt", None), "statistics", None) or {}
    if isinstance(stats, dict):
        stats_d = stats
    else:
        stats_d = getattr(stats, "__dict__", {})
    per_attempt = _num(mc_d["per_attempt_pass_probability"])
    margin = (EVAL_PASS_GATE - per_attempt) if per_attempt is not None else None
    strat_ref = {"source": "pipeline",
                 "display_name": getattr(result, "strategy_display_name", "strategy"),
                 "final_source_type": getattr(result, "final_source_type", "manual"),
                 "final_config": getattr(result, "final_config", None),
                 "final_code_text": getattr(result, "final_code_text", None),
                 "final_code_extension": getattr(result, "final_code_extension", None)}
    scorecard = getattr(result, "scorecard", None)

    # --- hard fails first -------------------------------------------------
    if getattr(result, "lookahead_hard_fail", False):
        return RecoveryPlan(
            stage="pipeline", failure_class="PIPE_LOOKAHEAD", passed=False,
            headline="NOT READY -- confirmed lookahead-bias leak. Every number in this report "
                     "is unreliable until the leak is fixed.",
            gate_name="lookahead check", gate_required=0, gate_actual=1, margin=1,
            actions=[
                RecoveryAction(
                    "Fix the leaking condition, then re-run from scratch",
                    "Open the lookahead check log in the report for the exact bar/condition. "
                    "The usual cause: referencing the still-forming bar's close/high/low "
                    "instead of a prior bar. Optimizing this configuration is pointless until "
                    "the leak is fixed -- fix first, re-run Full Pipeline second.",
                    "manual_fix",
                    {"leak_hint": "use prior-bar values in entry/exit conditions"}),
            ])

    if getattr(result, "risk_of_ruin_hard_fail", False):
        ruin = _num(mc_d["risk_of_ruin_pct"])
        cap = _num(getattr(result, "risk_of_ruin_cap", 20.0)) or 20.0
        return RecoveryPlan(
            stage="pipeline", failure_class="PIPE_RUIN", passed=False,
            headline=(f"NOT READY -- risk of ruin {_fmt(ruin)}% is above the {cap:.0f}% cap "
                      f"(over by {_fmt((ruin or 0) - cap)} points)."),
            gate_name="risk of ruin", gate_required=cap, gate_actual=ruin,
            margin=(ruin - cap) if ruin is not None else None,
            weakest_metrics=[("risk of ruin", ruin, f"cap is {cap:.0f}%")],
            actions=[
                _qopt_action(
                    "Quick Optimize for a lower-drawdown parameter set",
                    f"Ruin {_fmt(ruin)}% vs {cap:.0f}% cap: the GA should optimize directly "
                    "against drawdown-aware fitness (eval pass probability already penalizes "
                    "deep drawdowns through the prop simulator).",
                    strat_ref, dataset_label,
                    {"fitness_metric": "eval_pass_probability",
                     "ga_population": 32, "ga_generations": 12, "n_folds": 4}),
                RecoveryAction(
                    "Lower risk-per-trade and re-run Full Pipeline",
                    "Ruin scales roughly with the square of position size -- halving "
                    "risk-per-trade cuts ruin by roughly 4x. Re-run the pipeline with the "
                    "same strategy and lower risk to confirm.",
                    "rerun_pipeline",
                    {"risk_overrides": {"risk_value": (risk.get("risk_value", 1.0) / 2)},
                     "dataset_label": dataset_label}),
            ])

    # --- scorecard / verdict decomposition --------------------------------
    if verdict in ("NOT READY", "MARGINAL"):
        weakest = _decompose_mc_shortfall(mc_d, stats_d)
        weak_text = "; ".join(f"{n} ({_fmt(v)})" for n, v, _ in weakest[:3]) or "no sub-metric stood out"
        actions = []
        # Data-driven lever choice: cost drag -> widen stops/targets via new search knobs;
        # thin edge -> QO; noisy/low trades -> more data or higher-frequency families.
        names = [n for n, _, _ in weakest]
        if "cost drag" in names:
            pf_n = _fmt(_num(stats_d.get("profit_factor")), 2)
            pf_g = _fmt(_num(stats_d.get("profit_factor_gross") or stats_d.get("gross_profit_factor")), 2)
            _cfg = _search_cfg_base(
                dataset_label, risk, rules, timeframe, family="all",
                max_candidates=400, stop_mult_scale=1.5, seed=1234)
            actions.append(RecoveryAction(
                "Re-search with wider stops (stop_mult_scale 1.5)",
                f"Gross profit factor {pf_g} holds up but costs drag it to net {pf_n} -- the "
                "strategy churns. Wider stops/targets cut the trade count and the cost drag "
                "per unit of edge. This re-searches the same space with every candidate's "
                "ATR stop x1.5.",
                "new_search",
                {"search_cfg": _cfg}))
        if any(n in ("profit factor (net)", "win rate", "expectancy") for n in names):
            actions.append(_qopt_action(
                "Quick Optimize this exact candidate",
                f"Weakest: {weak_text}. The edge is thin but present -- a targeted GA pass "
                "over this candidate's own parameters is the cheapest way to find a "
                "stronger nearby combination before paying for a full re-search.",
                strat_ref, dataset_label,
                {"fitness_metric": "eval_pass_probability",
                 "ga_population": 32, "ga_generations": 12, "n_folds": 4}))
        if "trade count" in names:
            actions.append(RecoveryAction(
                "Get more data or use a lower timeframe",
                f"Only {_fmt(_num(stats_d.get('total_trades')), 0)} trades -- every metric is "
                "noisy at this sample size. Download more history with the in-app data "
                "downloader, or test a lower timeframe for more bars, then re-run.",
                "download_data", {}))
        if "risk of ruin" in names or "p95 drawdown" in names:
            _cfg2 = _search_cfg_base(
                dataset_label, risk, rules, timeframe, family="all",
                max_candidates=400, max_hold_bars=48, seed=1234)
            actions.append(RecoveryAction(
                "Cap holding time (max_hold_bars) in a re-search",
                "Tail drawdowns come from trades that overstay. Capping bars-in-trade cuts "
                "the worst paths without touching the entry logic.",
                "new_search",
                {"search_cfg": _cfg2}))
        if not actions:
            # Genuine fallback: still concrete, never "try different parameters".
            actions.append(_qopt_action(
                "Quick Optimize this exact candidate",
                f"Weakest: {weak_text}. Start with the cheapest targeted step: a GA pass "
                "over this candidate's own parameters.",
                strat_ref, dataset_label,
                {"fitness_metric": "eval_pass_probability",
                 "ga_population": 32, "ga_generations": 12, "n_folds": 4}))
        # Last resort is always explicit and always last.
        actions.append(RecoveryAction(
            "Last resort: fresh search with grammar invention",
            "Only if Quick Optimize and the targeted re-searches above still fail: a fresh "
            "search with grammar invention ON explores genuinely new structures instead of "
            "re-tuning the same families.",
            "new_search",
            {"search_cfg": _search_cfg_base(
                dataset_label, risk, rules, timeframe, family="all",
                max_candidates=600, candidate_source="grammar", seed=999)}))
        tier = "MARGINAL" if verdict == "MARGINAL" else "NOT READY"
        score_txt = ""
        if scorecard is not None and _num(getattr(scorecard, "score", None)) is not None:
            score_txt = f" T58 Score {_fmt(scorecard.score)}/100 ({getattr(scorecard, 'tier', '')})."
        return RecoveryPlan(
            stage="pipeline", failure_class="PIPE_MARGINAL" if verdict == "MARGINAL" else "PIPE_NOT_READY",
            passed=False,
            headline=(f"Verdict: {tier}.{score_txt} Per-attempt pass probability "
                      f"{_fmt(per_attempt)}% vs {EVAL_PASS_GATE:.0f}% gate "
                      f"(short by {_fmt(margin)} points). Weakest: {weak_text}."),
            gate_name="per-attempt pass probability", gate_required=EVAL_PASS_GATE,
            gate_actual=per_attempt, margin=margin,
            weakest_metrics=[(n, v, w) for n, v, w in weakest],
            actions=actions)

    if verdict == "READY":
        payout = _num(mc_d["per_attempt_payout_probability"])
        return RecoveryPlan(
            stage="pipeline", failure_class="", passed=True,
            headline=(f"Verdict: READY. Per-attempt pass probability {_fmt(per_attempt)}% "
                      f"clears the {EVAL_PASS_GATE:.0f}% gate by {_fmt(-(margin or 0))} points."
                      + (f" First-payout probability {_fmt(payout)}%." if payout is not None else "")),
            gate_name="per-attempt pass probability", gate_required=EVAL_PASS_GATE,
            gate_actual=per_attempt, margin=margin,
            actions=[
                RecoveryAction(
                    "Save to the Strategy Library as champion",
                    "The pipeline verdict is READY -- lock it in with the full report attached.",
                    "promote", {"display_name": getattr(result, "strategy_display_name", "")}),
                RecoveryAction(
                    "Run the Validate hub (CPCV + Regime Matrix) as a second opinion",
                    "READY is still a backtest verdict. An independent validation pass catches "
                    "what any single pipeline can miss.",
                    "validate", {"strategy_display_name": getattr(result, "strategy_display_name", ""),
                                 "dataset_label": dataset_label, "timeframe": timeframe,
                                 "risk": risk, "rules": rules}),
                RecoveryAction(
                    "Paper-trade / forward-test before risking an eval",
                    "No backtest, however clean, replaces live fills. Forward-test on demo "
                    "(MT5 Forward Test tab or a prop-firm demo) before paying for an evaluation.",
                    "deploy_checklist", {}),
            ])

    return RecoveryPlan(
        stage="pipeline", failure_class="PIPE_UNKNOWN", passed=False,
        headline="Pipeline finished with an unrecognized verdict -- insufficient data to diagnose.",
        actions=[RecoveryAction("Re-run Full Pipeline",
                                 "No usable verdict was produced.",
                                 "rerun_pipeline", {"dataset_label": dataset_label})])


# ---------------------------------------------------------------------------
# QUICK OPTIMIZE diagnosis
# ---------------------------------------------------------------------------

def _diagnose_quickopt_impl(result, dataset_label: str = "",
                      risk: dict | None = None, rules: dict | None = None,
                      timeframe: str = "") -> RecoveryPlan:
    """Diagnose a QuickOptimizeResult."""
    risk, rules = risk or {}, rules or {}
    timeframe = timeframe or ""
    # v9.16: judge on per-attempt eval-pass (the honest metric), falling
    # back to the chain-level fields for results saved before v9.16.
    base = _num(getattr(result, "baseline_per_attempt_pass_probability", None)
                or getattr(result, "baseline_eval_pass_probability", None))
    opt = _num(getattr(result, "optimized_per_attempt_pass_probability", None)
               or getattr(result, "optimized_eval_pass_probability", None))
    improved = bool(getattr(result, "improved", False))
    btr = _num(getattr(result, "baseline_trades", None))

    if btr is not None and btr == 0:
        return RecoveryPlan(
            stage="quick_optimize", failure_class="QO_BASELINE_BROKEN", passed=False,
            headline="Quick Optimize could not run: the baseline strategy took 0 trades on this data.",
            actions=[
                RecoveryAction(
                    "Fix the signal logic first",
                    "0 trades means the entry conditions never fire (or every signal is filtered). "
                    "Optimizing parameters of a strategy that never trades is meaningless -- "
                    "widen the entry conditions or check the data/timeframe, then re-run.",
                    "manual_fix", {"hint": "entry conditions never fire on this dataset"}),
            ])

    if improved and opt is not None and opt >= EVAL_PASS_GATE:
        return RecoveryPlan(
            stage="quick_optimize", failure_class="", passed=True,
            headline=(f"Quick Optimize improved per-attempt pass probability from {_fmt(base)}% "
                      f"to {_fmt(opt)}% -- clears the {EVAL_PASS_GATE:.0f}% gate."),
            gate_name="per-attempt pass probability", gate_required=EVAL_PASS_GATE,
            gate_actual=opt, margin=EVAL_PASS_GATE - opt,
            actions=[
                RecoveryAction(
                    "Run Full Pipeline on the optimized candidate",
                    "The optimizer's verdict is in-sample to its folds -- Full Pipeline is the "
                    "independent check (walk-forward, holdout, Monte Carlo) before champion.",
                    "rerun_pipeline", {"dataset_label": dataset_label}),
            ])

    if improved:
        margin = EVAL_PASS_GATE - (opt or 0)
        return RecoveryPlan(
            stage="quick_optimize", failure_class="QO_IMPROVED_SHORT", passed=False,
            headline=(f"Quick Optimize helped ({_fmt(base)}% -> {_fmt(opt)}%) but the candidate "
                      f"is still {_fmt(margin)} points short of the {EVAL_PASS_GATE:.0f}% gate."),
            gate_name="per-attempt pass probability", gate_required=EVAL_PASS_GATE,
            gate_actual=opt, margin=margin,
            actions=[
                RecoveryAction(
                    "Run Full Pipeline for the full decomposition",
                    f"Improved to {_fmt(opt)}% but short by {_fmt(margin)} points -- the pipeline's "
                    "recovery panel will name the exact sub-metric still dragging.",
                    "rerun_pipeline", {"dataset_label": dataset_label}),
                RecoveryAction(
                    "Widen the Quick Optimize budget and re-run",
                    "One GA pass found a better basin; a bigger budget (64x20) with a different "
                    "seed explores further before you pay for a full re-search.",
                    "quick_optimize",
                    {"strategy_ref": {"source": "quickopt"},
                     "dataset_label": dataset_label,
                     "qo_cfg": {"fitness_metric": "eval_pass_probability",
                                "ga_population": 64, "ga_generations": 20,
                                "n_folds": 4, "seed": 777}}),
            ])

    return RecoveryPlan(
        stage="quick_optimize", failure_class="QO_NO_IMPROVEMENT", passed=False,
        headline=(f"Quick Optimize did not improve the candidate ({_fmt(base)}% -> {_fmt(opt)}%). "
                  "The GA searched nearby parameter values and found nothing better -- this "
                  "basin is exhausted."),
        gate_name="per-attempt pass probability", gate_required=EVAL_PASS_GATE,
        gate_actual=opt, margin=(EVAL_PASS_GATE - (opt or 0)),
        actions=[
            RecoveryAction(
                "Change structure, not parameters: fresh grammar search",
                "When a full GA pass over a candidate's own parameters finds nothing, the "
                "structure itself is the limit. A grammar-invention search explores new "
                "structures instead of re-tuning this one.",
                "new_search",
                {"search_cfg": _search_cfg_base(
                    dataset_label, risk, rules, timeframe,
                    family="all", max_candidates=600,
                    candidate_source="grammar", seed=999)}),
            RecoveryAction(
                "Try a different strategy family entirely",
                "Pick the family with the best Stage-1 median from your last search and run a "
                "family-focused search on it -- different structure, same data.",
                "new_search",
                {"search_cfg": _search_cfg_base(
                    dataset_label, risk, rules, timeframe,
                    family="all", max_candidates=400, seed=1234)}),
        ])


# ---------------------------------------------------------------------------
# EVOLUTION diagnosis
# ---------------------------------------------------------------------------

def _diagnose_evolution_impl(leaderboard: list, total_evaluated: int = 0,
                       best_fitness: float | None = None,
                       dataset_label: str = "", risk: dict | None = None,
                       rules: dict | None = None) -> RecoveryPlan:
    """Diagnose a finished (or stopped) Evolution Lab run."""
    risk, rules = risk or {}, rules or {}
    lb = leaderboard or []
    if not lb:
        return RecoveryPlan(
            stage="evolution", failure_class="EVO_EMPTY", passed=False,
            headline=(f"Evolution Lab evaluated {total_evaluated} candidates and the "
                      "leaderboard is empty -- nothing survived evaluation."),
            actions=[
                RecoveryAction(
                    "Check the run log for the dominant error",
                    "An empty leaderboard with evaluations run usually means candidates are "
                    "crashing on evaluation (bad data, broken family code) rather than "
                    "scoring poorly. The per-instrument log names the exact error.",
                    "manual_fix", {"hint": "read the evolution run log first"}),
                RecoveryAction(
                    "Re-run with a smaller population to smoke-test",
                    "20 x 4 generations finishes fast and tells you whether evaluation itself "
                    "works before spending a real budget.",
                    "new_evolution",
                    {"evo_cfg": {"dataset_label": dataset_label, "risk": risk,
                                 "rules": rules, "population_size": 20,
                                 "max_generations": 4, "seed": 42}}),
            ])
    bf = _num(best_fitness)
    top = lb[0] if isinstance(lb[0], dict) else {}
    fam = top.get("family", "unknown")
    # The top record's spec (if present) lets the one-click endpoint rebuild
    # the strategy directly without a DB lookup.
    top_spec = top.get("spec") if isinstance(top, dict) else None
    return RecoveryPlan(
        stage="evolution", failure_class="EVO_REVIEW", passed=False,
        headline=(f"Evolution Lab finished: {len(lb)} on the leaderboard, "
                  f"best fitness {_fmt(bf, 3)} ({fam}). "
                  "Review the top candidates, then validate the best properly."),
        weakest_metrics=[("best fitness", bf, "higher is better; compare across runs")],
        actions=[
            RecoveryAction(
                "Quick Optimize the best evolution candidate",
                f"Best fitness {_fmt(bf, 3)} ({fam}) -- a targeted GA pass polishes what "
                "evolution found.",
                "quick_optimize",
                {"strategy_ref": {"source": "evolution", "family": fam,
                                 "spec": top_spec,
                                 "candidate_id": top.get("candidate_id")},
                 "dataset_label": dataset_label,
                 "qo_cfg": {"fitness_metric": "eval_pass_probability",
                            "ga_population": 32, "ga_generations": 12,
                            "n_folds": 4}}),
            RecoveryAction(
                "Run Full Pipeline on the best candidate",
                "Evolution fitness is in-sample to the run -- Full Pipeline re-validates "
                "out-of-sample before you trust it.",
                "rerun_pipeline", {"dataset_label": dataset_label}),
        ])


# ---------------------------------------------------------------------------
# Public entry points -- wrap the impls to stamp risk/rules context onto
# quick_optimize actions for the one-click POST handler.
# ---------------------------------------------------------------------------

def diagnose_search(summary, stage_cfg=None, dataset_label: str = "",
                    risk: dict | None = None, rules: dict | None = None,
                    db_diagnostics: dict | None = None) -> RecoveryPlan:
    return _attach_context(
        _diagnose_search_impl(summary, stage_cfg, dataset_label, risk, rules,
                              db_diagnostics),
        risk or {}, rules or {})


def diagnose_pipeline(result, dataset_label: str = "",
                      risk: dict | None = None, rules: dict | None = None,
                      timeframe: str = "") -> RecoveryPlan:
    return _attach_context(
        _diagnose_pipeline_impl(result, dataset_label, risk, rules, timeframe),
        risk or {}, rules or {})


def diagnose_quickopt(result, dataset_label: str = "",
                      risk: dict | None = None, rules: dict | None = None,
                      timeframe: str = "") -> RecoveryPlan:
    return _attach_context(
        _diagnose_quickopt_impl(result, dataset_label, risk, rules, timeframe),
        risk or {}, rules or {})


def diagnose_evolution(leaderboard: list, total_evaluated: int = 0,
                       best_fitness: float | None = None,
                       dataset_label: str = "", risk: dict | None = None,
                       rules: dict | None = None) -> RecoveryPlan:
    return _attach_context(
        _diagnose_evolution_impl(leaderboard, total_evaluated, best_fitness,
                                 dataset_label, risk, rules),
        risk or {}, rules or {})
