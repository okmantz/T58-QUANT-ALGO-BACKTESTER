"""Auto-Fix: act on the checks a Validate run flagged, and only keep a fix that provably helps.

The Validate page used to stop at "this failed, go to another page and work it out yourself". The
"Regime Matrix" link showed which market conditions lose money but did nothing about them, and the
"Walk-Forward GA" link went to a generic optimizer that does not address overfitting. This module is the
missing half: it takes the flagged checks, tries to repair the strategy, and then RE-RUNS the real checks
on the repaired copy so "fixed" means "the same test now passes", not "the app says so".

Two repairs, chosen by which checks failed (see simple_report.FIX_KIND_BY_CHECK):

  regime   (Regime Matrix)
      Finds the market regimes that lose money using ONLY the first 70% of the history, adds a filter that
      stands aside in the worst of them (one at a time, a few at most), then checks the result on the last
      30% -- bars it never looked at. A filter picked and judged on the same data proves nothing (any set
      of bad-luck cells looks "fixable" in-sample), so the fix is KEPT only if the unseen 30% is no worse
      and actually better, and enough trades remain.

  overfit  (CPCV / PBO / Walk-Forward / Walk-Forward GA / Sensitivity / Robustness)
      Runs the walk-forward-aware GA -- which scores every candidate only on held-out test folds --
      with progressively narrower search ranges (less room to memorise noise). The GA never sees the last
      20% of the data, so that tail is a genuine unseen check: a candidate is only kept if it is no worse
      than the original there, and the narrowest-gap candidate wins.

After either repair the originally-run checks are executed again on the repaired strategy (the same
dispatcher, the same settings). The outcome lists what now passes, what still fails and why, and anything
that got worse. Nothing is promoted automatically; a repaired strategy is saved as a NEW copy by the
caller, the original is never touched.

Flask-free: the web app and anything else can call run_auto_fix directly.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import build_run_context
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.refinement import RefinementConfig
from app.optimize.walkforward_ga import WalkforwardGACancelled, run_walkforward_aware_refinement
from app.strategy.manual import ManualStrategy
from app.validation import simple_report
from app.validation.regime_matrix import build_regime_matrix
from app.validation.runner import dispatch_validation_checks

EventCallback = Callable[..., None]
BuildFromCode = Callable[[str, str], Any]

_RANK = {"pass": 0, "info": 0, "caution": 1, "fail": 2, "error": 3}


class AutoFixError(Exception):
    """A repair could not even be attempted (bad input), as opposed to being attempted and rejected."""


# ---------------------------------------------------------------------------
# Strategy source helpers (shared with the web layer)
# ---------------------------------------------------------------------------

def strategy_source(strategy) -> tuple[str, str]:
    """(source_text, file_extension) for any of the four strategy types."""
    st = getattr(strategy, "source_type", None)
    if st == "manual":
        return json.dumps(strategy.config, indent=2), "json"
    if st == "python":
        return strategy.file_path.read_text(encoding="utf-8"), "py"
    if st == "pinescript":
        return strategy.code, "pine"
    if st == "mql5":
        return strategy.code, "mq5"
    raise AutoFixError(f"Unsupported strategy type '{st}'.")


def with_regime_exclusion(strategy, exclude_cells: list[dict], build_from_code: BuildFromCode):
    """A NEW strategy of the same type with `exclude_cells` merged into its regime-exclusion filter.
    Mirrors app.strategy.base.resolve_excluded_regimes in reverse: manual -> filters.regime_exclude,
    python -> EXCLUDE_REGIMES, pinescript/mql5 -> a T58_EXCLUDE_REGIMES directive. Returns
    (new_strategy, new_source_text, extension)."""
    st = getattr(strategy, "source_type", None)
    if st == "manual":
        cfg = copy.deepcopy(strategy.config)
        filters = cfg.setdefault("filters", {})
        existing = list(filters.get("regime_exclude") or [])
        for cell in exclude_cells:
            if cell not in existing:
                existing.append(cell)
        filters["regime_exclude"] = existing
        return ManualStrategy(cfg), json.dumps(cfg, indent=2), "json"
    if st == "python":
        code = strategy.file_path.read_text(encoding="utf-8")
        # a later top-level assignment overrides an earlier one, so appending is always safe
        # (merge with any exclusion the file already declares so earlier cells are kept)
        try:
            from app.strategy.base import resolve_excluded_regimes
            merged = list(resolve_excluded_regimes(strategy))
        except Exception:  # noqa: BLE001
            merged = []
        for cell in exclude_cells:
            if cell not in merged:
                merged.append(cell)
        new_code = code.rstrip("\n") + f"\n\nEXCLUDE_REGIMES = {merged!r}\n"
        return build_from_code("python", new_code), new_code, "py"
    if st in ("pinescript", "mql5"):
        code = strategy.code
        try:
            from app.strategy.base import resolve_excluded_regimes
            merged = list(resolve_excluded_regimes(strategy))
        except Exception:  # noqa: BLE001
            merged = []
        for cell in exclude_cells:
            if cell not in merged:
                merged.append(cell)
        directive = ";".join(",".join(f"{k}:{v}" for k, v in cell.items()) for cell in merged)
        new_code = code.rstrip("\n") + f"\n// T58_EXCLUDE_REGIMES={directive}\n"
        return build_from_code(st, new_code), new_code, ("pine" if st == "pinescript" else "mq5")
    raise AutoFixError(f"Cannot apply a regime exclusion to strategy type '{st}'.")


def strategy_from_wfga_candidate(strategy, candidate, build_from_code: BuildFromCode):
    """(strategy, source_text, extension) for a WalkforwardGACandidate's genome applied to `strategy`."""
    if getattr(strategy, "source_type", None) == "manual":
        cfg = candidate.config
        return ManualStrategy(cfg), json.dumps(cfg, indent=2), "json"
    text = candidate.code_text
    return build_from_code(strategy.source_type, text), text, (candidate.code_extension or "txt").lstrip(".")


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def window_metrics(trades: list, initial_balance: float = 0.0) -> dict:
    """Plain trade statistics for a list of trades (same PF convention as the regime matrix: no
    losers and some profit -> 10.0)."""
    pnls = [float(t.pnl) for t in trades]
    n = len(pnls)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    net = gross_profit - gross_loss
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)
    pf = min(pf, 10.0)  # above ~10 a profit factor only reflects how few losers there were; cap so it compares sensibly
    return {
        "trades": n,
        "net_profit": net,
        "profit_factor": pf,
        "win_rate": (sum(1 for p in pnls if p > 0) / n * 100.0) if n else 0.0,
        "expectancy": (net / n) if n else 0.0,
    }


def _entry_before(trade, split_ts: pd.Timestamp) -> bool:
    t = pd.Timestamp(trade.entry_time)
    if split_ts.tzinfo is not None and t.tzinfo is None:
        t = t.tz_localize(split_ts.tzinfo)
    elif split_ts.tzinfo is None and t.tzinfo is not None:
        t = t.tz_localize(None)
    return t < split_ts


def _split_trades(trades: list, split_ts: pd.Timestamp) -> tuple[list, list]:
    train = [t for t in trades if _entry_before(t, split_ts)]
    hold = [t for t in trades if not _entry_before(t, split_ts)]
    return train, hold


def _gi(form, key: str, default: int) -> int:
    try:
        return int(form.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _gs(form, key: str, default: str) -> str:
    return str(form.get(key, default) or default)


def check_score(check: str, summary: Optional[dict]) -> Optional[float]:
    """One comparable number per check, HIGHER = BETTER, so a repair that moves a result inside its
    pass/fail band (e.g. 4 losing regimes -> 3, 53% -> 57% eval-pass) still counts as real progress.
    None when the summary has nothing to compare."""
    s = summary or {}

    def f(k):
        try:
            v = float(s.get(k))
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None
    if check == "monte_carlo":
        return f("evaluation_pass_probability")
    if check == "walk_forward":
        return f("out_of_sample_efficiency")
    if check == "robustness":
        return f("parameter_robustness_score")
    if check == "regime":
        v = f("n_disable_recommended")
        return None if v is None else -v
    if check == "pbo":
        v = f("pbo")
        return None if v is None else -(v if v <= 1 else v / 100.0)
    if check == "wfga":
        gap, ins = f("overfitting_gap"), f("in_sample_fitness")
        return None if gap is None or ins is None else -(gap / (abs(ins) or 1.0))
    if check == "sensitivity":
        n, c = f("n_parameters"), f("n_cliffs")
        return None if not n or c is None else -(c / n)
    if check == "cpcv":
        oos, ins = f("mean_oos_metric"), f("mean_is_metric")
        return None if oos is None or not ins else oos / ins
    return None


def _score_improved(before: Optional[float], after: Optional[float]) -> bool:
    if before is None or after is None:
        return False
    return (after - before) > max(0.02 * abs(before), 0.005)


def _noise_note(summary: Optional[dict]) -> str:
    """For regime cells that are still flagged: say whether their losses are distinguishable from chance."""
    cells = (summary or {}).get("disable_cells") or []
    weak = [c for c in cells if c.get("t_stat") is None or c["t_stat"] > -1.0]
    if not cells:
        return ""
    if len(weak) == len(cells):
        return ("The regimes still flagged (" + ", ".join(f"{c['label']} PF {c['profit_factor']:.2f} on {c['n_trades']} trades" for c in cells)
                + ") are not statistically distinguishable from noise (every t-stat is above -1). The Regime check flags any cell under "
                "profit factor 1.0 with 15+ trades, and with this many cells a few always look like losers by luck, so removing them would "
                "be curve-fitting rather than a fix.")
    return ("Still flagged: " + ", ".join(f"{c['label']} (PF {c['profit_factor']:.2f}, {c['n_trades']} trades"
            + ("" if c.get("t_stat") is None else f", t={c['t_stat']:.1f}") + ")" for c in cells) + ".")


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: str                     # "regime" | "overfit"
    label: str
    attempted: bool
    accepted: bool
    headline: str
    reasons: list = field(default_factory=list)
    details: dict = field(default_factory=dict)
    strategy: Any = None           # the repaired strategy object when accepted (never serialised)
    source_text: str = ""
    extension: str = ""

    def to_dict(self) -> dict:
        return {"stage": self.stage, "label": self.label, "attempted": self.attempted, "accepted": self.accepted,
                "headline": self.headline, "reasons": list(self.reasons), "details": self.details}


@dataclass
class AutoFixOutcome:
    accepted: bool
    headline: str
    stages: list = field(default_factory=list)       # list[StageResult]
    before: list = field(default_factory=list)       # [{check,label,status,headline}]
    after: list = field(default_factory=list)
    newly_passing: list = field(default_factory=list)
    improved: list = field(default_factory=list)
    still_failing: list = field(default_factory=list)
    regressions: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    strategy: Any = None
    source_text: str = ""
    extension: str = ""
    after_results: list = field(default_factory=list)  # raw dispatcher results on the repaired strategy

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted, "headline": self.headline,
            "stages": [s.to_dict() for s in self.stages],
            "before": self.before, "after": self.after,
            "newly_passing": self.newly_passing, "improved": self.improved,
            "still_failing": self.still_failing, "regressions": self.regressions, "notes": self.notes,
            "has_candidate": bool(self.source_text), "extension": self.extension,
        }


def _row(item: dict) -> dict:
    return {"check": item.get("check"), "label": item.get("label"), "status": item.get("status"),
            "headline": item.get("headline")}


def regime_holdout_verdict(base_hold: dict, after_hold: dict, *, holdout_frac: float = 0.3, min_holdout_trades: int = 10,
                           min_keep_frac: float = 0.4) -> tuple[bool, list]:
    """The rule a regime filter must pass on the UNSEEN window to be kept. Returns (verifiable, reasons);
    the filter is accepted only when it is verifiable and `reasons` is empty."""
    reasons: list[str] = []
    if base_hold["trades"] < min_holdout_trades:
        reasons.append(f"Only {base_hold['trades']} trades fall in the unseen last {round(holdout_frac * 100)}% "
                       f"(need {min_holdout_trades}) -- too few to confirm a filter. Use a longer history.")
        return False, reasons
    keep_needed = max(5, int(math.ceil(min_keep_frac * base_hold["trades"])))
    if after_hold["trades"] < keep_needed:
        reasons.append(f"The filter would leave only {after_hold['trades']} of {base_hold['trades']} trades on the unseen data "
                       f"(need at least {keep_needed}) -- it removes too much of the strategy.")
    if after_hold["net_profit"] < base_hold["net_profit"]:
        reasons.append(f"On unseen data net profit gets worse ({base_hold['net_profit']:,.2f} -> {after_hold['net_profit']:,.2f}).")
    if after_hold["profit_factor"] < base_hold["profit_factor"]:
        reasons.append(f"On unseen data profit factor gets worse ({base_hold['profit_factor']:.2f} -> {after_hold['profit_factor']:.2f}).")
    if not reasons and after_hold["net_profit"] == base_hold["net_profit"] and after_hold["profit_factor"] == base_hold["profit_factor"]:
        reasons.append("The filter made no difference on the unseen data, so there is no evidence it helps.")
    return True, reasons


def tail_verdict(base_tail: dict, tail: dict, *, min_tail_trades: int = 5, tolerance: float = 0.95) -> list:
    """Why a re-tuned candidate is NOT acceptable on the unseen tail (empty list = acceptable). Quality is
    judged per trade, with a small tolerance for noise, because a re-tune that simply trades less is not worse."""
    reasons: list[str] = []
    if tail["trades"] < min_tail_trades:
        return [f"only {tail['trades']} trades on the unseen tail"]
    if tail["net_profit"] <= 0:
        reasons.append("it loses money on the unseen tail")
    if base_tail["profit_factor"] > 0 and tail["profit_factor"] < tolerance * base_tail["profit_factor"]:
        reasons.append(f"unseen-tail profit factor {tail['profit_factor']:.2f} is clearly below the original's {base_tail['profit_factor']:.2f}")
    if base_tail["expectancy"] > 0 and tail["expectancy"] < tolerance * base_tail["expectancy"]:
        reasons.append(f"unseen-tail profit per trade {tail['expectancy']:,.2f} is clearly below the original's {base_tail['expectancy']:,.2f}")
    if base_tail["trades"] >= 10 and tail["trades"] < 0.5 * base_tail["trades"]:
        reasons.append(f"it takes under half as many trades on the unseen tail ({tail['trades']} vs {base_tail['trades']})")
    return reasons


# ---------------------------------------------------------------------------
# Repair 1: regime filter, discovered in-sample, verified on unseen bars
# ---------------------------------------------------------------------------

def fix_regime(
    df: pd.DataFrame, strategy, risk, *, build_from_code: BuildFromCode,
    dimensions: tuple = ("volatility", "environment"), holdout_frac: float = 0.3, max_cells: int = 4,
    min_trades: int = 15, pf_threshold: float = 1.0, min_holdout_trades: int = 10,
    min_holdout_keep_frac: float = 0.4, event_cb: Optional[EventCallback] = None,
    cancel_event=None,
) -> StageResult:
    label = "Regime filter"
    rc = build_run_context(risk)

    def _stopped() -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    base_bt = run_backtest(df, strategy, rc)
    if not base_bt.trades:
        return StageResult("regime", label, False, False, "The strategy opened no trades, so there is nothing to filter.",
                           ["Run a plain test of this strategy on this data first."])
    if len(df) < 200:
        return StageResult("regime", label, False, False, "Not enough bars to classify regimes reliably.", [])

    split_idx = max(1, min(len(df) - 1, int(len(df) * (1.0 - holdout_frac))))
    split_ts = pd.Timestamp(df["timestamp"].iloc[split_idx])
    train0, hold0 = _split_trades(base_bt.trades, split_ts)
    base_train, base_hold = window_metrics(train0), window_metrics(hold0)

    details: dict = {
        "dimensions": list(dimensions), "holdout_fraction": holdout_frac,
        "discovery_window": "first %d%% of the bars" % round((1 - holdout_frac) * 100),
        "verification_window": "last %d%% of the bars (never used to choose the filter)" % round(holdout_frac * 100),
        "before_train": base_train, "before_holdout": base_hold, "steps": [],
    }

    current, bt = strategy, base_bt
    excluded: list[dict] = []
    labels_excluded: list[str] = []
    final_text, final_ext = strategy_source(strategy)

    for i in range(max_cells):
        if _stopped():
            break
        train, _hold = _split_trades(bt.trades, split_ts)
        matrix = build_regime_matrix(df, train, rc.initial_balance, dimensions=tuple(dimensions),
                                     min_trades_for_verdict=min_trades, disable_pf_threshold=pf_threshold)
        if matrix is None:
            return StageResult("regime", label, False, False,
                               "Not enough valid bars to classify regimes reliably.", [], details)
        flagged = [c for c in matrix.disable_regimes() if c.dims not in excluded]
        if not flagged:
            break
        worst = min(flagged, key=lambda c: c.net_profit)
        trial, text, ext = with_regime_exclusion(current, [dict(worst.dims)], build_from_code)
        trial_bt = run_backtest(df, trial, rc)
        t_train, t_hold = _split_trades(trial_bt.trades, split_ts)
        before_tr, after_tr = window_metrics(train), window_metrics(t_train)
        step = {
            "round": i + 1, "excluded_cell": worst.label, "dims": dict(worst.dims),
            "cell_trades": worst.n_trades, "cell_profit_factor": worst.profit_factor, "cell_net_profit": worst.net_profit,
            "train_net_before": before_tr["net_profit"], "train_net_after": after_tr["net_profit"],
            "train_pf_before": before_tr["profit_factor"], "train_pf_after": after_tr["profit_factor"],
            "holdout_net_after": window_metrics(t_hold)["net_profit"],
            "holdout_pf_after": window_metrics(t_hold)["profit_factor"],
            "holdout_trades_after": len(t_hold),
        }
        if after_tr["net_profit"] <= before_tr["net_profit"]:
            step["adopted"] = False
            step["note"] = "Skipping this regime: removing it did not improve the discovery window."
            details["steps"].append(step)
            break
        step["adopted"] = True
        details["steps"].append(step)
        current, bt = trial, trial_bt
        excluded.append(dict(worst.dims))
        labels_excluded.append(worst.label)
        final_text, final_ext = text, ext
        if event_cb:
            event_cb("regime_round", round=i + 1, cell=worst.label)

    if not excluded:
        why = ["No market regime had enough trades AND a losing profit factor in the first "
               f"{round((1 - holdout_frac) * 100)}% of the data, so there is nothing reliable to filter."]
        if base_hold["trades"] and base_hold["net_profit"] < 0:
            why.append("The weakness shows up in the most recent part of the data, which is exactly the part held back "
                       "for verification -- a regime filter chosen earlier cannot be justified from it.")
        return StageResult("regime", label, True, False, "No regime filter that can be justified from the data.", why, details)

    train_f, hold_f = _split_trades(bt.trades, split_ts)
    after_train, after_hold = window_metrics(train_f), window_metrics(hold_f)
    details.update({"excluded_cells": excluded, "excluded_labels": labels_excluded,
                    "after_train": after_train, "after_holdout": after_hold})

    verifiable, reasons = regime_holdout_verdict(
        base_hold, after_hold, holdout_frac=holdout_frac, min_holdout_trades=min_holdout_trades,
        min_keep_frac=min_holdout_keep_frac)
    accepted = verifiable and not reasons
    if accepted:
        headline = (f"Filter keeps the strategy out of {len(excluded)} regime(s): {', '.join(labels_excluded)}. "
                    f"On unseen data: profit factor {base_hold['profit_factor']:.2f} -> {after_hold['profit_factor']:.2f}, "
                    f"net {base_hold['net_profit']:,.2f} -> {after_hold['net_profit']:,.2f}.")
        return StageResult("regime", label, True, True, headline, [], details, current, final_text, final_ext)
    return StageResult("regime", label, True, False,
                       "The regime filter found on the first part of the data did not hold up on the unseen part, so it was not kept.",
                       reasons, details)


# ---------------------------------------------------------------------------
# Repair 2: overfitting -- walk-forward-aware GA with narrowing search ranges
# ---------------------------------------------------------------------------

def fix_overfit(
    df: pd.DataFrame, strategy, risk, rules, *, build_from_code: BuildFromCode, form,
    ladder: tuple = (1.0, 0.5, 0.25), tail_frac: float = 0.2, min_tail_trades: int = 5, tail_tolerance: float = 0.95,
    event_cb: Optional[EventCallback] = None, progress_cb: Optional[Callable[[str], None]] = None,
    cancel_event=None,
) -> StageResult:
    label = "Overfitting repair"
    rc = build_run_context(risk)
    cut = max(1, int(len(df) * (1.0 - tail_frac)))
    tail_df = df.iloc[cut:]
    if len(tail_df) < 100:
        return StageResult("overfit", label, False, False, "Not enough bars to keep a genuine unseen tail.",
                           ["Use a longer history so part of it can be held back for verification."])

    base_tail = window_metrics(run_backtest(tail_df, strategy, rc).trades)
    mc_sims = min(_gi(form, "mc_sims", 500), 500)
    mc_cfg = MonteCarloConfig(n_simulations=mc_sims)
    pop, gens = _gi(form, "wfga_population", 8), _gi(form, "wfga_generations", 4)
    folds, window_mode = _gi(form, "wf_folds", 3), _gs(form, "wf_window_mode", "rolling")
    metric = _gs(form, "fitness_metric", "eval_pass_probability")

    details: dict = {"unseen_tail": "last %d%% of the bars -- the GA never selects on it" % round(tail_frac * 100),
                     "original_tail": base_tail, "attempts": []}
    best: Optional[dict] = None
    last_error = ""

    for idx, scale in enumerate(ladder):
        if cancel_event is not None and cancel_event.is_set():
            raise WalkforwardGACancelled()
        if event_cb:
            event_cb("overfit_attempt", index=idx + 1, total=len(ladder), scale=scale)
        att: dict = {"search_range": "full" if scale >= 1.0 else f"{scale:.0%} of normal"}
        try:
            cfg = RefinementConfig(population_size=pop, generations=gens, search_monte_carlo_sims=200, fitness_metric=metric,
                                   plateau_robust_selection=True, random_seed=42 + idx)
            res = run_walkforward_aware_refinement(
                df, strategy, risk, rules, mc_cfg, cfg, n_folds=folds, window_mode=window_mode,
                progress_cb=progress_cb, cancel_event=cancel_event, reserve_tail_frac=tail_frac,
                gene_range_scale=(None if scale >= 1.0 else scale),
            )
            cand_strategy, text, ext = strategy_from_wfga_candidate(strategy, res.best, build_from_code)
            ins, oos, gap = res.best.in_sample_fitness, res.best.fitness, res.overfitting_gap
            ratio = (gap / (abs(ins) or 1.0)) if (gap is not None and ins is not None and math.isfinite(gap) and math.isfinite(ins)) else None
            tail = window_metrics(run_backtest(tail_df, cand_strategy, rc).trades)
            changed = []
            for g, new in zip(res.genes, res.best.genome):
                if abs(float(new) - float(g.base_value)) > 1e-9:
                    changed.append({"parameter": g.label, "before": g.base_value, "after": new})
            att.update({"gap_ratio": ratio, "in_sample_fitness": ins, "oos_fitness": oos, "tail": tail, "changed": changed})
            reasons = tail_verdict(base_tail, tail, min_tail_trades=min_tail_trades, tolerance=tail_tolerance)
            if ratio is None:
                reasons.append("no overfitting-gap figure was produced")
            att["eligible"] = not reasons
            att["rejected_because"] = reasons
            if not reasons and (best is None or ratio < best["ratio"]):
                best = {"ratio": ratio, "strategy": cand_strategy, "text": text, "ext": ext, "att": att}
        except WalkforwardGACancelled:
            raise
        except Exception as exc:  # noqa: BLE001 -- one rung failing must not sink the ladder
            last_error = str(exc)
            att["error"] = last_error
            att["eligible"] = False
        details["attempts"].append(att)

    if best is None:
        why = ["None of the candidates was at least as good as the current settings on the unseen tail."]
        if last_error and all(a.get("error") for a in details["attempts"]):
            why = [f"The search could not run: {last_error}"]
        return StageResult("overfit", label, True, False,
                           "No re-tuned version beat the current settings on data it had never seen, so the strategy was left as it is.",
                           why, details)

    details["chosen"] = best["att"]
    headline = (f"Re-tuned with a {best['att']['search_range']} search: the walk-forward gap is {best['ratio'] * 100:.0f}% of in-sample "
                f"and the result holds up on the unseen tail (profit factor {best['att']['tail']['profit_factor']:.2f} vs "
                f"{base_tail['profit_factor']:.2f} for the original).")
    return StageResult("overfit", label, True, True, headline, [], details, best["strategy"], best["text"], best["ext"])


# ---------------------------------------------------------------------------
# Orchestrator: repair, then re-run the REAL checks to confirm
# ---------------------------------------------------------------------------

def _plan(failed_checks: list[str]) -> list[str]:
    kinds = []
    for c in failed_checks:
        k = simple_report.FIX_KIND_BY_CHECK.get(c)
        if k and k not in kinds:
            kinds.append(k)
    # parameters first, then the regime filter built on top of the settled parameters
    return [k for k in ("overfit", "regime") if k in kinds]


def run_auto_fix(
    df: pd.DataFrame, strategy, risk, rules, *, failed_checks: list[str], before_items: list[dict], form,
    build_from_code: BuildFromCode, verify_checks: Optional[list[str]] = None,
    progress_cb: Optional[Callable[[str], None]] = None, event_cb: Optional[EventCallback] = None,
    cancel_event=None, on_check_done: Optional[Callable[[str, dict], None]] = None,
    before_raw: Optional[dict] = None,
) -> AutoFixOutcome:
    """Repair `strategy` for the flagged `failed_checks`, then re-run `verify_checks` (default: every
    check that ran originally) on the repaired copy with the same settings and compare with `before_items`
    (the interpreted items of the original run); `before_raw` is that run's {check: summary} so numeric progress inside
    a pass/fail band can be measured. Raises WalkforwardGACancelled if cancelled mid-GA."""
    log = progress_cb or (lambda _m: None)
    before = [_row(i) for i in before_items]
    before_status = {r["check"]: r["status"] for r in before}
    plan = _plan(failed_checks)
    if not plan:
        return AutoFixOutcome(
            False, "Nothing here can be fixed automatically.", before=before,
            notes=["Auto-Fix handles the Regime Matrix and the overfitting checks (CPCV, PBO, Walk-Forward, Walk-Forward GA, "
                   "Sensitivity, Robustness). A weak Monte Carlo result is a performance problem -- use Optimize for that."])

    current, text, ext = strategy, "", ""
    stages: list[StageResult] = []
    for kind in plan:
        if cancel_event is not None and cancel_event.is_set():
            raise WalkforwardGACancelled()
        if event_cb:
            event_cb("stage_start", stage=kind)
        log(f"Auto-Fix: {'repairing overfitting' if kind == 'overfit' else 'looking for a regime filter'}…")
        if kind == "overfit":
            res = fix_overfit(df, current, risk, rules, build_from_code=build_from_code, form=form,
                              event_cb=event_cb, progress_cb=log, cancel_event=cancel_event)
        else:
            res = fix_regime(df, current, risk, build_from_code=build_from_code,
                             dimensions=((_gs(form, "regime_dim_a", "volatility")), (_gs(form, "regime_dim_b", "environment"))),
                             event_cb=event_cb, cancel_event=cancel_event)
        stages.append(res)
        log(f"Auto-Fix {res.label}: {'KEPT' if res.accepted else 'not kept'} — {res.headline}")
        if res.accepted and res.strategy is not None:
            current, text, ext = res.strategy, res.source_text, res.extension
        if event_cb:
            event_cb("stage_done", stage=kind, accepted=res.accepted)

    if not any(s.accepted for s in stages):
        return AutoFixOutcome(
            False, "No repair was good enough to keep -- the strategy was left unchanged.", stages=stages, before=before,
            notes=["Every attempt is checked on data it never saw before being kept, so a rejection is the honest result "
                   "when the weakness is not something a parameter tweak or a regime filter can fix. "
                   "The reasons for each rejected attempt are listed above."])

    # ---- confirm: re-run the real checks on the repaired copy -----------------------------------
    order = simple_report.CHECK_ORDER
    to_verify = list(verify_checks) if verify_checks else [r["check"] for r in before]
    for c in failed_checks:
        if c not in to_verify:
            to_verify.append(c)
    to_verify = sorted(set(to_verify), key=lambda c: order.index(c) if c in order else 99)
    if event_cb:
        event_cb("verify_start", checks=to_verify)
    log("Auto-Fix: re-running the checks on the repaired strategy to confirm…")

    rc = build_run_context(risk)
    fixed_bt = run_backtest(df, current, rc)

    def _done(check, result):
        if event_cb:
            event_cb("verify_check_done", check=check)
        if on_check_done is not None:
            try:
                on_check_done(check, result)
            except Exception:  # noqa: BLE001
                pass

    after_results = dispatch_validation_checks(
        df, current, risk, rules, to_verify, form, baseline_trades=fixed_bt.trades,
        on_done=_done, cancel_event=cancel_event, log=log,
    )
    after_items = [simple_report.interpret_check(r) for r in after_results]
    after = [_row(i) for i in after_items]
    after_status = {r["check"]: r["status"] for r in after}

    before_raw = dict(before_raw or {})
    after_raw = {r["check"]: r.get("summary") for r in after_results}
    newly_passing, improved, regressions, still_failing = [], [], [], []
    for r in after:
        c = r["check"]
        b = before_status.get(c)
        a_rank, b_rank = _RANK.get(r["status"], 0), _RANK.get(b, 0) if b else None
        if r["status"] in ("fail", "caution", "error"):
            still_failing.append(r)
        if b is None:
            continue
        if a_rank < b_rank:
            (newly_passing if r["status"] == "pass" else improved).append({**r, "was": b})
        elif a_rank > b_rank:
            regressions.append({**r, "was": b})
        elif c in failed_checks and _score_improved(check_score(c, before_raw.get(c)), check_score(c, after_raw.get(c))):
            # same pass/fail band, but the number the check is built on moved the right way
            improved.append({**r, "was": b, "progress": "moved the right way inside the same band"})
    # a check that was flagged before and produced no result now is a gap, not a pass
    for c in failed_checks:
        if c not in after_status:
            still_failing.append({"check": c, "label": simple_report.CHECK_LABELS.get(c, c), "status": "error",
                                  "headline": "Could not be re-checked on the repaired strategy."})

    hard_regress = [r for r in regressions if r["status"] in ("fail", "error")]
    accepted = bool(newly_passing or improved) and not hard_regress
    if accepted and newly_passing:
        headline = "Fixed: " + ", ".join(r["label"] for r in newly_passing) + (" now pass." if len(newly_passing) > 1 else " now passes.")
        if still_failing:
            headline += " Still flagged: " + ", ".join(r["label"] for r in still_failing) + "."
    elif accepted:
        headline = "Improved: " + ", ".join(r["label"] for r in improved) + " moved up, but none reached a full pass yet."
    elif hard_regress:
        headline = ("The repair helped some checks but made " + ", ".join(r["label"] for r in hard_regress) +
                    " worse, so it is not recommended.")
    else:
        headline = "Re-running the checks showed no improvement, so the repair is not recommended."

    notes = []
    if "regime" in {r["check"] for r in still_failing}:
        n = _noise_note(after_raw.get("regime"))
        if n:
            notes.append(n)
    if regressions and not hard_regress:
        notes.append("Some checks moved down a notch (pass to borderline): " + ", ".join(r["label"] for r in regressions) + ".")
    return AutoFixOutcome(
        accepted, headline, stages=stages, before=before, after=after, newly_passing=newly_passing, improved=improved,
        still_failing=still_failing, regressions=regressions, notes=notes,
        strategy=current, source_text=text, extension=ext, after_results=after_results,
    )
