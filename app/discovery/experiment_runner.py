"""
EXPERIMENT RUNNER -- an idea across markets and timeframes, deflated for how
hard we looked, then broken, then recorded.

run_hypothesis(hyp, datasets, risk):
  datasets: {market: base_dataframe}. Each is resampled to every timeframe in
  hyp.timeframes ("5min", "15min", "1h", "4h", ...; the base frame is used as-is
  if none are given). One cell = (market, timeframe).

  * every cell runs the rule with the real engine (same sizing/costs);
  * every cell's per-trade Sharpe is deflated against the number of cells AND
    every variant already tried on this hypothesis (Bailey/Lopez de Prado
    deflated Sharpe, app.search.robustness) -- the best of N cells is
    expected to look good by chance;
  * the break-it battery runs on the best cell, using the other markets as the
    cross-market test;
  * the hypothesis status, an Experiment per run (win or lose) and the report
    text are persisted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.discovery.break_battery import BreakReport, run_break_battery
from app.discovery.hypothesis import Experiment, Hypothesis, HypothesisStore
from app.discovery.runner import metrics, resample_ohlc, run_spec


@dataclass
class CellResult:
    market: str
    timeframe: str
    n: int
    mean_r: float
    sharpe_per_trade: float
    net: float
    psr_deflated: float | None
    significant: bool | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class HypothesisRun:
    hypothesis: Hypothesis
    cells: list = field(default_factory=list)
    best_cell: CellResult | None = None
    battery: BreakReport | None = None
    n_trials: int = 0

    def render(self) -> str:
        lines = [f"HYPOTHESIS {self.hypothesis.id}: {self.hypothesis.idea}",
                 f"  rule: {self.hypothesis.spec}", f"  status: {self.hypothesis.status}   trials counted: {self.n_trials}"]
        for c in sorted(self.cells, key=lambda x: -x.mean_r):
            ds = "n/a" if c.psr_deflated is None else f"{c.psr_deflated:.2f}"
            lines.append(f"  {c.market:>8} {c.timeframe:>6}: {c.n:4d} trades  {c.mean_r:+.3f}R  net ${c.net:>9,.0f}  deflated-PSR {ds}")
        if self.battery:
            lines.append(self.battery.render())
        return "\n".join(lines)


def _spec_variants(spec: dict, cap: int = 9) -> list:
    """Base spec + bounded single-parameter neighbours (+/-25%, clamped
    to the schema range). Capped so a grid stays a grid, not a GA; every
    returned variant is schema-valid."""
    from app.discovery.rule_spec import SPEC_SCHEMA, SpecError, validate_spec

    out = [("base", spec)]
    seen = {str(sorted(spec.get("params", {}).items()))}
    schema = SPEC_SCHEMA.get(spec.get("kind"), {})
    for name, (_d, lo, hi, is_int) in schema.items():
        for mult in (0.75, 1.25):
            if len(out) >= cap:
                return out
            v = spec["params"][name] * mult
            v = min(max(v, lo), hi)
            v = int(round(v)) if is_int else float(v)
            if v == spec["params"][name]:
                continue
            params = {**spec["params"], name: v}
            key = str(sorted(params.items()))
            if key in seen:
                continue
            try:
                s2 = validate_spec({**spec, "params": params})
            except SpecError:
                continue
            seen.add(key)
            out.append((f"{name}x{mult:g}", s2))
    return out


def run_hypothesis(
    hyp: Hypothesis,
    datasets: dict,
    risk,
    *,
    store: HypothesisStore | None = None,
    n_null: int = 200,
    min_trades: int = 20,
    link_memory: bool = True,
) -> HypothesisRun:
    from app.search.robustness import deflated_sharpe_ratio
    try:
        from app.ai.experiment_memory import already_tested
        prior_rows = already_tested(hyp.id)
        if prior_rows:
            hyp.warnings.append(f"already tested {len(prior_rows)} time(s) before (see experiment memory); trials are deflated accordingly.")
    except Exception:  # noqa: BLE001
        pass
    hyp.status = "testing"
    tfs = list(hyp.timeframes) or [None]
    frames = {}
    for mk, base in datasets.items():
        for tf in tfs:
            frames[(mk, tf or "base")] = base if tf is None else resample_ohlc(base, tf)

    # v9.5: parameter-variant expansion. The QuantLab loop this layer is
    # modeled on does not test ONE parameter set per market/timeframe --
    # it tests variants. Each variant below (base spec + single-parameter
    # +/-25% neighbours, capped) runs on every cell, and EVERY run counts
    # in n_trials, so the deflated-Sharpe accounting charges for the
    # whole search, not just the cells. Displayed cells stay the base
    # spec (comparable across hypotheses); the battery attacks the best
    # (variant, cell) run overall, with the winning variant recorded.
    variants = _spec_variants(hyp.spec)
    raw = []  # ((mk, tf), variant_label, variant_spec, trades, metrics)
    for (mk, tf), d in frames.items():
        for v_label, v_spec in variants:
            tr = run_spec(d, v_spec, risk)
            raw.append(((mk, tf), v_label, v_spec, tr, metrics(tr)))

    prior = hyp.total_variants_tried() if hyp.experiments else 0
    n_trials = max(2, len(raw) + prior)
    sharpes = [m.sharpe_per_trade for _, _, _, _, m in raw if m.n >= 5]

    def _cell(mk, tf, m):
        if m.n >= 10:
            dsr = deflated_sharpe_ratio(m.sharpe_per_trade, sharpes, n_trials, m.n)
            return CellResult(mk, tf, m.n, m.mean_r, m.sharpe_per_trade, m.net,
                              float(dsr.deflated_sharpe), bool(dsr.is_significant))
        return CellResult(mk, tf, m.n, m.mean_r, m.sharpe_per_trade, m.net, None, None)

    cells = [_cell(mk, tf, m) for (mk, tf), lbl, _s, _t, m in raw if lbl == "base"]
    cells.sort(key=lambda c: -c.mean_r)
    eligible = [(mk, tf, lbl, s, m) for (mk, tf), lbl, s, _t, m in raw if m.n >= min_trades]
    run = HypothesisRun(hyp, cells=cells, n_trials=n_trials)
    if not eligible:
        hyp.status = "inconclusive"
        hyp.add_experiment(Experiment("grid", time.time(), {"cells": [c.to_dict() for c in cells]},
                                      n_variants=len(raw), passed=None,
                                      notes=f"no cell reached {min_trades} trades"))
    else:
        (bmk, btf, blbl, bspec, bm) = max(eligible, key=lambda e: e[4].mean_r)
        best = _cell(bmk, btf, bm)
        run.best_cell = best
        key = (bmk, btf)
        others = {f"{k[0]}/{k[1]}": v for k, v in frames.items() if k != key}
        rep = run_break_battery(frames[key], bspec, risk, other_markets=others,
                                n_null=n_null, min_trades=min_trades)
        run.battery = rep
        hyp.status = {"survived": "survived", "broken": "broken"}.get(rep.verdict, "inconclusive")
        hyp.add_experiment(Experiment("grid", time.time(), {"cells": [c.to_dict() for c in cells]},
                                      n_variants=len(raw), passed=None,
                                      notes=f"{len(variants)} variant(s) x {len(frames)} cell(s) = {len(raw)} runs"))
        hyp.add_experiment(Experiment("break_battery", time.time(), rep.to_dict(), n_variants=1,
                                      passed=(rep.verdict == "survived"),
                                      notes=f"best run {bmk}/{btf} variant '{blbl}' {bspec.get('params')}; deflated for {n_trials} trials"))
    if link_memory:
        _link_to_memory(hyp, run)
    if store is not None:
        store.save(hyp)
    return run


def _link_to_memory(hyp: Hypothesis, run: "HypothesisRun") -> None:
    """Best-effort: one experiment-memory row per cell (linked by hypothesis id) and, when the
    hypothesis is broken, a graveyard entry so the dead idea is not re-discovered. Never raises."""
    verdict = {"survived": "SURVIVED", "broken": "BROKEN"}.get(hyp.status, "INCONCLUSIVE")
    try:
        from app.ai.experiment_memory import record_experiment
        for c in run.cells:
            record_experiment(
                origin="discovery", strategy_name=f"{hyp.idea[:70]} [{c.market}/{c.timeframe}]", source_type="rule_spec",
                instrument=str(c.market), verdict=verdict, trades=int(c.n), net_profit=float(c.net),
                lesson=f"mean {c.mean_r:+.3f}R; deflated PSR {c.psr_deflated}; trials counted {run.n_trials}",
                config=hyp.spec, hypothesis_id=hyp.id, cell=f"{c.market}/{c.timeframe}",
            )
    except Exception:  # noqa: BLE001
        pass
    if hyp.status == "broken":
        try:
            from app.search.graveyard import GraveyardEntry, graveyard_path_for, param_signature, record_rejection
            reasons = [t.name + ": " + t.detail for t in (run.battery.tests if run.battery else []) if getattr(t, "status", "") == "fail"][:3]
            record_rejection(GraveyardEntry(
                candidate_id=f"hypothesis:{hyp.id}", family=str(hyp.spec.get("kind", "rule_spec")), generation=None,
                stage_died="break_battery", reason="; ".join(reasons) or "broken by the break-it battery",
                param_signature=param_signature(str(hyp.spec.get("kind", "rule_spec")), hyp.spec.get("params")),
                notes=[hyp.idea],
            ), path=(graveyard_path_for(str(run.best_cell.market), str(run.best_cell.timeframe)) if run.best_cell else None))
        except Exception:  # noqa: BLE001
            pass
