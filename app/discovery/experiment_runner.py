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


def run_hypothesis(
    hyp: Hypothesis,
    datasets: dict,
    risk,
    *,
    store: HypothesisStore | None = None,
    n_null: int = 100,
    min_trades: int = 20,
) -> HypothesisRun:
    from app.search.robustness import deflated_sharpe_ratio
    hyp.status = "testing"
    tfs = list(hyp.timeframes) or [None]
    frames = {}
    for mk, base in datasets.items():
        for tf in tfs:
            frames[(mk, tf or "base")] = base if tf is None else resample_ohlc(base, tf)

    raw = []
    for (mk, tf), d in frames.items():
        tr = run_spec(d, hyp.spec, risk)
        raw.append(((mk, tf), tr, metrics(tr)))

    prior = hyp.total_variants_tried() if hyp.experiments else 0
    n_trials = max(2, len(raw) + prior)
    sharpes = [m.sharpe_per_trade for _, _, m in raw if m.n >= 5]
    cells = []
    for (mk, tf), tr, m in raw:
        if m.n >= 10:
            dsr = deflated_sharpe_ratio(m.sharpe_per_trade, sharpes, n_trials, m.n)
            cells.append(CellResult(mk, tf, m.n, m.mean_r, m.sharpe_per_trade, m.net,
                                    float(dsr.deflated_sharpe), bool(dsr.is_significant)))
        else:
            cells.append(CellResult(mk, tf, m.n, m.mean_r, m.sharpe_per_trade, m.net, None, None))
    cells.sort(key=lambda c: -c.mean_r)
    eligible = [c for c in cells if c.n >= min_trades]
    run = HypothesisRun(hyp, cells=cells, n_trials=n_trials)
    if not eligible:
        hyp.status = "inconclusive"
        hyp.add_experiment(Experiment("grid", time.time(), {"cells": [c.to_dict() for c in cells]},
                                      n_variants=len(raw), passed=None,
                                      notes=f"no cell reached {min_trades} trades"))
    else:
        best = eligible[0]
        run.best_cell = best
        key = (best.market, best.timeframe)
        others = {f"{k[0]}/{k[1]}": v for k, v in frames.items() if k != key}
        rep = run_break_battery(frames[key], hyp.spec, risk, other_markets=others,
                                n_null=n_null, min_trades=min_trades)
        run.battery = rep
        hyp.status = {"survived": "survived", "broken": "broken"}.get(rep.verdict, "inconclusive")
        hyp.add_experiment(Experiment("grid", time.time(), {"cells": [c.to_dict() for c in cells]},
                                      n_variants=len(raw), passed=None))
        hyp.add_experiment(Experiment("break_battery", time.time(), rep.to_dict(), n_variants=1,
                                      passed=(rep.verdict == "survived"),
                                      notes=f"best cell {best.market}/{best.timeframe}; deflated for {n_trials} trials"))
    if store is not None:
        store.save(hyp)
    return run
