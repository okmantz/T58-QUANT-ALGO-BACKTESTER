"""v9.17 optimizer fixes."""
import math
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.optimize import refinement as rf
from app.optimize.refinement import FITNESS_METRICS, compute_fitness, _payout_window_fitness
from app.optimize.walkforward_ga import _top_across_batches
from app.prop.presets import get_preset


def _rules():
    p = get_preset("lucid_50k")
    return p.to_prop_rules()


def _mc(pass_p=0.5):
    return SimpleNamespace(per_attempt_pass_probability=pass_p, evaluation_pass_probability=pass_p,
                           n_source_trades=100)


def _trades(daily_pnl, days=120):
    t0 = datetime(2024, 1, 2, 10, 0)
    out = []
    for d in range(days):
        ts = t0 + timedelta(days=d)
        if ts.weekday() >= 5:
            continue
        out.append(SimpleNamespace(entry_time=ts, exit_time=ts + timedelta(hours=1), pnl=daily_pnl))
    return out


def test_metric_registered():
    assert "payout_30d" in FITNESS_METRICS


def test_payout_metric_orders_good_above_bad():
    r = _rules()
    good = _payout_window_fitness(_trades(400.0), r, _mc())
    bad = _payout_window_fitness(_trades(-300.0), r, _mc())
    flat = _payout_window_fitness(_trades(0.0), r, _mc())
    assert good > flat >= bad or good > bad
    assert good > 0.15 * 0.5


def test_payout_metric_no_trades_is_neg_inf_and_short_span_falls_back():
    r = _rules()
    assert _payout_window_fitness([], r, _mc()) == float("-inf")
    short = _trades(100.0, days=10)
    assert _payout_window_fitness(short, r, _mc(0.4)) == pytest.approx(0.06)


def test_compute_fitness_dispatches_payout_30d():
    r = _rules()
    f = compute_fitness({}, None, _mc(), "payout_30d", trades=_trades(400.0), prop_rules=r)
    assert math.isfinite(f) and f > 0


def test_top_across_batches_keeps_best_ever():
    c = lambda f: SimpleNamespace(fitness=f)
    allp = [c(0.9), c(0.1), c(0.2), c(float("-inf")), c(0.3), c(0.25)]
    top = _top_across_batches(allp, 3)
    assert [x.fitness for x in top] == [0.9, 0.3, 0.25]
    assert len(_top_across_batches(allp, 100)) == 6
