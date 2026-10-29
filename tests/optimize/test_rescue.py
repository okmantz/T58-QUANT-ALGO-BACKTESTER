"""Rescue search: structural variants run, selection never touches the holdout, and every variant is counted as a trial."""
from __future__ import annotations

import copy

import pytest

from app.backtest.risk import RiskConfig
from app.optimize import rescue as R
from app.search.strategy_space import FAMILIES
from tests.backtest.test_indicators_v12 import _intraday


def _cfg():
    fam = FAMILIES["trend_breakout"]
    return fam.build({k: v[len(v) // 2] for k, v in fam.param_grid.items()})


def _risk():
    return RiskConfig(initial_balance=100000.0, pip_size=0.0001)


def test_every_variant_is_a_valid_distinct_config():
    cfg = _cfg()
    seen = 0
    for _stage, make in R.STAGES:
        for label, v in make(cfg):
            assert isinstance(label, str) and v != cfg, label
            assert v is not cfg and cfg == _cfg()          # the original is never mutated
            seen += 1
    assert seen >= 20


def test_direction_variants_empty_the_other_side():
    names = dict(R.direction_variants(_cfg()))
    assert names["long only"]["entry_conditions"]["short"] == []
    assert names["short only"]["entry_conditions"]["long"] == []


def test_session_gate_is_added_to_both_sides_with_matching_connectors():
    for _l, v in R.session_variants(_cfg()):
        ec = v["entry_conditions"]
        for side in ("long", "short"):
            assert ec[side][-1]["left"]["type"] == "time_of_day"
            assert len(ec[side + "_connectors"]) == len(ec[side]) - 1


def test_selection_uses_only_training_data_and_trials_are_counted(monkeypatch):
    df = _intraday(days=60)
    lengths, recorded = [], []
    real = R.score_config

    def spy(d, cfg, risk):
        lengths.append(len(d))
        return real(d, cfg, risk)

    monkeypatch.setattr(R, "score_config", spy)
    import app.validation.trial_ledger as tl
    monkeypatch.setattr(tl, "record", lambda label, n, **k: recorded.append((label, n)) or n)
    out = R.rescue(df, _cfg(), _risk(), holdout_frac=0.3)
    cut = int(len(df) * 0.7)
    n_train_calls = out.trials                       # base + every variant, all scored on the training slice
    assert all(L == cut for L in lengths[:n_train_calls]), "a variant was scored outside the training slice"
    assert set(lengths[n_train_calls:]) <= {len(df) - cut}      # the holdout is scored only afterwards
    assert recorded == [("rescue", out.trials)]
    assert out.trials > 20


def test_headline_never_claims_a_rescue_that_failed_the_holdout():
    r = R.RescueResult({}, {}, -0.1, 0.2, -0.05, -0.2, 40, [], ["long only"])
    assert not r.rescued and "coincidence" in r.headline()
    r2 = R.RescueResult({}, {}, -0.1, 0.2, -0.05, 0.1, 40, [], ["long only"])
    assert r2.rescued and "Rescued on unseen data" in r2.headline()
    r3 = R.RescueResult({}, {}, -0.1, -0.1, -0.05, -0.05, 40, [], [])
    assert not r3.rescued and "No structural change" in r3.headline()
