"""Strategy Discovery Engine foundation: objective/floor, provenance, pool statistics, P&L store, holdout budget, trial accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.discovery_engine import holdout as ho
from app.discovery_engine import objective as ob
from app.discovery_engine import pool_stats as ps
from app.discovery_engine import provenance as pv
from app.discovery_engine import trial_accounting as ta
from app.discovery_engine.pnl_store import PnlStore
from app.validation import trial_ledger as tl


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "reports" / "trial_ledger.json")
    return tl


def _bars(n=500, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="h"), "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1000.0})


# ------------------------------------------------------------------ objective
def test_floor_defaults_match_search_lab():
    f = ob.DEFAULT_FLOOR
    assert (f.min_trades, f.min_t_stat, f.max_cost_in_r) == (100, 1.0, 0.15)
    assert f.as_filters()["min_trades"] == 100


def test_floor_rejects_twenty_trade_result_that_legacy_accepted():
    stats = {"total_trades": 20, "average_r": 0.4, "profit_factor": 1.3, "net_profit": 500.0}
    assert "min_trades" in ob.entry_floor_reasons(stats)
    assert ob.passes_entry_floor(stats, ob.LEGACY_FLOOR)


def test_floor_t_stat_gate():
    ok = {"total_trades": 200, "average_r": 0.2, "profit_factor": 1.2, "net_profit": 10.0}
    assert ob.passes_entry_floor(ok)
    weak = dict(ok, average_r=0.02)
    assert "t_stat" in ob.entry_floor_reasons(weak)


def test_complexity_penalty_monotone_and_free_band():
    simple = {"config": {"entry_conditions": {"long": [{"a": 1}, {"b": 2}], "short": []}, "exit_conditions": {"long": [], "short": []}}}
    complex_ = {"config": {"entry_conditions": {"long": [{"a": i} for i in range(10)], "short": [{"a": i} for i in range(10)]},
                           "exit_conditions": {"long": [], "short": []}}}
    assert ob.complexity_penalty(simple) == 0.0
    assert ob.complexity_penalty(complex_) > ob.complexity_penalty(simple)
    assert ob.complexity_penalty(complex_, weight=0) == 0.0


# ----------------------------------------------------------------- provenance
def test_fingerprint_stable_and_sensitive():
    a, b = _bars(), _bars()
    assert pv.data_fingerprint(a) == pv.data_fingerprint(b)
    assert pv.data_fingerprint(a) != pv.data_fingerprint(a.iloc[:-1])
    assert pv.data_fingerprint(None) == "empty"


def test_provenance_roundtrip_and_reproduction():
    df = _bars()
    p = pv.capture(seed=7, df=df, config={"a": 1}, edge={"edge_lb_r": 0.05, "mean_r": 0.1, "n_trades": 120})
    assert pv.Provenance.from_dict(p.to_dict()).digest() == p.digest()
    ok = pv.verify_reproduction(p, {"edge_lb_r": 0.052, "mean_r": 0.101, "n_trades": 121}, df=df)
    assert ok.reproduced
    bad = pv.verify_reproduction(p, {"edge_lb_r": -0.1, "mean_r": 0.1, "n_trades": 120}, df=df)
    assert not bad.reproduced
    assert not pv.verify_reproduction(p, {"edge_lb_r": 0.05, "mean_r": 0.1, "n_trades": 120}, df=_bars(seed=3)).reproduced


# ------------------------------------------------------------------ pool stats
def _noise_matrix(T=800, N=60, seed=1, planted=None):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 1, (T, N))
    if planted is not None:
        m[:, 0] += planted
    return m


def test_family_wise_does_not_reject_pure_noise():
    rejections = 0
    for s in range(6):
        r = ps.family_wise_tests(_noise_matrix(seed=s), n_boot=300, seed=s)
        rejections += int(r.passed)
    assert rejections <= 1


def test_family_wise_detects_planted_edge():
    r = ps.family_wise_tests(_noise_matrix(planted=0.25), list(range(60)), n_boot=400)
    assert r.passed and r.best_index == 0 and 0 in r.stepdown_rejected


def test_pool_pbo_near_half_on_noise():
    vals = [ps.pool_pbo(_noise_matrix(1000, 80, seed=s)).pbo for s in range(5)]
    assert 0.25 < float(np.mean(vals)) < 0.75


def test_stratified_sample_covers_every_stratum():
    ids = list(range(100))
    strata = ["a"] * 90 + ["b"] * 6 + ["c"] * 4
    sel = ps.stratified_sample(ids, strata, 20)
    assert len(sel) == 20 and {strata[i] for i in sel} == {"a", "b", "c"}


# ------------------------------------------------------------------ pnl store
def test_pnl_store_roundtrip_and_resume(tmp_path):
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    path = tmp_path / "run.pnl"
    st = PnlStore(path)
    assert st.add("a", "orb", pd.Series(np.arange(30.0), index=idx))
    assert not st.add("a", "orb", pd.Series(dtype=float))        # duplicate
    st.add("b", "gap", pd.Series(np.ones(10), index=idx[:10]))
    assert st.flush() == 2
    st2 = PnlStore(path)                                          # resume
    assert not st2.add("b", "gap", pd.Series(dtype=float))
    st2.add("c", "orb", pd.Series(np.ones(5), index=idx[:5]))
    st2.flush()
    mat, ids, fams, cal = PnlStore.load(path)
    assert sorted(ids) == ["a", "b", "c"] and mat.shape == (30, 3) and fams["c"] == "orb"
    assert PnlStore.delete(path) == 2


def test_pnl_store_respects_cap(tmp_path):
    st = PnlStore(tmp_path / "x", max_candidates=2)
    s = pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1))
    assert st.add("1", "f", s) and st.add("2", "f", s) and not st.add("3", "f", s)


# -------------------------------------------------------------------- holdout
def test_three_way_split_is_chronological_and_disjoint():
    df = _bars(1000)
    dev, gate, sealed = ho.split_three_way(df, 0.2, 0.1)
    assert len(dev) + len(gate) + len(sealed) == 1000 and len(sealed) == 100 and len(gate) == 200
    assert dev["timestamp"].iloc[-1] < gate["timestamp"].iloc[0] < sealed["timestamp"].iloc[0]


def test_budget_counts_looks_shrinks_and_seals():
    b = ho.HoldoutBudget(max_looks=3, decay=0.9, floor=0.5)
    gate = _bars(100)
    for _ in range(3):
        b.look()
    assert len(b.usable_slice(gate)) < 100
    with pytest.raises(ho.HoldoutExhausted):
        b.look()
    b.read_sealed(gate)
    with pytest.raises(ho.HoldoutExhausted):
        b.read_sealed(gate)
    assert ho.HoldoutBudget.from_dict(b.to_dict()).looks == 3


def test_thresholdout_returns_training_value_when_close_and_spends_when_far():
    t = ho.Thresholdout(threshold=0.05, sigma=0.001, budget=2, seed=1)
    v, revealed = t.query(0.10, 0.101)
    assert not revealed and v == 0.10
    v, revealed = t.query(0.10, 0.40)
    assert revealed and abs(v - 0.40) < 0.05 and t.remaining == 1


# ------------------------------------------------------------ trial accounting
def test_counter_books_once_with_kinds(ledger):
    c = ta.TrialCounter()
    c.add("orb", 100, "prefilter_kills")
    c.add("orb", 50, "stage1")
    c.add("gap", 7, "ga_inner_loop")
    assert c.total == 157
    assert c.book("run1", data_fp="fp1")
    assert not c.book("run1", data_fp="fp1")
    assert ledger.family_prior("orb") == 150 and ledger.run_kinds("run1")["prefilter_kills"] == 100


def test_same_window_trials_do_not_decay(ledger):
    ledger.record("family:orb", 1000, now=1.0, data_fp="fpA")      # booked a very long time ago
    assert ta.lifetime_trials("family:orb", data_fp="fpA") == 1000
    assert ta.lifetime_trials("family:orb", data_fp="fpB") < 1000   # window changed -> decays


def test_trial_inputs_add_ledger_and_never_loosen(ledger):
    ledger.record("my_strat", 500, data_fp="fp")
    ti = ta.trial_inputs(40, names=["my_strat"], data_fp="fp")
    assert ti.ledger_prior == 500 and ti.raw_trials == 540 and ti.n_tests == 540
    q = ta.quick_optimize_n_tests(40, True, ["my_strat"], data_fp="fp")
    assert q.n_tests >= 540
    assert ta.quick_optimize_n_tests(0, False, ["nobody"]).n_tests == 1


def test_effective_trials_cluster_duplicates():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=300), index=pd.date_range("2024-01-01", periods=300))
    streams = [base + rng.normal(0, 0.01, 300) for _ in range(6)] + [pd.Series(rng.normal(size=300), index=base.index) for _ in range(2)]
    out = ta.effective_trials(8000, streams)
    assert out["n_clusters"] == 3 and out["effective_trials"] < 8000
