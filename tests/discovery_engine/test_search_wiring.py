"""Search Lab wired to the Strategy Discovery Engine: pooled daily P&L, sealed slice, lifetime/effective trials, mean-R folds."""
from __future__ import annotations

import pytest

from app.discovery_engine.pnl_store import PnlStore
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchStageConfig, promote_champion, run_search
from app.search.strategy_space import generate_search_space
from app.validation import trial_ledger as tl
from tests.search.test_batch_runner import _fast_stage_cfg, _frictionless_risk, _trending_df


@pytest.fixture()
def space():
    return generate_search_space(mode="family", family="trend_breakout", max_candidates=8, seed=1)


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "reports" / "trial_ledger.json")


def test_defaults_follow_the_one_objective():
    c = SearchStageConfig()
    assert c.walk_forward_metric == "mean_r" and c.fitness_metric == "edge_lb_r"
    assert (c.min_trades, c.stage1_min_t_stat, c.stage1_max_cost_in_r) == (100, 1.0, 0.15)
    assert c.strict_entry_floor and c.capture_pool_pnl and c.sealed_final_frac == 0.10 and c.lifetime_deflation


def test_run_search_persists_pool_pnl_seals_a_slice_and_reports_effective_trials(tmp_path, space):
    df = _trending_df()
    logs = []
    cfg = _fast_stage_cfg()
    summary = run_search(df, _frictionless_risk(), PropRules(), space, cfg, db_path=str(tmp_path / "s.db"),
                         instrument="TEST", timeframe="5m", progress_cb=logs.append)
    assert summary.sealed_bars == int(len(df) * 0.10) and summary.dev_bars + summary.locked_bars + summary.sealed_bars == len(df)
    assert summary.data_fp and summary.pool_pnl_path
    series, fams = PnlStore.load_series(summary.pool_pnl_path)
    assert series and all(f for f in fams.values())
    assert summary.pool and summary.pool["n_candidates"] == len(series)
    assert summary.trial_info and summary.trial_info["raw_trials"] >= summary.trial_info["effective_trials"] >= 1
    assert any("Trial accounting" in line for line in logs)
    # run is booked once, with the data fingerprint, so a re-run on the same window does not decay it
    assert tl.family_prior("trend_breakout", current_fp=summary.data_fp) > 0


def test_second_run_on_same_data_deflates_for_the_first(tmp_path, space):
    df = _trending_df()
    run = lambda n: run_search(df, _frictionless_risk(), PropRules(), space, _fast_stage_cfg(), db_path=str(tmp_path / f"s{n}.db"),  # noqa: E731
                               instrument="TEST", timeframe="5m")
    a = run(1)
    b = run(2)
    assert b.trial_info["ledger_prior"] >= a.trial_info["run_evaluations"] > 0
    assert b.trial_info["raw_trials"] > a.trial_info["raw_trials"]


def test_promote_champion_never_reads_the_sealed_tail(tmp_path, space, monkeypatch):
    df = _trending_df()
    cfg = _fast_stage_cfg()
    s = run_search(df, _frictionless_risk(), PropRules(), space, cfg, db_path=str(tmp_path / "s.db"), instrument="TEST", timeframe="5m")
    cid = s.leaderboard[0]["candidate_id"] if s.leaderboard else None
    if cid is None:
        pytest.skip("no stage-3 candidate on this tiny fixture")
    seen = {}
    import app.search.batch_runner as br
    real = br.run_backtest

    def spy(frame, *a, **k):
        seen.setdefault("last_ts", []).append(frame["timestamp"].iloc[-1])
        return real(frame, *a, **k)

    monkeypatch.setattr(br, "run_backtest", spy)
    res = promote_champion(str(tmp_path / "s.db"), s.run_id, cid, df, _frictionless_risk(), PropRules(), str(tmp_path / "out"),
                           mc_sims=50, sealed_final_frac=s.sealed_final_frac)
    sealed_start = df["timestamp"].iloc[len(df) - s.sealed_bars]
    locked_ts = seen["last_ts"][0]            # first backtest in promote_champion is the gate-holdout read
    assert locked_ts < sealed_start
    assert res["holdout_looks"] and res["holdout_looks"]["looks"] >= 1
