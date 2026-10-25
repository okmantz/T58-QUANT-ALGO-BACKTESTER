"""Graveyard: optional 'map of dead space' fields, Search Lab / Evolution Lab read-back, and diagnostics."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.backtest.risk import RiskConfig
from app.search import batch_runner as br
from app.search import graveyard as gy


def _trade(pnl, risk=1.0, size=2.0, day="2024-01-02", comm=1.0):
    t = pd.Timestamp(day)
    return Trade(entry_time=t, exit_time=t + pd.Timedelta(minutes=5), direction=1, entry_price=100.0, exit_price=101.0,
                 size=size, pnl=pnl, pnl_pct=0.1, exit_reason="signal", commission=comm, equity_after=10_000.0,
                 initial_risk=risk)


def test_old_rows_still_load_and_new_fields_are_optional():
    old = {"candidate_id": "a", "family": "f", "generation": 1, "stage_died": "stress", "reason": "r"}
    e = gy.GraveyardEntry(**old)
    assert e.n_trades is None and e.null_gate_p is None and e.doubled_cost_net_profit is None and e.data_start is None
    assert "n_trades" in e.to_dict()
    assert "Edge" not in e.render()


def test_edge_diagnostics_numbers():
    risk = RiskConfig(pip_size=0.01, spread_pips=1.0, slippage_pips=0.5, commission_per_trade=0.5)
    trades = [_trade(+10.0, day="2024-01-02"), _trade(-4.0, day="2024-01-03"), _trade(+6.0, day="2024-01-04"),
              _trade(-2.0, day="2024-01-05")] * 4
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=100, freq="5min"), "open": 1.0, "high": 1.0,
                       "low": 1.0, "close": 1.0, "volume": 1.0})
    d = gy.edge_diagnostics(trades, risk=risk, df=df, instrument="ES1!", timeframe="5m", null_gate_p=0.12)
    assert d["n_trades"] == 16
    assert d["net_profit"] == pytest.approx(40.0)
    assert d["profit_factor"] == pytest.approx(64.0 / 24.0)
    # avg R = pnl / (initial_risk * size) = pnl / 2
    assert d["avg_r_net"] == pytest.approx(np.mean([5, -2, 3, -1]))
    # doubled costs: extra = size * 2 * side_cost + commission = 2*2*0.015 + 1.0 per trade
    extra = 2 * 2 * 0.015 + 1.0
    assert d["doubled_cost_net_profit"] == pytest.approx(40.0 - 16 * extra)
    assert d["doubled_cost_net_profit"] < d["net_profit"]
    assert d["doubled_cost_avg_r"] < d["avg_r_net"]
    assert d["null_gate_p"] == 0.12                       # passed through only when the real gate ran
    assert d["instrument"] == "ES1!" and d["timeframe"] == "5m"
    assert d["data_start"].startswith("2024-01-02") and d["data_bars"] == 100
    assert d["bootstrap_p"] is not None and 0 < d["bootstrap_p"] <= 1


def test_edge_diagnostics_never_raises_and_never_invents_null_p():
    assert gy.edge_diagnostics(None)["n_trades"] == 0
    d = gy.edge_diagnostics([_trade(1.0)], risk=None)
    assert d["null_gate_p"] is None and d.get("bootstrap_p") is None     # < 10 trades: no fake bootstrap
    assert gy.edge_diagnostics(object())["instrument"] is None


def test_entry_roundtrips_through_jsonl_and_renders(tmp_path):
    e = gy.GraveyardEntry(candidate_id="c", family="f", generation=None, stage_died="search_lab_stage3", reason="x",
                          n_trades=120, avg_r_net=-0.05, profit_factor=0.91, null_gate_p=0.4,
                          doubled_cost_net_profit=-300.0, doubled_cost_profit_factor=0.8,
                          instrument="NQ1!", timeframe="5m", data_start="2024-01-02", data_end="2024-06-01")
    p = tmp_path / "g.jsonl"
    gy.record_rejection(e, p)
    row = gy.load_graveyard(p, _install=False)[0]
    assert row["n_trades"] == 120 and row["doubled_cost_profit_factor"] == 0.8
    text = e.render()
    assert "120 trades" in text and "p = 0.400" in text and "At doubled costs" in text and "NQ1! 5m 2024-01-02 to 2024-06-01" in text


def test_search_writer_stamps_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "graveyard_path_for", lambda i, t: tmp_path / "g.jsonl")
    diag = {"n_trades": 77, "net_profit": -5.0, "avg_r_net": -0.1, "profit_factor": 0.9, "doubled_cost_net_profit": -50.0}
    recs = [{"candidate_id": "x", "family": "fam", "passed_stage3_gate": False, "config": {"a": 1},
             "gate_notes": "nope", "graveyard_diag": diag}]
    assert br._write_search_graveyard_entries(recs, "ES1!", "5m") is not None
    row = gy.load_graveyard(tmp_path / "g.jsonl", _install=False)[0]
    assert row["n_trades"] == 77 and row["instrument"] == "ES1!" and row["timeframe"] == "5m"
    assert row["doubled_cost_net_profit"] == -50.0 and row["null_gate_p"] is None


# --------------------------------------------------------------------------------------------- read-back
def _space(n_dead=3, n_live=2):
    cands, meta = {}, {}
    for i in range(n_dead):
        cands[f"dead{i}"] = {"source_type": "manual", "config": {"period": 14, "value": 30}}
        meta[f"dead{i}"] = {"family": "fam"}
    for i in range(n_live):
        cands[f"live{i}"] = {"source_type": "manual", "config": {"period": 50 + i * 10, "value": 70}}
        meta[f"live{i}"] = {"family": "fam"}
    return SimpleNamespace(candidates=cands, meta=meta, family="fam")


def _seed_rows(path, n, cfg):
    sig = gy.param_signature("fam", cfg)
    gy.record_rejections([gy.GraveyardEntry(candidate_id=f"old{i}", family="fam", generation=None, stage_died="stress",
                                            reason="r", param_signature=sig) for i in range(n)], path)


def test_search_readback_skips_repeat_offenders_only(tmp_path, monkeypatch):
    monkeypatch.delenv("T58_DISABLE_GRAVEYARD_READBACK", raising=False)
    monkeypatch.setattr(br, "graveyard_path_for", lambda i, t: tmp_path / "g.jsonl")
    monkeypatch.setattr(gy, "ensure_seed_installed", lambda force=False: 0)
    _seed_rows(tmp_path / "g.jsonl", 8, {"period": 14, "value": 30})
    logs = []
    space = _space()
    assert br._apply_graveyard_readback(space, "ES1!", "5m", 8, logs.append) == 3
    assert sorted(space.candidates) == ["live0", "live1"] and sorted(space.meta) == ["live0", "live1"]
    assert any("skipped 3" in m for m in logs)
    # below the attempt threshold nothing is skipped
    space2 = _space()
    assert br._apply_graveyard_readback(space2, "ES1!", "5m", 9, logs.append) == 0 and len(space2.candidates) == 5


def test_search_readback_never_empties_the_space_and_respects_env(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "graveyard_path_for", lambda i, t: tmp_path / "g.jsonl")
    monkeypatch.setattr(gy, "ensure_seed_installed", lambda force=False: 0)
    _seed_rows(tmp_path / "g.jsonl", 20, {"period": 14, "value": 30})
    monkeypatch.delenv("T58_DISABLE_GRAVEYARD_READBACK", raising=False)
    only_dead = _space(n_dead=3, n_live=0)
    assert br._apply_graveyard_readback(only_dead, "ES1!", "5m", 8, lambda m: None) == 0 and len(only_dead.candidates) == 3
    monkeypatch.setenv("T58_DISABLE_GRAVEYARD_READBACK", "1")
    s = _space()
    assert br._apply_graveyard_readback(s, "ES1!", "5m", 8, lambda m: None) == 0 and len(s.candidates) == 5


def test_indexed_lookup_matches_is_known_dead_neighborhood():
    rows = [{"param_signature": gy.param_signature("f", {"p": 1.0})}] * 8 + [{"family": "g"}] * 3
    counts = gy.dead_signature_counts(rows)
    for fam, cfg, k in (("f", {"p": 1.0}, 8), ("f", {"p": 2.0}, 8), ("g", None, 3), ("g", None, 4)):
        assert gy.is_dead_indexed(fam, cfg, counts, min_attempts=k) == gy.is_known_dead_neighborhood(fam, cfg, rows=rows, min_attempts=k)


def test_evolution_filter_and_booking(tmp_path, monkeypatch):
    from app.evolution import engine as eng
    from app.validation import trial_ledger as tl
    monkeypatch.delenv("T58_DISABLE_GRAVEYARD_READBACK", raising=False)
    monkeypatch.setattr(tl, "_PATH", tmp_path / "ledger.json")
    monkeypatch.setattr(eng, "graveyard_path_for", lambda i, t: tmp_path / "g.jsonl")
    monkeypatch.setattr(gy, "ensure_seed_installed", lambda force=False: 0)
    _seed_rows(tmp_path / "g.jsonl", 8, {"period": 14, "value": 30})
    logs = []
    runner = SimpleNamespace(cfg=SimpleNamespace(graveyard_skip_known_dead=True, graveyard_min_attempts=8,
                                                 book_trial_ledger=True, instrument="ES1!", timeframe="5m"),
                             _log=logs.append)
    pop = [("d1", {"config": {"period": 14, "value": 30}}, {"family": "fam"}),
           ("l1", {"config": {"period": 60, "value": 70}}, {"family": "fam"}),
           ("ctl", {"config": {"period": 14, "value": 30}}, {"family": "random_control", "control": True})]
    kept = eng.EvolutionRunner._graveyard_filter_population(runner, pop)
    assert [c for c, _s, _m in kept] == ["l1", "ctl"]            # control is never skipped
    assert any("GRAVEYARD" in m for m in logs)
    eng.EvolutionRunner._book_generation_trials(runner, kept, 0)
    eng.EvolutionRunner._book_generation_trials(runner, kept, 0)   # same (run, generation): idempotent
    assert tl.family_prior("fam") == 1 and tl.family_prior("random_control") == 0
    eng.EvolutionRunner._book_generation_trials(runner, kept, 1)
    assert tl.family_prior("fam") == 2


def test_evolution_filter_keeps_generation_alive(tmp_path, monkeypatch):
    from app.evolution import engine as eng
    monkeypatch.delenv("T58_DISABLE_GRAVEYARD_READBACK", raising=False)
    monkeypatch.setattr(eng, "graveyard_path_for", lambda i, t: tmp_path / "g.jsonl")
    monkeypatch.setattr(gy, "ensure_seed_installed", lambda force=False: 0)
    _seed_rows(tmp_path / "g.jsonl", 8, {"period": 14, "value": 30})
    runner = SimpleNamespace(cfg=SimpleNamespace(graveyard_skip_known_dead=True, graveyard_min_attempts=8, instrument="a", timeframe="b"),
                             _log=lambda m: None)
    pop = [("d1", {"config": {"period": 14, "value": 30}}, {"family": "fam"})]
    assert eng.EvolutionRunner._graveyard_filter_population(runner, pop) == pop
