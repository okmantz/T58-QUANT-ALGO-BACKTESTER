import json

import numpy as np
import pandas as pd

from app.discovery_lab import LabConfig, run_lab
from app.discovery_lab.engine import _lkey
from app.search.variant_ledger import VariantLedger, stable_key


def _bars(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    c = 4000 + np.cumsum(rng.normal(0, 2, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 1.5, n)
    l = np.minimum(o, c) - rng.uniform(0, 1.5, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="60min")
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1000})


def test_stable_key_is_deterministic_and_order_free():
    assert stable_key("a", {"x": 1, "y": 2}) == stable_key("a", {"y": 2, "x": 1})
    assert stable_key("a", 1) != stable_key("a", 2)


def test_ledger_record_known_and_stage_only_moves_up(tmp_path):
    led = VariantLedger(tmp_path / "l.sqlite")
    led.begin_run(source="t", scope="ES1!")
    led.record("ES1!", "k1", source="t", family="f", stage="train_pass", fit=2.0, spec={"a": 1})
    assert not led.is_known("ES1!", "k1")            # first seen THIS run is not "known"
    led.finish_run(tested=1)
    led.begin_run(source="t", scope="ES1!")
    assert led.is_known("ES1!", "k1") and not led.is_known("NQ1!", "k1")
    led.mark("ES1!", "k1", "winner")
    led.mark("ES1!", "k1", "searched")               # never moves down
    led.record("ES1!", "k1", source="t", family="f", stage="searched")
    led.finish_run()
    s = led.summary("ES1!")
    assert s["variants"] == 1 and s["winners"] == 1 and s["runs"] == 2 and s["evaluations"] == 2


def test_family_weights_never_starve_unexplored(tmp_path):
    led = VariantLedger(tmp_path / "l.sqlite")
    led.begin_run(source="t", scope="S")
    for i in range(100):
        led.record("S", f"a{i}", source="t", family="good", stage="train_pass" if i % 2 == 0 else "searched", fit=1.0 if i % 2 == 0 else None)
        led.record("S", f"b{i}", source="t", family="bad", stage="searched")
    led.record("S", "c0", source="t", family="new", stage="searched")
    w = led.family_weights("S", ["good", "bad", "new"])
    assert w["good"] > 1 and w["bad"] == 1 and w["new"] == 1


def test_second_lab_run_skips_known_and_finds_new_variants(tmp_path):
    led = VariantLedger(tmp_path / "l.sqlite")
    df = _bars()
    cfg = LabConfig(effort="quick", budget=500, prove_in_engine=False, ledger=led, ledger_scope="TEST")
    r1 = run_lab(df, 0.5, cfg)
    n1 = led.summary("TEST")["variants"]
    assert n1 >= 300 and r1.ledger_info["new_variants"] >= 300
    keys1 = {r[0] for r in led._con.execute("SELECT key FROM variants WHERE scope='TEST'")}
    r2 = run_lab(df, 0.5, LabConfig(effort="quick", budget=500, prove_in_engine=False, ledger=led, ledger_scope="TEST"))
    keys2_new = {r[0] for r in led._con.execute("SELECT key FROM variants WHERE scope='TEST' AND first_run=?", (r2.ledger_info["run_number"],))}
    assert keys2_new and not (keys2_new & keys1)      # run 2 only added variants run 1 never saw
    assert led.summary("TEST")["variants"] > n1
    assert r2.ledger_info["runs_on_market"] == 2
    json.dumps(r2.to_dict())


def test_lab_without_ledger_is_unchanged_and_deterministic():
    df = _bars(4000)
    a = run_lab(df, 0.5, LabConfig(effort="quick", budget=300, prove_in_engine=False))
    b = run_lab(df, 0.5, LabConfig(effort="quick", budget=300, prove_in_engine=False))
    assert a.tested == b.tested and a.train_pass == b.train_pass


def test_engine_search_backend_remembers_and_does_not_repeat(tmp_path, monkeypatch):
    from app.discovery_engine.engine import DiscoveryEngine, EngineRequest
    from app.prop.simulator import PropRules
    from app.search import variant_ledger as vl
    from tests.discovery_engine.test_engine_facade import _fake_steps
    from tests.search.test_batch_runner import _fast_stage_cfg, _frictionless_risk, _trending_df

    led = VariantLedger(tmp_path / "engine.sqlite")
    monkeypatch.setattr(vl, "_LEDGER", led)
    from app.search.ledger_search import scope_for
    SC = scope_for("ES1!")
    keep = ("min_trades", "stage1_top_n", "ga_population", "ga_generations", "ga_search_sims", "stage2_top_n", "full_mc_sims",
            "walk_forward_folds", "robustness_neighbors", "workers", "stage1_min_t_stat", "stage1_max_cost_in_r")

    def go(n):
        req = EngineRequest(preset="search_lab", df=_trending_df(n=2500), risk=_frictionless_risk(), prop_rules=PropRules(),
                            families=["trend_breakout"], work_dir=tmp_path / f"w{n}", max_candidates=6, register_retests=False, promote=False,
                            instrument="ES1!", timeframe="5m", overrides={k: v for k, v in _fast_stage_cfg().__dict__.items() if k in keep})
        return DiscoveryEngine(steps_factory=_fake_steps(False)).run(req)

    r1 = go(1)
    n1 = led.summary(SC)["variants"]
    assert n1 >= 1 and led.summary(SC)["runs"] == 1
    ids1 = {r[0] for r in led._con.execute("SELECT key FROM variants WHERE scope=?", (SC,))}
    go(2)
    ids2_new = {r[0] for r in led._con.execute("SELECT key FROM variants WHERE scope=? AND first_run=2", (SC,))}
    assert ids2_new and not (ids1 & ids2_new)
    assert led.summary(SC)["runs"] == 2


def test_graveyard_page_shows_searched_variants(tmp_path, monkeypatch):
    from app.search import variant_ledger as vl
    from app.web.server import app
    led = VariantLedger(tmp_path / "g.sqlite")
    monkeypatch.setattr(vl, "_LEDGER", led)
    led.begin_run(source="lab", scope="ES", effort="deep")
    for i in range(1234):
        led.record("ES", f"k{i}", source="lab", family="zscore_rev", stage="train_pass" if i % 10 == 0 else "searched",
                   fit=1.0 + i / 1000 if i % 10 == 0 else None, spec={"family": "zscore_rev"}, label=f"v{i}")
    led.finish_run(tested=1234)
    app.config["TESTING"] = True
    body = app.test_client().get("/graveyard").get_data(as_text=True)
    assert "Variants searched" in body and "1,234" in body and "zscore_rev" in body or "Z" in body
    body2 = app.test_client().get("/graveyard?market=ES").get_data(as_text=True)
    assert "1,234" in body2
