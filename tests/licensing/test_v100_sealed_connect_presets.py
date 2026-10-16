"""v10.0 additions: sealed final test slice, simple-report guidance, platform
normalisation, connect flow + in-app live-deploy enable, Apex/Lucid stamps."""
import pytest

from tests.evolution.test_evolution_lab import _trending_df
from app.evolution.engine import EvolutionConfig, EvolutionRunner
from app.backtest.risk import RiskConfig  # noqa: F401  (same import path as test_evolution_lab)
from app.prop.simulator import PropRules


def _cfg(tmp_path, **kw):
    return EvolutionConfig(
        population_size=10, elite_keep=2, max_generations=1,
        min_trades=3, min_profit_factor=0.0, max_drawdown_buffer_mult=20.0,
        mc_sims=50, robustness_neighbors=1, walk_forward_folds=2,
        cpcv_top_n=2, cpcv_max_paths=3, cpcv_n_groups=3,
        save_to_library=False, knowledge_graph_path=str(tmp_path / "kg.jsonl"),
        checkpoint_path=str(tmp_path / "c.json"), tested_log_path=str(tmp_path / "t.jsonl"), **kw)


def test_sealed_slice_is_disjoint_and_last(tmp_path):
    df = _trending_df()
    r = EvolutionRunner(df, RiskConfig(), PropRules(), _cfg(tmp_path, sealed_final_frac=0.15), progress_cb=None)
    assert r.sealed_final_df is not None and len(r.sealed_final_df)
    ts = lambda d: __import__("pandas").to_datetime(d["timestamp"])
    assert ts(r.locked_holdout_df).max() < ts(r.sealed_final_df).min()
    assert ts(r.df).max() < ts(r.locked_holdout_df).min()


def test_sealed_zero_disables(tmp_path):
    r = EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), _cfg(tmp_path, sealed_final_frac=0.0), progress_cb=None)
    assert r.sealed_final_df is None or not len(r.sealed_final_df)


def test_sealed_one_look_per_candidate(tmp_path):
    r = EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), _cfg(tmp_path), progress_cb=None)
    r._run_loop()
    if not r.leaderboard:
        pytest.skip("no survivors on this synthetic run")
    a = r.evaluate_champion_on_sealed_final()
    assert a["already_evaluated"] is False and a["pristine"] is True
    b = r.evaluate_champion_on_sealed_final()
    assert b["already_evaluated"] is True


def test_sealed_one_look_with_stub_candidate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    r = EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), _cfg(tmp_path), progress_cb=None)
    monkeypatch.setattr(r, "_evaluate_spec_on_slice", lambda df, rec, label, mc: {"label": label})
    c1 = SimpleNamespace(spec={}, sealed_check=None)
    c2 = SimpleNamespace(spec={}, sealed_check=None)
    a = r.evaluate_champion_on_sealed_final(c1)
    assert a["pristine"] and not a["already_evaluated"]
    assert r.evaluate_champion_on_sealed_final(c1)["already_evaluated"] is True
    b = r.evaluate_champion_on_sealed_final(c2)
    assert b["pristine"] is False and b["sealed_looks"] == 2


def test_sealed_refuses_while_running(tmp_path):
    r = EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), _cfg(tmp_path), progress_cb=None)
    r.is_running = True
    with pytest.raises(ValueError):
        r.evaluate_champion_on_sealed_final()


def test_normalize_platform():
    from app.live_deploy.live_settings import normalize_platform
    assert normalize_platform("MT5") == "MT4/MT5"
    assert normalize_platform("mt4") == "MT4/MT5"
    assert normalize_platform("ctrader") == "cTrader"
    assert normalize_platform("tradovate") == "Tradovate"


def test_connect_flow_validation_and_roundtrip():
    from app.live_deploy import connect_flow as cf
    errs = cf.validate_values("Tradovate", {})
    assert any("App ID" in e for e in errs)
    vals = {"login": "u", "password": "p", "app_id": "a", "app_secret": "s", "cid": "1", "sec": "x", "is_live": "false"}
    assert cf.validate_values("Tradovate", vals) == []
    acct = cf.account_from_values(platform="Tradovate", firm_name="Apex Trader Funding", nickname="n", values=vals)
    assert acct.extra_credentials["cid"] == "1" and acct.login == "u"
    pub = cf.public_values(acct)
    assert "password" not in pub and "sec" not in pub and pub["cid"] == "1"
    assert cf.account_is_live(acct) is False


def test_preflight_requires_ack_for_firm_banning_automation():
    from app.live_deploy import connect_flow as cf
    vals = {"login": "u", "password": "p", "app_id": "a", "app_secret": "s", "cid": "1", "sec": "x", "is_live": "false"}
    acct = cf.account_from_values(platform="Tradovate", firm_name="Apex Trader Funding", nickname="n", values=vals)
    acct.id = "x"
    res = cf.preflight_checklist(acct)
    assert res["needs_ack"] is True and res["can_deploy"] is True
    assert cf.preflight_checklist(None)["can_deploy"] is False


def test_enable_from_app_guards(tmp_path, monkeypatch):
    from app.live_deploy import web_deploy_config as w
    monkeypatch.setattr(w, "set_persisted_enabled", lambda v: None)
    ok = lambda **k: w.enable_from_app(**{"is_loopback": True, "has_password": True, "unlocked": True,
                                          "typed_phrase": w.ENABLE_PHRASE, **k})[0]
    assert ok() is True
    assert ok(is_loopback=False) is False
    assert ok(has_password=False) is False
    assert ok(unlocked=False) is False
    assert ok(typed_phrase="nope") is False


@pytest.mark.parametrize("key", ["apex_50k_eod", "apex_50k_intraday", "apex_100k_eod", "apex_100k_intraday", "lucid_100k", "lucid_50k"])
def test_apex_lucid_presets_verified(key):
    from app.prop.presets import get_preset, is_ready_eligible
    p = get_preset(key)
    assert p.verified_on and p.source_url and is_ready_eligible(p)


def test_simple_report_guidance_for_zero_trade_error():
    from app.validation.simple_report import interpret_check
    out = interpret_check({"name": "Sensitivity", "status": "error",
                           "detail": "Baseline produced 0 trades; nothing to compare."})
    assert out.get("guidance")
