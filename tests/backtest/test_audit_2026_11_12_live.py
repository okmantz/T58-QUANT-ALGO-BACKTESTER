from types import SimpleNamespace

from app.backtest.risk import RiskConfig
from app.live_deploy.execution_engine import LiveExecutionSession


def _engine(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    eng = LiveExecutionSession.__new__(LiveExecutionSession)
    eng.cfg = SimpleNamespace(prop_rules=SimpleNamespace(account_size=50_000.0), risk=RiskConfig(initial_balance=50_000.0, risk_value=1.0),
                              units_per_lot=None, contract_size=50.0, symbol="ES", journal_login="L1")
    eng.broker = SimpleNamespace(platform_name="Tradovate")
    eng.logs = []
    eng._log = lambda level, msg: eng.logs.append((level, msg))
    eng._dd_peak = 51_000.0
    return eng


def test_order_max_loss_check_shrinks_or_refuses(tmp_path, monkeypatch):
    eng = _engine(tmp_path, monkeypatch)
    # ES: 50 $/pt, 10-pt stop = $500/contract; cap = $500 -> 1 contract ok, 3 contracts shrink to 1
    assert eng._enforce_order_max_loss(1, 10.0) == 1
    assert eng._enforce_order_max_loss(3, 10.0) == 1
    assert eng._enforce_order_max_loss(1, 30.0) == 0       # one contract at a 30-pt stop is $1,500 > cap -> refused


def test_peak_persists_across_restarts(tmp_path, monkeypatch):
    eng = _engine(tmp_path, monkeypatch)
    eng._persist_peak()
    eng2 = _engine(tmp_path, monkeypatch)
    assert eng2._load_persisted_peak() == 51_000.0


def test_deploy_forms_refuse_above_cap_risk():
    from app.live_deploy.guardrails import deploy_risk_violation
    assert deploy_risk_violation("percent", 1.0, 50_000) is None
    assert "Refused" in deploy_risk_violation("percent", 2.0, 50_000)
    assert "Refused" in deploy_risk_violation("fixed", 800.0, 50_000)
    assert deploy_risk_violation("fixed", 500.0, 50_000) is None
