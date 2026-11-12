from types import SimpleNamespace

from app.reports.provenance import gate_provenance, gate_provenance_line, risk_cap_audit, risk_cap_audit_line


def test_gate_provenance_reflects_env(monkeypatch):
    monkeypatch.delenv("T58_SKIP_EXTRA_GATES", raising=False)
    monkeypatch.delenv("T58_USE_ALL_FAMILIES", raising=False)
    assert gate_provenance()["certification_clean"] is True
    monkeypatch.setenv("T58_SKIP_EXTRA_GATES", "1")
    p = gate_provenance()
    assert p["certification_clean"] is False and "NOT CERTIFIABLE" in gate_provenance_line(p)


def test_risk_cap_audit_flags_over_cap_trades():
    trades = [SimpleNamespace(risk_at_stop_dollars=400.0), SimpleNamespace(risk_at_stop_dollars=900.0)]
    a = risk_cap_audit(trades, None, SimpleNamespace(account_size=50_000))
    assert a["cap_dollars"] == 500.0 and a["over_cap_trades"] == 1 and not a["within_cap"]
    assert "OVER CAP" in risk_cap_audit_line(a)
