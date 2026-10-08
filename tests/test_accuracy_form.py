from flask import Flask

from app.backtest.risk import RiskConfig
from app.web.accuracy_form import accuracy_bp, check_form_mismatch, risk_config_from_request


def _app():
    a = Flask(__name__)
    a.register_blueprint(accuracy_bp)

    @a.post("/run")
    def run():
        r = risk_config_from_request(RiskConfig, initial_balance=50_000.0, pip_size=0.25)
        return {"mode": r.sizing_mode, "max": r.max_stop_dollars, "ib": r.intrabar_replay}
    return a


def test_instrument_endpoint_fills_from_spec():
    d = _app().test_client().get("/api/accuracy/instrument/ES").get_json()
    assert d["ok"] and d["pip_size"] == 1.0 and d["contract_size"] == 50
    assert 50 < d["round_trip_cost_dollars"] < 60


def test_sizing_mode_read_from_form_and_mismatch_blocked():
    c = _app().test_client()
    ok = c.post("/run", data={"sizing_mode": "fit_stop", "max_stop_dollars": "450", "intrabar_replay": "on", "acc_instrument": "ES", "pip_size": "1.0"})
    assert ok.status_code == 200 and ok.get_json() == {"mode": "fit_stop", "max": 450.0, "ib": True}
    bad = c.post("/run", data={"acc_instrument": "ES", "pip_size": "0.0001"})
    assert bad.status_code == 400 and "does not match" in bad.get_json()["error"]
    assert check_form_mismatch({"acc_instrument": "MES", "pip_size": "1.0", "contract_size": "50"}) is not None
