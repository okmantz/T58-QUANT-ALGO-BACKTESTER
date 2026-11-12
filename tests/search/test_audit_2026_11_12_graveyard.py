from app.search.graveyard import param_signature


def test_param_signature_walks_nested_manual_config():
    a = {"entry": {"conditions": [{"indicator": "rsi", "period": 14, "level": 30}]}}
    b = {"entry": {"conditions": [{"indicator": "rsi", "period": 14, "level": 70}]}}
    assert param_signature("manual", a) != param_signature("manual", b)
    assert param_signature("manual", a) != "manual"
    assert param_signature("manual", a) == param_signature("manual", dict(a))
