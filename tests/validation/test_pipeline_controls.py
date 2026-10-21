from app.validation.pipeline_controls import negative_control, positive_control


def test_negative_control_false_positive_rate_is_controlled():
    r = negative_control(n_sims=80, n_boot=200)
    assert r["false_positive_rate"] <= 0.15


def test_positive_control_pooled_beats_single_market():
    r = positive_control(edges=(0.30,), n_sims=40, n_boot=200)
    row = r["rows"][0]
    assert row["pooled"] >= row["single_market"]
    assert row["pooled"] >= 0.6
