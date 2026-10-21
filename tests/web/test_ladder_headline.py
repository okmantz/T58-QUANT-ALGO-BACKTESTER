from app.web.server import _ladder_class_filter, _ladder_filter


def test_ladder_headline_mapping():
    assert _ladder_filter("READY") == "FORWARD-TEST CANDIDATE"
    assert _ladder_filter("MARGINAL") == "LEAD"
    assert _ladder_filter("NOT READY") == "LEAD"
    assert _ladder_filter("NOT READY", "reject") == "REJECT"
    assert _ladder_filter("READY", "FUNDED") == "FUNDED"
    assert _ladder_filter(None) == "UNTESTED"
    assert _ladder_class_filter("REJECT") == "red"
