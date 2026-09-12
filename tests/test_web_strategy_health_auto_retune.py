from app.web.server import app


def test_strategy_health_form_loads_with_retune_picker():
    client = app.test_client()
    r = client.get("/quant-lab/strategy-health")
    assert r.status_code == 200
    assert b"retune_strategy" in r.data
    assert b"(health check only)" in r.data


def test_strategy_health_missing_journal_shows_clean_error():
    client = app.test_client()
    r = client.post(
        "/quant-lab/strategy-health",
        data={"session_id": "1", "strategy_label": "x", "account_balance": "10000"},
    )
    assert r.status_code == 200
    assert b"Please choose a journal .db file." in r.data


def test_strategy_health_unknown_retune_strategy_shows_clean_error():
    client = app.test_client()
    import io
    r = client.post(
        "/quant-lab/strategy-health",
        data={
            "session_id": "1", "strategy_label": "x", "account_balance": "10000",
            "retune_strategy": "python:definitely_not_a_real_strategy.py",
            "journal_db": (io.BytesIO(b"not a real db"), "journal.db"),
            "mc_result_json": (io.BytesIO(b"{}"), "mc.json"),
        },
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    # Either the MC json parse fails first, or the strategy lookup fails --
    # either way this must be a clean rendered error, never a 500.
    assert b"result-error" in r.data
