import time
from pathlib import Path

from app.web.server import app

_SAMPLE_CSV = Path("data/examples/EURUSD_5M_sample.csv")


def test_overnight_autopilot_form_loads():
    client = app.test_client()
    r = client.get("/overnight-autopilot")
    assert r.status_code == 200
    assert b"Overnight autopilot" in r.data


def test_overnight_autopilot_job_not_found():
    client = app.test_client()
    r = client.get("/overnight-autopilot/job/does-not-exist")
    assert r.status_code == 404


def test_overnight_autopilot_start_runs_to_completion_and_writes_report():
    client = app.test_client()
    with open(_SAMPLE_CSV, "rb") as f:
        data = dict(
            csv_file=(f, "EURUSD_5M_sample.csv"),
            initial_balance="10000", account_size="10000", risk_value="1.0", pip_size="0.0001",
            profit_target="2", daily_loss="100", max_dd="100",
            max_candidates="20", stage1_top_n="4", ga_population="4", ga_generations="1",
            top_k_to_validate="1", max_concurrent_validations="1", validation_folds="2",
            validation_final_mc_sims="100", auto_forward_test="on",
        )
        r = client.post("/overnight-autopilot/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    for _ in range(60):
        status = client.get(f"/overnight-autopilot/job/{job_id}/status.json").get_json()
        assert status["found"]
        if status["done"]:
            break
        time.sleep(1)
    else:
        raise AssertionError("Overnight Autopilot job did not finish in time")

    assert status["error"] is None
    summary = status["summary"]
    assert summary is not None
    assert "forward_test_message" in summary
    assert Path(summary["report_path"]).exists()

    job_page = client.get(f"/overnight-autopilot/job/{job_id}")
    assert job_page.status_code == 200
