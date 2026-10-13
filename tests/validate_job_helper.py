"""Helper: the Validate / Champion re-run pages start a BACKGROUND job and
redirect to /validate-simple/job/<id>; tests poll its status.json."""
import re
import time


def wait_for_validate_job(client, response, timeout=60.0) -> dict:
    assert response.status_code == 302, (response.status_code, response.get_data(as_text=True)[:300])
    m = re.search(r"/validate-simple/job/([0-9a-zA-Z_-]+)", response.headers["Location"])
    assert m, response.headers["Location"]
    job_id = m.group(1)
    deadline = time.time() + timeout
    data = {}
    while time.time() < deadline:
        data = client.get(f"/validate-simple/job/{job_id}/status.json").get_json()
        if data.get("done"):
            return data
        time.sleep(0.1)
    raise AssertionError(f"validate job {job_id} did not finish in {timeout}s: {data}")
