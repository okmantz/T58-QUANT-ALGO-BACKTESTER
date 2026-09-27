import time

from app.web.job_manager import JobManager


def test_create_returns_unique_ids_with_sane_defaults():
    jm = JobManager()
    id1 = jm.create()
    id2 = jm.create()
    assert id1 != id2

    job = jm.get(id1)
    assert job["done"] is False
    assert job["error"] is None
    assert job["result"] is None
    assert job["log"] == []
    assert isinstance(job["started_at"], float)


def test_create_accepts_initial_overrides():
    jm = JobManager()
    job_id = jm.create(instrument="EURUSD", log=["Loaded 100 bars."])
    job = jm.get(job_id)
    assert job["instrument"] == "EURUSD"
    assert job["log"] == ["Loaded 100 bars."]


def test_get_unknown_job_returns_none():
    jm = JobManager()
    assert jm.get("does-not-exist") is None


def test_get_returns_a_copy_not_the_live_dict():
    jm = JobManager()
    job_id = jm.create()
    snapshot = jm.get(job_id)
    snapshot["done"] = True  # mutating the caller's copy...
    assert jm.get(job_id)["done"] is False  # ...must not affect internal state


def test_log_appends_in_order():
    jm = JobManager()
    job_id = jm.create()
    jm.log(job_id, "step 1")
    jm.log(job_id, "step 2")
    assert jm.get(job_id)["log"] == ["step 1", "step 2"]


def test_log_on_unknown_job_does_not_raise():
    jm = JobManager()
    jm.log("nope", "should not raise")  # no assertion needed -- just must not throw


def test_update_merges_without_touching_other_fields():
    jm = JobManager()
    job_id = jm.create(instrument="XAUUSD")
    jm.log(job_id, "started")
    jm.update(job_id, report_html="/wfo_reports/x.html")
    job = jm.get(job_id)
    assert job["report_html"] == "/wfo_reports/x.html"
    assert job["instrument"] == "XAUUSD"
    assert job["log"] == ["started"]
    assert job["done"] is False


def test_finish_sets_done_and_merges_fields():
    jm = JobManager()
    job_id = jm.create()
    jm.finish(job_id, result={"n_paths": 30})
    job = jm.get(job_id)
    assert job["done"] is True
    assert job["error"] is None
    assert job["result"] == {"n_paths": 30}


def test_fail_sets_done_and_error():
    jm = JobManager()
    job_id = jm.create()
    jm.fail(job_id, "boom")
    job = jm.get(job_id)
    assert job["done"] is True
    assert job["error"] == "boom"


def test_prune_only_evicts_finished_jobs_past_max_age():
    jm = JobManager()
    running_job = jm.create()

    finished_old = jm.create()
    jm.finish(finished_old)
    jm._jobs[finished_old]["started_at"] = time.time() - 1000  # simulate age

    finished_recent = jm.create()
    jm.finish(finished_recent)

    evicted = jm.prune(max_age_seconds=500)

    assert evicted == 1
    assert jm.get(finished_old) is None
    assert jm.get(finished_recent) is not None
    # A still-running job is NEVER pruned, no matter how old it looks --
    # a slow overnight job must not vanish out from under its own status page.
    jm._jobs[running_job]["started_at"] = time.time() - 100000
    assert jm.prune(max_age_seconds=1) == 0
    assert jm.get(running_job) is not None


def test_count_reflects_current_job_dict_size():
    jm = JobManager()
    assert jm.count() == 0
    jm.create()
    jm.create()
    assert jm.count() == 2
