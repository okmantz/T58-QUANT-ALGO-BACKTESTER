from app.orchestration.gate_histogram import aggregate, load_runs, render


def _run(p, **status):
    return {"point_pass_pct": p, "gate_vector": {"gates": [{"id": k, "label": k, "status": v} for k, v in status.items()]}}


def test_histogram_counts_fail_rate_strong_rate_and_sole_killer():
    runs = [_run(90, pass_lb="cap", null="pass", dsr="pass"), _run(85, pass_lb="cap", dsr="cap"), _run(40, pass_lb="pass", null="fail"),
            _run(82, pass_lb="pass", null="pass", dsr="pass")]
    h = aggregate(runs, 80.0)
    assert h["n_runs"] == 4 and h["n_strong"] == 3
    assert h["gates"]["pass_lb"]["blocked"] == 2 and h["gates"]["pass_lb"]["fail_rate"] == 0.5
    assert h["gates"]["pass_lb"]["fail_rate_among_strong"] == 2 / 3
    assert h["gates"]["pass_lb"]["sole_killer"] == 1 and h["gates"]["null"]["sole_killer"] == 1 and h["gates"]["dsr"]["sole_killer"] == 0
    assert h["ranking"][0] in ("pass_lb", "null") and "pass_lb" in render(h)


def test_load_runs_reads_gate_vectors_and_skips_bad_files(tmp_path):
    import json
    (tmp_path / "a.json").write_text(json.dumps({"x": {"audit_evidence": {"gate_vector": {"gates": [{"id": "g", "status": "cap"}]}, "headline_pass_pct": 83.0}}}))
    (tmp_path / "bad.json").write_text("{nope")
    (tmp_path / "none.json").write_text("{}")
    runs, bad = load_runs(tmp_path)
    assert len(runs) == 1 and bad == 1 and runs[0]["point_pass_pct"] == 83.0
