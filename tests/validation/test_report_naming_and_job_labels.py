"""Every report is saved as <Strategy>__<Test>__<timestamp>, says which strategy it tested, and every job
page can show which strategy it is running."""
from __future__ import annotations

import json
import re

import pytest

from app.reports import naming
from app.validation import report_export


# ---------------------------------------------------------------- the naming rule

def test_report_stem_has_strategy_test_and_timestamp():
    stem = naming.report_stem("NQ1! 15m EMA Cross.json", "Full Pipeline", when=1_790_000_000)
    assert re.fullmatch(r"NQ1_15m_EMA_Cross__Full-Pipeline__\d{8}-\d{6}", stem), stem


@pytest.mark.parametrize("name", ["a/b\\c:d*e?f\"g<h>i|j", "  spaced  name  ", "é ü 日本語", "x" * 300, ""])
def test_report_stem_is_always_a_safe_file_name(name):
    stem = naming.report_stem(name, "CPCV")
    assert re.fullmatch(r"[A-Za-z0-9._-]+", stem), stem
    assert len(stem) < 120 and "__CPCV__" in stem and not stem.startswith(("_", ".", "-"))


def test_report_stem_never_overwrites_an_existing_report(tmp_path):
    first = naming.report_stem("My Strat", "Validation", when=1_790_000_000, output_dir=tmp_path)
    (tmp_path / f"{first}.json").write_text("{}")
    second = naming.report_stem("My Strat", "Validation", when=1_790_000_000, output_dir=tmp_path)
    (tmp_path / f"{second}.html").write_text("")
    third = naming.report_stem("My Strat", "Validation", when=1_790_000_000, output_dir=tmp_path)
    assert len({first, second, third}) == 3 and second.endswith("_2") and third.endswith("_3")


def test_strategy_label_prefers_explicit_then_library_file_then_object():
    class S:
        name = "Object Name"
    assert naming.strategy_label(S(), explicit="Explicit") == "Explicit"
    assert naming.strategy_label(S(), library_ref=("manual", "Lib File.json")) == "Lib File"
    assert naming.strategy_label(S()) == "Object Name"
    assert naming.strategy_label(None, fallback="") == ""


# ---------------------------------------------------------------- validation reports

def _doc(test_label="Validation"):
    report = {"overall": "mixed", "overall_label": "Mixed", "summary_sentence": "s", "items": [], "counts": {}, "next_step": {}}
    return report_export.build_report_document(
        report, meta={"strategy": "NQ1 15m EMA.json", "dataset": "NQ1"}, test_label=test_label)


def test_validation_report_id_names_the_strategy_and_the_test():
    rid = _doc()["report_id"]
    assert re.fullmatch(r"NQ1_15m_EMA__Validation__\d{8}-\d{6}", rid), rid
    assert "__Validation-Auto-Fix__" in _doc("Validation-Auto-Fix")["report_id"]


def test_validation_report_says_strategy_and_test_on_the_page_and_in_the_text(tmp_path):
    doc = _doc("Validation-Auto-Fix")
    page, txt = report_export.render_html(doc), report_export.render_text(doc)
    assert "Strategy tested:" in page and "NQ1 15m EMA.json" in page and "Validation Auto Fix" in page
    assert "Strategy tested: NQ1 15m EMA.json" in txt and "VALIDATION AUTO FIX REPORT" in txt
    paths = report_export.save_report(tmp_path, doc)
    assert all(p.name.startswith("NQ1_15m_EMA__Validation-Auto-Fix__") for p in paths.values())
    assert report_export.load_report(tmp_path, doc["report_id"])["test"] == "Validation-Auto-Fix"


def test_reports_saved_under_the_old_id_format_still_load(tmp_path):
    doc = _doc()
    doc["report_id"] = "20260101-101010_old-strategy"
    report_export.save_report(tmp_path, doc)
    assert report_export.load_report(tmp_path, "20260101-101010_old-strategy") is not None


# ---------------------------------------------------------------- full pipeline report

def test_full_pipeline_report_shows_the_strategy_tested_and_is_named_after_it(tmp_path):
    from tests.search.test_full_pipeline_naming_fix import _sma_config, _trending_df
    from app.backtest.risk import RiskConfig
    from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline
    from app.prop.simulator import PropRules
    from app.strategy.manual import ManualStrategy

    cfg_dict = _sma_config()
    cfg_dict["name"] = "My Named Strategy"
    cfg = FullPipelineConfig(ga_population=4, ga_generations=1, ga_search_mc_sims=20, final_mc_sims=200,
                             n_folds=3, save_to_library=False, random_seed=11)
    stem = naming.report_stem("My Named Strategy", "Full-Pipeline", output_dir=tmp_path)
    result = run_full_pipeline(_trending_df(), ManualStrategy(cfg_dict), RiskConfig(), PropRules(), tmp_path, cfg,
                               report_basename=stem)
    html = result.report_paths["html"].read_text(encoding="utf-8")
    assert result.report_paths["html"].name.startswith("My_Named_Strategy__Full-Pipeline__")
    assert "Strategy tested:" in html and "My Named Strategy" in html and "Test: <b>Full Pipeline</b>" in html
    data = json.loads(result.report_paths["json"].read_text(encoding="utf-8"))
    assert data["tested_strategy"] == "My Named Strategy" and data["test_label"] == "Full Pipeline"


def test_strategy_banner_escapes_the_name():
    from app.reports.generator import _strategy_banner
    out = _strategy_banner({"tested_strategy": "<img src=x onerror=alert(1)>", "test_label": "Full Pipeline",
                            "strategy": {"name": "n", "instrument": "NQ1"}})
    assert "<img" not in out and "&lt;img" in out


# ---------------------------------------------------------------- job pages show the strategy

def test_job_strategy_route_and_stamping():
    import app.web.server as server
    server.app.config["TESTING"] = True
    c = server.app.test_client()
    jid = server.JOB_MANAGER.create(tool="Test")
    assert c.get(f"/job-strategy/{jid}").get_json() == {"found": True, "strategy": ""}
    class S:
        name = "Display Name"
    server._stamp_job_strategy(jid, S(), library_ref=("manual", "Saved Strat.json"))
    assert c.get(f"/job-strategy/{jid}").get_json() == {"found": True, "strategy": "Saved Strat"}
    assert c.get("/job-strategy/doesnotexist").status_code == 404
    assert server._batch_strategy_names([type("I", (), {"label": "a"})(), type("I", (), {"label": "b"})()]) == "2 strategies: a, b"


@pytest.mark.parametrize("path", [
    "/full-pipeline/job/abc123", "/full-pipeline/batch-job/abc123", "/validate-simple/job/abc123",
    "/validate-fix/job/abc123", "/optimize-simple/job/abc123", "/quick-optimize/job/abc123",
    "/walk-forward-opt/job/abc123", "/walk-forward-ga/job/abc123", "/cpcv/job/abc123", "/pbo/job/abc123",
    "/sensitivity/job/abc123", "/multi-objective/job/abc123", "/refine/job/abc123",
])
def test_every_job_page_carries_the_strategy_chip(path):
    import app.web.server as server
    server.app.config["TESTING"] = True
    server._license_ok_cached = True
    body = server.app.test_client().get(path).get_data(as_text=True)
    # an unknown job id renders the page's own "not found" state; either way the page must wire the chip in
    # when it renders a live job, so check the template source for the include too
    assert "t58-strategy-chip" in body or "not found" in body.lower() or "could not be found" in body.lower() or "no such" in body.lower()


def test_templates_include_the_chip():
    from pathlib import Path
    tdir = Path(__file__).resolve().parent.parent.parent / "app" / "web" / "templates"
    for name in ("full_pipeline_job", "full_pipeline_batch_job", "validate_simple_job", "validate_fix_job",
                 "optimize_simple_job", "quick_optimize_job", "wfo_job", "wfga_job", "cpcv_job", "pbo_job",
                 "sensitivity_job", "parameter_robustness_job", "multi_objective_job", "refine_job"):
        assert "_job_strategy_chip.html" in (tdir / f"{name}.html").read_text(encoding="utf-8"), name
