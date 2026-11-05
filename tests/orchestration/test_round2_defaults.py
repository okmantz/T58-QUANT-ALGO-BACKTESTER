import re
from pathlib import Path

from app.orchestration.full_pipeline import FullPipelineConfig
from app.prop.presets import FIRM_FEES, PROP_FIRM_PRESETS

ROOT = Path(__file__).resolve().parents[2]


def test_adaptive_risk_is_on_by_default_for_the_full_pipeline():
    assert FullPipelineConfig().adaptive_risk_enabled is True


def test_test_page_adaptive_checkbox_is_pre_ticked():
    html = (ROOT / "app/web/templates/full_pipeline.html").read_text(encoding="utf-8")
    box = re.search(r'<input[^>]*name="adaptive_risk_enabled"[^>]*>', html).group(0)
    assert "checked" in box


def test_new_presets_exist_are_unverified_and_have_fee_rows():
    presets = PROP_FIRM_PRESETS if isinstance(PROP_FIRM_PRESETS, dict) else {p.key: p for p in PROP_FIRM_PRESETS}
    for key in ("e8_one_25k", "lucid_flex_25k", "lucid_flex_50k", "lucid_flex_100k"):
        assert key in presets, key
        assert key in FIRM_FEES and FIRM_FEES[key].challenge_price is None
        assert "UNVERIFIED" in presets[key].source_note
    assert presets["e8_one_25k"].account_size == 25_000
    assert presets["lucid_flex_25k"].consistency_rule_pct == 50.0
