"""v9.6 desktop lifecycle pass — source-level checks (headless, no Tk).

These tests parse app/ui/main_window.py as TEXT only: they never import
it and never instantiate a widget, so they run with no display. They pin
the approved sidebar reorganization (Lifecycle umbrella, ①-⑦ stages,
moved/new pointer entries, ACCOUNT reorder), the RunContextPanel
hardening hook, and the existence of the three new "pick your ..."
pages.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = (ROOT / "app" / "ui" / "main_window.py").read_text(encoding="utf-8")


def _nav_block() -> str:
    """The self._nav_items = [ ... ] literal, as text."""
    start = SOURCE.index("self._nav_items = [")
    end = SOURCE.index("]\n", start)
    return SOURCE[start:end]


def _key_pos(block: str, key: str) -> int:
    m = re.search(rf'\(\s*"{re.escape(key)}"\s*,', block)
    assert m, f"nav key {key!r} not found in _nav_items"
    return m.start()


def _header_pos(block: str, numeral: str) -> int:
    """Position of a circled-numeral stage header line, e.g. _header_pos(block, "①")."""
    m = re.search(rf'\(None, None, "\\u{ord(numeral):04x}', block)
    assert m, f"stage header {numeral!r} not found in _nav_items"
    return m.start()


def test_lifecycle_umbrella_replaces_strategy_lab():
    assert '(None, "SUPERHEADER", "Lifecycle"' in _nav_block()
    assert '(None, "SUPERHEADER", "Strategy Lab"' not in _nav_block()


def test_autopilot_pointer_lives_in_create():
    block = _nav_block()
    create = _header_pos(block, "①")
    test_hdr = _header_pos(block, "②")
    pos = _key_pos(block, "autopilot_pointer")
    assert create < pos < test_hdr


def test_full_pipeline_pointer_right_after_run_and_report():
    block = _nav_block()
    test_hdr = _header_pos(block, "②")
    opt_hdr = _header_pos(block, "③")
    run = _key_pos(block, "run")
    ptr = _key_pos(block, "fullpipeline_test_pointer")
    payout = _key_pos(block, "payout")
    assert test_hdr < run < ptr < payout < opt_hdr
    # The pointer must aim at the real Full Pipeline frame.
    assert '("fullpipeline_test_pointer", "", "Full Pipeline (all-in-one)", self.tab_fullpipeline' in block


def test_optimize_simple_right_after_start_here():
    block = _nav_block()
    opt_hdr = _header_pos(block, "③")
    val_hdr = _header_pos(block, "④")
    hub = _key_pos(block, "optimizehub")
    simple = _key_pos(block, "optimize_simple")
    full = _key_pos(block, "fullpipeline")
    assert opt_hdr < hub < simple < full < val_hdr
    assert '("optimize_simple", "", "\\u2726 Optimize (pick your engines)", self.tab_optimize_simple' in block


def test_validate_simple_and_youridea_pointer_in_validate():
    block = _nav_block()
    val_hdr = _header_pos(block, "④")
    champ_hdr = _header_pos(block, "⑤")
    hub = _key_pos(block, "validatehub")
    simple = _key_pos(block, "validate_simple")
    idea = _key_pos(block, "youridea_validate_pointer")
    wfo = _key_pos(block, "wfo")
    assert val_hdr < hub < simple < idea < wfo < champ_hdr
    assert '("validate_simple", "", "\\u2726 Validate (pick your checks)", self.tab_validate_simple' in block
    assert '("youridea_validate_pointer", "", "\\U0001F4A1 Your Idea", self.tab_youridea' in block


def test_champion_simple_and_moved_entries_in_champion():
    block = _nav_block()
    champ_hdr = _header_pos(block, "⑤")
    dep_hdr = _header_pos(block, "⑥")
    hub = _key_pos(block, "starthere_champion")
    simple = _key_pos(block, "champion_simple")
    ensemble = _key_pos(block, "ensemble")
    compare = _key_pos(block, "compare")
    health = _key_pos(block, "strathealth_pointer")
    assert champ_hdr < hub < simple < ensemble < compare < health < dep_hdr
    assert '("champion_simple", "", "\\u2726 Champion Board (promote)", self.tab_champion_simple' in block


def test_deployment_keeps_only_live_entries():
    block = _nav_block()
    dep_hdr = _header_pos(block, "⑥")
    grave_hdr = _header_pos(block, "⑦")
    for key in ("starthere_deployment", "forwardtest", "deploylive", "livemarket", "replay"):
        pos = _key_pos(block, key)
        assert dep_hdr < pos < grave_hdr
    assert '"Champion Checks"' not in block


def test_account_section_order():
    block = _nav_block()
    positions = {}
    for key in ("starthere_account", "account", "notifications", "apikeys", "datacenter", "support"):
        positions[key] = _key_pos(block, key)
    order = ["starthere_account", "account", "notifications", "apikeys", "datacenter", "support"]
    seq = [positions[k] for k in order]
    assert seq == sorted(seq), f"ACCOUNT order wrong: {positions}"
    # Notification Settings point at the Account frame (no separate frame exists).
    assert '("notifications", "", "Notification Settings", self.tab_account' in block


def test_new_tab_frames_registered():
    for frame in ("tab_optimize_simple", "tab_validate_simple", "tab_champion_simple"):
        assert f"self.{frame} = Frame(self.content, bg=BG)" in SOURCE
        # Registered in _all_tab_frames (so _show_page hides the others correctly).
        all_frames_span = SOURCE.index("self._all_tab_frames: list = []")
        nav_start = SOURCE.index("self._nav_items = [")
        assert f"self.{frame}" in SOURCE[all_frames_span:nav_start]


def test_new_page_builders_exist():
    for name in ("_build_optimize_simple_tab", "_build_validate_simple_tab", "_build_champion_simple_tab"):
        assert f"def {name}(self):" in SOURCE


def test_build_risk_config_hardened_through_run_context():
    m = re.search(r"def build_risk_config\(self\) -> RiskConfig:(.*?)\n    def ", SOURCE, re.S)
    assert m, "RunContextPanel.build_risk_config not found"
    assert "build_run_context" in m.group(1)
    assert "build_run_context(risk, self.build_prop_rules())" in m.group(1)


def test_build_run_context_imported():
    assert "from app.backtest.risk import" in SOURCE
    line = next(l for l in SOURCE.splitlines() if "from app.backtest.risk import" in l)
    assert "build_run_context" in line


def test_lse_dead_field_removed():
    assert "lse_asset_class" not in SOURCE
    assert "LSE_ASSET_CLASSES" not in SOURCE


def test_dashboard_primary_action_lifecycle_aware():
    assert 'self._show_page("optimize_simple")' in SOURCE
    assert 'self._show_page("champion_simple")' in SOURCE
    assert '"\\u25b6 RUN FULL PIPELINE"' in SOURCE
    # The old always-OPEN-VALIDATE fallback is gone.
    assert '"OPEN VALIDATE", lambda: self._show_page("cpcv")' not in SOURCE
