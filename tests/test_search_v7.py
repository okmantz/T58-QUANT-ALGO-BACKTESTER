"""v7 (2026-10-05, worker B) -- Search Lab P1 fixes.

Covers:
  fix #7  -- pip_size auto-apply logic (app.search.instrument_risk),
             grammar default-on wiring markers in the web template.
  fix #9  -- raised default search budgets (config + template defaults).
  fix #10 -- evolution exploitation: population/immigrant defaults and
             the n_children > 0 unit-level guard.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import app.search.batch_runner as _br_mod

REPO_ROOT = Path(_br_mod.__file__).resolve().parent.parent.parent


def _es_df(n=1500, seed=11):
    import numpy as np
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.0005, n)
    close = 6000.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    ts = pd.date_range("2026-09-01", periods=n, freq="5min", tz="America/Chicago")
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": np.full(n, 1000),
    })


# ---------------------------------------------------------------------------
# fix #7 -- resolve_leg_risk
# ---------------------------------------------------------------------------

def test_resolve_leg_risk_untouched_default_es_gets_spec():
    from app.backtest.risk import RiskConfig
    from app.search.instrument_risk import resolve_leg_risk
    df = _es_df()
    out, notes = resolve_leg_risk(RiskConfig(pip_size=0.0001), df, "ES")
    assert out.pip_size == 1.0
    assert out.contract_size == 50.0
    assert out.commission_per_contract > 0.0
    assert any("!!!" in n for n in notes), "must be LOUD"


def test_resolve_leg_risk_untouched_default_fx_data_keeps_fx_scale():
    from app.backtest.risk import RiskConfig
    from app.search.instrument_risk import resolve_leg_risk
    df = _es_df()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] / 5500.0
    out, notes = resolve_leg_risk(RiskConfig(pip_size=0.0001), df, "EURUSD")
    assert out.pip_size == 0.0001
    assert notes, "a decision must always be logged"


def test_resolve_leg_risk_unknown_label_falls_back_to_data():
    from app.backtest.risk import RiskConfig
    from app.search.instrument_risk import resolve_leg_risk
    df = _es_df()
    out, notes = resolve_leg_risk(RiskConfig(pip_size=0.0001), df, "MYSTERY")
    assert out.pip_size == 1.0  # data-detected ES scale
    assert out.contract_size is None  # not lot-rounded without a spec
    assert any("!!!" in n for n in notes)


def test_resolve_leg_risk_explicit_pip_size_never_overridden():
    from app.backtest.risk import RiskConfig
    from app.search.instrument_risk import resolve_leg_risk
    df = _es_df()
    out, notes = resolve_leg_risk(RiskConfig(pip_size=0.01), df, "ES")
    assert out.pip_size == 0.01, "explicit user value must survive"
    # ... but unset dependent fields may still be filled from the spec
    assert out.contract_size == 50.0
    assert any("kept your explicit" in n for n in notes)


def test_resolve_leg_risk_data_detected_branch_clears_contract_size():
    # Mirrors app.data.instrument_specs.resolve_risk_per_market's documented
    # semantics: with no known contract to anchor to, a contract $/point
    # typed for a different instrument would silently mis-size this leg, so
    # the data-detected branch clears it (positions are not lot-rounded).
    from app.backtest.risk import RiskConfig
    from app.search.instrument_risk import resolve_leg_risk
    df = _es_df()
    out, _ = resolve_leg_risk(RiskConfig(pip_size=0.0001, contract_size=25.0), df, "MYSTERY")
    assert out.contract_size is None
    assert out.pip_size == 1.0  # the pip itself was still auto-fixed


def test_resolve_leg_risk_never_raises():
    from app.backtest.risk import RiskConfig
    from app.search.instrument_risk import resolve_leg_risk
    out, notes = resolve_leg_risk(RiskConfig(), None, None)
    assert out is not None and notes


# ---------------------------------------------------------------------------
# fix #9 -- raised default budgets
# ---------------------------------------------------------------------------

def test_search_stage_config_budget_defaults_raised():
    from app.search.batch_runner import SearchStageConfig
    cfg = SearchStageConfig()
    assert cfg.ga_population == 40
    assert cfg.ga_generations == 15


def test_search_template_budget_defaults_raised():
    html = (REPO_ROOT / "app" / "web" / "templates" / "search.html").read_text()
    assert 'name="max_candidates" value="1000"' in html
    assert 'name="ga_population" value="40"' in html
    assert 'name="ga_generations" value="15"' in html


def test_search_template_grammar_toggle_default_on():
    html = (REPO_ROOT / "app" / "web" / "templates" / "search.html").read_text()
    assert 'name="invent_structures"' in html
    # the checkbox must be checked by default (grammar default-on for web)
    import re
    m = re.search(r'<input[^>]*name="invent_structures"[^>]*>', html)
    assert m is not None
    assert "checked" in m.group(0)


def test_search_template_pip_autodetect_wired():
    html = (REPO_ROOT / "app" / "web" / "templates" / "search.html").read_text()
    # Auto-detection must be WIRED (either via the shared partial or an
    # inline auto-trigger); the exact mechanism is an implementation detail.
    assert ("/data/detect-pip-size" in html and "existing_dataset" in html)
    assert ("_pip_size_autodetect.html" in html
            or "auto-run pip-size detection" in html.lower()
            or "addEventListener('change'" in html)


def test_lab_templates_pip_autodetect_wired():
    cases = [
        "evolution.html",
        "search_multi_instrument.html",
        "evolution_multi_instrument.html",
        "speed_run.html",
        "speed_run_multi_instrument.html",
        "quick_optimize.html",
        "full_pipeline.html",
        "refine.html",
        "wfga.html",
        "multi_market.html",
        "forge.html",
    ]
    for template in cases:
        html = (REPO_ROOT / "app" / "web" / "templates" / template).read_text()
        # Must POST to the detect endpoint on dataset selection -- via the
        # shared partial, an inline auto-trigger, or a t58DetectPipSize call
        # wired to a change listener.
        has_partial = "_pip_size_autodetect.html" in html
        if has_partial:
            # The endpoint reference lives in the partial itself.
            partial = (REPO_ROOT / "app" / "web" / "templates" / "_pip_size_autodetect.html").read_text()
            assert "/data/detect-pip-size" in partial, template
            continue
        has_auto_trigger = ("addEventListener('change'" in html
                            or 'addEventListener("change"' in html)
        has_detect_call = ("/data/detect-pip-size" in html
                           or "DetectPipSize" in html)
        assert has_auto_trigger and has_detect_call, template


# ---------------------------------------------------------------------------
# fix #10 -- evolution exploitation
# ---------------------------------------------------------------------------

def test_evolution_population_defaults_raised():
    from app.evolution.engine import EvolutionConfig
    cfg = EvolutionConfig()
    assert cfg.population_size == 200
    assert cfg.min_immigrants_per_family == 1


def test_evolution_n_children_positive_at_defaults():
    """Unit-level guard for fix #10: at default config, the elite-children
    loop must actually get budget (n_children > 0) instead of the
    immigrant floor eating the whole population.

    Mirrors _generate_population's documented arithmetic (see
    app.evolution.engine's v6 W1-A5 block): with elites present,
    n_immigrants = population_size * random_immigrant_frac, each family
    (plus the grammar pseudo-family) takes at least
    min_immigrants_per_family, structural breeding takes its dedicated
    max(2, 15%) budget first, and whatever is left goes to elite
    children. Uses the REAL defaults and the REAL family registry, so a
    future default/budget regression fails here, not silently in prod.
    """
    from app.evolution.engine import EvolutionConfig
    from app.search.strategy_space import list_families
    cfg = EvolutionConfig()
    n_fam = len(list_families()) + 1  # +1: the grammar pseudo-family
    n_immigrants = max(1, int(cfg.population_size * cfg.random_immigrant_frac))
    base_per_family = max(cfg.min_immigrants_per_family, n_immigrants // n_fam)
    immigrant_floor = base_per_family * n_fam
    structural_budget = max(2, int(cfg.population_size * 0.15)) if cfg.use_structural_operators else 0
    n_children = max(0, cfg.population_size - (immigrant_floor + structural_budget))
    assert n_children > 0, (
        f"elite compounding starved at defaults: population={cfg.population_size}, "
        f"immigrant_floor={immigrant_floor}, structural={structural_budget}, "
        f"n_children={n_children}"
    )


def test_evolution_template_population_default_raised():
    html = (REPO_ROOT / "app" / "web" / "templates" / "evolution.html").read_text()
    assert 'name="population_size" value="200"' in html
