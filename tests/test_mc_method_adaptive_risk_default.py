"""Tests for the regime-aware-throttle-vs-iid-resampling fix:
app.monte_carlo.engine.default_method_for_adaptive_risk.

P1-5 (2026-10-03): block bootstrap is now the engine-wide default
resampling (see MonteCarloConfig.method), so this helper returns
"block_bootstrap" for every input -- the i.i.d. "bootstrap" default it
used to fall back to for non-adaptive callers is gone. Explicit
method="bootstrap" remains the opt-out for callers that want the old
i.i.d. behavior.
"""
from __future__ import annotations

from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskRule
from app.monte_carlo.engine import default_method_for_adaptive_risk


def _enabled_adaptive_risk() -> AdaptiveRiskConfig:
    return AdaptiveRiskConfig(
        enabled=True,
        rules=[AdaptiveRiskRule(trigger="consecutive_losses", threshold=2, risk_multiplier=0.5)],
    )


def test_none_adaptive_risk_uses_block_bootstrap_default():
    assert default_method_for_adaptive_risk(None) == "block_bootstrap"


def test_disabled_adaptive_risk_uses_block_bootstrap_default():
    cfg = AdaptiveRiskConfig(enabled=False, rules=[])
    assert default_method_for_adaptive_risk(cfg) == "block_bootstrap"


def test_enabled_adaptive_risk_switches_to_block_bootstrap():
    assert default_method_for_adaptive_risk(_enabled_adaptive_risk()) == "block_bootstrap"


def test_object_without_enabled_attribute_is_safe():
    class NotAdaptiveRisk:
        pass

    assert default_method_for_adaptive_risk(NotAdaptiveRisk()) == "block_bootstrap"
