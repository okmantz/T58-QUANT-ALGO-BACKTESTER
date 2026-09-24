"""Tests for the regime-aware-throttle-vs-iid-resampling fix:
app.monte_carlo.engine.default_method_for_adaptive_risk."""
from __future__ import annotations

from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskRule
from app.monte_carlo.engine import default_method_for_adaptive_risk


def _enabled_adaptive_risk() -> AdaptiveRiskConfig:
    return AdaptiveRiskConfig(
        enabled=True,
        rules=[AdaptiveRiskRule(trigger="consecutive_losses", threshold=2, risk_multiplier=0.5)],
    )


def test_none_adaptive_risk_keeps_bootstrap_default():
    assert default_method_for_adaptive_risk(None) == "bootstrap"


def test_disabled_adaptive_risk_keeps_bootstrap_default():
    cfg = AdaptiveRiskConfig(enabled=False, rules=[])
    assert default_method_for_adaptive_risk(cfg) == "bootstrap"


def test_enabled_adaptive_risk_switches_to_block_bootstrap():
    assert default_method_for_adaptive_risk(_enabled_adaptive_risk()) == "block_bootstrap"


def test_object_without_enabled_attribute_is_safe():
    class NotAdaptiveRisk:
        pass

    assert default_method_for_adaptive_risk(NotAdaptiveRisk()) == "bootstrap"
