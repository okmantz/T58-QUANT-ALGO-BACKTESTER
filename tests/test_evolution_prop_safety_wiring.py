"""Tests for the EvolutionRunner -> with_prop_safety_defaults wiring fix
(2026-09-12): Evolution Lab's raw backtest stage previously ran with no
daily-loss / drawdown circuit breaker at all, unlike Speed Run and Full
Pipeline, which already both call with_prop_safety_defaults(). See
app.backtest.risk.with_prop_safety_defaults and
app.evolution.engine.EvolutionRunner.__init__ for the fix itself."""
from app.backtest.risk import RiskConfig
from app.evolution.engine import EvolutionConfig, EvolutionRunner
from app.prop.simulator import PropRules


def test_evolution_runner_wires_daily_loss_limit_from_prop_rules():
    risk = RiskConfig(initial_balance=25_000.0)
    rules = PropRules(account_size=25_000.0, daily_loss_limit_pct=5.0, max_drawdown_pct=10.0)
    runner = EvolutionRunner(df=None, risk=risk, prop_rules=rules, cfg=EvolutionConfig(population_size=4))
    assert runner.risk.daily_loss_limit_pct == 5.0
    assert runner.risk.max_account_drawdown_pct == 10.0
    # caller's original RiskConfig must be left untouched
    assert risk.daily_loss_limit_pct is None


def test_evolution_runner_never_overrides_explicit_risk_settings():
    risk = RiskConfig(initial_balance=25_000.0, daily_loss_limit_pct=2.0, max_account_drawdown_pct=6.0)
    rules = PropRules(account_size=25_000.0, daily_loss_limit_pct=5.0, max_drawdown_pct=10.0)
    runner = EvolutionRunner(df=None, risk=risk, prop_rules=rules, cfg=EvolutionConfig(population_size=4))
    assert runner.risk.daily_loss_limit_pct == 2.0
    assert runner.risk.max_account_drawdown_pct == 6.0
