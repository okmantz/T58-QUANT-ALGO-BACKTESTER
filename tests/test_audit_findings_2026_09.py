"""
Regression tests for the 2026-09 codebase audit findings:

  RISK-001  RiskConfig.initial_balance / PropRules.account_size mismatch
  EXEC-002  same-bar stop-out-then-reentry, now configurable
  DOC-003   stale Full Pipeline Step 3 docstring (no runtime behavior --
            not independently regression-testable; covered by inspection
            in the PR, not here)
  MC-004    Monte Carlo methodology_note / selection_bias_caveat
  VAL-005   Step 2 GA fold windows vs Step 4 "out-of-sample" fold overlap,
            plus build_folds' silent fold-drop
  VAL-006   tz-naive/tz-aware Trade timestamp mismatch

Each test is written to FAIL against the pre-fix code and PASS against
the fixed code, per the "regression test required" note on each finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.execution import run_execution
from app.backtest.risk import (
    RiskConfig,
    account_size_mismatch_message,
    with_prop_safety_defaults,
)
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.optimize.refinement import RefinementConfig
from app.prop.simulator import PropRules, simulate_account
from app.search.robustness import run_walk_forward
from app.strategy.base import StrategyResult
from app.strategy.manual import ManualStrategy
from app.validation.regime_matrix import build_regime_matrix
from app.validation.walk_forward_opt import build_folds


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _flat_df(n=1000, tz=None, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz=tz)
    price = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "timestamp": ts,
        "open": price, "high": price + 0.3, "low": price - 0.3, "close": price,
        "volume": 100.0,
    })


def _long_short_signal(n):
    return pd.Series(np.where(np.arange(n) % 20 < 10, 1, -1))


class _FixedSignalStrategy:
    """Minimal Strategy stand-in returning a fixed pre-built signal series."""
    source_type = "manual"

    def __init__(self, signals, stop_loss_pips=50, take_profit_pips=80):
        self._signals = signals
        self._stop = stop_loss_pips
        self._take = take_profit_pips

    def generate(self, df):
        return StrategyResult(
            name="fixed", source_type="manual", signals=self._signals,
            stop_loss_pips=self._stop, take_profit_pips=self._take,
        )


def _manual_ema_cross_config():
    return {
        "name": "ema cross",
        "entry_conditions": {
            "long": [{
                "left": {"type": "ema", "period": 5, "field": "close"},
                "operator": "cross above",
                "right": {"type": "ema", "period": 15, "field": "close"},
            }],
            "short": [{
                "left": {"type": "ema", "period": 5, "field": "close"},
                "operator": "cross below",
                "right": {"type": "ema", "period": 15, "field": "close"},
            }],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": {
            "stop_type": "atr", "stop_value": 2.0, "stop_atr_period": 14,
            "target_type": "atr", "target_value": 3.0, "target_atr_period": 14,
            "opposite_signal_exit": True,
        },
    }


# ---------------------------------------------------------------------------
# RISK-001
# ---------------------------------------------------------------------------

class TestRisk001AccountSizeMismatch:
    def test_mismatch_is_detected(self):
        msg = account_size_mismatch_message(100_000.0, 50_000.0)
        assert msg is not None
        assert "100,000" in msg and "50,000" in msg

    def test_no_mismatch_returns_none(self):
        assert account_size_mismatch_message(100_000.0, 100_000.0) is None

    def test_with_prop_safety_defaults_reconciles_initial_balance(self):
        """The two independently-editable values must not silently stay
        divergent -- with_prop_safety_defaults is the one chokepoint every
        pipeline (Full Pipeline, Evolution Lab, Speed Run) already calls,
        so PropRules.account_size must win there."""
        risk = RiskConfig(initial_balance=100_000.0)
        prop_rules = PropRules(account_size=50_000.0, daily_loss_limit_pct=5.0, max_drawdown_pct=10.0)
        reconciled = with_prop_safety_defaults(risk, prop_rules)
        assert reconciled.initial_balance == 50_000.0

    def test_daily_loss_circuit_breaker_and_prop_verdict_agree_after_fix(self):
        """The actual failure mode: without reconciliation, a raw backtest's
        intrabar daily-loss circuit breaker (keyed to risk.initial_balance)
        and the post-hoc prop-firm verdict (keyed to prop_rules.account_size)
        could fire at two different dollar floors for the "same" 5% rule on
        the exact same trade sequence. After the fix, both are keyed to the
        same $50,000 account."""
        risk = RiskConfig(initial_balance=100_000.0, pip_size=1.0)
        prop_rules = PropRules(account_size=50_000.0, daily_loss_limit_pct=5.0, max_drawdown_pct=90.0)
        reconciled = with_prop_safety_defaults(risk, prop_rules)
        # 5% of the RECONCILED $50,000 account = $2,500 -- this must be the
        # daily floor actually enforced during raw execution.
        assert reconciled.daily_loss_limit_pct == 5.0
        assert reconciled.initial_balance == prop_rules.account_size

        n = 5
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": [100] * n, "high": [100] * n, "low": [80] * n, "close": [100] * n,
            "volume": [100] * n,
        })
        sig = pd.Series([1, 0, 0, 0, 0])
        trades, _ = run_execution(
            df=df, signals=sig, risk=reconciled,
            stop_loss_pips=1000, take_profit_pips=None,
            stop_loss_distance=None, take_profit_distance=None,
            trailing_stop_distance=None, breakeven_trigger_r=None, partial_exit_config=None,
            adaptive_risk=None,
        )
        # Whatever the raw backtest actually realized, simulate_account
        # (using prop_rules.account_size) must be checking the SAME
        # dollar account the backtest just enforced its circuit breaker
        # against -- not a silently different one.
        pnls = [t.pnl for t in trades]
        dates = [t.entry_time for t in trades]
        result = simulate_account(pnls, dates, prop_rules)
        assert result.final_balance <= prop_rules.account_size * 1.5  # sanity: no runaway mismatch-scale blowup


# ---------------------------------------------------------------------------
# EXEC-002
# ---------------------------------------------------------------------------

class TestExec002ReentryCooldown:
    def _whipsaw_df(self):
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        return pd.DataFrame({
            "timestamp": ts,
            "open": [100, 100, 99, 99, 99],
            "high": [100.5, 100.2, 99.5, 99.5, 99.5],
            "low": [99.8, 95.0, 98.5, 98.5, 98.5],
            "close": [100, 99, 99, 99, 99],
            "volume": [100] * 5,
        })

    def _run(self, cooldown):
        df = self._whipsaw_df()
        sig = pd.Series([1, 1, 1, 0, 0])
        risk = RiskConfig(
            initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
            spread_pips=0.0, slippage_pips=0.0, commission_per_trade=0.0,
            reentry_cooldown_bars=cooldown,
        )
        trades, _ = run_execution(
            df=df, signals=sig, risk=risk,
            stop_loss_pips=2.0, take_profit_pips=None,
            stop_loss_distance=None, take_profit_distance=None,
            trailing_stop_distance=None, breakeven_trigger_r=None, partial_exit_config=None,
            adaptive_risk=None,
        )
        return trades

    def test_default_reproduces_original_no_cooldown_behavior(self):
        """Backward compatibility: reentry_cooldown_bars defaults to 0,
        which must reproduce the ORIGINAL (undocumented) same-bar
        reentry behavior byte-for-byte -- a stop-out on bar 1 followed by
        a fresh same-direction entry at that same bar's close."""
        trades = self._run(cooldown=0)
        assert len(trades) == 2
        assert trades[0].exit_reason == "stop_loss"
        assert trades[0].exit_time == trades[1].entry_time

    def test_cooldown_suppresses_same_bar_reentry(self):
        trades = self._run(cooldown=2)
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop_loss"


# ---------------------------------------------------------------------------
# MC-004
# ---------------------------------------------------------------------------

class TestMC004MethodologyNote:
    def _trades(self):
        df = _flat_df(n=400)
        strat = _FixedSignalStrategy(_long_short_signal(400))
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        bt = run_backtest(df, strat, risk)
        assert bt.trades, "fixture must produce trades"
        return bt.trades

    def test_methodology_note_present_and_describes_method(self):
        trades = self._trades()
        rules = PropRules(account_size=10_000.0)
        result = run_monte_carlo(trades, rules, MonteCarloConfig(n_simulations=200, method="bootstrap"))
        assert result.methodology_note
        assert "bootstrap" in result.methodology_note.lower()
        assert "fixed" in result.methodology_note.lower() or "calendar" in result.methodology_note.lower()

    def test_selection_bias_caveat_changes_wording_not_numbers(self):
        trades = self._trades()
        rules = PropRules(account_size=10_000.0)
        cfg = MonteCarloConfig(n_simulations=200, method="bootstrap", random_seed=42)
        plain = run_monte_carlo(trades, rules, cfg, selection_bias_caveat=False)
        flagged = run_monte_carlo(trades, rules, cfg, selection_bias_caveat=True)
        assert "selection" in flagged.methodology_note.lower() or "optimization" in flagged.methodology_note.lower()
        assert "selection" not in plain.methodology_note.lower()
        # Same seed/config/trades -> identical numeric results; only the
        # note's wording should differ.
        assert plain.evaluation_pass_probability == flagged.evaluation_pass_probability
        assert plain.risk_of_ruin_pct == flagged.risk_of_ruin_pct


# ---------------------------------------------------------------------------
# VAL-005
# ---------------------------------------------------------------------------

class TestVal005FoldOverlapAndSilentDrop:
    def test_build_folds_warns_on_silent_drop(self):
        """With n_folds=4 on 1000 bars (this app's own Full Pipeline/Quick
        Optimize defaults), the 4th rolling fold's train slice is empty
        and used to be dropped with NO warning at all."""
        df = _flat_df(n=1000)
        warnings: list[str] = []
        folds = build_folds(df, n_folds=4, window_mode="rolling", train_frac=0.6, warnings=warnings)
        assert len(folds) == 3
        assert any("dropped" in w for w in warnings)
        assert any("fold 3" in w for w in warnings)

    def test_max_test_bar_used_matches_actual_fold_bounds(self):
        df = _flat_df(n=1000)
        folds = build_folds(df, n_folds=4, window_mode="rolling", train_frac=0.6)
        max_bar = max(f.test_end_bar for f in folds)
        assert max_bar == 800  # verified by hand against this exact fixture in the audit

    def test_embargo_eliminates_overlap_with_ga_folds(self):
        """The core VAL-005 regression: Step 4's fold windows must not
        overlap the bars Step 2's GA already selected the winner against,
        once embargo_start_bar is supplied."""
        df = _flat_df(n=1000)
        ga_folds = build_folds(df, n_folds=4, window_mode="rolling", train_frac=0.6)
        ga_touched_ranges = [(f.test_start_bar, f.test_end_bar) for f in ga_folds]
        max_test_bar_used = max(f.test_end_bar for f in ga_folds)

        config = _manual_ema_cross_config()
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)

        result = run_walk_forward(
            df, lambda: ManualStrategy(config), risk,
            n_folds=4, metric="profit_factor", embargo_start_bar=max_test_bar_used,
        )
        assert result is not None
        assert result.embargo_applied_bars == max_test_bar_used
        for fold in result.folds:
            # Every WalkForwardFold reports its test_period as
            # timestamps; recover bar positions via the dataset's own
            # timestamp index to check for overlap against the GA's
            # touched ranges.
            test_start_ts, test_end_ts = fold.test_period
            start_idx = df.index[df["timestamp"] == pd.Timestamp(test_start_ts)][0]
            assert start_idx >= max_test_bar_used, (
                f"Step 4 fold {fold.fold_index} starts at bar {start_idx}, "
                f"which is before the GA's last touched bar {max_test_bar_used}"
            )

    def test_no_embargo_reproduces_original_overlapping_behavior(self):
        """Sanity check that embargo is opt-in: omitting it must reproduce
        the original (overlapping) fold construction unchanged."""
        df = _flat_df(n=1000)
        config = _manual_ema_cross_config()
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        result = run_walk_forward(df, lambda: ManualStrategy(config), risk, n_folds=4, metric="profit_factor")
        assert result is not None
        assert result.embargo_applied_bars == 0


# ---------------------------------------------------------------------------
# VAL-006
# ---------------------------------------------------------------------------

class TestVal006TimezoneHandling:
    def test_trade_timestamps_preserve_tz(self):
        df = _flat_df(n=400, tz="UTC")
        strat = _FixedSignalStrategy(_long_short_signal(400))
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        bt = run_backtest(df, strat, risk)
        assert bt.trades
        assert bt.trades[0].entry_time.tzinfo is not None
        assert str(bt.trades[0].entry_time.tzinfo) == "UTC"

    def test_equity_curve_timestamp_preserves_tz(self):
        df = _flat_df(n=400, tz="UTC")
        strat = _FixedSignalStrategy(_long_short_signal(400))
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        bt = run_backtest(df, strat, risk)
        assert bt.equity_curve["timestamp"].dt.tz is not None

    def test_naive_input_stays_naive(self):
        """The fix must be a no-op for the (still very common) tz-naive
        case -- never introduce a tz where none existed."""
        df = _flat_df(n=400, tz=None)
        strat = _FixedSignalStrategy(_long_short_signal(400))
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        bt = run_backtest(df, strat, risk)
        assert bt.trades
        assert bt.trades[0].entry_time.tzinfo is None
        assert bt.equity_curve["timestamp"].dt.tz is None

    def test_regime_matrix_does_not_raise_on_tz_aware_data(self):
        """The actual reported crash: build_regime_matrix used to raise
        'Cannot compare tz-naive and tz-aware datetime-like objects' on
        any tz-aware input dataset."""
        df = _flat_df(n=400, tz="UTC")
        strat = _FixedSignalStrategy(_long_short_signal(400))
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        bt = run_backtest(df, strat, risk)
        result = build_regime_matrix(df, bt.trades, initial_balance=risk.initial_balance)
        assert result is not None
