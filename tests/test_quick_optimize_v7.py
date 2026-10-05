"""v7 (2026-10-05, worker B) -- Workstream D: Quick Optimize must actually
optimize.

Regression test: on synthetic data with a PLANTED, stationary edge (strong
mean reversion a Bollinger fade can harvest), Quick Optimize -- starting
from a deliberately bad parameterization -- must IMPROVE the strategy on
held-out data, not merely change it. The comparison is winner-vs-baseline
on the same untouched holdout slice, so in-sample luck can't pass it.

Also covers the honest gate ledger (app.search.quickopt_gates): the
ran/skipped lists must actually reflect which gates ran.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.search.quickopt_gates import quickopt_gates_summary
from app.strategy.manual import ManualStrategy


def make_ou_es(n_bars=4000, seed=123, start_price=6000.0):
    """Strongly mean-reverting ES-scale 5-min data (stationary edge).

    Ornstein-Uhlenbeck around `start_price`: a Bollinger-band fade with a
    sensible lookback harvests it reliably; a too-twitchy lookback gets
    whipsawed. Deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    theta, sigma = 0.08, 2.2  # strong pull to the mean, $2.2/bar noise
    px = np.empty(n_bars)
    px[0] = start_price
    for i in range(1, n_bars):
        px[i] = px[i - 1] + theta * (start_price - px[i - 1]) + rng.normal(0, sigma)
    spread = np.abs(rng.normal(0.5, 0.2, n_bars)) + 0.2
    open_ = np.concatenate([[px[0]], px[:-1]]) + rng.normal(0, 0.25, n_bars)
    high = np.maximum(open_, px) + spread * rng.random(n_bars)
    low = np.minimum(open_, px) - spread * rng.random(n_bars)
    ts = pd.date_range("2026-09-01", periods=n_bars, freq="5min", tz="America/Chicago")
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": px, "volume": np.full(n_bars, 1500),
    })


def make_bad_fade_strategy(bb_period=15):
    """Bollinger fade with a deliberately bad risk setup.

    The edge (OU mean reversion) is real, but the stop is strangling it:
    stop_value=0.5 with target 1.5x is far too tight -- the GA should
    discover that widening the stop dramatically improves both net profit
    and eval-pass probability. The bb_period is fixed at a reasonable 15
    so the test isolates the risk-parameter optimization.

    NOTE: bb_period must be >= 8 -- a 5-period 2-std band is mathematically
    untouchable (max z-score of the last value in a 5-bar window is < 2.0),
    so period 5 would generate zero signals, not a bad strategy.
    """
    def _c(left, op, right):
        return {"left": left, "operator": op, "right": right}
    bb = lambda side: {"type": f"bollinger_{side}", "period": bb_period, "field": "close"}  # noqa: E731
    return ManualStrategy({
        "name": f"planted-edge fade (bb_period={bb_period}, tight stop)",
        "entry_conditions": {
            "long": [_c({"type": "close", "period": 1}, "<", bb("lower"))],
            "short": [_c({"type": "close", "period": 1}, ">", bb("upper"))],
        },
        "exit_conditions": {
            "long": [_c({"type": "close", "period": 1}, ">", bb("mid"))],
            "short": [_c({"type": "close", "period": 1}, "<", bb("mid"))],
        },
        "risk_management": {
            # Deliberately bad: 0.5x ATR stop is strangling the edge.
            "stop_type": "atr", "stop_value": 0.5, "stop_atr_period": 14,
            "target_type": "atr", "target_value": 0.75, "target_atr_period": 14,
            "opposite_signal_exit": True, "max_bars_in_trade": 48,
        },
    })


def _risk_and_rules():
    risk = RiskConfig(initial_balance=100_000, pip_size=1.0,
                      risk_mode="percent", risk_value=1.0)
    rules = PropRules(account_size=100_000)
    return risk, rules


def test_quick_optimize_improves_on_held_out_data():
    from app.orchestration.quick_optimize import QuickOptimizeConfig, run_quick_optimize

    # 8000 bars: enough trades for the Monte Carlo eval-pass fitness to
    # have gradient (with 4000 bars every candidate scored 0%, blinding
    # the GA). The OU edge is stationary, so the train-optimal period
    # generalizes to the holdout.
    df = make_ou_es(n_bars=8000)
    cut = int(len(df) * 0.8)
    holdout_df = df.iloc[cut:].reset_index(drop=True)
    risk, rules = _risk_and_rules()
    baseline_strategy = make_bad_fade_strategy(bb_period=15)

    # Sanity: the bad baseline actually trades and the edge is real.
    base_holdout_bt = run_backtest(holdout_df, baseline_strategy, risk)
    assert len(base_holdout_bt.trades) >= 20, "planted edge must trade on holdout"
    baseline_holdout_net = base_holdout_bt.statistics.net_profit

    cfg = QuickOptimizeConfig(
        ga_population=16, ga_generations=5, n_folds=3,
        ga_search_mc_sims=50, final_mc_sims=200,
        fitness_metric="eval_pass_probability",
        reserve_holdout=True, holdout_frac=0.2,
        save_to_library=False, random_seed=42,
    )
    result = run_quick_optimize(df, baseline_strategy, risk, rules, cfg)

    assert result.holdout_enabled
    assert result.holdout_trades and result.holdout_trades > 0
    # THE regression assertion: the optimized winner beats the unoptimized
    # baseline on data NEITHER ever saw during selection.
    assert result.holdout_net_profit is not None
    assert result.holdout_net_profit > baseline_holdout_net, (
        f"Quick Optimize did not improve on held-out data: winner "
        f"${result.holdout_net_profit:,.2f} vs baseline ${baseline_holdout_net:,.2f}"
    )
    # And it must say plainly that this is NOT a finished validation.
    assert result.validated is False
    assert "NOT OOS VALIDATED" in result.result_banner


def test_quickopt_gates_summary_marks_skipped_gates():
    class _FakeResult:
        icir_gate = None
        icir_gate_skip_reason = "too few trades"
        lookahead_note = None
        holdout_enabled = False

    s = quickopt_gates_summary(_FakeResult())
    assert any("Walk-forward-scored" in r for r in s["ran"])
    assert any("Prop-rule-aware" in r for r in s["ran"])
    assert any("ICIR" in x and "could not run" in x for x in s["skipped"])
    assert any("DSR / PBO" in x for x in s["skipped"])
    assert any("Holdout check (not requested" in x for x in s["skipped"])
    assert any("Lookahead-bias recheck (did not run)" in x for x in s["skipped"])


def test_quickopt_gates_summary_marks_ran_gates():
    class _Gate:
        pass

    class _FakeResult:
        icir_gate = _Gate()
        icir_gate_skip_reason = None
        lookahead_note = "recheck ran"
        holdout_enabled = True

    s = quickopt_gates_summary(_FakeResult())
    assert any("ICIR" in r for r in s["ran"])
    assert any("Lookahead-bias recheck" in r for r in s["ran"])
    assert any("Light holdout check" in r for r in s["ran"])
    assert not any("DSR / PBO" in r for r in s["ran"])
