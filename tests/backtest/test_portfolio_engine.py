"""Unified portfolio engine: parity with run_backtest / run_portfolio_backtest, clock, floating equity, caps."""
import dataclasses
import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.portfolio_engine import (
    EngineLeg, PortfolioEngineConfig, PortfolioEngineError, run_portfolio_engine,
)
from app.backtest.risk import RiskConfig
from app.data.portfolio_clock import PortfolioClockError, build_portfolio_clock
from app.portfolio.floating import floating_equity_overlay
from app.portfolio.portfolio import (
    InstrumentLeg, PortfolioConfig, run_portfolio_backtest, run_unified_portfolio,
)
from app.strategy.base import Strategy, StrategyResult


def _df(n=1200, seed=1, start=1.10, freq="30min", t0="2024-01-01"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(t0, periods=n, freq=freq)
    price, rows = start, []
    for i in range(n):
        o = price
        c = o + rng.normal(0.00002, 0.0007) * (start / 1.1)
        h = max(o, c) + abs(rng.normal(0, 0.0004)) * (start / 1.1)
        l = min(o, c) - abs(rng.normal(0, 0.0004)) * (start / 1.1)
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _signals(n, seed, active=None):
    rng = np.random.default_rng(seed + 1000)
    sig = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        run = int(rng.integers(3, 40))
        sig[i:i + run] = int(rng.choice([-1, 0, 1], p=[0.35, 0.3, 0.35]))
        i += run
    if active is not None:
        mask = np.zeros(n, dtype=bool)
        mask[active[0]:active[1]] = True
        sig[~mask] = 0
    return pd.Series(sig)


class _Fixed(Strategy):
    source_type = "python"

    def __init__(self, sig, sl=30.0, tp=60.0):
        self.sig, self.sl, self.tp = sig, sl, tp

    def generate(self, df):
        return StrategyResult(name="fixed", source_type="python", signals=self.sig,
                              stop_loss_pips=self.sl, take_profit_pips=self.tp)


def _key(t):
    d = dataclasses.asdict(t)
    d.pop("instrument", None)
    return {k: (str(v) if k in ("entry_time", "exit_time") else v) for k, v in d.items()}


RISKS = [
    RiskConfig(),
    RiskConfig(entry_fill_lag_bars=1, spread_pips=1.0, slippage_pips=0.3, commission_per_contract=2.0, contract_size=10.0),
    RiskConfig(daily_loss_limit_pct=1.0, max_account_drawdown_pct=4.0, reentry_cooldown_bars=2),
    RiskConfig(profit_target_pct=2.0, block_weekend_hold=True, risk_value=2.0),
]


@pytest.mark.parametrize("ri", range(len(RISKS)))
def test_one_leg_matches_run_backtest(ri):
    risk = RISKS[ri]
    df = _df(seed=ri + 3)
    st = _Fixed(_signals(len(df), ri))
    ref = run_backtest(df, st, risk, lookahead_check="skip")
    got = run_portfolio_engine([EngineLeg("A", df, st, risk)],
                               PortfolioEngineConfig(account=risk, lookahead_check="skip"))
    assert len(ref.trades) > 5
    assert [_key(t) for t in ref.trades] == [_key(t) for t in got.trades]
    assert ref.equity_curve["equity"].tolist() == got.equity_curve["equity"].tolist()
    assert ref.statistics.net_profit == pytest.approx(got.statistics.net_profit)
    assert all(t.instrument == "A" for t in got.trades)


def test_two_legs_without_overlap_match_standalone_runs_and_legacy_combination():
    n = 1200
    dfa, dfb = _df(n, 1), _df(n, 2, start=1.3)
    sa = _Fixed(_signals(n, 1, active=(0, n // 2 - 60)))
    sb = _Fixed(_signals(n, 2, active=(n // 2 + 60, n)))
    risk = RiskConfig()
    legs = [InstrumentLeg("A", dfa, sa, risk), InstrumentLeg("B", dfb, sb, risk)]
    legacy = run_portfolio_backtest(legs, PortfolioConfig(initial_balance=risk.initial_balance))
    weights = {l.name: l.final_weight for l in legacy.legs}
    eng = [EngineLeg(l.name, l.df, l.strategy, l.risk, weight=weights[l.name]) for l in legs]
    got = run_portfolio_engine(eng, PortfolioEngineConfig(account=risk, size_basis="leg", lookahead_check="skip"))

    assert sorted(_key(t)["pnl"] for t in got.trades) == pytest.approx(sorted(t.pnl for t in legacy.combined_trades))
    assert sorted(t.exit_time for t in got.trades) == sorted(t.exit_time for t in legacy.combined_trades)
    assert got.statistics.net_profit == pytest.approx(legacy.combined_statistics.net_profit)
    assert got.portfolio_stats["max_concurrent_positions"] == 1


def test_overlapping_legs_share_one_account_and_mark_to_market():
    n = 1200
    legs = [EngineLeg(f"L{i}", _df(n, i + 5, start=1.1 + i), _Fixed(_signals(n, i + 5)), RiskConfig()) for i in range(3)]
    res = run_portfolio_engine(legs, PortfolioEngineConfig(lookahead_check="skip"))
    assert res.portfolio_stats["max_concurrent_positions"] >= 2
    eq = res.equity_curve["equity"].to_numpy()
    assert len(eq) == len(res.clock)
    # floating drawdown is never smaller than the realised-only drawdown
    closed = floating_equity_overlay(res.trades, res.initial_balance)
    assert closed.worst_case_max_dd >= closed.closed_max_dd
    assert res.statistics.total_trades == len(res.trades)


def test_capacity_caps_are_enforced():
    n = 1200
    mk = lambda cap: run_portfolio_engine(
        [EngineLeg(f"L{i}", _df(n, i + 5, start=1.1 + i), _Fixed(_signals(n, i + 5)), RiskConfig()) for i in range(3)],
        PortfolioEngineConfig(max_open_positions=cap, lookahead_check="skip"))
    assert mk(1).portfolio_stats["max_concurrent_positions"] == 1
    assert mk(2).portfolio_stats["max_concurrent_positions"] <= 2
    assert mk(None).portfolio_stats["max_concurrent_positions"] >= 2


def test_unified_wrapper_on_portfolio_page_api():
    n = 900
    legs = [InstrumentLeg("A", _df(n, 1), _Fixed(_signals(n, 1)), RiskConfig()),
            InstrumentLeg("B", _df(n, 2, start=1.3), _Fixed(_signals(n, 2)), RiskConfig())]
    res = run_unified_portfolio(legs, PortfolioConfig(initial_balance=50_000))
    assert res.engine == "unified" and res.engine_result is not None
    d = res.to_summary_dict()
    assert d["engine"] == "unified" and "portfolio_stats" in d and "floating_overlay" in d
    assert res.floating_overlay.n_trades == len(res.combined_trades)


def test_clock_alignment_sessions_and_errors():
    a = _df(100, 1, freq="1h")
    b = _df(100, 2, freq="1h", t0="2024-01-01 03:00")        # offset start, same rhythm
    c = build_portfolio_clock([a, b], ["A", "B"])
    assert len(c) == 103
    assert (c.bar_at[0][:3] >= 0).all() and (c.bar_at[1][:3] == -1).all()
    assert c.last_bar[1][2] == -1 and c.last_bar[1][3] == 0       # as-of is backward-looking only
    one = build_portfolio_clock([a], ["A"])
    assert len(one) == len(a)
    ov = build_portfolio_clock([a, b], ["A", "B"], window="overlap")
    assert len(ov) == 97
    dup = pd.concat([a, a.iloc[:1]]).sort_values("timestamp")
    with pytest.raises(PortfolioClockError):
        build_portfolio_clock([dup], ["A"])
    with pytest.raises(PortfolioEngineError):
        run_portfolio_engine([], PortfolioEngineConfig())


def test_floating_overlay_bounds_realised_drawdown():
    n = 900
    legs = [InstrumentLeg(f"L{i}", _df(n, i + 1, start=1.1 + i), _Fixed(_signals(n, i + 1)), RiskConfig()) for i in range(2)]
    res = run_portfolio_backtest(legs, PortfolioConfig(initial_balance=100_000))
    ov = res.floating_overlay
    assert ov is not None and ov.worst_case_max_dd >= ov.closed_max_dd >= 0
    assert ov.n_measured + ov.n_proxied == len(res.combined_trades)
    assert "floating_overlay" in res.to_summary_dict()
