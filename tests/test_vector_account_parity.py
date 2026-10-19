"""2026-10-15: the vectorized Monte Carlo must be indistinguishable from the per-path loop.

Differential test: random trade lists x random PropRules (every feature the kernel claims to cover)
x every resampling method, run with T58_MC_VECTORIZE on and off, compared field by field (exact
equality, not approx). Also pins the batched day-block draw to the old scalar draw stream.
"""
from __future__ import annotations

import dataclasses
import random

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.monte_carlo import engine as eng
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules
from app.prop.vector_account import supports_vectorized


def _trades(rng: random.Random, n_days: int = 80, with_excursions: bool = True):
    out = []
    day = pd.Timestamp("2024-01-02 14:00")
    for d in range(n_days):
        if rng.random() < 0.15:          # a gap of a few calendar days now and then
            day += pd.Timedelta(days=rng.randint(2, 6))
        else:
            day += pd.Timedelta(days=1)
        for k in range(rng.randint(1, 5)):
            t = day + pd.Timedelta(minutes=30 * k)
            pnl = rng.gauss(60, 350)
            if rng.random() < 0.05:
                pnl = -rng.uniform(800, 2200)
            size = 1.0
            worst = 100.0 + min(pnl, 0.0) - rng.uniform(0, 200) if with_excursions else None
            best = 100.0 + max(pnl, 0.0) + rng.uniform(0, 200) if with_excursions else None
            out.append(Trade(
                entry_time=t, exit_time=t + pd.Timedelta(minutes=20), direction=1, entry_price=100.0,
                exit_price=100.0 + pnl, size=size, pnl=pnl, pnl_pct=0.0, exit_reason="signal",
                commission=0.0, equity_after=0.0, initial_risk=rng.uniform(100, 400),
                intended_risk_dollars=rng.uniform(100, 400), worst_price=worst, mfe_price=best,
            ))
    return out


def _random_rules(rng: random.Random, easy: bool = False) -> PropRules:
    """easy=True widens the loss limits and lowers the target so paths actually pass, get funded and get
    paid -- otherwise most random rule sets bust immediately and the payout logic goes unexercised."""
    kw = dict(
        account_size=50_000.0,
        evaluation_profit_target_pct=rng.choice([3.0, 6.0, 9.0]),
        daily_loss_limit_pct=rng.choice([1.5, 3.0, 5.0, 100.0]),
        max_drawdown_pct=rng.choice([3.0, 6.0, 10.0]),
        drawdown_type=rng.choice(["trailing", "static"]),
        drawdown_check_mode=rng.choice(["intrabar", "eod"]),
        consistency_rule_pct=rng.choice([None, 30.0, 50.0]),
        min_trading_days=rng.choice([1, 5, 10]),
        payout_threshold_pct=rng.choice([0.0, 2.0]),
        payout_cap_pct=rng.choice([None, 50.0]),
        payout_frequency_days=rng.choice([1, 5, 14]),
        required_buffer_pct=rng.choice([0.0, 1.0]),
        winning_days_for_payout=rng.choice([0, 3, 5]),
        min_winning_day_profit=rng.choice([0.0, 100.0]),
        max_inactive_days=rng.choice([None, 3, 10]),
        funded_consistency_rule_pct=rng.choice([None, 40.0]),
        daily_loss_action=rng.choice(["fail", "lock_day"]),
        daily_loss_base=rng.choice(["initial", "prior_day_high", "ratchet_up"]),
        max_eval_calendar_days=rng.choice([None, 45, 90]),
        payout_cap_dollars=rng.choice([None, 1500.0]),
        max_payouts=rng.choice([None, 2]),
        dd_basis=rng.choice(["legacy", "realized", "eod", "floating"]),
        daily_loss_basis=rng.choice(["realized", "floating"]),
        trailing_lock=rng.choice([False, True]),
        trailing_lock_offset_pct=rng.choice([0.0, 0.5]),
        trailing_distance_basis=rng.choice(["peak", "account"]),
        funded_starts_fresh=rng.choice([False, True]),
        floating_drawdown_mode=rng.choice(["realized", "adverse"]),
    )
    if easy:
        kw.update(evaluation_profit_target_pct=rng.choice([2.0, 3.0]), daily_loss_limit_pct=rng.choice([5.0, 100.0]),
                  max_drawdown_pct=rng.choice([10.0, 20.0]), min_trading_days=rng.choice([1, 3]),
                  winning_days_for_payout=rng.choice([0, 2]), payout_frequency_days=rng.choice([1, 3]),
                  max_inactive_days=rng.choice([None, 10]), max_eval_calendar_days=rng.choice([None, 120]))
    return PropRules(**kw)


def _run(trades, rules, cfg, vectorize: bool, monkeypatch):
    monkeypatch.setenv("T58_MC_VECTORIZE", "1" if vectorize else "0")
    return run_monte_carlo(trades, rules, cfg)


def _same(a, b, path=""):
    if dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            _same(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _same(a[k], b[k], f"{path}[{k}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, float):
        assert (a == b) or (a != a and b != b), f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


_OUTCOMES: list = []


@pytest.mark.parametrize("seed", list(range(36)))
def test_vectorized_monte_carlo_is_bit_identical_to_the_per_path_loop(seed, monkeypatch):
    rng = random.Random(seed)
    trades = _trades(rng, n_days=rng.choice([40, 80, 120]))
    rules = _random_rules(rng, easy=seed % 2 == 0)
    method = ["day_block_bootstrap", "day_block_bootstrap", "bootstrap", "shuffle", "block_bootstrap",
              "regime_shuffle"][seed % 6]
    cfg = MonteCarloConfig(n_simulations=96, method=method, random_seed=seed + 1,
                           slippage_stress_pct=rng.choice([0.0, 10.0]))
    assert supports_vectorized(rules, False, have_excursions=True)
    used = []
    real = eng._mc_rows_vectorized

    def spy(**kw):
        out = real(**kw)          # an exception here would silently fall back and make the test vacuous
        used.append(1)
        return out

    legacy = _run(trades, rules, cfg, False, monkeypatch)
    monkeypatch.setattr(eng, "_mc_rows_vectorized", spy)
    vec = _run(trades, rules, cfg, True, monkeypatch)
    assert used, "the vectorized kernel did not run (it fell back to the loop)"
    _same(legacy, vec)
    _OUTCOMES.append((legacy.evaluation_pass_probability, legacy.first_payout_probability,
                      legacy.risk_of_ruin_pct))


def test_random_rule_sets_exercise_pass_payout_and_failure():
    """Guard against a vacuous differential test: across the seeds above, some paths must pass, some must
    be paid, and some must bust (run after the parametrized cases, in file order)."""
    assert _OUTCOMES, "parametrized parity cases did not run first"
    assert any(p > 0 for p, _, _ in _OUTCOMES)
    assert any(f > 0 for _, f, _ in _OUTCOMES)
    assert any(r > 0 for _, _, r in _OUTCOMES)


def test_vectorized_path_is_actually_used(monkeypatch):
    calls = []
    real = eng._mc_rows_vectorized

    def spy(**kw):
        calls.append(1)
        return real(**kw)

    monkeypatch.setattr(eng, "_mc_rows_vectorized", spy)
    monkeypatch.setenv("T58_MC_VECTORIZE", "1")
    trades = _trades(random.Random(3))
    run_monte_carlo(trades, PropRules(account_size=50_000.0, dd_basis="eod"),
                    MonteCarloConfig(n_simulations=64, random_seed=1))
    assert calls
    calls.clear()
    monkeypatch.setenv("T58_MC_VECTORIZE", "0")
    run_monte_carlo(trades, PropRules(account_size=50_000.0, dd_basis="eod"),
                    MonteCarloConfig(n_simulations=64, random_seed=1))
    assert not calls


def test_reset_on_breach_falls_back_to_the_loop(monkeypatch):
    # S1 (2026-10-19): reset_on_breach is now supported by the vectorized kernel
    assert supports_vectorized(PropRules(), True, have_excursions=True)
    assert not supports_vectorized(PropRules(dd_basis="floating"), False, have_excursions=False)


def test_batched_day_block_draws_match_the_old_scalar_stream():
    def old(rng, groups, k):
        nd = len(groups)
        k = max(1, min(int(k), nd))
        chosen = []
        while len(chosen) < nd:
            s = int(rng.integers(0, nd - k + 1))
            chosen.extend(range(s, s + k))
        return np.array(chosen[:nd])

    for nd in (1, 2, 7, 50, 333):
        for k in (1, 2, 3, 5, 20, 400):
            groups = [np.arange(2)] * nd
            a, b = np.random.default_rng(9), np.random.default_rng(9)
            batch = eng._draw_day_block_choices(b, nd, k, 6)
            for p in range(6):
                assert (old(a, groups, k) == batch[p]).all()
            assert a.integers(0, 10 ** 9) == b.integers(0, 10 ** 9)   # streams stay aligned afterwards


def test_resample_day_blocks_keeps_its_public_contract():
    groups = [np.array([0, 1]), np.array([2]), np.array([3, 4, 5]), np.array([6])]
    idx, ds = eng._resample_day_blocks(np.random.default_rng(1), groups, [pd.Timestamp("2024-01-0%d" % (i + 1)) for i in range(4)], 2)
    assert len(idx) == len(ds.day_index_per_trade) == len(ds.is_last_of_day)
    assert ds.n_days == 4 and ds.day_index_per_trade == sorted(ds.day_index_per_trade)
    assert sum(ds.is_last_of_day) == 4
