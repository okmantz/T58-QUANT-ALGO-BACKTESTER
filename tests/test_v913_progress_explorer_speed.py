"""
v9.13 — completion percentage, baseline trade explorer, speed.

Four workstreams, one bundle. What this suite pins:

  WS-1  the weighted pipeline percent is real, monotonic, and reaches
        100 at done (status payload + PipelineProgress unit contract);
  WS-2  the baseline explorer endpoint serves Step 1's trades while
        the job is still running, heatmap aggregation matches the
        trade list exactly, and zero trades renders gracefully;
  WS-3  the speed work (vectorized trading days, incremental
        winning-days, parallel Monte Carlo / random-entry null /
        attempt replay) produces BIT-IDENTICAL numbers to the serial
        paths it replaced -- same seeds, same draws, same order;
  WS-4  the job page stays coherent (stop still fast, verdict path
        untouched -- covered by the v9.11 suite staying green).

Nothing here (or in the feature) touches a verdict threshold: the 70%
per-attempt READY bar, DSR/Bonferroni/PBO rules, seeds, and draw
counts are all exactly as before.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------------
# WS-3a: vectorized trading days == the scalar definition, elementwise
# --------------------------------------------------------------------------

def test_trading_days_matches_scalar_including_dst():
    from app.data.trading_day import trading_day, trading_days

    idx = pd.DatetimeIndex([
        # US DST spring forward 2026-03-08 02:00 local; fall back 2026-11-01.
        "2026-03-06 16:59:00", "2026-03-06 17:00:00", "2026-03-08 01:59:00",
        "2026-03-08 03:30:00", "2026-03-08 17:30:00", "2026-10-30 18:15:00",
        "2026-11-01 01:30:00", "2026-11-01 17:00:00", "2026-01-05 00:00:00",
        "2026-01-05 23:59:00",
    ])
    got = trading_days(idx, tz="America/Chicago", roll_hour=17)
    expected = [trading_day(ts, tz="America/Chicago", roll_hour=17) for ts in idx]
    assert list(got) == expected
    # tz-aware input takes the same path (no naive-only shortcut taken).
    # UTC localization: localizing these wall times to a US zone would
    # hit the fall-back ambiguity (2026-11-01 01:30 exists twice), which
    # neither implementation is meant to invent a ruling for.
    aware = idx.tz_localize("UTC")
    got2 = trading_days(aware, tz="America/Chicago", roll_hour=17)
    expected2 = [trading_day(ts, tz="America/Chicago", roll_hour=17) for ts in aware]
    assert list(got2) == expected2


# --------------------------------------------------------------------------
# WS-3b: _max_losing_streak vectorization == the original loop
# --------------------------------------------------------------------------

def _streak_reference(pnls: np.ndarray) -> int:
    cur = best = 0
    for v in pnls:
        if np.isnan(v):
            cur = 0
        elif v <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def test_max_losing_streak_matches_reference():
    from app.monte_carlo.engine import _max_losing_streak

    rng = np.random.default_rng(123)
    for _ in range(40):
        a = rng.normal(0, 100, size=int(rng.integers(0, 60)))
        a[rng.random(len(a)) < 0.1] = np.nan
        a[rng.random(len(a)) < 0.1] = 0.0
        assert _max_losing_streak(a) == _streak_reference(a)
    assert _max_losing_streak(a := np.array([])) == 0  # noqa: F841


# --------------------------------------------------------------------------
# WS-3c: parallel engines == serial engines (bit-identical)
# --------------------------------------------------------------------------

def _engine_trade(ts, direction: int, pnl: float):
    from app.backtest.execution import Trade

    return Trade(
        entry_time=ts, exit_time=ts + pd.Timedelta(hours=2),
        direction=direction, entry_price=100.0,
        exit_price=100.0 + direction * pnl / 10.0,
        size=10.0, pnl=pnl, pnl_pct=pnl / 1000.0,
        exit_reason="signal", commission=2.0, equity_after=50_000.0 + pnl,
    )


def _mc_fixture():
    from app.monte_carlo.engine import MonteCarloConfig
    from app.prop.simulator import PropRules

    ts = pd.date_range("2026-01-05", periods=400, freq="1D")
    trades = [
        _engine_trade(ts[i], 1 if i % 3 else -1, float(((i * 37) % 11) - 5) * 40.0)
        for i in range(400)
    ]
    rules = PropRules(account_size=50_000, evaluation_profit_target_pct=8,
                      daily_loss_limit_pct=5, max_drawdown_pct=10)
    # n_simulations must clear the engine's parallel floor (200) so the
    # parallel branch -- not the serial fallback -- is what gets compared.
    cfg = MonteCarloConfig(n_simulations=240, random_seed=7, reset_on_breach=True)
    return trades, rules, cfg


def test_monte_carlo_parallel_equals_serial():
    from dataclasses import asdict

    from app.monte_carlo.engine import run_monte_carlo

    trades, rules, cfg = _mc_fixture()
    serial = run_monte_carlo(trades, rules, cfg)
    parallel = run_monte_carlo(trades, rules, cfg, max_workers=2)
    assert asdict(parallel) == asdict(serial)

    # Progress callback fires and ends exactly at (n_simulations, n_simulations).
    seen: list[tuple[int, int]] = []
    run_monte_carlo(trades, rules, cfg, max_workers=2,
                    progress_cb=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1] == (cfg.n_simulations, cfg.n_simulations)
    assert all(d <= t for d, t in seen)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def _small_frame(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """Same shape as the v9.11 null test: small synthetic frame where
    full engine passes cost milliseconds, so serial-vs-parallel can be
    compared bit-for-bit inside a unit test."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.3,
        "low": close - 0.3, "close": close, "volume": 100.0,
    })


def test_random_entry_null_parallel_equals_serial_and_reports_progress():
    from app.backtest.risk import RiskConfig
    from app.research.director import random_entry_distribution

    df = _small_frame()
    risk = RiskConfig(initial_balance=50_000, pip_size=0.01, contract_size=1000.0)
    seen_serial: list[tuple[int, int]] = []
    serial = random_entry_distribution(
        df, risk, observed_value=0.0, n_entries=10,
        stop_loss_pips=50, take_profit_pips=100, n_seeds=32, seed=0,
        progress_cb=lambda d, t: seen_serial.append((d, t)),
    )
    seen_par: list[tuple[int, int]] = []
    parallel = random_entry_distribution(
        df, risk, observed_value=0.0, n_entries=10,
        stop_loss_pips=50, take_profit_pips=100, n_seeds=32, seed=0,
        max_workers=2, progress_cb=lambda d, t: seen_par.append((d, t)),
    )
    assert parallel == serial
    # Both paths report at identical 25-seed milestones (the v9.11
    # advisory-callback contract, unchanged by parallelization).
    assert seen_serial == [(25, 32)] and seen_par == seen_serial


def test_attempt_replay_parallel_equals_serial():
    from app.backtest.risk import RiskConfig
    from app.prop.attempt_replay import run_attempt_replay
    from app.prop.simulator import PropRules

    df = _small_frame(600, seed=5)
    signals = pd.Series(0, index=df.index, dtype=int)
    signals.iloc[::37] = 1
    signals.iloc[53::71] = -1
    risk = RiskConfig(initial_balance=50_000, pip_size=0.01, contract_size=1000.0)
    rules = PropRules(account_size=50_000, evaluation_profit_target_pct=8,
                      daily_loss_limit_pct=5, max_drawdown_pct=10)
    serial = run_attempt_replay(df, signals, risk, rules, 50, 100, n_starts=6)
    parallel = run_attempt_replay(df, signals, risk, rules, 50, 100,
                                  n_starts=6, max_workers=2)
    assert parallel.to_dict() == serial.to_dict()


# --------------------------------------------------------------------------
# WS-1: weighted percent + tracker monotonicity
# --------------------------------------------------------------------------

def test_pipeline_percent_weights_and_bounds():
    from app.orchestration.run_progress import (
        PIPELINE_STEP_WEIGHTS, pipeline_percent,
    )

    assert math.isclose(sum(PIPELINE_STEP_WEIGHTS), 1.0)
    assert pipeline_percent(1, 0.0) == 0.0
    assert pipeline_percent(7, 1.0) == 100.0
    assert 0.0 < pipeline_percent(2, 0.0) < pipeline_percent(2, 1.0) < 100.0
    # Out-of-range input clamps instead of exploding on a live job page.
    assert pipeline_percent(99, 5.0) == 100.0


def test_pipeline_progress_monotonic_and_finishes_at_100():
    from app.orchestration.run_progress import PipelineProgress

    tracker = PipelineProgress()
    seen = [tracker.update(1, 7, 0.0)["percent"]]
    for step, frac in [(1, 0.5), (1, 1.0), (2, 0.25), (2, 0.9), (3, 0.1),
                       (5, 0.0), (6, 0.5), (7, 0.4), (7, 0.95)]:
        seen.append(tracker.update(step, 7, frac)["percent"])
    assert seen == sorted(seen), f"percent must never go backwards: {seen}"
    assert tracker.finish()["percent"] == 100.0
    # Multi-run (timeframe sweep): rolling into run 2 must not reset to 0.
    sweep = PipelineProgress(run_count=2)
    sweep.update(1, 7, 0.0)
    sweep.update(7, 7, 1.0)
    mid = sweep.update(1, 7, 0.0)  # run 2 starts
    assert mid["percent"] >= 50.0
    assert sweep.finish()["percent"] == 100.0


# --------------------------------------------------------------------------
# WS-2: explorer aggregation math + empty case (pure functions)
# --------------------------------------------------------------------------

def _trade(day: str, hour: int, pnl: float, direction: int = 1) -> dict:
    entry = f"{day}T{hour:02d}:15:00"
    return {"entry_time": entry, "exit_time": f"{day}T{hour:02d}:45:00",
            "direction": direction, "pnl": pnl, "duration_minutes": 30.0}


def test_baseline_explorer_aggregation_math():
    from app.orchestration.baseline_explorer import summarize_baseline_trades

    # 2026-01-05 is a Monday; 2026-01-06 Tuesday; 2026-01-07 Wednesday.
    trades = [
        _trade("2026-01-05", 3, -100.0),
        _trade("2026-01-05", 3, -50.0, direction=-1),
        _trade("2026-01-06", 10, 300.0),
        _trade("2026-01-07", 3, -25.0),
        _trade("2026-01-07", 10, -75.0),
    ]
    s = summarize_baseline_trades(trades)
    assert s["has_trades"] is True and s["n_trades"] == 5
    assert s["net_pnl"] == 50.0 and s["gross_loss"] == 250.0
    assert s["matrix"][0][3] == -150.0          # Monday 03:00
    assert s["matrix"][1][10] == 300.0          # Tuesday 10:00
    assert s["matrix"][2][3] == -25.0           # Wednesday 03:00
    assert s["hour_profile"][3] == -175.0
    assert s["weekday_profile"][0] == -150.0
    worst = s["worst_hours"][0]
    assert worst["hour"] == 3 and worst["share_of_gross_losses_pct"] == 70.0
    # The two Monday 03:00 losers are adjacent chronologically -> one streak.
    assert any(st["trades"] == 2 for st in s["longest_losing_streaks"])
    assert s["suggestions"] and "03:00" in s["suggestions"][0]


def test_baseline_explorer_win_clusters_mirror_losses():
    """v9.14 WS-1: the win side aggregates the SAME trades with the same
    arithmetic as the loss side -- bucket sums must match the trade
    list exactly, and breakeven trades break winning streaks exactly
    like they break losing ones."""
    from app.orchestration.baseline_explorer import summarize_baseline_trades

    # 2026-01-05 Mon, 2026-01-06 Tue, 2026-01-07 Wed, 2026-01-08 Thu.
    trades = [
        _trade("2026-01-05", 14, 200.0),
        _trade("2026-01-05", 14, 150.0, direction=-1),
        _trade("2026-01-06", 9, -80.0),
        _trade("2026-01-06", 14, 100.0),
        _trade("2026-01-07", 10, 0.0),          # breakeven breaks the win streak
        _trade("2026-01-07", 14, 50.0),
        _trade("2026-01-08", 9, 25.0),
    ]
    s = summarize_baseline_trades(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    assert s["gross_profit"] == round(sum(t["pnl"] for t in wins), 2) == 525.0
    # Best hour = 14:00 with 200+150+100+50 = 500 of 525 gross profit.
    best = s["best_hours"][0]
    assert best["hour"] == 14 and best["pnl"] == 500.0
    assert best["share_of_gross_profits_pct"] == round(100.0 * 500.0 / 525.0, 1)
    # Weekday buckets are NET (same convention as worst_weekdays): Mon
    # 350, Tue -80+100=20, Wed 50, Thu 25 -- all positive, all listed.
    by_label = {w["label"]: w["pnl"] for w in s["best_weekdays"]}
    assert by_label == {"Mon": 350.0, "Tue": 20.0, "Wed": 50.0, "Thu": 25.0}
    # Winning streaks: [Mon14, Mon14] then the Tue 09:00 loser and the
    # Wed 10:00 breakeven break the rest into single-trade runs.
    assert s["longest_winning_streaks"][0]["trades"] == 2
    assert s["longest_winning_streaks"][0]["total_pnl"] == 350.0
    assert s["biggest_win_clusters"][0]["total_pnl"] == 350.0
    # The win-window suggestion cites the best hour with real numbers.
    assert any("14:00" in txt and "gross profit" in txt for txt in s["suggestions"])


def test_baseline_explorer_all_winners_has_no_loss_side():
    """All-win baseline: loss panel stays empty/graceful and the win
    panel still aggregates (not a division-by-zero anywhere)."""
    from app.orchestration.baseline_explorer import summarize_baseline_trades

    s = summarize_baseline_trades([_trade("2026-01-05", 10, 10.0), _trade("2026-01-05", 11, 20.0)])
    assert s["worst_hours"] == [] and s["gross_loss"] == 0.0
    assert s["best_hours"][0]["pnl"] == 20.0 and s["gross_profit"] == 30.0


def test_baseline_explorer_empty_is_graceful():
    from app.orchestration.baseline_explorer import summarize_baseline_trades

    s = summarize_baseline_trades([])
    assert s["has_trades"] is False
    assert s["n_trades"] == 0 and s["net_pnl"] == 0.0
    assert s["matrix"] == [[0.0] * 24 for _ in range(7)]
    assert s["worst_hours"] == [] and s["suggestions"] == []


# --------------------------------------------------------------------------
# WS-1 + WS-2 over HTTP: baseline served mid-run; progress monotonic;
# 100 at done.
# --------------------------------------------------------------------------

@pytest.mark.timeout(600)
def test_baseline_endpoint_serves_trades_mid_run():
    """Wide GA on a small frame: Step 1's baseline finishes in seconds
    while the search keeps the job running -- exactly the window the
    explorer and the percent exist for. Uses the v9.11 harness so the
    form plumbing matches the real route byte for byte."""
    import time as _t

    from app.web.server import JOB_MANAGER, app
    from tests.test_v911_stop_hang_layout import _start, _status

    client = app.test_client()
    job_id = _start(client, ga_population=40, ga_generations=40)
    try:
        # Poll the baseline endpoint until Step 1 publishes (should be
        # quick) while asserting the JOB ITSELF is still running --
        # that mid-run window is the whole point of the explorer.
        payload = None
        for _ in range(120):
            resp = client.get(f"/full-pipeline/job/{job_id}/baseline.json")
            assert resp.status_code == 200
            body = resp.get_json()
            if body.get("ready"):
                payload = body
                break
            _t.sleep(0.5)
        assert payload is not None, "baseline never became ready"
        status = _status(client, job_id)
        assert status["done"] is False, "job finished before baseline arrived -- test no longer proves mid-run serving"
        # Trades are the real baseline's: numbers consistent internally.
        trades = payload["trades"]
        assert isinstance(trades, list) and trades, "baseline should trade on this fixture"
        summary = payload["summary"]
        assert summary["n_trades"] == len(trades)
        assert summary["net_pnl"] == pytest.approx(sum(t["pnl"] for t in trades), abs=0.02)
        bucket_sum = sum(sum(row) for row in summary["matrix"])
        assert bucket_sum == pytest.approx(summary["net_pnl"], abs=0.05)
        # Progress payload present and sane while running.
        prog = status.get("progress")
        assert prog is not None and 0.0 <= prog["percent"] < 100.0
    finally:
        client.post(f"/full-pipeline/job/{job_id}/stop")
        for _ in range(120):
            if JOB_MANAGER.get(job_id)["done"]:
                break
            _t.sleep(0.25)


@pytest.mark.timeout(600)
def test_progress_reaches_100_when_done():
    """A fast end-to-end job (the v9.11 reduced shape): progress never
    goes backwards across the whole run and lands exactly on 100 once
    the job completes, with the verdict result stored as before."""
    import time as _t

    from app.web.server import JOB_MANAGER, app
    from tests.test_v911_stop_hang_layout import _start, _status

    client = app.test_client()
    job_id = _start(client)
    percents: list[float] = []
    status = None
    for _ in range(600):
        status = _status(client, job_id)
        prog = status.get("progress")
        assert prog is not None, "status payload lost its progress field"
        percents.append(prog["percent"])
        if status["done"]:
            break
        _t.sleep(0.5)
    assert status is not None and status["done"] is True
    assert status["error"] is None
    assert percents and percents == sorted(percents), f"progress went backwards: {percents}"
    assert percents[-1] == 100.0
    assert JOB_MANAGER.get(job_id)["result"] is not None  # verdict path intact
