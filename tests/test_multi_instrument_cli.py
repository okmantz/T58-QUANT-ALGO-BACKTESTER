"""Tests for app.main.run_multi_instrument_cli -- the CLI glue on top of
app.orchestration.multi_instrument_search (already covered end to end by
tests/test_multi_instrument_search.py). Kept small-scale (2 tiny CSVs,
workers=1, low candidate/sim counts) so this runs quickly while still
exercising the real argument-parsing -> dispatch -> report-writing path,
not a mock of it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.main import run_multi_instrument_cli


def _trending_csv(path, n=1200, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.00003)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00002))
        l = min(o, c) - abs(rng.normal(0, 0.00002))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.to_csv(path, index=False)
    return path


_FAST_KWARGS = dict(
    family="trend_breakout", max_candidates=6, workers=1,
    min_trades=3, min_profit_factor=0.0,
    ga_population=4, ga_generations=1, full_mc_sims=30,
    walk_forward_folds=0, robustness_neighbors=0,
)


def test_multi_instrument_cli_requires_at_least_two_jobs(tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_multi_instrument_cli(["XAUUSD:15m:x.csv"], str(tmp_path))
    assert "at least 2" in capsys.readouterr().out


def test_multi_instrument_cli_rejects_a_malformed_job_string(tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_multi_instrument_cli(["not-enough-colons"], str(tmp_path))


def test_multi_instrument_cli_end_to_end(tmp_path, capsys):
    csv_a = _trending_csv(tmp_path / "a.csv", seed=1)
    csv_b = _trending_csv(tmp_path / "b.csv", seed=2)
    run_multi_instrument_cli(
        [f"EURUSD:5m:{csv_a}", f"GBPUSD:5m:{csv_b}"],
        str(tmp_path / "out"), max_concurrent=2, **_FAST_KWARGS,
    )
    out = capsys.readouterr().out
    assert "MULTI-INSTRUMENT RESULTS" in out
    assert "EURUSD/5m" in out and "GBPUSD/5m" in out
    # Either a real champion was found and reported, or the run honestly
    # says nothing passed -- either way it must not silently do nothing.
    assert "Best result:" in out or "No instrument/timeframe produced a Stage 3 champion" in out


def test_multi_instrument_cli_unknown_family_exits_cleanly(tmp_path, capsys):
    csv_a = _trending_csv(tmp_path / "a.csv", seed=1)
    csv_b = _trending_csv(tmp_path / "b.csv", seed=2)
    with pytest.raises(SystemExit):
        run_multi_instrument_cli(
            [f"EURUSD:5m:{csv_a}", f"GBPUSD:5m:{csv_b}"],
            str(tmp_path / "out"), family="not_a_real_family",
        )
    assert "Unknown family" in capsys.readouterr().out
