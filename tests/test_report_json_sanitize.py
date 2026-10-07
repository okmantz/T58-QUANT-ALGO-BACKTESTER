"""Regression tests for the 2026-10-06 Full Pipeline report crash.

Owen ran the v9 champion through Full Pipeline on ES 1m data resampled to
5m. The pipeline correctly rejected the candidate at the PBO gate, but
Step 7/7 report generation crashed with "Object of type Timestamp is not
JSON serializable" -- a pandas Timestamp buried in a statistics/holdout
payload hit export_html's bare json.dumps() (the PDF-export hidden input),
which the timeframe sweep then surfaced as the misleading "produced no
usable timeframe".

Two layers of defense, both pinned here:
  1. build_report() sanitizes the report dict once via sanitize_for_json,
     so every consumer (JSON file, HTML, run history, web routes) gets
     JSON-safe values.
  2. export_html's hidden-input json.dumps now uses default=str, so even
     an unsanitized dict can never crash HTML export again.

Also pins Owen's explicit ask: every standard trading timeframe must
resample cleanly from 1-minute source data via build_timeframe_sweep --
the same path Full Pipeline, Search Lab, and Evolution Lab all use.
"""
import json

import pandas as pd
import pytest

from app.data.timeframe_sweep import DEFAULT_SWEEP_TIMEFRAMES, build_timeframe_sweep
from app.reports.generator import export_html, sanitize_for_json


def _payload_with_timestamp():
    return {
        "statistics": {
            "net_profit": 123.45,
            "first_trade_time": pd.Timestamp("2026-01-02 09:30:00"),
            "last_trade_time": pd.NaT,
            "nested": {"days": [pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-04")]},
        },
        "holdout": None,
    }


def test_sanitize_converts_timestamp_to_iso_string():
    out = sanitize_for_json(_payload_with_timestamp())
    assert out["statistics"]["first_trade_time"] == "2026-01-02T09:30:00"
    assert out["statistics"]["nested"]["days"] == ["2026-01-03T00:00:00", "2026-01-04T00:00:00"]


def test_sanitize_handles_nat_nan_inf_numpy_and_sets():
    np = pytest.importorskip("numpy")
    out = sanitize_for_json(
        {
            "nat": pd.NaT,
            "nan": float("nan"),
            "inf": float("inf"),
            "np_int": np.int64(7),
            "np_float": np.float64(1.5),
            "np_nan": np.float64("nan"),
            "np_bool": np.bool_(True),
            "arr": np.array([1, 2]),
            "td": pd.Timedelta("5 minutes"),
            "s": {3, 1, 2},
        }
    )
    assert out["nat"] is None
    assert out["nan"] is None
    assert out["inf"] is None
    assert out["np_int"] == 7 and isinstance(out["np_int"], int)
    assert out["np_float"] == 1.5
    assert out["np_nan"] is None
    assert out["np_bool"] is True
    assert out["arr"] == [1, 2]
    assert out["td"] == 300.0
    assert out["s"] == [1, 2, 3]


def test_sanitized_payload_dumps_with_bare_json_dumps():
    # The precise historical failure: json.dumps(report) with NO default=,
    # exactly as export_html's PDF-button line used to do.
    json.dumps(sanitize_for_json(_payload_with_timestamp()))


def _minimal_report():
    ts = pd.Timestamp("2026-01-02 09:30:00")
    return {
        "generated_at": "2026-10-06T00:00:00+00:00",
        "strategy": {
            "name": "S",
            "source_type": "manual",
            "instrument": "ES",
            "timeframe": "5m",
            "backtest_period_start": "2026-01-01",
            "backtest_period_end": "2026-02-01",
        },
        "verdict": "NOT READY",
        "verdict_reasons": ["rejected at PBO gate"],
        "historical_backtest": {"statistics": {"net_profit": 1.0}, "total_trades": 0},
        "prop_firm_rules": {"account_size": 50000},
        "prop_firm_single_run": {"passed_evaluation": False},
        "monte_carlo": {
            "n_simulations": 100,
            "evaluation_pass_probability": 0.1,
            "first_payout_probability": 0.05,
            "failure_before_payout_probability": 0.9,
            "median_days_to_first_payout": 20,
            "expected_payout": 0.0,
            "risk_of_ruin_pct": 90.0,
            "return_distribution": [1.0, 2.0],
            "return_percentiles": {5: 0.0, 50: 1.0, 95: 2.0},
            "drawdown_distribution": [1.0],
            "drawdown_percentiles": {50: 1.0},
        },
        # Timestamp buried where a real pipeline payload carries one --
        # this exact shape crashed export_html before the fix. Deliberately
        # NOT pre-sanitized: this pins defense layer 2 (default=str).
        "holdout_comparison": {"in_sample_statistics": {"first_bar": ts}},
    }


def test_export_html_survives_timestamp_in_report(tmp_path):
    out = export_html(_minimal_report(), tmp_path / "r.html")
    assert out.exists()
    assert out.stat().st_size > 0


def test_every_standard_timeframe_resamples_from_1m():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-01-04", periods=7200, freq="1min")
    opens = 7000 + np.cumsum(rng.standard_normal(7200) * 0.5)
    closes = opens + rng.standard_normal(7200) * 0.4
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": opens,
            "high": np.maximum(opens, closes) + abs(rng.standard_normal(7200)) * 0.5,
            "low": np.minimum(opens, closes) - abs(rng.standard_normal(7200)) * 0.5,
            "close": closes,
            "volume": rng.integers(1, 100, 7200).astype(float),
        }
    )
    plan = build_timeframe_sweep(df, list(DEFAULT_SWEEP_TIMEFRAMES))
    assert [t.label for t in plan.targets] == ["5m", "15m", "30m", "1h", "4h"]
    assert not plan.skipped
    for t in plan.targets:
        d = t.dataframe
        assert str(d["timestamp"].dtype).startswith("datetime")
        hi = d[["open", "close"]].max(axis=1)
        lo = d[["open", "close"]].min(axis=1)
        assert ((d["high"] >= hi) & (d["low"] <= lo)).all(), f"OHLC invariant broken on {t.label}"
    # Edge integrity: resampling invents no bars and drops no edge data.
    five = plan.targets[0].dataframe
    assert five["open"].iloc[0] == df["open"].iloc[0]
    assert five["close"].iloc[-1] == df["close"].iloc[-1]


def test_export_html_survives_infinite_profit_factor_in_cost_ladder(tmp_path):
    """All-winning trades give a cost-ladder rung with zero losses, i.e.
    profit_factor = inf; build_report's sanitizer maps inf to None, and
    the ladder's HTML formatter used to do f"{None:,.2f}" -> TypeError.
    The report crashed precisely for winning strategies."""
    report = _minimal_report()
    report["cost_ladder"] = [
        {"extra_cost_pct_per_trade": 0.0, "net_profit": 100.0, "profit_factor": float("inf"), "win_rate": 100.0},
        {"extra_cost_pct_per_trade": 0.05, "net_profit": 90.0, "profit_factor": None, "win_rate": 100.0},
        {"extra_cost_pct_per_trade": 0.1, "net_profit": 80.0, "profit_factor": 2.5, "win_rate": 90.0},
    ]
    out = export_html(report, tmp_path / "r2.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "∞" in text  # both the inf and the sanitized-None rung render as infinity
