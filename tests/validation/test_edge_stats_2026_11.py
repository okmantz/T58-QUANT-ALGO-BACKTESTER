"""2026-11-05 audit: measured per-trade SD, stationary-bootstrap sensitivity, market-pool dedupe, combined false-positive estimate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.validation import edge_stats as ES


def test_blended_sd_uses_prior_for_tiny_samples_and_measured_for_large():
    prior = ES.DEFAULT_TRADE_SD_R
    assert ES.blended_trade_sd_r(0.5, 0) == prior and ES.blended_trade_sd_r(None, 100) == prior
    assert abs(ES.blended_trade_sd_r(0.8, 200) - 0.8) < 1e-9
    mid = ES.blended_trade_sd_r(0.5, 15)                       # halfway: shrunk toward the prior
    assert 0.5 < mid < prior
    assert ES.blended_trade_sd_r(100.0, 500) <= ES.MEASURED_SD_CEIL_R
    assert ES.blended_trade_sd_r(0.0, 500) >= ES.MEASURED_SD_FLOOR_R
    assert "own" in ES.sd_basis_label(120) and ES.sd_basis_label(5) != ES.sd_basis_label(120)


def test_stationary_bootstrap_is_centred_on_the_sample_mean_and_reproducible():
    rng = np.random.default_rng(0)
    cnt = np.full(400, 2.0)
    sums = rng.normal(0.2, 1.0, 400) * cnt
    a = ES.stationary_bootstrap_means(sums, cnt, 4.0, n_boot=800, seed=5)
    b = ES.stationary_bootstrap_means(sums, cnt, 4.0, n_boot=800, seed=5)
    assert np.array_equal(a, b)
    assert abs(a.mean() - sums.sum() / cnt.sum()) < 0.05


def test_block_sensitivity_flags_a_zero_edge_and_not_a_strong_one():
    rng = np.random.default_rng(1)
    days = [d for d in pd.date_range("2024-01-01", periods=300, freq="B") for _ in range(2)]
    strong = list(rng.normal(0.5, 1.0, len(days)))
    res = ES.block_sensitivity(strong, days)
    assert res is not None and min(res["p_positive"].values()) > 0.95 and not res["sensitive"]
    null = list(rng.normal(0.0, 1.0, len(days)))
    res0 = ES.block_sensitivity(null, days)
    assert res0 is not None and res0["p_positive"]["1d"] < 0.95


def test_market_pool_dedupes_micro_twins_and_reads_bar_sizes():
    from app.data import market_pool as MP
    assert MP.underlying_symbol("MES") == "ES" and MP.underlying_symbol("MNQ") == "NQ" and MP.underlying_symbol("EURUSD") == "EURUSD"
    assert MP.bar_minutes_from_name("NQ_5min_2026.csv") == 5
    assert MP.bar_minutes_from_name("EURUSD240.csv") in (240, None)
    rng = np.random.default_rng(2)
    days = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(300)]
    r = list(rng.normal(0, 1, 300))
    kept, dropped = MP.dedupe_r_series({"ES": (r, days), "MES": (list(r), days), "GC": (list(rng.normal(0, 1, 300)), days)})
    assert len(kept) == 2 and len(dropped) == 1


def test_combined_false_positive_without_a_report_is_honest(tmp_path):
    from app.discovery_engine import surrogate
    out = surrogate.combined_false_positive(tmp_path / "missing.json")
    assert out["available"] is False and out["text"]
