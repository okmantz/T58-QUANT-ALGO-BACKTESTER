from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from app.quant_lab import factor_model
from app.quant_lab.factor_model import (
    FactorModelError,
    compute_factor_exposures,
    compute_returns_from_prices,
    fetch_fama_french_factors,
)

_SAMPLE_DAILY_CSV = """This file was created by CMPT_ME_BEME_RETS using the 202401 CRSP database.
The Fama/French factors are constructed using six value-weight portfolios formed on size and book-to-market.

,Mkt-RF,SMB,HML,RF
20220103,   0.85,  -0.24,   1.10,   0.000
20220104,  -0.10,   0.30,   0.90,   0.000
20220105,  -1.20,  -0.50,   0.40,   0.000
20220106,   0.30,   0.10,  -0.20,   0.000
20220107,   0.55,  -0.15,   0.25,   0.000
20220110,  -0.40,   0.05,   0.10,   0.000
20220111,   0.90,  -0.30,  -0.10,   0.000
20220112,   0.15,   0.20,   0.05,   0.000
20220113,  -0.75,   0.40,   0.30,   0.000
20220114,   0.20,  -0.05,  -0.15,   0.000
20220118,   0.60,   0.10,   0.20,   0.000
20220119,  -0.30,  -0.20,   0.10,   0.000

Copyright 2024 Kenneth R. French
"""


def _make_fake_zip(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_Research_Data_Factors_daily.CSV", csv_text)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def _synthetic_factors(n=500, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    mkt_rf = rng.normal(0.0003, 0.01, n)
    smb = rng.normal(0.0001, 0.005, n)
    hml = rng.normal(0.0001, 0.005, n)
    rf = np.full(n, 0.0001)
    factors = pd.DataFrame({"mkt_rf": mkt_rf, "smb": smb, "hml": hml, "rf": rf}, index=dates)
    return factors, rng


def test_fetch_fama_french_factors_parses_real_shaped_csv(monkeypatch):
    def fake_get(url, timeout=30.0, headers=None):
        return _FakeResponse(_make_fake_zip(_SAMPLE_DAILY_CSV))

    monkeypatch.setattr(factor_model.requests, "get", fake_get)
    df = fetch_fama_french_factors("daily")
    assert len(df) == 12
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rf"]
    # Values are converted from percent to decimal.
    assert df.iloc[0]["mkt_rf"] == pytest.approx(0.0085)


def test_fetch_fama_french_factors_invalid_frequency():
    with pytest.raises(FactorModelError):
        fetch_fama_french_factors("weekly")


def test_fetch_fama_french_factors_raises_on_unparseable_content(monkeypatch):
    def fake_get(url, timeout=30.0, headers=None):
        return _FakeResponse(_make_fake_zip("just some header text\nno data rows here\n"))

    monkeypatch.setattr(factor_model.requests, "get", fake_get)
    with pytest.raises(FactorModelError):
        fetch_fama_french_factors("daily")


def test_regression_recovers_known_zero_alpha_and_betas():
    factors, rng = _synthetic_factors(seed=1)
    true_beta_mkt, true_beta_smb, true_beta_hml = 1.1, 0.3, -0.2
    noise = rng.normal(0, 0.003, len(factors))
    stock_excess = (0.0 + true_beta_mkt * factors["mkt_rf"] + true_beta_smb * factors["smb"]
                    + true_beta_hml * factors["hml"] + noise)
    stock_returns = stock_excess + factors["rf"]

    result = compute_factor_exposures(stock_returns, factors)
    assert result.beta_mkt == pytest.approx(true_beta_mkt, abs=0.15)
    assert result.beta_smb == pytest.approx(true_beta_smb, abs=0.3)
    assert result.beta_hml == pytest.approx(true_beta_hml, abs=0.3)
    assert not result.alpha_significant
    assert result.r_squared > 0.5
    assert "Alpha is NOT statistically significant" in result.render_summary()


def test_regression_detects_a_real_injected_alpha():
    factors, rng = _synthetic_factors(seed=1)
    true_beta_mkt, true_beta_smb, true_beta_hml = 1.1, 0.3, -0.2
    noise = rng.normal(0, 0.003, len(factors))
    skilled_alpha = 0.0009  # a large, consistent per-period edge
    stock_excess = (skilled_alpha + true_beta_mkt * factors["mkt_rf"] + true_beta_smb * factors["smb"]
                    + true_beta_hml * factors["hml"] + noise)
    stock_returns = stock_excess + factors["rf"]

    result = compute_factor_exposures(stock_returns, factors)
    assert result.alpha_significant
    assert result.alpha > 0
    assert "Alpha IS statistically significant" in result.render_summary()


def test_too_little_overlap_raises():
    factors, rng = _synthetic_factors(n=10, seed=2)
    stock_returns = pd.Series(rng.normal(0, 0.01, 10), index=factors.index)
    with pytest.raises(FactorModelError):
        compute_factor_exposures(stock_returns, factors)


def test_compute_returns_from_prices():
    ts = pd.date_range("2024-01-01", periods=10, freq="1D")
    price = pd.Series([100, 101, 102, 101, 103, 104, 105, 104, 106, 107], dtype=float)
    df = pd.DataFrame({"timestamp": ts, "open": price, "high": price, "low": price, "close": price, "volume": 1.0})
    returns = compute_returns_from_prices(df)
    assert len(returns) == 9
    assert returns.iloc[0] == pytest.approx(0.01, abs=1e-6)
