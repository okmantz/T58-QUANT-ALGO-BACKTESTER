"""
Fama-French 3-Factor Model -- decomposes a stock's (or strategy's) return
series into exposure to three well-documented systematic factors:

    R_i - RF = alpha + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML + epsilon

  - Mkt-RF: the broad market's excess return over the risk-free rate.
  - SMB ("Small Minus Big"): the size factor -- small-cap minus large-cap returns.
  - HML ("High Minus Low"): the value factor -- high book-to-market (value)
    minus low book-to-market (growth) returns.

The regression itself is a plain multiple linear OLS, implemented from
scratch with matrix algebra (beta = (X'X)^-1 X'y) -- no statsmodels
dependency. What this answers, directly: is a stock/strategy's apparent
outperformance REAL skill (a statistically significant, non-zero alpha
left over after controlling for these three factors) or is it FULLY
EXPLAINED by simply being exposed to the market, small-caps, and/or value
stocks (an alpha that's statistically indistinguishable from zero once
those exposures are accounted for)?

Factor data comes from Kenneth French's public Data Library (Dartmouth
Tuck), the standard academic/practitioner source for these factors --
`fetch_fama_french_factors()` downloads and parses the official CSV
directly (real HTTP + zip + CSV parsing, no third-party
`pandas-datareader`-style dependency).

Honest statistical limitation, stated plainly rather than glossed over:
the t-statistics here use ORDINARY (non-robust) OLS standard errors. Real
daily/monthly return series are commonly autocorrelated and
heteroskedastic, which ordinary OLS standard errors don't correct for
(a full correction needs a Newey-West / HAC covariance estimator, which
this module does not implement) -- so treat the significance verdict as a
useful first read, not a publication-grade test. `n_observations` and the
raw regression output are always reported alongside the verdict so you
can judge sample size for yourself.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

_FACTOR_URLS = {
    "daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "monthly": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
}


class FactorModelError(Exception):
    """Raised when factor data can't be fetched/parsed, or there's too
    little overlapping data with the stock's own returns to regress."""


def fetch_fama_french_factors(frequency: str = "daily", timeout: float = 30.0) -> pd.DataFrame:
    """Downloads and parses Kenneth French's official Fama/French 3
    Factors CSV (daily or monthly). Returns a DataFrame indexed by date
    with columns mkt_rf, smb, hml, rf -- all as DECIMAL returns (the raw
    file reports percentages, e.g. 0.45 meaning 0.45%, so this divides by
    100 to match this app's own return convention elsewhere)."""
    frequency = frequency.lower()
    if frequency not in _FACTOR_URLS:
        raise FactorModelError(f"frequency must be one of {list(_FACTOR_URLS)}, got '{frequency}'.")
    url = _FACTOR_URLS[frequency]
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise FactorModelError(f"Failed to download Fama-French factor data from {url}: {exc}") from exc

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            inner_name = zf.namelist()[0]
            text = zf.read(inner_name).decode("utf-8", errors="ignore")
    except zipfile.BadZipFile as exc:
        raise FactorModelError("Downloaded file was not a valid zip -- Ken French's file format may have changed.") from exc

    return _parse_fama_french_csv(text, frequency)


def _parse_fama_french_csv(text: str, frequency: str) -> pd.DataFrame:
    """The raw file has descriptive header/footer text surrounding the
    actual data rows. Data rows are identified structurally (an 8-digit
    YYYYMMDD for daily, or 6-digit YYYYMM for monthly, followed by 4
    comma-separated numeric fields) rather than by a fixed line number,
    since the exact header row count has changed across Ken French's own
    file revisions over the years."""
    date_len = 8 if frequency == "daily" else 6
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        date_str = parts[0]
        if not (date_str.isdigit() and len(date_str) == date_len):
            continue
        try:
            values = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        date = pd.to_datetime(date_str, format="%Y%m%d" if frequency == "daily" else "%Y%m")
        rows.append((date, *values))

    if len(rows) < 10:
        raise FactorModelError(
            "Could not find enough structured data rows in the downloaded Fama-French file -- "
            "the file format may have changed."
        )
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"]).set_index("date")
    return df / 100.0


@dataclass
class FactorModelResult:
    alpha: float                  # per-period (matches the input return frequency), NOT annualized
    alpha_annualized: float
    alpha_t_stat: float
    alpha_significant: bool       # |t| > 1.96, i.e. roughly the 5% two-sided threshold
    beta_mkt: float
    beta_smb: float
    beta_hml: float
    t_stats: dict                 # {'mkt': t, 'smb': t, 'hml': t}
    r_squared: float
    n_observations: int
    periods_per_year: int

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d

    def render_summary(self) -> str:
        verdict = (
            f"Alpha IS statistically significant (t={self.alpha_t_stat:+.2f}) -- performance is not "
            "fully explained by market/size/value exposure alone."
            if self.alpha_significant else
            f"Alpha is NOT statistically significant (t={self.alpha_t_stat:+.2f}) -- observed "
            "performance is consistent with being fully explained by factor exposure, not skill."
        )
        return (
            f"n={self.n_observations} periods, R^2={self.r_squared:.3f}\n"
            f"Alpha: {self.alpha:+.4%} per period ({self.alpha_annualized:+.2%} annualized), t={self.alpha_t_stat:+.2f}\n"
            f"Market beta: {self.beta_mkt:+.3f} (t={self.t_stats['mkt']:+.2f})\n"
            f"Size (SMB) beta: {self.beta_smb:+.3f} (t={self.t_stats['smb']:+.2f})\n"
            f"Value (HML) beta: {self.beta_hml:+.3f} (t={self.t_stats['hml']:+.2f})\n"
            f"{verdict}"
        )


def _ols_with_t_stats(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Plain multiple OLS via normal equations: beta = (X'X)^-1 X'y.
    Returns (beta, t_stats, r_squared). X must already include an
    intercept column of ones."""
    n, k = X.shape
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError as exc:
        raise FactorModelError(
            "Design matrix is singular (e.g. two factors that are perfectly collinear over this "
            "sample) -- cannot fit the regression."
        ) from exc
    beta = xtx_inv @ X.T @ y
    fitted = X @ beta
    resid = y - fitted
    dof = max(n - k, 1)
    ssr = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ssr / sst if sst > 0 else 0.0
    sigma2 = ssr / dof
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t_stats = beta / np.where(se > 0, se, np.nan)
    return beta, t_stats, r_squared


def compute_factor_exposures(
    stock_returns: pd.Series, factors_df: pd.DataFrame, periods_per_year: int = 252,
) -> FactorModelResult:
    """stock_returns: a Series of period returns (decimal, e.g. 0.01 for
    1%), indexed by date, matching the SAME frequency as `factors_df`
    (daily returns need daily factors, monthly needs monthly).
    factors_df: output of fetch_fama_french_factors()."""
    s = stock_returns.copy()
    s.index = pd.to_datetime(s.index)
    merged = pd.concat([s.rename("stock"), factors_df], axis=1, join="inner").dropna()
    if len(merged) < 30:
        raise FactorModelError(
            f"Only {len(merged)} overlapping periods between the stock's returns and the factor "
            "data -- need at least 30 for a meaningful regression."
        )

    excess = (merged["stock"] - merged["rf"]).to_numpy()
    X = np.column_stack([
        np.ones(len(merged)), merged["mkt_rf"].to_numpy(), merged["smb"].to_numpy(), merged["hml"].to_numpy(),
    ])
    beta, t_stats, r_squared = _ols_with_t_stats(X, excess)
    alpha, beta_mkt, beta_smb, beta_hml = beta
    alpha_t = float(t_stats[0])

    return FactorModelResult(
        alpha=float(alpha), alpha_annualized=float((1 + alpha) ** periods_per_year - 1),
        alpha_t_stat=alpha_t, alpha_significant=abs(alpha_t) > 1.96,
        beta_mkt=float(beta_mkt), beta_smb=float(beta_smb), beta_hml=float(beta_hml),
        t_stats={"mkt": float(t_stats[1]), "smb": float(t_stats[2]), "hml": float(t_stats[3])},
        r_squared=float(r_squared), n_observations=len(merged), periods_per_year=periods_per_year,
    )


def compute_returns_from_prices(price_df: pd.DataFrame) -> pd.Series:
    """Convenience: turns a standard OHLCV DataFrame into a date-indexed
    daily close-to-close simple-return Series, ready for
    compute_factor_exposures()."""
    s = price_df[["timestamp", "close"]].copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"])
    if s["timestamp"].dt.tz is not None:
        s["timestamp"] = s["timestamp"].dt.tz_localize(None)
    s = s.set_index("timestamp").sort_index()
    daily_close = s["close"].resample("1D").last().dropna()
    return daily_close.pct_change().dropna()
