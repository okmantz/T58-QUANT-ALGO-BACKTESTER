"""Shared plumbing for the seven T58 research labs (Oct 2026).

Every lab is a *research engine*, not a dashboard: it takes data + an explicit parameter set, runs a reproducible
experiment, and returns an :class:`ExperimentRecord` that

* preserves its assumptions (what the lab assumed, what data it really had, which parts are PROXIES),
* is seeded (same inputs + same seed => identical numbers),
* is split chronologically into in-sample / out-of-sample so a finding is judged on data it was not tuned on,
* is saved as JSON under ``data/research_labs/<lab>/`` so it can be re-opened, compared, sent to the Trading Journal,
  or used to drive the backtester through the ``indicators_v10`` terminals (see ``app.strategy.indicators_v10``).

Statistics helpers here (bootstrap CI, Welch test, Cohen's d, Holm / Benjamini-Hochberg adjustment) are deliberately
dependency-light: numpy always, scipy only where it makes the p-value exact (a normal approximation is the fallback).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

ENGINE_VERSION = "1.0"
DEFAULT_SEED = 58
# Hard cap on bars handed to any lab so one click can never exhaust memory (most recent bars are kept).
MAX_BARS = 400_000


class ResearchLabError(ValueError):
    """A lab could not run, with a message that is safe to show the person verbatim."""


# --------------------------------------------------------------------------------------------------------------
# Experiment record
# --------------------------------------------------------------------------------------------------------------
@dataclass
class ExperimentRecord:
    lab: str
    title: str
    params: dict
    assumptions: list[str]
    data: dict                       # fingerprint + capability description of the data actually used
    results: dict                    # lab-specific, JSON-safe
    warnings: list[str] = field(default_factory=list)
    seed: int = DEFAULT_SEED
    status: str = "ok"               # "ok" | "needs_data"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_utc: float = field(default_factory=time.time)
    engine_version: str = ENGINE_VERSION
    spec_hash: str = ""

    def __post_init__(self) -> None:
        if not self.spec_hash:
            self.spec_hash = spec_hash({"lab": self.lab, "params": self.params, "seed": self.seed,
                                        "data": self.data.get("fingerprint")})

    def to_dict(self) -> dict:
        return json_safe(asdict(self))

    def summary_line(self) -> str:
        headline = self.results.get("headline") if isinstance(self.results, dict) else None
        return f"{self.title}: {headline}" if headline else self.title


def spec_hash(obj: Any) -> str:
    """Stable short hash of a JSON-able spec -- the 'pre-registration' fingerprint of an experiment."""
    blob = json.dumps(json_safe(obj), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def json_safe(obj: Any) -> Any:
    """Recursively converts numpy / pandas / NaN values to plain JSON-safe Python."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


# --------------------------------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------------------------------
def _base_dir() -> Path:
    from app.data.storage import get_app_base_dir

    return Path(get_app_base_dir())


def store_dir(lab: str) -> Path:
    d = _base_dir() / "data" / "research_labs" / lab
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_experiment(rec: ExperimentRecord) -> Path:
    path = store_dir(rec.lab) / f"{rec.id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec.to_dict(), indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load_experiment(lab: str, exp_id: str) -> Optional[dict]:
    if not exp_id.isalnum():
        return None
    path = store_dir(lab) / f"{exp_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_experiments(lab: str | None = None, limit: int = 30) -> list[dict]:
    root = _base_dir() / "data" / "research_labs"
    if not root.is_dir():
        return []
    out: list[dict] = []
    labs = [lab] if lab else [p.name for p in root.iterdir() if p.is_dir()]
    for name in labs:
        d = root / name
        if not d.is_dir():
            continue
        for f in d.glob("*.json"):
            if f.name == "ledger.json":
                continue
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"id": doc.get("id"), "lab": doc.get("lab"), "title": doc.get("title"),
                        "created_utc": doc.get("created_utc", 0), "status": doc.get("status"),
                        "headline": (doc.get("results") or {}).get("headline"), "spec_hash": doc.get("spec_hash")})
    out.sort(key=lambda r: r.get("created_utc") or 0, reverse=True)
    return out[:limit]


# --------------------------------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------------------------------
def prepare_ohlcv(df: pd.DataFrame, max_bars: int = MAX_BARS) -> pd.DataFrame:
    """Validated, time-sorted OHLCV copy with a tz-naive ``timestamp`` column and a clean RangeIndex."""
    if df is None or len(df) == 0:
        raise ResearchLabError("The dataset is empty.")
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "timestamp" not in out.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            out.insert(0, "timestamp", df.index)
        else:
            raise ResearchLabError("The dataset needs a 'timestamp' column.")
    missing = [c for c in ("open", "high", "low", "close") if c not in out.columns]
    if missing:
        raise ResearchLabError(f"The dataset is missing required column(s): {', '.join(missing)}.")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=False)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    out["timestamp"] = ts
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    if len(out) > max_bars:
        out = out.iloc[-max_bars:]
    out = out.reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    if len(out) < 50:
        raise ResearchLabError(f"Only {len(out)} usable bars -- a lab needs at least 50.")
    return out


def median_bar_minutes(df: pd.DataFrame) -> float:
    d = df["timestamp"].diff().dropna()
    if d.empty:
        return 1.0
    return max(float(d.median().total_seconds() / 60.0), 1.0 / 60.0)


def data_fingerprint(df: pd.DataFrame, label: str = "") -> dict:
    """Describes the data a result came from, and what it can / cannot support (so assumptions travel with it)."""
    cols = set(df.columns)
    has_trade_flow = bool({"bid_volume", "ask_volume"} <= cols or {"side", "size"} <= cols)
    has_l2 = any(c.startswith("bid_px_") or c.startswith("bid_price_") for c in cols) or {"bid_px", "ask_px"} <= cols
    first, last = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    sample = df["close"].to_numpy()[:: max(1, len(df) // 64)]
    return {
        "label": label, "rows": int(len(df)), "first": first.isoformat(), "last": last.isoformat(),
        "bar_minutes": round(median_bar_minutes(df), 4),
        "has_volume": bool(float(df["volume"].sum()) > 0), "has_trade_level_flow": has_trade_flow, "has_level2": has_l2,
        "fingerprint": hashlib.sha256(np.round(sample, 8).tobytes() + str(len(df)).encode()).hexdigest()[:16],
    }


def chrono_split(df: pd.DataFrame, oos_fraction: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological in-sample / out-of-sample split (never shuffled)."""
    oos_fraction = min(max(float(oos_fraction), 0.1), 0.6)
    cut = int(len(df) * (1.0 - oos_fraction))
    cut = min(max(cut, 10), len(df) - 10)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Return from this bar's close to the close ``horizon`` bars later (NaN at the tail). Uses FUTURE data on
    purpose -- it is the *outcome* being studied, never a feature."""
    return close.shift(-horizon) / close - 1.0


# --------------------------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------------------------
def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def welch_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Welch t-test; exact p via scipy when installed, normal approximation otherwise."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"t": None, "p": None, "n_a": int(len(a)), "n_b": int(len(b))}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return {"t": 0.0, "p": 1.0, "n_a": int(len(a)), "n_b": int(len(b))}
    t = (a.mean() - b.mean()) / se
    try:
        from scipy import stats

        dof = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
        p = float(2 * stats.t.sf(abs(t), dof))
    except Exception:  # noqa: BLE001
        p = float(2 * _norm_sf(abs(t)))
    return {"t": float(t), "p": p, "n_a": int(len(a)), "n_b": int(len(b))}


def one_sample_t(x: np.ndarray, mu: float = 0.0) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"t": None, "p": None, "n": int(len(x))}
    sd = x.std(ddof=1)
    if sd == 0:
        return {"t": 0.0, "p": 1.0, "n": int(len(x))}
    t = (x.mean() - mu) / (sd / math.sqrt(len(x)))
    try:
        from scipy import stats

        p = float(2 * stats.t.sf(abs(t), len(x) - 1))
    except Exception:  # noqa: BLE001
        p = float(2 * _norm_sf(abs(t)))
    return {"t": float(t), "p": p, "n": int(len(x))}


def cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def bootstrap_ci(x: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = DEFAULT_SEED, block: int = 1) -> tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI. ``block > 1`` resamples contiguous blocks (for autocorrelated series)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return None, None
    rng = np.random.default_rng(seed)
    stats_ = np.empty(n_boot)
    if block <= 1:
        for i in range(n_boot):
            stats_[i] = stat(x[rng.integers(0, n, n)])
    else:
        nb = int(math.ceil(n / block))
        for i in range(n_boot):
            starts = rng.integers(0, max(1, n - block + 1), nb)
            sample = np.concatenate([x[s:s + block] for s in starts])[:n]
            stats_[i] = stat(sample)
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
            seed: int = DEFAULT_SEED) -> tuple[Optional[float], Optional[float]]:
    """Bootstrap CI for mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return None, None
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm_adjust(pvals: Iterable[Optional[float]]) -> list[Optional[float]]:
    """Holm-Bonferroni step-down adjusted p-values (controls family-wise error). ``None`` stays ``None``."""
    ps = list(pvals)
    idx = [i for i, p in enumerate(ps) if p is not None]
    m = len(idx)
    out: list[Optional[float]] = [None] * len(ps)
    order = sorted(idx, key=lambda i: ps[i])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * float(ps[i]))
        running = max(running, adj)
        out[i] = running
    return out


def bh_adjust(pvals: Iterable[Optional[float]]) -> list[Optional[float]]:
    """Benjamini-Hochberg adjusted p-values (q-values; controls false discovery rate)."""
    ps = list(pvals)
    idx = [i for i, p in enumerate(ps) if p is not None]
    m = len(idx)
    out: list[Optional[float]] = [None] * len(ps)
    order = sorted(idx, key=lambda i: ps[i], reverse=True)
    running = 1.0
    for k, i in enumerate(order):
        rank = m - k
        running = min(running, float(ps[i]) * m / rank)
        out[i] = min(1.0, running)
    return out


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[Optional[float], Optional[float]]:
    """CI for a correlation via Fisher z (n is the effective sample size)."""
    if n is None or n <= 4 or r is None or not math.isfinite(r) or abs(r) >= 1:
        return None, None
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    zc = 1.959964 if abs(alpha - 0.05) < 1e-9 else abs(_inv_norm(alpha / 2))
    return float(math.tanh(z - zc * se)), float(math.tanh(z + zc * se))


def _inv_norm(p: float) -> float:
    # Acklam's approximation (ample accuracy for CI half-widths).
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - plow:
        return -_inv_norm(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def describe_group(x: np.ndarray, seed: int = DEFAULT_SEED) -> dict:
    """n / mean / median / hit-rate with a bootstrap CI for the mean -- the standard cell of every lab's tables."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": None, "median": None, "hit_rate": None, "ci_low": None, "ci_high": None}
    lo, hi = bootstrap_ci(x, seed=seed, n_boot=400)
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "hit_rate": float((x > 0).mean()), "ci_low": lo, "ci_high": hi}


# --------------------------------------------------------------------------------------------------------------
# Event detection shared by several labs
# --------------------------------------------------------------------------------------------------------------
def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 20, confirm_close: bool = True) -> pd.DataFrame:
    """Sell-side (SSL) and buy-side (BSL) liquidity sweeps, causal (uses only completed prior bars):

    * ``ssl_sweep`` -- the bar trades below the lowest low of the previous ``lookback`` bars, and (when
      ``confirm_close``) closes back above that level (a stop run that failed to continue).
    * ``bsl_sweep`` -- mirror image above the prior highest high.

    A sweep fires on the bar that completes it; the study then measures what happens AFTER that bar.
    """
    prior_low = df["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    prior_high = df["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    ssl = df["low"] < prior_low
    bsl = df["high"] > prior_high
    if confirm_close:
        ssl &= df["close"] > prior_low
        bsl &= df["close"] < prior_high
    return pd.DataFrame({"ssl_sweep": ssl.fillna(False), "bsl_sweep": bsl.fillna(False),
                         "prior_low": prior_low, "prior_high": prior_high}, index=df.index)


def sweep_directional_return(df: pd.DataFrame, sweeps: pd.DataFrame, horizon: int) -> pd.Series:
    """Forward return in the *reversal* direction of each sweep (SSL -> long, BSL -> short); NaN elsewhere."""
    fwd = forward_returns(df["close"], horizon)
    out = pd.Series(np.nan, index=df.index)
    out[sweeps["ssl_sweep"]] = fwd[sweeps["ssl_sweep"]]
    out[sweeps["bsl_sweep"]] = -fwd[sweeps["bsl_sweep"]]
    return out


def release_memory() -> None:
    """Best-effort RAM release after a lab finishes (never raises)."""
    try:
        from app.orchestration.memory_hygiene import release_memory as _rm

        _rm("research-lab")
    except Exception:  # noqa: BLE001
        import gc

        gc.collect()


def record_from(lab: str, title: str, params: dict, assumptions: list[str], df: pd.DataFrame, results: dict,
                warnings: list[str] | None = None, seed: int = DEFAULT_SEED, label: str = "",
                status: str = "ok", save: bool = True) -> ExperimentRecord:
    rec = ExperimentRecord(lab=lab, title=title, params=json_safe(params), assumptions=assumptions,
                           data=data_fingerprint(df, label), results=json_safe(results),
                           warnings=list(warnings or []), seed=seed, status=status)
    if save:
        try:
            save_experiment(rec)
        except OSError:
            rec.warnings.append("Could not save this experiment to disk (read-only folder?); results are shown only.")
    return rec


def group_compare(outcome: pd.Series, event_mask: pd.Series, groups: dict[str, pd.Series], *, oos_fraction: float = 0.3,
                  min_n: int = 15, seed: int = DEFAULT_SEED, fdr_q: float = 0.10) -> list[dict]:
    """The standard 'does this condition change the outcome?' table used by Labs 1, 2, 3, 6 and 7.

    For each named boolean ``groups`` mask (restricted to ``event_mask`` rows with a finite ``outcome``): n, mean, CI,
    hit rate, Welch t / p versus the REST of the events, Cohen's d, a bootstrap CI of the difference, a
    Benjamini-Hochberg q-value across all groups, and in-sample vs out-of-sample means (chronological split).
    """
    outcome = pd.Series(outcome).astype(float)
    ev = event_mask & outcome.notna()
    cut = int(len(outcome) * (1 - min(max(oos_fraction, 0.1), 0.6)))
    pos = np.arange(len(outcome))
    rows, pvals = [], []
    for name, mask in groups.items():
        mask = mask.fillna(False).astype(bool)
        sel = ev & mask
        in_s, out_s = sel & (pos < cut), sel & (pos >= cut)
        grp, rest = outcome[sel].to_numpy(), outcome[ev & ~mask].to_numpy()
        w = welch_test(grp, rest)
        pvals.append(w["p"])
        lo, hi = diff_ci(grp, rest, seed=seed, n_boot=400)
        rows.append({"group": name, **describe_group(grp, seed=seed), "rest_n": int(len(rest)),
                     "rest_mean": float(rest.mean()) if len(rest) else None, "t": w["t"], "p": w["p"],
                     "effect_d": cohens_d(grp, rest), "diff_ci_low": lo, "diff_ci_high": hi,
                     "in_sample_n": int(in_s.sum()), "in_sample_mean": float(outcome[in_s].mean()) if in_s.sum() else None,
                     "oos_n": int(out_s.sum()), "oos_mean": float(outcome[out_s].mean()) if out_s.sum() else None})
    for row, q in zip(rows, bh_adjust(pvals)):
        row["q_value_bh"] = q
        same_sign = (row["in_sample_mean"] is not None and row["oos_mean"] is not None
                     and np.sign(row["in_sample_mean"]) == np.sign(row["oos_mean"]) and row["oos_n"] >= max(5, min_n // 3))
        if row["n"] < min_n:
            row["verdict"] = "too few events"
        elif q is not None and q < fdr_q and same_sign:
            row["verdict"] = "promising: significant after correction AND same sign out-of-sample"
        else:
            row["verdict"] = "no reliable difference (not significant after multiple-testing correction, or OOS disagrees)"
    return rows
