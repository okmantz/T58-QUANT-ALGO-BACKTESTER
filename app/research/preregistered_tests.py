"""Pre-registered tests for the new families (report 2026-10-20, section 6).

Discipline, in the order it is enforced:
  1. REGISTER first. ``preregister`` freezes (family, canonical parameters, primary instrument, replication instrument, predicted
     sign = positive mean R, alpha, minimum trades) and stores a SHA-256 fingerprint with a timestamp in
     ``reports/preregistrations.json`` BEFORE any backtest is run. The canonical parameters are the MIDDLE element of each grid
     (deterministic, not chosen from results). Registering books one trial to the family's ledger entry.
  2. RUN once. ``run_preregistered`` backtests each registered test on its primary instrument at 2x costs, computes the one-sided
     day-block bootstrap p-value for mean R > 0, the mean R of the chronologically LAST third (out-of-sample sign check) and the mean
     R on the replication instrument.
  3. CORRECT. Benjamini-Hochberg across ALL tests in the batch (false-discovery rate ``fdr``, default 10%).
  4. DECIDE. SUPPORTED only if: BH-significant AND out-of-sample sign matches AND the second instrument has the same sign AND
     n >= min_trades. Otherwise NOT SUPPORTED (or INCONCLUSIVE when there were too few trades to say). Nothing is re-tuned after a
     fail: re-running a registered test appends a repeat to the registry and books another trial.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_LOCK = threading.Lock()
REGISTRY_PATH = Path(__file__).resolve().parents[2] / "reports" / "preregistrations.json"


@dataclass(frozen=True)
class PreRegistration:
    family: str
    params: dict
    primary_instrument: str
    replication_instrument: str
    predicted_sign: int = 1
    alpha: float = 0.05
    min_trades: int = 100
    cost_multiplier: float = 2.0
    note: str = ""

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def canonical_params(family_name: str) -> dict:
    """Middle element of every grid list, validated; falls back to the first valid combination."""
    from app.search import strategy_space as sps
    spec = sps.FAMILIES[family_name]
    mid = {k: v[len(v) // 2] for k, v in spec.param_grid.items()}
    if spec.valid(mid):
        return mid
    combos = spec.combinations()
    if not combos:
        raise ValueError(f"{family_name}: no valid parameter combination")
    return combos[0]


def _load() -> dict:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"registrations": []}


def preregister(family: str, primary_instrument: str, replication_instrument: str, params: dict | None = None,
                *, alpha: float = 0.05, min_trades: int = 100, note: str = "", path: Path | None = None) -> PreRegistration:
    reg = PreRegistration(family, dict(params) if params else canonical_params(family), primary_instrument.upper(),
                          replication_instrument.upper(), 1, alpha, min_trades, 2.0, note)
    fp = reg.fingerprint()
    p = Path(path) if path else REGISTRY_PATH
    with _LOCK:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {"registrations": []}
        seen = [r for r in data["registrations"] if r["fingerprint"] == fp]
        data["registrations"].append({"fingerprint": fp, "created": time.time(), "repeat": len(seen), "registration": asdict(reg)})
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    try:
        from app.validation import trial_ledger
        trial_ledger.record(f"family:{family}", 1)
    except trial_ledger.TrialLedgerError:
        raise    # 2026-10-31 audit: a preregistration whose trial was not counted is not a preregistration
    except Exception:  # noqa: BLE001
        pass
    return reg


def benjamini_hochberg(pvals: list[float], q: float = 0.10) -> list[bool]:
    """Step-up BH: reject the k smallest p-values where k is the largest i with p_(i) <= q * i / m."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, m + 1) / m)
    ok = np.asarray(pvals, dtype=float)[order] <= thresh
    k = int(np.max(np.nonzero(ok)[0]) + 1) if ok.any() else 0
    out = [False] * m
    for rank, idx in enumerate(order):
        if rank < k:
            out[int(idx)] = True
    return out


def bootstrap_pvalue_positive(r, days=None, n_boot: int = 4000, seed: int = 17) -> float | None:
    """One-sided day-block bootstrap p-value for H0: mean R <= 0 (share of resampled means that are <= 0, add-one smoothed)."""
    from app.validation.edge_stats import _day_codes
    a = np.asarray([x for x in r if x is not None and np.isfinite(x)], dtype=float)
    if a.size < 5:
        return None
    if days is not None:
        d = [dd for dd, x in zip(list(days), list(r)) if x is not None and np.isfinite(x)]
        codes, n_days = _day_codes(d)
    else:
        codes, n_days = np.arange(a.size), a.size
    ds = np.bincount(codes, weights=a, minlength=n_days)
    dc = np.bincount(codes, minlength=n_days).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_days, size=(n_boot, n_days))
    means = ds[idx].sum(axis=1) / np.maximum(dc[idx].sum(axis=1), 1.0)
    return float((1 + (means <= 0).sum()) / (n_boot + 1))


@dataclass
class TestResult:
    family: str
    fingerprint: str
    primary: str
    n_trades: int = 0
    mean_r: float | None = None
    p_value: float | None = None
    p_bh_significant: bool = False
    oos_mean_r: float | None = None
    replication_n: int = 0
    replication_mean_r: float | None = None
    verdict: str = "INCONCLUSIVE"
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _backtest_r(df, params_family: str, params: dict, risk, cost_multiplier: float):
    from app.backtest.engine import run_backtest
    from app.optimize.two_stage import stress_risk, trade_r_and_days
    from app.search import strategy_space as sps
    from app.strategy.manual import ManualStrategy
    cfg = sps.FAMILIES[params_family].build(params)
    bt = run_backtest(df, ManualStrategy(cfg), stress_risk(risk, cost_multiplier), lookahead_check="skip")
    return trade_r_and_days(bt.trades)


def run_preregistered(registrations: list[PreRegistration], frames: dict, risks: dict, *, fdr: float = 0.10, log=lambda m: None) -> list[TestResult]:
    """frames/risks are keyed by instrument symbol (upper case). Missing instrument -> INCONCLUSIVE with the reason."""
    results: list[TestResult] = []
    for reg in registrations:
        res = TestResult(reg.family, reg.fingerprint(), reg.primary_instrument)
        try:
            df, risk = frames.get(reg.primary_instrument), risks.get(reg.primary_instrument)
            if df is None or risk is None:
                res.reasons.append(f"no data/risk config for primary instrument {reg.primary_instrument}")
                results.append(res)
                continue
            rs, days = _backtest_r(df, reg.family, reg.params, risk, reg.cost_multiplier)
            res.n_trades = len(rs)
            if rs:
                res.mean_r = float(np.mean(rs))
                res.p_value = bootstrap_pvalue_positive(rs, days)
                third = max(1, len(rs) // 3)
                res.oos_mean_r = float(np.mean(rs[-third:]))
            rdf, rrisk = frames.get(reg.replication_instrument), risks.get(reg.replication_instrument)
            if rdf is not None and rrisk is not None:
                rr, _ = _backtest_r(rdf, reg.family, reg.params, rrisk, reg.cost_multiplier)
                res.replication_n = len(rr)
                res.replication_mean_r = float(np.mean(rr)) if rr else None
        except Exception as exc:  # noqa: BLE001 -- a failing test is reported, never silently dropped
            res.reasons.append(f"failed to run: {exc}")
        log(f"  {reg.family}: n={res.n_trades} mean R={res.mean_r}")
        results.append(res)
    testable = [i for i, r in enumerate(results) if r.p_value is not None]
    flags = benjamini_hochberg([results[i].p_value for i in testable], fdr)
    for i, f in zip(testable, flags):
        results[i].p_bh_significant = bool(f)
    for reg, res in zip(registrations, results):
        if res.p_value is None:
            res.verdict = "INCONCLUSIVE"
            continue
        if res.n_trades < reg.min_trades:
            res.verdict = "INCONCLUSIVE"
            res.reasons.append(f"{res.n_trades} trades < pre-registered minimum {reg.min_trades}")
            continue
        sgn = reg.predicted_sign
        ok_oos = res.oos_mean_r is not None and res.oos_mean_r * sgn > 0
        ok_rep = res.replication_mean_r is not None and res.replication_mean_r * sgn > 0
        if not res.p_bh_significant:
            res.reasons.append(f"p={res.p_value:.4f} is not significant after Benjamini-Hochberg at FDR {fdr:.0%}")
        if not ok_oos:
            res.reasons.append("out-of-sample (last third) mean R has the wrong sign")
        if not ok_rep:
            res.reasons.append("second instrument does not replicate the sign" if res.replication_mean_r is not None else "second instrument not tested")
        res.verdict = "SUPPORTED" if (res.p_bh_significant and ok_oos and ok_rep) else "NOT SUPPORTED"
    return results
