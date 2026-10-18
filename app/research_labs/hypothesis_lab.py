"""Lab 7 -- Alpha Discovery & Hypothesis Lab.

A *pre-registered* experiment: the hypothesis, sample, feature, baseline, outcome and acceptance criteria are all fixed
(and hashed) BEFORE any result is computed. The spec is the experiment; the same spec on the same data with the same
seed gives the same answer.

Example spec -- "After an SSL sweep in discount, does M15 confirmation improve the expectancy of long entries?"::

    {"hypothesis": "SSL sweep in discount + M15 bullish confirmation improves long expectancy",
     "direction": "long",
     "sample":  {"event": "ssl_sweep", "lookback": 20, "zone": "discount", "zone_lookback": 96, "hours": null},
     "feature": {"type": "htf_confirm", "tf": "15min", "rule": "bullish"},
     "baseline": "same events without the feature",
     "outcome": {"horizon": 12, "unit": "atr"},
     "acceptance": {"alpha": 0.05, "min_n": 30, "min_effect_d": 0.2, "require_validation": true, "validation_fraction": 0.3}}

Reported automatically: effect size (Cohen's d and mean difference in ATR), bootstrap confidence interval, sample sizes,
a multiple-testing adjustment that grows with the number of hypotheses already tested on the SAME data (a persistent
ledger -- repeated peeking raises the bar), and an independent validation on a chronological holdout the discovery
step never saw. The verdict is a function of the acceptance criteria you wrote down, nothing else.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from app.research_labs import common as C

LAB = "hypothesis_lab"
TITLE = "Alpha Discovery & Hypothesis Lab"
ASSUMPTIONS = [
    "The spec (hypothesis, sample, feature, baseline, outcome, acceptance) is fixed and hashed before results are computed.",
    "Events fire on the bar that completes them; the outcome is measured over the NEXT ``horizon`` bars (no look-ahead). Higher-timeframe features use only COMPLETED higher-timeframe bars.",
    "Outcome is the direction-adjusted forward return in ATR(14) units, measured before the event bar (ATR is shifted by one bar).",
    "Discovery uses the first (1 - validation_fraction) of the data; validation uses the held-out last part and must AGREE for acceptance.",
    "Multiple-testing: alpha is divided by (hypotheses already tested on this exact dataset + 1) (Bonferroni over the persistent ledger), so repeatedly trying variants makes acceptance harder.",
    "No transaction costs are applied to the outcome; subtract your own cost in ATR units before trusting a small edge.",
]
EVENTS = ("ssl_sweep", "bsl_sweep", "any_bar")
FEATURES = ("none", "htf_confirm", "trend_ema", "volume_spike", "range_expansion", "session_hours", "indicator")


def default_spec() -> dict:
    return {"hypothesis": "After an SSL sweep in discount, M15 bullish confirmation improves the expectancy of long entries.",
            "direction": "long",
            "sample": {"event": "ssl_sweep", "lookback": 20, "zone": "discount", "zone_lookback": 96, "hours": None},
            "feature": {"type": "htf_confirm", "tf": "15min", "rule": "bullish"},
            "baseline": "same events without the feature",
            "outcome": {"horizon": 12, "unit": "atr"},
            "acceptance": {"alpha": 0.05, "min_n": 30, "min_effect_d": 0.2, "require_validation": True, "validation_fraction": 0.3}}


def validate_spec(spec: dict) -> dict:
    s = json.loads(json.dumps(spec)) if isinstance(spec, dict) else None
    if not s:
        raise C.ResearchLabError("The experiment spec is empty.")
    d = default_spec()
    for k in ("hypothesis", "direction", "sample", "feature", "baseline", "outcome", "acceptance"):
        if k not in s or s[k] in (None, ""):
            if k in ("sample", "feature", "outcome", "acceptance", "direction"):
                raise C.ResearchLabError(f"Spec is missing '{k}'. Define it BEFORE running -- that is the point of this lab.")
            s[k] = d[k]
    if s["direction"] not in ("long", "short"):
        raise C.ResearchLabError("direction must be 'long' or 'short'.")
    smp = {**d["sample"], **s["sample"]}
    if smp["event"] not in EVENTS:
        raise C.ResearchLabError(f"sample.event must be one of {EVENTS}.")
    if smp.get("zone") not in (None, "", "discount", "premium"):
        raise C.ResearchLabError("sample.zone must be null, 'discount' or 'premium'.")
    ft = {"type": "none", **s["feature"]}
    if ft["type"] not in FEATURES:
        raise C.ResearchLabError(f"feature.type must be one of {FEATURES}.")
    out = {**d["outcome"], **s["outcome"]}
    out["horizon"] = int(min(max(int(out["horizon"]), 1), 500))
    acc = {**d["acceptance"], **s["acceptance"]}
    acc["alpha"] = float(min(max(acc["alpha"], 1e-4), 0.2))
    acc["min_n"] = int(max(acc["min_n"], 10))
    acc["validation_fraction"] = float(min(max(acc["validation_fraction"], 0.15), 0.5))
    s.update(sample=smp, feature=ft, outcome=out, acceptance=acc)
    return s


def _ledger_path():
    return C.store_dir(LAB) / "ledger.json"


def _ledger_count(fp: str) -> int:
    try:
        return len(json.loads(_ledger_path().read_text(encoding="utf-8")).get(fp, []))
    except (OSError, ValueError):
        return 0


def _ledger_add(fp: str, spec_h: str) -> None:
    try:
        p = _ledger_path()
        led = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if spec_h not in led.setdefault(fp, []):
            led[fp].append(spec_h)
        p.write_text(json.dumps(led), encoding="utf-8")
    except OSError:
        pass


def _htf_confirm(d: pd.DataFrame, tf: str, rule: str) -> pd.Series:
    """Direction of the last COMPLETED higher-timeframe bar, forward-filled onto the base bars."""
    g = d.set_index("timestamp")[["open", "close"]].resample(tf, label="left", closed="left").agg({"open": "first", "close": "last"}).dropna()
    bull = (g["close"] > g["open"]).astype(float)
    shifted = bull.shift(1)           # completed bars only
    shifted.index = shifted.index + pd.Timedelta(tf)   # a bar is usable only after it has closed
    ser = shifted.reindex(d["timestamp"], method="ffill")
    ser.index = d.index
    return (ser == 1.0) if rule == "bullish" else (ser == 0.0)


def build_masks(d: pd.DataFrame, spec: dict) -> tuple[pd.Series, pd.Series]:
    smp, ft = spec["sample"], spec["feature"]
    sweeps = C.detect_liquidity_sweeps(d, int(smp["lookback"]))
    if smp["event"] == "any_bar":
        ev = pd.Series(True, index=d.index)
    else:
        ev = sweeps[smp["event"]].copy()
    zl = int(smp.get("zone_lookback") or 96)
    if smp.get("zone"):
        hi, lo = d["high"].shift(1).rolling(zl, min_periods=zl // 2).max(), d["low"].shift(1).rolling(zl, min_periods=zl // 2).min()
        mid = (hi + lo) / 2
        ev &= (d["close"] < mid) if smp["zone"] == "discount" else (d["close"] > mid)
    if smp.get("hours"):
        a, b = smp["hours"]
        hr = d["timestamp"].dt.hour
        ev &= (hr >= a) & (hr < b) if a < b else ((hr >= a) | (hr < b))
    t = ft["type"]
    long = spec["direction"] == "long"
    if t == "none":
        feat = pd.Series(True, index=d.index)
    elif t == "htf_confirm":
        rule = ft.get("rule") or ("bullish" if long else "bearish")
        feat = _htf_confirm(d, ft.get("tf", "15min"), rule)
    elif t == "trend_ema":
        ema = d["close"].ewm(span=int(ft.get("span", 50)), adjust=False).mean()
        feat = (d["close"] > ema) if long else (d["close"] < ema)
    elif t == "volume_spike":
        v = d["volume"]
        feat = v > v.shift(1).rolling(20, min_periods=10).mean() * float(ft.get("mult", 1.5))
    elif t == "range_expansion":
        rng = d["high"] - d["low"]
        feat = rng > rng.shift(1).rolling(20, min_periods=10).mean() * float(ft.get("mult", 1.5))
    elif t == "indicator":
        # any registered indicator (gex_regime, vp_location, flow_absorption, ...) compared to a value, so a hypothesis can
        # use the GEX analyzer's regime or any lab feature as its condition.
        from app.strategy.indicators import build_indicator_series

        kind = str(ft.get("kind") or "")
        if not kind:
            raise C.ResearchLabError("feature.kind is required for feature.type='indicator' (e.g. 'gex_regime').")
        ser = build_indicator_series(d, kind, int(ft.get("period", 14)), "close", int(ft.get("period", 14))).astype(float)
        op, val = str(ft.get("op", ">")), float(ft.get("value", 0))
        feat = {">": ser > val, ">=": ser >= val, "<": ser < val, "<=": ser <= val, "==": ser == val}.get(op)
        if feat is None:
            raise C.ResearchLabError("feature.op must be one of >, >=, <, <=, ==.")
    else:   # session_hours
        a, b = ft.get("hours", [13, 17])
        hr = d["timestamp"].dt.hour
        feat = (hr >= a) & (hr < b)
    return ev.fillna(False), feat.fillna(False)


def _stats(x: np.ndarray, y: np.ndarray, seed: int) -> dict:
    w = C.welch_test(x, y)
    lo, hi = C.diff_ci(x, y, seed=seed, n_boot=500)
    return {"n_feature": int(len(x)), "n_baseline": int(len(y)), "mean_feature": float(x.mean()) if len(x) else None,
            "mean_baseline": float(y.mean()) if len(y) else None,
            "diff": float(x.mean() - y.mean()) if len(x) and len(y) else None, "ci_low": lo, "ci_high": hi, "p": w["p"], "t": w["t"],
            "effect_d": C.cohens_d(x, y), "hit_feature": float((x > 0).mean()) if len(x) else None,
            "hit_baseline": float((y > 0).mean()) if len(y) else None}


def run(df: pd.DataFrame, spec: dict | None = None, *, label: str = "series", seed: int = C.DEFAULT_SEED, save: bool = True,
        use_ledger: bool = True) -> C.ExperimentRecord:
    spec = validate_spec(spec or default_spec())
    d = C.prepare_ohlcv(df)
    if len(d) < 500:
        raise C.ResearchLabError("Need at least 500 bars for a hypothesis test.")
    fp = C.data_fingerprint(d, label)["fingerprint"]
    sh = C.spec_hash(spec)
    prior = _ledger_count(fp) if use_ledger else 0
    already = False
    try:
        already = sh in json.loads(_ledger_path().read_text(encoding="utf-8")).get(fp, [])
    except (OSError, ValueError):
        pass
    n_tests = prior if already else prior + 1
    acc = spec["acceptance"]
    alpha_adj = acc["alpha"] / max(n_tests, 1)
    h = spec["outcome"]["horizon"]
    atr = C.atr_series(d, 14).shift(1)
    sign = 1.0 if spec["direction"] == "long" else -1.0
    outcome = sign * (d["close"].shift(-h) - d["close"]) / atr
    ev, feat = build_masks(d, spec)
    cut = int(len(d) * (1 - acc["validation_fraction"]))
    pos = np.arange(len(d))
    finite = outcome.notna() & np.isfinite(outcome)
    disc_ev, val_ev = ev & finite & (pos < cut), ev & finite & (pos >= cut)

    def split(m):
        a = outcome[m & feat].to_numpy()
        b = outcome[m & ~feat].to_numpy()
        return a, b

    da, db = split(disc_ev)
    va, vb = split(val_ev)
    disc, val = _stats(da, db, seed), _stats(va, vb, seed)
    warnings = []
    if spec["feature"]["type"] == "none":
        warnings.append("Feature is 'none': the test compares an event with itself; use it only to read the baseline expectancy.")
    if len(da) < acc["min_n"] or len(db) < acc["min_n"]:
        warnings.append(f"Discovery sample is small (feature n={len(da)}, baseline n={len(db)}; need >= {acc['min_n']}).")
    crit = {
        "enough_samples": len(da) >= acc["min_n"] and len(db) >= acc["min_n"],
        "significant_after_adjustment": disc["p"] is not None and disc["p"] < alpha_adj,
        "effect_size": disc["effect_d"] is not None and abs(disc["effect_d"]) >= acc["min_effect_d"] and (disc["diff"] or 0) > 0,
        "ci_excludes_zero": disc["ci_low"] is not None and disc["ci_low"] > 0,
    }
    if acc["require_validation"]:
        crit["validation_agrees"] = (len(va) >= max(5, acc["min_n"] // 3) and val["diff"] is not None and val["diff"] > 0
                                     and val["p"] is not None and val["p"] < 0.10)
    passed = all(crit.values())
    verdict = ("ACCEPTED: every pre-registered criterion passed" if passed else
               "REJECTED: " + ", ".join(k.replace("_", " ") for k, v in crit.items() if not v) + " failed")
    base_exp = float(outcome[disc_ev & ~feat].mean()) if (disc_ev & ~feat).any() else None
    results = {"headline": f"{verdict}. Feature vs baseline difference {disc['diff'] if disc['diff'] is None else round(disc['diff'], 3)} ATR "
                           f"(discovery), {val['diff'] if val['diff'] is None else round(val['diff'], 3)} ATR (validation).",
               "spec": spec, "spec_hash": sh, "verdict": verdict, "accepted": bool(passed), "criteria": crit,
               "alpha_nominal": acc["alpha"], "alpha_adjusted": alpha_adj, "hypotheses_tested_on_this_data": n_tests,
               "discovery": disc, "validation": val, "events_total": int(ev.sum()), "baseline_expectancy_atr": base_exp,
               "interpretation": "Expectancy is the mean direction-adjusted forward move in ATR units; a positive feature-minus-baseline "
                                 "difference that survives the adjusted alpha AND the held-out validation is evidence, not proof."}
    rec = C.record_from(LAB, TITLE, {"spec": spec}, ASSUMPTIONS, d, results, warnings, seed, label, save=save)
    if save and use_ledger and not already:
        _ledger_add(fp, sh)
    return rec
