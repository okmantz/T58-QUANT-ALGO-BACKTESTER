"""
Research Integrity: is the lab honest? Two independent pieces of evidence, each labelled for what it is.

1. The SURROGATE STUDY (``app.discovery_engine.surrogate``) runs the REAL discovery engine on data with no edge (false discoveries) and on
   data with a planted edge (detection). It is slow, so it runs in the background and writes a report that gates new strategy families.
2. The BENCHMARK SUITE here is a fast, fully reproducible *statistical-chain* benchmark: a fixed, simple selection chain (pick the best of a
   grid of trend rules in-sample, then demand it holds up out-of-sample) run on the same kinds of surrogate data. It is NOT the production
   engine and says so everywhere. It exists to prove the measurement machinery works, to catch regressions in the statistics, and to be
   re-run in seconds from a seed. Its pass rule is the same as the gate: < 5% false discoveries over >= 60 null runs, >= 50% power on planted
   +0.20R and +0.30R edges, >= 10 runs per planted size.

The false-discovery rate on REAL market data is not measured by either: real data has an unknown amount of true edge, so a rate cannot be
counted. The dashboard says "not measured" rather than showing a number.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.discovery_engine import surrogate as sg

BENCH_VERSION = 1
GATE_NULL_RUNS = sg.GATE_MIN_NULL_RUNS
GATE_PLANT_RUNS = sg.GATE_MIN_PLANT_RUNS
MAX_FDR = 0.05
MIN_POWER = 0.5
FAST_GRID = [(f, s) for f in (4, 6, 8, 12, 16, 24) for s in (24, 36, 48, 72, 96) if s > f]
IS_FRACTION = 0.5
# Thresholds were chosen on CALIBRATION seeds (9,000,000+; never used for evaluation) as the least strict pair whose false-discovery rate on 300
# gross random-walk runs was about 1% (IS t >= 2.0 and OOS t >= 1.65: 3 of 300). They are not tuned on the evaluation seeds below.
IS_T_MIN = 2.0
OOS_T_MIN = 1.65
DEFAULT_DAYS = 250    # +0.20R is not detectable in 80 days of 5-minute bars by any honest test; 250 days is about a trading year


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc += a * (v - acc)
        out[i] = acc
    return out


def _tstat(r: np.ndarray) -> float:
    if len(r) < 30 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def fast_chain(frames: dict, seed: int = 0) -> dict:
    """In-sample best of the EMA-pair grid, then an out-of-sample test. Gross returns (no costs): the harshest setting for false discoveries,
    because on a no-edge walk costs would make every rule lose and hide a weak test."""
    df = next(iter(frames.values()))
    c = df["close"].to_numpy(dtype=float)
    ret = np.diff(np.log(c), prepend=np.log(c[0]))
    cut = int(len(c) * IS_FRACTION)
    best, best_t, best_pos = None, -1e9, None
    for f, s in FAST_GRID:
        pos = np.sign(_ema(c, f) - _ema(c, s))
        pos = np.concatenate([[0.0], pos[:-1]])                       # trade on the next bar: no look-ahead
        pnl = pos * ret
        t = _tstat(pnl[:cut])
        if t > best_t:
            best, best_t, best_pos = (f, s), t, pnl
    oos_t = _tstat(best_pos[cut:])
    cand = best_t >= IS_T_MIN and oos_t >= OOS_T_MIN
    return {"candidate": bool(cand), "is_t": round(best_t, 3), "oos_t": round(oos_t, 3), "rule": f"EMA{best[0]}/{best[1]}"}


def dataset_fingerprint(days: int = DEFAULT_DAYS) -> str:
    h = hashlib.sha256()
    for f in (sg.make_null_frame(days, 0), sg.make_planted_frame(days, 0, 0.05)):
        h.update(np.ascontiguousarray(f["close"].to_numpy(dtype=np.float64)).tobytes())
    h.update(json.dumps({"grid": FAST_GRID, "is": IS_T_MIN, "oos": OOS_T_MIN, "frac": IS_FRACTION, "v": BENCH_VERSION}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def engine_version() -> str:
    """Identity of the benchmark chain's code: a change to this file changes the version, so an old result is never read as current."""
    src = Path(__file__).read_bytes()
    return f"bench{BENCH_VERSION}-{hashlib.sha256(src).hexdigest()[:10]}"


def run_benchmark(*, n_null: int = GATE_NULL_RUNS, n_planted: int = GATE_PLANT_RUNS, targets=(0.10, 0.20, 0.30), days: int = DEFAULT_DAYS, seed: int = 0,
                  progress=None) -> dict:
    """Run the benchmark. Fully determined by its arguments: the same call always returns the same numbers."""
    log = progress or (lambda m: None)
    hits = 0
    for i in range(n_null):
        s = seed + 1000 * i
        out = fast_chain({"NULL0": sg.make_null_frame(days, s + 1)}, s)
        hits += int(out["candidate"])
        log(f"null {i + 1}/{n_null}")
    planted, power = [], {}
    for tgt in targets:
        cal = sg.calibrate_planted(tgt, days=days, seed=seed)
        found = 0
        for i in range(n_planted):
            s = seed + 500_000 + 1000 * i
            out = fast_chain({"EDGE0": sg.make_planted_frame(days, s + 1, cal["b"])}, s)
            found += int(out["candidate"])
        planted.append({"target_r": tgt, "measured_r": cal["measured_r"], "b": cal["b"], "runs": n_planted, "found": found})
        power[f"{tgt:.2f}"] = found / max(1, n_planted)
        log(f"planted +{tgt:.2f}R done")
    rate = hits / max(1, n_null)
    upper = sg.clopper_pearson_upper(hits, n_null)
    strong = [v for k, v in power.items() if float(k) >= 0.2 - 1e-9]
    reasons = []
    if n_null < GATE_NULL_RUNS:
        reasons.append(f"only {n_null} null runs (need >= {GATE_NULL_RUNS})")
    if any(p["runs"] < GATE_PLANT_RUNS for p in planted if p["target_r"] >= 0.2 - 1e-9):
        reasons.append(f"fewer than {GATE_PLANT_RUNS} runs on a planted size")
    if rate >= MAX_FDR:
        reasons.append(f"false-discovery rate {rate:.1%} is not under {MAX_FDR:.0%}")
    if not strong or min(strong) < MIN_POWER:
        reasons.append("power on the +0.20R / +0.30R plants is under 50%")
    return {"kind": "statistical_chain_benchmark", "label": "Fast statistical-chain benchmark (not the production engine)", "version": engine_version(),
            "dataset_fingerprint": dataset_fingerprint(days), "seed": seed, "days": days, "created_at": time.time(), "n_null": n_null,
            "null_candidates": hits, "null_rate": rate, "null_rate_upper95": upper, "planted": planted, "power": power,
            "passed": not reasons, "reasons": reasons}


# ------------------------------------------------------------------------------------------------------------------ storage
def _dir() -> Path:
    return sg.report_path().parent


def benchmark_path() -> Path:
    return _dir() / "integrity_benchmark.json"


def save_benchmark(res: dict) -> Path:
    p = benchmark_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    tmp.replace(p)
    return p


def load_benchmark() -> dict | None:
    try:
        d = json.loads(benchmark_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def history_path() -> Path:
    return _dir() / "surrogate_history.jsonl"


def append_history(rep: dict, gate_open: bool) -> None:
    try:
        row = {"created_at": rep.get("created_at"), "code_hash": rep.get("code_hash"), "n_null": rep.get("n_null"), "null_rate": rep.get("null_rate"),
               "open": bool(gate_open)}
        with history_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass


def last_successful_validation() -> dict | None:
    try:
        rows = [json.loads(x) for x in history_path().read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, ValueError):
        return None
    ok = [r for r in rows if r.get("open")]
    return ok[-1] if ok else None


def progress_path() -> Path:
    return _dir() / "surrogate_progress.json"


def write_progress(done: int, total: int, running: bool = True) -> None:
    try:
        progress_path().write_text(json.dumps({"done": done, "total": total, "running": running, "updated_at": time.time()}), encoding="utf-8")
    except OSError:
        pass


def _progress() -> dict:
    try:
        d = json.loads(progress_path().read_text(encoding="utf-8"))
        if d.get("running") and time.time() - float(d.get("updated_at", 0)) > 1800:
            d["running"] = False            # a study that stopped reporting is not "running"
        return d
    except (OSError, ValueError):
        return {"done": 0, "total": 0, "running": False}


# ------------------------------------------------------------------------------------------------------------------ the dashboard
def dashboard() -> dict:
    """Everything on the Research Integrity page. Each block says whether it was measured, when, and by which code."""
    from app.discovery_engine import surrogate_schedule as sch
    gate = sg.gate_status()
    rep = gate.get("report")
    fp = sg.combined_false_positive()
    prog = _progress()
    bench = load_benchmark()
    cur_ver = engine_version()
    last = last_successful_validation()
    planted = []
    if rep:
        for p in rep.get("planted", []):
            planted.append({"target_r": p["target_r"], "measured_r": p.get("measured_r"), "found": p["found"], "runs": p["runs"],
                            "detection_rate": p["found"] / max(1, p["runs"])})
    strong = [v for k, v in ((rep or {}).get("power") or {}).items() if float(k) >= 0.2 - 1e-9]
    return {
        "gate": {"open": gate["open"], "reasons": gate["reasons"], "state": sch.study_status()},
        "null_test_progress": {"done": int(prog.get("done", 0)), "total": int(prog.get("total", 0)), "running": bool(prog.get("running")),
                               "completed_runs": int((rep or {}).get("n_null", 0)), "required_runs": GATE_NULL_RUNS},
        "false_positive": {"measured": fp["available"], "rate": fp["rate"], "upper95": fp["upper95"], "n_null": fp["n_null"], "fresh": fp["fresh"], "text": fp["text"],
                           "target": MAX_FDR},
        "detection": {"measured": bool(rep), "by_size": planted, "mean_power_strong": (float(np.mean(strong)) if strong else None), "target": MIN_POWER},
        "statistical_power": {"strong_edges": (float(np.mean(strong)) if strong else None), "per_size": (rep or {}).get("power", {}),
                              "note": "Share of planted +0.20R / +0.30R edges the full engine found. Below 50% means the lab cannot find a known edge."},
        "engine": {"code_hash": (rep or {}).get("code_hash"), "report_code_matches_current": bool(rep) and rep.get("code_hash") == sg.code_hash(),
                   "current_code_hash": sg.code_hash(), "benchmark_version": cur_ver},
        "dataset_fingerprint": dataset_fingerprint(),
        "last_successful_validation": last,
        "real_data_fdr": {"measured": False, "text": "Not measured. Real markets contain an unknown amount of genuine edge, so a false-discovery rate cannot be counted on them. "
                                                      "The figures above are measured on surrogate data with a known answer and say how the chain behaves when the truth is known."},
        "benchmark": None if bench is None else {**bench, "current": bench.get("version") == cur_ver and bench.get("dataset_fingerprint") == dataset_fingerprint(bench.get("days", DEFAULT_DAYS))},
        "report_created_at": (rep or {}).get("created_at"),
    }
