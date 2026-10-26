"""End-to-end surrogate tests of the whole lab, and the gate that keeps the new family batches switched off until they pass.

Why: every statistic in the promotion path (Deflated Sharpe, null gate, pooled test, sealed slice) is only as good as the way they
are chained. The only honest check of the CHAIN is to feed it data where the truth is known:

  * NO-EDGE surrogates     pure random-walk bars. The share of runs that nevertheless produce a forward-test candidate is the lab's
                           empirical false-discovery rate (the report's target: < 5%).
  * PLANTED-EDGE surrogates  random-walk bars with a trend-following edge of a known size in R (+0.10 .. +0.30R) planted by
                           construction and CALIBRATED by simulation (the planted rule's measured mean R, not a nominal drift). The
                           share of runs that find it is the lab's power.

``run_surrogate_study`` takes any ``pipeline(df, frames, seed) -> {"candidate": bool, ...}``; ``engine_pipeline`` is the real one
(DiscoveryEngine on a family subset, full promotion sequence). The result is a ``SurrogateReport`` saved to ``reports/surrogate_gate.json``
together with a hash of the code that produced it: ``enabled_batches()`` (which ``strategy_space.active_family_names()`` consults)
returns ``[]`` unless a report exists, was produced by the CURRENT code, shows a false-discovery rate under 5% over >= 20 null runs and
shows the lab can find a planted edge. "Run after every major change" is therefore enforced, not requested: change the engine and the gate
closes until the study is re-run.

Honest limits: 20 null runs with 0 hits only bounds the false-discovery rate below ~14% at 95% confidence (Clopper-Pearson, reported in
the file); a gate that proves < 5% needs ~60 clean runs. ``min_null_runs`` is configurable for that reason.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GATE_SCHEMA = 1
MAX_REPORT_AGE_DAYS = 90.0
CODE_GLOBS = ("app/discovery_engine/*.py", "app/search/families_v13.py", "app/search/batch_runner.py", "app/evolution/engine.py",
              "app/evolution/edge_fitness.py", "app/validation/trial_ledger.py", "app/validation/pooled_edge.py",
              "app/validation/null_edge_gate.py", "app/strategy/indicators_v13*.py")
_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------------------------------------- surrogates
def _bars(log_ret: np.ndarray, rng, start_price: float, days: int, bars_per_day: int, sigma: float) -> pd.DataFrame:
    idx = []
    for d in pd.bdate_range("2023-01-02", periods=days):
        idx.extend(pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=bars_per_day, freq="5min"))
    n = len(idx)
    close = start_price * np.exp(np.cumsum(log_ret[:n]))
    op = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0.5 * sigma, 0.2 * sigma, n)) * close
    return pd.DataFrame({"timestamp": idx, "open": op, "high": np.maximum(op, close) + spread, "low": np.minimum(op, close) - spread,
                         "close": close, "volume": np.abs(rng.normal(1000, 100, n)).round()})


def make_null_frame(days: int = 80, seed: int = 0, bars_per_day: int = 78, sigma: float = 0.0006) -> pd.DataFrame:
    """Random-walk bars: by construction NO strategy has an edge on them (costs make every rule a small loser)."""
    rng = np.random.default_rng(seed)
    n = days * bars_per_day
    return _bars(rng.normal(0.0, sigma, n), rng, 5000.0, days, bars_per_day, sigma)


def make_planted_frame(days: int = 80, seed: int = 0, b: float = 0.05, bars_per_day: int = 78, sigma: float = 0.0006) -> pd.DataFrame:
    """Random walk whose next-bar drift is ``b`` sigma in the direction of the EMA12-vs-EMA48 state (computed from PAST bars only):
    a trend-following edge by construction. ``b`` is converted to a size in R by ``calibrate_planted``."""
    rng = np.random.default_rng(seed)
    n = days * bars_per_day
    eps = rng.normal(0.0, sigma, n)
    r = np.empty(n)
    a12, a48 = 2 / 13, 2 / 49
    price, e12, e48 = 0.0, 0.0, 0.0
    for i in range(n):
        state = 0.0 if i < 60 else (1.0 if e12 > e48 else -1.0)
        r[i] = eps[i] + b * sigma * state
        price += r[i]
        e12 += a12 * (price - e12)
        e48 += a48 * (price - e48)
    return _bars(r, rng, 5000.0, days, bars_per_day, sigma)


def planted_config(stop_atr: float = 2.0, target_atr: float = 4.0) -> dict:
    """The rule the edge is planted in (a plain EMA12/48 trend rule with ATR exits)."""
    from app.search.strategy_space import _cond, _ind, _risk_management
    return {"name": "planted EMA12/48 trend", "entry_conditions": {
        "long": [_cond(_ind("ema", 12), ">", _ind("ema", 48))], "long_connectors": [],
        "short": [_cond(_ind("ema", 12), "<", _ind("ema", 48))], "short_connectors": []},
        "risk_management": _risk_management(stop_atr, target_atr, max_bars_in_trade=96)}


def measure_planted_mean_r(df: pd.DataFrame, risk=None) -> tuple[float, int]:
    """Mean R and trade count of the planted rule on ``df`` (frictionless: the planted edge is gross, costs are the lab's problem)."""
    from app.backtest.engine import run_backtest
    from app.backtest.risk import RiskConfig
    from app.discovery_engine.objective import score_edge
    from app.search.strategy_space import build_strategy_from_spec
    risk = risk or RiskConfig(spread_pips=0.0, slippage_pips=0.0, commission_per_trade=0.0)
    trades = list(run_backtest(df, build_strategy_from_spec({"source_type": "manual", "config": planted_config()}), risk, lookahead_check="skip").trades)
    sc = score_edge(trades, None)
    return float(sc.mean_r), int(sc.n_trades)


def calibrate_planted(target_r: float, days: int = 80, seed: int = 0, iters: int = 7) -> dict:
    """Find the drift ``b`` for which the planted rule's measured mean R is ``target_r`` (bisection on a calibration seed that is NOT
    used for evaluation). Returns {"b", "measured_r", "n_trades"}."""
    lo, hi = 0.0, 0.6
    best = (0.0, 0.0, 0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        m, n = measure_planted_mean_r(make_planted_frame(days, seed + 10_000, mid))
        best = (mid, m, n)
        if m < target_r:
            lo = mid
        else:
            hi = mid
    return {"b": float(best[0]), "measured_r": float(best[1]), "n_trades": int(best[2])}


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(conf, k + 1, n - k))


# -------------------------------------------------------------------------------------------------------------- the study
@dataclass
class SurrogateReport:
    created_at: float
    code_hash: str
    n_null: int
    null_candidates: int
    null_rate: float
    null_rate_upper95: float
    planted: list = field(default_factory=list)        # [{"target_r", "measured_r", "b", "runs", "found"}]
    power: dict = field(default_factory=dict)          # {target_r: share found}
    runs: list = field(default_factory=list)
    min_null_runs: int = 20
    schema: int = GATE_SCHEMA
    opened_batches: list = field(default_factory=lambda: [1])
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def code_hash(root: Path | None = None) -> str:
    root = Path(root) if root else _ROOT
    h = hashlib.sha256()
    files = sorted({p for g in CODE_GLOBS for p in root.glob(g) if p.is_file() and p.name != "__pycache__"})
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:20]


def run_surrogate_study(pipeline, *, n_null: int = 20, planted_targets=(0.10, 0.20, 0.30), n_planted: int = 3, n_markets: int = 8, days: int = 80,
                        seed: int = 0, min_null_runs: int = 20, progress=None) -> SurrogateReport:
    """Run ``pipeline(df, frames, seed)`` on ``n_null`` no-edge surrogates and ``n_planted`` planted surrogates per target size.
    ``frames`` are ``n_markets`` independent surrogates of the same kind (what the pooled cross-market test consumes)."""
    log = progress or (lambda m: None)
    runs, null_hits = [], 0
    for i in range(n_null):
        s = seed + 1000 * i
        frames = {f"NULL{j}": make_null_frame(days, s + j + 1) for j in range(n_markets)}
        out = pipeline(frames["NULL0"], frames, s) or {}
        hit = bool(out.get("candidate"))
        null_hits += int(hit)
        runs.append({"kind": "null", "seed": s, "candidate": hit, **{k: v for k, v in out.items() if k != "candidate"}})
        log(f"null {i + 1}/{n_null}: {'FORWARD-TEST CANDIDATE (false discovery)' if hit else 'none'}")
    planted, power = [], {}
    for tgt in planted_targets:
        cal = calibrate_planted(tgt, days=days, seed=seed)
        found = 0
        for i in range(n_planted):
            s = seed + 500_000 + 1000 * i
            frames = {f"EDGE{j}": make_planted_frame(days, s + j + 1, cal["b"]) for j in range(n_markets)}
            out = pipeline(frames["EDGE0"], frames, s) or {}
            hit = bool(out.get("candidate"))
            found += int(hit)
            runs.append({"kind": "planted", "target_r": tgt, "seed": s, "candidate": hit, **{k: v for k, v in out.items() if k != "candidate"}})
            log(f"planted +{tgt:.2f}R {i + 1}/{n_planted}: {'found' if hit else 'missed'}")
        planted.append({"target_r": tgt, "measured_r": cal["measured_r"], "b": cal["b"], "runs": n_planted, "found": found})
        power[f"{tgt:.2f}"] = found / max(1, n_planted)
    rep = SurrogateReport(created_at=time.time(), code_hash=code_hash(), n_null=n_null, null_candidates=null_hits,
                          null_rate=null_hits / max(1, n_null), null_rate_upper95=clopper_pearson_upper(null_hits, n_null),
                          planted=planted, power=power, runs=runs, min_null_runs=min_null_runs)
    if n_null < 60:
        rep.notes.append(f"{n_null} null runs bound the false-discovery rate below {rep.null_rate_upper95:.1%} at 95% confidence only; ~60 clean runs are needed to prove < 5%.")
    return rep


def engine_pipeline(families, *, preset: str = "search_lab", null_seeds: int = 40, max_candidates: int = 120, work_root=None):
    """The REAL chain as a pipeline callable: DiscoveryEngine on ``families`` with the full mandatory promotion sequence
    (smaller null-gate seed count than production, so a study finishes in minutes; pass ``null_seeds=200`` for the real thing)."""
    import tempfile

    from app.backtest.risk import RiskConfig
    from app.discovery_engine.engine import DiscoveryEngine, EngineRequest
    from app.discovery_engine.promotion import PromotionConfig
    from app.prop.simulator import PropRules

    def run(df, frames, seed):
        risk = RiskConfig(spread_pips=0.0, slippage_pips=0.0, commission_per_trade=0.0)
        wd = Path(work_root or tempfile.mkdtemp(prefix="t58_surrogate_")) / f"run_{seed}"
        req = EngineRequest(preset=preset, df=df, risk=risk, prop_rules=PropRules(), families=list(families), seed=int(seed),
                            frames=frames, risks={k: risk for k in frames}, work_dir=wd, max_candidates=max_candidates, register_retests=False, promotion=PromotionConfig(null_seeds=null_seeds),
                            overrides={"min_trades": 100, "workers": 1})
        res = DiscoveryEngine().run(req)
        return {"candidate": bool(res.forward_test_candidates), "n_champions": len(res.champions),
                "failed_steps": [p.get("failed_step") for p in res.promotions]}
    return run


# ------------------------------------------------------------------------------------------------------------------ the gate
def report_path() -> Path:
    from app.validation import trial_ledger
    return Path(trial_ledger._PATH).parent / "surrogate_gate.json"  # noqa: SLF001


def save_report(rep: SurrogateReport, path: Path | None = None) -> Path:
    p = Path(path) if path else report_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rep.to_dict(), indent=1, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def load_report(path: Path | None = None) -> dict | None:
    try:
        d = json.loads(Path(path or report_path()).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def gate_status(path: Path | None = None, *, now: float | None = None, current_hash: str | None = None,
                max_null_rate: float = 0.05, min_power: float = 0.5) -> dict:
    """{"open": bool, "reasons": [...], "report": dict|None}. Closed unless a fresh report from THIS code shows a false-discovery rate
    below ``max_null_rate`` over >= ``min_null_runs`` null runs AND power >= ``min_power`` on the +0.20R and +0.30R plants."""
    rep = load_report(path)
    reasons = []
    if rep is None:
        return {"open": False, "reasons": ["no surrogate-study report on disk: run scripts/run_surrogate_study.py"], "report": None}
    if rep.get("schema") != GATE_SCHEMA:
        reasons.append("report schema is out of date")
    now = time.time() if now is None else now
    if (now - float(rep.get("created_at", 0))) / 86400.0 > MAX_REPORT_AGE_DAYS:
        reasons.append(f"report is older than {MAX_REPORT_AGE_DAYS:.0f} days")
    if rep.get("code_hash") != (current_hash or code_hash()):
        reasons.append("the engine code changed since the study ran: re-run it (report item 8: 'run after every major change')")
    n_null = int(rep.get("n_null", 0))
    if n_null < int(rep.get("min_null_runs", 20)):
        reasons.append(f"only {n_null} null runs (need >= {rep.get('min_null_runs', 20)})")
    if float(rep.get("null_rate", 1.0)) >= max_null_rate:
        reasons.append(f"false-discovery rate {float(rep.get('null_rate', 1.0)):.1%} is not under {max_null_rate:.0%}")
    strong = [v for k, v in (rep.get("power") or {}).items() if float(k) >= 0.2 - 1e-9]
    if not strong or float(np.mean(strong)) < min_power:
        reasons.append(f"power on planted +0.20..0.30R edges is {float(np.mean(strong)) if strong else 0.0:.0%} (< {min_power:.0%}): the lab cannot find a known edge")
    return {"open": not reasons, "reasons": reasons, "report": rep}


def enabled_batches(path: Path | None = None) -> list:
    """Rollout batches of the v13 families that 'all families' searches may run (empty while the gate is closed)."""
    st = gate_status(path)
    if not st["open"]:
        return []
    return sorted({int(b) for b in (st["report"].get("opened_batches") or [1])})


def new_batches_enabled(path: Path | None = None) -> bool:
    return bool(enabled_batches(path))
