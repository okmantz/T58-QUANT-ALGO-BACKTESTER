"""
BREAK-IT BATTERY -- try to kill a hypothesis before the market does.

Every test asks "what would we see if there were NO edge?" and checks that the
result is not that:

  null_random_timing  Same rule, same stops/targets, same trade count and hold
                      structure, but the entries are circularly shifted by a
                      random offset (destroying their timing relative to
                      price). p = (1 + #null runs >= observed) / (N + 1) on
                      total R. A real edge beats the shifted versions.
  out_of_sample       Last 40% of history only (rule fixed, nothing refit).
                      HONEST CAVEAT: this is only out-of-sample if the rule was
                      not tuned on that segment -- the report says so.
  cost_stress         Spread/slippage/commission x cost_mult (default 1.5).
  regimes             Mean R by volatility tercile (ATR percentile at entry);
                      >= 2 of the regimes with enough trades must be positive.
  param_neighbors     Each numeric parameter moved +-25%; most neighbors must
                      stay profitable (an edge on a knife-edge is overfit).
  time_folds          Trades split into K consecutive folds (CPCV-lite);
                      most folds positive.
  cross_market        Same rule on other supplied datasets.

Overall: SURVIVED only if the three hard tests (null, out-of-sample, costs)
pass AND at least 60% of the remaining ones that could be run pass. Tests that
cannot run (too few trades, no other markets) are reported "not_run", never
silently passed. Surviving is evidence, not proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from app.discovery.rule_spec import SPEC_SCHEMA, SpecError, validate_spec
from app.discovery.runner import metrics, run_spec, trade_r


@dataclass
class TestResult:
    name: str
    status: str            # "pass" | "fail" | "not_run"
    hard: bool
    detail: str
    value: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class BreakReport:
    verdict: str           # "survived" | "broken" | "inconclusive"
    tests: list
    n_baseline_trades: int
    notes: list = field(default_factory=list)

    def render(self) -> str:
        out = [f"BREAK-IT BATTERY: {self.verdict.upper()}  ({self.n_baseline_trades} baseline trades)"]
        for t in self.tests:
            out.append(f"  [{t.status.upper():7}] {t.name}{' (hard)' if t.hard else ''}: {t.detail}")
        out += [f"  NOTE: {n}" for n in self.notes]
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "n_baseline_trades": self.n_baseline_trades,
                "tests": [t.to_dict() for t in self.tests], "notes": list(self.notes)}


def _shifted_plan_trades(df, spec, risk, offset: int):
    """Run the rule with its ENTRY TIMING circularly shifted by `offset` bars."""
    from app.backtest.execution import run_execution
    from app.backtest.resting_orders import LimitOrder, simulate_resting_orders
    from app.discovery.rule_spec import build_plan
    import warnings
    plan = build_plan(df, spec)
    n = len(df)
    if plan.mode == "resting":
        c = df["close"].to_numpy(float)
        moved = []
        for od in plan.orders:
            nb = (od.created_bar + offset) % (n - 2)
            if nb < 1:
                continue
            rel = lambda x: x - c[od.created_bar]  # noqa: E731
            moved.append(LimitOrder(od.side, c[nb] + rel(od.price), nb, nb + (od.expire_bar - od.created_bar),
                                    c[nb] + rel(od.stop_price),
                                    None if od.target_price is None else c[nb] + rel(od.target_price), od.tag))
        return simulate_resting_orders(df, moved, risk)
    sig = np.roll(plan.signals.to_numpy(), offset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trades, _ = run_execution(
            df, pd.Series(sig, index=df.index), risk, stop_loss_pips=None, take_profit_pips=None,
            stop_loss_distance=plan.stop_loss_distance, take_profit_distance=plan.take_profit_distance,
        )
    return trades


def _atr_pct(df: pd.DataFrame) -> pd.Series:
    c = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c).abs(), (df["low"] - c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    return atr.rank(pct=True)


def run_break_battery(
    df: pd.DataFrame,
    spec: dict,
    risk,
    *,
    other_markets: dict | None = None,
    n_null: int = 100,
    cost_mult: float = 1.5,
    oos_fraction: float = 0.4,
    min_trades: int = 20,
    seed: int = 7,
) -> BreakReport:
    spec = validate_spec(spec)
    rng = np.random.default_rng(seed)
    tests: list[TestResult] = []
    notes: list[str] = []

    base = run_spec(df, spec, risk)
    bm = metrics(base)
    if bm.n < min_trades:
        return BreakReport("inconclusive", [TestResult("baseline", "not_run", True,
                           f"only {bm.n} trades (< {min_trades}); nothing can be concluded")], bm.n,
                           ["Too few trades to test. Use more history, a faster timeframe, or loosen the rule."])
    tests.append(TestResult("baseline", "pass" if bm.mean_r > 0 else "fail", True,
                            f"{bm.n} trades, mean {bm.mean_r:+.3f}R, PF {bm.profit_factor:.2f}, net ${bm.net:,.0f}",
                            bm.to_dict()))

    # 1 null: random timing -------------------------------------------------
    nulls = []
    n = len(df)
    for _ in range(n_null):
        off = int(rng.integers(max(50, n // 20), n - max(50, n // 20)))
        tr = _shifted_plan_trades(df, spec, risk, off)
        nulls.append(metrics(tr).total_r)
    nulls = np.array(nulls)
    p = float((1 + (nulls >= bm.total_r).sum()) / (len(nulls) + 1))
    tests.append(TestResult("null_random_timing", "pass" if p < 0.05 and bm.total_r > 0 else "fail", True,
                            f"observed total {bm.total_r:+.1f}R vs null median {np.median(nulls):+.1f}R "
                            f"(95th pct {np.percentile(nulls, 95):+.1f}R); p = {p:.3f} over {len(nulls)} shifted runs",
                            {"p_value": p, "observed_total_r": bm.total_r, "null_median": float(np.median(nulls)),
                             "null_p95": float(np.percentile(nulls, 95))}))

    # 2 OOS ----------------------------------------------------------------
    ts = pd.to_datetime(df["timestamp"])
    cut = ts.iloc[int(len(ts) * (1 - oos_fraction))]
    oos = [t for t in base if pd.Timestamp(t.entry_time) >= cut]
    ins = [t for t in base if pd.Timestamp(t.entry_time) < cut]
    om, im = metrics(oos), metrics(ins)
    if om.n < max(10, min_trades // 2):
        tests.append(TestResult("out_of_sample", "not_run", True, f"only {om.n} trades in the held-out segment"))
    else:
        ok = om.mean_r > 0 and om.mean_r >= 0.3 * max(im.mean_r, 0)
        tests.append(TestResult("out_of_sample", "pass" if ok else "fail", True,
                                f"in-sample {im.mean_r:+.3f}R ({im.n}) -> held-out {om.mean_r:+.3f}R ({om.n}); "
                                "rule fixed, nothing refit (only truly out-of-sample if you did not tune on this segment)",
                                {"is": im.to_dict(), "oos": om.to_dict()}))

    # 3 costs --------------------------------------------------------------
    stressed = replace(risk, spread_pips=risk.spread_pips * cost_mult, slippage_pips=risk.slippage_pips * cost_mult,
                       commission_per_contract=risk.commission_per_contract * cost_mult,
                       commission_per_trade=risk.commission_per_trade * cost_mult)
    sm = metrics(run_spec(df, spec, stressed))
    tests.append(TestResult("cost_stress", "pass" if sm.mean_r > 0 else "fail", True,
                            f"at {cost_mult:g}x costs: mean {sm.mean_r:+.3f}R, net ${sm.net:,.0f}", sm.to_dict()))

    # 4 regimes ------------------------------------------------------------
    pct = _atr_pct(df).to_numpy()
    ts_idx = {pd.Timestamp(t): i for i, t in enumerate(ts)}
    buckets = {"low": [], "mid": [], "high": []}
    for t in base:
        i = ts_idx.get(pd.Timestamp(t.entry_time))
        r = trade_r(t)
        if i is None or r is None or not np.isfinite(pct[i]):
            continue
        buckets["low" if pct[i] < 1 / 3 else ("mid" if pct[i] < 2 / 3 else "high")].append(r)
    valid = {k: v for k, v in buckets.items() if len(v) >= 8}
    if len(valid) < 2:
        tests.append(TestResult("regimes", "not_run", False, "fewer than 2 volatility regimes with >= 8 trades"))
    else:
        pos = sum(1 for v in valid.values() if np.mean(v) > 0)
        tests.append(TestResult("regimes", "pass" if pos >= 2 or pos == len(valid) else "fail", False,
                                ", ".join(f"{k}: {np.mean(v):+.2f}R ({len(v)})" for k, v in valid.items()),
                                {k: float(np.mean(v)) for k, v in valid.items()}))

    # 5 parameter neighbors --------------------------------------------------
    neighbors = []
    for name, (d, lo, hi, is_int) in SPEC_SCHEMA[spec["kind"]].items():
        for mult in (0.75, 1.25):
            v = spec["params"][name] * mult
            v = min(max(v, lo), hi)
            v = int(round(v)) if is_int else v
            if v == spec["params"][name]:
                continue
            try:
                s2 = validate_spec({**spec, "params": {**spec["params"], name: v}})
            except SpecError:
                continue
            neighbors.append((name, mult, metrics(run_spec(df, s2, risk)).mean_r))
    if len(neighbors) < 4:
        tests.append(TestResult("param_neighbors", "not_run", False, "too few valid neighbors"))
    else:
        share = sum(1 for _, _, r in neighbors if r > 0) / len(neighbors)
        tests.append(TestResult("param_neighbors", "pass" if share >= 0.6 else "fail", False,
                                f"{share:.0%} of {len(neighbors)} +-25% neighbors stay profitable",
                                {"share_positive": share}))

    # 6 time folds ---------------------------------------------------------
    K = 6
    if len(base) >= K * 5:
        chunks = np.array_split(np.arange(len(base)), K)
        fm = [metrics([base[i] for i in c]).mean_r for c in chunks]
        npos = sum(1 for x in fm if x > 0)
        tests.append(TestResult("time_folds", "pass" if npos >= 4 else "fail", False,
                                f"{npos}/{K} consecutive folds profitable ({', '.join(f'{x:+.2f}' for x in fm)})",
                                {"fold_mean_r": fm}))
    else:
        tests.append(TestResult("time_folds", "not_run", False, f"need >= {K * 5} trades"))

    # 7 cross-market -------------------------------------------------------
    if other_markets:
        res = {}
        for name, d2 in other_markets.items():
            m2 = metrics(run_spec(d2, spec, risk))
            if m2.n >= 10:
                res[name] = m2.mean_r
        if len(res) == 0:
            tests.append(TestResult("cross_market", "not_run", False, "no other market produced >= 10 trades"))
        else:
            share = sum(1 for v in res.values() if v > 0) / len(res)
            tests.append(TestResult("cross_market", "pass" if share >= 0.5 else "fail", False,
                                    ", ".join(f"{k}: {v:+.2f}R" for k, v in res.items()), res))
    else:
        tests.append(TestResult("cross_market", "not_run", False, "no other markets supplied"))

    # verdict ------------------------------------------------------------------
    hard = [t for t in tests if t.hard and t.name != "baseline"]
    soft = [t for t in tests if not t.hard and t.status != "not_run"]
    hard_fail = [t for t in hard if t.status == "fail"]
    hard_not_run = [t for t in hard if t.status == "not_run"]
    soft_pass = sum(1 for t in soft if t.status == "pass")
    if hard_fail or tests[0].status == "fail":
        verdict = "broken"
        notes.append("Failed: " + ", ".join(t.name for t in hard_fail + ([tests[0]] if tests[0].status == "fail" else [])))
    elif hard_not_run:
        verdict = "inconclusive"
        notes.append("Could not run: " + ", ".join(t.name for t in hard_not_run))
    elif soft and soft_pass / len(soft) < 0.6:
        verdict = "broken"
        notes.append("Passed the hard tests but only %d/%d robustness checks." % (soft_pass, len(soft)))
    else:
        verdict = "survived"
        notes.append("Survived is evidence, not proof: it was tested with %d null draws on this data set." % n_null)
    return BreakReport(verdict, tests, bm.n, notes)
