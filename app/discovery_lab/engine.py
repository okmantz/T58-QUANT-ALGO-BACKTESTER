"""The Discovery Lab search (v2, 2026-10-28).

One engine that combines what used to be separate tools -- broad random search, evolution, iterative refinement, robustness checks and
a final proof in the real backtester -- and treats every step as a way to stop a lucky backtest being called a discovery:

  0. DATA          the WHOLE history, at one or more bar sizes where trading costs are small next to a normal stop (not a recent slice).
  1. SEARCH        random sampling over 30 strategy families x their settings x regime filters (side / trend / session / volatility) x exits
                   (stop, target, time limit). Families are drawn from the research papers in the Library.
  2. EVOLVE        the best ideas breed (mutation + crossover) for several generations, always scored on the TRAIN slice only.
  3. REFINE        the top ideas are polished by hill-climbing one setting at a time (iterative refinement).
  4. VALIDATE      the best train candidates must hold up on the next slice of time they never trained on.
  5. ROBUSTNESS    neighbouring settings must mostly be profitable too: a real edge is a plateau, not a spike.
  6. SEALED TEST   the last 20% of history, read once per finalist, never used to choose anything. Must be profitable and survive DOUBLE
                   costs.
  7. PROOF         each finalist is written out exactly as it will be saved, loaded by the real backtester, run, and put through the same
                   lookahead test Full Pipeline uses. Anything that peeks at the future, or that the real engine does not reproduce, is
                   rejected here instead of failing later.
  8. ELIGIBILITY   every survivor gets an estimated chance of passing the chosen prop-firm evaluation at the best risk setting.

``winners`` passed every step; ``near_misses`` passed train+validate and show which step they failed. Neither is a guarantee -- the result
carries the number of ideas tried and an honest luck estimate, and each winner a confidence label deflated for how many ideas were tried.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import families as F

# Minimum number of ideas simulated by each effort level (Quick 5,000+, Standard 10,000+, Deep 15,000+) (random search + evolution + refinement; robustness checks come on top).
EFFORT_BUDGET = {"quick": 5200, "standard": 10400, "deep": 15600}
WARMUP = 250
MAX_HOLD = 400
CURVE_POINTS = 90
TARGETS = [0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 99.0]       # 99 = no profit target (exit on stop or time limit)
HOLDS = [6, 12, 24, 48, 120, 400]                      # bars before a trade is closed at the market
STOPS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
BAR_LADDER = (5, 15, 30, 60, 240)
SAMPLE_SHARE, EVOLVE_SHARE, REFINE_SHARE = 0.55, 0.38, 0.07
POOL_SHARE = 0.22                                      # share of random samples spent on the any-indicator family


@dataclass
class LabConfig:
    effort: str = "standard"
    seed: int = 7
    risk_pct: float = 0.5
    balance: float = 100_000.0
    max_dd_limit_pct: float = 10.0
    profit_target_pct: float = 8.0
    trailing_dd: bool = True
    families: Optional[list] = None
    train_frac: float = 0.60
    val_frac: float = 0.20
    min_trades: int = 40
    top_validate: int = 150
    max_finalists: int = 30
    target_cost_r: float = 0.12
    budget: Optional[int] = None            # override the effort's idea count (tests)
    prove_in_engine: bool = True            # run finalists through the real backtester + lookahead test
    # Searched-variant memory (app.search.variant_ledger). None = off (tests, one-off calls). With a ledger the Lab (1) skips variants any
    # earlier run on this market already tested, (2) varies its random seed per run so it does not walk the same path, (3) breeds new
    # variants from the best past ones (warm start) and (4) spends more random samples on the strategy types that have produced the most.
    ledger: object = None
    ledger_scope: str = ""                  # the market being searched, e.g. "ES1!"
    vary_seed: bool = True
    warm_start: bool = True
    seed_candidates: list = field(default_factory=list)   # starting ideas from papers / AI (app.discovery_lab.research_seeds); evaluated like any other idea


@dataclass
class LabResult:
    winners: list
    near_misses: list
    tested: int
    train_pass: int
    val_pass: int
    robust_pass: int
    families_tried: int
    cost_r_typical: float
    bars: int
    expected_by_luck: int = 0
    notes: list = field(default_factory=list)
    elapsed: float = 0.0
    bounds: tuple = (0, 0, 0)
    bar_sizes: list = field(default_factory=list)
    proven: int = 0
    ledger_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"winners": self.winners, "near_misses": self.near_misses, "tested": self.tested, "train_pass": self.train_pass,
                "val_pass": self.val_pass, "robust_pass": self.robust_pass, "expected_by_luck": self.expected_by_luck, "families_tried": self.families_tried,
                "cost_r_typical": self.cost_r_typical, "bars": self.bars, "notes": self.notes, "elapsed": round(self.elapsed, 1),
                "bounds": list(self.bounds), "bar_sizes": self.bar_sizes, "proven": self.proven, "ledger": self.ledger_info}


# ---------------------------------------------------------------------------------------------------------------- simulation core
def _sim_core(sig_idx, sig_dir, o, h, l, c, atr, stop_atr, target_r, max_hold, cost_price, cost_mult, warm):
    """Non-overlapping trades. Entry = next bar's open; the stop is checked before the target inside a bar (pessimistic); a gap through a
    level fills at the open; a trade still open after ``max_hold`` bars closes at that bar's close."""
    n = len(o)
    m = len(sig_idx)
    ent = np.empty(m, np.int64)
    ex = np.empty(m, np.int64)
    rr = np.empty(m, np.float64)
    k = 0
    free = 0
    for q in range(m):
        i = sig_idx[q]
        if i < free or i < warm or i >= n - 2:
            continue
        a = atr[i]
        if not (a > 0.0):
            continue
        d = sig_dir[q]
        e = o[i + 1]
        sd = stop_atr * a
        if d == 1:
            stop = e - sd
            tgt = e + target_r * sd
        else:
            stop = e + sd
            tgt = e - target_r * sd
        jend = i + max_hold
        if jend > n - 1:
            jend = n - 1
        exit_j = jend
        r = 0.0
        resolved = False
        for j in range(i + 1, jend + 1):
            if d == 1:
                if o[j] <= stop:
                    exit_j, r, resolved = j, (o[j] - e) / sd, True
                    break
                if l[j] <= stop:
                    exit_j, r, resolved = j, -1.0, True
                    break
                if o[j] >= tgt:
                    exit_j, r, resolved = j, (o[j] - e) / sd, True
                    break
                if h[j] >= tgt:
                    exit_j, r, resolved = j, target_r, True
                    break
            else:
                if o[j] >= stop:
                    exit_j, r, resolved = j, (e - o[j]) / sd, True
                    break
                if h[j] >= stop:
                    exit_j, r, resolved = j, -1.0, True
                    break
                if o[j] <= tgt:
                    exit_j, r, resolved = j, (e - o[j]) / sd, True
                    break
                if l[j] <= tgt:
                    exit_j, r, resolved = j, target_r, True
                    break
        if not resolved:
            r = d * (c[exit_j] - e) / sd
        r -= cost_mult * cost_price / sd
        ent[k] = i + 1
        ex[k] = exit_j
        rr[k] = r
        k += 1
        free = exit_j + 1
    return ent[:k], ex[:k], rr[:k]


try:  # numba makes the search ~30x faster; the plain-Python loop above is the always-available fallback
    from numba import njit as _njit

    _sim_fast = _njit(cache=False, nogil=True)(_sim_core)
except Exception:  # noqa: BLE001
    _sim_fast = _sim_core


def simulate(lg: np.ndarray, sh: np.ndarray, I: F.Indicators, stop_atr: float, target_r: float, cost_price: float, cost_mult: float = 1.0,
             max_hold: int = MAX_HOLD):
    """Returns (entry_idx, exit_idx, R_after_costs) for a long/short signal pair. Also used by the saved strategy files."""
    both = lg & sh
    sig = np.flatnonzero((lg | sh) & ~both)
    if len(sig) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    dirs = np.where(lg[sig], 1, -1).astype(np.int64)
    atr = np.nan_to_num(I.atr(14), nan=0.0)
    try:
        return _sim_fast(sig.astype(np.int64), dirs, I.o, I.h, I.l, I.c, atr, float(stop_atr), float(target_r), int(max_hold),
                         float(cost_price), float(cost_mult), WARMUP)
    except Exception:  # noqa: BLE001 -- if the compiled path ever fails, the plain loop gives the same answer
        return _sim_core(sig.astype(np.int64), dirs, I.o, I.h, I.l, I.c, atr, float(stop_atr), float(target_r), int(max_hold),
                         float(cost_price), float(cost_mult), WARMUP)


def stats(r: np.ndarray) -> dict:
    n = len(r)
    if n == 0:
        return {"n": 0, "mean": 0.0, "t": 0.0, "pf": 0.0, "win": 0.0, "dd": 0.0, "net": 0.0}
    m = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    t = m / sd * math.sqrt(n) if sd > 0 else 0.0
    pos, neg = float(r[r > 0].sum()), float(-r[r < 0].sum())
    pf = pos / neg if neg > 0 else (9.99 if pos > 0 else 0.0)
    cum = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum).max())
    return {"n": n, "mean": round(m, 4), "t": round(t, 3), "pf": round(min(pf, 9.99), 3), "win": round(float((r > 0).mean()) * 100, 1),
            "dd": round(dd, 3), "net": round(float(cum[-1]), 3)}


def curve(exit_idx: np.ndarray, r: np.ndarray, n_bars: int, upto: Optional[int] = None) -> list:
    """Cumulative R sampled at CURVE_POINTS evenly spaced bar positions (flat between trades)."""
    end = n_bars if upto is None else upto
    xs = np.linspace(0, end - 1, CURVE_POINTS).astype(int)
    cum = np.concatenate([[0.0], np.cumsum(r)])
    k = np.searchsorted(exit_idx, xs, side="right")
    return [round(float(v), 3) for v in cum[k]]


def _slice(ent, ex, r, lo, hi):
    m = (ent >= lo) & (ent < hi)
    return r[m]


# ---------------------------------------------------------------------------------------------------------------- candidates
def label_of(c: dict) -> str:
    fam = F.FAMILIES[c["family"]]
    ptxt = " ".join(f"{k}={v}" for k, v in c["params"].items() if v not in (0, None))
    flt = c.get("flt") or {}
    ftxt = ", ".join(f"{k}={v}" for k, v in flt.items() if v not in (0, None, "both", "all"))
    tgt = "no target" if c["target_r"] >= 50 else f"target {c['target_r']:g}R"
    hold = c.get("hold", MAX_HOLD)
    return (f"{fam.label} ({ptxt})" + (f" [{ftxt}]" if ftxt else "") + f" | stop {c['stop_atr']:g} ATR, {tgt}"
            + (f", exit after {hold} bars" if hold < MAX_HOLD else ""))


def _fkey(c): return (c["family"], tuple(sorted(c["params"].items())), tuple(sorted((c.get("flt") or {}).items())))
def _key(c): return _fkey(c) + (c["stop_atr"], c["target_r"], c.get("hold", MAX_HOLD))


def _lkey(c: dict, minutes: int) -> str:
    """Process-stable key of one variant (bar size + signal set + exits) for the searched-variant ledger."""
    from app.search.variant_ledger import stable_key
    return stable_key("lab", int(minutes), c["family"], sorted(c["params"].items()), sorted((c.get("flt") or {}).items()),
                      c["stop_atr"], c["target_r"], c.get("hold", MAX_HOLD))


def _lsig(c: dict, minutes: int) -> str:
    from app.search.variant_ledger import stable_key
    return stable_key("labsig", int(minutes), c["family"], sorted(c["params"].items()), sorted((c.get("flt") or {}).items()))


_SPEC_KEYS = ("family", "params", "flt", "stop_atr", "target_r", "hold")


def _warm_candidates(cfg: "LabConfig", minutes: int, fams: list, limit: int = 24) -> list:
    """Past top variants on this market/bar size, as candidate dicts the evolution phase can breed from. Anything that no longer fits the
    current family grids is dropped (families get edited between releases)."""
    led = cfg.ledger
    if led is None or not cfg.warm_start or not cfg.ledger_scope:
        return []
    out = []
    for row in led.top_performers(cfg.ledger_scope, bar_minutes=minutes, families=fams, limit=limit):
        sp = row.get("spec") or {}
        try:
            fam = F.FAMILIES[sp["family"]]
            if set(sp["params"]) != set(fam.grid) or not _valid_params(sp["family"], sp["params"]):
                continue
            c = {k: sp[k] for k in _SPEC_KEYS if k in sp}
            c.setdefault("hold", MAX_HOLD)
            c.setdefault("flt", {})
            c["fit"] = float(row.get("fit") or 0.0)
            c["_warm"] = True
            out.append(c)
        except Exception:  # noqa: BLE001
            continue
    return out


def _valid_params(fn: str, p: dict) -> bool:
    if fn == "ema_cross":
        return p["fast"] < p["slow"]
    if fn == "macd":
        return p["a"] < p["b"]
    return True


class _Ctx:
    """One bar size's data, indicators, cost and the signal cache the search reuses across exits."""

    def __init__(self, df: pd.DataFrame, cost_price: float, cfg: LabConfig, minutes: int):
        self.df = df
        self.I = F.Indicators(df)
        self.n = self.I.n
        self.minutes = int(minutes)
        self.cost = float(cost_price)
        self.atr_med = float(np.nanmedian(self.I.atr(14)[WARMUP:]))
        self.a1 = int(self.n * cfg.train_frac)
        self.a2 = int(self.n * (cfg.train_frac + cfg.val_frac))
        self.sigcache: dict = {}
        self.days = float(max(1, len(np.unique(self.I.tday))))
        self.intraday = ("timestamp" in df.columns) and self.I.hour.size and len(np.unique(self.I.hour)) >= 3 and minutes <= 240
        min_stop = self.cost / (cfg.target_cost_r * self.atr_med) if self.atr_med > 0 else 1.0
        self.stops = [s for s in STOPS if s >= min_stop] or [min(8.0, math.ceil(min_stop * 2) / 2 + k) for k in (0, 1, 2)]
        self.min_stop = min_stop

    def signals(self, c: dict):
        k = _fkey(c)
        hit = self.sigcache.get(k)
        if hit is not None:
            return hit
        lg, sh = F.FAMILIES[c["family"]].build(self.I, c["params"])
        lg, sh = np.nan_to_num(lg, nan=False).astype(bool), np.nan_to_num(sh, nan=False).astype(bool)
        lg, sh = F.apply_filters(self.I, lg, sh, c.get("flt"))
        both = lg & sh
        sig = np.flatnonzero((lg | sh) & ~both)
        dirs = np.where(lg[sig], 1, -1).astype(np.int64)
        if len(self.sigcache) > 700:
            self.sigcache.clear()
        self.sigcache[k] = (sig.astype(np.int64), dirs)
        return self.sigcache[k]

    def run(self, c: dict, cost_mult: float = 1.0):
        sig, dirs = self.signals(c)
        if len(sig) < 12:
            return None
        I = self.I
        atr = np.nan_to_num(I.atr(14), nan=0.0)
        args = (sig, dirs, I.o, I.h, I.l, I.c, atr, float(c["stop_atr"]), float(c["target_r"]), int(c.get("hold", MAX_HOLD)),
                self.cost, float(cost_mult), WARMUP)
        try:
            return _sim_fast(*args)
        except Exception:  # noqa: BLE001
            return _sim_core(*args)


def _train_fitness(ent, r, a1, min_trades, folds=4):
    """t-statistic of the train-slice R, scaled by how many of ``folds`` equal time slices of the train span are profitable. A system that
    earned everything in one burst scores lower than one that earns steadily."""
    m = ent < a1
    tr = r[m]
    n = len(tr)
    if n < min_trades:
        return None
    mean = float(tr.mean())
    if mean <= 0.02:
        return None
    sd = float(tr.std(ddof=1))
    if sd <= 0:
        return None
    t = mean / sd * math.sqrt(n)
    edges = np.linspace(WARMUP, a1, folds + 1)
    pos = 0
    used = 0
    for k in range(folds):
        mk = (ent[m] >= edges[k]) & (ent[m] < edges[k + 1])
        if mk.sum() >= 4:
            used += 1
            pos += int(tr[mk].mean() > 0)
    cons = (pos / used) if used else 0.0
    pf_pos, pf_neg = float(tr[tr > 0].sum()), float(-tr[tr < 0].sum())
    pf = pf_pos / pf_neg if pf_neg > 0 else 9.99
    if t < 1.5 or pf < 1.08:
        return None
    return t * (0.4 + 0.6 * cons), {"n": n, "mean": round(mean, 4), "t": round(t, 3), "pf": round(min(pf, 9.99), 3), "cons": round(cons, 2)}


def _rand_flt(rng: random.Random, ctx: _Ctx) -> dict:
    if rng.random() < 0.35:
        return {}
    flt = {}
    if rng.random() < 0.55:
        flt["side"] = rng.choice(["long", "long", "short"]) if rng.random() < 0.8 else "both"
    if rng.random() < 0.5:
        flt["trend"] = rng.choice([50, 100, 200])
    if ctx.intraday and ctx.minutes <= 120 and rng.random() < 0.4:
        flt["hours"] = rng.choice([h for h in F.HOUR_WINDOWS if h != "all"])
    if rng.random() < 0.25:
        flt["vol"] = rng.choice(["high", "low"])
    return {k: v for k, v in flt.items() if v not in ("both", "all", 0)}


def _sample(rng: random.Random, fams: list, ctx: _Ctx) -> dict:
    # The indicator pool is ONE family with ~300 indicators x settings behind it; a uniform pick would give it 1/30 of the search, so it is
    # sampled on its own share (POOL_SHARE) and the hand-built families split the rest.
    fn = "ind_rule" if ("ind_rule" in fams and rng.random() < POOL_SHARE) else rng.choice(fams)
    fam = F.FAMILIES[fn]
    for _ in range(20):
        p = {k: rng.choice(v) for k, v in fam.grid.items()}
        if _valid_params(fn, p):
            break
    return {"family": fn, "params": p, "flt": _rand_flt(rng, ctx), "stop_atr": rng.choice(ctx.stops), "target_r": rng.choice(TARGETS),
            "hold": rng.choice(HOLDS)}


def _neighbours(c: dict, ctx: _Ctx) -> list:
    """Every candidate that differs from ``c`` by ONE step in ONE setting (a parameter, a filter, or an exit)."""
    out = []
    fam = F.FAMILIES[c["family"]]
    for k, vals in fam.grid.items():
        if c["params"][k] in vals and len(vals) > 1 and k not in fam.categorical:
            i = vals.index(c["params"][k])
            for j in (i - 1, i + 1):
                if 0 <= j < len(vals):
                    q = dict(c["params"])
                    q[k] = vals[j]
                    if _valid_params(c["family"], q):
                        out.append({**c, "params": q})
    for key, lst in (("stop_atr", ctx.stops), ("target_r", TARGETS), ("hold", HOLDS)):
        cur = c.get(key, MAX_HOLD if key == "hold" else None)
        if cur in lst:
            i = lst.index(cur)
            for j in (i - 1, i + 1):
                if 0 <= j < len(lst):
                    out.append({**c, key: lst[j]})
    return out


def _mutate(rng: random.Random, c: dict, ctx: _Ctx, fams: list) -> dict:
    ch = {**c, "params": dict(c["params"]), "flt": dict(c.get("flt") or {})}
    for _ in range(rng.choice([1, 1, 2, 3])):
        g = rng.random()
        fam = F.FAMILIES[ch["family"]]
        if g < 0.40 and fam.grid:
            k = rng.choice(list(fam.grid))
            vals = fam.grid[k]
            i = vals.index(ch["params"][k]) if ch["params"][k] in vals else 0
            j = min(len(vals) - 1, max(0, i + rng.choice([-2, -1, 1, 2]))) if (rng.random() < 0.8 and k not in fam.categorical) else rng.randrange(len(vals))
            ch["params"][k] = vals[j]
        elif g < 0.60:
            ch["flt"] = _rand_flt(rng, ctx) if rng.random() < 0.5 else {k: v for k, v in ch["flt"].items() if rng.random() < 0.7}
        elif g < 0.78:
            ch["stop_atr"] = rng.choice(ctx.stops)
        elif g < 0.90:
            ch["target_r"] = rng.choice(TARGETS)
        else:
            ch["hold"] = rng.choice(HOLDS)
    if not _valid_params(ch["family"], ch["params"]):
        return c
    return ch


def _cross(rng: random.Random, a: dict, b: dict) -> dict:
    if a["family"] == b["family"]:
        p = {k: (a["params"][k] if rng.random() < 0.5 else b["params"][k]) for k in a["params"]}
        child = {**a, "params": p}
        if not _valid_params(a["family"], p):
            child = {**a}
    else:
        child = {**a}
    child["flt"] = dict(a.get("flt") or {}) if rng.random() < 0.5 else dict(b.get("flt") or {})
    for key in ("stop_atr", "target_r", "hold"):
        child[key] = a.get(key) if rng.random() < 0.5 else b.get(key, MAX_HOLD)
    return child


# ---------------------------------------------------------------------------------------------------------------- prop-firm odds
def eval_pass_estimate(r: np.ndarray, risk_pct: float, target_pct: float, dd_pct: float, trailing: bool, sims: int = 1500, seed: int = 11,
                       max_trades: int = 400) -> float:
    """Chance (%) that a fresh account following this trade sequence at ``risk_pct`` per trade reaches ``target_pct`` before breaching the
    drawdown limit. Trades are resampled in short blocks so streaks survive; static or trailing drawdown as the firm defines it."""
    n = len(r)
    if n < 20:
        return 0.0
    rng = np.random.default_rng(seed)
    block = 5
    nb = max_trades // block + 1
    starts = rng.integers(0, max(1, n - block), size=(sims, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(sims, -1)[:, :max_trades]
    pnl = r[np.minimum(idx, n - 1)] * risk_pct
    eq = np.cumsum(pnl, axis=1)
    peak = np.maximum.accumulate(np.maximum(eq, 0.0), axis=1)
    dd = (peak - eq) if trailing else (-eq)
    hit_t = eq >= target_pct
    hit_d = dd >= dd_pct
    first_t = np.where(hit_t.any(1), hit_t.argmax(1), 10 ** 9)
    first_d = np.where(hit_d.any(1), hit_d.argmax(1), 10 ** 9)
    return round(float(np.mean(first_t < first_d)) * 100.0, 1)


def best_risk(r: np.ndarray, cfg: LabConfig, trades_per_day: float = 1.0, horizon_days: int = 60) -> tuple:
    """Risk per trade (and its pass chance) that maximises the odds of finishing the evaluation within ``horizon_days`` trading days."""
    best = (0.0, cfg.risk_pct)
    horizon = int(max(20, min(600, trades_per_day * horizon_days)))
    for rp in (0.25, 0.4, 0.5, 0.75, 1.0, 1.5):
        pp = eval_pass_estimate(r, rp, cfg.profit_target_pct, cfg.max_dd_limit_pct, cfg.trailing_dd, max_trades=horizon)
        if pp > best[0] + 0.05:
            best = (pp, rp)
    return best


# ---------------------------------------------------------------------------------------------------------------- the search
def _search_one_size(ctx: _Ctx, fams: list, cfg: LabConfig, budget: int, rng: random.Random, emit, cancel, tally: dict, notes: list):
    """Random search -> evolution -> refinement on ONE bar size, scored on the train slice only. Returns the archive of train-passing
    candidates (each with ``fit`` and the cached trade arrays)."""
    archive: dict = {}
    seen: set = set()
    n_sample = int(budget * SAMPLE_SHARE)
    n_evolve = int(budget * EVOLVE_SHARE)
    n_refine = budget - n_sample - n_evolve
    start_tested = tally["tested"]

    def evaluate(c: dict) -> Optional[dict]:
        k = _key(c)
        if k in seen:
            return None
        seen.add(k)
        led, scope = cfg.ledger, cfg.ledger_scope
        lk = None
        if led is not None and scope:
            lk = _lkey(c, ctx.minutes)
            if led.is_known(scope, lk):                 # tested in an earlier run: do not spend a simulation on it again
                led.skipped_known += 1
                tally["skipped_known"] = tally.get("skipped_known", 0) + 1
                return None
        tally["tested"] += 1
        tally["distinct"].add((ctx.minutes,) + _fkey(c))
        res = ctx.run(c)
        spec = {kk: c[kk] for kk in _SPEC_KEYS if kk in c}

        def _book(stage, fit_=None, st_=None):
            if lk is not None:
                led.record(scope, lk, source="lab", family=c["family"], bar_minutes=ctx.minutes, spec=spec, label=label_of(c), stage=stage,
                           fit=fit_, train_t=(st_ or {}).get("t"), train_n=(st_ or {}).get("n"), sig=_lsig(c, ctx.minutes))
        if res is None:
            _book("searched")
            return None
        ent, ex, r = res
        fit = _train_fitness(ent, r, ctx.a1, cfg.min_trades)
        if fit is None:
            _book("searched")
            return None
        score, st = fit
        dof = len(c["params"]) + len(c.get("flt") or {})
        score = score / (1.0 + 0.06 * dof)               # every extra knob is another way to fit noise
        _book("train_pass", score, st)
        c = {**c, "fit": score, "train": st, "_trades": (ent, ex, r), "bar_minutes": ctx.minutes}
        archive[k] = c
        tally["train_pass"] += 1
        return c

    def live():
        top = sorted(archive.values(), key=lambda x: -x["fit"])[:4]
        emit({"phase": "search", "tested": tally["tested"], "total": tally["total"], "train_pass": tally["train_pass"],
              "bounds": [ctx.a1, ctx.a2, ctx.n], "minutes": ctx.minutes,
              "best": [{"id": i, "label": label_of(c), "t": c["train"]["t"],
                        "curve": curve(c["_trades"][1][c["_trades"][0] < ctx.a1], c["_trades"][2][c["_trades"][0] < ctx.a1], ctx.n, upto=ctx.a1)}
                       for i, c in enumerate(top)]})

    def stopped() -> bool:
        return cancel is not None and cancel.is_set()

    # --- 0. what earlier runs on this market learned: which strategy types pay, and which past variants are worth breeding from
    sample_fams = fams
    warm = []
    if cfg.ledger is not None and cfg.ledger_scope:
        try:
            wts = cfg.ledger.family_weights(cfg.ledger_scope, fams, bar_minutes=ctx.minutes)
            if any(w > 1 for w in wts.values()):
                sample_fams = [f for f in fams for _ in range(wts.get(f, 1))]
            warm = _warm_candidates(cfg, ctx.minutes, fams)
            cfg.ledger.warm_started += len(warm)
        except Exception as exc:  # noqa: BLE001 -- the memory is a bonus, never a reason to fail a search
            notes.append(f"Search memory could not be read ({type(exc).__name__}); searching without it.")
    # --- 0b. starting ideas from the Resource Library papers and the AI: evaluated first, exactly like any random idea (they count as tried)
    for seed_c in (cfg.seed_candidates or []):
        if stopped():
            break
        sc = {k: v for k, v in seed_c.items() if k in ("family", "params", "flt", "stop_atr", "target_r", "hold")}
        if sc.get("stop_atr") is None:
            sc["stop_atr"] = ctx.stops[len(ctx.stops) // 2]
        if sc["family"] in F.FAMILIES:
            evaluate(sc)
    # --- 1. random search
    guard = 0
    while tally["tested"] - start_tested < n_sample and guard < n_sample * 6 and not stopped():
        guard += 1
        evaluate(_sample(rng, sample_fams if (sample_fams is not fams and rng.random() < 0.6) else fams, ctx))
        if guard % 160 == 0:
            live()
    # --- 2. evolution
    pop_size = 48
    gens = 0
    while tally["tested"] - start_tested < n_sample + n_evolve and not stopped():
        pool = sorted(list(archive.values()) + warm, key=lambda x: -x["fit"])[:pop_size]
        if len(pool) < 4:                              # nothing to breed from: keep sampling, but around the best-scoring families
            for _ in range(160):
                evaluate(_sample(rng, fams, ctx))
            guard += 1
            if guard > n_sample * 8:
                break
            continue
        gens += 1
        made = 0
        tries = 0
        while made < 160 and tries < 900 and tally["tested"] - start_tested < n_sample + n_evolve and not stopped():
            tries += 1
            a = max(rng.sample(pool, min(3, len(pool))), key=lambda x: x["fit"])
            if rng.random() < 0.3:
                b = max(rng.sample(pool, min(3, len(pool))), key=lambda x: x["fit"])
                child = _cross(rng, a, b)
            else:
                child = _mutate(rng, a, ctx, fams)
            before = tally["tested"]
            evaluate({k: v for k, v in child.items() if k in ("family", "params", "flt", "stop_atr", "target_r", "hold")})
            made += tally["tested"] - before
        live()
    # --- 3. refinement (hill-climb one setting at a time around the best few)
    top = sorted(archive.values(), key=lambda x: -x["fit"])[:10]
    for base in top:
        cur = base
        for _ in range(3):
            if stopped() or tally["tested"] - start_tested >= budget:
                break
            best = cur
            for nb in _neighbours({k: v for k, v in cur.items() if k in ("family", "params", "flt", "stop_atr", "target_r", "hold")}, ctx):
                got = evaluate(nb)
                if got is not None and got["fit"] > best["fit"]:
                    best = got
            if best is cur:
                break
            cur = best
    live()
    return list(archive.values())


def _robust(ctx: _Ctx, c: dict, cfg: LabConfig) -> tuple:
    """Share of one-step neighbours that are profitable on the DEV span (train+validate). Returns (share, how_many_judged, evaluated)."""
    good = tot = 0
    base = {k: v for k, v in c.items() if k in ("family", "params", "flt", "stop_atr", "target_r", "hold")}
    nbrs = _neighbours(base, ctx)
    for nb in nbrs[:16]:
        res = ctx.run(nb)
        if res is None:
            continue
        e2, _x2, r2 = res
        rs = r2[(e2 >= 0) & (e2 < ctx.a2)]
        if len(rs) >= max(15, cfg.min_trades // 2):
            tot += 1
            good += int(rs.mean() > 0)
    return (round(good / tot, 2) if tot else 0.0), tot, len(nbrs[:16])


def run_lab(df: pd.DataFrame, cost_price: float, cfg: Optional[LabConfig] = None, progress: Optional[Callable] = None,
            cancel=None, frames: Optional[list] = None, verify: Optional[Callable] = None) -> LabResult:
    """Run the Lab. With ``cfg.ledger`` set, every variant tested is remembered (and known ones skipped) -- see ``_run_lab_impl``."""
    cfg = cfg or LabConfig()
    led = cfg.ledger if (cfg.ledger is not None and cfg.ledger_scope) else None
    try:
        res = _run_lab_impl(df, cost_price, cfg, progress, cancel, frames, verify)
    except BaseException:
        if led is not None:
            try:
                led.finish_run(tested=0)
            except Exception:  # noqa: BLE001
                pass
        raise
    if led is not None:
        try:
            skipped, warm, new = led.skipped_known, led.warm_started, led.new_variants
            led.finish_run(tested=res.tested, winners=len(res.winners), near_misses=len(res.near_misses))
            new = led.new_variants
            s = led.summary(cfg.ledger_scope)
            res.ledger_info = {"scope": cfg.ledger_scope, "run_number": led.run_id, "new_variants": new, "skipped_known": skipped,
                               "warm_started": warm, "total_variants_on_market": s["variants"], "runs_on_market": s["runs"]}
            if skipped or warm:
                res.notes.insert(0, f"Search memory: skipped {skipped:,} variants already tested in earlier runs on {cfg.ledger_scope}"
                                    + (f" and started evolution from {warm} of the best past variants" if warm else "")
                                    + f". {s['variants']:,} variants are now on record for this market.")
            elif s["variants"]:
                res.notes.insert(0, f"Search memory: {new:,} new variants recorded ({s['variants']:,} on record for {cfg.ledger_scope}). "
                                    "Next run skips these and explores new ground.")
        except Exception:  # noqa: BLE001 -- bookkeeping never fails a search
            pass
    return res


def _run_lab_impl(df: pd.DataFrame, cost_price: float, cfg: LabConfig, progress: Optional[Callable] = None,
                  cancel=None, frames: Optional[list] = None, verify: Optional[Callable] = None) -> LabResult:
    """``df`` is the primary frame (one bar size). Pass ``frames=[(minutes, df, cost_price), ...]`` to search several bar sizes in one run;
    ``verify(candidate, minutes) -> dict`` is the real-engine proof (see app.discovery_lab.verify) applied to each finalist."""
    emit = progress or (lambda ev: None)
    t0 = time.time()
    led = cfg.ledger if (cfg.ledger is not None and cfg.ledger_scope) else None
    run_no = 0
    if led is not None:
        run_no = led.begin_run(source="lab", scope=cfg.ledger_scope, effort=cfg.effort, seed=cfg.seed)
    # Same seed + same data would walk the same path every time (and land on the same strategy). With a memory the seed also depends on
    # how many runs this market has had, so each run explores a different region; without one it is exactly the old behaviour.
    rng = random.Random(f"{cfg.seed}:{led.run_count(cfg.ledger_scope)}" if (led is not None and cfg.vary_seed) else cfg.seed)
    if not frames:
        frames = [(int(round(native_minutes(df))), df, cost_price)]
    notes: list = []
    ctxs = []
    for minutes, d, cp in frames:
        if len(d) < 1500:
            notes.append(f"Skipped {minutes}-minute bars: only {len(d)} bars (needs 1,500 to train, validate and keep a sealed slice).")
            continue
        ctxs.append(_Ctx(d, cp, cfg, minutes))
    if not ctxs:
        raise ValueError("Not enough bars -- the Lab needs at least 1,500 to train, validate and keep a sealed test slice.")
    primary = ctxs[len(ctxs) // 2]
    for cx in ctxs:
        if cx.min_stop > 8.0:
            notes.append(f"Costs are heavy on {cx.minutes}-minute bars ({cx.cost / cx.atr_med:.2f} ATR round-trip); stops were widened to keep cost per trade near "
                         f"{cfg.target_cost_r * 100:.0f}% of R. A larger bar size would work better.")
    budget = cfg.budget or EFFORT_BUDGET.get(cfg.effort, EFFORT_BUDGET["standard"])
    tally = {"tested": 0, "train_pass": 0, "total": budget, "distinct": set(), "skipped_known": 0}

    # ---- search every bar size (the budget is split by how many bars each one has, with a floor so none is starved)
    weights = [max(0.6, math.log10(max(cx.n, 10))) for cx in ctxs]
    shares = [w / sum(weights) for w in weights]
    allcands: list = []
    for cx, sh_ in zip(ctxs, shares):
        fams = [f for f in (cfg.families or list(F.FAMILIES)) if f in F.FAMILIES]
        if not cx.intraday:
            fams = [f for f in fams if F.FAMILIES[f].group != "session"]
        if not fams:
            continue
        got = _search_one_size(cx, fams, cfg, max(300, int(budget * sh_)), rng, emit, cancel, tally, notes)
        for c in got:
            c["_ctx"] = cx
        allcands += got
        if cancel is not None and cancel.is_set():
            notes.append("Stopped early at your request -- results below are from the part of the search that finished.")
            break
    tally["total"] = tally["tested"]

    # ---- validate: the best train candidates, one per distinct signal set + exit, must hold up on the next slice of time
    allcands.sort(key=lambda c: -c["fit"])
    cand, seen_sig = [], {}
    for c in allcands:
        sk = (c["bar_minutes"], _fkey(c))
        if seen_sig.get(sk, 0) >= 2:                   # at most two exit variants per distinct idea
            continue
        seen_sig[sk] = seen_sig.get(sk, 0) + 1
        cand.append(c)
        if len(cand) >= cfg.top_validate:
            break
    emit({"phase": "validate", "tested": tally["tested"], "total": tally["tested"], "train_pass": tally["train_pass"], "candidates": len(cand),
          "bounds": [primary.a1, primary.a2, primary.n], "best": []})
    survivors = []
    for c in cand:
        cx = c["_ctx"]
        ent, ex, r = c["_trades"]
        sv = stats(_slice(ent, ex, r, cx.a1, cx.a2))
        c["val"] = sv
        if sv["n"] >= max(10, cfg.min_trades // 4) and sv["mean"] > 0.02 and sv["t"] >= 1.0 and sv["pf"] >= 1.05:
            survivors.append(c)
    survivors.sort(key=lambda c: -(c["train"]["t"] + 1.4 * c["val"]["t"]))
    survivors = survivors[: cfg.max_finalists]
    if led is not None:
        for c in survivors:
            led.mark(cfg.ledger_scope, _lkey(c, c["bar_minutes"]), "validated")

    # ---- robustness + sealed test + cost stress
    winners, near = [], []
    emit({"phase": "verify", "tested": tally["tested"], "total": tally["tested"], "train_pass": tally["train_pass"], "val_pass": len(survivors),
          "bounds": [primary.a1, primary.a2, primary.n], "best": []})
    robust_pass = 0
    # The best of N unrelated no-edge ideas is expected to reach t ~ sqrt(2 ln N); a winner has to clear a bar that high before it is
    # called "strong". N counts every distinct signal set tried (not just the ones that scored well).
    distinct = len(tally["distinct"]) or 1
    if led is not None:
        try:                       # every signal set EVER tried on this market counts toward the luck bar, not just this run's
            lifetime = led.lifetime_signal_sets(cfg.ledger_scope)
            if lifetime > distinct:
                notes.append(f"The luck bar counts all {lifetime:,} signal sets ever searched on {cfg.ledger_scope}, not just this run's {distinct:,}.")
                distinct = lifetime
        except Exception:  # noqa: BLE001
            pass
    deflated_t = math.sqrt(2.0 * math.log(max(distinct, 3)))
    for c in survivors:
        if cancel is not None and cancel.is_set():
            break
        cx = c["_ctx"]
        share, tot, _ = _robust(cx, c, cfg)
        tally["tested"] += tot
        c["robust"], c["robust_n"] = share, tot
        ent, ex, r = c["_trades"]
        te = stats(_slice(ent, ex, r, cx.a2, cx.n))
        c["test"] = te
        res2 = cx.run({k: v for k, v in c.items() if k in ("family", "params", "flt", "stop_atr", "target_r", "hold")}, cost_mult=2.0)
        tally["tested"] += 1
        r_stress = res2[2] if res2 is not None else np.zeros(0)
        c["stress_mean"] = round(float(r_stress.mean()), 4) if len(r_stress) else 0.0
        fails = []
        if tot < 3 or share < 0.6:
            fails.append(f"fragile: only {int(share * 100)}% of neighbouring settings are profitable")
        else:
            robust_pass += 1
        if te["n"] < 12:
            fails.append(f"too few trades ({te['n']}) in the sealed test slice")
        elif te["mean"] <= 0 or te["pf"] < 1.05:
            fails.append(f"lost money on the sealed test slice (mean {te['mean']:+.2f}R, PF {te['pf']:.2f})")
        if c["stress_mean"] <= 0:
            fails.append("does not survive double trading costs")
        c["fails"] = fails
        c["id"] = f"lab-{abs(hash((c['bar_minutes'],) + _key(c))) % 10**8:08d}"
        c["label"] = label_of(c)
        c["family_source"] = F.FAMILIES[c["family"]].source
        c["cost_r"] = round(cx.cost / (c["stop_atr"] * cx.atr_med), 3)
        dev_end = cx.n
        allr = r
        dd_pct = stats(allr)["dd"] * cfg.risk_pct
        c["dd_pct"] = round(dd_pct, 2)
        c["fits_firm_dd"] = dd_pct <= cfg.max_dd_limit_pct
        c["net_pct"] = round(float(allr.sum()) * cfg.risk_pct, 1)
        c["curve"] = curve(ex, r, dev_end)
        c["curve_pct"] = [round(v * cfg.risk_pct, 2) for v in c["curve"]]
        c["all"] = stats(allr)
        pooled = stats(r)
        c["pooled_t"] = pooled["t"]
        c["deflated_t"] = round(deflated_t, 2)
        c["confidence"] = "strong" if pooled["t"] >= deflated_t else ("plausible" if pooled["t"] >= 0.6 * deflated_t else "weak")
        pp, rp = best_risk(allr, cfg, trades_per_day=len(allr) / max(cx.days, 1.0))
        c["pass_prob"], c["suggested_risk_pct"] = pp, rp
        c["trades_per_week"] = round(len(allr) / max(cx.days, 1.0) * 5, 1)
        (winners if not fails else near).append(c)

    # ---- proof in the real engine (only for would-be winners): written out exactly as saved, run, lookahead-tested
    proven = 0
    if verify is not None and winners:
        emit({"phase": "prove", "tested": tally["tested"], "total": tally["tested"], "train_pass": tally["train_pass"], "val_pass": len(survivors),
              "bounds": [primary.a1, primary.a2, primary.n], "best": []})
        keep = []
        for c in winners:
            if cancel is not None and cancel.is_set():
                keep.append(c)
                continue
            try:
                proof = verify(c, c["bar_minutes"])
            except Exception as exc:  # noqa: BLE001 -- a crash in the proof is a failed proof, never a pass
                proof = {"ok": False, "reason": f"the real-engine check could not run ({type(exc).__name__}: {exc})"}
            c["engine"] = proof
            if proof.get("ok"):
                proven += 1
                keep.append(c)
            else:
                c["fails"] = list(c["fails"]) + [proof.get("reason") or "failed the real-engine check"]
                near.append(c)
        winners = keep

    if led is not None:
        for c in near:
            led.mark(cfg.ledger_scope, _lkey(c, c["bar_minutes"]), "near_miss")
        for c in winners:
            led.mark(cfg.ledger_scope, _lkey(c, c["bar_minutes"]), "winner")
    rank = {"strong": 0, "plausible": 1, "weak": 2}
    winners.sort(key=lambda c: (rank[c["confidence"]], -(c["pass_prob"]), -c["pooled_t"]))
    near.sort(key=lambda c: (len(c["fails"]), -(c["train"]["t"] + c["val"]["t"])))
    winners, near = _diverse(winners, per_family=3), _diverse(near, per_family=2)
    for c in winners + near + allcands:
        c.pop("_trades", None)
        c.pop("_ctx", None)
    if not winners:
        notes.append(f"Nothing cleared every check this time ({tally['tested']:,} ideas tried). The near-misses below show exactly which check each failed; "
                     "try Deep search, another market, or a different bar size.")
    else:
        notes.append(f"{tally['tested']:,} ideas were tried, so some 'winners' can still be luck -- the confidence label on each already accounts for how many "
                     "were tried. Run each through Full Pipeline and Validate before trusting it.")
    nval = len(cand)
    luck = int(round(nval * 0.159 * 0.42 * 0.5))
    emit({"phase": "done", "tested": tally["tested"], "total": tally["tested"], "train_pass": tally["train_pass"], "val_pass": len(survivors),
          "bounds": [primary.a1, primary.a2, primary.n],
          "best": [{"id": c["id"], "label": c["label"], "t": c["train"]["t"], "curve": c["curve"]} for c in (winners + near)[:4]]})
    return LabResult(winners=winners[:12], near_misses=near[:10], tested=tally["tested"], train_pass=tally["train_pass"], val_pass=len(survivors),
                     robust_pass=robust_pass, expected_by_luck=luck, families_tried=len({c["family"] for c in allcands}) or len(F.FAMILIES),
                     cost_r_typical=round(primary.cost / (primary.stops[len(primary.stops) // 2] * primary.atr_med), 3),
                     bars=sum(cx.n for cx in ctxs), notes=notes, elapsed=time.time() - t0, bounds=(primary.a1, primary.a2, primary.n),
                     bar_sizes=[cx.minutes for cx in ctxs], proven=proven)


def _diverse(rows: list, per_family: int = 2) -> list:
    """One exit variant per (bar size, family, params, filters) and at most ``per_family`` per family, so the list shows different IDEAS."""
    seen, count, out = set(), {}, []
    for c in rows:
        k = (c.get("bar_minutes"),) + _fkey(c)
        if k in seen or count.get(c["family"], 0) >= per_family:
            continue
        seen.add(k)
        count[c["family"]] = count.get(c["family"], 0) + 1
        out.append(c)
    return out


# ---------------------------------------------------------------------------------------------------------------- bar size
def native_minutes(df: pd.DataFrame) -> float:
    ts = pd.to_datetime(df["timestamp"])
    d = ts.diff().dropna().dt.total_seconds().to_numpy()
    return float(np.median(d) / 60.0) if len(d) else 1.0


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Same bars the backtester builds for a strategy that declares this TIMEFRAME (so what the Lab tests is what Full Pipeline runs)."""
    from app.data.timeframe_resample import resample_ohlcv
    label = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 240: "4h", 1440: "1d"}.get(int(minutes))
    if label is None:
        s = df.set_index(pd.to_datetime(df["timestamp"]))
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in s.columns:
            agg["volume"] = "sum"
        out = s.resample(f"{int(minutes)}min").agg(agg).dropna(subset=["open"]).reset_index()
        return out.rename(columns={out.columns[0]: "timestamp"})
    return resample_ohlcv(df, label)


def plan_bar_sizes(df: pd.DataFrame, cost_for: Callable, want_cost_r: float = 0.12, min_bars: int = 4000, max_sizes: int = 3, max_stop_atr: float = 3.0):
    """Bar sizes (>= native) where a stop of at most ``max_stop_atr`` ATR keeps round-trip cost under ``want_cost_r`` of R, and enough bars
    remain. Returns ([(minutes, frame, cost_price)], notes). Searches up to ``max_sizes`` of the coarser eligible sizes (cheap trading
    costs, the most history per bar)."""
    base = native_minutes(df)
    out, notes = [], []
    for m in BAR_LADDER:
        if m < base * 0.99:
            continue
        d = df if m <= base * 1.01 else resample(df, m)
        if len(d) < min_bars:
            continue
        a = float(np.nanmedian(F.Indicators(d).atr(14)[WARMUP:])) if len(d) > WARMUP + 20 else 0.0
        cp = float(cost_for(d))
        if a > 0 and cp / (max_stop_atr * a) <= want_cost_r:
            out.append((m, d, cp))
    if not out:                                             # nothing cheap enough: take the coarsest one with enough bars
        for m in reversed(BAR_LADDER):
            if m < base * 0.99:
                continue
            d = df if m <= base * 1.01 else resample(df, m)
            if len(d) >= min_bars:
                out.append((m, d, float(cost_for(d))))
                notes.append(f"Costs stay high even at {m}-minute bars; results may be thin.")
                break
    if not out:
        out.append((int(round(base)), df, float(cost_for(df))))
    return out[-max_sizes:], notes


def pick_bar_size(df: pd.DataFrame, cost_price: float, want_cost_r: float = 0.10, min_bars: int = 4000):
    """Smallest bar size (>= native) at which a 2-ATR stop costs <= want_cost_r, while keeping >= min_bars. Returns (df, minutes, note)."""
    base = native_minutes(df)
    ladder = [m for m in (1, 5, 15, 30, 60, 240) if m >= base * 0.99]
    best = (df, int(round(base)), None)
    for m in ladder:
        d = df if m <= base * 1.01 else resample(df, m)
        if len(d) < min_bars and m > base * 1.01:
            break
        a = float(np.nanmedian(F.Indicators(d).atr(14)[WARMUP:])) if len(d) > WARMUP + 20 else 0.0
        best = (d, m, None)
        if a > 0 and cost_price / (2.0 * a) <= want_cost_r:
            note = None if m <= base * 1.01 else f"Using {m}-minute bars (resampled from {int(round(base))}-minute) so trading costs stay small relative to each trade's risk."
            return d, m, note
    return best[0], best[1], f"Costs stay high even at {best[1]}-minute bars; results may be thin."
