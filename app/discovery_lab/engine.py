"""The Discovery Lab search.

Method (every step exists to stop a lucky backtest being called a discovery):

  1. COST-AWARE EXITS   stops are measured in ATR and the grid starts at the smallest stop whose round-trip cost is <= ~12% of R, so a
                        candidate is never thrown away just because a too-tight stop made costs huge (the old 'cost too high' wall).
  2. TRAIN              first 55% of bars. A candidate needs >= min trades, mean R after costs > 0, t >= 2.0 and PF >= 1.1.
  3. VALIDATE           next 25%. The best train candidates must hold up here (mean R > 0, t >= 0.7, PF >= 1.05).
  4. ROBUSTNESS         the neighbours of each survivor (one parameter or exit step either way) must mostly be profitable too: a real edge
                        is a plateau, not a spike.
  5. SEALED TEST        last 20%, read once per finalist, never used to choose anything. Must be profitable and survive DOUBLE costs.

``winners`` passed every step; ``near_misses`` passed train+validate and show which step they failed. Neither is a guarantee -- the
result carries the number of ideas tried so the false-positive risk is visible.
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

EFFORT_BUDGET = {"quick": 360, "standard": 2400, "deep": 9000}
EXITS_PER_SIGNAL = {"quick": 3, "standard": 8, "deep": 25}
WARMUP = 250
MAX_HOLD = 400
CURVE_POINTS = 90


@dataclass
class LabConfig:
    effort: str = "standard"
    seed: int = 7
    risk_pct: float = 0.5
    balance: float = 100_000.0
    max_dd_limit_pct: float = 10.0
    families: Optional[list] = None
    train_frac: float = 0.55
    val_frac: float = 0.25
    min_trades: int = 40
    top_train: int = 60
    max_finalists: int = 24
    target_cost_r: float = 0.12


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

    def to_dict(self) -> dict:
        return {"winners": self.winners, "near_misses": self.near_misses, "tested": self.tested, "train_pass": self.train_pass,
                "val_pass": self.val_pass, "robust_pass": self.robust_pass, "expected_by_luck": self.expected_by_luck, "families_tried": self.families_tried,
                "cost_r_typical": self.cost_r_typical, "bars": self.bars, "notes": self.notes, "elapsed": round(self.elapsed, 1),
                "bounds": list(self.bounds)}


# ---------------------------------------------------------------------------------------------------------------- simulation
def simulate(lg: np.ndarray, sh: np.ndarray, I: F.Indicators, stop_atr: float, target_r: float, cost_price: float, cost_mult: float = 1.0):
    """Non-overlapping trades. Returns (entry_idx, exit_idx, R_after_costs). Entry = next bar's open; stop checked before target inside a
    bar (pessimistic); unresolved trades close at MAX_HOLD bars."""
    n = I.n
    atr = I.atr(14)
    sig = np.flatnonzero((lg | sh))
    sig = sig[(sig >= WARMUP) & (sig < n - 2)]
    ent, ex, rr = [], [], []
    free = 0
    for i in sig:
        if i < free:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        d = 1 if lg[i] else -1
        if lg[i] and sh[i]:
            continue
        e = I.o[i + 1]
        sd = stop_atr * a
        j1 = min(n, i + 1 + MAX_HOLD)
        hi, lo = I.h[i + 1:j1], I.l[i + 1:j1]
        if d == 1:
            stop, tgt = e - sd, e + target_r * sd
            sh_hit, tg_hit = lo <= stop, hi >= tgt
        else:
            stop, tgt = e + sd, e - target_r * sd
            sh_hit, tg_hit = hi >= stop, lo <= tgt
        s_i = int(np.argmax(sh_hit)) if sh_hit.any() else 10 ** 9
        t_i = int(np.argmax(tg_hit)) if tg_hit.any() else 10 ** 9
        if s_i == 10 ** 9 and t_i == 10 ** 9:
            k = len(hi) - 1
            exit_p = I.c[i + 1 + k]
            r = d * (exit_p - e) / sd
        elif s_i <= t_i:
            k, r = s_i, -1.0
        else:
            k, r = t_i, float(target_r)
        r -= cost_mult * cost_price / sd
        ent.append(i + 1)
        ex.append(i + 1 + k)
        rr.append(r)
        free = i + 1 + k + 1
    return np.asarray(ent, int), np.asarray(ex, int), np.asarray(rr, float)


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


# ---------------------------------------------------------------------------------------------------------------- helpers
def _exit_grid(atr_med: float, cost_price: float, target_cost_r: float):
    min_stop = cost_price / (target_cost_r * atr_med) if atr_med > 0 else 1.0
    stops = [s for s in F.EXIT_GRID["stop_atr"] if s >= min_stop]
    note = None
    if not stops:
        base = math.ceil(min_stop * 2) / 2
        stops = [min(base, 8.0), min(base + 1, 9.0), min(base + 2, 10.0)]
        note = (f"Costs are heavy on this bar size ({cost_price / atr_med:.2f} ATR round-trip), so stops were widened to "
                f"{stops[0]:g}+ ATR to keep cost per trade near {target_cost_r * 100:.0f}% of R. A larger bar size would work better.")
    return stops, F.EXIT_GRID["target_r"], min_stop, note


def label_of(c: dict) -> str:
    fam = F.FAMILIES[c["family"]]
    ptxt = " ".join(f"{k}={v}" for k, v in c["params"].items() if v not in (0, None))
    return f"{fam.label} ({ptxt}) | stop {c['stop_atr']:g} ATR, target {c['target_r']:g}R"


def _key(c): return (c["family"], tuple(sorted(c["params"].items())), c["stop_atr"], c["target_r"])


# ---------------------------------------------------------------------------------------------------------------- the search
def run_lab(df: pd.DataFrame, cost_price: float, cfg: Optional[LabConfig] = None, progress: Optional[Callable] = None,
            cancel=None) -> LabResult:
    cfg = cfg or LabConfig()
    emit = progress or (lambda ev: None)
    t0 = time.time()
    rng = random.Random(cfg.seed)
    I = F.Indicators(df)
    n = I.n
    if n < 1500:
        raise ValueError(f"Only {n} bars -- the Lab needs at least 1,500 to train, validate and keep a sealed test slice.")
    a1, a2 = int(n * cfg.train_frac), int(n * (cfg.train_frac + cfg.val_frac))
    atr_med = float(np.nanmedian(I.atr(14)[WARMUP:]))
    stops, targets, min_stop, cost_note = _exit_grid(atr_med, cost_price, cfg.target_cost_r)
    cost_r_typ = cost_price / (stops[len(stops) // 2] * atr_med)
    notes = [cost_note] if cost_note else []
    fams = [f for f in (cfg.families or list(F.FAMILIES)) if f in F.FAMILIES]
    has_time = "timestamp" in df.columns
    if not has_time or I.ts.dt.hour.nunique() < 3:      # daily bars: the session family has nothing to work on
        fams = [f for f in fams if F.FAMILIES[f].group != "session"]
    budget = EFFORT_BUDGET.get(cfg.effort, EFFORT_BUDGET["standard"])
    per_sig = EXITS_PER_SIGNAL.get(cfg.effort, 8)
    pairs = []
    for fn in fams:
        for p in F.all_param_sets(F.FAMILIES[fn]):
            pairs.append((fn, p))
    rng.shuffle(pairs)
    # equal weight per family so a family with a huge grid doesn't crowd out the rest
    by_fam: dict = {}
    for fn, p in pairs:
        by_fam.setdefault(fn, []).append(p)
    want_pairs = max(len(fams), budget // per_sig)
    chosen: list = []
    i = 0
    while len(chosen) < want_pairs and any(by_fam.values()):
        for fn in fams:
            if by_fam.get(fn):
                chosen.append((fn, by_fam[fn].pop()))
                if len(chosen) >= want_pairs:
                    break
    results: list = []
    tested = 0
    train_pass = 0
    best_live: list = []
    seen = set()
    total = len(chosen) * per_sig
    emit({"phase": "search", "tested": 0, "total": total, "train_pass": 0, "best": [], "bounds": [a1, a2, n]})
    for ci, (fn, p) in enumerate(chosen):
        if cancel is not None and cancel.is_set():
            notes.append("Stopped early at your request -- results below are from the part of the search that finished.")
            break
        try:
            lg, sh, _ = F.signals(fn, df, p, I)
        except Exception:  # noqa: BLE001 -- one broken indicator combination must not stop the search
            continue
        if lg.sum() + sh.sum() < 20:
            tested += per_sig
            continue
        combos = [(sa, tr_) for sa in stops for tr_ in targets]
        rng.shuffle(combos)
        for sa, tr_ in combos[:per_sig]:
            c = {"family": fn, "params": p, "stop_atr": sa, "target_r": tr_}
            if _key(c) in seen:
                continue
            seen.add(_key(c))
            tested += 1
            ent, ex, r = simulate(lg, sh, I, c["stop_atr"], c["target_r"], cost_price)
            tr = _slice(ent, ex, r, 0, a1)
            st = stats(tr)
            if st["n"] >= cfg.min_trades and st["mean"] > 0.03 and st["t"] >= 2.0 and st["pf"] >= 1.1:
                train_pass += 1
                c["train"] = st
                c["_trades"] = (ent, ex, r)
                results.append(c)
                best_live.append((st["t"], c))
                best_live.sort(key=lambda x: -x[0])
                best_live = best_live[:4]
        if ci % 12 == 0 or ci == len(chosen) - 1:
            emit({"phase": "search", "tested": tested, "total": total, "train_pass": train_pass, "bounds": [a1, a2, n],
                  "best": [{"id": i_, "label": label_of(c_), "t": c_["train"]["t"],
                            "curve": curve(c_["_trades"][1][c_["_trades"][0] < a1], c_["_trades"][2][c_["_trades"][0] < a1], n, upto=a1)}
                           for i_, (_, c_) in enumerate(best_live)]})

    # ---- validate
    results.sort(key=lambda c: -c["train"]["t"])
    cand = results[: cfg.top_train]
    survivors = []
    emit({"phase": "validate", "tested": tested, "total": tested, "train_pass": train_pass, "candidates": len(cand), "bounds": [a1, a2, n], "best": []})
    for c in cand:
        ent, ex, r = c["_trades"]
        sv = stats(_slice(ent, ex, r, a1, a2))
        c["val"] = sv
        if sv["n"] >= max(8, cfg.min_trades // 4) and sv["mean"] > 0.02 and sv["t"] >= 0.7 and sv["pf"] >= 1.05:
            survivors.append(c)
    survivors.sort(key=lambda c: -(c["train"]["t"] + c["val"]["t"]))
    survivors = survivors[: cfg.max_finalists]

    # ---- robustness + sealed test + cost stress
    winners, near = [], []
    emit({"phase": "verify", "tested": tested, "total": tested, "train_pass": train_pass, "val_pass": len(survivors), "bounds": [a1, a2, n], "best": []})
    robust_pass = 0
    for c in survivors:
        if cancel is not None and cancel.is_set():
            break
        fam = F.FAMILIES[c["family"]]
        nbrs = [({"family": c["family"], "params": q, "stop_atr": c["stop_atr"], "target_r": c["target_r"]}) for q in F.neighbours(fam, c["params"])]
        for k, vals in (("stop_atr", stops), ("target_r", targets)):
            if c[k] in vals:
                j = vals.index(c[k])
                for jj in (j - 1, j + 1):
                    if 0 <= jj < len(vals):
                        nb = {"family": c["family"], "params": c["params"], "stop_atr": c["stop_atr"], "target_r": c["target_r"]}
                        nb[k] = vals[jj]
                        nbrs.append(nb)
        good = tot = 0
        for nb in nbrs[:14]:
            try:
                lg, sh, _ = F.signals(nb["family"], df, nb["params"], I)
                e2, x2, r2 = simulate(lg, sh, I, nb["stop_atr"], nb["target_r"], cost_price)
            except Exception:  # noqa: BLE001
                continue
            rs = r2[(e2 >= 0) & (e2 < a2)]
            if len(rs) >= max(15, cfg.min_trades // 2):
                tot += 1
                good += int(rs.mean() > 0)
        c["robust"] = round(good / tot, 2) if tot else 0.0
        c["robust_n"] = tot
        ent, ex, r = c["_trades"]
        te = stats(_slice(ent, ex, r, a2, n))
        c["test"] = te
        lg, sh, _ = F.signals(c["family"], df, c["params"], I)
        _, _, r_stress = simulate(lg, sh, I, c["stop_atr"], c["target_r"], cost_price, cost_mult=2.0)
        c["stress_mean"] = round(float(r_stress.mean()), 4) if len(r_stress) else 0.0
        fails = []
        if tot < 3 or c["robust"] < 0.6:
            fails.append(f"fragile: only {int(c['robust'] * 100)}% of neighbouring settings are profitable")
        else:
            robust_pass += 1
        if te["n"] < 12:
            fails.append(f"too few trades ({te['n']}) in the sealed test slice")
        elif te["mean"] <= 0 or te["pf"] < 1.05:
            fails.append(f"lost money on the sealed test slice (mean {te['mean']:+.2f}R, PF {te['pf']:.2f})")
        if c["stress_mean"] <= 0:
            fails.append("does not survive double trading costs")
        c["fails"] = fails
        c["id"] = f"lab-{abs(hash(_key(c))) % 10**8:08d}"
        c["label"] = label_of(c)
        c["cost_r"] = round(cost_price / (c["stop_atr"] * atr_med), 3)
        allr = r
        dd_pct = stats(allr)["dd"] * cfg.risk_pct
        c["dd_pct"] = round(dd_pct, 2)
        c["fits_firm_dd"] = dd_pct <= cfg.max_dd_limit_pct
        c["net_pct"] = round(float(allr.sum()) * cfg.risk_pct, 1)
        c["curve"] = curve(ex, r, n)
        c["curve_pct"] = [round(v * cfg.risk_pct, 2) for v in c["curve"]]
        c["all"] = stats(allr)
        (winners if not fails else near).append(c)
    winners.sort(key=lambda c: -(c["train"]["t"] + c["val"]["t"] + 2 * c["test"]["mean"] * 10))
    near.sort(key=lambda c: (len(c["fails"]), -(c["train"]["t"] + c["val"]["t"])))
    winners, near = _diverse(winners, per_family=2), _diverse(near, per_family=2)
    for c in winners + near:
        c.pop("_trades", None)
    for c in results:
        c.pop("_trades", None)
    if not winners:
        notes.append(f"Nothing cleared every check this time ({tested:,} ideas tried). The near-misses below show exactly which check each failed; "
                     "try a larger bar size, 'Deep' effort, or another market.")
    elif tested > 800:
        notes.append(f"{tested:,} ideas were tried, so a few 'winners' can still be luck -- run each through Validate before trusting it.")
    emit({"phase": "done", "tested": tested, "total": tested, "train_pass": train_pass, "val_pass": len(survivors), "bounds": [a1, a2, n],
          "best": [{"id": c["id"], "label": c["label"], "t": c["train"]["t"], "curve": c["curve"]} for c in (winners + near)[:4]]})
    return LabResult(winners=winners[:12], near_misses=near[:10], tested=tested, train_pass=train_pass, val_pass=len(survivors),
                     robust_pass=robust_pass, expected_by_luck=int(round(tested * 0.023)), families_tried=len({c["family"] for c in results}) or len(fams), cost_r_typical=round(cost_r_typ, 3),
                     bars=n, notes=notes, elapsed=time.time() - t0, bounds=(a1, a2, n))


def _diverse(rows: list, per_family: int = 2) -> list:
    """One exit variant per (family, params) and at most ``per_family`` per family, so the list shows different IDEAS, not one idea
    with five stop/target settings."""
    seen, count, out = set(), {}, []
    for c in rows:
        k = (c["family"], tuple(sorted(c["params"].items())))
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
    s = df.set_index(pd.to_datetime(df["timestamp"]))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in s.columns:
        agg["volume"] = "sum"
    out = s.resample(f"{int(minutes)}min").agg(agg).dropna(subset=["open"]).reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in out.columns:
        out = out.rename(columns={out.columns[0]: "timestamp"})
    return out


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
