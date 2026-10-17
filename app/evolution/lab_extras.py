"""
Evolution Lab extras (2026-10-17 audit, item 2).

Pure, dependency-light building blocks the Evolution Lab runner hooks into. Nothing here runs a backtest or touches
the network, so every piece is unit-testable on its own and cheap enough to call from the generation loop without
slowing it:

  1. parity_gate          -- compare our engine's trades with a SECOND engine (an MT5 report, or the real account) and
                             refuse the word "ready" until they agree.
  2. random controls      -- seeded random-entry strategies injected into every pool; ControlTracker answers "does the
                             search beat chance?" with a number instead of a feeling.
  3. tribes               -- a "consistency" tribe and a "growth" tribe evolve apart and merge late.
  4. decay_check          -- the last 15 trades against the rest: an edge that is already fading is not promoted.
  5. incubation ramp      -- how much risk a new strategy may take as it earns live trades.
  6. HourOfWeekSpread     -- a 168-bucket spread model, so a strategy that only works at 03:00 Sunday pays 03:00 Sunday
                             prices.
  7. EquityCurveStore     -- bounded, thread-safe store of one downsampled equity curve per evaluated mutation, for the
                             live gallery. Appending is O(points) and never blocks the generation loop.

Honesty notes live next to each function: where a number is a heuristic (decay thresholds, tribe weights, the spread
proxy) the docstring says so.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# 1. Parity gate against a second engine
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class ParityResult:
    status: str                      # "PASS" | "FAIL" | "NOT_RUN" | "INSUFFICIENT"
    n_engine: int = 0
    n_reference: int = 0
    n_matched: int = 0
    match_rate: float | None = None  # matched / max(n_engine, n_reference)
    pnl_corr: float | None = None
    total_pnl_engine: float | None = None
    total_pnl_reference: float | None = None
    total_pnl_gap_pct: float | None = None   # |engine - reference| / max(|reference|, 1e-9) * 100
    reasons: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict:
        return dict(self.__dict__) | {"passed": self.passed}


def _as_epoch(t) -> float:
    import pandas as pd
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.value / 1e9


def parity_gate(engine_trades: Sequence, reference_trades: Sequence, *, time_tol_s: float = 120.0,
                min_trades: int = 20, min_match_rate: float = 0.90, min_pnl_corr: float = 0.90,
                max_total_pnl_gap_pct: float = 15.0) -> ParityResult:
    """Do two independent engines tell the same story about the same strategy?

    Each trade (ours: Trade objects; theirs: dicts or objects with entry_time / direction / pnl) is matched to at most
    one reference trade of the same direction whose entry is within ``time_tol_s`` seconds. READY needs ALL of: enough
    trades, a high match rate, correlated per-trade P&L, and total P&L within ``max_total_pnl_gap_pct``. The
    thresholds are judgement calls, deliberately strict -- a backtest that cannot be reproduced elsewhere should not
    pass a prop firm's money through.
    """
    def _get(o, k):
        return o.get(k) if isinstance(o, dict) else getattr(o, k, None)

    eng = [(_as_epoch(_get(t, "entry_time")), int(_get(t, "direction")), float(_get(t, "pnl"))) for t in engine_trades]
    ref = [(_as_epoch(_get(t, "entry_time")), int(_get(t, "direction")), float(_get(t, "pnl"))) for t in reference_trades]
    res = ParityResult(status="NOT_RUN", n_engine=len(eng), n_reference=len(ref))
    if not ref:
        res.reasons.append("No second-engine trades were supplied -- parity not checked, so nothing can be called ready.")
        return res
    if min(len(eng), len(ref)) < min_trades:
        res.status = "INSUFFICIENT"
        res.reasons.append(f"Only {min(len(eng), len(ref))} trade(s) on one side (need >= {min_trades}) to compare engines.")
        return res
    ref_sorted = sorted(range(len(ref)), key=lambda j: ref[j][0])
    ref_t = np.array([ref[j][0] for j in ref_sorted])
    used = set()
    pairs = []
    for (t, d, p) in sorted(eng):
        lo = int(np.searchsorted(ref_t, t - time_tol_s, side="left"))
        hi = int(np.searchsorted(ref_t, t + time_tol_s, side="right"))
        best, best_gap = None, None
        for k in range(lo, hi):
            j = ref_sorted[k]
            if j in used or ref[j][1] != d:
                continue
            gap = abs(ref[j][0] - t)
            if best is None or gap < best_gap:
                best, best_gap = j, gap
        if best is not None:
            used.add(best)
            pairs.append((p, ref[best][2]))
    res.n_matched = len(pairs)
    res.match_rate = len(pairs) / max(len(eng), len(ref))
    res.total_pnl_engine = float(sum(p for _, _, p in eng))
    res.total_pnl_reference = float(sum(p for _, _, p in ref))
    res.total_pnl_gap_pct = abs(res.total_pnl_engine - res.total_pnl_reference) / max(abs(res.total_pnl_reference), 1e-9) * 100.0
    if len(pairs) >= 3:
        a = np.array([x for x, _ in pairs]); b = np.array([y for _, y in pairs])
        if a.std() > 0 and b.std() > 0:
            res.pnl_corr = float(np.corrcoef(a, b)[0, 1])
    ok = True
    if res.match_rate < min_match_rate:
        ok = False
        res.reasons.append(f"Only {res.match_rate*100:.0f}% of trades match between engines (need >= {min_match_rate*100:.0f}%).")
    if res.pnl_corr is None or res.pnl_corr < min_pnl_corr:
        ok = False
        res.reasons.append(f"Per-trade P&L correlation is {('n/a' if res.pnl_corr is None else f'{res.pnl_corr:.2f}')} (need >= {min_pnl_corr:.2f}).")
    if res.total_pnl_gap_pct > max_total_pnl_gap_pct:
        ok = False
        res.reasons.append(f"Total P&L differs by {res.total_pnl_gap_pct:.0f}% between engines (limit {max_total_pnl_gap_pct:.0f}%).")
    res.status = "PASS" if ok else "FAIL"
    if ok:
        res.reasons.append("Both engines reproduce the same trades and P&L within tolerance.")
    return res


def readiness_label(parity: ParityResult | dict | None, evidence_ready: bool) -> str:
    """The one place the Lab decides whether a finalist may be called READY."""
    status = parity.get("status") if isinstance(parity, dict) else (parity.status if parity is not None else "NOT_RUN")
    if status != "PASS":
        return "NOT READY -- second-engine parity " + ("failed" if status == "FAIL" else "not confirmed")
    return "READY" if evidence_ready else "NOT READY -- evidence incomplete"


# ---------------------------------------------------------------------------------------------------------------
# 2. Random controls and the "does the search beat chance?" tracker
# ---------------------------------------------------------------------------------------------------------------
CONTROL_FAMILY = "__random_control__"


def control_exit_pips(df, pip_size: float) -> tuple[float, float]:
    """Stop / target in pips scaled to THIS dataset (1.5 and 2.25 x the median 14-bar true range), so a control is
    sized like a plausible real strategy instead of tripping the pip-scale sanity check."""
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = float(np.nanmedian(tr)) * 3.0 if len(tr) else 0.0   # median bar range x3 ~ a 14-bar ATR without a rolling pass
    pip = float(pip_size or 0.0)
    if not (atr > 0 and pip > 0):
        return 20.0, 30.0
    return max(1.0, 1.5 * atr / pip), max(1.5, 2.25 * atr / pip)


def random_control_spec(seed: int, *, hold_bars: int = 12, n_entries: int = 60, stop_pips: float = 20.0,
                        take_pips: float = 30.0, long_fraction: float = 0.5) -> dict:
    """A python-source candidate whose entries are chosen by a seeded RNG. It keeps the same exit shape as a real
    candidate and nothing else, so beating it is evidence the entry LOGIC added something."""
    code = (
        "import numpy as np\nimport pandas as pd\n"
        f"STRATEGY_NAME = 'random_control_{int(seed)}'\n"
        f"STOP_LOSS_PIPS = {float(stop_pips)!r}\nTAKE_PROFIT_PIPS = {float(take_pips)!r}\n"
        "def generate_signals(df):\n"
        f"    n = len(df); hold = {int(hold_bars)}; k = {int(n_entries)}\n"
        f"    rng = np.random.default_rng({int(seed)})\n"
        "    sig = np.zeros(n, dtype=int)\n"
        "    slots = np.arange(0, max(1, n - hold - 1), hold + 1)\n"
        "    if len(slots) == 0:\n        return pd.Series(sig, index=df.index)\n"
        "    pick = rng.choice(slots, size=min(k, len(slots)), replace=False)\n"
        "    for st in pick:\n"
        f"        sig[st:st + hold] = 1 if rng.random() < {float(long_fraction)!r} else -1\n"
        "    return pd.Series(sig, index=df.index)\n"
    )
    return {"source_type": "python", "code_text": code}


def _fisher_one_sided(a_pass: int, a_n: int, b_pass: int, b_n: int) -> float:
    """P(search pass-rate this much above control by chance), one-sided Fisher exact."""
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a_pass, a_n - a_pass], [b_pass, b_n - b_pass]], alternative="greater")
        return float(p)
    except Exception:  # noqa: BLE001 -- normal approximation fallback
        p1, p2 = a_pass / max(a_n, 1), b_pass / max(b_n, 1)
        p = (a_pass + b_pass) / max(a_n + b_n, 1)
        se = math.sqrt(max(p * (1 - p) * (1 / max(a_n, 1) + 1 / max(b_n, 1)), 1e-12))
        z = (p1 - p2) / se
        return float(0.5 * math.erfc(z / math.sqrt(2)))


class ControlTracker:
    """Cumulative scoreboard: evolved candidates versus random-entry controls that went through the SAME filters.

    Two comparisons: (a) stage-1 pass rate (Fisher exact, one-sided), and (b) the best evolved profit factor against
    the distribution of control profit factors (a percentile). "Beats chance" needs BOTH p < alpha and the best
    evolved candidate above the 95th control percentile; with fewer than ``min_controls`` controls it says so instead
    of guessing."""

    def __init__(self, alpha: float = 0.05, min_controls: int = 20):
        self.alpha = alpha
        self.min_controls = min_controls
        self.evolved_n = 0
        self.evolved_pass = 0
        self.control_n = 0
        self.control_pass = 0
        self._evolved_pf: list[float] = []
        self._control_pf: list[float] = []
        self._lock = threading.Lock()

    def record(self, is_control: bool, passed: bool, profit_factor: float | None) -> None:
        with self._lock:
            if is_control:
                self.control_n += 1; self.control_pass += int(bool(passed))
                if profit_factor is not None and math.isfinite(profit_factor):
                    self._control_pf.append(float(profit_factor))
            else:
                self.evolved_n += 1; self.evolved_pass += int(bool(passed))
                if profit_factor is not None and math.isfinite(profit_factor):
                    self._evolved_pf.append(float(profit_factor))

    def report(self) -> dict:
        with self._lock:
            en, ep, cn, cp = self.evolved_n, self.evolved_pass, self.control_n, self.control_pass
            epf, cpf = list(self._evolved_pf), list(self._control_pf)
        out = {"evolved_tested": en, "evolved_passed": ep, "control_tested": cn, "control_passed": cp,
               "evolved_pass_rate": (ep / en) if en else None, "control_pass_rate": (cp / cn) if cn else None,
               "p_value": None, "best_evolved_pf": max(epf) if epf else None,
               "control_pf_p95": float(np.percentile(cpf, 95)) if len(cpf) >= 5 else None,
               "best_evolved_percentile_vs_control": None, "beats_chance": None}
        if cn < self.min_controls or en < self.min_controls:
            out["verdict"] = (f"Not enough data yet: {cn} control(s) and {en} evolved candidate(s) tested "
                              f"(need {self.min_controls} of each) -- cannot say whether the search beats chance.")
            return out
        out["p_value"] = _fisher_one_sided(ep, en, cp, cn)
        if epf and len(cpf) >= 5:
            out["best_evolved_percentile_vs_control"] = float(np.mean(np.array(cpf) < max(epf)) * 100.0)
        above = out["control_pf_p95"] is not None and out["best_evolved_pf"] is not None and out["best_evolved_pf"] > out["control_pf_p95"]
        out["beats_chance"] = bool(out["p_value"] < self.alpha and above)
        if out["beats_chance"]:
            out["verdict"] = (f"The search beats chance: {ep}/{en} evolved candidates passed the stage-1 filters versus {cp}/{cn} "
                              f"random controls (one-sided p = {out['p_value']:.3f}), and the best evolved profit factor sits above the "
                              "95th percentile of the controls.")
        else:
            out["verdict"] = (f"NOT distinguishable from chance yet: {ep}/{en} evolved vs {cp}/{cn} random controls passed stage 1 "
                              f"(p = {out['p_value']:.3f}). Treat every 'winner' as possibly luck.")
        return out


# ---------------------------------------------------------------------------------------------------------------
# 3. Tribes
# ---------------------------------------------------------------------------------------------------------------
TRIBES = ("consistency", "growth")


def tribe_scores(fitness) -> dict:
    """Two views of the same Monte-Carlo/OOS evidence (heuristic weights, documented rather than fitted):

    consistency = pass_probability x oos_consistency x robustness (a steady, repeatable edge),
    growth      = payout_probability x sqrt(pass_probability) (payouts are the point; passing is necessary, not sufficient)

    both on a 0-100 scale and reduced by the same penalties as the main fitness so a tribe cannot win by ignoring them.
    """
    if fitness is None:
        return {"consistency": 0.0, "growth": 0.0}
    pen = float(fitness.total_penalty()) if hasattr(fitness, "total_penalty") else 0.0
    cons = 100.0 * float(fitness.pass_probability) * float(fitness.oos_consistency) * float(fitness.robustness)
    grow = 100.0 * float(fitness.payout_probability) * math.sqrt(max(float(fitness.pass_probability), 0.0))
    return {"consistency": max(cons - pen, 0.0), "growth": max(grow - pen, 0.0)}


def tribes_active(gen: int, merge_generation: int, enabled: bool = True) -> bool:
    return bool(enabled) and int(gen) < int(merge_generation)


def tribe_of_index(i: int) -> str:
    return TRIBES[i % 2]


# ---------------------------------------------------------------------------------------------------------------
# 4. Fifteen-trade decay check
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class DecayResult:
    decayed: bool | None             # None = too few trades to say
    window: int
    last_mean: float | None = None
    prior_mean: float | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def decay_check(pnls: Sequence[float], window: int = 15, min_prior: int = 30, fade_ratio: float = 0.25) -> DecayResult:
    """Compare the mean P&L of the LAST ``window`` trades with the mean of everything before them.

    Decayed when the prior mean is positive and the recent mean is negative, or has fallen below ``fade_ratio`` of the
    prior mean. Needs ``window + min_prior`` trades. A heuristic tripwire, not a significance test -- 15 trades are noisy
    on purpose: it catches the strategy that has ALREADY gone cold, and costs nothing to compute."""
    x = np.asarray([p for p in pnls if p is not None and math.isfinite(p)], dtype=float)
    if len(x) < window + min_prior:
        return DecayResult(None, window, reason=f"only {len(x)} trades (need {window + min_prior})")
    last, prior = float(x[-window:].mean()), float(x[:-window].mean())
    if prior > 0 and last < 0:
        return DecayResult(True, window, last, prior, f"last {window} trades average {last:+.2f} against {prior:+.2f} before them")
    if prior > 0 and last < fade_ratio * prior:
        return DecayResult(True, window, last, prior, f"last {window} trades earn {last / prior * 100:.0f}% of the earlier average")
    return DecayResult(False, window, last, prior, "recent trades are in line with the earlier record")


# ---------------------------------------------------------------------------------------------------------------
# 5. Incubation risk ramp
# ---------------------------------------------------------------------------------------------------------------
DEFAULT_RAMP = ((0, 0.25), (20, 0.50), (50, 0.75), (100, 1.00))


def incubation_multiplier(n_live_trades: int, ramp: Sequence[tuple[int, float]] = DEFAULT_RAMP) -> float:
    """Fraction of full risk a strategy may use after ``n_live_trades`` live trades (step function)."""
    m = ramp[0][1]
    for threshold, mult in ramp:
        if n_live_trades >= threshold:
            m = mult
    return float(m)


def incubation_plan(full_risk_pct: float, ramp: Sequence[tuple[int, float]] = DEFAULT_RAMP) -> list[dict]:
    """The schedule to show next to a finalist. Stepping up is conditional on the 15-trade decay check staying clean
    and drawdown staying inside the backtest's worst case -- the plan says so in plain words."""
    steps = []
    for k, (n, mult) in enumerate(ramp):
        steps.append({"from_live_trade": n, "risk_pct": round(full_risk_pct * mult, 4), "fraction_of_full": mult,
                      "advance_if": "start here" if k == 0 else "the last 15 live trades are not decaying and drawdown is inside the backtest worst case"})
    return steps


# ---------------------------------------------------------------------------------------------------------------
# 6. Hour-of-week spread model
# ---------------------------------------------------------------------------------------------------------------
class HourOfWeekSpread:
    """168 buckets (weekday x hour) of relative spread. Real spreads are not observable from OHLC, so the shape is
    PROXIED from the bar range: thin, jumpy hours have wide ranges relative to the volume that trades through them. The
    proxy is normalised so the volume-weighted mean multiplier is 1.0 and clipped to [0.6, 4.0], then scaled by the
    user's flat spread. It is a stress shape, not a quote feed: replace ``mult`` with measured values when you have them."""

    def __init__(self, mult: np.ndarray | None = None):
        self.mult = np.ones(168) if mult is None else np.clip(np.asarray(mult, float), 0.6, 4.0)

    @classmethod
    def fit(cls, df) -> "HourOfWeekSpread":
        import pandas as pd
        ts = pd.DatetimeIndex(df["timestamp"])
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        how = (ts.dayofweek * 24 + ts.hour).to_numpy()
        rng = (df["high"].to_numpy(float) - df["low"].to_numpy(float))
        vol = df["volume"].to_numpy(float) if "volume" in df.columns else np.ones(len(df))
        vol = np.where(np.isfinite(vol) & (vol > 0), vol, 1.0)
        proxy = rng / np.sqrt(vol)
        mult = np.ones(168)
        overall = float(np.nanmedian(proxy)) or 1.0
        for b in range(168):
            sel = how == b
            if sel.sum() >= 5:
                mult[b] = float(np.nanmedian(proxy[sel])) / overall
        mult = mult / max(float(np.mean(mult)), 1e-9)
        return cls(mult)

    def spread_pips(self, ts, flat_pips: float) -> float:
        import pandas as pd
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return float(flat_pips) * float(self.mult[t.dayofweek * 24 + t.hour])


def recost_trades(trades: Iterable, model: HourOfWeekSpread, flat_spread_pips: float, pip_size: float) -> dict:
    """Re-prices each trade at the spread of the hour-of-week it ENTERED and EXITED in, versus the flat spread the
    backtest assumed. Returns adjusted per-trade P&L and the net change. Approximation: cost = size x pip_size x
    (spread_entry + spread_exit - 2 x flat) / 2 per side, i.e. half the spread each way, consistent with a flat model
    that charges ``spread_pips`` once per side."""
    adj, delta = [], 0.0
    for t in trades:
        s = float(getattr(t, "size", 0.0) or 0.0)
        extra = 0.5 * s * float(pip_size) * ((model.spread_pips(t.entry_time, flat_spread_pips) - flat_spread_pips)
                                              + (model.spread_pips(t.exit_time, flat_spread_pips) - flat_spread_pips))
        adj.append(float(t.pnl) - extra)
        delta -= extra
    return {"adjusted_pnls": adj, "net_change": delta,
            "adjusted_expectancy": float(np.mean(adj)) if adj else 0.0}


# ---------------------------------------------------------------------------------------------------------------
# 7. Equity curve store for the live gallery
# ---------------------------------------------------------------------------------------------------------------
def downsample_curve(values: Sequence[float], max_points: int = 90) -> list[float]:
    """Keeps the first and last point and the extremes' neighbourhood by uniform index sampling -- ``max_points`` floats
    per curve keeps the whole feed small enough to poll every few seconds."""
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if len(v) <= max_points:
        return [round(x, 2) for x in v]
    idx = np.unique(np.linspace(0, len(v) - 1, max_points).round().astype(int))
    return [round(v[i], 2) for i in idx]


def curve_from_pnls(pnls: Sequence[float], initial_balance: float, max_points: int = 90) -> list[float]:
    eq = initial_balance + np.cumsum(np.asarray(pnls, float)) if len(pnls) else np.array([])
    return downsample_curve([initial_balance, *eq.tolist()], max_points)


class EquityCurveStore:
    """Ring buffer of recent evaluated mutations (newest last) with a monotonically increasing ``seq`` so a browser can
    ask for "everything since N". One lock, O(1) append, bounded memory (``capacity`` x ``max_points`` floats)."""

    def __init__(self, capacity: int = 240):
        self._items: deque = deque(maxlen=int(capacity))
        self._seq = 0
        self._lock = threading.Lock()

    def add(self, item: dict) -> int:
        with self._lock:
            self._seq += 1
            row = dict(item); row["seq"] = self._seq
            self._items.append(row)
            return self._seq

    def since(self, seq: int = 0, limit: int = 120) -> dict:
        with self._lock:
            rows = [r for r in self._items if r["seq"] > int(seq)]
            return {"seq": self._seq, "items": rows[-int(limit):]}

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._seq = 0
