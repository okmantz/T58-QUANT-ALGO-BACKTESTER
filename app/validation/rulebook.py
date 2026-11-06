"""T58 Edge Rulebook (2026-11-03) as code: the math, the alpha ledger, the binary screen, the kill-switch and the stage map.

The Rulebook is the standing bar every idea is held to. This module is the single implementation of its numbers so the app, the docs and
the tests quote the same ones:

    E = p(b+1) - 1 - c                    net edge per trade (R) from win rate p, payoff b, round-turn cost c (R)
    p* = (1+c)/(b+1)                      break-even win rate
    Delta* = (E* + c)/(b+1)               probability lift over a no-skill walk (win rate 1/(b+1)) needed for E*
    z_bar(N) = Phi^-1(1 - 0.025/N)        deflation bar after N looks (two-sided 5% family error, Bonferroni)
    E[max N]                              expected best of N pure-noise z-scores (Bailey-Lopez de Prado approximation)
    n_needed = ((E[max N] + 1.645)/SR)^2 + 1   trades needed to confirm per-trade Sharpe SR after N looks
    k_eff = n/(1 + (n-1) rho)             independent bets in n edges of average correlation rho
    LLR_t = sum 0.15/sigma^2 (R_i - 0.075), kill at LLR <= -2.25        live kill line (H1 = healthy +0.15R vs H0 = no edge; the 0.075R in the formula is their midpoint)

Pure functions plus a small JSON-backed alpha ledger. numpy/stdlib only.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from app.validation.edge_stats import _norm_ppf

# ---- standing constants (Rulebook section 1) -------------------------------------------------------------------------
TARGET_EDGE_R = 0.15            # certified net edge per trade, after cost
BREAKEVEN_EV_EDGE_R = 0.035     # first-payout EV per attempt turns positive near this net edge (simulator, $172 fee)
DEFAULT_COST_R = 0.05           # round-turn cost assumed until measured
MAX_REQUIRED_LIFT_PTS = 7.0     # Stage 0: a mechanism needing more lift than this at the chosen payoff is not worth testing
MIN_TRADES_CERT = 600           # Stage 4 sample floor (3 years)
MIN_FORWARD_TRADES = 250        # Stage 6
ALPHA_FAMILY = 0.05
MAX_HYPOTHESES_PER_QUARTER = 10
Z_BAR_AT_BUDGET = 2.81          # z bar at N = 10
KILL_LLR = -2.25
KILL_H1_R = 0.15
KILL_H0_R = 0.0
KILL_SIGMA_AT_2_TO_1 = 1.46
ONE_SIDED_Z95 = 1.6448536269514722
EULER_GAMMA = 0.5772156649015329


def edge_per_trade(p: float, b: float, c: float = DEFAULT_COST_R) -> float:
    """E = p(b+1) - 1 - c."""
    return p * (b + 1.0) - 1.0 - c


def breakeven_win_rate(b: float, c: float = DEFAULT_COST_R) -> float:
    """p* = (1+c)/(b+1): the win rate at which E = 0."""
    return (1.0 + c) / (b + 1.0)


def no_skill_win_rate(b: float) -> float:
    return 1.0 / (b + 1.0)


def required_lift(b: float, e_star: float = TARGET_EDGE_R, c: float = DEFAULT_COST_R) -> float:
    """Delta* = (E* + c)/(b+1), as a probability (0.067 = 6.7 points)."""
    return (e_star + c) / (b + 1.0)


def z_bar(n_looks: int, alpha: float = ALPHA_FAMILY) -> float:
    """z bar = Phi^-1(1 - (alpha/2)/N). N = hypotheses, payoff cells, markets and windows looked at."""
    n = max(int(n_looks), 1)
    return float(_norm_ppf(1.0 - (alpha / 2.0) / n))


def expected_max_z(n_looks: int) -> float:
    """E[max of N pure-noise z-scores] (0 for N = 1), Bailey & Lopez de Prado's approximation."""
    n = int(n_looks)
    if n <= 1:
        return 0.0
    return float((1.0 - EULER_GAMMA) * _norm_ppf(1.0 - 1.0 / n) + EULER_GAMMA * _norm_ppf(1.0 - 1.0 / (n * math.e)))


def trades_needed(per_trade_sharpe: float, n_looks: int = 1) -> int:
    """n_needed = ((E[max N] + 1.645)/SR)^2 + 1, rounded to the nearest trade (the Rulebook table: 271 / 807 / 1,038 / 1,328 / 1,539)."""
    if per_trade_sharpe <= 0:
        raise ValueError("per-trade Sharpe must be positive")
    return int(round(((expected_max_z(n_looks) + ONE_SIDED_Z95) / per_trade_sharpe) ** 2 + 1.0))


def effective_bets(n_edges: int, avg_corr: float) -> float:
    """k_eff = n / (1 + (n-1) rho). Five ideas at rho 0.3 are 2.3 bets; four index futures at 0.8 are 1.2."""
    n = max(int(n_edges), 0)
    return 0.0 if n == 0 else n / (1.0 + (n - 1) * float(avg_corr))


def sharpe_per_trade_from_edge(edge_r: float, sd_r: float) -> float:
    return edge_r / sd_r if sd_r > 0 else 0.0


def lookup_table(per_trade_sharpe: float = 0.10, looks=(1, 5, 10, 25, 50)) -> list[dict]:
    return [{"looks": n, "z_bar": round(z_bar(n), 2), "trades_needed": trades_needed(per_trade_sharpe, n)} for n in looks]


def lift_table(e_star: float = TARGET_EDGE_R, c: float = DEFAULT_COST_R, payoffs=(1, 2, 3, 4)) -> list[dict]:
    return [{"payoff": f"{b}:1", "lift_pts": round(required_lift(b, e_star, c) * 100, 1)} for b in payoffs]


# ---- Stage 0: Frame --------------------------------------------------------------------------------------------------
@dataclass
class Frame:
    mechanism: str = ""
    sign: str = ""            # "+" or "-" (predicted direction), or free text
    instrument: str = ""
    horizon: str = ""
    payoff_b: float = 2.0
    cost_r: float = DEFAULT_COST_R

    def check(self) -> dict:
        lift_pts = required_lift(self.payoff_b, TARGET_EDGE_R, self.cost_r) * 100
        problems = []
        if not self.mechanism.strip():
            problems.append("Write down who pays you and why (the mechanism).")
        if not self.sign.strip():
            problems.append("State the predicted sign (long or short) before testing.")
        if lift_pts > MAX_REQUIRED_LIFT_PTS:
            problems.append(f"At {self.payoff_b:g}:1 the signal must lift the win rate by {lift_pts:.1f} points (limit {MAX_REQUIRED_LIFT_PTS:.0f}); use 2:1 or higher.")
        return {"passed": not problems, "required_lift_pts": round(lift_pts, 2), "problems": problems}


# ---- Stage 2: binary screen ------------------------------------------------------------------------------------------
def wilson_interval(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    ph = wins / n
    den = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(w1: int, n1: int, w0: int, n0: int) -> float:
    if n1 <= 0 or n0 <= 0:
        return 0.0
    p1, p0 = w1 / n1, w0 / n0
    pool = (w1 + w0) / (n1 + n0)
    se = math.sqrt(max(pool * (1 - pool) * (1.0 / n1 + 1.0 / n0), 1e-18))
    return (p1 - p0) / se


def binary_screen(trades, n_looks: int = 1, cost_r: float = DEFAULT_COST_R, payoff_b: float = 2.0) -> dict:
    """Rulebook Stage 2 on a backtest's trade list. Outcome = 1 if the entry-to-exit PRICE moved up; signal = +1 (long) / -1 (short).

    lift = P(up | long) - P(up | short) with a two-proportion z against z_bar(N); mean R per side and per half; gross edge vs cost.
    Passing needs: |z| >= z_bar(N), net Sharpe above E[max N]/sqrt(n-1), each side's mean R >= 0, both halves same sign and gross edge >= 2x cost.
    A strategy that trades one side only cannot show a lift (nothing to compare) -- reported as "not testable", never as a pass.
    """
    rows = []
    for t in trades or []:
        try:
            d = int(getattr(t, "direction"))
            moved_up = float(t.exit_price) > float(t.entry_price)
            # R = pnl / ACTUAL stop risk (stop distance x size taken) -- the same definition the core pipeline uses.
            # (It used to divide by the intended/configured risk, which misstates R whenever sizing floors or caps the position.)
            from app.scoring.edge_bounds import trade_r
            pnl = float(t.pnl)
            r = trade_r(t)
            rows.append((d, moved_up, r, pnl))
        except Exception:  # noqa: BLE001 -- a malformed trade is skipped, never fatal
            continue
    n = len(rows)
    zb = z_bar(n_looks)
    out = {"n": n, "n_looks": int(n_looks), "z_bar": round(zb, 3), "testable": False, "passed": False, "checks": [], "reason": ""}
    if n < 30:
        out["reason"] = f"Only {n} trades -- the binary screen needs at least 30."
        return out
    longs = [r for r in rows if r[0] > 0]
    shorts = [r for r in rows if r[0] < 0]
    rs = [r[2] for r in rows if r[2] is not None]
    mean_r = (sum(rs) / len(rs)) if rs else None
    sd_r = (math.sqrt(sum((x - mean_r) ** 2 for x in rs) / max(len(rs) - 1, 1)) if rs and mean_r is not None else None)
    out["mean_r"] = mean_r
    sharpe = (mean_r / sd_r) if (mean_r is not None and sd_r) else None
    out["net_sharpe_per_trade"] = sharpe
    hit = lambda grp: sum(1 for g in grp if g[3] > 0)  # noqa: E731
    out["hit_rate"] = {"all": hit(rows) / n, "wilson95": wilson_interval(hit(rows), n)}
    checks = []
    if longs and shorts:
        up_l = sum(1 for g in longs if g[1]); up_s = sum(1 for g in shorts if g[1])
        z = two_proportion_z(up_l, len(longs), up_s, len(shorts))
        out["lift"] = up_l / len(longs) - up_s / len(shorts)
        out["lift_z"] = z
        out["testable"] = True
        checks.append(("Lift is real at the honest N", abs(z) >= zb, f"|z| = {abs(z):.2f} vs z bar {zb:.2f} (N = {n_looks})"))
    else:
        out["reason"] = "The strategy trades one side only, so there is no long-vs-short lift to test; judge it by Stage 4 and its behaviour on falling windows."
    if sharpe is not None:
        need = expected_max_z(n_looks) / math.sqrt(max(n - 1, 1))
        checks.append(("Net Sharpe clears the noise ceiling", sharpe > need, f"{sharpe:.3f} per trade vs {need:.3f} (E[max N]/sqrt(n-1))"))
    for name, grp in (("Long side mean R >= 0", longs), ("Short side mean R >= 0", shorts)):
        g = [x[2] for x in grp if x[2] is not None]
        if g:
            checks.append((name, sum(g) / len(g) >= 0, f"mean {sum(g) / len(g):+.3f}R over {len(g)} trades"))
    half = len(rs) // 2
    if half >= 15:
        a, b2 = sum(rs[:half]) / half, sum(rs[half:]) / (len(rs) - half)
        checks.append(("Both halves have the same sign", (a > 0) == (b2 > 0), f"first half {a:+.3f}R, second half {b2:+.3f}R"))
    if mean_r is not None:
        checks.append(("Gross edge at least 2x cost", mean_r + cost_r >= 2 * cost_r, f"gross ~{mean_r + cost_r:+.3f}R vs 2x cost {2 * cost_r:.3f}R"))
    out["checks"] = [{"name": a, "ok": bool(b), "detail": c_} for a, b, c_ in checks]
    out["passed"] = bool(checks) and all(c["ok"] for c in out["checks"]) and out["testable"]
    return out


# ---- Kill switch (Rulebook section 4) --------------------------------------------------------------------------------
@dataclass
class LLRKillSwitch:
    """Running log-likelihood ratio of 'the edge is +0.15R' (H1) vs 'there is no edge' (H0): LLR += 0.15/sigma^2 * (R - 0.075). Cut the strategy at the kill line.

    In the Rulebook's simulation a dead edge (0R) is killed in a median of 309 trades, a half-decayed one (+0.05R) in 566, and a healthy
    +0.15R edge is wrongly killed in 14% of 3,000-trade paths."""
    sigma_r: float = KILL_SIGMA_AT_2_TO_1
    h1_r: float = KILL_H1_R
    h0_r: float = KILL_H0_R
    kill_at: float = KILL_LLR
    llr: float = 0.0
    n: int = 0
    killed_at_trade: int | None = None

    def add(self, r: float) -> dict:
        self.n += 1
        self.llr += (self.h1_r - self.h0_r) / (self.sigma_r ** 2) * (float(r) - (self.h1_r + self.h0_r) / 2.0)
        if self.killed_at_trade is None and self.llr <= self.kill_at:
            self.killed_at_trade = self.n
        return self.status()

    def status(self) -> dict:
        return {"llr": self.llr, "n": self.n, "kill_at": self.kill_at, "killed": self.killed_at_trade is not None,
                "killed_at_trade": self.killed_at_trade,
                "distance_to_kill": self.llr - self.kill_at}


def llr_path(r_values, **kw) -> dict:
    ks = LLRKillSwitch(**kw)
    for x in r_values:
        if x is not None and math.isfinite(float(x)):
            ks.add(float(x))
    return ks.status()


# ---- Alpha ledger (Rulebook sections 3-4) -----------------------------------------------------------------------------
def _ledger_path() -> Path:
    from app.data.storage import get_app_base_dir
    d = Path(get_app_base_dir()) / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d / "alpha_ledger.json"


def quarter_of(d: date) -> str:
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def load_ledger() -> list[dict]:
    try:
        raw = json.loads(_ledger_path().read_text(encoding="utf-8"))
        return [x for x in raw if isinstance(x, dict)]
    except Exception:  # noqa: BLE001
        return []


def _save_ledger(rows: list[dict]) -> None:
    _ledger_path().write_text(json.dumps(rows, indent=2), encoding="utf-8")


def budget_status(rows: list[dict] | None = None, today: date | None = None) -> dict:
    """This quarter's spend: hypotheses logged (failed ones count), the z bar now in force, and whether the budget is exhausted."""
    rows = load_ledger() if rows is None else rows
    q = quarter_of(today or date.today())
    used = [r for r in rows if r.get("quarter") == q]
    n_now = len(used)
    return {"quarter": q, "hypotheses": n_now, "max": MAX_HYPOTHESES_PER_QUARTER, "left": max(MAX_HYPOTHESES_PER_QUARTER - n_now, 0),
            "exhausted": n_now >= MAX_HYPOTHESES_PER_QUARTER, "z_bar_now": round(z_bar(max(n_now, 1)), 2),
            "z_bar_next": round(z_bar(n_now + 1), 2)}


def add_hypothesis(mechanism: str, segment: str, payoff_b: float = 2.0, cost_r: float = DEFAULT_COST_R, *, extra_looks: int = 0,
                   today: date | None = None, ledger_rows: list[dict] | None = None, persist: bool = True) -> dict:
    """Stage 1: add the hypothesis (and its N) to the ledger BEFORE testing. Refuses past the quarterly budget."""
    rows = load_ledger() if ledger_rows is None else ledger_rows
    d = today or date.today()
    st = budget_status(rows, d)
    if st["exhausted"]:
        return {"ok": False, "error": f"The alpha budget for {st['quarter']} is spent ({st['max']} hypotheses). Failed ideas spend it too; it resets when a new block of unseen data arrives.",
                "status": st}
    frame = Frame(mechanism=mechanism, sign="+", payoff_b=payoff_b, cost_r=cost_r).check()
    n_after = st["hypotheses"] + 1 + max(int(extra_looks), 0)
    row = {"id": f"H{len(rows) + 1}", "mechanism": mechanism.strip(), "segment": segment.strip(), "b": payoff_b, "c_r": cost_r,
           "n_after": n_after, "z_bar": round(z_bar(n_after), 2), "date": d.isoformat(), "quarter": quarter_of(d),
           "stage": 1, "verdict": "open", "required_lift_pts": frame["required_lift_pts"]}
    rows.append(row)
    if persist:
        _save_ledger(rows)
    return {"ok": True, "row": row, "status": budget_status(rows, d)}


def update_hypothesis(hid: str, *, stage: int | None = None, verdict: str | None = None) -> bool:
    rows = load_ledger()
    for r in rows:
        if r.get("id") == hid:
            if stage is not None:
                r["stage"] = int(stage)
            if verdict is not None:
                r["verdict"] = str(verdict)[:200]
            _save_ledger(rows)
            return True
    return False


# ---- Stage map: Rulebook stage -> where the app proves it --------------------------------------------------------------
STAGES = (
    {"stage": 0, "name": "Frame", "rule": "A mechanism and a sign exist; required lift at most 7 points (use 2:1 or higher if not).",
     "where": "Rulebook page (frame check). Test page: you fill in the strategy's mechanism and payoff.", "automatic": False},
    {"stage": 1, "name": "Budget", "rule": "On the menu; no more than 10 hypotheses this quarter (z bar 2.81).",
     "where": "Rulebook page alpha ledger; the Discovery Engine's lifetime trial ledger counts every variant.", "automatic": True},
    {"stage": 2, "name": "Binary screen", "rule": "|z of lift| >= z bar(N); net Sharpe above the noise ceiling; each side >= 0; halves agree; gross edge >= 2x cost.",
     "where": "Test report: 'Rulebook screen' block (binary_screen on the trade list).", "automatic": True},
    {"stage": 3, "name": "Payoff grid", "rule": "Fixed 1:1, 2:1, 3:1 with a pre-set ATR stop; a cell needs mean >= +0.15R net.",
     "where": "Optimize: the payoff grid is a search dimension; every cell counts in N (Discovery trial ledger).", "automatic": True},
    {"stage": 4, "name": "Certification", "rule": "Mean >= +0.15R on a sealed holdout read once; day-block 5th percentile > 0; PSR >= 0.95; still > 0 after +0.05R cost; no year (50+ trades) with PF < 0.9.",
     "where": "Test (holdout confirmation, unseen DSR, 2x-cost ladder, edge by year) and Discovery's sealed slice (lifetime-booked looks).", "automatic": True},
    {"stage": 5, "name": "Prop lifecycle", "rule": "Evaluation bust <= 20%; lower bound of pass >= the READY bar (10th percentile of an outer bootstrap over session days -- the same interval the gate, score and report use); EV per attempt > 0; record payout within 63 days.",
     "where": "Test: Monte Carlo + attempt replay + EV verdict (evaluation breach gate 20%, per-attempt pass lower bound 70%).", "automatic": True},
    {"stage": 6, "name": "Forward test", "rule": "At least 250 trades with a positive lower bound on mean R; kill line not hit. Only then pay for an attempt.",
     "where": "Deploy: Forward test + sequential monitor (SPRT) + the LLR kill line.", "automatic": True},
)

GATE_NAME_MAP = {
    "Evaluation breach (ruin)": "Stage 5 -- evaluation bust <= 20%",
    "Per-attempt pass": "Stage 5 -- bootstrap lower bound of pass >= the READY bar",
    "Holdout confirmation": "Stage 4 -- sealed holdout read once",
    "Edge after 2x costs": "Stage 4 -- still positive after further cost",
    "Unseen DSR": "Stage 4 -- PSR >= 0.95 at the honest N",
    "Null test": "Stage 2/4 -- not distinguishable from random entries",
    "Expected value": "Stage 5 -- EV per attempt > 0",
    "Trades": "Stage 4 -- minimum sample",
}


def stage_map() -> list[dict]:
    return [dict(s) for s in STAGES]


def snapshot() -> dict:
    """Everything the Rulebook page shows, in one call."""
    return {"constants": {"target_edge_r": TARGET_EDGE_R, "breakeven_ev_edge_r": BREAKEVEN_EV_EDGE_R, "cost_r": DEFAULT_COST_R,
                          "min_trades_cert": MIN_TRADES_CERT, "min_forward_trades": MIN_FORWARD_TRADES,
                          "max_hypotheses_per_quarter": MAX_HYPOTHESES_PER_QUARTER, "kill_llr": KILL_LLR},
            "lookup": lookup_table(), "lift": lift_table(), "stages": stage_map(), "budget": budget_status(), "ledger": load_ledger()}
