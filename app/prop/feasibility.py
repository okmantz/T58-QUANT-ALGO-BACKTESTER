"""Is a 30-day evaluation pass even mathematically plausible for a strategy with THIS trade frequency and THIS edge per trade?

This is the screening arithmetic behind "prioritise opportunity frequency". It answers, before any backtest of a full strategy,
whether a candidate could possibly reach +6% inside 30 calendar days without breaching the trailing drawdown -- and at what risk
per trade -- under a deliberately SIMPLE model:

    * trades arrive as a Poisson count over the window (mean ``trades_per_30d``);
    * each trade is a win of ``payoff_r`` R (prob ``win_prob``) or a loss of 1 R, minus ``cost_r`` R of friction;
    * R = ``risk_pct`` % of the account;  pass at +target%, bust when equity touches the (EOD-style, locking) trailing floor.

It is an UPPER-BOUND screen, not a verdict: i.i.d. trades hide streak clustering, intraday floating drawdown and daily-loss
locks, all of which only make a real strategy worse. A candidate that fails HERE cannot pass in the real simulator; a candidate that
passes here must still be measured by ``app.prop.thirty_day`` on real, dated trades.

Worked intuition (50k, +6% = $3,000, 4% trailing = $2,000): a strategy that makes 0.57 R per trade but trades five times a month
earns about 2.9 R a month. To bank $3,000 that needs ~$1,050 of risk per R (2.1% of the account) -- but a four-loss streak at that
size is an $4,200 drawdown against a $2,000 floor. Frequency is not a nicety; it is the lever. See ``required_frequency``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.prop.simulator import PropRules


@dataclass
class FeasibilityPoint:
    trades_per_30d: float
    win_prob: float
    payoff_r: float
    cost_r: float
    risk_pct: float
    expectancy_r: float
    p_pass_30: float
    p_bust_30: float
    mean_trades_to_pass: float | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def expectancy_r(win_prob: float, payoff_r: float, cost_r: float = 0.0) -> float:
    return float(win_prob * payoff_r - (1.0 - win_prob) * 1.0 - cost_r)


def pass_probability_iid(*, trades_per_30d: float, win_prob: float, payoff_r: float, risk_pct: float, rules: PropRules,
                         cost_r: float = 0.0, n_sims: int = 4000, seed: int = 1) -> FeasibilityPoint:
    """Monte Carlo P(pass inside the window) for an i.i.d. R-multiple strategy (see module docstring)."""
    rng = np.random.default_rng(seed)
    lam = max(float(trades_per_30d), 0.0)
    nmax = int(lam * 2.5 + 12)
    counts = rng.poisson(lam, size=n_sims)
    wins = rng.random((n_sims, nmax)) < float(win_prob)
    r = np.where(wins, float(payoff_r), -1.0) - float(cost_r)
    step = r * float(risk_pct) / 100.0                                 # fraction of the starting account per trade
    step = np.where(np.arange(nmax)[None, :] < counts[:, None], step, 0.0)
    eq = np.cumsum(step, axis=1)
    target = float(rules.evaluation_profit_target_pct) / 100.0
    dd = float(rules.max_drawdown_pct) / 100.0
    lock = float(getattr(rules, "trailing_lock_offset_pct", 0.0) or 0.0) / 100.0
    peak = np.maximum.accumulate(np.maximum(eq, 0.0), axis=1)
    floor = peak - dd
    if getattr(rules, "trailing_lock", False):
        floor = np.minimum(floor, lock)
    hit = eq >= target
    bust = eq <= floor
    idx = np.arange(nmax)[None, :]
    big = nmax + 1
    first_hit = np.where(hit.any(axis=1), np.argmax(hit, axis=1), big)
    first_bust = np.where(bust.any(axis=1), np.argmax(bust, axis=1), big)
    passed = first_hit < first_bust
    busted = first_bust <= first_hit
    busted &= first_bust < big
    mean_tp = float((first_hit[passed] + 1).mean()) if passed.any() else None
    return FeasibilityPoint(lam, float(win_prob), float(payoff_r), float(cost_r), float(risk_pct), expectancy_r(win_prob, payoff_r, cost_r),
                            float(passed.mean()), float(busted.mean()), mean_tp)


def best_over_risk(*, trades_per_30d: float, win_prob: float, payoff_r: float, rules: PropRules, cost_r: float = 0.0,
                   risk_grid=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0), n_sims: int = 3000, seed: int = 1) -> FeasibilityPoint:
    """The risk-per-trade (as % of account) that maximises P(pass<=30d) for this edge and frequency."""
    best = None
    for rp in risk_grid:
        pt = pass_probability_iid(trades_per_30d=trades_per_30d, win_prob=win_prob, payoff_r=payoff_r, risk_pct=rp, rules=rules,
                                  cost_r=cost_r, n_sims=n_sims, seed=seed)
        if best is None or pt.p_pass_30 > best.p_pass_30:
            best = pt
    return best


def required_frequency(*, win_prob: float, payoff_r: float, rules: PropRules, target_pass: float = 0.65, cost_r: float = 0.0,
                       freq_grid=(2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60), n_sims: int = 3000, seed: int = 1) -> dict:
    """Smallest trades-per-30-days (from ``freq_grid``) at which the best risk size reaches ``target_pass`` for this edge; ``None`` if
    none does. Returns the whole curve too, so the report can show where the knee is."""
    curve = []
    need = None
    for f in freq_grid:
        pt = best_over_risk(trades_per_30d=f, win_prob=win_prob, payoff_r=payoff_r, rules=rules, cost_r=cost_r, n_sims=n_sims, seed=seed)
        curve.append({"trades_per_30d": f, "best_risk_pct": pt.risk_pct, "p_pass_30": pt.p_pass_30, "p_bust_30": pt.p_bust_30})
        if need is None and pt.p_pass_30 >= target_pass:
            need = f
    return {"win_prob": win_prob, "payoff_r": payoff_r, "expectancy_r": expectancy_r(win_prob, payoff_r, cost_r), "required_trades_per_30d": need,
            "target_pass": target_pass, "curve": curve}


def frontier(rules: PropRules, expectancies=(0.1, 0.2, 0.3, 0.5), payoff_r: float = 1.5, freq_grid=(5, 10, 20, 40, 80), cost_r: float = 0.05,
             n_sims: int = 2000, seed: int = 1) -> list:
    """Grid of best-achievable P(pass<=30d) over (edge per trade, trades per 30 days) for a fixed payoff ratio."""
    rows = []
    for e in expectancies:
        # solve win_prob from E = p*b - (1-p) - cost
        p = (e + 1.0 + cost_r) / (payoff_r + 1.0)
        p = min(max(p, 0.01), 0.99)
        for f in freq_grid:
            pt = best_over_risk(trades_per_30d=f, win_prob=p, payoff_r=payoff_r, rules=rules, cost_r=cost_r, n_sims=n_sims, seed=seed)
            rows.append({"expectancy_r": e, "win_prob": round(p, 3), "trades_per_30d": f, "best_risk_pct": pt.risk_pct,
                         "p_pass_30": pt.p_pass_30, "p_bust_30": pt.p_bust_30})
    return rows


def screening_upper_bound(*, trades_per_30d: float, expectancy_r_net: float, win_prob: float, payoff_r: float, rules: PropRules,
                          n_sims: int = 2000, seed: int = 1) -> float:
    """Best P(pass<=30d) any position size could reach for a candidate with this frequency and net edge (i.i.d. model, so an upper
    bound for the real strategy). ``payoff_r`` and ``win_prob`` are the candidate's own; costs are taken as already in the edge."""
    return float(best_over_risk(trades_per_30d=trades_per_30d, win_prob=win_prob, payoff_r=payoff_r, rules=rules, cost_r=0.0,
                                n_sims=n_sims, seed=seed).p_pass_30) if expectancy_r_net > 0 else 0.0
