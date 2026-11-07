"""Net Profit Tester -- strategy validation for LIVE accounts (no prop-firm evaluation, no payout gate).

The prop-firm lifecycle test (``app.prop.lifecycle``) asks "does this configuration pass the evaluation and get paid?". That is the wrong
question for a personal or live account, which has no target to hit, no trailing drawdown that closes it, and no payout gate; it simply
keeps (or loses) what the equity curve earns. This module asks the live-account questions instead:

    * Does the strategy make money AFTER costs, and does the profit survive harsher costs (1.5x / 2x)?
    * Is the per-trade edge statistically distinguishable from zero (t-statistic and a day-block bootstrap interval)?
    * Over a realistic horizon, what is the distribution of outcomes: P(profit), the 5th-percentile result, the drawdown a trader must
      be willing to sit through (95th percentile), P(a 20% / 30% drawdown), P(losing half the account)?
    * Is the profit spread across time (positive months, worst month, longest underwater stretch) or concentrated in one lucky run?

It is selectable beside the prop-firm test (objective mode ``net_profit``); prop-firm testing remains the default everywhere.

Inputs are the same ``Trade`` objects every other engine produces (``pnl`` already includes commission, spread and slippage). Where a
cost stress cannot re-run the backtest, it is applied per trade from the trade's recorded commission and its stop-distance risk
(clearly labelled an approximation); callers that CAN re-run pass ``restress`` and get the exact figure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VERDICT_PASS = "PASS"
VERDICT_MARGINAL = "MARGINAL"
VERDICT_FAIL = "FAIL"
VERDICT_INSUFFICIENT = "INSUFFICIENT DATA"


@dataclass
class NetProfitTestConfig:
    starting_equity: float = 10_000.0
    n_simulations: int = 3_000
    horizon_trades: int | None = None            # None -> as many trades as the history
    day_block_days: int = 1                      # whole session days are resampled (keeps losing days together)
    cost_stress_multiples: tuple = (1.0, 1.5, 2.0)
    min_trades: int = 100
    min_t_stat: float = 1.5
    min_profit_factor: float = 1.15
    min_profit_probability_pct: float = 75.0
    max_p95_drawdown_pct: float = 30.0           # a LIVE account has no firm limit; this is the trader's own tolerance
    ruin_equity_fraction: float = 0.5            # "ruin" = equity at or below this fraction of the starting equity
    max_ruin_probability_pct: float = 5.0
    random_seed: int | None = 42


@dataclass
class NetProfitTestResult:
    verdict: str
    reasons: list
    baseline: dict
    edge: dict
    monte_carlo: dict
    cost_stress: dict
    stability: dict
    config: dict
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["headline"] = self.headline()
        return d

    def headline(self) -> str:
        b, m = self.baseline, self.monte_carlo
        return (f"{self.verdict}: net ${b.get('net_profit', 0):,.0f} over {b.get('n_trades', 0)} trades, "
                f"P(profit)={m.get('p_profit_pct', 0):.0f}%, 95th-pct drawdown {m.get('dd_p95_pct', 0):.1f}%")


def _sorted_arrays(trades):
    ts = sorted(trades, key=lambda t: pd.Timestamp(t.entry_time))
    pnl = np.array([float(t.pnl) for t in ts], dtype=float)
    when = pd.DatetimeIndex([pd.Timestamp(t.entry_time) for t in ts])
    return ts, pnl, when


def _max_drawdown(equity: np.ndarray, start: float) -> tuple[float, float]:
    """(max drawdown in $, max drawdown as a fraction of the running peak)."""
    path = np.concatenate([[start], equity])
    peak = np.maximum.accumulate(path)
    dd = peak - path
    frac = np.divide(dd, peak, out=np.zeros_like(dd), where=peak > 0)
    return float(dd.max()), float(frac.max())


def _profit_factor(pnl: np.ndarray) -> float:
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    if losses > 0:
        return float(wins / losses)
    return float("inf") if wins > 0 else 0.0


def _day_groups(when: pd.DatetimeIndex) -> list:
    days = when.normalize()
    order = np.argsort(days.asi8, kind="stable")
    d = days.asi8[order]
    change = np.ones(len(order), dtype=bool)
    change[1:] = d[1:] != d[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(order))
    return [order[a:b] for a, b in zip(starts, ends)]


def _bootstrap_mean_ci(pnl: np.ndarray, groups: list, rng, n_boot: int = 800, conf: float = 0.95) -> tuple[float, float]:
    """Day-block bootstrap interval of the mean pnl per trade (trades on the same day stay together)."""
    if len(groups) < 3 or len(pnl) < 5:
        return float("nan"), float("nan")
    sums = np.array([pnl[g].sum() for g in groups])
    cnts = np.array([len(g) for g in groups], dtype=float)
    k = len(groups)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, size=k)
        means[i] = sums[pick].sum() / max(cnts[pick].sum(), 1.0)
    lo, hi = np.percentile(means, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return float(lo), float(hi)


def _monte_carlo(pnl: np.ndarray, groups: list, cfg: NetProfitTestConfig, rng) -> dict:
    n = len(pnl)
    horizon = int(cfg.horizon_trades or n)
    sims = int(cfg.n_simulations)
    start = float(cfg.starting_equity)
    finals = np.empty(sims)
    dd_fracs = np.empty(sims)
    ruined = 0
    k = len(groups)
    group_pnls = [pnl[g] for g in groups]
    for s in range(sims):
        seq: list = []
        count = 0
        while count < horizon:
            gp = group_pnls[int(rng.integers(0, k))]
            seq.append(gp)
            count += len(gp)
        path = np.concatenate(seq)[:horizon]
        equity = start + np.cumsum(path)
        finals[s] = equity[-1]
        _, frac = _max_drawdown(equity, start)
        dd_fracs[s] = frac
        if equity.min() <= start * cfg.ruin_equity_fraction:
            ruined += 1
    ret = (finals - start) / start * 100.0
    return {
        "n_simulations": sims, "horizon_trades": horizon,
        "p_profit_pct": float((finals > start).mean() * 100.0),
        "return_pct_p05": float(np.percentile(ret, 5)), "return_pct_p50": float(np.percentile(ret, 50)),
        "return_pct_p95": float(np.percentile(ret, 95)),
        "final_equity_p05": float(np.percentile(finals, 5)), "final_equity_p50": float(np.percentile(finals, 50)),
        "dd_p50_pct": float(np.percentile(dd_fracs, 50) * 100.0), "dd_p95_pct": float(np.percentile(dd_fracs, 95) * 100.0),
        "p_dd_over_20_pct": float((dd_fracs > 0.20).mean() * 100.0), "p_dd_over_30_pct": float((dd_fracs > 0.30).mean() * 100.0),
        "p_ruin_pct": float(ruined / sims * 100.0), "ruin_definition": f"equity at or below {cfg.ruin_equity_fraction:.0%} of the starting equity",
    }


def _cost_stress(trades, pnl: np.ndarray, multiples, restress) -> dict:
    out = {}
    for m in multiples:
        m = float(m)
        if m == 1.0:
            out[m] = {"net_profit": float(pnl.sum()), "method": "baseline"}
            continue
        if restress is not None:
            try:
                tr = restress(m)
                out[m] = {"net_profit": float(sum(float(t.pnl) for t in tr)), "method": "re-run backtest at stressed costs", "n_trades": len(tr)}
                continue
            except Exception as exc:  # noqa: BLE001 -- fall back to the approximation, say so
                note = f"re-run failed ({exc}); approximated"
            else:
                note = ""
        else:
            note = ""
        extra = 0.0
        for t in trades:
            comm = abs(float(getattr(t, "commission", 0.0) or 0.0))
            # no recorded spread/slippage: assume the same again as commission (conservative proxy, labelled below)
            extra += (m - 1.0) * comm * 2.0
        out[m] = {"net_profit": float(pnl.sum() - extra),
                  "method": "approximation: extra cost = (multiple - 1) x 2 x each trade's commission" + (f" ({note})" if note else "")}
    return out


def _stability(pnl: np.ndarray, when: pd.DatetimeIndex, start: float) -> dict:
    s = pd.Series(pnl, index=when)
    monthly = s.groupby([when.year, when.month]).sum()
    n_months = int(len(monthly))
    pos_months = int((monthly > 0).sum())
    equity = start + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[start], equity]))[1:]
    under = equity < peak
    longest = cur = 0
    for flag in under:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    top_decile = float(np.sort(pnl)[::-1][: max(1, len(pnl) // 10)].sum())
    total = float(pnl.sum())
    return {"n_months": n_months, "positive_months": pos_months,
            "positive_months_pct": (pos_months / n_months * 100.0) if n_months else 0.0,
            "worst_month": float(monthly.min()) if n_months else 0.0, "best_month": float(monthly.max()) if n_months else 0.0,
            "longest_underwater_trades": int(longest),
            "top_decile_share_of_profit_pct": (top_decile / total * 100.0) if total > 0 else None}


def run_net_profit_test(trades, cfg: NetProfitTestConfig | None = None, restress=None) -> NetProfitTestResult:
    """Run the live-account test on ``trades``. ``restress(multiple) -> trades`` optionally re-runs the backtest at stressed costs."""
    cfg = cfg or NetProfitTestConfig()
    rng = np.random.default_rng(cfg.random_seed)
    cfg_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}
    if not trades:
        return NetProfitTestResult(VERDICT_INSUFFICIENT, ["No trades."], {}, {}, {}, {}, {}, cfg_dict)
    ts, pnl, when = _sorted_arrays(trades)
    n = len(pnl)
    start = float(cfg.starting_equity)
    equity = start + np.cumsum(pnl)
    dd_money, dd_frac = _max_drawdown(equity, start)
    net = float(pnl.sum())
    sd = float(pnl.std(ddof=1)) if n > 1 else 0.0
    t_stat = float(pnl.mean() / sd * math.sqrt(n)) if sd > 0 else (float("inf") if pnl.mean() > 0 else 0.0)
    groups = _day_groups(when)
    lo, hi = _bootstrap_mean_ci(pnl, groups, rng)
    baseline = {"n_trades": n, "net_profit": net, "return_pct": net / start * 100.0, "net_per_trade": net / n,
                "win_rate_pct": float((pnl > 0).mean() * 100.0), "profit_factor": _profit_factor(pnl),
                "max_drawdown": dd_money, "max_drawdown_pct": dd_frac * 100.0, "starting_equity": start,
                "history_days": int((when.max() - when.min()).days) + 1}
    edge = {"t_stat": t_stat, "mean_per_trade_ci95": [lo, hi],
            "ci_excludes_zero": bool(not math.isnan(lo) and lo > 0), "n_trade_days": len(groups)}
    mc = _monte_carlo(pnl, groups, cfg, rng)
    stress = _cost_stress(ts, pnl, cfg.cost_stress_multiples, restress)
    stab = _stability(pnl, when, start)

    reasons: list = []
    hard: list = []
    if n < cfg.min_trades:
        hard.append(f"only {n} trades (need {cfg.min_trades}+): too few to separate an edge from luck")
    if net <= 0:
        hard.append(f"net profit is ${net:,.0f} after costs")
    pf = baseline["profit_factor"]
    if pf < cfg.min_profit_factor:
        reasons.append(f"profit factor {pf:.2f} < {cfg.min_profit_factor:.2f}")
    if t_stat < cfg.min_t_stat:
        reasons.append(f"per-trade edge t-stat {t_stat:.2f} < {cfg.min_t_stat:.1f}")
    if mc["p_profit_pct"] < cfg.min_profit_probability_pct:
        reasons.append(f"only {mc['p_profit_pct']:.0f}% of simulated futures are profitable (need {cfg.min_profit_probability_pct:.0f}%)")
    if mc["dd_p95_pct"] > cfg.max_p95_drawdown_pct:
        reasons.append(f"95th-percentile drawdown {mc['dd_p95_pct']:.0f}% exceeds your {cfg.max_p95_drawdown_pct:.0f}% tolerance")
    if mc["p_ruin_pct"] > cfg.max_ruin_probability_pct:
        reasons.append(f"{mc['p_ruin_pct']:.1f}% chance of losing half the account (cap {cfg.max_ruin_probability_pct:.0f}%)")
    s15 = stress.get(1.5)
    if s15 is not None and s15["net_profit"] <= 0 and net > 0:
        hard.append("profit disappears at 1.5x costs: the edge is smaller than the cost uncertainty")
    s2 = stress.get(2.0)
    if s2 is not None and s2["net_profit"] <= 0 and net > 0 and not hard:
        reasons.append("profit disappears at 2x costs")
    if stab["n_months"] >= 6 and stab["positive_months_pct"] < 50:
        reasons.append(f"only {stab['positive_months_pct']:.0f}% of months are profitable")
    top = stab["top_decile_share_of_profit_pct"]
    if top is not None and top > 100.0:
        reasons.append("all of the profit (and more) comes from the best 10% of trades")

    if hard:
        verdict = VERDICT_INSUFFICIENT if (n < cfg.min_trades and net > 0 and len(hard) == 1) else VERDICT_FAIL
    elif not reasons:
        verdict = VERDICT_PASS
    elif len(reasons) <= 2:
        verdict = VERDICT_MARGINAL
    else:
        verdict = VERDICT_FAIL
    notes = ["Live-account test: no prop-firm rules are applied. Use the prop-firm lifecycle test for evaluations and funded accounts.",
             "Drawdown and ruin figures come from day-block resampling of this strategy's own trades; they are a tolerance check, not a forecast."]
    if any(v.get("method", "").startswith("approximation") for v in stress.values()):
        notes.append("Cost stress was approximated from recorded commissions because the backtest could not be re-run here.")
    return NetProfitTestResult(verdict, hard + reasons, baseline, edge, mc, stress, stab, cfg_dict, notes)
