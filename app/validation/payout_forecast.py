"""Payout forecast: "will this strategy pass the evaluation AND get paid, at which firm, and is it worth the fee?" (2026-11-07).

Why this exists. The pipeline's headline number is the T58 score and a READY / MARGINAL / NOT READY verdict. Both answer "does this
strategy satisfy our statistical gates?" -- not "if I buy this evaluation, what do I expect to take home?". A score of 85 and a score of
65 can have the same (or the reverse) payout economics once a firm's actual rules, its fee, the time the evaluation takes and the
strategy's trade clustering are accounted for. This module answers the economic question directly, firm by firm, from the strategy's
own trades:

  per firm preset
    p      per-attempt pass probability       Monte Carlo (block resampling, the app's engine) AND the real-start-date rolling replay;
                                              the DECISION uses the lower of the two lower confidence bounds, so a strategy that only
                                              passes under one of the two methods is not trusted
    q      P(paid | passed)                   from the same Monte Carlo (first payout reached)
    V      what the trader keeps per payout   profit split x first-payout size (capped at one profit target)
    c      what the attempt costs             fee table, subscription billing per calendar day lived
    EV     p x (q x V - activation) - c       evaluated at the point estimate AND at the lower bound; EV per calendar month as well
  and then the firms are RANKED BY CONSERVATIVE EV PER MONTH, not by any composite score.

Honesty rules that keep it from becoming another self-referential score:
  * trades are rescaled to each firm's account size at a constant risk %, so a $25k and a $200k account are compared like for like;
  * the grade says whether the numbers came from UNSEEN data ("out-of-sample") or from the same data the strategy was tuned on
    ("in-sample: optimistic") -- an in-sample forecast can rank firms for you but is never labelled a prediction;
  * a calibration ledger (``record_outcome``) stores forecast-vs-reality for evaluations actually bought / run, and
    ``calibration_summary`` / ``shrink_factor`` turn it into a haircut on future forecasts once there are enough outcomes;
  * ``score_vs_payout`` measures, on your own library, whether the T58 score even rank-correlates with forecast payout EV.
"""
from __future__ import annotations

import dataclasses
import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MIN_TRADES = 30
LEDGER_NAME = "payout_forecast_ledger.jsonl"
_LEDGER_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------------------------------------
# small stats helpers
# --------------------------------------------------------------------------------------------------------------------
def wilson_lcb(p: float, n_eff: float, z: float = 1.645) -> float:
    """One-sided 95% Wilson lower bound for proportion ``p`` observed over ``n_eff`` independent trials (fractions)."""
    n = max(float(n_eff), 1.0)
    p = min(max(float(p), 0.0), 1.0)
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return max(0.0, (c - a) / d)


def rescale_trades(trades: list, factor: float) -> list:
    """Same trades with every dollar field multiplied by ``factor`` (constant risk-% across account sizes)."""
    if abs(factor - 1.0) < 1e-12:
        return list(trades)
    money = ("pnl", "commission", "initial_risk", "intended_risk_dollars", "planned_risk_dollars", "risk_at_stop_dollars", "equity_after")
    out = []
    for t in trades:
        kw = {}
        for f in money:
            v = getattr(t, f, None)
            if v is not None:
                kw[f] = float(v) * factor
        out.append(dataclasses.replace(t, **kw))
    return out


# --------------------------------------------------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class FirmForecast:
    preset_key: str
    firm: str
    label: str
    account_size: float
    grade: str                              # "OUT-OF-SAMPLE" | "IN-SAMPLE (optimistic)"
    verdict: str                            # PAYS | MARGINAL | DOES NOT PAY | NO FEE DATA
    p_pass: float | None = None             # fractions throughout
    p_pass_mc: float | None = None
    p_pass_rolling: float | None = None
    rolling_windows: int = 0
    p_pass_lcb: float | None = None         # the number decisions use
    p_paid: float | None = None             # P(attempt reaches a first payout)
    q_paid_given_pass: float | None = None
    bust_before_pass: float | None = None
    median_days_to_pass: float | None = None
    payout_net: float | None = None         # $ the trader keeps per payout (after split)
    fee: float | None = None
    split_pct: float | None = None
    ev: float | None = None
    ev_lcb: float | None = None
    ev_per_month: float | None = None
    ev_per_month_lcb: float | None = None
    break_even_p: float | None = None
    methods_agree: bool | None = None
    rules_verified: bool = False
    algo_allowed: bool = False
    written_confirmation: bool = False
    calibration_factor: float = 1.0
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _fee_for(key: str):
    try:
        from app.prop.presets import get_fee_info
        return get_fee_info(key)
    except Exception:  # noqa: BLE001
        return None


def forecast_firm(trades: list, preset, *, source_account_size: float | None = None, sims: int = 1500, seed: int = 7,
                  out_of_sample: bool = False, rolling_stride: int | None = None, calibrate: bool = True) -> FirmForecast:
    """Forecast one firm preset. ``trades`` are the strategy's closed trades from ONE backtest (see rescale note in the module doc)."""
    from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
    from app.orchestration.ev_verdict import billing_profile, compute_ev, compute_ev_attempts, break_even_pass_rate, \
        break_even_pass_rate_attempts, billing_periods
    from app.prop.presets import is_ready_eligible
    from app.prop.rolling_evaluation import run_rolling_evaluation

    rules = preset.to_prop_rules()
    grade = "OUT-OF-SAMPLE" if out_of_sample else "IN-SAMPLE (optimistic)"
    fc = FirmForecast(preset_key=preset.key, firm=preset.firm, label=preset.label, account_size=float(preset.account_size), grade=grade,
                      verdict="NO FEE DATA", rules_verified=bool(is_ready_eligible(preset)), algo_allowed=bool(preset.algo_trading_allowed))
    if len(trades) < MIN_TRADES:
        fc.reasons.append(f"Only {len(trades)} trades; at least {MIN_TRADES} are needed for a forecast to mean anything.")
        fc.verdict = "MARGINAL"
        return fc
    k = (float(preset.account_size) / float(source_account_size)) if source_account_size else 1.0
    tr = rescale_trades(trades, k)
    mc = run_monte_carlo(tr, rules, MonteCarloConfig(n_simulations=int(sims), random_seed=int(seed)))
    p_mc = float(mc.per_attempt_pass_probability) / 100.0
    n_eff = float(getattr(mc, "effective_independent_attempts", 0.0) or 0.0) or float(min(len(tr), 200))
    # the real tape: every actual start date, no resampling
    p_roll, n_win, lcb_roll = None, 0, None
    try:
        win_days = int(max(10, round((rules.max_eval_calendar_days or 90) * 5 / 7)))
        rr = run_rolling_evaluation(tr, rules, window_trading_days=win_days, stride=rolling_stride or 1, max_windows=600)
        if rr.n_windows >= 8:
            p_roll, n_win = float(rr.pass_rate_pct) / 100.0, int(rr.n_windows)
            span_windows = max(1.0, len({t.entry_time.normalize() for t in tr}) / max(win_days, 1))   # overlapping starts are not independent
            lcb_roll = wilson_lcb(p_roll, max(2.0, span_windows))
    except Exception as exc:  # noqa: BLE001
        fc.reasons.append(f"Rolling real-start replay unavailable: {exc}")
    lcb_mc = wilson_lcb(p_mc, n_eff)
    p_lcb = min(lcb_mc, lcb_roll) if lcb_roll is not None else lcb_mc
    p_point = p_mc if p_roll is None else 0.5 * (p_mc + p_roll)
    fc.p_pass_mc, fc.p_pass_rolling, fc.rolling_windows = p_mc, p_roll, n_win
    if p_roll is not None:
        fc.methods_agree = abs(p_mc - p_roll) <= 0.15
        if not fc.methods_agree:
            fc.reasons.append(f"Resampled Monte Carlo says {p_mc * 100:.0f}% but starting on real historical dates passes {p_roll * 100:.0f}%: "
                              "the result depends on WHEN you start (regime dependence). Decisions use the lower bound.")
    # calibration haircut learned from outcomes actually bought / run
    cal = shrink_factor() if calibrate else 1.0
    p_point, p_lcb = p_point * cal, p_lcb * cal
    if cal < 0.999:
        fc.reasons.append(f"Pass odds scaled by {cal:.2f}: past forecasts of this kind over-predicted real outcomes (see calibration ledger).")
    fc.calibration_factor = cal
    fc.p_pass, fc.p_pass_lcb = p_point, p_lcb
    fc.p_paid = float(mc.per_attempt_paid_probability) / 100.0 * cal
    fc.q_paid_given_pass = min(fc.p_paid / p_point, 1.0) if p_point > 0 else 0.0
    fc.bust_before_pass = float(mc.bust_before_pass_probability) / 100.0
    fc.median_days_to_pass = mc.median_calendar_days_to_pass

    fee = _fee_for(preset.key)
    if fee is None or fee.challenge_price is None:
        fc.reasons.append("No fee row for this firm preset: probabilities are shown but no dollar verdict is possible.")
        return fc
    fc.fee, fc.split_pct = float(fee.challenge_price), float(fee.profit_split_pct)
    target_usd = float(preset.account_size) * float(preset.evaluation_profit_target_pct) / 100.0
    fp = float(mc.first_payout_probability) / 100.0
    gross = (float(mc.expected_payout) / fp) if fp > 0 else None
    gross = min(gross, target_usd) if gross else target_usd
    fc.payout_net = gross * float(fee.profit_split_pct) / 100.0
    a = float(getattr(fee, "activation_fee", 0.0) or 0.0)
    q = fc.q_paid_given_pass or 0.0
    profile = None
    if getattr(fee, "monthly_subscription", False):
        profile = billing_profile(mc.attempt_outcomes, mc.attempt_calendar_days, int(getattr(fee, "billing_period_days", 30)))
    if profile:
        fc.ev = compute_ev_attempts(p_point, q, fc.payout_net, profile, fc.fee, a)
        fc.ev_lcb = compute_ev_attempts(p_lcb, q, fc.payout_net, profile, fc.fee, a)
        fc.break_even_p = break_even_pass_rate_attempts(q, fc.payout_net, profile, fc.fee, a)
        days = profile.get("mean_calendar_days")
        period = float(profile.get("billing_period_days", 30))
    else:
        fc.ev = compute_ev(p_point, q, fc.payout_net, fc.fee, a)
        fc.ev_lcb = compute_ev(p_lcb, q, fc.payout_net, fc.fee, a)
        fc.break_even_p = break_even_pass_rate(fc.fee, q, fc.payout_net, a)
        cal_days = [d for d in (mc.attempt_calendar_days or []) if d and d > 0]
        days = (sum(cal_days) / len(cal_days)) if cal_days else (mc.median_calendar_days_to_pass or None)
        period = 30.0
    if days and days > 0:
        months = max(float(days) / period, 1.0)
        fc.ev_per_month, fc.ev_per_month_lcb = fc.ev / months, fc.ev_lcb / months
    # verdict is about money at the CONSERVATIVE rate, and is only called a prediction when the data was unseen
    if fc.ev_lcb > 0 and p_lcb >= (fc.break_even_p or 1.0) * 1.25:
        fc.verdict = "PAYS"
    elif fc.ev > 0:
        fc.verdict = "MARGINAL"
    else:
        fc.verdict = "DOES NOT PAY"
    if fc.verdict == "PAYS" and not out_of_sample:
        fc.reasons.append("Forecast was computed on the same data the strategy was built on, so treat it as an upper bound until it is confirmed on unseen data.")
    if not fc.rules_verified:
        fc.reasons.append("This firm's rule fields are not verified against a firm page: check the firm before paying.")
    if not fc.algo_allowed:
        fc.reasons.append("This firm does not explicitly allow automated trading; do not run this strategy there without written permission.")
    fc.written_confirmation = bool(getattr(fee, "written_confirmation", False))
    return fc


def forecast_across_firms(trades: list, presets: list | None = None, *, source_account_size: float | None = None, sims: int = 1200,
                          seed: int = 7, out_of_sample: bool = False, algo_only: bool = True, max_firms: int | None = 12,
                          progress_cb=None) -> list[FirmForecast]:
    """Forecast every candidate preset and rank by conservative EV per month (best first). Presets without fee data sort last."""
    from app.prop.presets import list_presets
    cands = list(presets) if presets is not None else list_presets()
    if algo_only:
        cands = [p for p in cands if getattr(p, "algo_trading_allowed", False)]
    if max_firms:
        # prefer rule-verified presets when capping, so the cap never drops the firms you can actually trust
        from app.prop.presets import is_ready_eligible
        cands = sorted(cands, key=lambda p: (not is_ready_eligible(p),))[: int(max_firms)]
    out = []
    for i, p in enumerate(cands):
        if progress_cb:
            try:
                progress_cb(f"Forecasting {p.label} ({i + 1}/{len(cands)})")
            except Exception:  # noqa: BLE001
                pass
        try:
            out.append(forecast_firm(trades, p, source_account_size=source_account_size, sims=sims, seed=seed, out_of_sample=out_of_sample))
        except Exception as exc:  # noqa: BLE001 -- one broken preset must not hide the rest
            f = FirmForecast(preset_key=p.key, firm=p.firm, label=p.label, account_size=float(p.account_size),
                             grade="OUT-OF-SAMPLE" if out_of_sample else "IN-SAMPLE (optimistic)", verdict="NO FEE DATA")
            f.reasons.append(f"Forecast failed: {exc}")
            out.append(f)
    out.sort(key=lambda f: (f.ev_per_month_lcb is None, -(f.ev_per_month_lcb if f.ev_per_month_lcb is not None else 0.0)))
    return out


def summarize(forecasts: list[FirmForecast]) -> dict:
    """Headline for a report: the best firm, how many would pay, and a one-line plain-language answer."""
    paying = [f for f in forecasts if f.verdict == "PAYS"]
    best = forecasts[0] if forecasts else None
    if best is None or best.ev_lcb is None:
        line = "No firm could be priced (missing fee data or too few trades)."
    elif paying:
        b = paying[0]
        line = (f"Best fit: {b.label}. Pass odds {b.p_pass * 100:.0f}% (conservative {b.p_pass_lcb * 100:.0f}%), paid {b.p_paid * 100:.0f}% of attempts, "
                f"expected ${b.ev:,.0f} per ${b.fee:,.0f} attempt ({b.grade.lower()}).")
    else:
        line = (f"No firm pays at the conservative rate. Closest: {best.label} (EV ${best.ev:,.0f}, break-even pass rate "
                f"{(best.break_even_p or 0) * 100:.0f}% vs conservative {(best.p_pass_lcb or 0) * 100:.0f}%).")
    return {"headline": line, "n_firms": len(forecasts), "n_paying": len(paying), "best_key": best.preset_key if best else None,
            "grade": forecasts[0].grade if forecasts else None}


# --------------------------------------------------------------------------------------------------------------------
# does the score predict payouts?
# --------------------------------------------------------------------------------------------------------------------
def _ranks(v: list) -> list:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(x: list, y: list) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return None if den == 0 else num / den


def score_vs_payout(rows: list) -> dict:
    """``rows``: [{"name", "score", "ev_per_month_lcb" (or "ev")}] -> rank correlation of the T58 score with forecast payout economics.

    A low or negative number means the score is not telling you which strategy will get paid; use the payout ranking instead."""
    pts = [(float(r["score"]), float(r.get("ev_per_month_lcb", r.get("ev")))) for r in rows
           if r.get("score") is not None and r.get("ev_per_month_lcb", r.get("ev")) is not None]
    rho = spearman([a for a, _ in pts], [b for _, b in pts])
    disagree = []
    if len(pts) >= 4:
        names = [r["name"] for r in rows if r.get("score") is not None and r.get("ev_per_month_lcb", r.get("ev")) is not None]
        sr, er = _ranks([a for a, _ in pts]), _ranks([b for _, b in pts])
        n = len(pts)
        for nm, a, b in zip(names, sr, er):
            if abs(a - b) >= max(2.0, 0.4 * n):
                disagree.append({"name": nm, "score_rank": int(n - a + 1), "payout_rank": int(n - b + 1)})
    if rho is None:
        msg = "Not enough strategies with both a score and a payout forecast to compare."
    elif rho >= 0.6:
        msg = f"The score and payout economics agree (rank correlation {rho:.2f})."
    elif rho >= 0.25:
        msg = f"The score only loosely tracks payout economics (rank correlation {rho:.2f}); rank by payout EV."
    else:
        msg = f"The score does NOT predict payout economics here (rank correlation {rho:.2f}); rank by payout EV, not score."
    return {"spearman": rho, "n": len(pts), "message": msg, "biggest_disagreements": disagree}


# --------------------------------------------------------------------------------------------------------------------
# calibration ledger: forecast vs what really happened
# --------------------------------------------------------------------------------------------------------------------
def _ledger_path() -> Path:
    try:
        from app.data.storage import user_data_dir
        base = Path(user_data_dir())
    except Exception:  # noqa: BLE001
        base = Path.home() / ".t58"
    base.mkdir(parents=True, exist_ok=True)
    return base / LEDGER_NAME


def record_outcome(strategy: str, preset_key: str, forecast_p_pass: float, *, passed: bool | None = None, paid: bool | None = None,
                   forecast_p_paid: float | None = None, note: str = "", path: Path | None = None) -> dict:
    """Append one forecast (and, when known, its real outcome) to the ledger. Call again with the same ids to record the outcome later."""
    import time
    rec = {"ts": time.time(), "strategy": strategy, "preset_key": preset_key, "forecast_p_pass": float(forecast_p_pass),
           "forecast_p_paid": None if forecast_p_paid is None else float(forecast_p_paid), "passed": passed, "paid": paid, "note": note}
    p = Path(path) if path is not None else _ledger_path()
    with _LEDGER_LOCK:
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    return rec


def read_ledger(path: Path | None = None) -> list:
    p = Path(path) if path is not None else _ledger_path()
    if not p.exists():
        return []
    out = []
    with _LEDGER_LOCK:
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    # the latest record per (strategy, preset) wins, so an outcome appended later replaces the open forecast
    latest = {}
    for r in out:
        latest[(r.get("strategy"), r.get("preset_key"))] = r
    return list(latest.values())


def calibration_summary(path: Path | None = None) -> dict:
    """Brier score and reliability of the pass forecasts that have a known outcome."""
    rows = [r for r in read_ledger(path) if r.get("passed") is not None]
    n = len(rows)
    if n == 0:
        return {"n": 0, "message": "No resolved outcomes yet. Record what happens to evaluations you buy or forward-test."}
    pred = sum(r["forecast_p_pass"] for r in rows) / n
    real = sum(1.0 for r in rows if r["passed"]) / n
    brier = sum((r["forecast_p_pass"] - (1.0 if r["passed"] else 0.0)) ** 2 for r in rows) / n
    buckets = []
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        b = [r for r in rows if lo <= r["forecast_p_pass"] < hi]
        if b:
            buckets.append({"range": f"{int(lo * 100)}-{min(int(hi * 100), 100)}%", "n": len(b),
                            "predicted": sum(r["forecast_p_pass"] for r in b) / len(b), "realised": sum(1.0 for r in b if r["passed"]) / len(b)})
    return {"n": n, "mean_predicted": pred, "realised_rate": real, "brier": brier, "buckets": buckets,
            "message": f"{n} resolved evaluations: forecast averaged {pred * 100:.0f}% pass, reality was {real * 100:.0f}%."}


def shrink_factor(path: Path | None = None, min_outcomes: int = 8) -> float:
    """<= 1.0 multiplier for future pass odds: realised / predicted once there are enough resolved outcomes (never raises odds)."""
    try:
        s = calibration_summary(path)
    except Exception:  # noqa: BLE001
        return 1.0
    if s.get("n", 0) < min_outcomes or not s.get("mean_predicted"):
        return 1.0
    return float(min(1.0, max(0.3, s["realised_rate"] / max(s["mean_predicted"], 1e-9))))
