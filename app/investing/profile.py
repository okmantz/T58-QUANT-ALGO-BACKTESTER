"""Onboarding questionnaire -> investor profile -> algo choice and parameters (the 'optimised to you' step)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

QUESTIONS = [
    {"id": "risk", "text": "How would you feel if your portfolio dropped 20% in a month?", "options": [
        ("sell", "I'd sell to stop the pain", 1), ("worry", "Uncomfortable, but I'd hold", 2), ("hold", "Fine - it's part of investing", 3), ("buy", "I'd buy more", 4)]},
    {"id": "horizon", "text": "When will you need this money?", "options": [
        ("lt1", "Within a year", 1), ("y1_3", "1-3 years", 2), ("y3_10", "3-10 years", 3), ("gt10", "10+ years", 4)]},
    {"id": "goal", "text": "What matters most?", "options": [
        ("preserve", "Protect what I have", 1), ("income", "Steady growth with low swings", 2), ("growth", "Grow wealth", 3), ("max", "Maximum upside", 4)]},
    {"id": "style", "text": "How hands-on do you want to be?", "options": [
        ("auto", "Fully automated", 1), ("hybrid", "Automated, with my approval", 2), ("manual", "I pick my own investments", 3)]},
    {"id": "universe", "text": "What should the algo be allowed to buy?", "multi": True, "options": [
        ("stocks", "Stocks", 0), ("etf", "Index funds / ETFs", 0), ("crypto", "Crypto", 0)]},
    {"id": "max_loss", "text": "Largest loss you could stomach?", "options": [
        ("5", "5%", 5), ("10", "10%", 10), ("20", "20%", 20), ("35", "35%+", 35)]},
]


@dataclass
class InvestorProfile:
    risk_score: float            # 1 (conservative) .. 4 (aggressive)
    label: str
    horizon: str                 # short | medium | long
    style: str
    universe: list
    max_drawdown_pct: float
    starting_cash: float = 10_000.0

    def to_dict(self):
        return asdict(self)


def build_profile(answers: dict, starting_cash: float = 10_000.0) -> InvestorProfile:
    def pts(qid, default=2):
        q = next(x for x in QUESTIONS if x["id"] == qid)
        for oid, _t, p in q["options"]:
            if oid == answers.get(qid):
                return p
        return default
    risk = (pts("risk") + pts("goal") + pts("horizon") * 0.75 + (pts("max_loss", 10) / 35 * 4) * 0.75) / 3.5
    risk = round(max(1.0, min(4.0, risk)), 2)
    label = "Conservative" if risk < 1.8 else "Moderate" if risk < 2.6 else "Growth" if risk < 3.3 else "Aggressive"
    h = answers.get("horizon", "y1_3")
    horizon = "short" if h in ("lt1", "y1_3") else "medium" if h == "y3_10" else "long"
    uni = answers.get("universe") or ["stocks", "etf"]
    if isinstance(uni, str):
        uni = [uni]
    return InvestorProfile(risk_score=risk, label=label, horizon=horizon, style=answers.get("style", "hybrid"), universe=list(uni),
                           max_drawdown_pct=float(answers.get("max_loss", 10)), starting_cash=float(starting_cash))


def algo_config(p: InvestorProfile) -> dict:
    """Map the profile to concrete algo parameters. Deterministic and explained, so the user can see why."""
    r = p.risk_score
    cfg = {
        "algo": "t58_composite",
        "max_positions": int(round(8 + (r - 1) * 3)),            # 8..17 names
        "max_weight": round(0.12 + 0.03 * (r - 1), 3),           # 12%..21% per name
        "cash_floor": round(max(0.0, 0.30 - 0.09 * (r - 1)), 3), # 30%..3% held in cash
        "value_weight": 0.6 if p.horizon == "long" else 0.35 if p.horizon == "medium" else 0.15,
        "momentum_weight": 0.2 if p.horizon == "long" else 0.35 if p.horizon == "medium" else 0.6,
        "crypto_cap": 0.0 if "crypto" not in p.universe else round(min(0.05 + 0.05 * (r - 1), 0.25), 3),
        "rebalance_days": 63 if p.horizon == "long" else 21 if p.horizon == "medium" else 5,
        "stop_loss_pct": round(min(p.max_drawdown_pct, 35) * 0.8, 1),
    }
    cfg["quality_weight"] = round(1 - cfg["value_weight"] - cfg["momentum_weight"], 3)
    cfg["why"] = [f"{p.label} profile (risk {p.risk_score}/4): up to {cfg['max_positions']} holdings, {cfg['max_weight']*100:.0f}% max each, {cfg['cash_floor']*100:.0f}% cash floor.",
                  f"{p.horizon.title()}-term: leans {'value' if p.horizon=='long' else 'momentum' if p.horizon=='short' else 'balanced'}; rebalances every {cfg['rebalance_days']} trading days.",
                  f"Positions are cut at -{cfg['stop_loss_pct']}% to stay inside your {p.max_drawdown_pct:.0f}% loss limit."]
    return cfg
